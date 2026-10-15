//! The demand-paged detached children the durable tests share (#2438, #2455): parents that spawn
//! paged children and serve their faults, the children, and a freeze → codec → thaw on the
//! tree-walker. Included by `durable_paged_detached.rs` and `park_freeze_matrix.rs`, each beside
//! [`super::detached_thaw`].
//!
//! The parent declares a `{ page: (i64) -> (i64) }` export and spawns the child through a v1 record
//! naming it as the pager (#1862, #1863). It serves the child's faults with `svc.poll` while it polls
//! the child, then joins it and returns `status << 56 | served << 48 | value`. Its pager supplies
//! every page from one scratch page holding the faulting page's number at the faulting byte, and
//! refuses any address below 65536, which is the runtime's: the control words and the shadow arena.
//! A durable window keeps those mapped, so the pager is never asked for them; were it asked, its
//! refusal would end the child with a fault.
//!
//! The child touches page 24 (address 100000), runs a polled loop, then touches page 26 (address
//! 110000) and reads `self.attest`. It returns `iters << 24 | b24 << 16 | b26 << 8 | exposure`:
//! `exposure` is the attestation's window and freeze bits, both set for a durable paged child.
#![allow(dead_code)] // each including binary uses a different subset

use super::detached_thaw::*;
use std::sync::Arc;
use temen_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, read_state, STATE_UNWINDING,
};
use temen_interp::Host;
use temen_ir::SpawnRec;

pub const ITERS: u64 = 200_000;

/// What the parent answers when the child ran to the end with its pager: no trap, two pages served,
/// and the child's value.
pub const EXPECT: i64 = (2 << 48) | ((ITERS as i64) << 24) | (24 << 16) | (26 << 8) | 3;

pub const PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65560
  i32.store vrm v1
  vrb = i64.const 65564
  i32.store vrb v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vz = i64.const 0
  br 1(v0, vc, vz)
}
block 1 (vh: i32, vch: i32, vt: i64) {
  vzp = i32.const 0
  vp = svc.poll vzp
  vt2 = i64.add vt vp
  vst = call.cap 6 9 (i32) -> (i32) vh (vch)
  vzz = i32.const 0
  vrun = i32.eq vst vzz
  br_if vrun 1(vh, vch, vt2) 2(vh, vch, vt2)
}
block 2 (vh2: i32, vch2: i32, vt3: i64) {
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh2 (vch2)
  v56 = i64.const 56
  vsb = i64.shl vj_st v56
  v48 = i64.const 48
  vtm = i64.shl vt3 v48
  vr0 = i64.or vsb vtm
  vr = i64.or vr0 vj
  return vr
  }
}
";

/// The parents' pager (#1862): any address below 65536 is refused, being the runtime's; any other
/// is supplied from one scratch page, holding the faulting page's number at the faulting byte.
pub const PAGER_SRC: &str = "func (i64) -> (i64) {
block 0 (vaddr: i64) {
  vlim = i64.const 65536
  vlow = i64.lt_u vaddr vlim
  br_if vlow 1() 2(vaddr)
}
block 1 () {
  vneg = i64.const -1
  return vneg
}
block 2 (va: i64) {
  vmask = i64.const 4095
  voff = i64.and va vmask
  vdt = i64.const 135168
  vdst = i64.add vdt voff
  v12 = i64.const 12
  vpg = i64.shr_u va v12
  vb = i32.wrap_i64 vpg
  i32.store8 vdst vb
  return vdst
  }
}
";

/// The parent, but it computes between serves: each round runs 100000 iterations before its
/// `svc.poll`, so a fault the child takes waits through most of a round for its page.
pub const SLOW_PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65560
  i32.store vrm v1
  vrb = i64.const 65564
  i32.store vrb v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vz = i64.const 0
  br 1(v0, vc, vz)
}
block 1 (vh: i32, vch: i32, vt: i64) {
  vi0 = i64.const 0
  br 2(vh, vch, vt, vi0)
}
block 2 (vh2: i32, vch2: i32, vt2: i64, vi: i64) {
  vk = i64.const 100000
  vlt = i64.lt_s vi vk
  br_if vlt 3(vh2, vch2, vt2, vi) 4(vh2, vch2, vt2)
}
block 3 (vh3: i32, vch3: i32, vt3: i64, vi3: i64) {
  vone = i64.const 1
  vin = i64.add vi3 vone
  br 2(vh3, vch3, vt3, vin)
}
block 4 (vh4: i32, vch4: i32, vt4: i64) {
  vzp = i32.const 0
  vp = svc.poll vzp
  vt5 = i64.add vt4 vp
  vst = call.cap 6 9 (i32) -> (i32) vh4 (vch4)
  vzz = i32.const 0
  vrun = i32.eq vst vzz
  br_if vrun 1(vh4, vch4, vt5) 5(vh4, vch4, vt5)
}
block 5 (vh5: i32, vch5: i32, vt6: i64) {
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh5 (vch5)
  v56 = i64.const 56
  vsb = i64.shl vj_st v56
  v48 = i64.const 48
  vtm = i64.shl vt6 v48
  vr0 = i64.or vsb vtm
  vr = i64.or vr0 vj
  return vr
  }
}
";

/// Two paged children, A and B, both served from the one pager: wait for A, serving, and join it;
/// then the same for B. Returns `a_status << 56 | b_status << 52 | (a_value + b_value)`.
pub const TWO_PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vam = i64.const 65560
  i32.store vam v1
  vab = i64.const 65564
  i32.store vab v3
  vbm = i64.const 65688
  i32.store vbm v2
  vbb = i64.const 65692
  i32.store vbb v3
  vpa = i64.const 65536
  vca = call.cap 6 17 (i64) -> (i32) v0 (vpa)
  vpb = i64.const 65664
  vcb = call.cap 6 17 (i64) -> (i32) v0 (vpb)
  br 1(v0, vca, vcb)
}
block 1 (vh: i32, vca1: i32, vcb1: i32) {
  vz = i32.const 0
  vp = svc.poll vz
  vsa = call.cap 6 9 (i32) -> (i32) vh (vca1)
  vrun = i32.eq vsa vz
  br_if vrun 1(vh, vca1, vcb1) 2(vh, vca1, vcb1)
}
block 2 (vh2: i32, vca2: i32, vcb2: i32) {
  vja_st, vja = call.cap 6 1 (i32) -> (i64, i64) vh2 (vca2)
  br 3(vh2, vcb2, vja_st, vja)
}
block 3 (vh3: i32, vcb3: i32, vas: i64, vav: i64) {
  vz3 = i32.const 0
  vp3 = svc.poll vz3
  vsb = call.cap 6 9 (i32) -> (i32) vh3 (vcb3)
  vrun3 = i32.eq vsb vz3
  br_if vrun3 3(vh3, vcb3, vas, vav) 4(vh3, vcb3, vas, vav)
}
block 4 (vh4: i32, vcb4: i32, vas4: i64, vav4: i64) {
  vjb_st, vjb = call.cap 6 1 (i32) -> (i64, i64) vh4 (vcb4)
  v56 = i64.const 56
  vsas = i64.shl vas4 v56
  v52 = i64.const 52
  vsbs = i64.shl vjb_st v52
  vst = i64.or vsas vsbs
  vsum = i64.add vav4 vjb
  vr = i64.or vst vsum
  return vr
  }
}
";

/// What [`TWO_PARENT_SRC`] answers when both children ran to the end: no trap, and each child's value.
pub const EXPECT_TWO: i64 = 2 * (EXPECT & ((1 << 48) - 1));

/// Two paged children: serve until a fault has been served, then kill A, serve until B has ended,
/// and join both. Returns `a_status << 56 | b_status << 52 | b_value`.
pub const KILL_PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vam = i64.const 65560
  i32.store vam v1
  vab = i64.const 65564
  i32.store vab v3
  vbm = i64.const 65688
  i32.store vbm v2
  vbb = i64.const 65692
  i32.store vbb v3
  vpa = i64.const 65536
  vca = call.cap 6 17 (i64) -> (i32) v0 (vpa)
  vpb = i64.const 65664
  vcb = call.cap 6 17 (i64) -> (i32) v0 (vpb)
  br 1(v0, vca, vcb)
}
block 1 (vh: i32, vca1: i32, vcb1: i32) {
  vzp = i32.const 0
  vp = svc.poll vzp
  vz = i64.const 0
  vnone = i64.eq vp vz
  br_if vnone 1(vh, vca1, vcb1) 2(vh, vca1, vcb1)
}
block 2 (vh2: i32, vca2: i32, vcb2: i32) {
  vk = call.cap 6 12 (i32) -> (i32) vh2 (vca2)
  br 3(vh2, vca2, vcb2)
}
block 3 (vh3: i32, vca3: i32, vcb3: i32) {
  vzp3 = i32.const 0
  vp3 = svc.poll vzp3
  vsb = call.cap 6 9 (i32) -> (i32) vh3 (vcb3)
  vzz = i32.const 0
  vrun = i32.eq vsb vzz
  br_if vrun 3(vh3, vca3, vcb3) 4(vh3, vca3, vcb3)
}
block 4 (vh4: i32, vca4: i32, vcb4: i32) {
  vja_st, vja = call.cap 6 1 (i32) -> (i64, i64) vh4 (vca4)
  vjb_st, vjb = call.cap 6 1 (i32) -> (i64, i64) vh4 (vcb4)
  v56 = i64.const 56
  vsas = i64.shl vja_st v56
  v52 = i64.const 52
  vsbs = i64.shl vjb_st v52
  vst = i64.or vsas vsbs
  vr = i64.or vst vjb
  return vr
  }
}
";

/// Two paged children: serve until a fault has been served, then return 42, joining neither.
pub const RETURNS_PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vam = i64.const 65560
  i32.store vam v1
  vab = i64.const 65564
  i32.store vab v3
  vbm = i64.const 65688
  i32.store vbm v2
  vbb = i64.const 65692
  i32.store vbb v3
  vpa = i64.const 65536
  vca = call.cap 6 17 (i64) -> (i32) v0 (vpa)
  vpb = i64.const 65664
  vcb = call.cap 6 17 (i64) -> (i32) v0 (vpb)
  br 1(v0)
}
block 1 (vh: i32) {
  vzp = i32.const 0
  vp = svc.poll vzp
  vz = i64.const 0
  vnone = i64.eq vp vz
  br_if vnone 1(vh) 2()
}
block 2 () {
  v42 = i64.const 42
  return v42
  }
}
";

/// One paged child: kill it as soon as it is spawned, then join it. Returns `status << 56 | value`.
pub const KILL_AT_ONCE_PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65560
  i32.store vrm v1
  vrb = i64.const 65564
  i32.store vrb v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vk = call.cap 6 12 (i32) -> (i32) v0 (vc)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  v56 = i64.const 56
  vsb = i64.shl vj_st v56
  vr = i64.or vsb vj
  return vr
  }
}
";

/// The child. `touch_first` puts its first page touch before the leaf call that is its first
/// safepoint, so a child the freeze rang before it ran faults before it can unwind (#2455).
pub fn child_src(touch_first: bool) -> String {
    let touch = "  va = i64.const 100000\n  vb1 = i32.load8_u va\n";
    let leaf = "  vs0 = i32.wrap_i64 v1\n  vz = i64.const 0\n  vu = call.cap 5 1 (i64, i64) -> (i64) vs0 (vz, vz)\n";
    let head = if touch_first {
        format!("{touch}{leaf}")
    } else {
        format!("{leaf}{touch}")
    };
    format!(
        "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
{head}  vb164 = i64.extend_i32_u vb1
  vz0 = i64.const 0
  br 1(vz0, vb164)
}}
block 1 (vi: i64, vb: i64) {{
  vn = i64.const {ITERS}
  vc = i64.lt_s vi vn
  br_if vc 2(vi, vb) 3(vi, vb)
}}
block 2 (vj: i64, vb2: i64) {{
  vo = i64.const 1
  vk = i64.add vj vo
  br 1(vk, vb2)
}}
block 3 (vfin: i64, vb3: i64) {{
  vaddr2 = i64.const 110000
  vc2 = i32.load8_u vaddr2
  vc264 = i64.extend_i32_u vc2
  vat = self.attest
  vat64 = i64.extend_i32_u vat
  v8 = i64.const 8
  vex = i64.shr_u vat64 v8
  v24 = i64.const 24
  vacc = i64.shl vfin v24
  v16 = i64.const 16
  vb1s = i64.shl vb3 v16
  vc2s = i64.shl vc264 v8
  vr0 = i64.or vacc vb1s
  vr1 = i64.or vr0 vc2s
  vr = i64.or vr1 vex
  return vr
  }}
}}
"
    )
}

/// #2455 — a child (1 MiB) that streams through memory its pager supplies: `iters` iterations, each
/// reading the first byte of 8 fresh pages from 64 KiB up. Its only safepoints are its loop's
/// back-edges, so between two of them it faults 8 times. It returns the sum of the bytes it read.
pub fn stream_src(iters: u64) -> String {
    let mut body = String::new();
    for k in 0..8u64 {
        body.push_str(&format!(
            "  vo{k} = i64.const {off}\n  va{k} = i64.add vbase vo{k}\n  vb{k} = i32.load8_u va{k}\n  vx{k} = i64.extend_i32_u vb{k}\n",
            off = k * 4096
        ));
    }
    body.push_str(
        "  vs1 = i64.add vx0 vx1\n  vs2 = i64.add vs1 vx2\n  vs3 = i64.add vs2 vx3\n  vs4 = i64.add vs3 vx4\n  vs5 = i64.add vs4 vx5\n  vs6 = i64.add vs5 vx6\n  vs7 = i64.add vs6 vx7\n",
    );
    format!(
        "memory 20 shadow 16448 65536
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vs0 = i32.wrap_i64 v1
  vz = i64.const 0
  vu = call.cap 5 1 (i64, i64) -> (i64) vs0 (vz, vz)
  br 1(vz, vz)
}}
block 1 (vi: i64, vacc: i64) {{
  vn = i64.const {iters}
  vc = i64.lt_s vi vn
  br_if vc 2(vi, vacc) 3(vacc)
}}
block 2 (vj: i64, vacc2: i64) {{
  vps = i64.const 32768
  vsh = i64.mul vj vps
  vlo = i64.const 65536
  vbase = i64.add vlo vsh
{body}  vacc3 = i64.add vacc2 vs7
  vone = i64.const 1
  vk = i64.add vj vone
  br 1(vk, vacc3)
}}
block 3 (vfin: i64) {{
  return vfin
  }}
}}
"
    )
}

/// The parent over one paged child per entry of `sizes`, each `size_log2`, through v1 records naming
/// the parent's `pager` export: the first at 65536 ([`parent_with`]), the next 128 bytes on.
pub fn parent(src: &str, sizes: &[i64]) -> temen_ir::Module {
    let paged = |size_log2| SpawnRec {
        size_log2,
        pager: 0,
        ..SpawnRec::v1()
    };
    let more: String = (1..sizes.len())
        .map(|i| rec::segment(65536 + 128 * i as u64, &paged(sizes[i])))
        .collect();
    parent_with(&format!("{src}{PAGER_SRC}{more}"), &paged(sizes[0]))
}

/// The parent's powerbox: its `Instantiator`, each of `children` as a durable `Module`, a `Budget` of
/// 4 MiB, and freeze authority; it knows its own program, against which op 17 checks the pager
/// export. The args are `(instantiator, modules.., budget)`.
pub fn host(parent: &temen_ir::Module, children: &[&temen_ir::Module]) -> (Host, Vec<i64>) {
    let mut h = Host::new();
    h.set_durable(true);
    let mut args = vec![h.grant_instantiator() as i64];
    for c in children {
        args.push(h.grant_durable_module(c) as i64);
    }
    args.push(h.grant_budget(-1, 1 << 22, 4) as i64);
    h.grant_freeze_authority();
    h.set_self_module(&Arc::new(parent.clone()));
    (h, args)
}

/// A run of the parent on the tree-walker from a fresh durable window, with `arm` applied to it.
pub fn run_from(
    parent: &temen_ir::Module,
    children: &[&temen_ir::Module],
    arm: impl FnOnce(&mut Vec<u8>),
) -> (Out, Vec<u8>, Host, Vec<i64>) {
    let (mut h, args) = host(parent, children);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    arm(&mut win);
    let (out, snap) =
        run(Engine::Interp, parent, &args, &win, &mut h).expect("the tree-walker runs");
    (out, snap, h, args)
}

/// Thaw `art` on `engine` over a restoring host that grants the children and knows the parent's
/// program. `None` when the engine refuses the thaw.
pub fn thaw_on(
    engine: Engine,
    art: &[u8],
    parent: &temen_ir::Module,
    children: &[&temen_ir::Module],
    args: &[i64],
) -> Option<Out> {
    let (mut th, twin) = restored(art, parent, children);
    run(engine, parent, args, &twin, &mut th).map(|(o, _)| o)
}

/// `art` restored for a thaw: a host that grants the children and knows the parent's program, and
/// the window, set to rewind.
pub fn restored(
    art: &[u8],
    parent: &temen_ir::Module,
    children: &[&temen_ir::Module],
) -> (Host, Vec<u8>) {
    let mut th = Host::new();
    th.set_durable(true);
    for c in children {
        th.grant_durable_module(c);
    }
    th.set_self_module(&Arc::new(parent.clone()));
    let mut twin = temen_snapshot::restore(art, parent, &mut th).expect("restore");
    begin_thaw(&mut twin, ARENA, 0);
    (th, twin)
}

/// Freeze with `arm`, require the root frozen with `captured` children captured live and none
/// unreached, and carry the tree through the codec.
pub fn artifact(
    parent: &temen_ir::Module,
    children: &[&temen_ir::Module],
    captured: usize,
    arm: impl FnOnce(&mut Vec<u8>),
) -> (Vec<u8>, Vec<i64>) {
    let (_, snap, mut fh, args) = run_from(parent, children, arm);
    assert_eq!(
        fh.take_freeze_declined(),
        None,
        "the freeze was not declined"
    );
    assert_eq!(read_state(&snap), STATE_UNWINDING, "the root froze");
    assert_eq!(
        fh.captured_detached().len(),
        captured,
        "children captured live"
    );
    assert!(fh.unreached_detached().is_empty(), "no child unreached");
    let art = temen_snapshot::freeze(parent, &snap, &fh).expect("serialize");
    (art, args)
}

/// #1940, #2455 — the freeze's trigger lands while the child is parked on a fault: the parent
/// computes between serves ([`SLOW_PARENT_SRC`]), so the fault waits through most of a round for its
/// page. The census used to decline the freeze there (`Parked(PageFault)`). Now the parent is held:
/// the fault park takes the freeze's phase and waits, the parent serves it in its next round, the
/// child unwinds at its next safepoint, and the thaw gives the uninterrupted answer.
pub fn fault_parked_at_the_trigger_thaws() {
    let child = child_module(&child_src(true));
    let parent = parent(SLOW_PARENT_SRC, &[17]);
    // Past the parent's first quantum, by when the child has run and faulted, and before the round
    // that serves the fault ends.
    let (art, args) = artifact(&parent, &[&child], 1, |w| {
        arm_freeze_after_backedges(w, 50_000)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &[&child], &args),
        Some(Out::Ret(EXPECT))
    );
}
