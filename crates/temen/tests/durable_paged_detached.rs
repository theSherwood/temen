//! #2438 — a durable run freezes a live **demand-paged** detached child, and the thaw gives it back
//! with its pager.
//!
//! The parent declares a `{ page: (i64) -> (i64) }` export and spawns the child through a record
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
#![cfg(all(unix, target_arch = "x86_64"))]

#[path = "support/detached_thaw.rs"]
mod detached_thaw;

use detached_thaw::*;
use std::sync::Arc;
use temen_durable::{arm_freeze_after_backedges, begin_thaw, init_durable_window, write_state};
use temen_interp::Host;
use temen_ir::SpawnRec;

const ITERS: u64 = 200_000;

/// What the parent answers when the child ran to the end with its pager: no trap, two pages served,
/// and the child's value.
const EXPECT: i64 = (2 << 48) | ((ITERS as i64) << 24) | (24 << 16) | (26 << 8) | 3;

const PARENT_SRC: &str = "memory 18 shadow 16448 65536
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface \"pager\" 1 { page: 1 }
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65544
  i32.store vrm v1
  vrb = i64.const 65548
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
func (i64) -> (i64) {
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

/// The child. `touch_first` puts its first page touch before the leaf call that is its first
/// safepoint, so a child the freeze rang before it ran faults before it can unwind.
fn child_src(touch_first: bool) -> String {
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

fn parent() -> temen_ir::Module {
    let paged = SpawnRec {
        size_log2: 17,
        pager: 0,
        ..SpawnRec::default()
    };
    parent_with(PARENT_SRC, &paged)
}

/// The parent's powerbox ([`powerbox`]), knowing its own program: op 17 checks the pager export
/// against it.
fn host(parent: &temen_ir::Module, child: &temen_ir::Module) -> (Host, Vec<i64>) {
    let (mut h, args) = powerbox(child);
    h.set_self_module(&Arc::new(parent.clone()));
    (h, args)
}

/// A run of the parent on the tree-walker from a fresh durable window, with `arm` applied to it.
fn run_from(
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
    arm: impl FnOnce(&mut Vec<u8>),
) -> (Out, Vec<u8>, Host, Vec<i64>) {
    let (mut h, args) = host(parent, child);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    arm(&mut win);
    let (out, snap) =
        run(Engine::Interp, parent, &args, &win, &mut h).expect("the tree-walker runs");
    (out, snap, h, args)
}

/// Thaw `art` on `engine` over a restoring host that grants the child and knows the parent's
/// program. `None` when the engine refuses the thaw.
fn thaw_on(
    engine: Engine,
    art: &[u8],
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
    args: &[i64],
) -> Option<Out> {
    let mut th = Host::new();
    th.set_durable(true);
    th.grant_durable_module(child);
    th.set_self_module(&Arc::new(parent.clone()));
    let mut twin = temen_snapshot::restore(art, parent, &mut th).expect("restore");
    begin_thaw(&mut twin, ARENA, 0);
    run(engine, parent, args, &twin, &mut th).map(|(o, _)| o)
}

/// Freeze with `arm`, require the child captured live, and carry the tree through the codec.
fn artifact(
    parent: &temen_ir::Module,
    child: &temen_ir::Module,
    arm: impl FnOnce(&mut Vec<u8>),
) -> (Vec<u8>, Vec<i64>) {
    let (_, snap, mut fh, args) = run_from(parent, child, arm);
    assert_eq!(
        fh.take_freeze_declined(),
        None,
        "the freeze was not declined"
    );
    assert_eq!(
        fh.captured_detached().len(),
        1,
        "the paged child was captured live"
    );
    let art = temen_snapshot::freeze(parent, &snap, &fh).expect("serialize");
    (art, args)
}

#[test]
fn a_paged_child_runs_to_its_answer_with_only_its_own_pages_served() {
    let (parent, child) = (parent(), child_module(&child_src(false)));
    let (out, ..) = run_from(&parent, &child, |_| {});
    assert_eq!(
        out,
        Out::Ret(EXPECT),
        "two guest pages served, the runtime's never asked for"
    );
}

/// The headline: cut mid-loop, after its first page was served, the child thaws with that page in
/// its image and its pager serving the next one.
#[test]
fn a_paged_child_cut_mid_run_thaws_with_its_pager() {
    let (parent, child) = (parent(), child_module(&child_src(false)));
    let (art, args) = artifact(&parent, &child, |w| arm_freeze_after_backedges(w, 2000));
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &child, &args),
        Some(Out::Ret(EXPECT)),
        "the thawed child resumes its loop and its pager serves the page it touches next"
    );
}

/// A child the freeze rang before it ran unwinds at its first safepoint, before it touches a page,
/// and thaws to the same answer.
#[test]
fn a_paged_child_frozen_before_it_runs_thaws_with_its_pager() {
    let (parent, child) = (parent(), child_module(&child_src(false)));
    let (art, args) = artifact(&parent, &child, |w| {
        write_state(w, temen_durable::STATE_UNWINDING)
    });
    assert_eq!(
        thaw_on(Engine::Interp, &art, &parent, &child, &args),
        Some(Out::Ret(EXPECT))
    );
}

/// The Cranelift JIT has no demand paging (#2164), so it refuses to thaw a paged child rather than
/// run it with no pager. A routed run folds a pager-capable program to the interpreter instead.
#[test]
fn the_jit_refuses_to_thaw_a_paged_child() {
    let (parent, child) = (parent(), child_module(&child_src(false)));
    let (art, args) = artifact(&parent, &child, |w| arm_freeze_after_backedges(w, 2000));
    assert_eq!(thaw_on(Engine::Jit, &art, &parent, &child, &args), None);
}

/// A child that faults after the freeze rang it, before its first safepoint, can never unwind: its
/// pager has unwound and serves nothing. It is unreached, and the freeze is refused whole, never
/// written as if the child had unwound.
#[test]
fn a_paged_child_that_faults_before_it_can_unwind_refuses_the_freeze() {
    let (parent, child) = (parent(), child_module(&child_src(true)));
    let (_, snap, fh, _) = run_from(&parent, &child, |w| {
        write_state(w, temen_durable::STATE_UNWINDING)
    });
    assert!(fh.captured_detached().is_empty(), "nothing was captured");
    assert_eq!(fh.unreached_detached().len(), 1, "the child is unreached");
    assert!(matches!(
        temen_snapshot::freeze(&parent, &snap, &fh),
        Err(temen_snapshot::FreezeError::DetachedUnreached { .. })
    ));
}
