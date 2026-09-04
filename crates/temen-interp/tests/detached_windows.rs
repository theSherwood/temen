//! PROCESS.md §5 — **detached windows**: a child spawned through a `WindowMinter` capability
//! (`Instantiator.instantiate_detached`, op 15) runs in a fresh platform window *outside* its
//! spawner's — no ancestor below the platform holds read authority, and the child attests
//! `window_exposed = false` (the jacl distrust-spawner trust anchor). Detachment severs READ,
//! not lifecycle (the spawner keeps kill/join) and not coordination (live offers work — the
//! linkage is the powerbox, not the window). The minter's byte quota is host-enforced at each
//! mint; misses refuse probeably.

use std::sync::Arc;
use temen_interp::{run_with_host, Host, Value};

fn module(text: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// The detached serving child: its own module, own offer, own serve loop — same shape as the
/// separate-module (nested) server, now in a window the parent cannot see.
const DETACHED_SERVER: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

/// The parent: spawns the server DETACHED (op 15 — minter, module, no grants), wires its live
/// offer (`child_offer` — identical to the nested form: the linkage is the powerbox Arc), calls
/// `add(40, 2)` through it (park, serve, reply), joins. Composite: join(1)*100 + 42 = 142.
const DETACHED_CALLER: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vB = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vex = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vB, vex)
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vcap (va, vb)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vB)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}
"#;

#[test]
fn a_detached_child_serves_live_calls_from_a_window_its_parent_cannot_see() {
    let a = module(DETACHED_CALLER);
    let b = module(DETACHED_SERVER);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "detached spawn → child_offer → park/serve/reply → join: detachment severs read, not coordination"
    );
}

/// A child that reports its own `self.attest` — the non-interposable trust anchor.
const ATTEST_MOD: &str = r#"
memory 12

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  va = call.cap 4294967295 4 () -> (i64) vz ()
  return va
  }
}
"#;

/// The trust anchor, side by side: the SAME module spawned **nested** (op 5, a carve of the
/// parent's window) attests `tier 1 | window_exposed` = 257; spawned **detached** (op 15) it
/// attests `tier 1` alone = 1 — the distrust-spawner report, platform-vouched. Composite:
/// nested*1000 + detached = 257_001.
const ATTEST_BOTH: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  ve = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  vN = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ve, voff, vlog, vq)
  vz = i64.const 0
  vD = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vjN = call.cap 6 1 (i32) -> (i64) v0 (vN)
  vjD = call.cap 6 1 (i32) -> (i64) v0 (vD)
  vk = i64.const 1000
  vm = i64.mul vjN vk
  vs = i64.add vm vjD
  return vs
  }
}
"#;

#[test]
fn a_detached_child_attests_window_unexposed_where_a_nested_one_attests_exposed() {
    let a = module(ATTEST_BOTH);
    let b = module(ATTEST_MOD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(257_001)],
        "nested attests tier|exposed (257); detached attests tier alone (1)"
    );
}

/// The minter's quota is the attenuation: with exactly one window's worth (4096 bytes), the
/// first detached spawn succeeds and the second refuses probeably (`-EINVAL`, nothing
/// charged) — a numeric quota, host-enforced at mint. Composite: first_failed*10 +
/// second_failed = 0*10 + 1 = 1.
const QUOTA_EXHAUSTS: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vfst = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vsnd = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vfst)
  vzero = i32.const 0
  vf1 = i32.lt_s vfst vzero
  vf2 = i32.lt_s vsnd vzero
  vten = i32.const 10
  vm = i32.mul vf1 vten
  vs = i32.add vm vf2
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

#[test]
fn the_minter_quota_bounds_detached_mints() {
    let a = module(QUOTA_EXHAUSTS);
    let b = module(ATTEST_MOD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 12); // exactly one 2^12 window
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(1)],
        "first mint fits the quota, second refuses probeably"
    );
}

/// A forged minter handle (the Instantiator handle itself, wrong type) refuses probeably —
/// the minter is spawn evidence, and no evidence means no detached window, never a trap.
const FORGED_MINTER: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v0
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vs = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = i64.extend_i32_s vs
  return vr
  }
}
"#;

#[test]
fn a_forged_minter_refuses_probeably() {
    let a = module(FORGED_MINTER);
    let b = module(ATTEST_MOD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm)],
        &mut fuel,
        &mut host,
    )
    .expect("run — refusal, not a trap");
    assert_eq!(r, vec![Value::I64(-22)], "-EINVAL, probeable");
}

/// A detached child that parks on an anonymous futex word in **its own** window (offset 32768) with
/// a 300 ms timeout and reports the wait status: `WAIT_TIMED_OUT` = 2 if nothing woke it.
const DETACHED_WAITER: &str = r#"
memory 16

func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 32768
  vexp = i32.const 0
  vto = i64.const 300000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

/// The parent spawns the waiter DETACHED, then hammers `atomic.notify` on the same offset 32768 of
/// **its own** window, OR-ing the woken counts, and joins. Composite: `join + 10 * woke_any`.
const NOTIFYING_PARENT: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vD = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vi0 = i64.const 0
  vw0 = i32.const 0
  br 1(v0, vD, vi0, vw0)
  }
block 1 (vh: i32, vd: i32, vi: i64, vw: i32) {
  vaddr = i64.const 32768
  vcnt = i32.const 1
  vn = atomic.notify vaddr vcnt
  vw2 = i32.or vw vn
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 200000
  vdone = i64.eq vi2 vlim
  br_if vdone 2(vh, vd, vw2) 1(vh, vd, vi2, vw2)
  }
block 2 (vh2: i32, vd2: i32, vwf: i32) {
  vj = call.cap 6 1 (i32) -> (i64) vh2 (vd2)
  vwe = i64.extend_i32_u vwf
  vk = i64.const 10
  vm = i64.mul vwe vk
  vs = i64.add vj vm
  return vs
  }
}
"#;

/// #1283: a detached child's anonymous futex words are **its own**. The parent and the detached child
/// both have `window.base() == 0`, so before the fix the same guest offset produced the same
/// `FutexKey::Anon` and the parent's `notify` woke the child (`join` = 0, `woke_any` = 1 → 10). Now
/// the key carries the backing identity: the child times out untouched (2) and no parent notify ever
/// finds a waiter (0). The only timing dependence is that a slow child park can make a *buggy* build
/// pass; a fixed build can never fail.
#[test]
fn a_detached_child_does_not_rendezvous_with_its_parent_on_anonymous_memory() {
    let a = module(NOTIFYING_PARENT);
    let b = module(DETACHED_WAITER);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 17);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(2)],
        "detached child's wait must time out (2) with no parent notify landing (woke_any = 0)"
    );
}

/// A detached child that reads the first argv word at `module_args_base() + 8` and adds its attest (1).
const ARGV_CHILD: &str = r#"
memory 16

func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  vs = i64.add va vat
  return vs
  }
}
"#;

/// The parent stores the args blob (`argc 1`, `"hello-detached\0"`) at 18432 in ITS window and passes
/// `(18432, 24)` as the optional 8th/9th op-15 args (#1286): the host copies it to the child's
/// `module_args_base()` before start — the detached twin of the op-13 "data segment in the carve".
const ARGV_PARENT: &str = r#"
memory 17

func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vb0 = i64.const 18432
  vw0 = i64.const 1
  i64.store vb0 vw0
  vb1 = i64.const 18440
  vw1 = i64.const 7306014452085450088
  i64.store vb1 vw1
  vb2 = i64.const 18448
  vw2 = i64.const 28265164885364
  i64.store vb2 vw2
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vap = i64.const 18432
  val = i64.const 24
  vD = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq, vap, val)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vD)
  return vj
  }
}
"#;

#[test]
fn a_detached_child_receives_the_spawn_time_args_payload() {
    let a = module(ARGV_PARENT);
    let b = module(ARGV_CHILD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 16);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    // "hello-de" as a little-endian i64, plus attest = 1 (tier 1, window_exposed = false).
    assert_eq!(
        r,
        vec![Value::I64(i64::from_le_bytes(*b"hello-de") + 1)],
        "the payload landed at the child's args base; the child is unexposed"
    );
}

/// A detached child that `vm_map`s past its declared 64 KiB window (via the child-manifest `vm_map`
/// import), stores the argv word on the grown page, loads it back and returns it plus attest.
const GROWING_CHILD: &str = r#"
memory 16
import 0 "vm_map" (i64, i64, i32) -> (i64)

func (i64) -> (i64) {
block 0 (v0: i64) {
  vab = i64.const 16520
  va = i64.load vab
  vz = i32.const 0
  vat = call.cap 4294967295 4 () -> (i64) vz ()
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vg = call.import 0 (voff, vlen, vprot)
  vp = i64.const 65600
  i64.store vp va
  vld = i64.load vp
  vs = i64.add vld vat
  return vs
  }
}
"#;

/// #1286 — a detached window **grows**: its starter `AddressSpace` spans the reservation (a root's
/// shape), not the declared size, so a `vm_map` past 64 KiB is admitted and the grown page is usable.
/// (Bounding the grant to the declared window refused the map and the store then faulted.)
#[test]
fn a_detached_child_grows_past_its_declared_window() {
    let a = module(ARGV_PARENT);
    let b = module(GROWING_CHILD);
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let hw = host.grant_window_minter(1 << 16);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm), Value::I32(hw)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(i64::from_le_bytes(*b"hello-de") + 1)],
        "the word round-tripped through a page above the declared window"
    );
}
