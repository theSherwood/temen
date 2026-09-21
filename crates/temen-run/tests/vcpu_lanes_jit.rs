//! D66 "broad lanes" (#1600, #1586) — **a domain's own `thread.spawn` vCPUs are lane-bounded on the
//! JIT too**, not only its §14 child domains.
//!
//! The owner ruling is that parallelism is a granted resource for the whole subtree. The interpreter
//! read it that way from the start: its `dispatch` gates every vCPU, including `thread.spawn`
//! siblings (`temen-interp/tests/lanes.rs`). The JIT did not gate them at all, so the same program
//! under the same cap ran four siblings at once on one engine and one at a time on the other. These
//! are the JIT half, written cross-engine so the two cannot drift apart again unnoticed.
//!
//! A vCPU stays 1:1 with an OS thread (D56 is unchanged). What the lane bounds is how many may be
//! *running*: the thread exists, it queues. So the load-bearing half is that **every park gives the
//! lane back** — under a cap of 1 a waiter that kept it would prevent the very peer that would
//! satisfy it from running, and each of these programs would deadlock instead of completing.

use core::ffi::c_void;
use temen_interp::{run_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};

fn grant_hooks(host: *mut Host) -> GrantChildHooks {
    temen_run::production_grant_hooks(temen_run::CapCtx::Raw(host))
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run `m` on both engines under `lane_cap` and return `(interp, jit)`.
///
/// The JIT learns its lane through the grant hooks, which carry the host's `(domain id, lane cap)` —
/// the same pair a §14 child's chain gets its parent entry from. That is why these runs pass hooks
/// even though none of these programs has an `Instantiator`.
fn both(m: &temen_ir::Module, lane_cap: i64) -> (i64, i64) {
    let one = |cap: i64, jit: bool| -> i64 {
        let mut host = Host::new();
        host.set_lane_cap(cap);
        if jit {
            let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
                m,
                0,
                &[],
                &[],
                temen_ir::DEFAULT_RESERVED_LOG2,
                temen_run::cap_thunk,
                &mut host as *mut Host as *mut c_void,
                Some(temen_run::module_resolver),
                Some(grant_hooks(&mut host as *mut Host)),
            )
            .expect("jit run");
            match jo {
                JitOutcome::Returned(ref v) => v.first().copied().unwrap_or(-1),
                ref o => panic!("jit ended abnormally: {o:?}"),
            }
        } else {
            let mut fuel = u64::MAX;
            match run_with_host(m, 0, &[], &mut fuel, &mut host)
                .expect("interp run")
                .first()
            {
                Some(Value::I64(x)) => *x,
                other => panic!("unexpected interp result {other:?}"),
            }
        }
    };
    (one(lane_cap, false), one(lane_cap, true))
}

/// Four `thread.spawn` siblings, each: bump a shared running counter, note a violation if it was
/// already non-zero, spin, drop the counter. The root joins all four and returns the violation
/// count. Byte-for-byte the interpreter's `lanes.rs` program, so the two engines are answering the
/// same question. Above the #1094 NULL guard: RUN at 16392, VIOL at 16400, handles at 16408+.
const FOUR_SPINNERS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vfour = i64.const 4
  vz2 = i64.const 0
  vlt = i64.lt_u vi vfour
  br_if vlt 2(vi) 3(vz2)
}
block 2 (vi2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vi2
  vh = i64.const 16408
  v8 = i64.const 4
  vm = i64.mul vi2 v8
  va = i64.add vh vm
  i32.store va vt
  vone = i64.const 1
  vn = i64.add vi2 vone
  br 1(vn)
}
block 3 (vj: i64) {
  vfour2 = i64.const 4
  vlt2 = i64.lt_u vj vfour2
  br_if vlt2 4(vj) 5()
}
block 4 (vj2: i64) {
  vh2 = i64.const 16408
  v82 = i64.const 4
  vm2 = i64.mul vj2 v82
  va2 = i64.add vh2 vm2
  vt2 = i32.load va2
  vr = thread.join vt2
  vone2 = i64.const 1
  vn2 = i64.add vj2 vone2
  br 3(vn2)
}
block 5 () {
  vviol = i64.const 16400
  vv = i64.atomic.load vviol
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vrun = i64.const 16392
  vone = i64.const 1
  vold = i64.atomic.rmw.add vrun vone
  vge = i64.ge_s vold vone
  br_if vge 1() 2()
}
block 1 () {
  vviol = i64.const 16400
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vviol vone1
  br 2()
}
block 2 () {
  vz = i64.const 0
  br 3(vz)
}
block 3 (vi: i64) {
  vlim = i64.const 400000
  vlt = i64.lt_u vi vlim
  vone3 = i64.const 1
  vi1 = i64.add vi vone3
  br_if vlt 3(vi1) 4()
}
block 4 () {
  vrun2 = i64.const 16392
  vneg = i64.const -1
  vd = i64.atomic.rmw.add vrun2 vneg
  vz4 = i64.const 0
  return vz4
  }
}
"#;

/// **The dispatch bound, on both engines.** A lane cap of 1 serializes four spinning siblings: each
/// holds the lane only while it is running, and a vCPU takes one before any guest code. The JIT used
/// to report violations here while the interpreter reported none — the divergence these tests exist
/// to close.
#[test]
fn a_lane_cap_of_one_serializes_four_spinning_siblings_on_both_engines() {
    let m = module(FOUR_SPINNERS);
    let (interp, jit) = both(&m, 1);
    assert_eq!(interp, 0, "the oracle: no two siblings overlapped");
    assert_eq!(
        jit, 0,
        "the JIT agrees — a vCPU runs only while it holds a lane"
    );
}

/// The default is unbounded and nothing changes: the same program completes on both engines. Its
/// violation count is whatever the hardware allowed and is deliberately not asserted — that is the
/// *mutation* for the test above, not a pin.
#[test]
fn an_unbounded_lane_leaves_both_engines_untouched() {
    let m = module(FOUR_SPINNERS);
    let (interp, jit) = both(&m, -1);
    assert!(interp >= 0 && jit >= 0, "both completed");
}

/// Two vCPUs rendezvous through the futex: the root parks in `atomic.wait` on a word the spawned
/// sibling stores and notifies, then joins it. Returns `100·status + word`, so `1` means the root
/// was woken (status 0) and read the sibling's `1`.
const WAIT_NOTIFY_PAIR: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vx = i64.const 16392
  ve = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vx ve vinf
  vr = thread.join vt
  vv = i32.load vx
  vs64 = i64.extend_i32_u vs
  vk = i64.const 100
  vm = i64.mul vs64 vk
  vv64 = i64.extend_i32_u vv
  vsum = i64.add vm vv64
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vx = i64.const 16392
  vone = i32.const 1
  i32.atomic.store vx vone
  vn = atomic.notify vx vone
  vz = i64.const 0
  return vz
  }
}
"#;

/// **A parked vCPU holds no lane.** Under a cap of 1 the root must step aside in `atomic.wait`, or
/// the sibling that would store-and-notify can never run and the program deadlocks. That it
/// completes at all is the assertion; the value pins that the wake was a real notify.
///
/// The root also `join`s, so this covers the join park on the same run.
#[test]
fn a_cap_of_one_still_lets_a_waiting_vcpu_be_woken_by_the_sibling_it_waits_for() {
    let m = module(WAIT_NOTIFY_PAIR);
    let (interp, jit) = both(&m, 1);
    // The oracle does not order the two, so the store may land before the root registers its wait —
    // the `futex_cross_domain.rs` racy shape. Either way it completes and reads the sibling's word.
    assert!(
        interp == 1 || interp == 101,
        "the oracle completes either way: {interp}"
    );
    assert!(jit == 1 || jit == 101, "the JIT completes too: {jit}");
}

/// The root spawns a sibling that does nothing but return a value, then joins it under a cap of 1.
/// The sibling cannot run until the joining root gives up its lane, so this deadlocks unless the
/// join park releases.
const JOIN_UNDER_CAP: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vv = i64.const 7
  return vv
  }
}
"#;

/// **A joining vCPU holds no lane.** Cap 1, the root joins the only sibling; it completes with the
/// sibling's `7` on both engines rather than hanging.
#[test]
fn a_cap_of_one_still_lets_a_joining_vcpu_reap_the_sibling_it_joins() {
    let m = module(JOIN_UNDER_CAP);
    assert_eq!(both(&m, 1), (7, 7));
}

/// A lane of **0** grants no parallelism at all, so nothing in the domain can ever run. That is
/// unsatisfiable rather than slow, and both engines answer it as a `ThreadFault` rather than hanging
/// — the #5 "errors are values, never a hang" line applied to a resource that can be granted empty.
#[test]
fn a_lane_of_zero_faults_rather_than_hanging() {
    let m = module(JOIN_UNDER_CAP);
    let mut host = Host::new();
    host.set_lane_cap(0);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host);
    assert!(
        r.is_err(),
        "the oracle refuses a domain granted no parallelism: {r:?}"
    );

    let mut host = Host::new();
    host.set_lane_cap(0);
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        &m,
        0,
        &[],
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        Some(grant_hooks(&mut host as *mut Host)),
    )
    .expect("jit run");
    assert!(
        !matches!(jo, JitOutcome::Returned(_)),
        "the JIT refuses it too rather than hanging: {jo:?}"
    );
}
