//! **#1623 — an infinite `atomic.wait` with no possible notifier is an error, not a hang.**
//!
//! INVARIANTS #5: an error is a value, never a hang. `futex_wait` already knows the answer — it
//! breaks `WAIT_DEADLOCK` when `peers_live()` is false, and `thread_wait` turns that into a
//! `ThreadFault` "matching the interpreter, never a guest-visible wait status". But both the check
//! and the bounded re-check used to sit inside an `epoch_addr != 0 || unwind_base != 0` guard, so on
//! an ordinary non-durable run with no kill path armed the wait fell through to an **untimed**
//! `cv.wait` that re-evaluated nothing. The detector did not exist unless something else armed the
//! run, and the JIT does not clamp an infinite wait the way the interpreter does.
//!
//! These runs arm nothing on purpose. That is the shape that hung.

use core::ffi::c_void;
use std::time::{Duration, Instant};
use temen_interp::{run_with_host, Host, Trap};
use temen_jit::compile_and_run_capture_reserved_with_host_ex;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run `m` on an **unarmed** JIT domain: no kill-path interrupt cell, no durable window, no grant
/// hooks. Returns `Err` if the run ended abnormally — which is what a detected deadlock looks like.
fn jit_unarmed(m: &temen_ir::Module) -> Result<temen_jit::JitOutcome, String> {
    let mut host = Host::new();
    let (jo, _) = compile_and_run_capture_reserved_with_host_ex(
        m,
        0,
        &[],
        &[],
        temen_ir::DEFAULT_RESERVED_LOG2,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut c_void,
        Some(temen_run::module_resolver),
        None,
    )
    .map_err(|e| format!("{e:?}"))?;
    Ok(jo)
}

/// Run `m` on the tree-walking oracle. Returns its `Trap` and how long the run took — the elapsed
/// time is half the assertion for #1624, because the bug it fixed was *also* a ten-second stall.
fn interp_trap(m: &temen_ir::Module) -> (Trap, Duration) {
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let t0 = Instant::now();
    let r = run_with_host(m, 0, &[], &mut fuel, &mut host);
    let dt = t0.elapsed();
    match r {
        Err(t) => (t, dt),
        Ok(v) => panic!("an unsatisfiable wait must not return a value, got {v:?} in {dt:?}"),
    }
}

/// Well under the interpreter's 10 s `MAX_WAIT` backstop, well over CI jitter: proof the verdict came
/// from the deadlock predicate and not from the clamp that used to decide it.
const PROMPT: Duration = Duration::from_secs(4);

/// The lone waiter: an infinite wait on a word nothing will ever store or notify, with no sibling
/// that could notify it. `live == parked == 1`, so no notify can ever arrive.
const LONE_INFINITE_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16392
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

/// **The hang, closed.** An unarmed JIT run must end rather than block forever. It used to print
/// nothing and sit until the test harness was killed.
#[test]
fn a_lone_infinite_wait_ends_instead_of_hanging_an_unarmed_jit_run() {
    let m = module(LONE_INFINITE_WAIT);
    let jo = jit_unarmed(&m);
    assert!(
        !matches!(jo, Ok(temen_jit::JitOutcome::Returned(_))),
        "an unsatisfiable wait must not come back as an ordinary result: {jo:?}"
    );
}

/// **The oracle agrees, and promptly (#1624).** `thread_wait` has always claimed it surfaces
/// `WAIT_DEADLOCK` as a `ThreadFault` *"matching the interpreter"*. It did not: `run_deadlocked` never
/// consulted `wait_waiters`, so a futex-parked vCPU was invisible to it and the run was ended by the
/// `MAX_WAIT` anti-wedge backstop instead — ten seconds, then `WAIT_TIMED_OUT`, a status nobody asked
/// for, chosen by a constant whose own doc calls it "never semantics".
///
/// Both halves are pinned here, because a fix that traded the wrong status for the wrong latency
/// would satisfy the first assertion alone.
#[test]
fn the_oracle_faults_on_a_lone_infinite_wait_without_waiting_out_the_backstop() {
    let (trap, dt) = interp_trap(&module(LONE_INFINITE_WAIT));
    assert_eq!(
        trap,
        Trap::ThreadFault,
        "an unsatisfiable wait is an error, not a timeout ({dt:?})"
    );
    assert!(
        dt < PROMPT,
        "the verdict must come from the predicate, not the 10 s MAX_WAIT backstop: took {dt:?}"
    );
}

/// **A mutual deadlock resolves for both vCPUs.** The root spawns a sibling; each parks in an
/// infinite wait on a word only the *other* would ever store. Both are blocked, so `live == parked`
/// and neither can be satisfied — the "mutual" half of the predicate's own comment, which the
/// lone-waiter case above does not reach.
///
/// The **exit** wake is what carries it, not the park. The waiter that parks second checks
/// `peers_live` after its own park is counted, so it breaks without being woken at all; the first is
/// freed by the second one finishing — a completing vCPU drops `live` and broadcasts, as
/// `Domain::child_finished` and `run_child` do. (An earlier draft of #1623 added a notify to
/// `ParkGuard` on the theory that the sleeping waiter had to learn its peer had blocked. Mutating it
/// back out showed the loom model and both of these tests passing without it, so it came out again.)
const MUTUAL_DEADLOCK: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vaddr = i64.const 16392
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vr = thread.join vt
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16400
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

#[test]
fn a_mutual_deadlock_ends_instead_of_hanging_an_unarmed_jit_run() {
    let m = module(MUTUAL_DEADLOCK);
    let jo = jit_unarmed(&m);
    assert!(
        !matches!(jo, Ok(temen_jit::JitOutcome::Returned(_))),
        "two vCPUs each blocked on the other must not come back as a result: {jo:?}"
    );
}
