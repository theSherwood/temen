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
use temen_interp::Host;
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

// The oracle is **not** pinned alongside these, and that is a finding rather than an omission:
// it answers the same program `WAIT_TIMED_OUT` after ten seconds, not `ThreadFault`. Its
// `run_deadlocked` predicate never consults `wait_waiters`, so an unsatisfiable futex wait is ended
// by the `MAX_WAIT` anti-wedge backstop instead of being detected — even though `thread_wait`'s own
// comment says the JIT surfaces `WAIT_DEADLOCK` as a `ThreadFault` *"matching the interpreter"*.
// It does not match. Which status is right is a semantics call, filed as #1624; a ten-second test
// pinning a behaviour we think is wrong would be poor value here.

/// **A mutual deadlock resolves for both vCPUs.** The root spawns a sibling; each parks in an
/// infinite wait on a word only the *other* would ever store. Both are blocked, so `live == parked`
/// and neither can be satisfied — the "mutual" half of the predicate's own comment, which the
/// lone-waiter case above does not reach.
///
/// This is the case that made the fix more than hoisting a check. The first waiter to park is
/// already asleep when the second one makes `live == parked` true, so detection used to wait for a
/// poll; `ParkGuard` now wakes the key's condvar as it increments `parked`, which is also what lets
/// loom model this at all (it has no timeouts).
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
