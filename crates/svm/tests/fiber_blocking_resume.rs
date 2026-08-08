//! I48 — **`cont.resume.block` is advisory**: returning `FIBER_PARKED (3)` is a conforming
//! result, so the fast backends (bytecode, Cranelift JIT) and the deterministic explorer alias
//! it to `cont.resume` (they have no M:N idle core), while only the tree-walk oracle actually
//! idles the vCPU. This pins that all three engines agree on the **final** guest-observable
//! result of a looping blocking-resume kernel — the oracle reaching it via idle-and-wake, the
//! others via the spin the guest's loop absorbs (invariant 9). The oracle's *idle* behaviour
//! (and that it does not spin) is pinned separately in `svm-interp/tests/blocking_resume.rs`.
//!
//! The kernel must **loop** on the status: a single `cont.resume.block` would return
//! FIBER_PARKED on the fast backends (the advisory downgrade) and diverge — which is exactly the
//! contract (may return FIBER_PARKED; loop for completion). The composite reads only the final
//! `(status, value)`, never the poll count, so idle-vs-spin is invisible to it.

use svm_run::{instantiate, Backend, Outcome, RunConfig, Value};

fn run_on(src: &str, backend: Backend) -> i64 {
    let module = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&module).expect("verify");
    let instance = instantiate(module).expect("instantiate");
    let run = instance
        .run(backend, &RunConfig::default())
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
    match run.outcome {
        Outcome::Returned(vals) => match vals.as_slice() {
            [Value::I64(x)] => *x,
            other => panic!("{backend:?}: expected one i64, got {other:?}"),
        },
        other => panic!("{backend:?}: did not return: {other:?}"),
    }
}

fn pin_all(src: &str, want: i64) {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        assert_eq!(run_on(src, backend), want, "{backend:?}");
    }
}

/// A poll loop over `cont.resume.block` of a fiber that does a 10 ms timed `atomic.wait` (never
/// notified). On the oracle the very first `cont.resume.block` idles until the timer fires, then
/// completes — the loop body runs ~once. On the fast backends `cont.resume.block` aliases to
/// `cont.resume`, so it returns FIBER_PARKED and the loop spins until the timer fires (the
/// bytecode logical clock / the JIT real deadline). All three converge on the fiber's
/// `WAIT_TIMED_OUT (2) * 1000 + 42 = 2042`.
const POLL_BLOCK_TIMED: &str = r#"
memory 16
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  br 1(v2)
}
block 1 (vk: i64) {
  vz = i64.const 0
  vs, vv = cont.resume.block vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vv) 1(vk)
}
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vk = i64.const 1000
  vst64 = i64.extend_i32_s vst
  va = i64.mul vst64 vk
  v42 = i64.const 42
  vr = i64.add va v42
  return vr
  }
}
"#;

#[test]
fn blocking_resume_poll_loop_agrees_on_all_backends() {
    // WAIT_TIMED_OUT(2)*1000 + 42 — the oracle reaches it by idling, the fast backends by
    // spinning the loop; the final result is identical (the advisory contract).
    pin_all(POLL_BLOCK_TIMED, 2042);
}

/// `cont.resume.block` on a fiber that **returns immediately** (never parks) is exactly
/// `cont.resume`: the resume switches in, the fiber runs to completion, and the first poll
/// yields `(FIBER_RETURNED=1, value)` on every backend — no idle, no spin, no divergence. Pins
/// that the blocking variant is a pure no-op when the fiber doesn't park. Fiber returns 7;
/// composite = RETURNED(1)*100 + 7 = 107.
const BLOCK_NO_PARK: &str = r#"
memory 16
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  vz = i64.const 0
  vs, vv = cont.resume.block v2 vz
  vk = i64.const 100
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v7 = i64.const 7
  return v7
  }
}
"#;

#[test]
fn blocking_resume_of_a_non_parking_fiber_is_plain_resume() {
    pin_all(BLOCK_NO_PARK, 107);
}

/// I48 parity — a **single** `cont.resume.block` with **no guest loop** of a fiber that does a
/// 10 ms timed `atomic.wait`. On a backend that genuinely idles, this returns the fiber's real
/// completion `WAIT_TIMED_OUT(2)*100 + 0 = 102`; a backend that merely aliased the op to
/// `cont.resume` (spins) would return the transient `FIBER_PARKED(3)*100 = 300` — the guest has no
/// loop to absorb it. So `== 102` on a no-loop kernel is a direct behavioral proof the vCPU idled
/// and re-resumed, not spun. Pinned on the tree-walk oracle and the **bytecode** engine (which
/// also backs the wasm-jit's `InterpDriven` fold); the Cranelift JIT still takes the advisory
/// downgrade (its OS-thread idle is the follow-up slice), so it is excluded here.
const BLOCK_NO_LOOP_TIMED: &str = r#"
memory 16
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  vz = i64.const 0
  vs, vv = cont.resume.block v2 vz
  vk = i64.const 100
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn no_loop_blocking_resume_idles_on_tree_walk_and_bytecode() {
    // WAIT_TIMED_OUT(2)*100 + 0 — reached only by actually idling until the timer, then resuming.
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        assert_eq!(
            run_on(BLOCK_NO_LOOP_TIMED, backend),
            102,
            "{backend:?}: a no-loop cont.resume.block must idle-and-complete, not spin to 300"
        );
    }
}
