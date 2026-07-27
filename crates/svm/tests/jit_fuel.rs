//! Safepoint-anchored **counted fuel on the JIT** (INTERP_PERF.md "Fuel unification", step 3). Unlike
//! the §5 kill-path (`jit_killpath.rs` — a host-written cell polled asynchronously, so interp and JIT
//! stop at *different* points and only the *outcome* matches), counted fuel is a **deterministic guest
//! budget**: the lowering decrements a host-owned `u64` at every function entry and taken back-edge and
//! traps `OutOfFuel` when it would underflow. So a runaway traps at a fixed point with no watchdog, and
//! a finite run charges a countable amount — the same unit the tree-walker and bytecode engines now
//! charge, so fuel becomes a checked cross-engine quantity rather than an excluded difference.
//!
//! Exact interp/JIT fuel *count* parity now holds: the interpreters were reconciled to charge the
//! top-level entry function too (`super::drive_arc` / bytecode `drive`), matching the JIT's entry-
//! prologue charge, so all three engines consume the identical amount on the same run and exhaust at
//! the identical safepoint. These tests assert strict equality (`== interp`), and the differential
//! harnesses assert `OutOfFuel` parity rather than excluding it (`bytecode_diff`, `jit_fuzz`).

use svm_interp::{run, Trap, Value};
use svm_jit::{compile_and_run, compile_and_run_with_host_fuel, JitOutcome, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

/// A non-terminating **intra-function loop** (block 1 branches to itself forever) — caught by the
/// per-back-edge fuel charge.
const INFINITE_LOOP: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

/// A non-terminating **tail-recursion** (function 0 tail-calls itself forever) — runs in O(1) native
/// stack, so only the *function-entry* fuel charge (the callee prologue) can stop it.
const INFINITE_TAIL_RECURSION: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return_call 0(v2)
  }
}
";

/// A **finite** countdown N → 0: block 1 loops back to itself while the counter is non-zero (a taken
/// back-edge each iteration), then exits forward to block 2 (no charge). The back-edge is taken `N-1`
/// times; both engines also charge the single top-level entry, so each charges `N` on the same run.
const FINITE_COUNTDOWN: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  br 1(v0)
}
block 1 (v1: i32) {
  v2 = i32.const -1
  v3 = i32.add v1 v2
  br_if v3 1(v3) 2(v3)
}
block 2 (v4: i32) {
  return v4
  }
}
";

/// Run `src` on the JIT with a counted-fuel budget of `budget`; returns the outcome and the *remaining*
/// fuel the guest wrote back into the cell.
fn jit_fuel(src: &str, args: &[i64], budget: u64) -> (JitOutcome, u64) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    // The guest decrements this cell in place through the baked address; after the run returns we read
    // the remainder back (no concurrent access — a single guest thread owns the budget).
    let mut cell: u64 = budget;
    let outcome = compile_and_run_with_host_fuel(
        &m,
        0,
        args,
        svm_run::cap_thunk,
        core::ptr::null_mut(),
        &mut cell as *mut u64,
    )
    .expect("jit compiles");
    (outcome, cell)
}

#[test]
fn jit_fuel_stops_infinite_loop() {
    // Deterministic: a small budget bounds the loop; the per-back-edge charge trips at exactly zero.
    let (outcome, remaining) = jit_fuel(INFINITE_LOOP, &[0i64], 1_000);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "counted fuel must stop the runaway loop"
    );
    assert_eq!(remaining, 0, "an exhausted budget is drained to zero");
}

#[test]
fn jit_fuel_stops_infinite_tail_recursion() {
    // The function-entry charge (the callee prologue), not the back-edge one, is what catches this —
    // tail calls never grow the stack, so without it the guest would spin forever.
    let (outcome, remaining) = jit_fuel(INFINITE_TAIL_RECURSION, &[0i64], 1_000);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "counted fuel must stop the runaway tail recursion"
    );
    assert_eq!(remaining, 0, "an exhausted budget is drained to zero");
}

#[test]
fn jit_fuel_finite_run_completes_and_charges() {
    const N: i32 = 1000;
    const BUDGET: u64 = 10_000_000;

    // Interpreter reference: run with the same generous budget and read how much it consumed.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let mut interp_fuel = BUDGET;
    let interp = run(&m, 0, &[Value::I32(N)], &mut interp_fuel).expect("interp ok");
    assert_eq!(interp, vec![Value::I32(0)], "countdown returns 0");
    let interp_consumed = BUDGET - interp_fuel;
    assert!(interp_consumed > 0, "the loop charges some fuel");

    // JIT: same program, same budget — must return the same result and charge the *exact* same amount.
    // Both engines now charge the top-level entry (fuel unification), so this is strict equality.
    let (outcome, remaining) = jit_fuel(FINITE_COUNTDOWN, &[N as i64], BUDGET);
    assert_eq!(
        outcome,
        JitOutcome::Returned(vec![0]),
        "an armed-but-sufficient finite run completes normally"
    );
    let jit_consumed = BUDGET - remaining;
    assert_eq!(
        jit_consumed, interp_consumed,
        "JIT fuel must exactly match interp after top-level-entry reconciliation"
    );
}

#[test]
fn jit_fuel_exhausts_one_short() {
    // A budget one below the exact need must trap `OutOfFuel` rather than complete — proving the charge
    // is exact at the boundary, not approximate. After fuel unification both engines charge the
    // top-level entry, so they consume the *same* amount and exhaust at the *same* point: budget
    // `interp_consumed` completes on both; `interp_consumed - 1` traps on both.
    const N: i32 = 1000;
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let mut interp_fuel = 10_000_000u64;
    run(&m, 0, &[Value::I32(N)], &mut interp_fuel).expect("interp ok");
    let interp_consumed = 10_000_000u64 - interp_fuel;
    let short = interp_consumed - 1;

    // Interp one short: traps.
    let mut interp_short = short;
    assert_eq!(
        run(&m, 0, &[Value::I32(N)], &mut interp_short),
        Err(Trap::OutOfFuel),
        "interp one short of its exact need must trap OutOfFuel"
    );
    // JIT one short: traps at the same boundary (exact interp/JIT fuel parity).
    let (outcome, remaining) = jit_fuel(FINITE_COUNTDOWN, &[N as i64], short);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT one short of the exact need must trap OutOfFuel, not complete"
    );
    assert_eq!(remaining, 0);
}

#[test]
fn jit_unarmed_path_is_unchanged() {
    // Arming is strictly opt-in: the ordinary (fuel-un-armed) entry runs the same finite program to
    // completion, byte-identical to before the feature existed.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let jit = compile_and_run(&m, 0, &[1000i64]).expect("jit");
    assert_eq!(jit, JitOutcome::Returned(vec![0]));
}
