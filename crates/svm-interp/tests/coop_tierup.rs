//! #926 slice 2 — native proof of wasm-JIT **tier-up on the cooperative multiplex driver**.
//!
//! [`CoopRun`](svm_interp::bytecode::CoopRun) is the single-thread, no-Worker analogue of the parallel
//! `svm_par_*` driver: it multiplexes every vCPU of a run (the root and its `thread.spawn` descendants)
//! on one host thread, and — when armed with a JIT-eligibility bitmap — pauses to the host on each
//! direct module-0 `Call` to an eligible function, exactly as the single-vCPU [`Vcpu`] does. This test
//! stands in for the browser host **without any wasm**: it services each `CoopEvent::TierUp` by running
//! the callee on a standalone bytecode run (what the emitted `f{func}` region computes) and asserts the
//! whole-run result is **identical** to a pure-interpreter cooperative run of the same guest — so the
//! tier-up marshalling (args in, results out, resume) is exact across a genuinely threaded schedule.
//!
//! The distinguishing coverage over the single-vCPU `vcpu_tierup` test: tier-up fires on **two tasks**
//! — the root and the spawned worker — proving the paused-task delivery routes to the right vCPU across
//! the scheduler's multiplexing, and that a spawned same-module child inherits the eligibility bitmap.
//!
//! Fuel is **bounded** (not `u64::MAX`) so any regression surfaces as an `OutOfFuel` trap rather than a
//! hang; the guests finish in well under the budget, so a bound never perturbs the differential.

use svm_interp::bytecode::TierUpConfig;
use svm_interp::{bytecode, Host, Trap, Value};
use svm_text::parse_module;

const FUEL: u64 = 10_000_000;

// func 0 (root) spawns one worker (func 1), calls the eligible leaf (func 2) itself, joins the worker,
// and sums the two leaf results. func 1 (worker) calls the same leaf on a constant. func 2 is the pure
// all-i64 leaf `f(x) = x*3 + 7` — the tier-up target. `memory 16` + `sp = 0` mirror the proven threaded
// guest shape (a fully-mapped window has a representable scalar extent, so tier-up does not decline).
const SRC: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  v3 = i64.const 3
  vlocal = call 2 (v3)
  vj = thread.join vt
  vr = i64.add vj vlocal
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 5
  vr = call 2 (v5)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v3 = i64.const 3
  vm = i64.mul vx v3
  v7 = i64.const 7
  va = i64.add vm v7
  return va
  }
}
"#;

fn oracle(m: &svm_ir::Module) -> Result<Vec<Value>, Trap> {
    // The differential oracle: a pure-interpreter cooperative run (`drive`, no eligibility bitmap).
    let mut fuel = FUEL;
    bytecode::compile_and_run(m, 0, &[], &mut fuel).expect("supported")
}

/// Drive the cooperative run with tier-up enabled, emulating the browser host: each `TierUp(func,
/// argv)` is serviced by running `func` standalone (what the emitted region computes) and delivering
/// its i64 results back. Returns the whole-run result and the number of tier-ups serviced.
fn coop_tierup_run(
    m: &svm_ir::Module,
    eligible: std::sync::Arc<[bool]>,
) -> (Result<Vec<Value>, Trap>, u32) {
    let tierup = TierUpConfig {
        eligible,
        page_checked: false,
    };
    let mut run = bytecode::CoopRun::new(m, 0, &[], FUEL, Host::new(), Some(tierup))
        .expect("supported")
        .expect("entry in range");
    let mut tierups = 0u32;
    loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => return (Ok(vals), tierups),
            bytecode::CoopEvent::Trapped(t) => return (Err(t), tierups),
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                tierups += 1;
                // A regression that fails to advance the paused task would tier up unboundedly; cap it
                // so the test fails fast instead of hanging (the real runs need at most one per call).
                assert!(
                    tierups < 50,
                    "runaway tier-ups (last func={func}, argv={argv:?})"
                );
                // Emulate `f{func}(win, env, ...argv)`: run the callee standalone over its i64 args.
                let args: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
                let mut fuel = FUEL;
                match bytecode::compile_and_run(m, func, &args, &mut fuel).expect("supported") {
                    Ok(vals) => {
                        let slots: Vec<i64> = vals
                            .iter()
                            .map(|v| match v {
                                Value::I64(x) => *x,
                                Value::I32(x) => *x as i64,
                                _ => panic!("non-integer tier-up result"),
                            })
                            .collect();
                        run.deliver_tierup(&slots);
                    }
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
        }
    }
}

// Bisection probe: does the hand-written threaded guest even terminate under the pure-interpreter
// cooperative oracle (identical to the verified slice-2a behaviour, `eligible = None`)? If this hangs,
// the guest is the problem, not the tier-up path.
#[test]
fn guest_oracle_terminates() {
    let m = parse_module(SRC).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    assert_eq!(oracle(&m), Ok(vec![Value::I64(38)]));
}

#[test]
fn coop_tierup_matches_pure_interp() {
    let m = parse_module(SRC).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    // Only func 2 (the leaf) is eligible; func 0 (root) and func 1 (worker) are the interp-driven
    // callers. Both the root and the spawned worker must therefore tier up their `call 2`.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);

    let want = oracle(&m);
    let (got, tierups) = coop_tierup_run(&m, eligible);

    // f(x) = 3x + 7: root f(3)=16, worker f(5)=22 → 16 + 22 = 38.
    assert_eq!(want, Ok(vec![Value::I64(38)]), "oracle value");
    assert_eq!(
        want, got,
        "cooperative tier-up run diverged from the pure-interp oracle"
    );
    // Non-vacuity: exactly two tier-ups — one on the root task, one on the worker. (If eligibility
    // failed to propagate to the spawned child this would be 1, not 2.)
    assert_eq!(
        tierups, 2,
        "expected 2 tier-ups (root + worker), got {tierups}"
    );
}

// A tier-up region that **traps** must surface exactly where the interpreter would. Here the leaf
// (func 2) divides by `(x - 5)`, trapping in the worker (which calls it with 5). The whole cooperative
// run must trap iff the pure-interp run does — proving `deliver_tierup_trap` propagates a spawned
// task's tier-up trap through the scheduler's domain teardown to the run's result, like the interpreter.
const SRC_TRAP: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vj = thread.join vt
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 5
  vr = call 2 (v5)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v5 = i64.const 5
  vd = i64.sub vx v5
  v100 = i64.const 100
  vq = i64.div_s v100 vd
  return vq
  }
}
"#;

#[test]
fn coop_tierup_trap_parity() {
    let m = parse_module(SRC_TRAP).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);

    let want = oracle(&m);
    let (got, _) = coop_tierup_run(&m, eligible);

    // The worker divides 100/(5-5) → div-by-zero: the run must trap, exactly as the interpreter does.
    assert!(
        want.is_err(),
        "the oracle run must trap (worker divides by zero)"
    );
    assert_eq!(
        want.is_err(),
        got.is_err(),
        "cooperative tier-up trap parity broke (oracle err={:?})",
        want.is_err()
    );
    assert_eq!(
        want, got,
        "cooperative tier-up value/trap diverged from the oracle"
    );
}

// A tier-up leaf whose emitted region **bounces** back to an interp-resident callee: func 1 (`L`, the
// eligible leaf) is `L(x) = C(x)`, and func 2 (`C`) is `C(x) = 3x + 7`. The host services the `TierUp`
// by calling `CoopRun::bounce` — emulating the emitted `f1` reaching `call_interp` for func 2 — then
// delivers `C`'s result as `L`'s. Proves the cross-tier bounce dispatches through the tiering-up task's
// env (masked table → `(module 0, func 2)`), marshals args/results, and threads the run-shared fiber
// registry — the `CoopSched` analogue of `Vcpu::bounce_call`. The dispatch-table slot for a module-0
// function `c` is just `c` (`SharedSlots::new` packs slot i ↦ `(0, i)`), so the bounce target is `2`.
const SRC_BOUNCE: &str = r#"
func () -> (i64) {
block 0 () {
  v = i64.const 4
  vr = call 1 (v)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vr = call 2 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v3 = i64.const 3
  vm = i64.mul vx v3
  v7 = i64.const 7
  va = i64.add vm v7
  return va
  }
}
"#;

#[test]
fn coop_tierup_bounce_matches_pure_interp() {
    let m = parse_module(SRC_BOUNCE).unwrap();
    svm_verify::verify_module(&m).expect("verify");
    // func 1 (L) is the eligible tier-up leaf; func 2 (C) is the interp callee it bounces to.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, true, false]);

    let want = {
        let mut fuel = FUEL;
        bytecode::compile_and_run(&m, 0, &[], &mut fuel).expect("supported")
    };

    let tierup = TierUpConfig {
        eligible,
        page_checked: false,
    };
    let mut run = bytecode::CoopRun::new(&m, 0, &[], FUEL, Host::new(), Some(tierup))
        .expect("supported")
        .expect("entry in range");
    let mut bounces = 0u32;
    let got = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                assert_eq!(func, 1, "only func 1 (L) is eligible / tiers up");
                // Emulate f1's emitted body: `call_interp(func 2, argv)` — the cross-tier bounce.
                let mut io: Vec<i64> = argv.to_vec();
                let n = run.bounce(2, &mut io).expect("bounce resolves + runs");
                bounces += 1;
                assert_eq!(n, 1, "C returns exactly one result");
                run.deliver_tierup(&io[..n]);
            }
        }
    };

    // L(x) = C(x) = 3x + 7; x = 4 → 19.
    assert_eq!(want, Ok(vec![Value::I64(19)]), "oracle value");
    assert_eq!(
        got, want,
        "cooperative tier-up + bounce diverged from the pure-interp oracle"
    );
    assert_eq!(bounces, 1, "expected exactly one cross-tier bounce");
}
