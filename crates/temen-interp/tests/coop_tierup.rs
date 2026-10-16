//! #926 slice 2 — native proof of wasm-JIT **tier-up on the cooperative multiplex driver**.
//!
//! [`CoopRun`](temen_interp::bytecode::CoopRun) is the single-thread, no-Worker analogue of the parallel
//! `temen_par_*` driver: it multiplexes every vCPU of a run (the root and its `thread.spawn` descendants)
//! on one host thread, and — when armed with a JIT-eligibility bitmap — pauses to the host on each
//! direct module-0 `Call` to an eligible function, exactly as the single-vCPU [`Vcpu`] does. This test
//! stands in for the browser host **without any wasm**: it services each `CoopEvent::TierUp` by running
//! the callee on a standalone bytecode run (what the emitted `f{func}` region computes) and asserts the
//! whole-run result is **identical** to a pure-interpreter cooperative run of the same guest — so the
//! tier-up marshalling (args in, results out, resume) is exact across a genuinely threaded schedule.
//!
//! Tier-up fires on **two tasks** — the root and the spawned worker — proving the paused-task delivery
//! routes to the right vCPU across the scheduler's multiplexing, and that a spawned same-module child
//! inherits the eligibility bitmap.
//! Plus the #926 "Differentials" concurrency shapes: tier-up co-runs with a **self-hosted fiber
//! scheduler** (`cont.new`/`cont.resume` worker fibers) and with a **single-threaded live futex** wake
//! cell (`atomic.wait`/`atomic.notify` park-and-wake) — the scheduler services the concurrency while the
//! root's leaf still tiers up, matching the pure-interp oracle. (A dead-linked-concurrency guest tiering
//! up its leaves is slice 1, #930; the browser-host wasm servicing is `browser/tests/coop_tierup_driver.rs`.)
//!
//! Fuel is **bounded** (not `u64::MAX`) so any regression surfaces as an `OutOfFuel` trap rather than a
//! hang; the guests finish in well under the budget, so a bound never perturbs the differential.

#[path = "support/coop.rs"]
mod coop;
#[path = "support/rec.rs"]
mod rec;

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{ModuleEmitter, ModuleOffer, ProgramEmit, TierUpConfig};
use temen_interp::{
    bytecode, CapState, ForkedProc, Host, HostProc, HostProcFork, ParkEvent, SignalSource, Trap,
    Value,
};
use temen_text::parse_module;

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

fn oracle(m: &temen_ir::Module) -> Result<Vec<Value>, Trap> {
    // The differential oracle: a pure-interpreter cooperative run (`drive`, no eligibility bitmap).
    let mut fuel = FUEL;
    bytecode::compile_and_run(m, 0, &[], &mut fuel).expect("supported")
}

/// Drive the cooperative run under `host` with tier-up enabled, emulating the browser host: each
/// `TierUp(func, argv)` is serviced by running `func` standalone (what the emitted region computes)
/// and delivering its i64 results back. Returns the whole-run result and the functions that tiered
/// up, in order.
fn coop_tierup_run(
    m: &temen_ir::Module,
    eligible: std::sync::Arc<[bool]>,
    host: Host,
) -> (Result<Vec<Value>, Trap>, Vec<u32>) {
    let tierup = TierUpConfig {
        root: ProgramEmit {
            eligible,
            page_checked: false,
        },
        leaf: None,
        modules: None,
    };
    let mut run = coop::coop_run(m, 0, &[], FUEL, host, Some(tierup))
        .expect("supported")
        .expect("entry in range");
    let mut tierups = Vec::new();
    loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => return (Ok(vals), tierups),
            bytecode::CoopEvent::Trapped(t) => return (Err(t), tierups),
            bytecode::CoopEvent::Idle
            | bytecode::CoopEvent::Paused
            | bytecode::CoopEvent::CapPark { .. } => {
                panic!("unexpected Idle (suspend_on_idle is never armed here)")
            }
            bytecode::CoopEvent::Resume { .. } => {
                panic!("unexpected Resume (no leaf image here)")
            }
            bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected JitInvoke (no vm_jit guest here)")
            }
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                tierups.push(func);
                // A regression that fails to advance the paused task would tier up unboundedly; cap it
                // so the test fails fast instead of hanging (the real runs need at most one per call).
                assert!(
                    tierups.len() < 50,
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
    temen_verify::verify_module(&m).expect("verify");
    assert_eq!(oracle(&m), Ok(vec![Value::I64(38)]));
}

#[test]
fn coop_tierup_matches_pure_interp() {
    let m = parse_module(SRC).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    // Only func 2 (the leaf) is eligible; func 0 (root) and func 1 (worker) are the interp-driven
    // callers. Both the root and the spawned worker must therefore tier up their `call 2`.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);

    let want = oracle(&m);
    let (got, tierups) = coop_tierup_run(&m, eligible, Host::new());

    // f(x) = 3x + 7: root f(3)=16, worker f(5)=22 → 16 + 22 = 38.
    assert_eq!(want, Ok(vec![Value::I64(38)]), "oracle value");
    assert_eq!(
        want, got,
        "cooperative tier-up run diverged from the pure-interp oracle"
    );
    // Non-vacuity: exactly two tier-ups — one on the root task, one on the worker. (If eligibility
    // failed to propagate to the spawned child this would be 1, not 2.)
    assert_eq!(
        tierups,
        [2, 2],
        "expected 2 tier-ups (root + worker), got {tierups:?}"
    );
}

/// #1414 ED-1 — a task the run starts tiers up at its entry when its program's emit makes the
/// entry eligible, as a call to it would, and its results end it: here the worker's entry, func 1.
/// The root's entry never does, since the root leaf is its rule. Nor does a durable run's thread:
/// a thaw replays its entry interpreted.
#[test]
fn a_thread_tiers_up_at_its_entry() {
    let m = parse_module(SRC).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    let want = oracle(&m);
    // `(eligible, durable, the functions that tier up)`.
    for (eligible, durable, want_funcs) in [
        ([false, true, true], false, &[1, 2][..]),
        ([true, false, true], false, &[2, 2]),
        ([false, true, true], true, &[2, 2]),
    ] {
        let mut host = Host::new();
        host.set_durable(durable);
        let (got, mut tierups) = coop_tierup_run(&m, std::sync::Arc::from(eligible), host);
        tierups.sort();
        assert_eq!(got, want, "eligible {eligible:?}, durable {durable}");
        assert_eq!(
            tierups, want_funcs,
            "eligible {eligible:?}, durable {durable}"
        );
    }
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
    temen_verify::verify_module(&m).expect("verify");
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);

    let want = oracle(&m);
    let (got, _) = coop_tierup_run(&m, eligible, Host::new());

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

// A task tiers up **once per call**, however many calls it makes: func 0 sums `f1(i)` over
// `0..N_ITERS`, and each `call 1` pauses to the host. No memory and an all-i64 leaf: the smallest shape
// that tiers up.
const SRC_LOOP: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = i64.const N_ITERS
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v0, v1, v2)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.lt_s v5 v3
  br_if v6 2(v3, v4, v5) 3(v4)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = call 1 (v9)
  v11 = i64.add v8 v10
  v12 = i64.const 1
  v13 = i64.add v9 v12
  br 1(v7, v11, v13)
}
block 3 (v14: i64) {
  return v14
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  v3 = i64.const 7
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn a_task_tiers_up_once_per_call() {
    for n in [0i64, 1, 2, 5, 10, 40] {
        let m = parse_module(&SRC_LOOP.replace("N_ITERS", &n.to_string())).unwrap();
        temen_verify::verify_module(&m).expect("verify");
        let want = oracle(&m);
        let eligible = std::sync::Arc::from(vec![false, true]);
        let (got, tierups) = coop_tierup_run(&m, eligible, Host::new());
        assert_eq!(
            want, got,
            "the tier-up run diverged from the oracle at n={n}"
        );
        assert_eq!(tierups.len() as i64, n, "one tier-up per call at n={n}");
        if n == 5 {
            // sum over i < 5 of (3i + 7) = 30 + 35.
            assert_eq!(want, Ok(vec![Value::I64(65)]));
        }
    }
}

// A tier-up leaf whose emitted region **bounces** back to an interp-resident callee: func 1 (`L`, the
// eligible leaf) is `L(x) = C(x)`, and func 2 (`C`) is `C(x) = 3x + 7`. The host services the `TierUp`
// by calling `CoopRun::bounce` — emulating the emitted `f1` reaching `call_interp` for func 2 — then
// delivers `C`'s result as `L`'s. Proves the cross-tier bounce dispatches through the tiering-up task's
// env (masked table → `(module 0, func 2)`), marshals args/results, and threads the run-shared fiber
// registry. The dispatch-table slot for a module-0
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
    temen_verify::verify_module(&m).expect("verify");
    // func 1 (L) is the eligible tier-up leaf; func 2 (C) is the interp callee it bounces to.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, true, false]);

    let want = {
        let mut fuel = FUEL;
        bytecode::compile_and_run(&m, 0, &[], &mut fuel).expect("supported")
    };

    let tierup = TierUpConfig {
        root: ProgramEmit {
            eligible,
            page_checked: false,
        },
        leaf: None,
        modules: None,
    };
    let mut run = coop::coop_run(&m, 0, &[], FUEL, Host::new(), Some(tierup))
        .expect("supported")
        .expect("entry in range");
    let mut bounces = 0u32;
    let got = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::Idle
            | bytecode::CoopEvent::Paused
            | bytecode::CoopEvent::CapPark { .. } => {
                panic!("unexpected Idle (suspend_on_idle is never armed here)")
            }
            bytecode::CoopEvent::Resume { .. } => {
                panic!("unexpected Resume (no leaf image here)")
            }
            bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected JitInvoke (no vm_jit guest here)")
            }
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                assert_eq!(func, 1, "only func 1 (L) is eligible / tiers up");
                // Emulate f1's emitted body: `call_interp(func 2, argv)` — the cross-tier bounce.
                let mut io: Vec<i64> = argv.to_vec();
                let n = run
                    .bounce(2, &mut io, None)
                    .expect("bounce resolves + runs")
                    .expect("the call returns: nothing here parks");
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

// #1414 slice 3d — a bounce resumes a fiber the pump parked. The root starts a fiber that parks in
// `memory.wait` (its resumer gets `(FIBER_PARKED, 0)`), then calls the eligible leaf `L(k)`, whose
// emitted body bounces to `C(k)`. `C` resumes the parked fiber over the run's registry: still parked,
// it answers the poll again, as the same resume does interpreted. Then the root notifies the word and
// resumes the fiber to its return. Returns `C's status * 10 + the last resume's status`.
const SRC_BOUNCE_POLLS_A_PARKED_FIBER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 3
  v1 = i64.const 0
  vk = cont.new v0 v1
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  vpoll = call 1 (vk)
  vaddr = i64.const 16384
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vs3, vv3 = cont.resume vk vz
  vs364 = i64.extend_i32_s vs3
  v10 = i64.const 10
  vhi = i64.mul vpoll v10
  vr = i64.add vhi vs364
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vk: i64) {
  vr = call 2 (vk)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vk: i64) {
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vs64 = i64.extend_i32_s vs
  return vs64
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

/// The bounce used to fault on the parked fiber (`FiberFault`): the nested drive claimed only
/// pending and suspended fibers, where the pump claims every state.
#[test]
fn a_bounce_polls_a_fiber_the_pump_parked() {
    let m = parse_module(SRC_BOUNCE_POLLS_A_PARKED_FIBER).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    // func 1 (L) is the eligible leaf; func 2 (C) is the interpreted callee its emitted body bounces to.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, true, false, false]);

    // FIBER_PARKED (3) from the bounce's resume, then FIBER_RETURNED (1).
    let want = oracle(&m);
    assert_eq!(want, Ok(vec![Value::I64(31)]), "oracle value");

    let tierup = TierUpConfig {
        root: ProgramEmit {
            eligible,
            page_checked: false,
        },
        leaf: None,
        modules: None,
    };
    let mut run = coop::coop_run(&m, 0, &[], FUEL, Host::new(), Some(tierup))
        .expect("supported")
        .expect("entry in range");
    let mut bounces = 0u32;
    let got = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                assert_eq!(func, 1, "only func 1 (L) is eligible / tiers up");
                // Emulate L's emitted body: `call_interp(func 2, argv)`, the cross-tier bounce.
                let mut io: Vec<i64> = argv.to_vec();
                match run.bounce(2, &mut io, None) {
                    Ok(n) => {
                        let n = n.expect("the call returns: nothing here parks");
                        run.deliver_tierup(&io[..n]);
                    }
                    Err(t) => run.deliver_tierup_trap(t),
                }
                bounces += 1;
            }
            bytecode::CoopEvent::Idle
            | bytecode::CoopEvent::Paused
            | bytecode::CoopEvent::CapPark { .. }
            | bytecode::CoopEvent::Resume { .. }
            | bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected event: only the leaf's tier-up pauses this run")
            }
        }
    };
    assert_eq!(
        got, want,
        "the bounce's resume diverged from the interpreted one"
    );
    assert_eq!(bounces, 1, "expected exactly one cross-tier bounce");
}

// #926 differential (the issue's "Differentials" list): a **self-hosted fiber scheduler** whose worker
// fibers each call the eligible leaf, plus the root — proving tier-up fires and marshals correctly amid
// the scheduler's fiber servicing (`cont.new`/`cont.resume`), not only across `thread.spawn` vCPUs. The
// root spins two worker fibers (func 1 `call 3 (3)`, func 2 `call 3 (5)`), runs each to completion, then
// calls the leaf itself (`call 3 (9)`); every `call 3` to the eligible leaf `f(x) = 3x + 7` must run
// observably identical to the pure-interp cooperative oracle. Single-vCPU — the fibers ARE the workers.
const SRC_FIBER_SCHED: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vf1 = ref.func 1
  vk1 = cont.new vf1 vz
  vsa, vva = cont.resume vk1 vz
  vf2 = ref.func 2
  vk2 = cont.new vf2 vz
  vsb, vvb = cont.resume vk2 vz
  v9 = i64.const 9
  vroot = call 3 (v9)
  vsum1 = i64.add vva vvb
  vr = i64.add vsum1 vroot
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v3 = i64.const 3
  vr = call 3 (v3)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 5
  vr = call 3 (v5)
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
fn coop_tierup_fiber_scheduler_matches_pure_interp() {
    let m = parse_module(SRC_FIBER_SCHED).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    // Only func 3 (the leaf) is eligible; the root and both worker fibers call it.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, false, true]);

    let want = oracle(&m);
    let (got, tierups) = coop_tierup_run(&m, eligible, Host::new());

    // fiber A f(3)=16, fiber B f(5)=22, root f(9)=34 → 72.
    assert_eq!(want, Ok(vec![Value::I64(72)]), "oracle value");
    assert_eq!(
        got, want,
        "cooperative tier-up under a fiber scheduler diverged from the pure-interp oracle"
    );
    // Non-vacuity: exactly **one** tier-up — the root's direct `call 3`. A leaf called from *inside* a
    // fiber runs interpreted (a resumed continuation executes on the fiber machinery, not the
    // eligibility-armed root frame), so fibers A and B compute their `f(3)`/`f(5)` on the interpreter —
    // and the parity check above proves that's identical to the oracle. The point this pins: tier-up
    // still fires and marshals correctly on the root while the cooperative scheduler juggles live fibers
    // (a regression that dropped tier-up under fiber activity would read 0).
    assert_eq!(
        tierups,
        [3],
        "expected 1 tier-up (the root's leaf; fibers run their leaf interpreted), got {tierups:?}"
    );
}

// #926 differential (the issue's "Differentials" list): a **single-threaded, cooperative live futex**
// wake cell whose compute leaf tiers up. The root creates a fiber that untimed-`atomic.wait`s on cell
// 16384 (above the #1094 NULL guard) (parking the FIBER, not the vCPU — the root keeps running),
// `atomic.notify`s the cell to wake it, runs
// it to completion, then calls the eligible compute leaf (`f(x) = 3x + 7`). The whole run — the live
// futex park/wake handshake AND the tier-up — must match the pure-interp cooperative oracle. No
// `thread.spawn`: the wake is delivered within the one vCPU, the `uses_futex` shape #845 gates on.
const SRC_LIVE_FUTEX: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  vk = cont.new v0 v1
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  vs2, vv2 = cont.resume vk vz
  vaddr = i64.const 16384
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vs3, vv3 = cont.resume vk vz
  v4 = i64.const 4
  vleaf = call 2 (v4)
  vr = i64.add vv3 vleaf
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
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
fn coop_tierup_live_futex_matches_pure_interp() {
    let m = parse_module(SRC_LIVE_FUTEX).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    // Only func 2 (the compute leaf) is eligible; the root calls it after the futex wake handshake.
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);

    let want = oracle(&m);
    let (got, tierups) = coop_tierup_run(&m, eligible, Host::new());

    // The fiber wakes WAIT_WOKEN(0), then the leaf f(4) = 19 → 0 + 19 = 19.
    assert_eq!(want, Ok(vec![Value::I64(19)]), "oracle value");
    assert_eq!(
        got, want,
        "cooperative tier-up over a live futex wake cell diverged from the pure-interp oracle"
    );
    // Non-vacuity: the compute leaf tiers up after the live futex park/wake completes single-threaded.
    assert_eq!(
        tierups,
        [2],
        "expected exactly one tier-up (the root's compute leaf), got {tierups:?}"
    );
}

// #816 env-routed tier-up — a task whose window lives in `extra_envs` (a §14 child, or a thread it
// spawns) TIERS UP over its own window: the engine routes the driver's per-event reads
// (`pending_win`, `window_scalar_extent`, `mem_map_info`/`mem_map_version`) to the pending
// round-trip's task env, and the eligibility gates (`tierup_servable`) admit a window the driver can
// address flatly. This test pins the inheritance: the child entry's and the child-env worker's calls
// to the eligible leaf both surface, and the run still matches the pure-interp oracle.
//
// func 0 (root; args = its Instantiator, a `Budget` and the child's `Module`): spawns a child
// detached, by the spawn record appended at 17408 (a window of its own, the module's declared memory),
// joins it, calls the eligible leaf f3 itself, and sums. The child runs func 1's child image
// (#2248). func 1 (child entry): calls the leaf directly, thread.spawns f2 INSIDE the child env (the
// thread-arm inheritance) and joins it. func 2 (child-env worker): calls the leaf. func 3: the pure
// all-i64 leaf f(x) = x*3 + 7.
const SRC_CHILD_ENV: &str = r#"
memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrb = i64.const 17420
  i32.store vrb v1
  vrm = i64.const 17416
  i32.store vrm v2
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
  v3 = i64.const 3
  vlocal = call 3 (v3)
  vr = i64.add vj vlocal
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  v7 = i64.const 7
  vlocal = call 3 (v7)
  vw = thread.join vt
  vr = i64.add vw vlocal
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 5
  vr = call 3 (v5)
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

/// The granted modules a run offered its host's emitter ([`modules_emitter`]): `(program, image)`.
type Offered = std::sync::Arc<std::sync::Mutex<Vec<(u32, temen_ir::Module)>>>;

/// A host's emitter for granted modules, as these tests fake one: it answers every module it is
/// offered with `emit` (`None` declines), and records the offer. The digest it is handed must be
/// the image's.
fn modules_emitter(emit: Option<ProgramEmit>) -> (ModuleEmitter, Offered) {
    let offered = Offered::default();
    let record = std::sync::Arc::clone(&offered);
    let emitter: ModuleEmitter = std::sync::Arc::new(move |o: &ModuleOffer| {
        assert_eq!(
            o.digest,
            temen_interp::module_digest(o.image),
            "the offer's digest"
        );
        record.lock().unwrap().push((o.module, o.image.clone()));
        emit.clone()
    });
    (emitter, offered)
}

/// What a [`TierUp`](bytecode::CoopEvent::TierUp) event's emitted region computes: function `func`
/// of the program it names, run standalone — `m`, the run's own (`0`), or an image the run offered.
fn emulate_region(
    m: &temen_ir::Module,
    offered: &Offered,
    module: u32,
    func: u32,
    argv: &[i64],
) -> Result<Vec<i64>, Trap> {
    let image = (module != 0).then(|| {
        let offered = offered.lock().unwrap();
        let (_, image) = offered
            .iter()
            .find(|(k, _)| *k == module)
            .expect("a tier-up names the run's program or one it offered");
        image.clone()
    });
    let args: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
    let mut fuel = FUEL;
    let vals = bytecode::compile_and_run(image.as_ref().unwrap_or(m), func, &args, &mut fuel)
        .expect("supported")?;
    Ok(vals
        .iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            Value::I32(x) => *x as i64,
            _ => panic!("non-integer tier-up result"),
        })
        .collect())
}

/// Run `m` cooperatively with `tierup` or without, its root handed an `Instantiator`, a `Budget`
/// and `module`, granting `image` for it when there is one. Each tier-up runs its region
/// ([`emulate_region`]). Returns the result and the program each tier-up named, in order.
fn run_spawner(
    m: &temen_ir::Module,
    image: &temen_ir::Module,
    tierup: Option<TierUpConfig>,
    offered: &Offered,
) -> (Result<Vec<Value>, Trap>, Vec<u32>) {
    let mut host = Host::new();
    host.set_self_module(&std::sync::Arc::new(m.clone()));
    let inst = host.grant_instantiator();
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let module = host.grant_module(image);
    let args = [inst, budget, module].map(Value::I32);
    let mut run = coop::coop_run(m, 0, &args, FUEL, host, tierup)
        .expect("supported")
        .expect("entry in range");
    let mut named = Vec::new();
    loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => return (Ok(vals), named),
            bytecode::CoopEvent::Trapped(t) => return (Err(t), named),
            bytecode::CoopEvent::Idle
            | bytecode::CoopEvent::Paused
            | bytecode::CoopEvent::CapPark { .. } => {
                panic!("unexpected Idle (suspend_on_idle is never armed here)")
            }
            bytecode::CoopEvent::Resume { .. } => {
                panic!("unexpected Resume (no leaf image here)")
            }
            bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected JitInvoke (no vm_jit guest here)")
            }
            bytecode::CoopEvent::TierUp {
                module, func, argv, ..
            } => {
                named.push(module);
                assert!(named.len() < 50, "runaway tier-ups (last func={func})");
                match emulate_region(m, offered, module, func, &argv) {
                    Ok(slots) => run.deliver_tierup(&slots),
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
        }
    }
}

/// The child's tasks tier up with the root: in a granted image of the root's program, through the
/// image's own emit (#2248) — or, where the host declines the image, they interpret. Differential
/// against the same run with no tier-up.
#[test]
fn coop_tierup_child_env_tasks_tier_up() {
    let leaf = || ProgramEmit {
        eligible: std::sync::Arc::from(vec![false, false, false, true]),
        page_checked: false,
    };
    // `(child, the image's emit, the programs the tier-ups name)`. The child's two calls surface
    // first: the root is parked on the join.
    for (child, emit, want_named) in [
        ("image", Some(leaf()), vec![1, 1, 0]),
        ("declined image", None, vec![0]),
    ] {
        let src = format!(
            "{SRC_CHILD_ENV}{}",
            rec::segment(17408, &temen_ir::SpawnRec::default())
        );
        let m = parse_module(&src).unwrap();
        temen_verify::verify_module(&m).expect("verify");
        let image = temen_ir::child_image_at(&m, 1).expect("child image");

        // Oracle: pure-interp cooperative run. Worker f(5)=22 + child entry f(7)=28 → child 50;
        // root f(3)=16 → 66.
        let (want, oracle_named) = run_spawner(&m, &image, None, &Offered::default());
        assert_eq!(want, Ok(vec![Value::I64(66)]), "{child}: oracle value");
        assert!(
            oracle_named.is_empty(),
            "{child}: the oracle never tiers up"
        );

        let (modules, offered) = modules_emitter(emit);
        let tierup = TierUpConfig {
            root: leaf(),
            leaf: None,
            modules: Some(modules),
        };
        let (got, named) = run_spawner(&m, &image, Some(tierup), &offered);
        assert_eq!(
            got, want,
            "{child}: cooperative tier-up run with a detached child diverged from the oracle"
        );
        // #816 env routing: all three eligible calls surface — the root's, the detached child
        // entry's and the child-env worker's — on every target, since a child that tiers up gets
        // a flat window (#2102).
        assert_eq!(named, want_named, "{child}: the programs the tier-ups name");
        let offers: Vec<u32> = offered.lock().unwrap().iter().map(|(k, _)| *k).collect();
        assert_eq!(offers, [1], "{child}: the modules offered to the host");
    }
}

/// #2248 — a granted module is offered to the host once per run, and every child that runs it
/// tiers up through the one emit. The root spawns two children of the image and joins each. The
/// image's emit takes its leaf, or its entry (#1414 ED-1): the image's bootstrap function 0, which
/// nothing calls, so each child tiers up at its entry and the entry's result ends it.
#[test]
fn coop_tierup_offers_a_granted_module_once() {
    const SRC: &str = r#"
memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrb = i64.const 17420
  i32.store vrb v1
  vrm = i64.const 17416
  i32.store vrm v2
  vrp = i64.const 17408
  vh1 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vj1_st, vj1 = call.cap 6 1 (i32) -> (i64, i64) v0 (vh1)
  vh2 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vj2_st, vj2 = call.cap 6 1 (i32) -> (i64, i64) v0 (vh2)
  vr = i64.add vj1 vj2
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  vr = call 2 (v7)
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
    let src = format!(
        "{SRC}{}",
        rec::segment(17408, &temen_ir::SpawnRec::default())
    );
    let m = parse_module(&src).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    let image = temen_ir::child_image_at(&m, 1).expect("child image");
    let (want, _) = run_spawner(&m, &image, None, &Offered::default());
    assert_eq!(
        want,
        Ok(vec![Value::I64(56)]),
        "oracle: two children, f(7) = 28 each"
    );

    for eligible in [[false, false, true], [true, false, false]] {
        let (modules, offered) = modules_emitter(Some(ProgramEmit {
            eligible: std::sync::Arc::from(eligible),
            page_checked: false,
        }));
        let tierup = TierUpConfig {
            root: ProgramEmit::default(),
            leaf: None,
            modules: Some(modules),
        };
        let (got, named) = run_spawner(&m, &image, Some(tierup), &offered);
        assert_eq!(
            got, want,
            "the tier-up run diverged from the oracle ({eligible:?})"
        );
        assert_eq!(
            named,
            vec![1, 1],
            "each child tiers up once, in the image ({eligible:?})"
        );
        let offers: Vec<u32> = offered.lock().unwrap().iter().map(|(k, _)| *k).collect();
        assert_eq!(offers, vec![1], "the image is offered once ({eligible:?})");
    }
}

// #816 item 3 — a **fork twin** tiers up over its own private flat window. The guest topology is
// the proven browser fork probe (`FORK_TWIN`, FORK.md §9): a manager (f0) spawns a server (f1,
// handlers f2 = `clone_caller`, f3 = `reap`) and a guest (f4) as detached §14 children, through
// the spawn records `fork_twin_src` lays at 17792 and 17664; the guest calls
// `svc.fork`, the servicer duplicates it (`Mem::fork_private` + `Host::fork_powerbox`), and BOTH
// copies — the original (reply 100) and the twin (reply 200) — call the eligible leaf f5 on their
// reply before writing the result to the shared stdout; the original then `svc.wait`s the twin.
// The manager takes a `Budget`, which each record charges, and the `Module`s its guest and server
// run: func 4's and func 1's child images. `fork_twin_src` fills in the guest's scratch (`@S0@`,
// `@S8@`, `@S16@`): its names and its stdout word, which it keeps in its own window.
const SRC_FORK_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32, vbud: i32, vmod: i32, vsrv: i32) {
  q1a0 = i64.const 17792
  q1b = i64.const 17804
  i32.store q1b vbud
  q1m = i64.const 17800
  i32.store q1m vsrv
  vs = call.cap 6 17 (i64) -> (i32) v0 (q1a0)
  vz0 = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  q2a0 = i64.const 17664
  q2b = i64.const 17676
  i32.store q2b vbud
  q2m = i64.const 17672
  i32.store q2m vmod
  vc = call.cap 6 17 (i64) -> (i32) v0 (q2a0)
  vjc_st, vjc = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vro = i64.const 100
  vrt = i64.const 200
  vt = call.cap 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = call.cap 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vs0 = i64.const @S0@
  i64.store vs0 vsvc
  voname = i64.const 111
  vs8 = i64.const @S8@
  i64.store vs8 voname
  vl3 = i64.const 3
  vhsvc = self.resolve vs0 vl3
  vl1 = i64.const 1
  vho = self.resolve vs8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  varg = i64.const 7
  vr = call.cap 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const 200
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho) 2(vr, vhsvc, vho)
  }
block 2 (vr: i64, vhsvc: i32, vho: i32) {
  vpid3 = i64.const 3
  vstatus = call.cap 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho) 3(vr, vstatus, vhsvc, vho)
  }
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32) {
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho) 4(vr, vho)
  }
block 4 (vr: i64, vho: i32) {
  vleaf = call 5 (vr)
  vs16 = i64.const @S16@
  i64.store vs16 vleaf
  vlen = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vho (vs16, vlen)
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

/// [`SRC_FORK_TWIN`]: its fork server runs func 1's child image and its guest func 4's, the manager's
/// `vsrv` and `vmod`, each detached in a window of its own (#2102, #2219). Its scratch is above the
/// NULL guard its window has (#964).
fn fork_twin_src() -> String {
    let guest = rec::segment(
        17664,
        &temen_ir::SpawnRec {
            grants_ptr: 16640,
            grants_n: 2,
            ..temen_ir::SpawnRec::default()
        },
    );
    SRC_FORK_TWIN
        .replace("@S0@", "16896")
        .replace("@S8@", "16904")
        .replace("@S16@", "16912")
        + &rec::segment(17792, &temen_ir::SpawnRec::default())
        + &guest
}

/// Every target (unlike the unix-only pending-win test): the run backing is an **owned flat**
/// buffer (`Region::owned_zeroed` — the same shape as the browser's `Region::shared` window), so
/// the raw window views resolve even where `Region::new` would fall back to `Paged` (Windows),
/// and the twin-backing seam's non-unix arm gets real end-to-end CI coverage there. A detached
/// guest's window and its twin's are flat on every target too (#2102). Differential against the
/// same run with no bitmap.
#[test]
fn coop_tierup_fork_twin_tiers_up_over_its_private_flat_window() {
    // Both leaf calls tier up: the original guest's and the fork twin's, whose private
    // `fork_private` window is its OWN flat allocation, outside the root backing. The guest's
    // 256 KiB window is its own, and so is its twin's. The guest's calls are its image's (program
    // 2: the server's image, spawned first, is program 1), and so are its twin's (#2248).
    let m = std::sync::Arc::new(parse_module(&fork_twin_src()).unwrap());
    temen_verify::verify_module(&m).expect("verify");
    // Oracle: pure-interp cooperative run. The original resumes with 100 and the twin with 200;
    // both write leaf(reply) to the shared stdout: leaf(100) = 307, leaf(200) = 607.
    let (want, want_out, oracle_events) = fork_twin_run(&m, None);
    assert_eq!(want, Ok(vec![Value::I64(100)]), "oracle value");
    assert_eq!(
        want_out,
        vec![307, 607],
        "both copies ran and wrote (oracle)"
    );
    assert!(oracle_events.is_empty(), "the oracle never tiers up");

    // The image keeps the program's functions where they are, so one emit fits both.
    let leaf = ProgramEmit {
        eligible: std::sync::Arc::from(vec![false, false, false, false, false, true]),
        page_checked: false,
    };
    let (modules, _) = modules_emitter(Some(leaf.clone()));
    let (got, got_out, mut events) = fork_twin_run(
        &m,
        Some(TierUpConfig {
            root: leaf,
            leaf: None,
            modules: Some(modules),
        }),
    );
    assert_eq!(got, want, "the tier-up run diverged from the oracle");
    assert_eq!(got_out, want_out, "stdout parity");
    events.sort_unstable();
    assert_eq!(
        events,
        [(2, false, 1 << 18), (2, false, 1 << 18)],
        "the original's event and the twin's, (program, in the root backing?, span)"
    );
}

/// Run `m`, a [`fork_twin_src`] program, on the cooperative driver with `tierup` or without, over an
/// owned flat 256 KiB root backing: its result, its sorted stdout words, and for each tier-up event,
/// the program it names, whether the pending window lies in the root backing, and its span.
#[allow(clippy::type_complexity)]
fn fork_twin_run(
    m: &std::sync::Arc<temen_ir::Module>,
    tierup: Option<TierUpConfig>,
) -> (Result<Vec<Value>, Trap>, Vec<i64>, Vec<(u32, bool, u64)>) {
    const FORK_FUEL: u64 = 40_000_000;
    let mut host = Host::new();
    host.set_self_module(m);
    let inst = host.grant_instantiator();
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(temen_interp::StreamRole::Out);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let module = host.grant_module(&temen_ir::child_image_at(m, 4).expect("child image"));
    let server = host.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    let args = [inst, out_h, budget, module, server].map(Value::I32);
    let back = std::sync::Arc::new(
        temen_interp::Region::owned_zeroed(1 << 18, 4096).expect("256 KiB allocates"),
    );
    let root_base = back.raw_base().expect("flat backing") as usize;
    let mut run = bytecode::CoopRun::new_over(
        m,
        0,
        &args,
        FORK_FUEL,
        host,
        tierup,
        &[],
        18,
        std::sync::Arc::clone(&back),
    )
    .expect("supported")
    .expect("entry in range");
    let mut events: Vec<(u32, bool, u64)> = Vec::new();
    let result = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::Idle
            | bytecode::CoopEvent::Paused
            | bytecode::CoopEvent::CapPark { .. } => {
                panic!("unexpected Idle (suspend_on_idle is never armed here)")
            }
            bytecode::CoopEvent::Resume { .. } => {
                panic!("unexpected Resume (no leaf image here)")
            }
            bytecode::CoopEvent::JitInvoke { .. } => panic!("unexpected JitInvoke"),
            bytecode::CoopEvent::TierUp {
                module, func, argv, ..
            } => {
                assert_eq!(func, 5, "only the leaf is eligible");
                assert!(events.len() < 10, "runaway tier-ups");
                let (ptr, len) = run
                    .pending_win()
                    .expect("every tiering task's window must resolve a flat view");
                let in_root = (ptr as usize) >= root_base && (ptr as usize) < root_base + (1 << 18);
                events.push((module, in_root, len));
                let x = argv[0];
                run.deliver_tierup(&[x * 3 + 7]);
            }
        }
    };
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).to_vec();
    let mut out: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    out.sort_unstable();
    (result, out, events)
}

// #2273 — a fork twin of the root runs program 0, whose emitted `call.dyn` dispatches through the
// root's dispatch table, over a table of its own. A `Jit` in the powerbox could make the two differ,
// so then the twin and its threads interpret; without one they tier up as the root does. The root
// forks through a personality reduced to its process doors ([`grant_proc_doors`]). The twin calls
// the eligible leaf f1 and joins a thread (f2) that calls it too; the root waits for the twin and
// calls f1 itself. The twin returns f1(5) + f1(9) = 56, and the root 1000 * 56 + f1(3) = 56016.
const SRC_ROOT_FORK: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (vproc: i32) {
  vpid = call.cap 13 0 () -> (i64) vproc ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1() 2(vpid, vproc)
}
block 1 () {
  vz1 = i64.const 0
  vt = thread.spawn 2 vz1 vz1
  v5 = i64.const 5
  vown = call 1 (v5)
  vth = thread.join vt
  vr = i64.add vown vth
  return vr
}
block 2 (vp: i64, vh: i32) {
  vst = call.cap 13 1 (i64) -> (i64) vh (vp)
  v3 = i64.const 3
  vown2 = call 1 (v3)
  vk = i64.const 1000
  vsk = i64.mul vst vk
  vr2 = i64.add vsk vown2
  return vr2
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
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v9 = i64.const 9
  vr = call 1 (v9)
  return vr
  }
}
"#;

#[test]
fn a_root_fork_twin_interprets_when_the_run_holds_a_jit() {
    let m = parse_module(SRC_ROOT_FORK).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    for jit in [false, true] {
        let (want, none) = root_fork_run(&m, jit, false);
        assert_eq!(want, Ok(vec![Value::I64(56_016)]), "oracle, jit {jit}");
        assert!(none.is_empty(), "the oracle never tiers up");
        let (got, envs) = root_fork_run(&m, jit, true);
        assert_eq!(
            got, want,
            "jit {jit}: the tier-up run diverged from the oracle"
        );
        // `-1` is the root's env; `0`, the twin's, which its thread shares.
        let tiered: &[i64] = if jit { &[-1] } else { &[-1, 0, 0] };
        assert_eq!(
            envs, tiered,
            "jit {jit}: the envs whose leaf calls tiered up"
        );
    }
}

/// Run [`SRC_ROOT_FORK`] on the cooperative driver, its powerbox holding a `Jit` or not, with the
/// leaf f1 eligible (`tierup`) or without a bitmap: its result, and the sorted envs of its tier-up
/// events. The session constructor wires the personality's park door, as for `bash -i`, over an
/// owned flat backing (as [`fork_twin_run`]'s).
fn root_fork_run(
    m: &temen_ir::Module,
    jit: bool,
    tierup: bool,
) -> (Result<Vec<Value>, Trap>, Vec<i64>) {
    let mut host = Host::new();
    let doors = grant_proc_doors(&mut host);
    if jit {
        host.grant_jit_with_table(Some(16), 4);
    }
    let tierup = tierup.then(|| TierUpConfig {
        root: ProgramEmit {
            eligible: Arc::from(vec![false, true, false]),
            page_checked: false,
        },
        leaf: None,
        modules: None,
    });
    let back =
        Arc::new(temen_interp::Region::owned_zeroed(1 << 16, 4096).expect("64 KiB allocates"));
    let program = bytecode::compile_reserved(m).expect("in subset");
    let mut run = bytecode::CoopRun::new_reserved_over_compiled(
        m,
        program,
        0,
        &[Value::I32(doors)],
        FUEL,
        host,
        tierup,
        &[],
        16,
        Some(back),
    )
    .expect("entry in range");
    let mut envs = Vec::new();
    let result = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                assert_eq!(func, 1, "only the leaf is eligible");
                assert!(envs.len() < 10, "runaway tier-ups");
                envs.push(run.pending_env());
                run.deliver_tierup(&[argv[0] * 3 + 7]);
            }
            _ => panic!("unexpected pump event"),
        }
    };
    envs.sort_unstable();
    (result, envs)
}

/// A personality reduced to its process doors: op 0 `fork`s and op 1 `waitpid`s, as temen-posix's
/// do. Each asks the core through the park door the driver installs on the domain's signal source;
/// `waitpid` re-runs when the twin ends, and answers the status its exit hook recorded.
fn grant_proc_doors(host: &mut Host) -> i32 {
    const ENOSYS: i64 = -38;
    let door = Arc::new(ParkDoor::default());
    let exited: Arc<Mutex<Option<i64>>> = Arc::default();
    let (ask, status) = (Arc::clone(&door), Arc::clone(&exited));
    let doors: HostProc = Box::new(move |op, args, _, _| {
        let (park, r) = match (op, *status.lock().unwrap()) {
            (0, _) => (Some(ParkEvent::ForkSelf), ENOSYS),
            (_, Some(s)) => (None, s),
            (_, None) => (Some(ParkEvent::TaskExit(args[0] as u64)), ENOSYS),
        };
        if let (Some(ev), Some(req)) = (park, ask.0.lock().unwrap().clone()) {
            req(ev);
        }
        Ok(vec![r])
    });
    // The twin's own door, as a per-process personality mints one (its park door is its own), and an
    // exit hook that records its status.
    let fork: HostProcFork = Arc::new(move |_pid| {
        let exited = Arc::clone(&exited);
        let source: Arc<dyn SignalSource + Send + Sync> = Arc::new(ParkDoor::default());
        Some(ForkedProc {
            signal: Some((source, Arc::default())),
            exit: Some(Arc::new(move |s| *exited.lock().unwrap() = Some(s))),
            ..ForkedProc::shared(Box::new(|_, _, _, _| Ok(vec![ENOSYS])), CapState::Stateless)
        })
    });
    host.set_signal_source(door, Arc::default());
    host.grant_host_proc_forkable(doors, fork, CapState::Stateless)
}

/// The park door the driver installs on a domain's signal source.
type ParkRequest = Arc<dyn Fn(ParkEvent) + Send + Sync>;

/// A signal source that holds only the park door the driver installs.
#[derive(Default)]
struct ParkDoor(Mutex<Option<ParkRequest>>);

impl SignalSource for ParkDoor {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_park_request(&self, req: ParkRequest) {
        *self.0.lock().unwrap() = Some(req);
    }
}

// #816 item 4 — `SharedProgram::coop_run_over_grown`, the warm session's cooperative constructor:
// a captured page-state map is re-established (`seed_pages`, no zeroing) over a caller-restored
// backing, and the run's eligible leaf tiers up — so a page-managing warm image evaluates on the
// coop drive instead of interpreter-only. The entry reads a marker in a "restored" `vm_map`-grown
// page (bytes planted directly in the backing, as the warm memcpy does) and calls the leaf; the
// negative arm proves the seeding is load-bearing: the same run without the seeded entries faults
// on the grown-page load (fail-closed, the `run_over_grown` contract).
#[test]
fn coop_run_over_grown_restores_the_page_map_and_tiers_up() {
    // func 0 (entry): load the marker at 65552 (inside the grown page [64 KiB, 68 KiB)), call the
    // eligible leaf f1 with 3, return marker + leaf. func 1: the pure all-i64 leaf f(x) = x*3 + 7.
    const SRC: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 65552
  vm = i64.load vaddr
  v3 = i64.const 3
  vleaf = call 1 (v3)
  vr = i64.add vm vleaf
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
    const MARKER: i64 = 424242;
    let m = parse_module(SRC).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    let prog = bytecode::SharedProgram::compile(&m).expect("in subset");
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, true]);
    let run_with = |prots: &temen_interp::PageMap| -> (Result<Vec<Value>, Trap>, u32) {
        // Flat, as the warm session's window is: a tier-up over a window without a flat span runs
        // interpreted (#1414 B6-3a), and `Region::new` is flat only where it is an `mmap`.
        let back = std::sync::Arc::new(
            temen_interp::Region::owned_zeroed(1 << 17, 4096).expect("a flat 128 KiB backing"),
        );
        // "Restore the warm image": plant the marker bytes directly in the backing, exactly what
        // the warm session's memcpy restore does before arming the run.
        for (i, b) in MARKER.to_le_bytes().iter().enumerate() {
            back.set_byte(65552 + i as u64, *b);
        }
        let mut run = prog
            .coop_run_over_grown(
                0,
                &[],
                FUEL,
                Host::new(),
                Some(TierUpConfig {
                    root: ProgramEmit {
                        eligible: std::sync::Arc::clone(&eligible),
                        page_checked: false,
                    },
                    leaf: None,
                    modules: None,
                }),
                back,
                17,
                prots,
                None,
            )
            .expect("entry in range");
        let mut tierups = 0u32;
        loop {
            match run.run() {
                bytecode::CoopEvent::Done(vals) => return (Ok(vals), tierups),
                bytecode::CoopEvent::Trapped(t) => return (Err(t), tierups),
                bytecode::CoopEvent::Idle
                | bytecode::CoopEvent::Paused
                | bytecode::CoopEvent::CapPark { .. } => {
                    panic!("unexpected Idle (suspend_on_idle is never armed here)")
                }
                bytecode::CoopEvent::Resume { .. } => {
                    panic!("unexpected Resume (no leaf image here)")
                }
                bytecode::CoopEvent::JitInvoke { .. } => panic!("unexpected JitInvoke"),
                bytecode::CoopEvent::TierUp { func, argv, .. } => {
                    tierups += 1;
                    assert!(tierups < 10, "runaway tier-ups");
                    assert_eq!(func, 1, "only the leaf is eligible");
                    run.deliver_tierup(&[argv[0] * 3 + 7]);
                }
            }
        }
    };

    // Seeded (one Rw entry for the grown page): the marker reads back and the leaf tiers up.
    let seeded =
        temen_interp::PageMap::from_entries(temen_interp::host_page_size(), 0, &[(65536, 1)])
            .expect("a plain Rw entry decodes");
    let (got, tierups) = run_with(&seeded);
    assert_eq!(
        got,
        Ok(vec![Value::I64(MARKER + 16)]),
        "the restored grown page must be readable and the leaf serviced"
    );
    assert_eq!(tierups, 1, "the eligible leaf tiers up exactly once");

    // Unseeded: the same bytes are in the backing, but with no page-state entry the grown-page
    // load faults — the restore is the page MAP, not just the bytes (fail-closed).
    let (got, _) = run_with(&temen_interp::PageMap::empty());
    assert!(
        got.is_err(),
        "an unseeded run must fault on the grown-page load, got {got:?}"
    );
}
