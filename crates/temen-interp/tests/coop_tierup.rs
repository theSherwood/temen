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
//! The distinguishing coverage over the single-vCPU `vcpu_tierup` test: tier-up fires on **two tasks**
//! — the root and the spawned worker — proving the paused-task delivery routes to the right vCPU across
//! the scheduler's multiplexing, and that a spawned same-module child inherits the eligibility bitmap.
//! Plus the #926 "Differentials" concurrency shapes: tier-up co-runs with a **self-hosted fiber
//! scheduler** (`cont.new`/`cont.resume` worker fibers) and with a **single-threaded live futex** wake
//! cell (`atomic.wait`/`atomic.notify` park-and-wake) — the scheduler services the concurrency while the
//! root's leaf still tiers up, matching the pure-interp oracle. (A dead-linked-concurrency guest tiering
//! up its leaves is slice 1, #930; the browser-host wasm servicing is `browser/tests/coop_tierup_driver.rs`.)
//!
//! Fuel is **bounded** (not `u64::MAX`) so any regression surfaces as an `OutOfFuel` trap rather than a
//! hang; the guests finish in well under the budget, so a bound never perturbs the differential.

use temen_interp::bytecode::TierUpConfig;
use temen_interp::{bytecode, Host, Trap, Value};
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

/// Drive the cooperative run with tier-up enabled, emulating the browser host: each `TierUp(func,
/// argv)` is serviced by running `func` standalone (what the emitted region computes) and delivering
/// its i64 results back. Returns the whole-run result and the number of tier-ups serviced.
fn coop_tierup_run(
    m: &temen_ir::Module,
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
            bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected JitInvoke (no vm_jit guest here)")
            }
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
    temen_verify::verify_module(&m).expect("verify");
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
    temen_verify::verify_module(&m).expect("verify");
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
            bytecode::CoopEvent::JitInvoke { .. } => {
                panic!("unexpected JitInvoke (no vm_jit guest here)")
            }
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
    let (got, tierups) = coop_tierup_run(&m, eligible);

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
        tierups, 1,
        "expected 1 tier-up (the root's leaf; fibers run their leaf interpreted), got {tierups}"
    );
}

// #926 differential (the issue's "Differentials" list): a **single-threaded, cooperative live futex**
// wake cell whose compute leaf tiers up. The root creates a fiber that untimed-`atomic.wait`s on cell 0
// (parking the FIBER, not the vCPU — the root keeps running), `atomic.notify`s the cell to wake it, runs
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
  vaddr = i64.const 0
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
  vaddr = i64.const 0
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
    let (got, tierups) = coop_tierup_run(&m, eligible);

    // The fiber wakes WAIT_WOKEN(0), then the leaf f(4) = 19 → 0 + 19 = 19.
    assert_eq!(want, Ok(vec![Value::I64(19)]), "oracle value");
    assert_eq!(
        got, want,
        "cooperative tier-up over a live futex wake cell diverged from the pure-interp oracle"
    );
    // Non-vacuity: the compute leaf tiers up after the live futex park/wake completes single-threaded.
    assert_eq!(
        tierups, 1,
        "expected exactly one tier-up (the root's compute leaf), got {tierups}"
    );
}

// #816 fail-closed — a task whose window lives in `extra_envs` must NOT tier up. The tier-up
// driver's `win` pointer and its page-state/extent introspection (`mem_map_info`/`mem_map_version`/
// `window_scalar_extent`) all read the run's ROOT window, so an emitted region for an env-carrying
// task (a §14 confined child's thread, a fork twin) would address the wrong backing and admit the
// wrong page map — the #839 JACL trap shape. The gate leaves such tasks interpreting (correct, just
// unaccelerated) while the root's own eligible call still tiers up.
//
// func 0 (root; arg = its Instantiator handle): instantiates a same-module confined child at f1
// (4 KiB carve at 64 KiB), joins it, calls the eligible leaf f3 itself, and sums. func 1 (child
// entry): thread.spawns f2 INSIDE the child env and joins it — the one shape that used to inherit
// the bitmap into an env-carrying task. func 2 (child-env worker): calls the leaf — the gated call.
// func 3: the pure all-i64 leaf f(x) = x*3 + 7.
const SRC_CHILD_ENV: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vh = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vsl, vq)
  vj = cap.call 6 1 (i32) -> (i64) v0 (vh)
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
  vw = thread.join vt
  return vw
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
fn coop_tierup_child_env_thread_stays_interpreted() {
    let m = parse_module(SRC_CHILD_ENV).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    // Drive with an Instantiator granted over the root window (the §14 spawn authority); the
    // shared `coop_tierup_run` harness has no grant seam, so inline its loop here.
    let run_with = |tierup: Option<TierUpConfig>| -> (Result<Vec<Value>, Trap>, u32) {
        let mut host = Host::new();
        let inst = host.grant_instantiator(0, 128 << 10);
        let args = [Value::I32(inst)];
        let mut run = bytecode::CoopRun::new(&m, 0, &args, FUEL, host, tierup)
            .expect("supported")
            .expect("entry in range");
        let mut tierups = 0u32;
        loop {
            match run.run() {
                bytecode::CoopEvent::Done(vals) => return (Ok(vals), tierups),
                bytecode::CoopEvent::Trapped(t) => return (Err(t), tierups),
                bytecode::CoopEvent::JitInvoke { .. } => {
                    panic!("unexpected JitInvoke (no vm_jit guest here)")
                }
                bytecode::CoopEvent::TierUp { func, argv, .. } => {
                    tierups += 1;
                    assert!(
                        tierups < 50,
                        "runaway tier-ups (last func={func}, argv={argv:?})"
                    );
                    let targs: Vec<Value> = argv.iter().map(|&s| Value::I64(s)).collect();
                    let mut fuel = FUEL;
                    match bytecode::compile_and_run(&m, func, &targs, &mut fuel).expect("supported")
                    {
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
    };

    // Oracle: pure-interp cooperative run. Child worker f(5)=22, root f(3)=16 → 38.
    let (want, oracle_tierups) = run_with(None);
    assert_eq!(want, Ok(vec![Value::I64(38)]), "oracle value");
    assert_eq!(oracle_tierups, 0, "the oracle never tiers up");

    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, false, true]);
    let (got, tierups) = run_with(Some(TierUpConfig {
        eligible,
        page_checked: false,
    }));
    assert_eq!(
        got, want,
        "cooperative tier-up run with a confined child diverged from the pure-interp oracle"
    );
    // The gate: exactly the root's `call 3` tiers up; the child-env worker's interprets. (Before
    // #816's gate the worker inherited the bitmap and this was 2.)
    assert_eq!(
        tierups, 1,
        "only the root-env call may tier up (#816), got {tierups}"
    );
}
