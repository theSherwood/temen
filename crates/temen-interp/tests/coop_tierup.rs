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

// #816 env-routed tier-up — a task whose window lives in `extra_envs` (a §14 confined child, or a
// thread it spawns) now TIERS UP over its own carve: the engine routes the driver's per-event reads
// (`pending_win`, `window_scalar_extent`, `mem_map_info`/`mem_map_version`) to the pending
// round-trip's task env, and the eligibility gates (`tierup_servable`) admit any window sharing the
// root backing. This test pins the inheritance: the child entry's and the child-env worker's calls
// to the eligible leaf both surface, and the run still matches the pure-interp oracle.
//
// func 0 (root; arg = its Instantiator handle): instantiates a same-module confined child at f1
// (4 KiB carve at 64 KiB), joins it, calls the eligible leaf f3 itself, and sums. func 1 (child
// entry): calls the leaf directly (the instantiate-arm inheritance), thread.spawns f2 INSIDE the
// child env (the spawn-arm inheritance) and joins it. func 2 (child-env worker): calls the leaf.
// func 3: the pure all-i64 leaf f(x) = x*3 + 7.
const SRC_CHILD_ENV: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
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

#[test]
fn coop_tierup_child_env_tasks_tier_up() {
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

    // Oracle: pure-interp cooperative run. Worker f(5)=22 + child entry f(7)=28 → child 50;
    // root f(3)=16 → 66.
    let (want, oracle_tierups) = run_with(None);
    assert_eq!(want, Ok(vec![Value::I64(66)]), "oracle value");
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
    // #816 env routing: all three eligible calls surface — the root's, the confined child entry's
    // (instantiate-arm inheritance), and the child-env worker's (spawn-arm inheritance).
    assert_eq!(
        tierups, 3,
        "root + child entry + child worker must all tier up (#816), got {tierups}"
    );
}

/// #816 — the routed **flat window view**: while a confined child's tier-up is pending,
/// [`CoopRun::pending_win`] must hand the driver the CHILD's carve (base = root backing + carve
/// offset, len = carve size), and a write through it must land where the parent reads it in its own
/// window (the shared backing). Emulates the browser host: each event writes a marker byte at the
/// pending window's offset 8 (what an emitted store would do), then delivers the leaf's value. The
/// guest sums the leaf results with the markers read back through the parent window — so a
/// mis-routed pointer (root base for a child event) or a mis-sized span fails the value assert.
/// Unix-only: the native test backing must be flat (`Region::Mapped`) for the raw view; on the
/// browser the backing is always flat (`Region::shared` over the cdylib buffer).
#[cfg(unix)]
#[test]
fn coop_tierup_pending_win_routes_to_the_child_carve() {
    // func 0 (root): instantiate child at 65536 (4 KiB), join → child result; call leaf(3) itself;
    // read marker bytes at [16392] (root's own event — above the #1094 guard, since the root window
    // is guarded) and [65536+8] (the child's event — its 4 KiB carve is sub-guard/unguarded, visible
    // through the parent window on the shared backing); return child + local + markers.
    // func 1 (child entry): call leaf(5), return it. func 2: the leaf f(x) = x*3 + 7.
    const SRC: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  v3 = i64.const 3
  vlocal = call 2 (v3)
  vma = i64.const 16392
  vm0 = i32.load8_u vma
  vm0e = i64.extend_i32_u vm0
  vca = i64.const 65544
  vm1 = i32.load8_u vca
  vm1e = i64.extend_i32_u vm1
  vs1 = i64.add vj vlocal
  vs2 = i64.add vs1 vm0e
  vs3 = i64.add vs2 vm1e
  return vs3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
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
    let m = parse_module(SRC).unwrap();
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let args = [Value::I32(inst)];
    let eligible: std::sync::Arc<[bool]> = std::sync::Arc::from(vec![false, false, true]);
    // The browser shape: the window lives over a caller-provided flat backing with the reservation
    // clamped to it, so `pending_win` resolves and spans exactly the run window / child carve.
    let back = std::sync::Arc::new(temen_interp::Region::new(1 << 17, 4096));
    let mut run = bytecode::CoopRun::new_over(
        &m,
        0,
        &args,
        FUEL,
        host,
        Some(TierUpConfig {
            eligible,
            page_checked: false,
        }),
        &[],
        17,
        back,
    )
    .expect("supported")
    .expect("entry in range");
    // Expected per-event windows: the root's event spans the full 2^17 window at the backing base;
    // the child's spans its 4 KiB carve at base + 65536.
    let mut spans: Vec<u64> = Vec::new();
    let result = loop {
        match run.run() {
            bytecode::CoopEvent::Done(vals) => break Ok(vals),
            bytecode::CoopEvent::Trapped(t) => break Err(t),
            bytecode::CoopEvent::JitInvoke { .. } => panic!("unexpected JitInvoke"),
            bytecode::CoopEvent::TierUp { func, argv, .. } => {
                assert_eq!(func, 2, "only the leaf is eligible");
                let (ptr, len) = run
                    .pending_win()
                    .expect("a flat native backing resolves the pending window");
                spans.push(len);
                // The emitted region's effect, emulated over the routed window: store the marker at
                // a window-relative offset the guest can read back. The root window is guarded
                // ([0,16384) unmapped), so its marker goes above the guard at 16392; the child's
                // 4 KiB carve is sub-guard (unguarded), so its marker stays at offset 8. (Keyed by
                // the span: only the root event spans the whole guarded window.)
                // SAFETY: the paused task is parked inside the event; `ptr` spans `len` bytes of the
                // run backing, exclusively ours until deliver (single-threaded pump).
                let moff = if len >= 16384 { 16392 } else { 8 };
                unsafe { std::ptr::write(ptr.cast_mut().add(moff), 21u8) };
                let x = argv[0];
                run.deliver_tierup(&[x * 3 + 7]);
            }
        }
    };
    // Child event first (the scheduler runs the child to completion inside the join), then root's.
    assert_eq!(
        spans,
        vec![4096, 1 << 17],
        "pending_win must span the child carve for the child's event and the run window for the root's"
    );
    // child leaf f(5)=22 + root leaf f(3)=16 + root marker 21 + child marker 21 = 80.
    assert_eq!(
        result,
        Ok(vec![Value::I64(80)]),
        "markers must land in each task's own window"
    );
}

// #816 item 3 — a **fork twin** tiers up over its own private flat window. The guest topology is
// the proven browser fork probe (`FORK_TWIN`, FORK.md §9): a manager (f0) spawns a server (f1,
// handlers f2 = `clone_caller`, f3 = `reap`) and a guest (f4) as §14 children; the guest calls
// `svc.fork`, the servicer duplicates it (`Mem::fork_private` + `Host::fork_powerbox`), and BOTH
// copies — the original (reply 100) and the twin (reply 200) — call the eligible leaf f5 on their
// reply before writing the result to the shared stdout; the original then `svc.wait`s the twin.
// With the flat twin-backing seam, the twin's private window is an owned flat buffer, so its leaf
// call TIERS UP (previously: interpreted fail-closed on wasm) — and its event's `pending_win` must
// resolve OUTSIDE the root backing (the twin's own allocation), while the original's resolves
// inside it (its §14 carve). Differential against the same run with no bitmap.
const SRC_FORK_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 17600
  i64.store q1a0 q1v0
  q1a1 = i64.const 17608
  i64.store q1a1 q1v1
  q1a2 = i64.const 17616
  i64.store q1a2 q1v2
  q1a3 = i64.const 17624
  i64.store q1a3 q1v3
  q1a4 = i64.const 17632
  i64.store q1a4 q1v4
  q1a5 = i64.const 17640
  i64.store q1a5 q1v4
  q1a6 = i64.const 17648
  i64.store q1a6 q1v4
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
  q2v0 = i64.const 17179869184
  q2v1 = i64.const 135168
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2v5 = i64.const 16640
  q2v6 = i64.const 2
  q2a0 = i64.const 17664
  i64.store q2a0 q2v0
  q2a1 = i64.const 17672
  i64.store q2a1 q2v1
  q2a2 = i64.const 17680
  i64.store q2a2 q2v2
  q2a3 = i64.const 17688
  i64.store q2a3 q2v3
  q2a4 = i64.const 17696
  i64.store q2a4 q2v4
  q2a5 = i64.const 17704
  i64.store q2a5 q2v5
  q2a6 = i64.const 17712
  i64.store q2a6 q2v6
  vc = call.cap 6 17 (i64) -> (i32) v0 (q2a0)
  vjc = call.cap 6 1 (i32) -> (i64) v0 (vc)
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
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = self.resolve vp8 vl1
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
  vp16 = i64.const 16
  i64.store vp16 vleaf
  vlen = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
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

/// Every target (unlike the unix-only pending-win test): the run backing is an **owned flat**
/// buffer (`Region::owned_zeroed` — the same shape as the browser's `Region::shared` window), so
/// the raw window views resolve even where `Region::new` would fall back to `Paged` (Windows),
/// and the twin-backing seam's non-unix arm gets real end-to-end CI coverage there.
#[test]
fn coop_tierup_fork_twin_tiers_up_over_its_private_flat_window() {
    const FORK_FUEL: u64 = 40_000_000;
    let m = std::sync::Arc::new(parse_module(SRC_FORK_TWIN).unwrap());
    temen_verify::verify_module(&m).expect("verify");
    // (ptr-in-root-backing?, len) per tier-up event, plus the whole-run result and stdout.
    #[allow(clippy::type_complexity)]
    let run_with =
        |tierup: Option<TierUpConfig>| -> (Result<Vec<Value>, Trap>, Vec<i64>, Vec<(bool, u64)>) {
            let mut host = Host::new();
            host.set_self_module(&m);
            let inst = host.grant_instantiator(0, 1u64 << 18);
            let sink = host.shared_stdout();
            let out_h = host.grant_stream(temen_interp::StreamRole::Out);
            let args = [Value::I32(inst), Value::I32(out_h)];
            let back = std::sync::Arc::new(
                temen_interp::Region::owned_zeroed(1 << 18, 4096).expect("256 KiB allocates"),
            );
            let root_base = back.raw_base().expect("flat backing") as usize;
            let mut run = bytecode::CoopRun::new_over(
                &m,
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
            let mut events: Vec<(bool, u64)> = Vec::new();
            let result = loop {
                match run.run() {
                    bytecode::CoopEvent::Done(vals) => break Ok(vals),
                    bytecode::CoopEvent::Trapped(t) => break Err(t),
                    bytecode::CoopEvent::JitInvoke { .. } => panic!("unexpected JitInvoke"),
                    bytecode::CoopEvent::TierUp { func, argv, .. } => {
                        assert_eq!(func, 5, "only the leaf is eligible");
                        assert!(events.len() < 10, "runaway tier-ups");
                        let (ptr, len) = run
                            .pending_win()
                            .expect("every tiering task's window must resolve a flat view");
                        let in_root =
                            (ptr as usize) >= root_base && (ptr as usize) < root_base + (1 << 18);
                        events.push((in_root, len));
                        let x = argv[0];
                        run.deliver_tierup(&[x * 3 + 7]);
                    }
                }
            };
            let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
            let mut out: Vec<i64> = bytes
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            out.sort_unstable();
            (result, out, events)
        };

    // Oracle: pure-interp cooperative run. The original resumes with 100 and the twin with 200;
    // both write leaf(reply) to the shared stdout: leaf(100) = 307, leaf(200) = 607.
    let (want, want_out, oracle_events) = run_with(None);
    assert_eq!(want, Ok(vec![Value::I64(100)]), "oracle value");
    assert_eq!(
        want_out,
        vec![307, 607],
        "both copies ran and wrote (oracle)"
    );
    assert!(oracle_events.is_empty(), "the oracle never tiers up");

    let eligible: std::sync::Arc<[bool]> =
        std::sync::Arc::from(vec![false, false, false, false, false, true]);
    let (got, got_out, events) = run_with(Some(TierUpConfig {
        eligible,
        page_checked: false,
    }));
    assert_eq!(got, want, "fork-twin tier-up run diverged from the oracle");
    assert_eq!(got_out, want_out, "stdout parity");
    // Both leaf calls tier up: the original guest's (a §14 child — its carve window lies INSIDE
    // the root backing) and the fork twin's (its private `fork_private` window is its OWN flat
    // allocation, outside the root backing). Both windows span the 4 KiB carve geometry.
    let mut sorted = events.clone();
    sorted.sort_unstable();
    assert_eq!(
        sorted,
        vec![(false, 4096), (true, 4096)],
        "one in-root-carve event (the original) + one private-window event (the twin), got {events:?}"
    );
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
    let run_with = |prots: &[(u64, u8)]| -> (Result<Vec<Value>, Trap>, u32) {
        let back = std::sync::Arc::new(temen_interp::Region::new(1 << 17, 4096));
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
                    eligible: std::sync::Arc::clone(&eligible),
                    page_checked: false,
                }),
                back,
                17,
                prots,
            )
            .expect("entry in range");
        let mut tierups = 0u32;
        loop {
            match run.run() {
                bytecode::CoopEvent::Done(vals) => return (Ok(vals), tierups),
                bytecode::CoopEvent::Trapped(t) => return (Err(t), tierups),
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
    let (got, tierups) = run_with(&[(65536, 1)]);
    assert_eq!(
        got,
        Ok(vec![Value::I64(MARKER + 16)]),
        "the restored grown page must be readable and the leaf serviced"
    );
    assert_eq!(tierups, 1, "the eligible leaf tiers up exactly once");

    // Unseeded: the same bytes are in the backing, but with no page-state entry the grown-page
    // load faults — the restore is the page MAP, not just the bytes (fail-closed).
    let (got, _) = run_with(&[]);
    assert!(
        got.is_err(),
        "an unseeded run must fault on the grown-page load, got {got:?}"
    );
}
