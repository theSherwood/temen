//! **Phase 3 — uniform run config across backends.** The same powerbox program runs on the
//! tree-walker, the bytecode engine, and the JIT through one `RunConfig`, and the resource limits
//! (fuel, spawn quota, window size) apply uniformly where each backend supports them. Proves the
//! "pick a backend, set the knobs, run" interface from `temen_run::Instance::run` / `run_diff`.
//!
//! Gated `#![cfg(unix)]` like the other JIT differential suites.
#![cfg(unix)]

use temen_run::{instantiate, Backend, Instance, Limits, Outcome, RunConfig, Value};

/// A minimal fixed-powerbox program: a paramless exported `_start` whose `write` manifest import
/// binds to the stdout slot at instantiation (the handle operand is a vestigial dummy).
const HELLO: &str = "\
memory 15
data ro 16384 \"hello, powerbox\\n\"
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 16
  v3 = call.sym \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  return v4
  }
}
";

fn hello_instance() -> Instance {
    let module = temen_text::parse_module(HELLO).expect("parse");
    instantiate(module).expect("instantiate")
}

/// All three backends run the same program through one `RunConfig` and produce identical output.
#[test]
fn every_backend_runs_under_one_config() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = hello_instance()
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(run.stdout, b"hello, powerbox\n", "{backend:?} stdout");
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I32(0)]),
            "{backend:?} outcome"
        );
    }
}

/// The differential entry (`run_diff`) cross-checks tree-walk vs JIT under the config.
#[test]
fn run_diff_under_config() {
    let run = hello_instance()
        .run_diff(&RunConfig::default())
        .expect("diff");
    assert_eq!(run.stdout, b"hello, powerbox\n");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
}

/// A looping variant: spins a bounded counter loop (each back-edge an IR **safepoint**) before the
/// same `write`, so a tight fuel budget runs the interpreters out of fuel *at a back-edge* — the unit
/// fuel is metered in since the fuel unification (straight-line code like `HELLO` is now free, so it
/// can't be out-of-fueled). The JIT compiles the same safepoint checks in when the fuel is bounded.
const LOOP_HELLO: &str = "\
memory 15
data ro 16384 \"hello, powerbox\\n\"
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  n0 = i32.const 100
  br 1(n0)
}
block 1 (n: i32) {
  one = i32.const 1
  n2 = i32.sub n one
  br_if n2 1(n2) 2()
}
block 2 () {
  v0 = i32.const 0
  v1 = i64.const 16384
  v2 = i64.const 16
  v3 = call.sym \"write\" (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i32.const 0
  return v4
  }
}
";

fn loop_hello_instance() -> Instance {
    let module = temen_text::parse_module(LOOP_HELLO).expect("parse");
    instantiate(module).expect("instantiate")
}

/// `fuel` bounds every backend alike (#1944 slice 3, #1705): the interpreters meter it at IR
/// safepoints (loop back-edges + function entries) and the JIT compiles the same checks in when the
/// run's fuel is bounded, all drawing from the run's budget chain.
#[test]
fn fuel_bounds_every_backend() {
    let tight = RunConfig {
        limits: Limits {
            fuel: Some(1),
            ..Limits::default()
        },
        ..RunConfig::default()
    };
    // A 1-safepoint budget out-of-fuels every backend before the program finishes its 100-iteration
    // counter loop.
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = loop_hello_instance().run(backend, &tight);
        assert!(
            run.as_ref().is_err_and(|e| e.contains("OutOfFuel")),
            "fuel=1 must out-of-fuel {backend:?}: {run:?}"
        );
    }
}

/// A thread and its spawner that each take `N` loop back-edges, one after the other: the root spawns
/// func 1 and joins it, then loops itself.
const THREAD_THEN_ROOT: &str = "\
memory 16
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  vsp = i64.const 32768
  vn = i64.const 10000
  vh = thread.spawn 1 vsp vn
  vj = thread.join vh
  br 1(vn)
}
block 1 (vi: i64) {
  one = i64.const 1
  vk = i64.sub vi one
  vz = i64.const 0
  vc = i64.ne vk vz
  br_if vc 1(vk) 2()
}
block 2 () {
  r = i32.const 0
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  br 1(v0)
}
block 1 (vi: i64) {
  one = i64.const 1
  vk = i64.sub vi one
  vz = i64.const 0
  vc = i64.ne vk vz
  br_if vc 1(vk) 2()
}
block 2 () {
  r = i64.const 0
  return r
  }
}
";

/// #1944 slice 3 — a domain's threads burn one fuel budget (INVARIANTS #6): 15 000 fuel covers the
/// thread's 10 000 back-edges or the root's, not both, on every backend. (The interpreters used to
/// hand each thread a copy of its spawner's fuel, so each could burn the whole budget.)
#[test]
fn a_domains_threads_share_one_fuel_budget() {
    let module = temen_text::parse_module(THREAD_THEN_ROOT).expect("parse");
    let with_fuel = |fuel| RunConfig {
        limits: Limits {
            fuel: Some(fuel),
            ..Limits::default()
        },
        ..RunConfig::default()
    };
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let inst = || instantiate(module.clone()).expect("instantiate");
        let run = inst().run(backend, &with_fuel(30_000));
        assert!(run.is_ok(), "{backend:?}: 30 000 fuel covers both: {run:?}");
        let run = inst().run(backend, &with_fuel(15_000));
        assert!(
            run.as_ref().is_err_and(|e| e.contains("OutOfFuel")),
            "{backend:?}: 15 000 fuel covers one: {run:?}"
        );
    }
}

/// The "amount of memory available" knob (`memory_size_log2`) overrides the module's declared window
/// uniformly across backends.
#[test]
fn memory_window_override_applies_to_every_backend() {
    let cfg = RunConfig {
        memory_size_log2: Some(22), // 4 MiB — larger than the module's declared window
        ..RunConfig::default()
    };
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let run = hello_instance()
            .run(backend, &cfg)
            .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
        assert_eq!(
            run.stdout, b"hello, powerbox\n",
            "{backend:?} under 4 MiB window"
        );
    }
}
