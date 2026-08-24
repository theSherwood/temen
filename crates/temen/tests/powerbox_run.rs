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
/// can't be out-of-fueled). The JIT still ignores the interpreter fuel counter.
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

/// `fuel` bounds the **interpreters** (metered at IR safepoints — loop back-edges + function entries)
/// but is ignored by the JIT (which bounds runaway guests with a `deadline` instead) — the one
/// documented backend-specific knob, shown uniform-API but honest about who honors it. Fuel is now
/// unified to safepoints across both interpreters, so the two agree on where they run out.
#[test]
fn fuel_bounds_interpreters_not_the_jit() {
    let tight = RunConfig {
        limits: Limits {
            fuel: Some(1),
            ..Limits::default()
        },
        ..RunConfig::default()
    };
    // A 1-safepoint budget out-of-fuels both interpreters at a loop back-edge, before the program
    // finishes its 100-iteration counter loop.
    assert!(
        loop_hello_instance()
            .run(Backend::TreeWalk, &tight)
            .is_err(),
        "fuel=1 must out-of-fuel the tree-walker"
    );
    assert!(
        loop_hello_instance()
            .run(Backend::Bytecode, &tight)
            .is_err(),
        "fuel=1 must out-of-fuel the bytecode engine"
    );
    // The JIT has no interpreter fuel counter, so it ignores `fuel` and runs to completion.
    let run = loop_hello_instance()
        .run(Backend::Jit, &tight)
        .expect("the JIT ignores interpreter fuel");
    assert_eq!(run.stdout, b"hello, powerbox\n");
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
