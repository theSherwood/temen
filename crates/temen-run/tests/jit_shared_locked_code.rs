//! temen#2389 — a program that runs over the **locked** powerbox compiles once for the runs handed
//! one `JitCodeCache`. It spawns a vCPU, so its `call.cap`s can come from two threads at once, and its
//! code bakes the locked thunk: the cache keeps it apart from a raw program's code. Each run's
//! instance of it stands up a thread domain of its own, which its vCPU runs and is joined on.
//!
//! The only test in its binary, so the process-wide compile counter moves only for its runs.

use temen_interp::Value;
use temen_run::{instantiate, Backend, JitCodeCache, Outcome, RunConfig};
use temen_text::parse_module;

/// Spawns a vCPU that answers its argument plus one, and joins it.
const SPAWNS: &str = "memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  varg = i64.const 41
  vt = thread.spawn 1 vsp varg
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  return v3
  }
}
";

fn run(backend: Backend, cfg: &RunConfig) -> Outcome {
    instantiate(parse_module(SPAWNS).expect("parse"))
        .expect("instantiate")
        .run_with_caps_and_host(backend, cfg, &[], None)
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"))
        .outcome
}

/// How many programs a JIT run under `cfg` compiles.
fn jit_compiles(cfg: &RunConfig) -> u64 {
    let before = temen_jit::module_compiles();
    assert_eq!(
        run(Backend::Jit, cfg),
        Outcome::Returned(vec![Value::I64(42)]),
        "Jit"
    );
    temen_jit::module_compiles() - before
}

#[test]
fn a_program_over_the_locked_powerbox_compiles_once_for_the_runs_that_keep_its_cache() {
    for backend in [Backend::TreeWalk, Backend::Bytecode] {
        assert_eq!(
            run(backend, &RunConfig::default()),
            Outcome::Returned(vec![Value::I64(42)]),
            "{backend:?}"
        );
    }
    let cache = JitCodeCache::default();
    let cached = RunConfig {
        jit_code: Some(cache.clone()),
        ..RunConfig::default()
    };
    assert_eq!(jit_compiles(&cached), 1, "the first run compiles it");
    assert_eq!(jit_compiles(&cached), 0, "a later run instances it");
    assert_eq!(cache.compiled(), 1);
}
