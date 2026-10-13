//! #2268 — a run the process's pool of mappings ([`temen_interp::maps_pool`]) cannot hold does not
//! start, on every engine alike: its activation, which no grant refuses, is held against the pool, and
//! the run ends with `ThreadFault` before its first instruction, its main vCPU refused. Once the runs
//! that held the pool end, it runs. Its own binary: the pool is the process's.

use temen_interp::{maps_pool, Host, RootWindow, Value, MAPS_PER_DOMAIN, MAPS_PER_VCPU};
use temen_run::{instantiate, Backend, Outcome, RunConfig};

const SEVEN: &str = "func () -> (i64) {\nblock 0 () {\n  v = i64.const 7\n  return v\n  }\n}\n";

/// `m` run on `backend`: what it returned, or why it did not.
fn run(m: &temen_ir::Module, backend: Backend) -> Result<Vec<Value>, String> {
    match instantiate(m.clone())?
        .run(backend, &RunConfig::default())?
        .outcome
    {
        Outcome::Returned(v) => Ok(v),
        Outcome::Exited(code) => Err(format!("exited {code}")),
    }
}

#[test]
fn a_run_the_pool_cannot_hold_does_not_start_on_any_engine() {
    let pool = maps_pool();
    if pool > 1 << 22 {
        eprintln!("skipped: a pool of {pool} mappings is too large to fill");
        return;
    }
    let m = temen_text::parse_module(SEVEN).expect("parse");
    let engines = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];
    for e in engines {
        assert_eq!(run(&m, e), Ok(vec![Value::I64(7)]), "{e:?} runs");
    }
    // Activations that hold the pool, as other runs of the process would, until one is refused.
    let mut roots = Vec::new();
    let refused = (0..=pool / (MAPS_PER_DOMAIN + MAPS_PER_VCPU)).any(|_| {
        let mut h = Host::new();
        let held = h.begin_activation(0, RootWindow::default()).is_some();
        roots.push(h);
        !held
    });
    assert!(refused, "the pool refuses an activation once it is full");
    for e in engines {
        let ended = run(&m, e).expect_err("no room");
        assert!(
            ended.contains("ThreadFault"),
            "{e:?} ends a run it has no room for with ThreadFault: {ended}"
        );
    }
    drop(roots);
    for e in engines {
        assert_eq!(run(&m, e), Ok(vec![Value::I64(7)]), "{e:?} runs again");
    }
}
