//! **Domain lifetime & teardown** (DESIGN.md §12 "Domain lifetime & teardown", INVARIANTS #6
//! "one lifetime" — owner decision 2026-07-24): executors never anchor a domain's lifetime.
//! `exit`/trap tears the whole domain down immediately (parked threads included, on every
//! engine), and root completion ends a batch run with its daemons abandoned — non-preemptively,
//! so the assertions here compare only the root outcome + effects that happened-before it.
//!
//! These are the fixtures from the jacl timed-wait regression report (2026-07-24), whose
//! JIT/tree-walker divergence was this teardown gap: pre-decision, the interpreter held the run
//! open until every parked daemon's `MAX_WAIT`-clamped wait fired (10 s per lap, forever for a
//! re-parking pool), while the JIT stored the exit trap that no parked waiter ever woke to
//! observe (a hang). Each test asserts **prompt** completion — generously below the 10 s clamp,
//! so a reintroduced wait-out-the-daemons regression fails loudly without being timing-flaky.
#![cfg(unix)]

use std::time::{Duration, Instant};
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig, Value};

/// Well under the 10 s `MAX_WAIT` clamp, well over CI jitter.
const PROMPT: Duration = Duration::from_secs(5);

fn powerbox(src: &str) -> temen_run::Instance {
    let module = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&module).expect("verify");
    let imports = Imports::new()
        .provide("write", HostCap::stdout())
        .provide("exit", HostCap::exit());
    instantiate_with_imports(module, imports).expect("instantiate")
}

/// The jacl daemon shape: `_start` spawns a worker that parks on an **indefinite** futex wait,
/// does a 10 ms timed wait of its own (nobody notifies — it times out), prints, and exits.
/// The parked daemon is abandoned at the exit; the run must end promptly on both engines.
const EXIT_WITH_PARKED_DAEMON: &str = r#"
memory 15
import 0 "write" (i64, i64) -> (i64)
import 1 "exit" (i32) -> ()
data ro 16384 "done\n"
export 0 func "_start" 0
func () -> () {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = i64.const 16400
  v3 = i32.const 0
  v4 = i64.const 10000000
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.const 16384
  v7 = i64.const 5
  v8 = call.import 0 (v6, v7)
  v9 = i32.const 0
  call.import 1 (v9)
  unreachable
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 16392
  v2 = i32.const 0
  v3 = i64.const -1
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 7
  return v5
  }
}
"#;

#[test]
fn exit_tears_down_parked_daemons() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let instance = powerbox(EXIT_WITH_PARKED_DAEMON);
        let t0 = Instant::now();
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        let took = t0.elapsed();
        assert_eq!(run.outcome, Outcome::Exited(0), "{backend:?}");
        assert_eq!(
            run.stdout, b"done\n",
            "{backend:?}: the pre-exit print lands"
        );
        assert!(
            took < PROMPT,
            "{backend:?}: exit must not wait out the daemon (took {took:?})"
        );
    }
}

/// Root **returns** (no exit call) with the daemon still parked: root completion ends the batch
/// activation, and with it the run — the C rule (`main` returning is `exit`), uniformly.
const RETURN_WITH_PARKED_DAEMON: &str = r#"
memory 15
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = i64.const 16400
  v3 = i32.const 0
  v4 = i64.const 10000000
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.const 42
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 16392
  v2 = i32.const 0
  v3 = i64.const -1
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 7
  return v5
  }
}
"#;

#[test]
fn root_return_tears_down_parked_daemons() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let instance = powerbox(RETURN_WITH_PARKED_DAEMON);
        let t0 = Instant::now();
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        let took = t0.elapsed();
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I64(42)]),
            "{backend:?}"
        );
        assert!(
            took < PROMPT,
            "{backend:?}: root return must not wait out the daemon (took {took:?})"
        );
    }
}

/// A **sibling's trap** is domain-wide (I37 / invariant 6): the daemon traps while the root is
/// parked in a long timed wait — the run must end with the trap promptly on both engines, not
/// after the root's full timeout, and never by waiting anything out.
const SIBLING_TRAP_KILLS_DOMAIN: &str = r#"
memory 15
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  v1 = thread.spawn 1 v0 v0
  v2 = i64.const 16400
  v3 = i32.const 0
  v4 = i64.const 8000000000
  v5 = i32.atomic.wait v2 v3 v4
  v6 = i64.const 42
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  unreachable
  }
}
"#;

#[test]
fn sibling_trap_tears_down_the_domain() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let instance = powerbox(SIBLING_TRAP_KILLS_DOMAIN);
        let t0 = Instant::now();
        let err = match instance.run(backend, &RunConfig::default()) {
            Err(e) => e,
            Ok(run) => panic!(
                "{backend:?}: sibling trap must end the run, got {:?}",
                run.outcome
            ),
        };
        let took = t0.elapsed();
        assert!(
            took < PROMPT,
            "{backend:?}: the trap must not wait for the root's 8 s timeout (took {took:?}, err {err})"
        );
    }
}

/// The #440 × #442 intersection (jacl fiber-timed-wait follow-up, 2026-07-25): teardown must reach
/// a **fiber** parked on an event wait, not just a spawned vCPU. `_start` creates a fiber that
/// parks on an indefinite `memory.wait` (§3.6 5a — the resumer sees `FIBER_PARKED`), then `exit`s
/// with the fiber still parked. The abandoned fiber must not hold the run open: `exit` completes
/// promptly on every engine (the JIT's `FiberWaitCell` cell is dropped with the domain; the
/// bytecode `WaitParked` fiber and the tree-walker's `Waiter::Fiber` drop with their vCPU).
const EXIT_WHILE_FIBER_PARKED: &str = r#"
memory 16
import 0 "exit" (i32) -> ()
export 0 func "_start" 0
func () -> () {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs, vv = cont.resume v2 v3
  v4 = i32.const 0
  call.import 0 (v4)
  unreachable
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 16392
  v6 = i32.const 0
  v7 = i64.const -1
  v8 = i32.atomic.wait v5 v6 v7
  v9 = i64.extend_i32_s v8
  return v9
  }
}
"#;

#[test]
fn exit_tears_down_a_parked_fiber() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let module = temen_text::parse_module(EXIT_WHILE_FIBER_PARKED).expect("parse");
        temen_verify::verify_module(&module).expect("verify");
        let imports = Imports::new().provide("exit", HostCap::exit());
        let instance = instantiate_with_imports(module, imports).expect("instantiate");
        let t0 = Instant::now();
        let run = instance
            .run(backend, &RunConfig::default())
            .unwrap_or_else(|e| panic!("run on {backend:?}: {e}"));
        let took = t0.elapsed();
        assert_eq!(run.outcome, Outcome::Exited(0), "{backend:?}");
        assert!(
            took < PROMPT,
            "{backend:?}: exit must not wait out the parked fiber (took {took:?})"
        );
    }
}
