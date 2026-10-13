//! #2305 — what the process's pool of address space ([`temen_interp::RESERVED_POOL`]) cannot hold is
//! refused at its admission, on every engine alike, before any window is built: a run ends with
//! `ThreadFault` before its first instruction, its main vCPU refused, and a detached spawn answers
//! `-EINVAL`, its budget charged nothing. Once the runs that held the pool end, both go through. Its
//! own binary: the pool is the process's.

#[path = "support/detached.rs"]
mod detached;

use detached::{host, interp_on_reserved, jit_on_reserved, module, parent_room_after_refusal};
use temen_interp::{
    maps_pool, Host, RootWindow, Value, MAPS_PER_DOMAIN, MAPS_PER_VCPU, RESERVED_POOL,
};
use temen_jit::JitOutcome;
use temen_run::{instantiate, Backend, Outcome, RunConfig};

const SEVEN: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v = i64.const 7
  return v
  }
}
"#;

/// A detached child with nothing to bind: its spawn is refused for its window or not at all.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 42
  return v
  }
}
"#;

const ENGINES: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// The address space a window of the default reservation holds of the pool.
const WINDOW: u64 = 1 << temen_ir::DEFAULT_RESERVED_LOG2;

/// The reservation of the spawning parents' own windows, small enough to fit what the test leaves.
const PARENT_LOG2: u8 = 24;

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

/// An activation holding `reserved` of the pool and nothing else, as another run of the process
/// would: `None` when the pool has no room for it.
fn hold(reserved: u64) -> Option<Host> {
    let mut h = Host::new();
    let window = RootWindow {
        reserved,
        ..RootWindow::default()
    };
    h.begin_activation(0, window)?;
    Some(h)
}

/// The detached spawn of [`CHILD`] on `backend` by a parent whose window reserves `2^PARENT_LOG2`
/// bytes: the room left in the funding budget when the spawn is refused, else `-1`.
fn spawn_detached(backend: Backend) -> i64 {
    let (p, c) = (module(&parent_room_after_refusal(0, 0)), module(CHILD));
    let (mut host, h) = host(&c, 1 << 20);
    match backend {
        Backend::TreeWalk => match interp_on_reserved(&p, &mut host, h, PARENT_LOG2).as_deref() {
            Ok([Value::I64(x)]) => *x,
            r => panic!("the parent ends abnormally on the interpreter: {r:?}"),
        },
        _ => match jit_on_reserved(&p, &mut host, h, PARENT_LOG2) {
            JitOutcome::Returned(v) => v[0],
            o => panic!("the parent ends abnormally on the JIT: {o:?}"),
        },
    }
}

#[test]
fn what_the_pool_cannot_hold_is_refused_at_its_admission_on_every_engine() {
    let slots = RESERVED_POOL / WINDOW;
    if RESERVED_POOL == u64::MAX || maps_pool() < 2 * slots * (MAPS_PER_DOMAIN + MAPS_PER_VCPU) {
        eprintln!("skipped: no address-space pool, or a pool of mappings too small to fill it");
        return;
    }
    // Runs that hold the pool: a window's worth at a time until it is full, then one handed back and
    // all of it but a parent's window held again.
    let mut roots: Vec<Host> = std::iter::from_fn(|| hold(WINDOW)).collect();
    assert_eq!(roots.len() as u64, slots, "the pool holds its windows");
    roots.pop();
    roots.push(hold(WINDOW - (1 << PARENT_LOG2)).expect("the window handed back"));
    let seven = temen_text::parse_module(SEVEN).expect("parse");
    for e in ENGINES {
        let ended = run(&seven, e).expect_err("no room");
        assert!(
            ended.contains("ThreadFault"),
            "{e:?} ends a run the pool cannot hold with ThreadFault: {ended}"
        );
    }
    for e in [Backend::TreeWalk, Backend::Jit] {
        assert_eq!(
            spawn_detached(e),
            1 << 20,
            "{e:?} refuses a detached spawn the pool cannot hold, its budget charged nothing"
        );
    }
    drop(roots);
    for e in ENGINES {
        assert_eq!(run(&seven, e), Ok(vec![Value::I64(7)]), "{e:?} runs again");
    }
    for e in [Backend::TreeWalk, Backend::Jit] {
        assert_eq!(spawn_detached(e), -1, "{e:?} spawns again");
    }
}
