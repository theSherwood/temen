//! A §22 unit's `Instantiator`, by route (#1726, #1578) — on the tree-walk oracle, the bytecode engine
//! and Cranelift alike.
//!
//! DESIGN §22 gives a unit two ways in. **Install** + `call.dyn` runs it in the caller's own frames,
//! where a spawn is an ordinary *module-aware* spawn: a same-module child runs the spawning frame's
//! module — the unit's function, not module 0's — exactly as `thread.spawn` does
//! (`bytecode_parallel_jit.rs::installed_unit_spawns_its_own_module`). **Invoke** runs it as a
//! seam-free leaf, where the whole `Instantiator` is unavailable: a child would outlive the synchronous
//! call over code nothing keeps (an invoked unit is never installed), and no one could join it.
//!
//! The base program *also* has a func 1, returning 7, while the unit's returns 42 — so a child built
//! from the wrong module still runs and answers 7, instead of hiding behind the same trap as "no such
//! function".
//!
//! Before #1726 the install route failed on all three engines, each differently (oracle `Malformed`,
//! bytecode `ThreadFault`, Cranelift `CapFault`); before #1578 the invoke route spawned on the oracle
//! and Cranelift and `CapFault`ed on bytecode.

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::{JitOutcome, TrapKind};
use temen_run::{grant_jit, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// The decoy every guest carries as func 1 — what a child built from module 0 would run.
const DECOY: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 7
  return v
  }
}
"#;

/// Install the unit and `call.dyn` it with the `Instantiator` handle.
const INSTALL: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vinst: i32) {
  vc = i64.extend_i32_u vcode
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vc)
  vs32 = i32.wrap_i64 vslot
  vh = i64.extend_i32_u vinst
  vr = call.dyn (i64) -> (i64) vs32 (vh)
  return vr
  }
}
"#;

/// `Jit.invoke` the unit with the `Instantiator` handle.
const INVOKE: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (vjit: i32, vcode: i32, vinst: i32) {
  vc = i64.extend_i32_u vcode
  vh = i64.extend_i32_u vinst
  vr = call.cap 11 1 (i64, i64) -> (i64) vjit (vc, vh)
  return vr
  }
}
"#;

/// The unit: instantiate a same-module child at its own func 1 (4 KiB carve at 64 KiB), join it,
/// and return the child's result.
const UNIT: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (vi64: i64) {
  vi = i32.wrap_i64 vi64
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vi (ve, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  return vj
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 42
  return v
  }
}
"#;

/// What a run came to, in one vocabulary across the three engines.
#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(i64),
    CapFault,
    Other(String),
}

fn guest(route: &str) -> temen_ir::Module {
    let m = parse_module(&format!("{route}{DECOY}")).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// A fresh host with the `Jit` and an `Instantiator` over the whole window, and the unit compiled
/// into it — deterministic, so every engine sees the same handles `[jit, code, instantiator]`.
fn setup(guest: &temen_ir::Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let jit = grant_jit(&mut host, guest, 4);
    let inst = host.grant_instantiator(0, 128 << 10);
    let unit = parse_module(UNIT).expect("parse unit");
    verify_module(&unit).expect("verify unit");
    let code = host
        .jit_compile(jit, &temen_encode::encode_module(&unit))
        .expect("no trap")
        .expect("compile ok")
        .handle;
    (host, [jit, code, inst])
}

fn interp(r: Result<Vec<Value>, Trap>) -> Outcome {
    match r.as_deref() {
        Ok([Value::I64(x)]) => Outcome::Returned(*x),
        Err(Trap::CapFault) => Outcome::CapFault,
        other => Outcome::Other(format!("{other:?}")),
    }
}

fn tree_walker(route: &str) -> Outcome {
    let m = guest(route);
    let (mut host, h) = setup(&m);
    let args: Vec<Value> = h.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        DEFAULT_RESERVED_LOG2,
        &mut host,
    );
    interp(res)
}

fn bytecode_engine(route: &str) -> Outcome {
    let m = guest(route);
    let (mut host, h) = setup(&m);
    let args: Vec<Value> = h.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let (res, _) = bytecode::compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        DEFAULT_RESERVED_LOG2,
        &mut host,
    )
    .expect("the bytecode engine accepts the guest");
    interp(res)
}

fn cranelift(route: &str) -> Outcome {
    let m = guest(route);
    let (mut host, h) = setup(&m);
    let slots: Vec<i64> = h.iter().map(|&x| x as i64).collect();
    let (out, _) =
        jit_cap_run(&m, 0, &slots, &[], DEFAULT_RESERVED_LOG2, 4, &mut host).expect("jit run");
    match out {
        JitOutcome::Returned(v) if v.len() == 1 => Outcome::Returned(v[0]),
        JitOutcome::Trapped(TrapKind::CapFault) => Outcome::CapFault,
        other => Outcome::Other(format!("{other:?}")),
    }
}

/// #1726 — an installed unit's same-module child runs the unit's function, and joins.
#[test]
fn an_installed_unit_spawns_its_own_function() {
    assert_eq!(tree_walker(INSTALL), Outcome::Returned(42), "tree-walker");
    assert_eq!(bytecode_engine(INSTALL), Outcome::Returned(42), "bytecode");
    assert_eq!(cranelift(INSTALL), Outcome::Returned(42), "cranelift");
}

/// #1578 — the same unit, invoked, reaches no `Instantiator` at all: a `CapFault` at the spawn.
#[test]
fn an_invoked_unit_has_no_instantiator() {
    assert_eq!(tree_walker(INVOKE), Outcome::CapFault, "tree-walker");
    assert_eq!(bytecode_engine(INVOKE), Outcome::CapFault, "bytecode");
    assert_eq!(cranelift(INVOKE), Outcome::CapFault, "cranelift");
}
