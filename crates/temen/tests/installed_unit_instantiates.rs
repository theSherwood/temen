//! #1726 — an **installed** §22 unit's `Instantiator.instantiate` (op 0) is a module-aware spawn.
//!
//! DESIGN §22's supported route for a unit to spawn is `install` + `call.dyn`: installed code runs
//! in the caller's own frames, so a spawn from it is an ordinary spawn that resolves its entry in
//! **the spawning frame's module** — the unit — exactly as `thread.spawn` does
//! (`bytecode_parallel_jit.rs::installed_unit_spawns_its_own_module`). A same-module child of an
//! installed unit therefore runs the unit's function, and the caller joins it like any child.
//!
//! The base program here *also* has a func 1, returning 7, while the unit's func 1 returns 42. A
//! child resolved against the wrong module still runs — and answers 7 — so the test tells "wrong
//! module" apart from "no such function" rather than seeing both as a trap.
//!
//! Before the fix all three engines failed, each differently: the tree-walk oracle validated the
//! entry against the unit but built the child from module 0, the bytecode engine validated it
//! against module 0, and Cranelift spawned but could not join.

use temen_interp::{bytecode, run_capture_reserved_with_host, Host, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::JitOutcome;
use temen_run::{grant_jit, jit_cap_run};
use temen_text::parse_module;
use temen_verify::verify_module;

/// The caller: install the unit, `call.dyn` its entry with the `Instantiator` handle, return what
/// it returns. Func 1 is the decoy a wrong-module child would run.
const GUEST: &str = r#"memory 17
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
func (i64) -> (i64) {
block 0 (v0: i64) {
  v = i64.const 7
  return v
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

const WANT: i64 = 42;

/// A fresh host with the `Jit` and an `Instantiator` over the whole window, and the unit compiled
/// into it — deterministic, so every engine sees the same handles.
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

fn guest() -> temen_ir::Module {
    let m = parse_module(GUEST).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

fn values(h: [i32; 3]) -> Vec<Value> {
    h.iter().map(|&x| Value::I32(x)).collect()
}

#[test]
fn tree_walker() {
    let m = guest();
    let (mut host, h) = setup(&m);
    let mut fuel = 10_000_000u64;
    let (res, _) = run_capture_reserved_with_host(
        &m,
        0,
        &values(h),
        &mut fuel,
        &[],
        DEFAULT_RESERVED_LOG2,
        &mut host,
    );
    assert_eq!(res, Ok(vec![Value::I64(WANT)]));
}

#[test]
fn bytecode_engine() {
    let m = guest();
    let (mut host, h) = setup(&m);
    let mut fuel = 10_000_000u64;
    let (res, _) = bytecode::compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &values(h),
        &mut fuel,
        &[],
        DEFAULT_RESERVED_LOG2,
        &mut host,
    )
    .expect("the bytecode engine accepts the guest");
    assert_eq!(res, Ok(vec![Value::I64(WANT)]));
}

#[test]
fn cranelift() {
    let m = guest();
    let (mut host, h) = setup(&m);
    let slots: Vec<i64> = h.iter().map(|&x| x as i64).collect();
    let (out, _) =
        jit_cap_run(&m, 0, &slots, &[], DEFAULT_RESERVED_LOG2, 4, &mut host).expect("jit run");
    assert!(
        matches!(out, JitOutcome::Returned(ref v) if v == &[WANT]),
        "cranelift: {out:?}"
    );
}
