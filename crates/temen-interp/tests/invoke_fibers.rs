//! §22 "Concurrency" (renegotiated 2026-07-30) on the **invoke** path — a `Jit.invoke`d unit may
//! host fibers, and the **bytecode** engine's nested invoke must agree with the **tree-walker**
//! oracle (#845 surfaced the gap: `run_invoke` had no fiber arms, so a fiber-hosting unit that the
//! canonical validator admits `CapFault`ed on the bytecode engine while running on the others).
//!
//! Shape under test: the unit `cont.new`s over a **module-0 (program) function** named by raw
//! natural-table slot — the shape `step_vcpu`'s renegotiated arms resolve (a fiber over an
//! *installed* unit function is the engine's documented deferred case, not pinned here) — and runs
//! its fiber to completion within the synchronous invoke. The seam-free half stays pinned too: a
//! `suspend` at the unit's own entry (which would park the invoke) traps on both engines.
//!
//! The blob validator is stubbed inline (the `parallel_jit_miri.rs` pattern) so this stays in the
//! temen-interp suite with no Cranelift dep.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Value};
use temen_text::parse_module;

/// What the fiber body adds to its first yield.
const K: i64 = 777;
/// The invoke arg.
const X: i64 = 5;

/// The unit: `f0(x)` spins a fiber over program function 1 (raw slot), resumes twice (suspend
/// yields `x + K`; the second resume runs it to completion returning `x`), sums the yields.
const FIBER_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vs1, vv1 = cont.resume vk v0
  vs2, vv2 = cont.resume vk v0
  vr = i64.add vv1 vv2
  return vr
  }
}
"#;

/// A unit whose **entry** suspends — parking the synchronous invoke, the seam §22 forbids.
const ROOT_SUSPEND_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vv = suspend v0
  return vv
  }
}
"#;

/// The guest program: func 0 `(jit, code) -> invoke2(code, X)`; func 1 is the fiber body the unit
/// names (suspends `arg + K`, then returns the second resume's arg).
fn guest() -> temen_ir::Module {
    let src = format!(
        r#"memory 16
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  vc64 = i64.extend_i32_u v1
  vx = i64.const {X}
  vr = cap.call 11 1 (i64, i64) -> (i64) v0 (vc64, vx)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vk = i64.const {K}
  vs = i64.add varg vk
  vv = suspend vs
  return vv
  }}
}}
"#
    );
    let m = parse_module(&src).expect("parse guest");
    temen_verify::verify_module(&m).expect("verify guest");
    m
}

/// Stub validator: the blob's first byte selects the fixed unit text (trusted test input, no
/// re-verify) — `0` = the fiber-hosting unit, `1` = the root-suspend unit.
fn stub_validator(
    bytes: &[u8],
    _mem_log2: Option<u8>,
    _symtab: &[u8],
) -> Result<Arc<[temen_ir::Func]>, i64> {
    let src = match bytes.first() {
        Some(1) => ROOT_SUSPEND_UNIT,
        _ => FIBER_UNIT,
    };
    match parse_module(src) {
        Ok(m) => Ok(m.funcs.into()),
        Err(_) => Err(-22),
    }
}

/// Grant a `Jit` domain with the stub validator, compile `blob_tag`'s unit, and hand back the
/// entry args `(jit, code)`.
fn setup(host: &mut Host, blob_tag: u8) -> Vec<Value> {
    let jit = host.grant_jit(Some(16));
    host.set_jit_validator(stub_validator);
    let code = match host.jit_compile(jit, &[blob_tag]) {
        Ok(Ok(c)) => c.handle,
        Ok(Err(e)) => panic!("stub compile refused: {e}"),
        Err(t) => panic!("stub compile trapped: {t:?}"),
    };
    vec![Value::I32(jit), Value::I32(code)]
}

#[test]
fn fiber_hosting_invoke_agrees_across_engines() {
    let m = guest();
    let want = 2 * X + K;

    let mut host = Host::new();
    let args = setup(&mut host, 0);
    let mut fuel = u64::MAX;
    let tw = temen_interp::run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("tree-walker services the fiber-hosting invoke");
    assert_eq!(tw, vec![Value::I64(want)], "tree-walker result");

    let mut host = Host::new();
    let args = setup(&mut host, 0);
    let mut fuel = u64::MAX;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("bytecode supports the module")
        .expect("bytecode services the fiber-hosting invoke (#845 — run_invoke fiber arms)");
    assert_eq!(bc, vec![Value::I64(want)], "bytecode result");
}

/// The seam-free half: a unit whose entry `suspend`s would park the synchronous invoke — both
/// engines must refuse with a trap (not hang, not return).
#[test]
fn root_suspend_in_an_invoked_unit_traps_on_both_engines() {
    let m = guest();

    let mut host = Host::new();
    let args = setup(&mut host, 1);
    let mut fuel = u64::MAX;
    let tw = temen_interp::run_with_host(&m, 0, &args, &mut fuel, &mut host);
    assert!(
        tw.is_err(),
        "tree-walker must trap the root suspend: {tw:?}"
    );

    let mut host = Host::new();
    let args = setup(&mut host, 1);
    let mut fuel = u64::MAX;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("bytecode supports the module");
    assert!(bc.is_err(), "bytecode must trap the root suspend: {bc:?}");
}
