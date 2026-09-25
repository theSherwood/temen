//! #1729 — every `Instantiator` op checks its `Instantiator` operand, on every engine.
//!
//! The child ops — `join` (1), `poll` (9), `detach` (10), `kill` (12), `child_offer` (14) — take a
//! child handle, and the child table is the caller's own, so skipping the check reaches nothing the
//! caller could not reach anyway. It is still the authority the op is made against: the oracle
//! resolves the `Instantiator` before it looks at the op, and a revoked or forged one is a
//! `CapFault` there. Cranelift passed only the child handle to these thunks, so it joined, polled,
//! detached, killed and minted through any handle at all.

use temen_interp::{bytecode, Host, MemLayout, Trap, Value};
use temen_ir::DEFAULT_RESERVED_LOG2;
use temen_jit::{JitOutcome, TrapKind};
use temen_run::jit_cap_run;
use temen_text::parse_module;
use temen_verify::verify_module;

/// A handle the guest was never granted.
const FORGED: i32 = 9999;

/// Spawn func 1 into a 4 KiB carve at 64 KiB through the real `Instantiator`, then apply `op` to the
/// live child through `FORGED`.
fn guest(op: u32) -> temen_ir::Module {
    let call = match op {
        1 => "  vr = call.cap 6 1 (i32) -> (i64) vf (vh)".to_string(),
        14 => "  vx = i64.const 0\n  vo = call.cap 6 14 (i32, i64) -> (i32) vf (vh, vx)\n  vr = i64.extend_i32_s vo".to_string(),
        _ => format!("  vs = call.cap 6 {op} (i32) -> (i32) vf (vh)\n  vr = i64.extend_i32_s vs"),
    };
    let src = format!(
        r#"memory 17
func (i32) -> (i64) {{
block 0 (vi: i32) {{
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vi (ve, voff, vsl, vq)
  vf = i32.const {FORGED}
{call}
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v = i64.const 42
  return v
  }}
}}
"#
    );
    let m = parse_module(&src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

fn host() -> (Host, i32) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    (host, inst)
}

fn oracle(m: &temen_ir::Module) -> Result<Vec<Value>, Trap> {
    let (mut host, inst) = host();
    let mut fuel = 10_000_000u64;
    temen_interp::run_with_host(m, 0, &[Value::I32(inst)], &mut fuel, &mut host)
}

/// `None` when the bytecode engine declines the module (it has no lowering for ops 9/10/12).
fn bytecode_engine(m: &temen_ir::Module) -> Option<Result<Vec<Value>, Trap>> {
    let (mut host, inst) = host();
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(m, 0, &[Value::I32(inst)], &mut fuel, &mut host)
}

fn cranelift(m: &temen_ir::Module) -> JitOutcome {
    let (mut host, inst) = host();
    jit_cap_run(
        m,
        0,
        &[inst as i64],
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
    )
    .expect("jit run")
    .0
}

#[test]
fn a_forged_instantiator_is_a_cap_fault_on_every_child_op() {
    for op in [1, 9, 10, 12, 14] {
        let m = guest(op);
        assert_eq!(oracle(&m), Err(Trap::CapFault), "op {op}: oracle");
        if let Some(bc) = bytecode_engine(&m) {
            assert_eq!(bc, Err(Trap::CapFault), "op {op}: bytecode");
        }
        let jit = cranelift(&m);
        assert!(
            matches!(jit, JitOutcome::Trapped(TrapKind::CapFault)),
            "op {op}: cranelift gave {jit:?}"
        );
    }
}
