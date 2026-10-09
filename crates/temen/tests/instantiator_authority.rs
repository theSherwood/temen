//! #1729 — every `Instantiator` op checks its `Instantiator` operand, on every engine.
//!
//! The child ops — `join` (1), `poll` (9), `detach` (10), `kill` (12), `child_offer` (14) — take a
//! child handle, and the child table is the caller's own, so skipping the check reaches nothing the
//! caller could not reach anyway. It is still the authority the op is made against: the oracle
//! resolves the `Instantiator` before it looks at the op, and a revoked or forged one is a
//! `CapFault` there. Cranelift passed only the child handle to these thunks, so it joined, polled,
//! detached, killed and minted through any handle at all.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, Host, MemLayout, Trap, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitOutcome, TrapKind};
use temen_run::jit_cap_run;
use temen_text::parse_module;
use temen_verify::verify_module;

/// A handle the guest was never granted.
const FORGED: i32 = 9999;

/// Spawn the child (func 1's child image, #2219) detached through the real `Instantiator`, paid from
/// the root's `Budget` (a v1 record at 17408), then apply `op` to the live child through `FORGED`.
fn guest(op: u32) -> temen_ir::Module {
    let call = match op {
        1 => "  vr = call.cap 6 1 (i32) -> (i64) vf (vh)".to_string(),
        14 => "  vx = i64.const 0\n  vo = call.cap 6 14 (i32, i64) -> (i32) vf (vh, vx)\n  vr = i64.extend_i32_s vo".to_string(),
        _ => format!("  vs = call.cap 6 {op} (i32) -> (i32) vf (vh)\n  vr = i64.extend_i32_s vs"),
    };
    let src = format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vb: i32, vm: i32) {{
  vam = i64.const 17432
  i32.store vam vm
  vab = i64.const 17436
  i32.store vab vb
  vp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vi (vp)
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
{rec}"#,
        rec = rec::segment(17408, &SpawnRec::v1(0))
    );
    let m = parse_module(&src).expect("parse guest");
    verify_module(&m).expect("verify guest");
    m
}

/// The entry's arguments: the `Instantiator`, the `Budget` that pays for the child, and the child.
fn host(m: &temen_ir::Module) -> (Host, [i32; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let child = host.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    (host, [inst, budget, child])
}

fn oracle(m: &temen_ir::Module) -> Result<Vec<Value>, Trap> {
    let (mut host, args) = host(m);
    let mut fuel = 10_000_000u64;
    temen_interp::run_with_host(m, 0, &args.map(Value::I32), &mut fuel, &mut host)
}

/// `None` when the bytecode engine declines the module (it has no lowering for ops 9/10/12).
fn bytecode_engine(m: &temen_ir::Module) -> Option<Result<Vec<Value>, Trap>> {
    let (mut host, args) = host(m);
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(m, 0, &args.map(Value::I32), &mut fuel, &mut host)
}

fn cranelift(m: &temen_ir::Module) -> JitOutcome {
    let (mut host, args) = host(m);
    jit_cap_run(
        m,
        0,
        &args.map(i64::from),
        &MemLayout::image(Vec::new()),
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
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
