//! CONSOLIDATION.md §3d — **the config-record spawn (op 17) is native on the bytecode tier**.
//! Every test here runs the same program on the tree-walk oracle (`run_with_host`) and the
//! bytecode engine (`compile_and_run_with_host`), asserting identical results — and the
//! `.expect(...)` on the engine's `Option` is the non-vacuity gate: if the engine *declined* the
//! module (fell back), the test fails. That is what distinguishes these pins from the
//! `svm`-crate record differentials, which run through svm-run's `Backend::Bytecode` — a path
//! that folds every `Instantiator` module to the tree-walker and so never exercises the native
//! `Op::InstantiateRec`.
//!
//! Tier boundaries pinned here:
//! - a **pager-capable** module (op 17 + impl exports) must *decline* to the oracle, which owns
//!   demand paging (`compile_module_for`'s veto — the bytecode twin of svm-run's
//!   `module_demand_spawns` fold on the Cranelift tier);
//! - a **budget** record funds natively (peek-then-drain at the driver's commit site), refusals
//!   leave the budget intact, and a *bounded spawn ceiling* is this tier's narrowed gap —
//!   `-EINVAL`, exactly the Cranelift thunk's — pinned divergent, to flip when child vCPU
//!   quotas land here.

use svm_interp::{bytecode, run_with_host, Host, Trap, Value};
use svm_text::parse_module;

use svm_ir::errno::EINVAL;

/// Parent (entry 0, `(i32 inst) -> (i64)`): build the record at 1152 — entry 1, off 64 KiB,
/// size_log2 12, no pager/module/budget/quota/grants — spawn via op 17, join, return the child's
/// result. Child (func 1) returns 42.
const REC_SPAWN: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  qv0 = i64.const 4294967296
  qv1 = i64.const 65536
  qv2 = i64.const -4294967284
  qv3 = i64.const 4294967295
  qv4 = i64.const 0
  qa0 = i64.const 1152
  i64.store qa0 qv0
  qa1 = i64.const 1160
  i64.store qa1 qv1
  qa2 = i64.const 1168
  i64.store qa2 qv2
  qa3 = i64.const 1176
  i64.store qa3 qv3
  qa4 = i64.const 1184
  i64.store qa4 qv4
  qa5 = i64.const 1192
  i64.store qa5 qv4
  qa6 = i64.const 1200
  i64.store qa6 qv4
  vch = cap.call 6 17 (i64) -> (i32) v0 (qa0)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vch)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
"#;

/// `REC_SPAWN`'s shape plus an **impl export** (a pager-capable module): op 17 present + impl
/// exports ⇒ the whole module must decline to the tree-walk oracle (the record's pager field is
/// runtime data, so admission is by module shape — `compile_module_for`).
const REC_WITH_IMPL_EXPORT: &str = r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface "pager" 1 { page: 1 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  qv0 = i64.const 4294967296
  qv1 = i64.const 65536
  qv2 = i64.const -4294967284
  qv3 = i64.const 4294967295
  qv4 = i64.const 0
  qa0 = i64.const 1152
  i64.store qa0 qv0
  qa1 = i64.const 1160
  i64.store qa1 qv1
  qa2 = i64.const 1168
  i64.store qa2 qv2
  qa3 = i64.const 1176
  i64.store qa3 qv3
  qa4 = i64.const 1184
  i64.store qa4 qv4
  qa5 = i64.const 1192
  i64.store qa5 qv4
  qa6 = i64.const 1200
  i64.store qa6 qv4
  vch = cap.call 6 17 (i64) -> (i32) v0 (qa0)
  vr = cap.call 6 1 (i32) -> (i64) v0 (vch)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 42
  return vr
  }
}
"#;

/// Budget-record parent (`(i32 inst, i64 budget) -> (i64)`): the budget handle arrives at run
/// time, so f24 is packed with `or(module=-1, budget<<32)`; quota stays 0 (mixing is a
/// `CapFault`). Spawn, join, return. Child (func 1) returns 42.
fn budget_rec_src(size_log2: u64, off: u64) -> String {
    let f0: i64 = 1 << 32; // version 0 | entry 1
    let f16: i64 = (size_log2 | (0xFFFF_FFFFu64 << 32)) as i64; // sl | pager MAX
    format!(
        "memory 17
func (i32, i64) -> (i64) {{
block 0 (v0: i32, v1: i64) {{
  ; spawn via record (op 17) funded by the budget handle in v1
  qv0 = i64.const {f0}
  qv1 = i64.const {off}
  qv2 = i64.const {f16}
  qsh = i64.const 32
  qbh = i64.shl v1 qsh
  qm1 = i64.const 4294967295
  qv3 = i64.or qm1 qbh
  qv4 = i64.const 0
  qa0 = i64.const 1152
  i64.store qa0 qv0
  qa1 = i64.const 1160
  i64.store qa1 qv1
  qa2 = i64.const 1168
  i64.store qa2 qv2
  qa3 = i64.const 1176
  i64.store qa3 qv3
  qa4 = i64.const 1184
  i64.store qa4 qv4
  qa5 = i64.const 1192
  i64.store qa5 qv4
  qa6 = i64.const 1200
  i64.store qa6 qv4
  vch = cap.call 6 17 (i64) -> (i32) v0 (qa0)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(v0, vch)
}}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
}}
block 2 (vi2: i32, vch2: i32) {{
  vr = cap.call 6 1 (i32) -> (i64) vi2 (vch2)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
"
    )
}

fn tw(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    run_with_host(&m, 0, args, &mut fuel, host)
}

fn bc(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, args, &mut fuel, host)
        .expect("bytecode engine must drive the record spawn natively (§3d)")
}

/// The vertical: record spawn + join is engine-native (compile succeeds — no oracle fallback)
/// and result-identical to the tree-walker.
#[test]
fn record_spawn_is_native_and_matches_the_oracle() {
    let m = parse_module(REC_SPAWN).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    assert!(
        bytecode::VcpuProgram::compile(&m).is_some(),
        "op-17 module must compile natively (no impl exports)"
    );

    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let r_tw = tw(REC_SPAWN, &[Value::I32(i_tw)], &mut h_tw);
    assert_eq!(r_tw, Ok(vec![Value::I64(42)]), "oracle");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let r_bc = bc(REC_SPAWN, &[Value::I32(i_bc)], &mut h_bc);
    assert_eq!(r_bc, r_tw, "record spawn: bytecode != tree-walker");
}

/// The pager admission veto: an op-17 module WITH impl exports declines whole (the oracle owns
/// demand paging), on both the orchestration entry and the cooperative runner.
#[test]
fn pager_capable_record_module_declines_to_the_oracle() {
    let m = parse_module(REC_WITH_IMPL_EXPORT).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    assert!(
        bytecode::VcpuProgram::compile(&m).is_none(),
        "op 17 + impl exports must decline (pager records are runtime data)"
    );
    let mut h = Host::new();
    let i = h.grant_instantiator(0, 1 << 17);
    let mut fuel = 10_000_000u64;
    assert!(
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(i)], &mut fuel, &mut h).is_none(),
        "cooperative entry must decline the pager-capable module too"
    );
}

/// A budget record funds the child natively: bounded fuel, unbounded mem/spawn → 42 on both
/// engines (and natively — the compile gate in `bc`).
#[test]
fn budget_record_funds_the_child_natively() {
    let src = budget_rec_src(12, 65536);
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let b_tw = h_tw.grant_budget(5_000_000, -1, -1);
    let r_tw = tw(
        &src,
        &[Value::I32(i_tw), Value::I64(b_tw as i64)],
        &mut h_tw,
    );
    assert_eq!(r_tw, Ok(vec![Value::I64(42)]), "oracle");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let b_bc = h_bc.grant_budget(5_000_000, -1, -1);
    let r_bc = bc(
        &src,
        &[Value::I32(i_bc), Value::I64(b_bc as i64)],
        &mut h_bc,
    );
    assert_eq!(r_bc, r_tw, "funded record spawn: bytecode != tree-walker");
}

/// A budget whose mem quota is short of the carve refuses the spawn `-EINVAL` on both engines —
/// and the drain never happened (peek-then-drain), which the guest can't observe here but the
/// engines must agree on the refusal path bit-for-bit.
#[test]
fn budget_mem_refusal_is_einval_on_both_engines() {
    let src = budget_rec_src(12, 65536); // carve 4 KiB > mem quota 1 KiB below
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let b_tw = h_tw.grant_budget(5_000_000, 1024, -1);
    let r_tw = tw(
        &src,
        &[Value::I32(i_tw), Value::I64(b_tw as i64)],
        &mut h_tw,
    );
    assert_eq!(r_tw, Ok(vec![Value::I64(EINVAL)]), "oracle refuses -EINVAL");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let b_bc = h_bc.grant_budget(5_000_000, 1024, -1);
    let r_bc = bc(
        &src,
        &[Value::I32(i_bc), Value::I64(b_bc as i64)],
        &mut h_bc,
    );
    assert_eq!(r_bc, r_tw, "mem refusal: bytecode != tree-walker");
}

/// A dangling budget handle fails the spawn closed (`CapFault`) on both engines — the exec arm's
/// peek is the bytecode twin of the tree-walker's parse-time peek.
#[test]
fn dangling_budget_handle_capfaults_on_both_engines() {
    let src = budget_rec_src(12, 65536);
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let r_tw = tw(&src, &[Value::I32(i_tw), Value::I64(99)], &mut h_tw);
    assert_eq!(r_tw, Err(Trap::CapFault), "oracle");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let m = parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    let r_bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(i_bc), Value::I64(99)],
        &mut fuel,
        &mut h_bc,
    )
    .expect("native");
    assert_eq!(r_bc, r_tw, "dangling budget: bytecode != tree-walker");
}

/// **The narrowed gap, pinned divergent** (flip when child vCPU quotas land on this tier): a
/// budget with a *bounded spawn ceiling* funds on the tree-walker (which tightens the child's
/// `Quota.max_vcpus`) but refuses `-EINVAL` on the bytecode engine — exactly the Cranelift
/// thunk's narrowed gap (`instantiate_record.rs::spawn_bounded_budget_record_is_the_narrowed_jit_gap`).
#[test]
fn bounded_spawn_budget_is_the_narrowed_bytecode_gap() {
    let src = budget_rec_src(12, 65536);
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let b_tw = h_tw.grant_budget(5_000_000, -1, 3);
    let r_tw = tw(
        &src,
        &[Value::I32(i_tw), Value::I64(b_tw as i64)],
        &mut h_tw,
    );
    assert_eq!(
        r_tw,
        Ok(vec![Value::I64(42)]),
        "oracle funds bounded-spawn budgets"
    );

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let b_bc = h_bc.grant_budget(5_000_000, -1, 3);
    let r_bc = bc(
        &src,
        &[Value::I32(i_bc), Value::I64(b_bc as i64)],
        &mut h_bc,
    );
    assert_eq!(
        r_bc,
        Ok(vec![Value::I64(EINVAL)]),
        "bytecode narrowed gap — flip this to 42 when vCPU quotas land on this tier"
    );
}
