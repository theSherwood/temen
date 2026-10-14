//! CONSOLIDATION.md §3d — **the config-record spawn (op 17) is native on the bytecode tier**.
//! Every test here runs the same program on the tree-walk oracle (`run_with_host`) and the
//! bytecode engine (`compile_and_run_with_host`), asserting identical results — and the
//! `.expect(...)` on the engine's `Option` is the non-vacuity gate: if the engine *declined* the
//! module (fell back), the test fails. That is what distinguishes these pins from the
//! `temen`-crate record differentials, which run through temen-run's `Backend::Bytecode` — a path
//! that folds every `Instantiator` module to the tree-walker and so never exercises the native
//! `Op::InstantiateRec`.
//!
//! Tier boundaries pinned here:
//! - a **pager-capable** module (op 17 + impl exports) must *decline* to the oracle, which owns
//!   demand paging (`compile_module_for`'s veto — the bytecode twin of temen-run's
//!   `module_demand_spawns` fold on the Cranelift tier);
//! - a **budget** funds the child natively, and a dangling one refuses the spawn probeably. What a
//!   budget bounds is pinned on every driver by `spawn_every_driver.rs`.

#[path = "support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, Host, Trap, Value};
use temen_ir::SpawnRec;
use temen_text::parse_module;

use temen_ir::errno::EINVAL;

/// A record spawn (the v1 record [`rec_with_impl_export`] appends at 17536) plus an **impl export**
/// (a pager-capable module): op 17 present + impl exports ⇒ the whole module must decline to the
/// tree-walk oracle (the record's pager field is runtime data, so admission is by module shape —
/// `compile_module_for`).
const REC_WITH_IMPL_EXPORT: &str = r#"memory 17
type 0 func (i64) -> (i64)
type 1 interface { page: 0 }
export 0 interface "pager" 1 { page: 1 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  qa0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) v0 (qa0)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vch)
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

/// [`REC_WITH_IMPL_EXPORT`] with its record. The module declines by its shape, before the record
/// is read.
fn rec_with_impl_export() -> String {
    format!(
        "{REC_WITH_IMPL_EXPORT}{}",
        rec::segment(17536, &SpawnRec::v1())
    )
}

/// A detached-record parent (`(i32 inst, i64 budget, i32 child) -> (i64)`): the budget and module
/// handles arrive at run time and are stored into the v1 record appended at 17536 (`child`, func 1's
/// child image, #2219, in its declared window). Spawn, join, return the child's 42 — or the spawn's
/// refusal.
fn budget_rec_v1_src() -> String {
    format!(
        "memory 17
func (i32, i64, i32) -> (i64) {{
block 0 (v0: i32, v1: i64, v2: i32) {{
  qma = i64.const 17560
  i32.store qma v2
  qb = i32.wrap_i64 v1
  qba = i64.const 17564
  i32.store qba qb
  qa0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) v0 (qa0)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vch) 2(v0, vch)
}}
block 1 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
}}
block 2 (vi2: i32, vch2: i32) {{
  vr = call.cap 6 1 (i32) -> (i64) vi2 (vch2)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vr = i64.const 42
  return vr
  }}
}}
{rec}",
        rec = rec::segment(17536, &SpawnRec::v1())
    )
}

fn tw(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    run_with_host(&m, 0, args, &mut fuel, host)
}

fn bc(src: &str, args: &[Value], host: &mut Host) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, args, &mut fuel, host)
        .expect("bytecode engine must drive the record spawn natively (§3d)")
}

/// The vertical: a v1 record spawn + join is engine-native (compile succeeds — no oracle fallback)
/// and result-identical to the tree-walker, the child funded by the budget.
#[test]
fn record_spawn_is_native_and_matches_the_oracle() {
    let src = budget_rec_v1_src();
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        bytecode::VcpuProgram::compile(&m).is_some(),
        "op-17 module must compile natively (no impl exports)"
    );
    let image = temen_ir::child_image_at(&m, 1).expect("image");
    let powerbox = || {
        let mut h = Host::new();
        let i = h.grant_instantiator(0, 1 << 17);
        let c = h.grant_module(&image);
        let b = h.grant_budget(5_000_000, -1, -1);
        (h, [Value::I32(i), Value::I64(b as i64), Value::I32(c)])
    };
    let (mut h, args) = powerbox();
    let r_tw = tw(&src, &args, &mut h);
    assert_eq!(r_tw, Ok(vec![Value::I64(42)]), "oracle");
    let (mut h, args) = powerbox();
    let r_bc = bc(&src, &args, &mut h);
    assert_eq!(r_bc, r_tw, "record spawn: bytecode != tree-walker");
}

/// The pager admission veto: an op-17 module WITH impl exports declines whole (the oracle owns
/// demand paging), on both the orchestration entry and the cooperative runner.
#[test]
fn pager_capable_record_module_declines_to_the_oracle() {
    let m = parse_module(&rec_with_impl_export()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
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

/// A dangling budget handle refuses the detached spawn probeably (`-EINVAL`) on both engines,
/// natively: op 15's rule, which a v1 record follows (a non-`Budget` arg refuses). This test pinned
/// `CapFault` while its record spawned `module = -1` from a host that registered no program: the
/// module failed to resolve before the budget was looked at (#2219).
#[test]
fn dangling_budget_handle_refuses_the_detached_spawn_on_both_engines() {
    let src = budget_rec_v1_src();
    let image = temen_ir::child_image_at(&parse_module(&src).expect("parse"), 1).expect("image");
    let mut h_tw = Host::new();
    let i_tw = h_tw.grant_instantiator(0, 1 << 17);
    let c_tw = h_tw.grant_module(&image);
    let args_tw = [Value::I32(i_tw), Value::I64(99), Value::I32(c_tw)];
    let r_tw = tw(&src, &args_tw, &mut h_tw);
    assert_eq!(r_tw, Ok(vec![Value::I64(EINVAL)]), "oracle");

    let mut h_bc = Host::new();
    let i_bc = h_bc.grant_instantiator(0, 1 << 17);
    let c_bc = h_bc.grant_module(&image);
    let r_bc = bc(
        &src,
        &[Value::I32(i_bc), Value::I64(99), Value::I32(c_bc)],
        &mut h_bc,
    );
    assert_eq!(r_bc, r_tw, "dangling budget: bytecode != tree-walker");
}
