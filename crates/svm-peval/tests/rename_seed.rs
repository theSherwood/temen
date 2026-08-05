//! Slice A of the mutable-VM-state rename feature: **seed rename cells from the constant image**.
//!
//! The plain `rename` region is zero-init scratch — an untouched renamed cell reads 0. That is wrong
//! for a region that aliases *real* captured memory (a `lua_State`, a `CallInfo`): its fields must
//! fold to their captured values, not zero, for the guards they drive to fold. With
//! `rename_seed_from_image` the untouched cell instead reads its seed from the constant-memory sources
//! (here a writable data segment declared constant via const_regions), while writes still shadow it. These tests pin both against the
//! interpreter oracle, and confirm the region is still fully SSA-lifted (no residual load/store).

use svm_interp::{Trap, Value};
use svm_ir::{Data, Module};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use svm_text::parse_module;
use svm_verify::verify_module;

fn run(m: &Module, x: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, &[Value::I64(x)], &mut fuel)
}

/// Attach a **writable** data segment holding `val` (little-endian i64) at window address `at`. It is
/// writable (not RO-protected) because a live VM-state region is mutated in place; the bytes are
/// declared constant *at specialization time* via `const_regions`, which is the actual seed source.
fn with_seed(mut m: Module, at: u64, val: i64) -> Module {
    m.data.push(Data {
        offset: at,
        readonly: false,
        bytes: val.to_le_bytes().to_vec(),
    });
    m
}

fn has_mem_op(m: &Module) -> bool {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .any(|i| matches!(i, svm_ir::Inst::Load { .. } | svm_ir::Inst::Store { .. }))
}

const SEED: i64 = 0x1234_5678;
const AT: u64 = 128;

#[test]
fn untouched_seeded_cell_reads_the_image_not_zero() {
    // f(x) = *S + x, where S is a renamed cell seeded from the constant image. Renaming it must fold to the
    // seed (0x12345678), matching the interpreter — not to the scratch zero.
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  vs = i64.load va
  vr = i64.add vs v0
  return vr
}
}
";
    let m = with_seed(parse_module(src).expect("parse"), AT, SEED);
    verify_module(&m).expect("source verifies");

    let cfg = SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_regions: vec![(AT, AT + 8)],
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert!(
        !has_mem_op(&residual),
        "the seeded cell should be SSA-lifted, leaving no residual load/store"
    );

    for x in [0i64, 1, -9, 1000, i64::MAX - SEED] {
        assert_eq!(
            run(&residual, x),
            run(&m, x),
            "diverged from interpreter at x={x}"
        );
        assert_eq!(
            run(&residual, x),
            Ok(vec![Value::I64(x.wrapping_add(SEED))])
        );
    }

    // Without the flag the region is zero-init scratch: the residual would read 0, *not* the seed —
    // i.e. it would diverge from the interpreter. This is exactly the unsoundness the flag fixes.
    let scratch = specialize_with_config(
        &m,
        0,
        &[SpecArg::Dynamic],
        &SpecConfig {
            rename: Some((AT, AT + 8)),
            rename_is_private: true,
            ..SpecConfig::default()
        },
    )
    .expect("specializes");
    assert_eq!(
        run(&scratch, 5),
        Ok(vec![Value::I64(5)]),
        "zero-init scratch reads 0, so f(5) = 5 — the wrong answer the seed fixes"
    );
}

#[test]
fn a_write_shadows_the_seed() {
    // f(x) = { *S = 999; return *S + x }. The store shadows the seed, so the read-back is 999, not
    // the seed — matching the interpreter. (The region is only observed via its return here, so no
    // write-back is needed yet; that is Slice B.)
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  vw = i64.const 999
  i64.store va vw
  vs = i64.load va
  vr = i64.add vs v0
  return vr
}
}
";
    let m = with_seed(parse_module(src).expect("parse"), AT, SEED);
    verify_module(&m).expect("source verifies");

    let cfg = SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_regions: vec![(AT, AT + 8)],
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert!(
        !has_mem_op(&residual),
        "write+read of the cell should fully fold"
    );

    for x in [0i64, 3, -1, 12345] {
        assert_eq!(run(&residual, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&residual, x), Ok(vec![Value::I64(x.wrapping_add(999))]));
    }
}
