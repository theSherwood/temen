//! Dynamic **float** rename cells (mutable-VM-state rename, follow-on). A `TValue`'s 8-byte value is
//! often moved with `f64` load/store, so once a register holds a *dynamic* value (e.g. a rolled
//! loop's accumulator) the renamer must handle a dynamic float cell. It does so by reinterpreting the
//! float's bits to a same-width integer on store, and back on load — so the cell stays uniformly
//! integer-typed and composes with integer loads (type-punning, exactly as Lua's value union does).
//! These tests pin the round-trip and both cross-type puns against the interpreter oracle.

use temen_interp::{Trap, Value};
use temen_ir::Module;
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

const AT: u64 = 16512; // above the #1094 NULL guard

fn cfg() -> SpecConfig {
    SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        ..SpecConfig::default()
    }
}
fn run(m: &Module, a: Value) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    temen_interp::run(m, 0, &[a], &mut fuel)
}
fn no_mem_ops(m: &Module) -> bool {
    !m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .any(|i| {
            matches!(
                i,
                temen_ir::Inst::Load { .. } | temen_ir::Inst::Store { .. }
            )
        })
}
fn residual(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg()).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert!(
        no_mem_ops(&r),
        "the dynamic float cell must be SSA-lifted (no residual load/store)"
    );
    r
}

#[test]
fn dynamic_f64_round_trips_through_a_rename_cell() {
    // f(x) = { *S = x; return *S } — a dynamic f64 stored then read back must be the identity.
    let src = "\
memory 16
func (f64) -> (f64) {
block 0 (v0: f64) {
  va = i64.const 16512
  f64.store va v0
  vr = f64.load va
  return vr
}
}
";
    let m = parse_module(src).unwrap();
    let r = residual(src);
    for x in [0.0f64, 1.5, -3.25, 1e300, f64::MIN_POSITIVE] {
        let v = Value::F64(x);
        assert_eq!(run(&r, v), run(&m, v), "diverged at x={x}");
        assert_eq!(run(&r, v), Ok(vec![Value::F64(x)]));
    }
}

#[test]
fn f64_store_then_i64_load_reinterprets_the_bits() {
    // f(x) = { *S = (f64)x; return (i64)*S } — the value union punned f64→i64, as Lua reads a TValue.
    let src = "\
memory 16
func (f64) -> (i64) {
block 0 (v0: f64) {
  va = i64.const 16512
  f64.store va v0
  vr = i64.load va
  return vr
}
}
";
    let m = parse_module(src).unwrap();
    let r = residual(src);
    for x in [0.0f64, 1.5, -3.25, 42.0, 1e-9] {
        let v = Value::F64(x);
        assert_eq!(run(&r, v), run(&m, v), "diverged at x={x}");
        assert_eq!(run(&r, v), Ok(vec![Value::I64(x.to_bits() as i64)]));
    }
}

#[test]
fn i64_store_then_f64_load_reinterprets_the_bits() {
    // The reverse pun: an i64 written, read back as the f64 with those bits.
    let src = "\
memory 16
func (i64) -> (f64) {
block 0 (v0: i64) {
  va = i64.const 16512
  i64.store va v0
  vr = f64.load va
  return vr
}
}
";
    let m = parse_module(src).unwrap();
    let r = residual(src);
    // Compare by *bits*, not `Value` equality — some inputs are NaN, and NaN != NaN under `==`.
    let bits_of = |r: Result<Vec<Value>, Trap>| match r {
        Ok(v) => match v[..] {
            [Value::F64(f)] => f.to_bits(),
            _ => panic!("unexpected result {v:?}"),
        },
        e => panic!("trapped: {e:?}"),
    };
    for bits in [0i64, 1, 0x4045_0000_0000_0000, -1, 0x7ff0_0000_0000_0000] {
        let res = bits_of(run(&r, Value::I64(bits)));
        let interp = bits_of(run(&m, Value::I64(bits)));
        assert_eq!(res, interp, "residual vs interp diverged at bits={bits:#x}");
        assert_eq!(
            res, bits as u64,
            "reinterpret must preserve the bit pattern (bits={bits:#x})"
        );
    }
}
