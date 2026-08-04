//! Approach-A differential spec: bounded-target lowering of a dynamic-index `call_indirect`.
//!
//! `lower_indirect_dispatch` must (1) re-verify, (2) be behaviorally identical to the source on the
//! reference interpreter for every *valid* dispatch index, and (3) let the specializer fold through
//! a dynamic-index `call_indirect` that it otherwise declines. The one documented divergence — an
//! out-of-signature index traps `Unreachable` instead of `IndirectCallType` — is pinned as a
//! known-gap test (both still trap).

use svm_interp::{Trap, Value};
use svm_ir::{BinOp, Block, Func, FuncType, Inst, IntTy, Module, Terminator, ValType};
use svm_peval::{lower_indirect_dispatch, specialize_with_config, SpecArg, SpecConfig};
use svm_verify::verify_module;

fn run(m: &Module, args: &[Value]) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    svm_interp::run(m, 0, args, &mut fuel)
}

const I32: ValType = ValType::I32;
const I64: ValType = ValType::I64;

/// A unary `i64 -> i64` leaf: `x <op> imm`.
fn unary(op: BinOp, imm: i64) -> Func {
    Func {
        params: vec![I64],
        results: vec![I64],
        blocks: vec![Block {
            params: vec![I64],
            insts: vec![
                Inst::ConstI64(imm),
                Inst::IntBin {
                    ty: IntTy::I64,
                    op,
                    a: 0,
                    b: 1,
                },
            ],
            term: Terminator::Return(vec![2]),
        }],
    }
}

/// Entry `dispatch(sel: i32, x: i64) -> i64 = call_indirect[I64->I64](sel)(x) + x`.
/// The trailing `+ x` forces the continuation to thread both the call result and a live input.
fn dispatch_module() -> Module {
    let ty = FuncType {
        params: vec![I64],
        results: vec![I64],
    };
    let entry = Func {
        params: vec![I32, I64],
        results: vec![I64],
        blocks: vec![Block {
            params: vec![I32, I64],
            insts: vec![
                Inst::CallIndirect {
                    ty,
                    idx: 0,        // sel
                    args: vec![1], // x
                },
                Inst::IntBin {
                    ty: IntTy::I64,
                    op: BinOp::Add,
                    a: 2, // call result
                    b: 1, // x
                },
            ],
            term: Terminator::Return(vec![3]),
        }],
    };
    Module {
        funcs: vec![
            entry,                  // 0: dispatch (sig [I32,I64]->[I64], not a candidate)
            unary(BinOp::Add, 1),   // 1: x + 1
            unary(BinOp::Mul, 2),   // 2: x * 2
            unary(BinOp::Sub, 100), // 3: x - 100
        ],
        memory: None,
        data: vec![],
        imports: vec![],
        exports: vec![],
        data_exports: vec![],
        data_ptrs: vec![],
        data_funcrefs: vec![],
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
    }
}

#[test]
fn lowering_verifies_and_matches_on_valid_indices() {
    let m = dispatch_module();
    verify_module(&m).expect("source verifies");
    let low = lower_indirect_dispatch(&m, 8);
    verify_module(&low).expect("lowered residual verifies");

    // No `call_indirect` survives; the candidate calls are now direct.
    let any_indirect = low.funcs.iter().flat_map(|f| &f.blocks).any(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i, Inst::CallIndirect { .. }))
            || matches!(b.term, Terminator::ReturnCallIndirect { .. })
    });
    assert!(
        !any_indirect,
        "no call_indirect should remain after lowering"
    );

    // Behaviorally identical for every valid (signature-matching) index 1,2,3.
    for sel in 1..=3i32 {
        for x in [-5i64, 0, 7, 1_000_000] {
            let a = &[Value::I32(sel), Value::I64(x)];
            assert_eq!(run(&m, a), run(&low, a), "diverged at sel={sel}, x={x}");
        }
    }
}

#[test]
fn out_of_signature_index_still_traps_known_gap() {
    // sel=0 selects func 0 (dispatch), whose signature does not match [I64]->[I64]: the source
    // traps IndirectCallType; the lowered form falls to the `unreachable` default. Both trap — the
    // one documented trap-*kind* divergence.
    let m = dispatch_module();
    let low = lower_indirect_dispatch(&m, 8);
    let a = &[Value::I32(0), Value::I64(9)];
    assert_eq!(run(&m, a), Err(Trap::IndirectCallType));
    assert!(run(&low, a).is_err(), "lowered must still trap");
}

#[test]
fn specializer_folds_through_dynamic_dispatch() {
    // Without the cap the specializer declines a dynamic-index call_indirect; with it, the residual
    // is produced and matches the interpreter with `sel` dynamic.
    let m = dispatch_module();
    let plain = SpecConfig::default();
    assert!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &plain).is_err(),
        "dynamic call_indirect is Unsupported without the cap"
    );

    let cfg = SpecConfig {
        indirect_targets_cap: Some(8),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &cfg)
        .expect("specializes with bounded indirect dispatch");
    verify_module(&residual).expect("residual verifies");

    for sel in 1..=3i32 {
        for x in [-3i64, 0, 42] {
            let a = &[Value::I32(sel), Value::I64(x)];
            assert_eq!(
                run(&m, a),
                run(&residual, a),
                "residual diverged sel={sel} x={x}"
            );
        }
    }
}

#[test]
fn constant_index_still_resolves_to_one_arm() {
    // With sel a static constant, lowering + specialization must collapse to the single resolved
    // target (no dynamic dispatch left).
    let m = dispatch_module();
    let cfg = SpecConfig {
        indirect_targets_cap: Some(8),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::ConstI32(2), SpecArg::Dynamic], &cfg)
        .expect("specializes");
    verify_module(&residual).expect("residual verifies");
    // `sel` is baked out, so the residual takes only the dynamic `x`.
    assert_eq!(
        residual.funcs[0].params,
        vec![I64],
        "only x remains dynamic"
    );
    // sel=2 → func 2 (x*2), then + x  ⇒  x*2 + x = 3x.
    for x in [0i64, 4, -6, 1000] {
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            run(&m, &[Value::I32(2), Value::I64(x)]),
            "residual != source at x={x}"
        );
        assert_eq!(
            run(&residual, &[Value::I64(x)]),
            Ok(vec![Value::I64(x * 3)])
        );
    }
}

/// A branchy tail: after the dynamic call, the continuation compares the result, references a live
/// input, and branches to two pre-existing blocks — stressing the continuation's terminator-operand
/// remapping and that untouched blocks keep their indices.
fn branchy_module() -> Module {
    let ty = FuncType {
        params: vec![I64],
        results: vec![I64],
    };
    let entry = Func {
        params: vec![I32, I64],
        results: vec![I64],
        blocks: vec![
            Block {
                params: vec![I32, I64], // sel=0, x=1
                insts: vec![
                    Inst::CallIndirect {
                        ty,
                        idx: 0,
                        args: vec![1],
                    }, // r = v2
                    Inst::ConstI64(0), // v3
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: svm_ir::CmpOp::LtS,
                        a: 2,
                        b: 3,
                    }, // r<0 → v4
                ],
                term: Terminator::BrIf {
                    cond: 4,
                    then_blk: 1,
                    then_args: vec![2], // pass r
                    else_blk: 2,
                    else_args: vec![1], // pass x
                },
            },
            Block {
                // b1(r) = -r
                params: vec![I64],
                insts: vec![
                    Inst::ConstI64(-1),
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Mul,
                        a: 0,
                        b: 1,
                    },
                ],
                term: Terminator::Return(vec![2]),
            },
            Block {
                // b2(x) = x
                params: vec![I64],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    Module {
        funcs: vec![
            entry,
            unary(BinOp::Add, 1),
            unary(BinOp::Mul, 2),
            unary(BinOp::Sub, 100),
        ],
        memory: None,
        data: vec![],
        imports: vec![],
        exports: vec![],
        data_exports: vec![],
        data_ptrs: vec![],
        data_funcrefs: vec![],
        impl_exports: vec![],
        types: vec![],
        debug_info: None,
    }
}

#[test]
fn branchy_tail_lowers_and_specializes_faithfully() {
    let m = branchy_module();
    verify_module(&m).expect("source verifies");
    let low = lower_indirect_dispatch(&m, 8);
    verify_module(&low).expect("lowered verifies");

    let cfg = SpecConfig {
        indirect_targets_cap: Some(8),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic, SpecArg::Dynamic], &cfg)
        .expect("specializes");
    verify_module(&residual).expect("residual verifies");

    for sel in 1..=3i32 {
        for x in [-9i64, -1, 0, 5, 250] {
            let a = &[Value::I32(sel), Value::I64(x)];
            assert_eq!(run(&m, a), run(&low, a), "lowered diverged sel={sel} x={x}");
            assert_eq!(
                run(&m, a),
                run(&residual, a),
                "residual diverged sel={sel} x={x}"
            );
        }
    }
}
