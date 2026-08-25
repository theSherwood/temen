//! ROI micro-benchmark for approach A (bounded-target dynamic `call_indirect`).
//!
//! A loop dispatches, each iteration, through a **data-dependent** `call_indirect` to one of four
//! handlers — the shape A targets (a closure / first-class-function call whose target is a runtime
//! value, not derivable from the constant program). Without A the specializer declines the whole
//! function (`Unsupported`); with A the dispatch becomes an inlined 4-way switch and the loop
//! specializes end to end. We measure the residual against the un-specialized interpreter module
//! (the "blocked today" baseline) on both the tree-walk interpreter and the Cranelift JIT.
//!
//! Run: `cargo test -p temen-peval --test indirect_bench -- --ignored --nocapture`

use std::hint::black_box;
use std::time::{Duration, Instant};

use temen_ir::{
    BinOp, Block, ConvOp, Func, FuncType, Inst, IntTy, Module, Terminator, TypeEntry, ValType,
    DEFAULT_RESERVED_LOG2,
};
use temen_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};
use temen_verify::verify_module;

const I64: ValType = ValType::I64;

/// A handler `h(acc: i64, _dummy: i64) -> i64 = acc <op> imm`. The two-arg signature keeps it
/// distinct from the entry, so the candidate set is exactly the four handlers.
fn handler(op: BinOp, imm: i64) -> Func {
    Func {
        params: vec![I64, I64],
        results: vec![I64],
        blocks: vec![Block {
            params: vec![I64, I64],
            insts: vec![
                Inst::ConstI64(imm),
                Inst::IntBin {
                    ty: IntTy::I64,
                    op,
                    a: 0,
                    b: 2,
                },
            ],
            term: Terminator::Return(vec![3]),
        }],
    }
}

/// `run(n)`: seed `acc = cnt = n` (both **dynamic**), then loop `cnt` times doing
/// `acc = handler[acc & 3](acc, acc)`, dispatched indirectly. Keeping `acc` and the countdown
/// dynamic makes the loop body a single memoized context (a residual loop), not an unroll — the way
/// a real interpreter's runtime loop behaves (the constant driver is the program, not the trip
/// count).
fn dispatch_loop_module() -> Module {
    let hty = FuncType {
        params: vec![I64, I64],
        results: vec![I64],
    };
    // run(n: i64) -> i64
    let run = Func {
        params: vec![I64],
        results: vec![I64],
        blocks: vec![
            // b0(n): acc=n, cnt=n → loop header
            Block {
                params: vec![I64],
                insts: vec![],
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 0],
                }, // (acc=n, cnt=n)
            },
            // b1(acc, cnt): if cnt > 0 → body else return acc
            Block {
                params: vec![I64, I64],
                insts: vec![
                    Inst::ConstI64(0), // 2
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: temen_ir::CmpOp::GtS,
                        a: 1,
                        b: 2,
                    }, // 3
                ],
                term: Terminator::BrIf {
                    cond: 3,
                    then_blk: 2,
                    then_args: vec![0, 1],
                    else_blk: 3,
                    else_args: vec![0],
                },
            },
            // b2(acc, cnt): acc' = call_indirect[acc&3](acc,acc); cnt' = cnt - 1
            Block {
                params: vec![I64, I64],
                insts: vec![
                    Inst::Convert {
                        op: ConvOp::WrapI64,
                        a: 0,
                    }, // 2: acc as i32
                    Inst::ConstI32(3), // 3
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::And,
                        a: 2,
                        b: 3,
                    }, // 4: sel
                    Inst::ConstI32(1), // 5: base handler idx
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 5,
                        b: 4,
                    }, // 6: idx = 1 + sel
                    Inst::CallIndirect {
                        ty: 0, // #922: hty interned at type-section index 0
                        idx: 6,
                        args: vec![0, 0],
                    }, // 7: acc'
                    Inst::ConstI64(1), // 8
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Sub,
                        a: 1,
                        b: 8,
                    }, // 9: cnt - 1
                ],
                term: Terminator::Br {
                    target: 1,
                    args: vec![7, 9],
                },
            },
            // b3(acc): return acc
            Block {
                params: vec![I64],
                insts: vec![],
                term: Terminator::Return(vec![0]),
            },
        ],
    };
    Module {
        funcs: vec![
            run,
            handler(BinOp::Add, 1),
            handler(BinOp::Mul, 3),
            handler(BinOp::Xor, 7),
            handler(BinOp::Sub, 2),
        ],
        memory: None,
        data: vec![],
        imports: vec![],
        exports: vec![],
        data_exports: vec![],
        data_ptrs: vec![],
        data_funcrefs: vec![],
        impl_exports: vec![],
        types: vec![TypeEntry::Func(hty)],
        debug_info: None,
    }
}

fn interp1(m: &Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    match temen_interp::run(m, 0, &[temen_interp::Value::I64(n)], &mut fuel) {
        Ok(v) => match v.as_slice() {
            [temen_interp::Value::I64(x)] => *x,
            o => panic!("bad interp result {o:?}"),
        },
        Err(t) => panic!("interp trapped: {t:?}"),
    }
}

fn jit_compile(m: &Module) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        Quota::default(),
        0,
    )
    .expect("jit compile")
}

fn jit1(cm: &mut CompiledModule, n: i64) -> i64 {
    match cm.run(&[n], None, None) {
        Ok((JitOutcome::Returned(v), _)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

fn best_of(reps: usize, mut f: impl FnMut() -> i64) -> Duration {
    f();
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        let r = f();
        best = best.min(start.elapsed());
        black_box(r);
    }
    best
}

#[test]
#[ignore = "benchmark — run with --ignored --nocapture"]
fn indirect_dispatch_roi() {
    let m = dispatch_loop_module();
    verify_module(&m).expect("interpreter verifies");

    // Baseline: the specializer cannot get past the dynamic dispatch.
    assert!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &SpecConfig::default()).is_err(),
        "dynamic call_indirect blocks specialization without approach A"
    );

    // Approach A: fold the dispatch into an inlined switch; the loop specializes.
    let cfg = SpecConfig {
        indirect_targets_cap: Some(4),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("A specializes");
    verify_module(&residual).expect("residual verifies");

    // Correctness across both backends before timing.
    for n in [0i64, 1, 7, 64, 1000] {
        let oracle = interp1(&m, n);
        assert_eq!(
            interp1(&residual, n),
            oracle,
            "interp(residual) wrong at n={n}"
        );
    }
    let mut cm_base = jit_compile(&m);
    let mut cm_res = jit_compile(&residual);
    for n in [0i64, 1, 7, 64, 1000] {
        assert_eq!(
            jit1(&mut cm_base, n),
            interp1(&m, n),
            "jit(base) wrong n={n}"
        );
        assert_eq!(
            jit1(&mut cm_res, n),
            interp1(&m, n),
            "jit(residual) wrong n={n}"
        );
    }

    let n: i64 = 1_000_000;
    let reps = 7;
    let ti_base = best_of(reps, || interp1(&m, n));
    let ti_res = best_of(reps, || interp1(&residual, n));
    let tj_base = best_of(reps, || jit1(&mut cm_base, n));
    let tj_res = best_of(reps, || jit1(&mut cm_res, n));

    let blocks = |mm: &Module| mm.funcs.iter().map(|f| f.blocks.len()).sum::<usize>();
    println!("\n=== approach-A indirect-dispatch ROI (n={n}) ===");
    println!(
        "residual: {} funcs, {} blocks (baseline {} funcs, {} blocks)",
        residual.funcs.len(),
        blocks(&residual),
        m.funcs.len(),
        blocks(&m)
    );
    println!(
        "interp : base {:>9.3?}  residual {:>9.3?}  speedup {:.2}x",
        ti_base,
        ti_res,
        ti_base.as_secs_f64() / ti_res.as_secs_f64()
    );
    println!(
        "jit    : base {:>9.3?}  residual {:>9.3?}  speedup {:.2}x",
        tj_base,
        tj_res,
        tj_base.as_secs_f64() / tj_res.as_secs_f64()
    );
}
