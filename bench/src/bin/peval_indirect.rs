//! Approach-A (bounded-target dynamic `call_indirect`) ROI across **all four backends**:
//! tree-walk interpreter, bytecode interpreter, Cranelift `svm-jit`, and `svm-wasm-jit` (its emitted
//! wasm run on Wasmtime — same Cranelift engine family as svm-jit).
//!
//! A loop dispatches, each iteration, through a data-dependent `call_indirect` to one of four
//! handlers — the shape A targets. Without A the specializer declines the whole function
//! (`Unsupported`); with A the dispatch folds into an inlined switch and the loop specializes. Each
//! backend runs the un-specialized interpreter module (the "blocked today" baseline) vs the
//! A-residual; we report the per-backend speedup.
//!
//! Run: `cargo run --release --manifest-path bench/Cargo.toml --bin peval_indirect`

use std::hint::black_box;
use std::time::{Duration, Instant};

use svm_interp::{Trap, Value};
use svm_ir::{
    BinOp, Block, ConvOp, Func, FuncType, Inst, IntTy, Module, Terminator, ValType,
    DEFAULT_RESERVED_LOG2,
};
use svm_jit::{CompiledModule, JitOutcome, Quota, INERT_CAP_THUNK};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};

// The svm-wasm-jit env ABI (mirrors crates/svm-wasm-jit/tests/differential.rs).
const WIN_BASE: u32 = 0x1_0000;
const ENV_PTR: u32 = 1024;

const I64: ValType = ValType::I64;

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

/// `run(n)`: seed `acc = cnt = n` (both dynamic), then loop `cnt` times doing
/// `acc = handler[acc & 3](acc, acc)` through a dynamic-index `call_indirect`.
fn dispatch_loop_module() -> Module {
    let hty = FuncType {
        params: vec![I64, I64],
        results: vec![I64],
    };
    let run = Func {
        params: vec![I64],
        results: vec![I64],
        blocks: vec![
            Block {
                params: vec![I64],
                insts: vec![],
                term: Terminator::Br {
                    target: 1,
                    args: vec![0, 0],
                },
            },
            Block {
                params: vec![I64, I64],
                insts: vec![
                    Inst::ConstI64(0),
                    Inst::IntCmp {
                        ty: IntTy::I64,
                        op: svm_ir::CmpOp::GtS,
                        a: 1,
                        b: 2,
                    },
                ],
                term: Terminator::BrIf {
                    cond: 3,
                    then_blk: 2,
                    then_args: vec![0, 1],
                    else_blk: 3,
                    else_args: vec![0],
                },
            },
            Block {
                params: vec![I64, I64],
                insts: vec![
                    Inst::Convert {
                        op: ConvOp::WrapI64,
                        a: 0,
                    },
                    Inst::ConstI32(3),
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::And,
                        a: 2,
                        b: 3,
                    },
                    Inst::ConstI32(1),
                    Inst::IntBin {
                        ty: IntTy::I32,
                        op: BinOp::Add,
                        a: 5,
                        b: 4,
                    },
                    Inst::CallIndirect {
                        ty: hty,
                        idx: 6,
                        args: vec![0, 0],
                    },
                    Inst::ConstI64(1),
                    Inst::IntBin {
                        ty: IntTy::I64,
                        op: BinOp::Sub,
                        a: 1,
                        b: 8,
                    },
                ],
                term: Terminator::Br {
                    target: 1,
                    args: vec![7, 9],
                },
            },
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
        types: vec![],
        debug_info: None,
    }
}

// ---- backend runners: (module, n) -> result i64 ----

fn tree_walk(m: &Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    one(svm_interp::run(m, 0, &[Value::I64(n)], &mut fuel))
}

fn bytecode(m: &Module, n: i64) -> i64 {
    let mut fuel = u64::MAX;
    match svm_interp::bytecode::compile_and_run(m, 0, &[Value::I64(n)], &mut fuel) {
        Some(r) => one(r),
        None => panic!("bytecode declined the module (out of subset)"),
    }
}

fn one(r: Result<Vec<Value>, Trap>) -> i64 {
    match r {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => *x,
            o => panic!("bad result {o:?}"),
        },
        Err(t) => panic!("trapped: {t:?}"),
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

fn jit_run(cm: &mut CompiledModule, n: i64) -> i64 {
    match cm.run(&[n], None, None, None) {
        Ok((JitOutcome::Returned(v), _)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

/// Compile `m` with the svm-wasm-jit emitter and run `f0` on Wasmtime with the confined-window ABI.
fn wasmjit_prepare(m: &Module) -> (wasmtime::Store<i32>, wasmtime::Instance) {
    use wasmtime::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store};
    let wasm = svm_wasm_jit::compile_module(m).expect("wasm-jit emit");
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = 2 + m.memory.map_or(0, |mc| (mc.size() >> 16) as u32);
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    // A generous fuel counter (i64 at env offset 0) — the loop is ~n dispatches, never OOF.
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 48).to_le_bytes())
        .unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define(&store, "env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call in this pure kernel")
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .expect("instantiate");
    (store, instance)
}

fn wasmjit_run(store: &mut wasmtime::Store<i32>, inst: &wasmtime::Instance, n: i64) -> i64 {
    use wasmtime::Val;
    let f = inst.get_func(&mut *store, "f0").expect("f0 exported");
    let params = [
        Val::I32(WIN_BASE as i32),
        Val::I32(ENV_PTR as i32),
        Val::I64(n),
    ];
    let mut results = [Val::I64(0)];
    f.call(&mut *store, &params, &mut results).expect("call f0");
    match results[0] {
        Val::I64(x) => x,
        o => panic!("bad wasm result {o:?}"),
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

fn main() {
    let m = dispatch_loop_module();
    svm_verify::verify_module(&m).expect("interpreter verifies");

    assert!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &SpecConfig::default()).is_err(),
        "dynamic call_indirect blocks specialization without approach A"
    );
    let cfg = SpecConfig {
        indirect_targets_cap: Some(4),
        ..SpecConfig::default()
    };
    let raw = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("A specializes");
    // Ship-shape: the residual is what you'd emit, so run the CFG cleanup the pipeline applies.
    let residual = svm_peval::optimize_module(&raw);
    svm_verify::verify_module(&residual).expect("residual verifies");

    // Correctness: every backend, base and residual, agrees with the tree-walk oracle.
    for n in [0i64, 1, 7, 64, 1000] {
        let oracle = tree_walk(&m, n);
        let mut cmb = jit_compile(&m);
        let mut cmr = jit_compile(&residual);
        let (mut sb, ib) = wasmjit_prepare(&m);
        let (mut sr, ir) = wasmjit_prepare(&residual);
        for (name, got) in [
            ("bytecode/base", bytecode(&m, n)),
            ("bytecode/res", bytecode(&residual, n)),
            ("treewalk/res", tree_walk(&residual, n)),
            ("svmjit/base", jit_run(&mut cmb, n)),
            ("svmjit/res", jit_run(&mut cmr, n)),
            ("wasmjit/base", wasmjit_run(&mut sb, &ib, n)),
            ("wasmjit/res", wasmjit_run(&mut sr, &ir, n)),
        ] {
            assert_eq!(got, oracle, "{name} wrong at n={n} (want {oracle})");
        }
    }

    let n: i64 = 2_000_000;
    let reps = 9;

    let mut cmb = jit_compile(&m);
    let mut cmr = jit_compile(&residual);
    let (mut sb, ib) = wasmjit_prepare(&m);
    let (mut sr, ir) = wasmjit_prepare(&residual);

    let rows = [
        (
            "tree-walk   ",
            best_of(reps, || tree_walk(&m, n)),
            best_of(reps, || tree_walk(&residual, n)),
        ),
        (
            "bytecode    ",
            best_of(reps, || bytecode(&m, n)),
            best_of(reps, || bytecode(&residual, n)),
        ),
        (
            "svm-jit     ",
            best_of(reps, || jit_run(&mut cmb, n)),
            best_of(reps, || jit_run(&mut cmr, n)),
        ),
        (
            "svm-wasm-jit",
            best_of(reps, || wasmjit_run(&mut sb, &ib, n)),
            best_of(reps, || wasmjit_run(&mut sr, &ir, n)),
        ),
    ];

    let blocks = |mm: &Module| mm.funcs.iter().map(|f| f.blocks.len()).sum::<usize>();
    println!(
        "\n=== approach-A indirect-dispatch ROI across all backends (n={n}, best of {reps}) ==="
    );
    println!(
        "baseline module: {} funcs / {} blocks   residual: {} funcs / {} blocks\n",
        m.funcs.len(),
        blocks(&m),
        residual.funcs.len(),
        blocks(&residual)
    );
    println!(
        "{:<14} {:>12} {:>12} {:>9}",
        "backend", "base", "residual", "speedup"
    );
    println!("{}", "-".repeat(50));
    for (name, base, res) in rows {
        println!(
            "{name} {base:>12.3?} {res:>12.3?} {:>8.2}x",
            base.as_secs_f64() / res.as_secs_f64()
        );
    }
}
