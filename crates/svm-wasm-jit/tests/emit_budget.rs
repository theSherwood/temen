//! **The module-total emit budget** (#1038): the wasm-compiled engine runs under a hard
//! linear-memory ceiling, and an over-large emit used to run out of memory *inside* `emit_module` —
//! aborting the whole engine instance instead of failing closed (reproduced with the shipped
//! Postgres module, 119 MiB estimated). Every emit entry now checks the `est_emitted_size` sum of
//! its emit set BEFORE allocating: the whole-program/reactor entries decline (`Unsupported` — the
//! open chain falls through), the tier-up entries **degrade** (drop the largest-estimate functions
//! to cross-tier leaves until the total fits), and the §22 unit emitter declines (the invoke runs
//! interpreted). These pins cover all three, plus the real-constant decline on a genuinely
//! over-budget module — which is fast precisely because the decline happens before any emit.

use svm_ir::{Block, Export, Func, Inst, LoadOp, Memory, Module, Terminator, ValType};

/// `_start` (func 0, `() -> (i64)`) calls `n_callees` load-padded leaves `f(x) -> (i64)`; each leaf
/// carries `loads_each` redundant `i64.load v0`s, so its estimated emitted body is
/// ~`128 * loads_each` bytes (the fat confine sequence per access). Built programmatically — a
/// half-million-instruction module parses too slowly as text.
fn wide_module(n_callees: usize, loads_each: usize) -> Module {
    let mut funcs = Vec::with_capacity(n_callees + 1);
    // f0: v0 = const 16384; v{i} = call i (v0); return v{n}.
    let mut insts = vec![Inst::ConstI64(16384)];
    for i in 1..=n_callees {
        insts.push(Inst::Call {
            func: i as u32,
            args: vec![0],
        });
    }
    funcs.push(Func {
        params: vec![],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![],
            insts,
            term: Terminator::Return(vec![n_callees as u32]),
        }],
    });
    for _ in 0..n_callees {
        let insts = vec![
            Inst::Load {
                op: LoadOp::I64,
                addr: 0,
                offset: 0,
            };
            loads_each
        ];
        funcs.push(Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64],
                insts,
                term: Terminator::Return(vec![loads_each as u32]),
            }],
        });
    }
    let m = Module {
        funcs,
        memory: Some(Memory { size_log2: 17 }),
        exports: vec![Export {
            name: "_start".into(),
            func: 0,
        }],
        ..Default::default()
    };
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// The real-constant pin: a module whose emit set estimates over the 104 MiB budget (20 leaves x
/// 48k loads, each under the 6.5 MB per-function cap, summing ~123 MiB — the Postgres shape in
/// miniature) must DECLINE the whole-program emit with
/// the budget error, before any allocation. Pre-#1038 this call OOM-aborted the wasm engine.
#[test]
fn over_budget_whole_program_declines_before_emitting() {
    let m = wide_module(20, 48_000);
    let err = svm_wasm_jit::compile_module_reactor(&m, 0, false)
        .expect_err("an over-budget module must decline, not emit");
    assert!(
        format!("{err:?}").contains("memory budget"),
        "the decline names the budget: {err:?}"
    );
}

/// The §22 unit-emitter guard: a guest-submitted unit big enough to OOM the engine mid-emit
/// declines instead (the seam then runs the invoke interpreted, fail-closed).
#[test]
fn over_budget_unit_emit_declines() {
    let m = wide_module(20, 48_000);
    let err = svm_wasm_jit::compile_module_b2(&m, false, 10)
        .expect_err("an over-budget unit must decline, not emit");
    assert!(
        format!("{err:?}").contains("memory budget"),
        "the decline names the budget: {err:?}"
    );
}

/// The reactor boundary, pinned with a small module + explicit budget (the `_budgeted` test seam):
/// over the budget declines; with the budget lifted the same module emits whole.
#[test]
fn reactor_budget_boundary() {
    let m = wide_module(3, 100); // ~12.8 KiB per leaf
    let err = svm_wasm_jit::compile_module_reactor_budgeted(&m, 0, false, usize::MAX, 20_000)
        .expect_err("over the explicit budget declines");
    assert!(format!("{err:?}").contains("memory budget"));
    let (wasm, emitted) =
        svm_wasm_jit::compile_module_reactor_budgeted(&m, 0, false, usize::MAX, usize::MAX)
            .expect("no budget: emits");
    assert!(emitted.iter().all(|&e| e), "everything emits unbudgeted");
    wasmi::Module::new(&wasmi::Engine::default(), &wasm[..]).expect("emitted wasm validates");
}

/// The tier-up entries DEGRADE instead of declining: over budget, the largest-estimate functions
/// drop to cross-tier leaves until the emitted total fits — the module still compiles, the entry
/// and the surviving leaves still emit, and the output validates.
#[test]
fn over_budget_tierup_degrades_to_fit() {
    let m = wide_module(3, 100); // f0 tiny; three ~12.8 KiB leaves; total ~38.6 KiB
    let (wasm, emitted) = svm_wasm_jit::compile_module_tierup_b2_budgeted(&m, false, 10, 30_000)
        .expect("over budget degrades — never a hard failure");
    assert!(emitted[0], "the entry stays emitted");
    assert_eq!(
        emitted.iter().filter(|&&e| e).count(),
        3,
        "exactly one leaf dropped to fit the 30 KiB budget: {emitted:?}"
    );
    wasmi::Module::new(&wasmi::Engine::default(), &wasm[..]).expect("degraded wasm validates");

    // Control: unbudgeted, the same module emits whole.
    let (_, emitted) =
        svm_wasm_jit::compile_module_tierup_b2_budgeted(&m, false, 10, usize::MAX).expect("emits");
    assert!(emitted.iter().all(|&e| e), "everything emits unbudgeted");
}
