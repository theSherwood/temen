//! **#1038 engine-survival pin**: an over-budget module offered whole to the emitter must leave the
//! engine ALIVE. Pre-#1038, the whole-program open on a large-enough module (the shipped Postgres,
//! 119 MiB estimated emit) ran out of linear memory *inside* `emit_module` and hit Rust's OOM
//! `unreachable`, aborting the whole cdylib instance — the fallback chain never ran, and every later
//! call was dead. Now the emitter declines on the `est_emitted_size` module total before allocating.
//! A module run offers its root to the emitter in the pump's one open (#1414 S4-3a), so this pins
//! the contract there: the open declines the root and admits the guest anyway (with a degraded
//! region emit), and a later small module on the same instance runs whole as a leaf.

use temen_browser::{
    temen_coop_close, temen_coop_open, temen_coop_program_wasm_len, temen_coop_wasm_len,
    COOP_LEAF_ROOT,
};
use temen_ir::{Block, Export, Func, Inst, LoadOp, Memory, Module, Terminator, ValType};

/// The over-budget shape (the Postgres miniature shared with temen-wasm-jit's `emit_budget.rs`):
/// `_start` calls 20 load-padded leaves, each under the 6.5 MB per-function cap, summing ~123 MiB
/// estimated — over the 104 MiB module budget.
fn wide_module(n_callees: usize, loads_each: usize) -> Module {
    let mut funcs = Vec::with_capacity(n_callees + 1);
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
            parent: 0,
            captures: vec![],
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
                parent: 0,
                captures: vec![],
                params: vec![ValType::I64],
                insts,
                term: Terminator::Return(vec![loads_each as u32]),
            }],
        });
    }
    let m = Module {
        funcs,
        memory: Some(Memory {
            size_log2: 17,
            shadow: None,
        }),
        exports: vec![Export {
            name: "_start".into(),
            func: 0,
        }],
        ..Default::default()
    };
    temen_verify::verify_module(&m).expect("verify");
    m
}

#[test]
fn over_budget_open_declines_and_the_engine_survives() {
    let open = |bytes: &[u8]| {
        temen_coop_open(
            bytes.as_ptr(),
            bytes.len(),
            core::ptr::null(),
            0,
            0,
            core::ptr::null(),
            0,
            COOP_LEAF_ROOT,
        )
    };
    let big = temen_encode::encode_module(&wide_module(20, 48_000));
    // 1. The root is declined whole (pre-#1038 this OOM-aborted the instance), and the open still
    // admits the guest: the interpreter drives it, with the region emit degraded to fit.
    assert_eq!(open(&big), 0, "the over-budget guest is admitted");
    assert_eq!(
        (temen_coop_program_wasm_len(0), temen_coop_wasm_len() > 0),
        (0, true),
        "no leaf; a region emit"
    );
    temen_coop_close();
    // 2. The engine is alive: a small module on the same instance runs whole as a leaf.
    let small_src = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 7
  return v0
  }
}
export 0 func "_start" 0
"#;
    let small = {
        let m = temen_text::parse_module(small_src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        temen_encode::encode_module(&m)
    };
    assert_eq!(open(&small), 0, "the engine still opens after the decline");
    assert!(
        temen_coop_program_wasm_len(0) > 0,
        "the small root runs whole"
    );
    temen_coop_close();
}
