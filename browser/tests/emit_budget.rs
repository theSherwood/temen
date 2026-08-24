//! **#1038 engine-survival pin**: an over-budget module through the whole-program open must leave
//! the engine ALIVE. Pre-#1038, `svm_onramp_jit_run_open` on a large-enough module (the shipped
//! Postgres, 119 MiB estimated emit) ran out of linear memory *inside* `emit_module` and hit Rust's
//! OOM `unreachable`, aborting the whole cdylib instance — the fallback chain never ran, and every
//! later call was dead. Now the reactor declines on the `est_emitted_size` module total before
//! allocating; this pins the observable contract: the open returns a clean `UNSUPPORTED`, the
//! status accessor still works, the tier-up fallback still admits the guest (with a degraded emit
//! set), and a subsequent small-module open on the same instance succeeds.

use svm_browser::{
    svm_coop_close, svm_coop_open, svm_onramp_jit_run_close, svm_onramp_jit_run_open, svm_status,
    STATUS_UNSUPPORTED,
};
use svm_ir::{Block, Export, Func, Inst, LoadOp, Memory, Module, Terminator, ValType};

/// The over-budget shape (the Postgres miniature shared with svm-wasm-jit's `emit_budget.rs`):
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

#[test]
fn over_budget_open_declines_and_the_engine_survives() {
    let big = svm_encode::encode_module(&wide_module(20, 48_000));
    // 1. The whole-program open DECLINES (pre-#1038 this OOM-aborted the instance here).
    let opened = svm_onramp_jit_run_open(big.as_ptr(), big.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened, -STATUS_UNSUPPORTED,
        "over-budget whole-program emit must refuse cleanly, not abort"
    );
    // 2. The engine is alive: the status accessor works…
    assert_eq!(svm_status(), STATUS_UNSUPPORTED);
    // …the tier-up fallback still ADMITS the same guest (its emit set degraded to fit — the chain
    // the playground actually takes)…
    let coop = svm_coop_open(big.as_ptr(), big.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        coop, 0,
        "the fallback tier serves the guest with a degraded emit set"
    );
    svm_coop_close();
    // …and a fresh small-module open on the same instance succeeds.
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
        let m = svm_text::parse_module(small_src).expect("parse");
        svm_verify::verify_module(&m).expect("verify");
        svm_encode::encode_module(&m)
    };
    let opened = svm_onramp_jit_run_open(small.as_ptr(), small.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "the engine still opens after the decline (status {})",
        svm_status()
    );
    svm_onramp_jit_run_close();
}
