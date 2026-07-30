//! Native coverage for the `svm_link_run` entry's **binary-object** input path (wire v9): a unit
//! is binary iff it opens with the `SVM\0` container magic, so `.svmo` bytes and SVM-IR text mix
//! freely across the two params. The program rides binary here, the library text — the sniff must
//! tell them apart and the linked entry must run to its value.

use svm_browser::{svm_link_run, svm_status};
use svm_ir::{Block, Func, Inst, Memory, Module, Terminator, ValType};

/// A minimal unit: one `(i64) -> (i64)` kernel returning `val` (the `i64` param is the data-stack
/// pointer the synthesized powerbox `_start` passes — the §3e entry shape), with a window.
fn unit(val: i64) -> Module {
    Module {
        memory: Some(Memory { size_log2: 16 }),
        funcs: vec![Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![Block {
                params: vec![ValType::I64],
                insts: vec![Inst::ConstI64(val)],
                term: Terminator::Return(vec![1]),
            }],
        }],
        ..Default::default()
    }
}

#[test]
fn link_run_accepts_binary_object_units() {
    let prog_bytes = svm_encode::encode_unit(&unit(42)); // binary object (v9 flag set)
    let lib_text = svm_text::print_module(&unit(7)); // text unit
    let entry = b"run";

    let ret = svm_link_run(
        prog_bytes.as_ptr(),
        prog_bytes.len(),
        lib_text.as_ptr(),
        lib_text.len(),
        entry.as_ptr(),
        entry.len(),
        core::ptr::null(),
        0,
    );
    assert_eq!(svm_status(), 0, "link+run should succeed (ret={ret})");
    assert_eq!(ret, 42, "the program unit's entry returns its value");
}
