//! Native coverage for the `temen_link_run` entry's **binary-object** input path (wire v9): a unit
//! is binary iff it opens with the `Temen\0` container magic, so `.temeno` bytes and TEMEN-IR text mix
//! freely across the two params. The program rides binary here, the library text — the sniff must
//! tell them apart and the linked entry must run to its value.

use temen_browser::{
    temen_link_lib_close, temen_link_lib_open, temen_link_run, temen_link_run_lib, temen_status,
};
use temen_ir::{Block, Func, Inst, Memory, Module, Terminator, ValType};

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
    let prog_bytes = temen_encode::encode_unit(&unit(42)); // binary object (v9 flag set)
    let lib_text = temen_text::print_module(&unit(7)); // text unit
    let entry = b"run";

    let ret = temen_link_run(
        prog_bytes.as_ptr(),
        prog_bytes.len(),
        lib_text.as_ptr(),
        lib_text.len(),
        entry.as_ptr(),
        entry.len(),
        core::ptr::null(),
        0,
    );
    assert_eq!(temen_status(), 0, "link+run should succeed (ret={ret})");
    assert_eq!(ret, 42, "the program unit's entry returns its value");
}

/// #1373: the resident-library path — open libraries once (two of them, as a frontend with a program
/// runtime and a macro-staging runtime does), run many programs against each by handle, each linking
/// to its own value; closing (or never opening) declines cleanly instead of faulting.
#[test]
fn resident_libraries_link_many_programs_by_handle() {
    let entry = b"run";
    let run = |h: i32, val: i64| -> (i64, i32) {
        let prog_text = temen_text::print_module(&unit(val));
        let ret = temen_link_run_lib(
            h,
            prog_text.as_ptr(),
            prog_text.len(),
            entry.as_ptr(),
            entry.len(),
            core::ptr::null(),
            0,
        );
        (ret, temen_status())
    };
    let lib_a = temen_encode::encode_unit(&unit(7)); // binary library
    let lib_b = temen_text::print_module(&unit(8)); // text library
    let ha = temen_link_lib_open(lib_a.as_ptr(), lib_a.len());
    let hb = temen_link_lib_open(lib_b.as_ptr(), lib_b.len());
    assert!(
        ha >= 0 && hb >= 0 && ha != hb,
        "two distinct handles ({ha}, {hb})"
    );
    drop((lib_a, lib_b)); // decoded, not borrowed — the host may free the bytes
    for val in [42i64, 43, 44] {
        assert_eq!(run(ha, val), (val, 0), "program {val} against library a");
        assert_eq!(
            run(hb, val + 100),
            (val + 100, 0),
            "program {} against library b",
            val + 100
        );
    }
    temen_link_lib_close(ha);
    assert_eq!(
        run(ha, 1),
        (0, temen_browser::STATUS_UNSUPPORTED),
        "closed handle declines"
    );
    assert_eq!(run(hb, 2), (2, 0), "the other library is untouched");
    temen_link_lib_close(hb);
    temen_link_lib_close(hb); // idempotent
    temen_link_lib_close(-1); // never a handle
    assert_eq!(
        run(99, 3),
        (0, temen_browser::STATUS_UNSUPPORTED),
        "unknown handle declines"
    );

    // A garbage library never becomes resident; the freed slot is reused by the next open.
    let junk = b"not a unit";
    assert_eq!(temen_link_lib_open(junk.as_ptr(), junk.len()), -1);
    assert_eq!(temen_status(), temen_browser::STATUS_DECODE_ERR);
    let lib_c = temen_text::print_module(&unit(9));
    let hc = temen_link_lib_open(lib_c.as_ptr(), lib_c.len());
    assert_eq!(hc, ha.min(hb), "lowest free slot is reused");
    assert_eq!(run(hc, 5), (5, 0));
    temen_link_lib_close(hc);
}
