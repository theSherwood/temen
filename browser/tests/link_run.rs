//! Native coverage for the `temen_link_run` entry's **binary-object** input path (wire v9): a unit
//! is binary iff it opens with the `Temen\0` container magic, so `.temeno` bytes and TEMEN-IR text mix
//! freely across the two params. The program rides binary here, the library text — the sniff must
//! tell them apart and the linked entry must run to its value.

use temen_browser::{
    temen_link_lib_close, temen_link_lib_open, temen_link_run, temen_link_run_lib,
    temen_link_text_libs, temen_status, temen_stderr_len, temen_stderr_ptr, STATUS_DECODE_ERR,
    STATUS_UNSUPPORTED,
};
use temen_ir::{Block, Export, Func, Inst, LinkUnitRef, Memory, Module, Terminator, ValType};

#[path = "support/ffi.rs"]
mod ffi;

/// A minimal unit: one `(i64) -> (i64)` kernel returning `val` (the `i64` param is the data-stack
/// pointer the synthesized powerbox `_start` passes — the §3e entry shape), with a window.
fn unit(val: i64) -> Module {
    Module {
        memory: Some(Memory {
            size_log2: 16,
            shadow: None,
        }),
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
    let _exports = ffi::lock();
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
    let _exports = ffi::lock();
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

/// The stderr slot, where a link entry leaves why it failed.
fn stderr() -> String {
    let (p, n) = (temen_stderr_ptr(), temen_stderr_len());
    if n == 0 {
        return String::new();
    }
    // SAFETY: the slot holds `n` bytes until the next entry replaces it, and the lock is held.
    String::from_utf8_lossy(unsafe { std::slice::from_raw_parts(p, n) }).into_owned()
}

/// #2297: a refused link says why. The status alone said only that the link was refused, so a C
/// card whose program defined its own `strlen` could show nothing better than "status 2". Natively
/// the reason is the `LinkFailure`'s; through the FFI it rides the stderr slot beside the status.
#[test]
fn a_refused_link_names_the_reason() {
    let _exports = ffi::lock();
    let exporting = |val: i64| Module {
        exports: vec![Export {
            name: "dup".into(),
            func: 0,
        }],
        ..unit(val)
    };
    let (lib, prog) = (exporting(7), exporting(42));

    // Both units define `dup`: a C linker's "multiple definition", not a bare status.
    let lib_exports: Vec<_> = lib
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();
    let lib_unit = LinkUnitRef {
        module: &lib,
        exports: &lib_exports,
        data_exports: &lib.data_exports,
        live: None,
    };
    let err = temen_browser::link_program_multi(&[lib_unit], &prog, "run")
        .expect_err("both units define `dup`");
    assert_eq!(err.status, STATUS_UNSUPPORTED);
    assert_eq!(err.message, "multiple definition of `dup`");

    // The same link through the FFI, against the library resident.
    let lib_text = temen_text::print_module(&lib);
    let h = temen_link_lib_open(lib_text.as_ptr(), lib_text.len());
    assert!(h >= 0, "the library opens");
    let link = |prog: &[u8]| -> (i32, i32, String) {
        let entry = b"run";
        let rc = temen_link_text_libs(
            [h].as_ptr(),
            1,
            prog.as_ptr(),
            prog.len(),
            entry.as_ptr(),
            entry.len(),
        );
        (rc, temen_status(), stderr())
    };
    let prog_text = temen_text::print_module(&prog);
    assert_eq!(
        link(prog_text.as_bytes()),
        (
            -STATUS_UNSUPPORTED,
            STATUS_UNSUPPORTED,
            "multiple definition of `dup`".to_string()
        )
    );
    // A program unit that does not load says so, under its own status.
    assert_eq!(
        link(b"\xff\xfe"),
        (
            -STATUS_DECODE_ERR,
            STATUS_DECODE_ERR,
            "the program unit is neither an encoded unit nor UTF-8 text".to_string()
        )
    );
    // So does a handle that is no longer open.
    temen_link_lib_close(h);
    assert_eq!(
        link(prog_text.as_bytes()),
        (
            -STATUS_UNSUPPORTED,
            STATUS_UNSUPPORTED,
            format!("a library handle in [{h}] is not open")
        )
    );
}
