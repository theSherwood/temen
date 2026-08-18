//! **#964 on the native JIT**, differentially vs. the interpreter oracle. A `__null_guard`-marked
//! module's window `mprotect`s `[0, POWERBOX_NULL_GUARD)` inaccessible, so a NULL dereference
//! faults into the §5 guard and reports `MemoryFault` exactly where the interpreter's
//! `Unmapped`-seeded page map traps — while the unmarked twin keeps the legacy behavior on both
//! backends. The reserved region is likewise **refused** to the Memory cap's page ops
//! (`MprotectWindow::prot_pages`, the `mmap_min_addr` analogue) and to `instantiate` carves (the
//! host seeds/copies a carve outside the guarded call, so a guarded carve would fault the host).
//!
//! Hardware enforcement is host-page-granular: on a host whose page doesn't divide the guard
//! (e.g. 64 KiB aarch64) both backends skip the guard identically, and so does this test.

use std::sync::Arc;

use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome, TrapKind};
use svm_text::parse_module;
use svm_verify::verify_module;

const GUARD: i64 = svm_ir::POWERBOX_NULL_GUARD as i64;

/// The guard engages only when host-page-exact (both backends skip identically otherwise).
fn guard_active() -> bool {
    svm_ir::POWERBOX_NULL_GUARD % svm_interp::host_page_size() == 0
}

/// f0: probe load at arg. f1: probe store at arg. f2 `(as, off, len)`: `unmap` via the
/// AddressSpace cap, returning its errno. `memory 17` = 128 KiB, fully mapped.
const BODY: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  i64.store v0 v0
  return v0
  }
}
func (i32, i64, i64) -> (i64) {
block 0 (vas: i32, voff: i64, vlen: i64) {
  vr = cap.call 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  return vr
  }
}
"#;

fn module(marked: bool) -> svm_ir::Module {
    let marker = if marked {
        "export 1 func \"__null_guard\" 0\n"
    } else {
        ""
    };
    let src = format!("memory 17\nexport 0 func \"_start\" 0\n{marker}{BODY}");
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// Run `func(args)` on both backends over a zero-seeded fully-mapped window, with the production
/// powerbox host shape (`set_self_module` + a Memory grant, as `Instance::grant_caps` does).
fn both(m: &svm_ir::Module, func: u32, args: &[i64]) -> (Result<Vec<Value>, Trap>, JitOutcome) {
    let init = vec![0u8; 1 << 17];
    let am = Arc::new(m.clone());

    let mut hi = Host::new();
    hi.set_self_module(&am);
    let ih = hi.grant_memory();
    let mut hj = Host::new();
    hj.set_self_module(&am);
    let jh = hj.grant_memory();
    assert_eq!(ih, jh, "the Memory grant must encode identically");

    // f2's first param is the i32 Memory-cap handle — patched in from the grant above.
    let vargs: Vec<Value> = args
        .iter()
        .enumerate()
        .map(|(i, &a)| {
            if func == 2 && i == 0 {
                Value::I32(ih)
            } else {
                Value::I64(a)
            }
        })
        .collect();
    let jargs: Vec<i64> = args
        .iter()
        .enumerate()
        .map(|(i, &a)| if func == 2 && i == 0 { jh as i64 } else { a })
        .collect();
    let mut fuel = 5_000_000u64;
    let (ir, _) = run_capture_reserved_with_host(m, func, &vargs, &mut fuel, &init, 0, &mut hi);
    let (jo, _) = compile_and_run_capture_reserved_with_host(
        m,
        func,
        &jargs,
        &init,
        0,
        svm_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    (ir, jo)
}

/// NULL loads and stores trap `MemoryFault` on the native JIT for a marked module — in lockstep
/// with the interpreter — and stay legal on the unmarked twin; the guard boundary and the window
/// top admit on both.
#[test]
fn marked_null_access_traps_on_native_jit() {
    if !guard_active() {
        return;
    }
    let marked = module(true);
    let legacy = module(false);
    for (func, what) in [(0u32, "load"), (1u32, "store")] {
        for probe in [0i64, 8, GUARD - 8, GUARD - 1] {
            let (ir, jo) = both(&marked, func, &[probe]);
            assert_eq!(
                ir,
                Err(Trap::MemoryFault),
                "interp {what}({probe}) must trap under the guard"
            );
            assert!(
                matches!(jo, JitOutcome::Trapped(TrapKind::MemoryFault)),
                "jit {what}({probe}) must trap under the guard, got {jo:?}"
            );
            let (ir, jo) = both(&legacy, func, &[probe]);
            assert!(
                ir.is_ok() && matches!(jo, JitOutcome::Returned(_)),
                "unmarked twin keeps the legacy behavior at {probe} ({jo:?})"
            );
        }
        // At and above the guard: legal on both backends, marked or not.
        for probe in [GUARD, (1 << 17) - 8] {
            for m in [&marked, &legacy] {
                let (ir, jo) = both(m, func, &[probe]);
                assert!(ir.is_ok(), "interp {what}({probe}) admits");
                assert!(
                    matches!(jo, JitOutcome::Returned(_)),
                    "jit {what}({probe}) admits, got {jo:?}"
                );
            }
        }
    }
}

/// The reserved region is refused to the Memory cap's page ops on the native backend
/// (`MprotectWindow::prot_pages`) exactly as on the interpreter: `unmap` below the guard returns a
/// negative errno on a marked module, stays legal above it, and is unchanged on the unmarked twin.
#[test]
fn page_ops_below_guard_refused_on_native_jit() {
    if !guard_active() {
        return;
    }
    let page = svm_interp::host_page_size() as i64;
    let unmap = |m: &svm_ir::Module, off: i64, len: i64| -> (i64, i64) {
        let (ir, jo) = both(m, 2, &[/* handle patched in `both` */ 0, off, len]);
        let iv = match ir.expect("interp unmap probe runs").pop() {
            Some(Value::I64(x)) => x,
            other => panic!("interp result {other:?}"),
        };
        let jv = match jo {
            JitOutcome::Returned(v) => v[0],
            other => panic!("jit unmap probe did not return: {other:?}"),
        };
        (iv, jv)
    };

    let marked = module(true);
    let legacy = module(false);
    for (off, len, below) in [
        (0, GUARD, true),
        (GUARD - page, page, true),
        (GUARD, page, false),
    ] {
        let (iv, jv) = unmap(&marked, off, len);
        assert_eq!(iv, jv, "backends agree on unmap({off}, {len})");
        assert_eq!(
            jv < 0,
            below,
            "unmap({off}, {len}) refusal (marked): got {jv}"
        );
    }
    let (iv, jv) = unmap(&legacy, 0, page);
    assert_eq!((iv, jv), (0, 0), "the unmarked twin keeps legacy page ops");
}

/// A §14 `instantiate` carve overlapping `[0, guard)` is refused `-EINVAL` on both backends for a
/// marked module (the host seeds/copies the carve outside the guarded call); a carve above the
/// guard nests fine, and the unmarked twin still carves anywhere.
#[test]
fn carve_below_guard_refused_on_native_jit() {
    if !guard_active() || !svm_jit::fiber_supported() {
        return;
    }
    // Parent (f0): instantiate f3 in a 4 KiB window at `off` and return the raw handle/errno —
    // no join, so a refused spawn is observable as the negative errno itself. f1/f2 keep the
    // probe indices of `BODY` stable for the marker export; f3 is the trivial child.
    let nest = |marked: bool, off: u64| -> String {
        let marker: &str = if marked {
            "export 1 func \"__null_guard\" 0\n"
        } else {
            ""
        };
        format!(
            "memory 17\nexport 0 func \"_start\" 0\n{marker}\
             func (i32) -> (i64) {{\n\
             block 0 (v0: i32) {{\n\
             \x20 v1 = i64.const 3\n\
             \x20 v2 = i64.const {off}\n\
             \x20 v3 = i64.const 12\n\
             \x20 v4 = i64.const 0\n\
             \x20 v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
             \x20 v6 = i64.extend_i32_s v5\n\
             \x20 return v6\n\
               }}\n\
             }}\n\
             func (i64) -> (i64) {{ block 0 (v0: i64) {{ return v0 }} }}\n\
             func (i64) -> (i64) {{ block 0 (v0: i64) {{ return v0 }} }}\n\
             func (i64) -> (i64) {{\n\
             block 0 (v0: i64) {{\n\
             \x20 v1 = i64.const 42\n\
             \x20 return v1\n\
               }}\n\
             }}\n"
        )
    };
    let run = |marked: bool, off: u64| -> (i64, i64) {
        let m = parse_module(&nest(marked, off)).expect("parse");
        verify_module(&m).expect("verify");
        let am = Arc::new(m.clone());
        let init = vec![0u8; 1 << 17];
        let mut hi = Host::new();
        hi.set_self_module(&am);
        let ih = hi.grant_instantiator(0, 1 << 17);
        let mut hj = Host::new();
        hj.set_self_module(&am);
        let jh = hj.grant_instantiator(0, 1 << 17);
        assert_eq!(ih, jh);
        let mut fuel = 5_000_000u64;
        let (ir, _) =
            run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &init, 0, &mut hi);
        let (jo, _) = compile_and_run_capture_reserved_with_host(
            &m,
            0,
            &[jh as i64],
            &init,
            0,
            svm_run::cap_thunk,
            &mut hj as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit");
        let iv = match ir.expect("interp parent runs").pop() {
            Some(Value::I64(x)) => x,
            other => panic!("interp result {other:?}"),
        };
        let jv = match jo {
            JitOutcome::Returned(v) => v[0],
            other => panic!("jit parent did not return: {other:?}"),
        };
        (iv, jv)
    };

    // Marked: a carve at 0 (inside the guard) is refused on both; one above the guard spawns.
    let (iv, jv) = run(true, 0);
    assert!(iv < 0 && jv < 0, "carve at 0 refused (marked): {iv}/{jv}");
    let (iv, jv) = run(true, 64 << 10);
    assert!(
        iv >= 0 && jv >= 0,
        "carve above the guard spawns (marked): {iv}/{jv}"
    );
    // Unmarked twin: a carve at 0 still spawns on both backends.
    let (iv, jv) = run(false, 0);
    assert!(
        iv >= 0 && jv >= 0,
        "unmarked twin carves at 0 as ever: {iv}/{jv}"
    );
}
