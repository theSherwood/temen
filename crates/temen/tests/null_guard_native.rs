//! **#964/#1094 on the native JIT**, differentially vs. the interpreter oracle. Every module's window
//! `mprotect`s `[0, POWERBOX_NULL_GUARD)` inaccessible (the guard is **unconditional** now — #1094, the
//! one canonical layout; no `__null_guard` marker), so a NULL dereference faults into the §5 guard and
//! reports `MemoryFault` exactly where the interpreter's `Unmapped`-seeded page map traps. The reserved
//! region is likewise **refused** to the Memory cap's page ops (`MprotectWindow::prot_pages`, the
//! `mmap_min_addr` analogue). A §14 child's window is its own, guarded like any root's
//! (`confined_child_null_guard.rs`).
//!
//! Hardware enforcement is host-page-granular: on a host whose page doesn't divide the guard
//! (e.g. 64 KiB aarch64) both backends skip the guard identically, and so does this test.

use std::sync::Arc;

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

const GUARD: i64 = temen_ir::POWERBOX_NULL_GUARD as i64;

/// The guard engages only when host-page-exact (both backends skip identically otherwise).
fn guard_active() -> bool {
    temen_ir::POWERBOX_NULL_GUARD.is_multiple_of(temen_interp::host_page_size())
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
  vr = call.cap 5 1 (i64, i64) -> (i64) vas (voff, vlen)
  return vr
  }
}
"#;

fn module() -> temen_ir::Module {
    let src = format!("memory 17\nexport 0 func \"_start\" 0\n{BODY}");
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// Run `func(args)` on both backends over a zero-seeded fully-mapped window, with the production
/// powerbox host shape (`set_self_module` + a Memory grant, as `Instance::grant_caps` does).
fn both(m: &temen_ir::Module, func: u32, args: &[i64]) -> (Result<Vec<Value>, Trap>, JitOutcome) {
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
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit");
    (ir, jo)
}

/// NULL loads and stores trap `MemoryFault` on the native JIT for every module — in lockstep with the
/// interpreter (#1094: the guard is unconditional); the guard boundary and the window top admit on
/// both backends.
#[test]
fn null_access_traps_on_native_jit() {
    if !guard_active() {
        return;
    }
    let m = module();
    for (func, what) in [(0u32, "load"), (1u32, "store")] {
        for probe in [0i64, 8, GUARD - 8, GUARD - 1] {
            let (ir, jo) = both(&m, func, &[probe]);
            assert_eq!(
                ir,
                Err(Trap::MemoryFault),
                "interp {what}({probe}) must trap under the guard"
            );
            assert!(
                matches!(jo, JitOutcome::Trapped(TrapKind::MemoryFault)),
                "jit {what}({probe}) must trap under the guard, got {jo:?}"
            );
        }
        // At and above the guard: legal on both backends.
        for probe in [GUARD, (1 << 17) - 8] {
            let (ir, jo) = both(&m, func, &[probe]);
            assert!(ir.is_ok(), "interp {what}({probe}) admits");
            assert!(
                matches!(jo, JitOutcome::Returned(_)),
                "jit {what}({probe}) admits, got {jo:?}"
            );
        }
    }
}

/// The reserved region is refused to the Memory cap's page ops on the native backend
/// (`MprotectWindow::prot_pages`) exactly as on the interpreter: `unmap` below the guard returns a
/// negative errno, and stays legal at/above it (#1094: unconditional).
#[test]
fn page_ops_below_guard_refused_on_native_jit() {
    if !guard_active() {
        return;
    }
    let page = temen_interp::host_page_size() as i64;
    let unmap = |m: &temen_ir::Module, off: i64, len: i64| -> (i64, i64) {
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

    let m = module();
    for (off, len, below) in [
        (0, GUARD, true),
        (GUARD - page, page, true),
        (GUARD, page, false),
    ] {
        let (iv, jv) = unmap(&m, off, len);
        assert_eq!(iv, jv, "backends agree on unmap({off}, {len})");
        assert_eq!(jv < 0, below, "unmap({off}, {len}) refusal: got {jv}");
    }
}
