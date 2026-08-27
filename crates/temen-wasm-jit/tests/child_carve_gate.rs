//! Unit tests for [`temen_wasm_jit::check_child_carve`] (#1123 slice 2) — the fail-closed §14 child
//! carve gate, the wasm-JIT twin of the native `mod_ok`/`fits` predicate. The gate is the confinement
//! precondition for routing a nested child's live `"mapped"` global to its carve, so it is exercised
//! directly here (its carve arithmetic is also fuzzed transitively via `temen_mask::Window::sub`).

use temen_wasm_jit::check_child_carve;

/// A minimal verified child module declaring `memory {log2}` (the only field the gate reads).
fn child(log2: u8) -> temen_ir::Module {
    let src = format!(
        "memory {log2}\nfunc () -> (i64) {{\nblock 0 () {{\n  vr = i64.const 0\n  return vr\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A child with **no** linear memory (`mod_ok` is vacuously satisfied).
fn child_no_mem() -> temen_ir::Module {
    let m = temen_text::parse_module(
        "func () -> (i64) {\nblock 0 () {\n  vr = i64.const 0\n  return vr\n  }\n}\n",
    )
    .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

const GUARD: u64 = 16384; // POWERBOX_NULL_GUARD
const PARENT: u64 = 1 << 16; // 64 KiB parent window
const BASE: u64 = 1 << 16; // parent window base, well above the guard

#[test]
fn carve_equal_to_declared_is_admitted() {
    // declared 10, carve 10, aligned, fits, above the guard.
    assert_eq!(
        check_child_carve(&child(10), 16384, 10, PARENT, BASE, GUARD).unwrap(),
        1024
    );
}

#[test]
fn carve_larger_than_declared_is_a_safe_superset() {
    // A bigger carve is admitted (the child masks to the carve; it needs the room for heap growth).
    assert_eq!(
        check_child_carve(&child(10), 16384, 11, PARENT, BASE, GUARD).unwrap(),
        2048
    );
}

#[test]
fn carve_smaller_than_declared_is_refused() {
    // declared 12 (4 KiB) into a `slog = 10` (1 KiB) carve — the child could reach past the carve.
    assert!(check_child_carve(&child(12), 16384, 10, PARENT, BASE, GUARD).is_err());
}

#[test]
fn misaligned_carve_is_refused() {
    // off = 100 is not `1<<10`-aligned.
    assert!(check_child_carve(&child(10), 100, 10, PARENT, BASE, GUARD).is_err());
}

#[test]
fn carve_straddling_the_parent_window_is_refused() {
    // off = parent window ⇒ `off + carve > parent`, even though `off` is aligned.
    assert!(check_child_carve(&child(10), PARENT, 10, PARENT, BASE, GUARD).is_err());
}

#[test]
fn carve_in_the_null_region_is_refused() {
    // base + off = 0 < guard: the carve would dip into the reserved NULL page.
    assert!(check_child_carve(&child(10), 0, 10, PARENT, 0, GUARD).is_err());
}

#[test]
fn out_of_range_size_log2_is_refused() {
    // A wild bounce arg must fault closed before the shift overflows.
    assert!(check_child_carve(&child(10), 0, 64, PARENT, BASE, GUARD).is_err());
    assert!(check_child_carve(&child(10), 0, 200, PARENT, BASE, GUARD).is_err());
}

#[test]
fn memoryless_child_passes_mod_ok_and_is_gated_only_by_fit() {
    // No declared memory ⇒ `mod_ok` is vacuous; the carve is still bounded by the fit predicate.
    assert!(check_child_carve(&child_no_mem(), 16384, 10, PARENT, BASE, GUARD).is_ok());
    assert!(check_child_carve(&child_no_mem(), PARENT, 10, PARENT, BASE, GUARD).is_err());
}
