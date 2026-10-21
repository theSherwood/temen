//! libFuzzer target for the **trap-confinement unit** (`DESIGN.md` §4, I1; §14; §18).
//!
//! The confinement unit is the part of escape-freedom the verifier does *not* cover, so
//! it is fuzzed in isolation against its crisp invariant: for any inputs,
//! `Window::checked` is total (never panics) and returns `Some(a)` **iff**
//! `[addr+offset, addr+offset+width) ⊆ [0, mapped)`, in which case `a == confine(addr, offset)`
//! (the raw, *unmasked* `addr + offset`) and `[a, a+width) ⊆ [0, mapped) ⊆ [0, reserved)`. Every
//! out-of-window access **faults** (`None`) — no aliasing back in. Both constructors are driven —
//! the fully-mapped form ([`Window::new`]) and the decoupled reserved/mapped form
//! ([`Window::with_mapped`], §4). Every window starts at 0 of its own address space (§14: a child
//! owns a window of its own), so there is no base to fuzz.
//!
//! Run: `cargo +nightly fuzz run mask`
#![no_main]

use libfuzzer_sys::fuzz_target;
use temen_mask::Window;

/// Assert the crisp invariant for one window under **trap-confinement**: `confine` is the raw
/// effective address `addr + offset` (no masking), and `checked` admits an access iff it lies fully
/// within the backed `[0, mapped)`, returning that address, and otherwise faults. Total — never
/// panics.
fn check(w: Window, addr: u64, offset: u64, width: u32) {
    let reserved = w.reserved();
    let mapped = w.mapped();
    assert!(mapped <= reserved, "mapped must not exceed reserved");

    // `confine` is the raw (unmasked) effective address — trap-confinement, no `& mask`.
    let a = w.confine(addr, offset);
    assert_eq!(a, addr.wrapping_add(offset), "confine = addr + offset");

    // In bounds iff `[addr+offset, addr+offset+width) ⊆ [0, mapped)`, computed overflow-free.
    let in_bounds = addr
        .checked_add(offset)
        .and_then(|e| e.checked_add(width as u64))
        .is_some_and(|end| end <= mapped);

    match w.checked(addr, offset, width) {
        Some(c) => {
            assert!(in_bounds, "admitted an out-of-window access");
            assert_eq!(c, a, "checked must return the confined address");
            assert!(
                c + width as u64 <= mapped,
                "confined access escaped the backed [0, mapped)"
            );
        }
        None => assert!(!in_bounds, "faulted on an in-mapped access"),
    }

    // `checked_reserved` is the same arithmetic bounded at `reserved` (the escape bound the
    // interpreter's scalar `confine_checked` uses) — assert it against an overflow-free oracle.
    let in_reserved = addr
        .checked_add(offset)
        .and_then(|e| e.checked_add(width as u64))
        .is_some_and(|end| end <= reserved);
    match w.checked_reserved(addr, offset, width) {
        Some(c) => {
            assert!(
                in_reserved,
                "checked_reserved admitted an out-of-reservation access"
            );
            assert_eq!(c, a, "checked_reserved must return the confined address");
        }
        None => assert!(
            !in_reserved,
            "checked_reserved faulted on an in-reservation access"
        ),
    }
}

/// Fuzz the **bulk-memory span confinement** reference (D62): `temen_mask::Window::span_checked` admits
/// a span `[ptr, ptr+len)` iff it lies in `[0, reserved)` (with `len == 0` an in-bounds no-op),
/// computed overflow-free. This asserts it against a clean `u128` oracle for every input — the
/// arithmetic is the subtle, security-critical part (a wrong bound would admit an out-of-window bulk
/// copy). The JIT's `confine_span` emits the overflow-avoiding form of exactly this predicate.
fn check_span(w: Window, ptr: u64, len: u64) {
    let reserved = w.reserved();
    // The one span-OOB reference (`temen_mask::Window::span_checked`) — the arithmetic the JIT's
    // `confine_span` emits and the interpreter calls; fuzzed here so it is not transcribed.
    let masked = w.span_checked(ptr, len);
    // Independent oracle in u128 (cannot overflow): a non-empty span is in-bounds iff its end lies
    // within the reservation; an empty span is always an in-bounds no-op.
    let in_bounds = len == 0 || (ptr as u128 + len as u128) <= reserved as u128;
    match masked {
        Some(a) => {
            assert!(
                in_bounds,
                "span_checked admitted an out-of-window span: ptr={ptr} len={len}"
            );
            assert_eq!(a, ptr, "span_checked must return the span's start");
        }
        None => assert!(
            !in_bounds,
            "span_checked faulted on an in-window span: ptr={ptr} len={len}"
        ),
    }
}

fuzz_target!(|data: &[u8]| {
    // Derive the inputs from the fuzz bytes (pad short inputs with zeros).
    let mut b = [0u8; 26];
    let n = data.len().min(b.len());
    b[..n].copy_from_slice(&data[..n]);

    let addr = u64::from_le_bytes(b[0..8].try_into().unwrap());
    let offset = u64::from_le_bytes(b[8..16].try_into().unwrap());
    // 1..=8 scalar widths, plus 16 for the §17 `v128` access (D58) — the masking unit is
    // width-parametric, so the wider SIMD load/store rides the same confinement invariant.
    let width = match b[16] % 9 {
        8 => 16,
        k => (k as u32) + 1,
    };
    let reserved_log2 = b[17]; // any byte, incl. out-of-range (Window clamps)
    let mapped = u64::from_le_bytes(b[18..26].try_into().unwrap()); // clamped to reserved

    // Fully-mapped form (mapped == reserved == size) and the decoupled form — both must keep every
    // access confined to their own window.
    let full = Window::new(reserved_log2);
    let decoupled = Window::with_mapped(reserved_log2, mapped);
    check(full, addr, offset, width);
    check(decoupled, addr, offset, width);

    // Bulk-memory span confinement (D62): drive the same reservation the scalar checks use, on both
    // forms. `offset`/`mapped`/`addr` cross-fill so both small and large lengths are exercised against
    // various pointers.
    check_span(full, addr, mapped);
    check_span(full, addr, offset);
    check_span(full, offset, addr);
    check_span(decoupled, addr, mapped);
    check_span(decoupled, offset, addr);
});
