//! Unit tests for [`temen_wasm_jit::check_child_carve`] (#1123 slice 2) — the fail-closed §14 child
//! carve gate, the wasm-JIT twin of the native `mod_ok`/`fits` predicate. The gate is the confinement
//! precondition for routing a nested child's live `"mapped"` global to its carve, so it is exercised
//! directly here (its carve arithmetic is also fuzzed transitively via `temen_mask::Window::sub`).

use temen_wasm_jit::{check_child_carve, child_carve_fits, child_carve_fits_growable};

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

// ---- #1123 slice 2: fuzz the carve **admission** gate as its own confinement unit --------------------
//
// The child masking (accesses confined to the carve) is fuzzed via `temen_mask::Window::sub` (the `mask`
// libFuzzer target). This property test covers the other half — the geometry predicate
// `child_carve_fits` that decides *whether a carve may be spawned at all* — against an independent
// `u128`, overflow-free oracle, exercising exactly the overflow edges a unit test can't enumerate:
// `off + carve` wrapping, `parent_base + off` wrapping, and `carve_log2` at/near 63. A wrong bound here
// would admit a carve that straddles the parent window or dips into the NULL guard (§2/§4 escape).
// Deterministic (SplitMix64, no dev-deps — the escape-TCB stays dependency-free), so it is a stable CI
// gate rather than a nightly-only libFuzzer run.

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The independent oracle: recompute `child_carve_fits`'s admission predicate in `u128` (which cannot
/// overflow for any `u64` input), returning the carve byte size iff every clause holds.
fn oracle(
    declared: Option<u8>,
    off: u64,
    carve_log2: u32,
    parent_mapped: u64,
    parent_base: u64,
    guard: u64,
) -> Option<u64> {
    if carve_log2 >= 64 {
        return None; // the shift itself is out of range — never compute `1 << carve_log2`
    }
    let carve = 1u128 << carve_log2; // carve_log2 < 64 ⇒ fits u128 and (as u64) is exact
    let mod_ok = declared.is_none_or(|d| u32::from(d) <= carve_log2);
    let aligned = (off as u128) & (carve - 1) == 0;
    let carve_le_mapped = carve <= parent_mapped as u128;
    // `off + carve` overflowing u64 ⇒ the sum exceeds `u64::MAX >= parent_mapped`, so `fits_win` is
    // already false; no extra overflow guard is needed here.
    let fits_win = (off as u128 + carve) <= parent_mapped as u128;
    // The child's absolute base `parent_base + off` must be a real (non-wrapping) u64 address that
    // clears the guard — a wrapping sum is a nonsensical placement the gate rejects (the impl's
    // `checked_add`). In u128 that means: the sum stays within u64 **and** is `>= guard`.
    let base_sum = parent_base as u128 + off as u128;
    let clears_guard = base_sum <= u64::MAX as u128 && base_sum >= guard as u128;
    (mod_ok && carve_le_mapped && aligned && fits_win && clears_guard).then_some(carve as u64)
}

#[test]
fn child_carve_fits_matches_u128_oracle() {
    let mut rng = Rng(0xC0FF_EE00_1123_5107);
    for i in 0..3_000_000u64 {
        // Half the draws are fully random (mostly land on the reject path via a wild bound); half are
        // *structured* to hit the accept path and its boundaries — a small aligned carve inside a small
        // window just above the guard — so both outcomes are exercised densely.
        let structured = i & 1 == 0;

        let declared = match rng.next() % 4 {
            0 => None,
            _ => Some((rng.next() % 66) as u8), // 0..=65, incl. out-of-range vs a small carve
        };
        // carve_log2 spans the valid range plus the >=64 fault edge and the just-below-63 shift edge.
        let carve_log2 = match rng.next() % 8 {
            0 => (rng.next() % 256) as u32,    // wild (incl. >= 64)
            1 => 60 + (rng.next() % 8) as u32, // straddle 63/64
            _ => (rng.next() % 40) as u32,     // realistic sizes
        };
        let guard: u64 = if rng.next().is_multiple_of(4) {
            rng.next() % 65536
        } else {
            16384
        };

        let (off, parent_mapped, parent_base) = if structured && carve_log2 < 40 {
            let mlog = 12 + (rng.next() % 16) as u32; // 4 KiB .. 128 MiB window
            let parent_mapped = 1u64 << mlog;
            let carve = 1u64 << carve_log2.min(63);
            // An aligned offset within (or just past) the window, so alignment/fit/guard all vary.
            let slots = (parent_mapped / carve.max(1)).max(1);
            let off = (rng.next() % (slots + 2)).wrapping_mul(carve);
            let parent_base = if rng.next().is_multiple_of(3) {
                rng.next() % 32768 // sometimes low enough that base+off dips into the guard
            } else {
                1u64 << (mlog + 1)
            };
            (off, parent_mapped, parent_base)
        } else {
            // Fully random, including values near u64::MAX to probe the `checked_add` overflow guards.
            (rng.next(), rng.next(), rng.next())
        };

        let got = child_carve_fits(declared, off, carve_log2, parent_mapped, parent_base, guard);
        let want = oracle(declared, off, carve_log2, parent_mapped, parent_base, guard);
        assert_eq!(
            got, want,
            "child_carve_fits != u128 oracle: declared={declared:?} off={off} carve_log2={carve_log2} \
             parent_mapped={parent_mapped} parent_base={parent_base} guard={guard}"
        );
        // Whenever a carve is admitted, it is a valid power-of-two byte size that fits the window and
        // clears the guard — the confinement post-condition, stated independently of the oracle.
        if let Some(carve) = got {
            assert!(carve.is_power_of_two(), "admitted a non-power-of-two carve");
            assert!(
                carve <= parent_mapped,
                "admitted carve exceeds the parent window"
            );
            assert!(
                off.checked_add(carve).is_some_and(|e| e <= parent_mapped),
                "admitted carve straddles the parent window"
            );
            assert!(
                parent_base.checked_add(off).is_some_and(|a| a >= guard),
                "admitted carve dips into the NULL guard"
            );
        }
    }
}

// ---- #1253: the growable-backing carve gate as its own confinement unit ------------------------------
//
// `child_carve_fits_growable` admits a §14 op-13 child whose backing grows under a fixed grant ceiling
// (no 8×/buddy pre-size): the offset is only **wasm-page**-aligned (the driver's grant records sit in
// `[0, off)` below the child — a small offset the strict, carve-aligned `child_carve_fits` would refuse)
// and the carve is a ceiling in the holder's **reservation**, not its backed extent. The escape property
// is unchanged — an admitted carve is a power of two, wholly inside `[0, parent_reserved)`,
// page-aligned, and clears the NULL guard — so the same `u128` overflow-free oracle re-derives it, with
// `parent_reserved` in place of `parent_mapped` and page- for carve-alignment. This pins that decoupling
// the backing from the grant did **not** widen what geometry is admitted (still no straddle, no
// guard-dip), and the alignment relaxation admits *exactly* the page-aligned offsets, no more.

const WASM_PAGE: u64 = 1 << 16;

/// The independent oracle for [`child_carve_fits_growable`] — the growable twin of [`oracle`]: carve is a
/// ceiling in `parent_reserved`, the offset is wasm-page-aligned. `u128` throughout so no clause can wrap.
fn oracle_growable(
    declared: Option<u8>,
    off: u64,
    carve_log2: u32,
    parent_reserved: u64,
    parent_base: u64,
    guard: u64,
) -> Option<u64> {
    if carve_log2 >= 64 {
        return None;
    }
    let carve = 1u128 << carve_log2;
    let mod_ok = declared.is_none_or(|d| u32::from(d) <= carve_log2);
    let aligned = (off as u128) & (WASM_PAGE as u128 - 1) == 0;
    let carve_le_res = carve <= parent_reserved as u128;
    let fits_win = (off as u128 + carve) <= parent_reserved as u128;
    let base_sum = parent_base as u128 + off as u128;
    let clears_guard = base_sum <= u64::MAX as u128 && base_sum >= guard as u128;
    (mod_ok && carve_le_res && aligned && fits_win && clears_guard).then_some(carve as u64)
}

#[test]
fn child_carve_fits_growable_matches_u128_oracle() {
    let mut rng = Rng(0x9053_1253_600D_5EED);
    for i in 0..3_000_000u64 {
        let structured = i & 1 == 0;

        let declared = match rng.next() % 4 {
            0 => None,
            _ => Some((rng.next() % 66) as u8),
        };
        let carve_log2 = match rng.next() % 8 {
            0 => (rng.next() % 256) as u32,
            1 => 60 + (rng.next() % 8) as u32,
            _ => (rng.next() % 40) as u32,
        };
        let guard: u64 = if rng.next().is_multiple_of(4) {
            rng.next() % 65536
        } else {
            16384
        };

        let (off, parent_reserved, parent_base) = if structured && carve_log2 < 40 {
            // A large reservation (the growable holder addresses far past its live backing) with a
            // **page-aligned** offset holding a small driver header below the child — the accept path the
            // buddy gate could never reach (a page-sized offset is never carve-aligned for a big carve).
            let rlog = 20 + (rng.next() % 20) as u32; // 1 MiB .. 1 TiB reservation
            let parent_reserved = 1u64 << rlog;
            // Offsets: exact page multiples (the header sizes a real driver would use), plus a few past
            // the reservation, plus the occasional non-page value to exercise the reject path.
            let off = match rng.next() % 6 {
                0 => 0,
                1 => WASM_PAGE, // one-page header (the real layout)
                2 => WASM_PAGE * (1 + rng.next() % 4), // a few pages of header
                3 => parent_reserved.wrapping_sub(WASM_PAGE), // near the top
                4 => (rng.next() % (parent_reserved + WASM_PAGE)) & !(WASM_PAGE - 1), // random aligned
                _ => rng.next() % (parent_reserved + WASM_PAGE), // random (mostly non-page ⇒ reject)
            };
            let parent_base = if rng.next().is_multiple_of(3) {
                rng.next() % 32768
            } else {
                1u64 << (rlog + 1)
            };
            (off, parent_reserved, parent_base)
        } else {
            (rng.next(), rng.next(), rng.next())
        };

        let got = child_carve_fits_growable(
            declared,
            off,
            carve_log2,
            parent_reserved,
            parent_base,
            guard,
        );
        let want = oracle_growable(
            declared,
            off,
            carve_log2,
            parent_reserved,
            parent_base,
            guard,
        );
        assert_eq!(
            got, want,
            "child_carve_fits_growable != u128 oracle: declared={declared:?} off={off} \
             carve_log2={carve_log2} parent_reserved={parent_reserved} parent_base={parent_base} \
             guard={guard}"
        );
        if let Some(carve) = got {
            assert!(carve.is_power_of_two(), "admitted a non-power-of-two carve");
            assert_eq!(
                off & (WASM_PAGE - 1),
                0,
                "admitted a non-page-aligned offset"
            );
            assert!(
                off.checked_add(carve).is_some_and(|e| e <= parent_reserved),
                "admitted carve straddles the holder reservation"
            );
            assert!(
                parent_base.checked_add(off).is_some_and(|a| a >= guard),
                "admitted carve dips into the NULL guard"
            );
        }
    }
}
