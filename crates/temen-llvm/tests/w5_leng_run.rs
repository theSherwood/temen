//! **W5 slice 2 (finish): the real `temen-leng` translator *runs* on temen to a correct result** (NIM.md §3e).
//!
//! `w5_leng_translates.rs` proved the whole `std` `temen-leng` translator lowers to verified TEMEN-IR — but
//! with libcore/liballoc symbols (`fmt`/`FromStr`/panic) *stubbed*, so a call would trap. This finishes
//! the slice: `-Z build-std` makes those symbols **real**, and the `leng_probe` guest actually executes
//! `temen_leng::translate_to_text` inside the sandbox, returning a checksum of the emitted Temen text that
//! must equal the *same* translation run **host-side** (the §18 differential — temen == native).
//!
//! Two things make the run integer-only, which is the point (W5 §3e path (b)):
//! - the fixture depends on `temen-leng` with `default-features = false`, so float literals fail closed
//!   and the module carries no `flt2dec`/`dec2flt` — the u128/vector float intrinsics the on-ramp does
//!   not yet lower (`llvm.fshl.v4i32`); and
//! - `build-std-features=panic_immediate_abort` collapses panics to `abort`, so no float *formatter* is
//!   pulled in for panic messages either.
//!
//! The Leng module translated is a fixed integer program (`main() -> 3 + 4*2`); the translator itself —
//! nif parse, the `Translator`, `String`/`Vec`/`HashMap`-driven TEMEN-text emission — is the real crate.
//! Interpreter-run: the JIT declines this ~255-func build-std module (a backend gap, not a translation
//! one — the module verifies), so this pins the tree-walker against the native oracle. Linux + the
//! `rustc`/`rust-src`/`llvm-*` build-std toolchain gated; skips otherwise.

#![cfg(target_os = "linux")]

mod common;

use temen_interp::Value;

/// The exact Leng module the `leng_probe` fixture's `main` translates — kept in sync with its `LENG`.
const LENG: &str =
    "(stmts (proc :main.0. . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";

/// The host-side oracle: translate the same module natively and FNV-1a its Temen text, truncated to the
/// non-negative i32 the guest's `main` returns (identical hash to `leng_probe/src/lib.rs`).
fn native_checksum() -> i32 {
    let text = temen_leng::translate_to_text(LENG).expect("native translate_to_text");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h & 0x7fff_ffff) as i32
}

#[test]
fn real_temen_leng_translator_runs_on_temen() {
    let Some(ll) = common::build_fixture_bc_std("leng_probe") else {
        return; // build-std toolchain absent — helper already logged the skip
    };

    let t = temen_llvm::translate_ll_path(&ll)
        .expect("temen-llvm translates the build-std temen-leng guest");
    // The untrusted producer's output is re-verified (DESIGN.md §2a) before it runs.
    temen_verify::verify_module(&t.module).expect("the translated temen-leng verifies");

    // Sanity: this is the whole translator, not a stub — a large module exporting the fixture's `main`.
    assert!(
        t.module.funcs.len() > 100,
        "expected the full temen-leng translator, got {} funcs",
        t.module.funcs.len()
    );
    let entry = t
        .exports
        .iter()
        .find(|(n, _)| n == "main")
        .expect("the guest's `main` is exported")
        .1;

    // `main(sp) -> i32`: a plain compute entry (no powerbox caps — the fixture allocates from a static
    // arena via its own `#[global_allocator]`), so the tree-walker runs it directly on the entry stack.
    let mut fuel = 5_000_000_000u64;
    let out = temen_interp::run(
        &t.module,
        entry,
        &[Value::I64(t.entry_sp as i64)],
        &mut fuel,
    )
    .expect("interp run of the temen-leng guest");

    // Byte-identical to the native translation (FNV over the full emitted text): the real temen-leng
    // produced the same TEMEN-IR *inside the sandbox* as it does natively.
    assert_eq!(
        out.as_slice(),
        [Value::I32(native_checksum())],
        "in-sandbox temen-leng translation must match the native oracle"
    );
}
