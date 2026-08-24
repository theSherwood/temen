//! **W5 slice 2 (partial): the real `temen-leng` translator lowers to verified TEMEN-IR** (NIM.md §3e).
//!
//! Slice 1 ran `temen-leng`-*shaped* Rust on temen. This drives the **actual** crate: the `leng_probe`
//! fixture is a `std` powerbox program that calls `temen_leng::translate_to_text` on a fixed Leng
//! module, built through the LLVM on-ramp (`rustc +1.81` → `llvm-link-18` → `opt-18` → `temen-llvm`).
//!
//! What this pins, and what it does not:
//! - **Does:** the whole `std` `temen-leng` translator (+ `temen-ir`/`temen-text`/`temen-encode`) compiles to
//!   LLVM IR and **translates to a re-verified TEMEN-IR module** — i.e. no `no_std` port is needed
//!   (Rust generics monomorphize into the crate's own IR; only a bounded set of libcore/liballoc
//!   symbols stay external). `stub_unresolved_externs` lowers those to trap stubs, exactly the
//!   large-real-program bring-up the option is built for.
//! - **Does not:** *run* the translator to a correct result. `translate_to_text` reaches
//!   `core::fmt::Display`/`FromStr` (it formats and parses numbers), which are **stubbed** here — a
//!   call would trap. A correct run needs those libcore/liballoc functions *real*, via `-Z build-std`,
//!   which needs a nightly whose LLVM matches the `llvm-*-18` tools (this env has only LLVM-18 tools
//!   vs. LLVM-23 nightly — a toolchain-alignment task, tracked in §3e, not a code one).
//!
//! Gated to Linux + the `rustc +1.81.0`/`llvm-link-18`/`opt-18` on-ramp toolchain; skips otherwise.

#![cfg(target_os = "linux")]

mod common;

use temen_llvm::TranslateOptions;

#[test]
fn real_temen_leng_translator_lowers_to_verified_temen_ir() {
    let Some(ll) = common::build_fixture_bc("leng_probe") else {
        return; // toolchain absent — build_fixture_bc already logged the skip
    };

    let opts = TranslateOptions {
        stub_unresolved_externs: true,
        ..Default::default()
    };
    let t = temen_llvm::translate_ll_path_with_options(&ll, opts)
        .expect("temen-llvm translates the real temen-leng translator's LLVM IR");

    // The untrusted producer's output is re-verified (DESIGN.md §2a): the whole translated temen-leng
    // is well-formed TEMEN-IR, trap stubs and all.
    temen_verify::verify_module(&t.module).expect("the translated temen-leng verifies");

    // Sanity: this really is the whole translator, not a stub — it's a large module that exports the
    // fixture's `main`.
    assert!(
        t.module.funcs.len() > 100,
        "expected the full temen-leng translator (many funcs), got {}",
        t.module.funcs.len()
    );
    assert!(
        t.exports.iter().any(|(n, _)| n.contains("temen_leng")),
        "the real temen-leng translator's functions are present (e.g. temen_leng::nif::parse)"
    );
}
