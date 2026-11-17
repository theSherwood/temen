//! Shared partial-evaluation harness for the peval benches: the interpreter-agnostic loop-cell
//! discovery driver (`capture`) and the zero-config Lua auto-rolled residual builder (`futamura`).
//! Copied from the proven `crates/temen-llvm/tests/{peval_capture,lua_rolled}` drivers so the
//! out-of-workspace bench (which links Wasmtime) can reuse them; the tests remain the correctness
//! source of truth. TODO: dedup into a shared `crates/temen-futamura` crate.

pub mod capture;
pub mod futamura;

/// #3068 — the `+<toolchain>` argument for a step that needs a nightly (`-Z build-std` for the
/// tier-3 wasm64 target): `TEMEN_RUST_TOOLCHAIN`, which CI sets to its dated pin (`RUST_NIGHTLY`),
/// or `nightly`. A dated toolchain is not named `nightly`, so a literal `+nightly` under a pin
/// would install and use the current one.
pub fn nightly_arg() -> String {
    let toolchain = std::env::var("TEMEN_RUST_TOOLCHAIN").unwrap_or_else(|_| "nightly".into());
    format!("+{toolchain}")
}
