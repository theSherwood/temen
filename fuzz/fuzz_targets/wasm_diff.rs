//! libFuzzer target: generative interpreter-vs-**wasm-JIT** differential (`DESIGN.md` §18; issue
//! #910). The input bytes drive the shared structured generator (`crates/temen/tests/support`), which
//! synthesizes a verifier-valid module; it is emitted with `temen_wasm_jit::compile_module`, run under
//! `wasmi`, and held against the tree-walk oracle — result, termination, and (for a float-free memory
//! module) the final window byte-for-byte. A crash here is a wasm-JIT confinement miscompile (the
//! third confinement lowering — `emit_confine`/`emit_span_check`, INVARIANTS #2's "fuzzed hinge") or
//! a generator bug. The stable `wasm_diff` test drives the *same* `fuzz_one_wasm` from seeds.
//!
//! Run: `cargo +nightly fuzz run wasm_diff`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/wasmdiff.rs"]
mod wasmdiff;

fuzz_target!(|data: &[u8]| {
    // `without_caps`: the wasm tier refuses `cap.call`/page-op modules, so suppressing them keeps the
    // coverage-guided search on modules the emitter actually accepts (memory ops are the hinge).
    let mut g = wasmdiff::Gen::from_bytes(data).without_caps();
    wasmdiff::fuzz_one_wasm(&mut g);
});
