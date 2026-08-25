//! libFuzzer target for the **#1081 paged bulk-memory per-page walk** — the emitted
//! `MemCopy`/`MemMove`/`MemFill` page check ([`temen_wasm_jit::emit_span_page_check`]) held against the
//! interpreter's `check_prot_span` oracle. Coverage-guided bytes pick a guest (fill/copy × unmap/protect),
//! a page-op region, and a span at fuzzer-chosen offsets/lengths; the leaf tiers up onto emitted wasm
//! (run under `wasmi`) and must trap `MemoryFault` at exactly the pages the interpreter does. A crash is a
//! paged-walk confinement miscompile — the fourth confinement lowering, unfuzzed until now (`wasm_diff`
//! suppresses call.cap/page-op modules). The stable `paged_walk` test drives the *same* `fuzz_one` from
//! seeds; INVARIANTS #2/#9, AGENTS.md ("fuzz the masking lowering as its own unit").
//!
//! Run: `cargo +nightly fuzz run paged_walk`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/paged.rs"]
mod paged;

fuzz_target!(|data: &[u8]| {
    let _ = paged::fuzz_one(data);
});
