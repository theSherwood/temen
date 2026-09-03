//! libFuzzer target for the **#810 live-`mapped` bound** — the mask-only wasm-JIT tier's runtime-aware
//! confinement (the emitted bounds check reads the live `"mapped"` global, synced by every driver from
//! the tier-up event) held against the interpreter oracle. Coverage-guided bytes pick a leaf (scalar
//! widths 1/2/4/8, `v128`, aligned atomics, bulk spans), two `vm_map` grows into the reserved tail
//! (contiguous, above a hole — the decline arm — or filling one), and an access address at
//! page/window/width edges; the
//! guest shapes its own window (interp-serviced on both tiers), the leaf tiers up onto emitted wasm
//! (run under `wasmi`), and must trap `MemoryFault` exactly where the interpreter's page map does —
//! with the window bytes identical and a canary past the reservation untouched (the escape property).
//! A crash is a live-bound confinement miscompile, or a sync-contract bug. The stable `live_mapped_diff`
//! test drives the *same* `fuzz_one` from seeds; INVARIANTS #2/#9, AGENTS.md ("fuzz the masking
//! lowering as its own unit").
//!
//! Run: `cargo +nightly fuzz run live_mapped`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/live_mapped.rs"]
mod live_mapped;

fuzz_target!(|data: &[u8]| {
    let _ = live_mapped::fuzz_one(data);
});
