//! libFuzzer target for the **#810 randomized page-state table** — the paged wasm-JIT tier's per-access
//! check (`"pagestate"` byte-per-page table + `"mapped"` coverage, #750/#1081) held against the
//! interpreter oracle over a live map richer than one carved range: coverage-guided bytes pick THREE
//! page ops (`unmap` / `protect` read-only / `protect` read-write / `map` read-write) over page-aligned
//! ranges, a leaf (scalar widths 1/2/4/8, `v128`, or a bulk fill/copy/move walk), and an access at a
//! page edge — so the table carries interleaved `Unmapped`/`Rw`/`Ro` runs, holes, and re-committed
//! pages, and the access straddles every kind of state transition. The emitted check must trap
//! `MemoryFault` exactly where the interpreter's `check_prot`/`check_prot_span` does, with identical
//! window bytes. A crash is a page-check confinement miscompile. The stable `pagestate_diff` test
//! drives the *same* `fuzz_one_pagestate` from seeds; INVARIANTS #2/#9, AGENTS.md ("fuzz the masking
//! lowering as its own unit").
//!
//! Run: `cargo +nightly fuzz run pagestate`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/paged.rs"]
mod paged;

fuzz_target!(|data: &[u8]| {
    let _ = paged::fuzz_one_pagestate(data);
});
