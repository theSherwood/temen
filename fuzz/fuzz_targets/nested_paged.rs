//! libFuzzer target for the **#1151 nested paged per-access check** — the emitted §14 nested tier's
//! scalar page check (`compile_module_nested_paged` after `outline_nested_cap_calls`) held against the
//! interpreter oracle, with the page-op bounce serviced on a real vCPU over the shared window
//! (`bounce_call`) and the page-state table rebuilt from its live `Mem::map_info`. Coverage-guided
//! bytes pick a guest (unmap/protect × load/store × 8/64-bit), a page-op region, and an access
//! address at page edges; the emitted access must trap `MemoryFault` exactly where the interpreter's
//! `check_prot` does, and the window bytes must match. A crash is a nested paged confinement
//! miscompile. The stable `nested_paged_diff` test drives the *same* `fuzz_one` from seeds;
//! INVARIANTS #2/#9, AGENTS.md ("fuzz the masking lowering as its own unit").
//!
//! Run: `cargo +nightly fuzz run nested_paged`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/nested_paged.rs"]
mod nested_paged;

fuzz_target!(|data: &[u8]| {
    let _ = nested_paged::fuzz_one(data);
});
