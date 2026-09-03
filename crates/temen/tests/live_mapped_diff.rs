//! Deterministic regression sweep for the **#810 live-`mapped` bound** — the mask-only wasm-JIT tier's
//! runtime-aware confinement (the emitted bounds check reading the driver-synced `"mapped"` global over
//! a guest-shaped window) held against the interpreter oracle (`support/live_mapped.rs`). This is the
//! stable-toolchain peer of the libFuzzer `live_mapped` target: it drives the *same* `fuzz_one` from
//! deterministic seeds so the confinement hinge is exercised on every PR (the CI fuzz job runs the
//! coverage-guided version nightly). Per AGENTS.md the masking lowering is fuzzed as its own unit; per
//! INVARIANTS #9 the interpreter is the oracle a mismatch fails against.

#[path = "support/live_mapped.rs"]
mod live_mapped;

use live_mapped::{case_from_seed, fuzz_one, Cat};

#[test]
fn live_mapped_access_matches_interp_on_generated_cases() {
    // Each case runs the guest twice (interpreter + tier-up with a wasmi instantiation), so keep the
    // sweep modest (the unbounded depth comes from the libFuzzer target); this is the regression +
    // coverage floor. macOS's 16-KiB pages leave fewer pages in the reservation — a coarser grid, fine.
    let iters: u64 = if cfg!(windows) { 400 } else { 1500 };
    let (mut trapped, mut passed, mut declined) = (0u32, 0u32, 0u32);
    for seed in 0..iters {
        match fuzz_one(&case_from_seed(seed)) {
            Cat::Trapped => trapped += 1,
            Cat::Passed => passed += 1,
            Cat::Declined => declined += 1,
            Cat::Skipped => {}
        }
    }

    eprintln!(
        "live_mapped sweep: {trapped} trapping, {passed} passing, {declined} declined ({iters} seeds)"
    );

    // Non-vacuity: the sweep must reach *both* sides of the live bound — accesses the synced bound
    // faults above the committed extent AND accesses it admits inside a grown region — and the
    // fail-closed decline arm (an unrepresentable hole), or it is proving nothing.
    assert!(
        trapped > 20,
        "too few trapping accesses ({trapped}) — the live bound's fault path is near-vacuous"
    );
    assert!(
        passed > 20,
        "too few passing accesses ({passed}) — the live bound's admit path is near-vacuous"
    );
    assert!(
        declined > 5,
        "too few declined cases ({declined}) — the unrepresentable-map decline arm is unreached"
    );
}
