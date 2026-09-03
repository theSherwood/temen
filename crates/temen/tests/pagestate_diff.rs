//! Deterministic regression sweep for the **#810 randomized page-state table** — the paged wasm-JIT
//! tier's per-access check over a live map built from THREE fuzzer-chosen page ops (interleaved
//! `Unmapped`/`Rw`/`Ro` runs, holes, re-commits) with scalar widths 1..16 and bulk walks straddling the
//! transitions, held against the interpreter oracle (`support/paged.rs::fuzz_one_pagestate`). This is
//! the stable-toolchain peer of the libFuzzer `pagestate` target: it drives the *same* function from
//! deterministic seeds so the confinement hinge is exercised on every PR (the CI fuzz job runs the
//! coverage-guided version nightly). Per AGENTS.md the masking lowering is fuzzed as its own unit; per
//! INVARIANTS #9 the interpreter is the oracle a mismatch fails against.

#[path = "support/paged.rs"]
mod paged;

use paged::{case_from_seed_pagestate, fuzz_one_pagestate, Cat};

#[test]
fn paged_access_matches_interp_over_randomized_page_maps() {
    // Each case builds + emits + wasmi-instantiates two runs, so keep the sweep modest (the unbounded
    // depth comes from the libFuzzer target); this is the regression + coverage floor.
    let iters: u64 = if cfg!(windows) { 400 } else { 1500 };
    let (mut trapped, mut passed) = (0u32, 0u32);
    for seed in 0..iters {
        match fuzz_one_pagestate(&case_from_seed_pagestate(seed)) {
            Cat::Trapped => trapped += 1,
            Cat::Passed => passed += 1,
            Cat::Skipped => {}
        }
    }

    eprintln!(
        "pagestate sweep: {trapped} trapping accesses, {passed} passing accesses ({iters} seeds)"
    );

    // Non-vacuity: the sweep must reach *both* sides of the check — accesses that trap on an
    // Unmapped/Ro page AND accesses that pass on admitted pages — or it is proving nothing.
    assert!(
        trapped > 20,
        "too few trapping accesses ({trapped}) — the check's fault path is near-vacuous"
    );
    assert!(
        passed > 20,
        "too few passing accesses ({passed}) — the check's admit path is near-vacuous"
    );
}
