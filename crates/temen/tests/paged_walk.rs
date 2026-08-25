//! Deterministic regression sweep for the **#1081 paged bulk-memory per-page walk** — the emitted
//! `MemCopy`/`MemMove`/`MemFill` page check held against the interpreter oracle (`support/paged.rs`).
//! This is the stable-toolchain peer of the libFuzzer `paged_walk` target: it drives the *same*
//! `fuzz_one` from deterministic seeds so the confinement hinge is exercised on every PR (the CI fuzz
//! job runs the coverage-guided version nightly). Per AGENTS.md the masking lowering is fuzzed as its
//! own unit; per INVARIANTS #9 the interpreter is the oracle a mismatch fails against.

#[path = "support/paged.rs"]
mod paged;

use paged::{case_from_seed, fuzz_one, Cat};

#[test]
fn paged_walk_matches_interp_on_generated_spans() {
    // Each case builds + emits + wasmi-instantiates two runs, so keep the sweep modest (the unbounded
    // depth comes from the libFuzzer target); this is the regression + coverage floor. macOS's 16-KiB
    // pages leave fewer pages in the 128-KiB window, so the same seeds explore a coarser grid — fine.
    let iters: u64 = if cfg!(windows) { 400 } else { 1500 };
    let (mut trapped, mut passed) = (0u32, 0u32);
    for seed in 0..iters {
        match fuzz_one(&case_from_seed(seed)) {
            Cat::Trapped => trapped += 1,
            Cat::Passed => passed += 1,
            Cat::Skipped => {}
        }
    }

    eprintln!("paged_walk sweep: {trapped} trapping spans, {passed} passing spans ({iters} seeds)");

    // Non-vacuity: the sweep must actually reach *both* sides of the walk — spans that trap on an
    // Unmapped/Ro page AND spans that pass on committed pages — or it is proving nothing (e.g. every
    // case out-of-window before the walk even runs). Floors well below today's counts catch a
    // generator regression without being flaky.
    assert!(
        trapped > 20,
        "too few trapping spans ({trapped}) — the walk's fault path is near-vacuous"
    );
    assert!(
        passed > 20,
        "too few passing spans ({passed}) — the walk's admit path is near-vacuous"
    );
}
