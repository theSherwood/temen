//! Deterministic regression sweep for the **#1151 nested paged per-access check** — the emitted §14
//! nested tier's scalar page check held against the interpreter oracle (`support/nested_paged.rs`),
//! with the page-op bounce serviced on a real vCPU over the shared window. This is the
//! stable-toolchain peer of the libFuzzer `nested_paged` target: it drives the *same* `fuzz_one` from
//! deterministic seeds so the confinement hinge is exercised on every PR (the CI fuzz job runs the
//! coverage-guided version nightly). Per AGENTS.md the masking lowering is fuzzed as its own unit; per
//! INVARIANTS #9 the interpreter is the oracle a mismatch fails against.

#[path = "support/nested_paged.rs"]
mod nested_paged;

use nested_paged::{case_from_seed, fuzz_one, Cat};

#[test]
fn nested_paged_access_matches_interp_on_generated_cases() {
    // Each case builds + emits + wasmi-instantiates one run and interprets another, so keep the sweep
    // modest (the unbounded depth comes from the libFuzzer target); this is the regression + coverage
    // floor. macOS's 16-KiB pages leave fewer pages in the 128-KiB window — a coarser grid, fine.
    let iters: u64 = if cfg!(windows) { 400 } else { 1500 };
    let (mut trapped, mut passed) = (0u32, 0u32);
    for seed in 0..iters {
        match fuzz_one(&case_from_seed(seed)) {
            Cat::Trapped => trapped += 1,
            Cat::Passed => passed += 1,
            Cat::Skipped => {}
        }
    }

    eprintln!("nested_paged sweep: {trapped} trapping accesses, {passed} passing accesses ({iters} seeds)");

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
