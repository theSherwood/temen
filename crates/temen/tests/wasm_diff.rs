//! Generative differential of the **wasm-JIT** against the tree-walk interpreter (DESIGN.md §18;
//! issue #910). For thousands of seeds we synthesize a verifier-valid module (`support/irgen.rs`),
//! emit it with `temen_wasm_jit::compile_module`, run its entry under `wasmi`, and assert it agrees
//! with the interpreter oracle on the result, on terminating, and — for a float-free memory module —
//! on the final window byte-for-byte (the escape-oracle: the wasm tier's `emit_confine`/
//! `emit_span_check` must mask every access into `[0, size)`, or the window diverges).
//!
//! This is the wasm-tier peer of `jit_fuzz.rs` (the Cranelift generator): it closes the coverage
//! asymmetry INVARIANTS #2 flags — masking is "the fuzzed hinge", but the wasm lowering had only
//! hand-written kernels (`temen-wasm-jit/tests/differential.rs`). Stable-toolchain (deterministic
//! seeds, runs in CI) as the regression + non-vacuity gate; the libFuzzer `wasm_diff` target drives
//! the *same* `fuzz_one_wasm` from coverage-guided input for the unbounded confinement exploration.

#[path = "support/wasmdiff.rs"]
mod wasmdiff;

use wasmdiff::{fuzz_one_wasm, Gen};

/// The seed transform: distinct from `jit_fuzz`'s so the two generators explore different modules.
fn seed_gen(seed: u64) -> Gen {
    // `without_caps`: the wasm tier refuses `cap.call`/page-op modules (`Unsupported`), so suppressing
    // them (the Cranelift path grants a Memory cap and covers them) roughly doubles the fraction of
    // modules that actually reach the emitter instead of being skipped.
    Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2A5D_C0DE_BEEF).without_caps()
}

#[test]
fn wasm_jit_matches_interp_on_generated_modules() {
    // Windows commits every page against the system limit (no overcommit), so trim the sweep there as
    // `jit_fuzz` does. The bulk of the confinement depth comes from the libFuzzer target; this sweep
    // is the deterministic regression + coverage floor.
    let iters: u64 = if cfg!(windows) { 1200 } else { 8000 };
    for seed in 0..iters {
        let mut g = seed_gen(seed);
        fuzz_one_wasm(&mut g);
    }
}

/// Non-vacuity: the generator must actually reach the wasm tier's **escape-oracle** — a float-free,
/// int-entry, RO-free module with memory that the tier accepts, whose final window is byte-compared —
/// on enough seeds that the sweep above is exercising the confinement lowering, not silently skipping
/// everything. (Float / cap / out-of-subset modules are legitimately skipped; this guards that the
/// *memory-safety* core is not.)
#[test]
fn generator_reaches_the_wasm_memory_oracle() {
    use temen_ir::ValType;
    let int = |t: &ValType| matches!(t, ValType::I32 | ValType::I64);
    let iters: u64 = if cfg!(windows) { 1200 } else { 8000 };
    let mut oracle = 0u32;
    for seed in 0..iters {
        let mut g = seed_gen(seed);
        let m = wasmdiff::gen_module(&mut g);
        let int_entry = m.funcs[0].params.iter().all(int) && m.funcs[0].results.iter().all(int);
        let ro_free = !m.data.iter().any(|d| d.readonly);
        if m.memory.is_some()
            && !wasmdiff::has_float(&m)
            && int_entry
            && ro_free
            && temen_wasm_jit::compile_module(&m).is_ok()
        {
            oracle += 1;
        }
    }
    // 8000 seeds yields ~47 today; a floor well below that catches a generator regression (e.g. the
    // wasm tier tightening its subset, or the entry-sig / float mix drifting) without being flaky.
    let floor = if cfg!(windows) { 3 } else { 25 };
    assert!(
        oracle > floor,
        "too few wasm-emittable memory modules ({oracle}) — the escape-oracle is near-vacuous"
    );
}
