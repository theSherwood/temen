//! libFuzzer target: the machine-code validator's accept-direction mutation test (#2799 layer 2).
//!
//! The input bytes drive the structured generator (`crates/temen/tests/support/irgen.rs`, shared
//! with `diff`/`jit_fuzz`): they synthesize a verifier-valid module, which the JIT compiles with the
//! validator's capture armed. `validator_mutants` then rewrites each accepted function's machine code
//! where a rewrite must make the validator reject it (a guest register in an addressing mode, a
//! narrowed reload, a dropped stack check, a jump into the middle of an instruction, …) and asserts
//! the validator rejects every mutant. A survivor is a soundness hole — code the validator accepts
//! that can escape — and a `unclean` baseline is a completeness gap the fuzzers get for free (an
//! accepted module whose own code the validator already flags). Either panics the target.
//!
//! The coverage the fuzzer guides is the generator's: it explores the lowerings the operators then
//! mutate. Run: `cargo +nightly fuzz run validator_mutants`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/irgen.rs"]
mod irgen;

fuzz_target!(|data: &[u8]| {
    // The validator — and so its mutation engine's entry point — is x86-64 only (aarch64 is #2642);
    // on another host there is nothing to drive, and the entry point is not even compiled.
    #[cfg(target_arch = "x86_64")]
    run(data);
    #[cfg(not(target_arch = "x86_64"))]
    let _ = data;
});

#[cfg(target_arch = "x86_64")]
fn run(data: &[u8]) {
    let mut g = irgen::Gen::from_bytes(data);
    let m = irgen::gen_module(&mut g);
    let report = temen_jit::validator_mutants(&m).expect("the generated module failed to compile");
    // Survivors are the test: a mutant the validator accepts is a soundness hole (accepted ⇒ it can
    // escape). `unclean` — the validator flagging its own unmutated baseline — is a completeness gap,
    // which #2799 scopes out (after #2536 a false rejection is a compile error, which irgen's own
    // harness panics on); it only costs that function its mutants here, so it is not asserted.
    assert!(
        report.survivors.is_empty(),
        "the validator accepted a mutant that must escape:\n{}",
        report.survivors.join("\n")
    );
}
