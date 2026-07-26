//! libFuzzer target: the generic optimizer (SCCP + the cleanup fixpoint, `OPT.md` Phase 2) is
//! semantics-preserving. The input bytes drive the structured generator (shared with `diff` /
//! `jit_fuzz`) to synthesize a verifier-valid module; `optimize_module` must (1) produce IR that
//! **re-verifies** — an optimizer bug is a clean verify error, never an escape — and (2) compute the
//! same result / trap as the original on the reference interpreter.
//!
//! Run: `cargo +nightly fuzz run opt_sccp`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/svm/tests/support/irgen.rs"]
mod irgen;

use svm_interp::Trap;
use svm_verify::verify_module;

fuzz_target!(|data: &[u8]| {
    let mut g = irgen::Gen::from_bytes(data);
    let m = irgen::gen_module(&mut g);
    if m.funcs.is_empty() || verify_module(&m).is_err() {
        return; // only optimize verifier-valid modules
    }

    let opt = svm_opt::optimize_module(&m);
    verify_module(&opt).expect("optimized module must re-verify");

    let args = irgen::gen_args(&mut g, &m.funcs[0].params);
    let mut fuel_a = 5_000_000u64;
    let mut fuel_b = 5_000_000u64;
    let r0 = svm_interp::run(&m, 0, &args, &mut fuel_a);
    let r1 = svm_interp::run(&opt, 0, &args, &mut fuel_b);

    // A smaller residual can finish where the original hit the fuel ceiling (or vice versa); that is
    // not a miscompile, so only compare when neither run ran out of fuel.
    if matches!(r0, Err(Trap::OutOfFuel)) || matches!(r1, Err(Trap::OutOfFuel)) {
        return;
    }
    // NaN-aware result comparison: the IR pins no NaN bit-pattern (a NaN-producing op may emit a
    // different-but-still-NaN payload after optimization), and `f32`/`f64` `PartialEq` makes
    // `NaN != NaN` — so a bare `assert_eq!` false-positives whenever a result is NaN, even when both
    // runs produced the identical value. Reuse the same value equivalence the JIT differential uses
    // (`irgen::values_equal`: bit-exact for non-NaN scalars, all-NaN-equal for NaN).
    let equiv = match (&r0, &r1) {
        (Ok(a), Ok(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| irgen::values_equal(x, y))
        }
        (Err(a), Err(b)) => a == b,
        _ => false,
    };
    assert!(
        equiv,
        "optimize_module changed observable behavior\n  orig: {r0:?}\n   opt: {r1:?}"
    );
});
