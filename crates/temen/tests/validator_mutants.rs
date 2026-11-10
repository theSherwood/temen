//! #2799 layer 2 — the machine-code validator's accept-direction mutation test, as a stable,
//! gating CI test (the libFuzzer twin is `fuzz/fuzz_targets/validator_mutants.rs`).
//!
//! For a fixed corpus of irgen seeds, each synthesizes a verifier-valid module, which the JIT
//! compiles with the validator's capture armed; `validator_mutants` then rewrites each accepted
//! function's machine code where a rewrite must make the validator reject it, and reports the
//! mutants it accepted (`survivors` — a soundness hole) and the findings it raised on unmutated
//! code (`unclean` — a completeness gap). The validator must reject every mutant and accept every
//! unmutated function. The claim it carries is *accepted ⇒ the function cannot escape* (DESIGN §2a).
//!
//! The validator is x86-64 only (aarch64 is #2642), and it models the SysV convention, so a Win64
//! host compile's trampolines read their tagged arguments in the wrong registers (#2820) — hence
//! `not(windows)`, until that lands.
#![cfg(all(target_arch = "x86_64", not(windows)))]

#[path = "support/irgen.rs"]
mod irgen;

use std::collections::BTreeMap;

use irgen::{gen_module, Gen};

/// The engine's run over one corpus of seeds: the mutants each operator made, the operators that
/// made none (they must be live somewhere in the corpus), the survivors, and the unclean baselines.
struct Sweep {
    made: BTreeMap<&'static str, usize>,
    unclean: Vec<String>,
    survivors: Vec<String>,
}

fn sweep(seeds: impl Iterator<Item = u64>, mix: u64) -> Sweep {
    let mut made: BTreeMap<&'static str, usize> = BTreeMap::new();
    let (mut unclean, mut survivors) = (Vec::new(), Vec::new());
    for seed in seeds {
        let mut g = Gen::from_seed(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ mix);
        let m = gen_module(&mut g);
        let r = temen_jit::validator_mutants(&m).expect("the generated module failed to compile");
        for u in r.unclean {
            unclean.push(format!("seed {seed}: {u}"));
        }
        for s in r.survivors {
            survivors.push(format!("seed {seed}: {s}"));
        }
        for (k, v) in r.made {
            *made.entry(k).or_default() += v;
        }
    }
    Sweep {
        made,
        unclean,
        survivors,
    }
}

/// Over a fixed seed corpus: every mutant is rejected and every baseline is clean. The mutation is
/// thorough — one validator run per mutation site — so the corpus is kept modest; the libFuzzer
/// target explores unboundedly from the same generator.
///
/// Which operators fire, and how many mutants each makes, is **not** asserted: the operators match
/// shapes in the compiled bytes, and the bytes depend on the host's Cranelift lowering (which
/// `cranelift_native` tunes to the runner's CPU), so the window/D63/jump-table operators that fire
/// freely on one host make nothing on another over the same seeds. The per-operator counts are
/// printed under `--nocapture` instead, and every operator's liveness is pinned locally (the
/// mutation tables in the PR) and explored by the libFuzzer twin. The gate is the two invariants.
#[test]
fn every_mutant_is_rejected() {
    let n: u64 = std::env::var("MUT_SEEDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let s = sweep(0..n, 0x2799_2799);
    for (op, made) in &s.made {
        eprintln!("made {op}: {made}");
    }
    assert!(
        s.unclean.is_empty(),
        "the validator flags its own accepted code ({} findings):\n{}",
        s.unclean.len(),
        s.unclean.join("\n")
    );
    assert!(
        s.survivors.is_empty(),
        "the validator accepted a mutant that must escape ({} survivors):\n{}",
        s.survivors.len(),
        s.survivors.join("\n")
    );
    // A weak host-robust tripwire: the engine made *some* mutants (every host has calls, frames and
    // branches, so the frame/call/jump operators always fire). A total of zero would mean the engine
    // or the capture broke, not a host difference.
    let total: usize = s.made.values().sum();
    assert!(
        total > 0,
        "the mutation engine made no mutants at all over {n} seeds"
    );
}

/// The jit_fuzz seed corpus too (a different generator mix, the one the interp↔JIT differential
/// uses), so the mutation test covers the same lowerings that differential explores.
#[test]
fn every_mutant_is_rejected_on_the_differential_corpus() {
    let s = sweep(0..1000, 0xD1CE_F00D);
    assert!(
        s.unclean.is_empty(),
        "unclean baseline:\n{}",
        s.unclean.join("\n")
    );
    assert!(
        s.survivors.is_empty(),
        "survivors:\n{}",
        s.survivors.join("\n")
    );
}
