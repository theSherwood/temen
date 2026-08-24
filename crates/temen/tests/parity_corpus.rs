//! **Phase 2 — the differential corpus is checked against the parity matrix.**
//!
//! The structured generator (`support/irgen.rs`) that feeds the Cranelift interp-vs-JIT differential
//! (`jit_fuzz.rs` / the libFuzzer `diff` target) carries a **hand-maintained set of avoidances** — ops
//! it does not emit because the JIT would fold them to `Unsupported`, which the differential silently
//! treats as a skip (`jit_interp_fuel_agree` returns `false` on `JitError::Unsupported`). That means
//! corpus/JIT drift is currently invisible: if the generator emits an op the JIT declines, coverage
//! quietly drops; if a lowering is removed, modules silently stop being checked.
//!
//! This test makes the [`temen_parity`] manifest the enforced oracle over that hand-maintained set, so
//! the corpus and the matrix cannot drift (INVARIANTS.md #9 — one shared predicate, one definition):
//!
//! * **Soundness (gating):** every instruction and terminator the generator emits must be classified
//!   Cranelift-supported by the manifest. A violation means either the generator drifted to emit a
//!   folded op (dead differential coverage) or a lowering was dropped — both real regressions.
//! * **Coverage (informational):** report the Cranelift-supported op *variants* the corpus never
//!   exercises, so gaps (whole SIMD/control families the differential doesn't reach) stay visible.
//!
//! The manifest is itself pinned to what the backends actually compile by
//! `temen-parity/tests/conformance.rs`, so the chain is closed: corpus ⊆ manifest-supported ⊆
//! backend-supported.

use std::collections::{HashMap, HashSet};
use std::mem::{discriminant, Discriminant};

use temen_ir::{Inst, Module, Terminator};
use temen_parity::{catalog, supports, supports_term, Backend, Focus};

#[path = "support/irgen.rs"]
mod irgen;

/// Seeds swept by the deterministic generator. Generation is pure and cheap (no compilation), so a
/// wide sweep is fast and exercises the generator's full op range.
const SEEDS: u64 = 20_000;

fn for_each_inst<'a>(
    m: &'a Module,
    mut on_inst: impl FnMut(&'a Inst),
    mut on_term: impl FnMut(&'a Terminator),
) {
    for func in &m.funcs {
        for blk in &func.blocks {
            for inst in &blk.insts {
                on_inst(inst);
            }
            on_term(&blk.term);
        }
    }
}

#[test]
fn generated_corpus_stays_within_the_cranelift_supported_set() {
    let mut emitted: HashSet<Discriminant<Inst>> = HashSet::new();

    for seed in 1..=SEEDS {
        let mut g = irgen::Gen::from_seed(seed);
        let m = irgen::gen_module(&mut g);
        for_each_inst(
            &m,
            |inst| {
                assert!(
                    supports(Backend::Cranelift, inst),
                    "irgen emitted `{inst:?}` (seed {seed}) which the manifest marks NOT \
                     Cranelift-supported — the corpus and the parity matrix have drifted \
                     (INVARIANTS #9). Either the generator now emits an op the JIT folds to the \
                     oracle (dead differential coverage), or a lowering was removed.",
                );
                emitted.insert(discriminant(inst));
            },
            |term| {
                assert!(
                    supports_term(Backend::Cranelift, term),
                    "irgen emitted terminator `{term:?}` (seed {seed}) not Cranelift-supported per \
                     the manifest.",
                );
            },
        );
    }

    // ---- Informational coverage report (non-gating) --------------------------------------------
    // Every Cranelift-supported catalog variant → a representative mnemonic. Variant granularity
    // (`Discriminant`): this surfaces whole families the differential never reaches, not per-sub-op
    // holes (e.g. it flags an unexercised `VExtMul`, not specifically `i8x16.mul`).
    let mut supported_variants: HashMap<Discriminant<Inst>, String> = HashMap::new();
    for op in catalog() {
        if let Focus::Inst(inst) = &op.focus {
            if supports(Backend::Cranelift, inst) {
                supported_variants
                    .entry(discriminant(inst))
                    .or_insert_with(|| op.mnemonic.clone());
            }
        }
    }
    let mut unexercised: Vec<&String> = supported_variants
        .iter()
        .filter(|(d, _)| !emitted.contains(d))
        .map(|(_, name)| name)
        .collect();
    unexercised.sort();
    eprintln!(
        "[parity-corpus] {} seeds · {} Cranelift-supported op variants exercised · {} not reached",
        SEEDS,
        emitted.len(),
        unexercised.len(),
    );
    for name in &unexercised {
        eprintln!("  · never generated (variant of): {name}");
    }
}
