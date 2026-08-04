//! Probe: how polymorphic are the `call_indirect` sites in real interpreters?
//!
//! This gates svm-peval "approach A" (bounded-target residualization of a *dynamic*-index
//! `call_indirect`): enumerate the signature-matching target functions, branch among them, fold
//! each arm. It only pays if the dispatch signatures are **low-polymorphism** — a handful of
//! matching functions, not hundreds. This test measures that on the lua on-ramp module (and
//! quickjs if the toolchain/network is present) and prints a histogram. Read-only; no specializer
//! changes involved.
//!
//! Run: `cargo test --manifest-path crates/svm-llvm/Cargo.toml --test indirect_poly_probe -- --nocapture`

use std::collections::BTreeMap;

use svm_ir::{Func, Inst, Module, Terminator};

/// A hashable/orderable signature key: `"[params]->[results]"` via Debug.
fn sig_key(params: &[svm_ir::ValType], results: &[svm_ir::ValType]) -> String {
    format!("{params:?}->{results:?}")
}

fn func_sig_key(f: &Func) -> String {
    sig_key(&f.params, &f.results)
}

/// `(sig -> number of module functions with that exact signature)`.
fn signature_population(m: &Module) -> BTreeMap<String, usize> {
    let mut pop = BTreeMap::new();
    for f in &m.funcs {
        *pop.entry(func_sig_key(f)).or_insert(0) += 1;
    }
    pop
}

/// Every `call_indirect` / `return_call_indirect` site's demanded signature key. One entry per
/// site (duplicates kept, so a hot signature used at many sites shows its true site count).
fn indirect_site_sigs(m: &Module) -> Vec<String> {
    let mut sites = Vec::new();
    for f in &m.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Inst::CallIndirect { ty, .. } = inst {
                    sites.push(sig_key(&ty.params, &ty.results));
                }
            }
            if let Terminator::ReturnCallIndirect { ty, .. } = &b.term {
                sites.push(sig_key(&ty.params, &ty.results));
            }
        }
    }
    sites
}

fn report(name: &str, m: &Module) {
    let pop = signature_population(m);
    let sites = indirect_site_sigs(m);
    let n_funcs = m.funcs.len();

    // Bucket sites by the polymorphism (population of their demanded signature).
    let mut buckets: BTreeMap<usize, usize> = BTreeMap::new(); // fan-out -> #sites
    let mut unmatched = 0usize;
    for s in &sites {
        match pop.get(s) {
            Some(&n) => *buckets.entry(n).or_insert(0) += 1,
            None => unmatched += 1,
        }
    }

    println!("\n================ {name} ================");
    println!("functions: {n_funcs}");
    println!("distinct signatures: {}", pop.len());
    println!("call_indirect sites: {}", sites.len());
    println!("  (sites whose signature matches NO function: {unmatched})");

    println!("\nsite fan-out histogram (targets per site -> #sites):");
    let mut cum_le = |thresh: usize| buckets.iter().filter(|(k, _)| **k <= thresh).map(|(_, v)| *v).sum::<usize>();
    for (fanout, count) in &buckets {
        println!("  {fanout:>5} targets : {count} site(s)");
    }
    let total_sites = sites.len().max(1);
    for cap in [1usize, 2, 4, 8, 16, 32] {
        let n = cum_le(cap);
        println!(
            "  fan-out <= {cap:>2}: {n}/{total_sites} sites ({:.1}%) would fire under a cap of {cap}",
            100.0 * n as f64 / total_sites as f64
        );
    }

    // The heaviest signatures by population — these are the ones that would blow up exhaustive
    // residualization if they appear at a dynamic-index site.
    let mut by_pop: Vec<(&String, &usize)> = pop.iter().collect();
    by_pop.sort_by(|a, b| b.1.cmp(a.1));
    println!("\ntop signatures by function population:");
    for (sig, n) in by_pop.iter().take(6) {
        let at_sites = sites.iter().filter(|s| *s == *sig).count();
        println!("  {n:>4} funcs, used at {at_sites:>3} indirect site(s): {sig}");
    }
}

#[test]
fn lua_indirect_polymorphism() {
    let fixtures = [
        "lua_eval.ll",
        "lua_all.ll",
        "lua_testsuite.ll",
    ];
    for fx in fixtures {
        let path = format!("{}/tests/fixtures/lua/{fx}", env!("CARGO_MANIFEST_DIR"));
        let t = svm_llvm::translate_ll_path(&path).expect("translate lua fixture");
        report(fx, &t.module);
    }
}
