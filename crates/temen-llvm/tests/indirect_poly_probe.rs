//! Probe: how polymorphic are the `call_indirect` sites in real interpreters?
//!
//! This gates temen-peval "approach A" (bounded-target residualization of a *dynamic*-index
//! `call_indirect`): enumerate the signature-matching target functions, branch among them, fold
//! each arm. It only pays if the dispatch signatures are **low-polymorphism** — a handful of
//! matching functions, not hundreds. This test measures that on the lua on-ramp module (and
//! quickjs if the toolchain/network is present) and prints a histogram. Read-only; no specializer
//! changes involved.
//!
//! Run: `cargo test --manifest-path crates/temen-llvm/Cargo.toml --test indirect_poly_probe -- --nocapture`

use std::collections::BTreeMap;

use temen_ir::{Func, Inst, Module, Terminator};

/// A hashable/orderable signature key: `"[params]->[results]"` via Debug.
fn sig_key(params: &[temen_ir::ValType], results: &[temen_ir::ValType]) -> String {
    format!("{params:?}->{results:?}")
}

/// #922: a call site's `ty`/`sig` is now a `u32` index into the module type section, not an inline
/// `FuncType`. Resolve it back to its signature key through `m.types`.
fn site_sig_key(m: &Module, ty: u32) -> String {
    match &m.types[ty as usize] {
        temen_ir::TypeEntry::Func(f) => sig_key(&f.params, &f.results),
        other => panic!("call_indirect ty {ty} names a non-Func type entry: {other:?}"),
    }
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
                    sites.push(site_sig_key(m, *ty));
                }
            }
            if let Terminator::ReturnCallIndirect { ty, .. } = &b.term {
                sites.push(site_sig_key(m, *ty));
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
    let cum_le = |thresh: usize| {
        buckets
            .iter()
            .filter(|(k, _)| **k <= thresh)
            .map(|(_, v)| *v)
            .sum::<usize>()
    };
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
    let fixtures = ["lua_eval.ll", "lua_all.ll", "lua_testsuite.ll"];
    for fx in fixtures {
        let path = format!("{}/tests/fixtures/lua/{fx}", env!("CARGO_MANIFEST_DIR"));
        let t = temen_llvm::translate_ll_path(&path).expect("translate lua fixture");
        report(fx, &t.module);
    }
}

/// For each eligible site, the candidate functions' **relative slot span** (max index − min index)
/// — the size an offset-by-min `br_table` would need. A small span means the same-signature targets
/// cluster in the function index space, so a jump table is cheap even at high absolute indices.
fn candidate_spans(m: &Module, cap: usize) -> Vec<usize> {
    let mut by_sig: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in m.funcs.iter().enumerate() {
        by_sig.entry(func_sig_key(f)).or_default().push(i);
    }
    let mut spans = Vec::new();
    for f in &m.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Inst::CallIndirect { ty, .. } = inst {
                    let key = site_sig_key(m, *ty);
                    if let Some(idxs) = by_sig.get(&key) {
                        if (1..=cap).contains(&idxs.len()) {
                            spans.push(idxs.last().unwrap() - idxs.first().unwrap());
                        }
                    }
                }
            }
        }
    }
    spans
}

/// How many functions contain at least one dynamic-index `call_indirect` site whose signature
/// fan-out is within `cap` (i.e. approach A would rewrite it).
fn functions_with_eligible_site(m: &Module, cap: usize) -> (usize, usize) {
    let pop = signature_population(m);
    let mut funcs_hit = 0usize;
    let mut sites_hit = 0usize;
    for f in &m.funcs {
        let mut hit = false;
        for b in &f.blocks {
            for inst in &b.insts {
                if let Inst::CallIndirect { ty, .. } = inst {
                    let n = pop.get(&site_sig_key(m, *ty)).copied().unwrap_or(0);
                    if (1..=cap).contains(&n) {
                        sites_hit += 1;
                        hit = true;
                    }
                }
            }
        }
        if hit {
            funcs_hit += 1;
        }
    }
    (funcs_hit, sites_hit)
}

/// Approach A on the *real* Lua module: at several caps, rewrite every eligible dynamic
/// `call_indirect`, re-verify, and report how many sites fire and the code-size cost. This measures
/// feasibility + blowup on real on-ramp code (the full Futamura *speedup* additionally needs a
/// program-as-constant harness, which does not exist yet).
#[test]
fn lua_approach_a_fires_and_verifies() {
    let path = format!(
        "{}/tests/fixtures/lua/lua_eval.ll",
        env!("CARGO_MANIFEST_DIR")
    );
    let m = temen_llvm::translate_ll_path(&path)
        .expect("translate")
        .module;
    temen_verify::verify_module(&m).expect("source lua module verifies");
    let base_bytes = temen_encode::encode_module(&m).len();
    let base_blocks: usize = m.funcs.iter().map(|f| f.blocks.len()).sum();

    println!("\n=== approach A on lua_eval.ll ===");
    println!(
        "baseline: {} funcs, {base_blocks} blocks, {base_bytes} encoded bytes",
        m.funcs.len()
    );
    for cap in [4usize, 8, 16] {
        let spans = candidate_spans(&m, cap);
        let le = |t: usize| spans.iter().filter(|&&s| s <= t).count();
        println!(
            "cap={cap:>2} candidate relative-span: {} sites | span<=32: {} | <=256: {} | max span {:?}",
            spans.len(), le(32), le(256), spans.iter().max()
        );
    }
    for cap in [4usize, 8, 16, 32] {
        let (funcs_hit, sites_hit) = functions_with_eligible_site(&m, cap);
        let mut low = temen_peval::lower_indirect_dispatch(&m, cap);
        // The lowering strips name-bearing link metadata (it must not clone a `String`, being itself
        // translated to temen-IR). Restore it from the source — function indices are unchanged, so the
        // exports/imports still resolve — for a faithful re-verify and an apples-to-apples size cost.
        low.imports = m.imports.clone();
        low.exports = m.exports.clone();
        low.data_exports = m.data_exports.clone();
        low.data_funcrefs = m.data_funcrefs.clone();
        low.impl_exports = m.impl_exports.clone();
        low.types = m.types.clone();
        temen_verify::verify_module(&low)
            .unwrap_or_else(|e| panic!("lowered lua (cap={cap}) must re-verify: {e:?}"));
        let bytes = temen_encode::encode_module(&low).len();
        let blocks: usize = low.funcs.iter().map(|f| f.blocks.len()).sum();
        println!(
            "cap={cap:>2}: {sites_hit:>3} sites in {funcs_hit:>3} funcs fire | \
             blocks {base_blocks}->{blocks} (+{:.1}%) | bytes {base_bytes}->{bytes} (+{:.1}%)",
            100.0 * (blocks as f64 - base_blocks as f64) / base_blocks as f64,
            100.0 * (bytes as f64 - base_bytes as f64) / base_bytes as f64,
        );
    }
}
