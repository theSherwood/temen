//! **Tier-up over-emission analyzer** (break-even de-risking, Step: instrument the trigger). The
//! single-shot module wasm-JIT emits **every in-subset function** of a guest up front; the profiler
//! showed runtime concentrates in ~one giant function. This decodes a real guest and quantifies that
//! gap statically: how many functions the whole-module path emits vs how concentrated the *code* is,
//! and — for the hot-function question — how many carry a **loop** (a back-edge: a branch to a block
//! index `<= its own`). A loop-bearing function can be "called once but hot" (needs back-edge hotness /
//! OSR to catch); a loop-free one is only hot if *called* often (a call-count gate suffices). Run:
//!   cargo run --release -p svm-wasm-jit --example tierup_analyze -- <guest.svmb> [more.svmb ...]

use svm_ir::{Module, Terminator};
use svm_wasm_jit::analyze;

fn func_insts(m: &Module, fi: usize) -> usize {
    m.funcs[fi].blocks.iter().map(|b| b.insts.len()).sum()
}

/// A function has a loop iff some block branches back to an index `<= its own` (the fuel back-edge).
fn has_loop(m: &Module, fi: usize) -> bool {
    m.funcs[fi].blocks.iter().enumerate().any(|(k, b)| {
        let k = k as u32;
        match &b.term {
            Terminator::Br { target, .. } => *target <= k,
            Terminator::BrIf {
                then_blk, else_blk, ..
            } => *then_blk <= k || *else_blk <= k,
            Terminator::BrTable {
                targets, default, ..
            } => targets.iter().any(|(t, _)| *t <= k) || default.0 <= k,
            _ => false,
        }
    })
}

fn analyze_guest(path: &str) {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let m = match svm_encode::decode_module(&bytes) {
        Ok(m) => m,
        Err(e) => {
            println!("\n=== {path} ===\n  decode failed: {e:?} (needs linking?) — skipped");
            return;
        }
    };
    let a = analyze(&m);
    let n = m.funcs.len();

    // Per-function size, restricted to what the whole-module path actually emits (in-subset).
    let mut sizes: Vec<(usize, usize, bool)> = (0..n)
        .filter(|&i| a.in_subset[i])
        .map(|i| (i, func_insts(&m, i), has_loop(&m, i)))
        .collect();
    sizes.sort_by(|x, y| y.1.cmp(&x.1)); // largest first

    let emitted = sizes.len();
    let total_insts: usize = sizes.iter().map(|(_, s, _)| s).sum();
    let loop_funcs = sizes.iter().filter(|(_, _, l)| *l).count();
    let leaves = a.interp_leaf.iter().filter(|&&x| x).count();

    println!("\n=== {path} ===");
    println!(
        "  {n} funcs total · {emitted} in-subset (whole-module emit) · {leaves} interp-leaf · {} out-of-subset",
        n - emitted - leaves
    );
    println!(
        "  emitted code: {} IR insts across {emitted} funcs · {loop_funcs} carry a loop ({}%)",
        total_insts,
        if emitted > 0 {
            100 * loop_funcs / emitted
        } else {
            0
        }
    );
    // Concentration: what fraction of emitted code is the top-K biggest functions?
    let cum = |k: usize| -> f64 {
        let s: usize = sizes.iter().take(k).map(|(_, s, _)| s).sum();
        100.0 * s as f64 / total_insts.max(1) as f64
    };
    println!(
        "  code concentration: top-1 fn = {:.0}% of emitted insts · top-5 = {:.0}% · top-20 = {:.0}%",
        cum(1),
        cum(5),
        cum(20)
    );
    println!("  biggest in-subset functions (the whole-module path emits ALL of them up front):");
    for (i, s, l) in sizes.iter().take(6) {
        println!(
            "    f{i:<4} {s:>7} insts  {}",
            if *l { "loop" } else { "straight-line" }
        );
    }
    println!(
        "  → whole-module emit produces {emitted} functions; if runtime concentrates in the top few\n    (per the profiler), a function-granularity + call/back-edge-gated trigger emits a small\n    fraction of this, bounding the cost of any wrong guess to one function."
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tierup_analyze <guest.svmb> [more.svmb ...]");
        std::process::exit(2);
    }
    for path in &args {
        analyze_guest(path);
    }
}
