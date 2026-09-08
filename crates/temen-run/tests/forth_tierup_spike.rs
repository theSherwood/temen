//! #1233 measurement spike (NOT a gate — `#[ignore]`d, run manually):
//!
//!   cargo test -p temen-run --test forth_tierup_spike -- --ignored --nocapture
//!
//! The question behind #1233 is whether running Forth's colon-definition **words** on an emitted tier
//! (instead of interpreted) is worth a cross-axis dispatch change (see the ticket). Rather than build
//! the wasm-JIT `call.dyn → JitInvoke` path first, this measures the *ceiling* of that win directly:
//! the same hot-word-loop program on the **bytecode interpreter** (words interpreted, today's
//! `jit: false` reality) vs the **Cranelift JIT** (`Backend::Jit` — which already compiles every §22
//! word unit and dispatches `call.dyn` to it natively, i.e. the words run emitted). The interp/emit
//! ratio is the best case the wasm-JIT tier could approach.
//!
//! Fixed costs (kernel compile, program compile) are cancelled by a **baseline subtraction**: time a
//! loop of `N` and of `2N` iterations; `t(2N) - t(N)` isolates `N` iterations of steady-state
//! execution. Two word weights bracket the per-call-overhead spectrum: a trivial word (`dup *`, where
//! per-call cost dwarfs the body) and a heavier word (an inner loop, where the body dominates).

#![cfg(all(unix, target_arch = "x86_64"))]

use std::time::Instant;
use temen_run::{instantiate, Backend, RunConfig};

/// Run `program` on `backend`, returning wall time. Fresh instance each call (the kernel recompiles,
/// but that's a fixed cost the N-vs-2N subtraction removes).
fn time_run(program: &str, backend: Backend) -> std::time::Duration {
    let m = temen_text::parse_module(include_str!("../demos/forth/forth.temt")).unwrap();
    let inst = instantiate(m).unwrap();
    let cfg = RunConfig {
        stdin: program.as_bytes().to_vec(),
        ..RunConfig::default()
    };
    let t0 = Instant::now();
    let run = inst.run(backend, &cfg).expect("run");
    let dt = t0.elapsed();
    // Touch stdout so the accumulator can't be dead-code-eliminated.
    assert!(!run.stdout.is_empty(), "bench must print its accumulator");
    dt
}

/// A hot loop that calls word `w` `count` times, accumulating the result (so nothing is DCE'd), then
/// prints the sum once.
fn prog(wdef: &str, count: u64) -> String {
    format!("{wdef}: bench ( -- s ) 0 {count} 0 do i callee + loop ;\nbench . cr\n")
}

/// Per-iteration steady-state cost (ns) on `backend` for word `wdef`, via the N/2N baseline subtraction.
fn per_iter_ns(wdef: &str, n: u64, backend: Backend) -> f64 {
    // Warm once (page-in, allocator) then take the min of a few samples to cut noise.
    let sample = |count: u64| -> std::time::Duration {
        (0..3)
            .map(|_| time_run(&prog(wdef, count), backend))
            .min()
            .unwrap()
    };
    let t1 = sample(n);
    let t2 = sample(2 * n);
    (t2.saturating_sub(t1)).as_nanos() as f64 / n as f64
}

fn report(label: &str, wdef: &str, n: u64) {
    let interp = per_iter_ns(wdef, n, Backend::Bytecode);
    let jit = per_iter_ns(wdef, n, Backend::Jit);
    let ratio = if jit > 0.0 { interp / jit } else { f64::NAN };
    println!(
        "  {label:<26} interp {interp:8.1} ns/call   jit {jit:8.1} ns/call   speedup {ratio:5.2}x"
    );
}

#[test]
#[ignore = "measurement spike for #1233; run with --ignored --nocapture"]
fn forth_word_emit_speedup_ceiling() {
    println!("\n#1233 spike — interpreted vs Cranelift-JIT Forth words (per call.dyn, baseline-subtracted):");
    // Trivial word: the body is one multiply; per-call dispatch overhead dominates.
    report(
        "trivial (dup *)",
        ": callee ( n -- n ) dup * ;\n",
        2_000_000,
    );
    // Heavier word: an inner 8-iteration accumulate — real compute per call.
    report(
        "heavy (8-iter inner loop)",
        ": callee ( n -- n ) 0 swap 8 0 do over + loop nip ;\n",
        400_000,
    );
    println!(
        "\n  Reading: the JIT column is the CEILING the wasm-JIT tier (#1233 Path 2) could approach.\n\
         \x20 A speedup near 1.0x means emitting that word shape isn't worth the dispatch change; a\n\
         \x20 large speedup means a hot-loop Forth *program* would benefit (the tour card, dominated by\n\
         \x20 one-shot compiles, is closer to the trivial/near-1.0 end).\n"
    );
}
