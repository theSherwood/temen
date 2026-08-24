//! **Subset / lazy-emit win measurement.** The single-shot module wasm-JIT emits every in-subset
//! function up front; the profiler showed a real run touches ~12% of them. This decodes a real guest,
//! reads the touched-function set (from `callprof.mjs` → `/tmp/touched-*.json`), and compares
//! whole-module emit ([`compile_module_reactor`]) against emitting only the touched set
//! ([`compile_module_reactor_keep`], the rest falling back to the interp) — reporting the emit TIME
//! and module SIZE reduction, i.e. exactly what lazy per-function emit would save on startup + memory.
//! (It does NOT change RUN throughput: the hot touched functions are emitted either way.) Run:
//!   cargo run --release -p temen-wasm-jit --example lazybench -- <guest.temen> <touched.json>

use std::time::Instant;
use temen_wasm_jit::{compile_module_reactor, compile_module_reactor_keep};

fn parse_indices(s: &str) -> Vec<usize> {
    s.trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect()
}

fn emit_time(f: impl Fn() -> (Vec<u8>, Vec<bool>)) -> (f64, usize, usize) {
    // Min over reps; report (ms, wasm bytes, #emitted functions).
    let mut best = f64::MAX;
    let mut bytes = 0;
    let mut nfun = 0;
    for _ in 0..7 {
        let t = Instant::now();
        let (wasm, emitted) = f();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        if ms < best {
            best = ms;
        }
        bytes = wasm.len();
        nfun = emitted.iter().filter(|&&b| b).count();
    }
    (best, bytes, nfun)
}

fn main() {
    let mut args = std::env::args().skip(1);
    let guest = args
        .next()
        .expect("usage: lazybench <guest.temen> <touched.json>");
    let touched_path = args
        .next()
        .expect("usage: lazybench <guest.temen> <touched.json>");
    let m = temen_encode::decode_module(&std::fs::read(&guest).unwrap()).expect("decode");
    let n = m.funcs.len();
    let touched = parse_indices(&std::fs::read_to_string(&touched_path).unwrap());
    let mut keep = vec![false; n];
    for &i in &touched {
        if i < n {
            keep[i] = true;
        }
    }

    let (whole_ms, whole_bytes, whole_fns) =
        emit_time(|| compile_module_reactor(&m, 0, false).expect("whole-module emit"));
    let (sub_ms, sub_bytes, sub_fns) =
        emit_time(|| compile_module_reactor_keep(&m, 0, &keep, false).expect("subset emit"));

    println!("\n=== {guest}  (touched set: {} funcs) ===", touched.len());
    println!("                       whole-module      subset (touched)     reduction");
    println!(
        "  functions emitted    {whole_fns:>10}      {sub_fns:>14}      {:.1}×",
        whole_fns as f64 / sub_fns.max(1) as f64
    );
    println!(
        "  emitted wasm         {:>7.0} KiB      {:>10.0} KiB      {:.1}×",
        whole_bytes as f64 / 1024.0,
        sub_bytes as f64 / 1024.0,
        whole_bytes as f64 / sub_bytes.max(1) as f64
    );
    println!(
        "  emit time            {whole_ms:>7.1} ms      {sub_ms:>10.1} ms      {:.1}×",
        whole_ms / sub_ms.max(f64::MIN_POSITIVE)
    );
    println!(
        "  → subset emit produces the {sub_fns}-function hot core; the other {} reachable in-subset\n    functions fall back to the interpreter (never called on this run, so no RUN cost).",
        whole_fns.saturating_sub(sub_fns)
    );
}
