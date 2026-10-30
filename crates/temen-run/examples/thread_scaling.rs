//! **Intra-guest scaling — one guest, many threads.** The companion to `parallel` (which runs many
//! *independent* guests): here a **single** guest spawns `W` worker threads via `thread.spawn` and
//! joins them, and we ask whether the runtime spreads them across cores — i.e. does the §12 scheduler
//! give one guest near-`W`× speedup? On the JIT, `thread.spawn` maps to a real OS thread per guest
//! thread (`os_thread_rt`), so this exercises actual parallel execution (the *interpreters*
//! model-check thread interleavings on one OS thread, so they don't apply). Driven through the
//! embedding runtime `run_powerbox` (the concurrent path takes the per-domain `Mutex<Host>`).
//! Runtime-only: **no libLLVM**.
//!
//! Each worker runs a serial `xorshift64*` loop (pure ALU), plus `S` private 8-byte stores per turn
//! into its own region, so the workers share no data. For width `W` we time the whole guest at
//! two per-worker iteration counts and take the slope `(T(LARGE) − T(SMALL))/Δn` — the wall time per
//! worker-iteration, with compile + spawn/join overhead subtracted out. If the workers run in
//! parallel that slope stays flat as `W` grows (each on its own core); if they serialized it would
//! grow ∝ `W`. **efficiency = slope(1) / slope(W)** (100% = perfect parallelism to that width).
//!
//! Each width runs metered (the default fuel grant, so the code carries fuel checks) and unmetered.
//! The stores are what expose a cost the workers share without sharing data (#2488): with a few
//! dozen between two fuel checks, a check's load of a cell another core wrote last waits for that
//! line. A pure-register loop (`S` = 0) hides it, since each load is served from the core's own
//! pending store.
//!
//! Run: cargo run -p temen-run --release --example thread_scaling [S]

use std::time::Instant;

use temen_run::{run_powerbox_cfg, Limits, Outcome, Value};

/// Turns per worker at the two points of the fit, for `S` = 0; more stores run fewer turns.
const SMALL: i64 = 1_000_000;
const LARGE: i64 = 200_000_000;

/// Worker = func 1: `(i64 sp, i64 n) -> i64`, xorshift64* `n` times, storing the state `stores`
/// times per turn into the 4 KiB below its `sp` (serial; no data shared with another worker).
fn worker(stores: usize) -> String {
    let mut s = String::from(
        r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vx = i64.const -7046029254386353131
  vi = i64.const 0
  vpage = i64.const 4096
  vd = i64.sub vsp vpage
  br 1(vi, vx, vn, vd)
}
block 1 (va: i64, vb: i64, vc: i64, vd: i64) {
  vs1 = i64.const 13
  vt1 = i64.shl vb vs1
  vb1 = i64.xor vb vt1
  vs2 = i64.const 7
  vt2 = i64.shr_u vb1 vs2
  vb2 = i64.xor vb1 vt2
  vs3 = i64.const 17
  vt3 = i64.shl vb2 vs3
  vb3 = i64.xor vb2 vt3
"#,
    );
    for k in 0..stores {
        s.push_str(&format!("  i64.store vd vb3 offset={}\n", 8 * k));
    }
    s.push_str(
        r#"  vone = i64.const 1
  va1 = i64.add va vone
  vlt = i64.lt_s va1 vc
  br_if vlt 1(va1, vb3, vc, vd) 2(vb3)
}
block 2 (vr: i64) {
  return vr
  }
}
"#,
    );
    s
}

/// A powerbox guest (entry func 0: `() -> (i32)`) that spawns `w` copies of the worker — each on
/// its own 64 KiB data-stack region — runs `n` iterations apiece, joins all, and returns 0.
fn guest(w: usize, n: i64, stores: usize) -> String {
    let mut s = String::from("memory 22\nfunc () -> (i32) {\nblock 0 () {\n");
    s.push_str(&format!("  vn = i64.const {n}\n"));
    for i in 0..w {
        let sp = (i as i64 + 1) * 0x10000; // 64 KiB apart, all inside the 4 MiB window
        s.push_str(&format!(
            "  vsp{i} = i64.const {sp}\n  vh{i} = thread.spawn 1 vsp{i} vn\n"
        ));
    }
    for i in 0..w {
        s.push_str(&format!("  vj{i} = thread.join vh{i}\n"));
    }
    s.push_str("  vrc = i32.const 0\n  return vrc\n  }\n}\n");
    s.push_str(&worker(stores));
    s
}

/// Min-of-reps wall time (ns) of the whole `w`-thread guest doing `n` iterations per worker.
fn wall(w: usize, n: i64, stores: usize, metered: bool) -> f64 {
    let m = temen_text::parse_module(&guest(w, n, stores)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let limits = Limits {
        fuel: (!metered).then_some(u64::MAX),
        ..Limits::default()
    };
    let mut best = f64::MAX;
    for _ in 0..4 {
        let t = Instant::now();
        let run = run_powerbox_cfg(&m, b"", &[], &[], limits.clone()).expect("run_powerbox");
        let e = t.elapsed().as_nanos() as f64;
        assert_eq!(
            run.outcome,
            Outcome::Returned(vec![Value::I32(0)]),
            "guest failed"
        );
        best = best.min(e);
    }
    best
}

/// Slope = wall-ns per worker-iteration (compile + spawn/join subtracted via the two-point fit).
fn slope(w: usize, stores: usize, metered: bool) -> f64 {
    let scale = 1 + stores as i64 / 4;
    let (small, large) = (SMALL / scale, LARGE / scale);
    (wall(w, large, stores, metered) - wall(w, small, stores, metered)) / (large - small) as f64
}

fn main() {
    let stores: usize = std::env::args()
        .nth(1)
        .map_or(0, |s| s.parse().expect("S: stores per turn"));
    let cores = std::thread::available_parallelism().map_or(1, |c| c.get());
    let mut counts = vec![1usize];
    let mut w = 2;
    while w < cores {
        counts.push(w);
        w *= 2;
    }
    counts.push(cores);
    counts.push(cores * 2); // one oversubscribed point

    println!(
        "host: {cores} logical cores | one guest, W threads via thread.spawn (JIT/os_thread_rt), \
         {stores} stores per turn\n\
         efficiency = slope(1)/slope(W); speedup = W × efficiency; 100% = perfect parallelism\n"
    );
    let mut at_cores = [0.0; 2];
    for (i, metered) in [true, false].into_iter().enumerate() {
        println!("{}", if metered { "metered" } else { "unmetered" });
        println!(
            "{:>8} {:>16} {:>14} {:>12}",
            "threads", "agg Miter/s", "speedup", "efficiency"
        );
        let base = slope(1, stores, metered);
        for &w in &counts {
            let s = slope(w, stores, metered);
            if w == cores {
                at_cores[i] = s;
            }
            let eff = base / s; // ≤ 1
            let agg = w as f64 / s * 1e9 / 1e6; // (w workers × iters/ns) → iters/s → Miter/s
            println!(
                "{w:>8} {:>16.1} {:>13.2}x {:>11.0}%",
                agg,
                w as f64 * eff,
                eff * 100.0
            );
        }
        println!();
    }
    println!(
        "metered / unmetered at {cores} threads: {:.2}x",
        at_cores[0] / at_cores[1]
    );
    println!(
        "\n(One guest's `thread.spawn` workers are scheduled onto real OS threads by the JIT runtime\n \
         (os_thread_rt), so a single sandbox uses multiple cores. Near-linear speedup to the core\n \
         count means the scheduler/host adds no serialization; the plateau past it is SMT/oversubscription.\n \
         The interpreters model-check thread interleavings on one OS thread, so they're excluded here.)"
    );
}
