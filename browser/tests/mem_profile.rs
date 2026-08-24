//! The W3 **run-mode memory profiler** in the cdylib (INTERACTIVE_EMBEDDING.md slice 4):
//! `temen_mem_profile` instruments a compute module with the mem-hook pass, runs it on the bytecode
//! engine feeding a `MemModel`, and stashes the stats JSON. Pins: (a) the hook-fed model equals a
//! **sink-fed** model on the same guest, stats-for-stats — the cdylib's local hook decode can't
//! drift from the sink vocabulary unnoticed; (b) the strided-vs-sequential miss ordering through
//! the entry; (c) a manifest-carrying module is refused fail-closed (the temen-run slot-0 rule).
//!
//! One `#[test]`: the entry's stash is a main-thread singleton.

use std::sync::{Arc, Mutex};
use temen_browser::{temen_mem_profile, temen_mem_profile_stats_len, temen_mem_profile_stats_ptr};
use temen_dap::models::{MemModel, MemModelCfg};
use temen_interp::bytecode::DebugRun;
use temen_text::parse_module;

fn store_loop(iters: i64, stride: i64) -> String {
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  v0 = i32.const {iters}
  v1 = i64.const 0
  br 1(v0, v1)
}}
block 1 (vk: i32, va: i64) {{
  v2 = i64.const 7
  i64.store va v2
  v3 = i64.const {stride}
  v4 = i64.add va v3
  v5 = i32.const -1
  v6 = i32.add vk v5
  br_if v6 1(v6, v4) 2()
}}
block 2 () {{
  v7 = i64.const 0
  return v7
  }}
}}
"#
    )
}

/// Run the profiler entry over `src` with default geometry; returns `(status, stats JSON text)`.
fn profile(src: &str) -> (i32, String) {
    let m = parse_module(src).expect("parses");
    let bytes = temen_encode::encode_module(&m);
    let status = temen_mem_profile(bytes.as_ptr(), bytes.len(), 0, 0, 0, 0, 0, 0, 0);
    let (p, l) = (temen_mem_profile_stats_ptr(), temen_mem_profile_stats_len());
    let raw = if p.is_null() || l == 0 {
        Vec::new()
    } else {
        unsafe { std::slice::from_raw_parts(p, l).to_vec() }
    };
    (status, String::from_utf8(raw).expect("utf8"))
}

/// The same guest observed through the **debug-session sink** (no instrumentation) — the other
/// feed of the two-feed design; must produce identical stats.
fn sink_stats(src: &str) -> String {
    let m = parse_module(src).expect("parses");
    let mut run = DebugRun::new(&m, 0, &[]).expect("subset");
    let model = Arc::new(Mutex::new(MemModel::new(MemModelCfg::default())));
    {
        let mut g = model.lock().unwrap();
        g.set_snapshots(false);
    }
    let feed = Arc::clone(&model);
    let mut n = 0u64; // match the profiler's synthetic event clock (1-based, forward-only)
    run.set_access_sink(Box::new(move |_clock, task, ev| {
        n += 1;
        feed.lock()
            .unwrap_or_else(|e| e.into_inner())
            .observe(n, task, ev);
    }));
    let mut fuel = u64::MAX;
    while run.tick(&mut fuel) {}
    let s = model.lock().unwrap().stats_json().to_string();
    s
}

#[test]
fn run_mode_profile_matches_the_sink_and_orders_misses() {
    // (a) Two feeds, one answer: hook-fed (instrumented run entry) ≡ sink-fed (debug session).
    let seq_src = store_loop(128, 8);
    let (status, hook_stats) = profile(&seq_src);
    assert_eq!(status, 0, "clean profiled run: {hook_stats}");
    assert_eq!(
        hook_stats,
        sink_stats(&seq_src),
        "hook-fed ≡ sink-fed, stats-for-stats"
    );

    // (b) The strided guest misses more than the sequential one, through the entry.
    let (st2, strided_stats) = profile(&store_loop(128, 512));
    assert_eq!(st2, 0);
    let misses = |s: &str| -> i64 {
        temen_dap::parse(s)
            .and_then(|j| {
                j.get("l1")
                    .and_then(|l| l.get("misses"))
                    .and_then(|v| v.as_i64())
            })
            .expect("l1.misses")
    };
    let (s_miss, t_miss) = (misses(&hook_stats), misses(&strided_stats));
    assert!(
        t_miss > 4 * s_miss,
        "strided ({t_miss}) ≫ sequential ({s_miss})"
    );

    // (c) A manifest-carrying module is refused fail-closed (slot-0 rule), like temen-run's hooks.
    let manifest_src = r#"memory 16
import 0 "write" (i64, i64) -> (i64)

func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  return v0
  }
}
"#;
    let m = parse_module(manifest_src).expect("parses");
    let bytes = temen_encode::encode_module(&m);
    assert_eq!(
        temen_mem_profile(bytes.as_ptr(), bytes.len(), 0, 0, 0, 0, 0, 0, 0),
        -2,
        "manifest-carrying module refused"
    );
}
