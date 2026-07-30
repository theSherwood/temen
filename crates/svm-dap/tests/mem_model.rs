//! The host-side **memory model** over the debug-session access sink (INTERACTIVE_EMBEDDING.md
//! slice 4): a strided-vs-sequential guest pair shows the expected miss-count ordering, the
//! first-touch fault counter counts pages, the model's state is **seek-consistent** (after
//! `seek(t)`, model state ≡ a fresh model observing a from-0 run to `t` — through the real
//! checkpoint ladder), and the DAP surface (`memModel` launch arg + `memModelStats`) works
//! end-to-end and fails closed on an engine without a sink.

use std::sync::{Arc, Mutex};
use svm_dap::models::{MemModel, MemModelCfg};
use svm_dap::{BytecodeBackend, DapServer, Debuggee, Json, SharedSink};
use svm_text::parse_module;

/// A store loop: `iters` stores of one i64 each, `stride` bytes apart, starting at 0.
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

/// Run `src` to completion on a `BytecodeBackend` with a model-feeding sink; returns the model.
fn model_run(src: &str) -> Arc<Mutex<MemModel>> {
    let m = parse_module(src).expect("parses");
    let mut b = BytecodeBackend::new(m, 0, &[], u64::MAX, false, Vec::new(), false)
        .expect("in the bytecode debug subset");
    let model = Arc::new(Mutex::new(MemModel::new(MemModelCfg::default())));
    let feed = Arc::clone(&model);
    let sink: SharedSink = Arc::new(Mutex::new(
        move |c: u64, t: usize, e: svm_interp::MemEvent| {
            feed.lock()
                .unwrap_or_else(|x| x.into_inner())
                .observe(c, t, e)
        },
    ));
    b.set_access_sink(sink);
    let _ = Debuggee::run_until_stop(&mut b);
    model
}

/// **The X2 acceptance ordering**: equal access counts, but the strided guest (512-byte stride —
/// every line lands in L1 set 0) misses far more than the sequential one (8-byte stride — 8
/// accesses share each 64-byte line).
#[test]
fn strided_misses_exceed_sequential_misses() {
    let seq = model_run(&store_loop(128, 8));
    let strided = model_run(&store_loop(128, 512));
    let (s_hits, s_miss, ..) = seq.lock().unwrap().counters();
    let (t_hits, t_miss, ..) = strided.lock().unwrap().counters();
    assert_eq!(s_hits + s_miss, t_hits + t_miss, "equal access counts");
    assert!(
        t_miss > 4 * s_miss,
        "strided thrashes ({t_miss} misses) vs sequential ({s_miss})"
    );
}

/// **The first-touch fault counter counts pages**: a 1 KiB footprint touches 1 page, a 64 KiB
/// footprint touches 16 (page size 4096).
#[test]
fn fault_counter_counts_first_touch_pages() {
    let (.., seq_faults) = model_run(&store_loop(128, 8)).lock().unwrap().counters();
    let (.., strided_faults) = model_run(&store_loop(128, 512)).lock().unwrap().counters();
    assert_eq!(seq_faults, 1, "1 KiB footprint = one page");
    assert_eq!(strided_faults, 16, "64 KiB footprint = sixteen pages");
}

/// **Seek consistency through the real checkpoint ladder** (the slice-4 acceptance): run a
/// >1024-op guest to completion, `seek` back to a mid-run clock — the engine restores a ladder
/// checkpoint and replays the tail, the sink feeds the replay, and the model's rollback lands on
/// its matching snapshot. The result must equal a fresh model observing a from-0 run to the same
/// clock, stats-for-stats (grids and LRU ticks included).
#[test]
fn model_state_is_seek_consistent() {
    let src = store_loop(512, 8); // ~4k ops: several stride boundaries deep
    let m = parse_module(&src).expect("parses");

    let mk = |m: &svm_ir::Module| -> (BytecodeBackend, Arc<Mutex<MemModel>>) {
        let mut b = BytecodeBackend::new(m.clone(), 0, &[], u64::MAX, false, Vec::new(), false)
            .expect("subset");
        let model = Arc::new(Mutex::new(MemModel::new(MemModelCfg::default())));
        let feed = Arc::clone(&model);
        let sink: SharedSink = Arc::new(Mutex::new(
            move |c: u64, t: usize, e: svm_interp::MemEvent| {
                feed.lock()
                    .unwrap_or_else(|x| x.into_inner())
                    .observe(c, t, e)
            },
        ));
        b.set_access_sink(sink);
        (b, model)
    };

    // A: run to completion, then seek back to 1500 (past the first ladder checkpoint at 1024).
    let (mut a, model_a) = mk(&m);
    let _ = Debuggee::run_until_stop(&mut a);
    let _ = Debuggee::seek(&mut a, 1500);

    // B: a fresh session seeks straight to 1500 (a forward from-0 replay).
    let (mut b, model_b) = mk(&m);
    let _ = Debuggee::seek(&mut b, 1500);

    let sa = model_a.lock().unwrap().stats_json().to_string();
    let sb = model_b.lock().unwrap().stats_json().to_string();
    assert_eq!(sa, sb, "seek(t) model state ≡ fresh from-0 run to t");
}

/// **The DAP surface**: `launch` with `memModel` arms the model, `memModelStats` reads it back
/// after the run, and the tree-walker (no sink) fails the launch closed.
#[test]
fn dap_mem_model_end_to_end_and_fail_closed() {
    let src = store_loop(64, 8);
    let req = |seq: i64, command: &str, args: Json| {
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("request")),
            ("command", Json::s(command)),
            ("arguments", args),
        ])
    };
    let response = |msgs: &[Json]| -> Json {
        msgs.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .expect("a response")
            .clone()
    };

    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&src)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("memModel", Json::obj(vec![])), // defaults
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "bytecode + memModel launches"
    );
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(4, "memModelStats", Json::obj(vec![])));
    let resp = response(&out);
    assert_eq!(resp.get("success"), Some(&Json::Bool(true)));
    let body = resp.get("body").expect("stats body");
    let l1_misses = body
        .get("l1")
        .and_then(|l| l.get("misses"))
        .and_then(|v| v.as_i64())
        .expect("l1.misses");
    assert!(l1_misses > 0, "the guest's stores reached the model");
    assert!(body.get("l1Grids").is_some() && body.get("l2Grid").is_some());

    // Fail-closed: the tree-walker has no access sink.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&src)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("memModel", Json::obj(vec![])),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "treewalk + memModel fails the launch"
    );
    // And a session without a model fails memModelStats cleanly.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(&src)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(3, "memModelStats", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}
