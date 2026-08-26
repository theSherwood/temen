//! The **debug-session access sink** ≡ the **W3 instrumentation-pass hook stream**
//! (INTERACTIVE_EMBEDDING.md slice 3). The sink observes a debug session's memory ops with *no
//! module rewrite* — so the machine view and op-clock are untouched — while the hook pass rewrites
//! the module. The two are each other's oracle: on the same program they must report the
//! **identical event sequence** (kind, address, width/span, order). This file drives the *pristine*
//! module under a `DebugRun` with a sink and asserts the exact stream `mem_hooks_diff.rs` pins for
//! the instrumented module on all three backends — the differential that also pins the sink's
//! bulk-op/atomic/v128 coverage.
//!
//! Plus the **bulk-op watchpoint fix** the same decode delivers: a watchpoint covered only by a
//! `mem.copy` span now stops both debug engines (it used to be silently missed on both — parity
//! hid the gap because the engines shared the single-range decode).

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::DebugRun;
use temen_interp::{Inspector, MemEvent, Stop, StopReason, WatchKind};
use temen_text::parse_module;

/// The `mem_hooks_diff.rs` fixture, verbatim: one of every event kind.
const SRC: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 16448
  v1 = i64.const 7
  i64.store v0 v1 offset=8
  v2 = i32.const 170
  v3 = i64.const 32
  mem.fill v0 v2 v3
  v4 = i64.const 16576
  mem.copy v4 v0 v3
  v5 = i32.const 1
  i32.atomic.store v0 v5
  v6 = i32.atomic.load v0
  v7 = i32.atomic.rmw.add v0 v5
  v8 = i32.atomic.cmpxchg v0 v5 v5
  v9 = v128.load v0
  v10 = i64.load v0 offset=8
  return v10
  }
}
"#;

/// The stream `mem_hooks_diff.rs` proves the instrumented module reports on every backend — the
/// sink over the *pristine* module must match it exactly.
fn expected_events() -> Vec<MemEvent> {
    vec![
        MemEvent::Store {
            addr: 16456,
            width: 8,
        },
        MemEvent::Fill {
            dst: 16448,
            len: 32,
        },
        MemEvent::Copy {
            dst: 16576,
            src: 16448,
            len: 32,
        },
        MemEvent::AtomicStore {
            addr: 16448,
            width: 4,
        },
        MemEvent::AtomicLoad {
            addr: 16448,
            width: 4,
        },
        MemEvent::AtomicRmw {
            addr: 16448,
            width: 4,
        },
        MemEvent::AtomicCmpxchg {
            addr: 16448,
            width: 4,
        },
        MemEvent::Load {
            addr: 16448,
            width: 16,
        },
        MemEvent::Load {
            addr: 16456,
            width: 8,
        },
    ]
}

/// Drive a fresh `DebugRun` over `SRC` to completion with a collecting sink; returns the events,
/// the final op clock, and the guest result.
fn sink_run() -> (Vec<MemEvent>, u64, Option<Vec<temen_interp::Value>>) {
    let m = parse_module(SRC).expect("parses");
    let mut run = DebugRun::new(&m, 0, &[]).expect("in the bytecode debug subset");
    let log: Arc<Mutex<Vec<MemEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&log);
    run.set_access_sink(Box::new(move |_clock, _task, ev| {
        sink.lock().unwrap_or_else(|e| e.into_inner()).push(ev);
    }));
    let mut fuel = u64::MAX;
    while run.tick(&mut fuel) {}
    let events = log.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let result = run.result().cloned().and_then(|r| r.ok());
    (events, run.op_clock(), result)
}

/// **The differential**: the sink's stream over the pristine module equals the hook pass's stream
/// over the instrumented one (the `mem_hooks_diff.rs` expectation), event for event.
#[test]
fn sink_stream_matches_the_hook_pass_stream() {
    let (events, _, _) = sink_run();
    assert_eq!(events, expected_events(), "sink ≡ hook stream");
}

/// **Observation never perturbs semantics** (invariant 9b): with and without a sink, the run's
/// result and op-clock are bit-identical.
#[test]
fn sink_is_inert() {
    let m = parse_module(SRC).expect("parses");
    let mut plain = DebugRun::new(&m, 0, &[]).expect("subset");
    let mut fuel = u64::MAX;
    while plain.tick(&mut fuel) {}
    let (_, sink_clock, sink_result) = sink_run();
    assert_eq!(
        plain.op_clock(),
        sink_clock,
        "op-clock unchanged by the sink"
    );
    let plain_result = plain.result().cloned().and_then(|r| r.ok());
    assert!(plain_result.is_some(), "the fixture runs to completion");
    assert_eq!(
        plain_result, sink_result,
        "guest result unchanged by the sink"
    );
}

/// **A write watchpoint covered only by a `mem.copy` destination span stops both engines** — the
/// bulk-op watchpoint fix. Range `[16584, 16588)` is written by nothing but the copy (dst
/// `16576..16608`, above the #1094 NULL guard); before the fix, both engines silently ran past it.
#[test]
fn bulk_copy_dst_write_watchpoint_fires_on_both_engines() {
    let m = parse_module(SRC).expect("parses");

    // Tree-walker: the oracle.
    let mut insp = Inspector::attach(&m, 0, &[], u64::MAX);
    insp.set_watchpoint(16584, 4, WatchKind::Write);
    let tw = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Watchpoint { addr, write },
            pc,
        } => (addr, write, pc),
        other => panic!("tree-walker missed the mem.copy write watchpoint: {other:?}"),
    };

    // Bytecode debug engine: must agree on the hit (address = the access span's base, a write)
    // and on the op it stopped before.
    let mut run = DebugRun::new(&m, 0, &[]).expect("subset");
    run.set_watchpoints(vec![(16584, 4, WatchKind::Write)]);
    let mut fuel = u64::MAX;
    let pc = run
        .run_to(&[], &mut fuel)
        .expect("bytecode missed the mem.copy write watchpoint");
    let (addr, write) = run.take_watch_hit().expect("a watch hit, not a breakpoint");
    assert_eq!((addr, write, pc), tw, "engines agree on the bulk watch hit");
    assert_eq!((addr, write), (16576, true), "the copy's dst span, a write");
}

/// **A read watchpoint covered only by the `mem.copy` source span fires too** — range `[16464, 16465)`
/// is read by nothing but the copy (src `16448..16480`, above the #1094 NULL guard; the v128/i64 loads
/// end at 16464).
#[test]
fn bulk_copy_src_read_watchpoint_fires_on_both_engines() {
    let m = parse_module(SRC).expect("parses");

    let mut insp = Inspector::attach(&m, 0, &[], u64::MAX);
    insp.set_watchpoint(16464, 1, WatchKind::Read);
    let tw = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Watchpoint { addr, write },
            pc,
        } => (addr, write, pc),
        other => panic!("tree-walker missed the mem.copy read watchpoint: {other:?}"),
    };

    let mut run = DebugRun::new(&m, 0, &[]).expect("subset");
    run.set_watchpoints(vec![(16464, 1, WatchKind::Read)]);
    let mut fuel = u64::MAX;
    let pc = run
        .run_to(&[], &mut fuel)
        .expect("bytecode missed the mem.copy read watchpoint");
    let (addr, write) = run.take_watch_hit().expect("a watch hit");
    assert_eq!((addr, write, pc), tw, "engines agree on the src-read hit");
    assert_eq!((addr, write), (16448, false), "the copy's src span, a read");
}
