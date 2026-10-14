//! STAGE1.md item 6 / PROCESS.md §4 — **concurrent stages**: two children running
//! CONCURRENTLY, piped through a granted `SharedRegion` + canonical-key futex. The parent
//! creates a region, spawns a producer and a consumer detached (each re-granted the region by
//! name), and joins both; the children map the region into their own windows and move four items
//! through a **one-slot bounded ring** (flag at region byte 0, datum at byte 8), parking on the
//! flag and waking each other by `notify` ([`pipeline`]).
//!
//! This is the shape sequential spawn/wait cannot run at all: with a 1-slot ring and 4
//! items, the producer MUST park mid-stream and be woken by the consumer (and vice versa) —
//! run-to-completion order deadlocks. It also pins the S1c futex-key residue closed: each
//! child maps the region in its OWN window (its own address space, its own per-window region
//! id), so wait/notify only rendezvous if the key is the **backing identity** — with
//! per-domain keys every wake misses.

#[path = "support/pipeline.rs"]
mod pipeline;

use temen_interp::{run_with_host, Host, Value};

/// The run's entry: the tree-walker, or the bytecode entry.
type Entry = fn(
    &temen_ir::Module,
    temen_ir::FuncIdx,
    &[Value],
    &mut u64,
    &mut Host,
) -> Result<Vec<Value>, temen_interp::Trap>;

/// Run the detached pipeline ([`pipeline::parent`] over [`pipeline::STAGES`]) through `entry`.
fn detached_pipeline(entry: Entry) -> Vec<Value> {
    let (parent, stages) = pipeline::modules();
    let (mut host, args) = pipeline::host(&stages);
    let mut fuel = 50_000_000u64;
    entry(&parent, 0, &args.map(Value::I32), &mut fuel, &mut host).expect("no trap, no hang")
}

#[test]
fn two_detached_stages_pipe_through_a_shared_region_ring() {
    assert_eq!(
        detached_pipeline(run_with_host),
        vec![Value::I64(410)],
        "the same ring, between two windows nobody's parent can read — private memory \
         and an explicit shared channel compose"
    );
}

/// The bytecode entry runs the detached pipeline identically to the tree-walker.
#[test]
fn the_bytecode_entry_runs_the_pipeline_identically() {
    assert_eq!(
        detached_pipeline(temen_interp::run_with_host_fast),
        vec![Value::I64(410)],
        "identical to the tree-walk run"
    );
}
