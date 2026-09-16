//! **#1528 — every run loop services `instantiate_detached` (op 15), and they agree.**
//!
//! The tree-walker hosts detached windows itself (`detached_windows.rs`); the resumable engine
//! surfaces the spawn as an event (`detached_engine.rs`). The three run loops that *drive* the
//! bytecode engine — the cooperative executor, the OS-thread parallel driver, and the debug
//! scheduler — used to differ: the executor spawned a fresh-window child, the other two declined
//! the op with `-EINVAL` (#1415 made that decline uniform and non-trapping; before it they trapped
//! `Malformed`). Now all three spawn: the child is a task of the driver over its own `Mem`
//! (`Mem::with_reservation`, its own guard — not a carve), admitted and powerboxed exactly as the
//! executor's child, and `join`ed through the shared seam.
//!
//! One guest, one powerbox, three drivers, one answer. The guest **joins** the child and returns
//! its result, so a driver that only minted a handle (or declined) cannot pass — the child has to
//! run to completion under that driver.

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{Host, Region, Trap, Value};

/// The parent: `v0` Instantiator, `v1` a granted `Module`, `v2` a detached-spawn `Budget`. Issues
/// op 15 (the 7-arg form) and `join`s the child, returning what the child returned.
const PARENT: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vz, vz, ve, vlog, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

/// The child (a child-entry module over its own `memory 15` window): stores and reloads a word in
/// its fresh window (above the null guard, below `1 << 15`) — proof the window is real and
/// writable — and returns the sentinel.
const CHILD: &str = r#"memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 24576
  vk = i64.const 42
  i64.store va vk
  vw = i64.load va
  return vw
  }
}
"#;

fn want() -> Result<Vec<Value>, Trap> {
    Ok(vec![Value::I64(42)])
}

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Grant the three caps op 15 needs and hand back the argument vector.
fn powerbox(host: &mut Host, child: &temen_ir::Module, window: u64) -> Vec<Value> {
    let inst = host.grant_instantiator(0, window);
    let modh = host.grant_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)]
}

/// A window the parallel driver can share across its vCPUs. The `unsafe` of borrowing host memory
/// lives here in the test embedder, exactly as in `bytecode_parallel.rs` — the engine stays
/// `#![forbid(unsafe_code)]` and just takes the `Arc<Region>`.
fn shared_window(size: u64) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    // SAFETY: non-zero, 8-aligned layout; freed below once every borrow of `base` is gone.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null(), "window allocation");
    // SAFETY: `base` owns `size` zeroed bytes and outlives every vCPU (the run joins before dealloc).
    let back = Arc::new(unsafe { Region::shared(base, size) });
    (back, base, layout)
}

#[test]
fn the_cooperative_executor_spawns_and_joins_a_detached_child() {
    let parent = module(PARENT);
    let child = module(CHILD);
    let mut host = Host::new();
    let args = powerbox(&mut host, &child, 1 << 16);
    let mut fuel = u64::MAX;

    let result = bytecode::compile_and_run_with_host(&parent, 0, &args, &mut fuel, &mut host)
        .expect("the cooperative executor runs this module");
    assert_eq!(result, want());
}

/// The OS-thread parallel driver: the child runs on its own OS thread over its own `Mem` and is
/// joined like a confined child. Before #1528 this arm landed `-EINVAL` (and before #1415 it
/// returned `Err(Trap::Malformed)`, abandoning the run).
#[test]
fn the_parallel_driver_spawns_and_joins_a_detached_child() {
    let parent = module(PARENT);
    let child = module(CHILD);
    let mut host = Host::new();
    let args = powerbox(&mut host, &child, 1 << 16);
    let (back, base, layout) = shared_window(1 << 16);
    let mut fuel = u64::MAX;

    let (result, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &parent,
        0,
        &args,
        &mut fuel,
        &[],
        Arc::clone(&back),
        &mut host,
    )
    .expect("the parallel driver runs this module");

    drop(back);
    // SAFETY: same layout; the region and every borrow of `base` are gone (the run joined its vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };

    assert_eq!(result, want());
}

/// The debug scheduler: the child is a `DbgTask` over its own `DbgEnv` (fresh `Mem`, child
/// powerbox), scheduled and joined like a confined child — so a spawning guest stays steppable
/// through its detached child. Before #1528 this arm landed `-EINVAL`.
#[test]
fn the_debug_scheduler_spawns_and_joins_a_detached_child() {
    let parent = module(PARENT);
    let child = module(CHILD);
    let mut host = Host::new();
    let args = powerbox(&mut host, &child, 1 << 16);
    let mut run = ScheduledDebugRun::new_with_host(&parent, 0, &args, host).expect("in subset");
    let mut fuel = u64::MAX;

    let result = loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => break r,
            SchedStop::Break { .. } => continue,
            other => panic!("the debug scheduler must drive op 15 to completion, got {other:?}"),
        }
    };
    assert_eq!(result, want());
}
