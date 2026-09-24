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
//! One guest, one powerbox, three drivers — and the tree-walk oracle — one answer. The guest
//! **joins** the child and returns its result, so a driver that only minted a handle (or declined)
//! cannot pass — the child has to run to completion under that driver.
//!
//! #1720 — and every **entry shape** a spawn admits gets the same answer everywhere: the child-entry
//! ABI, and a powerbox `_start` (no params) returning an `i32` status (read back sign-extended) or
//! nothing (read back as `0`). A card module nests as built through exactly these arms.

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_with_host, Host, Region, StreamRole, Trap, Value};

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

/// The children, one per admitted entry shape, each over its own `memory 15` window, with what the
/// parent's `join` reads back. The child-entry one stores and reloads a word in its fresh window
/// (above the null guard, below `1 << 15`) — proof the window is real and writable.
const CHILDREN: [(&str, &str, i64); 3] = [
    (
        "child entry `(i64) -> (i64)`",
        r#"memory 15
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 24576
  vk = i64.const 42
  i64.store va vk
  vw = i64.load va
  return vw
  }
}
"#,
        42,
    ),
    (
        "powerbox `_start: () -> (i32)`",
        r#"memory 15
export 0 func "_start" 0
func () -> (i32) {
block 0 () {
  vs = i32.const -7
  return vs
  }
}
"#,
        -7,
    ),
    (
        "powerbox `_start: () -> ()`",
        r#"memory 15
export 0 func "_start" 0
func () -> () {
block 0 () {
  return
  }
}
"#,
        0,
    ),
];

/// Run `check` once per child shape, with the parsed parent, the child, and the answer to expect.
fn for_each_child(
    check: impl Fn(&str, &temen_ir::Module, &temen_ir::Module, Result<Vec<Value>, Trap>),
) {
    for (shape, text, answer) in CHILDREN {
        check(
            shape,
            &module(PARENT),
            &module(text),
            Ok(vec![Value::I64(answer)]),
        );
    }
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
fn the_tree_walk_oracle_spawns_and_joins_a_detached_child() {
    for_each_child(|shape, parent, child, want| {
        let mut host = Host::new();
        let args = powerbox(&mut host, child, 1 << 16);
        let mut fuel = u64::MAX;
        let result = run_with_host(parent, 0, &args, &mut fuel, &mut host);
        assert_eq!(result, want, "{shape}");
    });
}

#[test]
fn the_cooperative_executor_spawns_and_joins_a_detached_child() {
    for_each_child(|shape, parent, child, want| {
        let mut host = Host::new();
        let args = powerbox(&mut host, child, 1 << 16);
        let mut fuel = u64::MAX;
        let result = bytecode::compile_and_run_with_host(parent, 0, &args, &mut fuel, &mut host)
            .expect("the cooperative executor runs this module");
        assert_eq!(result, want, "{shape}");
    });
}

/// The OS-thread parallel driver: the child runs on its own OS thread over its own `Mem` and is
/// joined like a confined child. Before #1528 this arm landed `-EINVAL` (and before #1415 it
/// returned `Err(Trap::Malformed)`, abandoning the run).
#[test]
fn the_parallel_driver_spawns_and_joins_a_detached_child() {
    for_each_child(|shape, parent, child, want| {
        let mut host = Host::new();
        let args = powerbox(&mut host, child, 1 << 16);
        let (back, base, layout) = shared_window(1 << 16);
        let mut fuel = u64::MAX;

        let (result, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
            parent,
            0,
            &args,
            &mut fuel,
            &[],
            Arc::clone(&back),
            &mut host,
        )
        .expect("the parallel driver runs this module");

        drop(back);
        // SAFETY: same layout; the region and every borrow of `base` are gone (the run joined its
        // vCPUs).
        unsafe { std::alloc::dealloc(base, layout) };

        assert_eq!(result, want, "{shape}");
    });
}

/// The debug scheduler: the child is a `DbgTask` over its own `DbgEnv` (fresh `Mem`, child
/// powerbox), scheduled and joined like a confined child — so a spawning guest stays steppable
/// through its detached child. Before #1528 this arm landed `-EINVAL`.
#[test]
fn the_debug_scheduler_spawns_and_joins_a_detached_child() {
    for_each_child(|shape, parent, child, want| {
        let mut host = Host::new();
        let args = powerbox(&mut host, child, 1 << 16);
        let mut run = ScheduledDebugRun::new_with_host(parent, 0, &args, host).expect("in subset");
        let mut fuel = u64::MAX;

        let result = loop {
            match run.run_until_stop(&mut fuel) {
                SchedStop::Finished(r) => break r,
                SchedStop::Break { .. } => continue,
                other => panic!("{shape}: the debug scheduler must drive op 15, got {other:?}"),
            }
        };
        assert_eq!(result, want, "{shape}");
    });
}

// ---- #1720: stdin rides into a child the way stdout does ------------------------------------------
//
// §7c stdio inheritance used to cover stdout/stderr only: a re-granted `Stream(In)` read the
// *child's* empty buffer, so a card that reads its program from stdin saw EOF when nested. Now a
// re-granted stdin carries its granter's (promoted) stdin, so the parent and the child read one
// stream from one position. With stdin `abc`: the parent reads `a`, the child reads `b` and writes it
// to its re-granted stdout, then the parent reads `c` — on every runner.

/// Params `(inst, module, budget, stdin, stdout)`. Reads one byte, spawns the child (op 15) with
/// `{"stdin", "stdout"}` re-granted by name, joins it, reads one more byte and returns it.
const STDIO_PARENT: &str = r#"memory 16
data 17472 "stdin"
data 17480 "stdout"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, vin: i32, vout: i32) {
  vb0 = i64.const 16384
  vone = i64.const 1
  vr0 = call.cap 0 0 (i64, i64) -> (i64) vin (vb0, vone)
  vg0 = i64.const 17408
  vw0 = i64.const 21474853952
  i64.store vg0 vw0
  vg0h = i64.const 17416
  vin64 = i64.extend_i32_u vin
  i64.store vg0h vin64
  vg1 = i64.const 17424
  vw1 = i64.const 25769821256
  i64.store vg1 vw1
  vg1h = i64.const 17432
  vout64 = i64.extend_i32_u vout
  i64.store vg1h vout64
  vmh = i64.extend_i32_u v1
  vmin = i64.extend_i32_u v2
  vgn = i64.const 2
  ve = i64.const 0
  vlog = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmin, vmh, vg0, vgn, ve, vlog, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  vb1 = i64.const 16392
  vr1 = call.cap 0 0 (i64, i64) -> (i64) vin (vb1, vone)
  vc = i64.load vb1
  return vc
  }
}
"#;

/// A powerbox `_start` child: resolves the stdin and stdout it was granted, copies one byte across.
const STDIO_CHILD: &str = r#"memory 15
data 16384 "stdin"
data 16392 "stdout"
export 0 func "_start" 0
func () -> (i32) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 5
  vin = self.resolve vp vl
  vq = i64.const 16392
  vm = i64.const 6
  vout = self.resolve vq vm
  vb = i64.const 16400
  vone = i64.const 1
  vr = call.cap 0 0 (i64, i64) -> (i64) vin (vb, vone)
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vb, vone)
  vz = i32.const 0
  return vz
  }
}
"#;

/// The parent's host: stdin `abc`, the three spawn caps, then stdin and stdout.
fn stdio_host(child: &temen_ir::Module) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.stdin = b"abc".to_vec();
    let mut args = powerbox(&mut host, child, 1 << 16);
    args.push(Value::I32(host.grant_stream(StreamRole::In)));
    args.push(Value::I32(host.grant_stream(StreamRole::Out)));
    (host, args)
}

#[test]
fn a_child_reads_its_parents_stdin_from_where_the_parent_left_it_on_every_runner() {
    let (parent, child) = (module(STDIO_PARENT), module(STDIO_CHILD));
    let want: Result<Vec<Value>, Trap> = Ok(vec![Value::I64(b'c' as i64)]);

    let (mut host, args) = stdio_host(&child);
    let mut fuel = u64::MAX;
    let r = run_with_host(&parent, 0, &args, &mut fuel, &mut host);
    assert_eq!(
        (r, host.take_stdout()),
        (want.clone(), b"b".to_vec()),
        "tree-walk oracle"
    );

    let (mut host, args) = stdio_host(&child);
    let mut fuel = u64::MAX;
    let r = bytecode::compile_and_run_with_host(&parent, 0, &args, &mut fuel, &mut host)
        .expect("the cooperative executor runs this module");
    assert_eq!(
        (r, host.take_stdout()),
        (want.clone(), b"b".to_vec()),
        "cooperative executor"
    );

    let (mut host, args) = stdio_host(&child);
    let (back, base, layout) = shared_window(1 << 16);
    let mut fuel = u64::MAX;
    let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
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
    assert_eq!(
        (r, host.take_stdout()),
        (want.clone(), b"b".to_vec()),
        "parallel driver"
    );

    let (host, args) = stdio_host(&child);
    let mut run = ScheduledDebugRun::new_with_host(&parent, 0, &args, host).expect("in subset");
    let mut fuel = u64::MAX;
    let r = loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => break r,
            SchedStop::Break { .. } => continue,
            other => panic!("the debug scheduler must drive op 15, got {other:?}"),
        }
    };
    assert_eq!(
        (r, run.host_mut().take_stdout()),
        (want, b"b".to_vec()),
        "debug scheduler"
    );
}
