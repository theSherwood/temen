//! **#1528 — every run loop services a detached spawn (a v1 `instantiate_rec` record, op 17), and
//! they agree.**
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

#[path = "support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_with_host, Host, Region, StreamRole, Trap, Value};
use temen_ir::SpawnRec;

/// A v1 spawn record for a `memory 15` child, granted the `grants_n` records at `grants_ptr`. Each
/// parent below stores the child's module and `Budget` handles into it at run time.
fn record(grants_ptr: u64, grants_n: u64) -> SpawnRec {
    SpawnRec {
        size_log2: 15,
        grants_ptr,
        grants_n,
        ..SpawnRec::v1()
    }
}

/// The parent: `v0` Instantiator, `v1` a granted `Module`, `v2` a detached-spawn `Budget`. Spawns
/// from its v1 record at 17408 (no grants, no payload) and `join`s the child, returning what the
/// child returned.
fn parent() -> String {
    format!(
        r#"memory 16
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vma = i64.const 17432
  i32.store vma v1
  vba = i64.const 17436
  i32.store vba v2
  vrp = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
  return vr
  }}
}}
{rec}"#,
        rec = rec::segment(17408, &record(0, 0)),
    )
}

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
            &module(&parent()),
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

/// Grant the three caps a detached spawn needs and hand back the argument vector.
fn powerbox(host: &mut Host, child: &temen_ir::Module) -> Vec<Value> {
    let inst = host.grant_instantiator();
    let modh = host.grant_module(child);
    let budget = host.grant_budget(-1, 1 << 20, -1);
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
        let args = powerbox(&mut host, child);
        let mut fuel = u64::MAX;
        let result = run_with_host(parent, 0, &args, &mut fuel, &mut host);
        assert_eq!(result, want, "{shape}");
    });
}

#[test]
fn the_cooperative_executor_spawns_and_joins_a_detached_child() {
    for_each_child(|shape, parent, child, want| {
        let mut host = Host::new();
        let args = powerbox(&mut host, child);
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
        let args = powerbox(&mut host, child);
        let (back, base, layout) = shared_window(1 << 16);
        let mut fuel = u64::MAX;

        let (result, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
            parent,
            0,
            &args,
            &mut fuel,
            &[],
            Some(Arc::clone(&back)),
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
        let args = powerbox(&mut host, child);
        let mut run = ScheduledDebugRun::new_with_host(parent, 0, &args, host).expect("in subset");
        let mut fuel = u64::MAX;

        let result = loop {
            match run.run_until_stop(&mut fuel) {
                SchedStop::Finished(r) => break r,
                SchedStop::Break { .. } => continue,
                other => panic!("{shape}: the debug scheduler must drive op 17, got {other:?}"),
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

/// Params `(inst, module, budget, stdin, stdout)`. Reads one byte, spawns the child (its v1 record
/// at 17536) with `{"stdin", "stdout"}` re-granted by name, joins it, reads one more byte and
/// returns it.
fn stdio_parent() -> String {
    format!(
        r#"memory 16
data 17472 "stdin"
data 17480 "stdout"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, vin: i32, vout: i32) {{
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
  vma = i64.const 17560
  i32.store vma v1
  vba = i64.const 17564
  i32.store vba v2
  vrp = i64.const 17536
  vh = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vh)
  vb1 = i64.const 16392
  vr1 = call.cap 0 0 (i64, i64) -> (i64) vin (vb1, vone)
  vc = i64.load vb1
  return vc
  }}
}}
{rec}"#,
        rec = rec::segment(17536, &record(17408, 2)),
    )
}

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
    let mut args = powerbox(&mut host, child);
    args.push(Value::I32(host.grant_stream(StreamRole::In)));
    args.push(Value::I32(host.grant_stream(StreamRole::Out)));
    (host, args)
}

/// What one runner made of a run: its result, or `None` for a declared decline (INVARIANTS #9 — the
/// debug scheduler leaves pipes to the run harness), and what it wrote to stdout.
type RunnerOutcome = (&'static str, Option<Result<Vec<Value>, Trap>>, Vec<u8>);

/// Run `parent` on every runner — the tree-walk oracle, the cooperative executor, the parallel
/// driver, the debug scheduler — each over a fresh `host()`.
fn on_every_runner(
    parent: &temen_ir::Module,
    host: impl Fn() -> (Host, Vec<Value>),
) -> Vec<RunnerOutcome> {
    let mut out = Vec::new();

    let (mut h, args) = host();
    let mut fuel = u64::MAX;
    let r = run_with_host(parent, 0, &args, &mut fuel, &mut h);
    out.push(("tree-walk oracle", Some(r), h.take_stdout()));

    let (mut h, args) = host();
    let mut fuel = u64::MAX;
    let r = bytecode::compile_and_run_with_host(parent, 0, &args, &mut fuel, &mut h)
        .expect("the cooperative executor runs this module");
    out.push(("cooperative executor", Some(r), h.take_stdout()));

    let (mut h, args) = host();
    let (back, base, layout) = shared_window(1 << 16);
    let mut fuel = u64::MAX;
    let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        parent,
        0,
        &args,
        &mut fuel,
        &[],
        Some(Arc::clone(&back)),
        &mut h,
    )
    .expect("the parallel driver runs this module");
    drop(back);
    // SAFETY: same layout; the region and every borrow of `base` are gone (the run joined its vCPUs).
    unsafe { std::alloc::dealloc(base, layout) };
    out.push(("parallel driver", Some(r), h.take_stdout()));

    let (h, args) = host();
    let mut run = ScheduledDebugRun::new_with_host(parent, 0, &args, h).expect("in subset");
    let mut fuel = u64::MAX;
    let r = loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => break Some(r),
            SchedStop::Break { .. } => continue,
            SchedStop::Declined => break None,
            other => panic!("the debug scheduler must drive op 17, got {other:?}"),
        }
    };
    out.push(("debug scheduler", r, run.host_mut().take_stdout()));
    out
}

#[test]
fn a_child_reads_its_parents_stdin_from_where_the_parent_left_it_on_every_runner() {
    let (parent, child) = (module(&stdio_parent()), module(STDIO_CHILD));
    for (runner, r, stdout) in on_every_runner(&parent, || stdio_host(&child)) {
        assert_eq!(
            (r, stdout),
            (Some(Ok(vec![Value::I64(b'c' as i64)])), b"b".to_vec()),
            "{runner}"
        );
    }
}

// ---- #1807: a pipe between two detached children --------------------------------------------------
//
// The parent mints a pipe (self-op 16), grants its write end to one child as `stdout` and its read end
// to another as `stdin`, drops its own copies, and joins both. The reader sees EOF only when every
// write end is gone — the parent's, closed explicitly, and the writer's, released when the writer's
// domain finishes. A runner that skipped that release would park the reader forever.

/// Params `(inst, writer, reader, budget)`. Spawns the writer from its v1 record at 17536 and the
/// reader from its own at 17664. Returns `reader * 1000 + writer`.
fn pipe_parent() -> String {
    format!(
        r#"memory 16
data 17472 "stdin"
data 17480 "stdout"
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, vmw: i32, vmr: i32, vbud: i32) {{
  vfds = i64.const 16896
  vz = i32.const 0
  vpm = call.cap 4294967295 16 (i64) -> (i32) vz (vfds)
  vrd = i32.load vfds
  vwa = i64.const 16900
  vwr = i32.load vwa
  vg0 = i64.const 17408
  vw0 = i64.const 25769821256
  i64.store vg0 vw0
  vg0h = i64.const 17416
  vwr64 = i64.extend_i32_u vwr
  i64.store vg0h vwr64
  vg1 = i64.const 17424
  vw1 = i64.const 21474853952
  i64.store vg1 vw1
  vg1h = i64.const 17432
  vrd64 = i64.extend_i32_u vrd
  i64.store vg1h vrd64
  vwma = i64.const 17560
  i32.store vwma vmw
  vwba = i64.const 17564
  i32.store vwba vbud
  vwrp = i64.const 17536
  vhw = call.cap 6 17 (i64) -> (i32) v0 (vwrp)
  vrma = i64.const 17688
  i32.store vrma vmr
  vrba = i64.const 17692
  i32.store vrba vbud
  vrrp = i64.const 17664
  vhr = call.cap 6 17 (i64) -> (i32) v0 (vrrp)
  vcr = call.cap 0 2 () -> (i64) vrd ()
  vcw = call.cap 0 2 () -> (i64) vwr ()
  vjr_st, vjr = call.cap 6 1 (i32) -> (i64, i64) v0 (vhr)
  vjw_st, vjw = call.cap 6 1 (i32) -> (i64, i64) v0 (vhw)
  vk = i64.const 1000
  vs = i64.mul vjr vk
  vt = i64.add vs vjw
  return vt
  }}
}}
{writer}{reader}"#,
        writer = rec::segment(17536, &record(17408, 1)),
        reader = rec::segment(17664, &record(17424, 1)),
    )
}

/// Writes `hello, pipe\n` (12 bytes) to its `stdout`; returns what the write returned.
const PIPE_WRITER: &str = r#"memory 15
data 16384 "stdout"
data 16400 "hello, pipe\n"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 6
  vh = self.resolve vp vl
  vb = i64.const 16400
  vn = i64.const 12
  vw = call.cap 0 1 (i64, i64) -> (i64) vh (vb, vn)
  return vw
  }
}
"#;

/// Reads its `stdin` 5 bytes at a time until EOF; returns the byte count.
const PIPE_READER: &str = r#"memory 15
data 16384 "stdin"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 5
  vh = self.resolve vp vl
  vz = i64.const 0
  br 1(vh, vz)
  }
block 1 (h: i32, tot: i64) {
  vb = i64.const 16400
  vc = i64.const 5
  vn = call.cap 0 0 (i64, i64) -> (i64) h (vb, vc)
  vzero = i64.const 0
  vdone = i64.le_s vn vzero
  br_if vdone 2(tot) 3(h, tot, vn)
  }
block 2 (t: i64) {
  return t
  }
block 3 (h3: i32, tot3: i64, n: i64) {
  vt = i64.add tot3 n
  br 1(h3, vt)
  }
}
"#;

#[test]
fn a_pipe_between_two_detached_children_reaches_eof_on_every_runner() {
    let (parent, writer, reader) = (
        module(&pipe_parent()),
        module(PIPE_WRITER),
        module(PIPE_READER),
    );
    let host = || {
        let mut h = Host::new();
        let inst = h.grant_instantiator();
        let mw = h.grant_module(&writer);
        let mr = h.grant_module(&reader);
        let budget = h.grant_budget(-1, 1 << 20, -1);
        let args = [inst, mw, mr, budget].map(Value::I32).to_vec();
        (h, args)
    };
    for (runner, r, _) in on_every_runner(&parent, host) {
        // The debug scheduler declines a pipe park (the run harness drives pipes, not the debugger):
        // the declared frontier, pinned here so a change to it is a decision.
        let want = (runner != "debug scheduler").then(|| Ok(vec![Value::I64(12_012)]));
        assert_eq!(r, want, "{runner}");
    }
}
