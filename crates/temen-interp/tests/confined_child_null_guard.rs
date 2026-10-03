//! **#1206 — a child's window carries the NULL guard, on every engine.** The guard
//! (`[0, POWERBOX_NULL_GUARD)` unmapped, #964/#1094) is "unconditional — every module reserves it",
//! and every root `Mem` constructor seeded it; the confined-child constructors (`nested_view`, the
//! in-engine cooperative/parallel arms' chokepoint, and the resumable `Vcpu`'s child constructors,
//! the host-orchestrated / browser path) did not — so a child in a carve of 16 KiB or more could store
//! at NULL on the interpreter where the emitted tier's guard compare (and any cross-tier bounce over
//! the same carve) traps. Now every child is guarded like a root; #1867 moves the child here to its
//! own detached window, the one placement once the carve is gone.
//!
//! Differential across the cooperative multiplex and the OS-thread parallel driver: a child storing at
//! 8 traps `MemoryFault` on both; the same child storing above the guard returns the value on both. A
//! spawned thread gets the same treatment.

#[path = "support/rec.rs"]
mod rec;

use std::sync::Arc;

use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_ir::SpawnRec;
use temen_text::parse_module;

/// Root `(inst, budget) -> child`: spawns func 1 detached through the v1 record at 17408, paid from
/// `budget`, and `join`s it. The child stores 7 at `addr` in its own window and loads it back.
fn child_src(addr: u64) -> String {
    format!(
        r#"memory 17
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32) {{
  vb = i64.const 17436
  i32.store vb vbud
  vrec = i64.const 17408
  vh = call.cap 6 17 (i64) -> (i32) vinst (vrec)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  va = i64.const {addr}
  vseven = i64.const 7
  i64.store va vseven
  vr = i64.load va
  return vr
  }}
}}
{}"#,
        rec::segment(17408, &SpawnRec::v1(1))
    )
}

/// Root `() -> thread`: `thread.spawn` func 1 (stores 7 at `addr`, loads it back), `join` it.
fn thread_src(addr: u64) -> String {
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vsp = i64.const 32768
  varg = i64.const 0
  vt = thread.spawn 1 vsp varg
  vr = thread.join vt
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  va = i64.const {addr}
  vseven = i64.const 7
  i64.store va vseven
  vr = i64.load va
  return vr
  }}
}}
"#
    )
}

/// A self-spawning root's powerbox: its program registered, an instantiator, and a budget.
fn spawn_host(m: &temen_ir::Module) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_self_module(&Arc::new(m.clone()));
    let inst = host.grant_instantiator(0, 1 << 17);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (host, vec![Value::I32(inst), Value::I32(budget)])
}

/// The cooperative multiplex.
fn coop(src: &str, mk: fn(&temen_ir::Module) -> (Host, Vec<Value>)) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (mut host, args) = mk(&m);
    let mut fuel = 50_000_000u64;
    bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
        .expect("bytecode subset")
}

/// The OS-thread parallel driver over a caller-owned backing.
fn parallel(
    src: &str,
    mk: fn(&temen_ir::Module) -> (Host, Vec<Value>),
) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    let (mut host, args) = mk(&m);
    let size = 1usize << 17;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero layout; `size` valid 8-aligned bytes owned here until freed below.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` is `size` valid bytes, exclusively this run's, freed only after every vCPU joined.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    let mut fuel = 50_000_000u64;
    let (r, _) = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        back,
        &mut host,
    )
    .expect("bytecode subset");
    // SAFETY: same layout; the run (and every region view) is finished.
    unsafe { std::alloc::dealloc(base, layout) };
    r
}

fn none(_: &temen_ir::Module) -> (Host, Vec<Value>) {
    (Host::new(), Vec::new())
}

#[test]
fn child_store_at_null_traps_on_both_engines() {
    let src = child_src(8);
    assert_eq!(
        coop(&src, spawn_host),
        Err(Trap::MemoryFault),
        "cooperative"
    );
    assert_eq!(
        parallel(&src, spawn_host),
        Err(Trap::MemoryFault),
        "parallel"
    );
}

#[test]
fn child_store_above_the_guard_passes_on_both_engines() {
    let src = child_src(20480);
    assert_eq!(
        coop(&src, spawn_host),
        Ok(vec![Value::I64(7)]),
        "cooperative"
    );
    assert_eq!(
        parallel(&src, spawn_host),
        Ok(vec![Value::I64(7)]),
        "parallel"
    );
}

#[test]
fn spawned_thread_store_at_null_traps_on_both_engines() {
    let src = thread_src(8);
    assert_eq!(coop(&src, none), Err(Trap::MemoryFault), "cooperative");
    assert_eq!(parallel(&src, none), Err(Trap::MemoryFault), "parallel");
}

#[test]
fn spawned_thread_store_above_the_guard_passes_on_both_engines() {
    let src = thread_src(16384);
    assert_eq!(coop(&src, none), Ok(vec![Value::I64(7)]), "cooperative");
    assert_eq!(parallel(&src, none), Ok(vec![Value::I64(7)]), "parallel");
}
