//! #2318 — a thaw rebuilds what its artifact holds, which its guest determined, so each rebuild is
//! held against the process's pools of mappings and address space: past any ceiling, since the
//! artifact's use was granted, but never past the pools. One a full pool has no room for is refused,
//! and what the thaw charged for it goes back. In its own binary: the pools are the process's, and
//! the tests fill them.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

#[path = "support/detached_thaw.rs"]
mod detached_thaw;

use detached_thaw::*;
use std::sync::Mutex;
use temen_interp::{maps_pool, Host, HostCharge, MAPPED_BUFFER, MAPS_PER_VCPU};
use temen_ir::errno::ENOMEM;
use temen_ir::trap_code::THREAD_FAULT;

/// The pools are the process's: one test at a time fills them.
static POOL: Mutex<()> = Mutex::new(());

/// The process's pool of mappings, filled to the last one: each held by a host buffer's charge of
/// [`MAPPED_BUFFER`] bytes, one `maps` each, on a node no ceiling bounds, so only the pool refuses.
struct Full {
    host: Host,
    held: Vec<HostCharge>,
}

impl Full {
    fn new() -> Full {
        let mut host = Host::new();
        host.set_grant(-1, -1, -1, -1);
        let mut full = Full {
            host,
            held: Vec::new(),
        };
        while full.take() {}
        full
    }

    /// One more mapping, if the pool has it.
    fn take(&mut self) -> bool {
        let mut charge = HostCharge::new(Some(self.host.own_node()));
        let took = charge.resize(MAPPED_BUFFER);
        if took {
            self.held.push(charge);
        }
        took
    }

    /// Hand `n` of them back.
    fn free(&mut self, n: usize) {
        self.held.truncate(self.held.len() - n);
    }

    /// Whether exactly `n` mappings are free: they can be taken again, and then no more.
    fn has_room(&mut self, n: usize) -> bool {
        (0..n).all(|_| self.take()) && !self.take()
    }
}

/// The child's probe: return 7.
const SEVEN: &str = "  pr = i64.const 7\n";

/// [`PARENT`](detached_thaw::PARENT)'s spawn and join, returning the room left in its budget (1 MiB)
/// once it has joined the child, times 100, plus the join's status: `0`, or a trap's wire code.
const PARENT_ROOM: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vrm = i64.const 65560
  i32.store vrm v1
  vrb = i64.const 65564
  i32.store vrb v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  vf = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) v2 (vf)
  vh = i64.const 100
  vrh = i64.mul vroom vh
  vt = i64.add vrh vr_st
  return vt
  }
}
";

/// The budget's whole room, which the parent reads back once the child has ended.
const ROOM: i64 = 1 << 20;

/// A thawed run of a parent whose detached child was live at the freeze, on each engine, given more
/// and more of the pool: first it has no room for the run (its activation refuses it, `ThreadFault`
/// before it starts), then room for the run but not its child, whose relaunch is refused (the
/// parent's join answers `ThreadFault`, and the child's window is back in its budget), then room
/// for both. Whatever the outcome, everything the thaw charged goes back with the run.
#[test]
fn a_relaunch_the_pool_has_no_room_for_is_refused_and_holds_nothing() {
    let _pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pool = maps_pool();
    if pool > 1 << 22 {
        eprintln!("skipped: a pool of {pool} mappings is too large to fill");
        return;
    }
    let parent = parent_of(PARENT_ROOM);
    let child = child_module(&child(SEVEN));
    let Some((art, args)) = artifact(Engine::Interp, &parent, &child) else {
        return;
    };
    let (refused_child, ran) = (Out::Ret(ROOM * 100 + THREAD_FAULT), Out::Ret(ROOM * 100));
    for engine in [Engine::Interp, Engine::Jit] {
        let mut full = Full::new();
        let mut seen = Vec::new();
        for room in 0..pool as usize {
            full.free(room);
            let Some(out) = thaw(&art, engine, &parent, &child, &args) else {
                break;
            };
            assert!(
                full.has_room(room),
                "{engine:?}: a thaw given {room} mappings left some of them held: {out:?}"
            );
            if seen.last() != Some(&out) {
                seen.push(out.clone());
            }
            if out == ran {
                break;
            }
        }
        if seen.is_empty() {
            continue; // a target without the JIT's child executor
        }
        assert_eq!(
            seen,
            [
                Out::Trap("ThreadFault".to_string()),
                refused_child.clone(),
                ran.clone()
            ],
            "{engine:?}: the run refused, then its child, then neither"
        );
    }
}

/// Each other rebuild a full pool has no room for is refused, holding nothing: a §13 region, which
/// would hold a mapping and its address space, and a vCPU, whose thread would hold its own.
#[test]
fn a_region_or_a_vcpu_the_pool_has_no_room_for_is_refused() {
    let _pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    if maps_pool() > 1 << 22 {
        return;
    }
    let mut full = Full::new();
    let host = Host::new();
    assert_eq!(
        host.rebuild_region(&[7; 64]).err(),
        Some(ENOMEM),
        "a region"
    );
    assert!(!host.own_node().rebuild_vcpu(), "a vCPU");
    assert!(full.has_room(0), "neither holds any of the pool");
    full.free(MAPS_PER_VCPU as usize);
    assert!(host.own_node().rebuild_vcpu(), "with room for its thread");
    host.own_node().vcpu_ended();
    full.free(1);
    let region = host
        .rebuild_region(&[7; 64])
        .expect("with room for its mapping");
    drop(region);
    assert!(
        full.has_room(MAPS_PER_VCPU as usize + 1),
        "and both handed it back"
    );
}
