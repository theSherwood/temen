//! PROCESS.md S0 spike (O2): do futex `wait`/`notify` rendezvous **across domains** on shared memory?
//! A parent and its detached child share a §13 `SharedRegion`: the parent maps it at 128 KiB in its
//! own window, and the spawn record pre-maps it into the child's window at 64 KiB. The futex key of a
//! region-aliased page is `(backing, byte offset within the region)` (`FutexKey::Region`), so the two
//! rendezvous on the *same region byte* under *different guest addresses* in *different domains*
//! (parent: `128 KiB + a`; child: `64 KiB + a`). That is the load-bearing fact for building endpoints
//! as a **library** over shared memory + futex (PROCESS.md §4 "library first") instead of new runtime
//! rendezvous machinery.
//!
//! Pinned on every driver that schedules a child itself (`drivers::SCHEDULING`): the `Vcpu` harness
//! runs a child to completion when it is spawned and orchestrates no `wait`.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{Host, Value};
use temen_ir::{Module, SpawnRec};

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The parent `(Instantiator, AddressSpace, child Module, Budget) -> i64`: mints a 64 KiB region, maps
/// it at 128 KiB, spawns the child detached through the spawn record at 17408 with the region pre-mapped
/// at 64 KiB of the child's window, then runs `body` (which sees the child as `vc` and the region
/// byte 20480 as `va`) and returns its `vr`.
fn parent(body: &str) -> Module {
    let spawn = SpawnRec {
        size_log2: 17,
        child_off: 65536,
        ..SpawnRec::default()
    };
    module(&format!(
        "memory 18
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 131072
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vpm = i64.const 17416
  i32.store vpm v2
  vpb = i64.const 17420
  i32.store vpb v3
  vpr = i64.const 17456
  i32.store vpr vrh
  vp = i64.const 17408
  vc = call.cap 6 17 (i64) -> (i32) v0 (vp)
  va = i64.const 151552
{body}
  return vr
  }}
}}
{rec}",
        rec = rec::segment(17408, &spawn)
    ))
}

/// The powerbox: an `Instantiator` and an `AddressSpace` over the parent's 256 KiB window, the child
/// module, and a `Budget` for exactly one 128 KiB child window.
fn setup(child: &str) -> impl Fn() -> (Host, Vec<Value>) {
    let child = module(child);
    move || {
        let mut host = Host::new();
        let inst = host.grant_instantiator();
        let aspace = host.grant_address_space(0, 1 << 18);
        let modh = host.grant_module(&child);
        let budget = host.grant_budget(-1, 1 << 17, -1);
        (host, [inst, aspace, modh, budget].map(Value::I32).to_vec())
    }
}

/// Deterministic rendezvous — the key-space proof. The parent parks on `atomic.wait` at its address
/// of region byte 20480 (expected 0; no store ever changes it). The child spin-`notify`s **its**
/// address of that byte, `64 KiB + 20480`, until the notify reports a waiter woken. If the keys did
/// not collide, the parent would never wake and the child would spin to fuel exhaustion; instead the
/// parent wakes (status 0 = woken, never 1 = not-equal since the value stays 0) and joins the
/// child's 7.
///
/// Ordering is safe on every executor shape: on a single cooperative worker the parent runs until
/// it parks (the spawn enqueues the child, no switch), so the child's first notify finds it; with
/// real parallel workers the child spins until the park lands.
#[test]
fn parent_waits_child_notifies_across_domains() {
    let m = parent(
        "  vexp = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait va vexp vinf
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  vw64 = i64.extend_i32_u vw
  vh = i64.const 100
  vwh = i64.mul vw64 vh
  vr = i64.add vwh vj",
    );
    let child = "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
}
block 1 () {
  v1 = i64.const 86016
  v2 = i32.const 1
  v3 = atomic.notify v1 v2
  v4 = i32.const 0
  v5 = i32.lt_u v4 v3
  br_if v5 2() 1()
}
block 2 () {
  v6 = i64.const 7
  return v6
  }
}
";
    // status 0 (woken) * 100 + child result 7 — any other status means the keys diverged.
    drivers::agree_on(
        &drivers::SCHEDULING,
        "the parent is woken by the child's notify of the same region byte",
        &m,
        &setup(child),
        &drivers::Ran {
            result: Ok(vec![Value::I64(7)]),
            stdout: Vec::new(),
            stderr: Vec::new(),
        },
    );
}

/// The inverse direction, racy by nature (the parent cannot spin-notify without starving a
/// single-worker executor, so it stores + notifies once). The child `atomic.wait`s its address of
/// region byte 20480 (expected 0); the parent stores 1 and notifies once at its own address of that
/// byte, then joins. The futex compare-and-park makes every interleaving safe, and both legal
/// outcomes prove the cross-domain semantics:
///   child parked first  → notify wakes it:   child status 0, parent saw 1 woken → result  1
///   store landed first  → compare sees 1:    child status 1, notify woke 0      → result 10
/// A hang or any other value means the shared-region futex contract is broken.
#[test]
fn child_waits_parent_notifies_across_domains() {
    let m = parent(
        "  vone = i32.const 1
  i32.atomic.store va vone
  vn = atomic.notify va vone
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  vten = i64.const 10
  vj10 = i64.mul vj vten
  vn64 = i64.extend_i32_u vn
  vr = i64.add vj10 vn64",
    );
    let child = "memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 86016
  v2 = i32.const 0
  v3 = i64.const -1
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.extend_i32_u v4
  return v5
  }
}
";
    for driver in drivers::SCHEDULING {
        let r = drivers::run_on(driver, &m, &setup(child))
            .unwrap_or_else(|| panic!("{driver:?} declined the module"))
            .result;
        assert!(
            r == Ok(vec![Value::I64(1)]) || r == Ok(vec![Value::I64(10)]),
            "{driver:?}: expected (woken,1)=1 or (not-equal,0)=10, got {r:?}"
        );
    }
}
