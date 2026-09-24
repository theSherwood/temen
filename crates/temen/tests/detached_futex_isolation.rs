//! #1731 — a detached (op 15) child's futex ops act on **its own** window, on every driver.
//!
//! Every detached window starts at base 0, so a raw guest address is not a rendezvous identity across
//! windows (#1283). The canonical key is `Mem::futex_key`: backing identity plus address. Two shapes:
//! - one detached child's `notify` at address A must not wake a sibling parked at A in its own window;
//! - a detached child's `wait` at A must compare against A in its own window, not the root's.
#![cfg(unix)]

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_capture_reserved_with_host, Host, Region, Trap, Value};
use temen_ir::Module;
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 256 << 10;

/// `WAIT_TIMED_OUT`, as `i32.atomic.wait` returns it.
const TIMED_OUT: i64 = 2;

/// A child (64 KiB, entry `(i64) -> (i64)`): `i32.atomic.wait` at 32768 for 0 with a 1 ms timeout,
/// and return the status. Its own window holds 0 there, so the wait parks and times out.
const WAITER: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 32768
  vexp = i32.const 0
  vto = i64.const 1000000
  vst = i32.atomic.wait va vexp vto
  vr = i64.extend_i32_u vst
  return vr
  }
}
";

/// A child (64 KiB): `atomic.notify` at 32768 a thousand times, and return the total woken. The loop
/// spans any interleaving with a sibling's park.
const NOTIFIER: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vi = i64.const 0
  vs = i64.const 0
  br 1(vi, vs)
}
block 1 (vi: i64, vs: i64) {
  vn = i64.const 1000
  vlt = i64.lt_s vi vn
  br_if vlt 2(vi, vs) 3(vs)
}
block 2 (vi: i64, vs: i64) {
  va = i64.const 32768
  vc = i32.const 1
  vw = atomic.notify va vc
  vw64 = i64.extend_i32_u vw
  vs2 = i64.add vs vw64
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2, vs2)
}
block 3 (vs: i64) {
  return vs
  }
}
";

/// The root (256 KiB, args `(instantiator, WAITER, NOTIFIER, budget)`): spawn `WAITER` then
/// `NOTIFIER` detached, join both, and return `waiter status * 10000 + notifier woken`.
const SIBLINGS: &str = "memory 18
func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vwait: i32, vnote: i32, vbud: i32) {
  mw = i64.extend_i32_s vwait
  mn = i64.extend_i32_s vnote
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 16
  hw = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, mw, gz, gz, gz, sl, gz)
  hn = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, mn, gz, gz, gz, sl, gz)
  jw = call.cap 6 1 (i32) -> (i64) vinst (hw)
  jn = call.cap 6 1 (i32) -> (i64) vinst (hn)
  vk = i64.const 10000
  vhi = i64.mul jw vk
  vr = i64.add vhi jn
  return vr
  }
}
";

/// The root: store 7 at 32768 in its own window, spawn `WAITER` detached, and return its status.
const ROOT_HOLDS_SEVEN: &str = "memory 18
func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vwait: i32, vnote: i32, vbud: i32) {
  va = i64.const 32768
  vseven = i32.const 7
  i32.store va vseven
  mw = i64.extend_i32_s vwait
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 16
  hw = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, mw, gz, gz, gz, sl, gz)
  jw = call.cap 6 1 (i32) -> (i64) vinst (hw)
  return jw
  }
}
";

fn parse(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

#[derive(Clone, Copy, Debug)]
enum Driver {
    Oracle,
    Cooperative,
    Parallel,
    Debug,
}

fn run(driver: Driver, root_src: &str) -> Result<Vec<Value>, Trap> {
    let root = parse(root_src);
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, WIN as u64);
    let waiter = host.grant_module(&parse(WAITER));
    let notifier = host.grant_module(&parse(NOTIFIER));
    let budget = host.grant_budget(0, 1 << 20, 0);
    let args = [inst, waiter, notifier, budget].map(Value::I32);
    let mut fuel = 50_000_000u64;
    match driver {
        Driver::Oracle => {
            let init = vec![0u8; WIN];
            run_capture_reserved_with_host(&root, 0, &args, &mut fuel, &init, 0, &mut host).0
        }
        Driver::Cooperative => {
            bytecode::compile_and_run_with_host(&root, 0, &args, &mut fuel, &mut host)
                .expect("the cooperative executor compiles this module")
        }
        Driver::Parallel => {
            let layout = std::alloc::Layout::from_size_align(WIN, 8).unwrap();
            // SAFETY: non-zero, 8-aligned layout; freed below once every borrow of `base` is gone.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            assert!(!base.is_null(), "window allocation");
            // SAFETY: `base` owns `WIN` zeroed bytes and outlives every vCPU (the run joins first).
            let back = Arc::new(unsafe { Region::shared(base, WIN as u64) });
            let (result, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
                &root,
                0,
                &args,
                &mut fuel,
                &[],
                Arc::clone(&back),
                &mut host,
            )
            .expect("the parallel driver compiles this module");
            drop(back);
            // SAFETY: same layout; the region and every borrow of `base` are gone.
            unsafe { std::alloc::dealloc(base, layout) };
            result
        }
        Driver::Debug => {
            let mut dbg = ScheduledDebugRun::new_with_host(&root, 0, &args, host).expect("subset");
            loop {
                match dbg.run_until_stop(&mut fuel) {
                    SchedStop::Finished(r) => break r,
                    SchedStop::Break { .. } => continue,
                    other => panic!("the debug scheduler stopped early: {other:?}"),
                }
            }
        }
    }
}

fn on_every_driver(root_src: &str, want: i64, what: &str) {
    for d in [
        Driver::Oracle,
        Driver::Cooperative,
        Driver::Parallel,
        Driver::Debug,
    ] {
        assert_eq!(
            run(d, root_src),
            Ok(vec![Value::I64(want)]),
            "{what} on {d:?}"
        );
    }
}

/// Both siblings use address 32768, each in its own window: the waiter times out and the notifier
/// woke no one.
#[test]
fn a_detached_siblings_notify_does_not_wake_the_other() {
    on_every_driver(
        SIBLINGS,
        TIMED_OUT * 10000,
        "waiter status * 10000 + notifier woken",
    );
}

/// The root holds 7 at 32768; the child's wait for 0 at 32768 reads its own window's 0, parks and
/// times out, rather than seeing the root's 7 and returning not-equal.
#[test]
fn a_detached_childs_wait_compares_its_own_window() {
    on_every_driver(ROOT_HOLDS_SEVEN, TIMED_OUT, "the child's wait status");
}
