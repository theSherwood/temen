//! A module that both spawns §14 children and uses §12 fibers runs on the bytecode engine, and each
//! domain has its **own** fiber registry, as on the tree-walk oracle. The parent parks a fiber of its
//! own, spawns a child (carved, op 5, or detached, op 15) that creates and drives fibers of its own,
//! joins it, and then resumes its parked fiber. The child reports its first fiber's handle, so a
//! registry shared across domains (the child's handles numbered after the parent's, or the parent's
//! fiber reachable from the child) shows up as a different result. Checked on the cooperative
//! driver, the OS-thread parallel driver and the debug scheduler.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_capture_reserved_with_host, Host, Region, Trap, Value};
use temen_ir::Module;
use temen_text::parse_module;
use temen_verify::verify_module;

const PARENT_LOG2: u8 = 23;
const CHILD_LOG2: u8 = 16;

/// A generator fiber: resumed with `a`, suspends with `a + 1`; resumed with `b`, returns `b + 5`.
const FIBER: &str = "func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
  }
}
";

/// The child: its first fiber's handle × 100, plus that fiber's two results (2 + 8).
fn child_src() -> String {
    format!(
        "memory {CHILD_LOG2}
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  v0 = ref.func 1
  v1 = i64.const 0
  h = cont.new v0 v1
  a = i64.const 1
  s1, r1 = cont.resume h a
  b = i64.const 3
  s2, r2 = cont.resume h b
  c = i64.const 100
  hh = i64.mul h c
  t = i64.add r1 r2
  out = i64.add hh t
  return out
  }}
}}
{FIBER}"
    )
}

/// The parent (args `(instantiator, child module, budget)`): parks a fiber at its suspend, spawns and
/// joins the child, resumes the fiber to its return. Result: child status × 1000 + 11 + 25.
fn parent_src(detached: bool) -> String {
    let spawn = if detached {
        "ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz)"
    } else {
        "ch = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, gz, off, sl, gz)"
    };
    format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  f0 = ref.func 1
  z = i64.const 0
  pf = cont.new f0 z
  a = i64.const 10
  s1, r1 = cont.resume pf a
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  off = i64.const 4194304
  sl = i64.const {CHILD_LOG2}
  {spawn}
  vr = call.cap 6 1 (i32) -> (i64) vinst (ch)
  b = i64.const 20
  s2, r2 = cont.resume pf b
  k = i64.const 1000
  x = i64.mul vr k
  y = i64.add r1 r2
  out = i64.add x y
  return out
  }}
}}
{FIBER}"
    )
}

fn parse(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

fn powerbox(child: &Module) -> (Host, [Value; 3]) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let module = host.grant_module(child);
    let budget = host.grant_budget(-1, -1, -1);
    (host, [inst, module, budget].map(Value::I32))
}

fn oracle(parent: &Module, child: &Module) -> Result<Vec<Value>, Trap> {
    let (mut host, args) = powerbox(child);
    let mut fuel = 10_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    run_capture_reserved_with_host(parent, 0, &args, &mut fuel, &init, 0, &mut host).0
}

fn cooperative(parent: &Module, child: &Module) -> Result<Vec<Value>, Trap> {
    let (mut host, args) = powerbox(child);
    let mut fuel = 10_000_000u64;
    bytecode::compile_and_run_with_host(parent, 0, &args, &mut fuel, &mut host)
        .expect("the bytecode engine lowers a module that spawns and uses fibers")
}

fn parallel(parent: &Module, child: &Module) -> Result<Vec<Value>, Trap> {
    let (mut host, args) = powerbox(child);
    let mut fuel = 10_000_000u64;
    let init = vec![0u8; 1 << PARENT_LOG2];
    let back = std::sync::Arc::new(Region::owned_zeroed(1 << PARENT_LOG2, 4096).expect("backing"));
    bytecode::compile_and_run_capture_over_parallel_with_host(
        parent, 0, &args, &mut fuel, &init, back, &mut host,
    )
    .expect("the parallel driver lowers it")
    .0
}

fn debug(parent: &Module, child: &Module) -> Result<Vec<Value>, Trap> {
    let (host, args) = powerbox(child);
    let mut run = ScheduledDebugRun::new_with_host(parent, 0, &args, host)
        .expect("the debug scheduler lowers it");
    let mut fuel = 10_000_000u64;
    loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => return r,
            SchedStop::Break { .. } => {}
            other => panic!("unexpected stop {other:?}"),
        }
    }
}

fn every_driver(detached: bool) {
    let how = if detached { "detached" } else { "carved" };
    let (parent, child) = (parse(&parent_src(detached)), parse(&child_src()));
    let want = oracle(&parent, &child);
    assert_eq!(
        want,
        Ok(vec![Value::I64(10 * 1000 + 11 + 25)]),
        "oracle, {how}: the child's first fiber is its handle 0"
    );
    assert_eq!(cooperative(&parent, &child), want, "cooperative, {how}");
    assert_eq!(parallel(&parent, &child), want, "parallel, {how}");
    assert_eq!(debug(&parent, &child), want, "debug, {how}");
}

#[test]
fn a_carved_child_has_its_own_fiber_registry() {
    every_driver(false);
}

#[test]
fn a_detached_child_has_its_own_fiber_registry() {
    every_driver(true);
}

/// The parent (args `(instantiator, child module, budget, address space)`): creates a 64 KiB region,
/// maps it at window offset 1 MiB, spawns the child detached with the region pre-mapped at the
/// child's 64 KiB, then spin-`notify`s region byte 0 until a waiter is woken, and joins.
const NOTIFIER: &str = "memory 23
func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32, vas: i32) {
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrh = i32.wrap_i64 vrh64
  vps = call.cap 4 3 () -> (i64) vrh ()
  vwin = i64.const 1048576
  vz = i64.const 0
  vprot = i32.const 3
  vm = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwin, vz, vps, vprot)
  me = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  gz = i64.const 0
  sl = i64.const 17
  coff = i64.const 65536
  ch = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, me, gz, gz, gz, sl, gz, gz, gz, vrh64, coff)
  br 1(vinst, ch)
  }
block 1 (i1: i32, c1: i32) {
  addr = i64.const 1048576
  one = i32.const 1
  n = atomic.notify addr one
  zz = i32.const 0
  more = i32.eq n zz
  br_if more 1(i1, c1) 2(i1, c1)
  }
block 2 (i2: i32, c2: i32) {
  vr = call.cap 6 1 (i32) -> (i64) i2 (c2)
  return vr
  }
}
";

/// The waiter: its fiber waits on the pre-mapped region byte (window 64 KiB, expected 0, no
/// timeout), which parks the FIBER; the child's root polls it with `cont.resume` while it reports
/// `FIBER_PARKED` (3), and returns the fiber's result, the wait status (0 = woken) + 7.
const WAITER: &str = "memory 17
func (i64) -> (i64) {
block 0 (vs: i64) {
  f = ref.func 1
  z = i64.const 0
  h = cont.new f z
  br 1(h)
  }
block 1 (hh: i64) {
  a = i64.const 0
  s, r = cont.resume hh a
  parked = i32.const 3
  again = i32.eq s parked
  br_if again 1(hh) 2(r)
  }
block 2 (res: i64) {
  return res
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  addr = i64.const 65536
  exp = i32.const 0
  inf = i64.const -1
  w = i32.atomic.wait addr exp inf
  w64 = i64.extend_i32_u w
  seven = i64.const 7
  out = i64.add w64 seven
  return out
  }
}
";

fn region_powerbox(child: &Module) -> (Host, [Value; 4]) {
    let (mut host, [i, m, b]) = powerbox(child);
    let aspace = host.grant_address_space(0, 1 << PARENT_LOG2);
    (host, [i, m, b, Value::I32(aspace)])
}

/// A child domain's fiber parked on a shared-region word is woken by a notify from ANOTHER domain's
/// window: fiber parks key on the region's canonical identity, and a notify scans every domain's
/// registry. On the oracle and the cooperative driver; the parallel driver does not yet rendezvous a
/// parent and a detached child on a region byte at all, fibers or not (#1787).
#[test]
fn a_notify_from_another_domain_wakes_a_childs_parked_fiber() {
    let (parent, child) = (parse(NOTIFIER), parse(WAITER));
    let want = Ok(vec![Value::I64(7)]);
    {
        let (mut host, args) = region_powerbox(&child);
        let mut fuel = 50_000_000u64;
        let init = vec![0u8; 1 << PARENT_LOG2];
        let got =
            run_capture_reserved_with_host(&parent, 0, &args, &mut fuel, &init, 0, &mut host).0;
        assert_eq!(got, want, "oracle");
    }
    {
        let (mut host, args) = region_powerbox(&child);
        let mut fuel = 50_000_000u64;
        let got = bytecode::compile_and_run_with_host(&parent, 0, &args, &mut fuel, &mut host)
            .expect("lowers");
        assert_eq!(got, want, "cooperative");
    }
}
