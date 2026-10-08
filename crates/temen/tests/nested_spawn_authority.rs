//! #1727 — a confined child's spawn handles resolve in **its own** powerbox, on every driver.
//!
//! Shape: the root spawns a *middle* child detached, and the middle child spawns a *grandchild* and
//! joins it, each through a v1 record. In the refusal cases the middle child names one handle it was
//! never granted but the root holds: another module, a cap it re-grants by name, or a `Budget`.
//! Invariant #3 (authority moves only down the grant graph) says each such spawn fails closed, and #9
//! says every driver fails it exactly as the tree-walk oracle does. The control passes only handles
//! the middle child holds, so the spawn machinery itself is known to work.
#![cfg(unix)]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_capture_reserved_with_host, Host, Region, StreamRole, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 128 << 10;

/// The grandchild: a 64 KiB window whose entry returns a fixed value — reaching it means the spawn
/// was admitted.
const GRAND: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 4242
  return v1
  }
}
";

/// The middle child's spawn of the grandchild: its own `mod`, paid from its own `budget`, except for
/// the one handle a refusal case swaps in.
#[derive(Clone, Copy)]
enum Spawn {
    /// Its own handles: the control.
    Own,
    /// A module it was never granted.
    Module(i32),
    /// Its own module, re-granting a cap it was never granted as `stdout`.
    Granting(i32),
    /// Its own module, paid from a budget it was never granted.
    Budget(i32),
}

/// The middle child (128 KiB, entry `(i64 instantiator) -> (i64)`): spawn the grandchild through the
/// record at 17408 and return what `join` returns.
fn middle(spawn: Spawn) -> String {
    let module = match spawn {
        Spawn::Module(h) => format!("  vmod = i32.const {h}\n"),
        _ => "  mnp = i64.const 16484\n  mnl = i64.const 3\n  vmod = self.resolve mnp mnl\n".into(),
    };
    let budget = match spawn {
        Spawn::Budget(h) => format!("  vbud = i32.const {h}\n"),
        _ => "  bnp = i64.const 16490\n  bnl = i64.const 6\n  vbud = self.resolve bnp bnl\n".into(),
    };
    // One grant record `{name_off=16500, name_len=6, handle, flags=0}` at 16384.
    let (grants, rec) = match spawn {
        Spawn::Granting(h) => (
            format!(
                "  g0 = i64.const 16384\n  gv0 = i32.const 16500\n  i32.store g0 gv0\n  \
                 g4 = i64.const 16388\n  gv4 = i32.const 6\n  i32.store g4 gv4\n  \
                 g8 = i64.const 16392\n  gv8 = i32.const {h}\n  i32.store g8 gv8\n"
            ),
            SpawnRec {
                grants_ptr: 16384,
                grants_n: 1,
                ..SpawnRec::v1(0)
            },
        ),
        _ => (String::new(), SpawnRec::v1(0)),
    };
    format!(
        "memory 17
data 16484 \"mod\"
data 16490 \"budget\"
data 16500 \"stdout\"
{rec}func (i64) -> (i64) {{
block 0 (vs: i64) {{
  vinst = i32.wrap_i64 vs
{module}{budget}{grants}  ma = i64.const 17432
  i32.store ma vmod
  ba = i64.const 17436
  i32.store ba vbud
  rp = i64.const 17408
  ch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  r = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return r
  }}
}}
",
        rec = rec::segment(17408, &rec),
    )
}

/// The root (args `(instantiator, middle module, grandchild module, budget)`): spawn the middle child
/// detached, paid from `budget` and granted the grandchild module as `mod`, and join it.
fn root() -> String {
    let rec = SpawnRec {
        grants_ptr: 16384,
        grants_n: 1,
        ..SpawnRec::v1(0)
    };
    format!(
        "memory 17
data 16484 \"mod\"
{rec}func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmid: i32, vgrand: i32, vbud: i32) {{
  g0 = i64.const 16384
  gv0 = i32.const 16484
  i32.store g0 gv0
  g4 = i64.const 16388
  gv4 = i32.const 3
  i32.store g4 gv4
  g8 = i64.const 16392
  i32.store g8 vgrand
  ma = i64.const 17432
  i32.store ma vmid
  ba = i64.const 17436
  i32.store ba vbud
  rp = i64.const 17408
  ch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  r = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return r
  }}
}}
",
        rec = rec::segment(17408, &rec),
    )
}

fn parse(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// The root's handles that nothing below it holds.
#[derive(Clone, Copy)]
struct Unheld {
    stdout: i32,
    module: i32,
    budget: i32,
}

/// The root's powerbox, laid out slot for slot against the middle child's. A detached child holds its
/// `Instantiator` and `AddressSpace` first, then its own `budget`, then its grants — here `mod`. The
/// root holds a padding stream opposite the `AddressSpace`, a budget opposite the middle's own, and
/// the **same** module opposite `mod`, so each of the middle's own handles names a cap of the same
/// kind in either table: on a driver that looks in the root's table, an unheld handle is then reached
/// rather than masked by an earlier lookup failing first. Everything after is the root's alone.
fn powerbox(spawn: impl Fn(Unheld) -> Spawn) -> (Host, [Value; 4]) {
    let grand = parse(GRAND);
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, WIN as u64);
    host.grant_stream(StreamRole::Err);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let grand_h = host.grant_module(&grand);
    let unheld = Unheld {
        stdout: host.grant_stream(StreamRole::Out),
        module: host.grant_module(&grand),
        budget: host.grant_budget(-1, 1 << 20, -1),
    };
    let mid = host.grant_module(&parse(&middle(spawn(unheld))));
    (
        host,
        [
            Value::I32(inst),
            Value::I32(mid),
            Value::I32(grand_h),
            Value::I32(budget),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
enum Driver {
    Oracle,
    Cooperative,
    Parallel,
    Debug,
}

fn run(driver: Driver, spawn: impl Fn(Unheld) -> Spawn) -> Result<Vec<Value>, Trap> {
    let root = parse(&root());
    let (mut host, args) = powerbox(spawn);
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
                Some(Arc::clone(&back)),
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

/// The oracle refuses the spawn; every other driver must give the oracle's answer.
fn refused_everywhere(spawn: impl Fn(Unheld) -> Spawn + Copy) {
    let want = run(Driver::Oracle, spawn);
    assert!(want.is_err(), "the oracle refuses the spawn, got {want:?}");
    let wrong: Vec<_> = [Driver::Cooperative, Driver::Parallel, Driver::Debug]
        .into_iter()
        .map(|d| (d, run(d, spawn)))
        .filter(|(_, got)| *got != want)
        .collect();
    assert!(
        wrong.is_empty(),
        "resolved the middle child's handle in the root's powerbox (oracle: {want:?}): {wrong:?}",
    );
}

#[test]
fn control_a_middle_child_spawns_with_its_own_handles() {
    for d in [
        Driver::Oracle,
        Driver::Cooperative,
        Driver::Parallel,
        Driver::Debug,
    ] {
        assert_eq!(run(d, |_| Spawn::Own), Ok(vec![Value::I64(4242)]), "{d:?}");
    }
}

#[test]
fn a_middle_child_cannot_spawn_a_module_only_the_root_holds() {
    refused_everywhere(|u| Spawn::Module(u.module));
}

#[test]
fn a_middle_child_cannot_grant_a_cap_only_the_root_holds() {
    refused_everywhere(|u| Spawn::Granting(u.stdout));
}

#[test]
fn a_middle_child_cannot_spend_a_budget_only_the_root_holds() {
    refused_everywhere(|u| Spawn::Budget(u.budget));
}
