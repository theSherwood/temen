//! #1727 — a confined child's spawn handles resolve in **its own** powerbox, on every driver.
//!
//! Shape: the root spawns a *middle* child, and the middle child spawns a *grandchild* and joins it.
//! In the refusal cases the middle child passes handle integers it was never granted but the root
//! holds: another module (op 5), a by-name grant of the root's `stdout` (op 13), or a module and a
//! `Budget` for a detached spawn (op 15). Invariant #3 (authority moves only down the grant graph)
//! says each such spawn fails closed, and #9 says every driver fails it exactly as the tree-walk
//! oracle does. The control passes only handles the middle child holds, so the spawn machinery itself
//! is known to work.
#![cfg(unix)]

use std::sync::Arc;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_capture_reserved_with_host, Host, Region, StreamRole, Trap, Value};
use temen_ir::Module;
use temen_text::parse_module;
use temen_verify::verify_module;

const WIN: usize = 256 << 10;

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

/// The module a spawn names: the one the middle child holds under the name `mod`, or a raw handle.
#[derive(Clone, Copy)]
enum Named {
    Own,
    Handle(i32),
}

/// The middle child's spawn of the grandchild.
#[derive(Clone, Copy)]
enum Spawn {
    /// op 5 (`instantiate_module`) into the upper half of the middle child's window.
    Carve(Named),
    /// op 13 (`instantiate_module_named`) of its own module, re-granting `handle` as `stdout`.
    CarveGranting(i32),
    /// op 15 (`instantiate_detached`) of `module`, charged to `budget`.
    Detached { module: i32, budget: i32 },
}

/// The middle child (128 KiB, entry `(i64 instantiator) -> (i64)`): spawn the grandchild and return
/// what `join` returns.
fn middle(spawn: Spawn) -> String {
    let resolve = "  np = i64.const 16484\n  nl = i64.const 3\n  vmod = self.resolve np nl\n  \
                   me = i64.extend_i32_s vmod\n";
    let (setup, call) = match spawn {
        Spawn::Carve(named) => (
            match named {
                Named::Own => resolve.to_string(),
                Named::Handle(h) => format!("  me = i64.const {h}\n"),
            },
            "call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, ent, off, sl, qz)",
        ),
        // One grant record `{name_off=16500, name_len=6, handle, flags=0}` at 16384.
        Spawn::CarveGranting(h) => (
            format!(
                "{resolve}  g0 = i64.const 16384\n  gv0 = i32.const 16500\n  i32.store g0 gv0\n  \
                 g4 = i64.const 16388\n  gv4 = i32.const 6\n  i32.store g4 gv4\n  \
                 g8 = i64.const 16392\n  gv8 = i32.const {h}\n  i32.store g8 gv8\n  \
                 g12 = i64.const 16396\n  gv12 = i32.const 0\n  i32.store g12 gv12\n  \
                 gp = i64.const 16384\n  gn = i64.const 1\n"
            ),
            "call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst \
             (me, gp, gn, ent, off, sl, qz)",
        ),
        Spawn::Detached { module, budget } => (
            format!("  me = i64.const {module}\n  vb = i64.const {budget}\n  gz = i64.const 0\n"),
            "call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst \
             (vb, me, gz, gz, ent, sl, qz)",
        ),
    };
    format!(
        "memory 17
data 16484 \"mod\"
data 16500 \"stdout\"
func (i64) -> (i64) {{
block 0 (vs: i64) {{
  vinst = i32.wrap_i64 vs
{setup}  ent = i64.const 0
  off = i64.const 65536
  sl = i64.const 16
  qz = i64.const 0
  ch = {call}
  r = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return r
  }}
}}
"
    )
}

/// The root (256 KiB, args `(instantiator, middle module, grandchild module)`): spawn the middle child
/// over the upper 128 KiB and join it. `named` spawns it with op 13, granting the grandchild module as
/// `mod`; otherwise with op 5 and no grants (the debug scheduler does not drive op 13).
fn root(named: bool) -> String {
    let call = if named {
        "  g0 = i64.const 16384\n  gv0 = i32.const 16484\n  i32.store g0 gv0\n  \
         g4 = i64.const 16388\n  gv4 = i32.const 3\n  i32.store g4 gv4\n  \
         g8 = i64.const 16392\n  i32.store g8 vgrand\n  \
         g12 = i64.const 16396\n  gv12 = i32.const 0\n  i32.store g12 gv12\n  \
         gp = i64.const 16384\n  gn = i64.const 1\n  \
         ch = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst \
         (me, gp, gn, ent, off, sl, qz)\n"
    } else {
        "  ch = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vinst (me, ent, off, sl, qz)\n"
    };
    format!(
        "memory 18
data 16484 \"mod\"
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmid: i32, vgrand: i32) {{
  me = i64.extend_i32_s vmid
  ent = i64.const 0
  off = i64.const 131072
  sl = i64.const 17
  qz = i64.const 0
{call}  r = call.cap 6 1 (i32) -> (i64) vinst (ch)
  return r
  }}
}}
"
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
    other: i32,
    budget: i32,
}

/// The root's powerbox, laid out slot for slot against the middle child's: the middle holds its
/// `Instantiator` (slot 0), `AddressSpace` (1) and — when spawned with op 13 — `mod` (2). The root
/// holds a padding stream at 1 and the **same** module at 2, so the middle's own `mod` names it in
/// either table: on a driver that looks in the root's table, an unheld grant is then reached rather
/// than masked by the module lookup failing first. Everything after slot 2 is the root's alone.
fn powerbox(spawn: impl Fn(Unheld) -> Spawn) -> (Host, [Value; 3]) {
    let grand = parse(GRAND);
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, WIN as u64);
    host.grant_stream(StreamRole::Err);
    let grand_h = host.grant_module(&grand);
    let unheld = Unheld {
        stdout: host.grant_stream(StreamRole::Out),
        other: host.grant_module(&grand),
        budget: host.grant_budget(0, 1 << 20, 0),
    };
    let mid = host.grant_module(&parse(&middle(spawn(unheld))));
    (
        host,
        [Value::I32(inst), Value::I32(mid), Value::I32(grand_h)],
    )
}

#[derive(Clone, Copy, Debug)]
enum Driver {
    Oracle,
    Cooperative,
    Parallel,
    Debug,
}

fn run(driver: Driver, named: bool, spawn: impl Fn(Unheld) -> Spawn) -> Result<Vec<Value>, Trap> {
    let root = parse(&root(named));
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

/// The oracle refuses the spawn; every driver in `drivers` must give the oracle's answer.
fn refused_everywhere(drivers: &[Driver], named: bool, spawn: impl Fn(Unheld) -> Spawn + Copy) {
    let want = run(Driver::Oracle, named, spawn);
    assert!(want.is_err(), "the oracle refuses the spawn, got {want:?}");
    let wrong: Vec<_> = drivers
        .iter()
        .map(|&d| (d, run(d, named, spawn)))
        .filter(|(_, got)| *got != want)
        .collect();
    assert!(
        wrong.is_empty(),
        "resolved the middle child's handle in the root's powerbox (oracle: {want:?}): {wrong:?}",
    );
}

#[test]
fn control_a_middle_child_spawns_with_its_own_handles() {
    for d in [Driver::Oracle, Driver::Cooperative, Driver::Parallel] {
        assert_eq!(
            run(d, true, |_| Spawn::Carve(Named::Own)),
            Ok(vec![Value::I64(4242)]),
            "{d:?}",
        );
    }
}

#[test]
fn a_middle_child_cannot_spawn_a_module_only_the_root_holds() {
    let drivers = [Driver::Cooperative, Driver::Parallel, Driver::Debug];
    refused_everywhere(&drivers, false, |u| Spawn::Carve(Named::Handle(u.other)));
}

#[test]
fn a_middle_child_cannot_grant_a_cap_only_the_root_holds() {
    let drivers = [Driver::Cooperative, Driver::Parallel];
    refused_everywhere(&drivers, true, |u| Spawn::CarveGranting(u.stdout));
}

#[test]
fn a_middle_child_cannot_spawn_detached_on_the_roots_module_and_budget() {
    let drivers = [Driver::Cooperative, Driver::Parallel, Driver::Debug];
    refused_everywhere(&drivers, false, |u| Spawn::Detached {
        module: u.other,
        budget: u.budget,
    });
}
