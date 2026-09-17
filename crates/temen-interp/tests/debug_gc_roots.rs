//! **#1563 — the debug scheduler scans `gc.roots` instead of declining it.**
//!
//! `service_advance` — the shared core of `ScheduledDebugRun`'s `drive` and its reverse-`seek`
//! `tick` — answered `gc.roots` with `Serviced::Declined`. A decline is not a skip: `drive` turns it
//! into `SchedStop::Declined` without ticking the clock, so a guest that collects stopped dead at
//! its first collection and a GC'd language runtime could not be debugged at all. That is un-wired
//! support behind a decline (INVARIANTS #14), on the axis the invariant lists by name, and the same
//! shape as #1528's op-15 decline.
//!
//! Nothing about the op resists stepping — it reads the vCPU's own continuation and writes a guest
//! buffer, without suspending, scheduling or leaving the domain — and the debug path already holds
//! every input: this task's `VTask`, the run's fiber registry, and the window each seam selects the
//! same way. The scan is `gc_scan`, shared with `step_vcpu`, so both tiers answer with the same set
//! rather than two approximations of it (INVARIANTS #9's observability corollary, #15's one path).

use std::collections::BTreeSet;
use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{Trap, Value};
use temen_text::parse_module;

/// Where the guest asks for its roots, and the heap range it declares.
const BUF: u64 = 16384;
const LO: u64 = 4096;
const HI: u64 = 8192;

/// The guest: four constants live in the calling frame, one duplicated and one out of range, then
/// `gc.roots` over `[LO, HI)`. It returns the total, so the result alone proves the op ran.
const GUEST: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  va = i64.const 4096
  vb = i64.const 5000
  vc = i64.const 5000
  vd = i64.const 9000
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 16384
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
"#;

/// A collecting guest that also **spawns a fiber and parks it**, so the scan has to reach past the
/// active frame into the resume chain and the registry — the coverage GC.md §3.1 requires and the
/// part a naive "scan the current frame" debug implementation would miss.
const GUEST_WITH_PARKED_FIBER: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  va = i64.const 4096
  vlo = i64.const 4096
  vhi = i64.const 8192
  vmask = i64.const -1
  vbuf = i64.const 16384
  vcap = i64.const 64
  vt = gc.roots vlo vhi vmask vbuf vcap
  return vt
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vroot = i64.const 6000
  vy = suspend vroot
  return vy
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Run to completion on the debug scheduler, returning `(result, the window image)`.
fn debug_run(m: &temen_ir::Module) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut run = ScheduledDebugRun::new(m, 0, &[]).expect("in the debug engine's subset");
    let mut fuel = 1_000_000u64;
    let result = loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => break r,
            SchedStop::Break { .. } => continue,
            other => panic!("the debug scheduler must drive gc.roots to completion, got {other:?}"),
        }
    };
    let image = run
        .read_window(0, (BUF + 8 * 64) as usize)
        .expect("the run's window");
    (result, image)
}

/// The roots a run wrote: its `i64` total, and the `total` little-endian words at `BUF`.
fn roots_of(res: &Result<Vec<Value>, Trap>, image: &[u8]) -> (i64, BTreeSet<u64>) {
    let total = match res {
        Ok(v) => match v.first() {
            Some(Value::I64(t)) => *t,
            other => panic!("expected an i64 total, got {other:?}"),
        },
        Err(e) => panic!("unexpected trap: {e:?}"),
    };
    let mut set = BTreeSet::new();
    for i in 0..total as usize {
        let off = BUF as usize + i * 8;
        set.insert(u64::from_le_bytes(image[off..off + 8].try_into().unwrap()));
    }
    (total, set)
}

/// The decline, gone: the session runs the op and finishes, instead of `SchedStop::Declined`.
#[test]
fn the_debug_scheduler_runs_a_collecting_guest_to_completion() {
    let m = module(GUEST);
    let (res, image) = debug_run(&m);
    let (total, set) = roots_of(&res, &image);
    assert_eq!(total as usize, set.len(), "total must equal the set size");
    assert_eq!(
        set,
        BTreeSet::from([4096, 5000]),
        "the caller's in-range constants, deduplicated, with 9000 filtered out"
    );
}

/// The observability corollary: the debug tier reports the **same** roots as the cooperative driver,
/// because both call `gc_scan`. A second implementation here would be a second answer to "what is a
/// root", which is what the shared scan exists to prevent.
#[test]
fn the_debug_scan_agrees_with_the_cooperative_driver() {
    for src in [GUEST, GUEST_WITH_PARKED_FIBER] {
        let m = module(src);
        let (dbg_res, dbg_image) = debug_run(&m);
        let (dbg_total, dbg_set) = roots_of(&dbg_res, &dbg_image);

        let init = vec![0u8; (BUF + 8 * 64) as usize];
        let mut fuel = 1_000_000u64;
        let (coop_res, coop_image) =
            bytecode::compile_and_run_capture(&m, 0, &[], &mut fuel, &init)
                .expect("the cooperative driver supports gc.roots");
        let (coop_total, coop_set) = roots_of(&coop_res, &coop_image);

        assert_eq!(dbg_total, coop_total, "totals must agree");
        assert_eq!(
            dbg_set, coop_set,
            "the debug tier must report the same roots as the driver it mirrors"
        );
        assert!(
            dbg_set.iter().all(|w| (LO..HI).contains(w)),
            "every reported word is an in-window guest value: {dbg_set:?}"
        );
    }
}

/// And the parked fiber's root is actually reached — the scan covers the registry, not just the
/// frame the debugger happens to be stopped in (GC.md §3.1).
#[test]
fn a_parked_fibers_root_is_scanned_under_the_debugger() {
    let m = module(GUEST_WITH_PARKED_FIBER);
    let (res, image) = debug_run(&m);
    let (_, set) = roots_of(&res, &image);
    assert!(
        set.contains(&6000),
        "the suspended fiber holds 6000 across its park; the scan must see it: {set:?}"
    );
}
