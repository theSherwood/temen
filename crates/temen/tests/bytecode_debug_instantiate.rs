//! Debugging §14 children on the multi-vCPU `ScheduledDebugRun` (debugging slice 15a): a child
//! runs on the cooperative debug scheduler as its **own scheduled vCPU**, joinable via the shared §12
//! thread machinery. Here a depth-2 chain of detached children: the run matches the production
//! bytecode engine and the tree-walk oracle, and a breakpoint fires in the grandchild, whose window is
//! its own. (The carve form this file started on, op 0, is retired — #1867; a detached child's
//! checkpoints and replay are `bytecode_debug_detached.rs`.)

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::bytecode::{SchedStop, ScheduledDebugRun};
use temen_interp::{bytecode, run_with_host, Host, IrPc, Value};
use temen_ir::SpawnRec;
use temen_text::parse_module;

/// Drive the schedule to completion (no breakpoints), returning the root's result.
fn drive_to_end(run: &mut ScheduledDebugRun, fuel: &mut u64) -> Result<Vec<Value>, ()> {
    loop {
        match run.run_until_stop(fuel) {
            SchedStop::Finished(r) => return r.map_err(|_| ()),
            SchedStop::Break { .. } => continue,
            // No blocking stdin in these runs: a stdin park would be as stuck as a deadlock.
            SchedStop::Blocked
            | SchedStop::StdinPark { .. }
            | SchedStop::CapPark { .. }
            | SchedStop::Declined => return Err(()),
        }
    }
}

// The root `(instantiator, budget, c1, c2) -> i64` spawns the child `c1`, func 1's child image
// (#2219), through a v1 record at 17408, paid from its budget and granted `c2`, func 2's image, as
// `"g2"` by the list at 16768. The child resolves its own `"budget"` and `"g2"` and spawns the
// grandchild through the record at 17536, paid from it. Each window is its own, so the grandchild
// writes its marker 200 above its NULL guard, at 16640. Each joins the next; the grandchild returns
// 77, propagated up.
const DETACHED_DEPTH_TWO: &str = r#"memory 17
data 16700 "budget"
data 16720 "g2"
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, vb: i32, vc1: i32, vc2: i32) {
  rb = i64.const 17436
  i32.store rb vb
  rm = i64.const 17432
  i32.store rm vc1
  ga = i64.const 16768
  gn = i32.const 16720
  i32.store ga gn
  gl = i32.const 2
  i32.store ga gl offset=4
  i32.store ga vc2 offset=8
  gz = i32.const 0
  i32.store ga gz offset=12
  rp = i64.const 17408
  v5 = call.cap 6 17 (i64) -> (i32) v0 (rp)
  v6 = call.cap 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 20480
  v3 = i32.const 171
  i32.store8 v2 v3
  np = i64.const 16700
  nl = i64.const 6
  hb = self.resolve np nl
  cb = i64.const 17564
  i32.store cb hb
  gp = i64.const 16720
  gl = i64.const 2
  hg = self.resolve gp gl
  cm = i64.const 17560
  i32.store cm hg
  cp = i64.const 17536
  v8 = call.cap 6 17 (i64) -> (i32) v1 (cp)
  v9 = call.cap 6 1 (i32) -> (i64) v1 (v8)
  return v9
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16640
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i64.const 77
  return v3
  }
}
"#;

/// [`DETACHED_DEPTH_TWO`] with its two spawn records, and a fresh powerbox granting the root the
/// child images of funcs 1 and 2.
fn detached_depth_two() -> (temen_ir::Module, Host, [Value; 4]) {
    let r1 = SpawnRec {
        grants_ptr: 16768,
        grants_n: 1,
        ..SpawnRec::v1()
    };
    let src = format!(
        "{DETACHED_DEPTH_TWO}{}{}",
        rec::segment(17408, &r1),
        rec::segment(17536, &SpawnRec::v1())
    );
    let m = parse_module(&src).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let [c1, c2] =
        [1, 2].map(|f| host.grant_module(&temen_ir::child_image_at(&m, f).expect("child image")));
    (m, host, [inst, budget, c1, c2].map(Value::I32))
}

fn detached_depth_two_session() -> ScheduledDebugRun {
    let (m, host, args) = detached_depth_two();
    ScheduledDebugRun::new_with_host(&m, 0, &args, host)
        .expect("scheduled debug engine drives a detached depth-2 chain")
}

/// The depth-2 chain runs on the scheduled debugger as on the production bytecode engine and the
/// tree-walker oracle.
#[test]
fn depth_two_debug_run_matches_the_oracle() {
    let mut r = detached_depth_two_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(77)]), "grandchild returns 77");

    let (m, mut h_bc, args) = detached_depth_two();
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut f_bc, &mut h_bc)
        .expect("bytecode engine drives the chain");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let (m, mut h_tw, args) = detached_depth_two();
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &args, &mut f_tw, &mut h_tw);
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// A breakpoint in the **grandchild** fires on its own distinct vCPU, and `read_window` reads the
/// grandchild's own window (its marker 200), not the child's or the root's — three windows, two deep.
#[test]
fn breakpoint_in_a_grandchild_fires() {
    let mut r = detached_depth_two_session();
    // The grandchild runs func 2's image, the second module the run admits at a spawn (module 0 is
    // the program, 1 the child's image), and an image keeps its function indices. Its func 2 is
    // `const 0` (inst 0), `const 200` (inst 1), `store8` (inst 2), … — so stop at inst 3, just
    // after the store, when its marker is in its window.
    let after_store = IrPc {
        module: 2,
        func: 2,
        block: 0,
        inst: 3,
    };
    r.set_breakpoints(vec![after_store]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, .. } => assert_eq!(pc, after_store, "stopped in the grandchild"),
        other => panic!("expected a grandchild breakpoint, got {other:?}"),
    }
    let gc = r.stopped_task().expect("stopped");
    assert_ne!(gc, 0, "grandchild is a distinct vCPU");
    assert!(r.select_task(gc));
    assert_eq!(r.frame_pc(0), Some(after_store));
    assert_eq!(
        r.read_window(16640, 1).unwrap(),
        vec![200u8],
        "reads the grandchild's own window (marker 200)"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(77)]));
}
