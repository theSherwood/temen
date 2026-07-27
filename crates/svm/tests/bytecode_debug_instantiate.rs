//! Debugging §14 **`instantiate`** (op 0) confined executor children on the multi-vCPU
//! `ScheduledDebugRun` (debugging slice 15a). Unlike a coroutine (driven inline on one vCPU), an
//! `instantiate` child runs on the cooperative debug scheduler as its **own scheduled vCPU** — confined
//! to a power-of-two sub-window of the holder's range (a `nested_view` over the shared backing), with an
//! attenuated powerbox (an `Instantiator` + `AddressSpace`) and a `quota` fuel sub-budget — joinable via
//! the shared §12 thread machinery. So a breakpoint fires **inside the child** on a distinct thread,
//! `select_task`/`read_window` inspect the child's confined window, and the run stays bit-identical to
//! the production bytecode engine + tree-walker oracle.
//!
//! Same `SHARED_MEM` fixture as `bytecode_instantiate.rs`, driven through the scheduled debug engine
//! directly (`ScheduledDebugRun::new_with_host`) rather than the DAP routing.

use svm_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use svm_interp::{bytecode, run_with_host, Host, IrPc, Value};
use svm_text::parse_module;

// Parent (func 0) instantiates the child (func 1) in a 4 KiB window at 64 KiB, joins it, then reads
// back the marker the child wrote into the shared backing. The child writes 123 at its own offset 7
// (→ backing 64 KiB + 7) and returns 42; the parent returns 42 * 1000 + 123 = 42123.
const SHARED_MEM: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
  }
}
"#;

const WANT: i64 = 42123;

/// A `ScheduledDebugRun` on the instantiate kernel carrying a powerbox with a granted `Instantiator`.
fn sched_session() -> ScheduledDebugRun {
    let m = parse_module(SHARED_MEM).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("scheduled debug engine must drive §14 instantiate")
}

/// Drive the schedule to completion (no breakpoints), returning the root's result.
fn drive_to_end(run: &mut ScheduledDebugRun, fuel: &mut u64) -> Result<Vec<Value>, ()> {
    loop {
        match run.run_until_stop(fuel) {
            SchedStop::Finished(r) => return r.map_err(|_| ()),
            SchedStop::Break { .. } => continue,
            SchedStop::Blocked | SchedStop::Declined => return Err(()),
        }
    }
}

/// The child body's first op (func 1, block 0, inst 0) and the op right after its `i32.store8` (inst 3).
fn child_first_op() -> IrPc {
    IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 0,
    }
}
fn child_after_store() -> IrPc {
    IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 3,
    }
}

/// An `instantiate` round-trip runs correctly on the scheduled debugger and matches the production
/// bytecode engine **and** the tree-walker oracle — the whole module was declined before this slice.
#[test]
fn instantiate_debug_run_matches_the_oracle() {
    let mut r = sched_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(WANT)]), "42 * 1000 + 123");

    let m = parse_module(SHARED_MEM).unwrap();
    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine drives instantiate");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// A breakpoint in the child's body fires on a **distinct thread** (not the root) — the `instantiate`
/// child is its own scheduled vCPU — and `select_task`/`frame_pc` inspect that thread's frame.
#[test]
fn breakpoint_in_an_instantiate_child_fires() {
    let mut r = sched_session();
    r.set_breakpoints(vec![child_first_op()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(pc, child_first_op(), "stopped at the child's first op");
            assert_eq!(reason, SchedBreak::Breakpoint);
        }
        other => panic!("expected a breakpoint in the child, got {other:?}"),
    }
    let child = r.stopped_task().expect("a thread is stopped");
    assert_ne!(
        child, 0,
        "the instantiate child is a distinct vCPU, not the root"
    );
    assert!(r.threads().contains(&child), "child is a live thread");
    assert!(r.select_task(child), "focus the child");
    assert_eq!(r.frame_pc(0), Some(child_first_op()), "child's frame");

    // Continue to completion — the child finishes and the joined root returns the right result.
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Stopped inside the child (just after its `i32.store8`), `read_window` reads the child's **confined**
/// window: byte 123 at the child's offset 7 (backing 64 KiB + 7), not the shared window's offset 7.
#[test]
fn inspect_an_instantiate_childs_confined_window() {
    let mut r = sched_session();
    r.set_breakpoints(vec![child_after_store()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, .. } => assert_eq!(pc, child_after_store()),
        other => panic!("expected the child breakpoint, got {other:?}"),
    }
    let child = r.stopped_task().expect("stopped");
    assert!(r.select_task(child));
    assert_eq!(
        r.read_window(7, 1).unwrap(),
        vec![123u8],
        "reads the instantiate child's confined window, not the shared one"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes with `instantiate`: a fresh session ticked to a turn reached inside the
/// child reproduces the exact child position — `seek` reconstructs the env/task set deterministically.
#[test]
fn instantiate_child_tick_replays_deterministically() {
    let mut a = sched_session();
    a.set_breakpoints(vec![child_first_op()]);
    let mut fuel = 5_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let child = a.stopped_task().unwrap();

    let mut b = sched_session();
    let mut f2 = 5_000_000u64;
    while b.op_turn() < turn && b.tick(&mut f2) {}
    b.locate();
    assert_eq!(b.op_turn(), turn, "replayed to the same turn");
    assert!(b.select_task(child));
    assert_eq!(
        b.frame_pc(0),
        Some(child_first_op()),
        "replay reproduced the instantiate child's position"
    );
}

// --- Depth-2 nesting (slice 15c) -----------------------------------------------------------------
// Confinement composes to depth 2 under the scheduled debugger: the root (func 0) instantiates a child
// (func 1) in a 4 KiB window at 64 KiB; the child, handed an `Instantiator` over *its* window, itself
// instantiates a grandchild (func 2) in a 1 KiB window at its own offset 2048. Each joins the next; the
// grandchild returns 77, propagated up. Same fixture as `bytecode_instantiate.rs::DEPTH_TWO`.
const DEPTH_TWO: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 1
  v2 = i64.const 65536
  v3 = i64.const 12
  v4 = i64.const 0
  v5 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 171
  i32.store8 v2 v3
  v4 = i64.const 2
  v5 = i64.const 2048
  v6 = i64.const 10
  v7 = i64.const 0
  v8 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (v4, v5, v6, v7)
  v9 = cap.call 6 1 (i32) -> (i64) v1 (v8)
  return v9
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i64.const 77
  return v3
  }
}
"#;

fn depth_two_session() -> ScheduledDebugRun {
    let m = parse_module(DEPTH_TWO).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("scheduled debug engine drives depth-2 instantiate")
}

/// Nested `instantiate` composes to depth 2 under the scheduled debugger and matches the oracle — a
/// confined child (`env = Some(k)`) itself carves a grandchild env within its own window.
#[test]
fn depth_two_instantiate_matches_the_oracle() {
    let mut r = depth_two_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(77)]), "grandchild returns 77");

    let m = parse_module(DEPTH_TWO).unwrap();
    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine drives depth-2 instantiate");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// A breakpoint in the **grandchild** fires on its own distinct vCPU (task 2), and `read_window` reads
/// the grandchild's confined window (its marker 200), not the child's (171) or the shared window's (0)
/// — three distinct confined windows nested two deep.
#[test]
fn breakpoint_in_a_grandchild_fires() {
    let mut r = depth_two_session();
    // The grandchild (func 2) is `const 0` (inst 0), `const 200` (inst 1), `store8` (inst 2), … — so
    // stop at inst 3, just after the store, when its marker is in its window.
    let after_store = IrPc {
        module: 0,
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
        r.read_window(0, 1).unwrap(),
        vec![200u8],
        "reads the grandchild's own confined window (marker 200)"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(77)]));
}
