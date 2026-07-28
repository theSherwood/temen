//! Debugging §14 **coroutine step-into on the multi-vCPU engine** (debugging slice 16). Single-vCPU
//! `DebugRun` got coroutine step-into in slices 14b/14c; this brings it to the `ScheduledDebugRun`, where
//! the coroutine's vCPU is **pinned** across the body — a `resume` is atomic w.r.t. other vCPUs, so the
//! scheduler never interleaves another thread mid-body. So a breakpoint fires inside a coroutine body on
//! the right thread while a sibling vCPU stays frozen, inspection reads the coroutine's confined window,
//! and reverse `tick`-replay stays deterministic — bit-identical to the production bytecode engine.
//!
//! The sibling vCPU here is a §14 `instantiate` **executor child**, not a `thread.spawn` worker: the
//! bytecode engine rejects `coroutine + thread` in one module (that combination is tree-walker-only,
//! `compile_module`), while `instantiate + coroutine` compiles and gives a real second scheduled vCPU.
//! Fixture: the root instantiates a child (func 2, → 50) and spawns a coroutine (func 1, yield 100 then
//! return 200), resumes the coroutine twice, joins the child, and returns 200 + 50 = 250.

use svm_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use svm_interp::{bytecode, Host, IrPc, Value};
use svm_text::parse_module;

const SRC: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (vinst: i32) {
  ventry_c = i64.const 2
  voff_c = i64.const 65536
  vslog = i64.const 15
  vq = i64.const 0
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry_c, voff_c, vslog, vq)
  ventry_co = i64.const 1
  voff_co = i64.const 98304
  vco = cap.call 6 2 (i64, i64, i64, i64) -> (i32) vinst (ventry_co, voff_co, vslog, vq)
  vrv = i64.const 0
  vs1, vcv1 = cap.call 6 3 (i32, i64) -> (i32, i64) vinst (vco, vrv)
  vs2, vcv2 = cap.call 6 3 (i32, i64) -> (i32, i64) vinst (vco, vrv)
  vcw = cap.call 6 1 (i32) -> (i64) vinst (vch)
  vsum = i64.add vcv2 vcw
  return vsum
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 100
  v3 = cap.call 7 0 (i64) -> (i64) v1 (v2)
  v4 = i64.const 200
  return v4
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 50
  return v1
  }
}
"#;

const WANT: i64 = 250; // coroutine returns 200, instantiate child returns 50

fn sched_session() -> ScheduledDebugRun {
    let m = parse_module(SRC).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst)], host)
        .expect("scheduled debug engine drives an instantiate + coroutine module")
}

fn drive_to_end(run: &mut ScheduledDebugRun, fuel: &mut u64) -> Result<Vec<Value>, ()> {
    loop {
        match run.run_until_stop(fuel) {
            SchedStop::Finished(r) => return r.map_err(|_| ()),
            SchedStop::Break { .. } => continue,
            SchedStop::Blocked | SchedStop::Declined => return Err(()),
        }
    }
}

/// The coroutine body's first op (func 1) and the instantiate child's first op (func 2).
fn coro_first_op() -> IrPc {
    IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 0,
    }
}
fn child_first_op() -> IrPc {
    IrPc {
        module: 0,
        func: 2,
        block: 0,
        inst: 0,
    }
}

/// The instantiate + coroutine round-trip runs correctly on the scheduled debugger and matches the
/// production bytecode engine (the oracle — `run_with_host` doesn't drive scheduler children).
#[test]
fn scheduled_coroutine_matches_the_oracle() {
    let mut r = sched_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(WANT)]), "200 + 50");

    let m = parse_module(SRC).unwrap();
    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine drives instantiate + coroutine");
    assert_eq!(bc.map_err(|_| ()), res, "debug run ≡ production bytecode");
}

/// A breakpoint inside the coroutine body fires on the scheduled engine, and while the coroutine is
/// single-stepped the sibling **instantiate-child vCPU stays frozen** — the coroutine's vCPU is pinned
/// (a `resume` is atomic w.r.t. other vCPUs).
#[test]
fn breakpoint_inside_a_scheduled_coroutine_and_sibling_stays_frozen() {
    let mut r = sched_session();
    r.set_breakpoints(vec![coro_first_op()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(pc, coro_first_op(), "stopped inside the coroutine body");
            assert_eq!(reason, SchedBreak::Breakpoint);
        }
        other => panic!("expected a coroutine breakpoint, got {other:?}"),
    }
    let coro_thread = r.stopped_task().expect("stopped");
    assert_eq!(
        r.frame_pc(0),
        Some(coro_first_op()),
        "focused on the coroutine body"
    );

    // The instantiate child is the other live vCPU; it hasn't run (the root pinned itself first).
    let child = r
        .threads()
        .into_iter()
        .find(|&t| t != coro_thread)
        .expect("the instantiate child vCPU is live");
    assert!(r.select_task(child));
    assert_eq!(
        r.frame_pc(0),
        Some(child_first_op()),
        "child parked at its entry — it never ran while the root was mid-coroutine"
    );

    // Single-step the coroutine a few times; the child must not advance (the pin holds).
    r.select_task(coro_thread);
    for _ in 0..2 {
        r.step(&mut fuel);
    }
    assert!(r.select_task(child));
    assert_eq!(
        r.frame_pc(0),
        Some(child_first_op()),
        "sibling vCPU stayed frozen while the coroutine was single-stepped"
    );

    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes: a fresh scheduled session ticked to the turn reached inside the coroutine
/// body reproduces the exact coroutine position — the pin makes the replay's op sequence deterministic.
#[test]
fn scheduled_coroutine_tick_replays_deterministically() {
    let mut a = sched_session();
    a.set_breakpoints(vec![coro_first_op()]);
    let mut fuel = 5_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let coro_thread = a.stopped_task().unwrap();

    let mut b = sched_session();
    let mut f2 = 5_000_000u64;
    while b.op_turn() < turn && b.tick(&mut f2) {}
    b.locate();
    assert_eq!(b.op_turn(), turn, "replayed to the same turn");
    assert!(b.select_task(coro_thread));
    assert_eq!(
        b.frame_pc(0),
        Some(coro_first_op()),
        "replay reproduced the coroutine-body position"
    );
}
