//! Debugging §14 **`instantiate_module`** (op 5) confined children on the multi-vCPU
//! `ScheduledDebugRun` (debugging slice 15b). Like op-0 `instantiate` (slice 15a), the child runs on the
//! cooperative debug scheduler as its own confined vCPU — but it runs a **host-granted separate
//! `Module`**: the debug engine resolves it from the powerbox, compiles it, and pushes it to the shared
//! source (the mutable-`Domain` step), then materializes its data segments into the carve. So a
//! breakpoint fires inside the **granted module's** body (a distinct pushed module index) on a distinct
//! thread, `read_window` reads the child's confined window, and the run stays bit-identical to the
//! production bytecode engine + tree-walker oracle.
//!
//! Same `PARENT`/`CHILD_SRC` fixture as `bytecode_separate_module.rs`, driven through the scheduled debug
//! engine directly (`ScheduledDebugRun::new_with_host`).

use svm_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use svm_interp::{bytecode, run_with_host, Host, IrPc, Value};
use svm_text::parse_module;

// The granted "plugin" module: a 64 KiB window with a data segment "VM" at offset 100. Its entry
// `(i64 instantiator) -> (i64)` loads its own data byte at 100 ('V' = 86), stores marker 7 at offset 0,
// and returns byte + 1000 = 1086 — a foreign module's code + data + window write, confined.
const CHILD_SRC: &str = r#"memory 16
data 100 "VM"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 100
  v2 = i32.load8_u v1
  v3 = i64.const 0
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
  }
}
"#;

// Parent `(instantiator, module) -> i64`: instantiate_module the granted module into a 64 KiB carve at
// 64 KiB, join it, return its value (1086).
const PARENT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.extend_i32_s v1
  v3 = i64.const 0
  v4 = i64.const 65536
  v5 = i64.const 16
  v6 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (v2, v3, v4, v5, v3)
  v7 = cap.call 6 1 (i32) -> (i64) v0 (v6)
  return v7
  }
}
"#;

const WANT: i64 = 1086;

/// A powerbox with a granted `Instantiator` (over the whole window) + a `Module` grant for `CHILD_SRC`.
fn powerbox() -> (Host, i32, i32) {
    let child = parse_module(CHILD_SRC).expect("parse child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    (host, inst, mh)
}

/// A `ScheduledDebugRun` on the instantiate_module kernel carrying the powerbox.
fn module_session() -> ScheduledDebugRun {
    let m = parse_module(PARENT).expect("parse");
    let (host, inst, mh) = powerbox();
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst), Value::I32(mh)], host)
        .expect("scheduled debug engine must drive §14 instantiate_module")
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

/// The granted child module's first body op (module **1** — the first push — func 0, block 0, inst 0)
/// and the op right after its `i32.store8` of the marker (inst 5).
fn child_module_first_op() -> IrPc {
    IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 0,
    }
}
fn child_module_after_store() -> IrPc {
    IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 5,
    }
}

/// An `instantiate_module` round-trip runs correctly on the scheduled debugger and matches the
/// production bytecode engine **and** the tree-walker oracle — declined before this slice.
#[test]
fn instantiate_module_debug_run_matches_the_oracle() {
    let mut r = module_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(WANT)]), "child returns 86 + 1000");

    let m = parse_module(PARENT).unwrap();
    let (mut h_bc, inst_bc, mh_bc) = powerbox();
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(inst_bc), Value::I32(mh_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine drives instantiate_module");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let (mut h_tw, inst_tw, mh_tw) = powerbox();
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(
        &m,
        0,
        &[Value::I32(inst_tw), Value::I32(mh_tw)],
        &mut f_tw,
        &mut h_tw,
    );
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// A breakpoint in the **granted module's** body fires on a distinct thread — the separate-module child
/// is its own scheduled vCPU running its own pushed module index.
#[test]
fn breakpoint_in_a_module_child_fires() {
    let mut r = module_session();
    r.set_breakpoints(vec![child_module_first_op()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(
                pc,
                child_module_first_op(),
                "stopped in the granted module's body"
            );
            assert_eq!(reason, SchedBreak::Breakpoint);
        }
        other => panic!("expected a breakpoint in the module child, got {other:?}"),
    }
    let child = r.stopped_task().expect("a thread is stopped");
    assert_ne!(
        child, 0,
        "the module child is a distinct vCPU, not the root"
    );
    assert!(r.select_task(child));
    assert_eq!(
        r.frame_pc(0),
        Some(child_module_first_op()),
        "child's frame (module 1)"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Stopped inside the module child after its `i32.store8`, `read_window` reads the child's **confined**
/// window: the marker byte 7 at the child's offset 0 (backing 64 KiB), not the shared window's offset 0.
#[test]
fn inspect_a_module_childs_confined_window() {
    let mut r = module_session();
    r.set_breakpoints(vec![child_module_after_store()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, .. } => assert_eq!(pc, child_module_after_store()),
        other => panic!("expected the child breakpoint, got {other:?}"),
    }
    let child = r.stopped_task().expect("stopped");
    assert!(r.select_task(child));
    assert_eq!(
        r.read_window(0, 1).unwrap(),
        vec![7u8],
        "reads the granted module child's confined window, not the shared one"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes with `instantiate_module`: a fresh session ticked to a turn reached inside
/// the granted module reproduces the exact child position — `seek` reconstructs the pushed module + env/
/// task set deterministically.
#[test]
fn module_child_tick_replays_deterministically() {
    let mut a = module_session();
    a.set_breakpoints(vec![child_module_first_op()]);
    let mut fuel = 5_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let child = a.stopped_task().unwrap();

    let mut b = module_session();
    let mut f2 = 5_000_000u64;
    while b.op_turn() < turn && b.tick(&mut f2) {}
    b.locate();
    assert_eq!(b.op_turn(), turn, "replayed to the same turn");
    assert!(b.select_task(child));
    assert_eq!(
        b.frame_pc(0),
        Some(child_module_first_op()),
        "replay reproduced the granted-module child's position"
    );
}
