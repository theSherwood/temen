//! Debugging a §14 child that runs a **host-granted separate `Module`** on the multi-vCPU
//! `ScheduledDebugRun` (debugging slice 15b). The child runs on the cooperative debug scheduler as its
//! own vCPU: the debug engine resolves the module from the powerbox, compiles it, and pushes it to the
//! shared source (the mutable-`Domain` step), then seeds its data segments into the child's window. So
//! a breakpoint fires inside the **granted module's** body (a distinct pushed module index) on a
//! distinct thread, `read_window` reads the child's own window, and the run stays bit-identical to the
//! production bytecode engine + tree-walker oracle. The child is spawned detached, by a spawn record; a
//! detached child's checkpoints are `bytecode_debug_detached.rs`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use temen_interp::{bytecode, run_with_host, Host, IrPc, Value, VarValue};
use temen_ir::SpawnRec;
use temen_text::parse_module;

// The granted "plugin" module: a 64 KiB window with a data segment "VM" at offset 16 KiB + 100 (above
// the NULL guard its carve reserves like any window, #1206). Its entry `(i64 instantiator) -> (i64)`
// loads its own data byte there ('V' = 86), stores marker 7 at offset 16 KiB, and returns
// byte + 1000 = 1086 — a foreign module's code + data + window write, confined.
const CHILD_SRC: &str = r#"memory 16
data 16484 "VM"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.const 16384
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
  }
}
"#;

// The parent `(instantiator, module, budget) -> i64` spawns the granted module through a spawn record at
// 17408 paid from the budget, joins it, and returns its value (1086).
const DETACHED_PARENT: &str = r#"memory 17
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vm = i64.const 17416
  i32.store vm v1
  vb = i64.const 17420
  i32.store vb v2
  vr = i64.const 17408
  v6 = call.cap 6 17 (i64) -> (i32) v0 (vr)
  v7_st, v7 = call.cap 6 1 (i32) -> (i64, i64) v0 (v6)
  return v7
  }
}
"#;

const WANT: i64 = 1086;

/// A powerbox with a granted `Instantiator` (over the whole window) + a `Module` grant for `CHILD_SRC`.
fn powerbox() -> (Host, i32, i32) {
    let child = parse_module(CHILD_SRC).expect("parse child");
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let mh = host.grant_module(&child);
    (host, inst, mh)
}

/// [`DETACHED_PARENT`] with its record, and its powerbox and args: [`powerbox`] and a `Budget` for the
/// child's window.
fn detached() -> (temen_ir::Module, Host, [Value; 3]) {
    let src = format!(
        "{DETACHED_PARENT}{}",
        rec::segment(17408, &SpawnRec::default())
    );
    let m = parse_module(&src).expect("parse");
    let (mut host, inst, mh) = powerbox();
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (m, host, [inst, mh, budget].map(Value::I32))
}

/// A `ScheduledDebugRun` on [`detached`].
fn detached_session() -> ScheduledDebugRun {
    let (m, host, args) = detached();
    ScheduledDebugRun::new_with_host(&m, 0, &args, host)
        .expect("scheduled debug engine must drive a detached module child")
}

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

/// A module child's round-trip runs correctly on the scheduled debugger and matches the production
/// bytecode engine **and** the tree-walker oracle.
#[test]
fn a_module_child_debug_run_matches_the_oracle() {
    let mut r = detached_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(res, Ok(vec![Value::I64(WANT)]), "child returns 86 + 1000");

    let (m, mut h_bc, args) = detached();
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &args, &mut f_bc, &mut h_bc)
        .expect("bytecode engine drives the module child");
    assert_eq!(
        bc.clone().map_err(|_| ()),
        res,
        "debug run ≡ production bytecode"
    );

    let (m, mut h_tw, args) = detached();
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &args, &mut f_tw, &mut h_tw);
    assert_eq!(tw, bc, "bytecode ≡ tree-walker");
}

/// A breakpoint in the **granted module's** body fires on a distinct thread — the separate-module child,
/// spawned detached, is its own scheduled vCPU running its own pushed module index.
#[test]
fn breakpoint_in_a_module_child_fires() {
    let mut r = detached_session();
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

/// #2239 — [`CHILD_SRC`]'s §6 info as its own source, `plugin.c`, would give it: the load at line
/// 3, what follows the marker's store at line 5, and the loaded byte named `byte`.
const PLUGIN_DEBUG: &str = r#"
debug.file 0 "plugin.c"
debug.fname 0 "plugin"
debug.loc 0 0 0 0 3 3
debug.loc 0 0 5 0 5 3
debug.var 0 "byte" ssa 2 "unsigned char"
"#;

/// #2239 — a line of a granted module's own source binds when the run admits the module. Before the
/// spawn no module of the run has `plugin.c`, so the lines resolve nowhere and stay where they are.
/// Line 5 then binds in the child and stops it there, its frame read against the module's own §6
/// info; line 4, without code in it, binds nowhere.
#[test]
fn a_line_of_a_childs_own_source_binds_when_the_run_admits_it() {
    let child = parse_module(&format!("{CHILD_SRC}{PLUGIN_DEBUG}")).expect("parse child");
    let src = format!(
        "{DETACHED_PARENT}{}",
        rec::segment(17408, &SpawnRec::default())
    );
    let m = parse_module(&src).expect("parse");
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let mh = host.grant_module(&child);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let args = [inst, mh, budget].map(Value::I32);
    let mut r = ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("in subset");
    assert_eq!(
        r.set_line_breakpoints("plugin.c", vec![4, 5]),
        vec![None, None],
        "no module of the run has plugin.c yet"
    );
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(
                (pc, reason),
                (child_module_after_store(), SchedBreak::Breakpoint)
            );
        }
        other => panic!("expected the child's line, got {other:?}"),
    }
    assert_eq!(
        r.read_var(0, "byte", 4),
        Some(VarValue::Value(Value::I32(86))),
        "the byte the child loaded, read against its own info"
    );
    assert_eq!(
        r.debug_info(1).map(|di| di.files.clone()),
        Some(vec!["plugin.c".to_string()])
    );
    assert!(r.debug_info(0).is_none(), "the parent has no §6 info");
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Stopped inside the module child after its `i32.store8`, `read_window` reads the child's **own**
/// window: the marker byte 7 at its offset 16 KiB, which the root's window does not hold.
#[test]
fn inspect_a_module_childs_window() {
    let mut r = detached_session();
    r.set_breakpoints(vec![child_module_after_store()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, .. } => assert_eq!(pc, child_module_after_store()),
        other => panic!("expected the child breakpoint, got {other:?}"),
    }
    let child = r.stopped_task().expect("stopped");
    assert!(r.select_task(child));
    assert_eq!(
        r.read_window(16384, 1).unwrap(),
        vec![7u8],
        "reads the granted module child's own window"
    );
    assert_eq!(drive_to_end(&mut r, &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes with a module child: a fresh session ticked to a turn reached inside the
/// granted module reproduces the exact child position — `seek` reconstructs the pushed module + env/
/// task set deterministically.
#[test]
fn module_child_tick_replays_deterministically() {
    let mut a = detached_session();
    a.set_breakpoints(vec![child_module_first_op()]);
    let mut fuel = 5_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let child = a.stopped_task().unwrap();

    let mut b = detached_session();
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
