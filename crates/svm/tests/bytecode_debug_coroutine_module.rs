//! Debugging §14 **separate-module** coroutines (`Instantiator.spawn_coroutine_module`, op 6) on the
//! single-vCPU bytecode `DebugRun` (slice 14c). Unlike a same-module coroutine (14a/14b), the child runs
//! a **host-granted separate `Module`**: the debug engine resolves it from the powerbox, compiles it, and
//! **pushes it to the shared source** (the mutable-`Domain` step that previously kept the whole op
//! declined under the debugger), then materializes its data segments into the confined carve. Thereafter
//! the child is `resume`d and stepped exactly like a same-module coroutine — so breakpoints fire in the
//! parent *and* inside the granted module's body, and the run stays bit-identical to the production
//! bytecode engine (`compile_and_run_with_host`), the oracle for this op — `run_with_host` drives only
//! the same-module form.

use svm_interp::bytecode::{self, DebugRun};
use svm_interp::{Host, IrPc, Value, VarValue};
use svm_text::parse_module;

/// The granted "plugin" module a coroutine runs (same as `bytecode_vcpu_coroutine_module.rs`): a 4 KiB
/// window (`memory 12`) with a data segment `"K"` (75) at offset 0; its entry `(i64 yielder) -> (i64)`
/// returns that own data byte (75) — deterministic, and proves data-segment materialization.
const MODULE_CHILD: &str = r#"memory 12
data 0 "K"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
"#;

/// Root `(instantiator, module) -> i64`: `spawn_coroutine_module` the granted module once into the carve
/// at `64 KiB`, `resume` it to completion, and return its value. The module coroutine returns 75.
const COROUTINE_MODULE: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (vinst: i32, vmod0: i32) {
  vmod = i64.extend_i32_s vmod0
  voff = i64.const 65536
  ventry = i64.const 0
  vslog = i64.const 12
  vcoro = cap.call 6 6 (i64, i64, i64, i64) -> (i32) vinst (vmod, ventry, voff, vslog)
  vrv = i64.const 0
  vstatus, vval = cap.call 6 3 (i32, i64) -> (i32, i64) vinst (vcoro, vrv)
  return vval
  }
}
"#;

const WANT: i64 = 75;

/// The same granted "plugin" module, now carrying `-g` §6 debug info: its own function's SSA source
/// variable `b` (the loaded data byte `v2`, value index 2) named over `plugin.c`. Execution is identical
/// to [`MODULE_CHILD`] (debug info is strippable, non-semantic) — it exists to prove a **step-into**
/// resolves the child's *own* source variables, against the child's funcs, not module 0's (the
/// `FrameReader`'s module-0 gate + per-module metadata follow-up, DEBUGGING.md §14).
const MODULE_CHILD_DBG: &str = r#"memory 12
data 0 "K"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  return v3
  }
}
debug.file 0 "plugin.c"
debug.fname 0 "load_k"
debug.type 0 base "int" signed 4
debug.var 0 "b" ssa 2 "int" 0
"#;

/// A fresh powerbox: `Instantiator` over the whole 128 KiB window + a `Module` grant for
/// [`MODULE_CHILD`]. Deterministic, so every engine gets identical handles.
fn powerbox() -> (Host, i32, i32) {
    let child = parse_module(MODULE_CHILD).expect("parse module child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    (host, inst, mh)
}

/// A `DebugRun` on the coroutine-module kernel carrying the powerbox (instantiator + module grant).
fn module_session() -> DebugRun {
    let m = parse_module(COROUTINE_MODULE).expect("parse");
    let (host, inst, mh) = powerbox();
    DebugRun::new_with_host(&m, 0, &[Value::I32(inst), Value::I32(mh)], host)
        .expect("bytecode debug engine must drive §14 spawn_coroutine_module")
}

/// Drive a `DebugRun` to completion, returning its result.
fn drive(run: &mut DebugRun, bps: &[IrPc], fuel: &mut u64) -> Result<Vec<Value>, ()> {
    loop {
        if run.run_to(bps, fuel).is_none() {
            return run.result().cloned().unwrap().map_err(|_| ());
        }
    }
}

/// The parent's `resume` (func 0, block 0, inst 6) and the granted child module's first body op
/// (module **1** — the first push — func 0, block 0, inst 0).
fn parent_resume() -> IrPc {
    IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 6,
    }
}
fn child_module_first_op() -> IrPc {
    IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 0,
    }
}

/// A separate-module coroutine round-trip runs correctly under the debugger and matches the production
/// bytecode engine — the whole module was declined before this slice. The oracle here is the bytecode
/// engine (`compile_and_run_with_host`), not the tree-walker: `run_with_host` returns `Malformed` for
/// `spawn_coroutine_module` (it drives the same-module form only), so the bytecode engine — itself
/// tree-walker-validated on the ops it shares — is the reference, exactly as `bytecode_vcpu_coroutine_
/// module.rs` uses it.
#[test]
fn separate_module_coroutine_debug_run_matches_the_oracle() {
    let mut r = module_session();
    let mut fuel = 5_000_000u64;
    let res = drive(&mut r, &[], &mut fuel);
    assert_eq!(
        res,
        Ok(vec![Value::I64(WANT)]),
        "coroutine-module returns 75"
    );

    let m = parse_module(COROUTINE_MODULE).unwrap();
    let (mut h_bc, inst_bc, mh_bc) = powerbox();
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(inst_bc), Value::I32(mh_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine drives spawn_coroutine_module");
    assert_eq!(
        bc.map_err(|_| ()),
        res,
        "debug run ≡ production bytecode engine"
    );
}

/// A breakpoint in the coroutine **parent** fires across the `spawn_coroutine_module`/`resume` handoff.
#[test]
fn breakpoint_in_a_separate_module_parent_fires() {
    let mut r = module_session();
    let mut fuel = 5_000_000u64;
    assert_eq!(
        r.run_to(&[parent_resume()], &mut fuel),
        Some(parent_resume())
    );
    assert_eq!(
        r.frame_pc(0),
        Some(parent_resume()),
        "stopped at the parent's resume"
    );
    assert_eq!(drive(&mut r, &[], &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Step-into reaches the **granted module's** body: a breakpoint set on the pushed child module (index 1)
/// fires inside the coroutine, the backtrace reports that module, and `read_window` reads the child's
/// confined window — the materialized data byte `75` — proving the separate module is driven op-by-op.
#[test]
fn step_into_a_separate_module_coroutine_body() {
    let mut r = module_session();
    let mut fuel = 5_000_000u64;
    assert_eq!(
        r.run_to(&[child_module_first_op()], &mut fuel),
        Some(child_module_first_op()),
        "descended into the granted module's coroutine body"
    );
    assert_eq!(
        r.frame_pc(0),
        Some(child_module_first_op()),
        "frame is in module 1"
    );
    assert_eq!(r.depth(), 1, "child's root activation");
    assert_eq!(
        r.read_window(0, 1).unwrap(),
        vec![75u8],
        "reads the granted child's confined window (its materialized data segment)"
    );
    assert_eq!(drive(&mut r, &[], &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes with separate-module coroutines: a fresh session ticked to a clock reached
/// inside the granted module's body reproduces the exact child position — `seek` reconstructs the pushed
/// module + active-coroutine state deterministically.
#[test]
fn separate_module_coroutine_body_tick_replays_deterministically() {
    let mut a = module_session();
    let mut fuel = 5_000_000u64;
    assert_eq!(
        a.run_to(&[child_module_first_op()], &mut fuel),
        Some(child_module_first_op())
    );
    let clock = a.op_clock();

    let mut b = module_session();
    let mut f2 = 5_000_000u64;
    while b.op_clock() < clock && b.tick(&mut f2) {}
    assert_eq!(b.op_clock(), clock, "replayed to the same op clock");
    assert_eq!(
        b.frame_pc(0),
        Some(child_module_first_op()),
        "replay reproduced the granted-module coroutine-body position"
    );
}

/// A `DebugRun` on the coroutine-module kernel whose granted child carries `-g` debug info.
fn module_session_dbg() -> DebugRun {
    let m = parse_module(COROUTINE_MODULE).expect("parse");
    let child = parse_module(MODULE_CHILD_DBG).expect("parse -g module child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    DebugRun::new_with_host(&m, 0, &[Value::I32(inst), Value::I32(mh)], host)
        .expect("bytecode debug engine must drive §14 spawn_coroutine_module")
}

/// The granted child's `v3 = i64.extend_i32_u v2` op (module **1**, func 0, block 0, inst 2) — the stop
/// *before* it, where its SSA source variable `b` (= `v2`, the loaded data byte) is live. The trailing
/// `return` is not a stoppable position (it ends the coroutine), so this is the last op inside the body
/// where a named value can be read.
fn child_module_load() -> IrPc {
    IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 2,
    }
}

/// **Separate-module source-variable inspection** (the module-0-gate follow-up): stepping into the granted
/// module's body, `read_var` resolves a name declared in the *child's* `-g` info against the *child's*
/// funcs. Before this slice the `FrameReader` gated every `module != 0` frame to `None` — position-level
/// step-into and window reads worked, but a named source variable in a separate-module child did not.
#[test]
fn read_a_source_variable_inside_a_separate_module_coroutine() {
    let mut r = module_session_dbg();
    let mut fuel = 5_000_000u64;
    // Descend into the granted module and stop where the child's `b` (= v2, the loaded byte) is live.
    assert_eq!(
        r.run_to(&[child_module_load()], &mut fuel),
        Some(child_module_load()),
        "stopped inside the granted module's body"
    );
    assert_eq!(r.frame_pc(0), Some(child_module_load()), "frame is in module 1");
    assert_eq!(r.depth(), 1, "child's root activation");

    // The child's own named SSA variable resolves to its live value (the data byte 75), read against the
    // child module's debug info — not module 0's (before this slice, a `module != 0` frame gave `None`).
    assert_eq!(
        r.read_var(0, "b", 4),
        Some(VarValue::Value(Value::I32(WANT as i32))),
        "read the granted module's source variable `b` by name"
    );
    // A name that isn't a variable of the child's function is unresolved (the child's info is consulted,
    // and it has no such name), never a stray hit from module 0.
    assert_eq!(r.read_var(0, "nope", 4), None, "unknown child variable is None");

    assert_eq!(drive(&mut r, &[], &mut fuel), Ok(vec![Value::I64(WANT)]));
}

/// Reverse debugging composes with separate-module source-variable inspection: a fresh session `tick`ed to
/// the clock reached at the child's `return` reproduces the same live value for `b` — the pushed module's
/// per-module debug metadata is reconstructed deterministically on replay.
#[test]
fn separate_module_source_variable_tick_replays_deterministically() {
    let mut a = module_session_dbg();
    let mut fuel = 5_000_000u64;
    assert_eq!(
        a.run_to(&[child_module_load()], &mut fuel),
        Some(child_module_load())
    );
    let clock = a.op_clock();
    let live = a.read_var(0, "b", 4);
    assert_eq!(live, Some(VarValue::Value(Value::I32(WANT as i32))));

    let mut b = module_session_dbg();
    let mut f2 = 5_000_000u64;
    while b.op_clock() < clock && b.tick(&mut f2) {}
    assert_eq!(b.op_clock(), clock, "replayed to the same op clock");
    assert_eq!(b.frame_pc(0), Some(child_module_load()), "replay landed at the child body");
    assert_eq!(
        b.read_var(0, "b", 4),
        live,
        "replay reproduced the child's source-variable value"
    );
}
