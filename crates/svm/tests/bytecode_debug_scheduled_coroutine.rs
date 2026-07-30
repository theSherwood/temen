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
use svm_interp::{bytecode, Host, IrPc, Value, VarValue};
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

// --- Separate-module source-variable inspection on the scheduled engine (slice 17) ---------------------
//
// The fixtures above use a *same-module* coroutine (func 1). Below, the coroutine runs a **host-granted
// separate `Module`** (op 6 `spawn_coroutine_module`), pushed to the shared source as module 1 — so
// stepping into its body, `read_var` must resolve the *child's* own `-g` source variables against the
// child's funcs, driven through `ScheduledDebugRun`'s scheduler arm (not the single-vCPU `DebugRun`).

/// The granted "plugin" coroutine module, carrying `-g` info: a 4 KiB window with a data byte `"K"` (75)
/// at offset 0; its entry `(i64 yielder) -> (i64)` returns that byte, and names the loaded value `b`
/// (SSA value index 2). Identical in execution to a debug-info-free module (debug info is strippable).
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

/// Root `(instantiator, module) -> i64`: `spawn_coroutine_module` (op 6) the granted plugin into the carve
/// at 64 KiB, `resume` it to completion, and return its value (75). Driven on the scheduled engine.
const COROUTINE_MODULE: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (vinst: i32, vmod0: i32) {
  vmod = i64.extend_i32_s vmod0
  ventry = i64.const 0
  voff = i64.const 65536
  vslog = i64.const 12
  vco = cap.call 6 6 (i64, i64, i64, i64) -> (i32) vinst (vmod, ventry, voff, vslog)
  vrv = i64.const 0
  vs1, vcv1 = cap.call 6 3 (i32, i64) -> (i32, i64) vinst (vco, vrv)
  return vcv1
  }
}
"#;

const MODWANT: i64 = 75;

fn module_sched_session() -> ScheduledDebugRun {
    let m = parse_module(COROUTINE_MODULE).expect("parse root");
    let child = parse_module(MODULE_CHILD_DBG).expect("parse -g plugin");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(inst), Value::I32(mh)], host)
        .expect("scheduled engine drives §14 spawn_coroutine_module")
}

/// The granted child's `v3 = i64.extend_i32_u v2` op (module **1**, func 0, block 0, inst 2) — the stop
/// *before* it, where the child's source variable `b` (= `v2`, the loaded data byte) is live. The trailing
/// `return` is not a stoppable position (it ends the coroutine).
fn child_module_load() -> IrPc {
    IrPc {
        module: 1,
        func: 0,
        block: 0,
        inst: 2,
    }
}

/// The scheduled separate-module coroutine round-trip runs correctly and matches the production bytecode
/// engine (the oracle) — the whole op-6 arm plus per-module debug metadata.
#[test]
fn scheduled_separate_module_coroutine_matches_the_oracle() {
    let mut r = module_sched_session();
    let mut fuel = 5_000_000u64;
    let res = drive_to_end(&mut r, &mut fuel);
    assert_eq!(
        res,
        Ok(vec![Value::I64(MODWANT)]),
        "coroutine-module returns 75"
    );

    let m = parse_module(COROUTINE_MODULE).unwrap();
    let child = parse_module(MODULE_CHILD_DBG).unwrap();
    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mh_bc = h_bc.grant_module(&child);
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(inst_bc), Value::I32(mh_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine drives spawn_coroutine_module");
    assert_eq!(bc.map_err(|_| ()), res, "debug run ≡ production bytecode");
}

/// **Separate-module source-variable inspection on the scheduled engine**: stepping into the granted
/// module's body, `ScheduledDebugRun::read_var` resolves a name declared in the *child's* `-g` info against
/// the *child's* funcs (module 1), not module 0's — the scheduled counterpart of the single-vCPU slice.
#[test]
fn read_a_source_variable_inside_a_scheduled_separate_module_coroutine() {
    let mut r = module_sched_session();
    r.set_breakpoints(vec![child_module_load()]);
    let mut fuel = 5_000_000u64;
    match r.run_until_stop(&mut fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(
                pc,
                child_module_load(),
                "stopped inside the granted module's body"
            );
            assert_eq!(reason, SchedBreak::Breakpoint);
        }
        other => panic!("expected a coroutine-body breakpoint, got {other:?}"),
    }
    assert_eq!(
        r.frame_pc(0),
        Some(child_module_load()),
        "focused on module 1"
    );

    // The child's own named SSA variable resolves to its live value (the data byte 75), read against the
    // child module's debug info — before this slice a `module != 0` frame gave `None` on this engine too.
    assert_eq!(
        r.read_var(0, "b", 4),
        Some(VarValue::Value(Value::I32(MODWANT as i32))),
        "read the granted module's source variable `b` by name on the scheduled engine"
    );
    assert_eq!(
        r.read_var(0, "nope", 4),
        None,
        "unknown child variable is None"
    );

    assert_eq!(
        drive_to_end(&mut r, &mut fuel),
        Ok(vec![Value::I64(MODWANT)])
    );
}

/// Reverse debugging composes with scheduled separate-module source-variable inspection: a fresh session
/// ticked to the turn reached at the child body reproduces the same live value for `b` — the pushed
/// module's per-module debug metadata is reconstructed deterministically on replay.
#[test]
fn scheduled_separate_module_source_variable_tick_replays_deterministically() {
    let mut a = module_sched_session();
    a.set_breakpoints(vec![child_module_load()]);
    let mut fuel = 5_000_000u64;
    assert!(matches!(
        a.run_until_stop(&mut fuel),
        SchedStop::Break { .. }
    ));
    let turn = a.op_turn();
    let coro_thread = a.stopped_task().unwrap();
    let live = a.read_var(0, "b", 4);
    assert_eq!(live, Some(VarValue::Value(Value::I32(MODWANT as i32))));

    let mut b = module_sched_session();
    let mut f2 = 5_000_000u64;
    while b.op_turn() < turn && b.tick(&mut f2) {}
    b.locate();
    assert_eq!(b.op_turn(), turn, "replayed to the same turn");
    assert!(b.select_task(coro_thread));
    assert_eq!(
        b.frame_pc(0),
        Some(child_module_load()),
        "replay landed at the child body"
    );
    assert_eq!(
        b.read_var(0, "b", 4),
        live,
        "replay reproduced the child's source-variable value"
    );
}

/// A per-turn observation of the scheduled coroutine run: the global turn, every live task's call stack
/// (frames as `m{module}f{func}b{block}i{inst}` — unambiguous for the func/block sanity checks below),
/// and the shared-window bytes spanning the instantiate child's carve (64 KiB) and the coroutine's
/// (96 KiB). A faithful checkpoint restore reproduces all of it.
fn sched_full_obs(run: &mut ScheduledDebugRun) -> (u64, String, Vec<u8>) {
    let turn = run.op_turn();
    let mut stacks = Vec::new();
    for tid in run.threads() {
        run.select_task(tid);
        let mut frames = Vec::new();
        for d in 0..run.depth() {
            if let Some(pc) = run.frame_pc(d) {
                frames.push(format!(
                    "m{}f{}b{}i{}",
                    pc.module, pc.func, pc.block, pc.inst
                ));
            }
        }
        stacks.push(format!("{tid}=[{}]", frames.join(",")));
    }
    let win = run.read_window(65536, 8).unwrap_or_default();
    (turn, stacks.join(";"), win)
}

/// Warm≡cold oracle for **§14 coroutine + `instantiate`-child checkpointing on the multi-vCPU engine**
/// (DEBUGGING.md W1). A scheduled coroutine only ever arises alongside an `instantiate` sibling (the
/// bytecode engine rejects `coroutine + thread`), so this exercises both at once: at every turn where a
/// checkpoint could be taken — including while stepped *inside* the coroutine body — a
/// `ScheduledDebugRun::restore`d run replays forward **identically** to the trusted single from-0 run.
/// The coroutine child `Vm` + its `nested_view`, the `instantiate` child's `DbgEnv` (window +
/// `Instantiator`/`AddressSpace` powerbox), and the whole task set must all round-trip.
#[test]
fn scheduled_coroutine_checkpoint_snapshot_restore_round_trips() {
    const FUEL: u64 = 5_000_000;

    // Reference: one observation per turn from a single from-0 run.
    let mut refr = sched_session();
    let mut f = FUEL;
    let mut ref_obs = vec![sched_full_obs(&mut refr)];
    while refr.tick(&mut f) {
        ref_obs.push(sched_full_obs(&mut refr));
    }
    let total = ref_obs.len() - 1;
    assert_eq!(
        refr.result().unwrap().as_ref().unwrap(),
        &[Value::I64(WANT)]
    );
    // The run genuinely steps inside the coroutine body (func 1) and runs the instantiate child (func 2),
    // so the round-trip below covers a live coroutine *and* a live child env.
    assert!(
        ref_obs.iter().any(|(_, s, _)| s.contains("f1b")),
        "the run steps inside the coroutine body (func 1)"
    );
    assert!(
        ref_obs.iter().any(|(_, s, _)| s.contains("f2b")),
        "the instantiate child runs (func 2)"
    );

    // For every turn C where `snapshot` succeeds, restore into a fresh run and replay forward, checking
    // it matches the reference from C onward. Track that at least one such C was inside the coroutine.
    let mut checkpointed_in_coro = false;
    for c in 0..=total {
        let mut at_c = sched_session();
        let mut f = FUEL;
        while at_c.op_turn() < c as u64 && at_c.tick(&mut f) {}
        let Some(snap) = at_c.snapshot() else {
            continue; // C is outside the checkpointable subset
        };
        if ref_obs[c].1.contains("f1b") {
            checkpointed_in_coro = true;
        }
        let mut warm = sched_session();
        warm.restore(&snap);
        let mut i = c;
        assert_eq!(
            sched_full_obs(&mut warm),
            ref_obs[i],
            "restore at C={c} lands at reference"
        );
        let mut f = FUEL;
        while warm.tick(&mut f) {
            i += 1;
            assert_eq!(
                sched_full_obs(&mut warm),
                ref_obs[i],
                "forward replay after restore at C={c}"
            );
        }
        assert_eq!(i, total, "warm run from C={c} reached the same end");
        assert_eq!(
            warm.result().unwrap().as_ref().unwrap(),
            &[Value::I64(WANT)]
        );
    }
    assert!(
        checkpointed_in_coro,
        "a checkpoint was taken (and restored) while stepped inside the coroutine body"
    );
}
