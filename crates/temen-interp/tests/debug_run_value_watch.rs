//! **#1517 slice 2 — value watchpoints (#1229) on the bytecode debug engine** (`ScheduledDebugRun`;
//! the former single-vCPU `DebugRun` carried them since #1229 and collapsed into it in slice 4). Same
//! module as the tree-walker suite (`value_watch.rs`): `x` is held by different SSA values through the
//! block, so it has no window address and only a *value* watch can reach it.
//!
//! Both verbs: arm at inst 1 (`x == 10`), then `continue` stops with the watch reason at inst 3 (the
//! op after `x` takes 11); the same arm plus repeated `step`s stops there too, and clearing the watch
//! lets the run finish with 12. A re-`set` that re-arms the same id keeps its running baseline (no
//! re-fire on the unchanged value).

use temen_interp::bytecode::{SchedBreak, SchedStop, ScheduledDebugRun};
use temen_interp::{IrPc, Value, WatchId, WatchKind};

const SRC: &str = r#"func () -> (i64) {
block 0 () {
  va = i64.const 10
  vb = i64.const 1
  vc = i64.add va vb
  vd = i64.add vc vb
  return vd
  }
}
debug.file 0 "t.c"
debug.var 0 "x" ssalist 2 0 1 0 0 3 2 "int"
"#;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module_debug(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn at(inst: usize) -> IrPc {
    IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst,
    }
}

const FUEL: u64 = 1_000_000;

/// Run to the next stop: the stop pc (`None` at completion), asserting the reason is `expect`.
fn sched_run(run: &mut ScheduledDebugRun, fuel: &mut u64, expect: SchedBreak) -> Option<IrPc> {
    match run.run_until_stop(fuel) {
        SchedStop::Break { pc, reason } => {
            assert_eq!(reason, expect, "stop reason at {pc:?}");
            Some(pc)
        }
        SchedStop::Finished(_) => None,
        other => panic!("unexpected scheduled stop {other:?}"),
    }
}

#[test]
fn value_watch_fires_on_continue_and_step() {
    let m = module();
    let mut fuel = FUEL;
    let id = WatchId::from_raw(7);

    // continue: arm at inst 1, stop at inst 3 with the watch reason, finish after clearing.
    let mut run = ScheduledDebugRun::new(&m, 0, &[]).expect("scheduled engine");
    run.set_breakpoints(vec![at(1)]);
    assert_eq!(
        sched_run(&mut run, &mut fuel, SchedBreak::Breakpoint),
        Some(at(1))
    );
    let target = run
        .resolve_value_watch(0, "x")
        .expect("x is SSA-held, so it takes a value watch");
    run.set_value_watches(vec![(id, target.clone(), WatchKind::Write)]);
    run.set_breakpoints(Vec::new());
    assert_eq!(
        sched_run(
            &mut run,
            &mut fuel,
            SchedBreak::Watchpoint {
                addr: 0,
                write: true
            }
        ),
        Some(at(3)),
        "pauses when x changes"
    );
    assert_eq!(run.take_watch_hit(), Some((0, true)));
    // Re-applying the same id keeps its running baseline: no re-fire on the unchanged 11.
    run.set_value_watches(vec![(id, target, WatchKind::Write)]);
    run.set_value_watches(Vec::new());
    assert_eq!(sched_run(&mut run, &mut fuel, SchedBreak::Step), None);
    assert_eq!(run.result(), Some(&Ok(vec![Value::I64(12)])));

    // step: the watch fires mid-step too (parity with continue and the tree-walker).
    let mut run = ScheduledDebugRun::new(&m, 0, &[]).expect("scheduled engine");
    run.set_breakpoints(vec![at(1)]);
    assert_eq!(
        sched_run(&mut run, &mut fuel, SchedBreak::Breakpoint),
        Some(at(1))
    );
    let target = run.resolve_value_watch(0, "x").expect("target");
    run.set_value_watches(vec![(id, target, WatchKind::Write)]);
    run.set_breakpoints(Vec::new());
    let mut hits = Vec::new();
    loop {
        match run.step(&mut fuel) {
            SchedStop::Break { pc, reason } => {
                if let SchedBreak::Watchpoint { addr, write } = reason {
                    hits.push((pc, (addr, write)));
                }
            }
            SchedStop::Finished(_) => break,
            other => panic!("unexpected scheduled stop {other:?}"),
        }
    }
    assert_eq!(
        hits,
        vec![(at(3), (0, true))],
        "one watch stop, at inst 3, while stepping"
    );
    assert_eq!(run.result(), Some(&Ok(vec![Value::I64(12)])));
}

/// Without a watch, the run never pauses mid-function (inertness, invariant 9b).
#[test]
fn no_value_watch_no_stop() {
    let m = module();
    let mut fuel = FUEL;
    let mut sched = ScheduledDebugRun::new(&m, 0, &[]).expect("scheduled engine");
    assert!(matches!(
        sched.run_until_stop(&mut fuel),
        SchedStop::Finished(Ok(_))
    ));
}

/// A value watch is refused for a name with no SSA location.
#[test]
fn value_watch_refused_without_an_ssa_location() {
    let m = module();
    let mut fuel = FUEL;
    let mut sched = ScheduledDebugRun::new(&m, 0, &[]).expect("scheduled engine");
    sched.set_breakpoints(vec![at(1)]);
    sched.run_until_stop(&mut fuel);
    assert!(sched.resolve_value_watch(0, "nonexistent").is_none());
}
