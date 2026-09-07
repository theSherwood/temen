//! **#1229 — value watchpoints on SSA-held (address-less) source variables.** A chibicc-promoted
//! scalar local has no window address (`VarLoc::Ssa`/`SsaList`), so the window-range watchpoint
//! can't reach it. `Inspector::set_value_watchpoint` watches such a variable *by value*: the run
//! pauses (`StopReason::Watchpoint`) when the variable's holding value changes. These pin the
//! tree-walker behavior; `dap_bytecode.rs` covers the bytecode-engine parity.

use temen_interp::{Inspector, IrPc, Stop, StopReason, Value, WatchKind};

/// A function whose source variable `x` is held by different SSA values through the block (the
/// `SsaList` promoted-scalar case): `x = 10` (value 0, `va`) until inst 3, then `x = 11` (value 2,
/// `vc`). The loclist entries start *after* each defining op so the holding value is always live.
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

/// Arm a value watch on `x` while stopped at inst 1 (where `x == 10`); continuing pauses with a
/// `Watchpoint` at inst 3, the op after `x` takes its new value (11).
#[test]
fn value_watch_fires_when_an_ssalist_local_changes() {
    let m = module();
    let mut ins = Inspector::attach(&m, 0, &[], 1_000_000);
    // Stop at inst 1 so `x` (value 0, = 10) is live to arm against.
    ins.set_breakpoint(at(1));
    assert!(
        matches!(ins.run_until_stop(), Stop::Break { reason: StopReason::Breakpoint, pc } if pc == at(1)),
        "reach the arming point"
    );
    let id = ins
        .set_value_watchpoint(0, "x", WatchKind::Write)
        .expect("x is an SSA-located local, so it takes a value watch");
    // The value watch fires when `x` becomes 11 — reported before the first op that observes the
    // new holding value (inst 3).
    match ins.run_until_stop() {
        Stop::Break {
            reason: StopReason::Watchpoint { .. },
            pc,
        } => assert_eq!(pc, at(3), "pauses when x's value changes"),
        other => panic!("expected a value-watch stop, got {other:?}"),
    }
    // Clearing it lets the run finish. `x` became 11 (`vc`) at the stop; the function returns
    // `vd = vc + vb = 12`.
    assert!(ins.clear_watchpoint(id), "the value watch was present");
    match ins.run_until_stop() {
        Stop::Finished(Ok(vals)) => {
            assert!(
                matches!(vals.first(), Some(Value::I64(12))),
                "returns vd = 12"
            )
        }
        other => panic!("expected a clean finish, got {other:?}"),
    }
}

/// Without the value watch, the same run never pauses mid-function — it runs straight to completion.
#[test]
fn no_value_watch_no_stop() {
    let m = module();
    let mut ins = Inspector::attach(&m, 0, &[], 1_000_000);
    assert!(
        matches!(ins.run_until_stop(), Stop::Finished(Ok(_))),
        "an unwatched run finishes without stopping"
    );
}

/// A value watch is refused for a variable with no SSA location — a name that doesn't exist, and
/// (were it present) a memory-located var, which is the window-range watch's job.
#[test]
fn value_watch_refused_without_an_ssa_location() {
    let m = module();
    let mut ins = Inspector::attach(&m, 0, &[], 1_000_000);
    ins.set_breakpoint(at(1));
    ins.run_until_stop();
    assert!(
        ins.set_value_watchpoint(0, "nonexistent", WatchKind::Write)
            .is_none(),
        "an unknown variable has no SSA location"
    );
}
