//! **#1059/#1094 — the NULL guard on the debugger attach/seek path.** The debug harness
//! (`Inspector::fresh_single_root` / `fresh_scheduled`, shared by `attach`, `attach_scheduled`, and
//! `seek`) rebuilds its `Mem` from bare funcs/data, so it once never saw the module's guard extent — a
//! guest that traps on NULL under a direct run silently returned `0` under the debugger (#964 gap 2).
//! Now `attach*` threads the module's guard extent through, so `[0, guard)` seeds `Unmapped` and a NULL
//! dereference faults identically when attached, when replayed via `seek`, and on a scheduled attach.
//! The guard is **unconditional** now (#1094 — the one canonical layout; no `__null_guard` marker), so
//! this holds for every module on every path.

use temen_dap::Debuggee;
use temen_interp::{Inspector, Stop, Trap, Value};

const GUARD: i64 = temen_ir::POWERBOX_NULL_GUARD as i64;

/// func 0: load from the arg (a NULL probe). `memory 17` = 128 KiB, fully mapped; func 0 is also the
/// `_start` export (the powerbox-entry shape), exactly as the interp oracle builds.
const BODY: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
"#;

fn module() -> temen_ir::Module {
    let src = format!("memory 17\nexport 0 func \"_start\" 0\n{BODY}");
    temen_text::parse_module(&src).expect("parse")
}

/// The `Stop` a fault produces: the guest ran to completion with a `MemoryFault`.
fn is_fault(stop: &Stop) -> bool {
    matches!(stop, Stop::Finished(Err(Trap::MemoryFault)))
}

/// A NULL load faults on **attach** across the low guard region, and stays legal (loads zero) at and
/// above the guard — the debugger guards exactly like a direct run.
#[test]
fn attach_null_load_faults_under_the_guard() {
    let m = module();
    for probe in [0i64, 8, GUARD - 8, GUARD - 1] {
        let mut ins = Inspector::attach(&m, 0, &[Value::I64(probe)], 1_000_000);
        assert!(
            is_fault(&Debuggee::run_until_stop(&mut ins)),
            "attach: NULL load({probe}) must fault under the guard"
        );
    }
    // At/above the guard the access is legal.
    for probe in [GUARD, (1 << 17) - 8] {
        let mut ins = Inspector::attach(&m, 0, &[Value::I64(probe)], 1_000_000);
        assert!(
            matches!(Debuggee::run_until_stop(&mut ins), Stop::Finished(Ok(_))),
            "load({probe}) at/above the guard admits"
        );
    }
}

/// A time-travel `seek` rebuilds the run's `Mem` from the same bare funcs/data — the guard must ride
/// that rebuild, so seeking past the faulting op still faults. This is the `seek_single` →
/// `fresh_single_root` reseed.
#[test]
fn seek_replay_faults_under_the_guard() {
    let m = module();
    let mut ins = Inspector::attach(&m, 0, &[Value::I64(0)], 1_000_000);
    assert!(
        is_fault(&Debuggee::seek(&mut ins, 1_000_000)),
        "seek replay re-seeds the guard and faults"
    );
}

/// The scheduled (multithreaded) attach harness (`fresh_scheduled`) reseeds the guard too, so a NULL
/// deref faults under `attach_scheduled` exactly as on the single-threaded path.
#[test]
fn scheduled_attach_faults_under_the_guard() {
    let m = module();
    let mut ins = Inspector::attach_scheduled(&m, 0, &[Value::I64(0)], 1_000_000, Vec::new());
    assert!(
        is_fault(&Debuggee::run_until_stop(&mut ins)),
        "scheduled attach faults on NULL"
    );
}
