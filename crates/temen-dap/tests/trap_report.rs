//! **#1190 — trap kind + fault address on the DAP `exited` event.** A run killed by an involuntary
//! trap (the SIGSEGV/SIGILL channel) is not a chosen `exit(k)`; `exitCode` alone collapses every such
//! trap to `1`, indistinguishable from a clean `exit(1)`. So a finished run that trapped now tags the
//! `exited` body with `trap` (the kind) and, for a `MemoryFault`, `faultAddr` (the window-relative
//! faulting guest address — a NULL deref → `0`). A clean exit carries neither field. This is what lets
//! the c_interpret consumer render `Segmentation fault (addr 0x0)` (#1059) instead of `Halted`.

use temen_dap::{DapServer, Json};

mod support;
use support::req;

/// A guest that dereferences NULL: load from address 0. Under the unconditional #1094 guard,
/// `[0, POWERBOX_NULL_GUARD)` is `Unmapped`, so this faults with a `MemoryFault` at address 0.
const NULL_LOAD: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  vl = i64.load v0
  return vl
  }
}
"#;

/// A clean compute session that returns a scalar — no trap.
const RETURN_7: &str = r#"func () -> (i64) {
block 0 () {
  v0 = i64.const 7
  return v0
  }
}
"#;

/// Launch `src` and run to completion; return the `exited` event's `body` (asserting it precedes
/// `terminated`). `engine` selects the backend (`None` = the tree-walker default).
fn exited_body(src: &str, engine: Option<&str>) -> Json {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(src)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
    ];
    if let Some(e) = engine {
        launch.push(("engine", Json::s(e)));
    }
    s.handle(&req(2, "launch", Json::obj(launch)));
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    let exited_at = out
        .iter()
        .position(|m| m.get("event").and_then(|e| e.as_str()) == Some("exited"))
        .expect("an exited event");
    let terminated_at = out
        .iter()
        .position(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated"))
        .expect("a terminated event");
    assert!(exited_at < terminated_at, "exited precedes terminated");
    out[exited_at].get("body").cloned().expect("an exited body")
}

/// Isolation: the `Debuggee::fault_addr` accessor reads the fault address directly after the trap
/// (tree-walker), independent of the DAP server's event plumbing.
#[test]
fn debuggee_fault_addr_isolated() {
    use temen_dap::Debuggee;
    use temen_interp::{Inspector, Stop, Trap};
    let m = temen_text::parse_module(NULL_LOAD).expect("parse");
    let mut ins = Inspector::attach(&m, 0, &[], 1_000_000);
    let stop = Debuggee::run_until_stop(&mut ins);
    assert!(
        matches!(stop, Stop::Finished(Err(Trap::MemoryFault))),
        "null load faults: {stop:?}"
    );
    assert_eq!(
        Debuggee::fault_addr(&ins),
        Some(0),
        "the accessor reads the faulting address after the trap"
    );
}

/// **A NULL deref reports a MemoryFault at address 0** on the bytecode engine — the path c_interpret's
/// DAP-over-wasm session drives.
#[test]
fn null_deref_reports_memory_fault_at_zero_bytecode() {
    let body = exited_body(NULL_LOAD, Some("bytecode"));
    assert_eq!(
        body.get("trap").and_then(|t| t.as_str()),
        Some("MemoryFault"),
        "the trap kind is surfaced: {body:?}"
    );
    assert_eq!(
        body.get("faultAddr").and_then(|a| a.as_i64()),
        Some(0),
        "the faulting address is the NULL pointer (0): {body:?}"
    );
}

/// **Parity: the tree-walker reports the same** trap kind + address.
#[test]
fn null_deref_reports_memory_fault_at_zero_tree_walker() {
    let body = exited_body(NULL_LOAD, None);
    assert_eq!(
        body.get("trap").and_then(|t| t.as_str()),
        Some("MemoryFault")
    );
    assert_eq!(body.get("faultAddr").and_then(|a| a.as_i64()), Some(0));
}

/// **A clean exit carries neither field** — `trap`/`faultAddr` are present only on an abnormal end, so
/// a normal `exit(k)` (here a returned scalar) is never misread as a crash.
#[test]
fn clean_exit_has_no_trap_or_fault_addr() {
    let body = exited_body(RETURN_7, Some("bytecode"));
    assert_eq!(
        body.get("exitCode").and_then(|c| c.as_i64()),
        Some(7),
        "the returned scalar is the exit code"
    );
    assert!(
        body.get("trap").is_none(),
        "no trap on a clean exit: {body:?}"
    );
    assert!(
        body.get("faultAddr").is_none(),
        "no faultAddr on a clean exit: {body:?}"
    );
}
