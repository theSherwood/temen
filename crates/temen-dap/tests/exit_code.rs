//! The DAP `exited` event (surfaced by the c_interpret migration: the UI's "exited with N"
//! readout): a finished run precedes `terminated` with an `exited` event carrying the guest's
//! exit code — `main`'s return through the on-ramp powerbox's `exit` capability, the returned
//! scalar of a compute session, or a non-zero abnormal code for a trap.

use temen_dap::{DapServer, Json};

mod support;
use support::req;

/// `_start` shape: call the `exit` capability with 42 (what the frontend synthesizes for
/// `int main(){ return 42; }` under the on-ramp powerbox).
const EXIT_42: &str = r#"memory 16
import 0 "exit" (i64) -> ()

func () -> () {
block 0 () {
  v0 = i64.const 42
  call.import 0 (v0)
  return
  }
}
"#;

/// A compute session (deny-all) that returns a scalar directly — the result becomes the code.
const RETURN_7: &str = r#"func () -> (i64) {
block 0 () {
  v0 = i64.const 7
  return v0
  }
}
"#;

/// A guest that traps (unreachable) — an abnormal, non-zero exit.
const TRAPS: &str = r#"func () -> (i64) {
block 0 () {
  unreachable
  }
}
"#;

/// Launch `src` (on the bytecode engine, under the powerbox iff `powerbox`), run to completion,
/// and return the `exited` event's `exitCode` — asserting it precedes `terminated`.
fn run_exit_code(src: &str, powerbox: bool) -> i64 {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(src)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s("bytecode")),
    ];
    if powerbox {
        launch.push(("powerbox", Json::s("onramp")));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(
        out.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .and_then(|r| r.get("success").cloned()),
        Some(Json::Bool(true)),
        "launch ok"
    );
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
    out[exited_at]
        .get("body")
        .and_then(|b| b.get("exitCode"))
        .and_then(|v| v.as_i64())
        .expect("an exitCode")
}

/// **The powerbox exit code**: `exit(42)` surfaces `exitCode: 42`.
#[test]
fn powerbox_exit_surfaces_the_code() {
    assert_eq!(run_exit_code(EXIT_42, true), 42);
}

/// **A returned scalar becomes the code** on a compute session (no powerbox).
#[test]
fn returned_scalar_is_the_exit_code() {
    assert_eq!(run_exit_code(RETURN_7, false), 7);
}

/// **A trap is a non-zero abnormal exit** (1), like a process killed by a fault.
#[test]
fn a_trap_exits_nonzero() {
    assert_eq!(run_exit_code(TRAPS, false), 1);
}

/// **The tree-walker reports the same code** — parity where both engines back the request.
#[test]
fn tree_walker_reports_the_same_code() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RETURN_7)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
        ]),
    ));
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    let code = out
        .iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("exited"))
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("exitCode"))
        .and_then(|v| v.as_i64());
    assert_eq!(
        code,
        Some(7),
        "tree-walker exit code matches the bytecode engine"
    );
}
