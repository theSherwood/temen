//! The on-ramp powerbox on the **scheduled (multi-vCPU) debug engine** (surfaced by the
//! c_interpret migration: a threaded C lesson needs `malloc`/`printf` under the debugger). Before
//! this, a `thread.spawn` module launched with a deny-all host, so any `write`/`exit`/`memory`
//! capability call `CapFault`ed. Now a threaded DAP session under `powerbox: "onramp"` reaches the
//! same caps a single-vCPU session does — its output surfaces as `output` events and `main`'s exit
//! becomes the `exited` code.

use svm_dap::{DapServer, Json};

/// A threaded guest under the powerbox: the root spawns a worker (which just returns), joins it,
/// then `write`s "hi\n" to stdout and `exit`s 7 — exercising the stream + exit caps across a spawn.
const THREADED_IO: &str = r#"memory 16
import 0 "write" (i64, i64) -> (i64)
import 1 "exit" (i32) -> ()
data ro 0 "hi\n"
export 0 func "_start" 0

func () -> () {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vj = thread.join vh
  vptr = i64.const 0
  vlen = i64.const 3
  vw = call.import 0 (vptr, vlen)
  vcode = i32.const 7
  call.import 1 (vcode)
  unreachable
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  return vz
  }
}
"#;

fn req(seq: i64, command: &str, args: Json) -> Json {
    Json::obj(vec![
        ("seq", Json::i(seq)),
        ("type", Json::s("request")),
        ("command", Json::s(command)),
        ("arguments", args),
    ])
}

fn response(msgs: &[Json]) -> &Json {
    msgs.iter()
        .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .expect("a response")
}

/// **The powerbox reaches the threaded engine**: a `thread.spawn` guest under `powerbox: "onramp"`
/// launches, its `write` output surfaces as an `output` event, and `main`'s `exit(7)` becomes the
/// `exited` code — none of which worked when the scheduled engine used a deny-all host.
#[test]
fn threaded_session_runs_under_the_powerbox() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(THREADED_IO)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "threaded powerbox launch ok"
    );
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    // The guest's stdout surfaced.
    let stdout: String = out
        .iter()
        .filter(|m| m.get("event").and_then(|e| e.as_str()) == Some("output"))
        .filter_map(|m| {
            m.get("body")
                .and_then(|b| b.get("output"))
                .and_then(|o| o.as_str())
        })
        .collect();
    assert!(
        stdout.contains("hi\n"),
        "the threaded guest's write reached stdout: {stdout:?}"
    );
    // The exit code surfaced.
    let code = out
        .iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("exited"))
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("exitCode"))
        .and_then(|v| v.as_i64());
    assert_eq!(code, Some(7), "main's exit(7) surfaced as the exit code");
}

/// **Reverse still works under the threaded powerbox**: after running out, `seek(0)` rebuilds the
/// session (the rebuild path now carries the powerbox too) and re-driving reproduces the output —
/// the powerbox host is reconstructed on every rebuild, not just at launch.
#[test]
fn threaded_powerbox_survives_seek() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(THREADED_IO)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
        ]),
    ));
    s.handle(&req(3, "continue", Json::obj(vec![])));
    // Rewind to the start: the rebuilt session's output resets to empty.
    let out = s.handle(&req(4, "seek", Json::obj(vec![("t", Json::i(0))])));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "seek(0) ok"
    );
    // Drive forward again: the powerbox is live on the rebuilt run, so the write reappears.
    let out = s.handle(&req(5, "continue", Json::obj(vec![])));
    let stdout: String = out
        .iter()
        .filter(|m| m.get("event").and_then(|e| e.as_str()) == Some("output"))
        .filter_map(|m| {
            m.get("body")
                .and_then(|b| b.get("output"))
                .and_then(|o| o.as_str())
        })
        .collect();
    assert!(
        stdout.contains("hi\n"),
        "re-driven threaded output reappears: {stdout:?}"
    );
}
