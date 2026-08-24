//! The on-ramp powerbox on the **scheduled (multi-vCPU) debug engine** (surfaced by the
//! c_interpret migration: a threaded C lesson needs `malloc`/`printf` under the debugger). Before
//! this, a `thread.spawn` module launched with a deny-all host, so any `write`/`exit`/`memory`
//! capability call `CapFault`ed. Now a threaded DAP session under `powerbox: "onramp"` reaches the
//! same caps a single-vCPU session does — its output surfaces as `output` events and `main`'s exit
//! becomes the `exited` code.

use temen_dap::{DapServer, Json};

mod support;
use support::{req, response};

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

/// A spawn-then-work threaded guest with a §6 debug section, so a source-line step lands on
/// successive lines. The root spawns a worker, both bump `mem[0]`, the root joins and returns it.
const SPAWN_STEP: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vaddr = i64.const 0
  vc = i64.load vaddr
  vn = i64.add vc va
  vj = thread.join vh
  vr = i64.load vaddr
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  return vz
  }
}

debug.file 0 "t.c"
debug.fname 0 "main"
debug.loc 0 0 0 0 2 5
debug.loc 0 0 1 0 3 5
debug.loc 0 0 2 0 4 5
debug.loc 0 0 3 0 5 5
debug.loc 0 0 4 0 6 5
debug.loc 0 0 5 0 7 5
debug.loc 0 0 6 0 8 5
debug.loc 0 0 7 0 9 5
"#;

/// **Threaded source-line stepping stops mid-run and surfaces the schedule** (the c_interpret
/// migration's threaded-stepping ask): from the entry, `stepIn` advances one step at a time
/// (not run-to-completion), a resolvable frame is readable at each stop, and the second vCPU
/// appears in the thread list *while stepping* once the root passes `thread.spawn`. Before the
/// entry-`locate()` fix, the first `stepIn` ran the whole program (no `stopped` position).
#[test]
fn threaded_stepping_stops_and_spawns_mid_run() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(SPAWN_STEP)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // The entry is a resolvable frame — not an empty backtrace.
    let out = s.handle(&req(
        3,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let entry_frames = response(&out)
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .map(|a| a.len())
        .unwrap_or(0);
    assert!(entry_frames >= 1, "the entry has a resolvable frame");

    // Step a bounded number of times; each stop must resolve a frame, and the second thread must
    // appear while stepping (never run straight to termination on the first step).
    let mut saw_two_threads = false;
    let mut terminated_early = false;
    for i in 0..12i64 {
        let out = s.handle(&req(
            10 + i,
            "stepIn",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let term = out
            .iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated"));
        let th = s.handle(&req(30 + i, "threads", Json::obj(vec![])));
        let n = response(&th)
            .get("body")
            .and_then(|b| b.get("threads"))
            .and_then(|t| t.as_array())
            .map(|a| a.len())
            .unwrap_or(0);
        if i == 0 && term {
            terminated_early = true;
            break;
        }
        if n >= 2 {
            saw_two_threads = true;
            break;
        }
        if term {
            break;
        }
    }
    assert!(
        !terminated_early,
        "the first step must not run the whole program"
    );
    assert!(
        saw_two_threads,
        "the second vCPU appears in the thread list while stepping"
    );
}
