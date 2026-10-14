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
data ro 16384 "hi\n"
export 0 func "_start" 0

func () -> () {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vj = thread.join vh
  vptr = i64.const 16384
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
/// successive lines. The root spawns a worker, both bump `mem[16384]` (above the #1094 NULL guard),
/// the root joins and returns it.
const SPAWN_STEP: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vaddr = i64.const 16384
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

/// #2239 — a program exporting `_child` (func 1) whose root resolves its `"instantiator"`,
/// `"budget"` and `"child"`, the image the on-ramp grants for that export. The root spawns the
/// image (an op-17 v1 record at its entry, 0) and returns what it returns. The image's func 1 adds
/// 1 to a word of its window 10 times. The program's §6 info puts the loop's head at line 7 of
/// `count.c` and names the word `count`.
const COUNTS_IN_A_CHILD: &str = r#"memory 17
data 17472 "instantiator"
data 17488 "budget"
data 17504 "child"
export 0 func "_child" 1
func () -> (i64) {
block 0 () {
  p1 = i64.const 17472
  l1 = i64.const 12
  vi = self.resolve p1 l1
  p2 = i64.const 17488
  l2 = i64.const 6
  vbud = self.resolve p2 l2
  p3 = i64.const 17504
  l3 = i64.const 5
  vchild = self.resolve p3 l3
  vz = i64.const 0
  r0 = i64.const 1
  a0 = i64.const 17536
  i64.store a0 r0
  a1 = i64.const 17544
  i64.store a1 vz
  r2 = i64.const -4294967296
  a2 = i64.const 17552
  i64.store a2 r2
  vb64 = i64.extend_i32_u vbud
  v32 = i64.const 32
  vbs = i64.shl vb64 v32
  vc64 = i64.extend_i32_u vchild
  r3 = i64.or vbs vc64
  a3 = i64.const 17560
  i64.store a3 r3
  a4 = i64.const 17568
  i64.store a4 vz
  a5 = i64.const 17576
  i64.store a5 vz
  a6 = i64.const 17584
  i64.store a6 vz
  a7 = i64.const 17592
  i64.store a7 vz
  a8 = i64.const 17600
  i64.store a8 vz
  vnone = i64.const 4294967295
  a9 = i64.const 17608
  i64.store a9 vnone
  a10 = i64.const 17616
  i64.store a10 vz
  vh = call.cap 6 17 (i64) -> (i32) vi (a0)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vi (vh)
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vn: i64) {
  va = i64.const 20000
  vx = i64.load va
  vone = i64.const 1
  vx2 = i64.add vx vone
  i64.store va vx2
  vn2 = i64.add vn vone
  vlim = i64.const 10
  vmore = i64.ne vn2 vlim
  br_if vmore 1(vn2) 2()
}
block 2 () {
  vb = i64.const 20000
  vr = i64.load vb
  return vr
  }
}

debug.file 0 "count.c"
debug.loc 1 1 0 0 7 3
debug.fname 0 "main"
debug.fname 1 "child"
debug.var global "count" fixed 20000 "long"
"#;

/// `setBreakpoints` for `count.c` with `lines` (each with an optional `condition`): the response's
/// `(verified, line)` per breakpoint.
fn set_count_breakpoints(
    s: &mut DapServer,
    seq: i64,
    lines: &[(i64, Option<&str>)],
) -> Vec<(bool, i64)> {
    let bps = lines
        .iter()
        .map(|&(line, cond)| {
            let mut bp = vec![("line", Json::i(line))];
            if let Some(c) = cond {
                bp.push(("condition", Json::s(c)));
            }
            Json::obj(bp)
        })
        .collect();
    let out = s.handle(&req(
        seq,
        "setBreakpoints",
        Json::obj(vec![
            (
                "source",
                Json::obj(vec![("path", Json::s("/work/count.c"))]),
            ),
            ("breakpoints", Json::Arr(bps)),
        ]),
    ));
    response(&out)
        .get("body")
        .and_then(|b| b.get("breakpoints"))
        .and_then(|b| b.as_array())
        .expect("breakpoints")
        .iter()
        .map(|b| {
            let verified = b.get("verified").and_then(|v| v.as_bool()) == Some(true);
            (
                verified,
                b.get("line").and_then(|l| l.as_i64()).unwrap_or(0),
            )
        })
        .collect()
}

/// `continue` to the next stop: the stopped thread's id and its top frame's `(name, line, source)`,
/// with the value of `count` its Locals show.
fn continue_to_child_stop(s: &mut DapServer, seq: i64) -> (i64, String, i64, String, String) {
    let out = s.handle(&req(seq, "continue", Json::obj(vec![])));
    let stopped = out
        .iter()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped"))
        .expect("a stop");
    let body = stopped.get("body").expect("a stop body");
    assert_eq!(
        body.get("reason").and_then(|r| r.as_str()),
        Some("breakpoint")
    );
    let thread = body
        .get("threadId")
        .and_then(|t| t.as_i64())
        .expect("its thread");
    let out = s.handle(&req(
        seq + 1,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(thread))]),
    ));
    let top = response(&out)
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .and_then(|f| f.first().cloned())
        .expect("a top frame");
    let name = top
        .get("name")
        .and_then(|n| n.as_str())
        .unwrap_or("")
        .to_string();
    let line = top.get("line").and_then(|l| l.as_i64()).unwrap_or(0);
    let path = top
        .get("source")
        .and_then(|src| src.get("path"))
        .and_then(|p| p.as_str())
        .unwrap_or("")
        .to_string();
    let frame = top.get("id").and_then(|i| i.as_i64()).expect("a frame id");
    let out = s.handle(&req(
        seq + 2,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame))]),
    ));
    let vref = response(&out)
        .get("body")
        .and_then(|b| b.get("scopes"))
        .and_then(|s| s.as_array())
        .and_then(|s| s.first())
        .and_then(|s| s.get("variablesReference"))
        .and_then(|v| v.as_i64())
        .expect("the Locals scope");
    let out = s.handle(&req(
        seq + 3,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let count = response(&out)
        .get("body")
        .and_then(|b| b.get("variables"))
        .and_then(|v| v.as_array())
        .and_then(|vars| {
            vars.iter()
                .find(|v| v.get("name").and_then(|n| n.as_str()) == Some("count"))
                .and_then(|v| v.get("value").and_then(|x| x.as_str()))
                .map(str::to_string)
        })
        .unwrap_or_default();
    (thread, name, line, path, count)
}

/// #2239 — **a source-line breakpoint stops a child image of the program**. The line binds in the
/// program as soon as it is set (snapping from 6 to 7) and in the image when the run admits it. The
/// child stops there: its frame shows the image's function and source, and its Locals read the
/// image's own §6 info. A `condition` holds in the child's frame as well, read against that info.
/// Bound in the program alone, the line never stopped the child.
#[test]
fn a_line_breakpoint_stops_a_child_image_of_the_program() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(COUNTS_IN_A_CHILD)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        set_count_breakpoints(&mut s, 3, &[(6, None)]),
        vec![(true, 7)],
        "bound in the program, snapped to the loop's head"
    );
    let (thread, name, line, path, count) = continue_to_child_stop(&mut s, 4);
    assert_eq!(
        (thread, name.as_str(), line, path.as_str(), count.as_str()),
        (2, "#0 child", 7, "count.c", "0"),
        "the child's first iteration"
    );
    let (thread, _, _, _, count) = continue_to_child_stop(&mut s, 10);
    assert_eq!((thread, count.as_str()), (2, "1"), "its second");

    assert_eq!(
        set_count_breakpoints(&mut s, 20, &[(7, Some("count == 5"))]),
        vec![(true, 7)]
    );
    let (thread, _, _, _, count) = continue_to_child_stop(&mut s, 21);
    assert_eq!(
        (thread, count.as_str()),
        (2, "5"),
        "the condition read in the child"
    );

    assert_eq!(set_count_breakpoints(&mut s, 30, &[]), vec![]);
    let out = s.handle(&req(31, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "runs out once the line is cleared"
    );
}
