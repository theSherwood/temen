//! Library frames and just-my-code stepping (#2341). A module's §6 debug info says which of its files
//! hold a library's code (`debug.lib`); the DAP marks those frames `subtle`, and under the `launch`
//! arg `justMyCode` a step does not end in them. Both engines, since the rule is the session's.

use temen_dap::{DapServer, Json};

mod support;
use support::{event, req, response};

// `main` (func 0, main.c) calls `lib` (func 1, lib.c — a library's file, `debug.lib 0`). The call is
// main.c:5 and the line after it main.c:6; `lib`'s two lines are lib.c:10 and lib.c:11.
const LIB_CALL_DBG: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 10
  v2 = i32.add v0 v1
  return v2
  }
}

debug.file 0 "lib.c"
debug.file 1 "main.c"
debug.lib 0
debug.fname 0 "main"
debug.fname 1 "lib"
debug.loc 0 0 0 1 5 1
debug.loc 0 0 1 1 6 1
debug.loc 1 0 0 0 10 1
debug.loc 1 0 1 0 11 1
"#;

// As [`LIB_CALL_DBG`], but `lib` calls back into the program (a `qsort` comparator): `cb` (func 2) is
// main.c's own code, lines main.c:20 and main.c:21, called from lib.c:10.
const LIB_CALLBACK_DBG: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i32.const 1
  v3 = i32.add v1 v2
  return v3
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 2 (v0)
  v2 = i32.const 10
  v3 = i32.add v1 v2
  return v3
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
  }
}

debug.file 0 "lib.c"
debug.file 1 "main.c"
debug.lib 0
debug.loc 0 0 0 1 5 1
debug.loc 0 0 1 1 6 1
debug.loc 1 0 0 0 10 1
debug.loc 1 0 1 0 11 1
debug.loc 2 0 0 1 20 1
debug.loc 2 0 1 1 21 1
"#;

const ENGINES: [Option<&str>; 2] = [None, Some("bytecode")];

/// A session over [`LIB_CALL_DBG`] on `engine`, stopped at the breakpoint `file:line`.
fn stopped_at(engine: Option<&str>, just_my_code: bool, file: &str, line: i64) -> DapServer {
    session(LIB_CALL_DBG, engine, just_my_code, file, line)
}

/// A session over `program` on `engine`, stopped at the breakpoint `file:line`.
fn session(
    program: &str,
    engine: Option<&str>,
    just_my_code: bool,
    file: &str,
    line: i64,
) -> DapServer {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(program)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![Json::i(5)])),
        ("justMyCode", Json::Bool(just_my_code)),
    ];
    if let Some(e) = engine {
        launch.push(("engine", Json::s(e)));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch)));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s(file))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(line))])]),
            ),
        ]),
    ));
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(
        event(&out, "stopped").is_some(),
        "{engine:?}: stops at {file}:{line}"
    );
    s
}

/// The thread's frames, innermost first.
fn frames(s: &mut DapServer) -> Vec<Json> {
    let out = s.handle(&req(
        100,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()
        .to_vec()
}

/// `file:line` of the innermost frame, and how deep the stack is.
fn top(s: &mut DapServer) -> (String, usize) {
    let frames = frames(s);
    let f = &frames[0];
    let file = f
        .get("source")
        .unwrap()
        .get("name")
        .unwrap()
        .as_str()
        .unwrap();
    let line = f.get("line").unwrap().as_i64().unwrap();
    (format!("{file}:{line}"), frames.len())
}

/// Run `command` on thread 1 and report where it stopped.
fn step(s: &mut DapServer, command: &str) -> (String, usize) {
    let out = s.handle(&req(50, command, Json::obj(vec![("threadId", Json::i(1))])));
    assert!(event(&out, "stopped").is_some(), "{command} stops");
    top(s)
}

/// A library's frame reads as one — DAP's `subtle` frame over `deemphasize`d source — and the
/// program's frame below it does not.
#[test]
fn a_library_frame_is_marked_subtle() {
    for engine in ENGINES {
        let mut s = stopped_at(engine, false, "lib.c", 10);
        let frames = frames(&mut s);
        assert_eq!(frames.len(), 2, "{engine:?}: lib over main");
        let hint = |f: &Json| {
            f.get("presentationHint")
                .and_then(|h| h.as_str())
                .map(str::to_owned)
        };
        let source_hint = |f: &Json| {
            f.get("source")
                .and_then(|src| src.get("presentationHint"))
                .and_then(|h| h.as_str())
                .map(str::to_owned)
        };
        assert_eq!(
            hint(&frames[0]).as_deref(),
            Some("subtle"),
            "{engine:?}: lib"
        );
        assert_eq!(
            source_hint(&frames[0]).as_deref(),
            Some("deemphasize"),
            "{engine:?}: lib.c"
        );
        assert_eq!(
            hint(&frames[1]),
            None,
            "{engine:?}: main is the program's own"
        );
        assert_eq!(source_hint(&frames[1]), None, "{engine:?}: main.c");
    }
}

/// Under `justMyCode`, `stepIn` on a call into a library goes through it to the program's next line,
/// where without it the step stops on the library's first line.
#[test]
fn step_in_passes_a_library_call_under_just_my_code() {
    for engine in ENGINES {
        let mut s = stopped_at(engine, false, "main.c", 5);
        assert_eq!(
            step(&mut s, "stepIn"),
            ("lib.c:10".to_string(), 2),
            "{engine:?}: without justMyCode the step goes in"
        );
        let mut s = stopped_at(engine, true, "main.c", 5);
        assert_eq!(
            step(&mut s, "stepIn"),
            ("main.c:6".to_string(), 1),
            "{engine:?}: with it the step stays in the program"
        );
    }
}

/// A breakpoint in a library's code still stops there under `justMyCode`, and a step from there ends
/// in the program's code: the library's next line is stepped out of, back to its caller.
#[test]
fn a_breakpoint_in_a_library_still_stops_and_a_step_leaves_it() {
    for engine in ENGINES {
        let mut s = stopped_at(engine, true, "lib.c", 10);
        assert_eq!(
            top(&mut s),
            ("lib.c:10".to_string(), 2),
            "{engine:?}: stopped in lib"
        );
        let (at, depth) = step(&mut s, "next");
        assert!(
            at.starts_with("main.c:") && depth == 1,
            "{engine:?}: next from lib ends in main, got {at} at depth {depth}"
        );
    }
}

/// `stepBack` does not end in a library's code either: from `lib`'s second line it goes back past
/// `lib` to the call, where without `justMyCode` it stops on `lib`'s first line.
#[test]
fn step_back_passes_a_library_under_just_my_code() {
    for engine in ENGINES {
        let mut s = stopped_at(engine, false, "lib.c", 11);
        assert_eq!(
            step(&mut s, "stepBack"),
            ("lib.c:10".to_string(), 2),
            "{engine:?}: without justMyCode"
        );
        let mut s = stopped_at(engine, true, "lib.c", 11);
        assert_eq!(
            step(&mut s, "stepBack"),
            ("main.c:5".to_string(), 1),
            "{engine:?}: with it"
        );
    }
}

/// Under `justMyCode`, `stepIn` goes on stepping *in* through a library's code, so a library calling
/// back into the program stops in the callback — where stepping out of the library would run past it.
/// `next` still steps over the whole call.
#[test]
fn step_in_finds_a_callback_the_library_makes() {
    for engine in ENGINES {
        let mut s = session(LIB_CALLBACK_DBG, engine, true, "main.c", 5);
        assert_eq!(
            step(&mut s, "stepIn"),
            ("main.c:20".to_string(), 3),
            "{engine:?}: stops in the callback, below lib"
        );
        let mut s = session(LIB_CALLBACK_DBG, engine, true, "main.c", 5);
        assert_eq!(
            step(&mut s, "next"),
            ("main.c:6".to_string(), 1),
            "{engine:?}: next steps over it all"
        );
    }
}

/// A step that leaves a callback returns into the library that called it (to `lib.c:11`, after the
/// call), and under `justMyCode` goes on out of it to the program's code: `stepOut` and `next` from
/// the callback both end in `main`, after its call into `lib`.
#[test]
fn leaving_a_callback_passes_the_library_that_called_it() {
    for engine in ENGINES {
        for (command, line) in [("stepOut", 20), ("next", 21)] {
            let mut s = session(LIB_CALLBACK_DBG, engine, false, "main.c", line);
            assert_eq!(
                step(&mut s, command),
                ("lib.c:11".to_string(), 2),
                "{engine:?}: {command} from main.c:{line} without justMyCode"
            );
            let mut s = session(LIB_CALLBACK_DBG, engine, true, "main.c", line);
            assert_eq!(
                step(&mut s, command),
                ("main.c:6".to_string(), 1),
                "{engine:?}: {command} from main.c:{line} with it"
            );
        }
    }
}

// Startup code with no source (func 0, like a linker's `_start`) calls a library's initialisation
// (func 1, lib.c:30–31) and then the program's `main` (func 2, main.c:2–3).
const STARTUP_DBG: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = call 2 (v1)
  return v2
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 2
  v2 = i32.mul v0 v1
  return v2
  }
}

debug.file 0 "lib.c"
debug.file 1 "main.c"
debug.lib 0
debug.loc 1 0 0 0 30 1
debug.loc 1 0 1 0 31 1
debug.loc 2 0 0 1 2 1
debug.loc 2 0 1 1 3 1
"#;

/// Under `justMyCode`, `stopOnEntry` stops on the program's own first line: past the sourceless
/// startup code and the library's initialisation, where without it the stop is the first op.
#[test]
fn stop_on_entry_stops_on_the_programs_first_line_under_just_my_code() {
    for engine in ENGINES {
        for just_my_code in [false, true] {
            let mut s = DapServer::new();
            s.handle(&req(1, "initialize", Json::obj(vec![])));
            let mut launch = vec![
                ("programText", Json::s(STARTUP_DBG)),
                ("function", Json::i(0)),
                ("args", Json::Arr(vec![Json::i(5)])),
                ("stopOnEntry", Json::Bool(true)),
                ("justMyCode", Json::Bool(just_my_code)),
            ];
            if let Some(e) = engine {
                launch.push(("engine", Json::s(e)));
            }
            s.handle(&req(2, "launch", Json::obj(launch)));
            let out = s.handle(&req(3, "configurationDone", Json::obj(vec![])));
            let reason = event(&out, "stopped")
                .and_then(|e| e.get("body"))
                .and_then(|b| b.get("reason"))
                .and_then(|r| r.as_str())
                .map(str::to_owned);
            assert_eq!(
                reason.as_deref(),
                Some("entry"),
                "{engine:?} {just_my_code}"
            );
            let frames = frames(&mut s);
            let line = frames[0].get("line").and_then(|l| l.as_i64());
            if just_my_code {
                assert_eq!(
                    top(&mut s),
                    ("main.c:2".to_string(), 2),
                    "{engine:?}: main's first line, below the startup code"
                );
            } else {
                assert_eq!(
                    (line, frames.len()),
                    (Some(0), 1),
                    "{engine:?}: the startup code's first op, which has no line"
                );
            }
        }
    }
}
