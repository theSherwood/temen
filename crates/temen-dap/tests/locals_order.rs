//! **A frame's Locals pane lists the frame's own variables before the module's globals.**
//!
//! The Locals scope deliberately includes module-scoped globals alongside the frame's locals (a C
//! debugger's Variables pane shows both), and their order used to follow the debug table's. That was
//! fine while a program's own debug info was the only debug info in the module. It stopped being fine
//! with **separate compilation** (#1392): a program linked against a prebuilt library gets the
//! *library's* debug tables first, so a separately compiled libc pushed `__pg_std`/`__pg_brk`/… above
//! the user's own `i`/`acc` in every lesson's Variables pane.
//!
//! The partition is stable, so declaration order is preserved within each group.

use temen_dap::{DapServer, Json};

mod support;
use support::{req, response};

/// Two globals declared *before* the frame's locals in the debug table — the shape a link produces
/// (unit 0's vars come first). `main` takes its arg, adds a global, and keeps two locals live.
const GLOBALS_FIRST: &str = r#"memory 32
data 8 "\x07\x00\x00\x00"
data 12 "\x09\x00\x00\x00"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 8
  v2 = i32.load v1
  v3 = i32.add v0 v2
  return v3
  }
}
debug.file 0 "lib.c"
debug.fname 0 "main"
debug.loc 0 1 0 0 5 5
debug.loc 0 1 0 1 5 5
debug.loc 0 1 0 2 6 5
debug.var global "__lib_first" fixed 8 "int"
debug.var global "__lib_second" fixed 12 "int"
debug.var 0 "acc" ssa 3 "int"
debug.var 0 "arg" ssa 0 "int"
"#;

#[test]
fn a_frames_own_locals_come_before_the_modules_globals() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(GLOBALS_FIRST)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(5)])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch"
    );
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));

    let out = s.handle(&req(
        4,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out);
    let frame_id = frames
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .and_then(|f| f.first())
        .and_then(|f| f.get("id"))
        .and_then(|i| i.as_i64())
        .expect("a stack frame");

    let out = s.handle(&req(
        5,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
    ));
    let scopes = response(&out);
    let vref = scopes
        .get("body")
        .and_then(|b| b.get("scopes"))
        .and_then(|sc| sc.as_array())
        .and_then(|sc| sc.first())
        .and_then(|sc| sc.get("variablesReference"))
        .and_then(|v| v.as_i64())
        .expect("a Locals scope");

    let out = s.handle(&req(
        6,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out);
    let names: Vec<String> = vars
        .get("body")
        .and_then(|b| b.get("variables"))
        .and_then(|v| v.as_array())
        .expect("a variables array")
        .iter()
        .filter_map(|v| v.get("name").and_then(|n| n.as_str()).map(str::to_string))
        .collect();

    // Both groups are present (the pane shows globals too) …
    for want in ["arg", "__lib_first", "__lib_second"] {
        assert!(
            names.iter().any(|n| n == want),
            "missing {want} in {names:?}"
        );
    }
    // … but every one of the frame's own locals precedes every global.
    let last_local = names
        .iter()
        .rposition(|n| !n.starts_with("__lib_"))
        .expect("a local");
    let first_global = names
        .iter()
        .position(|n| n.starts_with("__lib_"))
        .expect("a global");
    assert!(
        last_local < first_global,
        "the frame's locals must come first; got {names:?}"
    );
    // Declaration order survives within each group (a stable partition, not a sort by name).
    let globals: Vec<&String> = names.iter().filter(|n| n.starts_with("__lib_")).collect();
    assert_eq!(
        globals,
        vec!["__lib_first", "__lib_second"],
        "globals keep their declared order"
    );
}
