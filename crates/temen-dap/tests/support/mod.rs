//! Shared scaffolding for the `temen-dap` integration suites (#917): the DAP request builder and the
//! response/event accessors every `tests/*.rs` drives the server through, plus the fixtures used by
//! more than one suite. Each test binary `mod support;`s this and uses only the subset it needs, so
//! `#![allow(dead_code)]` keeps a binary that touches just `req` from warning on the rest. The
//! suites keep their own distinct engine matrices and per-suite helpers/fixtures; only this common
//! surface lives here.
#![allow(dead_code)]

use temen_dap::Json;

/// Build a DAP request message (`seq`/`type`/`command`/`arguments`).
pub fn req(seq: i64, command: &str, args: Json) -> Json {
    Json::obj(vec![
        ("seq", Json::i(seq)),
        ("type", Json::s("request")),
        ("command", Json::s(command)),
        ("arguments", args),
    ])
}

/// The single response message in a `handle()` result (type == "response").
pub fn response(msgs: &[Json]) -> &Json {
    msgs.iter()
        .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .expect("a response")
}

/// The first event with the given name, if any.
pub fn event<'a>(msgs: &'a [Json], name: &str) -> Option<&'a Json> {
    msgs.iter().find(|m| {
        m.get("type").and_then(|t| t.as_str()) == Some("event")
            && m.get("event").and_then(|e| e.as_str()) == Some(name)
    })
}

/// LOOP_SUM with a hand-written §6/W4 debug section: a source location at the loop body (sum.c:7)
/// and the two loop variables mapped to their block-relative SSA value indices.
pub const LOOP_SUM_DBG: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const -1
  v6 = i32.add v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}

debug.file 0 "sum.c"
debug.fname 0 "sum"
debug.loc 0 1 0 0 7 5
debug.var 0 "i" ssa 0 "int"
debug.var 0 "acc" ssa 1 "int"
"#;
