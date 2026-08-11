//! The custom `globals` request: enumerate module-scoped scalar globals (name, addr, byte size,
//! current value) for a consumer's shared-state view. DAP `scopes` only exposes per-frame Locals,
//! so this is the only way to list globals by name. Acceptance: a named global with a data-segment
//! initializer is reported with its value read from the window; aggregates are omitted; reserved
//! C names are *not* filtered here (that is the C-frontend consumer's presentation choice).

use svm_dap::{DapServer, Json};

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

/// A global `int counter` at window address 8, initialized to 42 (little-endian `2a 00 00 00`) by a
/// data segment, plus a reserved-name global `__hidden` at address 12 initialized to 7. The entry
/// just returns `counter`.
const GLOBALS_DBG: &str = r#"memory 32
data 8 "*\x00\x00\x00"
data 12 "\x07\x00\x00\x00"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 8
  v2 = i32.load v1
  return v2
  }
}
debug.file 0 "g.c"
debug.fname 0 "main"
debug.loc 0 1 0 0 5 5
debug.var global "counter" fixed 8 "int"
debug.var global "__hidden" fixed 12 "int"
"#;

/// **The `globals` request enumerates named scalars with their current values.** After launch the
/// data segments are applied, so `counter` reads back `42` (typed through `int`) and the reserved
/// `__hidden` reads `7`. Both are returned — the request is faithful; name filtering lives in the
/// consumer, not here.
#[test]
fn globals_enumerates_named_scalars_with_values() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(GLOBALS_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(0)])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));

    let out = s.handle(&req(4, "globals", Json::obj(vec![])));
    let resp = response(&out);
    assert_eq!(resp.get("success"), Some(&Json::Bool(true)), "globals ok");
    let list = resp
        .get("body")
        .and_then(|b| b.get("globals"))
        .and_then(|g| g.as_array())
        .expect("a globals array");

    let find = |name: &str| {
        list.iter()
            .find(|g| g.get("name").and_then(|n| n.as_str()) == Some(name))
            .unwrap_or_else(|| panic!("global {name} present"))
    };
    let counter = find("counter");
    assert_eq!(counter.get("addr"), Some(&Json::i(8)));
    assert_eq!(counter.get("size"), Some(&Json::i(4)));
    assert_eq!(
        counter.get("value").and_then(|v| v.as_str()),
        Some("42"),
        "value read from the window and formatted as int"
    );
    // Reserved names are returned by the engine — the consumer decides whether to hide them.
    assert_eq!(
        find("__hidden").get("value").and_then(|v| v.as_str()),
        Some("7")
    );
}

/// **Fails cleanly with no live session** — like the other custom requests, `globals` returns an
/// unsuccessful response rather than panicking when nothing has been launched.
#[test]
fn globals_without_session_fails_cleanly() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(2, "globals", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}
