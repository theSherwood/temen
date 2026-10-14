//! Structured value shapes (#2341): every DAP variable whose type is in §6's structured type table
//! carries a `typeInfo` — its `kind`, `size`, a scalar's `encoding`, an array's `count`, and the
//! pointee's or element's summary as `target` — so a client renders it from data instead of parsing
//! the type's render name (a C spelling like `char *`). Both engines.

use temen_dap::{DapServer, Json};

mod support;
use support::{req, response};

// One frame with an `int`, a `char *`, a `char[4]` and a two-int struct, all in the window, stopped
// at t.c:3.
const SHAPES_DBG: &str = r#"
memory 17
func (i64) -> (i32) {
block 0 (v0: i64) {
  v1 = i32.const 7
  i32.store v0 v1
  v2 = i32.const 0
  return v2
  }
}

debug.file 0 "t.c"
debug.loc 0 0 2 0 3 1
debug.type 0 base "int" signed 4
debug.type 1 base "char" signed 1
debug.type 2 ptr "char *" 1 8
debug.type 3 array "char[4]" 1 4
debug.type 4 agg "struct pt" 8
debug.field 4 "x" 0 0
debug.field 4 "y" 4 0
debug.var 0 "n" win 0 "int" 0
debug.var 0 "s" win 8 "char *" 2
debug.var 0 "buf" win 16 "char[4]" 3
debug.var 0 "pt" win 24 "struct pt" 4
"#;

/// The session stopped at t.c:3 on `engine`, its frame's id, and the frame's Locals.
fn locals(engine: Option<&str>) -> (DapServer, i64, Vec<Json>) {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut launch = vec![
        ("programText", Json::s(SHAPES_DBG)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![Json::i(0)])),
    ];
    if let Some(e) = engine {
        launch.push(("engine", Json::s(e)));
    }
    s.handle(&req(2, "launch", Json::obj(launch)));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("t.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(3))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frame = response(&out)
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .expect("frames")[0]
        .get("id")
        .and_then(|i| i.as_i64())
        .expect("a frame id");
    let out = s.handle(&req(
        6,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame))]),
    ));
    let vref = response(&out)
        .get("body")
        .and_then(|b| b.get("scopes"))
        .and_then(|sc| sc.as_array())
        .expect("scopes")[0]
        .get("variablesReference")
        .and_then(|v| v.as_i64())
        .expect("a Locals reference");
    let vars = variables(&mut s, vref);
    (s, frame, vars)
}

fn variables(s: &mut DapServer, vref: i64) -> Vec<Json> {
    let out = s.handle(&req(
        7,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    response(&out)
        .get("body")
        .and_then(|b| b.get("variables"))
        .and_then(|v| v.as_array())
        .expect("variables")
        .to_vec()
}

fn named<'a>(vars: &'a [Json], name: &str) -> &'a Json {
    vars.iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some(name))
        .unwrap_or_else(|| panic!("no variable {name}"))
}

/// A `typeInfo`'s `(kind, size, encoding, count)`, the fields a client renders by.
fn shape(info: &Json) -> (String, i64, Option<String>, Option<i64>) {
    (
        info.get("kind")
            .and_then(|k| k.as_str())
            .unwrap()
            .to_string(),
        info.get("size").and_then(|k| k.as_i64()).unwrap(),
        info.get("encoding")
            .and_then(|e| e.as_str())
            .map(str::to_string),
        info.get("count").and_then(|c| c.as_i64()),
    )
}

#[test]
fn every_structured_variable_says_its_shape() {
    for engine in [None, Some("bytecode")] {
        let (mut s, frame, vars) = locals(engine);
        let info = |name: &str| named(&vars, name).get("typeInfo").cloned().expect(name);
        let scalar = |size, enc: &str| ("scalar".to_string(), size, Some(enc.to_string()), None);

        assert_eq!(shape(&info("n")), scalar(4, "signed"), "{engine:?}: int");

        // A pointer, and what it points at: a one-byte signed scalar — a client can preview the
        // string behind a `char *` without matching the spelling.
        let s_info = info("s");
        assert_eq!(
            shape(&s_info),
            ("pointer".to_string(), 8, None, None),
            "{engine:?}: char *"
        );
        assert_eq!(
            shape(s_info.get("target").expect("pointee")),
            scalar(1, "signed"),
            "{engine:?}: char * points at a char"
        );

        let buf = info("buf");
        assert_eq!(
            shape(&buf),
            ("array".to_string(), 4, None, Some(4)),
            "{engine:?}: char[4]"
        );
        assert_eq!(
            shape(buf.get("target").expect("element")),
            scalar(1, "signed"),
            "{engine:?}: of char"
        );

        let pt = info("pt");
        assert_eq!(
            shape(&pt),
            ("aggregate".to_string(), 8, None, None),
            "{engine:?}: struct"
        );
        assert!(
            pt.get("target").is_none(),
            "{engine:?}: an aggregate has no target"
        );

        // Its members say theirs too.
        let vref = named(&vars, "pt")
            .get("variablesReference")
            .and_then(|v| v.as_i64())
            .expect("pt expands");
        let members = variables(&mut s, vref);
        let x = named(&members, "x").get("typeInfo").cloned().expect("x");
        assert_eq!(shape(&x), scalar(4, "signed"), "{engine:?}: pt.x");

        // `evaluate` says it as well, for a bare variable and for a member path.
        for expr in ["n", "pt.x"] {
            let out = s.handle(&req(
                8,
                "evaluate",
                Json::obj(vec![
                    ("expression", Json::s(expr)),
                    ("frameId", Json::i(frame)),
                ]),
            ));
            let info = response(&out)
                .get("body")
                .and_then(|b| b.get("typeInfo"))
                .cloned()
                .expect(expr);
            assert_eq!(
                shape(&info),
                scalar(4, "signed"),
                "{engine:?}: evaluate {expr}"
            );
        }
    }
}
