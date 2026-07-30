//! The DAP **array-pump** (INTERACTIVE_EMBEDDING.md, the step+reads bundle): `svm_dap_request`
//! accepts a top-level JSON *array* of requests and returns the concatenated reply array — a step
//! plus N state reads in a single FFI crossing. A single object stays the one-request pump with an
//! identical reply shape.
//!
//! One `#[test]` only: the pump's stashes (`DAP_SERVER`/`DAP_OUT`) are main-thread singletons, and
//! Rust runs a binary's tests on threads.

use svm_browser::{svm_dap_request, svm_dap_reset, svm_dap_response_len, svm_dap_response_ptr};
use svm_dap::Json;

// The dap_bytecode.rs LOOP_SUM fixture: a countdown-sum loop with a §6/W4 debug section binding
// sum.c:7 to the loop body and naming the two loop variables.
const LOOP_SUM_DBG: &str = r#"
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

fn req(seq: i64, command: &str, args: Json) -> Json {
    Json::obj(vec![
        ("seq", Json::i(seq)),
        ("type", Json::s("request")),
        ("command", Json::s(command)),
        ("arguments", args),
    ])
}

/// Feed `msg` through the cdylib pump and parse the stashed reply array.
fn pump(msg: &Json) -> Vec<Json> {
    let text = msg.to_string();
    assert_eq!(svm_dap_request(text.as_ptr(), text.len()), 0, "pump ok");
    let raw = unsafe {
        std::slice::from_raw_parts(svm_dap_response_ptr(), svm_dap_response_len()).to_vec()
    };
    let reply = svm_dap::parse(std::str::from_utf8(&raw).expect("utf8")).expect("json");
    match reply {
        Json::Arr(msgs) => msgs,
        other => panic!("reply is not an array: {other:?}"),
    }
}

fn responses(msgs: &[Json]) -> Vec<&Json> {
    msgs.iter()
        .filter(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
        .collect()
}

#[test]
fn array_pump_batches_requests_in_one_crossing() {
    svm_dap_reset();

    // A single object stays the one-request pump: one response.
    let one = pump(&req(1, "initialize", Json::obj(vec![])));
    assert_eq!(responses(&one).len(), 1, "singleton pump: one response");

    // A batch — launch + breakpoint + configurationDone + stackTrace — in ONE FFI crossing: four
    // responses in order, plus the `stopped` event from the breakpoint hit, all in one reply array.
    let batch = Json::Arr(vec![
        req(
            2,
            "launch",
            Json::obj(vec![
                ("programText", Json::s(LOOP_SUM_DBG)),
                ("function", Json::i(0)),
                ("args", Json::Arr(vec![Json::i(3)])),
                ("engine", Json::s("bytecode")),
            ]),
        ),
        req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
                ),
            ]),
        ),
        req(4, "configurationDone", Json::obj(vec![])),
        req(5, "stackTrace", Json::obj(vec![("threadId", Json::i(1))])),
    ]);
    let out = pump(&batch);
    let resp = responses(&out);
    assert_eq!(resp.len(), 4, "four responses from the batched crossing");
    for (r, cmd) in resp.iter().zip([
        "launch",
        "setBreakpoints",
        "configurationDone",
        "stackTrace",
    ]) {
        assert_eq!(
            r.get("command").and_then(|c| c.as_str()),
            Some(cmd),
            "responses arrive in request order"
        );
        assert_eq!(
            r.get("success"),
            Some(&Json::Bool(true)),
            "{cmd} succeeded in the batch"
        );
    }
    assert!(
        out.iter().any(|m| {
            m.get("type").and_then(|t| t.as_str()) == Some("event")
                && m.get("event").and_then(|e| e.as_str()) == Some("stopped")
        }),
        "the breakpoint's stopped event rides the same reply array"
    );
    // The batched stackTrace observed the paused frame — state reads batch with the step/run.
    let frames = resp[3]
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .expect("stackTrace body");
    assert!(!frames.is_empty(), "a paused frame in the batched read");
}
