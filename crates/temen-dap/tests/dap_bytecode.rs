//! DAP-over-bytecode parity (DEBUGGING.md G3). The same scripted DAP conversation is replayed against
//! both backends — the default tree-walker `Inspector` and the `"engine":"bytecode"` `BytecodeBackend`
//! (the engine the browser playground runs) — and the debugger observations must be **identical**:
//! breakpoint binding, the stop reason, the source frame, and the inspected locals at **every** loop
//! iteration (resume included), through to termination. This is the server-side proof that the backend
//! seam dispatches the forward-debug subset to either engine with the same result — the runtime
//! counterpart of `crates/temen/tests/debug_parity.rs`'s engine-level checks.

use temen_dap::{DapServer, Json};

mod support;
use support::{event, req, response, LOOP_SUM_DBG};

/// What the debugger reported over one full session — the fields that must match across engines.
#[derive(Debug, PartialEq)]
struct Observations {
    bp_verified: bool,
    bp_line: i64,
    first_reason: String,
    top_frame_name: String,
    top_frame_line: i64,
    /// `(i, acc)` read at each successive breakpoint hit, innermost frame — proves resume/loop parity.
    iter_locals: Vec<(String, String)>,
    terminated: bool,
}

/// Read the innermost frame's `(i, acc)` at the current stop via stackTrace → scopes → variables.
fn read_locals(s: &mut DapServer, seq: i64) -> (String, String) {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap();
    let top = &frames.as_array().unwrap()[0];
    let frame_id = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
    ));
    let scope = &response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let vref = scope.get("variablesReference").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let map: std::collections::HashMap<&str, &str> = vars
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap(),
                v.get("value").unwrap().as_str().unwrap(),
            )
        })
        .collect();
    (
        map.get("i").copied().unwrap_or("?").to_string(),
        map.get("acc").copied().unwrap_or("?").to_string(),
    )
}

/// Run the whole scripted session on `engine` (`None` ⇒ the default tree-walker) and record what the
/// debugger reported.
fn run_session(engine: Option<&str>) -> Observations {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));

    let mut launch_args = vec![
        ("programText", Json::s(LOOP_SUM_DBG)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![Json::i(3)])),
    ];
    if let Some(e) = engine {
        launch_args.push(("engine", Json::s(e)));
    }
    let out = s.handle(&req(2, "launch", Json::obj(launch_args)));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok ({engine:?})"
    );

    // Breakpoint on the loop body (sum.c:7).
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/work/sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    let bp0 = &response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let bp_verified = bp0.get("verified") == Some(&Json::Bool(true));
    let bp_line = bp0.get("line").and_then(|l| l.as_i64()).unwrap_or(-1);

    // Run to the first hit.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stopped = event(&out, "stopped").expect("stops at the breakpoint");
    let first_reason = stopped
        .get("body")
        .unwrap()
        .get("reason")
        .unwrap()
        .as_str()
        .unwrap()
        .to_string();

    // First frame + locals.
    let out = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let top = &response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let top_frame_name = top.get("name").unwrap().as_str().unwrap().to_string();
    let top_frame_line = top.get("line").unwrap().as_i64().unwrap();

    let mut iter_locals = vec![read_locals(&mut s, 6)];

    // Continue through every remaining hit, recording locals each time, until the guest terminates.
    let mut terminated = false;
    for seq in 10..200 {
        let out = s.handle(&req(seq, "continue", Json::obj(vec![])));
        if event(&out, "terminated").is_some() {
            terminated = true;
            break;
        }
        if event(&out, "stopped").is_some() {
            iter_locals.push(read_locals(&mut s, seq));
        }
    }

    Observations {
        bp_verified,
        bp_line,
        first_reason,
        top_frame_name,
        top_frame_line,
        iter_locals,
        terminated,
    }
}

#[test]
fn dap_over_bytecode_matches_the_tree_walker() {
    let treewalk = run_session(None);
    let bytecode = run_session(Some("bytecode"));

    // Absolute sanity: the bytecode backend binds the line, stops for a breakpoint, shows `#0 sum` at
    // line 7, reads i=3/acc=0 at the first hit, and terminates.
    assert!(bytecode.bp_verified, "bytecode: breakpoint binds");
    assert_eq!(bytecode.bp_line, 7);
    assert_eq!(bytecode.first_reason, "breakpoint");
    assert_eq!(bytecode.top_frame_name, "#0 sum");
    assert_eq!(bytecode.top_frame_line, 7);
    assert_eq!(
        bytecode.iter_locals.first(),
        Some(&("3".to_string(), "0".to_string()))
    );
    assert!(bytecode.terminated, "bytecode: the guest terminates");
    assert!(
        bytecode.iter_locals.len() >= 3,
        "loops at least 3 iterations"
    );

    // The load-bearing claim: the two engines report *identical* debugger observations — breakpoint,
    // stop reason, frame, and the (i, acc) at every iteration.
    assert_eq!(
        bytecode, treewalk,
        "DAP-over-bytecode ≡ DAP-over-tree-walker"
    );
}

// A program with NO explicit debug section — the server synthesizes one (line table + SSA names) so
// hand-written Temen text is debuggable. Same sum loop, sans the `debug.*` directives.
const PLAIN_SUM: &str = r#"
func () -> (i64) {
block 0 () {
  vn = i64.const 5
  vacc0 = i64.const 0
  br 1(vn, vacc0)
}
block 1 (vi: i64, vacc: i64) {
  vsum = i64.add vacc vi
  vone = i64.const 1
  vnext = i64.sub vi vone
  br_if vnext 1(vnext, vsum) 2(vsum)
}
block 2 (vr: i64) {
  return vr
  }
}
"#;

#[test]
fn dap_over_bytecode_debugs_plain_temen_via_synthesized_debug_info() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(PLAIN_SUM)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );

    // A breakpoint on the loop body (line 9, `vsum = i64.add`) binds against the synthesized table.
    let out = s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(9))])]),
            ),
        ]),
    ));
    let bp0 = &response(&out)
        .get("body")
        .unwrap()
        .get("breakpoints")
        .unwrap()
        .as_array()
        .unwrap()[0];
    assert_eq!(
        bp0.get("verified"),
        Some(&Json::Bool(true)),
        "breakpoint binds on plain Temen"
    );
    assert_eq!(bp0.get("line"), Some(&Json::i(9)));

    // Run to it; the loop variables read back by their **text names** (i / acc), not v-indices.
    let out = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    assert!(
        event(&out, "stopped").is_some(),
        "stops at the synthesized breakpoint"
    );
    let (i, acc) = read_named(&mut s, 5, "vi", "vacc");
    assert_eq!(
        (i.as_str(), acc.as_str()),
        ("5", "0"),
        "vi=5, vacc=0 at the first hit"
    );
}

/// Read two named locals at the current stop (innermost frame).
fn read_named(s: &mut DapServer, seq: i64, a: &str, b: &str) -> (String, String) {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let top = &response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0];
    let fid = top.get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "scopes",
        Json::obj(vec![("frameId", Json::i(fid))]),
    ));
    let vref = response(&out)
        .get("body")
        .unwrap()
        .get("scopes")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("variablesReference")
        .unwrap()
        .as_i64()
        .unwrap();
    let out = s.handle(&req(
        seq,
        "variables",
        Json::obj(vec![("variablesReference", Json::i(vref))]),
    ));
    let vars = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap();
    let map: std::collections::HashMap<&str, &str> = vars
        .as_array()
        .unwrap()
        .iter()
        .map(|v| {
            (
                v.get("name").unwrap().as_str().unwrap(),
                v.get("value").unwrap().as_str().unwrap(),
            )
        })
        .collect();
    (
        map.get(a).copied().unwrap_or("?").to_string(),
        map.get(b).copied().unwrap_or("?").to_string(),
    )
}

#[test]
fn dap_over_bytecode_reverse_matches_the_tree_walker() {
    // Run to the loop breakpoint three times (i = 3, 2, 1), then reverseContinue back to the previous
    // hit — on both engines. The bytecode backend does this by deterministic replay; the observations
    // must match the tree-walker's time-travel.
    fn script(engine: Option<&str>) -> (String, String, String) {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut la = vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ];
        if let Some(e) = engine {
            la.push(("engine", Json::s(e)));
        }
        s.handle(&req(2, "launch", Json::obj(la)));
        s.handle(&req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("/work/sum.c"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
                ),
            ]),
        ));
        s.handle(&req(4, "configurationDone", Json::obj(vec![]))); // hit 1: i=3
        s.handle(&req(5, "continue", Json::obj(vec![]))); // hit 2: i=2
        s.handle(&req(6, "continue", Json::obj(vec![]))); // hit 3: i=1
        let at3 = read_locals(&mut s, 7); // (i=1, acc=5)
                                          // reverseContinue → back to the previous breakpoint hit (i=2).
        let rev = s.handle(&req(8, "reverseContinue", Json::obj(vec![])));
        let reason = event(&rev, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let back = read_locals(&mut s, 9); // (i=2, acc=3)
        (
            format!("{},{}", at3.0, at3.1),
            reason,
            format!("{},{}", back.0, back.1),
        )
    }

    let bytecode = script(Some("bytecode"));
    assert_eq!(bytecode.0, "1,5", "3rd hit is i=1, acc=5");
    assert_eq!(
        bytecode.1, "breakpoint",
        "reverseContinue lands on a breakpoint"
    );
    assert_eq!(bytecode.2, "2,3", "…the previous hit: i=2, acc=3");
    assert_eq!(
        bytecode,
        script(None),
        "bytecode reverse ≡ tree-walker reverse"
    );
}

// A program that stores 8 bytes to window address 16384 (above the #1094 NULL guard; line 6), for
// the watchpoint test. Auto debug
// info makes line 5 (before the store) breakpointable.
const MEM_STORE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  a = i64.const 16384
  v = i64.const 42
  i64.store a v
  r = i64.load a
  return r
  }
}
"#;

#[test]
fn dap_over_bytecode_watchpoint_matches_the_tree_walker() {
    // Arm a write data breakpoint on window [16384, 16392) and Continue: the debugger stops *before* the store
    // that touches it (line 6) with reason "data breakpoint" — identically on both engines.
    fn script(engine: Option<&str>) -> (bool, String, i64) {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut la = vec![
            ("programText", Json::s(MEM_STORE)),
            ("function", Json::i(0)),
        ];
        if let Some(e) = engine {
            la.push(("engine", Json::s(e)));
        }
        s.handle(&req(2, "launch", Json::obj(la)));
        // Break on line 5 (the const just before the store), so we're stopped when we arm the watch.
        s.handle(&req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(6))])]),
                ),
            ]),
        ));
        s.handle(&req(4, "configurationDone", Json::obj(vec![])));
        // Arm a write watchpoint on window [16384, 16392) directly (dataId = "addr:len").
        let arm = s.handle(&req(
            5,
            "setDataBreakpoints",
            Json::obj(vec![(
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![
                    ("dataId", Json::s("16384:8")),
                    ("accessType", Json::s("write")),
                ])]),
            )]),
        ));
        let verified = arm
            .iter()
            .find(|m| m.get("command").and_then(|c| c.as_str()) == Some("setDataBreakpoints"))
            .and_then(|m| {
                m.get("body")?
                    .get("breakpoints")?
                    .as_array()?
                    .first()?
                    .get("verified")
                    .cloned()
            })
            == Some(Json::Bool(true));
        // Continue → stop before the store's write to [16384, 16392).
        let cont = s.handle(&req(6, "continue", Json::obj(vec![])));
        let reason = event(&cont, "stopped")
            .and_then(|e| e.get("body")?.get("reason")?.as_str().map(str::to_owned))
            .unwrap_or_default();
        let out = s.handle(&req(
            7,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let line = response(&out)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("line")
            .unwrap()
            .as_i64()
            .unwrap();
        (verified, reason, line)
    }

    let bytecode = script(Some("bytecode"));
    assert!(bytecode.0, "the data breakpoint verifies");
    assert_eq!(bytecode.1, "data breakpoint", "stops for the watchpoint");
    assert_eq!(
        bytecode.2, 7,
        "stops at the store's line (before it writes)"
    );
    assert_eq!(
        bytecode,
        script(None),
        "bytecode watchpoint ≡ tree-walker watchpoint"
    );
}

// The playground's watchpoint demo (browser/web/play.js `EXAMPLES['Debugger (watchpoints …)']`): a
// counter at a *fixed* window address, bumped each loop iteration, with a named source variable
// `count` over it (`debug.var … fixed 0`) so the panel can arm a data breakpoint by name — exactly
// what `dataBreakpointInfo` + `setDataBreakpoints` resolve. A breakpoint is pre-placed on the loop
// body (line 12) so a session pauses there to arm the watch. Kept in sync with the playground source.
const WATCH_COUNTER_DBG: &str = r#"; A counter lives at a fixed window address. Set a watch on `count` in the
; Variables pane (click its ● toggle), then Continue: the debugger stops the
; instant a store changes it — stop reason "data breakpoint".
memory 16
func () -> (i64) {
block 0 () {
  a0 = i64.const 16384
  z = i64.const 0
  i64.store a0 z
  br 1(z)
}
block 1 (i: i64) {
  a1 = i64.const 16384
  one = i64.const 1
  n = i64.add i one
  i64.store a1 n
  limit = i64.const 3
  done = i64.ge_s n limit
  br_if done 2(n) 1(n)
}
block 2 (r: i64) {
  a2 = i64.const 16384
  out = i64.load a2
  return out
  }
}

debug.file 0 "counter.temt"
debug.fname 0 "count_up"
debug.loc 0 0 0 0 7 3
debug.loc 0 0 1 0 8 3
debug.loc 0 0 2 0 9 3
debug.loc 0 1 0 0 12 3
debug.loc 0 1 1 0 13 3
debug.loc 0 1 2 0 14 3
debug.loc 0 1 3 0 15 3
debug.loc 0 1 4 0 16 3
debug.loc 0 1 5 0 17 3
debug.loc 0 2 0 0 20 3
debug.loc 0 2 1 0 21 3
debug.type 0 base "long" signed 8
debug.var 0 "count" fixed 16384 "long" 0
"#;

#[test]
fn dap_over_bytecode_named_watchpoint_matches_the_tree_walker() {
    // Arm a data breakpoint *by name* (the playground path: `dataBreakpointInfo` on the in-scope
    // variable `count`, then `setDataBreakpoints` on the minted `dataId`), then Continue: the debugger
    // stops *before* the loop-body store that touches `count`, reason "data breakpoint", on the same
    // source line — identically on both engines. This is the runtime, server-level proof that the
    // browser panel's watch-a-variable flow resolves and trips on the bytecode backend.
    fn script(engine: Option<&str>) -> (bool, String, i64, String) {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut la = vec![
            ("programText", Json::s(WATCH_COUNTER_DBG)),
            ("function", Json::i(0)),
        ];
        if let Some(e) = engine {
            la.push(("engine", Json::s(e)));
        }
        s.handle(&req(2, "launch", Json::obj(la)));
        // Break on line 12 (the loop body) so we pause with `count` in scope to arm the watch.
        s.handle(&req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("counter.temt"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(12))])]),
                ),
            ]),
        ));
        s.handle(&req(4, "configurationDone", Json::obj(vec![])));
        // Resolve the paused frame's Locals scope, then mint a `dataId` for `count` through it — the
        // exact request sequence `browser/web/dap.js` issues from the Variables pane.
        let st = s.handle(&req(
            5,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let frame_id = response(&st)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("id")
            .unwrap()
            .as_i64()
            .unwrap();
        let sc = s.handle(&req(
            6,
            "scopes",
            Json::obj(vec![("frameId", Json::i(frame_id))]),
        ));
        let var_ref = response(&sc)
            .get("body")
            .unwrap()
            .get("scopes")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("variablesReference")
            .unwrap()
            .as_i64()
            .unwrap();
        let info = s.handle(&req(
            7,
            "dataBreakpointInfo",
            Json::obj(vec![
                ("variablesReference", Json::i(var_ref)),
                ("name", Json::s("count")),
            ]),
        ));
        let data_id = response(&info)
            .get("body")
            .and_then(|b| b.get("dataId"))
            .and_then(|d| d.as_str())
            .map(str::to_owned);
        let arm = s.handle(&req(
            8,
            "setDataBreakpoints",
            Json::obj(vec![(
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![
                    ("dataId", data_id.clone().map(Json::s).unwrap_or(Json::Null)),
                    ("accessType", Json::s("write")),
                ])]),
            )]),
        ));
        let verified = response(&arm)
            .get("body")
            .and_then(|b| b.get("breakpoints"))
            .and_then(|b| b.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("verified"))
            .cloned()
            == Some(Json::Bool(true));
        // Continue → stop before the loop-body store's write to `count`.
        let cont = s.handle(&req(9, "continue", Json::obj(vec![])));
        let reason = event(&cont, "stopped")
            .and_then(|e| e.get("body")?.get("reason")?.as_str().map(str::to_owned))
            .unwrap_or_default();
        let out = s.handle(&req(
            10,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let line = response(&out)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("line")
            .unwrap()
            .as_i64()
            .unwrap();
        (verified, reason, line, data_id.unwrap_or_default())
    }

    let bytecode = script(Some("bytecode"));
    assert!(
        bytecode.0,
        "the named data breakpoint verifies on the bytecode engine"
    );
    assert_eq!(bytecode.1, "data breakpoint", "stops for the watchpoint");
    assert_eq!(
        bytecode.3, "16384:8",
        "`count` resolves to a fixed window range [16384, 16392)"
    );
    assert_eq!(
        bytecode.2, 15,
        "stops at the loop-body store's line (before it writes)"
    );
    assert_eq!(
        bytecode,
        script(None),
        "bytecode named watchpoint ≡ tree-walker named watchpoint"
    );
}

#[test]
fn dap_over_bytecode_step_back_rewinds_one_op() {
    // stepBack re-executes to one op earlier: after two forward steps the op clock advances, and a
    // stepBack lands strictly before the current position (still inside the guest, i.e. a `step` stop).
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("/work/sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    s.handle(&req(4, "configurationDone", Json::obj(vec![]))); // stop at the loop body, i=3
    let back = s.handle(&req(5, "stepBack", Json::obj(vec![])));
    assert_eq!(
        response(&back).get("success"),
        Some(&Json::Bool(true)),
        "stepBack succeeds"
    );
    // Rewinds to one op earlier — still inside the guest, so a `step` stop (not `terminated`).
    assert_eq!(
        event(&back, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason"),
        Some(&Json::s("step")),
        "lands as a step stop inside the guest",
    );
    // The frame is still readable after the rewind (we didn't fall off the start of the program).
    let out = s.handle(&req(
        6,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap();
    assert!(!frames.is_empty(), "a live frame after stepBack");
}

#[test]
fn dap_over_bytecode_step_back_after_forward_progress_matches_the_tree_walker() {
    // Exercises the `step_back` stoppable-position cache across a forward advance that moves the
    // position *past* the cache's high-water: a first stepBack builds the trace at i=3, two forward
    // `continue`s advance well beyond it (i=1), then a second stepBack must rebuild the trace to the
    // new high-water and still land correctly. The observations must match the tree-walker's
    // time-travel at every landing — the cache is a pure optimization, not a behavior change.
    fn script(engine: Option<&str>) -> (String, String, String) {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut la = vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ];
        if let Some(e) = engine {
            la.push(("engine", Json::s(e)));
        }
        s.handle(&req(2, "launch", Json::obj(la)));
        s.handle(&req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("/work/sum.c"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
                ),
            ]),
        ));
        s.handle(&req(4, "configurationDone", Json::obj(vec![]))); // hit 1: i=3
        s.handle(&req(5, "stepBack", Json::obj(vec![]))); // builds the trace (high-water at i=3)
                                                          // Forward again, well past that high-water: the first `continue` re-hits i=3 (we rewound to
                                                          // before it), then i=2, then i=1 — the furthest-forward position of the session.
        s.handle(&req(6, "continue", Json::obj(vec![]))); // hit: i=3 (re-hit)
        s.handle(&req(7, "continue", Json::obj(vec![]))); // hit: i=2
        s.handle(&req(8, "continue", Json::obj(vec![]))); // hit: i=1
        let at1 = read_locals(&mut s, 9); // (i=1, acc=5)
        let back = s.handle(&req(10, "stepBack", Json::obj(vec![]))); // now > high-water ⇒ rebuild, then seek
        let reason = event(&back, "stopped")
            .unwrap()
            .get("body")
            .unwrap()
            .get("reason")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();
        let after_back = read_locals(&mut s, 11); // one op before the i=1 stop
        (
            format!("{},{}", at1.0, at1.1),
            reason,
            format!("{},{}", after_back.0, after_back.1),
        )
    }

    let bytecode = script(Some("bytecode"));
    assert_eq!(bytecode.0, "1,5", "furthest-forward hit is i=1, acc=5");
    assert_eq!(
        bytecode.1, "step",
        "the rebuild-then-seek stepBack is a step stop"
    );
    assert_eq!(
        bytecode,
        script(None),
        "bytecode step_back across a cache rebuild ≡ tree-walker time-travel",
    );
}

// A `thread.spawn` guest (two workers each load/add/store mem[16384], one guard up; root spawns + joins). Auto debug
// info makes the worker's `vc = i64.load vaddr` breakpointable — it's on line 18 (leading newline is
// line 1). Drives the multithreaded scheduled bytecode engine over DAP.
const RACY_COUNTER: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 1
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 16384
  vr = i64.load vaddr
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vc = i64.load vaddr
  vn = i64.add vc varg
  i64.store vaddr vn
  vz = i64.const 0
  return vz
  }
}
"#;

#[test]
fn dap_over_bytecode_multithreaded_breakpoint_per_thread() {
    // A breakpoint on the worker's load (line 18) fires once per spawned thread on the multithreaded
    // bytecode engine, reporting a *distinct* DAP thread each time, with every live vCPU listed and
    // the stopped thread's own stack readable — then the guest terminates.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&launch).get("success"),
        Some(&Json::Bool(true)),
        "multithreaded launch on the bytecode engine"
    );
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(18))])]),
            ),
        ]),
    ));

    // Run to the first worker's breakpoint.
    let cfg = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stop1 = event(&cfg, "stopped").expect("stops at the worker breakpoint");
    assert_eq!(
        stop1.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("breakpoint")
    );
    let tid1 = stop1
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert!(tid1 >= 2, "a spawned worker (thread id ≥ 2), not the root");

    // `threads` lists every live vCPU: the root (join-blocked) plus the two workers.
    let th = s.handle(&req(5, "threads", Json::obj(vec![])));
    let ids: Vec<i64> = response(&th)
        .get("body")
        .unwrap()
        .get("threads")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t.get("id").unwrap().as_i64().unwrap())
        .collect();
    assert!(ids.contains(&1), "root thread listed: {ids:?}");
    assert!(ids.len() >= 3, "root + two workers live: {ids:?}");

    // The stopped thread's own stack: its top frame is at the worker load line (18).
    let st = s.handle(&req(
        6,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(tid1))]),
    ));
    let line = response(&st)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("line")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_eq!(line, 18, "stopped at the worker's load line");

    // Continue → the *other* worker hits the same breakpoint (a distinct DAP thread).
    let cont = s.handle(&req(7, "continue", Json::obj(vec![])));
    let stop2 = event(&cont, "stopped").expect("the second worker stops");
    let tid2 = stop2
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_ne!(tid1, tid2, "the two workers are distinct DAP threads");

    // Continue → the guest terminates (no more breakpoint hits).
    let done = s.handle(&req(8, "continue", Json::obj(vec![])));
    assert!(
        event(&done, "terminated").is_some(),
        "the multithreaded guest finished"
    );
}

/// The stack frames of DAP thread `tid` at the current stop; empty when the thread has no
/// resolvable frame.
fn frames(s: &mut DapServer, seq: i64, tid: i64) -> Vec<Json> {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(tid))]),
    ));
    response(&out)
        .get("body")
        .and_then(|b| b.get("stackFrames"))
        .and_then(|f| f.as_array())
        .map(|a| a.to_vec())
        .unwrap_or_default()
}

#[test]
fn dap_over_bytecode_threaded_step_keeps_a_resolvable_frame() {
    // Source-line **stepping** on the multithreaded bytecode engine keeps a resolvable frame — the
    // c_interpret threading-tier seam. Before the entry-`locate()` fix (commit that closed the
    // threaded-stepping ask) a fresh scheduled run launched un-positioned, so the first `stepIn` ran
    // to completion and a following `stackTrace` had no frame ("a bare stackTrace after a threaded
    // step returns no frames"). This pins the fact the browser threads panel relies on: after each
    // step the stopped thread resolves a source frame, and a spawned worker resolves its own frame
    // *while the session is mid-step* (so `select_task` → `stackTrace` lights the panel up during
    // stepping, not only after a run-to-completion).
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&launch).get("success"),
        Some(&Json::Bool(true)),
        "multithreaded launch on the bytecode engine"
    );

    // The entry itself is a resolvable frame (the single-vCPU engine gets this by construction; the
    // scheduled one via the entry `locate()`), not an empty backtrace.
    assert!(
        !frames(&mut s, 3, 1).is_empty(),
        "the entry has a resolvable frame"
    );

    let mut saw_worker_frame = false;
    for i in 0..24i64 {
        let out = s.handle(&req(
            100 + i,
            "stepIn",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        // The first step must advance one source line, never run the whole program to termination.
        if event(&out, "terminated").is_some() {
            assert!(i > 0, "the first step must not run the whole program");
            break;
        }
        // The stopped thread named by the step's `stopped` event resolves a source frame — the
        // "resolvable frame after a threaded step" the panel reads (regression: an empty backtrace).
        let tid = event(&out, "stopped")
            .and_then(|e| e.get("body"))
            .and_then(|b| b.get("threadId"))
            .and_then(|v| v.as_i64())
            .expect("a threaded step reports a stopped thread");
        let top = frames(&mut s, 200 + i, tid);
        assert!(
            !top.is_empty() && top[0].get("line").and_then(|l| l.as_i64()).is_some(),
            "the stopped thread resolves a source frame after step {i}"
        );

        // Once a `thread.spawn` has run, the spawned worker (DAP thread id ≥ 2) resolves its own
        // frame *while stepping* — the fact the threads panel needs mid-run, distinct from the
        // stopped (root) thread above.
        let th = s.handle(&req(300 + i, "threads", Json::obj(vec![])));
        let worker = response(&th)
            .get("body")
            .and_then(|b| b.get("threads"))
            .and_then(|t| t.as_array())
            .into_iter()
            .flatten()
            .filter_map(|t| t.get("id").and_then(|v| v.as_i64()))
            .find(|&id| id >= 2);
        if let Some(w) = worker {
            let wframes = frames(&mut s, 400 + i, w);
            assert!(
                !wframes.is_empty() && wframes[0].get("line").and_then(|l| l.as_i64()).is_some(),
                "the spawned worker (thread {w}) resolves a source frame while stepping"
            );
            saw_worker_frame = true;
            break;
        }
    }
    assert!(
        saw_worker_frame,
        "a spawned worker surfaced a resolvable frame while stepping"
    );
}

#[test]
fn dap_over_bytecode_multithreaded_reverse_continue() {
    // Reverse debugging on the scheduled engine (deterministic replay to a global `turn`): run forward
    // through both workers' breakpoints, then `reverseContinue` back to the *previous* one — landing on
    // the earlier worker's breakpoint, an earlier turn.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(18))])]),
            ),
        ]),
    ));
    // Forward: the first worker, then the second (a distinct DAP thread).
    let cfg = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let tid_a = event(&cfg, "stopped")
        .and_then(|e| e.get("body")?.get("threadId")?.as_i64())
        .unwrap();
    let cont = s.handle(&req(5, "continue", Json::obj(vec![])));
    let tid_b = event(&cont, "stopped")
        .and_then(|e| e.get("body")?.get("threadId")?.as_i64())
        .unwrap();
    assert_ne!(tid_a, tid_b, "forward reached the second worker");

    // reverseContinue → back to the previous breakpoint (the first worker).
    let rev = s.handle(&req(6, "reverseContinue", Json::obj(vec![])));
    assert_eq!(
        response(&rev).get("success"),
        Some(&Json::Bool(true)),
        "reverseContinue is supported on the scheduled engine"
    );
    let stop = event(&rev, "stopped").expect("reverse lands on a stop");
    assert_eq!(
        stop.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("breakpoint")
    );
    assert_eq!(
        stop.get("body").unwrap().get("threadId").unwrap().as_i64(),
        Some(tid_a),
        "reverse landed on the earlier worker's breakpoint"
    );
    // The stopped thread's stack is live at the worker's load line again.
    let st = s.handle(&req(
        7,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(tid_a))]),
    ));
    let line = response(&st)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("line")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_eq!(line, 18, "back at the worker's load line");
}

#[test]
fn dap_over_bytecode_multithreaded_cross_thread_watchpoint() {
    // A data breakpoint on the raced window range [16384, 16392) fires in whichever worker's store touches it,
    // over DAP — the cross-thread watch reaching a DAP client on the scheduled bytecode engine.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    // Arm a write watch on [16384, 16392) directly (dataId = "addr:len") *before* configurationDone runs the
    // guest; the server reports it verified.
    let arm = s.handle(&req(
        3,
        "setDataBreakpoints",
        Json::obj(vec![(
            "breakpoints",
            Json::Arr(vec![Json::obj(vec![
                ("dataId", Json::s("16384:8")),
                ("accessType", Json::s("write")),
            ])]),
        )]),
    ));
    let verified = response(&arm)
        .get("body")
        .and_then(|b| b.get("breakpoints"))
        .and_then(|b| b.as_array())
        .and_then(|a| a.first())
        .and_then(|b| b.get("verified"))
        .cloned();
    assert_eq!(
        verified,
        Some(Json::Bool(true)),
        "the cross-thread data breakpoint arms on the scheduled engine"
    );
    // Run → a worker's store to [16384, 16392) trips it, reason "data breakpoint", in a spawned worker.
    let cont = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stop = event(&cont, "stopped").expect("the store trips the watch");
    assert_eq!(
        stop.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("data breakpoint")
    );
    assert!(
        stop.get("body")
            .unwrap()
            .get("threadId")
            .unwrap()
            .as_i64()
            .unwrap()
            >= 2,
        "tripped inside a spawned worker, not the root"
    );
}

// A futex handoff: the root seeds mem[16392], spawns a worker, sets a flag + `atomic.notify`s mem[16384],
// joins; the worker `atomic.wait`s on mem[16384] then reads mem[16392] (→ 987654) — the window addresses
// are one guard up (above the #1094 NULL guard). The worker's read-after-wait is
// line 26 (leading newline = line 1). Drives `memory.wait`/`notify` on the scheduled engine over DAP.
const FUTEX_HANDOFF: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 16392
  v1 = i64.const 987654
  i64.atomic.store v0 v1
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = i64.const 16384
  v5 = i32.const 1
  i32.atomic.store v4 v5
  v6 = i64.const 16384
  v7 = i32.const 1
  v8 = atomic.notify v6 v7
  v9 = thread.join v3
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, v0: i64) {
  v1 = i64.const 16384
  v2 = i32.const 0
  v3 = i64.const 1000000000
  v4 = i32.atomic.wait v1 v2 v3
  v5 = i64.const 16392
  v6 = i64.atomic.load v5
  return v6
  }
}
"#;

#[test]
fn dap_over_bytecode_multithreaded_wait_notify() {
    // A breakpoint after the worker's `atomic.wait` (line 26) fires only once the root's `notify` wakes
    // it — proving `memory.wait`/`notify` drive under the scheduled engine over DAP — then the guest
    // finishes with the handed-off value.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(FUTEX_HANDOFF)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(26))])]),
            ),
        ]),
    ));
    let cfg = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stop = event(&cfg, "stopped").expect("the woken worker stops after the wait");
    assert_eq!(
        stop.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("breakpoint")
    );
    assert!(
        stop.get("body")
            .unwrap()
            .get("threadId")
            .unwrap()
            .as_i64()
            .unwrap()
            >= 2,
        "stopped inside the spawned worker (woken by notify)"
    );
    // Continue → the join completes and the guest terminates.
    let done = s.handle(&req(5, "continue", Json::obj(vec![])));
    assert!(
        event(&done, "terminated").is_some(),
        "the futex handoff finished"
    );
}

// A generator fiber: the root cont.new's + cont.resume's a fiber that suspends (11) then returns (25);
// the root sums 11+25 = 36. The fiber's `v1 = i64.add varg v0` is line 17 (leading newline = line 1).
// Drives §12 fiber debugging on the single-vCPU engine over DAP (a fiber-only program is spawn-free, so
// it routes to the single-vCPU `DebugRun`).
const SUSPEND_ROUNDTRIP: &str = r#"
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 20
  v7, v8 = cont.resume v2 v6
  v9 = i64.add v5 v8
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = i64.const 1
  v1 = i64.add varg v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn dap_over_bytecode_breakpoint_inside_a_fiber() {
    // A breakpoint inside the fiber body (line 17) fires once `cont.resume` switches the debugged
    // continuation into the fiber — proving fiber debugging reaches a DAP client — then the guest
    // finishes with the summed value.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(SUSPEND_ROUNDTRIP)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&launch).get("success"),
        Some(&Json::Bool(true)),
        "the fiber program launches on the bytecode engine"
    );
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(17))])]),
            ),
        ]),
    ));
    let cfg = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stop = event(&cfg, "stopped").expect("stops inside the fiber");
    assert_eq!(
        stop.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("breakpoint")
    );
    // The stopped frame is the fiber body (line 17).
    let st = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let line = response(&st)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("line")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_eq!(line, 17, "stopped at the fiber's add");
    // Continue → the fiber suspends, resumes, returns; the guest terminates.
    let done = s.handle(&req(6, "continue", Json::obj(vec![])));
    assert!(
        event(&done, "terminated").is_some(),
        "the fiber generator finished"
    );
}

// A fibers-on-threads guest: two spawned workers each run a §12 fiber (`cont.new`/`resume`×2), then
// atomically add the fiber's return (25) into mem[16384] (one guard up) → 50. The fiber body's add is on line 32; the
// module spawns threads, so DAP routes it to the scheduled `ScheduledDebugRun`.
const FIBER_WORKERS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh0 = thread.spawn 1 vsp va
  vh1 = thread.spawn 1 vsp va
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 16384
  vr = i64.atomic.load vaddr
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vf = ref.func 2
  vz0 = i64.const 0
  vk = cont.new vf vz0
  vc1 = i64.const 10
  vs1, vr1 = cont.resume vk vc1
  vc2 = i64.const 20
  vs2, vr2 = cont.resume vk vc2
  vaddr = i64.const 16384
  vrmw = i64.atomic.rmw.add vaddr vr2
  vz = i64.const 0
  return vz
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp2: i64, varg2: i64) {
  v0 = i64.const 1
  v1 = i64.add varg2 v0
  v2 = suspend v1
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn dap_over_bytecode_fiber_on_a_spawned_thread() {
    // A breakpoint inside the fiber body (line 32) fires on a *spawned worker* thread — proving fibers
    // compose with threads all the way to a DAP client — with the stopped worker's own stack showing the
    // fiber frame, then the guest terminates with 25 + 25 = 50.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let launch = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(FIBER_WORKERS)),
            ("function", Json::i(0)),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&launch).get("success"),
        Some(&Json::Bool(true)),
        "the fiber-on-threads program launches on the bytecode engine"
    );
    s.handle(&req(
        3,
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("source.temt"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(33))])]),
            ),
        ]),
    ));
    let cfg = s.handle(&req(4, "configurationDone", Json::obj(vec![])));
    let stop = event(&cfg, "stopped").expect("stops inside the fiber on a worker");
    assert_eq!(
        stop.get("body").unwrap().get("reason").unwrap().as_str(),
        Some("breakpoint")
    );
    // The stopped vCPU is a spawned worker (DAP thread id ≥ 2), not the root.
    let tid = stop
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert!(tid >= 2, "a spawned worker (thread id ≥ 2), not the root");
    // The stopped worker's top frame is the fiber body (line 32).
    let st = s.handle(&req(
        5,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(tid))]),
    ));
    let line = response(&st)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()[0]
        .get("line")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_eq!(line, 33, "stopped at the fiber's add on the worker");
    // Continue → the other worker's fiber hits it too, then the guest terminates.
    let cont = s.handle(&req(6, "continue", Json::obj(vec![])));
    let stop2 = event(&cont, "stopped").expect("the second worker stops inside its fiber");
    let tid2 = stop2
        .get("body")
        .unwrap()
        .get("threadId")
        .unwrap()
        .as_i64()
        .unwrap();
    assert_ne!(tid, tid2, "each worker ran its own fiber on its own vCPU");
    let done = s.handle(&req(7, "continue", Json::obj(vec![])));
    assert!(
        event(&done, "terminated").is_some(),
        "the fiber-on-threads guest finished"
    );
}

// ---- W4 blocking stdin: the launch gate is fail-closed --------------------------------------------
// `blockStdin: true` parks a `read` on exhausted stdin instead of returning EOF — a session mode only
// the bytecode powerbox path supports (single-vCPU or threaded — #1146 deeper). Every unsupported
// combination must *fail the launch* rather than silently keep EOF semantics (invariant 9: decline,
// never diverge).

/// Launch `text` with the given extra args; returns the launch response's `success`.
fn launch_succeeds(text: &str, extra: Vec<(&str, Json)>) -> bool {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut args = vec![
        ("programText", Json::s(text)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![Json::i(3)])),
    ];
    args.extend(extra);
    let out = s.handle(&req(2, "launch", Json::obj(args)));
    response(&out).get("success") == Some(&Json::Bool(true))
}

#[test]
fn block_stdin_launch_gate_is_fail_closed() {
    // The tree-walker has no blocking-stdin mode: fail the launch, don't silently EOF.
    assert!(
        !launch_succeeds(
            LOOP_SUM_DBG,
            vec![
                ("powerbox", Json::s("onramp")),
                ("blockStdin", Json::Bool(true)),
            ],
        ),
        "treewalk + blockStdin must fail the launch"
    );
    // A deny-all (no powerbox) session has no stdin capability to block on.
    assert!(
        !launch_succeeds(
            LOOP_SUM_DBG,
            vec![
                ("engine", Json::s("bytecode")),
                ("blockStdin", Json::Bool(true)),
            ],
        ),
        "bytecode without the powerbox + blockStdin must fail the launch"
    );
    // The multithreaded scheduled engine is out of scope this slice: declined at construction.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(RACY_COUNTER)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
            ("blockStdin", Json::Bool(true)),
        ]),
    ));
    // #1146 (deeper): the threaded engine carries the park too (invariant 14) — a thread.spawn
    // module under the powerbox + blockStdin LAUNCHES (the full round trip is pinned below).
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "a thread.spawn powerbox module + blockStdin launches on the threaded engine"
    );
    // And the same launches *without* blockStdin still succeed — the gate rejects only the mode.
    assert!(
        launch_succeeds(LOOP_SUM_DBG, vec![("powerbox", Json::s("onramp"))]),
        "treewalk without blockStdin launches"
    );
    assert!(
        launch_succeeds(LOOP_SUM_DBG, vec![("engine", Json::s("bytecode"))]),
        "bytecode deny-all without blockStdin launches"
    );
}

#[test]
fn provide_stdin_fails_cleanly_on_a_non_blocking_session() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(
        3,
        "provideStdin",
        Json::obj(vec![("data", Json::s("hello\n"))]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "provideStdin on a non-blocking session fails cleanly"
    );
}

// ---- W4 on the threaded engine (#1146 deeper): park, provideStdin, resume, faithful replay -------

/// A threaded echo guest under the powerbox: the root spawns a worker (which just returns), joins it,
/// then twice `read`s stdin into the window and `write`s back exactly the bytes read, and `exit`s 0.
/// Each `read` finds the buffer exhausted, so under `blockStdin` it parks the session (twice).
const THREADED_ECHO: &str = r#"memory 16
import 0 "read" (i64, i64) -> (i64)
import 1 "write" (i64, i64) -> (i64)
import 2 "exit" (i32) -> ()
export 0 func "_start" 0

func () -> () {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vj = thread.join vh
  vptr = i64.const 16384
  vlen = i64.const 16
  vn = call.import 0 (vptr, vlen)
  vw = call.import 1 (vptr, vn)
  vn2 = call.import 0 (vptr, vlen)
  vw2 = call.import 1 (vptr, vn2)
  vcode = i32.const 0
  call.import 2 (vcode)
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

/// The reason of the batch's `stopped` event, if any.
fn stopped_reason(msgs: &[Json]) -> Option<String> {
    event(msgs, "stopped")
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("reason"))
        .and_then(|r| r.as_str())
        .map(|s| s.to_string())
}

/// The text of the last `output` event in a batch — the guest's captured stdout at this stop (full,
/// not a delta: it rewinds on a reverse `seek`). `None` if the batch carried none.
fn output_text(msgs: &[Json]) -> Option<String> {
    msgs.iter()
        .rev()
        .find(|m| m.get("event").and_then(|e| e.as_str()) == Some("output"))
        .and_then(|m| m.get("body"))
        .and_then(|b| b.get("output"))
        .and_then(|o| o.as_str())
        .map(|s| s.to_string())
}

/// **#1146 (deeper) — the blocking-stdin park on the threaded debug engine (invariant 14's debugger
/// axis)**: a `thread.spawn` guest under `blockStdin` parks at each exhausted `read` (`stopped`
/// reason `"stdin"`, not a dead-end `Blocked`), resumes on `provideStdin`, echoes the provided
/// bytes, and its reverse replay reproduces both inputs from the cap tape with **no new park** —
/// the same W4 acceptance the single-vCPU engine meets (`browser/tests/chibicc_debug.rs`).
#[test]
fn threaded_blocking_stdin_parks_provides_and_replays() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(THREADED_ECHO)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
            ("blockStdin", Json::Bool(true)),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "threaded blockStdin launch ok"
    );

    // No breakpoints: the run spawns + joins the worker, then parks at the first read (reason
    // "stdin") instead of EOF-completing — and instead of reporting a deadlock.
    let out = s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    assert_eq!(
        stopped_reason(&out).as_deref(),
        Some("stdin"),
        "parked awaiting input at the first read (not terminated/blocked: {out:?})"
    );

    // Provide the first input and resume: it echoes, then parks at the second read.
    let out = s.handle(&req(
        4,
        "provideStdin",
        Json::obj(vec![("data", Json::s("A\n"))]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "provideStdin accepted on the threaded engine"
    );
    let out = s.handle(&req(5, "continue", Json::obj(vec![])));
    assert_eq!(
        stopped_reason(&out).as_deref(),
        Some("stdin"),
        "parked at the second read: {out:?}"
    );
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\n"),
        "the first provided input was echoed"
    );

    // Provide the second input and resume to completion.
    s.handle(&req(
        6,
        "provideStdin",
        Json::obj(vec![("data", Json::s("B!"))]),
    ));
    let out = s.handle(&req(7, "continue", Json::obj(vec![])));
    assert!(
        event(&out, "terminated").is_some(),
        "ran to completion: {out:?}"
    );
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\nB!"),
        "both provided inputs were echoed"
    );

    // Rewind to the start: the captured output rewinds with the program. The rebuilt run replays
    // from the cap tape (and any checkpoint that captured the park re-admits the parked thread).
    let out = s.handle(&req(8, "reverseContinue", Json::obj(vec![])));
    assert!(
        event(&out, "stopped").is_some(),
        "rewound to the start: {out:?}"
    );
    assert_eq!(
        output_text(&out).as_deref().unwrap_or(""),
        "",
        "output rewound to empty at the start"
    );

    // Forward again: the tape serves both provided reads byte-identically — the run completes with
    // the same output and **no new stdin park** (the W4 replay acceptance, threaded).
    let out = s.handle(&req(9, "continue", Json::obj(vec![])));
    assert!(
        stopped_reason(&out).is_none(),
        "replay served the provided inputs from the tape — no re-park: {out:?}"
    );
    assert!(
        event(&out, "terminated").is_some(),
        "replay ran to completion"
    );
    assert_eq!(
        output_text(&out).as_deref(),
        Some("A\nB!"),
        "replay reproduced the provided inputs byte-identically"
    );
}

/// **Inertness pin** (invariant 9b), threaded: without `blockStdin` the same guest keeps plain EOF
/// semantics on the threaded engine — exhausted reads return 0 and the run completes with no stop.
#[test]
fn threaded_without_block_stdin_exhausted_reads_stay_eof() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(THREADED_ECHO)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("powerbox", Json::s("onramp")),
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    let out = s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    assert!(
        stopped_reason(&out).is_none(),
        "no park without blockStdin: {out:?}"
    );
    assert!(
        event(&out, "terminated").is_some(),
        "EOF reads completed the run: {out:?}"
    );
    assert_eq!(
        output_text(&out).as_deref().unwrap_or(""),
        "",
        "two zero-length echoes"
    );
}

/// A promoted-scalar local `x` (an `ssalist` var — no window address) whose value changes: `x = 10`
/// (value 0) until the add at inst 3, then `x = 11` (value 2). Watching it needs a **value** watch
/// (#1229) — the window-range data breakpoint can't reach an address-less SSA local.
const VALUE_WATCH_DBG: &str = r#"func () -> (i64) {
block 0 () {
  va = i64.const 10
  vb = i64.const 1
  vc = i64.add va vb
  vd = i64.add vc vb
  return vd
  }
}
debug.file 0 "vw.temt"
debug.fname 0 "main"
debug.loc 0 0 0 0 2 3
debug.loc 0 0 1 0 3 3
debug.loc 0 0 2 0 4 3
debug.loc 0 0 3 0 5 3
debug.loc 0 0 4 0 6 3
debug.var 0 "x" ssalist 2 0 1 0 0 3 2 "int"
"#;

/// **#1229 — value watchpoint by name over DAP, engine parity.** Arm a data breakpoint on the
/// address-less SSA local `x` (the panel's watch-a-variable flow: `dataBreakpointInfo` mints a value
/// `dataId`, `setDataBreakpoints` arms it), then Continue: the debugger stops with reason
/// "data breakpoint" when `x`'s value changes — identically on the bytecode engine and the
/// tree-walker. The window-range watch path can't serve this variable, so this exercises the value
/// watch specifically (the minted `dataId` is the `val:…` form).
#[test]
fn dap_over_bytecode_value_watchpoint_matches_the_tree_walker() {
    fn script(engine: Option<&str>) -> (String, bool, String) {
        let mut s = DapServer::new();
        s.handle(&req(1, "initialize", Json::obj(vec![])));
        let mut la = vec![
            ("programText", Json::s(VALUE_WATCH_DBG)),
            ("function", Json::i(0)),
        ];
        if let Some(e) = engine {
            la.push(("engine", Json::s(e)));
        }
        s.handle(&req(2, "launch", Json::obj(la)));
        // Break on line 3 (inst 1) so we pause with `x` (= 10) live to arm the watch.
        s.handle(&req(
            3,
            "setBreakpoints",
            Json::obj(vec![
                ("source", Json::obj(vec![("path", Json::s("vw.temt"))])),
                (
                    "breakpoints",
                    Json::Arr(vec![Json::obj(vec![("line", Json::i(3))])]),
                ),
            ]),
        ));
        s.handle(&req(4, "configurationDone", Json::obj(vec![])));
        let st = s.handle(&req(
            5,
            "stackTrace",
            Json::obj(vec![("threadId", Json::i(1))]),
        ));
        let frame_id = response(&st)
            .get("body")
            .unwrap()
            .get("stackFrames")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("id")
            .unwrap()
            .as_i64()
            .unwrap();
        let sc = s.handle(&req(
            6,
            "scopes",
            Json::obj(vec![("frameId", Json::i(frame_id))]),
        ));
        let var_ref = response(&sc)
            .get("body")
            .unwrap()
            .get("scopes")
            .unwrap()
            .as_array()
            .unwrap()[0]
            .get("variablesReference")
            .unwrap()
            .as_i64()
            .unwrap();
        let info = s.handle(&req(
            7,
            "dataBreakpointInfo",
            Json::obj(vec![
                ("variablesReference", Json::i(var_ref)),
                ("name", Json::s("x")),
            ]),
        ));
        let data_id = response(&info)
            .get("body")
            .and_then(|b| b.get("dataId"))
            .and_then(|d| d.as_str())
            .map(str::to_owned)
            .unwrap_or_default();
        let arm = s.handle(&req(
            8,
            "setDataBreakpoints",
            Json::obj(vec![(
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![
                    ("dataId", Json::s(&data_id)),
                    ("accessType", Json::s("write")),
                ])]),
            )]),
        ));
        let verified = response(&arm)
            .get("body")
            .and_then(|b| b.get("breakpoints"))
            .and_then(|b| b.as_array())
            .and_then(|a| a.first())
            .and_then(|b| b.get("verified"))
            .cloned()
            == Some(Json::Bool(true));
        // Continue → stop when `x`'s value changes.
        let cont = s.handle(&req(9, "continue", Json::obj(vec![])));
        let reason = event(&cont, "stopped")
            .and_then(|e| e.get("body")?.get("reason")?.as_str().map(str::to_owned))
            .unwrap_or_default();
        (data_id, verified, reason)
    }

    let bytecode = script(Some("bytecode"));
    assert!(
        bytecode.0.starts_with("val:"),
        "`x` resolves to a value dataId (no window address): {:?}",
        bytecode.0
    );
    assert!(
        bytecode.1,
        "the value data breakpoint verifies on the bytecode engine"
    );
    assert_eq!(bytecode.2, "data breakpoint", "stops for the value watch");
    assert_eq!(
        bytecode,
        script(None),
        "bytecode value watchpoint ≡ tree-walker value watchpoint"
    );
}
