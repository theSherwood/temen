//! #1366 slice (c) — **declared host-completed caps over DAP**, end to end: a launch names the caps
//! the embedder services (`hostCaps`); the guest's flat `call.sym "<name>"` parks the session as
//! `stopped { reason: "cap", capId, capName, args }`; the client answers with `provideCap { id,
//! value }` and resumes; the delivered value lands in the call's result slot. A reverse `seek`
//! replays the delivered value from the cap tape — no re-park, no second request (the W4
//! `provideStdin` inertness pin generalized). Without `hostCaps` the import stays unbound and the
//! program traps (fail-closed opt-in); with a non-bytecode/non-powerbox launch the arg is refused.

mod support;
use support::{req, response};
use temen_dap::{DapServer, Json};

/// `blink(h, 7, 35)` — one flat call on a declared cap (the guest passes a placeholder handle;
/// the binding carries the granted one, like `vm_fs`). Returns whatever the host delivers.
const BLINK: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  vop = i64.const 7
  varg = i64.const 35
  vr = call.sym "blink" (i64, i64) -> (i64) vh (vop, varg)
  return vr
  }
}
"#;

fn events<'a>(msgs: &'a [Json], name: &str) -> Vec<&'a Json> {
    msgs.iter()
        .filter(|m| {
            m.get("type").and_then(|t| t.as_str()) == Some("event")
                && m.get("event").and_then(|e| e.as_str()) == Some(name)
        })
        .collect()
}

fn launch(s: &mut DapServer, host_caps: Option<Vec<&str>>, engine: &str, powerbox: bool) -> bool {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut args = vec![
        ("programText", Json::s(BLINK)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
        ("engine", Json::s(engine)),
    ];
    if powerbox {
        args.push(("powerbox", Json::s("onramp")));
    }
    if let Some(caps) = host_caps {
        args.push((
            "hostCaps",
            Json::Arr(caps.into_iter().map(Json::s).collect()),
        ));
    }
    let out = s.handle(&req(2, "launch", Json::obj(args)));
    let ok = response(&out).get("success") == Some(&Json::Bool(true));
    if ok {
        s.handle(&req(3, "configurationDone", Json::obj(vec![])));
    }
    ok
}

/// The round-trip: park with the request surfaced → `provideCap` → the value is the program's
/// result.
#[test]
fn declared_cap_parks_with_the_request_and_provide_cap_resumes() {
    let mut s = DapServer::new();
    assert!(
        launch(&mut s, Some(vec!["blink"]), "bytecode", true),
        "launch with hostCaps"
    );

    let out = s.handle(&req(
        4,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let stops = events(&out, "stopped");
    assert_eq!(stops.len(), 1, "one stop: {out:?}");
    let body = stops[0].get("body").expect("stopped body");
    assert_eq!(body.get("reason"), Some(&Json::s("cap")));
    assert_eq!(body.get("capName"), Some(&Json::s("blink")));
    assert_eq!(
        body.get("args"),
        Some(&Json::Arr(vec![Json::i(7), Json::i(35)])),
        "the guest's flat call arguments ride the event"
    );
    let id = body.get("capId").and_then(|v| v.as_i64()).expect("capId");
    assert!(
        events(&out, "exited").is_empty(),
        "not finished while parked"
    );

    // A resume without delivering does not run anything: still parked on the same call.
    let out = s.handle(&req(
        5,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let stops = events(&out, "stopped");
    assert_eq!(stops.len(), 1);
    assert_eq!(
        stops[0]
            .get("body")
            .and_then(|b| b.get("capId"))
            .and_then(|v| v.as_i64()),
        Some(id),
        "still parked on the same request"
    );

    // Deliver, resume: the program returns the delivered value.
    let out = s.handle(&req(
        6,
        "provideCap",
        Json::obj(vec![("id", Json::i(id)), ("value", Json::i(42))]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    let out = s.handle(&req(
        7,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let exited = events(&out, "exited");
    assert_eq!(exited.len(), 1, "finished: {out:?}");
    assert_eq!(
        exited[0].get("body").and_then(|b| b.get("exitCode")),
        Some(&Json::i(42)),
        "the delivered value is the program's result"
    );
}

/// The replay pin: after the run finished, a `stepBack` rebuilds and replays from the tape — the
/// declared call is served from the tape (no `cap` stop, no second request) and the program
/// finishes with the same value.
#[test]
fn reverse_step_replays_the_delivered_value_without_re_parking() {
    let mut s = DapServer::new();
    assert!(launch(&mut s, Some(vec!["blink"]), "bytecode", true));
    let out = s.handle(&req(
        4,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let id = events(&out, "stopped")[0]
        .get("body")
        .and_then(|b| b.get("capId"))
        .and_then(|v| v.as_i64())
        .expect("parked");
    s.handle(&req(
        5,
        "provideCap",
        Json::obj(vec![("id", Json::i(id)), ("value", Json::i(42))]),
    ));
    let out = s.handle(&req(
        6,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert_eq!(events(&out, "exited").len(), 1, "first run finished");

    // Rewind one step (a rebuild + tape replay), then run to the end again.
    let out = s.handle(&req(
        7,
        "stepBack",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "stepBack: {out:?}"
    );
    for st in events(&out, "stopped") {
        assert_ne!(
            st.get("body").and_then(|b| b.get("reason")),
            Some(&Json::s("cap")),
            "the replay never re-parks on the declared call"
        );
    }
    let out = s.handle(&req(
        8,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    for st in events(&out, "stopped") {
        assert_ne!(
            st.get("body").and_then(|b| b.get("reason")),
            Some(&Json::s("cap"))
        );
    }
    let exited = events(&out, "exited");
    assert_eq!(exited.len(), 1, "replayed to the end: {out:?}");
    assert_eq!(
        exited[0].get("body").and_then(|b| b.get("exitCode")),
        Some(&Json::i(42)),
        "the tape reproduced the delivered value"
    );
}

/// Opt-in, fail-closed: without `hostCaps` the `blink` import is unbound and the call traps.
#[test]
fn an_undeclared_cap_import_stays_unbound_and_traps() {
    let mut s = DapServer::new();
    assert!(launch(&mut s, None, "bytecode", true));
    let out = s.handle(&req(
        4,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert!(
        events(&out, "stopped")
            .iter()
            .all(|st| st.get("body").and_then(|b| b.get("reason")) != Some(&Json::s("cap"))),
        "no cap park without a declaration"
    );
    let exited = events(&out, "exited");
    assert_eq!(exited.len(), 1, "the program terminated: {out:?}");
    assert!(
        exited[0].get("body").and_then(|b| b.get("trap")).is_some(),
        "…by trapping on the unbound import: {out:?}"
    );
}

/// The launch gate: `hostCaps` needs the bytecode engine under the on-ramp powerbox.
#[test]
fn host_caps_are_refused_outside_the_bytecode_powerbox() {
    let mut s = DapServer::new();
    assert!(
        !launch(&mut s, Some(vec!["blink"]), "bytecode", false),
        "no powerbox ⇒ refused"
    );
    let mut s = DapServer::new();
    assert!(
        !launch(&mut s, Some(vec!["blink"]), "tree", true),
        "tree-walker ⇒ refused"
    );
}
