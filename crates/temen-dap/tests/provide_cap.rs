//! #1366 slice (b) — the custom **`provideCap`** request (the host-completed-cap twin of W4's
//! `provideStdin`): it fails cleanly without a session, with malformed arguments, and when the
//! session is not parked on the given completion id. (The park → `stopped{reason:"cap", capId}` →
//! `provideCap` → resume round-trip is pinned at the engine level in
//! `temen-interp/tests/debug_run_cap_park.rs`; a DAP-level round-trip needs a launch that grants
//! a host proc — the embedder-declared-caps launch, tracked under #1366 slice (c).)

mod support;
use support::{req, response};
use temen_dap::{DapServer, Json};

const TRIVIAL: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 1
  return v0
  }
}
"#;

fn provide(s: &mut DapServer, seq: i64, args: Json) -> bool {
    let out = s.handle(&req(seq, "provideCap", args));
    response(&out).get("success") == Some(&Json::Bool(true))
}

#[test]
fn provide_cap_fails_cleanly_without_a_session() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    assert!(
        !provide(
            &mut s,
            2,
            Json::obj(vec![("id", Json::i(0)), ("value", Json::i(1))])
        ),
        "no session ⇒ provideCap is refused, not a crash"
    );
}

#[test]
fn provide_cap_fails_cleanly_when_not_parked_or_malformed() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(TRIVIAL)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "bytecode launch"
    );
    s.handle(&req(3, "configurationDone", Json::obj(vec![])));

    // Malformed: missing value / negative id.
    assert!(!provide(&mut s, 4, Json::obj(vec![("id", Json::i(0))])));
    assert!(!provide(
        &mut s,
        5,
        Json::obj(vec![("id", Json::i(-1)), ("value", Json::i(1))])
    ));
    // Well-formed but the session is parked on nothing: refused, session untouched.
    assert!(!provide(
        &mut s,
        6,
        Json::obj(vec![("id", Json::i(0)), ("value", Json::i(1))])
    ));
    // The session still runs to completion normally afterwards.
    let out = s.handle(&req(
        7,
        "continue",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
}
