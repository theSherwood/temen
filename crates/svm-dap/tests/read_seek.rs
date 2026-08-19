//! The two request arms an interactive embedder's memory panel + history slider need (surfaced by
//! the c_interpret migration spike): the standard **`readMemory`** (base64 window reads — the
//! `writeMemory` twin) and the custom **`seek`** (jump to time `t` on the reverse coordinate,
//! forward or backward, landing as an ordinary stop). Both fail closed without a session.

use svm_dap::{DapServer, Json};

mod support;
use support::{req, response};

/// Loads the i64 at window address 8 and returns it (the slice-8 fixture).
const LOAD_CELL: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 8
  v1 = i64.load v0
  return v1
  }
}
"#;

/// The sum loop, storing the running total to mem[8] each iteration — seekable, memory-visible.
const STORE_LOOP: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vn = i64.const 5
  vacc0 = i64.const 0
  br 1(vn, vacc0)
}
block 1 (vi: i64, vacc: i64) {
  vsum = i64.add vacc vi
  vaddr = i64.const 8
  i64.store vaddr vsum
  vone = i64.const 1
  vnext = i64.sub vi vone
  br_if vnext 1(vnext, vsum) 2(vsum)
}
block 2 (vr: i64) {
  return vr
  }
}
"#;

fn launch(s: &mut DapServer, src: &str, engine: Option<&str>) {
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let mut args = vec![
        ("programText", Json::s(src)),
        ("function", Json::i(0)),
        ("args", Json::Arr(vec![])),
    ];
    if let Some(e) = engine {
        args.push(("engine", Json::s(e)));
    }
    let out = s.handle(&req(2, "launch", Json::obj(args)));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "launch ok"
    );
}

/// **`readMemory` round-trips `writeMemory`**: write 42 at 0x8, read it back (plain reference,
/// hex reference + offset, and a short count exercising the base64 padding shapes); the
/// capability is advertised; an unmapped range and a missing count fail cleanly.
#[test]
fn read_memory_round_trips_and_fails_closed() {
    let mut s = DapServer::new();
    let out = s.handle(&req(1, "initialize", Json::obj(vec![])));
    assert_eq!(
        response(&out)
            .get("body")
            .and_then(|b| b.get("supportsReadMemoryRequest"))
            .cloned(),
        Some(Json::Bool(true)),
        "readMemory advertised"
    );
    launch(&mut s, LOAD_CELL, Some("bytecode"));
    let out = s.handle(&req(
        3,
        "writeMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("0x8")),
            ("data", Json::s("KgAAAAAAAAA=")), // 42u64 little-endian
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    let read = |s: &mut DapServer, seq, args: Vec<(&str, Json)>| {
        let out = s.handle(&req(seq, "readMemory", Json::obj(args)));
        response(&out).clone()
    };
    let r = read(
        &mut s,
        4,
        vec![("memoryReference", Json::s("8")), ("count", Json::i(8))],
    );
    assert_eq!(r.get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        r.get("body").and_then(|b| b.get("data")).cloned(),
        Some(Json::s("KgAAAAAAAAA=")),
        "the write reads back"
    );
    assert_eq!(
        r.get("body").and_then(|b| b.get("address")).cloned(),
        Some(Json::s("0x8")),
        "the landed address echoes"
    );
    // Hex reference + offset resolve to the same cell; a 2-byte count exercises `=` padding.
    let r = read(
        &mut s,
        5,
        vec![
            ("memoryReference", Json::s("0x0")),
            ("offset", Json::i(8)),
            ("count", Json::i(2)),
        ],
    );
    assert_eq!(
        r.get("body").and_then(|b| b.get("data")).cloned(),
        Some(Json::s("KgA=")),
        "offset + short count"
    );
    // Fail-closed: an unmapped range; a missing count.
    let r = read(
        &mut s,
        6,
        vec![
            ("memoryReference", Json::s("0x20000000000")),
            ("count", Json::i(8)),
        ],
    );
    assert_eq!(r.get("success"), Some(&Json::Bool(false)), "unmapped range");
    let r = read(&mut s, 7, vec![("memoryReference", Json::s("8"))]);
    assert_eq!(r.get("success"), Some(&Json::Bool(false)), "missing count");
}

/// **`seek` rewinds and re-lands exactly**: run the store loop out, seek to 0 — memory reads the
/// *original* zeros and a stopped event fires; seek to a mid-run t lands on that exact t; a seek
/// past the end terminates again. The tree-walker honors the verb too.
#[test]
fn seek_jumps_both_ways_on_the_time_coordinate() {
    let mut s = DapServer::new();
    launch(&mut s, STORE_LOOP, Some("bytecode"));
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "ran out"
    );
    // Rewind to the start: landed t = 0, a stopped event, and the cell reads the original zeros.
    let out = s.handle(&req(4, "seek", Json::obj(vec![("t", Json::i(0))])));
    let r = response(&out);
    assert_eq!(r.get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        r.get("body").and_then(|b| b.get("t")).cloned(),
        Some(Json::i(0)),
        "landed at 0"
    );
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped")),
        "seek lands as a stop"
    );
    let out = s.handle(&req(
        5,
        "readMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("8")),
            ("count", Json::i(8)),
        ]),
    ));
    assert_eq!(
        response(&out)
            .get("body")
            .and_then(|b| b.get("data"))
            .cloned(),
        Some(Json::s("AAAAAAAAAAA=")),
        "at t=0 the store hasn't happened"
    );
    // Forward to a mid-run t: the slider lands exactly where asked.
    let out = s.handle(&req(6, "seek", Json::obj(vec![("t", Json::i(7))])));
    assert_eq!(
        response(&out).get("body").and_then(|b| b.get("t")).cloned(),
        Some(Json::i(7)),
        "landed exactly at the requested t"
    );
    // Past the end: terminates (again) rather than failing.
    let out = s.handle(&req(7, "seek", Json::obj(vec![("t", Json::i(1_000_000))])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "a seek past the end terminates"
    );

    // The tree-walker honors the verb too (its Inspector has native seek).
    let mut s = DapServer::new();
    launch(&mut s, STORE_LOOP, None);
    let out = s.handle(&req(3, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "tree-walker ran out"
    );
    let out = s.handle(&req(4, "seek", Json::obj(vec![("t", Json::i(0))])));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(true)),
        "tree-walker seek(0)"
    );

    // Fail-closed: no session.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(2, "seek", Json::obj(vec![("t", Json::i(0))])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
}
