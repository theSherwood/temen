//! Slice 8 (INTERACTIVE_EMBEDDING.md): **state writes on the input tape** — DAP
//! `setVariable`/`writeMemory` recorded at the clock they land and re-applied whenever execution
//! passes that clock on *any* path, so a live continue and a seek replay reach identical states.
//! Acceptance: write → seek back → seek forward re-observes the write; a seek to *before* the
//! write's clock reads the original state; everything fails closed off the bytecode path.

use temen_dap::{BytecodeBackend, DapServer, Debuggee, Json};

mod support;
use support::{req, response, LOOP_SUM_DBG};
use temen_interp::{IrPc, Stop, Value, VarValue};
use temen_text::parse_module;

/// Loads the i64 at window address 16392 (cell 8 shifted above the #1094 NULL guard) and returns
/// it — 0 unless a debugger write lands first.
const LOAD_CELL: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 16392
  v1 = i64.load v0
  return v1
  }
}
"#;

fn backend(src: &str, args: &[Value]) -> BytecodeBackend {
    let m = parse_module(src).expect("parses");
    BytecodeBackend::new(m, 0, args, u64::MAX, false, Vec::new(), false, None, None)
        .expect("bytecode subset")
}

fn finish(b: &mut BytecodeBackend) -> Vec<Value> {
    loop {
        match Debuggee::run_until_stop(b) {
            Stop::Finished(r) => return r.expect("clean finish"),
            Stop::Break { .. } => continue,
            Stop::Blocked => panic!("unexpected block"),
        }
    }
}

/// **A variable write shifts the result and survives `seek`**: break at the loop head (`i = 3`,
/// `acc = 0`), write `acc = 100`, run out → `106` instead of `6`; `seek(0)` and re-drive — the
/// recorded write re-applies at the same clock on the replay path, so the result is `106` again.
#[test]
fn var_write_shifts_the_result_and_survives_seek() {
    let mut b = backend(LOOP_SUM_DBG, &[Value::I32(3)]);
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 1,
        inst: 0,
    };
    Debuggee::set_breakpoint(&mut b, bp);
    let Stop::Break { pc, .. } = Debuggee::run_until_stop(&mut b) else {
        panic!("expected the loop-head breakpoint");
    };
    assert_eq!(pc, bp);
    assert_eq!(
        Debuggee::read_var(&b, 0, "acc", 4),
        Some(VarValue::Value(Value::I32(0))),
        "pre-write acc"
    );
    assert!(
        Debuggee::write_var(&mut b, 0, "acc", 100, 4),
        "the write lands"
    );
    assert_eq!(
        Debuggee::read_var(&b, 0, "acc", 4),
        Some(VarValue::Value(Value::I32(100))),
        "the live readback sees the write"
    );
    Debuggee::clear_breakpoint(&mut b, bp);
    assert_eq!(finish(&mut b), vec![Value::I32(106)], "3+2+1 on top of 100");
    // Rewind to the start and re-drive: the replay path re-applies the write at its clock.
    let _ = Debuggee::seek(&mut b, 0);
    assert_eq!(
        finish(&mut b),
        vec![Value::I32(106)],
        "seek(0) + rerun re-observes the write"
    );
}

/// **A window write shifts the result, a seek to before its clock reads the original bytes, and
/// the replay past its clock re-applies it** — the history slider stays truthful around an edit.
#[test]
fn window_write_survives_seek_and_predates_nothing() {
    let mut b = backend(LOAD_CELL, &[]);
    let bp = IrPc {
        module: 0,
        func: 0,
        block: 0,
        inst: 1,
    };
    Debuggee::set_breakpoint(&mut b, bp);
    let Stop::Break { .. } = Debuggee::run_until_stop(&mut b) else {
        panic!("expected the pre-load breakpoint");
    };
    assert_eq!(
        Debuggee::read_window(&b, 16392, 8).expect("readable"),
        vec![0u8; 8],
        "the cell starts zeroed"
    );
    assert!(
        Debuggee::write_window(&mut b, 16392, &42i64.to_le_bytes()),
        "the write lands"
    );
    Debuggee::clear_breakpoint(&mut b, bp);
    assert_eq!(finish(&mut b), vec![Value::I64(42)], "the load observes 42");
    // Seek to before the write's clock: the rebuilt memory is the *original* zeros.
    let _ = Debuggee::seek(&mut b, 0);
    assert_eq!(
        Debuggee::read_window(&b, 16392, 8).expect("readable"),
        vec![0u8; 8],
        "before the write's clock the original state shows"
    );
    // Drive forward again: passing the write's clock re-applies it on the replay path too.
    assert_eq!(
        finish(&mut b),
        vec![Value::I64(42)],
        "seek(0) + rerun re-observes the write"
    );
}

/// Two workers each `mem[16384] += 1` (counter cell shifted above the #1094 NULL guard); the root
/// joins both and returns the count — the threaded twin.
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

/// **The scheduled engine records writes at turns**: break in a worker, seed the counter cell to
/// 100 — the final count reflects it, and a `seek(0)` + re-drive reproduces the identical result.
#[test]
fn scheduled_window_write_survives_seek() {
    let mut b = backend(RACY_COUNTER, &[]);
    let bp = IrPc {
        module: 0,
        func: 1,
        block: 0,
        inst: 1,
    };
    Debuggee::set_breakpoint(&mut b, bp);
    let Stop::Break { .. } = Debuggee::run_until_stop(&mut b) else {
        panic!("expected the worker breakpoint");
    };
    assert!(
        Debuggee::write_window(&mut b, 16384, &100i64.to_le_bytes()),
        "the write lands on the scheduled engine"
    );
    Debuggee::clear_breakpoint(&mut b, bp);
    let live = finish(&mut b);
    let Some(Value::I64(count)) = live.first().copied() else {
        panic!("an i64 count");
    };
    assert!(count >= 100, "the seeded base shows in the count ({count})");
    let _ = Debuggee::seek(&mut b, 0);
    assert_eq!(
        finish(&mut b),
        live,
        "seek(0) + rerun re-observes the write at its turn"
    );
}

/// stackTrace → scopes → the top frame's `variablesReference` + the named var's current value.
fn top_scope_var(s: &mut DapServer, seq: i64, name: &str) -> (i64, String) {
    let out = s.handle(&req(
        seq,
        "stackTrace",
        Json::obj(vec![("threadId", Json::i(1))]),
    ));
    let frames = response(&out)
        .get("body")
        .unwrap()
        .get("stackFrames")
        .unwrap()
        .as_array()
        .unwrap()
        .to_vec();
    let frame_id = frames[0].get("id").unwrap().as_i64().unwrap();
    let out = s.handle(&req(
        seq,
        "scopes",
        Json::obj(vec![("frameId", Json::i(frame_id))]),
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
    let value = response(&out)
        .get("body")
        .unwrap()
        .get("variables")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .find(|v| v.get("name").and_then(|n| n.as_str()) == Some(name))
        .and_then(|v| v.get("value").and_then(|v| v.as_str()).map(String::from))
        .unwrap_or_else(|| "?".into());
    (vref, value)
}

/// **The DAP surface end-to-end**: `initialize` advertises both write capabilities; `setVariable`
/// on a stopped bytecode session writes through the scope reference and the Variables pane reads
/// the new value back; `writeMemory` takes base64 at a hex reference and reports `bytesWritten`.
#[test]
fn dap_set_variable_and_write_memory_round_trip() {
    let mut s = DapServer::new();
    let out = s.handle(&req(1, "initialize", Json::obj(vec![])));
    let caps = response(&out).get("body").cloned().expect("capabilities");
    assert_eq!(
        caps.get("supportsSetVariable"),
        Some(&Json::Bool(true)),
        "setVariable advertised"
    );
    assert_eq!(
        caps.get("supportsWriteMemoryRequest"),
        Some(&Json::Bool(true)),
        "writeMemory advertised"
    );
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
        "setBreakpoints",
        Json::obj(vec![
            ("source", Json::obj(vec![("path", Json::s("sum.c"))])),
            (
                "breakpoints",
                Json::Arr(vec![Json::obj(vec![("line", Json::i(7))])]),
            ),
        ]),
    ));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(true)));
    let out = s.handle(&req(4, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("stopped")),
        "stopped at the breakpoint"
    );
    let (vref, before) = top_scope_var(&mut s, 5, "acc");
    assert_eq!(before, "0", "acc before the write");
    let out = s.handle(&req(
        6,
        "setVariable",
        Json::obj(vec![
            ("variablesReference", Json::i(vref)),
            ("name", Json::s("acc")),
            ("value", Json::s("100")),
        ]),
    ));
    let resp = response(&out);
    assert_eq!(resp.get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        resp.get("body").and_then(|b| b.get("value")).cloned(),
        Some(Json::s("100")),
        "the response echoes the written value"
    );
    let (_, after) = top_scope_var(&mut s, 7, "acc");
    assert_eq!(after, "100", "the Variables pane reads the write back");

    // writeMemory needs a session with a window — a fresh LOAD_CELL launch.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOAD_CELL)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(
        3,
        "writeMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("0x4008")), // cell 8 above the #1094 NULL guard (16392)
            ("data", Json::s("KgAAAAAAAAA=")),      // 42u64 little-endian
        ]),
    ));
    let resp = response(&out);
    assert_eq!(resp.get("success"), Some(&Json::Bool(true)));
    assert_eq!(
        resp.get("body")
            .and_then(|b| b.get("bytesWritten"))
            .cloned(),
        Some(Json::i(8)),
        "eight bytes written"
    );
    let out = s.handle(&req(4, "continue", Json::obj(vec![])));
    assert!(
        out.iter()
            .any(|m| m.get("event").and_then(|e| e.as_str()) == Some("terminated")),
        "runs out"
    );
}

/// **Fail-closed**: the tree-walker declines both writes; malformed base64 and an unknown scope
/// reference fail cleanly on the bytecode engine.
#[test]
fn writes_fail_closed_off_the_bytecode_path() {
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOOP_SUM_DBG)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![Json::i(3)])),
        ]),
    ));
    let out = s.handle(&req(
        3,
        "writeMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("8")),
            ("data", Json::s("KgAAAAAAAAA=")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "the tree-walker declines writeMemory"
    );
    let out = s.handle(&req(
        4,
        "setVariable",
        Json::obj(vec![
            ("variablesReference", Json::i(1)),
            ("name", Json::s("acc")),
            ("value", Json::s("100")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "the tree-walker declines setVariable"
    );

    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(LOAD_CELL)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
        ]),
    ));
    let out = s.handle(&req(
        3,
        "writeMemory",
        Json::obj(vec![
            ("memoryReference", Json::s("8")),
            ("data", Json::s("not base64!")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "malformed base64 fails cleanly"
    );
    let out = s.handle(&req(
        4,
        "setVariable",
        Json::obj(vec![
            ("variablesReference", Json::i(99)),
            ("name", Json::s("acc")),
            ("value", Json::s("1")),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "an unknown scope reference fails cleanly"
    );
}
