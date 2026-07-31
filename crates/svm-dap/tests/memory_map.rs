//! Slice 5 (INTERACTIVE_EMBEDDING.md): the **Memory-capability growth cap** (the OOM-teaching
//! knob — a `vm_map` past the limit returns `-ENOMEM`, so a guest allocator observes NULL) and the
//! **window memory-map introspection** (`memoryMap`): geometry, explicit-state pages, segments,
//! and the powerbox stack/heap regions. Both fail closed where unsupported.

use svm_dap::{BytecodeBackend, DapServer, Debuggee, Json};
use svm_interp::{Stop, Value};
use svm_text::parse_module;

/// A guest that `vm_map`s two 64 KiB blocks into the reserved tail and returns the *second* map's
/// result — `0` unbounded, `-ENOMEM` under a one-block limit.
const MAP_TWICE: &str = r#"memory 16
import 0 "vm_map" (i64, i64, i64) -> (i64)

func () -> (i64) {
block 0 () {
  v0 = i64.const 65536
  v1 = i64.const 65536
  v2 = i64.const 3
  v3 = call.import 0 (v0, v1, v2)
  v4 = i64.const 131072
  v5 = call.import 0 (v4, v1, v2)
  return v5
  }
}
"#;

fn run_map_twice(mem_limit: Option<u64>) -> (i64, BytecodeBackend) {
    let m = parse_module(MAP_TWICE).expect("parses");
    let mut b = BytecodeBackend::new(
        m,
        0,
        &[],
        u64::MAX,
        true,
        Vec::new(),
        false,
        mem_limit,
        None,
    )
    .expect("subset");
    let stop = Debuggee::run_until_stop(&mut b);
    let Stop::Finished(Ok(vals)) = stop else {
        panic!("the map guest should finish cleanly: {stop:?}");
    };
    let Some(Value::I64(r)) = vals.first().copied() else {
        panic!("an i64 result");
    };
    (r, b)
}

/// **The growth cap**: unbounded, both maps succeed; with a one-block limit, the second map fails
/// probeably with `-ENOMEM` (the value a guest `malloc` turns into NULL) — and the guest still
/// runs to completion (invariant 5: errors are values, never traps).
#[test]
fn memory_limit_fails_the_map_past_it_probeably() {
    let (unbounded, _) = run_map_twice(None);
    assert!(unbounded >= 0, "unbounded: both maps succeed ({unbounded})");
    let (capped, _) = run_map_twice(Some(65536));
    assert_eq!(capped, -12, "the second map returns -ENOMEM at the limit");
}

/// **`vm_unmap` returns bytes to the budget**: map, unmap, map again under a one-block limit —
/// the third call succeeds because the unmap freed the budget.
#[test]
fn unmap_returns_bytes_to_the_budget() {
    const SRC: &str = r#"memory 16
import 0 "vm_map" (i64, i64, i64) -> (i64)
import 1 "vm_unmap" (i64, i64) -> (i64)

func () -> (i64) {
block 0 () {
  v0 = i64.const 65536
  v1 = i64.const 65536
  v2 = i64.const 3
  v3 = call.import 0 (v0, v1, v2)
  v4 = call.import 1 (v0, v1)
  v5 = i64.const 131072
  v6 = call.import 0 (v5, v1, v2)
  return v6
  }
}
"#;
    let m = parse_module(SRC).expect("parses");
    let mut b = BytecodeBackend::new(
        m,
        0,
        &[],
        u64::MAX,
        true,
        Vec::new(),
        false,
        Some(65536),
        None,
    )
    .expect("subset");
    let Stop::Finished(Ok(vals)) = Debuggee::run_until_stop(&mut b) else {
        panic!("finishes");
    };
    assert_eq!(
        vals.first().copied(),
        Some(Value::I64(0)),
        "map after unmap succeeds under the same limit"
    );
}

/// **The memory-map JSON**: geometry (mapped/reserved/page size), the grown tail pages as `rw`
/// entries after the guest's `vm_map`, the stack constants — and it fails cleanly on the
/// tree-walker.
#[test]
fn memory_map_reports_geometry_and_grown_pages() {
    let (_, b) = run_map_twice(None);
    let map = b.memory_map().expect("bytecode exposes the memory map");
    assert_eq!(
        map.get("mapped").and_then(|v| v.as_i64()),
        Some(65536),
        "declared window"
    );
    assert_eq!(
        map.get("reserved").and_then(|v| v.as_i64()),
        Some(1 << 40),
        "reservation"
    );
    let pages = map
        .get("pages")
        .and_then(|p| p.as_array())
        .expect("pages array");
    assert!(
        pages
            .iter()
            .any(|p| p.get("kind").and_then(|k| k.as_str()) == Some("rw")),
        "the vm_map'd tail pages appear as rw entries: {pages:?}"
    );
    assert!(map.get("stack").is_some(), "the powerbox stack constants");
    assert!(map.get("heap").is_some(), "the powerbox heap words");
}

/// **Fail-closed surfaces**: `memoryMap` on the tree-walker fails the request; `memoryLimit` off
/// the bytecode+powerbox path fails the launch.
#[test]
fn memory_map_and_limit_fail_closed() {
    let req = |seq: i64, command: &str, args: Json| {
        Json::obj(vec![
            ("seq", Json::i(seq)),
            ("type", Json::s("request")),
            ("command", Json::s(command)),
            ("arguments", args),
        ])
    };
    let response = |msgs: &[Json]| -> Json {
        msgs.iter()
            .find(|m| m.get("type").and_then(|t| t.as_str()) == Some("response"))
            .expect("a response")
            .clone()
    };
    // Tree-walker session: memoryMap unsupported.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(MAP_TWICE)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
        ]),
    ));
    let out = s.handle(&req(3, "memoryMap", Json::obj(vec![])));
    assert_eq!(response(&out).get("success"), Some(&Json::Bool(false)));
    // memoryLimit without the powerbox: launch fails.
    let mut s = DapServer::new();
    s.handle(&req(1, "initialize", Json::obj(vec![])));
    let out = s.handle(&req(
        2,
        "launch",
        Json::obj(vec![
            ("programText", Json::s(MAP_TWICE)),
            ("function", Json::i(0)),
            ("args", Json::Arr(vec![])),
            ("engine", Json::s("bytecode")),
            ("memoryLimit", Json::i(65536)),
        ]),
    ));
    assert_eq!(
        response(&out).get("success"),
        Some(&Json::Bool(false)),
        "memoryLimit without the powerbox fails the launch"
    );
}
