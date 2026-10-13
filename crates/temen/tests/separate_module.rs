//! §14 **`Module` capability** (iface 8) op 0, `resolve_export(name_ptr, name_len)`: the parent
//! looks a name up in a granted module's export table, so it can spawn the child at a named entry
//! (that spawn is `bytecode_separate_module.rs`'s `module_child_addressed_by_export_name_*`). The
//! name is untrusted guest memory, so the op is fail-closed: an unknown name is `-EINVAL`, a buffer
//! past the window `-EFAULT`.

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

/// A child ("plugin") with **two** exported entries, to prove name→funcidx resolution selects the
/// right one (not just func 0): `"alpha"` (func 0) returns `byte + 1000`, `"beta"` (func 1) returns
/// `byte + 2000`. Both are `(i64) -> (i64)` (the starter-cap convention).
fn named_child_src() -> &'static str {
    "memory 16
data 16484 \"VM\"
export 0 func \"alpha\" 0
export 1 func \"beta\" 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 2000
  v5 = i64.add v3 v4
  return v5
  }
}
"
}

/// Run `parent_src` with `(instantiator over the whole window, module handle for [`named_child_src`])`,
/// over a 128 KiB window seeded with the test pattern; return the result and final window.
fn run_with_named_child(parent_src: &str) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let parent = parse_module(parent_src).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(named_child_src()).expect("parse child");
    verify_module(&child).expect("verify child");

    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mh = host.grant_module(&child);
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ih), Value::I32(mh)],
        &mut fuel,
        &init,
        0,
        &mut host,
    )
}

#[test]
fn module_resolve_export_is_fail_closed() {
    // An unknown name → -EINVAL (-22); the parent returns the resolve result directly. Pins the
    // fail-closed contract of the new untrusted-name surface (a bad name must never trap or default).
    let parent = "memory 17
data 16384 \"nope\"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 16384
  v3 = i64.const 4
  v4 = call.cap 8 0 (i64, i64) -> (i64) v1 (v2, v3)
  return v4
  }
}
";
    let (res, _mem) = run_with_named_child(parent);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(-22)],
        "an unknown export name must resolve to -EINVAL"
    );
}

#[test]
fn module_resolve_export_out_of_bounds_name_is_efault() {
    // A name buffer past the window → -EFAULT (-14): the borrow is bounds-checked like every other
    // guest-buffer cap op (Stream/Jit.compile), never an over-read.
    let parent = "memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 4294967296
  v3 = i64.const 4
  v4 = call.cap 8 0 (i64, i64) -> (i64) v1 (v2, v3)
  return v4
  }
}
";
    let (res, _mem) = run_with_named_child(parent);
    assert_eq!(
        res.expect("run ok"),
        vec![Value::I64(-14)],
        "an out-of-window name buffer must resolve to -EFAULT"
    );
}
