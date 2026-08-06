//! Equality harness for the bytecode engine's **§14 executor-child seam** (INTERP_PERF.md Slice
//! 1c-5g): `Instantiator.instantiate` / `Instantiator.join`. Unlike a coroutine (driven inline),
//! an instantiated child runs on the cooperative scheduler — confined to a power-of-two sub-window
//! of the holder's range (a `nested_view` over the shared backing), with an attenuated powerbox (an
//! `Instantiator` + an `AddressSpace`, each over its own window) and a `quota` fuel sub-budget — and
//! is joined through the shared §12 thread machinery.
//!
//! Adapted from `crates/svm/tests/instantiator.rs`. Each case is checked **bit-identical** to the
//! tree-walker `run_with_host`; `.expect(Some)` gates that the bytecode engine drove the module
//! (didn't fall back). The host grants the `Instantiator` capability (iface 6); the handle reaches
//! the guest as func 0's argument. `instantiate` is `cap.call 6 0`, `join` is `cap.call 6 1`.

use svm_interp::{bytecode, run_with_host, Host, Value};
use svm_text::parse_module;

/// Run `src`'s entry on both engines with an `Instantiator` granted over `[0, 1<<win_log2)`, and
/// assert the results are identical and equal to `want`.
fn check(src: &str, want: Result<Vec<Value>, ()>) {
    let m = parse_module(src).expect("parse");

    let mut h_tw = Host::new();
    let inst_tw = h_tw.grant_instantiator(0, 128 << 10);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);

    let mut h_bc = Host::new();
    let inst_bc = h_bc.grant_instantiator(0, 128 << 10);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine must support instantiate/join (Slice 1c-5g)");

    assert_eq!(tw, bc, "instantiate: tree-walker != bytecode\n{src}");
    match want {
        Ok(vals) => assert_eq!(bc, Ok(vals), "instantiate result\n{src}"),
        Err(()) => assert!(bc.is_err(), "expected a trap, got {bc:?}\n{src}"),
    }
}

/// Parent (func 0) instantiates the child (func 1) in a 4 KiB window at 64 KiB, joins it, then reads
/// back the marker the child wrote into the **shared** backing — proving the child ran confined on
/// the executor and its writes are visible to the parent (the §14 shared data plane). The child
/// writes 123 at its own offset 7 (→ backing 64 KiB + 7) and returns 42; the parent returns
/// `42 * 1000 + 123 = 42123`.
const SHARED_MEM: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  v7 = i64.const 65543
  v8 = i32.load8_u v7
  v9 = i64.extend_i32_u v8
  v10 = i64.const 1000
  v11 = i64.mul v6 v10
  v12 = i64.add v11 v9
  return v12
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i32.const 123
  i32.store8 v1 v2
  v3 = i64.const 42
  return v3
  }
}
"#;

#[test]
fn instantiate_join_shares_backing() {
    check(SHARED_MEM, Ok(vec![Value::I64(42123)]));
}

/// Depth-2 VM-in-VM (from `instantiator.rs`): the child, handed an `Instantiator` over *its* window,
/// itself instantiates a grandchild — confinement composes. The grandchild returns 77, propagated up
/// through two joins.
const DEPTH_TWO: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 65536
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 1216
  i64.store q1a0 q1v0
  q1a1 = i64.const 1224
  i64.store q1a1 q1v1
  q1a2 = i64.const 1232
  i64.store q1a2 q1v2
  q1a3 = i64.const 1240
  i64.store q1a3 q1v3
  q1a4 = i64.const 1248
  i64.store q1a4 q1v4
  q1a5 = i64.const 1256
  i64.store q1a5 q1v4
  q1a6 = i64.const 1264
  i64.store q1a6 q1v4
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.wrap_i64 v0
  v2 = i64.const 0
  v3 = i32.const 171
  i32.store8 v2 v3
  ; spawn via record (op 17): entry=2 off=2048 sl=10 quota=0
  q2v0 = i64.const 8589934592
  q2v1 = i64.const 2048
  q2v2 = i64.const -4294967286
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2a0 = i64.const 1280
  i64.store q2a0 q2v0
  q2a1 = i64.const 1288
  i64.store q2a1 q2v1
  q2a2 = i64.const 1296
  i64.store q2a2 q2v2
  q2a3 = i64.const 1304
  i64.store q2a3 q2v3
  q2a4 = i64.const 1312
  i64.store q2a4 q2v4
  q2a5 = i64.const 1320
  i64.store q2a5 q2v4
  q2a6 = i64.const 1328
  i64.store q2a6 q2v4
  v8 = cap.call 6 17 (i64) -> (i32) v1 (q2a0)
  v9 = cap.call 6 1 (i32) -> (i64) v1 (v8)
  return v9
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i32.const 200
  i32.store8 v1 v2
  v3 = i64.const 77
  return v3
  }
}
"#;

#[test]
fn nesting_composes_to_depth_two() {
    check(DEPTH_TWO, Ok(vec![Value::I64(77)]));
}

/// A two-arg child receives its starter caps `(Instantiator, AddressSpace)`. It uses the
/// `AddressSpace` (iface 5, op 1 = `unmap`) to decommit the first 16 KiB of its **own** 64 KiB
/// window — a confined sub-window page op — and returns the unmap result (0). The parent returns it.
const ADDRESS_SPACE: &str = r#"memory 18
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q3v0 = i64.const 4294967296
  q3v1 = i64.const 65536
  q3v2 = i64.const -4294967280
  q3v3 = i64.const 4294967295
  q3v4 = i64.const 0
  q3a0 = i64.const 1344
  i64.store q3a0 q3v0
  q3a1 = i64.const 1352
  i64.store q3a1 q3v1
  q3a2 = i64.const 1360
  i64.store q3a2 q3v2
  q3a3 = i64.const 1368
  i64.store q3a3 q3v3
  q3a4 = i64.const 1376
  i64.store q3a4 q3v4
  q3a5 = i64.const 1384
  i64.store q3a5 q3v4
  q3a6 = i64.const 1392
  i64.store q3a6 q3v4
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q3a0)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i32.wrap_i64 v1
  v3 = i64.const 0
  v4 = i64.const 16384
  v5 = cap.call 5 1 (i64, i64) -> (i64) v2 (v3, v4)
  return v5
  }
}
"#;

#[test]
fn two_arg_child_manages_its_own_pages() {
    check(ADDRESS_SPACE, Ok(vec![Value::I64(0)]));
}

/// An out-of-range carve (a 4 KiB child at offset 128 KiB doesn't fit the 128 KiB holder) returns
/// `-EINVAL` (-22); the parent returns it without joining.
const BAD_CARVE: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=131072 sl=12 quota=0
  q4v0 = i64.const 4294967296
  q4v1 = i64.const 131072
  q4v2 = i64.const -4294967284
  q4v3 = i64.const 4294967295
  q4v4 = i64.const 0
  q4a0 = i64.const 1408
  i64.store q4a0 q4v0
  q4a1 = i64.const 1416
  i64.store q4a1 q4v1
  q4a2 = i64.const 1424
  i64.store q4a2 q4v2
  q4a3 = i64.const 1432
  i64.store q4a3 q4v3
  q4a4 = i64.const 1440
  i64.store q4a4 q4v4
  q4a5 = i64.const 1448
  i64.store q4a5 q4v4
  q4a6 = i64.const 1456
  i64.store q4a6 q4v4
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q4a0)
  v6 = i64.extend_i32_s v5
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  return v1
  }
}
"#;

#[test]
fn out_of_range_carve_rejected() {
    check(BAD_CARVE, Ok(vec![Value::I64(-22)]));
}

/// A child trap (`unreachable`) must propagate through `join` as the parent's trap — identically on
/// both engines.
const CHILD_TRAP: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=0 sl=12 quota=0
  q5v0 = i64.const 4294967296
  q5v1 = i64.const 0
  q5v2 = i64.const -4294967284
  q5v3 = i64.const 4294967295
  q5a0 = i64.const 4416
  i64.store q5a0 q5v0
  q5a1 = i64.const 4424
  i64.store q5a1 q5v1
  q5a2 = i64.const 4432
  i64.store q5a2 q5v2
  q5a3 = i64.const 4440
  i64.store q5a3 q5v3
  q5a4 = i64.const 4448
  i64.store q5a4 q5v1
  q5a5 = i64.const 4456
  i64.store q5a5 q5v1
  q5a6 = i64.const 4464
  i64.store q5a6 q5v1
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q5a0)
  v6 = cap.call 6 1 (i32) -> (i64) v0 (v5)
  return v6
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  unreachable
  }
}
"#;

#[test]
fn child_trap_propagates_through_join() {
    check(CHILD_TRAP, Err(()));
}
