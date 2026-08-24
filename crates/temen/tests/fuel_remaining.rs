//! CALLS.md §10.6 / increment 5 — `fuel.remaining` (self-namespace op 13): a domain reads its own
//! remaining fuel. Observable, so it must return the **identical** value on every backend — fuel is
//! a checked, safepoint-anchored cross-engine quantity (INVARIANTS.md 9), so the readout matches by
//! construction. This is that differential pin: the tree-walker (which services op 13 inline) and the
//! JIT (which lowers it to an inline load of the same host-owned fuel cell) must agree, with counted
//! fuel armed on both.

use temen_interp::{run, Value};
use temen_jit::{compile_and_run_with_host_fuel, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Read `fuel.remaining` once at entry and return it. `4294967295` = `CAP_SELF_TYPE_ID`, op `13`.
const FUEL_READBACK: &str = "\
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vr = cap.call 4294967295 13 () -> (i64) vz ()
  return vr
  }
}
";

/// Read fuel, spin a finite back-edge loop (charging fuel), then read again and return the
/// difference — the read-before-minus-read-after **call-metering** idiom (§10.6). The measured cost
/// must be identical on both engines.
const FUEL_METER: &str = "\
func (i32) -> (i64) {
block 0 (v0: i32) {
  vz = i32.const 0
  vbefore = cap.call 4294967295 13 () -> (i64) vz ()
  br 1(v0, vbefore)
}
block 1 (v1: i32, vb: i64) {
  vm1 = i32.const -1
  v2 = i32.add v1 vm1
  br_if v2 1(v2, vb) 2(vb)
}
block 2 (vb2: i64) {
  vzz = i32.const 0
  vafter = cap.call 4294967295 13 () -> (i64) vzz ()
  vcost = i64.sub vb2 vafter
  return vcost
  }
}
";

fn jit_i64(src: &str, args: &[i64], budget: u64) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut cell: u64 = budget;
    let outcome = compile_and_run_with_host_fuel(
        &m,
        0,
        args,
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        &mut cell as *mut u64,
    )
    .expect("jit compiles");
    match outcome {
        JitOutcome::Returned(v) => v.first().copied().unwrap_or(0),
        other => panic!("jit did not return cleanly: {other:?}"),
    }
}

fn interp_i64(src: &str, args: &[Value], budget: u64) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = budget;
    match run(&m, 0, args, &mut fuel).expect("interp ok").as_slice() {
        [Value::I64(v)] => *v,
        other => panic!("unexpected interp result shape: {other:?}"),
    }
}

#[test]
fn fuel_remaining_readback_matches_interp_and_jit() {
    const BUDGET: u64 = 1_000_000;
    let interp = interp_i64(FUEL_READBACK, &[], BUDGET);
    let jit = jit_i64(FUEL_READBACK, &[], BUDGET);
    assert_eq!(
        interp,
        (BUDGET - 1) as i64,
        "fuel.remaining at entry = budget minus the top-level-entry charge"
    );
    assert_eq!(
        jit, interp,
        "fuel.remaining must read identically on interp and JIT"
    );
}

#[test]
fn fuel_remaining_metering_matches_interp_and_jit() {
    const BUDGET: u64 = 10_000_000;
    let interp = interp_i64(FUEL_METER, &[Value::I32(1000)], BUDGET);
    let jit = jit_i64(FUEL_METER, &[1000], BUDGET);
    assert!(interp > 0, "the loop between the two reads charges fuel");
    assert_eq!(
        jit, interp,
        "read-before minus read-after must meter the same cost on both engines"
    );
}
