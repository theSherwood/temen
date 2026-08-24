//! R8: freeze/thaw across a `call_indirect` to a may-suspend target (DURABILITY.md §6/§11).
//!
//! A function-pointer call whose (signature-tainted) target can suspend is now a real suspend
//! point — the indirect twin of the propagated direct `Call` (`chain.rs`). On freeze the caller's
//! frame unwinds like any other; on thaw it reloads its pre-call live set **plus the table index**
//! and **re-issues the `call_indirect`**, leaving the state `REWINDING` so the resolved callee
//! rewinds in turn. Because the analysis instruments *every* function of the tainted signature, the
//! reloaded index can only re-select a `REWINDING`-aware callee. Before this, an indirect call to a
//! suspending target was silently under-instrumented (fail-open); the round-trip below would diverge.
//!
//! bash dispatches its builtins through function-pointer tables, so this is the shape `fork` needs.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, read_thaw_state, transform_module, write_state,
    STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18; // 256 KiB window — ample room for a handful of stacked frames
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

/// Confirm the transform actually recognized a suspend point in func 0 (the indirect caller) —
/// i.e. it was instrumented, not silently passed through. An instrumented function grows blocks
/// (segments + unwind/arm dispatch); the untransformed one has exactly its original blocks.
fn assert_instrumented(inst: &Module, func: usize, orig_blocks: usize) {
    assert!(
        inst.funcs[func].blocks.len() > orig_blocks,
        "func {func} must be instrumented (had {orig_blocks} blocks, now {})",
        inst.funcs[func].blocks.len()
    );
}

fn run(
    inst: &Module,
    clock_ns: i64,
    window: &[u8],
) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    )
}

/// Baseline (clock 42) vs. freeze→thaw on a fresh host (clock 0) — the whole chain unwinds, then a
/// thaw reproduces the uninterrupted result. Returns the agreed result for the caller to pin.
fn assert_roundtrips(inst: &Module) -> Vec<Value> {
    let (baseline, _) = run(inst, 42, &init_durable_window(WINDOW));
    let baseline = baseline.expect("baseline runs to completion");

    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (frozen, snapshot) = run(inst, 42, &win);
    assert!(frozen.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(
        read_state(&snapshot),
        STATE_UNWINDING,
        "artifact is still UNWINDING (the whole chain unwound, none completed)"
    );

    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, final_win) = run(inst, 0, &win);
    assert_eq!(
        thawed,
        Ok(baseline.clone()),
        "thaw equals the uninterrupted run"
    );
    assert_eq!(
        read_thaw_state(&final_win, 0),
        STATE_NORMAL,
        "the deepest frame flipped the state back to NORMAL exactly once"
    );
    baseline
}

// A →(call_indirect slot 1)→ B(leaf). Natural table: slot i = func i, so `i32.const 1` is the
// funcref for func 1. A adds 1000 to B's result; B adds 100 to the clock. Baseline: 1000+(100+42).
const TWO_LEVEL_INDIRECT: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = call_indirect (i32) -> (i64) v1 (v0)
  v3 = i64.const 1000
  v4 = i64.add v2 v3
  return v4
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn two_level_indirect_round_trips() {
    let inst = instrument(TWO_LEVEL_INDIRECT);
    assert_instrumented(&inst, 0, 1); // the indirect caller is a real suspend site now
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(1142)],
        "1000 + (100 + 42)"
    );
}

// A value live *across* the indirect call: `v1` is computed before the `call_indirect` and used
// after, so it must be spilled/reloaded alongside the table index — the re-issue must clobber
// neither. Baseline: leaf = 42+5 = 47; A = 47 + (7+7) = 61.
const LIVE_ACROSS_INDIRECT: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  v2 = i32.const 1
  v3 = call_indirect (i32) -> (i64) v2 (v0)
  v4 = i64.add v1 v1
  v5 = i64.add v3 v4
  return v5
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 5
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn live_value_across_indirect_call_survives() {
    let inst = instrument(LIVE_ACROSS_INDIRECT);
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(61)],
        "(42+5) + (7+7)"
    );
}

// A mixed chain: A →(direct)→ B →(call_indirect slot 2)→ C(leaf), each adding a distinct constant.
// Exercises an indirect frame stacked between two direct frames. Baseline: 42 + 1 + 20 + 300 = 363.
const MIXED_CHAIN: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i64.const 300
  v3 = i64.add v1 v2
  return v3
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 2
  v2 = call_indirect (i32) -> (i64) v1 (v0)
  v3 = i64.const 20
  v4 = i64.add v2 v3
  return v4
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 1
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

#[test]
fn mixed_direct_indirect_chain_round_trips() {
    let inst = instrument(MIXED_CHAIN);
    assert_instrumented(&inst, 1, 1); // the middle (indirect) frame is instrumented
    assert_eq!(
        assert_roundtrips(&inst),
        vec![Value::I64(363)],
        "42 + 1 + 20 + 300"
    );
}

// An indirect **tail** call to a may-suspend target must be rejected (no poll site to unwind at) —
// fail closed, matching direct `return_call`. Confirms R8 didn't reopen a hole via tail dispatch.
const INDIRECT_TAIL: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 1
  return_call_indirect (i32) -> (i64) v1 (v0)
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = cap.call 2 0 (i32) -> (i64) v0 (v1)
  return v2
  }
}
"#;

#[test]
fn indirect_tail_call_to_may_suspend_is_rejected() {
    let mut m = temen_text::parse_module(INDIRECT_TAIL).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    assert!(
        transform_module(&m).is_err(),
        "an indirect tail call to a suspending target has no poll site — must fail closed"
    );
}
