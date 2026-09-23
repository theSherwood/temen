//! #1671 — a freeze that cannot complete is **declined** at its trigger, before anything unwinds: the
//! freeze word goes back to `NORMAL`, the run finishes exactly as if it had never been armed, and the
//! embedder reads why from [`Host::take_freeze_declined`]. Before this, each shape below unwound the
//! whole run and then refused at the join — `Trap::ThreadFault`, the domain gone (INVARIANTS #5).
//!
//! Each shape is also a gap filed to be closed (#1703); when one closes, its test here becomes a
//! freeze that completes, and moves to that issue's tests.

use temen_durable::{arm_freeze_after, init_durable_window, read_state, transform_module};
use temen_interp::{
    run_capture_reserved_with_host, DeclineCause, FreezeDeclined, Host, Trap, Value, STATE_NORMAL,
};
use temen_ir::durable_abi::ShadowArena;

const ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};
const SIZE_LOG2: u8 = 18;

fn instrument(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// Run `parent` durable over a fresh window — armed to freeze at the second fiber resume when `arm`
/// — and return its result, its final freeze word, and what its powerbox says about a decline.
fn run(
    parent: &temen_ir::Module,
    arm: bool,
) -> (Result<Vec<Value>, Trap>, i32, Option<FreezeDeclined>) {
    let mut host = Host::new();
    host.set_durable(true);
    let ih = host.grant_instantiator(0, 1 << SIZE_LOG2);
    let mut win = init_durable_window(1 << SIZE_LOG2, ARENA);
    if arm {
        arm_freeze_after(&mut win, 2);
    }
    let mut fuel = 50_000_000u64;
    let (r, snap) = run_capture_reserved_with_host(
        parent,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    (r, read_state(&snap), host.take_freeze_declined())
}

/// The fiber both parents drive: suspends once, then returns 5 — two resumes, so `arm = 2` fires the
/// trigger at the second.
const FIBER: &str = "func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = suspend v2
  v4 = i64.const 5
  return v4
  }
}
";

/// A `0..100` loop (4950) — a §14 child that is still running at the trigger.
const LOOP: &str = "func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  v3 = i64.const 0
  br 1(v2, v3)
}
block 1 (v4: i64, v5: i64) {
  v6 = i64.const 100
  v7 = i64.lt_s v4 v6
  br_if v7 2(v4, v5) 3(v5)
}
block 2 (v8: i64, v9: i64) {
  v10 = i64.add v9 v8
  v11 = i64.const 1
  v12 = i64.add v8 v11
  br 1(v12, v10)
}
block 3 (v13: i64) {
  return v13
  }
}
";

/// Child B (slot 0) traps; child A (slot 1) loops. The parent joins A first — the single durable
/// worker runs B to its trap meanwhile — then drives the fiber (the trigger lands with B completed,
/// trapped and unjoined), then joins B, which propagates B's trap.
fn trapped_child_parent() -> temen_ir::Module {
    instrument(&format!(
        "memory 18 shadow 16448 65536
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  v1 = i64.const 2
  v2 = i64.const 196608
  v3 = i64.const 16
  v4 = i64.const 0
  v5 = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = i64.const 1
  v7 = i64.const 131072
  v8 = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (v6, v7, v3, v4)
  v9 = call.cap 6 1 (i32) -> (i64) v0 (v8)
  v10 = ref.func 3
  v11 = i64.const 4096
  v12 = cont.new v10 v11
  v13 = i64.const 0
  v14, v15 = cont.resume v12 v13
  v16, v17 = cont.resume v12 v15
  v18 = call.cap 6 1 (i32) -> (i64) v0 (v5)
  v19 = i64.add v9 v18
  v20 = i64.add v19 v17
  return v20
  }}
}}
{LOOP}func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  unreachable
  }}
}}
{FIBER}"
    ))
}

#[test]
fn a_child_that_completed_with_a_trap_declines_the_freeze_and_the_run_goes_on() {
    let parent = trapped_child_parent();
    let (base, base_state, base_declined) = run(&parent, false);
    assert!(
        base.is_err(),
        "uninterrupted, joining B propagates its trap: {base:?}"
    );
    assert_eq!((base_state, base_declined), (STATE_NORMAL, None));

    let (r, state, declined) = run(&parent, true);
    assert_eq!(
        r, base,
        "declined: the run finishes exactly as it would have unarmed"
    );
    assert_eq!(
        state, STATE_NORMAL,
        "nothing unwound — the freeze word went back to NORMAL"
    );
    assert_eq!(
        declined,
        Some(FreezeDeclined {
            cause: DeclineCause::ChildTrapped,
            task: 0,
            slot: Some(0),
        }),
        "the root declined over its child in slot 0"
    );
}

/// A §14 nested child (A, the loop) beside a `thread.spawn` thread (T → 7). The thaw's two seedings
/// would contend for one join table (#1673), so the cut declines; unarmed, the total is 4950 + 7 + 5.
#[test]
fn a_nested_child_beside_a_thread_declines_the_freeze_and_the_run_goes_on() {
    let parent = instrument(&format!(
        "memory 18 shadow 16448 65536
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  v1 = i64.const 1
  v2 = i64.const 131072
  v3 = i64.const 17
  v4 = i64.const 0
  v5 = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)
  v6 = thread.spawn 3 v4 v4
  v7 = ref.func 2
  v8 = i64.const 4096
  v9 = cont.new v7 v8
  v10 = i64.const 0
  v11, v12 = cont.resume v9 v10
  v13, v14 = cont.resume v9 v12
  v15 = call.cap 6 1 (i32) -> (i64) v0 (v5)
  v16 = thread.join v6
  v17 = i64.add v15 v16
  v18 = i64.add v17 v14
  return v18
  }}
}}
{LOOP}{FIBER}func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = i64.const 7
  return v2
  }}
}}
"
    ));
    let (base, _, _) = run(&parent, false);
    assert_eq!(
        base,
        Ok(vec![Value::I64(4950 + 7 + 5)]),
        "uninterrupted total"
    );

    let (r, state, declined) = run(&parent, true);
    assert_eq!(
        r, base,
        "declined: the run finishes exactly as it would have unarmed"
    );
    assert_eq!(state, STATE_NORMAL);
    assert_eq!(
        declined,
        Some(FreezeDeclined {
            cause: DeclineCause::NestedWithThread,
            task: 0,
            slot: None,
        })
    );
}
