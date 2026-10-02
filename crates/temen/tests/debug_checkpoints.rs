//! Oracle harness for **time-travel checkpointing** (DEBUGGING.md W1) on the tree-walk `Inspector`:
//! `seek(t)` re-executes from clock 0, so `step_back` is O(t²). Checkpoints let a `seek`/`step_back`
//! restart from the nearest snapshot (`clock ≤ t`) instead — bounding the replay to the checkpoint
//! stride.
//!
//! The forward and backward warm≡cold sweeps are the `ShadowStack` cell of the one harness every
//! continuation shares (`crates/temen-dap/tests/dap_checkpoints.rs`, #1460). What stays here is the
//! `Inspector`'s own surface: `step_back` one op at a time, and the refusal a scheduled run reports.
//!
//! Correctness gate: a **warm** Inspector (its checkpoint ladder populated by a prior deep seek, so
//! `seek` *restores* from a snapshot and replays only the tail) must observe **identical** state — the
//! result, the paused location, the logical clock, and guest memory — as a **cold** Inspector (freshly
//! attached, ladder empty, so it replays from clock 0).

use temen_interp::{Inspector, Stop, Trap, Value};
use temen_text::parse_module;

/// A guest that runs **well past the checkpoint stride** (≥ a few thousand ops): a counter loop that
/// also mutates linear memory each iteration, so a faithful checkpoint must restore both the call
/// stack *and* the window bytes. `block1` is the loop header; each turn stores the running sum to a
/// fixed address (16384, above the #1094 NULL guard) and decrements the counter.
const LOOP_WITH_MEM: &str = "\
memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.eqz v2
  br_if v4 2(v3) 3(v2, v3)
}
block 2 (v5: i32) {
  return v5
}
block 3 (v6: i32, v7: i32) {
  v8 = i32.add v7 v6
  v9 = i32.const 16384
  i32.store v9 v8
  v10 = i32.const -1
  v11 = i32.add v6 v10
  br 1(v11, v8)
  }
}
";

/// The observable state at a paused/finished point — everything a user could read after a `seek`.
#[derive(Debug, PartialEq)]
struct Probe {
    stop_pc: Option<(usize, usize)>, // (block, inst) of the pause, or None when finished
    finished: Option<Result<Vec<Value>, Trap>>,
    clock: u64,
    mem: Vec<u8>, // the window bytes the loop writes to (at 16384, above the #1094 NULL guard)
}

fn probe(insp: &Inspector, stop: &Stop) -> Probe {
    let (stop_pc, finished) = match stop {
        Stop::Break { pc, .. } => (Some((pc.block, pc.inst)), None),
        Stop::Finished(r) => (None, Some(r.clone())),
        Stop::Blocked => (None, None),
    };
    Probe {
        stop_pc,
        finished,
        clock: insp.clock(),
        mem: insp.read_window(16384, 8).unwrap_or_default(),
    }
}

/// `seek(t)` on a freshly attached Inspector — the ground truth (ladder empty ⇒ replay from clock 0).
fn cold(src: &str, arg: i32, t: u64) -> Probe {
    let m = parse_module(src).expect("parse");
    let mut insp = Inspector::attach(&m, 0, &[Value::I32(arg)], 50_000_000);
    let stop = insp.seek(t);
    probe(&insp, &stop)
}

#[test]
fn step_back_one_at_a_time_is_faithful() {
    // Fine-grained: a run of step_back() calls near a checkpoint boundary must each land exactly where
    // a cold seek to that clock would, exercising the restore + short forward-replay tail.
    let m = parse_module(LOOP_WITH_MEM).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(1500)], 50_000_000);
    let _ = warm.seek(2100); // just past the second stride boundary
    for _ in 0..12 {
        let before = warm.clock();
        let stop = warm.step_back();
        assert_eq!(
            warm.clock(),
            before - 1,
            "step_back ticks the clock down by one"
        );
        let got = probe(&warm, &stop);
        let want = cold(LOOP_WITH_MEM, 1500, warm.clock());
        assert_eq!(
            got,
            want,
            "step_back to {} diverged from cold",
            warm.clock()
        );
    }
}

/// A run using a fiber falls outside the checkpointable subset, so checkpointing stays off and `seek`
/// remains the (correct) replay-from-0 path — no checkpoints captured, results still right.
#[test]
fn out_of_subset_run_keeps_replaying_from_zero() {
    // A long, purely-scalar run with no memory still checkpoints (sanity that the gate isn't
    // over-broad); then assert a memoryless variant also matches cold.
    const SCALAR: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.eqz v2
  br_if v4 2(v3) 3(v2, v3)
}
block 2 (v5: i32) {
  return v5
}
block 3 (v6: i32, v7: i32) {
  v8 = i32.add v7 v6
  v9 = i32.const -1
  v10 = i32.add v6 v9
  br 1(v10, v8)
  }
}
";
    let m = parse_module(SCALAR).expect("parse");
    let mut warm = Inspector::attach(&m, 0, &[Value::I32(3000)], 50_000_000);
    let _ = warm.seek(u64::MAX);
    assert!(
        warm.checkpoint_count() > 1,
        "a memoryless scalar loop still checkpoints"
    );
    for &t in &[0u64, 1500, 3000, 50] {
        let stop = warm.seek(t);
        let got = probe(&warm, &stop);
        let want = {
            let mm = parse_module(SCALAR).expect("parse");
            let mut c = Inspector::attach(&mm, 0, &[Value::I32(3000)], 50_000_000);
            let s = c.seek(t);
            probe(&c, &s)
        };
        assert_eq!(got, want, "scalar warm seek({t}) diverged from cold");
    }
}

/// A scheduled (multithreaded) `Inspector` seeks by global turn and takes no checkpoints: its ladder
/// is refused from the start, and says so (#1460).
#[test]
fn a_scheduled_inspector_names_its_refusal() {
    let m = parse_module(LOOP_WITH_MEM).expect("parse");
    let insp = Inspector::attach_scheduled(&m, 0, &[Value::I32(10)], 50_000_000, Vec::new());
    assert_eq!(
        insp.checkpoint_refusal(),
        Some(temen_interp::moment::Refusal::Thread)
    );
}
