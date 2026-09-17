//! **`step_back` served by the undo journal** (#1556), against the replay path as the oracle.
//!
//! Reverse debugging used to be one mechanism: rebuild the run and re-drive it to the target, bounded
//! by the checkpoint ladder. The journal adds a second, cheaper one for targets whose history it
//! holds — put the window back from its pre-images and the continuation back from the nearest segment
//! boundary, in place, without rebuilding anything.
//!
//! Two things have to be true for that to be worth having, and they are separate claims:
//!
//! 1. **It agrees with replay.** `dap_checkpoints.rs` is the warm≡cold oracle for the whole backend
//!    and already covers this — it passes with undo serving `step_back`. What it cannot see is *which*
//!    path served each step.
//! 2. **It is actually used.** A fallback that silently never fires looks exactly like one that always
//!    does. `backward_counts()` distinguishes them, and these tests assert on it — the same lesson as
//!    a fuzz target that never reaches the code it guards.

use temen_dap::{BytecodeBackend, Debuggee};
use temen_interp::Value;
use temen_text::parse_module;

/// A counter loop that stores its running sum to the window each iteration, so a faithful backward
/// step has to put back both the call stack and the window bytes. (Same shape as
/// `dap_checkpoints.rs::LOOP_WITH_MEM`, deliberately — it is the fixture the replay oracle is pinned
/// on, so using it keeps the two comparable.)
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
}";

fn backend(arg: i32) -> BytecodeBackend {
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    BytecodeBackend::new(
        m,
        0,
        &[Value::I32(arg)],
        u64::MAX,
        false,
        Vec::new(),
        false,
        None,
        None,
    )
    .expect("the bytecode engine accepts the single-vCPU loop")
}

/// What a backward step lands on, in full: turn, logical clock, and the window word the loop writes.
fn observe(b: &mut BytecodeBackend) -> (u64, u64, Vec<u8>) {
    (
        b.turn(),
        b.clock(),
        b.read_window(16384, 4).unwrap_or_default(),
    )
}

/// **The wiring is live.** Stepping back repeatedly from deep in a run is served by undo, not by the
/// replay fallback — and the sequence of positions is the same one replay produces.
#[test]
fn step_back_is_served_by_undo_and_agrees_with_replay() {
    // Undo path: one backend, stepped forward then walked back.
    let mut hot = backend(400);
    for _ in 0..3000 {
        hot.step();
    }
    let start = hot.turn();
    let mut undone = Vec::new();
    for _ in 0..40 {
        hot.step_back();
        undone.push(observe(&mut hot));
    }
    let (undo_steps, replay_steps) = hot.backward_counts();

    // Replay path: the same walk on a backend with the journal off, so every step is a `seek`.
    let mut cold = backend(400);
    cold.set_journaling(false);
    for _ in 0..3000 {
        cold.step();
    }
    assert_eq!(cold.turn(), start, "both runs start from the same turn");
    let mut replayed = Vec::new();
    for _ in 0..40 {
        cold.step_back();
        replayed.push(observe(&mut cold));
    }

    assert_eq!(
        undone, replayed,
        "an undo-served step_back must land exactly where the replay path lands"
    );
    assert_eq!(
        undo_steps, 40,
        "every backward step should have been served by the journal, not by replay \
         (undo {undo_steps}, replay {replay_steps})"
    );
    assert_eq!(
        cold.backward_counts(),
        (0, 40),
        "the control used replay only"
    );
}

/// **It declines rather than lying.** With the journal disarmed the backend still steps back — through
/// the replay path — so the fallback is a real path and not a hypothetical one.
#[test]
fn a_disarmed_journal_falls_back_to_replay() {
    let mut b = backend(200);
    b.set_journaling(false);
    for _ in 0..500 {
        b.step();
    }
    let before = observe(&mut b);
    b.step_back();
    let after = observe(&mut b);

    assert_ne!(before.0, after.0, "the step went somewhere");
    assert_eq!(
        b.backward_counts(),
        (0, 1),
        "with no journal every backward step is a replay"
    );
}

/// **A backward walk past the journal's reach falls through mid-sequence**, and the two paths still
/// agree across the seam.
///
/// The budget drops the **oldest** history, so a walk has to be long enough to reach past what
/// remains before any target falls through — a short walk back from the end stays inside the retained
/// tail and is served entirely by undo. (That is worth stating because it is the first thing this test
/// got wrong: 60 steps back from turn 3000 under a 512-byte budget were all undo, and the seam this
/// test exists for was never exercised.)
#[test]
fn a_bounded_journal_hands_the_far_targets_back_to_replay() {
    let mut b = backend(400);
    b.set_journal_policy(temen_interp::journal::JournalPolicy {
        // No coalescing: this loop stores to one *fixed* address, so level 2 would collapse the whole
        // history to a single 4-byte entry and the budget below would never bite. (It didn't, the
        // first time this test was written — the drop path it exists to cover was never reached.)
        fine_turns: u64::MAX,
        byte_budget: 512, // ~128 of this loop's 4-byte stores
        ..Default::default()
    });
    for _ in 0..3000 {
        b.step();
    }
    const BACK: usize = 900; // far enough to walk off the end of what a 512-byte budget retains
    let mut seen = Vec::new();
    for _ in 0..BACK {
        b.step_back();
        seen.push(observe(&mut b));
    }
    let (undo_steps, replay_steps) = b.backward_counts();
    assert_eq!(undo_steps + replay_steps, BACK);
    assert!(
        undo_steps > 0,
        "the retained tail should still be served by undo (undo {undo_steps}, replay {replay_steps})"
    );
    assert!(
        replay_steps > 0,
        "a budget this small must push some targets onto the replay path \
         (undo {undo_steps}, replay {replay_steps})"
    );

    // And the walk is still correct across the seam — same oracle as the first test.
    let mut cold = backend(400);
    cold.set_journaling(false);
    for _ in 0..3000 {
        cold.step();
    }
    let replayed: Vec<_> = (0..BACK)
        .map(|_| {
            cold.step_back();
            observe(&mut cold)
        })
        .collect();
    assert_eq!(
        seen, replayed,
        "mixing undo and replay across one walk must not change where it lands"
    );
}
