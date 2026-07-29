//! Warm≡cold oracle for the bytecode `DebugRun` time-travel **checkpoint ladder** (DEBUGGING.md W1) —
//! the `BytecodeBackend` port of the tree-walker `crates/svm/tests/debug_checkpoints.rs`.
//!
//! A **warm** backend (its ladder populated by a prior deep `seek`, so `seek` *restores* from the
//! nearest snapshot `clock <= t` and replays only the tail) must observe **identical** state — the
//! logical clock, the call stack, and guest memory — as a **cold** backend (fresh, empty ladder,
//! replaying from clock 0). The cold path is the pre-existing, trusted replay; the warm path exercises
//! snapshot capture + restore. If restore is faithful they agree at every probed time, including
//! across stride boundaries and a full backward sweep. Behavior is a pure optimization: correctness is
//! *defined* by the cold path.

use svm_dap::{BytecodeBackend, Debuggee};
use svm_interp::Value;
use svm_text::parse_module;

/// A counter loop that also stores its running sum to window address 0 each iteration, so a faithful
/// checkpoint must restore both the call stack *and* the window bytes. Run with a large enough arg to
/// cross several `CHECKPOINT_STRIDE` (1024-op) boundaries.
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
  v9 = i32.const 0
  i32.store v9 v8
  v10 = i32.const -1
  v11 = i32.add v6 v10
  br 1(v11, v8)
  }
}";

/// A stable per-`seek` observation: the logical clock, the call stack (each frame's IR pc), and the
/// running-sum window bytes. Identical between a from-0 replay and a checkpoint-restored replay iff
/// restore is faithful.
fn obs(b: &mut BytecodeBackend, t: u64) -> (u64, String, Vec<u8>) {
    b.seek(t);
    let clock = b.clock();
    let stack = b
        .backtrace()
        .iter()
        .map(|f| format!("{}:{}:{}:{}", f.pc.module, f.pc.func, f.pc.block, f.pc.inst))
        .collect::<Vec<_>>()
        .join("|");
    let mem = b.read_window(0, 4).unwrap_or_default();
    (clock, stack, mem)
}

#[test]
fn bytecode_checkpoint_warm_seek_matches_cold_replay_from_zero() {
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    let args = [Value::I32(800)]; // several thousand ops ⇒ several stride boundaries
    let mk = || {
        BytecodeBackend::new(m.clone(), 0, &args, u64::MAX, false, Vec::new())
            .expect("the bytecode engine accepts the single-vCPU loop")
    };

    // Probe times spread across the run — deliberately not stride-aligned, so restores land at a
    // checkpoint strictly below the target and must replay a nonzero tail.
    let probes: Vec<u64> = (0..=6000).step_by(137).collect();

    // Cold baseline: a *fresh* backend per probe (empty ladder ⇒ always a from-0 replay, the trusted
    // path that defines correctness).
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs(&mut b, t)
        })
        .collect();

    // Warm: one backend, a deep seek first to populate the ladder, then the probes reuse it.
    let mut warm = mk();
    warm.seek(6000);
    assert!(
        warm.checkpoint_count() > 0,
        "a deep seek past the stride lays down checkpoints — the ladder is exercised, not dormant",
    );

    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm (checkpoint-restored) seek ≡ cold (replay-from-0) at every forward probe",
    );

    // A full backward sweep restarts each seek from a checkpoint below the target — must also match.
    let warm_back: Vec<_> = probes.iter().rev().map(|&t| obs(&mut warm, t)).collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(
        warm_back, cold_back,
        "warm backward sweep ≡ cold (restore is faithful seeking in either direction)",
    );

    // The loop never leaves the checkpointable subset, so checkpointing stayed on the whole run.
    assert!(
        warm.checkpoint_count() > 0,
        "checkpointing stays on for a pure single-vCPU memory loop",
    );
}

/// Wall-clock evidence that the ladder actually bounds the reverse-replay cost (DEBUGGING.md W1):
/// a backward `step_back` sweep over a long run is dramatically faster warm (restart from the nearest
/// checkpoint, replay ≤ one stride) than cold (rebuild + replay from clock 0 each time — the old
/// behavior). `#[ignore]`d because wall-clock ratios are runner-dependent and must never gate CI;
/// run with `cargo test -p svm-dap --test dap_checkpoints -- --ignored --nocapture` to see the ratio.
#[test]
#[ignore = "timing benchmark; run manually with --ignored --nocapture"]
fn bytecode_checkpoint_reverse_sweep_is_bounded() {
    use std::time::Instant;
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    let args = [Value::I32(8_000)]; // ~64k ops — many strides deep
    let mk = || BytecodeBackend::new(m.clone(), 0, &args, u64::MAX, false, Vec::new()).unwrap();
    let deep = 60_000u64;
    let steps = 60u64;

    // Warm sweep: one backend, step_back repeatedly from deep in the run (restart from the nearest
    // checkpoint, replay ≤ one stride each time).
    let mut warm = mk();
    warm.seek(deep);
    let warm_ckpts = warm.checkpoint_count();
    let t0 = Instant::now();
    for _ in 0..steps {
        warm.step_back();
    }
    let warm_ms = t0.elapsed().as_secs_f64() * 1e3;

    // Cold sweep: the pre-ladder behavior — each step_back rebuilds + replays from clock 0. Emulate by
    // a fresh backend per step (empty ladder ⇒ from-0 replay), seeking near the deep end each time.
    let t1 = Instant::now();
    for k in 0..steps {
        let mut cold = mk();
        cold.seek(deep - k);
        cold.step_back();
    }
    let cold_ms = t1.elapsed().as_secs_f64() * 1e3;

    println!(
        "reverse sweep ({steps} step_backs, ~64k-op run): warm={warm_ms:.1}ms cold={cold_ms:.1}ms \
         speedup={:.1}x (checkpoints={warm_ckpts})",
        cold_ms / warm_ms,
    );
    assert!(warm_ckpts > 0, "the warm run laid down a ladder");
}
