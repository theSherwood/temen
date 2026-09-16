//! Warm≡cold oracle for the bytecode engine's time-travel **checkpoint ladder** (DEBUGGING.md W1) —
//! the `BytecodeBackend` port of the tree-walker `crates/temen/tests/debug_checkpoints.rs`.
//!
//! A **warm** backend (its ladder populated by a prior deep `seek`, so `seek` *restores* from the
//! nearest snapshot `clock <= t` and replays only the tail) must observe **identical** state — the
//! logical clock, the call stack, and guest memory — as a **cold** backend (fresh, empty ladder,
//! replaying from clock 0). The cold path is the pre-existing, trusted replay; the warm path exercises
//! snapshot capture + restore. If restore is faithful they agree at every probed time, including
//! across stride boundaries and a full backward sweep. Behavior is a pure optimization: correctness is
//! *defined* by the cold path.

use temen_dap::{BytecodeBackend, Debuggee};
use temen_interp::Value;
use temen_text::parse_module;

/// A counter loop that also stores its running sum to window address 16384 (above the #1094 NULL
/// guard) each iteration, so a faithful
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
  v9 = i32.const 16384
  i32.store v9 v8
  v10 = i32.const -1
  v11 = i32.add v6 v10
  br 1(v11, v8)
  }
}";

/// A `thread.spawn`/`join` guest that runs **well past the checkpoint stride**: the root spawns two
/// workers (each looping its arg times, bumping a shared window counter), joins both, and returns the
/// counter. Drives the multi-vCPU `ScheduledDebugRun`, whose checkpoint must restore every task's `Vm`,
/// the join tables, the run states, and the shared window bytes.
const LOOP_THREADS: &str = "\
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vsp = i64.const 0
  vh0 = thread.spawn 1 vsp vn
  vh1 = thread.spawn 1 vsp vn
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 16384
  vr = i64.load vaddr
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1(varg)
}
block 1 (vi: i64) {
  vz = i64.eqz vi
  br_if vz 2() 3(vi)
}
block 2 () {
  vr = i64.const 0
  return vr
}
block 3 (vi2: i64) {
  vaddr = i64.const 16384
  vc = i64.load vaddr
  v1 = i64.const 1
  vsum = i64.add vc v1
  i64.store vaddr vsum
  vm1 = i64.const -1
  vnext = i64.add vi2 vm1
  br 1(vnext)
  }
}";

/// A §12 **fiber** generator whose fiber body runs a long internal loop before it first suspends, so
/// the *bulk* of the run's ops execute with the fiber as the active continuation and the root parked on
/// the resume chain — the state a checkpoint must capture (active fiber `Vm` + chain + registry). The
/// root creates the fiber, resumes it (it loops `arg` times summing a counter, then suspends the sum),
/// resumes again (it returns sum+5), and adds the two.
const FIBER_LOOP: &str = "\
func (i64) -> (i64) {
block 0 (vn: i64) {
  v0 = ref.func 1
  v1 = i64.const 0
  vc = cont.new v0 v1
  vs0, vy = cont.resume vc vn
  v6 = i64.const 0
  vs1, vr = cont.resume vc v6
  v9 = i64.add vy vr
  return v9
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vacc0 = i64.const 0
  br 1(varg, vacc0)
}
block 1 (vi: i64, vacc: i64) {
  vz = i64.eqz vi
  br_if vz 2(vacc) 3(vi, vacc)
}
block 2 (vsum: i64) {
  vres = suspend vsum
  v5 = i64.const 5
  vfin = i64.add vsum v5
  return vfin
}
block 3 (vi2: i64, vacc2: i64) {
  v1 = i64.const 1
  vnext = i64.add vacc2 v1
  vm1 = i64.const -1
  vid = i64.add vi2 vm1
  br 1(vid, vnext)
  }
}";

/// A **threaded** guest whose workers drive §12 fibers: the root spawns two workers, each of which
/// creates a fiber that loops `n` times incrementing the shared counter at address 0, then suspends and
/// returns. Because the increment loop runs *inside* the fiber, most turns execute with a fiber as the
/// worker's active continuation (worker parked on the resume chain) — so the scheduled checkpoints
/// capture live per-task fibers plus the run-shared fiber registry, interleaved across the two vCPUs.
const THREADS_WITH_FIBERS: &str = "\
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vsp = i64.const 0
  vh0 = thread.spawn 1 vsp vn
  vh1 = thread.spawn 1 vsp vn
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 16384
  vr = i64.load vaddr
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vf = ref.func 2
  vz = i64.const 0
  vc = cont.new vf vz
  vst, vy = cont.resume vc vn
  vst2, vr = cont.resume vc vz
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1(varg)
}
block 1 (vi: i64) {
  vz = i64.eqz vi
  br_if vz 2() 3(vi)
}
block 2 () {
  vzero = i64.const 0
  vres = suspend vzero
  vr = i64.const 0
  return vr
}
block 3 (vi2: i64) {
  vaddr = i64.const 16384
  vc = i64.load vaddr
  v1 = i64.const 1
  vsum = i64.add vc v1
  i64.store vaddr vsum
  vm1 = i64.const -1
  vnext = i64.add vi2 vm1
  br 1(vnext)
  }
}";

/// #1455 — a guest that holds a **host capability**: it opens a scratch file through the `vm_fs` seam
/// (chibicc's `__vm_fs` builtin shape — a flat `call.sym` with the fs op in arg0) and appends one byte
/// per iteration, accumulating the bytes written at 16384. Every iteration therefore crosses the
/// capability boundary, on both sides of every stride boundary.
///
/// Before #1455's ladder half this run was **not checkpointable at all**: `Host::checkpoint_safe`
/// required `host_procs.is_empty()`, so the ladder self-disabled the moment the powerbox granted
/// `vm_fs` and every `seek`/`step_back` replayed from clock 0. That is the class, not the corner: a
/// debugged C program that does file I/O is in it, and so is every playground reactor.
const FILE_WRITE_LOOP: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vzero = i64.const 0
  vpath = i64.const 16400
  vcf = i32.const 102
  i32.store8 vpath vcf
  vnp = i64.const 16420
  vcv = i32.const 118
  i32.store8 vnp vcv
  vn1 = i64.const 16421
  vcm = i32.const 109
  i32.store8 vn1 vcm
  vn2 = i64.const 16422
  vcu = i32.const 95
  i32.store8 vn2 vcu
  vn3 = i64.const 16423
  vcf2 = i32.const 102
  i32.store8 vn3 vcf2
  vn4 = i64.const 16424
  vcs = i32.const 115
  i32.store8 vn4 vcs
  vwbuf = i64.const 16440
  vbA = i32.const 65
  i32.store8 vwbuf vbA
  vnl = i64.const 5
  vh = self.resolve vnp vnl
  vopen = i64.const 0
  vplen = i64.const 1
  vflags = i64.const 19
  vfd = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vopen, vpath, vplen, vflags, vzero)
  vacc0 = i64.const 0
  br 1(vn, vacc0, vfd)
}
block 1 (vi: i64, vacc: i64, vfd: i64) {
  vz = i64.eqz vi
  br_if vz 2(vacc) 3(vi, vacc, vfd)
}
block 2 (vsum: i64) {
  return vsum
}
block 3 (vi2: i64, vacc2: i64, vfd2: i64) {
  vnp2 = i64.const 16420
  vnl2 = i64.const 5
  vh2 = self.resolve vnp2 vnl2
  vwbuf2 = i64.const 16440
  vone2 = i64.const 1
  vzero2 = i64.const 0
  vwrite = i64.const 2
  vwn = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh2 (vwrite, vfd2, vwbuf2, vone2, vzero2)
  vsum = i64.add vacc2 vwn
  vcell = i64.const 16384
  i64.store vcell vsum
  vm1 = i64.const -1
  vnext = i64.add vi2 vm1
  br 1(vnext, vsum, vfd2)
  }
}
"#;

/// The **threaded** twin of [`FILE_WRITE_LOOP`]: the root opens nothing, each of two spawned workers
/// opens the scratch file through `vm_fs` and appends bytes in a loop, accumulating into the shared
/// counter at 16384. The scheduled engine's `checkpointable` consults the *same* `Host::checkpoint_safe`
/// — over the root host and every `extra_envs` child — so admitting a named capability admits it here
/// too; this pins that rather than assuming it (INVARIANTS #14).
const FILE_WRITE_THREADS: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vpath = i64.const 16400
  vcf = i32.const 102
  i32.store8 vpath vcf
  vnp = i64.const 16420
  vcv = i32.const 118
  i32.store8 vnp vcv
  vn1 = i64.const 16421
  vcm = i32.const 109
  i32.store8 vn1 vcm
  vn2 = i64.const 16422
  vcu = i32.const 95
  i32.store8 vn2 vcu
  vn3 = i64.const 16423
  vcf2 = i32.const 102
  i32.store8 vn3 vcf2
  vn4 = i64.const 16424
  vcs = i32.const 115
  i32.store8 vn4 vcs
  vwbuf = i64.const 16440
  vbA = i32.const 65
  i32.store8 vwbuf vbA
  vsp = i64.const 0
  vh0 = thread.spawn 1 vsp vn
  vh1 = thread.spawn 1 vsp vn
  vj0 = thread.join vh0
  vj1 = thread.join vh1
  vaddr = i64.const 16384
  vr = i64.load vaddr
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vnp = i64.const 16420
  vnl = i64.const 5
  vh = self.resolve vnp vnl
  vzero = i64.const 0
  vopen = i64.const 0
  vpath = i64.const 16400
  vplen = i64.const 1
  vflags = i64.const 19
  vfd = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh (vopen, vpath, vplen, vflags, vzero)
  br 1(vn, vfd)
}
block 1 (vi: i64, vfd: i64) {
  vz = i64.eqz vi
  br_if vz 2() 3(vi, vfd)
}
block 2 () {
  vr = i64.const 0
  return vr
}
block 3 (vi2: i64, vfd2: i64) {
  vnp2 = i64.const 16420
  vnl2 = i64.const 5
  vh2 = self.resolve vnp2 vnl2
  vwbuf2 = i64.const 16440
  vone = i64.const 1
  vzero2 = i64.const 0
  vwrite = i64.const 2
  vwn = call.sym "vm_fs" (i64, i64, i64, i64, i64) -> (i64) vh2 (vwrite, vfd2, vwbuf2, vone, vzero2)
  vaddr = i64.const 16384
  vc = i64.load vaddr
  vsum = i64.add vc vwn
  i64.store vaddr vsum
  vm1 = i64.const -1
  vnext = i64.add vi2 vm1
  br 1(vnext, vfd2)
  }
}
"#;

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
    let mem = b.read_window(16384, 4).unwrap_or_default();
    (clock, stack, mem)
}

#[test]
fn bytecode_checkpoint_warm_seek_matches_cold_replay_from_zero() {
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    let args = [Value::I32(800)]; // several thousand ops ⇒ several stride boundaries
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            None,
        )
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

#[test]
fn bytecode_checkpoint_warm_seek_matches_cold_with_a_live_fiber() {
    // The bulk of this run executes *inside* the fiber (root parked on the resume chain), so nearly
    // every checkpoint captures a live §12 fiber continuation. A checkpoint-restored `seek` must
    // reproduce the fiber's stack and the final result exactly as a from-0 replay.
    let m = parse_module(FIBER_LOOP).expect("parses");
    let args = [Value::I64(900)]; // fiber loops 900× before its first suspend ⇒ several strides deep
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the single-vCPU engine accepts the fiber generator")
    };

    let probes: Vec<u64> = (0..=5000).step_by(131).collect();
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs(&mut b, t)
        })
        .collect();
    // Sanity: the run genuinely executes inside the fiber (func 1), so the checkpoints below capture a
    // live fiber continuation rather than only the root.
    assert!(
        cold.iter().any(|(_, stack, _)| stack.contains("0:1:")),
        "the run spends time inside the fiber body (func 1)",
    );

    let mut warm = mk();
    warm.seek(5000);
    assert!(
        warm.checkpoint_count() > 0,
        "a deep seek through the fiber body lays down checkpoints with a live fiber",
    );
    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm (checkpoint-restored) seek ≡ cold at every forward probe with a live fiber",
    );
    let warm_back: Vec<_> = probes.iter().rev().map(|&t| obs(&mut warm, t)).collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(
        warm_back, cold_back,
        "warm backward sweep ≡ cold with a live fiber (fiber chain + registry restore is faithful)",
    );
}

/// A stable per-`seek` observation on the **threaded** engine: the global turn, the live-thread count,
/// which thread is stopped, the shared counter bytes, and *every* live thread's call stack (selecting
/// each). A faithful checkpoint restore reproduces the whole cross-thread state; an unfaithful one
/// diverges in the counter, a stack, or the schedule position.
fn obs_sched(b: &mut BytecodeBackend, t: u64) -> (u64, usize, Option<u64>, Vec<u8>, String) {
    b.seek(t);
    let turn = b.turn();
    let threads = b.threads();
    let stopped = b.stopped_task();
    let mem = b.read_window(16384, 8).unwrap_or_default();
    let mut stacks = Vec::new();
    for &tid in &threads {
        b.select_task(tid);
        let s = b
            .backtrace()
            .iter()
            .map(|f| format!("{}:{}:{}:{}", f.pc.module, f.pc.func, f.pc.block, f.pc.inst))
            .collect::<Vec<_>>()
            .join(",");
        stacks.push(format!("{tid}=[{s}]"));
    }
    (turn, threads.len(), stopped, mem, stacks.join(";"))
}

#[test]
fn scheduled_checkpoint_warm_seek_matches_cold_replay_from_zero() {
    let m = parse_module(LOOP_THREADS).expect("parses");
    let args = [Value::I64(400)]; // two workers × 400 iters ⇒ several thousand turns, many strides
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the scheduled bytecode engine accepts the thread.spawn loop")
    };

    let probes: Vec<u64> = (0..=6000).step_by(149).collect();

    // Cold baseline: a fresh backend per probe (empty ladder ⇒ always a from-turn-0 replay).
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs_sched(&mut b, t)
        })
        .collect();

    // Warm: one backend, a deep seek to populate the scheduled ladder, then the probes reuse it.
    let mut warm = mk();
    warm.seek(6000);
    assert!(
        warm.checkpoint_count() > 0,
        "a deep scheduled seek past the stride lays down checkpoints",
    );
    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs_sched(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm (checkpoint-restored) scheduled seek ≡ cold (replay-from-turn-0) at every forward probe",
    );

    let warm_back: Vec<_> = probes
        .iter()
        .rev()
        .map(|&t| obs_sched(&mut warm, t))
        .collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(
        warm_back, cold_back,
        "warm scheduled backward sweep ≡ cold (restore is faithful across the whole task set)",
    );

    assert!(
        warm.checkpoint_count() > 0,
        "checkpointing stays on for a pure thread.spawn/join loop (no fibers/coroutines/§14 children)",
    );
}

#[test]
fn scheduled_checkpoint_warm_seek_matches_cold_with_live_fibers() {
    // Two worker vCPUs each drive a fiber (the increment loop runs inside the fiber), so nearly every
    // scheduled checkpoint captures live per-task fibers + the run-shared registry, interleaved. A
    // checkpoint-restored `seek` must reproduce the whole cross-thread state exactly as a from-turn-0
    // replay — proving the scheduled fiber snapshot/restore is faithful.
    let m = parse_module(THREADS_WITH_FIBERS).expect("parses");
    let args = [Value::I64(400)]; // two fibers × 400 iters ⇒ several thousand turns, many strides
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the scheduled bytecode engine accepts thread.spawn workers driving fibers")
    };

    let probes: Vec<u64> = (0..=6000).step_by(149).collect();
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs_sched(&mut b, t)
        })
        .collect();
    // Sanity: the run genuinely executes inside a fiber body (func 2) on a worker, so the checkpoints
    // below capture live fibers, not just the root/worker frames.
    assert!(
        cold.iter()
            .any(|(_, _, _, _, stacks)| stacks.contains(":2:")),
        "the run spends turns inside a fiber body (func 2)",
    );

    let mut warm = mk();
    warm.seek(6000);
    assert!(
        warm.checkpoint_count() > 0,
        "a deep scheduled seek through the fiber bodies lays down checkpoints with live fibers",
    );
    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs_sched(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm scheduled seek ≡ cold at every forward probe with live per-task fibers",
    );
    let warm_back: Vec<_> = probes
        .iter()
        .rev()
        .map(|&t| obs_sched(&mut warm, t))
        .collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(
        warm_back, cold_back,
        "warm scheduled backward sweep ≡ cold (per-task chains + run-shared registry restore faithfully)",
    );
}

/// Wall-clock evidence that the ladder actually bounds the reverse-replay cost (DEBUGGING.md W1):
/// a backward `step_back` sweep over a long run is dramatically faster warm (restart from the nearest
/// checkpoint, replay ≤ one stride) than cold (rebuild + replay from clock 0 each time — the old
/// behavior). `#[ignore]`d because wall-clock ratios are runner-dependent and must never gate CI;
/// run with `cargo test -p temen-dap --test dap_checkpoints -- --ignored --nocapture` to see the ratio.
#[test]
#[ignore = "timing benchmark; run manually with --ignored --nocapture"]
fn bytecode_checkpoint_reverse_sweep_is_bounded() {
    use std::time::Instant;
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    let args = [Value::I32(8_000)]; // ~64k ops — many strides deep
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            false,
            Vec::new(),
            false,
            None,
            None,
        )
        .unwrap()
    };
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

/// #1455 (the ladder half) — the warm≡cold oracle on a guest that **holds a host capability**.
///
/// This is the acceptance case the issue names. The powerbox grants `vm_fs` as a `HostProc`, which used
/// to disqualify the whole run from checkpointing (`host_procs.is_empty()`), so `checkpoint_count()` was
/// forced to `0` and every reverse step paid O(t). A named capability is now admitted, and the property
/// that has to survive is the same one the capability-free cases assert: a checkpoint-restored `seek`
/// observes exactly what a from-0 replay observes, forwards and backwards.
///
/// What makes it sound is the `CapTape`: every `HOST_PROC` crossing is recorded, so a replay — from a
/// checkpoint or from zero — serves them from the tape rather than re-entering the closure. The
/// capability's own declared state rides the checkpoint alongside (`Host::capture_cap_states`), which is
/// the half that matters once a replay runs past the tape's end.
#[test]
fn bytecode_checkpoint_warm_seek_matches_cold_with_a_host_capability() {
    let m = parse_module(FILE_WRITE_LOOP).expect("parses");
    let args = [Value::I64(600)]; // ~8k ops ⇒ several stride boundaries, ~600 cap crossings
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            true, // the on-ramp I/O powerbox — this is what grants `vm_fs`
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the bytecode engine accepts the file-writing loop")
    };

    let probes: Vec<u64> = (0..=6000).step_by(137).collect();
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs(&mut b, t)
        })
        .collect();
    // Sanity: the run really does cross the capability boundary and the accumulator really moves —
    // otherwise "warm ≡ cold" would be a statement about a guest that never touched its powerbox.
    assert!(
        cold.iter()
            .map(|(_, _, mem)| mem)
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] != w[1]),
        "the guest's byte count advances across the probes",
    );

    let mut warm = mk();
    warm.seek(6000);
    assert!(
        warm.checkpoint_count() > 0,
        "a cap-holding guest now lays down checkpoints — before #1455 the ladder self-disabled the \
         moment the powerbox granted `vm_fs`, and this count was forced to 0",
    );

    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm (checkpoint-restored) seek ≡ cold (replay-from-0) at every forward probe",
    );

    let warm_back: Vec<_> = probes.iter().rev().map(|&t| obs(&mut warm, t)).collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(
        warm_back, cold_back,
        "warm backward sweep ≡ cold (restore is faithful seeking in either direction)",
    );

    assert!(
        warm.checkpoint_count() > 0,
        "checkpointing stays on for the whole run — holding a named capability never leaves the subset",
    );
}

/// The scheduled (multi-vCPU) twin of the case above: two workers driving the same `vm_fs` capability
/// across global turns. One predicate governs both engines, so this is the propagation pin rather than
/// a second mechanism.
#[test]
fn scheduled_checkpoint_warm_seek_matches_cold_with_a_host_capability() {
    let m = parse_module(FILE_WRITE_THREADS).expect("parses");
    let args = [Value::I64(300)];
    let mk = || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            true, // the on-ramp I/O powerbox — this is what grants `vm_fs`
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the scheduled engine accepts the threaded file-writing loop")
    };

    let probes: Vec<u64> = (0..=6000).step_by(139).collect();
    let cold: Vec<_> = probes
        .iter()
        .map(|&t| {
            let mut b = mk();
            obs(&mut b, t)
        })
        .collect();
    assert!(
        cold.iter()
            .map(|(_, _, mem)| mem)
            .collect::<Vec<_>>()
            .windows(2)
            .any(|w| w[0] != w[1]),
        "the workers' byte count advances across the probes",
    );

    let mut warm = mk();
    warm.seek(6000);
    assert!(
        warm.checkpoint_count() > 0,
        "the scheduled ladder admits a cap-holding run too — same `checkpoint_safe`, over the root \
         host and every child env",
    );

    let warm_fwd: Vec<_> = probes.iter().map(|&t| obs(&mut warm, t)).collect();
    assert_eq!(
        warm_fwd, cold,
        "warm (checkpoint-restored) seek ≡ cold (replay-from-turn-0) at every forward probe",
    );

    let warm_back: Vec<_> = probes.iter().rev().map(|&t| obs(&mut warm, t)).collect();
    let cold_back: Vec<_> = cold.iter().rev().cloned().collect();
    assert_eq!(warm_back, cold_back, "warm backward sweep ≡ cold");
}
