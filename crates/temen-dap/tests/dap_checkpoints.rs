//! The **warm≡cold gate** for time travel: one harness, [`warm_matches_cold`], over every
//! continuation a [`Moment`](temen_interp::moment::Moment) carries (#1460) — the tree-walk oracle's
//! `ShadowStack`, the bytecode engine's `Bytecode`, and a reactor's `None` — and over every guest
//! shape the bytecode ladder admits (fibers, threads, host capabilities, regions).
//!
//! A **warm** driver (its ladder populated by a prior deep seek, so a seek *restores* the nearest
//! moment at or before the target and replays only the tail) must observe **identical** state to a
//! **cold** one (fresh, empty ladder, replaying from the start). The cold path is the trusted replay;
//! the warm path exercises capture and restore. Behavior is a pure optimization: correctness is
//! *defined* by the cold path.
//!
//! This harness is also the differential that keeps the tree-walker's continuation a separate
//! implementation from the bytecode engine's (INVARIANTS #15's exemption): both are held to the same
//! gate here, cell by cell.

use std::fmt::Debug;
use temen_dap::{BytecodeBackend, Debuggee};
use temen_interp::moment::{Moment, MomentReactor, ReactorTimeline, Refusal, SteppableReactor};
use temen_interp::{bytecode, Host, Inspector, Value};
use temen_text::parse_module;

/// The gate. `mk` opens a fresh driver; `observe` moves one to a coordinate and reads what a user
/// could read there; `rungs` counts its ladder. Every probe is observed cold (a fresh driver each) and
/// then warm (one driver, after a seek to the deepest probe laid its ladder down), forwards and then
/// backwards, and the two must agree. The ladder must still hold rungs at the end, which a refusal
/// anywhere in the sweep would have dropped. Returns the cold observations, for a cell's own sanity
/// checks on the guest, and the warm driver's rung count.
fn warm_matches_cold<T, O: PartialEq + Debug + Clone>(
    mk: impl Fn() -> T,
    observe: impl Fn(&mut T, u64) -> O,
    rungs: impl Fn(&T) -> usize,
    probes: &[u64],
) -> (Vec<O>, usize) {
    let cold: Vec<O> = probes.iter().map(|&t| observe(&mut mk(), t)).collect();

    let mut warm = mk();
    observe(&mut warm, *probes.iter().max().expect("probes"));
    assert!(
        rungs(&warm) > 0,
        "a seek to the deepest probe lays down a ladder — it is exercised, not dormant"
    );
    let warm_fwd: Vec<O> = probes.iter().map(|&t| observe(&mut warm, t)).collect();
    assert_eq!(warm_fwd, cold, "warm ≡ cold at every forward probe");
    let warm_back: Vec<O> = probes
        .iter()
        .rev()
        .map(|&t| observe(&mut warm, t))
        .collect();
    let cold_back: Vec<O> = cold.iter().rev().cloned().collect();
    assert_eq!(warm_back, cold_back, "warm ≡ cold seeking backwards");
    let held = rungs(&warm);
    assert!(held > 0, "the ladder was never refused");
    (cold, held)
}

/// A fresh bytecode debug session over `src` with `args`; `powerbox` grants the on-ramp I/O powerbox
/// (`vm_fs`, `memory`).
fn bytecode_session(src: &str, args: Vec<Value>, powerbox: bool) -> impl Fn() -> BytecodeBackend {
    let m = parse_module(src).expect("parses");
    move || {
        BytecodeBackend::new(
            m.clone(),
            0,
            &args,
            u64::MAX,
            powerbox,
            Vec::new(),
            false,
            None,
            None,
        )
        .expect("the bytecode engine accepts the guest")
    }
}

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
/// counter at 16384. The scheduled engine's checkpoint consults the *same* `Host::admits_checkpoint`
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
fn obs(b: &mut impl Debuggee, t: u64) -> (u64, String, Vec<u8>) {
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

/// The `ShadowStack` cell: the tree-walk oracle's `Inspector`, seeking by op clock.
#[test]
fn shadow_stack_warm_seek_matches_cold() {
    let m = parse_module(LOOP_WITH_MEM).expect("parses");
    let mk = || Inspector::attach(&m, 0, &[Value::I32(800)], 50_000_000);
    let probes: Vec<u64> = (0..=6000).step_by(137).chain([1023, 1024, 1025]).collect();
    warm_matches_cold(mk, obs, Inspector::checkpoint_count, &probes);
}

/// The `Bytecode` cell: the same loop on the bytecode debug engine, seeking by turn. Several thousand
/// ops cross several stride boundaries, and the probes are deliberately not stride-aligned, so a
/// restore lands strictly below its target and replays a nonzero tail.
#[test]
fn bytecode_warm_seek_matches_cold() {
    let probes: Vec<u64> = (0..=6000).step_by(137).collect();
    warm_matches_cold(
        bytecode_session(LOOP_WITH_MEM, vec![Value::I32(800)], false),
        obs,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
}

/// The bulk of this run executes *inside* the fiber (root parked on the resume chain), so nearly
/// every checkpoint captures a live §12 fiber continuation.
#[test]
fn bytecode_warm_seek_matches_cold_with_a_live_fiber() {
    let probes: Vec<u64> = (0..=5000).step_by(131).collect();
    let (cold, _) = warm_matches_cold(
        bytecode_session(FIBER_LOOP, vec![Value::I64(900)], false),
        obs,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
    assert!(
        cold.iter().any(|(_, stack, _)| stack.contains("0:1:")),
        "the run spends time inside the fiber body (func 1)",
    );
}

/// A reactor for the `None` cell: each tick adds 1 to a counter at 16384 and returns it. It holds no
/// capability, so the timeline's tape stays empty and a moment is the window alone.
const TICK: &str = "\
memory 16
func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vx = i64.load va
  v1 = i64.const 1
  vy = i64.add vx v1
  i64.store va vy
  return vy
  }
}
";

struct TickReactor {
    inst: bytecode::Reactor,
    host: Host,
}

impl MomentReactor for TickReactor {
    fn push_key(&self, _: i32, _: i32) {}
    fn push_mouse(&self, _: i32, _: i32) {}
    fn moment(&self) -> Result<Moment, Refusal> {
        Moment::capture(
            self.inst.window_layout().ok_or(Refusal::NoWindow)?,
            &self.host,
        )
    }
    fn restore(&mut self, m: &Moment) -> bool {
        let ok = self.inst.restore_window(&m.layout(), &self.host);
        m.restore_host(&mut self.host);
        ok
    }
}

impl SteppableReactor for TickReactor {
    fn step(&mut self) -> i32 {
        let mut fuel = u64::MAX;
        match self.inst.call(0, &[], &mut fuel, &mut self.host) {
            Ok(_) => 0,
            Err(_) => -1,
        }
    }
}

/// A reactor and its timeline, moved by tick: back inside the recording by a seek, past its end by
/// recording forward.
struct Scrub {
    r: TickReactor,
    t: ReactorTimeline,
}

fn scrub(s: &mut Scrub, tick: u64) -> (u64, Vec<u8>) {
    let tick = tick as usize;
    assert!(s.t.seek(&mut s.r, tick.min(s.t.len())), "seek to {tick}");
    while s.t.tick() < tick {
        assert_eq!(s.t.frame(&mut s.r), 0, "the tick runs");
    }
    let layout = s.r.inst.window_layout().expect("a window");
    (s.t.tick() as u64, layout.bytes()[16384..16392].to_vec())
}

/// The `None` cell: a reactor's keyframe ladder, seeking by tick.
#[test]
fn reactor_warm_seek_matches_cold() {
    let m = parse_module(TICK).expect("parses");
    let mk = || Scrub {
        r: TickReactor {
            inst: bytecode::Reactor::open(&m).expect("open the reactor"),
            host: Host::new(),
        },
        t: ReactorTimeline::new(16, 64, 0),
    };
    let probes: Vec<u64> = (0..=200).step_by(7).chain([16, 17, 200]).collect();
    let (cold, _) = warm_matches_cold(mk, scrub, |s| s.t.keyframe_ticks().len(), &probes);
    assert_eq!(
        cold.last().unwrap().1,
        200i64.to_le_bytes(),
        "200 ticks ran"
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

/// Two workers × 400 iterations ⇒ several thousand turns, many strides, on the scheduled engine.
#[test]
fn scheduled_warm_seek_matches_cold() {
    let probes: Vec<u64> = (0..=6000).step_by(149).collect();
    warm_matches_cold(
        bytecode_session(LOOP_THREADS, vec![Value::I64(400)], false),
        obs_sched,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
}

/// Two worker vCPUs each drive a fiber (the increment loop runs inside the fiber), so nearly every
/// scheduled checkpoint captures live per-task fibers + the run-shared registry, interleaved.
#[test]
fn scheduled_warm_seek_matches_cold_with_live_fibers() {
    let probes: Vec<u64> = (0..=6000).step_by(149).collect();
    let (cold, _) = warm_matches_cold(
        bytecode_session(THREADS_WITH_FIBERS, vec![Value::I64(400)], false),
        obs_sched,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
    assert!(
        cold.iter()
            .any(|(_, _, _, _, stacks)| stacks.contains(":2:")),
        "the run spends turns inside a fiber body (func 2)",
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
    let mk = bytecode_session(LOOP_WITH_MEM, vec![Value::I32(8_000)], false); // ~64k ops
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
fn bytecode_warm_seek_matches_cold_with_a_host_capability() {
    // ~8k ops ⇒ several stride boundaries, ~600 cap crossings; the on-ramp powerbox grants `vm_fs`.
    let probes: Vec<u64> = (0..=6000).step_by(137).collect();
    let (cold, _) = warm_matches_cold(
        bytecode_session(FILE_WRITE_LOOP, vec![Value::I64(600)], true),
        obs,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
    // The run really does cross the capability boundary and the accumulator really moves —
    // otherwise "warm ≡ cold" would be a statement about a guest that never touched its powerbox.
    assert!(
        cold.windows(2).any(|w| w[0].2 != w[1].2),
        "the guest's byte count advances across the probes",
    );
}

/// The scheduled (multi-vCPU) twin of the case above: two workers driving the same `vm_fs` capability
/// across global turns. One predicate governs both engines, so this is the propagation pin rather than
/// a second mechanism.
#[test]
fn scheduled_warm_seek_matches_cold_with_a_host_capability() {
    let probes: Vec<u64> = (0..=6000).step_by(139).collect();
    let (cold, _) = warm_matches_cold(
        bytecode_session(FILE_WRITE_THREADS, vec![Value::I64(300)], true),
        obs,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
    assert!(
        cold.windows(2).any(|w| w[0].2 != w[1].2),
        "the workers' byte count advances across the probes",
    );
}

/// #2026 — a guest that **creates, maps and writes a §13 region** mid-run. Phase one bumps a window
/// counter at 16384 `n` times; then it mints a 64 KiB region through its `memory` capability (the
/// handle arrives as the first argument), stores the handle at 16392, and maps the region at 128 KiB.
/// Phase two adds 3 to a counter at 131080 — in the region — `n` times. Last it mints a second
/// region, stores that handle at 16400, and returns the region counter.
///
/// So a checkpoint taken in phase two must carry the region's bytes, the handle naming it and the
/// window's alias; and the second mint, re-run after a restore, must land on the handle a from-0 run
/// gives it, which it does only if the restored table holds the first one where the run did.
const REGION_LOOP: &str = "\
memory 18
func (i32, i64) -> (i64) {
block 0 (vas: i32, vn: i64) {
  br 1(vas, vn, vn)
}
block 1 (va1: i32, vn1: i64, vi1: i64) {
  vz1 = i64.eqz vi1
  br_if vz1 3(va1, vn1) 2(va1, vn1, vi1)
}
block 2 (va2: i32, vn2: i64, vi2: i64) {
  vaddr2 = i64.const 16384
  vc2 = i64.load vaddr2
  vone2 = i64.const 1
  vs2 = i64.add vc2 vone2
  i64.store vaddr2 vs2
  vm2 = i64.const -1
  vnext2 = i64.add vi2 vm2
  br 1(va2, vn2, vnext2)
}
block 3 (va3: i32, vn3: i64) {
  vlen3 = i64.const 65536
  vrh3 = call.cap 5 5 (i64) -> (i64) va3 (vlen3)
  vh3 = i64.const 16392
  i64.store vh3 vrh3
  vr3 = i32.wrap_i64 vrh3
  voff3 = i64.const 131072
  vroff3 = i64.const 0
  vprot3 = i64.const 3
  vm3 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vr3 (voff3, vroff3, vlen3, vprot3)
  br 4(va3, vn3)
}
block 4 (va4: i32, vi4: i64) {
  vz4 = i64.eqz vi4
  br_if vz4 6(va4) 5(va4, vi4)
}
block 5 (va5: i32, vi5: i64) {
  vaddr5 = i64.const 131080
  vc5 = i64.load vaddr5
  vthree5 = i64.const 3
  vs5 = i64.add vc5 vthree5
  i64.store vaddr5 vs5
  vm5 = i64.const -1
  vnext5 = i64.add vi5 vm5
  br 4(va5, vnext5)
}
block 6 (va6: i32) {
  vlen6 = i64.const 65536
  vrh6 = call.cap 5 5 (i64) -> (i64) va6 (vlen6)
  vh6 = i64.const 16400
  i64.store vh6 vrh6
  vaddr6 = i64.const 131080
  vr6 = i64.load vaddr6
  return vr6
  }
}";

/// The `memory` capability's handle in the on-ramp powerbox: the fourth grant of the §3e prefix
/// (slot 3, first generation).
const MEMORY_HANDLE: i32 = (1 << 8) | 3;

/// The clock, the stack, and every value [`REGION_LOOP`] keeps: the window counter, both region
/// handles, and the region counter read through the alias.
fn obs_region(b: &mut BytecodeBackend, t: u64) -> (u64, String, Vec<u8>) {
    let (clock, stack, _) = obs(b, t);
    let mut mem = b.read_window(16384, 24).unwrap_or_default();
    mem.extend(b.read_window(131080, 8).unwrap_or_default());
    (clock, stack, mem)
}

#[test]
fn bytecode_warm_seek_matches_cold_with_a_region() {
    let end = 8100; // past the end: the run finishes at turn 8024
    let probes: Vec<u64> = (0..=end).step_by(131).chain([end]).collect();
    let (cold, held) = warm_matches_cold(
        bytecode_session(
            REGION_LOOP,
            vec![Value::I32(MEMORY_HANDLE), Value::I64(400)],
            true, // the on-ramp powerbox: its `memory` capability mints the regions
        ),
        obs_region,
        BytecodeBackend::checkpoint_count,
        &probes,
    );
    // The run really maps the region and writes through it, and really mints the second one.
    let last = &cold.last().expect("probes").2;
    assert_eq!(
        &last[24..],
        &1200i64.to_le_bytes(),
        "400 bumps of 3, through the alias"
    );
    assert_ne!(&last[8..16], &[0; 8], "the first region's handle");
    assert_ne!(&last[16..24], &[0; 8], "the second region's handle");
    assert!(
        held > 4,
        "checkpoints are laid down past the mint and the map — before #2026 the ladder dropped \
         itself the moment the guest minted a region",
    );
}
