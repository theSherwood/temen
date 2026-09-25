//! **#1638 — an unsatisfiable infinite `atomic.wait` is a fault on *every* backend.**
//!
//! INVARIANTS #9: the tree-walk interpreter is the oracle and the fast backends "run these
//! identically **or decline**, never diverge". `crates/temen-run/tests/futex_deadlock_jit.rs`
//! pins this contract for the oracle and the Cranelift JIT — and pins exactly those two, which
//! is how #1624 shipped a divergence: it taught those two engines to detect the deadlock and
//! left the **bytecode** interpreter answering `WAIT_TIMED_OUT`, a status nobody asked for,
//! chosen by the `MAX_WAIT` anti-wedge backstop. Measured on `main` before the fix:
//!
//! | backend  | 1:1 shape                  | fiber shape                   |
//! |----------|----------------------------|-------------------------------|
//! | TreeWalk | `ThreadFault`     0.55 ms  | `ThreadFault`        0.43 ms  |
//! | Bytecode | **`Returned(2)`** 0.25 ms  | **`Returned(2042)`** 0.20 ms  |
//! | Jit      | `ThreadFault`     6.4 ms   | `ThreadFault`       29 ms     |
//!
//! So this file pins **all three backends**, for **both shapes**, which is the concrete lesson
//! #1638 asks for: a two-engine test cannot catch a three-engine divergence, and the sibling
//! fiber files (`fiber_timed_wait.rs`, `fiber_blocking_resume.rs`) were already `pin_all`.
//!
//! Why the bytecode engine could get here at all: it is a cooperative scheduler with a
//! **logical clock** — when nothing is runnable it advances the clock to the earliest `wait`
//! deadline and wakes those waiters timed-out. An infinite wait clamped to `MAX_WAIT` carries a
//! deadline like any other, so the clock jumped to it. `drive`'s own doc already promised the
//! other answer — *"a stuck set advances a logical clock to the next `wait` deadline (**or
//! deadlocks → `ThreadFault`**)"* — and that path was simply unreachable while every infinite
//! wait carried a backstop deadline. The fix makes an infinite wait carry **no** deadline, so it
//! is not a clock-advance candidate and the deadlock exit is reached by construction.
//!
//! **Where this contract stops.** The fiber kernel polls with `cont.resume.block`, which idles
//! the resumer on the fiber (I48). Then nothing in the run is runnable, and that state — every
//! vCPU parked, no wake possible — is the only one in which "this wait can never be satisfied"
//! is decidable. A **non-blocking** `cont.resume` poll loop is not that state, and is correctly
//! *not* a deadlock: the poller is live guest code that may `notify` its own fiber on any later
//! poll. (#1642 proposed asking the predicate from the poll; it would fault correct programs.)
//! Such a loop is bounded by fuel like any other, and both halves are pinned three-engine in
//! `jit_fuel.rs`: the poller that wakes its fiber after 1000 polls completes, and the one that
//! never does exhausts its budget at the identical safepoint.

use std::time::{Duration, Instant};
use temen_run::{instantiate, Backend, RunConfig};

/// Every backend this contract binds. Named once so a future engine is added here, not
/// forgotten — the omission #1638 is about.
const ALL: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// Run `src` on `backend`, returning `Ok(description)` if it produced an ordinary result and
/// `Err(description)` if it ended abnormally, plus how long it took.
fn run_on(src: &str, backend: Backend) -> (Result<String, String>, Duration) {
    let module = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&module).expect("verify");
    let instance = instantiate(module).expect("instantiate");
    let t0 = Instant::now();
    let r = instance.run(backend, &RunConfig::default());
    let dt = t0.elapsed();
    (
        match r {
            Ok(run) => Ok(format!("{:?}", run.outcome)),
            Err(e) => Err(e),
        },
        dt,
    )
}

/// The deadlock verdict must come from the predicate, not from a 10 s `MAX_WAIT` backstop —
/// the same bound `futex_deadlock_jit.rs` uses, and half the point of the test: a fix that
/// traded the wrong *status* for the wrong *latency* would pass the status assertion alone.
const PROMPT: Duration = Duration::from_secs(4);

/// Assert every backend ends `src` abnormally — a `ThreadFault` — and promptly.
fn pin_all_fault(src: &str, what: &str) {
    for backend in ALL {
        let (r, dt) = run_on(src, backend);
        let e = match r {
            Ok(out) => panic!(
                "{backend:?}: {what} must not come back as an ordinary result, got {out} in {dt:?}"
            ),
            Err(e) => e,
        };
        assert!(
            e.contains("ThreadFault"),
            "{backend:?}: {what} must fault, got {e:?} in {dt:?}"
        );
        assert!(
            dt < PROMPT,
            "{backend:?}: {what} must be decided by the deadlock predicate, not waited out \
             on the 10 s MAX_WAIT backstop: took {dt:?}"
        );
    }
}

/// **Shape 1 — the 1:1 case.** A lone vCPU in an infinite wait on a word nothing will ever
/// store or notify, with no sibling that could. This is the exact kernel
/// `futex_deadlock_jit.rs` pins on two engines; here it is pinned on three.
const LONE_INFINITE_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16392
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

#[test]
fn a_lone_infinite_wait_faults_on_every_backend() {
    pin_all_fault(LONE_INFINITE_WAIT, "a lone unsatisfiable infinite wait");
}

/// **Shape 2 — the fiber case**, which #1638 calls out separately because it reaches a
/// different park: a `memory.wait` inside a `cont.new` fiber parks the FIBER, not the vCPU
/// (§3.6 slice 5a), so it lands in `FiberState::WaitParked` rather than the task-level
/// `BlockedWait` — a second deadline that the same clamp used to fabricate. The resumer polls
/// with `cont.resume.block` in a loop (the advisory contract from `fiber_blocking_resume.rs`),
/// so on the fast backends this spins rather than idling — and a spin is exactly where a
/// fabricated real-clock deadline would fire instead of the deadlock.
const FIBER_INFINITE_WAIT: &str = r#"
memory 16
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  br 1(v2)
}
block 1 (vk: i64) {
  vz = i64.const 0
  vs, vv = cont.resume.block vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vv) 1(vk)
}
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vk = i64.const 1000
  vst64 = i64.extend_i32_s vst
  va = i64.mul vst64 vk
  v42 = i64.const 42
  vr = i64.add va v42
  return vr
  }
}
"#;

#[test]
fn an_infinite_wait_inside_a_fiber_faults_on_every_backend() {
    pin_all_fault(
        FIBER_INFINITE_WAIT,
        "an unsatisfiable infinite wait inside a fiber",
    );
}

/// **The control.** The same two shapes with a *finite* timeout must still complete normally
/// with `WAIT_TIMED_OUT` on every backend — the deadlock exit must not swallow an ordinary
/// timed wait that simply has no notifier. Without this, "make it fault" passes by faulting
/// everything.
const LONE_TIMED_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16392
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

#[test]
fn a_finite_wait_with_no_notifier_still_times_out_on_every_backend() {
    for backend in ALL {
        let (r, dt) = run_on(LONE_TIMED_WAIT, backend);
        let out = r.unwrap_or_else(|e| {
            panic!("{backend:?}: a finite wait is a timeout, not a deadlock: {e} in {dt:?}")
        });
        assert!(
            out.contains("I64(2)"),
            "{backend:?}: expected WAIT_TIMED_OUT(2), got {out} in {dt:?}"
        );
    }
}

/// Assert every backend runs `src` to an ordinary result containing `want`.
fn pin_all_completes(src: &str, want: &str, what: &str) {
    for backend in ALL {
        let (r, dt) = run_on(src, backend);
        let out = r.unwrap_or_else(|e| panic!("{backend:?}: {what} must complete: {e} in {dt:?}"));
        assert!(
            out.contains(want),
            "{backend:?}: {what}: expected {want}, got {out} in {dt:?}"
        );
    }
}

/// **A joiner whose child has finished is not blocked.** Each round the root spawns C, which naps
/// 20 µs so the root is parked in `thread.join` when it ends, and W, which waits indefinitely for
/// the store the root makes after the join. A finishing vCPU drops `live` *before* it publishes
/// its result (so a joiner that sees the result also sees the spawn slot freed), and the JIT's
/// joiner used to stay counted in `parked` until it woke and dropped its guard. W, woken by the
/// same exit, could read `live(2) == parked(2)` in between — root and W — and trap `ThreadFault`
/// although the root was about to notify it: #1625's "a woken waiter still counted", on a join.
const JOIN_THEN_NOTIFY: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vz1 = i64.const 0
  vc = thread.spawn 1 vz1 vz1
  vw = thread.spawn 2 vz1 vi
  vrc = thread.join vc
  vk = i64.const 16392
  vone = i64.const 1
  vnext = i64.add vi vone
  vn32 = i32.wrap_i64 vnext
  i32.atomic.store vk vn32
  vcnt = i32.const 1
  vwoke = atomic.notify vk vcnt
  vrw = thread.join vw
  vrounds = i64.const 300
  vmore = i64.lt_u vnext vrounds
  br_if vmore 1(vnext) 2(vnext)
}
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 20000
  vst = i32.atomic.wait vaddr vexp vto
  vz = i64.const 0
  return vz
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vk = i64.const 16392
  vexp = i32.wrap_i64 varg
  vinf = i64.const -1
  vst = i32.atomic.wait vk vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

#[test]
fn a_joiner_whose_child_finished_is_not_counted_blocked() {
    pin_all_completes(
        JOIN_THEN_NOTIFY,
        "I64(300)",
        "join, then notify a waiting sibling",
    );
}

/// **A vCPU driving a fiber whose wait was just claimed is not blocked.** V drives, with
/// `cont.resume.block`, a fiber that waits indefinitely on K1 and then stores and notifies K2. The
/// root notifies K1 and at once waits indefinitely on K2, for 300 rounds. The JIT counts V as
/// parked while it idles on the fiber, and a `notify` claiming the *fiber's* wait used to leave that
/// count alone until V woke and re-polled. The root, parking right after its notify, could read
/// `live(2) == parked(2)` — V and itself — and trap `ThreadFault` against a fiber that was already
/// runnable.
const NOTIFY_A_BLOCK_DRIVEN_FIBER_THEN_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vone = i64.const 1
  br 1(vt, vone)
}
block 1 (vt1: i32, vi: i64) {
  vk1 = i64.const 16384
  vi32 = i32.wrap_i64 vi
  i32.atomic.store vk1 vi32
  vcnt = i32.const 1
  vwoke = atomic.notify vk1 vcnt
  vk2 = i64.const 16392
  vone1 = i64.const 1
  vprev = i64.sub vi vone1
  vp32 = i32.wrap_i64 vprev
  vinf = i64.const -1
  vst = i32.atomic.wait vk2 vp32 vinf
  vnext = i64.add vi vone1
  vlast = i64.const 300
  vmore = i64.le_u vnext vlast
  br_if vmore 1(vt1, vnext) 2(vt1)
}
block 2 (vt2: i32) {
  vr = thread.join vt2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vone = i64.const 1
  br 1(vone)
}
block 1 (vi: i64) {
  vf = ref.func 2
  vz = i64.const 0
  vk = cont.new vf vz
  br 2(vk, vi)
}
block 2 (vk2: i64, vi2: i64) {
  vs, vv = cont.resume.block vk2 vi2
  vdone = i32.const 1
  visdone = i32.eq vs vdone
  br_if visdone 3(vi2) 2(vk2, vi2)
}
block 3 (vi3: i64) {
  vone3 = i64.const 1
  vnext = i64.add vi3 vone3
  vlast = i64.const 300
  vmore = i64.le_u vnext vlast
  br_if vmore 1(vnext) 4(vi3)
}
block 4 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vk1 = i64.const 16384
  vone = i64.const 1
  vprev = i64.sub varg vone
  vp32 = i32.wrap_i64 vprev
  vinf = i64.const -1
  vst = i32.atomic.wait vk1 vp32 vinf
  vk2 = i64.const 16392
  vi32 = i32.wrap_i64 varg
  i32.atomic.store vk2 vi32
  vcnt = i32.const 1
  vwoke = atomic.notify vk2 vcnt
  return varg
  }
}
"#;

#[test]
fn a_vcpu_driving_a_just_notified_fiber_is_not_counted_blocked() {
    pin_all_completes(
        NOTIFY_A_BLOCK_DRIVEN_FIBER_THEN_WAIT,
        "I64(300)",
        "notify a block-driven fiber, then wait for its reply",
    );
}
