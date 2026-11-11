//! §5 the **kill-path across sibling vCPUs** — one host interrupt stops a whole *multithreaded*
//! runaway domain, not just its busy threads.
//!
//! Every vCPU runs the same finalized code, so a **spinning** sibling already polls the one baked
//! interrupt cell and trips `OutOfFuel` on its own. The subtle case is a **parked** sibling —
//! blocked in a futex `wait` or `thread.join` — which isn't executing guest code and so can't poll:
//! `os_thread_rt` re-checks the interrupt cell on a bounded interval while parked, so it wakes and
//! unwinds too. Without that, `join_all` at teardown would hang forever on a vCPU that never
//! finishes. These tests would *hang* (CI timeout) if the mechanism regressed; passing means the
//! whole domain terminates as `OutOfFuel`.
//!
//! Gated to the targets with the JIT thread runtime (`os_thread_rt`).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::time::Duration;

use temen_jit::{compile_and_run_with_host_interruptible, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Compile `src`, arm the kill-path, and let a watchdog request the kill after `delay_ms`. Returns
/// the outcome — which must be `Trapped(OutOfFuel)` once *every* vCPU (the root + its spawned
/// siblings) has unwound (`run_inner` joins them all before returning).
fn run_killed(src: &str, delay_ms: u64) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let interrupt = temen_jit::Interrupt::new();
    let wd = interrupt.clone();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(delay_ms));
        wd.fire(); // request the domain kill
    });
    // #2948 spike: under `TEMEN_JIT_KERNEL=1`, the spike path's run, killed through the same door.
    if let Some(o) = temen_run::jit_kernel::outcome(&m, 0, &[0i64], Some(&interrupt)) {
        watchdog.join().unwrap();
        return o;
    }
    // The guest makes no `call.cap`, so the thunk is never invoked (null ctx never read).
    let outcome = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[0i64],
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        &interrupt,
    )
    .expect("jit compiles");
    watchdog.join().unwrap();
    outcome
}

/// The root vCPU spawns a sibling that **spins**, then spins itself: both poll the shared baked
/// interrupt cell, so the single host kill stops both and the run reports `OutOfFuel`.
#[test]
fn killpath_stops_spinning_sibling() {
    let src = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = thread.spawn 1 v0 v0
  br 1(v0)
}
block 1 (v2: i64) {
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br 1(v4)
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  br 1(v0)
}
block 1 (v2: i64) {
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br 1(v4)
  }
}
";
    assert_eq!(
        run_killed(src, 100),
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "a multithreaded spinning domain must be killed whole"
    );
}

/// The root vCPU spawns a sibling that parks in an **infinite futex `wait`** (timeout `-1`, never
/// notified), then the root spins. The root trips on the interrupt; the *parked* sibling has no
/// guest code running to poll it, so the bounded re-check while parked is what wakes it — it then
/// unwinds (and trips `OutOfFuel` in the spin it falls into). The run terminates; without the
/// re-check, `join_all` would hang on the parked sibling forever.
#[test]
fn killpath_wakes_parked_futex_waiter() {
    let src = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = thread.spawn 1 v0 v0
  br 1(v0)
}
block 1 (v2: i64) {
  v3 = i64.const 1
  v4 = i64.add v2 v3
  br 1(v4)
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 16384
  v3 = i32.const 0
  v4 = i64.const -1
  v5 = i32.atomic.wait v2 v3 v4
  br 1(v0)
}
block 1 (v6: i64) {
  v7 = i64.const 1
  v8 = i64.add v6 v7
  br 1(v8)
  }
}
";
    assert_eq!(
        run_killed(src, 100),
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "a sibling parked in an infinite futex wait must be woken and killed"
    );
}

/// #2757 — a kill reaches a vCPU in a **timed** wait at once, not at the wait's own deadline. The
/// root is the run's only vCPU and sleeps 10 s, so nothing else runs to notice the kill: before the
/// kill door woke parked vCPUs, the timed wait slept to its deadline and the run ended 10 s in. The
/// door wakes it; it returns as woken and traps `OutOfFuel` at the loop's next poll.
#[test]
fn killpath_wakes_a_lone_vcpu_in_a_long_timed_wait() {
    let src = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16384
  v2 = i32.const 0
  v3 = i64.const 10000000000
  v4 = i32.atomic.wait v1 v2 v3
  br 1(v0)
}
block 1 (v5: i64) {
  v6 = i64.const 1
  v7 = i64.add v5 v6
  br 1(v7)
  }
}
";
    // The kill comes once the root has long since parked (its compile is far quicker than this).
    let started = std::time::Instant::now();
    assert_eq!(
        run_killed(src, 300),
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "the kill ends the run"
    );
    let took = started.elapsed();
    assert!(
        took < Duration::from_secs(5),
        "the kill reached the timed wait at once, not at its 10 s deadline: took {took:?}"
    );
}
