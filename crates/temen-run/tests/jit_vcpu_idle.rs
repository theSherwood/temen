//! #2757 slice 3 — **an idle run's parked vCPUs stay asleep**, and the kill door still reaches them.
//!
//! The root spawns two vCPUs and joins the second. The first parks forever in a futex wait nothing
//! notifies; the second sleeps 10 s in a timed wait. Nothing can end those parks but a kill, which
//! the host fires after [`IDLE_MS`]. Before this slice every parked vCPU re-checked the kill cell
//! every 20 ms (`KILL_RECHECK`), so the first vCPU and the joining root each woke ~15 times in the
//! idle window. Now the kill door's store wakes them, and nothing else does.
//!
//! The observable is [`temen_jit::vcpu_wakes`]: one tick each time a parked vCPU returns from its
//! sleep. The run must end killed (`OutOfFuel`) well before func 2's own 10 s deadline, which proves
//! the door's wake reached the parks: without it the run would end only when that wait ran out, and
//! an unwoken indefinite park would hang `join_all`. A watchdog fails the test instead of hanging.
//!
//! Its own test binary (one test, so the process-global counter is uncontended).
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::time::Duration;

use temen_jit::{compile_and_run_with_host_interruptible, JitOutcome, TrapKind};

/// How long the run idles before the host fires the kill door. The old 20 ms re-check would have
/// woken each of the two indefinitely parked vCPUs ~15 times in it, far past [`MAX_WAKES`].
const IDLE_MS: u64 = 300;

/// The most wakes a correct run spends: the kill wakes each of its three parked vCPUs about once,
/// and a condvar may wake spuriously.
const MAX_WAKES: u64 = 6;

/// Spurious condvar wakes the idle window tolerates. The old re-check woke ~30 times in it.
const MAX_IDLE_WAKES: u64 = 2;

/// How long the run gets to compile and park every vCPU before the idle window opens.
const SETTLE: Duration = Duration::from_millis(500);

/// How long the run may take before the test gives up on it, rather than hang.
const GIVE_UP: Duration = Duration::from_secs(60);

/// Root: spawn func 1 and func 2, join func 2, then spin (the kill traps it at the loop's poll).
/// Func 1 parks forever on word 16384. Func 2 sleeps 10 s on word 20480 (never notified), then
/// spins. Each spin is where a woken vCPU meets the kill.
const SRC: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = thread.spawn 1 v0 v0
  v2 = thread.spawn 2 v0 v0
  v3, v4 = thread.join v2
  br 1(v0)
}
block 1 (v5: i64) {
  v6 = i64.const 1
  v7 = i64.add v5 v6
  br 1(v7)
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
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 20480
  v3 = i32.const 0
  v4 = i64.const 10000000000
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

#[test]
fn an_idle_runs_parked_vcpus_wake_only_for_the_kill() {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let interrupt = temen_jit::Interrupt::new();
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let door = std::sync::Arc::clone(&interrupt);
    let run = std::thread::spawn(move || {
        // The guest makes no `call.cap`, so the thunk is never invoked: a null ctx is never read.
        let outcome = compile_and_run_with_host_interruptible(
            &m,
            0,
            &[0],
            temen_run::cap_thunk,
            core::ptr::null_mut(),
            &door,
        )
        .expect("the JIT compiles it");
        let _ = done_tx.send(outcome);
    });

    // Let every vCPU reach its park, then count the wakes across the idle window and the kill.
    std::thread::sleep(SETTLE);
    let before = temen_jit::vcpu_wakes();
    std::thread::sleep(Duration::from_millis(IDLE_MS));
    let idle = temen_jit::vcpu_wakes() - before;
    interrupt.fire();
    let fired = std::time::Instant::now();
    let outcome = done_rx
        .recv_timeout(GIVE_UP)
        .expect("the kill ended the run: a park the door did not wake would hang it");
    let took = fired.elapsed();
    run.join().expect("the run's thread");
    let wakes = temen_jit::vcpu_wakes() - before;

    assert_eq!(outcome, JitOutcome::Trapped(TrapKind::OutOfFuel), "killed");
    assert!(
        idle <= MAX_IDLE_WAKES,
        "parked vCPUs woke {idle} times while nothing could end their parks: a re-check timer is back"
    );
    assert!(
        wakes <= MAX_WAKES,
        "the kill took {wakes} wakes (bound {MAX_WAKES})"
    );
    // The door woke the parks: without it the run would end only when func 2's 10 s wait ran out.
    assert!(
        took < Duration::from_secs(5),
        "the kill reached the parked vCPUs at once, not at a wait's own deadline: took {took:?}"
    );
}
