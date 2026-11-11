//! #2757 slice 4 — **an idle run's parked threads stay asleep on executor 2**, and the stop door
//! still reaches them.
//!
//! The root spawns a reader that parks on an empty pipe whose write end the root holds, then sleeps
//! 10 s in a timed wait. Nothing can end those parks but the run's door, which the test stops after
//! [`IDLE_MS`]. Before this slice a pipe park slept 50 µs at a time and looked again
//! (`THREADS_POLL`), so the reader woke about 6,000 times in the idle window. Now a store into the
//! pipe would wake it, and nothing else does.
//!
//! The observable is [`bytecode::thread_wakes`]: one tick each time an executor 2 thread returns
//! from a sleep. The run must end stopped (`OutOfFuel`) well before the root's own 10 s deadline,
//! which proves the stop reached the parks. A watchdog fails the test instead of hanging.
//!
//! Its own test binary (one test, so the process-global counter is uncontended).

use std::time::{Duration, Instant};

use temen_interp::bytecode::{self, RunDoor};
use temen_interp::{Host, Trap};

/// How long the run idles before the test stops it. The old 50 µs poll woke the reader ~6,000 times
/// in it.
const IDLE_MS: u64 = 300;

/// Spurious condvar wakes the idle window tolerates.
const MAX_IDLE_WAKES: u64 = 2;

/// The most wakes the stop costs: each parked thread, and a watcher, wakes about once, and a condvar
/// may wake spuriously.
const MAX_WAKES: u64 = 8;

/// How long the run gets to start and park its tasks before the idle window opens.
const SETTLE: Duration = Duration::from_millis(500);

/// How long the run may take to end once stopped before the test gives up on it, rather than hang.
const GIVE_UP: Duration = Duration::from_secs(60);

/// Root `(r, w)`: spawn a reader of `r`, then sleep 10 s on a word nothing notifies, then write a
/// byte the reader would return. The reader reads one byte of the empty pipe: it parks, the write
/// end still open in the root.
const SRC: &str = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vr: i32, vw: i32) {
  vz = i64.const 0
  vr64 = i64.extend_i32_u vr
  vt = thread.spawn 1 vz vr64
  vword = i64.const 70000
  vzero = i32.const 0
  vns = i64.const 10000000000
  vwoke = i32.atomic.wait vword vzero vns
  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vlen)
  vj_st, vj = thread.join vt
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  return vb64
  }
}
"#;

#[test]
fn an_idle_runs_parked_threads_wake_only_for_the_stop() {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let door = std::sync::Arc::new(RunDoor::default());
    let (done_tx, done_rx) = std::sync::mpsc::channel();
    let stop = std::sync::Arc::clone(&door);
    let run = std::thread::spawn(move || {
        let mut host = Host::new();
        let (w, r) = host.grant_pipe();
        host.set_run_door(stop);
        let mut fuel = 1_000_000_000u64;
        let args = [temen_interp::Value::I32(r), temen_interp::Value::I32(w)];
        let ran = bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &args,
            &mut fuel,
            &[],
            None,
            &mut host,
        )
        .expect("executor 2 runs it");
        let _ = done_tx.send(ran.0);
    });

    // Let both tasks reach their parks, then count the wakes across the idle window and the stop.
    std::thread::sleep(SETTLE);
    let before = bytecode::thread_wakes();
    std::thread::sleep(Duration::from_millis(IDLE_MS));
    let idle = bytecode::thread_wakes() - before;
    while !door.stop() {
        std::thread::yield_now();
    }
    let stopped = Instant::now();
    let ended = done_rx
        .recv_timeout(GIVE_UP)
        .expect("the stop ended the run: a park it did not wake would hang it");
    let took = stopped.elapsed();
    run.join().expect("the run's thread");
    let wakes = bytecode::thread_wakes() - before;

    assert_eq!(ended, Err(Trap::OutOfFuel), "stopped");
    assert!(
        idle <= MAX_IDLE_WAKES,
        "parked threads woke {idle} times while nothing could end their parks: a poll is back"
    );
    assert!(
        wakes <= MAX_WAKES,
        "the stop took {wakes} wakes (bound {MAX_WAKES})"
    );
    // The stop woke the parks: without it the run would end only when the root's 10 s wait ran out.
    assert!(
        took < Duration::from_secs(5),
        "the stop reached the parked threads at once, not at a wait's own deadline: took {took:?}"
    );
}
