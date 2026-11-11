//! #2963 — on executor 2 a futex stop that moves only futex task state runs no settle and no wake
//! scan: it wakes the threads of the tasks its notify woke, and nothing else. What a settle would
//! have found besides is the world's doing, which rings the run's bell or wakes a parked thread
//! itself. These pin what that must not lose:
//!
//! - **The deadlock verdict.** A wait can leave nothing able to run, and no settle follows it. The
//!   park that would leave no thread awake to look settles first, so two threads each waiting for
//!   good on the other's word end the run as a deadlock at the second park, and with one of the
//!   waits timed there is no verdict.
//! - **What the world does to a pingpong.** Two threads handing a word back and forth make only
//!   futex stops. A stop through the run's door, and a kill an embedder's signal source rings for,
//!   still end the run at once, not at some later settle: in a run its opener watches, and in one
//!   whose opener runs the root, as a browser's Worker does, which nothing watches until a park
//!   starts a watcher. #2982 — a futex stop skips its settle only while the run's bell has not rung
//!   since the last one, so the kill's ring makes the next stop settle, watched or not, and the
//!   unwatched run takes the fast path until then.
//! - **Between an embedder's pumps.** A run its embedder pumps goes on between the pumps, and then
//!   nothing watches it: a kill rung then ends its threads at their next stop, not at the next
//!   pump. A door stop ends the run itself, pumped or not.
//! - **A door stop asked before the run starts** ends a run whose every stop is a futex stop at
//!   once: the run's open reads it.
//!
//! The tests share [`bytecode::thread_wakes`], a process-wide count, so they take turns.
//!
//! Each run is on a thread of its own, and a test that waits too long fails instead of hanging.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use temen_interp::bytecode::{self, CoopEvent, CoopRun, ThreadPlatform};
use temen_interp::{Host, SignalSource, Trap, Value};

/// One test at a time: [`threads_stop`] reads a count every thread of the process adds to.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// How long a run may take to end before the test gives up on it rather than hang.
const GIVE_UP: Duration = Duration::from_secs(30);

/// How long the world's stop or kill may take to end a pingpong: about one wake of the watcher and
/// the next stop of each thread, far less than this on any machine.
const PROMPT: Duration = Duration::from_secs(5);

/// The root spawns func 1, which waits for good on word 16392, then waits on word 16384 for
/// `timeout` ns (`-1`: for good) and returns what its wait answered. Nothing notifies either word.
fn mutual_wait(timeout: i64) -> String {
    format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vw = i64.const 16384
  vexp = i32.const 0
  vto = i64.const {timeout}
  vst = i32.atomic.wait vw vexp vto
  vst64 = i64.extend_i32_u vst
  return vst64
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vw = i64.const 16392
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vw vexp vto
  vst64 = i64.extend_i32_u vst
  return vst64
  }}
}}
"
    )
}

/// The root and func 1 hand word 16384 back and forth `n` times each (`n` its argument), each
/// waiting for its turn and notifying the other: nothing but futex stops.
const PINGPONG: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vz = i64.const 0
  vn64 = i64.extend_i32_u vn
  vt = thread.spawn 1 vz vn64
  br 1(vn64, vt)
  }
block 1 (vi: i64, vt: i32) {
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vzero = i32.const 0
  vmine = i32.eq vcur vzero
  br_if vmine 3(vi, vt) 2(vi, vt, vcur)
  }
block 2 (vi: i64, vt: i32, vcur: i32) {
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  br 1(vi, vt)
  }
block 3 (vi: i64, vt: i32) {
  vw = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vw vone
  vcnt = i32.const 1
  vk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2, vt) 4(vt)
  }
block 4 (vt: i32) {
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  br 1(vn)
  }
block 1 (vi: i64) {
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vone = i32.const 1
  vmine = i32.eq vcur vone
  br_if vmine 3(vi) 2(vi, vcur)
  }
block 2 (vi: i64, vcur: i32) {
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  br 1(vi)
  }
block 3 (vi: i64) {
  vw = i64.const 16384
  vzero = i32.const 0
  i32.atomic.store vw vzero
  vcnt = i32.const 1
  vk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2) 4()
  }
block 4 () {
  vz = i64.const 0
  return vz
  }
}
"#;

/// Hand-offs enough that no test run finishes them.
const FOREVER: i32 = i32::MAX;

/// Whether the opener watches the run (the native platform), or runs its root, as a browser's
/// Worker does, and nothing watches it.
const OPENERS: [bool; 2] = [false, true];

/// Run `src`'s function 0 with `args` on executor 2, over a host `setup` arms, on a thread of its
/// own: its result arrives on the receiver.
fn run(
    src: &str,
    args: Vec<Value>,
    root_on_opener: bool,
    setup: impl FnOnce(&mut Host) + Send + 'static,
) -> std::sync::mpsc::Receiver<Result<Vec<Value>, Trap>> {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.set_thread_platform(ThreadPlatform {
            root_on_opener,
            ..ThreadPlatform::OS
        });
        setup(&mut host);
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &args,
            &mut fuel,
            &[],
            None,
            &mut host,
        );
        let _ = tx.send(r.expect("executor 2 runs it").0);
    });
    rx
}

/// The second of two untimed waits on words nothing notifies leaves nothing able to run: the run
/// ends a deadlock (`ThreadFault`), as the settle after a wait found it before #2963. With the
/// root's wait timed, its timeout moves the run: the root returns `WAIT_TIMED_OUT` (2), with its
/// thread still waiting, and there is no verdict.
#[test]
fn the_last_park_of_two_mutual_waits_is_the_verdict_and_a_timed_one_moves() {
    let _turn = serial();
    for opener in OPENERS {
        let rx = run(&mutual_wait(-1), Vec::new(), opener, |_| {});
        let r = rx
            .recv_timeout(GIVE_UP)
            .expect("two untimed waits hung the run: no settle saw it go idle");
        assert_eq!(
            r,
            Err(Trap::ThreadFault),
            "a deadlock (opener runs root: {opener})"
        );

        let rx = run(&mutual_wait(10_000_000), Vec::new(), opener, |_| {});
        let r = rx.recv_timeout(GIVE_UP).expect("a timed wait hung the run");
        assert_eq!(
            r,
            Ok(vec![Value::I64(2)]),
            "timed out (opener runs root: {opener})"
        );
    }
}

/// A stop through the run's door ends two threads that make only futex stops at once.
#[test]
fn a_stop_ends_a_futex_pingpong_at_once() {
    let _turn = serial();
    for opener in OPENERS {
        let door = Arc::new(bytecode::RunDoor::default());
        let stop = Arc::clone(&door);
        let rx = run(PINGPONG, vec![Value::I32(FOREVER)], opener, move |h| {
            h.set_run_door(stop)
        });
        std::thread::sleep(Duration::from_millis(100));
        let asked = Instant::now();
        while !door.stop() {
            std::thread::yield_now();
        }
        let r = rx
            .recv_timeout(GIVE_UP)
            .expect("the stopped pingpong ran on");
        let took = asked.elapsed();
        assert_eq!(
            r,
            Err(Trap::OutOfFuel),
            "stopped (opener runs root: {opener})"
        );
        assert!(
            took < PROMPT,
            "the stop took {took:?} (opener runs root: {opener})"
        );
    }
}

/// A signal source whose kill the test lands: once `dead`, it rings the door it was handed.
#[derive(Default)]
struct KillOnDemand {
    dead: AtomicBool,
    ring: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl KillOnDemand {
    fn kill(&self) {
        self.dead.store(true, Ordering::SeqCst);
        let ring = self.ring.lock().unwrap().clone();
        ring.expect("the run wired its kill door")();
    }
}

impl SignalSource for KillOnDemand {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_kill(&self, kill: Arc<dyn Fn() + Send + Sync>) {
        *self.ring.lock().unwrap() = Some(kill);
    }
    fn killed(&self) -> bool {
        self.dead.load(Ordering::SeqCst)
    }
}

/// A kill an embedder's signal source rings for ends two threads that make only futex stops at
/// once: the bell's ring is the run's watcher's to settle, and the next stop of each thread sees its
/// task finalized. Before #2963 the next futex stop's own settle found it.
#[test]
fn a_kill_ends_a_futex_pingpong_at_once() {
    let _turn = serial();
    for opener in OPENERS {
        let source = Arc::new(KillOnDemand::default());
        let src = Arc::clone(&source);
        let rx = run(PINGPONG, vec![Value::I32(FOREVER)], opener, move |h| {
            h.arm_external_wake();
            h.set_signal_source(src, Arc::new(AtomicBool::new(false)));
        });
        std::thread::sleep(Duration::from_millis(100));
        let asked = Instant::now();
        source.kill();
        let r = rx
            .recv_timeout(GIVE_UP)
            .expect("the killed pingpong ran on");
        let took = asked.elapsed();
        assert_eq!(
            r,
            Err(Trap::ThreadFault),
            "finalized (opener runs root: {opener})"
        );
        assert!(
            took < PROMPT,
            "the kill took {took:?} (opener runs root: {opener})"
        );
    }
}

/// Whether the run's threads have stopped waking within [`PROMPT`]: two looks at
/// [`bytecode::thread_wakes`] [`QUIET`] apart that agree. A pingpong wakes thousands of times in
/// that span.
fn threads_stop() -> bool {
    let until = Instant::now() + PROMPT;
    let mut last = bytecode::thread_wakes();
    while Instant::now() < until {
        std::thread::sleep(QUIET);
        let now = bytecode::thread_wakes();
        if now == last {
            return true;
        }
        last = now;
    }
    false
}

/// How long the run's threads must stay asleep for [`threads_stop`] to call them stopped.
const QUIET: Duration = Duration::from_millis(200);

/// An endless pingpong its embedder pumps (`CoopRun::new_parallel`), over a host `setup` arms, on a
/// thread of its own. Its first pump returns once the door's wake hands it control back
/// (`Paused`); it pumps again when the test says so, and sends what that pump returned.
fn pumped(
    setup: impl FnOnce(&mut Host) + Send + 'static,
) -> (
    Arc<bytecode::RunDoor>,
    std::sync::mpsc::Sender<()>,
    std::sync::mpsc::Receiver<CoopEvent>,
) {
    let m = temen_text::parse_module(PINGPONG).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let door = Arc::new(bytecode::RunDoor::default());
    let (go, pump) = std::sync::mpsc::channel::<()>();
    let (tx, rx) = std::sync::mpsc::channel();
    let ours = Arc::clone(&door);
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.set_run_door(ours);
        setup(&mut host);
        let reserved = temen_ir::default_reserved_log2();
        let args = [Value::I32(FOREVER)];
        let mut run =
            CoopRun::new_parallel(&m, 0, &args, u64::MAX, host, &[], reserved, None, true)
                .expect("executor 2 runs it")
                .expect("the run opens");
        let _ = tx.send(run.run());
        if pump.recv().is_ok() {
            let _ = tx.send(run.run());
        }
    });
    // Let the pingpong start, then hand the pumping thread control back.
    std::thread::sleep(Duration::from_millis(100));
    while !door.wake() {
        std::thread::yield_now();
    }
    let first = rx
        .recv_timeout(GIVE_UP)
        .expect("the wake handed control back");
    assert!(
        matches!(first, CoopEvent::Paused),
        "the wake handed back control"
    );
    (door, go, rx)
}

/// #2963 — between an embedder's pumps nothing watches the run, so its futex stops settle: a kill
/// its signal source rings for then ends its threads at once, not at the next pump. Its next pump
/// hears the end.
#[test]
fn a_kill_between_an_embedders_pumps_ends_its_threads_at_once() {
    let _turn = serial();
    let source = Arc::new(KillOnDemand::default());
    let src = Arc::clone(&source);
    let (_door, go, rx) = pumped(move |h| {
        h.arm_external_wake();
        h.set_signal_source(src, Arc::new(AtomicBool::new(false)));
    });
    source.kill();
    assert!(
        threads_stop(),
        "the killed pingpong ran on between pumps: nothing settled on the kill's ring"
    );
    go.send(()).unwrap();
    let end = rx.recv_timeout(GIVE_UP).expect("the next pump returned");
    assert!(
        matches!(end, CoopEvent::Trapped(Trap::ThreadFault)),
        "the kill ended the run"
    );
}

/// A door stop between an embedder's pumps ends the run itself, as it does pumped or not.
#[test]
fn a_stop_between_an_embedders_pumps_ends_its_threads_at_once() {
    let _turn = serial();
    let (door, go, rx) = pumped(|_| {});
    while !door.stop() {
        std::thread::yield_now();
    }
    assert!(threads_stop(), "the stopped pingpong ran on between pumps");
    go.send(()).unwrap();
    let end = rx.recv_timeout(GIVE_UP).expect("the next pump returned");
    assert!(
        matches!(end, CoopEvent::Trapped(Trap::OutOfFuel)),
        "the stop ended the run"
    );
}

/// #2982 — a run whose opener runs the root, as a browser's Worker does, has no watcher until a ring
/// park starts one, and a futex pingpong never parks on a ring. Its futex stops still skip their
/// settles, and the kill's ring makes the next one settle: the kill ends the run at once.
#[test]
fn an_unwatched_pingpong_takes_the_fast_path_and_a_kill_still_ends_it() {
    let _turn = serial();
    let source = Arc::new(KillOnDemand::default());
    let src = Arc::clone(&source);
    let before = bytecode::futex_fast_stops();
    let rx = run(PINGPONG, vec![Value::I32(FOREVER)], true, move |h| {
        h.arm_external_wake();
        h.set_signal_source(src, Arc::new(AtomicBool::new(false)));
    });
    std::thread::sleep(Duration::from_millis(100));
    let fast = bytecode::futex_fast_stops() - before;
    // A release build keeps no count of its fast stops.
    if cfg!(debug_assertions) {
        assert!(
            fast > 100,
            "the unwatched pingpong settled its futex stops: {fast} skipped"
        );
    }
    let asked = Instant::now();
    source.kill();
    let r = rx
        .recv_timeout(GIVE_UP)
        .expect("the killed pingpong ran on");
    let took = asked.elapsed();
    assert_eq!(r, Err(Trap::ThreadFault), "the killed root is finalized");
    assert!(took < PROMPT, "the kill took {took:?}");
}

/// The root waits on word 16384 for 1 while it holds 0, for ever: every stop it makes is a futex
/// stop, the first one included, and none parks.
const NOT_EQUAL_FOREVER: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  br 1()
  }
block 1 () {
  vw = i64.const 16384
  vexp = i32.const 1
  vto = i64.const -1
  vst = i32.atomic.wait vw vexp vto
  br 1()
  }
}
"#;

/// #2982 — a door stop asked before the run starts ends a run whose every stop is a futex stop at
/// once: the run's open reads it, and its first stop settles besides.
#[test]
fn a_stop_asked_before_the_run_starts_ends_a_run_of_futex_stops_at_once() {
    let _turn = serial();
    for opener in OPENERS {
        let door = Arc::new(bytecode::RunDoor::default());
        while !door.stop() {
            std::thread::yield_now();
        }
        let asked = Instant::now();
        let stop = Arc::clone(&door);
        let rx = run(NOT_EQUAL_FOREVER, Vec::new(), opener, move |h| {
            h.set_run_door(stop)
        });
        let r = rx
            .recv_timeout(GIVE_UP)
            .expect("the waits ran on past an early stop");
        let took = asked.elapsed();
        assert_eq!(
            r,
            Err(Trap::OutOfFuel),
            "stopped (opener runs root: {opener})"
        );
        assert!(
            took < PROMPT,
            "the stop took {took:?} (opener runs root: {opener})"
        );
    }
}
