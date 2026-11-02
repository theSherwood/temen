//! The bytecode executors must not sleep through a kill that rings the external-wake bell (#1122).
//!
//! The pump's loop top finalizes every task of a killed domain (#1215), and when everything is parked
//! on something an embedder can wake it blocks on the external-wake doorbell (#1122) until the next
//! ring. A kill rings that bell. The pump snapshotted the bell generation *after* its kill sweep, so
//! a kill landing between the two had its ring folded into the snapshot: the pump then blocked until
//! some unrelated ring (the next keystroke) although the domain was already dead. The source below
//! lands the kill at exactly that point — right after the sweep reads `killed()` — so the ordering
//! is deterministic.
//!
//! The parallel driver (executor 2) has no all-parked block: its parked threads sleep until a rule
//! wakes them or their deadline comes. A watcher on the bell settles the run on each ring, so the
//! same kill ends a run whose every task sleeps, at once rather than at the deadline.
//!
//! #2480 — nor does the pump sleep through a kill while a host call is outstanding: its idle wait
//! for the call's answer is a wait on the same bell, which the call's store rings too.
//!
//! #1414 S5a — an embedder's stop ([`bytecode::RunDoor::stop`]) ends a parallel run the same way,
//! and asking never waits for the run: a browser page asks on its main thread, which cannot block.
//! #1414 5b-2 — a wake through the same door hands the thread that pumps the run control back.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;
use temen_interp::{
    bytecode, CapState, Host, HostProcGrant, OffloadOutcome, ProcHandler, SignalSource, StreamRole,
    Trap, Value,
};

/// Answers `killed()` truthfully, but on the `trigger`th query goes dead *after* answering and rings
/// the kill door — a terminate arriving from another thread just after the sweep looked.
struct KilledAfterTheSweep {
    queries: AtomicUsize,
    trigger: usize,
    dead: AtomicBool,
    ring: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl SignalSource for KilledAfterTheSweep {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_kill(&self, kill: Arc<dyn Fn() + Send + Sync>) {
        *self.ring.lock().unwrap() = Some(kill);
    }
    fn killed(&self) -> bool {
        let was = self.dead.load(Ordering::SeqCst);
        if self.queries.fetch_add(1, Ordering::SeqCst) + 1 == self.trigger {
            self.dead.store(true, Ordering::SeqCst);
            if let Some(ring) = self.ring.lock().unwrap().clone() {
                ring();
            }
        }
        was
    }
}

/// The root reads the (empty, blocking) stdin stream: it parks, and the pump — nothing runnable, an
/// externally wakeable park, an armed bell — blocks for a ring.
const READ_STDIN: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vbuf = i64.const 16392
  vcap = i64.const 4
  vr = call.cap 0 0 (i64, i64) -> (i64) v0 (vbuf, vcap)
  return vr
  }
}
"#;

#[test]
fn a_kill_that_lands_after_the_kill_sweep_is_not_slept_through() {
    let m = temen_text::parse_module(READ_STDIN).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        let h = host.grant_stream(StreamRole::In);
        host.set_stdin_blocking(true);
        host.arm_external_wake();
        // Query 1: the root is still runnable. Query 2: the root is parked on stdin — the kill lands
        // right after this sweep, and the pump's next move is the all-parked block.
        let source = Arc::new(KilledAfterTheSweep {
            queries: AtomicUsize::new(0),
            trigger: 2,
            dead: AtomicBool::new(false),
            ring: Mutex::new(None),
        });
        host.set_signal_source(source, Arc::new(AtomicBool::new(false)));
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
        let _ = tx.send(r);
    });
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the pump slept through a kill that rang its bell");
    let r = r.expect("the bytecode engine runs this module");
    assert!(
        r.is_err(),
        "the killed root is finalized, not returned: {r:?}"
    );
}

/// The root sleeps in a 30 s futex wait that nothing notifies.
const SLEEP_30S: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 30000000000
  vst = i32.atomic.wait va vexp vto
  vr = i64.extend_i32_u vst
  return vr
  }
}
"#;

#[test]
fn a_kill_that_rings_the_bell_ends_a_parallel_run_whose_tasks_all_sleep() {
    let m = temen_text::parse_module(SLEEP_30S).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.arm_external_wake();
        // Query 1 is the settle after the root parked in its wait: the kill lands right after it,
        // and nothing but the bell is left to wake the run before the deadline.
        let source = Arc::new(KilledAfterTheSweep {
            queries: AtomicUsize::new(0),
            trigger: 1,
            dead: AtomicBool::new(false),
            ring: Mutex::new(None),
        });
        host.set_signal_source(source, Arc::new(AtomicBool::new(false)));
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &[],
            &mut fuel,
            &[],
            None,
            &mut host,
        );
        let _ = tx.send(r.map(|(r, _image)| r));
    });
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the run slept through a kill that rang its bell");
    let r = r.expect("the parallel driver runs this module");
    assert_eq!(r, Err(Trap::ThreadFault), "the killed root is finalized");
}

/// The root makes a host call that answers only once the test lets it, or after 30 s.
const WAIT_ON_HOST: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vr = call.cap 13 0 (i64) -> (i64) vh (vz)
  return vr
  }
}
"#;

/// #2480 — the root waits on a host call and nothing else can run, so the pump blocks for the call's
/// answer. A kill lands right after the sweep that found the root parked, and rings the bell: the
/// pump must see it at once, not when the call answers. It waited on the call's store alone, deaf to
/// the bell, so the kill waited 30 s for the call.
#[test]
fn a_kill_is_not_slept_through_while_a_call_waits() {
    let m = temen_text::parse_module(WAIT_ON_HOST).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let let_go = Arc::new((Mutex::new(false), Condvar::new()));
    let gate = Arc::clone(&let_go);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        let h = host.grant_host_proc(HostProcGrant::new(
            ProcHandler::Offloadable(Box::new(move |_op, _args| {
                let gate = Arc::clone(&gate);
                OffloadOutcome::Offload(Box::new(move || {
                    let (open, rung) = &*gate;
                    let open = open.lock().unwrap();
                    let wait = rung.wait_timeout_while(open, Duration::from_secs(30), |o| !*o);
                    drop(wait.unwrap());
                    7
                }))
            })),
            CapState::Stateless,
        ));
        host.arm_external_wake();
        // Query 1: the root is still runnable. Query 2: the root is parked on its call — the kill
        // lands right after this sweep, and the pump's next move is the all-parked block.
        let source = Arc::new(KilledAfterTheSweep {
            queries: AtomicUsize::new(0),
            trigger: 2,
            dead: AtomicBool::new(false),
            ring: Mutex::new(None),
        });
        host.set_signal_source(source, Arc::new(AtomicBool::new(false)));
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
        let _ = tx.send(r);
    });
    let r = rx.recv_timeout(Duration::from_secs(10));
    *let_go.0.lock().unwrap() = true;
    let_go.1.notify_all();
    let r = r
        .expect("the pump slept through a kill while a call waited")
        .expect("the bytecode engine runs this module");
    assert!(
        r.is_err(),
        "the killed root is finalized, not returned: {r:?}"
    );
}

/// A thread that spins forever, which the root joins.
const SPIN_THREAD: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  varg = i64.const 0
  vt = thread.spawn 1 vsp varg
  vj_st, vj = thread.join vt
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  br 1(vi2)
  }
}
"#;

/// Run `src` on executor 2 with `stop`, on a thread of its own: its result arrives on the receiver.
fn run_parallel_with_stop(
    src: &str,
    stop: &Arc<bytecode::RunDoor>,
) -> std::sync::mpsc::Receiver<Result<Vec<Value>, Trap>> {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let stop = Arc::clone(stop);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        host.set_run_door(stop);
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &[],
            &mut fuel,
            &[],
            None,
            &mut host,
        );
        let _ = tx.send(r.expect("the parallel driver runs this module").0);
    });
    rx
}

/// Ask until the run is told: a busy lock answers `false`, and asking again is the caller's part.
fn stop_now(stop: &bytecode::RunDoor) {
    while !stop.stop() {
        std::thread::yield_now();
    }
}

#[test]
fn a_stop_ends_a_parallel_run_whose_tasks_all_sleep() {
    let stop = Arc::new(bytecode::RunDoor::default());
    let rx = run_parallel_with_stop(SLEEP_30S, &stop);
    // The root is in its wait (or about to be: a stop asked earlier ends it there too).
    std::thread::sleep(Duration::from_millis(50));
    stop_now(&stop);
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the stopped run slept on to its deadline");
    assert_eq!(
        r,
        Err(Trap::OutOfFuel),
        "a stopped run ends as one whose fuel ran out"
    );
}

#[test]
fn a_stop_ends_a_parallel_run_whose_thread_spins() {
    let stop = Arc::new(bytecode::RunDoor::default());
    let rx = run_parallel_with_stop(SPIN_THREAD, &stop);
    std::thread::sleep(Duration::from_millis(50));
    stop_now(&stop);
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the stopped run spun on");
    assert_eq!(r, Err(Trap::OutOfFuel));
}

/// A stop asked for before the run starts ends it at once; once withdrawn, the next run is its own.
#[test]
fn a_stop_asked_before_the_run_ends_it_and_a_reset_one_does_not() {
    let stop = Arc::new(bytecode::RunDoor::default());
    stop_now(&stop);
    let r = run_parallel_with_stop(SPIN_THREAD, &stop)
        .recv_timeout(Duration::from_secs(10))
        .expect("a run stopped before it started ran on");
    assert_eq!(r, Err(Trap::OutOfFuel));

    stop.reset();
    const RETURNS: &str =
        "memory 16\nfunc () -> (i64) {\nblock 0 () {\n  v = i64.const 7\n  return v\n  }\n}\n";
    let r = run_parallel_with_stop(RETURNS, &stop)
        .recv_timeout(Duration::from_secs(10))
        .expect("the run ends");
    assert_eq!(
        r,
        Ok(vec![Value::I64(7)]),
        "a withdrawn stop leaves the next run be"
    );
}

/// From a thread of the test's own, ask `ask` of `door` until the run hears it.
fn ask_from_a_thread(
    door: &Arc<bytecode::RunDoor>,
    ask: fn(&bytecode::RunDoor) -> bool,
) -> std::thread::JoinHandle<()> {
    let door = Arc::clone(door);
    std::thread::spawn(move || {
        // Most likely while the pumping thread sleeps in its watch, the case the wake is for.
        std::thread::sleep(Duration::from_millis(20));
        while !ask(&door) {
            std::thread::yield_now();
        }
    })
}

/// #1414 5b-2 — a wake through the door hands the thread that pumps a parallel run control back
/// (`Paused`) while the run's tasks go on: the root's thread still spins, so the stop that follows
/// through the same door ends the run.
#[test]
fn a_wake_hands_the_pumping_thread_control_back_while_the_run_goes_on() {
    let m = temen_text::parse_module(SPIN_THREAD).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let door = Arc::new(bytecode::RunDoor::default());
    let (tx, rx) = std::sync::mpsc::channel();
    {
        let door = Arc::clone(&door);
        std::thread::spawn(move || {
            let mut host = Host::new();
            host.set_run_door(door);
            let mut run = bytecode::CoopRun::new_parallel(
                &m,
                0,
                &[],
                u64::MAX,
                host,
                &[],
                temen_ir::default_reserved_log2(),
                None,
                true,
            )
            .expect("the parallel driver runs this module")
            .expect("the run opens");
            let woken = matches!(run.run(), bytecode::CoopEvent::Paused);
            let _ = tx.send(woken);
            let stopped = matches!(run.run(), bytecode::CoopEvent::Trapped(Trap::OutOfFuel));
            let _ = tx.send(stopped);
        });
    }
    ask_from_a_thread(&door, bytecode::RunDoor::wake)
        .join()
        .unwrap();
    let woken = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the pumping thread slept through the wake");
    assert!(woken, "a wake hands the pumping thread control back");
    ask_from_a_thread(&door, bytecode::RunDoor::stop)
        .join()
        .unwrap();
    let stopped = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the stopped run spun on");
    assert!(
        stopped,
        "the run went on past the wake, and the stop ended it"
    );
}
