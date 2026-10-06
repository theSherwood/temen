//! §5 the **fuel/epoch kill-path on the JIT** — a host can stop a *runaway* guest (an infinite
//! loop / unbounded tail recursion), matching the interpreter's `Trap::OutOfFuel`. The interpreter
//! has always bounded execution via its per-step fuel counter; this proves the production backend
//! now has the matching, **guest-undisableable** kill-path: the lowering polls a host-owned
//! interrupt cell at every loop back-edge and function entry and traps `OutOfFuel` the moment the
//! host sets it. Both backends agree on the outcome for a non-terminating program: it terminates,
//! reported as OutOfFuel, rather than hanging the host thread.
//!
//! The differential here is on the **outcome**, not the window/step-count: the interpreter trips on
//! a deterministic fuel budget, the JIT on a wall-clock watchdog (the realistic host mechanism), so
//! they stop at different points — but both must stop, and both must report OutOfFuel.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use temen_interp::{run, Trap, Value};
use temen_ir::SpawnRec;
use temen_jit::{compile_and_run, compile_and_run_with_host_interruptible, JitOutcome, TrapKind};
use temen_run::Limits;
use temen_text::parse_module;
use temen_verify::verify_module;

/// Serialize this binary's tests (the ISSUES.md I4 pattern, applied per I33): every test here
/// races a wall-clock watchdog against deliberately-runaway guest code, and sibling tests
/// competing for the process's cores distort exactly that timing — I33 recorded the runaway-child
/// leg flaking under full-workspace parallel load (twice locally, once on macOS CI, all
/// 2026-07-20) while passing consistently in isolation. A poisoned lock (an earlier test failed)
/// is fine to reuse — take the inner guard.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// A non-terminating **intra-function loop** (block1 branches to itself forever) — caught by the
/// per-back-edge kill-path check.
const INFINITE_LOOP: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

/// A non-terminating **tail-recursion** (function 0 tail-calls itself forever) — runs in O(1)
/// native stack, so it never faults; only the *function-entry* kill-path check can stop it.
const INFINITE_TAIL_RECURSION: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return_call 0(v2)
  }
}
";

/// A **finite** countdown N → 0 — used to prove an *armed-but-never-tripped* run still completes
/// correctly (the poll sees the cell stay zero every iteration and never false-trips).
const FINITE_COUNTDOWN: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  br 1(v0)
}
block 1 (v1: i32) {
  v2 = i32.const -1
  v3 = i32.add v1 v2
  br_if v3 1(v3) 2(v3)
}
block 2 (v4: i32) {
  return v4
  }
}
";

/// Run a non-terminating module on the JIT with the kill-path armed by a watchdog thread that sets
/// the interrupt cell after `delay`. Returns the JIT outcome — which must be `Trapped(OutOfFuel)`,
/// reached within a bounded time of the watchdog firing (else the mechanism is broken and the test
/// would hang — a CI timeout is then the failure signal).
fn jit_with_watchdog(src: &str, delay: Duration) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");

    let interrupt = Arc::new(AtomicU64::new(0));
    let wd = interrupt.clone();
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(delay);
        wd.store(1, Ordering::SeqCst); // request the kill
    });

    // The guest makes no `call.cap`, so the thunk is never invoked — a null ctx is never read.
    let outcome = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[0i64],
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        Arc::as_ptr(&interrupt),
    )
    .expect("jit compiles");
    watchdog.join().unwrap();
    outcome
}

#[test]
fn jit_killpath_stops_infinite_loop() {
    let _serial = serial();
    // Interp: a small fuel budget bounds the infinite loop → OutOfFuel.
    let m = parse_module(INFINITE_LOOP).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    let interp = run(&m, 0, &[Value::I64(0)], &mut fuel);
    assert!(
        matches!(interp, Err(Trap::OutOfFuel)),
        "interp must bound the infinite loop, got {interp:?}"
    );

    // JIT: a watchdog arms the kill-path; the per-back-edge poll trips it → OutOfFuel.
    let jit = jit_with_watchdog(INFINITE_LOOP, Duration::from_millis(100));
    assert_eq!(
        jit,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT must stop the runaway loop with OutOfFuel"
    );
}

#[test]
fn jit_killpath_stops_infinite_tail_recursion() {
    let _serial = serial();
    // Interp: fuel bounds the unbounded tail recursion → OutOfFuel.
    let m = parse_module(INFINITE_TAIL_RECURSION).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 100_000u64;
    let interp = run(&m, 0, &[Value::I64(0)], &mut fuel);
    assert!(
        matches!(interp, Err(Trap::OutOfFuel)),
        "interp must bound the tail recursion, got {interp:?}"
    );

    // JIT: the *function-entry* check (not the back-edge one) is what catches this — tail calls
    // never grow the stack, so without it the guest would spin forever.
    let jit = jit_with_watchdog(INFINITE_TAIL_RECURSION, Duration::from_millis(100));
    assert_eq!(
        jit,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT must stop the runaway tail recursion with OutOfFuel"
    );
}

#[test]
fn jit_armed_finite_run_completes_normally() {
    let _serial = serial();
    // Arm the kill-path on a *finite* program whose watchdog is set far enough out that it never
    // fires before the program finishes: the per-iteration poll sees the cell stay zero and never
    // false-trips, so the run returns its real result. (Also the interp/JIT agree on that result.)
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");

    let mut fuel = 10_000_000u64;
    let interp = run(&m, 0, &[Value::I32(1000)], &mut fuel).expect("interp ok");
    assert_eq!(interp, vec![Value::I32(0)], "countdown returns 0");

    let interrupt = Arc::new(AtomicU64::new(0)); // armed, but we never set it
    let jit = compile_and_run_with_host_interruptible(
        &m,
        0,
        &[1000i64],
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        Arc::as_ptr(&interrupt),
    )
    .expect("jit compiles");
    assert_eq!(
        jit,
        JitOutcome::Returned(vec![0]),
        "an armed-but-untripped finite run must complete normally"
    );
}

#[test]
fn jit_unarmed_path_is_unchanged() {
    let _serial = serial();
    // Sanity: the ordinary (kill-path-not-armed) entry still runs the same finite program to
    // completion — arming is strictly opt-in, so existing call sites are unaffected.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let jit = compile_and_run(&m, 0, &[1000i64]).expect("jit");
    assert_eq!(jit, JitOutcome::Returned(vec![0]));
}

/// A §14 parent (a powerbox `_start`) that spawns func 1 detached through a v1 record at 17408, paid
/// from its `"budget"`, and `join`s it. The child **spins forever**, so the run's kill must reach
/// *into the child*: the host's deadline watchdog sets the run's interrupt cell, the parent's parked
/// `join` traps `OutOfFuel`, and the run's teardown ends the running child with it. Without that, the
/// run waits on the child and hangs.
const PARENT_WITH_RUNAWAY_CHILD: &str = "\
memory 17
export 0 func \"_start\" 0
data 16640 \"instantiator\"
data 16656 \"budget\"
func () -> (i32) {
block 0 () {
  vip = i64.const 16640
  vil = i64.const 12
  vi = self.resolve vip vil
  vbp = i64.const 16656
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vrb = i64.const 17436
  i32.store vrb vb
  vrp = i64.const 17408
  v5 = call.cap 6 17 (i64) -> (i32) vi (vrp)
  v6 = call.cap 6 1 (i32) -> (i64) vi (v5)
  v7 = i32.wrap_i64 v6
  return v7
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

#[test]
fn jit_killpath_stops_runaway_child() {
    let _serial = serial();
    if !temen_jit::fiber_supported() {
        return; // no JIT nesting runtime here — a spawn is an inert CapFault, not a child run
    }
    let src = format!(
        "{PARENT_WITH_RUNAWAY_CHILD}{}",
        rec::segment(17408, &SpawnRec::v1(1))
    );
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let limits = Limits {
        deadline: Some(Duration::from_millis(100)),
        ..Limits::default()
    };
    // The run is on its own thread so a hang fails this test instead of stalling the suite.
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(temen_run::run_powerbox_cfg(&m, b"", &[], &[], limits));
    });
    let err = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the run hung on its runaway child")
        .expect_err("a runaway detached JIT child must be killed, not returned");
    assert!(
        err.contains("OutOfFuel"),
        "expected an OutOfFuel detect-and-kill, got: {err}"
    );
}
