//! PROCESS.md S3 — the §14 child lifecycle ops `poll` (9), `detach` (10) and `kill` (12), on the
//! interpreter and the JIT. Each guest spawns its child detached, through a v1 record paid from a
//! `Budget`; a JIT child runs as a task on the run's child executor (D66), concurrently with its
//! parent.
//!
//! - `poll(child) -> 0 running | 1 returned | 2 trapped` is a **WNOHANG** probe (I43): `0` is a valid
//!   answer at any time on any backend, and how many `0`s a caller sees before the child finishes is
//!   scheduling, not semantics — the interpreter's M:N scheduler defers a fresh child, while the JIT
//!   runs it on its own thread. The portable idiom loops `poll`, yielding between probes, until it is
//!   non-zero; the **terminal** value it reaches is the same on both engines.
//! - `detach(child) -> 0` drops the parent's join claim without waiting: the child keeps running
//!   until it ends, or the run does (a child domain ends with the run, DESIGN §23).
//! - `kill` and `detach` of a finished child are harmless successes returning `0`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

/// `src` with its spawn record (func 1, at 20480), and a host for it: the guest's two args, an
/// `Instantiator` and the `Budget` that pays for the child its own module spawns.
fn setup(src: &str) -> (Module, Host, [i32; 2]) {
    let src = format!("{src}{}", rec::segment(20480, &SpawnRec::v1(1)));
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    h.set_self_module(&Arc::new(m.clone()));
    let ih = h.grant_instantiator(0, 128 << 10);
    let bh = h.grant_budget(-1, 1 << 20, -1);
    (m, h, [ih, bh])
}

/// `run` on its own thread with a deadline, so a run that never ends (a child the run waits for)
/// fails the test instead of hanging the binary.
fn bounded<T: Send + 'static>(run: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run());
    });
    rx.recv_timeout(std::time::Duration::from_secs(60))
        .unwrap_or_else(|e| panic!("the run did not finish: {e}"))
}

fn run_interp(src: &str) -> Result<Vec<Value>, temen_interp::Trap> {
    let (m, mut h, args) = setup(src);
    bounded(move || {
        let mut fuel = 50_000_000u64;
        run_capture_reserved_with_host(
            &m,
            0,
            &args.map(Value::I32),
            &mut fuel,
            &[0u8; 128 << 10],
            0,
            &mut h,
        )
        .0
    })
}

fn run_jit(src: &str) -> JitOutcome {
    let (m, mut h, args) = setup(src);
    bounded(move || {
        temen_run::jit_cap_run(
            &m,
            0,
            &args.map(i64::from),
            &MemLayout::image(vec![0u8; 128 << 10]),
            0,
            0,
            &mut h,
            None,
        )
        .expect("jit")
        .0
    })
}

/// The portable poll idiom: spawn the child, `poll` it in a loop — yielding the worker with a short
/// `atomic.wait` on an anonymous byte between probes, so a single-worker pool schedules the child
/// rather than spinning — until `poll != 0`, then `end` the child (`vchc` is its handle) and return
/// the terminal poll status. `child` is func 1's body.
fn poll_loop(end: &str, child: &str) -> String {
    format!(
        "memory 17
func (i32, i32) -> (i64) {{
block 0 (v0: i32, vb: i32) {{
  vrb = i64.const 20508
  i32.store vrb vb
  vrp = i64.const 20480
  vch = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  br 1(v0, vch)
}}
block 1 (v0a: i32, vcha: i32) {{
  vp = call.cap 6 9 (i32) -> (i32) v0a (vcha)
  vz32 = i32.const 0
  vne = i32.ne vp vz32
  br_if vne 3(v0a, vcha, vp) 2(v0a, vcha)
}}
block 2 (v0b: i32, vchb: i32) {{
  vyield = i64.const 24576
  vexp = i32.const 0
  vto = i64.const 100000
  vy = i32.atomic.wait vyield vexp vto
  br 1(v0b, vchb)
}}
block 3 (v0c: i32, vchc: i32, vpf: i32) {{
  {end}
  vpf64 = i64.extend_i32_u vpf
  return vpf64
  }}
}}
func (i64) -> (i64) {{
block 0 (vci: i64) {{
{child}
  }}
}}
"
    )
}

/// A child that returns cleanly: the parent `join`s it once `poll` is non-zero. The terminal status
/// is `1` (returned) on every backend — the interpreter may take several loop turns to get there,
/// the JIT often reaches it on turn one.
#[test]
fn poll_terminal_status_converges_returning_child() {
    let src = poll_loop(
        "vj = call.cap 6 1 (i32) -> (i64) v0c (vchc)",
        "  v7 = i64.const 7\n  return v7",
    );
    let i = run_interp(&src);
    let j = run_jit(&src);
    assert_eq!(i, Ok(vec![Value::I64(1)]), "interp: returned");
    assert!(
        matches!(j, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: returned, as the interpreter; got {j:?}"
    );
}

/// A child that **traps** (the crash-handling shape — a bad command must not kill the shell): it
/// divides by zero; the parent `detach`es it once `poll` is non-zero (never `join`, which would
/// raise the child's trap in the parent). The terminal status is `2` (trapped) on every backend.
#[test]
fn poll_terminal_status_converges_trapping_child() {
    let src = poll_loop(
        "vd = call.cap 6 10 (i32) -> (i32) v0c (vchc)",
        "  va = i64.const 1\n  vb = i64.const 0\n  vd = i64.div_s va vb\n  return vd",
    );
    let i = run_interp(&src);
    let j = run_jit(&src);
    assert_eq!(i, Ok(vec![Value::I64(2)]), "interp: trapped");
    assert!(
        matches!(j, JitOutcome::Returned(ref s) if s == &[2]),
        "jit: trapped, as the interpreter; got {j:?}"
    );
}

/// `poll` of a child that never finishes is `0` (running), and `detach` lets the parent return
/// without waiting for it: the child is still spinning when the run ends, and ends with it.
const POLL_RUNNING_THEN_DETACH: &str = "memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, vb: i32) {
  vrb = i64.const 20508
  i32.store vrb vb
  vrp = i64.const 20480
  vch = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vp = call.cap 6 9 (i32) -> (i32) v0 (vch)
  vd = call.cap 6 10 (i32) -> (i32) v0 (vch)
  vp64 = i64.extend_i32_u vp
  return vp64
  }
}
func (i64) -> (i64) {
block 0 (vci: i64) {
  br 1()
}
block 1 () {
  br 1()
  }
}
";

#[test]
fn poll_running_is_zero_and_detach_does_not_block() {
    assert_eq!(
        run_interp(POLL_RUNNING_THEN_DETACH),
        Ok(vec![Value::I64(0)]),
        "interp: running, detached, the run ended"
    );
    let jo = run_jit(POLL_RUNNING_THEN_DETACH);
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[0]),
        "jit: running, detached, the run ended; got {jo:?}"
    );
}

/// `instantiate` a child (returns 7), then `kill` and `detach` it; return `kill_status*10 +
/// detach_status` = `0` (both are harmless successes on a finished child). No `poll` loop / futex, so
/// the result is backend-stable: `0` on the interpreter (kill flags the child, detach drops the claim)
/// and `0` on the JIT.
const KILL_DETACH: &str = "memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, vb: i32) {
  vrb = i64.const 20508
  i32.store vrb vb
  vrp = i64.const 20480
  vch = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vk = call.cap 6 12 (i32) -> (i32) v0 (vch)
  vd = call.cap 6 10 (i32) -> (i32) v0 (vch)
  vten = i32.const 10
  vkm = i32.mul vk vten
  vsum = i32.add vkm vd
  vr = i64.extend_i32_u vsum
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vci: i64) {
  v7 = i64.const 7
  return v7
  }
}
";

#[test]
fn kill_detach_match_interp() {
    let ir = run_interp(KILL_DETACH);
    let jo = run_jit(KILL_DETACH);
    assert_eq!(ir, Ok(vec![Value::I64(0)]), "interp: kill+detach both 0");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[0]),
        "jit: kill+detach must match interp (0), got {jo:?}"
    );
}

/// **Concurrency proof** (S1c): the child runs a long bounded loop before returning `7`, so when the
/// parent — running *concurrently on its own thread* — `poll`s it immediately after `instantiate`, the
/// child is still running (`poll` = 0). The parent records that first poll, then spins `poll` to
/// completion (`1`) and returns `first*10 + final`. The async OS-thread executor yields `0*10 + 1 = 1`;
/// a **synchronous** `instantiate` (child fully run before it returns) would see the child already done
/// at the first poll → `1*10 + 1 = 11`. So `== 1` is a deterministic witness that the child executed
/// concurrently with the parent — the whole point of async children (the substrate for a pipeline).
const POLL_RUNNING: &str = "memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, vb: i32) {
  vrb = i64.const 20508
  i32.store vrb vb
  vrp = i64.const 20480
  vch = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vfirst = call.cap 6 9 (i32) -> (i32) v0 (vch)
  br 1(v0, vch, vfirst)
}
block 1 (bv0: i32, bvch: i32, bfirst: i32) {
  vp = call.cap 6 9 (i32) -> (i32) bv0 (bvch)
  vzero = i32.const 0
  vrun = i32.eq vp vzero
  br_if vrun 1(bv0, bvch, bfirst) 2(bfirst, vp)
}
block 2 (bf: i32, vfin: i32) {
  vten = i32.const 10
  vfm = i32.mul bf vten
  vsum = i32.add vfm vfin
  vr = i64.extend_i32_u vsum
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vci: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (i: i64) {
  vlim = i64.const 20000000
  vlt = i64.lt_u i vlim
  vinc = i64.const 1
  vnext = i64.add i vinc
  br_if vlt 1(vnext) 2()
}
block 2 () {
  v7 = i64.const 7
  return v7
  }
}
";

#[test]
fn jit_poll_observes_a_concurrently_running_child() {
    let jo = run_jit(POLL_RUNNING);
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: the first poll must see the child still running (0) — proof it runs concurrently on its \
         own thread (async); got {jo:?} (11 would mean the child ran synchronously before instantiate \
         returned)"
    );
}
