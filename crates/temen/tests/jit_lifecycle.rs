//! PROCESS.md S3/S1c — §14 child lifecycle ops `poll` (9) / `detach` (10) / `kill` (12) on the **JIT**.
//!
//! A non-durable JIT child now runs on its **own OS thread** (S1c async children), so `poll` reports
//! the live state: `0` (running) while its thread is still executing, then `1` (returned) / `2`
//! (trapped) once it finishes; `kill`/`detach` of a child are harmless successes returning `0`. (The
//! child's thread is joined at run teardown either way.)
//!
//! - `kill_detach_match_interp` is a **cross-backend differential**: `instantiate` → `kill` → `detach`
//!   returns `0` on both engines (no futex/loop, so it stays on the nesting compile path — a program
//!   mixing §14 nesting with §12 `atomic.wait` is a separate, unsupported combination on the JIT).
//! - `jit_poll_reports_child_done` spins `poll` until the async child finishes and pins the terminal
//!   value (`1`, returned) — exercising the running→done transition the OS-thread child now goes
//!   through; poll's interp semantics live in `lifecycle_poll_detach.rs`.

use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

fn run_interp(src: &str) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let ih = h.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut h,
    )
    .0
}

fn run_jit(src: &str) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let jh = h.grant_instantiator(0, 128 << 10);
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        &mut h as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// `instantiate` a child (returns 7), then `kill` and `detach` it; return `kill_status*10 +
/// detach_status` = `0` (both are harmless successes on a finished child). No `poll` loop / futex, so
/// the result is backend-stable: `0` on the interpreter (kill flags the child, detach drops the claim)
/// and `0` on the JIT (the child already ran synchronously).
const KILL_DETACH: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ; spawn via record (op 17): entry=1 off=16384 sl=12 quota=0\n\
  q0v0 = i64.const 4294967296\n\
  q0v1 = i64.const 0\n\
  q0v2 = i64.const -4294967284\n\
  q0v3 = i64.const 4294967295\n\
  q0a0 = i64.const 20480\n\
  i64.store q0a0 q0v0\n\
  q0off = i64.const 16384\n\
  q0a1 = i64.const 20488\n\
  i64.store q0a1 q0off\n\
  q0a2 = i64.const 20496\n\
  i64.store q0a2 q0v2\n\
  q0a3 = i64.const 20504\n\
  i64.store q0a3 q0v3\n\
  q0a4 = i64.const 20512\n\
  i64.store q0a4 q0v1\n\
  q0a5 = i64.const 20520\n\
  i64.store q0a5 q0v1\n\
  q0a6 = i64.const 20528\n\
  i64.store q0a6 q0v1\n\
  vch = call.cap 6 17 (i64) -> (i32) v0 (q0a0)\n\
  vk = call.cap 6 12 (i32) -> (i32) v0 (vch)\n\
  vd = call.cap 6 10 (i32) -> (i32) v0 (vch)\n\
  vten = i32.const 10\n\
  vkm = i32.mul vk vten\n\
  vsum = i32.add vkm vd\n\
  vr = i64.extend_i32_u vsum\n\
  return vr\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

/// `instantiate` a child (returns 7), then **spin `poll` until it finishes** and return the terminal
/// status. The async child runs on its own OS thread, so early polls may report `0` (running); once its
/// thread completes, `poll` reports `1` (returned) and the loop exits. Deterministic (the child always
/// finishes) and exercises the running→done transition of an OS-thread child.
const POLL_DONE: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ; spawn via record (op 17): entry=1 off=16384 sl=12 quota=0\n\
  q1v0 = i64.const 4294967296\n\
  q1v1 = i64.const 0\n\
  q1v2 = i64.const -4294967284\n\
  q1v3 = i64.const 4294967295\n\
  q1a0 = i64.const 20480\n\
  i64.store q1a0 q1v0\n\
  q1off = i64.const 16384\n\
  q1a1 = i64.const 20488\n\
  i64.store q1a1 q1off\n\
  q1a2 = i64.const 20496\n\
  i64.store q1a2 q1v2\n\
  q1a3 = i64.const 20504\n\
  i64.store q1a3 q1v3\n\
  q1a4 = i64.const 20512\n\
  i64.store q1a4 q1v1\n\
  q1a5 = i64.const 20520\n\
  i64.store q1a5 q1v1\n\
  q1a6 = i64.const 20528\n\
  i64.store q1a6 q1v1\n\
  vch = call.cap 6 17 (i64) -> (i32) v0 (q1a0)\n\
  br 1(v0, vch)\n\
}\n\
block 1 (bv0: i32, bvch: i32) {\n\
  vp = call.cap 6 9 (i32) -> (i32) bv0 (bvch)\n\
  vzero = i32.const 0\n\
  vrun = i32.eq vp vzero\n\
  br_if vrun 1(bv0, bvch) 2(vp)\n\
}\n\
block 2 (vpf: i32) {\n\
  vr = i64.extend_i32_u vpf\n\
  return vr\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

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

#[test]
fn jit_poll_reports_child_done() {
    let jo = run_jit(POLL_DONE);
    // The async child runs on its own OS thread; the guest spins `poll` until it finishes, so the
    // terminal value is 1 (returned).
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1]),
        "jit: poll must reach 1 (returned) once the async child finishes, got {jo:?}"
    );
}

/// **Concurrency proof** (S1c): the child runs a long bounded loop before returning `7`, so when the
/// parent — running *concurrently on its own thread* — `poll`s it immediately after `instantiate`, the
/// child is still running (`poll` = 0). The parent records that first poll, then spins `poll` to
/// completion (`1`) and returns `first*10 + final`. The async OS-thread executor yields `0*10 + 1 = 1`;
/// a **synchronous** `instantiate` (child fully run before it returns) would see the child already done
/// at the first poll → `1*10 + 1 = 11`. So `== 1` is a deterministic witness that the child executed
/// concurrently with the parent — the whole point of async children (the substrate for a pipeline).
const POLL_RUNNING: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ; spawn via record (op 17): entry=1 off=16384 sl=12 quota=0\n\
  q2v0 = i64.const 4294967296\n\
  q2v1 = i64.const 0\n\
  q2v2 = i64.const -4294967284\n\
  q2v3 = i64.const 4294967295\n\
  q2a0 = i64.const 20480\n\
  i64.store q2a0 q2v0\n\
  q2off = i64.const 16384\n\
  q2a1 = i64.const 20488\n\
  i64.store q2a1 q2off\n\
  q2a2 = i64.const 20496\n\
  i64.store q2a2 q2v2\n\
  q2a3 = i64.const 20504\n\
  i64.store q2a3 q2v3\n\
  q2a4 = i64.const 20512\n\
  i64.store q2a4 q2v1\n\
  q2a5 = i64.const 20520\n\
  i64.store q2a5 q2v1\n\
  q2a6 = i64.const 20528\n\
  i64.store q2a6 q2v1\n\
  vch = call.cap 6 17 (i64) -> (i32) v0 (q2a0)\n\
  vfirst = call.cap 6 9 (i32) -> (i32) v0 (vch)\n\
  br 1(v0, vch, vfirst)\n\
}\n\
block 1 (bv0: i32, bvch: i32, bfirst: i32) {\n\
  vp = call.cap 6 9 (i32) -> (i32) bv0 (bvch)\n\
  vzero = i32.const 0\n\
  vrun = i32.eq vp vzero\n\
  br_if vrun 1(bv0, bvch, bfirst) 2(bfirst, vp)\n\
}\n\
block 2 (bf: i32, vfin: i32) {\n\
  vten = i32.const 10\n\
  vfm = i32.mul bf vten\n\
  vsum = i32.add vfm vfin\n\
  vr = i64.extend_i32_u vsum\n\
  return vr\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  vz = i64.const 0\n\
  br 1(vz)\n\
}\n\
block 1 (i: i64) {\n\
  vlim = i64.const 20000000\n\
  vlt = i64.lt_u i vlim\n\
  vinc = i64.const 1\n\
  vnext = i64.add i vinc\n\
  br_if vlt 1(vnext) 2()\n\
}\n\
block 2 () {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

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
