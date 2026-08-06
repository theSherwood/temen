//! I43 — the §14 child `poll` (op 9) contract is **WNOHANG**, and its *terminal* status is
//! **backend-portable**.
//!
//! `poll(child) -> 0 running | 1 returned | 2 trapped` is a non-blocking probe: `0` ("still
//! running") is a valid answer at any time on any backend — a caller does not control how many
//! `0`s it sees before the child is scheduled to completion. The two executors differ only in
//! that window's width: the interpreter's M:N scheduler defers a synchronously-spawned child
//! (so an *immediate* poll reads `0`), while the JIT runs it on its own OS thread (so an
//! immediate poll may already read `1`/`2`). That difference is the *defined* semantics of a
//! non-blocking probe, not a divergence to fix — the **portable idiom** is to loop poll (yielding
//! the worker between probes) until it is non-zero, and the **terminal** value that loop reaches
//! is identical across backends. (Making the interpreter "eager" cannot converge the immediate
//! poll for the deterministic single-worker configs — the JIT is a 1:1 OS-thread executor and a
//! single-worker interp run has no thread to run the child ahead of the parent — so the sound
//! contract is WNOHANG + a portable terminal, pinned here rather than a scheduler change.)
//!
//! This differential pins that terminal convergence — previously it lived only in a doc note
//! (`STAGE1.md` "Known caveat" / ISSUES.md I43) and the interp⇄JIT fuzzer never exercised the
//! poll+spawn shape, so the contract was asserted nowhere.

use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

const WIN: usize = 128 << 10;

fn run_interp(src: &str) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let ih = h.grant_instantiator(0, WIN as u64);
    let mut fuel = 50_000_000u64;
    let r =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &[0u8; WIN], 0, &mut h)
            .0
            .expect("interp run");
    match r.as_slice() {
        [Value::I64(x)] => *x,
        other => panic!("unexpected interp result: {other:?}"),
    }
}

fn run_jit(src: &str) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut h = Host::new();
    let jh = h.grant_instantiator(0, WIN as u64);
    let out = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[jh as i64],
        &[0u8; WIN],
        0,
        svm_run::cap_thunk,
        &mut h as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit compile")
    .0;
    match out {
        JitOutcome::Returned(vals) => match vals.as_slice() {
            [x] => *x,
            other => panic!("unexpected jit result: {other:?}"),
        },
        other => panic!("unexpected jit outcome: {other:?}"),
    }
}

/// The **portable poll idiom** over a child that returns cleanly: poll in a loop, yielding the
/// worker with a short `atomic.wait` on an anonymous byte between probes (so a single-worker pool
/// schedules the child rather than spinning), until `poll != 0`; then `join` and return the
/// terminal poll status. The terminal value must be `1` (returned) on every backend — the
/// interpreter may take several loop turns to get there, the JIT often reaches it on turn one, but
/// the value they converge on is the same.
const POLL_LOOP_RETURNS: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ; spawn via record (op 17): entry=1 off=0 sl=12 quota=0\n\
  q0v0 = i64.const 4294967296\n\
  q0v1 = i64.const 0\n\
  q0v2 = i64.const -4294967284\n\
  q0v3 = i64.const 4294967295\n\
  q0a0 = i64.const 4096\n\
  i64.store q0a0 q0v0\n\
  q0a1 = i64.const 4104\n\
  i64.store q0a1 q0v1\n\
  q0a2 = i64.const 4112\n\
  i64.store q0a2 q0v2\n\
  q0a3 = i64.const 4120\n\
  i64.store q0a3 q0v3\n\
  q0a4 = i64.const 4128\n\
  i64.store q0a4 q0v1\n\
  q0a5 = i64.const 4136\n\
  i64.store q0a5 q0v1\n\
  q0a6 = i64.const 4144\n\
  i64.store q0a6 q0v1\n\
  vch = cap.call 6 17 (i64) -> (i32) v0 (q0a0)\n\
  br 1(v0, vch)\n\
}\n\
block 1 (v0a: i32, vcha: i32) {\n\
  vp = cap.call 6 9 (i32) -> (i32) v0a (vcha)\n\
  vz32 = i32.const 0\n\
  vne = i32.ne vp vz32\n\
  br_if vne 3(v0a, vcha, vp) 2(v0a, vcha)\n\
}\n\
block 2 (v0b: i32, vchb: i32) {\n\
  v8192 = i64.const 8192\n\
  vexp = i32.const 0\n\
  vto = i64.const 100000\n\
  vy = i32.atomic.wait v8192 vexp vto\n\
  br 1(v0b, vchb)\n\
}\n\
block 3 (v0c: i32, vchc: i32, vpf: i32) {\n\
  vjr = cap.call 6 1 (i32) -> (i64) v0c (vchc)\n\
  vpf64 = i64.extend_i32_u vpf\n\
  return vpf64\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

/// The portable poll idiom over a child that **traps** (the crash-handling shape — a bad command
/// must not kill the shell): the child divides by zero; the parent poll-loops until `poll != 0`,
/// then **`detach`**s (never `join`, which would propagate the child's trap into the parent) and
/// returns the terminal poll status. It must be `2` (trapped) on every backend.
const POLL_LOOP_TRAPS: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  ; spawn via record (op 17): entry=1 off=0 sl=12 quota=0\n\
  q1v0 = i64.const 4294967296\n\
  q1v1 = i64.const 0\n\
  q1v2 = i64.const -4294967284\n\
  q1v3 = i64.const 4294967295\n\
  q1a0 = i64.const 4160\n\
  i64.store q1a0 q1v0\n\
  q1a1 = i64.const 4168\n\
  i64.store q1a1 q1v1\n\
  q1a2 = i64.const 4176\n\
  i64.store q1a2 q1v2\n\
  q1a3 = i64.const 4184\n\
  i64.store q1a3 q1v3\n\
  q1a4 = i64.const 4192\n\
  i64.store q1a4 q1v1\n\
  q1a5 = i64.const 4200\n\
  i64.store q1a5 q1v1\n\
  q1a6 = i64.const 4208\n\
  i64.store q1a6 q1v1\n\
  vch = cap.call 6 17 (i64) -> (i32) v0 (q1a0)\n\
  br 1(v0, vch)\n\
}\n\
block 1 (v0a: i32, vcha: i32) {\n\
  vp = cap.call 6 9 (i32) -> (i32) v0a (vcha)\n\
  vz32 = i32.const 0\n\
  vne = i32.ne vp vz32\n\
  br_if vne 3(v0a, vcha, vp) 2(v0a, vcha)\n\
}\n\
block 2 (v0b: i32, vchb: i32) {\n\
  v8192 = i64.const 8192\n\
  vexp = i32.const 0\n\
  vto = i64.const 100000\n\
  vy = i32.atomic.wait v8192 vexp vto\n\
  br 1(v0b, vchb)\n\
}\n\
block 3 (v0c: i32, vchc: i32, vpf: i32) {\n\
  vdt = cap.call 6 10 (i32) -> (i32) v0c (vchc)\n\
  vpf64 = i64.extend_i32_u vpf\n\
  return vpf64\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  va = i64.const 1\n\
  vb = i64.const 0\n\
  vd = i64.div_s va vb\n\
  return vd\n\
  }\n\
}\n";

#[test]
fn poll_terminal_status_converges_returning_child() {
    let i = run_interp(POLL_LOOP_RETURNS);
    let j = run_jit(POLL_LOOP_RETURNS);
    assert_eq!(i, 1, "interp terminal poll status is 1 (returned)");
    assert_eq!(j, 1, "jit terminal poll status is 1 (returned)");
    assert_eq!(i, j, "the terminal poll status converges across backends");
}

#[test]
fn poll_terminal_status_converges_trapping_child() {
    let i = run_interp(POLL_LOOP_TRAPS);
    let j = run_jit(POLL_LOOP_TRAPS);
    assert_eq!(i, 2, "interp terminal poll status is 2 (trapped)");
    assert_eq!(j, 2, "jit terminal poll status is 2 (trapped)");
    assert_eq!(i, j, "the terminal poll status converges across backends");
}
