//! PROCESS.md S3 — §14 child **lifecycle** ops `poll` (op 9) and `detach` (op 10) on the interpreter.
//!
//! - `poll(child) -> 0 running | 1 returned | 2 trapped` — never parks; the reap probe a shell loops
//!   for `WNOHANG`/`SIGCHLD`. Non-destructive: a later `join` still delivers the result.
//! - `detach(child) -> 0` — drop the parent's join claim; the child runs to completion on its own
//!   (detach is not kill), the parent never blocks on it.
//!
//! (`kill` needs a per-child §5 interrupt on the M:N executor — a follow-up. Interp-first, like the
//! rest of the §14 substrate.)

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Run `src`'s entry 0 with an `Instantiator` over the whole 128 KiB window.
fn run(src: &str) -> Result<Vec<Value>, Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    )
    .0
}

/// `poll` of a **blocked** child is `0` (running); `detach` lets the parent finish without joining.
///
/// func 0 (parent): spawn func 1 in a 4 KiB carve at offset 0; `poll` it (must be `0` — the child
/// spins until window byte 0 becomes 1, which is still 0, so it cannot have finished); `detach` it;
/// set byte 0 = 1 to release it; return the poll status. The child then runs to completion detached —
/// if `detach` blocked or the run waited wrong, this would hang.
const POLL_RUNNING_THEN_DETACH: &str = "memory 17\n\
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
  vp = call.cap 6 9 (i32) -> (i32) v0 (vch)\n\
  vd = call.cap 6 10 (i32) -> (i32) v0 (vch)\n\
  vz = i64.const 16384\n\
  vone = i32.const 1\n\
  i32.store8 vz vone\n\
  vp64 = i64.extend_i32_u vp\n\
  return vp64\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (vci: i64) {\n\
  br 1()\n\
}\n\
block 1 () {\n\
  vz = i64.const 0\n\
  vb = i32.load8_u vz\n\
  v1 = i32.const 1\n\
  veq = i32.eq vb v1\n\
  br_if veq 2() 1()\n\
}\n\
block 2 () {\n\
  v7 = i64.const 7\n\
  return v7\n\
  }\n\
}\n";

/// `poll` reaches `1` (returned) for a child that finishes. The parent polls in a loop, **yielding**
/// the worker between probes with a short `atomic.wait` on an anonymous byte (so a single-worker pool
/// schedules the child instead of spinning forever); once `poll != 0` it `join`s and returns the poll
/// status, which must be `1`.
const POLL_RETURNED: &str = "memory 17\n\
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
block 1 (v0a: i32, vcha: i32) {\n\
  vp = call.cap 6 9 (i32) -> (i32) v0a (vcha)\n\
  vz32 = i32.const 0\n\
  vne = i32.ne vp vz32\n\
  br_if vne 3(v0a, vcha, vp) 2(v0a, vcha)\n\
}\n\
block 2 (v0b: i32, vchb: i32) {\n\
  v8192 = i64.const 24576\n\
  vexp = i32.const 0\n\
  vto = i64.const 100000\n\
  vy = i32.atomic.wait v8192 vexp vto\n\
  br 1(v0b, vchb)\n\
}\n\
block 3 (v0c: i32, vchc: i32, vpf: i32) {\n\
  vjr = call.cap 6 1 (i32) -> (i64) v0c (vchc)\n\
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

#[test]
fn poll_running_is_zero_and_detach_does_not_block() {
    // The parent must observe the still-blocked child as running (0), detach it, and finish — with
    // the detached child completing on its own (no hang).
    assert_eq!(run(POLL_RUNNING_THEN_DETACH), Ok(vec![Value::I64(0)]));
}

#[test]
fn poll_reaches_returned() {
    // The yielding reap loop must see the child transition to returned (1), then join it cleanly.
    assert_eq!(run(POLL_RETURNED), Ok(vec![Value::I64(1)]));
}
