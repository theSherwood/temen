//! **Blocking-stdin park**: the persistent-session seam. With [`Host::set_stdin_blocking`] on, a `read`
//! on an exhausted stdin buffer must *suspend* the run instead of returning EOF, and resume, re-issuing
//! the same read, once the host pushes more bytes: no bytes lost, the read re-executes, and the guest
//! never sees a spurious EOF. Two engines suspend this way:
//! - the pump's suspend/resume session ([`CoopRun::set_suspend_on_idle`]), which the browser Postgres
//!   console runs on (#1414 S4-2): the pump returns [`CoopEvent::Idle`];
//! - the resumable [`bytecode::Vcpu`]: it returns [`VcpuEvent::StdinPark`].

use temen_interp::bytecode::{self, CoopEvent, CoopRun, VcpuEvent};
use temen_interp::{Host, StreamRole, Value, DEFAULT_FUEL};
use temen_text::parse_module;

// A guest that forever: read(stdin, buf, 64) → write(stdout, buf, n) → repeat. `_start(vin, vout)`
// takes the two stream handles positionally. Reading an empty stdin under blocking mode parks it.
const ECHO: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (vin: i32, vout: i32) {
  br 1(vin, vout)
}
block 1 (vin1: i32, vout1: i32) {
  vptr = i64.const 17408
  vlen = i64.const 64
  vn = call.cap 0 0 (i64, i64) -> (i64) vin1 (vptr, vlen)
  vw = call.cap 0 1 (i64, i64) -> (i64) vout1 (vptr, vn)
  br 1(vin1, vout1)
  }
}
"#;

/// The echo guest's session on the pump: every read on an empty stdin leaves nothing to run, and the
/// pump returns `Idle`. Feeding stdin and pumping again re-admits the reader.
#[test]
fn stdin_read_idles_the_pump_and_resumes() {
    let m = parse_module(ECHO).expect("parse echo guest");
    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    let vout = host.grant_stream(StreamRole::Out);
    host.set_stdin_blocking(true);
    host.push_stdin(b"abc"); // preload the first chunk

    let slots = [Value::I32(vin), Value::I32(vout)];
    // An engine-backed reservation, as the console's session has (small, for the test).
    let mut run = CoopRun::new_reserved(&m, 0, &slots, DEFAULT_FUEL, host, None, &[], 20)
        .expect("the echo guest is in the engine's subset")
        .expect("build the run");
    run.set_suspend_on_idle(true);

    // Echoes "abc", then the next read finds the buffer empty and parks: not EOF, not an exit.
    assert!(
        matches!(run.run(), CoopEvent::Idle),
        "an exhausted blocking stdin must idle the pump, not return EOF",
    );
    assert_eq!(
        run.host_mut().stdout,
        b"abc",
        "the preloaded chunk was echoed"
    );

    // Push more and pump: the parked read re-issues, echoes "de", and parks again.
    run.host_mut().push_stdin(b"de");
    assert!(matches!(run.run(), CoopEvent::Idle));
    assert_eq!(
        run.host_mut().stdout,
        b"abcde",
        "the re-admitted read took the new bytes"
    );

    // And again: a REPL parks after every prompt.
    run.host_mut().push_stdin(b"f");
    assert!(matches!(run.run(), CoopEvent::Idle));
    assert_eq!(run.host_mut().stdout, b"abcdef");
}

#[test]
fn stdin_read_parks_and_resumes() {
    let m = parse_module(ECHO).expect("parse echo guest");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile echo guest");

    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    let vout = host.grant_stream(StreamRole::Out);
    host.set_stdin_blocking(true);
    host.push_stdin(b"abc"); // preload the first chunk

    let slots = [Value::I32(vin), Value::I32(vout)];
    // Engine-backed reservation, the resumable twin of the one-shot reserved run (small reserve for the test).
    let mut vcpu = bytecode::Vcpu::new_root_reserved_with_powerbox(&prog, 0, &slots, &[], host, 20)
        .expect("build root vcpu");

    // First run: echoes "abc", then the next read finds the buffer empty and parks (not EOF/exit).
    assert!(
        matches!(vcpu.run(), VcpuEvent::StdinPark),
        "an exhausted blocking stdin must park the vCPU, not return EOF",
    );
    assert_eq!(
        vcpu.host_mut().stdout,
        b"abc",
        "the preloaded chunk was echoed"
    );

    // Push more and resume: the parked read re-issues, sees "de", echoes it, parks again — nothing lost.
    vcpu.push_stdin(b"de");
    assert!(matches!(vcpu.run(), VcpuEvent::StdinPark));
    assert_eq!(
        vcpu.host_mut().stdout,
        b"abcde",
        "resumed read appended the new bytes"
    );

    // A second push/resume proves the loop keeps parking (a real REPL parks after every prompt).
    vcpu.push_stdin(b"f");
    assert!(matches!(vcpu.run(), VcpuEvent::StdinPark));
    assert_eq!(vcpu.host_mut().stdout, b"abcdef");
}

// Sanity: with blocking OFF (the default — the one-shot `temen_run_pg` / oracle path), an exhausted read
// returns EOF and the guest runs to completion, unchanged. Here the guest reads once and returns `n`.
const READ_ONCE: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vin: i32) {
  vptr = i64.const 17408
  vlen = i64.const 64
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vptr, vlen)
  return vn
  }
}
"#;

#[test]
fn non_blocking_stdin_returns_eof_at_end() {
    let m = parse_module(READ_ONCE).expect("parse read-once guest");
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let mut host = Host::new();
    let vin = host.grant_stream(StreamRole::In);
    // stdin_block defaults to false; buffer is empty → read returns 0 (EOF), the guest returns 0.
    let slots = [Value::I32(vin)];
    let mut vcpu = bytecode::Vcpu::new_root_reserved_with_powerbox(&prog, 0, &slots, &[], host, 20)
        .expect("build vcpu");
    match vcpu.run() {
        VcpuEvent::Done(vals) => {
            assert_eq!(vals.first(), Some(&Value::I64(0)), "empty read = EOF = 0")
        }
        _ => panic!("expected Done(0) with blocking off (empty read should be EOF, not a park)"),
    }
}
