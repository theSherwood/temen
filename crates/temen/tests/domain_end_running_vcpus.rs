//! DESIGN §12 / D37 domain teardown on the Cranelift JIT — **running** vCPUs, not just parked ones.
//!
//! The owner decision (2026-07-24, `os_thread_rt`): the root's completion — a clean return included
//! — and a trap from any vCPU end the whole domain; sibling vCPUs "RUNNING or PARKED" unwind. The
//! JIT woke the *parked* ones (`Domain::begin_teardown`), but a vCPU in a call-free loop polled
//! nothing, so the run's `join_all` waited on it forever while the oracle returned. Each case here
//! pairs a root that finishes with a sibling that never would — a `thread.spawn` thread or a §14
//! child — and asks both engines for the root's answer.

use std::ffi::c_void;
use std::sync::mpsc;
use std::time::Duration;
use temen_interp::{Host, Trap, Value};
use temen_jit::{JitOutcome, Quota, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// A sibling's body: count forever, one back-edge per iteration, no calls.
const SPIN: &str = r#"
block 1 (vk: i64) {
  v1 = i64.const 1
  vk2 = i64.add vk v1
  br 1(vk2)
  }
}
"#;

/// A sibling that never ends without a back-edge: funcs 2 and 3 tail-call each other forever. It
/// polls at the entry of each (the functions that tail-call), where nothing else would stop it.
const TAIL_LOOP: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  return_call 3(v0)
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return_call 2(v2)
  }
}
"#;

/// The root spawns a `thread.spawn` spinner, then runs `tail`.
fn thread_guest(tail: &str) -> String {
    format!(
        "memory 17\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vh = thread.spawn 1 v0 v0\n{tail}\n  }}\n}}\nfunc (i64, i64) -> (i64) {{\nblock 0 (v0: i64, v9: i64) {{\n  br 1(v0)\n}}{SPIN}"
    )
}

/// The root spawns a §14 child (op 0, a 128 KiB carve) that spins, then runs `tail`.
fn child_guest(tail: &str) -> String {
    format!(
        "memory 19\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vi = i32.wrap_i64 v0\n  ve = i64.const 1\n  vo = i64.const 131072\n  vs = i64.const 17\n  vq = i64.const 0\n  vc = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vi (ve, vo, vs, vq)\n{tail}\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  br 1(v0)\n}}{SPIN}"
    )
}

const RETURN_5: &str = "  v5 = i64.const 5\n  return v5";
const TRAP: &str = "  unreachable";

fn module(src: &str) -> temen_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// The root's argument: an `Instantiator` over the whole window (unused by the thread guests).
fn host() -> (Host, i64) {
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 1 << 19);
    (host, ih as i64)
}

fn oracle(src: &str) -> Result<Vec<Value>, Trap> {
    let m = module(src);
    let (mut host, ih) = host();
    let mut fuel = 10_000_000u64;
    temen_interp::run_with_host(&m, 0, &[Value::I64(ih)], &mut fuel, &mut host)
}

unsafe extern "C" fn no_resolver(_t: u32, _o: u32, _na: u32, _nr: u32) -> *const c_void {
    core::ptr::null()
}

/// The JIT's answer, or `None` if the run is still going after `DEADLINE` — a hang. The run is on
/// its own thread so a hang fails this test instead of stalling the suite (the thread leaks).
fn cranelift(src: &str) -> Option<JitOutcome> {
    const DEADLINE: Duration = Duration::from_secs(20);
    let m = module(src);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let (mut host, ih) = host();
        let out = temen_jit::compile_and_run_with_host_fast(
            &m,
            0,
            &[ih],
            temen_run::cap_thunk,
            &mut host as *mut Host as *mut c_void,
            no_resolver,
            Quota {
                max_fibers: 1 << 16,
                max_vcpus: 8,
            },
        )
        .expect("jit compile");
        let _ = tx.send(out);
    });
    rx.recv_timeout(DEADLINE).ok()
}

#[test]
fn a_root_return_ends_a_running_thread() {
    let src = thread_guest(RETURN_5);
    assert_eq!(oracle(&src), Ok(vec![Value::I64(5)]), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running thread");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_trap_ends_a_running_thread() {
    let src = thread_guest(TRAP);
    assert_eq!(oracle(&src), Err(Trap::Unreachable), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running thread");
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::Unreachable)),
        "cranelift: {jit:?}"
    );
}

/// The thread tail-calls around a two-function cycle instead of looping.
#[test]
fn a_root_return_ends_a_thread_in_a_tail_call_cycle() {
    let src = format!(
        "memory 17\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vh = thread.spawn 1 v0 v0\n{RETURN_5}\n  }}\n}}\nfunc (i64, i64) -> (i64) {{\nblock 0 (v0: i64, v9: i64) {{\n  r = call 2 (v0)\n  return r\n  }}\n}}{TAIL_LOOP}"
    );
    assert_eq!(oracle(&src), Ok(vec![Value::I64(5)]), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the tail-calling thread");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_return_ends_a_running_nested_child() {
    let src = child_guest(RETURN_5);
    assert_eq!(oracle(&src), Ok(vec![Value::I64(5)]), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running child");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_trap_ends_a_running_nested_child() {
    let src = child_guest(TRAP);
    assert_eq!(oracle(&src), Err(Trap::Unreachable), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running child");
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::Unreachable)),
        "cranelift: {jit:?}"
    );
}
