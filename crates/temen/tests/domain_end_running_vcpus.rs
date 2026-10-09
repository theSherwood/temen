//! DESIGN §12 / D37 domain teardown on the Cranelift JIT — **running** vCPUs, not just parked ones.
//!
//! The owner decision (2026-07-24, `os_thread_rt`): the root's completion — a clean return included
//! — and a trap from any vCPU end the whole domain; sibling vCPUs "RUNNING or PARKED" unwind. The
//! JIT woke the *parked* ones (`Domain::begin_teardown`), but a vCPU in a call-free loop polled
//! nothing, so the run's `join_all` waited on it forever while the oracle returned. Each case here
//! pairs a root that finishes with a sibling that never would — a `thread.spawn` thread or a §14
//! child — and asks both engines for the root's answer.
//!
//! The oracle's fuel outlasts the deadline, so its run ends because the sibling stopped at its next
//! safepoint, the end of its preemption quantum. With a small budget, a sibling spinning on forever
//! still looks right: it runs out of fuel and the run ends. The tree-walker passed that way while it
//! re-queued a running sibling indefinitely (#2202).

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::mpsc;
use std::time::Duration;
use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{SpawnRec, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// A sibling's body from the end of its entry block: count forever, one back-edge per iteration, no
/// calls.
const SPIN: &str = r#"  br 1(v0)
}
block 1 (vk: i64) {
  v1 = i64.const 1
  vk2 = i64.add vk v1
  br 1(vk2)
  }
}
"#;

/// A sibling that never ends without a back-edge: funcs 2 and 3 tail-call each other forever. It
/// polls at the entry of each (the functions that tail-call), where nothing else would stop it.
const TAIL_LOOP: &str = r#"  r = call 2 (v0)
  return r
  }
}
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

/// Where a spawned sibling says it is running.
const UP: u64 = 20480;

/// The root spawns a `thread.spawn` sibling, waits until it is running, then runs `tail`. The
/// sibling says it is up, then runs `body`. The wait is what makes the sibling a *running* one: a
/// sibling still queued when the root finishes dies with the queue, and never reaches the safepoint
/// these cases are about.
fn thread_guest(body: &str, tail: &str) -> String {
    format!(
        "memory 17\nfunc (i64, i64, i64) -> (i64) {{\nblock 0 (v0: i64, vb: i64, vm: i64) {{\n  vh = thread.spawn 1 v0 v0\n  br 1()\n  }}\nblock 1 () {{\n  vu = i64.const {UP}\n  vup = i32.atomic.load vu\n  br_if vup 2() 1()\n  }}\nblock 2 () {{\n{tail}\n  }}\n}}\nfunc (i64, i64) -> (i64) {{\nblock 0 (v0: i64, v9: i64) {{\n  vu = i64.const {UP}\n  vone = i32.const 1\n  i32.atomic.store vu vone\n{body}"
    )
}

/// The root spawns a §14 child that spins (its func 1's child image, #2219), detached and paid from
/// its `Budget` (a v1 record at 17408), then runs `tail`.
fn child_guest(tail: &str) -> String {
    format!(
        "memory 19\nfunc (i64, i64, i64) -> (i64) {{\nblock 0 (v0: i64, vb: i64, vm: i64) {{\n  vi = i32.wrap_i64 v0\n  vbb = i32.wrap_i64 vb\n  vab = i64.const 17436\n  i32.store vab vbb\n  vmm = i32.wrap_i64 vm\n  vam = i64.const 17432\n  i32.store vam vmm\n  vp = i64.const 17408\n  vc = call.cap 6 17 (i64) -> (i32) vi (vp)\n{tail}\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n{SPIN}{}",
        rec::segment(17408, &SpawnRec::v1(0))
    )
}

const RETURN_5: &str = "  v5 = i64.const 5\n  return v5";
const TRAP: &str = "  unreachable";

fn module(src: &str) -> temen_ir::Module {
    let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// The root's arguments: an `Instantiator` over the whole window, the `Budget` that pays for a child,
/// and the child, func 1's child image (all three unused by the thread guests).
fn host(m: &temen_ir::Module) -> (Host, [i64; 3]) {
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 1 << 19);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let child = host.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    (host, [ih, budget, child].map(i64::from))
}

/// `run`'s answer, or `None` if it is still going after `DEADLINE` — a hang. It runs on its own
/// thread so a hang fails this test instead of stalling the suite (the thread leaks).
fn by_deadline<T: Send + 'static>(run: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    const DEADLINE: Duration = Duration::from_secs(20);
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run());
    });
    rx.recv_timeout(DEADLINE).ok()
}

/// The tree-walker's answer, or `None` on a hang.
fn oracle(src: &str) -> Option<Result<Vec<Value>, Trap>> {
    let m = module(src);
    by_deadline(move || {
        let (mut host, args) = host(&m);
        let mut fuel = 1u64 << 50;
        temen_interp::run_with_host(&m, 0, &args.map(Value::I64), &mut fuel, &mut host)
    })
}

/// The JIT's answer, or `None` on a hang.
fn cranelift(src: &str) -> Option<JitOutcome> {
    let m = module(src);
    by_deadline(move || {
        let (mut host, args) = host(&m);
        temen_run::jit_cap_run(
            &m,
            0,
            &args,
            &MemLayout::image(Vec::new()),
            DEFAULT_RESERVED_LOG2,
            0,
            &mut host,
            None,
        )
        .expect("jit run")
        .0
    })
}

#[test]
fn a_root_return_ends_a_running_thread() {
    let src = thread_guest(SPIN, RETURN_5);
    assert_eq!(oracle(&src), Some(Ok(vec![Value::I64(5)])), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running thread");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_trap_ends_a_running_thread() {
    let src = thread_guest(SPIN, TRAP);
    assert_eq!(oracle(&src), Some(Err(Trap::Unreachable)), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running thread");
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::Unreachable)),
        "cranelift: {jit:?}"
    );
}

/// The thread tail-calls around a two-function cycle instead of looping.
#[test]
fn a_root_return_ends_a_thread_in_a_tail_call_cycle() {
    let src = thread_guest(TAIL_LOOP, RETURN_5);
    assert_eq!(oracle(&src), Some(Ok(vec![Value::I64(5)])), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the tail-calling thread");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_return_ends_a_running_nested_child() {
    let src = child_guest(RETURN_5);
    assert_eq!(oracle(&src), Some(Ok(vec![Value::I64(5)])), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running child");
    assert!(
        matches!(jit, JitOutcome::Returned(ref v) if v == &[5]),
        "cranelift: {jit:?}"
    );
}

#[test]
fn a_root_trap_ends_a_running_nested_child() {
    let src = child_guest(TRAP);
    assert_eq!(oracle(&src), Some(Err(Trap::Unreachable)), "oracle");
    let jit = cranelift(&src).expect("cranelift hung on the running child");
    assert!(
        matches!(jit, JitOutcome::Trapped(TrapKind::Unreachable)),
        "cranelift: {jit:?}"
    );
}
