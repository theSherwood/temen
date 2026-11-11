//! #2948 spike — what the spike path (`temen_run::jit_kernel`) must hold that the five suites it is
//! routed through do not pin: a host-side read of the window never faults (the kernel's page map
//! and the JIT's agree where a wait reads), the domain's end reaches a spinning vCPU with no kill
//! door, a spawned vCPU's stack limit holds on a thread the kernel started, and a program out of
//! scope leaves its host as it was. Each case runs on both lanes, today's JIT and the spike path,
//! whatever `TEMEN_JIT_KERNEL` says. Scaffolding, deleted with the spike.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]

use std::time::{Duration, Instant};
use temen_interp::{Host, Trap};
use temen_jit::{compile_and_run, JitOutcome};
use temen_run::jit_kernel;

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Today's JIT's answer, as the spike's is spelled.
fn today(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    match compile_and_run(m, 0, &[]).expect("jit") {
        JitOutcome::Returned(v) => Ok(v),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k.code()).expect("a trap")),
        JitOutcome::Exited(c) => Err(Trap::Exit(c)),
        o => panic!("today: {o:?}"),
    }
}

/// The spike path's answer, checked to have run its vCPUs as kernel tasks, in under `PROMPT`.
fn spike(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    let mut host = Host::new();
    let t0 = Instant::now();
    let r = jit_kernel::run(m, 0, &[], temen_ir::DEFAULT_RESERVED_LOG2, &mut host, None)
        .expect("in the spike's scope");
    assert!(
        t0.elapsed() < PROMPT,
        "the spike's run ended promptly: {:?}",
        t0.elapsed()
    );
    assert!(jit_kernel::last_vcpus() >= 1, "it ran as kernel tasks");
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "every vCPU native"
    );
    r
}

const PROMPT: Duration = Duration::from_secs(5);

/// Both lanes, which must agree.
fn both(src: &str) -> Result<Vec<i64>, Trap> {
    let m = module(src);
    let (a, b) = (today(&m), spike(&m));
    assert_eq!(a, b, "today's JIT and the spike path agree");
    b
}

/// A spawned vCPU waits (or notifies) at `addr` in a `memory 16` window; the root joins it and
/// returns the join's value, or ends with what the vCPU's op raised.
fn op_in_thread(op: &str, addr: u64) -> String {
    let body = match op {
        "wait" => format!(
            "  va = i64.const {addr}\n  vz = i32.const 0\n  vt = i64.const 1000000\n  vs = i32.atomic.wait va vz vt\n  vr = i64.extend_i32_u vs\n"
        ),
        _ => format!(
            "  va = i64.const {addr}\n  vc = i32.const 1\n  vs = atomic.notify va vc\n  vr = i64.extend_i32_u vs\n"
        ),
    };
    format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vst, vv = thread.join vh
  return vv
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
{body}  return vr
  }}
}}
"
    )
}

/// The NULL guard is unmapped on both page maps: a wait there, on the root or a spawned vCPU, is a
/// guest `MemoryFault` on both lanes, read by neither in host code. A notify reads no memory, so it
/// answers 0 woken on both.
#[test]
fn a_wait_in_the_null_guard_faults_and_a_notify_there_wakes_none_on_both_lanes() {
    assert_eq!(both(&op_in_thread("wait", 0)), Err(Trap::MemoryFault));
    assert_eq!(both(&op_in_thread("wait", 4096)), Err(Trap::MemoryFault));
    assert_eq!(both(&op_in_thread("notify", 0)), Ok(vec![0]));
}

/// Past the declared window (64 KiB) in the reservation: unmapped on both maps, so a wait faults and
/// a notify wakes none; past the reservation, both fault.
#[test]
fn a_wait_past_the_window_faults_on_both_lanes() {
    assert_eq!(both(&op_in_thread("wait", 65_536)), Err(Trap::MemoryFault));
    assert_eq!(both(&op_in_thread("wait", 1 << 20)), Err(Trap::MemoryFault));
    assert_eq!(both(&op_in_thread("notify", 65_536)), Ok(vec![0]));
}

/// The domain's end reaches a vCPU that spins forever, with no kill door armed and no fuel: the root
/// returns, the end is raised, and the spinner stops at its next back-edge poll. The run ends with
/// the root's value, promptly, instead of waiting on the spinner.
#[test]
fn the_roots_return_stops_a_spinning_sibling_without_a_kill_door() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vr = i64.const 7
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  br 1(va)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vn = i64.add vi vone
  br 1(vn)
  }
}
";
    assert_eq!(both(src), Ok(vec![7]));
}

/// Unbounded recursion on a spawned vCPU, whose thread the kernel started (`ThreadPlatform::OS`),
/// not the JIT: its stack limit holds, and the run ends `StackOverflow` on both lanes.
#[test]
fn unbounded_recursion_on_a_spawned_vcpu_overflows_on_both_lanes() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vst, vv = thread.join vh
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vr = call 2(va)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vn: i64) {
  vone = i64.const 1
  vm = i64.add vn vone
  vr = call 2(vm)
  vs = i64.add vr vone
  return vs
  }
}
";
    assert_eq!(both(src), Err(Trap::StackOverflow));
}

/// A program outside the scope (here a page op, which would move the JIT's page map past the
/// kernel's) is declined before anything is taken: `None`, and the host as it was.
#[test]
fn a_page_op_is_declined_with_the_host_left_as_it_was() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 0
  va = i64.const 65536
  vl = i64.const 4096
  vp = i64.const 3
  vr = call.cap 5 0 (i64, i64, i64) -> (i64) vh(va, vl, vp)
  return vr
  }
}
";
    let m = module(src);
    assert_eq!(
        temen_interp::bytecode::native_scope(&m, &Host::new()),
        Err("cap")
    );
    let mut host = Host::new();
    host.set_jit_table_log2(7);
    let r = jit_kernel::run(&m, 0, &[], temen_ir::DEFAULT_RESERVED_LOG2, &mut host, None);
    assert!(r.is_none(), "declined");
    assert_eq!(host.jit_table_log2(), 7, "the host is the caller's still");
}
