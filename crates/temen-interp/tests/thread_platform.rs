//! **#1414 B6 — the parallel driver starts a run's threads and reads its clock through its host's
//! platform** ([`Host::set_thread_platform`]). The OS's is the default; a browser hands over one
//! that starts each thread as a Web Worker and reads `performance.now()`. These pin that the driver
//! uses whatever it was handed, for every thread it starts, and ends the run when it is refused one.

use std::sync::atomic::{AtomicUsize, Ordering};

use temen_interp::bytecode::{self, ThreadPlatform};
use temen_interp::{Host, Trap, Value};
use temen_ir::Module;
use temen_text::parse_module;

/// The root spawns three threads, each adding 1 to the cell at 16384, joins them, and returns the
/// cell.
const SPAWN_THREE: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  t0 = thread.spawn 1 vz vz
  t1 = thread.spawn 1 vz vz
  t2 = thread.spawn 1 vz vz
  j0 = thread.join t0
  j1 = thread.join t1
  j2 = thread.join t2
  va = i64.const 16384
  vr = i64.atomic.load va
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  va = i64.const 16384
  v1 = i64.const 1
  vo = i64.atomic.rmw.add va v1
  return vo
  }
}
";

fn module(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn run(m: &Module, platform: ThreadPlatform) -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    host.set_thread_platform(platform);
    let mut fuel = 50_000_000;
    bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &[],
        None,
        &mut host,
    )
    .expect("the parallel driver runs the module")
    .0
}

static STARTED: AtomicUsize = AtomicUsize::new(0);
static CLOCK_READS: AtomicUsize = AtomicUsize::new(0);

fn counting_spawn(f: Box<dyn FnOnce() + Send>) -> std::io::Result<()> {
    STARTED.fetch_add(1, Ordering::SeqCst);
    (ThreadPlatform::OS.spawn)(f)
}

fn counting_clock() -> u64 {
    CLOCK_READS.fetch_add(1, Ordering::SeqCst);
    (ThreadPlatform::OS.now_ns)()
}

#[test]
fn every_thread_of_a_run_starts_through_its_hosts_platform() {
    let platform = ThreadPlatform {
        spawn: counting_spawn,
        now_ns: counting_clock,
    };
    assert_eq!(run(&module(SPAWN_THREE), platform), Ok(vec![Value::I64(3)]));
    assert_eq!(
        STARTED.load(Ordering::SeqCst),
        3,
        "one start per spawned task"
    );
    assert!(
        CLOCK_READS.load(Ordering::SeqCst) > 0,
        "the run read its platform's clock"
    );
}

#[test]
fn a_platform_that_refuses_a_thread_ends_the_run() {
    let platform = ThreadPlatform {
        spawn: |_| Err(std::io::Error::other("no threads here")),
        now_ns: ThreadPlatform::OS.now_ns,
    };
    assert_eq!(run(&module(SPAWN_THREE), platform), Err(Trap::ThreadFault));
}
