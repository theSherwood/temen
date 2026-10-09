//! **#1414 B6 — the parallel driver starts a run's threads, reads its clock and runs its emitted
//! tier through its host's platform** ([`Host::set_thread_platform`]). The OS's is the default; a
//! browser hands over one that starts each thread as a Web Worker, reads `performance.now()`, and
//! runs a task's tier-ups on its Worker's emitted wasm. These pin that the driver uses whatever it
//! was handed: for every thread it starts, for every tier-up a thread meets, and for stopping
//! emitted code when the run ends.

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::Arc;

use temen_interp::bytecode::{self, ThreadPlatform, ThreadTier, TierUpCall};
use temen_interp::{Host, Region, Trap, Value};
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

/// Run `m` on the parallel driver under `platform`, over `back` (a backing of the test's choosing),
/// or a window the driver reserves itself.
fn run_over(
    m: &Module,
    platform: ThreadPlatform,
    back: Option<Arc<Region>>,
) -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    host.set_thread_platform(platform);
    let mut fuel = 50_000_000;
    bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &[],
        back,
        &mut host,
    )
    .expect("the parallel driver runs the module")
    .0
}

fn run(m: &Module, platform: ThreadPlatform) -> Result<Vec<Value>, Trap> {
    run_over(m, platform, None)
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
        ..ThreadPlatform::OS
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
        ..ThreadPlatform::OS
    };
    assert_eq!(run(&module(SPAWN_THREE), platform), Err(Trap::ThreadFault));
}

// ----- the emitted tier (#1414 B6-3) ----------------------------------------------------------

/// The root spawns four threads with argument 500, joins them, and returns the cell at 16384. Each
/// thread adds `count(500)` to the cell, a direct call to function 2, the one eligible function,
/// which counts to its argument: 4 × 500 = 2000.
const FOUR_COUNTERS: &str = "memory 16
func () -> (i64) {
block 0 () {
  vn = i64.const 500
  t0 = thread.spawn 1 vn vn
  t1 = thread.spawn 1 vn vn
  t2 = thread.spawn 1 vn vn
  t3 = thread.spawn 1 vn vn
  j0 = thread.join t0
  j1 = thread.join t1
  j2 = thread.join t2
  j3 = thread.join t3
  va = i64.const 16384
  vr = i64.atomic.load va
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vc = call 2 (vn)
  va = i64.const 16384
  vo = i64.atomic.rmw.add va vc
  vz = i64.const 0
  return vz
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v2: i64, v3: i64) {
  v4 = i64.lt_s v3 v2
  br_if v4 2(v2, v3) 3(v3)
}
block 2 (v5: i64, v6: i64) {
  v7 = i64.const 1
  v8 = i64.add v6 v7
  br 1(v5, v8)
}
block 3 (v9: i64) {
  return v9
  }
}
";

/// Function 2 alone runs on the emitted tier.
fn count_eligible() -> Arc<[bool]> {
    Arc::from([false, false, true])
}

/// A window the emitted tier can address on every target: one flat span.
fn flat_window() -> Option<Arc<Region>> {
    Some(Arc::new(
        Region::owned_zeroed(1 << 16, temen_interp::host_page_size()).expect("a flat window"),
    ))
}

/// Emitted code that hands the whole call back to the interpreter through its bounce: a tier-up
/// served end to end on the thread, with the interpreter as the emitted code.
fn bounce_whole_call(call: &mut TierUpCall<'_>) -> Result<(), Trap> {
    let mut io = [0i64; 8];
    io[..call.argv.len()].copy_from_slice(call.argv);
    let n = (call.bounce)(call.func, &mut io)?;
    call.results = io[..n].to_vec();
    Ok(())
}

fn tier(run: fn(&mut TierUpCall<'_>) -> Result<(), Trap>) -> ThreadPlatform {
    ThreadPlatform {
        tier: Some(ThreadTier {
            eligible: count_eligible(),
            cell_bytes: 64,
            run,
        }),
        ..ThreadPlatform::OS
    }
}

static BOUNCED: AtomicUsize = AtomicUsize::new(0);

fn counting_bounce(call: &mut TierUpCall<'_>) -> Result<(), Trap> {
    BOUNCED.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

#[test]
fn each_thread_runs_its_own_tasks_tier_ups() {
    let m = module(FOUR_COUNTERS);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(2000)]));
    assert_eq!(
        run_over(&m, tier(counting_bounce), flat_window()),
        interpreted
    );
    assert_eq!(
        BOUNCED.load(Ordering::SeqCst),
        4,
        "each thread's call to the eligible function tiered up"
    );
}

static SPARSE_TIERUPS: AtomicUsize = AtomicUsize::new(0);

fn sparse_bounce(call: &mut TierUpCall<'_>) -> Result<(), Trap> {
    SPARSE_TIERUPS.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

#[test]
fn a_window_with_no_flat_span_runs_interpreted() {
    let sparse = Region::sparse(1 << 16).expect("a sparse window");
    let ran = run_over(
        &module(FOUR_COUNTERS),
        tier(sparse_bounce),
        Some(Arc::new(sparse)),
    );
    assert_eq!(ran, Ok(vec![Value::I64(2000)]));
    assert_eq!(
        SPARSE_TIERUPS.load(Ordering::SeqCst),
        0,
        "emitted code addresses its window as one span: a sparse one declines"
    );
}

#[test]
fn a_trap_in_emitted_code_traps_its_task() {
    let ran = run_over(
        &module(FOUR_COUNTERS),
        tier(|_| Err(Trap::DivByZero)),
        flat_window(),
    );
    assert_eq!(ran, Err(Trap::DivByZero));
}

/// The root spawns a thread that calls function 2 forever, waits for the cell at 16392 to be set,
/// and returns 42 without joining it.
const ROOT_LEAVES_A_CALLER: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  br 1()
}
block 1 () {
  va = i64.const 16392
  vf = i64.atomic.load va
  vzero = i64.const 0
  vw = i64.eq vf vzero
  br_if vw 1() 2()
}
block 2 () {
  vr = i64.const 42
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
}
block 1 () {
  vn = i64.const 1
  vc = call 2 (vn)
  br 1()
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
";

/// Set the cell at 16392 of the call's window, which the root waits on.
fn signal_root(call: &TierUpCall<'_>) {
    // SAFETY: the test's window is a flat 64 KiB span, and 16392 is an aligned word in it.
    let cell = unsafe { &*(call.win.add(16392) as *const AtomicI64) };
    cell.store(1, Ordering::SeqCst);
}

static LOOPED: AtomicUsize = AtomicUsize::new(0);

/// Emitted code for the caller's loop: it answers at once, spending no fuel, so the loop between
/// tier-ups is a few interpreted ops that never spend a quantum. A thread that never looked at the
/// run would call it forever; far past the run's end, it fails the test instead.
fn signal_and_answer(call: &mut TierUpCall<'_>) -> Result<(), Trap> {
    signal_root(call);
    let calls = LOOPED.fetch_add(1, Ordering::SeqCst);
    assert!(
        calls < 1_000_000,
        "the thread never looked at the run, which ended"
    );
    call.results = vec![call.argv[0]];
    Ok(())
}

#[test]
fn a_thread_that_only_tiers_up_still_sees_the_run_end() {
    let ran = run_over(
        &module(ROOT_LEAVES_A_CALLER),
        tier(signal_and_answer),
        flat_window(),
    );
    assert_eq!(ran, Ok(vec![Value::I64(42)]));
}

/// The root spawns a thread whose leaf, function 2, sets the cell at 16392 and then counts forever;
/// the root waits for the cell and returns 42 without joining it. The browser corpus runs it too,
/// on real emitted code.
const ROOT_LEAVES_A_SPINNER: &str = include_str!("fixtures/root_leaves_a_spinner.temt");

static SPUN: AtomicUsize = AtomicUsize::new(0);

/// Emitted code that never returns on its own: it runs until the run is over, as emitted code's
/// fuel check would see it.
fn spin_until_stopped(call: &mut TierUpCall<'_>) -> Result<(), Trap> {
    SPUN.fetch_add(1, Ordering::SeqCst);
    signal_root(call);
    let since = std::time::Instant::now();
    while !call.over.load(Ordering::SeqCst) {
        // Far past the root's end: the run never stopped this code, and would wait for it forever.
        assert!(
            since.elapsed() < std::time::Duration::from_secs(60),
            "the run's end never stopped the emitted code"
        );
        std::hint::spin_loop();
    }
    Err(Trap::OutOfFuel)
}

#[test]
fn the_runs_end_stops_emitted_code_still_running() {
    let ran = run_over(
        &module(ROOT_LEAVES_A_SPINNER),
        tier(spin_until_stopped),
        flat_window(),
    );
    assert_eq!(ran, Ok(vec![Value::I64(42)]));
    assert_eq!(SPUN.load(Ordering::SeqCst), 1, "the thread was inside");
}
