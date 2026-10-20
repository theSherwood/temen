//! **#1414 B6 — the parallel driver starts a run's threads, reads its clock and runs its emitted
//! tier through its host's platform** ([`Host::set_thread_platform`]). The OS's is the default; a
//! browser hands over one that starts each thread as a Web Worker, reads `performance.now()`, and
//! runs a task's tier-ups on its Worker's emitted wasm. These pin that the driver uses whatever it
//! was handed: for every thread it starts, for every tier-up a thread meets, and for stopping
//! emitted code when the run ends.

#[path = "support/rec.rs"]
mod rec;

use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use temen_interp::bytecode::{
    self, EmittedCall, EmittedTarget, Hotness, ModuleOffer, ProgramEmit, ThreadPlatform,
    ThreadStart, ThreadTier,
};
use temen_interp::{ForeignOps, Host, JitValidated, Region, Trap, Value};
use temen_ir::{Module, SpawnRec, ValType};
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

fn counting_spawn(f: ThreadStart, window: Option<u32>) -> std::io::Result<()> {
    STARTED.fetch_add(1, Ordering::SeqCst);
    (ThreadPlatform::OS.spawn)(f, window)
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
        spawn: |_, _| Err(std::io::Error::other("no threads here")),
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
fn bounce_whole_call(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Region { func, .. } = call.target else {
        panic!("only tier-ups reach this tier");
    };
    let mut io = [0i64; 8];
    io[..call.argv.len()].copy_from_slice(call.argv);
    let n = (call.bounce)(func, &mut io, 0)?;
    call.results = io[..n].to_vec();
    Ok(())
}

fn tier(
    eligible: Arc<[bool]>,
    run: fn(&mut EmittedCall<'_>) -> Result<(), Trap>,
) -> ThreadPlatform {
    ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible,
                page_checked: false,
                ..Default::default()
            },
            spill: 0,
            modules: None,
            run,
        }),
        ..ThreadPlatform::OS
    }
}

static BOUNCED: AtomicUsize = AtomicUsize::new(0);

fn counting_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    BOUNCED.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

#[test]
fn each_thread_runs_its_own_tasks_tier_ups() {
    let m = module(FOUR_COUNTERS);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(2000)]));
    assert_eq!(
        run_over(&m, tier(count_eligible(), counting_bounce), flat_window()),
        interpreted
    );
    assert_eq!(
        BOUNCED.load(Ordering::SeqCst),
        4,
        "each thread's call to the eligible function tiered up"
    );
}

static ENTRIES: AtomicUsize = AtomicUsize::new(0);

/// Emitted code for the counters' entry, function 1, counted.
fn counting_entries(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    assert!(
        matches!(call.target, EmittedTarget::Region { module: 0, func: 1 }),
        "the threads' entry alone tiers up"
    );
    ENTRIES.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

/// #1414 ED-1 — a thread whose entry is eligible runs it on its own emitted tier, and the entry's
/// results end the thread.
#[test]
fn each_thread_runs_its_entry_on_its_emitted_tier() {
    let m = module(FOUR_COUNTERS);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    let entries = tier(Arc::from([false, true, false]), counting_entries);
    assert_eq!(run_over(&m, entries, flat_window()), interpreted);
    assert_eq!(
        ENTRIES.load(Ordering::SeqCst),
        4,
        "each thread's entry tiered up"
    );
}

/// The root calls function 1 with 3 and function 2 with 200, 50 times each, and returns what they
/// counted: 50 × (3 + 200) = 10150. Both count to their argument, a taken back edge a step.
const LIGHT_AND_HEAVY_CALLS: &str = "memory 16
func () -> (i64) {
block 0 () {
  vk = i64.const 50
  vz = i64.const 0
  br 1(vk, vz)
}
block 1 (vm: i64, vacc: i64) {
  vzero = i64.const 0
  vdone = i64.eq vm vzero
  br_if vdone 3(vacc) 2(vm, vacc)
}
block 2 (vk2: i64, va: i64) {
  v3 = i64.const 3
  vl = call 1 (v3)
  v200 = i64.const 200
  vh = call 2 (v200)
  vs1 = i64.add va vl
  vs = i64.add vs1 vh
  v1 = i64.const 1
  vk1 = i64.sub vk2 v1
  br 1(vk1, vs)
}
block 3 (vsum: i64) {
  return vsum
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

static HOT_CALLS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

fn recording_hot_calls(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Region { func, .. } = call.target else {
        panic!("only tier-ups reach this tier");
    };
    HOT_CALLS.lock().unwrap().push(func);
    // The work a host reports for the call, the fuel its emitted code burns: each callee here
    // counts to its argument, a back edge a count.
    call.work = call.argv[0] as u64;
    bounce_whole_call(call)
}

/// #2474 — calls tier up on the work they do, on either tier: once a function's calls average 100
/// safepoints over 4 calls, its later calls tier up, and they stay up while the work their host
/// reports keeps that average. The function counting to 200 tiers up after its first four calls;
/// the one counting to 3 never does, however often it is called.
#[test]
fn a_call_tiers_up_once_its_calls_have_done_enough_work() {
    let m = module(LIGHT_AND_HEAVY_CALLS);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(10150)]));
    let mut platform = tier(Arc::from([false, true, true]), recording_hot_calls);
    platform.tier.as_mut().expect("a tier").root.hot = Hotness {
        per_call: 100,
        window: 4,
    };
    assert_eq!(run_over(&m, platform, flat_window()), interpreted);
    let ran = HOT_CALLS.lock().unwrap().clone();
    assert_eq!(ran, [2; 46], "function 2, on its 5th to 50th calls");
}

static SPARSE_TIERUPS: AtomicUsize = AtomicUsize::new(0);

fn sparse_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    SPARSE_TIERUPS.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

#[test]
fn a_window_with_no_flat_span_runs_interpreted() {
    let sparse = Region::sparse(1 << 16).expect("a sparse window");
    let ran = run_over(
        &module(FOUR_COUNTERS),
        tier(count_eligible(), sparse_bounce),
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
        tier(count_eligible(), |_| Err(Trap::DivByZero)),
        flat_window(),
    );
    assert_eq!(ran, Err(Trap::DivByZero));
}

/// #2275 — a memory fault in emitted code reports where it faulted, as an interpreted one does: the
/// address its guard left, which the faulting task's window records for the run.
#[test]
fn a_memory_fault_in_emitted_code_reports_its_address() {
    let ran = run_over(
        &module(FOUR_COUNTERS),
        tier(count_eligible(), |call| {
            call.fault = Some(0x1234);
            Err(Trap::MemoryFault)
        }),
        flat_window(),
    );
    assert_eq!(ran, Err(Trap::MemoryFault));
    assert_eq!(temen_interp::last_capture_fault_addr(), Some(0x1234));
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
fn signal_root(call: &EmittedCall<'_>) {
    // SAFETY: the test's window is a flat 64 KiB span, and 16392 is an aligned word in it.
    let cell = unsafe { &*(call.win.add(16392) as *const AtomicI64) };
    cell.store(1, Ordering::SeqCst);
}

static LOOPED: AtomicUsize = AtomicUsize::new(0);

/// Emitted code for the caller's loop: it answers at once, spending no fuel, so the loop between
/// tier-ups is a few interpreted ops that never spend a quantum. A thread that never looked at the
/// run would call it forever; far past the run's end, it fails the test instead.
fn signal_and_answer(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
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
        tier(count_eligible(), signal_and_answer),
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
fn spin_until_stopped(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
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
        tier(count_eligible(), spin_until_stopped),
        flat_window(),
    );
    assert_eq!(ran, Ok(vec![Value::I64(42)]));
    assert_eq!(SPUN.load(Ordering::SeqCst), 1, "the thread was inside");
}

// ----- emitted §22 invokes (#1414 B6-3b) -------------------------------------------------------

/// The unit every worker invokes: `service() -> 7`.
const SERVICE: &str = "memory 16
func () -> (i32) {
block 0 () {
  v0 = i32.const 7
  return v0
  }
}
";

/// The root, given `(jit, code)`, spawns two workers that each `Jit.invoke` the unit and add what
/// it returns to the cell at 16392, joins them, and returns the cell: 2 × 7 = 14.
const INVOKE_TWICE: &str = "memory 16
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vje = i64.extend_i32_u v0
  vce = i64.extend_i32_u v1
  vc32 = i64.const 32
  vchi = i64.shl vce vc32
  vp = i64.or vchi vje
  vsp = i64.const 0
  t0 = thread.spawn 1 vsp vp
  t1 = thread.spawn 1 vsp vp
  j0 = thread.join t0
  j1 = thread.join t1
  va = i64.const 16392
  vr = i64.atomic.load va
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vp: i64) {
  vmask = i64.const 4294967295
  vjit64 = i64.and vp vmask
  vjit = i32.wrap_i64 vjit64
  vsh = i64.const 32
  vcode = i64.shr_u vp vsh
  vw = call.cap 11 1 (i64) -> (i32) vjit (vcode)
  vw64 = i64.extend_i32_u vw
  va = i64.const 16392
  vold = i64.atomic.rmw.add va vw64
  vret = i64.const 0
  return vret
  }
}
";

/// The host's unit gate: every blob is [`SERVICE`].
fn service_validator(_: &[u8], _: Option<u8>, _: &[u8]) -> Result<JitValidated, i64> {
    let m = parse_module(SERVICE).map_err(|_| -22)?;
    Ok(JitValidated {
        funcs: m.funcs.into(),
        types: m.types.into(),
        data: None,
    })
}

/// A host emitter that emits every unit. A fake tier never runs the bytes, so any will do.
fn emit_any(_: &[u8]) -> Option<Vec<u8>> {
    Some(vec![0])
}

/// Run [`INVOKE_TWICE`] under `platform` over `window`, its unit compiled on a host that emits
/// units when `emit`.
fn run_invokes(
    platform: ThreadPlatform,
    emit: bool,
    window: Option<Arc<Region>>,
) -> Result<Vec<Value>, Trap> {
    let m = module(INVOKE_TWICE);
    let mut host = Host::new();
    host.set_jit_validator(service_validator);
    if emit {
        host.set_jit_wasm_emitter(emit_any);
    }
    let jit = host.grant_jit_with_table(Some(16), 4);
    let code = host
        .jit_compile(jit, b"service")
        .expect("no trap")
        .expect("the unit compiles")
        .handle;
    host.set_thread_platform(platform);
    let mut fuel = 50_000_000;
    bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(jit), Value::I32(code)],
        &mut fuel,
        &[],
        window,
        &mut host,
    )
    .expect("the parallel driver runs the module")
    .0
}

/// A tier whose module tiers nothing up: it runs invokes only.
fn invoke_tier(run: fn(&mut EmittedCall<'_>) -> Result<(), Trap>) -> ThreadPlatform {
    ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: Arc::from([false, false]),
                page_checked: false,
                ..Default::default()
            },
            spill: 0,
            modules: None,
            run,
        }),
        ..ThreadPlatform::OS
    }
}

static INVOKED: AtomicUsize = AtomicUsize::new(0);

/// Emitted code for [`SERVICE`]: what it computes, counted.
fn answer_seven(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Unit {
        params, results, ..
    } = call.target
    else {
        panic!("no function of this guest tiers up");
    };
    assert_eq!((params, results), (&[][..], &[ValType::I32][..]));
    INVOKED.fetch_add(1, Ordering::SeqCst);
    call.results = vec![7];
    Ok(())
}

#[test]
fn each_thread_runs_its_own_tasks_emitted_invokes() {
    let interpreted = run_invokes(ThreadPlatform::OS, true, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(14)]));
    let emitted = run_invokes(invoke_tier(answer_seven), true, flat_window());
    assert_eq!(emitted, interpreted);
    assert_eq!(
        INVOKED.load(Ordering::SeqCst),
        2,
        "each worker's invoke ran on the emitted tier"
    );
}

#[test]
fn a_unit_with_no_emitted_wasm_runs_interpreted() {
    let ran = run_invokes(
        invoke_tier(|_| panic!("a unit with no wasm reached the emitted tier")),
        false,
        flat_window(),
    );
    assert_eq!(ran, Ok(vec![Value::I64(14)]));
}

#[test]
fn an_invoke_over_a_window_with_no_flat_span_runs_interpreted() {
    let sparse = Region::sparse(1 << 16).expect("a sparse window");
    let ran = run_invokes(
        invoke_tier(|_| panic!("emitted code addresses its window as one span")),
        true,
        Some(Arc::new(sparse)),
    );
    assert_eq!(ran, Ok(vec![Value::I64(14)]));
}

#[test]
fn a_trap_in_an_emitted_invoke_traps_its_task() {
    let ran = run_invokes(invoke_tier(|_| Err(Trap::DivByZero)), true, flat_window());
    assert_eq!(ran, Err(Trap::DivByZero));
}

/// The first page past the window's null guard ([`temen_ir::POWERBOX_NULL_GUARD`]).
fn guarded_page() -> u64 {
    temen_ir::POWERBOX_NULL_GUARD.max(temen_interp::host_page_size())
}

/// Run `src` on the parallel driver under a page-checked tier that runs function 2 with `run`, over
/// a flat window, with whole-window memory authority at the handle `{mem}`. `{addr}` is the
/// [`guarded_page`] and `{page}` the page size.
fn run_paged(
    src: &str,
    run: fn(&mut EmittedCall<'_>) -> Result<(), Trap>,
) -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    let mem = host.grant_memory();
    let src = src
        .replace("{mem}", &mem.to_string())
        .replace("{addr}", &guarded_page().to_string())
        .replace("{page}", &temen_interp::host_page_size().to_string());
    host.set_thread_platform(ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: count_eligible(),
                page_checked: true,
                ..Default::default()
            },
            spill: 0,
            modules: None,
            run,
        }),
        ..ThreadPlatform::OS
    });
    let mut fuel = 50_000_000;
    bytecode::compile_and_run_capture_over_parallel_with_host(
        &module(&src),
        0,
        &[],
        &mut fuel,
        &[],
        flat_window(),
        &mut host,
    )
    .expect("the parallel driver runs the module")
    .0
}

/// The page-state table a page-checked call reads now: a byte per page, up to its coverage.
fn pagestate_now(call: &EmittedCall<'_>) -> Vec<u8> {
    let (at, _) = call
        .pagestate()
        .expect("a page-checked call carries a page-state table");
    let pages = (call.mapped_now() / temen_interp::host_page_size()) as usize;
    // SAFETY: the table holds a byte per page of its coverage, and lives for the call.
    unsafe { std::slice::from_raw_parts(at, pages) }.to_vec()
}

/// The root seeds the cell at `{addr}` with 7, protects its page read-only, spawns a thread that
/// reads the cell through function 2, joins it, reads the cell through function 2 itself, and
/// returns the sum.
const PROTECT_THEN_READ: &str = "memory 16
func () -> (i64) {
block 0 () {
  vp = i64.const {addr}
  v7 = i64.const 7
  i64.store vp v7
  vas = i32.const {mem}
  vlen = i64.const {page}
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (vp, vlen, vprot)
  vz = i64.const 0
  t0 = thread.spawn 1 vz vz
  j0 = thread.join t0
  va = call 2 (vp)
  vs = i64.add j0 va
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vp = i64.const {addr}
  vl = call 2 (vp)
  return vl
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
";

static TABLES: std::sync::Mutex<Vec<(Vec<u8>, u64)>> = std::sync::Mutex::new(Vec::new());

fn record_table(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let table = pagestate_now(call);
    TABLES.lock().unwrap().push((table, call.mapped));
    bounce_whole_call(call)
}

#[test]
fn each_page_checked_call_reads_its_tasks_page_map() {
    assert_eq!(
        run_paged(PROTECT_THEN_READ, record_table),
        Ok(vec![Value::I64(14)])
    );
    let tables = TABLES.lock().unwrap();
    assert_eq!(
        tables.len(),
        2,
        "the thread's call and the root's tiered up"
    );
    let page = temen_interp::host_page_size();
    let at = (guarded_page() / page) as usize;
    for (table, mapped) in tables.iter() {
        assert_eq!(table[at], 2, "the protected page is read-only");
        assert_eq!(table[at + 1], 1, "the page after it is read-write");
        assert_eq!(
            *mapped,
            table.len() as u64 * page,
            "the call is bounded by the table's coverage"
        );
    }
}

/// The root seeds the cell at `{addr}` with 7 and reads it through function 2, whose emitted code
/// first calls function 3, which protects its page read-only.
const PROTECT_IN_A_BOUNCE: &str = "memory 16
func () -> (i64) {
block 0 () {
  vp = i64.const {addr}
  v7 = i64.const 7
  i64.store vp v7
  va = call 2 (vp)
  return va
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vl = i64.load v0
  return vl
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.const {mem}
  vlen = i64.const {page}
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (v0, vlen, vprot)
  return vr
  }
}
";

static AROUND_THE_BOUNCE: std::sync::Mutex<Vec<Vec<u8>>> = std::sync::Mutex::new(Vec::new());

fn protect_in_a_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let before = pagestate_now(call);
    let mut io = [0i64; 8];
    io[0] = call.argv[0];
    (call.bounce)(3, &mut io, 0)?;
    let after = pagestate_now(call);
    AROUND_THE_BOUNCE.lock().unwrap().extend([before, after]);
    bounce_whole_call(call)
}

#[test]
fn a_bounce_that_moves_the_page_map_rebuilds_the_calls_table() {
    assert_eq!(
        run_paged(PROTECT_IN_A_BOUNCE, protect_in_a_bounce),
        Ok(vec![Value::I64(7)])
    );
    let around = AROUND_THE_BOUNCE.lock().unwrap();
    assert_eq!(around.len(), 2);
    let at = (guarded_page() / temen_interp::host_page_size()) as usize;
    assert_eq!(around[0][at], 1, "before the bounce the page is read-write");
    assert_eq!(
        around[1][at], 2,
        "after it, the call's table has it read-only"
    );
}

/// A collecting guest: the root spawns 4 threads, and each task calls function 1 with `0x9100`. It
/// derives `0x9108`, which never leaves it, calls the collector (function 2: `gc.roots` over
/// `[0x9000, 0xA000)`), and returns the count plus 8. The tasks' results are summed.
const GC_FROM_THREADS: &str = include_str!("fixtures/threads_gc_spill.temt");

/// Emitted code for function 1 that spills what it holds live across its call to the collector, its
/// argument and the word it derived, then makes that call through its bounce.
fn spill_then_collect(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let (base, words) = call
        .spill()
        .expect("a spilling tier gives each thread a spill stack");
    assert!(words >= 2);
    let v0 = call.argv[0];
    // SAFETY: the thread's spill stack holds `words` words, and nothing else writes it meanwhile.
    unsafe {
        base.write(v0 as u64);
        base.add(1).write(v0 as u64 + 8);
    }
    let mut io = [0i64; 8];
    (call.bounce)(2, &mut io, 2)?;
    call.results = vec![io[0] + 8];
    Ok(())
}

#[test]
fn a_collector_under_each_threads_emitted_frame_sees_its_spilled_words() {
    let m = module(GC_FROM_THREADS);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(
        interpreted,
        Ok(vec![Value::I64(55)]),
        "5 tasks, each counting 3 roots, plus 8"
    );
    let spilling = ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: Arc::from([false, true, false, false]),
                page_checked: false,
                ..Default::default()
            },
            spill: 64,
            modules: None,
            run: spill_then_collect,
        }),
        ..ThreadPlatform::OS
    };
    assert_eq!(run_over(&m, spilling, flat_window()), interpreted);
}

// ----- a detached child's window in the platform's memory (#1414 B6-3c) -----------------------

/// The root spawns the granted module's entry 0, detached in its declared window, by the spawn record
/// at 17536, paid from `budget`. A refused spawn is tried once more. The root returns what the
/// child returned, or the second refusal's errno.
fn spawn_and_join() -> String {
    format!(
        "memory 17
func (i32, i32, i64) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i64) {{
  vma = i64.const 17544
  i32.store vma vmod
  vb = i32.wrap_i64 vbud
  vba = i64.const 17548
  i32.store vba vb
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 1(vinst) 2(vinst, vch)
}}
block 1 (vi: i32) {{
  vrp1 = i64.const 17536
  vch1 = call.cap 6 17 (i64) -> (i32) vi (vrp1)
  vz1 = i32.const 0
  vneg1 = i32.lt_s vch1 vz1
  br_if vneg1 3(vch1) 2(vi, vch1)
}}
block 2 (vi2: i32, vch2: i32) {{
  vst, vr = call.cap 6 1 (i32) -> (i64, i64) vi2 (vch2)
  return vr
}}
block 3 (ve: i32) {{
  vr = i64.extend_i32_s ve
  return vr
  }}
}}
{rec}",
        rec = rec::segment(17536, &SpawnRec::default()),
    )
}

/// The child: it stores 7 at 16400 of its window, and a thread of it stores 35 at 16408. It returns
/// their sum once the thread is done.
const CHILD_AND_THREAD: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 16400
  v7 = i64.const 7
  i64.store va v7
  vz = i64.const 0
  t = thread.spawn 1 vz vz
  j = thread.join t
  vb = i64.const 16408
  vt = i64.load vb
  vs = i64.add v7 vt
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vb = i64.const 16408
  v35 = i64.const 35
  i64.store vb v35
  return v35
  }
}
";

/// Run [`spawn_and_join`] under `platform`, granting it `child` (its program) and a budget of `mem`
/// bytes.
fn run_detached(platform: ThreadPlatform, mem: i64, child: &str) -> Result<Vec<Value>, Trap> {
    run_detached_root(&spawn_and_join(), platform, mem, child)
}

/// Run the root `root`, given `(instantiator, child module, budget)`, under `platform`, granting
/// it `child` (its program) and a budget of `mem` bytes. The root's window is flat, as executor 2's
/// is in a browser: a tier-up over a window without a flat span runs interpreted (#1414 B6-3a), and
/// the driver's own reservation is flat only where it is an `mmap`.
fn run_detached_root(
    root: &str,
    platform: ThreadPlatform,
    mem: i64,
    child: &str,
) -> Result<Vec<Value>, Trap> {
    let m = module(root);
    let size = 1 << m.memory.expect("the root has a window").size_log2;
    let back = Region::owned_zeroed(size, temen_interp::host_page_size()).expect("a flat window");
    let mut host = Host::new();
    host.set_thread_platform(platform);
    let inst = host.grant_instantiator();
    let child = host.grant_module(&module(child));
    let budget = host.grant_budget(-1, mem, -1);
    let args = [
        Value::I32(inst),
        Value::I32(child),
        Value::I64(budget as i64),
    ];
    let mut fuel = 50_000_000;
    bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &args,
        &mut fuel,
        &[],
        Some(Arc::new(back)),
        &mut host,
    )
    .expect("the parallel driver runs the module")
    .0
}

/// The test's own foreign memories, as a browser's are a child's `WebAssembly.Memory`: byte
/// vectors the engine reaches only through [`FAKE_OPS`].
static FAKE_MEMS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

fn fake_read(id: u32, off: u64, out: &mut [u8]) {
    let o = off as usize;
    out.copy_from_slice(&FAKE_MEMS.lock().unwrap()[id as usize][o..o + out.len()]);
}

fn fake_write(id: u32, off: u64, data: &[u8]) {
    let o = off as usize;
    FAKE_MEMS.lock().unwrap()[id as usize][o..o + data.len()].copy_from_slice(data);
}

fn fake_load(id: u32, off: u64, width: u32) -> u64 {
    let mut b = [0u8; 8];
    fake_read(id, off, &mut b[..width as usize]);
    u64::from_le_bytes(b)
}

fn fake_store(id: u32, off: u64, width: u32, val: u64) {
    fake_write(id, off, &val.to_le_bytes()[..width as usize]);
}

fn fake_fill(id: u32, off: u64, len: u64, b: u8) {
    FAKE_MEMS.lock().unwrap()[id as usize][off as usize..(off + len) as usize].fill(b);
}

fn fake_copy(id: u32, dst: u64, src: u64, len: u64) {
    let range = src as usize..(src + len) as usize;
    FAKE_MEMS.lock().unwrap()[id as usize].copy_within(range, dst as usize);
}

/// A load, a store or a compare-exchange: what these guests' runs perform.
fn fake_atomic(id: u32, kind: u32, off: u64, width: u32, a: u64, b: u64) -> u64 {
    let mut mems = FAKE_MEMS.lock().unwrap();
    let cell = &mut mems[id as usize][off as usize..(off + width as u64) as usize];
    let mut raw = [0u8; 8];
    raw[..cell.len()].copy_from_slice(cell);
    let old = u64::from_le_bytes(raw);
    let new = match kind {
        0 => return old,
        1 => a,
        8 if old == a => b,
        8 => return old,
        k => unreachable!("atomic kind {k}: these runs load, store and compare-exchange only"),
    };
    let len = cell.len();
    cell.copy_from_slice(&new.to_le_bytes()[..len]);
    old
}

fn fake_grow(id: u32, len: u64) -> bool {
    let mut mems = FAKE_MEMS.lock().unwrap();
    let m = &mut mems[id as usize];
    if m.len() < len as usize {
        m.resize(len as usize, 0);
    }
    true
}

static FAKE_OPS: ForeignOps = ForeignOps {
    read: fake_read,
    write: fake_write,
    load: fake_load,
    store: fake_store,
    fill: fake_fill,
    copy_within: fake_copy,
    atomic: fake_atomic,
    grow: fake_grow,
};

/// A child window in a fresh foreign memory of the test's own.
fn fake_child_window(size_log2: u8) -> Option<Arc<Region>> {
    let len = 1u64 << size_log2;
    let mut mems = FAKE_MEMS.lock().unwrap();
    mems.push(vec![0; len as usize]);
    let id = mems.len() as u32 - 1;
    Some(Arc::new(Region::foreign(id, len, &FAKE_OPS)))
}

/// The window each thread [`a_detached_childs_window_is_in_the_platforms_memory`]'s run starts was
/// spawned over.
static SPAWNED_OVER: Mutex<Vec<Option<u32>>> = Mutex::new(Vec::new());

fn recording_spawn(f: ThreadStart, window: Option<u32>) -> std::io::Result<()> {
    SPAWNED_OVER.lock().unwrap().push(window);
    (ThreadPlatform::OS.spawn)(f, window)
}

#[test]
fn a_detached_childs_window_is_in_the_platforms_memory() {
    let want = Ok(vec![Value::I64(42)]);
    assert_eq!(
        run_detached(ThreadPlatform::OS, 1 << 20, CHILD_AND_THREAD),
        want,
        "7 + 35, in a window of the engine's"
    );
    let platform = ThreadPlatform {
        spawn: recording_spawn,
        child_window: Some(fake_child_window),
        ..ThreadPlatform::OS
    };
    assert_eq!(run_detached(platform, 1 << 20, CHILD_AND_THREAD), want);
    let spawned = SPAWNED_OVER.lock().unwrap().clone();
    let id = spawned[0].expect("the child's task runs over its memory");
    assert_eq!(
        spawned,
        [Some(id), Some(id)],
        "the child's task and its thread were each spawned over its memory"
    );
    let word = |at: usize| fake_word(id, at);
    assert_eq!(
        (word(16400), word(16408)),
        (7, 35),
        "the child and its thread stored into the platform's memory"
    );
}

static WINDOWS_ASKED: AtomicUsize = AtomicUsize::new(0);

/// A platform that cannot make the first child window it is asked for, and makes the next in the
/// engine's own kind of memory.
fn refuse_the_first_window(size_log2: u8) -> Option<Arc<Region>> {
    match WINDOWS_ASKED.fetch_add(1, Ordering::SeqCst) {
        0 => None,
        _ => Some(Arc::new(
            Region::owned_zeroed(1 << size_log2, temen_interp::host_page_size()).expect("a window"),
        )),
    }
}

#[test]
fn a_child_window_the_platform_cannot_make_refuses_the_spawn_and_charges_nothing() {
    // The budget holds one window. The refused spawn hands it back, so the second spawn finds it.
    let platform = ThreadPlatform {
        child_window: Some(refuse_the_first_window),
        ..ThreadPlatform::OS
    };
    assert_eq!(
        run_detached(platform, 1 << 16, CHILD_AND_THREAD),
        Ok(vec![Value::I64(42)])
    );
    assert_eq!(WINDOWS_ASKED.load(Ordering::SeqCst), 2);
}

/// The word at `at` of the test's foreign memory `id`.
fn fake_word(id: u32, at: usize) -> i64 {
    let mems = FAKE_MEMS.lock().unwrap();
    i64::from_le_bytes(mems[id as usize][at..at + 8].try_into().unwrap())
}

/// #1414 ED-2 — the child: its entry spawns a thread (f1), calls f2 with 7 itself, joins the
/// thread, and returns the sum. The thread calls f2 with 35. f2 stores its argument at `16384 + 8x`
/// of the child's window and returns it.
const CHILD_CALLS_A_LEAF: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  t = thread.spawn 1 vz vz
  v7 = i64.const 7
  va = call 2 (v7)
  j = thread.join t
  vs = i64.add va j
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v35 = i64.const 35
  vr = call 2 (v35)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v8 = i64.const 8
  vo = i64.mul vx v8
  vb = i64.const 16384
  va = i64.add vb vo
  i64.store va vx
  return vx
  }
}
";

/// What each emitted call of [`a_childs_root_task_tiers_up_over_its_own_memory`] named:
/// `(program, function, memory)`.
static CHILD_CALLS: Mutex<Vec<(u32, u32, Option<u32>)>> = Mutex::new(Vec::new());

fn recording_child_calls(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Region { module, func } = call.target else {
        panic!("only tier-ups reach this tier");
    };
    CHILD_CALLS
        .lock()
        .unwrap()
        .push((module, func, call.memory));
    bounce_whole_call(call)
}

#[test]
fn a_childs_root_task_tiers_up_over_its_own_memory() {
    // The host emits the child's program, its leaf f2 alone eligible; the run's own program has
    // nothing eligible.
    let platform = ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: Arc::from([false]),
                page_checked: false,
                ..Default::default()
            },
            spill: 0,
            modules: Some(Arc::new(|_: &ModuleOffer| {
                Some(ProgramEmit {
                    eligible: Arc::from([false, false, true]),
                    page_checked: false,
                    ..Default::default()
                })
            })),
            run: recording_child_calls,
        }),
        child_window: Some(fake_child_window),
        ..ThreadPlatform::OS
    };
    assert_eq!(
        run_detached(platform, 1 << 20, CHILD_CALLS_A_LEAF),
        run_detached(ThreadPlatform::OS, 1 << 20, CHILD_CALLS_A_LEAF),
        "the oracle's 7 + 35"
    );
    let calls = CHILD_CALLS.lock().unwrap().clone();
    let [(module, func, Some(id))] = calls[..] else {
        panic!("one call ran emitted, over a memory of the platform's: {calls:?}");
    };
    assert_eq!(
        (module, func),
        (1, 2),
        "the child's root task tiered up its call to f2, in the child's program; its thread's \
         call interpreted"
    );
    assert_eq!(
        (fake_word(id, 16384 + 8 * 7), fake_word(id, 16384 + 8 * 35)),
        (7, 35),
        "both calls stored into the child's memory"
    );
}

// ----- a scheduler op in a tier-up's bounce (#2443) -------------------------------------------

/// The root calls function 1, which spawns three threads of function 2, joins them, and returns the
/// cell at 16384 that each thread adds 1 to.
const SPAWN_IN_A_CALL: &str = "memory 16
func () -> (i64) {
block 0 () {
  vr = call 1 ()
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  t0 = thread.spawn 2 vz vz
  t1 = thread.spawn 2 vz vz
  t2 = thread.spawn 2 vz vz
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

static SPAWNERS: AtomicUsize = AtomicUsize::new(0);

fn spawner_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    SPAWNERS.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

/// #2443 — a tier-up whose bounce spawns and joins threads. Each spawn and join stops the task from
/// inside the bounce, as the same op stops it interpreted, and the call runs on to its end.
#[test]
fn a_bounce_spawns_and_joins_threads() {
    let m = module(SPAWN_IN_A_CALL);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(3)]));
    let spawner = tier(Arc::from([false, true, false]), spawner_bounce);
    assert_eq!(run_over(&m, spawner, flat_window()), interpreted);
    assert_eq!(
        SPAWNERS.load(Ordering::SeqCst),
        1,
        "the root's call to the spawner ran emitted"
    );
}

/// The root spawns a thread of function 2, calls function 1, which waits until the word at 16384 is
/// set and returns it, and joins the thread. The thread sets the word to 5 and wakes a waiter.
const SIGNAL_ACROSS_BOUNCES: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  t = thread.spawn 2 vz vz
  vr = call 1 ()
  j = thread.join t
  return vr
  }
}
func () -> (i64) {
block 0 () {
  br 1()
}
block 1 () {
  va = i64.const 16384
  vv = i32.atomic.load va
  vz = i32.const 0
  vset = i32.ne vv vz
  br_if vset 2(vv) 3()
}
block 2 (vr: i32) {
  vr64 = i64.extend_i32_s vr
  return vr64
}
block 3 () {
  va2 = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait va2 vexp vto
  br 1()
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  va = i64.const 16384
  v5 = i32.const 5
  i32.atomic.store va v5
  vcnt = i32.const 1
  vw = atomic.notify va vcnt
  vz = i64.const 0
  return vz
  }
}
";

static SIGNALS: Mutex<Vec<u32>> = Mutex::new(Vec::new());

fn recording_signals(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Region { func, .. } = call.target else {
        panic!("only tier-ups reach this tier");
    };
    SIGNALS.lock().unwrap().push(func);
    bounce_whole_call(call)
}

/// #2443 — a wait in one task's bounce, woken by a notify in another's: the waiter's task parks from
/// inside its bounce, and the notifier's bounce wakes it, as the interpreted ops do.
#[test]
fn a_wait_in_a_bounce_is_woken_by_a_notify_in_another() {
    let m = module(SIGNAL_ACROSS_BOUNCES);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(5)]));
    let both = tier(Arc::from([false, true, true]), recording_signals);
    assert_eq!(run_over(&m, both, flat_window()), interpreted);
    let mut ran = SIGNALS.lock().unwrap().clone();
    ran.sort();
    assert_eq!(
        ran,
        [1, 2],
        "the waiter's call and the notifier's entry ran emitted"
    );
}

/// [`spawn_and_join`], its spawn and join in function 1, which the root calls with its arguments.
fn spawn_and_join_in_a_call() -> String {
    spawn_and_join().replacen(
        "memory 17\n",
        "memory 17
func (i32, i32, i64) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i64) {
  vr = call 1 (vinst, vmod, vbud)
  return vr
  }
}
",
        1,
    )
}

static CHILD_BOUNCES: Mutex<Vec<(u32, u32)>> = Mutex::new(Vec::new());

fn recording_child_bounces(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let EmittedTarget::Region { module, func } = call.target else {
        panic!("only tier-ups reach this tier");
    };
    CHILD_BOUNCES.lock().unwrap().push((module, func));
    bounce_whole_call(call)
}

/// #2443 — a tier-up whose bounce spawns a detached child and joins it, the child's entry running
/// emitted too and spawning and joining a thread of its own: each op stops its task from inside its
/// bounce.
#[test]
fn a_bounce_spawns_and_joins_a_child_whose_entry_spawns_a_thread() {
    let platform = ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: Arc::from([false, true]),
                page_checked: false,
                ..Default::default()
            },
            spill: 0,
            modules: Some(Arc::new(|_: &ModuleOffer| {
                Some(ProgramEmit {
                    eligible: Arc::from([true, false]),
                    page_checked: false,
                    ..Default::default()
                })
            })),
            run: recording_child_bounces,
        }),
        child_window: Some(fake_child_window),
        ..ThreadPlatform::OS
    };
    let root = spawn_and_join_in_a_call();
    let interpreted = run_detached_root(&root, ThreadPlatform::OS, 1 << 20, CHILD_AND_THREAD);
    assert_eq!(interpreted, Ok(vec![Value::I64(42)]), "the oracle's 7 + 35");
    assert_eq!(
        run_detached_root(&root, platform, 1 << 20, CHILD_AND_THREAD),
        interpreted
    );
    let mut ran = CHILD_BOUNCES.lock().unwrap().clone();
    ran.sort();
    assert_eq!(
        ran,
        [(0, 1), (1, 0)],
        "the root's spawner and the child's entry ran emitted; the child's thread interpreted"
    );
}

/// The root spawns a thread of function 1, waits until the thread has set the word at 16392, and
/// returns 42 without joining it. The thread sets that word, wakes the root, and waits forever on
/// the word at 16384, which nobody sets.
const ROOT_LEAVES_A_WAITER: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  t = thread.spawn 1 vz vz
  br 1()
}
block 1 () {
  vb = i64.const 16392
  vv = i32.atomic.load vb
  vzero = i32.const 0
  vset = i32.ne vv vzero
  br_if vset 2() 3()
}
block 2 () {
  v42 = i64.const 42
  return v42
}
block 3 () {
  vb2 = i64.const 16392
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vb2 vexp vto
  br 1()
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vb = i64.const 16392
  v1 = i32.const 1
  i32.atomic.store vb v1
  vcnt = i32.const 1
  vw = atomic.notify vb vcnt
  va = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait va vexp vto
  vr = i64.extend_i32_s vst
  return vr
  }
}
";

static WAITERS: AtomicUsize = AtomicUsize::new(0);

fn waiter_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    WAITERS.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

/// #2443 — the run ends while a task waits in its bounce: the task's thread wakes, its bounce traps
/// so the emitted frames unwind, and the run returns its root's result rather than waiting for it.
#[test]
fn the_runs_end_unwinds_a_task_waiting_in_a_bounce() {
    let m = module(ROOT_LEAVES_A_WAITER);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(42)]));
    let waiter = tier(Arc::from([false, true]), waiter_bounce);
    assert_eq!(run_over(&m, waiter, flat_window()), interpreted);
    assert_eq!(
        WAITERS.load(Ordering::SeqCst),
        1,
        "the waiter's entry ran emitted"
    );
}

/// The root calls function 1, which waits forever on a word nobody sets.
const WAIT_FOREVER_IN_A_CALL: &str = "memory 16
func () -> (i64) {
block 0 () {
  vr = call 1 ()
  return vr
  }
}
func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait va vexp vto
  vr = i64.extend_i32_s vst
  return vr
  }
}
";

/// #2443 — a task that waits in its bounce on what nothing can wake is part of the run's deadlock,
/// as an interpreted waiter is: the run ends with the deadlock's verdict.
#[test]
fn a_wait_in_a_bounce_that_nothing_wakes_is_a_deadlock() {
    let m = module(WAIT_FOREVER_IN_A_CALL);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Err(Trap::ThreadFault));
    let waiter = tier(Arc::from([false, true]), bounce_whole_call);
    assert_eq!(run_over(&m, waiter, flat_window()), interpreted);
}

/// The shared fork-join (the browser corpus runs it too, on real emitted code): the root calls
/// function 1, which spawns four threads of function 2, sums `1000..1` itself, waits until a thread
/// has set the word at 16384 and woken it, joins the four, and adds what each returned, its sum of
/// `500..1`: 500500 + 4 × 125250.
const FORK_JOIN_IN_A_CALL: &str = include_str!("fixtures/fork_join_in_a_call.temt");

static FORK_JOINS: AtomicUsize = AtomicUsize::new(0);

fn fork_join_bounce(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    FORK_JOINS.fetch_add(1, Ordering::SeqCst);
    bounce_whole_call(call)
}

/// #2443 — the fork-join with its spawner and its threads' entries on the emitted tier: the
/// spawner's spawns, wait and joins and each thread's notify stop their tasks from inside bounces,
/// and the run matches the interpreted one. Nothing it emits uses an atomic: the emitter keeps
/// those off the tier in a module with threads (#2468).
#[test]
fn a_fork_join_runs_with_its_spawner_and_threads_emitted() {
    let m = module(FORK_JOIN_IN_A_CALL);
    let interpreted = run_over(&m, ThreadPlatform::OS, flat_window());
    assert_eq!(interpreted, Ok(vec![Value::I64(1_001_500)]));
    let emitted = tier(Arc::from([false, true, true]), fork_join_bounce);
    assert_eq!(run_over(&m, emitted, flat_window()), interpreted);
    assert_eq!(
        FORK_JOINS.load(Ordering::SeqCst),
        5,
        "the root's call to the spawner and the four threads' entries ran emitted"
    );
}
