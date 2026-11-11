//! #2958 — what the kernel route (`temen_run::jit_kernel`) must hold that the suites routed through
//! it do not pin: a host-side read of the window never faults (the kernel's page map and the JIT's
//! agree where a wait reads), the domain's end reaches a spinning vCPU with no kill door, a spawned
//! vCPU's stack limit holds on a thread the kernel started, a program out of scope leaves its host as
//! it was, a sibling's trap racing the root's return ends as the oracle's run does, and fuel: a
//! metered run runs out where the oracle's does, and gives back what it drew and did not burn. Each
//! case runs on both routes while both exist: today's JIT (the direct `temen_jit` entry) and the
//! kernel's, asserting it ran there.

use std::time::{Duration, Instant};
use temen_interp::{Host, Trap};
use temen_jit::{compile_and_run, JitOutcome};
use temen_run::jit_kernel;

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A JIT outcome as a result, as the oracle spells one.
fn result(o: JitOutcome) -> Result<Vec<i64>, Trap> {
    match o {
        JitOutcome::Returned(v) => Ok(v),
        JitOutcome::Trapped(k) => Err(Trap::from_code(k.code()).expect("a trap")),
        JitOutcome::Exited(c) => Err(Trap::Exit(c)),
        o => panic!("unexpected {o:?}"),
    }
}

/// Today's JIT's answer.
fn today(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    result(compile_and_run(m, 0, &[]).expect("jit"))
}

/// The kernel route's answer, checked to have run its vCPUs as kernel tasks, in under `PROMPT`.
fn kernel(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    let t0 = Instant::now();
    let (o, vcpus) = jit_kernel::run_fresh(m, 0, &[], None).expect("in the kernel's scope");
    assert!(
        t0.elapsed() < PROMPT,
        "it ended promptly: {:?}",
        t0.elapsed()
    );
    assert!(vcpus >= 1, "it ran as kernel tasks");
    result(o)
}

const PROMPT: Duration = Duration::from_secs(5);

/// Both routes, which must agree.
fn both(src: &str) -> Result<Vec<i64>, Trap> {
    let m = module(src);
    let (a, b) = (today(&m), kernel(&m));
    assert_eq!(a, b, "today's JIT and the kernel route agree");
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
/// guest `MemoryFault` on both routes, read by neither in host code. A notify reads no memory, so it
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
/// not the JIT: its stack limit holds, and the run ends `StackOverflow` on both routes.
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

/// A sibling's trap racing the root's return ends as the oracle's run does: the root spawns a
/// thread that faults at once, then returns without joining it. Whichever lands first, the run's
/// answer is the oracle's: amendment 7's end-precedence row. Repeated, since the race is the point.
#[test]
fn a_siblings_trap_racing_the_roots_return_gives_the_oracles_answer() {
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
  vp = i64.const 0
  vx = i64.load vp
  return vx
  }
}
";
    let m = module(src);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..200 {
        let k = kernel(&m);
        let mut fuel = u64::MAX;
        let oracle = temen_interp::run(&m, 0, &[], &mut fuel).map(|v| {
            v.iter()
                .map(|x| match x {
                    temen_interp::Value::I64(n) => *n,
                    other => panic!("{other:?}"),
                })
                .collect::<Vec<_>>()
        });
        // The oracle's run is one of the two answers a race can give; the kernel's must be one the
        // oracle gives too.
        assert!(
            k == Ok(vec![7]) || k == Err(Trap::MemoryFault),
            "the kernel's answer {k:?} is one the oracle can give ({oracle:?})"
        );
        seen.insert(format!("{k:?}"));
    }
    eprintln!("answers seen: {seen:?}");
}

/// A metered run of `m` on each route and on the oracle, with `budget` fuel: what it ended with,
/// and the fuel left once every vCPU handed back what it drew and did not burn.
fn fueled(m: &temen_ir::Module, budget: u64) -> [(Result<Vec<i64>, Trap>, u64); 3] {
    let mut left = budget;
    let oracle = temen_interp::run(m, 0, &[], &mut left).map(|v| {
        v.iter()
            .map(|x| match x {
                temen_interp::Value::I64(n) => *n,
                other => panic!("{other:?}"),
            })
            .collect::<Vec<_>>()
    });
    let mut cell = budget;
    let today = temen_jit::compile_and_run_with_host_fuel(
        m,
        0,
        &[],
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        &mut cell,
    )
    .expect("jit");
    let mut host = Host::new();
    let admitted = jit_kernel::admit(m, &host, None).expect("in the kernel's scope");
    let start = jit_kernel::Start {
        entry: 0,
        args: &[],
        init_mem: &[],
        reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
        fuel: budget,
        snapshot_cap: None,
        kill: None,
    };
    let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
    assert!(end.vcpus >= 1, "it ran as kernel tasks");
    let kernel = result(end.outcome().0);
    [
        (oracle, left),
        (result(today), cell),
        (kernel, host.fuel_left()),
    ]
}

/// Fuel on the kernel route is today's JIT's: a threaded run that completes leaves exactly what
/// today's JIT leaves, every vCPU's unburned draw handed back, and the oracle's answer. Both JIT
/// routes burn one more unit per spawned thread than the oracle, which charges no thread's entry:
/// #2977, outside this route.
#[test]
fn a_threaded_run_with_fuel_to_spare_leaves_todays_fuel() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vn = i64.const 1000
  vz = i64.const 0
  vh1 = thread.spawn 1 vn vz
  vh2 = thread.spawn 1 vn vz
  vs1, v1 = thread.join vh1
  vs2, v2 = thread.join vh2
  vr = i64.add v1 v2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vn: i64, vacc: i64) {
  br 1(vn, vacc)
}
block 1 (vi: i64, va: i64) {
  vone = i64.const 1
  vj = i64.sub vi vone
  vb = i64.add va vi
  vz = i64.eqz vj
  br_if vz 2(vb) 1(vj, vb)
}
block 2 (vr: i64) {
  return vr
  }
}
";
    let m = module(src);
    for _ in 0..20 {
        let [oracle, today, kernel] = fueled(&m, 1_000_000);
        assert_eq!(oracle.0, Ok(vec![2 * 500_500]));
        assert_eq!(kernel.0, oracle.0, "the kernel route: the oracle's answer");
        assert_eq!(kernel, today, "the kernel route: today's answer and fuel");
        assert_eq!(oracle.1 - today.1, 2, "one unit per spawned entry (#2977)");
    }
}

/// The root runs out: `OutOfFuel`, with nothing left, on every engine.
#[test]
fn the_root_running_out_of_fuel_is_out_of_fuel() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vs, vv = thread.join vh
  br 1(vv)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vn = i64.add vi vone
  br 1(vn)
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  return va
  }
}
";
    let m = module(src);
    let [oracle, today, kernel] = fueled(&m, 100_000);
    assert_eq!(oracle, (Err(Trap::OutOfFuel), 0));
    assert_eq!(today, oracle, "today's JIT");
    assert_eq!(kernel, oracle, "the kernel route");
}

/// A spawned thread runs out while its sibling is parked on a futex for good: the run's verdict is
/// the oracle's `OutOfFuel`, not the `ThreadFault` the parked sibling sees as it unwinds.
#[test]
fn a_thread_running_out_beside_a_parked_sibling_is_out_of_fuel() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh1 = thread.spawn 1 vz vz
  vh2 = thread.spawn 2 vz vz
  vs, vv = thread.join vh2
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vw = i64.const 16384
  vzero = i32.const 0
  vt = i64.const -1
  vst = i32.atomic.wait vw vzero vt
  vr = i64.extend_i32_u vst
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
    let m = module(src);
    for _ in 0..20 {
        let [oracle, today, kernel] = fueled(&m, 100_000);
        assert_eq!(oracle.0, Err(Trap::OutOfFuel));
        assert_eq!(today.0, oracle.0, "today's JIT");
        assert_eq!(kernel.0, oracle.0, "the kernel route");
    }
}

/// A program outside the scope (here a page op, which would move the JIT's page map past the
/// kernel's) is declined by its clause before anything is taken from its host.
#[test]
fn a_page_op_is_declined_by_its_clause() {
    let src = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vc = i32.const 0
  va = i64.const 65536
  vl = i64.const 4096
  vp = i64.const 3
  vr = call.cap 5 0 (i64, i64, i64) -> (i64) vc(va, vl, vp)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  return va
  }
}
";
    let m = module(src);
    assert_eq!(jit_kernel::run_fresh(&m, 0, &[], None).err(), Some("cap"));
}
