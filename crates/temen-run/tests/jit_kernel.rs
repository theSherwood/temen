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
    let admitted = jit_kernel::admit(m, &host, None, false).expect("in the kernel's scope");
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

/// Fuel on the kernel route is exact: a threaded run that completes leaves exactly the oracle's
/// fuel, every vCPU's unburned draw handed back, but for one unit per spawned thread, which both JIT
/// routes burn and the oracle does not (#2977): the allowance is spelled out, so fixing #2977 fails
/// this test until it is taken away. (Today's JIT is no reference here: its threads share one fixed
/// cell, whose plain decrements race when they run at once.)
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
        assert_eq!(today.0, oracle.0, "today's JIT: the oracle's answer");
        assert_eq!(kernel.0, oracle.0, "the kernel route: the oracle's answer");
        assert_eq!(oracle.1 - kernel.1, 2, "one unit per spawned entry (#2977)");
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

/// Eight threads counting 1,000 each, joined and summed.
const EIGHT_COUNTERS: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz, vz)
}
block 1 (vi: i64, vacc: i64) {
  veight = i64.const 8
  vlt = i64.lt_u vi veight
  br_if vlt 2(vi, vacc) 3(vacc)
}
block 2 (vi2: i64, vacc2: i64) {
  vn = i64.const 1000
  vz2 = i64.const 0
  vh = thread.spawn 1 vn vz2
  vs, vv = thread.join vh
  vacc3 = i64.add vacc2 vv
  vone = i64.const 1
  vin = i64.add vi2 vone
  br 1(vin, vacc3)
}
block 3 (vr: i64) {
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

/// The least budget a run completes in, by bisection over `fueled`'s engine `which`.
fn least_budget(m: &temen_ir::Module, which: usize, hi: u64) -> u64 {
    let (mut lo, mut hi) = (0u64, hi); // lo runs out, hi completes
    assert!(fueled(m, hi)[which].0.is_ok());
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        match fueled(m, mid)[which].0 {
            Ok(_) => hi = mid,
            Err(Trap::OutOfFuel) => lo = mid,
            Err(t) => panic!("{t:?}"),
        }
    }
    hi
}

/// No task strands fuel the oracle would not. A run's least budget is set by the chunk rule: each
/// vCPU's draw takes up to half of the room, and the root holds what it drew while it waits in a
/// join, so the threads' burns come out of half of what is left. A chunk a native task's
/// kernel-side fuel drew and never burned would raise the kernel's least budget past the oracle's
/// (the root's did, before native tasks' fuel stopped drawing: 49,999 stranded of 100,000). A
/// completed run leaves exactly the oracle's fuel, but for #2977. Both JIT routes burn one unit per spawned
/// thread more than the oracle (#2977), which the half-room draw doubles in the least budget: the
/// allowance is spelled out, so fixing #2977 fails this test until it is taken away.
#[test]
fn the_kernel_route_strands_no_fuel_the_oracle_does_not() {
    const SPAWNS: u64 = 8;
    let m = module(EIGHT_COUNTERS);
    let [oracle, _, kernel] = fueled(&m, 1_000_000);
    assert_eq!(oracle.0, Ok(vec![8 * 500_500]));
    assert_eq!(kernel.0, oracle.0, "the oracle's answer");
    assert_eq!(oracle.1 - kernel.1, SPAWNS, "#2977's allowance");
    let least = least_budget(&m, 0, 1_000_000);
    assert_eq!(
        least_budget(&m, 2, 1_000_000),
        least + 2 * SPAWNS,
        "the kernel route's least budget is the oracle's, but for #2977's allowance"
    );
}

// ----- #2958 slice 2: the futex fast paths on a native vCPU -------------------------------------

/// The oracle's answer for `m`, as the JIT's are spelled.
fn oracle(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    let mut left = temen_interp::DEFAULT_FUEL;
    temen_interp::run(m, 0, &[], &mut left).map(|v| {
        v.iter()
            .map(|x| match x {
                temen_interp::Value::I64(n) => *n,
                other => panic!("{other:?}"),
            })
            .collect()
    })
}

/// #2958 slice 2 — a notify or a wait with no waiter at every kind of address answers as the oracle
/// does on both routes: the op's own checks (`prepare_wait`, `confine_for_notify`) trap a bad address
/// before any fast answer, a wait in range times out (its word is the 0 it expects), and a notify in
/// range answers 0 woken without the run lock.
#[test]
fn a_futex_op_with_no_waiter_answers_as_the_oracle_on_both_lanes() {
    for op in ["wait", "notify"] {
        for addr in [0, 4096, 16_384, 16_386, 65_536, 1 << 30] {
            let m = module(&op_in_thread(op, addr));
            let want = oracle(&m);
            assert_eq!(today(&m), want, "{op} at {addr}, today's JIT");
            assert_eq!(kernel(&m), want, "{op} at {addr}, the kernel route");
        }
    }
}

/// The root spawns a vCPU, and both run `op` forever: a notify of word 16400, which nothing waits
/// on, a wait on it for a 1 it never holds, or (to compare) nothing but the loop's add. With
/// `returns`, the root instead returns 7 once the vCPU has started its loop.
fn forever(op: &str, returns: bool) -> String {
    let op = match op {
        "notify" => "va = i64.const 16400\n  vc = i32.const 1\n  vk = atomic.notify va vc",
        "wait" => {
            "va = i64.const 16400\n  vx = i32.const 1\n  vto = i64.const -1\n  vk = i32.atomic.wait va vx vto"
        }
        _ => "vk = i32.const 0",
    };
    let root = if returns {
        "block 1 () {
  vc = i64.const 16392
  vs = i32.atomic.load vc
  vzero = i32.const 0
  vwait = i32.eq vs vzero
  br_if vwait 1() 2()
  }
block 2 () {
  vr = i64.const 7
  return vr
  }"
        .to_string()
    } else {
        format!("block 1 () {{\n  {op}\n  br 1()\n  }}")
    };
    format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  br 1()
  }}
{root}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
  vc = i64.const 16392
  vone = i32.const 1
  i32.atomic.store vc vone
  br 1()
  }}
block 1 () {{
  {op}
  br 1()
  }}
}}
"
    )
}

/// #2958 slice 2 — a vCPU that only notifies a word nothing waits on, or only waits on a word that
/// already differs, takes no stop: the root's return still ends it at its loop's back-edge poll of
/// the run's end word, as it ends a vCPU that only computes, and the run returns the root's value.
#[test]
fn the_roots_return_ends_a_vcpu_that_only_takes_the_futex_fast_paths() {
    for op in ["notify", "wait", "compute"] {
        let before = temen_interp::bytecode::futex_no_wakes();
        assert_eq!(kernel(&module(&forever(op, true))), Ok(vec![7]), "{op}");
        if cfg!(debug_assertions) && op != "compute" {
            let taken = temen_interp::bytecode::futex_no_wakes() > before;
            assert!(taken, "the fast path ran ({op})");
        }
    }
}

/// #2958 slice 2 — the JIT's kill door ends two vCPUs that only take the fast paths promptly, as it
/// ends two that only compute: the kill stops the kernel's run, whose end word their back-edges
/// poll. The time each took is printed beside the compute loop's.
#[test]
fn the_kill_door_ends_vcpus_that_only_take_the_futex_fast_paths() {
    for op in ["notify", "wait", "compute"] {
        let m = module(&forever(op, false));
        let kill = temen_jit::Interrupt::new();
        let k = std::sync::Arc::clone(&kill);
        let fired = std::sync::Arc::new(std::sync::Mutex::new(None));
        let at = std::sync::Arc::clone(&fired);
        let watchdog = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(100));
            *at.lock().unwrap() = Some(Instant::now());
            k.fire();
        });
        let (o, vcpus) = jit_kernel::run_fresh(&m, 0, &[], Some(&kill)).expect("in scope");
        let took = fired.lock().unwrap().expect("the kill fired").elapsed();
        watchdog.join().unwrap();
        eprintln!("kill of a {op} loop on the kernel route: {took:?}");
        assert!(vcpus >= 1, "it ran as kernel tasks");
        assert_eq!(result(o), Err(Trap::OutOfFuel), "{op}");
        assert!(took < PROMPT, "{op}: the kill took {took:?}");
    }
}
