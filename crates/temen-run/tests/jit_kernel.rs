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
        init: None,
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
/// fuel, every vCPU's unburned draw handed back. Each spawned thread's entry costs one on every
/// engine (#2977), as the root's does. (Today's JIT is no reference here: its threads share one
/// fixed cell, whose plain decrements race when they run at once.)
#[test]
fn a_threaded_run_with_fuel_to_spare_leaves_the_oracles_fuel() {
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
        assert_eq!(kernel.1, oracle.1, "the oracle's fuel left");
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

/// #2958 slice 3 — the root makes `page` (`map`, `unmap` or `protect` with its operands) through
/// the address-space capability (handle 0), then a spawned vCPU makes `op` at `addr` (as
/// [`op_in_thread`]) on the page it changed, and the root returns what the vCPU's op answered.
fn page_then(handle: i32, page: &str, op: &str, addr: u64) -> String {
    let (cap_op, args, sig) = match page {
        "map" => (
            0,
            format!("i64.const {addr}\n  vl = i64.const 4096\n  vp = i64.const 3"),
            3,
        ),
        "unmap" => (1, format!("i64.const {addr}\n  vl = i64.const 4096"), 2),
        "protect_none" => (
            2,
            format!("i64.const {addr}\n  vl = i64.const 4096\n  vp = i64.const 0"),
            3,
        ),
        _ => (
            2,
            format!("i64.const {addr}\n  vl = i64.const 4096\n  vp = i64.const 1"),
            3,
        ),
    };
    let (params, operands) = match sig {
        3 => ("i64, i64, i64", "va, vl, vp"),
        _ => ("i64, i64", "va, vl"),
    };
    let thread = op_in_thread(op, addr);
    let child = &thread[thread
        .find("func (i64, i64)")
        .expect("the thread's function")..];
    format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vc = i32.const {handle}
  va = {args}
  vpr = call.cap 5 {cap_op} ({params}) -> (i64) vc({operands})
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vst, vv = thread.join vh
  return vv
  }}
}}
{child}"
    )
}

/// #2958 slice 3 — a guest's `map`, `unmap` and `protect` come into the kernel's scope (the `cap`
/// clause's page-op part is gone), and a wait or a notify on the page each changed answers as the
/// oracle's and today's JIT's do, on both routes: the kernel reads the window through the one page
/// map its page ops write.
/// The [`page_then`] cases: a page op, a futex op on its page, the page, and the answer every route
/// gives. A page of the declared window (64 KiB), above the NULL guard, and one of the reserved tail.
const PAGE_CASES: [(&str, &str, u64, Result<i64, &str>); 10] = [
    // A tail page `map`ped read-write: a wait reads its 0 and times out (2); a notify wakes none.
    ("map", "wait", 1 << 20, Ok(2)),
    ("map", "notify", 1 << 20, Ok(0)),
    // An `unmap`ped page: a wait faults; a notify reads no memory and wakes none.
    ("unmap", "wait", 32768, Err("MemoryFault")),
    ("unmap", "notify", 32768, Ok(0)),
    // A page `protect`ed to nothing: likewise.
    ("protect_none", "wait", 32768, Err("MemoryFault")),
    ("protect_none", "notify", 32768, Ok(0)),
    // A page `protect`ed read-only: a wait reads it.
    ("protect_ro", "wait", 32768, Ok(2)),
    ("protect_ro", "notify", 32768, Ok(0)),
    // An `unmap` of a tail page never mapped: a wait there still faults.
    ("unmap", "wait", 1 << 20, Err("MemoryFault")),
    ("unmap", "notify", 1 << 20, Ok(0)),
];

/// #2958 slice 3 — a guest's `map`, `unmap` and `protect` come into the kernel's scope (the `cap`
/// clause's page-op part is gone), and a wait or a notify on the page each changed answers as the
/// oracle's and today's JIT's do: the kernel reads the window through the one page map its page ops
/// write.
#[test]
fn a_page_op_then_a_futex_op_on_its_page_answers_as_the_oracle_on_both_routes() {
    fn case<const N: usize>(handle: i32) -> temen_ir::Module {
        let (page, op, addr, _) = PAGE_CASES[N];
        module(&page_then(handle, page, op, addr))
    }
    macro_rules! each {
        ($($n:literal)*) => {[$((case::<$n> as fn(i32) -> temen_ir::Module, $n)),*]};
    }
    for (make, n) in each!(0 1 2 3 4 5 6 7 8 9) {
        let (page, op, addr, want) = PAGE_CASES[n];
        let want: Answer = want.map(|v| vec![v]).map_err(str::to_string);
        for (route, got) in granted(make, Host::grant_memory) {
            assert_eq!(got, want, "{route}: {page} then {op} at {addr:#x}");
        }
    }
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
/// completed run leaves exactly the oracle's fuel, each spawned thread's entry costing one on both
/// (#2977), and the least budget is the oracle's.
#[test]
fn the_kernel_route_strands_no_fuel_the_oracle_does_not() {
    let m = module(EIGHT_COUNTERS);
    let [oracle, _, kernel] = fueled(&m, 1_000_000);
    assert_eq!(oracle.0, Ok(vec![8 * 500_500]));
    assert_eq!(kernel.0, oracle.0, "the oracle's answer");
    assert_eq!(kernel.1, oracle.1, "the oracle's fuel left");
    let least = least_budget(&m, 0, 1_000_000);
    assert_eq!(
        least_budget(&m, 2, 1_000_000),
        least,
        "the kernel route's least budget is the oracle's"
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

/// #2995 — a root that writes, through a host procedure (handle `handle`), into a page of its
/// `readonly` data segment: the procedure's borrow of the page must be refused (`-EFAULT`), as the
/// hardware refuses the write. A notify keeps it in the kernel's scope.
fn ro_write(handle: i32) -> temen_ir::Module {
    module(&format!(
        "memory 18
data ro 98304 \"efghefgh\"
func () -> (i64) {{
block 0 () {{
  va = i64.const 16400
  vc = i32.const 1
  vk = atomic.notify va vc
  vh = i32.const {handle}
  vb = i64.const 98304
  vr = call.cap 13 0 (i64) -> (i64) vh(vb)
  return vr
  }}
}}
"
    ))
}

/// Grant `host` a procedure that writes eight bytes at its argument through the guest window, and
/// answers `0`, or `-EFAULT` when the window refuses the write.
fn grant_writer(host: &mut Host) -> i32 {
    use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
    host.grant_host_proc(HostProcGrant::new(
        ProcHandler::Sync(Box::new(|_op, args, out, mem, _minter| {
            let at = args.first().copied().unwrap_or(0) as u64;
            let wrote = mem.and_then(|m| m.write_bytes(at, b"XXXXXXXX"));
            put_results(out, &[if wrote.is_some() { 0 } else { -14 }])
        })),
        CapState::Stateless,
    ))
}

/// `f` on a thread of its own, its answer within [`PROMPT`]: a run that hangs fails the test
/// instead of the suite (the thread is left behind).
fn bounded<T: Send + 'static>(what: &str, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(PROMPT) {
        Ok(v) => v,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            panic!("{what} did not end within {PROMPT:?}")
        }
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => panic!("{what} panicked"),
    }
}

/// #2995 — on the kernel route the host's page map knew nothing of the readonly pages the kernel
/// protected, so the procedure's write passed its borrow check and faulted in host code under the
/// domain's `Host` lock; the fault handler unwound the vCPU with the lock held, and the run's end
/// hung on it. Both routes now refuse the borrow as the oracle does.
#[test]
fn a_host_procedures_write_into_a_readonly_page_is_refused_on_both_routes() {
    let want: Answer = Ok(vec![-14]);
    for (route, got) in granted(ro_write, grant_writer) {
        assert_eq!(got, want, "{route}");
    }
}

/// How a run ended, as the three routes all spell it: its results, or the trap's name.
type Answer = Result<Vec<i64>, String>;

/// The root's own ceilings on every route of [`granted`]: `mem` and `maps` bounded, so a `Budget`'s
/// room reads what the run holds.
const CEILING_MEM: u64 = 1 << 30;
const CEILING_MAPS: u64 = 1 << 20;

/// Run the program `make(handle)` makes on the oracle, the kernel route (asserting it ran there) and
/// today's route, each over a host `grant` granted one capability, `handle`, beside what that route's
/// host holds, with the root's ceilings [`CEILING_MEM`] and [`CEILING_MAPS`]. Each JIT route runs on
/// a thread of its own and must end within [`PROMPT`].
fn granted(
    make: fn(i32) -> temen_ir::Module,
    grant: fn(&mut Host) -> i32,
) -> [(&'static str, Answer); 3] {
    use temen_interp::Value;
    let values = |v: Vec<Value>| -> Vec<i64> {
        v.into_iter()
            .map(|x| match x {
                Value::I64(n) => n,
                other => panic!("{other:?}"),
            })
            .collect()
    };
    let ceilings = |h: &mut Host| h.set_grant(CEILING_MEM as i64, -1, -1, CEILING_MAPS as i64);
    let mut host = Host::new();
    ceilings(&mut host);
    let m = make(grant(&mut host));
    let oracle = temen_interp::run_with_host(&m, 0, &[], &mut 1_000_000, &mut host)
        .map(values)
        .map_err(|t| format!("{t:?}"));
    let kernel = bounded("the kernel route", move || {
        let mut host = Host::new();
        ceilings(&mut host);
        let m = make(grant(&mut host));
        let admitted = jit_kernel::admit(&m, &host, None, false).expect("in the kernel's scope");
        let start = jit_kernel::Start {
            entry: 0,
            args: &[],
            init_mem: &[],
            init: None,
            reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
            fuel: u64::MAX,
            snapshot_cap: None,
            kill: None,
        };
        let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
        assert!(end.vcpus >= 1, "it ran as kernel tasks");
        end.result.map_err(|t| format!("{t:?}"))
    });
    // Today's route: a run that keeps its code is declined by the kernel (`kept code`), so it runs on
    // the JIT's own runtime, whose window start seeds the host's page map (`window_aliases`).
    let today = bounded("today's route", move || {
        let handle = std::cell::Cell::new(0);
        let cfg = temen_run::RunConfig {
            jit_code: Some(temen_run::JitCodeCache::default()),
            limits: temen_run::Limits {
                mem: Some(CEILING_MEM),
                maps: Some(CEILING_MAPS),
                ..temen_run::Limits::default()
            },
            ..temen_run::RunConfig::default()
        };
        // The powerbox grants its own handles first: learn the one `grant` adds, then run the
        // program that names it. (The probe's own run is not wanted.)
        let mut learn = |h: &mut Host| handle.set(grant(h));
        let probe = temen_run::instantiate(make(0)).expect("instantiate");
        let _ =
            probe.run_with_caps_and_host(temen_run::Backend::TreeWalk, &cfg, &[], Some(&mut learn));
        let mut again = |h: &mut Host| {
            grant(h);
        };
        let ran = temen_run::instantiate(make(handle.get()))
            .expect("instantiate")
            .run_with_caps_and_host(temen_run::Backend::Jit, &cfg, &[], Some(&mut again));
        match ran {
            Ok(r) => match r.outcome {
                temen_run::Outcome::Returned(v) => Ok(values(v)),
                o => panic!("{o:?}"),
            },
            // "guest trapped (MemoryFault) — …": the trap's name.
            Err(e) => Err(e
                .split_once('(')
                .and_then(|(_, rest)| rest.split_once(')'))
                .map_or(e.clone(), |(name, _)| name.to_string())),
        }
    });
    [
        ("the oracle", oracle),
        ("the kernel route", kernel),
        ("today's route", today),
    ]
}

/// #2958 slice 3 — the race on the kernel route, overlapping by construction: a spawned vCPU sets a
/// start word, then loops `op` (a not-equal wait, the no-wake fast path's, or a notify) on a word
/// of page 8 until it reads a done word, and returns how many ops it made. The root, holding the
/// address-space capability `handle`, waits for the start word, unmaps and maps page 8 (or protects
/// it to nothing and back) `turns` times, sets the done word and joins. The start and done words
/// are on page 12, which no page op touches.
fn racing(handle: i32, op: &str, protect: bool, turns: u32) -> temen_ir::Module {
    let op = match op {
        "wait" => "vx = i32.const 1\n  vto = i64.const 0\n  vk = i32.atomic.wait va vx vto",
        _ => "vx = i32.const 1\n  vk = atomic.notify va vx",
    };
    let (take, give) = match protect {
        true => (
            "vp0 = i64.const 0\n  vr0 = call.cap 5 2 (i64, i64, i64) -> (i64) vc(va, vl, vp0)",
            "vp3 = i64.const 3\n  vr1 = call.cap 5 2 (i64, i64, i64) -> (i64) vc(va, vl, vp3)",
        ),
        false => (
            "vr0 = call.cap 5 1 (i64, i64) -> (i64) vc(va, vl)",
            "vp3 = i64.const 3\n  vr1 = call.cap 5 0 (i64, i64, i64) -> (i64) vc(va, vl, vp3)",
        ),
    };
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  br 1(vh)
}}
block 1 (vh0: i32) {{
  vsw = i64.const 49152
  vs = i32.atomic.load vsw
  vgo = i32.eqz vs
  vz0 = i64.const 0
  br_if vgo 1(vh0) 2(vh0, vz0)
}}
block 2 (vh1: i32, vi: i64) {{
  vc = i32.const {handle}
  va = i64.const 32768
  vl = i64.const 4096
  {take}
  {give}
  vone = i64.const 1
  vj = i64.add vi vone
  vlim = i64.const {turns}
  vdone = i64.ge_u vj vlim
  br_if vdone 3(vh1) 2(vh1, vj)
}}
block 3 (vh2: i32) {{
  vdw = i64.const 49156
  vd1 = i32.const 1
  i32.atomic.store vdw vd1
  vst, vv = thread.join vh2
  return vv
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vn: i64, vz: i64) {{
  vsw = i64.const 49152
  vs1 = i32.const 1
  i32.atomic.store vsw vs1
  br 1(vz)
}}
block 1 (vdone: i64) {{
  va = i64.const 32772
  {op}
  vone = i64.const 1
  vd = i64.add vdone vone
  vdw = i64.const 49156
  vfin = i32.atomic.load vdw
  vend = i32.eqz vfin
  br_if vend 1(vd) 2(vd)
}}
block 2 (vr: i64) {{
  return vr
  }}
}}
"
    ))
}

/// Busy threads for the length of a scope, so threads are descheduled at arbitrary points: set to
/// stop as this drops, however the scope ends, so a failed assertion fails the test instead of
/// leaving the scope joining spinners that never stop.
struct Spinners<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for Spinners<'_> {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

fn spin<'s, 'e>(s: &'s std::thread::Scope<'s, 'e>, stop: &'e std::sync::atomic::AtomicBool) {
    for _ in 0..2 {
        s.spawn(move || {
            while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                std::hint::spin_loop();
            }
        });
    }
}

/// The race test: under busy threads, a vCPU's futex ops on a page the root keeps unmapping or
/// protecting never fault the host. The ops overlap the page ops by construction ([`racing`]):
/// - a wait shape's root cycles until a wait meets the page revoked and its `MemoryFault` ends the
///   run (microseconds idle, a few scheduler slices under load), under a bound of `WAIT_TURNS`
///   cycles, tens of seconds, that a run reaches only if the two threads never raced at all;
/// - a notify never faults: its run ends with the vCPU's count (every op answered, at least one).
#[test]
fn futex_ops_racing_page_ops_never_fault_the_host() {
    const NOTIFY_TURNS: u32 = 400;
    const WAIT_TURNS: u32 = 2_000_000;
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let _stop = Spinners(&stop);
        spin(s, &stop);
        for (op, protect) in [
            ("wait", false),
            ("wait", true),
            ("notify", false),
            ("notify", true),
        ] {
            let turns = match op {
                "wait" => WAIT_TURNS,
                _ => NOTIFY_TURNS,
            };
            for _ in 0..30 {
                let mut host = Host::new();
                let m = racing(host.grant_memory(), op, protect, turns);
                let admitted = jit_kernel::admit(&m, &host, None, false).expect("in scope");
                let start = jit_kernel::Start {
                    entry: 0,
                    args: &[],
                    init_mem: &[],
                    init: None,
                    reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
                    fuel: u64::MAX,
                    snapshot_cap: None,
                    kill: None,
                };
                let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
                assert!(end.vcpus >= 1, "it ran as kernel tasks");
                match (op, end.result) {
                    ("wait", Err(Trap::MemoryFault)) => {}
                    ("wait", r) => panic!(
                        "wait (protect: {protect}): the root made {WAIT_TURNS} page-op cycles \
                         and no wait met the page revoked (the two never raced): {r:?}"
                    ),
                    // A notify reads no memory: it never faults.
                    (_, Ok(v)) => {
                        assert!(v[0] >= 1, "notify: every op answered, at least one: {v:?}")
                    }
                    (_, r) => panic!("notify (protect: {protect}): {r:?}"),
                }
            }
        }
    });
}

/// #2958 slice 3 — the root, holding the address-space capability `handle` and at `handle + 1` a
/// host procedure that reads its own budget node, reads it, maps a page of the reserved tail, reads
/// it, unmaps the page and reads it again, and returns the three readings. A notify keeps it in the
/// kernel's scope.
fn page_ops_read(handle: i32) -> temen_ir::Module {
    let p = handle + 1;
    module(&format!(
        "memory 16
func () -> (i64, i64, i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vc = i32.const {handle}
  vp = i32.const {p}
  va = i64.const 1048576
  vl = i64.const 4096
  vrw = i64.const 3
  vz = i64.const 0
  vr0 = call.cap 13 0 (i64) -> (i64) vp(vz)
  vx = call.cap 5 0 (i64, i64, i64) -> (i64) vc(va, vl, vrw)
  vr1 = call.cap 13 0 (i64) -> (i64) vp(vz)
  vy = call.cap 5 1 (i64, i64) -> (i64) vc(va, vl)
  vr2 = call.cap 13 0 (i64) -> (i64) vp(vz)
  return vr0, vr1, vr2
  }}
}}
"
    ))
}

/// The address-space capability, and after it a host procedure answering what the host's own budget
/// node holds of `mem` (in pages, the high half) and `maps` (the low half), as [`page_ops_read`]
/// names them.
fn grant_memory_and_reader(h: &mut Host) -> i32 {
    use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
    let m = h.grant_memory();
    let node = h.own_node();
    let page = temen_interp::host_page_size() as i64;
    let p = h.grant_host_proc(HostProcGrant::new(
        ProcHandler::Sync(Box::new(move |_op, _args, out, _mem, _minter| {
            let (mem, maps) = node.mem_and_maps();
            put_results(out, &[(mem / page) << 32 | maps])
        })),
        CapState::Stateless,
    ));
    assert_eq!(p, m + 1, "the reader follows");
    m
}

/// #2958 slice 3 — budget parity (INVARIANTS #3): while the run holds its window, the root's own node
/// holds the same `mem` and `maps` on the kernel route as on today's, before a guest's `map`, after
/// it and after its `unmap`. The kernel route's activation charges the bound the JIT's own runtime
/// charges (`jit_root_window`) and settles it to what the window's start made
/// (`Host::start_window`), as `window_aliases` does, and a page op charges through the host's
/// dispatch on both. (The oracle's plain entry opens no activation, so it charges no window: its
/// readings are not compared.)
#[test]
fn the_kernel_route_charges_the_window_as_today_before_and_after_a_page_op() {
    let [_, (_, kernel), (_, today)] = granted(page_ops_read, grant_memory_and_reader);
    let r = today.clone().expect("today's route runs");
    let (mem, maps) = (|x: i64| x >> 32, |x: i64| x & 0xFFFF_FFFF);
    assert!(mem(r[1]) > mem(r[0]), "a mapped page costs mem: {r:x?}");
    assert_eq!(r[2], r[0], "and its unmap gives it back: {r:x?}");
    assert!(maps(r[0]) > 0, "the window's runs are charged: {r:x?}");
    assert_eq!(kernel, today, "the kernel route against today's");
}

/// #2958 slice 3 — a restored start runs on the kernel route (the `restored` clause is gone): a run
/// that protects a page to nothing hands back a layout that says so, and a run started from it waits
/// on that page and faults, as the window it was captured from would, where a fresh start's wait
/// times out. The page map is the JIT's own, which a restored layout seeds (`reset_cap_pages`,
/// `set_restore`).
#[test]
fn a_restored_start_runs_on_the_kernel_route_under_its_page_map() {
    let mut host = Host::new();
    let handle = host.grant_memory();
    let protects = module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vc = i32.const {handle}
  va = i64.const 32768
  vl = i64.const 4096
  vnone = i64.const 0
  vr = call.cap 5 2 (i64, i64, i64) -> (i64) vc(va, vl, vnone)
  return vr
  }}
}}
"
    ));
    let fresh = temen_interp::MemLayout::image(Vec::new());
    let run = |m: &temen_ir::Module, init: &temen_interp::MemLayout, host: &mut Host| {
        assert!(
            jit_kernel::admit(m, host, Some(init), false).is_ok(),
            "the kernel route admits the start"
        );
        let r = temen_ir::DEFAULT_RESERVED_LOG2;
        temen_run::jit_cap_run(m, 0, &[], init, r, 0, host, None).expect("runs")
    };
    let (o, layout) = run(&protects, &fresh, &mut host);
    assert_eq!(result(o), Ok(vec![0]), "the protect");
    assert!(!layout.is_seed(), "the layout is a restored one");
    let waits = module(&op_in_thread("wait", 32768));
    let (o, _) = run(&waits, &layout, &mut Host::new());
    assert_eq!(
        result(o),
        Err(Trap::MemoryFault),
        "from the layout, the page is gone"
    );
    let (o, _) = run(&waits, &fresh, &mut Host::new());
    assert_eq!(
        result(o),
        Ok(vec![2]),
        "from a fresh start, the wait times out"
    );
}

/// #2958 slice 3 — no page of a native window is a §13 alias: a root host an embedder staged a
/// pre-map on (`Host::stage_premap`, the op-17 pre-map) is declined (`region start`), so its
/// futex words are keyed with the region's other mappings on today's route, as before. (The
/// interpreters apply an op-17 pre-map to a spawned child's window only, so no oracle answers a root
/// one.) (A captured layout that aliases a region reaches the same clause: `jit_cap_run`
/// stages its §13 pages as the host's pre-map, `Host::reset_cap_pages`.)
#[cfg(unix)]
#[test]
fn a_root_with_a_region_premapped_is_declined_and_answers_on_todays_route() {
    let off = 1u64 << 20;
    let staged = || {
        let mut parent = Host::new();
        parent.set_region_factory(temen_run::new_shared_region);
        let len = temen_interp::host_region_granularity() as usize;
        let backing = temen_run::new_shared_region(len).expect("a region");
        let region = parent.grant_shared_region_backed(backing);
        let mut root = Host::new();
        assert!(parent.stage_premap(region, off, &mut root), "staged");
        root
    };
    // A not-equal wait on the pre-mapped page: the region reads zero.
    let src = format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  va = i64.const {off}
  vx = i32.const 1
  vto = i64.const 0
  vk = i32.atomic.wait va vx vto
  vr = i64.extend_i32_u vk
  return vr
  }}
}}
"
    );
    let m = module(&src);
    let root = staged();
    assert_eq!(
        jit_kernel::admit(&m, &root, None, false).err(),
        Some("region start"),
        "the kernel route declines it"
    );
    let mut root = root;
    let init = temen_interp::MemLayout::image(Vec::new());
    let r = temen_ir::DEFAULT_RESERVED_LOG2;
    let (today, _) =
        temen_run::jit_cap_run(&m, 0, &[], &init, r, 0, &mut root, None).expect("today's route");
    assert_eq!(
        result(today),
        Ok(vec![1]),
        "today's route answers not-equal"
    );
}

/// #2958 slice 3 — a kernel-route window takes no §13 region alias, by any path: a host procedure
/// that maps a region into the window it is handed (`GuestMem::map_region`, a public interface) is
/// refused (`-EINVAL`) there, before anything changes, and a wait at that address faults as on any
/// unmapped page. Today's route maps it, and its wait reads the region. (The window carries the
/// refusal, `Host::refuse_cap_window_aliases`; the admission clauses keep the region-mapping
/// programs they can see on today's route.)
#[cfg(unix)]
#[test]
fn a_host_procedure_cannot_map_a_region_into_a_kernel_route_window() {
    const AT: u64 = 1 << 20;
    // The root calls the procedure, then (with `wait`) waits not-equal at the address it mapped.
    fn program(handle: i32, wait: bool) -> temen_ir::Module {
        let tail = match wait {
            true => {
                "va = i64.const 1048576\n  vx = i32.const 1\n  vto = i64.const 0\n  vk = i32.atomic.wait va vx vto\n  vr = i64.extend_i32_u vk"
            }
            false => "vr = i64.add vm vz",
        };
        module(&format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vn = atomic.notify vq vone
  vp = i32.const {handle}
  vz = i64.const 0
  vm = call.cap 13 0 (i64) -> (i64) vp(vz)
  {tail}
  return vr
  }}
}}
"
        ))
    }
    fn maps(h: &mut Host) -> i32 {
        use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
        h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Sync(Box::new(|_op, _args, out, mem, _minter| {
                let len = temen_interp::host_region_granularity();
                let backing = temen_run::new_shared_region(len as usize).expect("a region");
                let rc = mem.map_or(-1, |m| m.map_region(AT, 0, len, 3, 0, backing));
                put_results(out, &[rc])
            })),
            CapState::Stateless,
        ))
    }
    let [_, (_, kernel), (_, today)] = granted(|h| program(h, false), maps);
    assert_eq!(kernel, Ok(vec![-22]), "the kernel route refuses the alias");
    assert_eq!(today, Ok(vec![0]), "today's route maps it");
    let [_, (_, kernel), (_, today)] = granted(|h| program(h, true), maps);
    assert_eq!(kernel, Err("MemoryFault".into()), "the page stays unmapped");
    assert_eq!(today, Ok(vec![1]), "today's route reads the region");
}

/// Twelve spawned vCPUs, while the root counts to 200,000 and returns 7. `wait`: each loops on a 5 ms
/// timed `i32.atomic.wait` on one word nothing notifies, so the run ends with timed waiters parked,
/// whose deadlines pass while their threads are still leaving; otherwise each returns at once.
fn timed_waiters(wait: bool) -> temen_ir::Module {
    let spawns: String = (0..12)
        .map(|i| format!("  vh{i} = thread.spawn 1 vz vz\n"))
        .collect();
    let worker = match wait {
        true => "  va = i64.const 16400\n  vx = i32.const 0\n  vto = i64.const 5000000\n  vk = i32.atomic.wait va vx vto\n  br 1()",
        false => "  vr = i64.const 0\n  return vr",
    };
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
{spawns}  vi0 = i64.const 0
  br 1(vi0)
  }}
block 1 (vi: i64) {{
  vone = i64.const 1
  vn = i64.add vi vone
  vlim = i64.const 200000
  vdone = i64.ge_u vn vlim
  br_if vdone 2() 1(vn)
  }}
block 2 () {{
  vr = i64.const 7
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
  br 1()
  }}
block 1 () {{
{worker}
  }}
}}
"
    ))
}

/// What a run of `m` holds of a fresh host's node once it has ended ([`temen_interp::NodeRef::
/// mem_and_maps`]), on the kernel route (native tasks) and on executor 2 interpreting, each having
/// returned 7.
fn held_after(m: &temen_ir::Module) -> [(i64, i64); 2] {
    let mut host = Host::new();
    let node = host.own_node();
    let admitted = jit_kernel::admit(m, &host, None, false).expect("in the kernel's scope");
    let start = jit_kernel::Start {
        entry: 0,
        args: &[],
        init_mem: &[],
        init: None,
        reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
        fuel: temen_interp::DEFAULT_FUEL,
        snapshot_cap: None,
        kill: None,
    };
    let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
    assert_eq!(end.vcpus, 13, "every vCPU ran as a kernel task");
    assert_eq!(result(end.outcome().0), Ok(vec![7]), "kernel route");
    let kernel = node.mem_and_maps();
    let mut host = Host::new();
    let node = host.own_node();
    let mut fuel = temen_interp::DEFAULT_FUEL;
    let (r, _) = temen_interp::bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &[],
        None,
        &mut host,
    )
    .expect("in the engine's subset");
    assert_eq!(r, Ok(vec![temen_interp::Value::I64(7)]), "executor 2");
    [kernel, node.mem_and_maps()]
}

/// A run that ends with timed waiters parked returns the root's value and leaves its host's node
/// holding what a run whose threads all returned leaves (the root's window), on both executor 2
/// lanes. Once the verdict is in no rule advances a task (`ThreadRun::settle`): a leaver's wait whose
/// deadline passes after the verdict is never answered, and nothing a host reads after the run
/// depends on whether a thread happened to wake after it.
#[test]
fn a_run_that_ends_with_timed_waiters_parked_holds_what_a_returned_one_does() {
    let returned = held_after(&timed_waiters(false));
    for _ in 0..4 {
        assert_eq!(held_after(&timed_waiters(true)), returned);
    }
}
