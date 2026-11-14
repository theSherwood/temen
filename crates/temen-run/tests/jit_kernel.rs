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

/// Today's JIT's answer, within `PROMPT` ([`bounded`]).
fn today(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    let m = m.clone();
    result(bounded("today's route", move || {
        compile_and_run(&m, 0, &[]).expect("jit")
    }))
}

/// The kernel route's answer, checked to have run its vCPUs as kernel tasks, within `PROMPT`: a run
/// that hangs fails the test ([`bounded`]).
fn kernel(m: &temen_ir::Module) -> Result<Vec<i64>, Trap> {
    let m = m.clone();
    let (o, vcpus) = bounded("the kernel route", move || {
        jit_kernel::run_fresh(&m, 0, &[], None).expect("in the kernel's scope")
    });
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
    let admitted = jit_kernel::admit(m, &host, false).expect("in the kernel's scope");
    let start = fresh_start(budget);
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
        // Bounded: a kill that never lands fails the test instead of hanging it.
        let door = std::sync::Arc::clone(&kill);
        let (o, vcpus) = bounded("the killed run", move || {
            jit_kernel::run_fresh(&m, 0, &[], Some(&door)).expect("in scope")
        });
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
        let admitted = jit_kernel::admit(&m, &host, false).expect("in the kernel's scope");
        let start = fresh_start(u64::MAX);
        let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
        assert!(end.vcpus >= 1, "it ran as kernel tasks");
        end.result.map_err(|t| format!("{t:?}"))
    });
    // Today's route: a run that keeps its code is declined by the kernel (`kept code`), so it runs on
    // the JIT's own runtime, whose window start seeds the host's page map (`window_aliases`).
    let today = bounded("today's route", move || {
        let cfg = temen_run::RunConfig {
            jit_code: Some(temen_run::JitCodeCache::default()),
            limits: temen_run::Limits {
                mem: Some(CEILING_MEM),
                maps: Some(CEILING_MAPS),
                ..temen_run::Limits::default()
            },
            ..temen_run::RunConfig::default()
        };
        instance_run(make, grant, temen_run::Backend::Jit, &cfg)
    });
    [
        ("the oracle", oracle),
        ("the kernel route", kernel),
        ("today's route", today),
    ]
}

/// The program `make(handle)` makes, instanced as an embedder instances one (`temen_run::instantiate`,
/// its powerbox granted and its imports bound), with one capability more, `grant`'s `handle`, run on
/// `backend` under `cfg`. An exit answers `Err("Exit(code)")`, a trap its name.
fn instance_run(
    make: fn(i32) -> temen_ir::Module,
    grant: fn(&mut Host) -> i32,
    backend: temen_run::Backend,
    cfg: &temen_run::RunConfig,
) -> Answer {
    let handle = std::cell::Cell::new(0);
    // The powerbox grants its own handles first: learn the one `grant` adds, then run the program
    // that names it. (The probe's own run is not wanted.)
    let mut learn = |h: &mut Host| handle.set(grant(h));
    let probe = temen_run::instantiate(make(0)).expect("instantiate");
    let _ = probe.run_with_caps_and_host(temen_run::Backend::TreeWalk, cfg, &[], Some(&mut learn));
    let mut again = |h: &mut Host| {
        grant(h);
    };
    let inst = temen_run::instantiate(make(handle.get())).expect("instantiate");
    let before = jit_kernel::kernel_runs();
    let ran = inst.run_with_caps_and_host(backend, cfg, &[], Some(&mut again));
    // The route the run itself took: a JIT run without kept code must be the kernel's, any other
    // not (`jit_run` falls back to the JIT's own runtime when the kernel declines).
    let kernel = backend == temen_run::Backend::Jit && cfg.jit_code.is_none();
    assert_eq!(
        jit_kernel::kernel_runs() - before,
        u64::from(kernel),
        "the run's route (the kernel's: {kernel}; admit says {:?})",
        inst.kernel_route(cfg)
    );
    match ran {
        Ok(r) => match r.outcome {
            temen_run::Outcome::Returned(v) => Ok(v
                .into_iter()
                .map(|x| match x {
                    temen_interp::Value::I64(n) => n,
                    temen_interp::Value::I32(n) => n as i64,
                    other => panic!("{other:?}"),
                })
                .collect()),
            temen_run::Outcome::Exited(c) => Err(format!("Exit({c})")),
        },
        // "guest trapped (MemoryFault) — …": the trap's name.
        Err(e) => Err(e
            .split_once('(')
            .and_then(|(_, rest)| rest.split_once(')'))
            .map_or(e.clone(), |(name, _)| name.to_string())),
    }
}

/// #2958 slice 4 — `make(handle)` on the oracle (the tree-walker), the kernel route and today's route,
/// each instanced as an embedder instances it ([`instance_run`]) under the same limits, so a reading
/// of the run's own budget (its fuel included) compares across them; each JIT route asserts its
/// route, and every lane must end within [`PROMPT`]. `oracle` false leaves the oracle out, for a program whose
/// threads only overlap where they run at once (the oracle's cooperative scheduler runs one). `fuel`
/// is the run's fuel grant ([`METERED`] for a row that reads it; `None` for one whose end must not
/// race a budget).
fn by_instance(
    make: fn(i32) -> temen_ir::Module,
    grant: fn(&mut Host) -> i32,
    oracle: bool,
    fuel: Option<u64>,
) -> Vec<(&'static str, Answer)> {
    use temen_run::Backend;
    let cfg = move |keep: bool| temen_run::RunConfig {
        jit_code: keep.then(temen_run::JitCodeCache::default),
        limits: temen_run::Limits {
            fuel,
            mem: Some(CEILING_MEM),
            maps: Some(CEILING_MAPS),
            ..temen_run::Limits::default()
        },
        ..temen_run::RunConfig::default()
    };
    let mut lanes = Vec::new();
    if oracle {
        lanes.push((
            "the oracle",
            bounded("the oracle", move || {
                instance_run(make, grant, Backend::TreeWalk, &cfg(false))
            }),
        ));
    }
    lanes.push((
        "the kernel route",
        bounded("the kernel route", move || {
            instance_run(make, grant, Backend::Jit, &cfg(false))
        }),
    ));
    lanes.push((
        "today's route",
        bounded("today's route", move || {
            instance_run(make, grant, Backend::Jit, &cfg(true))
        }),
    ));
    lanes
}

/// The fuel a row that reads the run's budget is granted ([`by_instance`]).
const METERED: Option<u64> = Some(50_000_000);

/// Every lane of [`by_instance`] answers `want`.
fn all_answer(
    make: fn(i32) -> temen_ir::Module,
    grant: fn(&mut Host) -> i32,
    oracle: bool,
    fuel: Option<u64>,
    want: Answer,
) {
    for (route, answer) in by_instance(make, grant, oracle, fuel) {
        assert_eq!(answer, want, "{route}");
    }
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
        "notify" => "vx = i32.const 1\n  vk = atomic.notify va vx",
        // Any other op is the IR itself, over the racing word's address `va`.
        other => other,
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

/// #2958 slice 4 review — a host copy of guest memory races a page op on its buffer: a vCPU writes
/// eight bytes from page 8 to a captured stdout (`Stream.write`) in a loop while the root unmaps and
/// maps that page (or protects it to nothing and back), on both JIT routes ([`racing`]). Every host
/// copy of guest memory and every page op is made under the domain's `Host` lock, so a write meets
/// the page whole or refused (its errno), and the run ends with the vCPU's count. The host never
/// faults, which would end the process (or, recovered, surface as a `MemoryFault` the program
/// cannot raise itself).
#[cfg(any(unix, windows))]
#[test]
fn a_stream_write_racing_page_ops_on_its_buffer_never_faults_the_host() {
    /// The stream's handle in the high half, the address space's in the low.
    fn grant(h: &mut Host) -> i32 {
        let stream = h.grant_stream(temen_interp::StreamRole::Out);
        let space = h.grant_memory();
        assert!((0..1 << 15).contains(&stream) && (0..1 << 16).contains(&space));
        stream << 16 | space
    }
    fn write(handles: i32) -> String {
        format!(
            "vsh = i32.const {}\n  vwl = i64.const 8\n  vk = call.cap 0 1 (i64, i64) -> (i64) vsh(va, vwl)",
            handles >> 16
        )
    }
    fn unmapping(handles: i32) -> temen_ir::Module {
        racing(handles & 0xFFFF, &write(handles), false, 400)
    }
    fn protecting(handles: i32) -> temen_ir::Module {
        racing(handles & 0xFFFF, &write(handles), true, 400)
    }
    let stop = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let _stop = Spinners(&stop);
        spin(s, &stop);
        for make in [unmapping as fn(i32) -> temen_ir::Module, protecting] {
            for run in 0..10 {
                for (route, answer) in by_instance(make, grant, false, None) {
                    // Only the count: the program cannot fault (an unreadable buffer is the write's
                    // `-EFAULT`), and a recovered host fault would surface as `MemoryFault`.
                    assert!(
                        matches!(&answer, Ok(v) if v[0] >= 1),
                        "run {run}, {route}: {answer:?}"
                    );
                }
            }
        }
    });
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
///   run (microseconds idle, a few scheduler slices under load). A run whose two threads never
///   raced at all would cycle on, and the harness's bound ([`PROMPT`], far fewer than
///   `WAIT_TURNS` cycles) fails it; `WAIT_TURNS` only caps the program;
/// - a notify never faults: its run ends with the vCPU's count (every op answered, at least one).
///
/// Every wait run also asserts its window recovered no read
/// ([`temen_interp::CapPages::recovered`]): a wait meeting the page midway through a page op's
/// hardware change would be recovered to the same `MemoryFault`, so the count is what still tells
/// a broken contract from the map check. (A notify reads no word, so its runs leave it out.)
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
                // Bounded: a run that hangs fails the test.
                let (vcpus, result, recovered) = bounded("the race", move || {
                    let mut host = Host::new();
                    let m = racing(host.grant_memory(), op, protect, turns);
                    let admitted = jit_kernel::admit(&m, &host, false).expect("in scope");
                    let start = fresh_start(u64::MAX);
                    let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
                    (end.vcpus, end.result, host.cap_pages_recovered())
                });
                assert!(vcpus >= 1, "it ran as kernel tasks");
                // A wait's `MemoryFault` is the map check's, never a recovered read: the guarded
                // read keeps off the page ops' hardware changes ([`CapPages::recovered`]). (A
                // notify reads no word, so the count says nothing for it.)
                if op == "wait" {
                    assert_eq!(recovered, 0, "wait (protect: {protect}): a recovered read");
                }
                match (op, result) {
                    ("wait", Err(Trap::MemoryFault)) => {}
                    ("wait", r) => panic!(
                        "wait (protect: {protect}): the run ended without a wait meeting the \
                         page revoked: {r:?}"
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
            jit_kernel::admit(m, host, false).is_ok(),
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

/// #2958 slice 4 — a root host an embedder staged a pre-map on (`Host::stage_premap`, the op-17
/// pre-map) runs on the kernel route: the region is aliased into the window as it starts, and a
/// not-equal wait on the pre-mapped page reads the region (zero), as today's route answered it.
/// (The interpreters apply an op-17 pre-map to a spawned child's window only, so no oracle answers a
/// root one. A captured layout that aliases a region reaches the same path: `jit_cap_run` stages
/// its §13 pages as the host's pre-map, `Host::reset_cap_pages`.)
#[cfg(any(unix, windows))]
#[test]
fn a_root_with_a_region_premapped_runs_on_the_kernel_route() {
    let off = 1u64 << 20;
    let mut parent = Host::new();
    parent.set_region_factory(temen_run::new_shared_region);
    let len = temen_interp::host_region_granularity() as usize;
    let backing = temen_run::new_shared_region(len).expect("a region");
    let region = parent.grant_shared_region_backed(backing);
    let mut root = Host::new();
    assert!(parent.stage_premap(region, off, &mut root), "staged");
    let m = module(&format!(
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
    ));
    assert_eq!(jit_kernel::admit(&m, &root, false).err(), None, "admitted");
    let init = temen_interp::MemLayout::image(Vec::new());
    let r = temen_ir::DEFAULT_RESERVED_LOG2;
    let (o, _) = temen_run::jit_cap_run(&m, 0, &[], &init, r, 0, &mut root, None).expect("runs");
    assert_eq!(result(o), Ok(vec![1]), "the region reads zero: not-equal");
}

/// The two aliases [`aliased`] maps one region page at.
const ALIAS_A: u64 = 1 << 20;

/// #2958 slice 4 — a program over a `SharedRegion` `handle` of one granule: map it read-write at
/// [`ALIAS_A`] and at the next granule (`B`). A spawned vCPU sets a ready word, notifies it, and
/// waits at the first word of alias A for a `0` there, with no timeout. The root waits for the ready
/// word, then notifies the same word through alias B until a notify reports the one waiter woken,
/// and returns `100 +` the joined waiter's answer (`0`, woken) `+` both maps' answers. The overlap
/// is by construction: the root notifies until it wakes the waiter, and the waiter's one wait cannot
/// end otherwise. Words keyed apart never wake it: the harness's bound ([`PROMPT`]) fails the row.
fn aliased(handle: i32) -> temen_ir::Module {
    let g = temen_interp::host_region_granularity();
    let b = ALIAS_A + g;
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vc = i32.const {handle}
  va = i64.const {ALIAS_A}
  vb = i64.const {b}
  vro = i64.const 0
  vl = i64.const {g}
  vp = i64.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vc(va, vro, vl, vp)
  vm1 = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vc(vb, vro, vl, vp)
  vms = i64.add vm0 vm1
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vrw = i64.const 49152
  vx = i32.const 0
  vto = i64.const -1
  vk = i32.atomic.wait vrw vx vto
  br 1(vh, vms, vz)
}}
block 1 (vh1: i32, vms1: i64, vi: i64) {{
  vbw = i64.const {b}
  vone = i32.const 1
  vn = atomic.notify vbw vone
  br_if vn 2(vh1, vms1) 1(vh1, vms1, vi)
}}
block 2 (vh3: i32, vms3: i64) {{
  vst, vv = thread.join vh3
  vbase = i64.const 100
  vsum = i64.add vbase vv
  vres = i64.add vsum vms3
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vn: i64, vz: i64) {{
  vrw = i64.const 49152
  vone = i32.const 1
  i32.atomic.store vrw vone
  vk = atomic.notify vrw vone
  vaw = i64.const {ALIAS_A}
  vx = i32.const 0
  vto = i64.const -1
  vs = i32.atomic.wait vaw vx vto
  vr = i64.extend_i32_u vs
  return vr
  }}
}}
"
    ))
}

/// A host granted a fresh one-granule anonymous region.
fn grant_region(h: &mut Host) -> i32 {
    let len = temen_interp::host_region_granularity() as usize;
    h.grant_shared_region_backed(temen_run::new_shared_region(len).expect("a region"))
}

/// #2958 slice 4 — two mappings of one region byte in one window rendezvous: a wait through one alias
/// is woken by a notify through the other, on the oracle and both JIT routes. On the kernel route
/// the notify's no-wake fast path finds the waiter counted in under the region's key, and the rules
/// wake it there: a key on the window's own backing would leave the root notifying no one.
#[cfg(any(unix, windows))]
#[test]
fn two_aliases_of_one_region_byte_rendezvous_on_every_route() {
    for (route, answer) in granted(aliased, grant_region) {
        assert_eq!(answer, Ok(vec![100]), "{route}");
    }
}

/// #2958 slice 4 — a host procedure may map a region into the window it is handed
/// (`GuestMem::map_region`, a public interface): the kernel route maps it as today's route does,
/// and a not-equal wait at that address reads the region.
#[cfg(any(unix, windows))]
#[test]
fn a_host_procedure_maps_a_region_into_a_kernel_route_window_as_today() {
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
    assert_eq!(kernel, Ok(vec![0]), "the kernel route maps it");
    assert_eq!(today, Ok(vec![0]), "today's route maps it");
    let [_, (_, kernel), (_, today)] = granted(|h| program(h, true), maps);
    assert_eq!(kernel, Ok(vec![1]), "the kernel route reads the region");
    assert_eq!(today, Ok(vec![1]), "today's route reads the region");
}

/// #2958 slice 4 review — a trusted host procedure maps a file-backed region into its window itself
/// (`GuestMem::map_region`, which admission cannot see: the procedure is `Sync`, and the backing
/// never enters the host's regions), then truncates the file. A wait on the alias meets a page whose
/// backing is gone: `wait_word`'s recoverable load answers `MemoryFault` on both JIT routes, and
/// the process lives. (The oracle reads the backing in host code and dies of SIGBUS, #3005, so it
/// is left out.)
#[cfg(unix)]
#[test]
fn a_wait_on_a_file_alias_a_procedure_mapped_and_truncated_faults_the_guest() {
    // The root calls the procedure; a refused map returns its answer (never a fault), so a pass is
    // the wait's own. Then it waits not-equal on the alias.
    fn program(handle: i32) -> temen_ir::Module {
        module(&format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vp = i32.const {handle}
  vz = i64.const 0
  vm = call.cap 13 0 (i64) -> (i64) vp(vz)
  vok = i64.eqz vm
  br_if vok 1() 2(vm)
  }}
block 1 () {{
  va = i64.const 1048576
  vx = i32.const 1
  vto = i64.const 0
  vw = i32.atomic.wait va vx vto
  vr = i64.extend_i32_u vw
  return vr
  }}
block 2 (vrc: i64) {{
  return vrc
  }}
}}
"
        ))
    }
    // No panic in the handler, which runs inside the JIT's `extern "C"` thunk: each failure is an
    // answer the row then reports. The file is unlinked as soon as it is open.
    fn maps_and_truncates(h: &mut Host) -> i32 {
        use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
        h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Sync(Box::new(|_op, _args, out, mem, _minter| {
                static FILES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                let n = FILES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let path = std::env::temp_dir()
                    .join(format!("temen_jit_kernel_trunc_{}_{n}", std::process::id()));
                let opened = std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(true)
                    .open(&path);
                let _ = std::fs::remove_file(&path);
                let Ok(file) = opened else {
                    return put_results(out, &[-1001]);
                };
                let len = temen_interp::host_region_granularity();
                if file.set_len(len).is_err() {
                    return put_results(out, &[-1002]);
                }
                let Ok(backing) = temen_run::new_file_region(&file, len as usize) else {
                    return put_results(out, &[-1003]);
                };
                let rc = mem.map_or(-1004, |m| m.map_region(1 << 20, 0, len, 3, 0, backing));
                if rc == 0 && file.set_len(0).is_err() {
                    return put_results(out, &[-1005]);
                }
                put_results(out, &[rc])
            })),
            CapState::Stateless,
        ))
    }
    all_answer(
        program,
        maps_and_truncates,
        false,
        METERED,
        Err("MemoryFault".into()),
    );
    // And the answer is the recovered read's, which the window counts: the map called the page
    // readable, and the load faulted on the vanished backing.
    let (result, recovered) = bounded("the kernel run", || {
        let mut host = Host::new();
        let m = program(maps_and_truncates(&mut host));
        let admitted = jit_kernel::admit(&m, &host, false).expect("in scope");
        let start = fresh_start(temen_interp::DEFAULT_FUEL);
        let end = jit_kernel::run(admitted, start, &mut host).expect("compiles");
        (end.result, host.cap_pages_recovered())
    });
    assert_eq!(result, Err(Trap::MemoryFault));
    // At least one: the no-wake fast path's compare and the rules' each read the word.
    assert!(
        recovered >= 1,
        "the wait's read was the recovered one: {recovered}"
    );
}

/// A fresh run's start at entry 0 with `fuel`: no arguments, no initial memory, no kill door.
fn fresh_start(fuel: u64) -> jit_kernel::Start<'static> {
    jit_kernel::Start {
        entry: 0,
        args: &[],
        init_mem: &[],
        init: None,
        reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
        fuel,
        snapshot_cap: None,
        kill: None,
    }
}

/// A root that makes one §12 op (so the kernel admits it), then `body`, returning `vr`.
fn rooted(body: &str) -> temen_ir::Module {
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
{body}
  return vr
  }}
}}
"
    ))
}

/// #2958 slice 4 — a `Stream` write the kernel serves in the dispatch: five bytes to a captured
/// stdout, answered alike on every route.
#[test]
fn a_stream_write_answers_as_the_oracle_on_both_routes() {
    fn make(h: i32) -> temen_ir::Module {
        rooted(&format!(
            "  vc = i32.const {h}
  vp = i64.const 16384
  vw = i64.const 478560413032
  i64.store vp vw
  vl = i64.const 5
  vr = call.cap 0 1 (i64, i64) -> (i64) vc(vp, vl)"
        ))
    }
    all_answer(
        make,
        |h| h.grant_stream(temen_interp::StreamRole::Out),
        true,
        METERED,
        Ok(vec![5]),
    );
}

/// #2958 slice 4 — `Clock.now`: two reads a nanosecond apart on the host's deterministic clock.
#[test]
fn a_clock_read_answers_as_the_oracle_on_both_routes() {
    fn make(h: i32) -> temen_ir::Module {
        rooted(&format!(
            "  vc = i32.const {h}
  vid = i64.const 0
  vt0 = call.cap 2 0 (i64) -> (i64) vc(vid)
  vt1 = call.cap 2 0 (i64) -> (i64) vc(vid)
  vr = i64.sub vt1 vt0"
        ))
    }
    all_answer(make, Host::grant_clock, true, METERED, Ok(vec![1]));
}

/// #2958 slice 4, condition 5 — a `Budget` node under the run's own: its room of fuel, `mem` and
/// `maps`, read before any thread is spawned, so no other vCPU runs; then a `split` and its child's
/// `mem` room. Every route reads the same, the run's fuel included: its fuel cell draws from the same
/// node in the same chunks.
#[test]
fn a_budget_read_and_split_answer_as_the_oracle_on_both_routes() {
    fn make(h: i32) -> temen_ir::Module {
        rooted(&format!(
            "  vc = i32.const {h}
  vf0 = i64.const 0
  vfuel = call.cap 14 1 (i64) -> (i64) vc(vf0)
  vf1 = i64.const 1
  vmem = call.cap 14 1 (i64) -> (i64) vc(vf1)
  vf5 = i64.const 5
  vmaps = call.cap 14 1 (i64) -> (i64) vc(vf5)
  vneg = i64.const -1
  vhalf = i64.const 1048576
  vs = call.cap 14 0 (i64, i64, i64) -> (i64) vc(vneg, vhalf, vneg)
  vsh = i32.wrap_i64 vs
  vsm = call.cap 14 1 (i64) -> (i64) vsh(vf1)
  vr0 = i64.add vfuel vmem
  vr1 = i64.add vr0 vmaps
  vr = i64.add vr1 vsm"
        ))
    }
    let lanes = by_instance(make, |h| h.grant_budget(-1, -1, -1), true, METERED);
    let want = lanes[0].1.clone();
    assert!(want.as_ref().is_ok_and(|v| v[0] > 0), "{want:?}");
    for (route, answer) in lanes {
        assert_eq!(answer, want, "{route}");
    }
}

/// #2958 slice 4 — the `AddressSpace` `sub` mint (op 4): a child space over a sub-range, its handle
/// alike on every route. (`create_region`, op 5, under a region factory is declined, `file region`:
/// [`each_declined_call_names_its_clause`].)
#[test]
fn the_address_space_sub_mint_answers_as_the_oracle_on_both_routes() {
    fn make(h: i32) -> temen_ir::Module {
        rooted(&format!(
            "  vc = i32.const {h}
  voff = i64.const 32768
  vlog = i64.const 12
  vr = call.cap 5 4 (i64, i64) -> (i64) vc(voff, vlog)"
        ))
    }
    let lanes = by_instance(make, Host::grant_memory, true, METERED);
    let want = lanes[0].1.clone();
    assert!(want.as_ref().is_ok_and(|v| v[0] > 0), "minted: {want:?}");
    for (route, answer) in lanes {
        assert_eq!(answer, want, "{route}");
    }
}

/// #2958 slice 4 — the `self` namespace's reflection the kernel serves: the handle count (op 0), the
/// type of a handle (op 2) and `parallelism` (op 19), alike on every route.
#[test]
fn self_reflection_answers_as_the_oracle_on_both_routes() {
    fn make(h: i32) -> temen_ir::Module {
        rooted(&format!(
            "  vz = i32.const 0
  vn = call.cap 4294967295 0 () -> (i64) vz ()
  vh = i64.const {h}
  vt = call.cap 4294967295 2 (i64) -> (i64) vz (vh)
  vp = call.cap 4294967295 19 () -> (i64) vz ()
  vk1 = i64.const 1000
  vn1 = i64.mul vn vk1
  vt1 = i64.add vn1 vt
  vk2 = i64.const 1000000
  vp1 = i64.mul vp vk2
  vr = i64.add vt1 vp1"
        ))
    }
    let lanes = by_instance(make, Host::grant_clock, true, METERED);
    let want = lanes[0].1.clone();
    assert!(want.is_ok(), "{want:?}");
    for (route, answer) in lanes {
        assert_eq!(answer, want, "{route}");
    }
}

/// #2958 slice 4, condition 6 — `Exit` from the root, on handle `exit`, while a spawned sibling
/// `sibling` does: `wait`
/// parks it on a word nothing notifies; `spin` loops forever on a word nothing writes. The run ends
/// `Exit(7)` on every route, the sibling stopped (parked: woken by the end; spinning: at its next
/// check of the domain's end).
fn exits_beside(sibling: &str, exit: i32) -> temen_ir::Module {
    let body = match sibling {
        "wait" => {
            "  va = i64.const 16384\n  vx = i32.const 0\n  vto = i64.const -1\n  vs = i32.atomic.wait va vx vto\n  vr = i64.extend_i32_u vs\n  return vr\n  }\n"
        }
        _ => {
            "  br 1()\n}\nblock 1 () {\n  va = i64.const 16384\n  vs = i32.atomic.load va\n  vgo = i32.eqz vs\n  br_if vgo 1() 2()\n}\nblock 2 () {\n  vr = i64.const 0\n  return vr\n  }\n"
        }
    };
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  ve = i32.const {exit}
  vcode = i64.const 7
  vr = call.cap 1 0 (i64) -> (i64) ve(vcode)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
{body}}}
"
    ))
}

#[test]
fn an_exit_beside_a_parked_sibling_ends_the_run_on_every_route() {
    all_answer(
        |h| exits_beside("wait", h),
        Host::grant_exit,
        true,
        METERED,
        Err("Exit(7)".into()),
    );
}

#[test]
fn an_exit_beside_a_spinning_sibling_ends_the_run_on_every_route() {
    // On the JIT routes the sibling spins on a thread of its own until the exit ends it; the oracle,
    // which runs one task at a time, gives the answer they must match.
    all_answer(
        |h| exits_beside("spin", h),
        Host::grant_exit,
        true,
        // No fuel limit of the row's own: the run's default fuel (`DEFAULT_FUEL`), far more than the
        // sibling spins before the root's exit ends it.
        None,
        Err("Exit(7)".into()),
    );
}

/// #2958 slice 4 — `Exit` from a spawned vCPU ends its whole domain, the root parked in a join.
#[test]
fn an_exit_from_a_thread_ends_the_run_on_every_route() {
    fn make(h: i32) -> temen_ir::Module {
        module(&format!(
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
  ve = i32.const {h}
  vcode = i64.const 7
  vr = call.cap 1 0 (i64) -> (i64) ve(vcode)
  return vr
  }}
}}
"
        ))
    }
    all_answer(make, Host::grant_exit, true, METERED, Err("Exit(7)".into()));
}

/// #2958 slice 4, condition 6 — `Exit` from the root while a sibling is inside a host call: an
/// unlocked host procedure (#2645, which runs with the domain's `Host` lock let go) that marks its
/// entry in the window and returns once the root has marked its exit (its op 1 hands the root the
/// exit handle). The root waits for the entry
/// mark, marks, and exits, so the sibling is inside the call when the root's exit lands, by
/// construction. The run ends `Exit(7)` once the call returns and the sibling sees the domain's end.
/// (Not on the oracle, which runs one task at a time: its call would wait for a root that cannot run.)
#[test]
fn an_exit_beside_a_sibling_inside_a_host_call_ends_the_run_on_both_routes() {
    static HOST_CALL_LEFT: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    fn make(h: i32) -> temen_ir::Module {
        module(&format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vp = i32.const {h}
  vz = i64.const 0
  vex = call.cap 13 1 (i64) -> (i64) vp(vz)
  ve = i32.wrap_i64 vex
  vh = thread.spawn 1 vz vz
  br 1(ve)
}}
block 1 (ve1: i32) {{
  vin = i64.const 16384
  vs = i32.atomic.load vin
  vgo = i32.eqz vs
  br_if vgo 1(ve1) 2(ve1)
}}
block 2 (ve2: i32) {{
  vout = i64.const 16388
  vone = i32.const 1
  i32.atomic.store vout vone
  vcode = i64.const 7
  vr = call.cap 1 0 (i64) -> (i64) ve2(vcode)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
  vp = i32.const {h}
  vz = i64.const 0
  vr = call.cap 13 0 (i64) -> (i64) vp(vz)
  return vr
  }}
}}
"
        ))
    }
    fn grant(h: &mut Host) -> i32 {
        use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
        let exit = h.grant_exit();
        h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Unlocked(std::sync::Arc::new(move |op, _args, out, mem| {
                // Op 1 answers the exit handle, so the root holds both.
                if op == 1 {
                    return put_results(out, &[exit as i64]);
                }
                let mem = mem.ok_or(Trap::Malformed)?;
                mem.write_bytes(16384, &1u32.to_le_bytes())
                    .ok_or(Trap::MemoryFault)?;
                // The root marks its exit next. A run that never does is a failing row: the test's
                // end lets the call go ([`HOST_CALL_LEFT`]).
                while !mem.read_bytes(16388, 4).is_some_and(|b| b != [0; 4])
                    && !HOST_CALL_LEFT.load(std::sync::atomic::Ordering::Relaxed)
                {
                    std::hint::spin_loop();
                }
                put_results(out, &[0])
            })),
            CapState::Stateless,
        ))
    }
    // However the row ends, the procedure it leaves spinning stops.
    struct Left;
    impl Drop for Left {
        fn drop(&mut self) {
            HOST_CALL_LEFT.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let _left = Left;
    all_answer(make, grant, false, METERED, Err("Exit(7)".into()));
}

/// #2958 slice 4 — the calls the kernel route declines, inline, each with the clause that names what
/// removes it (`native_scope`'s `kernel_op`, default-deny), under a host that holds what makes it
/// park where that matters.
#[test]
fn each_declined_call_names_its_clause() {
    // A self op takes no arguments; every other call one.
    let call = |ty: u32, op: u32| {
        let args = match ty {
            u32::MAX => "() -> (i64) vc ()",
            _ => "(i64) -> (i64) vc(vz)",
        };
        rooted(&format!(
            "  vc = i32.const 0\n  vz = i64.const 0\n  vr = call.cap {ty} {op} {args}"
        ))
    };
    let blocking_stdin = || {
        let mut h = Host::new();
        h.stdin_block = true;
        h
    };
    let pipe = || {
        let mut h = Host::new();
        h.grant_pipe();
        h
    };
    let blocking = || {
        let mut h = Host::new();
        h.grant_blocking(Duration::ZERO, None);
        h
    };
    // A live-callee offer shaped as a `Stream` (read, write, close), over a callee with nothing in it.
    let live_impl = || {
        use temen_ir::{FuncType, ValType};
        let mut h = Host::new();
        let rw = FuncType {
            params: vec![ValType::I64, ValType::I64],
            results: vec![ValType::I64],
        };
        let close = FuncType {
            params: vec![],
            results: vec![],
        };
        let names: Vec<String> = ["read", "write", "close"].map(String::from).into();
        let callee = std::sync::Arc::new(std::sync::Mutex::new(Host::new()));
        h.wire_live_impl(&callee, 0, &names, &[rw.clone(), rw, close])
            .expect("wired");
        h
    };
    // A host holding a module's own offer (`Binding::Offer`), as a re-grant or an earlier run's
    // `export.handle` leaves one.
    let offered = || {
        let mut h = Host::new();
        let impls = module(
            "func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
",
        );
        let funcs: std::sync::Arc<[temen_ir::Func]> = impls.funcs.into();
        h.wire_offer_func(&funcs, &std::sync::Arc::from(Vec::new()), &[0])
            .expect("an offer");
        h
    };
    let factory = || {
        let mut h = Host::new();
        h.set_region_factory(temen_run::new_shared_region);
        h
    };
    let minting = || {
        use temen_interp::{put_results, CapState, HostProcGrant, ProcHandler};
        let mut h = Host::new();
        h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Minting(Box::new(|_, _, out, _, _| put_results(out, &[0]))),
            CapState::Stateless,
        ));
        h
    };
    let outside_written = || {
        let mut h = Host::new();
        h.grant_shared_region_backed(std::sync::Arc::new(OutsideWritten));
        h
    };
    // A host procedure its embedder answers later: a call parks.
    let host_answered = || {
        let mut h = Host::new();
        let requests = temen_interp::CapRequests::default();
        h.grant_host_proc(temen_interp::HostProcGrant::declared("later", &requests));
        h
    };
    let selfop = u32::MAX;
    let rows: Vec<(temen_ir::Module, Host, &str)> = vec![
        (call(0, 0), blocking_stdin(), "stream parks"),
        (call(0, 0), pipe(), "stream parks"),
        (call(0, 1), pipe(), "stream parks"),
        // A close of a pipe end wakes the other end's parks.
        (call(0, 2), pipe(), "stream parks"),
        (call(0, 3), Host::new(), "stream parks"),
        // An op no interface has: `Clock` has only op 0.
        (call(2, 1), Host::new(), "cap"),
        (call(selfop, 16), Host::new(), "stream parks"),
        // `fuel` is the table's, but the bytecode compile declines the op first.
        (call(selfop, 13), Host::new(), "bytecode"),
        (call(4, 7), Host::new(), "region signal"),
        (call(10, 0), blocking(), "host-proc parks"),
        (call(13, 0), host_answered(), "host-proc parks"),
        (call(6, 17), Host::new(), "§14 children"),
        (call(7, 0), Host::new(), "§14 children"),
        (call(8, 1), Host::new(), "§14 children"),
        (call(11, 0), Host::new(), "jit cap"),
        (call(16, 0), Host::new(), "cap"),
        (call(0x1000_0000, 0), Host::new(), "serving"),
        // A live callee's offer under a built-in's id: a call on it is the callee's to serve.
        (call(0, 1), live_impl(), "serving"),
        // The `self` ops `kernel_op` names `serving` for (9, 10): the bytecode compile declines
        // them first, so `serving` is the second line of defence. The `process` ops (11, 12, 14)
        // reach the kernel's own clause.
        (call(selfop, 9), Host::new(), "bytecode"),
        (call(selfop, 10), Host::new(), "bytecode"),
        (call(selfop, 11), Host::new(), "process"),
        (call(selfop, 12), Host::new(), "process"),
        (call(selfop, 14), Host::new(), "process"),
        // `export.handle`'s packed self op, inline: it mints an offer of the module's own impl.
        (call(selfop, 8), Host::new(), "serving"),
        (call(selfop, 8 | 1 << 8), Host::new(), "serving"),
        // A guest offer the host holds at admission, which may sit under a built-in's id.
        (call(0, 1), offered(), "serving"),
        // The three ways into `file region`: a guest mint under a region factory, a host procedure
        // that may mint a backing, and a held region something outside the VM writes.
        (call(5, 5), factory(), "file region"),
        (call(0, 1), minting(), "file region"),
        (call(0, 1), outside_written(), "file region"),
    ];
    let wrong: Vec<String> = rows
        .iter()
        .enumerate()
        .filter_map(|(i, (m, host, clause))| {
            let got = jit_kernel::admit(m, host, false).err();
            (got != Some(*clause)).then(|| format!("row {i}: {got:?}, not {clause:?}"))
        })
        .collect();
    assert!(wrong.is_empty(), "{wrong:#?}");
    // The kinds `native_scope` decides before the calls: a fiber (`fibers`) and a `gc.roots`
    // (`gc.roots`), which the instruction scan passes because these clauses already declined them.
    let fiber = module(
        "memory 16
func () -> (i64) {
block 0 () {
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vf = ref.func 1
  vz = i64.const 0
  vc = cont.new vf vz
  return vc
  }
}
func (i64) -> (i64) {
block 0 (va: i64) {
  return va
  }
}
",
    );
    assert_eq!(
        jit_kernel::admit(&fiber, &Host::new(), false).err(),
        Some("fibers")
    );
    let roots = rooted(
        "  vlo = i64.const 0\n  vhi = i64.const -1\n  vmask = i64.const -1\n  vbuf = i64.const 32768\n  vcap = i64.const 8\n  vr = gc.roots vlo vhi vmask vbuf vcap",
    );
    assert_eq!(
        jit_kernel::admit(&roots, &Host::new(), false).err(),
        Some("gc.roots")
    );
    // And what it admits beside them: a stream read with a non-blocking stdin and no pipe, and the
    // packed `type_id` and `covers` of a self interface past the first, as their typed forms are.
    assert!(jit_kernel::admit(&call(0, 0), &Host::new(), false).is_ok());
    for op in [6 | 1 << 8, 7 | 1 << 8] {
        assert!(
            jit_kernel::admit(&call(selfop, op), &Host::new(), false).is_ok(),
            "self op {op:#x}"
        );
    }
}

/// #2958 slice 4 — a threaded root over the mmap-capable fs (`fs::host_fs_mmap`, handle `fs`): open a
/// file, size it to a page, mint its region and map it at 1 MiB, truncate the file to nothing, then
/// wait not-equal on a word of the alias. The page under the alias has lost its backing: the read
/// faults, a guest `MemoryFault`, never a fault in host code.
#[cfg(unix)]
fn truncated(fs: i32) -> temen_ir::Module {
    let page = temen_interp::host_page_size();
    rooted(&format!(
        "  vp = i64.const 16384
  vch = i32.const 102
  i32.store8 vp vch
  vfs = i32.const {fs}
  vl1 = i64.const 1
  vfl = i64.const 19
  vfd = call.cap 13 0 (i64, i64, i64) -> (i64) vfs(vp, vl1, vfl)
  vpg = i64.const {page}
  vt1 = call.cap 13 7 (i64, i64) -> (i64) vfs(vfd, vpg)
  vz = i64.const 0
  vrg = call.cap 13 13 (i64, i64, i64) -> (i64) vfs(vfd, vz, vpg)
  vrh = i32.wrap_i64 vrg
  vat = i64.const 1048576
  vrw = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vrh(vat, vz, vpg, vrw)
  vt0 = call.cap 13 7 (i64, i64) -> (i64) vfs(vfd, vz)
  vx = i32.const 1
  vto = i64.const 0
  vw = i32.atomic.wait vat vx vto
  vr = i64.extend_i32_u vw"
    ))
}

/// The mmap-capable fs over a directory of this test's own.
#[cfg(unix)]
fn grant_mmap_fs(h: &mut Host) -> i32 {
    let dir = std::env::temp_dir().join(format!("temen_jit_kernel_mmap_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    temen_run::fs::host_fs_mmap(dir).install(h, 1 << 16)
}

/// #2958 slice 4 review, finding 1 — a file's alias can lose its pages under a host read (#3005),
/// so the kernel route declines a run over the mmap-capable fs (a minting host procedure), `file
/// region`, and the run takes today's route.
#[cfg(unix)]
#[test]
fn a_wait_on_an_alias_of_a_truncated_file_is_declined() {
    let mut host = Host::new();
    let fs = grant_mmap_fs(&mut host);
    assert_eq!(
        jit_kernel::admit(&truncated(fs), &host, false).err(),
        Some("file region")
    );
}

/// #2958 slice 4 review, finding 3 — a grouped import reaches every op its remap names, and the
/// kernel route decides each: a `{ write, read }` interface over a `Stream` is declined when its
/// `read` would park on a blocking stdin, though its first op, `write`, would not.
#[test]
fn a_grouped_import_is_decided_on_every_op_it_reaches() {
    let m = module(
        "memory 16
type 0 func (i64, i64) -> (i64)
type 1 interface { write: 0, read: 0 }
import 0 interface \"stdio\" 1
func 0 () -> (i64) {
block 0 () {
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  v0 = i64.const 16384
  v1 = i64.const 4
  v2 = call.import 0.read (v0, v1)
  return v2
  }
}
",
    );
    let bound = |block: bool| {
        let mut host = Host::new();
        let h = host.grant_stream(temen_interp::StreamRole::In);
        host.stdin_block = block;
        host.set_import_bindings(vec![temen_interp::BoundImport::required(
            temen_ir::cap_id::STREAM,
            1,
            h,
        )]);
        host.set_import_remap(0, std::sync::Arc::from(vec![1u32, 0]));
        host
    };
    assert_eq!(
        jit_kernel::admit(&m, &bound(true), false).err(),
        Some("stream parks")
    );
    assert!(jit_kernel::admit(&m, &bound(false), false).is_ok());
}

/// #2958 slice 4 review — the race on an aliased page, as slice 3's race on a private one is built:
/// a spawned vCPU waits not-equal on the first word of an alias at 1 MiB in a loop; the root, which
/// waits for its start word, then unmaps the alias and maps the region there again (`remap`, or maps
/// it over the live alias, replacing it) until the run ends. A wait that meets the page unmapped
/// faults, which ends the run `MemoryFault`; the host never faults. Not on the oracle, whose root
/// would run its loop alone. These rows run as an embedder runs, so they cannot read the window's
/// count of recovered reads: a broken page-map contract would be recovered to the same
/// `MemoryFault` and pass here. The count is asserted where the window is in reach: for the
/// revoke-first arm (unmap, map, protect of a private page) by
/// `futex_ops_racing_page_ops_never_fault_the_host` and `futex_window.rs`'s
/// `a_read_racing_an_unmap_or_a_protect_never_faults_the_host`; for this race's arm, a map over a
/// live alias under the page map's lock, by `futex_window.rs`'s
/// `a_read_racing_a_map_over_its_alias_meets_the_page_whole`.
fn alias_race(handle: i32, remap: bool) -> temen_ir::Module {
    let g = temen_interp::host_region_granularity();
    let unmap = match remap {
        true => "vu = call.cap 4 1 (i64, i64) -> (i64) vc(va, vl)",
        false => "vu = i64.const 0",
    };
    module(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vc = i32.const {handle}
  va = i64.const {ALIAS_A}
  vro = i64.const 0
  vl = i64.const {g}
  vp = i64.const 3
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vc(va, vro, vl, vp)
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  br 1()
}}
block 1 () {{
  vsw = i64.const 49152
  vs = i32.atomic.load vsw
  vgo = i32.eqz vs
  br_if vgo 1() 2()
}}
block 2 () {{
  vc = i32.const {handle}
  va = i64.const {ALIAS_A}
  vro = i64.const 0
  vl = i64.const {g}
  vp = i64.const 3
  {unmap}
  vm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vc(va, vro, vl, vp)
  br 2()
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vn: i64, vz: i64) {{
  vsw = i64.const 49152
  vs1 = i32.const 1
  i32.atomic.store vsw vs1
  br 1()
}}
block 1 () {{
  va = i64.const {ALIAS_A}
  vx = i32.const 1
  vto = i64.const 0
  vk = i32.atomic.wait va vx vto
  br 1()
  }}
}}
"
    ))
}

#[cfg(any(unix, windows))]
#[test]
fn a_wait_racing_an_unmap_of_its_alias_faults_the_guest_never_the_host() {
    let spinners = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let _stop = Spinners(&spinners);
        spin(s, &spinners);
        all_answer(
            |h| alias_race(h, true),
            grant_region,
            false,
            None,
            Err("MemoryFault".into()),
        );
    });
}

/// The same race with each map replacing the live alias, for a bounded count. Whether a wait meets
/// the page between the old alias's drop and the new one's map is the race's: each route ends either
/// returning or `MemoryFault` (the replacement is not atomic to the guest on macOS and Windows,
/// #3050). The row still fails when the host faults (the test process dies) or
/// the run hangs (the harness's [`PROMPT`]).
#[cfg(any(unix, windows))]
#[test]
fn a_wait_racing_a_map_over_its_alias_never_faults_the_host() {
    fn replacing(handle: i32) -> temen_ir::Module {
        let src = temen_text::print_module(&alias_race(handle, false));
        module(&src.replacen(
            "  br 2()\n",
            "  vdw = i64.const 49160\n  vcnt = i64.atomic.load vdw\n  vone = i64.const 1\n  vnext = i64.add vcnt vone\n  i64.atomic.store vdw vnext\n  vlim = i64.const 20000\n  vend = i64.ge_u vnext vlim\n  br_if vend 3() 2()\n}\nblock 3 () {\n  vr = i64.const 0\n  return vr\n",
            1,
        ))
    }
    let spinners = std::sync::atomic::AtomicBool::new(false);
    std::thread::scope(|s| {
        let _stop = Spinners(&spinners);
        spin(s, &spinners);
        // Many runs in one execution: the swap's window is narrow, and a host fault ends the
        // process (#3003's tripwire), so this is the one place a reader meeting it shows.
        for run in 0..10 {
            for (route, answer) in by_instance(replacing, grant_region, false, None) {
                assert!(
                    matches!(&answer, Ok(v) if v == &[0]) || answer == Err("MemoryFault".into()),
                    "run {run}, {route}: {answer:?}"
                );
            }
        }
    });
}

/// #2958 slice 4, condition 5 — a `Budget` read of fuel after the run's cells have drawn more than
/// once (a loop long enough to need a refill). A guest's read bounds fuel by the budgets below the
/// run's node, never by the run's own grant, whose room differs by engine (each draws on its own
/// schedule; INVARIANTS #3): so a node with no fuel ceiling of its own reads unbounded, `-1`, on
/// every route, however its run's cells drew. The equality is by that rule, not a measured parity of
/// the engines' draws.
#[test]
fn a_budget_read_of_fuel_after_a_refill_answers_alike_on_every_route() {
    fn make(h: i32) -> temen_ir::Module {
        module(&format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vinc = i64.const 1
  vj = i64.add vi vinc
  vlim = i64.const 3000000
  vdone = i64.ge_u vj vlim
  br_if vdone 2() 1(vj)
}}
block 2 () {{
  vc = i32.const {h}
  vf0 = i64.const 0
  vr = call.cap 14 1 (i64) -> (i64) vc(vf0)
  return vr
  }}
}}
"
        ))
    }
    all_answer(
        make,
        |h| h.grant_budget(-1, -1, -1),
        true,
        METERED,
        Ok(vec![-1]),
    );
}

/// #2958 slice 4 — `setjmp` and `longjmp` beside threads: no `native_scope` clause names them, and
/// the JIT's compile refuses the program on both of its routes (`Unsupported`), so neither runs it
/// and the embedder's JIT run fails as it did before. The oracle runs it: a spawned vCPU sets a jump
/// buffer, jumps back to it with `5`, and returns what the second return of `setjmp` gave.
#[test]
fn setjmp_beside_threads_is_the_jits_compile_to_refuse_on_both_routes() {
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
  vbuf = i64.const 16384
  vr = setjmp vbuf
  vfirst = i32.eqz vr
  br_if vfirst 1() 2(vr)
}
block 1 () {
  vb1 = i64.const 16384
  vfive = i32.const 5
  longjmp vb1 vfive
  unreachable
}
block 2 (vr2: i32) {
  vx = i64.extend_i32_u vr2
  return vx
  }
}
";
    let m = module(src);
    assert_eq!(oracle(&m), Ok(vec![5]));
    assert!(matches!(
        compile_and_run(&m, 0, &[]),
        Err(temen_jit::JitError::Unsupported(_))
    ));
    assert_eq!(
        jit_kernel::run_fresh(&m, 0, &[], None).err(),
        Some("the JIT's compile")
    );
}

/// #2958 slice 4 — an import is decided on the op its binding reaches, as an inline call on it is
/// ([`each_declined_call_names_its_clause`]): `self.fuel.remaining` (`fuel`, or the bytecode
/// compile's refusal first), a service point (`serving`, which the bytecode compile declines first
/// inline but cannot see through an import) and a region post's `signal` (`region signal`) are
/// declined through an import as they are inline.
#[test]
fn an_import_is_declined_by_its_targets_clause() {
    let m = module(
        "import 0 \"target\" (i64) -> (i64)
memory 16
func () -> (i64) {
block 0 () {
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vz = i64.const 0
  vr = call.import 0 (vz)
  return vr
  }
}
",
    );
    let bound = |type_id: u32, op: u32| {
        let mut host = Host::new();
        host.set_import_bindings(vec![temen_interp::BoundImport::required(type_id, op, 0)]);
        host
    };
    let rows = [
        (temen_ir::CAP_SELF_TYPE_ID, 13, "fuel"),
        (temen_ir::CAP_SELF_TYPE_ID, 8, "serving"),
        (temen_ir::CAP_SELF_TYPE_ID, 9, "serving"),
        (temen_ir::CAP_SELF_TYPE_ID, 10, "serving"),
        (temen_ir::cap_id::SHARED_REGION, 7, "region signal"),
    ];
    for (type_id, op, clause) in rows {
        assert_eq!(
            jit_kernel::admit(&m, &bound(type_id, op), false).err(),
            Some(clause),
            "({type_id}, {op})"
        );
    }
}

/// A pure-Rust backing written from outside the VM, as a host file is: `file region`.
struct OutsideWritten;

impl temen_interp::SharedBacking for OutsideWritten {
    fn size(&self) -> u64 {
        temen_interp::host_region_granularity()
    }
    fn read_byte(&self, _off: u64) -> u8 {
        0
    }
    fn write_byte(&self, _off: u64, _b: u8) {}
    fn atomic(&self, _off: u64, _width: u32, _f: &mut dyn FnMut(u64) -> Option<u64>) -> u64 {
        0
    }
    fn outside_writers(&self) -> bool {
        true
    }
}

/// Twelve spawned vCPUs, each of which checks in (a count at 16392) and then, with `wait`, loops on a
/// 5 ms timed `i32.atomic.wait` on a word nothing notifies, or else returns. The root returns 7 once
/// all twelve have checked in, so every vCPU runs, and with `wait` the run ends with timed waiters
/// parked, whose deadlines pass while their threads are still leaving.
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
{spawns}  br 1()
  }}
block 1 () {{
  vc = i64.const 16392
  vn = i32.atomic.load vc
  vall = i32.const 12
  vin = i32.eq vn vall
  br_if vin 2() 1()
  }}
block 2 () {{
  vr = i64.const 7
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va0: i64, vb0: i64) {{
  vc = i64.const 16392
  vone = i32.const 1
  vold = i32.atomic.rmw.add vc vone
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
    let admitted = jit_kernel::admit(m, &host, false).expect("in the kernel's scope");
    let start = fresh_start(temen_interp::DEFAULT_FUEL);
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
    let returned = bounded("the returned run", || held_after(&timed_waiters(false)));
    for _ in 0..4 {
        let parked = bounded("the run with waiters parked", || {
            held_after(&timed_waiters(true))
        });
        assert_eq!(parked, returned);
    }
}

/// #2958 slice 4 review — `export.handle` mints an offer of the module's own impl, under its
/// interface's interned id, which is a built-in's when the shapes match (here `Stream`'s `read`,
/// `write`, `close`): a call on that handle is a call through a guest interface, so the kernel route
/// declines the instruction (`serving`).
#[test]
fn an_export_handle_is_declined_as_serving() {
    let m = module(
        "type 0 func (i64, i64) -> (i64)
type 1 func () -> ()
type 2 interface { read: 0, write: 0, close: 1 }
export 0 interface \"stdio\" 2 { read: 1, write: 1, close: 2 }
memory 16
func () -> (i64) {
block 0 () {
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vh = export.handle 0
  vr = i64.extend_i32_u vh
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
func () -> () {
block 0 () {
  return
  }
}
",
    );
    assert_eq!(
        jit_kernel::admit(&m, &Host::new(), false).err(),
        Some("serving")
    );
}
