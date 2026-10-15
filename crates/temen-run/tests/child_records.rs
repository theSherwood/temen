//! **#2359 — a parent keeps no powerbox of a child it has joined or detached**, nor its kill flag or
//! its result. A supervisor or server that spawns a child per request or per connection otherwise
//! grows with every child it has ever had.
//!
//! A root spawns a small child (func 1's child image, from a spawn record) and joins it, or polls
//! until it has ended and detaches it, `n` times. The test binary counts every byte the process holds, and a
//! run's peak, less what was held before it, is what the run grew to. Two runs, of [`FEW`] and
//! [`MANY`] children, show what each child left behind: the difference over the children between
//! them. Before #2359 the tree-walker and the Cranelift JIT kept each child's whole powerbox, and the
//! bytecode drivers a fresh one in its place: about 11 KiB a child on each.
//!
//! The root polls each child to its end before it detaches it, so the peak holds what ended
//! children left, never children still running. A root that detaches each child at once outruns
//! them by as many as the host's scheduling lets it, and those still running hold their whole
//! powerbox: on CI's macOS runners, enough to read as 1.3 KiB a child.
//!
//! The debug driver is left out: it keeps every child domain for time travel, by design.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use temen_interp::{Host, MemLayout, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{run_on, Driver};

/// Every byte this process holds, and the most it has held since [`held_at_most`] last reset it.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn grew(by: usize) {
    let now = LIVE.fetch_add(by, Ordering::Relaxed) + by;
    PEAK.fetch_max(now, Ordering::Relaxed);
}

// SAFETY: every call forwards to `System` with the caller's arguments; the counters only observe.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            grew(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
            grew(new_size);
        }
        p
    }
}

#[global_allocator]
static COUNTING: Counting = Counting;

/// The counters see the whole process, so one measurement runs at a time.
static ONE_AT_A_TIME: Mutex<()> = Mutex::new(());

/// The children of the two runs compared.
const FEW: u64 = 40;
const MANY: u64 = 440;

/// What a joined or detached child may leave behind: its spent slot (#2380), with the JIT's
/// completion cell beside it.
const PER_CHILD: usize = 512;

/// ... and on the bytecode drivers its finished task and domain too, in tables a run never reuses
/// (#2377).
const PER_CHILD_BYTECODE: usize = 2048;

/// What `run` grew the process to, past what it held before.
fn held_at_most(run: impl FnOnce()) -> usize {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    run();
    PEAK.load(Ordering::Relaxed).saturating_sub(before)
}

/// The root's last act on each child: `join` (op 1), or `poll` (op 9) until it has ended, then
/// `detach` (op 10).
#[derive(Clone, Copy, Debug)]
enum End {
    Join,
    Detach,
}

/// A root that spawns func 1's child image `n` times from the record at 17408, paid from its
/// `Budget`, and joins or detaches each. It returns the sum of what the joined children returned (7
/// apiece), or, when it detaches them, how many answered `poll` that they had returned (`n`). Func 2
/// is the poll loop.
fn src(n: u64, end: End) -> String {
    let take = match end {
        End::Join => "vst, vr = call.cap 6 1 (i32) -> (i64, i64) vi (vch)",
        End::Detach => {
            "vw = call 2 (vi, vch)
  vd = call.cap 6 10 (i32) -> (i32) vi (vch)
  vreturned = i32.const 1
  vret = i32.eq vw vreturned
  vr = i64.extend_i32_u vret"
        }
    };
    format!(
        "memory 16
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32, vmod: i32) {{
  vma = i64.const {module_at}
  i32.store vma vmod
  vba = i64.const {budget_at}
  i32.store vba vbud
  vn = i64.const {n}
  vz = i64.const 0
  br 1(vn, vz, vinst)
}}
block 1 (vleft: i64, vsum: i64, vi: i32) {{
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vi (vrp)
  {take}
  vsum2 = i64.add vsum vr
  vone = i64.const 1
  vleft2 = i64.sub vleft vone
  vzero = i64.const 0
  vmore = i64.ne vleft2 vzero
  br_if vmore 1(vleft2, vsum2, vi) 2(vsum2)
}}
block 2 (vtotal: i64) {{
  return vtotal
}}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas: i64) {{
  vseven = i64.const 7
  return vseven
}}
}}
func (i32, i32) -> (i32) {{
block 0 (vpi: i32, vpc: i32) {{
  br 1(vpi, vpc)
}}
block 1 (vqi: i32, vqc: i32) {{
  vp = call.cap 6 9 (i32) -> (i32) vqi (vqc)
  vrunning = i32.const 0
  vstill = i32.eq vp vrunning
  br_if vstill 1(vqi, vqc) 2(vp)
}}
block 2 (vended: i32) {{
  return vended
}}
}}
{rec}",
        module_at = 17408 + rec::MODULE_AT,
        budget_at = 17408 + rec::BUDGET_AT,
        rec = rec::segment(17408, &SpawnRec::v1()),
    )
}

fn module(n: u64, end: End) -> Module {
    let m = temen_text::parse_module(&src(n, end)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// What the root returns.
fn want(n: u64, end: End) -> i64 {
    match end {
        End::Join => 7 * n as i64,
        End::Detach => n as i64,
    }
}

/// The root's powerbox: an `Instantiator` over its window, an unbounded `Budget` its children are paid
/// from, and the child image.
fn setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        let i = h.grant_instantiator();
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
        (h, [i, b, c].map(Value::I32).to_vec())
    }
}

/// The run on `driver`.
fn on_driver(driver: Driver, n: u64, end: End) {
    let m = module(n, end);
    let ran = run_on(driver, &m, &setup(&m)).expect("the driver runs it");
    assert_eq!(
        ran.result,
        Ok(vec![Value::I64(want(n, end))]),
        "{driver:?}, {end:?}, {n} children"
    );
}

/// The run on the Cranelift JIT.
fn on_the_jit(n: u64, end: End) {
    let m = Arc::new(module(n, end));
    let (mut host, args) = setup(&m)();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => i64::from(*x),
            other => panic!("unexpected arg {other:?}"),
        })
        .collect();
    let window = MemLayout::image(vec![0u8; 1 << 16]);
    let (jo, _) = temen_run::jit_cap_run(&m, 0, &args, &window, 0, 0, &mut host, None)
        .expect("the JIT compiles it");
    match jo {
        JitOutcome::Returned(v) => assert_eq!(v, [want(n, end)], "the JIT, {end:?}, {n} children"),
        other => panic!("the JIT, {end:?}, {n} children: {other:?}"),
    }
}

/// What a child left behind on `engine`, in bytes: the growth of [`MANY`] children's run over
/// [`FEW`]'s, over the children between them. A first run warms whatever the engine builds once.
fn left_per_child(engine: &str, run: impl Fn(u64)) -> usize {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    run(FEW);
    let few = held_at_most(|| run(FEW));
    let many = held_at_most(|| run(MANY));
    let per = many.saturating_sub(few) / (MANY - FEW) as usize;
    eprintln!("{engine}: {few} B for {FEW} children, {many} B for {MANY}: {per} B a child");
    per
}

fn check(end: End) {
    let mut over = Vec::new();
    for driver in [Driver::Oracle, Driver::Coop, Driver::Parallel, Driver::Vcpu] {
        if matches!((driver, end), (Driver::Vcpu, End::Detach)) {
            continue; // the resumable `Vcpu` has no `detach` (#2083)
        }
        let per = left_per_child(&format!("{driver:?} {end:?}"), |n| {
            on_driver(driver, n, end)
        });
        let most = match driver {
            Driver::Coop | Driver::Parallel => PER_CHILD_BYTECODE,
            _ => PER_CHILD,
        };
        if per > most {
            over.push(format!("{driver:?}: {per} B, past {most}"));
        }
    }
    let per = left_per_child(&format!("JIT {end:?}"), |n| on_the_jit(n, end));
    if per > PER_CHILD {
        over.push(format!("JIT: {per} B, past {PER_CHILD}"));
    }
    assert!(
        over.is_empty(),
        "a {end:?}ed child left too much behind: {over:?}"
    );
}

#[test]
fn a_joined_childs_powerbox_goes_on_every_engine() {
    check(End::Join);
}

#[test]
fn a_detached_childs_powerbox_goes_on_every_engine() {
    check(End::Detach);
}
