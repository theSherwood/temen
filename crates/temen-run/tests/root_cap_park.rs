//! **#2047 — a host call that waits on outside input parks its caller, on every engine.** A root
//! blocks in an offloadable host procedure (`wait`) until something outside the guest has
//! happened: here, until a ticker in the guest has called a second procedure (`tick`) [`TICKS`]
//! times. The ticker is a detached child, or a thread of the root's own domain; it runs only if
//! the root's wait parks the root and leaves everything else in the run running. An engine that
//! runs the wait inline on a thread the ticker needs never sees the ticks: the wait gives up after
//! [`GIVE_UP`] and returns `-1`.
//!
//! A **paced** ticker waits on a timer before each tick, so the run is idle except for a timer and
//! the root's outstanding wait: the timer must still fire while the root waits. Under **one lane**
//! (D66) the root and its ticker take turns: the root must give its lane up while it waits. And a
//! JIT root **killed** while it waits stops at once, as from every park (§5).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use temen_interp::{CapState, ForkedProc, Host, MemLayout, OffloadOutcome, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{JitOutcome, TrapKind};

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{run_on, Driver};

/// How many ticks the root waits for.
const TICKS: u64 = 5;

/// How long the root's wait waits for them before it gives up and returns `-1`: long enough for a
/// loaded CI runner to schedule the ticker, and only spent when an engine stalls it.
const GIVE_UP: Duration = Duration::from_secs(10);

/// The ticks so far, and the bell the root's wait listens on.
#[derive(Default)]
struct Ticks {
    n: Mutex<u64>,
    rung: Condvar,
}

impl Ticks {
    fn tick(&self) -> i64 {
        let mut n = self.n.lock().unwrap();
        *n += 1;
        self.rung.notify_all();
        *n as i64
    }

    /// `want` ticks' count once they have all come, or `-1` after [`GIVE_UP`].
    fn wait_for(&self, want: u64) -> i64 {
        let n = self.n.lock().unwrap();
        let (n, waited) = self
            .rung
            .wait_timeout_while(n, GIVE_UP, |n| *n < want)
            .unwrap();
        if waited.timed_out() {
            -1
        } else {
            *n as i64
        }
    }
}

/// Who ticks.
#[derive(Clone, Copy, Debug)]
enum Ticker {
    /// A detached child (func 1's child image), granted `tick` by name.
    Child,
    /// A thread of the root's domain (func 2), handed the root's `tick`.
    Thread,
}

/// A ticker's loop, after an entry block that leaves `tick`'s handle in `vh`: `TICKS` calls, each
/// after a 1 ms timed wait on a word nobody notifies when `paced`. Returns the last tick's count.
fn tick_loop(entry: &str, paced: bool) -> String {
    let pace = if paced {
        "  vpa = i64.const 17800\n  vpe = i32.const 0\n  vpt = i64.const 1000000\n  vps = i32.atomic.wait vpa vpe vpt\n"
    } else {
        ""
    };
    format!(
        "block 0 {entry}  vn = i64.const {TICKS}
  br 1(vh, vn)
}}
block 1 (vh1: i32, vleft: i64) {{
{pace}  vz = i64.const 0
  vt = call.cap 13 0 (i64) -> (i64) vh1 (vz)
  vone = i64.const 1
  vleft2 = i64.sub vleft vone
  vmore = i64.ne vleft2 vz
  br_if vmore 1(vh1, vleft2) 2(vt)
}}
block 2 (vlast: i64) {{
  return vlast
}}
}}"
    )
}

/// A root `(inst, tick, wait, budget, child) -> i64` that starts `ticker`, waits in `wait` for
/// `TICKS` ticks, joins the ticker and returns what the wait returned.
fn src(ticker: Ticker, paced: bool) -> String {
    let (start, end) = match ticker {
        Ticker::Child => (
            format!(
                "{grant}  vma = i64.const {module_at}
  i32.store vma vmod
  vba = i64.const {budget_at}
  i32.store vba vbud
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
",
                grant = rec::grant("g", 17600, 17700, 4, "vtick"),
                module_at = 17408 + SpawnRec::MODULE_AT,
                budget_at = 17408 + SpawnRec::BUDGET_AT,
            ),
            "  vst, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)\n",
        ),
        Ticker::Thread => (
            "  vsp = i64.const 0\n  vta = i64.extend_i32_u vtick\n  vth = thread.spawn 2 vsp vta\n"
                .to_string(),
            "  vj = thread.join vth\n",
        ),
    };
    let child = SpawnRec {
        grants_ptr: 17600,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        "memory 17
data 17700 \"tick\"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32, vmod: i32) {{
{start}  vwant = i64.const {TICKS}
  vw = call.cap 13 0 (i64) -> (i64) vwait (vwant)
{end}  return vw
}}
}}
func (i64) -> (i64) {{
{child_loop}
func (i64, i64) -> (i64) {{
{thread_loop}
{rec}",
        child_loop = tick_loop(
            "(v0: i64) {\n  vp = i64.const 17700\n  vl = i64.const 4\n  vh = self.resolve vp vl\n",
            paced
        ),
        thread_loop = tick_loop("(vsp: i64, varg: i64) {\n  vh = i32.wrap_i64 varg\n", paced),
        rec = rec::segment(17408, &child),
    )
}

fn module(ticker: Ticker, paced: bool) -> Module {
    let m = temen_text::parse_module(&src(ticker, paced)).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// `tick` (op 0): count a tick and ring the bell. Forkable, so a spawn can grant it to a child.
fn tick_proc(ticks: &Arc<Ticks>) -> temen_interp::HostProc {
    let t = Arc::clone(ticks);
    Box::new(move |_op, _args, _mem, _minter| Ok(vec![t.tick()]))
}

/// The root's powerbox: an `Instantiator` over its window, `tick`, `wait` (op 0 `wait(n)`: punt a
/// wait for `n` ticks to the offload pool), an unbounded `Budget` and the child image, on `lanes`
/// lanes (`-1`: as many as it likes). Fresh ticks for each run.
fn setup(m: &Module, lanes: i64) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let ticks = Arc::new(Ticks::default());
        let mut h = Host::new();
        h.set_lane_cap(lanes);
        let i = h.grant_instantiator();
        let forked = Arc::clone(&ticks);
        let tick = h.grant_host_proc_forkable(
            tick_proc(&ticks),
            Arc::new(move |_pid| ForkedProc::shared(tick_proc(&forked), CapState::Stateless)),
            CapState::Stateless,
        );
        let wait = h.grant_host_proc_offloadable(
            Box::new(move |_op, args| {
                let (t, want) = (Arc::clone(&ticks), args.first().copied().unwrap_or(0));
                OffloadOutcome::Offload(Box::new(move || t.wait_for(want as u64)))
            }),
            CapState::Stateless,
        );
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
        (h, [i, tick, wait, b, c].map(Value::I32).to_vec())
    }
}

/// What the root returned on `driver`, or why there is nothing to compare.
fn on_driver(driver: Driver, m: &Module, lanes: i64) -> Result<i64, String> {
    let ran = run_on(driver, m, &setup(m, lanes)).ok_or("declined the module")?;
    match ran.result {
        Ok(v) if v.len() == 1 => match v[0] {
            Value::I64(r) => Ok(r),
            other => Err(format!("returned {other:?}")),
        },
        other => Err(format!("ended {other:?}")),
    }
}

/// What the root returned on the Cranelift JIT.
fn on_the_jit(m: &Module, lanes: i64) -> Result<i64, String> {
    let (mut host, args) = setup(m, lanes)();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => i64::from(*x),
            other => panic!("unexpected arg {other:?}"),
        })
        .collect();
    let window = MemLayout::image(vec![0u8; 1 << 17]);
    let (jo, _) = temen_run::jit_cap_run(m, 0, &args, &window, 0, 0, &mut host, None)
        .map_err(|e| format!("did not compile: {e:?}"))?;
    match jo {
        JitOutcome::Returned(v) if v.len() == 1 => Ok(v[0]),
        other => Err(format!("ended {other:?}")),
    }
}

/// Every engine the ticker runs on, and what each root returned that was not the ticks.
fn check(ticker: Ticker, paced: bool, lanes: i64) {
    let m = module(ticker, paced);
    let mut wrong = Vec::new();
    let mut each = |engine: String, got: Result<i64, String>| {
        if got != Ok(TICKS as i64) {
            wrong.push(format!("  {engine}: {got:?}"));
        }
    };
    for driver in drivers::ALL {
        // The `Vcpu` harness orchestrates no futex `Wait`, which a paced ticker makes.
        if paced && driver == Driver::Vcpu {
            continue;
        }
        each(format!("{driver:?}"), on_driver(driver, &m, lanes));
    }
    each("JIT".into(), on_the_jit(&m, lanes));
    assert!(
        wrong.is_empty(),
        "a {ticker:?} ticker{}{}: want {TICKS} from the root's wait on every engine, but\n{}",
        if paced { ", paced" } else { "" },
        if lanes == 1 { ", one lane" } else { "" },
        wrong.join("\n")
    );
}

#[test]
fn a_root_waiting_on_the_host_lets_its_child_run() {
    check(Ticker::Child, false, -1);
}

#[test]
fn a_root_waiting_on_the_host_lets_its_thread_run() {
    check(Ticker::Thread, false, -1);
}

#[test]
fn a_root_waiting_on_the_host_lets_a_childs_timer_fire() {
    check(Ticker::Child, true, -1);
}

#[test]
fn a_root_waiting_on_the_host_lets_a_threads_timer_fire() {
    check(Ticker::Thread, true, -1);
}

#[test]
fn a_root_waiting_on_the_host_gives_its_lane_to_its_child() {
    check(Ticker::Child, false, 1);
}

#[test]
fn a_root_waiting_on_the_host_gives_its_lane_to_its_thread() {
    check(Ticker::Thread, false, 1);
}

/// A JIT root killed while it waits on a host call that will not answer: the watchdog sets the
/// run's interrupt cell, and the wait ends within a re-check, as a park's does. The run traps
/// `OutOfFuel` instead of waiting the call out, and the call's answer is left unclaimed.
#[test]
fn a_jit_root_killed_while_it_waits_on_the_host_stops() {
    let m = temen_text::parse_module(
        "func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vr = call.cap 13 0 (i64) -> (i64) vh (vz)
  return vr
}
}",
    )
    .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    // The call answers once the test lets it, or after `GIVE_UP`.
    let let_go = Arc::new((Mutex::new(false), Condvar::new()));
    let mut host = Host::new();
    let gate = Arc::clone(&let_go);
    let h = host.grant_host_proc_offloadable(
        Box::new(move |_op, _args| {
            let gate = Arc::clone(&gate);
            OffloadOutcome::Offload(Box::new(move || {
                let (open, rung) = &*gate;
                let open = open.lock().unwrap();
                drop(rung.wait_timeout_while(open, GIVE_UP, |o| !*o).unwrap());
                7
            }))
        }),
        CapState::Stateless,
    );
    let interrupt = Arc::new(AtomicU64::new(0));
    let kill = Arc::clone(&interrupt);
    let watchdog = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(50));
        kill.store(1, Ordering::SeqCst);
    });
    let started = Instant::now();
    let outcome = temen_jit::compile_and_run_with_host_interruptible(
        &m,
        0,
        &[i64::from(h)],
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        Arc::as_ptr(&interrupt),
    )
    .expect("the JIT compiles it");
    let took = started.elapsed();
    watchdog.join().unwrap();
    *let_go.0.lock().unwrap() = true;
    let_go.1.notify_all();
    assert_eq!(outcome, JitOutcome::Trapped(TrapKind::OutOfFuel));
    assert!(took < GIVE_UP / 2, "the wait outlived the kill: {took:?}");
}
