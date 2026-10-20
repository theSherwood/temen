//! **#2047, #2424 — a host call that waits on outside input parks its caller, on every engine,
//! whether the caller is a root or a §14 child.** The caller blocks in an offloadable host
//! procedure (`wait`) until something outside the guest has happened: here, until a ticker in the
//! guest has called a second procedure (`tick`) [`TICKS`] times. The ticker runs only if the wait
//! parks its caller and leaves everything else in the run running. An engine that runs the wait
//! inline on a thread the ticker needs never sees the ticks: the wait gives up after [`GIVE_UP`]
//! and returns `-1`.
//!
//! **A root waits** while a detached child, or a thread of its own domain, ticks. A **paced**
//! ticker waits on a timer before each tick, so the run is idle except for a timer and the root's
//! outstanding wait: the timer must still fire while the root waits. Under **one lane** (D66) the
//! root and its ticker take turns: the root must give its lane up while it waits. And a JIT root
//! **killed** while it waits stops at once, as from every park (§5).
//!
//! **A child waits** while its root ticks (#2424): the spawn grants it `wait` through the
//! procedure's fork factory, and its call parks the child alone. The root first waits, in `wait`'s
//! op 1, until the child's wait has begun, so the ticks always find it parked. Two children waiting
//! at once each get their answer when it comes: one's never waits behind the other's call. A child
//! may wait from a fiber of its own, resuming it, blocking, until it returns.
//!
//! Within a domain too, each answer comes as it arrives (#2477): a root's call answers while a
//! thread of its domain still waits on an earlier one.
//!
//! And a root that resumes a fiber, blocking, through a slow answer is not deadlocked: the answer
//! comes from outside the run, though nothing in the run is live meanwhile.
//!
//! **Nothing waits for a call whose caller is gone** (#2493): a root that kills a child waiting on
//! the host joins it at once, and a run whose root returns over a fiber still waiting ends at once.
//! The calls go on running, and their answers land where no one waits.
//!
//! **Every case runs twice** (#2495): with `wait` answered by the domain's offload pool, and with
//! `wait` answered by the embedder, through each call's [`Completer`], from a thread of its own.
//! A host-answered call parks and wakes exactly as a pool call does, on every engine and in every
//! domain.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use temen_interp::{
    CapState, Completer, ForkedProc, Host, MemLayout, OffloadHostProc, OffloadOutcome, Value,
};
use temen_ir::{Module, SpawnRec};
use temen_jit::{JitOutcome, TrapKind};

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use drivers::{run_on, Driver};

/// How many ticks a wait waits for.
const TICKS: u64 = 5;

/// How long a wait waits before it gives up and returns `-1`: long enough for a loaded CI runner
/// to schedule the ticker, and only spent when an engine stalls it.
const GIVE_UP: Duration = Duration::from_secs(10);

/// How long `wait`'s op 2 takes to answer: longer than the JIT's two 20 ms deadlock re-checks, so a
/// resumer that counted itself blocked on it would read the run as deadlocked.
const SLOW: Duration = Duration::from_millis(100);

/// The ticks so far and the waits begun, and the bell both ring.
#[derive(Default)]
struct Ticks {
    n: Mutex<(u64, u64)>,
    rung: Condvar,
}

impl Ticks {
    fn tick(&self) -> i64 {
        let mut n = self.n.lock().unwrap();
        n.0 += 1;
        self.rung.notify_all();
        n.0 as i64
    }

    /// `want` ticks' count once they have all come, or `-1` after [`GIVE_UP`]. Counts a wait
    /// begun first.
    fn wait_for(&self, want: u64) -> i64 {
        let mut n = self.n.lock().unwrap();
        n.1 += 1;
        self.rung.notify_all();
        self.until(n, |n| n.0 >= want, |n| n.0)
    }

    /// The waits begun once `want` have, or `-1` after [`GIVE_UP`].
    fn started(&self, want: u64) -> i64 {
        let n = self.n.lock().unwrap();
        self.until(n, |n| n.1 >= want, |n| n.1)
    }

    fn until(
        &self,
        n: std::sync::MutexGuard<'_, (u64, u64)>,
        done: impl Fn(&(u64, u64)) -> bool,
        count: impl Fn(&(u64, u64)) -> u64,
    ) -> i64 {
        let (n, waited) = self
            .rung
            .wait_timeout_while(n, GIVE_UP, |n| !done(n))
            .unwrap();
        if waited.timed_out() {
            -1
        } else {
            count(&n) as i64
        }
    }
}

/// Who ticks while a root waits.
#[derive(Clone, Copy, Debug)]
enum Ticker {
    /// A detached child (func 1's child image), granted `tick` by name.
    Child,
    /// A thread of the root's domain (func 2), handed the root's `tick`.
    Thread,
}

/// A 1 ms timed wait on a word nobody notifies, when `paced`.
fn pace(paced: bool) -> &'static str {
    if paced {
        "  vpa = i64.const 17800\n  vpe = i32.const 0\n  vpt = i64.const 1000000\n  vps = i32.atomic.wait vpa vpe vpt\n"
    } else {
        ""
    }
}

/// A ticker's loop, after an entry block that leaves `tick`'s handle in `vh`: `TICKS` calls, each
/// after a 1 ms timed wait when `paced`. Returns the last tick's count.
fn tick_loop(entry: &str, paced: bool) -> String {
    let pace = pace(paced);
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

/// Stores naming module `vmod` and budget `vbud` in the spawn record at `at`, and that record's
/// address in `vrp{k}` (registers suffixed `k`, so a root can spawn more than once).
fn spawn_prelude(at: u64, vmod: &str, k: &str) -> String {
    format!(
        "  vma{k} = i64.const {module_at}
  i32.store vma{k} {vmod}
  vba{k} = i64.const {budget_at}
  i32.store vba{k} vbud
  vrp{k} = i64.const {at}
",
        module_at = at + SpawnRec::MODULE_AT,
        budget_at = at + SpawnRec::BUDGET_AT,
    )
}

/// A root `(inst, tick, wait, budget, child) -> i64` that starts `ticker`, waits in `wait` for
/// `TICKS` ticks, joins the ticker and returns what the wait returned.
fn src(ticker: Ticker, paced: bool) -> String {
    let (start, end) = match ticker {
        Ticker::Child => (
            format!(
                "{grant}{prelude}  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
",
                grant = rec::grant("g", 17600, 17700, 4, "vtick"),
                prelude = spawn_prelude(17408, "vmod", ""),
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

/// The two children a waiting-child root spawns (#2424): func 1 waits for `2 * TICKS` ticks; func
/// 2 waits for `TICKS` and then ticks `TICKS` times itself. Each finds `wait` and `tick` by name.
/// With `in_fiber`, func 2 makes its wait from a fiber of its own (func 3), which it resumes until
/// the fiber returns: the fiber parks, and the child with it only while it resumes the fiber.
fn waiting_children(in_fiber: bool) -> String {
    let resolve = |name_at: u64, v: &str| {
        format!(
            "  {v}p = i64.const {name_at}\n  {v}l = i64.const 4\n  {v} = self.resolve {v}p {v}l\n"
        )
    };
    let (wait, tick) = (resolve(17710, "vh"), resolve(17700, "vt"));
    // Func 2's wait, which leaves its answer in `vw`, and the block its ticks start in.
    let (func2_wait, ticks_at) = if in_fiber {
        (
            format!(
                "{wait}  vf = ref.func 3
  vsp = i64.const 0
  vk = cont.new vf vsp
  vh64 = i64.extend_i32_u vh
  br 1(vk, vh64)
}}
block 1 (vk1: i64, varg: i64) {{
  vs, vv = cont.resume.block vk1 varg
  vthree = i32.const 3
  vparked = i32.eq vs vthree
  br_if vparked 1(vk1, varg) 2(vv)
}}
block 2 (vw: i64) {{
"
            ),
            3,
        )
    } else {
        (
            format!(
                "{wait}  vwant = i64.const {TICKS}
  vw = call.cap 13 0 (i64) -> (i64) vh (vwant)
"
            ),
            1,
        )
    };
    let done_at = ticks_at + 1;
    format!(
        "func (i64) -> (i64) {{
block 0 (v0: i64) {{
{wait}  vwant = i64.const {double}
  vw = call.cap 13 0 (i64) -> (i64) vh (vwant)
  return vw
}}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{func2_wait}{tick}  vn = i64.const {TICKS}
  br {ticks_at}(vt, vw, vn)
}}
block {ticks_at} (vt1: i32, vw1: i64, vleft: i64) {{
  vz = i64.const 0
  vtk = call.cap 13 0 (i64) -> (i64) vt1 (vz)
  vdec = i64.const 1
  vleft2 = i64.sub vleft vdec
  vmore = i64.ne vleft2 vz
  br_if vmore {ticks_at}(vt1, vw1, vleft2) {done_at}(vw1)
}}
block {done_at} (vw2: i64) {{
  return vw2
}}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vh = i32.wrap_i64 varg
  vwant = i64.const {TICKS}
  vw = call.cap 13 0 (i64) -> (i64) vh (vwant)
  return vw
}}
}}
",
        double = 2 * TICKS,
    )
}

/// A root `(inst, tick, wait, budget, child 1, child 2) -> i64` whose children wait (#2424). With
/// one child, it spawns func 2, waits in `wait` op 1 until that child's wait has begun, ticks
/// `TICKS` times (after a timer each, when `paced`) and returns what the child's wait returned.
/// With two, it first spawns func 1, waiting until its wait has begun too: func 1's wait ends only
/// if func 2's answer reaches func 2 while func 1 still waits, as only func 2's own ticks come
/// after. A wait the root did not see begin costs 1000.
fn child_waits_src(children: u64, paced: bool, in_fiber: bool) -> String {
    let grants = format!(
        "{}{}",
        rec::grant("gw", 17600, 17710, 4, "vwait"),
        rec::grant("gt", 17616, 17700, 4, "vtick")
    );
    let spawn = |k: u64, at: u64, vmod: &str| {
        format!(
            "{prelude}  vch{k} = call.cap 6 17 (i64) -> (i32) vinst (vrp{k})
  vwk{k} = i64.const {k}
  vs{k} = call.cap 13 1 (i64) -> (i64) vwait (vwk{k})
",
            prelude = spawn_prelude(at, vmod, &k.to_string()),
        )
    };
    let pace = pace(paced);
    let entry = "block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32, vm1: i32, vm2: i32) {";
    let root = if children == 1 {
        format!(
            "{entry}
{grants}{spawn}  vone = i64.const 1
  vlate = i64.sub vs1 vone
  vk = i64.const 1000
  vmiss = i64.mul vlate vk
  vn = i64.const {TICKS}
  br 1(vinst, vtick, vch1, vmiss, vn)
}}
block 1 (vi: i32, vh: i32, vc: i32, vm: i64, vleft: i64) {{
{pace}  vz = i64.const 0
  vt = call.cap 13 0 (i64) -> (i64) vh (vz)
  vdec = i64.const 1
  vleft2 = i64.sub vleft vdec
  vmore = i64.ne vleft2 vz
  br_if vmore 1(vi, vh, vc, vm, vleft2) 2(vi, vc, vm)
}}
block 2 (vi2: i32, vc2: i32, vm2: i64) {{
  vst, vj = call.cap 6 1 (i32) -> (i64, i64) vi2 (vc2)
  vr = i64.add vj vm2
  return vr
}}",
            spawn = spawn(1, 17504, "vm2"),
        )
    } else {
        format!(
            "{entry}
{grants}{first}{second}  vtwo = i64.const 2
  vlate = i64.sub vs2 vtwo
  vk = i64.const 1000
  vmiss = i64.mul vlate vk
  vn = i64.const {TICKS}
  br 1(vinst, vtick, vch1, vch2, vmiss, vn)
}}
block 1 (vi: i32, vh: i32, vca: i32, vcb: i32, vm: i64, vleft: i64) {{
{pace}  vz = i64.const 0
  vt = call.cap 13 0 (i64) -> (i64) vh (vz)
  vdec = i64.const 1
  vleft2 = i64.sub vleft vdec
  vmore = i64.ne vleft2 vz
  br_if vmore 1(vi, vh, vca, vcb, vm, vleft2) 2(vi, vca, vcb, vm)
}}
block 2 (vi2: i32, vca2: i32, vcb2: i32, vm2: i64) {{
  vstb, vjb = call.cap 6 1 (i32) -> (i64, i64) vi2 (vcb2)
  vsta, vja = call.cap 6 1 (i32) -> (i64, i64) vi2 (vca2)
  vr = i64.add vja vm2
  return vr
}}",
            first = spawn(1, 17408, "vm1"),
            second = spawn(2, 17504, "vm2"),
        )
    };
    let rec = |at| {
        rec::segment(
            at,
            &SpawnRec {
                grants_ptr: 17600,
                grants_n: 2,
                ..SpawnRec::default()
            },
        )
    };
    format!(
        "memory 17
data 17700 \"tick\"
data 17710 \"wait\"
func (i32, i32, i32, i32, i32, i32) -> (i64) {{
{root}
}}
{children}{first}{second}",
        children = waiting_children(in_fiber),
        first = rec(17408),
        second = rec(17504),
    )
}

fn module_of(src: &str) -> Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e}\n{src}"));
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// `tick` (op 0): count a tick and ring the bell. Forkable, so a spawn can grant it to a child.
fn tick_proc(ticks: &Arc<Ticks>) -> temen_interp::HostProc {
    let t = Arc::clone(ticks);
    Box::new(move |_op, _args, _mem, _minter| Ok(vec![t.tick()]))
}

/// Who answers a `wait` call (#2495).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Answer {
    /// The domain's offload pool ([`OffloadOutcome::Offload`]).
    Pool,
    /// The embedder, through the call's [`Completer`] ([`OffloadOutcome::Host`]), from a thread of
    /// its own.
    Host,
}

/// `wait`: op 0 `wait(n)` waits for `n` ticks, op 1 `started(k)` for `k` waits to have begun, and
/// op 2 for [`SLOW`], answering `n`; answered by `answer`. Forkable over the same ticks, so a spawn
/// can grant it to a child (#2424).
fn wait_proc(ticks: &Arc<Ticks>, answer: Answer) -> OffloadHostProc {
    let t = Arc::clone(ticks);
    Box::new(move |op, args| {
        let (t, n) = (Arc::clone(&t), args.first().copied().unwrap_or(0) as u64);
        let wait = move || match op {
            1 => t.started(n),
            2 => {
                std::thread::sleep(SLOW);
                n as i64
            }
            _ => t.wait_for(n),
        };
        match answer {
            Answer::Pool => OffloadOutcome::Offload(Box::new(wait)),
            Answer::Host => OffloadOutcome::Host(Box::new(move |c: Completer| {
                std::thread::spawn(move || c.complete(wait()));
            })),
        }
    })
}

/// The root's powerbox: an `Instantiator` over its window, `tick`, `wait` answered by `answer`, an
/// unbounded `Budget` and a child image of each of `kids`, on `lanes` lanes (`-1`: as many as it
/// likes). Fresh ticks for each run.
fn setup<'a>(
    m: &'a Module,
    lanes: i64,
    kids: &'a [u32],
    answer: Answer,
) -> impl Fn() -> (Host, Vec<Value>) + 'a {
    move || {
        let ticks = Arc::new(Ticks::default());
        let mut h = Host::new();
        h.set_lane_cap(lanes);
        let i = h.grant_instantiator();
        let forked = Arc::clone(&ticks);
        let tick = h.grant_host_proc_forkable(
            tick_proc(&ticks),
            Arc::new(move |_pid| Some(ForkedProc::shared(tick_proc(&forked), CapState::Stateless))),
            CapState::Stateless,
        );
        let forked = Arc::clone(&ticks);
        let wait = h.grant_host_proc_offloadable(
            wait_proc(&ticks, answer),
            Some(Arc::new(move |_pid| {
                Some(ForkedProc::offloadable(
                    wait_proc(&forked, answer),
                    CapState::Stateless,
                ))
            })),
            CapState::Stateless,
        );
        let b = h.grant_budget(-1, -1, -1);
        let images = kids
            .iter()
            .map(|&k| h.grant_module(&temen_ir::child_image_at(m, k).expect("child image")));
        let handles: Vec<i32> = [i, tick, wait, b]
            .into_iter()
            .chain(images.collect::<Vec<_>>())
            .collect();
        (h, handles.into_iter().map(Value::I32).collect())
    }
}

/// What the root returned on `driver`, or why there is nothing to compare.
fn on_driver(
    driver: Driver,
    m: &Module,
    lanes: i64,
    kids: &[u32],
    answer: Answer,
) -> Result<i64, String> {
    let ran = run_on(driver, m, &setup(m, lanes, kids, answer)).ok_or("declined the module")?;
    match ran.result {
        Ok(v) if v.len() == 1 => match v[0] {
            Value::I64(r) => Ok(r),
            other => Err(format!("returned {other:?}")),
        },
        other => Err(format!("ended {other:?}")),
    }
}

/// What the root returned on the Cranelift JIT.
fn on_the_jit(m: &Module, lanes: i64, kids: &[u32], answer: Answer) -> Result<i64, String> {
    let (mut host, args) = setup(m, lanes, kids, answer)();
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

/// `m` on every engine, `wait` answered each way, and each that returned anything but `want`.
/// `paced` leaves out the `Vcpu` harness, which orchestrates no futex `Wait`.
fn check_on_every_engine(what: &str, m: &Module, kids: &[u32], want: i64, paced: bool, lanes: i64) {
    let mut wrong = Vec::new();
    let mut each = |engine: String, got: Result<i64, String>| {
        if got != Ok(want) {
            wrong.push(format!("  {engine}: {got:?}"));
        }
    };
    for answer in [Answer::Pool, Answer::Host] {
        for driver in drivers::ALL {
            if paced && driver == Driver::Vcpu {
                continue;
            }
            let got = on_driver(driver, m, lanes, kids, answer);
            each(format!("{driver:?}, {answer:?}-answered"), got);
        }
        let got = on_the_jit(m, lanes, kids, answer);
        each(format!("JIT, {answer:?}-answered"), got);
    }
    assert!(
        wrong.is_empty(),
        "{what}{}{}: want {want} on every engine, but\n{}",
        if paced { ", paced" } else { "" },
        if lanes == 1 { ", one lane" } else { "" },
        wrong.join("\n")
    );
}

/// `m` on each of `drivers` and the JIT, `wait` answered each way, each run returning `want` in
/// under half of [`GIVE_UP`]: well before the call it left waiting could have answered.
fn check_prompt(what: &str, m: &Module, kids: &[u32], want: i64, drivers: &[Driver]) {
    let mut wrong = Vec::new();
    let mut each = |engine: String, run: &dyn Fn() -> Result<i64, String>| {
        let started = Instant::now();
        let got = run();
        let took = started.elapsed();
        if got != Ok(want) || took >= GIVE_UP / 2 {
            wrong.push(format!("  {engine}: {got:?} after {took:?}"));
        }
    };
    for answer in [Answer::Pool, Answer::Host] {
        for &driver in drivers {
            let engine = format!("{driver:?}, {answer:?}-answered");
            each(engine, &|| on_driver(driver, m, -1, kids, answer));
        }
        each(format!("JIT, {answer:?}-answered"), &|| {
            on_the_jit(m, -1, kids, answer)
        });
    }
    assert!(
        wrong.is_empty(),
        "{what}: want {want} on every engine, before its call could answer, but\n{}",
        wrong.join("\n")
    );
}

/// A root waiting while `ticker` ticks.
fn check(ticker: Ticker, paced: bool, lanes: i64) {
    let what = format!("a root waiting on a {ticker:?} ticker");
    let m = module_of(&src(ticker, paced));
    check_on_every_engine(&what, &m, &[1], TICKS as i64, paced, lanes);
}

/// `children` children waiting while their root ticks (#2424), the last from a fiber of its own
/// when `in_fiber`.
fn check_children(children: u64, in_fiber: bool, paced: bool, lanes: i64) {
    let what = format!(
        "{children} waiting child(ren){}",
        if in_fiber { ", in a fiber" } else { "" }
    );
    let m = module_of(&child_waits_src(children, paced, in_fiber));
    check_on_every_engine(&what, &m, &[1, 2], (children * TICKS) as i64, paced, lanes);
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

#[test]
fn a_child_waiting_on_the_host_lets_its_root_run() {
    check_children(1, false, false, -1);
}

#[test]
fn a_child_waiting_on_the_host_lets_its_roots_timer_fire() {
    check_children(1, false, true, -1);
}

#[test]
fn a_child_waiting_on_the_host_gives_its_lane_to_its_root() {
    check_children(1, false, false, 1);
}

#[test]
fn a_childs_answer_waits_for_no_other_childs_call() {
    check_children(2, false, false, -1);
}

/// The child's call is made from a fiber of the child's: the fiber parks in the child's registry,
/// and its answer comes through the child's store.
#[test]
fn a_childs_fiber_waiting_on_the_host_lets_its_root_run() {
    check_children(1, true, false, -1);
}

/// A thread waits on the host; the root, in `wait`'s op 1, waits until that wait has begun, then
/// makes a slow call of its own (op 2, answering 0 after [`SLOW`]), ticks the thread's ticks and
/// joins it. The slow call is its domain's later one: the oracle delivered a domain's answers in
/// the order of its calls, so the root's waited for the thread's, which waited for the root's
/// ticks, until it gave up (#2477). A wait the root did not see begin costs 1000.
#[test]
fn a_later_answer_waits_for_no_earlier_call_of_its_domain() {
    let m = module_of(&format!(
        "memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32) {{
  vsp = i64.const 0
  vwa = i64.extend_i32_u vwait
  vth = thread.spawn 1 vsp vwa
  vone = i64.const 1
  vs = call.cap 13 1 (i64) -> (i64) vwait (vone)
  vzero = i64.const 0
  vq = call.cap 13 2 (i64) -> (i64) vwait (vzero)
  vsq = i64.add vs vq
  vn = i64.const {TICKS}
  br 1(vtick, vth, vsq, vn)
}}
block 1 (vh: i32, vt: i32, vs1: i64, vleft: i64) {{
  vz = i64.const 0
  vtk = call.cap 13 0 (i64) -> (i64) vh (vz)
  vdec = i64.const 1
  vleft2 = i64.sub vleft vdec
  vmore = i64.ne vleft2 vz
  br_if vmore 1(vh, vt, vs1, vleft2) 2(vt, vs1)
}}
block 2 (vt2: i32, vs2: i64) {{
  vj = thread.join vt2
  vone2 = i64.const 1
  vlate = i64.sub vs2 vone2
  vk = i64.const 1000
  vmiss = i64.mul vlate vk
  vr = i64.add vj vmiss
  return vr
}}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vh = i32.wrap_i64 varg
  vwant = i64.const {TICKS}
  vw = call.cap 13 0 (i64) -> (i64) vh (vwant)
  return vw
}}
}}"
    ));
    let what = "a root's call while its thread waits";
    check_on_every_engine(what, &m, &[], TICKS as i64, false, -1);
}

/// A root resumes a fiber of its own, blocking, until the fiber's slow host call answers: the
/// resumer waits on the world through its fiber, and nothing else in the run is live. The JIT
/// counted it blocked and, the run then reading as deadlocked, trapped `ThreadFault`.
#[test]
fn a_root_resuming_its_fiber_through_a_slow_answer_is_not_deadlocked() {
    let m = module_of(
        "func (i32, i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32) {
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vh64 = i64.extend_i32_u vwait
  br 1(vk, vh64)
}
block 1 (vk1: i64, varg: i64) {
  vs, vv = cont.resume.block vk1 varg
  vthree = i32.const 3
  vparked = i32.eq vs vthree
  br_if vparked 1(vk1, varg) 2(vv)
}
block 2 (vr: i64) {
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vn = i64.const 7
  vr = call.cap 13 2 (i64) -> (i64) vh (vn)
  return vr
}
}",
    );
    check_on_every_engine("a root resuming its slow fiber", &m, &[], 7, false, -1);
}

/// A root kills a child that waits on a host call nobody will answer, and joins it (#2493). The
/// root first waits, in `wait`'s op 1, until the child's wait has begun, and returns what that
/// returned. The child's teardown used to wait for its call, through its pool, so the join did too.
/// On every driver that schedules its children (the `Vcpu` harness cannot kill one yet, #2083).
#[test]
fn a_killed_child_joins_while_its_call_still_waits() {
    let child = SpawnRec {
        grants_ptr: 17600,
        grants_n: 1,
        ..SpawnRec::default()
    };
    let m = module_of(&format!(
        "memory 17
data 17710 \"wait\"
func (i32, i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32, vmod: i32) {{
{grant}{prelude}  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vone = i64.const 1
  vs = call.cap 13 1 (i64) -> (i64) vwait (vone)
  vk = call.cap 6 12 (i32) -> (i32) vinst (vch)
  vst, vj = call.cap 6 1 (i32) -> (i64, i64) vinst (vch)
  return vs
}}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vp = i64.const 17710
  vl = i64.const 4
  vh = self.resolve vp vl
  vwant = i64.const {TICKS}
  vw = call.cap 13 0 (i64) -> (i64) vh (vwant)
  return vw
}}
}}
{rec}",
        grant = rec::grant("gw", 17600, 17710, 4, "vwait"),
        prelude = spawn_prelude(17408, "vmod", ""),
        rec = rec::segment(17408, &child),
    ));
    let what = "a killed child waiting on the host";
    check_prompt(what, &m, &[1], 1, &drivers::SCHEDULING);
}

/// A root returns over a fiber of its own still waiting on a host call nobody will answer
/// (#2493): the run ends at once, with the resume's `FIBER_PARKED` (3). The run's end used to wait
/// for the call. On every driver that parks the fiber alone: the debugger parks its thread, to keep
/// its schedule its own (#2047), and the `Vcpu` harness waits inline (I45).
#[test]
fn a_run_ends_while_its_fiber_waits_on_the_host() {
    let m = module_of(&format!(
        "func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vtick: i32, vwait: i32, vbud: i32) {{
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vh64 = i64.extend_i32_u vwait
  vs, vv = cont.resume vk vh64
  vr = i64.extend_i32_u vs
  return vr
}}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vh = i32.wrap_i64 varg
  vn = i64.const {TICKS}
  vr = call.cap 13 0 (i64) -> (i64) vh (vn)
  return vr
}}
}}"
    ));
    let fiber_parks = [Driver::Oracle, Driver::Coop, Driver::Parallel];
    check_prompt("a run over a waiting fiber", &m, &[], 3, &fiber_parks);
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
        None,
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
