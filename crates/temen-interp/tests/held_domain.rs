//! #2695 step 2 — a domain its embedder **holds** ([`HeldDomain`]), on every executor that holds
//! one: the tree-walker, the oracle, and the bytecode engine's pump. It outlives its entry and takes
//! calls until it ends itself or its embedder ends it. Between calls its threads keep running. Each
//! call, the embedder's or a domain's of its run, runs as an activation: a new fiber in the domain at
//! the op's handler, called with exactly the op's arguments, on the caller's thread, which funds it.
//! The domain's calls run one at a time, each until it parks (`Single`, DESIGN.md §12a).

#[path = "support/rec.rs"]
mod rec;

use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
use std::sync::{mpsc, Arc};
use std::time::Duration;
use temen_interp::bytecode::ThreadPlatform;
use temen_interp::{Executor, HeldDomain, Host, Reply, Trap, Value, GRANT_SERVE_LIVE_TAG};
use temen_ir::SpawnRec;

/// The ops of the domain's one offer, `"svc"`.
const PAIR: u32 = 0;
const BOOM: u32 = 1;
const QUIT: u32 = 2;
const JOIN_WORKER: u32 = 3;
const WAIT_SIGNAL: u32 = 4;
const SIGNAL: u32 = 5;
const TOTAL: u32 = 7;
const JOIN_CHILD: u32 = 8;
const TLS: u32 = 9;
const OBSERVE: u32 = 10;
const BUMP: u32 = 11;

/// The entries, each answering 1: the plain one, one that spawns a worker thread, one that spawns a
/// child domain, the same waiting 50 ms before it returns, and one that traps.
const ENTRY: u32 = 0;
const ENTRY_WITH_WORKER: u32 = 1;
const ENTRY_WITH_CHILD: u32 = 2;
const ENTRY_WITH_CHILD_THEN_WAIT: u32 = 3;
const ENTRY_TRAPS: u32 = 4;

/// `CAP_REVOKED`: what a call into a dead domain answers.
const CAP_REVOKED: i64 = -9;
const EINVAL: i64 = -22;

/// The held domain. Every entry stores 7 at 16384 and sets its thread-local word to 1234; each
/// handler is one op of `"svc"`:
///
/// - `pair(x)` answers `(x + mem[16384], 2x)`.
/// - `boom()` traps `Unreachable`; `quit()` exits 3 through the `Exit` handle the entry kept.
/// - `join_worker()` releases the worker thread [`ENTRY_WITH_WORKER`] spawned (it waits on 16416),
///   joins it and answers what it returns: `mem[16384] * 10`.
/// - `wait_signal()` waits until 16424 is set and answers it; `signal()` sets it to 1.
/// - `run(a, b)` answers `a + b` and adds it to 16432, which `total()` answers.
///   [`ENTRY_WITH_CHILD`]'s child calls `run` through the live self-serve grant (#744) its spawn
///   gives it, named `"exec"`, and `join_child()` joins the child and answers its result.
/// - `tls()` answers the thread-local word it runs with.
/// - `observe(n)` reads 16464, spins `n` times, and answers how far 16464 moved meanwhile; `bump()`
///   adds 1 to it.
fn parent() -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
data 16684 "exec"
{rec}type 0 func (i64) -> (i64, i64)
type 1 func () -> ()
type 2 func () -> (i64)
type 3 func (i64, i64) -> (i64)
type 4 func (i64) -> (i64)
type 5 interface {{ pair: 0, boom: 1, quit: 1, join_worker: 2, wait_signal: 2, signal: 2, run: 3, total: 2, join_child: 2, tls: 2, observe: 4, bump: 2 }}
export 0 interface "svc" 5 {{ pair: 5, boom: 6, quit: 7, join_worker: 8, wait_signal: 9, signal: 10, run: 11, total: 12, join_child: 13, tls: 14, observe: 15, bump: 16 }}

func (i32) -> (i64) {{
block 0 (vexit: i32) {{
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vh = i64.const 16400
  i32.store vh vexit
  vt = i64.const 1234
  vcpu.tls.set vt
  vr = i64.const 1
  return vr
  }}
}}

func (i32) -> (i64) {{
block 0 (vexit: i32) {{
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vz = i64.const 0
  vt = thread.spawn 17 vz vz
  vth = i64.const 16408
  i32.store vth vt
  vr = i64.const 1
  return vr
  }}
}}

func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vm: i32, vb: i32) {{
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vsi = i64.const 16444
  i32.store vsi vi
  vma = i64.const {module_at}
  i32.store vma vm
  vba = i64.const {budget_at}
  i32.store vba vb
  vgh = i32.const {tag}
{grant}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vsc = i64.const 16440
  i32.store vsc vch
  vr = i64.const 1
  return vr
  }}
}}

func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vm: i32, vb: i32) {{
  vr = call 2 (vi, vm, vb)
  vc = i64.const 16456
  vexp = i32.const 0
  vto = i64.const 50000000
  vst = i32.atomic.wait vc vexp vto
  return vr
  }}
}}

func (i32) -> (i64) {{
block 0 (vexit: i32) {{
  unreachable
  }}
}}

func (i64) -> (i64, i64) {{
block 0 (vx: i64) {{
  va = i64.const 16384
  vs = i64.load va
  vsum = i64.add vs vx
  vk = i64.const 2
  vd = i64.mul vx vk
  return vsum, vd
  }}
}}

func () -> () {{
block 0 () {{
  unreachable
  }}
}}

func () -> () {{
block 0 () {{
  vh = i64.const 16400
  vexit = i32.load vh
  vc = i32.const 3
  call.cap 1 0 (i32) -> () vexit (vc)
  return
  }}
}}

func () -> (i64) {{
block 0 () {{
  vc = i64.const 16416
  vone = i32.const 1
  i32.store vc vone
  vn = i32.const 1
  vw = atomic.notify vc vn
  vth = i64.const 16408
  vt = i32.load vth
  vst, vv = thread.join vt
  return vv
  }}
}}

func () -> (i64) {{
block 0 () {{
  vc = i64.const 16424
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vc vexp vto
  vv = i32.load vc
  vr = i64.extend_i32_s vv
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vc = i64.const 16424
  vone = i32.const 1
  i32.store vc vone
  vn = i32.const 1
  vw = atomic.notify vc vn
  vr = i64.const 1
  return vr
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vs = i64.add va vb
  vl = i64.const 16432
  vold = i64.load vl
  vnew = i64.add vold vs
  i64.store vl vnew
  return vs
  }}
}}

func () -> (i64) {{
block 0 () {{
  vl = i64.const 16432
  vt = i64.load vl
  return vt
  }}
}}

func () -> (i64) {{
block 0 () {{
  vsi = i64.const 16444
  vi = i32.load vsi
  vsc = i64.const 16440
  vch = i32.load vsc
  vst, vv = call.cap 6 1 (i32) -> (i64, i64) vi (vch)
  return vv
  }}
}}

func () -> (i64) {{
block 0 () {{
  vt = vcpu.tls.get
  return vt
  }}
}}

func (i64) -> (i64) {{
block 0 (vlim: i64) {{
  vx = i64.const 16464
  v0 = i64.load vx
  vn = i64.const 0
  br 1(v0, vn, vlim)
}}
block 1 (w0: i64, wn: i64, wl: i64) {{
  vgo = i64.lt_u wn wl
  br_if vgo 2(w0, wn, wl) 3(w0)
}}
block 2 (x0: i64, xn: i64, xl: i64) {{
  vone = i64.const 1
  xm = i64.add xn vone
  br 1(x0, xm, xl)
}}
block 3 (y0: i64) {{
  vy = i64.const 16464
  v1 = i64.load vy
  vd = i64.sub v1 y0
  return vd
  }}
}}

func () -> (i64) {{
block 0 () {{
  vx = i64.const 16464
  v0 = i64.load vx
  vone = i64.const 1
  v1 = i64.add v0 vone
  i64.store vx v1
  return v1
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vc = i64.const 16416
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vc vexp vto
  va = i64.const 16384
  vx = i64.load va
  vk = i64.const 10
  vr = i64.mul vx vk
  return vr
  }}
}}
"#,
        rec = rec::segment(17408, &child),
        grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        tag = GRANT_SERVE_LIVE_TAG as i32,
    )
}

/// The child [`ENTRY_WITH_CHILD`] spawns: it resolves `"exec"`, the offer over its parent, calls
/// `run(40, 2)`, then calls `run` again with that answer, `run(42, 0)`, and returns the second.
const CHILD: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 6 (i64, i64) -> (i64) vexec (va, vb)
  vz = i64.const 0
  vr2 = call.cap 268435456 6 (i64, i64) -> (i64) vexec (vr, vz)
  return vr2
  }
}
"#;

/// A child for [`ENTRY_WITH_CHILD`] that calls `bump()` through `"exec"` [`BUMPS`] times, and
/// returns how many it made.
const BUMPER: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vn = i64.const 0
  br 1(vexec, vn)
}
block 1 (we: i32, wn: i64) {
  vlim = i64.const 1000
  vgo = i64.lt_u wn vlim
  br_if vgo 2(we, wn) 3(wn)
}
block 2 (xe: i32, xn: i64) {
  vb = call.cap 268435456 11 () -> (i64) xe ()
  vone = i64.const 1
  xm = i64.add xn vone
  br 1(xe, xm)
}
block 3 (yn: i64) {
  return yn
  }
}
"#;

/// The calls [`BUMPER`] makes.
const BUMPS: i64 = 1000;

/// How long `observe` spins: long enough to measure, short of a quantum of the pump.
const SPIN: i64 = 20_000;

/// How long `observe` spins to outlast several of the pump's quanta, so the pump runs other tasks
/// while it does.
const LONG_SPIN: i64 = 400_000;

fn module(src: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    Arc::new(m)
}

/// `f` on a thread of its own, so a run that hangs fails the test instead of stalling the suite.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the run ends")
}

/// A powerbox for the domain: its offer, and an `Exit` handle, which [`args`] passes the entry.
fn host_for(m: &Arc<temen_ir::Module>) -> (Host, i32) {
    let mut host = Host::new();
    host.set_self_module(m);
    let exit = host.grant_exit();
    (host, exit)
}

/// Hold `entry` of `m` on `on` with `args` and `host`, which must be where it runs: the executor
/// does not decline the module.
fn hold(
    on: Executor,
    m: Arc<temen_ir::Module>,
    entry: u32,
    args: &[Value],
    host: Host,
) -> (HeldDomain, Vec<Value>) {
    let started = HeldDomain::start(on, m, entry, args, u64::MAX, host);
    let (d, results) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
    assert_eq!(d.executor(), on, "the domain runs where it was started");
    (d, results)
}

/// Start `entry` of the parent on `on` with the `Exit` handle.
fn start(on: Executor, entry: u32) -> HeldDomain {
    let m = module(&parent());
    let (host, exit) = host_for(&m);
    let (domain, results) = hold(on, m, entry, &[Value::I32(exit)], host);
    assert_eq!(results, vec![Value::I64(1)], "start answers the entry");
    domain
}

/// A call of `op` with `args` and fuel to spare.
fn call(d: &HeldDomain, op: u32, args: &[i64]) -> Reply {
    let mut fuel = 1_000_000;
    d.call(0, op, args, &mut fuel)
}

/// Each case runs on every executor that holds domains: the tree-walker, the oracle, the pump and
/// executor 2.
macro_rules! on_every_executor {
    ($($case:ident),* $(,)?) => {$(
        mod $case {
            #[test]
            fn tree_walker() {
                super::$case(temen_interp::Executor::TreeWalker);
            }

            #[test]
            fn pump() {
                super::$case(temen_interp::Executor::Pump);
            }

            #[test]
            fn threads() {
                super::$case(temen_interp::Executor::Threads);
            }
        }
    )*};
}

on_every_executor!(
    a_held_domain_takes_calls_after_its_entry_returns,
    a_call_runs_on_the_fuel_it_is_lent,
    an_activation_sees_the_main_threads_thread_local_word,
    a_call_the_domain_cannot_take_is_refused,
    a_trap_in_an_activation_ends_the_domain,
    an_exit_in_an_activation_ends_the_domain,
    a_thread_the_entry_spawned_runs_on_between_calls,
    a_resting_domain_whose_threads_all_wait_is_idle,
    a_parked_call_lets_the_next_call_in,
    a_parked_call_answers_the_domains_end,
    calls_run_one_at_a_time,
    calls_from_inside_the_run_run_one_at_a_time,
    a_call_from_a_child_is_served_while_its_parent_rests,
    a_call_that_arrives_during_the_entry_runs_once_it_returns,
    a_domain_that_ends_in_its_entry_is_not_held,
    a_durable_domain_is_refused,
    dropping_the_handle_ends_the_domain,
);

/// A held domain takes calls once its entry has returned. Each sees what the entry wrote and
/// replies with every result its op declares.
fn a_held_domain_takes_calls_after_its_entry_returns(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        assert_eq!(call(&d, PAIR, &[5]), Reply::Results(vec![12, 10]));
        assert_eq!(call(&d, PAIR, &[30]), Reply::Results(vec![37, 60]));
        let (_host, trap) = d.end();
        assert_eq!(trap, None, "the domain was alive until its end");
    });
}

/// A call runs on the fuel its caller lends it, and what it leaves comes back. One lent too little
/// runs out inside the handler, whose trap ends the domain, as any handler's does (INVARIANTS #6).
fn a_call_runs_on_the_fuel_it_is_lent(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        let mut fuel = 1_000_000u64;
        assert_eq!(
            d.call(0, OBSERVE, &[SPIN], &mut fuel),
            Reply::Results(vec![0])
        );
        assert!(
            fuel <= 1_000_000 - 20_000,
            "the call burned its loop: {fuel} left"
        );
        let mut fuel = 10u64;
        assert_eq!(
            d.call(0, OBSERVE, &[SPIN], &mut fuel),
            Reply::Failed(CAP_REVOKED)
        );
        assert_eq!(fuel, 0, "it burned all it was lent");
        assert_eq!(d.end().1, Some(Trap::OutOfFuel));
    });
}

/// An activation runs as the domain's main thread would: with the thread-local word it left.
fn an_activation_sees_the_main_threads_thread_local_word(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        assert_eq!(call(&d, TLS, &[]), Reply::Results(vec![1234]));
    });
}

/// A call of another arity than its handler, or of an op the export lacks, is refused `-EINVAL`,
/// and the domain goes on serving.
fn a_call_the_domain_cannot_take_is_refused(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        assert_eq!(call(&d, PAIR, &[5, 6]), Reply::Failed(EINVAL));
        assert_eq!(call(&d, 99, &[]), Reply::Failed(EINVAL));
        assert_eq!(call(&d, PAIR, &[5]), Reply::Results(vec![12, 10]));
    });
}

/// An activation's trap ends the domain, as any member's does: its call and every later one answer
/// `CAP_REVOKED`, and the end reports the trap.
fn a_trap_in_an_activation_ends_the_domain(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        assert_eq!(call(&d, BOOM, &[]), Reply::Failed(CAP_REVOKED));
        assert_eq!(call(&d, PAIR, &[5]), Reply::Failed(CAP_REVOKED));
        let (_host, trap) = d.end();
        assert_eq!(trap, Some(Trap::Unreachable));
    });
}

/// So does its exit.
fn an_exit_in_an_activation_ends_the_domain(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY);
        assert_eq!(call(&d, QUIT, &[]), Reply::Failed(CAP_REVOKED));
        assert_eq!(d.end().1, Some(Trap::Exit(3)));
    });
}

/// A held domain is a process: a thread its entry spawned runs on after the entry returns, and a
/// call releases it, joins it and answers what it computed.
fn a_thread_the_entry_spawned_runs_on_between_calls(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY_WITH_WORKER);
        assert_eq!(call(&d, JOIN_WORKER, &[]), Reply::Results(vec![70]));
    });
}

/// A resting domain whose every thread waits is idle, not deadlocked: its embedder can always call in.
/// Pumped, its worker waits on, and a call then releases it.
fn a_resting_domain_whose_threads_all_wait_is_idle(on: Executor) {
    bounded(move || {
        let d = start(on, ENTRY_WITH_WORKER);
        d.pump();
        assert_eq!(call(&d, JOIN_WORKER, &[]), Reply::Results(vec![70]));
        assert_eq!(d.end().1, None);
    });
}

/// A call that parks lets the next call in: here the second call is what releases the first.
fn a_parked_call_lets_the_next_call_in(on: Executor) {
    bounded(move || {
        let d = Arc::new(start(on, ENTRY));
        let waiter = std::thread::spawn({
            let d = Arc::clone(&d);
            move || call(&d, WAIT_SIGNAL, &[])
        });
        // The waiting call may not have parked yet; the signal releases it either way.
        assert_eq!(call(&d, SIGNAL, &[]), Reply::Results(vec![1]));
        assert_eq!(waiter.join().expect("the waiter"), Reply::Results(vec![1]));
    });
}

/// A call parked when the domain ends answers the end, as a call into a dead domain does: here one
/// waits on a signal that never comes, and another call traps.
fn a_parked_call_answers_the_domains_end(on: Executor) {
    bounded(move || {
        let d = Arc::new(start(on, ENTRY));
        let waiter = std::thread::spawn({
            let d = Arc::clone(&d);
            move || call(&d, WAIT_SIGNAL, &[])
        });
        // Let the waiter park first; if it has not, it finds the domain dead and answers the same.
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(call(&d, BOOM, &[]), Reply::Failed(CAP_REVOKED));
        assert_eq!(
            waiter.join().expect("the waiter"),
            Reply::Failed(CAP_REVOKED)
        );
    });
}

/// The domain's calls run one at a time: while `observe` spins, no `bump` lands, though they are
/// called at once from other threads.
fn calls_run_one_at_a_time(on: Executor) {
    bounded(move || {
        let d = Arc::new(start(on, ENTRY));
        for _ in 0..5 {
            let bumpers: Vec<_> = (0..3)
                .map(|_| {
                    let d = Arc::clone(&d);
                    std::thread::spawn(move || call(&d, BUMP, &[]))
                })
                .collect();
            assert_eq!(call(&d, OBSERVE, &[SPIN]), Reply::Results(vec![0]));
            for b in bumpers {
                assert!(matches!(b.join().expect("a bumper"), Reply::Results(_)));
            }
        }
    });
}

/// The powerbox for [`ENTRY_WITH_CHILD`], and its arguments: an `Instantiator`, the module of the
/// child it spawns, `child`, and a budget for it.
fn child_host(m: &Arc<temen_ir::Module>, child: &str) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_self_module(m);
    let args = vec![
        Value::I32(host.grant_instantiator()),
        Value::I32(host.grant_module(&module(child))),
        Value::I32(host.grant_budget(-1, 1 << 20, -1)),
    ];
    (host, args)
}

/// A call from inside the run reaches the held domain too: the child its entry spawned calls
/// `run(40, 2)` through the live self-serve grant, gets 42 back, and calls `run(42, 0)`.
fn a_child_calls_its_held_parent(on: Executor, entry: u32) {
    bounded(move || {
        let m = module(&parent());
        let (host, args) = child_host(&m, CHILD);
        let (d, _) = hold(on, m, entry, &args, host);
        assert_eq!(call(&d, JOIN_CHILD, &[]), Reply::Results(vec![42]));
        assert_eq!(call(&d, TOTAL, &[]), Reply::Results(vec![42 + 42]));
    });
}

/// A call from inside the run waits for the gate as the embedder's do: a child bumps while the
/// embedder's `observe` spins, and no bump lands while it does. On the pump the spin outlasts its
/// quanta, so the pump runs the child and its bumps' activations meanwhile.
fn calls_from_inside_the_run_run_one_at_a_time(on: Executor) {
    bounded(move || {
        let m = module(&parent());
        let (host, args) = child_host(&m, BUMPER);
        let (d, _) = hold(on, m, ENTRY_WITH_CHILD, &args, host);
        for _ in 0..3 {
            assert_eq!(call(&d, OBSERVE, &[LONG_SPIN]), Reply::Results(vec![0]));
        }
        assert_eq!(call(&d, JOIN_CHILD, &[]), Reply::Results(vec![BUMPS]));
    });
}

fn a_call_from_a_child_is_served_while_its_parent_rests(on: Executor) {
    a_child_calls_its_held_parent(on, ENTRY_WITH_CHILD);
}

/// The same, with the parent's entry waiting 50 ms before it returns, so the child's first call
/// most likely arrives while the entry runs: it waits, and runs once the domain rests.
fn a_call_that_arrives_during_the_entry_runs_once_it_returns(on: Executor) {
    a_child_calls_its_held_parent(on, ENTRY_WITH_CHILD_THEN_WAIT);
}

/// A domain whose entry traps is never held: `start` answers the trap and the powerbox.
fn a_domain_that_ends_in_its_entry_is_not_held(on: Executor) {
    bounded(move || {
        let m = module(&parent());
        let (host, exit) = host_for(&m);
        let started = HeldDomain::start(on, m, ENTRY_TRAPS, &[Value::I32(exit)], u64::MAX, host);
        assert!(matches!(started, Err((Trap::Unreachable, _))));
    });
}

/// A durable domain cannot be held yet: it is refused.
fn a_durable_domain_is_refused(on: Executor) {
    bounded(move || {
        let m = module(&parent());
        let (mut host, exit) = host_for(&m);
        host.set_durable(true);
        let started = HeldDomain::start(on, m, ENTRY, &[Value::I32(exit)], u64::MAX, host);
        assert!(matches!(started, Err((Trap::ThreadFault, _))));
    });
}

/// Dropping the handle ends the domain, its parked thread included.
fn dropping_the_handle_ends_the_domain(on: Executor) {
    bounded(move || drop(start(on, ENTRY_WITH_WORKER)));
}

/// The threads [`COUNTING`] has started, in this test binary.
static STARTED: AtomicUsize = AtomicUsize::new(0);

/// Executor 2's OS platform, counting the threads its runs start.
const COUNTING: ThreadPlatform = ThreadPlatform {
    spawn: |f, window| {
        STARTED.fetch_add(1, SeqCst);
        (ThreadPlatform::OS.spawn)(f, window)
    },
    ..ThreadPlatform::OS
};

/// Executor 2 starts a thread for each of the domain's own threads, and none for a call: the call
/// runs on its caller's thread, the embedder's or that of a task inside the run (INVARIANTS #4,
/// D56). The one test that counts.
#[test]
fn a_call_starts_no_thread_on_executor_2() {
    bounded(|| {
        let m = module(&parent());
        let (mut host, exit) = host_for(&m);
        host.set_thread_platform(COUNTING);
        let at = STARTED.load(SeqCst);
        let (d, _) = hold(
            Executor::Threads,
            m,
            ENTRY_WITH_WORKER,
            &[Value::I32(exit)],
            host,
        );
        assert_eq!(STARTED.load(SeqCst) - at, 1, "the worker's thread");
        for x in 0..10 {
            assert_eq!(call(&d, PAIR, &[x]), Reply::Results(vec![x + 7, 2 * x]));
        }
        assert_eq!(call(&d, JOIN_WORKER, &[]), Reply::Results(vec![70]));
        assert_eq!(
            STARTED.load(SeqCst) - at,
            1,
            "an embedder's call starts no thread"
        );
        drop(d);
        // A child's calls into its held parent run on the child's thread.
        let m = module(&parent());
        let (mut host, args) = child_host(&m, CHILD);
        host.set_thread_platform(COUNTING);
        let at = STARTED.load(SeqCst);
        let (d, _) = hold(Executor::Threads, m, ENTRY_WITH_CHILD, &args, host);
        assert_eq!(call(&d, JOIN_CHILD, &[]), Reply::Results(vec![42]));
        assert_eq!(
            STARTED.load(SeqCst) - at,
            1,
            "the child's thread, and none for its calls"
        );
    });
}
