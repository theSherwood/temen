//! #2695 step 2 — a domain its embedder **holds** ([`HeldDomain`]), on every executor
//! ([`temen_run::hold`]): the tree-walker, the oracle, the bytecode engine's pump and executor 2, and
//! the Cranelift JIT. It outlives its entry and takes
//! calls until it ends itself or its embedder ends it. Between calls its threads keep running. Each
//! call, the embedder's or a domain's of its run, runs as an activation: a new fiber in the domain at
//! the op's handler, called with exactly the op's arguments, on the caller's thread, which funds it.
//! The domain's calls run one at a time, each until it parks (`Single`, DESIGN.md §12a).

#[path = "../../temen-interp/tests/support/rec.rs"]
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
    let started = temen_run::hold(on, m, entry, args, u64::MAX, host);
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

/// Each case runs on every executor: the tree-walker, the oracle, the pump, executor 2 and the JIT.
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

            #[test]
            fn jit() {
                super::$case(temen_interp::Executor::Jit);
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
        let started = temen_run::hold(on, m, ENTRY_TRAPS, &[Value::I32(exit)], u64::MAX, host);
        assert!(matches!(started, Err((Trap::Unreachable, _))));
    });
}

/// A durable domain cannot be held yet: it is refused.
fn a_durable_domain_is_refused(on: Executor) {
    bounded(move || {
        let m = module(&parent());
        let (mut host, exit) = host_for(&m);
        host.set_durable(true);
        let started = temen_run::hold(on, m, ENTRY, &[Value::I32(exit)], u64::MAX, host);
        assert!(matches!(started, Err((Trap::ThreadFault, _))));
    });
}

/// Dropping the handle ends the domain, its parked thread included.
fn dropping_the_handle_ends_the_domain(on: Executor) {
    bounded(move || drop(start(on, ENTRY_WITH_WORKER)));
}

/// A domain whose entry answers 1 and whose one op, `seven()`, answers 7. Its third func serves in
/// `svc.wait`, though nothing runs it: what the module can do decides where it is held.
const SERVING: &str = r#"memory 16
type 0 func () -> (i64)
type 1 interface { seven: 0 }
export 0 interface "svc" 1 { seven: 1 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vr = i64.const 7
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}
"#;

/// A domain that can serve its offers in a serve loop (`svc.poll`, `svc.wait`) is held by the
/// tree-walker when it is asked for on the JIT: a thread of it parked in `svc.wait` takes its calls
/// first, which the JIT's activations do not reach yet (#2895). An executor declines what it would
/// run otherwise than the oracle does (INVARIANTS #9).
#[test]
fn a_domain_that_serves_is_held_by_the_tree_walker_on_the_jit() {
    bounded(|| {
        let m = module(SERVING);
        let mut host = Host::new();
        host.set_self_module(&m);
        let started = temen_run::hold(Executor::Jit, m, 0, &[], u64::MAX, host);
        let (d, results) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        assert_eq!(d.executor(), Executor::TreeWalker);
        assert_eq!(results, vec![Value::I64(1)]);
        assert_eq!(call(&d, 0, &[]), Reply::Results(vec![7]));
    });
}

/// #3133 — a domain whose handler makes a fiber that outlives its call: `mk(start)` makes a fiber
/// that counts up from 41 and suspends each count, resumes it once first when `start` is not 0, and
/// answers its handle; `go(h)` resumes it and answers `status · 1000 + value`.
const FIBER_MAKER: &str = r#"memory 16
type 0 func (i64) -> (i64)
type 1 interface { mk: 0, go: 0 }
export 0 interface "svc" 1 { mk: 1, go: 2 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vstart: i64) {
  vf = ref.func 3
  vz = i64.const 0
  vk = cont.new vf vz
  vgo = i64.ne vstart vz
  br_if vgo 1(vk) 2(vk)
}
block 1 (vk1: i64) {
  vz1 = i64.const 0
  vs, vv = cont.resume vk1 vz1
  br 2(vk1)
}
block 2 (vk2: i64) {
  return vk2
  }
}
func (i64) -> (i64) {
block 0 (vh: i64) {
  vz = i64.const 0
  vs, vv = cont.resume vh vz
  vs64 = i64.extend_i32_s vs
  vk = i64.const 1000
  vm = i64.mul vs64 vk
  vr = i64.add vm vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vc = i64.const 41
  br 1(vc)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vj = i64.add vi vone
  vx = suspend vj
  br 1(vj)
  }
}
"#;

/// #3133, #3080 — a domain that makes fibers is held on the JIT, and each call there answers and
/// leaves the fuel the tree-walker's does. A fiber made in an activation runs on a context of its own,
/// which outlives the call, and each resume names the cell of the activation that runs it, so the
/// call that resumes a fiber is the call that pays for it, as on the tree-walker. (Until #3080 the
/// JIT declined it: its fiber charged the call that made it, and on the JIT a later resume ran on
/// that call's freed context and cell, #3133.) Each fiber is made in one call and resumed by the
/// later ones, never started in its first call or started and suspended there, each call on a loan
/// and on none.
#[test]
fn a_domain_that_makes_fibers_is_held_on_the_jit_and_charges_as_the_tree_walker() {
    bounded(|| {
        for start in [0, 1] {
            for loan in [u64::MAX, 1_000_000] {
                let on = |on: Executor| {
                    let m = module(FIBER_MAKER);
                    let mut host = Host::new();
                    host.set_self_module(&m);
                    let started = temen_run::hold(on, m, 0, &[], u64::MAX, host);
                    let (d, results) =
                        started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
                    assert_eq!(d.executor(), on, "start {start}, loan {loan}");
                    assert_eq!(results, vec![Value::I64(1)]);
                    let mut calls = Vec::new();
                    let mut fuel = loan;
                    calls.push((d.call(0, 0, &[start], &mut fuel), fuel));
                    for _ in 0..3 {
                        let mut fuel = loan;
                        calls.push((d.call(0, 1, &[0], &mut fuel), fuel));
                    }
                    let (_host, trap) = d.end();
                    assert_eq!(trap, None);
                    calls
                };
                let oracle = on(Executor::TreeWalker);
                let replies: Vec<&Reply> = oracle.iter().map(|(r, _)| r).collect();
                let want: Vec<Reply> = [0, 42 + start, 43 + start, 44 + start]
                    .map(|v| Reply::Results(vec![v]))
                    .to_vec();
                assert_eq!(
                    replies,
                    want.iter().collect::<Vec<_>>(),
                    "the fiber's handle, then its counts"
                );
                // On no loan (`u64::MAX`) the JIT runs a call unmetered and reports none burned,
                // where the tree-walker counts down from it (#3146): the replies are compared there.
                let jit = on(Executor::Jit);
                if loan == u64::MAX {
                    let jit: Vec<&Reply> = jit.iter().map(|(r, _)| r).collect();
                    assert_eq!(jit, replies, "start {start}, no loan: each call's reply");
                } else {
                    assert_eq!(
                        jit, oracle,
                        "start {start}, loan {loan}: each call's reply and fuel left"
                    );
                }
            }
        }
    });
}

/// #3080 — `mk()` makes a fiber that runs 10,000 back-edges and returns 7, and answers its handle;
/// `go(h)` resumes it to its end and answers what it returned.
const A_LONG_FIBER: &str = r#"memory 16
type 0 func (i64) -> (i64)
type 1 interface { mk: 0, go: 0 }
export 0 interface "svc" 1 { mk: 1, go: 2 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (va: i64) {
  vf = ref.func 3
  vz = i64.const 0
  vk = cont.new vf vz
  return vk
  }
}
func (i64) -> (i64) {
block 0 (vh: i64) {
  vz = i64.const 0
  vs, vv = cont.resume vh vz
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vone = i64.const 1
  vj = i64.add vi vone
  vn = i64.const 10000
  vmore = i64.lt_u vj vn
  br_if vmore 1(vj) 2()
}
block 2 () {
  vr = i64.const 7
  return vr
  }
}
"#;

/// #3080 — the call that resumes a fiber is the one it charges: a fiber made in a call lent plenty
/// runs out of fuel in a later call lent 100, which ends the domain, on the JIT as on the
/// tree-walker. (Until #3080 the JIT charged the first call's leftover loan and completed.)
#[test]
fn a_fiber_made_in_one_call_runs_out_in_the_call_that_resumes_it() {
    bounded(|| {
        let on = |on: Executor| {
            let m = module(A_LONG_FIBER);
            let mut host = Host::new();
            host.set_self_module(&m);
            let (d, _) = temen_run::hold(on, m, 0, &[], u64::MAX, host)
                .unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
            assert_eq!(d.executor(), on);
            let mut fuel = 1_000_000;
            let made = (d.call(0, 0, &[0], &mut fuel), fuel);
            let mut fuel = 100;
            let resumed = (d.call(0, 1, &[0], &mut fuel), fuel);
            let (_host, trap) = d.end();
            (made, resumed, trap)
        };
        let oracle = on(Executor::TreeWalker);
        assert_eq!(
            oracle.2,
            Some(Trap::OutOfFuel),
            "the second call ends the domain"
        );
        assert_eq!(on(Executor::Jit), oracle);
    });
}

/// #3133 — an offer of one op, `seven()`, with no fiber op and no serve loop.
const SEVEN: &str = r#"memory 16
type 0 func () -> (i64)
type 1 interface { seven: 0 }
export 0 interface "svc" 1 { seven: 1 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vr = i64.const 7
  return vr
  }
}
"#;

/// #3080 — a domain whose host lets submitted units host fibers, but which cannot reach `Jit`, runs
/// no unit, so the grant alone keeps it off the JIT no longer (until #3080 it did); a plain call
/// there costs what it costs on the tree-walker: its handler's entry is free on both (#3138: the
/// JIT charged it). One that can reach `Jit` is the tree-walker's (#3164, below).
#[test]
fn a_domain_whose_units_may_host_fibers_is_held_on_the_jit() {
    bounded(|| {
        for hosts in [false, true] {
            let on = |on: Executor| {
                let m = module(SEVEN);
                let mut host = Host::new();
                host.set_self_module(&m);
                host.set_jit_hosts_fibers(hosts);
                let started = temen_run::hold(on, m, 0, &[], u64::MAX, host);
                let (d, results) =
                    started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
                assert_eq!(d.executor(), on, "hosting fibers: {hosts}");
                assert_eq!(results, vec![Value::I64(1)]);
                let mut fuel = 1_000_000;
                (d.call(0, 0, &[], &mut fuel), fuel)
            };
            let oracle = on(Executor::TreeWalker);
            assert_eq!(oracle, (Reply::Results(vec![7]), 1_000_000));
            assert_eq!(on(Executor::Jit), oracle, "hosting fibers: {hosts}");
        }
    });
}

/// #3133 — a §14 child whose thread serves one call of its offer `addmul` and ends: the handler
/// makes a fiber over its first argument and leaves its handle at 2048, and the child's root joins
/// the thread and resumes the fiber twice (it suspends 5, then returns `5 + 40`), answering 50.
const SERVES_A_FIBER: &str = r#"memory 12
type 0 func (i64, i64) -> (i64, i64)
type 1 interface { addmul: 0 }
export 0 interface "addmul" 1 { addmul: 1 }
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  vs, vv = thread.join vt
  vat = i64.const 2048
  vk = i64.load vat
  vrs, vr = cont.resume vk vz
  vrs2, vr2 = cont.resume vk vz
  vsum = i64.add vr vr2
  return vsum
  }
}
func (i64, i64) -> (i64, i64) {
block 0 (va: i64, vb: i64) {
  vf = ref.func 3
  vk = cont.new vf va
  vat = i64.const 2048
  i64.store vat vk
  vs = i64.add va vb
  vm = i64.mul va vb
  return vs, vm
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vc = i64.const 5
  vx = suspend vc
  vd = i64.add vc vsp
  return vd
  }
}
"#;

/// The root that spawns [`SERVES_A_FIBER`] detached, calls `addmul(40, 2)` through a live offer over
/// its export, joins it, and answers `join · 1_000_000 + sum · 1000 + product`.
fn calls_a_server() -> String {
    format!(
        r#"memory 17

func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vrm = i64.const 17416
  i32.store vrm v1
  vrb = i64.const 17420
  i32.store vrb v2
  vrp = i64.const 17408
  v5 = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  v6 = i64.const 0
  v7 = call.cap 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vsum, vprod = call.cap 268435456 0 (i64, i64) -> (i64, i64) v7 (va, vb)
  vthousand = i64.const 1000
  vsh = i64.mul vsum vthousand
  vcall = i64.add vsh vprod
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (v5)
  vk = i64.const 1000000
  vm = i64.mul vj vk
  vs = i64.add vm vcall
  return vs
  }}
}}
{rec}"#,
        rec = rec::segment(17408, &SpawnRec::default())
    )
}

/// #3133 — not a held domain; here so the `asan-held` lane runs it. A serve handler runs on its
/// window's context, and a fiber it makes lands in the run's table. When the vCPU that served ended,
/// its serve site went, and the window's context with it, so the child's root resumed the fiber onto
/// freed memory (reproduced on 5326ff0, under ASan with freed memory filled: a read of a pointer from
/// the freed context in compiled code). The fiber now runs on a context of its own. Both engines
/// answer the oracle's 50 · 1_000_000 + 42_080, and the JIT's lane is the JIT's throughout: its root
/// returns a `JitOutcome`, and a JIT root's spawn has no other engine for the child (a child it
/// cannot compile is the spawn's `CapFault`, `temen-jit`'s `instantiator_rt`), so the `42_080` the
/// child serves is a JIT child's.
#[test]
fn a_serve_handlers_fiber_outlives_the_thread_that_served() {
    let caller = module(&calls_a_server());
    let server = module(SERVES_A_FIBER);
    for jit in [false, true] {
        let (caller, server) = (Arc::clone(&caller), Arc::clone(&server));
        let got = bounded(move || {
            let mut host = Host::new();
            host.set_self_module(&caller);
            let args = [
                host.grant_instantiator(),
                host.grant_module(&server),
                host.grant_budget(-1, 1 << 20, -1),
                host.grant_clock(),
            ];
            if jit {
                let image = temen_interp::MemLayout::image(vec![0u8; 128 << 10]);
                let (out, _) = temen_run::jit_cap_run(
                    &caller,
                    0,
                    &args.map(i64::from),
                    &image,
                    0,
                    0,
                    &mut host,
                    None,
                )
                .expect("the JIT compiles the caller");
                format!("{out:?}")
            } else {
                let mut fuel = 5_000_000u64;
                let r = temen_interp::run_with_host(
                    &caller,
                    0,
                    &args.map(Value::I32),
                    &mut fuel,
                    &mut host,
                );
                format!("{r:?}")
            }
        });
        let want = if jit {
            "Returned([50042080])"
        } else {
            "Ok([I64(50042080)])"
        };
        assert_eq!(got, want, "the JIT: {jit}");
    }
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

/// #2877 — a held module that uses `setjmp` and serves no interface: its entry takes a checkpoint
/// and returns, so the domain rests, and every call is refused, because a held domain's activations
/// are calls of its served handlers and it has none. On the JIT no activation ever runs on another
/// thread over the instance's `setjmp` table. (A module that uses `setjmp` and serves is declined
/// at the JIT's compile.)
fn a_held_domain_with_setjmp_and_no_interface_takes_no_call(on: Executor) {
    bounded(move || {
        let m = module(
            "memory 17
func (i32) -> (i64) {
block 0 (vexit: i32) {
  vb = i64.const 40000
  vs = setjmp vb
  vr = i64.extend_i32_u vs
  vone = i64.const 1
  vt = i64.add vr vone
  return vt
  }
}
",
        );
        let (host, exit) = host_for(&m);
        // Not the `hold` helper: off unix the JIT declines `setjmp`, so an `Executor::Jit` hold of
        // this module runs on the interpreters, which must answer the same.
        let started = temen_run::hold(on, m, 0, &[Value::I32(exit)], u64::MAX, host);
        let (d, results) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        // Where the JIT has `setjmp`, it holds the module itself: nothing folds it.
        if cfg!(unix) {
            assert_eq!(d.executor(), on, "the domain runs where it was started");
        }
        assert_eq!(results, vec![Value::I64(1)], "start answers the entry");
        assert_eq!(call(&d, 0, &[]), Reply::Failed(EINVAL), "no op to call");
    });
}

on_every_executor!(a_held_domain_with_setjmp_and_no_interface_takes_no_call);

/// #2877 — a held module that uses `setjmp` and serves an interface. The JIT declines it before it
/// runs (`temen_jit::admits`: its handlers' guarded calls would not nest on the one checkpoint
/// table), so `hold` on `Executor::Jit` holds it on the interpreters, which answer its calls as
/// every other executor does. It used to end in the JIT's compile error as `ThreadFault`.
fn a_held_module_with_setjmp_that_serves_answers_its_calls(on: Executor) {
    bounded(move || {
        let m = module(
            r#"memory 17
type 0 func () -> (i64)
type 1 interface { seven: 0 }
export 0 interface "svc" 1 { seven: 1 }
func (i32) -> (i64) {
block 0 (vexit: i32) {
  vb = i64.const 40000
  vs = setjmp vb
  vr = i64.extend_i32_u vs
  vone = i64.const 1
  vt = i64.add vr vone
  return vt
  }
}
func () -> (i64) {
block 0 () {
  v7 = i64.const 7
  return v7
  }
}
"#,
        );
        let (host, exit) = host_for(&m);
        let started = temen_run::hold(on, m, 0, &[Value::I32(exit)], u64::MAX, host);
        let (d, results) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        assert_eq!(results, vec![Value::I64(1)], "start answers the entry");
        assert_eq!(call(&d, 0, &[]), Reply::Results(vec![7]));
    });
}

on_every_executor!(a_held_module_with_setjmp_that_serves_answers_its_calls);

/// #3000 — a held domain whose own activation calls the domain: its child hands the parent the live
/// offer over itself (`keep(me)`, a `cap` the call grants into the parent), and `outer()` then calls
/// `inner()` through it. `inner()` waits on a word nobody sets; `peek()` answers whether it has
/// started, and runs only once it has parked, since the calls run one at a time. `join_child()`
/// joins the child; `kept()` answers whether `keep` has run.
fn self_caller() -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
data 16684 "exec"
{rec}type 0 func () -> (i64)
type 1 func (cap) -> (i64)
type 2 func () -> ()
type 3 interface {{ keep: 1, inner: 0, outer: 0, peek: 0, boom: 2, join_child: 0, kept: 0 }}
export 0 interface "svc" 3 {{ keep: 1, inner: 2, outer: 3, peek: 4, boom: 5, join_child: 6, kept: 7 }}

func (i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vm: i32, vb: i32) {{
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

func (cap) -> (i64) {{
block 0 (vme: cap) {{
  va = i64.const 16500
  i32.store va vme
  vk = i64.const 16504
  vone = i32.const 1
  i32.store vk vone
  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vs = i64.const 16600
  vone = i32.const 1
  i32.store vs vone
  vc = i64.const 16424
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vc vexp vto
  vr = i64.const 5
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  va = i64.const 16500
  vme = i32.load va
  vr = call.cap 268435456 1 () -> (i64) vme ()
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vs = i64.const 16600
  vv = i32.load vs
  vr = i64.extend_i32_s vv
  return vr
  }}
}}

func () -> () {{
block 0 () {{
  unreachable
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
  vk = i64.const 16504
  vv = i32.load vk
  vr = i64.extend_i32_s vv
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

/// The child [`self_caller`] spawns: it resolves `"exec"` and hands it to its parent, `keep(exec)`.
const GIVER: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vr = call.cap 268435456 0 (cap) -> (i64) vexec (vexec)
  return vr
  }
}
"#;

/// #3000 — an activation's own call into its domain that parks, when the domain then ends: the
/// embedder's `outer()` calls `inner()` through the offer over the domain itself, `inner()` parks
/// for good, and the domain is finished. Both calls answer the end, and the run ends.
fn a_nested_self_call_parked_at_the_end(on: Executor) {
    bounded(move || {
        let m = module(&self_caller());
        let (host, args) = child_host(&m, GIVER);
        let (d, _) = hold(on, m, 0, &args, host);
        let d = Arc::new(d);
        assert_eq!(call(&d, 5, &[]), Reply::Results(vec![1]), "the child gave");
        let outer = std::thread::spawn({
            let d = Arc::clone(&d);
            move || call(&d, 2, &[])
        });
        let mut seen = Reply::Results(vec![0]);
        for _ in 0..100_000 {
            seen = call(&d, 3, &[]);
            if seen == Reply::Results(vec![1]) {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(seen, Reply::Results(vec![1]), "inner() started and parked");
        assert_eq!(
            call(&d, 4, &[]),
            Reply::Failed(CAP_REVOKED),
            "boom() ends it"
        );
        let outer = outer.join().expect("the outer call");
        assert_eq!(
            outer,
            Reply::Failed(CAP_REVOKED),
            "the outer call answers the end"
        );
    });
}

on_every_executor!(a_nested_self_call_parked_at_the_end);

/// The child [`self_caller`] spawns for the guest's own end: it hands its parent `"exec"`, waits
/// until `inner()` has parked (`peek()` answers 1), and ends the domain itself with `boom()`.
const GIVER_THEN_BOOM: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vr = call.cap 268435456 0 (cap) -> (i64) vexec (vexec)
  br 1(vexec)
}
block 1 (we: i32) {
  vp = call.cap 268435456 3 () -> (i64) we ()
  vone = i64.const 1
  vgo = i64.ne vp vone
  br_if vgo 1(we) 2(we)
}
block 2 (xe: i32) {
  call.cap 268435456 4 () -> () xe ()
  vz = i64.const 0
  return vz
  }
}
"#;

/// #3000 — the same, reached by the guest alone: the embedder makes one call, `outer()`, and the
/// domain's own child ends the domain once `inner()` has parked. The call answers the end.
fn a_nested_self_call_parked_when_the_guest_ends_its_domain(on: Executor) {
    bounded(move || {
        let m = module(&self_caller());
        let (host, args) = child_host(&m, GIVER_THEN_BOOM);
        let (d, _) = hold(on, m, 0, &args, host);
        // The child hands the offer over first: else `outer()` would call through handle 0.
        let mut kept = Reply::Results(vec![0]);
        for _ in 0..1_000_000 {
            kept = call(&d, 6, &[]);
            if kept != Reply::Results(vec![0]) {
                break;
            }
            std::thread::yield_now();
        }
        assert_eq!(
            kept,
            Reply::Results(vec![1]),
            "the child kept its offer in the parent"
        );
        assert_eq!(call(&d, 2, &[]), Reply::Failed(CAP_REVOKED));
        // Only the intended path ends the domain this way: the child's `boom()`, once `inner()` has
        // parked (an `outer()` that found no offer would trap `CapFault`).
        let (_, end) = d.end();
        assert_eq!(end, Some(Trap::Unreachable));
    });
}

on_every_executor!(a_nested_self_call_parked_when_the_guest_ends_its_domain);

/// #2993 — a held domain whose module names no `Instantiator`, thread or fiber op, so its run's
/// only nursery is the one its install room needs (#1726), and whose child comes from a §22 unit
/// it installs. Its entry takes an `Instantiator`, the child's `Module`, a budget and a `Jit`; it
/// compiles and installs four units, `spawn(inst, rec)`, `join(inst, child)`, `wait(word, _)` (a
/// futex wait on a word nobody sets) and `kill(inst, child)`, and spawns the child through the
/// first, granting it `"exec"`, the live offer over the domain. Its ops:
///
/// - `ping()` answers 1; `join()` joins the child and answers `status * 1000 + result` (a trap's wire
///   code as its status, #2066); `kill()` kills it and answers 0.
/// - `wait()` marks 16600 and waits for good, through the `wait` unit; `peek()` answers the mark.
/// - `boom()` traps.
fn unit_spawner() -> String {
    const SPAWN: &str = "memory 17\nfunc (i64, i64) -> (i64) {\nblock 0 (vi: i64, vr: i64) {\n  vi32 = i32.wrap_i64 vi\n  vch = call.cap 6 17 (i64) -> (i32) vi32 (vr)\n  vc64 = i64.extend_i32_u vch\n  return vc64\n  }\n}\n";
    const JOIN: &str = "memory 17\nfunc (i64, i64) -> (i64) {\nblock 0 (vi: i64, vc: i64) {\n  vi32 = i32.wrap_i64 vi\n  vc32 = i32.wrap_i64 vc\n  vst, vv = call.cap 6 1 (i32) -> (i64, i64) vi32 (vc32)\n  vk = i64.const 1000\n  vs = i64.mul vst vk\n  vr = i64.add vs vv\n  return vr\n  }\n}\n";
    const WAIT: &str = "memory 17\nfunc (i64, i64) -> (i64) {\nblock 0 (va: i64, vb: i64) {\n  vexp = i32.const 0\n  vto = i64.const -1\n  vst = i32.atomic.wait va vexp vto\n  vr = i64.extend_i32_s vst\n  return vr\n  }\n}\n";
    const KILL: &str = "memory 17\nfunc (i64, i64) -> (i64) {\nblock 0 (vi: i64, vc: i64) {\n  vi32 = i32.wrap_i64 vi\n  vc32 = i32.wrap_i64 vc\n  vr = call.cap 6 12 (i32) -> (i32) vi32 (vc32)\n  vr64 = i64.extend_i32_s vr\n  return vr64\n  }\n}\n";
    let units: Vec<Vec<u8>> = [SPAWN, JOIN, WAIT, KILL]
        .iter()
        .map(|u| temen_encode::encode_module(&module(u)))
        .collect();
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    // Each unit's bytes at 20480 + 1024 i, its installed slot at 16800 + 8 i.
    let data: String = (units.iter().enumerate())
        .map(|(i, u)| {
            let bytes: String = u.iter().map(|b| format!("\\x{b:02x}")).collect();
            format!("data {} \"{bytes}\"\n", 20480 + 1024 * i)
        })
        .collect();
    let install: String = (units.iter().enumerate())
        .map(|(i, u)| {
            format!(
                "  vu{i} = i64.const {at}\n  vn{i} = i64.const {len}\n  \
                 vk{i} = call.cap 11 0 (i64, i64) -> (i64) vjit (vu{i}, vn{i})\n  \
                 vc{i} = call.cap 11 3 (i64) -> (i64) vjit (vk{i})\n  \
                 vs{i} = i64.const {slot}\n  i64.store vs{i} vc{i}\n",
                at = 20480 + 1024 * i,
                len = u.len(),
                slot = 16800 + 8 * i,
            )
        })
        .collect();
    // Call unit `i` on `(a, b)`, two registers, into `r`; registers prefixed `p`.
    let unit = |i: usize, p: &str, a: &str, b: &str, r: &str| {
        format!(
            "  {p}sa = i64.const {slot}\n  {p}c = i64.load {p}sa\n  {p}cs = i32.wrap_i64 {p}c\n  \
             {r} = call.dyn (i64, i64) -> (i64) {p}cs ({a}, {b})\n",
            slot = 16800 + 8 * i,
        )
    };
    // The `Instantiator` and the child, as `i64`s, into `vi` and `vch`.
    let handles =
        "  vsi = i64.const 16444\n  vi32 = i32.load vsi\n  vi = i64.extend_i32_u vi32\n  \
                   vsc = i64.const 16440\n  vch32 = i32.load vsc\n  vch = i64.extend_i32_u vch32\n";
    format!(
        r#"memory 17
data 16684 "exec"
{data}{rec}type 0 func () -> (i64)
type 1 func () -> ()
type 2 interface {{ ping: 0, join: 0, wait: 0, kill: 0, peek: 0, boom: 1 }}
export 0 interface "svc" 2 {{ ping: 1, join: 2, wait: 3, kill: 4, peek: 5, boom: 6 }}

func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vm: i32, vb: i32, vjit: i32) {{
  vsi = i64.const 16444
  i32.store vsi vinst
  vma = i64.const {module_at}
  i32.store vma vm
  vba = i64.const {budget_at}
  i32.store vba vb
  vgh = i32.const {tag}
{grant}{install}  vi = i64.extend_i32_u vinst
  vrec = i64.const 17408
{spawn}  vch32 = i32.wrap_i64 vch
  vsc = i64.const 16440
  i32.store vsc vch32
  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
{handles}{join}  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vs = i64.const 16600
  vone = i32.const 1
  i32.store vs vone
  vw = i64.const 16424
  vz = i64.const 0
{wait}  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
{handles}{kill}  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vs = i64.const 16600
  vv = i32.load vs
  vr = i64.extend_i32_s vv
  return vr
  }}
}}

func () -> () {{
block 0 () {{
  unreachable
  }}
}}
"#,
        rec = rec::segment(17408, &child),
        grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        spawn = unit(0, "p", "vi", "vrec", "vch"),
        join = unit(1, "p", "vi", "vch", "vr"),
        wait = unit(2, "p", "vw", "vz", "vr"),
        kill = unit(3, "p", "vi", "vch", "vr"),
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        tag = GRANT_SERVE_LIVE_TAG as i32,
    )
}

/// A child for [`unit_spawner`] that calls `op` on its parent through `"exec"` and returns the
/// answer plus 40.
fn unit_child(op: u32) -> String {
    format!(
        r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vr = call.cap 268435456 {op} () -> (i64) vexec ()
  vk = i64.const 40
  vr2 = i64.add vr vk
  return vr2
  }}
}}
"#
    )
}

/// Hold [`unit_spawner`] on `on`, its child calling `op`.
fn hold_unit_spawner(on: Executor, op: u32) -> HeldDomain {
    let m = module(&unit_spawner());
    let mut host = Host::new();
    host.set_self_module(&m);
    let jit = temen_run::grant_jit(&mut host, &m, 0);
    host.set_jit_quota(4, 1 << 20);
    let args = [
        Value::I32(host.grant_instantiator()),
        Value::I32(host.grant_module(&module(&unit_child(op)))),
        Value::I32(host.grant_budget(-1, 1 << 20, -1)),
        Value::I32(jit),
    ];
    hold(on, m, 0, &args, host).0
}

/// Wait, within a bound, until the child's `wait()` has started: `peek()` answers 1.
fn until_the_child_waits(d: &HeldDomain) {
    let mut seen = Reply::Results(vec![0]);
    for _ in 0..1_000_000 {
        seen = call(d, 4, &[]);
        if seen != Reply::Results(vec![0]) {
            break;
        }
        std::thread::yield_now();
    }
    assert_eq!(seen, Reply::Results(vec![1]), "the child's wait() started");
}

/// #2993 — a child that an installed unit spawned calls its held parent and is answered. Before,
/// the install room's nursery was made before the run's domain, so the child had no `sched`, and
/// its call aborted a debug build.
fn a_child_an_installed_unit_spawned_calls_its_held_parent(on: Executor) {
    bounded(move || {
        let d = hold_unit_spawner(on, 0);
        assert_eq!(call(&d, 1, &[]), Reply::Results(vec![41]));
    });
}

/// #2993 — the domain ends while that child's call is parked in it: the call unwinds, and the
/// domain's end returns. Before, the child was a task of a second pool that the domain's end never
/// woke, so the rest waited for its activation for good.
fn a_unit_childs_call_parked_at_its_parents_end_unwinds(on: Executor) {
    bounded(move || {
        let d = hold_unit_spawner(on, 2);
        until_the_child_waits(&d);
        assert_eq!(
            call(&d, 5, &[]),
            Reply::Failed(CAP_REVOKED),
            "boom() ends it"
        );
        drop(d);
    });
}

/// #2993 — the child is killed while its call is parked in its held parent: the call's activation
/// runs on in the pool, and the parent's end waits for it before the run's window, code and root
/// context go.
fn a_killed_unit_childs_parked_call_ends_before_its_parent(on: Executor) {
    bounded(move || {
        let d = hold_unit_spawner(on, 2);
        until_the_child_waits(&d);
        assert_eq!(call(&d, 3, &[]), Reply::Results(vec![0]), "kill()");
        let killed = temen_ir::trap_code::THREAD_FAULT * 1000;
        assert_eq!(call(&d, 1, &[]), Reply::Results(vec![killed]), "join()");
        drop(d);
    });
}

/// #3030 — what the interpreters do today with the same program: an activation's call of a unit
/// the entry installed ends the domain with `IndirectCallType`, where the JIT answers. Pinned so the
/// divergence stays visible until #3030 settles it (INVARIANTS #9).
fn an_activations_call_of_an_installed_unit_traps_on_the_interpreters(on: Executor) {
    bounded(move || {
        let d = hold_unit_spawner(on, 0);
        assert_eq!(call(&d, 1, &[]), Reply::Failed(CAP_REVOKED));
        let (_, end) = d.end();
        assert_eq!(end, Some(Trap::IndirectCallType));
    });
}

mod unit_children_on_the_interpreters {
    #[test]
    fn tree_walker() {
        super::an_activations_call_of_an_installed_unit_traps_on_the_interpreters(
            temen_interp::Executor::TreeWalker,
        );
    }

    #[test]
    fn pump() {
        super::an_activations_call_of_an_installed_unit_traps_on_the_interpreters(
            temen_interp::Executor::Pump,
        );
    }

    #[test]
    fn threads() {
        super::an_activations_call_of_an_installed_unit_traps_on_the_interpreters(
            temen_interp::Executor::Threads,
        );
    }
}

/// On the JIT alone: the interpreters end the domain with `IndirectCallType` when an activation
/// calls a unit its entry installed (#3030, pinned above).
mod unit_children {
    #[test]
    fn a_child_an_installed_unit_spawned_calls_its_held_parent() {
        super::a_child_an_installed_unit_spawned_calls_its_held_parent(temen_interp::Executor::Jit);
    }

    #[test]
    fn a_unit_childs_call_parked_at_its_parents_end_unwinds() {
        super::a_unit_childs_call_parked_at_its_parents_end_unwinds(temen_interp::Executor::Jit);
    }

    #[test]
    fn a_killed_unit_childs_parked_call_ends_before_its_parent() {
        super::a_killed_unit_childs_parked_call_ends_before_its_parent(temen_interp::Executor::Jit);
    }
}

/// [`self_caller`] whose `outer()` calls `kept()` through the offer over the domain itself, and
/// whose `kept()` counts 100 down: a nested activation with work of its own.
fn nested_counter() -> String {
    let s = self_caller().replacen(
        "  vr = call.cap 268435456 1 () -> (i64) vme ()\n",
        "  vr = call.cap 268435456 6 () -> (i64) vme ()\n",
        1,
    );
    let kept = "func () -> (i64) {\nblock 0 () {\n  vk = i64.const 16504\n  vv = i32.load vk\n  vr = i64.extend_i32_s vv\n  return vr\n  }\n}\n";
    let counts = "func () -> (i64) {\nblock 0 () {\n  vn = i64.const 100\n  br 1(vn)\n}\nblock 1 (vi: i64) {\n  vone = i64.const 1\n  vj = i64.sub vi vone\n  vd = i64.eqz vj\n  br_if vd 2(vj) 1(vj)\n}\nblock 2 (vr: i64) {\n  vk = i64.const 42\n  return vk\n  }\n}\n";
    assert!(s.contains(kept), "the fixture's kept()");
    s.replacen(kept, counts, 1)
}

/// `outer()` of [`nested_counter`] on a loan of `loan`: its reply and what it burned.
fn nested_outer(on: Executor, loan: u64) -> (Reply, u64) {
    bounded(move || {
        let m = module(&nested_counter());
        let (host, args) = child_host(&m, GIVER);
        let (d, _) = hold(on, m, 0, &args, host);
        assert_eq!(call(&d, 5, &[]), Reply::Results(vec![1]), "the child gave");
        let mut fuel = loan;
        let reply = d.call(0, 2, &[], &mut fuel);
        (reply, loan - fuel)
    })
}

/// #3069 — a nested activation on a loan can burn all of it: the waiting caller hands back what its
/// draw left before the call, as it cannot run meanwhile. Each activation draws a cell of its own
/// from the loan, so a kept draw stranded half the loan, and the least loan was twice the cost.
/// The JIT alone: the interpreters charge the nested call nothing to the outer loan, so their least
/// loan here is not the JIT's (#3093). An activation's entry is free on every engine (#3138).
///
/// The read-back is ordered: `outer()` can call in, so it is carried (it may leave the caller's
/// thread), and its end reads its cell's `can_burn` (`HeldSite::carry`'s `done`). `kept()` never
/// parks, so it runs inline, on `outer()`'s own stack inside its `call.cap`: its context, and with it
/// its cell, drops as that `call.cap` returns, handing back what it drew before `outer()` runs on,
/// let alone ends.
#[test]
fn a_nested_activation_on_the_jit_can_burn_its_whole_loan() {
    let (done, cost) = nested_outer(Executor::Jit, 1_000_000);
    assert_eq!(done, Reply::Results(vec![42]));
    assert_eq!(
        cost, 99,
        "the inner loop's 99 back-edges: an activation's entry is free (#3138)"
    );
    assert_eq!(
        nested_outer(Executor::Jit, cost),
        (Reply::Results(vec![42]), cost)
    );
    assert_eq!(
        nested_outer(Executor::Jit, cost - 1),
        (Reply::Failed(CAP_REVOKED), cost - 1)
    );
}

/// #3080 (review F8a) — the unit a [`A_UNIT_FIBER_CROSSES_CALLS`] domain installs. Its entry, given
/// mode 0, makes a fiber over its own function 1, resumes it to its first suspend and answers its
/// handle; given mode 1 it resumes the fiber `h` and answers what it returned. The fiber's body calls
/// its parent's function 3 (slot 3, new to old) with the unit's suspender (function 2), so the
/// parent's code is on the fiber's stack when it suspends.
const A_UNIT_THAT_RUNS_A_PARENT_FIBER: &str = "memory 16
func (i64, i64) -> (i64) {
block 0 (vmode: i64, vh: i64) {
  vz = i64.const 0
  vresume = i64.ne vmode vz
  br_if vresume 2(vh) 1()
}
block 1 () {
  vf = ref.func 1
  vsp = i64.const 32768
  vk = cont.new vf vsp
  vz1 = i64.const 0
  vs, vv = cont.resume vk vz1
  return vk
}
block 2 (vk2: i64) {
  vz2 = i64.const 0
  vs2, vv2 = cont.resume vk2 vz2
  return vv2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vsus = ref.func 2
  vparent = i32.const 3
  vr = call.dyn (i32) -> (i64) vparent (vsus)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vy = suspend vx
  return vy
  }
}
";

/// #3080 (review F8a) — a held domain that drives `Jit` and has no fiber op of its own. Its entry
/// compiles [`A_UNIT_THAT_RUNS_A_PARENT_FIBER`] (laid in its window at 20480, `LEN` bytes), installs
/// it and answers the slot. `mk()` dispatches the unit to make a fiber; the fiber's body calls
/// function 3, which runs 1,000 back-edges and calls the unit's suspender, so it suspends with
/// function 3's frame on its stack, a frame that read `mk()`'s cell. `go(h)` dispatches the unit
/// again to resume it, on another call's cell, and function 3 runs 1,000 more and returns 7. The
/// parent has no fiber op, so only the grant (`hosts_fibers`) lowers function 3 to reload its cell.
const A_UNIT_FIBER_CROSSES_CALLS: &str = r#"memory 16
type 0 func () -> (i64)
type 1 func (i64) -> (i64)
type 2 interface { mk: 0, go: 1 }
export 0 interface "svc" 2 { mk: 1, go: 2 }
func (i32) -> (i64) {
block 0 (vjit: i32) {
  vblob = i64.const 20480
  vlen = i64.const LEN
  vu = call.cap 11 0 (i64, i64) -> (i64) vjit (vblob, vlen)
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vu)
  vat = i64.const 16384
  i64.store vat vslot
  return vslot
  }
}
func () -> (i64) {
block 0 () {
  vat = i64.const 16384
  vslot = i64.load vat
  vs = i32.wrap_i64 vslot
  vz = i64.const 0
  vh = call.dyn (i64, i64) -> (i64) vs (vz, vz)
  return vh
  }
}
func (i64) -> (i64) {
block 0 (vh: i64) {
  vat = i64.const 16384
  vslot = i64.load vat
  vs = i32.wrap_i64 vslot
  vone = i64.const 1
  vr = call.dyn (i64, i64) -> (i64) vs (vone, vh)
  return vr
  }
}
func (i32) -> (i64) {
block 0 (vsus: i32) {
  vz = i64.const 0
  br 1(vz, vsus)
}
block 1 (vi: i64, vs1: i32) {
  vone = i64.const 1
  vj = i64.add vi vone
  vn = i64.const 1000
  vmore = i64.lt_u vj vn
  br_if vmore 1(vj, vs1) 2(vs1)
}
block 2 (vs2: i32) {
  vz1 = i64.const 0
  vx = call.dyn (i64) -> (i64) vs2 (vz1)
  vz2 = i64.const 0
  br 3(vz2)
}
block 3 (vi2: i64) {
  vone2 = i64.const 1
  vj2 = i64.add vi2 vone2
  vn2 = i64.const 1000
  vmore2 = i64.lt_u vj2 vn2
  br_if vmore2 3(vj2) 4()
}
block 4 () {
  v7 = i64.const 7
  return v7
  }
}
"#;

/// #3080 (review B2) — a held domain that drives `Jit` under a fiber-hosting grant, whose installed
/// unit's fiber runs the domain's own code across two calls, is held by the tree-walker, whichever
/// executor is asked for: a unit's fiber could call into the domain (#3164), and its code is not
/// there to read before it arrives. Both runs are the tree-walker's, so the second pins only the
/// decline. That the JIT lowers such a domain's code to reload its cell (`hosts_fibers`) is pinned on the
/// route that runs it, `jit_fuel`'s `jit_fuel_an_installed_units_fiber_charges_the_vcpu_that_resumes_it`.
#[test]
fn a_domain_whose_units_may_host_fibers_and_that_drives_jit_is_held_by_the_tree_walker() {
    let unit = temen_text::parse_module(A_UNIT_THAT_RUNS_A_PARENT_FIBER).expect("parse the unit");
    temen_verify::verify_module(&unit).expect("verify the unit");
    let blob = temen_encode::encode_module(&unit);
    let escaped: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    let src = format!(
        "{}data 20480 \"{escaped}\"\n",
        A_UNIT_FIBER_CROSSES_CALLS.replace("LEN", &blob.len().to_string())
    );
    let on = |on: Executor| {
        let src = src.clone();
        bounded(move || {
            let m = module(&src);
            let mut host = Host::new();
            host.set_self_module(&m);
            let jit = temen_run::grant_jit_fibers(&mut host, &m, 4);
            let started = temen_run::hold(on, m, 0, &[Value::I32(jit)], u64::MAX, host);
            let (d, _) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
            assert_eq!(d.executor(), Executor::TreeWalker, "asked for {on:?}");
            let mut fuel = 1_000_000;
            let made = (d.call(0, 0, &[], &mut fuel), fuel);
            let handle = match &made.0 {
                Reply::Results(r) => r[0],
                other => panic!("mk(): {other:?}"),
            };
            let mut fuel = 1_000_000;
            let resumed = (d.call(0, 1, &[handle], &mut fuel), fuel);
            (made, resumed)
        })
    };
    let oracle = on(Executor::TreeWalker);
    assert_eq!(oracle.1 .0, Reply::Results(vec![7]), "the fiber returned 7");
    assert_eq!(on(Executor::Jit), oracle);
}

/// #3080 (review B2) — [`nested_counter`] with `outer()` making a fiber F over a new function 8,
/// which calls `kept()` through the domain's own offer; `outer()` resumes F twice and answers
/// `1e6·status₁ + 1e3·status₂ + value₂`.
fn fiber_self_caller() -> String {
    let s = nested_counter();
    let outer = "  va = i64.const 16500\n  vme = i32.load va\n  vr = call.cap 268435456 6 () -> (i64) vme ()\n  return vr\n";
    assert!(s.contains(outer), "outer()");
    let fiber_outer = "  vf = i32.const 8\n  vsp = i64.const 40960\n  vk = cont.new vf vsp\n  vz = i64.const 0\n  vs1, vv1 = cont.resume vk vz\n  vs2, vv2 = cont.resume vk vz\n  vs1w = i64.extend_i32_u vs1\n  vs2w = i64.extend_i32_u vs2\n  vm1 = i64.const 1000000\n  vm2 = i64.const 1000\n  va1 = i64.mul vs1w vm1\n  va2 = i64.mul vs2w vm2\n  vb = i64.add va1 va2\n  vr = i64.add vb vv2\n  return vr\n";
    let s = s.replacen(outer, fiber_outer, 1);
    format!("{s}\nfunc (i64, i64) -> (i64) {{\nblock 0 (vsp: i64, varg: i64) {{\n  va = i64.const 16500\n  vme = i32.load va\n  vr = call.cap 268435456 6 () -> (i64) vme ()\n  return vr\n  }}\n}}\n")
}

/// #3080 (review B2), #3164 — a guest fiber's call into its own held domain. The tree-walker keeps
/// the nested activation out while the calling one runs: F is still parked at `outer()`'s second
/// resume, `3003000`, and the call is charged 2. On the JIT the nested activation ran under the
/// caller's gate between the two resumes and then gave the gate away (`3001042`, 101), so `hold`
/// declines a fiber-making module that makes a guest-interface call, and the tree-walker holds it,
/// until #3164 carries such a call on the pool.
#[test]
fn a_fibers_call_into_its_own_held_domain_is_held_by_the_tree_walker() {
    for on in [Executor::TreeWalker, Executor::Jit] {
        let (reply, charged, executor) = bounded(move || {
            let m = module(&fiber_self_caller());
            let (host, args) = child_host(&m, GIVER);
            let started = temen_run::hold(on, m, 0, &args, u64::MAX, host);
            let (d, _) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
            assert_eq!(call(&d, 5, &[]), Reply::Results(vec![1]), "the child gave");
            let mut fuel = 1_000_000u64;
            let reply = d.call(0, 2, &[], &mut fuel);
            (reply, 1_000_000 - fuel, d.executor())
        });
        assert_eq!(executor, Executor::TreeWalker, "asked for {on:?}");
        assert_eq!(
            (reply, charged),
            (Reply::Results(vec![3_003_000]), 2),
            "asked for {on:?}"
        );
    }
}

/// #3158 — the back-edges of [`qh_held`]'s `spin()`: many quanta of a pool task's fuel.
const QH_HANDLER_TURNS: u64 = 200_000_000;
/// #3158 — the back-edges of [`qh_spinner`]: it outlasts the handler, so a task stays queued.
const QH_SPIN_TURNS: u64 = 2_000_000_000;

/// #3158 — a held domain whose entry spawns three §14 children, which run as pool tasks: two spinners, and
/// a caller holding the domain's own offer as `"exec"`. Its `spin()` never parks and runs
/// [`QH_HANDLER_TURNS`] back-edges; `join_c()` joins the caller and answers `1000·status + value`.
fn qh_held() -> String {
    let c = SpawnRec {
        grants_ptr: 16640,
        grants_n: 1,
        ..SpawnRec::default()
    };
    let d = SpawnRec::default();
    format!(
        r#"memory 17
data 16684 "exec"
{rec_c}{rec_d1}{rec_d2}type 0 func () -> (i64)
type 1 interface {{ spin: 0, join_c: 0 }}
export 0 interface "svc" 1 {{ spin: 1, join_c: 2 }}

func (i32, i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vmc: i32, vmd: i32, vb: i32) {{
  vsi = i64.const 16444
  i32.store vsi vi
  vd1m = i64.const {d1_module}
  i32.store vd1m vmd
  vd1b = i64.const {d1_budget}
  i32.store vd1b vb
  vd1r = i64.const 17536
  vd1 = call.cap 6 17 (i64) -> (i32) vi (vd1r)
  vd2m = i64.const {d2_module}
  i32.store vd2m vmd
  vd2b = i64.const {d2_budget}
  i32.store vd2b vb
  vd2r = i64.const 17664
  vd2 = call.cap 6 17 (i64) -> (i32) vi (vd2r)
  vcm = i64.const {c_module}
  i32.store vcm vmc
  vcb = i64.const {c_budget}
  i32.store vcb vb
  vgh = i32.const {tag}
{grant}  vcr = i64.const 17408
  vc = call.cap 6 17 (i64) -> (i32) vi (vcr)
  vsc = i64.const 16440
  i32.store vsc vc
  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vone = i64.const 1
  vj = i64.add vi vone
  vn = i64.const {QH_HANDLER_TURNS}
  vmore = i64.lt_u vj vn
  br_if vmore 1(vj) 2()
}}
block 2 () {{
  v42 = i64.const 42
  return v42
  }}
}}

func () -> (i64) {{
block 0 () {{
  vsi = i64.const 16444
  vi = i32.load vsi
  vsc = i64.const 16440
  vc = i32.load vsc
  vst, vv = call.cap 6 1 (i32) -> (i64, i64) vi (vc)
  vk = i64.const 1000
  vsk = i64.mul vst vk
  vr = i64.add vsk vv
  return vr
  }}
}}
"#,
        rec_c = rec::segment(17408, &c),
        rec_d1 = rec::segment(17536, &d),
        rec_d2 = rec::segment(17664, &d),
        grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        c_module = 17408 + SpawnRec::MODULE_AT,
        c_budget = 17408 + SpawnRec::BUDGET_AT,
        d1_module = 17536 + SpawnRec::MODULE_AT,
        d1_budget = 17536 + SpawnRec::BUDGET_AT,
        d2_module = 17664 + SpawnRec::MODULE_AT,
        d2_budget = 17664 + SpawnRec::BUDGET_AT,
        tag = GRANT_SERVE_LIVE_TAG as i32,
    )
}

/// The caller child: answers what `spin()` through `"exec"` answers.
const QH_CALLER: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vr = call.cap 268435456 0 () -> (i64) vexec ()
  return vr
  }
}
"#;

/// A child that spins [`QH_SPIN_TURNS`] back-edges and never parks.
fn qh_spinner() -> String {
    format!(
        "memory 15\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vz = i64.const 0\n  br 1(vz)\n}}\nblock 1 (vi: i64) {{\n  vone = i64.const 1\n  vj = i64.add vi vone\n  vn = i64.const {QH_SPIN_TURNS}\n  vmore = i64.lt_u vj vn\n  br_if vmore 1(vj) 2()\n}}\nblock 2 () {{\n  vz2 = i64.const 0\n  return vz2\n  }}\n}}\n"
    )
}

/// #3158 — a quantum that ends inside a handler run inline on a pool task's stack does not park the
/// task. The caller child's call into its held parent runs the handler on the caller's own stack
/// (`held_call` → `run_inline_as`, a guarded call nested in the task's frames); with one worker and
/// the spinners queued, the handler's quanta end there. Before the yield floor each such end parked
/// the task with the handler's recovery scope open on its stack, and the task resumed on another
/// worker could leave that scope there and fault into the first worker's frames (traced on main: 3
/// in 12 rounds). Now the task keeps its worker to the handler's end, and each park it would have
/// made is counted (`temen_jit::quantum_ends_held`); without the floor nothing counts them, so this
/// fails every run.
///
/// Under ASan too (`asan-held`): the stack checks read the stack pointer's register, not a local's
/// address, which ASan's fake stack puts on neither the thread's stack nor a fiber's.
#[test]
fn a_quantum_ending_inside_a_handler_run_inline_on_a_pool_task_does_not_park_it() {
    let held = temen_jit::quantum_ends_held();
    let reply = bounded(move || {
        let m = module(&qh_held());
        let mut host = Host::new();
        host.set_self_module(&m);
        host.set_workers(1);
        let args = vec![
            Value::I32(host.grant_instantiator()),
            Value::I32(host.grant_module(&module(QH_CALLER))),
            Value::I32(host.grant_module(&module(&qh_spinner()))),
            Value::I32(host.grant_budget(-1, 1 << 20, -1)),
        ];
        let (d, _) = hold(Executor::Jit, m, 0, &args, host);
        call(&d, 1, &[])
    });
    assert_eq!(
        reply,
        Reply::Results(vec![42]),
        "the caller joined, answering 42"
    );
    assert!(
        temen_jit::quantum_ends_held() > held,
        "the handler's quanta ended under its inline scope with the pool full, and the task kept its worker"
    );
}

/// What [`fiber_offer_caller`]'s fiber does with the offer over its own domain the child hands back.
#[derive(Clone, Copy)]
enum FiberCallsIn {
    /// G1: it calls the domain's `strm` export, which has the pre-seeded `Stream` shape and so the
    /// built-in id 0, below `GUEST_IMPL_BASE`.
    BuiltinsId,
    /// G2: it `Jit.invoke`s [`A_UNIT_THAT_CALLS_KEPT`], whose top level calls the domain's `kept()`;
    /// the domain holds `Jit` with no fiber-hosting grant.
    InvokedUnit,
    /// H1: the domain makes no fiber. Its function 6 is a `gc.roots` op it never calls, which stands
    /// the JIT's fiber runtime up, so `define_extra` admits a unit's `cont.*` with no fiber-hosting
    /// grant. Its entry installs [`A_UNIT_WITH_A_FIBER_THAT_CALLS_KEPT`], and `outer()` reaches it by
    /// `call.dyn`: the unit's own fiber calls the domain's `kept()`.
    InstalledUnitsFiber,
}

/// #3080 (review H1) — an installed unit whose entry makes a fiber over its own function 1, resumes
/// it twice and answers `1e6·status₁ + 1e3·status₂ + value₂`; the fiber calls `kept()` (op 3) on the
/// offer the domain stored at 16500.
const A_UNIT_WITH_A_FIBER_THAT_CALLS_KEPT: &str = "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  vf = ref.func 1\n  vsp = i64.const 49152\n  vk = cont.new vf vsp\n  vz = i64.const 0\n  vs1, vv1 = cont.resume vk vz\n  vs2, vv2 = cont.resume vk vz\n  vs1w = i64.extend_i32_u vs1\n  vs2w = i64.extend_i32_u vs2\n  vm1 = i64.const 1000000\n  vm2 = i64.const 1000\n  va1 = i64.mul vs1w vm1\n  va2 = i64.mul vs2w vm2\n  vsum = i64.add va1 va2\n  vr = i64.add vsum vv2\n  return vr\n  }\n}\nfunc (i64, i64) -> (i64) {\nblock 0 (vsp: i64, varg: i64) {\n  va = i64.const 16500\n  vme = i32.load va\n  vr = call.cap 268435456 3 () -> (i64) vme ()\n  return vr\n  }\n}\n";

/// A unit's text, verified and encoded: the `data` segment laying it at 20480, and its length.
fn unit_blob(src: &str) -> (String, usize) {
    let unit = temen_text::parse_module(src).expect("parse the unit");
    temen_verify::verify_module(&unit).expect("verify the unit");
    let blob = temen_encode::encode_module(&unit);
    let escaped: String = blob.iter().map(|b| format!("\\x{b:02x}")).collect();
    (format!("data 20480 \"{escaped}\"\n"), blob.len())
}

/// #3080 (review G2) — a unit whose top level calls `kept()` (op 3) on the offer the domain stored at
/// 16500. A unit shares the domain's window and dispatches into its powerbox.
const A_UNIT_THAT_CALLS_KEPT: &str = "memory 17\nfunc () -> (i64) {\nblock 0 () {\n  va = i64.const 16500\n  vme = i32.load va\n  vr = call.cap 268435456 3 () -> (i64) vme ()\n  return vr\n  }\n}\n";

/// #3080 (review G1, G2) — a held domain with two exports, `svc { keep, outer, join_child, kept }`
/// and `strm`, `Stream`'s canonical shape (`read`, `write`, `close`). Its entry spawns
/// [`offer_giver`] with a live grant of each (`"exec"`, `"strm"`); the child hands one back through
/// `keep()`. `outer()` makes a fiber F over function 6, resumes it twice and answers
/// `1e6·status₁ + 1e3·status₂ + value₂`, as [`fiber_self_caller`]. `kept()` and `strm`'s `read` and
/// `write` count 100 down and answer 42. F calls into the domain as `how` says. Its entry takes the
/// `Instantiator`, the child's module, its budget and a `Jit` handle (unused by G1's).
fn fiber_offer_caller(how: FiberCallsIn) -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::default()
    };
    let fiber_outer = "  vf = i32.const 6\n  vsp = i64.const 40960\n  vk = cont.new vf vsp\n  vz = i64.const 0\n  vs1, vv1 = cont.resume vk vz\n  vs2, vv2 = cont.resume vk vz\n  vs1w = i64.extend_i32_u vs1\n  vs2w = i64.extend_i32_u vs2\n  vm1 = i64.const 1000000\n  vm2 = i64.const 1000\n  va1 = i64.mul vs1w vm1\n  va2 = i64.mul vs2w vm2\n  vsum = i64.add va1 va2\n  vr = i64.add vsum vv2\n  return vr\n";
    let (body, blob, install, outer) = match how {
        FiberCallsIn::BuiltinsId => (
            "  va = i64.const 16500\n  vme = i32.load va\n  vz = i64.const 0\n  vr = call.cap 0 1 (i64, i64) -> (i64) vme (vz, vz)\n  return vr\n".to_string(),
            String::new(),
            String::new(),
            fiber_outer.to_string(),
        ),
        FiberCallsIn::InvokedUnit => {
            let (blob, len) = unit_blob(A_UNIT_THAT_CALLS_KEPT);
            (
                format!("  vj = i64.const 16520\n  vjit = i32.load vj\n  vb = i64.const 20480\n  vl = i64.const {len}\n  vu = call.cap 11 0 (i64, i64) -> (i64) vjit (vb, vl)\n  vr = call.cap 11 1 (i64) -> (i64) vjit (vu)\n  return vr\n"),
                blob,
                String::new(),
                fiber_outer.to_string(),
            )
        }
        FiberCallsIn::InstalledUnitsFiber => {
            let (blob, len) = unit_blob(A_UNIT_WITH_A_FIBER_THAT_CALLS_KEPT);
            (
                "  v0 = i64.const 0\n  v1 = i64.const 4096\n  vmask = i64.const -1\n  v2 = i64.const 0\n  v3 = i64.const 8\n  vr = gc.roots v0 v1 vmask v2 v3\n  return vr\n".to_string(),
                blob,
                format!("  vub = i64.const 20480\n  vul = i64.const {len}\n  vu = call.cap 11 0 (i64, i64) -> (i64) vjit (vub, vul)\n  vslot = call.cap 11 3 (i64) -> (i64) vjit (vu)\n  vsa = i64.const 16528\n  i64.store vsa vslot\n"),
                "  va = i64.const 16528\n  vslot = i64.load va\n  vs = i32.wrap_i64 vslot\n  vr = call.dyn () -> (i64) vs ()\n  return vr\n".to_string(),
            )
        }
    };
    let counts = "block 0 () {\n  vn = i64.const 100\n  br 1(vn)\n}\nblock 1 (vi: i64) {\n  vone = i64.const 1\n  vj = i64.sub vi vone\n  vd = i64.eqz vj\n  br_if vd 2(vj) 1(vj)\n}\nblock 2 (vr: i64) {\n  vk = i64.const 42\n  return vk\n  }\n}\n";
    let counts_rw = counts.replacen("block 0 () {", "block 0 (va: i64, vb: i64) {", 1);
    format!(
        r#"memory 17
data 16684 "exec"
data 16690 "strm"
{rec}{blob}type 0 func () -> (i64)
type 1 func (cap) -> (i64)
type 2 func () -> ()
type 3 func (i64, i64) -> (i64)
type 4 interface {{ keep: 1, outer: 0, join_child: 0, kept: 0 }}
type 5 interface {{ read: 3, write: 3, close: 2 }}
export 0 interface "svc" 4 {{ keep: 1, outer: 2, join_child: 3, kept: 7 }}
export 1 interface "strm" 5 {{ read: 4, write: 4, close: 5 }}

func (i32, i32, i32, i32) -> (i64) {{
block 0 (vi: i32, vm: i32, vb: i32, vjit: i32) {{
  vsi = i64.const 16444
  i32.store vsi vi
  vsj = i64.const 16520
  i32.store vsj vjit
{install}  vma = i64.const {module_at}
  i32.store vma vm
  vba = i64.const {budget_at}
  i32.store vba vb
  vgh = i32.const {tag}
{grant}  vgs = i32.const {tag_strm}
{grant_strm}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vsc = i64.const 16440
  i32.store vsc vch
  vr = i64.const 1
  return vr
  }}
}}

func (cap) -> (i64) {{
block 0 (vme: cap) {{
  va = i64.const 16500
  i32.store va vme
  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
{outer}  }}
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

func (i64, i64) -> (i64) {{
{counts_rw}
func () -> () {{
block 0 () {{
  return
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
{body}  }}
}}

func () -> (i64) {{
{counts}"#,
        rec = rec::segment(17408, &child),
        grant = rec::grant("ge", 16640, 16684, 4, "vgh"),
        grant_strm = rec::grant("gs", 16656, 16690, 4, "vgs"),
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        tag = GRANT_SERVE_LIVE_TAG as i32,
        tag_strm = (GRANT_SERVE_LIVE_TAG | 1) as i32,
    )
}

/// The child [`fiber_offer_caller`] spawns: it resolves the live grant `name` and hands it to its
/// parent, `keep(name)`, through the grant `"exec"`.
fn offer_giver(name: &str) -> String {
    format!(
        r#"memory 15
data 16384 "exec"
data 16392 "{name}"
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  vgp = i64.const 16392
  vgl = i64.const {len}
  vg = self.resolve vgp vgl
  vr = call.cap 268435456 0 (cap) -> (i64) vexec (vg)
  return vr
  }}
}}
"#,
        len = name.len()
    )
}

/// `outer()` of [`fiber_offer_caller`] asked for on `on`: the executor that held it, its reply and
/// what the call was charged.
fn fiber_offer_outer(on: Executor, how: FiberCallsIn) -> (Executor, Reply, u64) {
    bounded(move || {
        let m = module(&fiber_offer_caller(how));
        let give = match how {
            FiberCallsIn::BuiltinsId => "strm",
            FiberCallsIn::InvokedUnit | FiberCallsIn::InstalledUnitsFiber => "exec",
        };
        let (mut host, mut args) = child_host(&m, &offer_giver(give));
        args.push(Value::I32(match how {
            FiberCallsIn::BuiltinsId => -1,
            FiberCallsIn::InvokedUnit | FiberCallsIn::InstalledUnitsFiber => {
                temen_run::grant_jit(&mut host, &m, 4)
            }
        }));
        let started = temen_run::hold(on, m, 0, &args, u64::MAX, host);
        let (d, _) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        assert_eq!(call(&d, 2, &[]), Reply::Results(vec![1]), "the child gave");
        let mut fuel = 1_000_000u64;
        let reply = d.call(0, 1, &[], &mut fuel);
        (d.executor(), reply, 1_000_000 - fuel)
    })
}

/// #3080 (review G1), #3164 — a guest fiber's call into its own held domain through an offer whose
/// interface is a pre-seeded built-in shape (`Stream`, id 0, below `GUEST_IMPL_BASE`). The tree-walker
/// keeps the nested activation out while `outer()` runs, so F is still parked at its second resume:
/// `3003000`, charged 2. Held on the JIT (before `hold` read the built-in's id as one a guest can
/// implement), the nested activation ran under the caller's gate between the two resumes: `3001042`,
/// charged 101. So `hold` declines it, and the tree-walker holds it on both.
#[test]
fn a_fibers_call_into_its_own_held_domain_through_a_builtins_id_is_held_by_the_tree_walker() {
    for on in [Executor::TreeWalker, Executor::Jit] {
        assert_eq!(
            fiber_offer_outer(on, FiberCallsIn::BuiltinsId),
            (Executor::TreeWalker, Reply::Results(vec![3_003_000]), 2),
            "asked for {on:?}"
        );
    }
}

/// #3080 (review G2), #3164 — a fiber of a held domain that holds `Jit` without the fiber-hosting
/// grant `Jit.invoke`s a unit whose top level calls the domain's own `kept()`. No static read of the
/// module sees that call. The tree-walker's invoke child refuses it, which ends the domain
/// (`CAP_REVOKED`, charged 1): #3154's refusal, which #3155 removes. Held on the JIT (before `hold`
/// counted a unit as code that can run in a fiber), the nested activation ran at once, since below
/// the invoke's yield floor nothing parks (#3162), so F returned 42 at its first resume and the
/// second answered -9: `4294968287000`, charged 102. So `hold` declines a fiber-making module that can
/// reach `Jit`, and the tree-walker holds it on both.
#[test]
fn a_fibers_unit_that_calls_into_its_own_held_domain_is_held_by_the_tree_walker() {
    for on in [Executor::TreeWalker, Executor::Jit] {
        assert_eq!(
            fiber_offer_outer(on, FiberCallsIn::InvokedUnit),
            (Executor::TreeWalker, Reply::Failed(CAP_REVOKED), 1),
            "asked for {on:?}"
        );
    }
}

/// #3080 (review H1), #3164 — a held domain that makes no fiber but has a `gc.roots` op, under a
/// plain `Jit` grant: its compile stands the JIT's fiber runtime up, so an installed unit's own fiber
/// runs, and its call into the domain's own offer is a fiber's call into its held domain. The
/// tree-walker keeps the nested activation out while `outer()` runs, so the unit's fiber is still
/// parked at its second resume: `3003000`, charged 3. Held on the JIT (before `hold` read a unit's
/// fibers from the compile's own facts), the nested activation ran under the caller's gate between
/// the two resumes: `3001042`, charged 102. So `hold` declines it, and the tree-walker holds it on
/// both. The unit is installed and reached by `call.dyn`, so the tree-walker's answer is the gate's
/// (#3164's), not its invoke child's refusal (#3154).
#[test]
fn a_units_own_fiber_that_calls_into_its_held_domain_is_held_by_the_tree_walker() {
    for on in [Executor::TreeWalker, Executor::Jit] {
        assert_eq!(
            fiber_offer_outer(on, FiberCallsIn::InstalledUnitsFiber),
            (Executor::TreeWalker, Reply::Results(vec![3_003_000]), 3),
            "asked for {on:?}"
        );
    }
}

/// How [`import_caller`]'s fiber calls `kept()`, and what the import it calls through is bound to.
#[derive(Clone, Copy, Debug)]
enum ImportCall {
    /// `call.import 0`, bound (not rebindable) to an offer the host wires over the domain's own
    /// `kept()`: a guest-interned id. A host cannot hold an offer over a domain before it runs, so
    /// the wired offer stands in for the domain's own.
    BoundToAGuestOffer,
    /// `call.import 0`, bound (not rebindable) to a host procedure that answers 7.
    BoundToAHostCap,
    /// `call.import 0` through a `rebindable` slot, which the entry attaches to the domain's own
    /// offer (`export.handle 0`): what it reaches is not fixed when `hold` reads it.
    Rebindable,
    /// `call.import.dyn` on the domain's own offer: its interface is interned as it runs.
    Dynamic,
}

/// #3080 (review H2) — a held domain whose `outer()` makes a fiber that calls through its import once,
/// as `how` says, and answers `1e6·status₁ + 1e3·status₂ + value₁` over two resumes (the fiber returns
/// what the call answered at its first, and the second resume of the ended fiber answers -9); `kept()` counts 100 down
/// and answers 42. Its entry attaches a [`ImportCall::Rebindable`] domain's slot.
fn import_caller(how: ImportCall) -> String {
    let rebindable = matches!(how, ImportCall::Rebindable);
    let attach = if rebindable {
        "  vme = export.handle 0\n  vst = import.attach 0 vme\n"
    } else {
        ""
    };
    let call = if matches!(how, ImportCall::Dynamic) {
        "  vme = export.handle 0\n  vr = call.import.dyn 1.kept vme ()\n"
    } else {
        "  vr = call.import 0 ()\n"
    };
    format!(
        r#"memory 17
type 0 func () -> (i64)
type 1 interface {{ outer: 0, kept: 0 }}
import 0 "k" () -> (i64){mode}
export 0 interface "svc" 1 {{ outer: 1, kept: 2 }}

func () -> (i64) {{
block 0 () {{
{attach}  vr = i64.const 1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vf = i32.const 3
  vsp = i64.const 40960
  vk = cont.new vf vsp
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  vs2, vv2 = cont.resume vk vz
  vs1w = i64.extend_i32_u vs1
  vs2w = i64.extend_i32_u vs2
  vm1 = i64.const 1000000
  vm2 = i64.const 1000
  va1 = i64.mul vs1w vm1
  va2 = i64.mul vs2w vm2
  vsum = i64.add va1 va2
  vr = i64.add vsum vv1
  return vr
  }}
}}

func () -> (i64) {{
block 0 () {{
  vn = i64.const 100
  br 1(vn)
}}
block 1 (vi: i64) {{
  vone = i64.const 1
  vj = i64.sub vi vone
  vd = i64.eqz vj
  br_if vd 2(vj) 1(vj)
}}
block 2 (vr: i64) {{
  vk = i64.const 42
  return vk
  }}
}}

func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
{call}  return vr
  }}
}}
"#,
        mode = if rebindable { " rebindable" } else { "" },
    )
}

/// `outer()` of [`import_caller`] asked for on `on`: the executor that held it, its reply and what
/// the call was charged.
fn import_outer(on: Executor, how: ImportCall) -> (Executor, Reply, u64) {
    bounded(move || {
        let m = module(&import_caller(how));
        let mut host = Host::new();
        host.set_self_module(&m);
        let kept = temen_ir::FuncType {
            params: vec![],
            results: vec![temen_ir::ValType::I64],
        };
        let binding = match how {
            ImportCall::BoundToAGuestOffer => {
                let funcs: Arc<[temen_ir::Func]> = m.funcs.clone().into();
                let types: Arc<[temen_ir::TypeEntry]> = m.types.clone().into();
                let h = host
                    .wire_offer_func(&funcs, &types, &[2])
                    .expect("wire kept()");
                let id = host.intern_interface(&[], std::slice::from_ref(&kept));
                assert!(temen_interp::guest_implementable(id), "a guest id");
                temen_interp::BoundImport::required(id, 0, h)
            }
            ImportCall::BoundToAHostCap => {
                let h = host.grant_host_proc(temen_interp::HostProcGrant::new(
                    temen_interp::ProcHandler::Sync(Box::new(|_op, _args, out, _mem, _minter| {
                        temen_interp::put_results(out, &[7])
                    })),
                    temen_interp::CapState::Stateless,
                ));
                temen_interp::BoundImport::required(temen_ir::cap_id::HOST_PROC, 0, h)
            }
            ImportCall::Rebindable | ImportCall::Dynamic => temen_interp::BoundImport {
                type_id: temen_ir::cap_id::GUEST_IMPL_BASE,
                op: 1,
                handle: 0,
                bound: false,
                rebindable: matches!(how, ImportCall::Rebindable),
            },
        };
        host.set_import_bindings(vec![binding]);
        let started = temen_run::hold(on, m, 0, &[], u64::MAX, host);
        let (d, _) = started.unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        let mut fuel = 1_000_000u64;
        let reply = d.call(0, 0, &[], &mut fuel);
        (d.executor(), reply, 1_000_000 - fuel)
    })
}

/// #3080 (review H2, T1, T2) — a fiber's import call, by its binding. Each case's reply is
/// `1e6·1 + 1e3·(-9 as u32) + value₁`, the call's answer at the first resume:
/// - bound (not rebindable) to a guest's offer: `hold`'s fiber rule reads the binding's guest id
///   (`jit_proc::may_reach`) and declines it; the tree-walker answers `kept()`'s 42, charged 2 (the
///   wired offer's run is not the call's);
/// - bound for good to a host procedure: nothing declines it, and the JIT holds it with the
///   tree-walker's answer, the procedure's 7, charged 2;
/// - a `rebindable` slot and a `call.import.dyn`: `hold` declines both at `jit_proc::serves`, before
///   the fiber rule, through the same `may_reach` arms (either may reach a serve op), so this row
///   pins only that the tree-walker holds them. The answer is `kept()`'s 42, charged 102: the
///   attach took, and the call ran the domain's own `kept()`. The fiber rule's own arms for them are
///   pinned on the rule itself (`held::tests`).
#[test]
fn a_fibers_import_call_is_held_by_what_its_binding_may_reach() {
    let reply =
        |value: i64| Reply::Results(vec![1_000_000 + 1000 * i64::from(-9i32 as u32) + value]);
    for (how, held, answer, charged) in [
        (ImportCall::BoundToAGuestOffer, Executor::TreeWalker, 42, 2),
        (ImportCall::BoundToAHostCap, Executor::Jit, 7, 2),
        (ImportCall::Rebindable, Executor::TreeWalker, 42, 102),
        (ImportCall::Dynamic, Executor::TreeWalker, 42, 102),
    ] {
        assert_eq!(
            import_outer(Executor::TreeWalker, how),
            (Executor::TreeWalker, reply(answer), charged),
            "{how:?}, asked for the tree-walker"
        );
        assert_eq!(
            import_outer(Executor::Jit, how),
            (held, reply(answer), charged),
            "{how:?}, asked for the JIT"
        );
    }
}
