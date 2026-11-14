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
