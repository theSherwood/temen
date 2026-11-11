//! #2695 step 2 — what a held domain ([`HeldDomain`]) takes of the process. It keeps its window's
//! share of the process's address space (#2305) while its embedder holds it, as a process holds its
//! own, and hands it back when it ends, as an ordinary root does when its entry returns. On the
//! Cranelift JIT, no pool worker runs its calls: each runs on its caller's thread, parked or not. Its
//! own binary: the address space and the JIT's pool counter are the process's, and tests in one
//! binary share their process, so its tests take turns ([`PROCESS`]).

use std::sync::{Arc, Mutex};
use std::time::Duration;
use temen_interp::{
    maps_pool, reserved_pool, Executor, HeldDomain, Host, Reply, RootWindow, Value,
    MAPS_PER_DOMAIN, MAPS_PER_VCPU,
};

/// The address space a window of the default reservation holds of the pool.
const WINDOW: u64 = 1 << temen_ir::DEFAULT_RESERVED_LOG2;

/// One test at a time: each reads what the process holds.
static PROCESS: Mutex<()> = Mutex::new(());

/// A domain whose offer's ops are `twice(x)`, answering `2x`; `wait_signal()`, waiting until 16424
/// is set; `signal()`, setting it; and `join_worker()`, which releases the worker thread
/// [`WITH_WORKER`] spawned (it waits on 16416), joins it and answers its 70. Its entries answer 1.
const DOMAIN: &str = r#"memory 16
type 0 func (i64) -> (i64)
type 1 func () -> (i64)
type 2 interface { twice: 0, wait_signal: 1, signal: 1, join_worker: 1 }
export 0 interface "svc" 2 { twice: 2, wait_signal: 3, signal: 4, join_worker: 5 }
func () -> (i64) {
block 0 () {
  vr = i64.const 1
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 6 vz vz
  vth = i64.const 16408
  i32.store vth vt
  vr = i64.const 1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 2
  vr = i64.mul vx vk
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vc = i64.const 16424
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vc vexp vto
  vv = i32.load vc
  vr = i64.extend_i32_s vv
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vc = i64.const 16424
  vone = i32.const 1
  i32.store vc vone
  vn = i32.const 1
  vw = atomic.notify vc vn
  vr = i64.const 1
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vc = i64.const 16416
  vone = i32.const 1
  i32.store vc vone
  vn = i32.const 1
  vw = atomic.notify vc vn
  vth = i64.const 16408
  vt = i32.load vth
  vst, vv = thread.join vt
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vc = i64.const 16416
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vc vexp vto
  vr = i64.const 70
  return vr
  }
}
"#;

/// [`DOMAIN`]'s entries: the plain one, and one that spawns a worker thread.
const PLAIN: u32 = 0;
const WITH_WORKER: u32 = 1;

/// [`DOMAIN`]'s ops.
const TWICE: u32 = 0;
const WAIT_SIGNAL: u32 = 1;
const SIGNAL: u32 = 2;
const JOIN_WORKER: u32 = 3;

/// An activation holding a window of the default reservation and nothing else, as another run of the
/// process would: `None` when the pool has no room for it.
fn window() -> Option<Host> {
    let mut h = Host::new();
    let window = RootWindow {
        reserved: WINDOW,
        ..RootWindow::default()
    };
    h.begin_activation(0, window)?;
    Some(h)
}

/// Hold `entry` of [`DOMAIN`] on `on`, and run `f` with it on a thread of its own, so a run that
/// hangs fails the test instead of stalling the suite.
fn with_held<T: Send + 'static>(
    on: Executor,
    entry: u32,
    f: impl FnOnce(HeldDomain) -> T + Send + 'static,
) -> T {
    let m = temen_text::parse_module(DOMAIN).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let m = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&m);
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let (d, results) = temen_run::hold(on, m, entry, &[], u64::MAX, host)
            .unwrap_or_else(|(t, _)| panic!("the entry ends in {t:?}"));
        assert_eq!(d.executor(), on, "the domain runs where it was started");
        assert_eq!(results, vec![Value::I64(1)]);
        let _ = tx.send(f(d));
    });
    rx.recv_timeout(Duration::from_secs(60))
        .expect("the domain ends")
}

/// A call of `op` with `args` and fuel to spare.
fn call(d: &HeldDomain, op: u32, args: &[i64]) -> Reply {
    let mut fuel = 1_000_000;
    d.call(0, op, args, &mut fuel)
}

#[test]
fn a_held_domain_holds_its_window_until_it_ends() {
    let _turn = PROCESS.lock().unwrap_or_else(|e| e.into_inner());
    let pool = reserved_pool();
    let slots = pool / WINDOW;
    if pool == u64::MAX || maps_pool() < 2 * slots * (MAPS_PER_DOMAIN + MAPS_PER_VCPU) {
        eprintln!("skipped: no address-space pool, or a pool of mappings too small to fill it");
        return;
    }
    // In turn, on every executor: the pool is the process's, so one at a time.
    for on in [
        Executor::TreeWalker,
        Executor::Pump,
        Executor::Threads,
        Executor::Jit,
    ] {
        holds_its_window_until_it_ends(on, slots);
    }
}

fn holds_its_window_until_it_ends(on: Executor, slots: u64) {
    // Other runs of the process hold every window but one.
    let mut others: Vec<Host> = std::iter::from_fn(window).collect();
    assert_eq!(
        others.len() as u64,
        slots,
        "the pool holds its windows before the {on:?} run"
    );
    others.pop();
    let (while_held, trap, after) = with_held(on, PLAIN, |d| {
        assert_eq!(call(&d, TWICE, &[21]), Reply::Results(vec![42]));
        let while_held = window().is_some();
        let (host, trap) = d.end();
        // Back at the end, though the embedder still holds the host.
        let after = window().is_some();
        // The probes and the host let go before the report: the next run counts the pool's windows
        // as soon as it hears, and anything still held would be one of them.
        drop(host);
        (while_held, trap, after)
    });
    assert!(!while_held, "a held domain on {on:?} keeps its window");
    assert_eq!(trap, None);
    assert!(after, "its end on {on:?} hands the window back");
    drop(others);
}

/// What a held domain's call costs its caller on each executor, best of five rounds (a perf probe, run
/// on demand, not in CI):
///
///   cargo test --release -p temen-run --test held_domain_pool -- --ignored --nocapture
#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn held_call_cost() {
    let _turn = PROCESS.lock().unwrap_or_else(|e| e.into_inner());
    const N: i64 = 100_000;
    println!("\na held domain's call of `twice`, {N} a round:");
    for on in [
        Executor::TreeWalker,
        Executor::Pump,
        Executor::Threads,
        Executor::Jit,
    ] {
        let ns = with_held(on, PLAIN, |d| {
            let mut best = f64::INFINITY;
            for _ in 0..6 {
                let t = std::time::Instant::now();
                for x in 0..N {
                    assert_eq!(call(&d, TWICE, &[x]), Reply::Results(vec![2 * x]));
                }
                best = best.min(t.elapsed().as_nanos() as f64 / N as f64);
            }
            best
        });
        println!("  {on:?}: {ns:.0} ns a call");
    }
}

/// On the JIT, each call runs on its caller's thread, and one that parks resumes there (D2(a),
/// #2695): no pool worker resumes one, whatever it waits on. Here a call parks on a futex until
/// another call, from another thread, releases it, and a call parks on a join of the domain's own
/// thread, which is no pool task either.
#[test]
fn no_pool_worker_runs_a_call_on_the_jit() {
    let _turn = PROCESS.lock().unwrap_or_else(|e| e.into_inner());
    let at = temen_jit::pool_resumes();
    let trap = with_held(Executor::Jit, WITH_WORKER, |d| {
        for x in 0..10 {
            assert_eq!(call(&d, TWICE, &[x]), Reply::Results(vec![2 * x]));
        }
        let d = Arc::new(d);
        let waiter = std::thread::spawn({
            let d = Arc::clone(&d);
            move || call(&d, WAIT_SIGNAL, &[])
        });
        // Let the waiter park first; if it has not, it finds the signal and answers the same.
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(call(&d, SIGNAL, &[]), Reply::Results(vec![1]));
        assert_eq!(waiter.join().expect("the waiter"), Reply::Results(vec![1]));
        assert_eq!(call(&d, JOIN_WORKER, &[]), Reply::Results(vec![70]));
        let d = Arc::into_inner(d).expect("the waiter let go");
        d.end().1
    });
    assert_eq!(trap, None);
    assert_eq!(
        temen_jit::pool_resumes() - at,
        0,
        "a pool worker resumed a call"
    );
}
