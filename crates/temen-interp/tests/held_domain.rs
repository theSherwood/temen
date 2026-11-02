//! #2695 step 2 — a **held** domain ([`Host::hold`]) on the tree-walker, the oracle. Its entry
//! returns and it rests instead of ending the run: it serves each call queued on it as an
//! activation, a new vCPU of the domain at the op's handler, called with the call's arguments, and
//! its threads run on meanwhile. A member's trap or exit still ends it, and the run ends once nothing
//! in it can run. The other engines and the debugger decline a held domain, and a durable one is
//! refused.

#[path = "support/rec.rs"]
mod rec;

use std::sync::{mpsc, Arc};
use std::time::Duration;
use temen_interp::{
    run_capture_reserved_with_host, run_with_host_fast, Host, Inspector, Reply, Stop, Trap, Value,
    GRANT_SERVE_LIVE_TAG,
};
use temen_ir::SpawnRec;

/// The ops of the domain's one offer, `"svc"`.
const PAIR: u32 = 0;
const BOOM: u32 = 1;
const QUIT: u32 = 2;
const JOIN_WORKER: u32 = 3;
const STUCK: u32 = 4;

/// The entries: the plain one, one that spawns a worker thread, one that spawns a child domain.
const ENTRY: u32 = 0;
const ENTRY_WITH_WORKER: u32 = 1;
const ENTRY_WITH_CHILD: u32 = 2;

/// The held domain. Every entry stores 7 at 16384 and answers 1; each handler is one op of
/// `"svc"`:
///
/// - `pair(x)` answers `(x + mem[16384], 2x)`.
/// - `boom()` traps `Unreachable`; `quit()` exits 3 through the `Exit` handle the entry kept.
/// - `join_worker()` releases the worker thread [`ENTRY_WITH_WORKER`] spawned (it waits on 16416)
///   and answers what it returns: `mem[16384] * 10`.
/// - `stuck()` waits on 16424, which nothing notifies.
/// - `run(a, b)` answers `a + b` and adds it to 16432. [`ENTRY_WITH_CHILD`]'s child calls it through
///   the live self-serve grant (#744) the spawn gives it, named `"exec"`.
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
type 4 interface {{ pair: 0, boom: 1, quit: 1, join_worker: 2, stuck: 2, run: 3 }}
export 0 interface "svc" 4 {{ pair: 3, boom: 4, quit: 5, join_worker: 6, stuck: 7, run: 8 }}

func (i32) -> (i64) {{
block 0 (vexit: i32) {{
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vh = i64.const 16400
  i32.store vh vexit
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
  vt = thread.spawn 9 vz vz
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
  vma = i64.const {module_at}
  i32.store vma vm
  vba = i64.const {budget_at}
  i32.store vba vb
  vgh = i32.const {tag}
{grant}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vr = i64.extend_i32_s vch
  return vr
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
  vr = i64.extend_i32_s vst
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

/// The child [`ENTRY_WITH_CHILD`] spawns: it resolves `"exec"`, the offer over its parent, and calls
/// `run(40, 2)`, then `run` again with that answer, `run(42, 0)`.
const CHILD: &str = r#"memory 15
data 16384 "exec"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vnp = i64.const 16384
  vl = i64.const 4
  vexec = self.resolve vnp vl
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 5 (i64, i64) -> (i64) vexec (va, vb)
  vz = i64.const 0
  vr2 = call.cap 268435456 5 (i64, i64) -> (i64) vexec (vr, vz)
  return vr2
  }
}
"#;

fn module(src: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
    Arc::new(m)
}

/// A host whose domain is `m` and which the embedder holds.
fn held(m: &Arc<temen_ir::Module>) -> Host {
    let mut host = Host::new();
    host.set_self_module(m);
    host.hold();
    host
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

type Run = (Result<Vec<Value>, Trap>, Vec<u8>, Host);

/// Run `entry` of `m` over `host` on the tree-walker. Answers the run, its window and the host.
fn run(m: &Arc<temen_ir::Module>, entry: u32, args: Vec<Value>, mut host: Host) -> Run {
    let m = Arc::clone(m);
    bounded(move || {
        let mut fuel = 50_000_000u64;
        let init = vec![0u8; 1 << 17];
        let (r, window) =
            run_capture_reserved_with_host(&m, entry, &args, &mut fuel, &init, 0, &mut host);
        (r, window, host)
    })
}

fn i64_at(window: &[u8], at: usize) -> i64 {
    i64::from_le_bytes(window[at..at + 8].try_into().expect("8 bytes"))
}

/// Calls queued before the run are served once the entry has returned: each activation sees what
/// the entry wrote and replies with both of its results.
#[test]
fn a_held_domain_serves_the_calls_queued_on_it_after_its_entry_returns() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let t1 = host.svc_enqueue(0, PAIR, vec![5]).expect("queued");
    let t2 = host.svc_enqueue(0, PAIR, vec![30]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Ok(vec![Value::I64(1)]), "the run answers the entry");
    assert_eq!(host.svc_result(t1), Some(Reply::Results(vec![12, 10])));
    assert_eq!(host.svc_result(t2), Some(Reply::Results(vec![37, 60])));
}

/// The same calls on a domain no one holds: its entry's return ends the run, and nothing serves
/// them.
#[test]
fn a_domain_no_one_holds_ends_with_its_entry() {
    let m = module(&parent());
    let mut host = Host::new();
    host.set_self_module(&m);
    let exit = host.grant_exit();
    let t = host.svc_enqueue(0, PAIR, vec![5]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Ok(vec![Value::I64(1)]));
    assert_eq!(host.svc_result(t), None, "no activation ran");
}

/// A call of another arity than its handler is answered `-EINVAL`, as a serve loop answers it, and
/// the others are served.
#[test]
fn a_call_of_another_arity_is_answered_einval() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let bad = host.svc_enqueue(0, PAIR, vec![5, 6]).expect("queued");
    let good = host.svc_enqueue(0, PAIR, vec![5]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Ok(vec![Value::I64(1)]));
    assert_eq!(host.svc_result(bad), Some(Reply::Failed(-22)));
    assert_eq!(host.svc_result(good), Some(Reply::Results(vec![12, 10])));
}

/// An activation's trap ends its domain, as any member's does: the run answers the trap, and the
/// call has no reply.
#[test]
fn a_trap_in_an_activation_ends_the_domain() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let t = host.svc_enqueue(0, BOOM, vec![]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Err(Trap::Unreachable));
    assert_eq!(host.svc_result(t), None);
}

/// So does its exit.
#[test]
fn an_exit_in_an_activation_ends_the_domain() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    host.svc_enqueue(0, QUIT, vec![]).expect("queued");
    let (r, _, _) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Err(Trap::Exit(3)));
}

/// A held domain is a process: a thread its entry spawned runs on after the entry returns, and an
/// activation releases it, joins it and answers what it computed.
#[test]
fn a_thread_the_entry_spawned_runs_on_for_an_activation() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let t = host.svc_enqueue(0, JOIN_WORKER, vec![]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY_WITH_WORKER, vec![Value::I32(exit)], host);
    assert_eq!(r, Ok(vec![Value::I64(1)]));
    assert_eq!(host.svc_result(t), Some(Reply::Results(vec![70])));
}

/// With no call to serve and its one thread waiting on what nothing will send, nothing in the run
/// can run: it ends, and answers the entry, as a run whose root returned past a waiting thread does.
#[test]
fn the_run_ends_when_only_waiting_threads_remain() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let (r, _, _) = run(&m, ENTRY_WITH_WORKER, vec![Value::I32(exit)], host);
    assert_eq!(r, Ok(vec![Value::I64(1)]));
}

/// An activation that waits on what nothing will send strands its call: the stall verdict ends the
/// domain, as it ends a run whose root waits so, and the run answers `ThreadFault`.
#[test]
fn a_call_the_stall_verdict_strands_ends_the_domain() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let t = host.svc_enqueue(0, STUCK, vec![]).expect("queued");
    let (r, _, mut host) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Err(Trap::ThreadFault));
    assert_eq!(host.svc_result(t), None);
}

/// A call from inside the run lands on the resting domain too: the child its entry spawned calls
/// `run(40, 2)` through the live self-serve grant, gets 42 back, and calls `run(42, 0)`. The parent's
/// window holds what both activations added.
#[test]
fn a_call_from_a_child_in_the_run_is_served_while_its_parent_rests() {
    let m = module(&parent());
    let mut host = held(&m);
    let args = vec![
        Value::I32(host.grant_instantiator()),
        Value::I32(host.grant_module(&module(CHILD))),
        Value::I32(host.grant_budget(-1, 1 << 20, -1)),
    ];
    let (r, window, _) = run(&m, ENTRY_WITH_CHILD, args, host);
    let child = match r.as_deref() {
        Ok([Value::I64(h)]) => *h,
        other => panic!("the entry answers the child's handle, not {other:?}"),
    };
    assert!(child >= 0, "the spawn succeeds: {child}");
    assert_eq!(i64_at(&window, 16432), 42 + 42, "both calls were served");
}

/// A durable domain cannot rest yet: holding one is refused.
#[test]
fn a_held_durable_domain_is_refused() {
    let m = module(&parent());
    let mut host = held(&m);
    host.set_durable(true);
    let exit = host.grant_exit();
    let (r, _, _) = run(&m, ENTRY, vec![Value::I32(exit)], host);
    assert_eq!(r, Err(Trap::ThreadFault));
}

/// The bytecode engine declines a held domain, so the fast entry answers as the tree-walker does.
#[test]
fn the_fast_entry_serves_a_held_domain_on_the_tree_walker() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let t = host.svc_enqueue(0, PAIR, vec![5]).expect("queued");
    let m2 = Arc::clone(&m);
    let (r, mut host) = bounded(move || {
        let mut fuel = 50_000_000u64;
        let r = run_with_host_fast(&m2, ENTRY, &[Value::I32(exit)], &mut fuel, &mut host);
        (r, host)
    });
    assert_eq!(r, Ok(vec![Value::I64(1)]));
    assert_eq!(host.svc_result(t), Some(Reply::Results(vec![12, 10])));
}

/// The debugger ends a run at its entry, so it declines a held domain, and a seek does not re-run it
/// unheld.
#[test]
fn the_debugger_declines_a_held_domain() {
    let m = module(&parent());
    let mut host = held(&m);
    let exit = host.grant_exit();
    let mut dbg = Inspector::attach_with_host(&m, ENTRY, &[Value::I32(exit)], 1_000_000, host);
    assert!(matches!(
        dbg.run_until_stop(),
        Stop::Finished(Err(Trap::ThreadFault))
    ));
    assert!(matches!(dbg.seek(1), Stop::Blocked));
}
