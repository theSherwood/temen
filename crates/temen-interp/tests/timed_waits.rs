//! #2224 — a timed wait ends while another task keeps running. The tree-walker and the parallel
//! driver (executor 2) time a wait out on the wall clock. The cooperative pump and the debug
//! scheduler keep a logical clock for determinism, and it used to move only when no task was
//! runnable, so a thread that napped never woke while another spun on it. Now it moves with the
//! work done, one ns per op: the pump moves it by each quantum that runs out, the debug scheduler
//! by each turn.
//!
//! #1979 — and a wait's time passes when nothing runs. The pump used to jump its clock to the next
//! deadline the moment it went idle, so a finite wait ended at once. Now the time to the deadline
//! passes in real time first, as on the tree-walker and the parallel driver.

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/rec.rs"]
mod rec;

use std::time::{Duration, Instant};

use drivers::{agree_on, Driver, Ran, SCHEDULING};
use temen_interp::{CapState, Host, OffloadOutcome, Value};
use temen_ir::SpawnRec;

/// func 0 spawns func 1 and spins until the thread sets the flag at `FLAG`, then joins it and
/// returns its result. func 1 naps for 1 µs on a word nobody notifies, then sets the flag and returns
/// what the wait answered: `WAIT_TIMED_OUT`. Before #2224 the cooperative drivers ran main alone
/// until it ran out of fuel, since the thread's deadline never came.
const NAP: &str = "memory 18
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vchild = thread.spawn 1 vz vz
  br 1(vchild)
  }
block 1 (vchild: i32) {
  vf = i64.const 131072
  vflag = i32.atomic.load vf
  vone = i32.const 1
  vset = i32.eq vflag vone
  br_if vset 2(vchild) 1(vchild)
  }
block 2 (vchild: i32) {
  vr = thread.join vchild
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vw = i64.const 131080
  vexp = i32.const 0
  vto = i64.const 1000
  vs = i32.atomic.wait vw vexp vto
  vf = i64.const 131072
  vone = i32.const 1
  i32.atomic.store vf vone
  vs64 = i64.extend_i32_u vs
  return vs64
  }
}
";

#[test]
fn a_thread_that_naps_wakes_while_another_spins_on_it() {
    let m = temen_text::parse_module(NAP).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let want = Ran {
        result: Ok(vec![Value::I64(2)]), // `WAIT_TIMED_OUT`
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on(
        &SCHEDULING,
        "a nap beside a spinner",
        &m,
        &|| (Host::new(), Vec::new()),
        &want,
    );
}

/// The drivers whose waits take real time. The debug scheduler's time is logical, so that a replay
/// of a run is exact (#2224).
const REAL_TIME: [Driver; 3] = [Driver::Oracle, Driver::Coop, Driver::Parallel];

/// How long a lone wait waits: long enough that a jump to its deadline cannot pass for it.
const WAIT: Duration = Duration::from_millis(20);

/// A root that waits [`WAIT`] on a word nobody notifies, and returns what the wait answered.
fn lone_wait() -> String {
    format!(
        "memory 17
func () -> (i64) {{
block 0 () {{
  vw = i64.const 17800
  vexp = i32.const 0
  vto = i64.const {ns}
  vs = i32.atomic.wait vw vexp vto
  vs64 = i64.extend_i32_u vs
  return vs64
  }}
}}
",
        ns = WAIT.as_nanos()
    )
}

/// The same wait in a detached child: the root (`(i32 inst, i64 budget, i32 child)`) spawns func 1
/// from the record at 17536, joins it, and returns what its wait answered.
fn child_wait() -> String {
    format!(
        "memory 17
func (i32, i64, i32) -> (i64) {{
block 0 (vi: i32, vb: i64, vc: i32) {{
  vma = i64.const {module_at}
  i32.store vma vc
  vb32 = i32.wrap_i64 vb
  vba = i64.const {budget_at}
  i32.store vba vb32
  vrec = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vi (vrec)
  vst, vr = call.cap 6 1 (i32) -> (i64, i64) vi (vch)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vw = i64.const 17800
  vexp = i32.const 0
  vto = i64.const {ns}
  vs = i32.atomic.wait vw vexp vto
  vs64 = i64.extend_i32_u vs
  return vs64
  }}
}}
{rec}",
        module_at = 17536 + SpawnRec::MODULE_AT,
        budget_at = 17536 + SpawnRec::BUDGET_AT,
        ns = WAIT.as_nanos(),
        rec = rec::segment(17536, &SpawnRec::default())
    )
}

/// #1979 — a finite wait with nothing else to run takes its timeout, in a root and in a detached
/// child alike. The pump used to jump its clock to the deadline the moment it went idle, so a 10 ms
/// wait ended after about 80 µs there, where the other drivers waited 10 ms.
#[test]
fn a_lone_wait_takes_its_timeout() {
    let root = temen_text::parse_module(&lone_wait()).expect("parse");
    let child = temen_text::parse_module(&child_wait()).expect("parse");
    for m in [&root, &child] {
        temen_verify::verify_module(m).expect("verify");
    }
    let image = temen_ir::child_image_at(&child, 1).expect("child image");
    let powerbox = || {
        let mut h = Host::new();
        let i = h.grant_instantiator();
        let b = h.grant_budget(-1, -1, -1);
        let c = h.grant_module(&image);
        (h, vec![Value::I32(i), Value::I64(b as i64), Value::I32(c)])
    };
    let mut wrong = Vec::new();
    let mut check = |who: &str, m: &temen_ir::Module, setup: &dyn Fn() -> (Host, Vec<Value>)| {
        for driver in REAL_TIME {
            let began = Instant::now();
            let ran = drivers::run_on(driver, m, setup).map(|r| r.result);
            let took = began.elapsed();
            // `WAIT_TIMED_OUT`, after the wait's time.
            if ran != Some(Ok(vec![Value::I64(2)])) || took < WAIT {
                wrong.push(format!("  {who} on {driver:?}: {ran:?} after {took:?}"));
            }
        }
    };
    check("a root", &root, &|| (Host::new(), Vec::new()));
    check("a child", &child, &powerbox);
    assert!(
        wrong.is_empty(),
        "a lone {WAIT:?} wait must time out after {WAIT:?}, but\n{}",
        wrong.join("\n")
    );
}

/// How long the root's host call waits, and how long each of its thread's waits is.
const CALL_TIME: Duration = Duration::from_millis(200);
const NAP_TIME: Duration = Duration::from_millis(50);

/// A root given an offloadable host procedure: it starts func 1 as a thread, calls the procedure
/// (op 0), raises the flag at 17900, and returns what the thread returned. The thread naps
/// [`NAP_TIME`] at a time on a word nobody notifies until it sees the flag, and returns how many
/// naps it took.
fn naps_while_the_root_waits() -> String {
    format!(
        "memory 17
func (i32) -> (i64) {{
block 0 (vcall: i32) {{
  vz = i64.const 0
  vth = thread.spawn 1 vz vz
  vr = call.cap 13 0 (i64) -> (i64) vcall (vz)
  vf = i64.const 17900
  vone = i32.const 1
  i32.atomic.store vf vone
  vnaps = thread.join vth
  return vnaps
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vn0 = i64.const 0
  br 1(vn0)
  }}
block 1 (vn: i64) {{
  vf = i64.const 17900
  vflag = i32.atomic.load vf
  br_if vflag 3(vn) 2(vn)
  }}
block 2 (vn2: i64) {{
  vw = i64.const 17800
  vexp = i32.const 0
  vto = i64.const {nap}
  vs = i32.atomic.wait vw vexp vto
  vone = i64.const 1
  vn3 = i64.add vn2 vone
  br 1(vn3)
  }}
block 3 (vnaps: i64) {{
  return vnaps
  }}
}}
",
        nap = NAP_TIME.as_nanos()
    )
}

/// #1979 — while a root waits on the world (#2047: an offloadable host call, here one that sleeps
/// [`CALL_TIME`]), its thread's [`NAP_TIME`] naps keep real time: about four of them. The pump used
/// to time each out the moment it went idle, so the thread took 234,549 naps in those 200 ms.
#[test]
fn naps_keep_real_time_while_the_root_waits_on_the_world() {
    let m = temen_text::parse_module(&naps_while_the_root_waits()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let setup = || {
        let mut h = Host::new();
        let call = h.grant_host_proc_offloadable(
            Box::new(|_op, _args| {
                OffloadOutcome::Offload(Box::new(|| {
                    std::thread::sleep(CALL_TIME);
                    0
                }))
            }),
            CapState::Stateless,
        );
        (h, vec![Value::I32(call)])
    };
    let mut wrong = Vec::new();
    for driver in REAL_TIME {
        let began = Instant::now();
        let ran = drivers::run_on(driver, &m, &setup).map(|r| r.result);
        // As many naps as fit in the run, and the one under way when the flag rose.
        let most = (began.elapsed().as_nanos() / NAP_TIME.as_nanos()) as i64 + 2;
        match ran {
            Some(Ok(v)) if matches!(v[..], [Value::I64(n)] if (1..=most).contains(&n)) => {}
            other => wrong.push(format!("  {driver:?}: {other:?}, at most {most}")),
        }
    }
    assert!(
        wrong.is_empty(),
        "a thread napping {NAP_TIME:?} at a time while its root waits {CALL_TIME:?} must nap \
         in real time, but\n{}",
        wrong.join("\n")
    );
}
