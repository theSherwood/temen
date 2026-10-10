//! #2224 — a timed wait ends while another task keeps running. The tree-walker and the parallel
//! driver (executor 2) time a wait out on the wall clock. The cooperative pump and the debug
//! scheduler keep a logical clock for determinism, and it used to move only when no task was
//! runnable, so a thread that napped never woke while another spun on it. Now it moves with the
//! work done, one ns per op: the pump moves it by each quantum that runs out, the debug scheduler
//! by each turn.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{agree_on, Ran, SCHEDULING};
use temen_interp::{Host, Value};

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
