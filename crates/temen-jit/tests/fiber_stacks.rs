//! #3020 — a run gives back the control stack of every fiber it made, however each was left: a
//! fiber parked on an event (here a futex wait nothing notifies) holds a reference to its own slot in
//! a frame on its own stack, which no count can ever release. Alone in its binary: the count of live
//! stacks is the process's.

use temen_jit::{CompiledModule, JitOutcome, INERT_CAP_THUNK};

/// `func 0 (n)`: make `n` fibers, and resume each once: it waits on 16384, which nothing notifies,
/// and parks. Answer `n`.
const PARKS_ON_A_WAIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vz = i64.const 0
  br 1(vn, vz)
}
block 1 (vn1: i64, vi1: i64) {
  vlt = i64.lt_u vi1 vn1
  br_if vlt 2(vn1, vi1) 3(vn1)
}
block 2 (vn2: i64, vi2: i64) {
  vf = ref.func 1
  vsp = i64.const 4096
  vh = cont.new vf vsp
  va = i64.const 0
  vs, vr = cont.resume vh va
  vone = i64.const 1
  vin = i64.add vi2 vone
  br 1(vn2, vin)
}
block 3 (vn3: i64) {
  return vn3
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  vw = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.const 0
  return vr
  }
}
"#;

#[test]
fn a_run_gives_back_the_stacks_of_fibers_parked_on_an_event() {
    if !temen_fiber::supported() {
        return;
    }
    let Some(before) = temen_fiber::live_stacks() else {
        return; // the guard-paged backend counts nothing
    };
    let m = temen_text::parse_module(PARKS_ON_A_WAIT).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut cm = CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile");
    for run in 1..=5 {
        let (out, _) = cm.run(&[8], None, None).expect("run");
        assert_eq!(out, JitOutcome::Returned(vec![8]), "run {run}");
        assert_eq!(
            temen_fiber::live_stacks(),
            Some(before),
            "run {run} left its parked fibers' stacks live"
        );
    }
}
