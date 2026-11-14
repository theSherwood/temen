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

/// `func 0 (mode)`: mode 0 makes a fiber and resumes it, and the fiber stores at 8, in the NULL
/// guard below the window: a hardware fault, whose recovery jumps past the resume's return. Any other
/// mode makes the same store at the root.
const FAULTS_IN_A_FIBER: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vm: i64) {
  vz = i64.const 0
  vis0 = i64.eq vm vz
  br_if vis0 1() 2()
}
block 1 () {
  vf = ref.func 1
  vsp = i64.const 4096
  vh = cont.new vf vsp
  va = i64.const 0
  vs, vr = cont.resume vh va
  return vr
}
block 2 () {
  vp2 = i64.const 8
  vz2 = i64.const 0
  i64.store vp2 vz2
  vr2 = i64.const 0
  return vr2
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  vp = i64.const 8
  i64.store vp b
  vr = i64.const 0
  return vr
  }
}
"#;

fn compile(src: &str) -> CompiledModule {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    CompiledModule::compile(
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
    .expect("compile")
}

/// One test, its cases in sequence, since the count is the process's.
#[test]
fn a_run_gives_back_every_stack_it_took() {
    if !temen_fiber::supported() || temen_fiber::live_stacks().is_none() {
        return; // the guard-paged backend counts nothing
    }
    fibers_parked_on_an_event();
    a_fault_inside_a_fiber();
}

/// #3020 — a fiber that faults leaves its slot running (the fault's recovery skips the resume's
/// return): the run's end frees its stack all the same. #3047 — and the next run's root trap on the
/// same thread is the root's, not that fiber's.
fn a_fault_inside_a_fiber() {
    let before = temen_fiber::live_stacks();
    let mut cm = compile(FAULTS_IN_A_FIBER);
    for run in 1..=3 {
        let (out, _) = cm.run(&[0], None, None).expect("run");
        assert_eq!(
            out,
            JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault),
            "run {run}: the fiber's store faults"
        );
        assert_eq!(cm.last_trap_fiber(), Some(0), "run {run}: fiber 0 trapped");
        assert_eq!(
            temen_fiber::live_stacks(),
            before,
            "run {run} left the faulted fiber's stack live"
        );
        let (out, _) = cm.run(&[1], None, None).expect("run");
        assert_eq!(out, JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault));
        assert_eq!(
            cm.last_trap_fiber(),
            Some(-1),
            "run {run}'s root trap is the root's, not the earlier run's faulted fiber's"
        );
    }
}

fn fibers_parked_on_an_event() {
    let before = temen_fiber::live_stacks();
    let mut cm = compile(PARKS_ON_A_WAIT);
    for run in 1..=5 {
        let (out, _) = cm.run(&[8], None, None).expect("run");
        assert_eq!(out, JitOutcome::Returned(vec![8]), "run {run}");
        assert_eq!(
            temen_fiber::live_stacks(),
            before,
            "run {run} left its parked fibers' stacks live"
        );
    }
}
