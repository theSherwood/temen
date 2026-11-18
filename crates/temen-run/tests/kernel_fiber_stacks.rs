//! #3078 slice 2 (review B1) — a kernel run hands back the stacks of every fiber it leaves behind as
//! it returns, as today's route's run end does (`SharedFiberTable::end`), the one a fault abandoned
//! mid-run included: the fault's recovery jumps past that resume's return, so an `Arc` of the slot
//! in a frame on the vCPU's stack is never dropped, and only the table's end takes the fiber out of
//! it. The arena's count of live stacks is process-wide, so this is a test binary of its own.

use temen_jit::JitOutcome;
use temen_run::jit_kernel;

/// A spawned thread makes one fiber and resumes it; the fiber, while it runs, loads past the window.
const FAULTS_IN_A_FIBER: &str = "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vts, vtv = thread.join vt
  return vtv
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vz = i64.const 0
  vf = ref.func 2
  vk = cont.new vf vz
  vs, vv = cont.resume vk vz
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vfar = i64.const 1099511627776
  vbad = i64.load vfar
  return vbad
  }
}
";

#[test]
fn a_kernel_run_hands_back_the_stack_of_a_fiber_a_fault_abandoned() {
    if !temen_fiber::supported() {
        return;
    }
    let Some(_) = temen_fiber::live_stacks() else {
        return; // the guard-paged backend counts nothing
    };
    let m = temen_text::parse_module(FAULTS_IN_A_FIBER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    // Bounded, as every suite's kernel lane is: a join that never wakes after the fault fails the
    // test, saying so.
    let run = || {
        jit_kernel::expect_kernel(
            &m,
            0,
            &[],
            None,
            u64::MAX,
            jit_kernel::Expect::Kernel { vcpus: 2 },
        )
        .expect("in scope")
    };
    // Before the first run: a one-time leak is a leak too. The arena itself keeps no stack live.
    let before = temen_fiber::live_stacks();
    for i in 1..=5 {
        let out = run();
        assert_eq!(
            out,
            JitOutcome::Trapped(temen_jit::TrapKind::MemoryFault),
            "run {i}"
        );
        assert_eq!(
            temen_fiber::live_stacks(),
            before,
            "run {i} left its abandoned fiber's stack live"
        );
    }
}
