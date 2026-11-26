//! #3078 slice 2 (review B1) — a kernel run hands back the stacks of every fiber it leaves behind as
//! it returns, as today's route's run end does (`SharedFiberTable::end`), the one a fault abandoned
//! mid-run included: the fault's recovery jumps past that resume's return, so an `Arc` of the slot
//! in a frame on the vCPU's stack is never dropped, and only the table's end takes the fiber out of
//! it. #3078 slice 3 (review A1): and the stacks of fibers it leaves parked in the kernel's wait
//! index, whose slots stay `RUNNING` with no frame of theirs ever unwound. The arena's count of live
//! stacks is process-wide, so this is a test binary of its own, with one test.

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

/// #3078 slice 3 — a spawned thread makes four fibers and resumes each once: each waits on a word
/// nobody notifies, so each is left parked in the kernel's wait index as the run ends. It answers 4.
const PARKS_IN_FIBERS: &str = "memory 16
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
  vk0 = cont.new vf vz
  vs0, vv0 = cont.resume vk0 vz
  vk1 = cont.new vf vz
  vs1, vv1 = cont.resume vk1 vz
  vk2 = cont.new vf vz
  vs2, vv2 = cont.resume vk2 vz
  vk3 = cont.new vf vz
  vs3, vv3 = cont.resume vk3 vz
  vn = i64.const 4
  return vn
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vq = i64.const 16400
  ve = i32.const 0
  vto = i64.const -1
  vw = i32.atomic.wait vq ve vto
  vw64 = i64.extend_i32_u vw
  return vw64
  }
}
";

#[test]
fn a_kernel_run_hands_back_the_stacks_of_fibers_it_abandons() {
    if !temen_fiber::supported() {
        return;
    }
    let Some(_) = temen_fiber::live_stacks() else {
        return; // the guard-paged backend counts nothing
    };
    let m = temen_text::parse_module(FAULTS_IN_A_FIBER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let parks = temen_text::parse_module(PARKS_IN_FIBERS).expect("parse");
    temen_verify::verify_module(&parks).expect("verify");
    // Bounded, as every suite's kernel lane is: a join that never wakes after the fault fails the
    // test, saying so.
    let run = |m: &temen_ir::Module| {
        jit_kernel::expect_kernel(
            m,
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
        let out = run(&m);
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
        assert_eq!(
            run(&parks),
            JitOutcome::Returned(vec![4]),
            "parking run {i}"
        );
        assert_eq!(
            temen_fiber::live_stacks(),
            before,
            "parking run {i} left a parked fiber's stack live"
        );
    }
}
