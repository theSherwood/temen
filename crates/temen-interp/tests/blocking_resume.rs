//! I48 — **blocking `cont.resume`** (`cont.resume.block`): on the tree-walk M:N oracle, resuming
//! a still-event-parked fiber **idles the vCPU** on the fiber's own waiter instead of returning
//! the `FIBER_PARKED (3)` poll status, so a guest whose only pending work is one parked fiber
//! stops busy-spinning. Advisory: returning `FIBER_PARKED` is still conforming (the fast backends
//! and the explorer do exactly that — pinned in `temen/tests/fiber_blocking_resume.rs`); here we
//! pin the oracle's idle-and-wake behaviour and — sharply — that it does **not** spin (fuel).

use temen_interp::{run_capture_reserved_with_host, run_with_host, Host, Value};

/// A single `cont.resume.block` of a fiber that does a 10 ms **timed** `atomic.wait` on a
/// never-notified cell: the resumer idles on the fiber's timer (no guest loop!), the worker
/// wakes on the deadline, and the re-executed op re-claims the woken fiber and switches in — it
/// returns `(FIBER_RETURNED=1, WAIT_TIMED_OUT=2)`. Composite: s*100 + v = 1*100 + 2 = 102.
const BLOCK_ON_TIMED_WAIT: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vs, vv = cont.resume.block vk vz
  vk1 = i64.const 100
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk1
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn block_resume_idles_on_a_timed_wait_then_completes() {
    let m = temen_text::parse_module(BLOCK_ON_TIMED_WAIT).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let start = 1_000_000u64;
    let mut fuel = start;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(102)],
        "the resumer idled on the fiber's 10ms timer, then delivered WAIT_TIMED_OUT"
    );
    // The sharp proof it **idled, not spun**: a busy `cont.resume` poll loop over a 10 ms wait
    // burns millions of fuel (one charge per spin); the blocking resume executes exactly twice
    // (initial park + one wake re-execution) plus the fiber's handful of ops.
    let spent = start - fuel;
    assert!(
        spent < 100,
        "blocking resume must idle, not spin — spent {spent} fuel (a spin would burn millions)"
    );
}

/// The **cross-vCPU wake**: the root spawns a worker (vCPU 1), creates a fiber that futex-waits
/// (untimed) on cell 8, and `cont.resume.block`s it — idling vCPU 0 with no timer at all. The
/// worker stores 7 into cell 8 and `notify`s it; the fiber's wake (`wake_blocked` + `svc_wake`)
/// re-admits the idle resumer, which resumes the fiber to completion. Nothing here has a
/// deadline, so only the wake path can make progress: completing at all proves it. The fiber
/// returns `WAIT_WOKEN(0)*100 + mem[8]`; the root returns `status*10 + that` after joining.
/// Composite: RETURNED(1)*10 + (0 + 7) = 17.
const BLOCK_ON_NOTIFY: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  vf = ref.func 1
  vk = cont.new vf vz
  vs, vv = cont.resume.block vk vz
  vjr = thread.join vt
  vk1 = i64.const 10
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk1
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 8
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  vk = i64.const 100
  vsx = i64.mul vst64 vk
  vm8a = i64.const 8
  vm8 = i64.load vm8a
  vsum = i64.add vsx vm8
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vm8a = i64.const 8
  vseven = i64.const 7
  i64.store vm8a vseven
  vaddr = i64.const 8
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vz = i64.const 0
  return vz
  }
}
"#;

#[test]
fn block_resume_wakes_on_a_cross_vcpu_notify() {
    let m = temen_text::parse_module(BLOCK_ON_NOTIFY).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = 50_000_000u64;
    let init = vec![0u8; 1 << 16];
    let (r, _mem) = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut host);
    let r = r.expect("no trap, no hang");
    // The fiber's wait is untimed on cell 8, whose value the worker changes to 7: the wait
    // completes WAIT_WOKEN (a pre-notify race gives WAIT_NOT_EQUAL — value 1 — but the composite
    // reads mem[8]=7 either way; accept both statuses so the test is race-robust).
    match r.as_slice() {
        // RETURNED(1)*10 + [WAIT_WOKEN(0)*100 + mem8(7)] = 17 (the normal wake), or
        // RETURNED(1)*10 + [WAIT_NOT_EQUAL(1)*100 + mem8(7)] = 117 (the pre-notify race).
        [Value::I64(17)] | [Value::I64(117)] => {}
        o => panic!("unexpected blocking-notify result {o:?}"),
    }
}

/// A blocked resumer is freed on **domain teardown** (the #440 exit path): the root spawns a
/// worker that **traps** (`unreachable`) — domain-wide teardown, invariant 6 — while the root
/// is idle in `cont.resume.block` on an untimed, never-woken fiber. Teardown must reap the
/// parked resumer (swept from `svc_waiters` by identity, the `OfferPark` path) so the run ends
/// promptly rather than hanging on the unwakeable fiber. Uses a trap rather than an `exit`
/// import so no powerbox wiring is needed on the raw interpreter path.
const BLOCK_THEN_SIBLING_TRAPS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  vf = ref.func 1
  vk = cont.new vf vz
  vs, vv = cont.resume.block vk vz
  vse = i64.extend_i32_s vs
  return vse
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  unreachable
  }
}
"#;

#[test]
fn a_sibling_trap_frees_a_blocked_resumer() {
    let m = temen_text::parse_module(BLOCK_THEN_SIBLING_TRAPS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = 50_000_000u64;
    let init = vec![0u8; 1 << 16];
    let t0 = std::time::Instant::now();
    let (r, _mem) = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &init, 0, &mut host);
    let took = t0.elapsed();
    // The worker's trap tears the domain down; the idle blocked resumer is reaped, not left to
    // hang. The run terminates (the root's own outcome is unspecified post-teardown — invariant
    // 6 non-preemption — so we only require prompt termination, not a specific value).
    let _ = r;
    assert!(
        took < std::time::Duration::from_secs(10),
        "teardown must free the blocked resumer promptly (took {took:?})"
    );
}
