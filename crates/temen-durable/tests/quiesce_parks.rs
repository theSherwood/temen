//! **#1584 — a parked child must not veto its parent's freeze.** DURABILITY.md §13.4 4c-bis
//! generalized from one park kind to every park the scheduler owns.
//!
//! Every freeze mechanism in the tree asks the frozen domain to *cooperate*: it polls its window's
//! state word at a safepoint and unwinds itself. That is a fine implementation, but it must not be
//! the authority model. INVARIANTS #3 has authority moving **down** the grant graph — a parent
//! grants the window, the fuel, the lifecycle — and nothing in it says a child may decline to be
//! stopped. Yet a vCPU parked in `atomic.wait` runs no ops, so it reaches no safepoint, so it
//! silently vetoed an operation its parent was authorized to perform.
//!
//! 4c-bis already solved this for one park kind: when the run would otherwise block, promote each
//! `svc.wait`-parked vCPU's `dstate` to `UNWINDING` and re-admit it, so its re-executed `svc.wait`
//! unwinds. These pin the same rule for the rest — futex and join — which the `temen-durable`
//! transform already instruments as re-issue suspend points (`SuspendKind::MemoryWait`), exactly
//! like the serve op. The scheduler simply never woke them.

use std::time::{Duration, Instant};
use temen_durable::{
    arm_freeze_after_backedges, arm_freeze_on_quiesce, begin_thaw, init_durable_window,
    transform_module_assume_confined, write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::Memory;

const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};
const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrumented(src: &str) -> std::sync::Arc<temen_ir::Module> {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// The root parks in an **infinite** `atomic.wait` on a word nothing ever stores or notifies. No
/// timer, no peer, no safepoint it can reach on its own: the only thing that can end this park is
/// the freeze its owner asked for. Returns `1000 + status` if it ever comes back.
const SRC_FUTEX_PARKED_ROOT: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  vk = i64.const 1000
  vr = i64.add vk vst64
  return vr
  }
}
"#;

/// **A futex-parked vCPU freezes.** Armed for freeze-on-quiesce, the run reaches a state where
/// nothing is runnable and the only thing alive is a vCPU parked in an infinite `atomic.wait`.
///
/// Before #1584 the quiesce arm fired only for `svc.wait` waiters, and it was gated on an empty
/// timer heap — which an infinite wait is not, because the clamp gives it a `MAX_WAIT` backstop
/// entry. So the park vetoed the freeze, and not cleanly: the run stalled the full **10 s**
/// backstop, the wait "timed out", and the root returned `1002` as an ordinary result with a
/// snapshot that was not a freeze point (thawing it gave `Err(Unreachable)`) and nothing in the
/// `Ok` to say so. Now the freeze fires in ~0.1 s with the unwind return, and the snapshot thaws.
#[test]
fn a_futex_parked_vcpu_does_not_veto_its_owners_freeze() {
    let inst = instrumented(SRC_FUTEX_PARKED_ROOT);
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_on_quiesce(&mut win);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the owner asked for a freeze and must get one — the unwind return, not a result"
    );

    // And it is a real freeze, not an abandonment: thawing re-issues the wait, which this time
    // finds the word already changed and returns NOT_EQUAL (1) rather than parking again.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    let mut win2 = snap.clone();
    win2[66000..66004].copy_from_slice(&1i32.to_le_bytes());
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel2 = 1_000_000u64;
    let (r2, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel2, &win2, SIZE_LOG2, &mut h2);
    assert_eq!(
        r2,
        Ok(vec![Value::I64(1001)]),
        "the thawed root re-issues its wait and sees the changed word (1000 + NOT_EQUAL)"
    );
}

/// The root spawns a sibling that parks forever in `atomic.wait`, then parks itself in
/// `thread.join` on it. Two different scheduler-owned parks, stacked. Returns
/// `2000 + 100·sibling_status` if it ever comes back.
const SRC_JOIN_ON_A_FUTEX_PARKED_CHILD: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr = thread.join vt
  vk = i64.const 2000
  vs = i64.add vk vr
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  vk = i64.const 100
  vr = i64.mul vst64 vk
  return vr
  }
}
"#;

/// **A freeze is run-wide, so every parked vCPU takes the phase.** The sibling is re-admitted
/// (nothing else could wake it); the root is not, because its `join` *will* be woken by the
/// sibling completing. But it still needs `dstate = UNWINDING`, and that is the half worth a test
/// of its own: with the sibling re-admitted and the root left `NORMAL`, the root woke from its
/// join with the sibling's unwind value, never observed the freeze at its own safepoint, and ran
/// to completion — returning `2000` as an ordinary result, straight *through* the freeze its owner
/// had asked for. Re-admitting more parks without propagating the phase would have traded one
/// silent veto for a subtler one.
#[test]
fn a_join_parked_root_takes_the_freeze_its_child_was_re_admitted_for() {
    let inst = instrumented(SRC_JOIN_ON_A_FUTEX_PARKED_CHILD);
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_on_quiesce(&mut win);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "both vCPUs unwind: the root must not run through the freeze on its child's coat-tails"
    );
    let (frozen, root_sp) = (h.frozen_vcpus().to_vec(), h.frozen_root_sp());
    assert_eq!(frozen.len(), 1, "the child recorded its re-attach residue");

    // The round trip: hand the child's re-attach residue to the thaw host, change the word, thaw.
    // The child re-issues its wait and gets NOT_EQUAL (1 · 100), the root re-issues its join and
    // reaps it — 2000 + 100.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    h2.set_frozen_vcpus(frozen);
    if let Some(sp) = root_sp {
        h2.set_frozen_root_sp(sp);
    }
    let mut win2 = snap.clone();
    win2[66000..66004].copy_from_slice(&1i32.to_le_bytes());
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel2 = 1_000_000u64;
    let (r2, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel2, &win2, SIZE_LOG2, &mut h2);
    assert_eq!(
        r2,
        Ok(vec![Value::I64(2100)]),
        "the thawed subtree re-issues both suspend points and completes"
    );
}

// The shapes below reach a park from the other side: the freeze is already under way when the
// scheduler finds a vCPU parked. #1619 taught the quiesce arm to drain every park the scheduler
// owns, but only the quiesce arm; a freeze triggered any other way left a parked vCPU where it was,
// and the deadlock check then reaped it — so the "freeze" returned `Ok` with that vCPU missing from
// the cut. The same body now runs for both triggers.

type Inst = std::sync::Arc<temen_ir::Module>;

/// Run `inst` durably with the window prepared by `arm`; return the result, the snapshot, and the
/// host holding the freeze residue.
fn freeze(
    inst: &Inst,
    arm: impl FnOnce(&mut Vec<u8>),
) -> (Result<Vec<Value>, Trap>, Vec<u8>, Host) {
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm(&mut win);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    (r, snap, h)
}

/// Thaw `snap` with the residue `h` recorded, after changing the word every kernel here waits on —
/// so each re-issued wait returns `NOT_EQUAL` at once instead of parking again.
fn thaw_with_the_word_changed(inst: &Inst, snap: &[u8], h: &Host) -> Result<Vec<Value>, Trap> {
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(inst);
    h2.set_frozen_vcpus(h.frozen_vcpus().to_vec());
    if let Some(sp) = h.frozen_root_sp() {
        h2.set_frozen_root_sp(sp);
    }
    h2.set_frozen_fibers(h.frozen_fibers().to_vec());
    let mut win = snap.to_vec();
    win[66000..66004].copy_from_slice(&1i32.to_le_bytes());
    begin_thaw(&mut win, TEST_ARENA, 0);
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h2).0
}

/// The root spawns a sibling that parks forever, then sleeps 1 ms in a timed wait on a private word
/// — the single freeze worker runs the sibling meanwhile, so it is parked *before* anything else
/// happens — then runs a loop in which the back-edge countdown fires the freeze, then joins.
const SRC_SIBLING_PARKED_BEFORE_THE_FREEZE: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vsl = i64.const 66008
  vse = i32.const 0
  vto = i64.const 1000000
  vslept = i32.atomic.wait vsl vse vto
  vi0 = i64.const 0
  br 1(vt, vi0)
}
block 1 (vt1: i32, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 1000
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vt1, vi2) 2(vt1)
}
block 2 (vt2: i32) {
  vr = thread.join vt2
  vk = i64.const 2000
  vs = i64.add vk vr
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  vk = i64.const 100
  vr = i64.mul vst64 vk
  return vr
  }
}
"#;

/// **A freeze that arrives while a sibling is parked brings it through.** The sibling parked in
/// `NORMAL`, before the freeze existed, so it holds no phase and reaches no safepoint. It used to be
/// reaped once the root had unwound: the freeze returned `Ok` with **no residue for the sibling**, a
/// cut with a vCPU missing, and thawing it faulted `ThreadFault`. Now the in-flight freeze
/// re-admits it, it unwinds at its wait, and the round trip reproduces the uninterrupted answer.
#[test]
fn a_freeze_in_flight_brings_a_sibling_parked_before_it() {
    let inst = instrumented(SRC_SIBLING_PARKED_BEFORE_THE_FREEZE);
    let (r, snap, h) = freeze(&inst, |w| arm_freeze_after_backedges(w, 5));
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );
    assert_eq!(
        h.frozen_vcpus().len(),
        1,
        "the parked sibling is in the cut — it used to be reaped and silently left out"
    );
    assert_eq!(
        thaw_with_the_word_changed(&inst, &snap, &h),
        Ok(vec![Value::I64(2100)]),
        "the sibling re-issues its wait (NOT_EQUAL, 1·100), the root finishes its loop and joins"
    );
}

/// **A child that parks inside a freeze is captured.** The run starts `UNWINDING`, the root spawns a
/// child and joins it, and the child's first act is an infinite wait. Under a freeze a futex wait
/// still parks, so the child parked *with* the phase and nothing ever woke it: the deadlock check
/// reaped the whole run with `ThreadFault`, where the JIT (whose futex park observes a freeze on
/// its own) freezes the same program. A parked vCPU carrying the phase is a freeze in flight, so it
/// is re-admitted like any other.
#[test]
fn a_child_that_parks_inside_an_in_flight_freeze_is_captured() {
    let inst = instrumented(SRC_JOIN_ON_A_FUTEX_PARKED_CHILD);
    let (r, snap, h) = freeze(&inst, |w| write_state(w, STATE_UNWINDING));
    assert_eq!(r, Ok(vec![Value::I64(0)]), "a freeze, not a ThreadFault");
    assert_eq!(
        h.frozen_vcpus().len(),
        1,
        "the child recorded its re-attach residue"
    );
    assert_eq!(
        thaw_with_the_word_changed(&inst, &snap, &h),
        Ok(vec![Value::I64(2100)])
    );
}

/// The root drives a fiber with `cont.resume.block`; the fiber parks forever in `atomic.wait`.
/// Returns `1000 + fiber value` once the fiber returns, where the fiber returns `100 + status`.
const SRC_FIBER_PARKED_UNDER_A_BLOCKING_RESUME: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  br 1(v2)
}
block 1 (vk: i64) {
  vz = i64.const 0
  vs, vv = cont.resume.block vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vv) 1(vk)
}
block 2 (vr: i64) {
  vk2 = i64.const 1000
  vout = i64.add vk2 vr
  return vout
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  vk = i64.const 100
  vr = i64.add vk vst64
  return vr
  }
}
"#;

/// **A futex-parked fiber does not veto freeze-on-quiesce.** Two things stood in the way. A durable
/// run took I48's advisory downgrade unconditionally, so `cont.resume.block` returned `FIBER_PARKED`
/// and the resumer **spun** — the run never quiesced, and instead of freezing it ran its whole fuel
/// budget out (`OutOfFuel`, 2.6 s on this kernel). And `quiesced_parks_only` held the arm off for any
/// fiber waiter. Now the resumer idles like it does on a non-durable run, the fiber park counts as
/// quiesced, the resumer is re-admitted to unwind, and the fiber is flattened by `freeze_drive`.
#[test]
fn a_futex_parked_fiber_does_not_veto_freeze_on_quiesce() {
    let inst = instrumented(SRC_FIBER_PARKED_UNDER_A_BLOCKING_RESUME);
    let t = Instant::now();
    let (r, snap, h) = freeze(&inst, |w| arm_freeze_on_quiesce(w));
    assert_eq!(r, Ok(vec![Value::I64(0)]), "a freeze, not OutOfFuel");
    assert!(
        t.elapsed() < Duration::from_secs(4),
        "idle, not a spin: {:?}",
        t.elapsed()
    );
    assert_eq!(
        h.frozen_fibers().len(),
        1,
        "the parked fiber is flattened into the cut"
    );
    assert_eq!(
        thaw_with_the_word_changed(&inst, &snap, &h),
        Ok(vec![Value::I64(1101)]),
        "the fiber re-issues its wait (NOT_EQUAL → 100 + 1), the resumer collects it"
    );
}

/// The same kernel on a durable run **not** armed to freeze. The fiber's wait can never be
/// satisfied, and the resumer is now idle rather than spinning, so the run is a genuine deadlock
/// and faults promptly — what a non-durable run already does (#1639) — instead of burning its fuel
/// before `OutOfFuel`.
#[test]
fn an_unarmed_durable_run_idles_into_the_deadlock_verdict() {
    let inst = instrumented(SRC_FIBER_PARKED_UNDER_A_BLOCKING_RESUME);
    let t = Instant::now();
    let (r, _, _) = freeze(&inst, |_| {});
    assert_eq!(r, Err(Trap::ThreadFault));
    assert!(t.elapsed() < Duration::from_secs(4), "{:?}", t.elapsed());
}

/// **And the thawed run is the same run.** Thaw that freeze with the word *unchanged* and the quiesce
/// arm set again: the fiber re-issues its wait and parks, the resumer — its rewind complete — idles
/// on it once more, and the run freezes a second time. This is the path the durable idle opens that
/// nothing else exercises: a resumer parking after a thaw rather than before a freeze.
#[test]
fn a_thawed_fiber_park_idles_and_freezes_again() {
    let inst = instrumented(SRC_FIBER_PARKED_UNDER_A_BLOCKING_RESUME);
    let (r, snap, h) = freeze(&inst, |w| arm_freeze_on_quiesce(w));
    assert_eq!(r, Ok(vec![Value::I64(0)]));

    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    h2.set_frozen_fibers(h.frozen_fibers().to_vec());
    if let Some(sp) = h.frozen_root_sp() {
        h2.set_frozen_root_sp(sp);
    }
    let mut win2 = snap.clone();
    begin_thaw(&mut win2, TEST_ARENA, 0);
    arm_freeze_on_quiesce(&mut win2);
    let mut fuel = 1_000_000u64;
    let (r2, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win2, SIZE_LOG2, &mut h2);
    assert_eq!(
        r2,
        Ok(vec![Value::I64(0)]),
        "the thawed run parks again and re-freezes"
    );
    assert_eq!(h2.frozen_fibers().len(), 1, "the fiber is back in the cut");
}

/// The root spawns a sibling (handing it the pipe's write end), then reads one byte from the pipe's
/// read end: the pipe is empty and its writer open, so the root parks. The sibling loops (the
/// back-edge countdown fires the freeze inside it), then writes `x` and returns 7. The root joins
/// and returns `1000·n + byte + 7`.
const SRC_PIPE_PARKED_ROOT: &str = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vr: i32, vw: i32) {
  vz = i64.const 0
  vw64 = i64.extend_i32_u vw
  vt = thread.spawn 1 vz vw64
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vr (vbuf, vlen)
  vj = thread.join vt
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vs = i64.add vnk vb64
  vres = i64.add vs vj
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vi0 = i64.const 0
  br 1(varg, vi0)
}
block 1 (va: i64, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 1000
  vmore = i64.ne vi2 vlim
  br_if vmore 1(va, vi2) 2(va)
}
block 2 (va2: i64) {
  vw = i32.wrap_i64 va2
  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vlen)
  vr = i64.const 7
  return vr
  }
}
"#;

/// **#1672 — a pipe-parked vCPU is abandoned, not waited on, and its read is re-issued on thaw.**
/// The root parks reading an empty pipe, and the freeze lands in its sibling before the sibling has
/// written. Nothing in the cut would wake the read, so before #1672 the freeze waited on it for good.
/// Now the in-flight freeze re-admits the root, its re-executed read is abandoned (no effect, the
/// re-issue word set), and it unwinds. Two things would make the thaw wrong, and this pins both: a
/// *reloaded* result (the read's placeholder `0`) instead of a re-issued read, and the sibling's
/// unwind releasing the domain's pipe ends as if it had exited, which drops the writer count to 0 and
/// wakes the root to a false EOF mid-freeze. On thaw the sibling finishes its loop and writes, and the
/// root's re-issued read gets the byte: `1000·1 + 'x' + 7`.
#[test]
fn a_pipe_parked_root_is_abandoned_and_its_read_reissued_on_thaw() {
    let inst = instrumented(SRC_PIPE_PARKED_ROOT);
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    let (w, r) = h.grant_pipe();
    let args = [Value::I32(r), Value::I32(w)];
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after_backedges(&mut win, 5);
    let mut fuel = 1_000_000u64;
    let (res, snap) =
        run_capture_reserved_with_host(&inst, 0, &args, &mut fuel, &win, SIZE_LOG2, &mut h);
    assert_eq!(
        res,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );
    assert_eq!(h.frozen_vcpus().len(), 1, "the sibling is in the cut");

    // An in-memory thaw on the same powerbox: the pipe and its ends are still there.
    let mut win2 = snap.clone();
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel2 = 1_000_000u64;
    let (res2, _) =
        run_capture_reserved_with_host(&inst, 0, &args, &mut fuel2, &win2, SIZE_LOG2, &mut h);
    assert_eq!(
        res2,
        Ok(vec![Value::I64(1000 + i64::from(b'x') + 7)]),
        "the thawed root re-issues its read and gets the sibling's byte"
    );
}
