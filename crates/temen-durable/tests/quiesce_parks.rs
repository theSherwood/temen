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

const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);
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

/// #1851: the quiesce flag and an allocating program's heap pointer (`POWERBOX_HEAP_BRK`, one guard
/// up) were the same window word. A durable run whose heap pointer had a non-zero low byte came up
/// armed to freeze on quiesce though nobody armed it, and arming the flag rewrote the pointer.
#[test]
fn a_heap_pointer_neither_arms_nor_is_changed_by_the_quiesce_flag() {
    let brk = (temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_HEAP_BRK) as usize;
    // What a C or nim `_start` leaves there: a 16-aligned heap pointer.
    let heap = 0x2_0010u64.to_le_bytes();

    // Unarmed: the parked root is a deadlock, as without a heap.
    let inst = instrumented(SRC_FUTEX_PARKED_ROOT);
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    win[brk..brk + 8].copy_from_slice(&heap);
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert_eq!(r, Err(Trap::ThreadFault), "no one armed a freeze");

    // Nor on a run that is not durable at all, whatever it keeps where the flag would be: it has no
    // durable control words (the C on-ramp stages names in that scratch).
    let mut m = temen_text::parse_module(SRC_FUTEX_PARKED_ROOT).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: None,
    });
    let mut h = Host::new();
    let mut plain = vec![0u8; WINDOW];
    plain[brk..brk + 8].copy_from_slice(&heap);
    plain[temen_ir::durable_abi::ARM_QUIESCE_OFF as usize] = b'm';
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(&m, 0, &[], &mut fuel, &plain, SIZE_LOG2, &mut h);
    assert_eq!(r, Err(Trap::ThreadFault), "a plain run deadlocks");

    // Armed: the heap pointer is the program's, untouched.
    arm_freeze_on_quiesce(&mut win);
    assert_eq!(
        win[brk..brk + 8],
        heap,
        "arming leaves the heap pointer alone"
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

/// The root writes one byte to stdout and returns `1000 + written`.
const SRC_WRITES_ONCE: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vout: i32) {
  vbuf = i64.const 66100
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vout (vbuf, vlen)
  vk = i64.const 1000
  vr = i64.add vn vk
  return vr
  }
}
"#;

/// A job-control stop with no personality behind it. The first run to install its stop mirror here
/// keeps it, stopped at once when `at_start`; [`Stopper::stop`] and [`Stopper::cont`] drive it.
#[derive(Default)]
struct Stopper {
    at_start: bool,
    apply: std::sync::Mutex<Option<StopApply>>,
}

type StopApply = std::sync::Arc<dyn Fn(bool) + Send + Sync>;

impl Stopper {
    fn set(&self, stopped: bool) {
        let apply = self.apply.lock().unwrap().clone();
        apply.expect("installed by the run")(stopped);
    }
    fn stop(&self) {
        self.set(true);
    }
    fn cont(&self) {
        self.set(false);
    }
}

impl temen_interp::SignalSource for Stopper {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_stop_apply(&self, apply: StopApply) {
        let mut slot = self.apply.lock().unwrap();
        if slot.is_none() {
            if self.at_start {
                apply(true);
            }
            *slot = Some(apply);
        }
    }
}

/// **#1672 — a freeze sees through a job-control stop.** The domain is stopped from the start and
/// the freeze lands from the start. A stopped vCPU runs no ops, so it never reached its freeze point
/// and the freeze waited for a `SIGCONT` that might never come. A stop lands asynchronously, though,
/// so it may as well land at the next freeze point, provided nothing leaves the domain on the way:
/// the root runs on to its write, which is abandoned rather than performed (nothing reaches stdout
/// while it is stopped), and unwinds there. Continued and thawed, the root re-issues the write.
#[test]
fn a_stopped_domain_reaches_its_freeze_point_without_leaving_the_domain() {
    let inst = instrumented(SRC_WRITES_ONCE);
    let stopper = std::sync::Arc::new(Stopper {
        at_start: true,
        ..Default::default()
    });
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    h.set_signal_source(stopper.clone(), Default::default());
    let out = h.grant_stream(temen_interp::StreamRole::Out);
    let args = [Value::I32(out)];
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);

    let (tx, rx) = std::sync::mpsc::channel();
    let (inst2, win2) = (inst.clone(), win.clone());
    std::thread::spawn(move || {
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst2, 0, &args, &mut fuel, &win2, SIZE_LOG2, &mut h);
        let _ = tx.send((r, snap, h));
    });
    let (res, snap, mut h) = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the freeze completes while the domain is stopped");
    assert_eq!(
        res,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );
    assert!(
        h.stdout.is_empty(),
        "nothing left the domain while it was stopped"
    );

    stopper.cont();
    let mut win2 = snap;
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel = 1_000_000u64;
    let (res2, _) =
        run_capture_reserved_with_host(&inst, 0, &args, &mut fuel, &win2, SIZE_LOG2, &mut h);
    assert_eq!(
        res2,
        Ok(vec![Value::I64(1001)]),
        "the thaw re-issues the write"
    );
    assert_eq!(h.stdout, b"x");
}

/// The root parks a fiber in an infinite `atomic.wait`, stops its own domain through the host proc
/// `vstop`, then writes one byte to `vout` and returns `1000 + written`.
const SRC_STOPS_AFTER_A_FIBER_PARK: &str = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vstop: i32, vout: i32) {
  vf = ref.func 1
  vsp = i64.const 4096
  vk = cont.new vf vsp
  vz = i64.const 0
  vs, vx = cont.resume vk vz
  vq = call.cap 13 0 (i64) -> (i64) vstop (vz)
  vbuf = i64.const 66100
  vc = i32.const 120
  i32.store8 vbuf vc
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vout (vbuf, vlen)
  vt = i64.const 1000
  vr = i64.add vn vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;

/// **#1672 — a vCPU parked stopped before the freeze is brought through it.** The root parks a
/// fiber on a futex, then stops its own domain and parks stopped. Freeze-on-quiesce fires on the
/// fiber park, and the stopped root used to stay where it was: it reached no freeze point, so the run
/// never finished unwinding. Now the freeze re-admits it; it sees through its stop, abandons its
/// write (nothing reaches stdout), and unwinds. Continued and thawed, it re-issues the write.
#[test]
fn a_vcpu_stopped_before_the_freeze_is_brought_through_it() {
    let inst = instrumented(SRC_STOPS_AFTER_A_FIBER_PARK);
    let stopper = std::sync::Arc::new(Stopper::default());
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    h.set_signal_source(stopper.clone(), Default::default());
    let stop = {
        let stopper = stopper.clone();
        h.grant_host_proc(
            Box::new(move |_op, _args, _mem, _minter| {
                stopper.stop();
                Ok(vec![0])
            }),
            temen_interp::CapState::Stateless,
        )
    };
    let out = h.grant_stream(temen_interp::StreamRole::Out);
    let args = [Value::I32(stop), Value::I32(out)];
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_on_quiesce(&mut win);

    let (tx, rx) = std::sync::mpsc::channel();
    let (inst2, win2) = (inst.clone(), win.clone());
    std::thread::spawn(move || {
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst2, 0, &args, &mut fuel, &win2, SIZE_LOG2, &mut h);
        let _ = tx.send((r, snap, h));
    });
    let (res, snap, mut h) = rx
        .recv_timeout(Duration::from_secs(20))
        .expect("the freeze completes with the root parked stopped");
    assert_eq!(
        res,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );
    assert!(
        h.stdout.is_empty(),
        "nothing left the domain while it was stopped"
    );
    assert_eq!(h.frozen_fibers().len(), 1, "the fiber is in the cut");

    stopper.cont();
    let fibers = h.frozen_fibers().to_vec();
    h.set_frozen_fibers(fibers);
    let mut win2 = snap;
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel = 1_000_000u64;
    let (res2, _) =
        run_capture_reserved_with_host(&inst, 0, &args, &mut fuel, &win2, SIZE_LOG2, &mut h);
    assert_eq!(
        res2,
        Ok(vec![Value::I64(1001)]),
        "the thaw re-issues the write"
    );
    assert_eq!(h.stdout, b"x");
}

/// Run `inst` on `h` over `win` on a thread, failing (not hanging) after 20 s.
fn within_20s(
    inst: &Inst,
    args: &[Value],
    win: &[u8],
    mut h: Host,
) -> (Result<Vec<Value>, Trap>, Vec<u8>, Host) {
    let (tx, rx) = std::sync::mpsc::channel();
    let (inst, args, win) = (inst.clone(), args.to_vec(), win.to_vec());
    std::thread::spawn(move || {
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &args, &mut fuel, &win, SIZE_LOG2, &mut h);
        let _ = tx.send((r, snap, h));
    });
    rx.recv_timeout(Duration::from_secs(20))
        .expect("the run completes")
}

/// The root spawns a sibling, reads one byte from stdin, joins the sibling and returns
/// `1000·read + byte + joined`. The sibling loops, writes `y` to stdout and returns 7.
const SRC_STDIN_PARKED_ROOT: &str = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vin: i32, vout: i32) {
  vz = i64.const 0
  vo64 = i64.extend_i32_u vout
  vt = thread.spawn 1 vz vo64
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vlen)
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
  vy = i32.const 121
  i32.store8 vbuf vy
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vlen)
  vr = i64.const 7
  return vr
  }
}
"#;

/// A durable powerbox whose stdin blocks when empty (an interactive session), and the root's args:
/// its stdin and stdout handles.
fn stdin_host(inst: &Inst) -> (Host, [Value; 2]) {
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(inst);
    let vin = h.grant_stream(temen_interp::StreamRole::In);
    let vout = h.grant_stream(temen_interp::StreamRole::Out);
    h.set_stdin_blocking(true);
    (h, [Value::I32(vin), Value::I32(vout)])
}

/// **#1899 — a vCPU parked on a stream read is brought through a freeze, and its read re-issued.**
/// The root parks reading an empty stdin, then the freeze lands in its sibling. The stream read
/// was not rewound (its wake delivers the result), and no rule re-admitted it, so the freeze waited
/// for input that might never come. Now it is re-admitted with its read abandoned, and unwinds;
/// thawed with input waiting, it re-issues the read and gets the byte: `1000 + 'x' + 7`.
#[test]
fn a_stdin_parked_root_is_abandoned_and_its_read_reissued_on_thaw() {
    let inst = instrumented(SRC_STDIN_PARKED_ROOT);
    let (h, args) = stdin_host(&inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after_backedges(&mut win, 5);
    let (res, snap, mut h) = within_20s(&inst, &args, &win, h);
    assert_eq!(
        res,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );
    assert_eq!(h.frozen_vcpus().len(), 1, "the sibling is in the cut");

    h.push_stdin(b"x");
    let mut win2 = snap;
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let (res2, _, _) = within_20s(&inst, &args, &win2, h);
    assert_eq!(
        res2,
        Ok(vec![Value::I64(1000 + i64::from(b'x') + 7)]),
        "the thawed root re-issues its read and gets the byte"
    );
}

/// The same read reached under a landing freeze parks with the phase, and the freeze in flight
/// re-admits it the same way.
#[test]
fn a_stdin_read_under_a_landing_freeze_is_abandoned_and_reissued() {
    let inst = instrumented(SRC_STDIN_PARKED_ROOT);
    let (h, args) = stdin_host(&inst);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (res, snap, mut h) = within_20s(&inst, &args, &win, h);
    assert_eq!(
        res,
        Ok(vec![Value::I64(0)]),
        "the root unwinds for the freeze"
    );

    h.push_stdin(b"x");
    let mut win2 = snap;
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let (res2, _, _) = within_20s(&inst, &args, &win2, h);
    assert_eq!(res2, Ok(vec![Value::I64(1000 + i64::from(b'x') + 7)]));
}
