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

use temen_durable::{
    arm_freeze_on_quiesce, begin_thaw, init_durable_window, transform_module_assume_confined,
};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
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

    // **The thaw of this shape does not work yet (#1620), and that boundary is deliberate.**
    // A *spawned* vCPU frozen at an `atomic.wait` suspend point does not re-attach: the thaw
    // returns `ThreadFault`. Every existing multi-vCPU durable test freezes its child at a
    // `call.cap` instead (`temen-durable/tests/multivcpu.rs`), so the `MemoryWait` suspend kind
    // has never been exercised on a child — the veto fixed here is what kept this shape from ever
    // being frozen in the first place.
    //
    // Pinned as-is rather than left untested: this is strictly better than what it replaces. The
    // freeze above now takes a correct, consistent cut in ~0.1 s where it used to stall 10 s and
    // return `2000` as an ordinary result; and the thaw fails **closed and loudly**, which is a
    // value (#5), not a wrong answer. When #1620 lands, this flips to the `2100` round trip.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    let mut win2 = snap.clone();
    win2[66000..66004].copy_from_slice(&1i32.to_le_bytes());
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let mut fuel2 = 1_000_000u64;
    let (r2, _) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel2, &win2, SIZE_LOG2, &mut h2);
    assert!(
        r2.is_err(),
        "#1620: a spawned vCPU frozen at `atomic.wait` does not re-attach yet — but it must fail \
         closed rather than resume wrong. Flip this to `Ok([I64(2100)])` when #1620 lands: {r2:?}"
    );
}
