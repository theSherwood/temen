//! DURABILITY.md §13.4 step 3 — the serve-side freeze gates. The serve trio
//! (`svc_queue`/`svc_results`/`svc_next_ticket`) is snapshot data now (the codec's v13
//! serve section, pinned in `temen-snapshot`), but a freeze that lands **mid-handler** still
//! fails closed: under `UNWINDING` a handler's exit is an unwind return, whose
//! `(FIBER_RETURNED, 0)` would settle a bogus zero into the caller's completion cell — and
//! even a genuine mid-freeze return's reply linkage (`serve_run`) is not yet in the
//! snapshot (the step-4 record). The serve epilogue refuses the freeze instead
//! (`FiberFault`, the `handler_parks` gate's shape); the previous snapshot stays the
//! recovery point.

use temen_durable::{arm_freeze_after, init_durable_window, transform_module_assume_confined};
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::Memory;

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A durable serving domain whose HANDLER crosses a fiber safepoint: offer "counter" op 0 =
/// func 1, which spins a trivial sub-fiber (func 2) — the `cont.resume` is where an armed
/// countdown promotes to `UNWINDING`, mid-handler — then returns `x + 1`. The root just
/// `svc.poll`s and returns the served count.
const SRC_SERVING_HANDLER_FIBER: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  return vn
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  vf = ref.func 2
  vsp = i64.const 4096
  vk = cont.new vf vsp
  vs, vv = cont.resume vk vx
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  return vb
  }
}
"#;

#[test]
fn a_mid_handler_freeze_fails_closed_instead_of_settling_a_bogus_reply() {
    let mut m = temen_text::parse_module(SRC_SERVING_HANDLER_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Baseline (durable, un-armed = NORMAL): the dispatch serves through the fiber-spinning
    // handler and completes with x + 1.
    {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let t = h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        assert_eq!(r, Ok(vec![Value::I64(1)]), "one dispatch served");
        assert_eq!(h.svc_result(t), Some(42), "the handler's real reply");
    }

    // Armed at the first fiber safepoint — the handler's own `cont.resume` — so the freeze
    // lands mid-handler: the serve epilogue refuses it rather than settling the handler's
    // unwind-zero as a reply.
    {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
        let mut win = init_durable_window(WINDOW);
        arm_freeze_after(&mut win, 1);
        let mut fuel = 1_000_000u64;
        let (r, _) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert_eq!(
            r,
            Err(Trap::FiberFault),
            "a mid-handler freeze refuses fail-closed (no bogus zero reply)"
        );
    }
}

/// The transform's local serve-op numbers must match the interp's reserved self-namespace ops
/// (`temen-durable` depends only on `temen-ir`, so it carries copies).
#[test]
fn the_serve_op_numbers_match_the_interp() {
    assert_eq!(temen_durable::SVC_POLL_OP, temen_interp::CAP_SELF_SVC_POLL);
    assert_eq!(temen_durable::SVC_WAIT_OP, temen_interp::CAP_SELF_SVC_WAIT);
}

/// §13.4 slice 4b — a serving domain frozen AT its serve point thaws and drains the
/// **restored** queue. The freeze run reaches `svc.poll` under `UNWINDING`: the serve arm
/// delivers an inert sentinel (no drain — the queue survives untouched for the snapshot's
/// serve section) and the trailing poll spills. On thaw the `SvcServe` re-issue arm
/// re-executes the op against the restored trio: the handler finally runs, its effect lands
/// in the restored window, and the completion cell fills — identical to the uninterrupted
/// run (re-execution is the recovery).
const SRC_SERVING_POLL: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  vc = i64.const 65600
  vafter = i64.load vc
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 65600
  i64.store vc vx
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}
"#;

#[test]
fn a_domain_frozen_at_its_serve_point_thaws_and_drains_the_restored_queue() {
    use temen_durable::{begin_thaw, write_state, STATE_UNWINDING};

    let mut m = temen_text::parse_module(SRC_SERVING_POLL).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Baseline (durable, NORMAL): the queued bump(41) serves — count 1, cell 41 → 1041.
    {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &init_durable_window(WINDOW),
            SIZE_LOG2,
            &mut h,
        );
        assert_eq!(r, Ok(vec![Value::I64(1041)]), "served*1000 + cell");
    }

    // Freeze at the serve point: the run unwinds with the queue untouched.
    let (trio, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let t = h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
        assert_eq!(t, 0);
        let mut win = init_durable_window(WINDOW);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(r.is_ok(), "freeze returns a placeholder: {r:?}");
        let trio = h.svc_state();
        assert_eq!(trio.0.len(), 1, "the queue survives the freeze untouched");
        (trio, h.frozen_root_sp().expect("root extent"), snap)
    };

    // Thaw into a fresh host carrying the restored trio: the re-issued serve op drains it.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    let (q, res, next) = trio;
    h2.set_svc_state(q, res, next);
    h2.set_frozen_root_sp(root_sp);
    let r_thaw = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut fuel = 1_000_000u64;
        let (r, _) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h2);
        r
    };
    assert_eq!(
        r_thaw,
        Ok(vec![Value::I64(1041)]),
        "the thawed drain serves the restored dispatch identically"
    );
    assert_eq!(
        h2.svc_result(0),
        Some(42),
        "the completion cell filled on thaw"
    );
}

/// §13.4 slice 4c-bis — the **flagship**: freeze a server **idle in its `svc.wait` accept
/// loop** and thaw it still serving. The server parks in `svc.wait` (empty queue, no timeout —
/// forever); no safepoint or back-edge countdown can reach a parked vCPU, so `arm_freeze_on_
/// quiesce` triggers the freeze the instant the run would otherwise block on that park: the
/// scheduler promotes the parked window to `UNWINDING` and re-admits it, the re-executed
/// `svc.wait` takes the 4b sentinel, and the domain unwinds — capturing its (empty) serve trio
/// in the v13 section. On thaw the rewind re-issues `svc.wait` against the **restored** queue;
/// a dispatch enqueued before thaw is served (handler writes 41 to the cell, count 1) and the
/// server returns 1*1000 + 41.
const SRC_IDLE_SERVER: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  vc = i64.const 65600
  vafter = i64.load vc
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 65600
  i64.store vc vx
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}
"#;

#[test]
fn an_idle_server_freezes_on_quiesce_and_thaws_still_serving() {
    use temen_durable::{arm_freeze_on_quiesce, begin_thaw};

    let mut m = temen_text::parse_module(SRC_IDLE_SERVER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Freeze: the server parks in svc.wait with an empty queue; freeze-on-quiesce fires the
    // instant the run would block, so the domain unwinds instead of hanging.
    let (trio, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let mut win = init_durable_window(WINDOW);
        arm_freeze_on_quiesce(&mut win);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(r.is_ok(), "the idle server unwinds on quiesce: {r:?}");
        let trio = h.svc_state();
        assert!(
            trio.0.is_empty(),
            "the accept loop froze with an empty queue"
        );
        (trio, snap)
    };

    // Thaw: seed a dispatch into the restored queue, then thaw — the re-issued svc.wait drains
    // and serves it.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    let (q, res, next) = trio;
    h2.set_svc_state(q, res, next);
    let ticket = h2
        .svc_enqueue(0, 0, vec![41])
        .expect("enqueue into restored server");
    let r_thaw = {
        let mut win = snap.clone();
        begin_thaw(&mut win, 0);
        let mut fuel = 1_000_000u64;
        let (r, _) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h2);
        r
    };
    assert_eq!(
        r_thaw,
        Ok(vec![Value::I64(1041)]),
        "the thawed server drains and serves the restored dispatch (served*1000 + cell)"
    );
    assert_eq!(h2.svc_result(ticket), Some(42), "the handler's reply");
}

/// §13.4 4c-bis, the **nested** variant: a two-server subtree — root server + an instantiated
/// child server, both idle in `svc.wait` — freezes on quiesce and thaws back into the same
/// serving state. Freeze-on-quiesce drains both parked domains in `domain_id` order (root
/// first), so the root freezes first and records the still-running child as a re-attach
/// `FrozenNested` (`completed=None`), then the child freezes recording its `FrozenChildState`
/// (serve trio + grant handles) — the two keyed `(parent_task, slot)`-consistent. The child's
/// carve carries its own `UNWINDING` (its `dstate` routed to the carve's state word), not the
/// global one. On thaw the runtime re-creates the child, seeds its fresh host from the
/// residue, and rewinds it to its `svc.wait` re-issue arm; re-arming freeze-on-quiesce proves
/// the round trip — the thawed subtree re-parks both servers and **re-freezes**, recording the
/// child's residue a second time (a serving subtree came back serving).
const SRC_NESTED_TWO_SERVER: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=131072 sl=17 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 131072
  q0v2 = i64.const -4294967279
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 17536
  i64.store q0a0 q0v0
  q0a1 = i64.const 17544
  i64.store q0a1 q0v1
  q0a2 = i64.const 17552
  i64.store q0a2 q0v2
  q0a3 = i64.const 17560
  i64.store q0a3 q0v3
  q0a4 = i64.const 17568
  i64.store q0a4 q0v4
  q0a5 = i64.const 17576
  i64.store q0a5 q0v4
  q0a6 = i64.const 17584
  i64.store q0a6 q0v4
  v5 = call.cap 6 17 (i64) -> (i32) v0 (q0a0)
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

#[test]
fn a_nested_two_server_subtree_freezes_on_quiesce_and_thaws_still_serving() {
    use temen_durable::{arm_freeze_on_quiesce, begin_thaw};

    const P_LOG2: u8 = 18;
    const P_WINDOW: usize = 1 << P_LOG2;

    let mut m = temen_text::parse_module(SRC_NESTED_TWO_SERVER).expect("parse");
    m.memory = Some(Memory { size_log2: P_LOG2 });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Freeze the idle two-server subtree.
    let (nested, child_state, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let ih = h.grant_instantiator(0, P_WINDOW as u64);
        let mut win = init_durable_window(P_WINDOW);
        arm_freeze_on_quiesce(&mut win);
        let mut fuel = 5_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(ih)],
            &mut fuel,
            &win,
            P_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "the idle subtree freezes on quiesce: {r:?}");
        assert_eq!(
            h.frozen_nested().len(),
            1,
            "the child recorded as re-attach"
        );
        assert_eq!(
            h.frozen_nested()[0].completed_result,
            None,
            "still-running, not reloaded"
        );
        assert_eq!(
            h.frozen_child_state().len(),
            1,
            "the child's serve state was captured"
        );
        (
            h.frozen_nested().to_vec(),
            h.frozen_child_state().to_vec(),
            h.frozen_root_sp().expect("root extent"),
            snap,
        )
    };

    // Thaw with freeze-on-quiesce re-armed: the subtree re-parks both servers and re-freezes,
    // recording the child's residue again — a serving subtree came back serving.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    h2.set_frozen_nested(nested);
    h2.set_frozen_child_state(child_state);
    h2.set_frozen_root_sp(root_sp);
    let _ih2 = h2.grant_instantiator(0, P_WINDOW as u64);
    let mut win = snap.clone();
    begin_thaw(&mut win, 0);
    arm_freeze_on_quiesce(&mut win);
    let mut fuel = 5_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(_ih2)],
        &mut fuel,
        &win,
        P_LOG2,
        &mut h2,
    );
    assert!(r.is_ok(), "the thawed subtree re-freezes on quiesce: {r:?}");
    assert_eq!(
        h2.frozen_child_state().len(),
        1,
        "the thawed child came back a serving domain (re-froze with its serve state)"
    );
    assert_eq!(
        h2.frozen_nested()[0].completed_result,
        None,
        "the re-frozen child is still a live server, not a completed result"
    );
}

/// §13.4 slice 4c — **depth-2 (grandchild) serve capture**. A live-spawned nested child never
/// stamped its own `parent_task`, so a *grandchild* (C2, spawned by a child C1) defaulted to `0`
/// (the root). Its `FrozenChildState` was then keyed `(0, slot)` while its `FrozenNested` — which
/// the *recording parent* C1 keys with its own id — was `(C1, slot)`; on thaw the two failed to
/// match, so C2 fell to the fresh-grant path **without its serve module** and could not dispatch.
/// Fixed by stamping the spawning vCPU's id at the op-0 spawn (the intent the `parent_task`
/// destructure comment already stated). This freezes a three-level nested-server subtree, thaws it,
/// and re-freezes — the grandchild's serve state must be attributed to C1, not the root, at every
/// step.
///
/// (Child entries take the `(i64)` starter arg the spawn-ABI enforces; C1 stores it to scratch and
/// reloads the low word with `i32.load` to get the `i32` instantiator handle for spawning C2,
/// avoiding `i32.wrap_i64` which the durable transform does not type. Scratch sits above the 64 KiB
/// `DURABLE_RESERVE` and below C2's sub-carve.)
const SRC_NESTED_SERVERS_3: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { call: 0 }
export 0 interface "svc" 1 { call: 2 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=262144 sl=18 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 262144
  q1v2 = i64.const -4294967278
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 17600
  i64.store q1a0 q1v0
  q1a1 = i64.const 17608
  i64.store q1a1 q1v1
  q1a2 = i64.const 17616
  i64.store q1a2 q1v2
  q1a3 = i64.const 17624
  i64.store q1a3 q1v3
  q1a4 = i64.const 17632
  i64.store q1a4 q1v4
  q1a5 = i64.const 17640
  i64.store q1a5 q1v4
  q1a6 = i64.const 17648
  i64.store q1a6 q1v4
  vc1 = call.cap 6 17 (i64) -> (i32) v0 (q1a0)
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 65600
  i64.store vk v0
  vh = i32.load vk
  ; spawn via record (op 17): entry=2 off=131072 sl=17 quota=0
  q2v0 = i64.const 8589934592
  q2v1 = i64.const 131072
  q2v2 = i64.const -4294967279
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2a0 = i64.const 17664
  i64.store q2a0 q2v0
  q2a1 = i64.const 17672
  i64.store q2a1 q2v1
  q2a2 = i64.const 17680
  i64.store q2a2 q2v2
  q2a3 = i64.const 17688
  i64.store q2a3 q2v3
  q2a4 = i64.const 17696
  i64.store q2a4 q2v4
  q2a5 = i64.const 17704
  i64.store q2a5 q2v4
  q2a6 = i64.const 17712
  i64.store q2a6 q2v4
  vc2 = call.cap 6 17 (i64) -> (i32) vh (q2a0)
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

#[test]
fn a_three_level_nested_server_subtree_keys_the_grandchild_serve_state_to_its_real_parent() {
    use temen_durable::{arm_freeze_on_quiesce, begin_thaw};

    const P_LOG2: u8 = 19;
    const P_WINDOW: usize = 1 << P_LOG2;

    let mut m = temen_text::parse_module(SRC_NESTED_SERVERS_3).expect("parse");
    m.memory = Some(Memory { size_log2: P_LOG2 });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Freeze the idle three-level subtree (root -> C1 -> C2, each parked in svc.wait).
    let (nested, child_state, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let ih = h.grant_instantiator(0, P_WINDOW as u64);
        let mut win = init_durable_window(P_WINDOW);
        arm_freeze_on_quiesce(&mut win);
        let mut fuel = 20_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(ih)],
            &mut fuel,
            &win,
            P_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "the idle three-level subtree freezes: {r:?}");
        assert_eq!(h.frozen_nested().len(), 2, "C1 and C2 both recorded nested");
        assert_eq!(
            h.frozen_child_state().len(),
            2,
            "C1 and C2 serve state captured"
        );
        // The grandchild (C2) must be attributed to C1, not the root: the two captured child
        // states carry **distinct** parent tasks, and each matches a nested record's parent.
        // Before the fix both were `0` (the root) and C2's state was orphaned.
        let cs_parents: std::collections::BTreeSet<usize> = h
            .frozen_child_state()
            .iter()
            .map(|cs| cs.parent_task)
            .collect();
        assert_eq!(
            cs_parents.len(),
            2,
            "the grandchild's serve state is keyed to C1, distinct from C1's (keyed to root)"
        );
        for cs in h.frozen_child_state() {
            assert!(
                h.frozen_nested()
                    .iter()
                    .any(|n| n.parent_task == cs.parent_task && n.slot == cs.slot),
                "each child state matches a nested record's (parent_task, slot) key"
            );
        }
        (
            h.frozen_nested().to_vec(),
            h.frozen_child_state().to_vec(),
            h.frozen_root_sp().expect("root extent"),
            snap,
        )
    };

    // Thaw with freeze-on-quiesce re-armed: the subtree re-creates C1 + C2 (C2 matched to its
    // captured serve state, so it restores *with* its serve module), re-parks all three servers,
    // and re-freezes — recording the grandchild's serve state under C1 a second time.
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    h2.set_frozen_nested(nested);
    h2.set_frozen_child_state(child_state);
    h2.set_frozen_root_sp(root_sp);
    let ih2 = h2.grant_instantiator(0, P_WINDOW as u64);
    let mut win = snap.clone();
    begin_thaw(&mut win, 0);
    arm_freeze_on_quiesce(&mut win);
    let mut fuel = 20_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(ih2)],
        &mut fuel,
        &win,
        P_LOG2,
        &mut h2,
    );
    assert!(
        r.is_ok(),
        "the thawed three-level subtree re-freezes on quiesce: {r:?}"
    );
    assert_eq!(
        h2.frozen_child_state().len(),
        2,
        "the thawed subtree came back a live three-level server tree (re-captured both states)"
    );
    let cs_parents2: std::collections::BTreeSet<usize> = h2
        .frozen_child_state()
        .iter()
        .map(|cs| cs.parent_task)
        .collect();
    assert_eq!(
        cs_parents2.len(),
        2,
        "the re-frozen grandchild is still attributed to C1, not the root"
    );
}

/// §13.4 slice 4d follow-up — the **nested holder** end to end. A *child* C1 holds a live
/// `child_offer` cap onto a *grandchild* C2 (not the root onto a direct child). All three idle in
/// `svc.wait` at freeze: root holds a cap onto C1's **fwd** export (export 1 → func 3); C1 holds a
/// cap onto C2's **leaf** export (export 0 → func 1), stashed at `mem[65608]`. Freeze-on-quiesce
/// captures the subtree; C1's `LiveImpl` onto C2 rides its `FrozenChildState` as a durable handle.
/// On thaw the runtime re-creates C1 + C2 and re-links **C1's** placeholder to the re-created C2 by
/// the `(C1 task, C2 slot)` edge — the generalization the root-only re-link did not cover. A
/// dispatch seeded into the **root's** queue then drives `fwd(7)` through the root's cap onto C1;
/// C1's handler loads its re-linked cap and forwards `leaf(7)` to C2, which returns `7 + 100`. The
/// chained reply `107` is observable only if the nested re-link succeeded and the serve chain (I49)
/// threads the reply back C2 → C1's handler → root.
///
/// Fixture notes: child entries take the `(i64)` starter arg the spawn-ABI enforces; C1 stores it
/// to scratch and reloads the low word with `i32.load` for the `i32` instantiator handle (the
/// durable transform types loads/stores but not width conversions). Scratch sits above the 64 KiB
/// `DURABLE_RESERVE`, below each child's own sub-carve.
const SRC_NESTED_HOLDER: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { call: 0 }
export 0 interface "leaf" 1 { call: 1 }
export 1 interface "fwd" 1 { call: 3 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=2 off=262144 sl=18 quota=0
  q3v0 = i64.const 8589934592
  q3v1 = i64.const 262144
  q3v2 = i64.const -4294967278
  q3v3 = i64.const 4294967295
  q3v4 = i64.const 0
  q3a0 = i64.const 17728
  i64.store q3a0 q3v0
  q3a1 = i64.const 17736
  i64.store q3a1 q3v1
  q3a2 = i64.const 17744
  i64.store q3a2 q3v2
  q3a3 = i64.const 17752
  i64.store q3a3 q3v3
  q3a4 = i64.const 17760
  i64.store q3a4 q3v4
  q3a5 = i64.const 17768
  i64.store q3a5 q3v4
  q3a6 = i64.const 17776
  i64.store q3a6 q3v4
  vc1 = call.cap 6 17 (i64) -> (i32) v0 (q3a0)
  vexp = i64.const 1
  vh1 = call.cap 6 14 (i32, i64) -> (i32) v0 (vc1, vexp)
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  varg = i64.const 7
  vr = call.cap 268435456 0 (i64) -> (i64) vh1 (varg)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v1 = i64.const 100
  vr = i64.add vx v1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 65600
  i64.store vk v0
  vh = i32.load vk
  ; spawn via record (op 17): entry=4 off=131072 sl=17 quota=0
  q4v0 = i64.const 17179869184
  q4v1 = i64.const 131072
  q4v2 = i64.const -4294967279
  q4v3 = i64.const 4294967295
  q4v4 = i64.const 0
  q4a0 = i64.const 17792
  i64.store q4a0 q4v0
  q4a1 = i64.const 17800
  i64.store q4a1 q4v1
  q4a2 = i64.const 17808
  i64.store q4a2 q4v2
  q4a3 = i64.const 17816
  i64.store q4a3 q4v3
  q4a4 = i64.const 17824
  i64.store q4a4 q4v4
  q4a5 = i64.const 17832
  i64.store q4a5 q4v4
  q4a6 = i64.const 17840
  i64.store q4a6 q4v4
  vc2 = call.cap 6 17 (i64) -> (i32) vh (q4a0)
  vexp = i64.const 0
  vh2 = call.cap 6 14 (i32, i64) -> (i32) vh (vc2, vexp)
  vk2 = i64.const 65608
  i32.store vk2 vh2
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk2 = i64.const 65608
  vh2 = i32.load vk2
  vr = call.cap 268435456 0 (i64) -> (i64) vh2 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

#[test]
fn a_nested_holder_freezes_and_thaws_with_the_grandchild_cap_relinked() {
    use temen_durable::{arm_freeze_on_quiesce, begin_thaw};

    const P_LOG2: u8 = 19;
    const P_WINDOW: usize = 1 << P_LOG2;

    let mut m = temen_text::parse_module(SRC_NESTED_HOLDER).expect("parse");
    m.memory = Some(Memory { size_log2: P_LOG2 });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Freeze the idle three-level subtree (root holds a cap onto C1; C1 holds a cap onto C2).
    let (nested, child_state, root_handles, root_sp, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let ih = h.grant_instantiator(0, P_WINDOW as u64);
        let mut win = init_durable_window(P_WINDOW);
        arm_freeze_on_quiesce(&mut win);
        let mut fuel = 20_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(ih)],
            &mut fuel,
            &win,
            P_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "the idle nested-holder subtree freezes: {r:?}");
        assert_eq!(h.frozen_nested().len(), 2, "C1 and C2 both recorded nested");
        assert_eq!(
            h.frozen_child_state().len(),
            2,
            "both C1 and C2 captured serve state"
        );
        (
            h.frozen_nested().to_vec(),
            h.frozen_child_state().to_vec(),
            // The root's own durable handle table (its LiveImpl onto C1) — the snapshot codec
            // carries this in the artifact; the manual harness threads it explicitly.
            h.capture_durable_handles()
                .expect("root holds only durable handles"),
            h.frozen_root_sp().expect("root extent"),
            snap,
        )
    };

    // Thaw: re-create C1 + C2, re-link both edges (root→C1 direct, C1→C2 nested), then seed a
    // dispatch into the root's queue so it wakes and drives the forward chain fwd(7) → leaf(7).
    let mut h2 = Host::new();
    h2.set_durable(true);
    h2.set_self_module(&inst);
    h2.set_frozen_nested(nested);
    h2.set_frozen_child_state(child_state);
    h2.set_frozen_root_sp(root_sp);
    h2.restore_durable_handles(&root_handles);
    let ih2 = h2.grant_instantiator(0, P_WINDOW as u64);
    h2.svc_enqueue(0, 0, vec![0])
        .expect("seed a dispatch so the root's svc.wait returns");
    let mut win = snap.clone();
    begin_thaw(&mut win, 0);
    let mut fuel = 20_000_000u64;
    let (thawed, _) = run_capture_reserved_with_host(
        &inst,
        0,
        &[Value::I32(ih2)],
        &mut fuel,
        &win,
        P_LOG2,
        &mut h2,
    );
    assert_eq!(
        thawed,
        Ok(vec![Value::I64(107)]),
        "the thawed root drove fwd(7) → C1 forwarded leaf(7) through the re-linked grandchild cap → 107"
    );
}
