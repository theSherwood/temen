//! DURABILITY.md §13.4 step 3 — the serve-side freeze gates. The serve trio
//! (`svc_queue`/`svc_results`/`svc_next_ticket`) is snapshot data now (the codec's v13
//! serve section, pinned in `svm-snapshot`), but a freeze that lands **mid-handler** still
//! fails closed: under `UNWINDING` a handler's exit is an unwind return, whose
//! `(FIBER_RETURNED, 0)` would settle a bogus zero into the caller's completion cell — and
//! even a genuine mid-freeze return's reply linkage (`serve_run`) is not yet in the
//! snapshot (the step-4 record). The serve epilogue refuses the freeze instead
//! (`FiberFault`, the `handler_parks` gate's shape); the previous snapshot stays the
//! recovery point.

use svm_durable::{arm_freeze_after, init_durable_window, transform_module_assume_confined};
use svm_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use svm_ir::Memory;

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
  vn = cap.call 4294967295 9 () -> (i64) vz ()
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
    let mut m = svm_text::parse_module(SRC_SERVING_HANDLER_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    svm_verify::verify_module(&inst).expect("verify");

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
/// (`svm-durable` depends only on `svm-ir`, so it carries copies).
#[test]
fn the_serve_op_numbers_match_the_interp() {
    assert_eq!(svm_durable::SVC_POLL_OP, svm_interp::CAP_SELF_SVC_POLL);
    assert_eq!(svm_durable::SVC_WAIT_OP, svm_interp::CAP_SELF_SVC_WAIT);
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
  vn = cap.call 4294967295 9 () -> (i64) vz ()
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
    use svm_durable::{begin_thaw, write_state, STATE_UNWINDING};

    let mut m = svm_text::parse_module(SRC_SERVING_POLL).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    svm_verify::verify_module(&inst).expect("verify");

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
  vn = cap.call 4294967295 10 () -> (i64) vz ()
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
    use svm_durable::{arm_freeze_on_quiesce, begin_thaw};

    let mut m = svm_text::parse_module(SRC_IDLE_SERVER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    svm_verify::verify_module(&inst).expect("verify");

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
