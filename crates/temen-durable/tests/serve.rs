//! DURABILITY.md §13.4 — freezing a serving domain. The serve trio
//! (`svc_queue`/`svc_results`/`svc_next_ticket`) is snapshot data (the codec's serve section,
//! pinned in `temen-snapshot`), and so, since #1676, is a handler the freeze catches mid-run:
//! it unwinds as a fiber (`FIBER_FROZEN`, never a bogus `(FIBER_RETURNED, 0)` reply), its reply
//! linkage rides the powerbox, and the thaw's serve loop resumes it to reply for real.

use temen_durable::{arm_freeze_after, init_durable_window, transform_module_assume_confined};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::Memory;

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

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
fn a_mid_handler_freeze_thaws_and_the_handler_replies() {
    use temen_durable::{begin_thaw, read_state, STATE_UNWINDING};

    let mut m = temen_text::parse_module(SRC_SERVING_HANDLER_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
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
            &init_durable_window(WINDOW, TEST_ARENA),
            SIZE_LOG2,
            &mut h,
        );
        assert_eq!(r, Ok(vec![Value::I64(1)]), "one dispatch served");
        assert_eq!(h.svc_result(t), Some(42), "the handler's real reply");
    }

    // Armed at the first fiber safepoint — the handler's own `cont.resume` — so the freeze
    // lands mid-handler: the handler unwinds as a fiber and nothing is settled; its ticket rides.
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(&inst);
    let t = h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, 1);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "the freeze returns a placeholder: {r:?}");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "the cut was taken");
    assert_eq!(h.take_freeze_declined(), None, "not declined");
    assert_eq!(h.svc_result(t), None, "no bogus reply settled");
    assert_eq!(
        h.frozen_handlers(),
        vec![(0, t)],
        "the handler's ticket rides"
    );

    // Thaw: the serve loop adopts the handler, which rewinds, finishes and replies.
    let fibers = h.frozen_fibers().to_vec();
    h.set_frozen_fibers(fibers);
    let mut twin = snap;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &twin, SIZE_LOG2, &mut h);
    assert_eq!(r, Ok(vec![Value::I64(1)]), "one dispatch served");
    assert_eq!(h.svc_result(t), Some(42), "the handler's real reply");
    assert!(h.frozen_handlers().is_empty(), "the serve loop took it");
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
        shadow: Some(TEST_ARENA),
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
            &init_durable_window(WINDOW, TEST_ARENA),
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
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
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
        begin_thaw(&mut win, TEST_ARENA, 0);
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
        shadow: Some(TEST_ARENA),
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");

    // Freeze: the server parks in svc.wait with an empty queue; freeze-on-quiesce fires the
    // instant the run would block, so the domain unwinds instead of hanging.
    let (trio, snap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
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
        begin_thaw(&mut win, TEST_ARENA, 0);
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

/// #1676 — a serving domain whose handler is **parked**: `hold` (op 0) waits on a word only `poke`
/// (op 1) sets, and the server loops `svc.wait` until two dispatches are served. With only `hold`
/// queued, the handler parks, the serve loop idles, and freeze-on-quiesce fires: the handler rides
/// as a fiber with its futex wait abandoned, and its ticket rides the powerbox. `poke` arrives after
/// the freeze; on thaw the serve loop resumes the handler, which re-issues its wait, `poke` wakes it,
/// and both reply: `hold` 100, `poke` 7.
const SRC_PARKED_HANDLER: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { hold: 0, poke: 0 }
export 0 interface "latch" 1 { hold: 1, poke: 2 }

func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vacc: i64) {
  vs = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vs ()
  vacc2 = i64.add vacc vn
  vtwo = i64.const 2
  vmore = i64.lt_s vacc2 vtwo
  br_if vmore 1(vacc2) 2(vacc2)
}
block 2 (vr: i64) {
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vr = i64.const 100
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  vaddr = i64.const 66000
  vone = i32.const 1
  i32.atomic.store vaddr vone
  vn = atomic.notify vaddr vone
  vr = i64.const 7
  return vr
  }
}
"#;

#[test]
fn a_parked_handler_freezes_and_its_thaw_replies() {
    use temen_durable::{arm_freeze_on_quiesce, begin_thaw, read_state, STATE_UNWINDING};

    let mut m = temen_text::parse_module(SRC_PARKED_HANDLER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = std::sync::Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");
    let host = || {
        let mut h = Host::new();
        h.set_durable(true);
        h.set_self_module(&inst);
        h
    };
    let run = |h: &mut Host, win: &[u8]| {
        let mut fuel = 1_000_000u64;
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, win, SIZE_LOG2, h)
    };

    // Uninterrupted: both queued; `hold` parks, `poke` wakes it, both reply.
    let mut h = host();
    let t_hold = h.svc_enqueue(0, 0, vec![0]).expect("enqueue");
    let t_poke = h.svc_enqueue(0, 1, vec![0]).expect("enqueue");
    let (r, _) = run(&mut h, &init_durable_window(WINDOW, TEST_ARENA));
    assert_eq!(r, Ok(vec![Value::I64(2)]), "two served");
    assert_eq!(
        (h.svc_result(t_hold), h.svc_result(t_poke)),
        (Some(100), Some(7))
    );

    // Frozen with `hold` parked.
    let mut h = host();
    let t_hold = h.svc_enqueue(0, 0, vec![0]).expect("enqueue");
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_on_quiesce(&mut win);
    let (r, snap) = run(&mut h, &win);
    assert!(r.is_ok(), "the freeze returns a placeholder: {r:?}");
    assert_eq!(read_state(&snap), STATE_UNWINDING, "the cut was taken");
    assert_eq!(h.take_freeze_declined(), None, "not declined");
    assert_eq!(
        h.frozen_handlers(),
        vec![(0, t_hold)],
        "the parked handler's ticket rides"
    );

    // `poke` arrives; the thaw resumes `hold`, and `poke` wakes it.
    let t_poke = h.svc_enqueue(0, 1, vec![0]).expect("enqueue");
    let fibers = h.frozen_fibers().to_vec();
    h.set_frozen_fibers(fibers);
    let mut twin = snap;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (r, _) = run(&mut h, &twin);
    assert_eq!(r, Ok(vec![Value::I64(2)]), "two served across the cut");
    assert_eq!(
        (h.svc_result(t_hold), h.svc_result(t_poke)),
        (Some(100), Some(7))
    );
}
