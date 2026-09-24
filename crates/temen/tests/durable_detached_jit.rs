//! #1361 step 4 — a durable parent frozen on the **JIT** while its detached child (op 15) is live,
//! through the embedder's path (`temen_run::jit_cap_run`) and the §12 codec: the freeze rings the
//! child's own freeze word, the child unwinds in its own window, and the harvest puts its window and
//! powerbox on the `Host` as a `CapturedDetached` — the interpreter's form, so one artifact serves
//! both engines. The thaw re-launches the child at its join slot under `REWINDING`, on the JIT and on
//! the interpreter alike, and the parent's join delivers the uninterrupted total.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, write_state, STATE_NORMAL,
    STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, FreezeScope, Host, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_jit::{JitError, JitOutcome};

const ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};
const PARENT_LOG2: u8 = 18;

/// Spawn the child detached (op 15, 7-arg form: budget, module, no grants, entry 0, `size_log2` 17,
/// no quota), join it, return what it returns.
const PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vc)
  return vr
  }
}
";

/// Sums `0..100` with a back-edge poll per iteration — 4950 uninterrupted.
const CHILD: &str = "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = i64.const 0
  br 1(v1, v2)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 100
  v6 = i64.lt_s v3 v5
  br_if v6 2(v3, v4) 3(v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.add v8 v7
  v10 = i64.const 1
  v11 = i64.add v7 v10
  br 1(v11, v9)
}
block 3 (v12: i64) {
  return v12
  }
}
";

fn instrument(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// A durable powerbox granting what op 15 needs — freeze authority over detached progeny included —
/// and the handle values as JIT entry slots.
fn powerbox(child: &temen_ir::Module) -> (Host, Vec<i64>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    host.grant_freeze_authority(FreezeScope::DetachedProgeny);
    (host, vec![inst as i64, modh as i64, budget as i64])
}

fn returned(o: JitOutcome) -> Vec<i64> {
    match o {
        JitOutcome::Returned(v) => v,
        other => panic!("the run did not return: {other:?}"),
    }
}

#[test]
fn a_live_detached_child_freezes_and_thaws_on_the_jit_through_the_codec() {
    let parent = instrument(PARENT);
    let child = instrument(CHILD);

    // Control: uninterrupted on the JIT.
    let (mut host, args) = powerbox(&child);
    let base = match temen_run::jit_cap_run(
        &parent,
        0,
        &args,
        &init_durable_window(1 << PARENT_LOG2, ARENA),
        PARENT_LOG2,
        0,
        &mut host,
    ) {
        Ok((o, _)) => returned(o),
        Err(JitError::Unsupported(_)) => return, // a target without the child executor
        Err(e) => panic!("JIT run failed: {e:?}"),
    };
    assert_eq!(base, vec![4950], "uninterrupted total");

    // Freeze from the start: the parent spawns the child while already unwinding, so the child starts
    // with its own freeze word set (#1760) and unwinds at its first poll, however its thread is
    // scheduled; the harvest carries its window + powerbox onto the Host.
    let (mut fhost, fargs) = powerbox(&child);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (_, fsnap) = temen_run::jit_cap_run(&parent, 0, &fargs, &win, PARENT_LOG2, 0, &mut fhost)
        .expect("JIT freeze");
    assert_eq!(read_state(&fsnap), STATE_UNWINDING, "the parent froze");
    assert!(
        fhost.unreached_detached().is_empty(),
        "the child reached its poll"
    );
    assert_eq!(
        fhost.captured_detached().len(),
        1,
        "the live child was captured"
    );
    assert_eq!(fhost.captured_detached()[0].slot, 0);
    assert_eq!(
        read_state(fhost.captured_detached()[0].window.bytes()),
        STATE_UNWINDING,
        "the doorbell set the child's own freeze word, and it unwound"
    );

    // Through the codec, into a fresh host that re-grants the child's program (D-scope).
    let art = temen_snapshot::freeze(&parent, &fsnap, &fhost).expect("serialize");
    let restore = || {
        let mut h = Host::new();
        h.set_durable(true);
        h.grant_durable_module(&child);
        let w = temen_snapshot::restore(&art, &parent, &mut h).expect("restore");
        let mut w = w;
        begin_thaw(&mut w, ARENA, 0);
        (h, w)
    };

    // A thawing host that no longer grants the child's program refuses the JIT thaw whole, and the
    // detached residue stays on it for a host that does.
    let (mut granted, twin) = restore();
    let mut bare = Host::new();
    bare.set_durable(true);
    bare.set_thawed_detached(granted.take_thawed_detached());
    let refused = temen_run::jit_cap_run(&parent, 0, &fargs, &twin, PARENT_LOG2, 0, &mut bare);
    assert!(
        matches!(refused, Err(JitError::Unsupported(_))),
        "an ungranted child program refuses the thaw: {refused:?}"
    );
    assert_eq!(bare.thawed_detached().len(), 1, "and keeps the residue");
    granted.set_thawed_detached(bare.take_thawed_detached());

    // Thaw on the JIT: the child re-launches at its slot and the join delivers the total.
    let (mut thost, twin) = (granted, twin);
    let (tout, tsnap) =
        temen_run::jit_cap_run(&parent, 0, &fargs, &twin, PARENT_LOG2, 0, &mut thost)
            .expect("JIT thaw");
    assert_eq!(
        returned(tout),
        vec![4950],
        "JIT thaw: the re-launched child finished its loop"
    );
    assert_eq!(read_state(&tsnap), STATE_NORMAL, "back to NORMAL");

    // The same artifact thaws on the interpreter — one form for both engines.
    let (mut ihost, iwin) = restore();
    let iargs: Vec<Value> = fargs.iter().map(|&a| Value::I32(a as i32)).collect();
    let mut fuel = 50_000_000u64;
    let (ir, _) = run_capture_reserved_with_host(
        &parent,
        0,
        &iargs,
        &mut fuel,
        &iwin,
        PARENT_LOG2,
        &mut ihost,
    );
    assert_eq!(
        ir,
        Ok(vec![Value::I64(4950)]),
        "interpreter thaw of the JIT's artifact"
    );
}

/// The other direction: the **interpreter** freezes the tree, and the JIT thaws its artifact — the
/// child re-launched on the JIT from the interpreter's capture of it.
#[test]
fn an_interpreter_frozen_detached_child_thaws_on_the_jit() {
    let parent = instrument(PARENT);
    let child = instrument(CHILD);
    let (mut fhost, fargs) = powerbox(&child);
    let iargs: Vec<Value> = fargs.iter().map(|&a| Value::I32(a as i32)).collect();
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let mut fuel = 50_000_000u64;
    let (fr, fsnap) = run_capture_reserved_with_host(
        &parent,
        0,
        &iargs,
        &mut fuel,
        &win,
        PARENT_LOG2,
        &mut fhost,
    );
    assert!(fr.is_ok(), "the interpreter froze: {fr:?}");
    assert_eq!(fhost.captured_detached().len(), 1, "and captured the child");
    let art = temen_snapshot::freeze(&parent, &fsnap, &fhost).expect("serialize");

    let mut thost = Host::new();
    thost.set_durable(true);
    thost.grant_durable_module(&child);
    let mut twin = temen_snapshot::restore(&art, &parent, &mut thost).expect("restore");
    begin_thaw(&mut twin, ARENA, 0);
    match temen_run::jit_cap_run(&parent, 0, &fargs, &twin, PARENT_LOG2, 0, &mut thost) {
        Ok((o, _)) => assert_eq!(
            returned(o),
            vec![4950],
            "the JIT thawed the interpreter's cut"
        ),
        Err(JitError::Unsupported(_)) => {} // a target without the child executor
        Err(e) => panic!("JIT thaw failed: {e:?}"),
    }
}
