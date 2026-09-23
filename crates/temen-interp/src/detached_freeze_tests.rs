//! #1361 step 4 — a durable parent frozen while its **detached** child is live, thawed, and run to the
//! uninterrupted result: the child is captured as its own window + powerbox (the harvest), re-launched
//! under `REWINDING` from them (`relaunch_detached`), and the parent's rewound `thread.join` parks on it
//! exactly as before the cut.
//!
//! The hand-off between freeze and thaw is in memory here; `temen-snapshot`'s tests carry the same
//! residue through the codec, and `temen`'s `durable_detached_jit.rs` runs the whole arc on the JIT.

use super::*;
use temen_durable::{begin_thaw, init_durable_window, read_state, transform_module, write_state};
use temen_ir::durable_abi::ShadowArena;

const ARENA: ShadowArena = ShadowArena {
    base: 16448,
    end: 65536,
};
const PARENT_LOG2: u8 = 18;

/// The parent: spawn the child detached (op 15, 7-arg form: budget, module, no grants, entry 0,
/// `size_log2` 17, no quota), join it, return what it returns.
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

/// The child: sums `0..100` with a back-edge poll per iteration, so a freeze catches it mid-loop and
/// its continuation is a real one — 4950 uninterrupted.
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

fn instrument(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// A durable powerbox granting what the parent's op 15 needs, in a fixed order so a thaw host minted
/// the same way holds the same handle values (the in-memory stand-in for the codec's verbatim table).
fn powerbox(child: &Module) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(0, 1 << 20, 0);
    host.grant_freeze_authority(FreezeScope::DetachedProgeny);
    (
        host,
        vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)],
    )
}

fn run(
    parent: &Module,
    host: &mut Host,
    args: &[Value],
    win: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut fuel = 50_000_000u64;
    run_capture_reserved_with_host(parent, 0, args, &mut fuel, win, PARENT_LOG2, host)
}

/// A captured child as the thaw receives it — what `temen-snapshot`'s restore produces from the
/// artifact, here handed over in memory.
fn thawed(c: CapturedDetached) -> ThawedDetached {
    let host = Arc::try_unwrap(c.host)
        .unwrap_or_else(|_| panic!("the run released the child's powerbox"))
        .into_inner()
        .unwrap_or_else(|e| e.into_inner());
    ThawedDetached {
        parent_task: c.parent_task,
        slot: c.slot,
        window: c.window,
        reserved_log2: c.reserved_log2,
        host,
        launch: c.launch,
    }
}

#[test]
fn a_live_detached_child_is_captured_and_its_thaw_completes_the_join() {
    let parent = instrument(PARENT);
    let child = instrument(CHILD);

    // Control: uninterrupted, the child's total reaches the parent through the join.
    let (mut host, args) = powerbox(&child);
    let (base, _) = run(
        &parent,
        &mut host,
        &args,
        &init_durable_window(1 << PARENT_LOG2, ARENA),
    );
    assert_eq!(base, Ok(vec![Value::I64(4950)]), "uninterrupted total");

    // Freeze from the start: the parent spawns the child, unwinds at its next poll, and rings the
    // child's doorbell rather than waiting for it; the child unwinds at its own first poll, and the
    // driver harvests its window and powerbox.
    let (mut fhost, fargs) = powerbox(&child);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (fr, fsnap) = run(&parent, &mut fhost, &fargs, &win);
    assert!(
        fr.is_ok(),
        "the freeze returns a placeholder, not a refusal: {fr:?}"
    );
    assert_eq!(read_state(&fsnap), STATE_UNWINDING, "the parent froze");
    assert!(
        fhost.unreached_detached().is_empty(),
        "the child reached its poll"
    );
    let captured = std::mem::take(&mut fhost.captured_detached);
    assert_eq!(captured.len(), 1, "one live detached child was captured");
    let c = &captured[0];
    assert_eq!(
        (c.parent_task, c.slot),
        (0, 0),
        "the root's first join slot"
    );
    assert_eq!(c.launch.entry, 0);
    assert_eq!(c.launch.digest, module_digest(&child));
    assert!(
        c.launch.task > 0,
        "the child's own task id in the frozen run"
    );
    assert_eq!(
        read_state(c.window.bytes()),
        STATE_UNWINDING,
        "the doorbell set the child's own freeze word, and it unwound"
    );

    // Thaw: the parent's window rewinds; the child is re-launched from its own window and powerbox
    // under `REWINDING`; the parent's rewound join parks on it, and the loop finishes where it froze.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, ARENA, 0);
    let (mut thost, targs) = powerbox(&child);
    thost.set_thawed_detached(captured.into_iter().map(thawed).collect());
    let (tr, tsnap) = run(&parent, &mut thost, &targs, &twin);
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "the re-launched child finished its loop and the join delivered it"
    );
    assert_eq!(
        read_state(&tsnap),
        STATE_NORMAL,
        "the thaw ran back to NORMAL"
    );
    assert!(
        thost.captured_detached().is_empty() && thost.take_thawed_detached().is_empty(),
        "nothing left over for a later run"
    );
}

/// The gate's other half: without freeze authority over its detached progeny, a durable parent's
/// spawn is refused probeably (`-EINVAL`) — it cannot mint a child no freeze of it could capture.
#[test]
fn a_durable_parent_without_authority_is_refused_the_spawn() {
    const SPAWN_ONLY: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, vz, vlog, vz)
  vr = i64.extend_i32_s vc
  return vr
  }
}
";
    let parent = instrument(SPAWN_ONLY);
    let child = instrument(CHILD);
    let spawn = |authority: bool| {
        let mut host = Host::new();
        host.set_durable(true);
        let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
        let modh = host.grant_durable_module(&child);
        let budget = host.grant_budget(0, 1 << 20, 0);
        if authority {
            host.grant_freeze_authority(FreezeScope::DetachedProgeny);
        }
        let args = [Value::I32(inst), Value::I32(modh), Value::I32(budget)];
        run(
            &parent,
            &mut host,
            &args,
            &init_durable_window(1 << PARENT_LOG2, ARENA),
        )
        .0
    };
    assert_eq!(
        spawn(false),
        Ok(vec![Value::I64(EINVAL)]),
        "refused without authority"
    );
    assert_eq!(
        spawn(true),
        Ok(vec![Value::I64(0)]),
        "admitted at slot 0 with it"
    );
}
