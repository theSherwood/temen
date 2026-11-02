//! #1361 step 4 — a durable parent frozen while its **detached** child is live, thawed, and run to the
//! uninterrupted result: the child is captured as its own window + powerbox (the harvest), re-launched
//! under `REWINDING` from them (`relaunch_detached`), and the parent's rewound `thread.join` parks on it
//! exactly as before the cut.
//!
//! The hand-off between freeze and thaw is in memory here; `temen-snapshot`'s tests carry the same
//! residue through the codec, and `temen`'s `durable_detached_jit.rs` runs the whole arc on the JIT.

use super::*;
use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined, write_state,
};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::{Data, SpawnRec};

const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
const PARENT_LOG2: u8 = 18;

/// Where a parent lays its spawn record ([`spawn_rec`]): past the shadow arena, which spans
/// `[16448, 65536)`.
const REC_AT: u64 = 65536;

/// The parent: spawn the child detached (its spawn record at [`REC_AT`]: module, budget, no grants,
/// `size_log2` 17), join it, return what it returns.
const PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vma = i64.const 65544
  i32.store vma v1
  vba = i64.const 65548
  i32.store vba v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  return vr
  }
}
";

/// The child: sums `0..100` with a back-edge poll per iteration, so a freeze catches it mid-loop and
/// its continuation is a real one — 4950 uninterrupted. The zero-length `unmap` through its
/// `AddressSpace` (refused, its answer unused) is what gives the loop its poll: a function that
/// cannot suspend gets none, and runs to its end under a freeze rather than unwinding (#1937).
const CHILD: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, va: i64) {
  vas = i32.wrap_i64 va
  vz = i64.const 0
  vu = call.cap 5 1 (i64, i64) -> (i64) vas (vz, vz)
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

/// Parse `src`, add `data` to it, instrument it, and verify the result. Confined: the parent stores
/// its spawn record's handles at [`REC_AT`], past the shadow arena (R9).
fn instrument(src: &str, data: impl IntoIterator<Item = Data>) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.data.extend(data);
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented module verifies");
    inst
}

/// A parent's spawn record, at [`REC_AT`]: a `size_log2` 17 child, whose module and budget the
/// parent stores in at run time.
fn spawn_rec() -> Option<Data> {
    Some(spawn_rec_at(REC_AT))
}

/// A spawn record as [`spawn_rec`]'s, at `offset`.
fn spawn_rec_at(offset: u64) -> Data {
    Data {
        offset,
        readonly: false,
        bytes: SpawnRec {
            size_log2: 17,
            ..SpawnRec::default()
        }
        .encode(),
    }
}

/// A durable powerbox granting what the parent's spawn needs, in a fixed order so a thaw host minted
/// the same way holds the same handle values (the in-memory stand-in for the codec's verbatim table).
fn powerbox(child: &Module) -> (Host, Vec<Value>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator();
    let modh = host.grant_durable_module(child);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    host.grant_freeze_authority();
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
    let parent = instrument(PARENT, spawn_rec());
    let child = instrument(CHILD, None);

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
    // driver harvests its window and powerbox. One lane runs the two in turn, so the bell reaches
    // the child before its loop does: on two, the child can return between its last poll and the
    // bell, and it rides as completed instead ([`a_child_rung_after_its_last_poll_rides_completed`]).
    let (mut fhost, fargs) = powerbox(&child);
    fhost.set_lane_cap(1);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, ARENA, STATE_UNWINDING);
    let (fr, fsnap) = run(&parent, &mut fhost, &fargs, &win);
    assert!(
        fr.is_ok(),
        "the freeze returns a placeholder, not a refusal: {fr:?}"
    );
    assert_eq!(
        read_state(&fsnap, ARENA),
        STATE_UNWINDING,
        "the parent froze"
    );
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
    assert_eq!(c.launch.digest, module_digest(&child));
    assert!(
        c.launch.task > 0,
        "the child's own task id in the frozen run"
    );
    assert_eq!(
        read_state(c.window.bytes(), ARENA),
        STATE_UNWINDING,
        "the doorbell set the child's own freeze word, and it unwound"
    );

    // Thaw: the parent's window rewinds; the child is re-launched from its own window and powerbox
    // under `REWINDING`; the parent's rewound join parks on it, and the loop finishes where it froze.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, ARENA, 0);
    let (mut thost, targs) = powerbox(&child);
    thost.set_thawed_detached(captured.into_iter().map(thawed).collect());
    thost.set_child_generations(fhost.child_generations().to_vec());
    let (tr, tsnap) = run(&parent, &mut thost, &targs, &twin);
    assert_eq!(
        tr,
        Ok(vec![Value::I64(4950)]),
        "the re-launched child finished its loop and the join delivered it"
    );
    assert_eq!(
        read_state(&tsnap, ARENA),
        STATE_NORMAL,
        "the thaw ran back to NORMAL"
    );
    assert!(
        thost.captured_detached().is_empty() && thost.take_thawed_detached().is_empty(),
        "nothing left over for a later run"
    );
}

/// #1688 — a child the bell reaches after its last poll returns rather than unwinding, and what it
/// returned is its answer: it rides as a completed child, and the thaw hands its result to the
/// rewound join instead of running it again. This child has no poll at all, so one lane, which runs
/// it only once its parent has unwound and rung, puts the whole child after the bell.
#[test]
fn a_child_rung_after_its_last_poll_rides_completed() {
    const POLL_FREE: &str = "memory 17 shadow 16448 65536
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 4950
  return v1
  }
}
";
    let parent = instrument(PARENT, spawn_rec());
    let child = instrument(POLL_FREE, None);
    let (mut fhost, fargs) = powerbox(&child);
    fhost.set_lane_cap(1);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    write_state(&mut win, ARENA, STATE_UNWINDING);
    let (fr, fsnap) = run(&parent, &mut fhost, &fargs, &win);
    assert!(fr.is_ok(), "the parent froze: {fr:?}");
    assert!(
        fhost.unreached_detached().is_empty() && fhost.captured_detached().is_empty(),
        "nothing to resume"
    );
    let frozen = fhost.frozen_detached().to_vec();
    assert_eq!(frozen.len(), 1, "the child rides as completed");
    assert_eq!(
        (
            frozen[0].parent_task,
            frozen[0].slot,
            frozen[0].completed_result
        ),
        (0, 0, Ok(4950)),
    );

    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, ARENA, 0);
    let (mut thost, targs) = powerbox(&child);
    thost.set_frozen_detached(frozen);
    thost.set_child_generations(fhost.child_generations().to_vec());
    let (tr, _) = run(&parent, &mut thost, &targs, &twin);
    assert_eq!(tr, Ok(vec![Value::I64(4950)]), "the join took the result");
}

/// The gate's other half: without freeze authority over its detached progeny, a durable parent's
/// spawn is refused probeably (`-EINVAL`) — it cannot mint a child no freeze of it could capture.
#[test]
fn a_durable_parent_without_authority_is_refused_the_spawn() {
    const SPAWN_ONLY: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vma = i64.const 65544
  i32.store vma v1
  vba = i64.const 65548
  i32.store vba v2
  vrp = i64.const 65536
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  vr = i64.extend_i32_s vc
  return vr
  }
}
";
    let parent = instrument(SPAWN_ONLY, spawn_rec());
    let child = instrument(CHILD, None);
    let spawn = |authority: bool| {
        let mut host = Host::new();
        host.set_durable(true);
        let inst = host.grant_instantiator();
        let modh = host.grant_durable_module(&child);
        let budget = host.grant_budget(-1, 1 << 20, -1);
        if authority {
            host.grant_freeze_authority();
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
        Ok(vec![Value::I64(temen_ir::handles::pack(0, 1) as i64)]),
        "admitted at slot 0 with it"
    );
}

/// Where [`TWO_PARENT`] lays its second child's spawn record, past its first ([`REC_AT`]).
const REC_B: u64 = REC_AT + temen_ir::SPAWN_REC_LEN as u64;

/// A parent of two children, `A` and then `B` (its args: the Instantiator, `A`'s module, the budget,
/// `B`'s module). It joins `B`, then `A`, and returns `status_B·10¹² + status_A·10⁹ + B·10⁴ + A`.
/// `FLIP` requests the freeze just before `B`'s spawn, so the parent unwinds once both are spawned.
const TWO_PARENT: &str = "memory 18 shadow 16448 65536
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {
  vma = i64.const 65544
  i32.store vma v1
  vba = i64.const 65548
  i32.store vba v2
  vmb = i64.const 65608
  i32.store vmb v3
  vbb = i64.const 65612
  i32.store vbb v2
  vra = i64.const 65536
  vca = call.cap 6 17 (i64) -> (i32) v0 (vra)
FLIP
  vrb = i64.const 65600
  vcb = call.cap 6 17 (i64) -> (i32) v0 (vrb)
  vsb, vb = call.cap 6 1 (i32) -> (i64, i64) v0 (vcb)
  vsa, va = call.cap 6 1 (i32) -> (i64, i64) v0 (vca)
  vk12 = i64.const 1000000000000
  vk9 = i64.const 1000000000
  vk4 = i64.const 10000
  vx1 = i64.mul vsb vk12
  vx2 = i64.mul vsa vk9
  vx3 = i64.mul vb vk4
  vy1 = i64.add vx1 vx2
  vy2 = i64.add vy1 vx3
  vy3 = i64.add vy2 va
  return vy3
  }
}
";

/// #2430 — **a child the thaw does not relaunch ends with `ThreadFault`, whatever refused it.** The
/// parent freezes with both children live; the thaw grants `A`'s program, and another in `B`'s
/// place, so `B`'s relaunch finds its grant gone and drops it. `B`'s join answers `ThreadFault` and
/// `A`'s answers `A`'s total, as had `B` ended so. `B`'s slot used to stay unfiled, past the end of
/// the rebuilt table, where the join's mask folds `B`'s handle onto `A`'s: `B`'s join took `A`'s
/// total, and `A`'s faulted.
#[test]
fn a_child_the_thaw_does_not_relaunch_ends_with_thread_fault() {
    let flip = format!(
        "  vfw = i64.const {}\n  vfs = i32.const {STATE_UNWINDING}\n  i32.store vfw vfs",
        ARENA
            .freeze_word()
            .expect("the arena holds the freeze word")
    );
    let recs = || [spawn_rec_at(REC_AT), spawn_rec_at(REC_B)];
    // `A` sums `0..100` and `B` `0..10`; a third program stands in for `B`'s at the thaw.
    let summing = |n: u32| {
        instrument(
            &CHILD.replace("i64.const 100", &format!("i64.const {n}")),
            None,
        )
    };
    let (a, b) = (summing(100), summing(10));
    let powerbox = |b: &Module| {
        let mut host = Host::new();
        host.set_durable(true);
        let inst = host.grant_instantiator();
        let ha = host.grant_durable_module(&a);
        let budget = host.grant_budget(-1, 1 << 20, -1);
        let hb = host.grant_durable_module(b);
        host.grant_freeze_authority();
        (host, [inst, ha, budget, hb].map(Value::I32))
    };
    let window = || init_durable_window(1 << PARENT_LOG2, ARENA);

    let (mut host, args) = powerbox(&b);
    let plain = instrument(&TWO_PARENT.replace("FLIP", ""), recs());
    let (base, _) = run(&plain, &mut host, &args, &window());
    assert_eq!(
        base,
        Ok(vec![Value::I64(45 * 10_000 + 4950)]),
        "uninterrupted"
    );

    // One lane, so neither child runs before the parent has unwound and rung it: both ride live.
    let parent = instrument(&TWO_PARENT.replace("FLIP", &flip), recs());
    let (mut fhost, fargs) = powerbox(&b);
    fhost.set_lane_cap(1);
    let (fr, fsnap) = run(&parent, &mut fhost, &fargs, &window());
    assert!(fr.is_ok(), "the freeze returns a placeholder: {fr:?}");
    let captured = std::mem::take(&mut fhost.captured_detached);
    let slots: Vec<usize> = captured.iter().map(|c| c.slot).collect();
    assert_eq!(slots, [0, 1], "both children ride live");

    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, ARENA, 0);
    let (mut thost, targs) = powerbox(&summing(7));
    thost.set_thawed_detached(captured.into_iter().map(thawed).collect());
    thost.set_child_generations(fhost.child_generations().to_vec());
    let (tr, _) = run(&parent, &mut thost, &targs, &twin);
    assert_eq!(
        tr,
        Ok(vec![Value::I64(
            Trap::ThreadFault.code() * 1_000_000_000_000 + 4950
        )]),
        "B's join answers ThreadFault, A's its total"
    );
}
