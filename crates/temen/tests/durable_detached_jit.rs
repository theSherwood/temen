//! #1361 step 4 — a durable parent frozen on the **JIT** while its detached child (op 15) is live,
//! through the embedder's path (`temen_run::jit_cap_run`) and the §12 codec: the freeze rings the
//! child's own freeze word, the child unwinds in its own window, and the harvest puts its window and
//! powerbox on the `Host` as a `CapturedDetached` — the interpreter's form, so one artifact serves
//! both engines. The thaw re-launches the child at its join slot under `REWINDING`, on the JIT and on
//! the interpreter alike, and the parent's join delivers the uninterrupted total.

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, read_state, transform_module,
    transform_module_assume_confined, write_state, STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, FreezeScope, Host, MemLayout, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_jit::{JitError, JitOutcome};

const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
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

/// Sums `0..100` with a back-edge poll per iteration — 4950 uninterrupted. The zero-length `unmap`
/// through its `AddressSpace` (refused, its answer unused) is what gives the loop its poll: a function
/// that cannot suspend gets none, and runs to its end under a freeze rather than unwinding (#1937).
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
    let budget = host.grant_budget(-1, 1 << 20, -1);
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
        &MemLayout::image(init_durable_window(1 << PARENT_LOG2, ARENA).to_vec()),
        PARENT_LOG2,
        0,
        &mut host,
        None,
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
    let (_, fsnap) = temen_run::jit_cap_run(
        &parent,
        0,
        &fargs,
        &MemLayout::image(win.to_vec()),
        PARENT_LOG2,
        0,
        &mut fhost,
        None,
    )
    .expect("JIT freeze");
    assert_eq!(
        read_state(fsnap.bytes()),
        STATE_UNWINDING,
        "the parent froze"
    );
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
    let art = temen_snapshot::freeze(&parent, fsnap.bytes(), &fhost).expect("serialize");
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
    let refused = temen_run::jit_cap_run(
        &parent,
        0,
        &fargs,
        &MemLayout::image(twin.to_vec()),
        PARENT_LOG2,
        0,
        &mut bare,
        None,
    );
    assert!(
        matches!(refused, Err(JitError::Unsupported(_))),
        "an ungranted child program refuses the thaw: {:?}",
        refused.as_ref().map(|(o, _)| o)
    );
    assert_eq!(bare.thawed_detached().len(), 1, "and keeps the residue");
    granted.set_thawed_detached(bare.take_thawed_detached());

    // Thaw on the JIT: the child re-launches at its slot and the join delivers the total.
    let (mut thost, twin) = (granted, twin);
    let (tout, tsnap) = temen_run::jit_cap_run(
        &parent,
        0,
        &fargs,
        &MemLayout::image(twin.to_vec()),
        PARENT_LOG2,
        0,
        &mut thost,
        None,
    )
    .expect("JIT thaw");
    assert_eq!(
        returned(tout),
        vec![4950],
        "JIT thaw: the re-launched child finished its loop"
    );
    assert_eq!(read_state(tsnap.bytes()), STATE_NORMAL, "back to NORMAL");

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
    match temen_run::jit_cap_run(
        &parent,
        0,
        &fargs,
        &MemLayout::image(twin.to_vec()),
        PARENT_LOG2,
        0,
        &mut thost,
        None,
    ) {
        Ok((o, _)) => assert_eq!(
            returned(o),
            vec![4950],
            "the JIT thawed the interpreter's cut"
        ),
        Err(JitError::Unsupported(_)) => {} // a target without the child executor
        Err(e) => panic!("JIT thaw failed: {e:?}"),
    }
}

/// #2010 — the root of a depth-2 durable tree: `v0` its `Instantiator`, `v1` the [`NEST`] module, `v2`
/// a `Budget` whose `spawn` ceiling is 4, `v3` freeze authority over detached progeny, `v4` a settle
/// time in ms. It spawns `NEST` detached at entry 0 with two named grants (#2018): the authority as
/// `"freeze"`, so the durable child may spawn in turn, and the module as `"nest"`. It waits until the
/// budget's `spawn` room shows the child and the grandchild both live, or the child has ended,
/// parking 1 ms a time (a durable run on the oracle has one worker, so a waiter must give it up),
/// then the settle time. Then one fiber resume: the run's only fiber safepoint, where
/// `arm_freeze_after(win, 1)` freezes the tree with both live. Last, it joins the child and returns
/// what the child returns.
const NEST_ROOT: &str = "memory 18 shadow 16448 65536
data 70000 \"freeze\"
data 70008 \"nest\"
func (i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32) {
  vr0 = i64.const 70016
  vn0 = i32.const 70000
  i32.store vr0 vn0
  vr1 = i64.const 70020
  vl0 = i32.const 6
  i32.store vr1 vl0
  vr2 = i64.const 70024
  i32.store vr2 v3
  vr3 = i64.const 70032
  vn1 = i32.const 70008
  i32.store vr3 vn1
  vr4 = i64.const 70036
  vl1 = i32.const 4
  i32.store vr4 vl1
  vr5 = i64.const 70040
  i32.store vr5 v1
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vgp = i64.const 70016
  vgn = i64.const 2
  vz = i64.const 0
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vgp, vgn, vz, vlog, vz)
  br 1(v0, v2, vc, v4)
}
block 1 (vi: i32, vbu: i32, vch: i32, vms: i32) {
  vwa = i64.const 100000
  vwe = i32.const 0
  vwt = i64.const 1000000
  vww = i32.atomic.wait vwa vwe vwt
  vdim = i64.const 2
  vroom = call.cap 14 1 (i64) -> (i64) vbu (vdim)
  vtwo = i64.const 2
  vlive = i64.le_s vroom vtwo
  vst = call.cap 6 9 (i32) -> (i32) vi (vch)
  vz32 = i32.const 0
  vended = i32.ne vst vz32
  vgo = i32.or vlive vended
  br_if vgo 2(vi, vch, vms) 1(vi, vbu, vch, vms)
}
block 2 (vi2: i32, vch2: i32, vms2: i32) {
  vsa = i64.const 100000
  vse = i32.const 0
  vsm = i64.extend_i32_u vms2
  vmil = i64.const 1000000
  vsn = i64.mul vsm vmil
  vsw = i32.atomic.wait vsa vse vsn
  vf = ref.func 1
  vsp = i64.const 200000
  vk = cont.new vf vsp
  vz2 = i64.const 0
  vs, vx = cont.resume vk vz2
  vj = call.cap 6 1 (i32) -> (i64) vi2 (vch2)
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  return v1
  }
}
";

/// #2010 — the child [`NEST_ROOT`] spawns (entry 0) resolves its `"budget"` and `"nest"`, spawns
/// `"nest"` detached at entry 1, joins it, and returns 100 more than it. The grandchild (entry 1)
/// waits a second on a futex nothing notifies, so it is live when the freeze lands, then returns 7.
const NEST: &str = "memory 17 shadow 16448 65536
data 90112 \"budget\"
data 90120 \"nest\"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vbp = i64.const 90112
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vmp = i64.const 90120
  vml = i64.const 4
  vm = self.resolve vmp vml
  vbw = i64.extend_i32_u vb
  vmw = i64.extend_i32_u vm
  vz = i64.const 0
  ve = i64.const 1
  vlog = i64.const 17
  vinst = i32.wrap_i64 v0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vmw, vz, vz, ve, vlog, vz)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vh)
  vhundred = i64.const 100
  vr = i64.add vj vhundred
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vr = i64.const 7
  return vr
  }
}
";

/// What [`NEST_ROOT`] returns uninterrupted: the grandchild's 7, plus 100 from the child.
const NEST_TOTAL: i64 = 107;

/// [`NEST_ROOT`] and [`NEST`], instrumented. Both touch memory, so they take the confined transform.
fn nest_modules() -> (temen_ir::Module, temen_ir::Module) {
    let confined = |src: &str| {
        let m = transform_module_assume_confined(&temen_text::parse_module(src).expect("parse"))
            .expect("transform");
        temen_verify::verify_module(&m).expect("instrumented module verifies");
        m
    };
    (confined(NEST_ROOT), confined(NEST))
}

/// [`NEST_ROOT`]'s powerbox: its `Instantiator`, `nest` as a durable `Module`, a `Budget` of 1 MiB
/// with a `spawn` ceiling of 4, and freeze authority over detached progeny; and its arguments, with
/// `settle_ms` last.
fn nest_powerbox(nest: &temen_ir::Module, settle_ms: i32) -> (Host, Vec<i64>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(nest);
    let budget = host.grant_budget(-1, 1 << 20, 4);
    let freeze = host.grant_freeze_authority(FreezeScope::DetachedProgeny);
    let args = [inst, modh, budget, freeze, settle_ms].map(i64::from);
    (host, args.to_vec())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interp,
    Jit,
}

/// Run `root` over `win` on `engine`, the JIT with `fc` as its freeze controller: its answer and its
/// final window. `None` on a target without the JIT's child executor.
fn nest_run(
    engine: Engine,
    root: &temen_ir::Module,
    args: &[i64],
    win: &[u8],
    host: &mut Host,
    fc: Option<std::sync::Arc<temen_jit::FreezeController>>,
) -> Option<(i64, Vec<u8>)> {
    match engine {
        Engine::Interp => {
            let iargs: Vec<Value> = args.iter().map(|&a| Value::I32(a as i32)).collect();
            let mut fuel = u64::MAX / 2;
            match run_capture_reserved_with_host(root, 0, &iargs, &mut fuel, win, PARENT_LOG2, host)
            {
                (Ok(v), snap) => match v[..] {
                    [Value::I64(n)] => Some((n, snap)),
                    ref other => panic!("unexpected result {other:?}"),
                },
                (Err(t), _) => panic!("the interpreter run trapped: {t:?}"),
            }
        }
        Engine::Jit => match temen_run::jit_cap_run(
            root,
            0,
            args,
            &MemLayout::image(win.to_vec()),
            PARENT_LOG2,
            0,
            host,
            fc,
        ) {
            Ok((o, snap)) => Some((returned(o)[0], snap.bytes().to_vec())),
            Err(JitError::Unsupported(_)) => None, // a target without the child executor
            Err(e) => panic!("JIT run failed: {e:?}"),
        },
    }
}

/// For each captured child, how many children its own artifact carries.
fn carried(host: &Host) -> Vec<usize> {
    assert!(
        host.unreached_detached().is_empty(),
        "a child never reached its poll"
    );
    host.captured_detached()
        .iter()
        .map(|c| {
            let h = c.host.lock().unwrap_or_else(|e| e.into_inner());
            assert!(
                h.unreached_detached().is_empty(),
                "a grandchild never reached its poll"
            );
            h.captured_detached().len()
        })
        .collect()
}

/// Restore `art` into a fresh host that re-grants [`NEST`], ready to thaw.
fn nest_restore(art: &[u8], root: &temen_ir::Module, nest: &temen_ir::Module) -> (Host, Vec<u8>) {
    let mut host = Host::new();
    host.set_durable(true);
    host.grant_durable_module(nest);
    let mut win = temen_snapshot::restore(art, root, &mut host).expect("restore");
    begin_thaw(&mut win, ARENA, 0);
    (host, win)
}

/// Every engine's thaw of `art` that does not answer [`NEST_TOTAL`], described.
fn nest_thaws(
    art: &[u8],
    froze: Engine,
    root: &temen_ir::Module,
    nest: &temen_ir::Module,
    args: &[i64],
) -> Vec<String> {
    let mut wrong = Vec::new();
    for thaws in [Engine::Interp, Engine::Jit] {
        let (mut host, win) = nest_restore(art, root, nest);
        if let Some((r, _)) = nest_run(thaws, root, args, &win, &mut host, None) {
            if r != NEST_TOTAL {
                wrong.push(format!("frozen on {froze:?}, thawed on {thaws:?}: {r}"));
            }
        }
    }
    wrong
}

/// #2010 — **a durable detached child spawns and joins a grandchild, and a freeze carries both**, on
/// every engine. The freeze lands at the root's fiber safepoint: at once, while the child may still be
/// filing the grandchild, which then starts unwinding (#1760) or is rung once filed; or after a
/// settle, with the child parked joining the grandchild and the grandchild parked on its futex. The
/// root rings the child, and the child its own child, so each unwinds in its own window: the child
/// rides the root's artifact and the grandchild the child's, as the oracle's recursive harvest leaves
/// them. Every engine thaws every engine's artifact to the uninterrupted [`NEST_TOTAL`]. On the JIT
/// the settled freeze is a teardown's: its ring reaches the grandchild through the child's nursery
/// before any parked task is poisoned, and the child's join is left parked for the grandchild's end.
#[test]
fn a_durable_childs_grandchild_rides_its_freeze_on_every_engine() {
    use Engine::*;
    let (root, nest) = nest_modules();
    let mut wrong = Vec::new();
    for e in [Interp, Jit] {
        let (mut host, args) = nest_powerbox(&nest, 0);
        let win = init_durable_window(1 << PARENT_LOG2, ARENA);
        if let Some((r, _)) = nest_run(e, &root, &args, &win, &mut host, None) {
            if r != NEST_TOTAL {
                wrong.push(format!("uninterrupted on {e:?}: {r}"));
            }
        }
    }
    for froze in [Interp, Jit] {
        for settle_ms in [0, 100] {
            let (mut fhost, args) = nest_powerbox(&nest, settle_ms);
            let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
            arm_freeze_after(&mut win, 1);
            let Some((r, fsnap)) = nest_run(froze, &root, &args, &win, &mut fhost, None) else {
                continue;
            };
            if (r, carried(&fhost)) != (0, vec![1]) {
                wrong.push(format!(
                    "freeze on {froze:?} after {settle_ms} ms: {r}, carrying {:?}",
                    carried(&fhost)
                ));
                continue;
            }
            let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
            wrong.extend(nest_thaws(&art, froze, &root, &nest, &args));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// #2010 — **the JIT's freeze reaches a grandchild through the joins.** The embedder's
/// [`temen_jit::FreezeController`] requests the freeze while the root is parked joining the child and
/// the child is parked joining the grandchild. No teardown rings anyone yet: the root's join rings the
/// child, the child's join rings the grandchild, and each join is abandoned for re-issue once the
/// child it waits on has unwound. Without the child's ring the grandchild would wait out its second,
/// return, and be joined rather than carried.
#[test]
fn a_controller_freeze_reaches_a_grandchild_through_the_joins_on_the_jit() {
    let (root, nest) = nest_modules();
    let (mut fhost, args) = nest_powerbox(&nest, 0);
    let fc = temen_jit::FreezeController::new();
    // Well inside the grandchild's second-long wait, so the root and the child are parked in their
    // joins. (Landing sooner, it would cut the root's wait for both to be live, and the teardown's
    // ring would carry the same tree.)
    let ctl = {
        let fc = fc.clone();
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(200));
            fc.request_freeze();
        })
    };
    let win = init_durable_window(1 << PARENT_LOG2, ARENA);
    let run = nest_run(Engine::Jit, &root, &args, &win, &mut fhost, Some(fc));
    ctl.join().expect("controller");
    let Some((r, fsnap)) = run else {
        return;
    };
    assert_eq!(
        (r, carried(&fhost)),
        (0, vec![1]),
        "the freeze must carry the child, and the grandchild in the child's artifact"
    );
    let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
    let wrong = nest_thaws(&art, Engine::Jit, &root, &nest, &args);
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// #2010 — a JIT thaw refuses, whole, a captured child whose own powerbox carries residue the JIT
/// re-creates only for a run's root: here a grandchild recorded as having completed before the cut.
/// Re-launched without it, the child's rewound join would find nothing at its slot. The residue stays
/// on the host for a thaw on the interpreter.
#[test]
fn a_jit_thaw_refuses_a_captured_childs_residue_it_cannot_recreate() {
    let (root, nest) = nest_modules();
    let (mut fhost, args) = nest_powerbox(&nest, 0);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    arm_freeze_after(&mut win, 1);
    let (_, fsnap) =
        nest_run(Engine::Interp, &root, &args, &win, &mut fhost, None).expect("oracle");
    let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
    let (mut host, win) = nest_restore(&art, &root, &nest);
    let mut thawed = host.take_thawed_detached();
    assert_eq!(thawed.len(), 1, "the captured child");
    thawed[0]
        .host
        .set_frozen_detached(vec![temen_interp::FrozenDetached {
            parent_task: 0,
            slot: 1,
            completed_result: Ok(7),
        }]);
    host.set_thawed_detached(thawed);
    let refused = temen_run::jit_cap_run(
        &root,
        0,
        &args,
        &MemLayout::image(win),
        PARENT_LOG2,
        0,
        &mut host,
        None,
    );
    assert!(
        matches!(refused, Err(JitError::Unsupported(_))),
        "the JIT refuses the thaw: {:?}",
        refused.as_ref().map(|(o, _)| o)
    );
    assert_eq!(host.thawed_detached().len(), 1, "and keeps the residue");
}
