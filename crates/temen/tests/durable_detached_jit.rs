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
use temen_interp::{
    run_capture_reserved_with_host, FreezeScope, Host, MemLayout, StreamRole, Value,
};
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
/// a `Budget` whose `spawn` room falls to 2 once the tree has settled ([`nest_powerbox`]), `v3`
/// freeze authority over detached progeny, `v4` a settle time in ms, `v5` its stdout. It spawns
/// `NEST` detached at entry 0 with three named grants (#2018): the authority as `"freeze"`, so the
/// durable child may spawn in turn, the module as `"nest"`, and its stdout as `"stdout"` (#2054). It
/// waits until the budget's `spawn` room shows the tree settled, or the child has ended, parking 1 ms
/// a time (a durable run on the oracle has one worker, so a waiter must give it up), then the settle
/// time. Then one fiber resume: the run's only fiber safepoint, where `arm_freeze_after(win, 1)`
/// freezes the settled tree. Last, it joins the child and returns what the child returns.
const NEST_ROOT: &str = "memory 18 shadow 16448 65536
data 70000 \"freeze\"
data 70008 \"nest\"
data 70064 \"stdout\"
func (i32, i32, i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i32) {
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
  vr6 = i64.const 70048
  vn2 = i32.const 70064
  i32.store vr6 vn2
  vr7 = i64.const 70052
  vl2 = i32.const 6
  i32.store vr7 vl2
  vr8 = i64.const 70056
  i32.store vr8 v5
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vgp = i64.const 70016
  vgn = i64.const 3
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

/// #2010 — a child [`NEST_ROOT`] can spawn in [`NEST`]'s place: it spawns a thread, joins it, and
/// returns 100 more than it. The thread waits a second on a futex nothing notifies, so it is live
/// when the freeze lands, then returns 7.
const THREAD_NEST: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vj = thread.join vt
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

/// #2031 — a child [`NEST_ROOT`] can spawn in [`NEST`]'s place whose root holds a parked fiber when
/// the freeze lands. The fiber suspends 5; the root waits a second on a futex nothing notifies, then
/// resumes it, and it returns 2. Beside it runs a thread that waits as long and returns 0, which the
/// root joins, so [`NEST_ROOT`] sees two live vCPUs. The root returns 100 more than the three.
const FIBER_NEST: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
  vf = ref.func 1
  vsp = i64.const 100000
  vk = cont.new vf vsp
  vs, vx = cont.resume vk vz
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vs2, vx2 = cont.resume vk vz
  vj = thread.join vt
  vhundred = i64.const 100
  vr0 = i64.add vx vx2
  vr1 = i64.add vr0 vj
  vr = i64.add vr1 vhundred
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vfive = i64.const 5
  vy = suspend vfive
  vtwo = i64.const 2
  return vtwo
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  va = i64.const 66004
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vr = i64.const 0
  return vr
  }
}
";

/// #2031 — [`FIBER_NEST`] without the thread: the child's root is its only vCPU, so nothing but its
/// own capture can flatten the fiber. A first fiber, made before it, runs to its end, so the freeze
/// finds slot 0 free below the parked fiber in slot 1. A thaw that rebuilt only the parked one would
/// seed it at slot 0, where the root's handle to it does not resolve (#1684).
const FIBER_ONLY_NEST: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  vfa = ref.func 2
  vspa = i64.const 90000
  vka = cont.new vfa vspa
  vf = ref.func 1
  vsp = i64.const 100000
  vk = cont.new vf vsp
  vsa, vxa = cont.resume vka vz
  vs, vx = cont.resume vk vz
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vs2, vx2 = cont.resume vk vz
  vhundred = i64.const 100
  vr0 = i64.add vx vx2
  vr1 = i64.add vr0 vxa
  vr = i64.add vr1 vhundred
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vfive = i64.const 5
  vy = suspend vfive
  vtwo = i64.const 2
  return vtwo
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vr = i64.const 0
  return vr
  }
}
";

/// #2041 — a child [`NEST_ROOT`] can spawn in [`NEST`]'s place that returns at once, so the root's
/// wait for it ends and the freeze finds it **completed but not joined**: only its result crosses.
const QUICK_NEST: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vr = i64.const 107
  return vr
  }
}
";

/// #2041 — a child [`NEST_ROOT`] can spawn in [`NEST`]'s place that traps at once: the freeze finds
/// it completed **trapped** and not joined, so its trap crosses as its `join` outcome, which the
/// root's `join` re-raises.
const TRAP_NEST: &str = "memory 17 shadow 16448 65536
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  unreachable
  }
}
";

/// #2054 — a child [`NEST_ROOT`] can spawn in [`NEST`]'s place that resolves the stdout its spawner
/// re-granted it, waits a second on a futex nothing notifies, so it is live when the freeze lands,
/// then writes `X` to that stdout and returns [`NEST_TOTAL`].
const PRINT_NEST: &str = "memory 17 shadow 16448 65536
data 90112 \"stdout\"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vp = i64.const 90112
  vl = i64.const 6
  vo = self.resolve vp vl
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vb = i64.const 90200
  vx = i32.const 88
  i32.store8 vb vx
  vn = i64.const 1
  vwr = call.cap 0 1 (i64, i64) -> (i64) vo (vb, vn)
  vr = i64.const 107
  return vr
  }
}
";

/// #2054 — the same one level down: a child that re-grants the grandchild (entry 1) the stdout it
/// inherited, joins it, and returns 100 more than it. The grandchild resolves that stdout, waits a
/// second, writes `X` to it and returns 7, so its bytes reach the root through the child's stream.
const PRINT_GRAND_NEST: &str = "memory 17 shadow 16448 65536
data 90112 \"budget\"
data 90120 \"nest\"
data 90128 \"stdout\"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vbp = i64.const 90112
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vmp = i64.const 90120
  vml = i64.const 4
  vm = self.resolve vmp vml
  vop = i64.const 90128
  vol = i64.const 6
  vo = self.resolve vop vol
  vga = i64.const 90144
  vgw = i64.const 25769893904
  i64.store vga vgw
  vgh = i64.const 90152
  vo64 = i64.extend_i32_u vo
  i64.store vgh vo64
  vbw = i64.extend_i32_u vb
  vmw = i64.extend_i32_u vm
  vz = i64.const 0
  vgn = i64.const 1
  ve = i64.const 1
  vlog = i64.const 17
  vinst = i32.wrap_i64 v0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vmw, vga, vgn, ve, vlog, vz)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vh)
  vhundred = i64.const 100
  vr = i64.add vj vhundred
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vp = i64.const 90128
  vl = i64.const 6
  vo = self.resolve vp vl
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vb = i64.const 90200
  vx = i32.const 88
  i32.store8 vb vx
  vn = i64.const 1
  vwr = call.cap 0 1 (i64, i64) -> (i64) vo (vb, vn)
  vr = i64.const 7
  return vr
  }
}
";

/// #2041 — the same one level down: a child that spawns a grandchild that returns at once, waits for
/// it to end without joining it, then spawns two that wait a second, and joins those first. Its tree
/// settles with the child and the slow two live, which it reaches only after the quick one has ended:
/// the freeze finds the child live, with the quick one completed but not joined. It returns 100 more
/// than the quick one's 7 and the slow ones' 0.
const COMPLETED_NEST: &str = "memory 17 shadow 16448 65536
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
  vquick = i64.const 1
  vlog = i64.const 17
  vinst = i32.wrap_i64 v0
  vq = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vmw, vz, vz, vquick, vlog, vz)
  br 1(vinst, vbw, vmw, vq)
}
block 1 (vi: i32, vbw1: i64, vmw1: i64, vq1: i32) {
  vwa = i64.const 66000
  vwe = i32.const 0
  vwt = i64.const 1000000
  vww = i32.atomic.wait vwa vwe vwt
  vst = call.cap 6 9 (i32) -> (i32) vi (vq1)
  vz32 = i32.const 0
  vended = i32.ne vst vz32
  br_if vended 2(vi, vbw1, vmw1, vq1) 1(vi, vbw1, vmw1, vq1)
}
block 2 (vi2: i32, vbw2: i64, vmw2: i64, vq2: i32) {
  vz2 = i64.const 0
  vslow = i64.const 2
  vlog2 = i64.const 17
  vs1 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi2 (vbw2, vmw2, vz2, vz2, vslow, vlog2, vz2)
  vs2 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vi2 (vbw2, vmw2, vz2, vz2, vslow, vlog2, vz2)
  vj1 = call.cap 6 1 (i32) -> (i64) vi2 (vs1)
  vj2 = call.cap 6 1 (i32) -> (i64) vi2 (vs2)
  vjq = call.cap 6 1 (i32) -> (i64) vi2 (vq2)
  vhundred = i64.const 100
  vr0 = i64.add vjq vhundred
  vr1 = i64.add vr0 vj1
  vr = i64.add vr1 vj2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vr = i64.const 7
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 1000000000
  vw = i32.atomic.wait va vw0 vto
  vr = i64.const 0
  return vr
  }
}
";

/// [`NEST_ROOT`] and the `child` it spawns ([`NEST`] or [`THREAD_NEST`]), instrumented. Both touch
/// memory, so they take the confined transform.
fn nest_modules(child: &str) -> (temen_ir::Module, temen_ir::Module) {
    let confined = |src: &str| {
        let m = transform_module_assume_confined(&temen_text::parse_module(src).expect("parse"))
            .expect("transform");
        temen_verify::verify_module(&m).expect("instrumented module verifies");
        m
    };
    (confined(NEST_ROOT), confined(child))
}

/// [`NEST_ROOT`]'s powerbox: its `Instantiator`, `nest` as a durable `Module`, a `Budget` of 1 MiB,
/// freeze authority over detached progeny, and stdout; and its arguments, `settle_ms` among them. The
/// tree has settled once `live` of the root's descendants are live, so the budget's `spawn` ceiling
/// is 2 more than `live`: its room then falls to the 2 the root waits for.
fn nest_powerbox(nest: &temen_ir::Module, settle_ms: i32, live: i64) -> (Host, Vec<i64>) {
    let mut host = Host::new();
    host.set_durable(true);
    let inst = host.grant_instantiator(0, 1 << PARENT_LOG2);
    let modh = host.grant_durable_module(nest);
    let budget = host.grant_budget(-1, 1 << 20, live + 2);
    let freeze = host.grant_freeze_authority(FreezeScope::DetachedProgeny);
    let out = host.grant_stream(StreamRole::Out);
    let args = [inst, modh, budget, freeze, settle_ms, out].map(i64::from);
    (host, args.to_vec())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interp,
    Jit,
}

/// What a run of [`NEST_ROOT`] answers: its result, or the code of the trap it ended in, and what
/// it wrote to stdout.
type Answer<'a> = (Result<i64, i64>, &'a str);

/// Run `root` over `win` on `engine`, the JIT with `fc` as its freeze controller: its result (or
/// trap code), what it wrote to stdout, and its final window. `None` on a target without the JIT's
/// child executor.
fn nest_run(
    engine: Engine,
    root: &temen_ir::Module,
    args: &[i64],
    win: &[u8],
    host: &mut Host,
    fc: Option<std::sync::Arc<temen_jit::FreezeController>>,
) -> Option<(Result<i64, i64>, String, Vec<u8>)> {
    let (r, snap) = match engine {
        Engine::Interp => {
            let iargs: Vec<Value> = args.iter().map(|&a| Value::I32(a as i32)).collect();
            let mut fuel = u64::MAX / 2;
            match run_capture_reserved_with_host(root, 0, &iargs, &mut fuel, win, PARENT_LOG2, host)
            {
                (Ok(v), snap) => match v[..] {
                    [Value::I64(n)] => (Ok(n), snap),
                    ref other => panic!("unexpected result {other:?}"),
                },
                (Err(t), snap) => (Err(t.code()), snap),
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
            Ok((JitOutcome::Returned(v), snap)) => (Ok(v[0]), snap.bytes().to_vec()),
            Ok((JitOutcome::Trapped(t), snap)) => (Err(t as i64), snap.bytes().to_vec()),
            Ok((other, _)) => panic!("the run neither returned nor trapped: {other:?}"),
            Err(JitError::Unsupported(_)) => return None, // a target without the child executor
            Err(e) => panic!("JIT run failed: {e:?}"),
        },
    };
    let out = String::from_utf8_lossy(&host.take_stdout()).into_owned();
    Some((r, out, snap))
}

/// For each captured child, how many of its `thread.spawn` vCPUs its own artifact carries live.
fn carried_threads(host: &Host) -> Vec<usize> {
    assert!(
        host.unreached_detached().is_empty(),
        "a child never reached its poll"
    );
    host.captured_detached()
        .iter()
        .map(|c| {
            let h = c.host.lock().unwrap_or_else(|e| e.into_inner());
            h.frozen_vcpus()
                .iter()
                .filter(|v| v.completed_result.is_none())
                .count()
        })
        .collect()
}

/// For each captured child, how many fibers its own artifact carries, each as it parked.
fn carried_fibers(host: &Host) -> Vec<usize> {
    assert!(
        host.unreached_detached().is_empty(),
        "a child never reached its poll"
    );
    host.captured_detached()
        .iter()
        .map(|c| {
            let h = c.host.lock().unwrap_or_else(|e| e.into_inner());
            h.frozen_fibers().iter().filter(|f| !f.is_free()).count()
        })
        .collect()
}

/// How many detached children the root's artifact carries as completed but not joined (#2041).
fn completed_children(host: &Host) -> Vec<usize> {
    assert!(
        host.unreached_detached().is_empty(),
        "a child never reached its poll"
    );
    vec![host.frozen_detached().len()]
}

/// For each captured child, how many of its own children its artifact carries as completed but not
/// joined (#2041).
fn completed_grandchildren(host: &Host) -> Vec<usize> {
    assert!(
        host.unreached_detached().is_empty(),
        "a child never reached its poll"
    );
    host.captured_detached()
        .iter()
        .map(|c| {
            c.host
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .frozen_detached()
                .len()
        })
        .collect()
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

/// Each of `runs`' thaws of `art` that does not answer `answer`, described: what the thaw returns
/// and writes. `runs` are the engines that run the tree at all: one of them refusing the thaw is
/// wrong too.
fn nest_thaws(
    art: &[u8],
    froze: Engine,
    root: &temen_ir::Module,
    nest: &temen_ir::Module,
    args: &[i64],
    runs: &[Engine],
    answer: Answer,
) -> Vec<String> {
    let mut wrong = Vec::new();
    for &thaws in runs {
        let (mut host, win) = nest_restore(art, root, nest);
        match nest_run(thaws, root, args, &win, &mut host, None) {
            Some((r, out, _)) if (r, out.as_str()) == answer => {}
            Some((r, out, _)) => wrong.push(format!(
                "frozen on {froze:?}, thawed on {thaws:?}: {r:?}, writing {out:?}"
            )),
            None => wrong.push(format!("frozen on {froze:?}, thawed on {thaws:?}: refused")),
        }
    }
    wrong
}

/// Freeze [`NEST_ROOT`] over `child`, whose tree has settled once `live` of the root's descendants
/// are live, on every engine at the root's fiber safepoint: at once, and after a settle of 100 ms.
/// `carried` reads what each freeze carried, which must be `want`, and every engine thaws every
/// engine's artifact to `answer`, the tree's uninterrupted answer on every engine: its result, and
/// the rest of its output after what the frozen run wrote (#2054). An engine that runs the tree
/// uninterrupted must freeze and thaw it too.
fn rides_every_engine(
    child: &str,
    live: i64,
    carried: impl Fn(&Host) -> Vec<usize>,
    want: &[usize],
    answer: Answer,
) {
    use Engine::*;
    let (root, child) = nest_modules(child);
    let mut wrong = Vec::new();
    let mut runs = Vec::new();
    for e in [Interp, Jit] {
        let (mut host, args) = nest_powerbox(&child, 0, live);
        let win = init_durable_window(1 << PARENT_LOG2, ARENA);
        let Some((r, out, _)) = nest_run(e, &root, &args, &win, &mut host, None) else {
            continue; // a target without the JIT's child executor
        };
        if (r, out.as_str()) != answer {
            wrong.push(format!("uninterrupted on {e:?}: {r:?}, writing {out:?}"));
        }
        runs.push(e);
    }
    for &froze in &runs {
        for settle_ms in [0, 100] {
            let (mut fhost, args) = nest_powerbox(&child, settle_ms, live);
            let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
            arm_freeze_after(&mut win, 1);
            let Some((r, out, fsnap)) = nest_run(froze, &root, &args, &win, &mut fhost, None)
            else {
                wrong.push(format!("freeze on {froze:?} after {settle_ms} ms: refused"));
                continue;
            };
            let got = carried(&fhost);
            let rest = answer.1.strip_prefix(out.as_str());
            if (r, &got[..]) != (Ok(0), want) || rest.is_none() {
                wrong.push(format!(
                    "freeze on {froze:?} after {settle_ms} ms: {r:?}, writing {out:?}, carrying \
                     {got:?}"
                ));
                continue;
            }
            let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
            let thawed = (answer.0, rest.unwrap_or_default());
            wrong.extend(nest_thaws(&art, froze, &root, &child, &args, &runs, thawed));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
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
    rides_every_engine(NEST, 2, carried, &[1], (Ok(NEST_TOTAL), ""));
}

/// #2010 — **the JIT's freeze reaches a grandchild through the joins.** The embedder's
/// [`temen_jit::FreezeController`] requests the freeze while the root is parked joining the child and
/// the child is parked joining the grandchild. No teardown rings anyone yet: the root's join rings the
/// child, the child's join rings the grandchild, and each join is abandoned for re-issue once the
/// child it waits on has unwound. Without the child's ring the grandchild would wait out its second,
/// return, and be joined rather than carried.
#[test]
fn a_controller_freeze_reaches_a_grandchild_through_the_joins_on_the_jit() {
    controller_freeze(NEST, |host| {
        assert_eq!(
            carried(host),
            vec![1],
            "the freeze must carry the child, and the grandchild in the child's artifact"
        );
    });
}

/// #2010 — **the same, for a thread.** The freeze lands while the root is parked joining the child
/// and the child is parked joining its thread. The child's join ends on the word the root's join
/// rang; its root unwinds while the thread is still parked, so it has no outcome to publish until the
/// thread has unwound too and the capture is taken. Published early, the root's join would take the
/// child's placeholder for its result.
#[test]
fn a_controller_freeze_reaches_a_childs_thread_through_the_joins_on_the_jit() {
    controller_freeze(THREAD_NEST, |host| {
        assert_eq!(
            carried_threads(host),
            vec![1],
            "the freeze must carry the child, and its thread in the child's artifact"
        );
    });
}

/// The embedder's [`temen_jit::FreezeController`] freezes [`NEST_ROOT`] over `child` on the JIT, 200
/// ms in, while the child waits out its second: the root is parked joining it, or still polling it
/// (a child with no thread never shows [`NEST_ROOT`] two live vCPUs). `check` checks what the freeze
/// carried, and every engine's thaw must answer [`NEST_TOTAL`].
fn controller_freeze(child: &str, check: impl FnOnce(&Host)) {
    let (root, child) = nest_modules(child);
    let (mut fhost, args) = nest_powerbox(&child, 0, 2);
    let fc = temen_jit::FreezeController::new();
    // Well inside the innermost wait of a second, so the root and the child are parked in their
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
    let Some((r, _, fsnap)) = run else {
        return;
    };
    assert_eq!(r, Ok(0), "the root froze");
    check(&fhost);
    let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
    let wrong = nest_thaws(
        &art,
        Engine::Jit,
        &root,
        &child,
        &args,
        &[Engine::Interp, Engine::Jit],
        (Ok(NEST_TOTAL), ""),
    );
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// #2041 — a JIT thaw refuses, whole, two detached children of one domain that claim one join slot.
/// The interpreter keeps a join table per spawning vCPU, so children of different spawners can share
/// a slot number; the JIT keeps one per domain, where both handles would name one child. Here the
/// captured child's live grandchild holds slot 0, and a completed record from another of its vCPUs
/// claims slot 0 too. The residue stays on the host for a thaw on the interpreter.
#[test]
fn a_jit_thaw_refuses_two_detached_children_at_one_join_slot() {
    let (root, nest) = nest_modules(NEST);
    let (mut fhost, args) = nest_powerbox(&nest, 0, 2);
    let mut win = init_durable_window(1 << PARENT_LOG2, ARENA);
    arm_freeze_after(&mut win, 1);
    let (_, _, fsnap) =
        nest_run(Engine::Interp, &root, &args, &win, &mut fhost, None).expect("oracle");
    let art = temen_snapshot::freeze(&root, &fsnap, &fhost).expect("serialize");
    let (mut host, win) = nest_restore(&art, &root, &nest);
    let mut thawed = host.take_thawed_detached();
    assert_eq!(thawed.len(), 1, "the captured child");
    thawed[0]
        .host
        .set_frozen_detached(vec![temen_interp::FrozenDetached {
            parent_task: 3,
            slot: 0,
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

/// #2010 — **a durable detached child's thread unwinds with it, and rides its artifact**, on every
/// engine. The child spawns the thread while its window is `NORMAL`, so on the JIT the thread is an OS
/// thread with a shadow context of its own; the freeze lands while the child is parked joining it and
/// the thread is parked on its futex. The child's join ends on its own freeze word and the child
/// unwinds; its task waits for the thread to unwind into its own region, then the capture takes the
/// child's window with the thread's residue beside it. Every engine thaws every engine's artifact,
/// re-creating the thread, to the uninterrupted [`NEST_TOTAL`].
#[test]
fn a_durable_childs_thread_rides_its_freeze_on_every_engine() {
    rides_every_engine(THREAD_NEST, 2, carried_threads, &[1], (Ok(NEST_TOTAL), ""));
}

/// #2031 — **a durable detached child's parked fiber rides its freeze**, on every engine. The freeze
/// lands while the child's root is parked on its futex with the fiber suspended, and its thread
/// parked beside it. On the JIT the thread flattens the fiber as it unwinds, since a vCPU's flattening
/// walks its domain's whole table, so the fiber's record must say that the root already took its 5.
/// Recorded as unconsumed, a thaw re-delivers that 5 to the root's next resume, which then answers
/// 110. Every engine thaws every engine's artifact to the uninterrupted [`NEST_TOTAL`].
#[test]
fn a_durable_childs_fiber_rides_its_freeze_on_every_engine() {
    rides_every_engine(FIBER_NEST, 2, carried_fibers, &[1], (Ok(NEST_TOTAL), ""));
}

/// #2031 — **the same, for a child with no vCPU but its root.** No thread unwinds to flatten the
/// fiber, so the child's own capture flattens it, as a run's root's does, before the image is taken.
/// The embedder's controller lands the freeze while the child waits and the root polls it.
#[test]
fn a_controller_freeze_flattens_a_childs_own_fiber_on_the_jit() {
    controller_freeze(FIBER_ONLY_NEST, |host| {
        assert_eq!(
            carried_fibers(host),
            vec![1],
            "the freeze must carry the child, and its fiber in the child's artifact"
        );
    });
}

/// #2041 — **a detached child that completed but was not joined rides its parent's freeze as its
/// result**, on every engine. The root's wait for the child ends on the child's end, so the freeze
/// finds it finished and unjoined: the artifact carries only its `join` outcome. Every engine thaws
/// every engine's artifact, re-creating that outcome at the child's join slot, and the root's rewound
/// `join` reloads it instead of spawning the child again.
#[test]
fn a_completed_detached_child_rides_its_parents_freeze_on_every_engine() {
    rides_every_engine(
        QUICK_NEST,
        2,
        completed_children,
        &[1],
        (Ok(NEST_TOTAL), ""),
    );
}

/// #2041 — **the same for a child that trapped**: its trap rides the artifact as its `join` outcome,
/// not a value. Every engine's thaw re-raises it from the root's rewound `join`, as the uninterrupted
/// run does.
#[test]
fn a_completed_detached_childs_trap_rides_its_parents_freeze_on_every_engine() {
    let unreachable = (Err(temen_ir::trap_code::UNREACHABLE), "");
    rides_every_engine(TRAP_NEST, 2, completed_children, &[1], unreachable);
}

/// #2054 — **a durable detached child's inherited stdout rides its freeze**, on every engine. Its
/// spawner re-granted it the root's stdout, which aliases the root's buffer (§7c); the freeze lands
/// while the child waits, before it writes. A thaw re-aliases the stream to the thawing root's
/// stdout, so the `X` the child writes after the cut reaches the root as it does uninterrupted.
/// Captured without its sink, the stream came back as the child's own buffer, which nothing reads.
#[test]
fn a_detached_childs_inherited_stdout_rides_its_freeze_on_every_engine() {
    rides_every_engine(PRINT_NEST, 1, carried, &[0], (Ok(NEST_TOTAL), "X"));
}

/// #2054 — **the same one level down.** The grandchild's stdout is the one its spawner inherited,
/// so its thaw aliases it to the child's stream, which the child's own thaw aliased to the root's.
#[test]
fn a_grandchilds_inherited_stdout_rides_its_freeze_on_every_engine() {
    rides_every_engine(PRINT_GRAND_NEST, 2, carried, &[1], (Ok(NEST_TOTAL), "X"));
}

/// #2041 — **the same one level down.** The freeze captures the child live, joining its slow
/// grandchildren, with its quick grandchild completed but not joined. The child's artifact carries the
/// quick one's outcome beside the slow ones' captures, and a thaw re-creates all three in the child's
/// nursery before the child runs, so its rewound joins park on the slow ones and reload the quick one.
#[test]
fn a_captured_childs_completed_child_rides_its_freeze_on_every_engine() {
    rides_every_engine(
        COMPLETED_NEST,
        3,
        completed_grandchildren,
        &[1],
        (Ok(NEST_TOTAL), ""),
    );
}
