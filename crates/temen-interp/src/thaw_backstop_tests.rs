//! #3034 — the refusals that stay in front of a thaw's raw stores, behind the thaw pass
//! ([`temen_ir::thaw::validate`]). No residue the pass refuses reaches them from outside this crate
//! (a [`Host`]'s residue is seeded only from a [`Thaw`]), so these tests hand the seed hostile values
//! directly, through the private field, having said so ([`temen_ir::thaw::expect_backstops`]): a hole
//! in the pass is loud in debug otherwise. Each is refused, in the build CI runs and in release: a
//! fiber slot past the arena is not seeded, and its resume answers `FiberFault`; a vCPU with a join
//! slot past the table, no context of the pool, or an entry of another signature is not re-created.

use super::*;
use temen_durable::{
    begin_thaw, init_durable_window, transform_module_assume_confined, write_state,
};
use temen_ir::durable_abi::ShadowArena;

const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
const LOG2: u8 = 17;

/// The root spawns a child that parks on a futex word; it returns `2000 + 100 * the join's status`.
const THREADS: &str = "
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr_st, vr = thread.join vt
  vk = i64.const 100
  vs = i64.mul vr_st vk
  vb = i64.const 2000
  vt2 = i64.add vb vs
  return vt2
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
";

/// The root makes a fiber, resumes it (it suspends 42), then resumes handle 10, one it never made.
const FIBERS: &str = "
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 0
  v4, v5 = cont.resume v2 v3
  v6 = i64.const 7
  vten = i64.const 10
  v7, v8 = cont.resume vten v6
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 42
  v3 = suspend v2
  v4 = i64.const 100
  v5 = i64.add v3 v4
  return v5
  }
}
";

fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(temen_ir::Memory {
        size_log2: LOG2,
        shadow: Some(ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// `inst` frozen at its first safepoint, then thawed with its residue bent by `bend`: the thaw's result.
fn thawed(inst: &Module, bend: impl FnOnce(&mut Residue)) -> Result<Vec<Value>, Trap> {
    let mut win = init_durable_window(1 << LOG2, ARENA);
    write_state(&mut win, ARENA, temen_durable::STATE_UNWINDING);
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 1_000_000u64;
    let (r, mut snap) = run_capture_reserved_with_host(inst, 0, &[], &mut fuel, &win, LOG2, &mut h);
    assert!(r.is_ok(), "the freeze: {r:?}");
    bend(&mut h.residue);
    begin_thaw(&mut snap, ARENA, 0);
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(inst, 0, &[], &mut fuel, &snap, LOG2, &mut h).0
}

#[test]
fn a_fiber_slot_past_the_arena_is_not_seeded() {
    temen_ir::thaw::expect_backstops();
    let r = thawed(&instrument(FIBERS), |r| {
        let f = r.fibers[0];
        r.fibers = (0..11)
            .map(|slot| FrozenFiber {
                slot,
                shadow_sp: f.shadow_sp + slot as u64 * ARENA.stride,
                ..f
            })
            .collect();
    });
    assert_eq!(r, Err(Trap::FiberFault), "the resume of handle 10");
}

/// Each vCPU value the pass refuses. A join slot past the table pins no slot, so the root's rewound
/// join names a slot no child holds and answers `-EBADF`; a context outside the pool or an entry of
/// another signature ends the thread `ThreadFault`, which its join passes to the root.
#[test]
fn a_vcpu_the_pass_refuses_is_not_re_created() {
    temen_ir::thaw::expect_backstops();
    let inst = instrument(THREADS);
    let ebadf = 2000 + 100 * temen_ir::errno::EBADF;
    type Case = (fn(&mut Residue), Result<Vec<Value>, Trap>);
    let cases: [Case; 3] = [
        (
            |r| r.vcpus[0].slot = temen_ir::handles::SLOTS,
            Ok(vec![Value::I64(ebadf)]),
        ),
        (|r| r.vcpus[0].shadow_sp = 16, Err(Trap::ThreadFault)),
        (|r| r.vcpus[0].func = 0, Err(Trap::ThreadFault)),
    ];
    for (bend, want) in cases {
        assert_eq!(thawed(&inst, bend), want);
    }
}

/// A parent that may spawn a detached child (an `Instantiator` call on a path it never takes).
const SPAWNS: &str = "
func (i32) -> (i64) {
block 0 (v0: i32) {
  vz = i32.const 0
  vc = i32.eq v0 vz
  br_if vc 1() 2(v0)
}
block 1 () {
  vr = i64.const 1
  return vr
}
block 2 (vi: i32) {
  vrp = i64.const 0
  vh = call.cap 6 17 (i64) -> (i32) vi (vrp)
  vh64 = i64.extend_i32_u vh
  return vh64
  }
}
";

/// #3034 — a child's fault is the child's: a captured detached child whose residue holds an extent
/// its guest wrote ends with the push check's trap and rides its parent's cut as a completed child
/// answering it (`leave_cut`), while the parent's freeze goes on; the child is no longer carried.
/// Both engines record each extent at its own context's unwind, so no deterministic guest reaches
/// this through a run: the cut walk is handed the child directly.
#[test]
fn a_childs_refused_extent_ends_the_child_and_the_parents_freeze_goes_on() {
    let parent = instrument(SPAWNS);
    let child = Arc::new(instrument(THREADS));
    // The child's residue as its freeze left it, its vCPU's extent in a context's header.
    let real = {
        let mut win = init_durable_window(1 << LOG2, ARENA);
        write_state(&mut win, ARENA, temen_durable::STATE_UNWINDING);
        let mut h = Host::new();
        h.set_durable(true);
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(&child, 0, &[], &mut fuel, &win, LOG2, &mut h);
        assert!(r.is_ok(), "the child's freeze: {r:?}");
        h.take_residue()
    };
    assert_eq!(real.vcpus.len(), 1);
    let mut bad = real;
    bad.vcpus[0].shadow_sp = ARENA.region_base(3) + 8;
    let mut ch = Host::new();
    ch.residue = bad;
    let mut ph = Host::new();
    ph.residue.children = vec![1];
    ph.captured_detached.push(CapturedDetached {
        parent_task: 0,
        slot: 0,
        window: MemLayout::image(vec![0; 1 << LOG2]),
        reserved_log2: LOG2,
        host: Arc::new(Mutex::new(ch)),
        module: Arc::clone(&child),
        launch: DetachedLaunch {
            task: 1,
            digest: [0; 32],
            lane: -1,
            names: Vec::new(),
            pager: None,
            joinable: true,
        },
    });
    let ph = Arc::new(Mutex::new(ph));
    assert_eq!(leave_cut(&ph, (&parent).into(), 0, false), Ok(()));
    let h = ph.lock_unpoisoned();
    assert!(h.captured_detached().is_empty(), "the child is not carried");
    assert_eq!(
        h.frozen_detached(),
        [FrozenDetached {
            parent_task: 0,
            slot: 0,
            completed_result: Err(Trap::Unreachable.code()),
        }],
        "it rides as a completed child answering the trap"
    );
    assert!(!h.freeze_refused());
}
