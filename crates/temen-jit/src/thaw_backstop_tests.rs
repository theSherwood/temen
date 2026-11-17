//! #3034 — the refusals that stay in front of a JIT thaw's raw stores, behind the thaw pass
//! ([`temen_ir::thaw::validate`]). Only the pass builds a [`JitThaw`], so no hostile value reaches
//! them from outside this crate: these tests freeze a real run, bend one value of its residue, and
//! set it as the next run's seed directly, having said so ([`temen_ir::thaw::expect_backstops`]): a
//! hole in the pass is loud in debug otherwise. Each is refused, in the build CI runs and in release,
//! as the tree-walker refuses it (`temen-interp`'s `thaw_backstop_tests`).

use super::*;
use temen_durable::{init_durable_window, transform_module_assume_confined, write_state};
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

fn instrument(src: &str) -> IrModule {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(temen_ir::Memory {
        size_log2: LOG2,
        shadow: Some(ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

fn compiled(m: &IrModule) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        empty_cap_thunk,
        core::ptr::null_mut(),
        LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile")
}

/// `m` frozen at its first safepoint on the JIT, then thawed with its residue bent by `bend`, set as
/// the run's seed directly: the thaw's outcome.
fn thawed(m: &IrModule, bend: impl FnOnce(&mut DurableResidue)) -> JitOutcome {
    let mut win = init_durable_window(1 << LOG2, ARENA);
    write_state(&mut win, ARENA, temen_durable::STATE_UNWINDING);
    let mut cm = compiled(m);
    cm.durable = true;
    let (o, snap) = cm.run(&[], Some(&win), Some(SNAP_CAP)).expect("the freeze");
    assert!(matches!(o, JitOutcome::Returned(_)), "the freeze: {o:?}");
    let mut residue = cm.take_durable_residue();
    bend(&mut residue);
    let mut snap = snap;
    temen_durable::begin_thaw(&mut snap, ARENA, 0);
    let mut cm = compiled(m);
    cm.seed.vcpus = residue.vcpus;
    cm.seed.fibers = residue.fibers;
    cm.seed.root_sp = residue.root_sp;
    cm.seed.children = Some(residue.children);
    cm.durable = true;
    cm.run(&[], Some(&snap), Some(SNAP_CAP))
        .expect("the thaw")
        .0
}

/// Each vCPU value the pass refuses, as the tree-walker answers it: a join slot past the table pins
/// no slot, so the root's rewound join answers `-EBADF`; a context outside the pool or an entry of
/// another signature ends the thread `ThreadFault`, which its join passes to the root.
#[test]
fn a_vcpu_the_pass_refuses_is_not_re_created() {
    let _backstops = temen_ir::thaw::expect_backstops();
    let m = instrument(THREADS);
    let ebadf = 2000 + 100 * temen_ir::errno::EBADF;
    thawed(&m, |r| {
        assert_eq!(r.vcpus.len(), 1, "the parked child rides")
    });
    let slot = thawed(&m, |r| r.vcpus[0].slot = temen_ir::handles::SLOTS);
    assert!(
        matches!(&slot, JitOutcome::Returned(v) if *v == [ebadf]),
        "a join slot past the table: {slot:?}"
    );
    for (what, bend) in [
        (
            "a context outside the pool",
            (|r| r.vcpus[0].shadow_sp = 16) as fn(&mut DurableResidue),
        ),
        ("an entry of another signature", |r| r.vcpus[0].func = 0),
    ] {
        let o = thawed(&m, bend);
        assert!(
            matches!(o, JitOutcome::Trapped(TrapKind::ThreadFault)),
            "{what}: {o:?}"
        );
    }
}

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

/// A fiber slot past the arena is not seeded (the seed stops there, as `cont.new` refuses one), and
/// the resume of handle 10 answers `FiberFault`, as on the tree-walker.
#[test]
fn a_fiber_slot_past_the_arena_is_not_seeded() {
    let _backstops = temen_ir::thaw::expect_backstops();
    let o = thawed(&instrument(FIBERS), |r| {
        let f = r.fibers[0];
        r.fibers = (0..11)
            .map(|slot| FrozenFiber {
                slot,
                shadow_sp: f.shadow_sp + slot as u64 * ARENA.stride,
                ..f
            })
            .collect();
    });
    assert!(
        matches!(o, JitOutcome::Trapped(TrapKind::FiberFault)),
        "{o:?}"
    );
}

/// A module with no arena has no root region and no thaw words (INVARIANTS #16): a vCPU record seeded
/// into its run (here a finished one, so no context is asked for) does not reach the root's stores,
/// which would land at window offsets 0 and 8.
#[test]
fn a_vcpu_record_without_an_arena_writes_no_thaw_word() {
    let _backstops = temen_ir::thaw::expect_backstops();
    let real = {
        let m = instrument(THREADS);
        let mut win = init_durable_window(1 << LOG2, ARENA);
        write_state(&mut win, ARENA, temen_durable::STATE_UNWINDING);
        let mut cm = compiled(&m);
        cm.durable = true;
        cm.run(&[], Some(&win), Some(SNAP_CAP)).expect("the freeze");
        cm.take_durable_residue()
    };
    let src = "
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v5 = i64.const 5
  return v5
  }
}
";
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(temen_ir::Memory {
        size_log2: LOG2,
        shadow: None,
    });
    let mut cm = compiled(&m);
    cm.durable = true;
    cm.seed.vcpus = real
        .vcpus
        .into_iter()
        .map(|v| FrozenVCpu {
            completed_result: Some(5),
            ..v
        })
        .collect();
    cm.seed.root_sp = real.root_sp;
    cm.seed.children = Some(real.children);
    let win = vec![0u8; 1 << LOG2];
    let (_, out) = cm.run(&[], Some(&win), Some(SNAP_CAP)).expect("the run");
    assert_eq!(
        out[..16],
        [0; 16],
        "nothing is stored at window offsets 0 and 8"
    );
}
