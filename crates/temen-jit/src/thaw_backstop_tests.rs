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
    assert_eq!(residue.vcpus.len(), 1, "the parked child rides");
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
    temen_ir::thaw::expect_backstops();
    let m = instrument(THREADS);
    let ebadf = 2000 + 100 * temen_ir::errno::EBADF;
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
