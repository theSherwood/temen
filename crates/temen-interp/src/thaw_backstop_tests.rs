//! #3034 — the refusals that stay in front of a thaw's raw stores, behind the thaw pass
//! ([`temen_ir::thaw::validate`]). No residue the pass refuses reaches them from outside this crate
//! (a [`Host`]'s residue is seeded only from a [`Thaw`]), so these tests hand the seed hostile values
//! directly, through the private field. In a release build each is refused: a fiber slot past the
//! arena is not seeded, and its resume answers `FiberFault`; a vCPU with a join slot past the table,
//! no context of the pool, or an entry of another signature is not re-created. In a debug build each
//! is a hole in the pass, and loud.

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
  vst = i64.extend_i32_u vr_st
  vk = i64.const 100
  vs = i64.mul vst vk
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
    transform_module_assume_confined(&m).expect("transform")
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
#[cfg_attr(
    debug_assertions,
    should_panic(expected = "a fiber slot inside the arena")
)]
fn a_fiber_slot_past_the_arena_is_not_seeded() {
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
#[cfg_attr(debug_assertions, should_panic(expected = "the thaw pass admits only"))]
fn a_vcpu_the_pass_refuses_is_not_re_created() {
    let inst = instrument(THREADS);
    let ebadf = 2000 + 100 * i64::from(temen_ir::errno::EBADF as u32);
    let cases: [(fn(&mut Residue), Result<Vec<Value>, Trap>); 3] = [
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
