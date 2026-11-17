//! #3034 — a thaw artifact is untrusted input (owner, 2026-11-13). Its residue (frozen fibers and
//! vCPUs, the root's extent, the child table, detached children and fork twins) is admitted by one pass
//! at the trust boundary ([`temen_ir::thaw::validate`]), which every engine's thaw goes through.
//!
//! Each row freezes a real program, edits one value of the artifact's residue through the codec's own
//! writers ([`temen_snapshot::decode_residue`], [`temen_snapshot::encode_residue`]), and restores it.
//! A value no run could have made is refused as `Malformed` before any engine sees it. The decline rows
//! pin the engines that re-create less than the whole residue: they refuse it whole, to the oracle,
//! and never thaw part of it.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_durable::{
    begin_thaw, init_durable_window, transform_module_assume_confined, write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::thaw::{FrozenFiber, LiveTwin};
use temen_ir::{Memory, Module};
use temen_jit::JitError;
use temen_snapshot::{decode_residue, encode_residue, ResidueEdit, RestoreError, TwinRest};

const TEST_ARENA: ShadowArena = ShadowArena::new(16448, 65536);
const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

/// The root spawns a child that parks on a futex word; the freeze catches the child parked.
const THREADS: &str = r#"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr_st, vr = thread.join vt
  vk = i64.const 2000
  vs = i64.add vk vr
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  vk = i64.const 100
  vr = i64.mul vst64 vk
  return vr
  }
}
"#;

/// The root makes a fiber, resumes it (it suspends 42), then resumes handle 10, one it never made.
const FIBERS: &str = r#"
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
"#;

fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// `inst` frozen at its first safepoint on the tree-walker: the artifact.
fn frozen(inst: &Module) -> Vec<u8> {
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, TEST_ARENA, STATE_UNWINDING);
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 1_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "the freeze returns a placeholder: {r:?}");
    temen_snapshot::freeze(inst, &snap, &h).expect("freeze")
}

/// `art` with its residue edited by `edit`, through the codec's own writers.
fn bent(art: &[u8], edit: impl FnOnce(&mut ResidueEdit)) -> Vec<u8> {
    let mut r = decode_residue(art).expect("decode");
    edit(&mut r);
    encode_residue(art, &r).expect("encode")
}

fn assert_refused(what: &str, inst: &Module, art: &[u8]) {
    let got = temen_snapshot::restore(art, inst, &mut Host::new());
    assert!(
        matches!(got, Err(RestoreError::Malformed)),
        "{what}: {:?}",
        got.map(|_| ())
    );
}

type Bend = fn(&mut ResidueEdit);

/// Every vCPU value no spawn could have made. The first six are #2945's, which ended the vCPU
/// `ThreadFault` at its use; the rest are new. Each restored on `b099bbb` (and `task = usize::MAX`
/// then panicked both engines' thaw).
#[test]
fn a_vcpu_value_no_spawn_makes_is_refused_at_restore() {
    let inst = instrument(THREADS);
    let art = frozen(&inst);
    let real = decode_residue(&art).expect("decode");
    assert_eq!(real.vcpus.len(), 1, "the parked child rides");
    assert!(
        temen_snapshot::restore(&art, &inst, &mut Host::new()).is_ok(),
        "the real artifact restores"
    );
    let cases: &[(&str, Bend)] = &[
        ("shadow-SP past the arena", |r| {
            r.vcpus[0].shadow_sp = TEST_ARENA.end + 64
        }),
        ("shadow-SP below the arena", |r| r.vcpus[0].shadow_sp = 16),
        ("shadow-SP in the root's region", |r| {
            r.vcpus[0].shadow_sp = TEST_ARENA.base + 8
        }),
        ("an entry past the module", |r| r.vcpus[0].func = 99),
        ("an entry of another signature", |r| r.vcpus[0].func = 0),
        ("a join slot past the table", |r| {
            r.vcpus[0].slot = temen_ir::handles::SLOTS
        }),
        ("a task past TASK_MAX", |r| r.vcpus[0].task = usize::MAX),
        ("a spawner at or above its child", |r| {
            r.vcpus[0].parent_task = r.vcpus[0].task
        }),
        ("shadow-SP in a context's header", |r| {
            let ctx = (r.vcpus[0].shadow_sp - TEST_ARENA.base) / TEST_ARENA.stride;
            r.vcpus[0].shadow_sp = TEST_ARENA.base + ctx * TEST_ARENA.stride + 8;
        }),
        ("three spawn args", |r| r.vcpus[0].args.push(1)),
        ("a join slot the table never issued", |r| r.children.clear()),
        ("two vCPUs in one context", |r| {
            let mut v = r.vcpus[0].clone();
            v.task += 1;
            v.slot = 0;
            v.joinable = false;
            r.vcpus.push(v);
        }),
        ("a counter the table never leaves", |r| {
            r.children[0] = 1 << temen_ir::handles::GEN_BITS
        }),
    ];
    for (what, bend) in cases {
        assert_refused(what, &inst, &bent(&art, bend));
    }
}

/// A fiber residue past the arena: on `b099bbb` the restore admitted it, the tree-walker and the JIT
/// seeded the ten slots that fit and trapped `FiberFault` at the resume of handle 10, and the bytecode
/// engine seeded all eleven and returned `Ok(42)` — an engine divergence. Now it is refused, as are a
/// slot gap and an extent in another fiber's region.
#[test]
fn a_fiber_value_no_run_makes_is_refused_at_restore() {
    let inst = instrument(FIBERS);
    let art = frozen(&inst);
    let real = decode_residue(&art).expect("decode");
    assert_eq!(real.fibers.len(), 1, "the parked fiber rides");
    let cases: &[(&str, Bend)] = &[
        ("eleven fibers in an arena of ten", |r| {
            let f = r.fibers[0];
            r.fibers = (0..11).map(|slot| FrozenFiber { slot, ..f }).collect();
        }),
        ("a slot gap", |r| {
            r.fibers.push(FrozenFiber::free(2, 0));
        }),
        ("an extent in the next fiber's region", |r| {
            r.fibers[0].shadow_sp = TEST_ARENA.frame_base(2)
        }),
        ("a free slot with an entry", |r| {
            r.fibers.push(FrozenFiber {
                func: 1,
                ..FrozenFiber::free(1, 0)
            });
        }),
    ];
    for (what, bend) in cases {
        assert_refused(what, &inst, &bent(&art, bend));
    }
}

/// A fork twin's re-entry: on `b099bbb` an entry past the module reached `VCpu::new` on the
/// tree-walker. Now it is refused, as is a context past the arena.
#[test]
fn a_twin_value_no_fork_makes_is_refused_at_restore() {
    let inst = instrument(FIBERS);
    let art = frozen(&inst);
    let cases: &[(&str, Bend)] = &[
        ("an entry past the module", |r| r.twins[0].entry = 2),
        ("a context past the arena", |r| r.twins[0].ctx = 11),
        ("a pid past TASK_MAX", |r| r.twins[0].pid = u64::MAX),
    ];
    for (what, bend) in cases {
        let twinned = with_twin(&art);
        assert_refused(what, &inst, &bent(&twinned, bend));
    }
}

/// `art` carrying a live fork twin: a copy of itself, as a fork of its root at the cut.
fn with_twin(art: &[u8]) -> Vec<u8> {
    let nested = art.to_vec();
    bent(art, move |r| {
        r.twins.push(LiveTwin {
            pid: 5,
            entry: 0,
            ctx: 0,
            payload: TwinRest {
                servicer: true,
                names: Vec::new(),
                artifact: nested,
            },
        })
    })
}

/// An engine that does not re-create a whole residue declines it to the oracle whole. On `b099bbb`
/// the JIT and the bytecode engine dropped a fork twin silently and ran on without it.
#[test]
fn a_fork_twin_is_declined_whole_by_the_engines_that_do_not_recreate_one() {
    let inst = instrument(FIBERS);
    let art = with_twin(&frozen(&inst));
    let restored = || {
        let mut h = Host::new();
        let mut win = temen_snapshot::restore(&art, &inst, &mut h).expect("restore");
        begin_thaw(&mut win, TEST_ARENA, 0);
        h.set_durable(true);
        (h, win)
    };

    let (mut h, win) = restored();
    assert!(!temen_interp::bytecode::admits_thaw(&h));
    let mut fuel = 1_000_000u64;
    assert!(
        temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        )
        .is_none(),
        "the bytecode engine declines"
    );
    assert_eq!(h.thawed_twins().len(), 1, "and leaves the residue whole");

    let (mut h, win) = restored();
    let r = temen_run::jit_cap_run(
        &inst,
        0,
        &[],
        &MemLayout::image(win),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    );
    assert!(
        matches!(r, Err(JitError::Unsupported(_))),
        "the JIT declines: {:?}",
        r.as_ref().map(|(o, _)| o)
    );
    assert_eq!(h.thawed_twins().len(), 1, "and leaves the residue whole");
}
