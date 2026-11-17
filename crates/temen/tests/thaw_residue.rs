//! #3034 — a thaw artifact is untrusted input (owner, 2026-11-13). Its residue (frozen fibers and
//! vCPUs, the root's extent, the child table, detached children and fork twins) is admitted by one pass
//! at the trust boundary ([`temen_ir::thaw::validate`]), which every engine's thaw goes through.
//!
//! Each row freezes a real program, edits one value of the artifact's residue through the codec's own
//! writers ([`temen_snapshot::decode_residue`], [`temen_snapshot::encode_residue`]), and restores it.
//! A value no run could have made is refused as `Malformed` before any engine sees it. The decline rows
//! pin the engines that re-create less than the whole residue: they refuse it whole, to the oracle,
//! and never thaw part of it. The D3 rows pin a thaw that is transparent for a thread whose spawner
//! finished before the freeze, on every engine.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::thaw::{FrozenFiber, LiveTwin};
use temen_ir::{Memory, Module};
use temen_jit::{JitError, JitOutcome};
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
    assert_eq!(
        bent(&art, |_| {}),
        art,
        "the control: an unedited decode and encode"
    );
    let real = decode_residue(&art).expect("decode");
    assert_eq!(real.fibers.len(), 1, "the parked fiber rides");
    let cases: &[(&str, Bend)] = &[
        ("eleven fibers in an arena of ten", |r| eleven(r, 11)),
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
    // The control: ten, each at its own slot's extent, fill the arena and are admitted.
    let ten = bent(&art, |r| eleven(r, 10));
    assert!(temen_snapshot::restore(&ten, &inst, &mut Host::new()).is_ok());
}

/// `n` copies of the fiber, each in its own slot's region at the same depth.
fn eleven(r: &mut ResidueEdit, n: usize) {
    let f = r.fibers[0];
    let depth = f.shadow_sp - TEST_ARENA.region_base(1);
    r.fibers = (0..n)
        .map(|slot| FrozenFiber {
            slot,
            shadow_sp: TEST_ARENA.region_base(slot + 1) + depth,
            ..f
        })
        .collect();
}

/// A fork twin's re-entry: on `b099bbb` an entry past the module reached `VCpu::new` on the
/// tree-walker. Now it is refused, as is a context past the arena and a pid past `TASK_MAX`. The twin
/// carries an artifact with an empty residue, which its own pass admits in any context, so each
/// refusal is the parent's twin rule; the control, the same twin in the arena's top context, restores.
#[test]
fn a_twin_value_no_fork_makes_is_refused_at_restore() {
    let inst = instrument(FIBERS);
    let art = frozen(&inst);
    let bare = bent(&art, |r| *r = ResidueEdit::default());
    let twinned = |ctx: u32| with_twin_of(&art, &bare, ctx);
    assert!(
        temen_snapshot::restore(&twinned(10), &inst, &mut Host::new()).is_ok(),
        "the control"
    );
    let cases: &[(&str, Bend)] = &[
        ("an entry past the module", |r| r.twins[0].entry = 2),
        ("a pid past TASK_MAX", |r| r.twins[0].pid = u64::MAX),
    ];
    for (what, bend) in cases {
        assert_refused(what, &inst, &bent(&twinned(10), bend));
    }
    assert_refused("a context past the arena", &inst, &twinned(11));
}

/// A twin's own residue is admitted against the context its root runs in (`root_ctx`, the forking
/// vCPU's): a fiber of it may not hold that context. The twin here carries the frozen fiber, slot 0,
/// in context 1: refused for a twin in context 1, admitted for one in context 5.
#[test]
fn a_twins_residue_is_admitted_against_its_own_context() {
    let inst = instrument(FIBERS);
    let art = frozen(&inst);
    assert!(temen_snapshot::restore(&with_twin_of(&art, &art, 5), &inst, &mut Host::new()).is_ok());
    assert_refused(
        "a twin's fiber in the twin's own context",
        &inst,
        &with_twin_of(&art, &art, 1),
    );
}

/// `art` carrying a live fork twin: a copy of itself, as a fork of its root at the cut.
fn with_twin(art: &[u8]) -> Vec<u8> {
    with_twin_of(art, art, 0)
}

/// `art` carrying a live fork twin, whose own artifact is `nested`, running in context `ctx`.
fn with_twin_of(art: &[u8], nested: &[u8], ctx: u32) -> Vec<u8> {
    let nested = nested.to_vec();
    bent(art, move |r| {
        r.twins.push(LiveTwin {
            pid: 5,
            entry: 0,
            ctx,
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

    // Its entry over a cached program, too (on `f2aff3d` it skipped the gate).
    let (mut h, win) = restored();
    let shared = std::sync::Arc::new(inst.clone());
    let compiled = temen_interp::bytecode::compile_reserved(&shared).expect("compiles");
    let mut fuel = 1_000_000u64;
    assert!(
        temen_interp::bytecode::run_capture_reserved_over_compiled_with_host(
            &inst,
            compiled,
            0,
            &[],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        )
        .is_none(),
        "the bytecode engine declines over a cached program"
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
        matches!(
            r,
            Err(JitError::Unsupported(
                "durable JIT thaw: a detached child or fork twin the JIT cannot re-create"
            ))
        ),
        "the JIT declines for the twin: {:?}",
        r.as_ref().map(|(o, _)| o)
    );
    assert_eq!(h.thawed_twins().len(), 1, "and leaves the residue whole");
}

// ---- #3034 D3: a thread whose spawner finished before the freeze. Root spawns A; A spawns B and
// finishes; B owns a fiber, so the armed fiber trigger fires inside B while the root is parked in
// B's join. A real freeze writes B with `parent_task` naming A whether or not A rides a record. ----

/// B: owns a fiber that suspends 5; returns 5 + 100.
const B: &str = r#"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = ref.func 3
  v3 = i64.const 4096
  v4 = cont.new v2 v3
  v5 = i64.const 0
  v6, v7 = cont.resume v4 v5
  v8 = i64.const 100
  v9 = i64.add v7 v8
  return v9
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 5
  v3 = suspend v2
  return v3
  }
}
"#;

/// (a): A finishes unjoined. The root waits on the word at 66000 for B's handle, which A stores and
/// notifies before it returns 7.
const ROOT_UNJOINED: &str = r#"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  va = thread.spawn 1 vz vz
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vhb = i32.load vaddr
  vbst, vbr = thread.join vhb
  return vbr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  vb = thread.spawn 2 vz vz
  vaddr = i64.const 66000
  i32.store vaddr vb
  vone = i32.const 1
  vn = atomic.notify vaddr vone
  vr = i64.const 7
  return vr
  }
}
"#;

/// (b): A finishes and is joined; it returns B's handle.
const ROOT_JOINED: &str = r#"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  va = thread.spawn 1 vz vz
  vast, var = thread.join va
  vhb = i32.wrap_i64 var
  vbst, vbr = thread.join vhb
  return vbr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  vb = thread.spawn 2 vz vz
  vr = i64.extend_i32_u vb
  return vr
  }
}
"#;

fn interp_run(inst: &Module, win: &[u8], h: &mut Host) -> Result<Vec<Value>, temen_interp::Trap> {
    h.set_durable(true);
    let mut fuel = 10_000_000u64;
    run_capture_reserved_with_host(inst, 0, &[], &mut fuel, win, SIZE_LOG2, h).0
}

/// Freeze `root` + [`B`] with the trigger inside B, check what the freeze wrote, and thaw it on every
/// engine: each gives the uninterrupted answer. On `b099bbb` the tree-walker's thaw dropped B (its
/// spawner was not re-created) and the root's rewound join answered `-EBADF`, so it returned 0.
fn spawner_finished(root: &str, a_rides: bool) {
    let inst = instrument(&format!("{root}{B}"));
    let want = interp_run(
        &inst,
        &init_durable_window(WINDOW, TEST_ARENA),
        &mut Host::new(),
    );
    assert_eq!(want, Ok(vec![Value::I64(105)]), "uninterrupted");

    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, TEST_ARENA, 1);
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 10_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "the freeze returns a placeholder: {r:?}");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("freeze");

    // What the freeze wrote: B, its spawner A by id; A itself only while unjoined.
    let written = decode_residue(&art).expect("decode");
    let b = written
        .vcpus
        .iter()
        .find(|v| v.completed_result.is_none())
        .expect("B rides live");
    assert_eq!(b.parent_task, 1, "B names A, the thread that spawned it");
    assert_eq!(
        written.vcpus.iter().any(|v| v.task == 1),
        a_rides,
        "A rides a record iff no one joined it"
    );

    thaws_on_every_engine(&inst, &art, &want);
}

/// `art`, a real artifact of `inst`, re-freezes byte-identically once restored (§12.6), and thaws to
/// `want` on the tree-walker and the JIT; the bytecode engine, which runs no `thread.*` durably,
/// declines it to the tree-walker. Each engine is asserted to have run: this file builds only where
/// the JIT has its thread runtime (`fiber_rt`).
fn thaws_on_every_engine(inst: &Module, art: &[u8], want: &Result<Vec<Value>, temen_interp::Trap>) {
    // The control for every bent row: the codec's own writers, given the residue unedited, write
    // the artifact back byte for byte.
    assert_eq!(
        bent(art, |_| {}),
        art,
        "an unedited decode and encode is the artifact"
    );
    let restored = || {
        let mut h = Host::new();
        let mut win = temen_snapshot::restore(art, inst, &mut h).expect("a real artifact");
        begin_thaw(&mut win, TEST_ARENA, 0);
        (h, win)
    };

    let mut h = Host::new();
    let win = temen_snapshot::restore(art, inst, &mut h).expect("a real artifact");
    let again = temen_snapshot::freeze(inst, &win, &h).expect("re-freeze");
    assert_eq!(again, art, "a restored domain re-freezes byte-identically");

    let (mut h, win) = restored();
    assert_eq!(&interp_run(inst, &win, &mut h), want, "tree-walker thaw");

    let (mut h, win) = restored();
    let mut fuel = 10_000_000u64;
    assert!(
        temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
            inst,
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

    let (mut h, win) = restored();
    h.set_durable(true);
    let got = temen_run::jit_cap_run(
        inst,
        0,
        &[],
        &MemLayout::image(win),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    )
    .map(|(o, _)| o);
    let want: Vec<i64> = want
        .as_ref()
        .expect("an answer")
        .iter()
        .map(|v| match v {
            Value::I64(x) => *x,
            v => panic!("{v:?}"),
        })
        .collect();
    assert!(
        matches!(&got, Ok(JitOutcome::Returned(v)) if *v == want),
        "JIT thaw: {got:?}"
    );
}

#[test]
fn a_thread_whose_spawner_finished_unjoined_thaws_transparently() {
    spawner_finished(ROOT_UNJOINED, true);
}

#[test]
fn a_thread_whose_spawner_finished_and_was_joined_thaws_transparently() {
    spawner_finished(ROOT_JOINED, false);
}

// ---- #3034: a context whose unwound chain fills its shadow region exactly. The transform's
// push check traps only past the region's end (`newsp > region_end`), so a real freeze writes an
// extent equal to it. Each frame of `REC` is 16 bytes; the depths below fill both regions to the byte.

/// Root and spawned thread each recurse through their own `REC` to a depth; at the bottom the root
/// raises a flag and parks in the thread's join, and the thread waits for the flag and resumes a fiber
/// (the freeze trigger). Returns the fiber's 5.
fn full_region_src(thread_depth: i64, root_depth: i64) -> String {
    let rec = |idx: usize, bottom: &str| {
        format!(
            "func (i64) -> (i64) {{
block 0 (vn: i64) {{
  vz = i64.const 0
  vc = i64.eq vn vz
  br_if vc 1() 2(vn)
}}
block 1 () {{
{bottom}}}
block 2 (vm: i64) {{
  vone = i64.const 1
  vm1 = i64.sub vm vone
  vr = call {idx}(vm1)
  return vr
  }}
}}
"
        )
    };
    let thread_bottom = "  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vf = ref.func 4
  vss = i64.const 4096
  vk = cont.new vf vss
  vz2 = i64.const 0
  va, vb = cont.resume vk vz2
  return vb
";
    let root_bottom = "  vaddr = i64.const 66000
  vone = i32.const 1
  i32.store vaddr vone
  vn2 = atomic.notify vaddr vone
  vha = i64.const 66008
  vh = i32.load vha
  vst, vr = thread.join vh
  return vr
";
    format!(
        "func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vaddr = i64.const 66008
  i32.store vaddr vt
  vd = i64.const {root_depth}
  vr = call 3(vd)
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vd = i64.const {thread_depth}
  vr = call 2(vd)
  return vr
  }}
}}
{}{}func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = i64.const 5
  v3 = suspend v2
  return v3
  }}
}}
",
        rec(2, thread_bottom),
        rec(3, root_bottom)
    )
}

/// A vCPU, and the root beside it, frozen with their regions exactly full, thaw on every engine to
/// the uninterrupted answer. On `f2aff3d` the restore refused the artifact as `Malformed`
/// (`vcpu_of_sp` floored the vCPU's extent into the next context); on `b099bbb` its thaw answered
/// `ThreadFault` at the use site.
#[test]
fn a_vcpu_and_the_root_frozen_with_their_regions_exactly_full_thaw_on_every_engine() {
    let inst = instrument(&full_region_src(249, 250));
    let want = interp_run(
        &inst,
        &init_durable_window(WINDOW, TEST_ARENA),
        &mut Host::new(),
    );
    assert_eq!(want, Ok(vec![Value::I64(5)]), "uninterrupted");

    let armed = || {
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        arm_freeze_after(&mut win, TEST_ARENA, 1);
        win
    };
    // Frozen on the tree-walker.
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 10_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &armed(), SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "the freeze returns a placeholder: {r:?}");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("freeze");
    full_regions_written(&art);
    thaws_on_every_engine(&inst, &art, &want);

    // Frozen on the JIT, whose harvest reads the same extents (on `f2aff3d`: a refused residue, a
    // debug assert in `leave_residue`).
    let mut h = Host::new();
    h.set_durable(true);
    let snap = match temen_run::jit_cap_run(
        &inst,
        0,
        &[],
        &MemLayout::image(armed()),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    ) {
        Ok((JitOutcome::Returned(_), snap)) => snap,
        other => panic!("JIT freeze: {:?}", other.map(|(o, _)| o)),
    };
    let art = temen_snapshot::freeze(&inst, snap.bytes(), &h).expect("freeze");
    full_regions_written(&art);
    thaws_on_every_engine(&inst, &art, &want);
}

/// What a freeze of [`full_region_src`] wrote: both extents at their regions' ends.
fn full_regions_written(art: &[u8]) {
    let written = decode_residue(art).expect("decode");
    let ctx = |sp: u64| (sp - 1 - TEST_ARENA.base) / TEST_ARENA.stride;
    let ends = |sp: u64| TEST_ARENA.region_base(ctx(sp) as usize) + TEST_ARENA.stride;
    assert_eq!(written.vcpus.len(), 1, "the thread rides");
    let v = written.vcpus[0].shadow_sp;
    assert_eq!(v, ends(v), "the thread's chain fills its region");
    let root = written.root_sp.expect("a root extent");
    assert_eq!(
        root,
        TEST_ARENA.region_base(0) + TEST_ARENA.stride,
        "and the root's"
    );
}

/// A JIT thaw that refuses the residue (here: the run is not of the module the artifact was restored
/// against) declines and leaves the residue where it was, for the engine its caller declines to. On
/// `f2aff3d` the JIT took it first and dropped it on the refusal (#3034).
#[test]
fn a_refused_jit_thaw_leaves_the_residue_for_the_next_engine() {
    let inst = instrument(THREADS);
    let art = frozen(&inst);
    let mut h = Host::new();
    let mut win = temen_snapshot::restore(&art, &inst, &mut h).expect("restore");
    begin_thaw(&mut win, TEST_ARENA, 0);
    h.set_durable(true);
    let before = (
        h.residue().vcpus.clone(),
        h.residue().root_sp,
        h.residue().children.clone(),
    );
    assert_eq!(before.0.len(), 1, "the parked child rides");

    let other = instrument(FIBERS); // no `thread.spawn`: its runs make no vCPU
    let r = temen_run::jit_cap_run(
        &other,
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
    let r = h.residue();
    assert_eq!(
        (r.vcpus.clone(), r.root_sp, r.children.clone()),
        before,
        "and the residue is where it was"
    );
}

/// A kind of record no code of the module can make is refused (#3034): a fiber in a module
/// with no `cont.new`, a vCPU in one with no `thread.spawn`, a completed detached child in one with no
/// `Instantiator` call. On `f2aff3d` each restored, and the engines diverged: the tree-walker
/// re-created the fiber or vCPU, the JIT, standing up no runtime for a kind its module has no site for,
/// skipped it.
#[test]
fn a_record_kind_the_module_cannot_make_is_refused_at_restore() {
    let threads = instrument(THREADS);
    let fibers = instrument(FIBERS);
    let fiber = |r: &mut ResidueEdit| {
        r.fibers.push(FrozenFiber {
            slot: 0,
            func: 1,
            arg0: 0,
            shadow_sp: TEST_ARENA.frame_base(1),
            generation: 1,
            consumed: false,
        })
    };
    assert_refused(
        "a fiber, no cont.new",
        &threads,
        &bent(&frozen(&threads), fiber),
    );
    let fart = frozen(&fibers);
    let vcpu = {
        let t = decode_residue(&frozen(&threads)).expect("decode");
        (t.vcpus, t.children, t.root_sp)
    };
    assert_refused(
        "a vCPU, no thread.spawn",
        &fibers,
        &bent(&fart, |r| {
            r.vcpus = vcpu.0.clone();
            r.children = vcpu.1.clone();
            r.root_sp = vcpu.2;
        }),
    );
    assert_refused(
        "a detached child, no Instantiator",
        &fibers,
        &bent(&fart, |r| {
            r.children = vec![1];
            r.detached.push(temen_ir::thaw::FrozenDetached {
                parent_task: 0,
                slot: 0,
                completed_result: Ok(7),
            });
        }),
    );
}

/// A root that may spawn a detached child (an `Instantiator` call on a path it never takes).
const SPAWNS: &str = r#"
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
"#;

/// The pass checks a completed child's trap code itself, so an embedder's residue meets the rule the
/// decoder's does (#3034). On `f2aff3d` a trap code of 0 was admitted here, and on a thaw it
/// fired the tree-walker's `joined` assert and read as success on the JIT.
#[test]
fn a_trap_code_no_trap_has_is_refused_on_the_embedders_route() {
    let inst = instrument(SPAWNS);
    let completed = |code| temen_interp::Residue {
        children: vec![1],
        detached: vec![temen_ir::thaw::FrozenDetached {
            parent_task: 0,
            slot: 0,
            completed_result: Err(code),
        }],
        ..Default::default()
    };
    let mut h = Host::new();
    assert!(
        h.admit_thaw(&inst, completed(temen_ir::trap_code::THREAD_FAULT))
            .is_ok(),
        "a trap a join answers"
    );
    for code in [
        0,
        temen_ir::trap_code::DOMAIN_DONE,
        temen_ir::trap_code::THREAD_FAULT | 1 << 32,
    ] {
        let mut h = Host::new();
        assert!(
            h.admit_thaw(&inst, completed(code)).is_err(),
            "trap code {code:#x}"
        );
        assert!(h.residue().is_empty(), "and nothing is seeded");
    }
}

/// Root spawns a thread that raises the word at 66000 and returns 7; the root waits for the word,
/// resumes a fiber twice (the freeze triggers), then joins the thread: `7 * 1000 + 9`.
const FINISHED_UNJOINED: &str = r#"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vaddr vexp vinf
  vf = ref.func 2
  vss = i64.const 4096
  vk = cont.new vf vss
  va1, vb1 = cont.resume vk vz
  vnine = i64.const 9
  va2, vb2 = cont.resume vk vnine
  vst, vr = thread.join vt
  vth = i64.const 1000
  vm = i64.mul vr vth
  vs = i64.add vm vb2
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vaddr = i64.const 66000
  vone = i32.const 1
  i32.store vaddr vone
  vn = atomic.notify vaddr vone
  vr = i64.const 7
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 5
  v3 = suspend v2
  v4 = suspend v3
  return v4
  }
}
"#;

/// A thread that finished unjoined rides a freeze as its result, is delivered at the thaw, and rides
/// the next freeze again if no join takes it first: frozen at the first resume, thawed on the
/// tree-walker or the JIT, frozen again at the second, and thawed, the join still reads 7 (#3103).
/// On `f2aff3d` the second freeze, on either engine, carried nothing for it, and the
/// twice-thawed run returned 9.
#[test]
fn a_finished_unjoined_thread_rides_a_second_freeze_after_a_thaw() {
    let inst = instrument(FINISHED_UNJOINED);
    let want = Ok(vec![Value::I64(7009)]);
    assert_eq!(
        interp_run(
            &inst,
            &init_durable_window(WINDOW, TEST_ARENA),
            &mut Host::new()
        ),
        want,
        "uninterrupted"
    );
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, TEST_ARENA, 1);
    let mut h = Host::new();
    let (r, snap) = {
        h.set_durable(true);
        let mut fuel = 10_000_000u64;
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h)
    };
    assert!(r.is_ok(), "the first freeze: {r:?}");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("freeze");
    let finished = |art: &[u8]| {
        decode_residue(art)
            .expect("decode")
            .vcpus
            .iter()
            .map(|v| v.completed_result)
            .collect::<Vec<_>>()
    };
    assert_eq!(finished(&art), [Some(7)], "the thread rides as its result");

    // Thawed and frozen again before the join, on each engine that re-creates threads.
    let rearmed = || {
        let mut h = Host::new();
        let mut w = temen_snapshot::restore(&art, &inst, &mut h).expect("restore");
        begin_thaw(&mut w, TEST_ARENA, 0);
        arm_freeze_after(&mut w, TEST_ARENA, 2);
        h.set_durable(true);
        (h, w)
    };
    let (mut h, w) = rearmed();
    let mut fuel = 10_000_000u64;
    let (r, snap) = run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &w, SIZE_LOG2, &mut h);
    assert!(r.is_ok(), "the tree-walker's second freeze: {r:?}");
    let tw = temen_snapshot::freeze(&inst, &snap, &h).expect("freeze");
    let (mut h, w) = rearmed();
    let snap = match temen_run::jit_cap_run(
        &inst,
        0,
        &[],
        &MemLayout::image(w),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    ) {
        Ok((JitOutcome::Returned(_), snap)) => snap,
        other => panic!("the JIT's second freeze: {:?}", other.map(|(o, _)| o)),
    };
    let jit = temen_snapshot::freeze(&inst, snap.bytes(), &h).expect("freeze");

    for (engine, art) in [("tree-walker", tw), ("JIT", jit)] {
        assert_eq!(
            finished(&art),
            [Some(7)],
            "{engine}: it rides the second freeze"
        );
        thaws_on_every_engine(&inst, &art, &want);
    }
}

/// Makes `n` fibers, resuming each once (each suspends, staying live), and returns how many it made.
fn many_fibers(n: i64) -> String {
    format!(
        r#"
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  br 1(vz)
}}
block 1 (vi: i64) {{
  vn = i64.const {n}
  vc = i64.lt_s vi vn
  br_if vc 2(vi) 3(vi)
}}
block 2 (vj: i64) {{
  vf = ref.func 1
  vss = i64.const 4096
  vk = cont.new vf vss
  vz2 = i64.const 0
  va, vb = cont.resume vk vz2
  vone = i64.const 1
  vj1 = i64.add vj vone
  br 1(vj1)
}}
block 3 (vr: i64) {{
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = i64.const 5
  v3 = suspend v2
  return v3
  }}
}}
"#
    )
}

/// A durable run's fiber needs a shadow context of its own: the arena holds ten fiber contexts, so the
/// eleventh live `cont.new` faults, on every engine (#3101). On `f2aff3d` the bytecode engine
/// made it (and its freeze wrote an artifact a restore refuses) while the tree-walker and the JIT
/// answered `FiberFault`.
#[test]
fn a_durable_runs_eleventh_live_fiber_faults_alike_on_every_engine() {
    for (n, want) in [(10, Ok(10)), (11, Err(temen_interp::Trap::FiberFault))] {
        let inst = instrument(&many_fibers(n));
        let win = init_durable_window(WINDOW, TEST_ARENA);
        let tw = interp_run(&inst, &win, &mut Host::new());
        let mut h = Host::new();
        h.set_durable(true);
        let mut fuel = 10_000_000u64;
        let bc = temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
            &inst,
            0,
            &[],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        )
        .map(|c| c.0);
        let mut h = Host::new();
        h.set_durable(true);
        let jit = temen_run::jit_cap_run(
            &inst,
            0,
            &[],
            &MemLayout::image(win),
            SIZE_LOG2,
            0,
            &mut h,
            None,
        )
        .map(|(o, _)| o);
        let want_tw = want.clone().map(|v| vec![Value::I64(v)]);
        assert_eq!(tw, want_tw, "tree-walker, {n} fibers");
        assert_eq!(bc, Some(want_tw), "bytecode, {n} fibers");
        match (want, jit) {
            (Ok(v), Ok(JitOutcome::Returned(r))) => assert_eq!(r, vec![v], "JIT"),
            (Err(_), Ok(JitOutcome::Trapped(t))) => {
                assert_eq!(format!("{t:?}"), "FiberFault", "JIT")
            }
            (_, other) => panic!("JIT, {n} fibers: {other:?}"),
        }
    }
}

// ---- #3034: a freeze admits its own residue, through the one hand-off both engines take
// (`Host::leave_freeze`). Each context's shadow-SP word is guest memory, which the transform's push
// check bounds only at that context's own unwind.

/// Root spawns a thread, writes `val` at `addr`, raises the word at 66000 and joins it; the thread
/// waits for the word, then resumes a fiber, which triggers the freeze.
fn scribbling(addr: u64, val: i64) -> String {
    format!(
        "func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vsa = i64.const {addr}
  vsv = i64.const {val}
  i64.store vsa vsv
  vaddr = i64.const 66000
  vone = i32.const 1
  i32.store vaddr vone
  vn2 = atomic.notify vaddr vone
  vst, vres = thread.join vt
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vf = ref.func 2
  vss = i64.const 4096
  vk = cont.new vf vss
  vz2 = i64.const 0
  va, vb = cont.resume vk vz2
  return vb
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  v2 = i64.const 5
  v3 = suspend v2
  return v3
  }}
}}
"
    )
}

/// A root that writes its own shadow-SP word below its frames traps at its own unwind, at the push
/// check (`Unreachable`), while the thread it spawned has already unwound. The run's outcome is the
/// trap and nothing is carried, on both engines. On `f2aff3d` the tree-walker kept the thread's and
/// the fiber's records (a `freeze` then wrote an artifact the restore refuses), and the JIT reached
/// a debug assert in its hand-off (`Unsupported` in release, after the run had unwound).
///
/// What a `freeze` of that host writes: the window as the guest left it and its handles, with no
/// residue. Its restore seeds nothing, and its thaw rewinds the root from the word the guest wrote,
/// which faults (`MemoryFault`).
#[test]
fn a_root_that_writes_its_own_shadow_sp_word_traps_and_carries_nothing() {
    let inst = instrument(&scribbling(TEST_ARENA.region_base(0), 100));
    let armed = || {
        let mut w = init_durable_window(WINDOW, TEST_ARENA);
        arm_freeze_after(&mut w, TEST_ARENA, 1);
        w
    };
    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 10_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &armed(), SIZE_LOG2, &mut h);
    assert_eq!(r, Err(temen_interp::Trap::Unreachable), "tree-walker");
    assert!(h.residue().is_empty(), "the tree-walker carries nothing");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("the window and handles");
    assert!(
        decode_residue(&art).expect("decode").is_empty(),
        "no residue"
    );
    let mut th = Host::new();
    let mut w = temen_snapshot::restore(&art, &inst, &mut th).expect("restores");
    assert!(th.residue().is_empty(), "and seeds none");
    begin_thaw(&mut w, TEST_ARENA, 0);
    assert_eq!(
        interp_run(&inst, &w, &mut th),
        Err(temen_interp::Trap::MemoryFault),
        "its thaw rewinds the root from the word the guest wrote"
    );

    let mut h = Host::new();
    h.set_durable(true);
    let r = temen_run::jit_cap_run(
        &inst,
        0,
        &[],
        &MemLayout::image(armed()),
        SIZE_LOG2,
        0,
        &mut h,
        None,
    );
    assert!(
        matches!(
            r,
            Ok((JitOutcome::Trapped(temen_jit::TrapKind::Unreachable), _))
        ),
        "JIT: {:?}",
        r.as_ref().map(|(o, _)| o)
    );
    assert!(h.residue().is_empty(), "the JIT carries nothing");
}

/// A residue the pass refuses, handed off by a freeze, ends the run as the push check's own trap and
/// carries nothing: the case where a vCPU rewrites a sibling's word after the sibling unwound (every
/// unwind clean, the extent bad), which no guest reaches deterministically, handed to the hand-off
/// directly. A residue a run makes is admitted, and one from a run that trapped is dropped.
#[test]
fn a_freezes_refused_residue_is_unreachable_and_carries_nothing() {
    let inst = instrument(THREADS);
    let art = frozen(&inst);
    let real = || {
        let mut h = Host::new();
        temen_snapshot::restore(&art, &inst, &mut h).expect("restore");
        h.take_residue()
    };
    let mut h = Host::new();
    assert_eq!(h.leave_freeze(&inst, 0, real(), false), Ok(()));
    assert_eq!(h.residue().vcpus.len(), 1, "admitted");

    let mut h = Host::new();
    let mut bad = real();
    bad.vcpus[0].shadow_sp = TEST_ARENA.region_base(3) + 8; // in a context's header
    assert_eq!(
        h.leave_freeze(&inst, 0, bad, false),
        Err(temen_interp::Trap::Unreachable)
    );
    assert!(h.residue().is_empty(), "nothing is carried");

    let mut h = Host::new();
    assert_eq!(h.leave_freeze(&inst, 0, real(), true), Ok(()));
    assert!(h.residue().is_empty(), "a run that trapped leaves none");
}

/// #3034 — Section 5 gives the restored domain the `Jit` domains its artifact names, whether or not
/// the restoring host grants one (#3096 A1 ties it to the re-granted grant). With one, the pass admits
/// every record kind, since code installed through it may make any (the record-kind rule above): a
/// fiber in a module with no `cont.new` restores. The tree-walker re-creates it; the JIT, which stood
/// up no fiber runtime for the module (no `cont.*`, no fiber-hosting grant), declines the residue
/// whole and leaves it, where on `f2aff3d` it would have skipped the fiber.
#[test]
fn a_jit_domain_from_section_5_admits_every_kind_and_the_jit_declines_what_it_cannot_host() {
    let inst = instrument(THREADS);
    let art = {
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        write_state(&mut win, TEST_ARENA, STATE_UNWINDING);
        let mut h = Host::new();
        h.set_durable(true);
        h.grant_jit_with_table(Some(SIZE_LOG2), 0);
        let mut fuel = 1_000_000u64;
        let (r, snap) =
            run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
        assert!(r.is_ok(), "the freeze: {r:?}");
        temen_snapshot::freeze(&inst, &snap, &h).expect("freeze")
    };
    let grafted = bent(&art, |r| {
        r.fibers.push(FrozenFiber {
            slot: 0,
            func: 1,
            arg0: 0,
            shadow_sp: TEST_ARENA.frame_base(1),
            generation: 1,
            consumed: false,
        })
    });
    // The restoring host grants no `Jit`; Section 5 gives it the domain all the same.
    let mut h = Host::new();
    let mut win = temen_snapshot::restore(&grafted, &inst, &mut h).expect("admitted");
    assert!(h.installs_code(), "from Section 5");
    assert_eq!(h.residue().fibers.len(), 1);
    begin_thaw(&mut win, TEST_ARENA, 0);
    h.set_durable(true);
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
        matches!(
            r,
            Err(JitError::Unsupported(
                "durable JIT thaw: a record kind this run stands up no runtime for"
            ))
        ),
        "the JIT declines: {:?}",
        r.as_ref().map(|(o, _)| o)
    );
    assert_eq!(h.residue().fibers.len(), 1, "and leaves the residue");
}
