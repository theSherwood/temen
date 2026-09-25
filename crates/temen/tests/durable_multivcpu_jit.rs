//! Phase-3 slice 3.3 (DURABILITY.md §12.8): the **JIT** freezes *and thaws* a *multi-vCPU* durable
//! domain exactly as the interpreter does. A durable run is **single-worker** — the interp serializes
//! onto one cooperative worker, and the JIT (whose vCPUs are 1:1 OS threads) instead runs each
//! `thread.spawn`ed child **inline** (deferred during a freeze until the root unwinds; re-attached +
//! run before the root re-enters on a thaw). The one shared set of durable control words (state +
//! active shadow-SP) is never raced; each child unwinds into its own top-down shadow context.
//!
//! Pinned here:
//!   - **freeze** (`jit_freezes_a_spawned_vcpu_matching_interp`): freezing the *same* instrumented
//!     two-vCPU module on both backends flattens the child into a **byte-identical durable reserve**
//!     and exports the **same `FrozenVCpu` residue** — the cross-backend §7 property, extended to vCPUs.
//!   - **thaw** (`jit_thaws_its_own_multivcpu_freeze`): a JIT freeze → JIT thaw on an *advanced* clock
//!     reproduces the uninterrupted result (reloads the saved reads, never re-issues) — the §12.6
//!     equivalence, multi-vCPU, on the JIT.
//!   - **cross-backend thaw** (`interp_frozen_multivcpu_thaws_on_the_jit`): an interpreter-frozen
//!     domain thaws on the JIT to the uninterrupted result.
//!   - **child-owned fibers** (`jit_freezes_and_thaws_a_child_owned_fiber_matching_interp`, slice 3.4):
//!     a spawned child that owns a fiber flattens it with the child's *own* `freeze_drive`,
//!     byte-identical to the interp, and thaws it back.
//!
//! Native stack switching (for the inline child's guarded run) exists on x86-64 unix, aarch64 unix,
//! and x86-64 Windows; elsewhere the JIT bails `Unsupported` on `thread.*`/`cont.*`, so this is gated.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use core::ffi::c_void;
use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{Memory, Module};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};
use temen_jit::{
    compile_and_run_durable, DurableResidue, DurableRun, FrozenFiber as JitFiber,
    FrozenVCpu as JitVCpu, JitError, JitOutcome,
};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

// Same module as the interpreter's `two_vcpu_domain_freezes_and_thaws`: the root stashes the clock
// handle at a fixed guest byte (above the durable reserve), spawns a child over the shared window
// running it, reads the clock once, then joins the child and sums. The child loads the handle, reads
// the clock once, returns clock + 10.
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = thread.spawn 1 v2 v3
  v5 = i32.const 0
  v6 = call.cap 2 0 (i32) -> (i64) v0 (v5)
  v7 = thread.join v4
  v8 = i64.add v6 v7
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = call.cap 2 0 (i32) -> (i64) v3 (v4)
  v6 = i64.const 10
  v7 = i64.add v5 v6
  return v7
  }
}
"#;

fn instrument() -> Module {
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented multi-vCPU IR verifies");
    inst
}

/// The JIT runs a spawned child **inline** during a freeze and flattens it into the same durable
/// reserve, exporting the same `FrozenVCpu` residue as the interpreter.
#[test]
fn jit_freezes_a_spawned_vcpu_matching_interp() {
    let inst = instrument();

    // Interp freeze: UNWINDING from the start (single-worker). The root runs (spawns the child, reads
    // the clock → 42), unwinds at its poll; then the child runs (reads the clock → 43), unwinds into
    // its own top-down region. Capture the window image + the child's residue.
    let (ifrozen, isnap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "interp freeze returns a placeholder: {r:?}");
        (h.frozen_vcpus().to_vec(), snap)
    };
    assert_eq!(ifrozen.len(), 1, "interp captured the spawned vCPU");
    assert_eq!(ifrozen[0].task, 1, "the child is task 1 (root is task 0)");

    // JIT freeze: the child runs inline (single-worker) and unwinds into its own region. Skip on
    // Unsupported / host allocation pressure (mirroring the other cross-backend JIT durable tests).
    let mut jhost = Host::new();
    jhost.set_durable(true);
    jhost.clock_ns = 42;
    let clk = jhost.grant_clock();
    let mut jwin = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut jwin, STATE_UNWINDING);
    let (
        jout,
        jsnap,
        DurableResidue {
            fibers: jfibers,
            vcpus: jvcpus,
            ..
        },
    ) = match compile_and_run_durable(
        &inst,
        0,
        &[clk as i64],
        &jwin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => {
            panic!("JIT failed to compile a verified multi-vCPU module: {e:?}\n{inst:#?}")
        }
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze returns a placeholder, got {jout:?}"
    );
    assert!(jfibers.is_empty(), "no fibers in this module");

    // (1) The two backends flatten the child into a byte-identical durable reserve (control words +
    // both contexts' shadow regions): the same emitted IR spills the same values to the same offsets.
    let reserve = TEST_ARENA.end as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT freeze the spawned vCPU into a byte-identical durable reserve"
    );

    // (2) The exported `FrozenVCpu` residue matches field-for-field (task id, entry func, spawn args,
    // flattened shadow-SP) — so a JIT-frozen multi-vCPU domain re-attaches its children exactly as an
    // interp-frozen one does.
    assert_eq!(jvcpus.len(), 1, "the JIT exported the spawned vCPU");
    assert_eq!(jvcpus[0].task, ifrozen[0].task, "same task id");
    assert_eq!(jvcpus[0].func, ifrozen[0].func, "same entry func");
    assert_eq!(jvcpus[0].args, ifrozen[0].args, "same spawn args");
    assert_eq!(
        jvcpus[0].shadow_sp, ifrozen[0].shadow_sp,
        "same flattened shadow-SP extent"
    );
}

/// The JIT **thaws** a multi-vCPU domain it froze: the spawned child is re-attached + run (rewinds
/// from its restored extent, runs forward to completion), and the root re-enters under `REWINDING` and
/// resolves its `thread.join`. Thawing on a host whose clock has *advanced* must reproduce the
/// uninterrupted result — both vCPUs **reload** their saved clock reads (42, 43), they do not re-issue
/// them (which would read the advanced clock) — the §12.6 freeze/thaw equivalence, multi-vCPU, on the JIT.
#[test]
fn jit_thaws_its_own_multivcpu_freeze() {
    let inst = instrument();

    // Uninterrupted baseline: clock 42 → reads {42, 43}; result = 42 + (43 + 10) = 95 (order-invariant).
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW, TEST_ARENA),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(95)], "uninterrupted: 42 + (43 + 10)");

    // JIT freeze (UNWINDING): capture the window image, the child residue, and the root's extent.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    fhost.clock_ns = 42;
    let fclk = fhost.grant_clock();
    let mut fwin = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut fwin, STATE_UNWINDING);
    let (
        fout,
        fsnap,
        DurableResidue {
            vcpus: fvcpus,
            root_sp: froot_sp,
            ..
        },
    ) = match compile_and_run_durable(
        &inst,
        0,
        &[fclk as i64],
        &fwin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut fhost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT freeze failed: {e:?}\n{inst:#?}"),
    };
    assert!(
        matches!(fout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );
    assert_eq!(fvcpus.len(), 1, "the freeze captured the spawned child");
    assert_eq!(
        fhost.clock_ns, 44,
        "the freeze ran both clock reads once (42, 43 → 44)"
    );

    // JIT thaw on a host whose clock has *advanced* to 44: re-attach the child + restore the root's
    // extent, re-enter under REWINDING. Reload (42, 43) → 95; a re-issue would read {44, 45} → 99.
    let mut twin = fsnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 44;
    let tclk = thost.grant_clock();
    assert_eq!(tclk, fclk, "fresh host re-grants the same clock handle");
    let (tout, _tsnap, DurableResidue { vcpus: tvcpus, .. }) = match compile_and_run_durable(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                vcpus: fvcpus.to_vec(),
                root_sp: froot_sp,
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT thaw failed: {e:?}\n{inst:#?}"),
    };
    assert!(tvcpus.is_empty(), "a thaw re-freezes nothing");
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![95],
            "thawed two-vCPU domain reloads the saved clock reads (95), not re-issued ones (99)"
        ),
        other => panic!("thaw did not return cleanly: {other:?}"),
    }
}

/// An **interpreter-frozen** multi-vCPU domain **thaws on the JIT** and reproduces the uninterrupted
/// result — crossing the backend boundary. The interp's `FrozenVCpu` residue + root extent drive the
/// JIT thaw directly (the residue is a portable host-side record), so the re-attached child reloads its
/// saved clock read on the JIT just as it would on the interp.
#[test]
fn interp_frozen_multivcpu_thaws_on_the_jit() {
    let inst = instrument();

    // Interp freeze (UNWINDING): capture the window image, the child residue, and the root's extent.
    let (ivcpus, iroot_sp, isnap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "interp freeze placeholder: {r:?}");
        (
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
        )
    };
    assert_eq!(ivcpus.len(), 1, "interp captured the spawned child");

    // Bridge the interp residue to the JIT (same fields), then thaw on the JIT with an advanced clock.
    let seed: Vec<JitVCpu> = ivcpus
        .iter()
        .map(|v| JitVCpu {
            task: v.task,
            parent_task: v.parent_task,
            slot: v.slot,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
            completed_result: None,
        })
        .collect();
    let mut twin = isnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 44;
    let tclk = thost.grant_clock();
    let (tout, ..) = match compile_and_run_durable(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                vcpus: seed.to_vec(),
                root_sp: Some(iroot_sp),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT thaw of interp freeze failed: {e:?}\n{inst:#?}"),
    };
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![95],
            "interp-frozen multi-vCPU domain thaws on the JIT to the uninterrupted result (95)"
        ),
        other => panic!("cross-backend thaw did not return cleanly: {other:?}"),
    }
}

// Slice 3.4 — a spawned **child that owns a fiber**. The root reads the clock + spawns/joins; the
// child's first may-suspend op is its own `cont.resume`, so its fiber parks (yielding 5) before the
// child unwinds. The JIT must flatten the child's fiber with the child's own `freeze_drive` (its root
// drive ran before the child existed) → byte-identical durable reserve + residue vs the interp, and a
// thaw that re-seeds the child's fiber and reproduces the uninterrupted result. (Mirrors the interp's
// `temen-durable/tests/multivcpu.rs::child_owns_fiber_through_freeze_thaw`.)
const SRC_CHILD_FIBER: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 0
  v3 = thread.spawn 1 v1 v2
  v4 = i32.const 0
  v5 = call.cap 2 0 (i32) -> (i64) v0 (v4)
  v6 = thread.join v3
  v7 = i64.add v5 v6
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = ref.func 2
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
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
  }
}
"#;

fn instrument_child_fiber() -> Module {
    let mut m = temen_text::parse_module(SRC_CHILD_FIBER).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented child-fiber IR verifies");
    inst
}

/// The JIT flattens a **child-owned** fiber on freeze (the child's own `freeze_drive`), byte-identical
/// to the interp, and thaws it back to the uninterrupted result.
#[test]
fn jit_freezes_and_thaws_a_child_owned_fiber_matching_interp() {
    let inst = instrument_child_fiber();

    // Uninterrupted baseline: root clock 42 + child (fiber 5 + 100) = 42 + 105 = 147.
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW, TEST_ARENA),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(147)], "uninterrupted: 42 + (5 + 100)");

    // Interp freeze: capture window + residues (the child's fiber must be flattened).
    let (ifibers, ivcpus, _iroot_sp, isnap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "interp freeze placeholder: {r:?}");
        (
            h.frozen_fibers().to_vec(),
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent recorded"),
            snap,
        )
    };
    assert_eq!(ifibers.len(), 1, "interp flattened the child's fiber");
    assert_eq!(ivcpus.len(), 1, "interp captured the child vCPU");

    // JIT freeze.
    let mut jhost = Host::new();
    jhost.set_durable(true);
    jhost.clock_ns = 42;
    let clk = jhost.grant_clock();
    let mut jwin = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut jwin, STATE_UNWINDING);
    let (
        jout,
        jsnap,
        DurableResidue {
            fibers: jfibers,
            vcpus: jvcpus,
            root_sp: jroot_sp,
            ..
        },
    ) = match compile_and_run_durable(
        &inst,
        0,
        &[clk as i64],
        &jwin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT freeze of child-owned fiber failed: {e:?}\n{inst:#?}"),
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );

    // (1) Byte-identical durable reserve (control words + every context's flattened region): the
    // child's fiber (ctx 1) + the child vCPU (top-down ctx) flatten to the same bytes on both backends.
    let reserve = TEST_ARENA.end as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT flatten the child-owned fiber into a byte-identical durable reserve"
    );
    // (2) The JIT exported the child's fiber + the child vCPU, matching the interp field-for-field.
    assert_eq!(jfibers.len(), 1, "JIT flattened the child's fiber");
    assert_eq!(jvcpus.len(), 1, "JIT captured the child vCPU");
    assert_eq!(jfibers[0].slot, ifibers[0].slot, "same fiber slot");
    assert_eq!(jfibers[0].func, ifibers[0].func, "same fiber func");
    assert_eq!(
        jfibers[0].shadow_sp, ifibers[0].shadow_sp,
        "same fiber extent"
    );
    assert_eq!(jvcpus[0].task, ivcpus[0].task, "same child task");
    assert_eq!(
        jvcpus[0].shadow_sp, ivcpus[0].shadow_sp,
        "same child extent"
    );

    // (3) Thaw on the JIT with an advanced clock: re-seed the child's fiber + the child, restore the
    // root extent, re-enter under REWINDING → reload (147), not re-issue.
    let seed_fibers: Vec<JitFiber> = jfibers.clone();
    let seed_vcpus: Vec<JitVCpu> = jvcpus.clone();
    let mut twin = jsnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 99;
    let tclk = thost.grant_clock();
    let (tout, ..) = match compile_and_run_durable(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                fibers: seed_fibers.to_vec(),
                vcpus: seed_vcpus.to_vec(),
                root_sp: jroot_sp,
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT thaw of child-owned fiber failed: {e:?}\n{inst:#?}"),
    };
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![147],
            "thawed child-owned-fiber domain reloads (147), not a re-issued clock"
        ),
        other => panic!("child-fiber thaw did not return cleanly: {other:?}"),
    }
}

// Slice 3.4 — **nested spawns** on the JIT: root → child → grandchild. The child `thread.spawn`s the
// grandchild during the freeze (deferred, then drained by the loop in `drive_frozen_spawns`); the
// grandchild's guest handle is its index in the *child's* per-vCPU table (`0`), byte-identical to the
// interp's per-vCPU `threads`. Thaw rebuilds the per-parent join tables and runs children before
// parents so each join resolves on the single worker. (Mirrors the interp's
// `temen-durable/tests/multivcpu.rs::nested_spawn_tree_freezes_and_thaws`.)
const SRC_NESTED: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = i64.const 0
  v4 = thread.spawn 1 v2 v3
  v5 = i32.const 0
  v6 = call.cap 2 0 (i32) -> (i64) v0 (v5)
  v7 = thread.join v4
  v8 = i64.add v6 v7
  return v8
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i64.const 0
  v5 = i64.const 0
  v6 = thread.spawn 2 v4 v5
  v7 = i32.const 0
  v8 = call.cap 2 0 (i32) -> (i64) v3 (v7)
  v9 = thread.join v6
  v10 = i64.add v8 v9
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65536
  v3 = i32.load v2
  v4 = i32.const 0
  v5 = call.cap 2 0 (i32) -> (i64) v3 (v4)
  return v5
  }
}
"#;

fn instrument_nested() -> Module {
    let mut m = temen_text::parse_module(SRC_NESTED).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented nested IR verifies");
    inst
}

/// The JIT freezes a 3-level vCPU tree byte-identically to the interp (incl. the nested grandchild's
/// per-vCPU handle) and thaws it back to the uninterrupted result.
#[test]
fn jit_freezes_and_thaws_a_nested_tree_matching_interp() {
    let inst = instrument_nested();

    // Baseline: clock 42 → the three reads sum to 42 + 43 + 44 = 129 (order-invariant).
    let want = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut fuel = 1_000_000u64;
        let (r, _) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &init_durable_window(WINDOW, TEST_ARENA),
            SIZE_LOG2,
            &mut h,
        );
        r.expect("uninterrupted")
    };
    assert_eq!(want, vec![Value::I64(129)], "uninterrupted: 42 + 43 + 44");

    // Interp freeze: capture window + residues (child task 1 parent 0; grandchild task 2 parent 1).
    let (ivcpus, iroot_sp, isnap) = {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let mut win = init_durable_window(WINDOW, TEST_ARENA);
        write_state(&mut win, STATE_UNWINDING);
        let mut fuel = 1_000_000u64;
        let (r, snap) = run_capture_reserved_with_host(
            &inst,
            0,
            &[Value::I32(clk)],
            &mut fuel,
            &win,
            SIZE_LOG2,
            &mut h,
        );
        assert!(r.is_ok(), "interp freeze placeholder: {r:?}");
        (
            h.frozen_vcpus().to_vec(),
            h.frozen_root_sp().expect("root extent"),
            snap,
        )
    };
    assert_eq!(ivcpus.len(), 2, "interp captured child + grandchild");

    // JIT freeze.
    let mut jhost = Host::new();
    jhost.set_durable(true);
    jhost.clock_ns = 42;
    let clk = jhost.grant_clock();
    let mut jwin = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut jwin, STATE_UNWINDING);
    let (
        jout,
        jsnap,
        DurableResidue {
            vcpus: jvcpus,
            root_sp: jroot_sp,
            ..
        },
    ) = match compile_and_run_durable(
        &inst,
        0,
        &[clk as i64],
        &jwin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT freeze of nested tree failed: {e:?}\n{inst:#?}"),
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "freeze placeholder"
    );

    // (1) Byte-identical durable reserve — incl. the grandchild's spilled per-vCPU handle (= 0 in the
    // child's namespace, not a global running index).
    let reserve = TEST_ARENA.end as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT freeze the nested tree into a byte-identical durable reserve"
    );
    // (2) The JIT residue matches the interp field-for-field, parent_task included.
    let mut iv = ivcpus.clone();
    iv.sort_by_key(|v| v.task);
    let mut jv = jvcpus.clone();
    jv.sort_by_key(|v| v.task);
    assert_eq!(jv.len(), 2, "JIT captured child + grandchild");
    for (j, i) in jv.iter().zip(&iv) {
        assert_eq!(j.task, i.task, "same task");
        assert_eq!(j.parent_task, i.parent_task, "same parent_task");
        assert_eq!(j.func, i.func, "same func");
        assert_eq!(j.shadow_sp, i.shadow_sp, "same extent");
    }
    assert_eq!(jroot_sp, Some(iroot_sp), "same root extent");

    // (3) Thaw on the JIT with an advanced clock: rebuild the per-parent join tables, run children
    // before parents, reload all three clock reads → 129 (a re-issue would be 99+100+101 = 300).
    let mut twin = jsnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 99;
    let tclk = thost.grant_clock();
    let (tout, ..) = match compile_and_run_durable(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                vcpus: jvcpus.to_vec(),
                root_sp: jroot_sp,
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT thaw of nested tree failed: {e:?}\n{inst:#?}"),
    };
    match tout {
        JitOutcome::Returned(rs) => assert_eq!(
            rs,
            vec![129],
            "thawed nested tree reloads its saved clock reads (129), not re-issued ones (300)"
        ),
        other => panic!("nested thaw did not return cleanly: {other:?}"),
    }
}

// #1584 — a child **parked** in an infinite `atomic.wait` when the freeze reaches it. The run starts
// `UNWINDING`; the root spawns the child and joins it; the child's first act is the wait. The two
// engines used to disagree, and the oracle was the one that was wrong: the JIT's futex park observes a
// freeze on its own recheck cadence and unwinds, while the interpreter let the wait park under
// `UNWINDING`, nothing re-admitted it, and the deadlock check faulted the whole run `ThreadFault`. The
// in-flight freeze now re-admits it, and both engines capture it at its wait's re-issue point.
const SRC_FUTEX_PARKED_CHILD: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vr = thread.join vt
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

#[test]
fn jit_and_interp_freeze_a_futex_parked_child_identically() {
    let mut m = temen_text::parse_module(SRC_FUTEX_PARKED_CHILD).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR verifies");
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);

    let mut h = Host::new();
    h.set_durable(true);
    let mut fuel = 1_000_000u64;
    let (r, isnap) =
        run_capture_reserved_with_host(&inst, 0, &[], &mut fuel, &win, SIZE_LOG2, &mut h);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the oracle freezes it (it used to fault)"
    );
    let (ivcpus, iroot_sp) = (
        h.frozen_vcpus().to_vec(),
        h.frozen_root_sp().expect("root extent recorded"),
    );
    assert_eq!(ivcpus.len(), 1, "the parked child is in the oracle's cut");

    let mut jhost = Host::new();
    jhost.set_durable(true);
    let (
        jout,
        jsnap,
        DurableResidue {
            fibers: jfibers,
            vcpus: jvcpus,
            ..
        },
    ) = match compile_and_run_durable(
        &inst,
        0,
        &[],
        &win,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut jhost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
            ..Default::default()
        },
    ) {
        Ok(t) => t,
        Err(JitError::Unsupported(_)) => return,
        Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return,
        Err(e) => panic!("JIT failed on a verified module: {e:?}"),
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze returns a placeholder, got {jout:?}"
    );
    assert!(jfibers.is_empty(), "no fibers in this module");

    // The same cut, byte for byte: the reserve, and the child's re-attach residue field for field.
    let reserve = TEST_ARENA.end as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "interp/JIT freeze the parked child into a byte-identical durable reserve"
    );
    assert_eq!(jvcpus.len(), 1, "the JIT exported the parked child");
    assert_eq!(
        (
            jvcpus[0].task,
            jvcpus[0].func,
            &jvcpus[0].args,
            jvcpus[0].shadow_sp
        ),
        (
            ivcpus[0].task,
            ivcpus[0].func,
            &ivcpus[0].args,
            ivcpus[0].shadow_sp
        ),
        "same re-attach residue"
    );

    // And the oracle's cut thaws on the JIT: with the waited-on word changed, the child re-issues
    // its wait (NOT_EQUAL, 1·100) and the root reaps it — 2000 + 100.
    let seed: Vec<JitVCpu> = ivcpus
        .iter()
        .map(|v| JitVCpu {
            task: v.task,
            parent_task: v.parent_task,
            slot: v.slot,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
            completed_result: None,
        })
        .collect();
    let mut twin = isnap.clone();
    twin[66000..66004].copy_from_slice(&1i32.to_le_bytes());
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    let (tout, ..) = compile_and_run_durable(
        &inst,
        0,
        &[],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                vcpus: seed.to_vec(),
                root_sp: Some(iroot_sp),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("JIT thaw of the oracle's cut");
    assert_eq!(tout, JitOutcome::Returned(vec![2100]));
}

// #1655 — an **armed** durable run defers every spawn (the window is not `NORMAL`), so on the JIT a
// root that joins a child before the trigger fires used to wait on a child nothing would start. The
// interpreter's single worker runs the child when the root parks in the join. `SRC_CHILD_FIBER`'s
// root joins its child, and the child owns a fiber, so the fiber-safepoint countdown has somewhere to
// fire: 100 never fires, 1 fires inside the child while the root is parked in its join.

/// Run the JIT durable entry on `win` from a fresh host whose clock reads 42, on a helper thread
/// with a deadline, so a regression to the hang fails instead of wedging the test binary. `None` when
/// the JIT declines the module on this host.
#[allow(clippy::type_complexity)]
fn jit_durable_run_bounded(
    inst: &Module,
    win: Vec<u8>,
) -> Option<(JitOutcome, Vec<u8>, Vec<JitFiber>, Vec<JitVCpu>, u64)> {
    let inst = inst.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut h = Host::new();
        h.set_durable(true);
        h.clock_ns = 42;
        let clk = h.grant_clock();
        let r = compile_and_run_durable(
            &inst,
            0,
            &[clk as i64],
            &win,
            SIZE_LOG2,
            temen_run::cap_thunk,
            &mut h as *mut Host as *mut c_void,
            DurableRun {
                seed: DurableResidue {
                    root_sp: Some(TEST_ARENA.region_base(0)),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
        let _ = tx.send(match r {
            Ok((o, w, r)) => Some((o, w, r.fibers, r.vcpus, r.root_sp.unwrap_or(0))),
            Err(JitError::Unsupported(_)) => None,
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => None,
            Err(e) => panic!("JIT failed on a verified durable module: {e:?}"),
        });
    });
    rx.recv_timeout(std::time::Duration::from_secs(20))
        .expect("the JIT durable run hung (#1655)")
}

/// The interpreter's run of `inst` on `win` with the clock at `clock_ns`, seeded with `seed` (the
/// residues and root extent of a cut to thaw, or empty): result, window, and residues.
#[allow(clippy::type_complexity)]
fn interp_durable_run(
    inst: &Module,
    win: &[u8],
    clock_ns: i64,
    seed: Option<(
        Vec<temen_interp::FrozenFiber>,
        Vec<temen_interp::FrozenVCpu>,
        u64,
    )>,
) -> (
    Result<Vec<Value>, temen_interp::Trap>,
    Vec<u8>,
    Vec<temen_interp::FrozenFiber>,
    Vec<temen_interp::FrozenVCpu>,
    Option<u64>,
) {
    let mut h = Host::new();
    h.set_durable(true);
    h.clock_ns = clock_ns;
    let clk = h.grant_clock();
    if let Some((fibers, vcpus, root_sp)) = seed {
        h.set_frozen_fibers(fibers);
        h.set_frozen_vcpus(vcpus);
        h.set_frozen_root_sp(root_sp);
    }
    let mut fuel = 1_000_000u64;
    let (r, snap) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        win,
        SIZE_LOG2,
        &mut h,
    );
    (
        r,
        snap,
        h.frozen_fibers().to_vec(),
        h.frozen_vcpus().to_vec(),
        h.frozen_root_sp(),
    )
}

/// Armed, trigger never fires: both engines finish the ordinary run, `42 + (5 + 100)`.
#[test]
fn an_armed_run_that_joins_before_its_trigger_completes_on_both_engines() {
    let inst = instrument_child_fiber();
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, 100);

    let (ir, ..) = interp_durable_run(&inst, &win, 42, None);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(147)]),
        "the oracle's ordinary result"
    );

    let Some((jout, ..)) = jit_durable_run_bounded(&inst, win) else {
        return;
    };
    assert!(
        matches!(jout, JitOutcome::Returned(ref v) if v == &[147]),
        "the JIT runs the joined child instead of waiting on it: {jout:?}"
    );
}

/// Armed, trigger fires inside the child while the root is parked in its join: both engines freeze
/// the child and the root at the same points, byte-identically, and the cut thaws to the
/// uninterrupted result on both. Before #1655 the JIT hung here, and the oracle let the root run on
/// past the freeze (its dispatch restored the root's pre-freeze `ARMED` phase), returning a result
/// built from the child's unwind placeholder.
#[test]
fn a_trigger_that_fires_in_a_joined_child_freezes_identically_on_both_engines() {
    let inst = instrument_child_fiber();
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_freeze_after(&mut win, 1);

    let (ir, isnap, ifibers, ivcpus, iroot_sp) = interp_durable_run(&inst, &win, 42, None);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(0)]),
        "the oracle freezes: a placeholder result"
    );
    assert_eq!(ivcpus.len(), 1, "the oracle captured the child");
    assert_eq!(ifibers.len(), 1, "the oracle flattened the child's fiber");

    let Some((jout, jsnap, jfibers, jvcpus, jroot_sp)) = jit_durable_run_bounded(&inst, win) else {
        return;
    };
    assert!(
        matches!(jout, JitOutcome::Returned(_)),
        "JIT freeze: {jout:?}"
    );
    let reserve = TEST_ARENA.end as usize;
    assert_eq!(
        &isnap[..reserve],
        &jsnap[..reserve],
        "both engines cut the same durable reserve"
    );
    assert_eq!(Some(jroot_sp), iroot_sp, "same root extent");
    assert_eq!(jvcpus.len(), ivcpus.len(), "same vCPU residue count");
    assert_eq!(jfibers.len(), ifibers.len(), "same fiber residue count");
    for (j, i) in jvcpus.iter().zip(&ivcpus) {
        assert_eq!(
            (j.task, j.func, &j.args, j.shadow_sp),
            (i.task, i.func, &i.args, i.shadow_sp),
            "same vCPU residue"
        );
    }
    for (j, i) in jfibers.iter().zip(&ifibers) {
        assert_eq!(
            (j.slot, j.func, j.shadow_sp),
            (i.slot, i.func, i.shadow_sp),
            "same fiber residue"
        );
    }

    // The cut thaws to the uninterrupted result on both engines, reloading the clock read it took
    // (42) rather than re-reading the advanced clock (99).
    let mut iwin = isnap.clone();
    begin_thaw(&mut iwin, TEST_ARENA, 0);
    let (tr, ..) = interp_durable_run(&inst, &iwin, 99, Some((ifibers, ivcpus, jroot_sp)));
    assert_eq!(tr, Ok(vec![Value::I64(147)]), "the oracle thaws its cut");
    let mut twin = jsnap.clone();
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut thost = Host::new();
    thost.set_durable(true);
    thost.clock_ns = 99;
    let tclk = thost.grant_clock();
    let (tout, ..) = compile_and_run_durable(
        &inst,
        0,
        &[tclk as i64],
        &twin,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut thost as *mut Host as *mut c_void,
        DurableRun {
            seed: DurableResidue {
                fibers: jfibers.to_vec(),
                vcpus: jvcpus.to_vec(),
                root_sp: Some(jroot_sp),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .expect("JIT thaw");
    assert!(
        matches!(tout, JitOutcome::Returned(ref v) if v == &[147]),
        "the thawed cut finishes the uninterrupted run: {tout:?}"
    );
}

/// #1690 — the **embedder's** durable JIT path (`temen_run::jit_cap_run`, what an embedder that
/// snapshots through `temen_snapshot` runs) carries the spawned-vCPU residue and the root's extent
/// through the `Host` both ways. It used to hand back fibers only, so a frozen thread was silently
/// missing from the artifact and the thaw's `thread.join` found nothing. The interpreter is the
/// oracle for both the residue and the thawed result.
#[test]
fn the_embedder_jit_path_carries_the_vcpu_residue_both_ways() {
    let inst = instrument();
    let mut fwin = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut fwin, STATE_UNWINDING);
    let (_, isnap, _, ivcpus, iroot) = interp_durable_run(&inst, &fwin, 42, None);

    // Freeze on the JIT through the embedder path: the residue lands on the Host.
    let mut h = Host::new();
    h.set_durable(true);
    h.clock_ns = 42;
    let clk = h.grant_clock();
    let jsnap = match temen_run::jit_cap_run(
        &inst,
        0,
        &[clk as i64],
        &MemLayout::image(fwin.to_vec()),
        SIZE_LOG2,
        0,
        &mut h,
    ) {
        Ok((_, snap)) => snap.bytes().to_vec(),
        Err(JitError::Unsupported(_)) => return, // a target without the threads runtime
        Err(e) => panic!("JIT freeze failed: {e:?}"),
    };
    let jvcpus = h.frozen_vcpus().to_vec();
    assert_eq!(
        jvcpus.len(),
        ivcpus.len(),
        "the spawned child's residue reached the Host"
    );
    for (j, i) in jvcpus.iter().zip(&ivcpus) {
        assert_eq!(
            (j.task, j.parent_task, j.func),
            (i.task, i.parent_task, i.func)
        );
        assert_eq!(j.shadow_sp, i.shadow_sp, "same extent");
    }
    assert_eq!(
        h.frozen_root_sp(),
        iroot,
        "the root's extent reached the Host too"
    );

    // Thaw on the JIT through the same path, from the Host, under an advanced clock: the result is
    // the interpreter's thaw of the interpreter's own cut.
    let (iresult, ..) = {
        let mut twin = isnap.clone();
        begin_thaw(&mut twin, TEST_ARENA, 0);
        interp_durable_run(&inst, &twin, 99, Some((Vec::new(), ivcpus, iroot.unwrap())))
    };
    let mut twin = jsnap;
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let mut th = Host::new();
    th.set_durable(true);
    th.clock_ns = 99;
    let tclk = th.grant_clock();
    th.set_frozen_vcpus(jvcpus);
    th.set_frozen_root_sp(h.frozen_root_sp().expect("root extent"));
    let (tout, _) = temen_run::jit_cap_run(
        &inst,
        0,
        &[tclk as i64],
        &MemLayout::image(twin.to_vec()),
        SIZE_LOG2,
        0,
        &mut th,
    )
    .expect("JIT thaw");
    let want = match iresult {
        Ok(v) => v,
        Err(t) => panic!("interp thaw trapped: {t:?}"),
    };
    let JitOutcome::Returned(got) = tout else {
        panic!("JIT thaw did not return: {tout:?}");
    };
    assert_eq!(
        got,
        want.iter()
            .map(|v| match v {
                Value::I64(x) => *x,
                other => panic!("unexpected result {other:?}"),
            })
            .collect::<Vec<_>>(),
        "the JIT's thaw from the Host matches the interpreter's"
    );
    assert!(
        th.frozen_vcpus().is_empty(),
        "the thaw consumed the residue"
    );
}

/// #1690 — residue the JIT cannot re-create yet is refused whole, and left on the `Host` so the
/// embedder can thaw on the interpreter instead: never silently dropped.
#[test]
fn the_embedder_jit_path_refuses_residue_it_cannot_recreate_and_keeps_it() {
    let inst = instrument();
    let mut h = Host::new();
    h.set_durable(true);
    let clk = h.grant_clock();
    let detached = temen_interp::FrozenDetached {
        parent_task: 0,
        slot: 0,
        completed_result: Ok(7),
    };
    h.set_frozen_detached(vec![detached]);
    let r = temen_run::jit_cap_run(
        &inst,
        0,
        &[clk as i64],
        &MemLayout::image(init_durable_window(WINDOW, TEST_ARENA).to_vec()),
        SIZE_LOG2,
        0,
        &mut h,
    );
    assert!(
        matches!(r, Err(JitError::Unsupported(_))),
        "refused: {:?}",
        r.as_ref().map(|(o, _)| o)
    );
    assert_eq!(
        h.frozen_detached(),
        &[detached],
        "and the residue is still there"
    );
}
