//! #1685 — a `thread.spawn` child that is not frozen still keeps its join handle through a freeze.
//!
//! Two shapes, each thawed on the clock the freeze left behind (a re-issued read would move it on):
//!   - **finished, unjoined**: the root spawns A (no suspend point) and B (reads the clock), then
//!     unwinds. A runs to a genuine finish under the freeze; B unwinds. A must ride as a *completed*
//!     record at its slot — neither re-run (its effect would repeat) nor dropped (B's handle would
//!     shift into A's slot).
//!   - **joined before the freeze**: the root spawns A and joins it, spawns B, and resumes a fiber — the
//!     safepoint an armed trigger freezes it at. A's slot is empty at the freeze; B must thaw at slot
//!     1, not 0.
//!
//! A bumps a counter in guest memory, so a thaw that re-runs it reads 2, not 1. Each case runs on the
//! interpreter and, where it has native stack switching, on the JIT: both engines must record the same
//! residue and thaw it to the same answer.

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, FrozenFiber, FrozenVCpu, Host, Value};
use temen_ir::{Memory, Module};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

/// A (func 1): bump the counter at 65544, return 107. B (func 2): read the clock, return it + 10.
/// Func 3: a fiber that suspends at once — only its resume matters, as a freeze safepoint.
const CHILDREN: &str = r#"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 65544
  v3 = i64.load v2
  v4 = i64.const 1
  v5 = i64.add v3 v4
  i64.store v2 v5
  v6 = i64.const 107
  return v6
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
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  return v2
  }
}
"#;

/// Root: spawn A and B, read the clock, join A then B. Returns
/// `counter * 1_000_000 + join(A) * 1000 + clock + join(B)`.
pub const UNJOINED: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v4 = thread.spawn 2 v2 v2
  v5 = i32.const 0
  v6 = call.cap 2 0 (i32) -> (i64) v0 (v5)
  v7 = thread.join v3
  v8 = thread.join v4
  v9 = i64.const 65544
  v10 = i64.load v9
  v11 = i64.const 1000000
  v12 = i64.mul v10 v11
  v13 = i64.const 1000
  v14 = i64.mul v7 v13
  v15 = i64.add v12 v14
  v16 = i64.add v15 v6
  v17 = i64.add v16 v8
  return v17
  }
}
"#;

/// Root: spawn A and join it, spawn B, resume a fiber, then read the clock and join B. Same result
/// shape.
pub const JOINED_FIRST: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 65536
  i32.store v1 v0
  v2 = i64.const 0
  v3 = thread.spawn 1 v2 v2
  v7 = thread.join v3
  v4 = thread.spawn 2 v2 v2
  f0 = ref.func 3
  f1 = i64.const 4096
  f2 = cont.new f0 f1
  f3, f4 = cont.resume f2 v2
  v5 = i32.const 0
  v6 = call.cap 2 0 (i32) -> (i64) v0 (v5)
  v8 = thread.join v4
  v9 = i64.const 65544
  v10 = i64.load v9
  v11 = i64.const 1000000
  v12 = i64.mul v10 v11
  v13 = i64.const 1000
  v14 = i64.mul v7 v13
  v15 = i64.add v12 v14
  v16 = i64.add v15 v6
  v17 = i64.add v16 v8
  return v17
  }
}
"#;

/// `counter 1`, `join(A) = 107`, and the two clock reads 42 + 43 + 10 in either order.
const WANT: i64 = 1_000_000 + 107_000 + 95;

fn instrument(root: &str) -> Module {
    let mut m = temen_text::parse_module(&format!("{root}{CHILDREN}")).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR verifies");
    inst
}

/// The residue a freeze records and a thaw re-seeds, in the interpreter's types.
#[derive(Default)]
struct Residue {
    vcpus: Vec<FrozenVCpu>,
    fibers: Vec<FrozenFiber>,
    root_sp: Option<u64>,
}

/// One engine's run of a window: its result, the window after, the residue it recorded, and the clock
/// it left behind.
struct Ran {
    result: Result<i64, String>,
    window: Vec<u8>,
    residue: Residue,
    clock: i64,
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interp,
    Jit,
}

/// Run `inst` over `win` on `engine` with the clock at `clock`, thawing from `seed` if given.
/// `None` when the JIT can't run here.
fn run(
    engine: Engine,
    inst: &Module,
    win: &[u8],
    clock: i64,
    seed: Option<&Residue>,
) -> Option<Ran> {
    let mut h = Host::new();
    h.set_durable(true);
    h.clock_ns = clock;
    let clk = h.grant_clock();
    let (result, window, residue) = match engine {
        Engine::Interp => {
            if let Some(r) = seed {
                h.set_frozen_vcpus(r.vcpus.clone());
                h.set_frozen_fibers(r.fibers.clone());
                h.set_frozen_root_sp(r.root_sp.expect("root extent"));
            }
            let mut fuel = 1_000_000u64;
            let (r, window) = run_capture_reserved_with_host(
                inst,
                0,
                &[Value::I32(clk)],
                &mut fuel,
                win,
                SIZE_LOG2,
                &mut h,
            );
            let result = match r {
                Ok(v) if v.len() == 1 => match v[0] {
                    Value::I64(x) => Ok(x),
                    other => Err(format!("{other:?}")),
                },
                other => Err(format!("{other:?}")),
            };
            let residue = Residue {
                vcpus: h.frozen_vcpus().to_vec(),
                fibers: h.frozen_fibers().to_vec(),
                root_sp: h.frozen_root_sp(),
            };
            (result, window, residue)
        }
        Engine::Jit => jit::run(inst, win, &mut h, clk, seed)?,
    };
    Some(Ran {
        result,
        window,
        residue,
        clock: h.clock_ns,
    })
}

#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
mod jit {
    use super::*;
    use core::ffi::c_void;
    use temen_jit::{compile_and_run_durable, DurableResidue, DurableRun, JitError, JitOutcome};

    pub type Out = (Result<i64, String>, Vec<u8>, Residue);

    pub fn run(
        inst: &Module,
        win: &[u8],
        h: &mut Host,
        clk: i32,
        seed: Option<&Residue>,
    ) -> Option<Out> {
        let seed = match seed {
            Some(r) => DurableResidue {
                vcpus: r.vcpus.iter().map(to_jit).collect(),
                fibers: r.fibers.iter().map(fiber_to_jit).collect(),
                root_sp: r.root_sp,
                ..Default::default()
            },
            None => DurableResidue {
                root_sp: Some(TEST_ARENA.region_base(0)),
                ..Default::default()
            },
        };
        let (out, window, residue) = match compile_and_run_durable(
            inst,
            0,
            &[clk as i64],
            win,
            SIZE_LOG2,
            temen_run::cap_thunk,
            h as *mut Host as *mut c_void,
            DurableRun {
                seed,
                ..Default::default()
            },
        ) {
            Ok(t) => t,
            Err(JitError::Unsupported(_)) => return None,
            Err(JitError::Backend(msg)) if msg.contains("Allocation error") => return None,
            Err(e) => panic!("JIT failed: {e:?}"),
        };
        let result = match out {
            JitOutcome::Returned(rs) if rs.len() == 1 => Ok(rs[0]),
            other => Err(format!("{other:?}")),
        };
        let residue = Residue {
            vcpus: residue.vcpus.iter().map(to_interp).collect(),
            fibers: residue.fibers.iter().map(fiber_to_interp).collect(),
            root_sp: residue.root_sp,
        };
        Some((result, window, residue))
    }

    fn fiber_to_jit(f: &FrozenFiber) -> temen_jit::FrozenFiber {
        temen_jit::FrozenFiber {
            slot: f.slot,
            func: f.func,
            sp: f.sp,
            shadow_sp: f.shadow_sp,
            generation: f.generation,
            consumed: f.consumed,
        }
    }

    fn fiber_to_interp(f: &temen_jit::FrozenFiber) -> FrozenFiber {
        FrozenFiber {
            slot: f.slot,
            func: f.func,
            sp: f.sp,
            shadow_sp: f.shadow_sp,
            generation: f.generation,
            consumed: f.consumed,
        }
    }

    fn to_jit(v: &FrozenVCpu) -> temen_jit::FrozenVCpu {
        temen_jit::FrozenVCpu {
            task: v.task,
            parent_task: v.parent_task,
            slot: v.slot,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
            completed_result: v.completed_result,
        }
    }

    fn to_interp(v: &temen_jit::FrozenVCpu) -> FrozenVCpu {
        FrozenVCpu {
            task: v.task,
            parent_task: v.parent_task,
            slot: v.slot,
            func: v.func,
            args: v.args.clone(),
            shadow_sp: v.shadow_sp,
            completed_result: v.completed_result,
        }
    }
}

#[cfg(not(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
)))]
mod jit {
    use super::*;
    pub type Out = (Result<i64, String>, Vec<u8>, Residue);
    pub fn run(
        _: &Module,
        _: &[u8],
        _: &mut Host,
        _: i32,
        _: Option<(&[FrozenVCpu], u64)>,
    ) -> Option<Out> {
        None
    }
}

/// Freeze `root` on `engine` (`arm` sets the trigger), then thaw it on the same engine with the clock
/// where the freeze left it. Returns the freeze's residue and the thaw's result, or `None` when the
/// JIT can't run here.
fn freeze_thaw(
    engine: Engine,
    root: &str,
    arm: fn(&mut [u8]),
) -> Option<(Vec<FrozenVCpu>, Result<i64, String>)> {
    let inst = instrument(root);
    let fresh = init_durable_window(WINDOW, TEST_ARENA);
    let want = run(engine, &inst, &fresh, 42, None)?;
    assert_eq!(want.result, Ok(WANT), "{engine:?}: uninterrupted");

    let mut win = fresh;
    arm(&mut win);
    let frozen = run(engine, &inst, &win, 42, None)?;
    assert!(
        frozen.result.is_ok(),
        "{engine:?}: freeze placeholder: {:?}",
        frozen.result
    );
    assert!(
        frozen.residue.root_sp.is_some(),
        "{engine:?}: root extent recorded"
    );

    let mut win = frozen.window;
    begin_thaw(&mut win, TEST_ARENA, 0);
    let thawed = run(engine, &inst, &win, frozen.clock, Some(&frozen.residue))?;
    Some((frozen.residue.vcpus, thawed.result))
}

/// `(task, slot, completed_result)` per recorded child, in task order.
fn shape(vcpus: &[FrozenVCpu]) -> Vec<(usize, usize, Option<i64>)> {
    let mut v: Vec<_> = vcpus
        .iter()
        .map(|v| (v.task, v.slot, v.completed_result))
        .collect();
    v.sort();
    v
}

/// Freeze and thaw `root` on each engine that can run here; every one must record `residue` and thaw
/// to the uninterrupted answer.
fn check(root: &str, arm: fn(&mut [u8]), residue: &[(usize, usize, Option<i64>)]) {
    for engine in [Engine::Interp, Engine::Jit] {
        let Some((frozen, thawed)) = freeze_thaw(engine, root, arm) else {
            continue;
        };
        assert_eq!(shape(&frozen), residue, "{engine:?}: residue");
        assert_eq!(thawed, Ok(WANT), "{engine:?}: thaw");
    }
}

/// A is completed at slot 0, B frozen at slot 1; the thaw neither re-runs A nor shifts B.
#[test]
fn a_finished_unjoined_thread_rides_as_completed_at_its_slot() {
    check(
        UNJOINED,
        |w| write_state(w, STATE_UNWINDING),
        &[(1, 0, Some(107)), (2, 1, None)],
    );
}

/// A was joined before the freeze, so only B rides, at slot 1; its handle still resolves on thaw.
#[test]
fn a_slot_joined_before_the_freeze_stays_empty() {
    check(JOINED_FIRST, |w| arm_freeze_after(w, 1), &[(2, 1, None)]);
}

/// Frozen from the start, the root unwinds at its join of A — after the join got A's real result, and
/// before B exists. The join reloads that result on thaw; re-issuing it would need A again, which
/// finished and is not re-run (its bump is already in the image).
#[test]
fn a_join_that_got_its_result_reloads_it() {
    check(JOINED_FIRST, |w| write_state(w, STATE_UNWINDING), &[]);
}
