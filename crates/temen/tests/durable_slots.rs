//! Every slot a guest holds a handle to keeps its place through a freeze.
//!
//! **Threads (#1685).** A `thread.spawn` child that is not frozen still keeps its join handle:
//!   - **finished, unjoined**: the root spawns A (no suspend point) and B (reads the clock), then
//!     unwinds. A runs to a genuine finish under the freeze; B unwinds. A must ride as a *completed*
//!     record at its slot — neither re-run (its effect would repeat) nor dropped (B's handle would
//!     shift into A's slot).
//!   - **joined before the freeze**: the root spawns A and joins it, spawns B, and resumes a fiber — the
//!     safepoint an armed trigger freezes it at. A's slot is empty at the freeze; B must thaw at slot
//!     1, not 0.
//!
//! A bumps a counter in guest memory, so a thaw that re-runs it reads 2, not 1.
//!
//! **Fibers (#1684).** A fiber the freeze does not flatten still keeps its slot: a **fresh** one
//! (`cont.new`, never resumed) thaws to start from its entry, and a **free** one (finished) thaws
//! free, at its generation, so the next `cont.new` recycles it into the same handle.
//!
//! Each case runs on every engine that can run it here — the interpreter, the JIT (native stack
//! switching), and for fibers the bytecode engine — and each is thawed on the clock its freeze left
//! behind (a re-issued read would move it on). Every engine must thaw to the uninterrupted answer.

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{
    bytecode, run_capture_reserved_with_host, FrozenFiber, FrozenVCpu, Host, Value,
};
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
const THREADS: &str = r#"
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
const UNJOINED: &str = r#"
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
const JOINED_FIRST: &str = r#"
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

/// f (func 1): suspend its argument, then return what the next resume sends + 1. g (func 2): suspend
/// twice. The root runs k0 = f to completion (freeing slot 0), creates k1 = f and leaves it fresh,
/// and resumes k2 = g — the safepoint the trigger freezes at. After the cut it runs k1, then creates
/// k3, which recycles slot 0. Returns `x0 + 10 x1 + 100 x2 + 1000 x3 + 10^4 x4 + 10^5 k3`.
const FIBERS: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  f = ref.func 1
  g = ref.func 2
  sp = i64.const 4096
  k0 = cont.new f sp
  k1 = cont.new f sp
  k2 = cont.new g sp
  a0 = i64.const 1
  s0, x0 = cont.resume k0 a0
  a1 = i64.const 5
  s1, x1 = cont.resume k0 a1
  a2 = i64.const 2
  s2, x2 = cont.resume k2 a2
  a3 = i64.const 7
  s3, x3 = cont.resume k1 a3
  a4 = i64.const 8
  s4, x4 = cont.resume k1 a4
  k3 = cont.new f sp
  c10 = i64.const 10
  c100 = i64.const 100
  c1000 = i64.const 1000
  c10k = i64.const 10000
  c100k = i64.const 100000
  t1 = i64.mul x1 c10
  t2 = i64.mul x2 c100
  t3 = i64.mul x3 c1000
  t4 = i64.mul x4 c10k
  t5 = i64.mul k3 c100k
  r1 = i64.add x0 t1
  r2 = i64.add r1 t2
  r3 = i64.add r2 t3
  r4 = i64.add r3 t4
  r5 = i64.add r4 t5
  return r5
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  v3 = i64.const 1
  v4 = i64.add v2 v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  v3 = suspend v2
  return v3
  }
}
"#;

/// f as in [`FIBERS`]. The root creates k0 and k1, reads the clock — where a freeze from the start
/// lands — then runs k1 and k0 to completion. Returns `clock + 10 x1 + 100 x3`.
const FRESH: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  f = ref.func 1
  sp = i64.const 4096
  k0 = cont.new f sp
  k1 = cont.new f sp
  z = i32.const 0
  c = call.cap 2 0 (i32) -> (i64) v0 (z)
  a0 = i64.const 7
  s0, x0 = cont.resume k1 a0
  a1 = i64.const 8
  s1, x1 = cont.resume k1 a1
  a2 = i64.const 1
  s2, x2 = cont.resume k0 a2
  a3 = i64.const 5
  s3, x3 = cont.resume k0 a3
  c10 = i64.const 10
  c100 = i64.const 100
  t1 = i64.mul x1 c10
  t3 = i64.mul x3 c100
  r1 = i64.add c t1
  r2 = i64.add r1 t3
  return r2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = suspend v1
  v3 = i64.const 1
  v4 = i64.add v2 v3
  return v4
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
    Bytecode,
    Jit,
}

/// One `i64` result, or what came back instead.
fn one_i64(r: Result<Vec<Value>, temen_interp::Trap>) -> Result<i64, String> {
    match r {
        Ok(v) if v.len() == 1 => match v[0] {
            Value::I64(x) => Ok(x),
            other => Err(format!("{other:?}")),
        },
        other => Err(format!("{other:?}")),
    }
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
    if let (Engine::Interp | Engine::Bytecode, Some(r)) = (engine, seed) {
        h.set_frozen_vcpus(r.vcpus.clone());
        h.set_frozen_fibers(r.fibers.clone());
        if let Some(sp) = r.root_sp {
            h.set_frozen_root_sp(sp);
        }
    }
    let mut fuel = 1_000_000u64;
    let args = [Value::I32(clk)];
    let (result, window, residue) = match engine {
        Engine::Interp | Engine::Bytecode => {
            let (r, window) = match engine {
                Engine::Interp => run_capture_reserved_with_host(
                    inst, 0, &args, &mut fuel, win, SIZE_LOG2, &mut h,
                ),
                _ => bytecode::compile_and_run_capture_reserved_with_host(
                    inst, 0, &args, &mut fuel, win, SIZE_LOG2, &mut h,
                )
                .expect("the bytecode engine drives a single-vCPU durable module"),
            };
            let residue = Residue {
                vcpus: h.frozen_vcpus().to_vec(),
                fibers: h.frozen_fibers().to_vec(),
                root_sp: h.frozen_root_sp(),
            };
            (one_i64(r), window, residue)
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

/// Freeze `src` on each of `engines` (`arm` sets the trigger), then thaw it on the same engine with
/// the clock where the freeze left it. Every engine that runs here must thaw to its own uninterrupted
/// answer; returns each one's freeze residue.
fn freeze_thaw(engines: &[Engine], src: &str, arm: fn(&mut [u8])) -> Vec<(Engine, Residue)> {
    let inst = instrument(src);
    let mut out = Vec::new();
    for &engine in engines {
        let fresh = init_durable_window(WINDOW, TEST_ARENA);
        let Some(whole) = run(engine, &inst, &fresh, 42, None) else {
            continue;
        };
        assert!(whole.result.is_ok(), "{engine:?}: {:?}", whole.result);

        let mut win = fresh;
        arm(&mut win);
        let frozen = run(engine, &inst, &win, 42, None).expect("ran uninterrupted");
        assert!(
            frozen.result.is_ok(),
            "{engine:?}: freeze placeholder: {:?}",
            frozen.result
        );

        let mut win = frozen.window;
        begin_thaw(&mut win, TEST_ARENA, 0);
        let thawed =
            run(engine, &inst, &win, frozen.clock, Some(&frozen.residue)).expect("ran the freeze");
        assert_eq!(thawed.result, whole.result, "{engine:?}: thaw");
        out.push((engine, frozen.residue));
    }
    out
}

/// The engines with durable `thread.*` and an armed freeze trigger: the bytecode engine runs a
/// durable domain single-vCPU, frozen from the start.
const ARMED_ENGINES: &[Engine] = &[Engine::Interp, Engine::Jit];
const ALL_ENGINES: &[Engine] = &[Engine::Interp, Engine::Bytecode, Engine::Jit];

/// `(task, slot, completed_result)` per recorded child, in task order.
fn threads(r: &Residue) -> Vec<(usize, usize, Option<i64>)> {
    let mut v: Vec<_> = r
        .vcpus
        .iter()
        .map(|v| (v.task, v.slot, v.completed_result))
        .collect();
    v.sort();
    v
}

fn check_threads(root: &str, arm: fn(&mut [u8]), want: &[(usize, usize, Option<i64>)]) {
    for (engine, r) in freeze_thaw(ARMED_ENGINES, &format!("{root}{THREADS}"), arm) {
        assert_eq!(threads(&r), want, "{engine:?}: residue");
    }
}

/// A is completed at slot 0, B frozen at slot 1; the thaw neither re-runs A nor shifts B.
#[test]
fn a_finished_unjoined_thread_rides_as_completed_at_its_slot() {
    check_threads(
        UNJOINED,
        |w| write_state(w, STATE_UNWINDING),
        &[(1, 0, Some(107)), (2, 1, None)],
    );
}

/// A was joined before the freeze, so only B rides, at slot 1; its handle still resolves on thaw.
#[test]
fn a_slot_joined_before_the_freeze_stays_empty() {
    check_threads(JOINED_FIRST, |w| arm_freeze_after(w, 1), &[(2, 1, None)]);
}

/// Frozen from the start, the root unwinds at its join of A — after the join got A's real result, and
/// before B exists. The join reloads that result on thaw; re-issuing it would need A again, which
/// finished and is not re-run (its bump is already in the image).
#[test]
fn a_join_that_got_its_result_reloads_it() {
    check_threads(JOINED_FIRST, |w| write_state(w, STATE_UNWINDING), &[]);
}

/// Slot 0 is free at generation 1 (k0 finished), slot 1 fresh (k1 never resumed), slot 2 frozen (k2
/// parked at the freeze). The thaw starts k1 from its entry and recycles slot 0 into k3's handle.
#[test]
fn a_fresh_fiber_and_a_free_slot_keep_their_places() {
    let fresh = TEST_ARENA.frame_base(2);
    for (engine, r) in freeze_thaw(ARMED_ENGINES, FIBERS, |w| arm_freeze_after(w, 4)) {
        let mut fibers: Vec<_> = r
            .fibers
            .iter()
            .map(|f| (f.slot, f.is_free(), f.shadow_sp == fresh, f.generation))
            .collect();
        fibers.sort();
        assert_eq!(
            fibers,
            [
                (0, true, false, 1),
                (1, false, true, 0),
                (2, false, false, 0)
            ],
            "{engine:?}: residue"
        );
    }
}

/// Frozen from the start, before any resume: both fibers are fresh. The thaw starts each from its
/// entry, on every engine.
#[test]
fn fresh_fibers_start_from_their_entries() {
    let frame_base = |slot| TEST_ARENA.frame_base(slot + 1);
    for (engine, r) in freeze_thaw(ALL_ENGINES, FRESH, |w| write_state(w, STATE_UNWINDING)) {
        let mut fibers: Vec<_> = r
            .fibers
            .iter()
            .map(|f| (f.slot, f.shadow_sp == frame_base(f.slot)))
            .collect();
        fibers.sort();
        assert_eq!(fibers, [(0, true), (1, true)], "{engine:?}: residue");
    }
}
