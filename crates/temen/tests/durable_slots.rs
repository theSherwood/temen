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
//! **Resumes (#1835).** A `cont.resume` the freeze lands at re-issues on thaw only if its fiber is
//! still residue (it parked, or the freeze unwound it). One whose fiber returned reloads its results:
//! the slot is free.
//!
//! Each case runs on every engine that can run it here — the interpreter, the JIT (native stack
//! switching), and for fibers the bytecode engine — and each is thawed on the clock its freeze left
//! behind (a re-issued read would move it on). Every engine must thaw to the uninterrupted answer.
//!
//! **Deep chains (#1872).** A context's unwound call chain must fit its shadow region. A module whose
//! chains run deeper than the default region declares a wider stride, and freezes whole.

use temen_durable::{
    arm_freeze_after, begin_thaw, init_durable_window, transform_module_assume_confined,
    write_state, STATE_UNWINDING,
};
use temen_interp::{
    bytecode, run_capture_reserved_with_host, FrozenFiber, FrozenVCpu, Host, Value,
};
use temen_ir::durable_abi::{ShadowArena, DEFAULT_SHADOW_STRIDE};
use temen_ir::{Memory, Module};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

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

/// The root resumes a fiber that returns at once, then reads the clock. Returns
/// `clock + 10 status + 100 value`.
const RETURNS: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  f = ref.func 1
  sp = i64.const 4096
  k = cont.new f sp
  a = i64.const 5
  s, x = cont.resume k a
  z = i32.const 0
  c = call.cap 2 0 (i32) -> (i64) v0 (z)
  c10 = i64.const 10
  c100 = i64.const 100
  s64 = i64.extend_i32_u s
  t1 = i64.mul s64 c10
  t2 = i64.mul x c100
  r1 = i64.add c t1
  r2 = i64.add r1 t2
  return r2
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  return v3
  }
}
"#;

/// The root resumes a fiber that reads the clock (where a freeze from the start lands, inside the
/// fiber) and returns. Returns `10 status + value`.
const UNWINDS: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  f = ref.func 1
  sp = i64.const 4096
  k = cont.new f sp
  a = i64.extend_i32_u v0
  s, x = cont.resume k a
  s64 = i64.extend_i32_u s
  c10 = i64.const 10
  t1 = i64.mul s64 c10
  r = i64.add x t1
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  h = i32.wrap_i64 v1
  z = i32.const 0
  c = call.cap 2 0 (i32) -> (i64) h (z)
  return c
  }
}
"#;

/// #1872: a root and a fiber that each recurse [`DEPTH`] deep, eight `i64`s live across every call,
/// so each context's unwound chain is about twice a default (4 KiB) region. The fiber reads the clock
/// at the bottom, where a freeze from the start lands; both chains unwind whole. Its arena declares
/// 16 KiB regions: root, the fiber, and one spare.
const DEEP: &str = r#"
memory 17 shadow 16448 65600 stride 16384
func (i32) -> (i64) {
block 0 (v0: i32) {
  d = i64.const 100
  h = i64.extend_i32_u v0
  r = call 1 (d, h)
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (n: i64, h: i64) {
  c = i64.eqz n
  br_if c 1(h) 2(n, h)
}
block 1 (h: i64) {
  f = ref.func 2
  sp = i64.const 4096
  k = cont.new f sp
  s, x = cont.resume k h
  s64 = i64.extend_i32_u s
  c10 = i64.const 10
  t = i64.mul s64 c10
  r = i64.add x t
  return r
}
block 2 (n: i64, h: i64) {
  k3 = i64.const 3
  t1 = i64.mul n k3
  t2 = i64.add t1 n
  t3 = i64.mul t2 k3
  t4 = i64.add t3 t1
  t5 = i64.mul t4 k3
  t6 = i64.add t5 t2
  one = i64.const 1
  m = i64.sub n one
  r = call 1 (m, h)
  s1 = i64.add r t1
  s2 = i64.add s1 t2
  s3 = i64.add s2 t3
  s4 = i64.add s3 t4
  s5 = i64.add s4 t5
  s6 = i64.add s5 t6
  return s6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  d = i64.const 100
  r = call 3 (d, v1)
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (n: i64, h: i64) {
  c = i64.eqz n
  br_if c 1(h) 2(n, h)
}
block 1 (h: i64) {
  w = i32.wrap_i64 h
  z = i32.const 0
  c = call.cap 2 0 (i32) -> (i64) w (z)
  return c
}
block 2 (n: i64, h: i64) {
  k3 = i64.const 3
  t1 = i64.mul n k3
  t2 = i64.add t1 n
  t3 = i64.mul t2 k3
  t4 = i64.add t3 t1
  t5 = i64.mul t4 k3
  t6 = i64.add t5 t2
  one = i64.const 1
  m = i64.sub n one
  r = call 3 (m, h)
  s1 = i64.add r t1
  s2 = i64.add s1 t2
  s3 = i64.add s2 t3
  s4 = i64.add s3 t4
  s5 = i64.add s4 t5
  s6 = i64.add s5 t6
  return s6
  }
}
"#;

/// Parse and instrument `src`. A source without a `memory` line declares [`TEST_ARENA`].
fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory.get_or_insert(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module_assume_confined(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR verifies");
    inst
}

/// The shadow arena `inst` declares.
fn arena(inst: &Module) -> ShadowArena {
    inst.memory.and_then(|m| m.shadow).expect("a shadow arena")
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
    use std::sync::Mutex;
    use temen_jit::{compile_and_run_durable, DurableResidue, DurableRun, JitError, JitOutcome};
    use temen_run::CapCtx;

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
                root_sp: Some(arena(inst).region_base(0)),
                ..Default::default()
            },
        };
        // The guests spawn vCPUs that `call.cap` from their own OS threads: the host goes behind the
        // serialized thunk's lock for the run, and comes back after (#1166).
        let hm = Mutex::new(std::mem::take(h));
        let cc = CapCtx::Locked(&hm);
        let run = compile_and_run_durable(
            inst,
            0,
            &[clk as i64],
            win,
            SIZE_LOG2,
            cc.thunk(),
            cc.ptr(),
            DurableRun {
                seed,
                ..Default::default()
            },
        );
        *h = hm.into_inner().unwrap_or_else(|e| e.into_inner());
        let (out, window, residue) = match run {
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
/// answer; returns each one's freeze run (its window and residue).
fn freeze_thaw(engines: &[Engine], src: &str, arm: fn(&mut [u8])) -> Vec<(Engine, Ran)> {
    let inst = instrument(src);
    let mut out = Vec::new();
    for &engine in engines {
        let fresh = init_durable_window(WINDOW, arena(&inst));
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

        let mut win = frozen.window.clone();
        begin_thaw(&mut win, arena(&inst), 0);
        let thawed =
            run(engine, &inst, &win, frozen.clock, Some(&frozen.residue)).expect("ran the freeze");
        assert_eq!(thawed.result, whole.result, "{engine:?}: thaw");
        out.push((engine, frozen));
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
        assert_eq!(threads(&r.residue), want, "{engine:?}: residue");
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
            .residue
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
            .residue
            .fibers
            .iter()
            .map(|f| (f.slot, f.shadow_sp == frame_base(f.slot)))
            .collect();
        fibers.sort();
        assert_eq!(fibers, [(0, true), (1, true)], "{engine:?}: residue");
    }
}

/// #1835: the fiber returns inside the resume the freeze lands at. The resumer already has its
/// `(status, value)`, and the fiber's slot is free, so the thaw must reload them: re-issuing the
/// resume faults on the free slot.
#[test]
fn a_resume_whose_fiber_returned_reloads_its_result() {
    freeze_thaw(ALL_ENGINES, RETURNS, |w| write_state(w, STATE_UNWINDING));
    freeze_thaw(ARMED_ENGINES, RETURNS, |w| arm_freeze_after(w, 1));
}

/// And the case the re-issue is for: the fiber unwinds for the freeze inside the resume, so the thaw
/// re-issues the resume and the fiber rewinds to its clock read. The bytecode engine too: the unwound
/// fiber rides as residue there, not as a free slot.
#[test]
fn a_resume_whose_fiber_unwound_is_re_issued() {
    freeze_thaw(ALL_ENGINES, UNWINDS, |w| write_state(w, STATE_UNWINDING));
}

/// #1872: each context's chain is deeper than a default region, so with 4 KiB regions the freeze
/// traps (#1683: it would otherwise write the next context's frames). The module that declares
/// 16 KiB regions freezes both chains whole and thaws to the uninterrupted answer, on every engine.
#[test]
fn chains_deeper_than_a_default_region_freeze_in_wider_regions() {
    let narrow = DEEP.replace(" stride 16384", "").replace("65600", "65536");
    let inst = instrument(&narrow);
    assert_eq!(
        arena(&inst).stride,
        temen_ir::durable_abi::DEFAULT_SHADOW_STRIDE
    );
    for &engine in ALL_ENGINES {
        let mut win = init_durable_window(WINDOW, arena(&inst));
        write_state(&mut win, STATE_UNWINDING);
        if let Some(frozen) = run(engine, &inst, &win, 42, None) {
            assert!(frozen.result.is_err(), "{engine:?}: {:?}", frozen.result);
        }
    }

    let a = arena(&instrument(DEEP));
    for (engine, r) in freeze_thaw(ALL_ENGINES, DEEP, |w| write_state(w, STATE_UNWINDING)) {
        // Both chains unwound whole, each past what a default region holds.
        let depth = |sp: u64, ctx: usize| sp - a.frame_base(ctx);
        let fiber = r
            .residue
            .fibers
            .iter()
            .find(|f| !f.is_free())
            .expect("the fiber rides");
        assert!(
            depth(fiber.shadow_sp, 1) > DEFAULT_SHADOW_STRIDE,
            "{engine:?}: fiber"
        );
        // A single-vCPU freeze records no root extent: it is the root region's SP word.
        let root = r.residue.root_sp.unwrap_or_else(|| {
            let at = a.region_base(0) as usize;
            u64::from_le_bytes(r.window[at..at + 8].try_into().unwrap())
        });
        assert!(depth(root, 0) > DEFAULT_SHADOW_STRIDE, "{engine:?}: root");
    }
}
