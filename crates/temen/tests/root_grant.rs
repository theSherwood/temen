//! #2113 — **the root's grant is its node's ceilings**, on every engine: each interpreter driver and
//! the Cranelift JIT.
//!
//! The embedder grants a run `mem`, `channel` and `spawn` as the ceilings of the run's own budget
//! node (`Host::set_grant`, which `temen_run::Limits` maps to), and the root is charged against them
//! like any domain against its own budget: its window and main vCPU from the start, then each live
//! fiber `FIBER_STACK` of `mem`, each page it grows its window by, each live vCPU it spawns one
//! `spawn`, all or nothing up the chain. Past a ceiling the op fails (`cont.new` traps `FiberFault`,
//! `thread.spawn` traps `ThreadFault`, a `map` is `-ENOMEM`). A run whose embedder names no grant
//! holds the default one, large enough for every program here.
//!
//! These replace the retired `Quota` (`max_fibers`, `max_vcpus`): the hard ceilings
//! (`MAX_FIBERS`, `MAX_VCPUS`) stay, and the grant bounds a run below them.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, MemLayout, Trap, Value};
use temen_ir::{Module, FIBER_STACK};
use temen_jit::{JitError, JitOutcome, TrapKind};

/// The window every program here declares.
const WINDOW: i64 = 1 << 16;

/// Two fibers the root makes and never resumes.
const TWO_FIBERS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = cont.new v0 v1
  return v1
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

/// One vCPU, spawned and joined: its result, 5.
const SPAWN_JOIN: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 5
  v1 = thread.spawn 1 v0 v0
  v2 = thread.join v1
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

/// Eight vCPUs, each spawned and joined before the next: only one is ever live. Returns 42.
const SPAWN_JOIN_LOOP: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 8
  v3 = i64.lt_u v1 v2
  br_if v3 2(v1) 3()
}
block 2 (v4: i64) {
  v5 = i64.const 7
  v6 = thread.spawn 1 v5 v5
  v7 = thread.join v6
  v8 = i64.const 1
  v9 = i64.add v4 v8
  br 1(v9)
}
block 3 () {
  v10 = i64.const 42
  return v10
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

/// The root makes a fiber, then spawns a vCPU that makes one too and returns 7: both are the run's.
const FIBERS_ACROSS_VCPUS: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = thread.spawn 1 v1 v1
  v4 = thread.join v3
  return v4
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  v0 = ref.func 2
  v1 = cont.new v0 varg
  v2 = i64.const 7
  return v2
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

/// The root grows its window by 64 KiB twice through the `AddressSpace` it is handed, and returns
/// `first * 1000 + second`: each map's answer.
const GROW_TWICE: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (vas: i32) {
  a1 = i64.const 65536
  a2 = i64.const 131072
  n = i64.const 65536
  rw = i64.const 3
  r1 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (a1, n, rw)
  r2 = call.cap 5 0 (i64, i64, i64) -> (i64) vas (a2, n, rw)
  k = i64.const 1000
  h = i64.mul r1 k
  sum = i64.add h r2
  return sum
  }
}
"#;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// A powerbox granted `mem` bytes and `spawn` live vCPUs (`None`: the default grant), and the root's
/// arguments: none, or with `space` an `AddressSpace` over 1 MiB of its reservation.
fn granted(grant: Option<(i64, i64)>, space: bool) -> impl Fn() -> (Host, Vec<Value>) {
    move || {
        let mut h = Host::new();
        if let Some((mem, spawn)) = grant {
            h.set_grant(mem, -1, spawn);
        }
        let args = if space {
            vec![Value::I32(h.grant_address_space(0, 1 << 20))]
        } else {
            Vec::new()
        };
        (h, args)
    }
}

/// Run `src`'s function 0 under `grant` on every interpreter driver and on the Cranelift JIT, where
/// this target has its thread and fiber runtime, and assert each gives `want`.
fn every_engine(what: &str, src: &str, grant: Option<(i64, i64)>, want: Result<i64, Trap>) {
    run_everywhere(what, src, &granted(grant, false), want);
}

/// [`every_engine`] over the powerbox `setup` builds.
fn run_everywhere(
    what: &str,
    src: &str,
    setup: &dyn Fn() -> (Host, Vec<Value>),
    want: Result<i64, Trap>,
) {
    let m = module(src);
    let ran = Ran {
        result: want.map(|v| vec![Value::I64(v)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(what, &m, setup, &ran);
    let jit_want = match want {
        Ok(v) => JitOutcome::Returned(vec![v]),
        Err(Trap::FiberFault) => JitOutcome::Trapped(TrapKind::FiberFault),
        Err(Trap::ThreadFault) => JitOutcome::Trapped(TrapKind::ThreadFault),
        Err(t) => panic!("{what}: no JIT trap for {t:?}"),
    };
    let (mut host, args) = setup();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(h) => i64::from(*h),
            other => panic!("a handle, not {other:?}"),
        })
        .collect();
    let init = MemLayout::image(vec![0u8; WINDOW as usize]);
    match temen_run::jit_cap_run(&m, 0, &args, &init, 20, 0, &mut host, None) {
        Ok((o, _)) => assert_eq!(o, jit_want, "{what}: Cranelift"),
        Err(JitError::Unsupported(_)) => {} // a target without the fiber or thread runtime
        Err(e) => panic!("{what}: the JIT run failed: {e:?}"),
    }
}

/// The root's fibers spend its `mem`, past its window: room for one, and the second `cont.new` traps.
#[test]
fn a_fiber_past_the_roots_mem_grant_traps() {
    let fibers = |n: i64| Some((WINDOW + n * FIBER_STACK as i64, -1));
    every_engine("room for one", TWO_FIBERS, fibers(1), Err(Trap::FiberFault));
    every_engine("room for two", TWO_FIBERS, fibers(2), Ok(0));
    every_engine("the default grant", TWO_FIBERS, None, Ok(0));
}

/// The root's window counts against its `mem`: a byte short of the window and two fibers is room for
/// one fiber, not two.
#[test]
fn the_roots_window_spends_its_mem_grant() {
    let short = Some((WINDOW + 2 * FIBER_STACK as i64 - 1, -1));
    every_engine("a byte short", TWO_FIBERS, short, Err(Trap::FiberFault));
}

/// The root's growth spends its `mem`: room for one 64 KiB map past the window, and the second is
/// `-ENOMEM`.
#[test]
fn growth_past_the_roots_mem_grant_is_enomem() {
    let grown = |n: i64| granted(Some((WINDOW + n * WINDOW, -1)), true);
    run_everywhere("room for one", GROW_TWICE, &grown(1), Ok(-12));
    run_everywhere("room for two", GROW_TWICE, &grown(2), Ok(0));
    run_everywhere("the default grant", GROW_TWICE, &granted(None, true), Ok(0));
}

/// The root's vCPUs spend its `spawn`, its main vCPU first: room for the root alone, and
/// `thread.spawn` traps.
#[test]
fn a_vcpu_past_the_roots_spawn_grant_traps() {
    let vcpus = |n: i64| Some((-1, n));
    every_engine(
        "the root alone",
        SPAWN_JOIN,
        vcpus(1),
        Err(Trap::ThreadFault),
    );
    every_engine("one more", SPAWN_JOIN, vcpus(2), Ok(5));
    every_engine("the default grant", SPAWN_JOIN, None, Ok(5));
}

/// `spawn` bounds live vCPUs, not spawns: a joined vCPU hands its `spawn` back, so room for one
/// beside the root is room for any number of them in turn.
#[test]
fn the_spawn_grant_bounds_live_vcpus_not_spawns() {
    every_engine("one live", SPAWN_JOIN_LOOP, Some((-1, 2)), Ok(42));
}

/// `mem` is the run's, not a vCPU's: a spawned vCPU's fiber spends the same grant as the root's.
#[test]
fn the_mem_grant_spans_the_runs_vcpus() {
    let fibers = |n: i64| Some((WINDOW + n * FIBER_STACK as i64, -1));
    every_engine(
        "room for one",
        FIBERS_ACROSS_VCPUS,
        fibers(1),
        Err(Trap::FiberFault),
    );
    every_engine("room for two", FIBERS_ACROSS_VCPUS, fibers(2), Ok(7));
}
