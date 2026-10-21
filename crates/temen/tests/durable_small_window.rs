//! #2518 — **a window smaller than one NULL guard is durable.** The freeze word, the quiesce flag and
//! the two arm countdowns live in context 0's region header, inside the arena the module declares
//! (INVARIANTS.md #16), not at an offset the substrate fixes. So an 8 KiB window that declares an arena
//! freezes and thaws on every engine. A module that declares none has no control words: a durable
//! host runs it, and no engine reads or writes a word for it. The JIT used to read its freeze word at
//! `mem_base + 16384` for every durable window: SIGBUS on macOS, an access violation on Windows, and a
//! read of host memory on Linux.

use std::sync::mpsc;
use std::time::Duration;
use temen_durable::{
    begin_thaw, init_durable_window, read_state, read_thaw_state, transform_module_assume_confined,
    write_state, STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{bytecode, run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::Module;
use temen_jit::{FreezeController, JitOutcome};

const SIZE_LOG2: u8 = 13;
const WINDOW: usize = 1 << SIZE_LOG2;
/// The JIT's reservation: past the window, so the old read at `mem_base + 16384` faulted on every
/// host rather than reading whatever the process had mapped there.
const RESERVED_LOG2: u8 = 16;
/// The top half of the window; the guest keeps the bottom half.
const ARENA: ShadowArena = ShadowArena::new(4096, 8192);

#[derive(Clone, Copy, Debug)]
enum Engine {
    TreeWalker,
    Bytecode,
    Jit,
}
const ENGINES: [Engine; 3] = [Engine::TreeWalker, Engine::Bytecode, Engine::Jit];

/// A run's `i64` result and the window it left.
fn run(e: Engine, m: &Module, host: &mut Host, arg: i32, win: &[u8]) -> (i64, Vec<u8>) {
    let mut fuel = 10_000_000u64;
    let args = [Value::I32(arg)];
    let (r, left) = match e {
        Engine::TreeWalker => {
            run_capture_reserved_with_host(m, 0, &args, &mut fuel, win, SIZE_LOG2, host)
        }
        Engine::Bytecode => bytecode::compile_and_run_capture_reserved_with_host(
            m, 0, &args, &mut fuel, win, SIZE_LOG2, host,
        )
        .expect("the bytecode engine drives the module"),
        Engine::Jit => {
            let init = MemLayout::image(win.to_vec());
            let (o, left) =
                temen_run::jit_cap_run(m, 0, &[arg as i64], &init, RESERVED_LOG2, 0, host, None)
                    .expect("the JIT compiles the module");
            match o {
                JitOutcome::Returned(v) => (Ok(vec![Value::I64(v[0])]), left.bytes().to_vec()),
                other => panic!("{e:?}: {other:?}"),
            }
        }
    };
    match r.unwrap_or_else(|t| panic!("{e:?} trapped: {t:?}"))[..] {
        [Value::I64(n)] => (n, left),
        ref other => panic!("{e:?}: unexpected result {other:?}"),
    }
}

/// A durable host granting a clock that reads `clock` first.
fn clock_host(clock: i64) -> (Host, i32) {
    let mut h = Host::new();
    h.set_durable(true);
    h.clock_ns = clock;
    let clk = h.grant_clock();
    (h, clk)
}

/// A module with nothing to instrument, so no arena: it stores to the first and last words of its
/// window, which has no NULL guard below 16 KiB, and returns their sum.
const NO_ARENA: &str = "memory 13
func (i32) -> (i64) {
block 0 (v0: i32) {
  vlo = i64.const 0
  vhi = i64.const 8184
  va = i64.const 5
  vb = i64.const 2
  i64.store vlo va
  i64.store vhi vb
  vx = i64.load vlo
  vy = i64.load vhi
  vr = i64.add vx vy
  return vr
  }
}
";

#[test]
fn a_module_without_an_arena_runs_under_a_durable_host_on_every_engine() {
    let m = temen_text::parse_module(NO_ARENA).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    for e in ENGINES {
        let (mut h, clk) = clock_host(0);
        let (r, left) = run(e, &m, &mut h, clk, &vec![0u8; WINDOW]);
        assert_eq!(r, 7, "{e:?}");
        assert_eq!(left.len(), WINDOW, "{e:?}");
        assert!(
            left[8..8184].iter().all(|&b| b == 0),
            "{e:?}: the run wrote only the guest's two words"
        );
    }
}

/// As [`NO_ARENA`], but it loops long enough for a controller thread to reach it.
const NO_ARENA_LOOP: &str = "memory 13
func (i32) -> (i64) {
block 0 (v0: i32) {
  vz = i64.const 0
  br 1(vz)
  }
block 1 (vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 2000000
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vi2) 2(vi2)
  }
block 2 (vn: i64) {
  return vn
  }
}
";

/// A freeze request reaches a JIT run whose window has no freeze word. The run publishes none, so the
/// request returns at once and stores nowhere, and the run finishes as if it had not been asked. The
/// controller used to store `UNWINDING` at `mem_base + 16384`, outside this window.
#[test]
fn a_freeze_request_to_a_run_without_an_arena_lands_nowhere() {
    let m = temen_text::parse_module(NO_ARENA_LOOP).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let ctl = FreezeController::new();
    let (done, requested) = mpsc::channel();
    let req = {
        let ctl = ctl.clone();
        std::thread::spawn(move || {
            ctl.request_freeze();
            let _ = done.send(());
        })
    };
    let (mut h, _) = clock_host(0);
    let init = MemLayout::image(vec![0u8; WINDOW]);
    let (o, left) = temen_run::jit_cap_run(&m, 0, &[0], &init, RESERVED_LOG2, 0, &mut h, Some(ctl))
        .expect("the JIT compiles the module");
    assert_eq!(o, JitOutcome::Returned(vec![2_000_000]));
    assert!(left.bytes().iter().all(|&b| b == 0), "nothing was stored");
    requested
        .recv_timeout(Duration::from_secs(60))
        .expect("the request returns once the run has published no word");
    req.join().unwrap();
}

/// Stores `42` in guest memory, then reads the clock twice; the first read is live across the second.
/// Answers `42·10⁶ + first·10³ + second`.
const TWO_READS: &str = "memory 13 shadow 4096 8192
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 8
  vk = i64.const 42
  i64.store va vk
  v1 = call.cap 2 0 () -> (i64) v0 ()
  v2 = call.cap 2 0 () -> (i64) v0 ()
  v3 = i64.load va
  vm = i64.const 1000000
  v4 = i64.mul v3 vm
  vt = i64.const 1000
  v5 = i64.mul v1 vt
  v6 = i64.add v4 v5
  v7 = i64.add v6 v2
  return v7
  }
}
";

/// An 8 KiB window declaring the arena `[4096, 8192)` freezes at its first clock read on each engine
/// to one byte-identical artifact, which thaws on each engine to the uninterrupted answer: the frozen
/// read reloads rather than re-issues.
#[test]
fn an_8k_window_freezes_and_thaws_across_every_engine() {
    let m = temen_text::parse_module(TWO_READS).expect("parse");
    // The guest's memory is the window's bottom half, clear of its arena.
    let inst = transform_module_assume_confined(&m).expect("the arena fits the window");
    temen_verify::verify_module(&inst).expect("verify");
    let fresh = init_durable_window(WINDOW, ARENA);
    for e in ENGINES {
        let (mut h, clk) = clock_host(5);
        assert_eq!(run(e, &inst, &mut h, clk, &fresh).0, 42_005_006, "{e:?}");
    }

    let mut frozen = fresh.clone();
    write_state(&mut frozen, ARENA, STATE_UNWINDING);
    let mut artifacts = Vec::new();
    for e in ENGINES {
        let (mut h, clk) = clock_host(5);
        let (_, snap) = run(e, &inst, &mut h, clk, &frozen);
        assert_eq!(read_state(&snap, ARENA), STATE_UNWINDING, "{e:?}");
        assert_eq!(snap[8..16], 42i64.to_le_bytes(), "{e:?}: guest memory rode");
        let art = temen_snapshot::freeze(&inst, &snap, &h).expect("freeze");
        artifacts.push((e, art));
    }
    for (e, art) in &artifacts[1..] {
        assert_eq!(
            *art, artifacts[0].1,
            "{e:?}'s artifact is the tree-walker's"
        );
    }

    for e in ENGINES {
        // The clock goes on from where the freeze left it: one read was taken.
        let (mut h, clk) = clock_host(6);
        let mut win = temen_snapshot::restore(&artifacts[0].1, &inst, &mut h).expect("restore");
        begin_thaw(&mut win, ARENA, 0);
        let (r, left) = run(e, &inst, &mut h, clk, &win);
        assert_eq!(r, 42_005_006, "{e:?}: the thaw reloads the frozen read");
        assert_eq!(read_state(&left, ARENA), STATE_NORMAL, "{e:?}");
        assert_eq!(read_thaw_state(&left, ARENA, 0), STATE_NORMAL, "{e:?}");
    }
}
