//! #1688 — the fork-twin freeze the durable tests share: a guest that forks through the POSIX
//! personality, a freeze on the tree-walker, the §12 codec, and a thaw into a fresh host whose
//! registrar rebuilds the personality ([`temen_posix::Thaw`]). Included by `durable_fork.rs` and by
//! `park_freeze_matrix.rs`, whose `waitpid` row is [`waitpid_parked_at_the_trigger_thaws`].
#![allow(dead_code)] // each including binary uses a different subset

use std::sync::{Arc, Mutex};
use temen_durable::{
    arm_freeze_after_backedges, begin_thaw, init_durable_window, transform_module_assume_confined,
};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::durable_abi::ShadowArena;
use temen_ir::Memory;

pub const ARENA: ShadowArena = ShadowArena::new(16448, 65536);
pub const SIZE_LOG2: u8 = 17;
pub const WINDOW: usize = 1 << SIZE_LOG2;
const HEAP: (u64, u64) = (100_000, 120_000);

/// What each guest here answers, uninterrupted or thawed: its twin's pid (`2`, the first a twin can
/// take) times 1000, plus the twin's exit status, `2 · 10 + 7`. A thaw that gave the twin another
/// pid would change both halves.
pub const ANSWER: i64 = 2027;

/// `src` with its memory and shadow arena, instrumented confined and verified.
pub fn instrumented(src: &str) -> Arc<temen_ir::Module> {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(ARENA),
    });
    let inst = Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// A durable host granting the personality under the name `"libc"`.
pub fn granted() -> (Host, temen_posix::Posix, i32) {
    let mut h = Host::new();
    h.set_durable(true);
    let (lh, posix) = temen_posix::grant(&mut h, HEAP.0, HEAP.1, Vec::new());
    h.register_cap_name("libc", lh);
    (h, posix, lh)
}

/// Run `inst`'s func 0 over `win` on the tree-walker, handed the personality's handle `lh`: what it
/// returns, and the window it leaves.
pub fn run(inst: &temen_ir::Module, h: &mut Host, lh: i32, win: &[u8]) -> (i64, Vec<u8>) {
    let mut fuel = 10_000_000u64;
    let (r, snap) =
        run_capture_reserved_with_host(inst, 0, &[Value::I32(lh)], &mut fuel, win, SIZE_LOG2, h);
    match r.expect("the run returns")[..] {
        [Value::I64(n)] => (n, snap),
        ref other => panic!("unexpected result {other:?}"),
    }
}

/// A fresh durable host whose registrar rebuilds the personality from an artifact, the root's
/// process and every twin's ([`temen_posix::Thaw`]).
pub fn thaw_host() -> (Host, Arc<Mutex<temen_posix::Thaw>>) {
    let thaw: Arc<Mutex<temen_posix::Thaw>> = Arc::default();
    let mut h = Host::new();
    h.set_durable(true);
    let t = Arc::clone(&thaw);
    h.set_named_cap_registrar(Box::new(move |name, state, payer| {
        (name == "libc")
            .then(|| t.lock().unwrap().grant(state, payer))
            .flatten()
    }));
    (h, thaw)
}

/// Restore `art` into a fresh host and ready it to run on: the host, the pids of the twins it hands
/// back, and the window, under `REWINDING`.
pub fn thaw(art: &[u8], inst: &temen_ir::Module, lh: i32) -> (Host, Vec<u64>, Vec<u8>) {
    let (mut th, thaw) = thaw_host();
    let mut back = temen_snapshot::restore(art, inst, &mut th).expect("restore");
    let twins = th.thawed_twins().iter().map(|t| t.pid).collect();
    let posix = thaw
        .lock()
        .unwrap()
        .posix()
        .expect("the registrar re-granted libc");
    temen_posix::install(&mut th, &posix, lh);
    begin_thaw(&mut back, ARENA, 0);
    (th, twins, back)
}

/// Freeze `src` on the tree-walker, armed by `arm`, then thaw it and run it on: what the
/// uninterrupted run answers, what the thawed one does, and the pids of the twins the freeze
/// carried.
pub fn freeze_and_thaw(src: &str, arm: impl Fn(&mut [u8])) -> (i64, i64, Vec<u64>) {
    let inst = instrumented(src);
    let (mut h, _posix, lh) = granted();
    let (whole, _) = run(&inst, &mut h, lh, &init_durable_window(WINDOW, ARENA));

    let (mut h, _posix, lh) = granted();
    let mut win = init_durable_window(WINDOW, ARENA);
    arm(&mut win);
    let (r, snap) = run(&inst, &mut h, lh, &win);
    assert_eq!(
        (r, h.take_freeze_declined()),
        (0, None),
        "the root unwinds for the freeze"
    );
    let carried: Vec<u64> = h.captured_twins().iter().map(|t| t.pid).collect();
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("the twins ride");

    let (mut th, thawed, back) = thaw(&art, &inst, lh);
    assert_eq!(
        thawed, carried,
        "the restore hands each twin back under its pid"
    );
    let (r, _) = run(&inst, &mut th, lh, &back);
    (whole, r, carried)
}

/// The parent forks, starts a thread, and `waitpid`s the twin; the twin waits on a futex nothing
/// notifies, then exits with `getpid() · 10 + 7`. The thread asks `getpid` 64 times, a suspend point
/// that gives its loop the poll a back-edge countdown fires at (#1937). One worker runs a run that
/// can freeze, so the parent is benched in its `waitpid` and the twin parked before the thread runs.
const WAITPID_PARKED: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vl: i32) {
  vpid = call.cap 13 51 () -> (i64) vl ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1(vl) 2(vl, vpid)
}
block 1 (vl1: i32) {
  va = i64.const 66000
  vw0 = i32.const 0
  vto = i64.const 200000000
  vw = i32.atomic.wait va vw0 vto
  vme = call.cap 13 44 () -> (i64) vl1 ()
  vten = i64.const 10
  vm10 = i64.mul vme vten
  vseven = i64.const 7
  vr = i64.add vm10 vseven
  return vr
}
block 2 (vl2: i32, vpid2: i64) {
  vz2 = i64.const 0
  vh = i64.extend_i32_u vl2
  vt = thread.spawn 1 vz2 vh
  vsp = i64.const 66100
  vgot = call.cap 13 28 (i64, i64, i64) -> (i64) vl2 (vpid2, vsp, vz2)
  vst32 = i32.load vsp
  vst = i64.extend_i32_u vst32
  veight = i64.const 8
  vcode = i64.shr_u vst veight
  vk = i64.const 1000
  vg = i64.mul vgot vk
  vres = i64.add vg vcode
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vz = i64.const 0
  br 1(vh, vz)
}
block 1 (vh1: i32, vi: i64) {
  vme = call.cap 13 44 () -> (i64) vh1 ()
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 64
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vh1, vi2) 2(vi2)
}
block 2 (vr: i64) {
  return vr
  }
}
"#;

/// The park × engine matrix's `waitpid` row (#1903): the parent is benched in `waitpid` on its live
/// twin when the thread's countdown fires the freeze. The census lets it pass, since the twin's bell
/// reaches it; the bench re-admits the parent under `UNWINDING`, which abandons the `waitpid` for the
/// thaw to re-issue; the twin unwinds from its futex and rides. Thawed, the parent's `waitpid` waits
/// on the re-created twin, pid 2 still, and reaps it.
pub fn waitpid_parked_at_the_trigger_thaws() {
    let (whole, thawed, carried) = freeze_and_thaw(WAITPID_PARKED, |win| {
        arm_freeze_after_backedges(win, ARENA, 16)
    });
    assert_eq!(whole, ANSWER, "uninterrupted: twin 2 exits 27");
    assert_eq!(carried, [2], "the twin rides live, parked on its futex");
    assert_eq!(thawed, ANSWER, "the re-issued waitpid reaps twin 2");
}
