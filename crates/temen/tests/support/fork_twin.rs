//! #1688 — the fork-twin freeze the durable tests share: a guest that forks through the POSIX
//! personality, a freeze on the tree-walker, the §12 codec, and a thaw into a fresh host whose
//! registrar rebuilds the personality ([`temen_posix::Thaw`]). #2571 — and the servicer lane's: a
//! guest that forks through a servicer's `clone_caller` under FORK.md §8's manager topology
//! ([`servicer_twin_rides`]). Included by `durable_fork.rs` and by `park_freeze_matrix.rs`, whose
//! `waitpid` row is [`waitpid_parked_at_the_trigger_thaws`] and whose `wait(-1)` row is a live
//! servicer twin's.
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

/// #2571 — where a servicer's twin is at the cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TwinAtCut {
    /// Live: it waits out a timed futex nothing notifies, and its parent waits on it.
    Live,
    /// Live as [`Self::Live`], with a live detached child of its own, `D`, which it spawned before
    /// its nap and joins after it. The twin exits with `D`'s `5` plus `2`.
    LiveWithChild,
    /// Exited before the cut, unreaped: its parent waits out the futex before it waits on it.
    Exited,
    /// Crashed before the cut, unreaped, as [`Self::Exited`]: it traps where it would exit, and
    /// its parent's `wait` answers the crash status.
    Crashed,
}

/// #2571 — how a servicer twin's parent waits on it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wait {
    /// `wait(pid)`, which parks in `join_waiters`.
    Pid,
    /// `wait(-1)`, which parks in `reap_any_waiters`.
    Any,
}

/// FORK.md §8's manager topology, durable. The manager (func 0) spawns the server `S` (func 1) and
/// the caller `C` (func 5), both detached, and grants `C` its `child_offer` over `S` as `"svc"`, and
/// what a spawn of a durable detached child of its own takes beyond its starter `"instantiator"` and
/// `"budget"`: `D`'s image (func 7, which naps and returns `5`) and freeze authority, as `"m"` and
/// `"f"`. `S`
/// serves `fork` (func 2, pid-mode `clone_caller`), `wait` (func 3, `reap`) and `status` (func 4),
/// and counts each fork or wait it serves without an errno. `C` forks, waits on its twin and answers
/// `pid · 1000 + status`. The manager asks `S`'s count until it reaches `sync`,
/// resumes a fiber 20 times, where the fiber countdown fires, then joins `C`. `twin` is the twin's
/// code after the fork, to its end, `orig` `C`'s between the fork and the wait, and `arg` the wait's
/// argument.
fn servicer_src(sync: i64, twin: &str, orig: &str, arg: &str) -> String {
    format!(
        r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface {{ fork: 0, wait: 0, status: 0 }}
export 0 interface "svc" 1 {{ fork: 2, wait: 3, status: 4 }}
data 66048 "svc"
data 66060 "mf"
data 66070 "instantiator"
data 66090 "budget"
func (i32, i32, i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, vb: i32, vsi: i32, vci: i32, vdi: i32, vfa: i32) {{
  qs = i64.const 65536
  qsv = i64.const -4294967296
  i64.store qs qsv
  i32.store qs vsi offset=8
  i32.store qs vb offset=12
  qneg = i64.const 4294967295
  i64.store qs qneg offset=48
  vs = call.cap 6 17 (i64) -> (i32) v0 (qs)
  vz = i64.const 0
  vsvc = call.cap 6 14 (i32, i64) -> (i32) v0 (vs, vz)
  ql = i64.const 66100
  vnp = i32.const 66048
  i32.store ql vnp
  vthree = i32.const 3
  i32.store ql vthree offset=4
  i32.store ql vsvc offset=8
  vone32 = i32.const 1
  vnm = i32.const 66060
  i32.store ql vnm offset=16
  i32.store ql vone32 offset=20
  i32.store ql vdi offset=24
  vnf = i32.const 66061
  i32.store ql vnf offset=32
  i32.store ql vone32 offset=36
  i32.store ql vfa offset=40
  qc = i64.const 65600
  i64.store qc qsv
  i32.store qc vci offset=8
  i32.store qc vb offset=12
  i64.store qc ql offset=16
  vgn = i64.const 3
  i64.store qc vgn offset=24
  i64.store qc qneg offset=48
  vc = call.cap 6 17 (i64) -> (i32) v0 (qc)
  br 1(v0, vc, vsvc)
}}
block 1 (vi1: i32, vc1: i32, vsvc1: i32) {{
  vx = i64.const 0
  vn = call.cap 268435456 2 (i64) -> (i64) vsvc1 (vx)
  vsync = i64.const {sync}
  vready = i64.ge_s vn vsync
  br_if vready 2(vi1, vc1) 1(vi1, vc1, vsvc1)
}}
block 2 (vi2: i32, vc2: i32) {{
  vf = ref.func 6
  vfsp = i64.const 4096
  vk = cont.new vf vfsp
  vz2 = i64.const 0
  br 3(vi2, vc2, vk, vz2)
}}
block 3 (vi3: i32, vc3: i32, vk3: i64, vj: i64) {{
  vs3, vx3 = cont.resume vk3 vj
  vone3 = i64.const 1
  vj2 = i64.add vj vone3
  vlim = i64.const 20
  vmore = i64.ne vj2 vlim
  br_if vmore 3(vi3, vc3, vk3, vj2) 4(vi3, vc3)
}}
block 4 (vi4: i32, vc4: i32) {{
  vst, vres = call.cap 6 1 (i32) -> (i64, i64) vi4 (vc4)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  br 1()
  }}
block 1 () {{
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  br 1()
  }}
}}
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  vz = i32.const 0
  vzero = i64.const 0
  vr = call.cap 4294967295 11 (i64) -> (i64) vz (vzero)
  vok = i64.ge_s vr vzero
  br_if vok 1(vr) 2(vr)
}}
block 1 (vr1: i64) {{
  pc = i64.const 70000
  vc = i64.load pc
  vone = i64.const 1
  vc2 = i64.add vc vone
  i64.store pc vc2
  br 2(vr1)
}}
block 2 (vr2: i64) {{
  return vr2
  }}
}}
func (i64) -> (i64) {{
block 0 (vpid: i64) {{
  vz = i32.const 0
  vst = call.cap 4294967295 12 (i64) -> (i64) vz (vpid)
  vzero = i64.const 0
  vok = i64.ge_s vst vzero
  br_if vok 1(vst) 2(vst)
}}
block 1 (vst1: i64) {{
  pc = i64.const 70000
  vc = i64.load pc
  vone = i64.const 1
  vc2 = i64.add vc vone
  i64.store pc vc2
  br 2(vst1)
}}
block 2 (vst2: i64) {{
  return vst2
  }}
}}
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  pc = i64.const 70000
  vc = i64.load pc
  return vc
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vp = i64.const 66048
  vl = i64.const 3
  vh = self.resolve vp vl
  br 1(vh)
}}
block 1 (vh1: i32) {{
  vz = i64.const 0
  vr = call.cap 268435456 0 (i64) -> (i64) vh1 (vz)
  vneg = i64.lt_s vr vz
  br_if vneg 1(vh1) 2(vh1, vr)
}}
block 2 (vh2: i32, vr2: i64) {{
  vz2 = i64.const 0
  vtwin = i64.eq vr2 vz2
  br_if vtwin 3() 4(vh2, vr2)
}}
block 3 () {{
{twin}
}}
block 4 (vh4: i32, vpid: i64) {{
{orig}
  br 5(vh4, vpid)
}}
block 5 (vh5: i32, vpid5: i64) {{
{arg}
  vst = call.cap 268435456 1 (i64) -> (i64) vh5 (varg)
  veagain = i64.const -11
  vretry = i64.eq vst veagain
  br_if vretry 5(vh5, vpid5) 6(vpid5, vst)
}}
block 6 (vpid6: i64, vst6: i64) {{
  vk = i64.const 1000
  vm = i64.mul vpid6 vk
  vres = i64.add vm vst6
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (v0: i64, v1: i64) {{
  br 1(v1)
}}
block 1 (va: i64) {{
  vn = suspend va
  br 1(vn)
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{NAP}
  vfive = i64.const 5
  return vfive
  }}
}}
"#
    )
}

/// The twin's spawn of `D`, through the caps `C` holds, which its powerbox copied, with a spawn
/// record at 65700 in its own window: `D`'s declared window, no pager or pre-mapped region.
const SPAWN_D: &str = r#"
  vpi = i64.const 66070
  vli = i64.const 12
  vti = self.resolve vpi vli
  vpm = i64.const 66060
  vl1 = i64.const 1
  vtm = self.resolve vpm vl1
  vpb = i64.const 66090
  vlb = i64.const 6
  vtb = self.resolve vpb vlb
  qd = i64.const 65700
  qdv = i64.const -4294967296
  i64.store qd qdv
  i32.store qd vtm offset=8
  i32.store qd vtb offset=12
  qdn = i64.const 4294967295
  i64.store qd qdn offset=48
  vd = call.cap 6 17 (i64) -> (i32) vti (qd)"#;

/// The twin's join of `D`, and its exit with `D`'s `5` plus `2`.
const JOIN_D: &str = r#"
  vdst, vdv = call.cap 6 1 (i32) -> (i64, i64) vti (vd)
  vtwo = i64.const 2
  vr7 = i64.add vdv vtwo
  return vr7"#;

/// A timed futex wait nothing notifies, `0.2 s`, in the domain's own window.
const NAP: &str = r#"
  vna = i64.const 66000
  vnw = i32.const 0
  vnt = i64.const 200000000
  vnr = i32.atomic.wait vna vnw vnt"#;

/// `S`, `C` and `D`'s child images of `src` (#2219), each instrumented as [`instrumented`]
/// instruments `src`, so its bootstrap is too (#2245).
fn servicer_images(src: &str) -> [Arc<temen_ir::Module>; 3] {
    let m = temen_text::parse_module(src).expect("parse");
    [1, 5, 7].map(|f| {
        let mut image = temen_ir::child_image_at(&m, f).expect("child image");
        image.memory = Some(Memory {
            size_log2: SIZE_LOG2,
            shadow: Some(ARENA),
        });
        let image = Arc::new(transform_module_assume_confined(&image).expect("transform"));
        temen_verify::verify_module(&image).expect("verify the image");
        image
    })
}

/// A durable host granting the manager's arguments: the instantiator, a budget, `S`'s, `C`'s and
/// `D`'s images, and freeze authority. One lane when `one_lane`, so the freeze run's order is the
/// scheduler's, not the OS's.
fn servicer_host(images: &[Arc<temen_ir::Module>; 3], one_lane: bool) -> (Host, Vec<Value>) {
    let mut h = Host::new();
    h.set_durable(true);
    if one_lane {
        h.set_lane_cap(1);
    }
    let inst = h.grant_instantiator();
    let budget = h.grant_budget(-1, 1 << 22, -1);
    let [s, c, d] = images.each_ref().map(|m| h.grant_durable_module(m));
    let freeze = h.grant_freeze_authority();
    (h, [inst, budget, s, c, d, freeze].map(Value::I32).to_vec())
}

/// Run the manager on `h` over `win`: what it answers, and the window it leaves.
fn servicer_run(
    inst: &temen_ir::Module,
    h: &mut Host,
    args: &[Value],
    win: &[u8],
) -> (Result<i64, temen_interp::Trap>, Vec<u8>) {
    let mut fuel = 50_000_000u64;
    let (r, snap) = run_capture_reserved_with_host(inst, 0, args, &mut fuel, win, SIZE_LOG2, h);
    let r = r.map(|v| match v[..] {
        [Value::I64(n)] => n,
        ref other => panic!("unexpected result {other:?}"),
    });
    (r, snap)
}

/// #2571 — the manager topology frozen with `C`'s twin `at` the cut, `C` waiting on it as `wait`
/// says, then thawed through the codec into a fresh host, twice: the twin rides as a live domain or
/// as its outcome, under the pid `C`'s `fork()` returned, and the thawed `C` reaps it. The answer is
/// `3000 + status` either way: the twin is task 3, after the manager, `S` and `C`.
pub fn servicer_twin_rides(at: TwinAtCut, wait: Wait) {
    let exits = "  vseven = i64.const 7\n  return vseven";
    let (sync, twin, orig, status) = match at {
        TwinAtCut::Live => (2, format!("{NAP}\n{exits}"), "", 7),
        TwinAtCut::LiveWithChild => (2, format!("{SPAWN_D}\n{NAP}\n{JOIN_D}"), "", 7),
        TwinAtCut::Exited => (1, exits.to_string(), NAP, 7),
        // A trapped twin reaps as the crash status, 128 (`temen_interp::reap_status`).
        TwinAtCut::Crashed => (1, "  unreachable".to_string(), NAP, 128),
    };
    let answer = 3000 + status;
    let arg = match wait {
        Wait::Pid => "  vz5 = i64.const 0\n  varg = i64.add vpid5 vz5",
        Wait::Any => "  varg = i64.const -1",
    };
    let src = servicer_src(sync, &twin, orig, arg);
    let inst = instrumented(&src);
    let images = servicer_images(&src);
    let ctx = format!("{at:?} twin, wait({wait:?})");

    let (mut h, args) = servicer_host(&images, false);
    let (whole, _) = servicer_run(&inst, &mut h, &args, &init_durable_window(WINDOW, ARENA));
    assert_eq!(
        whole,
        Ok(answer),
        "{ctx}: uninterrupted, twin 3 ends {status}"
    );

    let exited = |outcome| (vec![], vec![temen_interp::FrozenTwin { pid: 3, outcome }]);
    let expect = match at {
        TwinAtCut::Live | TwinAtCut::LiveWithChild => (vec![3], vec![]),
        TwinAtCut::Exited => exited(Ok(7)),
        TwinAtCut::Crashed => exited(Err(temen_interp::Trap::Unreachable)),
    };
    // Frozen twice, on one lane: from a fresh run, then from its thaw, where the twin is the one the
    // thaw re-created or filed again.
    let (mut h, args) = servicer_host(&images, true);
    let mut win = init_durable_window(WINDOW, ARENA);
    for round in 1..=2 {
        temen_durable::arm_freeze_after(&mut win, ARENA, 5);
        let (r, snap) = servicer_run(&inst, &mut h, &args, &win);
        assert_eq!(
            (r, h.take_freeze_declined()),
            (Ok(0), None),
            "{ctx}, round {round}: the manager unwinds for the freeze"
        );
        // `C` is the child that forked: the one with a twin on its powerbox.
        let (carried, children) = h
            .captured_detached()
            .iter()
            .map(|c| {
                let ch = c.host.lock().unwrap();
                let live: Vec<u64> = ch.captured_twins().iter().map(|t| t.pid).collect();
                let children: Vec<usize> = ch
                    .captured_twins()
                    .iter()
                    .map(|t| t.host.lock().unwrap().captured_detached().len())
                    .collect();
                ((live, ch.frozen_twins().to_vec()), children)
            })
            .find(|((live, exited), _)| !live.is_empty() || !exited.is_empty())
            .expect("C rides with its twin");
        assert_eq!(
            carried, expect,
            "{ctx}, round {round}: what rides of the twin"
        );
        if at == TwinAtCut::LiveWithChild {
            assert_eq!(
                children,
                [1],
                "{ctx}, round {round}: D rides under the twin"
            );
        }
        let art = temen_snapshot::freeze(&inst, &snap, &h).expect("the twin rides");
        h = servicer_host(&images, round == 1).0;
        win = temen_snapshot::restore(&art, &inst, &mut h).expect("restore");
        begin_thaw(&mut win, ARENA, 0);
    }
    let (r, _) = servicer_run(&inst, &mut h, &args, &win);
    assert_eq!(
        r,
        Ok(answer),
        "{ctx}: the twice-thawed C reaps its twin, pid 3"
    );
}

/// The park × engine matrix's `wait(-1)` row (#1903): the parent is parked in `reap_any_waiters` on
/// its servicer's live twin when the fiber countdown fires ([`servicer_twin_rides`]).
pub fn servicer_wait_any_parked_at_the_trigger_thaws() {
    servicer_twin_rides(TwinAtCut::Live, Wait::Any);
}
