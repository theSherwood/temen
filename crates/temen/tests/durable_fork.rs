//! #1688 — a **fork twin rides the artifact**. A twin is a domain of its own: its own window and
//! powerbox, reaped by its parent. A freeze that lands while one lives carries it the way it carries
//! a live detached child, keyed by its pid where a child is keyed by its join slot, and the thaw
//! re-creates it under the pid its parent's `fork()` returned (DURABILITY.md §4).
//!
//! The guest forks through the POSIX personality. The twin waits on a futex nothing notifies, then
//! exits with `getpid() · 10 + 7`. The parent loops, then `waitpid`s the twin and answers
//! `pid · 1000 + WEXITSTATUS`. The freeze lands mid-loop, so the twin is parked when the cut is
//! taken. Uninterrupted and thawed alike the answer is `2027`: the twin's pid is `2`, the first a
//! twin can take, and a thaw that gave it another would change both halves.

#[path = "support/fork_twin.rs"]
mod fork_twin;

use fork_twin::{
    freeze_and_thaw, granted, instrumented, run, servicer_twin_rides, thaw, TwinAtCut, Wait,
    ANSWER, ARENA, WINDOW,
};
use temen_durable::{arm_freeze_after_backedges, init_durable_window, read_state, STATE_UNWINDING};

/// `op(args)` on the personality: `call.cap HOST_PROC op`. `51` is `fork`, `44` `getpid`, `28`
/// `waitpid`.
const SRC: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vl: i32) {
  vpid = call.cap 13 51 () -> (i64) vl ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1(vl) 2(vl, vpid, vz)
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
block 2 (vl2: i32, vpid2: i64, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 64
  vmore = i64.ne vi2 vlim
  br_if vmore 2(vl2, vpid2, vi2) 3(vl2, vpid2)
}
block 3 (vl3: i32, vpid3: i64) {
  vsp = i64.const 66100
  vz3 = i64.const 0
  vgot = call.cap 13 28 (i64, i64, i64) -> (i64) vl3 (vpid3, vsp, vz3)
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
"#;

#[test]
fn a_live_fork_twin_rides_the_freeze_and_keeps_its_pid() {
    let inst = instrumented(SRC);

    let (mut h, _posix, lh) = granted();
    let (r, _) = run(&inst, &mut h, lh, &init_durable_window(WINDOW, ARENA));
    assert_eq!(r, ANSWER, "uninterrupted: twin 2 exits 27");

    // Frozen mid-loop, the twin parked on its futex.
    let (mut h, _posix, lh) = granted();
    let mut win = init_durable_window(WINDOW, ARENA);
    arm_freeze_after_backedges(&mut win, ARENA, 16);
    let (r, snap) = run(&inst, &mut h, lh, &win);
    assert_eq!(
        (r, h.take_freeze_declined()),
        (0, None),
        "the root unwinds for the freeze"
    );
    assert_eq!(
        read_state(&snap, ARENA),
        STATE_UNWINDING,
        "the cut was taken"
    );
    let carried: Vec<u64> = h.captured_twins().iter().map(|t| t.pid).collect();
    assert_eq!(carried, [2], "the twin rides live, parked on its futex");
    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("the twin rides");

    // Frozen again, from the thaw: the thawed twin is the run's twin 2 again, rung and carried.
    let (mut th, thawed, mut back) = thaw(&art, &inst, lh);
    assert_eq!(thawed, [2], "the restore hands the twin back under its pid");
    arm_freeze_after_backedges(&mut back, ARENA, 8);
    let (r, snap) = run(&inst, &mut th, lh, &back);
    assert_eq!(
        (r, th.take_freeze_declined()),
        (0, None),
        "the thaw freezes again"
    );
    let carried: Vec<u64> = th.captured_twins().iter().map(|t| t.pid).collect();
    assert_eq!(
        carried,
        [2],
        "the thawed twin keeps its pid in the run, too"
    );
    let art = temen_snapshot::freeze(&inst, &snap, &th).expect("the twin rides again");

    let (mut th, _, back) = thaw(&art, &inst, lh);
    let (r, _) = run(&inst, &mut th, lh, &back);
    assert_eq!(
        r, ANSWER,
        "the twice-thawed parent reaps the twice-thawed twin, pid 2"
    );
}

/// Freeze armed `backedges` back-edges in.
fn after(backedges: i64) -> impl Fn(&mut [u8]) {
    move |win| arm_freeze_after_backedges(win, ARENA, backedges)
}

/// [`SRC`] with a twin that spins instead of parking: it runs when the freeze lands, so only its
/// doorbell reaches it.
const SPINNING: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vl: i32) {
  vpid = call.cap 13 51 () -> (i64) vl ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1(vl, vz) 3(vl, vpid, vz)
}
block 1 (vl1: i32, vj: i64) {
  vone1 = i64.const 1
  vj2 = i64.add vj vone1
  vlim1 = i64.const 200000
  vmore1 = i64.ne vj2 vlim1
  br_if vmore1 1(vl1, vj2) 2(vl1)
}
block 2 (vl4: i32) {
  vme = call.cap 13 44 () -> (i64) vl4 ()
  vten = i64.const 10
  vm10 = i64.mul vme vten
  vseven = i64.const 7
  vr = i64.add vm10 vseven
  return vr
}
block 3 (vl2: i32, vpid2: i64, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 64
  vmore = i64.ne vi2 vlim
  br_if vmore 3(vl2, vpid2, vi2) 4(vl2, vpid2)
}
block 4 (vl3: i32, vpid3: i64) {
  vsp = i64.const 66100
  vz3 = i64.const 0
  vgot = call.cap 13 28 (i64, i64, i64) -> (i64) vl3 (vpid3, vsp, vz3)
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
"#;

#[test]
fn a_running_twin_is_rung_and_rides() {
    let (whole, thawed, carried) = freeze_and_thaw(SPINNING, after(16));
    assert_eq!(whole, ANSWER);
    assert_eq!(
        carried,
        [2],
        "the twin was mid-spin, and its bell brought it to the cut"
    );
    assert_eq!(
        thawed, ANSWER,
        "it spins on from where it stopped, under pid 2"
    );
}

/// The parent reaps its twin first and only then loops: the twin is the personality's to forget,
/// and the core forgets it with it, so the freeze carries no twin at all — and is not declined, as
/// it was while the twin lingered in the scheduler (#1688).
const REAPED: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vl: i32) {
  vpid = call.cap 13 51 () -> (i64) vl ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1(vl) 2(vl, vpid)
}
block 1 (vl1: i32) {
  vme = call.cap 13 44 () -> (i64) vl1 ()
  vten = i64.const 10
  vm10 = i64.mul vme vten
  vseven = i64.const 7
  vr = i64.add vm10 vseven
  return vr
}
block 2 (vl3: i32, vpid3: i64) {
  vsp = i64.const 66100
  vz3 = i64.const 0
  vgot = call.cap 13 28 (i64, i64, i64) -> (i64) vl3 (vpid3, vsp, vz3)
  vst32 = i32.load vsp
  vst = i64.extend_i32_u vst32
  veight = i64.const 8
  vcode = i64.shr_u vst veight
  vk = i64.const 1000
  vg = i64.mul vgot vk
  vres = i64.add vg vcode
  br 3(vl3, vres, vz3)
}
block 3 (vl2: i32, vres2: i64, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 64
  vmore = i64.ne vi2 vlim
  br_if vmore 3(vl2, vres2, vi2) 4(vres2)
}
block 4 (vres4: i64) {
  return vres4
}
}
"#;

#[test]
fn a_reaped_twin_is_gone_and_the_next_freeze_proceeds() {
    let (whole, thawed, carried) = freeze_and_thaw(REAPED, after(16));
    assert_eq!(whole, ANSWER);
    assert!(carried.is_empty(), "nothing of the reaped twin rides");
    assert_eq!(thawed, ANSWER);
}

/// The twin exits at once and the parent reaps it only after the cut: the process is a zombie in
/// the personality's table by then, which rides with the personality, and the core has let it go.
const ZOMBIE: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (vl: i32) {
  vpid = call.cap 13 51 () -> (i64) vl ()
  vz = i64.const 0
  vtwin = i64.eq vpid vz
  br_if vtwin 1(vl) 2(vl, vpid, vz)
}
block 1 (vl1: i32) {
  vme = call.cap 13 44 () -> (i64) vl1 ()
  vten = i64.const 10
  vm10 = i64.mul vme vten
  vseven = i64.const 7
  vr = i64.add vm10 vseven
  return vr
}
block 2 (vl5: i32, vpid5: i64, vk5: i64) {
  vone5 = i64.const 1
  vk52 = i64.add vk5 vone5
  vlim5 = i64.const 20000
  vmore5 = i64.ne vk52 vlim5
  br_if vmore5 2(vl5, vpid5, vk52) 3(vl5, vpid5, vk52)
}
block 3 (vl2: i32, vpid2: i64, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 20064
  vmore = i64.ne vi2 vlim
  br_if vmore 3(vl2, vpid2, vi2) 4(vl2, vpid2)
}
block 4 (vl3: i32, vpid3: i64) {
  vsp = i64.const 66100
  vz3 = i64.const 0
  vgot = call.cap 13 28 (i64, i64, i64) -> (i64) vl3 (vpid3, vsp, vz3)
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
"#;

#[test]
fn an_exited_unreaped_twin_rides_as_the_personalitys_zombie() {
    // The freeze lands after the twin has long exited: 20,000 back-edges in.
    let (whole, thawed, carried) = freeze_and_thaw(ZOMBIE, after(20_016));
    assert_eq!(whole, ANSWER);
    assert!(
        carried.is_empty(),
        "an exited twin is no process of the core's"
    );
    assert_eq!(
        thawed, ANSWER,
        "the thawed parent reaps the zombie that rode"
    );
}

/// #2571 — a servicer's `clone_caller` twin rides the freeze as a personality's does, under FORK.md
/// §8's manager topology: the twin is live at the cut, waiting out a futex, and its parent waits on
/// it, by pid (`join_waiters`) or for any child (`reap_any_waiters`). The twin's unwind wakes the
/// parent, which abandons its wait; the thaw re-creates the twin under its pid, and the re-issued
/// wait reaps it. A twin with a live detached child of its own carries it, and the thaw gives the
/// child a task id of its own: the twin's pid stays reserved until the whole tree is seeded (a thaw
/// once released it as it re-created the twin, and minted it again for the child).
#[test]
fn a_servicers_live_twin_rides_and_its_parents_wait_reissues() {
    for at in [TwinAtCut::Live, TwinAtCut::LiveWithChild] {
        for wait in [Wait::Pid, Wait::Any] {
            servicer_twin_rides(at, wait);
        }
    }
}

/// #2571 — a servicer's twin that exited before the cut waits in the core for its parent's `wait`,
/// so it rides as its outcome, a return or a crash, and the thaw files it again for that `wait` to
/// reap. The freeze rings only live twins, so it knows a crash before the cut from its own end.
#[test]
fn a_servicers_exited_twin_rides_as_its_outcome() {
    for at in [TwinAtCut::Exited, TwinAtCut::Crashed] {
        for wait in [Wait::Pid, Wait::Any] {
            servicer_twin_rides(at, wait);
        }
    }
}
