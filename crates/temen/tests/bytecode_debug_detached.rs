//! Reverse-debugging a run with a §5 **detached** child on the multi-vCPU scheduler (#1866). A carve
//! child's window is a view of the root's, so its bytes ride the root's checkpoint image; a detached
//! child's window is its own. A checkpoint carries that window's image and the child's powerbox
//! (its handle table by the freeze's own capture, its inherited stdio aliased to the spawner's on
//! restore), and the run's budget tree, which the child's spawn charged. The undo journal records
//! none of a detached child's writes, so each of its anchors carries the child's window the same way
//! (#2058); and stdio a child's re-grant promoted into a shared cell is captured and rewound there
//! (#2055). A child of the debuggee's own program runs its spawner's unit, so breakpoints and §6
//! debug info reach it (#2076), and a debugger edit while a child is focused lands in the child's
//! window, live and on every replay (#2072).

use temen_interp::bytecode::{SchedStop, ScheduledDebugRun, ScheduledWrite};
use temen_interp::{Host, IrPc, StreamRole, Value, VarValue, WatchKind};
use temen_text::parse_module;

/// Root `(instantiator, module, budget, stdout) -> i64`: spawns the child (op 15, entry 0, a 64 KiB
/// window its budget pays for) with its stdout re-granted as `"stdout"`. Then 8 times it stores in
/// its window the budget's `mem` room, which the child's window charges; then it joins the child and
/// returns what it returns.
const ROOT: &str = r#"memory 17
data 17472 "stdout"
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, vout: i32) {
  vg = i64.const 17408
  vw = i64.const 25769821248
  i64.store vg vw
  vgh = i64.const 17416
  vout64 = i64.extend_i32_u vout
  i64.store vgh vout64
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vgn = i64.const 1
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vg, vgn, ve, vlog, vq)
  vz = i64.const 0
  br 1(v0, v2, vh, vz)
}
block 1 (vi: i32, vbu: i32, vch: i32, vn: i64) {
  vdim = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vbu (vdim)
  va = i64.const 17500
  i64.store va vroom
  vone = i64.const 1
  vn2 = i64.add vn vone
  vlim = i64.const 8
  vmore = i64.ne vn2 vlim
  br_if vmore 1(vi, vbu, vch, vn2) 2(vi, vch)
}
block 2 (vi2: i32, vch2: i32) {
  vj = call.cap 6 1 (i32) -> (i64) vi2 (vch2)
  return vj
  }
}
"#;

/// The child `(instantiator, address space) -> i64`: resolves its `"stdout"`, then 20 times adds the
/// count to a word of its window, writes a digit to stdout every fifth time, and halfway unmaps the
/// top page group of its window through its address space. Last it loads from that page, which
/// faults, so a replay that lost the page's protection would read zero and finish differently.
const CHILD: &str = r#"memory 16
data 16400 "stdout"
func (i64, i64) -> (i64) {
block 0 (vi0: i64, vas: i64) {
  vp = i64.const 16400
  vl = i64.const 6
  vo = self.resolve vp vl
  vz = i64.const 0
  br 1(vo, vas, vz)
}
block 1 (vo1: i32, vas1: i64, vi: i64) {
  va = i64.const 20000
  vx = i64.load va
  vx2 = i64.add vx vi
  i64.store va vx2
  vfive = i64.const 5
  vrem = i64.rem_u vi vfive
  vfour = i64.const 4
  vw = i64.eq vrem vfour
  br_if vw 2(vo1, vas1, vi) 3(vo1, vas1, vi)
}
block 2 (vo2: i32, vas2: i64, vi2: i64) {
  vf = i64.const 5
  vq = i64.div_u vi2 vf
  vzero = i64.const 48
  vd = i64.add vq vzero
  vd32 = i32.wrap_i64 vd
  vb = i64.const 20100
  i32.store8 vb vd32
  vn = i64.const 1
  vwr = call.cap 0 1 (i64, i64) -> (i64) vo2 (vb, vn)
  br 3(vo2, vas2, vi2)
}
block 3 (vo3: i32, vas3: i64, vi3: i64) {
  vten = i64.const 10
  vu = i64.eq vi3 vten
  br_if vu 4(vo3, vas3, vi3) 5(vo3, vas3, vi3)
}
block 4 (vo4: i32, vas4: i64, vi4: i64) {
  vuo = i64.const 49152
  vul = i64.const 16384
  vur = call.cap 5 1 (i64, i64) -> (i64) vas4 (vuo, vul)
  br 5(vo4, vas4, vi4)
}
block 5 (vo5: i32, vas5: i64, vi5: i64) {
  vone = i64.const 1
  vn5 = i64.add vi5 vone
  vlim = i64.const 20
  vmore = i64.ne vn5 vlim
  br_if vmore 1(vo5, vas5, vn5) 6()
}
block 6 () {
  vr = i64.const 20000
  vres = i64.load vr
  vgone = i64.const 49152
  vg8 = i64.load vgone
  vsum = i64.add vres vg8
  return vsum
  }
}
"#;

const FUEL: u64 = 50_000_000;

/// A run of `root` granting `child` as the module it spawns, with the journal armed or not.
fn session(root: &str, child: &str, journal: bool) -> ScheduledDebugRun {
    let root = parse_module(root).expect("parse the root");
    let child = parse_module(child).expect("parse the child");
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 17);
    let modh = host.grant_module(&child);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let out = host.grant_stream(StreamRole::Out);
    let args = [inst, modh, budget, out].map(Value::I32);
    let mut run = ScheduledDebugRun::new_with_host(&root, 0, &args, host).expect("in subset");
    run.set_journal_armed(journal);
    run
}

/// One turn's observation: the turn, every live task's call stack, and the page of each live task's
/// window that holds every byte these runs write.
fn observe(run: &mut ScheduledDebugRun) -> Turn {
    let mut seen = Vec::new();
    for tid in run.threads() {
        run.select_task(tid);
        let frames: Vec<String> = (0..run.depth())
            .filter_map(|d| run.frame_pc(d))
            .map(|pc| format!("m{}f{}b{}i{}", pc.module, pc.func, pc.block, pc.inst))
            .collect();
        seen.push((
            format!("{tid}: [{}]", frames.join(",")),
            run.read_window(16384, 4096).ok(),
        ));
    }
    (run.op_turn(), seen)
}

/// What [`observe`] sees at a turn.
type Turn = (u64, Vec<(String, Option<Vec<u8>>)>);

/// Run to the end: the result and everything the run wrote to stdout.
fn finish(run: &mut ScheduledDebugRun) -> (String, String) {
    let mut fuel = FUEL;
    let r = loop {
        match run.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => break r,
            SchedStop::Break { .. } => continue,
            other => panic!("unexpected stop {other:?}"),
        }
    };
    let out = String::from_utf8_lossy(&run.host_mut().take_stdout()).into_owned();
    (format!("{r:?}"), out)
}

/// #1866 — a run with a live detached child is checkpointable at **every** turn, and a restore at
/// any of them replays forward exactly as the uninterrupted run does: every task's stack and window
/// each turn, then the result and the output. The child's window rides its own image (the replay
/// reads its word back, and its last load still faults on the page it unmapped); its powerbox comes
/// back, its `"stdout"` aliased to the root's again, so its digits reach the root's stdout; and the
/// run's budget tree comes back charged for its window, so the root reads the same room the
/// uninterrupted run read. The digits the child wrote before a checkpoint ride it in the root's
/// promoted stdout (#2055).
#[test]
fn a_live_detached_child_rides_every_checkpoint() {
    let (result, out) = rides_every_checkpoint(CHILD);
    assert_eq!(
        (result.as_str(), out.as_str()),
        ("Err(MemoryFault)", "0123"),
        "the uninterrupted run: the child's last load faults on the page it unmapped, and its join \
         re-raises the fault"
    );
}

/// Checkpoint a run of [`ROOT`] spawning `child` at every turn, restore each checkpoint into a fresh
/// run, and check the restore lands where the uninterrupted run stood and replays forward exactly as
/// it did. Returns the uninterrupted run's result and output.
fn rides_every_checkpoint(child: &str) -> (String, String) {
    let mut refr = session(ROOT, child, false);
    let mut fuel = FUEL;
    let mut want = vec![observe(&mut refr)];
    let mut snaps = vec![refr.snapshot()];
    while refr.tick(&mut fuel) {
        want.push(observe(&mut refr));
        snaps.push(refr.snapshot());
    }
    let (result, out) = finish(&mut refr);

    let mut wrong = Vec::new();
    for (c, snap) in snaps.iter().enumerate() {
        let Ok(snap) = snap else {
            wrong.push(format!("turn {c}: no checkpoint"));
            continue;
        };
        let mut warm = session(ROOT, child, false);
        warm.restore(c as u64, snap);
        let mut i = c;
        if observe(&mut warm) != want[i] {
            wrong.push(format!("turn {c}: the restore lands elsewhere"));
            continue;
        }
        let mut fuel = FUEL;
        while warm.tick(&mut fuel) {
            i += 1;
            if want.get(i) != Some(&observe(&mut warm)) {
                wrong.push(format!("turn {c}: the replay diverges at turn {i}"));
                break;
            }
        }
        let got = finish(&mut warm);
        if got != (result.clone(), out.clone()) {
            wrong.push(format!("turn {c}: finishes as {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
    (result, out)
}

/// The child of [`an_undo_with_a_live_detached_child_matches_a_fresh_run`]: it adds 1 to a word of
/// its window 3000 times, long enough for the journal's stride to fall inside its life, and returns
/// the word.
const LONG_CHILD: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  va = i64.const 20000
  vx = i64.load va
  vone = i64.const 1
  vx2 = i64.add vx vone
  i64.store va vx2
  vi2 = i64.add vi vone
  vlim = i64.const 3000
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vi2) 2()
}
block 2 () {
  vb = i64.const 20000
  vr = i64.load vb
  return vr
  }
}
"#;

/// #2058 — an undo back into a live detached child's life finishes as a fresh run does. The journal
/// records none of the child's writes, which land in its own window, so the anchor an undo lands on
/// carries that window's image, as a checkpoint does. An anchor that rebuilt the child's window as a
/// view of the root's restarted the child's count from zero.
#[test]
fn an_undo_with_a_live_detached_child_matches_a_fresh_run() {
    let want = finish(&mut session(ROOT, LONG_CHILD, true));
    assert_eq!(want.0, "Ok([I64(3000)])");
    let mut total = 0u64;
    let mut r = session(ROOT, LONG_CHILD, true);
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {
        total += 1;
    }
    let mut wrong = Vec::new();
    for back in (1..total).step_by(997) {
        let mut r = session(ROOT, LONG_CHILD, true);
        let mut fuel = FUEL;
        while r.op_turn() < total - 1 && r.tick(&mut fuel) {}
        assert!(r.undo_to(back), "turn {back} is still in the journal");
        let got = finish(&mut r);
        if got != want {
            wrong.push(format!("undo to {back}: {got:?}"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}

/// The root of [`an_undo_truncates_output_a_child_promoted`]: it spawns a child that writes `X` to
/// the stdout it re-grants it, which promotes the root's stdout into a shared cell, joins it, spins
/// long enough for the journal to lay anchors, and writes `Y` to that stdout itself.
const PROMOTING_ROOT: &str = r#"memory 17
data 17472 "stdout"
func (i32, i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32, vout: i32) {
  vg = i64.const 17408
  vw = i64.const 25769821248
  i64.store vg vw
  vgh = i64.const 17416
  vout64 = i64.extend_i32_u vout
  i64.store vgh vout64
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vgn = i64.const 1
  ve = i64.const 0
  vlog = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vg, vgn, ve, vlog, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  vz = i64.const 0
  br 1(vout, vz)
}
block 1 (vo: i32, vi: i64) {
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 400
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vo, vi2) 2(vo)
}
block 2 (vo2: i32) {
  vb2 = i64.const 20000
  vy = i32.const 89
  i32.store8 vb2 vy
  vn = i64.const 1
  vwr = call.cap 0 1 (i64, i64) -> (i64) vo2 (vb2, vn)
  vr = i64.const 7
  return vr
  }
}
"#;

/// The child of [`an_undo_truncates_output_a_child_promoted`]: it writes `X` to its `"stdout"`.
const X_CHILD: &str = r#"memory 16
data 16400 "stdout"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vp = i64.const 16400
  vl = i64.const 6
  vo = self.resolve vp vl
  vb = i64.const 20000
  vx = i32.const 88
  i32.store8 vb vx
  vn = i64.const 1
  vwr = call.cap 0 1 (i64, i64) -> (i64) vo (vb, vn)
  vr = i64.const 0
  return vr
  }
}
"#;

/// #2055 — an undo truncates output in the cell a child's re-grant promoted the root's stdout into.
/// The child has ended, so the journal lays anchors again; an undo from after the root's `Y` to just
/// before it rewinds the cell to `X`, and the replay writes `Y` once. Truncating the root's own buffer
/// instead, which the promotion emptied, left `XY` in the cell for the replay to write `Y` after.
#[test]
fn an_undo_truncates_output_a_child_promoted() {
    let want = finish(&mut session(PROMOTING_ROOT, X_CHILD, true));
    assert_eq!(want, ("Ok([I64(7)])".to_string(), "XY".to_string()));
    let mut r = session(PROMOTING_ROOT, X_CHILD, true);
    let mut fuel = FUEL;
    while r.tick(&mut fuel) {}
    let end = r.op_turn();
    assert!(
        r.undo_to(end - 3),
        "the turns before the root's write are in the journal"
    );
    assert_eq!(finish(&mut r), want);
}

/// A root `(instantiator, budget) -> i64` that spawns a copy of itself (an op-17 v1 record, the
/// spawner's own module) and joins it. The copy, func 1, adds 1 to a word of its own window 3000
/// times and returns it, as [`LONG_CHILD`] does. The module's §6 debug info names that word `count`,
/// a global, so a variable write resolves to it in whichever task is focused.
const SELF_ROOT: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (vi: i32, vbud: i32) {
  vz = i64.const 0
  r0 = i64.const 4294967297
  a0 = i64.const 17536
  i64.store a0 r0
  a1 = i64.const 17544
  i64.store a1 vz
  r2 = i64.const -4294967296
  a2 = i64.const 17552
  i64.store a2 r2
  vb64 = i64.extend_i32_u vbud
  v32 = i64.const 32
  vbs = i64.shl vb64 v32
  vself = i64.const 4294967295
  r3 = i64.or vbs vself
  a3 = i64.const 17560
  i64.store a3 r3
  a4 = i64.const 17568
  i64.store a4 vz
  a5 = i64.const 17576
  i64.store a5 vz
  a6 = i64.const 17584
  i64.store a6 vz
  a7 = i64.const 17592
  i64.store a7 vz
  a8 = i64.const 17600
  i64.store a8 vz
  a9 = i64.const 17608
  i64.store a9 vself
  a10 = i64.const 17616
  i64.store a10 vz
  vh = call.cap 6 17 (i64) -> (i32) vi (a0)
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vn: i64) {
  va = i64.const 20000
  vx = i64.load va
  vone = i64.const 1
  vx2 = i64.add vx vone
  i64.store va vx2
  vn2 = i64.add vn vone
  vlim = i64.const 3000
  vmore = i64.ne vn2 vlim
  br_if vmore 1(vn2) 2()
}
block 2 () {
  vb = i64.const 20000
  vr = i64.load vb
  return vr
  }
}

debug.file 0 "count.c"
debug.fname 0 "main"
debug.fname 1 "child"
debug.var global "count" fixed 20000 "long"
"#;

/// A run of [`SELF_ROOT`], with the journal armed or not.
fn self_session(journal: bool) -> ScheduledDebugRun {
    let m = std::sync::Arc::new(parse_module(SELF_ROOT).expect("parse the root"));
    let mut host = Host::new();
    host.set_self_module(&m);
    let inst = host.grant_instantiator(0, 1 << 17);
    let budget = host.grant_budget(-1, 1 << 20, -1);
    let args = [inst, budget].map(Value::I32);
    let mut run = ScheduledDebugRun::new_with_host(&m, 0, &args, host).expect("in subset");
    run.set_journal_armed(journal);
    run
}

/// The head of the child's loop in [`SELF_ROOT`], where each of its 3000 iterations starts.
const LOOP_HEAD: IrPc = IrPc {
    module: 0,
    func: 1,
    block: 1,
    inst: 0,
};

/// Run `r` to the child's `n`th stop at [`LOOP_HEAD`], where the child is focused, and clear the
/// breakpoint. Returns the turn it stopped at.
fn stop_in_child(r: &mut ScheduledDebugRun, n: usize) -> u64 {
    r.set_breakpoints(vec![LOOP_HEAD]);
    let mut fuel = FUEL;
    for _ in 0..n {
        match r.run_until_stop(&mut fuel) {
            SchedStop::Break { pc, .. } => assert_eq!(pc, LOOP_HEAD),
            other => panic!("unexpected stop {other:?}"),
        }
    }
    assert_eq!(r.stopped_task(), Some(1), "the child stops there");
    r.set_breakpoints(Vec::new());
    r.op_turn()
}

/// The word the child counts in, read from the focused task's window.
fn count(r: &ScheduledDebugRun) -> u64 {
    u64::from_le_bytes(r.read_window(20000, 8).unwrap().try_into().unwrap())
}

/// #2076 — a detached child of the debuggee's own program runs its spawner's unit, so a breakpoint
/// in its code stops it, and its frames read the program's §6 debug info: a stop at each of its 3000
/// iterations, each reading the count so far. A child that ran a second copy of the program, a unit
/// of its own, never stopped, and its variables read as nothing.
#[test]
fn a_breakpoint_stops_a_child_of_its_own_program() {
    let mut r = self_session(false);
    r.set_breakpoints(vec![LOOP_HEAD]);
    let mut fuel = FUEL;
    let mut stops = 0u64;
    let result = loop {
        match r.run_until_stop(&mut fuel) {
            SchedStop::Break { pc, .. } => {
                assert_eq!((pc, r.stopped_task()), (LOOP_HEAD, Some(1)));
                assert_eq!(
                    r.read_var(0, "count", 8),
                    Some(VarValue::Bytes(stops.to_le_bytes().to_vec())),
                    "stop {stops}"
                );
                stops += 1;
            }
            SchedStop::Finished(res) => break res,
            other => panic!("unexpected stop {other:?}"),
        }
    };
    assert_eq!(
        (stops, format!("{result:?}")),
        (3000, "Ok([I64(3000)])".into())
    );
}

/// #2072 — a debugger write while the child is focused lands in the child's window, the one a read
/// there shows: a window write (`writeMemory`) and a variable write (`setVariable`) alike. The
/// root's word at the same address is untouched, and the child counts on from the write. Writing the
/// root's window instead left the child counting to 3000.
#[test]
fn a_write_to_a_focused_child_lands_in_its_window() {
    for var in [false, true] {
        let mut r = self_session(false);
        stop_in_child(&mut r, 100);
        let x = count(&r);
        let y = x + 1_000_000;
        let landed = if var {
            r.write_var(0, "count", y as i64, 8)
        } else {
            r.write_window(20000, &y.to_le_bytes())
        };
        assert!(landed, "var {var}");
        assert_eq!(
            count(&r),
            y,
            "var {var}: the child's window holds the write"
        );
        assert!(r.select_task(0));
        assert_eq!(count(&r), 0, "var {var}: the root's does not");
        assert_eq!(finish(&mut r).0, "Ok([I64(1003000)])", "var {var}");
    }
}

/// #2072 — a write recorded against the child, as the DAP backend records `writeMemory` and
/// `setVariable`, lands in the child's window on every path that passes its turn: a fresh run, a
/// restore from a checkpoint taken before the turn, and an undo from after it, which lands where a
/// fresh run stands.
#[test]
fn a_childs_recorded_writes_replay_into_its_window() {
    let mut r = self_session(false);
    let turn = stop_in_child(&mut r, 100);
    let x = count(&r);
    let writes = [
        (
            ScheduledWrite::Window {
                task: 1,
                addr: 20000,
                bytes: (x + 1_000_000).to_le_bytes().to_vec(),
            },
            "Ok([I64(1003000)])",
        ),
        (
            ScheduledWrite::Var {
                task: 1,
                frame: 0,
                name: "count".into(),
                value: (x + 2_000_000) as i64,
                width: 8,
            },
            "Ok([I64(2003000)])",
        ),
    ];
    for (write, want) in writes {
        let with = |journal| {
            let mut r = self_session(journal);
            r.set_scheduled_writes(vec![(turn, write.clone())]);
            r
        };
        assert_eq!(finish(&mut with(false)).0, want, "{write:?}: a fresh run");

        let mut fuel = FUEL;
        let mut a = with(false);
        while a.op_turn() < turn - 500 {
            assert!(a.tick(&mut fuel));
        }
        let snap = a.snapshot().expect("a checkpoint");
        let mut b = with(false);
        b.restore(turn - 500, &snap);
        assert_eq!(finish(&mut b).0, want, "{write:?}: a restore before it");

        let mut u = with(true);
        while u.op_turn() < turn + 2000 {
            assert!(u.tick(&mut fuel));
        }
        assert!(u.undo_to(turn - 500), "the turn is in the journal");
        let mut fresh = with(false);
        while fresh.op_turn() < turn - 500 {
            assert!(fresh.tick(&mut fuel));
        }
        assert_eq!(
            observe(&mut u),
            observe(&mut fresh),
            "{write:?}: an undo lands where a fresh run stands"
        );
        assert_eq!(finish(&mut u).0, want, "{write:?}: an undo across it");
    }
}

/// #2076 — the journal's pre-images are of the root's window alone. The child of [`SELF_ROOT`] runs
/// module 0 too, but in its own window, which each anchor carries whole (#2058); recording its 3000
/// stores against the root's window would hold pre-images of bytes nothing there wrote.
#[test]
fn the_journal_records_only_the_root_windows_writes() {
    let mut ticked = self_session(true);
    let mut fuel = FUEL;
    while ticked.tick(&mut fuel) {}
    let mut driven = self_session(true);
    finish(&mut driven);
    for r in [ticked, driven] {
        assert_eq!(
            r.journal_stats().appended,
            11,
            "the root's stores of the spawn record"
        );
    }
}

/// A child `(instantiator, address space) -> i64` that spawns a copy of its own program with the
/// budget its spawn paid with, `"budget"` (an op-17 v1 record, `module = -1`), and returns what the
/// copy returns. The copy, func 1, adds 1 to a word of its own window 20 times and returns it.
const SPAWNING_CHILD: &str = r#"memory 16
data 16400 "budget"
func (i64, i64) -> (i64) {
block 0 (vi: i64, vas: i64) {
  vp = i64.const 16400
  vl = i64.const 6
  vbud = self.resolve vp vl
  vz = i64.const 0
  r0 = i64.const 4294967297
  a0 = i64.const 17536
  i64.store a0 r0
  a1 = i64.const 17544
  i64.store a1 vz
  r2 = i64.const -4294967296
  a2 = i64.const 17552
  i64.store a2 r2
  vb64 = i64.extend_i32_u vbud
  v32 = i64.const 32
  vbs = i64.shl vb64 v32
  vself = i64.const 4294967295
  r3 = i64.or vbs vself
  a3 = i64.const 17560
  i64.store a3 r3
  a4 = i64.const 17568
  i64.store a4 vz
  a5 = i64.const 17576
  i64.store a5 vz
  a6 = i64.const 17584
  i64.store a6 vz
  a7 = i64.const 17592
  i64.store a7 vz
  a8 = i64.const 17600
  i64.store a8 vz
  a9 = i64.const 17608
  i64.store a9 vself
  a10 = i64.const 17616
  i64.store a10 vz
  vh = call.cap 6 17 (i64) -> (i32) vi (a0)
  vj = call.cap 6 1 (i32) -> (i64) vi (vh)
  return vj
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vn: i64) {
  va = i64.const 20000
  vx = i64.load va
  vone = i64.const 1
  vx2 = i64.add vx vone
  i64.store va vx2
  vn2 = i64.add vn vone
  vlim = i64.const 20
  vmore = i64.ne vn2 vlim
  br_if vmore 1(vn2) 2()
}
block 2 () {
  vb = i64.const 20000
  vr = i64.load vb
  return vr
  }
}
"#;

/// #2076 — a child of a child's own program runs that child's unit, not the root's: the grandchild
/// runs the child's func 1 and returns its count. And the run checkpoints at every turn, each env
/// rebuilt under the one that spawned it — the grandchild's powerbox under the child's, the child's
/// under the root's — with the unit it runs.
#[test]
fn a_child_of_a_childs_own_program_rides_every_checkpoint() {
    assert_eq!(
        rides_every_checkpoint(SPAWNING_CHILD),
        ("Ok([I64(20)])".to_string(), String::new())
    );
}

/// #2076 — the window watches and the access sink name the root's window. The child of
/// [`SELF_ROOT`] runs module 0 too, but in its own window: a write watch on the word it counts in
/// never stops it, and the sink sees none of its accesses, only the root's.
#[test]
fn window_watches_and_the_access_sink_see_the_roots_window_only() {
    let mut r = self_session(false);
    r.set_watchpoints(vec![(20000, 8, WatchKind::Write)]);
    let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&seen);
    r.set_access_sink(Box::new(move |_, task, _| sink.lock().unwrap().push(task)));
    let mut fuel = FUEL;
    let mut stops = 0;
    let result = loop {
        match r.run_until_stop(&mut fuel) {
            SchedStop::Break { .. } => stops += 1,
            SchedStop::Finished(res) => break res,
            other => panic!("unexpected stop {other:?}"),
        }
    };
    assert_eq!(
        (stops, format!("{result:?}")),
        (0, "Ok([I64(3000)])".into())
    );
    // A replay by `tick` feeds the sink the same way.
    let mut ticked = self_session(false);
    let sink = std::sync::Arc::clone(&seen);
    ticked.set_access_sink(Box::new(move |_, task, _| sink.lock().unwrap().push(task)));
    while ticked.tick(&mut fuel) {}
    let seen = seen.lock().unwrap();
    assert!(
        !seen.is_empty() && seen.iter().all(|&task| task == 0),
        "the root's accesses alone"
    );
}
