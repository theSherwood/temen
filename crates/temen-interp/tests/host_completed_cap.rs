//! #1366 — the **host-answered cap posture**: a cap call the *embedder* answers later while the
//! guest sees a plain synchronous `call.cap`.
//!
//! An offloadable handler returns [`OffloadOutcome::Host`] for the slow case; the dispatch mints a
//! completion id and hands the call's [`Completer`] to the submit hook (the host records the
//! request). On the cooperative multiplex driver ([`bytecode::CoopRun`], the engine behind the
//! browser's release session, #1953), a task or fiber parks on the call, the run tells of the call
//! once, as it is made ([`CoopEvent::CapPark`]), returns [`CoopEvent::Idle`] while nothing can run
//! until its answer, and the next pump after the answer resumes it (#1414 D4).
//! This is how a single-threaded embedder (the `wasm32` cdylib, where no offload pool can exist)
//! composes an asynchronous I/O powerbox. Pinned here: the park/answer round-trip on the root, a
//! spawned thread and a fiber, unsliced and sliced; byte-parity with the pool posture
//! (decline-never-diverge, INVARIANTS 9); the fail-closed decline where no caller can park (it must
//! not hang); and the §12 "sync ops never pay" pin. #2495 carries the posture to the drivers no
//! embedder pumps — the tree-walker and the native cooperative driver — where the embedder answers
//! from a thread of its own. #1414 5b runs the `CoopRun` cases on executor 2 too, where the run's
//! other tasks go on while the embedder answers.
//!
//! #1954 runs the **root program as an emitted leaf**: a root that parks only in declared
//! host-completed caps is offered to the host's leaf emitter, and each call parks in a bounce out of
//! the emitted frames, which the host holds until [`CoopEvent::Resume`]. Served here, as
//! `leaf_tierup.rs` does, by bouncing the entry — the nested interpretation the emitted image's own
//! calls bounce into.

#[path = "support/drivers.rs"]
mod drivers;

use drivers::{Pumped, PUMPED};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{
    self, CoopEvent, LeafOffer, ProgramEmit, ThreadPlatform, TierUpConfig,
};
use temen_interp::{
    run_with_host, BoundImport, CapRequests, Completer, Host, OffloadOutcome, Trap, Value,
};

/// Two `HOST_PROC` (iface 13) calls on one handle: op 0 with 5, op 1 with 7; the composite
/// `r0 * 1000 + r1` proves both results landed in the right slots and in order.
const TWO_CALLS: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vfive = i64.const 5
  vr0 = call.cap 13 0 (i64) -> (i64) vh (vfive)
  vseven = i64.const 7
  vr1 = call.cap 13 1 (i64) -> (i64) vh (vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
"#;

/// Both ops answer `arg + 100`: 105 and 107 → 105_107.
const WANT: i64 = 105_107;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module(TWO_CALLS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The host-answered requests the submit hook filed and nobody has answered yet: each call's
/// completer and the op's argument.
type Recorded = Arc<Mutex<Vec<(Completer, i64)>>>;

/// The filed requests, as `(completion id, argument)`.
fn filed(recorded: &Recorded) -> Vec<(u64, i64)> {
    let g = recorded.lock().unwrap();
    g.iter().map(|(c, a)| (c.id(), *a)).collect()
}

/// Answer the filed request `id` with its argument + 100: the argument, or `None` if none is
/// filed under `id`.
fn answer(recorded: &Recorded, id: u64) -> Option<i64> {
    let mut g = recorded.lock().unwrap();
    let at = g.iter().position(|(c, _)| c.id() == id)?;
    let (c, a) = g.remove(at);
    drop(g);
    c.complete(a + 100);
    Some(a)
}

/// Answer the next `n` requests from a thread of the embedder's own, each as it is filed: what a
/// driver no embedder pumps needs, since it returns only when the run ends.
fn answer_from_a_thread(recorded: &Recorded, n: usize) -> std::thread::JoinHandle<()> {
    let recorded = Arc::clone(recorded);
    std::thread::spawn(move || {
        for _ in 0..n {
            let id = loop {
                if let Some(&(id, _)) = filed(&recorded).first() {
                    break id;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            answer(&recorded, id);
        }
    })
}

/// The host-answered handler: op 0 answers inline (`Done`); op 1 punts to the *host*
/// (`Host`), whose submit hook files the call's completer and argument for the driver's embedder.
fn host_completed(recorded: &Recorded) -> (Host, i32) {
    let mut host = Host::new();
    let rec = Arc::clone(recorded);
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            let rec = Arc::clone(&rec);
            OffloadOutcome::Host(Box::new(move |c| rec.lock().unwrap().push((c, a))))
        }),
        None,
        temen_interp::CapState::Stateless,
    );
    (host, h)
}

/// The pool-completed twin: identical semantics, but op 1 punts a self-contained job to the
/// offload pool (`Offload`) — the posture the parity test compares against.
fn pool_completed() -> (Host, i32) {
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            OffloadOutcome::Offload(Box::new(move || a + 100))
        }),
        None,
        temen_interp::CapState::Stateless,
    );
    (host, h)
}

/// #2495 — where its caller could never park on it, a host-answered call **declines** with
/// `CapFault` before its embedder sees it, fail-closed — never blocking forever: in a durable run,
/// whose freeze has no cap park to re-derive, and on the tree-walker under a recording host, which
/// tapes no delivered value.
#[test]
fn a_host_answered_call_declines_where_its_caller_cannot_park() {
    declines(
        "a durable run on the tree-walker",
        |h| h.set_durable(true),
        true,
    );
    declines(
        "a durable run on the native coop driver",
        |h| h.set_durable(true),
        false,
    );
    declines(
        "a recording host on the tree-walker",
        |h| h.record_caps(),
        true,
    );
}

/// A run of [`module`] over a host `prepare`d so, on the tree-walker (`oracle`) or the native coop
/// driver, declines the host-answered call before the submit hook sees it.
fn declines(what: &str, prepare: fn(&mut Host), oracle: bool) {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    prepare(&mut host);
    let r = bounded(move || {
        let (args, mut fuel) = ([Value::I32(h)], 2_000_000_000u64);
        if oracle {
            run_with_host(&m, 0, &args, &mut fuel, &mut host)
        } else {
            bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
                .expect("in subset")
        }
    });
    assert_eq!(r, Err(Trap::CapFault), "{what}");
    assert!(
        filed(&recorded).is_empty(),
        "{what}: the declined call never reached the host's submit hook"
    );
}

/// #2495 — the tree-walk oracle, which no embedder pumps, parks the caller of a host-answered call
/// as it parks a pool call's, and the embedder answers from a thread of its own.
#[test]
fn the_oracle_parks_a_host_answered_call_until_its_embedder_answers() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let comps = host.completions();
    let answering = answer_from_a_thread(&recorded, 1);
    let r = bounded(move || {
        let mut fuel = 2_000_000u64;
        run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
    });
    answering.join().unwrap();
    assert_eq!(r, Ok(vec![Value::I64(WANT)]));
    assert_eq!(comps.outstanding(), 0, "the answered call settled");
}

// --- #1953: the cooperative multiplex driver (`CoopRun`) -----------------------------------------

/// Fuel for the cooperative runs: bounded, so a lost wake is a loud `OutOfFuel`, never a hang.
const FUEL: u64 = 2_000_000_000;

/// The root spawns a thread that makes the host-completed call (op 1, arg 7) while the root makes
/// the inline one (op 0, arg 5), then joins it: the same composite, `105 * 1000 + 107`, with the
/// park on a spawned thread. The thread gets the handle as its argument.
const THREADED: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vh64 = i64.extend_i32_u vh
  vt = thread.spawn 1 vz vh64
  vfive = i64.const 5
  vr0 = call.cap 13 0 (i64) -> (i64) vh (vfive)
  vr1 = thread.join vt
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  return vr
  }
}
"#;

/// The root runs a fiber that makes the host-completed call (op 1, arg 7) and `cont.resume.block`s
/// it, so the root idles on the parked fiber: status `1` (returned) and value 107 → `10_107`.
const FIBER: &str = r#"memory 16 shadow 16448 65536
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vh64 = i64.extend_i32_u v0
  vs, vv = cont.resume.block vk vh64
  vk4 = i64.const 10000
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk4
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  return vr
  }
}
"#;

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn coop(exec: Pumped, m: &temen_ir::Module, host: Host, h: i32) -> bytecode::CoopRun {
    exec.open(
        m,
        &[Value::I32(h)],
        FUEL,
        host,
        temen_ir::default_reserved_log2(),
    )
}

/// Pump a `CoopRun` to the end — unsliced, or in `slice`-op slices — answering every surfaced park
/// from what the submit hook filed (`arg + 100`). Returns the results and the parks surfaced.
fn drive_coop(
    run: &mut bytecode::CoopRun,
    recorded: &Recorded,
    slice: Option<u64>,
) -> (Vec<Value>, usize) {
    let mut parks = 0;
    loop {
        let ev = match slice {
            Some(n) => run.run_for(n),
            None => run.run(),
        };
        match ev {
            CoopEvent::Done(v) => return (v, parks),
            CoopEvent::Paused => {}
            CoopEvent::CapPark { id } => {
                parks += 1;
                answer(recorded, id)
                    .expect("the submit hook filed this id before the park surfaced");
            }
            CoopEvent::Trapped(t) => panic!("guest trapped: {t:?}"),
            _ => panic!("unexpected coop event"),
        }
    }
}

/// The round trip, on both executors (#1414 D4): the root parks on op 1, and the run tells of the
/// call once, as it is made, under the id the submit hook filed. With its answer owed and nothing
/// else to run, a pump hands back `Idle`, rather than tell of the call again or wait for it. The
/// answered value lands in the call's slot.
#[test]
fn a_host_answered_call_is_told_once_and_the_run_idles_until_its_answer() {
    for exec in PUMPED {
        let m = module();
        let (end, minted, outstanding) = bounded(move || {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let comps = host.completions();
            let mut run = coop(exec, &m, host, h);
            let id = match run.run() {
                CoopEvent::CapPark { id } => id,
                _ => panic!("{exec:?}: the host-answered call is told"),
            };
            assert_eq!(filed(&recorded), vec![(id, 7)], "{exec:?}");
            for _ in 0..2 {
                assert!(
                    matches!(run.run(), CoopEvent::Idle { timer: None }),
                    "{exec:?}: with the answer owed and nothing else to run, the run is idle"
                );
            }
            answer(&recorded, id);
            (run.run(), comps.minted(), comps.outstanding())
        });
        assert!(
            matches!(end, CoopEvent::Done(ref v) if *v == vec![Value::I64(WANT)]),
            "{exec:?}"
        );
        assert_eq!(
            minted, 1,
            "{exec:?}: one completion id minted (the Done op paid nothing)"
        );
        assert_eq!(outstanding, 0, "{exec:?}: the answered completion settled");
    }
}

/// #2531 — the root spins on a flag that its thread sets once its host-answered call returns:
/// `107`, the call's answer, through the join.
const SPIN_ON_FLAG: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vh64 = i64.extend_i32_u vh
  vt = thread.spawn 1 vz vh64
  br 1(vt)
}
block 1 (vt1: i32) {
  vaddr = i64.const 32768
  vflag = i64.load vaddr
  vzero = i64.const 0
  vwait = i64.eq vflag vzero
  br_if vwait 1(vt1) 2(vt1)
}
block 2 (vt2: i32) {
  vr = thread.join vt2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  vaddr = i64.const 32768
  vone = i64.const 1
  i64.store vaddr vone
  return vr
  }
}
"#;

/// #2531, #1414 D4 — a call is told while another task keeps running. A run that told of a call only
/// once nothing else could run never told of this one, as the root never stops spinning, and the
/// run never ended.
#[test]
fn a_call_is_told_while_another_task_keeps_running() {
    for exec in PUMPED {
        let m = parse(SPIN_ON_FLAG);
        let (v, parks) = bounded(move || {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let mut run = coop(exec, &m, host, h);
            drive_coop(&mut run, &recorded, None)
        });
        assert_eq!(v, vec![Value::I64(107)], "{exec:?}");
        assert_eq!(parks, 1, "{exec:?}: the call was told once");
    }
}

/// #1414 5b-2 — an embedder whose answers come from another of its threads waits out an idle
/// ([`bytecode::CoopRun::wait`]) rather than pump on at once: the answer's ring ends the wait, and
/// the next pump resumes the call. A browser's Worker pumps a run this way while its page answers.
#[test]
fn after_an_idle_the_embedder_waits_for_an_answer_from_another_thread() {
    let m = module();
    let (end, waits) = bounded(move || {
        let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
        let (host, h) = host_completed(&recorded);
        let mut run = coop(Pumped::Threads, &m, host, h);
        let id = match run.run() {
            CoopEvent::CapPark { id } => id,
            _ => panic!("the host-answered call is told"),
        };
        assert!(
            matches!(run.run(), CoopEvent::Idle { .. }),
            "with the answer owed, the run is idle"
        );
        let answerer = {
            let recorded = Arc::clone(&recorded);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_millis(50));
                answer(&recorded, id).expect("the call was filed");
            })
        };
        let mut waits = 0;
        let end = loop {
            run.wait();
            waits += 1;
            match run.run() {
                CoopEvent::Idle { .. } => {}
                ev => break ev,
            }
        };
        answerer.join().unwrap();
        (end, waits)
    });
    assert!(matches!(end, CoopEvent::Done(ref v) if *v == vec![Value::I64(WANT)]));
    // A wait that returned at once would have pumped thousands of times while the answer was due.
    assert!(
        waits <= 3,
        "the embedder waited for the answer: {waits} waits"
    );
}

/// The root and a thread it spawned each make a host-answered call (op 1, args 5 and 7), then the
/// root joins the thread: `105 * 1000 + 107`.
const TWO_PARKS: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vh64 = i64.extend_i32_u vh
  vt = thread.spawn 1 vz vh64
  vfive = i64.const 5
  vr0 = call.cap 13 1 (i64) -> (i64) vh (vfive)
  vr1 = thread.join vt
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  return vr
  }
}
"#;

/// #1414 D4 — each host-answered call is told once, on both executors: with one call told and not
/// yet answered, the next `run` tells of the other, and with both owed, the run is idle. The
/// embedder then answers both.
#[test]
fn each_host_answered_call_is_told_once_while_another_waits() {
    for exec in PUMPED {
        let m = parse(TWO_PARKS);
        let (r, told, filed_ids, idle) = bounded(move || {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let mut run = coop(exec, &m, host, h);
            let mut told: Vec<u64> = (0..2)
                .map(|_| match run.run() {
                    CoopEvent::CapPark { id } => id,
                    _ => panic!("{exec:?}: each host-answered call is told"),
                })
                .collect();
            let idle = matches!(run.run(), CoopEvent::Idle { timer: None });
            let mut filed_ids: Vec<u64> = filed(&recorded).into_iter().map(|(id, _)| id).collect();
            told.sort_unstable();
            filed_ids.sort_unstable();
            for &id in &told {
                answer(&recorded, id);
            }
            (run.run(), told, filed_ids, idle)
        });
        assert_eq!(told, filed_ids, "{exec:?}: each filed call was told once");
        assert!(idle, "{exec:?}: with both answers owed, the run is idle");
        assert!(
            matches!(r, CoopEvent::Done(ref v) if *v == vec![Value::I64(WANT)]),
            "{exec:?}"
        );
    }
}

/// #2557 — the root waits 1 ms on a word nobody notifies while its thread waits for a
/// host-answered call, reads the flag the thread raises once answered, and joins the thread:
/// `2` (timed out) * 10_000 + the flag * 1000 + the thread's 107.
const TIMER_WHILE_OWED: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vh64 = i64.extend_i32_u vh
  vt = thread.spawn 1 vz vh64
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 1000000
  vst = i32.atomic.wait vaddr vexp vto
  vflagat = i64.const 32768
  vflag = i64.load vflagat
  vr = thread.join vt
  vst64 = i64.extend_i32_s vst
  vk = i64.const 10000
  vs = i64.mul vst64 vk
  vk2 = i64.const 1000
  vf = i64.mul vflag vk2
  vsf = i64.add vs vf
  vsum = i64.add vsf vr
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  vaddr = i64.const 32768
  vone = i64.const 1
  i64.store vaddr vone
  return vr
  }
}
"#;

/// The timer of the [`CoopEvent::Idle`] a pump of `run` returns.
fn idle_timer(run: &mut bytecode::CoopRun) -> Option<u64> {
    match run.run() {
        CoopEvent::Idle { timer } => timer,
        _ => panic!("the run is idle until its answer"),
    }
}

/// #2557 — a timer fires while an answer is owed, once the embedder's time has reached it, on both
/// executors. The run hands back the time to the root's timer with `Idle`, and the embedder lets
/// that time pass before it answers: the root's wait times out before the answer comes. The pump's
/// clock waits for the embedder's: pumped again at once, it is idle with the same timer, since
/// its clock must not run ahead of an answer that is owed (#1414 D4). A pump that held its timers
/// while an answer was owed reported none here: the embedder answered at once, and the root read
/// the flag up (21_107).
#[test]
fn a_timer_fires_while_an_answer_is_owed() {
    for exec in PUMPED {
        let m = parse(TIMER_WHILE_OWED);
        let (end, timers) = bounded(move || {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let mut run = coop(exec, &m, host, h);
            let id = match run.run() {
                CoopEvent::CapPark { id } => id,
                _ => panic!("{exec:?}: the thread's call is told"),
            };
            let mut timers = vec![idle_timer(&mut run)];
            if exec == Pumped::Pump {
                timers.push(idle_timer(&mut run));
            }
            while let Some(&Some(ns)) = timers.last() {
                std::thread::sleep(std::time::Duration::from_nanos(ns));
                run.pass_time(ns);
                timers.push(idle_timer(&mut run));
            }
            answer(&recorded, id);
            (run.run(), timers)
        });
        assert!(
            matches!(end, CoopEvent::Done(ref v) if *v == vec![Value::I64(20_107)]),
            "{exec:?}: the root's wait timed out before the answer came"
        );
        if exec == Pumped::Pump {
            assert_eq!(timers, [Some(1_000_000), Some(1_000_000), None]);
        }
    }
}

/// A host that answers *inside* its submit hook (the request was serviceable synchronously after
/// all) never parks: the driver finds the result already posted and continues.
#[test]
fn host_completing_inside_submit_does_not_park() {
    for exec in PUMPED {
        let m = module();
        let mut host = Host::new();
        let comps = host.completions();
        let h = host.grant_host_proc_offloadable(
            Box::new(move |op, args| {
                let a = *args.first().unwrap_or(&0);
                if op == 0 {
                    return OffloadOutcome::Done(Ok(vec![a + 100]));
                }
                OffloadOutcome::Host(Box::new(move |c: Completer| c.complete(a + 100)))
            }),
            None,
            temen_interp::CapState::Stateless,
        );
        match bounded(move || coop(exec, &m, host, h).run()) {
            CoopEvent::Done(r) => assert_eq!(r, vec![Value::I64(WANT)], "{exec:?}"),
            CoopEvent::CapPark { .. } => {
                panic!("{exec:?}: an already-completed request must not park")
            }
            _ => panic!("{exec:?}: unexpected coop event"),
        }
        assert_eq!(comps.outstanding(), 0, "{exec:?}");
    }
}

/// §12 pin, host-completed form: a handler that always answers inline touches none of the
/// parking machinery — no id minted, nothing outstanding.
#[test]
fn sync_ops_never_pay_for_the_host_posture() {
    for exec in PUMPED {
        let m = module();
        let mut host = Host::new();
        let comps = host.completions();
        let h = host.grant_host_proc_offloadable(
            Box::new(|_op, args| OffloadOutcome::Done(Ok(vec![*args.first().unwrap_or(&0) + 100]))),
            None,
            temen_interp::CapState::Stateless,
        );
        match bounded(move || coop(exec, &m, host, h).run()) {
            CoopEvent::Done(r) => assert_eq!(r, vec![Value::I64(WANT)], "{exec:?}"),
            _ => panic!("{exec:?}: an all-inline handler never parks"),
        }
        assert_eq!(
            comps.minted(),
            0,
            "{exec:?}: no completion id was ever minted"
        );
        assert_eq!(comps.outstanding(), 0, "{exec:?}");
    }
}

/// #2495 — a completer dropped unanswered answers its call `CAP_REVOKED` (`-9`): a call its host
/// can no longer answer never hangs. The run tells of the park; dropping the request lets the run
/// end, the call having answered `-9` (`105 * 1000 - 9`).
#[test]
fn a_completer_dropped_unanswered_answers_cap_revoked() {
    for exec in PUMPED {
        let m = module();
        let end = bounded(move || {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let mut run = coop(exec, &m, host, h);
            assert!(matches!(run.run(), CoopEvent::CapPark { .. }), "{exec:?}");
            recorded.lock().unwrap().clear();
            run.run()
        });
        match end {
            CoopEvent::Done(r) => assert_eq!(r, vec![Value::I64(105 * 1000 - 9)], "{exec:?}"),
            _ => panic!("{exec:?}: the revoked call answers"),
        }
    }
}

/// **Host-completed ≡ pool-completed on `CoopRun`**, for each shape of parked caller — the root, a
/// spawned thread, and a fiber idled on by a blocking resume — unsliced and pumped in 1-op slices,
/// against the pool posture on the same driver and on the tree-walk oracle.
#[test]
fn coop_host_completed_matches_pool_completed() {
    for (name, m, want) in [
        ("root", module(), WANT),
        ("thread", parse(THREADED), WANT),
        ("fiber", parse(FIBER), 10_107),
    ] {
        for (exec, slice) in PUMPED.into_iter().flat_map(|e| [(e, None), (e, Some(1))]) {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let comps = host.completions();
            let m = m.clone();
            let (hosted, parks) = bounded(move || {
                let mut run = coop(exec, &m, host, h);
                drive_coop(&mut run, &recorded, slice)
            });
            assert_eq!(hosted, vec![Value::I64(want)], "{name} {exec:?} {slice:?}");
            assert_eq!(
                parks, 1,
                "{name} {exec:?} {slice:?}: exactly the host-completed call parked"
            );
            assert_eq!(comps.outstanding(), 0, "{name} {exec:?} {slice:?}");
        }
        let (host, h) = pool_completed();
        let mut run = coop(Pumped::Pump, &m, host, h);
        let pooled = match run.run() {
            CoopEvent::Done(r) => r,
            _ => panic!("{name}: the pool posture never parks the run"),
        };
        let (mut host, h) = pool_completed();
        let mut fuel = FUEL;
        let oracle = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
        host.quiesce_pool();
        assert_eq!(
            pooled,
            vec![Value::I64(want)],
            "{name}: pool ≡ host on CoopRun"
        );
        assert_eq!(
            oracle,
            vec![Value::I64(want)],
            "{name}: ≡ the tree-walk oracle"
        );
    }
}

/// #2495 — the native cooperative driver (the one-shot face with threads), which no embedder
/// pumps, parks a thread on a host-answered call as on a pool call, and the embedder answers from a
/// thread of its own.
#[test]
fn the_native_coop_driver_parks_a_host_answered_call_until_its_embedder_answers() {
    let m = parse(THREADED);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let answering = answer_from_a_thread(&recorded, 1);
    let r = bounded(move || {
        let mut fuel = FUEL;
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
            .expect("in subset")
    });
    answering.join().unwrap();
    assert_eq!(r, Ok(vec![Value::I64(WANT)]));
}

/// How many times the run below has read its clock (`counting_clock`).
static CLOCK_READS: AtomicUsize = AtomicUsize::new(0);

fn counting_clock() -> u64 {
    CLOCK_READS.fetch_add(1, Ordering::SeqCst);
    (ThreadPlatform::OS.now_ns)()
}

/// #2423 — on executor 2, a task waiting on a host-answered call sleeps until its answer comes:
/// the answer rings the run's bell, and the thread that opened the run settles the run, which wakes
/// the task. Counted through the run's clock, which a parked thread reads each time it looks:
/// meanwhile, while the answer is 100 ms away, its thread never looks. One that looked every
/// 50 µs read the clock thousands of times.
#[test]
fn a_task_waiting_on_a_host_answered_call_sleeps_until_its_answer() {
    let m = module();
    let reads_meanwhile = Arc::new(AtomicUsize::new(usize::MAX));
    let meanwhile = Arc::clone(&reads_meanwhile);
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            let meanwhile = Arc::clone(&meanwhile);
            OffloadOutcome::Host(Box::new(move |c| {
                std::thread::spawn(move || {
                    let before = CLOCK_READS.load(Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(100));
                    let after = CLOCK_READS.load(Ordering::SeqCst);
                    meanwhile.store(after - before, Ordering::SeqCst);
                    c.complete(a + 100);
                });
            }))
        }),
        None,
        temen_interp::CapState::Stateless,
    );
    host.set_thread_platform(ThreadPlatform {
        now_ns: counting_clock,
        ..ThreadPlatform::OS
    });
    let r = bounded(move || {
        let mut fuel = FUEL;
        bytecode::compile_and_run_capture_over_parallel_with_host(
            &m,
            0,
            &[Value::I32(h)],
            &mut fuel,
            &[],
            None,
            &mut host,
        )
        .expect("in subset")
        .0
    });
    assert_eq!(r, Ok(vec![Value::I64(WANT)]));
    let reads = reads_meanwhile.load(Ordering::SeqCst);
    assert!(
        reads < 50,
        "the waiting task's thread looked at the run {reads} times while its answer was 100 ms away"
    );
}

// --- #1954: the root program as an emitted leaf ----------------------------------------------------

/// The root calls the declared cap `ping` twice, through an import: `ping(5) * 1000 + ping(7)`.
const ROOT_PINGS: &str = r#"memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vfive = i64.const 5
  vr0 = call.import 0 (vfive)
  vseven = i64.const 7
  vr1 = call.import 0 (vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
"#;

/// The same `ping`, reached from function 1, which tiers up as a region (not a leaf): `ping(7)`.
const REGION_PING: &str = r#"memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vr = call 1 ()
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vseven = i64.const 7
  vr = call.import 0 (vseven)
  return vr
  }
}
"#;

/// A host granting `ping` as a declared host-completed cap, bound to import 0.
fn declared_ping(m: &temen_ir::Module) -> (Host, CapRequests) {
    let mut host = Host::new();
    let requests = CapRequests::default();
    let seams = host.grant_declared_host_caps(&m.imports, &["ping".to_string()], &requests);
    assert_eq!(seams.len(), 1, "ping is imported, so granted");
    host.set_import_bindings(vec![BoundImport::required(
        temen_ir::cap_id::HOST_PROC,
        0,
        seams[0].1,
    )]);
    (host, requests)
}

/// Offers seen by a leaf emitter: `(module, entry, parks)`.
type Offers = Arc<Mutex<Vec<(usize, u32, bool)>>>;

fn leaf_config(offers: &Offers, take: bool) -> TierUpConfig {
    let offers = Arc::clone(offers);
    TierUpConfig {
        root: ProgramEmit::default(),
        leaf: Some(Arc::new(move |o: &LeafOffer| {
            offers.lock().unwrap().push((o.module, o.entry, o.parks));
            take
        })),
        modules: None,
    }
}

/// What a run of `m` did: its result, the parks it surfaced, its tier-ups, and its resumes.
#[derive(Debug, PartialEq)]
struct Ran {
    end: Result<Vec<Value>, Trap>,
    parks: usize,
    tierups: usize,
    resumes: usize,
}

/// Pump `run` to the end, answering each park from the request its proc filed (`arg + 100`) and
/// serving each tier-up by bouncing its function, as `leaf_tierup.rs` does.
fn drive_leaf(mut run: bytecode::CoopRun, requests: &CapRequests) -> Ran {
    let (mut parks, mut tierups, mut resumes) = (0, 0, 0);
    let end = loop {
        match run.run() {
            CoopEvent::TierUp { func, argv, .. } => {
                tierups += 1;
                let mut io = argv.to_vec();
                io.resize(io.len().max(1), 0);
                match run.bounce(func, &mut io, None) {
                    Ok(Some(n)) => run.deliver_tierup(&io[..n]),
                    Ok(None) => {}
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
            CoopEvent::Resume { results } => {
                resumes += 1;
                run.deliver_tierup(&results);
            }
            CoopEvent::CapPark { id } => {
                parks += 1;
                let request = requests.lock().unwrap().remove(&id).expect("filed");
                request.completer.complete(request.args[0] + 100);
            }
            CoopEvent::Done(v) => break Ok(v),
            CoopEvent::Trapped(t) => break Err(t),
            _ => panic!("unexpected coop event"),
        }
    };
    Ran {
        end,
        parks,
        tierups,
        resumes,
    }
}

/// A run of `m` over a window **reserved at its declared size**. A root leaf needs a window emitted
/// code can address flat: unix's lazy `mmap` gives the default 128 GiB reservation one, but elsewhere
/// (Windows, wasm) that reservation is a sparse table, and only a reservation within the flat-buffer
/// cap is backed flat — so a run at the default runs its root interpreted there, by design.
fn coop_with(m: &temen_ir::Module, host: Host, tierup: Option<TierUpConfig>) -> bytecode::CoopRun {
    let declared = m.memory.expect("a window").size_log2;
    bytecode::CoopRun::new_reserved(m, 0, &[], FUEL, host, tierup, &[], declared)
        .expect("in subset")
        .expect("entry in range")
}

/// **The root runs as an emitted leaf and parks in it.** A root whose only parking calls are declared
/// host-completed caps is offered at its entry (module 0, parks = true), tiers up there, and each
/// call parks in a bounce out of the emitted frames: the park surfaces, the answer is delivered, and
/// `Resume` hands the frames the entry's results. The run ends exactly as the interpreted run does,
/// with the same parks.
#[test]
fn the_root_runs_as_an_emitted_leaf_and_parks_on_declared_caps() {
    let m = parse(ROOT_PINGS);
    let want = Ok(vec![Value::I64(WANT)]);

    let (host, requests) = declared_ping(&m);
    let interpreted = drive_leaf(coop_with(&m, host, None), &requests);
    assert_eq!(
        interpreted,
        Ran {
            end: want.clone(),
            parks: 2,
            tierups: 0,
            resumes: 0
        },
        "interpreted"
    );

    let offers: Offers = Arc::default();
    let (host, requests) = declared_ping(&m);
    let leaf = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, true))),
        &requests,
    );
    assert_eq!(
        offers.lock().unwrap().clone(),
        vec![(0, 0, true)],
        "the root, at its entry"
    );
    assert_eq!(
        leaf,
        Ran {
            end: want.clone(),
            parks: 2,
            tierups: 1,
            resumes: 1
        },
        "one tier-up at the entry, two parks inside it, one resume of its frames"
    );

    // Every constructor that holds the image offers it: `CoopRun::new` too, whose default 128 GiB
    // reservation only unix's lazy `mmap` backs flat (elsewhere that root interprets, by design).
    #[cfg(unix)]
    {
        let offers: Offers = Arc::default();
        let (host, requests) = declared_ping(&m);
        let run = bytecode::CoopRun::new(&m, 0, &[], FUEL, host, Some(leaf_config(&offers, true)))
            .expect("in subset")
            .expect("entry in range");
        let ran = drive_leaf(run, &requests);
        assert_eq!(
            offers.lock().unwrap().clone(),
            vec![(0, 0, true)],
            "CoopRun::new"
        );
        assert_eq!(
            (ran.end, ran.tierups, ran.resumes),
            (want, 1, 1),
            "CoopRun::new"
        );
    }
}

/// An emitter that declines the root (a host that cannot suspend its frames) leaves it interpreted,
/// with the same ending.
#[test]
fn a_declined_root_leaf_runs_interpreted() {
    let m = parse(ROOT_PINGS);
    let offers: Offers = Arc::default();
    let (host, requests) = declared_ping(&m);
    let ran = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, false))),
        &requests,
    );
    assert_eq!(offers.lock().unwrap().len(), 1, "offered once");
    assert_eq!(
        ran,
        Ran {
            end: Ok(vec![Value::I64(WANT)]),
            parks: 2,
            tierups: 0,
            resumes: 0
        }
    );
}

/// A tier-up **region** (not a leaf: nothing holds its frames) that reaches a host-completed call
/// declines with `CapFault` — it never waits on an id only the embedder could complete.
#[test]
fn a_host_completed_call_in_a_tierup_region_declines() {
    let m = parse(REGION_PING);
    let (host, requests) = declared_ping(&m);
    let tierup = TierUpConfig {
        root: ProgramEmit {
            eligible: Arc::from([false, true]),
            page_checked: false,
            ..Default::default()
        },
        leaf: None,
        modules: None,
    };
    let ran = drive_leaf(coop_with(&m, host, Some(tierup)), &requests);
    assert_eq!(
        ran,
        Ran {
            end: Err(Trap::CapFault),
            parks: 0,
            tierups: 1,
            resumes: 0
        }
    );
}

// --- #1954: what the root-leaf predicate admits -------------------------------------------------

/// A root that writes to stdout and exits through its imports — what every on-ramp C program
/// does — and imports a stdin read it never makes.
const ROOT_IO: &str = r#"memory 16
import 0 "write" (i64, i64) -> (i64)
import 1 "read" (i64, i64) -> (i64)
import 2 "exit" (i32) -> ()
data 16384 "hi\n"
func () -> (i64) {
block 0 () {
  vbuf = i64.const 16384
  vn = i64.const 3
  vw = call.import 0 (vbuf, vn)
  vseven = i32.const 7
  call.import 2 (vseven)
  unreachable
  }
}
"#;

/// A root that serves through `svc.wait` (self op 10), which parks until a caller arrives.
const ROOT_SVC_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

/// The §3e powerbox prefix bound to `write`/`read`/`exit`, as an on-ramp host binds them.
fn io_host(stdin_blocking: bool) -> Host {
    let mut host = Host::new();
    let [stdout, stdin, exit, _, _] = host.grant_powerbox_prefix(1 << 16);
    host.set_import_bindings(vec![
        BoundImport::required(temen_ir::cap_id::STREAM, 1, stdout),
        BoundImport::required(temen_ir::cap_id::STREAM, 0, stdin),
        BoundImport::required(temen_ir::cap_id::EXIT, 0, exit),
    ]);
    host.set_stdin_blocking(stdin_blocking);
    host
}

/// An `exit` never parks, and a stream import parks exactly as the same op inline: without a pipe
/// or a blocking stdin the root cannot park, so it is offered (`parks = false`) and runs whole; with
/// a blocking stdin its read can park, so it is offered only to a host that suspends.
#[test]
fn a_root_writing_and_exiting_through_its_imports_is_offered() {
    let m = parse(ROOT_IO);
    let none = CapRequests::default();

    let offers: Offers = Arc::default();
    let ran = drive_leaf(
        coop_with(&m, io_host(false), Some(leaf_config(&offers, true))),
        &none,
    );
    assert_eq!(offers.lock().unwrap().clone(), vec![(0, 0, false)]);
    assert_eq!((ran.end, ran.tierups), (Err(Trap::Exit(7)), 1));

    let offers: Offers = Arc::default();
    let _ = coop_with(&m, io_host(true), Some(leaf_config(&offers, false)));
    assert_eq!(
        offers.lock().unwrap().clone(),
        vec![(0, 0, true)],
        "a blocking stdin read can park"
    );
}

/// A root that reads its stdin with an inline `call.cap` (stream op 0) rather than an import.
const ROOT_INLINE_READ: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 1
  vbuf = i64.const 16384
  vn = i64.const 8
  vr = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vn)
  return vr
  }
}
"#;

/// #1414 S4-4a — a kept program reads its image's code once, and each offer still asks the run's
/// host. One program, compiled once and run over a host whose stdin cannot block, then over one
/// whose can, then the first kind again, is offered `parks = false`, `true`, `false`: the inline
/// read parks only on the blocking stdin.
#[test]
fn a_kept_programs_offers_ask_each_runs_host() {
    let m = parse(ROOT_INLINE_READ);
    let program = bytecode::compile_reserved(&m).expect("in subset");
    let declared = m.memory.expect("a window").size_log2;
    let offer = |stdin_blocking: bool| {
        let offers: Offers = Arc::default();
        let mut host = Host::new();
        host.set_stdin_blocking(stdin_blocking);
        let _ = bytecode::CoopRun::new_reserved_over_compiled(
            &m,
            Arc::clone(&program),
            0,
            &[],
            FUEL,
            host,
            Some(leaf_config(&offers, false)),
            &[],
            declared,
            None,
        )
        .expect("entry in range");
        let seen = offers.lock().unwrap().clone();
        seen
    };
    assert_eq!(offer(false), vec![(0, 0, false)]);
    assert_eq!(
        offer(true),
        vec![(0, 0, true)],
        "a blocking stdin read can park"
    );
    assert_eq!(offer(false), vec![(0, 0, false)], "and this host's cannot");
}

/// A self op that parks (`svc.wait`) keeps the root off the emitted tier: it is never offered.
#[test]
fn a_root_that_can_park_in_a_self_op_is_not_offered() {
    let m = parse(ROOT_SVC_WAIT);
    let offers: Offers = Arc::default();
    let _ = coop_with(&m, Host::new(), Some(leaf_config(&offers, true)));
    assert!(offers.lock().unwrap().is_empty());
}

/// A root that calls a host capability inline: a `call.cap` whose handle `h` is an operand, as
/// chibicc reaches its in-memory file system.
fn root_host_call(h: i32) -> temen_ir::Module {
    parse(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vh = i32.const {h}
  va = i64.const 5
  vr = call.cap 13 0 (i64) -> (i64) vh (va)
  return vr
  }}
}}
"
    ))
}

/// An inline host-capability call parks as any of the host's host capabilities could (#1414 S4-3b).
/// When each answers synchronously and no personality can ask one to park, the root cannot park:
/// it is offered (`parks = false`) and runs whole. An offloadable capability's call can punt, so a
/// host holding one keeps the root off the emitted tier.
#[test]
fn a_root_calling_a_host_capability_inline_is_offered_when_none_can_park() {
    let mut host = Host::new();
    let h = host.grant_host_proc(
        Box::new(|_op, args, _mem, _| Ok(vec![args[0] + 100])),
        temen_interp::CapState::Stateless,
    );
    let m = root_host_call(h);
    let offers: Offers = Arc::default();
    let ran = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, true))),
        &CapRequests::default(),
    );
    assert_eq!(offers.lock().unwrap().clone(), vec![(0, 0, false)]);
    assert_eq!((ran.end, ran.tierups), (Ok(vec![Value::I64(105)]), 1));

    let (host, h) = pool_completed();
    let m = root_host_call(h);
    let offers: Offers = Arc::default();
    let _ = coop_with(&m, host, Some(leaf_config(&offers, true)));
    assert!(
        offers.lock().unwrap().is_empty(),
        "an offloadable capability's call can punt"
    );
}

// --- #1722: a host-completed call whose reply cannot park -----------------------------------------

/// One host-completed call (op 1) declared with a reply that is not a single `i64`: C's `int`, as a
/// program declares an extern its embedder grants as a host cap, or no reply at all. A parked call
/// is delivered into one `i64` slot, so neither can park; each declines, never waiting on an answer
/// only the host could give.
fn narrow_call(reply: &str) -> temen_ir::Module {
    let (ret, sig) = match reply {
        "i32" => ("  vr64 = i64.extend_i32_s vr\n  return vr64\n", "(i32)"),
        _ => ("  vz = i64.const 0\n  return vz\n", "()"),
    };
    let call = if reply == "i32" {
        format!("  vr = call.cap 13 1 (i64) -> {sig} vh (vseven)\n")
    } else {
        format!("  call.cap 13 1 (i64) -> {sig} vh (vseven)\n")
    };
    parse(&format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (vh: i32) {{\n  vseven = i64.const 7\n{call}{ret}  }}\n}}\n"
    ))
}

/// Run `f` on its own thread, for at most 30 s: a call that waits on a completion only its host could
/// give would hang the test rather than fail it.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .expect("the run waited on a host-completed call that cannot park")
}

/// The trap each driver that parks host-answered calls ends a run of `m` with.
fn on_each_parking_driver(m: &temen_ir::Module) -> Vec<(&'static str, Option<Trap>)> {
    ["coop", "x2", "debug", "oracle"]
        .into_iter()
        .map(|driver| {
            let m = m.clone();
            let end = bounded(move || {
                let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
                let (mut host, h) = host_completed(&recorded);
                match driver {
                    "oracle" => {
                        run_with_host(&m, 0, &[Value::I32(h)], &mut 2_000_000, &mut host).err()
                    }
                    "coop" | "x2" => {
                        let exec = if driver == "x2" {
                            Pumped::Threads
                        } else {
                            Pumped::Pump
                        };
                        match coop(exec, &m, host, h).run() {
                            CoopEvent::Trapped(t) => Some(t),
                            _ => None,
                        }
                    }
                    _ => {
                        let args = [Value::I32(h)];
                        let mut run =
                            bytecode::ScheduledDebugRun::new_with_host(&m, 0, &args, host)
                                .expect("in subset");
                        match run.run_until_stop(&mut 2_000_000) {
                            bytecode::SchedStop::Finished(Err(t)) => Some(t),
                            _ => None,
                        }
                    }
                }
            });
            (driver, end)
        })
        .collect()
}

/// A host-answered call declared with an `int` reply declines with `CapFault` on every driver
/// that parks host-answered calls.
#[test]
fn a_host_completed_call_with_an_int_reply_declines() {
    for (driver, end) in on_each_parking_driver(&narrow_call("i32")) {
        assert_eq!(end, Some(Trap::CapFault), "{driver}");
    }
}

/// The same, declared with no reply.
#[test]
fn a_host_completed_call_with_no_reply_declines() {
    for (driver, end) in on_each_parking_driver(&narrow_call("none")) {
        assert_eq!(end, Some(Trap::CapFault), "{driver}");
    }
}
