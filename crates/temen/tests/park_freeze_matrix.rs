//! **#1903 — the park × engine freeze matrix** (DURABILITY §4, "Freeze liveness").
//!
//! Every place a vCPU can park is a [`ParkSite`] with one [`FreezeRule`] (#1898). This file holds
//! one row per `(site, engine)`, generated from [`ParkSite::ALL`] through an exhaustive match, so a
//! new site cannot be added without deciding its row:
//!
//! - **`Case`** runs it. At a `Reissue` or `Phase` site the root parks there, a sibling triggers the
//!   freeze, and the thaw must give the uninterrupted answer. At a `Decline` site the freeze must be
//!   declined and the run carry on untouched.
//! - **`Pending`** names the issue that builds the row.
//! - **`Unreachable`** says why the engine never parks there during a durable run.
//!
//! Each engine has a runner ([`run`]). The oracle and the JIT keep the fiber-safepoint countdown
//! ([`arm_freeze_after`]), which the rows that fire the freeze in a sibling use; the oracle and the
//! bytecode engine keep freeze-on-quiesce ([`arm_freeze_on_quiesce`]), which the one-vCPU rows use.

use std::sync::{Arc, Mutex};
use std::time::Duration;
use temen_durable::{
    arm_freeze_after, arm_freeze_on_quiesce, begin_thaw, init_durable_window,
    transform_module_assume_confined, ARM_QUIESCE_OFF,
};
use temen_interp::{
    run_capture_reserved_with_host, CapState, FreezeRule, FreezeScope, Host, ParkSite,
    SignalSource, StreamRole, Value,
};
use temen_ir::{Memory, Module};
use temen_jit::JitOutcome;

const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);
const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Engine {
    Interp,
    Bytecode,
    Jit,
}

const ENGINES: [Engine; 3] = [Engine::Interp, Engine::Bytecode, Engine::Jit];

enum Row {
    Case(fn(ParkSite, Engine)),
    Pending { issue: u32, why: &'static str },
    Unreachable { why: &'static str },
}

/// The one row per `(site, engine)`.
fn row(site: ParkSite, engine: Engine) -> Row {
    match engine {
        Engine::Interp => interp_row(site),
        Engine::Jit => jit_row(site),
        Engine::Bytecode => bytecode_row(site),
    }
}

fn interp_row(site: ParkSite) -> Row {
    match site {
        ParkSite::Futex => Row::Case(futex),
        ParkSite::Join => Row::Case(join),
        ParkSite::PipeRead => Row::Case(pipe_read),
        ParkSite::PipeWrite => Row::Case(pipe_write),
        ParkSite::StreamRead => Row::Case(stream_read),
        ParkSite::Stopped => Row::Case(stopped),
        ParkSite::Svc => Row::Case(svc),
        ParkSite::Reap => Row::Unreachable {
            why: "a blocking `waitpid` waits on a fork twin or a `pspawn`ed process, which declines \
                  the freeze first (#1688); the delegate `spawn` runs its child through the embedder \
                  and never parks",
        },
        ParkSite::Lane => Row::Unreachable {
            why: "a run that can freeze is serialized onto one worker, and a task gives its lane back \
                  when it parks, so no lane is ever full",
        },
        ParkSite::Reply => Row::Case(reply),
        ParkSite::PageFault => PAGE_FAULT_PENDING,
        ParkSite::Completion => COMPLETION_PENDING,
        ParkSite::ReapAny => Row::Unreachable {
            why: "`wait(-1)` needs a live fork twin, which declines the freeze first (#1688)",
        },
        ParkSite::Admit => ADMIT_UNREACHABLE,
    }
}

/// The JIT's rows, run through the embedder's durable entry (`temen_run::jit_cap_run`).
fn jit_row(site: ParkSite) -> Row {
    match site {
        ParkSite::Futex => Row::Case(futex),
        ParkSite::Join => Row::Case(join),
        ParkSite::PipeRead => Row::Case(pipe_read),
        ParkSite::PipeWrite => Row::Case(pipe_write),
        ParkSite::StreamRead => Row::Pending {
            issue: 1904,
            why: "the row's trigger is freeze-on-quiesce, which the JIT does not implement",
        },
        ParkSite::Stopped => Row::Pending {
            issue: 1826,
            why: "the JIT serves no job control",
        },
        ParkSite::Svc => Row::Unreachable {
            why:
                "the JIT's serve loop runs synchronously and never parks: an empty `svc.wait` has \
                  nothing that could enqueue mid-run, and faults `ThreadFault`",
        },
        ParkSite::Reap | ParkSite::ReapAny => Row::Unreachable {
            why: "only a JIT process tree (`jit_proc`) serves `waitpid`, and it never runs durable",
        },
        ParkSite::Lane => Row::Pending {
            issue: 1904,
            why: "a lane cap comes only from a granted §14 child; needs a granted-child runner",
        },
        ParkSite::Reply => Row::Pending {
            issue: 1904,
            why:
                "the JIT's `live_impl_call` reply park: its re-park on thaw, and the re-link of a \
                  `LiveImpl` onto a re-launched detached child",
        },
        ParkSite::PageFault => PAGE_FAULT_PENDING,
        ParkSite::Completion => COMPLETION_PENDING,
        ParkSite::Admit => ADMIT_UNREACHABLE,
    }
}

const PAGE_FAULT_PENDING: Row = Row::Pending {
    issue: 1940,
    why: "a faulting access is no call to re-issue, and cannot unwind until its page arrives",
};

const COMPLETION_PENDING: Row = Row::Pending {
    issue: 1902,
    why: "a punted host call is in flight outside the cut; its capability declares what a freeze does",
};

/// The bytecode engine's rows, run through its durable entry
/// (`bytecode::compile_and_run_capture_reserved_with_host`). That entry runs one vCPU: it refuses
/// `thread.*`, so the rows whose freeze fires in a sibling cannot run on it.
fn bytecode_row(site: ParkSite) -> Row {
    const NO_THREADS: &str =
        "the durable bytecode entry refuses `thread.*`, and this row's park waits \
                              on a sibling vCPU";
    match site {
        ParkSite::Svc => Row::Case(svc),
        ParkSite::StreamRead => Row::Case(stream_read),
        ParkSite::Join => Row::Unreachable { why: NO_THREADS },
        ParkSite::Futex => Row::Case(futex_alone),
        ParkSite::PipeRead | ParkSite::PipeWrite => Row::Pending {
            issue: 1904,
            why: "one vCPU's pipe park waits on the outside, and a pipe fed from outside the domain \
                  crosses the cut (#1680)",
        },
        ParkSite::Stopped => Row::Case(stopped),
        ParkSite::Reap | ParkSite::ReapAny => Row::Pending {
            issue: 1904,
            why: "a `waitpid` waits on a fork twin, and the bytecode census has no fork-twin decline \
                  outside freeze-on-quiesce",
        },
        ParkSite::Lane => Row::Unreachable {
            why: "the cooperative pump runs one task at a time and has no lanes",
        },
        ParkSite::Reply => Row::Pending {
            issue: 1904,
            why: "a live call's reply park: its re-park on thaw (#1901)",
        },
        ParkSite::PageFault => PAGE_FAULT_PENDING,
        ParkSite::Completion => COMPLETION_PENDING,
        ParkSite::Admit => ADMIT_UNREACHABLE,
    }
}

const ADMIT_UNREACHABLE: Row = Row::Unreachable {
    why: "a durable caller never animates an offer, so never waits to enter one (#1681)",
};

/// Every row: run the cases, and list what is not built yet. Fails on any case that fails.
#[test]
fn every_park_site_on_every_engine() {
    let mut gaps = Vec::new();
    for site in ParkSite::ALL {
        for engine in ENGINES {
            match row(site, engine) {
                Row::Case(run) => {
                    eprintln!("{site:?} × {engine:?}: running");
                    run(site, engine)
                }
                Row::Pending { issue, why } => {
                    gaps.push(format!("{site:?} × {engine:?}: #{issue} ({why})"))
                }
                Row::Unreachable { why } => {
                    eprintln!("{site:?} × {engine:?}: unreachable: {why}")
                }
            }
        }
    }
    eprintln!("pending rows:\n  {}", gaps.join("\n  "));
}

fn instrumented(src: &str) -> Arc<Module> {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = Arc::new(transform_module_assume_confined(&m).expect("transform"));
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

fn durable_host(inst: &Arc<Module>) -> Host {
    let mut h = Host::new();
    h.set_durable(true);
    h.set_self_module(inst);
    h
}

/// What a run answers, engine-neutral: the returned `i64`, or the trap's name.
type Answer = Result<i64, String>;

/// Run `inst` on `engine` with `h` over `win` on a thread, failing (not hanging) after 20 s.
fn run(
    engine: Engine,
    inst: &Arc<Module>,
    args: &[Value],
    win: &[u8],
    mut h: Host,
) -> (Answer, Vec<u8>, Host) {
    let (tx, rx) = std::sync::mpsc::channel();
    let (inst, args, win) = (inst.clone(), args.to_vec(), win.to_vec());
    std::thread::spawn(move || {
        let (r, snap) = match engine {
            Engine::Interp => {
                let mut fuel = 10_000_000u64;
                let (r, snap) = run_capture_reserved_with_host(
                    &inst, 0, &args, &mut fuel, &win, SIZE_LOG2, &mut h,
                );
                let r = match r {
                    Ok(v) => match v[..] {
                        [Value::I64(n)] => Ok(n),
                        ref other => panic!("unexpected result {other:?}"),
                    },
                    Err(t) => Err(format!("{t:?}")),
                };
                (r, snap)
            }
            Engine::Jit => {
                let slots: Vec<i64> = args
                    .iter()
                    .map(|v| match v {
                        Value::I32(x) => *x as i64,
                        Value::I64(x) => *x,
                        other => panic!("unsupported arg {other:?}"),
                    })
                    .collect();
                let layout = temen_interp::MemLayout::image(win);
                match temen_run::jit_cap_run(&inst, 0, &slots, &layout, SIZE_LOG2, 0, &mut h, None)
                {
                    Ok((JitOutcome::Returned(v), snap)) => (Ok(v[0]), snap.bytes().to_vec()),
                    Ok((JitOutcome::Trapped(t), snap)) => {
                        (Err(format!("{t:?}")), snap.bytes().to_vec())
                    }
                    Ok((other, _)) => panic!("unexpected outcome {other:?}"),
                    Err(e) => panic!("the JIT refused the module: {e:?}"),
                }
            }
            Engine::Bytecode => {
                let mut fuel = 10_000_000u64;
                let (r, snap) = temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
                    &inst, 0, &args, &mut fuel, &win, SIZE_LOG2, &mut h,
                )
                .expect("the bytecode engine runs the module");
                let r = match r {
                    Ok(v) => match v[..] {
                        [Value::I64(n)] => Ok(n),
                        ref other => panic!("unexpected result {other:?}"),
                    },
                    Err(t) => Err(format!("{t:?}")),
                };
                (r, snap)
            }
        };
        let _ = tx.send((r, snap, h));
    });
    rx.recv_timeout(Duration::from_secs(20))
        .expect("the run completes")
}

/// Whether a row's uninterrupted run can complete by itself, so the matrix checks `want` against it.
/// A row whose park is ended only from outside the domain (input arriving, a `SIGCONT`) cannot.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Uninterrupted {
    Runs,
    WaitsOnTheOutside,
}

/// The shared shape of a `Reissue` or `Phase` row. The root parks at `site`; `arm` sets the freeze
/// trigger, which fires elsewhere while it is parked. The frozen run must be the root's unwind with
/// the cut non-empty; `release` then does whatever the uninterrupted run would have seen happen
/// outside the domain, and the thaw must give `want`, the uninterrupted answer (checked against an
/// uninterrupted run when it `Runs`). Returns the host after the thaw.
#[allow(clippy::too_many_arguments)]
fn freeze_parked_then_thaw(
    engine: Engine,
    site: ParkSite,
    src: &str,
    host: impl Fn(&mut Host) -> Vec<Value>,
    uninterrupted: Uninterrupted,
    arm: fn(&mut [u8]),
    release: impl FnOnce(&mut Host),
    want: Answer,
) -> Host {
    assert_ne!(site.freeze_rule(), FreezeRule::Decline, "{site:?}");
    let inst = instrumented(src);
    if uninterrupted == Uninterrupted::Runs {
        let mut h = durable_host(&inst);
        let args = host(&mut h);
        let (res, _, _) = run(
            engine,
            &inst,
            &args,
            &init_durable_window(WINDOW, TEST_ARENA),
            h,
        );
        assert_eq!(res, want, "{site:?}: the uninterrupted answer");
    }
    let mut h = durable_host(&inst);
    let args = host(&mut h);
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm(&mut win);
    let (res, snap, mut h) = run(engine, &inst, &args, &win, h);
    assert_eq!(res, Ok(0), "{site:?}: the root unwinds for the freeze");
    assert_eq!(h.take_freeze_declined(), None, "{site:?}: not declined");

    release(&mut h);
    let fibers = h.frozen_fibers().to_vec();
    h.set_frozen_fibers(fibers);
    let mut win2 = snap;
    // The arm rides the window: a quiesce-frozen run that goes idle again would freeze again. The
    // thaw runs disarmed, to give the uninterrupted answer.
    win2[ARM_QUIESCE_OFF as usize] = 0;
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let (res2, _, h) = run(engine, &inst, &args, &win2, h);
    assert_eq!(
        res2, want,
        "{site:?}: the thaw gives the uninterrupted answer"
    );
    h
}

/// The sibling every shared-shape row spawns: it resumes a fiber (the fiber-safepoint countdown fires
/// the freeze inside it, with the root already parked), then does the row's release with its argument,
/// and returns 7. The fiber-safepoint countdown is the trigger the oracle and the JIT share, so the same
/// row runs on both. The fiber is [`FIBER`], which each such module defines as its func 2.
const SIBLING_LOOP: &str = r#"
block 0 (vsp: i64, varg: i64) {
  vf = ref.func 2
  vfsp = i64.const 4096
  vk = cont.new vf vfsp
  vi0 = i64.const 0
  br 1(varg, vk, vi0)
}
block 1 (va: i64, vk1: i64, vi: i64) {
  vs, vx = cont.resume vk1 vi
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 20
  vmore = i64.ne vi2 vlim
  br_if vmore 1(va, vk1, vi2) 2(va)
}
"#;

/// The sibling's fiber: it suspends forever.
const FIBER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  br 1(v1)
}
block 1 (va: i64) {
  vn = suspend va
  br 1(vn)
  }
}
"#;

fn arm_in_sibling(win: &mut [u8]) {
    arm_freeze_after(win, 5);
}

/// `atomic.wait` — the root waits on a word only its sibling sets. However the re-issued wait and
/// the sibling's store interleave on thaw, the root reads the stored word: `1000·1 + 7`.
fn futex(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vj = thread.join vt
  vw = i32.load vaddr
  vw64 = i64.extend_i32_u vw
  vk = i64.const 1000
  vwk = i64.mul vw64 vk
  vr = i64.add vwk vj
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
{SIBLING_LOOP}
block 2 (va2: i64) {{
  vaddr = i64.const 66000
  vone = i32.const 1
  i32.atomic.store vaddr vone
  vn = atomic.notify vaddr vone
  vr = i64.const 7
  return vr
  }}
}}
{FIBER}"#
    );
    freeze_parked_then_thaw(
        engine,
        site,
        &src,
        |_| vec![],
        Uninterrupted::Runs,
        arm_in_sibling,
        |_| {},
        Ok(1007),
    );
}

/// `atomic.wait` on one vCPU — the root waits, indefinitely, on a word nothing can store to: the
/// uninterrupted run is a deadlock (`ThreadFault`). Freeze-on-quiesce fires on the wait; the thaw
/// re-issues it (#1769), so it deadlocks exactly as the uninterrupted run, never reloading the freeze's
/// `WAIT_FROZEN` as the wait's answer.
fn futex_alone(site: ParkSite, engine: Engine) {
    let src = r#"
memory 17
func () -> (i64) {
block 0 () {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;
    freeze_parked_then_thaw(
        engine,
        site,
        src,
        |_| vec![],
        Uninterrupted::Runs,
        arm_freeze_on_quiesce,
        |_| {},
        Err("ThreadFault".into()),
    );
}

/// `thread.join` — the root joins a sibling that is still looping: `1000 + 7`.
fn join(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vj = thread.join vt
  vk = i64.const 1000
  vr = i64.add vk vj
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
{SIBLING_LOOP}
block 2 (va2: i64) {{
  vr = i64.const 7
  return vr
  }}
}}
{FIBER}"#
    );
    freeze_parked_then_thaw(
        engine,
        site,
        &src,
        |_| vec![],
        Uninterrupted::Runs,
        arm_in_sibling,
        |_| {},
        Ok(1007),
    );
}

/// A pipe read — the root reads one byte from an empty pipe whose write end the sibling holds and
/// writes `x` to after its loop: `1000·1 + 'x' + 7`.
fn pipe_read(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func (i32, i32) -> (i64) {{
block 0 (vr: i32, vw: i32) {{
  vz = i64.const 0
  vw64 = i64.extend_i32_u vw
  vt = thread.spawn 1 vz vw64
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vr (vbuf, vlen)
  vj = thread.join vt
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vs = i64.add vnk vb64
  vres = i64.add vs vj
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
{SIBLING_LOOP}
block 2 (va2: i64) {{
  vw = i32.wrap_i64 va2
  vbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vbuf vx
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vlen)
  vr = i64.const 7
  return vr
  }}
}}
{FIBER}"#
    );
    freeze_parked_then_thaw(
        engine,
        site,
        &src,
        |h| {
            let (w, r) = h.grant_pipe();
            vec![Value::I32(r), Value::I32(w)]
        },
        Uninterrupted::Runs,
        arm_in_sibling,
        |_| {},
        Ok(1000 + i64::from(b'x') + 7),
    );
}

/// A pipe write — the root fills the pipe (its capacity is the guest's whole 64 KiB) and then
/// writes one more byte, which parks until the sibling drains it after its loop: `1000·1 + 7`, the
/// second write's count.
fn pipe_write(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func (i32, i32) -> (i64) {{
block 0 (vr: i32, vw: i32) {{
  vz = i64.const 0
  vr64 = i64.extend_i32_u vr
  vt = thread.spawn 1 vz vr64
  vbuf = i64.const 65536
  vcap = i64.const 65536
  vfill = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vcap)
  vone = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vw (vbuf, vone)
  vj = thread.join vt
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vres = i64.add vnk vj
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
{SIBLING_LOOP}
block 2 (va2: i64) {{
  vrd = i32.wrap_i64 va2
  vbuf = i64.const 65536
  vcap = i64.const 65536
  vn = call.cap 0 0 (i64, i64) -> (i64) vrd (vbuf, vcap)
  vr = i64.const 7
  return vr
  }}
}}
{FIBER}"#
    );
    freeze_parked_then_thaw(
        engine,
        site,
        &src,
        |h| {
            let (w, r) = h.grant_pipe();
            vec![Value::I32(r), Value::I32(w)]
        },
        Uninterrupted::Runs,
        arm_in_sibling,
        |_| {},
        Ok(1007),
    );
}

/// A stream read — the root parks a fiber on a futex nothing will notify, then reads one byte from a
/// blocking stdin with nothing waiting. (One vCPU, so the row runs on an engine whose durable entry
/// refuses `thread.*`.) Freeze-on-quiesce fires with both parked; the input arrives only after the
/// freeze: `1000·1 + 'x'`.
fn stream_read(site: ParkSite, engine: Engine) {
    let src = r#"
memory 17
func (i32) -> (i64) {
block 0 (vin: i32) {
  vf = ref.func 1
  vsp = i64.const 4096
  vfk = cont.new vf vsp
  vz = i64.const 0
  vs, vx = cont.resume vfk vz
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vlen)
  vk = i64.const 1000
  vnk = i64.mul vn vk
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vres = i64.add vnk vb64
  return vres
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;
    freeze_parked_then_thaw(
        engine,
        site,
        src,
        |h| {
            let vin = h.grant_stream(StreamRole::In);
            h.set_stdin_blocking(true);
            vec![Value::I32(vin)]
        },
        Uninterrupted::WaitsOnTheOutside,
        arm_freeze_on_quiesce,
        |h| h.push_stdin(b"x"),
        Ok(1000 + i64::from(b'x')),
    );
}

/// A job-control stop with no personality behind it; [`Stopper::set`] drives it. The oracle applies it
/// through the door it installs ([`SignalSource::set_stop_apply`]); the bytecode engine reads
/// [`SignalSource::stopped`].
#[derive(Default)]
struct Stopper {
    apply: Mutex<Option<StopApply>>,
    stopped: std::sync::atomic::AtomicBool,
}

type StopApply = Arc<dyn Fn(bool) + Send + Sync>;

impl Stopper {
    fn set(&self, stopped: bool) {
        self.stopped
            .store(stopped, std::sync::atomic::Ordering::SeqCst);
        if let Some(apply) = self.apply.lock().unwrap().clone() {
            apply(stopped);
        }
    }
}

impl SignalSource for Stopper {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_stop_apply(&self, apply: StopApply) {
        self.apply.lock().unwrap().get_or_insert(apply);
    }
    fn stopped(&self) -> bool {
        self.stopped.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// A job-control stop — the root parks a fiber on a futex nothing will notify, stops its own domain
/// and parks stopped. (A fiber, not a thread: the stop would halt a sibling before it reached its
/// wait.) Freeze-on-quiesce fires on the fiber's park, and the stopped root is brought through it:
/// its write is abandoned, not performed. Continued and thawed, it re-issues the write (`1000 + 1`),
/// and the byte reaches stdout only then.
fn stopped(site: ParkSite, engine: Engine) {
    let src = r#"
memory 17
func (i32, i32) -> (i64) {
block 0 (vstop: i32, vout: i32) {
  vf = ref.func 1
  vsp = i64.const 4096
  vfk = cont.new vf vsp
  vz = i64.const 0
  vs, vx = cont.resume vfk vz
  vq = call.cap 13 0 (i64) -> (i64) vstop (vz)
  vbuf = i64.const 66100
  vc = i32.const 120
  i32.store8 vbuf vc
  vlen = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vout (vbuf, vlen)
  vk = i64.const 1000
  vr = i64.add vn vk
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 66000
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
  vst64 = i64.extend_i32_u vst
  return vst64
  }
}
"#;
    let stopper = Arc::new(Stopper::default());
    let h = freeze_parked_then_thaw(
        engine,
        site,
        src,
        |h| {
            h.set_signal_source(stopper.clone(), Default::default());
            let s = stopper.clone();
            let stop = h.grant_host_proc(
                Box::new(move |_op, _args, _mem, _minter| {
                    s.set(true);
                    Ok(vec![0])
                }),
                CapState::Stateless,
            );
            let out = h.grant_stream(StreamRole::Out);
            vec![Value::I32(stop), Value::I32(out)]
        },
        Uninterrupted::WaitsOnTheOutside,
        arm_freeze_on_quiesce,
        |h| {
            assert!(
                h.stdout.is_empty(),
                "{site:?}: nothing left the domain while stopped"
            );
            stopper.set(false);
        },
        Ok(1001),
    );
    assert_eq!(h.stdout, b"x", "{site:?}: the thaw re-issued the write");
}

/// `svc.wait` — the root is a server idle in its accept loop: its queue is empty, so it parks, and
/// freeze-on-quiesce fires on that park (no countdown can reach it). A dispatch arrives only after the
/// freeze; the thawed server re-issues its `svc.wait`, drains it (the handler stores 41 and replies
/// 42) and returns `served·1000 + stored`: `1041`.
fn svc(site: ParkSite, engine: Engine) {
    let src = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  vc = i64.const 65600
  vafter = i64.load vc
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 65600
  i64.store vc vx
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}
"#;
    let ticket = Arc::new(Mutex::new(None));
    let mut h = freeze_parked_then_thaw(
        engine,
        site,
        src,
        |_| vec![],
        Uninterrupted::WaitsOnTheOutside,
        arm_freeze_on_quiesce,
        |h| {
            let t = h.svc_enqueue(0, 0, vec![41]).expect("enqueue");
            *ticket.lock().unwrap() = Some(t);
        },
        Ok(1041),
    );
    let t = ticket.lock().unwrap().expect("enqueued");
    assert_eq!(h.svc_result(t), Some(42), "{site:?}: the handler's reply");
}

/// A live call's reply — thread `T` calls `bump(41)` on a detached child `C` that serves one dispatch
/// (op 15, then `child_offer`), while the root resumes a fiber, where the countdown fires. The root
/// unwinds and rings `C`, which unwinds in its `svc.wait` without serving; `T` then enqueues its call
/// and parks on the reply, and the freeze re-admits it: its wait is abandoned, its dispatch stays
/// queued on `C`, and the ticket rides the root's powerbox. Through the codec, the thaw's re-issued
/// call waits on that ticket and `C` serves the queued dispatch once: `served·1000 + reply`, `1042`.
/// Issued twice, `C` would serve two.
fn reply(site: ParkSite, engine: Engine) {
    reply_from(site, engine, T_CALLS);
}

/// `T` of [`reply`]: it makes the call itself.
const T_CALLS: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vcap: i64) {
  vh = i32.wrap_i64 vcap
  vx = i64.const 41
  vr = call.cap 268435456 0 (i64) -> (i64) vh (vx)
  return vr
  }
}
"#;

/// `T` of [`reply`] with its call in a fiber (#1677): it drives the fiber with `cont.resume.block`,
/// so it idles while the fiber waits on the reply. Its first resume comes before any poll, as the
/// call does in [`T_CALLS`]. Under the freeze `T` unwinds past the parked fiber, and the freeze drive
/// abandons the fiber's wait and records the fiber's ticket.
const T_CALLS_IN_A_FIBER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vcap: i64) {
  vf = ref.func 5
  vfsp = i64.const 8192
  vk = cont.new vf vfsp
  vs, vv = cont.resume.block vk vcap
  br 1(vk, vcap, vs, vv)
}
block 1 (vk1: i64, vc: i64, vs1: i32, vv1: i64) {
  vone = i32.const 1
  vdone = i32.eq vs1 vone
  br_if vdone 3(vv1) 2(vk1, vc)
}
block 2 (vk2: i64, vc2: i64) {
  vs2, vv2 = cont.resume.block vk2 vc2
  br 1(vk2, vc2, vs2, vv2)
}
block 3 (vr: i64) {
  return vr
  }
}
"#;

/// The fiber [`T_CALLS_IN_A_FIBER`] drives, func 5: the call.
const CALLING_FIBER: &str = r#"
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vcap: i64) {
  vh = i32.wrap_i64 vcap
  vx = i64.const 41
  vr = call.cap 268435456 0 (i64) -> (i64) vh (vx)
  return vr
  }
}
"#;

fn reply_from(site: ParkSite, engine: Engine, t: &str) {
    let src = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 3 }
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vmh = i64.extend_i32_u v1
  vb = i64.extend_i32_u v2
  vz = i64.const 0
  ve = i64.const 2
  vlog = i64.const 17
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb, vmh, vz, vz, ve, vlog, vz)
  vex = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vc, vex)
  vcap64 = i64.extend_i32_u vcap
  vt = thread.spawn 1 vz vcap64
  vf = ref.func 4
  vfsp = i64.const 4096
  vk = cont.new vf vfsp
  br 1(v0, vc, vt, vk, vz)
}
block 1 (vi0: i32, vc1: i32, vt1: i32, vk1: i64, vi: i64) {
  vs, vx = cont.resume vk1 vi
  vone = i64.const 1
  vi2 = i64.add vi vone
  vlim = i64.const 20
  vmore = i64.ne vi2 vlim
  br_if vmore 1(vi0, vc1, vt1, vk1, vi2) 2(vi0, vc1, vt1)
}
block 2 (vi3: i32, vc2: i32, vt2: i32) {
  vr = thread.join vt2
  vn = call.cap 6 1 (i32) -> (i64) vi3 (vc2)
  vth = i64.const 1000
  vm = i64.mul vn vth
  vres = i64.add vm vr
  return vres
  }
}
"#;
    let server = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vone = i64.const 1
  vr = i64.add vx vone
  return vr
  }
}
"#;
    let src = format!("{src}{t}{server}{FIBER}{CALLING_FIBER}");
    let inst = instrumented(&src);
    let powerbox = || {
        let mut h = durable_host(&inst);
        let i = h.grant_instantiator(0, WINDOW as u64);
        let m = h.grant_durable_module(&inst);
        let b = h.grant_budget(0, 1 << 20, 0);
        h.grant_freeze_authority(FreezeScope::DetachedProgeny);
        (h, vec![Value::I32(i), Value::I32(m), Value::I32(b)])
    };

    let (h, args) = powerbox();
    let (res, _, _) = run(
        engine,
        &inst,
        &args,
        &init_durable_window(WINDOW, TEST_ARENA),
        h,
    );
    assert_eq!(res, Ok(1042), "{site:?}: the uninterrupted answer");

    let (h, args) = powerbox();
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    arm_in_sibling(&mut win);
    let (res, snap, mut h) = run(engine, &inst, &args, &win, h);
    assert_eq!(res, Ok(0), "{site:?}: the root unwinds for the freeze");
    assert_eq!(h.take_freeze_declined(), None, "{site:?}: not declined");
    assert_eq!(h.captured_detached().len(), 1, "{site:?}: C rides the cut");
    assert_eq!(
        h.reply_waits().len(),
        1,
        "{site:?}: the caller's wait was abandoned, and its ticket rides"
    );

    let art = temen_snapshot::freeze(&inst, &snap, &h).expect("serialize");
    let mut th = durable_host(&inst);
    th.grant_durable_module(&inst);
    let mut twin = temen_snapshot::restore(&art, &inst, &mut th).expect("restore");
    begin_thaw(&mut twin, TEST_ARENA, 0);
    let (res, _, th) = run(engine, &inst, &args, &twin, th);
    assert_eq!(
        res,
        Ok(1042),
        "{site:?}: the thaw waits on the same ticket, and C serves it once"
    );
    assert!(th.reply_waits().is_empty(), "{site:?}: the wait was taken");
}

/// #1677 — a **fiber** parked at a re-issue site freezes as a vCPU parked there does: the freeze
/// drive abandons its park and the thaw re-issues it. A fiber parks on a futex (the `stopped` row's
/// fiber), a pipe op or a stream read (#1952, #1973, on every engine), or a live call's reply.
#[test]
fn a_fiber_parked_at_a_reissue_site_freezes_like_a_vcpu() {
    for engine in [Engine::Interp, Engine::Jit] {
        pipe_read_in_a_fiber(engine);
        pipe_write_in_a_fiber(engine);
        stdin_read_in_a_fiber(engine, Trigger::Countdown);
    }
    // The bytecode engine keeps no countdown, and under freeze-on-quiesce only input from outside the
    // domain can end the park: stdin, but not a pipe the domain minted (a pipe fed from outside crosses
    // the cut, #1680). Its pipe and stdin parks share one flatten arm (`HostParked`).
    for engine in [Engine::Interp, Engine::Bytecode] {
        stdin_read_in_a_fiber(engine, Trigger::Quiesce);
    }
    stream_read_in_a_fiber(ParkSite::StreamRead, Engine::Interp);
    reply_from(ParkSite::Reply, Engine::Interp, T_CALLS_IN_A_FIBER);
}

/// What fires a [`host_park_in_a_fiber`] row's freeze. The oracle keeps both triggers; the JIT keeps
/// only the countdown, and the bytecode engine only freeze-on-quiesce.
#[derive(Clone, Copy, Debug)]
enum Trigger {
    /// [`arm_in_sibling`]'s countdown, over the safepoints of a ticker fiber the root polls.
    Countdown,
    /// [`arm_freeze_on_quiesce`], when the root blocks on the parked fiber.
    Quiesce,
}

/// The one-vCPU shape of a fiber's host park, so the row runs on an engine whose durable entry refuses
/// `thread.*`. The root starts fiber `op` (func 1) with its first argument, and `op` parks on the host
/// op. The root then drives `op` to its return `n` and answers `1000·n + byte 66100`:
///
/// - [`Trigger::Countdown`]: it polls `op` twenty times beside the ticker [`FIBER`] (func 2), whose
///   safepoints fire the freeze with `op` parked, then does `release` with its second argument, which
///   ends the park from inside the domain. It never resumes `op` once it has returned, however early
///   the thaw's re-issued op completes.
/// - [`Trigger::Quiesce`]: it blocks on `op` at once (`cont.resume.block`), so the domain goes idle with
///   `op` parked; only `outside` can release it, and `release` must be empty.
///
/// `outside` is what arrives from outside the domain after the freeze.
#[allow(clippy::too_many_arguments)]
fn host_park_in_a_fiber(
    site: ParkSite,
    engine: Engine,
    trigger: Trigger,
    op: &str,
    release: &str,
    host: impl Fn(&mut Host) -> Vec<Value>,
    uninterrupted: Uninterrupted,
    outside: impl FnOnce(&mut Host),
    want: Answer,
) {
    let (polls, arm): (i64, fn(&mut [u8])) = match trigger {
        Trigger::Countdown => (20, arm_in_sibling),
        Trigger::Quiesce => (1, arm_freeze_on_quiesce),
    };
    let src = format!(
        r#"
memory 17
func (i32, i32) -> (i64) {{
block 0 (va: i32, vb: i32) {{
  vf = ref.func 1
  vfsp = i64.const 8192
  vk = cont.new vf vfsp
  vt = ref.func 2
  vtsp = i64.const 4096
  vtk = cont.new vt vtsp
  va64 = i64.extend_i32_u va
  vi0 = i64.const 0
  br 1(vk, vtk, va64, vb, vi0)
}}
block 1 (vk1: i64, vtk1: i64, va1: i64, vb1: i32, vi: i64) {{
  vs, vn = cont.resume vk1 va1
  vdone = i32.const 1
  vfin = i32.eq vs vdone
  br_if vfin 5(vn) 2(vk1, vtk1, va1, vb1, vi)
}}
block 2 (vk2: i64, vtk2: i64, va2: i64, vb2: i32, vi2: i64) {{
  vts, vtx = cont.resume vtk2 vi2
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  vlim = i64.const {polls}
  vmore = i64.ne vi3 vlim
  br_if vmore 1(vk2, vtk2, va2, vb2, vi3) 3(vk2, va2, vb2)
}}
block 3 (vk3: i64, va3: i64, vb3: i32) {{
{release}
  br 4(vk3, va3)
}}
block 4 (vk4: i64, va4: i64) {{
  vs4, vn4 = cont.resume.block vk4 va4
  vd4 = i32.const 1
  vfin4 = i32.eq vs4 vd4
  br_if vfin4 5(vn4) 4(vk4, va4)
}}
block 5 (vn5: i64) {{
  vth = i64.const 1000
  vnk = i64.mul vn5 vth
  vbuf = i64.const 66100
  vbyte = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vbyte
  vres = i64.add vnk vb64
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vh = i32.wrap_i64 varg
{op}
  }}
}}
{FIBER}"#
    );
    freeze_parked_then_thaw(engine, site, &src, host, uninterrupted, arm, outside, want);
}

/// A pipe grant: the read end, then the write end.
fn read_then_write(h: &mut Host) -> Vec<Value> {
    let (w, r) = h.grant_pipe();
    vec![Value::I32(r), Value::I32(w)]
}

/// A pipe grant: the write end, then the read end.
fn write_then_read(h: &mut Host) -> Vec<Value> {
    let (w, r) = h.grant_pipe();
    vec![Value::I32(w), Value::I32(r)]
}

/// A pipe read in a fiber: it reads one byte from an empty pipe, and the root writes `x` after its
/// polls: `1000·1 + 'x'`.
fn pipe_read_in_a_fiber(engine: Engine) {
    host_park_in_a_fiber(
        ParkSite::PipeRead,
        engine,
        Trigger::Countdown,
        r#"  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  return vn"#,
        r#"  vwbuf = i64.const 66200
  vx = i32.const 120
  i32.store8 vwbuf vx
  vwlen = i64.const 1
  vw = call.cap 0 1 (i64, i64) -> (i64) vb3 (vwbuf, vwlen)"#,
        read_then_write,
        Uninterrupted::Runs,
        |_| {},
        Ok(1000 + i64::from(b'x')),
    );
}

/// A pipe write in a fiber: it fills the pipe (its capacity is the guest's whole 64 KiB) and writes
/// one more byte, which parks until the root drains the pipe after its polls: `1000·1 + 0`, the second
/// write's count (the drained bytes are the zeroed buffer, so byte 66100 stays 0).
fn pipe_write_in_a_fiber(engine: Engine) {
    host_park_in_a_fiber(
        ParkSite::PipeWrite,
        engine,
        Trigger::Countdown,
        r#"  vbuf = i64.const 65536
  vcap = i64.const 65536
  vfill = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vcap)
  vone = i64.const 1
  vn = call.cap 0 1 (i64, i64) -> (i64) vh (vbuf, vone)
  return vn"#,
        r#"  vrbuf = i64.const 65536
  vrcap = i64.const 65536
  vr = call.cap 0 0 (i64, i64) -> (i64) vb3 (vrbuf, vrcap)"#,
        write_then_read,
        Uninterrupted::Runs,
        |_| {},
        Ok(1000),
    );
}

/// A stdin read in a fiber: it reads one byte from a blocking stdin with nothing waiting, which only
/// input from outside the domain ends; it arrives after the freeze: `1000·1 + 'x'`.
fn stdin_read_in_a_fiber(engine: Engine, trigger: Trigger) {
    host_park_in_a_fiber(
        ParkSite::StreamRead,
        engine,
        trigger,
        r#"  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vlen)
  return vn"#,
        "",
        |h| {
            let vin = h.grant_stream(StreamRole::In);
            h.set_stdin_blocking(true);
            vec![Value::I32(vin), Value::I32(0)]
        },
        Uninterrupted::WaitsOnTheOutside,
        |h| h.push_stdin(b"x"),
        Ok(1000 + i64::from(b'x')),
    );
}

/// A stream read in a fiber — the root drives a fiber with `cont.resume.block` that reads one byte
/// from a blocking stdin with nothing waiting, while its sibling loops. The input arrives only after
/// the freeze: `1000·1 + 'x' + 7`.
fn stream_read_in_a_fiber(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func (i32) -> (i64) {{
block 0 (vin: i32) {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vf = ref.func 3
  vfsp = i64.const 8192
  vk = cont.new vf vfsp
  vin64 = i64.extend_i32_u vin
  br 1(vt, vk, vin64)
}}
block 1 (vt1: i32, vk1: i64, va: i64) {{
  vs, vn = cont.resume.block vk1 va
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vt1, vn) 1(vt1, vk1, va)
}}
block 2 (vt2: i32, vn2: i64) {{
  vj = thread.join vt2
  vk = i64.const 1000
  vnk = i64.mul vn2 vk
  vbuf = i64.const 66100
  vb = i32.load8_u vbuf
  vb64 = i64.extend_i32_u vb
  vs = i64.add vnk vb64
  vres = i64.add vs vj
  return vres
  }}
}}
func (i64, i64) -> (i64) {{
{SIBLING_LOOP}
block 2 (va2: i64) {{
  vr = i64.const 7
  return vr
  }}
}}
{FIBER}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, va: i64) {{
  vin = i32.wrap_i64 va
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vlen)
  return vn
  }}
}}"#
    );
    freeze_parked_then_thaw(
        engine,
        site,
        &src,
        |h| {
            let vin = h.grant_stream(StreamRole::In);
            h.set_stdin_blocking(true);
            vec![Value::I32(vin)]
        },
        Uninterrupted::WaitsOnTheOutside,
        arm_in_sibling,
        |h| h.push_stdin(b"x"),
        Ok(1000 + i64::from(b'x') + 7),
    );
}
