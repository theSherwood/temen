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
//! The oracle and the JIT each have a runner ([`run`]). The bytecode rows wait on #1904, which gives
//! that engine the same park rules; enabling it is adding its runner and flipping its rows. Every
//! engine keeps the fiber-safepoint countdown ([`arm_freeze_after`]), so the rows that fire the
//! freeze in a sibling use it and run unchanged on each.

use std::sync::{Arc, Mutex};
use std::time::Duration;
use temen_durable::{
    arm_freeze_after, arm_freeze_on_quiesce, begin_thaw, init_durable_window,
    transform_module_assume_confined,
};
use temen_interp::{
    run_capture_reserved_with_host, CapState, FreezeRule, Host, ParkSite, SignalSource, StreamRole,
    Value,
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
        Engine::Bytecode => Row::Pending {
            issue: 1904,
            why: "the same park rules on this engine, and its runner here",
        },
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
            why: "a blocking `waitpid` waits on a fork twin, which declines the freeze first (#1688); \
                  `posix_spawn` runs its child through a host delegate and never parks",
        },
        ParkSite::Lane => Row::Unreachable {
            why: "a run that can freeze is serialized onto one worker, and a task gives its lane back \
                  when it parks, so no lane is ever full",
        },
        ParkSite::Reply | ParkSite::Completion => Row::Pending {
            issue: 1901,
            why: "a call in flight inside the cut; built with its re-park",
        },
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
        ParkSite::PipeRead | ParkSite::PipeWrite | ParkSite::StreamRead => Row::Pending {
            issue: 1826,
            why: "the JIT serves no host-call park outside a process tree: the op's placeholder \
                  answer stands: an empty read reads as EOF, a full write writes nothing",
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
        ParkSite::Reply | ParkSite::Completion => Row::Pending {
            issue: 1901,
            why: "a call in flight inside the cut (the JIT's `live_impl_call` reply and its \
                  punted-call completion); built with its re-park",
        },
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
            Engine::Bytecode => unreachable!("no bytecode runner yet"),
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
    want: i64,
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
        assert_eq!(res, Ok(want), "{site:?}: the uninterrupted answer");
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
    begin_thaw(&mut win2, TEST_ARENA, 0);
    let (res2, _, h) = run(engine, &inst, &args, &win2, h);
    assert_eq!(
        res2,
        Ok(want),
        "{site:?}: the thaw gives the uninterrupted answer"
    );
    h
}

/// The sibling every shared-shape row spawns: it resumes a fiber (the fiber-safepoint countdown fires
/// the freeze inside it, with the root already parked), then does the row's release with its argument,
/// and returns 7. The fiber-safepoint countdown is the trigger every engine keeps, so the same row runs
/// on each. The fiber is [`FIBER`], which each such module defines as its func 2.
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
        1007,
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
        1007,
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
        1000 + i64::from(b'x') + 7,
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
        1007,
    );
}

/// A stream read — the root reads one byte from a blocking stdin with nothing waiting, while its
/// sibling loops. The input arrives only after the freeze: `1000·1 + 'x' + 7`.
fn stream_read(site: ParkSite, engine: Engine) {
    let src = format!(
        r#"
memory 17
func (i32) -> (i64) {{
block 0 (vin: i32) {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vbuf = i64.const 66100
  vlen = i64.const 1
  vn = call.cap 0 0 (i64, i64) -> (i64) vin (vbuf, vlen)
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
            let vin = h.grant_stream(StreamRole::In);
            h.set_stdin_blocking(true);
            vec![Value::I32(vin)]
        },
        Uninterrupted::WaitsOnTheOutside,
        arm_in_sibling,
        |h| h.push_stdin(b"x"),
        1000 + i64::from(b'x') + 7,
    );
}

/// A job-control stop with no personality behind it; [`Stopper::stop`] and [`Stopper::cont`] drive it.
#[derive(Default)]
struct Stopper {
    apply: Mutex<Option<StopApply>>,
}

type StopApply = Arc<dyn Fn(bool) + Send + Sync>;

impl Stopper {
    fn set(&self, stopped: bool) {
        let apply = self.apply.lock().unwrap().clone();
        apply.expect("installed by the run")(stopped);
    }
}

impl SignalSource for Stopper {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_stop_apply(&self, apply: StopApply) {
        self.apply.lock().unwrap().get_or_insert(apply);
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
        1001,
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
        1041,
    );
    let t = ticket.lock().unwrap().expect("enqueued");
    assert_eq!(h.svc_result(t), Some(42), "{site:?}: the handler's reply");
}
