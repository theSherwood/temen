//! **The runners both parity matrices run on** (#2545, D3a): one way to run a guest in each
//! configuration of `run_modes::RUNNERS`, used by the run-mode matrix's feature probes
//! (`run_mode_conformance.rs`) and by the frontier matrix's capability probes
//! (`axis_conformance.rs`). Two harnesses with their own drivers could disagree about what a
//! configuration does with the same program; one cannot (INVARIANTS #15).
//!
//! Included with `#[path = "support/runners.rs"] mod runners;`, beside the `drivers`, `probes` and
//! `rec` modules it builds on.

// Each including binary uses a subset.
#![allow(dead_code)]

use std::sync::{mpsc, Arc};
use std::time::Duration;

use crate::drivers::{run_on, Driver, Pumped};
use crate::probes::{self, Embedder, Outside, Stopper, ARENA};
use temen_durable::{begin_thaw, init_durable_window, read_state, write_state};
use temen_interp::bytecode::{
    module_spawns_threads, CoopEvent, CoopRun, Reactor, SchedStop, ScheduledDebugRun,
};
use temen_interp::{Host, Inspector, MemLayout, Stop, Trap, Value};
use temen_ir::durable_abi::STATE_UNWINDING;
use temen_ir::Module;
use temen_parity::run_modes::{Config, Executor, Property, Props};
use temen_snapshot::FreezeError;

/// The fuel every run gets.
pub const FUEL: u64 = 50_000_000;

/// How long one run may take before the harness calls it hung. A harness must fail, never hang.
pub const BOUND: Duration = Duration::from_secs(30);

/// How long a runner waits before resuming a run idle on an answer the probe gives itself.
const ANSWER_POLL: Duration = Duration::from_millis(1);

/// How one run ended, whatever engine ran it. A trap is spelled as the oracle spells it, so the
/// same trap compares equal across engines.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum End {
    /// The entry returned this.
    Value(i64),
    /// The run trapped, as the oracle names the trap.
    Trap(String),
    /// The runner declined the module.
    Declined,
    /// Anything else: an answer of another shape, a stop the runner could not move on from.
    Stuck(String),
}

impl End {
    pub fn of(r: Result<Vec<Value>, Trap>) -> End {
        match r {
            Ok(v) => match v[..] {
                [Value::I64(x)] => End::Value(x),
                _ => End::Stuck(format!("returned {v:?}")),
            },
            Err(t) => End::Trap(format!("{t:?}")),
        }
    }
}

/// How the harness runs a configuration.
#[derive(Clone, Copy, Debug)]
pub enum Way {
    /// An executor, one-shot.
    OneShot(Executor),
    /// The tree-walk debugger (`Inspector`, the DAP's default engine).
    Inspector,
    /// The bytecode debug scheduler (`ScheduledDebugRun`, the DAP's `bytecode` engine).
    DebugScheduler,
    /// `CoopRun`, held and resumed by this harness.
    Held(Pumped),
    /// An executor's durable entry, run frozen and thawed ([`durable`]).
    Durable(Executor),
    /// The pump's reactor (`bytecode::Reactor`), its entry called as frames ([`reactor`]): each run
    /// to its end, or with `held`, each a `CoopRun` held and resumed by this harness.
    Reactor { held: bool },
}

impl Way {
    /// Whether nobody holds a run this way: it gets what its embedder gives it from outside from a
    /// thread of its own, rather than when it hands control back.
    pub fn unheld(self) -> bool {
        matches!(
            self,
            Way::OneShot(_) | Way::Inspector | Way::Durable(_) | Way::Reactor { held: false }
        )
    }
}

/// The harness's runner for `c`, if it has one. `RUNNERS` lists the same configurations, which
/// `run_mode_conformance.rs` holds.
pub fn way(c: Config) -> Option<Way> {
    use Executor::*;
    use Property::*;
    Some(if c.props.is_empty() {
        Way::OneShot(c.executor)
    } else if c == Config::new(TreeWalk, &[Debugger]) {
        Way::Inspector
    } else if c == Config::new(Pump, &[Debugger, Held]) {
        Way::DebugScheduler
    } else if c == Config::new(Pump, &[Held]) {
        Way::Held(Pumped::Pump)
    } else if c == Config::new(Parallel, &[Held]) {
        Way::Held(Pumped::Threads)
    } else if c.props == Props::of(&[Durable]) && c.executor != Parallel {
        Way::Durable(c.executor)
    } else if c == Config::new(Pump, &[Reactor]) {
        Way::Reactor { held: false }
    } else if c == Config::new(Pump, &[Reactor, Held]) {
        Way::Reactor { held: true }
    } else {
        return None;
    })
}

/// Run `f` on a thread of its own, for at most [`BOUND`].
pub fn bounded<T: Send + 'static>(what: String, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(BOUND)
        .unwrap_or_else(|_| panic!("{what}: no end within {BOUND:?}"))
}

/// How an embedder builds a run's powerbox and entry arguments, from the module the run runs.
pub type Setup<'a> = dyn Fn(&Module, &Embedder) -> (Host, Vec<Value>) + 'a;

/// A guest as a runner runs it: its module, the powerbox its embedder builds for each run, and what
/// the embedder does from outside the run.
pub struct Guest<'a> {
    pub module: &'a Module,
    pub setup: &'a Setup<'a>,
    pub outside: Outside,
}

/// Run `g` once on `way`, over a fresh embedder. `armed` arms the run's external-wake bell, saying
/// its embedder may act on it from outside, and has a thread continue the process each time it
/// stops. `want` is the oracle's answer, which a durable run must keep across its freeze; no other
/// way reads it.
pub fn once(way: Way, g: &Guest, armed: bool, want: Option<&End>) -> End {
    match way {
        Way::Durable(x) => {
            let want = want.expect("a durable run is held to the oracle's answer");
            return durable(x, g, armed, want);
        }
        Way::Reactor { held } => return reactor(g, held, armed),
        _ => {}
    }
    let m = g.module;
    let e = Embedder::default();
    let unheld = || unheld(m, g, &e, armed);
    let one_shot = |d| run_on(d, m, &unheld).map_or(End::Declined, |ran| End::of(ran.result));
    let end = match way {
        Way::OneShot(Executor::TreeWalk) => one_shot(Driver::Oracle),
        Way::OneShot(Executor::Pump) => one_shot(Driver::Coop),
        Way::OneShot(Executor::Parallel) => one_shot(Driver::Parallel),
        // Over a window seeded at its declared size in the default reservation, as
        // `temen_run::Instance` runs it.
        Way::OneShot(Executor::Cranelift) => {
            let window = MemLayout::image(vec![0u8; 1 << m.memory.map_or(16, |w| w.size_log2)]);
            on_cranelift(m, unheld(), &window, temen_ir::default_reserved_log2()).0
        }
        Way::Inspector => inspected(m, unheld()),
        Way::DebugScheduler => debugged(m, (g.setup)(m, &e), g.outside, &e.stopper),
        Way::Held(exec) => held(exec, m, (g.setup)(m, &e), g.outside, &e.stopper),
        Way::Durable(_) | Way::Reactor { .. } => unreachable!("driven above"),
    };
    // The run is over: a continue thread still waiting has nothing to continue.
    e.stopper.close();
    end
}

/// The powerbox `e` gives a run of `m` (the guest's module, or its instrumented twin) that nobody
/// holds, with what it gives the run from outside from a thread of its own: input fed from a
/// source, and, when `armed`, the run's external-wake bell armed and the process continued each
/// time it stops.
fn unheld(m: &Module, g: &Guest, e: &Embedder, armed: bool) -> (Host, Vec<Value>) {
    let (mut host, args) = (g.setup)(m, e);
    if e.durable {
        host.set_durable(true);
        // An embedder that freezes a run that spawns holds the authority to freeze its children
        // (#1440), without which a durable domain spawns none. Only such a run gets it: the grant
        // takes a handle, which would number the run's own grants unlike the oracle's.
        if temen_ir::spawns_detached(m) {
            host.grant_freeze_authority();
        }
    }
    if let Outside::Feed(bytes) = g.outside {
        feed_later(&mut host, bytes);
    }
    if armed {
        host.arm_external_wake();
        let s = Arc::clone(&e.stopper);
        std::thread::spawn(move || s.continue_each_stop());
    }
    (host, args)
}

/// Give `host` a stdin source that blocks until a thread supplies `bytes`, which it does only once
/// the read has asked: the read is always made before its input exists.
fn feed_later(host: &mut Host, bytes: &'static [u8]) {
    let (asked, on_ask) = mpsc::channel::<()>();
    let (supply, input) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        if on_ask.recv().is_ok() {
            let _ = supply.send(bytes.to_vec());
        }
    });
    host.set_stdin_source(Box::new(move || {
        let _ = asked.send(());
        // Once the input is spent, the thread is gone and the source says end of input.
        Some(input.recv().unwrap_or_default())
    }));
}

/// Run `m` on the Cranelift JIT through `temen_run`, from `window` in a reservation of
/// `2^reserved_log2` bytes, with the `call.dyn` table the powerbox's `Jit` grant reserved
/// (`temen_run::grant_jit` asks the compile for the same size): how it ended, the window it left,
/// and its host. The JIT refuses a module outside its subset with `Unsupported`; any other error is
/// an answer the oracle did not give. A trap kind is a trap wire code (#1735), so it is spelled as
/// the oracle's trap, and a guest's exit as the oracle's `Exit` trap.
fn on_cranelift(
    m: &Module,
    (mut host, args): (Host, Vec<Value>),
    window: &MemLayout,
    reserved_log2: u8,
) -> (End, Option<MemLayout>, Host) {
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => i64::from(*x),
            Value::I64(x) => *x,
            other => panic!("unexpected argument {other:?}"),
        })
        .collect();
    let table = host.jit_table_log2();
    let run = temen_run::jit_cap_run(m, 0, &args, window, reserved_log2, table, &mut host, None);
    let trap = |t: Trap| End::Trap(format!("{t:?}"));
    match run {
        Ok((temen_jit::JitOutcome::Returned(v), snap)) => match v[..] {
            [x] => (End::Value(x), Some(snap), host),
            _ => (End::Stuck(format!("returned {v:?}")), None, host),
        },
        Ok((temen_jit::JitOutcome::Trapped(k), snap)) => {
            let t = Trap::from_code(k.code()).expect("a trap kind is a trap");
            (trap(t), Some(snap), host)
        }
        Ok((temen_jit::JitOutcome::Exited(code), snap)) => {
            (trap(Trap::Exit(code)), Some(snap), host)
        }
        Ok((other, _)) => (End::Stuck(format!("{other:?}")), None, host),
        Err(temen_jit::JitError::Unsupported(_)) => (End::Declined, None, host),
        Err(e) => (End::Stuck(format!("{e:?}")), None, host),
    }
}

/// Run `m` under the tree-walk debugger to its end, as the best its API allows: scheduled, as the
/// DAP debugs a threaded program, when `m` spawns threads, else on one vCPU over the probe's
/// powerbox. The scheduled `Inspector` builds its own empty powerbox, so a threaded probe's handles
/// name nothing there.
fn inspected(m: &Module, (host, args): (Host, Vec<Value>)) -> End {
    let mut insp = if module_spawns_threads(m) {
        Inspector::attach_scheduled(m, 0, &args, FUEL, Vec::new())
    } else {
        Inspector::attach_with_host(m, 0, &args, FUEL, host)
    };
    loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => return End::of(r),
            Stop::Break { .. } => {}
            Stop::Blocked => return End::Stuck("Blocked".into()),
        }
    }
}

/// Run `m` under the debug scheduler to its end, as the DAP's bytecode engine does. A stop that
/// waits on the embedder is moved on: a stopped process is continued, a read the probe feeds gets
/// its bytes, and any other idle waits on a host-answered call the probe answers itself, so it is
/// looked at again shortly. The run says whether it waits on a read (`stdin_parked`), which only the
/// run changes; whether a call is still owed can change between the stop and the question.
fn debugged(
    m: &Module,
    (mut host, args): (Host, Vec<Value>),
    outside: Outside,
    stopper: &Stopper,
) -> End {
    let mut feed = match outside {
        Outside::Feed(bytes) => Some(bytes),
        Outside::Nothing | Outside::Continue => None,
    };
    host.set_stdin_blocking(feed.is_some());
    let Some(mut d) = ScheduledDebugRun::new_with_host(m, 0, &args, host) else {
        return End::Declined;
    };
    let mut fuel = FUEL;
    loop {
        match d.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => return End::of(r),
            SchedStop::Declined => return End::Declined,
            SchedStop::Break { .. } | SchedStop::CapPark { .. } => {}
            SchedStop::Idle { .. } if stopper.is_stopped() => stopper.set(false),
            SchedStop::Idle { .. } if !d.stdin_parked() => std::thread::sleep(ANSWER_POLL),
            SchedStop::Idle { .. } if feed.is_some() => {
                d.provide_stdin(feed.take().unwrap_or_default())
            }
            other => return End::Stuck(format!("{other:?}")),
        }
    }
}

/// Hold `m` as a `CoopRun` on `exec` and resume it to its end ([`pump`]).
fn held(
    exec: Pumped,
    m: &Module,
    (mut host, args): (Host, Vec<Value>),
    outside: Outside,
    stopper: &Stopper,
) -> End {
    host.set_stdin_blocking(matches!(outside, Outside::Feed(_)));
    let reserved = temen_ir::default_reserved_log2();
    let run = match exec {
        Pumped::Pump => CoopRun::new_reserved(m, 0, &args, FUEL, host, None, &[], reserved),
        Pumped::Threads => {
            CoopRun::new_parallel(m, 0, &args, FUEL, host, &[], reserved, None, true)
        }
    };
    match run {
        None => End::Declined,
        Some(Err(t)) => End::Stuck(format!("{t:?} at the start")),
        Some(Ok(mut run)) => pump(&mut run, outside, stopper),
    }
}

/// Resume a held `run` to its end, as its embedder. A run idle on a stopped process gets it
/// continued, and one idle on what the embedder feeds gets the probe's bytes once; idle otherwise,
/// it waits on an answer the probe gives itself, and is resumed shortly, the time it waited passed.
fn pump(run: &mut CoopRun, outside: Outside, stopper: &Stopper) -> End {
    let mut feed = match outside {
        Outside::Feed(bytes) => Some(bytes),
        Outside::Nothing | Outside::Continue => None,
    };
    loop {
        match run.run() {
            CoopEvent::Done(v) => return End::of(Ok(v)),
            CoopEvent::Trapped(t) => return End::of(Err(t)),
            CoopEvent::CapPark { .. } | CoopEvent::Paused => {}
            CoopEvent::Idle { .. } if stopper.is_stopped() => stopper.set(false),
            CoopEvent::Idle { .. } => match feed.take() {
                Some(bytes) => run.host_mut().push_stdin(bytes),
                None => {
                    std::thread::sleep(ANSWER_POLL);
                    // The run's timers see the time it waited, as an embedder's do (#2557).
                    run.pass_time(ANSWER_POLL.as_nanos() as u64);
                }
            },
            _ => return End::Stuck("a round-trip only the browser's tier makes".into()),
        }
    }
}

/// How many frames a reactor run calls its entry for: one, and one more over what it left.
const FRAMES: usize = 2;

/// Run `g` on the pump's reactor (`bytecode::Reactor`, the engine under the browser's reactors) as
/// [`FRAMES`] frames: its entry called again over the window and the powerbox the frame before it
/// left, each frame run to its end, or with `held`, each a `coop_call` this harness holds and
/// resumes ([`pump`]). Each frame is fed afresh. The frames are held to the oracle's
/// ([`oracle_frames`]): a first frame that does not give the oracle's answer is what the run gives,
/// a later one leaves the run stuck, and a frame that does not return is the reactor's last.
fn reactor(g: &Guest, held: bool, armed: bool) -> End {
    let want = oracle_frames(g, armed || held);
    let m = g.module;
    let Some(mut r) = Reactor::open(m) else {
        return End::Declined;
    };
    let e = Embedder::default();
    let (mut host, args) = if held {
        let (mut host, args) = (g.setup)(m, &e);
        host.set_stdin_blocking(matches!(g.outside, Outside::Feed(_)));
        (host, args)
    } else {
        unheld(m, g, &e, armed)
    };
    for (frame, want) in want.iter().enumerate() {
        let got = if held {
            match r.coop_call(0, &args, FUEL, std::mem::take(&mut host), None, None) {
                Err(t) => End::Stuck(format!("{t:?} at the start of frame {frame}")),
                Ok(mut run) => {
                    let got = pump(&mut run, g.outside, &e.stopper);
                    host = r.end_coop_call(&mut run);
                    got
                }
            }
        } else {
            if frame > 0 {
                feed_again(&mut host, g);
            }
            let mut fuel = FUEL;
            End::of(r.call(0, &args, &mut fuel, &mut host))
        };
        if got != *want {
            e.stopper.close();
            return match frame {
                0 => got,
                _ => End::Stuck(format!("frame {frame} gave {got:?}, the oracle's {want:?}")),
            };
        }
    }
    e.stopper.close();
    want[0].clone()
}

/// The oracle's frames for `g` as a reactor: the tree-walker's capture entry called again over the
/// window and the powerbox the frame before it left, each frame fed afresh. Fewer than [`FRAMES`]
/// when a frame does not return.
fn oracle_frames(g: &Guest, armed: bool) -> Vec<End> {
    let e = Embedder::default();
    let (mut host, args) = unheld(g.module, g, &e, armed);
    let mut window = MemLayout::image(Vec::new());
    let mut frames = Vec::new();
    for frame in 0..FRAMES {
        if frame > 0 {
            feed_again(&mut host, g);
        }
        let mut fuel = FUEL;
        let (r, left) = temen_interp::run_capture_reserved_with_host_layout(
            g.module,
            0,
            &args,
            &mut fuel,
            &window,
            temen_ir::default_reserved_log2(),
            &mut host,
        );
        let end = End::of(r);
        let returned = matches!(end, End::Value(_));
        frames.push(end);
        if !returned {
            break;
        }
        window = left.unwrap_or(window);
    }
    e.stopper.close();
    frames
}

/// What the embedder feeds a reactor's frame after its first: the probe's bytes again.
fn feed_again(host: &mut Host, g: &Guest) {
    if let Outside::Feed(bytes) = g.outside {
        feed_later(host, bytes);
    }
}

/// The reservation a durable run's window sits in: room for a probe that grows its window.
const DURABLE_RESERVED_LOG2: u8 = 18;

/// Run `g` durably on `x`, as DURABILITY.md §7's equivalence says: uninterrupted, then frozen at its
/// first safepoint (the poll after its first capability call), serialized by the §12 codec, restored
/// into a fresh powerbox and thawed. Both must give the oracle's answer, `want`; an answer that is
/// not a value has nothing to freeze across, so the uninterrupted run is the whole of it. The
/// freeze's state word is set before the run starts, the one deterministic trigger every engine
/// honours. The same embedder freezes and thaws the run: it gives the thaw the powerbox it gave the
/// run, whose module grants the restore re-pins handles to, and re-grants its named procedures by
/// name.
///
/// The uninterrupted run is continued if it stops itself, as its embedder continues it. The frozen
/// run is not: a freeze lands through a stop (#1672), so a process that stopped itself is frozen
/// stopped, and still is when it thaws. `armed` is then the thaw's, which the run-mode harness's
/// two-run rule holds: with nothing to continue it, the thaw must fault `ThreadFault`; continued, it
/// must finish.
fn durable(x: Executor, g: &Guest, armed: bool, want: &End) -> End {
    let inst = match probes::instrumented(g.module) {
        Ok(inst) => Arc::new(inst),
        Err(why) => return End::Stuck(why),
    };
    let size = 1usize << inst.memory.expect("instrumented declares it").size_log2;
    let fresh = init_durable_window(size, ARENA);

    let e = Embedder::durable();
    let seed = MemLayout::image(fresh.clone());
    let continued = g.outside == Outside::Continue;
    let whole = durable_run(x, &inst, unheld(&inst, g, &e, continued), &seed).0;
    e.stopper.close();
    if whole != *want || !matches!(want, End::Value(_)) {
        return whole;
    }

    let e = Embedder::durable();
    let mut cut = fresh;
    write_state(&mut cut, ARENA, STATE_UNWINDING);
    let (froze, snap, host) = durable_run(
        x,
        &inst,
        unheld(&inst, g, &e, false),
        &MemLayout::image(cut),
    );
    let thawed = match (froze, snap) {
        // A root that unwound returns 0, as one that ran to its end may: the state word tells them
        // apart.
        (End::Value(0), Some(snap)) if read_state(&snap.to_dense().0, ARENA) != STATE_UNWINDING => {
            End::Stuck("the run ended before its freeze landed".into())
        }
        (End::Value(0), Some(snap)) => thaw(x, &inst, g, &e, armed, &snap, host),
        (End::Value(0), None) => End::Stuck("the frozen run left no window".into()),
        (End::Declined, _) => End::Declined,
        (other, _) => End::Stuck(format!("the freeze did not unwind the root: {other:?}")),
    };
    e.stopper.close();
    match thawed {
        _ if thawed == *want => thawed,
        // The two-run rule's first half: a stopped process its thaw cannot continue.
        End::Trap(t) if t == "ThreadFault" && !armed && continued => End::Trap(t),
        End::Declined => End::Declined,
        other => End::Stuck(format!("frozen and thawed: {other:?}")),
    }
}

/// Serialize the run `e` froze, leaving `snap` and `host`, restore it into the powerbox `e` gives a
/// fresh run, and run it on: how the thaw ends.
fn thaw(
    x: Executor,
    inst: &Arc<Module>,
    g: &Guest,
    e: &Embedder,
    armed: bool,
    snap: &MemLayout,
    mut host: Host,
) -> End {
    if let Some(why) = host.take_freeze_declined() {
        return End::Stuck(format!("the freeze was declined: {why:?}"));
    }
    let art = match temen_snapshot::freeze_layout(inst, snap, DURABLE_RESERVED_LOG2, &host) {
        Ok(art) => art,
        // The cut's boundary (DURABILITY.md §12.5, #1680, #2025): a handle no artifact carries, or
        // a pipe or region something outside the cut shares. The run cannot be frozen, a decline.
        Err(
            FreezeError::NonDurableHandle(_)
            | FreezeError::PipeCrossesCut
            | FreezeError::RegionNotCaptured(_),
        ) => return End::Declined,
        Err(why) => return End::Stuck(format!("the codec refuses the freeze: {why:?}")),
    };
    let (mut thawing, args) = unheld(inst, g, e, armed);
    let registrar = e.clone();
    thawing.set_named_cap_registrar(Box::new(move |name, _state, _payer| {
        probes::regrant(name, &registrar)
    }));
    let twin = match temen_snapshot::restore_layout(&art, inst, &mut thawing) {
        Ok((twin, _)) => twin,
        Err(why) => return End::Stuck(format!("the codec refuses the restore: {why:?}")),
    };
    let (mut bytes, prots) = twin.to_dense();
    begin_thaw(&mut bytes, ARENA, 0);
    let twin = MemLayout::from_dense(bytes, &prots, twin.mapped());
    durable_run(x, inst, (thawing, args), &twin).0
}

/// One run of `inst` on `e`'s durable entry, starting from `window` in a reservation of
/// `2^DURABLE_RESERVED_LOG2` bytes: how it ended, the window it left, and its host. The bytecode
/// engine's entry takes and leaves a flat window, so its window carries no page map (#2628).
fn durable_run(
    e: Executor,
    inst: &Module,
    (mut host, args): (Host, Vec<Value>),
    window: &MemLayout,
) -> (End, Option<MemLayout>, Host) {
    let mut fuel = FUEL;
    match e {
        Executor::TreeWalk => {
            let (r, snap) = temen_interp::run_capture_reserved_with_host_layout(
                inst,
                0,
                &args,
                &mut fuel,
                window,
                DURABLE_RESERVED_LOG2,
                &mut host,
            );
            (End::of(r), snap, host)
        }
        Executor::Pump => match temen_interp::bytecode::compile_and_run_capture_reserved_with_host(
            inst,
            0,
            &args,
            &mut fuel,
            &window.to_dense().0,
            DURABLE_RESERVED_LOG2,
            &mut host,
        ) {
            Some((r, snap)) => (End::of(r), Some(MemLayout::image(snap)), host),
            None => (End::Declined, None, host),
        },
        Executor::Cranelift => on_cranelift(inst, (host, args), window, DURABLE_RESERVED_LOG2),
        Executor::Parallel => unreachable!("executor 2 has no durable runner (#2541)"),
    }
}
