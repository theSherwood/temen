//! **The honesty pin for the feature × run-mode matrix (#2545).**
//!
//! `crates/temen-parity/src/run_modes.rs` states, for each guest feature and each run property,
//! what every executor and property does with it. This runs each case and fails when a claim and
//! the run disagree, the same role `frontier_conformance.rs` plays for the frontier matrix.
//!
//! ## What is driven
//!
//! One **probe** per feature (`support/run_probes.rs`): a small program that uses the feature, the
//! powerbox it needs, and the answer the oracle gives. Each way the harness can run a probe is a
//! **runner**: the four native executors one-shot, the DAP's two debug engines for `debugger` (the
//! tree-walk `Inspector` and the bytecode debug scheduler), and `CoopRun` on the pump and on
//! executor 2 for `pumped`. A probe that needs its embedder to feed it
//! between steps has no one-shot runner: an executor runs it only where the embedder can hold the
//! run, pumped. A cell is checked against the verdict of every runner its column has:
//!
//! - `Full`: every runner gives the oracle's answer;
//! - `NotYet`: some runner does not;
//! - `Declines`: every runner declines the module, by design;
//! - `Conditional`: the runners give the answer where the note's condition holds, and decline
//!   elsewhere ([`condition`] says which the probe meets);
//! - `Unaudited`: not checked.
//!
//! Two rules keep the matrix from claiming more than the harness drives. A cell with no runner, or
//! in a feature's row with no probe, must be `Unaudited`. And a property × executor cell is `Full`
//! exactly where the property has a runner on that executor.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "support/run_probes.rs"]
mod probes;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;
mod support;

use std::sync::mpsc;
use std::time::Duration;

use drivers::{run_on, Driver, Pumped};
use probes::{module, probes, Probe};
use support::cites_an_issue;
use temen_interp::bytecode::{
    module_spawns_threads, CoopEvent, CoopRun, SchedStop, ScheduledDebugRun,
};
use temen_interp::{Inspector, Stop, Trap, Value};
use temen_ir::Module;
use temen_parity::run_modes::{
    feature_cells, property_cells, Executor, Feature, Mode, Property, Row,
};
use temen_parity::Status;

/// The fuel every run gets.
const FUEL: u64 = 50_000_000;

/// How long one run may take before the harness calls it hung. A harness must fail, never hang.
const BOUND: Duration = Duration::from_secs(30);

/// How long a runner waits before pumping again a run idle on an answer the probe gives itself.
const ANSWER_POLL: Duration = Duration::from_millis(1);

/// What one runner did with a probe.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Verdict {
    /// The oracle's answer.
    Matched,
    /// The runner declined the module: it is outside the engine's subset, or the run fell back.
    Declined,
    /// Anything else: a different answer, a trap the oracle did not take, a stop it could not
    /// move on from.
    Other(String),
}

/// A way to run a probe.
#[derive(Clone, Copy, Debug)]
enum Runner {
    /// An executor, one-shot.
    OneShot(Executor),
    /// The tree-walk debugger (`Inspector`, the DAP's default engine), for `debugger`.
    Inspect,
    /// The bytecode debug scheduler (`ScheduledDebugRun`, the DAP's `bytecode` engine), for
    /// `debugger`.
    Debug,
    /// `CoopRun`, pumped by this harness (`pumped`).
    Pumped(Pumped),
}

/// The runners column `m` has, for a probe that needs feeding between steps or not (`fed`). An
/// executor is its own runner, or, for a fed probe, its pumped run where it has one. A property
/// has one runner per executor it runs on, and none until its slice of #2545 drives it.
fn runners(m: Mode, fed: bool) -> Vec<Runner> {
    match m {
        Mode::Executor(e) if !fed => vec![Runner::OneShot(e)],
        Mode::Executor(Executor::Pump) => vec![Runner::Pumped(Pumped::Pump)],
        Mode::Executor(Executor::Parallel) => vec![Runner::Pumped(Pumped::Threads)],
        Mode::Executor(Executor::TreeWalk | Executor::Cranelift) => Vec::new(),
        Mode::Property(Property::Debugger) => vec![Runner::Inspect, Runner::Debug],
        Mode::Property(Property::Pumped) => {
            vec![
                Runner::Pumped(Pumped::Pump),
                Runner::Pumped(Pumped::Threads),
            ]
        }
        Mode::Property(
            Property::Durable
            | Property::Warm
            | Property::Reactor
            | Property::Browser
            | Property::Emitted,
        ) => Vec::new(),
    }
}

/// The executor a runner runs on, for the property × executor half.
fn executor_of(r: Runner) -> Executor {
    match r {
        Runner::OneShot(e) => e,
        Runner::Inspect => Executor::TreeWalk,
        Runner::Debug => Executor::Pump,
        Runner::Pumped(Pumped::Pump) => Executor::Pump,
        Runner::Pumped(Pumped::Threads) => Executor::Parallel,
    }
}

/// Run `f` on a thread of its own, for at most [`BOUND`].
fn bounded<T: Send + 'static>(what: String, f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(BOUND)
        .unwrap_or_else(|_| panic!("{what}: no end within {BOUND:?}"))
}

/// The verdict on a run that ended `r`, held to `want`.
fn judge(r: Result<Vec<Value>, Trap>, want: i64) -> Verdict {
    match r {
        Ok(v) if v == vec![Value::I64(want)] => Verdict::Matched,
        other => Verdict::Other(format!("{other:?}")),
    }
}

/// Run `probe` on `runner`.
fn run(runner: Runner, probe: &Probe) -> Verdict {
    let (src, setup, feed, want) = (probe.src, probe.setup, probe.feed, probe.want);
    bounded(format!("{:?} on {runner:?}", probe.feature), move || {
        let m = module(&src());
        let one_shot = |d| match run_on(d, &m, &|| setup(&m)) {
            Some(ran) => judge(ran.result, want),
            None => Verdict::Declined,
        };
        match runner {
            Runner::OneShot(Executor::TreeWalk) => one_shot(Driver::Oracle),
            Runner::OneShot(Executor::Pump) => one_shot(Driver::Coop),
            Runner::OneShot(Executor::Parallel) => one_shot(Driver::Parallel),
            Runner::OneShot(Executor::Cranelift) => on_cranelift(&m, setup, want),
            Runner::Inspect => inspected(&m, setup, feed, want),
            Runner::Debug => debugged(&m, setup, feed, want),
            Runner::Pumped(exec) => pumped(exec, &m, setup, feed, want),
        }
    })
}

/// Run `m` on the Cranelift JIT through `temen_run`, over a window seeded at its declared size in
/// the default reservation, as `temen_run::Instance` runs it, with the `call.dyn` table the
/// powerbox's `Jit` grant reserved (`temen_run::grant_jit` asks the compile for the same size).
/// The JIT refuses a module outside its subset with `Unsupported`; any other error is an answer
/// the oracle did not give.
fn on_cranelift(
    m: &Module,
    setup: fn(&Module) -> (temen_interp::Host, Vec<Value>),
    want: i64,
) -> Verdict {
    let (mut host, args) = setup(m);
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(x) => i64::from(*x),
            Value::I64(x) => *x,
            other => panic!("unexpected argument {other:?}"),
        })
        .collect();
    let size = 1usize << m.memory.map_or(16, |mem| mem.size_log2);
    let window = temen_interp::MemLayout::image(vec![0u8; size]);
    let reserved = temen_ir::default_reserved_log2();
    let table = host.jit_table_log2();
    match temen_run::jit_cap_run(m, 0, &args, &window, reserved, table, &mut host, None) {
        Ok((temen_jit::JitOutcome::Returned(v), _)) if v == vec![want] => Verdict::Matched,
        Ok((other, _)) => Verdict::Other(format!("{other:?}")),
        Err(temen_jit::JitError::Unsupported(_)) => Verdict::Declined,
        Err(e) => Verdict::Other(format!("{e:?}")),
    }
}

/// Run `m` under the tree-walk debugger to its end, as the best its API allows: scheduled, as the
/// DAP debugs a threaded program, when `m` spawns threads, else on one vCPU over the probe's
/// powerbox. The scheduled `Inspector` builds its own empty powerbox, so a threaded probe's handles
/// name nothing there. It has no way to feed a read either.
fn inspected(
    m: &Module,
    setup: fn(&Module) -> (temen_interp::Host, Vec<Value>),
    feed: Option<&'static [u8]>,
    want: i64,
) -> Verdict {
    if feed.is_some() {
        return Verdict::Other("the Inspector has no way to feed a read".into());
    }
    let (host, args) = setup(m);
    let mut insp = if module_spawns_threads(m) {
        Inspector::attach_scheduled(m, 0, &args, FUEL, Vec::new())
    } else {
        Inspector::attach_with_host(m, 0, &args, FUEL, host)
    };
    loop {
        match insp.run_until_stop() {
            Stop::Finished(r) => return judge(r, want),
            Stop::Break { .. } => {}
            Stop::Blocked => return Verdict::Other("Blocked".into()),
        }
    }
}

/// Run `m` under the debug scheduler to its end, as the DAP's bytecode engine does. A stop that
/// waits on the embedder is moved on: a read the probe feeds gets its bytes, and any other idle
/// waits on a host-answered call the probe answers itself, so it is looked at again shortly. The
/// run says whether it waits on a read (`stdin_parked`), which only the run changes; whether a call
/// is still owed can change between the stop and the question.
fn debugged(
    m: &Module,
    setup: fn(&Module) -> (temen_interp::Host, Vec<Value>),
    feed: Option<&'static [u8]>,
    want: i64,
) -> Verdict {
    let (host, args) = setup(m);
    let Some(mut d) = ScheduledDebugRun::new_with_host(m, 0, &args, host) else {
        return Verdict::Declined;
    };
    let mut fuel = FUEL;
    let mut feed = feed;
    loop {
        match d.run_until_stop(&mut fuel) {
            SchedStop::Finished(r) => return judge(r, want),
            SchedStop::Declined => return Verdict::Declined,
            SchedStop::Break { .. } | SchedStop::CapPark { .. } => {}
            SchedStop::Idle { .. } if !d.stdin_parked() => std::thread::sleep(ANSWER_POLL),
            SchedStop::Idle { .. } if feed.is_some() => {
                d.provide_stdin(feed.take().unwrap_or_default())
            }
            other => return Verdict::Other(format!("{other:?}")),
        }
    }
}

/// Hold `m` as a `CoopRun` on `exec` and pump it to its end. A run idle on what the embedder feeds
/// gets the probe's bytes once; idle otherwise, it waits on an answer the probe gives itself, and
/// is pumped again shortly, the time it waited passed.
fn pumped(
    exec: Pumped,
    m: &Module,
    setup: fn(&Module) -> (temen_interp::Host, Vec<Value>),
    feed: Option<&'static [u8]>,
    want: i64,
) -> Verdict {
    let (host, args) = setup(m);
    let reserved = temen_ir::default_reserved_log2();
    let run = match exec {
        Pumped::Pump => CoopRun::new_reserved(m, 0, &args, FUEL, host, None, &[], reserved),
        Pumped::Threads => {
            CoopRun::new_parallel(m, 0, &args, FUEL, host, &[], reserved, None, true)
        }
    };
    let mut run = match run {
        None => return Verdict::Declined,
        Some(Err(t)) => return Verdict::Other(format!("{t:?} at the start")),
        Some(Ok(run)) => run,
    };
    let mut feed = feed;
    loop {
        match run.run() {
            CoopEvent::Done(v) => return judge(Ok(v), want),
            CoopEvent::Trapped(t) => return judge(Err(t), want),
            CoopEvent::CapPark { .. } | CoopEvent::Paused => {}
            CoopEvent::Idle { .. } => match feed.take() {
                Some(bytes) => run.host_mut().push_stdin(bytes),
                None => {
                    std::thread::sleep(ANSWER_POLL);
                    // The run's timers see the time it waited, as an embedder's do (#2557).
                    run.pass_time(ANSWER_POLL.as_nanos() as u64);
                }
            },
            _ => return Verdict::Other("a round-trip only the browser's tier makes".into()),
        }
    }
}

/// Whether a `Conditional` cell's condition holds for the probe as this harness drives it: the
/// JIT's fiber runtime for the features that need it, and an OS-backed region, which the probe
/// uses, for a shared region on the JIT.
fn condition(f: Feature, m: Mode) -> bool {
    match (f, m) {
        (
            Feature::Threads | Feature::Fibers | Feature::Children,
            Mode::Executor(Executor::Cranelift),
        ) => temen_jit::fiber_supported(),
        _ => true,
    }
}

/// Whether `verdicts` (one per runner the column has) agree with the claimed `status`. `cond` is
/// whether a `Conditional` cell's condition holds ([`condition`]).
fn agrees(status: Status, verdicts: &[Verdict], cond: bool) -> bool {
    let all_matched = verdicts.iter().all(|v| *v == Verdict::Matched);
    let all_declined = verdicts.iter().all(|v| *v == Verdict::Declined);
    match status {
        Status::Full => all_matched,
        Status::NotYet => !all_matched,
        Status::Declines => all_declined,
        Status::Conditional if cond => all_matched,
        Status::Conditional => all_declined,
        Status::Unaudited => true,
    }
}

/// **Every driven cell of a feature's row matches what its column's runners do.** A probe runs on
/// every runner of every column its row claims anything in, and the verdicts must agree with the
/// claim. A failure lists every disagreeing cell, not just the first.
#[test]
fn every_driven_feature_cell_matches_what_its_runners_do() {
    let mut wrong = Vec::new();
    for probe in probes() {
        let cells = feature_cells(probe.feature);
        for m in Mode::ALL {
            let cell = cells[m.index()];
            if cell.status == Status::Unaudited {
                continue;
            }
            let verdicts: Vec<Verdict> = runners(m, probe.feed.is_some())
                .into_iter()
                .map(|r| run(r, &probe))
                .collect();
            if !agrees(cell.status, &verdicts, condition(probe.feature, m)) {
                wrong.push(format!(
                    "  {} / {}: claims {:?}, but the runners gave {verdicts:?}",
                    probe.feature.name(),
                    m.short(),
                    cell.status,
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the run-mode matrix disagrees with the runs — fix the manifest or the engine:\n{}",
        wrong.join("\n")
    );
}

/// **A cell claims nothing the harness cannot drive.** A feature with no probe, or a cell with no
/// runner, is `Unaudited`. This is what makes "a cell is claimed only where a test drives it" a
/// property of the tree rather than a promise.
#[test]
fn nothing_is_claimed_where_nothing_drives_it() {
    let all = probes();
    for row in Row::all() {
        let cells = row.cells();
        for m in Mode::ALL {
            let cell = cells[m.index()];
            if cell.status == Status::Unaudited {
                continue;
            }
            let driven = match row {
                Row::Feature(f) => all
                    .iter()
                    .find(|p| p.feature == f)
                    .is_some_and(|p| !runners(m, p.feed.is_some()).is_empty()),
                // A property's row: its executors are checked against its runners below, so it
                // needs some; a property combines with itself, and with another property only
                // where both have runners.
                Row::Property(p) => {
                    let has = |q| !runners(Mode::Property(q), false).is_empty();
                    match m {
                        Mode::Executor(_) => has(p),
                        Mode::Property(q) => p == q || (has(p) && has(q)),
                    }
                }
            };
            assert!(
                driven,
                "{} / {}: claims {:?}, but nothing in this test drives that cell",
                row.name(),
                m.short(),
                cell.status,
            );
        }
    }
}

/// **A property's row says where it runs.** A property × executor cell is `Full` exactly where the
/// property has a runner on that executor, and says why not (`Declines`, or a `NotYet` with its
/// issue) where it has none. A property with no runners at all is still `Unaudited`: slices 2 to 4
/// of #2545 drive them.
#[test]
fn a_property_runs_exactly_on_the_executors_it_has_runners_for() {
    for p in Property::ALL {
        let mine = runners(Mode::Property(p), false);
        if mine.is_empty() {
            continue;
        }
        let cells = property_cells(p);
        for e in Executor::ALL {
            let cell = cells[Mode::Executor(e).index()];
            let has = mine.iter().any(|&r| executor_of(r) == e);
            let ok = match cell.status {
                Status::Full | Status::Conditional => has,
                Status::Declines | Status::NotYet | Status::Unaudited => !has,
            };
            assert!(
                ok,
                "{} / {}: claims {:?}, but the property {} a runner there",
                p.short(),
                Mode::Executor(e).short(),
                cell.status,
                if has { "has" } else { "has no" },
            );
        }
    }
}

/// **The property × property half is symmetric, and a property combines with itself.**
#[test]
fn the_property_half_is_symmetric() {
    for p in Property::ALL {
        let mine = property_cells(p);
        assert_eq!(
            mine[Mode::Property(p).index()].status,
            Status::Full,
            "{}: a property combines with itself",
            p.short(),
        );
        for q in Property::ALL {
            let theirs = property_cells(q);
            assert_eq!(
                mine[Mode::Property(q).index()],
                theirs[Mode::Property(p).index()],
                "{} × {} must read the same from either row",
                p.short(),
                q.short(),
            );
        }
    }
}

/// A **known gap** must name the issue tracking it, as on the frontier matrix, with the same shape
/// check (`support::cites_an_issue`): `NotYet` is only actionable if the reader can find out what is
/// being done about it.
#[test]
fn every_known_gap_names_its_issue() {
    for row in Row::all() {
        for (cell, m) in row.cells().into_iter().zip(Mode::ALL) {
            if cell.status != Status::NotYet {
                continue;
            }
            assert!(
                cites_an_issue(cell.note),
                "{}/{}: a `NotYet` cell must cite the issue tracking the gap (`#1234`), got {:?}",
                row.name(),
                m.short(),
                cell.note,
            );
        }
    }
}

/// The audited count only grows: filling the matrix in is the work (#2545), and emptying it is a
/// regression.
#[test]
fn audited_coverage_does_not_regress() {
    let audited = Row::all()
        .flat_map(|r| r.cells())
        .filter(|c| c.status != Status::Unaudited)
        .count();
    assert!(
        audited >= 85,
        "audited cell count fell to {audited}; it was 85 when the matrix landed (#2545)"
    );
}
