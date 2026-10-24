//! **The honesty pin for the feature × run-mode matrix (#2545).**
//!
//! `crates/temen-parity/src/run_modes.rs` lists the runners this test has ([`RUNNERS`]) and the
//! exceptions to "a run gives the oracle's answer" ([`exception`], [`config_exception`]). Every cell
//! of the matrix is a fold of those runs. This runs every probe on every runner and fails when a run
//! and the manifest disagree, the role `frontier_conformance.rs` plays for the frontier matrix.
//!
//! ## What is driven
//!
//! One **probe** per feature (`support/run_probes.rs`): a small program that uses the feature, the
//! powerbox it needs, and the answer the oracle gives. A **runner** is one configuration of the
//! manifest, run the way its embedder runs it: the four executors one-shot, the DAP's two debug
//! engines (the tree-walk `Inspector` and the bytecode debug scheduler), and `CoopRun` on the pump
//! and on executor 2.
//!
//! A probe's embedder may have to act from outside the run ([`Outside`]), and each runner acts as
//! its embedder would. A run the embedder holds hands control back when it can only wait on the
//! embedder, and is fed or continued, and resumed. Any other run pulls its input from a stdin
//! source, which blocks until a thread supplies it, as the CLI's interactive mode does. A process
//! that stops itself in a run nobody holds is run twice (#2529). With nothing to continue it, the
//! run must fault `ThreadFault`. With its bell armed, saying the embedder may act from outside, it
//! must wait for the continue a thread sends once it has asked to stop, and give the oracle's
//! answer. The first run is what tells an engine that ignores the stop apart, and with no race:
//! that engine never faults.
//!
//! A run is held to its exception:
//!
//! - none: the oracle's answer;
//! - `Declines`: the runner declines the module;
//! - `NotYet`: anything but the oracle's answer, so a gap that closes fails the test until its
//!   exception goes;
//! - `Conditional`: the oracle's answer where the condition holds ([`condition`]), a decline
//!   elsewhere.
//!
//! The harness's runners are exactly the manifest's, both ways, so a cell can only come from a run
//! this file makes.

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "support/run_probes.rs"]
mod probes;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;
mod support;

use std::sync::{mpsc, Arc};
use std::time::Duration;

use drivers::{run_on, Driver, Pumped};
use probes::{module, probes, Embedder, Outside, Probe, Stopper};
use support::cites_an_issue;
use temen_interp::bytecode::{
    module_spawns_threads, CoopEvent, CoopRun, SchedStop, ScheduledDebugRun,
};
use temen_interp::{Host, Inspector, Stop, Trap, Value};
use temen_ir::Module;
use temen_parity::run_modes::{
    config_exception, exception, runner_for, Config, Executor, Feature, Property, Props, Row,
    RUNNERS, UNPROBED,
};
use temen_parity::Status;

/// The fuel every run gets.
const FUEL: u64 = 50_000_000;

/// How long one run may take before the harness calls it hung. A harness must fail, never hang.
const BOUND: Duration = Duration::from_secs(30);

/// How long a runner waits before resuming a run idle on an answer the probe gives itself.
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

/// How one run ended, whatever engine ran it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum End {
    /// The entry returned this.
    Value(i64),
    /// The run trapped, as its engine names the trap.
    Trap(String),
    /// The runner declined the module.
    Declined,
    /// Anything else: an answer of another shape, a stop the runner could not move on from.
    Stuck(String),
}

impl End {
    fn of(r: Result<Vec<Value>, Trap>) -> End {
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
enum Way {
    /// An executor, one-shot.
    OneShot(Executor),
    /// The tree-walk debugger (`Inspector`, the DAP's default engine).
    Inspector,
    /// The bytecode debug scheduler (`ScheduledDebugRun`, the DAP's `bytecode` engine).
    DebugScheduler,
    /// `CoopRun`, held and resumed by this harness.
    Held(Pumped),
}

/// The harness's runner for `c`, if it has one. [`RUNNERS`] lists the same configurations, which
/// [`the_harness_runs_exactly_the_manifests_runners`] holds.
fn way(c: Config) -> Option<Way> {
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
    } else {
        return None;
    })
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

/// The verdict on a run that ended `end`, held to the oracle's answer `want`.
fn judge(end: End, want: i64) -> Verdict {
    match end {
        End::Value(x) if x == want => Verdict::Matched,
        End::Declined => Verdict::Declined,
        other => Verdict::Other(format!("{other:?}")),
    }
}

/// Run `probe` on configuration `c`.
fn run(c: Config, probe: &Probe) -> Verdict {
    let way = way(c).expect("every runner of the manifest has a way to run it");
    let probe = *probe;
    bounded(format!("{:?} on {}", probe.feature, c.label()), move || {
        let m = module(&(probe.src)());
        let unheld = matches!(way, Way::OneShot(_) | Way::Inspector);
        if probe.outside != Outside::Continue || !unheld {
            return judge(once(way, &m, &probe, false), probe.want);
        }
        // A stop nobody can continue is a deadlock; one the embedder may continue from outside is
        // a wait (#2529).
        let alone = once(way, &m, &probe, false);
        let continued = once(way, &m, &probe, true);
        match (&alone, &continued) {
            (End::Trap(t), End::Value(x)) if t == "ThreadFault" && *x == probe.want => {
                Verdict::Matched
            }
            (End::Declined, End::Declined) => Verdict::Declined,
            _ => Verdict::Other(format!(
                "with nothing to continue it {alone:?}; continued from outside {continued:?}"
            )),
        }
    })
}

/// Run `probe` once on `way`, over a fresh embedder. `armed` arms the run's external-wake bell,
/// saying its embedder may act on it from outside, and has a thread continue the process once it
/// asks to stop.
fn once(way: Way, m: &Module, probe: &Probe, armed: bool) -> End {
    let e = Embedder::default();
    // A run nobody holds gets what its embedder gives it from outside from a thread of its own.
    let unheld = || {
        let (mut host, args) = (probe.setup)(m, &e);
        if let Outside::Feed(bytes) = probe.outside {
            feed_later(&mut host, bytes);
        }
        if armed {
            host.arm_external_wake();
            let s = Arc::clone(&e.stopper);
            std::thread::spawn(move || {
                s.await_stop();
                if s.is_stopped() {
                    s.set(false);
                }
            });
        }
        (host, args)
    };
    let one_shot = |d| run_on(d, m, &unheld).map_or(End::Declined, |ran| End::of(ran.result));
    let end = match way {
        Way::OneShot(Executor::TreeWalk) => one_shot(Driver::Oracle),
        Way::OneShot(Executor::Pump) => one_shot(Driver::Coop),
        Way::OneShot(Executor::Parallel) => one_shot(Driver::Parallel),
        Way::OneShot(Executor::Cranelift) => on_cranelift(m, unheld()),
        Way::Inspector => inspected(m, unheld()),
        Way::DebugScheduler => debugged(m, (probe.setup)(m, &e), probe.outside, &e.stopper),
        Way::Held(exec) => held(exec, m, (probe.setup)(m, &e), probe.outside, &e.stopper),
    };
    // The run is over: a continue thread still waiting has nothing to continue.
    e.stopper.close();
    end
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

/// Run `m` on the Cranelift JIT through `temen_run`, over a window seeded at its declared size in
/// the default reservation, as `temen_run::Instance` runs it, with the `call.dyn` table the
/// powerbox's `Jit` grant reserved (`temen_run::grant_jit` asks the compile for the same size).
/// The JIT refuses a module outside its subset with `Unsupported`; any other error is an answer
/// the oracle did not give.
fn on_cranelift(m: &Module, (mut host, args): (Host, Vec<Value>)) -> End {
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
        Ok((temen_jit::JitOutcome::Returned(v), _)) => match v[..] {
            [x] => End::Value(x),
            _ => End::Stuck(format!("returned {v:?}")),
        },
        Ok((temen_jit::JitOutcome::Trapped(t), _)) => End::Trap(format!("{t:?}")),
        Ok((other, _)) => End::Stuck(format!("{other:?}")),
        Err(temen_jit::JitError::Unsupported(_)) => End::Declined,
        Err(e) => End::Stuck(format!("{e:?}")),
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

/// Hold `m` as a `CoopRun` on `exec` and resume it to its end. A run idle on a stopped process gets
/// it continued, and one idle on what the embedder feeds gets the probe's bytes once; idle
/// otherwise, it waits on an answer the probe gives itself, and is resumed shortly, the time it
/// waited passed.
fn held(
    exec: Pumped,
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
    let reserved = temen_ir::default_reserved_log2();
    let run = match exec {
        Pumped::Pump => CoopRun::new_reserved(m, 0, &args, FUEL, host, None, &[], reserved),
        Pumped::Threads => {
            CoopRun::new_parallel(m, 0, &args, FUEL, host, &[], reserved, None, true)
        }
    };
    let mut run = match run {
        None => return End::Declined,
        Some(Err(t)) => return End::Stuck(format!("{t:?} at the start")),
        Some(Ok(run)) => run,
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

/// Whether a `Conditional` exception's condition holds for the probe as this harness drives it: the
/// JIT's fiber runtime for the features that need it, and an OS-backed region, which the probe
/// uses, for a shared region on the JIT.
fn condition(f: Feature, c: Config) -> bool {
    match (f, c.executor) {
        (Feature::Threads | Feature::Fibers | Feature::Children, Executor::Cranelift) => {
            temen_jit::fiber_supported()
        }
        _ => true,
    }
}

/// What is wrong with `verdict` as the run of `f` on `c`, or `None` if the manifest says it.
fn disagreement(f: Feature, c: Config, verdict: &Verdict) -> Option<String> {
    let ok = match exception(f, c).map(|x| x.status) {
        None => *verdict == Verdict::Matched,
        Some(Status::Declines) => *verdict == Verdict::Declined,
        Some(Status::NotYet) => *verdict != Verdict::Matched,
        Some(Status::Conditional) if condition(f, c) => *verdict == Verdict::Matched,
        Some(Status::Conditional) => *verdict == Verdict::Declined,
        Some(s @ (Status::Full | Status::Unaudited)) => {
            return Some(format!(
                "its exception is {s:?}, which an exception cannot be"
            ))
        }
    };
    if ok {
        return None;
    }
    Some(match exception(f, c) {
        None => format!(
            "gave {verdict:?}, not the oracle's answer, and the manifest records no exception: \
             fix the engine, or file an issue and add a NotYet exception citing it"
        ),
        Some(x) if x.status == Status::NotYet => format!(
            "now gives the oracle's answer, so its NotYet exception is stale: delete it and close \
             its issue ({})",
            x.note
        ),
        Some(x) => format!("is {:?} in the manifest, but gave {verdict:?}", x.status),
    })
}

/// **Every run gives what the manifest says.** Each probe runs on every runner, and the verdicts
/// must agree with the exceptions. A failure lists every disagreeing run, not just the first.
#[test]
fn every_run_matches_the_manifest() {
    let mut wrong = Vec::new();
    for probe in probes() {
        for r in RUNNERS {
            let verdict = run(r.config, &probe);
            if let Some(why) = disagreement(probe.feature, r.config, &verdict) {
                wrong.push(format!(
                    "  {} on {}: {why}",
                    probe.feature.label(),
                    r.config.label()
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the run-mode matrix disagrees with the runs:\n{}",
        wrong.join("\n")
    );
}

/// **The harness runs exactly the manifest's runners, and probes exactly its probed features.**
/// This is what makes every cell a fold of runs this file makes: a configuration the manifest names
/// has a runner here, a runner here is in the manifest, and a probed feature has one probe.
#[test]
fn the_harness_runs_exactly_the_manifests_runners() {
    for e in Executor::ALL {
        for props in Props::every() {
            let c = Config { executor: e, props };
            assert_eq!(
                way(c).is_some(),
                RUNNERS.iter().any(|r| r.config == c),
                "{}: the harness and RUNNERS must agree on whether it runs",
                c.label(),
            );
        }
    }
    let all = probes();
    for f in Feature::ALL {
        let n = all.iter().filter(|p| p.feature == f).count();
        let want = usize::from(!UNPROBED.contains(&f));
        assert_eq!(
            n,
            want,
            "{}: {n} probes, but the manifest says {want}",
            f.label()
        );
    }
}

/// **A cell's runner on each executor is unambiguous.** A cell uses, on each executor, the runner
/// that adds its properties and the fewest others; two such runners would make the cell depend on
/// the order of [`RUNNERS`].
#[test]
fn no_two_runners_tie() {
    for e in Executor::ALL {
        for need in Props::every().filter(|n| n.len() <= 2) {
            let Some(best) = runner_for(e, need) else {
                continue;
            };
            let ties = RUNNERS
                .iter()
                .filter(|r| {
                    r.config.executor == e
                        && r.config.props.contains(need)
                        && r.config.props.len() == best.props.len()
                })
                .count();
            assert_eq!(ties, 1, "{} for {need:?}: {ties} runners tie", e.label());
        }
    }
}

/// **A reason no run exists is never given where one does.** [`config_exception`] is consulted only
/// where no runner covers the combination, so an entry a runner now covers is stale: delete it.
#[test]
fn no_reason_is_given_for_a_configuration_that_runs() {
    for e in Executor::ALL {
        for props in Props::every() {
            if config_exception(e, props).is_none() {
                continue;
            }
            let covered = RUNNERS
                .iter()
                .any(|r| r.config.executor == e && r.config.props.contains(props));
            assert!(
                !covered,
                "{} with {props:?}: config_exception says why it does not run, but a runner \
                 covers it",
                e.label(),
            );
        }
    }
}

/// A **known gap** must name the issue tracking it, as on the frontier matrix, with the same shape
/// check (`support::cites_an_issue`): `NotYet` is only actionable if the reader can find out what is
/// being done about it. Checked over every configuration, not just the runners', since a rule
/// applies to a configuration as soon as a runner has it.
#[test]
fn every_known_gap_names_its_issue() {
    for e in Executor::ALL {
        for props in Props::every() {
            let c = Config { executor: e, props };
            let reasons = Feature::ALL
                .into_iter()
                .filter_map(|f| exception(f, c))
                .chain(config_exception(e, props));
            for x in reasons {
                assert!(
                    x.status != Status::NotYet || cites_an_issue(x.note),
                    "{}: a NotYet exception must cite the issue tracking the gap (`#1234`), got \
                     {:?}",
                    c.label(),
                    x.note,
                );
            }
        }
    }
}

/// The audited count only grows: filling the matrix in is the work (#2545), and emptying it is a
/// regression.
#[test]
fn audited_coverage_does_not_regress() {
    let audited = Row::all()
        .flat_map(|r| r.cells())
        .flatten()
        .filter(|c| c.status != Status::Unaudited)
        .count();
    assert!(
        audited >= 85,
        "audited cell count fell to {audited}; it was 85 when the job-control row got its probe \
         (#2529)"
    );
}
