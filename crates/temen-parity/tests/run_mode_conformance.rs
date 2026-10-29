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
//! engines (the tree-walk `Inspector` and the bytecode debug scheduler), `CoopRun` on the pump and
//! on executor 2, the durable entries of the tree-walker, the pump and the Cranelift JIT, each run
//! frozen and thawed, and the pump's reactor, its entry called as two frames, each run to its end
//! or held. The runners live in `support/runners.rs`, which the frontier matrix's run-driven axes
//! run on too (D3a).
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
#[path = "support/runners.rs"]
mod runners;
mod support;

use probes::{module, probes, Outside, Probe};
use runners::{bounded, once, way, End, Guest};
use support::cites_an_issue;
use temen_parity::run_modes::{
    config_exception, exception, runner_for, Config, Executor, Feature, Props, Row, RUNNERS,
    UNPROBED,
};
use temen_parity::Status;

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
        let g = Guest {
            module: &m,
            setup: &probe.setup,
            outside: probe.outside,
        };
        let want = End::Value(probe.want);
        if probe.outside != Outside::Continue || !way.unheld() {
            return judge(once(way, &g, false, Some(&want)), probe.want);
        }
        // A stop nobody can continue is a deadlock; one the embedder may continue from outside is
        // a wait (#2529).
        let alone = once(way, &g, false, Some(&want));
        let continued = once(way, &g, true, Some(&want));
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
        audited >= 124,
        "audited cell count fell to {audited}; it was 102 when the frozen-and-thawed runners \
         landed, and 124 once the reactor runners did (#2545)"
    );
}
