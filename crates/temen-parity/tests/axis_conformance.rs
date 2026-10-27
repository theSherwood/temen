//! **The honesty pin for the frontier matrix's run-driven axes (#1413, #2545 D3a):** `backend`,
//! `concurrency`, `debugger` and `durability`.
//!
//! Each of these axes is a set of the run-mode matrix's columns ([`Axis::modes`]), and runs on that
//! matrix's runners (`support/runners.rs`): the pump and the Cranelift JIT for `backend`, the pump and
//! executor 2 for `concurrency`, the tree-walk `Inspector` and the bytecode debug scheduler for
//! `debugger`, and the durable entries of the tree-walker, the pump and the Cranelift JIT, each run
//! frozen and thawed, for `durability`. One harness and one definition of each column, so the two
//! matrices cannot disagree about what a configuration does with the same program (INVARIANTS #15).
//! `target` is the browser's column, which has no runner yet, so its cells stay unaudited.
//!
//! ## What is run
//!
//! Each capability's probe (`support/capability_probe.rs`): for each op the capability declares, a
//! one-call module per call shape `(op, argc, result type)`, over a powerbox holding a real handle.
//! The lowerings are arity- and signature-guarded, so each shape is run on every runner and compared
//! to its own twin on the oracle (the tree-walker's one-shot run). Folding each runner's best answer
//! across arities would compare an arity one runner reached against another one the oracle did,
//! manufacturing divergences that are really the harness miscalling.
//!
//! A row whose durability turns on how its handle was granted is minted once for each side, and
//! every axis runs each mint. A durable runner also holds each handle across a freeze it is not
//! called in ([`hold`]), since an empty interface or `Exit`'s op never reaches a freeze.
//!
//! A run is held to the oracle's answer for the same shape, as every run-mode cell is:
//!
//! - the same answer, a value, an errno or a trap: **matched**;
//! - the runner declines the module (outside its subset, a debug engine's declined seam, or a freeze
//!   the codec refuses at the cut's boundary): **declined**;
//! - anything else: a **divergence**. A runner that answers a shape the oracle refuses diverges as
//!   much as one that refuses a shape the oracle answers (#1531 was a driver answering differently,
//!   not refusing).
//!
//! A shape that both the oracle and the runner refuse as miscalled (`CapFault`, `Malformed`) never
//! reached an op, so there is nothing to compare.
//!
//! ## The cell
//!
//! A row's runs on an axis fold as a run-mode cell does: every run matched ✅, any divergence 🚧, every
//! run declined ⛔, a mix of matched and declined 🔶. The manifest's cell must say what the fold says,
//! or 🚧, citing its issue, for a fold short of ✅: a known gap may decline rather than diverge, as a
//! run-mode exception's `NotYet` allows. A capability that declares no ops has nothing to diverge
//! about: a little past its empty interface is swept anyway, so that is checked rather than assumed.
//!
//! As on every column, a row is scored only when the harness can mint its handle and drive its ops.
//! `Jit`, `JitCode`, `Offer` and `LiveImpl` need a live unit or a live peer, so they stay `Unaudited`
//! on each axis this file runs but `durability`, where the codec's classifier decides them
//! (`frontier_conformance.rs`).

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;
#[path = "support/run_probes.rs"]
mod probes;
#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;
#[path = "support/runners.rs"]
mod runners;
mod support;

use std::sync::mpsc;

use probes::{Embedder, Outside};
use runners::{once, way, End, Guest, Setup, Way, BOUND};
use support::capability_probe::{probe_module_with, rows, Mint, Row, MAX_ARGC, PROBE_BEYOND};
use temen_interp::{Host, Value};
use temen_ir::Module;
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::run_modes::{fold_status, Config, Executor, Property};
use temen_parity::Status;

/// What one runner did with one call shape, held to the oracle's answer for it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Verdict {
    Matched,
    Declined,
    /// What the oracle and the runner each did.
    Diverged(End, End),
}

/// One run that had an answer to compare.
struct Run {
    /// What the run called: an op of the row's interface, or none, the handle only held ([`hold`]).
    call: String,
    shape: String,
    config: Config,
    verdict: Verdict,
}

/// The axes runs decide.
fn run_axes() -> impl Iterator<Item = Axis> {
    Axis::ALL.into_iter().filter(|a| !a.runners().is_empty())
}

/// Every runner any of them uses, once.
fn all_runners() -> Vec<Config> {
    let mut all: Vec<Config> = Vec::new();
    for c in run_axes().flat_map(Axis::runners) {
        if !all.contains(&c) {
            all.push(c);
        }
    }
    all
}

/// A call that reached no op: the shape does not match the op's signature. One engine names it
/// `CapFault` and another `Malformed`.
fn miscalled(e: &End) -> bool {
    matches!(e, End::Trap(t) if t == "CapFault" || t == "Malformed")
}

/// Every `(argc, result type)` shape the sweep tries.
fn shapes() -> impl Iterator<Item = (usize, &'static str)> {
    (0..MAX_ARGC).flat_map(|a| ["i64", "i32"].map(move |r| (a, r)))
}

/// The ops a row's sweep calls: its declared ones, or a little past an empty interface.
fn ops(row: &Row) -> Vec<u32> {
    if row.ops.is_empty() {
        (0..PROBE_BEYOND).collect()
    } else {
        row.ops.to_vec()
    }
}

/// Sweep `cap`'s row on `runners`, on a thread of its own. A sweep makes thousands of runs, so the
/// harness bounds each run rather than the whole: it fails once no run has ended for [`BOUND`].
fn swept(cap: Capability, runners: Vec<Config>) -> Vec<Run> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let row = rows()
            .into_iter()
            .find(|r| r.cap == cap)
            .expect("a drivable row");
        let ran = tx.clone();
        let runs = sweep(&row, &runners, &|| drop(ran.send(None)));
        let _ = tx.send(Some(runs));
    });
    loop {
        match rx.recv_timeout(BOUND) {
            Ok(Some(runs)) => return runs,
            Ok(None) => {}
            Err(_) => panic!("{}'s sweep: no run ended within {BOUND:?}", cap.name()),
        }
    }
}

/// Run every call shape of every op of `row`, over each of its mints, on the oracle and on each of
/// `runners`: the runs that had an answer to compare. Each run mints the row's powerbox afresh, in
/// the same order, so a real handle baked into a shape names the same grant on every runner. `ran`
/// hears each run end.
fn sweep(row: &Row, runners: &[Config], ran: &dyn Fn()) -> Vec<Run> {
    let mut out = Vec::new();
    for mint in &row.mints {
        let setup = |_: &Module, _: &Embedder| -> (Host, Vec<Value>) {
            let (host, handle, _) = mint();
            (host, vec![Value::I32(handle)])
        };
        for op in ops(row) {
            let mut reached = false;
            for (argc, res) in shapes() {
                let (_, _, real) = mint();
                let real: Vec<(usize, i64)> = real
                    .iter()
                    .filter(|(o, _, _)| *o == op)
                    .map(|(_, idx, v)| (*idx, *v))
                    .collect();
                let m = probe_module_with(row.iface, op, argc, res, &real);
                let shape = format!("argc {argc} -> {res}");
                let call = format!("op {op}");
                reached |= compare(&m, &setup, runners, &call, &shape, ran, &mut out);
            }
            assert!(
                row.ops.is_empty() || reached,
                "{}: op {op} reaches no op at any arity up to {MAX_ARGC} on any runner. Either it \
                 is not an op of this interface (drop it from the row) or it needs a handle this \
                 harness cannot mint (move the row to the undrivable set).",
                row.cap.name(),
            );
        }
        hold(mint, runners, ran, &mut out);
    }
    out
}

/// Hold `mint`'s handle across a freeze it is not called in, on each durable runner among
/// `runners`. A freeze lands at the first safepoint after a call, which a row's own ops may never
/// reach: an empty interface's are all miscalled, and `Exit`'s ends the run. So the run calls a
/// clock it is also granted, and freezes there with the handle in its powerbox.
fn hold(mint: &Mint, runners: &[Config], ran: &dyn Fn(), out: &mut Vec<Run>) {
    let setup = |_: &Module, _: &Embedder| -> (Host, Vec<Value>) {
        let (mut host, _, _) = mint();
        let clock = host.grant_clock();
        (host, vec![Value::I32(clock)])
    };
    let durable: Vec<Config> = runners
        .iter()
        .copied()
        .filter(|c| c.props.has(Property::Durable))
        .collect();
    let m = probe_module_with(temen_ir::cap_id::CLOCK, 0, 0, "i64", &[]);
    compare(
        &m,
        &setup,
        &durable,
        "held",
        "across a clock call",
        ran,
        out,
    );
}

/// Run `m` over the powerbox `setup` builds on the oracle and on each of `runners`, and push the
/// runs that had an answer to compare: whether any did. `ran` hears each run end.
fn compare(
    m: &Module,
    setup: &Setup,
    runners: &[Config],
    call: &str,
    shape: &str,
    ran: &dyn Fn(),
    out: &mut Vec<Run>,
) -> bool {
    let g = Guest {
        module: m,
        setup,
        outside: Outside::Nothing,
    };
    let want = once(Way::OneShot(Executor::TreeWalk), &g, false, None);
    ran();
    let mut reached = false;
    for &config in runners {
        let way = way(config).expect("an axis runs on the run-mode matrix's runners");
        let got = once(way, &g, false, Some(&want));
        ran();
        if miscalled(&want) && miscalled(&got) {
            continue;
        }
        reached = true;
        let verdict = match got {
            End::Declined => Verdict::Declined,
            _ if got == want => Verdict::Matched,
            _ => Verdict::Diverged(want.clone(), got),
        };
        out.push(Run {
            call: call.into(),
            shape: shape.into(),
            config,
            verdict,
        });
    }
    reached
}

/// The cell `runs` fold to, by the run-mode matrix's rule. A row with nothing to compare (no ops,
/// and nothing past its interface answers) has nothing to diverge about.
fn fold(runs: &[&Run]) -> Status {
    let statuses: Vec<Status> = runs
        .iter()
        .map(|r| match r.verdict {
            Verdict::Matched => Status::Full,
            Verdict::Declined => Status::Declines,
            Verdict::Diverged(..) => Status::NotYet,
        })
        .collect();
    fold_status(&statuses).unwrap_or(Status::Full)
}

/// Whether the manifest's `claimed` cell says what runs that fold to `observed` show: the fold
/// itself, or `NotYet` for any fold short of ✅. A known gap may decline or diverge, and its issue
/// tracks it either way, as a run-mode exception's `NotYet` holds any run short of the oracle's
/// answer. A gap that closes folds to ✅ and fails here until its cell says so.
fn says(claimed: Status, observed: Status) -> bool {
    claimed == observed || (claimed == Status::NotYet && observed != Status::Full)
}

/// The runs that did not match, grouped by call, runner and verdict, each with the shapes it took.
fn unmatched(runs: &[&Run]) -> Vec<(String, Vec<String>)> {
    let mut groups: Vec<(String, Vec<String>)> = Vec::new();
    for r in runs.iter().filter(|r| r.verdict != Verdict::Matched) {
        let what = format!("{} on {}: {:?}", r.call, r.config.label(), r.verdict);
        match groups.iter_mut().find(|(w, _)| *w == what) {
            Some((_, shapes)) => shapes.push(r.shape.clone()),
            None => groups.push((what, vec![r.shape.clone()])),
        }
    }
    groups
}

fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// **The pin.** For every drivable capability and every run-driven axis, the manifest's cell says
/// what the capability's runs on that axis's runners fold to. A failure lists every disagreeing
/// cell, with the runs that are not plain matches.
#[test]
fn every_run_driven_cell_matches_its_runs() {
    let runners = all_runners();
    let mut wrong = Vec::new();
    for cap in rows().iter().map(|r| r.cap) {
        let runs = swept(cap, runners.clone());
        for axis in run_axes() {
            let on: Vec<Config> = axis.runners();
            let these: Vec<&Run> = runs.iter().filter(|r| on.contains(&r.config)).collect();
            let observed = fold(&these);
            let claimed = capability_axes(cap)[col(axis)].status;
            if says(claimed, observed) {
                continue;
            }
            let odd: String = unmatched(&these)
                .iter()
                .map(|(what, shapes)| format!("    {what}: {}\n", shapes.join(", ")))
                .collect();
            wrong.push(format!(
                "  {} × {}: the manifest says {claimed:?}, the runs fold to {observed:?}\n{odd}",
                cap.name(),
                axis.short(),
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "the frontier matrix disagrees with the runs:\n{}",
        wrong.join("")
    );
}

/// The rows this file cannot drive, stated rather than left implicit, so the covered set cannot
/// quietly shrink while the axes keep claiming to be run: `JitCode` and `Jit`'s `invoke` need a unit
/// only a running guest can mint, and `Offer` and `LiveImpl` a live peer. Each is `Unaudited` on
/// every run-driven axis but `durability`, and only they are. On `durability` the codec's classifier
/// decides them instead (`frontier_conformance.rs`), which needs a handle, not a running guest.
#[test]
fn the_undrivable_rows_are_exactly_the_ones_needing_a_running_guest_or_a_live_peer() {
    let driven: Vec<Capability> = rows().into_iter().map(|r| r.cap).collect();
    let undriven: Vec<&'static str> = Capability::ALL
        .iter()
        .filter(|c| !driven.contains(c))
        .map(|c| c.name())
        .collect();
    assert_eq!(
        undriven,
        vec!["Jit", "JitCode", "Offer", "LiveImpl"],
        "the set of rows this file cannot drive changed. If a row became drivable, add it to \
         `rows()` rather than leaving the axes claiming coverage they do not have."
    );
    for axis in run_axes().filter(|a| *a != Axis::Durability) {
        for c in Capability::ALL {
            let unaudited = capability_axes(c)[col(axis)].status == Status::Unaudited;
            assert_eq!(
                unaudited,
                undriven.contains(&c.name()),
                "{} × {}: a row is `Unaudited` on a run-driven axis exactly when this file cannot \
                 drive it",
                c.name(),
                axis.short(),
            );
        }
    }
}

/// Non-vacuity for the spawn family: op 17's sweep reads its record and reaches the spawn's
/// admission, rather than agreeing with every runner by faulting early. `-EINVAL` is the answer for
/// a spawn no `Budget` pays for. A zero record pointer would fault in the NULL guard on every
/// runner, which is how the retired op 13's drivers once agreed on a forged module while disagreeing
/// about a real one (#1570).
#[test]
fn the_spawn_sweep_reaches_admission() {
    let row = rows()
        .into_iter()
        .find(|r| r.cap == Capability::Instantiator)
        .expect("the Instantiator row");
    let [mint] = &row.mints[..] else {
        panic!("the Instantiator row has one mint");
    };
    let (_, _, real) = mint();
    let real: Vec<(usize, i64)> = real
        .iter()
        .filter(|(o, _, _)| *o == 17)
        .map(|(_, i, v)| (*i, *v))
        .collect();
    let m = probe_module_with(row.iface, 17, 1, "i32", &real);
    let setup = |_: &Module, _: &Embedder| -> (Host, Vec<Value>) {
        let (host, handle, _) = mint();
        (host, vec![Value::I32(handle)])
    };
    let g = Guest {
        module: &m,
        setup: &setup,
        outside: Outside::Nothing,
    };
    assert_eq!(
        once(Way::OneShot(Executor::Pump), &g, false, None),
        End::Value(-22),
        "the op-17 sweep must get past reading its record into admission"
    );
}

/// The wasm-JIT's part of the `backend` axis, which has no runner yet: it is a leaf accelerator that
/// does not emit `call.cap` at all, so every capability folds to the interpreter underneath it. That
/// is a decline in invariant 9's sense, and `OPS_PARITY.md`'s `call.cap` row is its op-granularity
/// twin; this keeps the claim from going stale if that changes.
#[test]
fn the_wasm_jit_declines_every_capability_call() {
    for row in rows() {
        let op = row.ops.first().copied().unwrap_or(0);
        let m = probe_module_with(row.iface, op, 0, "i64", &[]);
        assert!(
            temen_wasm_jit::compile_module(&m).is_err(),
            "{}: the wasm-JIT emitted a module containing `call.cap`. If it now emits cap calls, \
             the backend axis must run it rather than record the fold",
            row.cap.name(),
        );
    }
}
