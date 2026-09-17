//! **The honesty pin for the frontier matrix's `debugger` column (#1413).**
//!
//! `crates/temen-parity/src/frontier.rs` states, per powerbox capability, whether it is *observable
//! under the debug tier* — INVARIANTS #14's seventh axis, disciplined by #9's observability
//! corollary. This test drives the **real** predicates and fails when the manifest disagrees, the
//! same role `frontier_conformance.rs` plays for `nesting`/`durability` and `conformance.rs` for the
//! op × backend matrix.
//!
//! ## What the predicate actually is
//!
//! Two gates in the tree decide it, and a capability op has to clear **both**:
//!
//! | gate | where | what a failure means |
//! |---|---|---|
//! | the bytecode lowering's `(type_id, op)` table | `bytecode::compile_module_unfused` | the module never compiles for the debug engine — `ScheduledDebugRun::new_with_host` is `None` and the whole run falls back to the tree-walk oracle |
//! | `service_advance`'s decline set | `bytecode::service_advance` | it compiled, but the scheduler seam the op produced is one the debug scheduler does not drive — `SchedStop::Declined` |
//!
//! Neither is keyed by capability, which is why this column could not be filled in when the matrix
//! landed: the first is a table over `(type_id, op)` pairs and the second a match over `Outcome`
//! (scheduler seams), so the capability-level answer only exists **downstream of running the thing**.
//! So this test runs it: for each capability it mints a handle on a bare `Host`, calls each of its
//! declared ops through a generated one-call module under `ScheduledDebugRun`, and folds the
//! per-op verdicts into the cell.
//!
//! The second gate became locatable only once the debugger driver's declines were named rather than
//! caught by a `_` (#1414) — before that a new `Outcome` joined the declined set silently, so a
//! column derived from it would have gone stale without anything going red.
//!
//! ## Why the arity sweep
//!
//! The lowering's arms are arity-guarded (`(INSTANTIATOR, 5) if args.len() >= 5`), so calling an op
//! with too few arguments makes it look unsupported when it is merely miscalled. The question the
//! column asks is "**can** the debug tier run this op", so each op is tried across a range of
//! arities and the **best** verdict wins. Argument *values* are all zero and deliberately so: a
//! `-EINVAL` or a trap is a serviced answer — the op reached its seam and the engine drove it —
//! and only the two gates above produce the other two verdicts.
//!
//! ## The audit rule
//!
//! A row is audited only when **every** declared op was actually driven. `Jit` misses that bar by
//! one op (`invoke` needs a live `JitCode` unit, which only a running guest can mint), so it stays
//! `Unaudited` alongside the rows that need a live peer, rather than being scored on a partial
//! sweep. That is the same discipline the other two columns keep: an unaudited cell is visible and
//! countable; a cell scored on what the test happened to reach is a wish.

mod support;
use support::capability_probe::{probe_module, rows, Row, MAX_ARGC, PROBE_BEYOND};
use temen_interp::bytecode::{SchedStop, ScheduledDebugRun};
use temen_interp::{Host, Trap, Value};
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::Status;

/// What the debug tier did with one capability op.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
enum Verdict {
    /// The generated module did not compile for the debug engine — the lowering's `(type_id, op)`
    /// table rejects it, so the whole run falls back to the oracle.
    NotInSubset,
    /// It compiled, but the seam it produced is in `service_advance`'s decline set.
    Declined,
    /// The debug scheduler drove it. A `-errno` result or a trap counts: the op reached its seam.
    Serviced,
}

/// Run `call.cap iface op` on `handle` with `argc` zero arguments, under the multi-vCPU debug
/// engine. `None` where the call faults for a reason that is not one of the two gates — a wrong
/// signature, or an op that needs a handle this harness cannot mint.
fn drive(host: Host, handle: i32, iface: u32, op: u32, argc: usize) -> Option<Verdict> {
    let m = probe_module(iface, op, argc);
    let Some(mut run) = ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(handle)], host) else {
        return Some(Verdict::NotInSubset);
    };
    let mut fuel = 1_000_000u64;
    match run.run_until_stop(&mut fuel) {
        SchedStop::Declined => Some(Verdict::Declined),
        // A cap fault is the harness miscalling, not an answer: the signature did not match any
        // dispatch arm. The sweep tries other arities.
        SchedStop::Finished(Err(Trap::CapFault)) => None,
        _ => Some(Verdict::Serviced),
    }
}

/// The best verdict `op` reaches across the arity sweep, or `None` where every arity cap-faults.
fn best(mint: &dyn Fn() -> (Host, i32), iface: u32, op: u32) -> Option<Verdict> {
    (0..MAX_ARGC)
        .filter_map(|argc| {
            let (host, h) = mint();
            drive(host, h, iface, op, argc)
        })
        .max()
}

/// Fold a row's per-op verdicts into its cell status. Every op serviced is `Full`; a mix is
/// `Conditional`; nothing serviced is `Declines`.
fn fold(verdicts: &[Verdict]) -> Status {
    let serviced = verdicts.iter().filter(|v| **v == Verdict::Serviced).count();
    match serviced {
        0 => Status::Declines,
        n if n == verdicts.len() => Status::Full,
        _ => Status::Conditional,
    }
}

/// A capability that declares no callable ops: assert nothing on it reaches either gate, which
/// makes its cell `Full` — there is no divergence for the debug tier to have.
fn no_callable_ops_reach_either_gate(row: &Row) -> Status {
    for op in 0..PROBE_BEYOND {
        if let Some(v) = best(row.mint.as_ref(), row.iface, op) {
            assert_eq!(
                v,
                Verdict::Serviced,
                "{}: op {op} gives {v:?}. This row is declared as having no callable ops, so \
                 nothing on it should reach the lowering's veto table or the scheduler's decline \
                 set — it has grown an op, and the cell needs scoring rather than asserting.",
                row.cap.name(),
            );
        }
    }
    Status::Full
}

fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// **The pin.** For every drivable capability, the manifest's `debugger` cell must match what the
/// debug tier actually does with that capability's ops.
#[test]
fn the_debugger_column_matches_what_the_debug_tier_actually_runs() {
    let d = col(Axis::Debugger);
    for row in rows() {
        if row.ops.is_empty() {
            let observed = no_callable_ops_reach_either_gate(&row);
            assert_eq!(
                observed,
                capability_axes(row.cap)[d].status,
                "{}: the frontier manifest disagrees with a capability that has no callable ops",
                row.cap.name(),
            );
            continue;
        }
        let verdicts: Vec<Verdict> = row
            .ops
            .iter()
            .map(|op| {
                best(row.mint.as_ref(), row.iface, *op).unwrap_or_else(|| {
                    panic!(
                        "{}: op {op} cap-faults at every arity up to {MAX_ARGC}. Either it is not \
                         an op of this interface (drop it from the row) or it needs a handle this \
                         harness cannot mint (move the row to the undrivable set) — leaving it here \
                         scores the cell on a sweep that never reached the op.",
                        row.cap.name(),
                    )
                })
            })
            .collect();
        let observed = fold(&verdicts);
        let claimed = capability_axes(row.cap)[d].status;
        let detail: Vec<String> = row
            .ops
            .iter()
            .zip(&verdicts)
            .map(|(op, v)| format!("op{op}={v:?}"))
            .collect();
        assert_eq!(
            observed,
            claimed,
            "{}: the frontier manifest says debugger = {claimed:?}, but the debug tier gives \
             {observed:?}. Per-op: {}. One of the two is wrong — and the manifest is the one nobody \
             runs.",
            row.cap.name(),
            detail.join(" "),
        );
    }
}

/// The rows this test cannot drive, stated rather than left implicit — so the covered set cannot
/// quietly shrink while the column keeps claiming to be conformance-tested.
///
/// `JitCode` and `Jit`'s `invoke` both need a unit minted by `Jit.compile` from guest bytes, which
/// only a running guest can produce; `Offer` and `LiveImpl` need a live peer domain. Filling these
/// in is the remaining work on this column.
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
        "the set of rows this test cannot drive changed. If a row became drivable, add it to \
         `rows()` rather than leaving the column claiming coverage it does not have."
    );

    let d = col(Axis::Debugger);
    for c in Capability::ALL {
        let unaudited = capability_axes(c)[d].status == Status::Unaudited;
        assert_eq!(
            unaudited,
            undriven.contains(&c.name()),
            "{}: a row is `Unaudited` on the debugger axis exactly when this test cannot drive it",
            c.name(),
        );
    }
}
