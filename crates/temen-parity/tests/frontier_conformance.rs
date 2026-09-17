//! **The honesty pin for the frontier matrix (#1413).**
//!
//! `crates/temen-parity/src/frontier.rs` states how each powerbox capability behaves on each
//! INVARIANTS #14 axis. This drives the **real** predicates and fails when the manifest disagrees
//! with them — the same role `conformance.rs` plays for the op × backend matrix.
//!
//! Without this the manifest is a wish: a hand-written table that looks authoritative, agrees with
//! nothing, and goes stale the first time someone changes `can_regrant`. That is precisely the shape
//! #1412 found three times over — three tests, in three crates, each individually coherent and each
//! asserting a divergence the invariants forbid, all green.
//!
//! Two axes are checked here because two axes are decided by an explicit, wildcard-free predicate in
//! the tree:
//!
//! | axis | predicate |
//! |---|---|
//! | nesting | `Host::can_regrant` (via `Host::regrant_into_child`, its only public route) |
//! | durability | the `NonDurableKind` match in `Host::capture_durable_handles` |
//!
//! The other five state the manifest's belief and are rendered `Unaudited` until their predicate is
//! locatable. That is deliberate: an unaudited cell is visible and countable, and this test asserts
//! the count only moves in one direction.

use temen_interp::{Host, NonDurableKind, StreamRole};
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::Status;

/// Index of an axis in the `[Cell; 7]` a row carries.
fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// Grant one handle of each capability kind onto a fresh host, where the kind can be minted without
/// a running guest. Returns `(capability, handle)` pairs.
///
/// Kinds needing a live peer (`Offer`, `LiveImpl`) or an embedder closure (`HostProc`) are absent —
/// their rows are checked by the tests that already exercise them (`regrant_into_child_carries_a_
/// forkable_host_proc_sharing_state`, the §3.6 offer suite). This test covers what it can actually
/// mint, and `every_mintable_capability_is_covered` below makes that set explicit rather than
/// silently partial.
fn mintable(host: &mut Host, m: &temen_ir::Module) -> Vec<(Capability, i32)> {
    vec![
        (Capability::Stream, host.grant_stream(StreamRole::Out)),
        (Capability::Exit, host.grant_exit()),
        (Capability::Clock, host.grant_clock()),
        (Capability::SharedRegion, host.grant_shared_region(4096)),
        (
            Capability::AddressSpace,
            host.grant_address_space(0, 1 << 16),
        ),
        (
            Capability::Instantiator,
            host.grant_instantiator(0, 1 << 16),
        ),
        (Capability::Budget, host.grant_budget(0, 1 << 16, 0)),
        (Capability::Module, host.grant_module(m)),
        (Capability::ModuleLoader, host.grant_module_loader()),
        (Capability::Jit, host.grant_jit(None)),
    ]
}

fn a_module() -> temen_ir::Module {
    let m = temen_text::parse_module(
        "memory 15\nfunc (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n",
    )
    .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// **Nesting.** For every mintable capability, the manifest's `nesting` cell must match what the
/// engine actually lets a §14 child be granted.
///
/// Driven through `Host::spawn_granted_child` — the public route, and the one the JIT backend uses
/// to build the same child powerbox host-side. It gates on `can_regrant` before building anything,
/// so `None` *is* "a child cannot hold this".
#[test]
fn the_nesting_column_matches_what_a_child_can_actually_be_granted() {
    let m = a_module();
    let n = col(Axis::Nesting);
    let mut host = Host::new();
    let pairs = mintable(&mut host, &m);

    for (cap, h) in pairs {
        let crossed = host.spawn_granted_child(h, 1 << 12).is_some();
        let claimed = capability_axes(cap)[n].status;
        let expected = match claimed {
            Status::Full => true,
            Status::Declines => false,
            // A conditional row (`HostProc`: only a *forkable* one crosses) is not mintable here, so
            // reaching this arm means a row changed classification without its checker.
            other => panic!(
                "{}: nesting cell is {other:?}, which this test cannot check — either add a checker \
                 for it or move the row out of `mintable`",
                cap.name()
            ),
        };
        assert_eq!(
            crossed, expected,
            "{}: the frontier manifest says nesting = {claimed:?}, but the engine {} a child holding \
             it. One of the two is wrong — and the manifest is the one nobody runs.",
            cap.name(),
            if crossed { "allowed" } else { "refused" },
        );
    }
}

/// **Durability.** The manifest's `durability` cell must match the `NonDurableKind` classifier: a
/// `Full` cell means a freeze carries the handle, a `Declines` cell means `drain_non_durable` takes
/// it.
///
/// Compared by *kind* rather than by slot: the kind is the classifier's own output, so a mismatch
/// names the binding the classifier disagreed about rather than a table index.
#[test]
fn the_durability_column_matches_the_durable_capture_classifier() {
    let m = a_module();
    let d = col(Axis::Durability);
    let mut host = Host::new();
    let pairs = mintable(&mut host, &m);
    let drained: Vec<NonDurableKind> = host
        .drain_non_durable()
        .into_iter()
        .map(|h| h.kind)
        .collect();

    for (cap, _) in pairs {
        let claimed = capability_axes(cap)[d].status;
        let kind = non_durable_kind_of(cap);
        let was_drained = kind.is_some_and(|k| drained.contains(&k));
        let expected_durable = match claimed {
            Status::Full => true,
            Status::Declines => false,
            other => panic!(
                "{}: durability cell is {other:?}, uncheckable here",
                cap.name()
            ),
        };
        assert_eq!(
            !was_drained,
            expected_durable,
            "{}: the frontier manifest says durability = {claimed:?}, but the capture classifier {} \
             it. Drained kinds: {drained:?}",
            cap.name(),
            if was_drained { "drained" } else { "kept" },
        );
    }
}

/// The `NonDurableKind` a capability is drained as, or `None` where it is durable (value-typed) and
/// so never appears in a drain. Exhaustive, no wildcard: a new capability must state which it is.
fn non_durable_kind_of(c: Capability) -> Option<NonDurableKind> {
    Some(match c {
        Capability::Stream
        | Capability::Exit
        | Capability::Clock
        | Capability::AddressSpace
        | Capability::Instantiator
        | Capability::Jit
        | Capability::JitCode
        // #1502: a Budget's remaining quotas ride the artifact verbatim.
        | Capability::Budget => return None,
        Capability::SharedRegion => NonDurableKind::SharedRegion,
        Capability::Module => NonDurableKind::Module,
        Capability::ModuleLoader => NonDurableKind::ModuleLoader,
        Capability::Blocking => NonDurableKind::Blocking,
        Capability::HostProc => NonDurableKind::HostProc,
        Capability::Offer => NonDurableKind::Offer,
        Capability::LiveImpl => NonDurableKind::LiveImpl,
        Capability::PipeEnd => NonDurableKind::Pipe,
    })
}

/// Every capability this test *can* mint must be in `mintable` — so the coverage set cannot quietly
/// shrink while the matrix keeps claiming to be conformance-tested.
#[test]
fn the_uncheckable_rows_are_exactly_the_ones_that_need_a_live_peer_or_a_host_closure() {
    let m = a_module();
    let mut host = Host::new();
    let covered: Vec<Capability> = mintable(&mut host, &m)
        .into_iter()
        .map(|(c, _)| c)
        .collect();
    let uncovered: Vec<&'static str> = Capability::ALL
        .iter()
        .filter(|c| !covered.contains(c))
        .map(|c| c.name())
        .collect();
    assert_eq!(
        uncovered,
        vec!["PipeEnd", "JitCode", "Blocking", "HostProc", "Offer", "LiveImpl"],
        "the set of rows this test cannot mint changed. Each one needs a live peer, an embedder \
         closure, or a running guest — if a row became mintable, add it to `mintable` rather than \
         leaving the matrix claiming coverage it does not have."
    );
}

/// Coverage may only improve. A cell that was audited must not silently revert to `Unaudited`, and
/// the two axes whose predicate classifies *every* row must stay fully audited.
///
/// The third conformance-tested axis, `debugger`, is not in that loop on purpose: its predicate is
/// reached by running the capability's ops, and four rows hold ops only a running guest or a live
/// peer can reach. `tests/debugger_conformance.rs` pins which four, so those cells cannot quietly
/// spread — the count floor below is what stops the column from emptying out.
#[test]
fn audited_coverage_does_not_regress() {
    let (n, d) = (col(Axis::Nesting), col(Axis::Durability));
    for c in Capability::ALL {
        let cells = capability_axes(c);
        for (axis, i) in [(Axis::Nesting, n), (Axis::Durability, d)] {
            assert_ne!(
                cells[i].status,
                Status::Unaudited,
                "{}: the `{}` axis is conformance-tested, so every row must be classified on it",
                c.name(),
                axis.short(),
            );
        }
    }

    let audited = Capability::ALL
        .iter()
        .flat_map(|c| capability_axes(*c))
        .filter(|cell| cell.status != Status::Unaudited)
        .count();
    assert!(
        audited >= 56,
        "audited cell count fell to {audited}; it was 32 when the matrix landed, 44 once the \
         `debugger` column was driven, and 56 once `concurrency` was. Filling axes in is the work \
         (#1413) — emptying them is a regression."
    );
}
