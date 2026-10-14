//! **Freeze authority as a capability** (#1440, INVARIANTS #14 R1, PROCESS.md O14).
//!
//! A snapshot is a complete read of a window, so "may an ancestor freeze me?" is an exposure question
//! and `self.attest`'s `freeze_exposed` bit is meant to answer it truthfully. It could not: authority
//! was implicit in nesting, so the bit reported a *placement* — conservative rather than true, and
//! unaskable, which is why the op-15 gate could only refuse outright instead of refusing *unless a
//! holder is registered*.
//!
//! Held as a capability it attenuates down the grant graph like any other authority (INVARIANTS #3),
//! survives freeze/thaw through the value-typed re-grant path, and can be asked. These pin the three
//! properties that makes true.

use temen_interp::{cap_id, Host, Lease, WINDOW_RUNS};

/// A 64 KiB detached window's lease on `budget`.
fn window(budget: i32) -> Lease {
    Lease {
        budget,
        bytes: 1 << 16,
        runs: WINDOW_RUNS,
    }
}

/// **A domain nobody holds authority over is confidential.** That is the whole of the
/// confidential/ancestor-freezable distinction (R1 §6): it is derived from the absence of a grant,
/// not tracked as a state of its own, so there is nothing to keep in step.
#[test]
fn a_host_with_no_grant_holds_no_authority() {
    assert!(!Host::new().holds_freeze_authority());
}

/// **It survives freeze/thaw.** A thawed parent must hold over its thawed children exactly what it
/// held before — otherwise a round trip launders their exposure away and `attest.freeze_exposed`
/// starts lying, and a thawed parent loses the ability to freeze children it could freeze a moment
/// earlier.
#[test]
fn authority_round_trips_through_the_durable_handle_capture() {
    let mut h = Host::new();
    let _ = h.grant_freeze_authority();

    let captured = h
        .capture_durable_handles()
        .expect("freeze authority is a durable binding");
    assert!(
        captured
            .iter()
            .any(|d| d.type_id == cap_id::FREEZE_AUTHORITY),
        "the capture must carry the authority, not drop it"
    );

    let mut thawed = Host::new();
    thawed.restore_durable_handles(&captured);
    assert!(
        thawed.holds_freeze_authority(),
        "a thawed holder holds what it held before the freeze"
    );
}

/// #2018 — **an ancestor hands freeze authority down by grant**, the one way INVARIANTS' R1 ruling
/// lets it reach a descendant. A durable detached child has no other source of it, and without it the
/// shared admission refuses every detached spawn of its own, so a durable tree could never be deeper
/// than one level. The parent keeps its own.
#[test]
fn freeze_authority_is_handed_down_by_grant() {
    let mut parent = Host::new();
    parent.set_durable(true);
    let authority = parent.grant_freeze_authority();
    let budget = parent.grant_budget(-1, 1 << 20, -1);
    let (mut child, _, _) = parent
        .spawn_detached_child(
            &[("freeze".to_string(), authority)],
            1 << 16,
            window(budget),
            -1,
        )
        .expect("freeze authority re-grants");
    assert!(child.holds_freeze_authority());
    assert!(parent.holds_freeze_authority(), "the parent keeps its own");
    let own = child
        .resolve_cap_name("budget")
        .expect("the child's budget");
    assert!(
        child.admit_detached_spawn(window(own)).is_some(),
        "so the durable child may spawn detached children of its own"
    );

    let (mut bare, _, _) = parent
        .spawn_detached_child(&[], 1 << 16, window(budget), -1)
        .expect("spawn");
    let own = bare.resolve_cap_name("budget").expect("the child's budget");
    assert!(
        bare.admit_detached_spawn(window(own)).is_none(),
        "without the grant it holds none, and its detached spawn is refused"
    );
}
