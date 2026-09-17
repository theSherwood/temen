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

use temen_interp::{cap_id, Host};

/// Authority over a range covers anything **inside** it — the same containment `AddressSpace` and
/// `Instantiator` sub-ranges use, and what lets one grant over an instantiator's range cover every
/// child carved from it rather than costing a handle-table slot per spawn.
#[test]
fn authority_covers_sub_ranges_not_just_the_exact_range() {
    let mut h = Host::new();
    let _ = h.grant_freeze_authority(0x1000, 0x1000);

    assert!(
        h.holds_freeze_authority_over(0x1000, 0x1000),
        "the range itself"
    );
    assert!(
        h.holds_freeze_authority_over(0x1400, 0x400),
        "a carve inside it"
    );
    assert!(h.holds_freeze_authority_over(0x1fff, 1), "its last byte");

    assert!(!h.holds_freeze_authority_over(0x0fff, 1), "one byte below");
    assert!(!h.holds_freeze_authority_over(0x2000, 1), "one byte above");
    assert!(
        !h.holds_freeze_authority_over(0x1000, 0x1001),
        "a range that overhangs the grant is not covered — partial authority is no authority"
    );
}

/// **A domain nobody holds authority over is confidential.** That is the whole of the
/// confidential/ancestor-freezable distinction (R1 §6): it is derived from the absence of a grant,
/// not tracked as a state of its own, so there is nothing to keep in step.
#[test]
fn a_host_with_no_grant_holds_no_authority() {
    let h = Host::new();
    assert!(!h.holds_freeze_authority_over(0, u64::MAX));
    assert!(!h.holds_freeze_authority_over(0x1000, 1));
}

/// **It survives freeze/thaw.** A thawed parent must hold over its thawed child exactly what it held
/// before — otherwise a round trip launders the domain's exposure away and `attest.freeze_exposed`
/// starts lying about a window an ancestor can still read.
#[test]
fn authority_round_trips_through_the_durable_handle_capture() {
    let mut h = Host::new();
    let handle = h.grant_freeze_authority(0x2000, 0x1000);

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
        thawed.holds_freeze_authority_over(0x2000, 0x1000),
        "a thawed holder holds what it held before the freeze"
    );
    assert!(
        thawed.holds_freeze_authority_over(0x2800, 0x400),
        "and the containment rule comes back with it"
    );
    let _ = handle;
}
