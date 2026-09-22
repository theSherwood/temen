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

use temen_interp::{cap_id, FreezeScope, Host};

/// Authority over a range covers anything **inside** it — the same containment `AddressSpace` and
/// `Instantiator` sub-ranges use, and what lets one grant over an instantiator's range cover every
/// child carved from it rather than costing a handle-table slot per spawn.
#[test]
fn authority_covers_sub_ranges_not_just_the_exact_range() {
    let mut h = Host::new();
    let _ = h.grant_freeze_authority(FreezeScope::Carve {
        base: 0x1000,
        size: 0x1000,
    });

    assert!(
        h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x1000,
            size: 0x1000
        }),
        "the range itself"
    );
    assert!(
        h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x1400,
            size: 0x400
        }),
        "a carve inside it"
    );
    assert!(
        h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x1fff,
            size: 1
        }),
        "its last byte"
    );

    assert!(
        !h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x0fff,
            size: 1
        }),
        "one byte below"
    );
    assert!(
        !h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x2000,
            size: 1
        }),
        "one byte above"
    );
    assert!(
        !h.holds_freeze_authority(FreezeScope::Carve {
            base: 0x1000,
            size: 0x1001
        }),
        "a range that overhangs the grant is not covered — partial authority is no authority"
    );
}

/// **A domain nobody holds authority over is confidential.** That is the whole of the
/// confidential/ancestor-freezable distinction (R1 §6): it is derived from the absence of a grant,
/// not tracked as a state of its own, so there is nothing to keep in step.
#[test]
fn a_host_with_no_grant_holds_no_authority() {
    let h = Host::new();
    assert!(!h.holds_freeze_authority(FreezeScope::Carve {
        base: 0,
        size: u64::MAX
    }));
    assert!(!h.holds_freeze_authority(FreezeScope::Carve {
        base: 0x1000,
        size: 1
    }));
}

/// **It survives freeze/thaw.** A thawed parent must hold over its thawed child exactly what it held
/// before — otherwise a round trip launders the domain's exposure away and `attest.freeze_exposed`
/// starts lying about a window an ancestor can still read.
#[test]
fn authority_round_trips_through_the_durable_handle_capture() {
    let mut h = Host::new();
    let handle = h.grant_freeze_authority(FreezeScope::Carve {
        base: 0x2000,
        size: 0x1000,
    });

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
        thawed.holds_freeze_authority(FreezeScope::Carve {
            base: 0x2000,
            size: 0x1000
        }),
        "a thawed holder holds what it held before the freeze"
    );
    assert!(
        thawed.holds_freeze_authority(FreezeScope::Carve {
            base: 0x2800,
            size: 0x400
        }),
        "and the containment rule comes back with it"
    );
    let _ = handle;
}

/// **Detached-progeny authority is all-or-nothing, and separate from carve authority.** A detached
/// child owns its own window, so no address range names it; holding authority over *ranges* therefore
/// says nothing about whether the holder may freeze its detached children, and vice versa.
#[test]
fn detached_progeny_authority_is_its_own_scope() {
    let mut carves = Host::new();
    let _ = carves.grant_freeze_authority(FreezeScope::Carve {
        base: 0,
        size: u64::MAX,
    });
    assert!(
        !carves.holds_freeze_authority(FreezeScope::DetachedProgeny),
        "authority over every address in the window still does not reach a detached child — \
         it lives outside that window entirely, which is what detached means"
    );

    let mut detached = Host::new();
    let _ = detached.grant_freeze_authority(FreezeScope::DetachedProgeny);
    assert!(detached.holds_freeze_authority(FreezeScope::DetachedProgeny));
    assert!(
        !detached.holds_freeze_authority(FreezeScope::Carve { base: 0, size: 1 }),
        "and the reverse: detached-progeny authority confers nothing over the holder's own carves"
    );
}

/// **A §14 nested spawn mints carve authority, never detached-progeny authority.**
///
/// This is the property that keeps detachment meaningful. A nested parent already reads its child's
/// carve, so self-granting authority there documents a fact. A detached child's window is *not*
/// readable by its parent, so authority over one is a real new power — and a parent able to mint it
/// for itself at spawn would dissolve the isolation it just asked for. It has to come from above.
///
/// Pinned on `Host` directly rather than through a spawn, because the guarantee is about what the
/// grant paths *can* produce: nothing in the host mints `DetachedProgeny`, and no guest-reachable op
/// mints a capability at all, so the only way in is an embedder grant or a re-grant from an ancestor.
#[test]
fn nothing_self_mints_detached_progeny_authority() {
    let mut h = Host::new();
    // The shape the §14 nested spawn path mints (see `child_attestation`'s call site).
    let _ = h.grant_freeze_authority(FreezeScope::Carve {
        base: 0,
        size: 1 << 20,
    });
    assert!(
        !h.holds_freeze_authority(FreezeScope::DetachedProgeny),
        "a spawn-minted carve grant must not confer authority over detached children"
    );
}

/// Detached-progeny authority survives freeze/thaw like the carve scope does — otherwise a round trip
/// would silently revoke it, and a thawed parent would lose the ability to freeze children it could
/// freeze a moment earlier.
#[test]
fn detached_progeny_authority_round_trips() {
    let mut h = Host::new();
    let _ = h.grant_freeze_authority(FreezeScope::DetachedProgeny);

    let captured = h
        .capture_durable_handles()
        .expect("detached-progeny authority is durable");
    assert!(captured
        .iter()
        .any(|d| d.type_id == cap_id::FREEZE_AUTHORITY));

    let mut thawed = Host::new();
    thawed.restore_durable_handles(&captured);
    assert!(thawed.holds_freeze_authority(FreezeScope::DetachedProgeny));
}
