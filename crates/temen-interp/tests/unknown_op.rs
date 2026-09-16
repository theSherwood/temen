//! **What a capability answers for an op it does not have** (#1515, found reconstructing the op
//! registry for #1416).
//!
//! Every tier routes `call.cap` through the one `Host::cap_dispatch_slots`, so this drives that
//! entry directly: what it pins holds on the oracle, the bytecode engine and both JITs by
//! construction (the `fast_cap_resolver` clock path gates on `(CLOCK, 0)` explicitly).
//!
//! The answer is **derived from the interface's seeded shape**: `op >= shape.len()` is a
//! `CapFault`. `op` is an immediate in the instruction, so an unknown op is a property of the
//! program, never of a runtime request — the "typing violation on a live handle" INVARIANTS #5
//! puts in the trap column, and the answer the import layer already gives an out-of-range consumer
//! op. One check, one place; an interface gets it the moment it is seeded.
//!
//! Interfaces **without** a seeded shape keep their old per-arm answer, and the last test pins that
//! honestly rather than hiding it: `Clock` still ignores `op` entirely. That is the visible,
//! countable form of "not fixed yet" — the same discipline as an `Unaudited` frontier cell.

use temen_interp::{cap_id, Host, StreamRole, Trap};

/// The op count of a seeded shape, or `None` where the built-in has none.
fn ops(id: u32) -> Option<usize> {
    temen_interp::builtin_iface_shape(id).map(|s| s.len())
}

#[test]
fn an_op_past_a_seeded_shape_is_a_cap_fault_on_the_one_shared_dispatch() {
    let mut h = Host::new();
    let stream = h.grant_stream(StreamRole::Out);
    let exit = h.grant_exit();
    for (name, id, handle) in [
        ("Stream", cap_id::STREAM, stream),
        ("Exit", cap_id::EXIT, exit),
    ] {
        let n = ops(id).unwrap_or_else(|| panic!("{name} is seeded")) as u32;
        for op in [n, n + 1, 999] {
            assert!(
                matches!(
                    h.cap_dispatch_slots(id, op, handle, &[7], None),
                    Err(Trap::CapFault)
                ),
                "{name}: op {op} is past its {n}-op shape and must CapFault",
            );
        }
    }
}

/// The fail-open this closes: before the shape-derived check, `Binding::Exit` never looked at `op`,
/// so `call.cap EXIT 999 (7)` exited the domain with code 7. An `Exit` handle now has exactly one
/// op, and only that one terminates.
#[test]
fn exit_terminates_only_on_its_one_op() {
    let mut h = Host::new();
    let exit = h.grant_exit();
    assert!(matches!(
        h.cap_dispatch_slots(cap_id::EXIT, 0, exit, &[7], None),
        Err(Trap::Exit(7))
    ));
    assert!(matches!(
        h.cap_dispatch_slots(cap_id::EXIT, 1, exit, &[7], None),
        Err(Trap::CapFault)
    ));
}

/// The seeded set is exactly the built-ins whose signature convention is pinned (#1515). A new
/// entry here is a decision that should be visible in the diff, not absorbed.
#[test]
fn the_seeded_built_ins_are_exactly_the_ones_with_a_pinned_convention() {
    let seeded: Vec<&str> = [
        ("Stream", cap_id::STREAM),
        ("Exit", cap_id::EXIT),
        ("Clock", cap_id::CLOCK),
        ("SharedRegion", cap_id::SHARED_REGION),
        ("AddressSpace", cap_id::ADDRESS_SPACE),
        ("Instantiator", cap_id::INSTANTIATOR),
        ("ModuleLoader", cap_id::MODULE_LOADER),
        ("Module", cap_id::MODULE),
        ("Blocking", cap_id::BLOCKING),
        ("Jit", cap_id::JIT),
        ("JitCode", cap_id::JIT_CODE),
        ("HostProc", cap_id::HOST_PROC),
        ("Budget", cap_id::BUDGET),
    ]
    .into_iter()
    .filter(|(_, id)| ops(*id).is_some())
    .map(|(n, _)| n)
    .collect();
    assert_eq!(
        seeded,
        vec!["Stream", "Exit"],
        "a built-in was seeded (or unseeded): update #1515's slice list and this pin together"
    );
}

/// **Not fixed yet, pinned so it cannot be forgotten.** `Clock` has no seeded shape (its `now` is
/// called both `(i32) -> (i64)` and `() -> (i64)` in the tree — #1515 decision 1), so nothing
/// derives an unknown-op answer for it and `Binding::Clock` still ignores `op`: op 999 reads the
/// clock. When `Clock` is seeded this test goes red, which is the point — replace it with the
/// seeded assertion above.
#[test]
fn clock_still_ignores_its_op_number_until_it_is_seeded() {
    assert!(
        ops(cap_id::CLOCK).is_none(),
        "Clock got seeded — retire this pin"
    );
    let mut h = Host::new();
    let clock = h.grant_clock();
    let a = h
        .cap_dispatch_slots(cap_id::CLOCK, 0, clock, &[], None)
        .unwrap();
    let b = h
        .cap_dispatch_slots(cap_id::CLOCK, 999, clock, &[], None)
        .unwrap();
    assert_eq!(
        b[0],
        a[0] + 1,
        "op 999 is served as `now` (the deterministic clock ticked once)"
    );
}

/// The other party's route (#1515): a **host** binds an import slot to `(EXIT, 5)`. Nothing the
/// guest wrote is wrong, so this cannot be caught at verify time; it is caught at the one sink every
/// binding passes through (`set_import_bindings`), which marks the slot unbound. A call through it
/// then gets exactly what a never-attached rebindable slot gets — `CapFault`, fail-closed — instead
/// of the op-0 behaviour (`Exit` would have terminated the domain).
#[test]
fn a_host_binding_past_a_seeded_shape_leaves_the_slot_unbound() {
    use temen_interp::BoundImport;
    let mut h = Host::new();
    let exit = h.grant_exit();
    h.set_import_bindings(vec![
        BoundImport::required(cap_id::EXIT, 0, exit), // slot 0: the real op
        BoundImport::required(cap_id::EXIT, 5, exit), // slot 1: an op Exit does not have
    ]);
    // `call.import` dispatches as `(CAP_IMPORT_TYPE_ID, slot | consumer_op << 16)`; a flat binding
    // takes consumer op 0.
    assert!(matches!(
        h.cap_dispatch_slots(temen_ir::CAP_IMPORT_TYPE_ID, 0, 0, &[3], None),
        Err(Trap::Exit(3))
    ));
    assert!(
        matches!(
            h.cap_dispatch_slots(temen_ir::CAP_IMPORT_TYPE_ID, 1, 0, &[3], None),
            Err(Trap::CapFault)
        ),
        "a slot bound past the interface's shape is unbound, never a live exit"
    );
}
