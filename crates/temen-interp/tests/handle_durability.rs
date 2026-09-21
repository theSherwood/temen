//! Handle-table durability classification + `(slot, generation)` pinning (DURABILITY.md
//! §12.5). These exercise the `Host` primitives the snapshot codec builds on:
//!
//! * `capture_durable_handles` — the live table classified into the re-grantable §12.5 set,
//!   or a clean refusal if any live slot is non-durable (freeze is all-or-nothing);
//! * `restore_durable_handles` / `grant_at` — reinstating that set into a fresh table while
//!   pinning each `(slot, generation)`, so a guest-held handle value (`(generation << 8) |
//!   slot`) still resolves after restore.
//!
//! The invariant tested is structural: a handle value is a pure function of `(generation,
//! slot)` and resolve re-checks `type_id` + `generation`, so equality of the captured set
//! across a restore ⇒ every guest-held handle stays valid. (The full freeze→serialize→
//! restore→thaw run lands with the snapshot-codec slice that wires this to the window image.)

use temen_interp::{cap_id, DurableBinding, DurableHandle, Host, NonDurableKind, StreamRole};

/// Grant a spread of durable bindings, capture, restore into a fresh table, and confirm the
/// captured set is byte-for-byte identical — slot, generation, type_id, and binding all pinned.
/// What a registrar was asked for: `(name, captured state)` per call.
type CapLog = std::sync::Arc<std::sync::Mutex<Vec<(String, Vec<u8>)>>>;

#[test]
fn durable_handles_round_trip_through_capture_restore() {
    let mut a = Host::new();
    a.grant_clock();
    a.grant_stream(StreamRole::Out);
    a.grant_memory();
    a.grant_exit();
    a.grant_address_space(0x2000, 0x1000);
    a.grant_instantiator(0x0, 0x4000);

    let captured = a
        .capture_durable_handles()
        .expect("every binding is durable");
    assert_eq!(captured.len(), 6, "all six live slots captured");
    // Ascending slot order, contiguous from 0 (grants fill the first free slots).
    assert_eq!(captured[0].slot, 0);
    assert!(
        captured.windows(2).all(|w| w[0].slot < w[1].slot),
        "ascending slot order"
    );
    // Value-typed bindings survive verbatim.
    assert_eq!(captured[0].binding, DurableBinding::Clock);
    assert_eq!(captured[1].binding, DurableBinding::Stream(StreamRole::Out));
    assert!(captured.iter().any(|h| h.binding
        == DurableBinding::AddressSpace {
            base: 0x2000,
            size: 0x1000
        }));

    let mut b = Host::new();
    b.restore_durable_handles(&captured);
    assert_eq!(
        b.capture_durable_handles().unwrap(),
        captured,
        "restore reinstates the exact (slot, generation, type_id, binding) set"
    );
}

/// A fresh table starts every slot at generation 0; restore must pin the *captured*
/// generation, not whatever the destination table happens to hold. Bump slot 0's generation
/// via close+re-grant so the distinction is observable.
#[test]
fn restore_pins_generation_not_destination_default() {
    let mut a = Host::new();
    let h0 = a.grant_clock(); // slot 0, generation 1
    a.close(h0);
    a.grant_clock(); // slot 0 again, generation 2 (close kept the generation)

    let captured = a.capture_durable_handles().unwrap();
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].slot, 0);
    assert_eq!(
        captured[0].generation, 2,
        "re-grant after close advanced the generation"
    );

    let mut b = Host::new(); // slot 0 is generation 0 here
    b.restore_durable_handles(&captured);
    let restored = b.capture_durable_handles().unwrap();
    assert_eq!(
        restored, captured,
        "generation 2 pinned, not reset to the fresh table's 0"
    );
}

/// A live non-durable handle (here a `Blocking`, which carries out-of-line host state) makes
/// the table non-snapshottable: capture refuses, naming the offending slot, rather than
/// dropping the authority.
#[test]
fn capture_refuses_a_non_durable_handle() {
    let mut a = Host::new();
    a.grant_clock(); // slot 0, durable
    a.grant_blocking(std::time::Duration::ZERO, None); // slot 1, NOT durable (out-of-line index)

    let err = a
        .capture_durable_handles()
        .expect_err("a blocking handle blocks the snapshot");
    assert_eq!(err.slot, 1);
    assert_eq!(err.kind, NonDurableKind::Blocking);
}

/// Draining the non-durable handles turns a capture refusal into a successful one: the out-of-line
/// bindings (a `Blocking` and a `HostProc`) are closed, the durable `Clock` is kept, and
/// `capture_durable_handles` then succeeds. The drained set comes back in ascending slot order so the
/// embedder can audit the relinquished authority (DURABILITY.md §12.5 handle hardening).
#[test]
fn drain_non_durable_makes_a_domain_snapshottable() {
    let mut a = Host::new();
    a.grant_clock(); // slot 0 — durable
    a.grant_blocking(std::time::Duration::ZERO, None); // slot 1 — non-durable
    a.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![0]))); // slot 2 — non-durable

    assert!(
        a.capture_durable_handles().is_err(),
        "a live non-durable handle blocks the snapshot"
    );

    let drained = a.drain_non_durable();
    assert_eq!(
        drained.len(),
        2,
        "the blocking handle and the host_proc were drained"
    );
    assert_eq!(
        (drained[0].slot, drained[0].kind),
        (1, NonDurableKind::Blocking)
    );
    assert_eq!(
        (drained[1].slot, drained[1].kind),
        (2, NonDurableKind::HostProc)
    );

    let captured = a
        .capture_durable_handles()
        .expect("a drained domain is snapshottable");
    assert_eq!(
        captured.len(),
        1,
        "only the durable Clock survives the drain"
    );
    assert_eq!(captured[0].slot, 0);
    assert_eq!(captured[0].binding, DurableBinding::Clock);
}

/// A drained handle's value is a dead generation: a `call.cap` on it answers the probeable
/// revocation errno (I41 — the drain IS a revocation of a once-valid handle), never authority
/// into the freed slot. The durable handles the drain left alone still resolve.
#[test]
fn drain_non_durable_kills_stale_handle_values() {
    let mut a = Host::new();
    a.grant_clock(); // durable — kept
    let ring = a.grant_blocking(std::time::Duration::ZERO, None); // non-durable — drained

    let drained = a.drain_non_durable();
    assert_eq!(drained.len(), 1, "only the blocking handle drained");

    // The drained handle completes with `-EBADF` at the use site (freed slot ⇒ resolve fails
    // before the op runs; the once-issued generation makes it the revocation errno, not a trap).
    let r = a.cap_dispatch_slots(cap_id::BLOCKING, 0, ring, &[], None);
    assert!(
        matches!(&r, Ok(v) if v.as_slice() == [-9]),
        "a drained handle is a revoked-once-valid generation, got {r:?}"
    );
    // The durable Clock is untouched.
    assert!(a
        .capture_durable_handles()
        .unwrap()
        .iter()
        .any(|h| h.binding == DurableBinding::Clock));
}

/// On an all-durable table draining is a no-op: nothing closes and the captured set is unchanged.
#[test]
fn drain_non_durable_is_a_noop_when_all_durable() {
    let mut a = Host::new();
    a.grant_clock();
    a.grant_memory();
    let before = a.capture_durable_handles().unwrap();

    assert!(
        a.drain_non_durable().is_empty(),
        "nothing non-durable to drain"
    );
    assert_eq!(
        a.capture_durable_handles().unwrap(),
        before,
        "the durable table is unchanged by a no-op drain"
    );
}

/// An empty table captures to an empty set; capacity is the table size, so the codec can
/// bounds-check a captured slot before restore.
#[test]
fn empty_table_captures_empty_and_capacity_is_table_size() {
    let a = Host::new();
    assert_eq!(
        a.capture_durable_handles().unwrap(),
        Vec::<DurableHandle>::new()
    );
    assert_eq!(Host::handle_capacity(), 256);
}

// ---- #1455: named host capabilities ---------------------------------------------------------
//
// A `HostProc` is an opaque closure: its code address is process-local and its captured state is the
// provider's, so neither can ride an artifact — which is why *any* live one used to make the whole
// freeze refuse, i.e. every capability-using guest. What can ride is the **name** the grant was
// registered under, because the reference powerboxes grant deterministically by name. These pin that
// the name is what makes the difference, that the provider's own state travels with it, and — the
// part that matters for authority — that the thawing embedder, not the artifact, decides what gets
// granted.

/// A capability with a registered name captures as `Named`; the same capability without one still
/// refuses. The name is the whole of the difference.
#[test]
fn a_named_host_cap_is_durable_and_an_unnamed_one_is_not() {
    let mut named = Host::new();
    let h = named.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![7])));
    named.register_cap_name("fs", h);
    let captured = named
        .capture_durable_handles()
        .expect("a named host capability is durable");
    assert_eq!(captured.len(), 1);
    assert_eq!(captured[0].binding, DurableBinding::Named { idx: 0 });

    let mut unnamed = Host::new();
    unnamed.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![7])));
    let err = unnamed
        .capture_durable_handles()
        .expect_err("an unnamed host capability has no reconstruction rule");
    assert_eq!(err.kind, NonDurableKind::HostProc);
}

/// The out-of-line half: the name and the provider's own state, positional over `host_procs` so a
/// captured `Named { idx }` re-resolves. A capability that declared no state captures an empty one.
#[test]
fn capture_named_carries_the_providers_state() {
    let cursors = std::sync::Arc::new(std::sync::Mutex::new(vec![3u8, 1, 4]));
    let mut a = Host::new();

    let stateless = a.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![0])));
    a.register_cap_name("display", stateless);

    let stateful = a.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![0])));
    a.register_cap_name("fs", stateful);
    let c = std::sync::Arc::clone(&cursors);
    a.set_cap_state_capture(stateful, Box::new(move || c.lock().unwrap().clone()));

    // State is read at capture time, not at registration time.
    cursors.lock().unwrap().push(1);

    let named = a.capture_durable_named();
    assert_eq!(named.len(), 2, "positional over host_procs");
    assert_eq!(named[0].as_ref().unwrap().name, "display");
    assert!(named[0].as_ref().unwrap().state.is_empty(), "stateless");
    assert_eq!(named[1].as_ref().unwrap().name, "fs");
    assert_eq!(named[1].as_ref().unwrap().state, vec![3, 1, 4, 1]);
}

/// The full re-grant: a fresh host with a registrar rebuilds the capabilities by name, re-seeded from
/// the captured state, and the restored handle still resolves *and dispatches* to them.
#[test]
fn a_registrar_re_grants_named_caps_and_the_handles_still_dispatch() {
    let mut a = Host::new();
    let h = a.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![1])));
    a.register_cap_name("fs", h);
    a.set_cap_state_capture(h, Box::new(|| vec![42]));

    let handles = a.capture_durable_handles().expect("named ⇒ durable");
    let named = a.capture_durable_named();

    let mut b = Host::new();
    let seen: CapLog = Default::default();
    let log = std::sync::Arc::clone(&seen);
    b.set_named_cap_registrar(Box::new(move |name, state| {
        log.lock().unwrap().push((name.to_string(), state.to_vec()));
        // Re-seed the fresh handler from the captured state — this is the provider reading back
        // exactly the bytes it wrote at freeze.
        let answer = state.first().copied().unwrap_or(0) as i64;
        Some(Box::new(move |_op: u32, _args: &[i64], _mem, _| {
            Ok(vec![answer])
        }))
    }));
    b.restore_durable_named(&named)
        .expect("the registrar serves `fs`");
    b.restore_durable_handles(&handles);

    assert_eq!(
        &*seen.lock().unwrap(),
        &[("fs".to_string(), vec![42u8])],
        "the registrar saw the captured name and state"
    );
    // The guest's handle value survives the restore and reaches the re-granted handler.
    assert_eq!(
        b.cap_dispatch_slots(cap_id::HOST_PROC, 0, h, &[], None),
        Ok(vec![42]),
        "the re-granted capability answers on the pinned handle"
    );
}

/// **The authority seam.** An artifact names a capability; it never carries one. A restoring host
/// whose registrar does not serve the name grants nothing and says which name it refused — so a
/// moment cannot smuggle authority into a host that would not have granted it fresh (INVARIANTS #3).
#[test]
fn a_restore_refuses_a_name_the_embedder_does_not_serve() {
    let mut a = Host::new();
    let h = a.grant_host_proc(Box::new(|_op, _args, _mem, _| Ok(vec![1])));
    a.register_cap_name("fs", h);
    let named = a.capture_durable_named();

    // No registrar at all: nothing is granted.
    let mut bare = Host::new();
    assert_eq!(
        bare.restore_durable_named(&named).unwrap_err().name,
        "fs",
        "a host with no registrar refuses, naming what it could not serve"
    );

    // A registrar that serves a *different* name refuses this one rather than substituting.
    let mut picky = Host::new();
    picky.set_named_cap_registrar(Box::new(|name, _state| {
        (name == "display")
            .then(|| -> temen_interp::HostProc { Box::new(|_op, _args, _mem, _| Ok(vec![0])) })
    }));
    assert_eq!(picky.restore_durable_named(&named).unwrap_err().name, "fs");
}

/// #1458: the restored table is **exactly** the captured set. An embedder that grants its powerbox
/// fresh and then restores an artifact over it (a reactor thaw) must not resurrect a capability the
/// guest had dropped before the freeze — the artifact does not carry it, so the slot comes back
/// closed, and the guest's stale handle value stays dead.
#[test]
fn restore_closes_the_slots_the_capture_does_not_carry() {
    let mut a = Host::new();
    a.grant_clock();
    let out = a.grant_stream(StreamRole::Out);
    let exit = a.grant_exit();
    a.close(exit); // the guest dropped it before the freeze
    let captured = a.capture_durable_handles().unwrap();
    assert_eq!(captured.len(), 2);

    // The thawing embedder grants the same powerbox fresh — `exit` included, at the same slot.
    let mut b = Host::new();
    b.grant_clock();
    assert_eq!(b.grant_stream(StreamRole::Out), out);
    assert_eq!(b.grant_exit(), exit);
    b.restore_durable_handles(&captured);
    assert_eq!(
        b.capture_durable_handles().unwrap(),
        captured,
        "the restored table carries what the artifact carries and nothing else"
    );
    assert!(
        !b.handle_live(exit),
        "the dropped capability is not resurrected by the fresh grant"
    );
    assert!(b.handle_live(out));
}

// ---------------------------------------------------------------------------------------------
// #1502 — a `Budget` is durable. Its whole state is four `i64`s of *remaining* quota, so it rides a
// capture verbatim and a restore re-mints it. Carrying the remaining rather than re-granting fresh is
// what keeps INVARIANTS #3's conservation across a freeze: what the domain already spent stays spent.
// ---------------------------------------------------------------------------------------------

use temen_interp::BudgetState;

/// A budget that has been partly spent captures as its **remaining** quotas, restores into a fresh
/// table at the same slot, and the guest-held handle then reads the same remaining — not the
/// original grant.
#[test]
fn a_budget_round_trips_through_capture_restore_with_its_remaining_intact() {
    let mut a = Host::new();
    a.grant_clock();
    let h = a.grant_budget_channel(7, 1 << 20, 3, -1); // `-1` channel: the unbounded encoding
    assert!(a.budget_mem_take(h, 4096), "spend some of the mem quota");
    let remaining = BudgetState {
        fuel: 7,
        mem: (1 << 20) - 4096,
        spawn: 3,
        channel: -1,
        lane: -1,
    };

    let captured = a
        .capture_durable_handles()
        .expect("a Budget is durable: the capture must not refuse it");
    assert!(
        captured
            .iter()
            .any(|c| c.binding == DurableBinding::Budget(remaining)),
        "the capture carries the remaining quotas, not the original grant: {captured:?}"
    );

    let mut b = Host::new();
    b.restore_durable_handles(&captured);
    assert_eq!(
        b.capture_durable_handles().unwrap(),
        captured,
        "restore reinstates the exact captured set, Budget included"
    );
    // The guest's handle value resolves to the re-minted entry, and `read(mem)` is the remaining.
    assert_eq!(
        b.cap_dispatch_slots(cap_id::BUDGET, 1, h, &[1], None),
        Ok(vec![remaining.mem]),
        "read(mem) after restore is what was left at capture"
    );
    assert_eq!(
        b.cap_dispatch_slots(cap_id::BUDGET, 1, h, &[3], None),
        Ok(vec![-1]),
        "an unbounded field survives the two's-complement uleb"
    );
}

/// The complement: a drain keeps a `Budget`, exactly as it keeps every other durable binding — a
/// domain holding one no longer has to give up its minting authority to become snapshottable.
#[test]
fn a_drain_keeps_a_budget() {
    let mut a = Host::new();
    let h = a.grant_budget(1, 2, 3);
    assert!(
        a.drain_non_durable().is_empty(),
        "nothing to drain: a Budget is durable"
    );
    assert_eq!(
        a.cap_dispatch_slots(cap_id::BUDGET, 1, h, &[2], None),
        Ok(vec![3]),
        "the budget is untouched by the drain"
    );
}
