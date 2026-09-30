//! #1944 — the budget tree. A node's ceilings cap its whole subtree; a use charges the node that pays
//! and every ancestor, all or nothing; ending the use gives it back to every level. Driven through
//! the guest-facing ops (`split` / `read` / `transfer` via `cap_dispatch_slots`) and the charge a
//! detached window's admission makes (`budget_mem_take` / `budget_mem_give`).

use super::*;

const KIB: i64 = 1024;
const MIB: i64 = 1024 * KIB;

/// `split(-1, mem, -1)` of `holder`: a child node capped on `mem` only.
fn split(h: &mut Host, holder: i32, mem: i64) -> i32 {
    h.cap_dispatch_slots(cap_id::BUDGET, 0, holder, &[-1, mem, -1], None)
        .expect("split")[0] as i32
}

/// `read(mem)`: the room left along `b`'s chain.
fn room(h: &mut Host, b: i32) -> i64 {
    h.cap_dispatch_slots(cap_id::BUDGET, 1, b, &[1], None)
        .expect("read")[0]
}

fn transfer(h: &mut Host, holder: i32, dst: i32, mem: i64) -> i64 {
    h.cap_dispatch_slots(cap_id::BUDGET, 2, holder, &[dst as i64, 0, mem, 0], None)
        .expect("transfer")[0]
}

/// #1944's example: A holds 2 MiB and grants B 1 MiB; B grants C 512 KiB. C's subtree is capped at
/// 512 KiB while B can still use its full 1 MiB, and B's at 1 MiB while A can still reach 2 MiB minus
/// what B's subtree uses.
#[test]
fn a_ceiling_caps_its_subtree_and_every_ancestor_is_charged() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, 2 * MIB, -1);
    let b = split(&mut h, a, MIB);
    let c = split(&mut h, b, MIB / 2);

    assert!(
        h.budget_mem_take(c, (MIB / 2) as u64),
        "C fills its ceiling"
    );
    assert!(
        !h.budget_mem_take(c, 1),
        "C's subtree is refused at 512 KiB"
    );
    assert_eq!(room(&mut h, b), MIB / 2, "C's use counts against B");
    assert!(
        h.budget_mem_take(b, (MIB / 2) as u64),
        "B still reaches its full 1 MiB"
    );
    assert!(!h.budget_mem_take(b, 1), "B's subtree is refused at 1 MiB");
    assert_eq!(room(&mut h, a), MIB, "A has 2 MiB less B's 1 MiB");
    assert!(
        h.budget_mem_take(a, MIB as u64),
        "A reaches 2 MiB minus B's use"
    );
    assert!(!h.budget_mem_take(a, 1));
    assert_eq!(room(&mut h, c), 0, "C is capped by its ancestors too");
}

/// A ceiling is a cap, not a reservation: siblings may be granted more between them than their parent
/// holds, and the first to charge wins, bounded by the parent.
#[test]
fn siblings_may_overcommit_their_parent_and_the_first_to_charge_wins() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, 2 * MIB, -1);
    let b = split(&mut h, a, MIB);
    let d = split(&mut h, a, 3 * MIB / 2);
    assert_eq!(room(&mut h, a), 2 * MIB, "a split deducts nothing");
    assert!(h.budget_mem_take(d, (3 * MIB / 2) as u64));
    assert_eq!(room(&mut h, b), MIB / 2, "B's room is what A has left");
    assert!(!h.budget_mem_take(b, (MIB / 2 + 1) as u64));
    assert!(h.budget_mem_take(b, (MIB / 2) as u64));
}

/// A charge that fits its own node but not an ancestor charges nothing anywhere; ending the use gives
/// it back to every level.
#[test]
fn a_charge_is_all_or_nothing_and_a_refund_returns_every_level() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, MIB, -1);
    let b = split(&mut h, a, -1); // unbounded on its own: only A caps it
    assert!(h.budget_mem_take(a, (MIB / 2) as u64));
    assert!(
        !h.budget_mem_take(b, (MIB / 2 + 1) as u64),
        "A lacks the room"
    );
    assert_eq!(
        (room(&mut h, a), room(&mut h, b)),
        (MIB / 2, MIB / 2),
        "the refused charge left nothing behind"
    );
    assert!(h.budget_mem_take(b, (MIB / 2) as u64));
    assert_eq!((room(&mut h, a), room(&mut h, b)), (0, 0));
    h.budget_mem_give(b, (MIB / 2) as u64);
    assert_eq!(
        (room(&mut h, a), room(&mut h, b)),
        (MIB / 2, MIB / 2),
        "the refund reached A too"
    );
}

/// `split` clamps a child's ceiling to the tightest along the holder's chain, and `transfer` raises a
/// child's ceiling up to the holder's, never past it.
#[test]
fn split_clamps_and_transfer_raises_within_the_holders_ceiling() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, MIB, -1);
    let b = split(&mut h, a, -1);
    let c = split(&mut h, b, 2 * MIB);
    assert_eq!(
        room(&mut h, c),
        MIB,
        "C's ceiling is clamped to A's, through B"
    );

    let d = split(&mut h, a, 100);
    assert_eq!(transfer(&mut h, a, d, 50), 0);
    assert_eq!(room(&mut h, d), 150, "D's ceiling raised by 50");
    assert_eq!(room(&mut h, a), MIB, "a transfer deducts nothing");
    assert_eq!(transfer(&mut h, a, d, 2 * MIB), 0);
    assert_eq!(room(&mut h, d), MIB, "raised no further than A's ceiling");
}

/// `transfer` raises only a node under the holder, whose every charge the holder pays too. Raising a
/// peer or an ancestor would let the caller spend the holder's ceiling a second time outside it.
#[test]
fn a_transfer_reaches_only_the_holders_subtree() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, MIB, -1);
    let b = split(&mut h, a, 100);
    let c = split(&mut h, b, 50);
    let d = split(&mut h, a, 100);
    assert_eq!(
        transfer(&mut h, b, d, 10),
        EINVAL,
        "a peer: D's use never charges B"
    );
    assert_eq!(transfer(&mut h, c, b, 10), EINVAL, "an ancestor");
    assert_eq!(transfer(&mut h, a, a, 10), EINVAL, "itself");
    assert_eq!(
        (room(&mut h, b), room(&mut h, d)),
        (100, 100),
        "the refusals raised nothing"
    );
    assert_eq!(transfer(&mut h, a, c, 10), 0, "a grandchild");
    assert_eq!(room(&mut h, c), 60);
}

/// A fork twin shares its parent's tree, as a forked process stays in its cgroup: its copied handle
/// names the parent's node, so a charge by either is seen by both.
#[test]
fn a_fork_twin_charges_its_parents_nodes() {
    let mut h = Host::new();
    let a = h.grant_budget(-1, MIB, -1);
    let mut twin = h
        .fork_powerbox(1)
        .expect("a plain budget-holding domain forks");
    assert!(twin.budget_mem_take(a, (MIB / 4) as u64));
    assert_eq!(
        room(&mut h, a),
        3 * MIB / 4,
        "the parent sees the twin's charge"
    );
}

/// Parent and child charge one tree at once (#1944's concurrency note): one lock makes every chain
/// charge all or nothing, so however the charges race, the shared ancestor is never overdrawn and
/// none is lost or doubled.
#[test]
fn concurrent_charges_never_overdraw_a_shared_ancestor() {
    let tree = Arc::new(BudgetTree::default());
    let root = tree.mint(None, [-1, 1000, -1, -1, -1], [0; BUDGET_DIMS]);
    let kids: Vec<u32> = (0..4)
        .map(|_| tree.split(root, [-1, 600, -1, -1], -1).expect("split"))
        .collect();
    let admitted: usize = std::thread::scope(|s| {
        let runs: Vec<_> = kids
            .iter()
            .map(|&k| {
                let tree = &tree;
                s.spawn(move || (0..200).filter(|_| tree.charge(k, BUDGET_MEM, 7)).count())
            })
            .collect();
        runs.into_iter().map(|r| r.join().expect("charger")).sum()
    });
    assert_eq!(admitted, 1000 / 7, "exactly what fits the root");
    assert_eq!(
        tree.room(root).expect("root")[BUDGET_MEM],
        1000 - 7 * (1000 / 7),
        "the root's charge is the sum of its children's"
    );
}

/// `read(fuel)`: the fuel room left along `b`'s chain.
fn fuel_room(h: &mut Host, b: i32) -> i64 {
    h.cap_dispatch_slots(cap_id::BUDGET, 1, b, &[0], None)
        .expect("read")[0]
}

/// #1944 slice 3 — fuel is drawn in chunks of at most half the chain's room, each charged to every
/// level, and what a consumer drew and did not burn goes back to every level.
#[test]
fn fuel_is_drawn_in_chunks_charged_to_every_level_and_refunded_unburned() {
    let mut h = Host::new();
    let root = h.grant_budget(1000, -1, -1);
    let node = h
        .cap_dispatch_slots(cap_id::BUDGET, 0, root, &[100, -1, -1], None)
        .expect("split")[0] as i32;
    let mut child = Host::new();
    h.give_child_budget(node, &mut child);
    let src = child.fuel_source();
    assert_eq!(src.draw(), Some(50), "half the chain's room");
    assert_eq!(fuel_room(&mut h, node), 50);
    assert_eq!(fuel_room(&mut h, root), 950, "charged to every level");
    src.give_back(20);
    assert_eq!(fuel_room(&mut h, node), 70);
    assert_eq!(fuel_room(&mut h, root), 970, "refunded to every level");

    // A vCPU that ends hands back the rest of its draw: it drew 35 of the 70 and burned one.
    let mut fuel = Fuel::drawn(child.fuel_source());
    fuel.burn().expect("room left");
    drop(fuel);
    assert_eq!(fuel_room(&mut h, node), 69);

    // A lone vCPU burns the chain to its last unit, then traps.
    let mut fuel = Fuel::drawn(child.fuel_source());
    for _ in 0..69 {
        fuel.burn().expect("room left");
    }
    assert_eq!(fuel.burn(), Err(Trap::OutOfFuel));
    drop(fuel);
    assert_eq!(fuel_room(&mut h, node), 0);
    assert_eq!(
        fuel_room(&mut h, root),
        900,
        "the node's 100, charged to the root"
    );
}

/// #1944 slice 3 — an activation's limit is its own node's fuel room, whatever earlier activations
/// burned, and a chain with no bounded level has nothing to meter.
#[test]
fn an_activation_sets_its_nodes_fuel_room_and_an_unbounded_chain_is_unmetered() {
    let mut h = Host::new();
    assert_eq!(h.fuel_source().draw(), None, "no activation yet: unbounded");
    let mut fuel = Fuel::drawn(h.begin_activation(10));
    for _ in 0..4 {
        fuel.burn().expect("room left");
    }
    drop(fuel);
    assert_eq!(
        h.fuel_left(),
        6,
        "read back once the vCPU handed back its rest"
    );
    let src = h.begin_activation(10);
    assert_eq!(
        h.fuel_left(),
        10,
        "a new activation starts with its own limit"
    );
    assert_eq!(src.draw(), Some(5));
    let _ = h.begin_activation(u64::MAX);
    assert_eq!(h.fuel_left(), u64::MAX, "u64::MAX: unbounded");
    let mut fuel = Fuel::drawn(h.fuel_source());
    fuel.burn().expect("unmetered");
    assert_eq!(
        fuel.remaining(),
        i64::MAX,
        "fuel.remaining reads unmetered as i64::MAX"
    );
}

/// #1944 slice 3 — a guest's `read` and `split` see the budgets it holds, not the activation's limit
/// above them (what is left of that moves with when each engine draws); a draw is bounded by it.
#[test]
fn a_guest_does_not_see_the_activation_limit_that_bounds_its_draws() {
    let mut h = Host::new();
    let b = h.grant_budget(-1, -1, -1);
    let mut root = Fuel::drawn(h.begin_activation(10));
    root.burn().expect("room left"); // draws 5 of the activation's 10
    assert_eq!(
        fuel_room(&mut h, b),
        -1,
        "the activation's limit is not the guest's to read"
    );
    let sub = h
        .cap_dispatch_slots(cap_id::BUDGET, 0, b, &[-1, -1, -1], None)
        .expect("split")[0] as i32;
    assert_eq!(fuel_room(&mut h, sub), -1, "nor does it clamp a split");

    let mut child = Host::new();
    h.give_child_budget(sub, &mut child);
    let mut fuel = Fuel::drawn(child.fuel_source());
    for _ in 0..5 {
        fuel.burn().expect("the activation's room");
    }
    assert_eq!(
        fuel.burn(),
        Err(Trap::OutOfFuel),
        "a draw is bounded by the activation"
    );
    drop((root, fuel));
    assert_eq!(h.fuel_left(), 4, "the root burned 1 and the child 5");
}
