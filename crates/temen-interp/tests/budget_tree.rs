//! #1944 — budgets are a **tree of ceilings** (INVARIANTS #3, ruling 2026-09-30). A node's ceiling
//! caps its whole subtree without being drawn from its parent; a charge lands on the paying node and
//! every ancestor, all-or-nothing; the node that pays for a detached child becomes the child's
//! `"budget"`, so the child's spawns charge inside its ceiling and its parent's.

use temen_interp::{cap_id, Host};

const KB: i64 = 1 << 10;

/// `split(-1, mem, -1)` on `budget`: a child node with a `mem` ceiling, the rest inherited.
fn split(h: &mut Host, budget: i32, mem: i64) -> i32 {
    let r = h
        .cap_dispatch_slots(cap_id::BUDGET, 0, budget, &[-1, mem, -1], None)
        .unwrap()[0];
    assert!(r >= 0, "split refused: {r}");
    r as i32
}

/// `read(mem)`: the room left along the node's chain.
fn room(h: &mut Host, budget: i32) -> i64 {
    h.cap_dispatch_slots(cap_id::BUDGET, 1, budget, &[1], None)
        .unwrap()[0]
}

/// A child host holding `budget`'s node as its `"budget"`, after `budget` paid `window` bytes for it —
/// what every spawn builder does ([`Host::admit_detached_spawn`], [`Host::give_child_budget`]).
fn spawn_child(parent: &mut Host, budget: i32, window: u64) -> (Host, i32) {
    assert!(
        parent.admit_detached_spawn(budget, window).is_some(),
        "admitted"
    );
    let mut child = Host::new();
    parent.give_child_budget(budget, &mut child);
    let b = child
        .resolve_cap_name("budget")
        .expect("the paying node is the child's budget");
    (child, b)
}

/// The owner's example: A holds 2 MB and grants B 1 MB; B grants C 500 KB. B's subtree is refused past
/// 1 MB while A still has room, A can use 2 MB minus what B's subtree uses, and C is capped at 500 KB
/// while B can still reach its full 1 MB.
#[test]
fn a_ceiling_caps_the_subtree_and_the_parent_keeps_its_own() {
    let mut a = Host::new();
    let ha = a.grant_budget(-1, 2048 * KB, -1);
    let hb = split(&mut a, ha, 1024 * KB);
    assert_eq!(room(&mut a, ha), 2048 * KB, "a split deducts nothing");

    // B's domain, paid 100 KB for its window from its own node.
    let (mut b, hb_b) = spawn_child(&mut a, hb, 100 * KB as u64);
    assert_eq!(room(&mut a, ha), 1948 * KB, "B's window charged A too");
    let hc = split(&mut b, hb_b, 500 * KB);
    assert_eq!(room(&mut b, hc), 500 * KB);

    // C's subtree stops at 500 KB, however much B and A have.
    assert!(b.budget_mem_take(hc, 400 * KB as u64));
    assert!(!b.budget_mem_take(hc, 200 * KB as u64), "C past its 500 KB");
    assert_eq!(
        room(&mut b, hb_b),
        1024 * KB - 500 * KB,
        "C's use counts inside B"
    );
    // B reaches its full 1 MB with C's use in it, and not a byte more.
    assert!(b.budget_mem_take(hb_b, 524 * KB as u64));
    assert!(
        !b.budget_mem_take(hb_b, 1),
        "B's subtree past 1 MB, while A has room"
    );
    // A: 2 MB minus B's subtree.
    assert_eq!(room(&mut a, ha), 1024 * KB);
    assert!(a.budget_mem_take(ha, 1024 * KB as u64));
    assert!(!a.budget_mem_take(ha, 1));

    // A refund goes back up the same chain.
    b.budget_mem_give(hc, 400 * KB as u64);
    assert_eq!(room(&mut a, ha), 400 * KB);
    assert_eq!(
        room(&mut b, hc),
        400 * KB,
        "C's own use is gone; its room is now what B has left (1 MB − 100 KB − 524 KB)"
    );
}

/// Overcommit: children's ceilings may sum past the parent's; the first to allocate wins, and the
/// parent's ceiling bounds them all. A split asking past the holder's ceiling is clamped to it.
#[test]
fn ceilings_may_overcommit_and_the_parent_bounds_them_all() {
    let mut a = Host::new();
    let ha = a.grant_budget(-1, 2048 * KB, -1);
    let hb = split(&mut a, ha, 1024 * KB);
    let hd = split(&mut a, ha, 1536 * KB);
    assert!(a.budget_mem_take(hd, 1536 * KB as u64), "D first");
    assert_eq!(room(&mut a, hb), 512 * KB, "B's room is what A has left");
    assert!(!a.budget_mem_take(hb, 513 * KB as u64));
    assert!(a.budget_mem_take(hb, 512 * KB as u64));
    let he = split(&mut a, ha, 4096 * KB);
    assert_eq!(
        a.cap_dispatch_slots(cap_id::BUDGET, 1, he, &[1], None),
        Ok(vec![0]),
        "clamped to A's 2 MB, all of it in use"
    );
}

/// Parent and child charge one tree from different threads: a charge is all-or-nothing under the
/// tree's one lock, so the live total never passes the root's ceiling, and every byte comes back.
#[test]
fn concurrent_charges_never_pass_a_ceiling() {
    use std::sync::atomic::{AtomicI64, Ordering};
    use std::sync::Arc;
    const CEIL: i64 = 64 * KB;
    const STEP: u64 = 5 * KB as u64;
    let mut root = Host::new();
    let hr = root.grant_budget(-1, CEIL, -1);
    let hb = split(&mut root, hr, CEIL);
    let hosts: Vec<(Host, i32)> = (0..3).map(|_| spawn_child(&mut root, hb, 0)).collect();
    let live = Arc::new(AtomicI64::new(0));
    let workers: Vec<_> = hosts
        .into_iter()
        .map(|(mut h, b)| {
            let live = Arc::clone(&live);
            std::thread::spawn(move || {
                for _ in 0..2000 {
                    if h.budget_mem_take(b, STEP) {
                        let now = live.fetch_add(STEP as i64, Ordering::SeqCst) + STEP as i64;
                        assert!(now <= CEIL, "{now} live bytes under a {CEIL} ceiling");
                        live.fetch_sub(STEP as i64, Ordering::SeqCst);
                        h.budget_mem_give(b, STEP);
                    }
                }
            })
        })
        .collect();
    for _ in 0..2000 {
        if root.budget_mem_take(hr, STEP) {
            let now = live.fetch_add(STEP as i64, Ordering::SeqCst) + STEP as i64;
            assert!(now <= CEIL, "{now} live bytes under a {CEIL} ceiling");
            live.fetch_sub(STEP as i64, Ordering::SeqCst);
            root.budget_mem_give(hr, STEP);
        }
    }
    for w in workers {
        w.join().expect("worker");
    }
    assert_eq!(room(&mut root, hr), CEIL, "every charge was refunded");
}
