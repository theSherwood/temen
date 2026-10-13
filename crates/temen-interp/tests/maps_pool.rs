//! #2268 — the process-wide pool of mappings (`maps_pool`): `vm.max_map_count` less what the host
//! keeps for itself, which every run's `maps` charges are held against whatever their grants. Once the
//! runs of a process hold it all, a charge that would take another mapping is refused, even under an
//! unbounded grant, and so is a run's activation, which no grant refuses; a run that ends hands back
//! what it held. One test in its own binary: the pool is the process's, and tests in one binary share
//! their process.

use temen_interp::{
    maps_pool, Host, Lease, RootWindow, MAPS_PER_DOMAIN, MAPS_PER_VCPU, WINDOW_RUNS,
};

/// A detached spawn's lease on `budget` for a 64 KiB window.
fn lease(budget: i32) -> Lease {
    Lease {
        budget,
        bytes: 1 << 16,
        runs: WINDOW_RUNS,
    }
}

#[test]
fn the_pool_refuses_what_no_grant_does_and_takes_back_what_ends() {
    let pool = maps_pool();
    if pool > 1 << 22 {
        eprintln!("skipped: a pool of {pool} mappings is too large to fill");
        return;
    }
    let mut spawner = Host::new();
    let budget = spawner.grant_budget(-1, -1, -1);
    assert!(
        spawner.admit_detached_spawn(lease(budget)).is_some(),
        "an empty pool admits"
    );
    // Runs whose roots hold the rest: an activation's mappings are its embedder's choice, held past
    // any ceiling but never past the pool, so they fill it to within one root's share, and the next
    // activation is refused, charging nothing.
    let per_root = MAPS_PER_DOMAIN + MAPS_PER_VCPU;
    let mut roots = Vec::new();
    for _ in 0..=pool / per_root {
        let mut h = Host::new();
        if h.begin_activation(0, RootWindow::default()).is_none() {
            break;
        }
        roots.push(h);
    }
    // The spawner's child, a fresh window's lease, holds one root's share already.
    assert_eq!(
        roots.len() as u64,
        (pool - per_root) / per_root,
        "the pool holds as many roots as fit beside the spawner's child"
    );
    assert!(
        spawner.admit_detached_spawn(lease(budget)).is_none(),
        "a full pool refuses, though the budget is unbounded"
    );
    drop(roots);
    assert!(
        spawner.admit_detached_spawn(lease(budget)).is_some(),
        "the runs that ended handed their mappings back"
    );
}
