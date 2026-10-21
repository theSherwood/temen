//! #2305 — the process-wide pool of address space (`reserved_pool`): every domain's window is held
//! against it at its admission, which no grant bounds. Once the runs of a process hold it all, a run's
//! activation, a detached spawn's admission and a fork's or a spawn's new process are each refused,
//! charging nothing, and a domain that ends hands its window's reservation back. One test in its own
//! binary: the pool is the process's, and tests in one binary share their process.

use temen_interp::{
    maps_pool, reserved_pool, Host, Lease, RootWindow, SpawnPlan, MAPS_PER_DOMAIN, MAPS_PER_VCPU,
    WINDOW_RUNS,
};

/// The address space a window of the default reservation holds of the pool.
const WINDOW: u64 = 1 << temen_ir::DEFAULT_RESERVED_LOG2;

/// The declared size of the windows the spawner's children and processes would have.
const DECLARED: u64 = 1 << 16;

/// A detached spawn's lease on `budget`: its window reserves the default.
fn lease(budget: i32) -> Lease {
    Lease {
        budget,
        bytes: DECLARED,
        runs: WINDOW_RUNS,
    }
}

/// An activation holding a window of the default reservation and nothing else, as another run of the
/// process would: `None` when the pool has no room for it.
fn hold() -> Option<Host> {
    let mut h = Host::new();
    let window = RootWindow {
        reserved: WINDOW,
        ..RootWindow::default()
    };
    h.begin_activation(0, window)?;
    Some(h)
}

#[test]
fn the_pool_refuses_every_window_past_it_and_takes_back_what_ends() {
    let pool = reserved_pool();
    let slots = pool / WINDOW;
    if pool == u64::MAX || maps_pool() < 2 * slots * (MAPS_PER_DOMAIN + MAPS_PER_VCPU) {
        eprintln!("skipped: no address-space pool, or a pool of mappings too small to fill it");
        return;
    }
    let mut spawner = Host::new();
    let budget = spawner.grant_budget(-1, -1, -1);
    let mut roots: Vec<Host> = std::iter::from_fn(hold).collect();
    assert_eq!(roots.len() as u64, slots, "the pool holds its windows");
    assert!(
        spawner.admit_detached_spawn(lease(budget)).is_none(),
        "a full pool refuses a detached spawn, though its budget is unbounded"
    );
    assert!(
        spawner.fork_powerbox_jit(7, 0, DECLARED).is_none(),
        "and a fork's process"
    );
    assert!(
        spawner
            .spawn_powerbox(8, SpawnPlan(Box::new(())), DECLARED)
            .is_none(),
        "and a spawned one"
    );
    // One window handed back: room for one domain, held until it ends.
    roots.pop();
    let lane = spawner
        .admit_detached_spawn(lease(budget))
        .expect("the refusals charged nothing");
    assert!(hold().is_none(), "the child's lease holds the window");
    spawner.undo_detached_spawn(lease(budget), lane);
    let mut twin = spawner
        .fork_powerbox_jit(9, 0, DECLARED)
        .expect("the lease handed its window back");
    assert!(hold().is_none(), "the process holds the window");
    twin.release_memory();
    roots.extend(hold());
    assert_eq!(roots.len() as u64, slots, "the process handed it back");
    // Runs that end hand back what their trees hold, so the pool is whole again.
    drop(roots);
    let again: Vec<Host> = std::iter::from_fn(hold).collect();
    assert_eq!(
        again.len() as u64,
        slots,
        "every run that ended handed its window back"
    );
}
