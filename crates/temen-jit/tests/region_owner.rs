//! **#1608 — a futex-key registry purge forgets only its own run's pages.**
//!
//! Window virtual addresses are recycled between runs, and the teardown purge runs *after* the
//! reservation is released, over the whole 1 TiB range. So a dying run could erase entries belonging
//! to whichever runs had since been handed those addresses — and then a waiter that had parked on
//! `Region(..)` and a notifier reading the map afterwards compute different keys for the same
//! address, the notify reaches nobody, and the run wedges with no error anywhere. Measured at 9 runs
//! in 50 of `child_exec_jit` at `--test-threads=6` before the owner tag, 0 in 50 after.
//!
//! These live in their own integration binary rather than in `temen-jit`'s unit tests **on purpose**.
//! Added there, they took that binary from 21 concurrent tests to 23 and the Windows
//! `fiber-scaling` lane began aborting `0xc0000005` inside `mem::tests`' PAL-guard tests — #1575,
//! whose whole character is a dependence on process warm-up ordering. These tests have no business
//! perturbing that, and the registry is public API, so this is where they belong.
//!
//! Addresses here are synthetic map keys, never mappings, and deliberately far from any real window
//! so this cannot disturb a run sharing the process.

use temen_jit::{region_canon_forget_window, region_canon_lookup, region_canon_new_owner};

/// 4 KiB everywhere temen runs; the registry pages with its own granule, so a test that stays a
/// whole page away from its neighbours does not need to know it exactly.
const PAGE: u64 = 4096;

/// A dead run's purge covers this page — by the time it runs the range is nobody's, and certainly not
/// still its own — so it must leave a live run's entry alone.
#[test]
fn a_dead_runs_purge_leaves_a_live_runs_entry_alone() {
    let abs = 0x5a5a_0000_0000u64;
    let (dead, live) = (region_canon_new_owner(), region_canon_new_owner());

    temen_jit::region_canon_record(abs, PAGE, 42, 0, live);
    assert_eq!(region_canon_lookup(abs), Some((42, 0)));

    region_canon_forget_window(abs - PAGE, PAGE * 4, dead);
    assert_eq!(
        region_canon_lookup(abs),
        Some((42, 0)),
        "another run's purge erased a live entry: a waiter and a notifier would now compute \
         different keys for this address, and the wakeup would be lost",
    );

    // The owner's own purge still works, so a reused address inherits nothing.
    region_canon_forget_window(abs - PAGE, PAGE * 4, live);
    assert_eq!(region_canon_lookup(abs), None);
}

/// A record by a live run replaces a dead one's entry for the same recycled page — correct, because
/// holding the reservation proves the previous owner is gone — and the dead one's late purge must
/// not take the replacement with it.
#[test]
fn a_live_run_takes_over_a_recycled_page() {
    let abs = 0x5a5a_1000_0000u64;
    let (first, second) = (region_canon_new_owner(), region_canon_new_owner());

    temen_jit::region_canon_record(abs, PAGE, 3, 0, first);
    temen_jit::region_canon_record(abs, PAGE, 7, 0, second);
    assert_eq!(region_canon_lookup(abs), Some((7, 0)));

    region_canon_forget_window(abs, PAGE, first);
    assert_eq!(region_canon_lookup(abs), Some((7, 0)));
}
