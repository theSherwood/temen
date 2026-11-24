//! #2804 — a serving window's handler admission, as a durable freeze reads it (owner decision 36,
//! "B′": quiesce, then decline). DESIGN.md §12a: "quiesce rides the admission word".
//!
//! One word holds the whole state, so every change is one read-modify-write and every reading one
//! load: whether the window admits a new handler, how many of its handlers are running, and how
//! many are parked.
//!
//! - A handler **starts** only while the window is open: it counts itself running, and the same RMW
//!   tells it whether the window was open. If it was not, it takes its count back and does not start.
//! - A running handler **parks** (running → parked) and is **woken** (parked → running) in one RMW
//!   each, so no reading ever sees a live handler in neither count. A wake is never refused: a woken
//!   handler runs to its end or its next park (B′'s row for a handler woken while a freeze is asked).
//! - A handler **ends** (running → gone) in one RMW.
//! - A freeze **closes** the window in one RMW and reads, at that instant, how many are running and
//!   how many are parked. After the close no handler starts, so the running count only falls.
//!
//! Separate words would not do: a census that read the running count, then the parked count, could
//! see neither of a handler that was woken between the two reads, and arm the freeze with it running.
//! The loom models below pin the one-word protocol.

#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering};

/// The window admits new handlers.
const OPEN: u64 = 1 << 63;
/// One running handler: the low 31 bits count them.
const RUNNING: u64 = 1;
/// One parked handler: bits 32..63 count them.
const PARKED: u64 = 1 << 32;
const COUNT: u64 = (1 << 31) - 1;

/// What a freeze's close found ([`Admission::close`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Census {
    /// Handlers running at the close. The freeze waits for these (B′'s quiesce): it arms once none
    /// runs.
    pub(crate) running: u32,
    /// Handlers parked at the close. With none running, any of these declines the freeze.
    pub(crate) parked: u32,
}

/// A serving window's handler admission.
pub(crate) struct Admission(AtomicU64);

impl Admission {
    /// Open, with no handler.
    pub(crate) fn new() -> Admission {
        Admission(AtomicU64::new(OPEN))
    }

    /// A handler starts: `true` if the window admits it, which counts it running. `false` once a
    /// freeze has closed the window; nothing is counted then.
    pub(crate) fn start(&self) -> bool {
        let was = self.0.fetch_add(RUNNING, Ordering::SeqCst);
        if was & OPEN != 0 {
            return true;
        }
        self.0.fetch_sub(RUNNING, Ordering::SeqCst);
        false
    }

    /// A running handler parks.
    pub(crate) fn park(&self) {
        // running − 1, parked + 1, in one RMW: `PARKED − RUNNING` never borrows across the fields
        // while the running count is at least one, which a parking handler's own count makes it.
        let was = self.0.fetch_add(PARKED - RUNNING, Ordering::SeqCst);
        debug_assert!(
            was & COUNT != 0,
            "a handler parks only while counted running"
        );
    }

    /// A parked handler is woken. Never refused (see the module doc).
    pub(crate) fn wake(&self) {
        let was = self.0.fetch_sub(PARKED - RUNNING, Ordering::SeqCst);
        debug_assert!(
            (was >> 32) & COUNT != 0,
            "a handler wakes only while counted parked"
        );
    }

    /// A running handler ends.
    pub(crate) fn end(&self) {
        let was = self.0.fetch_sub(RUNNING, Ordering::SeqCst);
        debug_assert!(
            was & COUNT != 0,
            "a handler ends only while counted running"
        );
    }

    /// A freeze closes the window and reads what is live at that instant.
    pub(crate) fn close(&self) -> Census {
        census(self.0.fetch_and(!OPEN, Ordering::SeqCst))
    }

    /// What is live now, the window left as it is: a freeze held by a running handler asks again at
    /// its next point.
    pub(crate) fn census(&self) -> Census {
        census(self.0.load(Ordering::SeqCst))
    }

    /// The freeze is over (declined, or the run thawed): new handlers start again.
    pub(crate) fn reopen(&self) {
        self.0.fetch_or(OPEN, Ordering::SeqCst);
    }
}

fn census(word: u64) -> Census {
    Census {
        running: (word & COUNT) as u32,
        parked: ((word >> 32) & COUNT) as u32,
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn a_handler_is_counted_running_then_parked_then_running_then_gone() {
        let a = Admission::new();
        assert!(a.start());
        assert_eq!(
            a.census(),
            Census {
                running: 1,
                parked: 0
            }
        );
        a.park();
        assert_eq!(
            a.census(),
            Census {
                running: 0,
                parked: 1
            }
        );
        a.wake();
        assert_eq!(
            a.census(),
            Census {
                running: 1,
                parked: 0
            }
        );
        a.end();
        assert_eq!(
            a.census(),
            Census {
                running: 0,
                parked: 0
            }
        );
    }

    #[test]
    fn a_closed_window_starts_nothing_but_still_wakes_and_ends_its_handlers() {
        let a = Admission::new();
        assert!(a.start());
        a.park();
        assert_eq!(
            a.close(),
            Census {
                running: 0,
                parked: 1
            }
        );
        assert!(
            !a.start(),
            "no handler starts while a freeze holds the window"
        );
        assert_eq!(
            a.census(),
            Census {
                running: 0,
                parked: 1
            },
            "the refused start left no count"
        );
        a.wake();
        a.end();
        assert_eq!(
            a.census(),
            Census {
                running: 0,
                parked: 0
            }
        );
        a.reopen();
        assert!(a.start());
    }
}

#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;

    /// A handler starting races the freeze's close: either the start is refused, or the close counts
    /// it. Never both missed (a freeze armed with a handler that then runs).
    #[test]
    fn a_start_racing_the_close_is_refused_or_counted() {
        loom::model(|| {
            let a = Arc::new(Admission::new());
            let h = {
                let a = Arc::clone(&a);
                loom::thread::spawn(move || a.start())
            };
            let seen = a.close();
            let started = h.join().unwrap();
            assert!(!started || seen.running == 1, "started unseen: {seen:?}");
            // And a refused start leaves no count behind.
            assert_eq!(a.census().running, u32::from(started));
        });
    }

    /// A parked handler woken while the freeze closes is always seen live, running or parked: the
    /// close never reads it in neither count.
    #[test]
    fn a_wake_racing_the_close_is_seen_live() {
        loom::model(|| {
            let a = Arc::new(Admission::new());
            assert!(a.start());
            a.park();
            let h = {
                let a = Arc::clone(&a);
                loom::thread::spawn(move || a.wake())
            };
            let seen = a.close();
            h.join().unwrap();
            assert_eq!(
                seen.running + seen.parked,
                1,
                "a live handler went unseen: {seen:?}"
            );
        });
    }

    /// A handler parking and another ending while the freeze closes: the close sees exactly the
    /// handlers not yet ended, each in one count.
    #[test]
    fn a_park_and_an_end_racing_the_close_are_each_counted_once() {
        loom::model(|| {
            let a = Arc::new(Admission::new());
            assert!(a.start());
            assert!(a.start());
            let p = {
                let a = Arc::clone(&a);
                loom::thread::spawn(move || a.park())
            };
            let e = {
                let a = Arc::clone(&a);
                loom::thread::spawn(move || a.end())
            };
            let seen = a.close();
            p.join().unwrap();
            e.join().unwrap();
            let live = seen.running + seen.parked;
            assert!(live == 1 || live == 2, "{seen:?}");
            assert_eq!(
                a.census(),
                Census {
                    running: 0,
                    parked: 1
                }
            );
        });
    }
}
