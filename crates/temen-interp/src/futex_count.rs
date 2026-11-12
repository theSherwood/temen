//! #2958 slice 2 — how many tasks may be waiting on each futex key, readable without the run lock,
//! so that a `memory.notify` with no waiter answers 0 without taking executor 2's lock
//! ([`crate::bytecode`]'s `ThreadRun::no_wake`).
//!
//! **The table.** [`STRIPES`] counters, each on a cache line of its own, indexed by a hash of the
//! [`FutexKey`] ([`stripe`], the one function both sides use). A run owns one table. Two keys on one
//! stripe make a notifier take the slow path when it need not: a count over-approximates "someone
//! may be waiting on this key", so a collision costs today's notify, never a wake.
//!
//! **Who counts.** Every waiter of every engine, inside the shared wait rule: a task counts in when
//! it joins its key's wait queue (`Tasks::park_wait`) and out when it leaves it (`Tasks::unpark_wait`),
//! and the wait rule parks *before* it compares the word (`wait_task`). A fiber's wait counts in and
//! never out ([`FutexCount::enter`] without a `leave`), because its exits are spread over the fiber
//! machinery; phase 2 of #2958 moves fiber waits into the kernel's wait index, and they count out
//! there.
//!
//! **Why no wake is lost (store buffering, Dekker).** Waiter, under the run lock: count in
//! ([`FutexCount::enter`]: an RMW, then `fence(SeqCst)`), then load the word and park only if it is
//! still the expected value. Notifier, after the guest's store to the word: `fence(SeqCst)`, then
//! load the key's count ([`FutexCount::quiet`]); 0 answers "no waiter" at once, anything else takes
//! the run lock and the rules' notify. The two `SeqCst` fences are totally ordered:
//! - the notifier's fence first: the waiter's load, after its own fence, sees the guest's store
//!   (or a later one), so it answers not-equal and does not sleep;
//! - the waiter's fence first: the notifier's load, after its own fence, sees the count-in, so it
//!   takes the lock, which serialises it with the waiter's park, and finds the waiter in the queue,
//!   or finds that it answered not-equal and left.
//!
//! Either way no waiter sleeps through the notify. The argument needs both fences and nothing
//! stronger on the accesses around them, which is also all [loom] models (it treats a `SeqCst`
//! access as `AcqRel` but models `fence(SeqCst)`; the models below use `Relaxed` for the guest's
//! store and the count's load, and show the protocol fails without either fence or with the count-in
//! after the compare). In machine terms: x86-64 is `lock xadd` + `mfence` against `mfence` + a plain
//! load; aarch64 `ldaddal` + `dmb ish` against `dmb ish` + `ldr`; wasm32 `atomic.rmw` +
//! `atomic.fence` against `atomic.fence` + a load. The guest's store is any store the guest made
//! before its notify, in program order on the notifying thread: an atomic one sits in the `SeqCst`
//! order itself, and a plain one (a native `mov`/`str`, an interpreted window write) is ordered
//! before the fence by every target's hardware mapping of it. (The C++ model gives a plain store
//! racing the waiter's atomic load no meaning at all; that race is the guest's own, as it is today.)
//!
//! [loom]: https://docs.rs/loom

#[cfg(all(loom, test))]
use loom::sync::atomic::{fence, AtomicU64, Ordering};
#[cfg(not(all(loom, test)))]
use std::sync::atomic::{fence, AtomicU64, Ordering};

use crate::FutexKey;

/// How many stripes a run's table has: a power of two. 64 stripes on 128-byte lines are 8 KiB per
/// run (#2958 condition 4); a run's idle threads parked on one key fill one stripe of them.
pub(crate) const STRIPES: usize = 64;

/// One stripe's count, on a line of its own (128 bytes: Apple's aarch64 line, two of x86's), so
/// waiters counting in on one key never contend with a notifier reading another.
#[repr(align(128))]
struct Stripe(AtomicU64);

/// A run's per-key wait counts ([the module](self)).
pub(crate) struct FutexCount {
    stripes: Box<[Stripe]>,
}

impl Default for FutexCount {
    fn default() -> FutexCount {
        FutexCount {
            stripes: (0..STRIPES).map(|_| Stripe(AtomicU64::new(0))).collect(),
        }
    }
}

/// The stripe of `key`: the one map from a key to its count, on the waiter's and the notifier's side
/// alike. A multiplicative hash of the key's identity and address, its top bits.
pub(crate) fn stripe(key: &FutexKey) -> usize {
    let (tag, ident, at) = match *key {
        FutexKey::Anon(i, a) => (0u64, i, a),
        FutexKey::Region(i, o) => (1u64, i, o),
    };
    let h = (at ^ ident.rotate_left(29) ^ tag.rotate_right(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    (h >> (64 - STRIPES.trailing_zeros())) as usize
}

impl FutexCount {
    /// A waiter on `key` counts itself in: before it compares its word ([the module](self)).
    pub(crate) fn enter(&self, key: &FutexKey) {
        self.bump(stripe(key));
        fence(Ordering::SeqCst);
    }

    /// A waiter on `key` that counted in has left its wait, however it ended.
    pub(crate) fn leave(&self, key: &FutexKey) {
        let was = self.stripes[stripe(key)].0.fetch_sub(1, Ordering::Relaxed);
        debug_assert!(was != 0, "a futex count went below zero");
    }

    /// Whether no task can be waiting on `key`: after the guest's store, so a `true` lets the notify
    /// answer 0 woken without the run lock ([the module](self)).
    pub(crate) fn quiet(&self, key: &FutexKey) -> bool {
        fence(Ordering::SeqCst);
        self.peek(stripe(key)) == 0
    }

    /// The sum of every stripe: what a run's waits account for (a debug check at their end).
    pub(crate) fn total(&self) -> u64 {
        (0..STRIPES).map(|s| self.peek(s)).sum()
    }

    fn bump(&self, s: usize) {
        self.stripes[s].0.fetch_add(1, Ordering::SeqCst);
    }

    fn peek(&self, s: usize) -> u64 {
        self.stripes[s].0.load(Ordering::Relaxed)
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;

    #[test]
    fn a_key_counts_in_and_out_on_its_one_stripe() {
        let c = FutexCount::default();
        let (k, other) = (FutexKey::Anon(7, 64), FutexKey::Anon(7, 68));
        assert!(c.quiet(&k));
        c.enter(&k);
        assert!(!c.quiet(&k));
        // Another key is quiet unless it shares the stripe, which costs only a slow path.
        assert_eq!(c.quiet(&other), stripe(&other) != stripe(&k));
        c.leave(&k);
        assert!(c.quiet(&k));
        assert_eq!(c.total(), 0);
    }

    /// Neighbouring words of one window, as a run's futexes sit, spread over the stripes rather than
    /// piling onto a few.
    #[test]
    fn neighbouring_words_spread_over_the_stripes() {
        let used: std::collections::BTreeSet<usize> = (0..STRIPES as u64)
            .map(|i| stripe(&FutexKey::Anon(0x7f00_1234_0000, 4096 + 4 * i)))
            .collect();
        assert!(
            used.len() >= STRIPES / 2,
            "{} of {STRIPES} stripes",
            used.len()
        );
    }

    #[test]
    fn a_stripe_is_a_line_of_its_own() {
        assert_eq!(std::mem::align_of::<Stripe>(), 128);
        assert_eq!(std::mem::size_of::<Stripe>(), 128);
    }
}

/// The count protocol under loom: `RUSTFLAGS="--cfg loom" cargo test -p temen-interp --lib loom`.
/// A waiter (count in, compare, park) against a notifier (the guest's store, then the count's read,
/// the rules' notify on the slow path), with the run lock a loom `Mutex` over the wait queue. Each
/// model asserts the futex property: a waiter that parked is woken, or the notifier ran before the
/// waiter's compare could see its store. The broken variants (a fence removed, the count-in after
/// the compare) must find the lost wake, which is how the model shows it can fail.
#[cfg(all(loom, test))]
mod loom_tests {
    use super::*;
    use loom::sync::atomic::AtomicU32;
    use loom::sync::{Arc, Mutex};

    const KEY: FutexKey = FutexKey::Anon(1, 8);

    /// The wait queue the run lock guards: whether the waiter parked on `KEY`, and whether a notify
    /// woke it.
    #[derive(Default)]
    struct Queue {
        parked: bool,
        woken: bool,
    }

    #[derive(Default)]
    struct World {
        word: AtomicU32,
        count: FutexCount,
        lock: Mutex<Queue>,
    }

    /// How a model's waiter and notifier are built.
    #[derive(Clone, Copy)]
    struct Shape {
        /// The waiter counts in before its compare (the protocol) or after it (broken).
        enter_first: bool,
        /// The waiter's fence after its count-in.
        waiter_fence: bool,
        /// The notifier's fence before its read.
        notifier_fence: bool,
    }

    const SOUND: Shape = Shape {
        enter_first: true,
        waiter_fence: true,
        notifier_fence: true,
    };

    fn enter(w: &World, s: Shape) {
        if s.waiter_fence {
            w.count.enter(&KEY);
        } else {
            w.count.bump(stripe(&KEY));
        }
    }

    /// The waiter: the rules' `wait_task` under the run lock, with `expected` 0.
    fn wait(w: &World, s: Shape) {
        let mut q = w.lock.lock().unwrap();
        if s.enter_first {
            enter(w, s);
        }
        // The guest's word, read as `atomic_value` reads it; loom takes `SeqCst` as `AcqRel`, so
        // the read contributes nothing the argument does not grant it.
        let equal = w.word.load(Ordering::SeqCst) == 0;
        if !s.enter_first && equal {
            enter(w, s);
        }
        if equal {
            q.parked = true;
        } else if s.enter_first {
            w.count.leave(&KEY);
        }
    }

    /// The notifier: the guest's plain store, as weak as the argument allows, then the fast path's
    /// read, and the rules' notify under the lock when it is not quiet.
    fn notify(w: &World, s: Shape) {
        w.word.store(1, Ordering::Relaxed);
        let quiet = if s.notifier_fence {
            w.count.quiet(&KEY)
        } else {
            w.count.peek(stripe(&KEY)) == 0
        };
        if quiet {
            return;
        }
        let mut q = w.lock.lock().unwrap();
        if q.parked && !q.woken {
            q.woken = true;
            w.count.leave(&KEY);
        }
    }

    fn model(s: Shape) {
        loom::model(move || {
            let w = Arc::new(World::default());
            let w2 = Arc::clone(&w);
            let waiter = loom::thread::spawn(move || wait(&w2, s));
            notify(&w, s);
            waiter.join().unwrap();
            let q = w.lock.lock().unwrap();
            assert!(
                !q.parked || q.woken,
                "the waiter parked and the notify missed it"
            );
        });
    }

    #[test]
    fn loom_no_wake_is_lost() {
        model(SOUND);
    }

    #[test]
    #[should_panic(expected = "missed it")]
    fn loom_without_the_waiters_fence_a_wake_is_lost() {
        model(Shape {
            waiter_fence: false,
            ..SOUND
        });
    }

    #[test]
    #[should_panic(expected = "missed it")]
    fn loom_without_the_notifiers_fence_a_wake_is_lost() {
        model(Shape {
            notifier_fence: false,
            ..SOUND
        });
    }

    #[test]
    #[should_panic(expected = "missed it")]
    fn loom_counting_in_after_the_compare_loses_a_wake() {
        model(Shape {
            enter_first: false,
            ..SOUND
        });
    }
}
