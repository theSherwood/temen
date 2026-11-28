//! Exhaustive Loom concurrency models for `temen-kernel` (red-team refinement #5).
//!
//! Run with:
//! `RUSTFLAGS="--cfg loom" cargo test -p temen-kernel --lib loom`

use loom::sync::atomic::{AtomicU64, Ordering};
use loom::sync::Arc;
use loom::thread;

use crate::budget::BudgetId;
use crate::domain::{Domain, DomainControl, FiberResumeClaim, NoopLifecycleHooks};
use crate::types::{
    CarrierId, ContextExit, ContextId, ContextKind, DomainId, MemSpaceId, Trap, WaitKey, WakeReason,
};
use crate::wait::{RegisteredPark, WaitIndex, WaiterSpec};

/// Model 1: Dekker futex protocol — a waiter checking `word == 0` vs. a notifier storing `1` and
/// calling `WaitIndex::notify`. Under every possible memory interleaving, either the waiter sees
/// `word != 0` immediately OR it parks and is guaranteed to be woken by the notifier (zero lost
/// wakes).
#[test]
fn loom_wait_index_dekker_no_lost_wake() {
    loom::model(|| {
        let waits = Arc::new(WaitIndex::default());
        let word = Arc::new(AtomicU64::new(0));
        let key = WaitKey::Futex {
            space: MemSpaceId(1),
            offset: 64,
        };

        let w_waits = Arc::clone(&waits);
        let w_word = Arc::clone(&word);
        let waiter = thread::spawn(move || {
            let spec = WaiterSpec {
                domain: DomainId(0),
                ctx: ContextId(1),
                kind: ContextKind::Thread,
                interruptible: true,
            };
            match w_waits.register_futex(key, spec, || w_word.load(Ordering::Relaxed) == 0) {
                RegisteredPark::Immediate(1) => true,
                RegisteredPark::Parked { slot, .. } => {
                    assert_eq!(slot.wait_blocking(), WakeReason::Ready);
                    true
                }
                _ => unreachable!(),
            }
        });

        let n_waits = Arc::clone(&waits);
        let n_word = Arc::clone(&word);
        let notifier = thread::spawn(move || {
            n_word.store(1, Ordering::Relaxed);
            n_waits.notify(key, 1, WakeReason::Ready)
        });

        assert!(waiter.join().unwrap());
        let _ = notifier.join().unwrap();
        assert_eq!(waits.total_parked(), 0);
        assert_eq!(waits.stripe_total(), 0);
    });
}

/// Model 2: `DomainControl::end` racing a thread registering a futex wait. Checking
/// `domain.poll_flags().is_empty()` inside `register_futex`'s Dekker-fenced predicate guarantees
/// that either the waiter observes `ENDED` immediately or `end()`'s `wake_domain` wakes it with
/// `WakeReason::Revoked` — no waiter is ever stranded past domain death (Invariant 6).
#[test]
fn loom_domain_end_never_strands_parking_waiter() {
    loom::model(|| {
        let waits = Arc::new(WaitIndex::default());
        let domain = Domain::new(
            DomainId(0),
            BudgetId::ROOT,
            std::sync::Arc::new(NoopLifecycleHooks),
        );
        let ctl = DomainControl::new(Arc::clone(&domain), Arc::clone(&waits));
        let key = WaitKey::Futex {
            space: MemSpaceId(1),
            offset: 128,
        };

        let w_waits = Arc::clone(&waits);
        let w_domain = Arc::clone(&domain);
        let waiter = thread::spawn(move || {
            let spec = WaiterSpec {
                domain: DomainId(0),
                ctx: ContextId(1),
                kind: ContextKind::Thread,
                interruptible: false,
            };
            match w_waits.register_futex(key, spec, || w_domain.poll_flags().is_empty()) {
                RegisteredPark::Immediate(_) => {
                    // Observed ENDED before parking — valid!
                }
                RegisteredPark::Parked { slot, .. } => {
                    assert_eq!(slot.wait_blocking(), WakeReason::Revoked);
                }
            }
        });

        let killer = thread::spawn(move || {
            ctl.end(ContextExit::Trapped(Trap::Unreachable));
        });

        waiter.join().unwrap();
        killer.join().unwrap();
        assert_eq!(waits.total_parked(), 0);
    });
}

/// Model 3: `GuestFiber` `FIBER_PARKED` cross-Carrier migration — Carrier 0 parks a fiber,
/// a notifier wakes it, and Carrier 1 races with Carrier 0 to claim it. At most ONE Carrier claims
/// the fiber, and once woken exactly one succeeds.
#[test]
fn loom_guest_fiber_exclusive_claim_on_migration() {
    loom::model(|| {
        let domain = Domain::new(
            DomainId(0),
            BudgetId::ROOT,
            std::sync::Arc::new(NoopLifecycleHooks),
        );
        let (ctx, handle) = domain.create_fiber(None);
        let key = WaitKey::Futex {
            space: MemSpaceId(1),
            offset: 256,
        };
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(0)),
            FiberResumeClaim::Claimed { ctx, wake: None }
        );
        assert!(domain.mark_fiber_parked(handle, key));
        assert!(domain.wake_fiber_by_ctx(ctx, WakeReason::Ready));

        let d0 = Arc::clone(&domain);
        let t0 = thread::spawn(move || d0.claim_fiber_resume(handle, CarrierId(0)));
        let d1 = Arc::clone(&domain);
        let t1 = thread::spawn(move || d1.claim_fiber_resume(handle, CarrierId(1)));

        let r0 = t0.join().unwrap();
        let r1 = t1.join().unwrap();
        let claimed = usize::from(matches!(r0, FiberResumeClaim::Claimed { .. }))
            + usize::from(matches!(r1, FiberResumeClaim::Claimed { .. }));
        let faulted = usize::from(matches!(r0, FiberResumeClaim::Fault))
            + usize::from(matches!(r1, FiberResumeClaim::Fault));
        assert_eq!(claimed, 1, "exactly one Carrier must claim the woken fiber");
        assert_eq!(faulted, 1, "the losing Carrier must observe FiberFault");
    });
}

/// Model 4: Decision 21 `multi_carrier` fast-path latch — Carrier 0 writes shared state before
/// promoting `multi_carrier` (`Release`), and Carrier 1 reads `is_multi_carrier()` (`Acquire`).
/// Whenever Carrier 1 observes `true`, it must see Carrier 0's prior write.
#[test]
fn loom_multi_carrier_latch_happens_before() {
    loom::model(|| {
        let domain = Domain::new(
            DomainId(0),
            BudgetId::ROOT,
            std::sync::Arc::new(NoopLifecycleHooks),
        );
        let payload = Arc::new(AtomicU64::new(0));

        let d0 = Arc::clone(&domain);
        let p0 = Arc::clone(&payload);
        let t0 = thread::spawn(move || {
            p0.store(42, Ordering::Relaxed);
            d0.promote_multi_carrier();
        });

        let d1 = Arc::clone(&domain);
        let p1 = Arc::clone(&payload);
        let t1 = thread::spawn(move || {
            if d1.is_multi_carrier() {
                assert_eq!(p1.load(Ordering::Relaxed), 42);
            }
        });

        t0.join().unwrap();
        t1.join().unwrap();
    });
}
