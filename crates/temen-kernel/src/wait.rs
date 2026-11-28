//! Unified [`WaitIndex`] for futexes and kernel waitable objects (`#3187` / `#3194`).
//!
//! Combines three hard-won properties in one `#![forbid(unsafe_code)]`, Loom-verified structure:
//! 1. **Lock-free no-wake fast path (`#2989`):** 64 cache-line-aligned (`#[repr(align(128))]`)
//!    atomic waiter-count stripes with Dekker `SeqCst` fences (`enter` -> `fence(SeqCst)` -> load
//!    word vs. guest store -> `fence(SeqCst)` -> `quiet`). `notify` with zero waiters returns `0`
//!    without taking any lock.
//! 2. **Non-blocking guest fiber parks (`FIBER_PARKED`, status 2 — red-team refinement #3):**
//!    A `ContextKind::GuestFiber` waiter enters the same `WaitKey` queue as any thread, under a
//!    single lock hold (fixing `#3153`'s 4-lock-hold fiber hand-off), and returns
//!    [`ParkDecision::YieldFiber`] instead of blocking the Carrier OS thread.
//! 3. **Domain-scoped wakes (Invariant 12) and last-sleeper deadlock settle (`#2967`):**
//!    `wake_domain` sweeps only waiters belonging to a specific `DomainId` (optionally only
//!    `interruptible` ones for `SIGINT` / `-EINTR`).

use std::collections::{HashMap, VecDeque};

use crate::sync::{
    fence, lock_unpoisoned, wait_unpoisoned, Arc, AtomicU64, Condvar, Mutex, Ordering,
};
use crate::types::{ContextId, ContextKind, DomainId, ParkDecision, WaitKey, WakeReason};

/// Power-of-two stripe count for lock-free futex no-wake checks (64 × 128 B = 8 KiB per runtime).
pub const STRIPES: usize = 64;

/// One cache-line-isolated waiter counter (128 bytes: Apple aarch64 line, two x86-64 lines).
#[repr(align(128))]
struct Stripe(AtomicU64);

struct FutexStripes {
    stripes: Box<[Stripe]>,
}

impl Default for FutexStripes {
    fn default() -> FutexStripes {
        FutexStripes {
            stripes: (0..STRIPES).map(|_| Stripe(AtomicU64::new(0))).collect(),
        }
    }
}

#[inline]
fn stripe_of(key: &WaitKey) -> Option<usize> {
    match *key {
        WaitKey::Futex { space, offset } => {
            let h = (offset ^ space.0.rotate_left(29)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            Some((h >> (64 - STRIPES.trailing_zeros())) as usize)
        }
        WaitKey::Object(_) => None,
    }
}

impl FutexStripes {
    #[inline]
    fn enter(&self, key: &WaitKey) {
        if let Some(s) = stripe_of(key) {
            self.stripes[s].0.fetch_add(1, Ordering::SeqCst);
            fence(Ordering::SeqCst);
        }
    }

    #[inline]
    fn leave(&self, key: &WaitKey) {
        if let Some(s) = stripe_of(key) {
            let prev = self.stripes[s].0.fetch_sub(1, Ordering::Relaxed);
            debug_assert!(prev != 0, "futex stripe count underflowed");
        }
    }

    #[inline]
    fn quiet(&self, key: &WaitKey) -> bool {
        let Some(s) = stripe_of(key) else {
            return false;
        };
        fence(Ordering::SeqCst);
        self.stripes[s].0.load(Ordering::Relaxed) == 0
    }

    #[cfg(test)]
    fn total(&self) -> u64 {
        self.stripes
            .iter()
            .map(|s| s.0.load(Ordering::Relaxed))
            .sum()
    }
}

/// Per-park synchronization cell shared between a parked waiter and its notifier.
///
/// For a blocking Carrier (`Root` / `Thread` / `ServeHandler`), the Carrier waits on `cv` until
/// `reason` becomes `Some(...)`. For a `GuestFiber` or cooperative Carrier, the caller polls
/// `take_reason()` or registers an optional wake callback (via the runtime's runnable queue).
pub struct WaitSlot {
    state: Mutex<Option<WakeReason>>,
    cv: Condvar,
}

impl WaitSlot {
    pub fn new() -> Arc<WaitSlot> {
        Arc::new(WaitSlot {
            state: Mutex::new(None),
            cv: Condvar::new(),
        })
    }

    /// Deliver `reason` once. Returns `true` if this call performed the wake (first writer wins).
    pub fn wake(&self, reason: WakeReason) -> bool {
        let mut g = lock_unpoisoned(&self.state);
        if g.is_some() {
            return false;
        }
        *g = Some(reason);
        self.cv.notify_one();
        true
    }

    /// Non-blocking check for whether this slot has already been woken.
    pub fn poll(&self) -> Option<WakeReason> {
        *lock_unpoisoned(&self.state)
    }

    /// Block the current OS thread until woken (used by Parallel Carrier for non-fiber contexts).
    pub fn wait_blocking(&self) -> WakeReason {
        let mut g = lock_unpoisoned(&self.state);
        while g.is_none() {
            g = wait_unpoisoned(&self.cv, g);
        }
        g.unwrap()
    }

    /// Block the current OS thread until woken or `timeout` elapses (real builds only).
    #[cfg(not(loom))]
    pub fn wait_timeout_blocking(&self, timeout: std::time::Duration) -> Option<WakeReason> {
        let deadline = std::time::Instant::now() + timeout;
        let mut g = lock_unpoisoned(&self.state);
        loop {
            if let Some(r) = *g {
                return Some(r);
            }
            let now = std::time::Instant::now();
            if now >= deadline {
                return None;
            }
            let (ng, timed_out) = crate::sync::wait_timeout_unpoisoned(&self.cv, g, deadline - now);
            g = ng;
            if let Some(r) = *g {
                return Some(r);
            }
            if timed_out {
                return None;
            }
        }
    }
}

/// Metadata identifying a waiter entering [`WaitIndex`].
#[derive(Clone, Copy, Debug)]
pub struct WaiterSpec {
    pub domain: DomainId,
    pub ctx: ContextId,
    pub kind: ContextKind,
    pub interruptible: bool,
}

struct WaiterEntry {
    spec: WaiterSpec,
    slot: Arc<WaitSlot>,
}

/// What [`WaitIndex::register_futex`] or [`WaitIndex::register_object`] returns to the Caller.
pub enum RegisteredPark {
    /// Did not park (e.g., futex word was not equal -> `1`).
    Immediate(i64),
    /// Waiter is parked in the queue; `slot` will receive the [`WakeReason`].
    /// If `spec.kind == ContextKind::GuestFiber`, the caller must yield `FIBER_PARKED` (status 2)
    /// without blocking the Carrier OS thread.
    Parked {
        slot: Arc<WaitSlot>,
        decision: ParkDecision,
    },
}

/// Optional host-level wake hook called *outside* `WaitIndex`'s lock whenever a waiter is woken
/// (e.g., to push a woken cooperative Context or `GuestFiber` back onto a runnable queue).
pub type WakeCallback = std::sync::Arc<dyn Fn(WaiterSpec, WakeReason) + Send + Sync>;

/// Unified wait table shared by all domains in a runtime instance.
pub struct WaitIndex {
    stripes: FutexStripes,
    queues: Mutex<HashMap<WaitKey, VecDeque<WaiterEntry>>>,
    on_wake: Option<WakeCallback>,
}

impl Default for WaitIndex {
    fn default() -> WaitIndex {
        WaitIndex::new(None)
    }
}

impl WaitIndex {
    /// Construct a new `WaitIndex`, with an optional `on_wake` callback invoked whenever any
    /// waiter leaves a queue with a [`WakeReason`].
    pub fn new(on_wake: Option<WakeCallback>) -> WaitIndex {
        WaitIndex {
            stripes: FutexStripes::default(),
            queues: Mutex::new(HashMap::new()),
            on_wake,
        }
    }

    /// Dekker-ordered futex registration:
    /// 1. Under `queues` lock, count in on `stripes` + `fence(SeqCst)`.
    /// 2. Run `check_eq()` to verify the guest word still equals `expected` (and domain is not
    ///    already terminal — callers pass that check inside `check_eq` or check `poll_flags` after
    ///    the fence!).
    /// 3. If not equal, count out immediately and return `RegisteredPark::Immediate(1)`.
    /// 4. If equal, enqueue FIFO waiter and return `RegisteredPark::Parked`.
    pub fn register_futex<F>(&self, key: WaitKey, spec: WaiterSpec, check_eq: F) -> RegisteredPark
    where
        F: FnOnce() -> bool,
    {
        let mut queues = lock_unpoisoned(&self.queues);
        self.stripes.enter(&key);
        if !check_eq() {
            self.stripes.leave(&key);
            return RegisteredPark::Immediate(1);
        }
        let slot = WaitSlot::new();
        let entry = WaiterEntry {
            spec,
            slot: Arc::clone(&slot),
        };
        queues.entry(key).or_default().push_back(entry);
        drop(queues);
        let decision = if spec.kind == ContextKind::GuestFiber {
            ParkDecision::YieldFiber
        } else {
            ParkDecision::Blocked(WakeReason::Ready)
        };
        RegisteredPark::Parked { slot, decision }
    }

    /// Register a waiter on a kernel waitable object (`WaitKey::Object(id)`), guarded by
    /// `still_waiting()` under the queue lock so a concurrent signal cannot be lost.
    pub fn register_object<F>(
        &self,
        key: WaitKey,
        spec: WaiterSpec,
        still_waiting: F,
    ) -> RegisteredPark
    where
        F: FnOnce() -> bool,
    {
        let mut queues = lock_unpoisoned(&self.queues);
        if !still_waiting() {
            return RegisteredPark::Immediate(0);
        }
        let slot = WaitSlot::new();
        let entry = WaiterEntry {
            spec,
            slot: Arc::clone(&slot),
        };
        queues.entry(key).or_default().push_back(entry);
        drop(queues);
        let decision = if spec.kind == ContextKind::GuestFiber {
            ParkDecision::YieldFiber
        } else {
            ParkDecision::Blocked(WakeReason::Ready)
        };
        RegisteredPark::Parked { slot, decision }
    }

    /// Wake up to `count` FIFO waiters on `key` with `reason`.
    ///
    /// For `WaitKey::Futex`, checks the lock-free stripe counter first (Dekker `fence(SeqCst)` +
    /// load): if 0, returns `0` immediately without touching the queue mutex (`#2989`).
    pub fn notify(&self, key: WaitKey, count: u32, reason: WakeReason) -> u32 {
        if count == 0 {
            return 0;
        }
        if matches!(key, WaitKey::Futex { .. }) && self.stripes.quiet(&key) {
            return 0;
        }
        let mut woken: Vec<WaiterEntry> = Vec::new();
        {
            let mut queues = lock_unpoisoned(&self.queues);
            if let Some(q) = queues.get_mut(&key) {
                while woken.len() < count as usize {
                    let Some(entry) = q.pop_front() else {
                        break;
                    };
                    self.stripes.leave(&key);
                    woken.push(entry);
                }
                if q.is_empty() {
                    queues.remove(&key);
                }
            }
        }
        let mut n = 0u32;
        for entry in woken {
            if entry.slot.wake(reason) {
                n += 1;
                if let Some(cb) = &self.on_wake {
                    cb(entry.spec, reason);
                }
            }
        }
        n
    }

    /// Wake *all* waiters on `key` with `reason` (e.g., thread exit / object completion).
    pub fn notify_all(&self, key: WaitKey, reason: WakeReason) -> u32 {
        self.notify(key, u32::MAX, reason)
    }

    /// Remove a specific waiter (e.g., on timeout) if it is still queued, returning `true` if
    /// removed before any notifier popped it.
    pub fn cancel(&self, key: WaitKey, ctx: ContextId, reason: WakeReason) -> bool {
        let removed = {
            let mut queues = lock_unpoisoned(&self.queues);
            let mut found = None;
            if let Some(q) = queues.get_mut(&key) {
                if let Some(pos) = q.iter().position(|e| e.spec.ctx == ctx) {
                    found = q.remove(pos);
                    self.stripes.leave(&key);
                }
                if q.is_empty() {
                    queues.remove(&key);
                }
            }
            found
        };
        if let Some(entry) = removed {
            if entry.slot.wake(reason) {
                if let Some(cb) = &self.on_wake {
                    cb(entry.spec, reason);
                }
                return true;
            }
        }
        false
    }

    /// Domain-scoped wake sweep (Invariant 12: never sweeps other domains).
    ///
    /// When `only_interruptible` is `true` (e.g., `DomainControl::interrupt_parks()`), non-
    /// interruptible waits remain queued. When `false` (e.g., `DomainControl::end()` or
    /// `set_paused(true)`), every waiter belonging to `domain` is woken.
    pub fn wake_domain(
        &self,
        domain: DomainId,
        reason: WakeReason,
        only_interruptible: bool,
    ) -> u32 {
        let mut woken: Vec<WaiterEntry> = Vec::new();
        {
            let mut queues = lock_unpoisoned(&self.queues);
            let mut empty_keys = Vec::new();
            for (&key, q) in queues.iter_mut() {
                let mut i = 0;
                while i < q.len() {
                    let matches_domain = q[i].spec.domain == domain;
                    let matches_intr = !only_interruptible || q[i].spec.interruptible;
                    if matches_domain && matches_intr {
                        let entry = q.remove(i).unwrap();
                        self.stripes.leave(&key);
                        woken.push(entry);
                    } else {
                        i += 1;
                    }
                }
                if q.is_empty() {
                    empty_keys.push(key);
                }
            }
            for k in empty_keys {
                queues.remove(&k);
            }
        }
        let mut n = 0u32;
        for entry in woken {
            if entry.slot.wake(reason) {
                n += 1;
                if let Some(cb) = &self.on_wake {
                    cb(entry.spec, reason);
                }
            }
        }
        n
    }

    /// Number of currently parked waiters belonging to `domain` (used by `#2967` last-sleeper
    /// deadlock settle).
    pub fn parked_count(&self, domain: DomainId) -> usize {
        let queues = lock_unpoisoned(&self.queues);
        queues
            .values()
            .flat_map(|q| q.iter())
            .filter(|e| e.spec.domain == domain)
            .count()
    }

    /// Total parked waiters across all keys (for diagnostics and tests).
    pub fn total_parked(&self) -> usize {
        let queues = lock_unpoisoned(&self.queues);
        queues.values().map(VecDeque::len).sum()
    }

    #[cfg(test)]
    pub(crate) fn stripe_total(&self) -> u64 {
        self.stripes.total()
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::types::{MemSpaceId, WaitableId};

    #[test]
    fn futex_fifo_notify_and_lock_free_quiet_path() {
        let idx = WaitIndex::default();
        let k = WaitKey::Futex {
            space: MemSpaceId(1),
            offset: 64,
        };
        assert_eq!(idx.notify(k, 1, WakeReason::Ready), 0);

        let s1 = WaiterSpec {
            domain: DomainId(0),
            ctx: ContextId(1),
            kind: ContextKind::Thread,
            interruptible: true,
        };
        let s2 = WaiterSpec {
            domain: DomainId(0),
            ctx: ContextId(2),
            kind: ContextKind::GuestFiber,
            interruptible: true,
        };
        let slot1 = match idx.register_futex(k, s1, || true) {
            RegisteredPark::Parked { slot, decision } => {
                assert_eq!(decision, ParkDecision::Blocked(WakeReason::Ready));
                slot
            }
            _ => panic!("expected parked"),
        };
        let slot2 = match idx.register_futex(k, s2, || true) {
            RegisteredPark::Parked { slot, decision } => {
                assert_eq!(decision, ParkDecision::YieldFiber);
                slot
            }
            _ => panic!("expected parked fiber"),
        };
        assert_eq!(idx.stripe_total(), 2);

        // FIFO: wakes slot1 first, slot2 stays parked.
        assert_eq!(idx.notify(k, 1, WakeReason::Ready), 1);
        assert_eq!(slot1.poll(), Some(WakeReason::Ready));
        assert_eq!(slot2.poll(), None);
        assert_eq!(idx.stripe_total(), 1);

        // Second notify wakes slot2.
        assert_eq!(idx.notify(k, 1, WakeReason::Ready), 1);
        assert_eq!(slot2.poll(), Some(WakeReason::Ready));
        assert_eq!(idx.stripe_total(), 0);
        assert_eq!(idx.total_parked(), 0);
    }

    #[test]
    fn domain_scoped_wake_never_touches_other_domains() {
        let idx = WaitIndex::default();
        let k = WaitKey::Object(WaitableId(99));
        let d0_intr = WaiterSpec {
            domain: DomainId(0),
            ctx: ContextId(1),
            kind: ContextKind::Root,
            interruptible: true,
        };
        let d0_non_intr = WaiterSpec {
            domain: DomainId(0),
            ctx: ContextId(2),
            kind: ContextKind::Thread,
            interruptible: false,
        };
        let d1_intr = WaiterSpec {
            domain: DomainId(1),
            ctx: ContextId(1),
            kind: ContextKind::Root,
            interruptible: true,
        };
        let slot_d0_i = match idx.register_object(k, d0_intr, || true) {
            RegisteredPark::Parked { slot, .. } => slot,
            _ => unreachable!(),
        };
        let slot_d0_n = match idx.register_object(k, d0_non_intr, || true) {
            RegisteredPark::Parked { slot, .. } => slot,
            _ => unreachable!(),
        };
        let slot_d1_i = match idx.register_object(k, d1_intr, || true) {
            RegisteredPark::Parked { slot, .. } => slot,
            _ => unreachable!(),
        };

        // Interrupt domain 0 only: only d0's interruptible waiter wakes.
        assert_eq!(
            idx.wake_domain(DomainId(0), WakeReason::Interrupted, true),
            1
        );
        assert_eq!(slot_d0_i.poll(), Some(WakeReason::Interrupted));
        assert_eq!(slot_d0_n.poll(), None);
        assert_eq!(slot_d1_i.poll(), None);

        // End domain 0: d0's non-interruptible waiter wakes; domain 1 stays untouched!
        assert_eq!(idx.wake_domain(DomainId(0), WakeReason::Revoked, false), 1);
        assert_eq!(slot_d0_n.poll(), Some(WakeReason::Revoked));
        assert_eq!(slot_d1_i.poll(), None);
    }

    #[test]
    fn stripe_is_cache_line_aligned() {
        assert_eq!(std::mem::align_of::<Stripe>(), 128);
        assert_eq!(std::mem::size_of::<Stripe>(), 128);
    }
}
