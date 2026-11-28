//! [`Domain`], [`DomainControl`], generational handle packing (Invariant 11), and
//! single-lock guest-fiber ownership/migration state transitions (`#3187` / `#3194`).
//!
//! ## Key invariants upheld here
//! - **Invariant 6 (One world / one lifetime):** `Domain` holds one shared `poll_flags: AtomicU64`
//!   and one terminal `Option<ContextExit>`. The first trap, `exit`, or root completion sets
//!   `PollFlags::ENDED` and wakes every waiter in this domain with `WakeReason::Revoked`.
//! - **Invariant 11 (Top byte belongs to guest tag):** 64-bit fiber handles lay out as
//!   `[tag:8][gen:32][slot:24]`. `unpack_fiber_handle` explicitly masks the generation to 32 bits
//!   (`(h >> 24) & 0xFFFF_FFFF`), so any guest pointer tag in bits 56..63 is strictly inert.
//! - **Decision 21 (Lock-free single-carrier fast path):** `multi_carrier: AtomicBool` starts
//!   `false` and flips to `true` with `Release` ordering only when a second Carrier attaches
//!   (`thread.spawn` or concurrent offer admission).
//! - **Red-team refinement #3 (`FIBER_PARKED`, status 2):** A `GuestFiber` parked in `WaitIndex`
//!   transitions `Running -> Parked`, returns `FIBER_PARKED` to its resumer, and becomes `Ready`
//!   when woken so *any* Carrier can claim and resume it exclusively (`FiberResumeClaim::Claimed`).

use crate::budget::BudgetId;
use crate::sync::{lock_unpoisoned, Arc, AtomicBool, AtomicU64, Mutex, Ordering};
use crate::types::{
    CarrierId, ContextExit, ContextId, ContextKind, ContextState, DomainId, LifecycleHooks,
    PollAction, PollFlags, Trap, WaitKey, WakeReason,
};
use crate::wait::WaitIndex;

/// Default no-op lifecycle hooks (used when no personality such as `temen-posix` is attached).
pub struct NoopLifecycleHooks;

impl LifecycleHooks for NoopLifecycleHooks {}

/// Slot bit width for 64-bit fiber handles (`[tag:8][gen:32][slot:24]` — Invariant 11).
const FIBER_SLOT_BITS: u32 = 24;
const FIBER_SLOT_MASK: u64 = (1u64 << FIBER_SLOT_BITS) - 1;
const FIBER_GEN_MASK: u64 = 0xFFFF_FFFF;

/// Pack a `(slot, generation)` pair into a 64-bit guest fiber handle obeying Invariant 11:
/// bits 0..24 = slot, bits 24..56 = generation (32 bits), bits 56..64 = 0 (reserved for guest tag).
#[inline]
pub const fn pack_fiber_handle(slot: u32, gen: u32) -> i64 {
    let lo = (slot as u64) & FIBER_SLOT_MASK;
    let mid = ((gen as u64) & FIBER_GEN_MASK) << FIBER_SLOT_BITS;
    (lo | mid) as i64
}

/// Unpack a guest fiber handle, masking out the top byte (bits 56..63) per Invariant 11 so any
/// guest pointer tag is strictly inert.
#[inline]
pub const fn unpack_fiber_handle(handle: i64) -> (u32, u32) {
    let raw = handle as u64;
    let slot = (raw & FIBER_SLOT_MASK) as u32;
    let gen = ((raw >> FIBER_SLOT_BITS) & FIBER_GEN_MASK) as u32;
    (slot, gen)
}

/// Generational handle slot for a thread or fiber context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct HandleSlot {
    pub slot: u32,
    pub gen: u32,
}

/// Exclusive ownership / migration state of a `GuestFiber` (replaces `fiber_registry.rs`'s
/// separate atomic + 4-lock-hold protocol with one state transition under the domain lock).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FiberSlotState {
    /// Created via `cont.new`, never resumed yet (or yielded via `suspend`), ready to be claimed.
    Suspended,
    /// Currently executing on `carrier`. Any concurrent `cont.resume` sees `FiberFault`.
    Running { carrier: CarrierId },
    /// Parked in [`WaitIndex`] (returned `FIBER_PARKED` status 2 to its resumer). Not yet woken.
    Parked { key: WaitKey },
    /// Was parked in `WaitIndex` and has now been woken with `reason`; ready for any Carrier's
    /// next `cont.resume` to claim and re-execute (Invariant 7).
    WakeReady { reason: WakeReason },
    /// Completed (returned or trapped); slot is dead and any further `cont.resume` traps
    /// `FiberFault`.
    Done,
}

/// Result of attempting to claim a `GuestFiber` for `cont.resume`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FiberResumeClaim {
    /// Claimed exclusively for `carrier`. `wake` is `Some(reason)` if resuming from a `WakeReady`
    /// park (so the engine re-executes per Invariant 7), or `None` for initial/suspended resume.
    Claimed {
        ctx: ContextId,
        wake: Option<WakeReason>,
    },
    /// Fiber is still parked in `WaitIndex` (`FIBER_PARKED` = 3 — caller gets status 3 without
    /// blocking).
    StillParked,
    /// Handle named a fiber that already returned (`(-EBADF, 0)`, Invariant 5 / `#2667`).
    Spent,
    /// Forged handle (generation never issued) or already running on another Carrier ->
    /// `Trap::FiberFault`.
    Fault,
}

/// Result of polling a `GuestFiber`'s readiness (used by guest schedulers / `cont.poll`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FiberPoll {
    StillParked,
    ClaimedReady,
    NotParked,
    Fault,
}

/// Kernel metadata tracked for every live [`Context`] in a [`Domain`].
#[derive(Clone, Debug)]
pub struct ContextMeta {
    pub id: ContextId,
    pub kind: ContextKind,
    pub state: ContextState,
    /// Optional durable shadow-arena slot (uniform for ALL `ContextKind`s — closes `#2804`).
    pub durable_slot: Option<u32>,
}

struct FiberEntry {
    gen: u32,
    ctx: ContextId,
    state: FiberSlotState,
}

struct DomainTable {
    next_ctx: u32,
    contexts: Vec<ContextMeta>,
    fibers: Vec<Option<FiberEntry>>,
    /// Highest generation ever issued per slot, so we can distinguish a **spent** handle
    /// (`gen <= last_gen` on a finished/recycled slot -> `(-EBADF, 0)`, `#2667`) from a
    /// **forged** handle (`gen == 0` or `gen > last_gen` -> `Trap::FiberFault`).
    fiber_issued_gen: Vec<u32>,
    free_fiber_slots: Vec<u32>,
    next_fiber_gen: u32,
    terminal: Option<ContextExit>,
    paused: bool,
}

/// One isolation domain: one window identity, one lifetime (Invariant 6), one shared safepoint
/// poll word (`poll`), one `multi_carrier` latch (Decision 21), and one context/fiber registry.
pub struct Domain {
    pub id: DomainId,
    pub budget: BudgetId,
    /// Shared safepoint poll word. Read-only on the hot path (L1-shared across Carriers); `VmCtx`
    /// points directly at this `AtomicU64` at offset 0 (red-team refinement #2).
    poll: Arc<AtomicU64>,
    /// Decision 21: `false` while only one Carrier runs this domain (lock-free fast path); flips
    /// `true` (`Release`) before a second Carrier begins executing in this domain.
    multi_carrier: AtomicBool,
    table: Mutex<DomainTable>,
    hooks: std::sync::Arc<dyn LifecycleHooks>,
}

impl Domain {
    /// Create a new domain with its root `ContextId(0)` registered.
    pub fn new(
        id: DomainId,
        budget: BudgetId,
        hooks: std::sync::Arc<dyn LifecycleHooks>,
    ) -> Arc<Domain> {
        let root = ContextMeta {
            id: ContextId(0),
            kind: ContextKind::Root,
            state: ContextState::Runnable,
            durable_slot: None,
        };
        Arc::new(Domain {
            id,
            budget,
            poll: Arc::new(AtomicU64::new(0)),
            multi_carrier: AtomicBool::new(false),
            table: Mutex::new(DomainTable {
                next_ctx: 1,
                contexts: vec![root],
                fibers: Vec::new(),
                fiber_issued_gen: Vec::new(),
                free_fiber_slots: Vec::new(),
                next_fiber_gen: 1,
                terminal: None,
                paused: false,
            }),
            hooks,
        })
    }

    /// Access the shared safepoint poll word (`VmCtx.domain_poll` points here).
    #[inline]
    pub fn poll_word(&self) -> &AtomicU64 {
        &self.poll
    }

    /// Clone the `Arc<AtomicU64>` poll word handle (for engine state / tests).
    pub fn poll_arc(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.poll)
    }

    /// Fast-path read of the current safepoint poll flags.
    #[inline]
    pub fn poll_flags(&self) -> PollFlags {
        PollFlags(self.poll.load(Ordering::Acquire))
    }

    /// Decision 21: whether this domain has transitioned to multi-carrier execution.
    #[inline]
    pub fn is_multi_carrier(&self) -> bool {
        self.multi_carrier.load(Ordering::Acquire)
    }

    /// Latch multi-carrier mode (`Release`) before handing work to a second Carrier.
    #[inline]
    pub fn promote_multi_carrier(&self) {
        self.multi_carrier.store(true, Ordering::Release);
    }

    /// Register a new `Thread` or `ServeHandler` context in this domain.
    pub fn register_context(&self, kind: ContextKind, durable_slot: Option<u32>) -> ContextId {
        let mut t = lock_unpoisoned(&self.table);
        let id = ContextId(t.next_ctx);
        t.next_ctx += 1;
        t.contexts.push(ContextMeta {
            id,
            kind,
            state: ContextState::Runnable,
            durable_slot,
        });
        if matches!(kind, ContextKind::Thread) {
            self.promote_multi_carrier();
        }
        id
    }

    /// Allocate a new `GuestFiber` context and generational handle (`[0:8][gen:32][slot:24]`).
    pub fn create_fiber(&self, durable_slot: Option<u32>) -> (ContextId, i64) {
        let mut t = lock_unpoisoned(&self.table);
        let ctx = ContextId(t.next_ctx);
        t.next_ctx += 1;
        t.contexts.push(ContextMeta {
            id: ctx,
            kind: ContextKind::GuestFiber,
            state: ContextState::Suspended,
            durable_slot,
        });
        let gen = t.next_fiber_gen;
        t.next_fiber_gen = t.next_fiber_gen.wrapping_add(1).max(1);
        let slot = if let Some(free) = t.free_fiber_slots.pop() {
            t.fibers[free as usize] = Some(FiberEntry {
                gen,
                ctx,
                state: FiberSlotState::Suspended,
            });
            t.fiber_issued_gen[free as usize] = gen;
            free
        } else {
            let idx = t.fibers.len() as u32;
            t.fibers.push(Some(FiberEntry {
                gen,
                ctx,
                state: FiberSlotState::Suspended,
            }));
            t.fiber_issued_gen.push(gen);
            idx
        };
        (ctx, pack_fiber_handle(slot, gen))
    }

    /// Exclusively claim a `GuestFiber` for `cont.resume` on `carrier` (masks top guest tag byte
    /// per Invariant 11).
    pub fn claim_fiber_resume(&self, handle: i64, carrier: CarrierId) -> FiberResumeClaim {
        let (slot, gen) = unpack_fiber_handle(handle);
        if gen == 0 {
            return FiberResumeClaim::Fault;
        }
        let mut t = lock_unpoisoned(&self.table);
        let Some(&issued) = t.fiber_issued_gen.get(slot as usize) else {
            return FiberResumeClaim::Fault;
        };
        if gen > issued {
            // Generation never issued for this slot -> forged handle traps FiberFault.
            return FiberResumeClaim::Fault;
        }
        let Some(Some(entry)) = t.fibers.get_mut(slot as usize) else {
            // Slot previously issued this generation and is now finished -> spent (-EBADF, #2667).
            return FiberResumeClaim::Spent;
        };
        if entry.gen != gen {
            // Earlier generation of a recycled slot -> spent (-EBADF, #2667).
            return FiberResumeClaim::Spent;
        }
        match entry.state {
            FiberSlotState::Suspended => {
                entry.state = FiberSlotState::Running { carrier };
                let ctx = entry.ctx;
                if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
                    m.state = ContextState::Running { carrier };
                }
                FiberResumeClaim::Claimed { ctx, wake: None }
            }
            FiberSlotState::WakeReady { reason } => {
                entry.state = FiberSlotState::Running { carrier };
                let ctx = entry.ctx;
                if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
                    m.state = ContextState::Running { carrier };
                }
                FiberResumeClaim::Claimed {
                    ctx,
                    wake: Some(reason),
                }
            }
            FiberSlotState::Parked { .. } => FiberResumeClaim::StillParked,
            FiberSlotState::Running { .. } => FiberResumeClaim::Fault,
            FiberSlotState::Done => FiberResumeClaim::Spent,
        }
    }

    /// Transition a running `GuestFiber` back to `Suspended` (when it executes `suspend`).
    pub fn suspend_fiber(&self, handle: i64) -> bool {
        let (slot, gen) = unpack_fiber_handle(handle);
        let mut t = lock_unpoisoned(&self.table);
        let Some(Some(entry)) = t.fibers.get_mut(slot as usize) else {
            return false;
        };
        if entry.gen != gen || !matches!(entry.state, FiberSlotState::Running { .. }) {
            return false;
        }
        entry.state = FiberSlotState::Suspended;
        let ctx = entry.ctx;
        if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
            m.state = ContextState::Suspended;
        }
        true
    }

    /// Mark a running `GuestFiber` as `Parked { key }` (`FIBER_PARKED` = 3 — red-team #3).
    /// Returns `false` if a concurrent wake already promoted it to `WakeReady` before park
    /// publication completed!
    pub fn mark_fiber_parked(&self, handle: i64, key: WaitKey) -> bool {
        let (slot, gen) = unpack_fiber_handle(handle);
        let mut t = lock_unpoisoned(&self.table);
        let Some(Some(entry)) = t.fibers.get_mut(slot as usize) else {
            return false;
        };
        if entry.gen != gen {
            return false;
        }
        match entry.state {
            FiberSlotState::Running { .. } => {
                entry.state = FiberSlotState::Parked { key };
                let ctx = entry.ctx;
                if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
                    m.state = ContextState::Parked {
                        key,
                        deadline: None,
                        interruptible: true,
                    };
                }
                true
            }
            FiberSlotState::WakeReady { .. } => false,
            _ => false,
        }
    }

    /// Notify a parked `GuestFiber` (by `ContextId`, invoked from `WaitIndex`'s wake path) so any
    /// Carrier can claim and resume it next.
    pub fn wake_fiber_by_ctx(&self, ctx: ContextId, reason: WakeReason) -> bool {
        let mut t = lock_unpoisoned(&self.table);
        for entry in t.fibers.iter_mut().flatten() {
            if entry.ctx == ctx {
                if matches!(
                    entry.state,
                    FiberSlotState::Parked { .. } | FiberSlotState::Running { .. }
                ) {
                    entry.state = FiberSlotState::WakeReady { reason };
                    if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
                        m.state = ContextState::Runnable;
                    }
                    return true;
                }
                return false;
            }
        }
        false
    }

    /// Mark a `GuestFiber` completed and recycle its slot with a fresh generation so stale handles
    /// return `Spent` (`-EBADF`, `#2667`) and unissued handles trap `FiberFault`.
    pub fn finish_fiber(&self, handle: i64) -> bool {
        let (slot, gen) = unpack_fiber_handle(handle);
        let mut t = lock_unpoisoned(&self.table);
        let Some(entry_opt) = t.fibers.get_mut(slot as usize) else {
            return false;
        };
        let Some(entry) = entry_opt else {
            return false;
        };
        if entry.gen != gen {
            return false;
        }
        let ctx = entry.ctx;
        *entry_opt = None;
        t.free_fiber_slots.push(slot);
        if let Some(m) = t.contexts.get_mut(ctx.0 as usize) {
            m.state = ContextState::Done;
        }
        true
    }

    /// Check whether this domain has reached a terminal exit (Invariant 6).
    pub fn terminal_exit(&self) -> Option<ContextExit> {
        lock_unpoisoned(&self.table).terminal.clone()
    }

    /// Record the domain's first terminal exit and set `PollFlags::ENDED`. Returns `true` if this
    /// call was the first to terminate the domain (invoking `LifecycleHooks::on_domain_exit`).
    pub fn record_terminal(&self, exit: ContextExit) -> bool {
        let first = {
            let mut t = lock_unpoisoned(&self.table);
            if t.terminal.is_some() {
                false
            } else {
                t.terminal = Some(exit.clone());
                true
            }
        };
        self.poll.fetch_or(PollFlags::ENDED, Ordering::Release);
        if first {
            self.hooks.on_domain_exit(self.id, &exit);
        }
        first
    }

    /// Interpret non-zero [`PollFlags`] observed at a safepoint.
    pub fn evaluate_poll(&self, flags: PollFlags) -> PollAction {
        if flags.contains(PollFlags::ENDED) {
            let trap = self
                .terminal_exit()
                .and_then(|e| e.as_trap())
                .unwrap_or(Trap::OutOfFuel);
            return PollAction::Terminate(trap);
        }
        if flags.contains(PollFlags::KILL) {
            return PollAction::Terminate(Trap::OutOfFuel);
        }
        if flags.contains(PollFlags::FREEZE) {
            return PollAction::Freeze;
        }
        if flags.contains(PollFlags::PAUSED) {
            return PollAction::Pause;
        }
        if flags.contains(PollFlags::UPCALL) {
            if let Some((func_idx, arg, alt_data_sp)) = self.hooks.take_upcall(self.id) {
                return PollAction::Upcall {
                    func_idx,
                    arg,
                    alt_data_sp,
                };
            }
            // No pending upcall left — clear the bit cleanly.
            self.poll.fetch_and(!PollFlags::UPCALL, Ordering::AcqRel);
        }
        PollAction::Continue
    }

    /// Access this domain's [`LifecycleHooks`].
    pub fn hooks(&self) -> &std::sync::Arc<dyn LifecycleHooks> {
        &self.hooks
    }
}

/// Cloneable control handle handed to personalities / embedders with the 4 OS-neutral verbs
/// (`#3187` decision 6): `set_paused`, `interrupt_parks`, `arm_upcall`, `end`.
#[derive(Clone)]
pub struct DomainControl {
    domain: Arc<Domain>,
    waits: Arc<WaitIndex>,
}

impl DomainControl {
    pub fn new(domain: Arc<Domain>, waits: Arc<WaitIndex>) -> DomainControl {
        DomainControl { domain, waits }
    }

    pub fn domain_id(&self) -> DomainId {
        self.domain.id
    }

    /// Verb 1: pause or resume all contexts in this domain (job control `SIGSTOP`/`SIGCONT`,
    /// debugger pause, or freeze quiesce).
    pub fn set_paused(&self, paused: bool) {
        {
            let mut t = lock_unpoisoned(&self.domain.table);
            t.paused = paused;
        }
        if paused {
            self.domain
                .poll
                .fetch_or(PollFlags::PAUSED, Ordering::Release);
            self.waits
                .wake_domain(self.domain.id, WakeReason::Paused, false);
        } else {
            self.domain
                .poll
                .fetch_and(!PollFlags::PAUSED, Ordering::Release);
        }
    }

    /// Verb 2: interrupt all interruptible parks in this domain (`^C` / `-EINTR`).
    /// Strictly scoped to `self.domain.id` (Invariant 12).
    pub fn interrupt_parks(&self) -> u32 {
        self.waits
            .wake_domain(self.domain.id, WakeReason::Interrupted, true)
    }

    /// Verb 3: arm an async upcall at the next safepoint (e.g., Bash `SIGINT` trap).
    pub fn arm_upcall(&self) {
        self.domain
            .poll
            .fetch_or(PollFlags::UPCALL, Ordering::Release);
    }

    /// Verb 4: terminate this domain with `exit` (Invariant 6: one lifetime) and revoke all
    /// domain waiters so none can hang.
    pub fn end(&self, exit: ContextExit) -> bool {
        let first = self.domain.record_terminal(exit);
        self.waits
            .wake_domain(self.domain.id, WakeReason::Revoked, false);
        first
    }

    /// Arm a kill / epoch interrupt on this domain.
    pub fn kill(&self) {
        self.domain
            .poll
            .fetch_or(PollFlags::KILL, Ordering::Release);
        self.waits
            .wake_domain(self.domain.id, WakeReason::Revoked, false);
    }

    /// Arm a freeze quiesce request on this domain.
    pub fn request_freeze(&self) {
        self.domain
            .poll
            .fetch_or(PollFlags::FREEZE, Ordering::Release);
        self.waits
            .wake_domain(self.domain.id, WakeReason::Freeze, false);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::types::MemSpaceId;

    #[test]
    fn fiber_handle_ignores_top_guest_tag_byte_invariant_11() {
        let domain = Domain::new(DomainId(0), BudgetId::ROOT, Arc::new(NoopLifecycleHooks));
        let (ctx, handle) = domain.create_fiber(None);
        // Stamp an arbitrary non-zero guest tag into bits 56..63.
        let tagged = ((handle as u64) | (0xABu64 << 56)) as i64;
        assert_eq!(unpack_fiber_handle(handle), unpack_fiber_handle(tagged));
        assert_eq!(
            domain.claim_fiber_resume(tagged, CarrierId(0)),
            FiberResumeClaim::Claimed { ctx, wake: None }
        );
        // Double-resume while running traps FiberFault even with a different tag.
        let tagged2 = ((handle as u64) | (0x55u64 << 56)) as i64;
        assert_eq!(
            domain.claim_fiber_resume(tagged2, CarrierId(1)),
            FiberResumeClaim::Fault
        );
        assert!(domain.finish_fiber(tagged));
        // Spent handle (finished fiber) returns Spent (-EBADF, #2667).
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(0)),
            FiberResumeClaim::Spent
        );
        // Stale generation after slot recycling also returns Spent.
        let (_ctx2, handle2) = domain.create_fiber(None);
        assert_ne!(handle, handle2);
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(0)),
            FiberResumeClaim::Spent
        );
        // Unissued generation or out-of-range slot is forged -> traps FiberFault.
        let forged = pack_fiber_handle(0, 999);
        assert_eq!(
            domain.claim_fiber_resume(forged, CarrierId(0)),
            FiberResumeClaim::Fault
        );
        assert_eq!(
            domain.claim_fiber_resume(pack_fiber_handle(77, 1), CarrierId(0)),
            FiberResumeClaim::Fault
        );
    }

    #[test]
    fn fiber_parked_and_migrated_across_carriers() {
        let domain = Domain::new(DomainId(0), BudgetId::ROOT, Arc::new(NoopLifecycleHooks));
        let (ctx, handle) = domain.create_fiber(None);
        let key = WaitKey::Futex {
            space: MemSpaceId(1),
            offset: 128,
        };
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(0)),
            FiberResumeClaim::Claimed { ctx, wake: None }
        );
        assert!(domain.mark_fiber_parked(handle, key));
        // Polling/resuming while still parked returns StillParked.
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(1)),
            FiberResumeClaim::StillParked
        );
        // Woken by notifier -> Carrier 1 claims and resumes it with WakeReason::Ready.
        assert!(domain.wake_fiber_by_ctx(ctx, WakeReason::Ready));
        assert_eq!(
            domain.claim_fiber_resume(handle, CarrierId(1)),
            FiberResumeClaim::Claimed {
                ctx,
                wake: Some(WakeReason::Ready)
            }
        );
    }

    #[test]
    fn domain_control_verbs_update_poll_word_and_terminal_once() {
        let waits = Arc::new(WaitIndex::default());
        let domain = Domain::new(DomainId(3), BudgetId::ROOT, Arc::new(NoopLifecycleHooks));
        let ctl = DomainControl::new(Arc::clone(&domain), Arc::clone(&waits));

        assert!(domain.poll_flags().is_empty());
        ctl.set_paused(true);
        assert!(domain.poll_flags().contains(PollFlags::PAUSED));
        assert_eq!(domain.evaluate_poll(domain.poll_flags()), PollAction::Pause);
        ctl.set_paused(false);
        assert!(domain.poll_flags().is_empty());

        assert!(ctl.end(ContextExit::Exited(7)));
        assert!(!ctl.end(ContextExit::Trapped(Trap::Unreachable)));
        assert_eq!(domain.terminal_exit(), Some(ContextExit::Exited(7)));
        assert_eq!(
            domain.evaluate_poll(domain.poll_flags()),
            PollAction::Terminate(Trap::Exit(7))
        );
    }
}
