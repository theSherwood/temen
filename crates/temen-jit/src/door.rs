//! The embedder's doors into a run, and the wakers they fire (#2757 slice 3).
//!
//! A run's parked vCPUs and tasks sleep until something wakes them: nothing re-checks on a timer.
//! So every store that can end a park fires the wakers registered on what it stores into. The
//! kill door ([`Interrupt`]) and a freeze ([`crate::FreezeController`]) are such stores: each holds
//! the wakers of the runs it reaches, and a run registers its own for as long as it runs.

use core::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// What a store calls to wake a park: it notifies whatever the park sleeps on.
pub(crate) type Waker = Box<dyn Fn() + Send + Sync>;

/// A set of wakers that one kind of store fires ([`Self::fire`]).
///
/// A waker runs under the set's lock, and its owner removes it under the same lock before it
/// frees what the waker names. So a waker may hold raw pointers into its owner, and no firing
/// outlives them. A waker must not register into, or fire, the set it is in.
#[derive(Default)]
pub(crate) struct Wakers {
    list: Mutex<Vec<(u64, Waker)>>,
    next: AtomicU64,
}

impl Wakers {
    /// Call every waker in the set.
    pub(crate) fn fire(&self) {
        let g = self.list.lock().unwrap_or_else(|e| e.into_inner());
        for (_, w) in g.iter() {
            w();
        }
    }

    /// Add `w`; [`Self::remove`] takes the id back.
    pub(crate) fn add(&self, w: Waker) -> u64 {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        self.list
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push((id, w));
        id
    }

    /// Remove waker `id`. Once this returns, no firing is still running it.
    pub(crate) fn remove(&self, id: u64) {
        self.list
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|(i, _)| *i != id);
    }

    /// #2953 — drop every waker, at a run's end: its registrations are its own run's.
    pub(crate) fn clear(&self) {
        self.list.lock().unwrap_or_else(|e| e.into_inner()).clear();
    }

    /// [`Self::add`] for as long as the returned registration lives.
    pub(crate) fn register(&self, w: Waker) -> Registration<'_> {
        Registration {
            wakers: self,
            id: self.add(w),
        }
    }
}

/// A waker's place in the set a store fires, which it leaves when dropped
/// ([`crate::wake_on_run_event`]).
pub struct Registration<'a> {
    wakers: &'a Wakers,
    id: u64,
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        self.wakers.remove(self.id);
    }
}

/// §5 — the embedder's **kill door** for JIT runs: the interrupt cell compiled code polls at its
/// safepoints (loop back-edges, function entries), together with the wakers of the runs that poll
/// it. [`Self::fire`] sets the cell, so a running vCPU traps `OutOfFuel` at its next safepoint, and
/// wakes every parked vCPU and task of those runs, so each unwinds too.
///
/// It replaces a bare cell the embedder stored into, which reached a parked vCPU only when that
/// vCPU next re-checked it, every 20 ms (#2757): the store and its wake are now one operation, so
/// no caller can forget the wake. A run keeps the door it was compiled with alive.
#[repr(C)]
pub struct Interrupt {
    /// Nonzero once fired. First, so a call's context, which holds the cell's address
    /// ([`crate::VmCtx::epoch`]), names the door too ([`Self::of_cell`]).
    cell: AtomicU64,
    /// The runs and parks that poll [`Self::cell`].
    wakers: Wakers,
}

impl Interrupt {
    /// A door that has not fired, for one or more runs.
    pub fn new() -> Arc<Interrupt> {
        Arc::new(Interrupt {
            cell: AtomicU64::new(0),
            wakers: Wakers::default(),
        })
    }

    /// Kill every run compiled with this door: each ends as one whose fuel ran out
    /// ([`crate::TrapKind::OutOfFuel`]). Its running vCPUs stop at their next safepoint, and its
    /// parked vCPUs and tasks are woken to stop at once. Firing twice does nothing more.
    pub fn fire(&self) {
        self.cell.store(1, Ordering::SeqCst);
        self.wakers.fire();
    }

    /// Whether the door has fired.
    pub fn fired(&self) -> bool {
        self.cell.load(Ordering::SeqCst) != 0
    }

    /// The cell compiled code polls.
    pub(crate) fn cell(&self) -> *const AtomicU64 {
        &self.cell
    }

    /// The wakers [`Self::fire`] calls.
    pub(crate) fn wakers(&self) -> &Wakers {
        &self.wakers
    }

    /// The door whose cell is at `cell`, as a call's context names it.
    ///
    /// # Safety
    /// `cell` is nonzero and is the cell of a live `Interrupt`, as every armed run's
    /// [`crate::VmCtx::epoch`] is: the run's module holds its door.
    pub(crate) unsafe fn of_cell<'a>(cell: usize) -> &'a Interrupt {
        &*(cell as *const Interrupt)
    }
}
