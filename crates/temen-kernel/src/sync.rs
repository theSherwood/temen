//! Synchronization primitives abstracted cleanly over `std::sync` and `loom::sync` (`--cfg loom`).
//!
//! Keeping this shim in one place lets the entire `temen-kernel` crate be exhaustively
//! model-checked under `RUSTFLAGS="--cfg loom"` without scattering `#[cfg(loom)]` across the
//! state machine.

#[cfg(loom)]
pub use loom::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(loom)]
pub use loom::sync::{Arc, Condvar, Mutex, MutexGuard};

#[cfg(not(loom))]
pub use std::sync::atomic::{fence, AtomicBool, AtomicU32, AtomicU64, Ordering};
#[cfg(not(loom))]
pub use std::sync::{Arc, Condvar, Mutex, MutexGuard};

/// Lock a mutex, recovering its inner data if a previous holder panicked (no poison cascade).
#[inline]
pub fn lock_unpoisoned<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    match m.lock() {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Wait on a condvar, recovering from poison.
#[inline]
pub fn wait_unpoisoned<'a, T>(cv: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    match cv.wait(g) {
        Ok(g) => g,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Wait on a condvar with a timeout (real builds only; loom does not model wall clocks).
#[cfg(not(loom))]
#[inline]
pub fn wait_timeout_unpoisoned<'a, T>(
    cv: &Condvar,
    g: MutexGuard<'a, T>,
    dur: std::time::Duration,
) -> (MutexGuard<'a, T>, bool) {
    match cv.wait_timeout(g, dur) {
        Ok((g, res)) => (g, res.timed_out()),
        Err(poisoned) => {
            let (g, res) = poisoned.into_inner();
            (g, res.timed_out())
        }
    }
}
