//! The temen futex primitive (copied to `sys/sync/futex/temen.rs` by `apply-overlay.sh`).
//!
//! A near-mechanical port of std's `sys/sync/futex/wasm.rs`: the same "an atomic word is a futex"
//! contract, but the wait/notify reach the temen §12 futex through the on-ramp intrinsics
//! `__vm_wait32` / `__vm_notify` (temen-llvm `lower_vm_builtin`) instead of the wasm
//! `memory.atomic.wait32` / `memory.atomic.notify` opcodes. The IR ops these lower to
//! (`MemoryWait` / `MemoryNotify`, temen-ir §12) share wasm's semantics exactly:
//!   * `__vm_wait32(addr, expected, timeout_ns) -> i32`: `0` woken / `1` not-equal / `2` timed-out.
//!   * `__vm_notify(addr, count) -> i32`: the number of vCPUs actually woken.
//! so the wasm module's return-value encoding (`< 2` = "not a timeout", `> 0` = "woke someone")
//! carries over unchanged. `timeout = -1` (all bits set) is the "no timeout" sentinel, as in wasm.
//!
//! Only compiled for the threaded target (`x86_64-unknown-temen-threads`, `target_env = "threads"`);
//! the lean single-threaded `x86_64-unknown-temen` target keeps std's `no_threads` sync and never
//! references this module.

use crate::sync::atomic::Atomic;
use crate::time::Duration;

unsafe extern "C" {
    /// §12 futex wait — the on-ramp lowers this to `Inst::MemoryWait { ty: i32, .. }`.
    /// Returns 0 (woken), 1 (value mismatch), or 2 (timed out).
    fn __vm_wait32(addr: *mut i32, expected: i32, timeout_ns: i64) -> i32;
    /// §12 futex notify — the on-ramp lowers this to `Inst::MemoryNotify`.
    /// Returns the number of vCPUs woken (`count` is the unsigned "wake up to N" bound).
    fn __vm_notify(addr: *mut i32, count: i32) -> i32;
}

/// An atomic for use as a futex that is at least 32-bits but may be larger.
pub type Futex = Atomic<Primitive>;
/// Must be the underlying type of Futex.
pub type Primitive = u32;

/// An atomic for use as a futex that is at least 8-bits but may be larger.
pub type SmallFutex = Atomic<SmallPrimitive>;
/// Must be the underlying type of SmallFutex.
pub type SmallPrimitive = u32;

/// Wait for a `futex_wake` operation to wake us.
///
/// Returns directly if the futex doesn't hold the expected value.
///
/// Returns false on timeout, and true in all other cases.
pub fn futex_wait(futex: &Atomic<u32>, expected: u32, timeout: Option<Duration>) -> bool {
    let timeout = timeout.and_then(|t| t.as_nanos().try_into().ok()).unwrap_or(-1);
    // `< 2`: 0 (woken) and 1 (mismatch) both return `true`; only 2 (timeout) returns `false`.
    unsafe { __vm_wait32(futex as *const Atomic<u32> as *mut i32, expected as i32, timeout) < 2 }
}

/// Wakes up one thread that's blocked on `futex_wait` on this futex.
///
/// Returns true if this actually woke up such a thread,
/// or false if no thread was waiting on this futex.
pub fn futex_wake(futex: &Atomic<u32>) -> bool {
    unsafe { __vm_notify(futex as *const Atomic<u32> as *mut i32, 1) > 0 }
}

/// Wakes up all threads that are waiting on `futex_wait` on this futex.
pub fn futex_wake_all(futex: &Atomic<u32>) {
    // `count = -1` reinterpreted as `u32::MAX` = "wake all" (temen-ir `MemoryNotify` caps at the real
    // waiter count), matching wasm's `memory.atomic.notify(_, i32::MAX)` wake-all idiom.
    unsafe {
        __vm_notify(futex as *const Atomic<u32> as *mut i32, i32::MAX);
    }
}
