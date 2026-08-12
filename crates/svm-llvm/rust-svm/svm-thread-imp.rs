//! Real `std::thread::spawn`/`join` for the threaded svm target (copied to `sys/thread/svm.rs`).
//!
//! Modeled on `sys/thread/hermit.rs` — the "single C-like entry taking a raw pointer, join by
//! handle" shape — over the svm §12 thread primitives the on-ramp lowers (`crates/svm-llvm`
//! `lower_vm_builtin`):
//!   * `__vm_thread_spawn(entry, stack, arg) -> handle` → `Inst::ThreadSpawn` (start a vCPU running
//!     `entry(stack, arg)`; `entry` must be a **direct** function symbol, `stack` is the new vCPU's
//!     data-stack base, `arg` is passed through).
//!   * `__vm_thread_join(handle) -> i64` → `Inst::ThreadJoin` (block until that vCPU finishes).
//!   * `__vm_vcpu_tls_set(base)` / `__vm_tls_size()` — the NIM.md §3d Tier-2 per-vCPU TLS block: a
//!     spawned thread allocates + zeroes a `__vm_tls_size()`-byte block and points `vcpu.tls` at it,
//!     so its thread-locals are isolated from every other vCPU's.
//!
//! Only compiled for `x86_64-unknown-svm-threads` (`target_env = "threads"`); the lean target keeps
//! `sys/thread/unsupported.rs` (spawn fails closed).
//!
//! Cooperative-scheduler notes: `set_name` is inert and `current_os_id` is `None` (no per-vCPU name
//! or OS-tid surface). `yield_now`/`sleep` ride the §12 futex — there is no dedicated yield/sleep op,
//! but a `memory.wait` on a private word with `value == expected` parks the vCPU (yielding to other
//! runnable vCPUs) until its deadline, which is exactly a cooperative yield / timed sleep. Spawn/join
//! + futex-backed `sys/sync` are the load-bearing surface.

use crate::alloc::{Layout, alloc, dealloc};
use crate::ffi::CStr;
use crate::io;
use crate::num::NonZero;
use crate::ptr;
use crate::thread::ThreadInit;
use crate::time::Duration;

pub const DEFAULT_MIN_STACK_SIZE: usize = 1 << 20; // 1 MiB, as on hermit.

unsafe extern "C" {
    fn __vm_thread_spawn(entry: extern "C" fn(usize) -> i64, stack: *mut u8, arg: usize) -> i32;
    fn __vm_thread_join(handle: i32) -> i64;
    fn __vm_vcpu_tls_set(base: usize);
    fn __vm_tls_size() -> usize;
    fn __vm_tls_template() -> *const u8;
    // §12 futex wait: park this vCPU while `*addr == expected`, until a notify or `timeout_ns`
    // nanoseconds elapse. Returns 0 (woken), 1 (value mismatch — no wait), 2 (timed out). Backs
    // `yield_now`/`sleep` here; the same op backs `sys/sync/futex/svm.rs`.
    fn __vm_wait32(addr: *mut i32, expected: i32, timeout_ns: i64) -> i32;
}

/// Park the current vCPU for `timeout_ns` nanoseconds by futex-waiting on a private stack word whose
/// value matches `expected`, so no notify can target it — it only wakes on the deadline. `timeout_ns
/// == 0` yields exactly one scheduler turn (a park with an already-reached deadline). A spurious wake
/// (`0`, possible only under the parallel driver's real futex) re-parks, so the vCPU sleeps at least
/// the requested time.
fn futex_park(timeout_ns: i64) {
    // A fresh, never-notified word: value 0, expected 0 → the wait parks rather than returning early.
    let word: i32 = 0;
    let addr = &word as *const i32 as *mut i32;
    loop {
        // SAFETY: `addr` is an aligned, in-window i32; the op only reads it and parks.
        let r = unsafe { __vm_wait32(addr, 0, timeout_ns) };
        // 0 = spurious/notify wake (re-park); 2 = timed out; 1 = mismatch (unreachable here). Only a
        // spurious wake loops — and a zero timeout never parks long enough to be worth re-parking.
        if r != 0 || timeout_ns == 0 {
            break;
        }
    }
}

/// A spawned vCPU and the data-stack region allocated for it (freed on `join`).
pub struct Thread {
    handle: i32,
    stack: *mut u8,
    stack_layout: Layout,
}

// A `Thread` is just a handle + an owning stack pointer; the vCPU it names runs independently.
unsafe impl Send for Thread {}
unsafe impl Sync for Thread {}

impl Thread {
    /// # Safety
    /// See `thread::Builder::spawn_unchecked`.
    pub unsafe fn new(stack: usize, init: Box<ThreadInit>) -> io::Result<Thread> {
        // The new vCPU runs on its own in-window data stack (frames grow upward from the base), so
        // its address-taken locals never collide with another vCPU's. Allocate it here; free on join.
        let stack_size = stack.max(DEFAULT_MIN_STACK_SIZE);
        let stack_layout = Layout::from_size_align(stack_size, 16)
            .map_err(|_| io::const_error!(io::ErrorKind::Uncategorized, "bad thread stack layout"))?;
        let stack_mem = unsafe { alloc(stack_layout) };
        if stack_mem.is_null() {
            return Err(io::const_error!(
                io::ErrorKind::Uncategorized,
                "unable to allocate a thread stack"
            ));
        }

        let data = Box::into_raw(init).expose_provenance();
        let handle = unsafe { __vm_thread_spawn(thread_start, stack_mem, data) };
        if handle < 0 {
            // Spawn failed: reclaim the boxed init and the stack.
            unsafe {
                drop(Box::from_raw(ptr::with_exposed_provenance_mut::<ThreadInit>(data)));
                dealloc(stack_mem, stack_layout);
            }
            return Err(io::const_error!(
                io::ErrorKind::Uncategorized,
                "unable to create thread!"
            ));
        }
        return Ok(Thread { handle, stack: stack_mem, stack_layout });

        extern "C" fn thread_start(data: usize) -> i64 {
            unsafe {
                // NIM.md §3d Tier-2: give this vCPU its own TLS block, initialized from the pristine
                // template (so non-zero thread-local initializers are honored), and point `vcpu.tls` at
                // it before any thread-local is touched — so `vcpu.tls.get() + off` is isolated. The
                // block is freed at the very end, once nothing reads a thread-local again.
                let mut tls_block: Option<(*mut u8, Layout)> = None;
                let tls_size = __vm_tls_size();
                if tls_size > 0 {
                    if let Ok(layout) = Layout::from_size_align(tls_size, 16) {
                        let block = alloc(layout);
                        if !block.is_null() {
                            ptr::copy_nonoverlapping(__vm_tls_template(), block, tls_size);
                            __vm_vcpu_tls_set(block.addr());
                            tls_block = Some((block, layout));
                        }
                    }
                }
                let init = Box::from_raw(ptr::with_exposed_provenance_mut::<ThreadInit>(data));
                let rust_start = init.init();
                rust_start();
                // Run the thread's TLS destructors and the std runtime cleanup, as every platform's
                // `thread_start` does — both may read thread-locals, so they precede the block free.
                crate::sys::thread_local::destructors::run();
                crate::rt::thread_cleanup();
                // Nothing touches a thread-local past this point (the vCPU is about to exit), so the
                // per-thread TLS block can be reclaimed — no leak.
                if let Some((block, layout)) = tls_block {
                    dealloc(block, layout);
                }
            }
            0
        }
    }

    pub fn join(self) {
        // `__vm_thread_join` blocks until the vCPU finishes, so the stack is safe to free afterward.
        unsafe {
            let _ = __vm_thread_join(self.handle);
            dealloc(self.stack, self.stack_layout);
        }
    }
}

pub fn available_parallelism() -> io::Result<NonZero<usize>> {
    // The embedder chooses cooperative vs parallel scheduling; std reporting 1 is the safe floor.
    Ok(NonZero::new(1).unwrap())
}

pub fn current_os_id() -> Option<u64> {
    None
}

pub fn yield_now() {
    // A zero-timeout park: yield one scheduler turn to other runnable vCPUs, then resume.
    futex_park(0);
}

pub fn set_name(_name: &CStr) {}

pub fn sleep(dur: Duration) {
    // Park until the deadline. On the cooperative driver the wait clock is deterministic (advanced to
    // the earliest deadline only when every vCPU is stuck-waiting), so a sleeping thread yields to
    // runnable ones and resumes in deadline order. Clamp the nanosecond count to the op's i64 range.
    let ns = i64::try_from(dur.as_nanos()).unwrap_or(i64::MAX);
    futex_park(ns);
}
