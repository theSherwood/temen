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
//! Cooperative-scheduler notes (fail-safe simplifications, revisited as the model firms up): the
//! per-thread TLS block leaks at thread exit (there is no thread-exit hook yet — the same
//! "leak everything" stance as the wasm/svm TLS guard); `yield_now`/`sleep`/`set_name` are inert;
//! `current_os_id` is `None`. Spawn/join + futex-backed `sys/sync` are the load-bearing surface.

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
                // it before any thread-local is touched — so `vcpu.tls.get() + off` is isolated.
                let tls_size = __vm_tls_size();
                if tls_size > 0 {
                    if let Ok(layout) = Layout::from_size_align(tls_size, 16) {
                        let block = alloc(layout);
                        if !block.is_null() {
                            ptr::copy_nonoverlapping(__vm_tls_template(), block, tls_size);
                            __vm_vcpu_tls_set(block.addr());
                        }
                    }
                    // (The block leaks at thread exit — no thread-exit hook yet.)
                }
                let init = Box::from_raw(ptr::with_exposed_provenance_mut::<ThreadInit>(data));
                let rust_start = init.init();
                rust_start();
                // Run the thread's TLS destructors and the std runtime cleanup, as every platform's
                // `thread_start` does.
                crate::sys::thread_local::destructors::run();
                crate::rt::thread_cleanup();
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

pub fn yield_now() {}

pub fn set_name(_name: &CStr) {}

pub fn sleep(_dur: Duration) {}
