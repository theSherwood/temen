//! svm PAL: the platform surface for the `x86_64-unknown-svm` target. Mirrors the `unsupported`
//! PAL (svm reaches the host through capability-bound named imports — `write`/`read`/`exit` — not a
//! syscall table, so the pal proper stays minimal), with one addition: `init` captures the
//! powerbox-threaded `argv` so `std::env::args` works (the unix PAL does the same via `args::init`).
#![deny(unsafe_op_in_unsafe_fn)]
use crate::io as std_io;

// SAFETY: must be called only once during runtime initialization.
pub unsafe fn init(argc: isize, argv: *const *const u8, _sigpipe: u8) {
    // The powerbox threads `argc`/`argv` into the guest's `main`; hand them to the args store.
    unsafe { crate::sys::args::init(argc, argv) };
}

// SAFETY: must be called only once during runtime cleanup.
pub unsafe fn cleanup() {}

pub fn unsupported<T>() -> std_io::Result<T> {
    Err(unsupported_err())
}

pub fn unsupported_err() -> std_io::Error {
    std_io::Error::UNSUPPORTED_PLATFORM
}

pub fn abort_internal() -> ! {
    core::intrinsics::abort();
}

/// The bridge to the **POSIX personality** capability the embedder grants by name (`run_with_caps`
/// with `("posix", …)`): `__vm_cap_resolve("posix")` → a handle, then `__vm_host_call(handle, op, …)`
/// drives the op table (svm-posix `OP_*`). This is how the richer `std::sys` surface — `time` here,
/// `fs`/`env` later — reaches the host, distinct from the powerbox stdout/exit streams. Each op has
/// its own wrapper so the `op` argument is a **compile-time constant** at the `__vm_host_call` site
/// (the on-ramp requires it).
pub(crate) mod host {
    use crate::sync::atomic::{AtomicI32, Ordering};

    unsafe extern "C" {
        fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
        fn __vm_host_call(handle: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
    }

    // `-1` = not-yet-resolved sentinel (a real handle is non-negative); resolved once, then cached.
    static POSIX: AtomicI32 = AtomicI32::new(-1);

    /// The `posix` personality handle, or a negative value if the embedder granted none.
    fn posix() -> i32 {
        let cached = POSIX.load(Ordering::Relaxed);
        if cached != -1 {
            return cached;
        }
        let h = unsafe { __vm_cap_resolve(b"posix".as_ptr(), 5) };
        POSIX.store(h, Ordering::Relaxed);
        h
    }

    /// Whether a posix personality is available (a time/fs/env op can be attempted).
    pub(crate) fn have_posix() -> bool {
        posix() >= 0
    }

    /// `clock(clock_id) -> nanos` (svm-posix `OP_CLOCK` = 33). `clock_id == 1` is monotonic, else
    /// realtime. `op` is the literal `33` at the call site.
    #[inline(always)]
    pub(crate) fn clock(clock_id: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 33, clock_id, 0, 0, 0) }
    }
}
