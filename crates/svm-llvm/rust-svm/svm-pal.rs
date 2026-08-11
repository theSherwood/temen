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
