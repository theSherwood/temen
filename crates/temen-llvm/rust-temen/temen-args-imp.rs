//! temen command-line arguments: captured from the powerbox-threaded `argv` at runtime init (the temen
//! PAL's `init` calls [`init`]), then walked as NUL-terminated C strings on demand — the "stored at
//! startup" half of the unix strategy, without the `os::unix` dependency.
pub use super::common::Args;
use crate::ffi::{CStr, OsString};
use crate::ptr;
use crate::sync::atomic::{AtomicIsize, AtomicPtr, Ordering};

static ARGC: AtomicIsize = AtomicIsize::new(0);
static ARGV: AtomicPtr<*const u8> = AtomicPtr::new(ptr::null_mut());

/// One-time global initialization, called from the temen PAL's `init` with the values the powerbox
/// threaded into `main(argc, argv)`.
pub unsafe fn init(argc: isize, argv: *const *const u8) {
    ARGC.store(argc, Ordering::Relaxed);
    ARGV.store(argv as *mut *const u8, Ordering::Relaxed);
}

pub fn args() -> Args {
    let argc = ARGC.load(Ordering::Relaxed);
    let argv = ARGV.load(Ordering::Relaxed) as *const *const u8;
    let mut vec = Vec::with_capacity(argc.max(0) as usize);
    if argv.is_null() {
        return Args::new(vec);
    }
    for i in 0..argc {
        // SAFETY: `argv` holds `argc` valid, NUL-terminated C strings (or a NULL terminator).
        let ptr = unsafe { argv.offset(i).read() };
        if ptr.is_null() {
            break;
        }
        let cstr = unsafe { CStr::from_ptr(ptr.cast()) };
        let bytes = cstr.to_bytes().to_vec();
        // SAFETY: the bytes came from a C string; temen has no OS-specific OsStr encoding.
        vec.push(unsafe { OsString::from_encoded_bytes_unchecked(bytes) });
    }
    Args::new(vec)
}
