//! The `temen-leng.temen` guest (W5 capstone, NIM.md §3e): the real `temen-leng` translator, wrapped as a
//! **powerbox program**. It reads a Leng-NIF module from stdin, translates it with
//! `temen_leng::translate_to_text`, and writes the emitted Temen text to stdout — the leng analog of
//! `chibicc.temen` (which reads C and emits Temen IR). Built once through the LLVM on-ramp into the
//! committed `temen-leng.temen`; a translation bug is a clean error (temen-verify re-checks it), never an
//! escape (DESIGN.md §2a).
//!
//! Two guest-libc-free choices keep the on-ramp surface minimal (only `read`/`write`/`bcmp` stay
//! external, all recognized by the on-ramp):
//! - a **static-arena bump `#[global_allocator]`** — no `malloc`/`free`/`calloc` externs to synthesize;
//! - **raw `extern "C"` `read`/`write`** (fd 0/1) instead of `std::io::{stdin,stdout}`, which would pull
//!   in the locked/buffered stdio machinery (`pthread_key_*`, `syscall`, `__errno_location`).

use core::alloc::{GlobalAlloc, Layout};

/// 32 MiB is comfortably above the largest real hexer module's translation working set (the biggest
/// committed corpus translates in a few hundred KiB); the guest never frees, so this is a high-water
/// bound, not steady-state use.
const ARENA: usize = 32 * 1024 * 1024;
static mut HEAP: [u8; ARENA] = [0; ARENA];
static mut OFF: usize = 0;

struct Bump;
unsafe impl GlobalAlloc for Bump {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let a = l.align();
        let s = (OFF + a - 1) & !(a - 1);
        if s + l.size() > ARENA {
            return core::ptr::null_mut();
        }
        OFF = s + l.size();
        (core::ptr::addr_of_mut!(HEAP) as *mut u8).add(s)
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {}
}
#[global_allocator]
static A: Bump = Bump;

extern "C" {
    fn read(fd: i32, buf: *mut u8, n: usize) -> isize;
    fn write(fd: i32, buf: *const u8, n: usize) -> isize;
}

/// Drain stdin (the powerbox `stdin` Stream cap) to a `Vec` — the Leng-NIF source.
fn read_stdin() -> Vec<u8> {
    let mut v = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let n = unsafe { read(0, chunk.as_mut_ptr(), chunk.len()) };
        if n <= 0 {
            break;
        }
        v.extend_from_slice(&chunk[..n as usize]);
    }
    v
}

/// Write the whole buffer to stdout (the powerbox `stdout` Stream cap), retrying short writes.
fn write_stdout(b: &[u8]) {
    let mut off = 0;
    while off < b.len() {
        let n = unsafe { write(1, b[off..].as_ptr(), b.len() - off) };
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
}

/// `main` — the powerbox entry the on-ramp wraps in a synthesized `_start`. Exit codes: `1` = stdin
/// was not UTF-8, `2` = the translator refused the module (an `Unsupported`/`Malformed` Leng
/// construct), `0` = the Temen text was emitted to stdout.
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let src = read_stdin();
    let src = match core::str::from_utf8(&src) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    match temen_leng::translate_to_text(src) {
        Ok(text) => {
            write_stdout(text.as_bytes());
            0
        }
        Err(_) => 2,
    }
}
