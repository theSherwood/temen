//! W5 slice-2 probe (NIM.md §3e): run the **real** `svm_leng::translate_to_text` as an svm guest.
//!
//! A powerbox program — `main` translates a fixed Leng module to SVM text and returns a checksum of
//! that text. A `std` crate (svm-leng is `std`), with a guest `#[global_allocator]` over the
//! on-ramp-synthesized `malloc`/`free` and `panic = abort`. The point is to learn whether a *real
//! std Rust translator* lowers cleanly through the LLVM on-ramp once its only genuine external is the
//! allocator (getrandom is gone — svm-leng now hashes with a seedless FNV, `dethash.rs`).
use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn malloc(n: usize) -> *mut u8;
    fn free(p: *mut u8);
}

struct Guest;
unsafe impl GlobalAlloc for Guest {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        malloc(l.size())
    }
    unsafe fn dealloc(&self, p: *mut u8, _l: Layout) {
        free(p)
    }
}
#[global_allocator]
static A: Guest = Guest;

/// A fixed Leng module: `main() -> int` returning `3 + 4*2 = 11`. Small, but it drives the real
/// translator — nif parse, the `Translator`, SVM-text emission (String/Vec/HashMap throughout).
const LENG: &str =
    "(stmts (proc :main.0. . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";

/// FNV-1a of the emitted SVM text — a single i64 the harness can compare against the host-side
/// translation of the same module.
pub fn checksum() -> i64 {
    match svm_leng::translate_to_text(LENG) {
        Ok(text) => {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for b in text.bytes() {
                h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
            }
            (h & 0x7fff_ffff_ffff_ffff) as i64
        }
        Err(_) => -1,
    }
}

#[no_mangle]
pub extern "C" fn main() -> i64 {
    checksum()
}
