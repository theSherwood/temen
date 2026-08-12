//! W5 slice-2 probe (NIM.md §3e): run the **real** `svm_leng::translate_to_text` as an svm guest.
//!
//! A powerbox program — `main` translates a fixed Leng module to SVM text and returns a checksum of
//! that text. A `std` crate (svm-leng is `std`, built with `float` off so no dragon4/dec2flt), with a
//! guest bump `#[global_allocator]` over a static arena (the first-light/rustbench model — no `malloc`
//! extern to synthesize), and `panic = abort`. Proves a real std Rust translator runs on svm.
use core::alloc::{GlobalAlloc, Layout};

const ARENA: usize = 8 * 1024 * 1024;
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

/// A fixed Leng module: `main() -> int` returning `3 + 4*2 = 11`. Small, but it drives the real
/// translator — nif parse, the `Translator`, SVM-text emission (String/Vec/HashMap throughout).
const LENG: &str =
    "(stmts (proc :main.0. . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";

/// FNV-1a of the emitted SVM text — a single i64 the harness compares against the host-side
/// translation of the same module. Returned truncated to i32 so it rides `main`'s exit code.
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let text = match svm_leng::translate_to_text(LENG) {
        Ok(t) => t,
        Err(_) => return -1,
    };
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.bytes() {
        h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3);
    }
    (h & 0x7fff_ffff) as i32
}
