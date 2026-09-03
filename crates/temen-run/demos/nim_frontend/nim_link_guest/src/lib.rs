//! The nim->powerbox linker as a powerbox program (#1025 3c). Reads the hexer Leng `.x.nif` units,
//! packed on stdin, links them with `temen_leng::link_nim_powerbox` (the compute shim + syscall adapter
//! + powerbox entry), and writes the `temen_encode`d linked module to stdout. Same guest-libc-free shape
//! as `leng_guest`: a static-arena bump allocator + raw `read`/`write`.
//!
//! Stdin packing (all lengths u32 LE): `count`, then per unit `stem_len, stem, src_len, src`.

use core::alloc::{GlobalAlloc, Layout};

const ARENA: usize = 512 * 1024 * 1024;
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

fn rd_u32(b: &[u8], p: &mut usize) -> Option<usize> {
    let e = *p + 4;
    if e > b.len() {
        return None;
    }
    let v = u32::from_le_bytes([b[*p], b[*p + 1], b[*p + 2], b[*p + 3]]) as usize;
    *p = e;
    Some(v)
}

/// `main`: parse the packed units, link, emit the encoded module. Exit 1 = malformed input, 2 = a unit
/// was not UTF-8, 3 = the linker refused the units, 0 = the linked module was written to stdout.
#[no_mangle]
pub extern "C" fn main() -> i32 {
    let input = read_stdin();
    let mut p = 0usize;
    let Some(count) = rd_u32(&input, &mut p) else { return 1 };
    let mut owned: Vec<(String, String)> = Vec::with_capacity(count);
    for _ in 0..count {
        let Some(sl) = rd_u32(&input, &mut p) else { return 1 };
        if p + sl > input.len() {
            return 1;
        }
        let stem = match core::str::from_utf8(&input[p..p + sl]) {
            Ok(s) => s.to_string(),
            Err(_) => return 2,
        };
        p += sl;
        let Some(cl) = rd_u32(&input, &mut p) else { return 1 };
        if p + cl > input.len() {
            return 1;
        }
        let src = match core::str::from_utf8(&input[p..p + cl]) {
            Ok(s) => s.to_string(),
            Err(_) => return 2,
        };
        p += cl;
        owned.push((stem, src));
    }
    let units: Vec<temen_leng::WholeModule> = owned
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    match temen_leng::link_nim_powerbox(&units) {
        Ok(m) => {
            let bytes = temen_encode::encode_module(&m);
            write_stdout(&bytes);
            0
        }
        Err(_) => 3,
    }
}
