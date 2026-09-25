//! `temen-link -o:<out.temen> [--libc:<path>] <main.c.nif> <dep.c.nif>...` — link a nim program's
//! modules into one POSIX-personality Temen module, in-guest (#1609).
//!
//! nimony's Temen backend (`nimony t`) plans this as the one whole-program node after DCE, in the
//! place `lengc` + `cc` + the linker take in the C backend; nifmake execs it through `/bin/sh`. Every
//! input is one module's DCE'd Leng (`<stem>.c.nif`). The link is [`temen_leng::link_nim_posix`] —
//! the call the host makes, which orders the modules itself, so the output depends on the module
//! set alone — against the personality's vocabulary ([`temen_posix_abi::vtable`]) and the prebuilt
//! guest libc (`/lib/temen/libc.temeno`, or `--libc:`; linked without one if absent).
//!
//! It reaches the world only through the personality's imports (`__px_*`, #1668), so it binds in an
//! exec'd powerbox exactly like the nim programs around it. Exit codes: 1 = bad arguments,
//! 2 = an input unreadable, 3 = the link refused, 4 = the output unwritable.
//!
//! Its heap is a size-class allocator over the on-ramp's `malloc` (see [`Heap`]).

use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn __px_open(path: i64, path_len: i64, flags: i64) -> i64;
    fn __px_read(fd: i64, buf: i64, len: i64) -> i64;
    fn __px_write(fd: i64, buf: i64, len: i64) -> i64;
    fn __px_close(fd: i64) -> i64;
    fn malloc(size: usize) -> *mut u8;
}

/// The global allocator: power-of-two **size classes** with intrusive free lists, carved from chunks
/// of the on-ramp's `malloc` (a bump allocator over the window that never frees). A freed block goes
/// back on its class's list and the next allocation of that class reuses it, so the heap tracks what
/// the linker has live, not everything it ever allocated: with a leaking allocator linking even a
/// small program outgrew the 32 MiB window every image in the self-hosted lane runs in. The guest is
/// single-threaded (one process, no threads), so the state needs no lock.
struct Classes {
    /// Free list per class `k` (blocks of `16 << k` bytes), threaded through the free blocks.
    free: [*mut u8; CLASSES],
    /// The current chunk's unused tail.
    bump: *mut u8,
    end: *mut u8,
}

const CLASSES: usize = 40;
const CHUNK: usize = 1 << 20;

struct Heap(core::cell::UnsafeCell<Classes>);
// SAFETY: the guest runs one thread; nothing shares the heap across threads.
unsafe impl Sync for Heap {}

/// The class that holds `size` bytes: blocks of `16 << class` bytes.
fn class_of(size: usize) -> usize {
    let blocks = size.max(16).next_power_of_two() / 16;
    blocks.trailing_zeros() as usize
}

unsafe impl GlobalAlloc for Heap {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        // `malloc` and every class block are 16-byte aligned; a block of `16 << k` bytes carved at a
        // 16-aligned bump is only 16-aligned, so a larger alignment takes a class at least that big
        // and an aligned carve.
        let c = &mut *self.0.get();
        let k = class_of(l.size().max(l.align()));
        if k >= CLASSES {
            return core::ptr::null_mut();
        }
        let head = c.free[k];
        if !head.is_null() && (head as usize) % l.align() == 0 {
            c.free[k] = *(head as *mut *mut u8);
            return head;
        }
        let size = 16usize << k;
        let align = l.align().max(16);
        let mut at = ((c.bump as usize + align - 1) & !(align - 1)) as *mut u8;
        if c.bump.is_null() || at as usize + size > c.end as usize {
            let want = size.max(CHUNK) + align;
            let chunk = malloc(want);
            if chunk.is_null() {
                return core::ptr::null_mut();
            }
            c.end = chunk.add(want);
            at = ((chunk as usize + align - 1) & !(align - 1)) as *mut u8;
        }
        c.bump = at.add(size);
        at
    }

    unsafe fn dealloc(&self, ptr: *mut u8, l: Layout) {
        let c = &mut *self.0.get();
        let k = class_of(l.size().max(l.align()));
        *(ptr as *mut *mut u8) = c.free[k];
        c.free[k] = ptr;
    }

    unsafe fn realloc(&self, ptr: *mut u8, l: Layout, new_size: usize) -> *mut u8 {
        // Still fits its block: nothing moves (a `Vec` growing inside its class, or shrinking).
        if class_of(new_size.max(l.align())) == class_of(l.size().max(l.align())) {
            return ptr;
        }
        let new = self.alloc(Layout::from_size_align_unchecked(new_size, l.align()));
        if !new.is_null() {
            core::ptr::copy_nonoverlapping(ptr, new, l.size().min(new_size));
            self.dealloc(ptr, l);
        }
        new
    }
}

#[global_allocator]
static A: Heap = Heap(core::cell::UnsafeCell::new(Classes {
    free: [core::ptr::null_mut(); CLASSES],
    bump: core::ptr::null_mut(),
    end: core::ptr::null_mut(),
}));

// The personality's `open` flags (POSIX values, temen-posix): O_RDONLY = 0; O_WRONLY|O_CREAT|O_TRUNC.
const O_RDONLY: i64 = 0;
const O_WRONLY_CREAT_TRUNC: i64 = 0o1 | 0o100 | 0o1000;

/// The whole file at `path`, or `None` if it cannot be opened.
fn read_file(path: &str) -> Option<Vec<u8>> {
    let fd = unsafe { __px_open(path.as_ptr() as i64, path.len() as i64, O_RDONLY) };
    if fd < 0 {
        return None;
    }
    let mut out = Vec::new();
    let mut chunk = vec![0u8; 1 << 16];
    loop {
        let n = unsafe { __px_read(fd, chunk.as_mut_ptr() as i64, chunk.len() as i64) };
        if n <= 0 {
            break;
        }
        out.extend_from_slice(&chunk[..n as usize]);
    }
    unsafe { __px_close(fd) };
    Some(out)
}

/// Write `bytes` to `path`, creating or truncating it.
fn write_file(path: &str, bytes: &[u8]) -> bool {
    let fd = unsafe {
        __px_open(
            path.as_ptr() as i64,
            path.len() as i64,
            O_WRONLY_CREAT_TRUNC,
        )
    };
    if fd < 0 {
        return false;
    }
    let mut off = 0;
    while off < bytes.len() {
        let n = unsafe {
            __px_write(fd, bytes[off..].as_ptr() as i64, (bytes.len() - off) as i64)
        };
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
    unsafe { __px_close(fd) };
    off == bytes.len()
}

/// A module's stem from its path: `nimcache/x.temen/sysvq0asl.c.nif` → `sysvq0asl`.
fn stem(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    name.split('.').next().unwrap_or(name)
}

fn run(args: &[&str]) -> i32 {
    let mut out = None;
    let mut libc_path = "/lib/temen/libc.temeno";
    let mut inputs = Vec::new();
    for &a in args {
        if let Some(o) = a.strip_prefix("-o:") {
            out = Some(o);
        } else if let Some(l) = a.strip_prefix("--libc:") {
            libc_path = l;
        } else {
            inputs.push(a);
        }
    }
    let Some(out) = out.filter(|_| !inputs.is_empty()) else {
        return 1;
    };
    let mut srcs = Vec::new();
    for &p in &inputs {
        let Some(bytes) = read_file(p) else { return 2 };
        srcs.push((stem(p), temen_leng::nif_text(&bytes).into_owned()));
    }
    let units: Vec<temen_leng::WholeModule> = srcs
        .iter()
        .map(|(stem, src)| temen_leng::WholeModule { stem, src })
        .collect();
    let libc = read_file(libc_path);
    let (names, sigs) = temen_posix_abi::vtable();
    let Ok(module) = temen_leng::link_nim_posix(&units, (&names, &sigs), libc.as_deref()) else {
        return 3;
    };
    if !write_file(out, &temen_encode::encode_module(&module)) {
        return 4;
    }
    0
}

/// `main(argc, argv)`: the on-ramp's powerbox `_start` parses the args region an `execve` wrote.
///
/// # Safety
/// `argv` points at `argc` NUL-terminated strings.
#[no_mangle]
pub unsafe extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    let mut args = Vec::new();
    for i in 1..argc.max(0) as usize {
        let p = *argv.add(i);
        let mut n = 0;
        while *p.add(n) != 0 {
            n += 1;
        }
        let Ok(s) = core::str::from_utf8(core::slice::from_raw_parts(p, n)) else {
            return 1;
        };
        args.push(s);
    }
    run(&args)
}
