//! The nim->powerbox linker as a **child-entry, memfs-I/O** phase (#1025 3c, assemble).
//!
//! `nim_link_guest` linked over stdin/stdout — fine for a top-level run, but a *driver guest* produces
//! the linker's input (the hexer `.x.nif`) at runtime, so it can't be a host-seeded stream. This guest
//! composes the way the other phases do: it is built `--child-entry` (op-13-spawnable) and hands off
//! through the **shared memfs** — read `argv[1]` (`nimcache/<stem>.x.nif`, written by hexer), link it
//! with `temen_leng::link_nim_powerbox`, write the `temen_encode`d linked module to `argv[2]`
//! (`nimcache/<stem>.temen`). `argv[3]` is the stem the `WholeModule` carries. Reaches the memfs through
//! the raw `__vm_cap_resolve`/`__vm_host_call` seam (the `fs` `HOST_PROC` cap, ops open/read/write/close
//! = 0/1/2/4), exactly as `child_entry_argv_fs.rs`. Paths are relative (the memfs is relative-only).
//!
//! Not part of the escape-TCB — everything it emits is re-verified by temen-verify (DESIGN.md §2a).
//!
//! Unlike `nim_link_guest`'s 512 MiB static bump arena, this backs its global allocator with the
//! on-ramp's **synthesized `malloc`** (the `vm_map`-growing bump allocator, proven by
//! `child_entry_malloc.rs`): the heap grows on-demand inside the carve instead of a huge static (a
//! 512 MiB static would force a >=512 MiB *declared* window). Using `malloc` also sets `need_malloc`,
//! which forces the synthesized powerbox `_start` — so `--child-entry` gets a `synth_start_argv` as
//! func 0 even though the guest reaches its caps only through raw `__vm_*` intrinsics (not §7 imports).

use core::alloc::{GlobalAlloc, Layout};

extern "C" {
    fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
    fn __vm_host_call(h: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
    // The on-ramp's synthesized allocator (16-byte aligned, never frees — a one-shot bump into the
    // carve via `vm_map`). Declaring `malloc` also flips `need_malloc`, forcing the powerbox `_start`.
    fn malloc(size: usize) -> *mut u8;
}

const PTR: usize = core::mem::size_of::<*mut u8>();

/// Global allocator over the synthesized `malloc`. The bump allocator returns 16-byte-aligned blocks
/// and never frees, so `dealloc` is a no-op (a one-shot linker leaks, exactly like `nim_link_guest`);
/// over-aligned requests (`align > 16`) over-allocate and stash the base pointer just below the aligned
/// address (unused by `dealloc`, but kept correct in shape).
struct Sys;
unsafe impl GlobalAlloc for Sys {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        if l.align() <= 16 {
            return malloc(l.size());
        }
        let base = malloc(l.size() + l.align() + PTR);
        if base.is_null() {
            return core::ptr::null_mut();
        }
        let aligned = ((base as usize + PTR + l.align() - 1) & !(l.align() - 1)) as *mut u8;
        (aligned.sub(PTR) as *mut *mut u8).write(base);
        aligned
    }
    unsafe fn dealloc(&self, _ptr: *mut u8, _l: Layout) {}
}
#[global_allocator]
static A: Sys = Sys;

// fs `HOST_PROC` op numbers (temen_fs): open/read/write/close.
const FS_OPEN: i32 = 0;
const FS_READ: i32 = 1;
const FS_WRITE: i32 = 2;
const FS_CLOSE: i32 = 4;
// open flags (temen_fs): O_READ=1; O_WRITE|O_CREATE|O_TRUNC = 2|16|8 = 26.
const O_READ: i64 = 1;
const O_WRITE_CREATE_TRUNC: i64 = 26;

unsafe fn clen(p: *const u8) -> usize {
    let mut n = 0;
    while *p.add(n) != 0 {
        n += 1;
    }
    n
}

/// Read the whole memfs file `path` (NUL-terminated, relative) through the `fs` cap.
unsafe fn read_file(fs: i32, path: *const u8) -> Option<Vec<u8>> {
    let fd = __vm_host_call(fs, FS_OPEN, path as i64, clen(path) as i64, O_READ, 0);
    if fd < 0 {
        return None;
    }
    let mut v = Vec::new();
    let mut chunk = [0u8; 65536];
    loop {
        let n = __vm_host_call(
            fs,
            FS_READ,
            fd,
            chunk.as_mut_ptr() as i64,
            chunk.len() as i64,
            0,
        );
        if n <= 0 {
            break;
        }
        v.extend_from_slice(&chunk[..n as usize]);
    }
    __vm_host_call(fs, FS_CLOSE, fd, 0, 0, 0);
    Some(v)
}

/// Write `bytes` to the memfs file `path` (NUL-terminated, relative), creating/truncating it.
unsafe fn write_file(fs: i32, path: *const u8, bytes: &[u8]) -> bool {
    let fd = __vm_host_call(
        fs,
        FS_OPEN,
        path as i64,
        clen(path) as i64,
        O_WRITE_CREATE_TRUNC,
        0,
    );
    if fd < 0 {
        return false;
    }
    let mut off = 0usize;
    while off < bytes.len() {
        let n = __vm_host_call(
            fs,
            FS_WRITE,
            fd,
            bytes[off..].as_ptr() as i64,
            (bytes.len() - off) as i64,
            0,
        );
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
    __vm_host_call(fs, FS_CLOSE, fd, 0, 0, 0);
    off == bytes.len()
}

/// `main(argc, argv)` — the §14 child-entry ABI (`main(argc, argv)` forces `synth_start_argv` as func 0;
/// the starter cap arrives ahead of it and is ignored). `argv = ["link", <in.x.nif>, <out.temen>,
/// <stem>]`. Exit codes: 1 = bad argv, 2 = input unreadable / not UTF-8, 3 = the linker refused the unit,
/// 4 = output unwritable, 0 = the linked module was written.
///
/// # Safety
/// `argv` points at `argc` NUL-terminated C strings the parent seeded into the carve.
#[no_mangle]
pub unsafe extern "C" fn main(argc: i32, argv: *const *const u8) -> i32 {
    if argc < 4 {
        return 1;
    }
    let in_path = *argv.add(1);
    let out_path = *argv.add(2);
    let stem_ptr = *argv.add(3);

    let fs = __vm_cap_resolve(b"fs".as_ptr(), 2);
    if fs < 0 {
        return 1;
    }

    let Some(src_bytes) = read_file(fs, in_path) else {
        return 2;
    };
    let Ok(src) = core::str::from_utf8(&src_bytes) else {
        return 2;
    };
    let stem_bytes = core::slice::from_raw_parts(stem_ptr, clen(stem_ptr));
    let Ok(stem) = core::str::from_utf8(stem_bytes) else {
        return 2;
    };

    let units = [temen_leng::WholeModule { stem, src }];
    let module = match temen_leng::link_nim_powerbox(&units) {
        Ok(m) => m,
        Err(_) => return 3,
    };
    let bytes = temen_encode::encode_module(&module);

    if !write_file(fs, out_path, &bytes) {
        return 4;
    }
    0
}
