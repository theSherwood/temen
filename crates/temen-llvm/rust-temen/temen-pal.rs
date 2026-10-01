//! temen PAL: the platform surface for the `x86_64-unknown-temen` target. Mirrors the `unsupported`
//! PAL (temen reaches the host through capability-bound named imports — `write`/`read`/`exit` — not a
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
/// drives the op table (temen-posix `OP_*`). This is how the richer `std::sys` surface — `time` here,
/// `fs`/`env` later — reaches the host, distinct from the powerbox stdout/exit streams. Each op has
/// its own wrapper so the `op` argument is a **compile-time constant** at the `__vm_host_call` site
/// (the on-ramp requires it).
pub(crate) mod host {
    use crate::sync::atomic::{AtomicI32, Ordering};

    unsafe extern "C" {
        fn __vm_cap_resolve(name: *const u8, len: i64) -> i32;
        fn __vm_host_call(handle: i32, op: i32, a: i64, b: i64, c: i64, d: i64) -> i64;
        // The core pipe builtins (temen-llvm, FORK.md §8.6): mint a pipe into this process's
        // powerbox, and read, write or close one of its ends by handle.
        fn __vm_pipe(fds: *mut i32) -> i64;
        fn __vm_read(h: i32, buf: *mut u8, len: i64) -> i64;
        fn __vm_write(h: i32, buf: *const u8, len: i64) -> i64;
        fn __vm_close(h: i32) -> i64;
    }

    /// The bound at or below which a personality answer is a **tag**, `TAG_BASE - handle` (temen-posix
    /// `PX_TAG_BASE`, #972): the fd is a core pipe end, and the call belongs on that end's own handle.
    const TAG_BASE: i64 = -(1 << 20);

    /// The pipe-end handle a personality answer `r` redirects to, if it is a tag.
    fn tagged(r: i64) -> Option<i32> {
        (r <= TAG_BASE).then(|| (TAG_BASE - r) as i32)
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

    /// `clock(clock_id) -> nanos` (temen-posix `OP_CLOCK` = 33). `clock_id == 1` is monotonic, else
    /// realtime. `op` is the literal `33` at the call site.
    #[inline(always)]
    pub(crate) fn clock(clock_id: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 33, clock_id, 0, 0, 0) }
    }

    /// `getenv_r(name, nlen, buf, cap) -> nbytes | -1` (temen-posix `OP_GETENV_R` = 34). Writes the
    /// value into `[buf, cap)` when it fits; the length is returned either way (size-then-fetch).
    #[inline(always)]
    pub(crate) fn getenv_r(name: *const u8, nlen: i64, buf: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 34, name as i64, nlen, buf as i64, cap) }
    }

    /// `setenv(name, nlen, val, vlen) -> 0 | -errno` (temen-posix `OP_SETENV` = 12; the 4-arg form
    /// always overwrites).
    #[inline(always)]
    pub(crate) fn setenv(name: *const u8, nlen: i64, val: *const u8, vlen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 12, name as i64, nlen, val as i64, vlen) }
    }

    /// `unsetenv(name, nlen) -> 0 | -errno` (temen-posix `OP_UNSETENV` = 35).
    #[inline(always)]
    pub(crate) fn unsetenv(name: *const u8, nlen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 35, name as i64, nlen, 0, 0) }
    }

    /// `environ(index, buf, cap) -> len | -1` (temen-posix `OP_ENVIRON` = 36). Writes the `index`-th
    /// `KEY=VALUE` into `[buf, cap)` when it fits; the length is returned either way.
    #[inline(always)]
    pub(crate) fn environ(index: i64, buf: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 36, index, buf as i64, cap, 0) }
    }

    // ---- filesystem (temen-posix `OP_OPEN`/`OP_READ`/… — the `std::fs` surface) -------------------
    // Each op's literal code is inlined at the `__vm_host_call` site (the on-ramp requires the `op`
    // argument to be a compile-time constant).

    /// `open(path, plen, flags) -> fd | -errno` (temen-posix `OP_OPEN` = 5).
    #[inline(always)]
    pub(crate) fn open(path: *const u8, plen: i64, flags: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 5, path as i64, plen, flags, 0) }
    }

    /// `close(fd) -> 0 | -errno` (temen-posix `OP_CLOSE` = 6). The last close of a core pipe end's
    /// descriptor answers a tag, and the end's handle is released here.
    #[inline(always)]
    pub(crate) fn close(fd: i64) -> i64 {
        let r = unsafe { __vm_host_call(posix(), 6, fd, 0, 0, 0) };
        match tagged(r) {
            Some(h) => {
                unsafe { __vm_close(h) };
                0
            }
            None => r,
        }
    }

    /// `lseek(fd, offset, whence) -> new_offset | -errno` (temen-posix `OP_LSEEK` = 7).
    #[inline(always)]
    pub(crate) fn lseek(fd: i64, offset: i64, whence: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 7, fd, offset, whence, 0) }
    }

    /// `unlink(path, plen) -> 0 | -errno` (temen-posix `OP_UNLINK` = 8).
    #[inline(always)]
    pub(crate) fn unlink(path: *const u8, plen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 8, path as i64, plen, 0, 0) }
    }

    /// `mkdir(path, plen, mode) -> 0 | -errno` (temen-posix `OP_MKDIR` = 37). Creates an explicit empty
    /// directory; `mode` is advisory (the memfs has no perm model).
    #[inline(always)]
    pub(crate) fn mkdir(path: *const u8, plen: i64, mode: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 37, path as i64, plen, mode, 0) }
    }

    /// `rename(old, olen, new, nlen) -> 0 | -errno` (temen-posix `OP_RENAME` = 38).
    #[inline(always)]
    pub(crate) fn rename(old: *const u8, olen: i64, new: *const u8, nlen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 38, old as i64, olen, new as i64, nlen) }
    }

    /// `rmdir(path, plen) -> 0 | -errno` (temen-posix `OP_RMDIR` = 39). Removes an empty directory.
    #[inline(always)]
    pub(crate) fn rmdir(path: *const u8, plen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 39, path as i64, plen, 0, 0) }
    }

    /// `read(fd, buf, len) -> n | -errno` (temen-posix `OP_READ` = 1). Distinct from the stdio PAL's
    /// `extern "C" read` (the powerbox stdin stream) — this drives a `posix`-personality file fd. A
    /// core pipe end's descriptor answers a tag, and the read is made on the end: it waits while the
    /// pipe is empty and a writer is left.
    #[inline(always)]
    pub(crate) fn read_fd(fd: i64, buf: *mut u8, len: i64) -> i64 {
        let r = unsafe { __vm_host_call(posix(), 1, fd, buf as i64, len, 0) };
        match tagged(r) {
            Some(h) => unsafe { __vm_read(h, buf, len) },
            None => r,
        }
    }

    /// `write(fd, buf, len) -> n | -errno` (temen-posix `OP_WRITE` = 0). The file-fd counterpart of the
    /// stdio PAL's powerbox `write`; a core pipe end's descriptor is written on the end, as
    /// [`read_fd`] reads it.
    #[inline(always)]
    pub(crate) fn write_fd(fd: i64, buf: *const u8, len: i64) -> i64 {
        let r = unsafe { __vm_host_call(posix(), 0, fd, buf as i64, len, 0) };
        match tagged(r) {
            Some(h) => unsafe { __vm_write(h, buf, len) },
            None => r,
        }
    }

    /// `stat(path, plen, statbuf) -> 0 | -errno` (temen-posix `OP_STAT` = 13). Fills the caller's
    /// `{ i64 st_mode; i64 st_size; }` (16 bytes).
    #[inline(always)]
    pub(crate) fn stat(path: *const u8, plen: i64, statbuf: *mut u8) -> i64 {
        unsafe { __vm_host_call(posix(), 13, path as i64, plen, statbuf as i64, 0) }
    }

    /// `opendir(path, plen) -> dir | -errno` (temen-posix `OP_OPENDIR` = 14).
    #[inline(always)]
    pub(crate) fn opendir(path: *const u8, plen: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 14, path as i64, plen, 0, 0) }
    }

    /// `readdir(dir, name_buf, cap) -> namelen | 0 | -errno` (temen-posix `OP_READDIR` = 15). Writes the
    /// next entry's NUL-terminated name; `0` is end-of-stream, `-ERANGE` means `cap` was too small.
    #[inline(always)]
    pub(crate) fn readdir(dir: i64, name_buf: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 15, dir, name_buf as i64, cap, 0) }
    }

    /// `closedir(dir) -> 0 | -errno` (temen-posix `OP_CLOSEDIR` = 16).
    #[inline(always)]
    pub(crate) fn closedir(dir: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 16, dir, 0, 0, 0) }
    }

    // ---- pipes + spawn/wait (temen-posix `OP_PIPE_ADOPT`/`OP_PSPAWN`/… — the `std::process` surface) --

    /// `pipe(fds_ptr) -> 0 | -errno`: mint a core pipe into this process's powerbox (`__vm_pipe`) and
    /// adopt its two ends as descriptors (temen-posix `OP_PIPE_ADOPT` = 52), written at `fds_ptr` as two
    /// little-endian `i32`s, read end first. A read of the empty pipe waits while a writer is left, and a
    /// write to the full pipe while a reader is — across processes: a spawned child's end is the same
    /// pipe.
    pub(crate) fn pipe(fds_ptr: *mut u8) -> i64 {
        let mut h = [0i32; 2];
        let r = unsafe { __vm_pipe(h.as_mut_ptr()) };
        if r != 0 {
            return r;
        }
        let r = unsafe { __vm_host_call(posix(), 52, h[0] as i64, h[1] as i64, fds_ptr as i64, 0) };
        if r != 0 {
            unsafe {
                __vm_close(h[0]);
                __vm_close(h[1]);
            }
        }
        r
    }

    /// `dup2(oldfd, newfd) -> newfd | -errno` (temen-posix `OP_DUP2` = 24). Re-points `newfd` at `oldfd`'s
    /// object, closing whatever `newfd` referred to.
    #[inline(always)]
    pub(crate) fn dup2(oldfd: i64, newfd: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 24, oldfd, newfd, 0, 0) }
    }

    /// `dup(oldfd) -> fd | -errno` (temen-posix `OP_DUP` = 25). Clones `oldfd` onto the lowest free fd.
    #[inline(always)]
    pub(crate) fn dup(oldfd: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 25, oldfd, 0, 0, 0) }
    }

    /// `pspawn(req) -> pid | -errno` (temen-posix `OP_PSPAWN` = 62): POSIX `posix_spawn`, a new process
    /// running the program at a path. `req` points at five little-endian words, `{path, argv, envp,
    /// actions, nactions}`: a NUL-terminated path, NULL-terminated arrays of NUL-terminated strings
    /// (`envp` may be NULL), and `nactions` file actions of four words each, `{op, fd, arg, path}`,
    /// applied in order to the child's copy of this process's descriptors.
    #[inline(always)]
    pub(crate) fn pspawn(req: *const u64) -> i64 {
        unsafe { __vm_host_call(posix(), 62, req as i64, 0, 0, 0) }
    }

    /// `waitpid(pid, status_ptr, options) -> pid | 0 | -errno` (temen-posix `OP_WAITPID` = 28). Reaps
    /// `pid` (or any child when `pid == -1`), writing the wait-encoded status (`WEXITSTATUS` in bits
    /// 8–15) as an `i32` to `status_ptr` when non-null. Waits for a running child to exit, unless
    /// `options` holds `WNOHANG` (`1`): then a running child answers `0`.
    #[inline(always)]
    pub(crate) fn waitpid(pid: i64, status_ptr: *mut u8, options: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 28, pid, status_ptr as i64, options, 0) }
    }

    /// `kill(pid, sig) -> 0 | -errno` (temen-posix `OP_KILL` = 31).
    #[inline(always)]
    pub(crate) fn kill(pid: i64, sig: i64) -> i64 {
        unsafe { __vm_host_call(posix(), 31, pid, sig, 0, 0) }
    }

    /// `getpid() -> pid` (temen-posix `OP_GETPID` = 44): this process's pid — `1` for a run's root.
    #[inline(always)]
    pub(crate) fn getpid() -> i64 {
        unsafe { __vm_host_call(posix(), 44, 0, 0, 0, 0) }
    }

    // ---- the `net` capability (POSIX.md §5a — `std::net`) ---------------------------------------
    // A **separate named handle** from `posix`: authority is its own grant. Addresses travel as the
    // blob `[family u8 (4|6), port u16 LE, addr 4|16 bytes]`. Connected sockets are ordinary fds —
    // `TcpStream` read/write ride the posix `read_fd`/`write_fd` ops above.

    // `-1` = not-yet-resolved sentinel; resolved once, then cached (like `POSIX`).
    static NET: AtomicI32 = AtomicI32::new(-1);

    /// The `net` capability handle, or a negative value if the embedder granted none.
    fn net() -> i32 {
        let cached = NET.load(Ordering::Relaxed);
        if cached != -1 {
            return cached;
        }
        let h = unsafe { __vm_cap_resolve(b"net".as_ptr(), 3) };
        NET.store(h, Ordering::Relaxed);
        h
    }

    /// Whether a `net` capability is available (a socket op can be attempted).
    pub(crate) fn have_net() -> bool {
        net() >= 0
    }

    /// `connect(addr, alen, laddr_out, cap) -> fd | -errno` (temen-posix `NET_CONNECT` = 1). Writes the
    /// connection's local address blob.
    #[inline(always)]
    pub(crate) fn net_connect(addr: *const u8, alen: i64, laddr_out: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(net(), 1, addr as i64, alen, laddr_out as i64, cap) }
    }

    /// `bind(addr, alen, bound_out, cap) -> listener_fd | -errno` (temen-posix `NET_BIND` = 2).
    /// Bind+listen folded; `:0` assigns an ephemeral port; writes the actual bound address.
    #[inline(always)]
    pub(crate) fn net_bind(addr: *const u8, alen: i64, bound_out: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(net(), 2, addr as i64, alen, bound_out as i64, cap) }
    }

    /// `accept(fd, peer_out, cap) -> fd | -EAGAIN | -errno` (temen-posix `NET_ACCEPT` = 3). Writes the
    /// peer's address blob.
    #[inline(always)]
    pub(crate) fn net_accept(fd: i64, peer_out: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(net(), 3, fd, peer_out as i64, cap, 0) }
    }

    /// `shutdown(fd, how) -> 0 | -errno` (temen-posix `NET_SHUTDOWN` = 4; `0` read / `1` write / `2` both).
    #[inline(always)]
    pub(crate) fn net_shutdown(fd: i64, how: i64) -> i64 {
        unsafe { __vm_host_call(net(), 4, fd, how, 0, 0) }
    }

    /// `resolve(name, nlen, out, cap) -> nbytes | -errno` (temen-posix `NET_RESOLVE` = 5). Writes address
    /// blobs back-to-back when they fit; the total length is returned either way (size-then-fetch).
    #[inline(always)]
    pub(crate) fn net_resolve(name: *const u8, nlen: i64, out: *mut u8, cap: i64) -> i64 {
        unsafe { __vm_host_call(net(), 5, name as i64, nlen, out as i64, cap) }
    }
}
