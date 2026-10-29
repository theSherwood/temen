//! **Configurable filesystem capability backends** — §7 embedder-registered `HostProc`s, *not* part of
//! the default powerbox grant. A consumer that wants its guest to see a filesystem offers one
//! explicitly (wasm-style dependency injection): grant it under a name via
//! [`Instance::run_with_caps`](crate::Instance::run_with_caps) (or bind it to an import with
//! [`instantiate_with_imports`](crate::instantiate_with_imports)), and the guest reaches it with
//! `__vm_cap_resolve("fs")` + `__vm_host_call(handle, op, …)`. No filesystem authority exists unless
//! the embedder injects it, and the two backends here are interchangeable behind the same op protocol:
//!
//! - [`mem_fs`] — a deterministic **in-memory** filesystem (fresh per run). The hermetic default for
//!   tests and differential runs: no real-fs state, no cleanup, parallel-safe.
//! - [`host_fs`] — the **real** filesystem, attenuated to a root directory: the capability *is* the
//!   rooted directory (relative paths only; `..` and absolute paths are refused), so the guest cannot
//!   name anything outside it. Every path resolves beneath that directory (#2643): a symlink that
//!   stays inside it works as its target does, and one that leads out is refused, as `..` is.
//!
//! ## Op protocol (`__vm_host_call(handle, op, a, b, c, d) -> i64`)
//!
//! | op | name | args | returns |
//! |----|------|------|---------|
//! | 0 | `open` | `(path_ptr, path_len, flags, _)` | fd ≥ 0 |
//! | 1 | `read` | `(fd, buf_ptr, len, _)` | bytes read (0 = EOF) |
//! | 2 | `write` | `(fd, buf_ptr, len, _)` | bytes written |
//! | 3 | `seek` | `(fd, whence 0/1/2, offset, _)` | new position |
//! | 4 | `close` | `(fd, _, _, _)` | 0 |
//! | 5 | `remove` | `(path_ptr, path_len, _, _)` | 0 |
//! | 6 | `rename` | `(from_ptr, from_len, to_ptr, to_len)` | 0 |
//! | 7 | `truncate` | `(fd, new_len, _, _)` | 0 |
//! | 8 | `sync` | `(fd, _, _, _)` | 0 |
//! | 9 | `mmap` | `(fd, file_offset, len, win_buf)` | 0 |
//! | 10 | `msync` | `(win_buf, len, _, _)` | 0 |
//! | 11 | `munmap` | `(win_buf, _, _, _)` | 0 |
//! | 12 | `crash_arm` | `(n, _, _, _)` | 0 / `-EINVAL` |
//! | 13 | `map_region` | `(fd, file_offset, len, _)` | region handle / `-errno` |
//! | 14 | `stat` | `(path_ptr, path_len, statbuf_ptr, statbuf_cap)` | 0 / `-errno` |
//! | 15 | `mkdir` | `(path_ptr, path_len, _, _)` | 0 / `-errno` |
//! | 16 | `rmdir` | `(path_ptr, path_len, _, _)` | 0 / `-errno` |
//! | 17 | `opendir` | `(path_ptr, path_len, _, _)` | dir handle ≥ 0 / `-errno` |
//! | 18 | `readdir` | `(dh, name_ptr, name_cap, _)` | name length > 0, `0` = end, `-errno` |
//! | 19 | `closedir` | `(dh, _, _, _)` | 0 / `-errno` |
//! | 20 | `replace` | `(path_ptr, path_len, buf_ptr, len)` | 0 / `-errno` |
//!
//! Op 12 (`crash_arm`) exists **only on the `*_crashy` test variants** (see [`FS_CRASH_ARM`]); op 13
//! (`map_region`, the §4b zero-copy path — see [`FS_MAP_REGION`]) exists **only on [`host_fs_mmap`]**.
//! The default [`mem_fs`]/[`host_fs`] return `-EINVAL` for both (unknown op / no minting authority).
//!
//! ## The metadata + directory surface (`stat`/`mkdir`/`rmdir`/`opendir`/`readdir`/`closedir`)
//!
//! The per-access VFS above names *files*; a real data tree (a natively-`initdb`'d Postgres cluster,
//! say) also needs to be **walked**. `stat` fills a fixed 72-byte little-endian [`StatBuf`] (mode with
//! the `S_IF*` type bits, size, mtime, ino/dev, …) so the guest libc can tell a directory from a file
//! and read a size without opening; `mkdir`/`rmdir` create and remove directories; `opendir` snapshots
//! a directory's immediate entries and `readdir` yields their names one per call (`0` when exhausted),
//! `closedir` drops the handle. `stat` uses **lstat** semantics: a symlink is reported as itself, not
//! as its target. Both backends implement the identical protocol; [`mem_fs`] models directories over
//! its flat name table (a path is a directory if it was `mkdir`'d or is a strict prefix of an
//! existing file), so a differential still runs identically on `mem_fs` and `host_fs`.
//!
//! ## The file-backed-mmap surface (`mmap`/`msync`/`munmap`) — the second storage shape
//!
//! `mmap` binds a **guest-owned window buffer** (`win_buf`, `len`) to a file region (`fd`,
//! `file_offset`): the host copies the file bytes *into* the buffer and records the binding. The
//! guest then reads and writes those bytes with ordinary loads/stores — **zero host calls on the
//! data-access path** (that is what makes this the memory-mapped shape, distinct from the per-access
//! `read`/`write` VFS). `msync(win_buf, len)` flushes a sub-range of a mapping back to its file at
//! `file_offset + (win_buf − mapping.base)`; `munmap` flushes the whole mapping and drops it. This is
//! coherent for a single mapping of a file (the buffer is the sole authority), exactly what an
//! `MDB_WRITEMAP`-mode LMDB needs: it writes every page — data and meta — through the map and asks
//! for durability via `msync`. (Not multi-process shared-memory coherence — that is a later slice.)
//!
//! Errors are negative errno values ([`ENOENT`]/[`EBADF`]/[`EINVAL`]/[`EACCES`]/[`EFAULT`]). `flags`
//! is a bitset ([`O_READ`]/[`O_WRITE`]/[`O_APPEND`]/[`O_TRUNC`]/[`O_CREATE`]) the guest libc derives
//! from the C `fopen` mode string. Buffers/paths are window-relative; an out-of-window range is
//! `-EFAULT` (fail-closed, never a host-side OOB).

use crate::Descriptor;
use crate::HostCap;
use cap_std::fs::Dir;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use temen_interp::{GuestMem, HostCharge, HostProc, NodeRef, RegionMinter, TABLE_ENTRY};

// The shared fs-cap wire protocol, the in-memory backend, and the shippable data-image format live in
// the wasm-safe `temen-fs` crate — so the browser cdylib can mount a `mem_fs` without the unix-only JIT
// that `temen-run` pulls in. Re-export them so `temen_run::fs::*` is unchanged; the real-filesystem
// `host_fs` backend and the `HostCap` wrappers (temen-run's capability type) stay here.
pub use temen_fs::*;
use temen_ir::errno::*; // the one shared negative-errno table (#905)

/// A deterministic **in-memory** filesystem capability (fresh, empty state per host). The hermetic
/// default for tests and differential runs.
pub fn mem_fs() -> HostCap {
    mem_fs_cap(|| temen_fs::MemFsHandle::new(false))
}

/// Like [`mem_fs`] but with the **test-only crash-injection** controller enabled (the [`FS_CRASH_ARM`]
/// op becomes live). Used to prove crash-consistency of a mapped store — see `demo_lmdb_crash_recovery`.
/// Never grant this to a real guest: a tripped crash freezes the store (a self-inflicted DoS on the
/// holder's own fs, no host effect, but pointless outside a test).
pub fn mem_fs_crashy() -> HostCap {
    mem_fs_cap(|| temen_fs::MemFsHandle::new(true))
}

/// A **pre-seeded** in-memory filesystem: `files` maps a normalized relative path to its contents, and
/// `dirs` names directories that must exist even with no files under them. Each host grant gets a fresh
/// clone of the seed, so a run's writes never leak back into it — re-runs are deterministic. This is how
/// a demo with no real filesystem (e.g. the browser) mounts a data-dir image on the `fs` cap.
pub fn mem_fs_seeded(files: Vec<(String, Vec<u8>)>, dirs: Vec<String>) -> HostCap {
    mem_fs_cap(move || temen_fs::MemFsHandle::seeded(&files, &dirs))
}

/// A memfs [`HostCap`] whose every grant mints a fresh store from `store` **and declares its state**
/// (#1491), so a checkpoint, a debugger rebuild and a §12 freeze carry the files, directories and
/// open descriptors the guest has — not just the continuation that expects them. A `fork()` twin
/// gets a fresh store, as [`HostCap::host_proc`] gives any `make`-built capability: charged within
/// its grantee's limits, so a fork with no room for it fails (#2428).
fn mem_fs_cap(store: impl Fn() -> temen_fs::MemFsHandle + Send + Sync + 'static) -> HostCap {
    HostCap::host_proc(0, move |payer, mint| {
        let fs = store();
        let bound = match mint {
            crate::Mint::Grant => {
                fs.bind(payer);
                true
            }
            crate::Mint::Inherit => fs.bind_within(payer),
        };
        bound.then(|| (fs.handler(), fs.cap_state()))
    })
}

/// Walk a host directory into a [`mem_fs_seeded`] capability — every regular file's bytes keyed by its
/// path relative to `root`, plus every directory (so empty ones survive). For building a demo data
/// image from an on-disk cluster; the resulting cap has no further tie to the host filesystem.
pub fn mem_fs_from_host_dir(root: &Path) -> std::io::Result<HostCap> {
    let (files, dirs) = temen_fs::read_host_dir(root)?;
    Ok(mem_fs_seeded(files, dirs))
}

/// Mount a [`encode_image`] data-image blob as an in-memory `fs` cap — the browser/demo path (a shipped
/// image, no host filesystem). Decode + [`mem_fs_seeded`].
pub fn mem_fs_from_archive(bytes: &[u8]) -> Result<HostCap, String> {
    let (files, dirs) = temen_fs::decode_image(bytes)?;
    Ok(mem_fs_seeded(files, dirs))
}

/// #2306 — what the names of an `opendir` snapshot cost, a [`TABLE_ENTRY`] each.
fn listing_cost(names: &[String]) -> u64 {
    names.iter().map(|n| TABLE_ENTRY + n.len() as u64).sum()
}

struct HostOpen {
    file: std::fs::File,
    readable: bool,
    writable: bool,
    _held: Descriptor, // #2306 — `file`'s share of the process's descriptors
}

/// A live `mmap` over the real fs: the guest buffer `[base, base+len)` is bound to the open file at
/// `open_idx`, starting at `file_off`. `msync` `pwrite`s a sub-range back through that file.
struct HostMapping {
    base: u64,
    len: u64,
    open_idx: usize,
    file_off: u64,
}

struct HostFsState {
    /// The granted directory's path.
    root: PathBuf,
    /// The granted directory itself, once a path op has opened it ([`Self::dir`]), and the
    /// descriptor of the process's it holds (#2306).
    dir: Option<(Dir, Descriptor)>,
    open: Vec<Option<HostOpen>>,
    /// Snapshots taken by [`FS_OPENDIR`]: `opendirs[dh]` is the remaining child names to yield
    /// (sorted, so a walk is deterministic and matches `mem_fs`).
    opendirs: Vec<Option<Vec<String>>>,
    maps: Vec<HostMapping>,
    /// `Some` only on the `host_fs_crashy` variant (test-only crash injection); `None` on `host_fs`.
    crash: Option<CrashCtl>,
    /// #2306 — what the guest's `opendir` snapshots and mappings cost, a [`TABLE_ENTRY`] each and
    /// their names, charged to the domain the capability was granted to. (Its open files are bounded
    /// by the process's descriptors.)
    tables: HostCharge,
}

impl HostFsState {
    /// A fresh state over `root`, its tables charged to `payer`.
    fn new(root: PathBuf, crash: Option<CrashCtl>, payer: &NodeRef) -> HostFsState {
        HostFsState {
            root,
            dir: None,
            open: Vec::new(),
            opendirs: Vec::new(),
            maps: Vec::new(),
            crash,
            tables: HostCharge::new(Some(payer.clone())),
        }
    }

    /// Charge `n` more to [`Self::tables`]: `false`, charging nothing, when the payer has no room.
    fn grow_tables(&mut self, n: u64) -> bool {
        self.tables.resize(self.tables.bytes() + n)
    }

    /// Hand `n` back from [`Self::tables`].
    fn shrink_tables(&mut self, n: u64) {
        self.tables.resize(self.tables.bytes().saturating_sub(n));
    }

    /// A durability barrier (`msync`/`sync`): `true` ⇒ drop this write (crashed or crashing now).
    fn crash_barrier(&mut self) -> bool {
        self.crash.as_mut().is_some_and(CrashCtl::barrier)
    }
    /// Whether the backing store is frozen by a tripped crash (persisting ops become no-ops).
    fn crash_frozen(&self) -> bool {
        self.crash.as_ref().is_some_and(|c| c.crashed)
    }

    /// The granted directory every path resolves beneath ([`open_root`]).
    fn dir(&mut self) -> Result<&Dir, i64> {
        open_root(&mut self.dir, &self.root)
    }

    /// The guest's path at window `(p, l)`, and the granted directory to resolve it beneath.
    fn path_at(
        &mut self,
        mem: Option<&dyn GuestMem>,
        p: i64,
        l: i64,
    ) -> Result<(&Dir, String), i64> {
        let path = read_path(mem, p, l)?;
        Ok((self.dir()?, path))
    }

    fn handle(
        &mut self,
        op: u32,
        args: &[i64],
        mem: Option<&mut dyn GuestMem>,
        minter: Option<&mut dyn RegionMinter>,
    ) -> i64 {
        let mut mem = mem;
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        match op {
            FS_OPEN => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                let flags = a(2);
                let mut oo = cap_std::fs::OpenOptions::new();
                oo.read(flags & O_READ != 0);
                if flags & O_APPEND != 0 {
                    oo.append(true);
                } else {
                    oo.write(flags & O_WRITE != 0);
                }
                oo.truncate(flags & O_TRUNC != 0)
                    .create(flags & O_CREATE != 0);
                // #2306 — the open file holds one of the process's descriptors while it is open.
                let held = match Descriptor::take() {
                    Ok(held) => held,
                    Err(errno) => return errno,
                };
                let file = match dir.open_with(path, &oo) {
                    Ok(f) => f.into_std(),
                    Err(e) => return io_errno(&e),
                };
                let o = HostOpen {
                    file,
                    readable: flags & O_READ != 0,
                    writable: flags & (O_WRITE | O_APPEND) != 0,
                    _held: held,
                };
                let fd = alloc_fd(&mut self.open);
                self.open[fd] = Some(o);
                fd as i64
            }
            FS_READ => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if !o.readable {
                    return EBADF;
                }
                let (buf, len) = (a(1), a(2));
                if buf < 0 || !(0..=(1 << 30)).contains(&len) {
                    return EINVAL;
                }
                if len == 0 {
                    return 0;
                }
                let Some(m) = mem.as_deref_mut() else {
                    return EFAULT;
                };
                // #2306 — a piece at a time, so a read as large as the guest's window never costs
                // the host more than a piece. A short read is the file's end (or a pipe's lull).
                let mut piece = vec![0u8; (len as usize).min(COPY_CHUNK)];
                let mut done = 0;
                while done < len as usize {
                    let want = (len as usize - done).min(piece.len());
                    let n = match o.file.read(&mut piece[..want]) {
                        Ok(n) => n,
                        Err(_) if done > 0 => break,
                        Err(e) => return io_errno(&e),
                    };
                    if m.write_bytes(buf as u64 + done as u64, &piece[..n])
                        .is_none()
                    {
                        return EFAULT;
                    }
                    done += n;
                    if n < want {
                        break;
                    }
                }
                done as i64
            }
            FS_WRITE => {
                if self.crash_frozen() {
                    return a(2).max(0); // power-loss: the un-synced write is silently dropped
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if !o.writable {
                    return EBADF;
                }
                let bytes = match mem.as_deref() {
                    Some(m) => match m.read_bytes(a(1) as u64, a(2) as u64) {
                        Some(b) => b,
                        None => return EFAULT,
                    },
                    None => return EFAULT,
                };
                match o.file.write_all(&bytes) {
                    Ok(()) => bytes.len() as i64,
                    Err(e) => io_errno(&e),
                }
            }
            FS_SEEK => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                let from = match a(1) {
                    0 => SeekFrom::Start(a(2).max(0) as u64),
                    1 => SeekFrom::Current(a(2)),
                    2 => SeekFrom::End(a(2)),
                    _ => return EINVAL,
                };
                match o.file.seek(from) {
                    Ok(p) => p as i64,
                    Err(e) => io_errno(&e),
                }
            }
            FS_CLOSE => {
                let Some(slot) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if slot.take().is_none() {
                    return EBADF;
                }
                0
            }
            FS_REMOVE => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                match unlink(dir, &path) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_RENAME => {
                let from = match read_path(mem.as_deref(), a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let (dir, to) = match self.path_at(mem.as_deref(), a(2), a(3)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                match rename(dir, &from, &to) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_TRUNCATE => {
                if self.crash_frozen() {
                    return 0; // power-loss: the resize never reaches the backing file
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if !o.writable {
                    return EBADF;
                }
                let len = a(1);
                if len < 0 {
                    return EINVAL;
                }
                match o.file.set_len(len as u64) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_SYNC => {
                // A durability barrier for crash injection, then the real fsync.
                if self.crash_barrier() {
                    return 0;
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                match o.file.sync_all() {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_MMAP => {
                let (fd, foff, len, buf) = (a(0), a(1), a(2), a(3));
                if foff < 0 || len < 0 || buf < 0 {
                    return EINVAL;
                }
                let Some(Some(o)) = self.open.get_mut(fd as usize) else {
                    return EBADF;
                };
                if o.file.seek(SeekFrom::Start(foff as u64)).is_err() {
                    return io_errno(&std::io::Error::last_os_error());
                }
                let Some(m) = mem.as_deref_mut() else {
                    return EFAULT;
                };
                // Copy the file region into the guest buffer (zero-fill past EOF), a piece at a time
                // (#2306).
                let (mut failed, mut eof) = (None, false);
                let copied = copy_to_guest(m, buf as u64, len as u64, |_, piece| {
                    let mut got = 0;
                    while !eof && failed.is_none() && got < piece.len() {
                        match o.file.read(&mut piece[got..]) {
                            Ok(0) => eof = true, // EOF — the rest stays zero
                            Ok(n) => got += n,
                            Err(e) => failed = Some(io_errno(&e)),
                        }
                    }
                });
                if let Some(errno) = failed {
                    return errno;
                }
                if !copied {
                    return EFAULT;
                }
                // #2306 — the mapping is an entry of the guest's tables.
                if !self.grow_tables(TABLE_ENTRY) {
                    return ENOSPC;
                }
                self.maps.push(HostMapping {
                    base: buf as u64,
                    len: len as u64,
                    open_idx: fd as usize,
                    file_off: foff as u64,
                });
                0
            }
            FS_MSYNC => {
                let (buf, len) = (a(0), a(1));
                if buf < 0 || len < 0 {
                    return EINVAL;
                }
                let Some(map) = self
                    .maps
                    .iter()
                    .find(|m| buf as u64 >= m.base && (buf as u64) < m.base + m.len)
                else {
                    return EINVAL;
                };
                let n = (len as u64).min(map.base + map.len - buf as u64) as usize;
                let file_pos = map.file_off + (buf as u64 - map.base);
                let open_idx = map.open_idx;
                let Some(m) = mem.as_deref() else {
                    return EFAULT;
                };
                let Some(bytes) = m.read_bytes(buf as u64, n as u64) else {
                    return EFAULT;
                };
                if self.crash_barrier() {
                    return 0; // power-loss: this msync's bytes never reach the file
                }
                let Some(Some(o)) = self.open.get_mut(open_idx) else {
                    return EBADF; // the mapped fd was closed
                };
                if o.file.seek(SeekFrom::Start(file_pos)).is_err() {
                    return io_errno(&std::io::Error::last_os_error());
                }
                match o.file.write_all(&bytes) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_MUNMAP => {
                let buf = a(0);
                let Some(idx) = self.maps.iter().position(|m| m.base == buf as u64) else {
                    return EINVAL;
                };
                let map = self.maps.remove(idx);
                self.shrink_tables(TABLE_ENTRY);
                // Final flush of the whole mapping (self-contained munmap) — unless a crash froze the
                // store, in which case a real munmap on a dead process would flush nothing.
                if !self.crash_frozen() {
                    let bytes = mem.as_deref().and_then(|m| m.read_bytes(map.base, map.len));
                    if let (Some(bytes), Some(Some(o))) = (bytes, self.open.get_mut(map.open_idx)) {
                        let _ = o.file.seek(SeekFrom::Start(map.file_off));
                        let _ = o.file.write_all(&bytes);
                    }
                }
                0
            }
            FS_CRASH_ARM => arm_crash(self.crash.as_mut(), a(0)),
            // §4b zero-copy: mint a file-backed `SharedRegion` over the open file and return its
            // handle for the guest to `SharedRegion.map` into its window (real MAP_SHARED aliasing).
            FS_MAP_REGION => {
                let Some(minter) = minter else {
                    return EINVAL; // not an mmap-capable grant (plain host_fs / mem_fs)
                };
                let (fd, foff, len) = (a(0), a(1), a(2));
                if foff != 0 || len <= 0 {
                    return EINVAL; // v1 maps the whole file from offset 0
                }
                let Some(Some(o)) = self.open.get(fd as usize) else {
                    return EBADF;
                };
                // The backing holds its **own** fd (a dup) over the same OS file, so it outlives the
                // guest's fd and both share one page cache — the map and the fs cap's pread/pwrite/
                // fsync stay coherent. Unix only for now (the `FileBacking` is `#[cfg(unix)]`);
                // elsewhere the op is unavailable and the guest shim falls back to copy-in.
                #[cfg(unix)]
                {
                    match crate::new_file_region(&o.file, len as usize) {
                        Ok(backing) => minter.grant_region(backing) as i64,
                        Err(errno) => errno,
                    }
                }
                #[cfg(not(unix))]
                {
                    let _ = (minter, o, len); // region-mapping is unix-only for now
                    EINVAL
                }
            }
            FS_STAT => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                // lstat semantics (symlink_metadata): a symlink is reported as itself.
                let md = match dir.symlink_metadata(path) {
                    Ok(m) => m,
                    Err(e) => return io_errno(&e),
                };
                let buf = host_stat_bytes(&md);
                if a(2) < 0 || a(3) < STATBUF_LEN as i64 {
                    return EINVAL;
                }
                let Some(m) = mem.as_deref_mut() else {
                    return EFAULT;
                };
                if m.write_bytes(a(2) as u64, &buf).is_some() {
                    0
                } else {
                    EFAULT
                }
            }
            FS_MKDIR => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                match mkdir(dir, &path) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_RMDIR => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                match rmdir(dir, &path) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            FS_OPENDIR => {
                let (dir, path) = match self.path_at(mem.as_deref(), a(0), a(1)) {
                    Ok(at) => at,
                    Err(e) => return e,
                };
                let listing = match names(dir, &path) {
                    Ok(names) => names,
                    Err(e) => return io_errno(&e),
                };
                // #2306 — the snapshot is the guest's until it closes it: an entry, and one per
                // name, charged as each is read, so a directory the payer has no room to list is
                // refused before it is all in memory.
                if !self.grow_tables(TABLE_ENTRY) {
                    return ENOSPC;
                }
                let mut names: Vec<String> = Vec::new();
                for name in listing {
                    let refused = match name {
                        // Names are OS strings; a non-UTF-8 entry can't round-trip through the guest's
                        // UTF-8 path protocol, so skip it (it was never nameable anyway).
                        Ok(name) => match name.to_str() {
                            Some(s) if !self.grow_tables(TABLE_ENTRY + s.len() as u64) => ENOSPC,
                            Some(s) => {
                                names.push(s.to_string());
                                continue;
                            }
                            None => continue,
                        },
                        Err(e) => io_errno(&e),
                    };
                    self.shrink_tables(TABLE_ENTRY + listing_cost(&names));
                    return refused;
                }
                names.sort();
                names.reverse(); // pop() yields sorted order
                let dh = self
                    .opendirs
                    .iter()
                    .position(|s| s.is_none())
                    .unwrap_or_else(|| {
                        self.opendirs.push(None);
                        self.opendirs.len() - 1
                    });
                self.opendirs[dh] = Some(names);
                dh as i64
            }
            FS_READDIR => {
                let Some(Some(entries)) = self.opendirs.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                let Some(name) = entries.pop() else {
                    return 0; // exhausted
                };
                let (ptr, cap) = (a(1), a(2));
                if ptr < 0 || cap < name.len() as i64 {
                    entries.push(name);
                    return EINVAL;
                }
                self.shrink_tables(TABLE_ENTRY + name.len() as u64);
                let Some(m) = mem else {
                    return EFAULT;
                };
                if m.write_bytes(ptr as u64, name.as_bytes()).is_some() {
                    name.len() as i64
                } else {
                    EFAULT
                }
            }
            FS_CLOSEDIR => {
                let Some(Some(names)) = self.opendirs.get_mut(a(0) as usize).map(Option::take)
                else {
                    return EBADF;
                };
                self.shrink_tables(TABLE_ENTRY + listing_cost(&names));
                0
            }
            FS_REPLACE => {
                let path = match read_path(mem.as_deref(), a(0), a(1)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let (buf, len) = (a(2), a(3));
                if buf < 0 || len < 0 {
                    return EINVAL;
                }
                let Some(bytes) = mem
                    .as_deref()
                    .and_then(|m| m.read_bytes(buf as u64, len as u64))
                else {
                    return EFAULT;
                };
                if self.crash_frozen() {
                    return 0; // power-loss: nothing reaches the disk
                }
                let root = match open_root(&mut self.dir, &self.root) {
                    Ok(root) => root,
                    Err(e) => return e,
                };
                match replace(root, &path, &bytes, &mut self.crash) {
                    Ok(()) => 0,
                    Err(e) => io_errno(&e),
                }
            }
            _ => EINVAL,
        }
    }
}

/// The granted directory every path resolves beneath (#2643), opened by the first path op into
/// `dir` and held there from then on: the grant names that directory wherever the host later moves
/// `root`. A root that can't be opened yet fails the op with the `-errno` the open got, and the
/// next op tries again.
///
/// Each op resolves its path beneath it with cap-std, the resolver Wasmtime's preopens use.
/// `read_path` has already refused absolute paths and `..`; cap-std refuses a symlink that would
/// lead out, `-EACCES` alike. The op then acts on what was resolved (on Windows, five of them
/// through `win`), so a link the host makes in the tree mid-op can't redirect it.
fn open_root<'a>(dir: &'a mut Option<(Dir, Descriptor)>, root: &Path) -> Result<&'a Dir, i64> {
    let open = match dir.take() {
        Some(open) => open,
        None => {
            let held = Descriptor::take()?;
            match Dir::open_ambient_dir(root, cap_std::ambient_authority()) {
                Ok(opened) => (opened, held),
                Err(e) => return Err(io_errno(&e)),
            }
        }
    };
    Ok(&dir.insert(open).0)
}

/// #2510 — [`FS_REPLACE`]: `path` beneath `root` holds exactly `bytes`, atomically and durably,
/// with the target's protection. `crash` counts the op's three crash points (`*_crashy`).
///
/// The bytes go to a temporary file in the target's directory, which takes the target's protection
/// once written (writing clears set-user-ID bits) and is then synced and renamed over the target.
/// Until the rename, an error removes it (`os::Temp` does as it is dropped).
fn replace(
    root: &Dir,
    path: &str,
    bytes: &[u8],
    crash: &mut Option<CrashCtl>,
) -> std::io::Result<()> {
    let mut crashed = || crash.as_mut().is_some_and(CrashCtl::barrier);
    let (dir, name) = replace_target(root, path)?;
    let target = os::target(&dir, &name)?;
    let mut temp = os::Temp::create(&dir, &temp_name(&name), target.as_ref())?;
    temp.file.write_all(bytes)?;
    if let Some(target) = &target {
        temp.protect(target)?;
    }
    // A crash at the temporary file's sync, or at the rename, leaves the old file, and the
    // temporary file as far as it got for the next replace to remove.
    if crashed() {
        temp.keep();
        return Ok(());
    }
    temp.sync()?;
    if crashed() {
        temp.keep();
        return Ok(());
    }
    temp.rename_over(&name)?;
    if crashed() {
        return Ok(()); // the directory's sync: the rename stands
    }
    temp.sync_dir()
}

/// R2 (#2510) — where [`FS_REPLACE`] writes for the guest's `path`: the directory beneath `root`
/// that holds the file, open, and the file's name there. A symlink is followed, beneath `root` as
/// every path's links are (#2643): the file it leads to is replaced and the link kept, and a
/// dangling one leads to the file the replace creates. Past 40 links, `-ELOOP`, as Linux counts.
fn replace_target(root: &Dir, path: &str) -> std::io::Result<(Dir, String)> {
    let mut path = path.to_owned();
    for _ in 0..=40 {
        let (parent, name) = path
            .rsplit_once(std::path::is_separator)
            .unwrap_or(("", &path));
        if matches!(name, "" | "." | "..") {
            root.open_dir(&path)?; // a directory, if it names anything
            return Err(refuse(EISDIR));
        }
        let dir = root.open_dir(if parent.is_empty() { "." } else { parent })?;
        match dir.symlink_metadata(name) {
            Ok(md) if md.file_type().is_symlink() => {
                // cap-std refuses an absolute link, -EACCES, as a path that leads out.
                let link = dir.read_link(name)?;
                let link = link.to_str().ok_or(std::io::ErrorKind::InvalidInput)?;
                path = match parent {
                    "" => link.to_owned(),
                    _ => format!("{parent}/{link}"),
                };
            }
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            _ => return Ok((dir, name.to_owned())),
        }
    }
    Err(refuse(ELOOP))
}

/// The name of [`FS_REPLACE`]'s temporary file for the file `name`: [`REPLACE_TEMP`] and the
/// name's 64-bit FNV-1a hash in hex. Each file has its own, so a crash's leftover is the next
/// replace's to remove, and any name's fits in a name.
fn temp_name(name: &str) -> String {
    let hash = name.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    });
    format!("{REPLACE_TEMP}{hash:016x}")
}

/// A protocol `-errno` carried in an `io::Error`, for a refusal of `host_fs`'s own that no
/// `io::ErrorKind` names exactly: `TooManyLinks` is `EMLINK` on Unix, but cap-std's symlink loop
/// on Windows.
#[derive(Debug)]
struct Errno(i64);

impl std::fmt::Display for Errno {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "errno {}", -self.0)
    }
}

impl std::error::Error for Errno {}

/// A refusal of `host_fs`'s own, which [`io_errno`] returns as `errno`.
fn refuse(errno: i64) -> std::io::Error {
    std::io::Error::other(Errno(errno))
}

// #2643 — the last step of five path ops, once cap-std has resolved the path beneath the root. Off
// Windows these are cap-std's own, which act on the handle it resolved. On Windows cap-std rebuilds
// a path from that handle for the OS to resolve again, which a directory turned into a junction in
// between redirects, so `win` acts on handles there.
#[cfg(not(windows))]
fn mkdir(dir: &Dir, path: &str) -> std::io::Result<()> {
    dir.create_dir(path)
}
#[cfg(not(windows))]
fn rmdir(dir: &Dir, path: &str) -> std::io::Result<()> {
    dir.remove_dir(path)
}
#[cfg(not(windows))]
fn unlink(dir: &Dir, path: &str) -> std::io::Result<()> {
    dir.remove_file(path)
}
#[cfg(not(windows))]
fn rename(dir: &Dir, from: &str, to: &str) -> std::io::Result<()> {
    dir.rename(from, dir, to)
}
/// The names in the directory `path` beneath `dir`, read as they are asked for.
#[cfg(not(windows))]
fn names(
    dir: &Dir,
    path: &str,
) -> std::io::Result<impl Iterator<Item = std::io::Result<std::ffi::OsString>>> {
    Ok(dir
        .read_dir(path)?
        .map(|entry| entry.map(|e| e.file_name())))
}
#[cfg(windows)]
mod win;
#[cfg(windows)]
use win::{mkdir, names, rename, rmdir, unlink};

// #2510 — `replace`'s steps on the target's directory: `unix` acts on descriptors, `win` on
// handles.
#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix as os;
#[cfg(windows)]
use win as os;

/// Fill a [`StatBuf`] from real-fs metadata. On unix the `st_*` fields come straight from the
/// `stat(2)` the OS already did; elsewhere the portable `Metadata` fills type + size and the rest
/// stays zero (the guest libc only ever needs type + size + mtime to walk an `initdb` tree).
fn host_stat_bytes(md: &cap_std::fs::Metadata) -> [u8; STATBUF_LEN] {
    #[cfg(unix)]
    {
        use cap_std::fs::MetadataExt;
        stat_bytes(
            md.mode(),
            md.nlink() as u32,
            md.size() as i64,
            md.mtime(),
            md.mtime_nsec(),
            md.ino(),
            md.dev(),
            md.uid(),
            md.gid(),
            md.blksize() as i64,
            md.blocks() as i64,
        )
    }
    #[cfg(not(unix))]
    {
        let ty = if md.is_dir() {
            S_IFDIR | 0o755
        } else if md.file_type().is_symlink() {
            S_IFLNK | 0o777
        } else {
            S_IFREG | 0o644
        };
        stat_bytes(ty, 1, md.len() as i64, 0, 0, 0, 0, 0, 0, 4096, 0)
    }
}

/// Map a host `io::Error` to the protocol's canonical (Linux) **negative** errno — ready to return
/// directly (the caller no longer negates). The wire protocol must be **host-OS-independent** — both
/// backends have to return the same value for the same failure, and on a differential a `host_fs` run
/// has to match a `mem_fs` run — but the raw OS error code is not portable: `ENOTEMPTY` is 39 on Linux
/// and 66 on macOS, and Windows codes are not errno at all. So canonicalize through the portable
/// `io::ErrorKind` (the shared `temen_ir::errno` table), falling back to the negated raw errno (the
/// canonical value on Linux, the differential's reference host) for kinds without a fixed mapping.
pub(crate) fn io_errno(e: &std::io::Error) -> i64 {
    use std::io::ErrorKind;
    if let Some(Errno(errno)) = e.get_ref().and_then(|inner| inner.downcast_ref()) {
        return *errno;
    }
    match e.kind() {
        ErrorKind::NotFound => ENOENT,
        ErrorKind::PermissionDenied => EACCES,
        ErrorKind::AlreadyExists => EEXIST,
        ErrorKind::DirectoryNotEmpty => ENOTEMPTY,
        ErrorKind::NotADirectory => ENOTDIR,
        ErrorKind::InvalidInput => EINVAL,
        _ => e.raw_os_error().map(|c| -(c as i64)).unwrap_or(EINVAL),
    }
}

/// The **real** filesystem, attenuated to `root` (relative paths only; `..`/absolute refused). The
/// guest sees exactly the subtree the embedder granted — nothing else is nameable, through a
/// symlink either: one that stays beneath `root` works as its target does, and one that leads out
/// is refused with `-EACCES`, as `..` is (#2643). A grant opens `root` at its first path op and holds
/// it from then on, so it keeps the directory it opened wherever the host later moves `root`.
pub fn host_fs(root: PathBuf) -> HostCap {
    host_fs_impl(root, false)
}

/// Like [`host_fs`] but with the **test-only crash-injection** controller enabled (the
/// [`FS_CRASH_ARM`] op becomes live). Proves crash-consistency of a real on-disk mapped store — see
/// `demo_lmdb_crash_recovery`. Never grant this to a real guest.
pub fn host_fs_crashy(root: PathBuf) -> HostCap {
    host_fs_impl(root, true)
}

/// Like [`host_fs`] but **mmap-capable** (§4b): granted with region-minting authority so the
/// [`FS_MAP_REGION`] op is live. A guest that maps a file gets a **real** `MAP_SHARED` alias of that
/// file into its window (zero-copy, page-cache coherent with the same file's `pread`/`pwrite`) instead
/// of the copy-in emulation. The guest shim falls back to `FS_MMAP` when granted a plain `host_fs`.
pub fn host_fs_mmap(root: PathBuf) -> HostCap {
    HostCap::host_proc_region(0, move |payer| {
        let mut st = HostFsState::new(root.clone(), None, payer);
        let h = Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  minter: Option<&mut dyn RegionMinter>| {
                Ok(vec![st.handle(op, args, mem, minter)])
            },
        ) as HostProc;
        // Open host files and directories, guest-visible cursors: `Uncaptured` (#1699).
        (h, temen_interp::CapState::Uncaptured)
    })
}

fn host_fs_impl(root: PathBuf, crashy: bool) -> HostCap {
    HostCap::host_proc(0, move |payer, _| {
        let mut st = HostFsState::new(root.clone(), crashy.then(CrashCtl::default), payer);
        let h = Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  _minter: Option<&mut dyn temen_interp::RegionMinter>| {
                Ok(vec![st.handle(op, args, mem, None)])
            },
        ) as HostProc;
        // Open host files and directories, guest-visible cursors: `Uncaptured` (#1699).
        Some((h, temen_interp::CapState::Uncaptured))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use temen_interp::{cap_id, Host, WindowMem};

    /// Drive fs op `op` on handle `h` over a fresh view of `win`, as a guest's `__vm_host_call`
    /// does. `win` persists between calls, so the paths put there stay put.
    fn call(host: &mut Host, h: i32, win: &mut [u8], op: u32, args: &[i64]) -> i64 {
        let n = win.len();
        let mut wm = WindowMem::new(win, n as u64);
        host.cap_dispatch_slots(cap_id::HOST_PROC, op, h, args, Some(&mut wm))
            .unwrap()[0]
    }

    /// Put `s` in `win` at `at`: the `(ptr, len)` a guest passes for a path or a buffer.
    fn put(win: &mut [u8], at: usize, s: impl AsRef<[u8]>) -> (i64, i64) {
        let s = s.as_ref();
        win[at..at + s.len()].copy_from_slice(s);
        (at as i64, s.len() as i64)
    }

    /// `temen-llvm` pins the `HostProc` interface id numerically (it produces `temen-ir` and cannot depend
    /// on the interpreter crate); this locks the pin to the real constant.
    #[test]
    fn host_proc_type_id_matches() {
        assert_eq!(temen_interp::cap_id::HOST_PROC, 13);
    }

    /// `temen-llvm` pins `SharedRegion`'s interface id numerically (`SHARED_REGION_TYPE_ID`, for the
    /// `__vm_region_call` the zero-copy bridge lowers to); this locks that pin to the real constant.
    #[test]
    fn shared_region_type_id_matches() {
        assert_eq!(temen_interp::cap_id::SHARED_REGION, 4);
    }

    /// `temen-llvm` pins `Instantiator`'s interface id numerically (`INSTANTIATOR_TYPE_ID`, for the
    /// `__vm_instantiate_rec`/`__vm_join` §14 spawn builtins a guest driver lowers to); this locks that
    /// pin to the real constant.
    #[test]
    fn instantiator_type_id_matches() {
        assert_eq!(temen_interp::cap_id::INSTANTIATOR, 6);
    }

    /// `temen-llvm` pins `ModuleLoader`'s interface id numerically (`MODULE_LOADER_TYPE_ID`, for the
    /// `__vm_module_from_bytes` run-in-guest builtin a guest driver lowers to); this locks that pin to
    /// the real constant.
    #[test]
    fn module_loader_type_id_matches() {
        assert_eq!(temen_interp::cap_id::MODULE_LOADER, 7);
    }

    /// The §4b zero-copy op end to end at the capability boundary: `host_fs_mmap`'s `FS_MAP_REGION`
    /// opens a real file and mints a **file-backed `SharedRegion`** whose size matches, while a plain
    /// `host_fs` refuses the op (no minting authority). Combined with `file_region_tests` (which prove
    /// a `FileBacking` aliases the real file's bytes), this establishes the whole delivery path:
    /// `FS_MAP_REGION` → `FileBacking` → a live `SharedRegion` the guest can `SharedRegion.map`.
    #[cfg(unix)]
    #[test]
    fn map_region_op_mints_a_file_backed_region_only_on_the_mmap_grant() {
        use std::io::Write;

        let dir = std::env::temp_dir().join(format!("temen_fs_mapregion_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        {
            let mut f = std::fs::File::create(dir.join("data")).unwrap();
            f.write_all(b"ZEROCOPY").unwrap();
            f.set_len(8192).unwrap();
        }

        // A tiny flat window holding the path "data" at offset 0.
        let mut win = vec![0u8; 4096];
        win[..4].copy_from_slice(b"data");
        let open_args = [0i64, 4, O_READ | O_WRITE, 0];

        // 1. The mmap-capable grant: open → map_region mints a live region of the file's size.
        {
            let mut host = Host::new();
            let cap = host_fs_mmap(dir.clone());
            let fs_h = (cap.grant)(&mut host, 0);
            let mut wm = WindowMem::new(&mut win, 4096);
            let fd = host
                .cap_dispatch_slots(cap_id::HOST_PROC, FS_OPEN, fs_h, &open_args, Some(&mut wm))
                .unwrap()[0];
            assert!(fd >= 0, "open failed: {fd}");
            let region = host
                .cap_dispatch_slots(
                    cap_id::HOST_PROC,
                    FS_MAP_REGION,
                    fs_h,
                    &[fd, 0, 8192],
                    Some(&mut wm),
                )
                .unwrap()[0];
            assert!(
                region >= 0,
                "FS_MAP_REGION should mint a region on host_fs_mmap: {region}"
            );
            // The returned handle is a live SharedRegion; op 2 `len()` reports the mapped size.
            let len = host
                .cap_dispatch_slots(cap_id::SHARED_REGION, 2, region as i32, &[], Some(&mut wm))
                .unwrap()[0];
            assert_eq!(len, 8192, "the minted region maps the whole file");
        }

        // 2. A plain host_fs has no minting authority → the op is unavailable (-EINVAL).
        {
            let mut host = Host::new();
            let cap = host_fs(dir.clone());
            let fs_h = (cap.grant)(&mut host, 0);
            let mut wm = WindowMem::new(&mut win, 4096);
            let fd = host
                .cap_dispatch_slots(cap_id::HOST_PROC, FS_OPEN, fs_h, &open_args, Some(&mut wm))
                .unwrap()[0];
            let region = host
                .cap_dispatch_slots(
                    cap_id::HOST_PROC,
                    FS_MAP_REGION,
                    fs_h,
                    &[fd, 0, 8192],
                    Some(&mut wm),
                )
                .unwrap()[0];
            assert_eq!(region, EINVAL, "plain host_fs must refuse FS_MAP_REGION");
        }

        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The metadata + directory surface (`stat`/`mkdir`/`rmdir`/`opendir`/`readdir`/`closedir`) is
    /// protocol-identical on both backends: the *same* scripted walk returns the *same* rc sequence,
    /// the same file/dir type bits, and the same size on `mem_fs` (flat-map dirs) and `host_fs` (real
    /// dirs). This is the invariant a Postgres differential relies on — the guest can't tell which
    /// backend it walks.
    #[test]
    fn os_metadata_ops_parity_mem_vs_host() {
        // Window layout: [0..64) path A, [64..128) path B, [256..328) statbuf, [512..576) name buf.
        const PA: usize = 0;
        const PB: usize = 64;
        const SB: usize = 256;
        const NB: usize = 512;

        // The scripted walk: returns (rc-sequence, stat-of-dir mode&S_IFMT, stat-of-file (mode,size),
        // readdir name). Run identically against whatever cap is granted.
        fn walk(host: &mut Host, h: i32, win: &mut [u8]) -> (Vec<i64>, u32, (u32, i64), String) {
            let mut rc = Vec::new();
            let (dp, dl) = put(win, PA, "d");
            rc.push(call(host, h, win, FS_MKDIR, &[dp, dl, 0, 0])); // 0
            rc.push(call(host, h, win, FS_MKDIR, &[dp, dl, 0, 0])); // -EEXIST
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[dp, dl, SB as i64, STATBUF_LEN as i64],
            )); // 0
            let dir_mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap()) & S_IFMT;

            let (fp, fl) = put(win, PB, "d/f");
            let fd = call(host, h, win, FS_OPEN, &[fp, fl, O_CREATE | O_WRITE, 0]);
            rc.push((fd >= 0) as i64); // 1 (fd is backend-specific; normalize to a bool)
            let (bp, bl) = put(win, NB, "hello");
            rc.push(call(host, h, win, FS_WRITE, &[fd, bp, bl, 0])); // 5
            rc.push(call(host, h, win, FS_CLOSE, &[fd, 0, 0, 0])); // 0

            // re-put the path (the write clobbered NB, not PB, but keep it explicit)
            let (fp, fl) = put(win, PB, "d/f");
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[fp, fl, SB as i64, STATBUF_LEN as i64],
            )); // 0
            let f_mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap()) & S_IFMT;
            let f_size = i64::from_le_bytes(win[SB + 8..SB + 16].try_into().unwrap());

            let (np, nl) = put(win, PA, "d/nope");
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[np, nl, SB as i64, STATBUF_LEN as i64],
            )); // -ENOENT

            let (dp, dl) = put(win, PA, "d");
            let dh = call(host, h, win, FS_OPENDIR, &[dp, dl, 0, 0]);
            rc.push((dh >= 0) as i64); // 1
            let got = call(host, h, win, FS_READDIR, &[dh, NB as i64, 64, 0]);
            rc.push(got); // 1 (len of "f")
            let name = String::from_utf8(win[NB..NB + got.max(0) as usize].to_vec()).unwrap();
            rc.push(call(host, h, win, FS_READDIR, &[dh, NB as i64, 64, 0])); // 0 (exhausted)
            rc.push(call(host, h, win, FS_CLOSEDIR, &[dh, 0, 0, 0])); // 0

            rc.push(call(host, h, win, FS_RMDIR, &[dp, dl, 0, 0])); // -ENOTEMPTY
            let (fp, fl) = put(win, PB, "d/f");
            rc.push(call(host, h, win, FS_REMOVE, &[fp, fl, 0, 0])); // 0
            rc.push(call(host, h, win, FS_RMDIR, &[dp, dl, 0, 0])); // 0
            rc.push(call(
                host,
                h,
                win,
                FS_STAT,
                &[dp, dl, SB as i64, STATBUF_LEN as i64],
            )); // -ENOENT
            (rc, dir_mode, (f_mode, f_size), name)
        }

        // mem_fs
        let (mem_rc, mem_dir, mem_file, mem_name) = {
            let mut host = Host::new();
            let cap = mem_fs();
            let h = (cap.grant)(&mut host, 0);
            let mut win = vec![0u8; 4096];
            walk(&mut host, h, &mut win)
        };

        // host_fs over a fresh temp root
        let root = std::env::temp_dir().join(format!("temen_fs_osmeta_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let (host_rc, host_dir, host_file, host_name) = {
            let mut host = Host::new();
            let cap = host_fs(root.clone());
            let h = (cap.grant)(&mut host, 0);
            let mut win = vec![0u8; 4096];
            walk(&mut host, h, &mut win)
        };
        let _ = std::fs::remove_dir_all(&root);

        let expected: Vec<i64> = vec![
            0, EEXIST, 0, 1, 5, 0, 0, ENOENT, 1, 1, 0, 0, ENOTEMPTY, 0, 0, ENOENT,
        ];
        assert_eq!(mem_rc, expected, "mem_fs rc sequence");
        assert_eq!(host_rc, expected, "host_fs rc sequence (must match mem_fs)");
        assert_eq!(mem_dir, S_IFDIR, "mem_fs: 'd' is a directory");
        assert_eq!(host_dir, S_IFDIR, "host_fs: 'd' is a directory");
        assert_eq!(
            mem_file,
            (S_IFREG, 5),
            "mem_fs: 'd/f' is a 5-byte regular file"
        );
        assert_eq!(
            host_file,
            (S_IFREG, 5),
            "host_fs: 'd/f' is a 5-byte regular file"
        );
        assert_eq!(mem_name, "f");
        assert_eq!(host_name, "f");
    }

    /// A data image round-trips through [`encode_image`]/[`decode_image`] byte-exact, and mounts via
    /// [`mem_fs_from_archive`] into a working fs — the shippable-artifact path (no host filesystem).
    #[test]
    fn data_image_roundtrip_and_mount() {
        let files = vec![
            ("PG_VERSION".to_string(), b"17\n".to_vec()),
            ("base/1/1259".to_string(), vec![0u8; 300]),
            ("global/pg_control".to_string(), b"\x01\x02\x03".to_vec()),
        ];
        let dirs = vec!["base/1".to_string(), "pg_logical/mappings".to_string()];

        let img = encode_image(&files, &dirs);
        let (df, dd) = decode_image(&img).expect("decode");
        assert_eq!(df, files, "files round-trip byte-exact");
        assert_eq!(dd, dirs, "dirs round-trip");
        assert!(decode_image(b"nope").is_err(), "bad magic fails closed");
        assert!(
            decode_image(&img[..img.len() - 5]).is_err(),
            "truncation fails closed"
        );

        // Mount the image and read a file back through the cap.
        let cap = mem_fs_from_archive(&img).expect("mount archive");
        let mut host = Host::new();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        win[..10].copy_from_slice(b"PG_VERSION");
        let mut wm = WindowMem::new(&mut win, 4096);
        let fd = host
            .cap_dispatch_slots(
                cap_id::HOST_PROC,
                FS_OPEN,
                h,
                &[0, 10, O_READ, 0],
                Some(&mut wm),
            )
            .unwrap()[0];
        assert!(fd >= 0, "open a file from the mounted image");
        let n = host
            .cap_dispatch_slots(
                cap_id::HOST_PROC,
                FS_READ,
                h,
                &[fd, 512, 8, 0],
                Some(&mut wm),
            )
            .unwrap()[0];
        assert_eq!(n, 3);
        assert_eq!(&win[512..515], b"17\n", "file contents from the image");
    }

    /// A **pre-seeded** in-memory fs ([`mem_fs_seeded`]) exposes its files and empty dirs, tolerates a
    /// `./` path prefix (all file ops normalize consistently), reports its data-dir-style `0700` dir
    /// mode, and lets a directory be `open`ed read-only and `sync`ed — the exact surface the Postgres
    /// demo boot needs from a virtual filesystem (`BOOTSPEED.md` Milestone A).
    #[test]
    fn seeded_mem_fs_open_norm_and_dir_fsync() {
        const PA: usize = 0;
        const SB: usize = 256;
        const RB: usize = 512;

        // Seed a file under a subdir + an *empty* subdir (the shape of `pg_logical/mappings`).
        let cap = mem_fs_seeded(
            vec![("sub/PG_VERSION".into(), b"17\n".to_vec())],
            vec!["sub/mappings".into()],
        );
        let mut host = Host::new();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];

        // Open the seeded file through a `./`-prefixed path — the raw key ("./sub/PG_VERSION") differs
        // from the normalized seed key, so this fails unless the op normalizes (the bug this guards).
        let (fp, fl) = put(&mut win, PA, "./sub/PG_VERSION");
        let fd = call(&mut host, h, &mut win, FS_OPEN, &[fp, fl, O_READ, 0]);
        assert!(fd >= 0, "open ./sub/PG_VERSION (normalized) succeeds");
        let n = call(&mut host, h, &mut win, FS_READ, &[fd, RB as i64, 8, 0]);
        assert_eq!(n, 3, "read the seeded 3 bytes");
        assert_eq!(&win[RB..RB + 3], b"17\n", "seeded file contents");
        call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]);

        // Open the *empty* seeded directory read-only and fsync it — Postgres does this at checkpoint.
        let (dp, dl) = put(&mut win, PA, "sub/mappings");
        let dfd = call(&mut host, h, &mut win, FS_OPEN, &[dp, dl, O_READ, 0]);
        assert!(dfd >= 0, "open a directory read-only (for fsync) succeeds");
        assert_eq!(
            call(&mut host, h, &mut win, FS_SYNC, &[dfd, 0, 0, 0]),
            0,
            "fsync of a directory fd succeeds"
        );
        call(&mut host, h, &mut win, FS_CLOSE, &[dfd, 0, 0, 0]);

        // A seeded directory stats as 0700 (Postgres' `checkDataDir` rejects group/other perms).
        let (sp, sl) = put(&mut win, PA, "sub");
        assert_eq!(
            call(
                &mut host,
                h,
                &mut win,
                FS_STAT,
                &[sp, sl, SB as i64, STATBUF_LEN as i64]
            ),
            0
        );
        let mode = u32::from_le_bytes(win[SB..SB + 4].try_into().unwrap());
        assert_eq!(mode & S_IFMT, S_IFDIR, "seeded 'sub' is a directory");
        assert_eq!(mode & 0o777, 0o700, "in-memory dir reports owner-only 0700");

        // The directory-open path is deliberately narrow: a read-only open of a name that is neither a
        // file nor a *tracked* directory is still `-ENOENT` — it must not resolve arbitrary paths to a
        // directory fd (the over-broad `is_dir` version silently opened file-key prefixes and the root,
        // which broke ordinary `open`s on the general in-memory fs).
        let (np, nl) = put(&mut win, PA, "sub/nope");
        assert_eq!(
            call(&mut host, h, &mut win, FS_OPEN, &[np, nl, O_READ, 0]),
            ENOENT,
            "read-only open of a non-file, non-tracked-dir path is ENOENT (dir-open stays narrow)"
        );
        // And a create still makes a *file* even where a dir subtree exists alongside (write intent
        // never resolves to a directory fd).
        let (cp, cl) = put(&mut win, PA, "sub/new");
        let cfd = call(
            &mut host,
            h,
            &mut win,
            FS_OPEN,
            &[cp, cl, O_CREATE | O_WRITE, 0],
        );
        assert!(
            cfd >= 0,
            "create under a seeded subtree opens a writable file"
        );
        call(&mut host, h, &mut win, FS_CLOSE, &[cfd, 0, 0, 0]);
    }

    /// `opendir` snapshots entries and `readdir` yields them **sorted**, deterministically, across
    /// several files and a nested directory — and a too-small `readdir` buffer fails closed
    /// (`-EINVAL`) without consuming the entry.
    #[test]
    fn readdir_is_sorted_and_bounded() {
        fn names(host: &mut Host, h: i32, win: &mut [u8]) -> Vec<String> {
            // opendir "." (the root) and drain it.
            win[0] = b'.';
            let dh = call(host, h, win, FS_OPENDIR, &[0, 1, 0, 0]);
            assert!(dh >= 0, "opendir . : {dh}");
            let mut out = Vec::new();
            loop {
                let n = call(host, h, win, FS_READDIR, &[dh, 512, 64, 0]);
                if n == 0 {
                    break;
                }
                assert!(n > 0, "readdir: {n}");
                out.push(String::from_utf8(win[512..512 + n as usize].to_vec()).unwrap());
            }
            call(host, h, win, FS_CLOSEDIR, &[dh, 0, 0, 0]);
            out
        }

        // Build "beta", "alpha", "gamma/x" (gamma implicit) on mem_fs; expect sorted [alpha,beta,gamma].
        let mut host = Host::new();
        let cap = mem_fs();
        let h = (cap.grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        for f in ["beta", "alpha", "gamma/x"] {
            win[100..100 + f.len()].copy_from_slice(f.as_bytes());
            let fd = call(
                &mut host,
                h,
                &mut win,
                FS_OPEN,
                &[100, f.len() as i64, O_CREATE | O_WRITE, 0],
            );
            assert!(fd >= 0);
            call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]);
        }
        assert_eq!(
            names(&mut host, h, &mut win),
            vec!["alpha", "beta", "gamma"]
        );

        // A readdir into a 2-byte buffer can't hold "alpha" (5) → -EINVAL, entry not consumed.
        win[0] = b'.';
        let dh = call(&mut host, h, &mut win, FS_OPENDIR, &[0, 1, 0, 0]);
        assert_eq!(
            call(&mut host, h, &mut win, FS_READDIR, &[dh, 512, 2, 0]),
            EINVAL,
            "a too-small readdir buffer fails closed"
        );
        // The entry survived: a full-size read still returns it.
        assert_eq!(
            call(&mut host, h, &mut win, FS_READDIR, &[dh, 512, 64, 0]),
            5
        );
    }

    /// #2643 — the tree the link tests run in, under a fresh directory named for `test`: `root`, the
    /// directory a grant names, and beside it `outside`, which holds `secret` and an empty `d`.
    /// Returns `(base, root, outside)`.
    fn link_fixture(test: &str) -> (PathBuf, PathBuf, PathBuf) {
        let base = std::env::temp_dir().join(format!("temen_fs_{test}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (root, outside) = (base.join("root"), base.join("outside"));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::create_dir_all(outside.join("d")).unwrap();
        std::fs::write(outside.join("secret"), "SECRET").unwrap();
        (base, root, outside)
    }

    /// Link `link` to the directory `target` with a symlink.
    #[cfg(unix)]
    fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }

    /// Link `link` to the directory `target` with a junction, which any user may make (a symlink
    /// needs a privilege on Windows).
    #[cfg(windows)]
    fn link_dir(target: &Path, link: &Path) -> std::io::Result<()> {
        let out = std::process::Command::new("cmd")
            .args(["/C", "mklink", "/J"])
            .arg(link)
            .arg(target)
            .output()?;
        match out.status.success() {
            true => Ok(()),
            false => Err(std::io::Error::other(
                String::from_utf8_lossy(&out.stderr).into_owned(),
            )),
        }
    }

    /// Remove the link [`link_dir`] made, not what it points to.
    fn unlink_dir(link: &Path) -> std::io::Result<()> {
        if cfg!(windows) {
            std::fs::remove_dir(link)
        } else {
            std::fs::remove_file(link)
        }
    }

    /// What `dir` holds: each entry's name, and a file's contents.
    fn listing(dir: &Path) -> Vec<(String, String)> {
        let mut all: Vec<_> = std::fs::read_dir(dir)
            .unwrap()
            .map(|e| {
                let path = e.unwrap().path();
                let name = path.file_name().unwrap().to_string_lossy().into_owned();
                (name, std::fs::read_to_string(&path).unwrap_or_default())
            })
            .collect();
        all.sort();
        all
    }

    /// #2643 — a link in the granted tree is no way out of it. `out` links to a directory outside
    /// the root (and on unix `up` does too, relatively, and `leak` links to a file there): every
    /// path op through one is `-EACCES`, as `..` is, and nothing outside changes.
    #[test]
    fn a_link_out_of_the_root_reaches_nothing() {
        let (base, root, outside) = link_fixture("link_out");
        std::fs::write(root.join("mine"), "MINE").unwrap();
        link_dir(&outside, &root.join("out")).unwrap();
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("../outside", root.join("up")).unwrap();
            std::os::unix::fs::symlink(outside.join("secret"), root.join("leak")).unwrap();
        }
        let links: &[&str] = if cfg!(unix) { &["out", "up"] } else { &["out"] };
        let before = listing(&outside);

        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        let (sb, sl) = (2048, STATBUF_LEN as i64);
        for link in links {
            let through = |s: &str| format!("{link}/{s}");
            for (op, path, c, d) in [
                (FS_OPEN, through("secret"), O_READ, 0),
                (FS_OPEN, through("secret"), O_WRITE | O_TRUNC, 0),
                (FS_OPEN, through("new"), O_WRITE | O_CREATE, 0),
                (FS_STAT, through("secret"), sb, sl),
                (FS_MKDIR, through("new"), 0, 0),
                (FS_RMDIR, through("d"), 0, 0),
                (FS_REMOVE, through("secret"), 0, 0),
                (FS_OPENDIR, link.to_string(), 0, 0),
                (FS_OPENDIR, through("d"), 0, 0),
            ] {
                let (p, l) = put(&mut win, 0, &path);
                let got = call(&mut host, h, &mut win, op, &[p, l, c, d]);
                assert_eq!(got, EACCES, "op {op} on {path}");
            }
            for (from, to) in [
                (through("secret"), "taken".into()),
                ("mine".into(), through("mine")),
            ] {
                let (fp, fl) = put(&mut win, 0, &from);
                let (tp, tl) = put(&mut win, 1024, &to);
                let got = call(&mut host, h, &mut win, FS_RENAME, &[fp, fl, tp, tl]);
                assert_eq!(got, EACCES, "rename {from} to {to}");
            }
        }
        #[cfg(unix)]
        for flags in [O_READ, O_WRITE | O_TRUNC] {
            let (p, l) = put(&mut win, 0, "leak");
            let got = call(&mut host, h, &mut win, FS_OPEN, &[p, l, flags, 0]);
            assert_eq!(got, EACCES, "open leak with flags {flags}");
        }

        assert_eq!(
            listing(&outside),
            before,
            "nothing outside the root changed"
        );
        assert_eq!(std::fs::read_to_string(root.join("mine")).unwrap(), "MINE");
        assert!(!root.join("taken").exists());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2643 — a link that stays inside the root works as its target does, as under Wasmtime's
    /// preopens. (On Windows only a junction can be made without a privilege, and a junction's
    /// target is absolute, so it is refused there, as an absolute symlink is everywhere.)
    #[cfg(unix)]
    #[test]
    fn a_link_within_the_root_works_as_its_target() {
        let (base, root, _) = link_fixture("link_in");
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/f"), "IN").unwrap();
        std::os::unix::fs::symlink("sub", root.join("in")).unwrap();
        std::os::unix::fs::symlink("in/f", root.join("f")).unwrap();

        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        for path in ["in/f", "f"] {
            let (p, l) = put(&mut win, 0, path);
            let fd = call(&mut host, h, &mut win, FS_OPEN, &[p, l, O_READ, 0]);
            assert!(fd >= 0, "open {path}: {fd}");
            assert_eq!(call(&mut host, h, &mut win, FS_READ, &[fd, 1024, 8, 0]), 2);
            assert_eq!(&win[1024..1026], b"IN", "{path} reads sub/f");
            assert_eq!(call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]), 0);
        }
        let (p, l) = put(&mut win, 0, "in/m");
        assert_eq!(call(&mut host, h, &mut win, FS_MKDIR, &[p, l, 0, 0]), 0);
        assert!(root.join("sub/m").is_dir());
        let (p, l) = put(&mut win, 0, "in");
        let dh = call(&mut host, h, &mut win, FS_OPENDIR, &[p, l, 0, 0]);
        assert!(dh >= 0, "opendir in: {dh}");
        let mut names = Vec::new();
        loop {
            let n = call(&mut host, h, &mut win, FS_READDIR, &[dh, 1024, 64, 0]);
            if n <= 0 {
                assert_eq!(n, 0);
                break;
            }
            names.push(String::from_utf8(win[1024..1024 + n as usize].to_vec()).unwrap());
        }
        assert_eq!(names, ["f", "m"]);
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2643 — a host that swaps a directory of the tree for a link out, and back, while the guest
    /// walks through it never lets an op reach outside: an op resolves its path beneath the root
    /// once and acts on what it resolved, so no swap lands between a check and a use. The walk
    /// takes every path op through `d`, and `outside` holds what each would find there: `secret`,
    /// which only an escaped read returns, and `victim` and `d`, which `d` itself never holds.
    #[test]
    fn swapping_a_directory_for_a_link_out_never_lets_an_op_out() {
        use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
        let (base, root, outside) = link_fixture("link_swap");
        std::fs::create_dir(root.join("d")).unwrap();
        std::fs::write(root.join("d/secret"), "MINE").unwrap();
        std::fs::write(outside.join("victim"), "VICTIM").unwrap();
        let before = listing(&outside);

        // The swapper takes each step once a whole walk has run since its last, so the guest walks
        // through every state the tree passes through, the link out among them, while each step
        // lands in the middle of some walk. A step can fail while the guest holds what it moves
        // (Windows refuses to rename a directory a handle holds open), so it is retried. A step that
        // waits a minute fails the test rather than hanging it.
        fn step(
            walks: &AtomicU64,
            seen: &mut u64,
            what: &str,
            mut f: impl FnMut() -> std::io::Result<()>,
        ) {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
            while walks.load(Relaxed) < *seen + 2 {
                assert!(
                    std::time::Instant::now() < deadline,
                    "{what}: the guest stopped walking"
                );
                std::thread::yield_now();
            }
            loop {
                match f() {
                    Ok(()) => break,
                    Err(e) => assert!(std::time::Instant::now() < deadline, "{what}: {e}"),
                }
                std::thread::yield_now();
            }
            *seen = walks.load(Relaxed);
        }
        // A junction costs a process on Windows, so fewer swaps there.
        let swaps = if cfg!(windows) { 50 } else { 500 };
        let walks = std::sync::Arc::new(AtomicU64::new(0));
        let swapper = std::thread::spawn({
            let walks = walks.clone();
            let (d, aside, outside) = (root.join("d"), root.join("d.aside"), outside.clone());
            move || {
                let seen = &mut 0;
                for _ in 0..swaps {
                    step(&walks, seen, "move d aside", || std::fs::rename(&d, &aside));
                    step(&walks, seen, "link d out", || link_dir(&outside, &d));
                    step(&walks, seen, "unlink d", || unlink_dir(&d));
                    step(&walks, seen, "move d back", || std::fs::rename(&aside, &d));
                }
            }
        });

        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 4096];
        // An op on one path, with its other two arguments; and a rename.
        fn at(host: &mut Host, h: i32, win: &mut [u8], op: u32, path: &str, cd: [i64; 2]) -> i64 {
            let (p, l) = put(win, 0, path);
            call(host, h, win, op, &[p, l, cd[0], cd[1]])
        }
        fn rename(host: &mut Host, h: i32, win: &mut [u8], from: &str, to: &str) -> i64 {
            let (tp, tl) = put(win, 1024, to);
            let (p, l) = put(win, 0, from);
            call(host, h, win, FS_RENAME, &[p, l, tp, tl])
        }
        let (host, win) = (&mut host, &mut win[..]);
        let stat = [2048, STATBUF_LEN as i64];
        while !swapper.is_finished() {
            let fd = at(host, h, win, FS_OPEN, "d/secret", [O_READ, 0]);
            if fd >= 0 {
                assert_eq!(call(host, h, win, FS_READ, &[fd, 1024, 8, 0]), 4);
                assert_eq!(&win[1024..1028], b"MINE", "d/secret is the root's");
                call(host, h, win, FS_CLOSE, &[fd, 0, 0, 0]);
            }
            let fd = at(host, h, win, FS_OPEN, "d/new", [O_WRITE | O_CREATE, 0]);
            if fd >= 0 {
                call(host, h, win, FS_CLOSE, &[fd, 0, 0, 0]);
                let left = match rename(host, h, win, "d/new", "d/new2") {
                    0 => "d/new2",
                    _ => "d/new",
                };
                at(host, h, win, FS_REMOVE, left, [0, 0]);
            }
            if at(host, h, win, FS_MKDIR, "d/m", [0, 0]) == 0 {
                at(host, h, win, FS_RMDIR, "d/m", [0, 0]);
            }
            // `d` never holds `victim` or `d`: only an op that got out finds them.
            assert_ne!(at(host, h, win, FS_STAT, "d/victim", stat), 0, "stat");
            assert_ne!(at(host, h, win, FS_REMOVE, "d/victim", [0, 0]), 0, "remove");
            assert_ne!(at(host, h, win, FS_RMDIR, "d/d", [0, 0]), 0, "rmdir");
            assert_ne!(rename(host, h, win, "d/victim", "d/moved"), 0, "rename");
            let dh = at(host, h, win, FS_OPENDIR, "d", [0, 0]);
            if dh >= 0 {
                loop {
                    let n = call(host, h, win, FS_READDIR, &[dh, 1024, 64, 0]);
                    if n <= 0 {
                        break;
                    }
                    let name = &win[1024..1024 + n as usize];
                    assert!(name != b"victim" && name != b"d", "d lists outside");
                }
                call(host, h, win, FS_CLOSEDIR, &[dh, 0, 0, 0]);
            }
            walks.fetch_add(1, Relaxed);
        }
        swapper.join().unwrap();

        assert_eq!(
            listing(&outside),
            before,
            "nothing outside the root changed"
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    // #2510 — `replace`. A test puts its path at `PATH` in the window, and the bytes at `BYTES`.
    const PATH: usize = 0;
    const BYTES: usize = 4096;

    /// `replace` the file `path` with `bytes`, as a guest does.
    fn replace_with(host: &mut Host, h: i32, win: &mut [u8], path: &str, bytes: &[u8]) -> i64 {
        let (p, l) = put(win, PATH, path);
        let (b, n) = put(win, BYTES, bytes);
        call(host, h, win, FS_REPLACE, &[p, l, b, n])
    }

    /// The bytes of the file `path`, read as a guest reads them, or the `-errno` of the open.
    fn read_back(host: &mut Host, h: i32, win: &mut [u8], path: &str) -> Result<Vec<u8>, i64> {
        let (p, l) = put(win, PATH, path);
        let fd = call(host, h, win, FS_OPEN, &[p, l, O_READ, 0]);
        if fd < 0 {
            return Err(fd);
        }
        let bytes = drain(host, h, win, fd);
        assert_eq!(call(host, h, win, FS_CLOSE, &[fd, 0, 0, 0]), 0);
        Ok(bytes)
    }

    /// What is left to read from the open file `fd`.
    fn drain(host: &mut Host, h: i32, win: &mut [u8], fd: i64) -> Vec<u8> {
        let mut bytes = Vec::new();
        loop {
            let n = call(host, h, win, FS_READ, &[fd, BYTES as i64, 4096, 0]);
            assert!(n >= 0, "read: {n}");
            if n == 0 {
                return bytes;
            }
            bytes.extend_from_slice(&win[BYTES..BYTES + n as usize]);
        }
    }

    /// The names of the `replace` temporary files in `dir`.
    fn temps(dir: &Path) -> Vec<String> {
        listing(dir)
            .into_iter()
            .map(|(name, _)| name)
            .filter(|name| name.starts_with(REPLACE_TEMP))
            .collect()
    }

    /// #2510 — `replace` behaves alike on both backends. It makes a missing file and replaces an
    /// existing one with exactly the bytes given; a descriptor open on the old bytes keeps them;
    /// and it refuses a directory, a path out of the root and a bad buffer. `host_fs` leaves no
    /// temporary file behind.
    #[test]
    fn replace_parity_mem_vs_host() {
        fn script(host: &mut Host, h: i32, win: &mut [u8]) -> Vec<String> {
            let text = |read: Result<Vec<u8>, i64>| match read {
                Ok(bytes) => String::from_utf8(bytes).unwrap(),
                Err(e) => format!("errno {e}"),
            };
            let mut seen = vec![];
            seen.push(format!("make: {}", replace_with(host, h, win, "f", b"one")));
            seen.push(format!("made: {}", text(read_back(host, h, win, "f"))));
            let (p, l) = put(win, PATH, "f");
            let old = call(host, h, win, FS_OPEN, &[p, l, O_READ, 0]);
            seen.push(format!(
                "replace: {}",
                replace_with(host, h, win, "f", b"two!")
            ));
            seen.push(format!("replaced: {}", text(read_back(host, h, win, "f"))));
            seen.push(format!(
                "old descriptor: {}",
                text(Ok(drain(host, h, win, old)))
            ));
            call(host, h, win, FS_CLOSE, &[old, 0, 0, 0]);
            let (p, l) = put(win, PATH, "d");
            seen.push(format!(
                "mkdir: {}",
                call(host, h, win, FS_MKDIR, &[p, l, 0, 0])
            ));
            seen.push(format!("in it: {}", replace_with(host, h, win, "d/g", b"")));
            seen.push(format!(
                "made empty: {}",
                text(read_back(host, h, win, "d/g"))
            ));
            seen.push(format!(
                "a directory: {}",
                replace_with(host, h, win, "d", b"x")
            ));
            seen.push(format!(
                "the root: {}",
                replace_with(host, h, win, ".", b"x")
            ));
            seen.push(format!("out: {}", replace_with(host, h, win, "../x", b"x")));
            let (p, l) = put(win, PATH, "f");
            let negative = call(host, h, win, FS_REPLACE, &[p, l, -1, 1]);
            seen.push(format!("negative: {negative}"));
            let past = call(host, h, win, FS_REPLACE, &[p, l, BYTES as i64, 1 << 40]);
            seen.push(format!("past the window: {past}"));
            seen.push(format!("kept: {}", text(read_back(host, h, win, "f"))));
            seen
        }
        let expected = [
            "make: 0".to_string(),
            "made: one".into(),
            "replace: 0".into(),
            "replaced: two!".into(),
            "old descriptor: one".into(),
            "mkdir: 0".into(),
            "in it: 0".into(),
            "made empty: ".into(),
            format!("a directory: {EISDIR}"),
            format!("the root: {EISDIR}"),
            format!("out: {EACCES}"),
            format!("negative: {EINVAL}"),
            format!("past the window: {EFAULT}"),
            "kept: two!".into(),
        ];

        let mut win = vec![0u8; 1 << 16];
        let mut host = Host::new();
        let h = (mem_fs().grant)(&mut host, 0);
        assert_eq!(script(&mut host, h, &mut win), expected, "mem_fs");

        let (base, root, _) = link_fixture("replace_parity");
        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        assert_eq!(script(&mut host, h, &mut win), expected, "host_fs");
        assert_eq!(temps(&root), Vec::<String>::new());
        assert_eq!(temps(&root.join("d")), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// What the host protects `path` with, as a test compares it: on Unix its mode, owner, group
    /// and an extended attribute; on Windows its DACL, as `icacls` lists it, whether it is hidden,
    /// and a named stream.
    #[cfg(unix)]
    fn protection(path: &Path) -> (u32, u32, u32, Option<Vec<u8>>) {
        use std::os::unix::fs::MetadataExt;
        let md = std::fs::metadata(path).unwrap();
        (md.mode() & 0o7777, md.uid(), md.gid(), get_xattr(path))
    }
    #[cfg(windows)]
    fn protection(path: &Path) -> (String, u32, Option<Vec<u8>>) {
        use std::os::windows::fs::MetadataExt;
        let out = std::process::Command::new("icacls")
            .arg(path)
            .output()
            .unwrap();
        // The listing names the file on its first line, which this leaves out.
        let acl = String::from_utf8_lossy(&out.stdout).replace(&*path.to_string_lossy(), "");
        let attributes = std::fs::metadata(path).unwrap().file_attributes();
        let stream = std::fs::read(format!("{}:temen.kept", path.display())).ok();
        (acl, attributes & 0x2, stream) // FILE_ATTRIBUTE_HIDDEN
    }

    /// The extended attribute these tests set: `None` where the filesystem has none.
    #[cfg(unix)]
    const XATTR: &str = if cfg!(target_vendor = "apple") {
        "com.temen.kept"
    } else {
        "user.temen.kept"
    };

    /// Set [`XATTR`] on `path`: `false` where the filesystem has no such attributes.
    #[cfg(unix)]
    fn set_xattr(path: &Path, value: &[u8]) -> bool {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let name = std::ffi::CString::new(XATTR).unwrap();
        let (v, n) = (value.as_ptr().cast(), value.len());
        // SAFETY: both strings are NUL-terminated and `value` is `n` bytes, for the call.
        #[cfg(target_vendor = "apple")]
        let set = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), v, n, 0, 0) };
        // SAFETY: as above.
        #[cfg(not(target_vendor = "apple"))]
        let set = unsafe { libc::setxattr(path.as_ptr(), name.as_ptr(), v, n, 0) };
        if set != 0 {
            let e = std::io::Error::last_os_error();
            assert_eq!(e.raw_os_error(), Some(libc::ENOTSUP), "setxattr: {e}");
        }
        set == 0
    }

    /// [`XATTR`]'s value on `path`, if it has one.
    #[cfg(unix)]
    fn get_xattr(path: &Path) -> Option<Vec<u8>> {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(path.as_os_str().as_bytes()).unwrap();
        let name = std::ffi::CString::new(XATTR).unwrap();
        let mut buf = [0u8; 64];
        let (v, n) = (buf.as_mut_ptr().cast(), buf.len());
        // SAFETY: both strings are NUL-terminated and `buf` is `n` bytes, for the call to fill.
        #[cfg(target_vendor = "apple")]
        let got = unsafe { libc::getxattr(path.as_ptr(), name.as_ptr(), v, n, 0, 0) };
        // SAFETY: as above.
        #[cfg(not(target_vendor = "apple"))]
        let got = unsafe { libc::getxattr(path.as_ptr(), name.as_ptr(), v, n) };
        usize::try_from(got).ok().map(|len| buf[..len].to_vec())
    }

    /// #2510 — `replace` keeps what the host protects the file with. On Unix that is its mode
    /// (set-user-ID bit and all), its owner and group, and its extended attributes, where the
    /// filesystem has them. On Windows it is its DACL, here with an entry of its own, its
    /// attributes, here hidden, and its named streams.
    #[test]
    fn replace_keeps_the_hosts_protection() {
        let (base, root, _) = link_fixture("replace_protection");
        let f = root.join("f");
        std::fs::write(&f, "old").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o4750)).unwrap();
            // As root, a file someone else owns: the replace gives the new file to them.
            // SAFETY: geteuid reads the process's effective user id.
            if unsafe { libc::geteuid() } == 0 {
                std::os::unix::fs::chown(&f, Some(65534), Some(65534)).unwrap();
            }
            set_xattr(&f, b"kept");
        }
        #[cfg(windows)]
        {
            // An entry of its own, read and execute for everyone, beside those it inherits.
            let run = |cmd: &str, args: &[&str]| {
                let out = std::process::Command::new(cmd)
                    .args(args)
                    .arg(&f)
                    .output()
                    .unwrap();
                assert!(out.status.success(), "{cmd}: {out:?}");
            };
            run("icacls", &["/grant", "*S-1-1-0:(RX)"]);
            run("attrib", &["+h"]);
            std::fs::write(format!("{}:temen.kept", f.display()), "kept").unwrap();
        }
        let before = protection(&f);

        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 1 << 16];
        assert_eq!(replace_with(&mut host, h, &mut win, "f", b"new"), 0);
        assert_eq!(std::fs::read(&f).unwrap(), b"new");
        assert_eq!(protection(&f), before);
        assert_eq!(temps(&root), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2510 — a file `replace` makes is made as `open` with `O_CREATE` makes one, with the host's
    /// defaults.
    #[test]
    fn replace_makes_a_missing_file_as_open_does() {
        let (base, root, _) = link_fixture("replace_make");
        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 1 << 16];
        let (p, l) = put(&mut win, PATH, "opened");
        let fd = call(
            &mut host,
            h,
            &mut win,
            FS_OPEN,
            &[p, l, O_CREATE | O_WRITE, 0],
        );
        assert_eq!(call(&mut host, h, &mut win, FS_CLOSE, &[fd, 0, 0, 0]), 0);
        assert_eq!(replace_with(&mut host, h, &mut win, "made", b"new"), 0);
        assert_eq!(std::fs::read(root.join("made")).unwrap(), b"new");
        assert_eq!(
            protection(&root.join("made")),
            protection(&root.join("opened"))
        );
        let _ = std::fs::remove_dir_all(&base);
    }

    /// Run `f` as a process the host's protection binds: as root on Linux, with the file-system
    /// ids of `nobody`, which drops the privilege to ignore a file's mode.
    fn unprivileged<T>(f: impl FnOnce() -> T) -> T {
        #[cfg(target_os = "linux")]
        // SAFETY: geteuid reads the process's effective user id.
        if unsafe { libc::geteuid() } == 0 {
            // SAFETY: setfsuid sets this thread's file-system user id, which is set back below.
            unsafe { libc::setfsuid(65534) };
            let out = f();
            // SAFETY: as above.
            unsafe { libc::setfsuid(0) };
            return out;
        }
        f()
    }

    /// Make the temporary file `replace` would make for `name` in `dir`, and hold it as a replace
    /// in progress does: locked on Unix, open without sharing deletes on Windows.
    fn hold_temp(dir: &Path, name: &str) -> std::fs::File {
        let path = dir.join(temp_name(name));
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let file = std::fs::File::create(&path).unwrap();
            // SAFETY: the descriptor is open for the call.
            let locked = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            assert_eq!(locked, 0);
            file
        }
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .share_mode(0x1 | 0x2) // FILE_SHARE_READ | FILE_SHARE_WRITE
                .open(&path)
                .unwrap()
        }
    }

    /// #2510 — `replace` refuses what it may not replace, and changes nothing: a file the host
    /// marks read-only for the process (`-EACCES`), a file with another hard link (R1,
    /// `-EMLINK`), and a file another replace is replacing (`-EBUSY`). Once that one is done, or
    /// a crash ended it, its temporary file is a leftover, which the next replace removes.
    #[test]
    fn replace_refuses_what_it_may_not_replace() {
        let (base, root, _) = link_fixture("replace_refusals");
        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 1 << 16];

        std::fs::write(root.join("a"), "linked").unwrap();
        std::fs::hard_link(root.join("a"), root.join("b")).unwrap();
        assert_eq!(replace_with(&mut host, h, &mut win, "a", b"new"), EMLINK);
        assert_eq!(std::fs::read(root.join("b")).unwrap(), b"linked");

        let ro = root.join("ro");
        std::fs::write(&ro, "ro").unwrap();
        // The directory is anyone's to write in, so it is the file's own mode that refuses.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&root, std::fs::Permissions::from_mode(0o777)).unwrap();
        }
        let mut read_only = std::fs::metadata(&ro).unwrap().permissions();
        read_only.set_readonly(true);
        std::fs::set_permissions(&ro, read_only.clone()).unwrap();
        let refused = unprivileged(|| replace_with(&mut host, h, &mut win, "ro", b"new"));
        assert_eq!(refused, EACCES);
        assert_eq!(std::fs::read(&ro).unwrap(), b"ro");
        #[allow(clippy::permissions_set_readonly_false)]
        read_only.set_readonly(false); // so it can be removed on Windows
        std::fs::set_permissions(&ro, read_only).unwrap();

        std::fs::write(root.join("f"), "old").unwrap();
        let held = hold_temp(&root, "f");
        assert_eq!(replace_with(&mut host, h, &mut win, "f", b"new"), EBUSY);
        assert_eq!(std::fs::read(root.join("f")).unwrap(), b"old");
        drop(held);
        assert_eq!(temps(&root).len(), 1, "the other replace's temporary file");
        assert_eq!(replace_with(&mut host, h, &mut win, "f", b"new"), 0);
        assert_eq!(std::fs::read(root.join("f")).unwrap(), b"new");
        assert_eq!(temps(&root), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2510 — a crash at each of `replace`'s three barriers leaves the old bytes or the new, on
    /// both backends alike: the old at the temporary file's sync and at the rename, the new at
    /// the directory's sync. What a crash leaves of `host_fs`'s temporary file, the next replace
    /// of the file removes.
    #[test]
    fn a_crash_at_each_step_of_replace_leaves_the_old_bytes_or_the_new() {
        let (base, root, _) = link_fixture("replace_crash");
        let f = root.join("f");
        let mut win = vec![0u8; 1 << 16];
        for (n, want) in ["old", "old", "new", "new"].into_iter().enumerate() {
            let mut host = Host::new();
            let h = (mem_fs_crashy().grant)(&mut host, 0);
            assert_eq!(replace_with(&mut host, h, &mut win, "f", b"old"), 0);
            assert_eq!(
                call(&mut host, h, &mut win, FS_CRASH_ARM, &[n as i64, 0, 0, 0]),
                0
            );
            assert_eq!(replace_with(&mut host, h, &mut win, "f", b"new"), 0);
            let read = read_back(&mut host, h, &mut win, "f");
            assert_eq!(read, Ok(want.into()), "mem_fs, a crash at barrier {n}");

            std::fs::write(&f, "old").unwrap();
            let mut host = Host::new();
            let h = (host_fs_crashy(root.clone()).grant)(&mut host, 0);
            assert_eq!(
                call(&mut host, h, &mut win, FS_CRASH_ARM, &[n as i64, 0, 0, 0]),
                0
            );
            assert_eq!(replace_with(&mut host, h, &mut win, "f", b"new"), 0);
            let read = std::fs::read_to_string(&f).unwrap();
            assert_eq!(read, want, "host_fs, a crash at barrier {n}");
            if n == 1 {
                assert_eq!(
                    temps(&root).len(),
                    1,
                    "a crash at the rename leaves the file"
                );
            }
            // A grant after the crash, as after a restart.
            let mut host = Host::new();
            let h = (host_fs(root.clone()).grant)(&mut host, 0);
            assert_eq!(replace_with(&mut host, h, &mut win, "f", b"after"), 0);
            assert_eq!(std::fs::read_to_string(&f).unwrap(), "after");
            assert_eq!(temps(&root), Vec::<String>::new());
        }
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2510 — a reader racing `replace` sees the old bytes or the new, never part of either.
    #[test]
    fn a_reader_racing_replace_sees_the_old_bytes_or_the_new() {
        use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
        let (base, root, _) = link_fixture("replace_race");
        let (a, b) = (vec![b'a'; 24 << 10], vec![b'b'; 40 << 10]);
        let f = root.join("f");
        std::fs::write(&f, &a).unwrap();
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let (f, a, b, stop) = (f.clone(), a.clone(), b.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Relaxed) {
                    let got = std::fs::read(&f).unwrap();
                    assert!(got == a || got == b, "a read of {} bytes", got.len());
                    reads += 1;
                }
                reads
            })
        };
        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 1 << 17];
        for i in 0..200 {
            let next = if i % 2 == 0 { &b } else { &a };
            assert_eq!(replace_with(&mut host, h, &mut win, "f", next), 0);
        }
        stop.store(true, Relaxed);
        assert!(reader.join().unwrap() > 0, "the reader read");
        let _ = std::fs::remove_dir_all(&base);
    }

    /// #2510 R2 — `replace` through a symlink beneath the root replaces the file it leads to and
    /// keeps the link, and a dangling one leads to the file the replace makes. A link out of the
    /// root is `-EACCES`, as it is for every op (#2643), a link to a directory `-EISDIR`, and a
    /// loop `-ELOOP`.
    #[cfg(unix)]
    #[test]
    fn replace_through_a_link_replaces_what_it_leads_to() {
        use std::os::unix::fs::symlink;
        let (base, root, outside) = link_fixture("replace_links");
        std::fs::create_dir(root.join("sub")).unwrap();
        std::fs::write(root.join("sub/f"), "old").unwrap();
        for (link, to) in [
            ("l", "sub/f"),
            ("chain", "l"),
            ("sub/up", "../l"),
            ("dangling", "sub/new"),
            ("out", "../outside/secret"),
            ("dir", "sub"),
            ("x", "y"),
            ("y", "x"),
        ] {
            symlink(to, root.join(link)).unwrap();
        }
        symlink(outside.join("secret"), root.join("abs")).unwrap();
        let mut host = Host::new();
        let h = (host_fs(root.clone()).grant)(&mut host, 0);
        let mut win = vec![0u8; 1 << 16];
        for via in ["l", "chain", "sub/up"] {
            assert_eq!(replace_with(&mut host, h, &mut win, via, via.as_bytes()), 0);
            assert_eq!(std::fs::read(root.join("sub/f")).unwrap(), via.as_bytes());
        }
        assert_eq!(replace_with(&mut host, h, &mut win, "dangling", b"made"), 0);
        assert_eq!(std::fs::read(root.join("sub/new")).unwrap(), b"made");
        for link in ["l", "chain", "sub/up", "dangling"] {
            let md = std::fs::symlink_metadata(root.join(link)).unwrap();
            assert!(md.file_type().is_symlink(), "{link} is still a link");
        }
        assert_eq!(replace_with(&mut host, h, &mut win, "out", b"x"), EACCES);
        assert_eq!(replace_with(&mut host, h, &mut win, "abs", b"x"), EACCES);
        assert_eq!(replace_with(&mut host, h, &mut win, "dir", b"x"), EISDIR);
        assert_eq!(replace_with(&mut host, h, &mut win, "x", b"x"), ELOOP);
        assert_eq!(
            listing(&outside),
            [
                ("d".into(), String::new()),
                ("secret".into(), "SECRET".into())
            ]
        );
        assert_eq!(temps(&root), Vec::<String>::new());
        assert_eq!(temps(&root.join("sub")), Vec::<String>::new());
        let _ = std::fs::remove_dir_all(&base);
    }
}
