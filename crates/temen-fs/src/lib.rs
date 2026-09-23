//! **In-memory `fs` capability backend + the shared fs-cap wire protocol.**
//!
//! This crate holds the wasm-safe half of the filesystem capability (`crates/temen-run/src/fs.rs`
//! §7): the op-code/`stat`-layout/path-vetting protocol both backends speak, and the deterministic
//! **in-memory** backend (`mem_fs`) plus its shippable **data-image** format. It depends only on
//! `temen-interp` (`HostProc`/`GuestMem`), so it builds for **wasm** — the browser cdylib mounts a
//! data-image `mem_fs` with no real filesystem. `temen-run` keeps the real-filesystem `host_fs`
//! backend (which pulls in the unix-only JIT/mmap machinery) and wraps these handlers in its
//! `HostCap`; it re-exports this crate's protocol + `mem_fs*` so `temen_run::fs::*` is unchanged.
//!
//! A store is a [`MemFsHandle`]: it mints the `HostProc` handlers granted over it and declares its
//! own state ([`MemFsHandle::declare_state`], #1491), so every checkpoint, debugger rebuild and §12
//! freeze carries the files a guest wrote. `temen-run` wraps it in its `HostCap`; the browser cdylib
//! and the debug on-ramp grant it through [`grant_vm_fs`].

use std::collections::HashMap;
use std::path::{Component, Path};
use std::sync::{Arc, Mutex};
use temen_interp::{GuestMem, HostProc};
use temen_ir::errno::*;

pub const FS_OPEN: u32 = 0;
pub const FS_READ: u32 = 1;
pub const FS_WRITE: u32 = 2;
pub const FS_SEEK: u32 = 3;
pub const FS_CLOSE: u32 = 4;
pub const FS_REMOVE: u32 = 5;
pub const FS_RENAME: u32 = 6;
pub const FS_TRUNCATE: u32 = 7;
pub const FS_SYNC: u32 = 8;
pub const FS_MMAP: u32 = 9;
pub const FS_MSYNC: u32 = 10;
pub const FS_MUNMAP: u32 = 11;
/// `map_region(fd, file_offset, len)` — the **zero-copy** file-mmap path (§4b of `MMAP_CAPABILITY.md`).
/// Mint a file-backed `SharedRegion` over `[file_offset, file_offset+len)` of the open file `fd` and
/// return its **handle** (a `SharedRegion` cap the guest maps into its window with `SharedRegion.map`),
/// so guest loads/stores hit the real file's pages with no copy-in and no per-access host call. Present
/// ONLY on the `host_fs_mmap` variant (which is granted with region-minting authority); returns
/// `-EINVAL` on `mem_fs`/`host_fs` (no minter, or no real fd). v1 requires `file_offset == 0`.
pub const FS_MAP_REGION: u32 = 13;
/// `crash_arm(n)` — **test-only** crash injection (§4d of `MMAP_CAPABILITY.md`). Present ONLY on the
/// `*_crashy` backend variants; the default [`mem_fs`]/[`host_fs`] leave the controller absent so this
/// op is an unknown op (`-EINVAL`) on a shipping grant. Arms a simulated power loss: after `n` further
/// durability barriers (`msync`/`sync`) have completed, the *next* barrier "crashes" — from then on
/// every write to the backing store is silently dropped (the un-synced page cache is lost, as on real
/// power loss) while **reads keep working** (a dead process's file is still readable on reopen). `n < 0`
/// disarms. Lets a test sweep the crash point across every sync boundary and prove the mapped store
/// recovers to its last *committed* state at each one.
pub const FS_CRASH_ARM: u32 = 12;

/// `stat(path_ptr, path_len, statbuf_ptr, statbuf_cap)` — fill a fixed [`StatBuf`] (72 bytes,
/// little-endian) for `path` with **lstat** semantics (symlinks are not followed). `statbuf_cap`
/// must be ≥ [`STATBUF_LEN`]. Returns `0` / `-errno`.
pub const FS_STAT: u32 = 14;
/// `mkdir(path_ptr, path_len)` — create a directory. `-EEXIST` if it already exists. (The `mode`
/// argument a guest `mkdir(2)` would pass is ignored: the granted root's umask governs.)
pub const FS_MKDIR: u32 = 15;
/// `rmdir(path_ptr, path_len)` — remove an empty directory (`-ENOTEMPTY` otherwise).
pub const FS_RMDIR: u32 = 16;
/// `opendir(path_ptr, path_len)` — open a directory for iteration; returns a **dir handle** (a small
/// non-negative integer, a separate namespace from file `fd`s) or `-errno`. The directory's immediate
/// entries are snapshotted at this call, so a concurrent create/remove does not perturb the walk
/// (matches a typical libc `readdir` buffering the getdents stream).
pub const FS_OPENDIR: u32 = 17;
/// `readdir(dh, name_ptr, name_cap)` — write the next entry's name (no trailing NUL) into the guest
/// buffer and return its byte length; `0` when the directory is exhausted; `-errno` on a bad handle
/// or `-EINVAL` if `name_cap` is too small for the next name. `.` and `..` are **not** yielded (the
/// guest libc synthesizes them if it wants them), matching what Postgres's `ReadDir` filters anyway.
pub const FS_READDIR: u32 = 18;
/// `closedir(dh)` — drop a dir handle opened by [`FS_OPENDIR`].
pub const FS_CLOSEDIR: u32 = 19;

/// Byte length of the fixed [`StatBuf`] the [`FS_STAT`] op writes.
pub const STATBUF_LEN: usize = 72;
/// `S_IFMT` mask and the two `S_IF*` type values the guest libc needs to tell files from directories
/// (Linux ABI values, so the guest shim can copy `mode` straight into a `struct stat`).
pub const S_IFMT: u32 = 0o170000;
pub const S_IFREG: u32 = 0o100000;
pub const S_IFDIR: u32 = 0o040000;
pub const S_IFLNK: u32 = 0o120000;

pub const O_READ: i64 = 1;
pub const O_WRITE: i64 = 2;
pub const O_APPEND: i64 = 4;
pub const O_TRUNC: i64 = 8;
pub const O_CREATE: i64 = 16;

/// Build the fixed 72-byte little-endian [`StatBuf`] payload from the fields the guest libc reads.
/// Layout (offset: field): `0:mode(u32) 4:nlink(u32) 8:size(i64) 16:mtime_sec(i64) 24:mtime_nsec(i64)
/// 32:ino(u64) 40:dev(u64) 48:uid(u32) 52:gid(u32) 56:blksize(i64) 64:blocks(i64)`.
#[allow(clippy::too_many_arguments)]
pub fn stat_bytes(
    mode: u32,
    nlink: u32,
    size: i64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ino: u64,
    dev: u64,
    uid: u32,
    gid: u32,
    blksize: i64,
    blocks: i64,
) -> [u8; STATBUF_LEN] {
    let mut b = [0u8; STATBUF_LEN];
    b[0..4].copy_from_slice(&mode.to_le_bytes());
    b[4..8].copy_from_slice(&nlink.to_le_bytes());
    b[8..16].copy_from_slice(&size.to_le_bytes());
    b[16..24].copy_from_slice(&mtime_sec.to_le_bytes());
    b[24..32].copy_from_slice(&mtime_nsec.to_le_bytes());
    b[32..40].copy_from_slice(&ino.to_le_bytes());
    b[40..48].copy_from_slice(&dev.to_le_bytes());
    b[48..52].copy_from_slice(&uid.to_le_bytes());
    b[52..56].copy_from_slice(&gid.to_le_bytes());
    b[56..64].copy_from_slice(&blksize.to_le_bytes());
    b[64..72].copy_from_slice(&blocks.to_le_bytes());
    b
}

/// Read a guest path (window `ptr`/`len`) as UTF-8. `-EFAULT` on an out-of-window range, `-EINVAL`
/// on non-UTF-8 or an unreasonable length, `-EACCES` on a path that could name anything outside the
/// granted root (absolute, `..`, or empty) — enforced by **both** backends so the protocol semantics
/// are backend-independent (a differential runs identically on `mem_fs` and `host_fs`).
pub fn read_path(mem: Option<&dyn GuestMem>, ptr: i64, len: i64) -> Result<String, i64> {
    let mem = mem.ok_or(EFAULT)?;
    if !(0..=4096).contains(&len) || ptr < 0 {
        return Err(EINVAL);
    }
    let bytes = mem.read_bytes(ptr as u64, len as u64).ok_or(EFAULT)?;
    let path = String::from_utf8(bytes).map_err(|_| EINVAL)?;
    let p = Path::new(&path);
    if path.is_empty()
        || p.is_absolute()
        || p.components()
            .any(|c| !matches!(c, Component::Normal(_) | Component::CurDir))
    {
        return Err(EACCES);
    }
    Ok(path)
}

/// Allocate a file descriptor, **reserving 0/1/2** (the POSIX stdin/stdout/stderr slots) so files
/// always start at 3. A guest libc that routes fds 0/1/2 to the powerbox `Stream` cap (stdout/stderr/
/// stdin) and everything else to this fs cap then never confuses a file fd with a stream fd — the two
/// namespaces are disjoint. The reserved slots stay permanently vacant (an op on fd 0/1/2 is `-EBADF`).
pub fn alloc_fd<T>(open: &mut Vec<Option<T>>) -> usize {
    const RESERVED: usize = 3;
    while open.len() < RESERVED {
        open.push(None);
    }
    match open.iter().skip(RESERVED).position(Option::is_none) {
        Some(off) => RESERVED + off,
        None => {
            open.push(None);
            open.len() - 1
        }
    }
}

/// One open file: a shared byte buffer (kept alive independently of the name table, so a `remove`
/// of an open file behaves POSIX-like — the data survives until the last close) + cursor + mode.
struct MemOpen {
    data: Arc<Mutex<Vec<u8>>>,
    pos: usize,
    readable: bool,
    writable: bool,
    append: bool,
}

/// Test-only crash-injection controller (the §4d "crash hook"), shared by both backends. Models a
/// power loss: [`FS_CRASH_ARM`] sets `countdown` to the number of durability barriers
/// (`msync`/`sync`) that may still complete; each barrier decrements it, and the one that finds it at
/// zero *trips* — sets `crashed`, and is itself dropped (the crash happened before it reached the
/// platter). Once `crashed`, every persisting op (`msync`/`sync`/`munmap` flush/`write`/`truncate`)
/// silently drops its effect, so the backing file is frozen at the last completed barrier; reads are
/// untouched. Present only on the `*_crashy` variants — a shipping grant has no controller at all.
#[derive(Default)]
pub struct CrashCtl {
    /// Barriers that may still complete before the crash trips; `None` = disarmed (never crash).
    pub countdown: Option<u64>,
    /// Once set, all persistence is frozen.
    pub crashed: bool,
}

impl CrashCtl {
    /// Call at each durability barrier (`msync`/`sync`). Returns `true` if this barrier's write must be
    /// **dropped** — either we have already crashed, or this very barrier trips the crash.
    pub fn barrier(&mut self) -> bool {
        if self.crashed {
            return true;
        }
        match self.countdown {
            Some(0) => {
                self.crashed = true;
                true // the crash struck mid-barrier: its bytes never reach the file
            }
            Some(n) => {
                self.countdown = Some(n - 1);
                false
            }
            None => false,
        }
    }
}

/// One live `mmap`: a guest window buffer `[base, base+len)` bound to `data` at `file_off`. The
/// guest reads/writes the window bytes directly; `msync` copies a sub-range back into `data`.
struct MemMapping {
    base: u64,
    len: u64,
    data: Arc<Mutex<Vec<u8>>>,
    file_off: u64,
}

#[derive(Default)]
struct MemFsState {
    files: HashMap<String, Arc<Mutex<Vec<u8>>>>,
    /// Explicitly-`mkdir`'d directories (normalized keys). A path is *also* treated as a directory
    /// when it is a strict prefix of an existing file key, so `initdb`-style trees created purely by
    /// writing files still walk correctly; `dirs` records the empty ones a walk would otherwise miss.
    dirs: std::collections::BTreeSet<String>,
    open: Vec<Option<MemOpen>>,
    /// Snapshots taken by [`FS_OPENDIR`]: `opendirs[dh]` is the remaining child names to yield.
    opendirs: Vec<Option<Vec<String>>>,
    maps: Vec<MemMapping>,
    /// `Some` only on the `mem_fs_crashy` variant (test-only crash injection); `None` on `mem_fs`.
    crash: Option<CrashCtl>,
}

/// Normalize a vetted relative path to a canonical key: drop `.`/empty segments, join with `/`.
/// The root (`.` or the effect of stripping everything) maps to `""`.
fn norm(p: &str) -> String {
    p.split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect::<Vec<_>>()
        .join("/")
}

impl MemFsState {
    /// Is `key` (already normalized) a directory in this flat store?
    fn is_dir(&self, key: &str) -> bool {
        key.is_empty() || self.dirs.contains(key) || {
            let prefix = format!("{key}/");
            self.files.keys().any(|k| k.starts_with(&prefix))
                || self.dirs.iter().any(|d| d.starts_with(&prefix))
        }
    }
    /// Immediate child names of directory `key` (normalized), deduplicated, sorted for determinism.
    fn children_of(&self, key: &str) -> Vec<String> {
        let prefix = if key.is_empty() {
            String::new()
        } else {
            format!("{key}/")
        };
        let mut set = std::collections::BTreeSet::new();
        let child = |full: &str| -> Option<String> {
            let rest = full.strip_prefix(&prefix)?;
            if rest.is_empty() {
                return None;
            }
            Some(rest.split('/').next().unwrap().to_string())
        };
        for k in self.files.keys() {
            if let Some(c) = child(k) {
                set.insert(c);
            }
        }
        for d in &self.dirs {
            if let Some(c) = child(d) {
                set.insert(c);
            }
        }
        set.into_iter().collect()
    }
}

impl MemFsState {
    /// Current contents as a `(files, dirs)` seed — the exact shape [`encode_image`] serializes and
    /// [`MemFsHandle::seeded`]/[`mem_fs_seeded_shared`] mount. A file's bytes are its live committed
    /// buffer (an open `fd` shares the same `Arc`, so bytes already `write`n are included); purely
    /// transient state that is not part of a filesystem *image* — open-fd cursors, `opendir` handles,
    /// live mmaps — is dropped. Entries are sorted for a byte-deterministic image (so re-snapshotting an
    /// unchanged store yields identical bytes).
    fn snapshot(&self) -> FsSeed {
        let mut files: Vec<(String, Vec<u8>)> = self
            .files
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    v.lock().unwrap_or_else(|e| e.into_inner()).clone(),
                )
            })
            .collect();
        files.sort_by(|a, b| a.0.cmp(&b.0));
        let dirs: Vec<String> = self.dirs.iter().cloned().collect(); // BTreeSet ⇒ already sorted
        (files, dirs)
    }

    /// A durability barrier (`msync`/`sync`): `true` ⇒ drop this write (crashed or crashing now).
    fn crash_barrier(&mut self) -> bool {
        self.crash.as_mut().is_some_and(CrashCtl::barrier)
    }
    /// Whether the backing store is frozen by a tripped crash (persisting ops become no-ops).
    fn crash_frozen(&self) -> bool {
        self.crash.as_ref().is_some_and(|c| c.crashed)
    }
}

/// The [`MemFsHandle::capture_state`] layout version; any other refuses to decode.
const STATE_VERSION: u8 = 1;

impl MemFsState {
    /// Serialize the whole store (#1491). Little-endian: the version byte; files (path, buffer id)
    /// sorted by path; dirs; the open table (per slot: absent, or buffer id + cursor + mode bits);
    /// the `opendir` table (per slot: absent, or the names still to yield); the live mappings
    /// (base, len, file offset, buffer id); the crash controller; then the buffers, each once. A
    /// buffer id names a shared `Arc`, so aliasing survives the round trip.
    fn encode(&self) -> Vec<u8> {
        let mut bufs: Vec<Vec<u8>> = Vec::new();
        let mut ids: HashMap<usize, u32> = HashMap::new();
        let mut id = |a: &Arc<Mutex<Vec<u8>>>| -> u32 {
            *ids.entry(Arc::as_ptr(a) as usize).or_insert_with(|| {
                bufs.push(a.lock().unwrap_or_else(|e| e.into_inner()).clone());
                (bufs.len() - 1) as u32
            })
        };
        let mut out = vec![STATE_VERSION];
        let put_u32 = |o: &mut Vec<u8>, v: usize| o.extend_from_slice(&(v as u32).to_le_bytes());
        let put_u64 = |o: &mut Vec<u8>, v: u64| o.extend_from_slice(&v.to_le_bytes());
        let put_str = |o: &mut Vec<u8>, s: &str| {
            o.extend_from_slice(&(s.len() as u32).to_le_bytes());
            o.extend_from_slice(s.as_bytes());
        };

        let mut files: Vec<(&String, &Arc<Mutex<Vec<u8>>>)> = self.files.iter().collect();
        files.sort_by(|a, b| a.0.cmp(b.0));
        put_u32(&mut out, files.len());
        for (path, data) in files {
            put_str(&mut out, path);
            put_u32(&mut out, id(data) as usize);
        }
        put_u32(&mut out, self.dirs.len());
        for d in &self.dirs {
            put_str(&mut out, d);
        }
        put_u32(&mut out, self.open.len());
        for o in &self.open {
            match o {
                None => out.push(0),
                Some(o) => {
                    out.push(1);
                    put_u32(&mut out, id(&o.data) as usize);
                    put_u64(&mut out, o.pos as u64);
                    out.push(o.readable as u8 | (o.writable as u8) << 1 | (o.append as u8) << 2);
                }
            }
        }
        put_u32(&mut out, self.opendirs.len());
        for d in &self.opendirs {
            match d {
                None => out.push(0),
                Some(names) => {
                    out.push(1);
                    put_u32(&mut out, names.len());
                    for n in names {
                        put_str(&mut out, n);
                    }
                }
            }
        }
        put_u32(&mut out, self.maps.len());
        for m in &self.maps {
            put_u64(&mut out, m.base);
            put_u64(&mut out, m.len);
            put_u64(&mut out, m.file_off);
            put_u32(&mut out, id(&m.data) as usize);
        }
        match &self.crash {
            None => out.push(0),
            Some(c) => {
                out.push(1);
                out.push(c.countdown.is_some() as u8);
                put_u64(&mut out, c.countdown.unwrap_or(0));
                out.push(c.crashed as u8);
            }
        }
        put_u32(&mut out, bufs.len());
        for b in &bufs {
            put_u64(&mut out, b.len() as u64);
            out.extend_from_slice(b);
        }
        out
    }

    /// The inverse of [`encode`](Self::encode). Fail-closed: a wrong version, a truncated field, a
    /// buffer id past the table or trailing bytes refuse the whole store rather than yield part of it.
    fn decode(bytes: &[u8]) -> Result<MemFsState, String> {
        struct Rd<'a>(&'a [u8]);
        impl<'a> Rd<'a> {
            fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
                if n > self.0.len() {
                    return Err("memfs state: truncated".into());
                }
                let (h, t) = self.0.split_at(n);
                self.0 = t;
                Ok(h)
            }
            fn u8(&mut self) -> Result<u8, String> {
                Ok(self.take(1)?[0])
            }
            fn u32(&mut self) -> Result<usize, String> {
                Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
            }
            fn u64(&mut self) -> Result<u64, String> {
                Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            }
            fn len(&mut self) -> Result<usize, String> {
                usize::try_from(self.u64()?).map_err(|_| "memfs state: too large".to_string())
            }
            fn str(&mut self) -> Result<String, String> {
                let n = self.u32()?;
                String::from_utf8(self.take(n)?.to_vec())
                    .map_err(|_| "memfs state: path is not UTF-8".to_string())
            }
        }
        let mut r = Rd(bytes);
        if r.u8()? != STATE_VERSION {
            return Err("memfs state: unsupported version".into());
        }
        let mut files = Vec::new();
        for _ in 0..r.u32()? {
            files.push((r.str()?, r.u32()?));
        }
        let mut dirs = std::collections::BTreeSet::new();
        for _ in 0..r.u32()? {
            dirs.insert(r.str()?);
        }
        let mut open = Vec::new();
        for _ in 0..r.u32()? {
            open.push(match r.u8()? {
                0 => None,
                1 => Some((r.u32()?, r.len()?, r.u8()?)),
                _ => return Err("memfs state: bad open-slot tag".into()),
            });
        }
        let mut opendirs = Vec::new();
        for _ in 0..r.u32()? {
            opendirs.push(match r.u8()? {
                0 => None,
                1 => {
                    let mut names = Vec::new();
                    for _ in 0..r.u32()? {
                        names.push(r.str()?);
                    }
                    Some(names)
                }
                _ => return Err("memfs state: bad opendir-slot tag".into()),
            });
        }
        let mut maps = Vec::new();
        for _ in 0..r.u32()? {
            maps.push((r.u64()?, r.u64()?, r.u64()?, r.u32()?));
        }
        let crash = match r.u8()? {
            0 => None,
            1 => {
                let armed = r.u8()? != 0;
                let countdown = r.u64()?;
                let crashed = r.u8()? != 0;
                Some(CrashCtl {
                    countdown: armed.then_some(countdown),
                    crashed,
                })
            }
            _ => return Err("memfs state: bad crash tag".into()),
        };
        let mut bufs = Vec::new();
        for _ in 0..r.u32()? {
            let n = r.len()?;
            bufs.push(Arc::new(Mutex::new(r.take(n)?.to_vec())));
        }
        if !r.0.is_empty() {
            return Err("memfs state: trailing bytes".into());
        }
        let buf = |i: usize| {
            bufs.get(i)
                .cloned()
                .ok_or_else(|| "memfs state: buffer id out of range".to_string())
        };
        Ok(MemFsState {
            files: files
                .into_iter()
                .map(|(p, i)| Ok((p, buf(i)?)))
                .collect::<Result<_, String>>()?,
            dirs,
            open: open
                .into_iter()
                .map(|o| {
                    o.map(|(i, pos, mode)| {
                        Ok(MemOpen {
                            data: buf(i)?,
                            pos,
                            readable: mode & 1 != 0,
                            writable: mode & 2 != 0,
                            append: mode & 4 != 0,
                        })
                    })
                    .transpose()
                })
                .collect::<Result<_, String>>()?,
            opendirs,
            maps: maps
                .into_iter()
                .map(|(base, len, file_off, i)| {
                    Ok(MemMapping {
                        base,
                        len,
                        data: buf(i)?,
                        file_off,
                    })
                })
                .collect::<Result<_, String>>()?,
            crash,
        })
    }

    fn handle(&mut self, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
        let mut mem = mem;
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        match op {
            FS_OPEN => {
                let path = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let flags = a(2);
                let data = match self.files.get(&path) {
                    Some(d) => {
                        if flags & O_TRUNC != 0 {
                            d.lock().unwrap_or_else(|e| e.into_inner()).clear();
                        }
                        d.clone()
                    }
                    None => {
                        // A **read-only** open of an explicitly-tracked directory (`mkdir`'d or seeded)
                        // — e.g. Postgres `fsync`s directories at checkpoint via `open(dir, O_RDONLY)` +
                        // `fsync`. Return a read-only fd over an empty buffer: `sync`/`close` succeed,
                        // reads yield EOF, writes are refused (matches a real directory fd). Narrow by
                        // design: only a read-only open (a write/create never resolves to a dir), and
                        // only `dirs` (not the `""` root or a mere file-key prefix), so an ordinary
                        // `open(name, "w")` that happens to prefix another file still creates a file.
                        let write_intent = flags & (O_CREATE | O_WRITE | O_APPEND | O_TRUNC) != 0;
                        if !write_intent && self.dirs.contains(&path) {
                            let o = MemOpen {
                                data: Arc::new(Mutex::new(Vec::new())),
                                pos: 0,
                                readable: true,
                                writable: false,
                                append: false,
                            };
                            let fd = alloc_fd(&mut self.open);
                            self.open[fd] = Some(o);
                            return fd as i64;
                        }
                        if flags & O_CREATE == 0 {
                            return ENOENT;
                        }
                        let d = Arc::new(Mutex::new(Vec::new()));
                        self.files.insert(path, d.clone());
                        d
                    }
                };
                let o = MemOpen {
                    data,
                    pos: 0,
                    readable: flags & O_READ != 0,
                    writable: flags & (O_WRITE | O_APPEND) != 0,
                    append: flags & O_APPEND != 0,
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
                if buf < 0 || len < 0 {
                    return EINVAL;
                }
                let data = o.data.lock().unwrap_or_else(|e| e.into_inner());
                let avail = data.len().saturating_sub(o.pos);
                let n = avail.min(len as usize);
                if n > 0 {
                    let Some(m) = mem.as_deref_mut() else {
                        return EFAULT;
                    };
                    if m.write_bytes(buf as u64, &data[o.pos..o.pos + n]).is_none() {
                        return EFAULT;
                    }
                }
                drop(data);
                o.pos += n;
                n as i64
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
                let (buf, len) = (a(1), a(2));
                if buf < 0 || len < 0 {
                    return EINVAL;
                }
                let bytes = match mem.as_deref() {
                    Some(m) => match m.read_bytes(buf as u64, len as u64) {
                        Some(b) => b,
                        None => return EFAULT,
                    },
                    None => return EFAULT,
                };
                let mut data = o.data.lock().unwrap_or_else(|e| e.into_inner());
                if o.append {
                    o.pos = data.len();
                }
                if o.pos > data.len() {
                    data.resize(o.pos, 0); // POSIX: writing past EOF zero-fills the gap
                }
                let end = o.pos + bytes.len();
                if end > data.len() {
                    data.resize(end, 0);
                }
                data[o.pos..end].copy_from_slice(&bytes);
                drop(data);
                o.pos = end;
                bytes.len() as i64
            }
            FS_SEEK => {
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                let size = o.data.lock().unwrap_or_else(|e| e.into_inner()).len() as i64;
                let base = match a(1) {
                    0 => 0,
                    1 => o.pos as i64,
                    2 => size,
                    _ => return EINVAL,
                };
                let Some(new) = base.checked_add(a(2)) else {
                    return EINVAL;
                };
                if new < 0 {
                    return EINVAL;
                }
                o.pos = new as usize;
                new
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
                let path = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                if self.files.remove(&path).is_none() {
                    return ENOENT;
                }
                0
            }
            FS_RENAME => {
                let from = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let to = match read_path(mem.as_deref(), a(2), a(3)).map(|p| norm(&p)) {
                    Ok(p) => p,
                    Err(e) => return e,
                };
                let Some(d) = self.files.remove(&from) else {
                    return ENOENT;
                };
                self.files.insert(to, d);
                0
            }
            FS_TRUNCATE => {
                if self.crash_frozen() {
                    return 0; // power-loss: the resize never reaches the backing file
                }
                let Some(Some(o)) = self.open.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if !o.writable {
                    return EBADF; // POSIX ftruncate needs a writable descriptor
                }
                let len = a(1);
                if len < 0 {
                    return EINVAL;
                }
                // POSIX: shrink discards, grow zero-fills; the cursor is untouched.
                o.data
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .resize(len as usize, 0);
                0
            }
            FS_SYNC => {
                // Memory is always "durable" here — a durability barrier for crash injection, then
                // validate the fd (nothing to flush).
                if self.crash_barrier() {
                    return 0;
                }
                let Some(Some(_)) = self.open.get(a(0) as usize) else {
                    return EBADF;
                };
                0
            }
            FS_MMAP => {
                let (fd, foff, len, buf) = (a(0), a(1), a(2), a(3));
                if foff < 0 || len < 0 || buf < 0 {
                    return EINVAL;
                }
                let Some(Some(o)) = self.open.get(fd as usize) else {
                    return EBADF;
                };
                let data = o.data.clone();
                // Copy the file region into the guest buffer (zero-fill past EOF, like a file-backed
                // mmap of a hole).
                let mut region = vec![0u8; len as usize];
                {
                    let d = data.lock().unwrap_or_else(|e| e.into_inner());
                    let start = (foff as usize).min(d.len());
                    let end = (foff as usize + len as usize).min(d.len());
                    if end > start {
                        region[..end - start].copy_from_slice(&d[start..end]);
                    }
                }
                let Some(m) = mem.as_deref_mut() else {
                    return EFAULT;
                };
                if m.write_bytes(buf as u64, &region).is_none() {
                    return EFAULT;
                }
                self.maps.push(MemMapping {
                    base: buf as u64,
                    len: len as u64,
                    data,
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
                    return EINVAL; // no mapping contains this address
                };
                let n = (len as u64).min(map.base + map.len - buf as u64) as usize;
                let file_pos = map.file_off + (buf as u64 - map.base);
                let data = map.data.clone(); // end the borrow of `self.maps` before `crash_barrier`
                let Some(m) = mem.as_deref() else {
                    return EFAULT;
                };
                let Some(bytes) = m.read_bytes(buf as u64, n as u64) else {
                    return EFAULT;
                };
                if self.crash_barrier() {
                    return 0; // power-loss: this msync's bytes never reach the file
                }
                let mut d = data.lock().unwrap_or_else(|e| e.into_inner());
                let end = file_pos as usize + n;
                if end > d.len() {
                    d.resize(end, 0);
                }
                d[file_pos as usize..end].copy_from_slice(&bytes);
                0
            }
            FS_MUNMAP => {
                let buf = a(0);
                // Flush the whole mapping, then drop it (LMDB msyncs explicitly before close, but a
                // final flush keeps `munmap` self-contained) — unless a crash has frozen the store,
                // in which case a real `munmap` on a dead process would flush nothing.
                let Some(idx) = self.maps.iter().position(|m| m.base == buf as u64) else {
                    return EINVAL;
                };
                let map = self.maps.remove(idx);
                if !self.crash_frozen() {
                    if let Some(m) = mem.as_deref() {
                        if let Some(bytes) = m.read_bytes(map.base, map.len) {
                            let mut d = map.data.lock().unwrap_or_else(|e| e.into_inner());
                            let end = (map.file_off + map.len) as usize;
                            if end > d.len() {
                                d.resize(end, 0);
                            }
                            d[map.file_off as usize..end].copy_from_slice(&bytes);
                        }
                    }
                }
                0
            }
            FS_CRASH_ARM => arm_crash(self.crash.as_mut(), a(0)),
            FS_STAT => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                // A file name wins over a same-named implicit dir prefix (there can't be both).
                let (mode, size) = if let Some(d) = self.files.get(&key) {
                    let len = d.lock().unwrap_or_else(|e| e.into_inner()).len() as i64;
                    (S_IFREG | 0o644, len)
                } else if self.is_dir(&key) {
                    // Owner-only (0700), the natural mode for a hermetic in-memory store — and the mode
                    // Postgres' `checkDataDir` demands of its data directory (0700 or 0750; a
                    // world/group-readable data dir is a fatal error). Differential stat tests compare
                    // only `S_IFMT`, so the perm bits are free to model an owner-private fs.
                    (S_IFDIR | 0o700, 0)
                } else {
                    return ENOENT;
                };
                // A stable synthetic identity: mtime 0, one link, a hash-free ino of 0 (Postgres uses
                // st_ino/st_dev only for cross-file identity, which a fresh per-run store never needs).
                let buf = stat_bytes(mode, 1, size, 0, 0, 0, 0, 0, 0, 4096, (size + 511) / 512);
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
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if key.is_empty() || self.is_dir(&key) || self.files.contains_key(&key) {
                    return EEXIST;
                }
                self.dirs.insert(key);
                0
            }
            FS_RMDIR => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if self.files.contains_key(&key) {
                    return ENOTDIR;
                }
                if !self.is_dir(&key) {
                    return ENOENT;
                }
                if !self.children_of(&key).is_empty() {
                    return ENOTEMPTY;
                }
                if !self.dirs.remove(&key) {
                    return ENOENT; // an implicit (non-empty) dir has no explicit entry to remove
                }
                0
            }
            FS_OPENDIR => {
                let key = match read_path(mem.as_deref(), a(0), a(1)).map(|p| norm(&p)) {
                    Ok(k) => k,
                    Err(e) => return e,
                };
                if self.files.contains_key(&key) {
                    return ENOTDIR;
                }
                if !self.is_dir(&key) {
                    return ENOENT;
                }
                let mut kids = self.children_of(&key);
                kids.reverse(); // pop() yields them in sorted order
                let dh = self.opendirs.iter().position(|s| s.is_none()).unwrap_or({
                    self.opendirs.push(None);
                    self.opendirs.len() - 1
                });
                self.opendirs[dh] = Some(kids);
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
                    entries.push(name); // leave the walk where it was
                    return EINVAL;
                }
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
                let Some(slot) = self.opendirs.get_mut(a(0) as usize) else {
                    return EBADF;
                };
                if slot.take().is_none() {
                    return EBADF;
                }
                0
            }
            _ => EINVAL,
        }
    }
}

/// [`FS_CRASH_ARM`] handler shared by both backends: `n < 0` disarms, `n >= 0` arms the crash to trip
/// after `n` further durability barriers. `-EINVAL` when the backend has no controller (the default,
/// non-`crashy` grants) — so the op simply does not exist on a shipping capability.
pub fn arm_crash(ctl: Option<&mut CrashCtl>, n: i64) -> i64 {
    let Some(c) = ctl else {
        return EINVAL;
    };
    c.countdown = if n < 0 { None } else { Some(n as u64) };
    c.crashed = false;
    0
}

/// A live handle onto a store mounted by [`mem_fs_seeded_shared`], letting the caller serialize the
/// **current** filesystem back out — e.g. to persist a browser Postgres session across page reloads
/// (snapshot the data dir, stash the image, reboot from it next visit). Cloneable; every clone observes
/// the same live store. The guest that owns the mount runs single-threaded, so a snapshot taken while it
/// is suspended (parked at a stdin read, between queries) is a quiescent, crash-consistent point-in-time
/// image — exactly the state Postgres' startup recovery expects to replay.
#[derive(Clone)]
pub struct MemFsHandle(Arc<Mutex<MemFsState>>);

impl MemFsHandle {
    /// A fresh, empty store. `crashy` enables the **test-only** crash-injection op ([`FS_CRASH_ARM`]).
    pub fn new(crashy: bool) -> MemFsHandle {
        MemFsHandle(Arc::new(Mutex::new(MemFsState {
            crash: crashy.then(CrashCtl::default),
            ..MemFsState::default()
        })))
    }

    /// A store holding `files` (path → bytes) and the `dirs` that must exist even when empty.
    pub fn seeded(files: &[(String, Vec<u8>)], dirs: &[String]) -> MemFsHandle {
        let mut st = MemFsState::default();
        for (p, data) in files {
            st.files.insert(norm(p), Arc::new(Mutex::new(data.clone())));
        }
        for d in dirs {
            st.dirs.insert(norm(d));
        }
        MemFsHandle(Arc::new(Mutex::new(st)))
    }

    /// A store rebuilt from a [`capture_state`](Self::capture_state) — what a thaw's registrar mints
    /// for a `vm_fs`/`fs` the artifact names. `Err` on bytes this version cannot fully read.
    pub fn from_state(bytes: &[u8]) -> Result<MemFsHandle, String> {
        Ok(MemFsHandle(Arc::new(Mutex::new(MemFsState::decode(
            bytes,
        )?))))
    }

    /// #1491 — the **whole** store as bytes: every file, directory, open descriptor (with its
    /// cursor and mode), `opendir` handle and live mapping. Files that share a buffer — an open
    /// descriptor and its path, two descriptors on one file, a removed file still held open — still
    /// share one after a round trip. There is no partial form: a store that came back with its files
    /// but not its cursors would look restored and read from the wrong offset (INVARIANTS #9c).
    pub fn capture_state(&self) -> Vec<u8> {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).encode()
    }

    /// Replace this live store's contents with a [`capture_state`](Self::capture_state), in place,
    /// so every handler already granted over it sees the restored store. `Err` (and the store left
    /// as it was) on bytes this version cannot fully read.
    pub fn restore_state(&self, bytes: &[u8]) -> Result<(), String> {
        let restored = MemFsState::decode(bytes)?;
        *self.0.lock().unwrap_or_else(|e| e.into_inner()) = restored;
        Ok(())
    }

    /// A handler over this live store (the grant half: every handler minted here shares it).
    pub fn handler(&self) -> HostProc {
        let st = self.0.clone();
        Box::new(
            move |op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn GuestMem>,
                  _minter: Option<&mut dyn temen_interp::RegionMinter>| {
                Ok(vec![st
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .handle(op, args, mem)])
            },
        ) as HostProc
    }

    /// Declare this store as the state of the capability granted at `handle` (#1455's pair), so a
    /// checkpoint, a moment, a cap tape and a §12 artifact all carry it, and an in-session rewind
    /// puts it back. One definition of the store's state, read every way.
    pub fn declare_state(&self, host: &mut temen_interp::Host, handle: i32) {
        let (c, r) = (self.clone(), self.clone());
        host.set_cap_state_capture(handle, Box::new(move || c.capture_state()));
        host.set_cap_state_restore(
            handle,
            Box::new(move |b| {
                // The bytes are this store's own capture, so a decode failure is a bug, not input.
                let restored = r.restore_state(b);
                debug_assert!(restored.is_ok(), "memfs state restore: {restored:?}");
            }),
        );
    }

    /// The current filesystem as a `(files, dirs)` seed (see [`MemFsState::snapshot`]).
    pub fn seed(&self) -> FsSeed {
        self.0.lock().unwrap_or_else(|e| e.into_inner()).snapshot()
    }

    /// The current filesystem serialized straight to a shippable [`encode_image`] blob — the bytes a
    /// caller persists and later re-mounts via [`decode_image`] + [`mem_fs_seeded_shared`].
    pub fn image(&self) -> Vec<u8> {
        let (files, dirs) = self.seed();
        encode_image(&files, &dirs)
    }
}

/// A seeded store's `HostProc` **and** its [`MemFsHandle`], so the mount can be snapshotted back out
/// later (the persistent-session persistence path): one live store shared between the handler and the
/// handle through an `Arc<Mutex<..>>`, locked per op (uncontended in the single-threaded browser).
pub fn mem_fs_seeded_shared(
    files: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> (HostProc, MemFsHandle) {
    let (factory, handle) = mem_fs_shared_factory(files, dirs);
    (factory(), handle)
}

/// Like [`mem_fs_seeded_shared`] but returns a **reusable factory** granting the *same* live store to
/// every domain it's called for, instead of a single `HostProc`. This is the **cross-domain shared
/// memfs**: unlike a fresh [`MemFsHandle::seeded`] per grant (so two domains get isolated
/// filesystems), every `HostProc` this yields closes over one shared
/// `Arc<Mutex<MemFsState>>`, so a file one domain writes another domain reads — the file hand-off a
/// multi-phase pipeline needs (NIM.md §3c, W4: phase N writes `x.nif`, phase N+1 reads it). The
/// returned [`MemFsHandle`] observes the same store (snapshot/seed it host-side). Granting this to a
/// spawned *child* (vs. a top-level domain) is a separate, security-shaped decision — what fs
/// authority a child inherits — and lives in the spawn path, not here.
pub fn mem_fs_shared_factory(
    files: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> (impl Fn() -> HostProc + Send + Sync + 'static, MemFsHandle) {
    let handle = MemFsHandle::seeded(&files, &dirs);
    let factory = {
        let handle = handle.clone();
        move || handle.handler()
    };
    (factory, handle)
}

/// #1491 — grant a guest-private, read-write in-memory filesystem on the **`vm_fs`** seam: the
/// chibicc `__vm_fs` builtin's `call.sym "vm_fs"`, one flat call with the fs op in `args[0]` and its
/// arguments after. Seeded from `seed` when given (a launch's fs-image), else empty. Registers the
/// name and declares the store's state, so every rebuild, rewind and freeze carries the files the
/// guest wrote. The one definition the browser Run path and the debug on-ramp both grant.
pub fn grant_vm_fs(host: &mut temen_interp::Host, seed: Option<&FsSeed>) -> i32 {
    let fs = match seed {
        Some((files, dirs)) => MemFsHandle::seeded(files, dirs),
        None => MemFsHandle::new(false),
    };
    let h = host.grant_host_proc(vm_fs_handler(&fs));
    host.register_cap_name("vm_fs", h);
    fs.declare_state(host, h);
    h
}

/// The `vm_fs` seam's handler over `fs`: the fs op in `args[0]`, its arguments after. What
/// [`grant_vm_fs`] grants, and what a thaw's registrar hands back for a `vm_fs` the artifact names
/// (over [`MemFsHandle::from_state`] of the state it carries).
pub fn vm_fs_handler(fs: &MemFsHandle) -> HostProc {
    let mut inner = fs.handler();
    Box::new(
        move |_slot_op: u32,
              args: &[i64],
              mem: Option<&mut dyn GuestMem>,
              minter: Option<&mut dyn temen_interp::RegionMinter>| {
            let (op, rest) = args
                .split_first()
                .map(|(o, r)| (*o as u32, r))
                .unwrap_or((0, &[][..]));
            inner(op, rest, mem, minter)
        },
    )
}

/// A filesystem seed: `(files as (relative-path, bytes), directory relative-paths)`. The material both
/// [`mem_fs_seeded`] mounts and [`encode_image`] serializes.
pub type FsSeed = (Vec<(String, Vec<u8>)>, Vec<String>);

/// Walk a host directory into a [`FsSeed`] — every regular file's `(relative-path, bytes)` plus every
/// directory's relative path (so empty ones survive). Symlinks are followed. The raw material for both
/// [`mem_fs_from_host_dir`] and [`encode_image`] (build a shippable data image once).
pub fn read_host_dir(root: &Path) -> std::io::Result<FsSeed> {
    let mut files: Vec<(String, Vec<u8>)> = Vec::new();
    let mut dirs: Vec<String> = Vec::new();
    fn walk(
        base: &Path,
        cur: &Path,
        files: &mut Vec<(String, Vec<u8>)>,
        dirs: &mut Vec<String>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(cur)? {
            let path = entry?.path();
            let rel = path
                .strip_prefix(base)
                .unwrap_or(&path)
                .to_string_lossy()
                .into_owned();
            // `is_dir`/`is_file` follow symlinks, so a symlinked file/dir is captured as its target.
            if path.is_dir() {
                dirs.push(rel);
                walk(base, &path, files, dirs)?;
            } else if path.is_file() {
                files.push((rel, std::fs::read(&path)?));
            }
        }
        Ok(())
    }
    walk(root, root, &mut files, &mut dirs)?;
    Ok((files, dirs))
}

use temen_encode::wire;

/// The fs-image payload format version — the TEMEN header's `version` for `kind = fs-image`.
const IMAGE_VERSION: u16 = 1;

/// Serialize a `(files, dirs)` seed into a **self-contained data image** — a flat, portable byte blob a
/// demo ships and mounts with [`mem_fs_from_archive`] (no host filesystem needed, e.g. in the browser).
/// Format (all little-endian): the TEMEN wire header (`kind` = fs-image, `version` = 1; `WIRE.md`);
/// `u32` dir count, then each dir `u32 len + path`; `u32`
/// file count, then each file `u32 path-len + path + u64 data-len + data`. Paths are stored verbatim
/// (normalization happens at mount time, as for any seed).
pub fn encode_image(files: &[(String, Vec<u8>)], dirs: &[String]) -> Vec<u8> {
    let mut out = Vec::new();
    wire::write_header(&mut out, wire::KIND_FS_IMAGE, IMAGE_VERSION, 0);
    out.extend_from_slice(&(dirs.len() as u32).to_le_bytes());
    for d in dirs {
        out.extend_from_slice(&(d.len() as u32).to_le_bytes());
        out.extend_from_slice(d.as_bytes());
    }
    out.extend_from_slice(&(files.len() as u32).to_le_bytes());
    for (p, data) in files {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p.as_bytes());
        out.extend_from_slice(&(data.len() as u64).to_le_bytes());
        out.extend_from_slice(data);
    }
    out
}

/// Parse a [`encode_image`] blob back into a [`FsSeed`]. `Err` on a bad magic or a truncated/oversized
/// field (fail-closed — a corrupt image never yields a partial mount).
pub fn decode_image(bytes: &[u8]) -> Result<FsSeed, String> {
    // The unified TEMEN wire header (WIRE.md): magic, then this format's own kind/version/flags,
    // each fail-closed before a single field is read.
    let (hdr, payload) = wire::read_header(bytes).map_err(|e| match e {
        wire::HeaderError::Truncated => "image: truncated".to_string(),
        wire::HeaderError::BadMagic => "image: bad magic".to_string(),
    })?;
    if hdr.kind != wire::KIND_FS_IMAGE {
        return Err("image: not an fs-image".into());
    }
    if hdr.version != IMAGE_VERSION {
        return Err(format!("image: unsupported version {}", hdr.version));
    }
    if hdr.flags != 0 {
        return Err("image: reserved flags set".into());
    }
    let bytes = payload;
    let mut p = 0usize;
    let take = |p: &mut usize, n: usize| -> Result<&[u8], String> {
        let end = p.checked_add(n).ok_or("image: length overflow")?;
        let s = bytes.get(*p..end).ok_or("image: truncated")?;
        *p = end;
        Ok(s)
    };
    let u32at = |p: &mut usize| -> Result<usize, String> {
        Ok(u32::from_le_bytes(take(p, 4)?.try_into().unwrap()) as usize)
    };
    let u64at = |p: &mut usize| -> Result<usize, String> {
        let v = u64::from_le_bytes(take(p, 8)?.try_into().unwrap());
        usize::try_from(v).map_err(|_| "image: entry too large".to_string())
    };
    let n_dirs = u32at(&mut p)?;
    let mut dirs = Vec::with_capacity(n_dirs);
    for _ in 0..n_dirs {
        let len = u32at(&mut p)?;
        let s = std::str::from_utf8(take(&mut p, len)?).map_err(|_| "image: non-UTF-8 dir")?;
        dirs.push(s.to_string());
    }
    let n_files = u32at(&mut p)?;
    let mut files = Vec::with_capacity(n_files);
    for _ in 0..n_files {
        let len = u32at(&mut p)?;
        let s = std::str::from_utf8(take(&mut p, len)?).map_err(|_| "image: non-UTF-8 path")?;
        let dlen = u64at(&mut p)?;
        files.push((s.to_string(), take(&mut p, dlen)?.to_vec()));
    }
    Ok((files, dirs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A flat `GuestMem` over a `Vec<u8>` — enough to drive the fs ops (they only read paths / read
    /// write-payloads / write read-results within `[0, len)`).
    struct VecMem(Vec<u8>);
    impl GuestMem for VecMem {
        fn read_bytes(&self, ptr: u64, len: u64) -> Option<Vec<u8>> {
            let end = (ptr as usize).checked_add(len as usize)?;
            self.0.get(ptr as usize..end).map(<[u8]>::to_vec)
        }
        fn write_bytes(&mut self, ptr: u64, data: &[u8]) -> Option<()> {
            let end = (ptr as usize).checked_add(data.len())?;
            self.0.get_mut(ptr as usize..end)?.copy_from_slice(data);
            Some(())
        }
    }

    /// #1491 — a captured store comes back **whole**: run one op sequence against the original and
    /// against a store rebuilt from its capture, and every answer, every byte read and the final
    /// capture agree. The sequence leans on what an image-only snapshot would lose: a cursor mid-file,
    /// two descriptors sharing one file, a removed file still held open, a half-read `opendir`.
    #[test]
    fn a_captured_store_restores_its_descriptors_and_their_sharing() {
        let call = |fs: &mut HostProc, op: u32, args: &[i64], mem: &mut VecMem| -> i64 {
            fs(op, args, Some(mem), None).expect("host fn")[0]
        };
        let a = MemFsHandle::seeded(&[("keep".into(), b"k".to_vec())], &["empty".into()]);
        let mut fs = a.handler();
        // Paths "a" at 0, "b" at 1 and "." at 2; payload "hello" at 16; reads land at 32.
        let mut mem = VecMem(vec![0u8; 64]);
        mem.0[..3].copy_from_slice(b"ab.");
        mem.0[16..21].copy_from_slice(b"hello");
        let rw = call(
            &mut fs,
            FS_OPEN,
            &[0, 1, O_CREATE | O_READ | O_WRITE],
            &mut mem,
        );
        assert_eq!(call(&mut fs, FS_WRITE, &[rw, 16, 5], &mut mem), 5);
        assert_eq!(call(&mut fs, FS_SEEK, &[rw, 0, 1], &mut mem), 1);
        let ro = call(&mut fs, FS_OPEN, &[0, 1, O_READ], &mut mem);
        let wb = call(&mut fs, FS_OPEN, &[1, 1, O_CREATE | O_WRITE], &mut mem);
        assert_eq!(call(&mut fs, FS_REMOVE, &[1, 1], &mut mem), 0);
        let dh = call(&mut fs, FS_OPENDIR, &[2, 1], &mut mem);
        assert!(call(&mut fs, FS_READDIR, &[dh, 40, 16], &mut mem) > 0);

        let bytes = a.capture_state();
        let b = MemFsHandle::from_state(&bytes).expect("a capture decodes");
        assert_eq!(
            b.capture_state(),
            bytes,
            "recapturing a restored store is byte-identical"
        );

        let after = |fs: &mut HostProc| {
            let mut mem = VecMem(vec![0u8; 64]);
            mem.0[16..17].copy_from_slice(b"!");
            let mut out = vec![
                call(fs, FS_READ, &[rw, 32, 8], &mut mem), // the cursor: "ello", not "hello"
                call(fs, FS_WRITE, &[rw, 16, 1], &mut mem), // at the end: "hello!"
                call(fs, FS_READ, &[ro, 48, 8], &mut mem), // the sharing: sees the "!"
                call(fs, FS_WRITE, &[wb, 16, 1], &mut mem), // the removed file is still writable
                call(fs, FS_READDIR, &[dh, 40, 16], &mut mem), // the rest of the listing
                call(fs, FS_READDIR, &[dh, 40, 16], &mut mem),
            ];
            out.extend(mem.0.iter().map(|&x| x as i64));
            out
        };
        let (mut fa, mut fb) = (a.handler(), b.handler());
        let want = after(&mut fa);
        assert_eq!(
            &want[..3],
            &[4, 1, 6],
            "cursor at 1, then append, then the sharer reads all six"
        );
        assert_eq!(
            after(&mut fb),
            want,
            "the restored store answers exactly as the original"
        );
        assert_eq!(
            b.capture_state(),
            a.capture_state(),
            "and ends in the same state"
        );
    }

    /// A capture is all or nothing: bytes this version cannot fully read refuse the whole store.
    #[test]
    fn a_damaged_capture_refuses_rather_than_restoring_part_of_a_store() {
        let bytes = MemFsHandle::seeded(&[("f".into(), b"data".to_vec())], &[]).capture_state();
        assert!(MemFsHandle::from_state(&bytes).is_ok());
        assert!(
            MemFsHandle::from_state(&bytes[..bytes.len() - 1]).is_err(),
            "truncated"
        );
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(MemFsHandle::from_state(&extra).is_err(), "trailing bytes");
        let mut version = bytes.clone();
        version[0] = STATE_VERSION + 1;
        assert!(
            MemFsHandle::from_state(&version).is_err(),
            "another version"
        );
        // The file's buffer id sits after the version, the file count and the path ("f").
        let mut dangling = bytes;
        dangling[1 + 4 + 4 + 1] = 7;
        assert!(
            MemFsHandle::from_state(&dangling).is_err(),
            "a buffer id past the table"
        );

        // A live store keeps its contents when handed bad bytes.
        let live = MemFsHandle::seeded(&[("f".into(), b"data".to_vec())], &[]);
        let before = live.capture_state();
        assert!(live.restore_state(&[0xff]).is_err());
        assert_eq!(live.capture_state(), before);
    }

    /// `mem_fs_seeded_shared` snapshots the **live** store — writes and removes made through the granted
    /// `HostProc` after the mount show up in `MemFsHandle::image`, and the image round-trips through
    /// `decode_image` back to a mountable seed. This is the persistence hinge: a Postgres session's data
    /// dir, mutated by DDL/DML, must serialize back out exactly.
    #[test]
    fn shared_handle_snapshots_live_writes() {
        // Seed: one existing file + one empty dir (the shapes an initdb tree has).
        let seed_files = vec![("base/1".to_string(), b"seed".to_vec())];
        let seed_dirs = vec!["pg_wal".to_string()];
        let (mut fs, handle) = mem_fs_seeded_shared(seed_files, seed_dirs);

        // The seed is visible immediately, before any op.
        let (f0, d0) = handle.seed();
        assert_eq!(f0, vec![("base/1".to_string(), b"seed".to_vec())]);
        assert_eq!(d0, vec!["pg_wal".to_string()]);

        // Lay out guest memory: path "base/2" at 0, payload "hello" at 16.
        let mut mem = VecMem(vec![0u8; 32]);
        mem.0[..6].copy_from_slice(b"base/2");
        mem.0[16..21].copy_from_slice(b"hello");
        let call = |fs: &mut HostProc, op: u32, args: &[i64], mem: &mut VecMem| -> i64 {
            fs(op, args, Some(mem), None).expect("host fn")[0]
        };

        // Create + write a new file through the granted handler (O_CREATE|O_WRITE).
        let fd = call(&mut fs, FS_OPEN, &[0, 6, O_CREATE | O_WRITE], &mut mem);
        assert!(fd >= 3, "fd = {fd}");
        assert_eq!(call(&mut fs, FS_WRITE, &[fd, 16, 5], &mut mem), 5);
        assert_eq!(call(&mut fs, FS_CLOSE, &[fd], &mut mem), 0);

        // Remove the seeded file (path "base/1" reuses the same 6-byte slot layout).
        mem.0[..6].copy_from_slice(b"base/1");
        assert_eq!(call(&mut fs, FS_REMOVE, &[0, 6], &mut mem), 0);

        // Snapshot → the write is captured and the removal propagated.
        let image = handle.image();
        let (files, dirs) = decode_image(&image).expect("round-trips");
        assert_eq!(files, vec![("base/2".to_string(), b"hello".to_vec())]);
        assert_eq!(dirs, vec!["pg_wal".to_string()]);

        // Re-mounting the image reproduces the same live state (the persistence loop closes).
        let (_fs2, handle2) = mem_fs_seeded_shared(files, dirs);
        assert_eq!(handle2.seed(), handle.seed());

        // Snapshotting an unchanged store is byte-identical (deterministic, sorted output).
        assert_eq!(handle.image(), handle.image());
    }

    /// The cross-domain file hand-off (NIM.md §3c, W4): two **independent** grants from one
    /// `mem_fs_shared_factory` — as two pipeline phases would each receive — see each other's writes.
    /// Phase A creates and writes "x"; phase B, a separate `HostProc`, opens and reads it back. Had
    /// the grants owned isolated stores (a fresh `MemFsHandle::seeded` per grant), B's read-only open
    /// would miss and this would fail — so the read-back witnesses one shared store across the two.
    #[test]
    fn two_grants_from_the_factory_share_one_store() {
        let (factory, _handle) = mem_fs_shared_factory(vec![], vec![]);
        let mut phase_a: HostProc = factory();
        let mut phase_b: HostProc = factory();
        let call = |fs: &mut HostProc, op: u32, args: &[i64], mem: &mut VecMem| -> i64 {
            fs(op, args, Some(mem), None).expect("host fn")[0]
        };

        // Phase A: create "x" (O_CREATE|O_WRITE) and write "hi".
        let mut mem_a = VecMem(vec![0u8; 32]);
        mem_a.0[..1].copy_from_slice(b"x");
        mem_a.0[16..18].copy_from_slice(b"hi");
        let fda = call(
            &mut phase_a,
            FS_OPEN,
            &[0, 1, O_CREATE | O_WRITE],
            &mut mem_a,
        );
        assert!(fda >= 3, "phase A open: fd = {fda}");
        assert_eq!(call(&mut phase_a, FS_WRITE, &[fda, 16, 2], &mut mem_a), 2);
        assert_eq!(call(&mut phase_a, FS_CLOSE, &[fda], &mut mem_a), 0);

        // Phase B (a separate grant): open "x" **read-only** (O_READ, no O_CREATE) and read it back.
        let mut mem_b = VecMem(vec![0u8; 32]);
        mem_b.0[..1].copy_from_slice(b"x");
        let fdb = call(&mut phase_b, FS_OPEN, &[0, 1, O_READ], &mut mem_b);
        assert!(fdb >= 3, "phase B open of A's file: fd = {fdb}");
        let n = call(&mut phase_b, FS_READ, &[fdb, 16, 8], &mut mem_b);
        assert_eq!(n, 2, "phase B read the 2 bytes A wrote");
        assert_eq!(
            &mem_b.0[16..18],
            b"hi",
            "phase B sees phase A's write across the domain boundary"
        );
    }
}
