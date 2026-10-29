//! #2510 — `host_fs`'s `replace` on Unix: each step on descriptors, relative to the target's
//! directory, which cap-std opened beneath the root (#2643).
//!
//! The temporary file is locked (`flock`) from before it has a name until it is renamed. That is
//! how a `replace` tells another's temporary file from one a crash left: a crash releases the
//! lock, so a leftover is one it can lock. On Linux the file is made without a name (`O_TMPFILE`)
//! and named once synced, so it can be seen only between then and the rename.

use super::refuse;
use cap_std::fs::{Dir, MetadataExt as _};
use std::ffi::{CStr, CString};
use std::fs::File;
use std::io;
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::fs::MetadataExt;
use temen_ir::errno::{EBUSY, EINVAL, EISDIR, EMLINK};

/// The file `name` in `dir`, open to read its protection from, or `None` if there is none.
/// Refused: a directory, anything but a regular file, a file with other hard links (R1), and a
/// file the process may not write in place, or read the protection of.
pub(super) fn target(dir: &Dir, name: &str) -> io::Result<Option<File>> {
    let at = dir.as_raw_fd();
    let name = CString::new(name)?;
    // Not following a link (`replace_target` followed them), and not waiting on a FIFO.
    let flags = libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_NOCTTY;
    let file = match openat(at, &name, flags, 0) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        opened => opened?,
    };
    let md = file.metadata()?;
    if md.is_dir() {
        return Err(refuse(EISDIR));
    }
    if !md.is_file() {
        return Err(refuse(EINVAL));
    }
    if md.nlink() > 1 {
        return Err(refuse(EMLINK));
    }
    // Writing it in place would fail, so replacing it does too: its mode, an ACL, a read-only
    // mount. The kernel decides, for the process's effective ids, as it would for the write.
    // SAFETY: `name` is NUL-terminated and lives for the call.
    cvt(unsafe { libc::faccessat(at, name.as_ptr(), libc::W_OK, libc::AT_EACCESS) })?;
    Ok(Some(file))
}

/// `replace`'s temporary file, in the target's directory.
pub(super) struct Temp<'a> {
    dir: &'a Dir,
    name: CString,
    pub(super) file: File,
    /// Whether `name` names it: not yet, for an `O_TMPFILE` file before [`Temp::sync`].
    named: bool,
    /// Whether to leave it as it is when dropped: renamed over the target, or left by a crash.
    kept: bool,
}

impl<'a> Temp<'a> {
    /// A new temporary file `name` in `dir`, locked. While it holds the new bytes of an existing
    /// `target` it is private (0600), until [`Self::protect`] gives it the target's mode. For a new
    /// file it has the host's defaults, as `open` with `O_CREAT` gives them: 0666 under the umask,
    /// or the directory's default ACL.
    pub(super) fn create(dir: &'a Dir, name: &str, target: Option<&File>) -> io::Result<Temp<'a>> {
        let at = dir.as_raw_fd();
        let name = CString::new(name)?;
        let mode = if target.is_some() { 0o600 } else { 0o666 };
        // [`Self::sync`] names it through /proc, so only where /proc is.
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if std::path::Path::new("/proc/self/fd").is_dir() {
            match openat(at, c".", libc::O_TMPFILE | libc::O_RDWR, mode) {
                Ok(file) => {
                    lock(&file)?;
                    let (named, kept) = (false, false);
                    return Ok(Temp {
                        dir,
                        name,
                        file,
                        named,
                        kept,
                    });
                }
                // The kernel, or this filesystem, has no `O_TMPFILE`: a named file, as elsewhere.
                Err(e) if matches!(e.raw_os_error(), Some(libc::EOPNOTSUPP | libc::EISDIR)) => {}
                Err(e) => return Err(e),
            }
        }
        let flags = libc::O_RDWR | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW;
        let file = match openat(at, &name, flags, mode) {
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                clear(dir, &name)?;
                openat(at, &name, flags, mode).map_err(busy)?
            }
            made => made?,
        };
        // Another `replace` took it for a leftover before it was locked, and it is that one's now.
        lock(&file)?;
        if !names(dir, &name, &file)? {
            return Err(refuse(EBUSY));
        }
        let (named, kept) = (true, false);
        Ok(Temp {
            dir,
            name,
            file,
            named,
            kept,
        })
    }

    /// Give the file `target`'s protection: its owner and group, where the process may set them,
    /// its extended attributes, which hold its ACLs, and its mode, last, as `chown` clears
    /// set-user-ID and set-group-ID bits.
    pub(super) fn protect(&self, target: &File) -> io::Result<()> {
        let md = target.metadata()?;
        let fd = self.file.as_raw_fd();
        // SAFETY: `fd` is open for the call.
        if unsafe { libc::fchown(fd, md.uid(), md.gid()) } != 0 {
            // A process that may not give a file away may still give it to a group it is in. If
            // it may not do that either, the file keeps the process's group.
            // SAFETY: as above; `uid_t::MAX` is -1, which leaves the owner as it is.
            unsafe { libc::fchown(fd, libc::uid_t::MAX, md.gid()) };
        }
        copy_xattrs(target, &self.file)?;
        // SAFETY: `fd` is open for the call.
        cvt(unsafe { libc::fchmod(fd, (md.mode() & 0o7777) as libc::mode_t) })
    }

    /// Make the file's bytes and protection durable, then give it its name if it has none yet.
    pub(super) fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()?;
        #[cfg(any(target_os = "linux", target_os = "android"))]
        if !self.named {
            let name_it = || link(&self.file, self.dir.as_raw_fd(), &self.name);
            match name_it() {
                Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                    clear(self.dir, &self.name)?;
                    name_it().map_err(busy)?;
                }
                named => named?,
            }
            self.named = true;
        }
        Ok(())
    }

    /// Rename the file over `target`, its directory's file of that name.
    pub(super) fn rename_over(&mut self, target: &str) -> io::Result<()> {
        let at = self.dir.as_raw_fd();
        let target = CString::new(target)?;
        // SAFETY: both names are NUL-terminated and live for the call.
        cvt(unsafe { libc::renameat(at, self.name.as_ptr(), at, target.as_ptr()) })?;
        self.kept = true;
        Ok(())
    }

    /// Make the rename durable: sync the directory.
    pub(super) fn sync_dir(&self) -> io::Result<()> {
        // cap-std holds a directory it resolves through as `O_PATH` where Linux has it, which
        // can't be synced, so open it again to sync it.
        openat(
            self.dir.as_raw_fd(),
            c".",
            libc::O_RDONLY | libc::O_DIRECTORY,
            0,
        )?
        .sync_all()
    }

    /// Leave the file as it is, as a crash does.
    pub(super) fn keep(&mut self) {
        self.kept = true;
    }
}

impl Drop for Temp<'_> {
    /// Remove the file after an error before the rename. It is locked, so its name is still this
    /// `replace`'s to remove.
    fn drop(&mut self) {
        if self.named && !self.kept {
            // SAFETY: `name` is NUL-terminated and lives for the call.
            unsafe { libc::unlinkat(self.dir.as_raw_fd(), self.name.as_ptr(), 0) };
        }
    }
}

/// Remove the temporary file `name` in `dir` that a crash left: one no `replace` holds, as a crash
/// releases its lock. `-EBUSY` if another `replace` holds it.
fn clear(dir: &Dir, name: &CStr) -> io::Result<()> {
    let at = dir.as_raw_fd();
    // For reading and writing, as `flock` over NFS takes a write lock for an exclusive one.
    let flags = libc::O_RDWR | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_NOCTTY;
    let file = match openat(at, name, flags, 0) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        opened => opened?,
    };
    lock(&file)?;
    // Removed and made again between the open and the lock: that one is another's.
    if !names(dir, name, &file)? {
        return Err(refuse(EBUSY));
    }
    // SAFETY: `name` is NUL-terminated and lives for the call.
    cvt(unsafe { libc::unlinkat(at, name.as_ptr(), 0) })
}

/// Lock `file` for this `replace`: `-EBUSY` if another holds it.
fn lock(file: &File) -> io::Result<()> {
    // SAFETY: the descriptor is open for the call.
    match cvt(unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) }) {
        Err(e) if e.kind() == io::ErrorKind::WouldBlock => Err(refuse(EBUSY)),
        locked => locked,
    }
}

/// Whether `name` in `dir` is `file`, not nothing or another file.
fn names(dir: &Dir, name: &CStr, file: &File) -> io::Result<bool> {
    let held = file.metadata()?;
    let name = name.to_str().map_err(|_| io::ErrorKind::InvalidInput)?;
    match dir.symlink_metadata(name) {
        Ok(named) => Ok(named.dev() == held.dev() && named.ino() == held.ino()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(e) => Err(e),
    }
}

/// A temporary file another `replace` made between this one's removing a leftover and making its
/// own is that one's: `-EBUSY`.
fn busy(e: io::Error) -> io::Error {
    match e.kind() {
        io::ErrorKind::AlreadyExists => refuse(EBUSY),
        _ => e,
    }
}

/// Give the unnamed file `file` the name `name` in the directory `at`.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn link(file: &File, at: RawFd, name: &CStr) -> io::Result<()> {
    let fd = CString::new(format!("/proc/self/fd/{}", file.as_raw_fd()))?;
    // SAFETY: both paths are NUL-terminated and live for the call.
    cvt(unsafe {
        libc::linkat(
            libc::AT_FDCWD,
            fd.as_ptr(),
            at,
            name.as_ptr(),
            libc::AT_SYMLINK_FOLLOW,
        )
    })
}

/// Give `to` exactly `from`'s extended attributes, which hold its ACLs: each of `from`'s set, and
/// each `to` has that `from` lacks removed, such as an ACL the directory's default gave it. One in
/// `security.` or `trusted.` the process may not set or remove stays as it is, as an owner and
/// group it may not set do.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn copy_xattrs(from: &File, to: &File) -> io::Result<()> {
    let (from, to) = (from.as_raw_fd(), to.as_raw_fd());
    let kept = xattr_names(from)?;
    for name in names_in(&kept) {
        let value = sized(|buf, len| {
            // SAFETY: `name` is NUL-terminated; `buf` is `len` bytes for the call to fill, or null
            // with `len` 0 to ask the size.
            unsafe { libc::fgetxattr(from, name.as_ptr(), buf.cast(), len) }
        })?;
        // SAFETY: `name` is NUL-terminated, and `value` is `value.len()` bytes.
        let set =
            unsafe { libc::fsetxattr(to, name.as_ptr(), value.as_ptr().cast(), value.len(), 0) };
        privileged(name, cvt(set))?;
    }
    for name in names_in(&xattr_names(to)?) {
        if !names_in(&kept).any(|k| k == name) {
            // SAFETY: `name` is NUL-terminated.
            privileged(name, cvt(unsafe { libc::fremovexattr(to, name.as_ptr()) }))?;
        }
    }
    Ok(())
}

/// The names of the extended attributes of `fd`, each NUL-terminated: none on a filesystem without
/// them.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn xattr_names(fd: RawFd) -> io::Result<Vec<u8>> {
    // SAFETY: `buf` is `len` bytes for the call to fill, or null with `len` 0 to ask the size.
    match sized(|buf, len| unsafe { libc::flistxattr(fd, buf.cast(), len) }) {
        Err(e) if e.raw_os_error() == Some(libc::EOPNOTSUPP) => Ok(Vec::new()),
        listed => listed,
    }
}

/// The NUL-terminated names in an extended-attribute listing.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn names_in(list: &[u8]) -> impl Iterator<Item = &CStr> {
    list.split_inclusive(|&b| b == 0)
        .filter_map(|name| CStr::from_bytes_with_nul(name).ok())
}

/// `done` for the attribute `name`, but `Ok` if it is one the process may not set or remove.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn privileged(name: &CStr, done: io::Result<()>) -> io::Result<()> {
    let name = name.to_bytes();
    let refused = |e: &io::Error| {
        matches!(
            e.raw_os_error(),
            Some(libc::EPERM | libc::EACCES | libc::EOPNOTSUPP)
        )
    };
    match done {
        Err(e)
            if refused(&e) && (name.starts_with(b"security.") || name.starts_with(b"trusted.")) =>
        {
            Ok(())
        }
        done => done,
    }
}

/// What a call that fills a buffer fills it with: `fill(null, 0)` gives the size, then
/// `fill(buf, len)` fills `buf`. If what it reads grew in between, it asks again.
#[cfg(any(target_os = "linux", target_os = "android"))]
fn sized(fill: impl Fn(*mut u8, usize) -> isize) -> io::Result<Vec<u8>> {
    loop {
        let len = usize::try_from(fill(std::ptr::null_mut(), 0))
            .map_err(|_| io::Error::last_os_error())?;
        let mut buf = vec![0u8; len];
        match usize::try_from(fill(buf.as_mut_ptr(), len)) {
            Ok(filled) => {
                buf.truncate(filled);
                return Ok(buf);
            }
            Err(_) => {
                let e = io::Error::last_os_error();
                if e.raw_os_error() != Some(libc::ERANGE) {
                    return Err(e);
                }
            }
        }
    }
}

/// Give `to` exactly `from`'s extended attributes and ACL: `fcopyfile` replaces `to`'s with them.
#[cfg(target_vendor = "apple")]
fn copy_xattrs(from: &File, to: &File) -> io::Result<()> {
    let what = libc::COPYFILE_ACL | libc::COPYFILE_XATTR;
    // SAFETY: both descriptors are open for the call, which takes no state.
    cvt(unsafe { libc::fcopyfile(from.as_raw_fd(), to.as_raw_fd(), std::ptr::null_mut(), what) })
}

/// The file `name` in the directory `at`, opened with `flags` and `O_CLOEXEC`, and made with
/// `mode` if `flags` make it.
fn openat(at: RawFd, name: &CStr, flags: libc::c_int, mode: libc::c_uint) -> io::Result<File> {
    // SAFETY: `name` is NUL-terminated and lives for the call.
    let fd = unsafe { libc::openat(at, name.as_ptr(), flags | libc::O_CLOEXEC, mode) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: `openat` succeeded, so `fd` is an open descriptor no one else owns.
    Ok(unsafe { File::from_raw_fd(fd) })
}

/// A libc call's result: its `-1` as the error it set.
fn cvt(ret: libc::c_int) -> io::Result<()> {
    match ret {
        -1 => Err(io::Error::last_os_error()),
        _ => Ok(()),
    }
}
