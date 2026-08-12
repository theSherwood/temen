//! svm filesystem: `File`, `read_dir`, `metadata` and friends reach the POSIX personality through the
//! PAL `host` bridge (svm-posix `OP_OPEN`/`OP_READ`/`OP_WRITE`/`OP_CLOSE`/`OP_LSEEK`/`OP_UNLINK`/
//! `OP_STAT`/`OP_OPENDIR`/`OP_READDIR`/`OP_CLOSEDIR`). Paths are passed as `(ptr, len)` byte slices and
//! buffers are written into guest-owned memory (no personality arena), matching the `env` overlay. The
//! backing store is the personality's in-memory filesystem, so there are no symlinks or mutable perms:
//! `lstat == stat`, `FilePermissions` is always writable, and the metadata-only ops (perms/times, mkdir,
//! rename, symlink) are `Unsupported`. Needs a granted `posix` cap (`run_with_caps`); without one every
//! entry point is `Unsupported`.
#![deny(unsafe_op_in_unsafe_fn)]
use crate::ffi::{OsStr, OsString};
use crate::fmt;
use crate::fs::TryLockError;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
use crate::sys::pal::host;
use crate::sys::time::SystemTime;
use crate::sys::{unsupported, unsupported_err};

// The metadata-only / directory-mutation surface has no host op on the memfs backend; borrow the flat
// `Unsupported` implementations (and the shared `Dir`/`FileTimes` types) from the `unsupported` PAL,
// exactly as `vexos` does. `mkdir` (via our own `DirBuilder`), `rename`, and `rmdir` are implemented
// below over the memfs directory ops; the rest (symlinks, hard links, times, canonicalize) have no
// memfs backing and stay `Unsupported`.
#[expect(dead_code)]
#[path = "unsupported.rs"]
mod unsupported_fs;
pub use crate::sys::fs::common::{copy, exists, remove_dir_all};
pub use unsupported_fs::{
    Dir, FileTimes, canonicalize, link, readlink, set_times, set_times_nofollow, symlink,
};

// `set_perm`/`set_perm_nofollow` take *this* backend's `FilePermissions`, so they can't be borrowed
// from the `unsupported` PAL (whose signatures name its own `FilePermissions`). The memfs has no
// mutable perms, so both are `Unsupported`.
pub fn set_perm(_path: &Path, _perm: FilePermissions) -> io::Result<()> {
    unsupported()
}

pub fn set_perm_nofollow(_path: &Path, _perm: FilePermissions) -> io::Result<()> {
    unsupported()
}

// svm-posix `struct stat` layout (`stat` op) and mode bits (see svm-posix `S_IF*`).
const S_IFMT: i64 = 0o170000;
const S_IFDIR: i64 = 0o040000;
// `open` flags (svm-posix `O_*`).
const O_WRONLY: i64 = 1;
const O_RDWR: i64 = 2;
const O_CREAT: i64 = 0o100;
const O_TRUNC: i64 = 0o1000;
const O_APPEND: i64 = 0o2000;
// `lseek` whence (svm-posix `SEEK_*`).
const SEEK_SET: i64 = 0;
const SEEK_CUR: i64 = 1;
const SEEK_END: i64 = 2;
// The errno values (`-code`) the memfs ops return, mapped to the `ErrorKind`s programs actually match.
const ENOENT: i64 = -2;
const EEXIST: i64 = -17;
const ENOTDIR: i64 = -20;
const EINVAL: i64 = -22;
const ERANGE: i64 = -34;
const ENOTEMPTY: i64 = -39;

/// Turn a negative errno returned by a host op into an `io::Error`, giving the common cases a precise
/// `ErrorKind` (the svm `io/error` leaf otherwise decodes everything to `Uncategorized`). `EEXIST`
/// maps to `AlreadyExists` so `create_dir_all` correctly treats an existing directory as success.
fn err(code: i64) -> io::Error {
    match code {
        ENOENT => io::ErrorKind::NotFound.into(),
        EEXIST => io::ErrorKind::AlreadyExists.into(),
        ENOTDIR => io::ErrorKind::NotADirectory.into(),
        ENOTEMPTY => io::ErrorKind::DirectoryNotEmpty.into(),
        EINVAL => io::ErrorKind::InvalidInput.into(),
        _ => io::Error::from_raw_os_error((-code) as i32),
    }
}

/// A path as the `(ptr, len)` UTF-8 byte slice the host ops expect. svm has no OS-specific `OsStr`
/// encoding, so the encoded bytes are the path verbatim.
fn path_bytes(path: &Path) -> &[u8] {
    path.as_os_str().as_encoded_bytes()
}

pub struct File {
    fd: i32,
}

/// Either a directory or a regular file of a known size — the two things a memfs `stat` distinguishes.
#[derive(Clone)]
pub enum FileAttr {
    Dir,
    File { size: u64 },
}

pub struct ReadDir {
    dir: i32,
    root: PathBuf,
}

pub struct DirEntry {
    path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FilePermissions;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FileType {
    is_dir: bool,
}

impl FileAttr {
    pub fn size(&self) -> u64 {
        match self {
            FileAttr::File { size } => *size,
            FileAttr::Dir => 0,
        }
    }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions
    }

    pub fn file_type(&self) -> FileType {
        FileType { is_dir: matches!(self, FileAttr::Dir) }
    }

    pub fn modified(&self) -> io::Result<SystemTime> {
        unsupported()
    }

    pub fn accessed(&self) -> io::Result<SystemTime> {
        unsupported()
    }

    pub fn created(&self) -> io::Result<SystemTime> {
        unsupported()
    }
}

impl FilePermissions {
    pub fn readonly(&self) -> bool {
        // The memfs has no read-only bit; every entry is writable.
        false
    }

    pub fn set_readonly(&mut self, _readonly: bool) {
        // No perms to set; a no-op keeps `fs::copy` (which round-trips permissions) working.
    }
}

impl FileType {
    pub fn is_dir(&self) -> bool {
        self.is_dir
    }

    pub fn is_file(&self) -> bool {
        !self.is_dir
    }

    pub fn is_symlink(&self) -> bool {
        // The memfs has no symlinks — entries are files or directories.
        false
    }
}

impl fmt::Debug for ReadDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ReadDir").field("root", &self.root).finish()
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;

    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        // `readdir` writes a NUL-terminated name and returns its length; `-ERANGE` means the buffer was
        // too small (and the stream did *not* advance), so grow and retry. `0` is end-of-stream.
        let mut cap = 256usize;
        loop {
            let mut buf = vec![0u8; cap];
            let n = host::readdir(self.dir as i64, buf.as_mut_ptr(), cap as i64);
            if n == 0 {
                return None;
            }
            if n == ERANGE {
                cap *= 2;
                continue;
            }
            if n < 0 {
                return Some(Err(err(n)));
            }
            buf.truncate(n as usize);
            // SAFETY: svm has no OS-specific `OsStr` encoding; the name bytes are the path component.
            let name = unsafe { OsStr::from_encoded_bytes_unchecked(&buf) };
            return Some(Ok(DirEntry { path: self.root.join(name) }));
        }
    }
}

impl Drop for ReadDir {
    fn drop(&mut self) {
        let _ = host::closedir(self.dir as i64);
    }
}

impl DirEntry {
    pub fn path(&self) -> PathBuf {
        self.path.clone()
    }

    pub fn file_name(&self) -> OsString {
        self.path.file_name().unwrap_or_default().into()
    }

    pub fn metadata(&self) -> io::Result<FileAttr> {
        stat(&self.path)
    }

    pub fn file_type(&self) -> io::Result<FileType> {
        Ok(self.metadata()?.file_type())
    }
}

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
        }
    }

    pub fn read(&mut self, read: bool) {
        self.read = read;
    }
    pub fn write(&mut self, write: bool) {
        self.write = write;
    }
    pub fn append(&mut self, append: bool) {
        self.append = append;
    }
    pub fn truncate(&mut self, truncate: bool) {
        self.truncate = truncate;
    }
    pub fn create(&mut self, create: bool) {
        self.create = create;
    }
    pub fn create_new(&mut self, create_new: bool) {
        self.create_new = create_new;
    }
}

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        if !host::have_posix() {
            return Err(unsupported_err());
        }
        let bytes = path_bytes(path);
        // The `open` op auto-creates with `O_CREAT` and never rejects an existing target, so enforce
        // `create_new`'s "must not exist" here (it maps to `O_EXCL`, which the memfs op does not model).
        if opts.create_new {
            let mut sb = [0u8; 16];
            let s = host::stat(bytes.as_ptr(), bytes.len() as i64, sb.as_mut_ptr());
            if s == 0 {
                return Err(io::const_error!(io::ErrorKind::AlreadyExists, "file exists"));
            }
        }
        let write = opts.write || opts.append;
        let mut flags = if opts.read && write {
            O_RDWR
        } else if write {
            O_WRONLY
        } else {
            0 // O_RDONLY
        };
        if opts.create || opts.create_new {
            flags |= O_CREAT;
        }
        if opts.truncate {
            flags |= O_TRUNC;
        }
        if opts.append {
            flags |= O_APPEND;
        }
        let fd = host::open(bytes.as_ptr(), bytes.len() as i64, flags);
        if fd < 0 {
            return Err(err(fd));
        }
        Ok(File { fd: fd as i32 })
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        // No `fstat` op; a `File` is only ever a regular file (directories arrive via `opendir`), so
        // recover its size by seeking to the end and restoring the offset.
        let cur = host::lseek(self.fd as i64, 0, SEEK_CUR);
        if cur < 0 {
            return Err(err(cur));
        }
        let end = host::lseek(self.fd as i64, 0, SEEK_END);
        if end < 0 {
            return Err(err(end));
        }
        let _ = host::lseek(self.fd as i64, cur, SEEK_SET);
        Ok(FileAttr::File { size: end as u64 })
    }

    pub fn fsync(&self) -> io::Result<()> {
        // Writes hit the personality store immediately; nothing to flush.
        Ok(())
    }

    pub fn datasync(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn lock(&self) -> io::Result<()> {
        // Single-threaded sandbox: advisory locks are uncontended, so acquisition always succeeds.
        Ok(())
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        Ok(())
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        Ok(())
    }

    pub fn unlock(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        // No `ftruncate` op (only `O_TRUNC` at open time).
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let n = host::read_fd(self.fd as i64, buf.as_mut_ptr(), buf.len() as i64);
        if n < 0 { Err(err(n)) } else { Ok(n as usize) }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        for b in bufs {
            if !b.is_empty() {
                return self.read(b);
            }
        }
        Ok(0)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_, u8>) -> io::Result<()> {
        crate::io::default_read_buf(|b| self.read(b), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let n = host::write_fd(self.fd as i64, buf.as_ptr(), buf.len() as i64);
        if n < 0 { Err(err(n)) } else { Ok(n as usize) }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        for b in bufs {
            if !b.is_empty() {
                return self.write(b);
            }
        }
        Ok(0)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(())
    }

    pub fn seek(&self, pos: SeekFrom) -> io::Result<u64> {
        let (whence, off) = match pos {
            SeekFrom::Start(n) => (SEEK_SET, n as i64),
            SeekFrom::Current(n) => (SEEK_CUR, n),
            SeekFrom::End(n) => (SEEK_END, n),
        };
        let r = host::lseek(self.fd as i64, off, whence);
        if r < 0 { Err(err(r)) } else { Ok(r as u64) }
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        Some(self.file_attr().map(|a| a.size()))
    }

    pub fn tell(&self) -> io::Result<u64> {
        let r = host::lseek(self.fd as i64, 0, SEEK_CUR);
        if r < 0 { Err(err(r)) } else { Ok(r as u64) }
    }

    pub fn duplicate(&self) -> io::Result<File> {
        // `try_clone`: a second fd onto the same open file, via the personality's `dup`.
        let fd = host::dup(self.fd as i64);
        if fd < 0 { Err(err(fd)) } else { Ok(File { fd: fd as i32 }) }
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        // No mutable perms; a no-op keeps `fs::copy` (which sets the destination's perms) working.
        Ok(())
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        unsupported()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        let _ = host::close(self.fd as i64);
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.fd).finish()
    }
}

pub fn readdir(path: &Path) -> io::Result<ReadDir> {
    if !host::have_posix() {
        return Err(unsupported_err());
    }
    let bytes = path_bytes(path);
    let dir = host::opendir(bytes.as_ptr(), bytes.len() as i64);
    if dir < 0 {
        return Err(err(dir));
    }
    Ok(ReadDir { dir: dir as i32, root: path.to_path_buf() })
}

pub fn unlink(path: &Path) -> io::Result<()> {
    if !host::have_posix() {
        return Err(unsupported_err());
    }
    let bytes = path_bytes(path);
    let r = host::unlink(bytes.as_ptr(), bytes.len() as i64);
    if r < 0 { Err(err(r)) } else { Ok(()) }
}

pub fn stat(path: &Path) -> io::Result<FileAttr> {
    if !host::have_posix() {
        return Err(unsupported_err());
    }
    let bytes = path_bytes(path);
    let mut sb = [0u8; 16];
    let r = host::stat(bytes.as_ptr(), bytes.len() as i64, sb.as_mut_ptr());
    if r < 0 {
        return Err(err(r));
    }
    let mode = i64::from_le_bytes(sb[0..8].try_into().unwrap());
    let size = i64::from_le_bytes(sb[8..16].try_into().unwrap());
    if mode & S_IFMT == S_IFDIR {
        Ok(FileAttr::Dir)
    } else {
        Ok(FileAttr::File { size: size as u64 })
    }
}

pub fn lstat(path: &Path) -> io::Result<FileAttr> {
    // No symlinks on the memfs, so `lstat` and `stat` coincide.
    stat(path)
}

pub fn rename(old: &Path, new: &Path) -> io::Result<()> {
    if !host::have_posix() {
        return Err(unsupported_err());
    }
    let ob = path_bytes(old);
    let nb = path_bytes(new);
    let r = host::rename(ob.as_ptr(), ob.len() as i64, nb.as_ptr(), nb.len() as i64);
    if r < 0 { Err(err(r)) } else { Ok(()) }
}

pub fn rmdir(path: &Path) -> io::Result<()> {
    if !host::have_posix() {
        return Err(unsupported_err());
    }
    let b = path_bytes(path);
    let r = host::rmdir(b.as_ptr(), b.len() as i64);
    if r < 0 { Err(err(r)) } else { Ok(()) }
}

/// The `std::fs::DirBuilder` PAL: a single `mkdir` over the memfs directory op. `create_dir_all` is
/// driven by the std layer (it calls `mkdir` per component and tolerates `AlreadyExists`).
#[derive(Debug)]
pub struct DirBuilder;

impl DirBuilder {
    pub fn new() -> DirBuilder {
        DirBuilder
    }

    pub fn mkdir(&self, p: &Path) -> io::Result<()> {
        if !host::have_posix() {
            return Err(unsupported_err());
        }
        let b = path_bytes(p);
        let r = host::mkdir(b.as_ptr(), b.len() as i64, 0o777);
        if r < 0 { Err(err(r)) } else { Ok(()) }
    }
}
