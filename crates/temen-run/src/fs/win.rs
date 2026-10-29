//! #2643 — `host_fs`'s mkdir, rmdir, remove, rename and listing on Windows, each on a handle cap-std
//! opened beneath the root.
//!
//! cap-std resolves a path on Windows as it does elsewhere, a component at a time from a handle, and
//! its open and stat act on what it resolved. Its make, remove, rename and list do not: they rebuild
//! a path from the parent's handle and hand it to the OS, which resolves it again from the volume's
//! root. cap-std keeps the parent from being renamed meanwhile, but not from becoming a junction in
//! place: `mklink /J` makes a directory, then turns that directory into the link. A directory the
//! host turns into a junction between the two lookups redirects the op out of the root. So each op
//! here names what it acts on by one component relative to a handle, or acts on the handle itself.
//!
//! #2510 — `replace`'s steps too. It writes a temporary file beside the target and renames that
//! file's handle over the target, rather than calling `ReplaceFileW`, which takes paths. The
//! temporary file takes the target's DACL when it is made, and its attributes and creation time
//! once written. It is held without sharing deletes until it is renamed: that is how a `replace`
//! tells another's temporary file from one a crash left, which it can open to remove.

use super::refuse;
use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io;
use std::mem::{offset_of, size_of};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::os::windows::fs::MetadataExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use temen_ir::errno::{EBUSY, EISDIR, EMLINK};
use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows_sys::Wdk::Storage::FileSystem::{
    FileRenameInformation, FileRenameInformationEx, NtCreateFile, NtSetInformationFile,
    FILE_CREATE, FILE_DIRECTORY_FILE, FILE_RENAME_INFORMATION, FILE_RENAME_POSIX_SEMANTICS,
    FILE_RENAME_REPLACE_IF_EXISTS, FILE_SYNCHRONOUS_IO_NONALERT,
};
use windows_sys::Win32::Foundation::{
    LocalFree, RtlNtStatusToDosError, ERROR_ACCESS_DENIED, ERROR_DIRECTORY, ERROR_NO_MORE_FILES,
    ERROR_SHARING_VIOLATION, ERROR_SUCCESS, GENERIC_READ, GENERIC_WRITE, NTSTATUS,
    STATUS_INVALID_INFO_CLASS, STATUS_INVALID_PARAMETER, STATUS_NOT_SUPPORTED, UNICODE_STRING,
};
use windows_sys::Win32::Security::Authorization::{
    GetSecurityInfo, SetSecurityInfo, SE_FILE_OBJECT,
};
use windows_sys::Win32::Security::{
    GetSecurityDescriptorControl, ACL, DACL_SECURITY_INFORMATION,
    PROTECTED_DACL_SECURITY_INFORMATION, PSECURITY_DESCRIPTOR, SE_DACL_PROTECTED,
    UNPROTECTED_DACL_SECURITY_INFORMATION,
};
use windows_sys::Win32::Storage::FileSystem::{
    FileBasicInfo, FileDispositionInfo, FileFullDirectoryInfo, FileFullDirectoryRestartInfo,
    GetFileInformationByHandle, GetFileInformationByHandleEx, SetFileInformationByHandle,
    BY_HANDLE_FILE_INFORMATION, DELETE, FILE_ATTRIBUTE_ARCHIVE, FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_HIDDEN, FILE_ATTRIBUTE_NORMAL, FILE_ATTRIBUTE_NOT_CONTENT_INDEXED,
    FILE_ATTRIBUTE_OFFLINE, FILE_ATTRIBUTE_SYSTEM, FILE_ATTRIBUTE_TEMPORARY, FILE_BASIC_INFO,
    FILE_DISPOSITION_INFO, FILE_FLAG_BACKUP_SEMANTICS, FILE_FLAG_OPEN_REPARSE_POINT,
    FILE_FULL_DIR_INFO, FILE_INFO_BY_HANDLE_CLASS, FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, READ_CONTROL, SYNCHRONIZE, WRITE_DAC,
};
use windows_sys::Win32::System::Kernel::OBJ_CASE_INSENSITIVE;
use windows_sys::Win32::System::IO::IO_STATUS_BLOCK;

/// Make the directory `path` beneath `dir`, by name in the directory its parent resolves to.
pub(super) fn mkdir(dir: &Dir, path: &str) -> io::Result<()> {
    let (parent, name) = parent_of(dir, path)?;
    let mut wide: Vec<u16> = OsStr::new(name).encode_wide().collect();
    let bytes =
        u16::try_from(wide.len() * 2).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let name = UNICODE_STRING {
        Length: bytes,
        MaximumLength: bytes,
        Buffer: wide.as_mut_ptr(),
    };
    let attributes = OBJECT_ATTRIBUTES {
        Length: size_of::<OBJECT_ATTRIBUTES>() as u32,
        RootDirectory: parent.as_raw_handle(),
        ObjectName: &name,
        Attributes: OBJ_CASE_INSENSITIVE as u32,
        SecurityDescriptor: std::ptr::null(),
        SecurityQualityOfService: std::ptr::null(),
    };
    let mut handle = std::ptr::null_mut();
    let mut status = io_status();
    // SAFETY: each pointer is to a local of the type NtCreateFile takes, live for the call:
    // `attributes` names `name`, whose buffer is `wide`, relative to `parent`, an open directory.
    nt(unsafe {
        NtCreateFile(
            &mut handle,
            FILE_LIST_DIRECTORY | SYNCHRONIZE,
            &attributes,
            &mut status,
            std::ptr::null(),
            FILE_ATTRIBUTE_NORMAL,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_CREATE,
            FILE_DIRECTORY_FILE | FILE_SYNCHRONOUS_IO_NONALERT,
            std::ptr::null(),
            0,
        )
    })?;
    // SAFETY: NtCreateFile succeeded, so `handle` is an open handle no one else owns.
    drop(unsafe { OwnedHandle::from_raw_handle(handle) });
    Ok(())
}

/// Remove the file `path` beneath `dir`, or a link to a file, as `DeleteFileW` does: a directory,
/// or a link to one, is refused.
pub(super) fn unlink(dir: &Dir, path: &str) -> io::Result<()> {
    let file = open_itself(dir, path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_DIRECTORY != 0 {
        return Err(io::Error::from_raw_os_error(ERROR_ACCESS_DENIED as i32));
    }
    delete(&file)
}

/// Remove the empty directory `path` beneath `dir`, or a link to a directory, as
/// `RemoveDirectoryW` does.
pub(super) fn rmdir(dir: &Dir, path: &str) -> io::Result<()> {
    let file = open_itself(dir, path)?;
    if file.metadata()?.file_attributes() & FILE_ATTRIBUTE_DIRECTORY == 0 {
        return Err(io::Error::from_raw_os_error(ERROR_DIRECTORY as i32));
    }
    delete(&file)
}

/// Rename `from` beneath `dir` to `to` beneath it, replacing a file there.
pub(super) fn rename(dir: &Dir, from: &str, to: &str) -> io::Result<()> {
    rename_handle(&open_itself(dir, from)?, dir, to)
}

/// Rename what `file` is open on to `to` beneath `dir`, replacing a file there. With POSIX
/// semantics where the volume has them, a file open elsewhere is replaced, as on Unix; where it
/// doesn't, as `MoveFileExW` does.
fn rename_handle(file: &File, dir: &Dir, to: &str) -> io::Result<()> {
    let (parent, name) = parent_of(dir, to)?;
    let name: Vec<u16> = OsStr::new(name).encode_wide().collect();
    // A FILE_RENAME_INFORMATION with the name in place of its one-element `FileName`, in u64s for
    // its alignment. The bytes past the name stay zero.
    let len = size_of::<FILE_RENAME_INFORMATION>() + name.len() * 2;
    let mut buf = vec![0u64; len.div_ceil(8)];
    let info = buf.as_mut_ptr().cast::<FILE_RENAME_INFORMATION>();
    // SAFETY: `buf` is `len` bytes, 8-aligned: the structure, and from `FileName` on the name's
    // `name.len()` u16s. Nothing else refers to it.
    unsafe {
        (*info).Anonymous.Flags = FILE_RENAME_REPLACE_IF_EXISTS | FILE_RENAME_POSIX_SEMANTICS;
        (*info).RootDirectory = parent.as_raw_handle();
        (*info).FileNameLength = (name.len() * 2) as u32;
        std::ptr::copy_nonoverlapping(
            name.as_ptr(),
            std::ptr::addr_of_mut!((*info).FileName).cast::<u16>(),
            name.len(),
        );
    }
    let set = move |class| {
        let mut status = io_status();
        // SAFETY: `file` is open with DELETE access, and `info` is a FILE_RENAME_INFORMATION of
        // `len` bytes whose directory, `parent`, is open for the call.
        unsafe {
            NtSetInformationFile(
                file.as_raw_handle(),
                &mut status,
                info.cast(),
                len as u32,
                class,
            )
        }
    };
    match set(FileRenameInformationEx) {
        // Windows before 10 1607, or a volume without POSIX semantics (FAT): the classic rename,
        // whose `ReplaceIfExists` is the flags' first byte.
        STATUS_INVALID_INFO_CLASS | STATUS_INVALID_PARAMETER | STATUS_NOT_SUPPORTED => {
            // SAFETY: as above; nothing else refers to `info`.
            unsafe { (*info).Anonymous.Flags = FILE_RENAME_REPLACE_IF_EXISTS };
            nt(set(FileRenameInformation))
        }
        status => nt(status),
    }
}

/// The names in the directory `path` beneath `dir`, read a piece at a time as they are asked for.
pub(super) fn names(dir: &Dir, path: &str) -> io::Result<Names> {
    Ok(Names {
        dir: dir.open_dir(path)?,
        buf: vec![0; 1024],
        next: None,
        restart: true,
        done: false,
    })
}

/// A listing from [`names`]: `FILE_FULL_DIR_INFO` records, a bufferful at a time, from its handle.
pub(super) struct Names {
    dir: Dir,
    /// The records, in u64s for their alignment.
    buf: Vec<u64>,
    /// Where in `buf` the next record starts, while one is left.
    next: Option<usize>,
    /// Whether the next read starts the scan, as the first does.
    restart: bool,
    done: bool,
}

impl Iterator for Names {
    type Item = io::Result<OsString>;

    fn next(&mut self) -> Option<io::Result<OsString>> {
        loop {
            if let Some(at) = self.next {
                let record = record(&self.buf, at);
                let Some((step, wide)) = record else {
                    self.done = true;
                    self.next = None;
                    return Some(Err(io::ErrorKind::InvalidData.into()));
                };
                self.next = (step != 0).then_some(at + step);
                if wide != [u16::from(b'.')] && wide != [u16::from(b'.'); 2] {
                    return Some(Ok(OsString::from_wide(&wide)));
                }
                continue;
            }
            if self.done {
                return None;
            }
            let class = match std::mem::take(&mut self.restart) {
                true => FileFullDirectoryRestartInfo,
                false => FileFullDirectoryInfo,
            };
            // SAFETY: `buf` is live and 8-aligned, of the size given; the call fills it with records.
            let read = unsafe {
                GetFileInformationByHandleEx(
                    self.dir.as_raw_handle(),
                    class,
                    self.buf.as_mut_ptr().cast(),
                    (self.buf.len() * 8) as u32,
                )
            };
            if read == 0 {
                self.done = true;
                let e = io::Error::last_os_error();
                return (e.raw_os_error() != Some(ERROR_NO_MORE_FILES as i32)).then_some(Err(e));
            }
            self.next = Some(0);
        }
    }
}

/// The `FILE_FULL_DIR_INFO` record at byte `at` of `buf`: the offset of the next one (0 if none),
/// and its name. `None` if the record runs past the buffer.
fn record(buf: &[u64], at: usize) -> Option<(usize, Vec<u16>)> {
    // SAFETY: `buf`'s memory is `buf.len() * 8` initialized bytes, at a looser alignment.
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), buf.len() * 8) };
    let field = |off: usize| {
        let b = bytes.get(at + off..at + off + 4)?;
        Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
    };
    let step = field(offset_of!(FILE_FULL_DIR_INFO, NextEntryOffset))?;
    let len = field(offset_of!(FILE_FULL_DIR_INFO, FileNameLength))?;
    let from = at + offset_of!(FILE_FULL_DIR_INFO, FileName);
    let name = bytes.get(from..from + len)?;
    Some((
        step,
        name.chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect(),
    ))
}

/// The file `name` in `dir`, open to read its protection from, or `None` if there is none.
/// Refused: a directory, a file with other hard links (R1), and a file the process may not write
/// in place: one marked read-only or whose DACL denies it writing, or (`-EBUSY`) one open
/// elsewhere without sharing writes.
pub(super) fn target(dir: &Dir, name: &str) -> io::Result<Option<File>> {
    // The file itself, not what a link there leads to: `replace_target` followed the links.
    let mut options = OpenOptions::new();
    options
        .access_mode(READ_CONTROL | FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
    let file = match dir.open_with(name, &options) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        opened => opened?.into_std(),
    };
    // SAFETY: BY_HANDLE_FILE_INFORMATION is plain data, for which all zeros is a value.
    let mut info: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: `file` is open, and `info` is the structure the call fills.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &mut info) } == 0 {
        return Err(io::Error::last_os_error());
    }
    if info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY != 0 {
        return Err(refuse(EISDIR));
    }
    if info.nNumberOfLinks > 1 {
        return Err(refuse(EMLINK));
    }
    // Writing it in place would fail, so replacing it does too.
    dir.open_with(name, OpenOptions::new().write(true))
        .map_err(busy)?;
    Ok(Some(file))
}

/// `replace`'s temporary file, in the target's directory.
pub(super) struct Temp<'a> {
    dir: &'a Dir,
    pub(super) file: File,
    /// Whether to leave it as it is when dropped: renamed over the target, or left by a crash.
    kept: bool,
}

impl<'a> Temp<'a> {
    /// A new temporary file `name` in `dir`. It takes an existing `target`'s DACL before it holds
    /// a byte, so its new bytes are never open to more than the target's are; a new file has the
    /// directory's default. It is held without sharing deletes until it is renamed.
    pub(super) fn create(dir: &'a Dir, name: &str, target: Option<&File>) -> io::Result<Temp<'a>> {
        let mut options = OpenOptions::new();
        options
            .read(true)
            .write(true)
            .create_new(true)
            .access_mode(GENERIC_READ | GENERIC_WRITE | DELETE | WRITE_DAC)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE);
        let file = match dir.open_with(name, &options) {
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {
                clear(dir, name)?;
                // Made in between by another `replace`, whose it is.
                dir.open_with(name, &options).map_err(|e| match e.kind() {
                    io::ErrorKind::AlreadyExists => refuse(EBUSY),
                    _ => e,
                })?
            }
            made => made?,
        };
        let temp = Temp {
            dir,
            file: file.into_std(),
            kept: false,
        };
        if let Some(target) = target {
            copy_dacl(target, &temp.file)?;
        }
        Ok(temp)
    }

    /// Give the file `target`'s attributes (hidden, system, archive, not indexed and the like) and
    /// creation time, as `ReplaceFileW` keeps them. It has had the target's DACL since it was made.
    pub(super) fn protect(&self, target: &File) -> io::Result<()> {
        // SAFETY: FILE_BASIC_INFO is plain data, for which all zeros is a value.
        let mut from: FILE_BASIC_INFO = unsafe { std::mem::zeroed() };
        // SAFETY: `target` is open with FILE_READ_ATTRIBUTES, and `from` is the structure its
        // class fills, of the size given.
        let read = unsafe {
            GetFileInformationByHandleEx(
                target.as_raw_handle(),
                FileBasicInfo,
                (&mut from as *mut FILE_BASIC_INFO).cast(),
                size_of::<FILE_BASIC_INFO>() as u32,
            )
        };
        if read == 0 {
            return Err(io::Error::last_os_error());
        }
        // The attributes a file can be given. A target is never read-only here: `target` refused it.
        let settable = FILE_ATTRIBUTE_ARCHIVE
            | FILE_ATTRIBUTE_HIDDEN
            | FILE_ATTRIBUTE_NORMAL
            | FILE_ATTRIBUTE_NOT_CONTENT_INDEXED
            | FILE_ATTRIBUTE_OFFLINE
            | FILE_ATTRIBUTE_SYSTEM
            | FILE_ATTRIBUTE_TEMPORARY;
        let attributes = match from.FileAttributes & settable {
            0 => FILE_ATTRIBUTE_NORMAL,
            some => some,
        };
        // A time of 0 leaves that time as it is.
        let info = FILE_BASIC_INFO {
            CreationTime: from.CreationTime,
            LastAccessTime: 0,
            LastWriteTime: 0,
            ChangeTime: 0,
            FileAttributes: attributes,
        };
        // SAFETY: `file` is open for writing, and FILE_BASIC_INFO is its class's structure.
        unsafe { set_info(&self.file, FileBasicInfo, &info) }
    }

    /// Make the file's bytes and protection durable.
    pub(super) fn sync(&mut self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Rename the file over `target`, its directory's file of that name.
    pub(super) fn rename_over(&mut self, target: &str) -> io::Result<()> {
        rename_handle(&self.file, self.dir, target)?;
        self.kept = true;
        Ok(())
    }

    /// Make the rename durable. Windows syncs no directory: NTFS journals the rename, and flushing
    /// the file it renamed flushes the journal through it.
    pub(super) fn sync_dir(&self) -> io::Result<()> {
        self.file.sync_all()
    }

    /// Leave the file as it is, as a crash does.
    pub(super) fn keep(&mut self) {
        self.kept = true;
    }
}

impl Drop for Temp<'_> {
    /// Remove the file after an error before the rename.
    fn drop(&mut self) {
        if !self.kept {
            let _ = delete(&self.file);
        }
    }
}

/// Remove the temporary file `name` in `dir` that a crash left: no `replace` holds it, so it can
/// be opened to be removed. `-EBUSY` if another `replace` holds it, without sharing deletes.
fn clear(dir: &Dir, name: &str) -> io::Result<()> {
    match open_itself(dir, name) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(busy(e)),
        Ok(file) => delete(&file),
    }
}

/// A sharing violation, a file open elsewhere that doesn't share what was asked: `-EBUSY`.
fn busy(e: io::Error) -> io::Error {
    match e.raw_os_error() {
        Some(code) if code == ERROR_SHARING_VIOLATION as i32 => refuse(EBUSY),
        _ => e,
    }
}

/// Give `to` the DACL of `from`, inherited or protected as `from`'s is.
fn copy_dacl(from: &File, to: &File) -> io::Result<()> {
    let mut dacl: *mut ACL = std::ptr::null_mut();
    let mut descriptor: PSECURITY_DESCRIPTOR = std::ptr::null_mut();
    // SAFETY: `from` is open with READ_CONTROL. The call allocates `descriptor` and points `dacl`
    // into it.
    let got = unsafe {
        GetSecurityInfo(
            from.as_raw_handle(),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            &mut dacl,
            std::ptr::null_mut(),
            &mut descriptor,
        )
    };
    if got != ERROR_SUCCESS {
        return Err(io::Error::from_raw_os_error(got as i32));
    }
    let (mut control, mut revision) = (0u16, 0u32);
    // SAFETY: `descriptor` is the one the call above allocated.
    let read = unsafe { GetSecurityDescriptorControl(descriptor, &mut control, &mut revision) };
    let done = if read == 0 {
        Err(io::Error::last_os_error())
    } else {
        let inherit = match control & SE_DACL_PROTECTED {
            0 => UNPROTECTED_DACL_SECURITY_INFORMATION,
            _ => PROTECTED_DACL_SECURITY_INFORMATION,
        };
        // SAFETY: `to` is open with WRITE_DAC, and `dacl` points into `descriptor`, live until it
        // is freed below.
        match unsafe {
            SetSecurityInfo(
                to.as_raw_handle(),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | inherit,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                dacl,
                std::ptr::null(),
            )
        } {
            ERROR_SUCCESS => Ok(()),
            e => Err(io::Error::from_raw_os_error(e as i32)),
        }
    };
    // SAFETY: GetSecurityInfo allocated `descriptor` to be freed with LocalFree, and nothing uses
    // it after.
    unsafe { LocalFree(descriptor) };
    done
}

/// `path`'s parent resolved beneath `dir`, open, and `path`'s last component, which is a name: never
/// empty or `.` (`read_path` refuses `..`).
fn parent_of<'p>(dir: &Dir, path: &'p str) -> io::Result<(Dir, &'p str)> {
    let (parent, name) = match path.rfind(['/', '\\']) {
        Some(at) => (&path[..at], &path[at + 1..]),
        None => ("", path),
    };
    if name.is_empty() || name == "." {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    let parent = if parent.is_empty() { "." } else { parent };
    Ok((dir.open_dir(parent)?, name))
}

/// `path` beneath `dir`, opened as itself (a link, not what it links to) to be removed or renamed.
fn open_itself(dir: &Dir, path: &str) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options
        .access_mode(DELETE | FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
    Ok(dir.open_with(path, &options)?.into_std())
}

/// Have what `file` is open on deleted when it closes.
fn delete(file: &File) -> io::Result<()> {
    // SAFETY: `file` is open with DELETE access, and FILE_DISPOSITION_INFO is its class's structure.
    unsafe {
        set_info(
            file,
            FileDispositionInfo,
            &FILE_DISPOSITION_INFO { DeleteFile: 1 },
        )
    }
}

/// Set `file`'s information of `class` to `info`.
///
/// # Safety
///
/// `T` is the structure `class` takes, and `file` is open with the access it needs.
unsafe fn set_info<T>(file: &File, class: FILE_INFO_BY_HANDLE_CLASS, info: &T) -> io::Result<()> {
    // SAFETY: the caller's: `info` is the structure `class` takes, of the size given.
    let done = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            class,
            (info as *const T).cast(),
            size_of::<T>() as u32,
        )
    };
    match done {
        0 => Err(io::Error::last_os_error()),
        _ => Ok(()),
    }
}

/// An IO_STATUS_BLOCK for a call to fill.
fn io_status() -> IO_STATUS_BLOCK {
    // SAFETY: IO_STATUS_BLOCK is plain data, for which all zeros is a value.
    unsafe { std::mem::zeroed() }
}

/// An NTSTATUS as a result, its error as the Win32 code `std` maps to a kind.
fn nt(status: NTSTATUS) -> io::Result<()> {
    if status >= 0 {
        return Ok(());
    }
    // SAFETY: RtlNtStatusToDosError reads nothing but the status it is passed.
    Err(io::Error::from_raw_os_error(
        unsafe { RtlNtStatusToDosError(status) } as i32,
    ))
}
