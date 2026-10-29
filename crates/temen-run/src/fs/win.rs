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

use cap_std::fs::{Dir, OpenOptions, OpenOptionsExt};
use std::ffi::{OsStr, OsString};
use std::io;
use std::mem::{offset_of, size_of};
use std::os::windows::ffi::{OsStrExt, OsStringExt};
use std::os::windows::fs::MetadataExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use windows_sys::Wdk::Foundation::OBJECT_ATTRIBUTES;
use windows_sys::Wdk::Storage::FileSystem::{
    FileRenameInformation, NtCreateFile, NtSetInformationFile, FILE_CREATE, FILE_DIRECTORY_FILE,
    FILE_RENAME_INFORMATION, FILE_SYNCHRONOUS_IO_NONALERT,
};
use windows_sys::Win32::Foundation::{
    RtlNtStatusToDosError, ERROR_ACCESS_DENIED, ERROR_DIRECTORY, ERROR_NO_MORE_FILES, NTSTATUS,
    UNICODE_STRING,
};
use windows_sys::Win32::Storage::FileSystem::{
    FileDispositionInfo, FileFullDirectoryInfo, FileFullDirectoryRestartInfo,
    GetFileInformationByHandleEx, SetFileInformationByHandle, DELETE, FILE_ATTRIBUTE_DIRECTORY,
    FILE_ATTRIBUTE_NORMAL, FILE_DISPOSITION_INFO, FILE_FLAG_BACKUP_SEMANTICS,
    FILE_FLAG_OPEN_REPARSE_POINT, FILE_FULL_DIR_INFO, FILE_LIST_DIRECTORY, FILE_READ_ATTRIBUTES,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE, SYNCHRONIZE,
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

/// Rename `from` beneath `dir` to `to` beneath it, replacing a file there, as `MoveFileExW` does.
pub(super) fn rename(dir: &Dir, from: &str, to: &str) -> io::Result<()> {
    let file = open_itself(dir, from)?;
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
        (*info).Anonymous.ReplaceIfExists = 1;
        (*info).RootDirectory = parent.as_raw_handle();
        (*info).FileNameLength = (name.len() * 2) as u32;
        std::ptr::copy_nonoverlapping(
            name.as_ptr(),
            std::ptr::addr_of_mut!((*info).FileName).cast::<u16>(),
            name.len(),
        );
    }
    let mut status = io_status();
    // SAFETY: `file` is open with DELETE access, and `info` is a FILE_RENAME_INFORMATION of `len`
    // bytes whose directory, `parent`, is open for the call.
    nt(unsafe {
        NtSetInformationFile(
            file.as_raw_handle(),
            &mut status,
            info.cast(),
            len as u32,
            FileRenameInformation,
        )
    })
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
fn open_itself(dir: &Dir, path: &str) -> io::Result<std::fs::File> {
    let mut options = OpenOptions::new();
    options
        .access_mode(DELETE | FILE_READ_ATTRIBUTES)
        .custom_flags(FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT);
    Ok(dir.open_with(path, &options)?.into_std())
}

/// Have what `file` is open on deleted when it closes.
fn delete(file: &std::fs::File) -> io::Result<()> {
    let info = FILE_DISPOSITION_INFO { DeleteFile: 1 };
    // SAFETY: `file` is open with DELETE access, and `info` is the FILE_DISPOSITION_INFO its class
    // takes, of the size given.
    let done = unsafe {
        SetFileInformationByHandle(
            file.as_raw_handle(),
            FileDispositionInfo,
            (&info as *const FILE_DISPOSITION_INFO).cast(),
            size_of::<FILE_DISPOSITION_INFO>() as u32,
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
