//! temen anonymous pipes: the `sys::pipe::Pipe` backing `std::process`'s child stdio and `std::io::pipe`.
//! A pipe is a core pipe (FORK.md §8.6) whose two ends are descriptors of this process (temen-posix
//! `OP_PIPE_ADOPT`), read/written through the same `OP_READ`/`OP_WRITE` file ops the fs overlay uses and
//! closed on drop (`OP_CLOSE`). A read of an empty pipe waits while a writer is left and reads `0` (EOF)
//! once none is; a write to a full pipe waits while a reader is. A spawned child's descriptor for the
//! pipe is an end of the same pipe, so a parent can stream into a live child and read it to EOF.
#![deny(unsafe_op_in_unsafe_fn)]
use crate::fmt;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut};
use crate::sys::pal::host;

pub struct Pipe {
    fd: i32,
}

impl Pipe {
    pub(crate) fn fd(&self) -> i32 {
        self.fd
    }
}

/// Create a pipe: `(read_end, write_end)`. Returns `Unsupported` without a granted `posix` cap.
pub fn pipe() -> io::Result<(Pipe, Pipe)> {
    if !host::have_posix() {
        return Err(io::Error::UNSUPPORTED_PLATFORM);
    }
    let mut fds = [0u8; 8];
    let r = host::pipe(fds.as_mut_ptr());
    if r < 0 {
        return Err(io::Error::from_raw_os_error((-r) as i32));
    }
    let rfd = i32::from_le_bytes(fds[0..4].try_into().unwrap());
    let wfd = i32::from_le_bytes(fds[4..8].try_into().unwrap());
    Ok((Pipe { fd: rfd }, Pipe { fd: wfd }))
}

impl Pipe {
    pub fn try_clone(&self) -> io::Result<Self> {
        let fd = host::dup(self.fd as i64);
        if fd < 0 { Err(io::Error::from_raw_os_error((-fd) as i32)) } else { Ok(Pipe { fd: fd as i32 }) }
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        let n = host::read_fd(self.fd as i64, buf.as_mut_ptr(), buf.len() as i64);
        if n < 0 { Err(io::Error::from_raw_os_error((-n) as i32)) } else { Ok(n as usize) }
    }

    pub fn read_buf(&self, buf: BorrowedCursor<'_, u8>) -> io::Result<()> {
        crate::io::default_read_buf(|b| self.read(b), buf)
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

    pub fn read_to_end(&self, buf: &mut Vec<u8>) -> io::Result<usize> {
        let mut total = 0usize;
        let mut chunk = [0u8; 512];
        loop {
            let n = self.read(&mut chunk)?;
            if n == 0 {
                return Ok(total); // EOF: the pipe is empty and no writer is left
            }
            buf.extend_from_slice(&chunk[..n]);
            total += n;
        }
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        let n = host::write_fd(self.fd as i64, buf.as_ptr(), buf.len() as i64);
        if n < 0 { Err(io::Error::from_raw_os_error((-n) as i32)) } else { Ok(n as usize) }
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
}

impl Drop for Pipe {
    fn drop(&mut self) {
        let _ = host::close(self.fd as i64);
    }
}

impl fmt::Debug for Pipe {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Pipe").field("fd", &self.fd).finish()
    }
}
