//! temen stdio: the three standard streams reach the host through the powerbox's `write`/`read`
//! bindings (the temen-llvm on-ramp's "Lane C" — `write`/`read`/`exit` bind to the powerbox
//! `Stream`/`Exit` handles the synthesized `_start` grants, POSIX.md ops 0/1/4). The `fd` argument
//! is threaded for POSIX shape but the on-ramp drops it (the handle selects the endpoint), so
//! stdout and stderr currently share the powerbox stdout stream. Errno rides the return value
//! in-band (negative), matching temen-posix's value-not-trap ABI.
use crate::io;

unsafe extern "C" {
    #[link_name = "write"]
    fn sys_write(fd: i32, buf: *const u8, len: usize) -> isize;
    #[link_name = "read"]
    fn sys_read(fd: i32, buf: *mut u8, len: usize) -> isize;
    // The powerbox stderr stream — a distinct `Stream` from stdout (the on-ramp binds this to the
    // `"stderr"` manifest handle), so `eprintln!` captures separately. `(buf, len) -> n | -errno`.
    #[link_name = "__vm_write_stderr"]
    fn sys_write_stderr(buf: *const u8, len: usize) -> isize;
}

const STDIN: i32 = 0;
const STDOUT: i32 = 1;

fn from_ret(n: isize) -> io::Result<usize> {
    if n < 0 { Err(io::Error::from_raw_os_error((-n) as i32)) } else { Ok(n as usize) }
}

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        from_ret(unsafe { sys_read(STDIN, buf.as_mut_ptr(), buf.len()) })
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        from_ret(unsafe { sys_write(STDOUT, buf.as_ptr(), buf.len()) })
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        from_ret(unsafe { sys_write_stderr(buf.as_ptr(), buf.len()) })
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = crate::sys::io::DEFAULT_BUF_SIZE;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
