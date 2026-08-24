//! temen environment variables: `getenv`/`setenv`/`unsetenv`/`vars` reach the POSIX personality through
//! the PAL `host` bridge (temen-posix `OP_GETENV_R`/`OP_SETENV`/`OP_UNSETENV`/`OP_ENVIRON`). The
//! buffer-writing `getenv_r`/`environ` ops copy into guest-owned memory (no personality arena), so
//! they never contend with the guest's own heap. Needs a granted `posix` cap (`run_with_caps`);
//! without one every lookup is unset and mutation is `Unsupported`.
pub use super::common::Env;
use crate::ffi::{OsStr, OsString};
use crate::io;
use crate::sys::pal::host;

pub fn getenv(name: &OsStr) -> Option<OsString> {
    if !host::have_posix() {
        return None;
    }
    let nb = name.as_encoded_bytes();
    let len = host::getenv_r(nb.as_ptr(), nb.len() as i64, crate::ptr::null_mut(), 0);
    if len < 0 {
        return None;
    }
    let mut buf = vec![0u8; len as usize];
    let got = host::getenv_r(nb.as_ptr(), nb.len() as i64, buf.as_mut_ptr(), buf.len() as i64);
    if got < 0 {
        return None;
    }
    buf.truncate(got as usize);
    // SAFETY: temen has no OS-specific OsStr encoding; the bytes are the value verbatim.
    Some(unsafe { OsString::from_encoded_bytes_unchecked(buf) })
}

pub unsafe fn setenv(name: &OsStr, value: &OsStr) -> io::Result<()> {
    if !host::have_posix() {
        return Err(io::const_error!(io::ErrorKind::Unsupported, "no posix cap: cannot set env vars"));
    }
    let nb = name.as_encoded_bytes();
    let vb = value.as_encoded_bytes();
    let r = host::setenv(nb.as_ptr(), nb.len() as i64, vb.as_ptr(), vb.len() as i64);
    if r < 0 { Err(io::Error::from_raw_os_error((-r) as i32)) } else { Ok(()) }
}

pub unsafe fn unsetenv(name: &OsStr) -> io::Result<()> {
    if !host::have_posix() {
        return Err(io::const_error!(io::ErrorKind::Unsupported, "no posix cap: cannot unset env vars"));
    }
    let nb = name.as_encoded_bytes();
    let r = host::unsetenv(nb.as_ptr(), nb.len() as i64);
    if r < 0 { Err(io::Error::from_raw_os_error((-r) as i32)) } else { Ok(()) }
}

pub fn env() -> Env {
    let mut v: Vec<(OsString, OsString)> = Vec::new();
    if host::have_posix() {
        let mut i = 0i64;
        loop {
            let len = host::environ(i, crate::ptr::null_mut(), 0);
            if len < 0 {
                break;
            }
            let mut buf = vec![0u8; len as usize];
            let got = host::environ(i, buf.as_mut_ptr(), buf.len() as i64);
            if got < 0 {
                break;
            }
            buf.truncate(got as usize);
            if let Some(eq) = buf.iter().position(|&b| b == b'=') {
                // SAFETY: temen has no OS-specific OsStr encoding.
                let key = unsafe { OsString::from_encoded_bytes_unchecked(buf[..eq].to_vec()) };
                let val = unsafe { OsString::from_encoded_bytes_unchecked(buf[eq + 1..].to_vec()) };
                v.push((key, val));
            }
            i += 1;
        }
    }
    Env::new(v)
}
