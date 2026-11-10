//! **The WebGPU capability's protocol** (`DESIGN.md` §17, #2665): one protocol for every host.
//!
//! A guest drives a GPU through the `webgpu` capability (`__vm_cap_resolve("webgpu")` +
//! `__vm_host_call`). It never holds a GPU object. It names objects by ids it chooses, and it
//! submits **command streams**: runs of records it builds in its own window ([`command`]), as a
//! WebGPU client's wire protocol does. A stream is copied out of the window, decoded totally, and
//! checked against the live ids ([`Session`]) before any of it runs, so a malformed or ill-named
//! stream runs nothing. What remains to validate (formats, usages, layouts, WGSL) WebGPU itself
//! validates, host-side: validation, not raw access, is the boundary (§2a).
//!
//! The protocol is all here, and both hosts share it through [`Gpu`]: the ops, the decoding, the
//! checks, and the errors. A host supplies only a [`Backend`]: natively wgpu (`temen-webgpu`), and
//! in the browser the page's `navigator.gpu`, through the JS form of [`js`]. A stream is plain data,
//! so a GPU broker in another process (§17) takes the same streams.
//!
//! ## Ops (`__vm_host_call(handle, op, a, b, c, d) -> i64`)
//!
//! | op | name | args | returns |
//! |----|------|------|---------|
//! | 0 | `submit` | `(ptr, len)`: a stream | `0`; `-EINVAL` for a stream refused, which ran nothing; `-ENODEV` with no adapter |
//! | 1 | `read` | `(buffer, ptr, len)` | the readback's length, with up to `len` bytes copied; `-EAGAIN` until it arrives; `-ENOENT` with none asked for |
//! | 2 | `error` | `(ptr, cap)` | the length of the first error since the last call, with up to `cap` bytes of it copied; `0` for none |
//! | 3 | `surface` | `(ptr)` | `0`, with `{u32 width, height, format, 0}` written; `-ENODEV` with no surface |
//!
//! Errors are values (D42): a refused stream says why through `error`, as does an error WebGPU
//! reports. WebGPU reports its own errors when it has checked the work, which in a browser is
//! after the call returns, so a guest reads them with `error` rather than from `submit`.
//!
//! A readback completes some time after its submit; a guest asks `read` until it stops saying
//! `-EAGAIN`, on every host alike. (#2838 makes it park the caller instead.)

#![no_std]

extern crate alloc;

pub mod command;
pub mod enums;
pub mod js;
pub mod session;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

pub use command::{decode, Command, Encoder};
pub use enums::TextureFormat;
pub use session::{Kind, Session};

/// An object's id, chosen by the guest: 1 up to [`session::MAX_IDS`].
pub type Id = u32;

/// The ops' numbers.
pub mod op {
    pub const SUBMIT: u32 = 0;
    pub const READ: u32 = 1;
    pub const ERROR: u32 = 2;
    pub const SURFACE: u32 = 3;
}

/// The negative errnos the ops return.
pub mod errno {
    pub const ENOENT: i64 = -2;
    pub const EAGAIN: i64 = -11;
    pub const EFAULT: i64 = -14;
    pub const ENODEV: i64 = -19;
    pub const EINVAL: i64 = -22;
}

/// The guest's window, as a host reaches it: reads copy out, writes copy in, and either fails for
/// a range outside it.
pub trait Window {
    fn read(&mut self, ptr: u64, len: u64) -> Option<Vec<u8>>;
    fn write(&mut self, ptr: u64, bytes: &[u8]) -> Option<()>;
}

/// The surface a host binds: its size and format, which a guest's pipelines target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceInfo {
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
}

/// Where a readback stands.
pub enum Readback {
    Ready(Vec<u8>),
    Pending,
    /// None was asked for.
    None,
}

/// Why a backend could not run a stream at all.
pub enum Fault {
    /// The host has no adapter, or lost its device.
    NoDevice(String),
}

/// What a host supplies: a GPU to run checked streams on.
pub trait Backend {
    /// Runs a stream that [`Session::check`] passed, in order.
    fn run(&mut self, cmds: &[Command<'_>]) -> Result<(), Fault>;
    /// Takes the readback of `buffer` once it has arrived.
    fn readback(&mut self, buffer: Id) -> Readback;
    /// The surface, if the host bound one.
    fn surface(&mut self) -> Option<SurfaceInfo>;
    /// The first error WebGPU reported since the last call.
    fn take_error(&mut self) -> Option<String>;
}

/// The capability's state for one guest: its ids, its backend, and the first error not yet read.
pub struct Gpu<B> {
    pub session: Session,
    pub backend: B,
    error: Option<String>,
}

impl<B: Backend> Gpu<B> {
    pub fn new(backend: B) -> Self {
        Gpu {
            session: Session::new(),
            backend,
            error: None,
        }
    }

    fn fail(&mut self, errno: i64, message: String) -> i64 {
        self.error.get_or_insert(message);
        errno
    }

    /// Answers one op. `args` are its arguments; `window`, the guest's window.
    pub fn call(&mut self, op: u32, args: &[i64], window: Option<&mut dyn Window>) -> i64 {
        let a = |i: usize| args.get(i).copied().unwrap_or(0);
        let Some(window) = window else {
            return errno::EFAULT;
        };
        match op {
            op::SUBMIT => {
                let (ptr, len) = (a(0), a(1));
                if !(0..=command::MAX_STREAM as i64).contains(&len) {
                    return self.fail(errno::EINVAL, format!("a stream of {len} bytes"));
                }
                let Some(stream) = window.read(ptr as u64, len as u64) else {
                    return errno::EFAULT;
                };
                self.submit(&stream)
            }
            op::READ => {
                let (buffer, ptr, len) = (a(0), a(1), a(2));
                if !(0..=u32::MAX as i64).contains(&buffer) || len < 0 {
                    return errno::EINVAL;
                }
                match self.backend.readback(buffer as Id) {
                    Readback::Ready(bytes) => {
                        let n = bytes.len().min(len as usize);
                        if window.write(ptr as u64, &bytes[..n]).is_none() {
                            return errno::EFAULT;
                        }
                        bytes.len() as i64
                    }
                    Readback::Pending => errno::EAGAIN,
                    Readback::None => errno::ENOENT,
                }
            }
            op::ERROR => {
                let (ptr, cap) = (a(0), a(1));
                let Some(message) = self.error.take().or_else(|| self.backend.take_error()) else {
                    return 0;
                };
                let n = message.len().min(cap.max(0) as usize);
                if window.write(ptr as u64, &message.as_bytes()[..n]).is_none() {
                    return errno::EFAULT;
                }
                message.len() as i64
            }
            op::SURFACE => {
                let Some(s) = self.backend.surface() else {
                    return errno::ENODEV;
                };
                let mut b = [0u8; 16];
                b[..4].copy_from_slice(&s.width.to_le_bytes());
                b[4..8].copy_from_slice(&s.height.to_le_bytes());
                b[8..12].copy_from_slice(&s.format.code().to_le_bytes());
                if window.write(a(0) as u64, &b).is_none() {
                    return errno::EFAULT;
                }
                0
            }
            _ => errno::EINVAL,
        }
    }

    /// Decodes, checks and runs one stream.
    pub fn submit(&mut self, stream: &[u8]) -> i64 {
        let cmds = match decode(stream) {
            Ok(c) => c,
            Err(e) => {
                return self.fail(
                    errno::EINVAL,
                    format!("command {} (code {}): {}", e.index, e.command, e.what),
                )
            }
        };
        if self.backend.surface().is_none() {
            if let Some(i) = cmds
                .iter()
                .position(|c| matches!(c, Command::Acquire { .. }))
            {
                return self.fail(
                    errno::EINVAL,
                    format!("command {i} (acquire): the host bound no surface"),
                );
            }
        }
        if let Err(r) = self.session.check(&cmds) {
            return self.fail(
                errno::EINVAL,
                format!("command {} ({}): {}", r.index, r.command, r.what),
            );
        }
        match self.backend.run(&cmds) {
            Ok(()) => 0,
            Err(Fault::NoDevice(why)) => self.fail(errno::ENODEV, why),
        }
    }
}

#[cfg(test)]
mod tests;
