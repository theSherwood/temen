//! The `webgpu` capability in the browser (#2665): the protocol every host shares (`temen-gpu`), with
//! the page's `navigator.gpu` as its [`Backend`]. The guest reaches it through the ordinary §7
//! host-defined-capability surface (`__vm_cap_resolve("webgpu")` + `__vm_host_call`). `temen-gpu`
//! decodes and checks each stream, here as natively, and hands this backend only checked streams,
//! in their JS form (`temen_gpu::js`). `web/webgpu.js` makes those calls by name, and holds no
//! knowledge of WebGPU of its own. The guest names objects and passes bytes; all GPU work, validation
//! included, happens in WebGPU: validation, not raw access, is the boundary (§2a).
//!
//! Each handler is a **session** of its own in the page, so two domains granted the capability (a
//! run and its fork twin) never see each other's ids. Its objects go when the handler does.

use std::sync::atomic::{AtomicU32, Ordering};

use temen_gpu::command::Command;
use temen_gpu::{Backend, Fault, Id, Readback, SurfaceInfo, TextureFormat};

#[link(wasm_import_module = "temen_host")]
extern "C" {
    /// Runs one stream of `session`'s, in its JS form: the steps as JSON, and the bytes they name.
    /// 0, or -19 with no device in the page. `json` and `blob` are this module's linear memory.
    fn webgpu_submit(
        session: u32,
        json: *const u8,
        json_len: usize,
        blob: *const u8,
        blob_len: usize,
    ) -> i64;
    /// The readback of `buffer`: its length, with that many bytes copied to `dst` and the readback
    /// released if `cap` holds them all; -11 until it arrives; -2 with none asked for.
    fn webgpu_read(session: u32, buffer: u32, dst: *mut u8, cap: usize) -> i64;
    /// The first error of `session`'s since the last call: its length, with up to `cap` bytes of it
    /// copied to `dst`; 0 for none.
    fn webgpu_error(session: u32, dst: *mut u8, cap: usize) -> i64;
    /// The page's canvas: its width and height written to `dims`, and its format's WebGPU name to
    /// `name`. The name's length, or -19 with no canvas.
    fn webgpu_surface(dims: *mut u32, name: *mut u8, cap: usize) -> i64;
    /// Frees `session`'s objects.
    fn webgpu_drop(session: u32);
}

/// The page's `navigator.gpu`, as one session's backend.
pub struct Js {
    session: u32,
}

impl Js {
    pub fn new() -> Self {
        static NEXT: AtomicU32 = AtomicU32::new(1);
        Js {
            session: NEXT.fetch_add(1, Ordering::Relaxed),
        }
    }
}

impl Drop for Js {
    fn drop(&mut self) {
        // SAFETY: a wasm import taking a scalar.
        unsafe { webgpu_drop(self.session) }
    }
}

impl Backend for Js {
    fn run(&mut self, cmds: &[Command<'_>]) -> Result<(), Fault> {
        let (mut json, mut blob) = (String::new(), Vec::new());
        temen_gpu::js::to_js(cmds, &mut json, &mut blob);
        // SAFETY: `json` and `blob` outlive the synchronous call, which only reads them.
        let r = unsafe {
            webgpu_submit(
                self.session,
                json.as_ptr(),
                json.len(),
                blob.as_ptr(),
                blob.len(),
            )
        };
        if r < 0 {
            return Err(Fault::NoDevice("no WebGPU device in this page".into()));
        }
        Ok(())
    }

    fn readback(&mut self, buffer: Id) -> Readback {
        // SAFETY: a capacity of 0 copies nothing; the second call writes at most `v.len()` bytes.
        let n = unsafe { webgpu_read(self.session, buffer, core::ptr::null_mut(), 0) };
        match n {
            -11 => Readback::Pending,
            n if n < 0 => Readback::None,
            n => {
                let mut v = vec![0u8; n as usize];
                unsafe { webgpu_read(self.session, buffer, v.as_mut_ptr(), v.len()) };
                Readback::Ready(v)
            }
        }
    }

    fn surface(&mut self) -> Option<SurfaceInfo> {
        let (mut dims, mut name) = ([0u32; 2], [0u8; 32]);
        // SAFETY: the import writes two `u32`s to `dims` and at most `name.len()` bytes to `name`.
        let n = unsafe { webgpu_surface(dims.as_mut_ptr(), name.as_mut_ptr(), name.len()) };
        if n < 0 {
            return None;
        }
        let name = std::str::from_utf8(name.get(..n as usize)?).ok()?;
        Some(SurfaceInfo {
            width: dims[0],
            height: dims[1],
            format: TextureFormat::from_name(name)?,
        })
    }

    fn take_error(&mut self) -> Option<String> {
        let mut b = vec![0u8; 1024];
        // SAFETY: the import writes at most `b.len()` bytes.
        let n = unsafe { webgpu_error(self.session, b.as_mut_ptr(), b.len()) };
        if n <= 0 {
            return None;
        }
        b.truncate((n as usize).min(b.len()));
        Some(String::from_utf8_lossy(&b).into_owned())
    }
}

/// The guest's window, as the protocol reaches it.
pub struct Window<'m>(pub &'m mut dyn temen_interp::GuestMem);

impl temen_gpu::Window for Window<'_> {
    fn read(&mut self, ptr: u64, len: u64) -> Option<Vec<u8>> {
        self.0.read_bytes(ptr, len)
    }
    fn write(&mut self, ptr: u64, bytes: &[u8]) -> Option<()> {
        self.0.write_bytes(ptr, bytes)
    }
}
