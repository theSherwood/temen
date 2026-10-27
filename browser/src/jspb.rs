//! **JS capabilities** (#1419, #2579) — capabilities the embedder names and implements *in JS*,
//! which the guest reaches the ordinary §7 way. The page stages each by name for the next on-ramp
//! open ([`crate::temen_onramp_stage_js_cap`]), [`grant_js_caps`] grants them into that open's
//! powerbox beside its others, and the module's §3.5 **import manifest** binds them by name with the
//! rest (IMPORTS.md §2.1), so a guest's `call.sym "<name>"` dispatches to the JS function the page
//! staged under that name. `web/powerbox.js`'s `definePowerbox` is exactly that: stage, then run.
//!
//! `webgpu` first proved the shape a JS-implemented capability takes — a `HostProc` closure that
//! marshals each op out through a wasm import — but as one hard-coded capability with a fixed op
//! vocabulary. This module is the general case: arbitrary names, each its own handle.
//!
//! Nothing here widens guest authority. A JS capability is an ordinary `HostProc` (iface
//! `HOST_PROC`): the guest names a masked, type-checked handle and passes scalars; guest memory is
//! reachable only through the bounds-checked [`temen_jspb_read`] / [`temen_jspb_write`] accessors
//! (the window pointer itself never crosses to JS), exactly the confinement the built-in
//! `Stream`/`Memory` caps get. An imported name the page did **not** stage refuses the open before a
//! single guest op executes, as every import nothing grants does.
//!
//! ABI: the page supplies one import, `temen_host.js_cap_call(slot, op, args_ptr, n_args, mem) ->
//! i64` — the whole JS side of the seam — and, from inside a call, uses the two window accessors.
//! On executor 2 a Worker's import sends the call to the page and waits inside it (`web/worker.js`),
//! and the page answers it with the same accessors over the calling task's window (`web/par.js`);
//! a run that names a `capModule` answers it in the Worker itself, from that module (#2645).

use std::cell::Cell;
use std::sync::Arc;

use temen_interp::{CapState, ForkedProc, GuestMem, Host, HostProcFork, UnlockedHostProc};

use crate::stash;

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "temen_host")]
extern "C" {
    /// The JS servicer for one capability call, supplied by the page as a wasm import. `slot` is the
    /// index [`crate::temen_onramp_stage_js_cap`] returned (which capability), `op` the guest's op
    /// selector, and `[args_ptr, n_args)` its `i64` arguments in this module's linear memory. `mem`
    /// is an opaque handle to the calling guest's window for [`temen_jspb_read`] /
    /// [`temen_jspb_write`] — null when the module declares no memory, and dangling after the call
    /// returns. Result: the capability's single `i64` (negative = an errno the guest can branch on,
    /// by convention).
    fn js_cap_call(
        slot: u32,
        op: u32,
        args_ptr: *const i64,
        n_args: usize,
        mem: *mut core::ffi::c_void,
    ) -> i64;
}

thread_local! {
    /// The most recent [`temen_jspb_read`] result on this thread — valid until its next read, like
    /// the other stashes. Per thread (#2645): on executor 2 each Worker serves its own task's calls.
    static SCRATCH: Cell<(*mut u8, usize)> = const { Cell::new((core::ptr::null_mut(), 0)) };
}

/// Grant `host` one capability per name in `names`, each served by the page's `js_cap_call` with its
/// slot, the name's index: the guest reaches the right JS function purely by which handle it
/// dispatches through (object-capability, not by name). Each is registered under its name. Returns
/// the `(name, handle)` seams for [`Host::bind_powerbox_manifest`]. Every on-ramp open grants the JS
/// capabilities the embedder staged for it through this (#2579).
///
/// Forkable (#1718): the JS function's state lives in JS, so a §14 child or `fork()` twin the
/// capability is re-granted to calls the same function, with its own window. Nothing here
/// serializes that state, so a run holding one cannot freeze (#1699).
///
/// Unlocked (#2645): the handler keeps nothing in the host, so where a domain's tasks share it under
/// a lock a call that waits on JS (on the page, or in a Worker) holds back no other task's host call.
pub(crate) fn grant_js_caps(host: &mut Host, names: &[String]) -> Vec<(String, i32)> {
    names
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            let call: UnlockedHostProc =
                Arc::new(move |op, args, mem| Ok(vec![dispatch(slot as u32, op, args, mem)]));
            let twin = Arc::clone(&call);
            let fork: HostProcFork = Arc::new(move |_pid| {
                Some(ForkedProc::unlocked(
                    Arc::clone(&twin),
                    CapState::Uncaptured,
                ))
            });
            let handle = host.grant_host_proc_unlocked(call, Some(fork), CapState::Uncaptured);
            // §7 F7/F9: the name is also the label, so a guest can `self.resolve` / `self.label` it.
            host.register_cap_name(name, handle);
            (name.clone(), handle)
        })
        .collect()
}

// ---- the calling guest's window, as seen from JS ----------------------------------------------

/// An opaque handle to the calling guest's window, live only for the duration of one capability
/// call. Reachable only through [`temen_jspb_read`] / [`temen_jspb_write`], each bounds-checked
/// against the window and fail-closed — the raw window pointer never crosses to JS. (The C ABI's
/// `TemenGuestMem` (F5) is the same shim for `temen-capi`'s function-pointer callbacks.)
#[repr(C)]
pub struct JsGuestMem {
    mem: *mut dyn GuestMem,
}

/// Copy `len` bytes from guest window offset `ptr` into a cdylib buffer and return its pointer, for
/// JS to read as `new Uint8Array(memory.buffer, p, len)`. Null (nothing copied) if `mem` is null or
/// `[ptr, ptr+len)` is not wholly inside the window. Valid until the next call.
///
/// # Safety
/// `mem` is the handle passed to the in-flight capability call (or null).
#[no_mangle]
pub unsafe extern "C" fn temen_jspb_read(mem: *mut JsGuestMem, ptr: u64, len: usize) -> *const u8 {
    let Some(shim) = mem.as_ref() else {
        return core::ptr::null();
    };
    let m = &*shim.mem;
    match m.read_bytes(ptr, len as u64) {
        Some(bytes) => SCRATCH.with(|s| {
            let mut slot = s.get();
            stash(&mut slot, bytes);
            s.set(slot);
            slot.0.cast_const()
        }),
        None => core::ptr::null(),
    }
}

/// Copy `len` bytes from `[src, src+len)` in this module's memory into the guest window at `ptr`.
/// `0` on success, `-1` (nothing written) if `mem`/`src` is null or the range is not wholly inside
/// the window and writable — a read-only or unmapped page fails closed, like the built-ins.
///
/// # Safety
/// `mem` is the handle passed to the in-flight capability call (or null); `src` points to at least
/// `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn temen_jspb_write(
    mem: *mut JsGuestMem,
    ptr: u64,
    src: *const u8,
    len: usize,
) -> i32 {
    let Some(shim) = mem.as_mut() else {
        return -1;
    };
    if src.is_null() {
        return -1;
    }
    let data = core::slice::from_raw_parts(src, len);
    let m = &mut *shim.mem;
    match m.write_bytes(ptr, data) {
        Some(()) => 0,
        None => -1,
    }
}

/// Marshal one capability call out to JS: wrap the live window borrow in a [`JsGuestMem`] that
/// lives exactly as long as the call, and hand the slot + op + args across.
fn dispatch(slot: u32, op: u32, args: &[i64], mem: Option<&mut dyn GuestMem>) -> i64 {
    match mem {
        Some(m) => {
            let mut shim = JsGuestMem {
                // SAFETY: erase the borrow's lifetime to carry it through the opaque handle. The
                // pointer is dereferenced (by `temen_jspb_read`/`write`) only while JS is inside
                // this call — `m`'s borrow is live and otherwise untouched, so no aliasing.
                mem: unsafe { core::mem::transmute::<&mut dyn GuestMem, *mut dyn GuestMem>(m) },
            };
            call_js(slot, op, args, &mut shim as *mut JsGuestMem)
        }
        None => call_js(slot, op, args, core::ptr::null_mut()),
    }
}

#[cfg(target_arch = "wasm32")]
fn call_js(slot: u32, op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    // SAFETY: wasm-only import; `args`/`mem` outlive the synchronous call. The window handle
    // crosses as an opaque pointer — JS only ever hands it straight back to the accessors.
    unsafe { js_cap_call(slot, op, args.as_ptr(), args.len(), mem.cast()) }
}

/// `-ENOSYS` — what a capability call returns with no servicer installed (natively; a page that
/// supplies a stub import returns the same by convention).
#[cfg(not(target_arch = "wasm32"))]
const ENOSYS: i64 = -38;

/// Natively there is no JS, so the seam's far side is a hook — which is what makes the *whole*
/// binding path (registry → manifest → handle → window accessors) testable off-browser.
#[cfg(not(target_arch = "wasm32"))]
pub type TestDispatch = fn(u32, u32, &[i64], *mut JsGuestMem) -> i64;

#[cfg(not(target_arch = "wasm32"))]
static mut TEST_DISPATCH: Option<TestDispatch> = None;

/// Install (or clear) the native stand-in for the JS servicer. Test-only; callers serialize.
#[cfg(not(target_arch = "wasm32"))]
pub fn set_test_dispatch(f: Option<TestDispatch>) {
    // SAFETY: single-threaded by the caller's FFI lock, like the other statics here.
    unsafe { *core::ptr::addr_of_mut!(TEST_DISPATCH) = f };
}

#[cfg(not(target_arch = "wasm32"))]
fn call_js(slot: u32, op: u32, args: &[i64], mem: *mut JsGuestMem) -> i64 {
    // SAFETY: see `set_test_dispatch`.
    match unsafe { *core::ptr::addr_of!(TEST_DISPATCH) } {
        Some(f) => f(slot, op, args, mem),
        None => ENOSYS,
    }
}
