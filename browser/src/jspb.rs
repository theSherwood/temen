//! **A powerbox defined in JavaScript** (`temen_jspb_*`, #1419) — the embedder names the capabilities and
//! implements them *in JS*; the guest reaches them the ordinary §7 way.
//!
//! Every other run entry in this cdylib grants a powerbox whose semantics are **Rust-side**
//! (`powerbox_exec`'s fixed §3e prefix, `grant_onramp_caps`' `display`/`keyboard`/`fs`/`webgpu`).
//! `webgpu` already proved the shape a JS-implemented capability takes — a `HostProc` closure that
//! marshals each op out through a wasm import — but it is one hard-coded capability with a fixed op
//! vocabulary. This module generalizes exactly that into the general case: a page registers
//! **arbitrary names** ([`temen_jspb_bind`]), each becomes its own handle, and the module's §3.5
//! **import manifest** binds slot `i` ↔ import `i` by name at instantiation (IMPORTS.md §2.1), so a
//! guest's `call.sym "<name>"` dispatches to the JS function the page bound to that name.
//!
//! Nothing here widens guest authority. A JS capability is an ordinary `HostProc` (iface
//! `HOST_PROC`): the guest names a masked, type-checked handle and passes scalars; guest memory is
//! reachable only through the bounds-checked [`temen_jspb_read`] / [`temen_jspb_write`] accessors
//! (the window pointer itself never crosses to JS), exactly the confinement the built-in
//! `Stream`/`Memory` caps get. An imported name the page did **not** bind fails the run closed,
//! before a single guest op executes.
//!
//! ABI (all pointer/length values follow this build's `temen_abi_is64` convention):
//! ```text
//! temen_jspb_reset()                        -- drop every binding
//! temen_jspb_bind(name_ptr, name_len) -> i32 -- register a name; returns its slot (JS dispatch key)
//! temen_jspb_run(mod_ptr, mod_len)    -> i64 -- run the module under those capabilities
//! temen_jspb_error_ptr() / _len()            -- why the last run failed (UTF-8), empty on success
//! ```
//! and, from inside a capability call, the two window accessors. The page supplies one import:
//! `temen_host.js_cap_call(slot, op, args_ptr, n_args, mem) -> i64` — the whole JS side of the seam.

use temen_interp::{bytecode, cap_id, BoundImport, GuestMem, Host, Trap, Value};

use crate::{
    stash, STATUS_BAD_RESULT, STATUS_DECODE_ERR, STATUS_EXIT, STATUS_OK, STATUS_TRAP,
    STATUS_UNSUPPORTED, STATUS_VERIFY_ERR,
};

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "temen_host")]
extern "C" {
    /// The JS servicer for one capability call, supplied by the page as a wasm import. `slot` is the
    /// index [`temen_jspb_bind`] returned (which capability), `op` the guest's op selector, and
    /// `[args_ptr, n_args)` its `i64` arguments in this module's linear memory. `mem` is an opaque
    /// handle to the calling guest's window for [`temen_jspb_read`] / [`temen_jspb_write`] — null
    /// when the module declares no memory, and dangling after the call returns. Result: the
    /// capability's single `i64` (negative = an errno the guest can branch on, by convention).
    fn js_cap_call(
        slot: u32,
        op: u32,
        args_ptr: *const i64,
        n_args: usize,
        mem: *mut core::ffi::c_void,
    ) -> i64;
}

/// The names bound by [`temen_jspb_bind`], in slot order: slot `i` is the `i`-th registered name,
/// and the key JS dispatches on. Single-threaded, like every other stash in this cdylib.
static mut CAPS: Vec<String> = Vec::new();

/// Why the last [`temen_jspb_run`] refused to run (UTF-8), or empty. Cdylib-managed, like `PARSE`.
static mut ERR: (*mut u8, usize) = (core::ptr::null_mut(), 0);

/// The most recent [`temen_jspb_read`] result — valid until the next read, like the other stashes.
static mut SCRATCH: (*mut u8, usize) = (core::ptr::null_mut(), 0);

fn set_err(msg: &str) {
    // SAFETY: single-threaded main-thread state, read back only through the accessors below.
    unsafe { stash(&mut *core::ptr::addr_of_mut!(ERR), msg.as_bytes().to_vec()) };
}

/// Drop every binding (and the last error). Call before a fresh series of [`temen_jspb_bind`]s so a
/// run sees exactly the capabilities the page just declared.
#[no_mangle]
pub extern "C" fn temen_jspb_reset() {
    // SAFETY: single-threaded main-thread state.
    unsafe { (*core::ptr::addr_of_mut!(CAPS)).clear() };
    set_err("");
}

/// Register the capability name at `[name_ptr, name_len)` (UTF-8) and return its **slot** — the
/// index the page dispatches on in `js_cap_call`, and the handle order the manifest binds against.
/// `-1` if the name is not UTF-8, is empty, or is already bound.
#[no_mangle]
pub extern "C" fn temen_jspb_bind(name_ptr: *const u8, name_len: usize) -> i32 {
    if name_ptr.is_null() || name_len == 0 {
        return -1;
    }
    // SAFETY: the host guarantees `[name_ptr, name_len)` is a live allocation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(name_ptr, name_len) };
    let Ok(name) = core::str::from_utf8(bytes) else {
        return -1;
    };
    // SAFETY: single-threaded main-thread state.
    let caps = unsafe { &mut *core::ptr::addr_of_mut!(CAPS) };
    if caps.iter().any(|n| n == name) {
        return -1; // a name is one capability; rebinding would silently shadow the first
    }
    caps.push(name.to_string());
    (caps.len() - 1) as i32
}

/// Decode + verify the module at `[mod_ptr, mod_len)`, bind its imports to the registered JS
/// capabilities **by name**, and run its entry (`_start` if exported, else function 0) on the
/// bytecode engine. Returns the entry's first `i64` result (`0` on any non-`OK` status — read
/// [`crate::temen_status`], and [`temen_jspb_error_ptr`] for the message).
///
/// Fail-closed: an import the page did not bind is named back as an error and **nothing runs**.
#[no_mangle]
pub extern "C" fn temen_jspb_run(mod_ptr: *const u8, mod_len: usize) -> i64 {
    let set = |s: i32| unsafe { crate::LAST_STATUS = s };
    set_err("");
    // SAFETY: the host guarantees `[mod_ptr, mod_len)` is a live `temen_alloc`ation it just filled.
    let bytes = unsafe { core::slice::from_raw_parts(mod_ptr, mod_len) };
    let m = match temen_encode::decode_module(bytes) {
        Ok(m) => m,
        Err(e) => {
            set(STATUS_DECODE_ERR);
            set_err(&format!("decode error: {e:?}"));
            return 0;
        }
    };
    if let Err(e) = temen_verify::verify_module(&m) {
        set(STATUS_VERIFY_ERR);
        set_err(&format!("verify error: {e:?}"));
        return 0;
    }
    // SAFETY: single-threaded main-thread state.
    let caps: &[String] = unsafe { &*core::ptr::addr_of!(CAPS) };
    // Every import must name a bound capability — report *all* the misses at once, then refuse.
    let missing: Vec<&str> = m
        .imports
        .iter()
        .map(|im| im.name.as_str())
        .filter(|n| !caps.iter().any(|c| c == n))
        .collect();
    if !missing.is_empty() {
        set(STATUS_UNSUPPORTED);
        set_err(&format!(
            "unbound import{}: {} — the page defined [{}]",
            if missing.len() == 1 { "" } else { "s" },
            missing.join(", "),
            caps.join(", ")
        ));
        return 0;
    }

    let mut host = Host::new();
    // One handle per registered name: the closure carries its slot, so the guest reaches the right
    // JS function purely by which handle it dispatches through (object-capability, not by name).
    let handles: Vec<i32> = caps
        .iter()
        .enumerate()
        .map(|(slot, name)| {
            let handle = host.grant_host_proc(Box::new(move |op, args, mem, _| {
                Ok(vec![dispatch(slot as u32, op, args, mem)])
            }));
            // §7 F7/F9: the name is also the label, so a guest can `self.resolve` / `self.label` it.
            host.register_cap_name(name, handle);
            handle
        })
        .collect();
    // IMPORTS.md §2.1 — the manifest binds import `i` to its named capability's handle. The module
    // bytes are never rewritten; `call.sym` dispatches through the binding.
    if !m.imports.is_empty() {
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                let slot = caps
                    .iter()
                    .position(|c| c == &im.name)
                    .expect("checked above");
                BoundImport::required(cap_id::HOST_PROC, 0, handles[slot])
            })
            .collect();
        host.set_import_bindings(bindings);
    }

    // A JS-powerbox guest's entry is paramless: capabilities arrive through the manifest, never as
    // positional handle arguments.
    let entry = m.resolve_export("_start").unwrap_or(0);
    let mut fuel = u64::MAX;
    match bytecode::compile_and_run_with_host(&m, entry, &[], &mut fuel, &mut host) {
        None => {
            set(STATUS_UNSUPPORTED);
            set_err("the bytecode engine does not support this module");
            0
        }
        Some(Err(Trap::Exit(code))) => {
            set(STATUS_EXIT);
            unsafe { crate::EXIT_CODE = code };
            0
        }
        Some(Err(t)) => {
            set(STATUS_TRAP);
            set_err(&format!("trap: {t:?}"));
            0
        }
        Some(Ok(vals)) => match vals.first() {
            Some(Value::I64(x)) => {
                set(STATUS_OK);
                *x
            }
            Some(Value::I32(x)) => {
                set(STATUS_OK);
                *x as i64
            }
            _ => {
                set(STATUS_BAD_RESULT);
                set_err("the entry returned no i32/i64 result");
                0
            }
        },
    }
}

/// Pointer / length of the last [`temen_jspb_run`] error message (UTF-8; empty on success).
#[no_mangle]
pub extern "C" fn temen_jspb_error_ptr() -> *const u8 {
    unsafe { (*core::ptr::addr_of!(ERR)).0 }
}
#[no_mangle]
pub extern "C" fn temen_jspb_error_len() -> usize {
    unsafe { (*core::ptr::addr_of!(ERR)).1 }
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
        Some(bytes) => {
            stash(&mut *core::ptr::addr_of_mut!(SCRATCH), bytes);
            (*core::ptr::addr_of!(SCRATCH)).0
        }
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
