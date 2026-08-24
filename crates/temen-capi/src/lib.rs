//! **`temen-capi` — the C ABI over the `temen-run` embedding surface** (POWERBOX.md Phase 5).
//!
//! A C program can: parse a module (text or binary IR) whose paramless exported `_start` declares
//! its capability imports (the IMPORTS.md manifest), bind host capabilities **by name** (built-ins,
//! or its own C function pointers — the wasm-style import registry of Phase 2), instantiate, and run on any backend
//! under a uniform config (Phase 3) — then read back the outcome and captured stdout/stderr. It is the
//! same pipeline as the Rust `Instance` API, exposed through `extern "C"`.
//!
//! **Discipline (FFI safety):** every entry point catches panics at the boundary (a panic never
//! unwinds into C — it becomes a null/error return), reports failures through a thread-local message
//! ([`temen_last_error`]), and owns memory through explicit `*_free` calls. Handles are opaque pointers;
//! `instantiate*` **consume** the module/imports handles passed to them.
//!
//! **Host-capability callbacks** receive the calling guest's linear-memory window as an opaque
//! `TemenGuestMem*` (its last parameter; `NULL` if the module declares no memory), readable/writable
//! through the **bounds-checked** [`temen_guest_read`] / [`temen_guest_write`] accessors (Followup F5). The
//! shim is valid only for the duration of one callback — never retain the pointer past the call. A
//! callback that only computes on its scalar args can ignore it.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::time::Duration;

use temen_interp::{GuestMem, HostProc, Trap};
use temen_run::{
    instantiate, instantiate_with_imports, Backend, HostCap, Imports, Instance, Limits, MemEvent,
    MemHookFn, Outcome, Run, RunConfig, Value,
};

// ----------------------------------------------------------------------------
// Status codes
// ----------------------------------------------------------------------------

/// Success.
pub const TEMEN_OK: i32 = 0;
/// A null handle was passed where a live one was required.
pub const TEMEN_ERR_NULL: i32 = 1;
/// A fallible operation failed; see [`temen_last_error`] for the message.
pub const TEMEN_ERR_FAILED: i32 = 2;
/// A panic was caught at the FFI boundary (a bug — please report); see [`temen_last_error`].
pub const TEMEN_ERR_PANIC: i32 = 3;

/// Backend selectors for [`temen_instance_run`] (mirror [`temen_run::Backend`]).
pub const TEMEN_BACKEND_TREEWALK: i32 = 0;
pub const TEMEN_BACKEND_BYTECODE: i32 = 1;
pub const TEMEN_BACKEND_JIT: i32 = 2;

/// `temen_run_outcome_kind` values.
pub const TEMEN_OUTCOME_RETURNED: i32 = 0;
pub const TEMEN_OUTCOME_EXITED: i32 = 1;

/// `TemenMemEvent::kind` values (mirror the [`temen_run::MemEvent`] variants). For scalar and atomic
/// events, `addr` is the effective guest address and `size` the access width in bytes; for `COPY`,
/// `addr` is the destination, `src` the source, `size` the byte length; for `FILL`, `addr` is the
/// destination and `size` the byte length (`src` is 0). v128 accesses are `LOAD`/`STORE`, size 16.
pub const TEMEN_MEM_LOAD: i32 = 0;
pub const TEMEN_MEM_STORE: i32 = 1;
pub const TEMEN_MEM_ATOMIC_LOAD: i32 = 2;
pub const TEMEN_MEM_ATOMIC_STORE: i32 = 3;
pub const TEMEN_MEM_ATOMIC_RMW: i32 = 4;
pub const TEMEN_MEM_ATOMIC_CMPXCHG: i32 = 5;
pub const TEMEN_MEM_COPY: i32 = 6;
pub const TEMEN_MEM_FILL: i32 = 7;

/// The max results a host-capability callback may return (the closure's scratch buffer size).
const TEMEN_MAX_RESULTS: usize = 16;

// ----------------------------------------------------------------------------
// Error reporting (thread-local; never panics across the boundary)
// ----------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error(msg: impl Into<Vec<u8>>) {
    // Strip interior NULs so the message always survives as a C string.
    let bytes: Vec<u8> = msg.into().into_iter().filter(|&b| b != 0).collect();
    let c = CString::new(bytes).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// The last error message on this thread (set by a failed call), or `NULL` if none. Valid until the
/// next `temen-capi` call on the same thread; copy it if you need to keep it.
#[no_mangle]
pub extern "C" fn temen_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ref().map_or(ptr::null(), |c| c.as_ptr()))
}

/// Run `f`, catching panics and `Err`s; on either, set the error and return a null pointer.
fn guard_ptr<T>(f: impl FnOnce() -> Result<*mut T, String>) -> *mut T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_error("panic caught at the temen-capi boundary");
            ptr::null_mut()
        }
    }
}

/// Run `f`, catching panics and `Err`s; map to a status code.
fn guard_status(f: impl FnOnce() -> Result<(), String>) -> i32 {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => TEMEN_OK,
        Ok(Err(e)) => {
            set_error(e);
            TEMEN_ERR_FAILED
        }
        Err(_) => {
            set_error("panic caught at the temen-capi boundary");
            TEMEN_ERR_PANIC
        }
    }
}

// ----------------------------------------------------------------------------
// Opaque handles
// ----------------------------------------------------------------------------

/// An IR module (opaque). Built by `temen_module_parse_text` / `temen_module_decode`, consumed by
/// `temen_instantiate*`.
pub struct TemenModule(temen_ir::Module);
/// A name → capability registry (opaque). Built by `temen_imports_new`, consumed by
/// `temen_instantiate_with_imports`.
pub struct TemenImports(Imports);
/// A resolved, verified instance (opaque).
pub struct TemenInstance(Instance);
/// The result of a run (opaque): outcome + captured stdout/stderr.
pub struct TemenRun(Run);

// ----------------------------------------------------------------------------
// Module
// ----------------------------------------------------------------------------

/// Parse a module from **text IR** (a NUL-terminated UTF-8 string). Returns a module handle, or
/// `NULL` on a parse error (see [`temen_last_error`]).
///
/// # Safety
/// `ir` must be a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn temen_module_parse_text(ir: *const c_char) -> *mut TemenModule {
    guard_ptr(|| {
        if ir.is_null() {
            return Err("temen_module_parse_text: null ir".into());
        }
        let s = CStr::from_ptr(ir)
            .to_str()
            .map_err(|_| "ir is not valid UTF-8".to_string())?;
        let m = temen_text::parse_module(s).map_err(|e| format!("parse: {e:?}"))?;
        Ok(Box::into_raw(Box::new(TemenModule(m))))
    })
}

/// Parse a module from the **binary IR** encoding (`temen-encode`).
///
/// # Safety
/// `bytes` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn temen_module_decode(bytes: *const u8, len: usize) -> *mut TemenModule {
    guard_ptr(|| {
        if bytes.is_null() && len != 0 {
            return Err("temen_module_decode: null bytes".into());
        }
        let slice = if len == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(bytes, len)
        };
        let m = temen_encode::decode_module(slice).map_err(|e| format!("decode: {e:?}"))?;
        Ok(Box::into_raw(Box::new(TemenModule(m))))
    })
}

/// Free a module handle (only if it was *not* consumed by an `temen_instantiate*` call).
///
/// # Safety
/// `m` must be a live module handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_module_free(m: *mut TemenModule) {
    if !m.is_null() {
        drop(Box::from_raw(m));
    }
}

// ----------------------------------------------------------------------------
// Imports registry
// ----------------------------------------------------------------------------

/// A C host-capability callback: compute `n_results` (≤ buffer capacity) outputs from `n_args` inputs
/// for operation `op`. Return the number of results written (`>= 0`), or a negative value to **trap**
/// the capability call (fail-closed). `ctx` is the opaque pointer registered alongside the callback.
/// `mem` is the calling guest's linear-memory window (`NULL` if the module declares none), readable /
/// writable through [`temen_guest_read`] / [`temen_guest_write`] — valid only for this call (F5).
pub type TemenHostProc = extern "C" fn(
    ctx: *mut c_void,
    op: u32,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    results_cap: usize,
    mem: *mut TemenGuestMem,
) -> i32;

/// An opaque handle to the calling guest's linear-memory window, handed to an [`TemenHostProc`] for the
/// duration of one call. Access it **only** through [`temen_guest_read`] / [`temen_guest_write`] (each
/// bounds-checked against the window, fail-closed) — the raw window pointer is never exposed, so a C
/// callback gets exactly the §7 confinement the built-in `Stream`/`Memory` caps do. The wrapped
/// borrow is live only during the callback; retaining the pointer past it is a use-after-free.
///
/// Holds a raw `*mut dyn GuestMem` (no lifetime param, so it appears in the C ABI as a plain opaque
/// pointer). The pointee is the live window borrow for the in-flight callback; the accessors below
/// reconstitute it only while C is calling back into us (single-threaded w.r.t. this borrow).
pub struct TemenGuestMem {
    mem: *mut dyn GuestMem,
}

/// Copy `len` bytes from guest window offset `ptr` into `dst`. Returns [`TEMEN_OK`] on success, or
/// [`TEMEN_ERR_FAILED`] (nothing copied) if `mem`/`dst` is null or `[ptr, ptr+len)` is not wholly within
/// the window — the same bounds check the built-in capabilities apply (fail-closed, never an over-read).
///
/// # Safety
/// `mem` is an `TemenGuestMem*` handed to the current [`TemenHostProc`] call (or null); `dst` points to at
/// least `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn temen_guest_read(
    mem: *const TemenGuestMem,
    ptr: u64,
    dst: *mut u8,
    len: usize,
) -> i32 {
    let Some(shim) = mem.as_ref() else {
        return TEMEN_ERR_FAILED;
    };
    if dst.is_null() {
        return TEMEN_ERR_FAILED;
    }
    let m = &*shim.mem;
    match m.read_bytes(ptr, len as u64) {
        Some(bytes) => {
            // `read_bytes` guarantees `bytes.len() == len` on success.
            ptr::copy_nonoverlapping(bytes.as_ptr(), dst, len);
            TEMEN_OK
        }
        None => TEMEN_ERR_FAILED,
    }
}

/// Copy `len` bytes from `src` into the guest window at offset `ptr`. Returns [`TEMEN_OK`] on success, or
/// [`TEMEN_ERR_FAILED`] (nothing written) if `mem`/`src` is null or `[ptr, ptr+len)` is not a wholly
/// in-window, writable range (a read-only / unmapped page fails closed, exactly like the built-ins).
///
/// # Safety
/// `mem` is an `TemenGuestMem*` handed to the current [`TemenHostProc`] call (or null); `src` points to at
/// least `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn temen_guest_write(
    mem: *mut TemenGuestMem,
    ptr: u64,
    src: *const u8,
    len: usize,
) -> i32 {
    let Some(shim) = mem.as_mut() else {
        return TEMEN_ERR_FAILED;
    };
    if src.is_null() {
        return TEMEN_ERR_FAILED;
    }
    let data = std::slice::from_raw_parts(src, len);
    let m = &mut *shim.mem;
    match m.write_bytes(ptr, data) {
        Some(()) => TEMEN_OK,
        None => TEMEN_ERR_FAILED,
    }
}

/// A `Send`/`Sync` carrier for the callback's opaque `ctx` so the grant closure can cross threads (a
/// concurrent guest's workers may invoke the cap). The embedder is responsible for `ctx` being safe to
/// use from multiple threads when the guest is concurrent.
#[derive(Clone, Copy)]
struct CtxPtr(*mut c_void);
// SAFETY: opaque to us; thread-safety of the pointee is the embedder's contract (documented).
unsafe impl Send for CtxPtr {}
unsafe impl Sync for CtxPtr {}

/// Create an empty capability registry.
#[no_mangle]
pub extern "C" fn temen_imports_new() -> *mut TemenImports {
    guard_ptr(|| Ok(Box::into_raw(Box::new(TemenImports(Imports::new())))))
}

unsafe fn provide(imports: *mut TemenImports, name: *const c_char, cap: HostCap) -> i32 {
    guard_status(|| {
        let imports = imports.as_mut().ok_or("null imports")?;
        if name.is_null() {
            return Err("null name".into());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "name is not valid UTF-8".to_string())?;
        // `provide` takes `self` by value (builder); swap through a temporary.
        let reg = std::mem::take(&mut imports.0);
        imports.0 = reg.provide(name, cap);
        Ok(())
    })
}

/// Bind `name` to a writable `Stream` (stdout).
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn temen_imports_provide_stdout(
    i: *mut TemenImports,
    name: *const c_char,
) -> i32 {
    provide(i, name, HostCap::stdout())
}
/// Bind `name` to a readable `Stream` (stdin).
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn temen_imports_provide_stdin(
    i: *mut TemenImports,
    name: *const c_char,
) -> i32 {
    provide(i, name, HostCap::stdin())
}
/// Bind `name` to the `Exit` capability.
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn temen_imports_provide_exit(
    i: *mut TemenImports,
    name: *const c_char,
) -> i32 {
    provide(i, name, HostCap::exit())
}
/// Bind `name` to the `Clock` capability.
///
/// # Safety
/// `i` is a live registry handle and `name` a valid NUL-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn temen_imports_provide_clock(
    i: *mut TemenImports,
    name: *const c_char,
) -> i32 {
    provide(i, name, HostCap::clock())
}

/// Bind `name` to a **host-defined** capability implemented by the C callback `f` (operation `op`,
/// opaque `ctx`). The guest reaches it as `call.sym "<name>"`.
///
/// # Safety
/// `i` is a live registry; `name` a valid C string; `f` a valid function pointer for the lifetime of
/// any instance built from this registry; `ctx` valid for that lifetime (and thread-safe if the guest
/// is concurrent).
#[no_mangle]
pub unsafe extern "C" fn temen_imports_provide_host_proc(
    i: *mut TemenImports,
    name: *const c_char,
    op: u32,
    f: TemenHostProc,
    ctx: *mut c_void,
) -> i32 {
    let ctx = CtxPtr(ctx);
    // `make` is called once per backend host; each builds a fresh `HostProc` that trampolines into `f`.
    let cap = HostCap::host_proc(op, move || -> HostProc {
        let ctx = ctx;
        Box::new(
            move |op, args, mem, _minter: Option<&mut dyn temen_interp::RegionMinter>| {
                // Force whole-`ctx` capture (the `Send`/`Sync` wrapper), not the disjoint `ctx.0` field
                // (a bare `*mut c_void`, which isn't `Send`) — Rust 2021 edition capture.
                let ctx = ctx;
                let mut buf = [0i64; TEMEN_MAX_RESULTS];
                // Wrap the guest window (if any) so the C callback can read/write it bounds-checked, via
                // `temen_guest_read`/`temen_guest_write`. The shim lives on this stack frame for the call only;
                // the pointer it hands C is dangling the instant `f` returns (documented contract).
                let mut shim = mem.map(|m| TemenGuestMem {
                    // SAFETY: erase the borrow's lifetime to carry it through the opaque C handle. The
                    // pointer is dereferenced (by `temen_guest_read`/`write`) only during this in-flight
                    // callback, while `m`'s borrow is live and otherwise untouched — no aliasing.
                    mem: unsafe { std::mem::transmute::<&mut dyn GuestMem, *mut dyn GuestMem>(m) },
                });
                let mem_ptr = shim
                    .as_mut()
                    .map_or(ptr::null_mut(), |s| s as *mut TemenGuestMem);
                let n = f(
                    ctx.0,
                    op,
                    args.as_ptr(),
                    args.len(),
                    buf.as_mut_ptr(),
                    buf.len(),
                    mem_ptr,
                );
                if n < 0 {
                    return Err(Trap::CapFault);
                }
                let n = (n as usize).min(buf.len());
                Ok(buf[..n].to_vec())
            },
        )
    });
    provide(i, name, cap)
}

/// Free a registry handle (only if it was *not* consumed by `temen_instantiate_with_imports`).
///
/// # Safety
/// `i` is a live registry handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_imports_free(i: *mut TemenImports) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

// ----------------------------------------------------------------------------
// Instantiate (consume the module / imports)
// ----------------------------------------------------------------------------

/// Instantiate `m` under the fixed §3e powerbox (resolve imports via the reference host policy, then
/// verify). **Consumes `m`** (do not use or free it afterward). `NULL` on failure.
///
/// # Safety
/// `m` is a live module handle from this library.
#[no_mangle]
pub unsafe extern "C" fn temen_instantiate(m: *mut TemenModule) -> *mut TemenInstance {
    guard_ptr(|| {
        if m.is_null() {
            return Err("temen_instantiate: null module".into());
        }
        let module = Box::from_raw(m).0;
        let inst = instantiate(module)?;
        Ok(Box::into_raw(Box::new(TemenInstance(inst))))
    })
}

/// Instantiate `m` against the name-keyed registry `imports` (wasm-style binding). **Consumes both
/// `m` and `imports`** (do not use or free them afterward). `NULL` on failure (e.g. an unbound import).
///
/// # Safety
/// `m` and `imports` are live handles from this library.
#[no_mangle]
pub unsafe extern "C" fn temen_instantiate_with_imports(
    m: *mut TemenModule,
    imports: *mut TemenImports,
) -> *mut TemenInstance {
    guard_ptr(|| {
        if m.is_null() || imports.is_null() {
            return Err("temen_instantiate_with_imports: null module or imports".into());
        }
        let module = Box::from_raw(m).0;
        let reg = Box::from_raw(imports).0;
        let inst = instantiate_with_imports(module, reg)?;
        Ok(Box::into_raw(Box::new(TemenInstance(inst))))
    })
}

/// Free an instance handle.
///
/// # Safety
/// `i` is a live instance handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_instance_free(i: *mut TemenInstance) {
    if !i.is_null() {
        drop(Box::from_raw(i));
    }
}

// ----------------------------------------------------------------------------
// Memory-access hooks (instrument the module; observe every guest access)
// ----------------------------------------------------------------------------

/// One guest memory access, handed to an [`TemenMemHook`] **before** the access executes. `kind` is one
/// of the `TEMEN_MEM_*` constants; the other fields are interpreted per that kind (see the constants).
#[repr(C)]
pub struct TemenMemEvent {
    pub kind: i32,
    /// Scalar/atomic: effective guest address. `COPY`/`FILL`: destination address.
    pub addr: u64,
    /// `COPY`: source address. Otherwise `0`.
    pub src: u64,
    /// Scalar/atomic: access width in bytes. `COPY`/`FILL`: span length in bytes.
    pub size: u64,
}

/// A memory-access hook callback. Invoked before each guest access with a flattened [`TemenMemEvent`]
/// (valid only for the call — do not retain the pointer). Return `0` to allow the access; return
/// non-zero to **veto** it — the run aborts with a capability trap, identically on every backend.
/// `ctx` is the opaque pointer registered alongside the callback.
pub type TemenMemHook = extern "C" fn(ctx: *mut c_void, ev: *const TemenMemEvent) -> i32;

/// Flatten a [`MemEvent`] into the C-ABI [`TemenMemEvent`].
fn c_mem_event(ev: MemEvent) -> TemenMemEvent {
    let scalar = |kind, addr, width: u32| TemenMemEvent {
        kind,
        addr,
        src: 0,
        size: width as u64,
    };
    match ev {
        MemEvent::Load { addr, width } => scalar(TEMEN_MEM_LOAD, addr, width),
        MemEvent::Store { addr, width } => scalar(TEMEN_MEM_STORE, addr, width),
        MemEvent::AtomicLoad { addr, width } => scalar(TEMEN_MEM_ATOMIC_LOAD, addr, width),
        MemEvent::AtomicStore { addr, width } => scalar(TEMEN_MEM_ATOMIC_STORE, addr, width),
        MemEvent::AtomicRmw { addr, width } => scalar(TEMEN_MEM_ATOMIC_RMW, addr, width),
        MemEvent::AtomicCmpxchg { addr, width } => scalar(TEMEN_MEM_ATOMIC_CMPXCHG, addr, width),
        MemEvent::Copy { dst, src, len } => TemenMemEvent {
            kind: TEMEN_MEM_COPY,
            addr: dst,
            src,
            size: len,
        },
        MemEvent::Fill { dst, len } => TemenMemEvent {
            kind: TEMEN_MEM_FILL,
            addr: dst,
            src: 0,
            size: len,
        },
    }
}

/// Opt `i` into **memory-access hooks**: instrument its module so every guest memory access (loads,
/// stores, v128, atomics, `mem.copy`/`move`/`fill`) calls `hook` before it executes, then re-verify.
/// **Consumes `i`** (do not use or free it afterward) and returns a new, hooked instance handle — run
/// it on any backend with `temen_instance_run`/`temen_instance_run_diff`. `NULL` on failure (e.g. the
/// instrumented module failed re-verification; see [`temen_last_error`]).
///
/// The un-hooked path is untouched — a program that never opts in pays nothing. A hooked run executes
/// more instructions, so give it more fuel than the pristine module. `hook` observes and may veto
/// (non-zero return aborts the run); it cannot rewrite values or addresses.
///
/// # Safety
/// `i` is a live instance handle from this library; `hook` is a valid function pointer for the
/// lifetime of the returned instance; `ctx` is valid for that lifetime (and thread-safe if the guest
/// is concurrent).
#[no_mangle]
pub unsafe extern "C" fn temen_instance_with_mem_hooks(
    i: *mut TemenInstance,
    hook: TemenMemHook,
    ctx: *mut c_void,
) -> *mut TemenInstance {
    guard_ptr(|| {
        if i.is_null() {
            return Err("temen_instance_with_mem_hooks: null instance".into());
        }
        // Consume the instance (`with_mem_hooks` takes `self` by value).
        let inst = Box::from_raw(i).0;
        let ctx = CtxPtr(ctx);
        // `make` is called once per backend host; each builds a fresh handler that trampolines into
        // the C callback. `hook` (a fn pointer) and `ctx` are `Copy`, so `make` stays `Fn`.
        let hooked = inst.with_mem_hooks(move || -> MemHookFn {
            let ctx = ctx;
            Box::new(move |ev| {
                let ctx = ctx;
                let cev = c_mem_event(ev);
                if hook(ctx.0, &cev as *const TemenMemEvent) != 0 {
                    return Err(Trap::CapFault);
                }
                Ok(())
            })
        })?;
        Ok(Box::into_raw(Box::new(TemenInstance(hooked))))
    })
}

// ----------------------------------------------------------------------------
// Run config (C-ABI mirror of `RunConfig`/`Limits`)
// ----------------------------------------------------------------------------

/// Run configuration. A `NULL` pointer means all defaults. `*_set` flags select whether the paired
/// field is applied (else the default is used); `max_fibers`/`max_vcpus` of `0` also mean "default".
#[repr(C)]
pub struct TemenRunConfig {
    /// Per-op fuel for the interpreters (applied iff `fuel_set`). Ignored by the JIT.
    pub fuel: u64,
    pub fuel_set: i32,
    /// JIT detect-and-kill deadline in milliseconds (applied iff `deadline_set`). Ignored by interps.
    pub deadline_ms: u64,
    pub deadline_set: i32,
    /// §15 spawn quota (`0` ⇒ default).
    pub max_fibers: usize,
    pub max_vcpus: usize,
    /// Guest stdin bytes (`NULL`/`0` ⇒ empty).
    pub stdin: *const u8,
    pub stdin_len: usize,
    /// Linear-memory window `size_log2` override (applied iff `memory_set`).
    pub memory_size_log2: u8,
    pub memory_set: i32,
}

/// Translate the C config (possibly null) into a Rust [`RunConfig`].
///
/// # Safety
/// `c` is null or points to a valid `TemenRunConfig`; its `stdin`/`stdin_len` describe a readable slice.
unsafe fn run_config(c: *const TemenRunConfig) -> RunConfig {
    let Some(c) = c.as_ref() else {
        return RunConfig::default();
    };
    let d = Limits::default();
    let stdin = if c.stdin.is_null() || c.stdin_len == 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(c.stdin, c.stdin_len).to_vec()
    };
    RunConfig {
        limits: Limits {
            fuel: (c.fuel_set != 0).then_some(c.fuel),
            deadline: (c.deadline_set != 0).then(|| Duration::from_millis(c.deadline_ms)),
            max_fibers: if c.max_fibers != 0 {
                c.max_fibers
            } else {
                d.max_fibers
            },
            max_vcpus: if c.max_vcpus != 0 {
                c.max_vcpus
            } else {
                d.max_vcpus
            },
        },
        stdin,
        memory_size_log2: (c.memory_set != 0).then_some(c.memory_size_log2),
        // argv/env are not part of the current C config surface (the C tests run arg-less programs);
        // an `temen_run_config_set_args`-style setter is a later C-ABI follow-up.
        ..RunConfig::default()
    }
}

fn backend_of(b: i32) -> Result<Backend, String> {
    match b {
        TEMEN_BACKEND_TREEWALK => Ok(Backend::TreeWalk),
        TEMEN_BACKEND_BYTECODE => Ok(Backend::Bytecode),
        TEMEN_BACKEND_JIT => Ok(Backend::Jit),
        other => Err(format!("unknown backend selector {other}")),
    }
}

/// Run the powerbox entry on a single `backend` under `config` (null ⇒ defaults). Returns a run handle
/// (read with `temen_run_*`), or `NULL` on a trap / failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `TemenRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn temen_instance_run(
    i: *mut TemenInstance,
    backend: i32,
    config: *const TemenRunConfig,
) -> *mut TemenRun {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("temen_instance_run: null instance")?;
        let backend = backend_of(backend)?;
        let cfg = run_config(config);
        let run = inst.0.run(backend, &cfg)?;
        Ok(Box::into_raw(Box::new(TemenRun(run))))
    })
}

/// Run the powerbox entry on the tree-walker **and** the JIT under `config`, asserting they agree (the
/// interp == jit oracle). Returns a run handle, or `NULL` on divergence / trap / failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `TemenRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn temen_instance_run_diff(
    i: *mut TemenInstance,
    config: *const TemenRunConfig,
) -> *mut TemenRun {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("temen_instance_run_diff: null instance")?;
        let cfg = run_config(config);
        let run = inst.0.run_diff(&cfg)?;
        Ok(Box::into_raw(Box::new(TemenRun(run))))
    })
}

// ----------------------------------------------------------------------------
// Run results
// ----------------------------------------------------------------------------

fn value_slot(v: &Value) -> i64 {
    match v {
        Value::I32(x) => *x as i64,
        Value::I64(x) => *x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::Ref(x) => *x as i64,
        Value::V128(b) => i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
    }
}

/// The captured stdout bytes (valid until `temen_run_free`); writes `*len`. Returns `NULL` (and `*len`
/// = 0) for a null handle.
///
/// # Safety
/// `r` is a live run handle (or null); `len` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn temen_run_stdout(r: *const TemenRun, len: *mut usize) -> *const u8 {
    bytes_field(r, len, |run| &run.0.stdout)
}

/// The captured stderr bytes (valid until `temen_run_free`); writes `*len`.
///
/// # Safety
/// `r` is a live run handle (or null); `len` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn temen_run_stderr(r: *const TemenRun, len: *mut usize) -> *const u8 {
    bytes_field(r, len, |run| &run.0.stderr)
}

unsafe fn bytes_field(
    r: *const TemenRun,
    len: *mut usize,
    pick: impl FnOnce(&TemenRun) -> &Vec<u8>,
) -> *const u8 {
    match r.as_ref() {
        Some(run) => {
            let v = pick(run);
            if !len.is_null() {
                *len = v.len();
            }
            v.as_ptr()
        }
        None => {
            if !len.is_null() {
                *len = 0;
            }
            ptr::null()
        }
    }
}

/// The outcome kind: [`TEMEN_OUTCOME_RETURNED`] or [`TEMEN_OUTCOME_EXITED`] (or `< 0` for a null handle).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_run_outcome_kind(r: *const TemenRun) -> i32 {
    match r.as_ref() {
        Some(run) => match run.0.outcome {
            Outcome::Returned(_) => TEMEN_OUTCOME_RETURNED,
            Outcome::Exited(_) => TEMEN_OUTCOME_EXITED,
        },
        None => -1,
    }
}

/// The exit code (valid when the outcome kind is [`TEMEN_OUTCOME_EXITED`]; else `0`).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_run_exit_code(r: *const TemenRun) -> i32 {
    match r.as_ref() {
        Some(run) => match run.0.outcome {
            Outcome::Exited(code) => code,
            Outcome::Returned(_) => 0,
        },
        None => 0,
    }
}

/// The number of returned result values (when the outcome kind is [`TEMEN_OUTCOME_RETURNED`]).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_run_result_count(r: *const TemenRun) -> usize {
    match r.as_ref() {
        Some(run) => match &run.0.outcome {
            Outcome::Returned(v) => v.len(),
            Outcome::Exited(_) => 0,
        },
        None => 0,
    }
}

/// The `idx`-th returned value as a raw `i64` slot (floats are bit-reinterpreted; `0` if out of range).
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_run_result(r: *const TemenRun, idx: usize) -> i64 {
    match r.as_ref() {
        Some(run) => match &run.0.outcome {
            Outcome::Returned(v) => v.get(idx).map_or(0, value_slot),
            Outcome::Exited(_) => 0,
        },
        None => 0,
    }
}

/// Free a run handle.
///
/// # Safety
/// `r` is a live run handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_run_free(r: *mut TemenRun) {
    if !r.is_null() {
        drop(Box::from_raw(r));
    }
}

// ----------------------------------------------------------------------------
// Reactor sessions (Phase 6): instantiate once, call exports repeatedly with
// persistent window state.
// ----------------------------------------------------------------------------

/// A live, stateful reactor session (opaque) — the C view of [`temen_run::Session`]. Built by
/// `temen_instance_start`, freed by `temen_session_free`.
pub struct TemenSession(temen_run::Session);

/// Start a reactor session on `backend` under `config` (null ⇒ defaults): grant the powerbox once,
/// run the bootstrap, and keep the window + host live for repeated `temen_session_call_export` calls.
/// Does **not** consume `i` (the instance can start more sessions). Returns `NULL` on failure.
///
/// # Safety
/// `i` is a live instance handle; `config` is null or a valid `TemenRunConfig`.
#[no_mangle]
pub unsafe extern "C" fn temen_instance_start(
    i: *const TemenInstance,
    backend: i32,
    config: *const TemenRunConfig,
) -> *mut TemenSession {
    guard_ptr(|| {
        let inst = i.as_ref().ok_or("temen_instance_start: null instance")?;
        let backend = backend_of(backend)?;
        let cfg = run_config(config);
        let session = inst.0.start(backend, &cfg)?;
        Ok(Box::into_raw(Box::new(TemenSession(session))))
    })
}

/// Call exported function `name` with `n_args` `i64` arguments, writing up to `results_cap` `i64`
/// results into `results` and the actual count into `*n_results`. The window (globals, stash, BSS) and
/// capability handles persist from prior calls. Returns `TEMEN_OK`, or an error status (message in
/// `temen_last_error`). Arguments are passed as raw `i64` slots (interpreted as `i64` values — the
/// common case; floats can be passed by bit pattern).
///
/// # Safety
/// `s` is a live session; `name` a valid C string; `args`/`results` describe readable/writable
/// `n_args`/`results_cap` slots; `n_results` is a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn temen_session_call_export(
    s: *mut TemenSession,
    name: *const c_char,
    args: *const i64,
    n_args: usize,
    results: *mut i64,
    results_cap: usize,
    n_results: *mut usize,
) -> i32 {
    guard_status(|| {
        let s = s
            .as_mut()
            .ok_or("temen_session_call_export: null session")?;
        if name.is_null() {
            return Err("null export name".into());
        }
        let name = CStr::from_ptr(name)
            .to_str()
            .map_err(|_| "export name is not valid UTF-8".to_string())?;
        let arg_slots = if n_args == 0 {
            &[][..]
        } else {
            std::slice::from_raw_parts(args, n_args)
        };
        let vals: Vec<Value> = arg_slots.iter().map(|&x| Value::I64(x)).collect();
        let out = s.0.call_export(name, &vals)?;
        if !n_results.is_null() {
            *n_results = out.len();
        }
        if !results.is_null() && results_cap > 0 {
            let dst = std::slice::from_raw_parts_mut(results, results_cap);
            for (d, v) in dst.iter_mut().zip(&out) {
                *d = value_slot(v);
            }
        }
        Ok(())
    })
}

/// The session's captured stdout so far (valid until the next call / `temen_session_free`); writes `*len`.
///
/// # Safety
/// `s` is a live session (or null); `len` a valid `size_t*` (or null).
#[no_mangle]
pub unsafe extern "C" fn temen_session_stdout(
    s: *const TemenSession,
    len: *mut usize,
) -> *const u8 {
    match s.as_ref() {
        Some(sess) => {
            let out = sess.0.stdout();
            if !len.is_null() {
                *len = out.len();
            }
            out.as_ptr()
        }
        None => {
            if !len.is_null() {
                *len = 0;
            }
            ptr::null()
        }
    }
}

/// Free a session handle.
///
/// # Safety
/// `s` is a live session handle from this library, or null.
#[no_mangle]
pub unsafe extern "C" fn temen_session_free(s: *mut TemenSession) {
    if !s.is_null() {
        drop(Box::from_raw(s));
    }
}

#[cfg(test)]
mod abi_tests;
