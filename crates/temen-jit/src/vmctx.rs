//! The JIT's **`vmctx`** — the per-instance context compiled code reaches through the one context
//! pointer every function already threads (#1768, FORK.md §9.5).
//!
//! Compiled code used to bake each per-instance address in as a constant: the `call.cap` ctx (the
//! domain's powerbox), the §5 kill-path cell, the counted-fuel cell, and the #932 signal-delivery
//! flag + ctx. That tied the machine code to one `Host`, so a fork twin — its own powerbox over the
//! parent's program — could not run its parent's code, and an exec'd image could not reuse a compile.
//! Now each of those addresses lives here, and the code loads it: Wasmtime's `VMContext`, placed where
//! the ABI already had a slot for it. Every function threads a pointer to its instance's trap cell;
//! that cell is this struct's first field, so the pointer *is* the vmctx, and every reader of the trap
//! cell (a trap store, the post-call propagation check, a host thunk's `*trap_out = …`) is unchanged.
//!
//! What stays baked is what describes the **code**, not the instance: the window geometry (the
//! confinement mask is an immediate on the hot path), the table mask, the thunk *functions* (process
//! globals), and whether each check is emitted at all (a compile armed with no kill-path, fuel budget
//! or signal source emits no check, so its code is byte-identical to before). A compile's per-instance
//! runtimes (the thread
//! `Domain`, the §14 `Nursery`, the `setjmp` table, a §22 unit's program) are still baked; a module
//! that has none is **instance-independent** — the property a fork twin sharing its parent's code,
//! and a compile cache, would rely on (#1825).
//!
//! **The trust argument is the entry.** Code that runs with the wrong vmctx dispatches its `call.cap`s
//! into the wrong powerbox — authority confusion, not a crash. So every entry into compiled code takes
//! a `*mut VmCtx` (never a bare trap cell), and each one names whose instance it is entering: the
//! root's, a child's, a fiber of the running instance (its creator's), a spawned vCPU (its domain's), a
//! handler invoked over a live window (the window's owner). The type makes the omission a compile
//! error; the comment at each site says which instance.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64};

/// The per-instance context (see the module docs). `#[repr(C)]`: compiled code addresses the fields
/// at the fixed offsets below.
#[repr(C)]
pub struct VmCtx {
    /// The trap cell: `0` = running / ok; the low 32 bits a [`crate::TrapKind`] / `EXIT_CODE` /
    /// internal code, the high 32 bits an exit code. **Offset 0** — a pointer to the vmctx is a
    /// pointer to the trap cell.
    pub trap: AtomicI64,
    /// The `call.cap` thunk's opaque ctx: the instance's powerbox (`*mut Host`, `*const Mutex<Host>`,
    /// or a child's own), handed to every `call.cap` as the thunk's first argument.
    pub cap_ctx: *mut c_void,
    /// The §5 kill-path interrupt cell a compile with epoch checks polls (null when not armed).
    pub epoch: *const AtomicU64,
    /// The counted-fuel cell a compile with fuel checks charges (null when not armed).
    pub fuel: *mut u64,
    /// #932 — the host `armed` flag a compile with signal checks polls (null when not armed).
    pub sig_armed: *const AtomicBool,
    /// #932 — the signal-delivery ctx handed to the take/return thunks (null when not armed).
    pub sig_ctx: *mut c_void,
    /// The embedder's own per-instance state, opaque to the JIT and never read by compiled code:
    /// what a `call.cap` thunk — which is handed this context as its `trap_out` — needs beyond the
    /// powerbox (temen-run's process-tree membership, #1768). Null when the embedder keeps none.
    pub embedder: *mut c_void,
}

/// The per-instance addresses a [`VmCtx`] is filled from — everything but the trap cell, which each
/// entry starts at `0`. `Copy`, so a compile can keep its default instance's addresses and each run
/// can mint a fresh context from them.
#[derive(Clone, Copy, Debug)]
pub struct InstanceAddrs {
    pub cap_ctx: *mut c_void,
    pub epoch: *const AtomicU64,
    pub fuel: *mut u64,
    pub sig_armed: *const AtomicBool,
    pub sig_ctx: *mut c_void,
    pub embedder: *mut c_void,
}

impl InstanceAddrs {
    /// No powerbox, no kill-path, no fuel, no signals, no embedder state.
    pub const NONE: InstanceAddrs = InstanceAddrs {
        cap_ctx: core::ptr::null_mut(),
        epoch: core::ptr::null(),
        fuel: core::ptr::null_mut(),
        sig_armed: core::ptr::null(),
        sig_ctx: core::ptr::null_mut(),
        embedder: core::ptr::null_mut(),
    };
}

impl VmCtx {
    /// A fresh context (trap cell `0`) for the instance at `a`.
    pub fn new(a: InstanceAddrs) -> VmCtx {
        VmCtx {
            trap: AtomicI64::new(0),
            cap_ctx: a.cap_ctx,
            epoch: a.epoch,
            fuel: a.fuel,
            sig_armed: a.sig_armed,
            sig_ctx: a.sig_ctx,
            embedder: a.embedder,
        }
    }

    /// The context a `call.cap` thunk was handed as its `trap_out` (a vmctx's field 0).
    ///
    /// # Safety
    /// `trap_out` is the `trap_out` argument of a thunk call made by compiled code, whose instance
    /// context is live for the call.
    pub unsafe fn of_trap_out<'a>(trap_out: *mut i64) -> &'a VmCtx {
        &*(trap_out as *const VmCtx)
    }

    /// The trap cell's address — what the FFI entry shims and the host thunks spell `trap_out`.
    pub fn trap_ptr(&self) -> *mut i64 {
        self.trap.as_ptr()
    }
}

// SAFETY: every pointer field is the address of a host-owned object that outlives each run entering
// through this context (the caller's contract at every entry), and compiled code only *reads* them; the
// one field written concurrently — the trap cell, by sibling vCPUs of one instance — is atomic. So one
// context may be shared by the threads of one instance (its spawned vCPUs, a §14 task's worker).
unsafe impl Send for VmCtx {}
unsafe impl Sync for VmCtx {}

/// Field offsets compiled code loads at (`#[repr(C)]` makes them fixed; `offset_of!` keeps them in
/// step with the struct).
pub(crate) const CAP_CTX: i32 = core::mem::offset_of!(VmCtx, cap_ctx) as i32;
pub(crate) const EPOCH: i32 = core::mem::offset_of!(VmCtx, epoch) as i32;
pub(crate) const FUEL: i32 = core::mem::offset_of!(VmCtx, fuel) as i32;
pub(crate) const SIG_ARMED: i32 = core::mem::offset_of!(VmCtx, sig_armed) as i32;
pub(crate) const SIG_CTX: i32 = core::mem::offset_of!(VmCtx, sig_ctx) as i32;

// The trap cell must be the first field: every trap-cell reader dereferences the vmctx pointer itself.
const _: () = assert!(core::mem::offset_of!(VmCtx, trap) == 0);
