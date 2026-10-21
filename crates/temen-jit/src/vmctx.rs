//! The JIT's **`vmctx`** — the per-instance context compiled code reaches through the one context
//! pointer every function already threads (#1768, FORK.md §9.5).
//!
//! Compiled code used to bake each per-instance address in as a constant: the `call.cap` ctx (the
//! domain's powerbox), the §5 kill-path cell, the counted-fuel cell, and the #932 signal-delivery
//! flag + ctx. That tied the machine code to one `Host`, so a fork twin — its own powerbox over the
//! parent's program — could not run its parent's code, and an exec'd image could not reuse a compile.
//! Now each of those addresses lives here, and the code loads it: Wasmtime's `VMContext`, placed where
//! the ABI already had a slot for it. Every function threads a pointer to its instance's context, whose
//! first field is the trap cell, so compiled code raises and checks the cell at offset 0 of the pointer
//! it already has, and a host thunk is handed the same pointer as a `&VmCtx`.
//!
//! What stays baked is what describes the **code**, not the instance: the window geometry (the
//! confinement mask is an immediate on the hot path), the table mask, and whether each check is
//! emitted at all (a compile armed with no kill-path, fuel budget or signal source emits no check,
//! so its code is byte-identical to before). The host functions the code calls (its thunks) it
//! names by symbol, which its link binds to their addresses in the process (`HostFn`, temen#2452).
//! The §12 thread `Domain` lives here too: the `thread.*`, futex and blocking-resume sites load it,
//! and so does the §14 `Nursery` the `Instantiator` sites spawn through (temen#2168), so every
//! spawn of a child runs one compile of its code, and so does the `setjmp` table (temen#2389). The
//! fiber runtime is found through a thread-local each entry sets. So the code names no per-instance
//! object, and each instance stands up runtimes of its own (`CompiledModule::instance`): the code
//! is **instance-independent** — the property a fork twin sharing its parent's code, and a compile
//! cache, rely on (#1825). Only a §22 unit's program, defined into a module after its compile, is
//! more than the compile; such a module is never shared.
//!
//! **The trust argument is the entry.** Code that runs with the wrong vmctx dispatches its `call.cap`s
//! into the wrong powerbox — authority confusion, not a crash. So every entry into compiled code takes
//! a `*mut VmCtx` (never a bare trap cell), and each one names whose instance it is entering: the
//! root's, a child's, a fiber of the running instance (its creator's), a spawned vCPU (its domain's), a
//! handler invoked over a live window (the window's owner). The type makes the omission a compile
//! error; the comment at each site says which instance.
//!
//! **The trap cell is how the domain ended, written once (#2499).** Every vCPU of a domain shares its
//! context, and by invariants 5 and 6 every code the cell can hold — a trap, an `exit`, the end
//! teardown records — ends the whole domain. So the cell is not a per-call status: [`VmCtx::raise`]
//! records an end only if none is recorded (the first end wins), nothing clears it while a run is in
//! it, and every check — the propagation guard after each call, a loop's poll, a park, the run's
//! outcome — reads that one word. The field is private: a host thunk raises and reads, never stores.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicU8, Ordering};

/// The per-instance context (see the module docs). `#[repr(C)]`: compiled code addresses the fields
/// at the fixed offsets below.
#[repr(C)]
pub struct VmCtx {
    /// The trap cell: `0` while the domain runs; once it ended, the low 32 bits a [`crate::TrapKind`] /
    /// `EXIT_CODE` / internal code, the high 32 bits an exit code. Written once, by [`Self::raise`].
    /// **Offset 0** — a pointer to the vmctx is a pointer to the trap cell.
    trap: AtomicI64,
    /// The `call.cap` thunk's opaque ctx: the instance's powerbox (`*mut Host`, `*const Mutex<Host>`,
    /// or a child's own), handed to every `call.cap` as the thunk's first argument.
    pub cap_ctx: *mut c_void,
    /// The §5 kill-path interrupt cell a compile with epoch checks polls (null when not armed).
    pub epoch: *const AtomicU64,
    /// The counted-fuel cell a compile with fuel checks charges (null when not armed).
    pub fuel: *mut crate::FuelCell,
    /// #932 — the host `armed` flag a compile with signal checks polls (null when not armed).
    pub sig_armed: *const AtomicBool,
    /// #932 — the signal-delivery ctx handed to the take/return thunks (null when not armed).
    pub sig_ctx: *mut c_void,
    /// The embedder's own per-instance state, opaque to the JIT and never read by compiled code:
    /// what a `call.cap` thunk — which is handed this context — needs beyond the powerbox
    /// (temen-run's process-tree membership, #1768). Null when the embedder keeps none.
    pub embedder: *mut c_void,
    /// The §12 thread domain (`os_thread_rt::Domain`) the instance's `thread.*`, futex and blocking
    /// `cont.resume` sites run on: its own, or for a §14 child its parent's. Null when it has none.
    pub sched: *const c_void,
    /// temen#2168 — the §14 `Nursery` (`instantiator_rt::Nursery`) the instance's `Instantiator`
    /// sites spawn and join through: its own, which holds its children. Null when it has none.
    pub nursery: *const c_void,
    /// temen#2389 — the `setjmp` table (`setjmp_rt::SetjmpRuntime`) the instance's `SetJmp` and
    /// `LongJmp` sites keep their host `jmp_buf`s in. Null when the code uses neither.
    pub setjmp: *const c_void,
    /// #2164 — a demand-paged child's **access table** (`temen_ir::PAGE_READ`), which a paged
    /// compile tests before each access. Null when the instance is not paged.
    pub pages: *const AtomicU8,
}

/// The per-instance addresses a [`VmCtx`] is filled from — everything but the trap cell, which each
/// entry starts at `0`. `Copy`, so a compile can keep its default instance's addresses and each run
/// can mint a fresh context from them.
#[derive(Clone, Copy, Debug)]
pub struct InstanceAddrs {
    pub cap_ctx: *mut c_void,
    pub epoch: *const AtomicU64,
    pub fuel: *mut crate::FuelCell,
    pub sig_armed: *const AtomicBool,
    pub sig_ctx: *mut c_void,
    pub embedder: *mut c_void,
    pub sched: *const c_void,
    pub nursery: *const c_void,
    pub setjmp: *const c_void,
    pub pages: *const AtomicU8,
}

impl InstanceAddrs {
    /// No powerbox, no kill-path, no fuel, no signals, no embedder state, no thread domain, no
    /// nursery, no `setjmp` table, no paging.
    pub const NONE: InstanceAddrs = InstanceAddrs {
        cap_ctx: core::ptr::null_mut(),
        epoch: core::ptr::null(),
        fuel: core::ptr::null_mut(),
        sig_armed: core::ptr::null(),
        sig_ctx: core::ptr::null_mut(),
        embedder: core::ptr::null_mut(),
        sched: core::ptr::null(),
        nursery: core::ptr::null(),
        setjmp: core::ptr::null(),
        pages: core::ptr::null(),
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
            sched: a.sched,
            nursery: a.nursery,
            setjmp: a.setjmp,
            pages: a.pages,
        }
    }

    /// End this context's domain with `code` — a trap, an `exit`, or an internal end — unless an end
    /// is already recorded: the first end wins, and a later one changes nothing (#2499). Every vCPU
    /// sharing the context sees it at its next check.
    pub fn raise(&self, code: i64) {
        debug_assert_ne!(code, 0, "an end is never 0");
        let _ = self
            .trap
            .compare_exchange(0, code, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// The code that ended this context's domain, or `None` while it runs.
    pub fn ended(&self) -> Option<i64> {
        match self.trap.load(Ordering::Relaxed) {
            0 => None,
            code => Some(code),
        }
    }

    /// The context at `vm`: the address compiled code threads, as a `u64`, to the runtime functions
    /// it calls.
    ///
    /// # Safety
    /// `vm` is the address of a context live for the returned borrow.
    pub(crate) unsafe fn at<'a>(vm: u64) -> &'a VmCtx {
        &*(vm as *const VmCtx)
    }
}

// SAFETY: every pointer field is the address of a host-owned object that outlives each run entering
// through this context (the caller's contract at every entry), and compiled code only *reads* them; the
// one field written concurrently — the trap cell, by sibling vCPUs of one instance and a parent ending
// it — is atomic. So one context may be shared by the threads of one instance (its spawned vCPUs, a
// §14 task's worker).
unsafe impl Send for VmCtx {}
unsafe impl Sync for VmCtx {}

/// Field offsets compiled code loads at (`#[repr(C)]` makes them fixed; `offset_of!` keeps them in
/// step with the struct).
pub(crate) const CAP_CTX: i32 = core::mem::offset_of!(VmCtx, cap_ctx) as i32;
pub(crate) const EPOCH: i32 = core::mem::offset_of!(VmCtx, epoch) as i32;
pub(crate) const FUEL: i32 = core::mem::offset_of!(VmCtx, fuel) as i32;
pub(crate) const SIG_ARMED: i32 = core::mem::offset_of!(VmCtx, sig_armed) as i32;
pub(crate) const SIG_CTX: i32 = core::mem::offset_of!(VmCtx, sig_ctx) as i32;
pub(crate) const SCHED: i32 = core::mem::offset_of!(VmCtx, sched) as i32;
pub(crate) const NURSERY: i32 = core::mem::offset_of!(VmCtx, nursery) as i32;
pub(crate) const SETJMP: i32 = core::mem::offset_of!(VmCtx, setjmp) as i32;
pub(crate) const PAGES: i32 = core::mem::offset_of!(VmCtx, pages) as i32;

// The trap cell must be the first field: every trap-cell reader dereferences the vmctx pointer itself.
const _: () = assert!(core::mem::offset_of!(VmCtx, trap) == 0);

#[cfg(test)]
mod tests {
    use super::{InstanceAddrs, VmCtx};

    /// #2499 — a domain ends once: the first end raised stands, and a later one changes nothing.
    #[test]
    fn the_first_end_wins() {
        let vm = VmCtx::new(InstanceAddrs::NONE);
        assert_eq!(vm.ended(), None);
        vm.raise(7);
        vm.raise(9);
        assert_eq!(vm.ended(), Some(7));
    }
}
