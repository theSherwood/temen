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
//! The §12 thread `Domain` lives here too, which the `thread.*`, futex and blocking-resume thunks
//! act in, and so does the §14 `Nursery` the `Instantiator` thunks spawn through (temen#2168), so
//! every spawn of a child runs one compile of its code, and so does the `setjmp` table (temen#2389).
//! The fiber runtime is found through a thread-local each entry sets. So the code names no
//! per-instance object, and each instance stands up runtimes of its own (`CompiledModule::instance`):
//! the code is **instance-independent** — the property a fork twin sharing its parent's code, and a
//! compile cache, rely on (#1825). Only a §22 unit's program, defined into a module after its
//! compile, is more than the compile; such a module is never shared.
//!
//! **A host function takes from the code only what the code alone can supply (#2534).** It is
//! handed the context, which the machine-code validator checks is this one, and reads here every
//! runtime object it acts on and the run's [`Window`] (the window base and reservation it confines
//! guest offsets to, and the function table), which the runtime records when it makes the context.
//! A backend bug that passed the wrong register could otherwise move the window a host function
//! trusts. Compiled code never reads the runtime objects itself, and the validator rejects a load
//! of one.
//!
//! **The trust argument is the entry.** Code that runs with the wrong vmctx dispatches its `call.cap`s
//! into the wrong powerbox — authority confusion, not a crash. So every entry into compiled code takes
//! a `*mut VmCtx` (never a bare trap cell), and each one names whose instance it is entering: the
//! root's, a child's, a fiber of the running instance (its creator's), a spawned vCPU (its domain's), a
//! handler invoked over a live window (the window's owner). The type makes the omission a compile
//! error; the comment at each site says which instance.
//!
//! **The domain's end is one word, written once (#2499).** By invariants 5 and 6 every code an end
//! can hold — a trap, an `exit`, the end teardown records — ends the whole domain. So the end is not a
//! per-call status: [`VmCtx::raise`] records one only if none is recorded (the first end wins), and
//! nothing clears it while a run is in it. The word belongs to the domain's root context; every
//! context of the domain points at it ([`END`]).
//!
//! **Each vCPU runs on a context of its own (#2488)**, made by [`VmCtx::vcpu`], so that it charges a
//! fuel cell of its own: the vCPUs of a domain once shared one, and every fuel check stored to the
//! same cache line. So the check after each call reads the context's **stop word** at offset 0, a
//! copy of the domain's end. The copy is written only after the end: by a raise, by a loop's poll
//! that read the end, and by the check after a host call, which reads the end itself because the host
//! may have seen another vCPU end the domain. The check after a guest call stays one load. A root
//! context's stop word is the root vCPU's. The fields are private: a host thunk raises and reads,
//! never stores.

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, Ordering};

/// The per-instance context (see the module docs). `#[repr(C)]`: compiled code addresses the fields
/// at the fixed offsets below.
#[repr(C)]
pub struct VmCtx {
    /// The stop word: `0` while the code running on this context may go on; once the domain ended,
    /// a copy of its end. **Offset 0** — a pointer to the vmctx is a pointer to the stop word.
    trap: AtomicI64,
    /// The `call.cap` thunk's opaque ctx: the instance's powerbox (`*mut Host`, `*const Mutex<Host>`,
    /// or a child's own), handed to every `call.cap` as the thunk's first argument.
    pub cap_ctx: *mut c_void,
    /// The §5 kill door whose cell a compile with epoch checks polls (null when not armed). The cell
    /// is the door's first field, so compiled code loads it through this pointer as a `u64`.
    pub epoch: *const crate::Interrupt,
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
    /// The domain's end: `0` while it runs; once it ended, the low 32 bits a [`crate::TrapKind`] /
    /// `EXIT_CODE` / internal code, the high 32 bits an exit code. Written once, by [`Self::raise`]
    /// (or compiled code's raise). The root context's [`Self::owned_end`].
    end: *const AtomicI64,
    /// #2594 — the compile service of the code, when it compiles lazily (`lazy::Lazy`): what a lazy
    /// entry's `resolve` asks for its function. A property of the code, not the instance, so the
    /// module sets it in every instance it makes; null for code compiled whole. Never read by
    /// compiled code.
    pub lazy: *const c_void,
    /// The domain's end word, owned by its root context ([`Self::new`]); `None` for a vCPU's
    /// context ([`Self::vcpu`]), which points at its root's. Never read by compiled code.
    owned_end: Option<Box<AtomicI64>>,
    /// #2742 — the vCPU's serve site, made at its first serve op ([`crate::ServeSite::of`]), so it
    /// goes with the vCPU. Never read by compiled code.
    #[cfg(fiber_rt)]
    pub(crate) serve_site: std::sync::OnceLock<std::sync::Arc<crate::child_exec::ServeSite>>,
    /// #2742 — the window this is the handler context of ([`crate::child_exec::Window::vm`]), null
    /// for every other context: what a park or a live call of code running as a handler finds its
    /// window's gate by. Never read by compiled code.
    #[cfg(fiber_rt)]
    pub(crate) handler_window: std::sync::atomic::AtomicPtr<crate::child_exec::Window>,
    /// #2742 — the domain's serve handlers not yet ended that its end must wait for: those the pool
    /// holds, and those running on a caller's stack. Shared by every context of the domain, as its
    /// end word is; owned by its root context ([`Self::owned_handlers`]). Never read by compiled
    /// code.
    #[cfg(fiber_rt)]
    handlers: *const std::sync::atomic::AtomicUsize,
    #[cfg(fiber_rt)]
    owned_handlers: Option<Box<std::sync::atomic::AtomicUsize>>,
    /// #2534 — the window and function table this run's code runs over, for the host functions,
    /// which read them here rather than take them from the code. Never read by compiled code, so
    /// it sits past every field the code loads, which keep their short offsets.
    window: Window,
    /// #2319 — the `call.cap` thunk's own state on this context, opaque to the JIT
    /// ([`Self::thunk_state`]): made by the thunk's first call, dropped with the context. Never read
    /// by compiled code. #2847 — beside it, the type it was made as.
    thunk_state: std::sync::OnceLock<(core::any::TypeId, Box<dyn core::any::Any + Send + Sync>)>,
    /// #2958 — on a vCPU that runs as a task of executor 2's scheduling core, where its §12 ops go
    /// ([`crate::kernel_rt`]): set by `KernelRun::run_vcpu` for its guarded call, `None` on every
    /// other context. Never read by compiled code, so it sits past every field the code loads.
    pub(crate) kernel: Option<core::ptr::NonNull<dyn crate::kernel_rt::KernelPark>>,
}

/// The per-instance addresses a [`VmCtx`] is filled from — everything but the trap cell, which each
/// entry starts at `0`. `Copy`, so a compile can keep its default instance's addresses and each run
/// can mint a fresh context from them.
#[derive(Clone, Copy, Debug)]
pub struct InstanceAddrs {
    pub cap_ctx: *mut c_void,
    pub epoch: *const crate::Interrupt,
    pub fuel: *mut crate::FuelCell,
    pub sig_armed: *const AtomicBool,
    pub sig_ctx: *mut c_void,
    pub embedder: *mut c_void,
    pub sched: *const c_void,
    pub nursery: *const c_void,
    pub setjmp: *const c_void,
    pub pages: *const AtomicU8,
    /// Set by the module from its own code ([`VmCtx::lazy`]): whatever an embedder passes is
    /// replaced.
    pub lazy: *const c_void,
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
        lazy: core::ptr::null(),
    };
}

/// #2534 — the window and function table a run's code runs over, recorded in its [`VmCtx`] so that
/// a host function derives them from the context compiled code hands it (which the validator
/// checks) instead of taking them as arguments the code computes. The base and the table are the
/// run's, the ones its entry is handed; the sizes are the geometry its compile baked in.
#[derive(Clone, Copy, Debug)]
pub struct Window {
    /// The window's base: what the entry is handed as `mem_base`. Null when there is no window.
    pub base: *mut u8,
    /// The declared window's bytes, `mapped`: what a `call.cap` is handed as its size.
    pub mapped: u64,
    /// The reservation, `mask + 1`, which bounds every confined offset. `0` when there is no window.
    pub reserved: u64,
    /// #2164 — how many pages a demand-paged child's access table covers; `0` when unpaged.
    pub pages: u64,
    /// #2164 — a demand-paged child's page-miss thunk, which asks its pager for a page its access
    /// table refuses; `None` when unpaged.
    pub page_miss: Option<crate::PageMissThunk>,
    /// The function table the entry is handed as `fn_table_base`.
    pub fn_table: *const u8,
}

impl Window {
    /// No window and no function table.
    pub const NONE: Window = Window {
        base: core::ptr::null_mut(),
        mapped: 0,
        reserved: 0,
        pages: 0,
        page_miss: None,
        fn_table: core::ptr::null(),
    };

    /// The host address of the `len ≥ 1` bytes at window offset `off`, if they lie in the
    /// reservation: what a host function reads or writes in place of a guest access. `None` for an
    /// empty span or window, or a span past the reservation, which the caller faults as a guest
    /// access there would. The bound is temen-mask's `span_checked`, the one definition of the
    /// span-OOB arithmetic the interpreters confine their bulk ops with.
    pub fn span(&self, off: u64, len: u64) -> Option<*mut u8> {
        if len == 0 || self.reserved == 0 {
            return None;
        }
        // `reserved` is `mask + 1`, a power of two.
        temen_mask::Window::new(self.reserved.trailing_zeros() as u8)
            .span_checked(off, len)
            .map(|off| self.base.wrapping_add(off as usize))
    }
}

impl VmCtx {
    /// A fresh root context for the instance at `a`, running over window `w`: a domain of its own,
    /// not yet ended.
    pub fn new(a: InstanceAddrs, w: Window) -> VmCtx {
        // Boxed, so the word stays put when the context moves.
        let owned_end = Box::new(AtomicI64::new(0));
        #[cfg(fiber_rt)]
        let owned_handlers = Box::new(std::sync::atomic::AtomicUsize::new(0));
        VmCtx {
            trap: AtomicI64::new(0),
            end: &*owned_end,
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
            lazy: a.lazy,
            window: w,
            owned_end: Some(owned_end),
            #[cfg(fiber_rt)]
            serve_site: std::sync::OnceLock::new(),
            #[cfg(fiber_rt)]
            handler_window: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            #[cfg(fiber_rt)]
            handlers: &*owned_handlers,
            #[cfg(fiber_rt)]
            owned_handlers: Some(owned_handlers),
            thunk_state: std::sync::OnceLock::new(),
            kernel: None,
        }
    }

    /// #2488 — a context for another vCPU of this context's domain: the same instance and the same
    /// end, charging `fuel`. It must not outlive the domain's root context, which owns the end.
    pub(crate) fn vcpu(&self, fuel: *mut crate::FuelCell) -> VmCtx {
        VmCtx {
            // A domain that already ended stops the vCPU at its first check.
            trap: AtomicI64::new(self.ended().unwrap_or(0)),
            end: self.end,
            cap_ctx: self.cap_ctx,
            epoch: self.epoch,
            fuel,
            sig_armed: self.sig_armed,
            sig_ctx: self.sig_ctx,
            embedder: self.embedder,
            sched: self.sched,
            nursery: self.nursery,
            setjmp: self.setjmp,
            pages: self.pages,
            lazy: self.lazy,
            window: self.window,
            owned_end: None,
            #[cfg(fiber_rt)]
            serve_site: std::sync::OnceLock::new(),
            #[cfg(fiber_rt)]
            handler_window: std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()),
            #[cfg(fiber_rt)]
            handlers: self.handlers,
            #[cfg(fiber_rt)]
            owned_handlers: None,
            thunk_state: std::sync::OnceLock::new(),
            kernel: None,
        }
    }

    /// #2742 — the domain's count of serve handlers its end waits for
    /// ([`crate::child_exec::HANDLERS_ENDING`]).
    #[cfg(fiber_rt)]
    pub(crate) fn handlers(&self) -> &std::sync::atomic::AtomicUsize {
        // SAFETY: as the end word's: the root context's boxed count, which outlives every context of
        // its domain.
        unsafe { &*self.handlers }
    }

    /// #2534 — the window and function table this context's code runs over.
    pub fn window(&self) -> &Window {
        &self.window
    }

    /// The domain's end word.
    fn end(&self) -> &AtomicI64 {
        // SAFETY: `end` is the root context's boxed word, which outlives every context of its domain
        // (`Self::vcpu`'s contract).
        unsafe { &*self.end }
    }

    /// End this context's domain with `code` — a trap, an `exit`, or an internal end — unless an end
    /// is already recorded: the first end wins, and a later one changes nothing (#2499). This
    /// context stops at its next check; the domain's other vCPUs, at their next poll or park.
    pub fn raise(&self, code: i64) {
        debug_assert_ne!(code, 0, "an end is never 0");
        let end = match self
            .end()
            .compare_exchange(0, code, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => code,
            Err(first) => first,
        };
        let _ = self
            .trap
            .compare_exchange(0, end, Ordering::Relaxed, Ordering::Relaxed);
    }

    /// #2742 — the domain this context is of, as a key: the address of its end word, which every
    /// context of the domain shares.
    pub(crate) fn domain_key(&self) -> usize {
        self.end as usize
    }

    /// #2319 — the `call.cap` thunk's state on this context, made by `init` on the first call that
    /// asks: what a thunk keeps from one call to the next (temen-run's view of the run's window), so
    /// a call need not build it again. Each context has its own, and none outlives its context.
    /// `None` when a state of another type is already here.
    pub fn thunk_state<T: core::any::Any + Send + Sync>(
        &self,
        init: impl FnOnce() -> T,
    ) -> Option<&T> {
        use core::any::{Any, TypeId};
        let (made, state) = self
            .thunk_state
            .get_or_init(|| (TypeId::of::<T>(), Box::new(init())));
        // #2847 — the type the state was made as is recorded beside it, so a call compares two ids
        // where `downcast_ref` asks the state for its type through its vtable on every call.
        // SAFETY: `state` was made as a `made`, which is `T`.
        (*made == TypeId::of::<T>())
            .then(|| unsafe { &*(&**state as *const (dyn Any + Send + Sync)).cast::<T>() })
    }

    /// The code that ended this context's domain, or `None` while it runs.
    pub fn ended(&self) -> Option<i64> {
        match self.end().load(Ordering::Relaxed) {
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
// words written concurrently — the domain's end and the stop word, by sibling vCPUs of one instance
// and a parent ending it — are atomic. So one context may be shared by the threads of one instance (a
// fiber that moved to another vCPU, a §14 task's worker).
unsafe impl Send for VmCtx {}
unsafe impl Sync for VmCtx {}

/// Field offsets compiled code loads at (`#[repr(C)]` makes them fixed; `offset_of!` keeps them in
/// step with the struct).
pub(crate) const END: i32 = core::mem::offset_of!(VmCtx, end) as i32;
pub(crate) const EPOCH: i32 = core::mem::offset_of!(VmCtx, epoch) as i32;
pub(crate) const FUEL: i32 = core::mem::offset_of!(VmCtx, fuel) as i32;
pub(crate) const SIG_ARMED: i32 = core::mem::offset_of!(VmCtx, sig_armed) as i32;
pub(crate) const PAGES: i32 = core::mem::offset_of!(VmCtx, pages) as i32;

// The stop word must be the first field: the check after each call dereferences the vmctx pointer
// itself.
const _: () = assert!(core::mem::offset_of!(VmCtx, trap) == 0);

#[cfg(test)]
mod tests {
    use super::{InstanceAddrs, VmCtx, Window};

    /// #2499 — a domain ends once: the first end raised stands, and a later one changes nothing.
    #[test]
    fn the_first_end_wins() {
        let vm = VmCtx::new(InstanceAddrs::NONE, Window::NONE);
        assert_eq!(vm.ended(), None);
        vm.raise(7);
        vm.raise(9);
        assert_eq!(vm.ended(), Some(7));
    }

    /// #2488 — a vCPU's context shares its root's end and stops on a copy of it: an end raised
    /// through either is the domain's, and the raiser's stop word holds the end that won.
    #[test]
    fn a_vcpu_context_shares_the_end_and_stops_on_its_copy() {
        let stop = |vm: &VmCtx| vm.trap.load(core::sync::atomic::Ordering::Relaxed);
        let root = VmCtx::new(InstanceAddrs::NONE, Window::NONE);
        let a = root.vcpu(core::ptr::null_mut());
        let b = root.vcpu(core::ptr::null_mut());
        a.raise(7);
        assert_eq!((root.ended(), b.ended()), (Some(7), Some(7)));
        assert_eq!(
            (stop(&a), stop(&b), stop(&root)),
            (7, 0, 0),
            "only the raiser stops at once"
        );
        b.raise(9);
        assert_eq!(
            (root.ended(), stop(&b)),
            (Some(7), 7),
            "b stops on the end that won"
        );
        assert_eq!(
            stop(&root.vcpu(core::ptr::null_mut())),
            7,
            "a late vCPU starts stopped"
        );
    }

    /// Runs `src` (the root takes one `i64`) and returns its outcome and whether anything stored the
    /// marker at window offset 16392, just past the NULL guard.
    fn run_marked(src: &str) -> (crate::JitOutcome, bool) {
        let m = temen_text::parse_module(src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let (outcome, mem) = crate::run_inner(
            &m,
            0,
            &[0],
            crate::empty_cap_thunk,
            core::ptr::null_mut(),
            crate::RunOpts {
                snapshot_cap: Some(16400),
                ..crate::RunOpts::default()
            },
        )
        .expect("compiles and runs");
        (outcome, mem[16392..16400] != [0; 8])
    }

    /// #2534 — `memory.wait` and `atomic.notify` hand their thunks the guest address, which the
    /// thunk confines against the context's window: a wait past the reservation is the
    /// `MemoryFault` a load there is, and a notify there wakes nobody (no wait can park there). A
    /// notify inside the window still reaches the thunk's futex.
    #[test]
    fn a_wait_past_the_window_faults_and_a_notify_there_wakes_none() {
        if !temen_fiber::supported() {
            return;
        }
        let run = |body: &str| {
            let src = format!(
                "memory 16\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n{body}  }}\n}}\n"
            );
            run_marked(&src).0
        };
        let wait = run(
            "  v1 = i64.const 2199023255552\n  v2 = i32.const 0\n  v3 = i64.const -1\n  \
             v4 = i32.atomic.wait v1 v2 v3\n  v5 = i64.extend_i32_u v4\n  return v5\n",
        );
        assert_eq!(
            wait,
            crate::JitOutcome::Trapped(crate::TrapKind::MemoryFault),
            "a wait past the window"
        );
        for addr in ["2199023255552", "16384"] {
            let notify = run(&format!(
                "  v1 = i64.const {addr}\n  v2 = i32.const -1\n  v3 = atomic.notify v1 v2\n  \
                 v4 = i64.extend_i32_u v3\n  return v4\n"
            ));
            assert_eq!(
                notify,
                crate::JitOutcome::Returned(vec![0]),
                "a notify at {addr} woke someone"
            );
        }
    }

    /// #2488 — a vCPU parked when a sibling ends the domain wakes as if woken, with nothing raised on
    /// its own context: only the check after the wait, which reads the domain's end, stops it. The
    /// root ends the domain once the vCPU has flagged that it is about to wait. Had the vCPU run on,
    /// it would have stored the marker.
    #[test]
    fn a_parked_vcpu_stops_when_a_sibling_ends_the_domain() {
        if !temen_fiber::supported() {
            return;
        }
        let src = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = thread.spawn 1 v1 v0
  br 1()
  }
block 1 () {
  v3 = i64.const 16400
  v4 = i32.atomic.load v3
  br_if v4 2() 1()
  }
block 2 () {
  unreachable
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 16400
  v3 = i32.const 1
  i32.atomic.store v2 v3
  v4 = i64.const 16384
  v5 = i32.const 0
  v6 = i64.const -1
  v7 = i32.atomic.wait v4 v5 v6
  v8 = i64.const 16392
  v9 = i64.const 99
  i64.store v8 v9
  return v1
  }
}
";
        let trapped = crate::JitOutcome::Trapped(crate::TrapKind::Unreachable);
        assert_eq!(run_marked(src), (trapped, false), "the waiter stopped");
    }

    /// #2488 — a fiber runs on its creator's context wherever it is resumed, so a fiber the root made
    /// that traps on another vCPU stops the root's context, not the resumer's: the resumer stops
    /// at the check after its `cont.resume`, which reads the domain's end.
    #[test]
    fn a_resumer_stops_when_a_fiber_made_elsewhere_traps() {
        if !temen_fiber::supported() {
            return;
        }
        let src = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = ref.func 2
  v2 = i64.const 8192
  v3 = cont.new v1 v2
  v4 = i64.const 0
  v5 = thread.spawn 1 v4 v3
  v6_st, v6 = thread.join v5
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  vs, vv = cont.resume v1 v2
  v3 = i64.const 16392
  v4 = i64.const 99
  i64.store v3 v4
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  unreachable
  }
}
";
        let trapped = crate::JitOutcome::Trapped(crate::TrapKind::Unreachable);
        assert_eq!(run_marked(src), (trapped, false), "the resumer stopped");
    }
}
