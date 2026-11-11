//! **#2948 spike — Cranelift vCPUs as tasks of executor 2's scheduling core.** Scaffolding
//! (INVARIANTS #13, #15), never a supported second route: it either becomes phase 1, which deletes
//! the thread half of [`crate::os_thread_rt`], or it is deleted.
//!
//! In this mode the run is executor 2's (`temen_interp::bytecode`'s `ThreadRun`): it owns the tasks,
//! starts their threads and applies the §12 rules. This module gives it what it needs of the JIT:
//!
//! - [`KernelRun`]: a compiled module's window, reserved here and seeded by the kernel, and one root
//!   context whose end every vCPU shares; [`KernelRun::run_vcpu`] runs a task's entry on the calling
//!   thread, as `os_thread_rt::run_child` runs a spawned vCPU's.
//! - [`KernelPark`]: the per-thread hook the `thread.spawn`, `thread.join`, `memory.wait` and
//!   `memory.notify` thunks call instead of the OS-thread runtime ([`routed`]). The kernel parks the
//!   thread there, with the emitted frames on its stack.
//!
//! Out of scope, refused by the embedder before it gets here: fibers, §14 children, durability, fuel
//! metering, signals.

use crate::{mem, slot_record, CompiledModule, TrapKind, VmCtx};
use std::cell::Cell;

/// A §12 op the emitted code made, for the kernel to apply ([`KernelPark::park`]). Addresses are
/// window-relative, as the guest wrote them.
#[derive(Clone, Copy, Debug)]
pub enum KStop {
    Spawn {
        func: u32,
        arg0: i64,
        arg1: i64,
    },
    Join {
        handle: i32,
    },
    /// `timeout` in ns, negative for none.
    Wait {
        off: u64,
        expected: u64,
        wide: bool,
        timeout: i64,
    },
    Notify {
        off: u64,
        count: i32,
    },
}

/// The kernel's side of a vCPU's §12 ops: apply `stop`, park the thread until the task can run, and
/// answer `(status, value)` for a join, the result in `[0]` otherwise. `Err` is the trap code to
/// raise: the task or its run ended meanwhile, or the op traps.
pub trait KernelPark {
    fn park(&mut self, stop: KStop) -> Result<[i64; 2], i64>;
}

thread_local! {
    /// The hook of the vCPU running on this thread, while [`KernelRun::run_vcpu`] runs it.
    static HOOK: Cell<Option<*mut (dyn KernelPark + 'static)>> = const { Cell::new(None) };
}

/// Route a §12 thunk's op to the kernel, if this thread runs a kernel vCPU: `None` otherwise, and the
/// thunk goes on as the OS-thread runtime. An `Err` is raised on `vm`, whose next check unwinds.
pub(crate) fn routed(vm: &VmCtx, stop: KStop) -> Option<[i64; 2]> {
    let hook = HOOK.with(Cell::get)?;
    // SAFETY: `run_vcpu` set the hook for the call it makes on this thread, and clears it before
    // the hook's borrow ends; a thunk runs only inside that call.
    match unsafe { (*hook).park(stop) } {
        Ok(r) => Some(r),
        Err(code) => {
            vm.raise(code);
            Some([0, 0])
        }
    }
}

/// Call `w` when `door` fires, for as long as the registration lives: how a kernel run hears of the
/// JIT kill door its code polls, to stop its parked tasks too.
pub fn on_kill(door: &crate::Interrupt, w: Box<dyn Fn() + Send + Sync>) -> crate::Registration<'_> {
    door.wakers().register(w)
}

/// What [`KernelRun::run_vcpu`] runs.
#[derive(Clone, Copy, Debug)]
pub enum VcpuEntry<'a> {
    /// The compiled entry, the root task's: its arguments.
    Root(&'a [i64]),
    /// A `thread.spawn` child's entry: function `func` of the table, with its two arguments.
    Thread { func: u32, arg0: i64, arg1: i64 },
}

/// One kernel run of a compiled module: its window and the root context whose end its vCPUs share.
pub struct KernelRun {
    cm: CompiledModule,
    window: mem::GuestWindow,
    root: VmCtx,
}

// SAFETY: the compiled code is finalized and only read; the window is shared by design (the guest's
// atomics are real ones); the root context is read by each vCPU (`vcpu`) and written only through its
// atomic end. Every vCPU is gone before the run drops it (executor 2 waits for its threads).
unsafe impl Send for KernelRun {}
unsafe impl Sync for KernelRun {}

impl KernelRun {
    /// Reserve `cm`'s window, read-write over its declared extent and unseeded: the kernel's window
    /// seeds it ([`Self::window`]), then [`Self::protect`]. `None` when the OS refuses it, or `cm` is
    /// outside the spike (fibers, nesting, durability).
    pub fn new(cm: CompiledModule) -> Option<KernelRun> {
        if cm.fibers || cm.nests || cm.durable || cm.call_tramp.is_none() && cm.spawns {
            return None;
        }
        let window = mem::GuestWindow::try_new(cm.win_mapped, cm.win_reserved)?;
        let root = VmCtx::new(cm.instance, cm.window_at(window.base()));
        Some(KernelRun { cm, window, root })
    }

    /// The window: `(base, reserved bytes)`.
    pub fn window(&self) -> (*mut u8, u64) {
        (self.window.base(), self.cm.win_reserved as u64)
    }

    /// Once the kernel has seeded the window: map the read-only data segments read-only, and the NULL
    /// guard, as a fresh run's window is (`CompiledModule::seed_window`). `false` if the OS refused one.
    pub fn protect(&self) -> bool {
        let cm = &self.cm;
        if let Some(size_log2) = cm.mem_size_log2 {
            let size = 1u64 << size_log2;
            for d in cm.data.iter().filter(|d| d.readonly) {
                let lo = d.offset.min(size);
                let hi = d.offset.saturating_add(d.bytes.len() as u64).min(size);
                if hi > lo && !self.window.protect(lo, hi - lo, mem::Prot::Ro) {
                    return false;
                }
            }
        }
        cm.null_guard > cm.win_mapped as u64 || self.window.seed_null_guard(cm.null_guard)
    }

    /// #2948 attribution — the futex word at window offset `off` (8 bytes if `wide`, else 4), read
    /// with a sequentially consistent load, if it lies where every vCPU may read: aligned, in the
    /// committed window and past the NULL guard. `None` anywhere else, which the kernel answers
    /// (a fault, or its own read). No page op is in scope, so the committed extent is fixed.
    pub fn load_word(&self, off: u64, wide: bool) -> Option<u64> {
        use std::sync::atomic::{AtomicU32, AtomicU64, Ordering::SeqCst};
        let cm = &self.cm;
        let width = if wide { 8 } else { 4 };
        let guard = if cm.null_guard > cm.win_mapped as u64 {
            0
        } else {
            cm.null_guard
        };
        let end = off.checked_add(width)?;
        if !off.is_multiple_of(width) || off < guard || end > cm.win_mapped as u64 {
            return None;
        }
        // SAFETY: `[off, off + width)` is aligned and inside the committed read-write window (the
        // read-only data pages are readable too), which outlives every vCPU of the run.
        unsafe {
            let p = self.window.base().add(off as usize);
            Some(if wide {
                (*(p as *const AtomicU64)).load(SeqCst)
            } else {
                u64::from((*(p as *const AtomicU32)).load(SeqCst))
            })
        }
    }

    /// End the domain: every vCPU still running stops at its next poll of the domain's end.
    pub fn end(&self) {
        self.root.raise(crate::DOMAIN_DONE_CODE as i64);
    }

    /// Run `entry` on this thread as a vCPU with `vcpu.tls` seed `tls`, its `call.cap`s through
    /// `cap_ctx` (a `*const Mutex<Host>` for the locked thunk), its §12 ops through `park`. Returns
    /// its results, or the domain's end code it stopped on.
    ///
    /// # Safety
    /// `cap_ctx` is what the module's `call.cap` thunk takes, live for the call; the kernel seeded and
    /// [`protect`](Self::protect)ed the window first.
    pub unsafe fn run_vcpu(
        &self,
        entry: VcpuEntry<'_>,
        tls: i64,
        cap_ctx: *mut core::ffi::c_void,
        park: &mut dyn KernelPark,
    ) -> Result<Vec<i64>, i64> {
        let cm = &self.cm;
        let mut vm = self.root.vcpu(std::ptr::null_mut());
        vm.cap_ctx = cap_ctx;
        let base = self.window.base();
        let fn_table = cm.fn_table.base();
        crate::vcpu_tls::seed(tls);
        crate::durable_shadow::seed(cm.shadow.region_base(0));
        mem::install_guard();
        // The hook outlives the call below, which is the only place a thunk reads it.
        let hook: *mut (dyn KernelPark + '_) = park;
        let hook: *mut (dyn KernelPark + 'static) = std::mem::transmute(hook);
        let prev = HOOK.with(|h| h.replace(Some(hook)));
        let mut results;
        let faulted = match entry {
            VcpuEntry::Root(args) => {
                if args.len() < cm.n_params {
                    HOOK.with(|h| h.set(prev));
                    return Err(TrapKind::ThreadFault as i64);
                }
                results = vec![0i64; cm.n_results];
                mem::run_guarded(
                    &self.window,
                    cm.tramp_code,
                    args.as_ptr(),
                    results.as_mut_ptr(),
                    base,
                    fn_table,
                    &vm,
                )
            }
            VcpuEntry::Thread { func, arg0, arg1 } => {
                let mut call = ThreadCall {
                    tramp: cm.call_tramp.expect("checked in `new`"),
                    code: slot_record(fn_table, func as usize).code(),
                    mem_base: base as u64,
                    fn_table: fn_table as u64,
                    vm: &vm as *const VmCtx as u64,
                    arg0: arg0 as u64,
                    arg1: arg1 as u64,
                    ret: 0,
                };
                let (lo, hi) = self.window.fault_range();
                let faulted = mem::run_guarded_range(
                    thread_entry as *const () as *const u8,
                    &mut call as *mut ThreadCall as *const i64,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    lo,
                    hi,
                );
                results = vec![call.ret];
                faulted
            }
        };
        HOOK.with(|h| h.set(prev));
        if faulted {
            vm.raise(mem::FAULT_TRAP);
        }
        match vm.ended() {
            Some(code) => Err(code),
            None => Ok(results),
        }
    }
}

/// A `thread.spawn` child's call through the guest call-trampoline ([`thread_entry`]).
#[repr(C)]
struct ThreadCall {
    tramp: crate::fiber_rt::FiberCallTramp,
    code: u64,
    mem_base: u64,
    fn_table: u64,
    vm: u64,
    arg0: u64,
    arg1: u64,
    ret: i64,
}

/// The `Entry`-shaped body [`KernelRun::run_vcpu`] guards for a child: `os_thread_rt`'s
/// `child_entry`, over a [`ThreadCall`].
extern "C" fn thread_entry(
    a: *const i64,
    _r: *mut i64,
    _m: *mut u8,
    _t: *const core::ffi::c_void,
    _tc: *mut i64,
) {
    // SAFETY: `a` is the `&mut ThreadCall` `run_vcpu` passed, live for the call.
    unsafe {
        let c = a as *mut ThreadCall;
        let v = ((*c).tramp)(
            (*c).code,
            (*c).mem_base,
            (*c).fn_table,
            (*c).vm,
            crate::stack_check::thread_limit(),
            (*c).arg0,
            (*c).arg1,
        );
        (*c).ret = v as i64;
    }
}
