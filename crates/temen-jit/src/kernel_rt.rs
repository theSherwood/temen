//! **#2958 — Cranelift vCPUs as tasks of executor 2's scheduling core.** For a program
//! `temen_interp::bytecode::native_scope` admits, the run is executor 2's (`ThreadRun`): it owns the
//! tasks, starts their threads and applies the §12 rules, and the Cranelift JIT is its emitted tier.
//! This module gives the run what it needs of the JIT:
//!
//! - [`KernelRun`]: a compiled module's window and the root context every vCPU's is made from;
//!   [`KernelRun::run_vcpu`] runs a task's entry on the calling thread, as `os_thread_rt::run_child`
//!   runs a spawned vCPU's, over the run's one end word, which executor 2 owns.
//! - [`routed`]: where the `thread.spawn`, `thread.join`, `memory.wait` and `memory.notify` thunks
//!   send a kernel vCPU's op, instead of the OS-thread runtime, through its [`KernelPark`] hook. The
//!   kernel parks the thread there, with the emitted frames on its stack (the embedder's hook calls
//!   `temen_interp::bytecode::EmittedCall::park`).
//!
//! Everything `native_scope` declines runs on [`crate::os_thread_rt`], unchanged.
//!
//! **The hook and its aliasing.** A vCPU's context carries its hook ([`crate::VmCtx`]'s `kernel`),
//! set by [`KernelRun::run_vcpu`] on the context it makes for that one guarded call, which drops with
//! it. Only the thunks of code running on that context read it, on its own thread, inside the guarded
//! call, while `run_vcpu` reaches the hook through nothing else: so the `&mut` a thunk takes through
//! it is the only live one, for the length of the thunk. A park may carry another task's activation
//! on this thread meanwhile (executor 2's `Next::Carry`); if that runs emitted code, it runs on a
//! context of its own with a hook of its own, and reaches this one through no pointer.

use crate::{mem, slot_record, CompiledModule, FuelCell, TrapKind, VmCtx};
use std::sync::atomic::AtomicI64;
pub use temen_ir::seams::NativeStop;

/// The scheduling core's side of a vCPU's §12 ops: apply `stop`, park the thread until the task can
/// run, and answer `(status, value)` for a join, the result in `[0]` otherwise. `Err` is the trap
/// code to raise: the task or its run ended meanwhile, or the op traps.
pub trait KernelPark {
    fn park(&mut self, stop: NativeStop) -> Result<[i64; 2], i64>;
}

/// Route a §12 thunk's op to the kernel, if `vm` is a kernel vCPU's context: `None` otherwise, and
/// the thunk goes on as the OS-thread runtime. A trap is raised on `vm`, whose next check unwinds.
#[cfg(fiber_rt)]
#[inline]
pub(crate) fn routed(vm: &VmCtx, stop: NativeStop) -> Option<[i64; 2]> {
    let hook = vm.kernel?;
    // SAFETY: `run_vcpu` set `kernel` on this context for the guarded call this thunk runs inside,
    // and nothing else reaches the hook meanwhile (the module docs).
    match unsafe { (*hook.as_ptr()).park(stop) } {
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

/// A started [`KernelRun`], and the byte ranges its window's data segments made read-only.
pub type Started = (KernelRun, Vec<[u64; 2]>);

/// One kernel run of a compiled module: its window and the root context its vCPUs' are made from.
///
/// The root context is a template only: no code runs on it, and its own end word is never read or
/// raised on this route. Every vCPU, the root task's included, runs on a copy over the run's end word
/// ([`Self::run_vcpu`]). It goes with slice 7 of #2958 (a native task with no interpreter VM).
pub struct KernelRun {
    cm: CompiledModule,
    window: mem::GuestWindow,
    root: VmCtx,
    /// The cell `cm`'s instance was compiled with, which arms its fuel checks: no vCPU charges it
    /// (each is handed its own, [`Self::run_vcpu`]), but the code holds its address.
    _armed: Option<Box<FuelCell>>,
}

// SAFETY: the compiled code is finalized and only read; the window is shared by design (the guest's
// atomics are real ones); the root context is only read, as each vCPU's is made from it. Every vCPU is
// gone before the run drops it (executor 2 waits for its threads).
unsafe impl Send for KernelRun {}
unsafe impl Sync for KernelRun {}

impl KernelRun {
    /// Start `cm`'s window as a fresh run's is started (`CompiledModule::seed_window`): `init_mem`,
    /// then the data segments with the read-only ones protected, or a thaw's layout
    /// (`CompiledModule::set_restore`), then the NULL guard. With it, the byte ranges the data
    /// segments made read-only, which the embedder's page map starts with. `armed` is the fuel cell
    /// `cm` was compiled with, if any, kept for as long as its code. `None` when `cm` is one the
    /// kernel does not run (fibers, §14 nesting, durability, a target without the fiber runtime);
    /// `Some(None)` when the OS refuses the window or a protection it needs (#2305, #2286), a run that
    /// ends as one the process cannot hold.
    pub fn new(
        cm: CompiledModule,
        armed: Option<Box<FuelCell>>,
        init_mem: &[u8],
    ) -> Option<Option<Started>> {
        if !crate::fiber_supported()
            || cm.fibers
            || cm.nests
            || cm.durable
            || cm.call_tramp.is_none() && cm.spawns
        {
            return None;
        }
        let Some((window, ro)) = cm.seed_window(Some(init_mem)) else {
            return Some(None);
        };
        let root = VmCtx::new(cm.instance, cm.window_at(window.base()));
        let run = KernelRun {
            cm,
            window,
            root,
            _armed: armed,
        };
        Some(Some((run, ro)))
    }

    /// The window: `(base, reserved bytes)`.
    pub fn window(&self) -> (*mut u8, u64) {
        (self.window.base(), self.cm.win_reserved as u64)
    }

    /// Once every vCPU has gone: the window's bytes as the run left them, as
    /// [`CompiledModule::run`]'s `snapshot_cap` asks for (`Some(cap)`: its committed extent, widened
    /// to `cap` bytes over the reserved tail). `None`, nothing read, when the OS refuses to make a
    /// page the guest protected readable again.
    pub fn snapshot(&self, cap: usize) -> Option<Vec<u8>> {
        let mapped = self.cm.win_mapped;
        let snap = match mapped {
            0 => 0,
            _ => cap.min(self.cm.win_reserved).max(mapped),
        };
        if !self.window.restore_rw() {
            return None;
        }
        self.window.read_low(snap)
    }

    /// The window's backed size, its declared memory.
    pub fn mapped(&self) -> u64 {
        self.cm.win_mapped as u64
    }

    /// Whether `cm`'s code carries fuel checks, so each vCPU's own cell must be handed to it
    /// ([`Self::run_vcpu`]).
    pub fn metered(&self) -> bool {
        !self.root.fuel.is_null()
    }

    /// Run `entry` on this thread as a vCPU with `vcpu.tls` seed `tls`, over the run's end word `end`
    /// (executor 2's, `EmittedCall::end`, which it sets as the run ends): its §12 ops through `hook`,
    /// its `call.cap`s through `cap_ctx` (a `*const Mutex<Host>` for the locked thunk), its fuel
    /// charged to `fuel` (code compiled without fuel checks never reads it). Returns its results, or
    /// the domain's end code it stopped on.
    ///
    /// # Safety
    /// `cap_ctx` is what the module's `call.cap` thunk takes, live for the call.
    pub unsafe fn run_vcpu(
        &self,
        entry: VcpuEntry<'_>,
        tls: i64,
        end: &AtomicI64,
        hook: &mut dyn KernelPark,
        cap_ctx: *mut core::ffi::c_void,
        fuel: &mut FuelCell,
    ) -> Result<Vec<i64>, i64> {
        let cm = &self.cm;
        // SAFETY: `end` outlives `vm`, which this call drops before it returns.
        let mut vm = unsafe { self.root.vcpu_ending(end, fuel) };
        vm.cap_ctx = cap_ctx;
        // The hook outlives the guarded call below, which is the only place a thunk reads it.
        let hook: *mut (dyn KernelPark + '_) = hook;
        let hook: *mut (dyn KernelPark + 'static) = std::mem::transmute(hook);
        vm.kernel = core::ptr::NonNull::new(hook);
        let base = self.window.base();
        let fn_table = cm.fn_table.base();
        crate::vcpu_tls::seed(tls);
        crate::vcpu_tls::set_serve_handlers(0);
        crate::durable_shadow::seed(cm.shadow.region_base(0));
        let mut results;
        let faulted = match entry {
            VcpuEntry::Root(args) => {
                if args.len() < cm.n_params {
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
            #[cfg(not(fiber_rt))]
            VcpuEntry::Thread { .. } => return Err(TrapKind::ThreadFault as i64),
            #[cfg(fiber_rt)]
            VcpuEntry::Thread { func, arg0, arg1 } => {
                mem::install_guard();
                let mut tc = ThreadCall {
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
                    &mut tc as *mut ThreadCall as *const i64,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    lo,
                    hi,
                );
                results = vec![tc.ret];
                faulted
            }
        };
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
#[cfg(fiber_rt)]
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
#[cfg(fiber_rt)]
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
