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
//! **The thread's record (#3078 slice 2).** A kernel vCPU's hook is found by the thread of control
//! running now, never through the context the code carries: a fiber's code carries its creator's
//! context ([`crate::VmCtx`], `vmctx.rs`'s note), so a fiber made on one vCPU and resumed on another
//! must reach the hook of the vCPU it runs on. [`KernelRun::run_vcpu`] publishes the record for its
//! guarded call, beside the vCPU's fiber runtime (`fiber_rt`'s `CURRENT_RT`), and one guard clears
//! both on every exit. The record names its run by the run's one end word, which every context of a
//! kernel run points at, a fiber's creator context included: a context of any other run, whose code
//! a synchronous host procedure may run on this thread (a held domain's call, a nested `jit_run`),
//! fails that compare and takes the JIT's own runtime, as it would on any other thread.
//!
//! **The borrow.** A thunk takes the hook `&mut` for one [`KernelPark`] call, marking the record
//! borrowed meanwhile. During it the thread runs no guest code: the kernel parks the thread in place
//! (`park_native`), and executor 2 carries another task's activation onto a parked thread only for a
//! held run (`Next::Carry`), which never reaches the kernel. A borrow never spans a fiber switch:
//! the call runs on this stack, and no switch happens inside it. A thunk that finds the record
//! borrowed, or the swap back that does not find it so, fails closed (`Malformed`) in every build
//! and asserts in a debug one.

use crate::{mem, slot_record, CompiledModule, FuelCell, TrapKind, VmCtx};
use core::ptr::NonNull;
use std::cell::Cell;
use std::sync::atomic::AtomicI64;
use std::sync::Mutex;
pub use temen_ir::seams::NativeStop;

/// The scheduling core's side of a vCPU's §12 ops: apply `stop`, park the thread until the task can
/// run, and answer `(status, value)` for a join, the result in `[0]` otherwise. `Err` is the trap
/// code to raise: the task or its run ended meanwhile, or the op traps.
pub trait KernelPark {
    fn park(&mut self, stop: NativeStop) -> Result<[i64; 2], i64>;

    /// #2958 slice 5 — the park a capability call left its vCPU to take (a pipe or stdin read, a
    /// punted call, a region post's wake), from the embedder's `call.cap` thunk rather than the
    /// emitted code: `call` is the embedder's own decision, a value this crate cannot name, handed
    /// from the embedder's thunk to its own hook through [`park_call`]. A later slice extends what
    /// that value carries, not this trait. Answers as [`Self::park`] does.
    fn park_call(&mut self, call: &mut dyn std::any::Any) -> Result<[i64; 2], i64>;
}

/// The thread's record (the module docs): the run its vCPU is of, by its end word, and its hook.
#[derive(Clone, Copy)]
struct Record {
    /// The run's end word; null when this thread runs no kernel vCPU.
    end: *const AtomicI64,
    hook: Hook,
}

#[derive(Clone, Copy)]
enum Hook {
    Off,
    Free(NonNull<dyn KernelPark>),
    /// A thunk holds the hook `&mut` (the module docs).
    Borrowed,
}

impl Record {
    const OFF: Record = Record {
        end: std::ptr::null(),
        hook: Hook::Off,
    };
}

thread_local! {
    static RECORD: Cell<Record> = const { Cell::new(Record::OFF) };
}

/// This thread's record, read on the thread running now: `#[inline(never)]` for the reason
/// `fiber_rt::current` is (#1466), since a thunk that reads it may run on a fiber's stack.
#[inline(never)]
fn record() -> Record {
    RECORD.with(Cell::get)
}

/// Replace this thread's record, answering the one it replaced.
#[inline(never)]
fn set_record(r: Record) -> Record {
    RECORD.with(|c| c.replace(r))
}

/// Whether code running on `vm` runs as a vCPU of a kernel run on this thread: its §12 ops and
/// parks are the kernel's, never this crate's own thread runtime's.
pub fn on_vcpu(vm: &VmCtx) -> bool {
    std::ptr::eq(record().end, vm.end_word())
}

/// Whether code running on `vm` runs as a kernel vCPU **inside a fiber**: a park there is a fiber's,
/// which the kernel does not take yet (`native_scope`'s `fiber parks`; #3078 slices 3 and 4).
pub fn in_fiber(vm: &VmCtx) -> bool {
    #[cfg(fiber_rt)]
    let fiber = || crate::fiber_rt::current_fiber_slot().is_some();
    #[cfg(not(fiber_rt))]
    let fiber = || false;
    on_vcpu(vm) && fiber()
}

/// Route a §12 thunk's op to the kernel, if `vm` is a kernel vCPU's context on this thread: `None`
/// otherwise, and the thunk goes on as the OS-thread runtime. A trap is raised on `vm`, whose next
/// check unwinds. A wait made in a fiber is a fiber's park, which `fiber parks` declines: reaching
/// one fails closed.
#[cfg(fiber_rt)]
#[inline]
pub(crate) fn routed(vm: &VmCtx, stop: NativeStop) -> Option<[i64; 2]> {
    if matches!(stop, NativeStop::Wait { .. }) && in_fiber(vm) {
        vm.raise(temen_ir::trap_code::MALFORMED);
        return Some([0, 0]);
    }
    // SAFETY: a §12 thunk runs on the vCPU's own thread, inside the guarded call `run_vcpu`
    // published the record for (the module docs).
    unsafe { through_hook(vm, |hook| hook.park(stop)) }
}

/// #2958 slice 5 — hand a capability call's park to the kernel, if `vm` is a kernel vCPU's context
/// on this thread ([`KernelPark::park_call`]): `None` otherwise. A trap is raised on `vm`, whose next
/// check unwinds.
///
/// # Safety
/// The caller is the embedder's `call.cap` thunk of code running on `vm`, on this thread, and holds
/// no other reference to the hook.
pub unsafe fn park_call(vm: &VmCtx, call: &mut dyn std::any::Any) -> Option<[i64; 2]> {
    // SAFETY: the caller's contract.
    unsafe { through_hook(vm, |hook| hook.park_call(call)) }
}

/// The one way into a vCPU's hook ([`routed`], [`park_call`]): run `f` on the hook this thread's
/// record holds for `vm`'s run, raising its error on `vm`.
///
/// # Safety
/// As [`park_call`].
#[inline]
unsafe fn through_hook(
    vm: &VmCtx,
    f: impl FnOnce(&mut dyn KernelPark) -> Result<[i64; 2], i64>,
) -> Option<[i64; 2]> {
    let r = record();
    if !std::ptr::eq(r.end, vm.end_word()) {
        return None;
    }
    let Hook::Free(hook) = r.hook else {
        // Re-entry while a borrow is live: never on a kernel run (the module docs).
        debug_assert!(false, "a kernel hook reached while it is borrowed");
        vm.raise(temen_ir::trap_code::MALFORMED);
        return Some([0, 0]);
    };
    set_record(Record {
        hook: Hook::Borrowed,
        ..r
    });
    // SAFETY: the record is marked borrowed, so nothing else reaches the hook meanwhile.
    let out = f(unsafe { &mut *hook.as_ptr() });
    let back = set_record(r);
    if !(std::ptr::eq(back.end, r.end) && matches!(back.hook, Hook::Borrowed)) {
        // The borrow spanned a switch onto another thread's record: never (the module docs).
        debug_assert!(false, "a kernel hook's borrow spanned a switch");
        set_record(back);
        vm.raise(temen_ir::trap_code::MALFORMED);
        return Some([0, 0]);
    }
    match out {
        Ok(r) => Some(r),
        Err(code) => {
            vm.raise(code);
            Some([0, 0])
        }
    }
}

/// [`KernelRun::run_vcpu`]'s publication of the thread's record and the vCPU's fiber runtime,
/// cleared as it drops, on every exit of the guarded call.
struct Published {
    prev: Record,
    #[cfg(fiber_rt)]
    fibers: Option<(
        Box<crate::fiber_rt::FiberRuntime>,
        *mut crate::fiber_rt::FiberRuntime,
    )>,
}

impl Published {
    /// # Safety
    /// `hook` outlives the returned guard.
    unsafe fn new(
        end: &AtomicI64,
        hook: &mut (dyn KernelPark + '_),
        #[cfg(fiber_rt)] fibers: Option<Box<crate::fiber_rt::FiberRuntime>>,
    ) -> Published {
        let hook: *mut (dyn KernelPark + '_) = hook;
        // SAFETY: the guard ends the record's use of the hook before the hook goes (the contract).
        let hook: *mut (dyn KernelPark + 'static) = unsafe { std::mem::transmute(hook) };
        let prev = set_record(Record {
            end,
            hook: Hook::Free(NonNull::new(hook).expect("a reference")),
        });
        #[cfg(fiber_rt)]
        let fibers = fibers.map(|mut rt| {
            let prev = crate::fiber_rt::set_current(&mut *rt);
            (rt, prev)
        });
        Published {
            prev,
            #[cfg(fiber_rt)]
            fibers,
        }
    }
}

impl Drop for Published {
    fn drop(&mut self) {
        #[cfg(fiber_rt)]
        if let Some((_, prev)) = &self.fibers {
            crate::fiber_rt::set_current(*prev);
        }
        set_record(self.prev);
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
    /// #3078 slice 2 — every vCPU context the run has made, for the next vCPU to reuse once its own
    /// ended: a context lives as long as its run, since a fiber its vCPU made goes on running on it
    /// after the vCPU ends (`vmctx.rs`'s note), as today's route keeps them (`Domain::vcpu_ctxs`).
    /// Boxed: code holds each one's address.
    #[allow(clippy::vec_box)]
    ctxs: Mutex<Vec<Box<VmCtx>>>,
    /// #3078 slice 2 — the run's one fiber table, when its module runs `cont.*`, and the config each
    /// vCPU's fiber runtime over it is built from.
    #[cfg(fiber_rt)]
    fibers: Option<RunFibers>,
}

/// A kernel run's fibers (#3078 slice 2): its table, charged to the run's budget node, and what each
/// vCPU's runtime over it needs.
#[cfg(fiber_rt)]
struct RunFibers {
    table: std::sync::Arc<crate::fiber_rt::SharedFiberTable>,
    type_id: u32,
    mask: u64,
    tramp: crate::fiber_rt::FiberCallTramp,
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
    /// `cm` was compiled with, if any, kept for as long as its code; `node` is the run's budget
    /// node, which its fibers' stacks are charged to (#2112), as executor 2 charges its own. `None` when `cm` is one the
    /// kernel does not run (§14 nesting, durability, a target without the fiber runtime);
    /// `Some(None)` when the OS refuses the window or a protection it needs (#2305, #2286), a run that
    /// ends as one the process cannot hold.
    pub fn new(
        cm: CompiledModule,
        armed: Option<Box<FuelCell>>,
        init_mem: &[u8],
        node: Option<std::sync::Arc<dyn crate::BudgetNode>>,
    ) -> Option<Option<Started>> {
        if !crate::fiber_supported()
            || cm.nests
            || cm.durable
            || cm.call_tramp.is_none() && (cm.spawns || cm.fibers)
        {
            return None;
        }
        #[cfg(fiber_rt)]
        let fibers = match (cm.fibers, cm.fiber_cfg, cm.call_tramp) {
            (false, _, _) => None,
            (true, Some((type_id, mask)), Some(tramp)) => Some(RunFibers {
                table: cm.run_fiber_table(node),
                type_id,
                mask,
                tramp,
            }),
            (true, _, _) => return None,
        };
        #[cfg(not(fiber_rt))]
        let _ = node;
        let Some((window, ro)) = cm.seed_window(Some(init_mem)) else {
            return Some(None);
        };
        let root = VmCtx::new(cm.instance, cm.window_at(window.base()));
        let run = KernelRun {
            cm,
            window,
            root,
            _armed: armed,
            ctxs: Mutex::new(Vec::new()),
            #[cfg(fiber_rt)]
            fibers,
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
        // Code compiled without fuel checks reads no cell, and a null one tells the runtime so
        // (`make_fiber`'s start refund).
        let fuel: *mut FuelCell = match self.metered() {
            true => fuel,
            false => std::ptr::null_mut(),
        };
        let vm = self.take_ctx(end, fuel, cap_ctx);
        #[cfg(fiber_rt)]
        let fibers = self.fibers.as_ref().map(|f| {
            let mut rt = Box::new(crate::fiber_rt::FiberRuntime::new(
                std::sync::Arc::clone(&f.table),
                f.type_id,
                f.mask,
            ));
            rt.set_call_tramp(f.tramp);
            rt
        });
        // SAFETY: `hook` outlives the guard, which drops before this call returns.
        let published = unsafe {
            Published::new(
                end,
                hook,
                #[cfg(fiber_rt)]
                fibers,
            )
        };
        // SAFETY: this function's contract.
        let ran = unsafe { self.run_on(entry, tls, &vm) };
        drop(published);
        self.give_ctx(vm);
        ran
    }

    /// A context for a vCPU about to run, over the run's end word `end`, charging `fuel`, its calls
    /// through `cap_ctx`: one an ended vCPU left ([`Self::ctxs`]), or a new one.
    ///
    /// A reused context may still be run on by a fiber its last vCPU made, on another thread, so its
    /// fields are only written where they differ: in a run whose module has fibers, every context's
    /// are the run's (`cap_ctx`, the domain's host cell, and a null `fuel`, `fiber fuel` declining a
    /// metered one), so none is written there.
    fn take_ctx(
        &self,
        end: &AtomicI64,
        fuel: *mut FuelCell,
        cap_ctx: *mut core::ffi::c_void,
    ) -> Box<VmCtx> {
        let pooled = self.ctxs.lock().unwrap_or_else(|e| e.into_inner()).pop();
        let mut vm = match pooled {
            Some(vm) => vm,
            // SAFETY: `end` is the run's, which outlives the run's contexts (executor 2 owns it for
            // the run).
            None => Box::new(unsafe { self.root.vcpu_ending(end, fuel) }),
        };
        debug_assert!(std::ptr::eq(vm.end_word(), end), "one end word per run");
        if vm.fuel != fuel {
            vm.fuel = fuel;
        }
        if vm.cap_ctx != cap_ctx {
            vm.cap_ctx = cap_ctx;
        }
        vm
    }

    /// A vCPU ended: its context waits for the next ([`Self::take_ctx`]).
    fn give_ctx(&self, vm: Box<VmCtx>) {
        self.ctxs.lock().unwrap_or_else(|e| e.into_inner()).push(vm);
    }

    /// [`Self::run_vcpu`]'s guarded call, on `vm`, with the thread's record published.
    ///
    /// # Safety
    /// As [`Self::run_vcpu`]'s: `vm`'s `cap_ctx` is live for the call.
    unsafe fn run_on(&self, entry: VcpuEntry<'_>, tls: i64, vm: &VmCtx) -> Result<Vec<i64>, i64> {
        let cm = &self.cm;
        let base = self.window.base();
        let fn_table = cm.fn_table.base();
        // #3047 — no fiber of an earlier run on this thread is the one running.
        #[cfg(fiber_rt)]
        crate::fiber_rt::enter_root();
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
                    vm,
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
                    vm: vm as *const VmCtx as u64,
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

#[cfg(all(test, fiber_rt))]
mod tests {
    use super::*;
    use crate::vmctx::{InstanceAddrs, Window};

    /// A hook whose `park` runs a §12 op on the same context again, as code that reached the hook
    /// while it is borrowed would.
    struct Reenters<'v>(&'v VmCtx);

    impl KernelPark for Reenters<'_> {
        fn park(&mut self, _: NativeStop) -> Result<[i64; 2], i64> {
            Ok(routed(self.0, NativeStop::Notify { addr: 0, count: 1 }).unwrap_or([7, 7]))
        }
        fn park_call(&mut self, _: &mut dyn std::any::Any) -> Result<[i64; 2], i64> {
            Ok([0, 0])
        }
    }

    /// #3078 slice 2 — a thunk that reaches the hook while it is borrowed fails closed: a debug
    /// build asserts, and any other traps the domain `Malformed` rather than alias the borrow.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "while it is borrowed"))]
    fn a_hook_reached_while_it_is_borrowed_fails_closed() {
        let root = VmCtx::new(InstanceAddrs::NONE, Window::NONE);
        let end = AtomicI64::new(0);
        // SAFETY: `end` outlives `vm`.
        let vm = unsafe { root.vcpu_ending(&end, std::ptr::null_mut()) };
        let mut hook = Reenters(&vm);
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, None) };
        assert!(on_vcpu(&vm), "the record names this run");
        let answer = routed(&vm, NativeStop::Notify { addr: 0, count: 1 });
        drop(published);
        assert_eq!(answer, Some([0, 0]));
        assert_eq!(vm.ended(), Some(temen_ir::trap_code::MALFORMED));
        assert!(!on_vcpu(&vm), "the record is cleared");
    }

    /// #3078 slice 2 — the record names its run: a context of another run on this thread is not a
    /// kernel vCPU's, and its ops take the JIT's own runtime.
    #[test]
    fn another_runs_context_on_a_kernel_vcpus_thread_is_not_the_kernels() {
        let root = VmCtx::new(InstanceAddrs::NONE, Window::NONE);
        let (end, other_end) = (AtomicI64::new(0), AtomicI64::new(0));
        // SAFETY: both ends outlive their contexts.
        let (vm, other) = unsafe {
            (
                root.vcpu_ending(&end, std::ptr::null_mut()),
                root.vcpu_ending(&other_end, std::ptr::null_mut()),
            )
        };
        let mut hook = Reenters(&vm);
        // SAFETY: `hook` outlives the guard.
        let _published = unsafe { Published::new(&end, &mut hook, None) };
        assert!(on_vcpu(&vm));
        assert!(!on_vcpu(&other));
        assert_eq!(
            routed(&other, NativeStop::Notify { addr: 0, count: 1 }),
            None
        );
    }
}
