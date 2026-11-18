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
//! must reach the hook of the vCPU it runs on. The context says only whether it is a kernel run's
//! (`VmCtx::kernel`, a fact of the run): a context of no kernel run never reads the record, so code
//! of another run that a synchronous host procedure runs on this thread (a held domain's call, a
//! nested `jit_run`) takes the JIT's own runtime, as on any other thread. A kernel run's context
//! asks the record, which names its run by the run's one end word (every context of the run, a
//! fiber's creator context included, points at it): one whose thread's record names another run, or
//! none, traps `Malformed`, since only a vCPU of the run runs its code.
//!
//! [`KernelRun::run_vcpu`] publishes the record for its guarded call, and one guard restores the
//! thread's earlier record on every exit, so a nested kernel run hands the outer vCPU its record
//! back. For a module with fibers it publishes the vCPU's fiber runtime (`fiber_rt`'s `CURRENT_RT`)
//! too, and restores that as well. The record names that runtime: whether a park is a fiber's is
//! asked of the run's own, never of the thread's `CURRENT_RT`, which may be another run's.
//!
//! **The borrow.** A thunk takes the hook `&mut` for one [`KernelPark`] call, marking the record
//! borrowed meanwhile. During it the thread runs no guest code: the kernel parks the thread in place
//! (`park_native`), and executor 2 carries another task's activation onto a parked thread only for a
//! held run (`Next::Carry`), which never reaches the kernel. A borrow never spans a fiber switch:
//! the call runs on this stack, and no switch happens inside it. A thunk that finds the record
//! lent, or a give-back that does not find this record lending this very hook, fails closed
//! (`Malformed`) in every build and asserts in a debug one.

use crate::{mem, slot_record, CompiledModule, FuelCell, JitFrameLoc, TrapKind, VmCtx};
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

/// The thread's record (the module docs): the run its vCPU is of, by its end word (null when this
/// thread runs no kernel vCPU), its hook, which a thunk borrows for one call (`lent`), and the vCPU's
/// own fiber runtime.
struct Record {
    end: Cell<*const AtomicI64>,
    hook: Cell<Option<NonNull<dyn KernelPark>>>,
    /// The hook is lent to a thunk now. It stays named in `hook` meanwhile, so the give-back can
    /// check that what it hands back is this record's own hook, not another vCPU's of the run.
    lent: Cell<bool>,
    /// The vCPU's own fiber runtime (`fiber_rt::FiberRuntime`), null for a run whose module has no
    /// fiber op: whether a park is a fiber's is asked of it, never of the thread's `CURRENT_RT`, which
    /// may be another run's (a fiber of a run that called the host procedure this run is in).
    fibers: Cell<*const core::ffi::c_void>,
}

/// What [`set_record`] writes and answers: the record's three cells.
#[derive(Clone, Copy)]
struct Fields {
    end: *const AtomicI64,
    hook: Option<NonNull<dyn KernelPark>>,
    lent: bool,
    fibers: *const core::ffi::c_void,
}

thread_local! {
    static RECORD: Record = const {
        Record {
            end: Cell::new(std::ptr::null()),
            hook: Cell::new(None),
            lent: Cell::new(false),
            fibers: Cell::new(std::ptr::null()),
        }
    };
}

// The record's accessors. Each is `#[inline(never)]` and reads or writes the record's cells, the rule
// #1466 made for per-thread state reached on a fiber's stack, as `fiber_rt::current` is: the thread's
// address is resolved inside the call, on the thread running it, and since the call reads memory a
// call between two of them (a park) keeps them apart. No address of the record is held across a
// park: one so held would be the thread's that took it, whichever thread runs on after.

/// The run this thread's record names: its end word, null when this thread runs no kernel vCPU.
#[inline(never)]
#[cfg(test)]
fn record_end() -> *const AtomicI64 {
    RECORD.with(|r| r.end.get())
}

/// What a thunk's take of the hook for the run of end word `end` found.
enum Taken {
    /// The record names another run, or none.
    NotOurs,
    /// The hook is already borrowed.
    Borrowed,
    Hook(NonNull<dyn KernelPark>),
}

/// Take the hook of this thread's record if it names the run of end word `end`.
#[inline(never)]
fn take_hook(end: *const AtomicI64) -> Taken {
    RECORD.with(|r| {
        if !std::ptr::eq(r.end.get(), end) {
            return Taken::NotOurs;
        }
        match (r.lent.get(), r.hook.get()) {
            (false, Some(h)) => {
                r.lent.set(true);
                Taken::Hook(h)
            }
            _ => Taken::Borrowed,
        }
    })
}

/// Give `hook` back to this thread's record, if it is the one that lent it: it names the run of end
/// word `end`, is lent, and lent this very hook (another vCPU's record of the same run, also
/// mid-borrow, lent its own). `false`, and nothing written, otherwise.
#[inline(never)]
fn give_back(end: *const AtomicI64, hook: NonNull<dyn KernelPark>) -> bool {
    RECORD.with(|r| {
        let own = r
            .hook
            .get()
            .is_some_and(|h| std::ptr::addr_eq(h.as_ptr(), hook.as_ptr()));
        let lent = std::ptr::eq(r.end.get(), end) && r.lent.get() && own;
        if lent {
            r.lent.set(false);
        }
        lent
    })
}

/// Set this thread's record, answering what it held.
#[inline(never)]
fn set_record(f: Fields) -> Fields {
    RECORD.with(|r| Fields {
        end: r.end.replace(f.end),
        hook: r.hook.replace(f.hook),
        lent: r.lent.replace(f.lent),
        fibers: r.fibers.replace(f.fibers),
    })
}

/// Whether this thread's record names the run of end word `end` and its vCPU is inside one of its
/// own fibers now.
#[inline(never)]
fn in_own_fiber(end: *const AtomicI64) -> bool {
    RECORD.with(|r| {
        let rt = r.fibers.get();
        #[cfg(fiber_rt)]
        let fiber = !rt.is_null()
            // SAFETY: the record names this vCPU's own runtime while it is published (`Published`).
            && unsafe { crate::fiber_rt::runs_a_fiber(rt.cast()) };
        #[cfg(not(fiber_rt))]
        let fiber = !rt.is_null();
        std::ptr::eq(r.end.get(), end) && fiber
    })
}

/// Whether code running on `vm` runs as a kernel vCPU **inside a fiber**: a park there is a fiber's,
/// which the kernel does not take yet (`native_scope`'s `fiber parks`; #3078 slices 3 and 4).
pub fn in_fiber(vm: &VmCtx) -> bool {
    vm.kernel && in_own_fiber(vm.domain_key() as *const AtomicI64)
}

/// Route a §12 thunk's op to the kernel, if `vm` is a kernel run's context: `None` otherwise, and the
/// thunk goes on as the OS-thread runtime. A trap is raised on `vm`, whose next
/// check unwinds. A wait made in a fiber is a fiber's park, which `fiber parks` declines: reaching
/// one fails closed.
#[cfg(fiber_rt)]
#[inline(always)]
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
/// ([`KernelPark::park_call`]): `None` otherwise. `parks`: whether the call parks its caller (a
/// rewound pipe or stdin park, a punted call), not only wakes another's (a region post's `signal`). A
/// trap is raised on `vm`, whose next check unwinds. A park made in a fiber is the fiber's, which
/// `fiber parks` declines: reaching one fails closed.
///
/// # Safety
/// The caller is the embedder's `call.cap` thunk of code running on `vm`, on this thread, and holds
/// no other reference to the hook.
pub unsafe fn park_call(vm: &VmCtx, parks: bool, call: &mut dyn std::any::Any) -> Option<[i64; 2]> {
    if parks && in_fiber(vm) {
        vm.raise(temen_ir::trap_code::MALFORMED);
        return Some([0, 0]);
    }
    // SAFETY: the caller's contract.
    unsafe { through_hook(vm, |hook| hook.park_call(call)) }
}

/// The one way into a vCPU's hook ([`routed`], [`park_call`]): run `f` on the hook this thread's
/// record holds for `vm`'s run, raising its error on `vm`.
///
/// # Safety
/// As [`park_call`].
#[inline(always)]
unsafe fn through_hook(
    vm: &VmCtx,
    f: impl FnOnce(&mut dyn KernelPark) -> Result<[i64; 2], i64>,
) -> Option<[i64; 2]> {
    if !vm.kernel {
        return None;
    }
    let end = vm.domain_key() as *const AtomicI64;
    let hook = match take_hook(end) {
        Taken::Hook(h) => h,
        // A kernel run's context on a thread whose record names no vCPU of its run: never, as only a
        // vCPU of the run runs its code (the module docs). Sent on to this crate's own runtime, its op
        // would run on the compile's dormant thread domain, so it fails closed.
        Taken::NotOurs => {
            debug_assert!(false, "a kernel run's context off its vCPUs' threads");
            vm.raise(temen_ir::trap_code::MALFORMED);
            return Some([0, 0]);
        }
        Taken::Borrowed => {
            // Re-entry while a borrow is live: never on a kernel run (the module docs).
            debug_assert!(false, "a kernel hook reached while it is borrowed");
            vm.raise(temen_ir::trap_code::MALFORMED);
            return Some([0, 0]);
        }
    };
    // SAFETY: the record is marked lent, so nothing else reaches the hook meanwhile.
    let out = f(unsafe { &mut *hook.as_ptr() });
    // Given back to the record of the thread running now, which must be the one that lent it.
    if !give_back(end, hook) {
        // The borrow spanned a switch onto another thread: never (the module docs). The lending
        // thread's record stays borrowed, so its next thunk fails closed too.
        debug_assert!(false, "a kernel hook's borrow spanned a switch");
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
    prev: Fields,
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
        #[cfg(fiber_rt)]
        let fibers = fibers.map(|mut rt| {
            let prev = crate::fiber_rt::set_current(&mut *rt);
            (rt, prev)
        });
        #[cfg(fiber_rt)]
        let own = fibers
            .as_ref()
            .map_or(std::ptr::null(), |(rt, _)| &**rt as *const _ as *const _);
        #[cfg(not(fiber_rt))]
        let own = std::ptr::null();
        let prev = set_record(Fields {
            end,
            hook: NonNull::new(hook),
            lent: false,
            fibers: own,
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
    /// #2958 slice 6 — the first trap-time capture (`pc`, return addresses, fiber) a vCPU published,
    /// the root's or a spawned one's, which [`Self::run_vcpu`] takes on the vCPU's own thread before
    /// the thread-local is lost with it ([`Self::trap_backtrace`]). First wins, as the
    /// JIT's own hub's does (`publish_trap_capture`): when two vCPUs trap at once, any trapping frame's
    /// chain is a valid kill backtrace (#2499). A leaf lock, held for one `get_or_insert`.
    trap: std::sync::Mutex<Option<(usize, Vec<usize>, i64)>>,
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
    /// kernel does not run (§14 nesting, a metered module with fibers, durability, a target without the fiber runtime);
    /// `Some(None)` when the OS refuses the window or a protection it needs (#2305, #2286), a run that
    /// ends as one the process cannot hold.
    pub fn new(
        cm: CompiledModule,
        armed: Option<Box<FuelCell>>,
        init_mem: &[u8],
        node: Option<std::sync::Arc<dyn crate::BudgetNode>>,
    ) -> Option<Option<Started>> {
        // A metered fiber is refused for memory safety, not only accounting: each vCPU's fuel cell
        // lives on its thread's stack (`temen_run::jit_kernel`'s `tier_run`), a fiber's code charges
        // its creator's cell wherever it runs (#3080), and a fiber may run after its creator ended.
        // `fiber fuel` declines such a run before this; this keeps it refused whatever admitted it.
        if !crate::fiber_supported()
            || cm.nests
            || cm.durable
            || cm.call_tramp.is_none() && (cm.spawns || cm.fibers)
            || cm.fibers && armed.is_some()
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
        let mut root = VmCtx::new(cm.instance, cm.window_at(window.base()));
        root.kernel = true;
        let run = KernelRun {
            cm,
            window,
            root,
            _armed: armed,
            ctxs: Mutex::new(Vec::new()),
            #[cfg(fiber_rt)]
            fibers,
            trap: std::sync::Mutex::new(None),
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

    /// #3078 slice 2 — the run has ended and no thread runs its guest code: every fiber of its table
    /// goes, its stack and its `mem` charge with it, as today's route's run end does
    /// (`SharedFiberTable::end`). A fault's recovery jumps past a resume's return, so a slot it
    /// abandoned keeps an `Arc` nothing ever drops: waiting for the last `Arc<KernelRun>` would keep
    /// that stack for the life of the process, and hand the charges back only after the run's host
    /// went back to its embedder.
    pub fn end_fibers(&self) {
        #[cfg(fiber_rt)]
        if let Some(f) = &self.fibers {
            f.table.end();
        }
    }

    /// #2958 slice 6 — the run's trap-time backtrace, innermost frame first, and the fiber that was
    /// running (`-1` for none): the first capture a vCPU published ([`Self::run_vcpu`]). Empty and
    /// `None` when none was, as when no vCPU trapped or its code carried no `-g`. A memory fault is
    /// captured whatever `-g` says, so its backtrace can be empty with the fiber `Some(-1)`. Read
    /// once every vCPU has gone; it takes the capture.
    pub fn trap_backtrace(&self) -> (Vec<JitFrameLoc>, Option<i64>) {
        let cap = self.trap.lock().unwrap_or_else(|e| e.into_inner()).take();
        match cap {
            Some((pc, rets, fiber)) => (self.cm.trap_backtrace(pc, &rets), Some(fiber)),
            None => (Vec::new(), None),
        }
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
        debug_assert_eq!(
            vm.domain_key(),
            end as *const AtomicI64 as usize,
            "one end word per run"
        );
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
        // A capture left on this thread by code that is not this run's: a held domain's activation
        // run inline from outside (`HeldSite::activate`), whose handler faulted, or a durable
        // freeze's deferred inline child, which runs after its root took the root's capture. Neither
        // takes what it made, and the kernel root runs on the caller's thread, so without this a
        // trap that makes no capture here (a host-raised fault, a trap without `-g`) would publish
        // theirs as this run's. Until it answers `None`: on Windows a memory fault and an explicit
        // trap capture into two thread-locals, and one take clears one of them.
        while mem::take_trap_frame().is_some() {}
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
        // This vCPU's own trap-time capture, taken on its thread, which may end with it, so none is
        // left there for a later run to read. Cleared above, a capture is made only where this vCPU
        // trapped, and the first one published is the run's ([`Self::trap_backtrace`]), whatever
        // end word won. A kill door's `DOMAIN_DONE` can win the word over an epoch trap's own raise,
        // and the run still ends in a trap whose capture this is. A vCPU stopped by another's end
        // made none.
        if let Some(cap) = mem::take_trap_frame() {
            self.trap
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .get_or_insert(cap);
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
    use crate::fiber_rt::{FiberRuntime, SharedFiberTable};
    use crate::vmctx::{InstanceAddrs, Window};
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    /// A hook whose `park` runs `then` first, as code on the vCPU's thread during its park would,
    /// and which counts the call parks that reach it.
    struct Then<F: FnMut() -> Option<[i64; 2]>>(F, usize);

    impl<F: FnMut() -> Option<[i64; 2]>> KernelPark for Then<F> {
        fn park(&mut self, _: NativeStop) -> Result<[i64; 2], i64> {
            Ok((self.0)().unwrap_or([7, 7]))
        }
        fn park_call(&mut self, _: &mut dyn std::any::Any) -> Result<[i64; 2], i64> {
            self.1 += 1;
            Ok([5, 5])
        }
    }

    /// A context of a kernel run (`kernel`) or of another (`!kernel`), over `end`.
    fn context(end: &AtomicI64, kernel: bool) -> VmCtx {
        let mut root = VmCtx::new(InstanceAddrs::NONE, Window::NONE);
        root.kernel = kernel;
        // SAFETY: each test's `end` outlives the context.
        unsafe { root.vcpu_ending(end, std::ptr::null_mut()) }
    }

    /// A vCPU runtime over a fresh table, running one fiber now (its slot the innermost resume), as
    /// the vCPU's own code is when a fiber of it makes an op.
    fn in_a_fiber() -> Box<FiberRuntime> {
        let table = Arc::new(SharedFiberTable::new(
            4,
            temen_ir::durable_abi::ShadowArena::EMPTY,
            None,
        ));
        let mut rt = Box::new(FiberRuntime::new(table, 0, 0));
        // SAFETY: the fiber's body never runs: it only stands for the one running.
        let fiber = unsafe { crate::fiber_rt::make_task_fiber(64 << 10, |_| {}) }.expect("a stack");
        rt.push_active(crate::fiber_rt::FiberSlot::platform(fiber));
        rt
    }

    const NOTIFY: NativeStop = NativeStop::Notify { addr: 0, count: 1 };
    const WAIT: NativeStop = NativeStop::Wait {
        addr: 0,
        expected: 0,
        wide: false,
        timeout: -1,
    };

    /// #3078 slice 2 — a thunk that reaches the hook while it is borrowed fails closed: a debug
    /// build asserts, and any other traps the domain `Malformed` rather than alias the borrow.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "while it is borrowed"))]
    fn a_hook_reached_while_it_is_borrowed_fails_closed() {
        let end = AtomicI64::new(0);
        let vm = context(&end, true);
        let vm_at = &vm as *const VmCtx;
        // SAFETY: `vm` outlives the hook.
        let mut hook = Then(|| routed(unsafe { &*vm_at }, NOTIFY), 0);
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, None) };
        assert_eq!(
            record_end(),
            &end as *const AtomicI64,
            "the record names this run"
        );
        let answer = routed(&vm, NOTIFY);
        drop(published);
        assert_eq!(answer, Some([0, 0]));
        assert_eq!(vm.ended(), Some(temen_ir::trap_code::MALFORMED));
        assert!(record_end().is_null(), "the record is cleared");
    }

    /// #3078 slice 2 — a context of no kernel run on a kernel vCPU's thread (a nested run's, a held
    /// domain's activation's) never reaches the record: its ops take the JIT's own runtime.
    #[test]
    fn another_runs_context_on_a_kernel_vcpus_thread_is_not_the_kernels() {
        let (end, other_end) = (AtomicI64::new(0), AtomicI64::new(0));
        let other = context(&other_end, false);
        let mut hook = Then(|| None, 0);
        // SAFETY: `hook` outlives the guard.
        let _published = unsafe { Published::new(&end, &mut hook, None) };
        assert!(!other.on_kernel());
        assert_eq!(routed(&other, NOTIFY), None);
        assert_eq!(other.ended(), None);
    }

    /// #3078 slice 2 (review R2) — a kernel run's context whose thread's record names another run, or
    /// none, fails closed: sent on to the JIT's own runtime, its op would run on the compile's
    /// dormant thread domain.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "off its vCPUs' threads"))]
    fn a_kernel_context_off_its_vcpus_threads_fails_closed() {
        let (end, other_end) = (AtomicI64::new(0), AtomicI64::new(0));
        let stray = context(&other_end, true);
        let mut hook = Then(|| None, 0);
        // SAFETY: `hook` outlives the guard.
        let _published = unsafe { Published::new(&end, &mut hook, None) };
        assert_eq!(routed(&stray, NOTIFY), Some([0, 0]));
        assert_eq!(stray.ended(), Some(temen_ir::trap_code::MALFORMED));
    }

    /// #3078 slice 2 (review B3) — the two park backstops `fiber parks` keeps unreachable, in any
    /// build: a futex wait and a call park made by a vCPU inside one of its own fibers trap the
    /// domain `Malformed`, and never reach the hook. A notify there is the task's stop, as anywhere.
    #[test]
    fn a_park_made_in_a_fiber_meets_its_backstop() {
        let end = AtomicI64::new(0);
        let vm = context(&end, true);
        let parks = std::cell::Cell::new(0);
        let mut hook = Then(
            || {
                parks.set(parks.get() + 1);
                None
            },
            0,
        );
        let hook_at = &hook as *const Then<_>;
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, Some(in_a_fiber())) };
        assert!(in_fiber(&vm), "the vCPU's own runtime runs a fiber");
        // A region post's wake (not a park) reaches the hook; a call park does not. Each on a context
        // of its own over the run's end, which nothing has ended yet.
        let wake = context(&end, true);
        // SAFETY: a call's wake on this thread, with no other reference to the hook.
        assert_eq!(unsafe { park_call(&wake, false, &mut ()) }, Some([5, 5]));
        assert_eq!(
            end.load(Ordering::Relaxed),
            0,
            "a wake in a fiber is the task's"
        );
        // SAFETY: as above.
        assert_eq!(unsafe { park_call(&vm, true, &mut ()) }, Some([0, 0]));
        assert_eq!(
            vm.ended(),
            Some(temen_ir::trap_code::MALFORMED),
            "the call park's backstop"
        );
        end.store(0, Ordering::Relaxed);
        let vm = context(&end, true);
        assert_eq!(
            routed(&vm, NOTIFY),
            Some([7, 7]),
            "a notify is the task's stop"
        );
        assert_eq!(vm.ended(), None);
        assert_eq!(routed(&vm, WAIT), Some([0, 0]));
        assert_eq!(
            vm.ended(),
            Some(temen_ir::trap_code::MALFORMED),
            "the wait's backstop"
        );
        drop(published);
        // SAFETY: the guard is gone; nothing else holds the hook.
        let call_parks = unsafe { (*hook_at).1 };
        assert_eq!(
            (parks.get(), call_parks),
            (1, 1),
            "only the notify and the wake reached it"
        );
    }

    /// #3078 slice 2 (review B2) — whether a park is a fiber's is asked of the run's own runtime: a
    /// run with no fiber op, on a thread whose `CURRENT_RT` is running a fiber of another run, parks.
    #[test]
    fn a_fiberless_run_on_another_runs_fiber_parks() {
        let end = AtomicI64::new(0);
        let vm = context(&end, true);
        let outer = Box::into_raw(in_a_fiber());
        let prev = crate::fiber_rt::set_current(outer);
        let mut hook = Then(|| None, 0);
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, None) };
        assert!(!in_fiber(&vm));
        assert_eq!(routed(&vm, WAIT), Some([7, 7]), "the wait went to the hook");
        drop(published);
        crate::fiber_rt::set_current(prev);
        // SAFETY: from `Box::into_raw` above, no longer published.
        drop(unsafe { Box::from_raw(outer) });
    }

    /// #3078 slice 2 (review B3) — a borrow whose park ended on a thread whose record is not the
    /// lender's (a switch moved it, which never happens on a kernel run) writes nothing there and
    /// fails closed: the hook given back is checked against the record of the thread running now.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "spanned a switch"))]
    fn a_borrow_that_ends_on_another_record_fails_closed() {
        let (end, elsewhere) = (AtomicI64::new(0), AtomicI64::new(0));
        let vm = context(&end, true);
        // The park ends with another run's record on this thread, as a switch onto it would leave.
        let mut hook = Then(
            || {
                set_record(Fields {
                    end: &elsewhere,
                    hook: None,
                    lent: true,
                    fibers: std::ptr::null(),
                });
                None
            },
            0,
        );
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, None) };
        assert_eq!(routed(&vm, NOTIFY), Some([0, 0]));
        drop(published);
        assert_eq!(vm.ended(), Some(temen_ir::trap_code::MALFORMED));
    }

    /// #3078 slice 2 (review round 2) — a borrow whose park ended on another vCPU's record of the
    /// same run, itself mid-borrow (a switch moved it, which never happens on a kernel run), is not
    /// given back there: that record lent a hook of its own, and the give-back compares the hook, so
    /// it writes nothing and fails closed rather than cross-wire the two vCPUs' hooks.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "spanned a switch"))]
    fn a_borrow_that_ends_on_a_sibling_vcpus_borrowed_record_fails_closed() {
        let end = AtomicI64::new(0);
        let vm = context(&end, true);
        let mut sibling = Then(|| None, 0);
        let sibling_hook: *mut (dyn KernelPark + '_) = &mut sibling;
        // SAFETY: only the pointer is kept, as a record's; it is never called.
        let sibling_hook: *mut dyn KernelPark = unsafe { std::mem::transmute(sibling_hook) };
        // The park ends on a record of the same run whose own hook is lent too.
        let mut hook = Then(
            || {
                set_record(Fields {
                    end: &end,
                    hook: NonNull::new(sibling_hook),
                    lent: true,
                    fibers: std::ptr::null(),
                });
                None
            },
            0,
        );
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, None) };
        assert_eq!(routed(&vm, NOTIFY), Some([0, 0]));
        drop(published);
        assert_eq!(vm.ended(), Some(temen_ir::trap_code::MALFORMED));
    }

    /// #3078 slice 2 (review R1) — a module with fibers compiled with fuel checks is never a kernel
    /// run, whatever admitted it: a vCPU's cell lives on its thread's stack, and a fiber charges its
    /// creator's cell wherever it runs, after its creator ended too (#3080). Unmetered, it is one.
    #[test]
    fn a_metered_module_with_fibers_is_no_kernel_run() {
        let m = temen_text::parse_module(
            "memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vf = ref.func 1
  vk = cont.new vf vz
  vs, vv = cont.resume vk vz
  vt = thread.spawn 1 vz vz
  vts, vtv = thread.join vt
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  return va
  }
}
",
        )
        .expect("parse");
        let compile = |fuel: Option<*mut FuelCell>| {
            CompiledModule::compile(
                &m,
                0,
                crate::empty_cap_thunk,
                std::ptr::null_mut(),
                temen_ir::DEFAULT_RESERVED_LOG2,
                None,
                None,
                fuel,
                None,
                None,
                0,
            )
            .expect("compile")
        };
        let mut armed = FuelCell::fixed(0);
        let metered = compile(Some(&mut *armed as *mut FuelCell));
        assert!(KernelRun::new(metered, Some(armed), &[], None).is_none());
        let unmetered = compile(None);
        assert!(matches!(
            KernelRun::new(unmetered, None, &[], None),
            Some(Some(_))
        ));
    }

    /// #3078 slice 2 (review B3) — `cont.resume.block` of a target its resume finds event-parked
    /// (`FIBER_PARKED`), which no fiber on an admitted run can be (`fiber parks`), fails closed on a
    /// kernel vCPU instead of idling on the compile's dormant thread domain.
    #[test]
    fn a_blocking_resume_of_a_parked_fiber_meets_its_backstop_on_a_kernel_vcpu() {
        let end = AtomicI64::new(0);
        let vm = context(&end, true);
        let table = Arc::new(SharedFiberTable::new(
            4,
            temen_ir::durable_abi::ShadowArena::EMPTY,
            None,
        ));
        // SAFETY: the body runs under the resume below, with this thread's runtime published.
        let fiber = unsafe {
            crate::fiber_rt::make_task_fiber(64 << 10, |_| {
                let slot = crate::fiber_rt::current_fiber_slot().expect("running in its slot");
                crate::fiber_rt::fiber_event_park(&slot, None, crate::fiber_rt::ParkOn::Event);
            })
        }
        .expect("a stack");
        let handle = table.create_for_test(fiber).expect("a slot");
        let rt = Box::new(FiberRuntime::new(Arc::clone(&table), 0, 0));
        let mut hook = Then(|| None, 0);
        // SAFETY: `hook` outlives the guard.
        let published = unsafe { Published::new(&end, &mut hook, Some(rt)) };
        let mut status = 0i64;
        // SAFETY: `vm` is a live context and `status` a live cell; this thread's runtime is published.
        unsafe { crate::os_thread_rt::fiber_resume_block(&vm, handle, 0, &mut status) };
        drop(published);
        assert_eq!(status, 3, "its resume found it parked");
        assert_eq!(vm.ended(), Some(temen_ir::trap_code::MALFORMED));
        table.end();
    }
}
