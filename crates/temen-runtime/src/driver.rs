//! Unified `DomainWorld` + Cooperative & Parallel `Carrier` drivers (`#3187` / `#3196`).
//!
//! One state machine (`DomainWorld` over `temen-kernel`'s `Domain`, `WaitIndex`, and `BudgetTree`)
//! drives BOTH [`crate::RunMode::Cooperative`] and [`crate::RunMode::Parallel`] Carriers with
//! identical semantics for threads, futexes, migratable guest fibers (`FIBER_PARKED` = 3),
//! Carrier-owned fuel metering (Decision 30), and domain-wide termination (Invariant 6).

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};

use temen_bytecode::{step as bc_step, Code, ExecState, MemView, StepConfig};
use temen_kernel::wait::RegisteredPark;
use temen_kernel::{
    BudgetId, BudgetTree, CapOutcome, CarrierId, ContextExit, ContextId, ContextKind, Domain,
    DomainControl, DomainId, FiberResumeClaim, FuelCell, MemSpaceId, ParkDecision, PollAction,
    Step, Trap, TypeId, WaitIndex, WaitKey, WaitSlot, WaitableId, WaiterSpec, WakeReason,
    YieldReason, ERRNO_EBADF, FIBER_PARKED, FIBER_RETURNED, FIBER_SUSPENDED, WAIT_NOT_EQUAL,
    WAIT_OK, WAIT_TIMED_OUT,
};
use temen_mask::Window;
use temen_mem::Region;

use crate::{CapHandler, Engine, RunConfig, RunMode};

/// Result of running a domain entry to completion.
pub struct RunOutput {
    pub exit: ContextExit,
    pub fuel_used: u64,
    pub memory: Option<Arc<Region>>,
}

/// Generational 32-bit thread handle packing (`[gen:16][slot:16]`, generation >= 1).
#[inline]
const fn pack_thread_handle(slot: u16, gen: u16) -> i32 {
    (((gen as u32) << 16) | (slot as u32)) as i32
}

#[inline]
const fn unpack_thread_handle(handle: i32) -> (u16, u16) {
    let raw = handle as u32;
    ((raw & 0xFFFF) as u16, (raw >> 16) as u16)
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) type Deadline = Instant;
#[cfg(target_arch = "wasm32")]
pub(crate) type Deadline = u64;

pub(crate) struct FiberRecord {
    pub(crate) handle: i64,
    pub(crate) exec: Option<ExecState>,
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    pub(crate) kfiber: Option<temen_jit::kernel_engine::KernelFiber>,
    pub(crate) started: bool,
    pub(crate) parked: Option<FiberParkInfo>,
}

pub(crate) struct FiberParkInfo {
    pub(crate) key: WaitKey,
    pub(crate) slot: Arc<WaitSlot>,
    pub(crate) deadline: Option<Deadline>,
    pub(crate) ctx: ContextId,
    pub(crate) reexec_on_wake: bool,
}

struct ThreadRecord {
    gen: u16,
    waitable: WaitableId,
    exit: Option<ContextExit>,
    joined: bool,
}

pub(crate) struct ThreadTable {
    slots: Vec<ThreadRecord>,
    issued_gen: Vec<u16>,
    next_gen: u16,
}

impl ThreadTable {
    fn new() -> ThreadTable {
        ThreadTable {
            slots: Vec::new(),
            issued_gen: Vec::new(),
            next_gen: 1,
        }
    }

    pub(crate) fn alloc(&mut self, waitable: WaitableId) -> (i32, u16) {
        let gen = self.next_gen;
        self.next_gen = self.next_gen.wrapping_add(1).max(1);
        let slot = self.slots.len() as u16;
        self.slots.push(ThreadRecord {
            gen,
            waitable,
            exit: None,
            joined: false,
        });
        self.issued_gen.push(gen);
        (pack_thread_handle(slot, gen), slot)
    }
}

pub(crate) enum ThreadJoinLookup {
    Finished(ContextExit),
    Running(WaitableId),
    Spent,
    Fault,
}

/// Live runtime state of one domain (shared across all Carriers executing contexts in it).
pub struct DomainWorld {
    pub domain: Arc<Domain>,
    pub waits: Arc<WaitIndex>,
    pub budgets: Arc<BudgetTree>,
    pub ctl: DomainControl,
    pub(crate) engine: Engine,
    pub(crate) region: Option<Arc<Region>>,
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    pub(crate) kernel_mem: Option<Arc<temen_jit::kernel_engine::KernelMemory>>,
    window: Option<Window>,
    readonly_ranges: Vec<(u64, u64)>,
    null_guard: u64,
    pub(crate) mem_space: MemSpaceId,
    pub(crate) caps: Arc<dyn CapHandler>,
    fuel_chunk: u64,
    quantum: u32,
    pub(crate) metered: bool,
    pub(crate) next_carrier: AtomicU32,
    next_waitable: AtomicU64,
    pub(crate) fibers: Mutex<HashMap<ContextId, FiberRecord>>,
    pub(crate) threads: Mutex<ThreadTable>,
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) os_threads: Mutex<Vec<std::thread::JoinHandle<()>>>,
}

impl DomainWorld {
    /// Instantiate a fresh [`DomainWorld`] for `engine` under `cfg`.
    pub fn new(engine: Engine, cfg: RunConfig) -> Result<Arc<DomainWorld>, Trap> {
        let metered = cfg.limits.fuel != u64::MAX;
        let budgets = Arc::new(BudgetTree::new(cfg.limits));

        // Charge initial linear memory window + root task against the 3-dimension BudgetTree.
        let mut region = None;
        let mut window = None;
        let mut readonly_ranges = Vec::new();
        let mut null_guard = 0u64;
        #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
        let mut kernel_mem = None;

        match &engine {
            Engine::Bytecode(code) => {
                if let Some(size_log2) = code.memory_size_log2 {
                    let bytes = 1u64
                        .checked_shl(u32::from(size_log2))
                        .ok_or(Trap::MemoryFault)?;
                    if !budgets.charge_mem(BudgetId::ROOT, bytes) {
                        return Err(Trap::MemoryFault);
                    }
                    let reg = Arc::new(Region::new(bytes, 4096));
                    for d in &code.initial_data {
                        if d.offset
                            .checked_add(d.bytes.len() as u64)
                            .is_none_or(|end| end > bytes)
                        {
                            return Err(Trap::MemoryFault);
                        }
                        reg.write_from(d.offset, &d.bytes);
                        if d.readonly && !d.bytes.is_empty() {
                            readonly_ranges.push((d.offset, d.offset + d.bytes.len() as u64));
                        }
                    }
                    // Invariant 13 (#1094): unconditionally guard [0, POWERBOX_NULL_GUARD) when
                    // the window is at least that large, matching `Mem::seed_null_guard`.
                    let guard = temen_ir::module_null_guard();
                    null_guard = if guard > 0 && guard <= bytes {
                        guard
                    } else {
                        0
                    };
                    region = Some(reg);
                    window = Some(Window::new(size_log2));
                }
            }
            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
            Engine::Cranelift(compiled) => {
                if let Some(size_log2) = compiled.memory_size_log2 {
                    let bytes = 1u64
                        .checked_shl(u32::from(size_log2))
                        .ok_or(Trap::MemoryFault)?;
                    if !budgets.charge_mem(BudgetId::ROOT, bytes) {
                        return Err(Trap::MemoryFault);
                    }
                    let kmem = temen_jit::kernel_engine::KernelMemory::try_new(
                        size_log2,
                        &compiled.initial_data,
                    )
                    .ok_or(Trap::MemoryFault)?;
                    kernel_mem = Some(kmem);
                } else {
                    kernel_mem = Some(temen_jit::kernel_engine::KernelMemory::empty());
                }
            }
        }

        if !budgets.charge_tasks(BudgetId::ROOT, 1) {
            return Err(Trap::ThreadFault);
        }

        let domain = Domain::new(DomainId(0), BudgetId::ROOT, cfg.hooks);

        // Wire `WaitIndex`'s `on_wake` hook to promote woken `GuestFiber`s to `WakeReady` so any
        // Carrier can claim and resume them (including across OS threads).
        let dom_for_wake = Arc::clone(&domain);
        let on_wake = Arc::new(move |spec: WaiterSpec, reason: WakeReason| {
            if spec.kind == ContextKind::GuestFiber {
                dom_for_wake.wake_fiber_by_ctx(spec.ctx, reason);
            }
        });
        let waits = Arc::new(WaitIndex::new(Some(on_wake)));
        let ctl = DomainControl::new(Arc::clone(&domain), Arc::clone(&waits));

        Ok(Arc::new(DomainWorld {
            domain,
            waits,
            budgets,
            ctl,
            engine,
            region,
            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
            kernel_mem,
            window,
            readonly_ranges,
            null_guard,
            mem_space: MemSpaceId(1),
            caps: cfg.caps,
            fuel_chunk: cfg.fuel_chunk.max(1),
            quantum: cfg.quantum.max(1),
            metered,
            next_carrier: AtomicU32::new(1),
            next_waitable: AtomicU64::new(1),
            fibers: Mutex::new(HashMap::new()),
            threads: Mutex::new(ThreadTable::new()),
            #[cfg(not(target_arch = "wasm32"))]
            os_threads: Mutex::new(Vec::new()),
        }))
    }

    #[inline]
    fn code(&self) -> &Arc<Code> {
        match self.engine {
            Engine::Bytecode(ref code) => code,
            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
            Engine::Cranelift(_) => unreachable!("code() called on Engine::Cranelift"),
        }
    }

    #[inline]
    fn mem_view(&self) -> MemView<'_> {
        MemView {
            region: self.region.as_deref(),
            window: self.window,
            readonly_ranges: &self.readonly_ranges,
            null_guard: self.null_guard,
        }
    }

    pub(crate) fn alloc_waitable(&self) -> WaitableId {
        WaitableId(self.next_waitable.fetch_add(1, Ordering::Relaxed))
    }

    pub(crate) fn draw_fuel(&self, cell: &mut FuelCell) -> bool {
        if !self.metered {
            return true;
        }
        let got = self.budgets.draw_fuel(self.domain.budget, self.fuel_chunk);
        if got == 0 {
            return false;
        }
        cell.remaining = cell.remaining.saturating_add(got);
        true
    }

    pub(crate) fn refund_fuel(&self, cell: &mut FuelCell) {
        if self.metered {
            let left = cell.take_remaining();
            self.budgets.refund_fuel(self.domain.budget, left);
        }
    }

    pub(crate) fn make_fuel_cell(&self) -> FuelCell {
        if self.metered {
            let got = self.budgets.draw_fuel(self.domain.budget, self.fuel_chunk);
            FuelCell::metered(got)
        } else {
            FuelCell::unlimited()
        }
    }

    pub(crate) fn check_futex_eq(&self, addr: u64, expected: i64, width64: bool) -> bool {
        if !self.domain.poll_flags().is_empty() {
            return false;
        }
        if let Some(ref reg) = self.region {
            return if width64 {
                reg.atomic_load(addr, 8) == expected as u64
            } else {
                (reg.atomic_load(addr, 4) as u32) == (expected as u32)
            };
        }
        #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
        if let Some(ref kmem) = self.kernel_mem {
            let Some(got) = kmem.atomic_load(addr, width64) else {
                return false;
            };
            return if width64 {
                got == expected as u64
            } else {
                (got as u32) == (expected as u32)
            };
        }
        false
    }

    pub(crate) fn lookup_thread_join(&self, handle: i32) -> ThreadJoinLookup {
        let (slot, gen) = unpack_thread_handle(handle);
        if gen == 0 {
            return ThreadJoinLookup::Fault;
        }
        let mut tbl = self.threads.lock().unwrap_or_else(|e| e.into_inner());
        let Some(&issued) = tbl.issued_gen.get(slot as usize) else {
            return ThreadJoinLookup::Fault;
        };
        if gen > issued {
            return ThreadJoinLookup::Fault;
        }
        let rec = &mut tbl.slots[slot as usize];
        if rec.gen != gen || rec.joined {
            return ThreadJoinLookup::Spent;
        }
        if let Some(ref exit) = rec.exit {
            rec.joined = true;
            ThreadJoinLookup::Finished(exit.clone())
        } else {
            ThreadJoinLookup::Running(rec.waitable)
        }
    }

    pub(crate) fn finish_thread(&self, slot: u16, exit: ContextExit) {
        let waitable = {
            let mut tbl = self.threads.lock().unwrap_or_else(|e| e.into_inner());
            let rec = &mut tbl.slots[slot as usize];
            rec.exit = Some(exit.clone());
            rec.waitable
        };
        self.budgets.refund_tasks(self.domain.budget, 1);
        if exit.is_domain_terminal() {
            self.ctl.end(exit);
        } else {
            self.waits
                .notify_all(WaitKey::Object(waitable), WakeReason::Ready);
        }
    }

    pub(crate) fn expire_fiber_if_timed_out(&self, handle: i64) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let info = {
                let fibers = self.fibers.lock().unwrap_or_else(|e| e.into_inner());
                let (slot, _) = temen_kernel::unpack_fiber_handle(handle);
                let _ = slot;
                // Check all parked fibers whose deadline elapsed.
                let now = Instant::now();
                let mut expired = Vec::new();
                for rec in fibers.values() {
                    if let Some(ref p) = rec.parked {
                        if p.deadline.is_some_and(|dl| now >= dl) {
                            expired.push((p.key, p.ctx));
                        }
                    }
                }
                expired
            };
            for (key, ctx) in info {
                self.waits.cancel(key, ctx, WakeReason::Timeout);
            }
        }
        #[cfg(target_arch = "wasm32")]
        let _ = handle;
    }

    pub(crate) fn fiber_park_slot(
        &self,
        handle: i64,
    ) -> Option<(WaitKey, Arc<WaitSlot>, Option<Deadline>, ContextId)> {
        let target = temen_kernel::unpack_fiber_handle(handle);
        let fibers = self.fibers.lock().unwrap_or_else(|e| e.into_inner());
        for rec in fibers.values() {
            if temen_kernel::unpack_fiber_handle(rec.handle) == target {
                if let Some(ref p) = rec.parked {
                    return Some((p.key, Arc::clone(&p.slot), p.deadline, p.ctx));
                }
            }
        }
        None
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn join_all_os_threads(&self) {
        let handles: Vec<_> = {
            let mut g = self.os_threads.lock().unwrap_or_else(|e| e.into_inner());
            std::mem::take(&mut *g)
        };
        for h in handles {
            let _ = h.join();
        }
    }
}

/// Active nested fiber frame pushed onto a Carrier when `cont.resume` runs a `GuestFiber` inline.
struct ActiveFiber {
    handle: i64,
    ctx: ContextId,
    resumer_ctx: ContextId,
    resumer_kind: ContextKind,
    resumer_exec: ExecState,
    block: bool,
}

/// Action returned by [`step_carrier_once`] to tell either the Cooperative or Parallel loop
/// whether the context finished, yielded, or needs to block on a [`WaitSlot`].
enum CarrierPause {
    Finished(ContextExit),
    Yielded,
    BlockOn {
        key: WaitKey,
        slot: Arc<WaitSlot>,
        deadline: Option<Deadline>,
        kind: ParkKind,
    },
    SpawnCoop {
        ctx: ContextId,
        slot: u16,
        exec: ExecState,
        tls: i64,
    },
}

#[derive(Clone, Copy)]
enum ParkKind {
    Futex,
    Reexec,
}

/// Per-Carrier execution bundle (owned exclusively by one Cooperative task or one Parallel OS
/// thread).
struct CarrierTask {
    ctx: ContextId,
    kind: ContextKind,
    exec: ExecState,
    fiber_stack: Vec<ActiveFiber>,
    tls: i64,
    thread_slot: Option<u16>,
}

impl CarrierTask {
    fn new_root(exec: ExecState) -> CarrierTask {
        CarrierTask {
            ctx: ContextId(0),
            kind: ContextKind::Root,
            exec,
            fiber_stack: Vec::new(),
            tls: 0,
            thread_slot: None,
        }
    }

    fn new_thread(ctx: ContextId, slot: u16, exec: ExecState, tls: i64) -> CarrierTask {
        CarrierTask {
            ctx,
            kind: ContextKind::Thread,
            exec,
            fiber_stack: Vec::new(),
            tls,
            thread_slot: Some(slot),
        }
    }
}

pub(crate) fn wake_to_futex_status(reason: WakeReason) -> Result<i32, Trap> {
    match reason {
        WakeReason::Ready => Ok(WAIT_OK),
        WakeReason::Timeout => Ok(WAIT_TIMED_OUT),
        WakeReason::Interrupted => Ok(WAIT_OK),
        WakeReason::Revoked | WakeReason::Freeze | WakeReason::Paused => Err(Trap::OutOfFuel),
    }
}

/// Advance `task` on `carrier_id` using ONE shared state-transition loop for both Cooperative and
/// Parallel modes.
fn step_carrier_once(
    world: &Arc<DomainWorld>,
    carrier_id: CarrierId,
    task: &mut CarrierTask,
    fuel: &mut FuelCell,
    parallel: bool,
) -> CarrierPause {
    let code = world.code();
    let mem = world.mem_view();
    let cfg = StepConfig {
        quantum: if parallel { u32::MAX } else { world.quantum },
        max_call_depth: temen_bytecode::vm::DEFAULT_MAX_CALL_DEPTH,
    };

    loop {
        let s = bc_step(
            code,
            &mut task.exec,
            &mem,
            fuel,
            world.domain.poll_word(),
            &mut task.tls,
            cfg,
        );

        match s {
            Step::Finished(exit) => {
                if let Some(active) = task.fiber_stack.pop() {
                    // A running GuestFiber finished!
                    if exit.is_domain_terminal() {
                        world.ctl.end(exit.clone());
                        return CarrierPause::Finished(exit);
                    }
                    let ret_val = match exit {
                        ContextExit::Returned(ref v) => v.first().copied().unwrap_or(0),
                        _ => 0,
                    };
                    world.domain.finish_fiber(active.handle);
                    world
                        .fibers
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .remove(&active.ctx);
                    task.ctx = active.resumer_ctx;
                    task.kind = active.resumer_kind;
                    task.exec = active.resumer_exec;
                    task.exec.deliver_fiber_resume(FIBER_RETURNED, ret_val);
                    continue;
                }
                return CarrierPause::Finished(exit);
            }

            Step::Yield(YieldReason::Quantum) => {
                return CarrierPause::Yielded;
            }

            Step::Yield(YieldReason::FuelRefill) => {
                if world.draw_fuel(fuel) {
                    continue;
                }
                let exit = ContextExit::Trapped(Trap::OutOfFuel);
                world.ctl.end(exit.clone());
                return CarrierPause::Finished(exit);
            }

            Step::Yield(YieldReason::Poll(flags)) => match world.domain.evaluate_poll(flags) {
                PollAction::Continue => continue,
                PollAction::Terminate(trap) => {
                    return CarrierPause::Finished(ContextExit::Trapped(trap));
                }
                PollAction::Pause => {
                    return CarrierPause::Yielded;
                }
                PollAction::Freeze => {
                    return CarrierPause::Finished(ContextExit::Trapped(Trap::OutOfFuel));
                }
                PollAction::Upcall { .. } => {
                    world.domain.hooks().upcall_finished(world.domain.id);
                    continue;
                }
            },

            Step::FutexNotify { addr, count } => {
                let key = WaitKey::Futex {
                    space: world.mem_space,
                    offset: addr,
                };
                let woken = world.waits.notify(key, count, WakeReason::Ready);
                task.exec.deliver_i32(woken as i32);
            }

            Step::FutexWait {
                addr,
                expected,
                width64,
                timeout_ns,
            } => {
                let key = WaitKey::Futex {
                    space: world.mem_space,
                    offset: addr,
                };
                if timeout_ns == 0 {
                    let eq = world.check_futex_eq(addr, expected, width64);
                    task.exec
                        .deliver_i32(if eq { WAIT_TIMED_OUT } else { WAIT_NOT_EQUAL });
                    continue;
                }
                #[cfg(not(target_arch = "wasm32"))]
                let deadline = (timeout_ns > 0)
                    .then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64));
                #[cfg(target_arch = "wasm32")]
                let deadline = (timeout_ns > 0).then_some(timeout_ns as u64);

                let spec = WaiterSpec {
                    domain: world.domain.id,
                    ctx: task.ctx,
                    kind: task.kind,
                    interruptible: true,
                };
                match world
                    .waits
                    .register_futex(key, spec, || world.check_futex_eq(addr, expected, width64))
                {
                    RegisteredPark::Immediate(_) => {
                        if !world.domain.poll_flags().is_empty() {
                            continue; // poll safepoint will handle ENDED / PAUSED
                        }
                        task.exec.deliver_i32(WAIT_NOT_EQUAL);
                    }
                    RegisteredPark::Parked { slot, decision } => {
                        if decision == ParkDecision::YieldFiber {
                            // Red-team refinement #3: GuestFiber parks non-blockingly!
                            let active = task.fiber_stack.pop().expect("GuestFiber has resumer");
                            let fiber_exec = std::mem::replace(&mut task.exec, active.resumer_exec);
                            {
                                let mut fibers =
                                    world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(rec) = fibers.get_mut(&active.ctx) {
                                    rec.exec = Some(fiber_exec);
                                    rec.parked = Some(FiberParkInfo {
                                        key,
                                        slot: Arc::clone(&slot),
                                        deadline,
                                        ctx: active.ctx,
                                        reexec_on_wake: false,
                                    });
                                }
                            }
                            let parked = world.domain.mark_fiber_parked(active.handle, key);
                            task.ctx = active.resumer_ctx;
                            task.kind = active.resumer_kind;
                            if !parked {
                                // Concurrent wake already promoted the fiber to WakeReady!
                                task.exec.rewind_for_reexec();
                                continue;
                            }
                            if active.block {
                                task.exec.rewind_for_reexec();
                                return CarrierPause::BlockOn {
                                    key,
                                    slot,
                                    deadline,
                                    kind: ParkKind::Reexec,
                                };
                            }
                            task.exec.deliver_fiber_resume(FIBER_PARKED, 0);
                            continue;
                        }
                        return CarrierPause::BlockOn {
                            key,
                            slot,
                            deadline,
                            kind: ParkKind::Futex,
                        };
                    }
                }
            }

            Step::FiberNew { func_idx, arg0 } => {
                // Mask func_idx into the function table (§3c); signature is checked at first resume
                // (or right here if out of range / wrong type -> FiberFault on resume).
                let slot_idx = func_idx & code.fn_table_mask;
                let fiber_exec = ExecState::new_fiber_entry(code, slot_idx, arg0);
                let (ctx, handle) = world.domain.create_fiber(None);
                world
                    .fibers
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .insert(
                        ctx,
                        FiberRecord {
                            handle,
                            exec: fiber_exec.ok(),
                            #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
                            kfiber: None,
                            started: false,
                            parked: None,
                        },
                    );
                task.exec.deliver_results(&[handle]);
            }

            Step::FiberResume { handle, val, block } => {
                world.expire_fiber_if_timed_out(handle);
                match world.domain.claim_fiber_resume(handle, carrier_id) {
                    FiberResumeClaim::Claimed { ctx, wake } => {
                        let (mut fiber_exec, started, reexec_on_wake) = {
                            let mut fibers = world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                            let Some(rec) = fibers.get_mut(&ctx) else {
                                let exit = ContextExit::Trapped(Trap::FiberFault);
                                world.ctl.end(exit.clone());
                                return CarrierPause::Finished(exit);
                            };
                            let reexec = rec.parked.take().is_some_and(|p| p.reexec_on_wake);
                            let Some(exec) = rec.exec.take() else {
                                // Invalid funcref / wrong entry signature checked at first resume.
                                let exit = ContextExit::Trapped(Trap::IndirectCallType);
                                world.ctl.end(exit.clone());
                                return CarrierPause::Finished(exit);
                            };
                            let was_started = rec.started;
                            rec.started = true;
                            (exec, was_started, reexec)
                        };
                        if !started {
                            fiber_exec.set_entry_arg1(val);
                        } else if let Some(reason) = wake {
                            if !reexec_on_wake {
                                match wake_to_futex_status(reason) {
                                    Ok(code) => fiber_exec.deliver_i32(code),
                                    Err(t) => {
                                        let exit = ContextExit::Trapped(t);
                                        world.ctl.end(exit.clone());
                                        return CarrierPause::Finished(exit);
                                    }
                                }
                            } else if matches!(reason, WakeReason::Revoked) {
                                let exit = world
                                    .domain
                                    .terminal_exit()
                                    .unwrap_or(ContextExit::Trapped(Trap::OutOfFuel));
                                return CarrierPause::Finished(exit);
                            }
                        } else {
                            fiber_exec.deliver_results(&[val]);
                        }
                        let resumer_exec = std::mem::replace(&mut task.exec, fiber_exec);
                        task.fiber_stack.push(ActiveFiber {
                            handle,
                            ctx,
                            resumer_ctx: task.ctx,
                            resumer_kind: task.kind,
                            resumer_exec,
                            block,
                        });
                        task.ctx = ctx;
                        task.kind = ContextKind::GuestFiber;
                    }
                    FiberResumeClaim::StillParked => {
                        if block {
                            if let Some((key, slot, deadline, _)) = world.fiber_park_slot(handle) {
                                task.exec.rewind_for_reexec();
                                return CarrierPause::BlockOn {
                                    key,
                                    slot,
                                    deadline,
                                    kind: ParkKind::Reexec,
                                };
                            }
                        }
                        task.exec.deliver_fiber_resume(FIBER_PARKED, 0);
                    }
                    FiberResumeClaim::Spent => {
                        task.exec.deliver_fiber_resume(ERRNO_EBADF, 0);
                    }
                    FiberResumeClaim::Fault => {
                        let exit = ContextExit::Trapped(Trap::FiberFault);
                        world.ctl.end(exit.clone());
                        return CarrierPause::Finished(exit);
                    }
                }
            }

            Step::FiberSuspend { val } => {
                let Some(active) = task.fiber_stack.pop() else {
                    // Suspending from Root or Thread traps FiberFault.
                    let exit = ContextExit::Trapped(Trap::FiberFault);
                    world.ctl.end(exit.clone());
                    return CarrierPause::Finished(exit);
                };
                let fiber_exec = std::mem::replace(&mut task.exec, active.resumer_exec);
                {
                    let mut fibers = world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(rec) = fibers.get_mut(&active.ctx) {
                        rec.exec = Some(fiber_exec);
                    }
                }
                world.domain.suspend_fiber(active.handle);
                task.ctx = active.resumer_ctx;
                task.kind = active.resumer_kind;
                task.exec.deliver_fiber_resume(FIBER_SUSPENDED, val);
            }

            Step::ThreadSpawn {
                func_idx,
                arg0,
                arg1,
            } => {
                let Some(f) = code.funcs.get(func_idx as usize) else {
                    let exit = ContextExit::Trapped(Trap::ThreadFault);
                    world.ctl.end(exit.clone());
                    return CarrierPause::Finished(exit);
                };
                if f.type_id != TypeId::fiber_entry() {
                    let exit = ContextExit::Trapped(Trap::IndirectCallType);
                    world.ctl.end(exit.clone());
                    return CarrierPause::Finished(exit);
                }
                if !world.budgets.charge_tasks(world.domain.budget, 1) {
                    let exit = ContextExit::Trapped(Trap::ThreadFault);
                    world.ctl.end(exit.clone());
                    return CarrierPause::Finished(exit);
                }
                let exec = match ExecState::new_entry(code, func_idx, &[arg0, arg1]) {
                    Ok(e) => e,
                    Err(t) => {
                        world.budgets.refund_tasks(world.domain.budget, 1);
                        let exit = ContextExit::Trapped(t);
                        world.ctl.end(exit.clone());
                        return CarrierPause::Finished(exit);
                    }
                };
                let ctx = world.domain.register_context(ContextKind::Thread, None);
                let waitable = world.alloc_waitable();
                let (handle, slot) = world
                    .threads
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .alloc(waitable);
                let dense_tls = world.next_carrier.fetch_add(1, Ordering::Relaxed) as i64;
                task.exec.deliver_i32(handle);

                #[cfg(not(target_arch = "wasm32"))]
                if parallel {
                    let w = Arc::clone(world);
                    let cid = CarrierId(dense_tls as u32);
                    let jh = std::thread::spawn(move || {
                        let child_task = CarrierTask::new_thread(ctx, slot, exec, dense_tls);
                        drive_parallel_task(&w, cid, child_task);
                    });
                    world
                        .os_threads
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .push(jh);
                    continue;
                }

                return CarrierPause::SpawnCoop {
                    ctx,
                    slot,
                    exec,
                    tls: dense_tls,
                };
            }

            Step::ThreadJoin { handle } => match world.lookup_thread_join(handle) {
                ThreadJoinLookup::Finished(exit) => match exit {
                    ContextExit::Returned(v) => {
                        task.exec
                            .deliver_results(&[0, v.first().copied().unwrap_or(0)]);
                    }
                    other => {
                        world.ctl.end(other.clone());
                        return CarrierPause::Finished(other);
                    }
                },
                ThreadJoinLookup::Spent => {
                    task.exec.deliver_results(&[i64::from(ERRNO_EBADF), 0]);
                }
                ThreadJoinLookup::Fault => {
                    let exit = ContextExit::Trapped(Trap::ThreadFault);
                    world.ctl.end(exit.clone());
                    return CarrierPause::Finished(exit);
                }
                ThreadJoinLookup::Running(waitable) => {
                    let key = WaitKey::Object(waitable);
                    let spec = WaiterSpec {
                        domain: world.domain.id,
                        ctx: task.ctx,
                        kind: task.kind,
                        interruptible: false,
                    };
                    let still_waiting = || {
                        if !world.domain.poll_flags().is_empty() {
                            return false;
                        }
                        let (slot, gen) = unpack_thread_handle(handle);
                        let tbl = world.threads.lock().unwrap_or_else(|e| e.into_inner());
                        tbl.slots
                            .get(slot as usize)
                            .is_some_and(|r| r.gen == gen && r.exit.is_none())
                    };
                    match world.waits.register_object(key, spec, still_waiting) {
                        RegisteredPark::Immediate(_) => {
                            task.exec.rewind_for_reexec();
                        }
                        RegisteredPark::Parked { slot, decision } => {
                            if decision == ParkDecision::YieldFiber {
                                let active =
                                    task.fiber_stack.pop().expect("GuestFiber has resumer");
                                let fiber_exec =
                                    std::mem::replace(&mut task.exec, active.resumer_exec);
                                {
                                    let mut fibers =
                                        world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                                    if let Some(rec) = fibers.get_mut(&active.ctx) {
                                        let mut fe = fiber_exec;
                                        fe.rewind_for_reexec();
                                        rec.exec = Some(fe);
                                        rec.parked = Some(FiberParkInfo {
                                            key,
                                            slot: Arc::clone(&slot),
                                            deadline: None,
                                            ctx: active.ctx,
                                            reexec_on_wake: true,
                                        });
                                    }
                                }
                                let parked = world.domain.mark_fiber_parked(active.handle, key);
                                task.ctx = active.resumer_ctx;
                                task.kind = active.resumer_kind;
                                if !parked {
                                    task.exec.rewind_for_reexec();
                                    continue;
                                }
                                task.exec.deliver_fiber_resume(FIBER_PARKED, 0);
                                continue;
                            }
                            task.exec.rewind_for_reexec();
                            return CarrierPause::BlockOn {
                                key,
                                slot,
                                deadline: None,
                                kind: ParkKind::Reexec,
                            };
                        }
                    }
                }
            },

            Step::CallCap {
                handle,
                type_id,
                op,
                args,
                nargs,
                nresults: _,
            } => {
                match world.caps.call(
                    world.domain.id,
                    task.ctx,
                    handle,
                    type_id,
                    op,
                    &args[..nargs as usize],
                    world.region.as_deref(),
                ) {
                    CapOutcome::Ready(vals) => task.exec.deliver_results(&vals),
                    CapOutcome::Trap(t) => {
                        let exit = ContextExit::Trapped(t);
                        world.ctl.end(exit.clone());
                        return CarrierPause::Finished(exit);
                    }
                    CapOutcome::Park {
                        key,
                        deadline: dl_ns,
                        interruptible,
                    } => {
                        let spec = WaiterSpec {
                            domain: world.domain.id,
                            ctx: task.ctx,
                            kind: task.kind,
                            interruptible,
                        };
                        #[cfg(not(target_arch = "wasm32"))]
                        let deadline = dl_ns.map(|ns| Instant::now() + Duration::from_nanos(ns));
                        #[cfg(target_arch = "wasm32")]
                        let deadline = dl_ns;

                        if let RegisteredPark::Parked { slot, .. } =
                            world.waits.register_object(key, spec, || true)
                        {
                            task.exec.rewind_for_reexec();
                            return CarrierPause::BlockOn {
                                key,
                                slot,
                                deadline,
                                kind: ParkKind::Reexec,
                            };
                        }
                        task.exec.rewind_for_reexec();
                    }
                }
            }

            Step::Park {
                key,
                deadline: dl_ns,
                interruptible,
            } => {
                let spec = WaiterSpec {
                    domain: world.domain.id,
                    ctx: task.ctx,
                    kind: task.kind,
                    interruptible,
                };
                #[cfg(not(target_arch = "wasm32"))]
                let deadline = dl_ns.map(|ns| Instant::now() + Duration::from_nanos(ns));
                #[cfg(target_arch = "wasm32")]
                let deadline = dl_ns;
                if let RegisteredPark::Parked { slot, .. } =
                    world.waits.register_object(key, spec, || true)
                {
                    return CarrierPause::BlockOn {
                        key,
                        slot,
                        deadline,
                        kind: ParkKind::Reexec,
                    };
                }
            }
        }
    }
}

/// Parallel Carrier OS-thread driver (runs `Root` on the caller's thread and each spawned `Thread`
/// on its own OS thread, sharing ONE `DomainWorld`).
#[cfg(not(target_arch = "wasm32"))]
fn drive_parallel_task(
    world: &Arc<DomainWorld>,
    carrier_id: CarrierId,
    mut task: CarrierTask,
) -> ContextExit {
    let mut fuel = world.make_fuel_cell();
    let exit = loop {
        match step_carrier_once(world, carrier_id, &mut task, &mut fuel, true) {
            CarrierPause::Finished(exit) => break exit,
            CarrierPause::Yielded => {
                world.refund_fuel(&mut fuel);
                while world
                    .domain
                    .poll_flags()
                    .contains(temen_kernel::PollFlags::PAUSED)
                    && !world
                        .domain
                        .poll_flags()
                        .contains(temen_kernel::PollFlags::ENDED)
                {
                    std::thread::yield_now();
                }
                if world
                    .domain
                    .poll_flags()
                    .contains(temen_kernel::PollFlags::ENDED)
                {
                    let t = world
                        .domain
                        .terminal_exit()
                        .unwrap_or(ContextExit::Trapped(Trap::OutOfFuel));
                    break t;
                }
                let _ = world.draw_fuel(&mut fuel);
            }
            CarrierPause::BlockOn {
                key,
                slot,
                deadline,
                kind,
            } => {
                // Refund unburned Carrier fuel while blocked so sibling Carriers aren't starved.
                world.refund_fuel(&mut fuel);
                let reason = match deadline {
                    Some(dl) => {
                        let now = Instant::now();
                        let rem = dl.saturating_duration_since(now);
                        match slot.wait_timeout_blocking(rem) {
                            Some(r) => r,
                            None => {
                                if world.waits.cancel(key, task.ctx, WakeReason::Timeout) {
                                    WakeReason::Timeout
                                } else {
                                    slot.poll().unwrap_or(WakeReason::Timeout)
                                }
                            }
                        }
                    }
                    None => slot.wait_blocking(),
                };
                if !world.draw_fuel(&mut fuel) && world.metered {
                    let exit = ContextExit::Trapped(Trap::OutOfFuel);
                    world.ctl.end(exit.clone());
                    break exit;
                }
                match kind {
                    ParkKind::Futex => match wake_to_futex_status(reason) {
                        Ok(st) => task.exec.deliver_i32(st),
                        Err(t) => {
                            let exit = world
                                .domain
                                .terminal_exit()
                                .unwrap_or(ContextExit::Trapped(t));
                            break exit;
                        }
                    },
                    ParkKind::Reexec => {
                        // Already rewound to re-execute cleanly on wake.
                        if matches!(reason, WakeReason::Revoked) {
                            let exit = world
                                .domain
                                .terminal_exit()
                                .unwrap_or(ContextExit::Trapped(Trap::OutOfFuel));
                            break exit;
                        }
                    }
                }
            }
            CarrierPause::SpawnCoop { .. } => unreachable!("parallel carrier spawns OS threads"),
        }
    };
    world.refund_fuel(&mut fuel);
    if let Some(slot) = task.thread_slot {
        world.finish_thread(slot, exit.clone());
    }
    exit
}

/// Cooperative single-Carrier round-robin driver (native + wasm32/wasm64 clean, with `#2967`
/// last-sleeper deadlock settle).
fn drive_cooperative(world: &Arc<DomainWorld>, root_exec: ExecState) -> ContextExit {
    struct ParkedCoop {
        task: CarrierTask,
        key: WaitKey,
        slot: Arc<WaitSlot>,
        deadline: Option<Deadline>,
        kind: ParkKind,
    }

    let mut runnable: VecDeque<CarrierTask> = VecDeque::new();
    let mut parked: Vec<ParkedCoop> = Vec::new();
    runnable.push_back(CarrierTask::new_root(root_exec));

    let mut fuel = world.make_fuel_cell();
    let carrier_id = CarrierId(0);

    let root_exit = loop {
        // 1. Poll any parked cooperative tasks whose WaitSlot became ready or deadline elapsed.
        let mut i = 0;
        while i < parked.len() {
            let mut ready_reason = parked[i].slot.poll();
            #[cfg(not(target_arch = "wasm32"))]
            if ready_reason.is_none() {
                if let Some(dl) = parked[i].deadline {
                    if Instant::now() >= dl {
                        let p = &parked[i];
                        if world.waits.cancel(p.key, p.task.ctx, WakeReason::Timeout) {
                            ready_reason = Some(WakeReason::Timeout);
                        } else {
                            ready_reason = p.slot.poll().or(Some(WakeReason::Timeout));
                        }
                    }
                }
            }
            if let Some(reason) = ready_reason {
                let mut p = parked.remove(i);
                match p.kind {
                    ParkKind::Futex => match wake_to_futex_status(reason) {
                        Ok(st) => {
                            p.task.exec.deliver_i32(st);
                            runnable.push_back(p.task);
                        }
                        Err(t) => {
                            let exit = world
                                .domain
                                .terminal_exit()
                                .unwrap_or(ContextExit::Trapped(t));
                            world.ctl.end(exit.clone());
                            if p.task.kind == ContextKind::Root {
                                break;
                            }
                        }
                    },
                    ParkKind::Reexec => {
                        runnable.push_back(p.task);
                    }
                }
            } else {
                i += 1;
            }
        }

        if let Some(exit) = world.domain.terminal_exit() {
            break exit;
        }

        // 2. If no task is runnable, either sleep until earliest deadline or settle deadlock (#2967).
        let Some(mut task) = runnable.pop_front() else {
            if parked.is_empty() {
                break ContextExit::Trapped(Trap::ThreadFault);
            }
            #[cfg(not(target_arch = "wasm32"))]
            {
                let earliest = parked.iter().filter_map(|p| p.deadline).min();
                if let Some(dl) = earliest {
                    let now = Instant::now();
                    if dl > now {
                        std::thread::sleep((dl - now).min(Duration::from_millis(5)));
                    }
                    continue;
                }
            }
            // #2967: every live context is parked with no external bell and no finite timeout ->
            // deterministic deadlock settle (`Trap::ThreadFault`).
            let exit = ContextExit::Trapped(Trap::ThreadFault);
            world.ctl.end(exit.clone());
            break exit;
        };

        // 3. Step the selected task for one cooperative quantum.
        match step_carrier_once(world, carrier_id, &mut task, &mut fuel, false) {
            CarrierPause::Finished(exit) => {
                if let Some(slot) = task.thread_slot {
                    world.finish_thread(slot, exit);
                } else {
                    // Root finished -> terminate domain (Invariant 6).
                    world.ctl.end(exit.clone());
                    break exit;
                }
            }
            CarrierPause::Yielded => {
                runnable.push_back(task);
            }
            CarrierPause::SpawnCoop {
                ctx,
                slot,
                exec,
                tls,
            } => {
                runnable.push_back(task);
                runnable.push_back(CarrierTask::new_thread(ctx, slot, exec, tls));
            }
            CarrierPause::BlockOn {
                key,
                slot,
                deadline,
                kind,
            } => {
                parked.push(ParkedCoop {
                    task,
                    key,
                    slot,
                    deadline,
                    kind,
                });
            }
        }
    };

    world.refund_fuel(&mut fuel);
    root_exit
}

/// Top-level entry point called by [`crate::run`].
pub fn run_entry(engine: Engine, entry_func: u32, args: &[i64], cfg: RunConfig) -> RunOutput {
    let mode = cfg.mode;
    let world = match DomainWorld::new(engine, cfg) {
        Ok(w) => w,
        Err(t) => {
            return RunOutput {
                exit: ContextExit::Trapped(t),
                fuel_used: 0,
                memory: None,
            };
        }
    };

    let exit = match &world.engine {
        Engine::Bytecode(_) => {
            let root_exec = match ExecState::new_entry(world.code(), entry_func, args) {
                Ok(e) => e,
                Err(t) => {
                    return RunOutput {
                        exit: ContextExit::Trapped(t),
                        fuel_used: 0,
                        memory: world.region.clone(),
                    };
                }
            };
            match mode {
                RunMode::Cooperative => drive_cooperative(&world, root_exec),
                #[cfg(not(target_arch = "wasm32"))]
                RunMode::Parallel => {
                    let root_exit =
                        drive_parallel_task(&world, CarrierId(0), CarrierTask::new_root(root_exec));
                    // Invariant 6: root completion ends the domain and joins all spawned sibling OS
                    // threads before returning, so zero background threads outlive the run.
                    world.ctl.end(root_exit.clone());
                    world.join_all_os_threads();
                    world.domain.terminal_exit().unwrap_or(root_exit)
                }
            }
        }
        #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
        Engine::Cranelift(_) => crate::cranelift::run_cranelift_entry(&world, entry_func, args),
    };

    let fuel_used = world
        .budgets
        .snapshot()
        .first()
        .map(|s| s.used_fuel)
        .unwrap_or(0);

    RunOutput {
        exit,
        fuel_used,
        memory: world.region.clone(),
    }
}
