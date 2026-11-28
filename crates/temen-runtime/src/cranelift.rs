//! Stage 1d (`#3197`): `Engine::Cranelift` bridge over the shared [`DomainWorld`].
//!
//! 100% `#![forbid(unsafe_code)]` (all hardware-guard, FFI, and `temen-fiber` stack-switching
//! unsafe lives inside `temen_jit::kernel_engine`). Shares ONE `Domain`, `WaitIndex`, `BudgetTree`,
//! `DomainControl`, `ThreadTable`, `FiberRecord`, and `CapHandler` with `Engine::Bytecode`
//! (Invariant 15).

use std::cell::RefCell;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, Instant};

use temen_jit::kernel_engine::{
    park_current_fiber, suspend_current_fiber, CompiledKernelModule, KernelBridge, KernelFiber,
    KernelFiberStep, KernelMemory,
};
use temen_kernel::wait::RegisteredPark;
use temen_kernel::{
    CapOutcome, CarrierId, ContextExit, ContextId, ContextKind, FiberResumeClaim, FuelCell,
    ParkDecision, PollFlags, Trap, TypeId, WaitKey, WaitSlot, WaiterSpec, WakeReason, ERRNO_EBADF,
    FIBER_PARKED, FIBER_RETURNED, FIBER_SUSPENDED, WAIT_NOT_EQUAL, WAIT_TIMED_OUT,
};

use crate::driver::{
    wake_to_futex_status, Deadline, DomainWorld, FiberParkInfo, FiberRecord, ThreadJoinLookup,
};
use crate::Engine;

struct CarrierFrame {
    carrier_id: CarrierId,
    /// Active context stack on this Carrier: `[Root/Thread, Fiber1, ...]`.
    ctx_stack: Vec<(ContextId, ContextKind)>,
}

std::thread_local! {
    static CARRIER: RefCell<Option<CarrierFrame>> = const { RefCell::new(None) };
}

fn current_carrier_id() -> CarrierId {
    CARRIER.with(|c| {
        c.borrow()
            .as_ref()
            .map(|f| f.carrier_id)
            .unwrap_or(CarrierId(0))
    })
}

fn current_ctx_and_kind() -> (ContextId, ContextKind) {
    CARRIER.with(|c| {
        c.borrow()
            .as_ref()
            .and_then(|f| f.ctx_stack.last().copied())
            .unwrap_or((ContextId(0), ContextKind::Root))
    })
}

fn push_fiber_ctx(ctx: ContextId) {
    CARRIER.with(|c| {
        if let Some(ref mut f) = *c.borrow_mut() {
            f.ctx_stack.push((ctx, ContextKind::GuestFiber));
        }
    });
}

fn pop_fiber_ctx() {
    CARRIER.with(|c| {
        if let Some(ref mut f) = *c.borrow_mut() {
            let _ = f.ctx_stack.pop();
        }
    });
}

struct FiberCtxGuard;

impl Drop for FiberCtxGuard {
    fn drop(&mut self) {
        pop_fiber_ctx();
    }
}

struct CarrierGuard;

impl Drop for CarrierGuard {
    fn drop(&mut self) {
        CARRIER.with(|c| {
            *c.borrow_mut() = None;
        });
    }
}

fn install_carrier(carrier_id: CarrierId, ctx: ContextId, kind: ContextKind) -> CarrierGuard {
    CARRIER.with(|c| {
        *c.borrow_mut() = Some(CarrierFrame {
            carrier_id,
            ctx_stack: vec![(ctx, kind)],
        });
    });
    CarrierGuard
}

/// Bridge from `temen_jit::kernel_engine` thunks into the shared [`DomainWorld`].
struct CraneliftBridge {
    world: Arc<DomainWorld>,
}

impl CraneliftBridge {
    #[inline]
    fn compiled(&self) -> &Arc<CompiledKernelModule> {
        match self.world.engine {
            Engine::Cranelift(ref c) => c,
            Engine::Bytecode(_) => unreachable!("CraneliftBridge on Engine::Bytecode"),
        }
    }

    #[inline]
    fn kernel_mem(&self) -> &Arc<KernelMemory> {
        self.world
            .kernel_mem
            .as_ref()
            .expect("Engine::Cranelift always initializes KernelMemory")
    }

    fn terminal_trap(&self, fallback: Trap) -> Trap {
        match self.world.domain.terminal_exit() {
            Some(ContextExit::Trapped(t)) => t,
            _ => fallback,
        }
    }

    /// Block the calling Carrier OS thread on `slot` until woken or `deadline` elapses, refunding
    /// unburned Carrier fuel while blocked (Decision 30).
    fn block_carrier_on_slot(
        &self,
        key: WaitKey,
        ctx: ContextId,
        slot: &Arc<WaitSlot>,
        deadline: Option<Deadline>,
        fuel: &mut FuelCell,
    ) -> Result<WakeReason, Trap> {
        self.world.refund_fuel(fuel);
        let reason = match deadline {
            Some(dl) => {
                let rem = dl.saturating_duration_since(Instant::now());
                match slot.wait_timeout_blocking(rem) {
                    Some(r) => r,
                    None => {
                        if self.world.waits.cancel(key, ctx, WakeReason::Timeout) {
                            WakeReason::Timeout
                        } else {
                            slot.poll().unwrap_or(WakeReason::Timeout)
                        }
                    }
                }
            }
            None => slot.wait_blocking(),
        };
        if !self.world.draw_fuel(fuel) && self.world.metered {
            let exit = ContextExit::Trapped(Trap::OutOfFuel);
            self.world.ctl.end(exit);
            return Err(Trap::OutOfFuel);
        }
        Ok(reason)
    }
}

impl KernelBridge for CraneliftBridge {
    fn on_poll(&self, flags: u64, fuel: &mut FuelCell) -> Result<(), Trap> {
        let pf = PollFlags(flags);
        if pf.contains(PollFlags::ENDED) {
            return Err(self.terminal_trap(Trap::OutOfFuel));
        }
        if pf.contains(PollFlags::PAUSED) {
            self.world.refund_fuel(fuel);
            while self.world.domain.poll_flags().contains(PollFlags::PAUSED)
                && !self.world.domain.poll_flags().contains(PollFlags::ENDED)
            {
                std::thread::yield_now();
            }
            if self.world.domain.poll_flags().contains(PollFlags::ENDED) {
                return Err(self.terminal_trap(Trap::OutOfFuel));
            }
            if !self.world.draw_fuel(fuel) && self.world.metered {
                self.world.ctl.end(ContextExit::Trapped(Trap::OutOfFuel));
                return Err(Trap::OutOfFuel);
            }
        }
        Ok(())
    }

    fn on_fuel_refill(&self, fuel: &mut FuelCell) -> Result<(), Trap> {
        if !self.world.metered {
            fuel.remaining = u64::MAX;
            return Ok(());
        }
        if !self.world.draw_fuel(fuel) {
            self.world.ctl.end(ContextExit::Trapped(Trap::OutOfFuel));
            return Err(Trap::OutOfFuel);
        }
        // Charge the 1 safepoint unit that hit `remaining == 0`.
        fuel.remaining = fuel.remaining.saturating_sub(1);
        Ok(())
    }

    fn futex_wait(
        &self,
        addr: u64,
        expected: i64,
        width64: bool,
        timeout_ns: i64,
        fuel: &mut FuelCell,
    ) -> Result<i32, Trap> {
        // Validate alignment, NULL guard, and window bounds via hardware-backed `KernelMemory`.
        if self.kernel_mem().atomic_load(addr, width64).is_none() {
            return Err(Trap::MemoryFault);
        }
        if timeout_ns == 0 {
            let eq = self.world.check_futex_eq(addr, expected, width64);
            return Ok(if eq { WAIT_TIMED_OUT } else { WAIT_NOT_EQUAL });
        }
        let deadline =
            (timeout_ns > 0).then(|| Instant::now() + Duration::from_nanos(timeout_ns as u64));
        let key = WaitKey::Futex {
            space: self.world.mem_space,
            offset: addr,
        };
        let (ctx, kind) = current_ctx_and_kind();
        let spec = WaiterSpec {
            domain: self.world.domain.id,
            ctx,
            kind,
            interruptible: true,
        };
        match self.world.waits.register_futex(key, spec, || {
            self.world.check_futex_eq(addr, expected, width64)
        }) {
            RegisteredPark::Immediate(_) => {
                if self.world.domain.poll_flags().contains(PollFlags::ENDED) {
                    return Err(self.terminal_trap(Trap::OutOfFuel));
                }
                Ok(WAIT_NOT_EQUAL)
            }
            RegisteredPark::Parked { slot, decision } => {
                if decision == ParkDecision::YieldFiber {
                    // Record park metadata BEFORE yielding; `fiber_resume` stores the suspended
                    // `KernelFiber` back into `world.fibers` FIRST and calls `mark_fiber_parked`
                    // SECOND so no concurrent Carrier can ever claim a still-running stack.
                    {
                        let mut fibers =
                            self.world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                        if let Some(rec) = fibers.get_mut(&ctx) {
                            rec.parked = Some(FiberParkInfo {
                                key,
                                slot,
                                deadline,
                                ctx,
                                reexec_on_wake: false,
                            });
                        }
                    }
                    let wake_val = park_current_fiber().ok_or(Trap::FiberFault)?;
                    if self.world.domain.poll_flags().contains(PollFlags::ENDED) {
                        return Err(self.terminal_trap(Trap::OutOfFuel));
                    }
                    Ok(wake_val as i32)
                } else {
                    let reason = self.block_carrier_on_slot(key, ctx, &slot, deadline, fuel)?;
                    wake_to_futex_status(reason).map_err(|t| self.terminal_trap(t))
                }
            }
        }
    }

    fn futex_notify(&self, addr: u64, count: u32) -> Result<i32, Trap> {
        if self.kernel_mem().atomic_load(addr, false).is_none() {
            return Err(Trap::MemoryFault);
        }
        let key = WaitKey::Futex {
            space: self.world.mem_space,
            offset: addr,
        };
        let woken = self.world.waits.notify(key, count, WakeReason::Ready);
        Ok(woken as i32)
    }

    fn thread_spawn(&self, func_idx: u32, arg0: i64, arg1: i64) -> Result<i32, Trap> {
        let Some(tid) = self.compiled().func_type_id(func_idx) else {
            return Err(Trap::ThreadFault);
        };
        if tid != TypeId::fiber_entry() {
            return Err(Trap::IndirectCallType);
        }
        if !self.world.budgets.charge_tasks(self.world.domain.budget, 1) {
            return Err(Trap::ThreadFault);
        }
        let ctx = self
            .world
            .domain
            .register_context(ContextKind::Thread, None);
        let waitable = self.world.alloc_waitable();
        let (handle, slot) = self
            .world
            .threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .alloc(waitable);
        let cid = CarrierId(self.world.next_carrier.fetch_add(1, Ordering::Relaxed));
        let w = Arc::clone(&self.world);
        let jh = std::thread::spawn(move || {
            run_cranelift_thread(w, cid, ctx, slot, func_idx, arg0, arg1);
        });
        self.world
            .os_threads
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(jh);
        Ok(handle)
    }

    fn thread_join(&self, handle: i32, fuel: &mut FuelCell) -> Result<(i64, i64), Trap> {
        loop {
            match self.world.lookup_thread_join(handle) {
                ThreadJoinLookup::Finished(exit) => match exit {
                    ContextExit::Returned(v) => return Ok((0, v.first().copied().unwrap_or(0))),
                    ContextExit::Trapped(t) => {
                        self.world.ctl.end(ContextExit::Trapped(t));
                        return Err(t);
                    }
                    _ => return Err(Trap::ThreadFault),
                },
                ThreadJoinLookup::Spent => return Ok((i64::from(ERRNO_EBADF), 0)),
                ThreadJoinLookup::Fault => return Err(Trap::ThreadFault),
                ThreadJoinLookup::Running(waitable) => {
                    let key = WaitKey::Object(waitable);
                    let (ctx, kind) = current_ctx_and_kind();
                    let spec = WaiterSpec {
                        domain: self.world.domain.id,
                        ctx,
                        kind,
                        interruptible: true,
                    };
                    match self.world.waits.register_object(key, spec, || {
                        self.world.domain.poll_flags().is_empty()
                            && matches!(
                                self.world.lookup_thread_join(handle),
                                ThreadJoinLookup::Running(_)
                            )
                    }) {
                        RegisteredPark::Immediate(_) => {
                            if self.world.domain.poll_flags().contains(PollFlags::ENDED) {
                                return Err(self.terminal_trap(Trap::OutOfFuel));
                            }
                        }
                        RegisteredPark::Parked { slot, decision } => {
                            if decision == ParkDecision::YieldFiber {
                                {
                                    let mut fibers =
                                        self.world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                                    if let Some(rec) = fibers.get_mut(&ctx) {
                                        rec.parked = Some(FiberParkInfo {
                                            key,
                                            slot,
                                            deadline: None,
                                            ctx,
                                            reexec_on_wake: true,
                                        });
                                    }
                                }
                                let _ = park_current_fiber().ok_or(Trap::FiberFault)?;
                                if self.world.domain.poll_flags().contains(PollFlags::ENDED) {
                                    return Err(self.terminal_trap(Trap::OutOfFuel));
                                }
                            } else {
                                let reason =
                                    self.block_carrier_on_slot(key, ctx, &slot, None, fuel)?;
                                if matches!(reason, WakeReason::Revoked)
                                    || self.world.domain.poll_flags().contains(PollFlags::ENDED)
                                {
                                    return Err(self.terminal_trap(Trap::OutOfFuel));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn fiber_new(&self, func_idx: u32, arg0: i64) -> Result<i64, Trap> {
        let compiled = self.compiled();
        let slot_idx = ((func_idx as u64) & compiled.fn_table_mask) as u32;
        let bridge: Arc<dyn KernelBridge> = Arc::new(CraneliftBridge {
            world: Arc::clone(&self.world),
        });
        let kfiber = KernelFiber::try_new(
            Arc::clone(compiled),
            slot_idx,
            arg0,
            Arc::clone(self.kernel_mem()),
            self.world.domain.poll_word(),
            bridge,
        )
        .ok();
        let (ctx, handle) = self.world.domain.create_fiber(None);
        self.world
            .fibers
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(
                ctx,
                FiberRecord {
                    handle,
                    exec: None,
                    kfiber,
                    started: false,
                    parked: None,
                },
            );
        Ok(handle)
    }

    fn fiber_resume(
        &self,
        handle: i64,
        val: i64,
        block: bool,
        fuel: &mut FuelCell,
        carrier_tls: &mut i64,
    ) -> Result<(i32, i64), Trap> {
        let carrier_id = current_carrier_id();
        loop {
            self.world.expire_fiber_if_timed_out(handle);
            match self.world.domain.claim_fiber_resume(handle, carrier_id) {
                FiberResumeClaim::Claimed { ctx, wake } => {
                    let (mut kfiber, started, reexec_on_wake) = {
                        let mut fibers =
                            self.world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                        let Some(rec) = fibers.get_mut(&ctx) else {
                            return Err(Trap::FiberFault);
                        };
                        let reexec = rec.parked.take().is_some_and(|p| p.reexec_on_wake);
                        let Some(kf) = rec.kfiber.take() else {
                            return Err(Trap::IndirectCallType);
                        };
                        let was_started = rec.started;
                        rec.started = true;
                        (kf, was_started, reexec)
                    };

                    let resume_arg = if !started {
                        val
                    } else if let Some(reason) = wake {
                        if !reexec_on_wake {
                            i64::from(
                                wake_to_futex_status(reason).map_err(|t| self.terminal_trap(t))?,
                            )
                        } else if matches!(reason, WakeReason::Revoked) {
                            return Err(self.terminal_trap(Trap::OutOfFuel));
                        } else {
                            0
                        }
                    } else {
                        val
                    };

                    push_fiber_ctx(ctx);
                    let step = {
                        let _guard = FiberCtxGuard;
                        kfiber.resume(resume_arg, fuel, carrier_tls)
                    };

                    match step {
                        KernelFiberStep::Suspended(yval) => {
                            {
                                let mut fibers =
                                    self.world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(rec) = fibers.get_mut(&ctx) {
                                    rec.kfiber = Some(kfiber);
                                }
                            }
                            self.world.domain.suspend_fiber(handle);
                            return Ok((FIBER_SUSPENDED, yval));
                        }
                        KernelFiberStep::Parked => {
                            // Store the suspended `KernelFiber` back FIRST, then publish `Parked`
                            // on `Domain` so any concurrent wake/claim sees a fully parked stack.
                            let park_info = {
                                let mut fibers =
                                    self.world.fibers.lock().unwrap_or_else(|e| e.into_inner());
                                if let Some(rec) = fibers.get_mut(&ctx) {
                                    rec.kfiber = Some(kfiber);
                                    rec.parked
                                        .as_ref()
                                        .map(|p| (p.key, Arc::clone(&p.slot), p.deadline, p.ctx))
                                } else {
                                    None
                                }
                            };
                            let Some((key, slot, deadline, park_ctx)) = park_info else {
                                return Err(Trap::FiberFault);
                            };
                            let still_parked = self.world.domain.mark_fiber_parked(handle, key);
                            if !still_parked {
                                // Concurrent wake already promoted the fiber to `WakeReady`!
                                continue;
                            }
                            if block {
                                let reason = self
                                    .block_carrier_on_slot(key, park_ctx, &slot, deadline, fuel)?;
                                if matches!(reason, WakeReason::Revoked)
                                    || self.world.domain.poll_flags().contains(PollFlags::ENDED)
                                {
                                    return Err(self.terminal_trap(Trap::OutOfFuel));
                                }
                                continue;
                            }
                            return Ok((FIBER_PARKED, 0));
                        }
                        KernelFiberStep::Complete(rval) => {
                            self.world
                                .fibers
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .remove(&ctx);
                            self.world.domain.finish_fiber(handle);
                            return Ok((FIBER_RETURNED, rval));
                        }
                        KernelFiberStep::Trapped(t) => {
                            self.world
                                .fibers
                                .lock()
                                .unwrap_or_else(|e| e.into_inner())
                                .remove(&ctx);
                            self.world.domain.finish_fiber(handle);
                            self.world.ctl.end(ContextExit::Trapped(t));
                            return Err(t);
                        }
                    }
                }
                FiberResumeClaim::StillParked => {
                    if block {
                        if let Some((key, slot, deadline, park_ctx)) =
                            self.world.fiber_park_slot(handle)
                        {
                            let reason =
                                self.block_carrier_on_slot(key, park_ctx, &slot, deadline, fuel)?;
                            if matches!(reason, WakeReason::Revoked)
                                || self.world.domain.poll_flags().contains(PollFlags::ENDED)
                            {
                                return Err(self.terminal_trap(Trap::OutOfFuel));
                            }
                            continue;
                        }
                    }
                    return Ok((FIBER_PARKED, 0));
                }
                FiberResumeClaim::Spent => return Ok((ERRNO_EBADF, 0)),
                FiberResumeClaim::Fault => return Err(Trap::FiberFault),
            }
        }
    }

    fn fiber_suspend(&self, val: i64) -> Result<i64, Trap> {
        suspend_current_fiber(val).ok_or(Trap::FiberFault)
    }

    fn cap_call(
        &self,
        handle: i32,
        type_id: u32,
        op: u32,
        args: &[i64],
        results: &mut [i64],
    ) -> Result<(), Trap> {
        let (ctx, _) = current_ctx_and_kind();
        match self
            .world
            .caps
            .call(self.world.domain.id, ctx, handle, type_id, op, args, None)
        {
            CapOutcome::Ready(vals) => {
                for (dst, &src) in results.iter_mut().zip(vals.iter()) {
                    *dst = src;
                }
                Ok(())
            }
            CapOutcome::Trap(t) => Err(t),
            CapOutcome::Park { .. } => Err(Trap::CapFault),
        }
    }
}

fn run_cranelift_thread(
    world: Arc<DomainWorld>,
    carrier_id: CarrierId,
    ctx: ContextId,
    slot: u16,
    func_idx: u32,
    arg0: i64,
    arg1: i64,
) {
    let compiled = match world.engine {
        Engine::Cranelift(ref c) => Arc::clone(c),
        Engine::Bytecode(_) => unreachable!(),
    };
    let kmem = Arc::clone(
        world
            .kernel_mem
            .as_ref()
            .expect("Engine::Cranelift initializes KernelMemory"),
    );
    let _guard = install_carrier(carrier_id, ctx, ContextKind::Thread);
    let mut fuel = world.make_fuel_cell();
    let mut tls = i64::from(carrier_id.0);
    let bridge: Arc<dyn KernelBridge> = Arc::new(CraneliftBridge {
        world: Arc::clone(&world),
    });

    let res = compiled.run_entry(
        func_idx,
        &[arg0, arg1],
        &kmem,
        world.domain.poll_word(),
        &mut fuel,
        &mut tls,
        bridge,
    );
    world.refund_fuel(&mut fuel);
    let exit = match res {
        Ok(vals) => ContextExit::Returned(vals),
        Err(t) => world
            .domain
            .terminal_exit()
            .unwrap_or(ContextExit::Trapped(t)),
    };
    world.finish_thread(slot, exit);
}

/// Top-level `Engine::Cranelift` driver entry point invoked by [`crate::driver::run_entry`].
pub(crate) fn run_cranelift_entry(
    world: &Arc<DomainWorld>,
    entry_func: u32,
    args: &[i64],
) -> ContextExit {
    let compiled = match world.engine {
        Engine::Cranelift(ref c) => Arc::clone(c),
        Engine::Bytecode(_) => unreachable!(),
    };
    let kmem = Arc::clone(
        world
            .kernel_mem
            .as_ref()
            .expect("Engine::Cranelift initializes KernelMemory"),
    );
    let _guard = install_carrier(CarrierId(0), ContextId(0), ContextKind::Root);
    let mut fuel = world.make_fuel_cell();
    let mut tls = 0i64;
    let bridge: Arc<dyn KernelBridge> = Arc::new(CraneliftBridge {
        world: Arc::clone(world),
    });

    let res = compiled.run_entry(
        entry_func,
        args,
        &kmem,
        world.domain.poll_word(),
        &mut fuel,
        &mut tls,
        bridge,
    );
    world.refund_fuel(&mut fuel);
    let root_exit = match res {
        Ok(vals) => ContextExit::Returned(vals),
        Err(t) => world
            .domain
            .terminal_exit()
            .unwrap_or(ContextExit::Trapped(t)),
    };
    // Invariant 6: root completion ends the domain and joins all spawned sibling OS threads
    // before returning, so zero background threads outlive the run.
    world.ctl.end(root_exit.clone());
    world.join_all_os_threads();
    world.domain.terminal_exit().unwrap_or(root_exit)
}
