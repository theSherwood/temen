//! #1944 slice 3 — the counted-fuel cell a compile with fuel checks charges ([`crate::VmCtx::fuel`]).
//!
//! Compiled code burns [`FuelCell::left`] one unit per safepoint (word 0, a plain load/decrement/store)
//! and, when it is spent, calls the cell's refill (word 1) instead of trapping at once: the refill
//! draws the next chunk from the budget chain the embedder accounts in ([`BudgetNode`]), and only a
//! spent chain traps `OutOfFuel`. A chain unbounded at every level hands the cell a [`QUANTUM`] at a
//! time all the same, drawn from nothing, so none of it goes back. A harness's `u64` budget is a
//! one-node chain of its own ([`FuelCell::fixed`], #3069), so every cell draws, and each vCPU charges
//! a cell of its own. The JIT meters; the embedder's budget tree accounts.
//!
//! #1991 — the refill is also where a child pool's task ends its quantum: it hands its worker to the
//! next runnable task, if one waits ([`crate::child_exec::quantum_end`]). Every pool task counts, so
//! every one reaches a refill within a chunk, metered or not.
//!
//! The refill is called through a `PreserveAll` trampoline ([`refill_trampoline`]), which clobbers
//! no register. A plain call would clobber the caller-saved registers at every safepoint, so a value
//! live across one would have to sit in a callee-saved register, which the function's prologue then
//! saves and restores on every call, refill or not (#2147).

#[cfg(loom)]
use loom::sync::atomic::{AtomicI64, Ordering};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

/// What an unmetered chain hands a cell at each refill: as many safepoints as the interpreter's pool
/// runs a vCPU for before its quantum ends.
pub(crate) const QUANTUM: u64 = temen_ir::POOL_QUANTUM;

/// A domain's node in the embedder's budget chain (temen-run's, over a `temen_interp` budget node):
/// where a [`FuelCell`] draws its next chunk, and what a domain's vCPUs (#2001) and fibers (#2112)
/// are charged to.
pub trait BudgetNode: Send + Sync {
    /// Up to a chunk of fuel from the chain, charged to every level: `None` when every level is
    /// unbounded (nothing to meter), `Some(0)` when the chain is spent.
    fn draw(&self) -> Option<u64>;
    /// Hand back `unspent`, the rest of a draw its cell did not burn.
    fn give_back(&self, unspent: u64);
    /// #3081 — take exactly `n` from the chain, charged to every level, all or nothing: a spawned
    /// vCPU's entry, paid at the spawn. `None` when every level is unbounded (nothing charged),
    /// `Some(false)` when the chain has less than `n` of room (nothing charged), `Some(true)` when `n`
    /// was charged. One step, never a draw and a give-back, which would empty half the room for a
    /// moment under another vCPU's refill.
    fn take_exact(&self, n: u64) -> Option<bool>;
    /// The fuel room left along the chain, `-1` when every level is unbounded.
    fn room(&self) -> i64;
    /// Charge one live vCPU to every level, all or nothing: `false` when a level's `spawn` is full.
    fn charge_vcpu(&self) -> bool;
    /// [`Self::charge_vcpu`] past any ceiling, since it lived before the freeze, but never past the
    /// process's pools (#2318): a vCPU a thaw re-creates. `false` when they have no room.
    fn rebuild_vcpu(&self) -> bool;
    /// Hand back a vCPU's charge when it ends.
    fn vcpu_ended(&self);
    /// #2112 — charge one live fiber, [`temen_ir::FIBER_STACK`] of `mem`, to every level, all or
    /// nothing: `false` when a level's `mem` is full.
    fn charge_fiber(&self) -> bool;
    /// [`Self::charge_fiber`] past any ceiling: a fiber a thaw re-creates, which lived before the
    /// freeze.
    fn force_fiber(&self);
    /// Hand back a fiber's charge when it ends.
    fn fiber_ended(&self);
}

/// #3069 — a fixed allowance, as a one-node chain: what is left of it, `-1` when it is unbounded (an
/// allowance past `i64::MAX`, as the oracle's root reads one). Each vCPU's cell draws from it as from
/// any chain, by the oracle's rule ([`temen_ir::fuel_draw`]), and hands back what it did not burn,
/// so no two vCPUs charge one cell. It counts no vCPU and no fiber: a domain's own node does.
pub(crate) struct Allowance(AtomicI64);

impl Allowance {
    pub(crate) fn new(n: u64) -> Allowance {
        Allowance(AtomicI64::new(i64::try_from(n).unwrap_or(-1)))
    }
}

impl BudgetNode for Allowance {
    fn draw(&self) -> Option<u64> {
        let mut room = self.0.load(Ordering::Relaxed);
        loop {
            let take = temen_ir::fuel_draw(u64::try_from(room).ok()?);
            // `take <= room`, so the new room is never negative.
            match self.0.compare_exchange_weak(
                room,
                room - take as i64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(take),
                Err(now) => room = now,
            }
        }
    }
    fn give_back(&self, unspent: u64) {
        // Only a bounded allowance hands out draws, so only it takes any back, and never more than it
        // handed out: the sum stays within `i64`. An unbounded one (`-1`) stays unbounded.
        if unspent > 0 && self.0.load(Ordering::Relaxed) >= 0 {
            self.0.fetch_add(unspent as i64, Ordering::Relaxed);
        }
    }
    fn take_exact(&self, n: u64) -> Option<bool> {
        let mut room = self.0.load(Ordering::Relaxed);
        loop {
            let left = u64::try_from(room).ok()?.checked_sub(n);
            let Some(left) = left else { return Some(false) };
            match self.0.compare_exchange_weak(
                room,
                left as i64,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Some(true),
                Err(now) => room = now,
            }
        }
    }
    fn room(&self) -> i64 {
        self.0.load(Ordering::Relaxed)
    }
    fn charge_vcpu(&self) -> bool {
        true
    }
    fn rebuild_vcpu(&self) -> bool {
        true
    }
    fn vcpu_ended(&self) {}
    fn charge_fiber(&self) -> bool {
        true
    }
    fn force_fiber(&self) {}
    fn fiber_ended(&self) {}
}

/// A domain's counted-fuel cell. `repr(C)`: compiled code reads `left` at word 0 and calls `refill`
/// (word 1) and `remaining` (word 2) through the cell's own pointer, so no address but the cell's is
/// baked into code.
#[repr(C)]
pub struct FuelCell {
    /// What is left of the last draw.
    pub left: u64,
    /// [`refill_trampoline`]'s address: called with the cell, returns nothing, clobbers no register.
    refill: usize,
    remaining: unsafe extern "C" fn(*const FuelCell) -> i64,
    /// What the last refill handed its caller to charge from: the allowance it drew or found, `0`
    /// when the chain is spent. Compiled code reads it after a refill instead of `left`, which another
    /// context's plain store can have overwritten ([`fuel_refill`]).
    granted: u64,
    /// The chain the next draw comes from. Locked by a refill, so contexts sharing the cell draw one
    /// at a time.
    src: Mutex<Arc<dyn BudgetNode>>,
    /// `left` is not the chain's to take back: #1991, the chain was unbounded at every level at the
    /// last refill, which granted a [`QUANTUM`] drawn from nothing; or #3081, `left` is the credit
    /// seeded for a spawned vCPU's entry, which its spawn alone charges ([`Self::seed_entry`]), and
    /// which is never refunded. None of it
    /// goes back to the chain (`settle`, `rearm`, `Drop`). Written by a refill and by the seeding,
    /// under `src`'s lock. A refill that draws rewrites it (to `true` only on an unbounded chain); the
    /// refill's early return, which keeps another context's refill, leaves it as it is.
    free: bool,
}

impl FuelCell {
    /// A fixed allowance of `n`: a cell drawing from a one-node chain of `n` ([`Allowance`], #3069),
    /// unbounded past `i64::MAX`. What the run did not burn is [`Self::can_burn`] once its vCPUs end.
    pub fn fixed(n: u64) -> Box<FuelCell> {
        Self::drawing(Arc::new(Allowance::new(n)))
    }

    /// A cell that draws from `src`, none drawn yet: the first safepoint draws.
    pub fn drawing(src: Arc<dyn BudgetNode>) -> Box<FuelCell> {
        Box::new(Self::value(src))
    }

    /// [`Self::drawing`] when `src`'s chain is bounded; `None` when every level is unbounded — code
    /// with nothing to meter compiles without fuel checks.
    pub fn metering(src: Arc<dyn BudgetNode>) -> Option<Box<FuelCell>> {
        (src.room() >= 0).then(|| Self::drawing(src))
    }

    fn value(src: Arc<dyn BudgetNode>) -> FuelCell {
        FuelCell {
            left: 0,
            refill: refill_trampoline(),
            remaining: fuel_remaining,
            granted: 0,
            src: Mutex::new(src),
            free: false,
        }
    }

    /// The chain this cell draws from.
    fn chain(&self) -> Arc<dyn BudgetNode> {
        self.src.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// #2488 — a cell for another vCPU of this cell's domain, drawing from the same chain, none drawn
    /// yet.
    pub(crate) fn sibling(&self) -> Box<FuelCell> {
        Self::drawing(self.chain())
    }

    /// #3081 — pay a spawned vCPU's entry at its spawn, as the interpreters do: exactly one unit,
    /// taken from this cell's chain ([`BudgetNode::take_exact`]), all or nothing. It touches no cell:
    /// the vCPU seeds its own with the unit once it runs ([`Self::seed_entry`]). `None` when the chain is
    /// unbounded (nothing to pay), `Some(false)` when it is spent (the spawn runs out of fuel).
    pub(crate) fn pay_entry(&self) -> Option<bool> {
        self.chain().take_exact(1)
    }

    /// #3081 — seed `cell` with a spawned vCPU's entry: one unit, a credit against the charge its
    /// entry's prologue makes, so the entry is charged once, by its spawn ([`Self::pay_entry`]),
    /// whatever the spawn paid: one unit on a bounded chain, nothing on an unbounded one, as on
    /// every engine. The seed is never refunded: the cell holds exactly the unit, marked as none of
    /// the chain's (`free`), so whether the vCPU burns it or stops first (its run ended before its
    /// first charge: a kill check or a domain poll comes first in the prologue, or lazy code never
    /// ran), nothing goes back. The refill that follows the burn rewrites the mark as any refill
    /// does: cleared on a bounded chain, set again on an unbounded one, whose quantum is drawn from
    /// nothing. What a recycled cell held goes back to its chain first,
    /// under the lock a refill takes and through the raw pointer, as [`Self::settle`] does; a fiber
    /// of the cell's earlier vCPU that refills it in between races it as any shared charge does
    /// (#3080).
    ///
    /// Both of the JIT's routes seed through it, each on the thread's own cell. The kernel route
    /// (temen-run's `jit_kernel::tier_run`) seeds every thread entry on a node-backed cell. Today's
    /// (`os_thread_rt::run_child`) gates on the answer its spawn carried (`paid_entry`), which is the
    /// same answer in every reachable case. The two differ only for a chain unbounded at the spawn
    /// and bounded when the thread starts, and nothing bounds a chain during a run: `transfer` only
    /// raises a ceiling, and an activation's limit is set before its run starts.
    ///
    /// # Safety
    /// - `cell` is a live cell that no context but this vCPU's (and #3080's fibers) charges.
    /// - It runs before this vCPU's first guest code: none of the vCPU's code charges the cell yet.
    /// - The vCPU's spawn completed: its entry is charged by the spawn alone (`pay_entry` answered
    ///   `Some(true)` or `None`), and the seed stands for the prologue's own charge of it.
    pub unsafe fn seed_entry(cell: *mut FuelCell) {
        let src = (*cell).src.lock().unwrap_or_else(|e| e.into_inner());
        let left = core::ptr::addr_of_mut!((*cell).left);
        let free = core::ptr::addr_of_mut!((*cell).free);
        let unspent = core::ptr::read_volatile(left);
        if !core::ptr::read_volatile(free) {
            src.give_back(unspent);
        }
        core::ptr::write_volatile(left, 1);
        core::ptr::write_volatile(free, true);
    }

    /// #2488 — a vCPU that charged this cell has ended: hand back what its draw left, as an ended
    /// vCPU does (invariant 3). The cell keeps its chain, so code that still charges it draws again.
    ///
    /// # Safety
    /// `cell` is a live cell. A fiber that outlived its vCPU may still charge it: a charge racing this
    /// one can be lost or counted twice, by at most one chunk.
    pub(crate) unsafe fn settle(cell: *mut FuelCell) {
        let src = (*cell).src.lock().unwrap_or_else(|e| e.into_inner());
        let left = core::ptr::addr_of_mut!((*cell).left);
        let unspent = core::ptr::read_volatile(left);
        core::ptr::write_volatile(left, 0);
        if !(*cell).free {
            src.give_back(unspent);
        }
    }

    /// #2113 — point a cell that compiled code keeps across runs (a compile-once program's) at the
    /// chain its next run draws from: what the last draw left goes back to its chain first. `None`
    /// points it at a spent allowance, so code run before the next re-arm traps `OutOfFuel`. Only
    /// between runs: no code may be charging the cell.
    pub fn rearm(&mut self, src: Option<Arc<dyn BudgetNode>>) {
        let src = src.unwrap_or_else(|| Arc::new(Allowance::new(0)));
        let slot = self.src.get_mut().unwrap_or_else(|e| e.into_inner());
        let left = std::mem::take(&mut self.left);
        if !std::mem::take(&mut self.free) {
            slot.give_back(left);
        }
        *slot = src;
    }

    /// What this cell's vCPUs can still burn: `left` plus the chain's room, `u64::MAX` unmetered.
    pub fn can_burn(&self) -> u64 {
        let left = unsafe { core::ptr::read_volatile(&self.left) };
        match self.chain().room() {
            r if r < 0 => u64::MAX,
            r => left.saturating_add(r as u64),
        }
    }
}

/// #3000 — a cell several contexts may charge, owned by all of them: it goes with the last. Compiled
/// code charges it through [`Self::get`], whose address never moves, so sharing it costs the charge
/// path nothing; it is one allocation, as a boxed cell is.
#[cfg(fiber_rt)]
pub(crate) struct SharedCell(core::cell::UnsafeCell<FuelCell>);

// SAFETY: the cell is a `FuelCell`, whose shared state is behind its lock, and whose `left` compiled
// code charges through the raw pointer, as it does any cell.
#[cfg(fiber_rt)]
unsafe impl Send for SharedCell {}
#[cfg(fiber_rt)]
unsafe impl Sync for SharedCell {}

#[cfg(fiber_rt)]
impl SharedCell {
    /// [`FuelCell::fixed`], shared.
    pub(crate) fn fixed(n: u64) -> Arc<SharedCell> {
        Self::drawing(Arc::new(Allowance::new(n)))
    }

    /// [`FuelCell::sibling`] of `cell`, shared.
    pub(crate) fn sibling_of(cell: &FuelCell) -> Arc<SharedCell> {
        Self::drawing(cell.chain())
    }

    fn drawing(src: Arc<dyn BudgetNode>) -> Arc<SharedCell> {
        Arc::new(SharedCell(core::cell::UnsafeCell::new(FuelCell::value(
            src,
        ))))
    }

    /// The cell, live while this is.
    pub(crate) fn get(&self) -> *mut FuelCell {
        self.0.get()
    }
}

impl Drop for FuelCell {
    fn drop(&mut self) {
        if !self.free {
            let src = self.src.get_mut().unwrap_or_else(|e| e.into_inner());
            src.give_back(self.left);
        }
    }
}

/// A spent cell's refill, called by compiled code through [`refill_trampoline`]: it puts the new
/// allowance (at least 1) in `left`, and in [`FuelCell::granted`] what the caller charges from: that
/// allowance, or `0` when the chain is spent, on which the caller traps `OutOfFuel`. Each vCPU charges
/// a cell of its own (#2488, #3069), but a fiber charges its creator's ([`crate::fiber_rt`]), wherever
/// it is resumed, so two contexts can still share one: a refill another made while this one waited
/// for the lock is kept, not drawn again, and what it left is granted.
///
/// The caller charges from `granted`, never from a reload of `left`. Contexts charge a shared cell
/// with plain loads and stores, so one that loaded `1` before a refill can store its `0` just after
/// it. A reload of `left` that read that `0` trapped `OutOfFuel` with the chain still full (#2202:
/// five threads in a loop trapped within milliseconds). Only a refill writes `granted`, under the
/// lock, and it writes `0` only when the chain is spent.
///
/// A child pool's task ends its quantum here first, before the lock ([`crate::child_exec::quantum_end`]):
/// it may park, as a runnable task, until a worker takes it again.
///
/// # Safety
/// `cell` is a live [`FuelCell`].
pub(crate) unsafe extern "C" fn fuel_refill(cell: *mut FuelCell) {
    crate::child_exec::quantum_end();
    // Only the `src` field is borrowed: compiled code writes `left` through the raw cell pointer.
    let src = (*cell).src.lock().unwrap_or_else(|e| e.into_inner());
    let left = core::ptr::addr_of_mut!((*cell).left);
    let granted = core::ptr::addr_of_mut!((*cell).granted);
    let refilled = core::ptr::read_volatile(left);
    if refilled > 0 {
        core::ptr::write_volatile(granted, refilled);
        return;
    }
    let (drawn, free) = match src.draw() {
        // Every level unbounded: nothing to meter, but a quantum to count (#1991).
        None => (QUANTUM, true),
        Some(n) => (n, false),
    };
    *core::ptr::addr_of_mut!((*cell).free) = free;
    core::ptr::write_volatile(left, drawn);
    core::ptr::write_volatile(granted, drawn);
}

/// The address of [`fuel_refill`] behind a trampoline in Cranelift's `PreserveAll` convention, which
/// every [`FuelCell`] calls through. The trampoline saves whatever the platform call to
/// [`fuel_refill`] clobbers, so the safepoint that calls it clobbers nothing. Compiled once per
/// process, by the same ISA configuration as every guest compile, and never freed: every cell points
/// at it. The static owns the module, not just the code's address, so the module's own allocations
/// stay reachable; a `mem::forget` left LeakSanitizer to report them at every exit (#2277).
pub(crate) fn refill_trampoline() -> usize {
    static TRAMPOLINE: std::sync::OnceLock<(crate::OwnedJit, usize)> = std::sync::OnceLock::new();
    TRAMPOLINE
        .get_or_init(|| {
            compile_refill_trampoline().unwrap_or_else(|e| {
                panic!("the fuel refill trampoline must compile wherever the JIT does: {e:?}")
            })
        })
        .1
}

fn compile_refill_trampoline() -> Result<(crate::OwnedJit, usize), crate::JitError> {
    use cranelift_codegen::ir::{types::I64, AbiParam, InstBuilder, Signature};
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_module::{Linkage, Module};
    let backend = |e: cranelift_module::ModuleError| crate::JitError::Backend(e.to_string());
    let mut module = crate::new_jit_module(crate::Imports::NONE, &crate::FastClaims::new())?;
    let mut sig = Signature::new(CallConv::PreserveAll);
    sig.params.push(AbiParam::new(I64)); // the cell
    let id = module
        .declare_function("fuel_refill_trampoline", Linkage::Local, &sig)
        .map_err(backend)?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let cell = b.block_params(entry)[0];
    let callees = crate::Callees::new(&mut module, &[], &[], &[], &crate::FastClaims::new())?;
    crate::call_host(&callees, &mut b, crate::HostFn::FuelRefill, &[cell])?;
    b.ins().return_(&[]);
    b.finalize();
    crate::define_verified(&mut module, id, &mut ctx)?;
    module.finalize_definitions().map_err(backend)?;
    let code = module.get_finalized_function(id) as usize;
    Ok((crate::OwnedJit::new(module), code))
}

/// `fuel.remaining` (self op 13) on a metered compile: [`FuelCell::can_burn`], `i64::MAX` unmetered.
///
/// # Safety
/// `cell` is a live [`FuelCell`].
unsafe extern "C" fn fuel_remaining(cell: *const FuelCell) -> i64 {
    (*cell).can_burn().min(i64::MAX as u64) as i64
}

/// Word offsets compiled code uses.
pub(crate) const REFILL_OFF: i32 = 8;
pub(crate) const REMAINING_OFF: i32 = 16;
pub(crate) const GRANTED_OFF: i32 = 24;

const _: () = {
    assert!(core::mem::offset_of!(FuelCell, left) == 0);
    assert!(core::mem::offset_of!(FuelCell, refill) == REFILL_OFF as usize);
    assert!(core::mem::offset_of!(FuelCell, remaining) == REMAINING_OFF as usize);
    assert!(core::mem::offset_of!(FuelCell, granted) == GRANTED_OFF as usize);
};

#[cfg(test)]
mod tests {
    use super::{BudgetNode, FuelCell};
    use crate::{JitOutcome, RunOpts, TrapKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use temen_interp::Value;
    use temen_ir::{
        BinOp, Block, CastOp, CmpOp, FBinOp, FloatTy, Func, Inst, IntTy, Module, Terminator,
        ValType,
    };

    /// A chain that hands out one unit per draw, `left` of them, then is spent: every safepoint
    /// after the first runs the refill.
    struct Drip {
        left: AtomicU64,
        draws: AtomicU64,
    }

    impl Drip {
        fn new(units: u64) -> Arc<Drip> {
            Arc::new(Drip {
                left: AtomicU64::new(units),
                draws: AtomicU64::new(0),
            })
        }
    }

    impl BudgetNode for Drip {
        fn draw(&self) -> Option<u64> {
            // Work the float registers, as a real chain's host code may: a trampoline that did not
            // save them would hand the loop back clobbered floats.
            let x: f64 = (0..16)
                .map(|i| std::hint::black_box(f64::from(i)) * 1.5)
                .sum();
            std::hint::black_box(x);
            self.draws.fetch_add(1, Ordering::Relaxed);
            let had = self
                .left
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
            Some(u64::from(had.is_ok()))
        }
        fn give_back(&self, _: u64) {}
        fn take_exact(&self, n: u64) -> Option<bool> {
            let mut had = self.left.load(Ordering::Relaxed);
            loop {
                let Some(left) = had.checked_sub(n) else {
                    return Some(false);
                };
                match self.left.compare_exchange_weak(
                    had,
                    left,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(true),
                    Err(now) => had = now,
                }
            }
        }
        fn room(&self) -> i64 {
            self.left.load(Ordering::Relaxed).min(i64::MAX as u64) as i64
        }
        fn charge_vcpu(&self) -> bool {
            true
        }
        fn rebuild_vcpu(&self) -> bool {
            true
        }
        fn vcpu_ended(&self) {}
        fn charge_fiber(&self) -> bool {
            true
        }
        fn force_fiber(&self) {}
        fn fiber_ended(&self) {}
    }

    const INTS: u32 = 12;
    const FLOATS: u32 = 8;

    /// `f(n)`: `n` turns of a loop that carries 12 `i64`s and 8 `f64`s around its back edge, which
    /// charges fuel. That is more values than the callee-saved registers hold (and every `xmm`
    /// register is caller-saved on System V), so some ride across the refill in registers a platform
    /// call would clobber. Returns all of them folded into one `i64`.
    fn many_live_values() -> Module {
        let (k, j) = (INTS, FLOATS);
        let carried: Vec<ValType> = std::iter::once(ValType::I64)
            .chain((0..k).map(|_| ValType::I64))
            .chain((0..j).map(|_| ValType::F64))
            .collect();
        let all = |n: u32| (0..n).collect::<Vec<u32>>();
        let p = 1 + k + j; // values a loop block's params take
                           // b0(n): the starting values, then into the loop.
        let mut b0 = vec![];
        for x in 0..k {
            b0.push(Inst::ConstI64(i64::from(x) + 1));
        }
        for x in 0..j {
            b0.push(Inst::ConstF64((f64::from(x) + 1.0).to_bits()));
        }
        let b0 = Block {
            parent: 0,
            captures: vec![],
            params: vec![ValType::I64],
            insts: b0,
            term: Terminator::Br {
                target: 1,
                args: all(1 + k + j),
            },
        };
        // b1(i, a.., f..): i != 0 ? b2 : b3.
        let b1 = Block {
            parent: 0,
            captures: vec![],
            params: carried.clone(),
            insts: vec![
                Inst::ConstI64(0),
                Inst::IntCmp {
                    ty: IntTy::I64,
                    op: CmpOp::Ne,
                    a: 0,
                    b: p,
                },
            ],
            term: Terminator::BrIf {
                cond: p + 1,
                then_blk: 2,
                then_args: all(p),
                else_blk: 3,
                else_args: (1..p).collect(),
            },
        };
        // b2(i, a.., f..): i - 1, each a += its neighbour, each f += its neighbour; back to b1.
        let mut b2 = vec![
            Inst::ConstI64(1),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: p,
            },
        ];
        for x in 0..k {
            b2.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1 + x,
                b: 1 + (x + 1) % k,
            });
        }
        for x in 0..j {
            b2.push(Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Add,
                a: 1 + k + x,
                b: 1 + k + (x + 1) % j,
            });
        }
        let b2 = Block {
            parent: 0,
            captures: vec![],
            params: carried.clone(),
            insts: b2,
            term: Terminator::Br {
                target: 1,
                args: (p + 1..p + 2 + k + j).collect(),
            },
        };
        // b3(a.., f..): the sum of the ints and of the floats' bits.
        let mut b3 = vec![];
        for x in 0..j {
            b3.push(Inst::Cast {
                op: CastOp::ReinterpF64I64,
                a: k + x,
            });
        }
        let mut acc = 0;
        for (n, x) in (1..k).chain(k + j..k + j + j).enumerate() {
            b3.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: acc,
                b: x,
            });
            acc = k + j + j + n as u32; // the sum so far: the add just pushed
        }
        let b3 = Block {
            parent: 0,
            captures: vec![],
            params: carried[1..].to_vec(),
            insts: b3,
            term: Terminator::Return(vec![acc]),
        };
        let m = Module {
            funcs: vec![Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![b0, b1, b2, b3],
            }],
            ..Default::default()
        };
        temen_verify::verify_module(&m).expect("the fixture verifies");
        m
    }

    fn run_jit(m: &Module, n: i64, cell: &mut FuelCell) -> JitOutcome {
        crate::run_inner(
            m,
            0,
            &[n],
            crate::empty_cap_thunk,
            core::ptr::null_mut(),
            RunOpts {
                fuel: Some(cell),
                ..RunOpts::default()
            },
        )
        .expect("compiles and runs")
        .0
    }

    #[test]
    fn values_live_across_a_refill_keep_their_registers() {
        let m = many_live_values();
        let n = 2000;
        let mut fuel = u64::MAX;
        let want = match temen_interp::run(&m, 0, &[Value::I64(n)], &mut fuel)
            .expect("the interpreter runs it")[..]
        {
            [Value::I64(v)] => v,
            ref other => panic!("one i64 result, got {other:?}"),
        };
        let drip = Drip::new(u64::MAX);
        let mut cell = FuelCell::drawing(drip.clone());
        assert_eq!(run_jit(&m, n, &mut cell), JitOutcome::Returned(vec![want]));
        assert_eq!(
            drip.draws.load(Ordering::Relaxed),
            n as u64 + 1,
            "the entry and every back edge refilled"
        );
    }

    /// A chain that hands out [`CHUNK`] per draw, never runs dry, and counts what goes back; or, when
    /// `unbounded`, one unbounded at every level, which has nothing to hand out.
    #[derive(Default)]
    struct Chunks {
        unbounded: bool,
        draws: AtomicU64,
        back: AtomicU64,
    }

    const CHUNK: u64 = 1 << 20;

    impl BudgetNode for Chunks {
        fn draw(&self) -> Option<u64> {
            self.draws.fetch_add(1, Ordering::Relaxed);
            (!self.unbounded).then_some(CHUNK)
        }
        fn give_back(&self, unspent: u64) {
            self.back.fetch_add(unspent, Ordering::Relaxed);
        }
        fn take_exact(&self, _: u64) -> Option<bool> {
            (!self.unbounded).then_some(true)
        }
        fn room(&self) -> i64 {
            if self.unbounded {
                -1
            } else {
                i64::MAX
            }
        }
        fn charge_vcpu(&self) -> bool {
            true
        }
        fn rebuild_vcpu(&self) -> bool {
            true
        }
        fn vcpu_ended(&self) {}
        fn charge_fiber(&self) -> bool {
            true
        }
        fn force_fiber(&self) {}
        fn fiber_ended(&self) {}
    }

    /// The root spawns a vCPU that turns a loop its argument's times, and returns what it joins.
    const SPAWNS_A_LOOP: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  v2 = thread.spawn 1 v1 v0
  v3_st, v3 = thread.join v2
  return v3
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i64.const 0
  br 1(v2, v1)
}
block 1 (v3: i64, v4: i64) {
  v5 = i64.const 1
  v6 = i64.add v3 v5
  v7 = i64.lt_s v6 v4
  br_if v7 1(v6, v4) 2(v6)
}
block 2 (v8: i64) {
  return v8
  }
}
";

    /// #2488 — a spawned vCPU charges a cell of its own, drawn from the root cell's chain, and hands
    /// back what it did not burn when it ends. The root's cell pays only the root's safepoints, so
    /// the domain's vCPUs no longer store to one cache line.
    #[test]
    fn a_spawned_vcpu_charges_a_cell_of_its_own() {
        if !temen_fiber::supported() {
            return;
        }
        let m = temen_text::parse_module(SPAWNS_A_LOOP).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let chain = Arc::new(Chunks::default());
        let mut root = FuelCell::drawing(chain.clone());
        let n = 10_000;
        assert_eq!(run_jit(&m, n, &mut root), JitOutcome::Returned(vec![n]));
        assert_eq!(
            chain.draws.load(Ordering::Relaxed),
            2,
            "the root's cell and the vCPU's each drew one chunk"
        );
        let root_burned = CHUNK - root.left;
        assert!(
            root_burned < 10,
            "the root's cell paid the root's safepoints only, not the loop's: {root_burned}"
        );
        let vcpu_burned = CHUNK - chain.back.load(Ordering::Relaxed);
        assert_eq!(
            vcpu_burned,
            n as u64 - 1,
            "the vCPU's cell paid the loop's back-edges and handed back the rest; its entry was paid \
             at the spawn, outside the chunk (#3081)"
        );
    }

    /// #1991 — a chain unbounded at every level hands a cell a [`super::QUANTUM`] at a time, so a
    /// pool task reaches the end of its quantum, and none of it goes back: nothing was drawn. So
    /// it is for the spawned vCPU's own cell (#2488).
    #[test]
    fn an_unbounded_chains_quanta_go_back_to_nothing() {
        if !temen_fiber::supported() {
            return;
        }
        let m = temen_text::parse_module(SPAWNS_A_LOOP).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let chain = Arc::new(Chunks {
            unbounded: true,
            ..Chunks::default()
        });
        let mut root = FuelCell::drawing(chain.clone());
        let n = 3 * super::QUANTUM as i64;
        assert_eq!(run_jit(&m, n, &mut root), JitOutcome::Returned(vec![n]));
        assert_eq!(root.can_burn(), u64::MAX, "nothing to meter");
        drop(root);
        assert!(
            chain.draws.load(Ordering::Relaxed) > 3,
            "the loop drew a quantum at a time, three times, and the root once"
        );
        assert_eq!(chain.back.load(Ordering::Relaxed), 0, "nothing went back");
    }

    /// #3081 — a seeded entry is never refunded, and a recycled cell's leftover never adds to the
    /// next one. A spawn pays a unit from the allowance; the cell holds it, marked `free`. A vCPU that
    /// never charged it settles and hands back nothing. Seeded again while it still holds that unit,
    /// it holds one, not two, and hands nothing back. Recycled to a vCPU that is not seeded, its first
    /// refill draws as any does and clears the mark, so seeding it next hands that draw's leftover
    /// back. An unbounded allowance pays nothing.
    #[test]
    fn a_seeded_entry_is_never_refunded() {
        use super::{fuel_refill, Allowance};
        let chain = Arc::new(Allowance::new(10));
        let root = FuelCell::drawing(chain.clone());
        let mut cell = root.sibling();
        let cell = &mut *cell as *mut FuelCell;
        assert_eq!(root.pay_entry(), Some(true));
        // SAFETY: `cell` is live and nothing else charges it.
        unsafe { FuelCell::seed_entry(cell) };
        assert_eq!((unsafe { (*cell).left }, chain.room()), (1, 9));
        // Its vCPU stops before its first charge: nothing goes back.
        unsafe { FuelCell::settle(cell) };
        assert_eq!(
            (unsafe { (*cell).left }, chain.room()),
            (0, 9),
            "the paid unit stays paid"
        );
        // Recycled to a seeded vCPU while it still holds an unburned paid unit: one unit, not the
        // leftover plus one, and the leftover does not go back.
        assert_eq!(root.pay_entry(), Some(true));
        unsafe { FuelCell::seed_entry(cell) };
        assert_eq!((unsafe { (*cell).left }, chain.room()), (1, 8));
        assert_eq!(root.pay_entry(), Some(true));
        unsafe { FuelCell::seed_entry(cell) };
        assert_eq!(
            (unsafe { (*cell).left }, chain.room()),
            (1, 7),
            "one unit, nothing back"
        );
        unsafe { FuelCell::settle(cell) };
        assert_eq!((unsafe { (*cell).left }, chain.room()), (0, 7));
        // Recycled to a vCPU that is not seeded: its first charge draws, which clears the mark.
        unsafe { fuel_refill(cell) };
        assert_eq!(
            (unsafe { (*cell).left }, chain.room()),
            (4, 3),
            "half of 7, rounded up, drawn"
        );
        // Then to a seeded vCPU, unsettled: the draw's leftover goes back, and the cell holds 1.
        assert_eq!(root.pay_entry(), Some(true));
        unsafe { FuelCell::seed_entry(cell) };
        assert_eq!(
            (unsafe { (*cell).left }, chain.room()),
            (1, 6),
            "4 back, 1 paid"
        );
        unsafe { FuelCell::settle(cell) };
        assert_eq!(chain.room(), 6, "the paid unit stays paid");
        // A spent allowance cannot pay; an unbounded one has nothing to pay.
        let spent = FuelCell::drawing(Arc::new(Allowance::new(0)));
        assert_eq!(spent.pay_entry(), Some(false));
        let unbounded = FuelCell::fixed(u64::MAX);
        assert_eq!(unbounded.pay_entry(), None);
    }

    #[test]
    fn a_spent_chain_still_traps_out_of_fuel() {
        let m = many_live_values();
        let drip = Drip::new(10);
        let mut cell = FuelCell::drawing(drip.clone());
        assert_eq!(
            run_jit(&m, 100, &mut cell),
            JitOutcome::Trapped(TrapKind::OutOfFuel)
        );
        assert_eq!(
            drip.draws.load(Ordering::Relaxed),
            11,
            "ten draws of one unit, then the one that found the chain spent"
        );
    }
}

/// #3069 — a fixed allowance's draw and give-back under loom, on [`Allowance`]'s own steps.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::{Allowance, BudgetNode};
    use loom::sync::Arc;

    /// Two vCPUs each pay an entry with an exact take and then draw from a small allowance at once,
    /// each burning one unit of what it drew and handing the rest back, then drawing again. Every
    /// interleaving hands out no more than the allowance holds, and what is left at the end is the
    /// allowance less exactly what they burned.
    #[test]
    fn loom_an_allowance_hands_out_no_more_than_it_holds() {
        loom::model(|| {
            const N: u64 = 6;
            let a = Arc::new(Allowance::new(N));
            let vcpu = |a: Arc<Allowance>| {
                move || {
                    assert_eq!(a.take_exact(1), Some(true), "an entry fits");
                    let mut burned = 1;
                    for _ in 0..2 {
                        match a.draw() {
                            Some(0) => {}
                            Some(n) => {
                                assert!(n <= N);
                                burned += 1;
                                a.give_back(n - 1);
                            }
                            None => panic!("a bounded allowance meters"),
                        }
                    }
                    burned
                }
            };
            let t = loom::thread::spawn(vcpu(Arc::clone(&a)));
            let here = vcpu(Arc::clone(&a))();
            let burned = here + t.join().unwrap();
            assert!(a.room() >= 0);
            assert_eq!(a.room() as u64, N - burned);
            // A draw can meet a room the other vCPU's draw holds for a moment (the chunk rule), so
            // not every draw finds fuel; both entries always do, and nothing is lost or made.
            assert!((2..=N).contains(&burned), "{burned}");
        });
    }
}
