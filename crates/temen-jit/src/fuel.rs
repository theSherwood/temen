//! #1944 slice 3 — the counted-fuel cell a compile with fuel checks charges ([`crate::VmCtx::fuel`]).
//!
//! Compiled code burns [`FuelCell::left`] one unit per safepoint (word 0, a plain load/decrement/store)
//! and, when it is spent, calls the cell's refill (word 1) instead of trapping at once: the refill
//! draws the next chunk from the budget chain the embedder accounts in ([`BudgetNode`]), and only a
//! spent chain traps `OutOfFuel`. A cell with no source is a fixed allowance, refilled never — a
//! carve child's, or a harness's `u64` budget. The JIT meters; the embedder's budget tree accounts.

use std::sync::{Arc, Mutex};

/// A domain's node in the embedder's budget chain (temen-run's, over a `temen_interp` budget node):
/// where a [`FuelCell`] draws its next chunk, and what a domain's vCPUs (#2001) and fibers (#2112)
/// are charged to.
pub trait BudgetNode: Send + Sync {
    /// Up to a chunk of fuel from the chain, charged to every level: `None` when every level is
    /// unbounded (nothing to meter), `Some(0)` when the chain is spent.
    fn draw(&self) -> Option<u64>;
    /// Hand back `unspent`, the rest of a draw its cell did not burn.
    fn give_back(&self, unspent: u64);
    /// The fuel room left along the chain, `-1` when every level is unbounded.
    fn room(&self) -> i64;
    /// Charge one live vCPU to every level, all or nothing: `false` when a level's `spawn` is full.
    fn charge_vcpu(&self) -> bool;
    /// [`Self::charge_vcpu`] past any ceiling: a vCPU a thaw re-creates, which lived before the freeze.
    fn force_vcpu(&self);
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

/// A domain's counted-fuel cell. `repr(C)`: compiled code reads `left` at word 0 and calls `refill`
/// (word 1) and `remaining` (word 2) through the cell's own pointer, so no address but the cell's is
/// baked into code.
#[repr(C)]
pub struct FuelCell {
    /// What is left of the last draw (or of a fixed allowance).
    pub left: u64,
    refill: unsafe extern "C" fn(*mut FuelCell) -> u64,
    remaining: unsafe extern "C" fn(*const FuelCell) -> i64,
    /// The chain the next draw comes from; `None` for a fixed allowance (or once the chain proved
    /// unbounded). Locked by a refill, so the vCPUs of a domain sharing the cell draw one at a time.
    src: Mutex<Option<Arc<dyn BudgetNode>>>,
}

impl FuelCell {
    /// A fixed allowance of `n`, never refilled.
    pub fn fixed(n: u64) -> Box<FuelCell> {
        Self::with(n, None)
    }

    /// A cell that draws from `src`, none drawn yet: the first safepoint draws.
    pub fn drawing(src: Arc<dyn BudgetNode>) -> Box<FuelCell> {
        Self::with(0, Some(src))
    }

    /// [`Self::drawing`] when `src`'s chain is bounded; `None` when every level is unbounded — code
    /// with nothing to meter compiles without fuel checks.
    pub fn metering(src: Arc<dyn BudgetNode>) -> Option<Box<FuelCell>> {
        (src.room() >= 0).then(|| Self::drawing(src))
    }

    fn with(left: u64, src: Option<Arc<dyn BudgetNode>>) -> Box<FuelCell> {
        Box::new(FuelCell {
            left,
            refill: fuel_refill,
            remaining: fuel_remaining,
            src: Mutex::new(src),
        })
    }

    /// What this cell's vCPUs can still burn: `left` plus the chain's room, `u64::MAX` unmetered.
    pub fn can_burn(&self) -> u64 {
        let left = unsafe { core::ptr::read_volatile(&self.left) };
        match &*self.src.lock().unwrap_or_else(|e| e.into_inner()) {
            None => left,
            Some(s) => match s.room() {
                r if r < 0 => u64::MAX,
                r => left.saturating_add(r as u64),
            },
        }
    }
}

impl Drop for FuelCell {
    fn drop(&mut self) {
        if let Some(s) = self.src.get_mut().unwrap_or_else(|e| e.into_inner()) {
            s.give_back(self.left);
        }
    }
}

/// A spent cell's refill, called by compiled code: the new `left` (at least 1), or `0` when the chain
/// (or a fixed allowance) is spent — the caller then traps `OutOfFuel`. A domain's vCPUs share one
/// cell, so a refill another vCPU made while this one waited for the lock is kept, not drawn again.
///
/// # Safety
/// `cell` is a live [`FuelCell`].
unsafe extern "C" fn fuel_refill(cell: *mut FuelCell) -> u64 {
    // Only the `src` field is borrowed: compiled code writes `left` through the raw cell pointer.
    let mut src = (*cell).src.lock().unwrap_or_else(|e| e.into_inner());
    let left = core::ptr::addr_of_mut!((*cell).left);
    if core::ptr::read_volatile(left) > 0 {
        return core::ptr::read_volatile(left);
    }
    let drawn = match src.as_ref().map(|s| s.draw()) {
        None | Some(Some(0)) => return 0,
        Some(None) => {
            *src = None; // every level unbounded: nothing left to meter
            u64::MAX
        }
        Some(Some(n)) => n,
    };
    core::ptr::write_volatile(left, drawn);
    drawn
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

const _: () = {
    assert!(core::mem::offset_of!(FuelCell, left) == 0);
    assert!(core::mem::offset_of!(FuelCell, refill) == REFILL_OFF as usize);
    assert!(core::mem::offset_of!(FuelCell, remaining) == REMAINING_OFF as usize);
};
