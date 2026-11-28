//! The passive, engine-agnostic runtime kernel (`#3187` / `#3194`).
//!
//! `temen-kernel` sits below every execution engine (`temen-bytecode`, `temen-jit`,
//! `temen-wasm-jit`, `OpOracle`) and above `temen-ir` only. It is `#![forbid(unsafe_code)]`,
//! wasm32/wasm64-clean, Miri-testable, and Loom-verified from commit 1.
//!
//! ## Six orthogonal primitives
//! 1. **Domain** ([`Domain`], [`DomainControl`]) — one isolation boundary, one window, one
//!    lifetime (Invariant 6), one shared safepoint poll word (`poll_flags`), and one
//!    `multi_carrier` fast-path latch (Decision 21).
//! 2. **Context** ([`Context`], [`ContextKind`], [`ContextState`]) — uniform runnable stack
//!    for `Root`, `Thread`, `GuestFiber`, and `ServeHandler`.
//! 3. **Carrier** ([`CarrierId`], [`FuelCell`]) — execution vehicle (OS thread or cooperative
//!    pump) that owns active fuel metering (Decision 30).
//! 4. **WaitIndex** ([`WaitIndex`], [`WaitKey`], [`WakeReason`], [`ParkDecision`]) — unified
//!    waiter table with lock-free per-stripe futex counts (`#2989`), last-sleeper deadlock settle
//!    (`#2967`), and non-blocking `GuestFiber` `YieldFiber` hand-off (`FIBER_PARKED`, status 2).
//! 5. **Budget** ([`BudgetTree`], [`BudgetId`], [`Limits`]) — 3-dimension durable ceiling tree
//!    (`mem`, `fuel`, `tasks`) plus D66 parallelism `lanes` (Invariant 3).
//! 6. **LifecycleHooks** ([`LifecycleHooks`]) — 4 OS-neutral hooks replacing personality-specific
//!    substrate state (Invariant 4 / 12).

#![forbid(unsafe_code)]

pub mod budget;
pub mod domain;
pub mod sync;
pub mod types;
pub mod wait;

#[cfg(all(test, loom))]
mod loom_tests;

pub use budget::{BudgetId, BudgetSnapshot, BudgetTree, Limits};
pub use domain::{
    pack_fiber_handle, unpack_fiber_handle, ContextMeta, Domain, DomainControl, FiberPoll,
    FiberResumeClaim, FiberSlotState, HandleSlot, NoopLifecycleHooks,
};
pub use types::{
    CallId, CapOutcome, CarrierId, ContextExit, ContextId, ContextKind, ContextState, DomainId,
    FuelCell, KernelOps, LifecycleHooks, MemSpaceId, ParkDecision, PollAction, PollFlags, Step,
    Trap, TypeId, WaitKey, WaitableId, WakeReason, YieldReason, ERRNO_EBADF, FIBER_PARKED,
    FIBER_RETURNED, FIBER_SUSPENDED, MAX_CAP_ARGS, WAIT_NOT_EQUAL, WAIT_OK, WAIT_TIMED_OUT,
};
pub use wait::{WaitIndex, WaitSlot, WaiterSpec, STRIPES};
