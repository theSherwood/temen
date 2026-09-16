//! Per-OS-thread **durable shadow-region base** register (`durable.shadow_base`, DURABILITY.md §12.8
//! Phase 4 Slice A.5) for the JIT.
//!
//! One `u64` window byte offset per OS thread (a vCPU): the base of the shadow region the running
//! durable context spills into. The durable transform reads it (via the `durable.shadow_base` IR op)
//! to address *this* context's per-context shadow-SP word, so concurrent vCPUs each unwind into their
//! own region with **no shared SP word** — retiring the single `SHADOW_SP_OFF` word (and its
//! `workers > 1` lock).
//!
//! Like [`crate::vcpu_tls`] it is a baked thunk over a thread-local — substrate-independent and unable
//! to fault — but **runtime-private**: the runtime seeds it (per dispatch / per child) and there is no
//! guest write thunk, so a guest cannot redirect its own shadow stack (unlike the guest-overwritable
//! `vcpu.tls`). Seeded at vCPU entry to `shadow_region_base(ctx)` (root = `ShadowArena::region_base(0)`).

use std::cell::Cell;

/// Default: the root context's region base (`ShadowArena::region_base(0)`). The runtime
/// re-seeds at every root entry / inline child / fiber switch before any instrumented code runs, so
/// this default is only a never-stale fallback.
const ROOT_SHADOW_BASE: u64 = temen_ir::durable_abi::ShadowArena::LEGACY.region_base(0);

thread_local! {
    /// This OS thread's (vCPU's) active durable shadow-SP **word address** — the base of the region
    /// the running context spills into (§12.8 4A.5). [`seed`] resets it per root entry / inline child /
    /// fiber switch so a reused worker thread can't leak a prior run's value.
    static DURABLE_SHADOW_BASE: Cell<u64> = const { Cell::new(ROOT_SHADOW_BASE) };
}

/// Seed/reset the current OS thread's durable shadow-SP word address. Called when the runtime makes a
/// context active: the root entry, each inline child, and both edges of a fiber resume swap.
///
/// `#[inline(never)]` is load-bearing correctness (#1466), for the same reason as
/// [`crate::fiber_rt`]'s `current`: the **exit** edge of the resume swap runs after a stack switch
/// (`fiber_rt::fiber_resume`, which is itself on a fiber stack whenever a fiber resumes a fiber), so
/// an inlined copy would let LLVM reuse the thread-pointer it resolved for the *entry* edge and seed
/// the suspending thread's slot instead. That miswrite is silent — unlike `CURRENT_RT` there is no
/// null to trip over — so this held only by the accident of not being inlined.
#[inline(never)]
pub(crate) fn seed(base: u64) {
    DURABLE_SHADOW_BASE.with(|c| c.set(base));
}

/// `durable.shadow_base` thunk — the current context's shadow-region base. A pure thread-local read;
/// it cannot fault, so it takes no window/trap context (unlike the `call.cap`/`gc.roots` thunks).
///
/// `#[inline(never)]`: same reason as [`seed`] (#1466). It is also handed to JIT-emitted code as a
/// raw function pointer, so it must keep a callable body regardless.
#[inline(never)]
pub(crate) extern "C" fn get() -> u64 {
    DURABLE_SHADOW_BASE.with(|c| c.get())
}
