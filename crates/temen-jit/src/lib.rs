//! Cranelift JIT backend (`DESIGN.md` §9, §18).
//!
//! Cranelift is the chosen codegen **by design** (§1): it is the security mechanism,
//! not a liability — we share Wasmtime's most security-critical component, so the
//! escape-TCB *delta* we own is just this CLIF generation plus the §4 masking
//! lowering. Correctness is established by **differential testing against the
//! reference interpreter** (§18, invariants I1/I4), the oracle in `temen-interp`.
//!
//! ## Status: the whole IR
//! Lowers every IR op: `i32`/`i64`/`f32`/`f64` consts, all integer and float
//! arithmetic/bitwise/shift/rotate/compare ops (incl. trapping `div`/`rem`),
//! `eqz`/`select`/`clz`/`ctz`/`popcnt`, every conversion (extend/wrap/demote/promote/
//! reinterpret, int↔float, saturating **and** trapping `trunc`), **loads/stores with
//! confinement masking** (I1), **indirect calls with function-table dispatch** (I2),
//! **`call.cap` through a host thunk** (§9, see [`CapThunk`]), and every terminator —
//! `br`/`br_if`/`br_table`/`return`/`unreachable` plus direct and indirect tail calls.
//!
//! ## Traps ([`JitOutcome`])
//! A trap is terminal (the guest domain is killed, §5 detect-and-kill), but the host
//! must *observe* it rather than crash. Arithmetic traps (div/rem-by-zero, trapping
//! `trunc`, `unreachable`, indirect-call type) are detected with **explicit checks that
//! store a [`TrapKind`] code and `return` early**; **memory faults** use a real
//! **`PROT_NONE` guard page + a SIGSEGV/SIGBUS handler** that unwinds the call as
//! [`TrapKind::MemoryFault`] (see `mem.rs` / `trap_shim.c`, unix). Either way the
//! *observable semantics* are identical, so the differential harness checks the JIT and
//! interpreter agree on traps too.
//!
//! ## The confinement lowering (§4, invariant I1, trap-confinement)
//! Every access **bounds-checks the final effective address** against the reserved
//! window (`addr` vs the compile-time constant `reserved − offset − width` — the
//! predicate of [`temen_mask::Window::checked_reserved`], the isolated, separately-fuzzed
//! spec) and redirects an out-of-bounds one, branchlessly (`select_spectre_guard`), to
//! the window's trailing guard page, where the access itself faults as `MemoryFault`
//! (§4, D63). The redirect is a speculation barrier, so a misspeculated OOB access lands
//! on the guard too — the Spectre-v1 hardening. Every window starts at 0 of its own
//! reservation (§14), so the address is added to the window base unchanged.
//!
//! **Check/clamp elision (§1a guard-when-bounded, D36–D38).** A conservative per-block
//! upper-bound analysis ([`ub_of`]) proves some effective addresses are *already* `< size`;
//! for those the check and clamp are both dropped ([`in_window`] / `mask_addr`'s `elide`) —
//! closing part of the gap to wasm32's free guard-page accesses (the proof is over data
//! dependencies, so it holds speculatively too). This is the subset of guard-when-bounded
//! that needs no guard region (it only elides *provably in-window* accesses, never relying
//! on a fault); the full wasm32-style large-guard version awaits real guard pages (§5). A
//! wrong bound would be a confinement escape, so the analysis is upper-bound-only
//! (unknown ⇒ check+clamp) and the elision is differentially guarded by the escape-oracle
//! (final-memory equality, §18).
//!
//! ## Indirect-call dispatch (§3c, invariant I2)
//! `call.dyn` masks the guest index into a host-owned, power-of-two-padded
//! function table, checks the slot's `type_id` against the call's signature (trap on
//! mismatch — a forged/wrong-type index is inert), and calls the slot's code pointer.
//!
//! ## Calling convention
//! All functions share a natural CLIF ABI `(mem_base, fn_table_base, params…) ->
//! (results…)` (the `tail` call conv, so `return_call` works); the two context
//! pointers are threaded through every call. The entry is wrapped in a fixed
//! buffer-ABI trampoline `fn(args: *const i64, results: *mut i64, mem_base: *mut u8,
//! fn_table_base: *const FnEntry)` so [`compile_and_run`] can call any arity from Rust.

use core::sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{
    F32, F32X4, F64, F64X2, I16, I16X8, I32, I32X4, I64, I64X2, I8, I8X16,
};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp as ClifRmwOp, BlockArg, BlockCall, ConstantData, Endianness, FuncRef,
    Function, InstBuilder, JumpTableData, MemFlags, Signature, SourceLoc, StackSlotData,
    StackSlotKind, Type, UserFuncName, Value, ValueLabel,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::CompiledCode;
use cranelift_codegen::LabelValueLoc;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, ModuleReloc};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use temen_ir::bounds::{in_window, ub_at, ub_of, UB_TOP};
use temen_ir::cap_id;
use temen_ir::{
    AtomicRmwOp, BinOp, Block, CastOp, ConvOp, Data, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncIdx,
    FuncType, Inst, IntTy, IntUnOp, LoadOp, Module as IrModule, StoreOp, Terminator, VBitBinOp,
    VCvtOp, VFCmpOp, VFloatBinOp, VFloatUnOp, VICmpOp, VIntBinOp, VIntUnOp, VNarrowOp, VPMinMaxOp,
    VSatBinOp, VShape, VShiftOp, ValType,
};

mod dwarf;
pub mod gdb; // W5 JIT/DWARF tier (Stage 2c): in-memory ELF wrapper + the GDB JIT registration interface
mod mem; // guest-window allocation + the §4/§5 guard-page / detect-and-kill handler // W5 JIT/DWARF tier: synthesize DWARF from the Stage 1 machine-address → source map

// JIT fiber runtime (§12): host-side fiber table + `extern "C"` thunks for `cont.new`/`resume`/
// `suspend`, on top of the `temen-fiber` stack-switch substrate. Available where `temen_fiber::supported()`.
#[cfg(fiber_rt)]
mod fiber_rt;

// JIT `setjmp`/`longjmp` runtime (LLVM.md §"JIT `longjmp`", Option B): a host-side `jmp_buf` table +
// `extern "C"` slot thunks, with libc `_setjmp`/`_longjmp` called inline from JITted code. Unix-only
// among the fiber_rt targets (`setjmp_rt` cfg); elsewhere the JIT keeps bailing `Unsupported` and the
// interpreters cover it.
#[cfg(setjmp_rt)]
mod setjmp_rt;

// 1:1 OS-thread executor for `thread.spawn`/`thread.join` + the `wait`/`notify` futex (§12): the VM
// exposes these as *primitives*, not a scheduler — a spawned vCPU is one real OS thread; any M:N model
// is built by the guest runtime over these + `cont.*` (D22: no built-in scheduler). The futex core is
// loom-verified. Available where `temen_fiber::supported()` (x86-64 unix).
#[cfg(fiber_rt)]
mod os_thread_rt;
// PROCESS.md S1b/S1c: the canonical-key futex region registry — `temen-run` records a §13 `map`'s pages
// (so the JIT futex thunks canonicalize `Backed` addresses) and purges them at teardown. Real-runtime
// only; the loom futex model has no regions (`futex_key_of` is `Anon`-only there).
#[cfg(not(loom))]
pub use os_thread_rt::{
    outside_waker, park_host_call, region_canon_forget_window, region_canon_lookup,
    region_canon_new_owner, region_canon_record, wait_outside, wake_host_parks, wake_host_parks_in,
    FreezeWatch, HostPark, Parks,
};

/// F3 (FIBER_PARK.md): whether a guest fiber is the current continuation on this OS thread —
/// the host thunk's fiber-context probe (the futex thunk's `current_fiber_slot`, public face).
/// `false` for the root computation, so a root-context punt keeps its inline/blocking wait.
pub fn fiber_active() -> bool {
    fiber_rt::current_fiber_slot().is_some()
}

/// #1166 — whether this OS thread runs a `thread.spawn`ed vCPU, which only a concurrent guest has.
/// A host thunk that takes its host state as a raw `&mut`, and so serves one caller at a time, must
/// refuse a call made here: a concurrent guest's calls arrive from several threads at once.
pub fn on_spawned_vcpu() -> bool {
    os_thread_rt::on_spawned_vcpu()
}

/// #1768 — whether compiled code on this OS thread is running beneath a **host frame that
/// re-entered it** over a live window ([`CompiledModule::invoke_extra`]: a `Jit.invoke`d unit, a
/// serve handler). The embedder's fork gate: a fork is reified by unwinding the caller's instrumented
/// frames into the window's shadow stack, and a host frame between them and the run's entry is not a
/// frame that unwinding can save or a rewind rebuild — so a fork made beneath one is refused.
pub fn reentered() -> bool {
    reentry::DEPTH.with(|d| d.get() != 0)
}

/// The per-thread re-entry depth behind [`reentered`], held up for exactly the span of a re-entry.
mod reentry {
    use core::cell::Cell;

    thread_local! {
        pub(super) static DEPTH: Cell<u32> = const { Cell::new(0) };
    }

    /// One re-entry in progress on this thread; dropping it ends it (so a caught fault's return,
    /// which drops it too, cannot leave the count raised).
    pub(super) struct Guard;

    pub(super) fn enter() -> Guard {
        DEPTH.with(|d| d.set(d.get() + 1));
        Guard
    }

    impl Drop for Guard {
        fn drop(&mut self) {
            DEPTH.with(|d| d.set(d.get() - 1));
        }
    }
}

/// F3 (FIBER_PARK.md) — park the current fiber once on a host call's answer from outside the run
/// (a punted job): unwinds `FIBER_PARKED` to the resumer (the suspend the guest didn't write) and
/// returns when a `cont.resume` polls this fiber again. The host thunk's event-park seam — exactly
/// what the futex thunk's wait loop does between polls, minus the futex cell. A §14 child's task
/// parked here (#2424) is not counted blocked, as an OS vCPU in [`wait_outside`] is not: the answer
/// may yet end a peer's wait. [`outside_waker`] re-offers it once the answer is in.
///
/// # Safety
/// Must be called from a host thunk on a thread currently running a guest fiber
/// ([`fiber_active`] is `true`) — the same contract as the futex thunk's event park.
pub unsafe fn fiber_park_outside() {
    let slot = fiber_rt::current_fiber_slot().expect("fiber_park_outside outside a fiber");
    fiber_rt::fiber_event_park(&slot, None, fiber_rt::ParkOn::Outside);
}

// §12 per-vCPU TLS register (`vcpu.tls.get`/`set`): one i64 per OS thread (a vCPU). Always compiled
// (substrate-independent), so a plain non-fiber root has a TLS word too.
mod vcpu_tls;
pub use vcpu_tls::serve_handlers;

// #1768 — the per-instance context compiled code reaches through its threaded context pointer.
mod vmctx;
pub use vmctx::{InstanceAddrs, VmCtx};
mod fuel;
pub use fuel::{BudgetNode, FuelCell};
// temen#2452 — compiled code kept on disk, so a later process defines it instead of compiling.
mod code_dir;
pub use code_dir::CodeDir;
use code_dir::{Codegen, EmittedCode};

// §12.8 4A.5 durable-runtime-internal per-OS-thread shadow-region base (`durable.shadow_base`): the
// base of the region the running durable context spills into, so concurrent vCPUs each have their own
// per-context shadow-SP word (retiring the shared `SHADOW_SP_OFF`). Runtime-private (no guest setter).
mod durable_shadow;

// Migratable-fiber ownership protocol (D57 / DESIGN.md §23): the loom-verified single-owner atomic
// state machine that guarantees a stolen fiber is resumed by exactly one thread — the gating safety
// core of stackful work-stealing, proven in isolation before the runtime integration + cross-thread
// resume land. Pure atomics; not yet wired into the live runtime.
#[cfg(fiber_rt)]
mod fiber_registry;

// §14 nesting runtime: the host side of the `Instantiator` capability for the JIT — a spawn compiles
// the child's module for a window of its own (nesting cost paid at setup) and runs it on the child
// executor; `join` returns its result. Available where children can run (`fiber_rt`).
/// D66 — the child-domain executor: detached §14 children as migrating tasks over a lane-bounded
/// worker pool (DESIGN.md §23).
#[cfg(fiber_rt)]
mod child_exec;
#[cfg(fiber_rt)]
mod instantiator_rt;

// The windows placeholder-window commit and protect primitives, reused by `temen-run`'s Memory-cap
// backend (it commits/grows tail pages of this same window; a plain `VirtualAlloc(MEM_COMMIT)` cannot
// commit a placeholder reservation, nor one `VirtualProtect` span a §13 view). See `mem::win_commit_rw`.
#[cfg(windows)]
pub use mem::{win_commit_rw, win_protect};

// The §5 W3 frame-pointer walk, under its own fault recovery — the `trap_walk` fuzz target's entry
// (#1487). See `mem::walk_trap_frame_chain`.
#[cfg(all(unix, fiber_rt))]
pub use mem::walk_trap_frame_chain;

/// Whether this build's JIT lowers the §12 fiber/thread/futex ops (`cont.*`, `thread.*`,
/// `atomic.wait`/`notify`) instead of bailing [`JitError::Unsupported`]. True on the targets where
/// `temen-fiber` provides a real stack switch — the `fiber_rt` cfg derived in `build.rs`, kept in
/// lockstep with `temen_fiber::supported()`. Exposed so tests can assert the platform gating against the
/// single source of truth rather than re-deriving the target set.
pub const fn fiber_supported() -> bool {
    cfg!(fiber_rt)
}

/// Largest window the reference JIT will back with a host allocation. Real deployments
/// reserve a huge guard-paged virtual range (§4); for the differential harness we map
/// `1 << size_log2` bytes (+ a guard page on unix), so cap it. This bounds the harness's
/// backed allocation only — it is **not** a confinement invariant (the security mask,
/// `child_size − 1`, holds at any power-of-two size), and the browser wasm-JIT (whose linear
/// memory grows on demand) has no such cap. Sized (owner-approved 2026-08-25) so the op-13
/// nimony front-end chain fits: `nimsem`/`hexer` semchecking the system module peak in
/// (128, 256] MiB, needing a 256 MiB child carve and thus a 512 MiB parent window. `mmap` is
/// lazy (RSS follows touched pages, not the reservation), so the VA bump is cheap and real fuzz
/// seeds — far below even the old cap — are unaffected.
///
/// **Re-measured 2026-09-21 (#1591, #1603): 29 → 31.** The premise above was accurate when written,
/// then two of our own defects inflated the workload past it. `nifler_shim.c`'s `mmap` was not
/// page-aligned, so Nim's allocator — which recovers a chunk header with `pageAddr(p) = p & ~0xFFF`
/// — crashed in `rawDealloc`; that was worked around with `-d:useMalloc`, which routes every object
/// to the on-ramp's `synth_malloc`, whose `free` is a no-op. The guest's peak then tracked total
/// allocation *churn* rather than its live set: 2007 MiB. With the alignment fixed and the flag
/// dropped it is **666 MiB** for byte-identical output — still 2.6x the original (128, 256], so the
/// chain needs a 1 GiB carve and hence a 2 GiB parent window.
///
/// Raising the cap is safe in the sense that matters: it is **not** a confinement boundary. Masking
/// enforces that (DESIGN §4 / INVARIANTS #2); this is a guard rail so a fuzzed or malformed module
/// cannot ask the reference JIT for an absurd allocation. The cost is address space, not RAM — the
/// mapping is lazy, so RSS follows touched pages, and a guest that needs 666 MiB needs it on every
/// engine. `MAX_JIT_RESERVED_LOG2` has allowed 1 TiB of reserved VA all along.
///
/// (Recorded honestly because the line above says "owner-approved": the *figure* went stale, the
/// decision behind it did not, and what actually changed was on our side of the boundary.)
pub const MAX_JIT_WINDOW_LOG2: u8 = 31; // 2 GiB (the backed `mapped` extent)

/// Largest **reserved** virtual range (the mask domain) the reference JIT will `mmap` per
/// window. The reservation is `PROT_NONE` + `MAP_NORESERVE`, so this is virtual address space,
/// not committed memory. `2^40` was the default until #2305 lowered it to 2^37 (`DESIGN.md` §4),
/// so a window frozen under it still thaws at the reservation its artifact records. A real host
/// chooses this per its VA budget; the reference just caps it so a fuzzed/oversized request
/// can't ask for an absurd reservation.
const MAX_JIT_RESERVED_LOG2: u8 = 40; // 1 TiB of reserved VA (lazy)

/// Escape-oracle snapshot span (the `_with_host` capture): byte-compare the low `SNAP_CAP` bytes of
/// the window across interp + JIT, *including* reserved-tail pages the guest grew via the Memory cap
/// (not just the backed prefix). Bounds the per-seed snapshot cost while covering a generous growth
/// region. **Must match `temen_interp`'s `SNAP_CAP`** so both backends snapshot the same span.
const SNAP_CAP: usize = 1 << 18; // 256 KiB

/// A function-table entry (§3c `FnEntry`): host-owned, guest-unwritable. `type_id`
/// identifies the signature (distinct-`FuncType` index); `code` is the finalized
/// function address. `call.dyn` masks the guest index into the table, checks
/// `type_id` (offset 0), then calls `code` (offset 8) — confinement at the use site (invariant I2).
///
/// The fields are **atomic** so a guest-driven `install`/`uninstall` (a host-side write) is sound
/// against a *concurrent* `call.dyn` from another thread (DESIGN.md §22 threaded install). The
/// `#[repr(C)]` layout (`type_id` @0, `code` @8) is exactly what [`indirect_dispatch`] bakes its
/// loads against, unchanged. Two distinct guarantees, both platform-uniform:
/// - **Visibility** (a synchronized reader sees a complete install) rides the **guest's own**
///   acquire/release — the worker only dispatches a slot it learned about through a guest atomic, so
///   the install's stores are ordered-before the worker's dispatch loads via that synchronization,
///   on weakly-ordered targets too. The dispatch's own plain loads need no acquire. `publish` still
///   stores `code` before `type_id` (release-ordered) as belt-and-suspenders for a reader that
///   happens to synchronize on `type_id` directly.
/// - **No torn pointer** (a *racy* reader never sees a half-written `code`): each field is a single
///   atomic word, so a concurrent dispatch reads either the old or the new value — never a wild
///   pointer. A racy reinstall is the guest's own bug and is contained (trap), never an escape.
#[repr(C)]
pub(crate) struct FnEntry {
    type_id: AtomicU32,
    _pad: u32,
    code: AtomicU64,
}

impl FnEntry {
    /// A real/installed entry.
    pub(crate) fn new(type_id: u32, code: u64) -> FnEntry {
        FnEntry {
            type_id: AtomicU32::new(type_id),
            _pad: 0,
            code: AtomicU64::new(code),
        }
    }
    /// A trapping padding entry (`call.dyn` here is inert).
    pub(crate) fn padding() -> FnEntry {
        FnEntry::new(PADDING_TYPE_ID, 0)
    }
    /// The slot's signature id (the host-side bookkeeping read; the hot dispatch read is in
    /// generated code). `Acquire` pairs with [`Self::publish`]'s release of `type_id`.
    pub(crate) fn type_id(&self) -> u32 {
        self.type_id.load(Ordering::Acquire)
    }
    /// The slot's code pointer (host-side bookkeeping read).
    pub(crate) fn code(&self) -> u64 {
        self.code.load(Ordering::Acquire)
    }
    /// **Publish** an installed function: `code` first, then `type_id` (release-ordered), so a
    /// concurrent reader that sees the new `type_id` also sees the new `code`.
    pub(crate) fn publish(&self, type_id: u32, code: u64) {
        self.code.store(code, Ordering::Release);
        self.type_id.store(type_id, Ordering::Release);
    }
    /// **Clear** to trapping padding: `type_id` (the ready field) first so a new dispatch traps
    /// promptly, then zero `code`.
    pub(crate) fn clear(&self) {
        self.type_id.store(PADDING_TYPE_ID, Ordering::Release);
        self.code.store(0, Ordering::Relaxed);
    }
}

/// `type_id` for a table slot that holds no function (the power-of-two padding) — it
/// matches no call site, so a forged index landing here always traps.
const PADDING_TYPE_ID: u32 = u32::MAX;
/// `type_id` a `call.dyn` uses when its signature matches no function in the
/// module: distinct from every real id and from the padding sentinel, so it always
/// traps (no function could satisfy it).
const NO_MATCH_TYPE_ID: u32 = u32::MAX - 1;

/// The distinct function signatures in a module; a function's (or call site's) type id
/// is an index into this list — structural equality, matching the interpreter's check.
fn distinct_types(funcs: &[Func]) -> Vec<FuncType> {
    let mut out: Vec<FuncType> = Vec::new();
    for f in funcs {
        let ft = FuncType {
            params: f.params.clone(),
            results: f.results.clone(),
        };
        if !out.contains(&ft) {
            out.push(ft);
        }
    }
    out
}

/// Resolve an interned call type index (#922) into its [`FuncType`]. The call variants
/// (`CallIndirect`, `CapCall`, …) carry a `u32` index into the unit's type section; a verified
/// module always resolves to a `TypeEntry::Func`. A miss returns a static empty signature so
/// callers stay total (the verifier already rejects a dangling / non-func index up front).
fn sig_of(type_section: &[temen_ir::TypeEntry], t: u32) -> &FuncType {
    static EMPTY: FuncType = FuncType {
        params: Vec::new(),
        results: Vec::new(),
    };
    match type_section.get(t as usize) {
        Some(temen_ir::TypeEntry::Func(ft)) => ft,
        _ => &EMPTY,
    }
}

/// The type id of `ty` among `distinct`, or [`NO_MATCH_TYPE_ID`] if absent.
fn type_id_of(distinct: &[FuncType], ty: &FuncType) -> u32 {
    distinct
        .iter()
        .position(|t| t == ty)
        .map(|i| i as u32)
        .unwrap_or(NO_MATCH_TYPE_ID)
}

/// Intern `ty` into the **append-only** type-id registry (DESIGN.md §22: the per-domain id space
/// made incremental), returning its stable id. Soundness of the `call.dyn` dispatch check
/// reduces to this map being an *injection*, *stable over time* (an id never remaps — appends
/// only), and *total over participants* (every signature that can appear at a call site or in
/// a table slot is interned before any code referencing it is lowered) — then id-equality
/// coincides exactly with the interpreter's structural equality. The registry is consulted
/// only at compile/install time (inside a synchronous `call.cap`, guest suspended); compiled
/// code holds ids as immediates and never reads it at runtime.
fn intern_type(distinct: &mut Vec<FuncType>, ty: &FuncType) -> Result<u32, JitError> {
    if let Some(i) = distinct.iter().position(|t| t == ty) {
        return Ok(i as u32);
    }
    // Defensive: never collide with the `NO_MATCH_TYPE_ID` / `PADDING_TYPE_ID` sentinels
    // (unreachable in practice — it would take ~2^32 distinct signatures).
    if distinct.len() as u64 >= NO_MATCH_TYPE_ID as u64 {
        return Err(JitError::Unsupported("type-id registry full"));
    }
    distinct.push(ty.clone());
    Ok((distinct.len() - 1) as u32)
}

/// Intern every signature `funcs` can put into play for table dispatch: each function's own
/// signature (what a table slot holding it would carry) and every `call.dyn` /
/// `return_call.dyn` **site** signature (what the check compares against). Site
/// signatures matter: an id is baked into the call site as an immediate when the unit is
/// lowered, so a site whose signature is only defined by a *later* unit must already hold the
/// real id — interning up front keeps id-equality ≡ structural equality across units instead
/// of freezing a site to the always-trapping `NO_MATCH_TYPE_ID`.
fn intern_unit_sigs(
    distinct: &mut Vec<FuncType>,
    funcs: &[Func],
    type_section: &[temen_ir::TypeEntry],
) -> Result<(), JitError> {
    for f in funcs {
        intern_type(
            distinct,
            &FuncType {
                params: f.params.clone(),
                results: f.results.clone(),
            },
        )?;
        for b in &f.blocks {
            for i in &b.insts {
                if let Inst::CallIndirect { ty, .. } = i {
                    intern_type(distinct, sig_of(type_section, *ty))?;
                }
            }
            if let Terminator::ReturnCallIndirect { ty, .. } = &b.term {
                intern_type(distinct, sig_of(type_section, *ty))?;
            }
        }
    }
    Ok(())
}

/// Why the JIT could not compile (or run) a function. The integer slice rejects
/// anything it does not yet lower with [`JitError::Unsupported`]; the differential
/// harness treats that as "skip", not "fail".
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JitError {
    /// An instruction/terminator the current slice does not lower yet.
    Unsupported(&'static str),
    /// Structurally invalid (a verified module never hits this; defensive only).
    Malformed,
    /// Cranelift rejected the generated CLIF or failed to compile it.
    Backend(String),
}

/// What a JIT'd run produced: either the result slots, or a **trap** with a kind code.
///
/// A trap is terminal (the guest domain is killed, §5 "detect-and-kill") — this just
/// reports it to the host instead of aborting the process. The reference JIT detects
/// traps with **explicit checks that store a code and return early**; a production JIT
/// would instead take a hardware fault (guard page / `#DE`) caught by a signal handler
/// (§5). The *observable semantics* — which inputs trap, and the kind — are identical,
/// which is what the differential oracle checks.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum JitOutcome {
    Returned(Vec<i64>),
    Trapped(TrapKind),
    /// The guest invoked the `Exit` capability with this code (§3e) — terminal, but not
    /// an error.
    Exited(i32),
    /// #1768 — the host thunk unwound the run on the host's behalf ([`HOST_UNWIND_CODE`]): the
    /// guest's continuation is discarded, and the host's own state says what it asked for and what
    /// runs next (an `execve` image-replace is the one user). Only a host that stores the code sees it.
    HostUnwound,
}

/// Per-page protection to re-establish on a guest window before a run — the durable-restore
/// step (DURABILITY.md §12.3). One entry per [`DURABLE_SNAPSHOT_PAGE`]-byte page of the image the
/// window is seeded with, and one per [`TailPage`] it holds past that (#2427): a `Rw` entry in the
/// backed prefix keeps the default, and past it every entry is a page the window holds. Lets a
/// thawed guest fault on a restored `Ro`/`Unmapped` page exactly as the frozen one would,
/// matching the interpreter (`temen-interp`'s `Mem::restore_layout`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WindowProt {
    Rw,
    Ro,
    Unmapped,
}

/// #2427 — a page a durable window holds past the image a thaw seeds it with, as the thaw
/// re-establishes it: its byte offset, its protection, and its bytes (`None` reads zero). A capture
/// carries each page the window holds there, and nothing between them.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct TailPage {
    pub off: u64,
    pub prot: WindowProt,
    pub bytes: Option<Vec<u8>>,
}

/// #2441 — the window a thaw's run starts from (DURABILITY.md §12.3): the image it is seeded with is
/// the whole window, its data segments holding what the guest left in them, and `prots` its page map,
/// one [`WindowProt`] per [`DURABLE_SNAPSHOT_PAGE`] of that image — read-only only where it says,
/// never where a `readonly` segment lies (#2307); then `tail`, each page it holds past the image
/// (#2427). A run without one builds a fresh instance's window, its data segments laid in.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct WindowRestore {
    pub prots: Vec<WindowProt>,
    pub tail: Vec<TailPage>,
}

/// The host-side residue of a fiber the durable freeze driver flattened (DURABILITY.md §12.8) — the
/// JIT mirror of `temen_interp::FrozenFiber`. Its continuation lives in its in-window shadow region;
/// this carries what a thaw must re-seed: the registry slot (= guest handle), entry funcref +
/// data-stack base (to re-enter it), and the flattened shadow-SP extent. A durable **freeze** run
/// returns one per flattened fiber; a **thaw** run is handed them back to re-create the fibers.
/// Every slot of the table rides (#1684): a fresh fiber at its empty frame base, and a free slot as
/// [`FrozenFiber::free`] — the same records as the interpreter's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FrozenFiber {
    pub slot: usize,
    pub func: i32,
    pub sp: i64,
    /// `0` for a free slot: no shadow region lies at offset 0.
    pub shadow_sp: u64,
    /// The slot's generation at freeze (recycling step 2): re-seeded on thaw so a guest handle to a
    /// recycled fiber still resolves. 0 for a non-recycled fiber. Mirrors `temen_interp::FrozenFiber`
    /// (48-bit field — the `i64` handle's generation bits).
    pub generation: u64,
    /// #1538 — the fiber's last `suspend` value was already consumed by a resumer that ran on, so a
    /// thaw claim delivers its argument at the rewound `suspend` instead of re-parking. Mirrors
    /// `temen_interp::FrozenFiber::consumed`.
    pub consumed: bool,
}

impl FrozenFiber {
    /// #1684 — a free slot (its fiber finished): only its generation rides.
    pub fn free(slot: usize, generation: u64) -> FrozenFiber {
        FrozenFiber {
            slot,
            func: 0,
            sp: 0,
            shadow_sp: 0,
            generation,
            consumed: false,
        }
    }

    /// Whether this is a [`FrozenFiber::free`] slot.
    pub fn is_free(&self) -> bool {
        self.shadow_sp == 0
    }
}

/// The host-side residue of a **spawned vCPU** (a `thread.spawn` child) flattened by a multi-vCPU
/// durable freeze (DURABILITY.md §12.8 slice 3.3) — the JIT mirror of `temen_interp::FrozenVCpu`. Its
/// continuation lives in its own in-window shadow region (`shadow_region_base(task)`); this carries
/// what a thaw must re-attach: its task id (= shadow context index + spawn order), entry function +
/// spawn args (to re-enter it), and the flattened shadow-SP extent. A multi-vCPU **freeze** returns
/// one per flattened child; a **thaw** is handed them back to re-create the children.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FrozenVCpu {
    pub task: usize,
    /// The task that **spawned** this child (slice 3.4: nested spawns) — `0` for the root's direct
    /// children, a child's task for a grandchild. Thaw rebuilds the per-parent join tables from this so
    /// a grandchild's reloaded handle resolves in its parent's table. Mirror of `temen_interp`'s field.
    pub parent_task: usize,
    /// The child's handle in its spawner's join table (#1685) — the thaw re-attaches it there.
    pub slot: usize,
    pub func: i32,
    pub args: Vec<i64>,
    pub shadow_sp: u64,
    /// §12.8 4A.5 follow-up A: `Some(result)` for a **completed-but-unjoined** concurrent child — one
    /// that finished before the freeze point, so its `thread.join` result must survive in the snapshot
    /// (the host-side Done cell isn't captured otherwise). The thaw delivers this result into the
    /// spawner's join table **without re-running** the child (no double side effects). `None` for a
    /// normal frozen child (re-spawned + rewound on thaw); `shadow_sp`/`func`/`args` are then inert.
    pub completed_result: Option<i64>,
}

/// The durable snapshot's window-image page granularity (must match `temen-snapshot`'s `PAGE` /
/// `temen-interp`'s `DURABLE_SNAPSHOT_PAGE`): a restored protection map has one entry per this many
/// bytes. Host-page-independent for artifact portability; a 4 KiB codec page sits within one host
/// page, so protecting it protects (at most) its host page — exact on a 4 KiB-page host.
pub const DURABLE_SNAPSHOT_PAGE: usize = 4096;

/// The trap kinds the JIT can raise (a subset of the interpreter's `Trap`), numbered by the one trap
/// wire code ([`temen_ir::trap_code`]) — what the lowered checks and the host thunks store into the
/// trap cell, and what the interpreters' `Trap::code` returns, so `Trap` ↔ `TrapKind` is
/// `Trap::from_code(k.code())` / `TrapKind::from_code(t.code())`, never a hand-written map (#1735).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum TrapKind {
    DivByZero = temen_ir::trap_code::DIV_BY_ZERO as u32,
    IntOverflow = temen_ir::trap_code::INT_OVERFLOW as u32,
    BadConversion = temen_ir::trap_code::BAD_CONVERSION as u32,
    Unreachable = temen_ir::trap_code::UNREACHABLE as u32,
    IndirectCallType = temen_ir::trap_code::INDIRECT_CALL_TYPE as u32,
    /// Forged / closed / wrong-type capability handle (§3c).
    CapFault = temen_ir::trap_code::CAP_FAULT as u32,
    /// A guest memory access faulted into the window's guard region (§4/§5) — caught by
    /// the signal handler and turned into detect-and-kill. The masking lowering confines
    /// every access to `[0, size)`, so in practice this is a width-overrun at the very top
    /// of the window, or (defense-in-depth) a masking/elision bug that the guard caught.
    MemoryFault = temen_ir::trap_code::MEMORY_FAULT as u32,
    /// Forged / out-of-range / already-running / finished fiber handle, a bad fiber-entry funcref, a
    /// `suspend` with no running fiber, or a fiber-bomb (§12). Matches `Trap::FiberFault`.
    FiberFault = temen_ir::trap_code::FIBER_FAULT as u32,
    /// Forged / out-of-range / already-joined thread handle, or a thread-bomb (§12). Matches
    /// `Trap::ThreadFault`.
    ThreadFault = temen_ir::trap_code::THREAD_FAULT as u32,
    /// The host **interrupted** a runaway guest (§5 the fuel/epoch kill-path): a non-terminating
    /// guest is stopped because the host set the interrupt cell (e.g. a watchdog timer). The
    /// lowering polls that cell at loop back-edges and function entries and traps here. Matches the
    /// interpreter's `Trap::OutOfFuel` — both report "the host bounded this run".
    OutOfFuel = temen_ir::trap_code::OUT_OF_FUEL as u32,
    /// A `longjmp` to a `jmp_buf` that was never `setjmp`'d (a stale/forged token) — caught by the
    /// host `setjmp` table's lookup before the (skipped) `_longjmp` (§3b totality). Matches the
    /// interpreters' `Trap::Malformed` for the same condition (LLVM.md §"JIT `longjmp`").
    SetjmpFault = temen_ir::trap_code::MALFORMED as u32,
    /// A guest **control-stack overflow** caught by the software stack-limit check the JIT emits in
    /// each prologue when `feature = "stack-check"` (the arena/software-guard fiber model, which drops
    /// the per-fiber hardware guard page). A function whose frame would grow the native stack past the
    /// running fiber's low bound traps here rather than corrupting an adjacent fiber's stack —
    /// detect-and-kill (§5). NOTE: on the default guard-page backend a *fiber* overflow is NOT a clean
    /// `MemoryFault` — the fault happens at stack exhaustion and the SIGSEGV handler (`SA_ONSTACK` set,
    /// but no `sigaltstack` installed) double-faults on the exhausted stack and kills the process. Only
    /// in-window memory faults (handler has ample stack) surface as `MemoryFault`. The software check
    /// here traps through `vmctx` with no signal, so it is the only path that catches fiber overflow
    /// survivably (see `temen-jit/STACK_GUARD_FLIP.md`, "sigaltstack finding").
    StackOverflow = temen_ir::trap_code::STACK_OVERFLOW as u32,
}

/// Trap-cell code the host thunk stores for an `Exit` (the exit code rides in the high
/// 32 bits of the `i64` cell). Distinct from every [`TrapKind`].
pub const EXIT_CODE: u32 = temen_ir::trap_code::EXIT as u32;

/// #1768 — trap-cell code a host thunk stores to **unwind the whole run on the host's behalf**, with
/// the guest's continuation discarded: an `execve` image-replace, which never returns to its caller.
/// The run unwinds exactly as it does for an `Exit` and ends [`JitOutcome::HostUnwound`]. Distinct
/// from every [`TrapKind`], from [`EXIT_CODE`] and from the internal `DOMAIN_DONE_CODE`.
pub const HOST_UNWIND_CODE: u32 = temen_ir::trap_code::HOST_UNWIND as u32;

/// Trap-cell code `run_inner` stores at **domain teardown** when the root vCPU completed cleanly
/// (owner decision 2026-07-24: "root completion ends the activation; the owner's departure ends the
/// domain" — DESIGN.md §12, ISSUES.md I37). Purely internal, never surfaced: it makes every
/// surviving sibling vCPU's trap-propagation guard unwind (parked waiters are woken to observe it,
/// `os_thread_rt::Domain::begin_teardown`), and the outcome read maps it back to
/// [`JitOutcome::Returned`] — the run result stays the root's own return, uncontaminated. Distinct
/// from every [`TrapKind`] and from [`EXIT_CODE`], mirroring how the freeze unwind keeps its
/// sentinel (the `UNWINDING` state word) out of the trap namespace.
pub(crate) const DOMAIN_DONE_CODE: u32 = temen_ir::trap_code::DOMAIN_DONE as u32;

impl TrapKind {
    /// The kind a trap-cell code names (its low 32 bits), if it is a trap — not `0`, [`EXIT_CODE`] or
    /// a JIT-internal sentinel.
    pub fn from_code(c: u32) -> Option<TrapKind> {
        use temen_ir::trap_code as t;
        Some(match c as i64 {
            t::DIV_BY_ZERO => TrapKind::DivByZero,
            t::INT_OVERFLOW => TrapKind::IntOverflow,
            t::BAD_CONVERSION => TrapKind::BadConversion,
            t::UNREACHABLE => TrapKind::Unreachable,
            t::INDIRECT_CALL_TYPE => TrapKind::IndirectCallType,
            t::CAP_FAULT => TrapKind::CapFault,
            t::MEMORY_FAULT => TrapKind::MemoryFault,
            t::FIBER_FAULT => TrapKind::FiberFault,
            t::THREAD_FAULT => TrapKind::ThreadFault,
            t::OUT_OF_FUEL => TrapKind::OutOfFuel,
            t::MALFORMED => TrapKind::SetjmpFault,
            t::STACK_OVERFLOW => TrapKind::StackOverflow,
            _ => return None,
        })
    }

    /// This kind's trap wire code ([`temen_ir::trap_code`]).
    pub fn code(self) -> i64 {
        self as u32 as i64
    }
}

/// The host callback the JIT invokes for `call.cap` (§9's trampoline). The caller wires
/// it to its capability host; the JIT bakes the function + ctx addresses in as constants
/// and calls it. Scalars cross as `i64` slots (`i32` in the low bits), buffers as the
/// `(ptr, len)` window borrow. A call that traps or exits ends the domain: it raises a
/// [`TrapKind`] code, or `EXIT_CODE | (exit_code << 32)` for an `Exit`, on `vm`
/// ([`VmCtx::raise`]). A call that succeeds leaves the trap cell alone — it holds the domain's
/// end, which a sibling vCPU may have raised during the call (#2499).
///
/// # Safety
/// `ctx` is the caller's host pointer; `args`/`results` point at `n_args`/`n_results`
/// `i64` slots; `[mem_base, mem_base+mem_size)` is the guest window's backed prefix
/// (`mem_base` null if none) and `mem_reserved` is the full reserved mask domain
/// (`>= mem_size`) the guest may `map`-grow into via the Memory cap (§3e/§4); `vm` is the
/// calling instance's live context. All must outlive the call.
pub type CapThunk = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    mem_base: *mut u8,
    mem_size: u64,
    mem_reserved: u64,
    type_id: u32,
    op: u32,
    handle: i32,
    args: *const i64,
    n_args: u64,
    results: *mut i64,
    n_results: u64,
    vm: &VmCtx,
);

/// The **devirtualized `call.cap` fast path** (§9 / D45). A `call.cap` to a statically-known
/// `(type_id, op)` normally goes through the generic [`CapThunk`]: it marshals args through a stack
/// buffer, passes a 12-wide ABI (incl. `n_args`/`n_res`/`type_id`/`op`), and the host dispatches on
/// `(type_id, op)` at runtime. A `FastCapResolver` lets an embedder hand the JIT a **specialized**
/// host function for a given `(type_id, op)`, which the JIT calls **register-to-register** — args and
/// the result in registers, no stack marshalling, no runtime dispatch. Resolution happens once at
/// **compile time** (the JIT calls the resolver during codegen and bakes the returned address); a
/// `null` return falls back to the generic thunk, so an embedder can fast-path only its hot ops.
///
/// The specialized function's ABI is, for a `call.cap` with `n_args` args and **one** result
/// (multi-result ops fall back to the generic thunk):
/// `unsafe extern "C" fn(ctx, mem_base, mem_size, handle: i32, vm: &VmCtx, a0: i64, …, aN: i64) -> i64`
/// — `ctx`/`mem_base`/`mem_size`/`vm` are exactly as in [`CapThunk`]; the `handle` is the
/// resolved capability handle (the fn still does the authority check); each `ai` is the i'th argument
/// widened to its i64 slot (an i32/f32 in the low bits, an f64 bit-pattern); the i64 return is decoded
/// to the result type. A 0-result op returns an ignored 0. The fn raises a trap on `vm` exactly like
/// the generic thunk.
///
/// **The resolver MUST gate on `(n_args, n_res)`**: the JIT builds the call signature from the IR
/// `call.cap`'s arity, so a returned fn whose Rust signature has a *different* arity is a C-ABI
/// mismatch. A frontend may emit a `call.cap` to any `(type_id, op)` with any sig (the verifier checks
/// only `args.len() == sig.params.len()`, not that it matches the host op), so the resolver must return
/// a fn **only** when `(n_args, n_res)` equals that fn's own arity — otherwise `null` (the generic
/// slot-based path handles the odd arity safely). Types never mismatch (every arg is passed as an i64
/// register, the result decoded from i64), so only arity matters.
///
/// # Safety
/// The resolver and every function it returns must honour the ABI above (incl. the arity gate) and
/// stay valid for the run.
pub type FastCapResolver = unsafe extern "C" fn(
    type_id: u32,
    op: u32,
    n_args: u32,
    n_res: u32,
) -> *const core::ffi::c_void;

/// A resolved §14 **`Module` grant**, filled in by a [`ModuleResolver`]: the module, in host-owned
/// storage (the powerbox's module table), and what its grant records of it. The pointer must stay
/// valid for the whole run (the host's table is append-only and the host outlives the run — the
/// same lifetime contract as the `call.cap` ctx). `memory_log2 < 0` means the module declares no
/// memory.
#[repr(C)]
pub struct ResolvedModule {
    /// The granted module: what a spawn of it compiles and seeds — its functions, its type section
    /// (#922: the child's interned `call.dyn` type indices resolve against it), its data segments
    /// and its impl exports (#2247: the child serves their handlers).
    pub module: *const temen_ir::Module,
    pub memory_log2: i32,
    /// The module's declared durable shadow arena (INVARIANTS.md #16) — `ShadowArena::EMPTY` when
    /// it declared none (then a durable child of it is refused, never placed by default).
    pub shadow: temen_ir::durable_abi::ShadowArena,
    /// The granting host attested the module **freezable** (DURABILITY.md §4: it was instrumented). A
    /// durable domain admits only such a child (#1501).
    pub durable: bool,
    /// temen#2168 — the module's content digest, as its grant records it: what names its code to the
    /// run's [`ChildCodeCache`], however it was granted.
    pub digest: [u8; 32],
}

/// The host callback the §14 nesting runtime uses to resolve a guest's **`Module` handle** to the
/// granted module's code/data (so a guest can only instantiate modules it was given). Returns
/// nonzero and fills `out` on success, `0` for a forged/closed/wrong handle. Deliberately a
/// *separate* callback from [`CapThunk`]: module resolution yields host pointers, which must never
/// be reachable from a guest-issued `call.cap` (the generic dispatch on a Module handle is an inert
/// `CapFault`) — only the host-side nesting runtime calls this.
///
/// # Safety
/// `ctx` is the same host pointer as the run's `cap_ctx`; `out` points at a writable
/// [`ResolvedModule`]. The filled views must outlive the run (see [`ResolvedModule`]).
pub type ModuleResolver =
    unsafe extern "C" fn(ctx: *mut core::ffi::c_void, handle: i32, out: *mut ResolvedModule) -> i32;

/// A §14 **granted child** powerbox built host-side (PROCESS.md S2 JIT parity): an opaque child
/// `Host` (`ctx`) holding an `Instantiator` + `AddressSpace` over the child's own window and the
/// parent's re-granted coordinate-free capability, plus the three entry-arg handles the child
/// receives (`Instantiator`, `AddressSpace`, grant). Filled by a [`GrantDetachedChildBuilder`];
/// `ctx` is owned by the caller and must be freed with the paired [`GrantChildReleaser`] after the
/// child runs.
#[repr(C)]
pub struct GrantChild {
    pub ctx: *mut core::ffi::c_void,
    /// CALLS.md 5c.0 — a **second reference** to the same child powerbox, retained by the nursery
    /// so `child_offer` (op 14) can mint a live-impl over the child and callers can reach its
    /// world after the child thread exits (the interp's `child_hosts` retention, JIT twin), and so
    /// `grant` (op 19, #2220) can grant into it while it runs. Null for a builder that does not
    /// share the child `Host` (then ops 14 and 19 answer `-EINVAL`, fail-closed). When non-null,
    /// `ctx` and `retained_ctx` are two counted refs to one shared `Host` (host-side an
    /// `Arc<Mutex<Host>>`): the child thread's ref is released at child exit as before, the
    /// retained one by the nursery at teardown — each exactly once, via the same
    /// [`GrantChildReleaser`].
    pub retained_ctx: *mut core::ffi::c_void,
    pub inst_handle: i32,
    pub as_handle: i32,
    pub grant_handle: i32,
    /// #1296 — the child's `call.dyn` table reservation (`log2` slots; `0` ⇒ the natural
    /// `next_pow2(funcs)`), read off the child powerbox: a child holding a re-granted `Jit`
    /// capability installs units into padding slots of its **own** table, so the builder reports
    /// the reservation its grant carried and the child compiles with that many slots.
    pub jit_table_log2: u8,
    /// temen#2168 — the child holds a re-granted `Jit` capability, with install room or none, so it
    /// may extend its own code (`define_extra`, #1296): it compiles its own rather than run the
    /// run's one compile of the child ([`ChildCodeCache`]), which no instance may extend.
    pub holds_jit: bool,
    /// D66 — the child's **lane chain** coordinates, read off the two powerboxes by the builder:
    /// the child domain's id and lane cap (`-1` = unbounded), and its parent's. The child-domain
    /// executor gates each task's dispatch on every bounded cap along this chain
    /// (`temen_ir::lanes`), so a task runs only while its own lane *and* every enclosing one has
    /// room — the oracle's `lane_chain`, arriving on this backend.
    pub domain: u64,
    pub lane_cap: i64,
    pub parent_domain: u64,
    pub parent_lane_cap: i64,
    /// #2164 — a demand-paged child's access table ([`VmCtx::pages`]), which the builder made for a
    /// spawn that named a pager. Null for any other child. The child's powerbox owns it.
    pub pages: *const AtomicU8,
}

/// The host callback for a **detached spawn** (Instantiator op 17, the record): build a
/// fresh child powerbox from `grants_n` 16-byte records `{name_off, name_len, handle, flags}` at
/// window-relative `grants_ptr` (bounded to `[0, mem_size)`), re-granting each record's copyable
/// handle **under its name** (the child finds them by `self.resolve`), plus what the admission
/// ([`BudgetMemTaker`]) just did: the `(budget, bytes, runs)`
/// lease it charged for the child's window, whose budget becomes the child's `"budget"` and whose
/// runs the child's window is paid for up to (#2268), and `lane`, the lane it reserved, which the
/// child is stamped with. The thunk passes both from one spawn to the next hook, so a second vCPU's
/// spawn between the two cannot hand this child its budget or lane (#1972). Returns nonzero and fills
/// `out` on success; `0` with a `MemoryFault` (out-of-window record/name) or `CapFault` (non-UTF-8
/// name / forged / non-copyable handle) raised on `vm` — the whole spawn fails closed, as on the
/// interpreter.
///
/// #2164 — `pager` is the record's pager export (`u32::MAX` for none), which [`PagerAdmit`] already
/// checked: the child is demand-paged, its faults go to that export of the spawner's program, and
/// `out.pages` is its access table.
///
/// # Safety
/// `ctx` is the run's `cap_ctx` (the parent `Host`); `[mem_base, mem_base+mem_size)` is the parent's
/// mapped window; `out` is writable; `vm` is the spawner's live context. The filled `ctx` is released
/// with the [`GrantChildReleaser`] in the same [`GrantChildHooks`].
pub type GrantDetachedChildBuilder = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    mem_base: *mut u8,
    mem_size: u64,
    grants_ptr: u64,
    grants_n: u64,
    child_size: u64,
    budget: i32,
    bytes: u64,
    runs: u64,
    lane: i64,
    pager: u32,
    out: *mut GrantChild,
    vm: &VmCtx,
) -> i32;

/// §2.2 / #2164 — whether the spawner's program has a pager export `export`, the one a spawn record
/// may name to supply its child's pages (`temen_interp::Host::has_pager_export`): `1`, or `0` and the
/// spawn traps `CapFault`, before any other check of the record, as on the interpreters.
///
/// # Safety
/// `ctx` is the parent host pointer the hook family decodes ([`GrantChildHooks::parent_ctx`]).
pub type PagerAdmit = unsafe extern "C" fn(ctx: *mut core::ffi::c_void, export: u32) -> i32;

/// §2.2 / #2164 — a demand-paged child's **miss thunk**: its code calls it when an access's page
/// check fails ([`emit_page_check`]), with the access's `len` bytes at window offset `off` of the
/// window at `mem_base`, and [`PAGE_MISS_WRITE`] / [`PAGE_MISS_FATAL`] in `flags`. It returns once
/// every page of the access admits it, having asked the child's pager for each one that may be
/// paged in, or with a trap raised on `vm`, which the code propagates.
///
/// # Safety
/// `vm` is the faulting instance's live context; `[mem_base, mem_base + off + len)` lies in its
/// window's reservation.
pub type PageMissThunk =
    unsafe extern "C" fn(vm: &VmCtx, mem_base: *mut u8, off: u64, len: u64, flags: u64);

/// [`PageMissThunk`] flag: the access stores.
pub const PAGE_MISS_WRITE: u64 = 1;
/// [`PageMissThunk`] flag: no page of the access may be paged in, so a refused one is fatal — a
/// `memory.wait`, which faults on an unsupplied page on every engine.
pub const PAGE_MISS_FATAL: u64 = 2;

/// Free a child `Host` built by a [`GrantDetachedChildBuilder`] — called once, after the child has run and its outcome is stashed for `join`. Deliberately paired with the
/// builder (rather than, say, leaking the host for the run's lifetime like a `Module` grant) because a
/// shell respawning applets would otherwise accumulate one child `Host` per spawn.
///
/// # Safety
/// `ctx` is a [`GrantChild::ctx`] a builder returned and that has not yet been released.
pub type GrantChildReleaser = unsafe extern "C" fn(ctx: *mut core::ffi::c_void);

/// #1810, #2427 — the embedder's report of the pages the window at `base` holds past `mapped` (its
/// declared size) with contents of its own, committed through the Memory capability: the byte offset
/// of each, ascending, in host pages. The host keeps that page map; the JIT does not. A durable
/// capture reads these pages and nothing between them. `ctx` is the pointer the hook was installed
/// with ([`CompiledModule::set_held_tail`]). A plain Rust fn: it hands back a `Vec`.
pub type HeldTail = unsafe fn(ctx: *mut core::ffi::c_void, base: usize, mapped: u64) -> Vec<u64>;

/// PROCESS.md §5 / #1287, D66 — the admission for a detached spawn, the interpreter's
/// `Host::admit_detached_spawn`: reserve the lane of the budget behind `budget` against the parent's
/// Σ and charge `bytes` (the child's declared window), the child's first vCPU, and the mappings its
/// domain and window will hold — `runs` bounds the window's runs of page state
/// (`temen_ir::window_runs_bound`, #2268) — to that budget, or none of them. Returns the lane it
/// reserved (`-1` = unbounded), which the build then stamps on the child
/// ([`GrantDetachedChildBuilder`]), or [`ADMIT_REFUSED`] for a forged/wrong-type handle or no room:
/// the spawn refuses probeably (`-EINVAL`), charging nothing.
///
/// # Safety
/// `ctx` is the run's `cap_ctx` (the parent `Host`).
pub type BudgetMemTaker =
    unsafe extern "C" fn(ctx: *mut core::ffi::c_void, budget: i32, bytes: u64, runs: u64) -> i64;

/// A [`BudgetMemTaker`]'s refusal. Never a lane: a lane is `-1` (unbounded) or non-negative.
pub const ADMIT_REFUSED: i64 = i64::MIN;

/// D66 — return a detached child's **lane** to its parent's `granted_lanes` once the child is reaped
/// (the JIT twin of the interpreter's `credit_child_lane`). Called by the child-domain executor at
/// task finish with the lane the child was stamped with (`-1` = unbounded, a no-op).
pub type LaneGiver = unsafe extern "C" fn(ctx: *mut core::ffi::c_void, lane: i64);

/// #1587, #1877 — settle what a [`BudgetMemTaker`] admitted over `(budget, bytes, runs)`: the
/// window's bytes, the child's first vCPU and its mappings go back to `budget` on the parent, at the
/// child's end or when its spawn fails *after* the take. The OS-thread spawn is the one refusal on the detached path that happens after the
/// commit, so without this a guest that trips it leaks its allowance per attempt. At the child's end
/// `child` is its powerbox (a [`GrantChild::ctx`]), and what its window grew goes back too (#1909);
/// null for a child that never ran.
pub type BudgetMemGiver = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    child: *mut core::ffi::c_void,
    budget: i32,
    bytes: u64,
    runs: u64,
);

/// The **pre-map admission** (the parent side of a spawn record's `region` and `child_off`): may the
/// parent's `SharedRegion` `region` be aliased whole into a child window of
/// `child_size` bytes at `child_off`? `1` admitted, `0` refused (bad geometry — the spawn answers
/// `-EINVAL`, charging nothing), `-1` with a `CapFault` raised on `vm` for a forged handle. Runs
/// **before** the budget take. Host-side it is the interpreter's `Host::premap_admit`, so the two
/// backends admit identically.
///
/// # Safety
/// `ctx` is the run's `cap_ctx` (the parent `Host`); `vm` the spawner's live context.
pub type PremapAdmit = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    region: i32,
    child_off: u64,
    child_size: u64,
    vm: &VmCtx,
) -> i32;

/// Spawn **pre-map staging**: re-grant `region` from the parent into the built child powerbox
/// (`child_ctx`, a [`GrantChild::ctx`]) and record the pending alias at `child_off` — the interpreter's
/// `Host::stage_premap`. Nonzero on success.
///
/// # Safety
/// `ctx` is the parent host; `child_ctx` a child powerbox from a [`GrantDetachedChildBuilder`].
pub type PremapStage = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    child_ctx: *mut core::ffi::c_void,
    region: i32,
    child_off: u64,
) -> i32;

/// The embedder's hook over a fresh window once it is seeded and protected, before its guest runs: a
/// root's ([`CompiledModule::set_window_ready`]) and a detached child's (the nursery's, on the child
/// thread). `ro` lists the `[start, end)` byte ranges its `readonly` data segments made read-only, for
/// the embedder's page map to record as the interpreter's own does (#2268); `shadow` is the arena its
/// module declared, whose runtime-private pages the embedder's page ops refuse (#2426); then a
/// spawn's pre-map, or a thaw's §13 pages, are aliased onto the window `[base, base+mapped)` (inside
/// `reserved`) — the interpreter's `Host::apply_premap` through the host's own window backend, i.e.
/// the same `map_region` a `SharedRegion.map` would take. Nonzero on success; `0` ⇒ the window's
/// guest never runs (a child's `CapFault` outcome, a root's failed run).
///
/// # Safety
/// `ctx` is the window's powerbox; `[base, base+reserved)` its live reservation; `ro` points at `n_ro`
/// ranges (any pointer when `n_ro` is `0`).
pub type WindowReady = unsafe extern "C" fn(
    ctx: *mut core::ffi::c_void,
    base: *mut u8,
    mapped: u64,
    reserved: u64,
    ro: *const [u64; 2],
    n_ro: usize,
    shadow: temen_ir::durable_abi::ShadowArena,
) -> i32;

/// #1944 slice 3 — a detached child's own budget node (the budget that paid for its window), read off
/// child powerbox `child_ctx` (always the shared form): its vCPUs draw their fuel from it, and its
/// threads are charged to it (#2001). `None` when the child holds none. A plain Rust fn: it hands back
/// a Rust trait object.
pub type ChildBudgetNode =
    unsafe fn(child_ctx: *mut core::ffi::c_void) -> Option<std::sync::Arc<dyn BudgetNode>>;

#[derive(Clone, Copy)]
pub struct GrantChildHooks {
    /// PROCESS.md §5 / #1287 — build a **detached** child's powerbox: the by-name grant list, the
    /// child attesting `window_exposed = false` (true when it is paged, #2164), its starter caps
    /// spanning `child_size` = the window
    /// **reservation** (a root's shape), and the budget that paid for its window its own (see
    /// [`GrantDetachedChildBuilder`]).
    pub build_detached: GrantDetachedChildBuilder,
    /// #1287 — the `Budget` quota take (see [`BudgetMemTaker`]).
    pub budget_mem_take: BudgetMemTaker,
    /// #1587 — its undo for a spawn that fails after the take (see [`BudgetMemGiver`]).
    pub budget_mem_give: BudgetMemGiver,
    /// D66 — return a reaped detached child's lane (see [`LaneGiver`]).
    pub lane_give: LaneGiver,
    /// A spawn's pre-mapped region — the host sides of one child-side `map` (see [`PremapAdmit`],
    /// [`PremapStage`]), the last applied as the child's window starts ([`WindowReady`], which every
    /// detached child's window is handed to).
    pub premap_admit: PremapAdmit,
    pub premap_stage: PremapStage,
    pub premap_apply: WindowReady,
    /// #1854, #2427 — [`HeldTail`] over a **child** powerbox (always the shared form): the pages a
    /// durable detached child holds past its declared window, so a freeze's capture of it carries
    /// them, as the root's does.
    pub held_tail: HeldTail,
    /// #1944 slice 3 — a **detached** child's own budget node (see [`ChildBudgetNode`]).
    pub budget_node: ChildBudgetNode,
    /// #2164 — a spawn record's pager check (see [`PagerAdmit`]), and the miss thunk a paged child's
    /// code calls (see [`PageMissThunk`]).
    pub pager_admit: PagerAdmit,
    pub page_miss: PageMissThunk,
    pub release: GrantChildReleaser,
    /// IMPORTS.md phase 3 / S2.1: bind a spawned child module's import manifest against its freshly
    /// built powerbox (`(parent_ctx, child_ctx, module_handle)`) — the JIT-side twin of the
    /// interpreter's inline `Host::bind_child_manifest` at spawn.
    pub bind_imports: ChildManifestBinder,
    /// CALLS.md 5c.0 — mint a `child_offer` (op 14) live-impl in the **parent's** powerbox over a
    /// retained child powerbox (`GrantChild::retained_ctx`): `(parent_ctx, child_ctx, export)` →
    /// the granted handle, or a negative errno (`-EINVAL` for a bad export / unshared child).
    pub mint: ChildOfferMint,
    /// #2220 — grant a capability of the **parent's** into a running child's retained powerbox
    /// (`grant`, op 19; see [`ChildGrant`]).
    pub grant: ChildGrant,
    /// CALLS.md 5c.0 — the cap thunk granted-child compiles run against. A shared child `Host`
    /// (non-null `retained_ctx`) is reachable from the parent while the child runs, so its own
    /// `call.cap`s must synchronize: this is the lock-taking thunk variant (`ctx` = the shared
    /// cell), replacing the run's unsynchronized `cap_thunk` for granted children only.
    pub thunk: CapThunk,
    /// CALLS.md 5c.1b — register a spawned granted child's **serve context** on its shared
    /// powerbox: `(child_ctx, serve_ctx)` where `serve_ctx` is the live child `CompiledModule` address
    /// (resolve handlers via [`handler_tramp_of`], invoke via [`invoke_handler`]).
    /// Called after compile, before the child thread starts; cleared by the releaser (host-side
    /// `Host::set_serve_ctx(0)`) so a stale pointer is never read after child exit.
    pub register_serve: ChildServeRegistrar,
    /// #1234 — the **parent host pointer these hooks decode**, in the shape *this* family expects.
    ///
    /// Every hook above needs the run's parent host, and each one casts this pointer itself. The
    /// run bakes one of two shapes for its `call.cap` sites — a raw host pointer (decoded by the
    /// single-threaded thunk) or a lock cell (decoded by the concurrent one) — and the hooks have
    /// to agree with whichever the run chose. The thunk never disagrees because it and its ctx are
    /// arguments to the *same* [`CompiledModule::compile`] call; the hooks were the outlier, set by
    /// a separate `set_grant_child_hooks` and left to interpret a pointer chosen elsewhere. When
    /// those two sites drifted, a hook read a lock header as a host — memory corruption reachable
    /// from guest code. Carrying the pointer here, set in the same call as the family that decodes
    /// it, makes that drift unrepresentable.
    ///
    /// Null is legal only when the hooks are never invoked.
    pub parent_ctx: *mut core::ffi::c_void,
    /// D66 — the parent domain's `(id, lane cap)`, read off the parent host when this family is
    /// built (the same call that chooses the pointer above): the run's own lane, which its domain's
    /// vCPUs and its children's tasks draw on ([`CompiledModule::set_grant_child_hooks`]). `-1` =
    /// unbounded.
    pub parent_domain: u64,
    pub parent_lane_cap: i64,
    /// #1956 — the [`ModuleResolver`] for `parent_ctx`'s shape: what a detached child's spawns
    /// resolve their `Module` against. (A run's own spawns use the resolver it was compiled with.)
    pub resolve_module: ModuleResolver,
    /// #1956 — the family a **detached** child spawns its own children with (see [`AsParent`]).
    pub as_parent: AsParent,
}

/// #1956 — this hook family for a detached child as a parent: the same hooks over its powerbox
/// `child_ctx` (always the shared form), which becomes `parent_ctx`, with the child's lane
/// coordinates as `parent_domain` / `parent_lane_cap`. A plain Rust fn: it hands back a Rust struct.
pub type AsParent = unsafe fn(child_ctx: *mut core::ffi::c_void) -> GrantChildHooks;

/// Register / clear a granted child's serve context on its shared powerbox — see
/// [`GrantChildHooks::register_serve`].
pub type ChildServeRegistrar =
    unsafe extern "C" fn(child_ctx: *mut core::ffi::c_void, serve_ctx: usize);

/// Mint a `child_offer` live-impl in the parent powerbox over a retained child powerbox — see
/// [`GrantChildHooks::mint`]. Host-side it fetches the export shape from the child's registered
/// module and installs the live-impl entry (the interp op-14 arm's body); the two powerbox locks
/// are never held together.
pub type ChildOfferMint = unsafe extern "C" fn(
    parent_ctx: *mut core::ffi::c_void,
    child_ctx: *mut core::ffi::c_void,
    export: i64,
) -> i32;

/// #2220 — grant `handle`, resolved in the parent powerbox, into the retained powerbox `child_ctx` of
/// a running child (null: the nursery holds no live child there) — see [`GrantChildHooks::grant`].
/// Host-side it is the interp op-19 arm's body: the child's handle, or a negative errno. A forged
/// `handle` raises `CapFault` on `vm`, as a spawn's grant list does, and the result is unread.
pub type ChildGrant = unsafe extern "C" fn(
    parent_ctx: *mut core::ffi::c_void,
    child_ctx: *mut core::ffi::c_void,
    handle: i64,
    vm: &VmCtx,
) -> i32;

/// Bind a child module's import manifest against its built powerbox host — see
/// [`GrantChildHooks::bind_imports`]. Returns `0` on success, nonzero when a `required` import
/// had nothing to bind (IMPORTS.md §3.3 withhold) — the spawn then fails closed with `-EINVAL`
/// before any child code runs, matching the interpreter's inline spawn. `module` names the child's
/// program, as for a [`ModuleResolver`].
pub type ChildManifestBinder = unsafe extern "C" fn(
    parent_ctx: *mut core::ffi::c_void,
    child_ctx: *mut core::ffi::c_void,
    module: i64,
) -> i32;

/// The default thunk for [`compile_and_run`] (no host): an empty powerbox, so every
/// `call.cap` is inert — a `CapFault` — exactly like the interpreter's `run`.
unsafe extern "C" fn empty_cap_thunk(
    _ctx: *mut core::ffi::c_void,
    _mem_base: *mut u8,
    _mem_size: u64,
    _mem_reserved: u64,
    _type_id: u32,
    _op: u32,
    _handle: i32,
    _args: *const i64,
    _n_args: u64,
    _results: *mut i64,
    _n_results: u64,
    vm: &VmCtx,
) {
    vm.raise(TrapKind::CapFault as i64);
}

/// The inert [`CapThunk`] (every `call.cap` is a `CapFault`) for callers with no host — the
/// long-lived [`CompiledModule::compile`] counterpart of [`compile_and_run`]'s empty powerbox.
/// Pass with a null `cap_ctx`.
pub const INERT_CAP_THUNK: CapThunk = empty_cap_thunk;

/// The CLIF type backing an IR value type.
fn clif_ty(t: ValType) -> Type {
    match t {
        ValType::I32 => I32,
        ValType::I64 => I64,
        ValType::F32 => F32,
        ValType::F64 => F64,
        // §17/D58: a `v128` SSA value is canonically held as `I8X16`; lane ops bitcast to the
        // shape-specific vector type and back.
        ValType::V128 => I8X16,
        // An opaque `ref` lowers exactly as `i64` (GC forward-compat reservation, GC.md §6).
        ValType::Ref => I64,
        // §3.5 `cap` is i32-width handle data (translation reservation).
        ValType::Cap => I32,
    }
}

/// The shape-specific CLIF vector type for a lane op (all 128-bit).
fn vec_ty(shape: VShape) -> Type {
    match shape {
        VShape::I8x16 => I8X16,
        VShape::I16x8 => I16X8,
        VShape::I32x4 => I32X4,
        VShape::I64x2 => I64X2,
        VShape::F32x4 => F32X4,
        VShape::F64x2 => F64X2,
    }
}

/// The CLIF **lane** type for a shape (the scalar a lane holds in CLIF).
fn lane_clif(shape: VShape) -> Type {
    match shape {
        VShape::I8x16 => I8,
        VShape::I16x8 => I16,
        VShape::I32x4 => I32,
        VShape::I64x2 => I64,
        VShape::F32x4 => F32,
        VShape::F64x2 => F64,
    }
}

/// Reinterpret a 128-bit vector value to another 128-bit vector type (a no-op bitcast,
/// little-endian lane order). Used to move between the canonical `I8X16` and a shape type.
fn vcast(b: &mut FunctionBuilder, v: Value, to: Type) -> Value {
    if b.func.dfg.value_type(v) == to {
        return v;
    }
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    b.ins().bitcast(to, mf, v)
}

/// The CLIF type for an integer-class IR type (operands to int↔float conversions).
fn int_clif_ty(t: IntTy) -> Type {
    match t {
        IntTy::I32 => I32,
        IntTy::I64 => I64,
    }
}

/// The CLIF type for a float-class IR type.
fn float_clif_ty(t: FloatTy) -> Type {
    match t {
        FloatTy::F32 => F32,
        FloatTy::F64 => F64,
    }
}

/// Compile the whole module and run `func` on slot-encoded `args` (each `i64` is one
/// parameter slot; `i32`/`f32` occupy the low 32 bits). Returns the result slots, or a
/// [`JitOutcome::Trapped`] if the run trapped. Intended for the differential harness.
///
/// All functions are compiled with a **natural CLIF ABI** — `(mem_base, fn_table_base,
/// vmctx, params…) -> (results…)` — so direct/indirect/tail calls are ordinary CLIF
/// calls; the entry is wrapped in a fixed buffer-ABI trampoline (any arity).
pub fn compile_and_run(m: &IrModule, func: FuncIdx, args: &[i64]) -> Result<JitOutcome, JitError> {
    // No host: an empty powerbox, so any `call.cap` is an inert CapFault (like `run`).
    compile_and_run_with_host(m, func, args, empty_cap_thunk, core::ptr::null_mut())
}

/// Compile `func` of `m` to a reusable [`CompiledModule`] with the default no-host policy (an empty
/// powerbox — any `call.cap` is an inert CapFault, exactly like [`compile_and_run`]), so the caller can
/// **compile once and [`CompiledModule::run`] many times** (DESIGN.md §22's long-lived split). The
/// one-shot [`compile_and_run`] recompiles the whole module on *every* call (~ms of Cranelift codegen);
/// a hot loop or a benchmark isolating per-iteration compute from compile jitter should compile here and
/// reuse the returned module. For `call.cap` dispatch to a real host, call [`CompiledModule::compile`]
/// directly with a thunk.
pub fn compile(m: &IrModule, func: FuncIdx) -> Result<CompiledModule, JitError> {
    CompiledModule::compile(
        m,
        func,
        empty_cap_thunk,
        core::ptr::null_mut(),
        temen_ir::default_reserved_log2(),
        None, // resolve_module
        None, // interrupt
        None, // fuel
        None, // fast_resolver
        None, // node
        0,    // natural (non-B2-reserved) function-table size
    )
}

/// Like [`compile_and_run`], but `call.cap`s dispatch through `cap_thunk` with the
/// caller's `cap_ctx` (the powerbox host). The compiled code calls the thunk, and its instance
/// holds the ctx — valid because the module is compiled, run once, then discarded here.
///
/// # Safety
/// `cap_thunk`/`cap_ctx` must stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_with_host(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<JitOutcome, JitError> {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed. Callers
    // wanting a specific reservation use the `_reserved` capture entry.
    Ok(run_inner(m, func, args, cap_thunk, cap_ctx, RunOpts::default())?.0)
}

/// Like [`compile_and_run_with_host`], but also supply a [`FastCapResolver`] so hot `call.cap`s to
/// the resolver's known `(type_id, op)` pairs take the **devirtualized fast path** (register-to-
/// register, no stack marshalling, no runtime dispatch — §9 / D45). Calls the resolver doesn't claim
/// fall back to `cap_thunk`. This is the entry an embedder uses once it has specialized host functions
/// for its hot capabilities (the generic `cap_thunk` stays the correctness fallback).
///
/// # Safety
/// `cap_thunk`/`cap_ctx`/`fast_resolver` (and every function it returns) must stay valid for the call
/// and honour their respective ABIs.
pub fn compile_and_run_with_host_fast(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    fast_resolver: FastCapResolver,
    node: Option<std::sync::Arc<dyn BudgetNode>>,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            fast_resolver: Some(fast_resolver),
            node,
            ..RunOpts::default()
        },
    )?
    .0)
}

/// Like [`compile_and_run_with_host`], but **arm the §5 fuel/epoch kill-path**: the lowering polls
/// `*interrupt` at every loop back-edge and function entry, and traps [`TrapKind::OutOfFuel`] as
/// soon as the host stores a non-zero value there. This is how a host bounds a *runaway* JIT guest
/// (an infinite loop / unbounded recursion) — the interpreter has always had this via its fuel
/// counter; this gives the production backend the matching, **guest-undisableable** kill-path
/// (DESIGN §5 / preemption). The caller owns `interrupt` (typically an `Arc<AtomicU64>`) and sets it
/// from a watchdog timer, a cross-domain preemption, a signal handler, etc.
///
/// # Safety
/// `interrupt` must point at a live `AtomicU64` that outlives the call; `cap_thunk`/`cap_ctx` must
/// stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_with_host_interruptible(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    interrupt: *const AtomicU64,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            interrupt: Some(interrupt),
            ..RunOpts::default()
        },
    )?
    .0)
}

/// #932 — [`compile_and_run_with_host`] with **async signal delivery armed**: the lowering polls the
/// host `armed` flag (`Host::sig_armed`) at its safepoints (function entries + taken back-edges) and,
/// when a signal is deliverable, redirects into the guest handler `void(int)` on the guest-registered
/// signal stack — the JIT tier of the interpreters' per-op redirect (`#796`/`#1146`), so a
/// compute-bound guest on the JIT delivers its handler mid-loop. The `SignalArm`'s decision/return
/// thunks carry the shared `SignalSource` logic (nesting bound, `take_deliverable`, `handler_returned`),
/// so delivery is byte-identical to the interpreter under the differential oracle. Build the arm from
/// the granted host with `temen_run::jit_sig_delivery` + `jit_sig_take`/`jit_sig_return`.
///
/// # Safety
/// As [`compile_and_run_with_host`], plus every address in `signal` (the `armed` flag, the two thunks,
/// and the delivery ctx) must stay valid and honour its ABI for the whole call (they are baked into
/// the compiled code).
pub fn compile_and_run_with_host_signals(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    signal: SignalArm,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            signal: Some(signal),
            ..RunOpts::default()
        },
    )?
    .0)
}

/// [`compile_and_run_with_host_interruptible`] + the §9/D45 [`FastCapResolver`]: the production run
/// path — a guest-undisableable kill-path **and** hot `call.cap`s devirtualized. The resolver's
/// unclaimed ops fall back to `cap_thunk` unchanged.
///
/// # Safety
/// As [`compile_and_run_with_host_interruptible`], plus `fast_resolver` (and every fn it returns) must
/// honour the [`FastCapResolver`] ABI and stay valid for the call.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_with_host_interruptible_fast(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    interrupt: *const AtomicU64,
    fast_resolver: FastCapResolver,
    node: Option<std::sync::Arc<dyn BudgetNode>>,
) -> Result<JitOutcome, JitError> {
    Ok(run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            interrupt: Some(interrupt),
            fast_resolver: Some(fast_resolver),
            node,
            ..RunOpts::default()
        },
    )?
    .0)
}

/// Like [`compile_and_run_with_host`], but **arm safepoint-anchored counted fuel** (INTERP_PERF.md
/// "Fuel unification"): the lowering decrements `*fuel` by one at every function entry, taken
/// back-edge and `cont.resume`, and traps [`TrapKind::OutOfFuel`] when the budget would go below zero.
/// Unlike the §5 kill-path ([`compile_and_run_with_host_interruptible`], a host-written cell polled
/// asynchronously), this is a deterministic **guest budget**: `fuel` counts exactly the function
/// entries, taken back-edges and resumes executed — the unit the tree-walker and bytecode engines
/// charge — so a run either completes or traps `OutOfFuel` at the same safepoint on all three
/// backends. The caller seeds `fuel` with the budget and reads the remainder back after the call: a
/// fixed allowance, never refilled (#1944 slice 3).
///
/// # Safety
/// `cap_thunk`/`cap_ctx` must stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_with_host_fuel(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    fuel: &mut u64,
) -> Result<JitOutcome, JitError> {
    let mut cell = FuelCell::fixed(*fuel);
    let r = run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            fuel: Some(&mut *cell),
            ..RunOpts::default()
        },
    );
    *fuel = cell.left;
    Ok(r?.0)
}

/// Like [`compile_and_run`], but seed the guest window with `init_mem` (its low bytes) and
/// return the final window contents — the JIT side of the **escape-oracle** (§18). A
/// verified module that runs to completion must leave a window byte-identical to the
/// interpreter's [`temen_interp::run_capture`]; any divergence is a confinement/codegen
/// escape — a load/store whose effective address was not masked into `[0, size)`.
pub fn compile_and_run_capture(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    // Default reservation policy (§4): a large reserved range, only `mapped` backed.
    compile_and_run_capture_reserved(m, func, args, init_mem, temen_ir::default_reserved_log2())
}

/// Like [`compile_and_run_capture`], but with a host **reservation policy**: the window masks
/// into `[0, 2^reserved_log2)` (the mask domain) while only the declared `1 << size_log2` bytes
/// are backed — an access into the reserved-but-unmapped tail faults (§4 "guard-when-bounded";
/// detect-and-kill, §5). `reserved_log2` is raised to at least `size_log2` (so `0` ⇒ fully
/// mapped) and capped at the reference JIT's [`MAX_JIT_RESERVED_LOG2`]. This is the JIT side of
/// the escape-oracle under the decoupled `reserved`/`mapped` model; both backends must be driven
/// with the *same* `reserved_log2` to stay in differential lockstep.
pub fn compile_and_run_capture_reserved(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        empty_cap_thunk,
        core::ptr::null_mut(),
        RunOpts {
            init_mem: Some(init_mem),
            reserved_log2,
            snapshot_cap: Some(0), // the backed prefix
            ..RunOpts::default()
        },
    )
}

/// [`compile_and_run_capture_reserved`] + a live powerbox: `call.cap`s dispatch through
/// `cap_thunk`/`cap_ctx` (so a granted handle takes its **success** path) *and* the final window
/// is captured for the escape-oracle. Pairs with the interpreter's
/// [`temen_interp::run_capture_reserved_with_host`] to byte-compare the effects of the §3e Memory
/// capability (`map`/`unmap`/`protect`) across both backends.
///
/// # Safety
/// `cap_thunk`/`cap_ctx` must stay valid for the call and honour the [`CapThunk`] contract.
pub fn compile_and_run_capture_reserved_with_host(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    compile_and_run_capture_reserved_with_host_ex(
        m,
        func,
        args,
        init_mem,
        reserved_log2,
        cap_thunk,
        cap_ctx,
        None,
        None, // no child-powerbox hooks: a spawn is an inert CapFault
    )
}

/// [`compile_and_run_capture_reserved_with_host`] + a §14 **module resolver**: the host callback the
/// nesting runtime uses to resolve a guest's `Module` handle when it `instantiate`s a
/// **separate-module child** (the Instantiator's module ops). `None` ⇒ module ops are an inert
/// `CapFault` (same as a host that granted no modules).
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`]; `resolve_module` (with `cap_ctx`) must honour
/// the [`ModuleResolver`] contract — in particular the resolved views must outlive the run.
/// `grant_child` (PROCESS.md S2 JIT parity) supplies the hooks that build and release a detached
/// child's powerbox ([`GrantChildHooks`]); `None` ⇒ a spawn is an inert `CapFault`. Both hooks share
/// `cap_ctx` as their host.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_ex(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    resolve_module: Option<ModuleResolver>,
    grant_child: Option<GrantChildHooks>,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            init_mem: Some(init_mem),
            reserved_log2,
            // Escape-oracle over the §1a growth path: snapshot the low `SNAP_CAP` bytes (not just the
            // backed prefix) so guest-grown / `unmap`-ed reserved-tail pages are byte-compared too.
            snapshot_cap: Some(SNAP_CAP),
            resolve_module,
            grant_child,
            ..RunOpts::default()
        },
    )
}

/// [`compile_and_run_capture_reserved_with_host`] with a **counted-fuel budget armed** (INTERP_PERF.md
/// "Fuel unification"): the caller seeds `fuel` with the budget and reads the remainder back after the
/// call (a fixed allowance, never refilled — #1944 slice 3); the run traps [`TrapKind::OutOfFuel`]
/// when the budget would underflow, at the same IR safepoints (function entries + taken back-edges +
/// `cont.resume`) the interpreters charge.
/// This lets the differential fuzzer **assert** cross-engine `OutOfFuel` parity rather than exclude it.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_fuel(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    fuel: &mut u64,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let mut cell = FuelCell::fixed(*fuel);
    let r = run_inner(
        m,
        func,
        args,
        cap_thunk,
        cap_ctx,
        RunOpts {
            init_mem: Some(init_mem),
            reserved_log2,
            snapshot_cap: Some(SNAP_CAP),
            // counted-fuel budget armed — traps OutOfFuel at the shared safepoints
            fuel: Some(&mut *cell),
            ..RunOpts::default()
        },
    );
    *fuel = cell.left;
    r
}

/// [`compile_and_run_capture_reserved_with_host`] that first **re-establishes** a captured
/// per-page protection map on the window — the durable-restore step (DURABILITY.md §12.3): a
/// thawed guest faults on a restored `Ro`/`Unmapped` page exactly as the frozen one would,
/// matching `temen-interp`. `init_prots[i]` is the protection of the backed-prefix page at
/// `[i*DURABLE_SNAPSHOT_PAGE, …)`; pages beyond `init_prots` (or `Rw`) keep the default RW.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`].
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_capture_reserved_with_host_prots(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    init_prots: &[WindowProt],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let mut cm = CompiledModule::compile(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        None, // no module resolver
        None, // no interrupt
        None, // no fuel budget
        None, // no fast cap resolver
        None, // no budget node
        0,    // one-shot path: natural table size
    )?;
    cm.restore = (!init_prots.is_empty()).then(|| WindowRestore {
        prots: init_prots.to_vec(),
        tail: Vec::new(),
    });
    cm.run(args, Some(init_mem), Some(SNAP_CAP))
}

/// An async **freeze controller** (DURABILITY.md Phase-4 Slice A, 4A.3): a caller-owned handle a
/// *controller thread* uses to request a stop-the-world freeze of an in-flight durable JIT run. The
/// run publishes its live window's freeze word here once the window is mapped ([`CompiledModule::run`]);
/// [`Self::request_freeze`] then stores `UNWINDING` into it, which the guest's
/// loop-header back-edge poll (4A.1) observes at its next iteration and begins unwinding — closing
/// the R6 latency caveat for a poll-free compute loop with no deterministic countdown.
///
/// This is the real async trigger (vs. the deterministic `arm_freeze_after_backedges` test oracle,
/// which the interpreter keeps because its window is private/synchronous). The interpreter has no
/// equivalent — only the JIT's window is a shared mmap a second thread can reach.
///
/// # Lifetime
/// [`Self::request_freeze`] may be called at any time; only the first call stores. Its store and the run's
/// [`Self::retire`] exclude each other through `word` (a request holds [`STORING`] across its store;
/// retire waits it out), so the store only ever lands in a live window, and a request after the run
/// ended is a no-op. A run of a module with no arena has no freeze word (#2518): nothing in it polls,
/// so nothing can stop it mid-run, and its controller is retired from the start. A request that lands
/// after the guest's last poll but before retire freezes nothing: the run sees it at retire and treats
/// it as never made (see [`CompiledModule::run`]).
pub struct FreezeController {
    /// `0` = window not live yet (spin); the address of a live window's freeze word; [`STORING`]
    /// while a request writes; [`LANDED`] once it wrote; [`RETIRED`] once the run ended.
    word: AtomicUsize,
}

/// Store `UNWINDING` into the freeze word at `word` — what a request does to the run's window.
///
/// # Safety
/// `word` is a live durable window's freeze word, in the arena its module declared and the verifier
/// bounded into the window.
unsafe fn ring(word: usize) {
    // An aligned atomic i32 store the guest's back-edge poll loads (defined under §12 races).
    (*(word as *const AtomicI32)).store(temen_ir::durable_abi::STATE_UNWINDING, Ordering::Release);
}

/// [`FreezeController::word`] while a request's store is in flight.
const STORING: usize = usize::MAX - 2;
/// [`FreezeController::word`] once a request's store has landed.
const LANDED: usize = usize::MAX - 1;
/// [`FreezeController::word`] once the run has ended.
const RETIRED: usize = usize::MAX;

impl FreezeController {
    /// A fresh controller, shareable with a run and a controller thread.
    pub fn new() -> Arc<Self> {
        Arc::new(FreezeController {
            word: AtomicUsize::new(0),
        })
    }

    /// Request a freeze: spin until the run publishes its window's freeze word, then store
    /// `UNWINDING` into it. A no-op if the run already ended or another request got there first.
    pub fn request_freeze(&self) {
        loop {
            match self.word.load(Ordering::Acquire) {
                0 => std::hint::spin_loop(),          // window not mapped yet
                STORING | LANDED | RETIRED => return, // another request has it, or the run ended
                word => {
                    if self
                        .word
                        .compare_exchange(word, STORING, Ordering::AcqRel, Ordering::Acquire)
                        .is_err()
                    {
                        continue; // retired meanwhile: the next load says so
                    }
                    // SAFETY: `word` was the live window's and `retire` cannot complete while this
                    // request holds `STORING`, so the window is mapped.
                    unsafe { ring(word) };
                    self.word.store(LANDED, Ordering::Release);
                    return;
                }
            }
        }
    }

    /// Run-side: publish the live window's freeze word (the run is now blocked in the guest), or
    /// retire at once for a window with none, whose run no request can stop.
    fn publish(&self, word: Option<usize>) {
        self.word.store(word.unwrap_or(RETIRED), Ordering::Release);
    }

    /// Run-side: retire the base once the guest returns, so a later `request_freeze` is a no-op.
    /// Waits out a store in flight. `true` if a request's store landed.
    fn retire(&self) -> bool {
        loop {
            let cur = self.word.load(Ordering::Acquire);
            if cur == STORING {
                std::hint::spin_loop();
                continue;
            }
            if self
                .word
                .compare_exchange(cur, RETIRED, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return cur == LANDED;
            }
        }
    }
}

#[cfg(test)]
mod freeze_controller_tests {
    use super::*;

    /// A live window's freeze word, as a request reaches it.
    fn word_of(w: &AtomicI32) -> Option<usize> {
        Some(w as *const AtomicI32 as usize)
    }

    #[test]
    fn a_request_lands_in_the_live_window_and_retire_reports_it() {
        let w = AtomicI32::new(0);
        let fc = FreezeController::new();
        fc.publish(word_of(&w));
        fc.request_freeze();
        assert_eq!(w.load(Ordering::Relaxed), 1, "UNWINDING stored");
        fc.request_freeze(); // a second request is a no-op
        assert!(fc.retire(), "retire reports the landed request");
    }

    /// #2518: a run whose window has no freeze word publishes none: a request returns at once and
    /// lands nowhere.
    #[test]
    fn a_window_without_a_freeze_word_retires_its_controller_at_once() {
        let fc = FreezeController::new();
        fc.publish(None);
        fc.request_freeze();
        assert!(!fc.retire(), "nothing landed");
    }

    /// #1693: a run that fails before publishing its window (here a bad entry index) still retires
    /// its controller, so a request returns rather than spinning for a window that never comes.
    #[test]
    fn a_run_that_fails_before_publishing_retires_its_controller() {
        let m = temen_text::parse_module(
            "memory 16\nfunc () -> (i64) {\nblock 0 () {\n  v0 = i64.const 0\n  return v0\n  }\n}\n",
        )
        .expect("parse");
        let fc = Arc::new(FreezeController::new());
        let run = DurableRun {
            freeze: Some(Arc::clone(&fc)),
            ..DurableRun::default()
        };
        let r = compile_and_run_durable(
            &m,
            7,
            &[],
            &[],
            16,
            empty_cap_thunk,
            core::ptr::null_mut(),
            run,
        );
        assert!(r.is_err(), "entry 7 does not exist");
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            fc.request_freeze();
            let _ = tx.send(());
        });
        assert!(
            rx.recv_timeout(std::time::Duration::from_secs(10)).is_ok(),
            "request_freeze returned"
        );
    }

    #[test]
    fn a_request_after_retire_stores_nothing() {
        let w = AtomicI32::new(0);
        let fc = FreezeController::new();
        fc.publish(word_of(&w));
        assert!(!fc.retire(), "nothing landed");
        fc.request_freeze();
        assert_eq!(
            w.load(Ordering::Relaxed),
            0,
            "the retired window is untouched"
        );
    }
}

/// A durable JIT run's **whole** freeze/thaw residue (DURABILITY.md §12.4, §12.8): what a thaw
/// re-creates before the root re-enters under `REWINDING`, and what a freeze hands back — the same
/// pieces the interpreter's `ThawResidue` carries, so one embedder hand-off serves both engines
/// (#1690: each durable entry used to carry a different subset, silently dropping the rest).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DurableResidue {
    /// Fibers the freeze flattened (§12.4).
    pub fibers: Vec<FrozenFiber>,
    /// `thread.spawn`ed vCPUs that unwound under the freeze (slice 3.3).
    pub vcpus: Vec<FrozenVCpu>,
    /// The root vCPU's flattened shadow-SP extent (slice 3.3): reported beside the vCPUs because the
    /// shared active-SP word ends at the last child's. `None` ⇒ the root's extent is the window's own
    /// active-SP word (a single-vCPU run).
    pub root_sp: Option<u64>,
}

/// #1361 step 4 — a **detached** child a durable JIT freeze reached: what the embedder needs to put
/// it in the parent's artifact (as the interpreter's `CapturedDetached`) or to tell why it cannot.
pub struct DetachedHarvest {
    /// The parent's join slot for it.
    pub slot: usize,
    /// Its window geometry.
    pub mapped_log2: u8,
    pub reserved_log2: u8,
    /// Its powerbox: the nursery's retained ref ([`GrantChild::retained_ctx`]) — one counted ref to
    /// the child `Host`, which the embedder now owns and must release. Null for a child a thaw
    /// delivered as completed (#2044): its `outcome` is all there is, and the geometry is unused.
    pub powerbox: *mut core::ffi::c_void,
    /// What it left, **iff** it unwound for the freeze.
    pub capture: Option<DetachedCapture>,
    /// `(result, trap)` once its task finished: it completed before the cut (`trap == 0` and no
    /// capture), or was torn down unreached (it never polled — parked, #1584).
    pub outcome: Option<(i64, i64)>,
    /// #2010 — its own unjoined durable detached children, harvested the same way.
    pub children: Vec<DetachedHarvest>,
}

/// #1361 step 4 — what a durable detached child that unwound for a freeze left for the harvest.
pub struct DetachedCapture {
    /// Its window image: its declared window.
    pub image: Vec<u8>,
    /// #2427 — each page it holds past that with contents of its own, by byte offset.
    pub tail: Vec<(u64, Vec<u8>)>,
    /// What a run's root leaves beside its image: its fibers (#2031), its `thread.spawn` vCPUs,
    /// including each that finished unjoined, which its root's re-issued join is owed (#2010), and
    /// its root's extent.
    pub residue: DurableResidue,
}

/// #1361 step 4 — a detached child a JIT **thaw** re-creates in its parent's nursery, at the join
/// slot it was spawned at (see [`CompiledModule::set_detached_seed`]).
pub enum DetachedSeed {
    /// One the freeze captured live: re-launched under `REWINDING`.
    Captured(Box<CapturedSeed>),
    /// #2041 — one that had completed but was not joined: its `join` outcome, which the parent's
    /// rewound `join` reloads, as the interpreter's thaw delivers it.
    Completed { slot: usize, result: i64, trap: i64 },
}

/// #1361 step 4 — what a captured detached child's re-launch needs.
pub struct CapturedSeed {
    pub slot: usize,
    pub mapped_log2: u8,
    pub reserved_log2: u8,
    /// Its own program, resolved by the embedder from the module digest the artifact names.
    pub funcs: Arc<[Func]>,
    pub types: Arc<[temen_ir::TypeEntry]>,
    /// #2247 — its program's impl exports, whose handlers it serves.
    pub impl_exports: Vec<temen_ir::ImplExport>,
    pub shadow: temen_ir::durable_abi::ShadowArena,
    /// temen#2168 — that digest, which names its code to the run's [`ChildCodeCache`], as a
    /// spawn's grant does.
    pub digest: [u8; 32],
    /// Its restored window image.
    pub image: Vec<u8>,
    /// Its restored page map over `image`, one entry per [`DURABLE_SNAPSHOT_PAGE`] (#1733); empty
    /// when no page deviates from a fresh window's.
    pub prots: Vec<WindowProt>,
    /// #2427 — and each page it holds past `image`.
    pub tail: Vec<TailPage>,
    /// Its restored powerbox, as a builder fills it: two counted refs (`ctx`, `retained_ctx`).
    pub child: GrantChild,
    /// #1971 — the lease its window was charged under, as the spawn filed it: the parent's handle on
    /// the budget that paid, the bytes it paid, and the bound on its runs (#2268). Handed back when
    /// the thawed child ends. `None` when the parent holds no handle on that budget.
    pub window: Option<(i32, u64, u64)>,
    /// #2010 — its own children, re-created in its nursery before it runs.
    pub children: Vec<DetachedSeed>,
    /// Its [`DetachedCapture::residue`], re-created before its root runs: its fibers re-seeded in its
    /// table (#2031), and its `thread.spawn` vCPUs re-spawned into its domain, each rewinding from its
    /// own extent (#2010).
    pub residue: DurableResidue,
}

/// #1768 — the embedder's side of a **fork** on the JIT (FORK.md §9.5): fork is durable freeze →
/// clone the window → thaw both, with each copy's reply injected (FORK.md §1, §3).
///
/// The program is compiled with its fork calls as durable suspend points (the `temen-durable`
/// transform, sites = the fork calls), and the run is armed with this hook
/// ([`CompiledModule::set_fork_hook`]). When the guest forks, the embedder's `call.cap` thunk sets
/// the window's freeze word, the instrumented frames unwind into the root's shadow stack, and the
/// root returns — unwound, not finished. The run then calls the hook with the frozen window: the
/// hook duplicates the process — mints the twin's pid and powerbox, takes the twin's window from
/// [`ForkPoint::twin`] and starts it ([`CompiledModule::run_twin`], on its own thread) — and returns
/// the parent's reply: the twin's pid, or a negative errno when it refused (nothing duplicated).
/// The run injects that reply, rewinds the root in place, and carries on past the fork call.
pub type ForkHook = Box<dyn FnMut(&ForkPoint<'_>) -> i64 + Send>;

/// #1768 — a run frozen at a fork, as its [`ForkHook`] sees it: the live window, unwound into its
/// root's shadow stack, and what a private duplicate of it needs.
pub struct ForkPoint<'a> {
    window: &'a mem::GuestWindow,
    /// The NULL guard every window of this module seeds inaccessible (`None` when it does not fit).
    null_guard: Option<u64>,
    shadow: temen_ir::durable_abi::ShadowArena,
    args: &'a [i64],
}

/// #1768 — a fork twin's window: a private duplicate of its parent's at the fork, its reply injected
/// and its root set rewinding — what [`CompiledModule::run_twin`] runs. Moves to the twin's thread.
pub struct TwinWindow {
    window: mem::GuestWindow,
    /// The twin's `vcpu.tls` register at its first entry: its parent's at the fork.
    tls: i64,
}

impl TwinWindow {
    /// The duplicate's base — the key the twin's page-state map is filed under.
    pub fn base(&self) -> usize {
        self.window.base() as usize
    }

    /// The duplicate's backed prefix: the declared window the twin pays for (#2106).
    pub fn mapped(&self) -> u64 {
        self.window.mapped() as u64
    }
}

impl ForkPoint<'_> {
    /// The forking entry's arguments — what the twin's run re-enters the entry with.
    pub fn args(&self) -> &[i64] {
        self.args
    }

    /// The base of the frozen window — the key its page-state map is filed under (the embedder's
    /// `call.cap` view of the window keeps one; see [`Self::twin`]).
    pub fn window_base(&self) -> usize {
        self.window.base() as usize
    }

    /// The twin's window: a private duplicate of this one (the JIT's `Mem::fork_private`) — every
    /// committed page's bytes and protection — with `reply` injected as the fork call's result and
    /// its root set rewinding. `pages` is the run's page-state map (host page → its
    /// [`temen_ir::PageState`], absent ⇒ the default): the read-only pages the window started with
    /// (#2268) and every page the guest committed or protected since, a `readonly` data segment's
    /// page it made writable left out of it (#2307). With the NULL guard, which no page op changes,
    /// it describes every page. `None` when the window cannot be duplicated: a page aliases a §13
    /// shared region (fork shares no memory — the interpreter's `fork_private` refuses the same
    /// window), or the address space has no room for another window.
    pub fn twin(
        &self,
        pages: &BTreeMap<u64, temen_ir::PageState>,
        reply: i64,
    ) -> Option<TwinWindow> {
        use temen_ir::PageState as ps;
        let page = mem::page_size() as u64;
        let mut prots: BTreeMap<usize, mem::Prot> = BTreeMap::new();
        // The NULL guard (host-page granular, as `GuestWindow::protect` applies it).
        if let Some(guard) = self.null_guard.filter(|g| g.is_multiple_of(page)) {
            for p in 0..guard / page {
                prots.insert(p as usize, mem::Prot::None);
            }
        }
        // Then the page map.
        let prefix = (self.window.mapped() as u64).div_ceil(page);
        for (&p, &state) in pages {
            let prot = match state {
                ps::Rw => mem::Prot::Rw,
                ps::Ro => mem::Prot::Ro,
                ps::Unmapped => mem::Prot::None,
                ps::Backed { .. } => return None, // a §13 alias: refuse
            };
            if prot == mem::Prot::Rw && p < prefix {
                prots.remove(&(p as usize)); // the prefix default
            } else {
                prots.insert(p as usize, prot);
            }
        }
        // SAFETY: the run is unwound — no guest code runs on this window until it rewinds — and
        // every index is a page of the window (the page map only ever records its own pages).
        let window = unsafe { self.window.fork_copy(&prots) }?;
        // SAFETY: the duplicate is this window's layout, so the reply slot and control words lie in
        // its backed prefix.
        unsafe { arm_fork_rewind(window.base(), self.shadow, reply) };
        Some(TwinWindow {
            window,
            tls: vcpu_tls::get(),
        })
    }
}

/// The window's freeze word, read by the host between entries; `NORMAL` for a module with no arena,
/// which has none (#2518).
///
/// # Safety
/// `base` is a live window of a module declaring `shadow`.
unsafe fn window_state(base: *mut u8, shadow: temen_ir::durable_abi::ShadowArena) -> i32 {
    shadow
        .freeze_word()
        .map_or(temen_ir::durable_abi::STATE_NORMAL, |w| {
            (base.add(w as usize) as *const i32).read_unaligned()
        })
}

/// #1768 — set a window frozen at a fork up to **rewind** its root with `reply` as the fork call's
/// result (FORK.md §3, reply injection): the reply into the root's leaf frame
/// ([`temen_ir::durable_abi::ShadowArena::leaf_reply`]), the root context's thaw word `REWINDING`, and
/// the global freeze word back to `NORMAL` (else the rewinding code's polls would unwind again) —
/// `temen_durable::inject_leaf_reply` + `begin_thaw`, on a live window.
///
/// # Safety
/// `base` is a live window of a module declaring `shadow`, frozen at a fork (its root's deepest frame
/// is the fork call's leaf), with nothing running on it.
unsafe fn arm_fork_rewind(base: *mut u8, shadow: temen_ir::durable_abi::ShadowArena, reply: i64) {
    use temen_ir::durable_abi::{STATE_NORMAL, STATE_REWINDING};
    (base.add(shadow.leaf_reply(0) as usize) as *mut i64).write_unaligned(reply);
    (base.add(shadow.thaw_state_off(0) as usize) as *mut i32).write_unaligned(STATE_REWINDING);
    if let Some(w) = shadow.freeze_word() {
        (base.add(w as usize) as *mut i32).write_unaligned(STATE_NORMAL);
    }
}

/// How a durable JIT run is driven beyond its residue.
#[derive(Clone, Default)]
pub struct DurableRun {
    /// Per-page protections to re-establish on a thawed window (§12.3); empty on a freeze.
    pub init_prots: Vec<WindowProt>,
    /// The thaw seed (empty on a freeze or an ordinary durable run).
    pub seed: DurableResidue,
    /// An async freeze controller (Phase-4 Slice A): a controller thread's
    /// [`FreezeController::request_freeze`] lands mid-run. Its presence also engages the **concurrent**
    /// durable path (§12.8 4A.5): a child spawned while `NORMAL` runs on its own OS thread with its own
    /// reserved shadow context, so an async freeze reaches it in its own region — never the root's
    /// (#1691: an async freeze without it unwound children into the root's region).
    pub freeze: Option<Arc<FreezeController>>,
}

/// The one **durable** JIT entry (DURABILITY.md §12.8): [`compile_and_run_capture_reserved_with_host_prots`]
/// with the per-fiber shadow-SP swap armed, the freeze driven (fibers flattened, spawned vCPUs and §14
/// children unwound), and the whole residue round-tripped.
///
/// - **Freeze:** an empty [`DurableRun::seed`]; returns every piece of residue the freeze produced.
/// - **Thaw:** the artifact's page map as `init_prots` and its residue as `seed`; returns an empty
///   residue.
///
/// # Safety
/// As [`compile_and_run_capture_reserved_with_host`]; with a controller, its lifetime contract (call
/// `request_freeze` at most once, concurrently with this run) must hold.
#[allow(clippy::too_many_arguments)]
pub fn compile_and_run_durable(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    init_mem: &[u8],
    reserved_log2: u8,
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    run: DurableRun,
) -> Result<(JitOutcome, Vec<u8>, DurableResidue), JitError> {
    let freeze = run.freeze.clone();
    let r = (|| {
        let mut cm = CompiledModule::compile(
            m,
            func,
            cap_thunk,
            cap_ctx,
            reserved_log2,
            None, // resolve_module
            None, // interrupt
            None, // fuel
            None, // fast_resolver
            None, // node
            0,
        )?;
        cm.restore = (!run.init_prots.is_empty()).then(|| WindowRestore {
            prots: run.init_prots,
            tail: Vec::new(),
        });
        cm.set_durable(run.seed);
        cm.concurrent_durable = run.freeze.is_some();
        cm.freeze_ctl = run.freeze;
        let (outcome, win) = cm.run(args, Some(init_mem), Some(SNAP_CAP))?;
        Ok((outcome, win, cm.take_durable_residue()))
    })();
    // #1693: retire on every exit. A run that failed before publishing its window would otherwise
    // leave a request spinning forever on base 0; after the run's own retire this is a no-op.
    if let Some(fc) = freeze {
        fc.retire();
    }
    r
}

#[allow(clippy::too_many_arguments)]
/// The optional policy knobs [`run_inner`] threads to the lowering + runtime. Every field defaults
/// to a no-op (the empty-powerbox reservation, no kill-path / fuel / quota / nesting / capture), so a
/// public entry point sets only the one or two knobs it arms instead of threading ten positional
/// `None`s. Crate-private — the `compile_and_run*` wrappers are the stable named surface (the
/// differential harnesses call them by name); this just collapses their shared internal plumbing.
struct RunOpts<'a> {
    /// Low bytes to seed the guest window with (and, with `snapshot_cap`, capture back) — escape-oracle.
    init_mem: Option<&'a [u8]>,
    /// Reservation policy: mask domain is `[0, 2^reserved_log2)`; `0` ⇒ fully mapped (reserved == size).
    reserved_log2: u8,
    /// What of the window to snapshot back ([`CompiledModule::run`]): `None`, nothing; `Some(cap)`,
    /// the backed prefix, widened to `cap` bytes.
    snapshot_cap: Option<usize>,
    /// §14 module resolver for a guest `instantiate` of a separate-module child.
    resolve_module: Option<ModuleResolver>,
    /// The detached-child powerbox builder/releaser ([`GrantChildHooks`], PROCESS.md S2).
    grant_child: Option<GrantChildHooks>,
    /// §5 async kill-path cell, host-written and polled at safepoints.
    interrupt: Option<*const AtomicU64>,
    /// Safepoint-anchored counted-fuel budget cell (a deterministic guest budget).
    fuel: Option<*mut FuelCell>,
    /// §9/D45 devirtualized fast cap resolver.
    fast_resolver: Option<FastCapResolver>,
    /// #2113 — the run's own budget node: what its vCPUs, fibers and processes are charged to
    /// (`None`: charged to nothing, a host-less run's).
    node: Option<std::sync::Arc<dyn BudgetNode>>,
    /// #932 — the async-signal delivery arm (host `armed` flag + take/return thunks + ctx) the
    /// safepoints poll; `None` ⇒ no signal check emitted.
    signal: Option<SignalArm>,
}

impl Default for RunOpts<'_> {
    fn default() -> Self {
        RunOpts {
            init_mem: None,
            // The default reservation policy (§4): a large reserved range, only `mapped` backed.
            reserved_log2: temen_ir::default_reserved_log2(),
            snapshot_cap: None,
            resolve_module: None,
            grant_child: None,
            interrupt: None,
            fuel: None,
            fast_resolver: None,
            node: None,
            signal: None,
        }
    }
}

fn run_inner(
    m: &IrModule,
    func: FuncIdx,
    args: &[i64],
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    opts: RunOpts,
) -> Result<(JitOutcome, Vec<u8>), JitError> {
    let RunOpts {
        init_mem,
        reserved_log2,
        snapshot_cap,
        resolve_module,
        grant_child,
        interrupt,
        fuel,
        fast_resolver,
        node,
        signal,
    } = opts;
    // The historical one-shot lifecycle, now compile → run over the long-lived split
    // (DESIGN.md §22): `CompiledModule` owns the `JITModule` for the whole run and the
    // executable memory is freed when it drops, after `run` returns — behavior-identical
    // to the old inline compile→run→drop.
    #[cfg_attr(not(fiber_rt), allow(unused_mut))]
    let mut cm = CompiledModule::compile_with_signal(
        m,
        func,
        cap_thunk,
        cap_ctx,
        reserved_log2,
        resolve_module,
        interrupt,
        fuel,
        fast_resolver,
        node,
        0,      // one-shot path: natural table size (no B2 reservation)
        signal, // #932 — the one-shot signal-delivery arm (None on every non-signal entry)
        None,
    )?;
    // PROCESS.md S2 (JIT parity): install the child-powerbox host callbacks into the §14 nursery
    // before the guest runs (the nursery only exists when the module holds an `Instantiator`). `None`
    // leaves a spawn an inert `CapFault`.
    // Through the one setter, so the nursery hooks and the domain's lane are installed together
    // (they come from the same host and must agree).
    cm.set_grant_child_hooks(grant_child);
    cm.run(args, init_mem, snapshot_cap)
}

/// A [`JITModule`] whose memory is actually **released on drop**. cranelift-jit deliberately
/// leaks all code memory when a bare `JITModule` drops (its `Memory` `mem::forget`s every
/// allocation so stale `fn` pointers can never fault) — reclaiming it requires the explicit
/// `unsafe free_memory()`, which this crate never called. That leak is per *compile*, and the
/// [`ArenaMemoryProvider`] both compile paths install reserves **256 MiB per module** — which on
/// Windows is `MEM_RESERVE | MEM_COMMIT` (the region crate commits eagerly; see the note in
/// cranelift's `arena.rs`), i.e. every compile permanently charged 256 MiB against the system
/// commit limit. A compile-heavy process (the differential fuzz loops, `durable_jit`
/// freeze/thaw, a §14 nursery, a REPL) pinned the runner's commit ceiling within ~dozens of
/// compiles, after which *unrelated* heap allocations abort (`memory allocation of N bytes
/// failed` → `0xc0000409`) and window commits fail (`os error 1455`) — the ISSUES.md **I3**
/// Windows CI flake family. On unix, overcommit hid the same leak as unbounded VA growth.
///
/// Freeing on drop is sound because the owner ([`CompiledModule`], root or §14 child) already pins
/// the lifetime contract "nothing that points into the code may outlive this struct": the field
/// is declared last, so runtimes/tables/trampolines drop first, and no fiber, thread, or
/// installed table entry survives the owner (documented on the structs). Instances sharing code
/// ([`SharedCode`]) each hold the arena's `Arc`, so it is freed with the last of them.
struct OwnedJit(Option<JITModule>);

impl OwnedJit {
    fn new(m: JITModule) -> OwnedJit {
        OwnedJit(Some(m))
    }
}

/// #1825 — the code arena, to extend it (`define_extra`, the hosting trampolines): only while one
/// instance holds it alone. Instances sharing code ([`SharedCode`]) run the same finalized bytes, on
/// several threads at once, so none of them may add to it. The one way to the `JITModule` of an arena
/// that is already an instance's.
fn arena_mut(module: &mut Arc<OwnedJit>) -> Result<&mut JITModule, JitError> {
    let arena = Arc::get_mut(module).ok_or(JitError::Unsupported(
        "this code is shared with another instance, so it cannot be extended",
    ))?;
    Ok(arena.0.as_mut().expect("JITModule present until drop"))
}

// SAFETY: a shared `&OwnedJit` reaches nothing. Its `JITModule` is touched only through `&mut`
// (`arena_mut`, on a unique `Arc`, and `Drop`), so instances sharing an arena on several threads
// (#1825) share only the finalized code they execute.
unsafe impl Sync for OwnedJit {}

impl Drop for OwnedJit {
    fn drop(&mut self) {
        if let Some(m) = self.0.take() {
            // SAFETY: per the owners' documented contract, by the time this field drops no guest
            // code from this module is executing and no `fn` pointer into it is ever called again.
            unsafe { m.free_memory() };
        }
    }
}

/// A compiled module whose executable code **outlives a single run** (DESIGN.md §22: the
/// long-lived `JITModule` split). Owns the [`JITModule`] (the executable memory lives until
/// drop), the power-of-two-padded function table, the entry's buffer-ABI trampoline, and the
/// §12/§14 runtimes its instance's code runs on. Produced by
/// [`CompiledModule::compile`]; [`CompiledModule::run`] executes the entry over a fresh guest
/// window (allocated per run — the window base is threaded as a runtime argument, not baked).
///
/// [`CompiledModule::define_extra`] then declares + defines + finalizes **additional**
/// functions into the same live module — the enabling primitive for the guest-driven `Jit`
/// capability (DESIGN.md §22). Extra functions are *thunk-reachable only*: they are **never**
/// installed in the function table, so the table mask baked into every `call.dyn` site
/// (`fn_table_mask`) never changes — the escape-relevant dispatch is byte-identical to the
/// one-shot path.
/// One function produced by [`CompiledModule::define_extra`]: its buffer-ABI **trampoline**
/// (for `invoke` over a fresh/live window, any arity) and its natural-ABI **entry** + interned
/// **`type_id`** (for B2 [`CompiledModule::install`] into the `call.dyn` table). Pointers
/// are valid for the life of the `CompiledModule`.
#[derive(Clone, Copy)]
pub struct DefinedFn {
    pub tramp: *const u8,
    pub code: *const u8,
    pub type_id: u32,
}

/// `(func, block, inst) → index into the module's `debug_info.locs`` — the source-loc lookup the
/// JIT codegen consults to stamp each emitted op with a `cranelift SourceLoc` (W5 JIT/DWARF tier,
/// Stage 0). Built once per compile only when the module carries `-g` debug info.
type SrcLocMap = std::collections::HashMap<(u32, u32, u32), u32>;

/// Per-function captured `MachSrcLoc` ranges before address resolution: `(func index, [(start
/// offset, end offset, `debug_info.locs` index)])`, relative to the function's start until
/// `finalize` resolves the base address (W5 JIT/DWARF Stage 1).
type RawSrcLocs = Vec<(u32, Vec<(u32, u32, u32)>)>;

/// One source variable's machine-location ranges before address resolution (W5 JIT/DWARF Stage 3a):
/// `[(start offset, end offset, location)]`, relative to its function's start until `finalize`.
type VarLocRanges = Vec<(u32, u32, VarMachineLoc)>;

/// Per-function captured value-label ranges before address resolution (Stage 3a): `(func index,
/// [(value-label id, ranges)])`.
type RawVarLocs = Vec<(u32, Vec<(u32, VarLocRanges)>)>;

/// A machine-address → source range in finalized JIT code (W5 JIT/DWARF tier, Stage 1): the absolute
/// `[lo, hi)` code range a source position covers, resolved after `finalize` from Cranelift's
/// per-function `MachSrcLoc` ranges (relative offsets) + the function's finalized base address.
/// Strippable host-side tooling, untrusted-for-escape (§2a) — never read by running guest code.
#[derive(Clone, Copy, Debug)]
pub struct SrcRange {
    pub lo: u64,
    pub hi: u64,
    pub func: u32,
    pub file: u32,
    pub line: u32,
    pub col: u32,
}

/// A symbolized JIT code address (W5 JIT/DWARF tier): the source position [`CompiledModule::
/// symbolize`] resolves a machine `pc` to.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct JitFrameLoc {
    pub func: u32,
    /// The source function name (`debug_info.func_names`), or `None` when the module carried no name
    /// for `func` — renderers fall back to `fn{func}`.
    pub func_name: Option<String>,
    pub file: String,
    pub line: u32,
    pub col: u32,
}

/// Symbolize a captured trap stack into source frames (§5 W3 Stage 1): the innermost frame from the
/// faulting `pc`, then one per **return address** in `rets` (the frame-pointer chain the guard
/// handler walked while the guest stack was intact), stopping at the first that isn't guest code per
/// `sym`. A return address points at the instruction *after* the call, so callers are symbolized at
/// `ret - 1` — landing inside the call instruction, the caller's real source position (the standard
/// backtrace adjustment); the innermost `pc` is the faulting instruction itself, symbolized
/// directly. Adjacent duplicate positions are collapsed, as in [`CompiledModule::fiber_backtrace`].
/// Pure (the handler already did the stack reads): the seed for [`CompiledModule::trap_backtrace`].
fn symbolize_capture(
    pc: usize,
    rets: &[usize],
    sym: impl Fn(usize) -> Option<JitFrameLoc>,
) -> Vec<JitFrameLoc> {
    let mut frames: Vec<JitFrameLoc> = Vec::new();
    if let Some(loc) = sym(pc) {
        frames.push(loc);
    }
    for &ret in rets {
        match sym(ret.wrapping_sub(1)) {
            Some(loc) => {
                if frames.last() != Some(&loc) {
                    frames.push(loc);
                }
            }
            None => break, // reached the host boundary (the outermost guest frame's caller) — stop.
        }
    }
    frames
}

#[cfg(test)]
mod trap_capture_tests {
    use super::{symbolize_capture, JitFrameLoc};

    fn loc(func: u32, line: u32) -> JitFrameLoc {
        JitFrameLoc {
            func,
            func_name: None,
            file: "f.c".into(),
            line,
            col: 0,
        }
    }

    #[test]
    fn symbolizes_pc_directly_callers_at_ret_minus_one_and_stops_at_host() {
        // pc → func2; ret0-1 → func1; ret1-1 → func0; ret2-1 → host (None, stop). A guest entry
        // is fed `pc` directly but callers via `ret - 1` (the byte inside the call instruction).
        let sym = |a: usize| match a {
            0x100 => Some(loc(2, 5)),  // pc (faulting instruction)
            0x199 => Some(loc(1, 10)), // ret0 (0x19a) - 1
            0x299 => Some(loc(0, 20)), // ret1 (0x29a) - 1
            _ => None,
        };
        let frames = symbolize_capture(0x100, &[0x19a, 0x29a, 0x999], sym);
        assert_eq!(
            frames,
            vec![loc(2, 5), loc(1, 10), loc(0, 20)],
            "three guest frames, host trimmed"
        );
    }

    #[test]
    fn collapses_adjacent_duplicate_positions() {
        // A recursive self-call: the same source position on consecutive frames collapses to one.
        let sym = |a: usize| match a {
            0x10 | 0x1f | 0x2f => Some(loc(0, 7)),
            _ => None,
        };
        let frames = symbolize_capture(0x10, &[0x20, 0x30, 0x40], sym);
        assert_eq!(
            frames,
            vec![loc(0, 7)],
            "duplicate adjacent positions collapse"
        );
    }
}

/// Owner decision 2026-07-24 — **domain-teardown semantics** (DESIGN.md §12; ISSUES.md I37; D37
/// death-is-revocation): `exit(n)` / any trap from ANY vCPU, and the **root vCPU's completion**
/// (plain clean return included), tear the whole domain down — PARKED siblings included — and the
/// run returns promptly with the root's outcome. Regression fixtures for the hang where a daemon
/// parked on an indefinite `atomic.wait` (or a `thread.join`) kept `join_all` blocked forever
/// because nothing woke it to observe the domain's end. Gated to the targets with a JIT thread
/// runtime (like `crates/temen/tests/jit_threads.rs`).
#[cfg(test)]
#[cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]
mod domain_teardown_tests {
    use super::{compile_and_run, compile_and_run_with_host, JitOutcome, TrapKind, EXIT_CODE};
    use std::time::{Duration, Instant};
    use temen_text::parse_module;
    use temen_verify::verify_module;

    /// The teardown latency bound: generous for CI (compile included), but far below the
    /// MAX-WAIT scale of the hang this guards against (the daemon parks with timeout −1).
    const PROMPT: Duration = Duration::from_secs(15);

    /// Func 1 in every fixture: a daemon vCPU that parks forever on `i32.atomic.wait` (timeout −1)
    /// at address 16384 (one guard up — #1094's unconditional NULL guard unmaps `[0, 16384)`, so the
    /// futex word must sit above it), whose word never changes and which nothing ever notifies.
    const PARKED_DAEMON: &str = "func (i64, i64) -> (i64) {\n\
        block 0 (vsp: i64, v0: i64) {\n\
        \x20 v1 = i64.const 16384\n\
        \x20 v2 = i32.const 0\n\
        \x20 v3 = i64.const -1\n\
        \x20 v4 = i32.atomic.wait v1 v2 v3\n\
        \x20 v5 = i64.const 7\n\
        \x20 return v5\n\
          }\n\
        }\n";

    fn run_promptly(src: &str) -> JitOutcome {
        let m = parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
        let start = Instant::now();
        let out = compile_and_run(&m, 0, &[]).expect("jit compile/run");
        let elapsed = start.elapsed();
        assert!(
            elapsed < PROMPT,
            "domain teardown must be prompt, took {elapsed:?}"
        );
        out
    }

    /// Rule 2: the **root's clean return** ends the run — the parked daemon (indefinite
    /// `atomic.wait`) is torn down, its outcome unobservable, and the result is the root's own
    /// return value. Before the fix this hung: nothing re-evaluated the parked daemon's wait when
    /// the root finished.
    #[test]
    fn root_return_tears_down_parked_daemon() {
        let src = format!(
            "memory 16\n\
             func () -> (i64) {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = i64.const 16392\n\
             \x20 v3 = i32.const 0\n\
             \x20 v4 = i64.const 10000000\n\
             \x20 v5 = i32.atomic.wait v2 v3 v4\n\
             \x20 v6 = i64.const 42\n\
             \x20 return v6\n\
               }}\n\
             }}\n\
             {PARKED_DAEMON}"
        );
        match run_promptly(&src) {
            JitOutcome::Returned(r) => assert_eq!(r, vec![42]),
            other => panic!("expected Returned([42]), got {other:?}"),
        }
    }

    /// Rule 1 (`exit`): the root calls the exit capability while the daemon is parked forever —
    /// the whole domain unwinds and the run reports `Exited` promptly. Before the fix this hung:
    /// the exit code sat in the trap cell but the parked daemon was never woken to observe it.
    #[test]
    fn exit_tears_down_parked_daemon() {
        /// A one-op powerbox: `call.cap 1 0` is `exit(code)` — raise `EXIT_CODE | code << 32`,
        /// exactly as temen-run's host thunk does for `Trap::Exit`.
        unsafe extern "C" fn exit_thunk(
            _ctx: *mut core::ffi::c_void,
            _mem_base: *mut u8,
            _mem_size: u64,
            _mem_reserved: u64,
            _type_id: u32,
            _op: u32,
            _handle: i32,
            args: *const i64,
            _n_args: u64,
            _results: *mut i64,
            _n_results: u64,
            vm: &crate::VmCtx,
        ) {
            let code = *args as i32;
            vm.raise(EXIT_CODE as i64 | ((code as i64) << 32));
        }

        let src = format!(
            "memory 16\n\
             func () -> () {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = i64.const 16392\n\
             \x20 v3 = i32.const 0\n\
             \x20 v4 = i64.const 10000000\n\
             \x20 v5 = i32.atomic.wait v2 v3 v4\n\
             \x20 v6 = i32.const 0\n\
             \x20 v7 = i32.const 0\n\
             \x20 v8 = call.cap 1 0 (i32) -> (i64) v6 (v7)\n\
             \x20 unreachable\n\
               }}\n\
             }}\n\
             {PARKED_DAEMON}"
        );
        let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
        verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
        let start = Instant::now();
        let out = compile_and_run_with_host(&m, 0, &[], exit_thunk, core::ptr::null_mut())
            .expect("jit compile/run");
        let elapsed = start.elapsed();
        assert!(
            elapsed < PROMPT,
            "domain teardown must be prompt, took {elapsed:?}"
        );
        match out {
            JitOutcome::Exited(code) => assert_eq!(code, 0),
            other => panic!("expected Exited(0), got {other:?}"),
        }
    }

    /// Rule 1 (trap): a root trap (`unreachable`) tears down the parked daemon and surfaces the
    /// root's trap promptly — same wake path as `exit`, with a `TrapKind` in the cell instead.
    #[test]
    fn root_trap_tears_down_parked_daemon() {
        let src = format!(
            "memory 16\n\
             func () -> (i64) {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = i64.const 16392\n\
             \x20 v3 = i32.const 0\n\
             \x20 v4 = i64.const 10000000\n\
             \x20 v5 = i32.atomic.wait v2 v3 v4\n\
             \x20 unreachable\n\
               }}\n\
             }}\n\
             {PARKED_DAEMON}"
        );
        match run_promptly(&src) {
            JitOutcome::Trapped(k) => assert_eq!(k, TrapKind::Unreachable),
            other => panic!("expected Trapped(Unreachable), got {other:?}"),
        }
    }

    /// Rule 2 for the **`thread.join` park**: a sibling blocked joining the forever-parked daemon
    /// (handle 0, the first spawn) must also unwind when the root returns — the join park site has
    /// its own teardown check and its own wake.
    #[test]
    fn root_return_tears_down_parked_joiner() {
        let src = format!(
            "memory 16\n\
             func () -> (i64) {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = thread.spawn 2 v0 v0\n\
             \x20 v3 = i64.const 16392\n\
             \x20 v4 = i32.const 0\n\
             \x20 v5 = i64.const 10000000\n\
             \x20 v6 = i32.atomic.wait v3 v4 v5\n\
             \x20 v7 = i64.const 42\n\
             \x20 return v7\n\
               }}\n\
             }}\n\
             {PARKED_DAEMON}\
             func (i64, i64) -> (i64) {{\n\
             block 0 (vsp: i64, v0: i64) {{\n\
             \x20 v1 = i32.const 0\n\
             \x20 v2 = thread.join v1\n\
             \x20 return v2\n\
               }}\n\
             }}\n"
        );
        match run_promptly(&src) {
            JitOutcome::Returned(r) => assert_eq!(r, vec![42]),
            other => panic!("expected Returned([42]), got {other:?}"),
        }
    }

    /// #2498 — a powerbox whose calls are in flight when the domain ends. Op 0 is `exit(code)`.
    /// Op 1 answers once the domain has ended — a sibling's call in flight when the end lands — and
    /// succeeds, which leaves the end in place (#2499). Op 2 waits the same way, then traps. Each
    /// waits at most half of [`PROMPT`], so a run that never ends its domain still ends, with
    /// whatever outcome it then reports.
    unsafe extern "C" fn in_flight_thunk(
        _ctx: *mut core::ffi::c_void,
        _mem_base: *mut u8,
        _mem_size: u64,
        _mem_reserved: u64,
        _type_id: u32,
        op: u32,
        _handle: i32,
        args: *const i64,
        _n_args: u64,
        _results: *mut i64,
        _n_results: u64,
        vm: &crate::VmCtx,
    ) {
        if op == 0 {
            vm.raise(EXIT_CODE as i64 | ((*args as i32 as i64) << 32));
            return;
        }
        let start = Instant::now();
        while vm.ended().is_none() && start.elapsed() < PROMPT / 2 {
            std::thread::sleep(Duration::from_millis(1));
        }
        if op == 2 {
            vm.raise(TrapKind::CapFault as i64);
        }
    }

    /// Run `src` over [`in_flight_thunk`] on a thread of its own, and fail if it has not ended
    /// within [`PROMPT`]: the hang this guards against would otherwise hang the test too.
    fn run_in_flight(src: String) -> JitOutcome {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let m = parse_module(&src).unwrap_or_else(|e| panic!("parse: {e:?}\n{src}"));
            verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{src}"));
            let out = compile_and_run_with_host(&m, 0, &[], in_flight_thunk, core::ptr::null_mut());
            let _ = tx.send(out.expect("jit compile/run"));
        });
        rx.recv_timeout(PROMPT)
            .unwrap_or_else(|_| panic!("the run did not end within {PROMPT:?}"))
    }

    /// The root half of every in-flight fixture: spawn func 1, wait until it has stored `1` at
    /// 16392 (it does just before its host call), then `end`.
    fn root_then(ret: &str, end: &str) -> String {
        format!(
            "memory 16\n\
             func () -> ({ret}) {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = i64.const 16392\n\
             \x20 v3 = i32.const 0\n\
             \x20 v4 = i64.const -1\n\
             \x20 v5 = i32.atomic.wait v2 v3 v4\n\
             \x20 {end}\n\
               }}\n\
             }}\n"
        )
    }

    /// Func 1 of every in-flight fixture: store `1` at 16392 and notify the root, make host call
    /// `op`, then `then`, with `blocks` after.
    fn sibling_calls(op: u32, then: &str, blocks: &str) -> String {
        format!(
            "func (i64, i64) -> (i64) {{\n\
             block 0 (vsp: i64, v0: i64) {{\n\
             \x20 v1 = i64.const 16392\n\
             \x20 v2 = i32.const 1\n\
             \x20 i32.atomic.store v1 v2\n\
             \x20 v3 = atomic.notify v1 v2\n\
             \x20 v4 = i32.const 0\n\
             \x20 v5 = call.cap 1 {op} (i32) -> (i64) v4 (v4)\n\
             \x20 {then}\n\
               }}\n\
             {blocks}\
             }}\n"
        )
    }

    /// #2498 — the root's completion ends a sibling whose host call is in flight at teardown: the
    /// call answers into a domain that has ended, and the sibling unwinds there instead of spinning
    /// on (nim's `parfor` corpus case: an idle pool worker's host call). That call used to reset
    /// the trap cell, so the sibling never saw the end and `join_all` waited for it forever.
    #[test]
    fn root_return_tears_down_a_sibling_whose_host_call_answers_after_teardown() {
        let src = format!(
            "{}{}",
            root_then("i64", "v6 = i64.const 42\n  return v6"),
            sibling_calls(1, "br 1()", "block 1 () {\n  br 1()\n  }\n")
        );
        match run_in_flight(src) {
            JitOutcome::Returned(r) => assert_eq!(r, vec![42]),
            other => panic!("expected Returned([42]), got {other:?}"),
        }
    }

    /// #2498 — and one that would park after such a call: it unwinds rather than parking, which a
    /// deadlock check read as `ThreadFault` for a root that returned cleanly.
    #[test]
    fn root_return_tears_down_a_sibling_that_would_park_after_its_host_call() {
        let src = format!(
            "{}{}",
            root_then("i64", "v6 = i64.const 42\n  return v6"),
            sibling_calls(
                1,
                "v6 = i64.const 16400\n  v7 = i32.const 0\n  v8 = i64.const -1\n  \
                 v9 = i32.atomic.wait v6 v7 v8\n  v10 = i64.const 7\n  return v10",
                ""
            )
        );
        match run_in_flight(src) {
            JitOutcome::Returned(r) => assert_eq!(r, vec![42]),
            other => panic!("expected Returned([42]), got {other:?}"),
        }
    }

    /// #2498/#2499 — the root's `exit` survives a sibling's host call that answers after it: the
    /// call leaves the exit in the cell, and the run reports it.
    #[test]
    fn root_exit_survives_a_sibling_host_call_that_answers_after_it() {
        let src = format!(
            "{}{}",
            root_then(
                "",
                "v6 = i32.const 0\n  v7 = i32.const 3\n  \
                 v8 = call.cap 1 0 (i32) -> (i64) v6 (v7)\n  unreachable"
            ),
            sibling_calls(1, "v6 = i64.const 7\n  return v6", "")
        );
        match run_in_flight(src) {
            JitOutcome::Exited(code) => assert_eq!(code, 3),
            other => panic!("expected Exited(3), got {other:?}"),
        }
    }

    /// #2499 — what a sibling does once the root has returned is unobservable (invariant 6): a trap
    /// its host call raises after teardown loses to the end teardown raised, and the run reports the
    /// root's result. It used to be stored over that end and reported instead.
    #[test]
    fn a_sibling_trap_after_the_root_returned_leaves_the_roots_result() {
        let src = format!(
            "{}{}",
            root_then("i64", "v6 = i64.const 42\n  return v6"),
            sibling_calls(2, "v6 = i64.const 7\n  return v6", "")
        );
        match run_in_flight(src) {
            JitOutcome::Returned(r) => assert_eq!(r, vec![42]),
            other => panic!("expected Returned([42]), got {other:?}"),
        }
    }

    /// #2499 — the root's own fault is what ends its domain: the run reports it, with a sibling
    /// still parked.
    #[test]
    fn a_root_fault_ends_its_domain_with_the_fault() {
        let src = format!(
            "memory 16\n\
             func () -> (i64) {{\n\
             block 0 () {{\n\
             \x20 v0 = i64.const 0\n\
             \x20 v1 = thread.spawn 1 v0 v0\n\
             \x20 v2 = i64.load v0\n\
             \x20 return v2\n\
               }}\n\
             }}\n\
             {PARKED_DAEMON}"
        );
        match run_promptly(&src) {
            JitOutcome::Trapped(k) => assert_eq!(k, TrapKind::MemoryFault),
            other => panic!("expected Trapped(MemoryFault), got {other:?}"),
        }
    }
}

/// `(func, block) → [(block-local value index, value-label id)]` — the source variables to stamp
/// with a Cranelift `ValueLabel` during lowering (W5 JIT/DWARF Stage 3a). Built once per compile from
/// the §6 `debug_info.vars`' SSA-resident `VarLoc`s, only when the module carries `-g`. The label id
/// is the variable's index into [`CompiledModule::var_locs`]; the block-local value index maps onto
/// the JIT's per-block value map (see [`lower_block`]).
type VarLabelMap = std::collections::HashMap<(u32, u32), Vec<(u32, u32)>>;

/// Where a source variable's value physically lives over a machine-code range (W5 JIT/DWARF Stage
/// 3a) — Cranelift's `LabelValueLoc` translated to DWARF terms, the bridge to a `DW_AT_location`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarMachineLoc {
    /// In machine register `dwarf_reg` (a DWARF register number, via `map_regalloc_reg_to_dwarf`) —
    /// emits `DW_OP_regN` (Stage 3c).
    Reg(u16),
    /// At canonical-frame-address + `off` — emits `DW_OP_call_frame_cfa` + offset / `DW_OP_fbreg`.
    CfaOffset(i64),
}

/// One `[lo, hi)` absolute machine-code range over which a source variable lives in [`VarMachineLoc`]
/// (W5 JIT/DWARF Stage 3a) — one entry of a DWARF location list.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VarRange {
    pub lo: u64,
    pub hi: u64,
    pub loc: VarMachineLoc,
}

/// A source variable's machine-location list in finalized JIT code (W5 JIT/DWARF Stage 3a): the
/// `value_labels_ranges` Cranelift produced for the CLIF value that backs it, resolved to absolute
/// pcs. Empty `ranges` ⇒ the optimizer dropped the value everywhere (gdb will show `<optimized
/// out>`). Host-side tooling, off the runtime path (§2a).
#[derive(Clone, Debug)]
pub struct VarMachineInfo {
    pub func: u32,
    pub name: String,
    /// The variable's structured type as a `debug_info.types` index (Stage 3b), or `None` when the
    /// module carried only a render-name. The Stage 3c `DW_TAG_variable` points `DW_AT_type` at it.
    pub type_id: Option<u32>,
    pub ranges: Vec<VarRange>,
}

pub struct CompiledModule {
    /// The padded function table `call.dyn` dispatches through. Its address is threaded as
    /// a runtime argument (not baked), but running code reads it — boxed so it never moves, and
    /// declared before `module` so drop order matches the old `drop(fn_table); drop(module)`.
    fn_table: Box<[FnEntry]>,
    /// The entry's buffer-ABI trampoline (finalized code, owned by `module`).
    tramp_code: *const u8,
    /// D66 — the entry's **limit-taking** trampoline (`build_trampoline(.., with_limit = true)`), the
    /// one a child-domain task's fiber body enters through with its own stack low bound. Only a
    /// `compile_child_windowed` product has one; null on a root (`compile`) module.
    tramp_code_limited: *const u8,
    /// I36 slice 3 (§3.6 JIT serve loop) — one buffer-ABI trampoline per **impl-export
    /// handler**: `(funcidx, code, n_params, n_results)`, invocable over the live window via
    /// [`Self::invoke_extra`] from the embedder's serve arm ([`Self::handler_tramp`]). Empty
    /// for a module with no impl exports.
    serve_tramps: Vec<(u32, *const u8, usize, usize)>,
    /// The entry's parameter count — `run` rejects shorter `args` (the trampoline reads
    /// exactly this many slots).
    n_params: usize,
    /// The entry's result count (`run` sizes the result buffer).
    n_results: usize,
    /// The module's own function count — where the `fn_table`'s installable padding begins.
    /// `uninstall` refuses to clear a slot `< n_real_funcs` (a real module function), so a guest
    /// can only reclaim slots it `install`ed.
    n_real_funcs: usize,
    // --- the baked lowering environment, reused verbatim by `define_extra` so an extra
    // --- function compiles against the *same* constants as the parent (same confinement
    // --- mask, same cap thunk, same table mask — DESIGN.md §22 "vmctx sharing").
    distinct: Vec<FuncType>,
    /// §9/D45 — the resolver the code's fast-path `call.cap`s were claimed by.
    fast_resolver: Option<FastCapResolver>,
    /// The instance has a §12 fiber runtime its code's `cont.*` sites call into.
    fibers: bool,
    /// The instance has a §12 thread domain its code's futex and `thread.*` sites run on.
    threads: bool,
    /// The code's `thread.spawn`/`join` sites have a domain to spawn in: always with
    /// [`Self::threads`] on a root; on a §14 child only when it uses them, which then stands up a
    /// domain of its own (`child_exec`).
    spawns: bool,
    /// The code lowers its `Instantiator` `call.cap`s to the §14 nesting runtime, through the
    /// instance's nursery.
    nesting: bool,
    mask: u64,
    /// Whether this module's code polls a kill-path cell / charges a fuel cell (see [`Lower::epoch`]
    /// / [`Lower::fuel`]) — the code's shape, which `define_extra` compiles new units to match.
    epoch: bool,
    fuel: bool,
    /// #2164 — the page check a demand-paged child's code makes before each access (see
    /// [`PageEnv`]), which `define_extra` compiles new units to match. [`PageEnv::NONE`] otherwise.
    page: PageEnv,
    /// #1768 — the addresses this module's **own** instance fills its [`VmCtx`] from: the powerbox
    /// ctx, kill-path cell, fuel cell and signal arm the compile was handed. Nothing reads them from
    /// the code; each run mints its context from these ([`VmCtx::new`]), and a run of the same code
    /// as another instance supplies its own.
    instance: InstanceAddrs,
    /// The `call.dyn` index mask fixed at compile time (`next_pow2(n_funcs) - 1`) and baked
    /// into every call site. `define_extra` compiles new units against this same constant.
    fn_table_mask: u64,
    /// Monotonic counter for unique `declare_function` symbol names across `define_extra` calls.
    next_extra: usize,
    /// Cumulative machine-code bytes of every `define_extra`'d function + trampoline lowered over
    /// this module's life — a **byte-accurate** code-arena occupancy measure (the actual emitted
    /// code size, summed at finalize; the dominant term in arena consumption). Like `next_extra` it
    /// only grows, and restarts near zero in a freshly-compacted module. An embedder watermarks on
    /// [`Self::extra_byte_count`] to auto-compact (DESIGN.md §22).
    extra_bytes: usize,
    /// Finalized machine-code bytes of the **base module** (every function body lowered by
    /// [`CompiledModule::compile`], summed at define time from Cranelift's `code_buffer`). The
    /// analogue of [`Self::extra_bytes`] for the initial compile — a byte-accurate measure of the
    /// module's emitted code size, exposed via [`Self::code_byte_count`]. Excludes the small
    /// buffer-ABI trampoline and any later `define_extra` units (those are in `extra_bytes`).
    base_bytes: usize,
    /// The in-flight run's window fault range, published by `run_code_raw` for the duration of
    /// the guarded call so a mid-run [`Self::invoke_extra`] can arm its nested recovery.
    /// `None` ⇒ no run in flight (invoke is rejected).
    live_fault_range: Option<(usize, usize)>,
    /// #1296 — a §14 **child** module: its window is minted and owned by the nursery's spawn, never by
    /// this module, so `live_fault_range` is never recorded and `invoke_extra` derives the range
    /// from the `mem_base` the thunk was handed. `false` for a root, whose `run_code_raw` records its own window.
    caller_window: bool,
    /// The source backtrace of the most recent [`Self::run`] that **trapped** (§5 W3 Stage 1):
    /// innermost guest frame first, symbolized from the trap site the guard handler captured.
    /// Empty after a non-trapping run (or a trap with no usable frame). Read via
    /// [`Self::last_trap_backtrace`]; host-side, off the runtime path (§2a).
    last_trap_backtrace: Vec<JitFrameLoc>,
    /// The guest **fiber handle** running when the most recent [`Self::run`] trapped (§5 W3 / §23-D57
    /// per-fiber attribution): `Some(handle)` for a trap inside a fiber (named even if it had migrated
    /// across vCPU threads — captured at the trap instant, not inferred from the thread), `Some(-1)`
    /// for the root computation, `None` after a clean run. Read via [`Self::last_trap_fiber`].
    last_trap_fiber: Option<i64>,
    // --- the per-run window *plan* (the window itself is allocated in `run`; the mask and
    // --- extents are baked into the code, so they were fixed at compile time).
    win_mapped: usize,
    win_reserved: usize,
    mem_size_log2: Option<u8>,
    /// #964/#1094: the module's NULL guard (`0` = unguarded). Captured at compile from
    /// [`temen_ir::module_null_guard`] (unconditional now — #1094); each run `mprotect`s `[0, guard)`
    /// inaccessible after the window is seeded, so a NULL dereference faults into the §5 guard exactly
    /// where the interpreter's `Unmapped`-seeded page map traps.
    null_guard: u64,
    /// Initialized data segments, owned so a run can seed a fresh window (the module may
    /// outlive the borrowed `IrModule`).
    data: Vec<Data>,
    /// The thaw's window the next run restores ([`WindowRestore`], the durable-restore step,
    /// DURABILITY.md §12.3); `None` (the common path) builds a fresh instance's window.
    restore: Option<WindowRestore>,
    /// #2427 — the pages a durable run's capture read past its declared window, by byte offset
    /// ([`Self::take_tail_out`]).
    tail_out: Vec<(u64, Vec<u8>)>,
    /// This run is **durable** (DURABILITY.md §12.8): the fiber runtime keeps the active shadow-SP
    /// word pointing at the running context's region (swapped on every fiber switch). `false` (the
    /// default) ⇒ an ordinary run that never touches the durable reserve. Set per-run at entry.
    durable: bool,
    /// The module-declared shadow arena (INVARIANTS.md #16) — [`ShadowArena::EMPTY`] for a module
    /// that declared none, which a durable run refuses at entry rather than placing by default.
    shadow: temen_ir::durable_abi::ShadowArena,
    /// §12.8 4A.5 stage (ii): engage the **concurrent** durable path — children spawned during NORMAL
    /// run as real OS threads with their own reserved shadow contexts and self-unwind concurrently on a
    /// freeze (vs. the single-worker deferred model). Set by the concurrent multi-vCPU interruptible
    /// entry; `false` everywhere else, so those paths are byte-identical.
    concurrent_durable: bool,
    /// Durable **thaw** seed (slice 3.3.3): frozen fibers to re-create in the table before a
    /// `REWINDING` run, so a thaw `cont.resume` re-enters them. Empty for a freeze / ordinary run.
    frozen_seed: Vec<FrozenFiber>,
    /// Durable **freeze** residue (slice 3.3.3): the fibers the freeze driver flattened this run,
    /// read back by the durable entry point. Empty unless a freeze flattened fibers.
    frozen_out: Vec<FrozenFiber>,
    /// Durable **freeze** residue (slice 3.3): the spawned vCPUs that unwound under the freeze (each
    /// run inline single-worker by `thread.spawn`). Drained from the `Domain` after the root unwinds,
    /// read back by the durable entry point. Empty unless a freeze caught a live child.
    frozen_vcpus_out: Vec<FrozenVCpu>,
    /// Durable **freeze** residue (slice 3.3): the **root** vCPU's flattened shadow-SP extent, captured
    /// after the freeze driver (before the children overwrite the shared active-SP word). The root's
    /// continuation rides in the window image like everyone's, but the active-SP word ends at the last
    /// child's extent, so the root's own extent is reported separately for a thaw to restore. `0` on a
    /// non-freeze run.
    frozen_root_sp_out: u64,
    /// Durable **thaw** seed (slice 3.3): the spawned vCPUs to re-attach + run before the root re-enters
    /// under `REWINDING`. Empty for a freeze / ordinary run.
    frozen_vcpu_seed: Vec<FrozenVCpu>,
    /// #1361 step 4 — the durable **detached** children a freeze reached, after teardown (see
    /// [`DetachedHarvest`]); taken by the embedder. Empty unless a freeze caught one.
    detached_out: Vec<DetachedHarvest>,
    /// #1361 step 4 — the captured detached children a **thaw** re-launches before the root re-enters.
    detached_seed: Vec<DetachedSeed>,
    /// Durable **thaw** input (slice 3.3): the root vCPU's restored shadow-SP extent (from the
    /// artifact), set as the active word before the root rewinds. The empty root extent (`ShadowArena::frame_base(0)`) otherwise.
    thaw_root_sp: u64,
    /// Async freeze controller (Phase-4 Slice A, 4A.3): if set, the run publishes its live window base
    /// here before the guarded call and retires it after, so a controller thread's `request_freeze`
    /// can write `UNWINDING` into the running window. `None` for every non-interruptible run.
    freeze_ctl: Option<Arc<FreezeController>>,
    /// #1810, #2427 — a durable run's capture reads the pages the guest holds past its declared
    /// window (see [`HeldTail`]).
    held_tail: Option<(HeldTail, *mut core::ffi::c_void)>,
    /// #2025 — the embedder's hook over a fresh root window, before the guest runs
    /// ([`Self::set_window_ready`]).
    window_ready: Option<(WindowReady, *mut core::ffi::c_void)>,
    /// #1768 — the embedder's fork hook ([`Self::set_fork_hook`]): set on a run armed to fork, whose
    /// root unwinds at a fork call for the hook to duplicate. `None` for every other run.
    fork_hook: Option<ForkHook>,
    // --- §12/§14 runtimes this instance's code runs on, which its vmctx names or an entry publishes;
    // --- they must live exactly as long as the instance can run. Each instance has its own
    // --- ([`Self::instance`], temen#2389).
    #[cfg(fiber_rt)]
    fiber_rt: Option<Box<fiber_rt::FiberRuntime>>,
    #[cfg(fiber_rt)]
    domain: Option<Box<os_thread_rt::Domain>>,
    /// Kept alive because the instance names it ([`VmCtx::nursery`]) for its `Instantiator` sites.
    #[cfg(fiber_rt)]
    _nursery: Option<Box<instantiator_rt::Nursery>>,
    /// Kept alive because the instance names it ([`VmCtx::setjmp`]) for its `SetJmp`/`LongJmp` sites
    /// (LLVM.md §"JIT `longjmp`"). Holds the per-run host `jmp_buf` table.
    #[cfg(setjmp_rt)]
    _setjmp_rt: Option<Box<setjmp_rt::SetjmpRuntime>>,
    /// temen#2389 — the `call.cap` thunk and module resolver a root's §14 nursery spawns with: what
    /// [`Self::instance`] stands an instance's own nursery up over.
    #[cfg(fiber_rt)]
    cap_thunk: CapThunk,
    #[cfg(fiber_rt)]
    resolve_module: Option<ModuleResolver>,
    #[cfg(fiber_rt)]
    call_tramp: Option<fiber_rt::FiberCallTramp>,
    /// `(fiber_type_id, fiber_mask)` when the module uses `cont.*` — the per-vCPU fiber config
    /// spawned vCPUs build their runtimes from (`Domain::set_env`).
    #[cfg(fiber_rt)]
    fiber_cfg: Option<(u32, u64)>,
    /// The **domain-shared fiber table** (D57 3b-ii) the root's `fiber_rt` and every spawned
    /// vCPU's runtime are built over — one handle namespace + one fiber ceiling per domain.
    #[cfg(fiber_rt)]
    fiber_table: Option<std::sync::Arc<fiber_rt::SharedFiberTable>>,
    /// #2113 — the run's own budget node ([`Self::compile`]'s `node`), which the fiber table and
    /// thread domain the hosting enablers stand up later charge too.
    #[cfg(fiber_rt)]
    node: Option<std::sync::Arc<dyn BudgetNode>>,
    /// Machine-address → source map for finalized code (W5 JIT/DWARF Stage 1), sorted by `lo`.
    /// Empty unless the module carried `-g` debug info. Host-side tooling, off the runtime path.
    src_ranges: Vec<SrcRange>,
    /// Source file paths (from `debug_info.files`), indexed by [`SrcRange::file`].
    src_files: Vec<String>,
    /// Source function names (`func → name`, from `debug_info.func_names`): for [`Self::symbolize`],
    /// the DWARF `DW_AT_name`, and kill messages. Empty unless the module carried `-g` names.
    func_names: std::collections::HashMap<u32, String>,
    /// Per-source-variable machine-location lists (W5 JIT/DWARF Stage 3a), resolved from Cranelift's
    /// `value_labels_ranges` after finalize. Empty unless the module carried `-g` vars. Host-side
    /// tooling, off the runtime path.
    var_locs: Vec<VarMachineInfo>,
    /// The §6 structured type graph (`debug_info.types`), emitted as `DW_TAG_*_type` DIEs (W5
    /// JIT/DWARF Stage 3b). Empty unless the module carried `-g` types. Host-side tooling.
    debug_types: Vec<temen_ir::TypeDef>,
    /// #1825 — whether a §22 hosting entry stood a runtime up after the compile
    /// ([`Self::enable_fiber_hosting`], [`Self::enable_thread_hosting`]): one [`Self::instance`] does
    /// not make, so the code cannot be shared. Every runtime the compile itself stands up (the thread
    /// domain, the fiber table and runtime, the §14 nursery, the `setjmp` table) an instance makes
    /// anew (temen#2389). A §14 child's runtimes are its task's and its nursery its spawn's
    /// (temen#2168).
    hosted: bool,
    /// temen#2168 — whether the code's own `Instantiator` sites spawn through its instance's nursery
    /// ([`VmCtx::nursery`]): every instance of it has one exactly then.
    nests: bool,
    /// Owns the executable memory — the whole point of the long-lived split. Dropped last
    /// (declaration order), after everything that points into it — and the drop **releases** the
    /// code arena back to the OS (see [`OwnedJit`]; a bare `JITModule` would leak it). Shared by every
    /// instance of the same code ([`SharedCode`]); only an instance that holds it alone can extend it
    /// ([`arena_mut`]).
    module: Arc<OwnedJit>,
}

impl CompiledModule {
    /// Symbolize a finalized-code machine address to its source position (W5 JIT/DWARF Stage 1), or
    /// `None` if `pc` is outside any source-mapped op (a non-`-g` build, a trampoline, a prologue
    /// gap). The [`SrcRange`] map is sorted and disjoint, so this is a binary search. Host-side
    /// tooling, off the running guest's path (§2a) — the seed for trap symbolization + DWARF emit.
    pub fn symbolize(&self, pc: usize) -> Option<JitFrameLoc> {
        let pc = pc as u64;
        let i = self.src_ranges.partition_point(|r| r.lo <= pc);
        let r = self.src_ranges[..i].last().filter(|r| pc < r.hi)?;
        Some(JitFrameLoc {
            func: r.func,
            func_name: self.func_names.get(&r.func).cloned(),
            file: self
                .src_files
                .get(r.file as usize)
                .cloned()
                .unwrap_or_default(),
            line: r.line,
            col: r.col,
        })
    }

    /// The finalized machine-address → source map (W5 JIT/DWARF Stage 1), sorted by address. Empty
    /// unless the module carried `-g`. For tooling/tests and the forthcoming DWARF line-program emit.
    pub fn src_ranges(&self) -> &[SrcRange] {
        &self.src_ranges
    }

    /// A backtrace of a **suspended fiber** (W5 JIT/DWARF Stage 4c — the W3-JIT fiber-rooted stack
    /// walk). Rooted at a fiber *handle* (§23/D57 migratable fibers, not the OS thread), it scans the
    /// parked fiber's live control stack `[ctx, top)` low→high (innermost frame first) and symbolizes
    /// every word that lands in this module's JIT'd guest code — each is a return address, i.e. a
    /// guest call frame. A conservative scan (like the GC-root walk) rather than a frame-pointer
    /// chase, so it is robust to the host runtime glue sitting between the guest frames and the
    /// suspend switch. Adjacent duplicate positions are collapsed. Empty if `handle` names no parked
    /// fiber or the module carried no `-g`. Host-side tooling, off the running guest's path (§2a).
    #[cfg(fiber_rt)]
    pub fn fiber_backtrace(&self, handle: i64) -> Vec<JitFrameLoc> {
        let Some(table) = self.fiber_table.as_ref() else {
            return Vec::new();
        };
        table
            .with_parked_stack(handle, |stack| {
                let mut frames: Vec<JitFrameLoc> = Vec::new();
                for w in stack.chunks_exact(8) {
                    let pc = u64::from_le_bytes(w.try_into().unwrap()) as usize;
                    if let Some(loc) = self.symbolize(pc) {
                        if frames.last() != Some(&loc) {
                            frames.push(loc);
                        }
                    }
                }
                frames
            })
            .unwrap_or_default()
    }

    /// Symbolize a JIT **trap** into a source backtrace (§5 W3 Stage 1): the innermost frame from the
    /// faulting `pc`, then one frame per guest caller from `rets` — the frame-pointer chain's return
    /// addresses the SIGSEGV/SIGBUS handler walked *while the guest stack was intact* and stashed
    /// (`mem::take_trap_frame`); the walk can't run on the host side because the trap unwinds back
    /// onto the same stack the host then reuses. Callers are symbolized at `ret - 1` (inside the call
    /// instruction); adjacent duplicate positions are collapsed, as in [`Self::fiber_backtrace`].
    /// Empty when the module carried no `-g` (nothing symbolizes) or the capture is host-only. The
    /// engine stashes the last run's trap in [`Self::last_trap_backtrace`]. Pure host-side tooling,
    /// off the running guest's path (§2a).
    pub fn trap_backtrace(&self, pc: usize, rets: &[usize]) -> Vec<JitFrameLoc> {
        symbolize_capture(pc, rets, |a| self.symbolize(a))
    }

    /// The source backtrace of this module's most recent [`Self::run`] that **trapped** (§5 W3 Stage
    /// 1): innermost guest frame first, symbolized from the trap site captured during that run. Empty
    /// when the last run returned/exited, the trap carried no usable frame (an explicit-check trap —
    /// Stage 2 — or a platform whose handler doesn't decode the fault frame), or the module carried
    /// no `-g`. Host-side, off the runtime path (§2a) — fold it into a kill/trap report.
    pub fn last_trap_backtrace(&self) -> &[JitFrameLoc] {
        &self.last_trap_backtrace
    }

    /// The guest **fiber** that was running when this module's most recent [`Self::run`] trapped (§5 W3
    /// / §23-D57 per-fiber attribution): `Some(handle)` for a trap inside a fiber, `Some(-1)` for the
    /// root computation (no fiber), `None` after a clean run. The handle is captured at the trap
    /// instant, so it names the right fiber even under work-stealing migration (where the fiber may have
    /// resumed on a different vCPU thread than it last suspended on). Pairs with
    /// [`Self::last_trap_backtrace`] to render *which fiber* trapped *where*; host-side, off the runtime
    /// path (§2a).
    pub fn last_trap_fiber(&self) -> Option<i64> {
        self.last_trap_fiber
    }

    /// The per-source-variable machine-location lists (W5 JIT/DWARF Stage 3a): for each `-g` source
    /// variable whose value the JIT could track, the `[lo, hi)` machine ranges over which it lives in
    /// a register or CFA-relative slot. The seed for the Stage 3c `DW_AT_location` loclists. Empty
    /// without `-g`; a variable with empty `ranges` was optimized out everywhere.
    pub fn var_locations(&self) -> &[VarMachineInfo] {
        &self.var_locs
    }

    /// The synthesized DWARF `.debug_line` section for this module's finalized code (W5 JIT/DWARF
    /// Stage 2): a line-number program over the JIT'd machine addresses, for gdb/lldb (via the
    /// forthcoming GDB JIT registration) to resolve addresses to source lines. Empty without `-g`.
    pub fn debug_line_section(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        dwarf::debug_line(&self.src_ranges, &self.src_files)
    }

    /// The synthesized `(.debug_info, .debug_abbrev, .debug_loc)` for this module's finalized code (W5
    /// JIT/DWARF Stages 2b/3b/3c): a compile unit holding the §6 `TypeDef` graph as `DW_TAG_*_type`
    /// DIEs (3b) and one `DW_TAG_subprogram` per function (2b), each carrying its source variables as
    /// `DW_TAG_variable` children with `.debug_loc` location lists (3c). All empty without `-g`.
    fn synth_debug_info(&self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        if self.src_ranges.is_empty() {
            return (Vec::new(), Vec::new(), Vec::new());
        }
        dwarf::debug_info(
            &self.func_extents(),
            &self.debug_types,
            &self.var_locs,
            &self.func_names,
        )
    }

    /// The synthesized DWARF `(.debug_info, .debug_abbrev)` (W5 JIT/DWARF Stages 2b/3b/3c) — a compile
    /// unit with the §6 type DIEs, one `DW_TAG_subprogram` per function, and a `DW_TAG_variable` per
    /// tracked source variable (its `DW_AT_location` referring into [`Self::debug_loc_section`]). Both
    /// empty without `-g`.
    pub fn debug_info_sections(&self) -> (Vec<u8>, Vec<u8>) {
        let (info, abbrev, _) = self.synth_debug_info();
        (info, abbrev)
    }

    /// The synthesized DWARF `.debug_loc` (W5 JIT/DWARF Stage 3c): the variable location lists the
    /// `DW_TAG_variable` DIEs' `DW_AT_location`s point into — one `DW_OP_reg{N}` entry per
    /// register-resident machine range. Empty without `-g` (or when no variable is register-resident).
    pub fn debug_loc_section(&self) -> Vec<u8> {
        self.synth_debug_info().2
    }

    /// The synthesized DWARF `.debug_frame` CFI (W5 JIT/DWARF Stage 4a): one CIE with the JIT's
    /// uniform frame-pointer unwind rules + one FDE per function, so gdb can unwind a stopped JIT
    /// frame (`bt`) and compute the CFA the subprograms' `DW_AT_frame_base` refers to. Empty without
    /// `-g`.
    pub fn debug_frame_section(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        dwarf::debug_frame(&self.func_extents())
    }

    /// Each function's machine `[low_pc, high_pc)` extent, derived as the span (min `lo`, max `hi`)
    /// of its source-mapped ranges, sorted by start address. The basis for both the `.debug_info`
    /// subprograms (Stage 2b) and the ELF `.symtab`/`.text` extent (Stage 2c). Empty without `-g`.
    fn func_extents(&self) -> Vec<(u32, u64, u64)> {
        let mut per_fn: std::collections::BTreeMap<u32, (u64, u64)> =
            std::collections::BTreeMap::new();
        for r in &self.src_ranges {
            let e = per_fn.entry(r.func).or_insert((r.lo, r.hi));
            e.0 = e.0.min(r.lo);
            e.1 = e.1.max(r.hi);
        }
        let mut funcs: Vec<(u32, u64, u64)> = per_fn
            .into_iter()
            .map(|(f, (lo, hi))| (f, lo, hi))
            .collect();
        funcs.sort_by_key(|&(_, lo, _)| lo);
        funcs
    }

    /// The in-memory ELF object that wraps this module's finalized code + synthesized DWARF for the
    /// GDB JIT interface (W5 JIT/DWARF Stage 2c): an `SHT_NOBITS` `.text` at the live code address,
    /// the `.debug_line`/`.debug_info`/`.debug_abbrev` sections, and a `.symtab` naming each
    /// function. This is the `symfile` [`Self::register_with_gdb`] hands to gdb. Empty without `-g`.
    pub fn elf_object(&self) -> Vec<u8> {
        if self.src_ranges.is_empty() {
            return Vec::new();
        }
        let funcs = self.func_extents();
        let code_base = funcs.iter().map(|f| f.1).min().unwrap_or(0);
        let code_end = funcs.iter().map(|f| f.2).max().unwrap_or(0);
        let (info, abbrev, loc) = self.synth_debug_info();
        let line = self.debug_line_section();
        let frame = self.debug_frame_section();
        gdb::build_elf(
            code_base,
            code_end.saturating_sub(code_base),
            &funcs,
            &self.func_names,
            &info,
            &abbrev,
            &line,
            &loc,
            &frame,
        )
    }

    /// Register this module's code with a live gdb/lldb via the GDB JIT interface (W5 JIT/DWARF
    /// Stage 2c) — the **headline milestone**: with the returned guard held, gdb can bind a
    /// source-line breakpoint inside the JIT'd code and show the guest source frame. The returned
    /// [`gdb::GdbRegistration`] **unregisters on drop**; hold it as long as the code is debuggable.
    /// `None` for a non-`-g` module (nothing to register). Host-side tooling, off the runtime path.
    pub fn register_with_gdb(&self) -> Option<gdb::GdbRegistration> {
        let elf = self.elf_object();
        if elf.is_empty() {
            return None;
        }
        Some(gdb::GdbRegistration::register(elf))
    }

    /// Compile the whole module (the compile half of the old one-shot `compile_and_run*`):
    /// declare + define every function, the entry's buffer-ABI trampoline, finalize once, and
    /// build the function table. What the code is compiled for — the confinement mask, the table
    /// mask, the checks it emits — and its instance's `call.cap` ctx, runtimes and §5 interrupt
    /// cell are fixed here; per-run state (the window, the trap cell) is supplied by [`Self::run`].
    ///
    /// # Safety-relevant contract (not `unsafe`, but load-bearing)
    /// The compiled code calls `cap_thunk` and the targets `fast_resolver` claims, and its instance
    /// holds `cap_ctx` and `interrupt`: they must stay valid for the **lifetime of this
    /// `CompiledModule`** (not just one run) and honour their respective ABIs — the same contract
    /// the one-shot entry points documented per call, stretched over the module's life.
    #[allow(clippy::too_many_arguments)]
    pub fn compile(
        m: &IrModule,
        func: FuncIdx,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        reserved_log2: u8,
        resolve_module: Option<ModuleResolver>,
        interrupt: Option<*const AtomicU64>,
        fuel: Option<*mut FuelCell>,
        fast_resolver: Option<FastCapResolver>,
        node: Option<std::sync::Arc<dyn BudgetNode>>,
        table_reserve_log2: u8,
    ) -> Result<CompiledModule, JitError> {
        // #932 — the stable public signature: no async signal delivery armed (the common case). The
        // signal-arming variant is [`Self::compile_with_signal`], reached only by the one-shot
        // `compile_and_run_with_host_signals` entry, so every existing caller is unchanged.
        Self::compile_with_signal(
            m,
            func,
            cap_thunk,
            cap_ctx,
            reserved_log2,
            resolve_module,
            interrupt,
            fuel,
            fast_resolver,
            node,
            table_reserve_log2,
            None,
            None,
        )
    }

    /// [`Self::compile`], keeping the code in a [`CodeDir`] when `kept` names one, under the
    /// module's content digest: a compile of the same module with the same arguments, in this
    /// process or a later one, defines the code kept instead of compiling it (temen#2452). A module
    /// with debug info compiles as by [`Self::compile`] and keeps nothing: its code is not all its
    /// compile makes.
    #[allow(clippy::too_many_arguments)]
    pub fn compile_kept(
        kept: Option<(&CodeDir, [u8; 32])>,
        m: &IrModule,
        func: FuncIdx,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        reserved_log2: u8,
        resolve_module: Option<ModuleResolver>,
        interrupt: Option<*const AtomicU64>,
        fuel: Option<*mut FuelCell>,
        fast_resolver: Option<FastCapResolver>,
        node: Option<std::sync::Arc<dyn BudgetNode>>,
        table_reserve_log2: u8,
    ) -> Result<CompiledModule, JitError> {
        Self::compile_with_signal(
            m,
            func,
            cap_thunk,
            cap_ctx,
            reserved_log2,
            resolve_module,
            interrupt,
            fuel,
            fast_resolver,
            node,
            table_reserve_log2,
            None,
            kept,
        )
    }

    /// [`Self::compile`] with an optional #932 async-signal delivery arm (`signal`) baked into the
    /// module's safepoints. `None` ⇒ byte-identical to [`Self::compile`]. Crate-internal — the stable
    /// public surface is [`Self::compile`] + the named `compile_and_run_with_host_signals` entry.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_with_signal(
        m: &IrModule,
        func: FuncIdx,
        cap_thunk: CapThunk,
        cap_ctx: *mut core::ffi::c_void,
        reserved_log2: u8,
        resolve_module: Option<ModuleResolver>,
        interrupt: Option<*const AtomicU64>,
        fuel: Option<*mut FuelCell>,
        fast_resolver: Option<FastCapResolver>,
        node: Option<std::sync::Arc<dyn BudgetNode>>,
        table_reserve_log2: u8,
        signal: Option<SignalArm>,
        kept: Option<(&CodeDir, [u8; 32])>,
    ) -> Result<CompiledModule, JitError> {
        let entry = m.funcs.get(func as usize).ok_or(JitError::Malformed)?;
        let shadow = m
            .memory
            .and_then(|x| x.shadow)
            .unwrap_or(temen_ir::durable_abi::ShadowArena::EMPTY);
        // The `call.dyn` function table is power-of-two padded; `table_reserve_log2`
        // (DESIGN.md §22) reserves a *larger* table than the module needs so `install` can
        // fill the padding slots without moving the Spectre-safe mask constant (which is baked
        // from this length into every call site). `0` ⇒ the natural `next_pow2(funcs.len())`,
        // i.e. behavior-identical to before B2.
        let table_len = (1usize << table_reserve_log2)
            .max(m.funcs.len().next_power_of_two())
            .max(1);
        // §5 fuel/epoch kill-path: the address of the host-owned interrupt cell the lowering polls at
        // loop back-edges + function entries. `0` when the caller armed no kill-path (then no checks are
        // emitted — guest code is byte-identical to before). The cell must outlive every run of this
        // module's own instance (it is read from the instance's `VmCtx`); the caller owns it (e.g. an
        // `Arc<AtomicU64>` a watchdog thread sets).
        let epoch_addr = interrupt.map_or(0, |p| p as i64);
        // Fuel unification: the address of the host-owned counted-fuel cell the lowering decrements at
        // safepoints (function entries + taken back-edges + `cont.resume`). `0` ⇒ no fuel armed (no
        // fuel checks emitted — guest code byte-identical). The caller owns the `u64` cell (it must
        // outlive the runs that charge it) and reads the remaining budget back after the run.
        let fuel_addr = fuel.map_or(0, |p| p as i64);
        // #932 — the async-signal delivery arm `emit_signal_check` polls at safepoints: none (no
        // checks emitted) unless the caller supplied a `SignalArm`. The code calls its two thunks
        // ([`HostFn::SigTake`], [`HostFn::SigReturn`]); the `Host::sig_armed` flag + the delivery
        // ctx are the instance's, read from its `VmCtx`, so the caller keeps them valid for the
        // runs of this instance (the one-shot signal entry holds the delivery ctx for the whole
        // run).
        let signals = signal.is_some();
        // The thread domain and the nursery, if any, are stood up below.
        #[cfg_attr(not(fiber_rt), allow(unused_mut))]
        let mut instance = InstanceAddrs {
            cap_ctx,
            epoch: epoch_addr as *const AtomicU64,
            fuel: fuel_addr as *mut FuelCell,
            sig_armed: signal.as_ref().map_or(core::ptr::null(), |s| s.armed),
            sig_ctx: signal.as_ref().map_or(core::ptr::null_mut(), |s| s.ctx),
            ..InstanceAddrs::NONE
        };
        // Calls can reach any function, so every function must be lowerable.
        for f in &m.funcs {
            ensure_supported(f, &m.types)?;
        }

        // Plan the guest window if the module declares memory (allocation happens per `run`):
        // `mapped` backed RW bytes inside a host-configured `reserved` virtual range whose
        // unmapped tail + guard page fault (§4). `mask` is the §4 confinement mask (`reserved − 1`,
        // the mask domain). All of these are baked into the code, so they are fixed at compile time.
        let (win_mapped, win_reserved, mask): (usize, usize, u64) = match m.memory {
            Some(mc) => {
                if mc.size_log2 > MAX_JIT_WINDOW_LOG2 {
                    return Err(JitError::Unsupported(
                        "window too large for the reference JIT",
                    ));
                }
                // Host reservation policy: at least `mapped` (fully mapped if `reserved_log2` is
                // smaller, e.g. 0), capped so the reference JIT's reservation stays sane.
                let reserved_log2 = reserved_log2.max(mc.size_log2).min(MAX_JIT_RESERVED_LOG2);
                (
                    1usize << mc.size_log2,
                    1usize << reserved_log2,
                    (1u64 << reserved_log2) - 1,
                )
            }
            None => (0, 0, 0),
        };

        let imports = Imports {
            cap_thunk: cap_thunk as *const u8,
            sig: signal.as_ref().map(|s| (s.take, s.ret)),
            fast_resolver,
            page_miss: None, // only a §14 child is demand-paged (#2164)
        };
        let mut module = new_jit_module(imports)?;

        // Declare every function (natural ABI) up front so calls can reference any of them.
        let ids: Vec<FuncId> = declare_all_funcs(&mut module, &m.funcs)?;

        // temen#2452 — the code a `CodeDir` kept for this compile, defined in place of compiling,
        // or else a record of the code the compile emits, to keep. A module's debug info or a signal
        // arm makes more than code, so neither compile is kept.
        let kept = kept
            .filter(|_| m.debug_info.is_none() && signal.is_none())
            .map(|(dir, digest)| {
                let key = code_dir::root_key(
                    &digest,
                    func,
                    epoch_addr != 0,
                    fuel_addr != 0,
                    fast_resolver.is_some(),
                    reserved_log2,
                    table_reserve_log2,
                );
                (dir, key)
            });
        let replay = kept
            .and_then(|(dir, key)| dir.load(&key))
            .filter(|code| code.fits(&module, &ids, &imports));
        let mut codegen = match &replay {
            Some(kept) => Codegen::Replay(kept, &imports),
            None => Codegen::Compile(kept.map(|_| EmittedCode::default())),
        };

        // Distinct signatures give each function (and call site) a structural type id, the
        // basis for the `call.dyn` type check (matching the interpreter). Function
        // signatures come first (`distinct_types`, ids identical to the historical one-shot
        // compile — the fn-table and `fiber_type_id` depend on those positions), then every
        // call-site signature is interned after them. Today a site whose signature matches no
        // function traps either way (a fresh id ≥ the function-sig count matches no table
        // entry, exactly like `NO_MATCH_TYPE_ID`) — but interning it now means a *later*
        // `define_extra`/install of a function with that signature can satisfy the site,
        // keeping id-equality ≡ structural equality across units (DESIGN.md §22).
        let mut distinct = distinct_types(&m.funcs);
        intern_unit_sigs(&mut distinct, &m.funcs, &m.types)?;
        // §12: a module that uses `cont.*` may `cont.new` a fiber over ANY dispatch-table slot —
        // including a `Jit.install` slot into an installed unit whose `(i64,i64)->(i64)` entry is
        // interned only when that unit is later `define_extra`'d. Intern the fiber-entry signature NOW
        // (the compile-path twin of `enable_fiber_hosting`) so its id is stable and a later install of
        // that signature gets the SAME id — keeping id-equality ≡ structural equality across units, so
        // the root's fiber first-resume type-check matches an installed-unit entry (#1226) instead of
        // seeing `NO_MATCH_TYPE_ID` (the root program itself has no such function). Append-only +
        // idempotent, so a fiber-free module never reaches this and function ids are unchanged.
        #[cfg(fiber_rt)]
        if module_uses_fibers(m) {
            intern_type(&mut distinct, &fiber_func_type())?;
        }
        let distinct = distinct;

        // §12 fibers: if the module uses `cont.*`, stand up a host fiber runtime (an entry publishes it
        // in a thread-local the `cont.*` thunks read). The `FiberRuntime` box is created now; the
        // call-trampoline address is filled in after finalize. On targets without stack-switch support,
        // `ensure_supported` has already rejected the fiber ops, so this stays `null`.
        #[cfg(fiber_rt)]
        let uses_fibers = module_uses_fibers(m);
        #[cfg(fiber_rt)]
        let uses_threads = module_uses_threads(m);
        // Per-run fiber constants (the §12 fiber entry type id + the function-table mask), used to build
        // fiber runtimes — one standalone, or one per vCPU when threads use `cont.*`.
        #[cfg(fiber_rt)]
        let fiber_type_id = type_id_of(&distinct, &fiber_func_type());
        // The fiber-entry funcref is a `call.dyn` dispatch-table index (a `ref.func` of a module-0
        // function OR a `Jit.install` slot into an installed §22 unit), so it must be masked into the
        // **reserved** table exactly as `call.dyn` (the `(table_len as u64) - 1` mask baked per call
        // site, below, and stored in `fn_table_mask`) — not `next_pow2(nfuncs) - 1`, which is 0 for a
        // 1-function root and folds every install slot onto func 0, trapping `FiberFault` for an
        // installed-unit fiber body (#1226; the tree-walk oracle + bytecode engine resolve module-aware).
        #[cfg(fiber_rt)]
        let fiber_mask = (table_len as u64) - 1;
        // Fibers + threads compose via a per-vCPU fiber runtime (execution context), published through a
        // thread-local, all over **one domain-shared fiber table** (D57 3b-ii: a unified handle
        // namespace + a per-domain fiber ceiling, matching the interpreter's run-shared registry). This is
        // the *root* vCPU's runtime (the one running `main` on the caller's thread); each spawned vCPU
        // builds its own over the same table from `fiber_cfg` (`os_thread_rt`). Created whenever the
        // module uses `cont.*` **or** `thread.spawn` — a threaded module needs the table for the
        // durable vCPU-context allocator (slice 3.3), even when it uses no fibers (the table is then
        // dormant: a fiber-free module never resumes a fiber, so the root's runtime just routes the
        // durable shadow-SP swap).
        #[cfg(fiber_rt)]
        let fiber_table: Option<std::sync::Arc<fiber_rt::SharedFiberTable>> =
            if uses_fibers || uses_threads {
                Some(std::sync::Arc::new(fiber_rt::SharedFiberTable::new(
                    temen_ir::MAX_FIBERS,
                    shadow,
                    node.clone(),
                )))
            } else {
                None
            };
        // The *root* vCPU's fiber runtime is built only when the module actually uses `cont.*` — a
        // fiber-free module (even a threaded one) never resumes a fiber, so it needs no execution
        // context, and the durable shadow-SP word it does use is driven by the instrumented IR
        // directly (the multi-vCPU deferred/thaw spawn paths go through the `Domain`, not this
        // runtime). The *table* is still created for `uses_threads` above (the durable vCPU-context
        // allocator); only the per-vCPU runtime is fiber-gated, so a thread-only run publishes no
        // `CURRENT_RT` and allocates no idle runtime (matching the pre-slice-3.3 behavior).
        #[cfg(fiber_rt)]
        let mut fiber_rt: Option<Box<fiber_rt::FiberRuntime>> = if uses_fibers {
            fiber_table.as_ref().map(|t| {
                Box::new(fiber_rt::FiberRuntime::new(
                    std::sync::Arc::clone(t),
                    fiber_type_id,
                    fiber_mask,
                ))
            })
        } else {
            None
        };
        // The code's `cont.*` sites call the fiber thunks, which find the runtime via a
        // thread-local.
        #[cfg(fiber_rt)]
        let fibers = uses_fibers;
        #[cfg(not(fiber_rt))]
        let fibers = false;

        // §12 threads: stand up the 1:1 OS-thread executor `Domain` the `thread.*` sites load from the
        // instance's vmctx. It owns no scheduling policy — `thread.spawn` launches a real OS thread (the
        // guest builds any M:N model itself, D22). The per-run `Env` (call-trampoline, window, trap cell)
        // is supplied after finalize via `set_env`; the address is stable now. Also stood up for a
        // **nesting** module with no thread ops of its own: §14 children compile their `atomic.wait`/
        // `notify` against this domain's futex (via the nursery), so concurrent children — and the
        // parent's own waits, if any — rendezvous in one table (the granted-children pipeline).
        // I48: a fiber-using module needs the `Domain` even without threads — the
        // `cont.resume.block` thunk (`fiber_resume_block`) idles on `Domain.futex_cv`. (A fiber
        // that *parks* already implies `memory.wait` ⇒ `uses_threads`; this also covers a
        // fiber-only module whose blocking resume never reaches the park — the thunk still takes
        // the domain pointer.)
        #[cfg(fiber_rt)]
        let domain: Option<Box<os_thread_rt::Domain>> =
            if uses_threads || uses_fibers || module_uses_instantiator(m) {
                Some(Box::new(os_thread_rt::Domain::new(node.clone())))
            } else {
                None
            };
        #[cfg(fiber_rt)]
        if let Some(d) = &domain {
            instance.sched = &**d as *const os_thread_rt::Domain as *const core::ffi::c_void;
        }
        #[cfg(fiber_rt)]
        let threads = domain.is_some();
        #[cfg(not(fiber_rt))]
        let threads = false;

        // §14 nesting: if the module holds an `Instantiator` (a `call.cap` to iface 6), stand up the
        // per-run `Nursery` those sites spawn through, which they load from the instance's vmctx.
        #[cfg(fiber_rt)]
        let spawns = module_uses_instantiator(m);
        #[cfg(fiber_rt)]
        // #1726: also for a module that can **install** §22 units (a reserved `call.dyn` table): an
        // installed unit that instantiates runs in this module's frames, so it needs this nursery
        // even when the module itself names no `Instantiator` (it may only pass the handle on).
        let nursery = (spawns || table_reserve_log2 > 0).then(|| {
            nursery_over(
                cap_thunk,
                cap_ctx,
                resolve_module,
                epoch_addr as usize,
                // Always stood up for a nesting module, above.
                domain.as_deref(),
                shadow,
            )
        });
        // temen#2168: the code loads the nursery from its instance's vmctx, as it loads the domain.
        #[cfg(fiber_rt)]
        if let Some(n) = &nursery {
            instance.nursery = &**n as *const instantiator_rt::Nursery as *const core::ffi::c_void;
        }
        #[cfg(fiber_rt)]
        let nesting = nursery.is_some();
        #[cfg(not(fiber_rt))]
        let nesting = false;

        // `setjmp`/`longjmp` (LLVM.md §"JIT `longjmp`", Option B): when the module uses them, stand up
        // the per-run host `jmp_buf` table the `SetJmp`/`LongJmp` sites load from the instance's vmctx.
        // Owned by the `CompiledModule` (`_setjmp_rt`), so it outlives every (re-)run. Unix-only
        // (the `setjmp_rt` cfg); elsewhere `ensure_supported` has already rejected the ops.
        #[cfg(setjmp_rt)]
        let uses_setjmp = module_uses_setjmp(m);
        // Fail-closed (guard against cross-stack corruption): the per-run `jmp_buf` table is keyed by
        // the guest buffer address and shared across the run's native stacks, so a module that mixes
        // `setjmp` with fibers/threads could let one stack's `setjmp` overwrite another's saved native
        // SP (a `longjmp` would then restore into the wrong stack). The on-ramp never emits that combo,
        // but a hand-crafted IR module could — so decline the JIT and let the interpreters (which key
        // `setjmp_points` per-vCPU) cover it. Per-fiber JIT keying is a documented follow-on.
        #[cfg(setjmp_rt)]
        if uses_setjmp && (uses_fibers || uses_threads) {
            return Err(JitError::Unsupported(
                "setjmp/longjmp combined with fibers/threads is not supported on the JIT yet",
            ));
        }
        #[cfg(setjmp_rt)]
        let setjmp_runtime: Option<Box<setjmp_rt::SetjmpRuntime>> = if uses_setjmp {
            Some(Box::new(setjmp_rt::SetjmpRuntime::new()))
        } else {
            None
        };
        // temen#2389: the sites load the table from the instance's vmctx, as they load the domain.
        #[cfg(setjmp_rt)]
        if let Some(r) = &setjmp_runtime {
            instance.setjmp = &**r as *const setjmp_rt::SetjmpRuntime as *const core::ffi::c_void;
        }

        // W5 JIT/DWARF tier (Stages 0–1): when the module carries `-g` debug info, build the
        // `(func,block,inst) → debug_info.locs` index lookup that stamps each op with a `SourceLoc`,
        // and capture each function's `MachSrcLoc` ranges (relative offsets) to resolve to a
        // machine-address → source map after finalize. Off the runtime path; absent ⇒ no-op.
        let srcloc_map: Option<SrcLocMap> = m.debug_info.as_ref().map(|di| {
            di.locs
                .iter()
                .enumerate()
                .map(|(i, l)| ((l.func, l.block, l.inst), i as u32))
                .collect()
        });
        let mut raw_srclocs: RawSrcLocs = Vec::new();

        // W5 JIT/DWARF Stage 3a: assign each SSA-resident source variable a `ValueLabel` and record
        // the `(func, block) → [(block-local value, label)]` points `lower_block` stamps. Only the
        // SSA forms (`Ssa`/`SsaList`) map to a Cranelift value-location; the window/fixed *memory*
        // forms are left to Stage 3c (a DWARF memory expression, not a value label). `var_meta[label]`
        // names the variable a label belongs to. Empty unless the module carries `-g` vars.
        let mut var_meta: Vec<(u32, String, Option<u32>)> = Vec::new();
        let mut var_label_map: VarLabelMap = std::collections::HashMap::new();
        if let Some(di) = m.debug_info.as_ref() {
            for v in &di.vars {
                let points: Vec<(u32, u32)> = match &v.loc {
                    // A function-wide SSA index ≈ the block-0 local index (parameters, single-block
                    // promoted scalars — the cases where the two numberings coincide).
                    temen_ir::VarLoc::Ssa { value } => vec![(0, *value)],
                    temen_ir::VarLoc::SsaList(locs) => {
                        locs.iter().map(|l| (l.block, l.value)).collect()
                    }
                    // `Window`/`WindowVia`/`Fixed` are memory locations — no value label here.
                    _ => continue,
                };
                let label = var_meta.len() as u32;
                var_meta.push((v.func, v.name.clone(), v.type_id));
                for (block, value) in points {
                    var_label_map
                        .entry((v.func, block))
                        .or_default()
                        .push((value, label));
                }
            }
        }
        let var_labels = (!var_label_map.is_empty()).then_some(&var_label_map);
        let mut raw_var_locs: RawVarLocs = Vec::new();

        // Define each function body (#2450: compiled on worker threads, defined in order).
        // Optional codegen dump for performance work: `TEMEN_JIT_DUMP_CLIF=1` prints each function's
        // CLIF and (post-compile) the Cranelift VCODE disassembly to stderr. A pure diagnostic — no
        // effect unless the env var is set; used to compare temen-jit's generated code against other
        // backends (e.g. the Embench cross-engine bench).
        let dump_codegen = std::env::var_os("TEMEN_JIT_DUMP_CLIF").is_some();
        let base_bytes = define_funcs(
            &mut module,
            &ids,
            &mut codegen,
            |module, ctx, fi| {
                // Stage 3a: enable Cranelift's value-label tracking so `set_val_label` (in
                // `lower_block`) takes effect and `value_labels_ranges` is populated. Gated on `-g`
                // vars ⇒ no effect on an ordinary build. Must be set on the fresh `ctx.func` before
                // lowering populates it.
                if var_labels.is_some() {
                    ctx.func.collect_debug_info();
                }
                build_clif(
                    module,
                    &ids,
                    &m.funcs,
                    &distinct,
                    &m.types,
                    fast_resolver,
                    nesting,
                    signals, // #932 async signal delivery (off unless a `SignalArm` was supplied)
                    PageEnv::NONE, // #2164 — only a §14 child is demand-paged
                    &mut ctx.func,
                    &m.funcs[fi],
                    mask,
                    win_mapped as u64,
                    guard_offset_of(win_reserved as u64),
                    epoch_addr != 0,
                    fuel_addr != 0,
                    threads,                // spawned vCPUs run this code beside the root
                    (table_len as u64) - 1, // the (possibly B2-reserved) table mask, baked per call site
                    None,                   // top-level: `ref.func N` = module-0 slot N (no remap)
                    fi as u32,
                    srcloc_map.as_ref(),
                    var_labels,
                )?;
                if dump_codegen {
                    eprintln!("=== TEMEN-JIT CLIF fn{fi} ===\n{}", ctx.func.display());
                    ctx.set_disasm(true);
                }
                Ok(())
            },
            |module, cc, fi| {
                // NOTE (feature `stack-check`, STACK_GUARD.md §2): soundness wants a post-compile
                // bound rejecting any function whose frame exceeds `RED_ZONE` (so a function can't
                // grow the stack past the limit before its callee re-checks). Cranelift 0.132 doesn't
                // expose the final frame size on `CompiledCode` (`FrameLayout`/`total_stacksize` are
                // ABI-internal), so this bound is DEFERRED — see the increment plan. Until then the
                // check assumes frames ≤ RED_ZONE.
                if dump_codegen {
                    if let Some(d) = cc.vcode.as_ref() {
                        eprintln!("=== TEMEN-JIT VCODE fn{fi} ===\n{d}");
                    }
                }
                if srcloc_map.is_some() {
                    let ranges: Vec<(u32, u32, u32)> = cc
                        .buffer
                        .get_srclocs_sorted()
                        .iter()
                        .filter(|s| !s.loc.is_default())
                        .map(|s| (s.start, s.end, s.loc.bits()))
                        .collect();
                    if !ranges.is_empty() {
                        raw_srclocs.push((fi as u32, ranges));
                    }
                }
                // Stage 3a: capture this function's value-label ranges, translating each Cranelift
                // `LabelValueLoc` to DWARF terms now (the ISA is in hand; a `Reg` we cannot map is
                // dropped — that sub-range simply has no location, like an optimized-out gap).
                if var_labels.is_some() {
                    let isa = module.isa();
                    let mut per_label: Vec<(u32, VarLocRanges)> = Vec::new();
                    for (label, ranges) in &cc.value_labels_ranges {
                        let mut out: VarLocRanges = Vec::new();
                        for r in ranges {
                            if r.end <= r.start {
                                continue;
                            }
                            let loc = match r.loc {
                                LabelValueLoc::Reg(reg) => {
                                    match isa.map_regalloc_reg_to_dwarf(reg) {
                                        Ok(d) => VarMachineLoc::Reg(d),
                                        Err(_) => continue,
                                    }
                                }
                                LabelValueLoc::CFAOffset(off) => VarMachineLoc::CfaOffset(off),
                            };
                            out.push((r.start, r.end, loc));
                        }
                        if !out.is_empty() {
                            per_label.push((label.as_u32(), out));
                        }
                    }
                    if !per_label.is_empty() {
                        raw_var_locs.push((fi as u32, per_label));
                    }
                }
            },
        )?;
        let mut ctx = module.make_context();

        // The buffer-ABI trampoline for the entry, exported so Rust can call it.
        build_trampoline(&mut module, &mut ctx.func, ids[func as usize], entry, false)?;
        let tramp = module
            .declare_function("trampoline", Linkage::Export, &ctx.func.signature)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        define_verified(&mut module, tramp, &mut ctx)?;
        module.clear_context(&mut ctx);

        // I36 slice 3 (§3.6 JIT serve loop): a buffer-ABI trampoline for every **impl-export
        // handler**, so the embedder's serve arm (its cap thunk's native `svc.poll`/`svc.wait`)
        // can invoke a handler of any signature over the live window via [`Self::invoke_extra`]
        // — the same `build_trampoline` shape as the entry's, one per distinct handler funcidx.
        let mut serve_ids: Vec<(u32, cranelift_module::FuncId)> = Vec::new();
        {
            let mut seen = std::collections::BTreeSet::new();
            for e in &m.impl_exports {
                for &f in &e.ops {
                    if (f as usize) < m.funcs.len() && seen.insert(f) {
                        build_trampoline(
                            &mut module,
                            &mut ctx.func,
                            ids[f as usize],
                            &m.funcs[f as usize],
                            false,
                        )?;
                        let id = module
                            .declare_function(
                                &format!("serve_tramp_{f}"),
                                Linkage::Export,
                                &ctx.func.signature,
                            )
                            .map_err(|e| JitError::Backend(e.to_string()))?;
                        define_verified(&mut module, id, &mut ctx)?;
                        module.clear_context(&mut ctx);
                        serve_ids.push((f, id));
                    }
                }
            }
        }

        // The generic call-trampoline (one per module; calls any `Tail`-ABI `(sp, arg) -> i64` entry from
        // Rust). Needed by both the fiber runtime (`cont.*`) and the thread scheduler (vCPU entries).
        #[cfg(fiber_rt)]
        let fiber_tramp = if uses_fibers || uses_threads {
            build_fiber_call_trampoline(&mut module, &mut ctx.func);
            let id = module
                .declare_function("fiber_call_tramp", Linkage::Export, &ctx.func.signature)
                .map_err(|e| JitError::Backend(e.to_string()))?;
            define_verified(&mut module, id, &mut ctx)?;
            module.clear_context(&mut ctx);
            Some(id)
        } else {
            None
        };

        module
            .finalize_definitions()
            .map_err(|e| JitError::Backend(e.to_string()))?;

        // W5 JIT/DWARF Stage 1: now that code is finalized, resolve each captured `MachSrcLoc` range
        // (relative to its function's start) to an absolute machine address, building the sorted
        // `machine-pc → source` map `symbolize` consults. Empty without `-g`.
        let (src_ranges, src_files, func_names) = match m.debug_info.as_ref() {
            Some(di) => {
                let mut ranges = Vec::new();
                for (fi, rs) in &raw_srclocs {
                    let base = module.get_finalized_function(ids[*fi as usize]) as u64;
                    for &(start, end, loc) in rs {
                        if let Some(l) = di.locs.get(loc as usize) {
                            ranges.push(SrcRange {
                                lo: base + start as u64,
                                hi: base + end as u64,
                                func: l.func,
                                file: l.file,
                                line: l.line,
                                col: l.col,
                            });
                        }
                    }
                }
                ranges.sort_by_key(|r| r.lo);
                // §6 function names (`func → name`), for `symbolize`, the DWARF `DW_AT_name`, and kill
                // messages — `compute` instead of `fn3`.
                let names = di
                    .func_names
                    .iter()
                    .map(|f| (f.func, f.name.clone()))
                    .collect();
                (ranges, di.files.clone(), names)
            }
            None => (Vec::new(), Vec::new(), std::collections::HashMap::new()),
        };

        // W5 JIT/DWARF Stage 3a: resolve the captured value-label ranges (relative offsets) to
        // absolute machine pcs and group them per source variable. A variable whose label produced
        // no range (the optimizer dropped its value everywhere) still gets a `VarMachineInfo` with
        // empty `ranges` — Stage 3c emits that as a `<optimized out>` location.
        let var_locs: Vec<VarMachineInfo> = if var_meta.is_empty() {
            Vec::new()
        } else {
            let mut by_label: std::collections::HashMap<u32, Vec<VarRange>> =
                std::collections::HashMap::new();
            for (fi, per_label) in &raw_var_locs {
                let base = module.get_finalized_function(ids[*fi as usize]) as u64;
                for (label, ranges) in per_label {
                    let entry = by_label.entry(*label).or_default();
                    for &(s, e, loc) in ranges {
                        entry.push(VarRange {
                            lo: base + s as u64,
                            hi: base + e as u64,
                            loc,
                        });
                    }
                }
            }
            var_meta
                .iter()
                .enumerate()
                .map(|(label, (func, name, type_id))| {
                    let mut ranges = by_label.remove(&(label as u32)).unwrap_or_default();
                    ranges.sort_by_key(|r| r.lo);
                    VarMachineInfo {
                        func: *func,
                        name: name.clone(),
                        type_id: *type_id,
                        ranges,
                    }
                })
                .collect()
        };

        // Now that code is finalized, hand the root fiber runtime its call-trampoline address (and keep it
        // to seed the thread `Domain`'s `Env`, which spawned vCPUs use to call their entry).
        #[cfg(fiber_rt)]
        let mut call_tramp: Option<fiber_rt::FiberCallTramp> = None;
        #[cfg(fiber_rt)]
        if let Some(id) = fiber_tramp {
            let addr = module.get_finalized_function(id);
            // SAFETY: `addr` is the finalized `fiber_call_tramp` with exactly the `FiberCallTramp` ABI.
            let t: fiber_rt::FiberCallTramp = unsafe { std::mem::transmute(addr) };
            if let Some(rt) = &mut fiber_rt {
                rt.set_call_tramp(t);
            }
            call_tramp = Some(t);
        }

        // Build the function table (§3c) now that code addresses are known: power-of-two
        // padded (to `table_len`, possibly B2-reserved beyond the module), AoS, host-owned.
        // `call.dyn` masks the guest index into this; padding/reserved slots trap until
        // `install` fills them.
        let fn_table: Box<[FnEntry]> = (0..table_len)
            .map(|slot| match m.funcs.get(slot) {
                Some(f) => FnEntry::new(
                    type_id_of(
                        &distinct,
                        &FuncType {
                            params: f.params.clone(),
                            results: f.results.clone(),
                        },
                    ),
                    module.get_finalized_function(ids[slot]) as u64,
                ),
                None => FnEntry::padding(),
            })
            .collect();

        let tramp_code = module.get_finalized_function(tramp);
        // I36 slice 3: resolve the handler trampolines now that code is finalized.
        let serve_tramps: Vec<(u32, *const u8, usize, usize)> = serve_ids
            .iter()
            .map(|&(f, id)| {
                let fu = &m.funcs[f as usize];
                (
                    f,
                    module.get_finalized_function(id),
                    fu.params.len(),
                    fu.results.len(),
                )
            })
            .collect();
        #[cfg(not(fiber_rt))]
        let _ = (&node, resolve_module);
        if let (Some((dir, key)), Codegen::Compile(Some(emitted))) = (kept, &codegen) {
            dir.keep(&key, emitted);
        }
        if replay.is_none() {
            MODULE_COMPILES.fetch_add(1, Ordering::Relaxed);
        }
        Ok(CompiledModule {
            hosted: false,
            nests: cfg!(fiber_rt) && module_uses_instantiator(m),
            page: PageEnv::NONE,
            fn_table,
            tramp_code,
            tramp_code_limited: core::ptr::null(), // a root is never a child-domain task
            serve_tramps,
            n_params: entry.params.len(),
            n_results: entry.results.len(),
            n_real_funcs: m.funcs.len(),
            distinct,
            fast_resolver,
            fibers,
            threads,
            spawns: threads,
            nesting,
            mask,
            epoch: epoch_addr != 0,
            fuel: fuel_addr != 0,
            instance,
            fn_table_mask: (table_len as u64) - 1,
            next_extra: 0,
            extra_bytes: 0,
            base_bytes,
            live_fault_range: None,
            caller_window: false,
            last_trap_backtrace: Vec::new(),
            last_trap_fiber: None,
            win_mapped,
            win_reserved,
            // #964: the marker opts this module's window into the NULL guard.
            null_guard: temen_ir::module_null_guard(),
            mem_size_log2: m.memory.map(|mc| mc.size_log2),
            data: m.data.clone(),
            restore: None,
            tail_out: Vec::new(),
            durable: false,
            shadow,
            concurrent_durable: false,
            frozen_seed: Vec::new(),
            frozen_out: Vec::new(),
            frozen_vcpus_out: Vec::new(),
            frozen_root_sp_out: 0,
            frozen_vcpu_seed: Vec::new(),
            detached_out: Vec::new(),
            detached_seed: Vec::new(),
            thaw_root_sp: shadow.frame_base(0), // §12.8 4A.5: empty root extent
            freeze_ctl: None,
            held_tail: None,
            window_ready: None,
            fork_hook: None,
            #[cfg(fiber_rt)]
            fiber_rt,
            #[cfg(fiber_rt)]
            domain,
            #[cfg(fiber_rt)]
            _nursery: nursery,
            #[cfg(setjmp_rt)]
            _setjmp_rt: setjmp_runtime,
            #[cfg(fiber_rt)]
            cap_thunk,
            #[cfg(fiber_rt)]
            resolve_module,
            #[cfg(fiber_rt)]
            call_tramp,
            #[cfg(fiber_rt)]
            fiber_cfg: if uses_fibers {
                Some((fiber_type_id, fiber_mask))
            } else {
                None
            },
            #[cfg(fiber_rt)]
            fiber_table,
            #[cfg(fiber_rt)]
            node,
            src_ranges,
            src_files,
            func_names,
            var_locs,
            debug_types: m
                .debug_info
                .as_ref()
                .map(|di| di.types.clone())
                .unwrap_or_default(),
            module: Arc::new(OwnedJit::new(module)),
        })
    }

    /// Run the compiled entry over a **fresh guest window** on slot-encoded `args` (the run half
    /// of the old one-shot `compile_and_run*`): allocate + seed the window (init bytes, data
    /// segments, RO protection), seed the per-run runtime env, execute under the §5
    /// detect-and-kill guard, copy back what `snapshot_cap` asks for (`None`: nothing;
    /// `Some(cap)`: the backed prefix, widened to `cap` bytes), and tear the window down. The
    /// executable code and the runtimes stay alive in `self`, so `run` can be called again (and
    /// [`Self::define_extra`]-d code stays valid across runs).
    pub fn run(
        &mut self,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        let (code, n_params, n_results) = (self.tramp_code, self.n_params, self.n_results);
        // SAFETY: `self` is a unique borrow for the whole call and this path hands no pointer
        // to any handler — the re-entrant powerbox path goes through `run_raw` instead.
        unsafe {
            Self::run_code_raw(
                self,
                code,
                n_params,
                n_results,
                args,
                init_mem,
                snapshot_cap,
                None,
            )
        }
    }

    /// [`Self::run`] through a **caller-managed raw pointer** — the entry the Phase-2 `Jit`
    /// capability path uses, because its handlers re-enter this module mid-run: the host gives
    /// the `Jit` binding a copy of `this`, calls `run_raw(this, …)`, and while the guest is
    /// suspended inside a synchronous `call.cap` the handler may call
    /// [`Self::define_extra`] / [`Self::invoke_extra`] through its copy. `run_raw` keeps no
    /// Rust reference into `*this` alive across the guarded call (see `run_code_raw`), so the
    /// handler's transient `&mut *this` aliases nothing.
    ///
    /// # Safety
    /// - `this` must point at a live `CompiledModule`, not concurrently accessed by any other
    ///   thread, and **the same pointer value** must be the one handlers use (don't re-derive a
    ///   fresh `&mut` elsewhere while the run is in flight).
    /// - Handlers may call `define_extra` / `invoke_extra` through `this` only while the guest
    ///   is suspended in a synchronous `call.cap` on this thread, and must not call `run` /
    ///   `run_raw` re-entrantly.
    pub unsafe fn run_raw(
        this: *mut CompiledModule,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        let (code, n_params, n_results) = {
            let t = &*this;
            (t.tramp_code, t.n_params, t.n_results)
        };
        Self::run_code_raw(
            this,
            code,
            n_params,
            n_results,
            args,
            init_mem,
            snapshot_cap,
            None,
        )
    }

    /// #1768 — run a **fork twin** (FORK.md §9.5): this module's entry, re-entered over `twin` — the
    /// window a [`ForkPoint::twin`] duplicated from the forking run, its reply injected and its root
    /// set rewinding — so the entry's prologue rebuilds the parent's frames and the twin resumes past
    /// the fork call with its own reply. `this` is the twin's own instance of the parent's program
    /// (its own powerbox); `args` are the parent entry's (the rewind reads none of them). Otherwise
    /// exactly [`Self::run_raw`]: the twin may fork again through this module's [`ForkHook`].
    ///
    /// # Safety
    /// As [`Self::run_raw`]; `this` is compiled from the same (fork-instrumented) program as the
    /// module whose run produced `twin`, with the same window geometry.
    pub unsafe fn run_twin(
        this: *mut CompiledModule,
        twin: TwinWindow,
        args: &[i64],
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        let (code, n_params, n_results) = {
            let t = &*this;
            (t.tramp_code, t.n_params, t.n_results)
        };
        Self::run_code_raw(
            this,
            code,
            n_params,
            n_results,
            args,
            None,
            None,
            Some(twin),
        )
    }

    /// #1825 — this module's code as [`SharedCode`], for more instances to run. `None` unless the code
    /// is **only code**: no hosting entry stood a runtime up past the compile (see `hosted`), and
    /// nothing was defined past it. From here on no instance can extend it ([`arena_mut`]) — this
    /// module included, which goes on running as the first instance.
    pub fn share(&self) -> Option<SharedCode> {
        self.is_only_code().then(|| {
            // The kept instance is never run: it names none of this one's state, and no run's node.
            SharedCode(self.instance(
                InstanceAddrs {
                    cap_ctx: core::ptr::null_mut(),
                    embedder: core::ptr::null_mut(),
                    sched: core::ptr::null(),
                    nursery: core::ptr::null(),
                    setjmp: core::ptr::null(),
                    pages: core::ptr::null(),
                    ..self.instance
                },
                None,
            ))
        })
    }

    /// See [`Self::share`].
    fn is_only_code(&self) -> bool {
        !self.hosted && self.next_extra == 0
    }

    /// Whether an instance of this code stands up its own §14 nursery ([`Self::instance`]): a root's
    /// that spawns does; a child's is its spawn's.
    fn makes_nursery(&self) -> bool {
        #[cfg(fiber_rt)]
        {
            self.nests && self._nursery.is_some()
        }
        #[cfg(not(fiber_rt))]
        {
            false
        }
    }

    /// A new instance of this module's code at `addrs` ([`SharedCode::instance`]): the compile's
    /// products, the function table copied (a run is handed its address; nothing bakes it), and fresh
    /// run state. Only for code [`Self::is_only_code`] admits.
    ///
    /// temen#2389 — each runtime the compile stood up, the instance gets one of its own, charged to
    /// `node`, its run's budget: a thread domain, the fiber table and the root vCPU's fiber runtime,
    /// the §14 nursery its `Instantiator` sites spawn through, the `setjmp` table. Its vmctx names
    /// them; nothing in the code does. Not the nursery a module with install room gets for its units
    /// (#1726): no unit can be defined into shared code. A §14 child's runtimes are its task's, and
    /// its domain and nursery are the ones `addrs` names: its parent's domain, and the nursery its
    /// spawn made for it (temen#2168).
    #[cfg_attr(not(fiber_rt), allow(unused_mut))]
    fn instance(
        &self,
        mut addrs: InstanceAddrs,
        node: Option<std::sync::Arc<dyn BudgetNode>>,
    ) -> CompiledModule {
        #[cfg(fiber_rt)]
        let domain = self
            .domain
            .as_ref()
            .map(|_| Box::new(os_thread_rt::Domain::new(node.clone())));
        #[cfg(fiber_rt)]
        if let Some(d) = &domain {
            addrs.sched = &**d as *const os_thread_rt::Domain as *const core::ffi::c_void;
        }
        #[cfg(fiber_rt)]
        let fiber_table = self.fiber_table.as_ref().map(|_| {
            std::sync::Arc::new(fiber_rt::SharedFiberTable::new(
                temen_ir::MAX_FIBERS,
                self.shadow,
                node.clone(),
            ))
        });
        #[cfg(fiber_rt)]
        let fiber_rt = match (
            &self.fiber_rt,
            &fiber_table,
            self.fiber_cfg,
            self.call_tramp,
        ) {
            (Some(_), Some(t), Some((type_id, mask)), Some(tramp)) => {
                let mut rt = Box::new(fiber_rt::FiberRuntime::new(
                    std::sync::Arc::clone(t),
                    type_id,
                    mask,
                ));
                rt.set_call_tramp(tramp);
                Some(rt)
            }
            _ => None,
        };
        #[cfg(fiber_rt)]
        let nursery = self.makes_nursery().then(|| {
            nursery_over(
                self.cap_thunk,
                addrs.cap_ctx,
                self.resolve_module,
                addrs.epoch as usize,
                domain.as_deref(),
                self.shadow,
            )
        });
        #[cfg(fiber_rt)]
        if let Some(n) = &nursery {
            addrs.nursery = &**n as *const instantiator_rt::Nursery as *const core::ffi::c_void;
        }
        #[cfg(setjmp_rt)]
        let setjmp_runtime = self
            ._setjmp_rt
            .as_ref()
            .map(|_| Box::new(setjmp_rt::SetjmpRuntime::new()));
        #[cfg(setjmp_rt)]
        if let Some(r) = &setjmp_runtime {
            addrs.setjmp = &**r as *const setjmp_rt::SetjmpRuntime as *const core::ffi::c_void;
        }
        #[cfg(not(fiber_rt))]
        let _ = node;
        CompiledModule {
            hosted: self.hosted,
            nests: self.nests,
            page: self.page,
            fn_table: self
                .fn_table
                .iter()
                .map(|e| FnEntry::new(e.type_id(), e.code()))
                .collect(),
            tramp_code: self.tramp_code,
            tramp_code_limited: self.tramp_code_limited,
            serve_tramps: self.serve_tramps.clone(),
            n_params: self.n_params,
            n_results: self.n_results,
            n_real_funcs: self.n_real_funcs,
            distinct: self.distinct.clone(),
            fast_resolver: self.fast_resolver,
            fibers: self.fibers,
            threads: self.threads,
            spawns: self.spawns,
            nesting: false,
            mask: self.mask,
            epoch: self.epoch,
            fuel: self.fuel,
            instance: addrs,
            fn_table_mask: self.fn_table_mask,
            next_extra: self.next_extra,
            extra_bytes: self.extra_bytes,
            base_bytes: self.base_bytes,
            live_fault_range: None,
            caller_window: self.caller_window,
            last_trap_backtrace: Vec::new(),
            last_trap_fiber: None,
            win_mapped: self.win_mapped,
            win_reserved: self.win_reserved,
            null_guard: self.null_guard,
            mem_size_log2: self.mem_size_log2,
            data: self.data.clone(),
            restore: None,
            tail_out: Vec::new(),
            durable: false,
            shadow: self.shadow,
            concurrent_durable: false,
            frozen_seed: Vec::new(),
            frozen_out: Vec::new(),
            frozen_vcpus_out: Vec::new(),
            frozen_root_sp_out: 0,
            frozen_vcpu_seed: Vec::new(),
            detached_out: Vec::new(),
            detached_seed: Vec::new(),
            thaw_root_sp: self.shadow.frame_base(0),
            freeze_ctl: None,
            held_tail: None,
            window_ready: None,
            fork_hook: None,
            #[cfg(fiber_rt)]
            fiber_rt,
            #[cfg(fiber_rt)]
            domain,
            #[cfg(fiber_rt)]
            _nursery: nursery,
            #[cfg(setjmp_rt)]
            _setjmp_rt: setjmp_runtime,
            #[cfg(fiber_rt)]
            cap_thunk: self.cap_thunk,
            #[cfg(fiber_rt)]
            resolve_module: self.resolve_module,
            #[cfg(fiber_rt)]
            call_tramp: self.call_tramp,
            #[cfg(fiber_rt)]
            fiber_cfg: self.fiber_cfg,
            #[cfg(fiber_rt)]
            fiber_table,
            #[cfg(fiber_rt)]
            node,
            src_ranges: self.src_ranges.clone(),
            src_files: self.src_files.clone(),
            func_names: self.func_names.clone(),
            var_locs: self.var_locs.clone(),
            debug_types: self.debug_types.clone(),
            module: Arc::clone(&self.module),
        }
    }

    /// #1768 — hand this module's own instance the embedder's per-instance state (the vmctx
    /// `embedder` slot, [`VmCtx::embedder`]): its `call.cap` thunk finds it through `vmctx`.
    pub fn set_embedder_ctx(&mut self, ctx: *mut core::ffi::c_void) {
        self.instance.embedder = ctx;
    }

    /// #1768 — arm this module's runs to **fork**: a run whose root unwinds at a fork hands the
    /// live window to `hook` and, with the reply the hook returns, rewinds the root in place and runs
    /// on (see [`ForkHook`]). Only for a program instrumented to unwind at its fork calls, run as a
    /// single-threaded, non-durable root — the embedder's contract; `None` disarms.
    pub fn set_fork_hook(&mut self, hook: Option<ForkHook>) {
        self.fork_hook = hook;
    }

    /// Make the next run **durable** (DURABILITY.md §12.8) on a caller-built module — the
    /// fiber-hosting / `Jit`-domain setup (`enable_fiber_hosting`, reconstructed units) stays the
    /// caller's, so an embedder like `temen_run::jit_cap_run` needs no second durable entry point.
    /// `seed` is the thaw seed: the residue to re-create before a `REWINDING` run (empty for a freeze
    /// or an ordinary durable run). The freeze residue comes back via [`Self::take_durable_residue`].
    pub fn set_durable(&mut self, seed: DurableResidue) {
        self.frozen_seed = seed.fibers;
        self.frozen_vcpu_seed = seed.vcpus;
        if let Some(sp) = seed.root_sp {
            self.thaw_root_sp = sp;
        }
        self.durable = true;
    }

    /// The residue the last durable run's freeze produced (empty unless a freeze caught something);
    /// taken, so a later run starts clean.
    pub fn take_durable_residue(&mut self) -> DurableResidue {
        let root_sp = std::mem::take(&mut self.frozen_root_sp_out);
        DurableResidue {
            fibers: std::mem::take(&mut self.frozen_out),
            vcpus: std::mem::take(&mut self.frozen_vcpus_out),
            root_sp: (root_sp != 0).then_some(root_sp),
        }
    }

    /// #1361 step 4 — the durable detached children the last run's freeze reached (see
    /// [`DetachedHarvest`]); taken. Each carries a powerbox ref the caller now owns.
    pub fn take_detached_harvest(&mut self) -> Vec<DetachedHarvest> {
        std::mem::take(&mut self.detached_out)
    }

    /// #1361 step 4 — the detached children the next (thaw) run re-creates, each at its recorded join
    /// slot, before the root re-enters under `REWINDING`.
    pub fn set_detached_seed(&mut self, seed: Vec<DetachedSeed>) {
        self.detached_seed = seed;
    }

    /// Run an **incrementally defined** function (a trampoline pointer returned by
    /// [`Self::define_extra`]) over a fresh guest window, exactly like [`Self::run`] runs the
    /// entry, and return the window's backed prefix with the outcome. This is the test/demo
    /// surface; the Phase-2 `Jit` capability instead uses [`Self::invoke_extra`] over the *live*
    /// window of an in-flight run.
    ///
    /// # Safety
    /// `code` must be a trampoline pointer returned by `define_extra` **on this module**, and
    /// `n_params`/`n_results` must match the parameter/result counts of the function it wraps
    /// (the trampoline reads exactly `n_params` arg slots and writes exactly `n_results` result
    /// slots).
    pub unsafe fn run_extra(
        &mut self,
        code: *const u8,
        n_params: usize,
        n_results: usize,
        args: &[i64],
        init_mem: Option<&[u8]>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        Self::run_code_raw(
            self,
            code,
            n_params,
            n_results,
            args,
            init_mem,
            Some(0),
            None,
        )
    }

    /// #2173 — the run's park hub: the address of its thread domain, whose futex table every park
    /// of the run's vCPUs and §14 children keys into. A waker that holds no call's trap cell rings
    /// those parks through it ([`wake_host_parks_in`]). `0` when the run has no domain — no threads,
    /// fibers or children, so nothing else of the run can park.
    pub fn park_hub(&self) -> usize {
        #[cfg(fiber_rt)]
        {
            self.domain
                .as_ref()
                .map_or(0, |d| &**d as *const os_thread_rt::Domain as usize)
        }
        #[cfg(not(fiber_rt))]
        {
            0
        }
    }

    /// CALLS.md 5c.1c — install the granted-child host callbacks into this module's §14 nursery
    /// (the production twin of the `compile_and_run_*_ex` install; a module with no `Instantiator`
    /// has no nursery and this is a no-op). `None` leaves the granted spawns inert `CapFault`s.
    pub fn set_grant_child_hooks(&self, hooks: Option<GrantChildHooks>) {
        #[cfg(fiber_rt)]
        if let Some(n) = &self._nursery {
            n.set_grant_hooks(hooks);
        }
        // D66 — the same call installs this run's own lane: the hooks already carry the host's
        // `(domain_id, lane_cap)` (that is how a §14 child's chain gets its parent entry), and the
        // domain needs the identical pair so its 1:1 vCPUs and its children's tasks draw on **one**
        // lane rather than two copies of the cap. Done here rather than at the nursery so a module
        // that only uses `thread.*` — and therefore has no nursery at all — is still bounded.
        #[cfg(fiber_rt)]
        if let (Some(d), Some(h)) = (&self.domain, hooks.as_ref()) {
            d.set_lane_chain(h.parent_domain, h.parent_lane_cap);
        }
        #[cfg(not(fiber_rt))]
        let _ = hooks;
    }

    /// temen#2168 — compile this run's §14 children into `cache` ([`ChildCodeCache`]), which an
    /// embedder that keeps it hands its later runs too, so each child compiles once for all of them.
    /// Call before the run. Without it the run's children share a cache of the run's own; a module
    /// with no nursery spawns nothing, and ignores it.
    pub fn set_child_code(&self, cache: ChildCodeCache) {
        #[cfg(fiber_rt)]
        if let Some(n) = &self._nursery {
            n.set_child_code(cache);
        }
        #[cfg(not(fiber_rt))]
        let _ = cache;
    }

    /// Install the run's async freeze trigger (see [`FreezeController`]); `None` for none. What
    /// [`DurableRun::freeze`] sets for [`compile_and_run_durable`], for an embedder that compiles and runs
    /// the module itself.
    pub fn set_freeze_controller(&mut self, fc: Option<Arc<FreezeController>>) {
        self.freeze_ctl = fc;
    }

    /// #1810, #2427 — make a snapshotting run's capture its declared window and each page the guest
    /// holds past it, as `hook` reports them for the live window ([`Self::take_tail_out`]). For a
    /// **durable** run, whose capture is its freeze image; other captures keep their fixed span (a
    /// grown heap is not worth copying for them).
    pub fn set_held_tail(&mut self, hook: Option<(HeldTail, *mut core::ffi::c_void)>) {
        self.held_tail = hook;
    }

    /// #2427 — the pages the last durable run's capture read past its declared window, by byte offset
    /// ([`Self::set_held_tail`]): its image's tail, taken.
    pub fn take_tail_out(&mut self) -> Vec<(u64, Vec<u8>)> {
        std::mem::take(&mut self.tail_out)
    }

    /// #2025 — run `hook` over each fresh root window once it is seeded and protected, before the
    /// guest runs ([`WindowReady`], with `ctx` the root's powerbox): the embedder's page map records
    /// the read-only data pages (#2268), and a durable thaw's §13 pages are aliased back onto their
    /// regions, as a child's pre-map is. `0` from it fails the run.
    pub fn set_window_ready(&mut self, hook: Option<(WindowReady, *mut core::ffi::c_void)>) {
        self.window_ready = hook;
    }

    /// #1834, #2441 — the thaw's window the next run restores ([`WindowRestore`]): its `init_mem` is
    /// that window, under its page map (DURABILITY.md §12.3), a page past the backed prefix committed
    /// and seeded from it when `Rw`/`Ro`; then (#2427) each page the window holds past `init_mem`.
    /// `None` (the default) builds a fresh instance's window, `init_mem` its low bytes and its data
    /// segments laid over them.
    pub fn set_restore(&mut self, restore: Option<WindowRestore>) {
        self.restore = restore;
    }

    pub fn handler_tramp(&self, func: u32) -> Option<(*const u8, usize, usize)> {
        self.serve_tramps
            .iter()
            .find(|&&(f, ..)| f == func)
            .map(|&(_, code, np, nr)| (code, np, nr))
    }

    /// Invoke an extra trampoline **over the live window of an in-flight run** — the engine of
    /// the `Jit` capability's `invoke` op. Called from inside a `call.cap` handler while the
    /// guest is suspended; `mem_base`/`vmctx` are the values the cap thunk received (the run's
    /// window base, and its `vmctx` — the running instance's [`VmCtx`]), so the invoked code
    /// reads/writes the guest's own memory in place, dispatches its own `call.cap`s into the same
    /// powerbox, and a trap in it propagates exactly like a guest trap.
    ///
    /// Runs under a **nested** detect-and-kill recovery (`run_guarded_range` is re-entrant): a
    /// memory fault in the invoked code is caught *here*, written to the trap cell as `MemoryFault`,
    /// and this returns normally — the guest's `call.cap` trap-propagation check then unwinds the
    /// domain. Traps in invoked code are **terminal for the domain** (DESIGN.md §22); a guest wanting
    /// trap isolation uses the `Instantiator`, not `Jit`.
    ///
    /// # Safety
    /// - `this` must be the pointer an in-flight [`Self::run_raw`] on **this thread** is
    ///   executing on (the guest suspended in its synchronous `call.cap`), and the caller must
    ///   hold no Rust reference into `*this` across this call.
    /// - `code` must be a trampoline returned by `define_extra` on this module; `args` must
    ///   cover its param count and `results` its result count.
    /// - `mem_base` and `vmctx` must be the live run's window base and instance context (the cap
    ///   thunk's `mem_base` / `vmctx`).
    pub unsafe fn invoke_extra(
        this: *mut CompiledModule,
        code: *const u8,
        args: &[i64],
        results: &mut [i64],
        mem_base: *mut u8,
        vmctx: *const VmCtx,
    ) -> Result<(), JitError> {
        let (fn_table_ptr, live, caller_window, reserved) = {
            let t = &*this;
            (
                t.fn_table.as_ptr() as *const core::ffi::c_void,
                t.live_fault_range,
                t.caller_window,
                t.win_reserved,
            )
        };
        // A root run records its window at entry (`run_code_raw`); outside a run there is no live
        // window to confine against — refuse. A §14 child's module never records one (see
        // `caller_window`): its live window is exactly the `mem_base` its thunk was handed, whose
        // extent (reservation + guard) is baked in `win_reserved` (#1296).
        let (lo, hi) = match live {
            Some(r) => r,
            None if caller_window => mem::fault_range_of(mem_base, reserved),
            None => {
                return Err(JitError::Unsupported(
                    "invoke_extra outside an in-flight run",
                ))
            }
        };
        let _reentry = reentry::enter();
        let faulted = mem::run_guarded_range(
            code,
            args.as_ptr(),
            results.as_mut_ptr(),
            mem_base,
            fn_table_ptr,
            vmctx,
            lo,
            hi,
        );
        if faulted {
            // Detect-and-kill (§5), reported the same way the outer run reports it; the
            // guest's call.cap propagation check sees the cell and unwinds the domain.
            (*vmctx).raise(mem::FAULT_TRAP);
        }
        Ok(())
    }

    /// A run's window: a fresh instance's — `init_mem` (the escape-oracle's or an embedder's low
    /// bytes), the data segments over it with the `readonly` ones made read-only — or a thaw's
    /// ([`WindowRestore`]): `init_mem` as it is, under its captured page map. Then the NULL guard.
    /// With it, the read-only ranges the data segments made, for the embedder's page map
    /// (`window_ready`). `None`, nothing kept, when the OS refuses the reservation or any commit or
    /// protection (#2305, #2286): a window missing one would let the guest write or read where the
    /// interpreter traps.
    fn seed_window(&self, init_mem: Option<&[u8]>) -> Option<(mem::GuestWindow, Vec<[u64; 2]>)> {
        let mut window = mem::GuestWindow::try_new(self.win_mapped, self.win_reserved)?;
        // Escape-oracle: seed the window's low bytes so a divergent read/store is observable.
        if let Some(init) = init_mem {
            let n = init.len().min(self.win_mapped);
            window.rw_mut()[..n].copy_from_slice(&init[..n]);
        }
        // Initialized data segments (§3a / D40), a fresh instance's alone: a thaw's image holds them
        // as the guest left them (#2441), and its page map says which of their pages are read-only
        // (#2307). Copy each segment's bytes into the window, then map the `readonly` ones RO (so a
        // guest write to const data faults into the guard, §4/§5). The verifier already bounds every
        // segment to `[0, size)`. Done while fully RW.
        let mut ro: Vec<[u64; 2]> = Vec::new();
        if let (None, Some(size_log2)) = (&self.restore, self.mem_size_log2) {
            let size = 1u64 << size_log2;
            let rw = window.rw_mut();
            for d in &self.data {
                let lo = d.offset.min(size);
                let hi = (d.offset + d.bytes.len() as u64).min(size);
                let (start, end) = (lo as usize, hi as usize);
                rw[start..end].copy_from_slice(&d.bytes[..end - start]);
            }
            // Map the `readonly` segments RO (so a guest write to const data faults into the guard,
            // §4/§5). Clamp the range to `[0, size)` exactly as the copy loop above: the verifier
            // already bounds every data segment into the window, so this is defensive consistency —
            // an out-of-window segment must never `mprotect` past the backed region.
            for d in self.data.iter().filter(|d| d.readonly) {
                let lo = d.offset.min(size);
                let hi = d.offset.saturating_add(d.bytes.len() as u64).min(size);
                if hi > lo {
                    if !window.protect(lo, hi - lo, mem::Prot::Ro) {
                        return None;
                    }
                    ro.push([lo, hi]);
                }
            }
        }
        // Durable restore (DURABILITY.md §12.3): re-establish a thaw's captured per-page protections
        // on its image so the thawed guest faults on an `Ro`/`Unmapped` page exactly as the frozen
        // one would — matching `temen-interp`'s `restore_layout`. A page held past the prefix is
        // committed and seeded here (#1834, #2427).
        // #964: then reserve the marked module's NULL region last, after every host write above
        // (init seed, data segments, durable prots) has landed — the marked layout keeps all
        // live data at or above the guard, so nothing legitimate is covered. A window smaller than
        // the guard skips, matching the interpreter's fit check. The snapshot paths re-assert RW
        // (`restore_rw`/`read_low`) before any host read.
        let restored = self.restore.as_ref().is_none_or(|r| {
            window.apply_prots(
                &r.prots,
                self.win_mapped as u64,
                init_mem.unwrap_or(&[]),
                &r.tail,
            )
        });
        let seeded = restored
            && (self.null_guard > self.win_mapped as u64
                || window.seed_null_guard(self.null_guard));
        seeded.then_some((window, ro))
    }

    /// The shared run body: window setup → guarded call → snapshot → teardown. `code` is a
    /// buffer-ABI trampoline owned by the module (the entry's, or an extra function's).
    /// `snapshot_cap` is what of the window the run copies back: `None`, nothing; `Some(cap)`, the
    /// backed prefix, widened to `cap` bytes (so `Some(0)` is the backed prefix alone).
    ///
    /// Structured for **mid-run re-entry**: every reference into `*this` is derived
    /// transiently and dropped before the guarded call (raw pointers extracted up front), so a
    /// `call.cap` handler re-entering through its own copy of `this` while the guest is
    /// suspended aliases no live Rust reference. The fields a re-entrant `define_extra`
    /// mutates (`module`, `distinct`, `next_extra`) are disjoint from everything the in-flight
    /// call reads through raw pointers (`fn_table` is boxed, never grown or moved; the window
    /// is a local).
    ///
    /// # Safety
    /// As [`Self::run_raw`]; additionally `code`/`n_params`/`n_results` must describe a
    /// trampoline owned by this module.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run_code_raw(
        this: *mut CompiledModule,
        code: *const u8,
        n_params: usize,
        n_results: usize,
        args: &[i64],
        init_mem: Option<&[u8]>,
        snapshot_cap: Option<usize>,
        // #1768 — a fork twin's window, already duplicated and set rewinding ([`Self::run_twin`]);
        // `None` builds a fresh window from the module (and `init_mem`).
        twin: Option<TwinWindow>,
    ) -> Result<(JitOutcome, Vec<u8>), JitError> {
        // The trampoline reads exactly `n_params` arg slots; a shorter buffer would be an
        // out-of-bounds read from safe code. (The one-shot wrappers always pass exact-length
        // args; this check makes the now-public entry sound rather than contractual.)
        if args.len() < n_params {
            return Err(JitError::Malformed);
        }
        // §4 (DURABILITY.md): a durable run's §14 nursery must know it — its `instantiate` /
        // `coro_spawn` thunks fail closed there (this JIT child runner cannot yet run a durable
        // child; the interpreter is the reference for durable nesting). Set here, the common
        // bottom of every run entry, because the durable flag is applied by the entry wrappers
        // *after* compile (where the nursery is built).
        {
            let t = &*this;
            if let Some(n) = &t._nursery {
                n.set_durable(t.durable);
            }
        }
        // ---- Setup: references into `*this` live only inside this block. ----
        // Allocate the guest window for this run: `mapped` backed RW bytes inside the reserved
        // virtual range planned at compile time (§4); zero-sized when the module has no memory.
        let twin_tls = twin.as_ref().map(|t| t.tls);
        let (mut window, win_size, mask, fn_table_ptr) = if let Some(twin) = twin {
            // A fork twin's window is its parent's, byte for byte and page for page: seeded,
            // protected and set rewinding by the fork (`ForkPoint::twin`), so none of the fresh
            // window's seeding below applies.
            let t = &*this;
            let fn_table_ptr = t.fn_table.as_ptr() as *const core::ffi::c_void;
            (twin.window, t.win_mapped, t.mask, fn_table_ptr)
        } else {
            let t = &mut *this;
            // #2305/#2286 — a run the OS will not give its window to ends as one the process cannot
            // hold (#2268): `ThreadFault` before its first instruction, never a host panic. That is
            // no room for the reservation, or a commit or protection the window's seeding needs
            // refused. Nothing ran, so a snapshot is the window it was handed, unchanged (a session
            // keeps its state).
            let Some((window, ro)) = t.seed_window(init_mem) else {
                t.last_trap_backtrace = Vec::new();
                t.last_trap_fiber = None;
                let image = snapshot_cap
                    .and(init_mem)
                    .map_or_else(Vec::new, <[u8]>::to_vec);
                return Ok((JitOutcome::Trapped(TrapKind::ThreadFault), image));
            };
            let win_size = t.win_mapped;
            if let Some((ready, ctx)) = t.window_ready {
                if ready(
                    ctx,
                    window.base(),
                    t.win_mapped as u64,
                    t.win_reserved as u64,
                    ro.as_ptr(),
                    ro.len(),
                    t.shadow,
                ) == 0
                {
                    return Err(JitError::Unsupported(
                        "durable JIT thaw: a §13 page did not re-alias",
                    ));
                }
            }
            // #1768 — a run armed to fork keeps its root's shadow stack in context 0 of the
            // declared arena; a fresh window starts it empty (the durable `init_durable_window`
            // seed), so the first fork's unwind pushes from the frame base.
            if t.fork_hook.is_some() && t.shadow.region_fits(0) {
                let sp_word = window.base().add(t.shadow.region_base(0) as usize) as *mut u64;
                sp_word.write_unaligned(t.shadow.frame_base(0));
            }
            let fn_table_ptr = t.fn_table.as_ptr() as *const core::ffi::c_void;
            (window, win_size, t.mask, fn_table_ptr)
        };

        let mem_base = window.base();
        let mut results = vec![0i64; n_results];
        // This run's instance context: the module's own instance (#1768 — the powerbox, kill-path
        // cell, fuel cell and signal arm its compile was handed), entered by the root and shared with
        // every vCPU and fiber of the run. Its field 0 is the trap cell — how the domain ended: 0
        // while it runs; low 32 bits = a TrapKind / EXIT_CODE; high 32 bits = the exit code for an
        // Exit. It is **shared across vCPU threads** (every spawned vCPU gets its address via
        // `set_env`), so every end is a compare-and-swap from 0 — emitted code's and the runtime's
        // alike ([`VmCtx::raise`]) — and the first end wins (#2499).
        let vm = VmCtx::new((*this).instance);

        // §12: the root vCPU (`main`) runs on this thread under the §5 detect-and-kill guard; any spawned
        // vCPUs run on their own OS threads via the instance's `Domain`. Seed its per-run `Env` now
        // that the window / fn-table / trap-cell / call-trampoline are all known.
        #[cfg(fiber_rt)]
        {
            let t = &*this;
            if let Some(d) = &t.domain {
                d.set_env(
                    mem_base as u64,
                    fn_table_ptr as u64,
                    &vm, // spawned vCPUs are this instance's
                    // `None` on a futex-only domain (a nesting module with no `thread.*`/`cont.*` ops
                    // of its own): no spawn site exists to call through it.
                    t.call_tramp,
                    window.fault_range(),
                    t.fiber_cfg,
                    t.fiber_table.clone(), // the domain-shared table spawned vCPUs build over
                    t.instance.epoch as usize, // §5 kill-path: so parked vCPUs (futex/join) observe the interrupt
                    t.durable, // slice 3.3: run spawned children inline (single-worker) under a freeze/thaw
                    t.shadow,  // #2518: where its freeze word is
                );
            }
        }

        // Set if a durable thaw re-seed (below) hit a control-stack alloc failure (I1): the trap cell
        // already carries the `FiberFault`, so we skip the root re-entry and report it post-run.
        // (Only the `fiber_rt` build re-seeds, so it's never mutated otherwise.)
        #[cfg_attr(not(fiber_rt), allow(unused_mut))]
        let mut seed_faulted = false;

        // Publish the root fiber runtime (when the module uses `cont.*`) so its thunks find it via the
        // thread-local for the duration of the entry; spawned vCPUs publish their own.
        #[cfg(fiber_rt)]
        let prev_rt = {
            // The OS-thread stack pointer in *this* (`run_inner`) frame — above the guarded guest
            // call below and every guest root frame it pushes — is the high bound for `gc.roots`'
            // scan of the root computation's frames. Captured via a local's address here (not a
            // sub-call) so it provably dominates the guest's frames.
            let entry_probe = 0u8;
            let entry_sp = std::hint::black_box(&entry_probe as *const u8 as usize);
            let t = &mut *this;
            let durable = t.durable;
            // Durable **thaw** (slice 3.3.3): re-create the frozen fibers in the run-shared table
            // before the root re-enters under REWINDING, so a thaw `cont.resume` resolves + re-enters
            // them. Done before `set_current` (and the run), while the window/table/trap cell are set.
            let seed = std::mem::take(&mut t.frozen_seed);
            t.fiber_rt.as_mut().map(|rt| {
                rt.set_root_entry_sp(entry_sp);
                // Arm the durable fiber-switch swap for the root vCPU (DURABILITY.md §12.8): the
                // window base is known now. (Spawned vCPUs are multi-vCPU durability, Phase 3.2.)
                rt.set_durable_env(mem_base as u64, durable);
                if durable && !seed.is_empty() {
                    // A thaw re-seed that the OS refuses (I1) writes a `FiberFault` to the trap cell
                    // and returns false; the post-run trap read below reports it. We still publish
                    // the runtime and fall through — the guest re-entry simply won't resolve the
                    // missing fibers — rather than abort the host.
                    seed_faulted = !fiber_rt::seed_frozen_fibers(
                        &mut **rt as *mut fiber_rt::FiberRuntime,
                        &seed,
                        mem_base as u64,
                        fn_table_ptr as u64,
                        &vm as *const VmCtx as u64, // the fibers are this instance's
                    );
                }
                fiber_rt::set_current(&mut **rt as *mut fiber_rt::FiberRuntime)
            })
        };

        // §12.8 concurrent-thaw stage 2: the parked-vCPU thaw no longer needs a `Domain.thawing` flag —
        // the thaw re-spawns frozen vCPUs as concurrent OS threads (below), so a re-issued `atomic.wait`
        // parks on the real futex and a sibling's re-issued `notify` wakes it; a wait with no possible
        // notifier left fails closed via `futex_wait`'s shared deadlock detection (no thaw-specific path).

        // Durable **multi-vCPU thaw** (slice 3.3, thaw side): re-attach + run the spawned children a
        // freeze flattened, *before* the root re-enters — the JIT's single-worker thaw (the root's
        // rewind skips its prologue `thread.spawn`, so the runtime reconstructs the children). Each
        // child rewinds from its restored extent, runs forward to completion, and publishes its result
        // so the root's re-executed `thread.join` resolves; the root's active shadow-SP is then set to
        // its restored extent. Done after the fiber seed (the table is live) and `set_env`.
        #[cfg(fiber_rt)]
        if (*this).durable && !(*this).frozen_vcpu_seed.is_empty() {
            let seed = std::mem::take(&mut (*this).frozen_vcpu_seed);
            let root_sp = (*this).thaw_root_sp;
            if let Some(d) = &(*this).domain {
                d.thaw_reattach_and_run(&seed, root_sp);
            }
        }

        // #1361 step 4 — re-create the detached children at their recorded slots before the root
        // re-enters: a captured one re-launched on its own restored window under `REWINDING`, a
        // completed one (#2041) as its outcome. The root's rewound `join` then parks on, or reloads,
        // each exactly as before the cut.
        #[cfg(fiber_rt)]
        if (*this).durable && !(*this).detached_seed.is_empty() {
            let seed = std::mem::take(&mut (*this).detached_seed);
            if let Some(n) = &(*this)._nursery {
                for d in seed {
                    n.relaunch_detached(d);
                }
            }
        }

        // Publish the live window's fault range so a mid-run `invoke_extra` (from a call.cap
        // handler) can arm its nested recovery against this run's window.
        (*this).live_fault_range = Some(window.fault_range());

        // ---- The guarded call: NO Rust reference into `*this` is live here, so a handler's
        // ---- transient `&mut *this` (define_extra / invoke_extra mid-run) aliases nothing.
        // SAFETY: `code` is a finalized buffer-ABI trampoline honouring the `Entry` ABI. It reads the
        // arg slots, writes the result slots, accesses only the guarded window (any escape faults into
        // the guard page), reads `fn_table`, and raises on `vm`. All buffers outlive the call;
        // the module owns the executable pages until `*this` drops (after every spawned vCPU is
        // joined below).
        // §12 seed the root vCPU's TLS register to 0 (its dense id), resetting any value a reused
        // worker thread carries from a prior run before guest code can `vcpu.tls.get` it.
        // A fork twin's register is the parent's at the fork (POSIX: the child's thread state is
        // a copy of the forking thread's).
        vcpu_tls::seed(twin_tls.unwrap_or(0));
        // §12.8 4A.5: seed the durable shadow-base register to the root's region (context 0 =
        // `DURABLE_SHADOW_BASE`), so the root's instrumented code addresses its own per-context
        // shadow-SP word.
        durable_shadow::seed((*this).shadow.region_base(0));
        // §12.8 4A.5 stage (ii): engage the concurrent durable path before the guarded call (where the
        // root may `thread.spawn` children) so each child reserves its own shadow context.
        #[cfg(fiber_rt)]
        if (*this).concurrent_durable {
            if let Some(d) = &(*this).domain {
                d.engage_concurrent_durable();
            }
        }
        // Phase-4 Slice A (4A.3): publish the live window base for an async controller right before the
        // guarded call (the guest is about to block in its loop), and retire it right after — so a
        // `request_freeze` can only ever store into the window while it is mapped (freed below).
        if let Some(fc) = &(*this).freeze_ctl {
            let word = (*this).shadow.freeze_word();
            fc.publish(word.map(|w| mem_base as usize + w as usize));
        }
        // D66 — the root vCPU takes a lane like any other task before it runs guest code, and gives
        // it back when its computation ends. Without this the root would be exempt from the cap its
        // own children are held to, which is the whole of INVARIANTS #3 read backwards. A run with
        // no cap has an empty chain and skips every line of this.
        #[cfg(fiber_rt)]
        let root_lane = (*this)
            .domain
            .as_ref()
            .map(|d| d.lane_chain())
            .unwrap_or_default();
        #[cfg(fiber_rt)]
        let root_has_lane = match &(*this).domain {
            // The root is the first task of its domain, so this only ever waits when a cap of 0
            // makes it unsatisfiable — which is a `ThreadFault`, matching the interpreter.
            Some(d) => d.lane_acquire(&root_lane, || false),
            None => true,
        };
        #[cfg(fiber_rt)]
        if !root_has_lane {
            vm.raise(TrapKind::ThreadFault as i64);
        }
        #[cfg(not(fiber_rt))]
        let root_has_lane = true;
        let faulted = if seed_faulted || !root_has_lane {
            // A thaw re-seed already failed and wrote the trap; don't re-enter with missing fibers.
            false
        } else {
            loop {
                let faulted = mem::run_guarded(
                    &window,
                    code,
                    args.as_ptr(),
                    results.as_mut_ptr(),
                    mem_base,
                    fn_table_ptr,
                    &vm, // the root enters as the module's own instance
                );
                // #1768 — the root returned *unwound*, at a fork (the only unwind a fork-armed,
                // non-durable run has): hand the frozen window to the embedder, which duplicates the
                // process, then rewind this root in place with the reply it gives, as the twin will
                // with its own. A trap or fault ends the run as usual.
                let forked = !faulted
                    && vm.ended().is_none()
                    && !(*this).durable
                    && (*this).fork_hook.is_some()
                    && window_state(mem_base, (*this).shadow)
                        == temen_ir::durable_abi::STATE_UNWINDING;
                if !forked {
                    break faulted;
                }
                let reply = {
                    // The hook is taken out for the call, so the fork point's shared view of the
                    // module never overlaps the hook's own `&mut`.
                    let mut hook = (*this).fork_hook.take().expect("fork-armed");
                    let t = &*this;
                    let point = ForkPoint {
                        window: &window,
                        null_guard: (t.null_guard <= t.win_mapped as u64).then_some(t.null_guard),
                        shadow: t.shadow,
                        args,
                    };
                    let reply = hook(&point);
                    (*this).fork_hook = Some(hook);
                    reply
                };
                arm_fork_rewind(mem_base, (*this).shadow, reply);
            }
        };
        // The root's computation has ended: its lane goes back, so a still-running child task or a
        // spawned vCPU queued behind it can take it (teardown below joins them).
        #[cfg(fiber_rt)]
        if root_has_lane {
            if let Some(d) = &(*this).domain {
                d.lane_give_back(&root_lane);
            }
        }
        if let Some(fc) = &(*this).freeze_ctl {
            let landed = fc.retire();
            // A request that landed after the root's last poll froze nothing: the root returned (or
            // trapped) without unwinding, so its shadow-SP never left its frame base (an unwinding
            // root spills at least its entry frame). Undo the mark, so the run reads as the
            // uninterrupted one it was rather than as a freeze whose artifact holds a finished run.
            // Only when the root is the last live vCPU: a concurrent child may have seen the mark and
            // be mid-unwind, and flipping the word under it would resume it on placeholder values.
            #[cfg(fiber_rt)]
            if landed
                && fiber_rt::read_shadow_sp(mem_base as u64, (*this).shadow.region_base(0))
                    <= (*this).shadow.frame_base(0)
                && (*this).domain.as_ref().is_none_or(|d| d.live_vcpus() == 1)
            {
                // A request landed, so the window has a freeze word: the published one.
                if let Some(w) = (*this).shadow.freeze_word() {
                    (*((mem_base as usize + w as usize) as *const AtomicI32))
                        .store(temen_ir::durable_abi::STATE_NORMAL, Ordering::Release);
                }
            }
            #[cfg(not(fiber_rt))]
            let _ = landed;
        }

        // §5 W3 — the **root vCPU's** trap-time backtrace capture (this run thread's thread-local): a
        // memory fault's SIGSEGV/SIGBUS-handler walk (Stage 1) or an explicit trap's helper walk
        // (Stage 2). Taken raw now (and cleared, so a clean run stays empty); symbolized after
        // `join_all` below, where it is reconciled with any **spawned vCPU's** capture (Stage 3 —
        // collected into the domain because a worker's thread-local dies with the worker).
        let root_trap_cap = mem::take_trap_frame();

        // Durable freeze driver (DURABILITY.md §12.8 slice 3.3.2): on a durable **freeze** run
        // (state word UNWINDING) the root has now unwound into context 0's shadow region; flatten
        // every still-parked fiber into its own region before the window is snapshotted, so the
        // artifact captures their continuations. CURRENT_RT is still the root runtime here. A
        // flattening fiber touches only the committed reserve, so it's sound outside the guard.
        // This drives the **root's** own fibers; a spawned child flattens *its* fibers in
        // `run_child_inline` (slice 3.4), merged below. Skipped on a fault or a non-freeze run. The
        // residue (incl. any fiber unwound mid-resume-chain during the root run, slice 3.2) is
        // accumulated in the runtime by each fiber's `Complete` arm; drain it after driving — then the
        // deferred children run (`drive_frozen_spawns`).
        #[cfg(fiber_rt)]
        if (*this).durable
            && !faulted
            && fiber_rt::window_is_unwinding(mem_base as u64, (*this).shadow)
        {
            if let Some(rt) = (*this).fiber_rt.as_mut() {
                let rt = &mut **rt as *mut fiber_rt::FiberRuntime;
                fiber_rt::freeze_drive(rt, &vm as *const VmCtx as u64);
                (*this).frozen_out = fiber_rt::take_frozen(rt); // read back by the durable entry
            }
            // Slice 3.3: capture the **root's** flattened extent now — the freeze driver restored the
            // active shadow-SP to context 0's region (root rewinds first on thaw), but the children
            // below overwrite the shared word with their own extents, so the root's must be read here
            // (its implicit residue, reported separately for a thaw to restore).
            // §12.8 4A.5: the root's shadow-SP word lives in its own region (context 0); children no
            // longer share it.
            (*this).frozen_root_sp_out =
                fiber_rt::read_shadow_sp(mem_base as u64, (*this).shadow.region_base(0));
            // Slice 3.3: each `thread.spawn` during the freeze *deferred* its child (recording the
            // request, returning the handle) so the root could unwind first — matching the interp,
            // which enqueues a child and runs it only after the spawning vCPU yields. Now that the
            // root has unwound and its fibers are flattened, run the deferred children **inline**
            // (single-worker) in spawn order: each unwinds into its own top-down context and records a
            // `FrozenVCpu`. This reproduces the interp's dispatch order (root → root's fibers →
            // children), so the side-effect interleaving — and the frozen window — is byte-identical.
            if let Some(d) = &(*this).domain {
                d.drive_frozen_spawns(0, (*this).shadow.region_base(0));
                // §12.8 4A.5 stage (ii): drains the **deferred** (single-worker) children now;
                // **concurrent** children record their residue on their own OS threads, drained again
                // after `join_all` below. `extend` (vs. assign) so both contribute. Canonical sort at
                // serialize means the append order can't affect the artifact (§12.6).
                (*this).frozen_vcpus_out.extend(d.take_frozen_vcpus());
                // Slice 3.4: a spawned child that owns fibers flattened them with its own
                // `freeze_drive` (in `run_child_inline`) into the domain accumulator; merge that into
                // the run residue alongside the root's. Sort-by-slot at serialize is canonical, so the
                // append order doesn't affect the artifact.
                (*this).frozen_out.extend(d.take_frozen_fibers());
            }
            // #1361 step 4 — reach its detached children, which own windows the freeze word above
            // is not in: ring each one's own; they unwind and are harvested at teardown.
            if let Some(n) = &(*this)._nursery {
                n.ring_detached();
            }
        }

        // ---- Teardown: transient references again. ----
        (*this).live_fault_range = None;
        #[cfg(fiber_rt)]
        if let Some(p) = prev_rt {
            fiber_rt::set_current(p);
        }
        // S1c: join every async §14 child OS thread before freeing the window — a child copies to/from
        // the parent window, so none may outlive it (mirrors the vCPU `join_all` just below).
        #[cfg(fiber_rt)]
        if let Some(n) = &(*this)._nursery {
            let froze = (*this).durable
                && !faulted
                && fiber_rt::window_is_unwinding(mem_base as u64, (*this).shadow);
            (*this).detached_out = n.join_children(froze);
        }
        // A caught guard fault is detect-and-kill (§5): the root's `MemoryFault` ends its domain,
        // unless a sibling's end came first (#2499).
        if faulted {
            vm.raise(mem::FAULT_TRAP);
        }
        // Join every spawned vCPU OS thread before freeing the window — no vCPU may outlive it.
        #[cfg(fiber_rt)]
        if let Some(d) = &(*this).domain {
            // Owner decision 2026-07-24 — domain teardown (DESIGN.md §12; ISSUES.md I37; D37
            // death-is-revocation): the root vCPU has finished, which ends the domain — a trap or
            // `exit` from any vCPU already sits in the shared cell, and a clean root return raises
            // the internal `DOMAIN_DONE_CODE` sentinel ("root completion ends the activation").
            // Either way every parked sibling (futex wait / join) is woken to observe the non-zero
            // cell and unwind through its trailing trap-propagation guard, so `join_all` below
            // returns promptly instead of hanging on a parked daemon. A durable **freeze** is not
            // teardown: the root unwound under `UNWINDING` and parked siblings return through the
            // freeze re-issue machinery (the sentinel is skipped so their residue is recorded).
            let freezing = (*this).durable
                && !faulted
                && fiber_rt::window_is_unwinding(mem_base as u64, (*this).shadow);
            d.begin_teardown(freezing);
            d.join_all(true);
            // §12.8 4A.5 stage (ii): concurrent durable children self-unwound into their own regions
            // and recorded their `FrozenVCpu` residue before their OS threads ended; `join_all` is the
            // coordinator-wait (every child finished). Collect that residue now — after the join, so the
            // snapshot below captures a fully-quiesced window. No-op off the concurrent path.
            if (*this).durable
                && !faulted
                && fiber_rt::window_is_unwinding(mem_base as u64, (*this).shadow)
            {
                (*this).frozen_vcpus_out.extend(d.take_frozen_vcpus());
                // §12.8 4A.5 follow-up A: carry completed concurrent children's join results, so a
                // `thread.join` of a child that finished before the freeze point resolves on thaw.
                (*this)
                    .frozen_vcpus_out
                    .extend(d.take_completed_children_residue());
                // §12.8 4A.5 follow-up B: a concurrent child that owns fibers flattened them in its own
                // `run_child` `freeze_drive`, recorded during `join_all` — drain after the join.
                (*this).frozen_out.extend(d.take_frozen_fibers());
            }
        }
        // #1684 — every fiber slot nobody flattened rides too (a fresh one, a free one), so the thaw
        // rebuilds the table slot for slot. After `join_all`: no vCPU can still create a fiber.
        #[cfg(fiber_rt)]
        if (*this).durable
            && !faulted
            && fiber_rt::window_is_unwinding(mem_base as u64, (*this).shadow)
        {
            if let Some(rt) = (*this).fiber_rt.as_ref() {
                let flattened: Vec<usize> = (*this).frozen_out.iter().map(|f| f.slot).collect();
                let rest = rt.table().unflattened_for_freeze(&flattened);
                (*this).frozen_out.extend(rest);
            }
        }
        // §5 W3 Stage 3 — a trap that originated on a *spawned* vCPU stashed its backtrace capture in
        // that worker's thread-local, which dies with the worker; the worker handed it to the domain
        // before finishing, so collect it now (all vCPUs are joined). Used only when the root vCPU
        // itself didn't trap (`root_trap_cap` below takes precedence — the entry's own trap is the
        // primary one in the common single-vCPU case).
        #[cfg(fiber_rt)]
        let worker_trap_cap = (*this).domain.as_ref().and_then(|d| d.take_trap_capture());
        #[cfg(not(fiber_rt))]
        let worker_trap_cap: Option<(usize, Vec<usize>, i64)> = None;
        // Snapshot the in-window bytes when asked (the escape-oracle; a reactor's state). #2176: a run
        // that asks for none copies nothing — a copy of a 32 MiB window is a page fault per page, for
        // bytes no one reads.
        (*this).tail_out = Vec::new();
        let final_mem = match snapshot_cap {
            None => Some(Vec::new()),
            Some(cap) => {
                // The guest may have made pages non-readable via the Memory cap (unmap/protect), so
                // restore RW first — else this read faults outside the guarded call and crashes the
                // host. #2286 — one the OS refuses fails the snapshot instead: nothing is read, and
                // the run reports it once its own state is settled (below).
                // `cap` widens the snapshot past the backed prefix to also cover reserved-tail pages
                // the guest grew/`unmap`-ed (§1a growth path), `commit`-ing them so the read sees
                // zero/their content instead of faulting. `read_low` clamps to the reservation.
                // #1810, #2427: a durable run's capture is its declared window and, read apart while the
                // window lives, each page the guest holds past it: never the span between them.
                let snap = match (*this).held_tail {
                    Some(_) => win_size,
                    None if win_size > 0 => cap.min((mask + 1) as usize).max(win_size),
                    None => win_size,
                };
                let image = if !window.restore_rw() {
                    None
                } else if snap > win_size {
                    window.read_low(snap)
                } else {
                    Some(window.rw_mut()[..win_size].to_vec())
                };
                match (image, (*this).held_tail) {
                    (Some(image), Some((f, ctx))) => {
                        let held = f(ctx, mem_base as usize, win_size as u64);
                        window.read_pages(&held).map(|tail| {
                            (*this).tail_out = tail;
                            image
                        })
                    }
                    (image, _) => image,
                }
            }
        };
        // The window dies with this run; the code, function table, and runtimes stay alive in
        // `*this` for the next `run` / `define_extra` / drop.
        drop(window);

        // What ended the run: the first end raised on its domain, which nothing after it can change
        // (#2499). Post-`join_all`, so the load sees every raise (the join is a synchronization
        // point); Relaxed suffices.
        let cell = vm.ended().unwrap_or(0);
        let code = cell as u32;

        // Publish the trap-time backtrace + fiber for `last_trap_backtrace`/`last_trap_fiber`: the root
        // vCPU's own capture if it trapped, else a spawned vCPU's (Stage 3). The fiber handle (§23-D57)
        // rides along, captured at the trap instant. Symbolizing is pure (reads the address map), so it
        // is fine here after teardown. Reset on a clean run (every successful run resets both), and on
        // one that did not end in a trap: a sibling's capture from after its domain ended is no
        // outcome's.
        let trapped = TrapKind::from_code(code).is_some();
        match root_trap_cap.or(worker_trap_cap).filter(|_| trapped) {
            Some((pc, rets, fiber)) => {
                (*this).last_trap_backtrace = (*this).trap_backtrace(pc, &rets);
                (*this).last_trap_fiber = Some(fiber);
            }
            None => {
                (*this).last_trap_backtrace = Vec::new();
                (*this).last_trap_fiber = None;
            }
        };

        let outcome = if code == 0 || code == DOMAIN_DONE_CODE {
            // `DOMAIN_DONE_CODE` is the internal root-completed teardown sentinel (owner decision
            // 2026-07-24): the root returned cleanly and the sentinel only unwound leftover sibling
            // vCPUs, whose outcomes are unobservable — the run outcome is the root's own results.
            JitOutcome::Returned(results)
        } else if code == EXIT_CODE {
            JitOutcome::Exited((cell >> 32) as i32)
        } else if code == HOST_UNWIND_CODE {
            JitOutcome::HostUnwound
        } else {
            JitOutcome::Trapped(TrapKind::from_code(code).ok_or(JitError::Malformed)?)
        };
        let final_mem = final_mem.ok_or(JitError::Unsupported(
            "the OS refused to open the window for its snapshot",
        ))?;
        Ok((outcome, final_mem))
    }

    /// Declare + define + finalize **additional functions** into the live module (DESIGN.md §22:
    /// the enabling primitive for the guest-driven `Jit` capability). The slice is a
    /// self-contained unit: its `FuncIdx` space is unit-local, so direct calls resolve within the
    /// unit only (cross-unit calls go through `call.dyn` against the parent table, or the
    /// guest re-emits the callee — DESIGN.md §22 "Recommendation"). Returns one buffer-ABI trampoline
    /// pointer per function, in order; invoke via [`Self::run_extra`] (or, in the capability
    /// layer, directly over a live run's window).
    ///
    /// **The function table is deliberately untouched**: extra functions are thunk-reachable
    /// only, so the table mask baked into every existing `call.dyn` site never changes
    /// (DESIGN.md §22 — zero new escape-relevant dispatch surface). Extra code is lowered
    /// against the *parent's* environment: same confinement mask, same `call.cap` thunk, same
    /// table mask, and the module's shared **append-only type-id registry** (see
    /// [`Self::interned_type_id`]) — the unit's signatures are interned before lowering, so
    /// id-equality coincides with structural equality across every unit this module has
    /// compiled or will compile. A `call.dyn` whose signature no table entry carries
    /// still traps, fail-closed — but a signature first introduced here keeps a stable id, so
    /// a future table install of a structurally equal function can satisfy it.
    ///
    /// Functions using §12 fibers/threads are rejected (`Unsupported`) — the MVP restricts
    /// incremental definition to single-threaded code (DESIGN.md §22 "Concurrency"), and lowering
    /// `cont.*`/`thread.*` here would need per-unit runtime wiring this slice doesn't do.
    pub fn define_extra(
        &mut self,
        funcs: &[Func],
        types: &[temen_ir::TypeEntry],
    ) -> Result<Vec<DefinedFn>, JitError> {
        if funcs.is_empty() {
            return Ok(Vec::new());
        }
        for f in funcs {
            ensure_supported(f, types)?;
            // Threads/futex ARE hosted (CONSOLIDATION.md §11) — but only when the domain stood up a
            // thread scheduler (`enable_thread_hosting`, driven by a thread-hosting `Jit` grant, or a
            // parent that itself uses threads). Without it the instance has no domain for the
            // unit's sites to run on, so a thread/futex-using unit fails closed — exactly the fiber
            // gate below.
            if (f.uses_threads() || f.uses_futex()) && !self.threads {
                return Err(JitError::Unsupported(
                    "a submitted unit using threads/futex requires a thread-hosting Jit domain",
                ));
            }
            // Fibers ARE hosted — but only when the domain stood up a fiber runtime
            // (`enable_fiber_hosting`, driven by a fiber-hosting `Jit` grant). Without it there is
            // no runtime for the unit's `cont.*` sites, so a fiber-using unit must fail closed.
            if f.uses_fibers() && !self.fibers {
                return Err(JitError::Unsupported(
                    "a submitted unit using fibers requires a fiber-hosting Jit domain",
                ));
            }
            // `setjmp`/`longjmp` keep their buffers in the instance's `setjmp` table, which only a
            // module that uses them stands up: without it the unit's sites would run over none.
            if f.uses_setjmp() && self.instance.setjmp.is_null() {
                return Err(JitError::Unsupported(
                    "a submitted unit using setjmp/longjmp requires a module that uses them",
                ));
            }
        }
        // Shared code cannot take the unit (#1825): refused before anything is recorded.
        let module = arena_mut(&mut self.module)?;
        // Intern the unit's signatures (its functions' own + its call sites') into the
        // append-only registry BEFORE lowering, so the ids baked into this unit's dispatch
        // checks are real, stable ids — id-equality ≡ structural equality across all units
        // sharing this module, past and future (DESIGN.md §22; see `intern_type`).
        intern_unit_sigs(&mut self.distinct, funcs, types)?;
        // Declare the unit's functions first so intra-unit direct calls can reference any of them.
        let ids: Vec<FuncId> = funcs
            .iter()
            .map(|f| {
                let name = format!("x{}", self.next_extra);
                self.next_extra += 1;
                let sig = natural_sig(&mut *module, f);
                module
                    .declare_function(&name, Linkage::Local, &sig)
                    .map_err(|e| JitError::Backend(e.to_string()))
            })
            .collect::<Result<_, _>>()?;

        // Auto-install for unit-own funcrefs (DESIGN.md §22 "unit-own funcref"): if the unit names
        // its OWN functions by index — `ref.func`, **or** `thread.spawn` (whose entry `func` is a
        // module-relative index resolved through the shared `fn_table`, CONSOLIDATION.md §11) — each
        // such reference must land on a real shared-table slot or it would resolve against the
        // *parent's* table (a confused-deputy spawn of the parent's function N instead of the unit's).
        // Reserve one padding slot per unit function up front, so `ref.func N` / `thread.spawn N`
        // lower to `ref_slots[N]` (below); the finalized code is published into these slots after the
        // build. `None` when the unit names no own function — then no slots are consumed. Single
        // `call.cap`, so the picked padding slots stay free until we `install_at` them (no concurrent
        // install races within a compile). The test is the one both interpreters install on
        // (`Func::names_own_funcs`), so every engine fills the same slots.
        let ref_slots: Option<Vec<u32>> = if funcs.iter().any(Func::names_own_funcs) {
            let free: Vec<u32> = self
                .fn_table
                .iter()
                .enumerate()
                .filter(|(_, e)| e.type_id() == PADDING_TYPE_ID)
                .map(|(i, _)| i as u32)
                .take(funcs.len())
                .collect();
            if free.len() < funcs.len() {
                return Err(JitError::Unsupported(
                    "not enough reserved call.dyn slots for a submitted unit's own funcrefs \
                     (grant a larger Jit table)",
                ));
            }
            Some(free)
        } else {
            None
        };

        let nesting = self.nesting;
        // Byte-accurate occupancy: each function's emitted code size. A unit's code is not kept.
        let extra_bytes = define_funcs(
            &mut *module,
            &ids,
            &mut Codegen::Compile(None),
            |module, ctx, fi| {
                build_clif(
                    module,
                    &ids,
                    funcs,
                    &self.distinct,
                    types,
                    self.fast_resolver,
                    nesting,
                    false, // #932 — define_extra units are not signal-delivery armed (slice 1b)
                    self.page, // #2164 — a paged child's units check its access table, as its own code does
                    &mut ctx.func,
                    &funcs[fi],
                    self.mask,
                    self.win_mapped as u64,
                    guard_offset_of(self.win_reserved as u64),
                    self.epoch,
                    self.fuel, // charges the running instance's fuel cell, as the parent's functions do
                    self.threads, // as the parent module's functions
                    self.fn_table_mask, // the parent's table mask, NOT derived from this unit's size
                    ref_slots.as_deref(), // remap `ref.func N` -> the unit's auto-installed slot
                    0,
                    None, // extra/installed units carry no source-loc map (W5 JIT/DWARF)
                    None, // …nor value-label points (Stage 3a)
                )
            },
            |_, _, _| {},
        )?;
        self.extra_bytes += extra_bytes;
        let mut ctx = module.make_context();
        // One buffer-ABI trampoline per function, so the host can invoke any of them (any arity).
        let tramp_ids: Vec<FuncId> = funcs
            .iter()
            .zip(&ids)
            .map(|(f, id)| {
                build_trampoline(&mut *module, &mut ctx.func, *id, f, false)?;
                let name = format!("xt{}", self.next_extra);
                self.next_extra += 1;
                let t = module
                    .declare_function(&name, Linkage::Export, &ctx.func.signature)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                self.extra_bytes += define_verified(module, t, &mut ctx)?.code_buffer().len();
                module.clear_context(&mut ctx);
                Ok(t)
            })
            .collect::<Result<_, JitError>>()?;
        // Incremental finalize: mprotects only the newly defined code pages; already-finalized,
        // possibly-running code is untouched (the DESIGN.md §22 Phase-1 W^X spike is the test asserting
        // exactly this).
        module
            .finalize_definitions()
            .map_err(|e| JitError::Backend(e.to_string()))?;
        // Each function's finalized natural-ABI entry and buffer-ABI trampoline.
        let finalized: Vec<(*const u8, *const u8)> = ids
            .iter()
            .zip(&tramp_ids)
            .map(|(id, t)| {
                (
                    module.get_finalized_function(*id),
                    module.get_finalized_function(*t),
                )
            })
            .collect();
        // Publish the unit's now-finalized functions into the slots reserved above, so a `ref.func`
        // funcref (remapped to `ref_slots[N]`) resolves to the unit's own function through the
        // ordinary masked dispatch (DESIGN.md §22 "unit-own funcref"). Slots were picked from the
        // padding pool and are still free (single `call.cap`), so each `install_at` succeeds.
        if let Some(slots) = &ref_slots {
            for ((f, &(code, _)), &slot) in funcs.iter().zip(&finalized).zip(slots) {
                let type_id = type_id_of(
                    &self.distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                );
                if !self.install_at(slot, code, type_id) {
                    return Err(JitError::Backend(format!(
                        "auto-install of a unit funcref into slot {slot} failed"
                    )));
                }
            }
        }
        // Per function: the buffer-ABI trampoline (for `invoke` over a window) **and** the
        // natural-ABI entry + interned `type_id` (for B2 `install` into the function table —
        // `call.dyn` calls the natural ABI, not the trampoline).
        Ok(funcs
            .iter()
            .zip(finalized)
            .map(|(f, (code, tramp))| DefinedFn {
                tramp,
                code,
                type_id: type_id_of(
                    &self.distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
            })
            .collect())
    }

    /// Stand up the §12 **fiber runtime** on an already-compiled module so guest-submitted units
    /// (`define_extra` / `invoke_extra`) may host `cont.*` (DESIGN.md §22 "Concurrency",
    /// renegotiated 2026-07-30). A `Jit`-granting parent typically does not itself use fibers, so
    /// its `compile` stood up no fiber runtime — and `define_extra` rejects a fiber-using
    /// submitted unit (its `cont.*` sites would have none to call into). This installs the
    /// **same** runtime `compile` builds when the top-level module uses fibers — the domain-shared
    /// table, the root vCPU's [`fiber_rt::FiberRuntime`], and the call
    /// trampoline — reusing the incremental define+finalize path (`define_extra`'s, W^X-safe: only
    /// the fresh trampoline page is mprotected; running code is untouched).
    ///
    /// Called by the `Jit` run entries between `compile` and `run` for a **fiber-hosting** grant.
    /// The subsequent `run` publishes `CURRENT_RT` from `self.fiber_rt` (set here), so a submitted
    /// unit's `cont.*`, running on the caller's stack inside the synchronous `call.cap`, resolve it.
    /// A submitted unit whose own scheduler creates fibers over unit-local entries needs those
    /// entries reachable as funcrefs (§22 install / the shared `fn_table`) — same as any `cont.new`.
    ///
    /// **Idempotent** and behavior-preserving: a no-op if fibers are already hosted (the parent used
    /// `cont.*`/`gc.roots`), reusing an existing table/trampoline a thread-only parent already built.
    #[cfg(fiber_rt)]
    pub fn enable_fiber_hosting(&mut self) -> Result<(), JitError> {
        if self.fibers {
            return Ok(()); // already hosting fibers (parent used cont.*/gc.roots at compile time)
        }
        // Append the fiber-entry signature to the per-domain type registry (idempotent; an id never
        // remaps — DESIGN.md §22). A later `define_extra` interning the same signature gets this id,
        // so id-equality stays ≡ structural equality. Consulted only here, guest not yet running.
        let fiber_type_id = intern_type(&mut self.distinct, &fiber_func_type())?;
        // `cont.new` masks a funcref into the shared table; the table mask is safe (indices stay in
        // `[0, table_len)`; padding slots trap on the type check) and equals `compile`'s funcs-based
        // mask for a domain with no reserved install table (the fiber-hosting consumer).
        let fiber_mask = self.fn_table_mask;
        // Domain-shared handle namespace — reuse the one a thread-only parent built.
        let table = match &self.fiber_table {
            Some(t) => std::sync::Arc::clone(t),
            None => {
                let t = std::sync::Arc::new(fiber_rt::SharedFiberTable::new(
                    temen_ir::MAX_FIBERS,
                    self.shadow,
                    self.node.clone(),
                ));
                self.fiber_table = Some(std::sync::Arc::clone(&t));
                t
            }
        };
        let mut rt = Box::new(fiber_rt::FiberRuntime::new(
            table,
            fiber_type_id,
            fiber_mask,
        ));
        // The generic call-trampoline (calls any Tail-ABI `(sp, arg) -> i64` entry from Rust). A
        // thread-only parent already built it (`compile`, `uses_threads` arm); else build it now,
        // incrementally like a submitted unit — a fresh function, a second `finalize_definitions`.
        let tramp = match self.call_tramp {
            Some(t) => t,
            None => {
                let module = arena_mut(&mut self.module)?;
                let mut ctx = module.make_context();
                build_fiber_call_trampoline(&mut *module, &mut ctx.func);
                let id = module
                    .declare_function("fiber_call_tramp", Linkage::Export, &ctx.func.signature)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                define_verified(module, id, &mut ctx)?;
                module.clear_context(&mut ctx);
                module
                    .finalize_definitions()
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                let addr = module.get_finalized_function(id);
                // SAFETY: `addr` is the finalized `fiber_call_tramp` with exactly the ABI its
                // builder emitted (`code` + the guest entry's Tail args) — the same `transmute`
                // `compile` does for its own trampoline.
                unsafe { std::mem::transmute::<*const u8, fiber_rt::FiberCallTramp>(addr) }
            }
        };
        rt.set_call_tramp(tramp);
        self.fiber_rt = Some(rt);
        self.hosted = true;
        self.call_tramp = Some(tramp);
        self.fiber_cfg = Some((fiber_type_id, fiber_mask));
        self.fibers = true;
        Ok(())
    }

    /// Fiber hosting on a target without stack-switch support: a no-op — fibers stay unsupported, so
    /// a fiber-using submitted unit remains rejected by `define_extra`'s fiber gate.
    #[cfg(not(fiber_rt))]
    pub fn enable_fiber_hosting(&mut self) -> Result<(), JitError> {
        Ok(())
    }

    /// Stand up the §12 **thread scheduler** on an already-compiled module so guest-submitted units
    /// (`define_extra` / `install`) may host `thread.spawn`/`thread.join` and the `memory.wait`/
    /// `notify` futex (CONSOLIDATION.md §11 — installed units join the caller's concurrency model).
    /// The twin of [`Self::enable_fiber_hosting`]: a `Jit`-granting parent that itself uses no threads
    /// stood up no thread domain at `compile`, so `define_extra` rejects a thread-using submitted
    /// unit (its `thread.*` sites would have none to run on). This installs the
    /// **same** runtime `compile` builds for a threaded module: the 1:1 OS-thread executor `Domain`
    /// (which the unit's `thread.*` sites load from the vmctx), the domain-shared fiber table the
    /// spawned vCPUs allocate their durable contexts over (D57 3b-ii — created even for a fiber-free
    /// threaded module), and the generic call-trampoline vCPU entries run through.
    ///
    /// Called by the `Jit` run entries between `compile` and `run` for a **thread-hosting** grant. The
    /// subsequent `run` seeds the `Domain`'s per-run `Env` (window / fn-table / trap cell / trampoline)
    /// via [`os_thread_rt::Domain::set_env`], exactly as a natively-threaded run does. A submitted
    /// unit's `thread.spawn N` dispatches its entry through the shared `fn_table`; the unit's own
    /// functions are auto-installed there (`define_extra`'s `ref_slots`), so the spawn resolves to the
    /// unit's function, not the parent's slot `N`.
    ///
    /// **Idempotent** and behavior-preserving: a no-op if threads are already hosted (the parent used
    /// `thread.*`, or a nesting parent stood up the `Domain` for its children's futex), reusing the
    /// existing `Domain`/table/trampoline. `fiber_cfg` is left untouched — `None` for a pure-thread
    /// host (matching a normally-compiled fiber-free threaded module), already `Some` if fibers are
    /// also hosted, so the two `enable_*` calls compose in either order.
    #[cfg(fiber_rt)]
    pub fn enable_thread_hosting(&mut self) -> Result<(), JitError> {
        if self.threads {
            return Ok(()); // already hosting threads (parent used thread.*, or a nesting Domain exists)
        }
        // The domain-shared fiber table (D57 3b-ii): every spawned vCPU builds its `FiberRuntime` /
        // durable vCPU-context over it. `compile` creates it for *any* threaded module (even a
        // fiber-free one — the durable-context allocator needs it), so create it here too. Reuse an
        // existing table if a fiber-hosting parent already built one (unified handle namespace). The
        // `Domain` reads it via `set_env`'s `fiber_table.clone()` at run time, so populating the field
        // is all that's needed here.
        if self.fiber_table.is_none() {
            self.fiber_table = Some(std::sync::Arc::new(fiber_rt::SharedFiberTable::new(
                temen_ir::MAX_FIBERS,
                self.shadow,
                self.node.clone(),
            )));
        }
        // The generic call-trampoline (calls any Tail-ABI `(sp, arg) -> i64` entry from Rust). Spawned
        // vCPUs call their entry through it; a fiber-hosting parent already built it, else build it now
        // incrementally like a submitted unit — a fresh function, a second `finalize_definitions`
        // (W^X-safe: only the fresh trampoline page is mprotected; running code is untouched).
        let tramp = match self.call_tramp {
            Some(t) => t,
            None => {
                let module = arena_mut(&mut self.module)?;
                let mut ctx = module.make_context();
                build_fiber_call_trampoline(&mut *module, &mut ctx.func);
                let id = module
                    .declare_function("fiber_call_tramp", Linkage::Export, &ctx.func.signature)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                define_verified(module, id, &mut ctx)?;
                module.clear_context(&mut ctx);
                module
                    .finalize_definitions()
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                let addr = module.get_finalized_function(id);
                // SAFETY: `addr` is the finalized `fiber_call_tramp` with exactly the ABI its builder
                // emitted — the same `transmute` `compile`/`enable_fiber_hosting` do for their own.
                unsafe { std::mem::transmute::<*const u8, fiber_rt::FiberCallTramp>(addr) }
            }
        };
        // Stand up the executor `Domain` the unit's `thread.*` sites load from this instance's context.
        // Its per-run `Env` (window / fn-table / trap cell / this trampoline) is supplied later by the
        // run entry's `set_env`; the address is stable now, which is all the context needs.
        let domain = Box::new(os_thread_rt::Domain::new(self.node.clone()));
        self.instance.sched = &*domain as *const os_thread_rt::Domain as *const core::ffi::c_void;
        self.threads = true;
        self.spawns = true;
        self.domain = Some(domain);
        self.hosted = true;
        self.call_tramp = Some(tramp);
        // `fiber_table` set above; `fiber_cfg` left as-is (None ⇒ pure-thread host, matching a
        // fiber-free threaded module — spawned vCPUs then build no per-vCPU fiber runtime).
        Ok(())
    }

    /// Thread hosting on a target without the fiber/thread runtime: a no-op — threads stay unsupported,
    /// so a thread-using submitted unit remains rejected by `define_extra`'s thread gate.
    #[cfg(not(fiber_rt))]
    pub fn enable_thread_hosting(&mut self) -> Result<(), JitError> {
        Ok(())
    }

    /// **Install** an incrementally-defined function into the live `call.dyn` table (DESIGN.md §22
    /// Model B2): write its natural-ABI `code` + interned `type_id` into the next reserved
    /// padding slot, returning that slot index — a funcref the guest (or another unit) can
    /// `call.dyn` at native speed (old→new / new→new). `None` if the table is full (every
    /// reserved slot taken). The base never moves (the table was pre-reserved at compile, the
    /// mask is a baked constant), so this is just a slot write. The write is **release-ordered and
    /// atomic** ([`FnEntry::publish`]), so it is sound against a concurrent `call.dyn` from
    /// another thread — the JIT §6 #2 threaded-install path — not only the single-threaded MVP.
    /// Takes `&self`: the running generated code reads the table through raw pointers, so the host
    /// install must not claim a Rust exclusive borrow over it.
    pub fn install(&self, code: *const u8, type_id: u32) -> Option<u32> {
        let slot = self
            .fn_table
            .iter()
            .position(|e| e.type_id() == PADDING_TYPE_ID)?;
        self.fn_table[slot].publish(type_id, code as u64);
        Some(slot as u32)
    }

    /// **Uninstall** a previously-`install`ed `call.dyn` slot (DESIGN.md §22 reclaim): set
    /// it back to a trapping padding slot so the index is reusable by a later `install` and a
    /// stale `call.dyn` of it fails closed (`IndirectCallType`). Returns `true` on success;
    /// `false` for an out-of-range slot, a real module-function slot (`< n_real_funcs`), or an
    /// already-empty slot — a guest may only reclaim what it installed. (The code memory itself
    /// is not freed — cranelift-jit has no per-function free; this reclaims the *slot*.)
    pub fn uninstall(&self, slot: u32) -> bool {
        let i = slot as usize;
        if i < self.n_real_funcs || i >= self.fn_table.len() {
            return false;
        }
        if self.fn_table[i].type_id() == PADDING_TYPE_ID {
            return false; // already empty
        }
        self.fn_table[i].clear();
        true
    }

    /// **Install at a specific slot** — the compaction counterpart of [`Self::install`] (DESIGN.md §22
    /// code-memory compaction). `install` fills the *next* padding slot, which reproduces a dense
    /// install history but not one with `uninstall` gaps; recompaction must reproduce **exact** slot
    /// indices so a funcref a guest already holds keeps resolving to the same unit across the swap.
    /// Writes the unit's natural-ABI `code` + interned `type_id` into `slot`, returning `true` on
    /// success. Refuses (`false`) an out-of-range slot, a real module-function slot
    /// (`< n_real_funcs`, guarding the original program's funcrefs), or an already-occupied slot
    /// (the target must be padding — a fresh module's reserved slots all are). Same trust class as
    /// `install`: a host-driven slot write into a pre-reserved table whose base never moves.
    pub fn install_at(&self, slot: u32, code: *const u8, type_id: u32) -> bool {
        let i = slot as usize;
        if i < self.n_real_funcs || i >= self.fn_table.len() {
            return false;
        }
        if self.fn_table[i].type_id() != PADDING_TYPE_ID {
            return false; // occupied — install_at never overwrites a live slot
        }
        self.fn_table[i].publish(type_id, code as u64);
        true
    }

    /// The currently-occupied **installable** slots (`≥ n_real_funcs`) of the `call.dyn`
    /// table, as `(slot, code, type_id)`. The reclaim driver (DESIGN.md §22) reads this from the
    /// *old* module to learn which units occupy which slots, joins each `code` back to its owning
    /// unit (via the embedder's per-unit install record), and reproduces the exact slot in the
    /// fresh module with [`Self::install_at`]. Real module-function slots (`< n_real_funcs`) are
    /// excluded — they are reproduced by `compile` itself, not by the driver.
    pub fn installed_slots(&self) -> Vec<(u32, u64, u32)> {
        self.fn_table
            .iter()
            .enumerate()
            .skip(self.n_real_funcs)
            .filter(|(_, e)| e.type_id() != PADDING_TYPE_ID)
            .map(|(i, e)| (i as u32, e.code(), e.type_id()))
            .collect()
    }

    /// The number of extra (`define_extra`) functions+trampolines this module has lowered over its
    /// life — a monotonic proxy for code-arena occupancy (cranelift-jit exposes no byte count, and
    /// has no per-function free, so this only grows). An embedder watches it to decide when to
    /// **compact** (DESIGN.md §22): rebuild the live unit set into a fresh module — whose count restarts
    /// near zero — and drop this one, reclaiming the arena. See `tests/jit_compaction.rs`.
    pub fn extra_fn_count(&self) -> usize {
        self.next_extra
    }

    /// **Byte-accurate** code-arena occupancy: the cumulative machine-code bytes of every
    /// `define_extra`'d function + trampoline this module has lowered (the actual emitted size,
    /// summed at finalize — the dominant term in arena consumption; alignment/rodata padding is
    /// excluded, so this slightly *under*counts the true arena bytes). Monotonic; restarts near zero
    /// in a freshly-compacted module. Prefer this over [`Self::extra_fn_count`] for a watermark when
    /// units vary widely in size (a few large functions vs many tiny ones). See
    /// [`crate::CompiledModule`] / `tests/jit_compaction.rs`.
    pub fn extra_byte_count(&self) -> usize {
        self.extra_bytes
    }

    /// Finalized machine-code bytes of the **base module** — every function body emitted by the
    /// initial [`CompiledModule::compile`], summed from Cranelift's `code_buffer` at define time.
    /// The byte-accurate "how big is the JIT'd code" measure (excludes the tiny buffer-ABI
    /// trampoline and later `define_extra` units — those are [`Self::extra_byte_count`]).
    pub fn code_byte_count(&self) -> usize {
        self.base_bytes
    }

    /// Whether a run is in flight on this module (a guarded call published its window fault range).
    /// Compaction (and any rebuild that drops `self`) is only sound at a **quiescent** point — DESIGN.md §22
    /// §6: "it can only run at a quiescent point — the guest is suspended *inside* the very module
    /// being compacted." An embedder asserts `!is_running()` before swapping a freshly-compacted
    /// module in for this one.
    pub fn is_running(&self) -> bool {
        self.live_fault_range.is_some()
    }

    /// #2113 — the fuel cell this module's code charges, as [`Self::compile`] was handed it: what a
    /// recompaction compiles the fresh module against, so the fresh code meters as the old did.
    pub fn fuel_cell(&self) -> Option<*mut FuelCell> {
        (!self.instance.fuel.is_null()).then_some(self.instance.fuel)
    }

    /// The stable type id `ty` was interned under, or `None` if no unit this module compiled
    /// has mentioned it (as a function signature or a call-site type). Ids are append-only —
    /// once returned, an id never remaps — and id-equality coincides with structural equality
    /// over everything this module compiled (see `intern_type`). This is the lookup a table
    /// `install` operation uses to stamp a slot's `type_id` (DESIGN.md §22).
    pub fn interned_type_id(&self, ty: &FuncType) -> Option<u32> {
        self.distinct.iter().position(|t| t == ty).map(|i| i as u32)
    }
}

/// #2166 — invoke a serve handler's trampoline over its domain's **live** window: the one handler
/// call of every JIT serve route — the domain's own serve loop (root or §14 child) and a caller's
/// direct handoff (CALLS.md 5c.2). Arms detect-and-kill over the window's whole fault range
/// (`[base, base + round_up(reserved) + guard)`, the range its own run arms), so a handler's fault is
/// the domain's trap: written to `vmctx`'s trap cell, exactly as a guest trap. Counts the handler in
/// the running vCPU's [`serve_handlers`] for its span, which is what refuses a nested serve (#2160).
///
/// # Safety
/// `cc` is the live [`CompiledModule`] the serving domain runs (registered as its serve context);
/// `code` is one of its finalized serve trampolines; `args`/`results` match the trampoline's arity;
/// `mem_base` is the domain's live window with a `mem_reserved` reservation (what its `call.cap`
/// thunk is handed); `vmctx` is the **domain's** instance context — its own run's (the serve loop's
/// thunk `vmctx`), or one minted from [`instance_of`] for a handler run on another thread.
#[cfg(fiber_rt)]
pub unsafe fn invoke_handler(
    cc: *const core::ffi::c_void,
    code: *const u8,
    args: &[i64],
    results: &mut [i64],
    mem_base: *mut u8,
    mem_reserved: u64,
    vmctx: *const VmCtx,
) {
    let cc = cc as *const CompiledModule;
    let fn_table_ptr = (*cc).fn_table.as_ptr() as *const core::ffi::c_void;
    let (lo, hi) = mem::fault_range_of(mem_base, mem_reserved as usize);
    vcpu_tls::set_serve_handlers(serve_handlers() + 1);
    let faulted = mem::run_guarded_range(
        code,
        args.as_ptr(),
        results.as_mut_ptr(),
        mem_base,
        fn_table_ptr,
        vmctx,
        lo,
        hi,
    );
    vcpu_tls::set_serve_handlers(serve_handlers() - 1);
    if faulted {
        (*vmctx).raise(mem::FAULT_TRAP);
    }
}

/// #1768 — the instance a serving domain's module runs as (its powerbox, and the kill-path and fuel
/// cells its compile was handed): what a caller running one of its handlers **on another thread**
/// (the CALLS.md 5c.2 handoff) mints the handler's [`VmCtx`] from, so the handler's `call.cap`s reach
/// the server's powerbox, never the caller's.
///
/// # Safety
/// `cc` is a live registered serve context (see [`invoke_handler`]).
#[cfg(fiber_rt)]
pub unsafe fn instance_of(cc: *const core::ffi::c_void) -> InstanceAddrs {
    (*(cc as *const CompiledModule)).instance
}

/// CALLS.md 5c.1a — resolve handler `func`'s serve trampoline on a raw serve-context
/// [`CompiledModule`] (the `Host::serve_ctx` registration, opaque outside this crate).
///
/// # Safety
/// `cc` is a live registered serve context (see [`invoke_handler`]).
#[cfg(fiber_rt)]
pub unsafe fn handler_tramp_of(
    cc: *const core::ffi::c_void,
    func: u32,
) -> Option<(*const u8, usize, usize)> {
    (*(cc as *const CompiledModule)).handler_tramp(func)
}

// PROCESS.md S1c — a compiled §14 child is shareable across OS threads. `CompiledModule` is
// `!Send`/`!Sync` only because of its raw code pointers; once `finalize_definitions` has run (before
// a child is ever stored) the code arena and `fn_table` are read-execute memory the entry trampoline
// reads and jumps into, never writes. So handing an `Arc<CompiledModule>` to a spawned child thread
// and running the same **plain** child concurrently on N threads is sound: every thread only reads the same finalized bytes, and the single `OwnedJit` frees the arena
// once when the last `Arc` drops (after `join_children`, so no thread still runs the code). The one
// mutable path — `define_extra`/`install` behind a re-granted `Jit` capability (#1296) — is reached
// only through a child holding that `Jit`, whose code is compiled for it alone (never cached,
// temen#2168) and owned by exactly one thread: its module is mutated by that thread alone, inside the
// child's own suspended `call.cap` (the `run_raw` re-entry discipline), never shared. A root module is
// never sent anywhere.
#[cfg(fiber_rt)]
unsafe impl Send for CompiledModule {}
#[cfg(fiber_rt)]
unsafe impl Sync for CompiledModule {}

/// #1825 — compiled code more than one instance runs ([`CompiledModule::share`]): a fork twin its
/// parent's program, every `execve` of a command the one compile of it. Each [`Self::instance`] is a
/// [`CompiledModule`] of its own — its own powerbox, function table and run state — over the same
/// finalized code, which none of them can extend.
pub struct SharedCode(CompiledModule);

// SAFETY: the wrapped module is never run and never mutated — `instance` only reads it. Its code
// pointers name finalized, read-execute memory its `Arc<OwnedJit>` keeps alive, and every instance holds
// a clone of that `Arc`, so the arena outlives the last run of it. The code names no per-instance
// runtime: the runtimes the kept module holds are its own, never run, and `instance` reads only which
// of them it has, to stand up its own (temen#2389). So any thread may hold it, and any number may make
// instances of it at once — the argument above for a §14 child's code.
unsafe impl Send for SharedCode {}
unsafe impl Sync for SharedCode {}

impl SharedCode {
    /// A new instance of the code at `addrs`: `cap_ctx` is its own powerbox, and `epoch`, `fuel` and
    /// `sig_armed` the cells its checks poll — its run's, which need not be the cells of the run that
    /// compiled it (#2145). A cell is set exactly when the code was compiled with its check: code
    /// without the check never reads it, and a check of a null cell would fault in the host.
    ///
    /// The instance stands up the runtimes its code runs on (temen#2389), and `node` is the budget
    /// its threads and fibers are charged to: its run's. A §14 child's are its task's instead, and
    /// its nursery is its spawn's, set in `addrs` exactly when the code spawns (temen#2168); a root
    /// makes its own.
    pub fn instance(
        &self,
        addrs: InstanceAddrs,
        node: Option<std::sync::Arc<dyn BudgetNode>>,
    ) -> CompiledModule {
        assert_eq!(addrs.epoch.is_null(), !self.0.epoch, "kill-path cell");
        assert_eq!(addrs.fuel.is_null(), !self.0.fuel, "fuel cell");
        assert_eq!(
            addrs.sig_armed.is_null(),
            self.0.instance.sig_armed.is_null(),
            "signal cell"
        );
        assert_eq!(
            addrs.nursery.is_null(),
            !self.0.nests || self.0.makes_nursery(),
            "nursery"
        );
        // A paged compile tests the access table it loads before each access (#2164).
        assert_eq!(
            addrs.pages.is_null(),
            !self.0.page.is_paged(),
            "access table"
        );
        self.0.instance(addrs, node)
    }
}

// Compile-time proof that the child artifact stays thread-shareable — if a future field reintroduced a
// `!Send`/`!Sync` type (e.g. an `Rc` or `Cell`) without a matching soundness review, this assertion
// would fail to compile, catching the regression at the type level (the S1c executor relies on it).
#[cfg(fiber_rt)]
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<CompiledModule>();
};

/// A host function compiled code calls. The code names it by a symbol, through a relocation, never
/// by its address: [`new_jit_module`] binds each symbol to the function's address in this process
/// when the code is linked, so the same code links into a process whose functions sit elsewhere
/// (#2452). A call costs what a call to a baked address cost: Cranelift lowers a call to an
/// imported symbol to the same `movabs` and indirect `call` on x86-64, a literal load and `blr` on
/// AArch64. The symbols are not C identifiers, so no lookup in the process's own symbol table can
/// bind one: only [`new_jit_module`] does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HostFn {
    /// The `call.cap` thunk ([`CapThunk`]): the embedder's, or a §14 child's coroutine thunk.
    CapThunk,
    /// #932 — the delivery-decision thunk of the compile's [`SignalArm`].
    SigTake,
    /// #932 — the handler-return thunk of the compile's [`SignalArm`].
    SigReturn,
    FiberNew,
    FiberResume,
    /// I48 — `cont.resume.block`'s thunk.
    FiberResumeBlock,
    FiberSuspend,
    /// The `gc.roots` register-flush trampoline.
    GcRoots,
    ThreadSpawn,
    ThreadJoin,
    ThreadWait,
    ThreadNotify,
    InstJoin,
    InstPoll,
    InstDetach,
    InstKill,
    /// §3c — the config-record spawn (op 17).
    InstSpawn,
    ChildOffer,
    ChildGrant,
    SetjmpSlot,
    SetjmpLookup,
    /// libc `_setjmp`, called inline in the guest frame (LLVM.md §"JIT `longjmp`").
    Setjmp,
    /// libc `_longjmp`, called inline in the guest frame.
    Longjmp,
    VcpuTlsGet,
    VcpuTlsSet,
    DurableShadowBase,
    /// The calling thread's stack limit, which a root's entry trampoline takes (#1983).
    StackLimit,
    /// §5 W3 — the explicit-trap backtrace capture (`temen_capture_explicit_trap`).
    TrapCapture,
    /// The counted-fuel refill, which `fuel`'s trampoline calls.
    FuelRefill,
    /// #2164 — a demand-paged child's miss thunk ([`PageMissThunk`]), the compile's own.
    PageMiss,
}

impl HostFn {
    /// Every host function, for [`new_jit_module`] to bind.
    const ALL: [HostFn; 30] = [
        HostFn::CapThunk,
        HostFn::SigTake,
        HostFn::SigReturn,
        HostFn::FiberNew,
        HostFn::FiberResume,
        HostFn::FiberResumeBlock,
        HostFn::FiberSuspend,
        HostFn::GcRoots,
        HostFn::ThreadSpawn,
        HostFn::ThreadJoin,
        HostFn::ThreadWait,
        HostFn::ThreadNotify,
        HostFn::InstJoin,
        HostFn::InstPoll,
        HostFn::InstDetach,
        HostFn::InstKill,
        HostFn::InstSpawn,
        HostFn::ChildOffer,
        HostFn::ChildGrant,
        HostFn::SetjmpSlot,
        HostFn::SetjmpLookup,
        HostFn::Setjmp,
        HostFn::Longjmp,
        HostFn::VcpuTlsGet,
        HostFn::VcpuTlsSet,
        HostFn::DurableShadowBase,
        HostFn::StackLimit,
        HostFn::TrapCapture,
        HostFn::FuelRefill,
        HostFn::PageMiss,
    ];

    /// The symbol compiled code calls it by.
    fn symbol(self) -> &'static str {
        match self {
            HostFn::CapThunk => "temen.cap_thunk",
            HostFn::SigTake => "temen.sig_take",
            HostFn::SigReturn => "temen.sig_return",
            HostFn::FiberNew => "temen.fiber_new",
            HostFn::FiberResume => "temen.fiber_resume",
            HostFn::FiberResumeBlock => "temen.fiber_resume_block",
            HostFn::FiberSuspend => "temen.fiber_suspend",
            HostFn::GcRoots => "temen.gc_roots",
            HostFn::ThreadSpawn => "temen.thread_spawn",
            HostFn::ThreadJoin => "temen.thread_join",
            HostFn::ThreadWait => "temen.thread_wait",
            HostFn::ThreadNotify => "temen.thread_notify",
            HostFn::InstJoin => "temen.inst_join",
            HostFn::InstPoll => "temen.inst_poll",
            HostFn::InstDetach => "temen.inst_detach",
            HostFn::InstKill => "temen.inst_kill",
            HostFn::InstSpawn => "temen.inst_spawn",
            HostFn::ChildOffer => "temen.child_offer",
            HostFn::ChildGrant => "temen.child_grant",
            HostFn::SetjmpSlot => "temen.setjmp_slot",
            HostFn::SetjmpLookup => "temen.setjmp_lookup",
            HostFn::Setjmp => "temen.setjmp",
            HostFn::Longjmp => "temen.longjmp",
            HostFn::VcpuTlsGet => "temen.vcpu_tls_get",
            HostFn::VcpuTlsSet => "temen.vcpu_tls_set",
            HostFn::DurableShadowBase => "temen.durable_shadow_base",
            HostFn::StackLimit => "temen.stack_limit",
            HostFn::TrapCapture => "temen.trap_capture",
            HostFn::FuelRefill => "temen.fuel_refill",
            HostFn::PageMiss => "temen.page_miss",
        }
    }

    /// Its C signature, in the target's calling convention `cc`.
    fn signature(self, cc: cranelift_codegen::isa::CallConv) -> Signature {
        use HostFn::*;
        let (params, returns): (&[Type], &[Type]) = match self {
            // (ctx, mem_base, mem_size, mem_reserved, type_id, op, handle, args, n_args, results,
            // n_results, vmctx)
            CapThunk => (
                &[I64, I64, I64, I64, I32, I32, I32, I64, I64, I64, I64, I64],
                &[],
            ),
            // (ctx, out) -> 1 when a signal is deliverable, 0 when not
            SigTake => (&[I64, I64], &[I32]),
            // (ctx)
            SigReturn => (&[I64], &[]),
            // (mem_base, fn_table, vmctx, funcref, sp, fuel) -> handle (16-bit slot, 48-bit
            // generation)
            FiberNew => (&[I64, I64, I64, I32, I64, I64], &[I64]),
            // (handle, arg, status_out, vmctx) -> value
            FiberResume => (&[I64, I64, I64, I64], &[I64]),
            // (domain, handle, arg, status_out, vmctx) -> value
            FiberResumeBlock => (&[I64, I64, I64, I64, I64], &[I64]),
            // (value, vmctx) -> value
            FiberSuspend => (&[I64, I64], &[I64]),
            // (args) -> the candidates found
            GcRoots => (&[I64], &[I64]),
            // (domain, mem_base, fn_table, vmctx, func, sp, arg) -> handle
            ThreadSpawn => (&[I64, I64, I64, I64, I32, I64, I64], &[I32]),
            // (domain, handle, vmctx) -> result
            ThreadJoin => (&[I64, I32, I64], &[I64]),
            // (domain, addr, expected, width, timeout, vmctx) -> how the wait ended
            ThreadWait => (&[I64, I64, I64, I32, I64, I64], &[I32]),
            // (domain, addr, count) -> the waiters woken
            ThreadNotify => (&[I64, I64, I32], &[I32]),
            // (nursery, mem_base, instantiator, child, status_out, vmctx) -> value
            InstJoin => (&[I64, I64, I32, I32, I64, I64], &[I64]),
            // (nursery, mem_base, instantiator, child, vmctx) -> status
            InstPoll | InstDetach | InstKill => (&[I64, I64, I32, I32, I64], &[I32]),
            // (nursery, mem_base, mem_size, instantiator, record, vmctx) -> child
            InstSpawn => (&[I64, I64, I64, I32, I64, I64], &[I32]),
            // (nursery, mem_base, instantiator, child, arg, vmctx) -> handle or -errno
            ChildOffer | ChildGrant => (&[I64, I64, I32, I32, I64, I64], &[I32]),
            // (table, buf) -> jmp_buf
            SetjmpSlot => (&[I64, I64], &[I64]),
            // (table, buf, vmctx) -> jmp_buf
            SetjmpLookup => (&[I64, I64, I64], &[I64]),
            // (jmp_buf) -> 0, or the value a `longjmp` to it passes
            Setjmp => (&[I64], &[I32]),
            // (jmp_buf, value)
            Longjmp => (&[I64, I32], &[]),
            VcpuTlsGet | DurableShadowBase | StackLimit => (&[], &[I64]),
            // (value)
            VcpuTlsSet => (&[I64], &[]),
            // (the trapping frame pointer)
            TrapCapture => (&[I64], &[]),
            // (the fuel cell)
            FuelRefill => (&[I64], &[]),
            // (vmctx, mem_base, off, len, flags)
            PageMiss => (&[I64, I64, I64, I64, I64], &[]),
        };
        c_signature(cc, params, returns)
    }

    /// The function's address in this process: the compile's own from `imports`. `None` for one
    /// the compile has none of, or this target lacks: no code it emits calls those.
    fn addr(self, imports: &Imports) -> Option<*const u8> {
        let f: *const () = match self {
            HostFn::CapThunk => return Some(imports.cap_thunk).filter(|p| !p.is_null()),
            HostFn::SigTake => return imports.sig.map(|(take, _)| take as *const u8),
            HostFn::SigReturn => return imports.sig.map(|(_, ret)| ret as *const u8),
            HostFn::PageMiss => return imports.page_miss.map(|miss| miss as *const u8),
            #[cfg(fiber_rt)]
            HostFn::FiberNew => fiber_rt::fiber_new as *const (),
            #[cfg(fiber_rt)]
            HostFn::FiberResume => fiber_rt::fiber_resume as *const (),
            #[cfg(fiber_rt)]
            HostFn::FiberResumeBlock => os_thread_rt::fiber_resume_block as *const (),
            #[cfg(fiber_rt)]
            HostFn::FiberSuspend => fiber_rt::fiber_suspend as *const (),
            // The register-flush trampoline, not `gc_roots` itself: it spills the callee-saved
            // registers before the scan, so a guest root parked in one is captured.
            #[cfg(fiber_rt)]
            HostFn::GcRoots => fiber_rt::temen_gc_roots_flush as *const (),
            #[cfg(fiber_rt)]
            HostFn::ThreadSpawn => os_thread_rt::thread_spawn as *const (),
            #[cfg(fiber_rt)]
            HostFn::ThreadJoin => os_thread_rt::thread_join as *const (),
            #[cfg(fiber_rt)]
            HostFn::ThreadWait => os_thread_rt::thread_wait as *const (),
            #[cfg(fiber_rt)]
            HostFn::ThreadNotify => os_thread_rt::thread_notify as *const (),
            #[cfg(fiber_rt)]
            HostFn::InstJoin => instantiator_rt::join as *const (),
            #[cfg(fiber_rt)]
            HostFn::InstPoll => instantiator_rt::poll as *const (),
            #[cfg(fiber_rt)]
            HostFn::InstDetach => instantiator_rt::detach as *const (),
            #[cfg(fiber_rt)]
            HostFn::InstKill => instantiator_rt::kill as *const (),
            #[cfg(fiber_rt)]
            HostFn::InstSpawn => instantiator_rt::instantiate_rec as *const (),
            #[cfg(fiber_rt)]
            HostFn::ChildOffer => instantiator_rt::child_offer as *const (),
            #[cfg(fiber_rt)]
            HostFn::ChildGrant => instantiator_rt::child_grant as *const (),
            #[cfg(setjmp_rt)]
            HostFn::SetjmpSlot => setjmp_rt::rt_setjmp_slot as *const (),
            #[cfg(setjmp_rt)]
            HostFn::SetjmpLookup => setjmp_rt::rt_setjmp_lookup as *const (),
            #[cfg(setjmp_rt)]
            HostFn::Setjmp => setjmp_rt::setjmp_addr(),
            #[cfg(setjmp_rt)]
            HostFn::Longjmp => setjmp_rt::longjmp_addr(),
            HostFn::VcpuTlsGet => vcpu_tls::get as *const (),
            HostFn::VcpuTlsSet => vcpu_tls::set as *const (),
            HostFn::DurableShadowBase => durable_shadow::get as *const (),
            HostFn::StackLimit => stack_check::thread_limit as *const (),
            HostFn::TrapCapture => trap_capture_addr()?,
            HostFn::FuelRefill => fuel::fuel_refill as *const (),
            #[allow(unreachable_patterns)]
            _ => return None,
        };
        Some(f as *const u8)
    }
}

/// The embedder's functions a compile's code calls: what [`new_jit_module`] binds their
/// [`HostFn`] symbols to, beside temen-jit's own functions. Each compile supplies its own.
#[derive(Clone, Copy)]
struct Imports {
    /// The `call.cap` thunk ([`HostFn::CapThunk`]); null for code that makes no `call.cap`.
    cap_thunk: *const u8,
    /// #932 — the signal-delivery thunks, when the compile arms delivery.
    sig: Option<(SigTakeThunk, SigReturnThunk)>,
    /// §9/D45 — the resolver that binds a fast-path `call.cap`'s target ([`fast_cap_symbol`]).
    fast_resolver: Option<FastCapResolver>,
    /// #2164 — a demand-paged child's miss thunk.
    page_miss: Option<PageMissThunk>,
}

impl Imports {
    /// None: code that calls temen-jit's own functions alone.
    const NONE: Imports = Imports {
        cap_thunk: core::ptr::null(),
        sig: None,
        fast_resolver: None,
        page_miss: None,
    };
}

/// A C signature in the calling convention `cc`.
fn c_signature(
    cc: cranelift_codegen::isa::CallConv,
    params: &[Type],
    returns: &[Type],
) -> Signature {
    let mut sig = Signature::new(cc);
    sig.params.extend(params.iter().map(|&t| AbiParam::new(t)));
    sig.returns
        .extend(returns.iter().map(|&t| AbiParam::new(t)));
    sig
}

/// Call the host function `f`: a call to the symbol the code names it by.
fn call_host(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    f: HostFn,
    args: &[Value],
) -> Result<cranelift_codegen::ir::Inst, JitError> {
    let callee = import_host(module, b.func, f)?;
    Ok(b.ins().call(callee, args))
}

/// Import the host function `f` into `func`.
fn import_host(
    module: &mut JITModule,
    func: &mut Function,
    f: HostFn,
) -> Result<FuncRef, JitError> {
    let sig = f.signature(module.isa().default_call_conv());
    import_symbol(module, func, f.symbol(), &sig)
}

/// Import the host symbol `name`, whose C signature is `sig`, into `func`.
fn import_symbol(
    module: &mut JITModule,
    func: &mut Function,
    name: &str,
    sig: &Signature,
) -> Result<FuncRef, JitError> {
    let id = module
        .declare_function(name, Linkage::Import, sig)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    Ok(module.declare_func_in_func(id, func))
}

/// The symbol a §9/D45 fast-path `call.cap` calls its target by: the `(type_id, op, n_args,
/// n_res)` the compile's resolver claimed, which [`fast_cap_addr`] asks it again at link time.
fn fast_cap_symbol(type_id: u32, op: u32, n_args: usize, n_res: usize) -> String {
    format!("temen.fast_cap.{type_id}.{op}.{n_args}.{n_res}")
}

/// A fast-path target's C signature: `(ctx, mem_base, mem_size, handle, vmctx, args…) ->
/// [result]`, each arg and the result in its `i64` register slot.
fn fast_cap_signature(
    cc: cranelift_codegen::isa::CallConv,
    n_args: usize,
    n_res: usize,
) -> Signature {
    let mut params = vec![I64, I64, I64, I32, I64];
    params.resize(5 + n_args, I64);
    c_signature(cc, &params, &[I64][..n_res])
}

/// The `(type_id, op, n_args, n_res)` a [`fast_cap_symbol`] names; `None` for any other symbol.
fn fast_cap_key(name: &str) -> Option<[u32; 4]> {
    let key: Vec<u32> = name
        .strip_prefix("temen.fast_cap.")?
        .split('.')
        .map(|n| n.parse().ok())
        .collect::<Option<_>>()?;
    key.try_into().ok()
}

/// The target `resolver` binds a [`fast_cap_symbol`] to; `None` for any other symbol.
fn fast_cap_addr(resolver: FastCapResolver, name: &str) -> Option<*const u8> {
    let [type_id, op, n_args, n_res] = fast_cap_key(name)?;
    // SAFETY: `resolver` honours the `FastCapResolver` contract, which the compile's caller
    // guaranteed: a pure lookup, which the compile already made for this key (`fast_cap_target`).
    let target = unsafe { resolver(type_id, op, n_args, n_res) };
    (!target.is_null()).then_some(target as *const u8)
}

/// The C signature, in the calling convention `cc`, of the host function the symbol `name` names
/// ([`HostFn`], or a fast-path target), if a link with the compile's `imports` binds it: what a
/// compile declares the imports of code a [`CodeDir`] kept with (temen#2452).
fn host_import(
    name: &str,
    cc: cranelift_codegen::isa::CallConv,
    imports: &Imports,
) -> Option<Signature> {
    if let Some(f) = HostFn::ALL.into_iter().find(|f| f.symbol() == name) {
        return f.addr(imports).map(|_| f.signature(cc));
    }
    let [_, _, n_args, n_res] = fast_cap_key(name)?;
    let bound = fast_cap_addr(imports.fast_resolver?, name).is_some();
    (bound && n_res <= 1).then(|| fast_cap_signature(cc, n_args as usize, n_res as usize))
}

/// Build a fresh `JITModule` with the escape-TCB ISA configuration **every** compile path shares
/// (top-level [`CompiledModule::compile`] and nested [`compile_child_windowed`]) — one audited copy,
/// so the soundness-critical flags cannot drift between them. Most load-bearing is
/// `enable_probestack=false`: the software stack-overflow guard (`emit_stack_check` / STACK_GUARD.md)
/// is sound *only because* a large frame's `sub rsp` is a pure pointer move touching no pages before
/// the entry-block check — a probestack that page-walked the frame downward would touch below the
/// fiber's low bound *before* the check (silent neighbour-slot corruption under the arena, which has
/// no guard page). Pinning it here locks that assumption against a future Cranelift default flip.
///
/// It binds every symbol the code calls the host by ([`HostFn`]): temen-jit's functions, and the
/// compile's `imports`; and Cranelift's libcalls ([`libcalls`]).
fn new_jit_module(imports: Imports) -> Result<JITModule, JitError> {
    let mut builder = JITBuilder::with_isa(jit_isa()?, cranelift_module::default_libcall_names());
    for f in HostFn::ALL {
        if let Some(addr) = f.addr(&imports) {
            builder.symbol(f.symbol(), addr);
        }
    }
    for (name, addr) in libcalls() {
        builder.symbol(name, addr);
    }
    if let Some(resolver) = imports.fast_resolver {
        builder.symbol_lookup_fn(Box::new(move |name| fast_cap_addr(resolver, name)));
    }
    // Code + read-only data (float constants, jump tables) from a single contiguous 256 MiB arena
    // (VA only, committed on demand) rather than separate `mmap`s: x64's 32-bit PC-relative
    // relocations (`X86CallPCRel4`/`X86PCRel4`) overflow if ASLR places independent mmaps > 2 GiB
    // apart (a `compiled_blob.rs` panic on large modules with rodata). Falls back to the default
    // provider if it cannot map.
    if let Ok(arena) = cranelift_jit::ArenaMemoryProvider::new_with_size(256 << 20) {
        builder.memory_provider(Box::new(arena));
    }
    Ok(JITModule::new(builder))
}

/// Cranelift's libcalls that name a C runtime function, by the names
/// [`cranelift_module::default_libcall_names`] gives them, with that function's address.
/// [`new_jit_module`] binds them as it binds each [`HostFn`], so the JIT never looks a name up in the
/// process's dynamic symbol table, which a statically linked host doesn't have: there, a
/// `memory.copy` could not link (#2514). The others (`Probestack`, which is off, and the TLS and x86
/// shuffle helpers) name nothing a C runtime has.
fn libcalls() -> [(&'static str, *const u8); 14] {
    use core::ffi::{c_int, c_void};
    extern "C" {
        fn memcpy(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;
        fn memmove(dst: *mut c_void, src: *const c_void, n: usize) -> *mut c_void;
        fn memset(dst: *mut c_void, c: c_int, n: usize) -> *mut c_void;
        fn memcmp(a: *const c_void, b: *const c_void, n: usize) -> c_int;
        fn ceilf(x: f32) -> f32;
        fn ceil(x: f64) -> f64;
        fn floorf(x: f32) -> f32;
        fn floor(x: f64) -> f64;
        fn truncf(x: f32) -> f32;
        fn trunc(x: f64) -> f64;
        fn nearbyintf(x: f32) -> f32;
        fn nearbyint(x: f64) -> f64;
        fn fmaf(a: f32, b: f32, c: f32) -> f32;
        fn fma(a: f64, b: f64, c: f64) -> f64;
    }
    [
        ("memcpy", memcpy as *const u8),
        ("memmove", memmove as *const u8),
        ("memset", memset as *const u8),
        ("memcmp", memcmp as *const u8),
        ("ceilf", ceilf as *const u8),
        ("ceil", ceil as *const u8),
        ("floorf", floorf as *const u8),
        ("floor", floor as *const u8),
        ("truncf", truncf as *const u8),
        ("trunc", trunc as *const u8),
        ("nearbyintf", nearbyintf as *const u8),
        ("nearbyint", nearbyint as *const u8),
        ("fmaf", fmaf as *const u8),
        ("fma", fma as *const u8),
    ]
}

/// The ISA every compile uses: [`new_jit_module`]'s, and the one [`define_funcs`]'s workers compile
/// with, so the two cannot differ.
fn jit_isa() -> Result<cranelift_codegen::isa::OwnedTargetIsa, JitError> {
    let mut flags = settings::builder();
    // A JIT'd function is called directly, not relocated into a shared object.
    let _ = flags.set("is_pic", "false");
    // Cranelift's x64 `return_call` (tail calls, §3b) lowering requires frame pointers.
    let _ = flags.set("preserve_frame_pointers", "true");
    // Run Cranelift's mid-end optimizer (GVN/CSE, constant materialization, store-to-load
    // forwarding), as Wasmtime does; "SSA on the wire" keeps cold start ahead even with it on.
    let _ = flags.set("opt_level", "speed");
    // The escape-TCB pin — see `new_jit_module`'s doc.
    let _ = flags.set("enable_probestack", "false");
    // [`compile_verified`] runs Cranelift's verifier before and after Cranelift's passes on every
    // compile (#2449). With debug assertions and under the fuzzers, Cranelift also runs it after
    // each pass, which names the pass that broke the CLIF.
    let per_pass = cfg!(any(debug_assertions, fuzzing));
    let _ = flags.set("enable_verifier", if per_pass { "true" } else { "false" });
    cranelift_native::builder()
        .map_err(|e| JitError::Backend(e.to_string()))?
        .finish(settings::Flags::new(flags))
        .map_err(|e| JitError::Backend(e.to_string()))
}

/// What a worker hands back for one function: its code and relocations, the compile's error, or
/// the payload of a panic in the compile.
type CompiledFn = std::thread::Result<Result<(CompiledCode, Vec<ModuleReloc>), String>>;

/// Lower and define functions `0..ids.len()` (#2450), every compile path's one loop. `lower` builds
/// function `i`'s CLIF on this thread, where the module's declarations are. Worker threads compile
/// the CLIF, each function in a context of its own, and this thread defines the code in function
/// order, `defined` seeing each function's as it goes. A function's code depends on its CLIF and the
/// ISA alone, so the module's code is the same on any number of threads; a compile that fails or
/// panics fails or panics here, at its function, as it would on one thread. `code` says where the
/// code comes from: the compile, which keeps it while its record holds code, or code a [`CodeDir`]
/// kept, defined as is (temen#2452). Returns the bytes of code defined.
fn define_funcs(
    module: &mut JITModule,
    ids: &[FuncId],
    code: &mut Codegen<'_>,
    mut lower: impl FnMut(
        &mut JITModule,
        &mut cranelift_codegen::Context,
        usize,
    ) -> Result<(), JitError>,
    mut defined: impl FnMut(&JITModule, &CompiledCode, usize),
) -> Result<usize, JitError> {
    let keep = match code {
        Codegen::Replay(kept, imports) => return kept.replay(module, ids, imports),
        Codegen::Compile(keep) => keep,
    };
    let isa = jit_isa()?;
    let workers = std::thread::available_parallelism()
        .map_or(1, |n| n.get())
        .clamp(1, ids.len().max(1));
    let (work_tx, work_rx) =
        std::sync::mpsc::sync_channel::<(usize, cranelift_codegen::Context)>(2 * workers);
    let work_rx = std::sync::Mutex::new(work_rx);
    let (done_tx, done_rx) = std::sync::mpsc::channel::<(usize, CompiledFn)>();
    std::thread::scope(|scope| {
        for _ in 0..workers {
            let (work_rx, done_tx, isa) = (&work_rx, done_tx.clone(), &*isa);
            scope.spawn(move || loop {
                let job = work_rx.lock().unwrap_or_else(|p| p.into_inner()).recv();
                let Ok((i, mut ctx)) = job else { break };
                let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let code = compile_verified(isa, &mut ctx)?;
                    let relocs = code
                        .buffer
                        .relocs()
                        .iter()
                        .map(|r| ModuleReloc::from_mach_reloc(r, &ctx.func, ids[i]))
                        .collect();
                    Ok((code, relocs))
                }));
                if done_tx.send((i, out)).is_err() {
                    break;
                }
            });
        }
        drop(done_tx);
        let mut done: Vec<Option<CompiledFn>> = (0..ids.len()).map(|_| None).collect();
        let mut next = 0;
        let mut bytes = 0;
        // Define, in order, every function compiled so far.
        let mut define = |module: &mut JITModule, done: &mut [Option<CompiledFn>]| {
            while let Some(out) = done.get_mut(next).and_then(Option::take) {
                let (code, relocs) = match out {
                    Ok(compiled) => compiled.map_err(JitError::Backend)?,
                    Err(panic) => std::panic::resume_unwind(panic),
                };
                let align = code.buffer.alignment as u64;
                module
                    .define_function_bytes(ids[next], align, code.code_buffer(), &relocs)
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                #[cfg(test)]
                record_emitted(module, code.code_buffer(), &relocs);
                let kept = keep
                    .as_mut()
                    .is_some_and(|k| k.push(module, ids, align, code.code_buffer(), &relocs));
                if !kept {
                    *keep = None;
                }
                bytes += code.code_buffer().len();
                defined(module, &code, next);
                next += 1;
            }
            Ok::<(), JitError>(())
        };
        for i in 0..ids.len() {
            let mut ctx = module.make_context();
            lower(module, &mut ctx, i)?;
            work_tx
                .send((i, ctx))
                .map_err(|_| JitError::Backend("no compile worker".into()))?;
            for (j, out) in done_rx.try_iter() {
                done[j] = Some(out);
            }
            define(module, &mut done)?;
        }
        drop(work_tx);
        for (j, out) in done_rx.iter() {
            done[j] = Some(out);
            define(module, &mut done)?;
        }
        Ok(bytes)
    })
}

/// Compile `ctx`'s CLIF for `isa`, with Cranelift's verifier at the two points #2449 settled on:
/// on the CLIF temen lowered, before Cranelift's passes rely on it being well formed, and on the CLIF
/// those passes hand the backend. A lowering bug that makes malformed CLIF fails the compile there,
/// and so does a pass that breaks it. With debug assertions and under the fuzzers, [`jit_isa`]'s
/// `enable_verifier` also runs the verifier after each pass. No verifier sees the machine code the
/// backend emits (#2481).
fn compile_verified(
    isa: &dyn cranelift_codegen::isa::TargetIsa,
    ctx: &mut cranelift_codegen::Context,
) -> Result<CompiledCode, String> {
    let mut ctrl = cranelift_codegen::control::ControlPlane::default();
    ctx.verify(isa)
        .map_err(|e| format!("CLIF verifier, on the lowered CLIF: {e}"))?;
    ctx.optimize(isa, &mut ctrl)
        .map_err(|e| cranelift_module::ModuleError::from(e).to_string())?;
    ctx.verify(isa)
        .map_err(|e| format!("CLIF verifier, after Cranelift's passes: {e}"))?;
    let code = isa
        .compile_function(&ctx.func, &ctx.domtree, ctx.want_disasm, &mut ctrl)
        .map_err(|e| cranelift_module::ModuleError::from(e).to_string())?;
    Ok(code.apply_params(&ctx.func.params))
}

/// Define `id` in `module` from `ctx`'s CLIF, compiled by [`compile_verified`], and return its code.
/// Every function the JIT defines outside [`define_funcs`] (the trampolines, and `fuel`'s refill
/// trampoline) comes through here.
fn define_verified(
    module: &mut JITModule,
    id: FuncId,
    ctx: &mut cranelift_codegen::Context,
) -> Result<CompiledCode, JitError> {
    let code = compile_verified(module.isa(), ctx).map_err(JitError::Backend)?;
    let relocs: Vec<ModuleReloc> = code
        .buffer
        .relocs()
        .iter()
        .map(|r| ModuleReloc::from_mach_reloc(r, &ctx.func, id))
        .collect();
    module
        .define_function_bytes(
            id,
            code.buffer.alignment as u64,
            code.code_buffer(),
            &relocs,
        )
        .map_err(|e| JitError::Backend(e.to_string()))?;
    #[cfg(test)]
    record_emitted(module, code.code_buffer(), &relocs);
    Ok(code)
}

#[cfg(test)]
thread_local! {
    /// temen#2452 — what the compiling thread defines while a test records it: each function's code
    /// as Cranelift emitted it, before its link, and its relocations, their targets by symbol.
    static EMITTED: core::cell::RefCell<Option<Vec<u8>>> = const { core::cell::RefCell::new(None) };
}

/// Record a function `module` defines from `code` and `relocs`, if a test is recording.
#[cfg(test)]
fn record_emitted(module: &JITModule, code: &[u8], relocs: &[ModuleReloc]) {
    EMITTED.with(|e| {
        let mut e = e.borrow_mut();
        let Some(out) = e.as_mut() else { return };
        out.extend_from_slice(code);
        for r in relocs {
            let target = if cranelift_module::ModuleDeclarations::is_function(&r.name) {
                let id = FuncId::from_name(&r.name);
                format!("{:?}", module.declarations().get_function_decl(id).name)
            } else {
                format!("{:?}", r.name)
            };
            let line = format!("\n{} {:?} {target} {}\n", r.offset, r.kind, r.addend);
            out.extend_from_slice(line.as_bytes());
        }
    });
}

/// temen#2452 — the code a compile emits names no address in its process: two processes emit the
/// same code for one program, though each placed the functions it calls elsewhere.
#[cfg(all(test, fiber_rt, setjmp_rt))]
mod relocatable_code_tests {
    use super::*;
    use core::ffi::c_void;

    /// A program whose code calls every host function a root's can but `setjmp`'s, which a module
    /// that spawns may not use: `call.cap`s (one on the D45 fast path), fibers, threads, the futex,
    /// `gc.roots`, the vCPU TLS register, the durable shadow base, every `Instantiator` op, and a
    /// trap, which captures a backtrace under `-g`.
    const CALLS: &str = "memory 16
func (i32, i32, i64) -> (i64) {
block 0 (vh: i32, vi: i32, vx: i64) {
  vgen = call.cap 3 7 (i64) -> (i64) vh (vx)
  vfast = call.cap 3 8 (i64) -> (i64) vh (vx)
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vs1, vv1 = cont.resume vk vx
  vs2, vv2 = cont.resume.block vk vx
  vt = thread.spawn 1 vsp vx
  vj = thread.join vt
  va = i64.const 64
  ve = i32.const 0
  vto = i64.const 0
  vw = i32.atomic.wait va ve vto
  vc = i32.const 1
  vn = atomic.notify va vc
  vlo = i64.const 0
  vhi = i64.const 4096
  vm = i64.const -1
  vbuf = i64.const 128
  vcap = i64.const 4
  vr = gc.roots vlo vhi vm vbuf vcap
  vtls = vcpu.tls.get
  vcpu.tls.set vtls
  vsb = durable.shadow_base
  vkid = call.cap 6 17 (i64) -> (i32) vi (vx)
  vst, vval = call.cap 6 1 (i32) -> (i64, i64) vi (vkid)
  vp = call.cap 6 9 (i32) -> (i32) vi (vkid)
  vd = call.cap 6 10 (i32) -> (i32) vi (vkid)
  vkl = call.cap 6 12 (i32) -> (i32) vi (vkid)
  vo = call.cap 6 14 (i32, i64) -> (i32) vi (vkid, vx)
  vg = call.cap 6 19 (i32, i64) -> (i32) vi (vkid, vx)
  vz = i64.const 0
  vq = i64.div_s vx vz
  return vq
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vs = suspend varg
  return vs
  }
}
";

    /// `setjmp` and `longjmp`.
    const JUMPS: &str = "memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 64
  v2 = setjmp v1
  v3 = i32.const 0
  v4 = i32.eq v2 v3
  br_if v4 1() 2(v2)
  }
block 1 () {
  v5 = i64.const 64
  v6 = i32.const 7
  longjmp v5 v6
  unreachable
  }
block 2 (v7: i32) {
  v8 = i64.extend_i32_u v7
  return v8
  }
}
";

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn cap_thunk(
        _: *mut c_void,
        _: *mut u8,
        _: u64,
        _: u64,
        _: u32,
        _: u32,
        _: i32,
        _: *const i64,
        _: u64,
        _: *mut i64,
        _: u64,
        _: &VmCtx,
    ) {
    }

    unsafe extern "C" fn fast_target(
        _: *mut c_void,
        _: *mut u8,
        _: u64,
        _: i32,
        _: &VmCtx,
        _: i64,
    ) -> i64 {
        0
    }

    /// Claims `call.cap 3 8` with one argument and one result.
    unsafe extern "C" fn resolver(type_id: u32, op: u32, n_args: u32, n_res: u32) -> *const c_void {
        if (type_id, op, n_args, n_res) == (3, 8, 1, 1) {
            fast_target as *const c_void
        } else {
            core::ptr::null()
        }
    }

    unsafe extern "C" fn sig_take(_: *mut c_void, _: *mut i64) -> i32 {
        0
    }

    unsafe extern "C" fn sig_return(_: *mut c_void) {}

    unsafe extern "C" fn page_miss(_: &VmCtx, _: *mut u8, _: u64, _: u64, _: u64) {}

    /// What the test programs' compiles bind beside temen-jit's own functions.
    fn imports() -> Imports {
        Imports {
            cap_thunk: cap_thunk as *const u8,
            sig: Some((sig_take, sig_return)),
            fast_resolver: Some(resolver),
            page_miss: Some(page_miss),
        }
    }

    fn program(src: &str) -> IrModule {
        let m = temen_text::parse_module(src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    }

    /// What this process's compiles of the programs emit: a root's of each, and two §14 children's
    /// of the calls, one of them demand-paged (#2164).
    fn emit() -> Vec<u8> {
        EMITTED.with(|e| *e.borrow_mut() = Some(Vec::new()));
        // Its refill trampoline compiles now, in a process that has compiled none.
        let mut fuel = FuelCell::fixed(1);
        let interrupt = AtomicU64::new(0);
        let armed = core::sync::atomic::AtomicBool::new(false);
        let mut calls = program(CALLS);
        calls.debug_info = Some(temen_ir::DebugInfo::default());
        let mut root = |m: &IrModule, signal| {
            CompiledModule::compile_with_signal(
                m,
                0,
                cap_thunk,
                core::ptr::null_mut(),
                0,
                None,
                Some(&interrupt as *const AtomicU64),
                Some(&mut *fuel as *mut FuelCell),
                Some(resolver),
                None,
                0,
                signal,
                None,
            )
            .expect("a root's compile")
        };
        let calls_root = root(
            &calls,
            Some(SignalArm {
                armed: &armed,
                take: sig_take,
                ret: sig_return,
                ctx: core::ptr::null_mut(),
            }),
        );
        let jumps_root = root(&program(JUMPS), None);
        // Never run, so the domain, the nursery and the access table it names are never reached.
        let child_of = |pages: *const AtomicU8, page_miss: Option<PageMissThunk>| {
            compile_child_windowed(
                &ChildCompile {
                    funcs: &calls.funcs,
                    types: &calls.types,
                    entry: 0,
                    mapped_log2: 16,
                    reserved_log2: 16,
                    thunk: cap_thunk,
                    addrs: InstanceAddrs {
                        sched: core::ptr::NonNull::<c_void>::dangling().as_ptr(),
                        nursery: core::ptr::NonNull::<c_void>::dangling().as_ptr(),
                        pages,
                        ..InstanceAddrs::NONE
                    },
                    impl_exports: &[],
                    table_reserve_log2: 0,
                    shadow: temen_ir::durable_abi::ShadowArena::EMPTY,
                    page_miss,
                },
                None,
            )
            .expect("a child's compile")
        };
        let child = child_of(core::ptr::null(), None);
        let paged = child_of(
            core::ptr::NonNull::<AtomicU8>::dangling().as_ptr(),
            Some(page_miss),
        );
        drop((calls_root, jumps_root, child, paged));
        EMITTED.with(|e| e.borrow_mut().take()).expect("recorded")
    }

    const EMIT_TO: &str = "TEMEN_JIT_EMIT_TO";

    /// The half of [`two_processes_emit_the_same_code`] that runs in a process of its own: it
    /// writes where this process placed each host function, then what it emitted.
    #[test]
    #[ignore = "run in a process of its own by two_processes_emit_the_same_code"]
    fn emit_in_this_process() {
        let Some(path) = std::env::var_os(EMIT_TO) else {
            return;
        };
        let mut out: Vec<u8> = HostFn::ALL
            .iter()
            .flat_map(|f| (f.addr(&imports()).expect("bound") as u64).to_le_bytes())
            .collect();
        out.extend(emit());
        std::fs::write(path, out).expect("write what this process emitted");
    }

    #[test]
    // A static Android executable is never position-independent (rustc links it `-no-pie`), so two
    // processes place the host functions alike, and this test can't tell an address from a symbol.
    // `scripts/ci/android.sh` links its test binaries that way (#2514).
    #[cfg_attr(
        all(target_os = "android", target_feature = "crt-static"),
        ignore = "needs ASLR, which a static Android executable lacks"
    )]
    fn two_processes_emit_the_same_code() {
        let exe = std::env::current_exe().expect("this test binary");
        let run = |name: &str| {
            let path =
                std::env::temp_dir().join(format!("temen-jit-emit-{}-{name}", std::process::id()));
            let run = std::process::Command::new(&exe)
                .args([
                    "--exact",
                    "relocatable_code_tests::emit_in_this_process",
                    "--ignored",
                ])
                .env(EMIT_TO, &path)
                .output()
                .expect("run this test binary");
            assert!(
                run.status.success(),
                "process {name}: {}\n{}{}",
                run.status,
                String::from_utf8_lossy(&run.stdout),
                String::from_utf8_lossy(&run.stderr)
            );
            let out = std::fs::read(&path).expect("read what the process emitted");
            let _ = std::fs::remove_file(&path);
            let (addrs, code) = out.split_at(8 * HostFn::ALL.len());
            let addrs: Vec<u64> = addrs
                .chunks(8)
                .map(|a| u64::from_le_bytes(a.try_into().unwrap()))
                .collect();
            (addrs, code.to_vec())
        };
        let (addrs_a, a) = run("a");
        let (addrs_b, b) = run("b");
        assert_ne!(
            addrs_a, addrs_b,
            "both processes placed the host functions at the same addresses, so their code \
             cannot tell an address from a symbol: this test needs ASLR"
        );
        for (addrs, code) in [(&addrs_a, &a), (&addrs_b, &b)] {
            for (f, addr) in HostFn::ALL.iter().zip(addrs) {
                let named = code
                    .windows(f.symbol().len())
                    .any(|w| w == f.symbol().as_bytes());
                assert!(named, "no code calls {}", f.symbol());
                let held = code.windows(8).any(|w| w == addr.to_le_bytes());
                assert!(!held, "the code holds the address of {}", f.symbol());
            }
        }
        let fast = fast_cap_symbol(3, 8, 1, 1);
        assert!(a.windows(fast.len()).any(|w| w == fast.as_bytes()));
        assert!(
            a == b,
            "two processes emitted different code for the same programs ({} and {} bytes)",
            a.len(),
            b.len()
        );
    }
}

#[cfg(test)]
mod libcall_tests {
    use super::*;
    use cranelift_codegen::ir::LibCall;

    /// #2514 — every libcall that names a C runtime function is bound, so a Cranelift bump that adds
    /// one fails here, not at a statically linked host's first call of it.
    #[test]
    fn every_c_runtime_libcall_is_bound() {
        let name = cranelift_module::default_libcall_names();
        let bound: Vec<&str> = libcalls().iter().map(|&(n, _)| n).collect();
        for &lc in LibCall::all_libcalls() {
            let runtime = !matches!(
                lc,
                LibCall::Probestack
                    | LibCall::ElfTlsGetAddr
                    | LibCall::ElfTlsGetOffset
                    | LibCall::X86Pshufb
            );
            let n = name(lc);
            assert_eq!(bound.contains(&n.as_str()), runtime, "{lc:?} ({n})");
        }
        assert_eq!(bound.len(), LibCall::all_libcalls().len() - 4);
        assert!(libcalls().iter().all(|&(_, addr)| !addr.is_null()));
    }
}

#[cfg(test)]
mod compile_verified_tests {
    use super::*;

    /// A function adding an `i32` to an `i64`, and returning the sum from an `i64` function.
    fn malformed(isa: &dyn cranelift_codegen::isa::TargetIsa) -> Function {
        let mut sig = Signature::new(isa.default_call_conv());
        sig.returns.push(AbiParam::new(I64));
        let mut func = Function::with_name_signature(UserFuncName::default(), sig);
        let mut fctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut func, &mut fctx);
        let block = b.create_block();
        b.switch_to_block(block);
        b.seal_block(block);
        let x = b.ins().iconst(I32, 1);
        let y = b.ins().iconst(I64, 2);
        let sum = b.ins().iadd(x, y);
        b.ins().return_(&[sum]);
        b.finalize();
        func
    }

    #[test]
    fn malformed_clif_fails_the_compile_before_cranelifts_passes() {
        let isa = jit_isa().unwrap();
        let mut ctx = cranelift_codegen::Context::for_function(malformed(&*isa));
        let err = compile_verified(&*isa, &mut ctx).unwrap_err();
        assert!(
            err.starts_with("CLIF verifier, on the lowered CLIF"),
            "{err}"
        );
    }

    /// A loop summing `0..n`: block params, a branch and a back edge for Cranelift's passes.
    fn summing(isa: &dyn cranelift_codegen::isa::TargetIsa) -> Function {
        let mut sig = Signature::new(isa.default_call_conv());
        sig.params.push(AbiParam::new(I64));
        sig.returns.push(AbiParam::new(I64));
        let mut func = Function::with_name_signature(UserFuncName::default(), sig);
        let mut fctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut func, &mut fctx);
        let [entry, head, body, exit] = [(); 4].map(|_| b.create_block());
        b.append_block_params_for_function_params(entry);
        let (i, sum) = (
            b.append_block_param(head, I64),
            b.append_block_param(head, I64),
        );
        let total = b.append_block_param(exit, I64);
        b.switch_to_block(entry);
        let n = b.block_params(entry)[0];
        let zero = b.ins().iconst(I64, 0);
        b.ins()
            .jump(head, &[BlockArg::Value(zero), BlockArg::Value(zero)]);
        b.switch_to_block(head);
        let done = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, i, n);
        b.ins().brif(done, exit, &[BlockArg::Value(sum)], body, &[]);
        b.switch_to_block(body);
        let (sum, i) = (b.ins().iadd(sum, i), b.ins().iadd_imm(i, 1));
        b.ins()
            .jump(head, &[BlockArg::Value(i), BlockArg::Value(sum)]);
        b.switch_to_block(exit);
        b.ins().return_(&[total]);
        b.seal_all_blocks();
        b.finalize();
        func
    }

    /// [`compile_verified`] repeats the steps of Cranelift's `Context::compile` with the verifier
    /// around its passes, so a Cranelift whose compile gains a step fails here rather than
    /// compiling without it.
    #[test]
    fn compile_verified_emits_what_cranelifts_compile_emits() {
        let isa = jit_isa().unwrap();
        let mut ours = cranelift_codegen::Context::for_function(summing(&*isa));
        let code = compile_verified(&*isa, &mut ours).unwrap();
        let mut theirs = cranelift_codegen::Context::for_function(summing(&*isa));
        let want = theirs.compile(&*isa, &mut Default::default()).unwrap();
        assert_eq!(code.code_buffer(), want.code_buffer());
        assert_eq!(code.buffer.relocs().len(), want.buffer.relocs().len());
    }
}

/// Declare every function (natural ABI, `f{i}` local linkage) up front so any call site can
/// reference any of them — shared by both compile paths.
fn declare_all_funcs(module: &mut JITModule, funcs: &[Func]) -> Result<Vec<FuncId>, JitError> {
    funcs
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let sig = natural_sig(module, f);
            module
                .declare_function(&format!("f{i}"), Linkage::Local, &sig)
                .map_err(|e| JitError::Backend(e.to_string()))
        })
        .collect()
}

/// What a §14 child's compile is given ([`compile_child_windowed`]). All of it but the module itself
/// (its functions, types and shadow arena, which the module's digest names) is what else its code
/// depends on, so a [`ChildCodeKey`] is made of exactly these fields.
#[derive(Clone, Copy)]
pub(crate) struct ChildCompile<'a> {
    pub(crate) funcs: &'a [Func],
    /// #922 — the child module's type section, so `call.dyn`/`call.cap` interned type indices in
    /// `funcs` resolve to their `FuncType` during lowering.
    pub(crate) types: &'a [temen_ir::TypeEntry],
    pub(crate) entry: FuncIdx,
    /// Its committed window, inside a reservation of `reserved_log2`.
    pub(crate) mapped_log2: u8,
    pub(crate) reserved_log2: u8,
    /// The `call.cap` thunk its sites call.
    pub(crate) thunk: CapThunk,
    /// The first instance's addresses: its powerbox, the run's kill-path cell, its own budget cell
    /// (null: un-metered), the domain its futex sites rendezvous in (null keeps futex ops rejected:
    /// the durable path), and its own nursery (null: it cannot spawn). The code polls, charges, waits
    /// and nests exactly when its instance has the cell, so every instance of it has the same set.
    pub(crate) addrs: InstanceAddrs,
    /// CALLS.md 5c.1a — the child module's impl exports: each **handler** is given a buffer-ABI
    /// serve trampoline so a cross-domain dispatch can invoke it over the child's live window, as a
    /// root's compile gives its own (#2247). Empty ⇒ a child that offers nothing.
    pub(crate) impl_exports: &'a [temen_ir::ImplExport],
    /// #1296 — the child's `call.dyn` table reservation (`0` ⇒ natural), see [`GrantChild`].
    pub(crate) table_reserve_log2: u8,
    /// The child module's declared shadow arena (its ctx-0 words + regions live in its own window).
    pub(crate) shadow: temen_ir::durable_abi::ShadowArena,
    /// #2164 — the thunk a demand-paged child's page checks miss into: set exactly when its instance
    /// has an access table (`addrs.pages`), which the code then tests before each access.
    pub(crate) page_miss: Option<PageMissThunk>,
}

/// Compile a §14 child: every function confined to its own window — `mapped_log2` committed bytes
/// inside a `reserved_log2` reservation (#1287, the detached child — a root's shape) — its
/// `call.cap`s calling its thunk, and the entry wrapped in a buffer-ABI trampoline. "Nesting cost
/// is paid at setup" (§14): this is the setup. The confinement mask is the reservation, the thunks
/// see `(mapped, reserved)` so the child's `vm_map` commits tail pages through the same path a root
/// run's does, and an access past the committed extent faults on the inaccessible tail.
/// `mapped_log2 == reserved_log2` is a fully-mapped window.
///
/// The result is a [`CompiledModule`] — **the same shape a root compiles to** (#1296: a domain is a
/// domain however it came into being). What differs is only what its code is compiled for: the
/// `call.cap` thunk its link binds (its own), its window geometry, and no setjmp runtime of its own
/// (those ops are rejected below). Its powerbox, the cells it polls and charges, the domain it
/// waits in and the nursery it spawns through are its instance's, in its vmctx. A child that runs
/// as an executor task may use `cont.*` fibers, `gc.roots` and `thread.spawn`/`join` (#1469; a
/// durable one since #2010): the thunks are static and find the running fiber runtime and thread
/// domain through thread-locals, and each task brings its own (`child_exec`). So the code is shared
/// by every spawn of it ([`ChildCodeCache`], temen#2168). Being the root's shape is what lets a
/// child hold a re-granted `Jit` capability: `define_extra` / `invoke_extra` / `install` lower a
/// submitted unit as this compile lowered the child's code, into the child's own table — so such a
/// child's code is its own.
///
/// `kept` is where its code is kept and under what key ([`ChildCodeKey::kept`]): the code kept
/// there is defined in place of compiling, or else the compile's is kept (temen#2452).
fn compile_child_windowed(
    c: &ChildCompile<'_>,
    kept: Option<(&CodeDir, [u8; 32])>,
) -> Result<CompiledModule, JitError> {
    let ChildCompile {
        funcs,
        types,
        entry: child_entry,
        mapped_log2,
        reserved_log2,
        thunk: cap_thunk,
        addrs,
        impl_exports,
        table_reserve_log2,
        shadow,
        page_miss,
    } = *c;
    let epoch_addr = addrs.epoch as usize;
    let fuel_addr = addrs.fuel as usize;
    let futex_sched = addrs.sched as usize;
    #[cfg(fiber_rt)]
    let nesting = !addrs.nursery.is_null();
    #[cfg(not(fiber_rt))]
    let nesting = false;
    // Audit #3: reject an oversize child window explicitly rather than silently clamping with
    // `.min(MAX_JIT_WINDOW_LOG2)`, so the window built here always equals the size the Instantiator
    // *validated* (which requires `child ≤ parent ≤ 2^MAX`, so this is unreachable in practice — but
    // it keeps the invariant local instead of relying on the cross-module parent-size cap). The
    // reservation may exceed the committed cap (a root's lazy reservation), never the other way, and
    // is capped as a root's is: a thawed child keeps the reservation its artifact records.
    if mapped_log2 > MAX_JIT_WINDOW_LOG2
        || reserved_log2 < mapped_log2
        || reserved_log2 > MAX_JIT_RESERVED_LOG2
    {
        return Err(JitError::Unsupported(
            "child window exceeds the reference JIT's max",
        ));
    }
    let entry = funcs
        .get(child_entry as usize)
        .ok_or(JitError::Malformed)?
        .clone();
    for f in funcs {
        ensure_supported(f, types)?;
        // `thread.spawn`/`join` need the task's own domain, and its parent's to share the futex
        // with; without them they would compile against null thread thunks — reject. Likewise
        // `cont.*` with no domain for `cont.resume.block` to bake.
        if f.uses_threads() && !(cfg!(fiber_rt) && futex_sched != 0) {
            return Err(JitError::Unsupported(
                "a §14 JIT child using thread.spawn/join needs a task and its parent's domain",
            ));
        }
        if !(cfg!(fiber_rt) && futex_sched != 0) && f.uses_fibers_or_threads() {
            return Err(JitError::Unsupported(
                "a §14 JIT child using fibers needs a task and its parent's domain",
            ));
        }
        // `atomic.wait`/`notify` compile against the parent domain's shared futex when one was
        // supplied; without it (coroutine / durable children) they would bake a null sched — reject.
        if futex_sched == 0 && f.uses_futex() {
            return Err(JitError::Unsupported(
                "a §14 JIT child using atomic.wait/notify needs the parent's futex domain",
            ));
        }
        // Likewise `setjmp`/`longjmp`: a child has no `setjmp` table yet (its vmctx's is null), so
        // reject rather than run a `SetJmp` site over none.
        if f.uses_setjmp() {
            return Err(JitError::Unsupported(
                "a §14 JIT child using setjmp/longjmp is not supported yet",
            ));
        }
    }
    let mapped = 1u64 << mapped_log2; // bounded ≤ MAX by the reject above (audit #3)
    let reserved = 1u64 << reserved_log2;
    let mask = reserved - 1;
    // #2164 — a demand-paged child checks its instance's access table before each access.
    let page = match (addrs.pages.is_null(), page_miss) {
        (true, _) => PageEnv::NONE,
        (false, Some(_)) => PageEnv::paged(mapped),
        (false, None) => {
            return Err(JitError::Unsupported(
                "a demand-paged child needs a page-miss thunk",
            ))
        }
    };
    // The `call.dyn` table, power-of-two padded and — like a root's (`CompiledModule::compile`) —
    // reserved larger than the module needs when the child holds a `Jit` grant, so `install` fills
    // padding slots without moving the mask constant baked into every call site.
    let table_len = (1usize << table_reserve_log2)
        .max(funcs.len().next_power_of_two())
        .max(1);
    let fn_table_mask = (table_len as u64) - 1;

    let imports = Imports {
        cap_thunk: cap_thunk as *const u8,
        page_miss,
        ..Imports::NONE
    };
    let mut module = new_jit_module(imports)?;
    let ids: Vec<FuncId> = declare_all_funcs(&mut module, funcs)?;
    // temen#2452 — the code kept for this compile, defined in place of compiling, or else a record
    // of the code it emits, to keep.
    let replay = kept
        .and_then(|(dir, key)| dir.load(&key))
        .filter(|code| code.fits(&module, &ids, &imports));
    let mut codegen = match &replay {
        Some(kept) => Codegen::Replay(kept, &imports),
        None => Codegen::Compile(kept.map(|_| EmittedCode::default())),
    };
    // Function signatures first, then every call-site signature (as the root does), so a unit a
    // `Jit`-holding child later submits interns against stable ids — id-equality ≡ structural
    // equality across the child's program and its units (DESIGN.md §22).
    let mut distinct = distinct_types(funcs);
    intern_unit_sigs(&mut distinct, funcs, types)?;
    // A task child using `cont.*` or `gc.roots` gets the fiber thunks, as a root does
    // (`compile_with_signal`); the fiber-entry signature is interned the same way.
    let uses_fibers = funcs.iter().any(|f| f.uses_fibers() || f.uses_gc_roots());
    // Thread ops passed the check above only in a task that can host them.
    let uses_threads = funcs.iter().any(|f| f.uses_threads());
    #[cfg(fiber_rt)]
    if uses_fibers {
        intern_type(&mut distinct, &fiber_func_type())?;
    }
    let distinct = distinct;
    // The **parent's** futex domain: wait/notify rendezvous there, and a task's
    // `thread.spawn`/`join` act in its own domain, which the thunks look up (`CURRENT_DOMAIN`) — so
    // the child spawns only when it uses them. Neither when no domain was supplied.
    let threads = futex_sched != 0;
    let spawns = threads && uses_threads;
    let base_bytes = define_funcs(
        &mut module,
        &ids,
        &mut codegen,
        |module, ctx, fi| {
            build_clif(
                module,
                &ids,
                funcs,
                &distinct,
                types,
                None,    // nested child: `call.cap`s go to the coroutine thunk, not a fast path
                nesting, // a child that spawns loads its own nursery from its vmctx (temen#2168)
                false,   // #932 — a §14 child is not signal-delivery armed (slice 1b)
                page,    // #2164 — its access table's check, when it is demand-paged
                &mut ctx.func,
                &funcs[fi],
                mask,
                mapped,                    // the committed prefix
                guard_offset_of(reserved), // its own window's trailing guard
                epoch_addr != 0, // §5 kill-path: the child polls the parent's interrupt cell
                fuel_addr != 0, // counted fuel: the child charges its own budget cell (else un-metered)
                true,           // a §14 child always runs beside its parent, which may tear it down
                fn_table_mask,  // the child's own (reserved) table mask
                None,           // §14 child: own window/table, `ref.func N` = slot N (no remap)
                0,
                None, // nested-child units carry no source-loc map (W5 JIT/DWARF)
                None, // …nor value-label points (Stage 3a)
            )
        },
        |_, _, _| {},
    )?;
    let mut ctx = module.make_context();
    build_trampoline(
        &mut module,
        &mut ctx.func,
        ids[child_entry as usize],
        &entry,
        false,
    )?;
    let tramp = module
        .declare_function("child_trampoline", Linkage::Export, &ctx.func.signature)
        .map_err(|e| JitError::Backend(e.to_string()))?;
    define_verified(&mut module, tramp, &mut ctx)?;
    module.clear_context(&mut ctx);
    // D66 — the same entry through the limit-taking variant, for a child-domain **task** (a detached
    // child resumed on a platform fiber by the executor, `child_exec`): its body passes the fiber's
    // stack low bound so the guest's prologue checks guard the fiber stack, not an inert `0`.
    build_trampoline(
        &mut module,
        &mut ctx.func,
        ids[child_entry as usize],
        &entry,
        true,
    )?;
    let tramp_limited = module
        .declare_function(
            "child_trampoline_limited",
            Linkage::Export,
            &ctx.func.signature,
        )
        .map_err(|e| JitError::Backend(e.to_string()))?;
    define_verified(&mut module, tramp_limited, &mut ctx)?;
    module.clear_context(&mut ctx);
    // CALLS.md 5c.1a — a serve trampoline per impl-export handler (the `CompiledModule::compile`
    // serve_ids block, child twin), so the locked thunk's child serve arm can invoke a handler of
    // any signature over the child's live window.
    let mut serve_ids: Vec<(u32, cranelift_module::FuncId)> = Vec::new();
    {
        let mut seen = std::collections::BTreeSet::new();
        for &f in impl_exports.iter().flat_map(|e| &e.ops) {
            if (f as usize) < funcs.len() && seen.insert(f) {
                build_trampoline(
                    &mut module,
                    &mut ctx.func,
                    ids[f as usize],
                    &funcs[f as usize],
                    false,
                )?;
                let id = module
                    .declare_function(
                        &format!("child_serve_tramp_{f}"),
                        Linkage::Export,
                        &ctx.func.signature,
                    )
                    .map_err(|e| JitError::Backend(e.to_string()))?;
                define_verified(&mut module, id, &mut ctx)?;
                module.clear_context(&mut ctx);
                serve_ids.push((f, id));
            }
        }
    }
    // The call-trampoline (a fiber's first resume, and a spawned vCPU's entry, call through it), as
    // a root's.
    #[cfg(fiber_rt)]
    let fiber_tramp = if uses_fibers || uses_threads {
        build_fiber_call_trampoline(&mut module, &mut ctx.func);
        let id = module
            .declare_function("fiber_call_tramp", Linkage::Export, &ctx.func.signature)
            .map_err(|e| JitError::Backend(e.to_string()))?;
        define_verified(&mut module, id, &mut ctx)?;
        module.clear_context(&mut ctx);
        Some(id)
    } else {
        None
    };
    module
        .finalize_definitions()
        .map_err(|e| JitError::Backend(e.to_string()))?;
    // What a task's own fiber runtime needs (`child_exec`): the trampoline, and the fiber-entry
    // type id + the funcref mask (the reserved table's, as a root's `fiber_mask`).
    #[cfg(fiber_rt)]
    let (call_tramp, fiber_cfg): (Option<fiber_rt::FiberCallTramp>, Option<(u32, u64)>) =
        match fiber_tramp {
            Some(id) => {
                let addr = module.get_finalized_function(id);
                // SAFETY: `addr` is the finalized `fiber_call_tramp` with exactly this ABI.
                let t: fiber_rt::FiberCallTramp = unsafe { std::mem::transmute(addr) };
                // A pure-thread child has the trampoline but no fibers, as a pure-thread root.
                (
                    Some(t),
                    uses_fibers.then(|| (type_id_of(&distinct, &fiber_func_type()), fn_table_mask)),
                )
            }
            None => (None, None),
        };

    let fn_table: Box<[FnEntry]> = (0..table_len)
        .map(|slot| match funcs.get(slot) {
            Some(f) => FnEntry::new(
                type_id_of(
                    &distinct,
                    &FuncType {
                        params: f.params.clone(),
                        results: f.results.clone(),
                    },
                ),
                module.get_finalized_function(ids[slot]) as u64,
            ),
            None => FnEntry::padding(),
        })
        .collect();

    let code = module.get_finalized_function(tramp);
    let code_limited = module.get_finalized_function(tramp_limited);
    let serve_tramps: Vec<(u32, *const u8, usize, usize)> = serve_ids
        .iter()
        .map(|&(f, id)| {
            let fu = &funcs[f as usize];
            (
                f,
                module.get_finalized_function(id),
                fu.params.len(),
                fu.results.len(),
            )
        })
        .collect();
    if let (Some((dir, key)), Codegen::Compile(Some(emitted))) = (kept, &codegen) {
        dir.keep(&key, emitted);
    }
    // PROCESS.md S1: a child module was actually JIT-compiled. Counting successful compiles is what
    // lets a test prove a child ran compiled; code a `CodeDir` kept is defined, not compiled.
    if replay.is_none() {
        CHILD_COMPILES.fetch_add(1, Ordering::Relaxed);
    }
    Ok(CompiledModule {
        // A §14 child's runtimes are its task's (`child_exec`) and its nursery is its instance's
        // (temen#2168), so its code is only code: every spawn of it can run one compile
        // ([`ChildCodeCache`]).
        hosted: false,
        nests: nesting && funcs_use_instantiator(funcs),
        page,
        fn_table,
        tramp_code: code,
        tramp_code_limited: code_limited,
        serve_tramps,
        n_params: entry.params.len(),
        n_results: entry.results.len(),
        n_real_funcs: funcs.len(),
        distinct,
        fast_resolver: None,
        fibers: cfg!(fiber_rt) && uses_fibers,
        threads,
        spawns,
        nesting,
        mask,
        epoch: epoch_addr != 0,
        fuel: fuel_addr != 0,
        // The child's own instance: its powerbox, the parent's kill-path cell, its own budget cell,
        // the parent's domain its `thread.*` and futex sites run on, and its own nursery. §14
        // children are not signal-delivery armed.
        instance: addrs,
        fn_table_mask,
        next_extra: 0,
        extra_bytes: 0,
        base_bytes,
        live_fault_range: None,
        caller_window: true,
        last_trap_backtrace: Vec::new(),
        last_trap_fiber: None,
        win_mapped: mapped as usize,
        win_reserved: reserved as usize,
        null_guard: temen_ir::module_null_guard(),
        mem_size_log2: Some(mapped_log2),
        data: Vec::new(), // a child's data segments are materialized into its window by the spawner
        restore: None,
        tail_out: Vec::new(),
        durable: false,
        concurrent_durable: false,
        frozen_seed: Vec::new(),
        frozen_out: Vec::new(),
        frozen_vcpus_out: Vec::new(),
        frozen_root_sp_out: 0,
        frozen_vcpu_seed: Vec::new(),
        detached_out: Vec::new(),
        detached_seed: Vec::new(),
        thaw_root_sp: shadow.frame_base(0),
        shadow,
        freeze_ctl: None,
        held_tail: None,
        window_ready: None,
        fork_hook: None,
        fiber_rt: None,
        domain: None,
        _nursery: None,
        #[cfg(setjmp_rt)]
        _setjmp_rt: None,
        // A child's nursery is its spawn's: its instances never stand one up of their own.
        cap_thunk: INERT_CAP_THUNK,
        resolve_module: None,
        call_tramp,
        fiber_cfg,
        fiber_table: None,
        node: None,
        src_ranges: Vec::new(),
        src_files: Vec::new(),
        func_names: std::collections::HashMap::new(),
        var_locs: Vec::new(),
        debug_types: Vec::new(),
        module: Arc::new(OwnedJit::new(module)),
    })
}

/// temen#2168 — what a §14 child's code depends on: its module, by digest, and every other input of
/// its compile ([`ChildCompile`]). A spawn whose key is equal instantiates the cache's compile.
#[derive(PartialEq, Eq, Hash)]
struct ChildCodeKey {
    /// The module's content digest, from its grant: the same code however it was granted. Identity
    /// is structural (INVARIANTS #10). It names the functions, types, impl exports and shadow arena
    /// the compile read.
    digest: [u8; 32],
    entry: FuncIdx,
    mapped_log2: u8,
    reserved_log2: u8,
    thunk: usize,
    /// Which cells the code's instances have, which decides the checks it emits
    /// ([`ChildCompile::addrs`]): the kill path, fuel, the futex domain and the nursery.
    cells: [bool; 4],
    table_reserve_log2: u8,
    /// The miss thunk a paged compile calls, or `0`: a paged and an unpaged spawn of one module run
    /// two compiles (#2164).
    page_miss: usize,
}

impl ChildCodeKey {
    /// The key a [`CodeDir`] keeps the code under: this one's but the `call.cap` and page-miss
    /// thunks, which only the code's link binds.
    fn kept(&self) -> [u8; 32] {
        code_dir::child_key(
            &self.digest,
            self.entry,
            self.mapped_log2,
            self.reserved_log2,
            self.cells,
            self.table_reserve_log2,
            self.page_miss != 0,
        )
    }

    fn of(digest: [u8; 32], c: &ChildCompile<'_>) -> ChildCodeKey {
        let a = &c.addrs;
        ChildCodeKey {
            digest,
            entry: c.entry,
            mapped_log2: c.mapped_log2,
            reserved_log2: c.reserved_log2,
            thunk: c.thunk as usize,
            cells: [
                !a.epoch.is_null(),
                !a.fuel.is_null(),
                !a.sched.is_null(),
                !a.nursery.is_null(),
            ],
            table_reserve_log2: c.table_reserve_log2,
            page_miss: c.page_miss.map_or(0, |t| t as usize),
        }
    }
}

/// One child's compile, made by its first spawn while later ones wait on the lock; `None` until then.
type ChildCodeSlot = Arc<std::sync::Mutex<Option<SharedCode>>>;

/// temen#2168 — compiled §14 children, by what their code depends on ([`ChildCodeKey`]): every
/// spawn of the same child runs one compile of it, each over its own powerbox, window, cells and
/// nursery ([`SharedCode::instance`]). A run's nurseries share one, made for the run unless the
/// embedder hands its own ([`CompiledModule::set_child_code`]), which then serves its later runs too.
/// The code lives as long as the cache: a run keeps a compile of each child it spawned, as the
/// bytecode engine keeps one of each module a run grants.
#[derive(Clone, Default)]
pub struct ChildCodeCache {
    slots: Arc<std::sync::Mutex<std::collections::HashMap<ChildCodeKey, ChildCodeSlot>>>,
    /// temen#2452 — where its children's code is kept, so a later process's first spawn of one
    /// defines the code kept instead of compiling it.
    dir: Option<CodeDir>,
}

impl ChildCodeCache {
    /// A cache that keeps its children's code in `dir` too ([`CodeDir`]).
    pub fn kept_in(dir: CodeDir) -> ChildCodeCache {
        ChildCodeCache {
            slots: Arc::default(),
            dir: Some(dir),
        }
    }

    /// How many children it holds a compile of. One being compiled right now is not counted (its
    /// slot is held for the compile, and this does not wait for it).
    pub fn compiled(&self) -> usize {
        self.slots
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|slot| slot.try_lock().is_ok_and(|code| code.is_some()))
            .count()
    }

    /// The code for a spawn of the child `c` describes, whose module `digest` names: an instance of
    /// the cache's compile of it. The first spawn makes the compile and runs it itself. Only for a
    /// child that cannot extend its code: one holding a `Jit` ([`GrantChild::holds_jit`]) has code
    /// of its own ([`Self::own`]).
    pub(crate) fn instance(
        &self,
        digest: [u8; 32],
        c: &ChildCompile<'_>,
    ) -> Result<CompiledModule, JitError> {
        let key = ChildCodeKey::of(digest, c);
        let disk = self.disk(&key);
        let slot = Arc::clone(
            self.slots
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .entry(key)
                .or_default(),
        );
        let mut kept = slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(code) = &*kept {
            // A child's runtimes are its task's: its instance stands none up, and charges no node.
            return Ok(code.instance(c.addrs, None));
        }
        let first = compile_child_windowed(c, disk)?;
        *kept = first.share();
        Ok(first)
    }

    /// The code for a spawn of a child that extends its code, one holding a `Jit`: a compile no
    /// other spawn shares, since the units it installs are its own. It starts from the code the
    /// cache's directory kept, if any, as [`Self::instance`]'s compile does (temen#2452).
    pub(crate) fn own(
        &self,
        digest: [u8; 32],
        c: &ChildCompile<'_>,
    ) -> Result<CompiledModule, JitError> {
        compile_child_windowed(c, self.disk(&ChildCodeKey::of(digest, c)))
    }

    /// Where the code for `key` is kept, and under what key, when the cache keeps code on disk.
    fn disk(&self, key: &ChildCodeKey) -> Option<(&CodeDir, [u8; 32])> {
        self.dir.as_ref().map(|dir| (dir, key.kept()))
    }
}

impl std::fmt::Debug for ChildCodeCache {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChildCodeCache")
            .field("compiled", &self.compiled())
            .finish()
    }
}

/// PROCESS.md S1: process-wide count of §14 child modules JIT-compiled — one per child and window
/// shape a run spawns ([`ChildCodeCache`]). Monotone; the public [`child_compiles`] reads it (a real
/// metric, and the tests' proof that a child ran compiled).
#[cfg(fiber_rt)]
pub(crate) static CHILD_COMPILES: AtomicU64 = AtomicU64::new(0);

/// #1825 — every whole-module compile ([`CompiledModule::compile`]); see [`module_compiles`].
static MODULE_COMPILES: AtomicU64 = AtomicU64::new(0);

/// #1825 — how many modules this process has JIT-compiled whole ([`CompiledModule::compile`]; a §14
/// child's compile counts in [`child_compiles`] instead). An instance of shared code
/// ([`SharedCode::instance`]) does **not** advance it, nor does a compile that defines code a
/// [`CodeDir`] kept: the metric is the observable of the process tree's compile-once.
pub fn module_compiles() -> u64 {
    MODULE_COMPILES.load(Ordering::Relaxed)
}

/// PROCESS.md S1: how many §14 child modules this process has JIT-compiled (0 where nesting is
/// unsupported — no child ever compiles): one per child a run spawns, not per spawn
/// ([`ChildCodeCache`]).
pub fn child_compiles() -> u64 {
    #[cfg(fiber_rt)]
    {
        CHILD_COMPILES.load(Ordering::Relaxed)
    }
    #[cfg(not(fiber_rt))]
    {
        0
    }
}

/// The natural CLIF signature for an IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. Both context pointers are threaded through every call so loads/
/// stores reach the window and `call.dyn` reaches the function table.
fn natural_sig(module: &mut JITModule, f: &Func) -> cranelift_codegen::ir::Signature {
    sig_from(module, &f.params, &f.results)
}

/// Max function results returned **in registers**. Above this the JIT spills results to a
/// caller-provided memory **return-area (sret) pointer** — like wasm engines do for multi-value —
/// so a many-result function compiles **uniformly on every target**. (Cranelift's `Tail` calling
/// convention caps register returns at a per-ABI budget: x86-64 fits 8, aarch64 fewer, so returning
/// the count in registers was the one place a *valid* module compiled on one supported target and was
/// rejected on another.) `4` keeps every real signature — including the §12 `(sp,arg)->i64` fiber/
/// thread entry and the multi-result test cases — on the fast register path, while being safely
/// within the tightest target's budget; only `>4`-result functions take the sret path. The decision
/// is by result **count**, a property of the function *type*, so it is identical at every call site —
/// direct, `call.dyn` (its type id pins the same choice), and tail calls.
const MAX_REG_RESULTS: usize = 4;

/// Whether a function with these results returns via the memory return-area pointer (sret) rather
/// than registers — see [`MAX_REG_RESULTS`].
fn uses_sret(results: &[ValType]) -> bool {
    results.len() > MAX_REG_RESULTS
}

/// The sret return-area uses 8-byte slots (`encode_slot`/`decode_slot`, the buffer-ABI encoding), so
/// a `v128` result cannot be carried through it. A `>4`-result signature containing a `v128` is
/// therefore rejected uniformly (`Unsupported`) rather than miscompiled — an exotic non-case (`v128`
/// buffer slots are already out of MVP scope, §17), and the interpreter still covers it.
fn sret_blocked_by_v128(results: &[ValType]) -> bool {
    uses_sret(results) && results.iter().any(|t| matches!(t, ValType::V128))
}

/// The natural signature for an explicit param/result list (shared by `natural_sig`
/// and the `call.dyn` signature import).
fn sig_from(
    module: &mut JITModule,
    params: &[ValType],
    results: &[ValType],
) -> cranelift_codegen::ir::Signature {
    let mut sig = module.make_signature();
    // The `tail` calling convention so `return_call` (guaranteed tail calls, §3b) is
    // available; a normal `call` from the trampoline works against it too.
    sig.call_conv = cranelift_codegen::isa::CallConv::Tail;
    sig.params.push(AbiParam::new(I64)); // mem_base
    sig.params.push(AbiParam::new(I64)); // fn_table_base
    sig.params.push(AbiParam::new(I64)); // vmctx (the instance's context; field 0 its trap cell)
                                         // §2b path B: a 4th context param — the running stack's limit: a fiber's `usable_low`, or the
                                         // root/thread-top's thread limit (`stack_check::thread_limit`, #1983). Threaded through every call (constant within
                                         // a stack's call tree; set anew at each fiber/root entry), so the prologue check reads a per-vCPU
                                         // limit from a register with no cell and no TLS. Always present (the software stack-overflow guard
                                         // is in the always-on escape-TCB path; see `emit_stack_check`).
    sig.params.push(AbiParam::new(I64)); // stack_limit
    let sret = uses_sret(results);
    if sret {
        // The return-area pointer: the callee writes its results here (8-byte slots) instead of
        // returning them in registers, so the result count is target-independent. Placed right
        // after the context pointers, before the user params — the order every call site mirrors.
        sig.params.push(AbiParam::new(I64));
    }
    for p in params {
        sig.params.push(AbiParam::new(clif_ty(*p)));
    }
    if !sret {
        for r in results {
            sig.returns.push(AbiParam::new(clif_ty(*r)));
        }
    }
    sig
}

/// Reject functions using any op outside the integer slice, so `build_clif` can lower
/// the remainder totally. Keeping the check separate keeps the lowering readable.
fn ensure_supported(f: &Func, type_section: &[temen_ir::TypeEntry]) -> Result<(), JitError> {
    // The sret return-area (used for `>MAX_REG_RESULTS` results) carries 8-byte slots, so a
    // many-result signature containing a `v128` can't pass through it — reject uniformly on every
    // target (the interpreter still covers it). This function's own results + any indirect
    // call/tail-call target type; direct callees are checked as their own definitions.
    if sret_blocked_by_v128(&f.results) {
        return Err(JitError::Unsupported("v128 in a many-result signature"));
    }
    for blk in &f.blocks {
        if let Terminator::ReturnCallIndirect { ty, .. } = &blk.term {
            if sret_blocked_by_v128(&sig_of(type_section, *ty).results) {
                return Err(JitError::Unsupported("v128 in a many-result signature"));
            }
        }
        for inst in &blk.insts {
            if let Inst::CallIndirect { ty, .. } = inst {
                if sret_blocked_by_v128(&sig_of(type_section, *ty).results) {
                    return Err(JitError::Unsupported("v128 in a many-result signature"));
                }
            }
        }
    }
    for blk in &f.blocks {
        for inst in &blk.insts {
            match inst {
                Inst::ConstI32(_)
                | Inst::ConstI64(_)
                | Inst::ConstF32(_)
                | Inst::ConstF64(_)
                | Inst::Select { .. }
                | Inst::IntCmp { .. }
                | Inst::Eqz { .. }
                | Inst::FBin { .. }
                | Inst::FUn { .. }
                | Inst::Fma { .. }
                | Inst::FCmp { .. }
                | Inst::FToISat { .. }
                | Inst::FToITrap { .. }
                | Inst::IToFConv { .. }
                | Inst::Cast { .. }
                | Inst::Load { .. }
                | Inst::Store { .. }
                | Inst::MemCopy { .. }
                | Inst::MemMove { .. }
                | Inst::MemFill { .. }
                | Inst::AtomicLoad { .. }
                | Inst::AtomicStore { .. }
                | Inst::AtomicRmw { .. }
                | Inst::AtomicCmpxchg { .. }
                | Inst::AtomicFence { .. }
                | Inst::Call { .. }
                | Inst::CallIndirect { .. }
                | Inst::CapCall { .. }
                | Inst::CallImport { .. }
                | Inst::CallImportDyn { .. }
                | Inst::CallSym { .. }
                | Inst::ExportHandle { .. }
                | Inst::CapSelfTypeId { .. }
                | Inst::CapSelfCovers { .. }
                | Inst::ImportAttach { .. }
                | Inst::RefFunc { .. }
                | Inst::IntBin { .. }
                | Inst::Convert { .. } => {}
                Inst::IntUn { .. } => {}
                // §17 SIMD (D58): all lowered via Cranelift's native vector ops.
                Inst::ConstV128(_)
                | Inst::V128Load { .. }
                | Inst::V128Store { .. }
                | Inst::Splat { .. }
                | Inst::ExtractLane { .. }
                | Inst::ReplaceLane { .. }
                | Inst::VFloatBin { .. }
                | Inst::VFloatUn { .. }
                | Inst::VBitBin { .. }
                | Inst::VNot { .. }
                | Inst::Bitselect { .. }
                | Inst::Shuffle { .. }
                | Inst::Swizzle { .. } => {}
                // §12 per-vCPU TLS register: a thunk over a thread-local — substrate-independent
                // (works for a plain non-fiber root), so supported on every target.
                Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } => {}
                // §12.8 4A.5 durable-runtime-internal: a thunk over a per-OS-thread word (like
                // the TLS register), so supported on every target.
                Inst::DurableShadowBase => {}
                // Every `VIntBin` lane op is lowered: `i8x16.mul` via widen → `i16x8` multiply →
                // low-byte pack, and `i64x2` min/max via per-lane compare + `bitselect` (x86/aarch64
                // have no native `i64` vector min/max) — see the `VIntBin` lowering.
                Inst::VIntBin { .. } => {}
                // Lane compares lower to a single Cranelift `icmp`/`fcmp` (legalize on every target).
                Inst::VIntCmp { .. } | Inst::VFloatCmp { .. } => {}
                // Lane shifts lower to vector `ishl`/`ushr`/`sshr`; Cranelift legalizes every shape
                // (incl. `i8x16`, which has no native per-byte shift on x86 — it emits a sequence).
                Inst::VShift { .. } => {}
                // Lane `abs`/`neg` lower to vector `iabs`/`ineg`; Cranelift legalizes every shape.
                Inst::VIntUn { .. } => {}
                // `i8x16.popcnt` lowers to a vector `popcnt` (native `cnt` on aarch64, a byte
                // shuffle sequence on x86 — Cranelift legalizes both).
                Inst::VPopcnt { .. } => {}
                // `avgr_u` (`i8x16`/`i16x8` only, verifier-enforced) → native `avg_round`.
                Inst::VAvgr { .. } => {}
                // `i32x4.dot_i16x8_s` / `i16x8.dot_i8x16_s` → `swiden_low/high` + `imul` +
                // `iadd_pairwise` (all legalize).
                Inst::VDot { .. } | Inst::VDotI8 { .. } => {}
                // Extended multiply → widen low/high both operands + `imul` on the wide shape.
                // `imul` legalizes for every wide shape (incl. `i64x2`, unlike `i8x16.mul`).
                Inst::VExtMul { .. } => {}
                // Extended pairwise add → `swiden/uwiden` low+high + `iadd_pairwise` (all legalize).
                Inst::VExtAddPairwise { .. } => {}
                // Q15 rounding multiply → native `sqmul_round_sat`.
                Inst::VQ15MulrSat { .. } => {}
                // Fused multiply-add (relaxed_madd/nmadd) → vector `fma` (one rounding; the same
                // correctly-rounded result as the interp's `libm` `fma`, so the differential holds).
                Inst::VFma { .. } => {}
                // Boolean reductions → a scalar `i32` (`vany_true`/`vall_true`/`vhigh_bits`).
                Inst::VAnyTrue { .. } | Inst::VAllTrue { .. } | Inst::VBitmask { .. } => {}
                // Saturating add/sub (`i8x16`/`i16x8` only, verifier-enforced) lower to native
                // `sadd_sat`/`uadd_sat`/`ssub_sat`/`usub_sat`.
                Inst::VSatBin { .. } => {}
                // Widen lowers to `swiden_low`/`uwiden_low`/`*_high`; narrow to `snarrow`/`unarrow`.
                Inst::VWiden { .. } | Inst::VNarrow { .. } => {}
                // Int↔float / float↔float conversions → Cranelift `fcvt_*` / `fvdemote`/`fvpromote_low`.
                Inst::VConvert { .. } => {}
                // pmin/pmax lower to a single `fcmp` plus `bitselect` (both legalize on every target).
                Inst::VPMinMax { .. } => {}
                // §12 fibers/threads: lowered to host runtime calls, but only where the stack-switch
                // substrate exists (`temen_fiber::supported()` — x86-64 unix). Elsewhere, bail so the
                // differential harness skips rather than miscompiles.
                // I48: the `block: true` form lowers to a resume thunk that parks the OS thread on
                // the domain condvar (see below); `block: false` returns FIBER_PARKED and the guest
                // loops. Both accepted wherever `cont.resume` is.
                Inst::ContNew { .. }
                | Inst::ContResume { .. }
                | Inst::Suspend { .. }
                | Inst::ThreadSpawn { .. }
                | Inst::ThreadJoin { .. }
                | Inst::MemoryWait { .. }
                | Inst::MemoryNotify { .. }
                // §GC `gc.roots`: scans the live fiber stacks via the fiber runtime — supported only
                // where the stack-switch substrate exists (else it bails like the other fiber ops and
                // the interpreter covers it).
                | Inst::GcRoots { .. }
                    if cfg!(fiber_rt) => {}
                // `setjmp`/`longjmp` (LLVM.md §"JIT `longjmp`", Option B): libc `_setjmp`/`_longjmp`
                // called inline from JITted code with a host-side `jmp_buf` table. Supported on the
                // `setjmp_rt` targets (unix among `fiber_rt`); elsewhere bail so the interpreters cover
                // it (module-granular fallback).
                Inst::SetJmp { .. } | Inst::LongJmp { .. } if cfg!(setjmp_rt) => {}
                _ => return Err(JitError::Unsupported("instruction")),
            }
        }
        match &blk.term {
            Terminator::Br { .. }
            | Terminator::BrIf { .. }
            | Terminator::BrTable { .. }
            | Terminator::Return(_)
            | Terminator::ReturnCall { .. }
            | Terminator::ReturnCallIndirect { .. }
            | Terminator::Unreachable => {}
        }
    }
    Ok(())
}

/// #932 — the JIT-side ABI of the async-signal delivery **decision** thunk a host supplies to
/// [`compile_and_run_with_host_signals`]: given the delivery ctx and a 3-slot `out` buffer, gate on
/// the nesting bound + consume one deliverable signal, writing `(handler_funcref, signum, handler_sp)`
/// into `out[0..3]` and returning `1` (delivered) or `0` (nothing / bound reached). Mirrors the
/// interpreter's per-op redirect — the host wires it to `temen_interp::SignalSource::take_deliverable`.
pub type SigTakeThunk = unsafe extern "C" fn(ctx: *mut core::ffi::c_void, out: *mut i64) -> i32;
/// #932 — the JIT-side ABI of the async-signal **handler-return** thunk: fire after the guest handler
/// the redirect dispatched returns, so the host restores the block-during-handler mask
/// (`SignalSource::handler_returned`) and pops the nesting count.
pub type SigReturnThunk = unsafe extern "C" fn(ctx: *mut core::ffi::c_void);

/// #932 — the host's async-signal delivery **arm** for a JIT run: the `Host::sig_armed` flag the
/// lowering polls at its safepoints (function entries + taken back-edges), plus the two call-out
/// thunks and their ctx. Passed to [`compile_and_run_with_host_signals`]. The compiled code calls
/// the thunks, and its instance holds the flag and the ctx, so all must stay valid for the whole
/// run. See `temen_run::JitSigDelivery` for the host side.
pub struct SignalArm {
    /// The `AtomicBool` "a signal is armed" flag (`Host::sig_armed`) the lowering polls inline.
    pub armed: *const core::sync::atomic::AtomicBool,
    /// The delivery-decision thunk (gate + consume). See [`SigTakeThunk`].
    pub take: SigTakeThunk,
    /// The handler-return thunk (restore mask + pop nesting). See [`SigReturnThunk`].
    pub ret: SigReturnThunk,
    /// The opaque host delivery context passed to both thunks (a `*mut temen_run::JitSigDelivery`).
    pub ctx: *mut core::ffi::c_void,
}

/// #2164 — a demand-paged child's page-check shape, baked into [`emit_page_check`]: the host page's
/// `log2`, and how many pages its declared window has, which its access table covers. The table is
/// per-instance, read from its [`VmCtx`], and a miss calls [`HostFn::PageMiss`], which the link
/// binds. Unpaged ⇒ no check is emitted, so the code is the unpaged code.
#[derive(Clone, Copy)]
struct PageEnv {
    paged: bool,
    shift: u8,
    pages: u64,
}

impl PageEnv {
    const NONE: PageEnv = PageEnv {
        paged: false,
        shift: 0,
        pages: 0,
    };

    /// The shape for a child whose declared window is `mapped` bytes.
    fn paged(mapped: u64) -> PageEnv {
        let page = mem::page_size() as u64;
        PageEnv {
            paged: true,
            shift: page.trailing_zeros() as u8,
            pages: mapped.div_ceil(page),
        }
    }

    fn is_paged(&self) -> bool {
        self.paged
    }
}

/// Per-function lowering context shared across blocks.
struct Lower<'a> {
    /// Holds `mem_base` (the window base) for load/store lowering and call threading.
    mem_var: Variable,
    /// Holds `fn_table_base` for `call.dyn` dispatch and call threading.
    fn_table_var: Variable,
    /// Holds the instance's [`VmCtx`] pointer — which is also its trap cell (field 0), the host-owned
    /// `*mut i64` a trap (or the cap thunk) writes before returning (the host reads it to learn the
    /// run trapped, §5). Every other per-instance address is loaded from it ([`vmctx_load`]).
    vmctx_var: Variable,
    /// §2b path B: holds `stack_limit` (the running stack's `usable_low`, or 0 for the root), for the
    /// prologue [`emit_stack_check`] and to thread on to callees. Always present (the guard is in the
    /// always-on escape-TCB path).
    limit_var: Variable,
    /// This function's result CLIF types, so a trapping path can `return` dummy zeros.
    result_tys: Vec<Type>,
    /// The §4 confinement mask (`reserved - 1`); `0` when the module has no memory.
    mask: u64,
    /// The backed `mapped` extent in bytes — the guest window length handed to the `call.cap`
    /// thunk (`[mem_base, mem_base+mapped)`), so buffer borrows and Memory-cap ops bound against
    /// the *backed* region, not the larger reserved mask domain. `0` when the module has no memory.
    mapped: u64,
    /// Offset from `mem_base` of the **window's trailing guard page** — `round_up(win_reserved,
    /// page)`. Always `PROT_NONE`, so the D63 branchless lowering redirects an out-of-bounds access
    /// here to fault it. `0` when the module has no memory (the branchless path isn't taken).
    guard_offset: u64,
    /// The target's frontend config (pointer width + default call conv), for the Cranelift
    /// `call_memcpy`/`call_memmove`/`call_memset` helpers that lower the bulk-memory ops (D62).
    frontend_config: cranelift_codegen::isa::TargetFrontendConfig,
    /// The function-table index mask (`next_pow2(nfuncs) - 1`) for `call.dyn`.
    fn_table_mask: u64,
    /// **Submitted-unit `ref.func` remap** (DESIGN.md §22 "unit-own funcref"). For a top-level
    /// compile this is `None` and `ref.func N` lowers to the bare index `N` (module-0 slot N). For a
    /// guest-submitted unit that takes its own functions' addresses, `define_extra` auto-installs the
    /// unit's functions into reserved `call.dyn` slots and passes the map here (indexed by the
    /// unit-local function index), so `ref.func N` lowers to `iconst(ref_slots[N])` — the real shared
    /// table slot — instead of `N`, which would resolve against the *parent's* table. This is what
    /// lets a unit `cont.new`/`thread.spawn`/`call.dyn` over its OWN function; it changes only
    /// *which constant* `ref.func` emits, never the masked-dispatch lowering (invariant I2).
    ref_slots: Option<&'a [u32]>,
    /// The optional D45 devirtualize-to-direct-call resolver (top-level compile only; `None` for
    /// nested children, whose `call.cap`s go to the coroutine thunk). Invoked at compile time.
    fast_resolver: Option<FastCapResolver>,
    /// Lower `Instantiator` `call.cap`s to the §14 nesting runtime (`false` ⇒ they take the
    /// ordinary `call.cap` path — an inert `CapFault`).
    nesting: bool,
    /// #932 — emit the async-signal delivery check ([`emit_signal_check`]) at safepoints (`false` ⇒
    /// no signal delivery armed — the check is not emitted, guest code byte-identical to the
    /// un-armed build).
    signals: bool,
    /// Whether this compile polls the host-owned **interrupt cell** (`AtomicU64`, the instance's
    /// [`VmCtx::epoch`]) for the §5 fuel/epoch kill-path. `false` ⇒ no kill-path is armed (the checks
    /// are not emitted — guest code is byte-identical to the un-armed build). When set, the lowering
    /// polls the cell at loop back-edges and function entries and traps [`TrapKind::OutOfFuel`] if the
    /// host has set it non-zero, so a non-terminating guest is stopped. The guest cannot disable the
    /// poll — only the host (who chose to arm it) writes the cell.
    epoch: bool,
    /// Whether this compile charges the host-owned **counted-fuel cell** (`u64`, the instance's
    /// [`VmCtx::fuel`]) for safepoint-anchored fuel metering (INTERP_PERF.md "Fuel unification").
    /// `false` ⇒ no fuel budget is armed (no fuel checks are emitted — guest code is byte-identical to
    /// the un-armed build). When set, the lowering decrements the cell by one at every function entry,
    /// taken back-edge and `cont.resume` (the same IR safepoints the interpreters charge at; a fiber's
    /// entry is refunded, since the resume that started it paid — #1642) and traps
    /// [`TrapKind::OutOfFuel`] when it would go below zero — so `fuel` = (function entries + taken
    /// back-edges + resumes) executed, identical to the interpreters. Unlike [`Self::epoch`] this is
    /// the **guest's own budget**, not written by the host mid-run, so the load/decrement/store is
    /// plain (non-atomic); the store⇒load dependency on the same address keeps Cranelift from hoisting
    /// the check out of a loop.
    fuel: bool,
    /// DESIGN §12 domain teardown for **running** vCPUs: poll this vCPU's own trap cell at every loop
    /// back-edge (and the entry of a function that tail-calls), and unwind when it is non-zero — a sibling's trap, the root's
    /// completion sentinel ([`DOMAIN_DONE_CODE`]), or a parent's teardown of a §14 child. On for code
    /// that can run beside other vCPUs (a threaded module, every §14 child); a single-vCPU compile
    /// emits nothing.
    domain_poll: bool,
    /// Every function's `FuncId`, so `call`/`return_call` can reference callees.
    ids: &'a [FuncId],
    /// The functions of this compilation unit, indexed like [`Self::ids`], so a **direct** `call`
    /// can read its callee's result types to decide the sret ABI (see [`uses_sret`]). `call.dyn`
    /// uses its own carried type instead.
    funcs: &'a [Func],
    /// Distinct module signatures, for `call.dyn` type ids.
    distinct: &'a [FuncType],
    /// The unit's **type section** (#922): the call variants (`call.dyn`, `call.cap`, …)
    /// carry a `u32` index into this, which the lowering resolves to the `FuncType` it needs for
    /// the call signature and (via [`type_id_of`]/[`intern_type`]) the dispatch type id.
    type_section: &'a [temen_ir::TypeEntry],
    /// The current function's **return-area pointer** variable when it returns via sret
    /// ([`uses_sret`] of its results), else `None`. A `Return` stores results through it; a tail
    /// call forwards it (the tail callee shares the caller's result type, so its sret-ness matches).
    sret_var: Option<Variable>,
    /// `-g` source-loc lookup (W5 JIT/DWARF Stage 0): `(func, block, inst) → debug_info.locs` index.
    /// `None` ⇒ no debug info, so no `SourceLoc`s are stamped (codegen is byte-identical to before).
    srclocs: Option<&'a SrcLocMap>,
    /// This function's index, the `func` half of the [`Self::srclocs`] key.
    func_idx: u32,
    /// `-g` value-label points (W5 JIT/DWARF Stage 3a): `(func, block) → [(block-local value, label)]`
    /// — the source variables whose backing CLIF value `lower_block` stamps with a `ValueLabel`, so
    /// Cranelift records its machine-location ranges. `None` ⇒ no `-g` vars (no labels, codegen
    /// unchanged).
    var_labels: Option<&'a VarLabelMap>,
    /// §5 W3 Stage 2 explicit-trap backtrace: `Some(helper)` to call the trap-capture helper
    /// (`temen_capture_explicit_trap`, [`HostFn::TrapCapture`]) from every [`emit_trap`] site, so
    /// an explicit trap (div-by-zero, `unreachable`, `OutOfFuel`, indirect-call-type) records a
    /// source backtrace before it unwinds — the way memory faults do via the signal handler. `None`
    /// without `-g` (the backtrace would be empty) or where the helper isn't linked (a target with
    /// no trap runtime), leaving the trap path byte-identical to before.
    trap_capture: Option<FuncRef>,
    /// #2164 — a paged compile's page check ([`emit_page_check`]): its shape, the miss thunk
    /// ([`HostFn::PageMiss`]) imported into the function, and the variable holding the instance's
    /// access table, loaded once at entry. `None` for an unpaged compile.
    page: Option<(PageEnv, FuncRef, Variable)>,
}

/// The address of the §5 W3 explicit-trap capture helper (`temen_capture_explicit_trap` in
/// `trap_capture.c`), which each [`emit_trap`] site calls under `-g` ([`HostFn::TrapCapture`]). The
/// helper takes the trapping frame pointer as an argument (the JIT threads it in via Cranelift
/// `get_frame_pointer`), so it works on every target the shim compiles for — **unix and windows**
/// (MSVC has no `__builtin_frame_address`, but the passed `fp` + `_ReturnAddress` sidestep it).
/// `None` on a target with no trap runtime.
#[cfg(any(unix, windows))]
fn trap_capture_addr() -> Option<*const ()> {
    extern "C" {
        fn temen_capture_explicit_trap(fp: usize);
    }
    Some(temen_capture_explicit_trap as unsafe extern "C" fn(usize) as *const ())
}
#[cfg(not(any(unix, windows)))]
fn trap_capture_addr() -> Option<*const ()> {
    None
}

/// Build the natural-ABI CLIF for one IR function: `(mem_base, fn_table_base, params…)
/// -> (results…)`. The CLIF entry block holds the native params and jumps into IR
/// block 0 passing the parameters as its block args.
///
/// `fn_table_mask` is the `call.dyn` index mask — `next_pow2(table_len) - 1` for the
/// table this function will dispatch through. It is an explicit parameter (not derived from
/// `ids.len()`) because an **incrementally defined** function (`CompiledModule::define_extra`)
/// has its own unit-local `ids` for direct calls but dispatches through the *parent's*
/// function table, whose mask was fixed when the parent was compiled (the mask is baked as a
/// constant into every call site — DESIGN.md §22 "the baked function-table mask").
#[allow(clippy::too_many_arguments)]
fn build_clif(
    module: &mut JITModule,
    ids: &[FuncId],
    funcs: &[Func],
    distinct: &[FuncType],
    type_section: &[temen_ir::TypeEntry],
    fast_resolver: Option<FastCapResolver>,
    nesting: bool,
    signals: bool,
    page: PageEnv,
    clif: &mut Function,
    f: &Func,
    mask: u64,
    mapped: u64,
    guard_offset: u64,
    epoch: bool,
    fuel: bool,
    domain_poll: bool,
    fn_table_mask: u64,
    ref_slots: Option<&[u32]>,
    func_idx: u32,
    srclocs: Option<&SrcLocMap>,
    var_labels: Option<&VarLabelMap>,
) -> Result<(), JitError> {
    if f.blocks.is_empty() {
        return Err(JitError::Malformed);
    }
    clif.signature = natural_sig(module, f);
    clif.name = UserFuncName::user(0, 0);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);

    // One CLIF block per IR block, with params mirroring the IR block params, but those the
    // lowering takes as the values they forward (#2448). A separate CLIF entry block holds the
    // native params and jumps into IR block 0.
    let mut blocks = Blocks {
        clif: f.blocks.iter().map(|_| b.create_block()).collect(),
        fwd: f.forwards(),
        vals: vec![Vec::new(); f.blocks.len()],
    };
    for (i, blk) in f.blocks.iter().enumerate() {
        for (j, p) in blk.params.iter().enumerate() {
            if blocks.taken(i, j).is_none() {
                b.append_block_param(blocks.clif[i], clif_ty(*p));
            }
        }
    }
    let sret = uses_sret(&f.results);
    let entry = b.create_block();
    b.append_block_param(entry, I64); // mem_base
    b.append_block_param(entry, I64); // fn_table_base
    b.append_block_param(entry, I64); // vmctx (its field 0 is the trap cell)
    b.append_block_param(entry, I64); // stack_limit (§2b path B) — mirrors sig_from
    if sret {
        b.append_block_param(entry, I64); // return-area pointer (results spilled here, not returned)
    }
    for p in &f.params {
        b.append_block_param(entry, clif_ty(*p));
    }
    b.switch_to_block(entry);
    b.seal_block(entry);
    let mem_base = b.block_params(entry)[0];
    let fn_table_base = b.block_params(entry)[1];
    let vmctx = b.block_params(entry)[2];

    // The context pointers are needed across blocks; stash them in variables.
    let mem_var = b.declare_var(I64);
    b.def_var(mem_var, mem_base);
    let fn_table_var = b.declare_var(I64);
    b.def_var(fn_table_var, fn_table_base);
    let vmctx_var = b.declare_var(I64);
    b.def_var(vmctx_var, vmctx);
    // §2b path B: stash the stack-limit param (block index 3, right after the context pointers) so the
    // prologue check and every call can reach it.
    let limit_var = {
        let var = b.declare_var(I64);
        b.def_var(var, b.block_params(entry)[3]);
        var
    };
    // The return-area pointer (when sret) is likewise needed in the `Return`/tail-call blocks. Its
    // block index is after the context pointers + the stack-limit param (index 4).
    let sret_var = if sret {
        let var = b.declare_var(I64);
        b.def_var(var, b.block_params(entry)[4]);
        Some(var)
    } else {
        None
    };
    // §5 W3 Stage 2: under `-g`, import the trap-capture helper (its `(i64) -> ()` host-C signature
    // takes the trapping frame pointer), so `emit_trap` can record an explicit trap's source
    // backtrace before unwinding. Disabled without `-g` or where the helper isn't linked.
    let trap_capture = match trap_capture_addr() {
        Some(_) if srclocs.is_some() => Some(import_host(module, b.func, HostFn::TrapCapture)?),
        _ => None,
    };
    // #2164 — a paged compile loads its instance's access table once, here, and imports the miss
    // thunk ([`HostFn::PageMiss`]) once.
    let page = if page.is_paged() {
        let table = b.declare_var(I64);
        let t = b.ins().load(I64, MemFlags::trusted(), vmctx, vmctx::PAGES);
        b.def_var(table, t);
        Some((page, import_host(module, b.func, HostFn::PageMiss)?, table))
    } else {
        None
    };
    let lower = Lower {
        mem_var,
        fn_table_var,
        vmctx_var,
        limit_var,
        sret_var,
        funcs,
        result_tys: f.results.iter().map(|t| clif_ty(*t)).collect(),
        mask,
        mapped,
        guard_offset,
        frontend_config: module.target_config(),
        fn_table_mask,
        ref_slots,
        fast_resolver,
        nesting,
        signals,
        epoch,
        fuel,
        domain_poll,
        ids,
        distinct,
        type_section,
        srclocs,
        func_idx,
        var_labels,
        trap_capture,
        page,
    };

    // Jump into IR block 0 passing the function parameters (entry params after the three context
    // pointers, plus the sret pointer when present). A §5 kill-path check guards the *entry* (caught
    // before any work): this is what stops unbounded recursion and tail-call loops — each (re-)entry
    // polls the interrupt cell. Intra-function loops are caught by the per-back-edge check in
    // `lower_block`.
    let pbase = 4 + usize::from(sret); // 3 context ptrs + stack_limit, then sret
    let entry_args: Vec<BlockArg> = b.block_params(entry)[pbase..]
        .iter()
        .map(|v| BlockArg::from(*v))
        .collect();
    emit_epoch_check(&mut b, &lower);
    // A cycle of tail calls passes no back-edge and no call return, so a function that tail-calls
    // polls on entry; any other re-entry comes back through a caller's post-call trap guard.
    if f.blocks.iter().any(|blk| {
        matches!(
            blk.term,
            Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. }
        )
    }) {
        emit_domain_poll(&mut b, &lower);
    }
    emit_signal_check(module, &mut b, &lower)?; // #932 async signal delivery (no-op when disarmed)
    emit_fuel_check(&mut b, &lower);
    emit_stack_check(&mut b, &lower);
    b.ins().jump(blocks.clif[0], &entry_args);

    for (i, blk) in f.blocks.iter().enumerate() {
        lower_block(module, &mut b, blk, i, &mut blocks, &lower)?;
    }

    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

/// Build the fixed buffer-ABI trampoline `fn(args_ptr, results_ptr, mem_base,
/// fn_table_base, vmctx)` that decodes the entry function's args from `args_ptr`,
/// calls it (natural ABI), and stores its results to `results_ptr`. This is what Rust
/// calls, so any arity works.
///
/// `with_limit` (D66) appends a sixth `stack_limit` param and threads it into the entry's §2b
/// stack-limit context slot instead of the root's constant `0`: the variant a **child-domain task**
/// enters through, since its body runs on a platform fiber's control stack whose low bound only the
/// executor knows at resume time (the fiber's own `stack_low`), exactly as a guest fiber's entry does
/// via the fiber call-trampoline. One flag on the one builder, not a second trampoline family.
fn build_trampoline(
    module: &mut JITModule,
    clif: &mut Function,
    entry_id: FuncId,
    entry: &Func,
    with_limit: bool,
) -> Result<(), JitError> {
    clif.signature.params.push(AbiParam::new(I64)); // args_ptr
    clif.signature.params.push(AbiParam::new(I64)); // results_ptr
    clif.signature.params.push(AbiParam::new(I64)); // mem_base
    clif.signature.params.push(AbiParam::new(I64)); // fn_table_base
    clif.signature.params.push(AbiParam::new(I64)); // vmctx
    if with_limit {
        clif.signature.params.push(AbiParam::new(I64)); // stack_limit (D66 child-domain task)
    }
    clif.name = UserFuncName::user(0, 1);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    for _ in 0..(if with_limit { 6 } else { 5 }) {
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(blk);
    b.seal_block(blk);
    let args_ptr = b.block_params(blk)[0];
    let results_ptr = b.block_params(blk)[1];
    let mem_base = b.block_params(blk)[2];
    let fn_table_base = b.block_params(blk)[3];
    let vmctx = b.block_params(blk)[4];

    // Decode args (context pointers first), call the entry, store results. When the entry returns
    // via sret, hand it `results_ptr` directly as the return-area pointer — it writes its results
    // (8-byte `encode_slot` slots) straight into the buffer Rust reads, so no register read-back.
    let sret = uses_sret(&entry.results);
    let mut call_args = vec![mem_base, fn_table_base, vmctx];
    // §2b path B: the root runs on the calling thread's own stack, so it takes that stack's limit from
    // the host (#1983, `stack_check::thread_limit`); fibers get a real limit at their own entry.
    let limit = if with_limit {
        b.block_params(blk)[5]
    } else {
        let call = call_host(module, &mut b, HostFn::StackLimit, &[])?;
        b.inst_results(call)[0]
    };
    call_args.push(limit);
    if sret {
        call_args.push(results_ptr);
    }
    for (i, p) in entry.params.iter().enumerate() {
        let slot = b
            .ins()
            .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
        call_args.push(decode_slot(&mut b, slot, *p));
    }
    let callee = module.declare_func_in_func(entry_id, b.func);
    let call = b.ins().call(callee, &call_args);
    if !sret {
        let rets: Vec<Value> = b.inst_results(call).to_vec();
        for (i, r) in rets.iter().enumerate() {
            // A result slot holds what the interpreter's `Reg` would (#1720): an `i32` sign-extends,
            // so a §14 child's `_start` status reaches its joiner identically on every tier. Every
            // other reader decodes the slot by the declared type, where the extension is invisible.
            let slot = if b.func.dfg.value_type(*r) == I32 {
                b.ins().sextend(I64, *r)
            } else {
                encode_slot(&mut b, *r)
            };
            b.ins()
                .store(MemFlags::trusted(), slot, results_ptr, (i * 8) as i32);
        }
    }
    b.ins().return_(&[]);
    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

/// The fixed signature of a §12 fiber/thread entry: `(i64 sp, i64 arg) -> i64` (the unified
/// frontend convention). Its structural id is what a `cont.new` funcref is type-checked against.
#[cfg(fiber_rt)]
fn fiber_func_type() -> FuncType {
    FuncType {
        params: vec![ValType::I64, ValType::I64],
        results: vec![ValType::I64],
    }
}

/// Whether `m` contains any fiber op (or a `gc.roots`, which needs the fiber runtime to walk live
/// stacks), so `run_inner` knows to stand it up. Delegates to temen-ir's per-function predicates — the
/// single source of truth for the op sets backends must agree on (#913) — instead of re-enumerating.
#[cfg(fiber_rt)]
fn module_uses_fibers(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| f.uses_fibers() || f.uses_gc_roots())
}

/// Whether `m` contains any thread op (spawn/join/wait/notify), so `run_inner` knows to run under the
/// thread scheduler. `uses_threads` is spawn/join and `uses_futex` is wait/notify (temen-ir), so their
/// union is the "thread scheduler needed" set this gate wants.
#[cfg(fiber_rt)]
fn module_uses_threads(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| f.uses_threads() || f.uses_futex())
}

/// Whether `m` contains any `setjmp`/`longjmp` op, so `run_inner` knows to stand up the per-run
/// [`setjmp_rt::SetjmpRuntime`] those sites load from the vmctx.
#[cfg(setjmp_rt)]
fn module_uses_setjmp(m: &IrModule) -> bool {
    m.funcs.iter().any(|f| f.uses_setjmp())
}

/// Whether `m` holds a §14 `Instantiator` — a `call.cap` to `cap_id::INSTANTIATOR`
/// — so `run_inner` knows to stand up the nesting [`instantiator_rt::Nursery`].
#[cfg(fiber_rt)]
fn module_uses_instantiator(m: &IrModule) -> bool {
    funcs_use_instantiator(&m.funcs)
}

/// A run's §14 nursery, which its `Instantiator` sites spawn and join through: a compile's for the
/// instance it runs as, or one more instance's of shared code (temen#2389). Its children poll the
/// run's kill-path cell (§5), and compile their `atomic.wait`/`notify` against the run's thread
/// `domain`, so concurrent children and the parent rendezvous in one futex table.
#[cfg(fiber_rt)]
fn nursery_over(
    cap_thunk: CapThunk,
    cap_ctx: *mut core::ffi::c_void,
    resolve_module: Option<ModuleResolver>,
    epoch_addr: usize,
    domain: Option<&os_thread_rt::Domain>,
    shadow: temen_ir::durable_abi::ShadowArena,
) -> Box<instantiator_rt::Nursery> {
    Box::new(instantiator_rt::Nursery::new(
        cap_thunk,
        cap_ctx,
        resolve_module,
        epoch_addr,
        domain.map_or(0, |d| d as *const os_thread_rt::Domain as usize),
        None, // the run's own child executor
        shadow,
    ))
}

/// [`module_uses_instantiator`] over a bare function list — a §22 unit's, in `define_extra`.
#[cfg(fiber_rt)]
fn funcs_use_instantiator(funcs: &[Func]) -> bool {
    funcs.iter().any(|f| {
        f.blocks.iter().any(|blk| {
            blk.insts.iter().any(|i| {
                matches!(
                    i,
                    Inst::CapCall {
                        type_id: cap_id::INSTANTIATOR,
                        ..
                    }
                )
            })
        })
    })
}

/// Build the generic fiber **call-trampoline**: `extern "C" fn(code, mem_base, fn_table_base,
/// vmctx, sp, arg) -> i64` that `call.dyn`s a guest fiber entry under its `Tail` ABI. Rust
/// cannot call a `Tail`-convention function directly, so the fiber runtime calls this (default C ABI)
/// instead; one trampoline serves all fibers since every entry is `(i64 sp, i64 arg) -> i64`.
#[cfg(fiber_rt)]
fn build_fiber_call_trampoline(module: &mut JITModule, clif: &mut Function) {
    // Params: `code`, then the guest entry's Tail args in order — mem_base, fn_table_base, vmctx,
    // stack_limit (§2b path B, always present), sp, arg -> i64. `stack_limit` sits in the same relative
    // slot as in `sig_from`, so the entry call args are exactly `p[1..]` (see [`FiberCallTramp`]).
    let np = 7; // code + (mem_base, fn_table_base, vmctx, stack_limit, sp, arg)
    for _ in 0..np {
        clif.signature.params.push(AbiParam::new(I64));
    }
    clif.signature.returns.push(AbiParam::new(I64));
    clif.name = UserFuncName::user(0, 2);

    let mut fbctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(clif, &mut fbctx);
    let blk = b.create_block();
    for _ in 0..np {
        b.append_block_param(blk, I64);
    }
    b.switch_to_block(blk);
    b.seal_block(blk);
    let p = b.block_params(blk).to_vec();
    let code = p[0];
    let call_args = p[1..].to_vec(); // (mem_base, fn_table_base, vmctx, [stack_limit], sp, arg)
    let sig = b.import_signature(sig_from(
        module,
        &[ValType::I64, ValType::I64],
        &[ValType::I64],
    ));
    let call = b.ins().call_indirect(sig, code, &call_args);
    let r = b.inst_results(call)[0];
    b.ins().return_(&[r]);
    b.seal_all_blocks();
    b.finalize();
}

/// Lower one IR block's body + terminator into its CLIF block.
fn lower_block(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    blk: &Block,
    block_idx: usize,
    blocks: &mut Blocks,
    lower: &Lower,
) -> Result<(), JitError> {
    let cb = blocks.clif[block_idx];
    b.switch_to_block(cb);
    // Seed the value map with the IR block params: each a CLIF block param, or the value a taken
    // param forwards, which a block lowered before this one holds (#2448).
    let mut own = b.block_params(cb).to_vec().into_iter();
    let mut vals: Vec<Value> = Vec::with_capacity(blk.params.len() + blk.insts.len());
    for i in 0..blk.params.len() {
        vals.push(match blocks.taken(block_idx, i) {
            Some((sb, si)) => *blocks.vals[sb].get(si).ok_or(JitError::Malformed)?,
            None => own.next().ok_or(JitError::Malformed)?,
        });
    }
    // Parallel upper-bound map (for mask elision); block params are unknown. Kept in lockstep
    // with `vals` so `ubs[i]` describes IR value `i` — a misalignment could mis-elide a mask (the
    // INVARIANTS #2 escape class). Rather than re-growing it by hand in every multi-result arm (a
    // 25th-arm slip away from a mis-elide), it is repaired to `vals.len()` at the **top of each
    // iteration** (below), before any bound is consulted. So a slip now pads with `UB_TOP`
    // ("unknown ⇒ emit the check") — degrading to a redundant mask (perf pennies), never a missing
    // one — and the single-value tail's `ubs.push` is the only hand-placed growth. `size` confines
    // via `mask` (= size−1).
    let mut ubs: Vec<u64> = vec![UB_TOP; vals.len()];
    let size = lower.mask.wrapping_add(1);
    // #2147 — the bounds checks this block's loads and stores share.
    let mut checked = CheckedBases::new(lower);

    for (inst_idx, inst) in blk.insts.iter().enumerate() {
        // Single-site `ubs` maintenance (#893): repair any lag a previous multi-result arm left —
        // those arms grow `vals` but no longer touch `ubs` — before any elision bound is read this
        // iteration. A same-length resize is a no-op, so there is no cost on the common path, and
        // the lockstep is pinned by a `debug_assert_eq!` at each consult site.
        ubs.resize(vals.len(), UB_TOP);
        // W5 JIT/DWARF Stage 0: stamp the ops this IR instruction lowers to with a `SourceLoc`
        // (= its `debug_info.locs` index), so the finalized code's address map carries source
        // positions. Only when the module has `-g` debug info; otherwise codegen is unchanged.
        if let Some(map) = lower.srclocs {
            if let Some(&loc) = map.get(&(lower.func_idx, block_idx as u32, inst_idx as u32)) {
                b.set_srcloc(SourceLoc::new(loc));
            }
        }
        // `call`/`call.dyn` append 0..N results — handle before the single-value
        // match (which produces exactly one value).
        if let Inst::Call { func, args } = inst {
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let results = &lower
                .funcs
                .get(*func as usize)
                .ok_or(JitError::Malformed)?
                .results;
            let mut cargs = ctx_args(b, lower);
            let sret = sret_call_slot(b, &mut cargs, results);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call(callee, &cargs);
            // A trap raised inside the callee leaves the trap cell set and returns zeros; propagate
            // it here so it unwinds immediately (else the caller would run on with bogus results,
            // and a later successful `call.cap` could reset the cell, masking the trap). On the sret
            // path this also returns *before* reading the unwritten return-area slots.
            emit_trap_propagate(b, lower);
            read_call_results(b, call, sret, results, &mut vals);
            continue;
        }
        if let Inst::CallIndirect { ty, idx, args } = inst {
            let ty = sig_of(lower.type_section, *ty); // #922: resolve interned call type index
            let code = indirect_dispatch(b, lower, get(&vals, *idx)?, ty);
            let sig = b.import_signature(sig_from(module, &ty.params, &ty.results));
            let mut cargs = ctx_args(b, lower);
            let sret = sret_call_slot(b, &mut cargs, &ty.results);
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            let call = b.ins().call_indirect(sig, code, &cargs);
            // Propagate a callee trap immediately (see the direct-call case above).
            emit_trap_propagate(b, lower);
            read_call_results(b, call, sret, &ty.results, &mut vals);
            continue;
        }
        if let Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } = inst
        {
            // #922: resolve interned call type index.
            let sig = sig_of(lower.type_section, *sig);
            // §3c: the verifier checks a `call.cap` against its *declared* `sig` only, so a
            // verifier-valid module can call an op an engine serves itself with any shape. Held to
            // the op's contract (`temen_ir::served_contract`), as every engine holds it: a call
            // that doesn't match, or an Instantiator op that isn't live, lowers to an
            // unconditional runtime `CapFault`, as the interpreters trap — never a compile-time
            // rejection of a verified module (#2066, #2361). (Found by the libFuzzer `diff`
            // target: Instantiator ops 5/6/7 with short args made the compile fail `Malformed`,
            // crashing the differential — nightlies Jun 19 / Jul 2 / Jul 4, ISSUES.md I16.)
            if !temen_ir::served_shape_ok(*type_id, *op, sig) {
                emit_trap_set(b, lower, TrapKind::CapFault);
                emit_trap_propagate(b, lower);
                // Dead at runtime (the cell is already set), but keep the verifier's value
                // accounting for the rest of the block: push zeros of the *declared* result types.
                for t in &sig.results {
                    let z = zero_of(b, clif_ty(*t));
                    vals.push(z);
                }
                continue;
            }
            // CALLS.md §10.6 / increment 5 — `fuel.remaining` (self-namespace op 13): read the
            // domain's remaining fuel **inline**, from the same host-owned cell `emit_fuel_check`
            // charges, mirroring the interp arm that pushes `*fuel` (and charging none of its own).
            // Fuel is safepoint-anchored and bit-exact across engines, so this returns the identical
            // value the tree-walker does under the differential oracle (which arms counted fuel on
            // the JIT). When no fuel is armed for this compile (`fuel_addr == 0` — the production CLI
            // bounds runaways via the interrupt kill-path, not a counter) there is no cell to read,
            // so report `i64::MAX` ("unmetered"): the honest answer for an unbudgeted run.
            if *type_id == temen_ir::CAP_SELF_TYPE_ID && *op == 13 {
                let v = if lower.fuel {
                    // #1944 slice 3 — what the domain can still burn: the cell's `left` plus its
                    // budget chain's room, asked of the cell (word 2).
                    let addr = vmctx_load(b, lower, vmctx::FUEL);
                    let rsig = {
                        let mut s = Signature::new(lower.frontend_config.default_call_conv);
                        s.params.push(AbiParam::new(I64)); // the cell
                        s.returns.push(AbiParam::new(I64)); // the remaining fuel
                        b.import_signature(s)
                    };
                    let remaining =
                        b.ins()
                            .load(I64, MemFlags::trusted(), addr, crate::fuel::REMAINING_OFF);
                    let call = b.ins().call_indirect(rsig, remaining, &[addr]);
                    b.inst_results(call)[0]
                } else {
                    b.ins().iconst(I64, i64::MAX)
                };
                vals.push(v);
                continue;
            }
            // FORK.md §8.6 — `exec_module` (`execve` image-replace, op 14) is eval-loop-only: the
            // tree-walker folds it to `Step::Exec` before any host dispatch. The generic `call.cap`
            // thunk here would answer `-EINVAL` where the oracle image-replaces — a silent divergence
            // (INVARIANTS.md #9). Decline so the run folds to a reifiable tier (OPS_PARITY.md `exec`).
            if *type_id == temen_ir::CAP_SELF_TYPE_ID && *op == 14 {
                return Err(JitError::Unsupported(
                    "exec_module: eval-loop-only image-replace (op 14)",
                ));
            }
            // §14 `Instantiator` (iface 6): when this (parent) compile has a live `Nursery`, lower
            // `instantiate`/`join` to its thunks instead of the generic `call.cap` — spawning a child
            // needs the host compiler, which the flat `call.cap` thunk can't reach. Otherwise (a child
            // compile, or no nesting runtime) it falls through to the ordinary path (an inert CapFault).
            if *type_id == cap_id::INSTANTIATOR && lower.nesting {
                lower_instantiator(module, b, lower, *op, sig, *handle, args, &mut vals)?;
            } else if let Some(target) = fast_cap_target(lower, *type_id, *op, sig) {
                // D45 devirtualized fast path: a register-to-register direct call to the specialized
                // host fn the resolver claimed for this `(type_id, op)`.
                lower_cap_call_fast(module, b, lower, &target, sig, *handle, args, &mut vals)?;
            } else {
                let h = get(&vals, *handle)?;
                lower_cap_call(module, b, lower, *type_id, *op, sig, h, args, &mut vals)?;
            }
            continue;
        }
        // §7 executable named import (IMPORTS.md phase 1): lower like the `self.*` family — a
        // `call.cap` thunk with the reserved `CAP_IMPORT_TYPE_ID` and the **import index** as the
        // op. The host's dispatch translates it through the instantiation-time binding table
        // (import `i` → bound `(type_id, op)` + granted handle) and re-dispatches — one shared
        // implementation with the interpreter and the bytecode engine. The vestigial handle operand
        // is not read (constant 0, like `self.*`); the module bytes are never rewritten, so the
        // compiled code is identical across instantiations (the binding is host-side state).
        //
        // §3.5: the reserved import dispatch packs `(slot | consumer_op << 16)`; a bound `call.sym`
        // (§7/§22 symbolic call) is a flat import dispatch (consumer op 0) whose legacy handle
        // operand is not read either. The dynamic-mode dispatch by type-section reference packs
        // `(type_idx | op << 16)` into the reserved dyn entry, and its handle operand is live (the
        // value being driven). [`Inst::host_dispatch`] is the one definition of the packings.
        if let Inst::CallImport { sig, args, .. }
        | Inst::CallSym { sig, args, .. }
        | Inst::CallImportDyn { sig, args, .. } = inst
        {
            let sig = sig_of(lower.type_section, *sig); // #922: resolve interned call type index
            let (type_id, op) = inst.host_dispatch().expect("an import-form call");
            let h = match inst {
                Inst::CallImportDyn { handle, .. } => get(&vals, *handle)?,
                _ => b.ins().iconst(I32, 0),
            };
            lower_cap_call(module, b, lower, type_id, op, sig, h, args, &mut vals)?;
            continue;
        }
        // §3.5 self-namespace extensions — the shared dispatch entry with `CAP_SELF_TYPE_ID` and
        // op packed `(selfop | idx << 8)`: 6 = `self.type_id`, 7 = `self.covers`
        // (handle as the one argument), 8 = `export.handle`. One `i32` result each.
        if matches!(
            inst,
            Inst::CapSelfTypeId { .. } | Inst::CapSelfCovers { .. } | Inst::ExportHandle { .. }
        ) {
            let h0 = b.ins().iconst(I32, 0);
            let i32_result = FuncType {
                params: vec![],
                results: vec![ValType::I32],
            };
            let (op, sig, call_args): (u32, FuncType, &[u32]) = match inst {
                Inst::CapSelfTypeId { ty } => (6 | (*ty << 8), i32_result.clone(), &[]),
                Inst::CapSelfCovers { handle, ty } => (
                    7 | (*ty << 8),
                    FuncType {
                        params: vec![ValType::I32],
                        results: vec![ValType::I32],
                    },
                    std::slice::from_ref(handle),
                ),
                Inst::ExportHandle { export } => (8 | (*export << 8), i32_result.clone(), &[]),
                _ => unreachable!(),
            };
            lower_cap_call(
                module,
                b,
                lower,
                temen_ir::CAP_SELF_TYPE_ID,
                op,
                &sig,
                h0,
                call_args,
                &mut vals,
            )?;
            continue;
        }
        // Phase-2 `import.attach` (IMPORTS.md): the attach sentinel with the handle value as the
        // one `i32` argument; `i32` status result. Same shared host entry as the interpreter and
        // the bytecode engine.
        if let Inst::ImportAttach { import, handle } = inst {
            let h0 = b.ins().iconst(I32, 0);
            let sig = FuncType {
                params: vec![ValType::I32],
                results: vec![ValType::I32],
            };
            let call_args = [*handle];
            lower_cap_call(
                module,
                b,
                lower,
                temen_ir::CAP_IMPORT_ATTACH_TYPE_ID,
                *import,
                &sig,
                h0,
                &call_args,
                &mut vals,
            )?;
            continue;
        }
        // §7/§6 capability reflection (`self.count`/`get`/`resolve`/`label`/`attest`) is no longer
        // a distinct IR op — the frontend emits its `call.cap CAP_SELF op N` form, lowered by the
        // generic `Inst::CapCall` arm above (a `lower_cap_call` thunk the host services directly).
        // §12 fibers: lower `cont.*` to calls to the host fiber thunks ([`HostFn`]), threading
        // `mem_base`/`fn_table_base`/`vmctx` like `call.cap`. A thunk that sets the trap cell
        // (forged handle, bad funcref, fiber-bomb, root suspend) propagates here.
        if let Inst::ContNew { func, sp } = inst {
            // fiber_new(mem_base, fn_table_base, vmctx, funcref:i32, sp:i64, fuel_addr:i64) -> i32
            // handle. The running vCPU's fiber runtime is read from a thread-local, so threads +
            // fibers compose. `fuel_addr` (`0` ⇒ un-armed) lets the fiber's first entry refund the
            // prologue charge its starting `cont.resume` already paid (#1642).
            let mem_base = b.use_var(lower.mem_var);
            let fnt = b.use_var(lower.fn_table_var);
            let vmctx = b.use_var(lower.vmctx_var);
            let funcref = get(&vals, *func)?;
            let spv = get(&vals, *sp)?;
            let fuel = if lower.fuel {
                vmctx_load(b, lower, vmctx::FUEL)
            } else {
                b.ins().iconst(I64, 0)
            };
            let call = call_host(
                module,
                b,
                HostFn::FiberNew,
                &[mem_base, fnt, vmctx, funcref, spv, fuel],
            )?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        // I48: `cont.resume` returns `FIBER_PARKED` on a still-parked fiber (the guest polls);
        // `cont.resume.block` instead calls the blocking thunk, which idles this OS thread on the
        // domain condvar and re-resumes until the fiber completes/suspends — a sleep, not a spin,
        // so the JIT idles like the oracle and the cooperative bytecode driver (INVARIANTS.md #9).
        // Both append the two results (status:i32, value:i64) to match the IR's shape.
        if let Inst::ContResume { k, arg, block } = inst {
            // Fuel unification (#1642): one fuel per `cont.resume`, charged here, before the claim —
            // where the interpreters take theirs, and one per op whether the resume starts the fiber,
            // switches back into a suspended one, or finds it still parked. This site used to charge
            // nothing: the fiber's entry prologue paid for its *start*, so a fiber resumed once cost
            // the same as on the interpreters and every re-resume after that went unmetered. A start
            // now pays here, and `fiber_rt::make_fiber` refunds the entry prologue's second charge.
            emit_fuel_check(b, lower);
            let blocking = *block;
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let kh = get(&vals, *k)?;
            let av = get(&vals, *arg)?;
            let vmctx = b.use_var(lower.vmctx_var);
            let call = if blocking {
                // fiber_resume_block(sched, handle, arg, status_out, vmctx) -> value:i64.
                let sched = vmctx_load(b, lower, vmctx::SCHED);
                call_host(
                    module,
                    b,
                    HostFn::FiberResumeBlock,
                    &[sched, kh, av, status_ptr, vmctx],
                )?
            } else {
                // fiber_resume(handle, arg, status_out, vmctx) -> value:i64.
                call_host(module, b, HostFn::FiberResume, &[kh, av, status_ptr, vmctx])?
            };
            emit_trap_propagate(b, lower);
            let value = b.inst_results(call)[0];
            let status64 = b.ins().stack_load(I64, ss, 0);
            let status = b.ins().ireduce(I32, status64);
            vals.push(status); // result 0: status (i32)
            vals.push(value); // result 1: value (i64)
            continue;
        }
        if let Inst::Suspend { value } = inst {
            // fiber_suspend(value:i64, vmctx:i64) -> next-resume arg:i64
            let v = get(&vals, *value)?;
            let vmctx = b.use_var(lower.vmctx_var);
            let call = call_host(module, b, HostFn::FiberSuspend, &[v, vmctx])?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        // `setjmp` (LLVM.md §"JIT `longjmp`", Option B): two calls. First a host thunk returns the
        // stable host `jmp_buf` slot for this guest `buf` address (`rt_setjmp_slot`); then
        // `_setjmp` is called **inline in this guest frame** — the frame a later `longjmp` returns
        // to — so it saves *this* frame's SP/return-addr. The code calls libc `_setjmp` itself (not
        // wrapped in a Rust thunk, whose frame would be gone by `longjmp` time — UB). Result is the
        // `i32` 0 (direct) / long-jump value (re-entry). The slot alloc is infallible, so no
        // trap-propagate.
        if let Inst::SetJmp { buf } = inst {
            let bufv = get(&vals, *buf)?; // i64 guest jmp_buf window address (the table key)
                                          // slot = rt_setjmp_slot(rt_addr, buf) -> *mut jmp_buf
            let rt_addr = vmctx_load(b, lower, vmctx::SETJMP);
            let c1 = call_host(module, b, HostFn::SetjmpSlot, &[rt_addr, bufv])?;
            let slot = b.inst_results(c1)[0];
            // r = _setjmp(slot) -> i32 — emitted inline in this JIT frame.
            let c2 = call_host(module, b, HostFn::Setjmp, &[slot])?;
            vals.push(b.inst_results(c2)[0]);
            continue;
        }
        // `longjmp` (LLVM.md §"JIT `longjmp`"): look up the host `jmp_buf` slot for this `buf` (set by a
        // prior `setjmp`); a miss writes the trap cell and `emit_trap_propagate` bails *before* the
        // (skipped) `_longjmp`. Otherwise `_longjmp(slot, val)` restores the saved frame and never
        // returns — the IR's trailing `unreachable` terminator caps the block (the call isn't marked
        // noreturn, but the dead fall-through is terminated by it). No result.
        if let Inst::LongJmp { buf, val } = inst {
            let bufv = get(&vals, *buf)?;
            let valv = get(&vals, *val)?; // i32 long-jump value (0 → 1 is applied by libc `_longjmp`)
            let vmctx = b.use_var(lower.vmctx_var);
            // slot = rt_setjmp_lookup(rt_addr, buf, vmctx) -> *mut jmp_buf (null + trap on miss)
            let rt_addr = vmctx_load(b, lower, vmctx::SETJMP);
            let c1 = call_host(module, b, HostFn::SetjmpLookup, &[rt_addr, bufv, vmctx])?;
            let slot = b.inst_results(c1)[0];
            emit_trap_propagate(b, lower); // bail on a miss (forged/stale token) before `_longjmp`
                                           // _longjmp(slot, val) — inline, noreturn.
            call_host(module, b, HostFn::Longjmp, &[slot, valv])?;
            continue;
        }
        // §12 per-vCPU TLS register: thunks over a per-OS-thread word (no window/trap context —
        // a pure thread-local access that cannot fault). `get` reads the *current* vCPU's word, so it
        // is correct after a fiber migrates here.
        if let Inst::VcpuTlsGet = inst {
            let call = call_host(module, b, HostFn::VcpuTlsGet, &[])?;
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        // §12.8 4A.5 durable-runtime-internal: read the current context's shadow-region base from the
        // per-OS-thread register (a thunk, like `vcpu.tls.get`; cannot fault, no window/trap
        // context). The durable transform emits this to address this context's own shadow-SP word.
        if let Inst::DurableShadowBase = inst {
            let call = call_host(module, b, HostFn::DurableShadowBase, &[])?;
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        if let Inst::VcpuTlsSet { val } = inst {
            let v = get(&vals, *val)?;
            call_host(module, b, HostFn::VcpuTlsSet, &[v])?;
            continue;
        }
        if let Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } = inst
        {
            // gc_roots(heap_lo, heap_hi, payload_mask, buf, cap, mem_base, mask, mapped, vmctx)
            // -> i64 count. The thunk walks the live fiber stacks (runtime via the thread-local),
            // masks each word with `payload_mask` (§GC tagged pointers; distinct from the
            // window-confinement `mask`), filters the masked value to `[heap_lo, heap_hi)`, and
            // writes the first `cap` deduped words to guest `buf` — confining/bounds-checking it
            // with the same `mask`/bound as `mask_addr`, so a forged buffer faults
            // (below). #826: the bound is the **reserved** mask domain (not the compile-time
            // `mapped`), so a roots buffer in a legitimately `map`-grown tail page is admitted
            // where the interpreter's live page map admits it; committed-ness and write-protection
            // are the hardware's to enforce — a write into an uncommitted or `Ro` page faults through
            // the SIGSEGV shim to `MemoryFault`, exactly like a guest store.
            let lo = get(&vals, *heap_lo)?;
            let hi = get(&vals, *heap_hi)?;
            let payload_mask = get(&vals, *mask)?;
            let dst = get(&vals, *buf)?;
            let cap_v = get(&vals, *cap)?;
            let mem_base = b.use_var(lower.mem_var);
            let maskv = b.ins().iconst(I64, lower.mask as i64);
            let win_reserved = if lower.mapped == 0 { 0 } else { lower.mask + 1 };
            let mappedv = b.ins().iconst(I64, win_reserved as i64);
            let vmctx = b.use_var(lower.vmctx_var);
            // #2164 — a demand-paged child's thunk checks the access table before it writes: the
            // declared window's page count bounds it.
            let pages = b
                .ins()
                .iconst(I64, lower.page.map_or(0, |(p, ..)| p.pages as i64));
            // Marshal the ten args into one 8-byte-aligned stack slot (matching `GcRootsArgs`'s
            // `#[repr(C)]` field order) and pass a single pointer. The thunk is the register-flush
            // trampoline, which spills the callee-saved registers before the scan; a one-pointer ABI
            // lets it do so with no per-argument reshuffle (see `fiber_rt::temen_gc_roots_flush`).
            let slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                80, // 10 × 8 bytes
                3,  // 8-byte aligned
            ));
            for (i, v) in [
                lo,
                hi,
                payload_mask,
                dst,
                cap_v,
                mem_base,
                maskv,
                mappedv,
                vmctx,
                pages,
            ]
            .into_iter()
            .enumerate()
            {
                b.ins().stack_store(v, slot, (i * 8) as i32);
            }
            let args_ptr = b.ins().stack_addr(I64, slot, 0);
            let call = call_host(module, b, HostFn::GcRoots, &[args_ptr])?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        // §12 threads: lower `thread.spawn`/`thread.join` to calls to the host scheduler thunks
        // ([`HostFn`]), threading `mem_base`/`fn_table_base`/`vmctx` like `call.cap`. A thunk
        // that sets the trap cell (forged handle, thread-bomb) propagates here.
        if let Inst::ThreadSpawn { func, sp, arg } = inst {
            // thread_spawn(sched, mem_base, fn_table_base, vmctx, func_idx:i32, sp:i64, arg:i64) -> i32
            let sched = vmctx_load(b, lower, vmctx::SCHED);
            let mem_base = b.use_var(lower.mem_var);
            let fnt = b.use_var(lower.fn_table_var);
            let vmctx = b.use_var(lower.vmctx_var);
            // The spawn entry is dispatched through the shared `fn_table` (`os_thread_rt` masks it
            // into the table like any `call.dyn`). A submitted unit's own functions live in
            // auto-installed padding slots, not at their module-relative indices — so remap `func`
            // through `ref_slots` exactly like `ref.func` (DESIGN.md §22 "unit-own funcref"). Without
            // this a unit's `thread.spawn N` would launch the *parent's* function N. Top-level
            // compiles have no remap: `func` is module-0 slot N verbatim.
            let entry = match lower.ref_slots {
                Some(slots) => *slots.get(*func as usize).unwrap_or(func),
                None => *func,
            };
            let func_idx = b.ins().iconst(I32, entry as i64);
            let spv = get(&vals, *sp)?;
            let av = get(&vals, *arg)?;
            let call = call_host(
                module,
                b,
                HostFn::ThreadSpawn,
                &[sched, mem_base, fnt, vmctx, func_idx, spv, av],
            )?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        if let Inst::ThreadJoin { handle } = inst {
            // thread_join(sched, handle:i32, vmctx:i64) -> result:i64
            let sched = vmctx_load(b, lower, vmctx::SCHED);
            let h = get(&vals, *handle)?;
            let vmctx = b.use_var(lower.vmctx_var);
            let call = call_host(module, b, HostFn::ThreadJoin, &[sched, h, vmctx])?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        if let Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } = inst
        {
            // thread_wait(sched, phys:i64, expected:i64, width:i32, timeout:i64, vmctx:i64) ->
            // status:i32. `vmctx` carries the §12.8 thaw fail-closed (a re-issued wait that would
            // park on the single worker traps `ThreadFault`); on a fresh run it is never written.
            let w = atomic_width(*ty);
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false, w);
            guard_atomic_align(b, lower, phys, w); // misaligned wait traps (like the other atomics)
                                                   // #2164 — a wait reads its word, and a word on an unsupplied page faults it: fatal on
                                                   // every engine, never paged.
            emit_page_check(b, lower, phys, w, PAGE_MISS_FATAL);
            let sched = vmctx_load(b, lower, vmctx::SCHED);
            let exp_raw = get(&vals, *expected)?;
            let exp = if w < 8 {
                b.ins().uextend(I64, exp_raw) // compare is bit-equality on the low `w` bytes
            } else {
                exp_raw
            };
            let width = b.ins().iconst(I32, w as i64);
            let to = get(&vals, *timeout)?;
            let vmctx = b.use_var(lower.vmctx_var);
            let call = call_host(
                module,
                b,
                HostFn::ThreadWait,
                &[sched, phys, exp, width, to, vmctx],
            )?;
            emit_trap_propagate(b, lower);
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        if let Inst::MemoryNotify { addr, count } = inst {
            // thread_notify(sched, phys:i64, count:i32) -> woken:i32. Accesses no memory (the address
            // is only confined, no alignment requirement — matching the interpreter).
            let phys = mask_addr(b, lower, get(&vals, *addr)?, 0, false, 4);
            let sched = vmctx_load(b, lower, vmctx::SCHED);
            let cnt = get(&vals, *count)?;
            let call = call_host(module, b, HostFn::ThreadNotify, &[sched, phys, cnt])?;
            vals.push(b.inst_results(call)[0]);
            continue;
        }
        let v = match inst {
            Inst::ConstI32(c) => b.ins().iconst(I32, *c as i64),
            Inst::ConstI64(c) => b.ins().iconst(I64, *c),
            Inst::IntBin { ty, op, a, b: rb } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                match op {
                    // div/rem trap on a zero divisor (and signed div on INT_MIN/-1):
                    // guard with explicit checks that branch to a trap-return.
                    BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => {
                        lower_div_rem(b, lower, *ty, *op, x, y)
                    }
                    _ => int_bin(b, *op, x, y),
                }
            }
            Inst::IntUn { op, a, ty } => {
                let x = get(&vals, *a)?;
                let rt = int_clif_ty(*ty);
                // `extendN_s` sign-extends the low N bits: reduce to iN, then sign-extend
                // back to the (same) result width. When N == the result width (`extend32_s`
                // on `i32`) it's the identity — `ireduce`/`sextend` both require a strict
                // width change, so pass `x` through.
                let sext_low = |b: &mut FunctionBuilder, nt: Type| {
                    if nt == rt {
                        x
                    } else {
                        let r = b.ins().ireduce(nt, x);
                        b.ins().sextend(rt, r)
                    }
                };
                match op {
                    IntUnOp::Clz => b.ins().clz(x),
                    IntUnOp::Ctz => b.ins().ctz(x),
                    IntUnOp::Popcnt => b.ins().popcnt(x),
                    IntUnOp::Extend8S => sext_low(b, I8),
                    IntUnOp::Extend16S => sext_low(b, I16),
                    IntUnOp::Extend32S => sext_low(b, I32),
                }
            }
            Inst::IntCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().icmp(int_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Eqz { a, .. } => {
                let x = get(&vals, *a)?;
                let c = b.ins().icmp_imm(IntCC::Equal, x, 0);
                b.ins().uextend(I32, c)
            }
            Inst::Select { cond, a, b: rb } => {
                let (c, x, y) = (get(&vals, *cond)?, get(&vals, *a)?, get(&vals, *rb)?);
                b.ins().select(c, x, y)
            }
            Inst::ConstF32(bits) => {
                // Materialize via the exact bit pattern (NaN-safe), then bitcast.
                let i = b.ins().iconst(I32, *bits as i64);
                b.ins().bitcast(F32, MemFlags::new(), i)
            }
            Inst::ConstF64(bits) => {
                let i = b.ins().iconst(I64, *bits as i64);
                b.ins().bitcast(F64, MemFlags::new(), i)
            }
            Inst::FBin { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                float_bin(b, *op, x, y)
            }
            Inst::FUn { op, a, .. } => {
                let x = get(&vals, *a)?;
                float_un(b, *op, x)
            }
            Inst::Fma { a, b: rb, c, .. } => {
                // Scalar fused multiply-add — `a·b + c`, one rounding, as the interp's `libm` `fma`.
                let (x, y, z) = (get(&vals, *a)?, get(&vals, *rb)?, get(&vals, *c)?);
                b.ins().fma(x, y, z)
            }
            Inst::FCmp { op, a, b: rb, .. } => {
                let (x, y) = (get(&vals, *a)?, get(&vals, *rb)?);
                let c = b.ins().fcmp(float_cc(*op), x, y);
                b.ins().uextend(I32, c) // bool (I8) -> i32 0/1
            }
            Inst::Convert { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    ConvOp::ExtendI32S => b.ins().sextend(I64, x),
                    ConvOp::ExtendI32U => b.ins().uextend(I64, x),
                    ConvOp::WrapI64 => b.ins().ireduce(I32, x),
                }
            }
            Inst::Cast { op, a } => {
                let x = get(&vals, *a)?;
                match op {
                    CastOp::Demote => b.ins().fdemote(F32, x),
                    CastOp::Promote => b.ins().fpromote(F64, x),
                    CastOp::ReinterpI32F32 => b.ins().bitcast(F32, MemFlags::new(), x),
                    CastOp::ReinterpF32I32 => b.ins().bitcast(I32, MemFlags::new(), x),
                    CastOp::ReinterpI64F64 => b.ins().bitcast(F64, MemFlags::new(), x),
                    CastOp::ReinterpF64I64 => b.ins().bitcast(I64, MemFlags::new(), x),
                }
            }
            Inst::IToFConv { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let fty = float_clif_ty(to);
                if signed {
                    b.ins().fcvt_from_sint(fty, x)
                } else {
                    b.ins().fcvt_from_uint(fty, x)
                }
            }
            Inst::FToISat { op, a } => {
                let x = get(&vals, *a)?;
                let (_, to, signed) = op.parts();
                let ity = int_clif_ty(to);
                // Saturating (wasm trunc_sat): NaN→0, out-of-range→clamp — exactly
                // Cranelift's saturating fcvt, so it matches the interpreter.
                if signed {
                    b.ins().fcvt_to_sint_sat(ity, x)
                } else {
                    b.ins().fcvt_to_uint_sat(ity, x)
                }
            }
            Inst::FToITrap { op, a } => {
                let (from, to, signed) = op.parts();
                lower_trunc_trap(b, lower, get(&vals, *a)?, from, to, signed)
            }
            // ----- §17 SIMD (D58): native Cranelift vector lowering -----
            Inst::ConstV128(bytes) => {
                let c = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
                b.ins().vconst(I8X16, c)
            }
            Inst::V128Load { addr, offset, .. } => {
                // The 16-byte masked access — the lone escape-TCB delta SIMD adds (§17/D58).
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, 16, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, 16);
                emit_page_check(b, lower, phys, 16, 0);
                b.ins().load(I8X16, mem_flags(), phys, 0)
            }
            Inst::V128Store {
                addr,
                value,
                offset,
                ..
            } => {
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, 16, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, 16);
                emit_page_check(b, lower, phys, 16, PAGE_MISS_WRITE);
                b.ins().store(mem_flags(), get(&vals, *value)?, phys, 0);
                continue; // store produces no value
            }
            Inst::Splat { shape, a } => {
                let s = get(&vals, *a)?;
                // The scalar arrives as the lane's `lane_val` (i32 for narrow ints); narrow to the
                // CLIF lane type, splat, then canonicalize to I8X16.
                let lane = lane_clif(*shape);
                let s = if b.func.dfg.value_type(s) == lane {
                    s
                } else if lane == I8 || lane == I16 {
                    b.ins().ireduce(lane, s)
                } else {
                    s
                };
                let v = b.ins().splat(vec_ty(*shape), s);
                vcast(b, v, I8X16)
            }
            Inst::ExtractLane {
                shape,
                lane,
                signed,
                a,
            } => {
                let v = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let raw = b.ins().extractlane(v, *lane);
                match shape {
                    // Narrow integer lanes widen to the i32 result (sign/zero per `signed`).
                    VShape::I8x16 | VShape::I16x8 => {
                        if *signed {
                            b.ins().sextend(I32, raw)
                        } else {
                            b.ins().uextend(I32, raw)
                        }
                    }
                    // i32x4/i64x2/f32x4/f64x2 extract to the lane type directly.
                    _ => raw,
                }
            }
            Inst::ReplaceLane {
                shape,
                lane,
                a,
                b: rb,
            } => {
                let v = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let s = get(&vals, *rb)?;
                let lty = lane_clif(*shape);
                let s = if b.func.dfg.value_type(s) == lty {
                    s
                } else if lty == I8 || lty == I16 {
                    b.ins().ireduce(lty, s)
                } else {
                    s
                };
                let r = b.ins().insertlane(v, s, *lane);
                vcast(b, r, I8X16)
            }
            Inst::VIntBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = match op {
                    VIntBinOp::Add => b.ins().iadd(x, y),
                    VIntBinOp::Sub => b.ins().isub(x, y),
                    // x86 has no per-byte multiply (no `PMULLB`), so Cranelift can't legalize an
                    // `imul` on `i8x16`. Emulate it: widen each half to `i16x8` and multiply (the low
                    // byte of an `i16` product equals the low byte of the `i8` product, and that low
                    // byte is sign-independent), mask each product to its low byte, then pack the two
                    // halves back with unsigned-saturating narrow (every lane is ≤ 0xFF so nothing
                    // saturates — it's an exact low-byte truncation). Matches the interp's wrapping mul.
                    VIntBinOp::Mul if *shape == VShape::I8x16 => {
                        let (xl, yl) = (b.ins().uwiden_low(x), b.ins().uwiden_low(y));
                        let (xh, yh) = (b.ins().uwiden_high(x), b.ins().uwiden_high(y));
                        let pl = b.ins().imul(xl, yl);
                        let ph = b.ins().imul(xh, yh);
                        let m = b.ins().iconst(I16, 0x00ff);
                        let mask = b.ins().splat(I16X8, m);
                        let pl = b.ins().band(pl, mask);
                        let ph = b.ins().band(ph, mask);
                        b.ins().unarrow(pl, ph)
                    }
                    VIntBinOp::Mul => b.ins().imul(x, y),
                    // `i64x2` min/max: x86/aarch64 have no native `i64` vector min/max, so Cranelift
                    // cannot legalize `smin`/`umin`/`smax`/`umax` here. Synthesize per-lane
                    // compare + `bitselect` (both legalize on every target): `bitselect(m, x, y)` ==
                    // `(x & m) | (y & !m)` keeps `x` in the lanes where the compare mask is all-ones.
                    VIntBinOp::MinS if *shape == VShape::I64x2 => {
                        let m = b.ins().icmp(IntCC::SignedLessThan, x, y);
                        b.ins().bitselect(m, x, y)
                    }
                    VIntBinOp::MinU if *shape == VShape::I64x2 => {
                        let m = b.ins().icmp(IntCC::UnsignedLessThan, x, y);
                        b.ins().bitselect(m, x, y)
                    }
                    VIntBinOp::MaxS if *shape == VShape::I64x2 => {
                        let m = b.ins().icmp(IntCC::SignedGreaterThan, x, y);
                        b.ins().bitselect(m, x, y)
                    }
                    VIntBinOp::MaxU if *shape == VShape::I64x2 => {
                        let m = b.ins().icmp(IntCC::UnsignedGreaterThan, x, y);
                        b.ins().bitselect(m, x, y)
                    }
                    VIntBinOp::MinS => b.ins().smin(x, y),
                    VIntBinOp::MinU => b.ins().umin(x, y),
                    VIntBinOp::MaxS => b.ins().smax(x, y),
                    VIntBinOp::MaxU => b.ins().umax(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VIntCmp {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Vector `icmp` yields a per-lane all-ones/all-zeros mask of the lane width — exactly
                // the wasm/interp semantics — so this is a single instruction on the right vector type.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let cc = match op {
                    VICmpOp::Eq => IntCC::Equal,
                    VICmpOp::Ne => IntCC::NotEqual,
                    VICmpOp::LtS => IntCC::SignedLessThan,
                    VICmpOp::LtU => IntCC::UnsignedLessThan,
                    VICmpOp::GtS => IntCC::SignedGreaterThan,
                    VICmpOp::GtU => IntCC::UnsignedGreaterThan,
                    VICmpOp::LeS => IntCC::SignedLessThanOrEqual,
                    VICmpOp::LeU => IntCC::UnsignedLessThanOrEqual,
                    VICmpOp::GeS => IntCC::SignedGreaterThanOrEqual,
                    VICmpOp::GeU => IntCC::UnsignedGreaterThanOrEqual,
                };
                let r = b.ins().icmp(cc, x, y);
                vcast(b, r, I8X16)
            }
            Inst::VShift { shape, op, a, amt } => {
                // One scalar shift amount, masked to the lane bit-width (the wasm rule), broadcast
                // across the lanes by Cranelift's vector `ishl`/`ushr`/`sshr`.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let bits = (shape.lane_bytes() * 8) as i64;
                let sh = b.ins().band_imm(get(&vals, *amt)?, bits - 1);
                let r = match op {
                    VShiftOp::Shl => b.ins().ishl(x, sh),
                    VShiftOp::ShrU => b.ins().ushr(x, sh),
                    VShiftOp::ShrS => b.ins().sshr(x, sh),
                };
                vcast(b, r, I8X16)
            }
            Inst::VIntUn { shape, op, a } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let r = match op {
                    VIntUnOp::Abs => b.ins().iabs(x),
                    VIntUnOp::Neg => b.ins().ineg(x),
                };
                vcast(b, r, I8X16)
            }
            Inst::VPopcnt { a } => {
                // Canonical vectors are already I8X16, matching the op's fixed shape.
                b.ins().popcnt(get(&vals, *a)?)
            }
            Inst::VAvgr { shape, a, b: rb } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = b.ins().avg_round(x, y);
                vcast(b, r, I8X16)
            }
            Inst::VDot { a, b: rb } => {
                // Widen each i16x8 operand to two i32x4 halves, multiply lane-wise, then
                // horizontally add adjacent products: `iadd_pairwise([a0b0,a1b1,a2b2,a3b3],
                // [a4b4,..]) = [a0b0+a1b1, a2b2+a3b3, a4b4+a5b5, a6b6+a7b7]` — the wasm dot result.
                let x = vcast(b, get(&vals, *a)?, I16X8);
                let y = vcast(b, get(&vals, *rb)?, I16X8);
                let xl = b.ins().swiden_low(x);
                let xh = b.ins().swiden_high(x);
                let yl = b.ins().swiden_low(y);
                let yh = b.ins().swiden_high(y);
                let pl = b.ins().imul(xl, yl);
                let ph = b.ins().imul(xh, yh);
                let r = b.ins().iadd_pairwise(pl, ph);
                vcast(b, r, I8X16)
            }
            Inst::VDotI8 { a, b: rb } => {
                // The i8→i16 signed dot, same shape as `VDot` one width down: widen each i8x16 to
                // two i16x8 halves, multiply, pairwise-add. (Deterministic relaxed_dot_i8x16_i7x16_s.)
                let x = vcast(b, get(&vals, *a)?, I8X16);
                let y = vcast(b, get(&vals, *rb)?, I8X16);
                let xl = b.ins().swiden_low(x);
                let xh = b.ins().swiden_high(x);
                let yl = b.ins().swiden_low(y);
                let yh = b.ins().swiden_high(y);
                let pl = b.ins().imul(xl, yl);
                let ph = b.ins().imul(xh, yh);
                let r = b.ins().iadd_pairwise(pl, ph);
                vcast(b, r, I8X16)
            }
            Inst::VExtMul {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Widen the same (low/high, sign) half of both operands to the wide shape, then
                // multiply lane-wise — the wasm extmul.
                let (low, signed) = op.parts();
                let src = vec_ty(
                    shape
                        .narrower()
                        .expect("verifier ensures a narrower source"),
                );
                let x = vcast(b, get(&vals, *a)?, src);
                let y = vcast(b, get(&vals, *rb)?, src);
                let (wx, wy) = match (low, signed) {
                    (true, true) => (b.ins().swiden_low(x), b.ins().swiden_low(y)),
                    (false, true) => (b.ins().swiden_high(x), b.ins().swiden_high(y)),
                    (true, false) => (b.ins().uwiden_low(x), b.ins().uwiden_low(y)),
                    (false, false) => (b.ins().uwiden_high(x), b.ins().uwiden_high(y)),
                };
                let r = b.ins().imul(wx, wy);
                vcast(b, r, I8X16)
            }
            Inst::VExtAddPairwise { shape, signed, a } => {
                // Widen the low and high halves of the source, then pairwise-add: the two halves'
                // pairwise sums concatenate to `out[i] = w(a[2i]) + w(a[2i+1])`.
                let src = vec_ty(
                    shape
                        .narrower()
                        .expect("verifier ensures a narrower source"),
                );
                let x = vcast(b, get(&vals, *a)?, src);
                let (lo, hi) = if *signed {
                    (b.ins().swiden_low(x), b.ins().swiden_high(x))
                } else {
                    (b.ins().uwiden_low(x), b.ins().uwiden_high(x))
                };
                let r = b.ins().iadd_pairwise(lo, hi);
                vcast(b, r, I8X16)
            }
            Inst::VQ15MulrSat { a, b: rb } => {
                let x = vcast(b, get(&vals, *a)?, I16X8);
                let y = vcast(b, get(&vals, *rb)?, I16X8);
                let r = b.ins().sqmul_round_sat(x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFma {
                shape,
                neg,
                a,
                b: rb,
                c,
            } => {
                let ty = vec_ty(*shape);
                let xa = vcast(b, get(&vals, *a)?, ty);
                // `nmadd` is `−a·b + c`: negate the product by negating `a`.
                let x = if *neg { b.ins().fneg(xa) } else { xa };
                let y = vcast(b, get(&vals, *rb)?, ty);
                let z = vcast(b, get(&vals, *c)?, ty);
                let r = b.ins().fma(x, y, z);
                vcast(b, r, I8X16)
            }
            Inst::VSatBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let r = match op {
                    VSatBinOp::AddS => b.ins().sadd_sat(x, y),
                    VSatBinOp::AddU => b.ins().uadd_sat(x, y),
                    VSatBinOp::SubS => b.ins().ssub_sat(x, y),
                    VSatBinOp::SubU => b.ins().usub_sat(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VWiden { shape, op, a } => {
                // The source is the half-width shape; widen low/high → the wide result.
                let src_ty = vec_ty(shape.narrower().expect("verifier ensures a narrower shape"));
                let x = vcast(b, get(&vals, *a)?, src_ty);
                let (low, signed) = op.parts();
                let r = match (low, signed) {
                    (true, true) => b.ins().swiden_low(x),
                    (false, true) => b.ins().swiden_high(x),
                    (true, false) => b.ins().uwiden_low(x),
                    (false, false) => b.ins().uwiden_high(x),
                };
                vcast(b, r, I8X16)
            }
            Inst::VNarrow {
                shape,
                op,
                a,
                b: rb,
            } => {
                // The sources are the wider shape; `snarrow`/`unarrow` saturate `a`'s lanes then
                // `b`'s into the narrow result.
                let src_ty = vec_ty(shape.wider().expect("verifier ensures a wider source"));
                let x = vcast(b, get(&vals, *a)?, src_ty);
                let y = vcast(b, get(&vals, *rb)?, src_ty);
                let r = match op {
                    VNarrowOp::S => b.ins().snarrow(x, y),
                    VNarrowOp::U => b.ins().unarrow(x, y),
                };
                vcast(b, r, I8X16)
            }
            Inst::VConvert { op, a } => {
                let raw = get(&vals, *a)?;
                let r = match op {
                    VCvtOp::F32x4ConvertI32x4S => {
                        let x = vcast(b, raw, I32X4);
                        b.ins().fcvt_from_sint(F32X4, x)
                    }
                    VCvtOp::F32x4ConvertI32x4U => {
                        let x = vcast(b, raw, I32X4);
                        b.ins().fcvt_from_uint(F32X4, x)
                    }
                    VCvtOp::I32x4TruncSatF32x4S => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fcvt_to_sint_sat(I32X4, x)
                    }
                    VCvtOp::I32x4TruncSatF32x4U => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fcvt_to_uint_sat(I32X4, x)
                    }
                    VCvtOp::F32x4DemoteF64x2Zero => {
                        let x = vcast(b, raw, F64X2);
                        b.ins().fvdemote(x)
                    }
                    VCvtOp::F64x2PromoteLowF32x4 => {
                        let x = vcast(b, raw, F32X4);
                        b.ins().fvpromote_low(x)
                    }
                    // Lane-count changes (2↔4). Widen/narrow through the i64x2 intermediate, the
                    // same recipe Cranelift's own wasm frontend uses.
                    VCvtOp::F64x2ConvertLowI32x4S => {
                        let x = vcast(b, raw, I32X4);
                        let w = b.ins().swiden_low(x); // low 2 i32 → i64x2
                        b.ins().fcvt_from_sint(F64X2, w)
                    }
                    VCvtOp::F64x2ConvertLowI32x4U => {
                        let x = vcast(b, raw, I32X4);
                        let w = b.ins().uwiden_low(x); // low 2 u32 → i64x2
                        b.ins().fcvt_from_uint(F64X2, w)
                    }
                    VCvtOp::I32x4TruncSatF64x2SZero => {
                        let x = vcast(b, raw, F64X2);
                        let conv = b.ins().fcvt_to_sint_sat(I64X2, x); // i64x2
                        let zc = b
                            .func
                            .dfg
                            .constants
                            .insert(ConstantData::from(&[0u8; 16][..]));
                        let zero = b.ins().vconst(I64X2, zc);
                        // snarrow packs [conv | zero] → i32x4: low 2 lanes = conv, high 2 = 0.
                        b.ins().snarrow(conv, zero)
                    }
                    VCvtOp::I32x4TruncSatF64x2UZero => {
                        let x = vcast(b, raw, F64X2);
                        let conv = b.ins().fcvt_to_uint_sat(I64X2, x); // i64x2
                        let zc = b
                            .func
                            .dfg
                            .constants
                            .insert(ConstantData::from(&[0u8; 16][..]));
                        let zero = b.ins().vconst(I64X2, zc);
                        b.ins().uunarrow(conv, zero)
                    }
                };
                vcast(b, r, I8X16)
            }
            // Boolean reductions → an `i32`. `vany_true`/`vall_true` yield an `I8` bool (zero/one),
            // widened to `i32`; `vhigh_bits` produces the bitmask directly into an `i32`.
            Inst::VAnyTrue { a } => {
                let x = get(&vals, *a)?; // shape-agnostic; the canonical I8X16 view is fine
                let t = b.ins().vany_true(x);
                b.ins().uextend(I32, t)
            }
            Inst::VAllTrue { shape, a } => {
                let x = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                let t = b.ins().vall_true(x);
                b.ins().uextend(I32, t)
            }
            Inst::VBitmask { shape, a } => {
                let x = vcast(b, get(&vals, *a)?, vec_ty(*shape));
                b.ins().vhigh_bits(I32, x)
            }
            Inst::VFloatCmp {
                shape,
                op,
                a,
                b: rb,
            } => {
                // Vector `fcmp` yields a per-lane all-ones/all-zeros mask — the wasm/interp semantics.
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                let cc = match op {
                    VFCmpOp::Eq => FloatCC::Equal,
                    VFCmpOp::Ne => FloatCC::NotEqual,
                    VFCmpOp::Lt => FloatCC::LessThan,
                    VFCmpOp::Gt => FloatCC::GreaterThan,
                    VFCmpOp::Le => FloatCC::LessThanOrEqual,
                    VFCmpOp::Ge => FloatCC::GreaterThanOrEqual,
                };
                let r = b.ins().fcmp(cc, x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFloatBin {
                shape,
                op,
                a,
                b: rb,
            } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let y = vcast(b, get(&vals, *rb)?, ty);
                // Reuse the scalar float lowering — Cranelift's `fadd`/`fmin`/… are polymorphic over
                // scalar and vector, so lanes lower the same way scalars do.
                let r = float_bin(b, vf_bin(*op), x, y);
                vcast(b, r, I8X16)
            }
            Inst::VFloatUn { shape, op, a } => {
                let ty = vec_ty(*shape);
                let x = vcast(b, get(&vals, *a)?, ty);
                let r = float_un(b, vf_un(*op), x);
                vcast(b, r, I8X16)
            }
            Inst::VPMinMax {
                shape,
                op,
                a,
                b: rb,
            } => {
                // pmin(a,b) = b < a ? b : a ; pmax(a,b) = a < b ? b : a.
                // Both select the second operand `b` where a one-sided `<` holds — only the
                // compare operand order differs. `fcmp` yields the lane mask (`I32X4`/`I64X2`),
                // which we blend in the canonical `I8X16` domain so `bitselect`'s three args
                // share a type. This matches the interp's NaN/-0 propagation (no IEEE min/max).
                let ty = vec_ty(*shape);
                let xc = get(&vals, *a)?;
                let yc = get(&vals, *rb)?;
                let x = vcast(b, xc, ty);
                let y = vcast(b, yc, ty);
                let mask = match op {
                    VPMinMaxOp::Pmin => b.ins().fcmp(FloatCC::LessThan, y, x),
                    VPMinMaxOp::Pmax => b.ins().fcmp(FloatCC::LessThan, x, y),
                };
                let m = vcast(b, mask, I8X16);
                b.ins().bitselect(m, yc, xc)
            }
            Inst::VBitBin { op, a, b: rb } => {
                // Whole-vector — operate on the canonical I8X16 directly.
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                match op {
                    VBitBinOp::And => b.ins().band(x, y),
                    VBitBinOp::Or => b.ins().bor(x, y),
                    VBitBinOp::Xor => b.ins().bxor(x, y),
                    VBitBinOp::AndNot => b.ins().band_not(x, y),
                }
            }
            Inst::VNot { a } => b.ins().bnot(get(&vals, *a)?),
            Inst::Bitselect { a, b: rb, mask } => {
                // IR `(a & mask) | (b & !mask)` == Cranelift `bitselect(mask, a, b)`.
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                let m = get(&vals, *mask)?;
                b.ins().bitselect(m, x, y)
            }
            Inst::Shuffle { lanes, a, b: rb } => {
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                let imm = b.func.dfg.immediates.push(ConstantData::from(&lanes[..]));
                b.ins().shuffle(x, y, imm)
            }
            Inst::Swizzle { a, b: rb } => {
                let x = get(&vals, *a)?;
                let y = get(&vals, *rb)?;
                b.ins().swizzle(x, y)
            }
            Inst::Load {
                op, addr, offset, ..
            } => {
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, op.info().2, size);
                let a = get(&vals, *addr)?;
                let phys = if elide {
                    mask_addr(b, lower, a, *offset, true, op.info().2)
                } else {
                    checked.confine(b, lower, *addr, a, *offset, op.info().2)
                };
                emit_page_check(b, lower, phys, op.info().2, 0);
                lower_load(b, *op, phys)
            }
            Inst::Store {
                op,
                addr,
                value,
                offset,
                ..
            } => {
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, op.info().2, size);
                let a = get(&vals, *addr)?;
                let phys = if elide {
                    mask_addr(b, lower, a, *offset, true, op.info().2)
                } else {
                    checked.confine(b, lower, *addr, a, *offset, op.info().2)
                };
                emit_page_check(b, lower, phys, op.info().2, PAGE_MISS_WRITE);
                lower_store(b, *op, phys, get(&vals, *value)?);
                continue; // store produces no value
            }
            // Bulk-memory ops (D62). Each span is confined once (single range check + base clamp),
            // then the copy/fill is the platform libcall — `memory.copy`-class lowering, not a
            // per-byte confined loop. `MemCopy`/`MemMove` differ only in the libcall (overlap safety).
            // #2177: a short copy of one of this block's constant lengths is loads and stores instead
            // ([`copy_inline`]); a zero-length one is inert, as the libcall path makes it.
            Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
                let short = checked.consts.get(len).copied();
                if let Some(n) =
                    short.filter(|n| lower.mask != 0 && (0..=INLINE_COPY_MAX).contains(n))
                {
                    if n > 0 {
                        copy_inline(b, lower, &mut checked, &ubs, &vals, *dst, *src, n as u32)?;
                    }
                    continue;
                }
                let n = get(&vals, *len)?;
                let dphys = confine_span(b, lower, get(&vals, *dst)?, n);
                let sphys = confine_span(b, lower, get(&vals, *src)?, n);
                // #2164: the source's pages, then the destination's, as the interpreters check them.
                emit_span_page_check(b, lower, sphys, n, 0);
                emit_span_page_check(b, lower, dphys, n, PAGE_MISS_WRITE);
                // I21: fault an overrun before the libcall (no partial write; catches `dst==src`).
                probe_span(b, dphys, n);
                probe_span(b, sphys, n);
                if matches!(inst, Inst::MemCopy { .. }) {
                    b.call_memcpy(lower.frontend_config, dphys, sphys, n);
                } else {
                    b.call_memmove(lower.frontend_config, dphys, sphys, n);
                }
                continue;
            }
            Inst::MemFill { dst, val, len } => {
                let n = get(&vals, *len)?;
                let dphys = confine_span(b, lower, get(&vals, *dst)?, n);
                emit_span_page_check(b, lower, dphys, n, PAGE_MISS_WRITE); // #2164
                probe_span(b, dphys, n);
                // `call_memset` uextends the fill byte to i32, so hand it the low byte (i8).
                let v8 = b.ins().ireduce(I8, get(&vals, *val)?);
                b.call_memset(lower.frontend_config, dphys, v8, n);
                continue;
            }
            // §12 atomics. Confine like a normal access, then a natural-alignment guard (a misaligned
            // address traps — `atomic_*` require alignment, and it matches the interpreter), then a
            // hardware atomic. Elision uses the same upper-bound analysis.
            // The `order` is ignored: Cranelift atomics are seq-cst, which soundly implements every
            // requested ordering and keeps the interpreter↔JIT oracle exact (see `temen_ir::Ordering`).
            Inst::AtomicLoad {
                ty, addr, offset, ..
            } => {
                let w = atomic_width(*ty);
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                emit_page_check(b, lower, phys, w, 0);
                b.ins().atomic_load(int_clif_ty(*ty), atomic_flags(), phys)
            }
            Inst::AtomicStore {
                ty,
                addr,
                value,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                emit_page_check(b, lower, phys, w, PAGE_MISS_WRITE);
                b.ins()
                    .atomic_store(atomic_flags(), get(&vals, *value)?, phys);
                continue; // atomic store produces no value
            }
            Inst::AtomicRmw {
                ty,
                op,
                addr,
                value,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w);
                emit_page_check(b, lower, phys, w, PAGE_MISS_WRITE);
                b.ins().atomic_rmw(
                    int_clif_ty(*ty),
                    atomic_flags(),
                    clif_rmw_op(*op),
                    phys,
                    get(&vals, *value)?,
                )
            }
            Inst::AtomicCmpxchg {
                ty,
                addr,
                expected,
                replacement,
                offset,
                ..
            } => {
                let w = atomic_width(*ty);
                debug_assert_eq!(
                    ubs.len(),
                    vals.len(),
                    "ubs/vals lockstep (mask-elision, #893)"
                );
                let elide = in_window(ub_at(&ubs, *addr), *offset, w, size);
                let phys = mask_addr(b, lower, get(&vals, *addr)?, *offset, elide, w);
                guard_atomic_align(b, lower, phys, w); // type is inferred from the operands
                emit_page_check(b, lower, phys, w, PAGE_MISS_WRITE);
                b.ins().atomic_cas(
                    atomic_flags(),
                    phys,
                    get(&vals, *expected)?,
                    get(&vals, *replacement)?,
                )
            }
            // A funcref is just the function index as plain i32 data (§3c) — the same
            // value the interpreter materializes; `call.dyn` masks it into the table.
            Inst::RefFunc { func } => {
                // A submitted unit's own-function address must be a real shared-table slot (the unit
                // is auto-installed there — DESIGN.md §22 "unit-own funcref"), so it resolves to the
                // unit's function through the ordinary masked dispatch instead of the parent's table.
                // Top-level compiles have no remap: `ref.func N` is module-0 slot N verbatim.
                let idx = match lower.ref_slots {
                    Some(slots) => *slots.get(*func as usize).unwrap_or(func),
                    None => *func,
                };
                b.ins().iconst(I32, idx as i64)
            }
            // §12 standalone fence. Cranelift emits a full (seq-cst) barrier regardless of the
            // requested `order` — the same sound strengthening the atomics use.
            Inst::AtomicFence { .. } => {
                b.ins().fence();
                continue; // produces no value
            }
            _ => return Err(JitError::Unsupported("instruction")),
        };
        // Single-result instruction: record its value and a sound upper bound in lockstep.
        let u = ub_of(inst, &ubs);
        checked.note(vals.len() as u32, inst);
        vals.push(v);
        ubs.push(u);
    }

    // W5 JIT/DWARF Stage 3a: now that this block's value map is fully populated, stamp the CLIF
    // values backing source variables with their `ValueLabel`, so Cranelift records the
    // machine-location ranges (`value_labels_ranges`) `compile` reads back. `set_val_label` is inert
    // unless `collect_debug_info` was enabled (it is, exactly when `var_labels` is `Some`), so
    // non-`-g` codegen is untouched. The label association drives liveness-based range computation;
    // the block-local `value` index maps straight onto `vals`.
    if let Some(map) = lower.var_labels {
        if let Some(points) = map.get(&(lower.func_idx, block_idx as u32)) {
            for &(value, label) in points {
                if let Some(&v) = vals.get(value as usize) {
                    b.set_val_label(v, ValueLabel::from_u32(label));
                }
            }
        }
    }

    match &blk.term {
        Terminator::Br { target, args } => {
            let ba = map_edge(&vals, blocks, *target, args)?;
            let t = *blocks
                .clif
                .get(*target as usize)
                .ok_or(JitError::Malformed)?;
            // §5 kill-path: poll the interrupt cell before taking any branch — every loop body ends
            // in one of these terminators, so this bounds a non-terminating intra-function loop.
            emit_epoch_check(b, lower);
            if *target as usize <= block_idx {
                emit_domain_poll(b, lower);
            }
            emit_signal_check(module, b, lower)?; // #932 async signal delivery (no-op when disarmed)
                                                  // Fuel unification: an unconditional branch charges one fuel iff it is a back-edge — the
                                                  // target block index is at-or-before this block's, matching the interpreters' `backedge!`
                                                  // (`$t <= pc`). Back-edge-ness is fully static here (one target).
            if *target as usize <= block_idx {
                emit_fuel_check(b, lower);
            }
            b.ins().jump(t, &ba);
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let c = get(&vals, *cond)?;
            let ta = map_edge(&vals, blocks, *then_blk, then_args)?;
            let ea = map_edge(&vals, blocks, *else_blk, else_args)?;
            let tb = *blocks
                .clif
                .get(*then_blk as usize)
                .ok_or(JitError::Malformed)?;
            let eb = *blocks
                .clif
                .get(*else_blk as usize)
                .ok_or(JitError::Malformed)?;
            emit_epoch_check(b, lower); // §5 kill-path (see `Br`)
            if *then_blk as usize <= block_idx || *else_blk as usize <= block_idx {
                emit_domain_poll(b, lower); // (see `Br`)
            }
            emit_signal_check(module, b, lower)?; // #932 async signal delivery (no-op when disarmed)
                                                  // Fuel unification: charge one fuel iff the *taken* edge is a back-edge (target block index
                                                  // <= this block's), matching the interpreters' per-edge `backedge!`. Back-edge-ness is
                                                  // static per edge, but which edge is taken is a runtime value — so an edge that is a
                                                  // back-edge *alone* is charged via a trampoline block entered only when that edge is taken.
                                                  // Gated on `fuel_addr != 0` so an un-armed compile emits the identical plain `brif`.
            let then_be = lower.fuel && *then_blk as usize <= block_idx;
            let else_be = lower.fuel && *else_blk as usize <= block_idx;
            match (then_be, else_be) {
                // Neither edge charges (or fuel un-armed) — the plain conditional branch.
                (false, false) => {
                    b.ins().brif(c, tb, &ta, eb, &ea);
                }
                // Both edges charge → charge unconditionally before the branch (either way is a
                // back-edge), avoiding two trampolines.
                (true, true) => {
                    emit_fuel_check(b, lower);
                    b.ins().brif(c, tb, &ta, eb, &ea);
                }
                // Only the then-edge is a back-edge: route it through a trampoline that charges then
                // jumps to the real target; the else-edge branches direct.
                (true, false) => {
                    let tramp = b.create_block();
                    b.ins().brif(c, tramp, &[], eb, &ea);
                    b.switch_to_block(tramp);
                    emit_fuel_check(b, lower);
                    b.ins().jump(tb, &ta);
                }
                // Symmetric: only the else-edge is a back-edge.
                (false, true) => {
                    let tramp = b.create_block();
                    b.ins().brif(c, tb, &ta, tramp, &[]);
                    b.switch_to_block(tramp);
                    emit_fuel_check(b, lower);
                    b.ins().jump(eb, &ea);
                }
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let index = get(&vals, *idx)?;
            emit_epoch_check(b, lower); // §5 kill-path (see `Br`)
            if default.0 as usize <= block_idx
                || targets.iter().any(|(t, _)| *t as usize <= block_idx)
            {
                emit_domain_poll(b, lower); // (see `Br`)
            }
            emit_signal_check(module, b, lower)?; // #932 async signal delivery (no-op when disarmed)
                                                  // Fuel unification: charge one fuel iff the *selected* target is a back-edge (target block
                                                  // index <= this block's), matching the interpreters' `backedge!` on the arm actually taken.
                                                  // Back-edge-ness is static per arm; which arm is selected is runtime. So: no arm a back-edge
                                                  // ⇒ no charge; every arm a back-edge ⇒ charge once up front; mixed ⇒ route each back-edge arm
                                                  // through a trampoline that charges then jumps, leaving forward arms direct. Gated on
                                                  // `fuel_addr != 0` so an un-armed compile builds the identical table.
            let is_be = |t: u32| lower.fuel && t as usize <= block_idx;
            let all_be = is_be(default.0) && targets.iter().all(|(t, _)| is_be(*t));
            let any_be = is_be(default.0) || targets.iter().any(|(t, _)| is_be(*t));
            if all_be {
                emit_fuel_check(b, lower);
            }
            let mixed = any_be && !all_be; // some arms back-edges, some not
                                           // Build a BlockCall (target block + its edge args) for each table entry and the default;
                                           // Cranelift masks the index and selects, default on OOB. A `(tramp, dest, args)` is deferred
                                           // and filled after the `br_table` (a block can only be built once its predecessors exist).
            let mut tramps: Vec<(
                cranelift_codegen::ir::Block,
                cranelift_codegen::ir::Block,
                Vec<BlockArg>,
            )> = Vec::new();
            let mut entries = Vec::with_capacity(targets.len());
            for (t, args) in targets {
                let ba = map_edge(&vals, blocks, *t, args)?;
                let blk = *blocks.clif.get(*t as usize).ok_or(JitError::Malformed)?;
                if mixed && is_be(*t) {
                    let tramp = b.create_block();
                    entries.push(BlockCall::new(
                        tramp,
                        core::iter::empty::<BlockArg>(),
                        &mut b.func.dfg.value_lists,
                    ));
                    tramps.push((tramp, blk, ba));
                } else {
                    entries.push(BlockCall::new(
                        blk,
                        ba.iter().copied(),
                        &mut b.func.dfg.value_lists,
                    ));
                }
            }
            let (dt, dargs) = default;
            let dba = map_edge(&vals, blocks, *dt, dargs)?;
            let dblk = *blocks.clif.get(*dt as usize).ok_or(JitError::Malformed)?;
            let dcall = if mixed && is_be(*dt) {
                let tramp = b.create_block();
                let call = BlockCall::new(
                    tramp,
                    core::iter::empty::<BlockArg>(),
                    &mut b.func.dfg.value_lists,
                );
                tramps.push((tramp, dblk, dba));
                call
            } else {
                BlockCall::new(dblk, dba.iter().copied(), &mut b.func.dfg.value_lists)
            };
            let jt = b.create_jump_table(JumpTableData::new(dcall, &entries));
            b.ins().br_table(index, jt);
            // Fill the back-edge trampolines: charge one fuel, then jump to the real target with the
            // arm's edge args (they come from this block's `vals`, which dominate the trampoline).
            for (tramp, dest, args) in tramps {
                b.switch_to_block(tramp);
                emit_fuel_check(b, lower);
                b.ins().jump(dest, &args);
            }
        }
        Terminator::Return(outs) => {
            let rets: Vec<Value> = outs
                .iter()
                .map(|o| get(&vals, *o))
                .collect::<Result<_, _>>()?;
            if let Some(sret_var) = lower.sret_var {
                // sret ABI: write each result into the caller-provided return-area (8-byte slots,
                // `encode_slot`-encoded like the buffer ABI), then return void.
                let ptr = b.use_var(sret_var);
                for (i, r) in rets.iter().enumerate() {
                    let slot = encode_slot(b, *r);
                    b.ins()
                        .store(MemFlags::trusted(), slot, ptr, (i * 8) as i32);
                }
                b.ins().return_(&[]);
            } else {
                // Natural ABI: return the result values directly (CLIF multi-return).
                b.ins().return_(&rets);
            }
        }
        Terminator::ReturnCall { func, args } => {
            // Tail call (§3b): replace this frame with the callee, threading the context.
            let callee_id = *lower.ids.get(*func as usize).ok_or(JitError::Malformed)?;
            let callee = module.declare_func_in_func(callee_id, b.func);
            let mut cargs = ctx_args(b, lower);
            // The tail callee shares this function's result type (verifier-enforced), so its sret-ness
            // matches ours: when we return via sret, forward our own return-area pointer.
            if let Some(sret_var) = lower.sret_var {
                cargs.push(b.use_var(sret_var));
            }
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            b.ins().return_call(callee, &cargs);
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            // Indirect tail call: table dispatch (§3c) then a guaranteed tail call.
            let ty = sig_of(lower.type_section, *ty); // #922: resolve interned call type index
            let code = indirect_dispatch(b, lower, get(&vals, *idx)?, ty);
            let sig = b.import_signature(sig_from(module, &ty.params, &ty.results));
            let mut cargs = ctx_args(b, lower);
            // `ty.results` equals this function's results (tail-call contract), so sret-ness matches:
            // forward our return-area pointer when we return via sret.
            if let Some(sret_var) = lower.sret_var {
                cargs.push(b.use_var(sret_var));
            }
            for a in args {
                cargs.push(get(&vals, *a)?);
            }
            b.ins().return_call_indirect(sig, code, &cargs);
        }
        Terminator::Unreachable => {
            emit_trap(b, lower, TrapKind::Unreachable);
        }
    }
    blocks.vals[block_idx] = vals;
    Ok(())
}

/// The leading context arguments threaded into every guest call: `(mem_base,
/// fn_table_base, vmctx, stack_limit)`.
/// Load one of the instance's per-run addresses from its [`VmCtx`] (a field offset from
/// [`vmctx`](crate::vmctx)). The fields are written before an entry and never during it, so the load is
/// `readonly` + `can_move`: Cranelift may hoist it out of a loop or share it between uses — the
/// address is loop-invariant; what it points *at* (the interrupt cell, the armed flag) is still read
/// by the caller with its own ordering.
fn vmctx_load(b: &mut FunctionBuilder, lower: &Lower, field: i32) -> Value {
    let vmctx = b.use_var(lower.vmctx_var);
    let flags = MemFlags::trusted().with_readonly().with_can_move();
    b.ins().load(I64, flags, vmctx, field)
}

fn ctx_args(b: &mut FunctionBuilder, lower: &Lower) -> Vec<Value> {
    vec![
        b.use_var(lower.mem_var),
        b.use_var(lower.fn_table_var),
        b.use_var(lower.vmctx_var),
        // §2b path B: pass our own stack-limit on to the callee (same stack, same limit) — mirrors
        // sig_from. Constant within a stack's call tree; set anew at each fiber/root entry.
        b.use_var(lower.limit_var),
    ]
}

/// For a callee whose `results` use the sret ABI ([`uses_sret`]), allocate a stack **return-area**,
/// push its address onto `cargs` (right after the context pointers, before the user args — the order
/// [`sig_from`] lays out), and return the slot; else push nothing and return `None`.
fn sret_call_slot(
    b: &mut FunctionBuilder,
    cargs: &mut Vec<Value>,
    results: &[ValType],
) -> Option<cranelift_codegen::ir::StackSlot> {
    if !uses_sret(results) {
        return None;
    }
    let ss = b.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (results.len() * 8) as u32,
        3, // 8-byte aligned slots
    ));
    let addr = b.ins().stack_addr(I64, ss, 0);
    cargs.push(addr);
    Some(ss)
}

/// Append a call's results to `vals`: from registers ([`FunctionBuilder::inst_results`]) on the
/// normal path, or by loading the sret return-area's 8-byte slots ([`decode_slot`]-encoded, matching
/// the storing `Return`) on the sret path.
fn read_call_results(
    b: &mut FunctionBuilder,
    call: cranelift_codegen::ir::Inst,
    sret: Option<cranelift_codegen::ir::StackSlot>,
    results: &[ValType],
    vals: &mut Vec<Value>,
) {
    match sret {
        None => vals.extend_from_slice(b.inst_results(call)),
        Some(ss) => {
            for (i, r) in results.iter().enumerate() {
                let raw = b.ins().stack_load(I64, ss, (i * 8) as i32);
                vals.push(decode_slot(b, raw, *r));
            }
        }
    }
}

/// Lower a trap (§5 detect-and-kill): store the kind code into the host trap cell, then
/// `return` dummy zero results so the run unwinds to the trampoline, which reports the
/// trap. (The reference JIT detects traps this way; production uses hardware faults.)
///
/// Caveat: this returns from the *current* function only. The current scalar tests put
/// every trap in the entry function (or its dispatch), so that suffices; propagating a
/// trap *out of a callee* would need a post-call check, added when a case needs it.
fn emit_trap(b: &mut FunctionBuilder, lower: &Lower, kind: TrapKind) {
    emit_trap_set(b, lower, kind);
    emit_trap_return(b, lower);
}

/// Record a trap in the cell (with the §5 W3 backtrace capture) **without** terminating the block.
/// Callers pair it with [`emit_trap_return`] (an unconditional trap, via [`emit_trap`]) or with
/// [`emit_trap_propagate`] (the shared branch-on-cell exit) when compilation of the rest of the
/// block must continue — the branch keeps the continuation formally reachable, so later
/// instructions still lower against a well-formed value stack even though they are dead at runtime.
fn emit_trap_set(b: &mut FunctionBuilder, lower: &Lower, kind: TrapKind) {
    // §5 W3 Stage 2: record a source backtrace for this explicit trap *before* it unwinds — the trap
    // stores its kind and returns, and that return propagates up tearing down every guest frame, so
    // the helper must walk the frame-pointer chain from this live site now. The current op's
    // `SourceLoc` is in effect, so the call inherits it and symbolizes to the trapping op's line.
    // Only present under `-g` (else the backtrace would be empty); the trap path is otherwise
    // unchanged.
    if let Some(capture) = lower.trap_capture {
        // Pass the trapping function's frame pointer so the helper can walk the caller chain without
        // `__builtin_frame_address` (which MSVC lacks); it pairs `fp` with its own return address (the
        // trap site) for the innermost frame.
        let fp = b.ins().get_frame_pointer(I64);
        b.ins().call(capture, &[fp]);
    }
    let cell = b.use_var(lower.vmctx_var);
    let none = b.ins().iconst(I64, 0);
    let code = b.ins().iconst(I64, kind as u32 as i64); // full i64 cell (high bits 0)
                                                        // #2499 — raise, never store: the cell holds the domain's end, and an end a sibling vCPU raised
                                                        // first stands ([`VmCtx::raise`]).
    b.ins().atomic_cas(atomic_flags(), cell, none, code);
}

/// Emit the function `return` on a trap / early-exit path: **void** for an sret function (its results
/// flow through the return-area pointer, left unwritten — the trap cell is set, so the caller returns
/// before reading them), else dummy zeros of the result types (the register-ABI convention).
fn emit_trap_return(b: &mut FunctionBuilder, lower: &Lower) {
    if lower.sret_var.is_some() {
        b.ins().return_(&[]);
    } else {
        let zeros: Vec<Value> = lower.result_tys.iter().map(|t| zero_of(b, *t)).collect();
        b.ins().return_(&zeros);
    }
}

/// Emit the §5 fuel/epoch **kill-path** check: if the host has set the interrupt cell non-zero
/// (a watchdog timer, a cross-domain preemption, …), trap [`TrapKind::OutOfFuel`]; otherwise fall
/// through. A no-op when no kill-path is armed (`epoch_addr == 0`) — then the guest code is emitted
/// exactly as before. Placed at function entry and every loop back-edge, so any non-terminating
/// guest polls the cell within a bounded number of steps and is stopped — the guest can't disable
/// the poll (only the host writes the cell). The load is **not** `readonly`, so it is re-evaluated
/// each iteration (never hoisted out of the loop); it never faults (a host-owned aligned cell).
///
/// On return the builder is positioned at a fresh continuation block (the not-interrupted path);
/// the caller emits the real terminator / jump there.
fn emit_epoch_check(b: &mut FunctionBuilder, lower: &Lower) {
    if !lower.epoch {
        return; // no kill-path armed for this compile — emit nothing
    }
    let cont = b.create_block();
    let trap_blk = b.create_block();
    let addr = vmctx_load(b, lower, vmctx::EPOCH);
    // The host stores into this cell **concurrently** (the watchdog thread), so the poll must reload
    // it every check. An **atomic** load is the reliable way to say so: under `opt_level=speed`
    // Cranelift's alias analysis sees no *guest* store to the cell and would hoist/CSE a plain load out
    // of the loop (the poll would fire once and never again — the runaway would never be killed); an
    // atomic load is a synchronization op the optimizer won't hoist. (The cell is a host `AtomicU64`.)
    let flag = b.ins().atomic_load(I64, atomic_flags(), addr);
    b.ins().brif(flag, trap_blk, &[], cont, &[]);
    b.switch_to_block(trap_blk);
    emit_trap(b, lower, TrapKind::OutOfFuel);
    b.switch_to_block(cont);
    // `cont`/`trap_blk` are sealed by the caller's `seal_all_blocks`.
}

/// Emit the domain-teardown poll ([`Lower::domain_poll`]) — at loop back-edges and at the entry of
/// a function that tail-calls: if this vCPU's domain has ended — a sibling's trap or `exit`, the
/// root's completion, or a parent's `kill` or teardown, all raised on the trap cell — return
/// straight up the stack, as a call's trap-propagation guard does. The cell is only read, so the
/// value that ended the domain is the one reported. The load is atomic for the reason
/// [`emit_epoch_check`]'s is: another thread writes the cell, and a plain load could be hoisted out
/// of the loop it bounds. A no-op when `domain_poll` is off.
///
/// On return the builder is positioned at the continuation block.
fn emit_domain_poll(b: &mut FunctionBuilder, lower: &Lower) {
    if !lower.domain_poll {
        return;
    }
    let cell = b.use_var(lower.vmctx_var);
    let ended = b.ins().atomic_load(I64, atomic_flags(), cell);
    let stop = b.create_block();
    let cont = b.create_block();
    b.set_cold_block(stop); // taken once per vCPU lifetime: keep it off the loop's fall-through path
    b.ins().brif(ended, stop, &[], cont, &[]);
    b.switch_to_block(stop);
    emit_trap_return(b, lower);
    b.switch_to_block(cont);
    // `stop`/`cont` are sealed by the caller's `seal_all_blocks`.
}

/// #932 — emit the async-signal **delivery safepoint**: poll the host `armed` flag
/// (`Host::sig_armed`) and, when set, call out to the host's decision thunk; if it returns a
/// deliverable signal, redirect into the guest handler `void(int)` (dispatched through the §3c
/// function table like a `call.dyn`) on the guest-registered signal stack, then notify
/// handler-return and **re-poll** (so a signal that arrived during the handler, or a queued one
/// exposed by unblocking, is delivered before continuing). A no-op when no signal delivery is armed
/// ([`Lower::signals`]) — then the guest code is byte-identical to the un-armed build. Placed at
/// the same safepoints as [`emit_epoch_check`] (function entry + every branch), so a compute-bound
/// guest polls within a bounded number of steps and delivers its handler mid-loop — the JIT twin of
/// the interpreters' per-op redirect (`#796` / `#1146`). The **decision** (nesting bound + consume)
/// and **return** bookkeeping (mask restore) live in the shared `SignalSource` behind the thunks,
/// so delivery is byte-identical to the interpreter under the differential oracle (DESIGN.md §18);
/// only the redirect codegen is here.
///
/// On return the builder is positioned at a fresh continuation block (the not-delivered path); the
/// caller emits the real terminator / next check there. The armed flag is loaded **atomically** (as
/// [`emit_epoch_check`] does) so Cranelift's alias analysis cannot hoist the poll out of a loop — the
/// host sets it from another thread and the guest never stores it.
fn emit_signal_check(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
) -> Result<(), JitError> {
    if !lower.signals {
        return Ok(()); // no signal delivery armed for this compile — emit nothing
    }
    let poll = b.create_block(); // re-poll head (revisited after a handler returns)
    let cont = b.create_block(); // the not-delivered continuation
    b.ins().jump(poll, &[]);
    b.switch_to_block(poll);
    // Poll the host `AtomicBool` armed flag. Atomic load (like the epoch cell) so it is re-read every
    // iteration — a plain load would be hoisted out of the loop and the poll would fire only once.
    let armed_addr = vmctx_load(b, lower, vmctx::SIG_ARMED);
    let armed = b.ins().atomic_load(I8, atomic_flags(), armed_addr);
    let maybe = b.create_block();
    b.ins().brif(armed, maybe, &[], cont, &[]);

    // Armed: ask the host decision thunk `fn(ctx, *mut [i64;3]) -> i32` for a deliverable signal.
    b.switch_to_block(maybe);
    let out_ss = b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 24, 3)); // [fref,signum,sp]
    let out = b.ins().stack_addr(I64, out_ss, 0);
    let ctx = vmctx_load(b, lower, vmctx::SIG_CTX);
    let got = call_host(module, b, HostFn::SigTake, &[ctx, out])?;
    let got = b.inst_results(got)[0];
    let deliver = b.create_block();
    b.ins().brif(got, deliver, &[], cont, &[]);

    // Deliver: read `(handler_funcref, signum, handler_sp)` and dispatch the guest handler
    // `void(int)` — IR type `(i64 sp, i32 signum) -> ()` — through the §3c table (the `call.dyn`
    // machinery), on the guest's registered signal stack `sp` (the handler's v0). The interrupted
    // safepoint is NOT retired, so control resumes here (and re-polls) after the handler returns.
    b.switch_to_block(deliver);
    let fref64 = b.ins().load(I64, MemFlags::trusted(), out, 0);
    let fref = b.ins().ireduce(I32, fref64);
    let signum64 = b.ins().load(I64, MemFlags::trusted(), out, 8);
    let signum = b.ins().ireduce(I32, signum64);
    let sp = b.ins().load(I64, MemFlags::trusted(), out, 16);
    let handler_ty = FuncType {
        params: vec![ValType::I64, ValType::I32],
        results: vec![],
    };
    let code = indirect_dispatch(b, lower, fref, &handler_ty);
    let hsig = b.import_signature(sig_from(module, &handler_ty.params, &handler_ty.results));
    let mut cargs = ctx_args(b, lower);
    cargs.push(sp);
    cargs.push(signum);
    b.ins().call_indirect(hsig, code, &cargs);
    // The handler may `exit`/trap or `longjmp`; propagate a set trap cell now (unwinds this function),
    // exactly as after a `call.dyn`. Leaves the builder in a fresh continuation block.
    emit_trap_propagate(b, lower);
    // Handler returned normally: notify the host (restore the block-during-handler mask, pop nesting).
    let ctx2 = vmctx_load(b, lower, vmctx::SIG_CTX);
    call_host(module, b, HostFn::SigReturn, &[ctx2])?;
    b.ins().jump(poll, &[]); // re-poll: deliver a nested/queued signal before resuming
    b.switch_to_block(cont);
    // `poll`/`maybe`/`deliver`/`cont` are sealed by the caller's `seal_all_blocks`.
    Ok(())
}

/// Emit the safepoint-anchored **counted-fuel** decrement (INTERP_PERF.md "Fuel unification"): charge
/// one fuel — trap [`TrapKind::OutOfFuel`] if the budget is already exhausted, else store the
/// decremented budget back. A no-op when no fuel is armed (`fuel_addr == 0`) — then guest code is
/// byte-identical to the un-armed build, so an ordinary (non-fuel) run pays nothing. Placed at the
/// [`emit_epoch_check`] sites (function entry + every taken back-edge) **and at every `cont.resume`**
/// (#1642), so `fuel` counts exactly (function entries + back-edges + resumes) executed — what the
/// tree-walker and bytecode engines charge, so `OutOfFuel` is a checked cross-engine parity. A fiber's
/// entry is the one function entry the interpreters do not charge (its starting resume is the whole
/// cost), so `fiber_rt::make_fiber` refunds that prologue's charge.
///
/// Distinct from [`emit_epoch_check`] in two ways: (1) it is the **guest's own** budget, not written
/// by the host from another thread, so a plain (non-atomic) load/store is enough — and cheaper. A
/// domain's vCPUs share the cell, so their plain stores race and their charge is approximate (and
/// vCPUs running at once contend for its cache line, #2488); after a refill the check charges from
/// the allowance it granted, so a sibling's stale `0` in `left` is never read as a spent chain; (2)
/// the store of `fuel-1` back to the same address it just loaded creates a store⇒load dependency that
/// stops Cranelift's alias analysis from treating the load as loop-invariant and hoisting it out (the
/// reason `emit_epoch_check` needs an *atomic* load to achieve the same). Trap when the loaded value
/// is `0` (would underflow), i.e. before the charge that would make it negative — matching the
/// interpreters' `checked_sub(1)` at the safepoint.
fn emit_fuel_check(b: &mut FunctionBuilder, lower: &Lower) {
    if !lower.fuel {
        return; // no fuel armed for this compile — emit nothing
    }
    let cont = b.create_block();
    let refill_blk = b.create_block();
    let trap_blk = b.create_block();
    b.append_block_param(cont, I64);
    let addr = vmctx_load(b, lower, vmctx::FUEL);
    // Plain load (not `readonly`) — the store below writes the same address, so the load is not
    // loop-invariant and is re-evaluated each iteration. `trusted()` = aligned + notrap (a host-owned
    // aligned cell that never faults), no atomic ordering (the cell's `left`, word 0 of a `FuelCell`).
    let fuel = b.ins().load(I64, MemFlags::trusted(), addr, 0);
    // Spent? (`fuel == 0` ⇒ the next charge would underflow) → refill before charging.
    b.ins()
        .brif(fuel, cont, &[BlockArg::from(fuel)], refill_blk, &[]);
    // #1944 slice 3 — the cell's refill (word 1) draws the next chunk from the budget chain into
    // `left`; a `left` still `0` is a spent chain (or a spent fixed allowance), which traps as a spent
    // counter always did. #2147 — the refill is a `PreserveAll` trampoline, so this call clobbers no
    // register: a value live across the safepoint keeps its register, and the function need not hold
    // it in a callee-saved one (saved and restored on every call) just because a refill might happen.
    b.switch_to_block(refill_blk);
    b.set_cold_block(refill_blk);
    let sig = {
        let mut s = Signature::new(cranelift_codegen::isa::CallConv::PreserveAll);
        s.params.push(AbiParam::new(I64)); // the cell
        b.import_signature(s)
    };
    let refill = b
        .ins()
        .load(I64, MemFlags::trusted(), addr, crate::fuel::REFILL_OFF);
    b.ins().call_indirect(sig, refill, &[addr]);
    // What the refill granted, not a reload of `left`: a sibling vCPU's plain store may have put a
    // stale `0` there since (see `fuel_refill`).
    let drawn = b
        .ins()
        .load(I64, MemFlags::trusted(), addr, crate::fuel::GRANTED_OFF);
    b.ins()
        .brif(drawn, cont, &[BlockArg::from(drawn)], trap_blk, &[]);
    b.switch_to_block(trap_blk);
    b.set_cold_block(trap_blk);
    emit_trap(b, lower, TrapKind::OutOfFuel);
    b.switch_to_block(cont);
    // Charge one: store `fuel - 1` back. The store⇒load dependency (same `addr`) is what keeps the
    // load above from being hoisted out of a loop.
    let fuel = b.block_params(cont)[0];
    let one = b.ins().iconst(I64, 1);
    let charged = b.ins().isub(fuel, one);
    b.ins().store(MemFlags::trusted(), charged, addr, 0);
    // `cont`/`refill_blk`/`trap_blk` are sealed by the caller's `seal_all_blocks`.
}

/// The software stack-overflow guard's tunables. See `crates/temen-jit/STACK_GUARD.md`. Under §2b
/// **path B** the per-thread limit is NOT a global cell: it is a value ABI param (`stack_limit`,
/// [`Lower::limit_var`]) threaded through every call and set anew at each fiber entry from
/// `Yielder::stack_low()` — per-vCPU by construction, no cell and no TLS. The prologue check
/// ([`emit_stack_check`]) reads it straight from that register. Always compiled: the guard is in the
/// always-on escape-TCB path (it becomes the sole overflow defense once the arena drops the guard
/// page, so it must not be feature-gated out).
pub(crate) mod stack_check {
    /// Headroom (bytes) the prologue check reserves below the limit. Because the check runs *after* the
    /// machine prologue's `sub rsp` (it's in the entry block), it validates this function's frame
    /// directly; `RED_ZONE` only needs to cover the prologue's pre-check register pushes plus the
    /// check's own scratch. A single frame larger than the whole stack is a residual backstop concern
    /// (STACK_GUARD.md §2b) — Cranelift 0.132 doesn't expose the final frame size to enforce it.
    pub(crate) const RED_ZONE: u64 = 1 << 14; // 16 KiB

    /// #1983 — the stack limit for guest code entered on the calling thread's **own** stack: the root
    /// (the entry trampoline) and a spawned vCPU's top. The thread stack's low bound plus a quarter of
    /// its size, which stays free for the host code a guest call runs, so unbounded recursion traps
    /// `StackOverflow` instead of reaching the OS guard page, where the process aborts. `0` (the check
    /// inert, as a fiber-less stack had it before) when the bounds are unknown or the caller is on
    /// some other stack, such as a host re-entry from inside a guest fiber. The bounds are looked up
    /// once per thread.
    pub(crate) extern "C" fn thread_limit() -> u64 {
        thread_local! {
            static BOUNDS: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
        }
        let (low, high) = BOUNDS.with(|b| {
            b.get().unwrap_or_else(|| {
                let v = stack_bounds();
                b.set(Some(v));
                v
            })
        });
        let here = 0u8;
        let sp = &here as *const u8 as u64;
        if low == 0 || sp <= low || sp > high {
            return 0;
        }
        low + (high - low) / 4
    }

    /// The calling thread's stack as `(low, high)`, or `(0, 0)` where the platform can't say.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    fn stack_bounds() -> (u64, u64) {
        // SAFETY: `attr` is initialised by `pthread_getattr_np` before it is read, and destroyed once.
        unsafe {
            let mut attr = std::mem::zeroed::<libc::pthread_attr_t>();
            if libc::pthread_getattr_np(libc::pthread_self(), &mut attr) != 0 {
                return (0, 0);
            }
            let (mut addr, mut size) = (std::ptr::null_mut(), 0usize);
            let ok = libc::pthread_attr_getstack(&attr, &mut addr, &mut size) == 0;
            libc::pthread_attr_destroy(&mut attr);
            if !ok {
                return (0, 0);
            }
            (addr as u64, addr as u64 + size as u64)
        }
    }

    #[cfg(target_os = "macos")]
    fn stack_bounds() -> (u64, u64) {
        // SAFETY: both calls only read the calling thread's own descriptor.
        unsafe {
            let this = libc::pthread_self();
            let high = libc::pthread_get_stackaddr_np(this) as u64;
            let size = libc::pthread_get_stacksize_np(this) as u64;
            (high - size, high)
        }
    }

    #[cfg(windows)]
    fn stack_bounds() -> (u64, u64) {
        let (mut low, mut high) = (0usize, 0usize);
        // SAFETY: writes the calling thread's stack bounds into the two locals.
        unsafe {
            windows_sys::Win32::System::Threading::GetCurrentThreadStackLimits(&mut low, &mut high)
        };
        (low as u64, high as u64)
    }

    #[cfg(not(any(
        target_os = "linux",
        target_os = "android",
        target_os = "macos",
        windows
    )))]
    fn stack_bounds() -> (u64, u64) {
        (0, 0)
    }
}

/// The per-prologue software stack-limit check — overflow protection for the arena/software-guard
/// fiber model, which drops the per-fiber hardware guard page. Always emitted (escape-TCB). Reads the
/// running fiber's low bound from our own `stack_limit` ABI param ([`Lower::limit_var`], §2b path B —
/// per-vCPU by construction, no cell and no TLS), gets the current SP, and traps
/// [`TrapKind::StackOverflow`] if `SP - RED_ZONE < limit` — i.e. this function's frame would grow the
/// native stack past the fiber's low bound. The check sits in the entry block, *after* the machine
/// prologue's `sub rsp`, so `get_stack_pointer` already reflects this frame and it is validated
/// directly (soundness relies on Cranelift not page-probing during `sub rsp` — `enable_probestack` is
/// off; see the ISA flags). The root and a spawned vCPU's top take their thread's limit
/// ([`stack_check::thread_limit`], #1983). `limit == 0` (an entry off its thread's stack) ⇒
/// `SP - RED_ZONE` is a huge address, never unsigned-`< 0`, so the check is inert there. The limit is a
/// constant within a stack's call tree, so the callee re-checks before its own frame.
fn emit_stack_check(b: &mut FunctionBuilder, lower: &Lower) {
    let cont = b.create_block();
    let trap_blk = b.create_block();
    // §2b path B: the running stack's low bound, from our own ABI param (per-vCPU by construction — no
    // cell, no TLS). 0 ⇒ `SP - RED_ZONE < 0` is never true ⇒ inert (the OS guard only).
    let limit = b.use_var(lower.limit_var);
    let sp = b.ins().get_stack_pointer(I64);
    let guard = b.ins().iadd_imm(sp, -(stack_check::RED_ZONE as i64)); // SP - RED_ZONE
    let lt = b.ins().icmp(IntCC::UnsignedLessThan, guard, limit);
    b.ins().brif(lt, trap_blk, &[], cont, &[]);
    b.switch_to_block(trap_blk);
    emit_trap(b, lower, TrapKind::StackOverflow);
    b.switch_to_block(cont);
}

/// A zero constant of CLIF type `t` (for a trapping path's dummy return).
fn zero_of(b: &mut FunctionBuilder, t: Type) -> Value {
    if t == F32 {
        b.ins().f32const(0.0)
    } else if t == F64 {
        b.ins().f64const(0.0)
    } else if t.is_vector() {
        // A vector result (e.g. a `float4`/v128 returned in a register, per §17): `iconst` is
        // scalar-integer only and produces malformed IR (the verifier hits an internal `unreachable`),
        // so materialize an all-zero vector constant instead — the same idiom the SIMD lowering uses.
        let bytes = vec![0u8; t.bytes() as usize];
        let zc = b.func.dfg.constants.insert(ConstantData::from(&bytes[..]));
        b.ins().vconst(t, zc)
    } else {
        b.ins().iconst(t, 0)
    }
}

/// The §3c indirect-call dispatch (invariant I2): mask the guest index into the
/// host-owned function table, load the slot's `type_id`, trap if it does not match the
/// call's signature (a forged/wrong-type index is inert), and return the slot's code
/// pointer. Leaves the builder positioned in the post-check ("matched") block.
fn indirect_dispatch(b: &mut FunctionBuilder, lower: &Lower, idx: Value, ty: &FuncType) -> Value {
    // slot = (idx as u32) & (next_pow2(nfuncs) - 1) — mask, not branch (Spectre-safe).
    let idx64 = b.ins().uextend(I64, idx);
    let m = b.ins().iconst(I64, lower.fn_table_mask as i64);
    let slot = b.ins().band(idx64, m);
    // entry_addr = fn_table_base + slot * sizeof(FnEntry=16)
    let off = b.ins().imul_imm(slot, 16);
    let base = b.use_var(lower.fn_table_var);
    let entry_addr = b.ins().iadd(base, off);

    // type_id check against the call's expected id.
    let tid = b.ins().load(I32, MemFlags::trusted(), entry_addr, 0);
    let expected = type_id_of(lower.distinct, ty);
    let want = b.ins().iconst(I32, expected as i32 as i64);
    let cond = b.ins().icmp(IntCC::Equal, tid, want);
    let matched = b.create_block();
    let bad = b.create_block();
    b.ins().brif(cond, matched, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::IndirectCallType);
    b.switch_to_block(matched);
    b.seal_block(matched);
    // code pointer at offset 8.
    b.ins().load(I64, MemFlags::trusted(), entry_addr, 8)
}

/// Emit "if the host trap cell is non-zero, propagate the trap now": branch to an early
/// `return` of zero-valued results (the trap kind / exit code already sits in the cell, which the
/// entry trampoline reads to decide `Trapped`/`Exited`). Used after **every** `call.cap` *and*
/// every `call`/`call.dyn`, so a trap raised deep in a callee unwinds the whole guest stack
/// immediately — before any later op can observe bogus zero results. A sibling vCPU's end stops
/// this one here too: the cell is the domain's, and nothing clears it during a run (#2499).
fn emit_trap_propagate(b: &mut FunctionBuilder, lower: &Lower) {
    let vmctx = b.use_var(lower.vmctx_var);
    let tc = b.ins().load(I64, MemFlags::trusted(), vmctx, 0);
    let trapped = b.ins().icmp_imm(IntCC::NotEqual, tc, 0);
    let trapret = b.create_block();
    let cont = b.create_block();
    b.ins().brif(trapped, trapret, &[], cont, &[]);
    b.switch_to_block(trapret);
    b.seal_block(trapret);
    emit_trap_return(b, lower);
    b.switch_to_block(cont);
    b.seal_block(cont);
}

/// Resolve the §9/D45 fast-path target for a `call.cap`, if one applies: there is a
/// [`FastCapResolver`] (top-level compile), the op has **at most one result** (the register ABI
/// returns a single i64), and the resolver claims this `(type_id, op)` (returns a non-null
/// specialized fn). Returns the symbol the call names the target by ([`fast_cap_symbol`]).
/// Resolution is a **compile-time** call into the embedder's resolver, which the code's link makes
/// again to bind the symbol.
fn fast_cap_target(lower: &Lower, type_id: u32, op: u32, sig: &FuncType) -> Option<String> {
    let resolver = lower.fast_resolver?;
    if sig.results.len() > 1 {
        return None; // the fast ABI returns a single register; multi-result falls back to the thunk
    }
    // SAFETY: `resolver` honours the `FastCapResolver` contract (caller guarantee); it's a pure
    // `(type_id, op, n_args, n_res) -> *const fn` lookup with no side effects, safe to call during
    // codegen. The arity is passed so the resolver only claims an op when its specialized fn's arity
    // matches the IR `call.cap`'s — else it returns null and the generic slot-based path is used.
    let target = unsafe {
        resolver(
            type_id,
            op,
            sig.params.len() as u32,
            sig.results.len() as u32,
        )
    };
    (!target.is_null()).then(|| fast_cap_symbol(type_id, op, sig.params.len(), sig.results.len()))
}

/// Lower a `call.cap` via the **devirtualized fast path** (§9 / D45): a direct register-to-register
/// call to the specialized host fn the symbol `target` names, passing
/// `ctx`/`mem_base`/`mem_size`/`handle`/`vmctx` then each argument **in a register** (widened to
/// its i64 slot), and reading the single result back from a register — no stack-slot marshalling,
/// no `n_args`/`n_res`/`type_id`/`op` dispatch. The trap cell is checked exactly as in
/// [`lower_cap_call`].
#[allow(clippy::too_many_arguments)]
fn lower_cap_call_fast(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    target: &str,
    sig: &FuncType,
    handle: u32,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    let n_res = sig.results.len();
    let ctx = vmctx_load(b, lower, vmctx::CAP_CTX);
    let mem_base = b.use_var(lower.mem_var);
    // #826: this frozen `lower.mapped` is deliberate — the ABI slot is the window's
    // default-committed **prefix** (the interpreter's equally-frozen `window.mapped`; growth lives
    // as explicit page-map entries, never by moving this value). A fast-path fn that needs to
    // admit grown-tail buffers must consult live page state like the generic thunk's
    // `MprotectWindow` does, not this bound alone.
    let mem_size = b.ins().iconst(I64, lower.mapped as i64);
    let h = get(vals, handle)?;
    let vmctx = b.use_var(lower.vmctx_var);

    let tsig = fast_cap_signature(module.isa().default_call_conv(), args.len(), n_res);
    let mut call_args = vec![ctx, mem_base, mem_size, h, vmctx];
    for a in args {
        let v = get(vals, *a)?;
        call_args.push(encode_slot(b, v)); // each arg widened to its i64 register slot
    }
    let callee = import_symbol(module, b.func, target, &tsig)?;
    let call = b.ins().call(callee, &call_args);

    // If the specialized fn set the trap cell, propagate (the cell already holds the kind / exit code).
    emit_trap_propagate(b, lower);

    if n_res == 1 {
        let slot = b.inst_results(call)[0]; // the i64 return register
        vals.push(decode_slot(b, slot, sig.results[0]));
    }
    Ok(())
}

/// Lower a `call.cap` (§3c/§9): marshal the arg slots into a stack buffer, call the host
/// thunk ([`HostFn::CapThunk`]) with the cap immediates + the guest window, and
/// — unless it set the trap cell — read the result slots back. A trap from the thunk
/// (CapFault / Exit) propagates like any other (return early; the cell is already set).
#[allow(clippy::too_many_arguments)]
fn lower_cap_call(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    type_id: u32,
    op: u32,
    sig: &FuncType,
    // The already-resolved `i32` handle value (the caller does `get(vals, handle_idx)`; §7 reflection
    // passes a constant 0 — the `CAP_SELF_TYPE_ID` dispatch ignores it).
    h: Value,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    let n_args = args.len();
    let n_res = sig.results.len();

    // Marshal the args into a stack buffer of i64 slots (null pointer when there are 0).
    let args_ptr = if n_args == 0 {
        b.ins().iconst(I64, 0)
    } else {
        let ss = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (n_args * 8) as u32,
            3,
        ));
        let addr = b.ins().stack_addr(I64, ss, 0);
        for (i, a) in args.iter().enumerate() {
            let v = get(vals, *a)?;
            let slot = encode_slot(b, v);
            b.ins()
                .store(MemFlags::trusted(), slot, addr, (i * 8) as i32);
        }
        addr
    };
    let res_ss = if n_res == 0 {
        None
    } else {
        Some(b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (n_res * 8) as u32,
            3,
        )))
    };
    let res_ptr = match res_ss {
        Some(ss) => b.ins().stack_addr(I64, ss, 0),
        None => b.ins().iconst(I64, 0),
    };
    // Audit #4: pre-zero the result slots, so a host op that writes fewer than `n_res` results can't
    // leave uninitialized stack for the read-back to decode. The verifier pins the sig arity, so a
    // correct host fills all of them — this guards a buggy host, not a guest (bounded to this slot).
    if let Some(ss) = res_ss {
        let z = b.ins().iconst(I64, 0);
        for i in 0..n_res {
            b.ins().stack_store(z, ss, (i * 8) as i32);
        }
    }

    // Assemble the thunk arguments (see `CapThunk`): the ctx is the instance's powerbox.
    let ctx = vmctx_load(b, lower, vmctx::CAP_CTX);
    let mem_base = b.use_var(lower.mem_var);
    // #826: this frozen `lower.mapped` is deliberate — it is the window's default-committed
    // **prefix** (the interpreter's equally-frozen `window.mapped`): the thunk builds its
    // `MprotectWindow` view as `(mapped default, reserved, live shared page map)`, so grown-tail
    // borrows are admitted by the page map, not by this value. Do not "fix" it to `reserved` —
    // that would default-commit the whole tail and admit borrows of never-mapped pages.
    let mem_size = b.ins().iconst(I64, lower.mapped as i64);
    // The reserved mask domain (`mask + 1`) the guest may `map`-grow into; 0 when no memory.
    let reserved = if lower.mapped == 0 { 0 } else { lower.mask + 1 };
    let mem_reserved = b.ins().iconst(I64, reserved as i64);
    let tid = b.ins().iconst(I32, type_id as i64);
    let opc = b.ins().iconst(I32, op as i64);
    let na = b.ins().iconst(I64, n_args as i64);
    let nr = b.ins().iconst(I64, n_res as i64);
    let vmctx = b.use_var(lower.vmctx_var);

    let call_args = [
        ctx,
        mem_base,
        mem_size,
        mem_reserved,
        tid,
        opc,
        h,
        args_ptr,
        na,
        res_ptr,
        nr,
        vmctx,
    ];
    call_host(module, b, HostFn::CapThunk, &call_args)?;

    // If the thunk set the trap cell, propagate (return early; the cell already holds
    // the kind / exit code).
    emit_trap_propagate(b, lower);

    // Read the result slots back.
    if let Some(ss) = res_ss {
        for (i, rty) in sig.results.iter().enumerate() {
            let slot = b.ins().stack_load(I64, ss, (i * 8) as i32);
            vals.push(decode_slot(b, slot, *rty));
        }
    }
    Ok(())
}

/// Lower a §14 `Instantiator` `call.cap` (iface 6) to the nesting runtime ([`instantiator_rt`]) — only
/// reached when this (parent) compile has a live `Nursery` (`lower.nesting`). Each served op
/// (`temen_ir::served_contract`: 17 `instantiate_rec`, 1 `join`, 9/10/12 supervision, 14
/// `child_offer`, 19 `grant`) calls its thunk, threading `mem_base` (the live parent window) +
/// `vmctx`; a thunk that sets the trap cell (forged handle, refused spawn, child trap) propagates
/// here like any `call.cap`.
#[allow(clippy::too_many_arguments)] // mirrors the CapCall fields + the shared lowering context
fn lower_instantiator(
    module: &mut JITModule,
    b: &mut FunctionBuilder,
    lower: &Lower,
    op: u32,
    sig: &FuncType,
    handle: u32,
    args: &[u32],
    vals: &mut Vec<Value>,
) -> Result<(), JitError> {
    // This lowering dispatches on `op` **statically** and can neither index missing args nor pass
    // mistyped values through the fixed thunk ABIs. The `call.cap` lowering has held the call to
    // its op's contract (`temen_ir::served_contract`) before routing it here, so every arg the
    // contract names is present. The contract is width-tolerant: the per-arg coercions below
    // (`slot_i64`/`slot_i32`/`result_as`) reconcile the declared widths with each thunk's fixed ABI.
    // temen#2168: the instance's own nursery, from its vmctx, so one compile serves every instance.
    let nursery = vmctx_load(b, lower, vmctx::NURSERY);
    let mem_base = b.use_var(lower.mem_var);
    let vmctx = b.use_var(lower.vmctx_var);
    match op {
        1 => {
            // join(nursery, mem_base, instantiator:i32, child_handle:i32, status_out:i64,
            // vmctx:i64) -> value:i64, the status through `status_out` as `cont.resume`'s is
            // (#2066). The nursery owns the child table for this run; the thunk still resolves the
            // call.cap's `Instantiator` first, as the oracle does for every op (#1729).
            let h = slot_i32(b, get(vals, handle)?);
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let ss =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let status_ptr = b.ins().stack_addr(I64, ss, 0);
            let call = call_host(
                module,
                b,
                HostFn::InstJoin,
                &[nursery, mem_base, h, child, status_ptr, vmctx],
            )?;
            emit_trap_propagate(b, lower);
            let value = b.inst_results(call)[0];
            let status = b.ins().stack_load(I64, ss, 0);
            vals.push(result_as(b, status, sig.results[0]));
            vals.push(result_as(b, value, sig.results[1]));
        }
        9 | 10 | 12 => {
            // S3 lifecycle: poll / detach / kill (nursery, mem_base, instantiator:i32, child:i32,
            // vmctx:i64) -> status:i32 — resolved and dispatched as `join` is.
            let h = slot_i32(b, get(vals, handle)?);
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let thunk = match op {
                9 => HostFn::InstPoll,
                10 => HostFn::InstDetach,
                _ => HostFn::InstKill,
            };
            let call = call_host(module, b, thunk, &[nursery, mem_base, h, child, vmctx])?;
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        17 => {
            // §3c instantiate_rec(nursery, mem_base, mem_size:i64, handle:i32, record_ptr:i64,
            //   vmctx:i64) -> handle:i32 — the config-record spawn: the thunk parses the
            //   record from the window (`mem_size` bounds the read) and spawns it. Pager records CapFault (sound: modules with impl exports fold to the
            //   oracle).
            //
            // #826: the bound is the **reserved** mask domain, not the compile-time `mapped` — a
            // record in a legitimately `map`-grown tail page must be read where the interpreter's
            // live page map reads it. Committed-ness is the hardware's to enforce, exactly like an
            // ordinary guest access: a read into a still-unmapped (PROT_NONE) or guarded page
            // faults through the SIGSEGV shim to `MemoryFault` (the thunk runs inside the armed
            // guest call, and the faulting address is in the guarded window range).
            let h = slot_i32(b, get(vals, handle)?);
            let win_reserved = if lower.mapped == 0 { 0 } else { lower.mask + 1 };
            let mem_size = b.ins().iconst(I64, win_reserved as i64);
            let record_ptr = slot_i64(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let call = call_host(
                module,
                b,
                HostFn::InstSpawn,
                &[nursery, mem_base, mem_size, h, record_ptr, vmctx],
            )?;
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        14 | 19 => {
            // CALLS.md 5c.0 child_offer(nursery, mem_base, instantiator:i32, child:i32, export:i64,
            // vmctx) -> handle:i32. Mint a live-callee offer over a granted child's
            // nursery-retained shared powerbox; every miss is the probeable -EINVAL (the interp op-14
            // arm, errno-for-errno). A forged `Instantiator` is the `CapFault` every op gives (#1729);
            // `mem_base` is only for that resolve — the mint reads no guest memory. #2220's `grant`
            // has the same shape over the same powerbox, its last arg the handle to grant.
            let h = slot_i32(b, get(vals, handle)?);
            let child = slot_i32(b, get(vals, *args.first().ok_or(JitError::Malformed)?)?);
            let arg = slot_i64(b, get(vals, *args.get(1).ok_or(JitError::Malformed)?)?);
            let thunk = match op {
                14 => HostFn::ChildOffer,
                _ => HostFn::ChildGrant,
            };
            let call = call_host(module, b, thunk, &[nursery, mem_base, h, child, arg, vmctx])?;
            emit_trap_propagate(b, lower);
            let r = result_as(b, b.inst_results(call)[0], sig.results[0]);
            vals.push(r);
        }
        // Unknown ops were rejected by the shape check above (→ runtime CapFault, matching the
        // interpreter's default arm) — this match only sees contract-validated ops.
        _ => unreachable!("shape check admitted an unknown Instantiator op"),
    }
    Ok(())
}

fn get(vals: &[Value], i: u32) -> Result<Value, JitError> {
    vals.get(i as usize).copied().ok_or(JitError::Malformed)
}

/// Widen a fetched call.cap arg to the `i64` slot an Instantiator thunk param takes — the JIT analogue
/// of the interpreter reading every arg as `Reg::i64()` off a **sign-extended** slot (`from_i32`). A
/// chibicc-emitted call already passes i64 (passthrough); a frontend that declares a slot arg `i32`
/// gets sign-extended, matching the interp exactly. Keeps the two backends in lockstep even though the
/// verifier only checks a call.cap against its *declared* sig (§3c), not the host op's contract.
fn slot_i64(b: &mut FunctionBuilder, v: Value) -> Value {
    if b.func.dfg.value_type(v) == I32 {
        b.ins().sextend(I64, v)
    } else {
        v
    }
}

/// Narrow a fetched call.cap arg to the `i32` a handle/child thunk param takes. chibicc widens every
/// scalar to i64, but handles cross as i32 (the host table masks them); an already-i32 value passes
/// through. Mirrors the interpreter's `get_i32` on the same slot.
fn slot_i32(b: &mut FunctionBuilder, v: Value) -> Value {
    if b.func.dfg.value_type(v) == I64 {
        b.ins().ireduce(I32, v)
    } else {
        v
    }
}

/// Coerce a thunk result to the guest's *declared* result type — the JIT analogue of the interpreter's
/// `slot_to_val(ty, slot)`. The thunks return canonical widths (an i32 child handle / status, an i64
/// join value); a chibicc sig declares i64 for an `int`-returning call, so **sign-extend** the i32 up
/// (a spawn yields a small non-negative handle or a negative errno — sign-extension matches both the
/// interp's i64 slot and C's `(int)`→`long`). Narrows the reverse case for completeness.
fn result_as(b: &mut FunctionBuilder, v: Value, declared: ValType) -> Value {
    let want = clif_ty(declared);
    let have = b.func.dfg.value_type(v);
    if have == want {
        v
    } else if want == I64 && have == I32 {
        b.ins().sextend(I64, v)
    } else if want == I32 && have == I64 {
        b.ins().ireduce(I32, v)
    } else {
        v
    }
}

/// Offset from `mem_base` of the window's trailing guard page (`round_up(win_reserved, page)`),
/// the always-`PROT_NONE` redirect target for the D63 branchless out-of-bounds fault. `0` (unused)
/// when the window has no reservation.
fn guard_offset_of(win_reserved: u64) -> u64 {
    if win_reserved == 0 {
        return 0;
    }
    let page = mem::page_size() as u64;
    (win_reserved + page - 1) & !(page - 1)
}

/// How far past `reserved` an access may reach and still fault on the window's trailing guard
/// ([`mem::trailing_guard`], DESIGN.md D67), when that guard starts exactly at `reserved`: a window
/// whose reservation is page-aligned. `0` otherwise (an unaligned reservation shares its last page
/// with the window), and then [`CheckedBases`] checks every access exactly, as [`mask_addr`] does.
fn guard_reach(lower: &Lower) -> u64 {
    let reserved = lower.mask.wrapping_add(1);
    let guard_at_reserved = lower.mask != 0
        && reserved != 0
        && reserved.is_multiple_of(mem::page_size() as u64)
        && lower.guard_offset == reserved;
    if guard_at_reserved {
        mem::trailing_guard() as u64
    } else {
        0
    }
}

/// #2147 — the bounds checks one IR block's loads and stores share. A **check point** is an access
/// whose own address `a` is tested against the reservation alone: `a ≥ reserved` redirects the base to
/// the guard page (`select_spectre_guard`, as [`mask_addr`] redirects), and the access's offset and
/// width fall within [`guard_reach`]. A later access in the same CLIF block whose address adds a
/// constant to the same **root** value — `d ≥ 0` past the check point's, with `d + offset + width`
/// within the reach — takes the checked base plus a constant, and tests nothing.
///
/// Exact, not approximate. If the check point's `a < reserved`, the later address is `a + d` with no
/// wrap (`a + d < reserved + page`), so it is the interpreter's address, and whatever of the access
/// lies past `reserved` lies on the guard page and faults there, at that access, where the
/// interpreter's `checked_reserved` faults it. If `a ≥ reserved`, the check point itself faulted
/// first. Under misspeculation the check point's base is the guard page's, so every access derived
/// from it stays on that page. A base is reused only in the CLIF block that computed it, so it always
/// dominates its uses; a call's trap check or a safepoint starts a new block, and with it new checks.
///
/// A store that straddles `reserved` writes nothing before it faults only on hardware that never
/// tears a faulting store: x86-64 guarantees it, Arm and RISC-V don't. [`mask_addr`] already relies
/// on the same at `mapped` and at every page the guest protects (#2162). Stores share checks anyway:
/// with a check of their own, a store's address is a different value from a load's at the same
/// address, and Cranelift can no longer forward the stored value to the load.
struct CheckedBases {
    /// [`guard_reach`]: `0` checks every access exactly.
    reach: i128,
    /// The block's `i64.const` values, by value index.
    consts: HashMap<u32, i64>,
    /// A value the block computes as `root + c` by adding (or subtracting) constants, by value index.
    roots: HashMap<u32, (u32, i128)>,
    /// The check points still usable: one per root, the latest.
    points: Vec<CheckPoint>,
}

struct CheckPoint {
    root: u32,
    /// The check point's address is `root + c`.
    c: i128,
    /// `mem_base + (a ≥ reserved ? guard_offset : a)`.
    base: Value,
    /// The CLIF block the base was computed in, the only one that may use it.
    at: cranelift_codegen::ir::Block,
}

impl CheckedBases {
    fn new(lower: &Lower) -> CheckedBases {
        CheckedBases {
            reach: i128::from(guard_reach(lower)),
            consts: HashMap::new(),
            roots: HashMap::new(),
            points: Vec::new(),
        }
    }

    /// Record what value `idx`, which `inst` defines, adds to a root.
    fn note(&mut self, idx: u32, inst: &Inst) {
        match *inst {
            Inst::ConstI64(k) => {
                self.consts.insert(idx, k);
            }
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a,
                b,
            } => {
                if let Some(&k) = self.consts.get(&b) {
                    self.roots.insert(idx, self.root_of(a, k));
                } else if let Some(&k) = self.consts.get(&a) {
                    self.roots.insert(idx, self.root_of(b, k));
                }
            }
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a,
                b,
            } => {
                if let Some(&k) = self.consts.get(&b) {
                    self.roots.insert(idx, self.root_of(a, -i128::from(k)));
                }
            }
            _ => {}
        }
    }

    /// `v + k` as `(root, c)`.
    fn root_of(&self, v: u32, k: impl Into<i128>) -> (u32, i128) {
        let (root, c) = self.roots.get(&v).copied().unwrap_or((v, 0));
        (root, c + k.into())
    }

    /// The physical address of a non-elided `width`-byte access at IR value `idx` (`addr`) + `offset`.
    fn confine(
        &mut self,
        b: &mut FunctionBuilder,
        lower: &Lower,
        idx: u32,
        addr: Value,
        offset: u64,
        width: u32,
    ) -> Value {
        let reach = self.reach;
        let span = i128::from(offset) + i128::from(width);
        if span > reach {
            return mask_addr(b, lower, addr, offset, false, width);
        }
        let (root, c) = self.root_of(idx, 0);
        let here = b.current_block().expect("lowering inside a block");
        if let Some(p) = self.points.iter().find(|p| p.root == root && p.at == here) {
            let d = c - p.c;
            if d >= 0 && d + span <= reach {
                let base = p.base;
                return b.ins().iadd_imm(base, (d + i128::from(offset)) as i64);
            }
        }
        let reserved = b.ins().iconst(I64, lower.mask.wrapping_add(1) as i64);
        let oob = b
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, addr, reserved);
        let guard = b.ins().iconst(I64, lower.guard_offset as i64);
        let confined = b.ins().select_spectre_guard(oob, guard, addr);
        let mem_base = b.use_var(lower.mem_var);
        let base = b.ins().iadd(mem_base, confined);
        let point = CheckPoint {
            root,
            c,
            base,
            at: here,
        };
        match self.points.iter_mut().find(|p| p.root == root) {
            Some(p) => *p = point,
            None => self.points.push(point),
        }
        if offset == 0 {
            base
        } else {
            b.ins().iadd_imm(base, offset as i64)
        }
    }
}

/// The §4 confinement lowering (invariant I1, **branchless confinement**, D63): bounds-test the
/// final effective address `addr + offset` against `reserved − offset − width`, then
/// `select_spectre_guard(oob, guard_offset, addr + offset)` and add the base, giving the physical
/// address `mem_base + (oob ? guard_offset : addr + offset)`. An out-of-bounds access is redirected
/// to `guard_offset` — the offset of the window's trailing guard page (`round_up(win_reserved,
/// page)`) — so the access *itself* faults there (`MemoryFault`, the architectural fault, mirroring
/// `temen_mask::Window::checked`), with no per-access branch. The
/// `select_spectre_guard` is also a speculation barrier: a misspeculated OOB access likewise receives
/// `guard_offset`, confining it to the guard — the Spectre-v1 hardening (§4, D63; supersedes the D38
/// `trapnz` + `& mask` clamp).
///
/// When `elide` is set the check and clamp are both dropped — but **only** the caller's
/// [`in_window`] proof (the address is provably `< size`) may set it, so the unclamped
/// `addr + offset` already stays in `[0, size)` (a proven bound holds speculatively too:
/// the proof is over data dependencies, e.g. `(i & K)*W`). This is the
/// elide-when-provably-bounded half of §1a (D36–D38); a wrong proof is a confinement
/// escape, caught by the escape-oracle (final-memory differential, §18).
fn mask_addr(
    b: &mut FunctionBuilder,
    lower: &Lower,
    addr: Value,
    offset: u64,
    elide: bool,
    width: u32,
) -> Value {
    // Fold the immediate only when non-zero, so an offset-0 access keeps a minimal address
    // expression (helps Cranelift's GVN / store-to-load forwarding recognize equal addresses).
    let eff = if offset == 0 {
        addr
    } else {
        let off = b.ins().iconst(I64, offset as i64);
        b.ins().iadd(addr, off)
    };
    // A non-elided access is confined by a bounds test + a branchless spectre-guard redirect (invariant
    // I1, **branchless confinement**, §4/D63): the access `[addr+offset, addr+offset+width)` must lie in
    // the reserved domain `[0, reserved)`, else a clear `MemoryFault` fires *at the offending access*
    // (vs the old mask silently wrapping a wild access to some other in-window byte). The bounds test
    // compares the dynamic `addr` against the compile-time constant `reserved - offset - width`, so it
    // never itself overflows; an out-of-bounds address is redirected (via `select_spectre_guard`, no
    // branch) to the enclosing window's trailing guard page and faults there. Within the reservation,
    // the committed-ness of `[0, mapped)` (and any page the guest `grow`s into the tail) is enforced by
    // the `PROT_NONE` guard region — a stray `[mapped, reserved)` access faults there. This mirrors the
    // interpreter's `confine_checked` (reserved-domain bound) + `check_prot` (per-page) — the redirect
    // is invisible to that mirror (it only fires on an access the interpreter also faults; the
    // interpreter doesn't speculate) — so the two agree on the same fault under the §18 escape oracle.
    // Elided accesses (proven `< reserved`) need no check. The no-memory / degenerate `mask == 0`
    // case keeps the mask (a 1-byte / memoryless window has no reservation to bound against).
    if !elide && lower.mask != 0 {
        // `reserved = mask + 1` (a power of two ≤ 2^63); the reservation this window is confined to.
        let reserved = lower.mask.wrapping_add(1);
        let need = (width as u64).saturating_add(offset);
        // Out-of-bounds test: `addr > reserved − offset − width` ⇒ some byte of the access falls
        // outside `[0, reserved)`. The compare is against a compile-time constant, so it never itself
        // overflows.
        let oob = match reserved.checked_sub(need) {
            Some(limit) => {
                let lim = b.ins().iconst(I64, limit as i64);
                b.ins().icmp(IntCC::UnsignedGreaterThan, addr, lim)
            }
            // `offset + width` alone exceeds the reservation ⇒ no `addr` is in-bounds; always fault.
            None => b.ins().iconst(I8, 1),
        };
        // **Branchless confinement** (invariant I1, D63), matching Wasmtime's spectre-guarded heap.
        // No `trapnz`: `select_spectre_guard(oob, guard_offset, …)` redirects an out-of-bounds access
        // to `guard_offset` — the offset of the **window's trailing guard page** from `mem_base` — so
        // the load/store *itself* faults there (SIGSEGV → `MemoryFault`, unwound by the guard handler
        // like any in-window guard fault), with no per-access branch.
        //
        // `guard_offset = round_up(win_reserved, page)` is always `PROT_NONE` (`GuestWindow` maps
        // `[round_up(mapped,page), round_up(reserved,page)+page)` inaccessible), and
        // `win_reserved == reserved`, so this is the window's own guard.
        //
        // `select_spectre_guard` is a **speculation barrier** (a data-dependent cmov, not a predicted
        // branch): a *misspeculated* OOB access also receives `guard_offset`, so it can never
        // speculatively read past the window — the same Spectre-v1 confinement the old AND mask gave,
        // via the identical primitive Wasmtime uses (in-process nesting is defense-in-depth, not a
        // promised Spectre boundary — DESIGN.md §2a). Dropping the per-access branch **and** the
        // memory-operand AND (a 40-bit mask is not an x86 immediate, so it was a RIP-relative load each
        // access) closes the residual array-kernel gap to Wasmtime-w64 (matmul 1.06→0.94,
        // matmul_eb 1.34→0.97 temen÷wt64; see the `confine` harness / DESIGN.md §4). The
        // clear-`MemoryFault`-at-the-offending-access property of §4/D38 is preserved — only the
        // mechanism (guard-page fault vs `ud2`) differs.
        let guard = b.ins().iconst(I64, lower.guard_offset as i64);
        let confined = b.ins().select_spectre_guard(oob, guard, eff);
        let base = b.use_var(lower.mem_var);
        return b.ins().iadd(base, confined);
    }
    // Elided (proven in-window) or the degenerate `mask == 0` (memoryless) path: no bounds check, and
    // the mask kept only for `mask == 0`.
    let confined = if elide {
        eff
    } else {
        let m = b.ins().iconst(I64, lower.mask as i64);
        b.ins().band(eff, m)
    };
    let base = b.use_var(lower.mem_var);
    b.ins().iadd(base, confined)
}

/// §2.2 / #2164 — a demand-paged child's **page check** of a `len`-byte access at `phys`, emitted after
/// the access's confinement (and, for an atomic, its alignment check: the interpreters' order). An
/// unpaged compile emits nothing, so its code is the unpaged code.
///
/// `phys` is `mem_base + off`, `off` the access's window offset, or the guard page's where confinement
/// redirected an out-of-bounds access. The access table ([`VmCtx::pages`]) has a byte for each page of
/// the declared window and one for the page past it, which admits every access: past the declared
/// window the hardware decides, as in an unpaged child, and the pager supplies nothing there. So an
/// access starting in the declared window tests the bytes of its first and last pages (`len` is never
/// more than a page, so those are its only pages), and one starting past it tests nothing.
///
/// A miss calls the miss thunk ([`PageMissThunk`]): it returns once every page of the access admits
/// it, its pager having supplied the rest, or with a trap set, which unwinds. The bytes load with
/// acquire ordering: a supply writes a page's bytes before its table byte, so a vCPU that reads the
/// byte reads the bytes.
fn emit_page_check(b: &mut FunctionBuilder, lower: &Lower, phys: Value, len: u32, flags: u64) {
    let Some((page, thunk, table)) = lower.page else {
        return;
    };
    let base = b.use_var(lower.mem_var);
    let off = b.ins().isub(phys, base);
    let check = b.create_block();
    let miss = b.create_block();
    let done = b.create_block();
    let declared = (page.pages << page.shift) as i64;
    let in_window = b.ins().icmp_imm(IntCC::UnsignedLessThan, off, declared);
    b.ins().brif(in_window, check, &[], done, &[]);

    b.switch_to_block(check);
    b.seal_block(check);
    let t = b.use_var(table);
    let state = |b: &mut FunctionBuilder, at: Value| {
        let index = b.ins().ushr_imm(at, i64::from(page.shift));
        let byte = b.ins().iadd(t, index);
        b.ins().atomic_load(I8, MemFlags::trusted(), byte)
    };
    let mut admits = state(b, off);
    if len > 1 {
        let last = b.ins().iadd_imm(off, i64::from(len - 1));
        let last = state(b, last);
        admits = b.ins().band(admits, last);
    }
    let need = if flags & PAGE_MISS_WRITE != 0 {
        temen_ir::PAGE_WRITE
    } else {
        temen_ir::PAGE_READ
    };
    let ok = b.ins().band_imm(admits, i64::from(need));
    b.ins().brif(ok, done, &[], miss, &[]);

    b.set_cold_block(miss);
    b.switch_to_block(miss);
    b.seal_block(miss);
    let len = b.ins().iconst(I64, i64::from(len));
    emit_page_miss(b, lower, thunk, base, off, len, flags);
    b.ins().jump(done, &[]);

    b.switch_to_block(done);
    b.seal_block(done);
}

/// #2164 — a demand-paged child's page check of a bulk op's span `[phys, phys + n)`: the miss thunk
/// walks every page of a non-empty one before the op touches any, as the interpreters check a span
/// whole before they copy (a span may cover many pages, so no inline test). An unpaged compile emits
/// nothing.
fn emit_span_page_check(b: &mut FunctionBuilder, lower: &Lower, phys: Value, n: Value, flags: u64) {
    let Some((_, thunk, _)) = lower.page else {
        return;
    };
    let base = b.use_var(lower.mem_var);
    let off = b.ins().isub(phys, base);
    let walk = b.create_block();
    let done = b.create_block();
    b.ins().brif(n, walk, &[], done, &[]);
    b.switch_to_block(walk);
    b.seal_block(walk);
    emit_page_miss(b, lower, thunk, base, off, n, flags);
    b.ins().jump(done, &[]);
    b.switch_to_block(done);
    b.seal_block(done);
}

/// #2164 — whether a demand-paged instance's access table `table` admits a `len`-byte access at
/// window offset `off` that needs `need` ([`temen_ir::PAGE_READ`] or [`temen_ir::PAGE_WRITE`]): every
/// page of it among the declared window's `pages` has the bit. Past the declared window the hardware
/// decides. For a host thunk that touches the window itself, as [`emit_page_check`] is for the code.
///
/// # Safety
/// `table` is the instance's live access table, `pages + 1` bytes.
#[cfg(fiber_rt)]
pub(crate) unsafe fn table_admits(
    table: *const AtomicU8,
    pages: u64,
    off: u64,
    len: u64,
    need: u8,
) -> bool {
    if len == 0 {
        return true;
    }
    let shift = (mem::page_size() as u64).trailing_zeros();
    let (first, last) = (off >> shift, (off + (len - 1)) >> shift);
    (first..=last.min(pages.saturating_sub(1)))
        .all(|p| (*table.add(p as usize)).load(Ordering::Acquire) & need != 0)
}

/// Call the miss thunk `thunk` on `len` bytes at window offset `off` of the window at `base`, and
/// unwind if it set a trap.
fn emit_page_miss(
    b: &mut FunctionBuilder,
    lower: &Lower,
    thunk: FuncRef,
    base: Value,
    off: Value,
    len: Value,
    flags: u64,
) {
    let vm = b.use_var(lower.vmctx_var);
    let flags = b.ins().iconst(I64, flags as i64);
    b.ins().call(thunk, &[vm, base, off, len, flags]);
    emit_trap_propagate(b, lower);
}

/// **Bulk-span backed-extent probe (ISSUES.md I21).** Fault a bulk span that overruns the
/// currently-*backed* region **before** the libcall runs. [`confine_span`] only bounds the span
/// against `reserved`, delegating the `[mapped, reserved)` guard hole to the libcall's own accesses
/// — but that leaks in two ways the interpreter (`Mem::check_prot_span`, which validates every page
/// up front) does not: (1) libc `memcpy`/`memmove` **short-circuit `dst == src`**, so a self-copy
/// overrunning `mapped` never faults; (2) the libcall writes a prefix before hitting the guard, so a
/// faulting run diverges from the interpreter's fault-before-any-write. Both are interp↔JIT
/// divergences (§3 parity), not escapes (every byte stays in `[0, reserved)`).
///
/// The fix is a guarded 1-byte read of the span's **first** byte (`phys`) and **last** byte
/// (`phys + len − 1`, where `phys` is [`confine_span`]'s confined base). Each faults
/// [`TrapKind::MemoryFault`] on a `PROT_NONE` guard page exactly where the interpreter faults —
/// **consulting the live page tables**, so it honors guest `grow` (unlike the compile-time
/// `Lower::mapped`, which would over-fault a grown window). The last-byte read catches an overrun
/// past the backed extent; the first-byte read catches a span starting inside the #1094 NULL guard
/// `[0, POWERBOX_NULL_GUARD)` — which the last-byte read alone misses when the span *ends* in backed
/// memory (e.g. a whole-window self-copy from 0: macOS libc `memmove` short-circuits `dst == src`,
/// so nothing else would ever touch the guard, an interp↔JIT divergence the spec-mem differential
/// caught on macOS while glibc happened to fault). For the contiguous unmapped-guard + backed-prefix
/// model (the production window + the spec window), first-and-last-in-bounds ⟺ whole-span-in-bounds.
/// Emitted per span (`dst` and `src`) before the copy, so no partial write survives. `len == 0`
/// skips the probe entirely (a branch on `len != 0`), keeping a zero-length op inert even at a wild
/// pointer (D62).
///
/// Residual (documented, not silently dropped): a bulk op whose span straddles a guest-created
/// **interior** hole (`unmap`) or a read-only page mid-span is not caught by a last-byte read probe
/// — the same boundary the interpreter's contiguous fast path assumes away. Left to a per-page probe
/// loop if a real workload needs bulk ops over a deliberately-punched window.
fn probe_span(b: &mut FunctionBuilder, phys: Value, len: Value) {
    let do_probe = b.create_block();
    let after = b.create_block();
    let nonzero = b.ins().icmp_imm(IntCC::NotEqual, len, 0);
    b.ins().brif(nonzero, do_probe, &[], after, &[]);

    b.switch_to_block(do_probe);
    b.seal_block(do_probe);
    // May-trap loads: each faults `MemoryFault` on a guard page — the first byte on the #1094 NULL
    // guard, the last byte on an overrun past the backed region. The results are unused, but the
    // loads are preserved because they can trap (side-effecting).
    b.ins().load(I8, mem_flags(), phys, 0);
    let one = b.ins().iconst(I64, 1);
    let last_off = b.ins().isub(len, one);
    let last = b.ins().iadd(phys, last_off);
    b.ins().load(I8, mem_flags(), last, 0);
    b.ins().jump(after, &[]);

    b.switch_to_block(after);
    b.seal_block(after);
}

/// #2177 — the most bytes a constant-length `mem.copy`/`mem.move` copies inline ([`copy_inline`]).
const INLINE_COPY_MAX: i64 = 64;

/// #2177 — a `mem.copy`/`mem.move` of a constant `n` bytes, `0 < n ≤` [`INLINE_COPY_MAX`], as loads and
/// stores: what gcc makes of a small struct copy, where the libcall path costs a call that clobbers
/// every caller-saved register. Each span is confined as one `n`-byte access, exactly as a load or
/// store of that width is ([`CheckedBases`], or no check where the bound proof elides one). Every
/// source byte is loaded, and the destination's first and last bytes are probed, before the first
/// store. So overlapping spans copy as `mem.move` does, and a span that faults faults before anything
/// is written (I21, as [`probe_span`] keeps the libcall path): `n` is under a page, so each span
/// touches at most two pages, and those reads reach both.
#[allow(clippy::too_many_arguments)]
fn copy_inline(
    b: &mut FunctionBuilder,
    lower: &Lower,
    checked: &mut CheckedBases,
    ubs: &[u64],
    vals: &[Value],
    dst: u32,
    src: u32,
    n: u32,
) -> Result<(), JitError> {
    let size = lower.mask.wrapping_add(1);
    let mut confine = |b: &mut FunctionBuilder, idx: u32| -> Result<Value, JitError> {
        let a = get(vals, idx)?;
        Ok(if in_window(ub_at(ubs, idx), 0, n, size) {
            mask_addr(b, lower, a, 0, true, n)
        } else {
            checked.confine(b, lower, idx, a, 0, n)
        })
    };
    let s = confine(b, src)?;
    let d = confine(b, dst)?;
    // #2164: the source's pages, then the destination's, as the interpreters check a copy's spans.
    emit_page_check(b, lower, s, n, 0);
    emit_page_check(b, lower, d, n, PAGE_MISS_WRITE);
    // Widest words first: 8 bytes, then the 4/2/1-byte tail.
    let mut words = Vec::new();
    let mut off = 0;
    for w in [8, 4, 2, 1] {
        while n - off >= w {
            words.push((
                off as i32,
                b.ins().load(width_ty(w), mem_flags(), s, off as i32),
            ));
            off += w;
        }
    }
    b.ins().load(I8, mem_flags(), d, 0);
    b.ins().load(I8, mem_flags(), d, (n - 1) as i32);
    for (off, word) in words {
        b.ins().store(mem_flags(), word, d, off);
    }
    Ok(())
}

/// Confine a **dynamic-length span** `[ptr, ptr+len)` to the reserved domain `[0, reserved)` and
/// return its physical base `mem_base + (ptr & mask)` — the bulk-memory hinge (D62). Where
/// [`mask_addr`] confines one fixed-width access, this confines a whole span with a *single* range
/// check: it traps [`TrapKind::MemoryFault`] (native `trapnz`) when `len != 0 && (len > reserved ||
/// ptr > reserved − len)`, i.e. any byte of the span would fall outside `[0, reserved)`. The two
/// sub-checks avoid overflow: `len > reserved` catches an oversized (or negative-as-u64) length before
/// the `reserved − len` subtraction can wrap. `len == 0` is a no-op span that never faults (matching
/// the interpreter and C `memcpy(_,_,0)`), so a 0-length op on a wild pointer is inert.
///
/// This predicate is the overflow-avoiding lowering of the **one span-OOB reference**,
/// `temen_mask::Window::span_checked` (the arithmetic the interpreter calls and the `mask` fuzz target
/// asserts against a `u128` oracle). Any change to that reference must be mirrored in the CLIF here —
/// the fuzz oracle pins the reference, and the `diff.rs` escape oracle pins this emission to it.
///
/// The returned base is then Spectre-clamped (`& mask`) exactly as in [`mask_addr`]: architecturally a
/// no-op (the check proved `ptr < reserved`), but on a *mispredicted* path it pins the copy's base
/// inside `[0, reserved)`. As with `mask_addr`, this matches Wasmtime's bounds-checked `memory.copy`
/// posture ("as secure as wasm", DESIGN.md §1a): the length is checked, the base is confined, and the
/// bulk copy itself is a libcall the CPU does not speculate byte-by-byte — so no per-byte clamp is
/// needed. Callers pass the **same** `len` to the copy, so both spans (dst and src) are checked.
fn confine_span(b: &mut FunctionBuilder, lower: &Lower, ptr: Value, len: Value) -> Value {
    let reserved = lower.mask.wrapping_add(1);
    let resv = b.ins().iconst(I64, reserved as i64);
    let oob_len = b.ins().icmp(IntCC::UnsignedGreaterThan, len, resv);
    let rem = b.ins().isub(resv, len); // reserved − len; meaningful only when !oob_len
    let oob_ptr = b.ins().icmp(IntCC::UnsignedGreaterThan, ptr, rem);
    let oob_span = b.ins().bor(oob_len, oob_ptr);
    let zero = b.ins().iconst(I64, 0);
    let nonzero = b.ins().icmp(IntCC::NotEqual, len, zero);
    let oob = b.ins().band(nonzero, oob_span);
    b.ins()
        .trapnz(oob, cranelift_codegen::ir::TrapCode::HEAP_OUT_OF_BOUNDS);
    let m = b.ins().iconst(I64, lower.mask as i64);
    let clamped = b.ins().band(ptr, m);
    let base = b.use_var(lower.mem_var);
    b.ins().iadd(base, clamped)
}

// The confinement bound-proof (`UB_TOP`/`ub_at`/`ub_of`/`in_window`) now lives in one audited place,
// `temen_ir::bounds`, so the native and wasm JITs share a single definition of this veto predicate
// rather than diverging copies (INVARIANTS #9). Imported at the top of this module.

/// Little-endian, may-trap memory access flags (the window is host memory; the guard
/// margin absorbs width overrun, so this never faults in practice).
fn mem_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf
}

/// The CLIF type holding `width` raw bytes.
fn width_ty(width: u32) -> Type {
    match width {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    }
}

fn lower_load(b: &mut FunctionBuilder, op: LoadOp, phys: Value) -> Value {
    let (_, rty, width, signed) = op.info();
    // Float loads read the float type directly (no extension).
    if matches!(rty, ValType::F32 | ValType::F64) {
        return b.ins().load(clif_ty(rty), mem_flags(), phys, 0);
    }
    let load_ty = width_ty(width);
    let raw = b.ins().load(load_ty, mem_flags(), phys, 0);
    let result_ty = clif_ty(rty);
    if load_ty == result_ty {
        raw
    } else if signed {
        b.ins().sextend(result_ty, raw) // narrow signed load: sign-extend
    } else {
        b.ins().uextend(result_ty, raw) // narrow unsigned load: zero-extend
    }
}

fn lower_store(b: &mut FunctionBuilder, op: StoreOp, phys: Value, value: Value) {
    let (_, vty, width) = op.info();
    // Float stores write the float bits directly.
    if matches!(vty, ValType::F32 | ValType::F64) {
        b.ins().store(mem_flags(), value, phys, 0);
        return;
    }
    let store_ty = width_ty(width);
    // Narrow stores keep only the low `width` bytes (matches the interpreter).
    let v = if b.func.dfg.value_type(value) == store_ty {
        value
    } else {
        b.ins().ireduce(store_ty, value)
    };
    b.ins().store(mem_flags(), v, phys, 0);
}

/// Access width (and natural-alignment requirement) of a §12 atomic `ty`.
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Memory flags for an atomic access: little-endian (the window is LE) and aligned — a preceding
/// [`guard_atomic_align`] traps a misaligned address, so the hardware atomic only ever sees a
/// naturally-aligned one.
fn atomic_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf.set_aligned();
    mf
}

/// Map an IR atomic RMW op to Cranelift's.
fn clif_rmw_op(op: AtomicRmwOp) -> ClifRmwOp {
    match op {
        AtomicRmwOp::Add => ClifRmwOp::Add,
        AtomicRmwOp::Sub => ClifRmwOp::Sub,
        AtomicRmwOp::And => ClifRmwOp::And,
        AtomicRmwOp::Or => ClifRmwOp::Or,
        AtomicRmwOp::Xor => ClifRmwOp::Xor,
        AtomicRmwOp::Xchg => ClifRmwOp::Xchg,
    }
}

/// Trap (`MemoryFault`) if physical address `phys` is not `width`-aligned, else fall through —
/// mirrors the §12 interpreter's `check_align`. `atomic_*` lowerings require natural alignment
/// (e.g. aarch64 `LDAXR`/`STLXR` fault otherwise), so this precedes every atomic. Leaves the
/// builder positioned in the aligned ("ok") block.
fn guard_atomic_align(b: &mut FunctionBuilder, lower: &Lower, phys: Value, width: u32) {
    if width <= 1 {
        return;
    }
    let rem = b.ins().band_imm(phys, (width - 1) as i64);
    let aligned = b.ins().icmp_imm(IntCC::Equal, rem, 0);
    let ok = b.create_block();
    let bad = b.create_block();
    b.ins().brif(aligned, ok, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::MemoryFault);
    b.switch_to_block(ok);
    b.seal_block(ok);
}

/// Decode an `i64` calling-convention slot to a value of IR type `ty`.
fn decode_slot(b: &mut FunctionBuilder, slot: Value, ty: ValType) -> Value {
    match ty {
        ValType::I64 | ValType::Ref => slot, // `ref` is an opaque i64 in the cap ABI
        ValType::I32 | ValType::Cap => b.ins().ireduce(I32, slot),
        ValType::F32 => {
            let i = b.ins().ireduce(I32, slot);
            b.ins().bitcast(F32, MemFlags::new(), i)
        }
        ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), slot),
        // `v128` cap-ABI slots are out of MVP scope (§17); a zero vector keeps this total.
        ValType::V128 => {
            let z = b.ins().iconst(I8, 0);
            b.ins().splat(I8X16, z)
        }
    }
}

/// Encode a value into its `i64` calling-convention slot (the harness reads back the
/// low 32 bits for i32/f32 results).
fn encode_slot(b: &mut FunctionBuilder, v: Value) -> Value {
    match b.func.dfg.value_type(v) {
        I64 => v,
        I32 => b.ins().uextend(I64, v),
        F32 => {
            let i = b.ins().bitcast(I32, MemFlags::new(), v);
            b.ins().uextend(I64, i)
        }
        F64 => b.ins().bitcast(I64, MemFlags::new(), v),
        _ => v,
    }
}

/// Map a vector float op to the scalar [`FBinOp`]/[`FUnOp`] so vector lanes lower exactly
/// like scalars (§17/D58).
fn vf_bin(op: VFloatBinOp) -> FBinOp {
    match op {
        VFloatBinOp::Add => FBinOp::Add,
        VFloatBinOp::Sub => FBinOp::Sub,
        VFloatBinOp::Mul => FBinOp::Mul,
        VFloatBinOp::Div => FBinOp::Div,
        VFloatBinOp::Min => FBinOp::Min,
        VFloatBinOp::Max => FBinOp::Max,
    }
}
fn vf_un(op: VFloatUnOp) -> FUnOp {
    match op {
        VFloatUnOp::Abs => FUnOp::Abs,
        VFloatUnOp::Neg => FUnOp::Neg,
        VFloatUnOp::Sqrt => FUnOp::Sqrt,
        VFloatUnOp::Ceil => FUnOp::Ceil,
        VFloatUnOp::Floor => FUnOp::Floor,
        VFloatUnOp::Trunc => FUnOp::Trunc,
        VFloatUnOp::Nearest => FUnOp::Nearest,
    }
}

fn float_bin(b: &mut FunctionBuilder, op: FBinOp, x: Value, y: Value) -> Value {
    match op {
        FBinOp::Add => b.ins().fadd(x, y),
        FBinOp::Sub => b.ins().fsub(x, y),
        FBinOp::Mul => b.ins().fmul(x, y),
        FBinOp::Div => b.ins().fdiv(x, y),
        // The IR defines `min`/`max` to yield a **canonical** NaN on any NaN input — the reference
        // interpreter's `fmin`/`fmax` force `0x7FC0_0000` / `0x7FF8_..` (§18 oracle). Cranelift's
        // `fmin`/`fmax` instead *propagate* the input NaN's payload and sign, so without a fixup the
        // JIT and interp would store different NaN bits and the byte-exact escape-oracle window
        // comparison would diverge. (Add/Sub/Mul/Div need no fixup: both backends emit the same host
        // float instruction, so their NaN results already match — only the hand-written min/max
        // canonicalizes.) Canonicalize scalar and per-lane v128 alike.
        FBinOp::Min => {
            let r = b.ins().fmin(x, y);
            canonicalize_nan(b, r)
        }
        FBinOp::Max => {
            let r = b.ins().fmax(x, y);
            canonicalize_nan(b, r)
        }
        FBinOp::Copysign => b.ins().fcopysign(x, y),
    }
}

/// Replace any NaN in `r` with the IR's canonical quiet NaN — `0x7FC0_0000` (f32) /
/// `0x7FF8_0000_0000_0000` (f64) — matching the reference interpreter. Works for a scalar float and,
/// lane-wise, for a `v128` float vector (`Unordered(r, r)` is true exactly on the NaN lanes).
fn canonicalize_nan(b: &mut FunctionBuilder, r: Value) -> Value {
    let ty = b.func.dfg.value_type(r);
    // `Unordered(r, r)` is true exactly where `r` is NaN (a per-lane mask for vectors).
    let is_nan = b.ins().fcmp(FloatCC::Unordered, r, r);
    if ty.is_vector() {
        // Blend in the integer-vector domain so the fcmp mask and the operands share a type
        // (`bitselect` requires it): fcmp on F32X4 / F64X2 yields the I32X4 / I64X2 lane mask.
        let (ity, bits): (Type, i64) = if ty.lane_type() == F32 {
            (I32X4, 0x7FC0_0000)
        } else {
            (I64X2, 0x7FF8_0000_0000_0000u64 as i64)
        };
        let canon_lane = b.ins().iconst(ity.lane_type(), bits);
        let canon = b.ins().splat(ity, canon_lane);
        let ri = vcast(b, r, ity);
        let sel = b.ins().bitselect(is_nan, canon, ri);
        vcast(b, sel, ty)
    } else {
        let canon = if ty == F32 {
            b.ins().f32const(f32::from_bits(0x7FC0_0000))
        } else {
            b.ins().f64const(f64::from_bits(0x7FF8_0000_0000_0000))
        };
        b.ins().select(is_nan, canon, r)
    }
}

fn float_un(b: &mut FunctionBuilder, op: FUnOp, x: Value) -> Value {
    match op {
        FUnOp::Abs => b.ins().fabs(x),
        FUnOp::Neg => b.ins().fneg(x),
        FUnOp::Sqrt => b.ins().sqrt(x),
        FUnOp::Ceil => b.ins().ceil(x),
        FUnOp::Floor => b.ins().floor(x),
        FUnOp::Trunc => b.ins().trunc(x),
        FUnOp::Nearest => b.ins().nearest(x),
    }
}

fn float_cc(op: FCmpOp) -> FloatCC {
    match op {
        FCmpOp::Eq => FloatCC::Equal,
        FCmpOp::Ne => FloatCC::NotEqual, // unordered ≠ (NaN ne x is true), wasm semantics
        FCmpOp::Lt => FloatCC::LessThan,
        FCmpOp::Le => FloatCC::LessThanOrEqual,
        FCmpOp::Gt => FloatCC::GreaterThan,
        FCmpOp::Ge => FloatCC::GreaterThanOrEqual,
    }
}

/// Map IR edge args to CLIF block-call args (`BlockArg`, the 0.132 block-call type).
/// An edge's arguments, but those for the params of `target` the lowering takes (#2448).
fn map_edge(
    vals: &[Value],
    blocks: &Blocks,
    target: u32,
    args: &[u32],
) -> Result<Vec<BlockArg>, JitError> {
    args.iter()
        .enumerate()
        .filter(|&(i, _)| blocks.taken(target as usize, i).is_none())
        .map(|(_, a)| get(vals, *a).map(BlockArg::from))
        .collect()
}

/// A function's IR blocks as its lowering goes: each one's CLIF block, which of its params the
/// lowering takes as the values they forward (#2448), with no CLIF param and no edge argument, and,
/// once it is lowered, its values, which a later block's taken params read.
struct Blocks {
    clif: Vec<cranelift_codegen::ir::Block>,
    fwd: temen_ir::Forwards,
    vals: Vec<Vec<Value>>,
}

impl Blocks {
    /// The block and index of the value block `b`'s param `i` is taken as, if it is taken. Blocks
    /// lower in index order, so a param is taken only where the block of its value comes before its
    /// own. That block is lowered by then, and it dominates the param's ([`temen_ir::Forwards`]), so
    /// the value is defined wherever the param is used.
    fn taken(&self, b: usize, i: usize) -> Option<(usize, usize)> {
        self.fwd
            .source(b, i)
            .map(|(sb, si)| (sb as usize, si as usize))
            .filter(|&(sb, _)| sb < b)
    }
}

fn int_bin(b: &mut FunctionBuilder, op: BinOp, x: Value, y: Value) -> Value {
    match op {
        BinOp::Add => b.ins().iadd(x, y),
        BinOp::Sub => b.ins().isub(x, y),
        BinOp::Mul => b.ins().imul(x, y),
        BinOp::And => b.ins().band(x, y),
        BinOp::Or => b.ins().bor(x, y),
        BinOp::Xor => b.ins().bxor(x, y),
        BinOp::Shl => b.ins().ishl(x, y),
        BinOp::ShrS => b.ins().sshr(x, y),
        BinOp::ShrU => b.ins().ushr(x, y),
        BinOp::Rotl => b.ins().rotl(x, y),
        BinOp::Rotr => b.ins().rotr(x, y),
        // div/rem are guarded and lowered by `lower_div_rem`, never here.
        BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => unreachable!("guarded elsewhere"),
    }
}

/// Lower a trapping `div`/`rem` with explicit guards: a zero divisor traps
/// `DivByZero`, and signed `div` of `INT_MIN / -1` traps `IntOverflow` (matching the
/// interpreter). On the non-trapping path the division runs on a value Cranelift's
/// `sdiv`/`srem` will not fault on (so the hardware op never traps). Returns the
/// quotient/remainder in the final ("safe") block.
fn lower_div_rem(
    b: &mut FunctionBuilder,
    lower: &Lower,
    ty: IntTy,
    op: BinOp,
    x: Value,
    y: Value,
) -> Value {
    let ity = int_clif_ty(ty);
    // Trap on a zero divisor.
    let is_zero = b.ins().icmp_imm(IntCC::Equal, y, 0);
    let after_zero = b.create_block();
    let dz = b.create_block();
    b.ins().brif(is_zero, dz, &[], after_zero, &[]);
    b.switch_to_block(dz);
    b.seal_block(dz);
    emit_trap(b, lower, TrapKind::DivByZero);
    b.switch_to_block(after_zero);
    b.seal_block(after_zero);

    // Signed div additionally traps on INT_MIN / -1 (overflow).
    if op == BinOp::DivS {
        let min = match ty {
            IntTy::I32 => i32::MIN as i64,
            IntTy::I64 => i64::MIN,
        };
        let x_is_min = b.ins().icmp_imm(IntCC::Equal, x, min);
        let y_is_neg1 = b.ins().icmp_imm(IntCC::Equal, y, -1);
        let overflow = b.ins().band(x_is_min, y_is_neg1);
        let after_ov = b.create_block();
        let ov = b.create_block();
        b.ins().brif(overflow, ov, &[], after_ov, &[]);
        b.switch_to_block(ov);
        b.seal_block(ov);
        emit_trap(b, lower, TrapKind::IntOverflow);
        b.switch_to_block(after_ov);
        b.seal_block(after_ov);
    }

    let _ = ity;
    match op {
        BinOp::DivS => b.ins().sdiv(x, y),
        BinOp::DivU => b.ins().udiv(x, y),
        // `INT_MIN % -1 == 0` is representable so `rem_s` does not trap (only `div_s`
        // does); Cranelift's `srem` yields 0 here, matching the interpreter.
        BinOp::RemS => b.ins().srem(x, y),
        BinOp::RemU => b.ins().urem(x, y),
        _ => unreachable!("non-div/rem routed here"),
    }
}

/// Lower a trapping float→int (`trunc`): trap `BadConversion` on NaN or out-of-range,
/// else convert. The bounds are the interpreter's exact bounds (computed in `f64`, so
/// `f32` is promoted first), so the JIT and interpreter trap on identical inputs.
fn lower_trunc_trap(
    b: &mut FunctionBuilder,
    lower: &Lower,
    x: Value,
    from: FloatTy,
    to: IntTy,
    signed: bool,
) -> Value {
    // Promote f32 -> f64 (exact) so one set of bounds covers both.
    let xf = match from {
        FloatTy::F32 => b.ins().fpromote(F64, x),
        FloatTy::F64 => x,
    };
    // (lower bound, upper bound, lower-inclusive). Ordered comparisons are false for
    // NaN, so NaN falls out of range and traps — no separate NaN check needed.
    let (lo, hi, lo_incl) = match (to, signed) {
        (IntTy::I32, true) => (-2_147_483_649.0_f64, 2_147_483_648.0_f64, false),
        (IntTy::I32, false) => (-1.0_f64, 4_294_967_296.0_f64, false),
        (IntTy::I64, true) => (
            -9_223_372_036_854_775_808.0_f64,
            9_223_372_036_854_775_808.0_f64,
            true,
        ),
        (IntTy::I64, false) => (-1.0_f64, 18_446_744_073_709_551_616.0_f64, false),
    };
    let lo_c = b.ins().f64const(lo);
    let hi_c = b.ins().f64const(hi);
    let ge_lo = if lo_incl {
        b.ins().fcmp(FloatCC::GreaterThanOrEqual, xf, lo_c)
    } else {
        b.ins().fcmp(FloatCC::GreaterThan, xf, lo_c)
    };
    let lt_hi = b.ins().fcmp(FloatCC::LessThan, xf, hi_c);
    let in_range = b.ins().band(ge_lo, lt_hi);

    let ok = b.create_block();
    let bad = b.create_block();
    b.ins().brif(in_range, ok, &[], bad, &[]);
    b.switch_to_block(bad);
    b.seal_block(bad);
    emit_trap(b, lower, TrapKind::BadConversion);
    b.switch_to_block(ok);
    b.seal_block(ok);
    // In range: the saturating cast is exact and never faults.
    let ity = int_clif_ty(to);
    if signed {
        b.ins().fcvt_to_sint_sat(ity, x)
    } else {
        b.ins().fcvt_to_uint_sat(ity, x)
    }
}

fn int_cc(op: temen_ir::CmpOp) -> IntCC {
    use temen_ir::CmpOp::*;
    match op {
        Eq => IntCC::Equal,
        Ne => IntCC::NotEqual,
        LtS => IntCC::SignedLessThan,
        LtU => IntCC::UnsignedLessThan,
        LeS => IntCC::SignedLessThanOrEqual,
        LeU => IntCC::UnsignedLessThanOrEqual,
        GtS => IntCC::SignedGreaterThan,
        GtU => IntCC::UnsignedGreaterThan,
        GeS => IntCC::SignedGreaterThanOrEqual,
        GeU => IntCC::UnsignedGreaterThanOrEqual,
    }
}
