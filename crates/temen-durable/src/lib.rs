//! `temen-durable` — the IR→IR freeze/thaw transform (DESIGN.md / DURABILITY.md D60).
//!
//! A **tooling-tier, non-TCB** crate (like `temen-text`): it depends only on `temen-ir`
//! and emits ordinary, verifier-passing IR — no new instructions, no escape-TCB
//! surface. An embedder running pre-instrumented modules links none of it.
//!
//! This is the **Phase 1** slice of the plan (DURABILITY.md §9): it instruments a
//! function so an in-flight may-suspend op (a `call.cap`, or a `Call` into a suspended
//! chain) can be *unwound* into guest-resident shadow state and later *rewound* back into
//! execution, byte-for-byte. The codec is exactly the §2 mechanism:
//!
//! * a **state word** (`NORMAL | UNWINDING | REWINDING`) in the window,
//! * a per-fiber **shadow stack** in the window (DURABILITY.md §12.7),
//! * **unwind** = after a may-suspend call, if `UNWINDING`, spill the live values +
//!   resume id and return out to the host;
//! * **rewind** = in the prologue, if `REWINDING`, `br_table` on the saved resume id,
//!   reload the live values, and continue from the resume point.
//!
//! # Scope (arbitrary single-vCPU CFGs)
//!
//! The transform handles an arbitrary-CFG function (branches, loops, joins) with any
//! number of may-suspend operations, each either:
//!
//! * a **leaf** `call.cap` — the host performs the operation; on thaw the deepest frame
//!   reloads the saved result and flips the state word back to `NORMAL`; or
//! * a **propagated** `Call` to a may-suspend callee (a function that transitively
//!   reaches a `call.cap`) — frames stack up across the call chain. On thaw a non-deepest
//!   frame reloads its pre-call live set and **re-issues the call** (leaving the state
//!   `REWINDING` so the callee rewinds in turn); only the innermost leaf flips to
//!   `NORMAL`. This is the DURABILITY.md §12.7 "re-issue vs. continue" branch (R8). The
//!   re-issued call then polls like the original: the program runs on inside it, and a later
//!   freeze can land beneath it.
//!
//! Each original block is split at its suspend ops into forward segments; branch targets
//! are remapped to the target block's first segment; a `br_table` in the prologue dispatch
//! routes a thaw to the resume point that was in flight (one arm per point). A function is
//! **may-suspend** iff it contains a `call.cap` or (transitively) a `Call` to a may-suspend
//! function; only may-suspend functions are instrumented. Covered end-to-end on the real
//! interpreter (`tests/roundtrip.rs`, `chain.rs`, `multipoint.rs`, `multiblock.rs`) and
//! across the interp/JIT differential (`crates/temen/tests/durable_jit.rs`).
//!
//! Each resume point spills only its **minimal live set** — the values used after the op
//! (block-local SSA makes this a backward scan within the block), plus a propagated call's
//! own operands. An unmodelled instruction in the tail falls back to spilling the whole
//! range, so the analysis never under-spills.
//!
//! A **`call.dyn`** to a may-suspend target is instrumented too (R8, the fork-critical
//! case): the target is a runtime table index, so the analysis taints *by signature* (a
//! `call.dyn` of type `T` suspends iff some function of type `T` does — the natural table
//! admits any signature match), and the site re-issues the call on thaw with the reloaded index
//! (`SuspendKind::PropagatedIndirect`), so the re-selected — and, by the taint rule, instrumented
//! — callee rewinds in turn. An unwind the runtime may *decline* (a fork) narrows that to the
//! functions the module takes the address of, and fronts every other may-suspend function's table
//! slot with a **barrier** that declines an unwind beneath it ([`IndirectReach`]). Out of scope
//! (rejected — the frame is replaced, so there is no poll to unwind at): a **tail** call, direct or
//! indirect, into a may-suspend callee.
//!
//! The remaining extensions (DURABILITY.md §9) are fibers / multi-vCPU / STW (Phase 3).

#![forbid(unsafe_code)]

use temen_ir::{
    BinOp, Block, BlockIdx, CmpOp, Func, FuncIdx, FuncType, Inst, IntTy, LoadOp, Module, StoreOp,
    Terminator, TypeEntry, ValIdx, ValType,
};

/// Resolve an interned call type index (#922) into its [`FuncType`]. The call variants
/// (`CallIndirect`, `CapCall`, …) carry a `u32` index into the module's type section; a
/// well-formed (verified) module always resolves to a `TypeEntry::Func`. A miss returns a
/// static empty signature so callers stay total — the verifier already rejects a dangling
/// or non-func index, so this only ever fires on already-invalid IR.
fn sig_of(types: &[TypeEntry], t: u32) -> &FuncType {
    static EMPTY: FuncType = FuncType {
        params: Vec::new(),
        results: Vec::new(),
    };
    match types.get(t as usize) {
        Some(TypeEntry::Func(ft)) => ft,
        _ => &EMPTY,
    }
}

// `FuncIdx` is used by `SuspendKind::Propagated` below.

/// The reserved self-namespace op for `svc.poll` (IMPORTS.md §3.6). A local copy — this crate
/// depends only on `temen-ir` — pinned equal to `temen_interp::CAP_SELF_SVC_POLL` by a dev-test
/// (`tests/serve.rs`).
pub use temen_ir::durable_abi::SVC_POLL_OP;
/// The reserved self-namespace op for `svc.wait`; pinned equal to `temen_interp::CAP_SELF_SVC_WAIT`.
pub use temen_ir::durable_abi::SVC_WAIT_OP;

// ---- State word values (the §2 state machine) ----

/// Freeze **armed** (the deterministic mid-run trigger): the run executes normally, but the runtime
/// counts down [`ARM_COUNTDOWN_OFF`] at each **fiber safepoint** (`cont.resume`/`suspend`) and, on
/// reaching 0, promotes the word to [`STATE_UNWINDING`] so that op's trailing poll begins the freeze.
/// Transparent to the instrumented IR — every emitted poll/prologue tests only `UNWINDING`/`REWINDING`,
/// so an `ARMED` run reads as `NORMAL` until the runtime promotes it. Lets a single-threaded test
/// freeze *after N fiber safepoints of forward progress* (e.g. after a fiber has been recycled), which
/// the freeze-before-start harness cannot; it also models an async controller flipping `UNWINDING` from
/// another thread, which the existing mechanism already picks up at the next poll. Both backends count
/// the same set (the fiber ops, routed through runtime thunks), so an armed freeze lands at the same
/// safepoint on each — `call.cap` is not counted (no cross-backend choke; its freeze is already
/// reachable at the first safepoint).
pub use temen_ir::durable_abi::STATE_ARMED;
/// Normal forward execution; polls and prologues fall straight through.
pub use temen_ir::durable_abi::STATE_NORMAL;
/// Thaw in progress: every prologue rebuilds its frame from the shadow stack.
pub use temen_ir::durable_abi::STATE_REWINDING;
/// Freeze in progress: every poll after a may-suspend call unwinds out to the host.
pub use temen_ir::durable_abi::STATE_UNWINDING;

// ---- Durable runtime region layout ----
//
// The control words + shadow arena occupy a **reserved low slice** `[0, arena.end)` of the
// domain's *own* window (the arena is the module's own declaration, `Memory::shadow`, one
// definition of placement — #1503 / INVARIANTS.md #16); the guest's memory is `[arena.end, window)`. This is the wasm shadow-stack
// convention (runtime metadata + call stack below `__heap_base`, the program's heap above it):
// the reserve is part of the guest's memory allotment, and a cooperating toolchain bases the
// guest's data/heap at the arena `end` so the two never overlap (see
// `transform_module_assume_confined`).
//
// This is *placement*, not an isolation boundary: the window is per-domain and the runtime
// masks every access into it, so a guest that writes the reserve can only corrupt its own
// durability — never another domain or the host — and that fails safe (a forged resume id
// hits the `br_table` default → `Unreachable`; a wild shadow-SP stays masked in-window; the
// host validates the artifact on restore). Hardening the reserve against an *adversarial*
// guest (a guard-paged, per-fiber placement, DURABILITY.md §12.7) is optional
// defense-in-depth, not required for a cooperating toolchain.

/// Window byte offset of the `i64` **back-edge arm countdown** — the number of loop back-edges
/// (branch terminators) still to pass before an [`STATE_ARMED`] run promotes itself to
/// [`STATE_UNWINDING`], so a loop-header poll begins the freeze. The deterministic trigger for the
/// Phase-4 Slice A back-edge polls, separate from [`ARM_COUNTDOWN_OFF`] (which counts only fiber
/// safepoints) so an ordinary or fiber-armed run is byte-identical (this slot stays 0). Lives in the
/// reserve's `[24, 64)` gap.
pub use temen_ir::durable_abi::ARM_BACKEDGE_OFF;
/// Window byte offset of the `i64` **arm countdown** — the number of fiber safepoints still to pass
/// before an [`STATE_ARMED`] run promotes itself to [`STATE_UNWINDING`]. Decremented by the runtime at
/// each fiber safepoint (`cont.resume`/`suspend`); inert unless the state word is `ARMED`. Lives in the
/// reserve's previously-unused `[16, 64)` gap, so it is byte-identical to before for any run that never
/// arms (countdown stays 0).
pub use temen_ir::durable_abi::ARM_COUNTDOWN_OFF;
/// Window byte offset of the `i8` **freeze-on-quiesce** flag (DURABILITY.md §13.4 slice 4c-bis):
/// non-zero arms the runtime to freeze the instant the run would otherwise block on
/// `svc.wait`-parked consumers only — a server idle in its accept loop, which no safepoint or
/// back-edge countdown can reach (a parked vCPU runs no ops). Distinct from the two countdown
/// slots (which trigger mid-execution); this triggers at quiescence. Lives in the reserve's
/// `[32, 64)` gap, so an unarmed run stays byte-identical. Must equal `temen-interp`'s copy.
pub use temen_ir::durable_abi::ARM_QUIESCE_OFF;
/// §12.8 concurrent-thaw stage 1: bytes reserved at a context region's base before its shadow frames —
/// the 8-byte shadow-SP word, the 4-byte thaw state word at [`STATE_IN_REGION_OFF`] and the 4-byte
/// re-issue word at [`REISSUE_IN_REGION_OFF`]. [`shadow_frame_base`]-equivalents in every backend start here. Must equal
/// `temen-interp`/`temen-jit`'s copy.
pub use temen_ir::durable_abi::REGION_HEADER_LEN;
/// #1672: byte offset of the per-context **re-issue** word within a context's region — set by the
/// runtime when it abandons a host call that would have parked under a landing freeze, and moved into
/// that call's shadow frame by its unwind, so the thaw re-issues the call instead of reloading a result.
pub use temen_ir::durable_abi::REISSUE_IN_REGION_OFF;
/// Window byte offset of the `i64` shadow-stack pointer (a window byte offset itself).
pub use temen_ir::durable_abi::SHADOW_SP_OFF;
/// §12.8 concurrent-thaw stage 1: byte offset of the per-context **thaw** state word
/// (`REWINDING`/`NORMAL`) **within a context's region** — just past the 8-byte in-region shadow-SP word
/// at the region base, addressed via the `durable.shadow_base` register (like the SP word). Each frozen
/// vCPU rewinds against its *own* thaw word, so the thaw can run them as concurrent OS threads with no
/// shared word: one vCPU finishing its rewind (flipping its word to `NORMAL`) can't disturb a sibling
/// still `REWINDING`, and a forward vCPU's callee prologue can't read another's `REWINDING`. The
/// **freeze** state (`UNWINDING`) stays at the global [`STATE_OFF`] — a freeze is stop-the-world, so the
/// single word is the natural broadcast every poll reads. Must equal `temen-interp`/`temen-jit`'s copy.
pub use temen_ir::durable_abi::STATE_IN_REGION_OFF;
/// Window byte offset of the `i32` state word.
pub use temen_ir::durable_abi::STATE_OFF;

/// Per-context shadow-region stride: context `i` owns `[arena.region_base(i), +SHADOW_STRIDE)`
/// (§12.8 4A.5). The transform itself never addresses a region (it emits `durable.shadow_base`-relative
/// loads the runtime resolves), but the [`write_thaw_state`] host helper indexes a context's region.
pub use temen_ir::durable_abi::SHADOW_STRIDE;
/// The shadow arena — one definition of placement (see the region-layout note above) — and the end
/// of the always-live control words below it.
pub use temen_ir::durable_abi::{ShadowArena, DURABLE_CONTROL_END, WAIT_FROZEN};

// Block layout of an instrumented function with `S` forward segments (each original
// block is split at its suspend ops into `points+1` segments; non-suspend blocks are one
// segment) and `P` resume points total (an 8-block shape for one block / one point):
//   0                  PROLOGUE  — dispatch on the state word, then enter segment 0 of blk 0
//   1 ..= S            forward segments (each original block's segments, in block order)
//   1+S                DISPATCH  — read resume id, br_table to an arm
//   2+S ..= 1+S+2P     UNWIND_g  — a (check, spill) pair per point: trap if the push would
//                                  overflow the reserve, else spill + return placeholder
//   2+S+2P ..= 1+S+3P  ARM_g     — reload + resume, per resume point
//   2+S+3P             TRAP      — forged/reserved resume id, or shadow-stack overflow

/// Reasons the Phase-1 transform declines a module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum TransformError {
    /// A function uses `call.cap` but the module declares no memory window — the
    /// shadow stack and state word have nowhere to live.
    NoMemory,
    /// The declared window is too small to hold the durable region + a shadow frame.
    MemoryTooSmall,
    /// A function must be instrumented but the module's `memory` declares no shadow arena
    /// (`memory N shadow BASE END`): there is nowhere to keep shadow frames. Placement is the
    /// module's to declare (INVARIANTS.md #16), so this fails closed rather than defaulting.
    NoShadowArena,
    /// A `call.cap`-bearing function is outside the Phase-1 shape (not a single block,
    /// not exactly one `call.cap`, or not a `return` terminator).
    UnsupportedShape,
    /// A prefix instruction's result type isn't modelled by the Phase-1 transform
    /// (e.g. SIMD, conversions, concurrency ops before the call).
    UnsupportedInst,
    /// The module is being instrumented via the strict [`transform_module`] path but a
    /// function uses a guest linear-memory instruction (load/store/atomic), which could
    /// alias the reserved durable region `[0, arena.end)` (R9). The strict path fails
    /// closed for *untrusted* modules. A durable module from a cooperating toolchain that
    /// reserves `[0, arena.end)` (basing the guest's data/heap at the arena `end`)
    /// should instead use [`transform_module_assume_confined`]. (Cap-mediated window effects
    /// — e.g. a Memory capability's map/unmap — are a separate facet the embedder withholds.)
    GuestUsesMemory,
}

/// How [`transform`] instruments a module — the one transform, parameterized by who unwinds it.
#[derive(Clone, Copy)]
pub struct TransformOpts<'a> {
    /// Refuse a module any of whose functions touch linear memory (R9) — the strict path for an
    /// untrusted module ([`transform_module`]). Off for a cooperating toolchain's module
    /// ([`transform_module_assume_confined`]).
    pub enforce_r9: bool,
    /// Which suspendable operations (a capability call, a fiber switch, a `thread.join`, an
    /// `atomic.wait`) are suspend points. `None` instruments every one: a durable domain can be
    /// caught by a freeze inside any of them. `Some(site)` instruments only the ops `site` admits,
    /// and so only the functions that can reach one: a **fork** (#1768, FORK.md §9.5) is an unwind
    /// the guest's own call triggers, so only the calls that can fork need a poll, and a program
    /// that cannot reach one is left byte-identical. The runtime unwinds only at an op this admitted.
    pub sites: Option<&'a dyn Fn(&Inst) -> bool>,
    /// Poll at every loop header too (Phase-4 Slice A: an async freeze lands in a poll-free loop
    /// at bounded latency). An unwind the guest's own call starts is never inside a loop body, so
    /// a fork needs none.
    pub loop_polls: bool,
    /// The unwind duplicates the running thread **alive** — a fork — rather than freezing it into a
    /// snapshot, so the thread state an engine keeps outside the window travels with it: the vCPU TLS
    /// register (`vcpu.tls.*`) needs no frame slot, and the ops on it instrument like any other. A
    /// snapshot carries the window alone, so for a freeze (`false`) those ops fail closed
    /// (`UnsupportedInst`) rather than thaw with the register lost.
    pub carries_thread: bool,
    /// Which functions a `call.dyn` can select, for the may-suspend taint (see [`IndirectReach`]).
    pub indirect: IndirectReach,
}

/// Which functions a `call.dyn` of a given signature can select — the may-suspend taint's reading
/// of an indirect call (R8), whose target is a runtime table index.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IndirectReach {
    /// **Any** function of that signature. A funcref is a forgeable integer (§3c) and the table maps
    /// every function into a slot, so this is the only reading sound for every program: every
    /// may-suspend function taints its signature. The reading for an unwind the runtime must be
    /// able to complete wherever it lands — a freeze.
    Signature,
    /// Only a function whose address the module **takes** (`ref.func`) — the indirect calls a
    /// program makes through the funcrefs it was given. Every other may-suspend function whose
    /// signature is left untainted is one no instrumented `call.dyn` can select, so an index that
    /// selects it is forged (or baked into the data image, which a linked module no longer records)
    /// and lands from an uninstrumented site: its table slot is fronted by a **barrier** that holds
    /// the running context's shadow stack occupied ([`BARRIER_SP`]) while the real body runs. The
    /// body moves to the end of the module, and every static reference enters it directly
    /// ([`Instrumented::body`]). The runtime's half of the contract: it starts an unwind only when
    /// the running context's shadow-SP is its empty frame base ([`ShadowArena::frame_base`]), so a
    /// barrier below the call declines it. The reading for an unwind the runtime may decline — a
    /// fork (FORK.md §9.5), which one call site triggers: it instruments the calls that can reach a
    /// fork, not every function that shares a signature with one.
    AddressTaken,
}

/// The shadow-SP a barrier ([`IndirectReach::AddressTaken`]) holds while its body runs: `0`, below
/// every arena (which starts at or above [`DURABLE_CONTROL_END`]), so it never reads as a context's
/// empty frame base — and a push at it would fault on the NULL guard rather than land anywhere.
pub const BARRIER_SP: u64 = 0;
const _: () = assert!(BARRIER_SP < DURABLE_CONTROL_END);

/// A module [`transform`] instrumented.
#[derive(Clone, Debug)]
pub struct Instrumented {
    pub module: Module,
    /// `body[f]` is where input function `f`'s body lives in `module`: `f` itself, unless a
    /// barrier took its place ([`IndirectReach::AddressTaken`]) and the body moved to the end. Enter
    /// a function at its body to run it directly (a run's entry); its own index is the table slot a
    /// `call.dyn` lands on.
    pub body: Vec<FuncIdx>,
}

impl<'a> TransformOpts<'a> {
    /// A durable domain's instrumentation: every suspendable op, loop polls, strict R9.
    pub const DURABLE: TransformOpts<'static> = TransformOpts {
        enforce_r9: true,
        sites: None,
        loop_polls: true,
        carries_thread: false,
        indirect: IndirectReach::Signature,
    };

    /// A **fork**'s instrumentation (#1768, FORK.md §9.5): the unwind a guest's own call starts, at
    /// the ops `sites` admits — so no loop polls; the thread travels alive (`carries_thread`); and,
    /// the runtime being free to decline it, a `call.dyn` reads only the address-taken functions
    /// ([`IndirectReach::AddressTaken`]). A forking program is a cooperating toolchain's, which keeps
    /// its data and heap clear of its own declared arena (the verifier holds its data segments to
    /// that; R9's contract covers the rest), so R9 is not enforced.
    pub fn fork(sites: &'a dyn Fn(&Inst) -> bool) -> Self {
        TransformOpts {
            enforce_r9: false,
            sites: Some(sites),
            loop_polls: false,
            carries_thread: true,
            indirect: IndirectReach::AddressTaken,
        }
    }

    /// Whether `x` is a suspend point under these options.
    fn is_site(&self, x: &Inst) -> bool {
        is_suspendable_op(x) && self.sites.is_none_or(|site| site(x))
    }
}

/// The operations that can suspend a frame to the host or another stack: `call.cap` suspends to
/// the host; a fiber `cont.resume`/`suspend` switches stacks and is a freeze safepoint too
/// (`cont.new` alone merely allocates); `call.import` / `call.import.dyn` / `call.sym` are
/// capability calls bound at run time (IMPORTS.md) — the same host suspend as `call.cap` (#1300
/// Phase 2); a vCPU blocked in `thread.join` or `atomic.wait` is a safepoint (§12.8).
fn is_suspendable_op(x: &Inst) -> bool {
    matches!(
        x,
        Inst::CapCall { .. }
            | Inst::CallImport { .. }
            | Inst::CallImportDyn { .. }
            | Inst::CallSym { .. }
            | Inst::ContResume { .. }
            | Inst::Suspend { .. }
            | Inst::ThreadJoin { .. }
            | Inst::MemoryWait { .. }
    )
}

/// Instrument every may-suspend function in `m` for freeze/thaw. Functions that can
/// never suspend are returned unchanged. The result is ordinary IR; run it through
/// `temen_verify::verify_module` before executing.
pub fn transform_module(m: &Module) -> Result<Module, TransformError> {
    transform(m, &TransformOpts::DURABLE).map(|t| t.module)
}

/// Like [`transform_module`], but **allows the guest to use linear memory**, on the caller's
/// guarantee that the guest is confined to its usable region `[arena.end, window)` and
/// never touches the reserved durable slice `[0, arena.end)` (its own declared arena's end).
///
/// This is the intended path for durable modules produced by a **cooperating toolchain**:
/// just as a wasm toolchain reserves low memory for the shadow stack and bases the heap at
/// `__heap_base`, the producer reserves `[0, arena.end)` and bases the guest's data/heap at
/// the arena `end`. The reserve is budget-accounted (the declared window must be ≥ `arena.end`).
/// The contract is *not* statically enforced here — a guest that
/// violates it can corrupt only its own durability, and fails safe (see the region notes) —
/// so prefer [`transform_module`] (fails closed, no memory) for *untrusted* modules.
pub fn transform_module_assume_confined(m: &Module) -> Result<Module, TransformError> {
    transform(
        m,
        &TransformOpts {
            enforce_r9: false,
            ..TransformOpts::DURABLE
        },
    )
    .map(|t| t.module)
}

/// Instrument `m` as `opts` says (see [`TransformOpts`]); [`transform_module`] and
/// [`transform_module_assume_confined`] are its two durable presets.
pub fn transform(m: &Module, opts: &TransformOpts) -> Result<Instrumented, TransformError> {
    let enforce_r9 = opts.enforce_r9;
    let func_results: Vec<Vec<ValType>> = m.funcs.iter().map(|f| f.results.clone()).collect();
    let targets = indirect_targets(&m.funcs, opts.indirect);
    let may_suspend = compute_may_suspend(&m.funcs, &m.types, opts, &targets);
    let tainted_sigs = tainted_signatures(&m.funcs, &may_suspend, &targets);
    let any_instrumented = may_suspend.iter().any(|&s| s);

    // R9 enforcement: the durable region shares the window with guest memory at fixed low
    // addresses, with nothing confining the guest away from it. Rather than risk silent
    // mutual corruption, refuse to instrument a module any of whose functions touch linear
    // memory. (The generated/tested durable guests are pure SSA + `call.cap`, so they pass;
    // the relocation that would lift this restriction is §12.7 future work.)
    if enforce_r9
        && any_instrumented
        && m.funcs
            .iter()
            .any(|f| f.blocks.iter().any(|b| b.insts.iter().any(is_guest_mem_op)))
    {
        return Err(TransformError::GuestUsesMemory);
    }

    // The shadow arena is the module's declaration (INVARIANTS.md #16): an instrumented module must
    // have one; a module with nothing to instrument needs none — and gets none (no default).
    let arena = if any_instrumented {
        let mem = m.memory.ok_or(TransformError::NoMemory)?;
        mem.shadow.ok_or(TransformError::NoShadowArena)?
    } else {
        ShadowArena::EMPTY
    };

    let mut out = m.clone();
    let mut max_frame = 0u64;

    for (i, f) in m.funcs.iter().enumerate() {
        if may_suspend[i] {
            let (nf, frame_size) = transform_func(
                f,
                &func_results,
                &may_suspend,
                &tainted_sigs,
                &m.types,
                opts,
            )?;
            out.funcs[i] = nf;
            max_frame = max_frame.max(frame_size);
        }
    }

    if any_instrumented {
        let mem = out.memory.ok_or(TransformError::NoMemory)?;
        // The reserved region `[0, arena.end)` must fit in the declared window (it is part of
        // the guest's allotment; guest memory is the remainder `[arena.end, window)`), and a
        // single shadow frame must fit in one context's region.
        // A live call chain stacks one frame per suspended activation; the region bounds the
        // total depth, and the UNWIND check traps a chain deeper than that (#1683).
        if mem.size() < arena.end || REGION_HEADER_LEN + max_frame > SHADOW_STRIDE {
            return Err(TransformError::MemoryTooSmall);
        }
    }

    // A may-suspend function whose signature no selectable may-suspend function shares is one no
    // instrumented `call.dyn` can reach: under `AddressTaken`, its slot gets a barrier and its body
    // moves to the end (see [`IndirectReach::AddressTaken`]). Under `Signature` every may-suspend
    // function taints its own signature, so there is none.
    let mut body: Vec<FuncIdx> = (0..m.funcs.len() as FuncIdx).collect();
    for (g, f) in m.funcs.iter().enumerate() {
        if may_suspend[g] && !tainted_sigs.iter().any(|s| sig_matches(s, f)) {
            let moved = out.funcs.len() as FuncIdx;
            let real = std::mem::replace(&mut out.funcs[g], barrier(f, moved));
            out.funcs.push(real);
            body[g] = moved;
        }
    }
    retarget_static_refs(&mut out, &body);
    Ok(Instrumented { module: out, body })
}

/// The functions a `call.dyn` can select under `reach` (see [`IndirectReach`]).
fn indirect_targets(funcs: &[Func], reach: IndirectReach) -> Vec<bool> {
    let mut targets = vec![reach == IndirectReach::Signature; funcs.len()];
    if reach == IndirectReach::AddressTaken {
        for x in funcs
            .iter()
            .flat_map(|f| f.blocks.iter().flat_map(|b| &b.insts))
        {
            if let Inst::RefFunc { func } = x {
                if let Some(t) = targets.get_mut(*func as usize) {
                    *t = true;
                }
            }
        }
    }
    targets
}

/// Whether `f` has signature `s`.
fn sig_matches(s: &FuncType, f: &Func) -> bool {
    s.params == f.params && s.results == f.results
}

/// The **barrier** that takes a function's table slot when its body moves to `body`
/// ([`IndirectReach::AddressTaken`]): it holds the running context's shadow stack occupied
/// ([`BARRIER_SP`]) for the duration of a call to the body, then restores it. It has the body's
/// signature and returns its results, so a `call.dyn` that selects the slot computes exactly what it
/// did — only an unwind beneath it is declined.
fn barrier(f: &Func, body: FuncIdx) -> Func {
    let mut b = Bb::new(f.params.clone());
    let sp_a = b.one(Inst::DurableShadowBase);
    let sp = b.one(load(LoadOp::I64, sp_a, 0));
    let occupied = b.one(Inst::ConstI64(BARRIER_SP as i64));
    b.zero(store(StoreOp::I64, sp_a, occupied, 0));
    let args = (0..f.params.len() as ValIdx).collect();
    let results = b.many(Inst::Call { func: body, args }, f.results.len());
    b.zero(store(StoreOp::I64, sp_a, sp, 0));
    Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks: vec![b.finish(Terminator::Return(results))],
    }
}

/// Point every **static** reference to a function at its body (`body`, see [`Instrumented`]): the
/// calls, a thread's entry, the exports and impl-export ops a host enters directly, and the debug
/// info that describes the body. A `ref.func` is left alone: its value is a table slot — a barrier's,
/// for a moved function — and none names one the analysis gave a barrier.
fn retarget_static_refs(m: &mut Module, body: &[FuncIdx]) {
    let to = |f: &mut FuncIdx| {
        if let Some(&b) = body.get(*f as usize) {
            *f = b;
        }
    };
    for f in &mut m.funcs {
        for blk in &mut f.blocks {
            for inst in &mut blk.insts {
                if let Inst::Call { func, .. } | Inst::ThreadSpawn { func, .. } = inst {
                    to(func);
                }
            }
            if let Terminator::ReturnCall { func, .. } = &mut blk.term {
                to(func);
            }
        }
    }
    for e in &mut m.exports {
        to(&mut e.func);
    }
    for e in &mut m.impl_exports {
        e.ops.iter_mut().for_each(to);
    }
    if let Some(di) = &mut m.debug_info {
        di.locs.iter_mut().for_each(|l| to(&mut l.func));
        di.func_names.iter_mut().for_each(|n| to(&mut n.func));
        for v in &mut di.vars {
            if v.func != temen_ir::GLOBAL_SCOPE {
                to(&mut v.func);
            }
        }
    }
}

/// A guest linear-memory instruction (one that reads or writes a window address). These
/// can alias the durable region, so they are rejected in an instrumented module (R9). An
/// `AtomicFence` carries no address and so is not included.
fn is_guest_mem_op(inst: &Inst) -> bool {
    matches!(
        inst,
        Inst::Load { .. }
            | Inst::Store { .. }
            | Inst::AtomicLoad { .. }
            | Inst::AtomicStore { .. }
            | Inst::AtomicRmw { .. }
            | Inst::AtomicCmpxchg { .. }
            | Inst::V128Load { .. }
            | Inst::V128Store { .. }
            | Inst::MemCopy { .. }
            | Inst::MemMove { .. }
            | Inst::MemFill { .. }
            | Inst::MemoryWait { .. } // reads the window value at `addr`
    )
}

/// The value operands an instruction reads, or `None` for a variant this pass does not
/// model — in which case liveness conservatively assumes it uses everything (so we never
/// drop a still-live value). Covers the set `result_types` admits (anything else is already
/// rejected as `UnsupportedInst` before liveness runs).
fn inst_operands(i: &Inst) -> Option<Vec<ValIdx>> {
    use Inst::*;
    Some(match i {
        ConstI32(_) | ConstI64(_) | ConstF32(_) | ConstF64(_) | ConstV128(_) | RefFunc { .. } => {
            vec![]
        }
        IntBin { a, b, .. } | IntCmp { a, b, .. } | FBin { a, b, .. } | FCmp { a, b, .. } => {
            vec![*a, *b]
        }
        IntUn { a, .. }
        | FUn { a, .. }
        | Eqz { a, .. }
        | Convert { a, .. }
        | FToISat { a, .. }
        | FToITrap { a, .. }
        | IToFConv { a, .. }
        | Cast { a, .. }
        | Splat { a, .. }
        | ExtractLane { a, .. }
        | VIntUn { a, .. }
        | VFloatUn { a, .. }
        | VPopcnt { a, .. }
        | VWiden { a, .. }
        | VConvert { a, .. }
        | VExtAddPairwise { a, .. }
        | VAnyTrue { a, .. }
        | VAllTrue { a, .. }
        | VBitmask { a, .. }
        | VNot { a, .. } => vec![*a],
        Fma { a, b, c, .. } | VFma { a, b, c, .. } => vec![*a, *b, *c],
        ReplaceLane { a, b, .. }
        | VIntBin { a, b, .. }
        | VIntCmp { a, b, .. }
        | VFloatBin { a, b, .. }
        | VFloatCmp { a, b, .. }
        | VPMinMax { a, b, .. }
        | VSatBin { a, b, .. }
        | VAvgr { a, b, .. }
        | VDot { a, b, .. }
        | VDotI8 { a, b, .. }
        | VExtMul { a, b, .. }
        | VQ15MulrSat { a, b, .. }
        | VNarrow { a, b, .. }
        | VBitBin { a, b, .. }
        | Shuffle { a, b, .. }
        | Swizzle { a, b, .. } => vec![*a, *b],
        VShift { a, amt, .. } => vec![*a, *amt],
        Bitselect { a, b, mask } => vec![*a, *b, *mask],
        DataSym { .. } | DataSelf { .. } | DataTop | CapSelfTypeId { .. } | ExportHandle { .. } => {
            vec![]
        }
        CapSelfCovers { handle, .. } => vec![*handle],
        MemCopy { dst, src, len } | MemMove { dst, src, len } => vec![*dst, *src, *len],
        MemFill { dst, val, len } => vec![*dst, *val, *len],
        Select { cond, a, b } => vec![*cond, *a, *b],
        Load { addr, .. } | AtomicLoad { addr, .. } | V128Load { addr, .. } => vec![*addr],
        Store { addr, value, .. }
        | AtomicStore { addr, value, .. }
        | AtomicRmw { addr, value, .. }
        | V128Store { addr, value, .. } => vec![*addr, *value],
        AtomicCmpxchg {
            addr,
            expected,
            replacement,
            ..
        } => vec![*addr, *expected, *replacement],
        AtomicFence { .. } => vec![],
        MemoryWait {
            addr,
            expected,
            timeout,
            ..
        } => vec![*addr, *expected, *timeout],
        MemoryNotify { addr, count } => vec![*addr, *count],
        Call { args, .. } => args.clone(),
        CapCall { handle, args, .. }
        | CallImportDyn { handle, args, .. }
        | CallSym { handle, args, .. } => {
            let mut v = Vec::with_capacity(args.len() + 1);
            v.push(*handle);
            v.extend_from_slice(args);
            v
        }
        CallImport { args, .. } => args.clone(),
        CallIndirect { idx, args, .. } => {
            let mut v = Vec::with_capacity(args.len() + 1);
            v.push(*idx);
            v.extend_from_slice(args);
            v
        }
        ContNew { func, sp } => vec![*func, *sp],
        ContResume { k, arg, .. } => vec![*k, *arg],
        Suspend { value } => vec![*value],
        _ => return None,
    })
}

/// The value operands a terminator reads (a closed set, all handled).
fn term_operands(t: &Terminator) -> Vec<ValIdx> {
    match t {
        Terminator::Br { args, .. } => args.clone(),
        Terminator::BrIf {
            cond,
            then_args,
            else_args,
            ..
        } => {
            let mut v = vec![*cond];
            v.extend_from_slice(then_args);
            v.extend_from_slice(else_args);
            v
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let mut v = vec![*idx];
            for (_, a) in targets {
                v.extend_from_slice(a);
            }
            v.extend_from_slice(&default.1);
            v
        }
        Terminator::Return(vals) => vals.clone(),
        Terminator::ReturnCall { args, .. } => args.clone(),
        Terminator::ReturnCallIndirect { idx, args, .. } => {
            let mut v = vec![*idx];
            v.extend_from_slice(args);
            v
        }
        Terminator::Unreachable => vec![],
    }
}

/// The block targets a terminator branches to (a closed set; tail calls / returns carry none).
fn term_targets(t: &Terminator) -> Vec<BlockIdx> {
    match t {
        Terminator::Br { target, .. } => vec![*target],
        Terminator::BrIf {
            then_blk, else_blk, ..
        } => vec![*then_blk, *else_blk],
        Terminator::BrTable {
            targets, default, ..
        } => {
            let mut v: Vec<BlockIdx> = targets.iter().map(|(t, _)| *t).collect();
            v.push(default.0);
            v
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => vec![],
    }
}

/// Mark each function that can suspend: it contains a `call.cap` (or fiber/thread/futex
/// safepoint), or (transitively) a **direct** `Call`, or an **indirect** `call.dyn`
/// whose target could suspend. A least-fixed-point over the call graph.
///
/// **R8 — `call.dyn`.** The target is a runtime table index, so the transform can't
/// name the callee statically. But dispatch is signature-checked and the natural table
/// maps *every* function into a slot (and `Jit.install` can add more at run time), so the
/// only sound static rule is by **signature**: a `call.dyn` of type `T` can reach any
/// function whose signature equals `T`, hence it suspends iff **some may-suspend function
/// shares its signature**. This is the ceiling of static precision for an unwind that must
/// complete wherever it lands — there is no element/table section in the IR to narrow it
/// (DURABILITY.md §6; the breadth cost is R7). An unwind the runtime may decline reads only the
/// address-taken functions as selectable (`targets`), and a barrier declines it beneath any other
/// ([`IndirectReach::AddressTaken`]). The taint set grows with `ms`, so it is re-read each fixpoint
/// round (a newly-may-suspend function taints its own signature). Marking the caller (rather than
/// ignoring the indirect call) is what flips R8 from fail-**open** — silent under-instrumentation —
/// to sound: `transform_func` then either instruments the site or fails the module closed.
fn compute_may_suspend(
    funcs: &[Func],
    types: &[TypeEntry],
    opts: &TransformOpts,
    targets: &[bool],
) -> Vec<bool> {
    let mut ms: Vec<bool> = funcs
        .iter()
        .map(|f| {
            f.blocks
                .iter()
                .any(|b| b.insts.iter().any(|x| opts.is_site(x)))
        })
        .collect();
    loop {
        // A `call.dyn` of type `ty` reaches a may-suspend target iff some already-may-suspend
        // function it can select (`targets`, per [`IndirectReach`]) has that exact signature.
        // Re-derived each round from the live `ms`; collect the newly-tainted functions first, then
        // apply.
        let sigs = tainted_signatures(funcs, &ms, targets);
        let tainted = |ty: u32| sigs.contains(sig_of(types, ty));
        let to_mark: Vec<usize> = funcs
            .iter()
            .enumerate()
            .filter(|&(i, f)| {
                !ms[i]
                    && f.blocks.iter().any(|b| {
                        b.insts.iter().any(|x| match x {
                            Inst::Call { func, .. } => ms[*func as usize],
                            Inst::CallIndirect { ty, .. } => tainted(*ty),
                            _ => false,
                        })
                        // A direct or indirect **tail** call into a may-suspend callee also suspends
                        // (both rejected by `transform_func` as out of scope — a replaced frame has
                        // no poll to unwind at — but marked so the module fails closed, not silently
                        // under-instrumented).
                        || match &b.term {
                            Terminator::ReturnCall { func, .. } => ms[*func as usize],
                            Terminator::ReturnCallIndirect { ty, .. } => tainted(*ty),
                            _ => false,
                        }
                    })
            })
            .map(|(i, _)| i)
            .collect();
        if to_mark.is_empty() {
            break;
        }
        for i in to_mark {
            ms[i] = true;
        }
    }
    ms
}

/// The distinct signatures of the may-suspend functions a `call.dyn` can select (`targets`) — the
/// tainted set a `call.dyn` checks its type against (see [`compute_may_suspend`]). Computed once
/// from the final `ms` and threaded into `transform_func` so it recognizes indirect suspend sites
/// the same way.
fn tainted_signatures(funcs: &[Func], ms: &[bool], targets: &[bool]) -> Vec<temen_ir::FuncType> {
    let mut sigs: Vec<temen_ir::FuncType> = Vec::new();
    for (i, f) in funcs.iter().enumerate() {
        if ms[i] && targets[i] {
            let ty = temen_ir::FuncType {
                params: f.params.clone(),
                results: f.results.clone(),
            };
            if !sigs.contains(&ty) {
                sigs.push(ty);
            }
        }
    }
    sigs
}

/// The distinct signatures a *program* instruments for suspension — i.e. every `call.dyn`
/// of one of these types has a poll/unwind seam. Computed on the program's (pre-transform)
/// functions, this is the set a durable host stashes so it can gate later `Jit.compile`s
/// ([`unit_suspends_untainted`]). Exposed for the durable-JIT install fence (DURABILITY.md §12.5).
pub fn tainted_signatures_of(funcs: &[Func], types: &[TypeEntry]) -> Vec<temen_ir::FuncType> {
    let opts = TransformOpts::DURABLE;
    let targets = indirect_targets(funcs, opts.indirect);
    let ms = compute_may_suspend(funcs, types, &opts, &targets);
    tainted_signatures(funcs, &ms, &targets)
}

/// The durable-JIT install fence (DURABILITY.md §12.5, R8 fork-critical case). Returns `true`
/// — *reject this unit* — when the unit's entry (func 0, the `invoke`/`call.dyn` target)
/// transitively **suspends** yet its signature is **not** in `program_tainted` (the caller
/// program's [`tainted_signatures_of`]). In that case a program `call.dyn` reaching the
/// installed unit would be at an *un-instrumented* site (the taint is by-signature), so a freeze
/// mid-unit would silently lose the continuation on thaw. Fences fail-closed at compile so the
/// unit can never be installed. A non-suspending unit, or one whose signature the program taints
/// (the seam exists), returns `false` — admitted. Runs on the already-instrumented unit funcs;
/// the entry's `call.cap`/`cont.*` survive instrumentation, so `ms[0]` is unchanged, and the
/// transform preserves signatures. Injected into the durable `Host` as its taint gate.
pub fn unit_suspends_untainted(
    unit_funcs: &[Func],
    unit_types: &[TypeEntry],
    program_tainted: &[temen_ir::FuncType],
) -> bool {
    if unit_funcs.is_empty() {
        return false; // no entry to invoke; the empty-unit case is rejected elsewhere
    }
    let opts = TransformOpts::DURABLE;
    let targets = indirect_targets(unit_funcs, opts.indirect);
    let ms = compute_may_suspend(unit_funcs, unit_types, &opts, &targets);
    if !ms[0] {
        return false; // entry cannot suspend → no continuation to lose → safe
    }
    let entry = temen_ir::FuncType {
        params: unit_funcs[0].params.clone(),
        results: unit_funcs[0].results.clone(),
    };
    !program_tainted.contains(&entry)
}

/// The single may-suspend operation in an instrumented block.
enum SuspendKind {
    /// A host call (`call.cap` / `call.import` / `call.import.dyn` / `call.sym`); `op` is the original
    /// instruction. It is the deepest frame, and its thaw arm does one of two things. A call the host
    /// **performed** before the cut reloads its result. A call the runtime **abandoned** without effect
    /// (it would have parked while a freeze was landing, #1672) is **re-issued** with its reloaded
    /// operands. The runtime marks an abandoned call by setting the context's re-issue word
    /// ([`REISSUE_IN_REGION_OFF`]); the unwind spills that word into the frame and clears it.
    Leaf { op: Inst },
    /// `Call` to a may-suspend callee: re-issued on thaw so the callee rewinds in turn.
    Propagated { callee: FuncIdx, args: Vec<ValIdx> },
    /// `call.dyn` to a (by-signature, conservatively) may-suspend target (R8): the indirect
    /// analog of `Propagated`. Re-issued on thaw with the **reloaded table index** — the runtime
    /// re-selects the same slot, and because the taint rule instruments *every* function of that
    /// signature, whatever the `idx` resolves to has a `REWINDING`-aware prologue and rewinds in
    /// turn. `ty` reconstructs the op; `idx`/`args` are its block-local operands (spilled + reloaded).
    PropagatedIndirect {
        /// Interned call type index (#922) into the module type section — reconstructs the op.
        ty: u32,
        idx: ValIdx,
        args: Vec<ValIdx>,
    },
    /// `cont.resume` (resumer side): like a propagated call, **re-issued on thaw** so the fiber
    /// rewinds in turn and redelivers its `(status: i32, value: i64)` (slice 3.1.2). The
    /// re-issued resume reconstructs the fiber via its own rewind (the `Yield` re-park, slice
    /// 3.1.3) — until that lands, a thaw that actually re-enters a suspended fiber still relies
    /// on the fiber side being wired.
    Resume { k: ValIdx, arg: ValIdx },
    /// `thread.join` (§12.8 next slice): a vCPU blocked joining a child is a freeze safepoint too — the
    /// trailing poll unwinds it. Like a host call it carries the re-issue word (#1685): a join the freeze
    /// ended (the runtime returned without the child's result, or with the placeholder of a child that
    /// itself unwound) is **re-issued on thaw** against the re-spawned child; a join that got the
    /// child's real result reloads it, since that child is not re-run. `handle` is the joined vCPU
    /// handle's block-local index (spilled + reloaded).
    ThreadJoin { handle: ValIdx },
    /// `<ty>.atomic.wait` (§12.8 parked-vCPU slice): a vCPU blocked in a futex wait is a freeze
    /// safepoint too — the `thread_wait` thunk returns on observing `UNWINDING` (with `WAIT_FROZEN`),
    /// the trailing poll unwinds, and the status is spilled with the frame (#1769). On thaw a wait
    /// that completed before the cut delivers its status as it was; one the freeze ended is
    /// **re-issued** (like `thread.join`): the re-executed wait re-checks the guest value, so a wake
    /// that landed as a value change (already in the snapshot, or replayed by another re-run vCPU)
    /// resolves it immediately with `WAIT_NOT_EQUAL`. A re-issue that
    /// would still *park* on the single-worker thaw can't be satisfied (no concurrent notifier) and
    /// fails closed (`ThreadFault`, matching the interp's join-deadlock). `ty` reconstructs the op;
    /// `addr`/`expected`/`timeout` are its block-local operands (spilled + reloaded).
    MemoryWait {
        ty: IntTy,
        addr: ValIdx,
        expected: ValIdx,
        timeout: ValIdx,
    },
    /// `suspend` (fiber side): unwinds the fiber's stack like a leaf, and thaw **re-parks** the
    /// fiber — flips the state word to `NORMAL` (a parked fiber's suspend is the globally-deepest
    /// frozen frame) and re-executes `suspend` so control returns to the resumer awaiting a future
    /// `cont.resume` (slice 3.1.3). `value` is the suspended value's block-local index.
    Yield { value: ValIdx },
    /// A **serve op** (`call.cap CAP_SELF svc.poll/svc.wait` — DURABILITY.md §13.4 slice 4b): a
    /// serving domain frozen at (or parked in) its serve point is a freeze safepoint — the serve
    /// arm delivers an inert sentinel on observing `UNWINDING` (no drain, no park; the queue
    /// stays untouched for the snapshot's serve section), the trailing poll unwinds, and the op
    /// is **re-issued on thaw** (like `atomic.wait`): the re-executed drain runs against the
    /// *restored* queue — re-execution is the recovery, so the sentinel is never captured. A
    /// mid-handler freeze is refused up front (the serve epilogue's fail-closed gate), so this
    /// point is always the globally-deepest frozen frame on its thread: flip the state word to
    /// `NORMAL` itself, then reload the handle + args and re-execute. The op immediates
    /// (`type_id`/`op`/`sig`) reconstruct the instruction; `handle`/`args` are its block-local
    /// operands (spilled + reloaded).
    SvcServe {
        type_id: u32,
        op: u32,
        /// Interned call type index (#922) into the module type section — reconstructs the op.
        sig: u32,
        handle: ValIdx,
        args: Vec<ValIdx>,
    },
    /// A **loop-header poll** (Phase-4 Slice A): a state-word check prepended to a loop header's
    /// entry (the header dominates its body, so a poll-free compute loop is caught every iteration
    /// at bounded latency, closing the R6 latency caveat). It has no in-flight op — it is always
    /// the globally-deepest frozen frame (any may-suspend op in the body would have unwound at its
    /// own poll first), so on thaw it behaves like a leaf: flip the state word to `NORMAL`, reload
    /// the header's block params, and re-enter the header body.
    LoopHeader,
}

impl SuspendKind {
    /// The block-local value operands this op consumes — the values that must stay **live** (spilled
    /// and reloaded) across a freeze so the op can be re-issued on thaw. One definition for both the
    /// liveness marking and the re-issue arg mapping (#915), so a missed operand can't silently
    /// under-spill (a corrupted thaw). Immediates (`callee`/`ty`/`type_id`/`op`/`sig`) are not
    /// operands; `Leaf`/`LoopHeader` have none (they only reload their own result / block params).
    fn operands(&self) -> Vec<ValIdx> {
        match self {
            SuspendKind::Leaf { op } => {
                inst_operands(op).expect("a host call's operands are modeled")
            }
            SuspendKind::LoopHeader => Vec::new(),
            SuspendKind::Propagated { args, .. } => args.clone(),
            SuspendKind::PropagatedIndirect { idx, args, .. } => {
                let mut v = Vec::with_capacity(args.len() + 1);
                v.push(*idx); // the table index must be reloaded to re-select the slot
                v.extend_from_slice(args);
                v
            }
            SuspendKind::Resume { k, arg } => vec![*k, *arg],
            SuspendKind::ThreadJoin { handle } => vec![*handle],
            SuspendKind::MemoryWait {
                addr,
                expected,
                timeout,
                ..
            } => vec![*addr, *expected, *timeout],
            SuspendKind::Yield { value } => vec![*value],
            SuspendKind::SvcServe { handle, args, .. } => {
                let mut v = Vec::with_capacity(args.len() + 1);
                v.push(*handle);
                v.extend_from_slice(args);
                v
            }
        }
    }

    /// Whether thaw flips the shadow thaw-state word back to `NORMAL` in **this** op's arm — true for
    /// the globally-deepest frozen frame (a leaf `call.cap`, a loop-header poll, or a re-parked
    /// `suspend` / `thread.join` / `atomic.wait` / serve op, whose blocking peer is a separate vCPU).
    /// A propagated `call` / `call.dyn` / `cont.resume` does **not** flip — the callee it
    /// re-issues owns the flip at its own deepest leaf (#915).
    fn flips_thaw_word(&self) -> bool {
        matches!(
            self,
            SuspendKind::Leaf { .. }
                | SuspendKind::LoopHeader
                | SuspendKind::Yield { .. }
                | SuspendKind::ThreadJoin { .. }
                | SuspendKind::MemoryWait { .. }
                | SuspendKind::SvcServe { .. }
        )
    }
}

/// One resume point's metadata across the whole function (global id by vector order).
struct PointPlan {
    kind: SuspendKind,        // leaf call.cap or propagated call
    nres: usize,              // result count of the suspend op
    out: usize,               // value count after the op (= continuation param count)
    save_end: usize,          // spillable range `[0, save_end)` (excludes a call's results)
    slot_types: Vec<ValType>, // types of values `0..out` (continuation params / dead-slot zeros)
    spilled: Vec<usize>,      // block-local indices actually spilled (sorted, live ∪ call-args)
    frame_offsets: Vec<u64>,  // window offset of each spilled value (parallel to `spilled`)
    frame_size: u64,
    rid_off: u64,
    flag_off: Option<u64>, // the spilled re-issue word (`Leaf` and `ThreadJoin`)
    cont_seg: u32,         // new block index of the continuation segment (after the op)
}

/// Per-original-block analysis: value types, the value count after each instruction, and
/// the positions of its may-suspend ops.
struct BlockInfo {
    types: Vec<ValType>,
    vend: Vec<usize>,
    scs: Vec<usize>,
    plen: usize,
}

/// Instrument one may-suspend function. `Ok((func, max_frame_size))` on success, or an
/// error for an out-of-scope shape. (Non-may-suspend functions are not passed here.)
///
/// Each original block is split at its may-suspend ops into forward segments; a poll after
/// each op unwinds (per-point spill + resume id) or continues to the next segment, and the
/// prologue's `br_table` dispatch routes a thaw to the in-flight point's arm, which reloads
/// and resumes into the continuation segment. Branch targets are remapped to segment 0 of
/// the target block. See the block-layout map near the constants.
fn transform_func(
    f: &Func,
    func_results: &[Vec<ValType>],
    may_suspend: &[bool],
    tainted_sigs: &[temen_ir::FuncType],
    type_section: &[TypeEntry],
    opts: &TransformOpts,
) -> Result<(Func, u64), TransformError> {
    // Whether a `call.dyn` of this signature could reach a may-suspend target (R8) — the same
    // by-signature rule `compute_may_suspend` used to mark this function may-suspend in the first
    // place. Kept identical to that rule so the instrumentation set == the taint set (re-issue
    // soundness: every possible indirect target is instrumented, so the reloaded `idx` can only
    // resolve to a `REWINDING`-aware callee). `ty` is an interned type index (#922).
    let is_tainted = |ty: u32| tainted_sigs.iter().any(|s| s == sig_of(type_section, ty));
    // Out of scope: a **tail** call (direct or indirect) into a may-suspend callee — the frame is
    // replaced, so there is no poll to unwind at. Rejected (fail closed), so a tail-dispatched
    // suspending callee never leaves a caller silently under-instrumented.
    for blk in &f.blocks {
        let reject = match &blk.term {
            Terminator::ReturnCall { func, .. } => may_suspend[*func as usize],
            Terminator::ReturnCallIndirect { ty, .. } => is_tainted(*ty),
            _ => false,
        };
        if reject {
            return Err(TransformError::UnsupportedShape);
        }
    }

    let nb = f.blocks.len();
    // Per-block analysis (value types / counts / suspend positions).
    let mut binfo: Vec<BlockInfo> = Vec::with_capacity(nb);
    for blk in &f.blocks {
        let mut types = blk.params.clone();
        let mut vend = Vec::with_capacity(blk.insts.len());
        for inst in &blk.insts {
            types.extend(result_types(
                inst,
                &types,
                func_results,
                type_section,
                opts,
            )?);
            vend.push(types.len());
        }
        let scs: Vec<usize> = blk
            .insts
            .iter()
            .enumerate()
            .filter(|(_, inst)| match inst {
                Inst::Call { func, .. } => may_suspend[*func as usize],
                Inst::CallIndirect { ty, .. } => is_tainted(*ty),
                x => opts.is_site(x),
            })
            .map(|(pos, _)| pos)
            .collect();
        binfo.push(BlockInfo {
            types,
            vend,
            scs,
            plen: blk.params.len(),
        });
    }

    let inblock_points: usize = binfo.iter().map(|bi| bi.scs.len()).sum();
    if inblock_points == 0 {
        return Err(TransformError::UnsupportedShape); // may-suspend, but no in-block op
    }

    // A block is a *loop header* if a back-edge (a branch whose target index ≤ its source block)
    // targets it. A poll prepended to the header's entry — which dominates the loop body — is hit
    // every iteration, so a poll-free compute loop freezes at bounded latency (Phase-4 Slice A,
    // the R6 caveat). Each header adds one resume point (a `LoopHeader` poll) and one segment (the
    // poll itself, ahead of the header's body segments).
    let mut is_header = vec![false; nb];
    if opts.loop_polls {
        for (b, blk) in f.blocks.iter().enumerate() {
            for t in term_targets(&blk.term) {
                if (t as usize) <= b {
                    is_header[t as usize] = true;
                }
            }
        }
    }
    let header_count = is_header.iter().filter(|&&h| h).count();
    let total_points = inblock_points + header_count;

    // Block-index layout (see the map near the constants).
    let mut seg_base = Vec::with_capacity(nb);
    let mut acc = 1u32; // segment indices start right after the PROLOGUE
    for (b, bi) in binfo.iter().enumerate() {
        seg_base.push(acc);
        // points + 1 segments, + 1 poll segment ahead of the body when this block is a header.
        acc += bi.scs.len() as u32 + 1 + is_header[b] as u32;
    }
    let s_total = acc - 1;
    // Body segment `k` of block `b`. A header's poll segment sits at `seg_base[b]` (= `seg0(b)`,
    // the branch-entry target), so body segments start one past it.
    let seg = |b: usize, k: usize| seg_base[b] + is_header[b] as u32 + k as u32;
    let p_total = total_points as u32;
    let dispatch_blk = 1 + s_total;
    // Each resume point has a UNWIND *pair*: a check block (traps if the push would exceed
    // the reserve) and a spill block. `check_blk(g) = unwind_base + 2g`, spill is +1.
    let unwind_base = 2 + s_total;
    let arm_base = unwind_base + 2 * p_total;
    let trap_blk = arm_base + p_total;
    let p = f.params.len();

    // Remap a terminator's block targets to segment 0 of each target block.
    let seg0 = |t: BlockIdx| seg_base[t as usize];
    let remap = |term: &Terminator| -> Terminator {
        match term {
            Terminator::Br { target, args } => Terminator::Br {
                target: seg0(*target),
                args: args.clone(),
            },
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => Terminator::BrIf {
                cond: *cond,
                then_blk: seg0(*then_blk),
                then_args: then_args.clone(),
                else_blk: seg0(*else_blk),
                else_args: else_args.clone(),
            },
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => Terminator::BrTable {
                idx: *idx,
                targets: targets.iter().map(|(t, a)| (seg0(*t), a.clone())).collect(),
                default: (seg0(default.0), default.1.clone()),
            },
            // Return / Unreachable / (in)direct tail calls carry no block targets.
            other => other.clone(),
        }
    };

    // ---- PROLOGUE — dispatch on the state word ----
    let mut pb = Bb::new(f.params.clone());
    let (st_a, st_off) = pb.thaw_word_addr();
    let st = pb.one(load(LoadOp::I32, st_a, st_off));
    let rw = pb.one(Inst::ConstI32(STATE_REWINDING));
    let is_rw = pb.one(icmp(IntTy::I32, CmpOp::Eq, st, rw));
    let prologue = pb.finish(Terminator::BrIf {
        cond: is_rw,
        then_blk: dispatch_blk,
        then_args: vec![], // the arm reloads everything from the frame
        else_blk: seg(0, 0),
        else_args: (0..p as u32).collect(),
    });

    // ---- forward segments + collect the per-point resume plans (global order) ----
    let mut seg_blocks: Vec<Block> = Vec::with_capacity(s_total as usize);
    let mut points: Vec<PointPlan> = Vec::with_capacity(total_points);
    for (b, blk) in f.blocks.iter().enumerate() {
        let bi = &binfo[b];
        let m = bi.scs.len();
        // segment k's incoming value count: block params for k==0, else the previous op's out
        let in_of = |k: usize| {
            if k == 0 {
                bi.plen
            } else {
                bi.vend[bi.scs[k - 1]]
            }
        };
        // Loop-header poll: a state-word check at the header's entry (`seg_base[b]` = `seg0(b)`,
        // the branch-entry target), ahead of the body segments. On UNWINDING it spills the
        // header's block params (the loop-carried live set) into a fresh `LoopHeader` resume
        // point and returns; otherwise it falls through into the body. Built before the in-block
        // points so its `gid` precedes them (the index layout only requires `points` order to
        // match the unwind/arm block order, which it does).
        if is_header[b] {
            let plen = bi.plen;
            let slot_types = bi.types[0..plen].to_vec();
            let spilled: Vec<usize> = (0..plen).collect(); // all params (loop-carried, live)
            let mut frame_offsets = Vec::with_capacity(plen);
            let mut off = 0u64;
            for &i in &spilled {
                off = align_up(off, vsize(slot_types[i]));
                frame_offsets.push(off);
                off += vsize(slot_types[i]);
            }
            let frame_size = align_up(off + 4, 16);
            let gid = points.len() as u32;
            points.push(PointPlan {
                kind: SuspendKind::LoopHeader,
                nres: 0,
                out: plen,
                save_end: plen,
                slot_types: slot_types.clone(),
                spilled,
                frame_offsets,
                rid_off: frame_size - 4,
                frame_size,
                flag_off: None,
                cont_seg: seg(b, 0), // re-enter the header body, past the poll
            });
            let mut psb = Bb::new(slot_types);
            let live: Vec<ValIdx> = (0..plen as u32).collect();
            // On to the header body (the point's continuation) unless a freeze is unwinding.
            let term = poll(&mut psb, unwind_base + 2 * gid, seg(b, 0), live);
            seg_blocks.push(psb.finish(term));
        }
        for k in 0..=m {
            let mut sb = Bb::new(bi.types[0..in_of(k)].to_vec());
            if k < m {
                // segment body up to & including the suspend op, then the poll
                let pos = bi.scs[k];
                let seg_start = if k == 0 { 0 } else { bi.scs[k - 1] + 1 };
                sb.insts.extend_from_slice(&blk.insts[seg_start..=pos]);
                let out = bi.vend[pos];
                sb.next = out as u32;
                let gid = points.len() as u32;
                let live: Vec<ValIdx> = (0..out as u32).collect();
                let term = poll(&mut sb, unwind_base + 2 * gid, seg(b, k + 1), live);
                seg_blocks.push(sb.finish(term));

                // resume plan for this point
                let kind = match &blk.insts[pos] {
                    // §13.4 slice 4b: a serve op re-issues (the sentinel it returned under
                    // `UNWINDING` must never reload as the served count) — before the
                    // generic `Leaf` arm.
                    Inst::CapCall {
                        type_id: temen_ir::CAP_SELF_TYPE_ID,
                        op: sop @ (SVC_POLL_OP | SVC_WAIT_OP),
                        sig,
                        handle,
                        args,
                    } => SuspendKind::SvcServe {
                        type_id: temen_ir::CAP_SELF_TYPE_ID,
                        op: *sop,
                        sig: *sig,
                        handle: *handle,
                        args: args.clone(),
                    },
                    // A runtime-bound capability call is a leaf exactly like `call.cap`: the host
                    // effect happened before the freeze, so the thaw reloads its result. (A
                    // durable domain never resolves an import to a serve op — `Host::import_binding`
                    // fails those closed — so the `SvcServe` re-issue case cannot hide behind one.)
                    op @ (Inst::CapCall { .. }
                    | Inst::CallImport { .. }
                    | Inst::CallImportDyn { .. }
                    | Inst::CallSym { .. }) => SuspendKind::Leaf { op: op.clone() },
                    Inst::Call { func, args } => SuspendKind::Propagated {
                        callee: *func,
                        args: args.clone(),
                    },
                    Inst::CallIndirect { ty, idx, args } => SuspendKind::PropagatedIndirect {
                        ty: *ty,
                        idx: *idx,
                        args: args.clone(),
                    },
                    // `block` is advisory scheduling only — not preserved across the durable
                    // suspend record; replay re-issues a plain (non-blocking) resume.
                    Inst::ContResume { k, arg, .. } => SuspendKind::Resume { k: *k, arg: *arg },
                    Inst::Suspend { value } => SuspendKind::Yield { value: *value },
                    Inst::ThreadJoin { handle } => SuspendKind::ThreadJoin { handle: *handle },
                    Inst::MemoryWait {
                        ty,
                        addr,
                        expected,
                        timeout,
                    } => SuspendKind::MemoryWait {
                        ty: *ty,
                        addr: *addr,
                        expected: *expected,
                        timeout: *timeout,
                    },
                    _ => unreachable!(
                        "suspend position is a call.cap / call.import / call / fiber / thread.join / atomic.wait op"
                    ),
                };
                let nres = match (&kind, &blk.insts[pos]) {
                    (
                        SuspendKind::Leaf { .. },
                        Inst::CapCall { sig, .. }
                        | Inst::CallImport { sig, .. }
                        | Inst::CallImportDyn { sig, .. }
                        | Inst::CallSym { sig, .. },
                    ) => sig_of(type_section, *sig).results.len(),
                    (SuspendKind::SvcServe { .. }, Inst::CapCall { sig, .. }) => {
                        sig_of(type_section, *sig).results.len()
                    }
                    (SuspendKind::Propagated { callee, .. }, _) => {
                        func_results[*callee as usize].len()
                    }
                    (SuspendKind::PropagatedIndirect { ty, .. }, _) => {
                        sig_of(type_section, *ty).results.len()
                    }
                    (SuspendKind::Resume { .. }, _) => 2, // (status, value)
                    (SuspendKind::Yield { .. }, _) => 1,  // the resume arg
                    (SuspendKind::ThreadJoin { .. }, _) => 1, // the join result (i64)
                    (SuspendKind::MemoryWait { .. }, _) => 1, // the wait status (i32)
                    _ => unreachable!(),
                };
                // Spillable range: values `[0, save_end)`. A leaf reloads its own result too
                // (`save_end == out`); a propagated frame re-issues its call, so the call's
                // results `[save_end, out)` are recomputed, not spilled.
                let save_end = match kind {
                    // A wait's status is spilled too (#1769): one the wait completed with before the
                    // cut is delivered on thaw; only the freeze's own `WAIT_FROZEN` re-issues it. A
                    // join's result likewise, unless the freeze ended it (#1685).
                    SuspendKind::Leaf { .. }
                    | SuspendKind::MemoryWait { .. }
                    | SuspendKind::ThreadJoin { .. } => out,
                    // The op's results are recomputed (re-issue) or redelivered (resume), so
                    // they aren't spilled — same as a propagated call.
                    SuspendKind::Propagated { .. }
                    | SuspendKind::PropagatedIndirect { .. }
                    | SuspendKind::Resume { .. }
                    | SuspendKind::Yield { .. }
                    | SuspendKind::SvcServe { .. } => out - nres,
                    // Header polls are built separately (above), never from an in-block op.
                    SuspendKind::LoopHeader => unreachable!("loop-header point not from an op"),
                };
                let slot_types = bi.types[0..out].to_vec();

                // Minimal live-set: spill only values used *after* the op (block-local SSA ⇒
                // a value's whole live range is in this block, so "live across" = referenced
                // by a later instruction or the terminator), plus a propagated call's own
                // operands (needed to re-issue it). An unrecognized instruction in the tail ⇒
                // fall back to spilling the whole spillable range (never under-spill).
                let mut used = vec![false; out];
                let mut conservative = false;
                for inst in &blk.insts[pos + 1..] {
                    match inst_operands(inst) {
                        Some(ops) => ops.iter().for_each(|&o| {
                            if (o as usize) < out {
                                used[o as usize] = true;
                            }
                        }),
                        None => {
                            conservative = true;
                            break;
                        }
                    }
                }
                for o in term_operands(&blk.term) {
                    if (o as usize) < out {
                        used[o as usize] = true;
                    }
                }
                // The re-issued op's operands must be reloaded across the freeze — mark them live
                // from the single operand table (#915).
                for o in kind.operands() {
                    used[o as usize] = true;
                }
                // A leaf's own results are always in its frame, even ones the continuation never
                // reads: they are the reply slot a host injects into before a thaw (#1768).
                if matches!(kind, SuspendKind::Leaf { .. }) {
                    used[out - nres..out].iter_mut().for_each(|u| *u = true);
                }
                // A wait's thaw arm reads its own status to decide re-issue vs. deliver (#1769).
                if matches!(kind, SuspendKind::MemoryWait { .. }) {
                    used[out - 1] = true;
                }
                let spilled: Vec<usize> = if conservative {
                    (0..save_end).collect()
                } else {
                    (0..save_end).filter(|&i| used[i]).collect()
                };
                // Frame layout (DURABILITY.md §12.7): packed spilled values, resume id on top. A
                // leaf's results come first, at the frame base, in declaration order — the reply
                // slot (`ShadowArena::leaf_reply`); the rest of its live set follows. `spilled`
                // itself stays in value order (the reload order), so each value keeps its own offset.
                let is_reply =
                    |i: usize| matches!(kind, SuspendKind::Leaf { .. }) && i >= out - nres;
                let mut frame_offsets = vec![0u64; spilled.len()];
                let mut off = 0u64;
                let replies_first = (0..spilled.len())
                    .filter(|&j| is_reply(spilled[j]))
                    .chain((0..spilled.len()).filter(|&j| !is_reply(spilled[j])));
                for j in replies_first.collect::<Vec<_>>() {
                    let i = spilled[j];
                    off = align_up(off, vsize(slot_types[i]));
                    frame_offsets[j] = off;
                    off += vsize(slot_types[i]);
                }
                // A host call's frame also carries the re-issue word it saw (#1672), below the id; so
                // does a join's (#1685).
                let flag_off = matches!(
                    kind,
                    SuspendKind::Leaf { .. } | SuspendKind::ThreadJoin { .. }
                )
                .then(|| {
                    off = align_up(off, 4);
                    off += 4;
                    off - 4
                });
                let frame_size = align_up(off + 4, 16);
                points.push(PointPlan {
                    flag_off,
                    kind,
                    nres,
                    out,
                    save_end,
                    slot_types,
                    spilled,
                    frame_offsets,
                    rid_off: frame_size - 4,
                    frame_size,
                    cont_seg: seg(b, k + 1),
                });
            } else {
                // last segment: the tail after the final suspend op + the remapped terminator
                let seg_start = if m == 0 { 0 } else { bi.scs[m - 1] + 1 };
                sb.insts.extend_from_slice(&blk.insts[seg_start..]);
                seg_blocks.push(sb.finish(remap(&blk.term)));
            }
        }
    }

    // ---- DISPATCH — read the resume id at SP-4 and br_table to the matching arm ----
    // `sp_a` is the **active context's shadow-SP word address** from the runtime-private register
    // (`durable.shadow_base`, §12.8 4A.5) — per-context, so concurrent vCPUs each address their own SP
    // word with no shared location (vs. the former fixed global `SHADOW_SP_OFF`). The runtime seeds the
    // register; a guest cannot redirect it. Used identically at every SP site (dispatch/unwind/arm).
    let mut db = Bb::new(vec![]);
    let sp_a = db.one(Inst::DurableShadowBase);
    let sp = db.one(load(LoadOp::I64, sp_a, 0));
    let four = db.one(Inst::ConstI64(4));
    let sp_m4 = db.one(ibin(IntTy::I64, BinOp::Sub, sp, four));
    let rid = db.one(load(LoadOp::I32, sp_m4, 0));
    // id 0 is reserved ("no resume" ⇒ trap); id g+1 selects ARM_g.
    let mut targets: Vec<(BlockIdx, Vec<ValIdx>)> = vec![(trap_blk, vec![])];
    for g in 0..p_total {
        targets.push((arm_base + g, vec![]));
    }
    let dispatch = db.finish(Terminator::BrTable {
        idx: rid,
        targets,
        default: (trap_blk, vec![]),
    });

    // ---- UNWIND (check + spill pair) / ARM_g, per resume point ----
    let mut unwind_blocks: Vec<Block> = Vec::with_capacity(2 * total_points);
    let mut arm_blocks: Vec<Block> = Vec::with_capacity(total_points);
    // Blocks an arm needs beyond its own, appended after the trap block so no index above moves:
    // a wait's re-issue (#1769).
    let mut extra_blocks: Vec<Block> = Vec::new();
    for (gid, pt) in points.iter().enumerate() {
        // index in `pt.spilled` (and thus the reloaded vec) of a block-local value, if spilled
        let spill_slot = |i: usize| pt.spilled.binary_search(&i).ok();
        // The point's UNWIND check block — where its polls send an unwinding frame.
        let unwind_blk = unwind_base + 2 * gid as u32;

        // UNWIND check: a push of this frame must not run past the running context's own region
        // `[region base, +SHADOW_STRIDE)` — past it lies the next context's shadow frames, or
        // guest memory for the last region (R9 / DURABILITY.md §12.7, #1683). The shadow stack
        // mirrors the call stack, so this only trips for a chain deeper than a region holds — a
        // clean trap, never silent corruption. It lives on the (cold) freeze path, not the
        // per-call path.
        let mut cb = Bb::new(pt.slot_types.clone());
        let sp_a = cb.one(Inst::DurableShadowBase);
        let sp = cb.one(load(LoadOp::I64, sp_a, 0));
        let fsz = cb.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = cb.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        let stride = cb.one(Inst::ConstI64(SHADOW_STRIDE as i64));
        let region_end = cb.one(ibin(IntTy::I64, BinOp::Add, sp_a, stride));
        let over = cb.one(icmp(IntTy::I64, CmpOp::GtU, newsp, region_end));
        let live: Vec<ValIdx> = (0..pt.out as u32).collect();
        unwind_blocks.push(cb.finish(Terminator::BrIf {
            cond: over,
            then_blk: trap_blk,
            then_args: vec![],
            else_blk: unwind_blk + 1, // the spill block
            else_args: live,
        }));

        // UNWIND spill: spill the live (∪ call-arg) values + the resume id, commit the new SP.
        let mut ub = Bb::new(pt.slot_types.clone());
        let sp_a = ub.one(Inst::DurableShadowBase);
        let sp = ub.one(load(LoadOp::I64, sp_a, 0)); // this activation's frame base
        for (j, &i) in pt.spilled.iter().enumerate() {
            ub.zero(spill(pt.slot_types[i], sp, i as u32, pt.frame_offsets[j]));
        }
        // A host call (or a join) moves the context's re-issue word into its frame and clears it, so the word
        // is set only between the abandoning runtime and this spill.
        if let Some(flag_off) = pt.flag_off {
            let flag = ub.one(load(LoadOp::I32, sp_a, REISSUE_IN_REGION_OFF));
            ub.zero(store(StoreOp::I32, sp, flag, flag_off));
            let zero = ub.one(Inst::ConstI32(0));
            ub.zero(store(StoreOp::I32, sp_a, zero, REISSUE_IN_REGION_OFF));
        }
        let rid = ub.one(Inst::ConstI32(gid as i32 + 1));
        ub.zero(store(StoreOp::I32, sp, rid, pt.rid_off));
        let fsz = ub.one(Inst::ConstI64(pt.frame_size as i64));
        let newsp = ub.one(ibin(IntTy::I64, BinOp::Add, sp, fsz));
        ub.zero(store(StoreOp::I64, sp_a, newsp, 0));
        let ret: Vec<ValIdx> = f.results.iter().map(|&t| ub.one(zero_const(t))).collect();
        unwind_blocks.push(ub.finish(Terminator::Return(ret)));

        // ARM: reload the spilled set (self-contained — no incoming params), pop, resume.
        let mut ab = Bb::new(vec![]);
        let sp_a = ab.one(Inst::DurableShadowBase);
        let sp = ab.one(load(LoadOp::I64, sp_a, 0));
        let fsz = ab.one(Inst::ConstI64(pt.frame_size as i64));
        let base = ab.one(ibin(IntTy::I64, BinOp::Sub, sp, fsz));
        let reloaded: Vec<ValIdx> = pt
            .spilled
            .iter()
            .enumerate()
            .map(|(j, &i)| ab.one(reload(pt.slot_types[i], base, pt.frame_offsets[j])))
            .collect();
        let flag = pt.flag_off.map(|o| ab.one(load(LoadOp::I32, base, o)));
        ab.zero(store(StoreOp::I64, sp_a, base, 0)); // pop: SP = frame base

        // Flip the shadow thaw-state word back to `NORMAL` for the globally-deepest frozen frame
        // (leaf / loop-header / re-parked suspend·join·wait·serve); a propagated call/indirect/resume
        // leaves it — the callee it re-issues owns the flip at its own leaf. One copy, driven by
        // `flips_thaw_word()` (#915), emitted before the re-issue exactly as each arm did.
        if pt.kind.flips_thaw_word() {
            let (st_a, st_off) = ab.thaw_word_addr();
            let normal_v = ab.one(Inst::ConstI32(STATE_NORMAL));
            ab.zero(store(StoreOp::I32, st_a, normal_v, st_off));
        }
        // For a propagated call, re-issue it (its operands were all spilled). For a leaf, the state
        // word was flipped above.
        let op_results: Vec<ValIdx> = match &pt.kind {
            // A leaf call.cap and a loop-header poll are both the globally-deepest frozen frame with
            // no op to re-issue: the leaf then reloads its call.cap result; the header reloads its
            // block params and re-enters the body (`cont_seg`). Neither produces an `op_results` value.
            SuspendKind::Leaf { .. } | SuspendKind::LoopHeader => vec![],
            SuspendKind::Propagated { callee, args } => {
                let mapped: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| reloaded[spill_slot(a as usize).expect("call arg spilled")])
                    .collect();
                ab.many(
                    Inst::Call {
                        func: *callee,
                        args: mapped,
                    },
                    pt.nres,
                )
            }
            // `call.dyn` re-issue (R8): reload the (spilled) table index + args and re-execute
            // the indirect call. The reloaded `idx` re-selects the same slot, whose (instrumented)
            // callee rewinds in turn — the indirect twin of `Propagated`; no state-word flip (the
            // resolved callee is the frame that flips at its own deepest leaf).
            SuspendKind::PropagatedIndirect { ty, idx, args } => {
                let ridx = reloaded[spill_slot(*idx as usize).expect("call.dyn idx spilled")];
                let mapped: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| reloaded[spill_slot(a as usize).expect("call.dyn arg spilled")])
                    .collect();
                ab.many(
                    Inst::CallIndirect {
                        ty: *ty,
                        idx: ridx,
                        args: mapped,
                    },
                    pt.nres,
                )
            }
            // `cont.resume` re-issue (slice 3.1.2): reload the (spilled) handle + arg and resume
            // the fiber again. On thaw the fiber rewinds in turn (its `Yield` re-park) and
            // redelivers `(status, value)` — the resumer threads those two results into its
            // continuation just like a propagated call. The resumer does **not** flip the state
            // word: the resumee's `Yield` arm (the globally-deepest frame) does.
            SuspendKind::Resume { k, arg } => {
                let kk = reloaded[spill_slot(*k as usize).expect("resume handle spilled")];
                let aa = reloaded[spill_slot(*arg as usize).expect("resume arg spilled")];
                ab.many(
                    Inst::ContResume {
                        k: kk,
                        arg: aa,
                        block: false,
                    },
                    pt.nres,
                )
            }
            // `suspend` re-park (slice 3.1.3): a parked fiber's suspend is the globally-deepest
            // frozen frame, so flip the state word to NORMAL, then re-execute `suspend` — which
            // parks this fiber and hands `value` back to the resumer (in NORMAL). Its result, the
            // value the *next* resume delivers, threads into the continuation exactly as a leaf's
            // reloaded call.cap result does.
            SuspendKind::Yield { value } => {
                let v = reloaded[spill_slot(*value as usize).expect("suspend value spilled")];
                ab.many(Inst::Suspend { value: v }, pt.nres)
            }
            // `thread.join`: the joined child rewinds as a *separate* vCPU, so on this thread the join is
            // the globally-deepest frozen frame — it flipped the state to `NORMAL` above, like a leaf.
            // Its terminator below reloads the result or re-issues the join (#1685).
            SuspendKind::ThreadJoin { .. } => vec![],
            // `atomic.wait`: like `thread.join`, the wait is the globally-deepest frozen frame on this
            // thread (the notifier is a *separate* vCPU), so the state word was flipped to `NORMAL`
            // above. Its status was spilled and is reloaded into the continuation; the terminator
            // below branches on it (#1769). A wait that completed before the cut (woken, not-equal,
            // timed out) delivers that status as it was. One the freeze ended (`WAIT_FROZEN`) is
            // re-issued with its reloaded operands, and re-checks the restored value.
            SuspendKind::MemoryWait { .. } => vec![],
            // Serve-op re-issue (§13.4 slice 4b): the mid-handler gate guarantees this point is
            // the globally-deepest frozen frame on its thread (no handler was in flight), so —
            // like `atomic.wait` — flip the state word to `NORMAL` itself, then reload the
            // handle + args and re-execute. The re-executed drain runs against the restored
            // queue (the snapshot's serve section); an empty queue re-parks `svc.wait` exactly
            // as an uninterrupted run would.
            SuspendKind::SvcServe {
                type_id,
                op,
                sig,
                handle,
                args,
            } => {
                let hh = reloaded[spill_slot(*handle as usize).expect("serve-op handle spilled")];
                let aa: Vec<ValIdx> = args
                    .iter()
                    .map(|&a| reloaded[spill_slot(a as usize).expect("serve-op arg spilled")])
                    .collect();
                ab.many(
                    Inst::CapCall {
                        type_id: *type_id,
                        op: *op,
                        sig: *sig,
                        handle: hh,
                        args: aa,
                    },
                    pt.nres,
                )
            }
        };

        // Assemble the continuation's `out` args slot-by-slot: a reloaded value, a re-issued
        // call result (`[save_end, out)`), or a zero placeholder for a dead-but-present slot.
        let cont_args: Vec<ValIdx> = (0..pt.out)
            .map(|i| {
                if let Some(j) = spill_slot(i) {
                    reloaded[j]
                } else if i >= pt.save_end {
                    op_results[i - pt.save_end]
                } else {
                    ab.one(zero_const(pt.slot_types[i])) // dead across the op; value unused
                }
            })
            .collect();
        let term = match &pt.kind {
            // A wait that completed before the cut (woken, not-equal, timed out) delivers its status
            // as it was; one the freeze ended (`WAIT_FROZEN`) is re-issued and re-checks the restored
            // value (#1769).
            SuspendKind::MemoryWait {
                ty,
                addr,
                expected,
                timeout,
            } => {
                let status = cont_args[pt.out - 1];
                let frozen = ab.one(Inst::ConstI32(WAIT_FROZEN));
                let cond = ab.one(icmp(IntTy::I32, CmpOp::Eq, status, frozen));
                let ty = *ty;
                reissue_branch(
                    &mut extra_blocks,
                    trap_blk + 1,
                    unwind_blk,
                    pt,
                    cont_args,
                    cond,
                    &[*addr, *expected, *timeout],
                    &reloaded,
                    |p| Inst::MemoryWait {
                        ty,
                        addr: p[0],
                        expected: p[1],
                        timeout: p[2],
                    },
                )
            }
            // A host call the runtime abandoned (its spilled re-issue word is set) is re-issued; one
            // it performed reloads its result (#1672).
            SuspendKind::Leaf { op } => {
                let flag = flag.expect("a host call's frame carries its re-issue word");
                let zero = ab.one(Inst::ConstI32(0));
                let cond = ab.one(icmp(IntTy::I32, CmpOp::Ne, flag, zero));
                let operands = inst_operands(op).expect("a host call's operands are modeled");
                reissue_branch(
                    &mut extra_blocks,
                    trap_blk + 1,
                    unwind_blk,
                    pt,
                    cont_args,
                    cond,
                    &operands,
                    &reloaded,
                    |p| with_operands(op, p),
                )
            }
            // A join the freeze ended (its spilled re-issue word is set) is re-issued against the
            // re-spawned child; one that got its child's real result reloads it (#1685).
            SuspendKind::ThreadJoin { handle } => {
                let flag = flag.expect("a join's frame carries its re-issue word");
                let zero = ab.one(Inst::ConstI32(0));
                let cond = ab.one(icmp(IntTy::I32, CmpOp::Ne, flag, zero));
                reissue_branch(
                    &mut extra_blocks,
                    trap_blk + 1,
                    unwind_blk,
                    pt,
                    cont_args,
                    cond,
                    &[*handle],
                    &reloaded,
                    |p| Inst::ThreadJoin { handle: p[0] },
                )
            }
            // A loop header re-enters its body; nothing ran again.
            SuspendKind::LoopHeader => Terminator::Br {
                target: pt.cont_seg,
                args: cont_args,
            },
            // Every other kind ran its op again above — a live call, as on the forward path, so it
            // polls as the forward path does: an unwind that starts beneath it later (the next
            // freeze, the next fork) must find this frame unwinding too, not running on with the
            // placeholder its callee returned.
            _ => poll(&mut ab, unwind_blk, pt.cont_seg, cont_args),
        };
        arm_blocks.push(ab.finish(term));
    }

    // ---- TRAP — br_table default / forged resume id ----
    let trap = Block {
        params: vec![],
        insts: vec![],
        term: Terminator::Unreachable,
    };

    // Assemble in the order the index layout assumes.
    let mut blocks = Vec::with_capacity((2 + s_total + 2 * p_total + 1) as usize);
    blocks.push(prologue);
    blocks.extend(seg_blocks);
    blocks.push(dispatch);
    blocks.extend(unwind_blocks);
    blocks.extend(arm_blocks);
    blocks.push(trap);
    blocks.extend(extra_blocks);

    let max_frame = points.iter().map(|pt| pt.frame_size).max().unwrap_or(0);
    let func = Func {
        params: f.params.clone(),
        results: f.results.clone(),
        blocks,
    };
    Ok((func, max_frame))
}

// ---- window helpers for freeze/thaw drivers and tests ----

/// A fresh durable window of `size` bytes: state = `NORMAL`, and the root context's per-context
/// shadow-SP word (§12.8 4A.5) — the first 8 bytes of its region at `arena.region_base(0)` — set to
/// its empty frame base (`arena.frame_base(0)`, just past the SP + thaw words). The legacy global
/// `SHADOW_SP_OFF` is unused; the per-context thaw words default to `NORMAL` (zero).
pub fn init_durable_window(size: usize, a: ShadowArena) -> Vec<u8> {
    let mut w = vec![0u8; size];
    write_state(&mut w, STATE_NORMAL);
    let b = a.region_base(0) as usize;
    w[b..b + 8].copy_from_slice(&a.frame_base(0).to_le_bytes());
    w
}

/// Window byte offset of context `ctx`'s **thaw** state word (§12.8 concurrent-thaw stage 1) — its
/// region base plus [`STATE_IN_REGION_OFF`]. Per-context, so a thaw can set each frozen vCPU rewinding
/// independently (vs. the global [`STATE_OFF`] freeze word).
pub fn thaw_state_off(arena: ShadowArena, ctx: usize) -> u64 {
    arena.thaw_state_off(ctx)
}

/// Overwrite the global **freeze** state word (`UNWINDING`/`ARMED`/`NORMAL`) in a window image — the
/// stop-the-world trigger every poll reads. Thaw (`REWINDING`) goes through [`write_thaw_state`].
pub fn write_state(window: &mut [u8], state: i32) {
    window[STATE_OFF as usize..STATE_OFF as usize + 4].copy_from_slice(&state.to_le_bytes());
}

/// Overwrite context `ctx`'s per-context **thaw** state word (`REWINDING`/`NORMAL`) — used to drive a
/// thaw (the runtime sets each frozen context `REWINDING` before its rewinding re-entry).
pub fn write_thaw_state(window: &mut [u8], arena: ShadowArena, ctx: usize, state: i32) {
    let off = thaw_state_off(arena, ctx) as usize;
    window[off..off + 4].copy_from_slice(&state.to_le_bytes());
}

/// Set up a window for a **thaw** of context `ctx` (§12.8 concurrent-thaw stage 1): clear the global
/// **freeze** word back to `NORMAL` (the frozen artifact left it `UNWINDING`, but a thaw is not a
/// freeze — leaving it would make the rewinding code's polls re-unwind) and set `ctx`'s per-context
/// **thaw** word to `REWINDING`. Mirrors what the runtime does on a real snapshot-restore thaw (the
/// interp's `drive` clear + per-context `REWINDING`; the JIT thaw driver).
pub fn begin_thaw(window: &mut [u8], arena: ShadowArena, ctx: usize) {
    write_state(window, STATE_NORMAL);
    write_thaw_state(window, arena, ctx, STATE_REWINDING);
}

/// **Inject** `reply` as the result the frozen call of context `ctx` returns on thaw, in place of
/// the one it returned before the freeze (FORK.md §3 — reply-injection, never re-issue; #1768). The
/// deepest frame of a context frozen at a capability call is that call's leaf frame, which holds the
/// call's results first ([`ShadowArena::leaf_reply`]), so this writes the first result's slot as an
/// `i64`. A fork writes a different reply into each copy of one frozen window, and each thaw resumes
/// past the same call with its own answer — return-twice.
pub fn inject_leaf_reply(window: &mut [u8], arena: ShadowArena, ctx: usize, reply: i64) {
    let off = arena.leaf_reply(ctx) as usize;
    window[off..off + 8].copy_from_slice(&reply.to_le_bytes());
}

/// Read context `ctx`'s per-context **thaw** state word — after a thaw, a completed rewind reads
/// `NORMAL` (the deepest frame's re-issue flipped it).
pub fn read_thaw_state(window: &[u8], arena: ShadowArena, ctx: usize) -> i32 {
    let off = thaw_state_off(arena, ctx) as usize;
    let mut b = [0u8; 4];
    b.copy_from_slice(&window[off..off + 4]);
    i32::from_le_bytes(b)
}

/// Arm a window to **freeze after `safepoints` further fiber safepoints** (the deterministic mid-run
/// trigger): the run proceeds normally, and the runtime promotes the state word to `UNWINDING` at the
/// `safepoints`-th fiber safepoint (`cont.resume`/`suspend`) so that op's trailing poll begins the
/// freeze. `safepoints == 1` freezes at the first fiber safepoint; larger values let the run make
/// forward progress first (e.g. past a fiber recycle). A non-positive count is clamped to 1.
pub fn arm_freeze_after(window: &mut [u8], safepoints: i64) {
    let n = safepoints.max(1);
    window[ARM_COUNTDOWN_OFF as usize..ARM_COUNTDOWN_OFF as usize + 8]
        .copy_from_slice(&n.to_le_bytes());
    write_state(window, STATE_ARMED);
}

/// Arm a window to **freeze on quiesce** (DURABILITY.md §13.4 slice 4c-bis): the deterministic
/// trigger for a server idle in its accept loop. The run proceeds normally; the instant it would
/// otherwise block with only `svc.wait`-parked consumers left (no runnable work, no futex/svc
/// timers pending), the runtime promotes each parked domain's window to `UNWINDING` and re-admits
/// it — the re-executed `svc.wait` takes the trailing-poll sentinel, so the domain unwinds and the
/// freeze completes instead of the run hanging. Sets the quiesce flag ([`ARM_QUIESCE_OFF`]); the
/// two countdown slots stay 0 so the triggers never interfere. Unlike the countdown arms, the
/// state word stays `NORMAL` — the trigger is quiescence, detected scheduler-side, not a poll.
pub fn arm_freeze_on_quiesce(window: &mut [u8]) {
    window[ARM_QUIESCE_OFF as usize] = 1;
}

/// Arm a window to **freeze after `backedges` further loop back-edges** (the deterministic Phase-4
/// Slice A trigger for back-edge polls): the run proceeds normally, and the runtime promotes the
/// state word to `UNWINDING` at the `backedges`-th branch terminator so the next loop-header poll
/// begins the freeze — reaching a poll-free compute loop that no fiber-safepoint countdown can. Sets
/// the back-edge countdown ([`ARM_BACKEDGE_OFF`]); the fiber-safepoint countdown stays 0 so the two
/// triggers never interfere. A non-positive count is clamped to 1.
pub fn arm_freeze_after_backedges(window: &mut [u8], backedges: i64) {
    let n = backedges.max(1);
    window[ARM_BACKEDGE_OFF as usize..ARM_BACKEDGE_OFF as usize + 8]
        .copy_from_slice(&n.to_le_bytes());
    write_state(window, STATE_ARMED);
}

/// Read the state word from a window image.
pub fn read_state(window: &[u8]) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&window[STATE_OFF as usize..STATE_OFF as usize + 4]);
    i32::from_le_bytes(b)
}

// ---- small IR construction helpers ----

/// A block under construction that tracks the next block-local value index.
struct Bb {
    params: Vec<ValType>,
    insts: Vec<Inst>,
    next: u32,
}

impl Bb {
    fn new(params: Vec<ValType>) -> Self {
        let next = params.len() as u32;
        Bb {
            params,
            insts: Vec::new(),
            next,
        }
    }
    /// Push a single-result instruction; returns its value index.
    fn one(&mut self, i: Inst) -> ValIdx {
        let idx = self.next;
        self.insts.push(i);
        self.next += 1;
        idx
    }
    /// Push an instruction that defines `nres` consecutive values; returns their indices.
    fn many(&mut self, i: Inst, nres: usize) -> Vec<ValIdx> {
        let start = self.next;
        self.insts.push(i);
        self.next += nres as u32;
        (start..self.next).collect()
    }
    /// Push a zero-result instruction (a store).
    fn zero(&mut self, i: Inst) {
        self.insts.push(i);
    }
    /// §12.8 concurrent-thaw stage 1: the **freeze** state word's address (`UNWINDING`), as
    /// `(base, offset)` for a `load`. **Always global** ([`STATE_OFF`]) — a freeze is genuinely
    /// stop-the-world, so the single word is the natural broadcast every context's poll reads (the arm
    /// trigger / `request_freeze` set it). Read by the loop-header and in-block `UNWINDING` polls.
    fn freeze_word_addr(&mut self) -> (ValIdx, u64) {
        (self.one(Inst::ConstI64(STATE_OFF as i64)), 0)
    }
    /// §12.8 concurrent-thaw stage 1: the **thaw** state word's address (`REWINDING`/`NORMAL`), as
    /// `(base, offset)` for a `load`/`store` — the running context's own region word
    /// (`durable.shadow_base` + [`STATE_IN_REGION_OFF`], like the per-context shadow-SP word), so
    /// concurrent vCPUs each rewind against their own (the relocation's whole point). Read by the
    /// prologue's `REWINDING` dispatch and written `NORMAL` by the deepest frame's re-issue (thaw end).
    fn thaw_word_addr(&mut self) -> (ValIdx, u64) {
        (self.one(Inst::DurableShadowBase), STATE_IN_REGION_OFF)
    }
    fn finish(self, term: Terminator) -> Block {
        Block {
            params: self.params,
            insts: self.insts,
            term,
        }
    }
}

fn load(op: LoadOp, addr: ValIdx, offset: u64) -> Inst {
    Inst::Load { op, addr, offset }
}

fn store(op: StoreOp, addr: ValIdx, value: ValIdx, offset: u64) -> Inst {
    Inst::Store {
        op,
        addr,
        value,
        offset,
    }
}

fn ibin(ty: IntTy, op: BinOp, a: ValIdx, b: ValIdx) -> Inst {
    Inst::IntBin { ty, op, a, b }
}

fn icmp(ty: IntTy, op: CmpOp, a: ValIdx, b: ValIdx) -> Inst {
    Inst::IntCmp { ty, op, a, b }
}

/// The **poll** that follows a suspend op, ending the block that ran it: while a freeze or fork is
/// unwinding (the state word reads `UNWINDING`), on to the point's UNWIND check block `unwind_blk`,
/// which spills `live` and returns; otherwise on to the continuation `cont` with the same values. The
/// forward path polls after the op, and so does an arm that ran the op again on a thaw — a re-issued
/// call is as live as the original, and an unwind can start beneath it later.
fn poll(b: &mut Bb, unwind_blk: BlockIdx, cont: BlockIdx, live: Vec<ValIdx>) -> Terminator {
    let (st_a, st_off) = b.freeze_word_addr();
    let st = b.one(load(LoadOp::I32, st_a, st_off));
    let unw = b.one(Inst::ConstI32(STATE_UNWINDING));
    let is_unw = b.one(icmp(IntTy::I32, CmpOp::Eq, st, unw));
    Terminator::BrIf {
        cond: is_unw,
        then_blk: unwind_blk,
        then_args: live.clone(),
        else_blk: cont,
        else_args: live,
    }
}

/// An arm's terminator for a point that either delivers its reloaded result or **re-issues** its op:
/// `cond` selects a re-issue block, appended to `extra_blocks` (numbered from `first_extra`). That
/// block takes the continuation's args minus the op's results, then the op's reloaded `operands`;
/// `make` builds the op over those operand params, and its results complete the continuation's args,
/// which the re-issued op's poll ([`poll`], unwinding to `unwind_blk`) hands on.
#[allow(clippy::too_many_arguments)]
fn reissue_branch(
    extra_blocks: &mut Vec<Block>,
    first_extra: u32,
    unwind_blk: BlockIdx,
    pt: &PointPlan,
    cont_args: Vec<ValIdx>,
    cond: ValIdx,
    operands: &[ValIdx],
    reloaded: &[ValIdx],
    make: impl FnOnce(&[ValIdx]) -> Inst,
) -> Terminator {
    let slot = |v: ValIdx| {
        pt.spilled
            .binary_search(&(v as usize))
            .expect("a re-issued op's operand is spilled")
    };
    let kept = pt.out - pt.nres;
    let mut params = pt.slot_types[..kept].to_vec();
    params.extend(operands.iter().map(|&v| pt.slot_types[v as usize]));
    let mut rb = Bb::new(params);
    let n = kept as u32;
    let op_params: Vec<ValIdx> = (n..n + operands.len() as u32).collect();
    let r = rb.many(make(&op_params), pt.nres);
    let mut args: Vec<ValIdx> = (0..n).collect();
    args.extend(r);
    let blk = first_extra + extra_blocks.len() as u32;
    let term = poll(&mut rb, unwind_blk, pt.cont_seg, args);
    extra_blocks.push(rb.finish(term));
    let mut then_args = cont_args[..kept].to_vec();
    then_args.extend(operands.iter().map(|&v| reloaded[slot(v)]));
    Terminator::BrIf {
        cond,
        then_blk: blk,
        then_args,
        else_blk: pt.cont_seg,
        else_args: cont_args,
    }
}

/// A host call `op` rebuilt over new operands, in [`inst_operands`] order (handle first, then args).
fn with_operands(op: &Inst, ops: &[ValIdx]) -> Inst {
    match op {
        Inst::CapCall {
            type_id, op, sig, ..
        } => Inst::CapCall {
            type_id: *type_id,
            op: *op,
            sig: *sig,
            handle: ops[0],
            args: ops[1..].to_vec(),
        },
        Inst::CallImportDyn { ty, op, sig, .. } => Inst::CallImportDyn {
            ty: *ty,
            op: *op,
            sig: *sig,
            handle: ops[0],
            args: ops[1..].to_vec(),
        },
        Inst::CallSym { import, sig, .. } => Inst::CallSym {
            import: *import,
            sig: *sig,
            handle: ops[0],
            args: ops[1..].to_vec(),
        },
        Inst::CallImport {
            import, op, sig, ..
        } => Inst::CallImport {
            import: *import,
            op: *op,
            sig: *sig,
            args: ops.to_vec(),
        },
        _ => unreachable!("a leaf is a host call"),
    }
}

fn zero_const(t: ValType) -> Inst {
    match t {
        ValType::I32 => Inst::ConstI32(0),
        ValType::I64 => Inst::ConstI64(0),
        ValType::F32 => Inst::ConstF32(0),
        ValType::F64 => Inst::ConstF64(0),
        ValType::V128 => Inst::ConstV128([0; 16]),
        // An opaque `ref` is i64-width (GC.md §6 reservation); its zero is the i64 zero word.
        ValType::Ref => Inst::ConstI64(0),
        // §3.5 `cap` is i32-width handle data.
        ValType::Cap => Inst::ConstI32(0),
    }
}

fn vsize(t: ValType) -> u64 {
    match t {
        ValType::I32 | ValType::F32 | ValType::Cap => 4,
        ValType::I64 | ValType::F64 | ValType::Ref => 8,
        ValType::V128 => 16,
    }
}

fn align_up(x: u64, a: u64) -> u64 {
    (x + a - 1) & !(a - 1)
}

/// Spill one frame slot of type `t` at `addr + offset` (#1300 Phase 2: a `v128` slot spills
/// through `v128.store`, 16-byte aligned by the frame layout; every scalar through `store`).
fn spill(t: ValType, addr: ValIdx, value: ValIdx, offset: u64) -> Inst {
    let op = match t {
        ValType::I32 | ValType::Cap => StoreOp::I32,
        ValType::I64 | ValType::Ref => StoreOp::I64, // `ref` spills as its opaque i64 word
        ValType::F32 => StoreOp::F32,
        ValType::F64 => StoreOp::F64,
        ValType::V128 => {
            return Inst::V128Store {
                addr,
                value,
                offset,
            }
        }
    };
    store(op, addr, value, offset)
}

/// Reload one frame slot of type `t` from `addr + offset` — the twin of [`spill`].
fn reload(t: ValType, addr: ValIdx, offset: u64) -> Inst {
    let op = match t {
        ValType::I32 | ValType::Cap => LoadOp::I32,
        ValType::I64 | ValType::Ref => LoadOp::I64, // `ref` reloads as its opaque i64 word
        ValType::F32 => LoadOp::F32,
        ValType::F64 => LoadOp::F64,
        ValType::V128 => return Inst::V128Load { addr, offset },
    };
    load(op, addr, offset)
}

/// Result types of an instruction, given the types of all earlier values in the block
/// and each function's result types. Covers the scalar/memory/call subset a Phase-1
/// prefix can use; returns `UnsupportedInst` for anything else — the ops whose state does not
/// live in values the shadow frame can carry: `setjmp`/`longjmp` (an interpreter-frame jump
/// buffer), `gc.roots`, `import.attach` (host binding-table mutation) and vCPU TLS (unless the
/// unwind carries the thread, [`TransformOpts::carries_thread`]) — so the transform fails closed
/// rather than mis-typing a frame.
///
/// Deliberately **not** `temen_verify::func_value_types` (#913): that one is whole-function and
/// **total** — it types every op and degrades gracefully (an underivable value is simply absent)
/// because it feeds the debugger's best-effort view. This one is per-instruction and **fail-closed**
/// — an op outside the durable Phase-1 subset must *error*, not be typed, or the transform would
/// spill/reload a frame slot it can't safely freeze. The narrower, erroring shape is the point; the
/// two are kept separate on purpose rather than merged.
fn result_types(
    inst: &Inst,
    types: &[ValType],
    func_results: &[Vec<ValType>],
    type_section: &[TypeEntry],
    opts: &TransformOpts,
) -> Result<Vec<ValType>, TransformError> {
    use Inst::*;
    Ok(match inst {
        ConstI32(_) => vec![ValType::I32],
        ConstI64(_) => vec![ValType::I64],
        ConstF32(_) => vec![ValType::F32],
        ConstF64(_) => vec![ValType::F64],
        ConstV128(_) => vec![ValType::V128],
        IntBin { ty, .. } | IntUn { ty, .. } => vec![ty.val()],
        FBin { ty, .. } | FUn { ty, .. } => vec![ty.val()],
        IntCmp { .. } | FCmp { .. } | Eqz { .. } => vec![ValType::I32],
        // Scalar conversions (#1300 Phase 2, item 3): each yields one scalar the shadow frame
        // already spills/reloads (i32/i64/f32/f64), typed from the op itself — width conversions,
        // saturating and trapping float→int, int→float, and the float casts/reinterprets.
        Convert { op, .. } => vec![op.sig().2],
        FToISat { op, .. } | FToITrap { op, .. } => vec![op.parts().1.val()],
        IToFConv { op, .. } => vec![op.parts().1.val()],
        Cast { op, .. } => vec![op.sig().2],
        Fma { ty, .. } => vec![ty.val()],
        // Address constants and §3.5 reflection: pure, one scalar each.
        DataSym { .. } | DataSelf { .. } | DataTop => vec![ValType::I64],
        CapSelfTypeId { .. } | CapSelfCovers { .. } | ExportHandle { .. } => vec![ValType::I32],
        // Bulk memory ops: no results (guest-memory ops — the strict gate refuses them, the
        // confined path admits them, like `load`/`store`).
        MemCopy { .. } | MemMove { .. } | MemFill { .. } => vec![],
        // §17 SIMD (#1300 Phase 2): a `v128` spills/reloads through its own load/store ops, so
        // every vector op types like any scalar — one `v128`, a lane scalar, or an `i32` mask.
        V128Load { .. }
        | Splat { .. }
        | ReplaceLane { .. }
        | VIntBin { .. }
        | VIntCmp { .. }
        | VShift { .. }
        | VFloatBin { .. }
        | VFloatCmp { .. }
        | VPMinMax { .. }
        | VFma { .. }
        | VFloatUn { .. }
        | VIntUn { .. }
        | VPopcnt { .. }
        | VSatBin { .. }
        | VAvgr { .. }
        | VDot { .. }
        | VDotI8 { .. }
        | VExtMul { .. }
        | VExtAddPairwise { .. }
        | VQ15MulrSat { .. }
        | VWiden { .. }
        | VNarrow { .. }
        | VConvert { .. }
        | VBitBin { .. }
        | VNot { .. }
        | Bitselect { .. }
        | Shuffle { .. }
        | Swizzle { .. } => vec![ValType::V128],
        ExtractLane { shape, .. } => vec![shape.lane_val()],
        VAnyTrue { .. } | VAllTrue { .. } | VBitmask { .. } => vec![ValType::I32],
        V128Store { .. } => vec![],
        AtomicLoad { ty, .. } | AtomicRmw { ty, .. } | AtomicCmpxchg { ty, .. } => vec![ty.val()],
        Store { .. } | AtomicStore { .. } | AtomicFence { .. } => vec![],
        Select { a, .. } => vec![types[*a as usize]],
        Load { op, .. } => vec![load_result_ty(*op)],
        Call { func, .. } => func_results
            .get(*func as usize)
            .cloned()
            .ok_or(TransformError::UnsupportedShape)?,
        CapCall { sig, .. }
        | CallImport { sig, .. }
        | CallImportDyn { sig, .. }
        | CallSym { sig, .. } => sig_of(type_section, *sig).results.clone(),
        CallIndirect { ty, .. } => sig_of(type_section, *ty).results.clone(),
        RefFunc { .. } => vec![ValType::I32],
        // Fiber control ops (§12 / Phase 3): an i64 handle, a `(status, value)` pair, a resume arg.
        ContNew { .. } => vec![ValType::I64],
        ContResume { .. } => vec![ValType::I32, ValType::I64],
        Suspend { .. } => vec![ValType::I64],
        // §12 thread ops (Phase 3.2): `thread.spawn` yields an `i32` handle, `thread.join` an `i64`
        // result. Neither is a may-suspend checkpoint — they are copied verbatim into their segment and
        // their results spill/reload like any scalar; the multi-vCPU freeze/thaw choreography is the
        // runtime's (durable §12.8 slice 3.2.1), so the transform only needs to type them.
        ThreadSpawn { .. } => vec![ValType::I32],
        ThreadJoin { .. } => vec![ValType::I64],
        // §12 futex ops: `atomic.wait` yields an `i32` status (woken / not-equal / timed-out),
        // `atomic.notify` an `i32` woken count. `atomic.wait` is a may-suspend re-issue safepoint
        // (the parked-vCPU slice); `atomic.notify` is copied verbatim into its segment.
        MemoryWait { .. } | MemoryNotify { .. } => vec![ValType::I32],
        // §12 vCPU TLS: the register is the thread's, so only an unwind that carries the thread keeps
        // it — `get` then yields a scalar like any other, `set` nothing.
        VcpuTlsGet if opts.carries_thread => vec![ValType::I64],
        VcpuTlsSet { .. } if opts.carries_thread => vec![],
        _ => return Err(TransformError::UnsupportedInst),
    })
}

fn load_result_ty(op: LoadOp) -> ValType {
    use LoadOp::*;
    match op {
        I32 | I32_8S | I32_8U | I32_16S | I32_16U => ValType::I32,
        I64 | I64_8S | I64_8U | I64_16S | I64_16U | I64_32S | I64_32U => ValType::I64,
        F32 => ValType::F32,
        F64 => ValType::F64,
    }
}

#[cfg(test)]
mod tests {
    /// The arena every test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
    const TEST_ARENA: ShadowArena = ShadowArena {
        base: 16448,
        end: 65536,
    };
    use super::*;
    use temen_ir::Memory;

    fn parse_with_mem(src: &str, size_log2: u8) -> Module {
        let mut m = temen_text::parse_module(src).expect("parse");
        m.memory = Some(Memory {
            size_log2,
            shadow: Some(TEST_ARENA),
        });
        m
    }

    #[test]
    fn no_cap_call_is_left_unchanged() {
        let m = parse_with_mem(
            "func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
            12,
        );
        let out = transform_module(&m).expect("transform");
        assert_eq!(out.funcs, m.funcs, "function without call.cap is untouched");
    }

    #[test]
    fn instrumented_function_verifies() {
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        temen_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            9,
            "one host-call point: 4n+4 blocks + its re-issue block"
        );
    }

    #[test]
    fn two_cap_calls_become_two_resume_points() {
        // Two host-call points: 4·2 + 4 blocks, plus one re-issue block each.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  v3 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("two resume points are in scope");
        temen_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            14,
            "two-point layout: 4n+4 with n=2, plus two re-issue blocks"
        );
    }

    /// #1300 Phase 2 (item 3): scalar conversions in the may-suspend prefix — a value converted before
    /// the suspend point and used after it spills as its (converted) scalar type and reloads on
    /// resume. The instrumented function verifies; the four conversion families each yield one
    /// scalar the shadow frame already models.
    #[test]
    fn conversions_in_the_prefix_instrument_and_verify() {
        let m = parse_with_mem(
            "func (i32, f64) -> (i64) {\nblock 0 (v0: i32, vf: f64) {\n  \
             v1 = i64.extend_i32_u v0\n  v2 = i32.wrap_i64 v1\n  \
             v3 = i32.trunc_sat_f64_s vf\n  v4 = f32.convert_i32_s v3\n  v5 = f64.promote_f32 v4\n  \
             v6 = i64.trunc_f64_s v5\n  \
             v7 = call.cap 2 0 (i32) -> (i64) v0 (v2)\n  \
             v8 = i64.add v1 v7\n  v9 = i64.add v8 v6\n  return v9\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("conversions are in the Phase-2 prefix model");
        temen_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            9,
            "one host-call point: 4n+4 blocks + its re-issue block"
        );
    }

    /// #1300 Phase 2 (item 3, v128 half): a `v128` live across the suspend point spills through
    /// `v128.store` into a 16-byte-aligned frame slot and reloads through `v128.load`; the
    /// instrumented function verifies.
    #[test]
    fn a_v128_live_across_the_suspend_point_instruments_and_verifies() {
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  \
             v1 = i64.const 7\n  v2 = i64x2.splat v1\n  v3 = i32.const 0\n  \
             v4 = call.cap 2 0 (i32) -> (i64) v0 (v3)\n  \
             v5 = i64x2.extract_lane 1 v2\n  v6 = i64.add v4 v5\n  return v6\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("a v128 in the live set is in scope");
        temen_verify::verify_module(&out).expect("instrumented IR must verify");
        let spills = out.funcs[0]
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, Inst::V128Store { .. }))
            .count();
        let reloads = out.funcs[0]
            .blocks
            .iter()
            .flat_map(|b| b.insts.iter())
            .filter(|i| matches!(i, Inst::V128Load { .. }))
            .count();
        assert_eq!(
            (spills, reloads),
            (1, 1),
            "the v128 spills once and reloads once"
        );
    }

    /// #1300 Phase 2: a `call.import` (a capability call bound at run time) is a suspend point
    /// exactly like `call.cap` — the function is may-suspend, the site gets a poll + resume arm,
    /// and the instrumented IR verifies.
    #[test]
    fn a_call_import_is_a_leaf_suspend_point() {
        let m = parse_with_mem(
            "import 0 \"clock\" (i32) -> (i64)\n\
             func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  \
             v2 = call.import 0 (v1)\n  v3 = i64.const 100\n  v4 = i64.add v2 v3\n  return v4\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("call.import is a modeled suspend point");
        temen_verify::verify_module(&out).expect("instrumented IR must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            9,
            "one host-call point: 4n+4 blocks + its re-issue block"
        );
    }

    #[test]
    fn propagated_chain_instruments_each_frame() {
        // A two-level chain: the caller suspends on its `call` to the leaf, the leaf on
        // its `call.cap`. Both are may-suspend, so both get the 7-block instrumentation.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = call 1 (v0)\n  return v1\n  }\n}\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("transform");
        temen_verify::verify_module(&out).expect("instrumented chain must verify");
        assert_eq!(
            out.funcs[0].blocks.len(),
            8,
            "caller (propagated) instrumented"
        );
        assert_eq!(out.funcs[1].blocks.len(), 9, "callee (leaf) instrumented");
    }

    #[test]
    fn non_suspending_callee_is_left_unchanged() {
        // func 0 (leaf call.cap) calls func 1 (a pure helper) as a *prefix* op. The helper
        // never suspends, so it is not instrumented and func 0's only suspend point is its
        // own call.cap; the helper's result is spilled/reloaded, never re-issued.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = call 1 (v0)\n  v2 = i32.const 0\n  v3 = call.cap 2 0 (i32) -> (i64) v0 (v2)\n  v4 = i64.add v1 v3\n  return v4\n  }\n}\nfunc (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i64.const 5\n  return v1\n  }\n}\n",
            18,
        );
        let helper_before = m.funcs[1].clone();
        let out = transform_module(&m).expect("transform");
        temen_verify::verify_module(&out).expect("verify");
        assert_eq!(out.funcs[0].blocks.len(), 9, "leaf instrumented");
        assert_eq!(
            out.funcs[1], helper_before,
            "non-suspending helper untouched"
        );
    }

    #[test]
    fn instrumented_module_with_guest_memory_op_is_rejected() {
        // A guest store could alias the durable region below the arena → R9 fails closed.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  v3 = i64.const 7\n  i64.store v1 v3\n  return v2\n  }\n}\n",
            18,
        );
        assert_eq!(transform_module(&m), Err(TransformError::GuestUsesMemory));
    }

    #[test]
    fn guest_memory_op_in_uninstrumented_module_is_fine() {
        // No `call.cap` anywhere ⇒ nothing is instrumented ⇒ no durable region ⇒ the
        // guest's own memory use is left untouched.
        let m = parse_with_mem(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i64.const 7\n  i64.store v0 v1\n  v2 = i64.load v0\n  return v2\n  }\n}\n",
            18,
        );
        let out = transform_module(&m).expect("no instrumentation, memory use is fine");
        assert_eq!(out.funcs, m.funcs, "left unchanged");
    }

    #[test]
    fn cap_call_without_memory_is_rejected() {
        let mut m = temen_text::parse_module(
            "func (i32) -> (i64) {\nblock 0 (v0: i32) {\n  v1 = i32.const 0\n  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)\n  return v2\n  }\n}\n",
        )
        .unwrap();
        m.memory = None;
        assert_eq!(transform_module(&m), Err(TransformError::NoMemory));
    }
}
