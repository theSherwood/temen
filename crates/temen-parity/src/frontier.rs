//! # The capability × axis frontier matrix
//!
//! INVARIANTS #14 says an accepted capability must hold across **seven axes** — runtime backend,
//! host target, concurrency model, code origin, nesting, debugger, durability. Exactly one of those
//! had a machine-checked matrix ([`crate::catalog`], `op × backend`). This is the machine for the
//! rest, and #1413 tracks filling it in.
//!
//! ## Why a second matrix rather than a second crate
//!
//! Same three properties that make the op matrix work, and the same reasons (INVARIANTS #15 — a
//! second position is a *parameter* of one structure, not a copy of it):
//!
//! 1. **Exhaustive** — [`capability_axes`] matches [`Capability`] with **no wildcard arm**, so a new
//!    capability fails to compile until it is classified on every axis.
//! 2. **Code-derived** — rows are `temen_ir::cap_id` kinds, so the row set tracks the powerbox
//!    rather than drifting from it.
//! 3. **Honesty-pinned** — `tests/frontier_conformance.rs` drives the *real* predicates
//!    (`Host::can_regrant`, the durable-capture classifier) and fails when this manifest disagrees
//!    with them. A manifest nothing checks is a wish.
//!
//! It shares [`crate::Status`] deliberately: one status vocabulary across both matrices means the
//! renderer, the JSON view and the playground page work for either without a second set of glyphs,
//! ids and colours to keep in step.
//!
//! ## Which axes are filled in, and why those
//!
//! **Nesting** and **durability** came first: both are *already* decided by an explicit,
//! wildcard-free predicate in the tree — `Host::can_regrant` and the `NonDurableKind` match in
//! `capture_durable_handles`. So they can be derived and conformance-tested rather than asserted,
//! which is the difference between a matrix and a wish-list.
//!
//! **Debugger** followed, and is a different shape. No single function classifies capabilities on
//! it: two gates decide, the bytecode lowering's `(type_id, op)` table and `service_advance`'s
//! decline set, and neither is keyed by capability — the first is over op pairs, the second over
//! scheduler seams. So the column is derived by *running* each capability's ops under
//! `ScheduledDebugRun` and folding the verdicts (`tests/debugger_conformance.rs`). It became
//! derivable at all only once the debugger driver's declines were named rather than caught by a `_`
//! (#1414): before that a new seam joined the declined set silently, so a column reading it would
//! have gone stale with nothing going red.
//!
//! Four rows on that column stay [`Status::Unaudited`] — `Jit` (its `invoke` needs a unit only a
//! running guest can mint), `JitCode`, `Offer` and `LiveImpl` (a live peer). A row is scored only
//! when *every* one of its ops was actually driven; scoring one on a partial sweep would be the
//! wish the matrix exists to avoid.
//!
//! **Concurrency** is the third shape again. There is no predicate to read at all: whether a
//! capability is "carried by both drivers" is only answerable by running it on both, so the column
//! is derived by driving each capability's ops on `bytecode::drive` and `bytecode::run_vcpu_parallel`
//! and comparing the answers shape by shape (`tests/concurrency_conformance.rs`). The comparison is
//! of *answers*, not of refusals: the op-15 gap #1531 closed was a driver returning a different
//! value, not refusing, so a column that only asked "does it trap" would have scored it `Full`.
//!
//! Its first rendering found one: `child_offer` answers `-EINVAL` on the coop driver and traps on
//! the parallel one (#1566). The same four rows that need a live unit or peer stay `Unaudited` here
//! as on the debugger column.
//!
//! The remaining three axes are populated as their predicates become locatable; until then their
//! cells read `Unaudited`, which is the point — an unaudited cell is visible, countable, and cannot
//! be mistaken for a passing one.
//!
//! ## What the first rendering already showed
//!
//! Put side by side, the two axes are near-complements: almost everything a §14 child can be granted
//! is **non-durable** (`SharedRegion`, `Module`, `Offer`, pipe ends, `HostProc`), and the two purely
//! window-coordinate caps a child can *never* be granted — `AddressSpace`, `Instantiator` — are
//! exactly the ones that **are** durable. `Stream`/`Exit`/`Clock` and `Jit` are both; `Budget` is
//! durable but never crosses into a child (it hands down a *sub*-budget by `split`/`transfer`, not
//! the handle — #1502 made it durable, it was neither before). Nobody could see that before, because
//! the two classifications live ~2,000 lines apart in two functions that never mention each other.

use crate::{Cell, Status};

/// A powerbox capability kind — one matrix row. Mirrors `temen_ir::cap_id`, which is the registry
/// (including which ids are retired-not-reused), plus the two guest-side shapes that have no
/// `cap_id` of their own because they live above [`temen_ir::cap_id::GUEST_IMPL_BASE`].
///
/// Retired ids (`3` ex-`Memory`, `9` ex-`IoRing`, `15` ex-`WindowMinter`) are deliberately absent:
/// they name nothing a guest can hold, so they have no frontier to be consistent across.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Capability {
    Stream,
    Exit,
    Clock,
    SharedRegion,
    AddressSpace,
    Instantiator,
    ModuleLoader,
    Module,
    Blocking,
    Jit,
    JitCode,
    HostProc,
    Budget,
    /// A pipe end — a `Stream`-typed handle carrying a shared FIFO backing, classified separately
    /// because its index-carrying shape makes it behave unlike a plain `Stream` on both axes.
    PipeEnd,
    /// A wired interface offer (IMPORTS.md §3.2).
    Offer,
    /// A §3.6 live-callee offer — points at a *running* domain's powerbox.
    LiveImpl,
}

impl Capability {
    /// Every row, in matrix order. Grouped so the rendered table reads as the powerbox does:
    /// coordinate-free caps, then window-coordinate caps, then code//authority, then the guest-side
    /// shapes.
    pub const ALL: [Capability; 16] = [
        Capability::Stream,
        Capability::Exit,
        Capability::Clock,
        Capability::PipeEnd,
        Capability::SharedRegion,
        Capability::AddressSpace,
        Capability::Instantiator,
        Capability::Budget,
        Capability::Module,
        Capability::ModuleLoader,
        Capability::Jit,
        Capability::JitCode,
        Capability::Blocking,
        Capability::HostProc,
        Capability::Offer,
        Capability::LiveImpl,
    ];

    /// The name the matrix prints — the `cap_id` constant's own spelling, so a reader can grep it.
    pub fn name(self) -> &'static str {
        match self {
            Capability::Stream => "Stream",
            Capability::Exit => "Exit",
            Capability::Clock => "Clock",
            Capability::PipeEnd => "PipeEnd",
            Capability::SharedRegion => "SharedRegion",
            Capability::AddressSpace => "AddressSpace",
            Capability::Instantiator => "Instantiator",
            Capability::Budget => "Budget",
            Capability::Module => "Module",
            Capability::ModuleLoader => "ModuleLoader",
            Capability::Jit => "Jit",
            Capability::JitCode => "JitCode",
            Capability::Blocking => "Blocking",
            Capability::HostProc => "HostProc",
            Capability::Offer => "Offer",
            Capability::LiveImpl => "LiveImpl",
        }
    }

    /// The `temen_ir::cap_id` value, where the capability has one. `None` for the two guest-side
    /// shapes above `GUEST_IMPL_BASE` and for `PipeEnd` (a `Stream`-typed handle, not its own id).
    pub fn cap_id(self) -> Option<u32> {
        use temen_ir::cap_id as c;
        Some(match self {
            Capability::Stream | Capability::PipeEnd => c::STREAM,
            Capability::Exit => c::EXIT,
            Capability::Clock => c::CLOCK,
            Capability::SharedRegion => c::SHARED_REGION,
            Capability::AddressSpace => c::ADDRESS_SPACE,
            Capability::Instantiator => c::INSTANTIATOR,
            Capability::ModuleLoader => c::MODULE_LOADER,
            Capability::Module => c::MODULE,
            Capability::Blocking => c::BLOCKING,
            Capability::Jit => c::JIT,
            Capability::JitCode => c::JIT_CODE,
            Capability::HostProc => c::HOST_PROC,
            Capability::Budget => c::BUDGET,
            Capability::Offer | Capability::LiveImpl => return None,
        })
    }
}

/// The INVARIANTS #14 axes, in matrix-column order. All seven are listed even though only two are
/// populated: an axis missing from the enum is an axis nobody remembers to ask about, which is the
/// failure this matrix exists to prevent.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Axis {
    /// Can a **§14 child** (nested carve or detached) be granted this? Decided by
    /// `Host::can_regrant`.
    Nesting,
    /// Does a handle of this kind **survive a freeze**? Decided by the `NonDurableKind` match in
    /// `Host::capture_durable_handles`.
    Durability,
    /// bytecode / Cranelift JIT / wasm-JIT vs. the tree-walk oracle.
    RuntimeBackend,
    /// native (x86/arm), wasm32, Windows.
    HostTarget,
    /// The cooperative multiplex driver and the genuinely-parallel per-Worker driver.
    ConcurrencyModel,
    /// The host-translated base module and a §22 guest-JIT unit.
    CodeOrigin,
    /// Observable under the debug tier (disciplined by #9's observability corollary).
    Debugger,
}

impl Axis {
    pub const ALL: [Axis; 7] = [
        Axis::Nesting,
        Axis::Durability,
        Axis::RuntimeBackend,
        Axis::HostTarget,
        Axis::ConcurrencyModel,
        Axis::CodeOrigin,
        Axis::Debugger,
    ];

    /// The column header.
    pub fn short(self) -> &'static str {
        match self {
            Axis::Nesting => "nesting",
            Axis::Durability => "durability",
            Axis::RuntimeBackend => "backend",
            Axis::HostTarget => "target",
            Axis::ConcurrencyModel => "concurrency",
            Axis::CodeOrigin => "code origin",
            Axis::Debugger => "debugger",
        }
    }

    /// The question the column answers, for the rendered legend.
    pub fn question(self) -> &'static str {
        match self {
            Axis::Nesting => "can a §14 child hold it? (`Host::can_regrant`)",
            Axis::Durability => "does it survive a freeze? (`capture_durable_handles`)",
            Axis::RuntimeBackend => "same on every engine? (see OPS_PARITY.md for op-level detail)",
            Axis::HostTarget => "same on native / wasm32 / Windows?",
            Axis::ConcurrencyModel => "carried by both the coop and per-Worker drivers?",
            Axis::CodeOrigin => "usable from a §22 guest-JIT unit as from the base module?",
            Axis::Debugger => "observable under the debug tier?",
        }
    }

    /// Whether this column's cells are checked against a live predicate — `nesting` and
    /// `durability` by `tests/frontier_conformance.rs`, `debugger` by
    /// `tests/debugger_conformance.rs`. An unconformed column states the manifest's belief and
    /// nothing more — worth saying out loud in the rendered matrix.
    ///
    /// A conformed column may still hold `Unaudited` cells (`debugger` holds four): the claim is
    /// that the column is *checked*, not that every row in it could be reached.
    pub fn is_conformed(self) -> bool {
        matches!(
            self,
            Axis::Nesting | Axis::Durability | Axis::Debugger | Axis::ConcurrencyModel
        )
    }
}

/// `Full`, no note.
const F: Cell = Cell {
    status: Status::Full,
    note: "",
};
/// Not audited on this axis yet — see the module docs. Distinct from `NotYet` (a *known* gap).
const U: Cell = Cell {
    status: Status::Unaudited,
    note: "",
};
/// `Full` on the **concurrency** axis: both drivers, given the same call, give the same answer.
/// `tests/concurrency_conformance.rs` drives each of the row's ops on `bytecode::drive` and on
/// `bytecode::run_vcpu_parallel` and compares them shape by shape.
const K: Cell = Cell {
    status: Status::Full,
    note: "",
};
/// The one concurrency-axis gap the column's first rendering found (#1566): `child_offer` (op 14)
/// answers `-EINVAL` on the cooperative driver and **traps** `ThreadFault` on the parallel one, so a
/// guest probing a stale child handle survives on one driver and dies on the other — INVARIANTS #5
/// (errors are values) and #9 (refuse probeably, never diverge). A `NotYet`, not a `Declines`: the
/// op works, the two drivers disagree about how it fails.
const CHILD_OFFER_DIVERGES: Cell = Cell {
    status: Status::NotYet,
    note: "child_offer (op 14) answers -EINVAL on the coop driver and traps on the parallel one (#1566)",
};

const fn declines(note: &'static str) -> Cell {
    Cell {
        status: Status::Declines,
        note,
    }
}

const fn conditional(note: &'static str) -> Cell {
    Cell {
        status: Status::Conditional,
        note,
    }
}

/// **The classifier.** Exhaustive over [`Capability`], no wildcard arm — a new capability fails to
/// compile until it is classified on all seven axes.
///
/// Cell order matches [`Axis::ALL`].
pub fn capability_axes(c: Capability) -> [Cell; 7] {
    // Read as: [nesting, durability, backend, target, concurrency, code origin, debugger].
    match c {
        // Coordinate-free value caps: copyable into a child (`resolve_copyable`) and value-typed, so
        // they ride a freeze. The only rows that are unconditionally `Full` on both audited axes.
        Capability::Stream | Capability::Exit | Capability::Clock => [F, F, U, U, K, U, F],

        // A pipe end is `Stream`-typed but index-carrying: `regrant_into_child` aliases its shared
        // FIFO into the child (the cross-domain `cmd1 | cmd2` grant), while a freeze cannot carry the
        // live queue.
        Capability::PipeEnd => [
            F,
            declines("the live FIFO backing cannot be serialized (NonDurableKind::Pipe)"),
            U,
            U,
            K,
            U,
            F,
        ],

        // Re-granting aliases the SAME backing into the child (the explicit data plane); a byte
        // snapshot cannot reproduce a live alias into shared backing — INVARIANTS #14's one recorded
        // (provisional) exception.
        Capability::SharedRegion => [
            F,
            declines("a snapshot cannot reproduce a live alias into shared backing (#14 exception)"),
            U,
            U,
            K,
            U,
            conditional(
                "map/unmap/len/page_size run; op 4 (the guest-minted-region grant) is vetoed by \
                 name in the bytecode lowering",
            ),
        ],

        // Window-coordinate authority: a child is minted its OWN `AddressSpace`/`Instantiator` over
        // its own window, never handed the parent's — the parent's names coordinates meaningless in
        // the child (PROCESS.md §4: never implicit carve addresses). Both are value-typed, so they
        // survive a freeze.
        Capability::AddressSpace => [
            declines("the child is minted its own over its own window; the parent's names coordinates the child cannot use"),
            F,
            U,
            U,
            K,
            U,
            F,
        ],
        // Identical to `AddressSpace` on both predicate-audited axes, and deliberately its own arm
        // because the **debugger** axis splits them: the memory half of the §14 pair compiles whole
        // for the debug tier, the spawn half does not.
        Capability::Instantiator => [
            declines("the child is minted its own over its own window; the parent's names coordinates the child cannot use"),
            F,
            U,
            U,
            CHILD_OFFER_DIVERGES,
            U,
            conditional(
                "instantiate/join/instantiate_module_named/instantiate_detached compile; the \
                 coroutine spawns and instantiate_rec fall back, and child_offer (op 14) reaches \
                 the debug scheduler and is declined",
            ),
        ],

        // Declines on nesting (its index into `Host::budgets` is meaningless in another table — a child
        // is granted a *sub*-budget by `split`/`transfer`, never the handle) but **durable** since
        // #1502: the artifact carries the remaining quotas verbatim (`DurableBinding::Budget`) and the
        // thaw may only attenuate them. This closed the caveat INVARIANTS #3's R2 ruling had recorded
        // against itself ("minting authority does not survive a freeze").
        Capability::Budget => [
            declines("index-carrying: the child is granted a sub-budget by split/transfer, not the handle"),
            F,
            U,
            U,
            K,
            U,
            F,
        ],

        // An immutable instantiable artifact: shared into the child (FORK.md §8.6), but its host-side
        // registration cannot be serialized, so the embedder re-grants after restore.
        Capability::Module => [
            F,
            declines("NonDurableKind::Module — re-granted by the embedder after restore"),
            U,
            U,
            K,
            U,
            F,
        ],
        Capability::ModuleLoader => [
            declines("not in `can_regrant`: a child that may mint modules must be granted one explicitly"),
            declines("NonDurableKind::ModuleLoader — a live loader makes the domain non-snapshottable"),
            U,
            U,
            K,
            U,
            F,
        ],

        // #1296 — a §22 `Jit` grant crosses into a §14 child with a FRESH, empty unit table (sharing
        // one across the boundary would let either side's `install` alias into the other's
        // `call.dyn`), quota-attenuated to the parent's remaining. Durable since slice 2: the unit
        // state rides the artifact via `capture_durable_jit`.
        Capability::Jit => [F, F, U, U, U, U, U],
        // Not in `can_regrant` — a compiled-code handle names a unit in ITS table.
        Capability::JitCode => [
            declines("not in `can_regrant`: names a unit in its own table, meaningless in the child's"),
            F,
            U,
            U,
            U,
            U,
            U,
        ],

        Capability::Blocking => [
            declines("not in `can_regrant`: index-carrying into the parent's blocking table"),
            declines("NonDurableKind::Blocking"),
            U,
            U,
            K,
            U,
            F,
        ],
        // Only a *forkable* host proc crosses (one carrying a provider fork factory); a factory-less
        // opaque closure cannot be re-minted over the shared provider state.
        Capability::HostProc => [
            Cell {
                status: Status::Conditional,
                note: "only a forkable host proc (one carrying a fork factory) crosses; a factory-less one cannot",
            },
            declines("NonDurableKind::HostProc — the host closure cannot be serialized"),
            U,
            U,
            K,
            U,
            F,
        ],

        // Guest-side shapes: both re-grant into a child (an offer wires the interface; a live impl
        // wires the SAME running domain, so two siblings coordinate through a peer their parent
        // introduced), and neither survives a freeze — both carry out-of-line references to another
        // domain.
        Capability::Offer => [
            F,
            declines("NonDurableKind::Offer — an out-of-line reference to the offering domain"),
            U,
            U,
            U,
            U,
            U,
        ],
        Capability::LiveImpl => [
            F,
            declines("NonDurableKind::LiveImpl — points at a *running* domain's powerbox"),
            U,
            U,
            U,
            U,
            U,
        ],
    }
}
