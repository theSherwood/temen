//! # The op × backend parity matrix
//!
//! One **code-derived, test-checked** source of truth for how every IR op behaves across the four
//! backends (DESIGN.md §3): the tree-walk interpreter (the oracle), the bytecode interpreter, the
//! Cranelift JIT, and the wasm-JIT. It answers, at a glance, *where the backends diverge and why* —
//! driving parity work, recording where parity is **not expected** (a leaf-accelerator fold), and
//! flagging where parity is **not yet achieved** (a real gap with a convergence plan).
//!
//! Two pieces, deliberately split so the classification cannot lie:
//!
//! * [`parity`] — an **exhaustive** classifier over [`svm_ir::Inst`]/[`svm_ir::Terminator`]. No
//!   wildcard arm, exactly like [`svm_ir::Inst::effects`]: a new op fails to compile until it is
//!   classified for all four backends. This is the anti-drift forcing function.
//! * [`catalog`] — the full **mnemonic** fan-out (every `i32.add`, `f64.sqrt`, `i8x16.add`, …),
//!   built from `svm-ir`'s own sub-op vocabulary so the row set tracks the IR.
//!
//! The **conformance test** (`tests/conformance.rs`) is the honesty pin: it builds a minimal module
//! for each catalogued op and checks the real backends' `compile` results against this manifest, so a
//! `Full`/`Declines` here that the backend contradicts is a hard test failure — not silent drift
//! (INVARIANTS.md #9: "one shared predicate, one definition", differential tests gate every backend
//! feature). The human-readable [`render_markdown`] view (`OPS_PARITY.md`) is regenerated from the
//! same data by the `svm-parity-gen` bin and kept fresh by a golden test.

use svm_ir::*;

/// The four execution strategies (DESIGN.md §3), in matrix-column order. The names match the
/// crate/short-id vocabulary the owner uses (`svm-tree-walk`, `svm-bytecode`, `svm-jit`,
/// `svm-wasm-jit`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Backend {
    /// `svm-interp::run` — walks the SSA IR directly. The **oracle**; runs every op.
    TreeWalk,
    /// `svm-interp::bytecode` — operand-resolved threaded bytecode. Held bit-exact; runs every op.
    Bytecode,
    /// `svm-jit` — the Cranelift native speed path. Fail-closed: folds its non-subset to the oracle.
    Cranelift,
    /// `svm-wasm-jit` — emits wasm for a host engine's optimizing tiers. A **leaf accelerator**
    /// (DESIGN.md §3): never runs concurrency / cap-call / fiber ops itself.
    WasmJit,
}

impl Backend {
    pub const ALL: [Backend; 4] = [
        Backend::TreeWalk,
        Backend::Bytecode,
        Backend::Cranelift,
        Backend::WasmJit,
    ];

    /// The owner-facing short id / crate name — the matrix column header.
    pub fn short(self) -> &'static str {
        match self {
            Backend::TreeWalk => "svm-tree-walk",
            Backend::Bytecode => "svm-bytecode",
            Backend::Cranelift => "svm-jit",
            Backend::WasmJit => "svm-wasm-jit",
        }
    }
}

/// A backend's parity status for one op — the four buckets the owner asked the matrix to
/// distinguish.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Status {
    /// Runs the op with **oracle-identical** observable behavior — modulo the deliberately unpinned
    /// cross-backend deltas (float-NaN bit patterns on the JITs, backend-local handle/funcref index
    /// numbering; DESIGN.md §3/§3a). Parity **achieved**.
    Full,
    /// **Folds to the oracle by design** — parity is *not expected*. The op runs correctly (some
    /// faithful engine runs it), just never on *this* backend: a wasm-JIT concurrency/cap/fiber op
    /// (leaf accelerator), or a lowering that has no counterpart on the target (scalar `fma` has no
    /// core-wasm opcode).
    Declines,
    /// **Not yet lowered** — a real gap this backend *could* close but hasn't. Parity *not yet
    /// achieved*; carries a convergence note (an ISSUES.md id where one is tracked).
    NotYet,
    /// **Target-conditional** — `Full` where a build/target cfg holds, `Declines` elsewhere (the
    /// Cranelift fiber/thread/`setjmp` ops need the x86-64-unix stack-switch substrate; the wasm-JIT
    /// is absent on native and the Cranelift JIT on the browser). The note names the condition.
    Conditional,
}

impl Status {
    /// A compact glyph for the human-readable matrix.
    pub fn glyph(self) -> &'static str {
        match self {
            Status::Full => "✅",
            Status::Declines => "⛔",
            Status::NotYet => "🚧",
            Status::Conditional => "🔶",
        }
    }
}

/// One backend's verdict for one op: a [`Status`] plus a short human note (reason / issue id /
/// condition). The note is `""` for a plain `Full`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Cell {
    pub status: Status,
    pub note: &'static str,
}

const fn cell(status: Status, note: &'static str) -> Cell {
    Cell { status, note }
}

/// `Full`, no note — the common case.
const F: Cell = cell(Status::Full, "");

/// Both interpreters run every op (INVARIANTS.md #9: the oracle never declines, the bytecode engine
/// is held bit-exact against it). So a manifest row is really "what do the two JITs do?" — this
/// helper fills the two interpreter columns and takes the two JIT cells.
const fn row(cranelift: Cell, wasm_jit: Cell) -> [Cell; 4] {
    [F, F, cranelift, wasm_jit]
}

// ---- shared notes (kept as consts so identical reasons read identically across the matrix) --------

const LEAF: &str = "leaf accelerator: folds to the bytecode interp underneath (DESIGN §3)";
const HOST_OP: &str = "host cap/handle op — serviced by the oracle, not emitted";
const FIBER_RT: &str = "Full on x86-64-unix (fiber_rt); Declines to the interp elsewhere";
const SETJMP_RT: &str = "Full on x86-64-unix (setjmp_rt); Declines to the interp elsewhere";
const NO_WASM_OP: &str = "no core-wasm opcode (relaxed-SIMD / scalar-fma only)";

/// **The manifest.** Classify one instruction across the four backends. Exhaustive by design — a new
/// [`Inst`] variant must be classified here (and rendered by [`catalog`]) before this crate compiles.
///
/// Ground truth for the two JIT columns: the Cranelift support gate (`svm_jit`'s `ensure_supported`)
/// and the wasm-JIT subset (`svm_wasm_jit`'s `block_value_types` typing pass). The conformance test
/// pins this against what those backends actually compile.
pub fn parity(inst: &Inst) -> [Cell; 4] {
    match inst {
        // ---- Pure scalar value ops: both JITs lower them natively. -----------------------------
        Inst::ConstI32(_)
        | Inst::ConstI64(_)
        | Inst::ConstF32(_)
        | Inst::ConstF64(_)
        | Inst::IntBin { .. }
        | Inst::IntCmp { .. }
        | Inst::Eqz { .. }
        | Inst::Convert { .. }
        | Inst::Select { .. }
        | Inst::FBin { .. }
        | Inst::FUn { .. }
        | Inst::FCmp { .. }
        | Inst::FToISat { .. }
        | Inst::FToITrap { .. }
        | Inst::IToFConv { .. }
        | Inst::Cast { .. }
        | Inst::RefFunc { .. } => row(F, F),

        // Unary integer ops. wasm has `extend8_s`/`extend16_s` (both widths) and `i64.extend32_s`,
        // but **not** `i32.extend32_s` (it is the identity, so wasm defines no opcode) — the one
        // sub-op split in this family.
        Inst::IntUn {
            ty: IntTy::I32,
            op: IntUnOp::Extend32S,
            ..
        } => row(
            F,
            cell(
                Status::Declines,
                "i32.extend32_s is the identity; no wasm opcode",
            ),
        ),
        Inst::IntUn { .. } => row(F, F),

        // Scalar fused multiply-add: Cranelift has `fma`; core wasm has no scalar FMA opcode.
        Inst::Fma { .. } => row(F, cell(Status::Declines, NO_WASM_OP)),

        // Pointer provenance ops (off-CHERI plain i64): both JITs lower them — Cranelift natively,
        // the wasm-JIT as a wrapping `i64.add` / identity forward.
        Inst::PtrAdd { .. } | Inst::PtrCast { .. } => row(F, F),

        // ---- Guest linear memory: both JITs emit confined accesses / bulk memory. ---------------
        Inst::Load { .. }
        | Inst::Store { .. }
        | Inst::MemCopy { .. }
        | Inst::MemMove { .. }
        | Inst::MemFill { .. } => row(F, F),

        // ---- §12 atomics. Cranelift emits hardware atomics; the wasm-JIT lowers them to the
        // single-threaded load/rmw/store sequence (gated module-wide to concurrency-free guests). --
        Inst::AtomicLoad { .. }
        | Inst::AtomicStore { .. }
        | Inst::AtomicRmw { .. }
        | Inst::AtomicCmpxchg { .. } => row(
            F,
            cell(
                Status::Full,
                "single-threaded lowering (concurrency-free module)",
            ),
        ),
        // Standalone fence: both JITs handle it — Cranelift emits it; the wasm-JIT (single-threaded)
        // treats it as an observable no-op.
        Inst::AtomicFence { .. } => row(F, F),

        // ---- Calls. Direct/indirect calls lower on both JITs. -----------------------------------
        Inst::Call { .. } | Inst::CallIndirect { .. } => row(F, F),
        // Host capability calls: Cranelift emits a cap-call thunk into the host; the wasm-JIT is a
        // leaf accelerator and folds every cap op to the interpreter underneath.
        Inst::CapCall { .. }
        | Inst::CallImport { .. }
        | Inst::CallImportDyn { .. }
        | Inst::CallSym { .. }
        | Inst::ExportHandle { .. }
        | Inst::ImportAttach { .. } => row(F, cell(Status::Declines, LEAF)),

        // ---- §7 capability reflection (`cap.self.*`): Cranelift services them via a host thunk;
        // the wasm-JIT leaf folds them to the interpreter. ---------------------------------------
        Inst::CapSelfCount
        | Inst::CapSelfGet { .. }
        | Inst::CapSelfResolve { .. }
        | Inst::CapSelfLabel { .. }
        | Inst::CapSelfTypeId { .. }
        | Inst::CapSelfCovers { .. }
        | Inst::CapSelfAttest => row(F, cell(Status::Declines, HOST_OP)),

        // Per-vCPU TLS register + durable shadow base: baked thunks over a thread-local, supported on
        // every Cranelift target; the wasm-JIT leaf folds them.
        Inst::VcpuTlsGet | Inst::VcpuTlsSet { .. } | Inst::DurableShadowBase => {
            row(F, cell(Status::Declines, LEAF))
        }

        // ---- §12 fibers / threads / futex: Cranelift lowers them to host-runtime calls only where
        // the stack-switch substrate exists; the wasm-JIT (leaf) never runs concurrency itself. ---
        Inst::ContNew { .. }
        | Inst::ContResume { .. }
        | Inst::Suspend { .. }
        | Inst::ThreadSpawn { .. }
        | Inst::ThreadJoin { .. }
        | Inst::MemoryWait { .. }
        | Inst::MemoryNotify { .. }
        | Inst::GcRoots { .. } => row(
            cell(Status::Conditional, FIBER_RT),
            cell(Status::Declines, LEAF),
        ),
        // setjmp/longjmp: Cranelift on the setjmp_rt targets; the wasm-JIT leaf folds them.
        Inst::SetJmp { .. } | Inst::LongJmp { .. } => row(
            cell(Status::Conditional, SETJMP_RT),
            cell(Status::Declines, LEAF),
        ),

        // ---- §17 SIMD (D58). The core lane ops lower on both JITs. -------------------------------
        Inst::ConstV128(_)
        | Inst::V128Load { .. }
        | Inst::V128Store { .. }
        | Inst::Splat { .. }
        | Inst::ExtractLane { .. }
        | Inst::ReplaceLane { .. }
        | Inst::VFloatCmp { .. }
        | Inst::VShift { .. }
        | Inst::VIntUn { .. }
        | Inst::VSatBin { .. }
        | Inst::VWiden { .. }
        | Inst::VNarrow { .. }
        | Inst::VConvert { .. }
        | Inst::VPMinMax { .. }
        | Inst::VPopcnt { .. }
        | Inst::VAvgr { .. }
        | Inst::VDot { .. }
        | Inst::VExtMul { .. }
        | Inst::VExtAddPairwise { .. }
        | Inst::VQ15MulrSat { .. }
        | Inst::VAnyTrue { .. }
        | Inst::VAllTrue { .. }
        | Inst::VBitmask { .. }
        | Inst::VFloatBin { .. }
        | Inst::VFloatUn { .. }
        | Inst::VBitBin { .. }
        | Inst::VNot { .. }
        | Inst::Bitselect { .. }
        | Inst::Shuffle { .. }
        | Inst::Swizzle { .. }
        | Inst::SimdWidthBytes => row(F, F),

        // `i64x2` min/max has no single-instruction lowering on the target ISAs, so Cranelift can't
        // legalize it; wasm has no `i64x2` min/max op at all (so it never appears there — but as an
        // `Inst` it's outside the wasm subset just the same).
        Inst::VIntBin { shape, op, .. }
            if matches!(
                (*shape, *op),
                (
                    VShape::I64x2,
                    VIntBinOp::MinS | VIntBinOp::MinU | VIntBinOp::MaxS | VIntBinOp::MaxU
                )
            ) =>
        {
            row(
                cell(
                    Status::NotYet,
                    "i64x2 min/max: no single-instr ISA lowering",
                ),
                cell(Status::Declines, "no i64x2 min/max op in wasm"),
            )
        }
        // `i8x16.mul` has no wasm opcode (wasm omits byte-lane multiply); Cranelift synthesizes it
        // (widen → `i16x8` multiply → low-byte pack).
        Inst::VIntBin {
            shape: VShape::I8x16,
            op: VIntBinOp::Mul,
            ..
        } => row(F, cell(Status::Declines, "no i8x16.mul opcode in wasm")),
        Inst::VIntBin { .. } => row(F, F),

        // Lane integer compares. wasm omits the **unsigned `i64x2`** compares (`lt_u`/`gt_u`/`le_u`/
        // `ge_u`); Cranelift legalizes every shape.
        Inst::VIntCmp {
            shape: VShape::I64x2,
            op: VICmpOp::LtU | VICmpOp::GtU | VICmpOp::LeU | VICmpOp::GeU,
            ..
        } => row(
            F,
            cell(Status::Declines, "no unsigned i64x2 compare in wasm"),
        ),
        Inst::VIntCmp { .. } => row(F, F),

        // The two **relaxed-SIMD** ops (fused multiply-add, i8 dot): Cranelift picks the fused
        // behavior; core wasm has no opcode for either.
        Inst::VFma { .. } | Inst::VDotI8 { .. } => row(F, cell(Status::Declines, NO_WASM_OP)),
    }
}

/// Classify a [`Terminator`]. Every terminator lowers on all four backends today (irreducible
/// control flow is native; tail calls are lowered on both JITs), but the match stays exhaustive so a
/// new terminator forces a decision.
pub fn parity_term(term: &Terminator) -> [Cell; 4] {
    match term {
        Terminator::Br { .. }
        | Terminator::BrIf { .. }
        | Terminator::BrTable { .. }
        | Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => row(F, F),
    }
}

/// Whether `backend` **runs `inst` itself** (`Full`, or target-conditional where its cfg holds) as
/// opposed to folding it to the oracle (`Declines`/`NotYet`). The single programmatic source of a
/// backend's supported-op set — the differential-test corpora consult it so their op selection can't
/// drift from the matrix (INVARIANTS.md #9: one shared predicate, one definition). `Conditional`
/// counts as supported because the harnesses that generate those ops run on the target where the
/// substrate exists.
pub fn supports(backend: Backend, inst: &Inst) -> bool {
    matches!(
        parity(inst)[backend as usize].status,
        Status::Full | Status::Conditional
    )
}

/// [`supports`] for a block terminator.
pub fn supports_term(backend: Backend, term: &Terminator) -> bool {
    matches!(
        parity_term(term)[backend as usize].status,
        Status::Full | Status::Conditional
    )
}

mod catalog;
pub use catalog::{catalog, Focus, Op};

mod render;
pub use render::render_markdown;
