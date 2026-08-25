//! The verifier — security-critical TCB (`DESIGN.md` §2a, invariants I2/I3/I4;
//! §3b "verifier validity rules").
//!
//! A single linear forward pass, O(module size), no dominance analysis and no
//! fixups (block parameters make cross-block dataflow explicit). For each block we
//! seed a local type vector with the block's declared parameter types, walk the
//! instructions (checking each operand is defined *earlier* and exactly the right
//! type, then appending the result type), and finally check the terminator's branch
//! arguments against each target block's declared parameter types.
//!
//! Result types are computed here from opcode + operand types (§3a "inferred result
//! types"); for the one polymorphic op (`select`) the result is the operand type.
//!
//! **Fail-closed:** any violation returns `Err`; the verifier never panics on any
//! input (that property is fuzzed — see the `temen` crate). A module that verifies is
//! the precondition for the escape-freedom contract (§2a); soundness of *this code*
//! is the separate hard problem (§18).
#![forbid(unsafe_code)]
#![cfg_attr(not(test), no_std)]

extern crate alloc;
use alloc::vec::Vec;

use temen_ir::{Block, BlockIdx, Func, Inst, Module, Terminator, VShape, ValIdx, ValType};

/// Why verification rejected a module. Carries enough location to debug, never
/// enough to be load-bearing for safety (the boolean accept/reject is the contract).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum VerifyError {
    /// Entry block parameter types must equal the function signature's parameters.
    EntryParamsMismatch { func: u32 },
    /// A branch/return references a block index that does not exist.
    BlockOutOfRange { func: u32, block: u32, target: u32 },
    /// An operand references a value index not yet defined in this block.
    ValueOutOfRange {
        func: u32,
        block: u32,
        value: ValIdx,
        defined: u32,
    },
    /// An operand had the wrong type for its opcode.
    TypeMismatch {
        func: u32,
        block: u32,
        expected: ValType,
        found: ValType,
    },
    /// Branch argument count did not match the target block's parameter count.
    ArgCountMismatch {
        func: u32,
        block: u32,
        target: u32,
        expected: usize,
        found: usize,
    },
    /// Return value count did not match the function's result count.
    ResultCountMismatch {
        func: u32,
        block: u32,
        expected: usize,
        found: usize,
    },
    /// A load/store appeared but the module declares no linear memory.
    MemoryNotDeclared { func: u32, block: u32 },
    /// The declared window size (`1 << size_log2`) is not representable.
    MemorySizeTooLarge { size_log2: u8 },
    /// A `data` segment was declared but the module has no linear memory to place it in.
    DataWithoutMemory { seg: u32 },
    /// A `data` segment's `[offset, offset+len)` does not fit within the declared window.
    DataOutOfWindow { seg: u32 },
    /// A `data.ptr` relocation survived into a would-be-runnable module. Unlike the code→data
    /// link forms (which trap at execution), a data-image pointer has no execution site, so its
    /// placeholder bytes would be read unpatched — fail-closed here. `link` clears these; a
    /// module still carrying one was never linked.
    UnlinkedDataPtr { at: u64 },
    /// A `data.funcref` relocation survived into a would-be-runnable module — the funcref twin of
    /// [`UnlinkedDataPtr`](VerifyError::UnlinkedDataPtr). Its placeholder bytes (a would-be function
    /// index) would be loaded and `call.dyn`'d unpatched, so it is fail-closed here. `link`
    /// resolves and clears these; a survivor was never linked.
    UnlinkedDataFuncref { at: u64 },
    /// A `call` referenced a function index that does not exist.
    CallFuncOutOfRange { func: u32, block: u32, callee: u32 },
    /// A `call`'s argument count did not match the callee's parameter count.
    CallArgCountMismatch {
        func: u32,
        block: u32,
        expected: usize,
        found: usize,
    },
    /// A `thread.spawn` named a function whose signature is not the fixed thread entry type
    /// `(i64 sp, i64 arg) -> i64` (§12).
    ThreadEntrySignature { func: u32, block: u32, callee: u32 },
    /// A `<shape>.extract_lane`/`replace_lane` named a lane index `>= shape.lanes()`, or an
    /// `i8x16.shuffle` byte index `>= 32` (§17). Lane indices are immediates, so this is a
    /// structural check.
    BadSimdLane { func: u32, block: u32 },
    /// A lane-wise op was given a shape of the wrong category — an integer op on a float shape
    /// or a float op on an integer shape (§17).
    BadSimdShape { func: u32, block: u32 },
    /// A [`Inst::CallImport`] referenced an import index at or past the end of the module's
    /// [`Module::imports`] manifest (§7 / IMPORTS.md phase 1). A `call.import` is executable when
    /// its index names a declared import; one that names nothing is fail-closed. (This variant also
    /// covers the pre-manifest legacy shape — a `CallImport` in a module with an empty import
    /// section is by definition out of range.)
    UnresolvedImport { func: u32, block: u32, import: u32 },
    /// A [`Inst::CallImport`]'s self-describing `sig` disagreed with the declared signature of the
    /// import it references (IMPORTS.md phase 1). The manifest is the canonical interface; a call
    /// site asserting a different one is fail-closed (the §7 structural signature check).
    ImportSigMismatch { func: u32, block: u32, import: u32 },
    /// A call variant's interned signature index (FuncType interning, #922) is out of range or
    /// does not name a [`temen_ir::TypeEntry::Func`]. Fail-closed: the index feeds arg/result
    /// typing and SSA-slot layout ([`temen_ir::Inst::result_count`]), so a bad index is rejected.
    CallSigInvalid { func: u32, block: u32, sig: u32 },
    /// Two [`Module::imports`] entries share a name — imports must be uniquely resolvable by the
    /// host's instantiation policy (IMPORTS.md phase 1), mirroring [`VerifyError::DuplicateExport`].
    DuplicateImport { import: u32 },
    /// An [`Inst::ImportAttach`] targeted an import that is not declared
    /// [`temen_ir::ImportMode::Rebindable`] (IMPORTS.md phase 2). `required` bindings are
    /// immutable-per-instance by construction — that immutability is what makes them always legal
    /// to devirtualize — so attaching to one is fail-closed here, statically.
    AttachNotRebindable { func: u32, block: u32, import: u32 },
    /// A `gc.roots` carried a **constant** payload mask that clears more than the top byte (its
    /// low 56 bits are not all-ones). Such a mask could fold a canonical host pointer down into the
    /// guest window and leak host-address bits past the range filter (GC.md §3, §6). Only
    /// top-byte-strip masks (`mask | 0xFF00_0000_0000_0000 == !0`) are allowed; the runtime also
    /// rejects a non-constant mask that violates this at execution.
    GcRootsMaskUnsafe { func: u32, block: u32, mask: u64 },
    /// A named [`Module::exports`] entry points at a funcidx past the end of `funcs`.
    ExportFuncOutOfRange { export: u32, func: u32 },
    /// Two [`Module::exports`] entries share a name — exports must be uniquely addressable.
    DuplicateExport { export: u32 },
    /// An [`Module::impl_exports`] offer's op list names a funcidx past the end of `funcs`.
    ImplExportFuncOutOfRange { export: u32, op: u32, func: u32 },
    /// An [`Module::impl_exports`] offer with an empty op list — an interface with no operations
    /// can never be wired; fail-closed at verify rather than at wiring.
    ImplExportEmpty { export: u32 },
    /// Two export entries share a name. Function exports and impl exports (interface offers,
    /// IMPORTS.md §3.2) are one namespace: the host addresses both by name.
    DuplicateImplExport { export: u32 },
    /// An [`Module::impl_exports`] offer's `iface` does not name a well-formed
    /// [`temen_ir::TypeDef::Interface`] in [`Module::types`] (out of range, a `Func` entry, or
    /// an interface whose elements don't all name `Func` entries).
    ImplExportIfaceOutOfRange { export: u32, iface: u32 },
    /// An [`Module::impl_exports`] offer does not implement its declared interface: the op
    /// count differs, or op `op`'s function type differs from the interface's op-`op`
    /// signature (structural, exact — IMPORTS.md OQ3/v6).
    ImplExportIfaceMismatch { export: u32, op: u32 },
    /// A [`Module::imports`] entry's shape reference (§3.5, v7) does not name a well-formed
    /// type-section entry of its declared kind (`func` → a `Func` entry; `interface` → an
    /// `Interface` entry whose elements all resolve to `Func` entries).
    ImportShapeInvalid { import: u32 },
    /// A [`Inst::CallImport`]'s `op` immediate is out of range for the import's declared
    /// shape (a flat `func` import has exactly op 0; a grouped import has its interface's
    /// op count).
    ImportOpOutOfRange {
        func: u32,
        block: u32,
        import: u32,
        op: u32,
    },
    /// A §3.5 instruction's type-section reference (`call.import.dyn`, `self.type_id`,
    /// `self.covers`) does not name a well-formed interface entry, or its `op` is out of
    /// range / its self-describing `sig` differs from the type-section resolution.
    DynIfaceInvalid { func: u32, block: u32, ty: u32 },
    /// An [`Inst::ExportHandle`]'s index is past the end of [`Module::impl_exports`].
    ExportHandleOutOfRange { func: u32, block: u32, export: u32 },
}

/// Verify an entire module. `Ok(())` is the only "accept".
pub fn verify_module(m: &Module) -> Result<(), VerifyError> {
    // A declared window must have a representable size (`1 << size_log2`, with the
    // mask `size - 1` well-defined). `size_log2 == 63` is the largest window.
    if let Some(mem) = &m.memory {
        if mem.size_log2 >= 64 {
            return Err(VerifyError::MemorySizeTooLarge {
                size_log2: mem.size_log2,
            });
        }
    }
    // Data segments must fit within the declared window `[0, size)` (§3a / D40). The runtime
    // copies them in (and protects `readonly` ones) at instantiation, so an out-of-window or
    // memory-less segment is rejected here, fail-closed.
    for (i, d) in m.data.iter().enumerate() {
        let seg = i as u32;
        let Some(mem) = &m.memory else {
            return Err(VerifyError::DataWithoutMemory { seg });
        };
        // Reject if `offset + len` overflows (`None`) or exceeds the window. Written as an explicit
        // match (not `is_none_or`, stabilized in 1.82) so this crate also compiles on the on-ramp's
        // pinned `rustc 1.81` (LLVM-18) toolchain — see DESIGN.md §20c.
        let end = d.offset.checked_add(d.bytes.len() as u64);
        let out_of_window = match end {
            Some(e) => e > mem.size(),
            None => true,
        };
        if out_of_window {
            return Err(VerifyError::DataOutOfWindow { seg });
        }
    }
    // A runnable module carries no `data.ptr` relocations — `link` resolves them into the data
    // image and clears the list. A survivor means the module was never linked; its placeholder
    // bytes would be read unpatched, so it is fail-closed here (a data pointer has no execution
    // site to trap at, unlike the `data.sym`/`data.self` instruction forms).
    if let Some(p) = m.data_ptrs.first() {
        return Err(VerifyError::UnlinkedDataPtr { at: p.at });
    }
    // Likewise a runnable module carries no `data.funcref` relocations — `link` resolves each into a
    // baked function index and clears the list. A survivor was never linked; its placeholder bytes
    // would be loaded and dispatched unpatched.
    if let Some(r) = m.data_funcrefs.first() {
        return Err(VerifyError::UnlinkedDataFuncref { at: r.at });
    }
    // Validate the import manifest (§7 / IMPORTS.md phase 1) in one pass, before `verify_func`
    // checks any call against it. Two properties per entry:
    //   - names must be uniquely resolvable — the instantiation policy binds by name, so an
    //     ambiguous manifest is fail-closed here, mirroring the export-name check below;
    //   - §3.5 shape: every import must reference a well-formed type-section entry of its declared
    //     kind — a `func` import a `Func` entry, an `interface` import an `Interface` entry whose
    //     elements all resolve to `Func` entries. Fail-closed before any binding act.
    for (ii, imp) in m.imports.iter().enumerate() {
        if m.imports[..ii].iter().any(|o| o.name == imp.name) {
            return Err(VerifyError::DuplicateImport { import: ii as u32 });
        }
        let shape_ok = match imp.shape {
            temen_ir::ImportShape::Func(t) => {
                matches!(m.types.get(t as usize), Some(temen_ir::TypeEntry::Func(_)))
            }
            temen_ir::ImportShape::Interface(t) => m.interface_ops(t).is_some(),
        };
        if !shape_ok {
            return Err(VerifyError::ImportShapeInvalid { import: ii as u32 });
        }
    }
    let has_memory = m.memory.is_some();
    for (fi, f) in m.funcs.iter().enumerate() {
        verify_func(
            fi as u32,
            f,
            &m.funcs,
            &m.imports,
            &m.types,
            m.impl_exports.len(),
            has_memory,
        )?;
    }
    // Named exports must point at a real function and be uniquely addressable (backends ignore the
    // table, but the host resolves `call("name")` through it, so a dangling/ambiguous name is
    // fail-closed here).
    for (ei, e) in m.exports.iter().enumerate() {
        if e.func as usize >= m.funcs.len() {
            return Err(VerifyError::ExportFuncOutOfRange {
                export: ei as u32,
                func: e.func,
            });
        }
        if m.exports[..ei].iter().any(|o| o.name == e.name) {
            return Err(VerifyError::DuplicateExport { export: ei as u32 });
        }
    }
    // Interface offers (IMPORTS.md §3.2): every per-op funcidx must be a real function (op `i`'s
    // signature IS that function's declared type, so a dangling index would leave the interface
    // unspecifiable), the op list must be non-empty, and names must be unique across *both*
    // export namespaces (the host wires offers by name).
    for (ei, e) in m.impl_exports.iter().enumerate() {
        if e.ops.is_empty() {
            return Err(VerifyError::ImplExportEmpty { export: ei as u32 });
        }
        for (oi, &f) in e.ops.iter().enumerate() {
            if f as usize >= m.funcs.len() {
                return Err(VerifyError::ImplExportFuncOutOfRange {
                    export: ei as u32,
                    op: oi as u32,
                    func: f,
                });
            }
        }
        // v6 (OQ3): the offer must implement its **declared** interface exactly — same op
        // count, and op `i`'s function type equal to the interface's op-`i` signature
        // (resolved through the type section's one index space). Makes "implemented the
        // wrong interface" a verify error, not a wiring surprise.
        let Some(iface) = m.interface_ops(e.interface) else {
            return Err(VerifyError::ImplExportIfaceOutOfRange {
                export: ei as u32,
                iface: e.interface,
            });
        };
        if e.ops.len() != iface.len() {
            return Err(VerifyError::ImplExportIfaceMismatch {
                export: ei as u32,
                op: e.ops.len().min(iface.len()) as u32,
            });
        }
        for (oi, (&f, want)) in e.ops.iter().zip(&iface).enumerate() {
            let ft = &m.funcs[f as usize];
            if ft.params != want.params || ft.results != want.results {
                return Err(VerifyError::ImplExportIfaceMismatch {
                    export: ei as u32,
                    op: oi as u32,
                });
            }
        }
        if m.impl_exports[..ei].iter().any(|o| o.name == e.name)
            || m.exports.iter().any(|o| o.name == e.name)
        {
            return Err(VerifyError::DuplicateImplExport { export: ei as u32 });
        }
    }
    Ok(())
}

/// Whether `m`'s capability egress is **manifest-complete** (IMPORTS.md §2.2): the module contains
/// no dynamic-mode capability dispatch (`call.cap` — dispatch on a runtime handle value), so the
/// import manifest is the *complete* list of interfaces the module can ever drive. A statically
/// checkable per-module property — tooling can report it, and a host policy may require it for
/// high-assurance slots. Reflection (`self.*`) is authority-neutral and does not affect the
/// bit: discovering a handle confers nothing without a `call.cap` to drive it, which the bit
/// catches. That exemption extends to a `call.cap` whose `type_id` **immediate** is the reserved
/// [`temen_ir::CAP_SELF_TYPE_ID`] — the self namespace's dispatch-form ops (e.g. §3.1
/// `provenance`) are the same authority-neutral reflection, statically identifiable from the
/// immediate, so querying them does not cost the completeness bit. Modules needing open-world
/// discovery (shells, plugin hosts) legitimately report `false` — their egress is bounded by
/// grants instead (the ocap bound).
pub fn manifest_complete(m: &Module) -> bool {
    m.funcs.iter().all(|f| {
        f.blocks.iter().all(|b| {
            !b.insts.iter().any(|i| {
                matches!(i, Inst::CapCall { type_id, .. } if *type_id != temen_ir::CAP_SELF_TYPE_ID)
                    // §3.5: dynamic-mode dispatch by type-section reference is dispatch on a
                    // runtime handle value — it costs the bit exactly like `call.cap`.
                    || matches!(i, Inst::CallImportDyn { .. })
            })
        })
    })
}

/// Resolve import `import`'s op-`op` signature through the type section (the §3.5 view):
/// a flat `func` import has exactly op 0; a grouped import resolves its interface element.
fn import_op_sig<'a>(
    imports: &[temen_ir::Import],
    ts: &'a [temen_ir::TypeEntry],
    import: u32,
    op: u32,
) -> Option<&'a temen_ir::FuncType> {
    match imports.get(import as usize)?.shape {
        temen_ir::ImportShape::Func(t) => match (op, ts.get(t as usize)?) {
            (0, temen_ir::TypeEntry::Func(ft)) => Some(ft),
            _ => None,
        },
        temen_ir::ImportShape::Interface(t) => iface_op_sig(ts, t, op),
    }
}

/// Resolve an interned signature index (FuncType interning, #922) to its `FuncType`: the index
/// must be in range and name a `TypeEntry::Func`. `None` is a fail-closed verify error.
fn func_sig(ts: &[temen_ir::TypeEntry], t: u32) -> Option<&temen_ir::FuncType> {
    match ts.get(t as usize)? {
        temen_ir::TypeEntry::Func(ft) => Some(ft),
        temen_ir::TypeEntry::Interface(_) => None,
    }
}

/// Resolve interface entry `t`'s op-`op` signature, or `None` if `t` is not a well-formed
/// interface reference or `op` is out of range.
fn iface_op_sig(ts: &[temen_ir::TypeEntry], t: u32, op: u32) -> Option<&temen_ir::FuncType> {
    match ts.get(t as usize)? {
        temen_ir::TypeEntry::Interface(elems) => {
            match ts.get(elems.get(op as usize)?.ty as usize)? {
                temen_ir::TypeEntry::Func(ft) => Some(ft),
                temen_ir::TypeEntry::Interface(_) => None,
            }
        }
        temen_ir::TypeEntry::Func(_) => None,
    }
}

/// Whether `t` names a well-formed interface entry (every element a `Func` reference).
fn iface_well_formed(ts: &[temen_ir::TypeEntry], t: u32) -> bool {
    match ts.get(t as usize) {
        Some(temen_ir::TypeEntry::Interface(elems)) => elems
            .iter()
            .all(|e| matches!(ts.get(e.ty as usize), Some(temen_ir::TypeEntry::Func(_)))),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn verify_func(
    fi: u32,
    f: &Func,
    funcs: &[Func],
    imports: &[temen_ir::Import],
    type_section: &[temen_ir::TypeEntry],
    n_impl_exports: usize,
    has_memory: bool,
) -> Result<(), VerifyError> {
    // Per function: the entry block's parameters are the function's parameters.
    match f.blocks.first() {
        Some(entry) if entry.params == f.params => {}
        Some(_) => return Err(VerifyError::EntryParamsMismatch { func: fi }),
        // A function with no blocks cannot return; treat as ill-formed.
        None => return Err(VerifyError::EntryParamsMismatch { func: fi }),
    }

    let nblocks = f.blocks.len() as u32;
    // Per-function result arity, for `Inst::result_count` (used to trace a `gc.roots` constant mask
    // through the block's value numbering).
    let fn_results: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    for (bi, b) in f.blocks.iter().enumerate() {
        let bi = bi as u32;
        // Seed the local type vector with the block's declared parameter types.
        let mut types: Vec<ValType> = b.params.clone();

        for inst in &b.insts {
            // §GC security: a constant `gc.roots` payload mask may only clear the top byte — a
            // fold-down mask is rejected statically here (a non-constant mask is enforced at runtime
            // on both backends). This is the one check that needs *block* context (tracing the mask
            // operand to its defining `i64.const`), so it stays out of `type_inst`. `mask | 0xFF00..
            // == !0` ⇔ the low 56 bits are all 1.
            if let Inst::GcRoots { mask, .. } = inst {
                if let Some(m) = const_i64_in_block(b, &fn_results, type_section, *mask) {
                    if (m as u64) | 0xFF00_0000_0000_0000 != u64::MAX {
                        return Err(VerifyError::GcRootsMaskUnsafe {
                            func: fi,
                            block: bi,
                            mask: m as u64,
                        });
                    }
                }
            }

            // The single per-instruction dispatch (#895): check operands + append result types.
            let before = types.len();
            type_inst(
                fi,
                bi,
                inst,
                &types,
                has_memory,
                funcs,
                imports,
                type_section,
                n_impl_exports,
                true,
            )?
            .append_to(&mut types);
            // The appended arity is the value-numbering contract `const_i64_in_block` reads back
            // through `result_count`; a drift would mis-trace the `gc.roots` constant mask. One
            // dispatch now feeds both, so this asserts they agree (debug builds; the fuzz gates the
            // release contract).
            debug_assert_eq!(
                types.len() - before,
                inst.result_count(&fn_results, type_section),
                "type_inst arity disagrees with Inst::result_count"
            );
        }

        check_terminator(fi, bi, &b.term, &types, type_section, nblocks, f, funcs)?;
    }
    Ok(())
}

/// Per-block SSA value **types** for a function: each block's declared params followed by every
/// instruction's result type(s) — exactly the assignment [`verify_func`] performs while checking.
/// Indexing `func_value_types(..)[block][value_idx]` gives the type of block-local value
/// `value_idx`, which the interpreter's debugger uses to reconstruct a typed value from its
/// (untyped) storage slot. Assumes `f` is **verified**: it derives every value's type from the
/// *same* [`type_inst`] dispatch `verify_func` uses (run with `check = false`), so the two cannot
/// drift — the "keep in sync" hazard the `gc.roots` mask-trace once depended on is gone. On an
/// unverified function it degrades gracefully (a value whose type can't be derived is simply absent
/// from the vector) rather than erroring.
pub fn func_value_types(
    f: &Func,
    funcs: &[Func],
    type_section: &[temen_ir::TypeEntry],
    has_memory: bool,
) -> Vec<Vec<ValType>> {
    let fn_results: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    f.blocks
        .iter()
        .map(|b| block_value_types(b, funcs, &fn_results, type_section, has_memory))
        .collect()
}

fn block_value_types(
    b: &Block,
    funcs: &[Func],
    fn_results: &[usize],
    type_section: &[temen_ir::TypeEntry],
    has_memory: bool,
) -> Vec<ValType> {
    let _ = fn_results;
    let mut types: Vec<ValType> = b.params.clone();
    for inst in &b.insts {
        // The typing-only walk shares the one [`type_inst`] dispatch with `check = false`: operand
        // and structural checks short-circuit, so it derives result types from an unverified function
        // without rejecting it (`imports`/`n_impl_exports` are only read on the checked paths, so the
        // empties here are never consulted). An op whose type can't be derived contributes nothing.
        if let Ok(res) = type_inst(
            0,
            0,
            inst,
            &types,
            has_memory,
            funcs,
            &[],
            type_section,
            0,
            false,
        ) {
            res.append_to(&mut types);
        }
    }
    types
}

/// The result types one instruction appends to the running SSA type vector — the return of the
/// single [`type_inst`] dispatch. Alloc-free for the common 0/1/2-result cases; a call's variable
/// result row is a borrow of the callee/signature it resolved (`funcs`/`types` outlive the walk).
enum InstResults<'a> {
    /// No value (`store`, the bulk-memory ops, `longjmp`, `set`, a bare fence).
    None,
    /// One result — the vast majority of ops.
    One(ValType),
    /// `cont.resume`'s `(status: i32, value: i64)` pair.
    Two(ValType, ValType),
    /// A call's result row (`callee.results` / `types[sig].results`).
    Row(&'a [ValType]),
}

impl InstResults<'_> {
    /// Append these results to the running type vector, in order.
    fn append_to(&self, out: &mut Vec<ValType>) {
        match self {
            InstResults::None => {}
            InstResults::One(t) => out.push(*t),
            InstResults::Two(a, b) => {
                out.push(*a);
                out.push(*b);
            }
            InstResults::Row(r) => out.extend_from_slice(r),
        }
    }
}

/// A resolved signature's result row, or nothing when it didn't resolve — the typing-only
/// (`check = false`) shape for the call-import ops: [`block_value_types`] derives result types from
/// the interned `sig` when it resolves and simply omits the value otherwise (graceful degradation on
/// an unverified function), never rejecting.
fn row_or_none(ft: Option<&temen_ir::FuncType>) -> InstResults<'_> {
    match ft {
        Some(ft) => InstResults::Row(&ft.results),
        None => InstResults::None,
    }
}

/// Check one instruction's operands against the running type vector and return the
/// result type to append (`None` for `Store`). Operands must reference
/// strictly-earlier indices.
/// The constant `i64` an operand resolves to, if it's defined by an `i64.const` **earlier in this
/// block** (the only place a value can be defined in this block-param SSA — see `verify_func`).
/// Returns `None` for a block parameter or any non-constant definition. Mirrors the value
/// numbering of `verify_func`: params occupy `0..params.len()`, then each instruction owns its
/// `result_count` consecutive indices.
fn const_i64_in_block(
    b: &temen_ir::Block,
    fn_results: &[usize],
    type_section: &[temen_ir::TypeEntry],
    v: ValIdx,
) -> Option<i64> {
    let mut idx = b.params.len() as u32;
    for inst in &b.insts {
        let n = inst.result_count(fn_results, type_section) as u32;
        if v >= idx && v < idx + n {
            return match inst {
                Inst::ConstI64(c) => Some(*c),
                _ => None,
            };
        }
        idx += n;
    }
    None
}

/// The **one** per-instruction dispatch (#895): check an instruction's operands against the running
/// SSA type vector and return the result types it appends. Every opcode lives in exactly one arm —
/// the whole-module-dependent ops (`call`/`call.dyn`/`call.cap`/the import & iface calls,
/// `ref.func`, `thread.spawn`, `cont.resume`, the reflection ops) fold in here rather than being
/// checked in a parallel chain in [`verify_func`], so there is no ordering-sensitive `if let` cascade
/// and no `unreachable!()` arm. `check` selects the two callers: `verify_func` runs `check = true`
/// (operands + structural rules enforced); [`block_value_types`] runs `check = false`, where the
/// checks short-circuit and the same dispatch derives result types from an unverified function
/// (`Cx::check`). Operands must reference strictly-earlier indices. `Store` and the other no-result
/// ops return [`InstResults::None`].
#[allow(clippy::too_many_arguments)]
fn type_inst<'a>(
    fi: u32,
    bi: u32,
    inst: &Inst,
    types: &[ValType],
    has_memory: bool,
    funcs: &'a [Func],
    imports: &[temen_ir::Import],
    type_section: &'a [temen_ir::TypeEntry],
    n_impl_exports: usize,
    check: bool,
) -> Result<InstResults<'a>, VerifyError> {
    let cx = Cx {
        fi,
        bi,
        types,
        check,
    };
    // `Store` is the only instruction that yields no value; handle it up front so the
    // main match can produce a single result type.
    if let Inst::Store {
        op, addr, value, ..
    } = inst
    {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, op.info().1)?;
        return Ok(InstResults::None);
    }
    // Bulk-memory ops (from `llvm.memcpy`/`memmove`/`memset`) — no-result, whole-span confined.
    // `dst`/`src`/`len` are `i64`; `MemFill`'s `val` is the `i32` fill byte.
    if let Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*dst, ValType::I64)?;
        cx.expect(*src, ValType::I64)?;
        cx.expect(*len, ValType::I64)?;
        return Ok(InstResults::None);
    }
    if let Inst::MemFill { dst, val, len } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*dst, ValType::I64)?;
        cx.expect(*val, ValType::I32)?;
        cx.expect(*len, ValType::I64)?;
        return Ok(InstResults::None);
    }
    // §12 atomic store — the other no-result memory op.
    if let Inst::AtomicStore {
        ty, addr, value, ..
    } = inst
    {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, ty.val())?;
        return Ok(InstResults::None);
    }
    // §17 `v128.store` — the third no-result memory op (a 16-byte masked access).
    if let Inst::V128Store { addr, value, .. } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*addr, ValType::I64)?;
        cx.expect(*value, ValType::V128)?;
        return Ok(InstResults::None);
    }
    // `setjmp`/`longjmp` (the non-local jump): both touch the guest `jmp_buf` token in window memory,
    // so both require a declared window. `setjmp` takes an `i64` buffer address, yields `i32` (0 on the
    // direct call, the long-jump value on re-entry); `longjmp` takes the `i64` address + an `i32` value
    // and yields no result (a `noreturn` control op).
    if let Inst::SetJmp { buf } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*buf, ValType::I64)?;
        return Ok(InstResults::One(ValType::I32));
    }
    if let Inst::LongJmp { buf, val } = inst {
        if !has_memory {
            return Err(VerifyError::MemoryNotDeclared {
                func: fi,
                block: bi,
            });
        }
        cx.expect(*buf, ValType::I64)?;
        cx.expect(*val, ValType::I32)?;
        return Ok(InstResults::None);
    }
    let ty = match inst {
        Inst::ConstI32(_) => ValType::I32,
        Inst::ConstI64(_) => ValType::I64,
        // Link-form data addresses (the `call.sym` analogue for data): type as the `i64` address
        // they resolve to, so a pre-link *unit* type-checks. `link` rewrites them to `ConstI64`
        // before anything runs; if one ever survives into an executed module the backends fail
        // closed (they never reach a legitimate execution path).
        Inst::DataSym { .. } | Inst::DataSelf { .. } | Inst::DataTop => ValType::I64,
        // §7 executable named import (IMPORTS.md phase 1): the index must name a declared import
        // and the call's self-describing `sig` must equal the manifest's — the canonical-interface
        // check `call.cap` cannot have (its sig is self-asserted). No handle operand (v8): the slot
        // binding identifies the capability. Appends the interned sig's result row.
        Inst::CallImport {
            import,
            op,
            sig,
            args,
        } => {
            if check {
                if imports.get(*import as usize).is_none() {
                    return Err(VerifyError::UnresolvedImport {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                // §3.5: resolve the consumer-local op through the import's declared shape + the type
                // section (op range + exact signature), then require the interned `sig` to agree.
                let Some(want) = import_op_sig(imports, type_section, *import, *op) else {
                    return Err(VerifyError::ImportOpOutOfRange {
                        func: fi,
                        block: bi,
                        import: *import,
                        op: *op,
                    });
                };
                let Some(sig_ft) = func_sig(type_section, *sig) else {
                    return Err(VerifyError::CallSigInvalid {
                        func: fi,
                        block: bi,
                        sig: *sig,
                    });
                };
                if want != sig_ft {
                    return Err(VerifyError::ImportSigMismatch {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                cx.check_args(args, &sig_ft.params)?;
                return Ok(InstResults::Row(&sig_ft.results));
            }
            return Ok(row_or_none(func_sig(type_section, *sig)));
        }
        // §3.5 dynamic-mode dispatch by type-section reference: `ty` names a well-formed interface,
        // `op` in range, and the self-describing `sig` equals the resolution — the same
        // canonical-interface check as static mode, at load.
        Inst::CallImportDyn {
            ty,
            op,
            sig,
            handle,
            args,
        } => {
            if check {
                let Some(want) = iface_op_sig(type_section, *ty, *op) else {
                    return Err(VerifyError::DynIfaceInvalid {
                        func: fi,
                        block: bi,
                        ty: *ty,
                    });
                };
                let Some(sig_ft) = func_sig(type_section, *sig) else {
                    return Err(VerifyError::CallSigInvalid {
                        func: fi,
                        block: bi,
                        sig: *sig,
                    });
                };
                if want != sig_ft {
                    return Err(VerifyError::DynIfaceInvalid {
                        func: fi,
                        block: bi,
                        ty: *ty,
                    });
                }
                cx.expect(*handle, ValType::I32)?;
                cx.check_args(args, &sig_ft.params)?;
                return Ok(InstResults::Row(&sig_ft.results));
            }
            return Ok(row_or_none(func_sig(type_section, *sig)));
        }
        // §7/§22 symbolic call (v8): `call.sym` verifies exactly like a *flat* manifest import call
        // (op 0), plus its legacy handle operand (i32, ignored by manifest dispatch — live only to
        // the linker). Executable when the instance binds the name; the linker's rewrite target when
        // resolved first. One rule, both consumers.
        Inst::CallSym {
            import,
            sig,
            handle,
            args,
        } => {
            if check {
                if imports.get(*import as usize).is_none() {
                    return Err(VerifyError::UnresolvedImport {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                let Some(want) = import_op_sig(imports, type_section, *import, 0) else {
                    return Err(VerifyError::ImportOpOutOfRange {
                        func: fi,
                        block: bi,
                        import: *import,
                        op: 0,
                    });
                };
                let Some(sig_ft) = func_sig(type_section, *sig) else {
                    return Err(VerifyError::CallSigInvalid {
                        func: fi,
                        block: bi,
                        sig: *sig,
                    });
                };
                if want != sig_ft {
                    return Err(VerifyError::ImportSigMismatch {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                cx.expect(*handle, ValType::I32)?;
                cx.check_args(args, &sig_ft.params)?;
                return Ok(InstResults::Row(&sig_ft.results));
            }
            return Ok(row_or_none(func_sig(type_section, *sig)));
        }
        // Phase-2 `import.attach`: the index must name a declared **rebindable** import (attaching to
        // a `required` slot would break its immutability-per-instance); the handle is an ordinary
        // forgeable `i32` (validity is the runtime's §3c check at attach). Appends the `i32` status.
        Inst::ImportAttach { import, handle } => {
            if check {
                if imports.get(*import as usize).is_none() {
                    return Err(VerifyError::UnresolvedImport {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                if imports[*import as usize].mode != temen_ir::ImportMode::Rebindable {
                    return Err(VerifyError::AttachNotRebindable {
                        func: fi,
                        block: bi,
                        import: *import,
                    });
                }
                cx.expect(*handle, ValType::I32)?;
            }
            return Ok(InstResults::One(ValType::I32));
        }
        // §3.5 `export.handle`: the index must name a declared impl export; appends the reified `i32`.
        Inst::ExportHandle { export } => {
            if check && *export as usize >= n_impl_exports {
                return Err(VerifyError::ExportHandleOutOfRange {
                    func: fi,
                    block: bi,
                    export: *export,
                });
            }
            return Ok(InstResults::One(ValType::I32));
        }
        // §3.5 reflection: reference a well-formed interface entry of *this* module's type section;
        // `covers` also takes a forgeable `i32` handle. Each appends an `i32`.
        Inst::CapSelfTypeId { ty } => {
            if check && !iface_well_formed(type_section, *ty) {
                return Err(VerifyError::DynIfaceInvalid {
                    func: fi,
                    block: bi,
                    ty: *ty,
                });
            }
            return Ok(InstResults::One(ValType::I32));
        }
        Inst::CapSelfCovers { handle, ty } => {
            if check && !iface_well_formed(type_section, *ty) {
                return Err(VerifyError::DynIfaceInvalid {
                    func: fi,
                    block: bi,
                    ty: *ty,
                });
            }
            cx.expect(*handle, ValType::I32)?;
            return Ok(InstResults::One(ValType::I32));
        }
        // §12 per-vCPU TLS register: ambient, no memory/module dependency. `get` yields an i64;
        // `set` consumes an i64 and yields nothing (handled like `store`).
        Inst::VcpuTlsGet => ValType::I64,
        // Durable-runtime-internal: the current context's shadow region base (a window byte offset).
        Inst::DurableShadowBase => ValType::I64,
        Inst::VcpuTlsSet { val } => {
            cx.expect(*val, ValType::I64)?;
            return Ok(InstResults::None);
        }
        Inst::IntBin { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::IntCmp { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            ValType::I32
        }
        Inst::IntUn { ty, a, .. } => {
            cx.expect(*a, ty.val())?;
            ty.val()
        }
        Inst::Eqz { ty, a } => {
            cx.expect(*a, ty.val())?;
            ValType::I32
        }
        Inst::Convert { op, a } => {
            let (_, src, dst) = op.sig();
            cx.expect(*a, src)?;
            dst
        }
        Inst::Select { cond, a, b } => {
            cx.expect(*cond, ValType::I32)?;
            // Polymorphic: `a` defines the result type, `b` must match it.
            let t = cx.type_of(*a)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::ConstF32(_) => ValType::F32,
        Inst::ConstF64(_) => ValType::F64,
        Inst::FBin { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            t
        }
        Inst::FUn { ty, a, .. } => {
            cx.expect(*a, ty.val())?;
            ty.val()
        }
        // Scalar fused multiply-add: all three operands and the result are `ty`.
        Inst::Fma { ty, a, b, c } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            cx.expect(*c, t)?;
            t
        }
        Inst::FCmp { ty, a, b, .. } => {
            let t = ty.val();
            cx.expect(*a, t)?;
            cx.expect(*b, t)?;
            ValType::I32
        }
        Inst::FToISat { op, a } | Inst::FToITrap { op, a } => {
            let (from, to, _) = op.parts();
            cx.expect(*a, from.val())?;
            to.val()
        }
        Inst::IToFConv { op, a } => {
            let (from, to, _) = op.parts();
            cx.expect(*a, from.val())?;
            to.val()
        }
        Inst::Cast { op, a } => {
            let (_, src, dst) = op.sig();
            cx.expect(*a, src)?;
            dst
        }
        Inst::Load { op, addr, .. } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            op.info().1
        }
        Inst::AtomicLoad { ty, addr, .. } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            ty.val()
        }
        Inst::AtomicRmw {
            ty, addr, value, ..
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*value, ty.val())?;
            ty.val()
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            ..
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*expected, ty.val())?;
            cx.expect(*replacement, ty.val())?;
            ty.val()
        }
        // §12 fibers. `cont.new` takes an i32 funcref, yields an i64 handle (16-bit slot +
        // 48-bit generation); `suspend` takes an i64, yields the i64 of the next resume.
        // (`cont.resume` is multi-result and handled in the main loop.)
        Inst::ContNew { func, sp } => {
            cx.expect(*func, ValType::I32)?;
            cx.expect(*sp, ValType::I64)?; // the fiber's data-stack base
            ValType::I64
        }
        Inst::Suspend { value } => {
            cx.expect(*value, ValType::I64)?;
            ValType::I64
        }
        // §GC conservative root enumeration: i64 heap_lo, heap_hi, buf, cap ⇒ i64 count. Writes the
        // candidate words into guest memory at `buf`, so it requires a declared window.
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*heap_lo, ValType::I64)?;
            cx.expect(*heap_hi, ValType::I64)?;
            cx.expect(*mask, ValType::I64)?;
            cx.expect(*buf, ValType::I64)?;
            cx.expect(*cap, ValType::I64)?;
            // The top-byte-strip constraint on a *constant* `mask` is checked in `verify_func`'s
            // block loop (which can trace the operand to its defining `i64.const`); a non-constant
            // mask is enforced defensively at runtime on both backends.
            ValType::I64
        }
        // §12 thread join: an i32 thread handle in, the joined vCPU's i64 result out. (The handle
        // is forgeable; safety is the runtime use-site check, like a fiber/capability handle.)
        Inst::ThreadJoin { handle } => {
            cx.expect(*handle, ValType::I32)?;
            ValType::I64
        }
        // §12 futex wait: i64 addr, `ty` expected value, i64 timeout ⇒ i32 status. Touches memory.
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*expected, ty.val())?;
            cx.expect(*timeout, ValType::I64)?;
            ValType::I32
        }
        // §12 futex notify: i64 addr, i32 count ⇒ i32 woken. Requires declared memory.
        Inst::MemoryNotify { addr, count } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            cx.expect(*count, ValType::I32)?;
            ValType::I32
        }
        // A standalone fence produces no value and needs no memory or operands (any ordering is
        // valid for a fence) — accept it directly.
        Inst::AtomicFence { .. } => return Ok(InstResults::None),

        // ----- §17 SIMD (D58): total lane-typing rules -----
        Inst::ConstV128(_) => ValType::V128,
        Inst::V128Load { addr, .. } => {
            if !has_memory {
                return Err(VerifyError::MemoryNotDeclared {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*addr, ValType::I64)?;
            ValType::V128
        }
        Inst::Splat { shape, a } => {
            cx.expect(*a, shape.lane_val())?;
            ValType::V128
        }
        Inst::ExtractLane { shape, lane, a, .. } => {
            if *lane >= shape.lanes() {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            shape.lane_val()
        }
        Inst::ReplaceLane {
            shape, lane, a, b, ..
        } => {
            if *lane >= shape.lanes() {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, shape.lane_val())?;
            ValType::V128
        }
        Inst::VIntBin { shape, a, b, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VIntCmp { shape, a, b, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VShift { shape, a, amt, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*amt, ValType::I32)?;
            ValType::V128
        }
        Inst::VFloatBin { shape, a, b, .. }
        | Inst::VFloatCmp { shape, a, b, .. }
        | Inst::VPMinMax { shape, a, b, .. } => {
            if !shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Fused multiply-add (`relaxed_madd`/`nmadd`): a ternary float-lane op.
        Inst::VFma { shape, a, b, c, .. } => {
            if !shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            cx.expect(*c, ValType::V128)?;
            ValType::V128
        }
        Inst::VFloatUn { shape, a, .. } => {
            if !shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        Inst::VIntUn { shape, a, .. } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // `i8x16.popcnt`: shape is fixed (i8x16), so there is no lane rule to enforce.
        Inst::VPopcnt { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Saturating add/sub is `i8x16`/`i16x8` only (the wasm spec has no wider sat).
        Inst::VSatBin { shape, a, b, .. } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Unsigned rounding average: `i8x16`/`i16x8` only (the only shapes wasm defines `avgr_u`).
        Inst::VAvgr { shape, a, b } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Dot product: fixed shapes (i16x8 → i32x4), so there is no lane rule to enforce.
        Inst::VDot { a, b } | Inst::VDotI8 { a, b } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Extended multiply: the result `shape` must be a wide integer shape (has a half-width
        // source to widen from) — same rule as widen.
        Inst::VExtMul { shape, a, b, .. } => {
            if shape.narrower().is_none() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Extended pairwise add: wide integer result only. `i16x8`/`i32x4` (i64x2 has no wasm op,
        // but a half-width source exists, so restrict to the two wasm shapes explicitly).
        Inst::VExtAddPairwise { shape, a, .. } => {
            if !matches!(shape, VShape::I16x8 | VShape::I32x4) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Q15 rounding multiply: fixed `i16x8`, so there is no lane rule to enforce.
        Inst::VQ15MulrSat { a, b } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Widen: the result shape must be an integer shape that has a (half-width) source.
        Inst::VWiden { shape, a, .. } => {
            if shape.narrower().is_none() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Narrow: `i8x16`/`i16x8` results only (the wasm spec has no wider narrow).
        Inst::VNarrow { shape, a, b, .. } => {
            if !matches!(shape, VShape::I8x16 | VShape::I16x8) {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        // Lane conversions: `v128` → `v128`, fully described by the op.
        Inst::VConvert { a, .. } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        // Boolean reductions: a `v128` → an `i32`. `all_true`/`bitmask` carry an integer shape;
        // `any_true` is shape-agnostic.
        Inst::VAnyTrue { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::I32
        }
        Inst::VAllTrue { shape, a } | Inst::VBitmask { shape, a } => {
            if shape.is_float() {
                return Err(VerifyError::BadSimdShape {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            ValType::I32
        }
        Inst::VBitBin { a, b, .. } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::VNot { a } => {
            cx.expect(*a, ValType::V128)?;
            ValType::V128
        }
        Inst::Bitselect { a, b, mask } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            cx.expect(*mask, ValType::V128)?;
            ValType::V128
        }
        Inst::Shuffle { lanes, a, b } => {
            // Each byte index selects from the 32-byte `a ++ b`; ≥32 is structurally invalid.
            if lanes.iter().any(|&l| l >= 32) {
                return Err(VerifyError::BadSimdLane {
                    func: fi,
                    block: bi,
                });
            }
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }
        Inst::Swizzle { a, b } => {
            cx.expect(*a, ValType::V128)?;
            cx.expect(*b, ValType::V128)?;
            ValType::V128
        }

        // ----- whole-module-dependent calls (results are a callee/signature row) -----
        // `call` needs the callee's signature; a dangling funcidx is rejected (and appends nothing
        // in the typing-only walk, where `result_count` also reports 0).
        Inst::Call { func, args } => {
            let callee = match funcs.get(*func as usize) {
                Some(c) => c,
                None => {
                    return if check {
                        Err(VerifyError::CallFuncOutOfRange {
                            func: fi,
                            block: bi,
                            callee: *func,
                        })
                    } else {
                        Ok(InstResults::None)
                    };
                }
            };
            cx.check_args(args, &callee.params)?;
            return Ok(InstResults::Row(&callee.results));
        }
        // `ref.func` materializes a function reference (`i32`); the index must name a function.
        Inst::RefFunc { func } => {
            if check && *func as usize >= funcs.len() {
                return Err(VerifyError::CallFuncOutOfRange {
                    func: fi,
                    block: bi,
                    callee: *func,
                });
            }
            return Ok(InstResults::One(ValType::I32));
        }
        // `call.dyn`: the interned `sig` (#922) must name a `Func`; `idx` is the `i32` table slot.
        Inst::CallIndirect { ty, idx, args } => {
            if check {
                let Some(ft) = func_sig(type_section, *ty) else {
                    return Err(VerifyError::CallSigInvalid {
                        func: fi,
                        block: bi,
                        sig: *ty,
                    });
                };
                cx.expect(*idx, ValType::I32)?;
                cx.check_args(args, &ft.params)?;
                return Ok(InstResults::Row(&ft.results));
            }
            return Ok(row_or_none(func_sig(type_section, *ty)));
        }
        // `call.cap`: self-asserted signature (no manifest to resolve against), so the interned `sig`
        // need only name a `Func`; the runtime use-site check (host-owned table type_id/generation)
        // carries safety, not typing. `handle` is a forgeable `i32`.
        Inst::CapCall {
            sig, handle, args, ..
        } => {
            if check {
                let Some(ft) = func_sig(type_section, *sig) else {
                    return Err(VerifyError::CallSigInvalid {
                        func: fi,
                        block: bi,
                        sig: *sig,
                    });
                };
                cx.expect(*handle, ValType::I32)?;
                cx.check_args(args, &ft.params)?;
                return Ok(InstResults::Row(&ft.results));
            }
            return Ok(row_or_none(func_sig(type_section, *sig)));
        }
        // §12 `cont.resume` (blocking and non-blocking) appends `(status: i32, value: i64)`; the
        // `block` flag does not affect typing. `k` is a forgeable i64 fiber handle, `arg` an i64.
        Inst::ContResume { k, arg, block: _ } => {
            cx.expect(*k, ValType::I64)?;
            cx.expect(*arg, ValType::I64)?;
            return Ok(InstResults::Two(ValType::I32, ValType::I64));
        }
        // §12 `thread.spawn` resolves a static funcidx whose signature must be the fixed thread-entry
        // type `(i64 sp, i64 arg) -> i64`; appends the spawned vCPU handle (`i32`).
        Inst::ThreadSpawn { func, sp, arg } => {
            if check {
                let callee = funcs
                    .get(*func as usize)
                    .ok_or(VerifyError::CallFuncOutOfRange {
                        func: fi,
                        block: bi,
                        callee: *func,
                    })?;
                if callee.params != [ValType::I64, ValType::I64] || callee.results != [ValType::I64]
                {
                    return Err(VerifyError::ThreadEntrySignature {
                        func: fi,
                        block: bi,
                        callee: *func,
                    });
                }
                cx.expect(*sp, ValType::I64)?;
                cx.expect(*arg, ValType::I64)?;
            }
            return Ok(InstResults::One(ValType::I32));
        }

        // Handled by the no-result pre-match guards above; listed for exhaustiveness (never reached —
        // each `return`s before the match — so a new op forces a decision here, no `_` catch-all).
        Inst::Store { .. }
        | Inst::MemCopy { .. }
        | Inst::MemMove { .. }
        | Inst::MemFill { .. }
        | Inst::AtomicStore { .. }
        | Inst::V128Store { .. }
        | Inst::SetJmp { .. }
        | Inst::LongJmp { .. } => return Ok(InstResults::None),
    };
    Ok(InstResults::One(ty))
}

#[allow(clippy::too_many_arguments)]
fn check_terminator(
    fi: u32,
    bi: u32,
    term: &Terminator,
    types: &[ValType],
    type_section: &[temen_ir::TypeEntry],
    nblocks: u32,
    f: &Func,
    funcs: &[Func],
) -> Result<(), VerifyError> {
    let cx = Cx {
        fi,
        bi,
        types,
        check: true,
    };
    match term {
        Terminator::Br { target, args } => {
            check_edge(&cx, *target, args, nblocks, f)?;
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            cx.expect(*cond, ValType::I32)?;
            check_edge(&cx, *then_blk, then_args, nblocks, f)?;
            check_edge(&cx, *else_blk, else_args, nblocks, f)?;
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            cx.expect(*idx, ValType::I32)?;
            for (t, args) in targets {
                check_edge(&cx, *t, args, nblocks, f)?;
            }
            let (t, args) = default;
            check_edge(&cx, *t, args, nblocks, f)?;
        }
        Terminator::Return(vals) => {
            if vals.len() != f.results.len() {
                return Err(VerifyError::ResultCountMismatch {
                    func: fi,
                    block: bi,
                    expected: f.results.len(),
                    found: vals.len(),
                });
            }
            for (v, want) in vals.iter().zip(&f.results) {
                cx.expect(*v, *want)?;
            }
        }
        Terminator::ReturnCall { func, args } => {
            let callee = funcs
                .get(*func as usize)
                .ok_or(VerifyError::CallFuncOutOfRange {
                    func: fi,
                    block: bi,
                    callee: *func,
                })?;
            check_tail_call(&cx, args, &callee.params, &callee.results, &f.results)?;
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            // Interned signature (#922): the `types` index must name a `Func` entry.
            let Some(ftype) = func_sig(type_section, *ty) else {
                return Err(VerifyError::CallSigInvalid {
                    func: fi,
                    block: bi,
                    sig: *ty,
                });
            };
            cx.expect(*idx, ValType::I32)?;
            check_tail_call(&cx, args, &ftype.params, &ftype.results, &f.results)?;
        }
        // Aborts unconditionally; references nothing, so nothing to check.
        Terminator::Unreachable => {}
    }
    Ok(())
}

/// Shared checks for `return_call`/`return_call.dyn`: the args match the
/// callee's parameters, and the callee's results equal *this* function's results
/// (a tail call returns the callee's results as our own).
fn check_tail_call(
    cx: &Cx,
    args: &[ValIdx],
    callee_params: &[ValType],
    callee_results: &[ValType],
    func_results: &[ValType],
) -> Result<(), VerifyError> {
    cx.check_args(args, callee_params)?;
    if callee_results != func_results {
        return Err(VerifyError::ResultCountMismatch {
            func: cx.fi,
            block: cx.bi,
            expected: func_results.len(),
            found: callee_results.len(),
        });
    }
    Ok(())
}

/// Check a single branch edge: target in range, arg count + types match the target
/// block's declared parameters exactly.
fn check_edge(
    cx: &Cx,
    target: BlockIdx,
    args: &[ValIdx],
    nblocks: u32,
    f: &Func,
) -> Result<(), VerifyError> {
    if target >= nblocks {
        return Err(VerifyError::BlockOutOfRange {
            func: cx.fi,
            block: cx.bi,
            target,
        });
    }
    let target_params = &f.blocks[target as usize].params;
    if args.len() != target_params.len() {
        return Err(VerifyError::ArgCountMismatch {
            func: cx.fi,
            block: cx.bi,
            target,
            expected: target_params.len(),
            found: args.len(),
        });
    }
    for (v, want) in args.iter().zip(target_params) {
        cx.expect(*v, *want)?;
    }
    Ok(())
}

/// Bundles the location + running type vector for concise operand checks. `check` distinguishes the
/// two callers of [`type_inst`]: `verify_func` runs with `check = true` (operand checks enforced);
/// the typing-only walk [`block_value_types`] runs with `check = false`, where `expect`/`check_args`
/// short-circuit to `Ok` so the walk derives result types from the *same* dispatch without rejecting
/// an unverified function — the leniency `func_value_types`' consumers (the debugger, `temen-opt`)
/// rely on. Structural checks that aren't operand-typing (import range, sig/iface resolution, SIMD
/// shape) are likewise gated on `check` at their sites.
struct Cx<'a> {
    fi: u32,
    bi: u32,
    types: &'a [ValType],
    check: bool,
}

impl Cx<'_> {
    /// The type of an earlier-defined operand, or `ValueOutOfRange`.
    fn type_of(&self, v: ValIdx) -> Result<ValType, VerifyError> {
        self.types
            .get(v as usize)
            .copied()
            .ok_or(VerifyError::ValueOutOfRange {
                func: self.fi,
                block: self.bi,
                value: v,
                defined: self.types.len() as u32,
            })
    }

    /// Check a call's arguments: exactly `params.len()` of them, each defined earlier in this
    /// block with `params[i]`'s type. The shared body behind every call-shaped op — `call`,
    /// `call.dyn`, `call.cap`, `call.sym`, `call.import`, `call.import.dyn`, and the two
    /// `return_call` forms (via [`check_tail_call`]) — mirroring [`check_edge`] on the terminator
    /// side. (Handle/index operands, where present, are checked by the caller before this.)
    fn check_args(&self, args: &[ValIdx], params: &[ValType]) -> Result<(), VerifyError> {
        if !self.check {
            return Ok(());
        }
        if args.len() != params.len() {
            return Err(VerifyError::CallArgCountMismatch {
                func: self.fi,
                block: self.bi,
                expected: params.len(),
                found: args.len(),
            });
        }
        for (a, want) in args.iter().zip(params) {
            self.expect(*a, *want)?;
        }
        Ok(())
    }

    /// An operand must be defined earlier in this block and have exactly `want`'s type.
    fn expect(&self, v: ValIdx, want: ValType) -> Result<(), VerifyError> {
        if !self.check {
            return Ok(());
        }
        let found = self.type_of(v)?;
        // `cap` is `i32`-width data in guest code (IMPORTS.md §3.5): the two are value-compatible
        // everywhere operands flow — a `cap` is usable as a handle (`i32`), and an `i32` fills a
        // `cap` slot. They stay *distinct* in signatures, so structural interface matching and the
        // boundary `cap` translation both still key on the marker; those compare `FuncType`s
        // directly, never through `expect`.
        let ok = found == want
            || matches!(
                (found, want),
                (ValType::I32 | ValType::Cap, ValType::I32 | ValType::Cap)
            );
        if !ok {
            return Err(VerifyError::TypeMismatch {
                func: self.fi,
                block: self.bi,
                expected: want,
                found,
            });
        }
        Ok(())
    }
}
