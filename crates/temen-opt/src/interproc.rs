//! Interprocedural passes (see `OPT.md` Phase 3). These are **module-level** — they add, remove, or
//! renumber whole functions — unlike the per-function scalar passes, so they run once over the module
//! rather than inside `optimize_func`. Output is re-verified like everything else in this crate, so a
//! bug here is a clean verify error, never an escape (untrusted-for-escape posture, §20a).
//!
//! Four passes: **constant-funcref devirtualization** (an indirect call on a constant funcref → direct
//! call), **interprocedural constant propagation** (specialize a function on constants its callers
//! agree on — a monotone fixpoint that also resolves constant funcrefs flowing into dispatchers), the
//! **budgeted direct-call inliner** (splice a small callee — straight-line in place, or multi-block by
//! splicing its CFG in and threading values across the call), and **dead-function elimination** (drop
//! functions no reachable code can call). They compose into the end-to-end interprocedural story:
//! devirtualization turns a constant `call.dyn` into a direct `call`, const_prop feeds constants
//! (including funcrefs, which a re-run of devirt then resolves) into callees, the inliner splices a
//! small callee in, and DFE sweeps the now-uncalled leaf — and, because devirtualization removes the
//! indirect dispatch, DFE's conservative gate lifts too.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;
use temen_ir::{Block, Export, Func, FuncIdx, Inst, Module, Terminator, ValType};
use temen_verify::block_value_types;

use crate::{each_operand, get, map_operands, map_term_operands, Known};

/// Visit every **static function index** a function references: a direct `call`, a `ref.func`, a
/// `thread.spawn` entry, and the `return_call` terminator. This mirrors `temen_ir::offset_func_indices`
/// (the merged-module reindexer) exactly — `call.dyn` / `cont.*` dispatch on runtime funcref
/// *values* (an `i32` that equals the function index; the identity table), and `call.import` carries
/// an import index, so none of those name a *static* callee. A `ref.func` **does**: it materializes a
/// callable funcref, so a function whose reference is taken by reachable code must be kept — it may be
/// reached by a later `call.dyn`. Counting `ref.func` as a call edge is the sound over-
/// approximation that keeps such functions live.
fn referenced_funcs(f: &Func, mut visit: impl FnMut(FuncIdx)) {
    for b in &f.blocks {
        for inst in &b.insts {
            match inst {
                Inst::Call { func, .. }
                | Inst::RefFunc { func }
                | Inst::ThreadSpawn { func, .. } => visit(*func),
                _ => {}
            }
        }
        if let Terminator::ReturnCall { func, .. } = &b.term {
            visit(*func);
        }
    }
}

/// Whether the module dispatches on a runtime **funcref value** — `call.dyn`,
/// `return_call.dyn`, or `cont.new` (whose `func` operand is a funcref value, not a static
/// index). Because a funcref equals its function index (the identity table) and can be a plain
/// `ConstI32` indistinguishable from ordinary data, an indirect dispatch could reach *any* in-range
/// function, so removing or renumbering functions is unsound without funcref value-flow analysis.
/// Slice-1 dead-function elimination therefore leaves such modules untouched; the later
/// devirtualization pass folds constant indirect calls to direct ones, after which this pass applies.
/// (`thread.spawn` carries a *static* funcidx — [`referenced_funcs`] already handles it — so it is not
/// a value dispatch and does not gate here.)
fn has_indirect_funcref_dispatch(m: &Module) -> bool {
    m.funcs.iter().flat_map(|f| &f.blocks).any(|b| {
        b.insts
            .iter()
            .any(|i| matches!(i, Inst::CallIndirect { .. } | Inst::ContNew { .. }))
            || matches!(b.term, Terminator::ReturnCallIndirect { .. })
    })
}

/// Whether the module hands out any function index as a value: a `ref.func`, or an index its data
/// image holds ([`Module::data_funcref_slots`], #1830). That value is an observable `i32`, so
/// renumbering functions would change it wherever it flows to data (a return, a store, a comparison)
/// rather than an immediately-consumed dispatch — and a data image's index would name another
/// function. [`dead_func_elim`] bails to the identity when one is present — see the call site.
fn has_ref_func(m: &Module) -> bool {
    !m.data_funcref_slots.is_empty()
        || m.funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i, Inst::RefFunc { .. }))
}

/// Rewrite every static function index in `f` through the old→new map (the exact set
/// [`referenced_funcs`] reads).
fn remap_func_indices(f: &mut Func, map: &[u32]) {
    for b in &mut f.blocks {
        for inst in &mut b.insts {
            match inst {
                Inst::Call { func, .. }
                | Inst::RefFunc { func }
                | Inst::ThreadSpawn { func, .. } => *func = map[*func as usize],
                _ => {}
            }
        }
        if let Terminator::ReturnCall { func, .. } = &mut b.term {
            *func = map[*func as usize];
        }
    }
}

/// **Dead-function elimination.** Drop every function unreachable in the call graph from the roots —
/// the conventional entry (`func 0`, what `run(m, 0, …)` invokes) and every named export — following
/// `call` / `return_call` / `thread.spawn` edges and, conservatively, `ref.func` (a materialized
/// funcref can be reached by a later `call.dyn`). Surviving functions are renumbered densely and
/// every static funcidx reference (and each export) is remapped; `call.dyn` is untouched because
/// it dispatches on the funcref value, which equals the function index and so rides the same map only
/// where a `ref.func` produced it (already remapped above).
///
/// Sound: a dropped function is provably uncallable from any reachable code, so removing it changes no
/// observable behavior. Conservative on `ref.func` (a reference taken but never indirectly called
/// still keeps its function) — correct, never over-eager. Debug info is dropped when any function is
/// removed (its `(func, …)` positions would be stale after renumbering; it is strippable and untrusted
/// for escape, §3a).
pub fn dead_func_elim(m: &Module) -> Module {
    let n = m.funcs.len();
    if n == 0 {
        return m.clone();
    }
    // Unsound to remove/renumber functions while any indirect funcref dispatch is live (see
    // [`has_indirect_funcref_dispatch`]) — bail to the identity until devirtualization removes it.
    if has_indirect_funcref_dispatch(m) {
        return m.clone();
    }
    // Also unsound while any `ref.func` is live: its result **is the function index as a plain i32**
    // (the identity table), so a program can return/store/compare that integer — its value is
    // directly observable, not only a `call.dyn` dispatch target. Renumbering rewrites the
    // `ref.func` operand to preserve *dispatch*, but that silently changes the observed integer
    // (nightly `opt_sccp`, input `[0xfc]`: `ref.func 2` returned as the entry's i32 result folded to
    // `ref.func 1` after func 1 was dropped, so the result went `2 → 1`). A dead `ref.func` is cleared
    // by the per-function DCE before this pass runs in the fixpoint, so this only pins live ones.
    if has_ref_func(m) {
        return m.clone();
    }

    // Reachability closure from the roots.
    let mut reachable = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mark = |i: usize, reachable: &mut [bool], stack: &mut Vec<usize>| {
        if i < n && !reachable[i] {
            reachable[i] = true;
            stack.push(i);
        }
    };
    mark(0, &mut reachable, &mut stack); // conventional entry
    for e in &m.exports {
        mark(e.func as usize, &mut reachable, &mut stack);
    }
    // Interface offers (IMPORTS.md §3.2) are roots too: their op functions are entered from
    // another domain via wiring, invisible to intra-module reachability.
    for e in &m.impl_exports {
        for &f in &e.ops {
            mark(f as usize, &mut reachable, &mut stack);
        }
    }
    while let Some(fi) = stack.pop() {
        referenced_funcs(&m.funcs[fi], |g| {
            let g = g as usize;
            if g < n && !reachable[g] {
                reachable[g] = true;
                stack.push(g);
            }
        });
    }

    if reachable.iter().all(|&r| r) {
        return m.clone(); // nothing dead — identity (no renumbering)
    }

    // old funcidx → new (dense over the survivors, order-preserving).
    let mut map = vec![0u32; n];
    let mut next = 0u32;
    for (i, &live) in reachable.iter().enumerate() {
        if live {
            map[i] = next;
            next += 1;
        }
    }

    let funcs: Vec<Func> = (0..n)
        .filter(|&i| reachable[i])
        .map(|i| {
            let mut f = m.funcs[i].clone();
            remap_func_indices(&mut f, &map);
            f
        })
        .collect();
    // Every export is a root, hence reachable; remap its funcidx through the survivor map.
    let exports: Vec<Export> = m
        .exports
        .iter()
        .map(|e| Export {
            name: e.name.clone(),
            func: map[e.func as usize],
        })
        .collect();
    let impl_exports: Vec<temen_ir::ImplExport> = m
        .impl_exports
        .iter()
        .map(|e| temen_ir::ImplExport {
            name: e.name.clone(),
            interface: e.interface, // interface indices are untouched by func renumbering
            ops: e.ops.iter().map(|&f| map[f as usize]).collect(),
            threaded: e.threaded,
        })
        .collect();

    Module {
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        // None: a module whose data holds a function index bails above.
        data_funcref_slots: Vec::new(),
        tls: Vec::new(),
        funcs,
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports,
        // Offset-based (not function indices), untouched by renumbering — carry through unchanged.
        data_exports: m.data_exports.clone(),
        impl_exports,
        types: m.types.clone(),
        debug_info: None, // positions go stale once functions are renumbered
    }
}

// ---------------------------------------------------------------------------------------
// Budgeted direct-call inliner (OPT.md Phase 3).
// ---------------------------------------------------------------------------------------

/// What [`inline_calls_with`] inlines: direct callees of at most `max_callee` instructions in all,
/// until `budget` instructions have been spliced. Spending the budget bounds growth *and* guarantees
/// termination even through cycles of small functions (each inline spends budget).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InlineLimits {
    /// The largest callee inlined, in instructions across all its blocks — a code-size guard.
    pub max_callee: usize,
    /// Instructions the pass may splice in all.
    pub budget: usize,
}

impl InlineLimits {
    /// [`inline_calls`]'s limits: what a residual gets ahead of the full cleanup pipeline
    /// ([`crate::optimize_module`]).
    pub const OPTIMIZER: InlineLimits = InlineLimits {
        max_callee: 24,
        budget: 4096,
    };
}

/// Inline one **single-block, straight-line** callee at a direct `call` site, in place. The callee's
/// block is `Return`-terminated with no internal control flow, so its body is spliced directly into
/// the caller block — no block split, no cross-block value threading (which block-local SSA would
/// otherwise force for a multi-block callee). The callee's parameters bind to the call's arguments,
/// its instruction results take fresh caller-local indices right where the call was, and the call's
/// result values are forwarded to the callee's returned values. Every operand after the call is
/// renumbered through the same old→new map used by the intra-block passes ([`map_operands`]).
///
/// Sound because a single-block callee is pure straight-line substitution: the same instructions run,
/// in the same order, on the same operands (its params replaced by the caller's argument values), and
/// its result flows exactly where the call's result did. Any effects/traps in the callee stay in the
/// identical position relative to the caller's surrounding code.
fn inline_single_block_call(
    caller: &Block,
    call_idx: usize,
    callee: &Block,
    fn_results: &[usize],
    types: &[temen_ir::TypeEntry],
) -> Block {
    let p = caller.params.len() as u32;

    // First result index of each caller instruction, and the caller block's total value count.
    let mut result_start = Vec::with_capacity(caller.insts.len());
    let mut n = p;
    for inst in &caller.insts {
        result_start.push(n);
        n += inst.result_count(fn_results, types) as u32;
    }
    let base_c = result_start[call_idx];
    let rc = caller.insts[call_idx].result_count(fn_results, types) as u32;
    let args: Vec<u32> = match &caller.insts[call_idx] {
        Inst::Call { args, .. } => args.clone(),
        _ => unreachable!("call site must be a direct call"),
    };

    // old caller value → new caller value. Params keep their indices; results are reassigned as the
    // rebuilt instruction stream is emitted (so post-call values shift by the callee's net size).
    let mut map: Vec<Option<u32>> = vec![None; n as usize];
    for i in 0..p {
        map[i as usize] = Some(i);
    }
    let mut new_insts: Vec<Inst> = Vec::new();
    let mut next = p;

    // Instructions before the call: operands reference only earlier values (identity map so far).
    let emit = |i: usize, new_insts: &mut Vec<Inst>, map: &mut Vec<Option<u32>>, next: &mut u32| {
        let mut inst = caller.insts[i].clone();
        map_operands(&mut inst, &mut |o| {
            map[o as usize].expect("operand defined before use")
        });
        let rcount = caller.insts[i].result_count(fn_results, types) as u32;
        for r in 0..rcount {
            map[(result_start[i] + r) as usize] = Some(*next);
            *next += 1;
        }
        new_insts.push(inst);
    };
    for i in 0..call_idx {
        emit(i, &mut new_insts, &mut map, &mut next);
    }

    // Splice the callee: its params bind to the call's argument values, its results take fresh indices.
    let cp = callee.params.len();
    let mut c_result_start = Vec::with_capacity(callee.insts.len());
    let mut cn = cp as u32;
    for inst in &callee.insts {
        c_result_start.push(cn);
        cn += inst.result_count(fn_results, types) as u32;
    }
    let mut cmap: Vec<u32> = vec![0; cn as usize];
    for (j, cslot) in cmap.iter_mut().enumerate().take(cp) {
        *cslot = map[args[j] as usize].expect("call argument defined before the call");
    }
    for (ci, inst) in callee.insts.iter().enumerate() {
        let mut inst = inst.clone();
        map_operands(&mut inst, &mut |o| cmap[o as usize]);
        let rcount = callee.insts[ci].result_count(fn_results, types) as u32;
        for r in 0..rcount {
            cmap[(c_result_start[ci] + r) as usize] = next;
            next += 1;
        }
        new_insts.push(inst);
    }

    // The call's result values forward to the callee's returned values.
    match &callee.term {
        Terminator::Return(rvals) => {
            for r in 0..rc {
                map[(base_c + r) as usize] = Some(cmap[rvals[r as usize] as usize]);
            }
        }
        _ => unreachable!("inlinable callee must end in `return`"),
    }

    // Instructions after the call: operands referencing the call's results now hit the callee's
    // returned values; everything else is renumbered through the map.
    for i in (call_idx + 1)..caller.insts.len() {
        emit(i, &mut new_insts, &mut map, &mut next);
    }
    let mut term = caller.term.clone();
    map_term_operands(&mut term, &mut |o| {
        map[o as usize].expect("terminator operand defined")
    });

    Block {
        params: caller.params.clone(),
        insts: new_insts,
        term,
    }
}

/// Total instructions across every block of a function — the size charged against the inline budget
/// (a multi-block callee's cost is its whole body, not just its entry).
fn callee_total_insts(callee: &Func) -> usize {
    callee.blocks.iter().map(|b| b.insts.len()).sum()
}

/// Whether `callee` is an inlining candidate for a direct call: no larger than `max_callee`
/// instructions total, every block exits only by an internal branch (`br`/`br_if`/`br_table` — targets
/// stay inside the callee), a value `return`, or `unreachable`, and at least one block actually
/// `return`s (so the spliced-in continuation has a predecessor). Tail-call exits
/// (`return_call`/`return_call.dyn`) are excluded — turning a callee tail call into a caller
/// non-tail call is a separate transform. A single-block `return` callee takes the in-place fast path
/// ([`inline_single_block_call`]); anything else with internal control flow takes the CFG-splicing path
/// ([`inline_multi_block_call`]).
fn is_inlinable(callee: &Func, max_callee: usize) -> bool {
    if callee.blocks.is_empty() || callee_total_insts(callee) > max_callee {
        return false;
    }
    let exits_ok = callee.blocks.iter().all(|b| {
        matches!(
            b.term,
            Terminator::Br { .. }
                | Terminator::BrIf { .. }
                | Terminator::BrTable { .. }
                | Terminator::Return(_)
                | Terminator::Unreachable
        )
    });
    let has_return = callee
        .blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::Return(_)));
    exits_ok && has_return
}

/// Read every value operand of a terminator (the read-only counterpart of [`map_term_operands`]).
fn each_term_operand(term: &Terminator, mut visit: impl FnMut(u32)) {
    let mut t = term.clone();
    map_term_operands(&mut t, &mut |o| {
        visit(o);
        o
    });
}

/// Rewrite a callee block's terminator for splicing into the caller: internal branch targets are
/// shifted by `off` (where the callee's blocks now live), operand indices ≥ `np` are shifted by `capc`
/// (the block grew by `capc` captured pass-through parameters at slots `[np, np+capc)`), every out-edge
/// carries this block's captured parameters along (`cap_params`), and a `return` becomes a branch to
/// the continuation block `cont` passing the return values plus the captured parameters.
fn transform_callee_term(
    term: &Terminator,
    np: u32,
    capc: u32,
    off: u32,
    cont: u32,
    cap_params: &[u32],
) -> Terminator {
    let shift = |idx: u32| if idx < np { idx } else { idx + capc };
    // Edge args: shift each, then append the captured pass-through params.
    let ext = |args: &[u32]| -> Vec<u32> {
        let mut v: Vec<u32> = args.iter().map(|&a| shift(a)).collect();
        v.extend_from_slice(cap_params);
        v
    };
    match term {
        Terminator::Br { target, args } => Terminator::Br {
            target: off + target,
            args: ext(args),
        },
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => Terminator::BrIf {
            cond: shift(*cond),
            then_blk: off + then_blk,
            then_args: ext(then_args),
            else_blk: off + else_blk,
            else_args: ext(else_args),
        },
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => Terminator::BrTable {
            idx: shift(*idx),
            targets: targets.iter().map(|(t, a)| (off + t, ext(a))).collect(),
            default: (off + default.0, ext(&default.1)),
        },
        Terminator::Return(rvals) => Terminator::Br {
            target: cont,
            args: ext(rvals),
        },
        Terminator::Unreachable => Terminator::Unreachable,
        Terminator::ReturnCall { .. } | Terminator::ReturnCallIndirect { .. } => {
            unreachable!("tail-call callee exits are excluded by is_inlinable")
        }
    }
}

/// Inline a **multi-block** callee at a direct `call` site by splicing its CFG into the caller. The
/// caller block is split at the call: the instructions before it stay (branching into the callee, whose
/// parameters bind to the call's arguments), and the instructions after it move to a fresh
/// **continuation** block whose parameters receive the callee's return values. The callee's blocks are
/// appended (targets shifted past the caller's existing blocks), and every `return` becomes a branch to
/// the continuation.
///
/// Block-local SSA forbids the continuation from naming values defined before the call, so each such
/// **captured** value (a pre-call value used after the call) is **threaded** through the callee: it is
/// appended as a pass-through parameter to every callee block and passed along every edge, arriving at
/// the continuation. This over-threads (a block that doesn't need a captured value still carries it);
/// the always-on dead-block-parameter cleanup prunes the unused ones afterward.
///
/// Sound because it is the call's own control/data flow made explicit: the callee body runs between the
/// pre- and post-call code exactly as the call did, its arguments bind to the callee's parameters, its
/// return values flow to where the call's results were used, and each captured value reaches the
/// continuation unchanged (one definition, threaded verbatim along every path). Edits the caller's
/// `blocks` in place: block `bi` is split, and the callee's blocks and the continuation are appended.
fn inline_multi_block_call(
    blocks: &mut Vec<Block>,
    bi: usize,
    call_idx: usize,
    callee: &Func,
    fn_results: &[usize],
    types: &[temen_ir::TypeEntry],
    caller_block_types: &[ValType],
) {
    let placeholder = Block {
        params: Vec::new(),
        insts: Vec::new(),
        term: Terminator::Unreachable,
    };
    let b = core::mem::replace(&mut blocks[bi], placeholder);
    let p = b.params.len() as u32;

    // First result index of each caller-block instruction, and the block's total value count.
    let mut result_start = Vec::with_capacity(b.insts.len());
    let mut n = p;
    for inst in &b.insts {
        result_start.push(n);
        n += inst.result_count(fn_results, types) as u32;
    }
    let base_c = result_start[call_idx];
    let rc = b.insts[call_idx].result_count(fn_results, types) as u32;
    let call_args: Vec<u32> = match &b.insts[call_idx] {
        Inst::Call { args, .. } => args.clone(),
        _ => unreachable!("call site must be a direct call"),
    };

    // Captured = pre-call values (local index < base_c) referenced by post-call insts or the terminator.
    let mut used: BTreeSet<u32> = BTreeSet::new();
    for inst in &b.insts[(call_idx + 1)..] {
        each_operand(inst, |o| {
            if o < base_c {
                used.insert(o);
            }
        });
    }
    each_term_operand(&b.term, |o| {
        if o < base_c {
            used.insert(o);
        }
    });
    let cap: Vec<u32> = used.into_iter().collect();
    let capc = cap.len() as u32;
    let cap_types: Vec<ValType> = cap
        .iter()
        .map(|&c| caller_block_types[c as usize])
        .collect();

    let off = blocks.len() as u32; // callee entry lands at off + 0
    let k = callee.blocks.len() as u32;
    let cont = off + k;

    // Pre-call block (keeps index bi): the pre-call insts, then a branch into the callee passing the
    // call arguments followed by the captured values. The post-call insts move to the continuation.
    let Block {
        params,
        insts: mut pre_insts,
        term,
    } = b;
    let post_insts = pre_insts.split_off(call_idx + 1);
    pre_insts.truncate(call_idx);
    let mut pre_args = call_args;
    pre_args.extend(cap.iter().copied());
    blocks[bi] = Block {
        params,
        insts: pre_insts,
        term: Terminator::Br {
            target: off,
            args: pre_args,
        },
    };

    // Callee blocks: append the captured params to each, shift internal operands/targets, thread.
    for cb in &callee.blocks {
        let np = cb.params.len() as u32;
        let mut params = cb.params.clone();
        params.extend(cap_types.iter().copied());
        let mut insts = cb.insts.clone();
        for inst in &mut insts {
            map_operands(inst, &mut |o| if o < np { o } else { o + capc });
        }
        let cap_params: Vec<u32> = (np..np + capc).collect();
        let term = transform_callee_term(&cb.term, np, capc, off, cont, &cap_params);
        blocks.push(Block {
            params,
            insts,
            term,
        });
    }

    // Continuation block: post-call insts + original terminator, with pre-call/call values remapped to
    // the continuation's own locals. Its parameters are the call's results then the captured values.
    let mut map: Vec<u32> = vec![u32::MAX; n as usize];
    for r in 0..rc {
        map[(base_c + r) as usize] = r; // call result r → continuation param r
    }
    for (i, &c) in cap.iter().enumerate() {
        map[c as usize] = rc + i as u32; // captured value → continuation param rc+i
    }
    let mut next_cont = rc + capc;
    let mut cont_insts: Vec<Inst> = Vec::with_capacity(post_insts.len());
    for (j, mut inst) in post_insts.into_iter().enumerate() {
        let rcount = inst.result_count(fn_results, types) as u32;
        map_operands(&mut inst, &mut |o| map[o as usize]);
        for r in 0..rcount {
            map[(result_start[call_idx + 1 + j] + r) as usize] = next_cont;
            next_cont += 1;
        }
        cont_insts.push(inst);
    }
    let mut cont_term = term;
    map_term_operands(&mut cont_term, &mut |o| map[o as usize]);
    let mut cont_params = callee.results.clone();
    cont_params.extend(cap_types.iter().copied());
    blocks.push(Block {
        params: cont_params,
        insts: cont_insts,
        term: cont_term,
    });
}

/// **Budgeted direct-call inliner**, with [`InlineLimits::OPTIMIZER`] (see [`inline_calls_with`]).
pub fn inline_calls(m: &Module) -> Module {
    inline_calls_with(m, InlineLimits::OPTIMIZER)
}

/// **Budgeted direct-call inliner.** Splice small callees into direct `call` sites in one forward
/// pass — functions in index order, their blocks in order (including the blocks a splice appends),
/// and each spliced body rescanned in place so its own calls inline too — until no eligible site
/// remains or `limits.budget` is spent. A straight-line single-block callee is spliced
/// in place ([`inline_single_block_call`]); a callee with internal control flow has its CFG spliced in
/// ([`inline_multi_block_call`]). Direct self-recursion is skipped, and every inline spends budget, so
/// cycles of small functions terminate.
///
/// The one pass inlines the sites rescanning the module from the top after each inline would, in the
/// same order, except that it never goes back to a site it has passed. A passed site stays ineligible,
/// since the budget only falls and a splice only grows its caller, unless the splice was of a callee
/// with no instructions: that removes the call, so its caller shrinks and could newly fit a site
/// already passed. Inlining does not change any function's signature, so caller/callee
/// indices stay valid; the now-uncalled callee is swept later by [`dead_func_elim`]. Debug info
/// survives for every function nothing was inlined into; a caller's own positions go stale, so its
/// locations and variables are dropped ([`crate::drop_stale_debug`]).
pub fn inline_calls_with(m: &Module, limits: InlineLimits) -> Module {
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let types = &m.types;
    let has_memory = m.memory.is_some();
    let mut funcs = m.funcs.clone();
    let mut budget = limits.budget;
    let mut touched = vec![false; funcs.len()];

    for ci in 0..funcs.len() {
        // `blocks.len()` is re-read each time: a multi-block splice appends the callee's blocks and
        // the continuation, and the pass reaches them in turn.
        let mut bi = 0;
        while bi < funcs[ci].blocks.len() {
            let mut ii = 0;
            while ii < funcs[ci].blocks[bi].insts.len() {
                let callee = match funcs[ci].blocks[bi].insts[ii] {
                    Inst::Call { func, .. } => func as usize,
                    _ => {
                        ii += 1;
                        continue;
                    }
                };
                // Skip direct self-recursion, an out-of-range index, an ineligible or too-big callee.
                let fits = callee != ci
                    && callee < funcs.len()
                    && is_inlinable(&funcs[callee], limits.max_callee)
                    && callee_total_insts(&funcs[callee]) <= budget;
                if !fits {
                    ii += 1;
                    continue;
                }
                budget -= callee_total_insts(&funcs[callee]);
                touched[ci] = true;
                if funcs[callee].blocks.len() == 1 {
                    // Straight-line callee: its body now starts at `ii`, where the scan resumes.
                    let callee_block = funcs[callee].blocks[0].clone();
                    funcs[ci].blocks[bi] = inline_single_block_call(
                        &funcs[ci].blocks[bi],
                        ii,
                        &callee_block,
                        &fn_results,
                        types,
                    );
                } else {
                    // Callee has internal control flow: splice its CFG in, threading captured values
                    // through. The block now ends at the call; the rest of it moved to the appended
                    // continuation.
                    let block_types =
                        block_value_types(&funcs[ci].blocks[bi], &funcs, types, has_memory);
                    let callee_fn = funcs[callee].clone();
                    inline_multi_block_call(
                        &mut funcs[ci].blocks,
                        bi,
                        ii,
                        &callee_fn,
                        &fn_results,
                        types,
                        &block_types,
                    );
                    break;
                }
            }
            bi += 1;
        }
    }

    if !touched.contains(&true) {
        return m.clone();
    }
    Module {
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        data_funcref_slots: m.data_funcref_slots.clone(),
        tls: Vec::new(),
        funcs,
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports: m.exports.clone(),
        // Offset-based (not function indices), untouched by renumbering — carry through unchanged.
        data_exports: m.data_exports.clone(),
        impl_exports: m.impl_exports.clone(),
        types: m.types.clone(),
        debug_info: m
            .debug_info
            .as_ref()
            .and_then(|d| crate::drop_stale_debug(d, &touched)),
    }
}

// ---------------------------------------------------------------------------------------
// Interprocedural constant propagation (OPT.md Phase 3).
// ---------------------------------------------------------------------------------------

/// Whether block `0` (the entry) is the target of any branch — i.e. it is a loop header, so its
/// parameters are phis fed by back edges as well as the function's arguments. In that case a parameter
/// is *not* simply its incoming call argument, so it must not be replaced by a call-site constant.
fn entry_has_predecessors(f: &Func) -> bool {
    !crate::cfg::Cfg::new(&f.blocks).preds[0].is_empty()
}

/// Materialize each `(param, constant)` in `subs` at the top of the entry block and rewrite the
/// parameter's uses to it. Prepending `c = subs.len()` constants shifts every instruction result by
/// `c`; a use of a substituted parameter becomes the matching constant, other parameter uses are
/// unchanged. The parameter list is untouched (the signature must stay valid) — the parameter is simply
/// left dead.
fn substitute_params(entry: &Block, subs: &[(usize, Known)]) -> Block {
    let np = entry.params.len();
    let c = subs.len() as u32;
    let mut param_to_const: BTreeMap<u32, u32> = BTreeMap::new();
    let mut insts: Vec<Inst> = Vec::with_capacity(subs.len() + entry.insts.len());
    for (pos, (j, k)) in subs.iter().enumerate() {
        param_to_const.insert(*j as u32, np as u32 + pos as u32);
        insts.push(k.to_const_inst());
    }
    let remap = |o: u32| -> u32 {
        if (o as usize) < np {
            param_to_const.get(&o).copied().unwrap_or(o)
        } else {
            o + c
        }
    };
    for inst in &entry.insts {
        let mut ni = inst.clone();
        map_operands(&mut ni, &mut |o| remap(o));
        insts.push(ni);
    }
    let mut term = entry.term.clone();
    map_term_operands(&mut term, &mut |o| remap(o));
    Block {
        params: entry.params.clone(),
        insts,
        term,
    }
}

/// A constant-propagation lattice value for a function parameter: `Bottom` (no call reaches it yet), a
/// single known constant, or `Top` (could be anything). `join` moves up the lattice only.
#[derive(Clone, Copy, PartialEq)]
enum Cp {
    Bottom,
    Const(Known),
    Top,
}

impl Cp {
    fn join(self, other: Cp) -> Cp {
        match (self, other) {
            (Cp::Bottom, x) | (x, Cp::Bottom) => x,
            (Cp::Const(a), Cp::Const(b)) if a == b => Cp::Const(a),
            _ => Cp::Top,
        }
    }
}

/// Per block-local value, the constant it statically holds here — like [`crate::block_consts`] but a
/// `ref.func` counts as its funcidx (a funcref **is** its index; the identity table), so a constant
/// funcref flows through the analysis as an `i32` and can resolve a `call.dyn`.
fn block_knowns(
    b: &Block,
    fn_results: &[usize],
    types: &[temen_ir::TypeEntry],
) -> Vec<Option<Known>> {
    let mut k = vec![None; b.params.len()];
    for inst in &b.insts {
        let rc = inst.result_count(fn_results, types);
        if rc == 1 {
            k.push(match inst {
                Inst::RefFunc { func } => Some(Known::I32(*func as i32)),
                other => crate::const_value(other),
            });
        } else {
            for _ in 0..rc {
                k.push(None);
            }
        }
    }
    k
}

/// **Interprocedural constant propagation.** A monotone fixpoint computes, per function parameter, the
/// join of the value passed at every call that can reach it; a parameter that resolves to a single
/// constant is substituted in the entry block ([`substitute_params`]). The per-function passes then fold
/// through it (branch resolution, arithmetic), and — because a **constant funcref** propagated into a
/// dispatcher's parameter makes its `call.dyn` index constant — the devirtualization that runs
/// after this pass resolves that indirect call to a direct one. Signatures are unchanged (a substituted
/// parameter is left dead, callers keep passing it), so all funcidxs stay valid.
///
/// **Soundness** rests on seeing *every* call that can reach a parameter. A direct `call`/`return_call`
/// feeds its callee; an indirect `call.dyn`/`return_call.dyn` whose index resolves to a
/// constant funcref feeds exactly that (signature-matching) target — a mismatched or out-of-range index
/// traps and reaches no one. Values we cannot see are seeded `Top` and never substituted: the entry
/// (`func 0`), exports, `ref.func`-taken and `thread.spawn` functions (reachable via a path the fixpoint
/// doesn't model), and loop-header entries (a parameter is a phi there, not the call argument). The two
/// hard gates: any `cont.new` (a funcref value run *later* with resume-time arguments) makes the pass
/// bail, and if **any** indirect index is still `Top` at the fixpoint — an unknown funcref that could
/// reach any function with arguments we never counted — the pass bails entirely. Output is re-verified.
pub fn const_prop(m: &Module) -> Module {
    let n = m.funcs.len();
    if n == 0 {
        return m.clone();
    }
    // A `cont.new` dispatches on a funcref value but runs it *later* with resume-time arguments we
    // cannot see — too subtle to model, so bail if any is present.
    if m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .any(|b| b.insts.iter().any(|i| matches!(i, Inst::ContNew { .. })))
    {
        return m.clone();
    }
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let types = &m.types;

    // Parameters we cannot fully see are seeded `Top`.
    let mut opaque = vec![false; n];
    opaque[0] = true;
    for e in &m.exports {
        if (e.func as usize) < n {
            opaque[e.func as usize] = true;
        }
    }
    for (i, f) in m.funcs.iter().enumerate() {
        if entry_has_predecessors(f) {
            opaque[i] = true;
        }
    }
    for (o, taken) in opaque.iter_mut().zip(temen_ir::taken_funcs(m)) {
        *o |= taken;
    }
    for f in &m.funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                if let Inst::ThreadSpawn { func, .. } = inst {
                    if (*func as usize) < n {
                        opaque[*func as usize] = true;
                    }
                }
            }
        }
    }

    let mut val: Vec<Vec<Cp>> = m
        .funcs
        .iter()
        .enumerate()
        .map(|(i, f)| vec![if opaque[i] { Cp::Top } else { Cp::Bottom }; f.params.len()])
        .collect();

    // Resolve a block-local operand to a lattice value. `fp` is the count of leading locals that are
    // **function parameters** — the function's arity in the *entry* block (where block params equal the
    // function's), and `0` in every other block (whose low locals are phis this analysis doesn't track,
    // so they read as their static constant, i.e. `Top`). A function parameter reads its current `val`;
    // anything else reads its static constant, or `Top`. Pure over `val` so the caller can then mutate.
    let eval = |val: &[Vec<Cp>], ci: usize, fp: usize, knowns: &[Option<Known>], local: u32| {
        if (local as usize) < fp {
            val[ci][local as usize]
        } else {
            match get(knowns, local) {
                Some(k) => Cp::Const(k),
                None => Cp::Top,
            }
        }
    };
    // The target of an indirect call whose index resolves to a signature-matching constant funcref.
    // `ty` is an interned `types` index (FuncType interning, #922); resolve it to the signature.
    let target = |cp: Cp, ty: u32| -> Option<usize> {
        let Some(temen_ir::TypeEntry::Func(ft)) = m.types.get(ty as usize) else {
            return None;
        };
        if let Cp::Const(Known::I32(g)) = cp {
            let g = g as usize;
            if g < n && m.funcs[g].params == ft.params && m.funcs[g].results == ft.results {
                return Some(g);
            }
        }
        None
    };

    loop {
        let mut changed = false;
        for ci in 0..n {
            let caller = &m.funcs[ci];
            let np = caller.params.len();
            for (bi, b) in caller.blocks.iter().enumerate() {
                let fp = if bi == 0 { np } else { 0 };
                let knowns = block_knowns(b, &fn_results, types);
                // Collect (callee, arg-lattice-values) for every call this block makes, reading `val`.
                let mut feeds: Vec<(usize, Vec<Cp>)> = Vec::new();
                let mut push = |callee: usize, args: &[u32], val: &[Vec<Cp>]| {
                    let cps = args
                        .iter()
                        .map(|&a| eval(val, ci, fp, &knowns, a))
                        .collect();
                    feeds.push((callee, cps));
                };
                for inst in &b.insts {
                    match inst {
                        Inst::Call { func, args } => push(*func as usize, args, &val),
                        Inst::CallIndirect { ty, idx, args } => {
                            if let Some(g) = target(eval(&val, ci, fp, &knowns, *idx), *ty) {
                                push(g, args, &val);
                            }
                        }
                        _ => {}
                    }
                }
                match &b.term {
                    Terminator::ReturnCall { func, args } => push(*func as usize, args, &val),
                    Terminator::ReturnCallIndirect { ty, idx, args } => {
                        if let Some(g) = target(eval(&val, ci, fp, &knowns, *idx), *ty) {
                            push(g, args, &val);
                        }
                    }
                    _ => {}
                }
                // Apply the joins (writing `val`), now that the reads are done.
                for (callee, cps) in feeds {
                    if callee >= n {
                        continue;
                    }
                    for (j, cp) in cps.into_iter().enumerate() {
                        if j >= val[callee].len() {
                            break;
                        }
                        let nv = val[callee][j].join(cp);
                        if nv != val[callee][j] {
                            val[callee][j] = nv;
                            changed = true;
                        }
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }

    // If any indirect index is still `Top` at the fixpoint, an unknown funcref could reach any function
    // with arguments we never counted — unsound to specialize anyone.
    for ci in 0..n {
        let caller = &m.funcs[ci];
        let np = caller.params.len();
        for (bi, b) in caller.blocks.iter().enumerate() {
            let fp = if bi == 0 { np } else { 0 };
            let knowns = block_knowns(b, &fn_results, types);
            let idx_top = |idx: u32| eval(&val, ci, fp, &knowns, idx) == Cp::Top;
            for inst in &b.insts {
                if let Inst::CallIndirect { idx, .. } = inst {
                    if idx_top(*idx) {
                        return m.clone();
                    }
                }
            }
            if let Terminator::ReturnCallIndirect { idx, .. } = &b.term {
                if idx_top(*idx) {
                    return m.clone();
                }
            }
        }
    }

    let mut funcs = m.funcs.clone();
    let mut changed = false;
    for i in 0..n {
        if opaque[i] {
            continue;
        }
        let subs: Vec<(usize, Known)> = (0..funcs[i].params.len())
            .filter_map(|j| match val[i][j] {
                Cp::Const(k) => Some((j, k)),
                _ => None,
            })
            .collect();
        if subs.is_empty() {
            continue;
        }
        funcs[i].blocks[0] = substitute_params(&funcs[i].blocks[0], &subs);
        changed = true;
    }

    if !changed {
        return m.clone();
    }
    Module {
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        data_funcref_slots: m.data_funcref_slots.clone(),
        tls: Vec::new(),
        funcs,
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports: m.exports.clone(),
        // Offset-based (not function indices), untouched by renumbering — carry through unchanged.
        data_exports: m.data_exports.clone(),
        impl_exports: m.impl_exports.clone(),
        types: m.types.clone(),
        debug_info: None, // instruction positions shift in a specialized entry block
    }
}

// ---------------------------------------------------------------------------------------
// Constant-funcref devirtualization (OPT.md Phase 3).
// ---------------------------------------------------------------------------------------

/// Per block-local value, the function index it is known to hold *as a funcref* — from a `ref.func`
/// (a funcref **is** its funcidx) or an in-range `ConstI32` (a funcref is a plain `i32`; the identity
/// table). Parameters, out-of-range constants, and everything else are `None`. Same block-local
/// forward scan as [`crate::block_consts`], specialized to funcref constants.
fn block_funcrefs(
    b: &Block,
    fn_results: &[usize],
    types: &[temen_ir::TypeEntry],
    num_funcs: usize,
) -> Vec<Option<u32>> {
    let mut known: Vec<Option<u32>> = vec![None; b.params.len()];
    for inst in &b.insts {
        let rc = inst.result_count(fn_results, types);
        if rc == 1 {
            known.push(match *inst {
                Inst::RefFunc { func } => Some(func),
                Inst::ConstI32(v) if v >= 0 && (v as usize) < num_funcs => Some(v as u32),
                _ => None,
            });
        } else {
            for _ in 0..rc {
                known.push(None);
            }
        }
    }
    known
}

/// If block-local value `idx` holds a constant funcref whose target function's signature matches `ty`,
/// return that callee index — otherwise `None`, leaving the indirect call untouched so it dispatches
/// (and, on a signature mismatch or out-of-range index, **traps**) exactly as before. Direct-calling a
/// signature-mismatched target would run the wrong function instead of trapping, so the sig check is
/// load-bearing for soundness, not just an optimization guard.
fn resolve_devirt(
    known: &[Option<u32>],
    idx: u32,
    ty: u32,
    types: &[temen_ir::TypeEntry],
    funcs: &[Func],
) -> Option<u32> {
    let k = known.get(idx as usize).copied().flatten()?;
    let f = funcs.get(k as usize)?;
    // `ty` is an interned `types` index (FuncType interning, #922); resolve to the signature.
    let temen_ir::TypeEntry::Func(ft) = types.get(ty as usize)? else {
        return None;
    };
    (f.params == ft.params && f.results == ft.results).then_some(k)
}

/// **Constant-funcref devirtualization.** Rewrite a `call.dyn` / `return_call.dyn` whose
/// index is a compile-time-constant funcref (a `ref.func` or an in-range `ConstI32`) into the
/// equivalent direct `call` / `return_call`, when the target's signature matches the call's `ty`.
/// Because the signatures match, the result arity is identical, so the rewrite is **in place** — no
/// block-local value renumbering. The dead `ref.func`/`const` feeding the index is then DCE'd, the
/// direct call becomes an inlining candidate, and — with the indirect dispatch gone — dead-function
/// elimination's conservative gate lifts.
///
/// Sound because a `call.dyn` on a constant, in-range, signature-matching funcref deterministically
/// calls `funcs[idx]` (the identity table; cf. the interpreter's `table_lookup`), which is exactly what
/// the direct call does. A mismatched or out-of-range index is left as an indirect call so it still
/// traps identically. Debug info is dropped on any rewrite (an instruction changed).
pub fn devirtualize(m: &Module) -> Module {
    let fn_results: Vec<usize> = m.funcs.iter().map(|f| f.results.len()).collect();
    let types = &m.types;
    let num_funcs = m.funcs.len();
    let mut funcs = m.funcs.clone();
    let mut changed = false;

    for f in &mut funcs {
        for b in &mut f.blocks {
            let known = block_funcrefs(b, &fn_results, types, num_funcs);
            for inst in &mut b.insts {
                let repl = if let Inst::CallIndirect { ty, idx, args } = inst {
                    resolve_devirt(&known, *idx, *ty, types, &m.funcs).map(|k| Inst::Call {
                        func: k,
                        args: core::mem::take(args),
                    })
                } else {
                    None
                };
                if let Some(r) = repl {
                    *inst = r;
                    changed = true;
                }
            }
            let repl = if let Terminator::ReturnCallIndirect { ty, idx, args } = &mut b.term {
                resolve_devirt(&known, *idx, *ty, types, &m.funcs).map(|k| Terminator::ReturnCall {
                    func: k,
                    args: core::mem::take(args),
                })
            } else {
                None
            };
            if let Some(r) = repl {
                b.term = r;
                changed = true;
            }
        }
    }

    if !changed {
        return m.clone();
    }
    Module {
        data_ptrs: Vec::new(),
        data_funcrefs: Vec::new(),
        data_funcref_slots: m.data_funcref_slots.clone(),
        tls: Vec::new(),
        funcs,
        memory: m.memory,
        data: m.data.clone(),
        imports: m.imports.clone(),
        exports: m.exports.clone(),
        // Offset-based (not function indices), untouched by renumbering — carry through unchanged.
        data_exports: m.data_exports.clone(),
        impl_exports: m.impl_exports.clone(),
        types: m.types.clone(),
        debug_info: None, // an instruction/terminator changed
    }
}
