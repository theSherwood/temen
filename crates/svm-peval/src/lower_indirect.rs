//! **Approach A** — bounded-target lowering of a *dynamic-index* `call_indirect`.
//!
//! The specializer inlines a `call_indirect` only when its table index folds to a **constant**
//! (see [`crate::specialize`]'s `resolve_indirect`); a genuinely runtime index returns
//! [`crate::SpecError::Unsupported`], because the single-function residual carries no table to
//! dispatch through. That blocks specialization past the dispatch point in exactly the interpreters
//! that motivate peval (closures / first-class functions / builtin function-pointer tables).
//!
//! This pass is the offline analogue of a polymorphic inline cache that simply caches *all* arms:
//! for a site whose demanded signature `ty` is matched by a **small** set of module functions
//! `S = { i : funcs[i] : ty }`, rewrite
//!
//! ```text
//!     v = call_indirect ty idx args
//! ```
//!
//! into an explicit, in-IR dispatch over the (masked) index:
//!
//! ```text
//!     m = idx & MASK                 // MASK = next_pow2(nfuncs) - 1, the call_indirect table mask
//!     if m == s0 { v = call s0 args } // s0 .. s_{k-1} = the signature-matching functions
//!     elif m == s1 { v = call s1 args }
//!     ...
//!     else { unreachable }           // (see the trap-kind note below)
//!     // continue with v
//! ```
//!
//! Each arm is now a **direct** call to a constant function, which the specializer already inlines /
//! outlines and folds through. The dynamic branch on `m` survives as a residual branch — dispatch is
//! kept, but everything *around* it (and inside each target) still specializes. When `idx` turns out
//! constant in context, the specializer folds `m` and the compare chain, collapsing back to the
//! single resolved arm, so lowering never pessimizes the constant-index case.
//!
//! **Soundness of the masked compare.** The interpreter dispatches `slot = idx & (next_pow2(n)-1)`
//! then checks the target's signature, trapping `IndirectCallType` on a mismatch or empty slot
//! ([`svm_interp`]'s `dispatch_indirect`). Baking the *same* mask and comparing against the
//! signature-matching function indices reproduces that exactly — including a wrapped out-of-range
//! index that lands back on a matching slot (it matches the same arm here).
//!
//! **Trap-kind gap (the one known divergence).** The `else` arm is reached only when `m` selects a
//! slot whose function has a *different* signature (or an empty pad slot) — where the source traps
//! `IndirectCallType`. There is no terminator that raises that specific trap (only
//! [`Terminator::Unreachable`], `Trap::Unreachable`), so this arm diverges in **trap kind** (both
//! still trap; neither escapes, and the residual is re-verified regardless). For a well-formed
//! interpreter the index is always a real function object, so the arm is dead and cleaned up. This
//! is the prototype boundary: making it fully faithful wants a trap-carrying terminator (a small
//! cross-crate follow-up), which the ROI measurement should justify before it is taken on.
//!
//! **Cap and decline.** A site whose signature is matched by more than `cap` functions is left
//! untouched (the specializer still returns `Unsupported` for it) — the megamorphic dispatch (e.g.
//! Lua's generic "call any function" path, ~160–250 targets) is not worth exhaustive residualizing.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use svm_ir::{BinOp, Block, CmpOp, Func, FuncType, Inst, IntTy, Module, Terminator, ValType};
use svm_verify::func_value_types;

use crate::{map_operands, map_term_operands};

/// The mask a `call_indirect` applies to its index: `next_power_of_two(nfuncs) - 1` (the natural
/// module-0 table is padded to a power of two; real functions occupy `[0, nfuncs)`).
fn table_mask(nfuncs: usize) -> i32 {
    (nfuncs.max(1).next_power_of_two() - 1) as u32 as i32
}

/// The signature-matching functions for `ty`, or `None` if there are none or more than `cap`
/// (both cases leave the site un-lowered).
fn candidates(module: &Module, ty: &FuncType, cap: usize) -> Option<Vec<u32>> {
    let mut s = Vec::new();
    for (i, f) in module.funcs.iter().enumerate() {
        if f.params == ty.params && f.results == ty.results {
            s.push(i as u32);
            if s.len() > cap {
                return None;
            }
        }
    }
    (!s.is_empty()).then_some(s)
}

/// One eligible dynamic `call_indirect`: its position and resolved target set.
struct Site {
    block: usize,
    k: usize,
    ty: FuncType,
    cands: Vec<u32>,
}

/// Find the first body `call_indirect` with a small, non-empty signature-matching target set.
/// (Over-cap / unmatched sites are skipped, not returned, so the driver loop terminates once every
/// eligible site has been rewritten into direct calls.)
fn find_site(f: &Func, module: &Module, cap: usize) -> Option<Site> {
    for (bi, b) in f.blocks.iter().enumerate() {
        for (k, inst) in b.insts.iter().enumerate() {
            if let Inst::CallIndirect { ty, .. } = inst {
                if let Some(cands) = candidates(module, ty, cap) {
                    return Some(Site {
                        block: bi,
                        k,
                        ty: ty.clone(),
                        cands,
                    });
                }
            }
        }
    }
    None
}

/// Rewrite a module so every eligible dynamic-index body `call_indirect` becomes an explicit
/// bounded dispatch (see the module docs). `cap` bounds the target fan-out a site may have to be
/// lowered. `cap == 0` is a no-op. Function *signatures* are unchanged, so `call_indirect` target
/// indices — and every other reference — stay valid.
pub fn lower_indirect_dispatch(module: &Module, cap: usize) -> Module {
    if cap == 0 {
        return module.clone();
    }
    let mask = table_mask(module.funcs.len());
    let fn_results: Vec<usize> = module.funcs.iter().map(|f| f.results.len()).collect();
    let has_memory = module.memory.is_some();
    let mut out = module.clone();
    for fi in 0..out.funcs.len() {
        // Rewrite one eligible site per pass, re-typing the (growing) function each time; a lowered
        // site becomes direct calls and is never re-found, so this drains in finitely many passes.
        while let Some(site) = find_site(&out.funcs[fi], module, cap) {
            let types = func_value_types(&out.funcs[fi], &module.funcs, has_memory);
            split_site(&mut out.funcs[fi], site, &types, mask, &fn_results);
        }
    }
    out
}

/// Split block `site.block` at the `call_indirect` (inst `site.k`) into a masked dispatch, one
/// direct-call arm per candidate, an `unreachable` default, and a continuation carrying the
/// original block tail. Live values crossing the split are threaded as block parameters (dead ones
/// are removed by the optimizer). `types` is the pre-split value typing of the function.
fn split_site(f: &mut Func, site: Site, types: &[Vec<ValType>], mask: i32, fn_results: &[usize]) {
    let Site {
        block: bi,
        k,
        ty,
        cands,
    } = site;

    // --- snapshot everything needed from the original block before we overwrite it ---
    let b_params = f.blocks[bi].params.clone();
    let head_insts_src = f.blocks[bi].insts[..k].to_vec();
    let ci = f.blocks[bi].insts[k].clone();
    let tail_insts = f.blocks[bi].insts[k + 1..].to_vec();
    let tail_term = f.blocks[bi].term.clone();
    let btypes = &types[bi];

    let (idx, args) = match &ci {
        Inst::CallIndirect { idx, args, .. } => (*idx, args.clone()),
        _ => unreachable!("find_site only returns a CallIndirect"),
    };
    let r = ty.results.len();

    // Value index just past the last head value (params + results of insts[0..k]); the CI results
    // occupy [head_count, head_count + r), and the tail's own values follow.
    let mut head_count = b_params.len();
    for inst in &head_insts_src {
        head_count += inst.result_count(fn_results);
    }
    let head_count = head_count as u32;

    // Live head values referenced across the split: the call's operands and every tail use.
    let mut refs: BTreeSet<u32> = BTreeSet::new();
    let mut note = |v: u32| {
        if v < head_count {
            refs.insert(v);
        }
    };
    note(idx);
    for &a in &args {
        note(a);
    }
    for inst in &tail_insts {
        let mut c = inst.clone();
        map_operands(&mut c, &mut |o| {
            note(o);
            o
        });
    }
    {
        let mut t = tail_term.clone();
        map_term_operands(&mut t, &mut |o| {
            note(o);
            o
        });
    }
    let live: Vec<u32> = refs.into_iter().collect();
    let live_pos: BTreeMap<u32, u32> = live
        .iter()
        .enumerate()
        .map(|(i, &v)| (v, i as u32))
        .collect();
    let nl = live.len() as u32;
    let live_types: Vec<ValType> = live.iter().map(|&v| btypes[v as usize]).collect();

    // --- block index layout (all new blocks appended in this exact order) ---
    let s = cands.len() as u32;
    let base = f.blocks.len() as u32;
    let d_base = base; // dispatch chain D_0..D_{s-1}
    let c_base = d_base + s; // call arms  C_0..C_{s-1}
    let default_blk = c_base + s; // unreachable default
    let cont_blk = default_blk + 1; // continuation (original tail)

    // --- head block (reuses index `bi`): run insts[0..k], mask the index, branch to D_0 ---
    let mut head_insts = head_insts_src;
    head_insts.push(Inst::ConstI32(mask));
    let mask_val = head_count; // ConstI32 result
    head_insts.push(Inst::IntBin {
        ty: IntTy::I32,
        op: BinOp::And,
        a: idx,
        b: mask_val,
    });
    let m_val = head_count + 1; // masked index
    let mut d0_args = live.clone();
    d0_args.push(m_val);
    f.blocks[bi] = Block {
        params: b_params,
        insts: head_insts,
        term: Terminator::Br {
            target: d_base,
            args: d0_args,
        },
    };

    let mut new_blocks: Vec<Block> = Vec::with_capacity((s * 2 + 2) as usize);

    // --- dispatch chain: D_j tests `m == cands[j]`, branch to C_j else D_{j+1} (last → default) ---
    // Params of every D_j: live values, then the masked index `m` (at index nl).
    for (j, &target) in cands.iter().enumerate() {
        let insts = vec![
            Inst::ConstI32(target as i32), // @ nl+1
            Inst::IntCmp {
                ty: IntTy::I32,
                op: CmpOp::Eq,
                a: nl,     // m
                b: nl + 1, // the candidate const
            },
        ];
        let eq_val = nl + 2;
        // then → C_j with the live values (C_j's params are exactly `live`).
        let then_args: Vec<u32> = (0..nl).collect();
        let (else_blk, else_args): (u32, Vec<u32>) = if (j as u32) + 1 < s {
            (d_base + j as u32 + 1, (0..=nl).collect()) // forward live ++ m
        } else {
            (default_blk, Vec::new())
        };
        let mut params = live_types.clone();
        params.push(ValType::I32); // m
        new_blocks.push(Block {
            params,
            insts,
            term: Terminator::BrIf {
                cond: eq_val,
                then_blk: c_base + j as u32,
                then_args,
                else_blk,
                else_args,
            },
        });
    }

    // --- call arms: C_j = direct call to cands[j], then branch to the continuation ---
    // Params of C_j: the live values (indices [0, nl)); the call results land at [nl, nl+r).
    for &target in &cands {
        let call_args: Vec<u32> = args.iter().map(|a| live_pos[a]).collect();
        let cont_args: Vec<u32> = (0..nl + r as u32).collect(); // live ++ call results
        new_blocks.push(Block {
            params: live_types.clone(),
            insts: vec![Inst::Call {
                func: target,
                args: call_args,
            }],
            term: Terminator::Br {
                target: cont_blk,
                args: cont_args,
            },
        });
    }

    // --- default: no matching signature at the selected slot ⇒ the source's IndirectCallType trap
    // (raised here as `unreachable`; see the module-level trap-kind note). ---
    new_blocks.push(Block {
        params: Vec::new(),
        insts: Vec::new(),
        term: Terminator::Unreachable,
    });

    // --- continuation: the original tail, with operands remapped into CONT's value space.
    // CONT params: live values [0, nl), then the CI results [nl, nl+r). A tail operand `o` maps to
    // its live-param slot if it was a head value, else it shifts down by `head_count - nl` (the CI
    // results and every later tail value line up after the params). ---
    let remap = |o: u32| -> u32 {
        if o < head_count {
            live_pos[&o]
        } else {
            o - head_count + nl
        }
    };
    let mut cont_insts = tail_insts;
    for inst in &mut cont_insts {
        map_operands(inst, &mut |o| remap(o));
    }
    let mut cont_term = tail_term;
    map_term_operands(&mut cont_term, &mut |o| remap(o));
    let mut cont_params = live_types;
    cont_params.extend_from_slice(&ty.results);
    new_blocks.push(Block {
        params: cont_params,
        insts: cont_insts,
        term: cont_term,
    });

    f.blocks.extend(new_blocks);
}
