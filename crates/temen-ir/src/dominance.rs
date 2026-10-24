//! Converting a function between block-local form and dominance form (#2454, DESIGN.md §3a).
//!
//! In block-local form a value crosses blocks only as a block param: every parent is the entry and
//! no block captures. In dominance form a block captures the values of blocks that dominate it, so
//! a value that lives long in a large function stops being threaded through every block between.
//!
//! [`Func::localize`] goes to block-local form, for the passes that reason one block at a time.
//! [`Func::raise`] goes to dominance form: a param that only forwards one value
//! ([`Func::forwards`]) becomes a capture of that value. Both report where each value went
//! ([`Moves`]), so debug info can follow it.

use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::{
    Block, BlockIdx, DebugInfo, Func, FuncIdx, SsaLoc, Terminator, ValIdx, ValType, VarLoc,
};

/// Where each block, and each value of each block, went when a pass renumbered a function: one
/// entry per old block. Empty when nothing moved.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Moves(Vec<BlockMoves>);

#[derive(Clone, Debug, PartialEq)]
struct BlockMoves {
    /// The block's new index, or `None` when the pass dropped it.
    to: Option<BlockIdx>,
    /// Where each of the block's old params and captures went, or `None` where it is gone.
    inputs: Vec<Option<ValIdx>>,
    /// How many params and captures the block has now: its first result's new index.
    new_inputs: u32,
}

impl Moves {
    /// Moves from each old block's new index (or `None` where it was dropped), where each of its
    /// params and captures went (or `None` where it is gone), and how many it has now.
    pub(crate) fn new(blocks: Vec<(Option<BlockIdx>, Vec<Option<ValIdx>>, ValIdx)>) -> Moves {
        Moves(
            blocks
                .into_iter()
                .map(|(to, inputs, new_inputs)| BlockMoves {
                    to,
                    inputs,
                    new_inputs,
                })
                .collect(),
        )
    }

    /// Whether the pass moved nothing.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Where old block `b`'s value `v` went, as its new block and index, or `None` when it is gone.
    pub fn value(&self, b: BlockIdx, v: ValIdx) -> Option<(BlockIdx, ValIdx)> {
        let Some(m) = self.0.get(b as usize) else {
            return Some((b, v));
        };
        let to = m.to?;
        match m.inputs.get(v as usize) {
            Some(&nv) => nv.map(|nv| (to, nv)),
            None => Some((to, v - m.inputs.len() as ValIdx + m.new_inputs)),
        }
    }

    /// Re-point function `func`'s block- and value-keyed debug info at where the pass moved them,
    /// so a debugger reads the variable it read before. A location that went with its value or
    /// block is dropped (the variable is not live there). A function-wide [`VarLoc::Ssa`] names one
    /// index in every block (a φ-model slot), which no longer holds, so it becomes the list of where
    /// that slot now sits.
    pub fn remap_debug(&self, di: &mut DebugInfo, func: FuncIdx) {
        if self.is_empty() {
            return;
        }
        di.locs.retain_mut(|l| {
            if l.func != func {
                return true;
            }
            match self.0.get(l.block as usize).map(|m| m.to) {
                Some(Some(to)) => {
                    l.block = to;
                    true
                }
                Some(None) => false,
                None => true,
            }
        });
        let remap = |locs: &mut Vec<SsaLoc>| {
            locs.retain_mut(|l| match self.value(l.block, l.value) {
                Some((b, v)) => {
                    l.block = b;
                    l.value = v;
                    true
                }
                None => false,
            });
        };
        for var in di.vars.iter_mut().filter(|v| v.func == func) {
            match &mut var.loc {
                VarLoc::Ssa { value } => {
                    let value = *value;
                    var.loc = VarLoc::SsaList(
                        (0..self.0.len() as BlockIdx)
                            .filter_map(|b| {
                                self.value(b, value).map(|(block, value)| SsaLoc {
                                    block,
                                    inst: 0,
                                    value,
                                })
                            })
                            .collect(),
                    );
                }
                VarLoc::SsaList(locs) => remap(locs),
                VarLoc::WindowVia { base, .. } => remap(base),
                _ => {}
            }
        }
    }
}

impl Func {
    /// Rewrite this function into block-local form: each capture becomes a block param at the same
    /// index, a captured value is threaded as a param through every block between its own and the
    /// one capturing it, and every parent becomes the entry. `types` are the function's value types
    /// per block (temen-verify's `func_value_types`), for the params this adds. Assumes a verified
    /// function. A function with no captures only has its parents reset, and moves nothing.
    pub fn localize(&mut self, types: &[Vec<ValType>]) -> Moves {
        for b in &mut self.blocks {
            b.parent = 0;
        }
        if self.blocks.iter().all(|b| b.captures.is_empty()) {
            return Moves::default();
        }
        let n = self.blocks.len();
        // The definition each capture names (a param or a result), following a capture of a
        // capture to the value it takes: ancestors come first, so theirs are already resolved.
        let mut origin: Vec<Vec<(BlockIdx, ValIdx)>> = Vec::with_capacity(n);
        for b in &self.blocks {
            let o = b
                .captures
                .iter()
                .map(|&(a, v)| {
                    let blk = &self.blocks[a as usize];
                    match (v as usize).checked_sub(blk.params.len()) {
                        Some(j) if j < blk.captures.len() => origin[a as usize][j],
                        _ => (a, v),
                    }
                })
                .collect();
            origin.push(o);
        }
        // Number the captured definitions, and record each one's block.
        let mut id: BTreeMap<(BlockIdx, ValIdx), usize> = BTreeMap::new();
        for &o in origin.iter().flatten() {
            let k = id.len();
            id.entry(o).or_insert(k);
        }
        let mut def: Vec<(BlockIdx, ValIdx)> = vec![(0, 0); id.len()];
        for (&o, &k) in &id {
            def[k] = o;
        }
        // Which captured definitions each block needs on entry: its own captures', and every one a
        // successor needs that the block does not define. A backward fixpoint over bitsets.
        let words = id.len().div_ceil(64);
        let mut live = vec![0u64; n * words];
        let mut kill = vec![0u64; n * words];
        for (k, &(b, _)) in def.iter().enumerate() {
            kill[b as usize * words + k / 64] |= 1 << (k % 64);
        }
        for (b, os) in origin.iter().enumerate() {
            for o in os {
                let k = id[o];
                live[b * words + k / 64] |= 1 << (k % 64);
            }
        }
        let succs = Cfg::new(&self.blocks).succs;
        let mut changed = true;
        while changed {
            changed = false;
            for b in (0..n).rev() {
                for &s in &succs[b] {
                    for w in 0..words {
                        let flow = live[s as usize * words + w] & !kill[b * words + w];
                        let cur = &mut live[b * words + w];
                        if flow & !*cur != 0 {
                            *cur |= flow;
                            changed = true;
                        }
                    }
                }
            }
        }
        // Each block's added params: one per live definition its captures do not already bring.
        // `local[b]` maps a live definition to the param that holds it in `b`.
        let mut moves = Vec::with_capacity(n);
        let mut local: Vec<BTreeMap<usize, ValIdx>> = Vec::with_capacity(n);
        let mut added: Vec<Vec<usize>> = Vec::with_capacity(n);
        for (b, blk) in self.blocks.iter().enumerate() {
            let inputs = blk.inputs() as ValIdx;
            let mut here = BTreeMap::new();
            for (j, o) in origin[b].iter().enumerate() {
                here.entry(id[o])
                    .or_insert(blk.params.len() as ValIdx + j as ValIdx);
            }
            let mut more = Vec::new();
            for k in 0..id.len() {
                if live[b * words + k / 64] >> (k % 64) & 1 == 1 && !here.contains_key(&k) {
                    here.insert(k, inputs + more.len() as ValIdx);
                    more.push(k);
                }
            }
            moves.push(BlockMoves {
                to: Some(b as BlockIdx),
                inputs: (0..inputs).map(Some).collect(),
                new_inputs: inputs + more.len() as ValIdx,
            });
            local.push(here);
            added.push(more);
        }
        let moves = Moves(moves);
        // A value the block defines keeps its (shifted) index; a live definition from elsewhere is
        // the param holding it.
        let in_block = |b: usize, k: usize| -> ValIdx {
            let (db, dv) = def[k];
            if db as usize == b {
                moves
                    .value(db, dv)
                    .expect("a definition is never dropped")
                    .1
            } else {
                local[b][&k]
            }
        };
        let mut edge_args: Vec<Vec<Vec<ValIdx>>> = Vec::with_capacity(n);
        for (b, blk) in self.blocks.iter().enumerate() {
            edge_args.push(
                blk.term
                    .edges()
                    .iter()
                    .map(|&(s, _)| {
                        let s = s as usize;
                        origin[s]
                            .iter()
                            .map(|o| id[o])
                            .chain(added[s].iter().copied())
                            .map(|k| in_block(b, k))
                            .collect()
                    })
                    .collect(),
            );
        }
        for (b, blk) in self.blocks.iter_mut().enumerate() {
            let m = &moves.0[b];
            let first_result = blk.inputs() as ValIdx;
            let shift = m.new_inputs - first_result;
            let mut to = |v: &mut ValIdx| {
                if *v >= first_result {
                    *v += shift;
                }
            };
            for inst in &mut blk.insts {
                inst.for_each_operand_mut(&mut to);
            }
            blk.term.for_each_operand_mut(&mut to);
            let mut extra = edge_args[b].iter();
            for_each_edge_mut(&mut blk.term, |args| {
                args.extend_from_slice(extra.next().expect("one argument list per edge"));
            });
            let captured: Vec<ValType> = blk
                .captures
                .iter()
                .map(|&(a, v)| types[a as usize][v as usize])
                .collect();
            blk.params.extend(captured);
            blk.params.extend(
                added[b]
                    .iter()
                    .map(|&k| types[def[k].0 as usize][def[k].1 as usize]),
            );
            blk.captures.clear();
        }
        moves
    }

    /// Rewrite this function into dominance form: renumber its reachable blocks in reverse
    /// postorder, drop the unreachable ones, make each block's parent its immediate dominator, and
    /// turn every param that only forwards one value ([`Func::forwards`]) into a capture of that
    /// value — or into nothing, where the block only passed it on. Its edge arguments go with it.
    /// Assumes a verified function.
    pub fn raise(&mut self) -> Moves {
        let n = self.blocks.len();
        let cfg = Cfg::new(&self.blocks);
        let order = cfg.rpo();
        let idom = cfg.dominators();
        let fw = self.forwards();
        let np: Vec<usize> = self.blocks.iter().map(|b| b.params.len()).collect();
        let forwarded = |b: usize, i: usize| (i < np[b]).then(|| fw.source(b, i)).flatten();
        if order.len() == n
            && order.iter().enumerate().all(|(i, &b)| b as usize == i)
            && (0..n).all(|b| (0..np[b]).all(|i| forwarded(b, i).is_none()))
        {
            for (b, d) in self.blocks.iter_mut().zip(&idom) {
                b.parent = d.unwrap_or(0);
            }
            return Moves::default();
        }
        // An argument into a param that forwards goes with the param. What a block still reads
        // after that is what it needs.
        for b in &mut self.blocks {
            b.term
                .retain_edge_args(|s, k| forwarded(s as usize, k).is_none());
        }
        let mut pos: Vec<Option<BlockIdx>> = vec![None; n];
        for (i, &b) in order.iter().enumerate() {
            pos[b as usize] = Some(i as BlockIdx);
        }
        // Each block's captures: its own, each resolved through a forwarding param of its block,
        // then the source of every forwarding param it reads, once, in first-read order.
        let mut caps: Vec<Vec<(BlockIdx, ValIdx)>> = vec![Vec::new(); n];
        for &b in &order {
            let b = b as usize;
            let blk = &self.blocks[b];
            let mut c: Vec<(BlockIdx, ValIdx)> = blk
                .captures
                .iter()
                .map(|&(a, v)| forwarded(a as usize, v as usize).unwrap_or((a, v)))
                .collect();
            let mut note = |&v: &ValIdx| {
                if let Some(s) = forwarded(b, v as usize) {
                    if !c.contains(&s) {
                        c.push(s);
                    }
                }
            };
            for inst in &blk.insts {
                inst.for_each_operand(&mut note);
            }
            blk.term.for_each_operand(&mut note);
            caps[b] = c;
        }
        // Where each block's params and captures go: its own params first, then its captures.
        let moves = Moves(
            self.blocks
                .iter()
                .enumerate()
                .map(|(b, blk)| {
                    let own = (0..np[b]).filter(|&i| forwarded(b, i).is_none()).count() as ValIdx;
                    let mut next = 0;
                    let mut inputs: Vec<Option<ValIdx>> = (0..np[b])
                        .map(|i| match forwarded(b, i) {
                            None => {
                                next += 1;
                                Some(next - 1)
                            }
                            Some(s) => caps[b]
                                .iter()
                                .position(|&c| c == s)
                                .map(|j| own + j as ValIdx),
                        })
                        .collect();
                    inputs.extend((0..blk.captures.len() as ValIdx).map(|j| Some(own + j)));
                    BlockMoves {
                        to: pos[b],
                        inputs,
                        new_inputs: own + caps[b].len() as ValIdx,
                    }
                })
                .collect(),
        );
        let mut old: Vec<Option<Block>> = core::mem::take(&mut self.blocks)
            .into_iter()
            .map(Some)
            .collect();
        let new_index = |b: BlockIdx| pos[b as usize].expect("a reachable block's successor");
        for &b in &order {
            let bu = b as usize;
            let mut blk = old[bu].take().expect("each block once");
            let mut to = |v: &mut ValIdx| {
                *v = moves.value(b, *v).expect("a value still read is kept").1;
            };
            for inst in &mut blk.insts {
                inst.for_each_operand_mut(&mut to);
            }
            blk.term.for_each_operand_mut(&mut to);
            blk.term.for_each_target_mut(&mut |t| *t = new_index(*t));
            let mut params = blk.params.iter();
            blk.params = (0..np[bu])
                .filter_map(|i| {
                    let t = params.next().copied();
                    forwarded(bu, i).is_none().then_some(t).flatten()
                })
                .collect();
            blk.captures = caps[bu]
                .iter()
                .map(|&(a, v)| moves.value(a, v).expect("a captured value is kept"))
                .collect();
            blk.parent = idom[bu].map_or(0, new_index);
            self.blocks.push(blk);
        }
        moves
    }
}

/// Apply `f` to each edge's argument list of a terminator, in [`Terminator::edges`] order.
fn for_each_edge_mut(t: &mut Terminator, mut f: impl FnMut(&mut Vec<ValIdx>)) {
    match t {
        Terminator::Br { args, .. } => f(args),
        Terminator::BrIf {
            then_args,
            else_args,
            ..
        } => {
            f(then_args);
            f(else_args);
        }
        Terminator::BrTable {
            targets, default, ..
        } => {
            for (_, args) in targets.iter_mut() {
                f(args);
            }
            f(&mut default.1);
        }
        Terminator::Return(_)
        | Terminator::ReturnCall { .. }
        | Terminator::ReturnCallIndirect { .. }
        | Terminator::Unreachable => {}
    }
}
