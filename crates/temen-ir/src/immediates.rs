//! Immediate operands (#2585): an integer op whose operand is a constant of its own block carries
//! the constant ([`Inst::IntBinImm`], [`Inst::IntCmpImm`]) instead of naming it, and the constant,
//! read by nothing else, goes. Fewer instructions to load, and fewer for the interpreters to run.
//!
//! Producers run [`Module::fold_immediates`] after [`Module::raise`]. Their own lowerings keep
//! emitting a constant and the register form.

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    Block, BlockIdx, DebugInfo, Func, Inst, IntTy, Module, SsaLoc, TypeEntry, ValIdx, VarLoc,
};

/// Where [`Func::fold_immediates`] moved each block's values and instructions: the constants it
/// deleted are gone, and what followed each moved down.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Folded(Vec<FoldedBlock>);

#[derive(Clone, Debug, PartialEq)]
struct FoldedBlock {
    /// Each old value's new index, or `None` for a deleted constant.
    values: Vec<Option<ValIdx>>,
    /// Whether each old instruction was deleted.
    dead: Vec<bool>,
    /// For each old instruction, and one past the last (the terminator), how many kept
    /// instructions precede it: its new index, if it was kept.
    before: Vec<u32>,
}

impl FoldedBlock {
    /// The new index of old instruction `i`, or `None` if it was deleted.
    fn inst(&self, i: u32) -> Option<u32> {
        match self.dead.get(i as usize) {
            Some(true) => None,
            _ => Some(self.from(i)),
        }
    }

    /// The new index of the first kept instruction at or after old instruction `i`.
    fn from(&self, i: u32) -> u32 {
        let last = self.before.len() - 1;
        match self.before.get(i as usize) {
            Some(&n) => n,
            None => self.before[last] + (i - last as u32),
        }
    }

    /// The new index of old value `v`, or `None` if it was deleted.
    fn value(&self, v: ValIdx) -> Option<ValIdx> {
        match self.values.get(v as usize) {
            Some(&n) => n,
            None => Some(v),
        }
    }
}

impl Folded {
    /// Whether the pass deleted nothing.
    pub fn is_empty(&self) -> bool {
        self.0.iter().all(|b| !b.dead.contains(&true))
    }
}

impl Func {
    /// Fold each integer op's constant operand into the op ([`Inst::IntBinImm`],
    /// [`Inst::IntCmpImm`]) where the constant is an instruction of the op's own block, then delete
    /// the constants nothing reads any more. A constant on the left moves right where the op allows:
    /// a commutative op swaps its operands, and a compare swaps them and turns around
    /// ([`CmpOp::swapped`](crate::CmpOp::swapped)). A constant `keep` names stays (debug info's).
    ///
    /// Semantics-preserving: an immediate is the constant it replaces. The output verifies wherever
    /// the input did. `fn_results` and `types` count each instruction's results, as
    /// [`Inst::result_count`] does.
    pub fn fold_immediates(
        &mut self,
        fn_results: &[usize],
        types: &[TypeEntry],
        keep: impl Fn(BlockIdx, ValIdx) -> bool,
    ) -> Folded {
        // Each block's result count per instruction, and the value index of each one's first.
        let counts: Vec<Vec<u32>> = (self.blocks.iter())
            .map(|b| {
                (b.insts.iter())
                    .map(|i| i.result_count(fn_results, types) as u32)
                    .collect()
            })
            .collect();
        let firsts: Vec<(Vec<ValIdx>, ValIdx)> = (self.blocks.iter().zip(&counts))
            .map(|(b, counts)| first_results(b, counts))
            .collect();

        // Fold: an operand a constant of this block defines becomes the op's immediate.
        for (b, (first, n)) in self.blocks.iter_mut().zip(&firsts) {
            let mut constant: Vec<Option<(IntTy, i64)>> = vec![None; *n as usize];
            for (inst, &v) in b.insts.iter().zip(first) {
                match *inst {
                    Inst::ConstI32(c) => constant[v as usize] = Some((IntTy::I32, i64::from(c))),
                    Inst::ConstI64(c) => constant[v as usize] = Some((IntTy::I64, c)),
                    _ => {}
                }
            }
            let k = |v: ValIdx, ty: IntTy| match constant.get(v as usize) {
                Some(&Some((t, c))) if t == ty => Some(c),
                _ => None,
            };
            for inst in &mut b.insts {
                *inst = match *inst {
                    Inst::IntBin { ty, op, a, b } => match (k(a, ty), k(b, ty)) {
                        (_, Some(imm)) => Inst::IntBinImm { ty, op, a, imm },
                        (Some(imm), None) if op.commutes() => Inst::IntBinImm { ty, op, a: b, imm },
                        _ => continue,
                    },
                    Inst::IntCmp { ty, op, a, b } => match (k(a, ty), k(b, ty)) {
                        (_, Some(imm)) => Inst::IntCmpImm { ty, op, a, imm },
                        (Some(imm), None) => Inst::IntCmpImm {
                            ty,
                            op: op.swapped(),
                            a: b,
                            imm,
                        },
                        _ => continue,
                    },
                    _ => continue,
                };
            }
        }

        // What is still read: by an instruction, a terminator, or another block's capture.
        let mut read: Vec<Vec<bool>> = firsts
            .iter()
            .map(|(_, n)| vec![false; *n as usize])
            .collect();
        let mut mark = |b: usize, v: ValIdx| {
            if let Some(r) = read.get_mut(b).and_then(|r| r.get_mut(v as usize)) {
                *r = true;
            }
        };
        for (bi, b) in self.blocks.iter().enumerate() {
            for &(a, v) in &b.captures {
                mark(a as usize, v);
            }
            for inst in &b.insts {
                inst.for_each_operand(&mut |&v| mark(bi, v));
            }
            b.term.for_each_operand(&mut |&v| mark(bi, v));
        }

        // Delete the constants nothing reads, and renumber what follows them.
        let mut folded = Vec::with_capacity(self.blocks.len());
        for (bi, (b, ((first, n), counts))) in
            (self.blocks.iter_mut().zip(firsts.iter().zip(&counts))).enumerate()
        {
            let dead: Vec<bool> = (b.insts.iter().zip(first))
                .map(|(inst, &v)| {
                    matches!(inst, Inst::ConstI32(_) | Inst::ConstI64(_))
                        && !read[bi][v as usize]
                        && !keep(bi as BlockIdx, v)
                })
                .collect();
            let inputs = (b.params.len() + b.captures.len()) as ValIdx;
            let mut values: Vec<Option<ValIdx>> = Vec::with_capacity(*n as usize);
            values.extend((0..inputs).map(Some));
            let mut before = Vec::with_capacity(dead.len() + 1);
            let (mut kept, mut next) = (0u32, inputs);
            for (&d, &count) in dead.iter().zip(counts) {
                before.push(kept);
                for _ in 0..count {
                    values.push((!d).then_some(next));
                    next += !d as ValIdx;
                }
                kept += !d as u32;
            }
            before.push(kept);
            if dead.contains(&true) {
                let mut d = dead.iter();
                b.insts.retain(|_| !d.next().copied().unwrap_or(false));
                let mut to = |v: &mut ValIdx| {
                    if let Some(&Some(n)) = values.get(*v as usize) {
                        *v = n;
                    }
                };
                for inst in &mut b.insts {
                    inst.for_each_operand_mut(&mut to);
                }
                b.term.for_each_operand_mut(&mut to);
            }
            folded.push(FoldedBlock {
                values,
                dead,
                before,
            });
        }
        // A capture names its source's value where that value went.
        for b in &mut self.blocks {
            for (a, v) in &mut b.captures {
                if let Some(n) = folded.get(*a as usize).and_then(|f| f.value(*v)) {
                    *v = n;
                }
            }
        }
        Folded(folded)
    }
}

/// The value index of each of `b`'s instructions' first result, given their result `counts`, and
/// the block's value count.
fn first_results(b: &Block, counts: &[u32]) -> (Vec<ValIdx>, ValIdx) {
    let mut v = (b.params.len() + b.captures.len()) as ValIdx;
    let mut first = Vec::with_capacity(counts.len());
    for &n in counts {
        first.push(v);
        v += n;
    }
    (first, v)
}

impl Module {
    /// [`Func::fold_immediates`] every function, keeping the constants debug info names a variable
    /// by, and re-point the debug info at where each function's instructions and values went.
    pub fn fold_immediates(&mut self) {
        let fn_results: Vec<usize> = self.funcs.iter().map(|f| f.results.len()).collect();
        let named = named_values(self.debug_info.as_ref(), self.funcs.len());
        let folded: Vec<Folded> = (self.funcs.iter_mut().zip(&named))
            .map(|(f, named)| f.fold_immediates(&fn_results, &self.types, |b, v| named.holds(b, v)))
            .collect();
        if let Some(di) = &mut self.debug_info {
            remap_debug(di, &folded);
        }
    }
}

/// The values a function's variables live in, per its debug info.
#[derive(Default)]
struct Named {
    /// `(block, value)` pairs, sorted.
    at: Vec<(BlockIdx, ValIdx)>,
    /// Values named in every block ([`VarLoc::Ssa`]), sorted.
    everywhere: Vec<ValIdx>,
}

impl Named {
    fn holds(&self, b: BlockIdx, v: ValIdx) -> bool {
        self.at.binary_search(&(b, v)).is_ok() || self.everywhere.binary_search(&v).is_ok()
    }
}

/// Each of `funcs` functions' [`Named`] values.
fn named_values(di: Option<&DebugInfo>, funcs: usize) -> Vec<Named> {
    let mut named: Vec<Named> = (0..funcs).map(|_| Named::default()).collect();
    for var in di.iter().flat_map(|di| &di.vars) {
        let Some(n) = named.get_mut(var.func as usize) else {
            continue;
        };
        match &var.loc {
            VarLoc::Ssa { value } => n.everywhere.push(*value),
            VarLoc::SsaList(locs) | VarLoc::WindowVia { base: locs, .. } => {
                n.at.extend(locs.iter().map(|l| (l.block, l.value)));
            }
            _ => {}
        }
    }
    for n in &mut named {
        n.at.sort_unstable();
        n.everywhere.sort_unstable();
    }
    named
}

/// Re-point debug info at where [`Func::fold_immediates`] moved each function's instructions and
/// values (`folded[func]`). A deleted constant's source location goes with it. A variable location
/// stays on its value, which was kept; one that started at a deleted constant starts at the next
/// instruction.
fn remap_debug(di: &mut DebugInfo, folded: &[Folded]) {
    let block = |func: u32, b: u32| {
        folded
            .get(func as usize)
            .filter(|f| !f.is_empty())
            .and_then(|f| f.0.get(b as usize))
    };
    di.locs
        .retain_mut(|l| match block(l.func, l.block).map(|b| b.inst(l.inst)) {
            Some(Some(i)) => {
                l.inst = i;
                true
            }
            Some(None) => false,
            None => true,
        });
    for var in &mut di.vars {
        let func = var.func;
        let remap = |locs: &mut Vec<SsaLoc>| {
            for l in locs.iter_mut() {
                if let Some(b) = block(func, l.block) {
                    l.inst = b.from(l.inst);
                    l.value = b.value(l.value).unwrap_or(l.value);
                }
            }
        };
        match &mut var.loc {
            // One value index for every block: where it moved, it becomes a list.
            VarLoc::Ssa { value } => {
                let value = *value;
                let Some(f) = folded.get(func as usize).filter(|f| !f.is_empty()) else {
                    continue;
                };
                if f.0.iter().any(|b| b.value(value) != Some(value)) {
                    var.loc = VarLoc::SsaList(
                        (f.0.iter().enumerate())
                            .filter_map(|(block, b)| {
                                b.value(value).map(|value| SsaLoc {
                                    block: block as BlockIdx,
                                    inst: 0,
                                    value,
                                })
                            })
                            .collect(),
                    );
                }
            }
            VarLoc::SsaList(locs) => remap(locs),
            VarLoc::WindowVia { base, .. } => remap(base),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BinOp, CmpOp, DebugInfo, Loc, Terminator, ValType, VarInfo};
    use alloc::string::String;

    fn block(
        params: Vec<ValType>,
        captures: Vec<(BlockIdx, ValIdx)>,
        insts: Vec<Inst>,
        term: Terminator,
    ) -> Block {
        Block {
            parent: 0,
            captures,
            params,
            insts,
            term,
        }
    }

    fn bin(ty: IntTy, op: BinOp, a: ValIdx, b: ValIdx) -> Inst {
        Inst::IntBin { ty, op, a, b }
    }

    /// A right constant, a commutative op's left constant and a compare's left constant fold; a
    /// `sub` from a constant does not. The constants nothing reads go, and the values after them
    /// move down.
    #[test]
    fn a_constant_of_the_block_folds_and_goes() {
        let i32 = IntTy::I32;
        let mut f = Func {
            params: vec![ValType::I32],
            results: vec![ValType::I32],
            blocks: vec![block(
                vec![ValType::I32],
                vec![],
                vec![
                    Inst::ConstI32(5),          // v1
                    bin(i32, BinOp::Add, 0, 1), // v2 = v0 + 5
                    Inst::ConstI32(7),          // v3
                    bin(i32, BinOp::Sub, 3, 0), // v4 = 7 - v0: stays
                    Inst::ConstI32(-2),         // v5
                    bin(i32, BinOp::Mul, 5, 2), // v6 = -2 * v2
                    Inst::ConstI32(10),         // v7
                    Inst::IntCmp {
                        ty: i32,
                        op: CmpOp::LtS,
                        a: 7,
                        b: 6,
                    }, // v8 = 10 < v6
                ],
                Terminator::Return(vec![8]),
            )],
        };
        let folded = f.fold_immediates(&[], &[], |_, _| false);
        assert!(!folded.is_empty());
        let b = &f.blocks[0];
        assert_eq!(
            b.insts,
            vec![
                Inst::IntBinImm {
                    ty: i32,
                    op: BinOp::Add,
                    a: 0,
                    imm: 5
                },
                Inst::ConstI32(7),
                bin(i32, BinOp::Sub, 2, 0),
                Inst::IntBinImm {
                    ty: i32,
                    op: BinOp::Mul,
                    a: 1,
                    imm: -2
                },
                Inst::IntCmpImm {
                    ty: i32,
                    op: CmpOp::GtS,
                    a: 4,
                    imm: 10
                },
            ]
        );
        assert_eq!(b.term, Terminator::Return(vec![5]));
    }

    /// A constant another block captures stays, though its own op folds it; one nothing reads
    /// goes, and a capture of a value after it follows the value.
    #[test]
    fn a_captured_constant_stays_and_a_capture_follows_its_value() {
        let i64 = IntTy::I64;
        let mut f = Func {
            params: vec![ValType::I64],
            results: vec![ValType::I64],
            blocks: vec![
                block(
                    vec![ValType::I64],
                    vec![],
                    vec![
                        Inst::ConstI64(8),          // v1: captured below
                        bin(i64, BinOp::Add, 0, 1), // v2
                        Inst::ConstI64(16),         // v3: goes
                        bin(i64, BinOp::Xor, 3, 2), // v4 = 16 ^ v2
                    ],
                    Terminator::Br {
                        target: 1,
                        args: vec![],
                    },
                ),
                block(
                    vec![],
                    vec![(0, 1), (0, 4)],
                    vec![bin(i64, BinOp::Add, 0, 1)],
                    Terminator::Return(vec![2]),
                ),
            ],
        };
        f.fold_immediates(&[], &[], |_, _| false);
        assert_eq!(
            f.blocks[0].insts,
            vec![
                Inst::ConstI64(8),
                Inst::IntBinImm {
                    ty: i64,
                    op: BinOp::Add,
                    a: 0,
                    imm: 8
                },
                Inst::IntBinImm {
                    ty: i64,
                    op: BinOp::Xor,
                    a: 2,
                    imm: 16
                },
            ]
        );
        assert_eq!(f.blocks[1].captures, vec![(0, 1), (0, 3)]);
    }

    /// Debug info follows: a deleted constant's line goes, later ones move down; a variable kept
    /// in a constant keeps it, and a variable location that began at a deleted constant begins at
    /// the next instruction.
    #[test]
    fn debug_info_follows_the_fold() {
        let i32 = IntTy::I32;
        let mut m = Module {
            funcs: vec![Func {
                params: vec![ValType::I32],
                results: vec![ValType::I32],
                blocks: vec![block(
                    vec![ValType::I32],
                    vec![],
                    vec![
                        Inst::ConstI32(3),          // v1: a variable's value; stays
                        bin(i32, BinOp::Shl, 0, 1), // v2
                        Inst::ConstI32(1),          // v3: goes
                        bin(i32, BinOp::Add, 2, 3), // v4
                    ],
                    Terminator::Return(vec![4]),
                )],
            }],
            ..Default::default()
        };
        let loc = |inst: u32| Loc {
            func: 0,
            block: 0,
            inst,
            file: 0,
            line: 10 + inst,
            col: 0,
        };
        let var = |name: &str, inst: u32, value: ValIdx| VarInfo {
            func: 0,
            name: String::from(name),
            ty: String::from("int"),
            loc: VarLoc::SsaList(vec![SsaLoc {
                block: 0,
                inst,
                value,
            }]),
            type_id: None,
            scope: None,
        };
        m.debug_info = Some(DebugInfo {
            locs: (0..=4).map(loc).collect(),
            vars: vec![var("k", 0, 1), var("sum", 2, 4)],
            ..Default::default()
        });
        m.fold_immediates();
        let b = &m.funcs[0].blocks[0];
        assert_eq!(
            b.insts,
            vec![
                Inst::ConstI32(3),
                Inst::IntBinImm {
                    ty: i32,
                    op: BinOp::Shl,
                    a: 0,
                    imm: 3
                },
                Inst::IntBinImm {
                    ty: i32,
                    op: BinOp::Add,
                    a: 2,
                    imm: 1
                },
            ]
        );
        let di = m.debug_info.as_ref().unwrap();
        let lines: Vec<(u32, u32)> = di.locs.iter().map(|l| (l.inst, l.line)).collect();
        assert_eq!(lines, [(0, 10), (1, 11), (2, 13), (3, 14)]);
        let at = |i: usize| match &di.vars[i].loc {
            VarLoc::SsaList(l) => (l[0].inst, l[0].value),
            other => panic!("{other:?}"),
        };
        assert_eq!(at(0), (0, 1), "the variable's constant stays");
        assert_eq!(
            at(1),
            (2, 3),
            "from the deleted constant on: the next instruction"
        );
    }
}
