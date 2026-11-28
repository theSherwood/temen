//! Compile verified [`temen_ir::Module`] into flat register-window bytecode (`#3187` / `#3195`).
//!
//! ## Register layout
//! Each function gets a flat register window of `num_regs` 128-bit slots (`Reg`). Every block `bi`
//! owns a contiguous slice `[block_base[bi] .. block_base[bi] + n_vals)`:
//! - `0..b.params.len()` hold block parameters (written by incoming branch edges).
//! - Captures `(anc_blk, anc_val)` are resolved **statically** to `block_base[anc_blk] + anc_val`
//!   so entering a nested block copies zero captures at runtime.
//! - Each instruction that produces `k` values appends them sequentially after the block's params.

use temen_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func,
    FuncType, IToF, Inst, IntTy, IntUnOp, LoadOp, Module, Ordering as IrOrdering, StoreOp,
    Terminator, TypeEntry, VBitBinOp, VFloatBinOp, VIntBinOp, VShape, ValIdx,
};
use temen_kernel::TypeId;

/// Compilation failure (unsupported Stage-2/link-form op or malformed module).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum CompileError {
    EmptyFunc(u32),
    Unsupported(&'static str),
    InvalidTypeRef(u32),
    TooManyRegs(u32),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::EmptyFunc(idx) => write!(f, "function {idx} has no blocks"),
            CompileError::Unsupported(op) => write!(f, "unsupported op in temen-bytecode: {op}"),
            CompileError::InvalidTypeRef(ty) => write!(f, "invalid function type index {ty}"),
            CompileError::TooManyRegs(idx) => {
                write!(f, "function {idx} exceeds u32 register limit")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// One parallel-safe register move (`dst = src`) on a branch edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegMove {
    pub dst: u32,
    pub src: u32,
}

/// One compiled branch edge (target PC, edge moves, and whether it is a taken back-edge for
/// IR-anchored fuel/poll safepoints per Invariant 9).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct BranchEdge {
    pub target_pc: u32,
    pub moves_start: u32,
    pub moves_len: u16,
    pub is_back_edge: bool,
}

/// Flat bytecode instruction in a [`CompiledFunc`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum BcOp {
    // ---- Constants & refs ----
    ConstI32 {
        dst: u32,
        val: i32,
    },
    ConstI64 {
        dst: u32,
        val: i64,
    },
    ConstF32 {
        dst: u32,
        bits: u32,
    },
    ConstF64 {
        dst: u32,
        bits: u64,
    },
    ConstV128 {
        dst: u32,
        bytes: [u8; 16],
    },
    RefFunc {
        dst: u32,
        func: u32,
    },

    // ---- Scalar integers ----
    IntBin32 {
        dst: u32,
        op: BinOp,
        a: u32,
        b: u32,
    },
    IntBin64 {
        dst: u32,
        op: BinOp,
        a: u32,
        b: u32,
    },
    IntBinImm32 {
        dst: u32,
        op: BinOp,
        a: u32,
        imm: i32,
    },
    IntBinImm64 {
        dst: u32,
        op: BinOp,
        a: u32,
        imm: i64,
    },
    IntCmp32 {
        dst: u32,
        op: CmpOp,
        a: u32,
        b: u32,
    },
    IntCmp64 {
        dst: u32,
        op: CmpOp,
        a: u32,
        b: u32,
    },
    IntCmpImm32 {
        dst: u32,
        op: CmpOp,
        a: u32,
        imm: i32,
    },
    IntCmpImm64 {
        dst: u32,
        op: CmpOp,
        a: u32,
        imm: i64,
    },
    IntUn32 {
        dst: u32,
        op: IntUnOp,
        a: u32,
    },
    IntUn64 {
        dst: u32,
        op: IntUnOp,
        a: u32,
    },
    Eqz32 {
        dst: u32,
        a: u32,
    },
    Eqz64 {
        dst: u32,
        a: u32,
    },
    Convert {
        dst: u32,
        op: ConvOp,
        a: u32,
    },
    Select {
        dst: u32,
        cond: u32,
        a: u32,
        b: u32,
    },

    // ---- Scalar floats ----
    FBin32 {
        dst: u32,
        op: FBinOp,
        a: u32,
        b: u32,
    },
    FBin64 {
        dst: u32,
        op: FBinOp,
        a: u32,
        b: u32,
    },
    FUn32 {
        dst: u32,
        op: FUnOp,
        a: u32,
    },
    FUn64 {
        dst: u32,
        op: FUnOp,
        a: u32,
    },
    Fma32 {
        dst: u32,
        a: u32,
        b: u32,
        c: u32,
    },
    Fma64 {
        dst: u32,
        a: u32,
        b: u32,
        c: u32,
    },
    FCmp32 {
        dst: u32,
        op: FCmpOp,
        a: u32,
        b: u32,
    },
    FCmp64 {
        dst: u32,
        op: FCmpOp,
        a: u32,
        b: u32,
    },
    FToISat {
        dst: u32,
        op: FToI,
        a: u32,
    },
    FToITrap {
        dst: u32,
        op: FToI,
        a: u32,
    },
    IToF {
        dst: u32,
        op: IToF,
        a: u32,
    },
    Cast {
        dst: u32,
        op: CastOp,
        a: u32,
    },

    // ---- Linear memory ----
    Load {
        dst: u32,
        op: LoadOp,
        addr: u32,
        offset: u64,
    },
    Store {
        op: StoreOp,
        addr: u32,
        val: u32,
        offset: u64,
    },
    MemCopy {
        dst_addr: u32,
        src_addr: u32,
        len: u32,
    },
    MemMove {
        dst_addr: u32,
        src_addr: u32,
        len: u32,
    },
    MemFill {
        dst_addr: u32,
        val: u32,
        len: u32,
    },

    // ---- Atomics, futex, TLS ----
    AtomicLoad {
        dst: u32,
        ty: IntTy,
        addr: u32,
        offset: u64,
    },
    AtomicStore {
        ty: IntTy,
        addr: u32,
        val: u32,
        offset: u64,
    },
    AtomicRmw {
        dst: u32,
        ty: IntTy,
        op: AtomicRmwOp,
        addr: u32,
        val: u32,
        offset: u64,
    },
    AtomicCmpxchg {
        dst: u32,
        ty: IntTy,
        addr: u32,
        expected: u32,
        replacement: u32,
        offset: u64,
    },
    AtomicFence {
        order: IrOrdering,
    },
    MemoryWait {
        dst: u32,
        ty: IntTy,
        addr: u32,
        expected: u32,
        timeout: u32,
    },
    MemoryNotify {
        dst: u32,
        addr: u32,
        count: u32,
    },
    VcpuTlsGet {
        dst: u32,
    },
    VcpuTlsSet {
        val: u32,
    },

    // ---- Threads & fibers ----
    ThreadSpawn {
        dst: u32,
        func: u32,
        arg0: u32,
        arg1: u32,
    },
    ThreadJoin {
        dst: u32,
        handle: u32,
    },
    ContNew {
        dst: u32,
        func: u32,
        arg0: u32,
    },
    ContResume {
        dst: u32,
        k: u32,
        arg: u32,
        block: bool,
    },
    Suspend {
        dst: u32,
        val: u32,
    },

    // ---- Intra-Context setjmp/longjmp ----
    SetJmp {
        dst: u32,
        buf: u32,
    },
    LongJmp {
        buf: u32,
        val: u32,
    },

    // ---- Calls & capability dispatch ----
    Call {
        dst: u32,
        nresults: u8,
        func: u32,
        args_start: u32,
        args_len: u16,
    },
    CallIndirect {
        dst: u32,
        nresults: u8,
        expected_type: TypeId,
        idx: u32,
        args_start: u32,
        args_len: u16,
    },
    CapCall {
        dst: u32,
        nresults: u8,
        type_id: u32,
        op: u32,
        handle: u32,
        args_start: u32,
        args_len: u16,
    },

    // ---- Terminators ----
    Br {
        edge: u32,
    },
    BrIf {
        cond: u32,
        then_edge: u32,
        else_edge: u32,
    },
    BrTable {
        idx: u32,
        edges_start: u32,
        edges_len: u32,
        default_edge: u32,
    },
    Return {
        vals_start: u32,
        vals_len: u16,
    },
    ReturnCall {
        func: u32,
        args_start: u32,
        args_len: u16,
    },
    ReturnCallIndirect {
        expected_type: TypeId,
        idx: u32,
        args_start: u32,
        args_len: u16,
    },
    Unreachable,

    // ---- Core SIMD ----
    V128Load {
        dst: u32,
        addr: u32,
        offset: u64,
    },
    V128Store {
        addr: u32,
        val: u32,
        offset: u64,
    },
    Splat {
        dst: u32,
        shape: VShape,
        a: u32,
    },
    ExtractLane {
        dst: u32,
        shape: VShape,
        lane: u8,
        signed: bool,
        a: u32,
    },
    ReplaceLane {
        dst: u32,
        shape: VShape,
        lane: u8,
        a: u32,
        val: u32,
    },
    VIntBin {
        dst: u32,
        shape: VShape,
        op: VIntBinOp,
        a: u32,
        b: u32,
    },
    VFloatBin {
        dst: u32,
        shape: VShape,
        op: VFloatBinOp,
        a: u32,
        b: u32,
    },
    VBitBin {
        dst: u32,
        op: VBitBinOp,
        a: u32,
        b: u32,
    },
    VNot {
        dst: u32,
        a: u32,
    },
    Bitselect {
        dst: u32,
        a: u32,
        b: u32,
        mask: u32,
    },
    Shuffle {
        dst: u32,
        lanes: [u8; 16],
        a: u32,
        b: u32,
    },
    Swizzle {
        dst: u32,
        a: u32,
        b: u32,
    },
}

/// One compiled function in [`Code`].
#[derive(Clone, Debug)]
pub struct CompiledFunc {
    pub num_params: u16,
    pub num_results: u8,
    pub num_regs: u32,
    pub type_id: TypeId,
    pub ops: Vec<BcOp>,
    pub edges: Vec<BranchEdge>,
    pub moves: Vec<RegMove>,
    pub arg_regs: Vec<u32>,
}

/// Initial static data segment copied into a fresh domain window.
#[derive(Clone, Debug)]
pub struct InitialData {
    pub offset: u64,
    pub readonly: bool,
    pub bytes: Vec<u8>,
}

/// Compiled, immutable module artifact for `Engine::Bytecode`.
#[derive(Clone, Debug)]
pub struct Code {
    pub funcs: Vec<CompiledFunc>,
    pub func_type_ids: Vec<TypeId>,
    pub fn_table_mask: u32,
    pub memory_size_log2: Option<u8>,
    pub initial_data: Vec<InitialData>,
    pub exports: Vec<(String, u32)>,
}

impl Code {
    /// Look up an exported function index by name.
    pub fn export_func(&self, name: &str) -> Option<u32> {
        self.exports
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, idx)| *idx)
    }

    /// Resolve an indirect function-table index with §3c power-of-two masking and canonical
    /// 64-bit structural [`TypeId`] check.
    #[inline]
    pub fn resolve_indirect(&self, idx: i32, expected: TypeId) -> Option<u32> {
        if self.funcs.is_empty() {
            return None;
        }
        let slot = (idx as u32) & self.fn_table_mask;
        let &got = self.func_type_ids.get(slot as usize)?;
        (got == expected).then_some(slot)
    }
}

fn func_sig(m: &Module, ty: u32) -> Result<&FuncType, CompileError> {
    match m.types.get(ty as usize) {
        Some(TypeEntry::Func(ft)) => Ok(ft),
        _ => Err(CompileError::InvalidTypeRef(ty)),
    }
}

/// Compile a verified [`Module`] into [`Code`].
pub fn compile_module(m: &Module) -> Result<Code, CompileError> {
    let func_results: Vec<u8> = m.funcs.iter().map(|f| f.results.len() as u8).collect();
    let func_type_ids: Vec<TypeId> = m
        .funcs
        .iter()
        .map(|f| TypeId::of_sig(&f.params, &f.results))
        .collect();
    let mut funcs = Vec::with_capacity(m.funcs.len());
    for (idx, f) in m.funcs.iter().enumerate() {
        funcs.push(compile_func(
            m,
            idx as u32,
            f,
            &func_results,
            func_type_ids[idx],
        )?);
    }
    let fn_table_mask = (m.funcs.len().max(1).next_power_of_two() - 1) as u32;
    let initial_data = m
        .data
        .iter()
        .map(|d| InitialData {
            offset: d.offset,
            readonly: d.readonly,
            bytes: d.bytes.clone(),
        })
        .collect();
    let exports = m.exports.iter().map(|e| (e.name.clone(), e.func)).collect();

    Ok(Code {
        funcs,
        func_type_ids,
        fn_table_mask,
        memory_size_log2: m.memory.as_ref().map(|mem| mem.size_log2),
        initial_data,
        exports,
    })
}

fn inst_result_count(m: &Module, inst: &Inst, func_results: &[u8]) -> Result<u32, CompileError> {
    Ok(match inst {
        Inst::Store { .. }
        | Inst::MemCopy { .. }
        | Inst::MemMove { .. }
        | Inst::MemFill { .. }
        | Inst::AtomicStore { .. }
        | Inst::AtomicFence { .. }
        | Inst::VcpuTlsSet { .. }
        | Inst::LongJmp { .. }
        | Inst::V128Store { .. } => 0,
        Inst::Call { func, .. } => u32::from(*func_results.get(*func as usize).unwrap_or(&0)),
        Inst::CallIndirect(c) => func_sig(m, c.ty)?.results.len() as u32,
        Inst::CapCall(c) => func_sig(m, c.sig)?.results.len() as u32,
        Inst::ContResume { .. } | Inst::ThreadJoin { .. } => 2,
        _ => 1,
    })
}

fn compile_func(
    m: &Module,
    func_idx: u32,
    f: &Func,
    func_results: &[u8],
    type_id: TypeId,
) -> Result<CompiledFunc, CompileError> {
    if f.blocks.is_empty() {
        return Err(CompileError::EmptyFunc(func_idx));
    }

    // Pass 1: assign a flat register index to every block-local ValIdx.
    // Captures alias the ancestor's register directly (zero runtime copies).
    let mut val_regs: Vec<Vec<u32>> = Vec::with_capacity(f.blocks.len());
    let mut param_bases: Vec<u32> = Vec::with_capacity(f.blocks.len());
    let mut next_reg: u32 = 0;

    for b in &f.blocks {
        let mut regs = Vec::with_capacity(b.params.len() + b.captures.len() + b.insts.len());
        param_bases.push(next_reg);
        for _ in 0..b.params.len() {
            regs.push(next_reg);
            next_reg = next_reg
                .checked_add(1)
                .ok_or(CompileError::TooManyRegs(func_idx))?;
        }
        for &(anc_blk, anc_val) in &b.captures {
            let r = val_regs[anc_blk as usize][anc_val as usize];
            regs.push(r);
        }
        for inst in &b.insts {
            let n = inst_result_count(m, inst, func_results)?;
            for _ in 0..n {
                regs.push(next_reg);
                next_reg = next_reg
                    .checked_add(1)
                    .ok_or(CompileError::TooManyRegs(func_idx))?;
            }
        }
        val_regs.push(regs);
    }

    // Pass 2: emit instructions per block, recording each block's starting PC and pending
    // branch edge targets to patch in Pass 3.
    let mut ops: Vec<BcOp> = Vec::new();
    let mut edges: Vec<BranchEdge> = Vec::new();
    let mut edge_targets: Vec<u32> = Vec::new();
    let mut moves: Vec<RegMove> = Vec::new();
    let mut arg_regs: Vec<u32> = Vec::new();
    let mut block_pcs: Vec<u32> = Vec::with_capacity(f.blocks.len());

    let push_args = |arg_regs: &mut Vec<u32>, map: &[u32], args: &[ValIdx]| -> (u32, u16) {
        let start = arg_regs.len() as u32;
        for &v in args {
            arg_regs.push(map[v as usize]);
        }
        (start, args.len() as u16)
    };

    let push_edge = |edges: &mut Vec<BranchEdge>,
                     edge_targets: &mut Vec<u32>,
                     moves: &mut Vec<RegMove>,
                     cur_blk: u32,
                     cur_map: &[u32],
                     target_blk: u32,
                     args: &[ValIdx]|
     -> u32 {
        let moves_start = moves.len() as u32;
        let dst_base = param_bases[target_blk as usize];
        for (i, &v) in args.iter().enumerate() {
            moves.push(RegMove {
                dst: dst_base + i as u32,
                src: cur_map[v as usize],
            });
        }
        let edge_idx = edges.len() as u32;
        edges.push(BranchEdge {
            target_pc: 0, // patched in Pass 3
            moves_start,
            moves_len: args.len() as u16,
            is_back_edge: target_blk <= cur_blk,
        });
        edge_targets.push(target_blk);
        edge_idx
    };

    for (bi, b) in f.blocks.iter().enumerate() {
        block_pcs.push(ops.len() as u32);
        let map = &val_regs[bi];
        let mut cursor = b.params.len() + b.captures.len();
        let r = |v: ValIdx| map[v as usize];

        for inst in &b.insts {
            let nres = inst_result_count(m, inst, func_results)?;
            let dst = if nres > 0 { map[cursor] } else { 0 };
            cursor += nres as usize;

            let bc = match inst {
                Inst::ConstI32(val) => BcOp::ConstI32 { dst, val: *val },
                Inst::ConstI64(val) => BcOp::ConstI64 { dst, val: *val },
                Inst::ConstF32(bits) => BcOp::ConstF32 { dst, bits: *bits },
                Inst::ConstF64(bits) => BcOp::ConstF64 { dst, bits: *bits },
                Inst::ConstV128(bytes) => BcOp::ConstV128 { dst, bytes: *bytes },
                Inst::RefFunc { func } => BcOp::RefFunc { dst, func: *func },

                Inst::IntBin {
                    ty: IntTy::I32,
                    op,
                    a,
                    b,
                } => BcOp::IntBin32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::IntBin {
                    ty: IntTy::I64,
                    op,
                    a,
                    b,
                } => BcOp::IntBin64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::IntBinImm {
                    ty: IntTy::I32,
                    op,
                    a,
                    imm,
                } => BcOp::IntBinImm32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    imm: *imm as i32,
                },
                Inst::IntBinImm {
                    ty: IntTy::I64,
                    op,
                    a,
                    imm,
                } => BcOp::IntBinImm64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    imm: *imm,
                },
                Inst::IntCmp {
                    ty: IntTy::I32,
                    op,
                    a,
                    b,
                } => BcOp::IntCmp32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::IntCmp {
                    ty: IntTy::I64,
                    op,
                    a,
                    b,
                } => BcOp::IntCmp64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::IntCmpImm {
                    ty: IntTy::I32,
                    op,
                    a,
                    imm,
                } => BcOp::IntCmpImm32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    imm: *imm as i32,
                },
                Inst::IntCmpImm {
                    ty: IntTy::I64,
                    op,
                    a,
                    imm,
                } => BcOp::IntCmpImm64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    imm: *imm,
                },
                Inst::IntUn {
                    ty: IntTy::I32,
                    op,
                    a,
                } => BcOp::IntUn32 {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::IntUn {
                    ty: IntTy::I64,
                    op,
                    a,
                } => BcOp::IntUn64 {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::Eqz { ty: IntTy::I32, a } => BcOp::Eqz32 { dst, a: r(*a) },
                Inst::Eqz { ty: IntTy::I64, a } => BcOp::Eqz64 { dst, a: r(*a) },
                Inst::Convert { op, a } => BcOp::Convert {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::Select { cond, a, b } => BcOp::Select {
                    dst,
                    cond: r(*cond),
                    a: r(*a),
                    b: r(*b),
                },

                Inst::FBin {
                    ty: FloatTy::F32,
                    op,
                    a,
                    b,
                } => BcOp::FBin32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::FBin {
                    ty: FloatTy::F64,
                    op,
                    a,
                    b,
                } => BcOp::FBin64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::FUn {
                    ty: FloatTy::F32,
                    op,
                    a,
                } => BcOp::FUn32 {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::FUn {
                    ty: FloatTy::F64,
                    op,
                    a,
                } => BcOp::FUn64 {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::Fma {
                    ty: FloatTy::F32,
                    a,
                    b,
                    c,
                } => BcOp::Fma32 {
                    dst,
                    a: r(*a),
                    b: r(*b),
                    c: r(*c),
                },
                Inst::Fma {
                    ty: FloatTy::F64,
                    a,
                    b,
                    c,
                } => BcOp::Fma64 {
                    dst,
                    a: r(*a),
                    b: r(*b),
                    c: r(*c),
                },
                Inst::FCmp {
                    ty: FloatTy::F32,
                    op,
                    a,
                    b,
                } => BcOp::FCmp32 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::FCmp {
                    ty: FloatTy::F64,
                    op,
                    a,
                    b,
                } => BcOp::FCmp64 {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::FToISat { op, a } => BcOp::FToISat {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::FToITrap { op, a } => BcOp::FToITrap {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::IToFConv { op, a } => BcOp::IToF {
                    dst,
                    op: *op,
                    a: r(*a),
                },
                Inst::Cast { op, a } => BcOp::Cast {
                    dst,
                    op: *op,
                    a: r(*a),
                },

                Inst::Load { op, addr, offset } => BcOp::Load {
                    dst,
                    op: *op,
                    addr: r(*addr),
                    offset: *offset,
                },
                Inst::Store {
                    op,
                    addr,
                    value,
                    offset,
                } => BcOp::Store {
                    op: *op,
                    addr: r(*addr),
                    val: r(*value),
                    offset: *offset,
                },
                Inst::MemCopy { dst: d, src, len } => BcOp::MemCopy {
                    dst_addr: r(*d),
                    src_addr: r(*src),
                    len: r(*len),
                },
                Inst::MemMove { dst: d, src, len } => BcOp::MemMove {
                    dst_addr: r(*d),
                    src_addr: r(*src),
                    len: r(*len),
                },
                Inst::MemFill { dst: d, val, len } => BcOp::MemFill {
                    dst_addr: r(*d),
                    val: r(*val),
                    len: r(*len),
                },

                Inst::AtomicLoad { ty, addr, offset } => BcOp::AtomicLoad {
                    dst,
                    ty: *ty,
                    addr: r(*addr),
                    offset: *offset,
                },
                Inst::AtomicStore {
                    ty,
                    addr,
                    value,
                    offset,
                } => BcOp::AtomicStore {
                    ty: *ty,
                    addr: r(*addr),
                    val: r(*value),
                    offset: *offset,
                },
                Inst::AtomicRmw {
                    ty,
                    op,
                    addr,
                    value,
                    offset,
                } => BcOp::AtomicRmw {
                    dst,
                    ty: *ty,
                    op: *op,
                    addr: r(*addr),
                    val: r(*value),
                    offset: *offset,
                },
                Inst::AtomicCmpxchg {
                    ty,
                    addr,
                    expected,
                    replacement,
                    offset,
                } => BcOp::AtomicCmpxchg {
                    dst,
                    ty: *ty,
                    addr: r(*addr),
                    expected: r(*expected),
                    replacement: r(*replacement),
                    offset: *offset,
                },
                Inst::AtomicFence { order } => BcOp::AtomicFence { order: *order },
                Inst::MemoryWait {
                    ty,
                    addr,
                    expected,
                    timeout,
                } => BcOp::MemoryWait {
                    dst,
                    ty: *ty,
                    addr: r(*addr),
                    expected: r(*expected),
                    timeout: r(*timeout),
                },
                Inst::MemoryNotify { addr, count } => BcOp::MemoryNotify {
                    dst,
                    addr: r(*addr),
                    count: r(*count),
                },
                Inst::VcpuTlsGet => BcOp::VcpuTlsGet { dst },
                Inst::VcpuTlsSet { val } => BcOp::VcpuTlsSet { val: r(*val) },

                Inst::ThreadSpawn { func, arg0, arg1 } => BcOp::ThreadSpawn {
                    dst,
                    func: *func,
                    arg0: r(*arg0),
                    arg1: r(*arg1),
                },
                Inst::ThreadJoin { handle } => BcOp::ThreadJoin {
                    dst,
                    handle: r(*handle),
                },
                Inst::ContNew { func, arg0 } => BcOp::ContNew {
                    dst,
                    func: r(*func),
                    arg0: r(*arg0),
                },
                Inst::ContResume { k, arg, block } => BcOp::ContResume {
                    dst,
                    k: r(*k),
                    arg: r(*arg),
                    block: *block,
                },
                Inst::Suspend { value } => BcOp::Suspend {
                    dst,
                    val: r(*value),
                },
                Inst::SetJmp { buf } => BcOp::SetJmp { dst, buf: r(*buf) },
                Inst::LongJmp { buf, val } => BcOp::LongJmp {
                    buf: r(*buf),
                    val: r(*val),
                },

                Inst::Call { func, args } => {
                    let (args_start, args_len) = push_args(&mut arg_regs, map, args);
                    BcOp::Call {
                        dst,
                        nresults: nres as u8,
                        func: *func,
                        args_start,
                        args_len,
                    }
                }
                Inst::CallIndirect(c) => {
                    let ft = func_sig(m, c.ty)?;
                    let (args_start, args_len) = push_args(&mut arg_regs, map, &c.args);
                    BcOp::CallIndirect {
                        dst,
                        nresults: ft.results.len() as u8,
                        expected_type: TypeId::of_func_type(ft),
                        idx: r(c.idx),
                        args_start,
                        args_len,
                    }
                }
                Inst::CapCall(c) => {
                    let ft = func_sig(m, c.sig)?;
                    let (args_start, args_len) = push_args(&mut arg_regs, map, &c.args);
                    BcOp::CapCall {
                        dst,
                        nresults: ft.results.len() as u8,
                        type_id: c.type_id,
                        op: c.op,
                        handle: r(c.handle),
                        args_start,
                        args_len,
                    }
                }

                // Core SIMD
                Inst::V128Load { addr, offset } => BcOp::V128Load {
                    dst,
                    addr: r(*addr),
                    offset: *offset,
                },
                Inst::V128Store {
                    addr,
                    value,
                    offset,
                } => BcOp::V128Store {
                    addr: r(*addr),
                    val: r(*value),
                    offset: *offset,
                },
                Inst::Splat { shape, a } => BcOp::Splat {
                    dst,
                    shape: *shape,
                    a: r(*a),
                },
                Inst::ExtractLane {
                    shape,
                    lane,
                    signed,
                    a,
                } => BcOp::ExtractLane {
                    dst,
                    shape: *shape,
                    lane: *lane,
                    signed: *signed,
                    a: r(*a),
                },
                Inst::ReplaceLane { shape, lane, a, b } => BcOp::ReplaceLane {
                    dst,
                    shape: *shape,
                    lane: *lane,
                    a: r(*a),
                    val: r(*b),
                },
                Inst::VIntBin { shape, op, a, b } => BcOp::VIntBin {
                    dst,
                    shape: *shape,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::VFloatBin { shape, op, a, b } => BcOp::VFloatBin {
                    dst,
                    shape: *shape,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::VBitBin { op, a, b } => BcOp::VBitBin {
                    dst,
                    op: *op,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::VNot { a } => BcOp::VNot { dst, a: r(*a) },
                Inst::Bitselect { a, b, mask } => BcOp::Bitselect {
                    dst,
                    a: r(*a),
                    b: r(*b),
                    mask: r(*mask),
                },
                Inst::Shuffle { lanes, a, b } => BcOp::Shuffle {
                    dst,
                    lanes: **lanes,
                    a: r(*a),
                    b: r(*b),
                },
                Inst::Swizzle { a, b } => BcOp::Swizzle {
                    dst,
                    a: r(*a),
                    b: r(*b),
                },

                // Link-form / Stage-2 / extended SIMD ops fail closed in Stage 1b.
                Inst::CallSym(_)
                | Inst::DataSym(_)
                | Inst::DataSelf { .. }
                | Inst::DataTop
                | Inst::RefSym { .. } => return Err(CompileError::Unsupported("link-form op")),
                Inst::CallImport(_)
                | Inst::CallImportDyn(_)
                | Inst::ExportHandle { .. }
                | Inst::ImportAttach { .. }
                | Inst::CapSelfTypeId { .. }
                | Inst::CapSelfCovers { .. }
                | Inst::DurableShadowBase => return Err(CompileError::Unsupported("stage2-op")),
                _ => return Err(CompileError::Unsupported("extended-simd-op")),
            };
            ops.push(bc);
        }

        let term = match &b.term {
            Terminator::Br { target, args } => {
                let edge = push_edge(
                    &mut edges,
                    &mut edge_targets,
                    &mut moves,
                    bi as u32,
                    map,
                    *target,
                    args,
                );
                BcOp::Br { edge }
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let then_edge = push_edge(
                    &mut edges,
                    &mut edge_targets,
                    &mut moves,
                    bi as u32,
                    map,
                    *then_blk,
                    then_args,
                );
                let else_edge = push_edge(
                    &mut edges,
                    &mut edge_targets,
                    &mut moves,
                    bi as u32,
                    map,
                    *else_blk,
                    else_args,
                );
                BcOp::BrIf {
                    cond: r(*cond),
                    then_edge,
                    else_edge,
                }
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let edges_start = edges.len() as u32;
                for (t_blk, t_args) in targets {
                    push_edge(
                        &mut edges,
                        &mut edge_targets,
                        &mut moves,
                        bi as u32,
                        map,
                        *t_blk,
                        t_args,
                    );
                }
                let default_edge = push_edge(
                    &mut edges,
                    &mut edge_targets,
                    &mut moves,
                    bi as u32,
                    map,
                    default.0,
                    &default.1,
                );
                BcOp::BrTable {
                    idx: r(*idx),
                    edges_start,
                    edges_len: targets.len() as u32,
                    default_edge,
                }
            }
            Terminator::Return(vals) => {
                let (vals_start, vals_len) = push_args(&mut arg_regs, map, vals);
                BcOp::Return {
                    vals_start,
                    vals_len,
                }
            }
            Terminator::ReturnCall { func, args } => {
                let (args_start, args_len) = push_args(&mut arg_regs, map, args);
                BcOp::ReturnCall {
                    func: *func,
                    args_start,
                    args_len,
                }
            }
            Terminator::ReturnCallIndirect { ty, idx, args } => {
                let ft = func_sig(m, *ty)?;
                let (args_start, args_len) = push_args(&mut arg_regs, map, args);
                BcOp::ReturnCallIndirect {
                    expected_type: TypeId::of_func_type(ft),
                    idx: r(*idx),
                    args_start,
                    args_len,
                }
            }
            Terminator::Unreachable => BcOp::Unreachable,
        };
        ops.push(term);
    }

    // Pass 3: patch branch target PCs now that every block's PC is known.
    for (edge, target_blk) in edges.iter_mut().zip(edge_targets.iter()) {
        edge.target_pc = block_pcs[*target_blk as usize];
    }

    Ok(CompiledFunc {
        num_params: f.params.len() as u16,
        num_results: f.results.len() as u8,
        num_regs: next_reg,
        type_id,
        ops,
        edges,
        moves,
        arg_regs,
    })
}
