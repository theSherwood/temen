//! Stackless register-window VM loop for `Engine::Bytecode` (`#3187` / `#3195`).
//!
//! Pure, `#![forbid(unsafe_code)]`, wasm32/wasm64-clean. Steps a single [`ExecState`] up to a
//! kernel boundary and returns [`temen_kernel::Step`]; never blocks OS threads or locks domain
//! tables.

use core::sync::atomic::Ordering;
use temen_ir::{IntTy, LoadOp, Ordering as IrOrdering, StoreOp};
use temen_kernel::sync::AtomicU64;
use temen_kernel::{
    ContextExit, FuelCell, PollFlags, Step, Trap, TypeId, YieldReason, MAX_CAP_ARGS,
};
use temen_mask::Window;
use temen_mem::Region;

use crate::compile::{BcOp, BranchEdge, Code, CompiledFunc};
use crate::ops::{
    bin32, bin64, cast, cmp32, cmp64, convert, extract_lane, fbin32, fbin64, fcmp32, fcmp64,
    fto_i_sat, fto_i_trap, fun32, fun64, i_to_f, intun32, intun64, replace_lane, rmw_op, splat,
    vbit_bin, vfloat_bin, vint_bin, vshuffle, vswizzle, Reg,
};

/// Default call-frame depth cap (prevents unbounded host stack/Vec growth on infinite recursion).
pub const DEFAULT_MAX_CALL_DEPTH: usize = 4096;

/// Per-step configuration passed by the Carrier.
#[derive(Clone, Copy, Debug)]
pub struct StepConfig {
    /// Maximum IR-anchored safepoints to run before yielding `YieldReason::Quantum`
    /// (`u32::MAX` for parallel Carriers that do not cooperatively time-slice).
    pub quantum: u32,
    /// Maximum call-frame nesting depth before trapping `Trap::StackOverflow`.
    pub max_call_depth: usize,
}

impl Default for StepConfig {
    fn default() -> StepConfig {
        StepConfig {
            quantum: 4096,
            max_call_depth: DEFAULT_MAX_CALL_DEPTH,
        }
    }
}

/// Borrowed view of a domain's linear memory window for one VM step.
pub struct MemView<'a> {
    pub region: Option<&'a Region>,
    pub window: Option<Window>,
    /// Inclusive-exclusive `[start, end)` byte ranges marked read-only (data segments).
    pub readonly_ranges: &'a [(u64, u64)],
    /// Invariant 13 NULL-guard extent (`[0, null_guard)` is unmapped and traps `MemoryFault`).
    pub null_guard: u64,
}

impl<'a> MemView<'a> {
    pub const NONE: MemView<'static> = MemView {
        region: None,
        window: None,
        readonly_ranges: &[],
        null_guard: 0,
    };

    #[inline]
    fn check_read(&self, addr: u64, offset: u64, width: u32) -> Result<(&'a Region, u64), Trap> {
        let (Some(region), Some(win)) = (self.region, self.window) else {
            return Err(Trap::MemoryFault);
        };
        let eff = win.checked(addr, offset, width).ok_or(Trap::MemoryFault)?;
        if eff < self.null_guard {
            return Err(Trap::MemoryFault);
        }
        Ok((region, eff))
    }

    #[inline]
    fn check_write(&self, addr: u64, offset: u64, width: u32) -> Result<(&'a Region, u64), Trap> {
        let (region, eff) = self.check_read(addr, offset, width)?;
        let end = eff + u64::from(width);
        for &(ro_start, ro_end) in self.readonly_ranges {
            if eff < ro_end && end > ro_start {
                return Err(Trap::MemoryFault);
            }
        }
        Ok((region, eff))
    }

    #[inline]
    fn check_span(&self, ptr: u64, len: u64, write: bool) -> Result<&'a Region, Trap> {
        let (Some(region), Some(win)) = (self.region, self.window) else {
            return Err(Trap::MemoryFault);
        };
        if len == 0 {
            return Ok(region);
        }
        let eff = win.span_checked(ptr, len).ok_or(Trap::MemoryFault)?;
        if eff < self.null_guard {
            return Err(Trap::MemoryFault);
        }
        let end = eff.checked_add(len).ok_or(Trap::MemoryFault)?;
        if end > win.mapped() {
            return Err(Trap::MemoryFault);
        }
        if write {
            for &(ro_start, ro_end) in self.readonly_ranges {
                if eff < ro_end && end > ro_start {
                    return Err(Trap::MemoryFault);
                }
            }
        }
        Ok(region)
    }

    #[inline]
    fn confine_only(&self, addr: u64, width: u32) -> Result<u64, Trap> {
        let Some(win) = self.window else {
            return Err(Trap::MemoryFault);
        };
        win.checked(addr, 0, width).ok_or(Trap::MemoryFault)
    }
}

#[derive(Clone, Debug)]
struct Frame {
    func: u32,
    pc: u32,
    base_reg: usize,
    /// Where to write results in the *caller's* register window (`usize::MAX` for root frame).
    ret_dst: usize,
}

/// One intra-Context `setjmp` checkpoint (never crosses Context boundaries).
#[derive(Clone, Debug)]
struct SetJmpCheckpoint {
    token: u64,
    frame_depth: usize,
    resume_pc: u32,
    dst_reg: usize,
    saved_regs: Vec<Reg>,
}

/// Pending destination register(s) waiting on a kernel step result before resuming.
#[derive(Clone, Copy, Debug)]
struct PendingResult {
    base_reg: usize,
    dst_reg: u32,
    count: u8,
    /// PC of the op that yielded (for `rewind_for_reexec` per Invariant 7).
    op_pc: u32,
}

/// Mutable execution state of one `Engine::Bytecode` context (Root, Thread, GuestFiber, or
/// ServeHandler). Owned exclusively by the Carrier while running; zero locks on the hot loop.
#[derive(Clone, Debug)]
pub struct ExecState {
    regs: Vec<Reg>,
    frames: Vec<Frame>,
    /// Whether the very first function entry safepoint still needs to be charged.
    entry_pending_safepoint: bool,
    pending: Option<PendingResult>,
    setjmps: Vec<SetJmpCheckpoint>,
    next_setjmp_token: u64,
    scratch_moves: Vec<Reg>,
}

impl ExecState {
    /// Prepare a fresh execution state positioned at the entry of `code.funcs[func_idx]`.
    pub fn new_entry(code: &Code, func_idx: u32, args: &[i64]) -> Result<ExecState, Trap> {
        let f = code
            .funcs
            .get(func_idx as usize)
            .ok_or(Trap::IndirectCallType)?;
        if args.len() != f.num_params as usize {
            return Err(Trap::IndirectCallType);
        }
        let mut regs = vec![Reg::default(); f.num_regs as usize];
        for (i, &a) in args.iter().enumerate() {
            regs[i] = Reg::from_i64(a);
        }
        Ok(ExecState {
            regs,
            frames: vec![Frame {
                func: func_idx,
                pc: 0,
                base_reg: 0,
                ret_dst: usize::MAX,
            }],
            entry_pending_safepoint: true,
            pending: None,
            setjmps: Vec::new(),
            next_setjmp_token: 0x5E71_0001,
            scratch_moves: Vec::new(),
        })
    }

    /// Prepare a fresh `GuestFiber` entry (`(i64, i64) -> i64`) binding `arg0`; the first
    /// `cont.resume` supplies `arg1` via [`Self::deliver_fiber_start_arg`].
    pub fn new_fiber_entry(code: &Code, func_idx: u32, arg0: i64) -> Result<ExecState, Trap> {
        let f = code
            .funcs
            .get(func_idx as usize)
            .ok_or(Trap::IndirectCallType)?;
        if f.type_id != TypeId::fiber_entry() {
            return Err(Trap::IndirectCallType);
        }
        let mut s = Self::new_entry(code, func_idx, &[arg0, 0])?;
        // `cont.resume` already charges the IR-anchored safepoint on entry to the fiber (Invariant 9).
        s.entry_pending_safepoint = false;
        Ok(s)
    }

    /// Deliver `arg1` on a `GuestFiber`'s very first resume before stepping it.
    pub fn set_entry_arg1(&mut self, arg1: i64) {
        if self.regs.len() > 1 {
            self.regs[1] = Reg::from_i64(arg1);
        }
    }

    /// Write result words from a completed kernel step (`FutexWait`, `FutexNotify`, `CallCap`,
    /// `FiberNew`, `FiberResume`, `FiberSuspend`, `ThreadSpawn`, `ThreadJoin`) into the waiting
    /// register(s).
    pub fn deliver_results(&mut self, vals: &[i64]) {
        if let Some(p) = self.pending.take() {
            let n = (p.count as usize).min(vals.len());
            for (i, &v) in vals.iter().take(n).enumerate() {
                self.regs[p.base_reg + p.dst_reg as usize + i] = Reg::from_i64(v);
            }
        }
    }

    /// Convenience helper for delivering a single `i32` result (e.g. `memory.wait` status).
    pub fn deliver_i32(&mut self, val: i32) {
        if let Some(p) = self.pending.take() {
            if p.count >= 1 {
                self.regs[p.base_reg + p.dst_reg as usize] = Reg::from_i32(val);
            }
        }
    }

    /// Convenience helper for delivering a `(status: i32, value: i64)` fiber resume pair.
    pub fn deliver_fiber_resume(&mut self, status: i32, value: i64) {
        if let Some(p) = self.pending.take() {
            if p.count >= 2 {
                self.regs[p.base_reg + p.dst_reg as usize] = Reg::from_i32(status);
                self.regs[p.base_reg + p.dst_reg as usize + 1] = Reg::from_i64(value);
            }
        }
    }

    /// Rewind the current frame's `pc` back to the op that yielded, so a parked op re-executes
    /// cleanly on wake (Invariant 7).
    pub fn rewind_for_reexec(&mut self) {
        if let Some(p) = self.pending.take() {
            if let Some(frame) = self.frames.last_mut() {
                frame.pc = p.op_pc;
            }
        }
    }

    /// Whether this context still has active frames (not finished).
    pub fn has_frames(&self) -> bool {
        !self.frames.is_empty()
    }

    #[inline]
    fn apply_edge(&mut self, f: &CompiledFunc, base: usize, edge: &BranchEdge) {
        let start = edge.moves_start as usize;
        let len = edge.moves_len as usize;
        if len == 0 {
            return;
        }
        if len == 1 {
            let mv = f.moves[start];
            self.regs[base + mv.dst as usize] = self.regs[base + mv.src as usize];
            return;
        }
        // Parallel copy via reusable scratch buffer so cyclic swaps `(a, b) -> (b, a)` are exact.
        self.scratch_moves.clear();
        for mv in &f.moves[start..start + len] {
            self.scratch_moves.push(self.regs[base + mv.src as usize]);
        }
        for (mv, &val) in f.moves[start..start + len]
            .iter()
            .zip(self.scratch_moves.iter())
        {
            self.regs[base + mv.dst as usize] = val;
        }
    }
}

#[inline]
fn load_val(region: &Region, eff: u64, op: LoadOp) -> Reg {
    match op {
        LoadOp::I32 => Reg::from_i32(region.read_word(eff, 4) as u32 as i32),
        LoadOp::I64 => Reg::from_i64(region.read_word(eff, 8) as i64),
        LoadOp::F32 => Reg::from_f32_bits(region.read_word(eff, 4) as u32),
        LoadOp::F64 => Reg::from_f64_bits(region.read_word(eff, 8)),
        LoadOp::I32_8S => Reg::from_i32(region.byte(eff) as i8 as i32),
        LoadOp::I32_8U => Reg::from_i32(region.byte(eff) as i32),
        LoadOp::I32_16S => Reg::from_i32(region.read_word(eff, 2) as u16 as i16 as i32),
        LoadOp::I32_16U => Reg::from_i32(region.read_word(eff, 2) as u16 as i32),
        LoadOp::I64_8S => Reg::from_i64(region.byte(eff) as i8 as i64),
        LoadOp::I64_8U => Reg::from_i64(region.byte(eff) as i64),
        LoadOp::I64_16S => Reg::from_i64(region.read_word(eff, 2) as u16 as i16 as i64),
        LoadOp::I64_16U => Reg::from_i64(region.read_word(eff, 2) as u16 as i64),
        LoadOp::I64_32S => Reg::from_i64(region.read_word(eff, 4) as u32 as i32 as i64),
        LoadOp::I64_32U => Reg::from_i64(region.read_word(eff, 4) as u32 as i64),
    }
}

#[inline]
fn store_val(region: &Region, eff: u64, op: StoreOp, val: Reg) {
    match op {
        StoreOp::I32 | StoreOp::F32 => region.write_word(eff, 4, val.0 as u64),
        StoreOp::I64 | StoreOp::F64 => region.write_word(eff, 8, val.0 as u64),
        StoreOp::I32_8 | StoreOp::I64_8 => region.set_byte(eff, val.0 as u8),
        StoreOp::I32_16 | StoreOp::I64_16 => region.write_word(eff, 2, val.0 as u64),
        StoreOp::I64_32 => region.write_word(eff, 4, val.0 as u64),
    }
}

#[inline]
fn atomic_width(ty: IntTy) -> u32 {
    match ty {
        IntTy::I32 => 4,
        IntTy::I64 => 8,
    }
}

/// Step `state` forward on `code` until it hits a kernel boundary, completes, or traps.
///
/// - `fuel` is the active **Carrier's** [`FuelCell`] (Decision 30).
/// - `poll_word` is the enclosing [`temen_kernel::Domain`]'s shared `AtomicU64`.
/// - `carrier_tls` is the executing Carrier's per-vCPU TLS word (so migrated fibers see their new
///   Carrier's TLS word on `vcpu.tls.get`/`set`).
pub fn step(
    code: &Code,
    state: &mut ExecState,
    mem: &MemView<'_>,
    fuel: &mut FuelCell,
    poll_word: &AtomicU64,
    carrier_tls: &mut i64,
    cfg: StepConfig,
) -> Step {
    let mut quantum = cfg.quantum.max(1);

    // Charge initial entry safepoint once before running the first instruction.
    if state.entry_pending_safepoint {
        let p = poll_word.load(Ordering::Relaxed);
        if p != 0 {
            return Step::Yield(YieldReason::Poll(PollFlags(p)));
        }
        if !fuel.charge_one() {
            return Step::Yield(YieldReason::FuelRefill);
        }
        state.entry_pending_safepoint = false;
    }

    macro_rules! safepoint {
        ($restore_pc:expr) => {{
            let p = poll_word.load(Ordering::Relaxed);
            if p != 0 {
                if let Some(fr) = state.frames.last_mut() {
                    fr.pc = $restore_pc;
                }
                return Step::Yield(YieldReason::Poll(PollFlags(p)));
            }
            if !fuel.charge_one() {
                if let Some(fr) = state.frames.last_mut() {
                    fr.pc = $restore_pc;
                }
                return Step::Yield(YieldReason::FuelRefill);
            }
            quantum -= 1;
            if quantum == 0 {
                return Step::Yield(YieldReason::Quantum);
            }
        }};
    }

    loop {
        let Some(frame) = state.frames.last_mut() else {
            return Step::Finished(ContextExit::Returned(Vec::new()));
        };
        let func_idx = frame.func as usize;
        let pc = frame.pc;
        let base = frame.base_reg;
        frame.pc = pc + 1;

        let f = &code.funcs[func_idx];
        let op = &f.ops[pc as usize];

        macro_rules! r {
            ($idx:expr) => {
                state.regs[base + ($idx) as usize]
            };
        }
        macro_rules! w {
            ($idx:expr, $val:expr) => {
                state.regs[base + ($idx) as usize] = $val
            };
        }

        match *op {
            // ---- Constants & refs ----
            BcOp::ConstI32 { dst, val } => w!(dst, Reg::from_i32(val)),
            BcOp::ConstI64 { dst, val } => w!(dst, Reg::from_i64(val)),
            BcOp::ConstF32 { dst, bits } => w!(dst, Reg::from_f32_bits(bits)),
            BcOp::ConstF64 { dst, bits } => w!(dst, Reg::from_f64_bits(bits)),
            BcOp::ConstV128 { dst, bytes } => w!(dst, Reg::from_v128(bytes)),
            BcOp::RefFunc { dst, func } => w!(dst, Reg::from_i32(func as i32)),

            // ---- Scalar integers ----
            BcOp::IntBin32 { dst, op, a, b } => match bin32(op, r!(a).i32(), r!(b).i32()) {
                Ok(v) => w!(dst, Reg::from_i32(v)),
                Err(t) => return Step::Finished(ContextExit::Trapped(t)),
            },
            BcOp::IntBin64 { dst, op, a, b } => match bin64(op, r!(a).i64(), r!(b).i64()) {
                Ok(v) => w!(dst, Reg::from_i64(v)),
                Err(t) => return Step::Finished(ContextExit::Trapped(t)),
            },
            BcOp::IntBinImm32 { dst, op, a, imm } => match bin32(op, r!(a).i32(), imm) {
                Ok(v) => w!(dst, Reg::from_i32(v)),
                Err(t) => return Step::Finished(ContextExit::Trapped(t)),
            },
            BcOp::IntBinImm64 { dst, op, a, imm } => match bin64(op, r!(a).i64(), imm) {
                Ok(v) => w!(dst, Reg::from_i64(v)),
                Err(t) => return Step::Finished(ContextExit::Trapped(t)),
            },
            BcOp::IntCmp32 { dst, op, a, b } => {
                w!(
                    dst,
                    Reg::from_i32(i32::from(cmp32(op, r!(a).i32(), r!(b).i32())))
                )
            }
            BcOp::IntCmp64 { dst, op, a, b } => {
                w!(
                    dst,
                    Reg::from_i32(i32::from(cmp64(op, r!(a).i64(), r!(b).i64())))
                )
            }
            BcOp::IntCmpImm32 { dst, op, a, imm } => {
                w!(dst, Reg::from_i32(i32::from(cmp32(op, r!(a).i32(), imm))))
            }
            BcOp::IntCmpImm64 { dst, op, a, imm } => {
                w!(dst, Reg::from_i32(i32::from(cmp64(op, r!(a).i64(), imm))))
            }
            BcOp::IntUn32 { dst, op, a } => w!(dst, Reg::from_i32(intun32(op, r!(a).i32()))),
            BcOp::IntUn64 { dst, op, a } => w!(dst, Reg::from_i64(intun64(op, r!(a).i64()))),
            BcOp::Eqz32 { dst, a } => w!(dst, Reg::from_i32(i32::from(r!(a).i32() == 0))),
            BcOp::Eqz64 { dst, a } => w!(dst, Reg::from_i32(i32::from(r!(a).i64() == 0))),
            BcOp::Convert { dst, op, a } => w!(dst, convert(op, r!(a))),
            BcOp::Select { dst, cond, a, b } => {
                let chosen = if r!(cond).i32() != 0 { r!(a) } else { r!(b) };
                w!(dst, chosen);
            }

            // ---- Scalar floats ----
            BcOp::FBin32 { dst, op, a, b } => {
                w!(dst, Reg::from_f32(fbin32(op, r!(a).f32(), r!(b).f32())))
            }
            BcOp::FBin64 { dst, op, a, b } => {
                w!(dst, Reg::from_f64(fbin64(op, r!(a).f64(), r!(b).f64())))
            }
            BcOp::FUn32 { dst, op, a } => w!(dst, Reg::from_f32(fun32(op, r!(a).f32()))),
            BcOp::FUn64 { dst, op, a } => w!(dst, Reg::from_f64(fun64(op, r!(a).f64()))),
            BcOp::Fma32 { dst, a, b, c } => {
                w!(
                    dst,
                    Reg::from_f32(libm::fmaf(r!(a).f32(), r!(b).f32(), r!(c).f32()))
                )
            }
            BcOp::Fma64 { dst, a, b, c } => {
                w!(
                    dst,
                    Reg::from_f64(libm::fma(r!(a).f64(), r!(b).f64(), r!(c).f64()))
                )
            }
            BcOp::FCmp32 { dst, op, a, b } => {
                w!(
                    dst,
                    Reg::from_i32(i32::from(fcmp32(op, r!(a).f32(), r!(b).f32())))
                )
            }
            BcOp::FCmp64 { dst, op, a, b } => {
                w!(
                    dst,
                    Reg::from_i32(i32::from(fcmp64(op, r!(a).f64(), r!(b).f64())))
                )
            }
            BcOp::FToISat { dst, op, a } => w!(dst, fto_i_sat(op, r!(a))),
            BcOp::FToITrap { dst, op, a } => match fto_i_trap(op, r!(a)) {
                Ok(v) => w!(dst, v),
                Err(t) => return Step::Finished(ContextExit::Trapped(t)),
            },
            BcOp::IToF { dst, op, a } => w!(dst, i_to_f(op, r!(a))),
            BcOp::Cast { dst, op, a } => w!(dst, cast(op, r!(a))),

            // ---- Linear memory ----
            BcOp::Load {
                dst,
                op,
                addr,
                offset,
            } => {
                let width = op.info().2;
                match mem.check_read(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => w!(dst, load_val(region, eff, op)),
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::Store {
                op,
                addr,
                val,
                offset,
            } => {
                let width = op.info().2;
                match mem.check_write(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => store_val(region, eff, op, r!(val)),
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::MemCopy {
                dst_addr,
                src_addr,
                len,
            }
            | BcOp::MemMove {
                dst_addr,
                src_addr,
                len,
            } => {
                let d = r!(dst_addr).u64();
                let s = r!(src_addr).u64();
                let n = r!(len).u64();
                if let Err(t) = mem.check_span(s, n, false) {
                    return Step::Finished(ContextExit::Trapped(t));
                }
                match mem.check_span(d, n, true) {
                    Ok(region) => {
                        if n > 0 {
                            region.copy_within(d, s, n);
                        }
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::MemFill { dst_addr, val, len } => {
                let d = r!(dst_addr).u64();
                let b = r!(val).i32() as u8;
                let n = r!(len).u64();
                match mem.check_span(d, n, true) {
                    Ok(region) => {
                        if n > 0 {
                            region.fill(d, n, b);
                        }
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }

            // ---- Atomics, futex, TLS ----
            BcOp::AtomicLoad {
                dst,
                ty,
                addr,
                offset,
            } => {
                let width = atomic_width(ty);
                match mem.check_read(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => {
                        if eff % u64::from(width) != 0 {
                            return Step::Finished(ContextExit::Trapped(Trap::MemoryFault));
                        }
                        let v = region.atomic_load(eff, width);
                        w!(
                            dst,
                            match ty {
                                IntTy::I32 => Reg::from_i32(v as u32 as i32),
                                IntTy::I64 => Reg::from_i64(v as i64),
                            }
                        );
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::AtomicStore {
                ty,
                addr,
                val,
                offset,
            } => {
                let width = atomic_width(ty);
                match mem.check_write(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => {
                        if eff % u64::from(width) != 0 {
                            return Step::Finished(ContextExit::Trapped(Trap::MemoryFault));
                        }
                        region.atomic_store(eff, width, r!(val).u64());
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::AtomicRmw {
                dst,
                ty,
                op,
                addr,
                val,
                offset,
            } => {
                let width = atomic_width(ty);
                match mem.check_write(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => {
                        if eff % u64::from(width) != 0 {
                            return Step::Finished(ContextExit::Trapped(Trap::MemoryFault));
                        }
                        let old = region.atomic_rmw(eff, width, rmw_op(op), r!(val).u64());
                        w!(
                            dst,
                            match ty {
                                IntTy::I32 => Reg::from_i32(old as u32 as i32),
                                IntTy::I64 => Reg::from_i64(old as i64),
                            }
                        );
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::AtomicCmpxchg {
                dst,
                ty,
                addr,
                expected,
                replacement,
                offset,
            } => {
                let width = atomic_width(ty);
                match mem.check_write(r!(addr).u64(), offset, width) {
                    Ok((region, eff)) => {
                        if eff % u64::from(width) != 0 {
                            return Step::Finished(ContextExit::Trapped(Trap::MemoryFault));
                        }
                        let old = region.atomic_cmpxchg(
                            eff,
                            width,
                            r!(expected).u64(),
                            r!(replacement).u64(),
                        );
                        w!(
                            dst,
                            match ty {
                                IntTy::I32 => Reg::from_i32(old as u32 as i32),
                                IntTy::I64 => Reg::from_i64(old as i64),
                            }
                        );
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::AtomicFence { order } => {
                let ord = match order {
                    IrOrdering::Relaxed => None,
                    IrOrdering::Acquire => Some(Ordering::Acquire),
                    IrOrdering::Release => Some(Ordering::Release),
                    IrOrdering::AcqRel => Some(Ordering::AcqRel),
                    IrOrdering::SeqCst => Some(Ordering::SeqCst),
                };
                if let Some(o) = ord {
                    core::sync::atomic::fence(o);
                }
            }
            BcOp::MemoryWait {
                dst,
                ty,
                addr,
                expected,
                timeout,
            } => {
                let width = atomic_width(ty);
                let eff = match mem.check_read(r!(addr).u64(), 0, width) {
                    Ok((_, eff)) if eff % u64::from(width) == 0 => eff,
                    _ => return Step::Finished(ContextExit::Trapped(Trap::MemoryFault)),
                };
                let exp = match ty {
                    IntTy::I32 => r!(expected).i32() as u32 as i64,
                    IntTy::I64 => r!(expected).i64(),
                };
                let timeout_ns = r!(timeout).i64();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 1,
                    op_pc: pc,
                });
                return Step::FutexWait {
                    addr: eff,
                    expected: exp,
                    width64: ty == IntTy::I64,
                    timeout_ns,
                };
            }
            BcOp::MemoryNotify { dst, addr, count } => {
                let eff = match mem.confine_only(r!(addr).u64(), 4) {
                    Ok(eff) if eff % 4 == 0 => eff,
                    _ => return Step::Finished(ContextExit::Trapped(Trap::MemoryFault)),
                };
                let c = r!(count).i32() as u32;
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 1,
                    op_pc: pc,
                });
                return Step::FutexNotify {
                    addr: eff,
                    count: c,
                };
            }
            BcOp::VcpuTlsGet { dst } => w!(dst, Reg::from_i64(*carrier_tls)),
            BcOp::VcpuTlsSet { val } => *carrier_tls = r!(val).i64(),

            // ---- Threads & fibers ----
            BcOp::ThreadSpawn {
                dst,
                func,
                arg0,
                arg1,
            } => {
                let a0 = r!(arg0).i64();
                let a1 = r!(arg1).i64();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 1,
                    op_pc: pc,
                });
                return Step::ThreadSpawn {
                    func_idx: func,
                    arg0: a0,
                    arg1: a1,
                };
            }
            BcOp::ThreadJoin { dst, handle } => {
                let h = r!(handle).i32();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 2,
                    op_pc: pc,
                });
                return Step::ThreadJoin { handle: h };
            }
            BcOp::ContNew { dst, func, arg0 } => {
                let func_idx = r!(func).i32() as u32;
                let a0 = r!(arg0).i64();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 1,
                    op_pc: pc,
                });
                return Step::FiberNew { func_idx, arg0: a0 };
            }
            BcOp::ContResume { dst, k, arg, block } => {
                safepoint!(pc);
                let handle = r!(k).i64();
                let val = r!(arg).i64();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 2,
                    op_pc: pc,
                });
                return Step::FiberResume { handle, val, block };
            }
            BcOp::Suspend { dst, val } => {
                let v = r!(val).i64();
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: 1,
                    op_pc: pc,
                });
                return Step::FiberSuspend { val: v };
            }

            // ---- Intra-Context setjmp/longjmp ----
            BcOp::SetJmp { dst, buf } => {
                let (region, eff) = match mem.check_write(r!(buf).u64(), 0, 8) {
                    Ok(x) => x,
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                };
                let token = state.next_setjmp_token;
                state.next_setjmp_token = state.next_setjmp_token.wrapping_add(1).max(1);
                region.write_word(eff, 8, token);
                let frame_depth = state.frames.len();
                let saved_regs = state.regs[base..base + f.num_regs as usize].to_vec();
                state.setjmps.push(SetJmpCheckpoint {
                    token,
                    frame_depth,
                    resume_pc: pc + 1,
                    dst_reg: base + dst as usize,
                    saved_regs,
                });
                w!(dst, Reg::from_i32(0));
            }
            BcOp::LongJmp { buf, val } => {
                let (region, eff) = match mem.check_read(r!(buf).u64(), 0, 8) {
                    Ok(x) => x,
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                };
                let token = region.read_word(eff, 8);
                let jump_val = {
                    let v = r!(val).i32();
                    if v == 0 {
                        1
                    } else {
                        v
                    }
                };
                let Some(pos) = state.setjmps.iter().rposition(|c| c.token == token) else {
                    return Step::Finished(ContextExit::Trapped(Trap::FiberFault));
                };
                let cp = state.setjmps[pos].clone();
                if cp.frame_depth == 0 || cp.frame_depth > state.frames.len() {
                    return Step::Finished(ContextExit::Trapped(Trap::FiberFault));
                }
                // Discard any checkpoints deeper than the target frame, keep target reusable.
                state.setjmps.truncate(pos + 1);
                state.frames.truncate(cp.frame_depth);
                let target_frame = state.frames.last_mut().unwrap();
                let target_base = target_frame.base_reg;
                target_frame.pc = cp.resume_pc;
                state.regs.truncate(target_base + cp.saved_regs.len());
                state.regs[target_base..target_base + cp.saved_regs.len()]
                    .copy_from_slice(&cp.saved_regs);
                state.regs[cp.dst_reg] = Reg::from_i32(jump_val);
            }

            // ---- Calls & capability dispatch ----
            BcOp::Call {
                dst,
                nresults: _,
                func,
                args_start,
                args_len,
            } => {
                safepoint!(pc);
                if state.frames.len() >= cfg.max_call_depth {
                    return Step::Finished(ContextExit::Trapped(Trap::StackOverflow));
                }
                let callee = &code.funcs[func as usize];
                let new_base = state.regs.len();
                state
                    .regs
                    .resize(new_base + callee.num_regs as usize, Reg::default());
                let a_start = args_start as usize;
                let a_len = args_len as usize;
                for i in 0..a_len {
                    let src_reg = f.arg_regs[a_start + i] as usize;
                    state.regs[new_base + i] = state.regs[base + src_reg];
                }
                state.frames.push(Frame {
                    func,
                    pc: 0,
                    base_reg: new_base,
                    ret_dst: base + dst as usize,
                });
            }
            BcOp::CallIndirect {
                dst,
                nresults: _,
                expected_type,
                idx,
                args_start,
                args_len,
            } => {
                safepoint!(pc);
                let Some(func) = code.resolve_indirect(r!(idx).i32(), expected_type) else {
                    return Step::Finished(ContextExit::Trapped(Trap::IndirectCallType));
                };
                if state.frames.len() >= cfg.max_call_depth {
                    return Step::Finished(ContextExit::Trapped(Trap::StackOverflow));
                }
                let callee = &code.funcs[func as usize];
                let new_base = state.regs.len();
                state
                    .regs
                    .resize(new_base + callee.num_regs as usize, Reg::default());
                let a_start = args_start as usize;
                let a_len = args_len as usize;
                for i in 0..a_len {
                    let src_reg = f.arg_regs[a_start + i] as usize;
                    state.regs[new_base + i] = state.regs[base + src_reg];
                }
                state.frames.push(Frame {
                    func,
                    pc: 0,
                    base_reg: new_base,
                    ret_dst: base + dst as usize,
                });
            }
            BcOp::CapCall {
                dst,
                nresults,
                type_id,
                op: cap_op,
                handle,
                args_start,
                args_len,
            } => {
                let h = r!(handle).i32();
                let mut args = [0i64; MAX_CAP_ARGS];
                let a_start = args_start as usize;
                let a_len = (args_len as usize).min(MAX_CAP_ARGS);
                for (i, slot) in args.iter_mut().take(a_len).enumerate() {
                    let src_reg = f.arg_regs[a_start + i] as usize;
                    *slot = state.regs[base + src_reg].i64();
                }
                state.pending = Some(PendingResult {
                    base_reg: base,
                    dst_reg: dst,
                    count: nresults,
                    op_pc: pc,
                });
                return Step::CallCap {
                    handle: h,
                    type_id,
                    op: cap_op,
                    args,
                    nargs: a_len as u8,
                    nresults,
                };
            }

            // ---- Terminators ----
            BcOp::Br { edge } => {
                let e = &f.edges[edge as usize];
                if e.is_back_edge {
                    safepoint!(pc);
                }
                let target_pc = e.target_pc;
                state.apply_edge(f, base, e);
                state.frames.last_mut().unwrap().pc = target_pc;
            }
            BcOp::BrIf {
                cond,
                then_edge,
                else_edge,
            } => {
                let edge_idx = if r!(cond).i32() != 0 {
                    then_edge
                } else {
                    else_edge
                };
                let e = &f.edges[edge_idx as usize];
                if e.is_back_edge {
                    safepoint!(pc);
                }
                let target_pc = e.target_pc;
                state.apply_edge(f, base, e);
                state.frames.last_mut().unwrap().pc = target_pc;
            }
            BcOp::BrTable {
                idx,
                edges_start,
                edges_len,
                default_edge,
            } => {
                let raw = r!(idx).i32() as u32;
                let edge_idx = if raw < edges_len {
                    edges_start + raw
                } else {
                    default_edge
                };
                let e = &f.edges[edge_idx as usize];
                if e.is_back_edge {
                    safepoint!(pc);
                }
                let target_pc = e.target_pc;
                state.apply_edge(f, base, e);
                state.frames.last_mut().unwrap().pc = target_pc;
            }
            BcOp::Return {
                vals_start,
                vals_len,
            } => {
                let finished = state.frames.pop().unwrap();
                // Prune any setjmp checkpoints created in the returning frame.
                let new_depth = state.frames.len();
                state.setjmps.retain(|c| c.frame_depth <= new_depth);

                let v_start = vals_start as usize;
                let v_len = vals_len as usize;
                if state.frames.is_empty() {
                    let mut out = Vec::with_capacity(v_len);
                    for i in 0..v_len {
                        let reg = f.arg_regs[v_start + i] as usize;
                        out.push(state.regs[base + reg].i64());
                    }
                    state.regs.clear();
                    return Step::Finished(ContextExit::Returned(out));
                }
                // Copy return values into caller's destination window, then truncate callee regs.
                let ret_dst = finished.ret_dst;
                for i in 0..v_len {
                    let reg = f.arg_regs[v_start + i] as usize;
                    let val = state.regs[base + reg];
                    state.regs[ret_dst + i] = val;
                }
                state.regs.truncate(base);
            }
            BcOp::ReturnCall {
                func,
                args_start,
                args_len,
            } => {
                safepoint!(pc);
                let callee = &code.funcs[func as usize];
                let a_start = args_start as usize;
                let a_len = args_len as usize;
                state.scratch_moves.clear();
                for i in 0..a_len {
                    let src_reg = f.arg_regs[a_start + i] as usize;
                    state.scratch_moves.push(state.regs[base + src_reg]);
                }
                // Invalidate setjmps belonging to the replaced frame.
                let cur_depth = state.frames.len();
                state.setjmps.retain(|c| c.frame_depth < cur_depth);
                // Reuse the current frame's register window in O(1) space.
                state.regs.truncate(base);
                state
                    .regs
                    .resize(base + callee.num_regs as usize, Reg::default());
                for (i, &v) in state.scratch_moves.iter().enumerate() {
                    state.regs[base + i] = v;
                }
                let fr = state.frames.last_mut().unwrap();
                fr.func = func;
                fr.pc = 0;
            }
            BcOp::ReturnCallIndirect {
                expected_type,
                idx,
                args_start,
                args_len,
            } => {
                safepoint!(pc);
                let Some(func) = code.resolve_indirect(r!(idx).i32(), expected_type) else {
                    return Step::Finished(ContextExit::Trapped(Trap::IndirectCallType));
                };
                let callee = &code.funcs[func as usize];
                let a_start = args_start as usize;
                let a_len = args_len as usize;
                state.scratch_moves.clear();
                for i in 0..a_len {
                    let src_reg = f.arg_regs[a_start + i] as usize;
                    state.scratch_moves.push(state.regs[base + src_reg]);
                }
                let cur_depth = state.frames.len();
                state.setjmps.retain(|c| c.frame_depth < cur_depth);
                state.regs.truncate(base);
                state
                    .regs
                    .resize(base + callee.num_regs as usize, Reg::default());
                for (i, &v) in state.scratch_moves.iter().enumerate() {
                    state.regs[base + i] = v;
                }
                let fr = state.frames.last_mut().unwrap();
                fr.func = func;
                fr.pc = 0;
            }
            BcOp::Unreachable => return Step::Finished(ContextExit::Trapped(Trap::Unreachable)),

            // ---- Core SIMD ----
            BcOp::V128Load { dst, addr, offset } => {
                match mem.check_read(r!(addr).u64(), offset, 16) {
                    Ok((region, eff)) => {
                        let mut bytes = [0u8; 16];
                        region.read_into(eff, &mut bytes);
                        w!(dst, Reg::from_v128(bytes));
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::V128Store { addr, val, offset } => {
                match mem.check_write(r!(addr).u64(), offset, 16) {
                    Ok((region, eff)) => {
                        let bytes = r!(val).v128();
                        region.write_from(eff, &bytes);
                    }
                    Err(t) => return Step::Finished(ContextExit::Trapped(t)),
                }
            }
            BcOp::Splat { dst, shape, a } => w!(dst, splat(shape, r!(a))),
            BcOp::ExtractLane {
                dst,
                shape,
                lane,
                signed,
                a,
            } => w!(dst, extract_lane(shape, lane, signed, r!(a))),
            BcOp::ReplaceLane {
                dst,
                shape,
                lane,
                a,
                val,
            } => w!(dst, replace_lane(shape, lane, r!(a), r!(val))),
            BcOp::VIntBin {
                dst,
                shape,
                op,
                a,
                b,
            } => match vint_bin(shape, op, r!(a), r!(b)) {
                Some(v) => w!(dst, v),
                None => return Step::Finished(ContextExit::Trapped(Trap::Malformed)),
            },
            BcOp::VFloatBin {
                dst,
                shape,
                op,
                a,
                b,
            } => match vfloat_bin(shape, op, r!(a), r!(b)) {
                Some(v) => w!(dst, v),
                None => return Step::Finished(ContextExit::Trapped(Trap::Malformed)),
            },
            BcOp::VBitBin { dst, op, a, b } => w!(dst, vbit_bin(op, r!(a), r!(b))),
            BcOp::VNot { dst, a } => w!(dst, Reg(!r!(a).0)),
            BcOp::Bitselect { dst, a, b, mask } => {
                let m = r!(mask).0;
                w!(dst, Reg((r!(a).0 & m) | (r!(b).0 & !m)));
            }
            BcOp::Shuffle { dst, lanes, a, b } => w!(dst, vshuffle(&lanes, r!(a), r!(b))),
            BcOp::Swizzle { dst, a, b } => w!(dst, vswizzle(r!(a), r!(b))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile::compile_module;
    use temen_ir::{Block, Func, FuncType, Inst, Memory, Module, Terminator, TypeEntry, ValType};
    use temen_kernel::sync::AtomicU64;

    fn run_simple(m: &Module, args: &[i64], fuel_limit: Option<u64>) -> (Step, u64) {
        let code = compile_module(m).expect("compile");
        let region = m
            .memory
            .as_ref()
            .map(|mem| Region::new(1u64 << mem.size_log2, 4096));
        let window = m.memory.as_ref().map(|mem| Window::new(mem.size_log2));
        let mut ro = Vec::new();
        if let Some(ref reg) = region {
            for d in &code.initial_data {
                reg.write_from(d.offset, &d.bytes);
                if d.readonly {
                    ro.push((d.offset, d.offset + d.bytes.len() as u64));
                }
            }
        }
        let mem = MemView {
            region: region.as_ref(),
            window,
            readonly_ranges: &ro,
            null_guard: 0,
        };
        let mut state = ExecState::new_entry(&code, 0, args).expect("entry");
        let mut fuel = match fuel_limit {
            Some(n) => FuelCell::metered(n),
            None => FuelCell::unlimited(),
        };
        let poll = AtomicU64::new(0);
        let mut tls = 0i64;
        let out = step(
            &code,
            &mut state,
            &mem,
            &mut fuel,
            &poll,
            &mut tls,
            StepConfig {
                quantum: u32::MAX,
                max_call_depth: 1024,
            },
        );
        (out, fuel.remaining)
    }

    #[test]
    fn loop_sum_and_exact_fuel_metering() {
        // sum(n): loop i=0, acc=0; while i < n { acc += i; i += 1 } return acc
        let m = Module {
            funcs: vec![Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        parent: 0,
                        captures: vec![],
                        params: vec![ValType::I64],
                        insts: vec![Inst::ConstI64(0)],
                        term: Terminator::Br {
                            target: 1,
                            args: vec![1, 1], // i=0, acc=0
                        },
                    },
                    Block {
                        parent: 0,
                        captures: vec![(0, 0)],                   // v2 = n
                        params: vec![ValType::I64, ValType::I64], // v0=i, v1=acc
                        insts: vec![
                            Inst::IntCmp {
                                ty: IntTy::I64,
                                op: temen_ir::CmpOp::LtS,
                                a: 0,
                                b: 2,
                            }, // v3 = i < n
                        ],
                        term: Terminator::BrIf {
                            cond: 3,
                            then_blk: 2,
                            then_args: vec![],
                            else_blk: 3,
                            else_args: vec![],
                        },
                    },
                    Block {
                        parent: 1,
                        captures: vec![(1, 0), (1, 1)], // v0=i, v1=acc
                        params: vec![],
                        insts: vec![
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: temen_ir::BinOp::Add,
                                a: 1,
                                b: 0,
                            }, // v2 = acc + i
                            Inst::IntBinImm {
                                ty: IntTy::I64,
                                op: temen_ir::BinOp::Add,
                                a: 0,
                                imm: 1,
                            }, // v3 = i + 1
                        ],
                        term: Terminator::Br {
                            target: 1,
                            args: vec![3, 2], // back-edge!
                        },
                    },
                    Block {
                        parent: 1,
                        captures: vec![(1, 1)],
                        params: vec![],
                        insts: vec![],
                        term: Terminator::Return(vec![0]),
                    },
                ],
            }],
            ..Module::default()
        };
        // Entry = 1 fuel, 10 back-edges = 10 fuel => total 11 fuel.
        let (out, rem) = run_simple(&m, &[10], Some(20));
        assert_eq!(out, Step::Finished(ContextExit::Returned(vec![45])));
        assert_eq!(rem, 9);

        // With only 5 fuel, yields FuelRefill before finishing.
        let (out2, rem2) = run_simple(&m, &[10], Some(5));
        assert_eq!(out2, Step::Yield(YieldReason::FuelRefill));
        assert_eq!(rem2, 0);
    }

    #[test]
    fn tail_recursion_runs_deep_in_constant_frames() {
        // f(n, acc) = if n == 0 { acc } else { return_call f(n - 1, acc + n) }
        let m = Module {
            funcs: vec![Func {
                params: vec![ValType::I64, ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![
                    Block {
                        parent: 0,
                        captures: vec![],
                        params: vec![ValType::I64, ValType::I64],
                        insts: vec![Inst::Eqz {
                            ty: IntTy::I64,
                            a: 0,
                        }],
                        term: Terminator::BrIf {
                            cond: 2,
                            then_blk: 1,
                            then_args: vec![],
                            else_blk: 2,
                            else_args: vec![],
                        },
                    },
                    Block {
                        parent: 0,
                        captures: vec![(0, 1)],
                        params: vec![],
                        insts: vec![],
                        term: Terminator::Return(vec![0]),
                    },
                    Block {
                        parent: 0,
                        captures: vec![(0, 0), (0, 1)],
                        params: vec![],
                        insts: vec![
                            Inst::IntBinImm {
                                ty: IntTy::I64,
                                op: temen_ir::BinOp::Sub,
                                a: 0,
                                imm: 1,
                            },
                            Inst::IntBin {
                                ty: IntTy::I64,
                                op: temen_ir::BinOp::Add,
                                a: 1,
                                b: 0,
                            },
                        ],
                        term: Terminator::ReturnCall {
                            func: 0,
                            args: vec![2, 3],
                        },
                    },
                ],
            }],
            ..Module::default()
        };
        // Depth 2000 exceeds max_call_depth (1024), proving O(1) tail-call frame reuse!
        let (out, _) = run_simple(&m, &[2000, 0], None);
        assert_eq!(
            out,
            Step::Finished(ContextExit::Returned(vec![2000 * 2001 / 2]))
        );
    }

    #[test]
    fn indirect_call_checks_structural_type_id() {
        let m = Module {
            types: vec![
                TypeEntry::Func(FuncType {
                    params: vec![ValType::I64],
                    results: vec![ValType::I64],
                }),
                TypeEntry::Func(FuncType {
                    params: vec![ValType::I32],
                    results: vec![ValType::I32],
                }),
            ],
            funcs: vec![
                Func {
                    params: vec![ValType::I32, ValType::I64],
                    results: vec![ValType::I64],
                    blocks: vec![Block {
                        parent: 0,
                        captures: vec![],
                        params: vec![ValType::I32, ValType::I64],
                        insts: vec![Inst::CallIndirect(Box::new(temen_ir::CallIndirect {
                            ty: 0,
                            idx: 0,
                            args: Box::new([1]),
                        }))],
                        term: Terminator::Return(vec![2]),
                    }],
                },
                Func {
                    params: vec![ValType::I64],
                    results: vec![ValType::I64],
                    blocks: vec![Block {
                        parent: 0,
                        captures: vec![],
                        params: vec![ValType::I64],
                        insts: vec![Inst::IntBinImm {
                            ty: IntTy::I64,
                            op: temen_ir::BinOp::Add,
                            a: 0,
                            imm: 7,
                        }],
                        term: Terminator::Return(vec![1]),
                    }],
                },
            ],
            ..Module::default()
        };
        // Slot 1 matches `(i64) -> (i64)`.
        assert_eq!(
            run_simple(&m, &[1, 35], None).0,
            Step::Finished(ContextExit::Returned(vec![42]))
        );
        // Slot 0 has signature `(i32, i64) -> (i64)` -> traps IndirectCallType.
        assert_eq!(
            run_simple(&m, &[0, 35], None).0,
            Step::Finished(ContextExit::Trapped(Trap::IndirectCallType))
        );
    }

    #[test]
    fn setjmp_and_longjmp_unwind_callee_within_context() {
        // f0(): if setjmp(buf=0) == 0 { call f1(0, 99); return -1 } else { return r }
        // f1(buf, v): longjmp(buf, v); unreachable
        let m = Module {
            memory: Some(Memory {
                size_log2: 16,
                shadow: None,
            }),
            funcs: vec![
                Func {
                    params: vec![],
                    results: vec![ValType::I32],
                    blocks: vec![
                        Block {
                            parent: 0,
                            captures: vec![],
                            params: vec![],
                            insts: vec![
                                Inst::ConstI64(0),
                                Inst::SetJmp { buf: 0 }, // v1 = 0 first time, 99 on longjmp
                                Inst::Eqz {
                                    ty: IntTy::I32,
                                    a: 1,
                                }, // v2
                            ],
                            term: Terminator::BrIf {
                                cond: 2,
                                then_blk: 1,
                                then_args: vec![],
                                else_blk: 2,
                                else_args: vec![],
                            },
                        },
                        Block {
                            parent: 0,
                            captures: vec![(0, 0)],
                            params: vec![],
                            insts: vec![
                                Inst::ConstI32(99),
                                Inst::Call {
                                    func: 1,
                                    args: Box::new([0, 1]),
                                },
                                Inst::ConstI32(-1),
                            ],
                            term: Terminator::Return(vec![2]),
                        },
                        Block {
                            parent: 0,
                            captures: vec![(0, 1)],
                            params: vec![],
                            insts: vec![],
                            term: Terminator::Return(vec![0]),
                        },
                    ],
                },
                Func {
                    params: vec![ValType::I64, ValType::I32],
                    results: vec![],
                    blocks: vec![Block {
                        parent: 0,
                        captures: vec![],
                        params: vec![ValType::I64, ValType::I32],
                        insts: vec![Inst::LongJmp { buf: 0, val: 1 }],
                        term: Terminator::Unreachable,
                    }],
                },
            ],
            ..Module::default()
        };
        assert_eq!(
            run_simple(&m, &[], None).0,
            Step::Finished(ContextExit::Returned(vec![99]))
        );
    }
}
