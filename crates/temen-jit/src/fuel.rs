//! #1944 slice 3 — the counted-fuel cell a compile with fuel checks charges ([`crate::VmCtx::fuel`]).
//!
//! Compiled code burns [`FuelCell::left`] one unit per safepoint (word 0, a plain load/decrement/store)
//! and, when it is spent, calls the cell's refill (word 1) instead of trapping at once: the refill
//! draws the next chunk from the budget chain the embedder accounts in ([`BudgetNode`]), and only a
//! spent chain traps `OutOfFuel`. A cell with no source is a fixed allowance, refilled never — a
//! carve child's, or a harness's `u64` budget. The JIT meters; the embedder's budget tree accounts.
//!
//! The refill is called through a `PreserveAll` trampoline ([`refill_trampoline`]), which clobbers
//! no register. A plain call would clobber the caller-saved registers at every safepoint, so a value
//! live across one would have to sit in a callee-saved register, which the function's prologue then
//! saves and restores on every call, refill or not (#2147).

use std::sync::{Arc, Mutex};

/// A domain's node in the embedder's budget chain (temen-run's, over a `temen_interp` budget node):
/// where a [`FuelCell`] draws its next chunk, and what a domain's vCPUs (#2001) and fibers (#2112)
/// are charged to.
pub trait BudgetNode: Send + Sync {
    /// Up to a chunk of fuel from the chain, charged to every level: `None` when every level is
    /// unbounded (nothing to meter), `Some(0)` when the chain is spent.
    fn draw(&self) -> Option<u64>;
    /// Hand back `unspent`, the rest of a draw its cell did not burn.
    fn give_back(&self, unspent: u64);
    /// The fuel room left along the chain, `-1` when every level is unbounded.
    fn room(&self) -> i64;
    /// Charge one live vCPU to every level, all or nothing: `false` when a level's `spawn` is full.
    fn charge_vcpu(&self) -> bool;
    /// [`Self::charge_vcpu`] past any ceiling: a vCPU a thaw re-creates, which lived before the freeze.
    fn force_vcpu(&self);
    /// Hand back a vCPU's charge when it ends.
    fn vcpu_ended(&self);
    /// #2112 — charge one live fiber, [`temen_ir::FIBER_STACK`] of `mem`, to every level, all or
    /// nothing: `false` when a level's `mem` is full.
    fn charge_fiber(&self) -> bool;
    /// [`Self::charge_fiber`] past any ceiling: a fiber a thaw re-creates, which lived before the
    /// freeze.
    fn force_fiber(&self);
    /// Hand back a fiber's charge when it ends.
    fn fiber_ended(&self);
}

/// A domain's counted-fuel cell. `repr(C)`: compiled code reads `left` at word 0 and calls `refill`
/// (word 1) and `remaining` (word 2) through the cell's own pointer, so no address but the cell's is
/// baked into code.
#[repr(C)]
pub struct FuelCell {
    /// What is left of the last draw (or of a fixed allowance).
    pub left: u64,
    /// [`refill_trampoline`]'s address: called with the cell, returns nothing, clobbers no register.
    refill: usize,
    remaining: unsafe extern "C" fn(*const FuelCell) -> i64,
    /// What the last refill handed its caller to charge from: the allowance it drew or found, `0`
    /// when the chain is spent. Compiled code reads it after a refill instead of `left`, which a
    /// sibling vCPU's plain store can have overwritten ([`fuel_refill`]).
    granted: u64,
    /// The chain the next draw comes from; `None` for a fixed allowance (or once the chain proved
    /// unbounded). Locked by a refill, so the vCPUs of a domain sharing the cell draw one at a time.
    src: Mutex<Option<Arc<dyn BudgetNode>>>,
}

impl FuelCell {
    /// A fixed allowance of `n`, never refilled.
    pub fn fixed(n: u64) -> Box<FuelCell> {
        Self::with(n, None)
    }

    /// A cell that draws from `src`, none drawn yet: the first safepoint draws.
    pub fn drawing(src: Arc<dyn BudgetNode>) -> Box<FuelCell> {
        Self::with(0, Some(src))
    }

    /// [`Self::drawing`] when `src`'s chain is bounded; `None` when every level is unbounded — code
    /// with nothing to meter compiles without fuel checks.
    pub fn metering(src: Arc<dyn BudgetNode>) -> Option<Box<FuelCell>> {
        (src.room() >= 0).then(|| Self::drawing(src))
    }

    fn with(left: u64, src: Option<Arc<dyn BudgetNode>>) -> Box<FuelCell> {
        Box::new(FuelCell {
            left,
            refill: refill_trampoline(),
            remaining: fuel_remaining,
            granted: 0,
            src: Mutex::new(src),
        })
    }

    /// #2113 — point a cell that compiled code keeps across runs (a compile-once program's) at the
    /// chain its next run draws from: what the last draw left goes back to its chain first. `None`
    /// detaches it, so code run before the next re-arm traps `OutOfFuel`. Only between runs: no
    /// code may be charging the cell.
    pub fn rearm(&mut self, src: Option<Arc<dyn BudgetNode>>) {
        let slot = self.src.get_mut().unwrap_or_else(|e| e.into_inner());
        let left = std::mem::take(&mut self.left);
        if let Some(s) = slot.take() {
            s.give_back(left);
        }
        *slot = src;
    }

    /// What this cell's vCPUs can still burn: `left` plus the chain's room, `u64::MAX` unmetered.
    pub fn can_burn(&self) -> u64 {
        let left = unsafe { core::ptr::read_volatile(&self.left) };
        match &*self.src.lock().unwrap_or_else(|e| e.into_inner()) {
            None => left,
            Some(s) => match s.room() {
                r if r < 0 => u64::MAX,
                r => left.saturating_add(r as u64),
            },
        }
    }
}

impl Drop for FuelCell {
    fn drop(&mut self) {
        if let Some(s) = self.src.get_mut().unwrap_or_else(|e| e.into_inner()) {
            s.give_back(self.left);
        }
    }
}

/// A spent cell's refill, called by compiled code through [`refill_trampoline`]: it puts the new
/// allowance (at least 1) in `left`, and in [`FuelCell::granted`] what the caller charges from: that
/// allowance, or `0` when the chain (or a fixed allowance) is spent, on which the caller traps
/// `OutOfFuel`. A domain's vCPUs share one cell, so a refill another vCPU made while this one waited
/// for the lock is kept, not drawn again: what it left is granted.
///
/// The caller charges from `granted`, never from a reload of `left`. The vCPUs charge the shared
/// cell with plain loads and stores, so a sibling that loaded `1` before a refill can store its `0`
/// just after it. A reload of `left` that read that `0` trapped `OutOfFuel` with the chain still
/// full (#2202: five threads in a loop trapped within milliseconds). Only a refill writes `granted`,
/// under the lock, and it writes `0` only when the chain is spent.
///
/// # Safety
/// `cell` is a live [`FuelCell`].
unsafe extern "C" fn fuel_refill(cell: *mut FuelCell) {
    // Only the `src` field is borrowed: compiled code writes `left` through the raw cell pointer.
    let mut src = (*cell).src.lock().unwrap_or_else(|e| e.into_inner());
    let left = core::ptr::addr_of_mut!((*cell).left);
    let granted = core::ptr::addr_of_mut!((*cell).granted);
    let refilled = core::ptr::read_volatile(left);
    if refilled > 0 {
        core::ptr::write_volatile(granted, refilled);
        return;
    }
    let drawn = match src.as_ref().map(|s| s.draw()) {
        None | Some(Some(0)) => 0,
        Some(None) => {
            *src = None; // every level unbounded: nothing left to meter
            u64::MAX
        }
        Some(Some(n)) => n,
    };
    core::ptr::write_volatile(left, drawn);
    core::ptr::write_volatile(granted, drawn);
}

/// The address of [`fuel_refill`] behind a trampoline in Cranelift's `PreserveAll` convention, which
/// every [`FuelCell`] calls through. The trampoline saves whatever the platform call to
/// [`fuel_refill`] clobbers, so the safepoint that calls it clobbers nothing. Compiled once per
/// process, by the same ISA configuration as every guest compile, and never freed: every cell points
/// at it.
fn refill_trampoline() -> usize {
    static TRAMPOLINE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *TRAMPOLINE.get_or_init(|| {
        compile_refill_trampoline().unwrap_or_else(|e| {
            panic!("the fuel refill trampoline must compile wherever the JIT does: {e:?}")
        })
    })
}

fn compile_refill_trampoline() -> Result<usize, crate::JitError> {
    use cranelift_codegen::ir::{types::I64, AbiParam, InstBuilder, Signature};
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use cranelift_module::{Linkage, Module};
    let backend = |e: cranelift_module::ModuleError| crate::JitError::Backend(e.to_string());
    let mut module = crate::new_jit_module()?;
    let mut sig = Signature::new(CallConv::PreserveAll);
    sig.params.push(AbiParam::new(I64)); // the cell
    let id = module
        .declare_function("fuel_refill_trampoline", Linkage::Local, &sig)
        .map_err(backend)?;
    let mut ctx = module.make_context();
    ctx.func.signature = sig;
    let mut fctx = FunctionBuilderContext::new();
    let mut b = FunctionBuilder::new(&mut ctx.func, &mut fctx);
    let entry = b.create_block();
    b.append_block_params_for_function_params(entry);
    b.switch_to_block(entry);
    b.seal_block(entry);
    let cell = b.block_params(entry)[0];
    let mut inner = Signature::new(module.isa().default_call_conv());
    inner.params.push(AbiParam::new(I64));
    let inner = b.import_signature(inner);
    let refill = b
        .ins()
        .iconst(I64, fuel_refill as *const () as usize as i64);
    b.ins().call_indirect(inner, refill, &[cell]);
    b.ins().return_(&[]);
    b.finalize();
    module.define_function(id, &mut ctx).map_err(backend)?;
    module.finalize_definitions().map_err(backend)?;
    let code = module.get_finalized_function(id) as usize;
    // The code lives in the module's memory, which must outlive every cell: keep it for the process.
    std::mem::forget(module);
    Ok(code)
}

/// `fuel.remaining` (self op 13) on a metered compile: [`FuelCell::can_burn`], `i64::MAX` unmetered.
///
/// # Safety
/// `cell` is a live [`FuelCell`].
unsafe extern "C" fn fuel_remaining(cell: *const FuelCell) -> i64 {
    (*cell).can_burn().min(i64::MAX as u64) as i64
}

/// Word offsets compiled code uses.
pub(crate) const REFILL_OFF: i32 = 8;
pub(crate) const REMAINING_OFF: i32 = 16;
pub(crate) const GRANTED_OFF: i32 = 24;

const _: () = {
    assert!(core::mem::offset_of!(FuelCell, left) == 0);
    assert!(core::mem::offset_of!(FuelCell, refill) == REFILL_OFF as usize);
    assert!(core::mem::offset_of!(FuelCell, remaining) == REMAINING_OFF as usize);
    assert!(core::mem::offset_of!(FuelCell, granted) == GRANTED_OFF as usize);
};

#[cfg(test)]
mod tests {
    use super::{BudgetNode, FuelCell};
    use crate::{JitOutcome, RunOpts, TrapKind};
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;
    use temen_interp::Value;
    use temen_ir::{
        BinOp, Block, CastOp, CmpOp, FBinOp, FloatTy, Func, Inst, IntTy, Module, Terminator,
        ValType,
    };

    /// A chain that hands out one unit per draw, `left` of them, then is spent: every safepoint
    /// after the first runs the refill.
    struct Drip {
        left: AtomicU64,
        draws: AtomicU64,
    }

    impl Drip {
        fn new(units: u64) -> Arc<Drip> {
            Arc::new(Drip {
                left: AtomicU64::new(units),
                draws: AtomicU64::new(0),
            })
        }
    }

    impl BudgetNode for Drip {
        fn draw(&self) -> Option<u64> {
            // Work the float registers, as a real chain's host code may: a trampoline that did not
            // save them would hand the loop back clobbered floats.
            let x: f64 = (0..16)
                .map(|i| std::hint::black_box(f64::from(i)) * 1.5)
                .sum();
            std::hint::black_box(x);
            self.draws.fetch_add(1, Ordering::Relaxed);
            let had = self
                .left
                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1));
            Some(u64::from(had.is_ok()))
        }
        fn give_back(&self, _: u64) {}
        fn room(&self) -> i64 {
            self.left.load(Ordering::Relaxed).min(i64::MAX as u64) as i64
        }
        fn charge_vcpu(&self) -> bool {
            true
        }
        fn force_vcpu(&self) {}
        fn vcpu_ended(&self) {}
        fn charge_fiber(&self) -> bool {
            true
        }
        fn force_fiber(&self) {}
        fn fiber_ended(&self) {}
    }

    const INTS: u32 = 12;
    const FLOATS: u32 = 8;

    /// `f(n)`: `n` turns of a loop that carries 12 `i64`s and 8 `f64`s around its back edge, which
    /// charges fuel. That is more values than the callee-saved registers hold (and every `xmm`
    /// register is caller-saved on System V), so some ride across the refill in registers a platform
    /// call would clobber. Returns all of them folded into one `i64`.
    fn many_live_values() -> Module {
        let (k, j) = (INTS, FLOATS);
        let carried: Vec<ValType> = std::iter::once(ValType::I64)
            .chain((0..k).map(|_| ValType::I64))
            .chain((0..j).map(|_| ValType::F64))
            .collect();
        let all = |n: u32| (0..n).collect::<Vec<u32>>();
        let p = 1 + k + j; // values a loop block's params take
                           // b0(n): the starting values, then into the loop.
        let mut b0 = vec![];
        for x in 0..k {
            b0.push(Inst::ConstI64(i64::from(x) + 1));
        }
        for x in 0..j {
            b0.push(Inst::ConstF64((f64::from(x) + 1.0).to_bits()));
        }
        let b0 = Block {
            params: vec![ValType::I64],
            insts: b0,
            term: Terminator::Br {
                target: 1,
                args: all(1 + k + j),
            },
        };
        // b1(i, a.., f..): i != 0 ? b2 : b3.
        let b1 = Block {
            params: carried.clone(),
            insts: vec![
                Inst::ConstI64(0),
                Inst::IntCmp {
                    ty: IntTy::I64,
                    op: CmpOp::Ne,
                    a: 0,
                    b: p,
                },
            ],
            term: Terminator::BrIf {
                cond: p + 1,
                then_blk: 2,
                then_args: all(p),
                else_blk: 3,
                else_args: (1..p).collect(),
            },
        };
        // b2(i, a.., f..): i - 1, each a += its neighbour, each f += its neighbour; back to b1.
        let mut b2 = vec![
            Inst::ConstI64(1),
            Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Sub,
                a: 0,
                b: p,
            },
        ];
        for x in 0..k {
            b2.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: 1 + x,
                b: 1 + (x + 1) % k,
            });
        }
        for x in 0..j {
            b2.push(Inst::FBin {
                ty: FloatTy::F64,
                op: FBinOp::Add,
                a: 1 + k + x,
                b: 1 + k + (x + 1) % j,
            });
        }
        let b2 = Block {
            params: carried.clone(),
            insts: b2,
            term: Terminator::Br {
                target: 1,
                args: (p + 1..p + 2 + k + j).collect(),
            },
        };
        // b3(a.., f..): the sum of the ints and of the floats' bits.
        let mut b3 = vec![];
        for x in 0..j {
            b3.push(Inst::Cast {
                op: CastOp::ReinterpF64I64,
                a: k + x,
            });
        }
        let mut acc = 0;
        for (n, x) in (1..k).chain(k + j..k + j + j).enumerate() {
            b3.push(Inst::IntBin {
                ty: IntTy::I64,
                op: BinOp::Add,
                a: acc,
                b: x,
            });
            acc = k + j + j + n as u32; // the sum so far: the add just pushed
        }
        let b3 = Block {
            params: carried[1..].to_vec(),
            insts: b3,
            term: Terminator::Return(vec![acc]),
        };
        let m = Module {
            funcs: vec![Func {
                params: vec![ValType::I64],
                results: vec![ValType::I64],
                blocks: vec![b0, b1, b2, b3],
            }],
            ..Default::default()
        };
        temen_verify::verify_module(&m).expect("the fixture verifies");
        m
    }

    fn run_jit(m: &Module, n: i64, cell: &mut FuelCell) -> JitOutcome {
        crate::run_inner(
            m,
            0,
            &[n],
            crate::empty_cap_thunk,
            core::ptr::null_mut(),
            RunOpts {
                fuel: Some(cell),
                ..RunOpts::default()
            },
        )
        .expect("compiles and runs")
        .0
    }

    #[test]
    fn values_live_across_a_refill_keep_their_registers() {
        let m = many_live_values();
        let n = 2000;
        let mut fuel = u64::MAX;
        let want = match temen_interp::run(&m, 0, &[Value::I64(n)], &mut fuel)
            .expect("the interpreter runs it")[..]
        {
            [Value::I64(v)] => v,
            ref other => panic!("one i64 result, got {other:?}"),
        };
        let drip = Drip::new(u64::MAX);
        let mut cell = FuelCell::drawing(drip.clone());
        assert_eq!(run_jit(&m, n, &mut cell), JitOutcome::Returned(vec![want]));
        assert_eq!(
            drip.draws.load(Ordering::Relaxed),
            n as u64 + 1,
            "the entry and every back edge refilled"
        );
    }

    #[test]
    fn a_spent_chain_still_traps_out_of_fuel() {
        let m = many_live_values();
        let drip = Drip::new(10);
        let mut cell = FuelCell::drawing(drip.clone());
        assert_eq!(
            run_jit(&m, 100, &mut cell),
            JitOutcome::Trapped(TrapKind::OutOfFuel)
        );
        assert_eq!(
            drip.draws.load(Ordering::Relaxed),
            11,
            "ten draws of one unit, then the one that found the chain spent"
        );
    }
}
