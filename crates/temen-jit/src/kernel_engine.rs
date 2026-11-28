//! Stage 1d (`#3197`): Lean Cranelift backend targeting `temen-kernel` + `temen-runtime`.
//!
//! ## Architecture (`#3187` red-team refinements #1–#4)
//! - **Acyclic DAG:** `temen-kernel` (`#![forbid(unsafe_code)]`) <- `temen-jit::kernel_engine`
//!   <- `temen-runtime` (`Engine::Cranelift`).
//! - **Lean `KernelVmCtx`:** offset 0 is `domain_poll: *const AtomicU64` (pointing directly to
//!   `Domain`'s shared poll word; L1-shared read-only on the hot path), followed by Carrier-owned
//!   `fuel: *mut FuelCell` (Decision 30), `runtime_ctx`, window geometry (`mem_base`,
//!   `mem_reserved`, `mem_mapped`), unified `fn_table` carrying canonical 64-bit structural
//!   `TypeId`s (Invariant 10), `pages`, Stage 4 `got` slot, `trap_code`, and `carrier_tls`.
//! - **Hardware guard reuse:** runs guest code through `crate::mem::GuestWindow` +
//!   `crate::mem::run_guarded_kernel` (audited Linux/macOS/Windows SIGSEGV/VEH trap capture),
//!   with one outermost guard per Carrier OS thread (zero nested `sigsetjmp` scopes, `#3158`).
//! - **Configurable confinement (`BoundMode`):** supports both `BakedImmediate` and
//!   `DynamicVmCtx` (`mem_reserved` loaded once at function entry from `KernelVmCtx`) so we can
//!   benchmark dynamic bounds directly for Stage 4 compile-once PIC shared libraries.

use core::ffi::c_void;
use core::sync::atomic::{AtomicI64, AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types::{F32, F64, I16, I32, I64, I8};
use cranelift_codegen::ir::{
    AbiParam, AtomicRmwOp as ClifRmwOp, Block as ClifBlock, BlockArg, Endianness, FuncRef,
    InstBuilder, JumpTableData, MemFlags, SigRef, Signature, StackSlotData, StackSlotKind, Type,
    Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};

use temen_ir::{
    AtomicRmwOp, BinOp, CastOp, CmpOp, ConvOp, FBinOp, FCmpOp, FUnOp, FloatTy, Func, FuncType,
    Inst, IntTy, IntUnOp, Module as IrModule, Ordering as IrOrdering, Terminator, TypeEntry,
    ValIdx, ValType,
};
use temen_kernel::{FuelCell, Trap, TypeId, MAX_CAP_ARGS};

use crate::mem::{self, GuestWindow, Prot};

/// How memory bounds checks lower the window reservation bound (red-team refinement #4).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BoundMode {
    /// Bake the compile-time `reserved` constant as an immediate in each bounds check (current
    /// `temen-jit` posture; ties machine code to one `size_log2`).
    BakedImmediate,
    /// Load `mem_reserved` once per function entry from `KernelVmCtx` into an SSA register so the
    /// same compiled machine code runs on any domain's window size (Stage 4 PIC-library ready).
    DynamicVmCtx,
}

/// Unified indirect dispatch table entry (canonical 64-bit structural `TypeId`, Invariant 10).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FnTableEntry {
    pub code_ptr: *const u8,
    pub type_id: u64,
}

// SAFETY: code pointers point into immutable finalized JIT memory; `type_id` is plain data.
unsafe impl Send for FnTableEntry {}
unsafe impl Sync for FnTableEntry {}

/// Lean per-Carrier execution context passed as hidden parameter 0 (`vmctx`) to every compiled
/// guest function (`#3187` comment 1 + red-team refinement #2).
#[repr(C)]
pub struct KernelVmCtx {
    /// Offset 0: pointer to `Domain`'s shared safepoint poll `AtomicU64`.
    pub domain_poll: *const AtomicU64,
    /// Offset 8: pointer to the active Carrier's `FuelCell` (Decision 30).
    pub fuel: *mut FuelCell,
    /// Offset 16: opaque pointer to `&KernelBridgeHolder` for runtime thunks.
    pub runtime_ctx: *mut c_void,
    /// Offset 24: domain linear-memory window host base (`null` if memoryless).
    pub mem_base: *mut u8,
    /// Offset 32: domain window reservation length in bytes (power of two; also `guard_offset`).
    pub mem_reserved: u64,
    /// Offset 40: domain window backed/mapped length in bytes.
    pub mem_mapped: u64,
    /// Offset 48: unified dispatch table base (`*const FnTableEntry`).
    pub fn_table: *const FnTableEntry,
    /// Offset 56: `next_pow2(nfuncs) - 1` function-table index mask.
    pub fn_table_mask: u64,
    /// Offset 64: optional demand-paged access table (`null` when not demand-paged).
    pub pages: *const AtomicU8,
    /// Offset 72: Stage 4 forward-compat per-domain GOT pointer (`null` until Stage 4).
    pub got: *const c_void,
    /// Offset 80: per-Context software trap word (`0` = running, else `temen_ir::trap_code::*`).
    pub trap_code: AtomicI64,
    /// Offset 88: pointer to active Carrier's `tls: i64` (`vcpu.tls.get`/`set`).
    pub carrier_tls: *mut i64,
}

const OFF_DOMAIN_POLL: i32 = 0;
const OFF_FUEL: i32 = 8;
const OFF_MEM_BASE: i32 = 24;
const OFF_MEM_RESERVED: i32 = 32;
const OFF_FN_TABLE: i32 = 48;
const OFF_FN_TABLE_MASK: i32 = 56;
const OFF_TRAP_CODE: i32 = 80;
const OFF_CARRIER_TLS: i32 = 88;

const _: () = {
    assert!(core::mem::offset_of!(KernelVmCtx, domain_poll) == OFF_DOMAIN_POLL as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, fuel) == OFF_FUEL as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, mem_base) == OFF_MEM_BASE as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, mem_reserved) == OFF_MEM_RESERVED as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, fn_table) == OFF_FN_TABLE as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, fn_table_mask) == OFF_FN_TABLE_MASK as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, trap_code) == OFF_TRAP_CODE as usize);
    assert!(core::mem::offset_of!(KernelVmCtx, carrier_tls) == OFF_CARRIER_TLS as usize);
    assert!(core::mem::size_of::<FnTableEntry>() == 16);
};

impl KernelVmCtx {
    /// Record the first software trap on this context (`0` -> `code`).
    #[inline]
    pub fn raise_trap(&self, trap: Trap) {
        let _ =
            self.trap_code
                .compare_exchange(0, trap.code(), Ordering::AcqRel, Ordering::Acquire);
    }

    /// Read any recorded software trap on this context.
    #[inline]
    pub fn take_trap(&self) -> Option<Trap> {
        let code = self.trap_code.load(Ordering::Acquire);
        if code == 0 {
            None
        } else {
            Trap::from_code(code)
        }
    }
}

/// Runtime callbacks implemented by `temen-runtime` (`DomainWorld`) and invoked from compiled
/// native Cranelift code via `extern "C"` thunks.
///
/// Every method takes safe Rust references and returns `Result<..., Trap>`, keeping both
/// `temen-jit` acyclic with `temen-runtime` (red-team refinement #1) and `temen-runtime`
/// `#![forbid(unsafe_code)]`.
pub trait KernelBridge: Send + Sync {
    /// Handle non-zero `PollFlags` observed at a safepoint.
    fn on_poll(&self, flags: u64, fuel: &mut FuelCell) -> Result<(), Trap>;

    /// Refill `fuel` from the domain's `BudgetTree` and charge 1 unit for the current safepoint.
    fn on_fuel_refill(&self, fuel: &mut FuelCell) -> Result<(), Trap>;

    /// `memory.wait`: compare and park via `WaitIndex`. When running inside a `KernelFiber`,
    /// parks via [`park_current_fiber`] (`FIBER_PARKED`) without blocking the Carrier OS thread.
    fn futex_wait(
        &self,
        addr: u64,
        expected: i64,
        width64: bool,
        timeout_ns: i64,
        fuel: &mut FuelCell,
    ) -> Result<i32, Trap>;

    /// `memory.notify`: wake up to `count` waiters on `addr`.
    fn futex_notify(&self, addr: u64, count: u32) -> Result<i32, Trap>;

    /// `thread.spawn`: start a concurrent OS thread running `func_idx(arg0, arg1)`.
    fn thread_spawn(&self, func_idx: u32, arg0: i64, arg1: i64) -> Result<i32, Trap>;

    /// `thread.join`: wait for `handle`; returns `(status, exit_val)` where `status` is `0` or
    /// `-EBADF`.
    fn thread_join(&self, handle: i32, fuel: &mut FuelCell) -> Result<(i64, i64), Trap>;

    /// `cont.new`: allocate a `GuestFiber` backed by a [`KernelFiber`] control stack.
    fn fiber_new(&self, func_idx: u32, arg0: i64) -> Result<i64, Trap>;

    /// `cont.resume` / `cont.resume.block`: claim & switch into `GuestFiber` `handle`. Returns
    /// `(status, value)` where `status` is `0`/`1`/`3`/`-9`.
    fn fiber_resume(
        &self,
        handle: i64,
        val: i64,
        block: bool,
        fuel: &mut FuelCell,
        carrier_tls: &mut i64,
    ) -> Result<(i32, i64), Trap>;

    /// `suspend`: yield from the running `GuestFiber` back to its resumer delivering `val`,
    /// returning the next resume's delivered argument.
    fn fiber_suspend(&self, val: i64) -> Result<i64, Trap>;

    /// `call.cap`: dispatch capability call; write results into `results`.
    #[allow(clippy::too_many_arguments)]
    fn cap_call(
        &self,
        handle: i32,
        type_id: u32,
        op: u32,
        args: &[i64],
        results: &mut [i64],
    ) -> Result<(), Trap>;
}

/// Thin stack/heap holder whose address lives in `KernelVmCtx.runtime_ctx`.
pub struct KernelBridgeHolder {
    pub bridge: Arc<dyn KernelBridge>,
}

/// Hardware guard-paged linear memory window for `Engine::Cranelift` (reuses `mem::GuestWindow`).
pub struct KernelMemory {
    window: GuestWindow,
    reserved: u64,
    mapped: u64,
}

// SAFETY: `GuestWindow` is an OS virtual memory mapping with no thread affinity; concurrent guest
// accesses synchronize via hardware atomic instructions and memory barriers.
unsafe impl Send for KernelMemory {}
unsafe impl Sync for KernelMemory {}

impl KernelMemory {
    /// Allocate a hardware guard-paged window for `size_log2`, seed `initial_data`, apply
    /// read-only page protections, and install the Invariant 13 `POWERBOX_NULL_GUARD`.
    pub fn try_new(size_log2: u8, initial_data: &[KernelInitialData]) -> Option<Arc<KernelMemory>> {
        let bytes = 1usize.checked_shl(u32::from(size_log2))?;
        let page = mem::page_size();
        let alloc_reserved = bytes.max(page);
        let mut window = GuestWindow::try_new(bytes, alloc_reserved)?;
        {
            let slice = window.rw_mut();
            for d in initial_data {
                let start = usize::try_from(d.offset).ok()?;
                let end = start.checked_add(d.bytes.len())?;
                if end > slice.len() {
                    return None;
                }
                slice[start..end].copy_from_slice(&d.bytes);
            }
        }
        // Apply read-only protections on read-only data segments (page-aligned spans).
        for d in initial_data {
            if d.readonly && !d.bytes.is_empty() {
                let p_u64 = page as u64;
                let aligned_off = d.offset & !(p_u64 - 1);
                let end = d.offset.checked_add(d.bytes.len() as u64)?;
                let aligned_len = (end - aligned_off).div_ceil(p_u64) * p_u64;
                if !window.protect(aligned_off, aligned_len, Prot::Ro) {
                    return None;
                }
            }
        }
        // Invariant 13 (#1094): unconditionally unmap [0, POWERBOX_NULL_GUARD) when the window is
        // at least that large.
        let guard = temen_ir::module_null_guard();
        if guard > 0 && guard <= bytes as u64 {
            let _ = window.seed_null_guard(guard);
        }
        Some(Arc::new(KernelMemory {
            window,
            reserved: bytes as u64,
            mapped: bytes as u64,
        }))
    }

    /// Empty placeholder window for memoryless modules.
    pub fn empty() -> Arc<KernelMemory> {
        Arc::new(KernelMemory {
            window: GuestWindow::try_new(0, 0).expect("empty GuestWindow"),
            reserved: 0,
            mapped: 0,
        })
    }

    #[inline]
    pub fn base(&self) -> *mut u8 {
        self.window.base()
    }

    #[inline]
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    #[inline]
    pub fn mapped(&self) -> u64 {
        self.mapped
    }

    /// Atomic load from a naturally-aligned, in-bounds guest offset (used by `WaitIndex` futex
    /// compare predicates in `temen-runtime`).
    pub fn atomic_load(&self, addr: u64, width64: bool) -> Option<u64> {
        let width = if width64 { 8u64 } else { 4u64 };
        let guard = temen_ir::module_null_guard();
        if guard > 0 && guard <= self.mapped && addr < guard {
            return None;
        }
        if !addr.is_multiple_of(width) || addr.checked_add(width).is_none_or(|e| e > self.mapped) {
            return None;
        }
        // SAFETY: `[addr, addr+width)` is committed and naturally aligned inside `self.window`.
        unsafe {
            let ptr = self.window.base().add(addr as usize);
            if width64 {
                let a = &*(ptr as *const AtomicU64);
                Some(a.load(Ordering::SeqCst))
            } else {
                let a = &*(ptr as *const core::sync::atomic::AtomicU32);
                Some(u64::from(a.load(Ordering::SeqCst)))
            }
        }
    }

    /// Run `code` (a `CompiledKernelModule` trampoline) under the OS hardware guard (`SIGSEGV`/VEH).
    /// Returns `true` if a hardware memory fault in the window range was caught (`MemoryFault`).
    ///
    /// # Safety
    /// `code`, `args`, `results`, `fn_table`, and `vmctx` must be valid for the call duration.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run_guarded(
        &self,
        code: *const u8,
        args: *const i64,
        results: *mut i64,
        fn_table: *const FnTableEntry,
        vmctx: *mut KernelVmCtx,
    ) -> bool {
        mem::run_guarded_kernel(
            &self.window,
            code,
            args,
            results,
            self.window.base(),
            fn_table as *const c_void,
            vmctx as *mut i64,
        )
    }
}

/// Static data segment copied into a fresh `KernelMemory`.
#[derive(Clone, Debug)]
pub struct KernelInitialData {
    pub offset: u64,
    pub readonly: bool,
    pub bytes: Vec<u8>,
}

/// Compilation error for `kernel_engine`.
#[derive(Clone, Debug)]
pub enum KernelCompileError {
    Unsupported(&'static str),
    InvalidTypeRef(u32),
    Backend(String),
}

impl std::fmt::Display for KernelCompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelCompileError::Unsupported(op) => {
                write!(f, "unsupported op in kernel_engine: {op}")
            }
            KernelCompileError::InvalidTypeRef(t) => write!(f, "invalid type index {t}"),
            KernelCompileError::Backend(msg) => write!(f, "cranelift backend error: {msg}"),
        }
    }
}

impl std::error::Error for KernelCompileError {}

/// Immutable finalized Cranelift native code artifact for `Engine::Cranelift`.
pub struct CompiledKernelModule {
    _jit: crate::code_ranges::NewModule,
    pub fn_table: Box<[FnTableEntry]>,
    pub fn_table_mask: u64,
    pub trampolines: Vec<*const u8>,
    pub func_arities: Vec<(u16, u8)>,
    pub memory_size_log2: Option<u8>,
    pub initial_data: Vec<KernelInitialData>,
    pub exports: Vec<(String, u32)>,
    pub bound_mode: BoundMode,
}

// SAFETY: `JITModule`'s finalized code pages are immutable and executable from any OS thread.
unsafe impl Send for CompiledKernelModule {}
unsafe impl Sync for CompiledKernelModule {}

impl CompiledKernelModule {
    /// Check whether `func_idx` exists and matches canonical `TypeId::thread_entry()` or
    /// `TypeId::fiber_entry()`.
    #[inline]
    pub fn func_type_id(&self, func_idx: u32) -> Option<TypeId> {
        let idx = func_idx as usize;
        if idx >= self.func_arities.len() {
            return None;
        }
        Some(TypeId(self.fn_table[idx].type_id))
    }

    /// Execute `func_idx(args)` under the OS hardware guard (`SIGSEGV`/VEH) on the calling
    /// Carrier thread. Safe entry point for `temen-runtime` (`#![forbid(unsafe_code)]`).
    #[allow(clippy::too_many_arguments)]
    pub fn run_entry(
        &self,
        func_idx: u32,
        args: &[i64],
        memory: &KernelMemory,
        domain_poll: &AtomicU64,
        fuel: &mut FuelCell,
        carrier_tls: &mut i64,
        bridge: Arc<dyn KernelBridge>,
    ) -> Result<Vec<i64>, Trap> {
        let idx = func_idx as usize;
        let &(nparams, nresults) = self.func_arities.get(idx).ok_or(Trap::IndirectCallType)?;
        if args.len() != usize::from(nparams) {
            return Err(Trap::IndirectCallType);
        }
        let tramp = self.trampolines[idx];
        let holder = KernelBridgeHolder { bridge };
        let mut results = vec![0i64; usize::from(nresults)];
        let mut vmctx = KernelVmCtx {
            domain_poll: domain_poll as *const AtomicU64,
            fuel: fuel as *mut FuelCell,
            runtime_ctx: (&holder as *const KernelBridgeHolder) as *mut c_void,
            mem_base: memory.base(),
            mem_reserved: memory.reserved(),
            mem_mapped: memory.mapped(),
            fn_table: self.fn_table.as_ptr(),
            fn_table_mask: self.fn_table_mask,
            pages: core::ptr::null(),
            got: core::ptr::null(),
            trap_code: AtomicI64::new(0),
            carrier_tls: carrier_tls as *mut i64,
        };
        let saved_fiber_depth = active_fiber_depth();
        // SAFETY: `holder`, `args`, `results`, `memory`, `domain_poll`, `fuel`, `carrier_tls`,
        // and `vmctx` are all pinned on the current stack frame for the entire guarded call.
        let faulted = unsafe {
            memory.run_guarded(
                tramp,
                args.as_ptr(),
                results.as_mut_ptr(),
                self.fn_table.as_ptr(),
                &mut vmctx as *mut KernelVmCtx,
            )
        };
        // If a hardware `SIGSEGV`/VEH unwound across a nested `KernelFiber::resume`, trim any
        // orphaned thread-local active-fiber frames immediately.
        truncate_active_fibers(saved_fiber_depth);
        if faulted {
            return Err(Trap::MemoryFault);
        }
        if let Some(trap) = vmctx.take_trap() {
            return Err(trap);
        }
        Ok(results)
    }

    /// Direct raw-ABI fiber entry trampoline: invokes `trampolines[slot](vmctx, arg0, arg1)`
    /// after verifying canonical `TypeId::fiber_entry()`.
    ///
    /// # Safety
    /// `vmctx` must be a live, initialized `KernelVmCtx` for the active Carrier.
    pub unsafe fn call_fiber_entry(
        &self,
        func_idx: u32,
        vmctx: *mut KernelVmCtx,
        arg0: i64,
        arg1: i64,
    ) -> Result<i64, Trap> {
        let idx = func_idx as usize;
        if idx >= self.func_arities.len() {
            return Err(Trap::IndirectCallType);
        }
        let entry = &self.fn_table[idx];
        if entry.type_id != TypeId::fiber_entry().0 {
            return Err(Trap::IndirectCallType);
        }
        let tramp = self.trampolines[idx];
        let args = [arg0, arg1];
        let mut results = [0i64; 1];
        let f: unsafe extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64) =
            core::mem::transmute(tramp);
        let ctx = &*vmctx;
        f(
            args.as_ptr(),
            results.as_mut_ptr(),
            ctx.mem_base,
            ctx.fn_table as *const c_void,
            vmctx as *mut i64,
        );
        if let Some(t) = ctx.take_trap() {
            return Err(t);
        }
        Ok(results[0])
    }
}

// ---- Native migratable `KernelFiber` over `temen-fiber` ----------------------------------------

/// Default native control-stack size for a `KernelFiber` (256 KiB, matching `temen-jit`).
const KERNEL_FIBER_STACK_BYTES: usize = 256 * 1024;

const YIELD_KIND_NONE: u8 = 0;
const YIELD_KIND_SUSPEND: u8 = 1;
const YIELD_KIND_PARK: u8 = 2;

/// Out-of-band discriminator between guest `suspend(val)` (arbitrary 64-bit `val`) and
/// `FIBER_PARKED` (blocking `memory.wait` / `thread.join` inside a fiber).
struct FiberYieldSlot {
    kind: core::cell::Cell<u8>,
}

// SAFETY: only accessed by the single Carrier currently holding exclusive ownership of the fiber.
unsafe impl Send for FiberYieldSlot {}

#[cfg(fiber_rt)]
struct ActiveFiberFrame {
    yielder: *const temen_fiber::Yielder,
    yield_slot: *const FiberYieldSlot,
}

#[cfg(fiber_rt)]
std::thread_local! {
    static ACTIVE_FIBERS: core::cell::RefCell<Vec<ActiveFiberFrame>> =
        const { core::cell::RefCell::new(Vec::new()) };
}

#[inline]
fn active_fiber_depth() -> usize {
    #[cfg(fiber_rt)]
    {
        ACTIVE_FIBERS.with(|s| s.borrow().len())
    }
    #[cfg(not(fiber_rt))]
    {
        0
    }
}

#[inline]
fn truncate_active_fibers(depth: usize) {
    #[cfg(fiber_rt)]
    {
        ACTIVE_FIBERS.with(|s| s.borrow_mut().truncate(depth));
    }
    #[cfg(not(fiber_rt))]
    {
        let _ = depth;
    }
}

/// Whether the calling Carrier thread is currently executing inside a [`KernelFiber`].
#[inline]
pub fn in_kernel_fiber() -> bool {
    active_fiber_depth() > 0
}

/// Suspend the currently running [`KernelFiber`] for guest `suspend(val)`, returning the value
/// delivered by the next `resume`. Returns `None` if not inside a [`KernelFiber`].
pub fn suspend_current_fiber(val: i64) -> Option<i64> {
    #[cfg(fiber_rt)]
    {
        let frame = ACTIVE_FIBERS.with(|s| {
            let b = s.borrow();
            b.last().map(|f| (f.yielder, f.yield_slot))
        })?;
        // SAFETY: `frame.yielder` and `frame.yield_slot` live on the active fiber's stack/box and
        // remain valid until this `suspend` returns.
        unsafe {
            (*frame.1).kind.set(YIELD_KIND_SUSPEND);
            Some((*frame.0).suspend(val as u64) as i64)
        }
    }
    #[cfg(not(fiber_rt))]
    {
        let _ = val;
        None
    }
}

/// Suspend the currently running [`KernelFiber`] for a non-blocking `FIBER_PARKED` wait
/// (`memory.wait` / `thread.join`). Returns `Some(wake_val)` when any Carrier later wakes and
/// resumes the fiber, or `None` if called outside a [`KernelFiber`].
pub fn park_current_fiber() -> Option<i64> {
    #[cfg(fiber_rt)]
    {
        let frame = ACTIVE_FIBERS.with(|s| {
            let b = s.borrow();
            b.last().map(|f| (f.yielder, f.yield_slot))
        })?;
        // SAFETY: `frame.yielder` and `frame.yield_slot` live for the active fiber's duration.
        unsafe {
            (*frame.1).kind.set(YIELD_KIND_PARK);
            Some((*frame.0).suspend(0) as i64)
        }
    }
    #[cfg(not(fiber_rt))]
    {
        None
    }
}

/// Result of one [`KernelFiber::resume`] step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelFiberStep {
    /// Guest `suspend(val)` yielded `val` and stays resumable.
    Suspended(i64),
    /// Guest parked on a `WaitIndex` key (`FIBER_PARKED`, status 3) and stays migratable.
    Parked,
    /// Guest returned `val`; fiber is finished.
    Complete(i64),
    /// Guest raised a software trap; fiber is finished.
    Trapped(Trap),
}

/// Owned migratable native guest fiber over `temen_fiber::Fiber` + stable `Box<KernelVmCtx>`.
///
/// Because `vmctx` has a stable heap address for the fiber's lifetime, any Carrier OS thread can
/// claim this fiber, refresh `vmctx.fuel` and `vmctx.carrier_tls` to its own Carrier cells, and
/// resume the native machine stack cleanly across OS threads.
#[cfg(fiber_rt)]
pub struct KernelFiber {
    fiber: Option<temen_fiber::Fiber>,
    vmctx: *mut KernelVmCtx,
    bridge_holder: *mut KernelBridgeHolder,
    yield_slot: *mut FiberYieldSlot,
    _memory: Arc<KernelMemory>,
    _module: Arc<CompiledKernelModule>,
    started: bool,
}

// SAFETY: `Domain::claim_fiber_resume` enforces exclusive single-Carrier access to `KernelFiber`
// whenever it is running or migrated; `temen_fiber::Fiber` is `Send`, and `vmctx` /
// `bridge_holder` / `yield_slot` are exclusively owned by this `KernelFiber`.
#[cfg(fiber_rt)]
unsafe impl Send for KernelFiber {}

#[cfg(fiber_rt)]
impl Drop for KernelFiber {
    fn drop(&mut self) {
        // Drop the `Fiber` stack FIRST before freeing the allocations its closure references.
        self.fiber.take();
        // SAFETY: all three pointers were created via `Box::into_raw` in `try_new` and are no
        // longer aliased once `self.fiber` has been dropped.
        unsafe {
            drop(Box::from_raw(self.vmctx));
            drop(Box::from_raw(self.yield_slot));
            drop(Box::from_raw(self.bridge_holder));
        }
    }
}

#[cfg(fiber_rt)]
impl KernelFiber {
    /// Create a fresh unstarted [`KernelFiber`] targeting `func_idx(arg0, first_resume_val)`.
    pub fn try_new(
        module: Arc<CompiledKernelModule>,
        func_idx: u32,
        arg0: i64,
        memory: Arc<KernelMemory>,
        domain_poll: &AtomicU64,
        bridge: Arc<dyn KernelBridge>,
    ) -> Result<KernelFiber, Trap> {
        let idx = func_idx as usize;
        if idx >= module.func_arities.len()
            || module.fn_table[idx].type_id != TypeId::fiber_entry().0
        {
            return Err(Trap::IndirectCallType);
        }
        let bridge_holder = Box::into_raw(Box::new(KernelBridgeHolder { bridge }));
        let vmctx = Box::into_raw(Box::new(KernelVmCtx {
            domain_poll: domain_poll as *const AtomicU64,
            fuel: core::ptr::null_mut(),
            runtime_ctx: bridge_holder as *mut c_void,
            mem_base: memory.base(),
            mem_reserved: memory.reserved(),
            mem_mapped: memory.mapped(),
            fn_table: module.fn_table.as_ptr(),
            fn_table_mask: module.fn_table_mask,
            pages: core::ptr::null(),
            got: core::ptr::null(),
            trap_code: AtomicI64::new(0),
            carrier_tls: core::ptr::null_mut(),
        }));
        let yield_slot = Box::into_raw(Box::new(FiberYieldSlot {
            kind: core::cell::Cell::new(YIELD_KIND_NONE),
        }));
        let vmctx_addr = vmctx as usize;
        let slot_addr = yield_slot as usize;
        let mod_clone = Arc::clone(&module);

        let Some(fiber) = temen_fiber::Fiber::new(KERNEL_FIBER_STACK_BYTES, move |y, first_arg| {
            struct ActiveGuard;
            impl Drop for ActiveGuard {
                fn drop(&mut self) {
                    ACTIVE_FIBERS.with(|s| {
                        let _ = s.borrow_mut().pop();
                    });
                }
            }
            let y_ptr = y as *const temen_fiber::Yielder;
            let s_ptr = slot_addr as *const FiberYieldSlot;
            ACTIVE_FIBERS.with(|s| {
                s.borrow_mut().push(ActiveFiberFrame {
                    yielder: y_ptr,
                    yield_slot: s_ptr,
                });
            });
            let _guard = ActiveGuard;
            let vm_ptr = vmctx_addr as *mut KernelVmCtx;
            // SAFETY: `vm_ptr` points into `KernelFiber.vmctx`, which outlives `KernelFiber.fiber`
            // (`Drop` releases `fiber` first) and has fresh `fuel` / `carrier_tls` written by the
            // resuming Carrier before each `resume`.
            match unsafe { mod_clone.call_fiber_entry(func_idx, vm_ptr, arg0, first_arg as i64) } {
                Ok(v) => v as u64,
                Err(t) => {
                    unsafe {
                        (*vm_ptr).raise_trap(t);
                    }
                    0
                }
            }
        }) else {
            // SAFETY: `fiber` was never created; free the three boxes before returning error.
            unsafe {
                drop(Box::from_raw(vmctx));
                drop(Box::from_raw(yield_slot));
                drop(Box::from_raw(bridge_holder));
            }
            return Err(Trap::FiberFault);
        };

        Ok(KernelFiber {
            fiber: Some(fiber),
            vmctx,
            bridge_holder,
            yield_slot,
            _memory: memory,
            _module: module,
            started: false,
        })
    }

    /// Resume this fiber on the calling Carrier, binding `fuel` and `carrier_tls` for the slice.
    pub fn resume(
        &mut self,
        val: i64,
        fuel: &mut FuelCell,
        carrier_tls: &mut i64,
    ) -> KernelFiberStep {
        let Some(ref mut fiber) = self.fiber else {
            return KernelFiberStep::Trapped(Trap::FiberFault);
        };
        // Exact IR fuel parity: `cont.resume` already charged 1 safepoint on entry to the fiber;
        // compensate for the fiber function prologue's `emit_safepoint` on first entry only.
        if !self.started {
            self.started = true;
            if fuel.metered {
                fuel.remaining = fuel.remaining.saturating_add(1);
            }
        }
        // SAFETY: `&mut self` guarantees exclusive access while not running; `vmctx` and
        // `yield_slot` remain live until `Drop`.
        unsafe {
            (*self.vmctx).fuel = fuel as *mut FuelCell;
            (*self.vmctx).carrier_tls = carrier_tls as *mut i64;
            (*self.yield_slot).kind.set(YIELD_KIND_NONE);
        }

        let st = fiber.resume(val as u64);

        // Detach Carrier pointers before returning so no dangling pointer can ever be observed.
        unsafe {
            (*self.vmctx).fuel = core::ptr::null_mut();
            (*self.vmctx).carrier_tls = core::ptr::null_mut();
        }

        match st {
            temen_fiber::State::Yielded(v) => {
                // SAFETY: exclusive access after `resume` returned.
                let kind = unsafe { (*self.yield_slot).kind.get() };
                match kind {
                    YIELD_KIND_PARK => KernelFiberStep::Parked,
                    _ => KernelFiberStep::Suspended(v as i64),
                }
            }
            temen_fiber::State::Complete(v) => {
                self.fiber = None;
                // SAFETY: exclusive access after `resume` returned.
                if let Some(trap) = unsafe { (*self.vmctx).take_trap() } {
                    KernelFiberStep::Trapped(trap)
                } else {
                    KernelFiberStep::Complete(v as i64)
                }
            }
        }
    }
}

#[cfg(not(fiber_rt))]
pub struct KernelFiber;

#[cfg(not(fiber_rt))]
impl KernelFiber {
    pub fn try_new(
        _module: Arc<CompiledKernelModule>,
        _func_idx: u32,
        _arg0: i64,
        _memory: Arc<KernelMemory>,
        _domain_poll: &AtomicU64,
        _bridge: Arc<dyn KernelBridge>,
    ) -> Result<KernelFiber, Trap> {
        Err(Trap::FiberFault)
    }

    pub fn resume(
        &mut self,
        _val: i64,
        _fuel: &mut FuelCell,
        _carrier_tls: &mut i64,
    ) -> KernelFiberStep {
        KernelFiberStep::Trapped(Trap::FiberFault)
    }
}

// ---- `extern "C"` runtime thunks called by compiled Cranelift code ----------------------------

#[inline]
unsafe fn bridge_of(vmctx: &KernelVmCtx) -> &dyn KernelBridge {
    let holder = &*(vmctx.runtime_ctx as *const KernelBridgeHolder);
    holder.bridge.as_ref()
}

unsafe extern "C" fn krt_on_poll(vmctx: *mut KernelVmCtx, flags: u64) {
    let ctx = &*vmctx;
    let fuel = &mut *ctx.fuel;
    if let Err(t) = bridge_of(ctx).on_poll(flags, fuel) {
        ctx.raise_trap(t);
    }
}

unsafe extern "C" fn krt_fuel_refill(vmctx: *mut KernelVmCtx) {
    let ctx = &*vmctx;
    let fuel = &mut *ctx.fuel;
    if let Err(t) = bridge_of(ctx).on_fuel_refill(fuel) {
        ctx.raise_trap(t);
    }
}

unsafe extern "C" fn krt_futex_wait(
    vmctx: *mut KernelVmCtx,
    addr: u64,
    expected: i64,
    width64: i32,
    timeout_ns: i64,
) -> i32 {
    let ctx = &*vmctx;
    let fuel = &mut *ctx.fuel;
    match bridge_of(ctx).futex_wait(addr, expected, width64 != 0, timeout_ns, fuel) {
        Ok(rc) => rc,
        Err(t) => {
            ctx.raise_trap(t);
            0
        }
    }
}

unsafe extern "C" fn krt_futex_notify(vmctx: *mut KernelVmCtx, addr: u64, count: u32) -> i32 {
    let ctx = &*vmctx;
    match bridge_of(ctx).futex_notify(addr, count) {
        Ok(rc) => rc,
        Err(t) => {
            ctx.raise_trap(t);
            0
        }
    }
}

unsafe extern "C" fn krt_thread_spawn(
    vmctx: *mut KernelVmCtx,
    func_idx: u32,
    arg0: i64,
    arg1: i64,
) -> i32 {
    let ctx = &*vmctx;
    match bridge_of(ctx).thread_spawn(func_idx, arg0, arg1) {
        Ok(h) => h,
        Err(t) => {
            ctx.raise_trap(t);
            0
        }
    }
}

unsafe extern "C" fn krt_thread_join(
    vmctx: *mut KernelVmCtx,
    handle: i32,
    out_val: *mut i64,
) -> i64 {
    let ctx = &*vmctx;
    let fuel = &mut *ctx.fuel;
    match bridge_of(ctx).thread_join(handle, fuel) {
        Ok((st, val)) => {
            *out_val = val;
            st
        }
        Err(t) => {
            ctx.raise_trap(t);
            *out_val = 0;
            0
        }
    }
}

unsafe extern "C" fn krt_fiber_new(vmctx: *mut KernelVmCtx, func_idx: u32, arg0: i64) -> i64 {
    let ctx = &*vmctx;
    match bridge_of(ctx).fiber_new(func_idx, arg0) {
        Ok(h) => h,
        Err(t) => {
            ctx.raise_trap(t);
            0
        }
    }
}

unsafe extern "C" fn krt_fiber_resume(
    vmctx: *mut KernelVmCtx,
    handle: i64,
    val: i64,
    block: i32,
    out_val: *mut i64,
) -> i32 {
    let ctx = &*vmctx;
    let fuel = &mut *ctx.fuel;
    let tls = &mut *ctx.carrier_tls;
    match bridge_of(ctx).fiber_resume(handle, val, block != 0, fuel, tls) {
        Ok((st, rv)) => {
            *out_val = rv;
            st
        }
        Err(t) => {
            ctx.raise_trap(t);
            *out_val = 0;
            0
        }
    }
}

unsafe extern "C" fn krt_fiber_suspend(vmctx: *mut KernelVmCtx, val: i64) -> i64 {
    let ctx = &*vmctx;
    match bridge_of(ctx).fiber_suspend(val) {
        Ok(rv) => rv,
        Err(t) => {
            ctx.raise_trap(t);
            0
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe extern "C" fn krt_cap_call(
    vmctx: *mut KernelVmCtx,
    handle: i32,
    type_id: u32,
    op: u32,
    args_ptr: *const i64,
    nargs: u32,
    res_ptr: *mut i64,
    nresults: u32,
) {
    let ctx = &*vmctx;
    let args = if nargs == 0 {
        &[]
    } else {
        core::slice::from_raw_parts(args_ptr, (nargs as usize).min(MAX_CAP_ARGS))
    };
    let res = if nresults == 0 {
        &mut []
    } else {
        core::slice::from_raw_parts_mut(res_ptr, (nresults as usize).min(MAX_CAP_ARGS))
    };
    if let Err(t) = bridge_of(ctx).cap_call(handle, type_id, op, args, res) {
        ctx.raise_trap(t);
    }
}

unsafe extern "C" fn krt_mem_copy(vmctx: *mut KernelVmCtx, dst: u64, src: u64, len: u64) {
    if len == 0 {
        return;
    }
    let ctx = &*vmctx;
    let win =
        temen_mask::Window::with_mapped(ctx.mem_reserved.trailing_zeros() as u8, ctx.mem_mapped);
    if win.span_checked(src, len).is_none()
        || win.span_checked(dst, len).is_none()
        || src.checked_add(len).is_none_or(|e| e > ctx.mem_mapped)
        || dst.checked_add(len).is_none_or(|e| e > ctx.mem_mapped)
    {
        ctx.raise_trap(Trap::MemoryFault);
        return;
    }
    let w = crate::Window {
        base: ctx.mem_base,
        mapped: ctx.mem_mapped,
        reserved: ctx.mem_reserved.max(mem::page_size() as u64),
        pages: 0,
        page_miss: None,
        fn_table: ctx.fn_table as *const u8,
    };
    let mut acc = mem::Access::Copy {
        dst: ctx.mem_base.add(dst as usize),
        src: ctx.mem_base.add(src as usize),
        len: len as usize,
    };
    if !mem::try_access(&w, &mut acc) {
        ctx.raise_trap(Trap::MemoryFault);
    }
}

unsafe extern "C" fn krt_mem_fill(vmctx: *mut KernelVmCtx, dst: u64, val: i32, len: u64) {
    if len == 0 {
        return;
    }
    let ctx = &*vmctx;
    let win =
        temen_mask::Window::with_mapped(ctx.mem_reserved.trailing_zeros() as u8, ctx.mem_mapped);
    if win.span_checked(dst, len).is_none()
        || dst.checked_add(len).is_none_or(|e| e > ctx.mem_mapped)
    {
        ctx.raise_trap(Trap::MemoryFault);
        return;
    }
    let w = crate::Window {
        base: ctx.mem_base,
        mapped: ctx.mem_mapped,
        reserved: ctx.mem_reserved.max(mem::page_size() as u64),
        pages: 0,
        page_miss: None,
        fn_table: ctx.fn_table as *const u8,
    };
    let mut acc = mem::Access::Fill {
        dst: ctx.mem_base.add(dst as usize),
        val: val as u8,
        len: len as usize,
    };
    if !mem::try_access(&w, &mut acc) {
        ctx.raise_trap(Trap::MemoryFault);
    }
}

// ---- Cranelift lowering ------------------------------------------------------------------------

#[inline]
fn clif_ty(t: ValType) -> Result<Type, KernelCompileError> {
    match t {
        ValType::I32 | ValType::Ref | ValType::Cap => Ok(I32),
        ValType::I64 => Ok(I64),
        ValType::F32 => Ok(F32),
        ValType::F64 => Ok(F64),
        ValType::V128 => Err(KernelCompileError::Unsupported("v128")),
    }
}

#[inline]
fn mem_flags() -> MemFlags {
    let mut mf = MemFlags::new();
    mf.set_endianness(Endianness::Little);
    mf
}

#[inline]
fn width_ty(width: u32) -> Type {
    match width {
        1 => I8,
        2 => I16,
        4 => I32,
        _ => I64,
    }
}

fn func_sig(m: &IrModule, ty: u32) -> Result<&FuncType, KernelCompileError> {
    match m.types.get(ty as usize) {
        Some(TypeEntry::Func(ft)) => Ok(ft),
        _ => Err(KernelCompileError::InvalidTypeRef(ty)),
    }
}

struct HostThunks {
    on_poll: FuncId,
    fuel_refill: FuncId,
    futex_wait: FuncId,
    futex_notify: FuncId,
    thread_spawn: FuncId,
    thread_join: FuncId,
    fiber_new: FuncId,
    fiber_resume: FuncId,
    fiber_suspend: FuncId,
    cap_call: FuncId,
    mem_copy: FuncId,
    mem_fill: FuncId,
}

struct LocalThunks {
    on_poll: FuncRef,
    fuel_refill: FuncRef,
    futex_wait: FuncRef,
    futex_notify: FuncRef,
    thread_spawn: FuncRef,
    thread_join: FuncRef,
    fiber_new: FuncRef,
    fiber_resume: FuncRef,
    fiber_suspend: FuncRef,
    cap_call: FuncRef,
    mem_copy: FuncRef,
    mem_fill: FuncRef,
}

/// Compile a verified [`IrModule`] into a [`CompiledKernelModule`] using `bound_mode`.
pub fn compile_kernel_module(
    m: &IrModule,
    bound_mode: BoundMode,
) -> Result<CompiledKernelModule, KernelCompileError> {
    let mut flag_builder = settings::builder();
    flag_builder
        .set("is_pic", "false")
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    flag_builder
        .set("opt_level", "speed")
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    flag_builder
        .set("preserve_frame_pointers", "true")
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    flag_builder
        .set("enable_probestack", "false")
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    let isa_builder =
        cranelift_native::builder().map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    let isa = isa_builder
        .finish(settings::Flags::new(flag_builder))
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;

    let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());
    for (name, ptr) in crate::libcalls() {
        builder.symbol(name, ptr);
    }
    builder.symbol("krt_on_poll", krt_on_poll as *const u8);
    builder.symbol("krt_fuel_refill", krt_fuel_refill as *const u8);
    builder.symbol("krt_futex_wait", krt_futex_wait as *const u8);
    builder.symbol("krt_futex_notify", krt_futex_notify as *const u8);
    builder.symbol("krt_thread_spawn", krt_thread_spawn as *const u8);
    builder.symbol("krt_thread_join", krt_thread_join as *const u8);
    builder.symbol("krt_fiber_new", krt_fiber_new as *const u8);
    builder.symbol("krt_fiber_resume", krt_fiber_resume as *const u8);
    builder.symbol("krt_fiber_suspend", krt_fiber_suspend as *const u8);
    builder.symbol("krt_cap_call", krt_cap_call as *const u8);
    builder.symbol("krt_mem_copy", krt_mem_copy as *const u8);
    builder.symbol("krt_mem_fill", krt_mem_fill as *const u8);

    let mut jit = crate::code_ranges::arena_module(builder)
        .map_err(|e| KernelCompileError::Backend(format!("{e:?}")))?;
    let c_cc = jit.isa().default_call_conv();

    let decl_host = |jit: &mut JITModule,
                     name: &str,
                     params: &[Type],
                     rets: &[Type]|
     -> Result<FuncId, KernelCompileError> {
        let mut sig = Signature::new(c_cc);
        for &p in params {
            sig.params.push(AbiParam::new(p));
        }
        for &r in rets {
            sig.returns.push(AbiParam::new(r));
        }
        jit.declare_function(name, Linkage::Import, &sig)
            .map_err(|e| KernelCompileError::Backend(e.to_string()))
    };

    let host = HostThunks {
        on_poll: decl_host(&mut jit, "krt_on_poll", &[I64, I64], &[])?,
        fuel_refill: decl_host(&mut jit, "krt_fuel_refill", &[I64], &[])?,
        futex_wait: decl_host(
            &mut jit,
            "krt_futex_wait",
            &[I64, I64, I64, I32, I64],
            &[I32],
        )?,
        futex_notify: decl_host(&mut jit, "krt_futex_notify", &[I64, I64, I32], &[I32])?,
        thread_spawn: decl_host(&mut jit, "krt_thread_spawn", &[I64, I32, I64, I64], &[I32])?,
        thread_join: decl_host(&mut jit, "krt_thread_join", &[I64, I32, I64], &[I64])?,
        fiber_new: decl_host(&mut jit, "krt_fiber_new", &[I64, I32, I64], &[I64])?,
        fiber_resume: decl_host(
            &mut jit,
            "krt_fiber_resume",
            &[I64, I64, I64, I32, I64],
            &[I32],
        )?,
        fiber_suspend: decl_host(&mut jit, "krt_fiber_suspend", &[I64, I64], &[I64])?,
        cap_call: decl_host(
            &mut jit,
            "krt_cap_call",
            &[I64, I32, I32, I32, I64, I32, I64, I32],
            &[],
        )?,
        mem_copy: decl_host(&mut jit, "krt_mem_copy", &[I64, I64, I64, I64], &[])?,
        mem_fill: decl_host(&mut jit, "krt_mem_fill", &[I64, I64, I32, I64], &[])?,
    };

    // Declare all guest functions with `CallConv::Tail` and hidden param 0 = `vmctx: I64`.
    let mut func_ids: Vec<FuncId> = Vec::with_capacity(m.funcs.len());
    let mut type_ids: Vec<TypeId> = Vec::with_capacity(m.funcs.len());
    let mut func_arities: Vec<(u16, u8)> = Vec::with_capacity(m.funcs.len());

    for (idx, f) in m.funcs.iter().enumerate() {
        let mut sig = Signature::new(CallConv::Tail);
        sig.params.push(AbiParam::new(I64)); // vmctx
        for &p in &f.params {
            sig.params.push(AbiParam::new(clif_ty(p)?));
        }
        for &r in &f.results {
            sig.returns.push(AbiParam::new(clif_ty(r)?));
        }
        let id = jit
            .declare_function(&format!("kf_{idx}"), Linkage::Local, &sig)
            .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
        func_ids.push(id);
        type_ids.push(TypeId::of_sig(&f.params, &f.results));
        func_arities.push((f.params.len() as u16, f.results.len() as u8));
    }

    let baked_reserved: u64 = m
        .memory
        .as_ref()
        .and_then(|mem| 1u64.checked_shl(u32::from(mem.size_log2)))
        .unwrap_or(0);

    let mut fb_ctx = FunctionBuilderContext::new();
    let mut ctx = jit.make_context();

    for (idx, f) in m.funcs.iter().enumerate() {
        ctx.clear();
        let mut sig = Signature::new(CallConv::Tail);
        sig.params.push(AbiParam::new(I64));
        for &p in &f.params {
            sig.params.push(AbiParam::new(clif_ty(p)?));
        }
        for &r in &f.results {
            sig.returns.push(AbiParam::new(clif_ty(r)?));
        }
        ctx.func.signature = sig;

        let local_thunks = LocalThunks {
            on_poll: jit.declare_func_in_func(host.on_poll, &mut ctx.func),
            fuel_refill: jit.declare_func_in_func(host.fuel_refill, &mut ctx.func),
            futex_wait: jit.declare_func_in_func(host.futex_wait, &mut ctx.func),
            futex_notify: jit.declare_func_in_func(host.futex_notify, &mut ctx.func),
            thread_spawn: jit.declare_func_in_func(host.thread_spawn, &mut ctx.func),
            thread_join: jit.declare_func_in_func(host.thread_join, &mut ctx.func),
            fiber_new: jit.declare_func_in_func(host.fiber_new, &mut ctx.func),
            fiber_resume: jit.declare_func_in_func(host.fiber_resume, &mut ctx.func),
            fiber_suspend: jit.declare_func_in_func(host.fiber_suspend, &mut ctx.func),
            cap_call: jit.declare_func_in_func(host.cap_call, &mut ctx.func),
            mem_copy: jit.declare_func_in_func(host.mem_copy, &mut ctx.func),
            mem_fill: jit.declare_func_in_func(host.mem_fill, &mut ctx.func),
        };
        let local_funcs: Vec<FuncRef> = func_ids
            .iter()
            .map(|&id| jit.declare_func_in_func(id, &mut ctx.func))
            .collect();

        lower_func(
            m,
            f,
            &local_funcs,
            &local_thunks,
            bound_mode,
            baked_reserved,
            &mut ctx.func,
            &mut fb_ctx,
        )?;

        jit.define_function(func_ids[idx], &mut ctx)
            .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
    }

    // Build C-ABI entry trampolines matching `mem::Entry` for each function so `run_guarded` can
    // enter any function cleanly.
    let mut tramp_ids: Vec<FuncId> = Vec::with_capacity(m.funcs.len());
    for (idx, f) in m.funcs.iter().enumerate() {
        let mut sig = Signature::new(c_cc);
        for _ in 0..5 {
            sig.params.push(AbiParam::new(I64));
        }
        let tid = jit
            .declare_function(&format!("ktramp_{idx}"), Linkage::Local, &sig)
            .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
        ctx.clear();
        ctx.func.signature = sig;
        let callee_ref = jit.declare_func_in_func(func_ids[idx], &mut ctx.func);
        {
            let mut b = FunctionBuilder::new(&mut ctx.func, &mut fb_ctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);
            let args_ptr = b.block_params(entry)[0];
            let res_ptr = b.block_params(entry)[1];
            let vmctx = b.block_params(entry)[4];

            let mut call_args = Vec::with_capacity(1 + f.params.len());
            call_args.push(vmctx);
            for (i, &pt) in f.params.iter().enumerate() {
                let raw = b
                    .ins()
                    .load(I64, MemFlags::trusted(), args_ptr, (i * 8) as i32);
                let v = match pt {
                    ValType::I32 | ValType::Ref | ValType::Cap => b.ins().ireduce(I32, raw),
                    ValType::I64 => raw,
                    ValType::F32 => {
                        let lo = b.ins().ireduce(I32, raw);
                        b.ins().bitcast(F32, MemFlags::new(), lo)
                    }
                    ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), raw),
                    ValType::V128 => unreachable!(),
                };
                call_args.push(v);
            }
            let call = b.ins().call(callee_ref, &call_args);
            let results: Vec<Value> = b.inst_results(call).to_vec();
            for (i, (&rt, &val)) in f.results.iter().zip(results.iter()).enumerate() {
                let wide = match rt {
                    ValType::I32 | ValType::Ref | ValType::Cap => b.ins().uextend(I64, val),
                    ValType::I64 => val,
                    ValType::F32 => {
                        let bits = b.ins().bitcast(I32, MemFlags::new(), val);
                        b.ins().uextend(I64, bits)
                    }
                    ValType::F64 => b.ins().bitcast(I64, MemFlags::new(), val),
                    ValType::V128 => unreachable!(),
                };
                b.ins()
                    .store(MemFlags::trusted(), wide, res_ptr, (i * 8) as i32);
            }
            b.ins().return_(&[]);
            b.finalize();
        }
        jit.define_function(tid, &mut ctx)
            .map_err(|e| KernelCompileError::Backend(e.to_string()))?;
        tramp_ids.push(tid);
    }

    jit.finalize_definitions()
        .map_err(|e| KernelCompileError::Backend(e.to_string()))?;

    let table_len = m.funcs.len().max(1).next_power_of_two();
    let fn_table_mask = (table_len - 1) as u64;
    let mut fn_table = Vec::with_capacity(table_len);
    for idx in 0..table_len {
        if idx < m.funcs.len() {
            fn_table.push(FnTableEntry {
                code_ptr: jit.get_finalized_function(func_ids[idx]),
                type_id: type_ids[idx].0,
            });
        } else {
            fn_table.push(FnTableEntry {
                code_ptr: core::ptr::null(),
                type_id: 0,
            });
        }
    }
    let trampolines: Vec<*const u8> = tramp_ids
        .iter()
        .map(|&id| jit.get_finalized_function(id))
        .collect();
    let initial_data = m
        .data
        .iter()
        .map(|d| KernelInitialData {
            offset: d.offset,
            readonly: d.readonly,
            bytes: d.bytes.clone(),
        })
        .collect();
    let exports = m.exports.iter().map(|e| (e.name.clone(), e.func)).collect();

    Ok(CompiledKernelModule {
        _jit: jit,
        fn_table: fn_table.into_boxed_slice(),
        fn_table_mask,
        trampolines,
        func_arities,
        memory_size_log2: m.memory.as_ref().map(|mem| mem.size_log2),
        initial_data,
        exports,
        bound_mode,
    })
}

struct LowerCtx<'a> {
    m: &'a IrModule,
    local_funcs: &'a [FuncRef],
    thunks: &'a LocalThunks,
    bound_mode: BoundMode,
    baked_reserved: u64,
    vmctx: Variable,
    mem_base: Variable,
    mem_guard: Variable,
    /// Precomputed loop-invariant `usub_sat(mem_reserved, need - 1)` values in `entry_setup` for
    /// `BoundMode::DynamicVmCtx`, keyed by `need = width + offset`.
    dyn_limits: std::collections::HashMap<u64, Variable>,
    poll_ptr: Variable,
    trap_blk: ClifBlock,
    unwind_blk: ClifBlock,
}

/// Collect every distinct `need = width + offset` used by memory ops in `f` so `DynamicVmCtx` can
/// hoist `excl_limit = usub_sat(mem_reserved, need - 1)` once in `entry_setup` outside all loops.
fn collect_mem_needs(f: &Func) -> Vec<u64> {
    let mut needs = Vec::new();
    let mut add = |offset: u64, width: u32| {
        if let Some(n) = (width as u64).checked_add(offset) {
            if !needs.contains(&n) {
                needs.push(n);
            }
        }
    };
    for blk in &f.blocks {
        for inst in &blk.insts {
            match inst {
                Inst::Load { op, offset, .. } => add(*offset, op.info().2),
                Inst::Store { op, offset, .. } => add(*offset, op.info().2),
                Inst::AtomicLoad { ty, offset, .. }
                | Inst::AtomicStore { ty, offset, .. }
                | Inst::AtomicRmw { ty, offset, .. }
                | Inst::AtomicCmpxchg { ty, offset, .. } => {
                    add(*offset, if *ty == IntTy::I32 { 4 } else { 8 });
                }
                _ => {}
            }
        }
    }
    needs
}

#[allow(clippy::too_many_arguments)]
fn lower_func(
    m: &IrModule,
    f: &Func,
    local_funcs: &[FuncRef],
    thunks: &LocalThunks,
    bound_mode: BoundMode,
    baked_reserved: u64,
    clif_func: &mut cranelift_codegen::ir::Function,
    fb_ctx: &mut FunctionBuilderContext,
) -> Result<(), KernelCompileError> {
    let ret_tys: Vec<Type> = f
        .results
        .iter()
        .map(|&t| clif_ty(t))
        .collect::<Result<_, _>>()?;
    let val_types = temen_verify::func_value_types(f, &m.funcs, &m.types, m.memory.is_some());
    let mem_needs = if bound_mode == BoundMode::DynamicVmCtx {
        collect_mem_needs(f)
    } else {
        Vec::new()
    };

    let mut b = FunctionBuilder::new(clif_func, fb_ctx);
    let vmctx_var = b.declare_var(I64);
    let mem_base_var = b.declare_var(I64);
    let mem_guard_var = b.declare_var(I64);
    let poll_ptr_var = b.declare_var(I64);

    // Pre-create one CLIF block per IR block, plus dedicated trap/unwind blocks.
    let entry_setup = b.create_block();
    b.append_block_params_for_function_params(entry_setup);

    let mut clif_blocks: Vec<ClifBlock> = Vec::with_capacity(f.blocks.len());
    for blk in &f.blocks {
        let cb = b.create_block();
        for &pt in &blk.params {
            b.append_block_param(cb, clif_ty(pt)?);
        }
        clif_blocks.push(cb);
    }
    let trap_blk = b.create_block();
    b.append_block_param(trap_blk, I64);
    let unwind_blk = b.create_block();

    // Entry setup block: load vmctx, mem_base, mem_reserved, mem_guard, poll_ptr, and hoist each
    // distinct `DynamicVmCtx` exclusive limit `usub_sat(mem_reserved, need - 1)` once.
    b.switch_to_block(entry_setup);
    b.seal_block(entry_setup);
    let entry_params: Vec<Value> = b.block_params(entry_setup).to_vec();
    let vmctx_val = entry_params[0];
    b.def_var(vmctx_var, vmctx_val);
    let poll_ptr = b
        .ins()
        .load(I64, MemFlags::trusted(), vmctx_val, OFF_DOMAIN_POLL);
    b.def_var(poll_ptr_var, poll_ptr);
    let mem_base = b
        .ins()
        .load(I64, MemFlags::trusted(), vmctx_val, OFF_MEM_BASE);
    b.def_var(mem_base_var, mem_base);

    let mut dyn_limits = std::collections::HashMap::with_capacity(mem_needs.len());
    if !mem_needs.is_empty() {
        let mem_reserved = b
            .ins()
            .load(I64, MemFlags::trusted(), vmctx_val, OFF_MEM_RESERVED);
        // Both `mem_reserved` and `page_size` are powers of two, so the trailing `PROT_NONE` guard
        // offset `round_up(mem_reserved, page_size)` equals `umax(mem_reserved, page_size)`.
        let page_v = b.ins().iconst(I64, mem::page_size() as i64);
        let mem_guard = b.ins().umax(mem_reserved, page_v);
        b.def_var(mem_guard_var, mem_guard);

        for &need in &mem_needs {
            let var = b.declare_var(I64);
            let sub = need - 1;
            let excl = if sub == 0 {
                mem_reserved
            } else {
                let sub_v = b.ins().iconst(I64, sub as i64);
                let zero = b.ins().iconst(I64, 0);
                let ok = b
                    .ins()
                    .icmp(IntCC::UnsignedGreaterThanOrEqual, mem_reserved, sub_v);
                let diff = b.ins().isub(mem_reserved, sub_v);
                b.ins().select(ok, diff, zero)
            };
            b.def_var(var, excl);
            dyn_limits.insert(need, var);
        }
    } else {
        let page_v = b.ins().iconst(I64, mem::page_size() as i64);
        b.def_var(mem_guard_var, page_v);
    }

    let lctx = LowerCtx {
        m,
        local_funcs,
        thunks,
        bound_mode,
        baked_reserved,
        vmctx: vmctx_var,
        mem_base: mem_base_var,
        mem_guard: mem_guard_var,
        dyn_limits,
        poll_ptr: poll_ptr_var,
        trap_blk,
        unwind_blk,
    };

    // Charge function-entry IR safepoint (Invariant 9).
    emit_safepoint(&mut b, &lctx);

    let init_args: Vec<BlockArg> = entry_params[1..]
        .iter()
        .copied()
        .map(BlockArg::Value)
        .collect();
    b.ins().jump(clif_blocks[0], &init_args);

    // Track per-block SSA values as Variables so captures from any dominating ancestor block work
    // seamlessly across CFG diamonds and loops.
    let mut block_vars: Vec<Vec<Variable>> = Vec::with_capacity(f.blocks.len());
    for (bi, blk) in f.blocks.iter().enumerate() {
        let mut vars = Vec::with_capacity(val_types[bi].len());
        for &vt in &val_types[bi] {
            vars.push(b.declare_var(clif_ty(vt)?));
        }
        let _ = blk;
        block_vars.push(vars);
    }

    for (bi, blk) in f.blocks.iter().enumerate() {
        b.switch_to_block(clif_blocks[bi]);
        let params: Vec<Value> = b.block_params(clif_blocks[bi]).to_vec();
        let mut vals: Vec<Value> = Vec::with_capacity(val_types[bi].len());
        for (i, &pv) in params.iter().enumerate() {
            b.def_var(block_vars[bi][i], pv);
            vals.push(pv);
        }
        for (j, &(anc_blk, anc_val)) in blk.captures.iter().enumerate() {
            let cv = b.use_var(block_vars[anc_blk as usize][anc_val as usize]);
            let slot = params.len() + j;
            b.def_var(block_vars[bi][slot], cv);
            vals.push(cv);
        }

        let mut next_slot = vals.len();
        for inst in &blk.insts {
            let produced = lower_inst(&mut b, &lctx, inst, &vals)?;
            for v in produced {
                b.def_var(block_vars[bi][next_slot], v);
                vals.push(v);
                next_slot += 1;
            }
        }

        lower_term(&mut b, &lctx, bi as u32, &blk.term, &vals, &clif_blocks)?;
    }

    // Trap block: record software trap code in vmctx.trap_code and return zeroed results.
    b.switch_to_block(trap_blk);
    b.set_cold_block(trap_blk);
    let code = b.block_params(trap_blk)[0];
    let vm = b.use_var(vmctx_var);
    b.ins().store(MemFlags::trusted(), code, vm, OFF_TRAP_CODE);
    b.ins().jump(unwind_blk, &[]);

    b.switch_to_block(unwind_blk);
    b.set_cold_block(unwind_blk);
    let zeros: Vec<Value> = ret_tys.iter().map(|&t| zero_val(&mut b, t)).collect();
    b.ins().return_(&zeros);

    b.seal_all_blocks();
    b.finalize();
    Ok(())
}

fn zero_val(b: &mut FunctionBuilder, ty: Type) -> Value {
    match ty {
        I32 => b.ins().iconst(I32, 0),
        I64 => b.ins().iconst(I64, 0),
        F32 => b.ins().f32const(0.0),
        F64 => b.ins().f64const(0.0),
        _ => b.ins().iconst(ty, 0),
    }
}

fn emit_check_trap(b: &mut FunctionBuilder, lctx: &LowerCtx) {
    let vm = b.use_var(lctx.vmctx);
    let code = b.ins().load(I64, MemFlags::trusted(), vm, OFF_TRAP_CODE);
    let cont = b.create_block();
    b.ins().brif(code, lctx.unwind_blk, &[], cont, &[]);
    b.switch_to_block(cont);
}

fn emit_safepoint(b: &mut FunctionBuilder, lctx: &LowerCtx) {
    // 1. Poll shared `Domain.poll` word (read-only L1-shared on hot path).
    let poll_ptr = b.use_var(lctx.poll_ptr);
    let poll = b.ins().load(I64, MemFlags::trusted(), poll_ptr, 0);
    let poll_slow = b.create_block();
    let after_poll = b.create_block();
    b.ins().brif(poll, poll_slow, &[], after_poll, &[]);

    b.switch_to_block(poll_slow);
    b.set_cold_block(poll_slow);
    let vm = b.use_var(lctx.vmctx);
    b.ins().call(lctx.thunks.on_poll, &[vm, poll]);
    emit_check_trap(b, lctx);
    b.ins().jump(after_poll, &[]);

    // 2. Carrier-owned `FuelCell` check (Decision 30).
    b.switch_to_block(after_poll);
    let vm = b.use_var(lctx.vmctx);
    let fuel_ptr = b.ins().load(I64, MemFlags::trusted(), vm, OFF_FUEL);
    let rem = b.ins().load(I64, MemFlags::trusted(), fuel_ptr, 0);
    let fuel_ok = b.create_block();
    let fuel_slow = b.create_block();
    let done = b.create_block();
    b.ins().brif(rem, fuel_ok, &[], fuel_slow, &[]);

    b.switch_to_block(fuel_ok);
    let rem1 = b.ins().iadd_imm(rem, -1);
    b.ins().store(MemFlags::trusted(), rem1, fuel_ptr, 0);
    b.ins().jump(done, &[]);

    b.switch_to_block(fuel_slow);
    b.set_cold_block(fuel_slow);
    let vm2 = b.use_var(lctx.vmctx);
    b.ins().call(lctx.thunks.fuel_refill, &[vm2]);
    emit_check_trap(b, lctx);
    b.ins().jump(done, &[]);

    b.switch_to_block(done);
}

/// Branchless spectre-guarded confinement lowering (supports both `BakedImmediate` and
/// `DynamicVmCtx` for the Stage 1d VA reservation benchmark).
fn emit_confine(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    addr: Value,
    offset: u64,
    width: u32,
) -> Value {
    let base = b.use_var(lctx.mem_base);
    let Some(need) = (width as u64).checked_add(offset) else {
        let guard = b.use_var(lctx.mem_guard);
        return b.ins().iadd(base, guard);
    };
    let eff = if offset == 0 {
        addr
    } else {
        let off = b.ins().iconst(I64, offset as i64);
        b.ins().iadd(addr, off)
    };

    match lctx.bound_mode {
        BoundMode::BakedImmediate => {
            if lctx.baked_reserved == 0 {
                return base;
            }
            let guard_off = lctx.baked_reserved.max(mem::page_size() as u64);
            let guard = b.ins().iconst(I64, guard_off as i64);
            let oob = match lctx.baked_reserved.checked_sub(need) {
                Some(limit) => {
                    let lim = b.ins().iconst(I64, limit as i64);
                    b.ins().icmp(IntCC::UnsignedGreaterThan, addr, lim)
                }
                None => b.ins().iconst(I8, 1),
            };
            let confined = b.ins().select_spectre_guard(oob, guard, eff);
            b.ins().iadd(base, confined)
        }
        BoundMode::DynamicVmCtx => {
            let guard = b.use_var(lctx.mem_guard);
            let excl_lim = b.use_var(lctx.dyn_limits[&need]);
            let oob = b
                .ins()
                .icmp(IntCC::UnsignedGreaterThanOrEqual, addr, excl_lim);
            let confined = b.ins().select_spectre_guard(oob, guard, eff);
            b.ins().iadd(base, confined)
        }
    }
}

fn emit_align_check(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    addr: Value,
    offset: u64,
    width: u32,
) {
    let eff = if offset == 0 {
        addr
    } else {
        b.ins().iadd_imm(addr, offset as i64)
    };
    let rem = b.ins().band_imm(eff, i64::from(width - 1));
    let ok = b.create_block();
    let bad = b.create_block();
    b.ins().brif(rem, bad, &[], ok, &[]);
    b.switch_to_block(bad);
    b.set_cold_block(bad);
    let tc = b.ins().iconst(I64, Trap::MemoryFault.code());
    b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);
    b.switch_to_block(ok);
}

fn lower_int_bin(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    ty: IntTy,
    op: BinOp,
    a: Value,
    rhs: Value,
) -> Value {
    let clif = match ty {
        IntTy::I32 => I32,
        IntTy::I64 => I64,
    };
    match op {
        BinOp::Add => b.ins().iadd(a, rhs),
        BinOp::Sub => b.ins().isub(a, rhs),
        BinOp::Mul => b.ins().imul(a, rhs),
        BinOp::And => b.ins().band(a, rhs),
        BinOp::Or => b.ins().bor(a, rhs),
        BinOp::Xor => b.ins().bxor(a, rhs),
        BinOp::Shl => b.ins().ishl(a, rhs),
        BinOp::ShrS => b.ins().sshr(a, rhs),
        BinOp::ShrU => b.ins().ushr(a, rhs),
        BinOp::Rotl => b.ins().rotl(a, rhs),
        BinOp::Rotr => b.ins().rotr(a, rhs),
        BinOp::DivS | BinOp::DivU | BinOp::RemS | BinOp::RemU => {
            // Check division by zero.
            let is_zero = b.ins().icmp_imm(IntCC::Equal, rhs, 0);
            let dz_blk = b.create_block();
            let nz_blk = b.create_block();
            b.ins().brif(is_zero, dz_blk, &[], nz_blk, &[]);
            b.switch_to_block(dz_blk);
            b.set_cold_block(dz_blk);
            let tc = b.ins().iconst(I64, Trap::DivByZero.code());
            b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);

            b.switch_to_block(nz_blk);
            if matches!(op, BinOp::DivS | BinOp::RemS) {
                let min_v = match ty {
                    IntTy::I32 => b.ins().iconst(I32, i64::from(i32::MIN)),
                    IntTy::I64 => b.ins().iconst(I64, i64::MIN),
                };
                let is_min = b.ins().icmp(IntCC::Equal, a, min_v);
                let is_m1 = b.ins().icmp_imm(IntCC::Equal, rhs, -1);
                let ov = b.ins().band(is_min, is_m1);
                let ov_blk = b.create_block();
                let ok_blk = b.create_block();
                let done_blk = b.create_block();
                b.append_block_param(done_blk, clif);
                b.ins().brif(ov, ov_blk, &[], ok_blk, &[]);

                b.switch_to_block(ov_blk);
                b.set_cold_block(ov_blk);
                if op == BinOp::DivS {
                    let tc = b.ins().iconst(I64, Trap::IntOverflow.code());
                    b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);
                } else {
                    let z = b.ins().iconst(clif, 0);
                    b.ins().jump(done_blk, &[BlockArg::Value(z)]);
                }

                b.switch_to_block(ok_blk);
                let res = if op == BinOp::DivS {
                    b.ins().sdiv(a, rhs)
                } else {
                    b.ins().srem(a, rhs)
                };
                b.ins().jump(done_blk, &[BlockArg::Value(res)]);

                b.switch_to_block(done_blk);
                b.block_params(done_blk)[0]
            } else if op == BinOp::DivU {
                b.ins().udiv(a, rhs)
            } else {
                b.ins().urem(a, rhs)
            }
        }
    }
}

fn int_cc(op: CmpOp) -> IntCC {
    match op {
        CmpOp::Eq => IntCC::Equal,
        CmpOp::Ne => IntCC::NotEqual,
        CmpOp::LtS => IntCC::SignedLessThan,
        CmpOp::LtU => IntCC::UnsignedLessThan,
        CmpOp::LeS => IntCC::SignedLessThanOrEqual,
        CmpOp::LeU => IntCC::UnsignedLessThanOrEqual,
        CmpOp::GtS => IntCC::SignedGreaterThan,
        CmpOp::GtU => IntCC::UnsignedGreaterThan,
        CmpOp::GeS => IntCC::SignedGreaterThanOrEqual,
        CmpOp::GeU => IntCC::UnsignedGreaterThanOrEqual,
    }
}

fn float_cc(op: FCmpOp) -> FloatCC {
    match op {
        FCmpOp::Eq => FloatCC::Equal,
        FCmpOp::Ne => FloatCC::NotEqual,
        FCmpOp::Lt => FloatCC::LessThan,
        FCmpOp::Le => FloatCC::LessThanOrEqual,
        FCmpOp::Gt => FloatCC::GreaterThan,
        FCmpOp::Ge => FloatCC::GreaterThanOrEqual,
    }
}

fn clif_rmw_op(op: AtomicRmwOp) -> ClifRmwOp {
    match op {
        AtomicRmwOp::Add => ClifRmwOp::Add,
        AtomicRmwOp::Sub => ClifRmwOp::Sub,
        AtomicRmwOp::And => ClifRmwOp::And,
        AtomicRmwOp::Or => ClifRmwOp::Or,
        AtomicRmwOp::Xor => ClifRmwOp::Xor,
        AtomicRmwOp::Xchg => ClifRmwOp::Xchg,
    }
}

fn lower_inst(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    inst: &Inst,
    vals: &[Value],
) -> Result<Vec<Value>, KernelCompileError> {
    let v = |idx: ValIdx| vals[idx as usize];
    let out: Vec<Value> = match inst {
        Inst::ConstI32(c) => vec![b.ins().iconst(I32, i64::from(*c))],
        Inst::ConstI64(c) => vec![b.ins().iconst(I64, *c)],
        Inst::ConstF32(bits) => vec![b.ins().f32const(f32::from_bits(*bits))],
        Inst::ConstF64(bits) => vec![b.ins().f64const(f64::from_bits(*bits))],
        Inst::RefFunc { func } => vec![b.ins().iconst(I32, i64::from(*func))],

        Inst::IntBin { ty, op, a, b: rhs } => {
            vec![lower_int_bin(b, lctx, *ty, *op, v(*a), v(*rhs))]
        }
        Inst::IntBinImm { ty, op, a, imm } => {
            let clif = if *ty == IntTy::I32 { I32 } else { I64 };
            let rhs = b.ins().iconst(clif, *imm);
            vec![lower_int_bin(b, lctx, *ty, *op, v(*a), rhs)]
        }
        Inst::IntCmp {
            ty: _,
            op,
            a,
            b: rhs,
        } => {
            let c = b.ins().icmp(int_cc(*op), v(*a), v(*rhs));
            vec![b.ins().uextend(I32, c)]
        }
        Inst::IntCmpImm { ty, op, a, imm } => {
            let clif = if *ty == IntTy::I32 { I32 } else { I64 };
            let rhs = b.ins().iconst(clif, *imm);
            let c = b.ins().icmp(int_cc(*op), v(*a), rhs);
            vec![b.ins().uextend(I32, c)]
        }
        Inst::IntUn { ty, op, a } => {
            let x = v(*a);
            let r = match op {
                IntUnOp::Clz => b.ins().clz(x),
                IntUnOp::Ctz => b.ins().ctz(x),
                IntUnOp::Popcnt => b.ins().popcnt(x),
                IntUnOp::Extend8S => b.ins().ireduce(I8, x),
                IntUnOp::Extend16S => b.ins().ireduce(I16, x),
                IntUnOp::Extend32S => {
                    if *ty == IntTy::I64 {
                        b.ins().ireduce(I32, x)
                    } else {
                        x
                    }
                }
            };
            let clif = if *ty == IntTy::I32 { I32 } else { I64 };
            let r = match op {
                IntUnOp::Extend8S | IntUnOp::Extend16S => b.ins().sextend(clif, r),
                IntUnOp::Extend32S if *ty == IntTy::I64 => b.ins().sextend(I64, r),
                _ => r,
            };
            vec![r]
        }
        Inst::Eqz { ty: _, a } => {
            let c = b.ins().icmp_imm(IntCC::Equal, v(*a), 0);
            vec![b.ins().uextend(I32, c)]
        }
        Inst::Convert { op, a } => {
            let r = match op {
                ConvOp::WrapI64 => b.ins().ireduce(I32, v(*a)),
                ConvOp::ExtendI32S => b.ins().sextend(I64, v(*a)),
                ConvOp::ExtendI32U => b.ins().uextend(I64, v(*a)),
            };
            vec![r]
        }
        Inst::Select { cond, a, b: bv } => {
            vec![b.ins().select(v(*cond), v(*a), v(*bv))]
        }

        Inst::FBin {
            ty: _,
            op,
            a,
            b: bv,
        } => {
            let r = match op {
                FBinOp::Add => b.ins().fadd(v(*a), v(*bv)),
                FBinOp::Sub => b.ins().fsub(v(*a), v(*bv)),
                FBinOp::Mul => b.ins().fmul(v(*a), v(*bv)),
                FBinOp::Div => b.ins().fdiv(v(*a), v(*bv)),
                FBinOp::Min => b.ins().fmin(v(*a), v(*bv)),
                FBinOp::Max => b.ins().fmax(v(*a), v(*bv)),
                FBinOp::Copysign => b.ins().fcopysign(v(*a), v(*bv)),
            };
            vec![r]
        }
        Inst::FUn { ty: _, op, a } => {
            let r = match op {
                FUnOp::Abs => b.ins().fabs(v(*a)),
                FUnOp::Neg => b.ins().fneg(v(*a)),
                FUnOp::Sqrt => b.ins().sqrt(v(*a)),
                FUnOp::Ceil => b.ins().ceil(v(*a)),
                FUnOp::Floor => b.ins().floor(v(*a)),
                FUnOp::Trunc => b.ins().trunc(v(*a)),
                FUnOp::Nearest => b.ins().nearest(v(*a)),
            };
            vec![r]
        }
        Inst::Fma { ty: _, a, b: bv, c } => {
            vec![b.ins().fma(v(*a), v(*bv), v(*c))]
        }
        Inst::FCmp {
            ty: _,
            op,
            a,
            b: bv,
        } => {
            let c = b.ins().fcmp(float_cc(*op), v(*a), v(*bv));
            vec![b.ins().uextend(I32, c)]
        }
        Inst::FToISat { op, a } => {
            let (_, to, signed) = op.parts();
            let dst_ty = if to == IntTy::I32 { I32 } else { I64 };
            let r = if signed {
                b.ins().fcvt_to_sint_sat(dst_ty, v(*a))
            } else {
                b.ins().fcvt_to_uint_sat(dst_ty, v(*a))
            };
            vec![r]
        }
        Inst::FToITrap { op, a } => {
            let (from, to, signed) = op.parts();
            let f64_v = if from == FloatTy::F32 {
                b.ins().fpromote(F64, v(*a))
            } else {
                v(*a)
            };
            let is_nan = b.ins().fcmp(FloatCC::NotEqual, f64_v, f64_v);
            let (lo, hi, lo_inclusive) = match (to, signed) {
                (IntTy::I32, true) => (-2_147_483_649.0_f64, 2_147_483_648.0_f64, false),
                (IntTy::I32, false) => (-1.0_f64, 4_294_967_296.0_f64, false),
                (IntTy::I64, true) => (
                    -9_223_372_036_854_775_808.0_f64,
                    9_223_372_036_854_775_808.0_f64,
                    true,
                ),
                (IntTy::I64, false) => (-1.0_f64, 18_446_744_073_709_551_616.0_f64, false),
            };
            let lo_v = b.ins().f64const(lo);
            let hi_v = b.ins().f64const(hi);
            let low_bad = b.ins().fcmp(
                if lo_inclusive {
                    FloatCC::LessThan
                } else {
                    FloatCC::LessThanOrEqual
                },
                f64_v,
                lo_v,
            );
            let high_bad = b.ins().fcmp(FloatCC::GreaterThanOrEqual, f64_v, hi_v);
            let bad1 = b.ins().bor(is_nan, low_bad);
            let bad = b.ins().bor(bad1, high_bad);
            let bad_blk = b.create_block();
            let ok_blk = b.create_block();
            b.ins().brif(bad, bad_blk, &[], ok_blk, &[]);
            b.switch_to_block(bad_blk);
            b.set_cold_block(bad_blk);
            let tc = b.ins().iconst(I64, Trap::BadConversion.code());
            b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);

            b.switch_to_block(ok_blk);
            let dst_ty = if to == IntTy::I32 { I32 } else { I64 };
            let r = if signed {
                b.ins().fcvt_to_sint_sat(dst_ty, f64_v)
            } else {
                b.ins().fcvt_to_uint_sat(dst_ty, f64_v)
            };
            vec![r]
        }
        Inst::IToFConv { op, a } => {
            let (_, to, signed) = op.parts();
            let dst_ty = if to == FloatTy::F32 { F32 } else { F64 };
            let r = if signed {
                b.ins().fcvt_from_sint(dst_ty, v(*a))
            } else {
                b.ins().fcvt_from_uint(dst_ty, v(*a))
            };
            vec![r]
        }
        Inst::Cast { op, a } => {
            let r = match op {
                CastOp::Demote => b.ins().fdemote(F32, v(*a)),
                CastOp::Promote => b.ins().fpromote(F64, v(*a)),
                CastOp::ReinterpI32F32 => b.ins().bitcast(F32, MemFlags::new(), v(*a)),
                CastOp::ReinterpF32I32 => b.ins().bitcast(I32, MemFlags::new(), v(*a)),
                CastOp::ReinterpI64F64 => b.ins().bitcast(F64, MemFlags::new(), v(*a)),
                CastOp::ReinterpF64I64 => b.ins().bitcast(I64, MemFlags::new(), v(*a)),
            };
            vec![r]
        }

        // ---- Linear memory ----
        Inst::Load { op, addr, offset } => {
            let (_, rty, width, signed) = op.info();
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            let res = if matches!(rty, ValType::F32 | ValType::F64) {
                b.ins().load(clif_ty(rty)?, mem_flags(), phys, 0)
            } else {
                let load_ty = width_ty(width);
                let raw = b.ins().load(load_ty, mem_flags(), phys, 0);
                let dst_ty = clif_ty(rty)?;
                if load_ty == dst_ty {
                    raw
                } else if signed {
                    b.ins().sextend(dst_ty, raw)
                } else {
                    b.ins().uextend(dst_ty, raw)
                }
            };
            vec![res]
        }
        Inst::Store {
            op,
            addr,
            value,
            offset,
        } => {
            let (_, vty, width) = op.info();
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            if matches!(vty, ValType::F32 | ValType::F64) {
                b.ins().store(mem_flags(), v(*value), phys, 0);
            } else {
                let store_ty = width_ty(width);
                let val = if store_ty == clif_ty(vty)? {
                    v(*value)
                } else {
                    b.ins().ireduce(store_ty, v(*value))
                };
                b.ins().store(mem_flags(), val, phys, 0);
            }
            vec![]
        }
        Inst::MemCopy { dst, src, len } | Inst::MemMove { dst, src, len } => {
            let vm = b.use_var(lctx.vmctx);
            b.ins()
                .call(lctx.thunks.mem_copy, &[vm, v(*dst), v(*src), v(*len)]);
            emit_check_trap(b, lctx);
            vec![]
        }
        Inst::MemFill { dst, val, len } => {
            let vm = b.use_var(lctx.vmctx);
            b.ins()
                .call(lctx.thunks.mem_fill, &[vm, v(*dst), v(*val), v(*len)]);
            emit_check_trap(b, lctx);
            vec![]
        }

        // ---- Atomics, futex, TLS ----
        Inst::AtomicLoad { ty, addr, offset } => {
            let width = if *ty == IntTy::I32 { 4 } else { 8 };
            emit_align_check(b, lctx, v(*addr), *offset, width);
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            let clif = if *ty == IntTy::I32 { I32 } else { I64 };
            vec![b.ins().atomic_load(clif, mem_flags(), phys)]
        }
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
        } => {
            let width = if *ty == IntTy::I32 { 4 } else { 8 };
            emit_align_check(b, lctx, v(*addr), *offset, width);
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            b.ins().atomic_store(mem_flags(), v(*value), phys);
            vec![]
        }
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
        } => {
            let width = if *ty == IntTy::I32 { 4 } else { 8 };
            emit_align_check(b, lctx, v(*addr), *offset, width);
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            let clif = if *ty == IntTy::I32 { I32 } else { I64 };
            vec![b
                .ins()
                .atomic_rmw(clif, mem_flags(), clif_rmw_op(*op), phys, v(*value))]
        }
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
        } => {
            let width = if *ty == IntTy::I32 { 4 } else { 8 };
            emit_align_check(b, lctx, v(*addr), *offset, width);
            let phys = emit_confine(b, lctx, v(*addr), *offset, width);
            vec![b
                .ins()
                .atomic_cas(mem_flags(), phys, v(*expected), v(*replacement))]
        }
        Inst::AtomicFence { order } => {
            if !matches!(order, IrOrdering::Relaxed) {
                b.ins().fence();
            }
            vec![]
        }
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => {
            let width = if *ty == IntTy::I32 { 4 } else { 8 };
            emit_align_check(b, lctx, v(*addr), 0, width);
            let exp64 = if *ty == IntTy::I32 {
                b.ins().uextend(I64, v(*expected))
            } else {
                v(*expected)
            };
            let w64 = b.ins().iconst(I32, i64::from(*ty == IntTy::I64));
            let vm = b.use_var(lctx.vmctx);
            let call = b.ins().call(
                lctx.thunks.futex_wait,
                &[vm, v(*addr), exp64, w64, v(*timeout)],
            );
            let res = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            vec![res]
        }
        Inst::MemoryNotify { addr, count } => {
            emit_align_check(b, lctx, v(*addr), 0, 4);
            let vm = b.use_var(lctx.vmctx);
            let call = b
                .ins()
                .call(lctx.thunks.futex_notify, &[vm, v(*addr), v(*count)]);
            let res = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            vec![res]
        }
        Inst::VcpuTlsGet => {
            let vm = b.use_var(lctx.vmctx);
            let ptr = b.ins().load(I64, MemFlags::trusted(), vm, OFF_CARRIER_TLS);
            vec![b.ins().load(I64, MemFlags::trusted(), ptr, 0)]
        }
        Inst::VcpuTlsSet { val } => {
            let vm = b.use_var(lctx.vmctx);
            let ptr = b.ins().load(I64, MemFlags::trusted(), vm, OFF_CARRIER_TLS);
            b.ins().store(MemFlags::trusted(), v(*val), ptr, 0);
            vec![]
        }

        // ---- Threads & fibers ----
        Inst::ThreadSpawn { func, arg0, arg1 } => {
            let vm = b.use_var(lctx.vmctx);
            let fidx = b.ins().iconst(I32, i64::from(*func));
            let call = b
                .ins()
                .call(lctx.thunks.thread_spawn, &[vm, fidx, v(*arg0), v(*arg1)]);
            let res = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            vec![res]
        }
        Inst::ThreadJoin { handle } => {
            let slot =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let out_ptr = b.ins().stack_addr(I64, slot, 0);
            let vm = b.use_var(lctx.vmctx);
            let call = b
                .ins()
                .call(lctx.thunks.thread_join, &[vm, v(*handle), out_ptr]);
            let status = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            let val = b.ins().stack_load(I64, slot, 0);
            vec![status, val]
        }
        Inst::ContNew { func, arg0 } => {
            let vm = b.use_var(lctx.vmctx);
            let call = b
                .ins()
                .call(lctx.thunks.fiber_new, &[vm, v(*func), v(*arg0)]);
            let res = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            vec![res]
        }
        Inst::ContResume { k, arg, block } => {
            emit_safepoint(b, lctx);
            let slot =
                b.create_sized_stack_slot(StackSlotData::new(StackSlotKind::ExplicitSlot, 8, 3));
            let out_ptr = b.ins().stack_addr(I64, slot, 0);
            let vm = b.use_var(lctx.vmctx);
            let blk_v = b.ins().iconst(I32, i64::from(*block));
            let call = b.ins().call(
                lctx.thunks.fiber_resume,
                &[vm, v(*k), v(*arg), blk_v, out_ptr],
            );
            let status = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            let val = b.ins().stack_load(I64, slot, 0);
            vec![status, val]
        }
        Inst::Suspend { value } => {
            let vm = b.use_var(lctx.vmctx);
            let call = b.ins().call(lctx.thunks.fiber_suspend, &[vm, v(*value)]);
            let res = b.inst_results(call)[0];
            emit_check_trap(b, lctx);
            vec![res]
        }

        // ---- Calls & capabilities ----
        Inst::Call { func, args } => {
            let vm = b.use_var(lctx.vmctx);
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(vm);
            for &a in args.iter() {
                call_args.push(v(a));
            }
            let call = b.ins().call(lctx.local_funcs[*func as usize], &call_args);
            let results: Vec<Value> = b.inst_results(call).to_vec();
            emit_check_trap(b, lctx);
            results
        }
        Inst::CallIndirect(c) => {
            let ft = func_sig(lctx.m, c.ty)?;
            let (code_ptr, sig_ref) = emit_indirect_resolve(b, lctx, v(c.idx), ft)?;
            let vm = b.use_var(lctx.vmctx);
            let mut call_args = Vec::with_capacity(1 + c.args.len());
            call_args.push(vm);
            for &a in c.args.iter() {
                call_args.push(v(a));
            }
            let call = b.ins().call_indirect(sig_ref, code_ptr, &call_args);
            let results: Vec<Value> = b.inst_results(call).to_vec();
            emit_check_trap(b, lctx);
            results
        }
        Inst::CapCall(c) => {
            let ft = func_sig(lctx.m, c.sig)?;
            let nargs = c.args.len();
            let nresults = ft.results.len();
            let args_ptr = if nargs == 0 {
                b.ins().iconst(I64, 0)
            } else {
                let slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (nargs * 8) as u32,
                    3,
                ));
                for (i, (&pt, &arg)) in ft.params.iter().zip(c.args.iter()).enumerate() {
                    let wide = widen_to_i64(b, pt, v(arg));
                    b.ins().stack_store(wide, slot, (i * 8) as i32);
                }
                b.ins().stack_addr(I64, slot, 0)
            };
            let (res_slot, res_ptr) = if nresults == 0 {
                (None, b.ins().iconst(I64, 0))
            } else {
                let slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    (nresults * 8) as u32,
                    3,
                ));
                (Some(slot), b.ins().stack_addr(I64, slot, 0))
            };
            let vm = b.use_var(lctx.vmctx);
            let tid = b.ins().iconst(I32, i64::from(c.type_id));
            let op = b.ins().iconst(I32, i64::from(c.op));
            let na = b.ins().iconst(I32, nargs as i64);
            let nr = b.ins().iconst(I32, nresults as i64);
            b.ins().call(
                lctx.thunks.cap_call,
                &[vm, v(c.handle), tid, op, args_ptr, na, res_ptr, nr],
            );
            emit_check_trap(b, lctx);
            let mut res = Vec::with_capacity(nresults);
            if let Some(slot) = res_slot {
                for (i, &rt) in ft.results.iter().enumerate() {
                    let wide = b.ins().stack_load(I64, slot, (i * 8) as i32);
                    res.push(narrow_from_i64(b, rt, wide)?);
                }
            }
            res
        }

        _ => return Err(KernelCompileError::Unsupported("stage2-or-simd-op")),
    };
    Ok(out)
}

fn widen_to_i64(b: &mut FunctionBuilder, ty: ValType, val: Value) -> Value {
    match ty {
        ValType::I32 | ValType::Ref | ValType::Cap => b.ins().uextend(I64, val),
        ValType::I64 => val,
        ValType::F32 => {
            let bits = b.ins().bitcast(I32, MemFlags::new(), val);
            b.ins().uextend(I64, bits)
        }
        ValType::F64 => b.ins().bitcast(I64, MemFlags::new(), val),
        ValType::V128 => val,
    }
}

fn narrow_from_i64(
    b: &mut FunctionBuilder,
    ty: ValType,
    wide: Value,
) -> Result<Value, KernelCompileError> {
    Ok(match ty {
        ValType::I32 | ValType::Ref | ValType::Cap => b.ins().ireduce(I32, wide),
        ValType::I64 => wide,
        ValType::F32 => {
            let lo = b.ins().ireduce(I32, wide);
            b.ins().bitcast(F32, MemFlags::new(), lo)
        }
        ValType::F64 => b.ins().bitcast(F64, MemFlags::new(), wide),
        ValType::V128 => return Err(KernelCompileError::Unsupported("v128")),
    })
}

fn emit_indirect_resolve(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    idx: Value,
    ft: &FuncType,
) -> Result<(Value, SigRef), KernelCompileError> {
    let vm = b.use_var(lctx.vmctx);
    let tbl = b.ins().load(I64, MemFlags::trusted(), vm, OFF_FN_TABLE);
    let mask = b
        .ins()
        .load(I64, MemFlags::trusted(), vm, OFF_FN_TABLE_MASK);
    let idx64 = b.ins().uextend(I64, idx);
    let slot = b.ins().band(idx64, mask);
    let byte_off = b.ins().ishl_imm(slot, 4); // size_of::<FnTableEntry>() == 16
    let entry_ptr = b.ins().iadd(tbl, byte_off);
    let code_ptr = b.ins().load(I64, MemFlags::trusted(), entry_ptr, 0);
    let got_tid = b.ins().load(I64, MemFlags::trusted(), entry_ptr, 8);
    let want_tid = b.ins().iconst(I64, TypeId::of_func_type(ft).0 as i64);
    let ok = b.ins().icmp(IntCC::Equal, got_tid, want_tid);
    let good_blk = b.create_block();
    let bad_blk = b.create_block();
    b.ins().brif(ok, good_blk, &[], bad_blk, &[]);

    b.switch_to_block(bad_blk);
    b.set_cold_block(bad_blk);
    let tc = b.ins().iconst(I64, Trap::IndirectCallType.code());
    b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);

    b.switch_to_block(good_blk);
    let mut sig = Signature::new(CallConv::Tail);
    sig.params.push(AbiParam::new(I64));
    for &p in &ft.params {
        sig.params.push(AbiParam::new(clif_ty(p)?));
    }
    for &r in &ft.results {
        sig.returns.push(AbiParam::new(clif_ty(r)?));
    }
    let sig_ref = b.import_signature(sig);
    Ok((code_ptr, sig_ref))
}

fn lower_term(
    b: &mut FunctionBuilder,
    lctx: &LowerCtx,
    cur_blk: u32,
    term: &Terminator,
    vals: &[Value],
    clif_blocks: &[ClifBlock],
) -> Result<(), KernelCompileError> {
    let v = |idx: ValIdx| vals[idx as usize];
    let edge_args = |args: &[ValIdx]| -> Vec<BlockArg> {
        args.iter().map(|&a| BlockArg::Value(v(a))).collect()
    };
    match term {
        Terminator::Br { target, args } => {
            if *target <= cur_blk {
                emit_safepoint(b, lctx);
            }
            b.ins()
                .jump(clif_blocks[*target as usize], &edge_args(args));
        }
        Terminator::BrIf {
            cond,
            then_blk,
            then_args,
            else_blk,
            else_args,
        } => {
            let then_back = *then_blk <= cur_blk;
            let else_back = *else_blk <= cur_blk;
            if !then_back && !else_back {
                b.ins().brif(
                    v(*cond),
                    clif_blocks[*then_blk as usize],
                    &edge_args(then_args),
                    clif_blocks[*else_blk as usize],
                    &edge_args(else_args),
                );
            } else {
                let t_tramp = b.create_block();
                let e_tramp = b.create_block();
                b.ins().brif(v(*cond), t_tramp, &[], e_tramp, &[]);

                b.switch_to_block(t_tramp);
                if then_back {
                    emit_safepoint(b, lctx);
                }
                b.ins()
                    .jump(clif_blocks[*then_blk as usize], &edge_args(then_args));

                b.switch_to_block(e_tramp);
                if else_back {
                    emit_safepoint(b, lctx);
                }
                b.ins()
                    .jump(clif_blocks[*else_blk as usize], &edge_args(else_args));
            }
        }
        Terminator::BrTable {
            idx,
            targets,
            default,
        } => {
            let mut tramps = Vec::with_capacity(targets.len());
            for (t_blk, t_args) in targets {
                let tb = b.create_block();
                tramps.push((tb, *t_blk, t_args.as_slice()));
            }
            let def_b = b.create_block();
            let def_call = b.func.dfg.block_call(def_b, &[]);
            let entries: Vec<_> = tramps
                .iter()
                .map(|&(tb, _, _)| b.func.dfg.block_call(tb, &[]))
                .collect();
            let jt = JumpTableData::new(def_call, &entries);
            let jt_ref = b.create_jump_table(jt);
            b.ins().br_table(v(*idx), jt_ref);

            for (tb, t_blk, t_args) in tramps {
                b.switch_to_block(tb);
                if t_blk <= cur_blk {
                    emit_safepoint(b, lctx);
                }
                b.ins()
                    .jump(clif_blocks[t_blk as usize], &edge_args(t_args));
            }
            b.switch_to_block(def_b);
            if default.0 <= cur_blk {
                emit_safepoint(b, lctx);
            }
            b.ins()
                .jump(clif_blocks[default.0 as usize], &edge_args(&default.1));
        }
        Terminator::Return(rvals) => {
            let rv: Vec<Value> = rvals.iter().map(|&a| v(a)).collect();
            b.ins().return_(&rv);
        }
        Terminator::ReturnCall { func, args } => {
            let vm = b.use_var(lctx.vmctx);
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(vm);
            for &a in args {
                call_args.push(v(a));
            }
            b.ins()
                .return_call(lctx.local_funcs[*func as usize], &call_args);
        }
        Terminator::ReturnCallIndirect { ty, idx, args } => {
            let ft = func_sig(lctx.m, *ty)?;
            let (code_ptr, sig_ref) = emit_indirect_resolve(b, lctx, v(*idx), ft)?;
            let vm = b.use_var(lctx.vmctx);
            let mut call_args = Vec::with_capacity(1 + args.len());
            call_args.push(vm);
            for &a in args {
                call_args.push(v(a));
            }
            b.ins().return_call_indirect(sig_ref, code_ptr, &call_args);
        }
        Terminator::Unreachable => {
            let tc = b.ins().iconst(I64, Trap::Unreachable.code());
            b.ins().jump(lctx.trap_blk, &[BlockArg::Value(tc)]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    struct StubBridge;

    impl KernelBridge for StubBridge {
        fn on_poll(&self, _flags: u64, _fuel: &mut FuelCell) -> Result<(), Trap> {
            Ok(())
        }
        fn on_fuel_refill(&self, _fuel: &mut FuelCell) -> Result<(), Trap> {
            Err(Trap::OutOfFuel)
        }
        fn futex_wait(
            &self,
            _addr: u64,
            _expected: i64,
            _width64: bool,
            _timeout_ns: i64,
            _fuel: &mut FuelCell,
        ) -> Result<i32, Trap> {
            Err(Trap::MemoryFault)
        }
        fn futex_notify(&self, _addr: u64, _count: u32) -> Result<i32, Trap> {
            Err(Trap::MemoryFault)
        }
        fn thread_spawn(&self, _func_idx: u32, _arg0: i64, _arg1: i64) -> Result<i32, Trap> {
            Err(Trap::ThreadFault)
        }
        fn thread_join(&self, _handle: i32, _fuel: &mut FuelCell) -> Result<(i64, i64), Trap> {
            Err(Trap::ThreadFault)
        }
        fn fiber_new(&self, _func_idx: u32, _arg0: i64) -> Result<i64, Trap> {
            Err(Trap::FiberFault)
        }
        fn fiber_resume(
            &self,
            _handle: i64,
            _val: i64,
            _block: bool,
            _fuel: &mut FuelCell,
            _carrier_tls: &mut i64,
        ) -> Result<(i32, i64), Trap> {
            Err(Trap::FiberFault)
        }
        fn fiber_suspend(&self, _val: i64) -> Result<i64, Trap> {
            Err(Trap::FiberFault)
        }
        fn cap_call(
            &self,
            _handle: i32,
            _type_id: u32,
            _op: u32,
            _args: &[i64],
            _results: &mut [i64],
        ) -> Result<(), Trap> {
            Err(Trap::CapFault)
        }
    }

    fn run_smoke(src: &str, args: &[i64], mode: BoundMode) -> Result<Vec<i64>, Trap> {
        let m = temen_text::parse_module(src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let compiled = compile_kernel_module(&m, mode).expect("compile");
        let mem = match compiled.memory_size_log2 {
            Some(log2) => KernelMemory::try_new(log2, &compiled.initial_data).expect("mem"),
            None => KernelMemory::empty(),
        };
        let poll = AtomicU64::new(0);
        let mut fuel = FuelCell::unlimited();
        let mut tls = 0i64;
        compiled.run_entry(
            0,
            args,
            &mem,
            &poll,
            &mut fuel,
            &mut tls,
            Arc::new(StubBridge),
        )
    }

    #[test]
    fn smoke_both_bound_modes_and_traps() {
        let fib_src = r#"
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vz = i64.const 0
  vo = i64.const 1
  br 1 (vz, vo, vz)
}
block 1 (va: i64, vb: i64, vi: i64, vn = 0.vn) {
  vlt = i64.lt_s vi vn
  br_if vlt 2 () 3 ()
}
block 2 in 1 (va = 1.va, vb = 1.vb, vi = 1.vi) {
  vnext = i64.add va vb
  vone = i64.const 1
  vi1 = i64.add vi vone
  br 1 (vb, vnext, vi1)
}
block 3 in 1 (va = 1.va) {
  vp = i64.const 16448
  i64.store vp va
  vr = i64.load vp
  return vr
}
}
"#;
        let oob_src = r#"
memory 8
func (i64) -> (i64) {
block 0 (vp: i64) {
  vr = i64.load vp
  return vr
}
}
"#;
        for mode in [BoundMode::BakedImmediate, BoundMode::DynamicVmCtx] {
            assert_eq!(run_smoke(fib_src, &[10], mode), Ok(vec![55]));
            // Tiny 256-byte window: offset 248 is in-bounds, offset 256 must fault MemoryFault!
            assert_eq!(run_smoke(oob_src, &[248], mode), Ok(vec![0]));
            assert_eq!(run_smoke(oob_src, &[256], mode), Err(Trap::MemoryFault));
        }
    }

    /// Stage 1d (`#3197`, red-team refinement #4): measure `BoundMode::DynamicVmCtx` vs
    /// `BoundMode::BakedImmediate` across 3 memory-intensive kernels:
    /// 1. `sum_i64`: tight sequential load loop (tests LICM hoisting of `mem_reserved`).
    /// 2. `saxpy_f64`: 2 loads + 1 FMA + 1 store per iteration (tests register pressure).
    /// 3. `ptr_chase_calls`: dependent load inside a non-inlined leaf call per iteration
    ///    (tests function-prologue cost of loading `mem_reserved` + `umax(mem_reserved, page)`).
    #[test]
    #[ignore = "benchmark: run explicitly with --release -- --ignored --nocapture"]
    fn bench_bound_mode_baked_vs_dynamic() {
        // 1 MiB window (size_log2 = 20); usable data starts at 16384 (above NULL guard).
        let sum_src = r#"
memory 20
func (i64, i64) -> (i64) {
block 0 (vbase: i64, vn: i64) {
  vz = i64.const 0
  br 1 (vz, vz)
}
block 1 (vi: i64, vacc: i64, vbase = 0.vbase, vn = 0.vn) {
  vlt = i64.lt_s vi vn
  br_if vlt 2 () 3 ()
}
block 2 in 1 (vi = 1.vi, vacc = 1.vacc, vbase = 0.vbase) {
  v3 = i64.const 3
  voff = i64.shl vi v3
  vptr = i64.add vbase voff
  vx = i64.load vptr
  vacc1 = i64.add vacc vx
  v1 = i64.const 1
  vi1 = i64.add vi v1
  br 1 (vi1, vacc1)
}
block 3 in 1 (vacc = 1.vacc) {
  return vacc
}
}
"#;

        let saxpy_src = r#"
memory 20
func (i64, i64, i64, f64) -> (f64) {
block 0 (vxbase: i64, vybase: i64, vn: i64, va: f64) {
  vz = i64.const 0
  vfz = f64.const 0.0
  br 1 (vz, vfz)
}
block 1 (vi: i64, vsum: f64, vxbase = 0.vxbase, vybase = 0.vybase, vn = 0.vn, va = 0.va) {
  vlt = i64.lt_s vi vn
  br_if vlt 2 () 3 ()
}
block 2 in 1 (vi = 1.vi, vsum = 1.vsum, vxbase = 0.vxbase, vybase = 0.vybase, va = 0.va) {
  v3 = i64.const 3
  voff = i64.shl vi v3
  vpx = i64.add vxbase voff
  vpy = i64.add vybase voff
  vx = f64.load vpx
  vy = f64.load vpy
  vnw = f64.fma va vx vy
  f64.store vpy vnw
  vsum1 = f64.add vsum vnw
  v1 = i64.const 1
  vi1 = i64.add vi v1
  br 1 (vi1, vsum1)
}
block 3 in 1 (vsum = 1.vsum) {
  return vsum
}
}
"#;

        let ptr_chase_src = r#"
memory 20
func (i64, i64) -> (i64) {
block 0 (vcur: i64, vn: i64) {
  vz = i64.const 0
  br 1 (vz, vcur)
}
block 1 (vi: i64, vcur: i64, vn = 0.vn) {
  vlt = i64.lt_s vi vn
  br_if vlt 2 () 3 ()
}
block 2 in 1 (vi = 1.vi, vcur = 1.vcur) {
  vnxt = call 1 (vcur)
  v1 = i64.const 1
  vi1 = i64.add vi v1
  br 1 (vi1, vnxt)
}
block 3 in 1 (vcur = 1.vcur) {
  return vcur
}
}
func (i64) -> (i64) {
block 0 (vp: i64) {
  vr = i64.load vp
  return vr
}
}
"#;

        let bench_one = |name: &str,
                         src: &str,
                         init: &dyn Fn(&mut [u8]),
                         args: &[i64],
                         runs: usize| {
            let m = temen_text::parse_module(src).expect("parse");
            temen_verify::verify_module(&m).expect("verify");

            let measure = |mode: BoundMode| -> (Vec<u64>, Vec<i64>) {
                let compiled = compile_kernel_module(&m, mode).expect("compile");
                let mem = KernelMemory::try_new(20, &[]).expect("mem");
                // SAFETY: exclusive setup before running guest code.
                unsafe {
                    let s = core::slice::from_raw_parts_mut(mem.base(), mem.mapped() as usize);
                    init(s);
                }
                let poll = AtomicU64::new(0);
                let mut fuel = FuelCell::unlimited();
                let mut tls = 0i64;
                let bridge: Arc<dyn KernelBridge> = Arc::new(StubBridge);
                // Warmup
                let mut last = compiled
                    .run_entry(
                        0,
                        args,
                        &mem,
                        &poll,
                        &mut fuel,
                        &mut tls,
                        Arc::clone(&bridge),
                    )
                    .expect("warmup");
                let mut samples = Vec::with_capacity(runs);
                for _ in 0..runs {
                    let t0 = Instant::now();
                    last = compiled
                        .run_entry(
                            0,
                            args,
                            &mem,
                            &poll,
                            &mut fuel,
                            &mut tls,
                            Arc::clone(&bridge),
                        )
                        .expect("run");
                    samples.push(t0.elapsed().as_nanos() as u64);
                }
                samples.sort_unstable();
                (samples, last)
            };

            let (b_samples, b_res) = measure(BoundMode::BakedImmediate);
            let (d_samples, d_res) = measure(BoundMode::DynamicVmCtx);
            assert_eq!(
                b_res, d_res,
                "{name}: BakedImmediate vs DynamicVmCtx result mismatch"
            );
            let b_med = b_samples[b_samples.len() / 2];
            let d_med = d_samples[d_samples.len() / 2];
            let ratio = d_med as f64 / b_med as f64;
            let delta_pct = (ratio - 1.0) * 100.0;
            println!(
                "{name:<18} | BakedImmediate median: {b_med:>9} ns | DynamicVmCtx median: {d_med:>9} ns | ratio: {ratio:.4}x ({delta_pct:+.2}%)"
            );
        };

        let base = 16384i64;
        let n_sum = 65_536i64;
        bench_one(
            "sum_i64 (64K)",
            sum_src,
            &|buf| {
                for i in 0..65_536usize {
                    let off = 16384 + i * 8;
                    buf[off..off + 8].copy_from_slice(&((i as i64 * 3 + 1).to_le_bytes()));
                }
            },
            &[base, n_sum],
            41,
        );

        let x_base = 16384i64;
        let y_base = 16384i64 + 32_768 * 8;
        let a_bits = 1.0001f64.to_bits() as i64;
        bench_one(
            "saxpy_f64 (32K)",
            saxpy_src,
            &|buf| {
                for i in 0..32_768usize {
                    let xo = 16384 + i * 8;
                    let yo = 16384 + 32_768 * 8 + i * 8;
                    buf[xo..xo + 8].copy_from_slice(&((i as f64 * 0.25).to_le_bytes()));
                    buf[yo..yo + 8].copy_from_slice(&((i as f64 * 0.5).to_le_bytes()));
                }
            },
            &[x_base, y_base, 32_768, a_bits],
            41,
        );

        // Pointer chase through a pseudo-random permutation of 16384 slots (100K steps).
        bench_one(
            "ptr_chase (100K)",
            ptr_chase_src,
            &|buf| {
                let slots = 16_384usize;
                for i in 0..slots {
                    let next = (i * 4099 + 1) % slots;
                    let off = 16384 + i * 8;
                    let target = (16384 + next * 8) as i64;
                    buf[off..off + 8].copy_from_slice(&target.to_le_bytes());
                }
            },
            &[base, 100_000],
            41,
        );
    }
}
