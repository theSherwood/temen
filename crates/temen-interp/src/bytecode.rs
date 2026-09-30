//! Phase-1b bytecode engine (see `INTERP_PERF.md`).
//!
//! Compiles a function once into a flat, operand-resolved op stream over a **function-wide
//! global-slot register file**, executed with **register windows** for calls (each activation
//! occupies `[base, base + nslots)` of one shared `regs` vector — a call opens the next window with
//! no per-call allocation, a return writes results back and restores the caller's window). This is
//! the production form of the Phase-1 ROI spike; it reuses the crate's audited semantic helpers
//! (`bin64`, `cmp32`, `fto_i`, …) and `Mem` — **no op semantics are duplicated here**, only the
//! dispatch/layout.
//!
//! Scope so far: scalar + memory + SIMD/`v128` + fences + direct & indirect calls; the synchronous
//! capability seam (generic `call.cap` + `self.*`, via `host.cap_dispatch_slots`); §12 **fibers**
//! (`cont.*`/`suspend`, cooperative single-vCPU switching in [`step_vcpu`]); and §12 **threads**
//! (`thread.spawn`/`join` + `memory.wait`/`notify`) on a cooperative single-threaded scheduler
//! ([`drive`]) over one shared `Mem`; and §14 **coroutines** (`Instantiator.spawn_coroutine`/`resume`
//! + `Yielder.yield`, inline-driven over a confined `nested_view` child window — including the
//! separate-**module** and **demand** (fault-driven-yield, lazy-paged) variants) and §14 **executor
//! children** (`Instantiator.instantiate`/`join` + the separate-module variant, scheduler-driven over
//! a confined child env with an attenuated `Instantiator`+`AddressSpace` powerbox and a `quota`
//! sub-budget) — §14 is fully covered (ops 0–7). Faithful for the
//! interleaving-invariant programs the oracle uses; and §22 **guest-driven JIT units**
//! (`Jit.install`/`uninstall`/`invoke` + cross-module `call.dyn` into an installed unit) over a
//! multi-module [`Domain`] (a runtime dispatch table spanning `mods`; `invoke` runs a unit nested
//! over the shared window/table). Hot scalar/memory ops dispatch inline; the SIMD/`v128`/fence long
//! tail is delegated to the reference [`super::eval_inst`]. Threads and fibers compose (the fiber
//! registry is run-shared, so fibers migrate across vCPUs); and **tail calls** (`return_call`/
//! `return_call.dyn`, reusing the current window — O(1) deep tail recursion); §GC **`gc.roots`**
//! (conservative root enumeration over the whole vCPU continuation — sound, not bit-identical, per
//! GC.md §3.2); and **durability** freeze/thaw for single-fiber vCPUs (IR-driven by the `temen-durable`
//! transform — the engine just runs the transformed module over a seeded window, via
//! [`compile_and_run_capture_reserved_with_host`]). [`compile_module`] returns `None` when a function
//! needs a seam not yet driven here — instantiate-mixed-with-fibers, `gc.roots`-mixed-with-threads, or
//! **multi-fiber** durable freeze — so callers (`super::run_with_host_fast`) fall back to the
//! tree-walker for those.
//!
//! `run`/`run_with_host` stay the tree-walker (the reference oracle); the bytecode engine is reached
//! via `run_fast`/`run_with_host_fast` (and, with a trap-time backtrace, `run_with_host_fast_traced`).
//! Correctness is gated by exact-equality harnesses against the tree-walker (`bytecode_diff.rs` — which
//! also checks trap-backtrace parity on every trapping generated module, `bytecode_{caps,fibers,threads,
//! coroutines,instantiate,separate_module,demand_coroutine,tailcall,debug,traced,gc_roots,durable,
//! dynlink}.rs`; `gc_roots` checks soundness rather than equality; `durable` checks freeze/thaw artifact
//! + round-trip equality; `traced` checks trap-time backtrace `IrPc`-equality with `run_with_host_traced`).
//!
//! Like the reference interpreter, it is total and panic-free: every slot/pc index is in range by
//! construction of the compiler, and `compile_module` rejects anything it can't lower.

use temen_ir::{
    BinOp, CastOp, CmpOp, ConvOp, DebugInfo, FBinOp, FCmpOp, FToI, FUnOp, FloatTy, Func, FuncIdx,
    IToF, Inst, IntTy, IntUnOp, LoadOp, Module, SpawnRec, StoreOp, Terminator, ValType, VarLoc,
};

use super::{
    bin32, bin64, cast, cmp32, cmp64, fbin32, fbin64, fcmp32, fcmp64, fto_i, fun32, fun64, i_to_f,
    intun32, intun64, slot_to_val, step, trunc_trap, val_to_slot, GuestMem, Host, LockUnpoisoned,
    Mem, MemLayout, Reg, Trap, Value, VarValue, DEFAULT_RESERVED_LOG2,
};

// ---- Per-function call profiler (opt-in `callprof` feature; tier-up break-even measurement) -------
// A thread-local histogram indexed by primary-module function index, bumped once per `Op::Call`. Off
// by default: with the feature disabled these items don't exist and the hot path is byte-identical.
#[cfg(feature = "callprof")]
mod callprof {
    use std::cell::RefCell;
    thread_local! {
        static COUNTS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
    }
    /// Arm the profiler with a zeroed histogram of `n` functions (call before the run).
    pub fn reset(n: usize) {
        COUNTS.with(|c| {
            let mut c = c.borrow_mut();
            c.clear();
            c.resize(n, 0);
        });
    }
    /// Record one call to `func` (a no-op if the histogram is smaller — non-primary-module callees).
    #[inline]
    pub fn hit(func: usize) {
        COUNTS.with(|c| {
            if let Some(slot) = c.borrow_mut().get_mut(func) {
                *slot += 1;
            }
        });
    }
    /// Snapshot the per-function call counts.
    pub fn snapshot() -> Vec<u64> {
        COUNTS.with(|c| c.borrow().clone())
    }

    thread_local! {
        /// Per-op execution counts keyed by the op's IR location `(func, block, inst)` — `inst` carries
        /// [`super::SRC_TERM`] for a terminator op — so a caller can weight any per-instruction static
        /// analysis (a call site's spill cost, a function's instruction count) by how often it ran.
        static OPS: RefCell<std::collections::HashMap<(u32, u32, u32), u64>> =
            RefCell::new(std::collections::HashMap::new());
    }
    /// Record one execution of the op at IR location `(func, block, inst)`.
    pub fn op(func: usize, loc: (u32, u32)) {
        OPS.with(|o| {
            *o.borrow_mut()
                .entry((func as u32, loc.0, loc.1))
                .or_insert(0) += 1
        });
    }
    /// Zero the per-op histogram.
    pub fn reset_ops() {
        OPS.with(|o| o.borrow_mut().clear());
    }
    /// Snapshot the per-op execution counts.
    pub fn op_snapshot() -> Vec<((u32, u32, u32), u64)> {
        OPS.with(|o| o.borrow().iter().map(|(k, v)| (*k, *v)).collect())
    }
}
/// Arm the per-function call profiler with a zeroed `n`-function histogram (opt-in `callprof`).
#[cfg(feature = "callprof")]
pub fn callprof_reset(n: usize) {
    callprof::reset(n);
    callprof::reset_ops();
}
/// Snapshot per-function call counts since the last [`callprof_reset`].
#[cfg(feature = "callprof")]
pub fn callprof_snapshot() -> Vec<u64> {
    callprof::snapshot()
}
/// Per-op execution counts since the last [`callprof_reset`], keyed by each op's IR location
/// `(func, block, inst)` in the primary module; a terminator op's `inst` carries [`SRC_TERM`]. An op
/// with no recorded location, and the `IntCmp` a fused `BrIfCmp` absorbs, are not counted.
#[cfg(feature = "callprof")]
pub fn callprof_op_snapshot() -> Vec<((u32, u32, u32), u64)> {
    callprof::op_snapshot()
}

/// Block-argument moves applied on a taken edge: `(src_slot, dst_slot)` pairs (frame-relative), with
/// a precomputed `aliasing` flag. A **non-aliasing** edge — the common case (an induction variable /
/// accumulator reads distinct value slots and writes the successor's param slots) — is applied by a
/// single direct pass. An **aliasing** edge, where some destination slot is also read as a source (a
/// parallel move that permutes/swaps params), must gather into `scratch` then scatter, so a value
/// isn't clobbered before it is read. Classifying once at compile time keeps the value off the hot
/// path (the `scratch` `push`+read per copy is only paid where correctness needs it).
struct Copies {
    pairs: Box<[(u32, u32)]>,
    aliasing: bool,
}
impl Copies {
    /// Build from resolved `(src, dst)` pairs, classifying `aliasing` = some `dst` is also some `src`
    /// (so a one-pass sequential copy could clobber a not-yet-read source). O(n²) over a tiny edge.
    fn new(pairs: Box<[(u32, u32)]>) -> Copies {
        let aliasing = pairs
            .iter()
            .any(|&(_, d)| pairs.iter().any(|&(s, _)| s == d));
        Copies { pairs, aliasing }
    }
}
/// A resolved branch edge: its arg copies plus the target op index (`pc`).
type Edge = (Copies, u32);

/// One resolved operation. Operands and results are **frame-window-relative slot indices** (added
/// to the activation's `base` at run time); branch targets are op indices (`pc`) within the same
/// function. Edge copies are `(src_slot, dst_slot)` pairs applied on a taken branch.
enum Op {
    Const {
        dst: u32,
        val: Reg,
    },
    IntBin {
        dst: u32,
        a: u32,
        b: u32,
        ty: IntTy,
        op: BinOp,
    },
    IntCmp {
        dst: u32,
        a: u32,
        b: u32,
        ty: IntTy,
        op: CmpOp,
    },
    IntUn {
        dst: u32,
        a: u32,
        ty: IntTy,
        op: IntUnOp,
    },
    Eqz {
        dst: u32,
        a: u32,
        ty: IntTy,
    },
    Convert {
        dst: u32,
        a: u32,
        op: ConvOp,
    },
    Select {
        dst: u32,
        cond: u32,
        a: u32,
        b: u32,
    },
    FBin {
        dst: u32,
        a: u32,
        b: u32,
        ty: FloatTy,
        op: FBinOp,
    },
    FUn {
        dst: u32,
        a: u32,
        ty: FloatTy,
        op: FUnOp,
    },
    FCmp {
        dst: u32,
        a: u32,
        b: u32,
        ty: FloatTy,
        op: FCmpOp,
    },
    FToISat {
        dst: u32,
        a: u32,
        op: FToI,
    },
    FToITrap {
        dst: u32,
        a: u32,
        op: FToI,
    },
    IToFConv {
        dst: u32,
        a: u32,
        op: IToF,
    },
    Cast {
        dst: u32,
        a: u32,
        op: CastOp,
    },
    RefFunc {
        dst: u32,
        func: u32,
    },
    Load {
        dst: u32,
        addr: u32,
        op: LoadOp,
        offset: u64,
    },
    Store {
        addr: u32,
        value: u32,
        op: StoreOp,
        offset: u64,
    },
    // Bulk-memory ops (D62). `MemCopy`/`MemMove` share the overlap-safe `Mem::mem_copy`.
    MemCopy {
        dst: u32,
        src: u32,
        len: u32,
    },
    MemMove {
        dst: u32,
        src: u32,
        len: u32,
    },
    MemFill {
        dst: u32,
        val: u32,
        len: u32,
    },
    AtomicLoad {
        dst: u32,
        addr: u32,
        ty: IntTy,
        offset: u64,
    },
    AtomicStore {
        addr: u32,
        value: u32,
        ty: IntTy,
        offset: u64,
    },
    AtomicRmw {
        dst: u32,
        addr: u32,
        value: u32,
        ty: IntTy,
        op: temen_ir::AtomicRmwOp,
        offset: u64,
    },
    AtomicCmpxchg {
        dst: u32,
        addr: u32,
        expected: u32,
        replacement: u32,
        ty: IntTy,
        offset: u64,
    },
    Br {
        copies: Copies,
        target: u32,
    },
    BrIf {
        cond: u32,
        then_copies: Copies,
        then_pc: u32,
        else_copies: Copies,
        else_pc: u32,
    },
    /// Slice 5a superinstruction: a block-final `IntCmp` fused with the `BrIf` that is its sole
    /// consumer — compare `a`/`b` (`ty`, `op`) and branch on the result, dropping one dispatch plus
    /// the boolean's write-then-reread. Emitted only by the **fused** compile (the fast path); the
    /// debug/trace compile is unfused so its step trace keeps one location per source instruction.
    BrIfCmp {
        a: u32,
        b: u32,
        ty: IntTy,
        op: CmpOp,
        then_copies: Copies,
        then_pc: u32,
        else_copies: Copies,
        else_pc: u32,
    },
    BrTable {
        idx: u32,
        arms: Box<[Edge]>,
        default: Edge,
    },
    Call {
        callee: u32,
        args: Box<[u32]>,
        dst: u32,
    },
    /// `call.dyn` through module 0's natural function table (slot `i` ⇒ func `i`; padding to a
    /// power of two traps). Resolved at run time from `idx` masked to the table length, then the
    /// resolved function's signature is checked against `want_params`/`want_results` (a forged or
    /// mistyped slot is an inert [`Trap::IndirectCallType`], matching [`super::dispatch_indirect`]).
    CallIndirect {
        idx: u32,
        args: Box<[u32]>,
        dst: u32,
        want_params: Box<[ValType]>,
        want_results: Box<[ValType]>,
    },
    /// Synchronous capability call (§3c) through the host powerbox — the guest is suspended, the
    /// host computes a result, and execution continues in the same activation (no scheduler/fiber).
    /// Only the **generic** powerbox path is lowered here; the executor/fiber capability variants
    /// (`Instantiator`, `Yielder`, `JIT`, `SharedRegion` op 4) are rejected by [`compile_inst`] and
    /// fall back to the tree-walker. Args/results cross as `i64` slots (the host-dispatch ABI);
    /// `results` carries `sig.results` so each returned slot is re-typed exactly as the tree-walker
    /// does.
    CapCall {
        type_id: u32,
        op: u32,
        handle: u32,
        args: Box<[u32]>,
        dst: u32,
        /// The call's `sig.params` — carried so an **import-bound** call that resolves to a §22 `Jit`
        /// driver op (`invoke`/`install`/`uninstall`) can be marshalled to the driver like a static
        /// `call.cap (JIT, op)` (which lowers straight to [`Op::JitInvoke`]). Empty when unused.
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §3.6 serve-loop core (ISSUES.md I36 slice 1): `svc.poll` (`call.cap CAP_SELF 9`) — drain
    /// the domain's inbound queue, running each servable dispatch as a handler activation over
    /// the one world. Rewind-driven like the tree-walk serve arm: an admitted handler's return
    /// linkage re-enters THIS op (pc un-advanced) with its result in `dst` (the linkage's result
    /// slot), which the re-execution settles into the ticket's completion cell before admitting
    /// the next dispatch; the final execution overwrites `dst` with the served count. Compiled
    /// only when the module-level qualification veto admits it (no park-capable seams — see
    /// [`compile_module`]), so a handler always runs to completion or traps.
    SvcPoll {
        dst: u32,
        /// `svc.wait` (op 10): identical drain, but a no-progress empty-queue execution parks
        /// the task on its domain ([`Outcome::SvcWait`]) instead of delivering a zero count; a
        /// caller's enqueue re-admits it and the rewound op re-executes the whole drain.
        wait: bool,
    },
    /// FORK.md §9.2 — `clone_caller` (self-op 11): fork-returns-twice, servicer side. Compiled only
    /// in a fork-serving module ([`Seams::bytecode_serves_fork`]). From within a serve handler, it
    /// duplicates the caller parked on this dispatch into a live twin and replies differently to each.
    /// The twin build needs the driver's task/env set, so the op resolves its reply args here and
    /// surfaces to the cooperative driver ([`Outcome::CloneCaller`]); the driver reads the running
    /// handler's `serve_ticket` to name the parked caller. Arity picks the mode (mirrors the oracle):
    /// 2 args = explicit `(reply_orig, reply_twin)`; 0/1 args = **pid mode** (`fork()`) — the parent
    /// sees the twin's task id, the child sees the arg (0). `-EINVAL` outside a handler.
    CloneCaller {
        /// The reply-value arg registers (0, 1, or 2), resolved to i64s in the driver.
        args: Box<[u32]>,
        dst: u32,
        /// Whether the `call.cap` has a result slot (the twin handle / errno lands here).
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap` (self-op 12): the servicer side of `wait(pid)`. From within a serve
    /// handler, reap a twin `pid` a prior `clone_caller` minted, on behalf of the caller parked on
    /// this dispatch — delivering the twin's exit status (now, or when it finishes). Surfaces to the
    /// driver ([`Outcome::Reap`]), which owns the task set + the `forked_twins` allow-set. `-EINVAL`
    /// outside a handler; `-ECHILD` for a `pid` this servicer did not mint (never a hang).
    Reap {
        /// The `pid` arg register (`None` = no arg → an out-of-range pid → `-ECHILD`).
        pid: Option<u32>,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §8.6 — `exec_module` (self-op 14): **`execve` image-replace** (#1080). The running vCPU
    /// replaces its own image with a granted *separate command module*, in place, keeping its task id
    /// and fuel. The rebuild needs the driver's task/env set + `dom.source`, so the op resolves its
    /// register operands here and surfaces to the cooperative driver ([`Outcome::Exec`]); the driver
    /// compiles the command, builds its powerbox from the by-name grant list, materializes its data
    /// into the caller's window, and swaps the task's activation. On any refusal the driver writes a
    /// probeable `-EINVAL` to `dst` and the caller keeps running (POSIX: `execve` returns only on
    /// failure). Cooperative-driver-only (like [`Op::CloneCaller`]); other drivers `ThreadFault`.
    ExecModule {
        /// The command `Module` handle register.
        module: u32,
        /// The by-name grant list `(ptr, count)` register pair (16-byte `{name_off, name_len, handle,
        /// flags}` records, the same layout op-13 `instantiate_module_named` reads).
        grants_ptr: u32,
        grants_n: u32,
        /// The command entry-function index register.
        entry: u32,
        /// The advisory `size_log2` register (the real bound is the caller's window).
        size_log2: u32,
        /// The `call.cap` result slot — receives `-EINVAL` on a refused exec (a successful exec never
        /// returns to this activation).
        dst: u32,
    },
    /// §3.6 (I36 slice 2) — `Instantiator.child_offer` (op 14): mint a live-callee offer over a
    /// running child's impl-export into the wirer's table. The authority check (the Instantiator
    /// handle) runs in the op exec; the mint itself needs the child's env/host, so it surfaces to
    /// the driver ([`Outcome::ChildOffer`]).
    ChildOffer {
        handle: u32,
        child: u32,
        export: u32,
        dst: u32,
    },
    // §7/§6 capability reflection `self.count`/`get`/`resolve`/`label`/`attest` are no longer
    // dedicated bytecode ops — they arrive as `call.cap CAP_SELF op 0/1/2/3/4` and compile to the
    // generic `Op::CapCall` (host `cap_dispatch_slots`), the same path the JIT thunk takes.
    /// §3.5 self-namespace extensions through the shared dispatch entry: `op` packs
    /// `(selfop | idx << 8)` (6 = `self.type_id`, 7 = `self.covers`, 8 =
    /// `export.handle`); `handle` is the optional live handle-register (covers only). One
    /// `i32` result.
    CapSelfExt {
        op: u32,
        handle: Option<u32>,
        dst: u32,
    },
    /// §12 fiber create (`cont.new`): register a pending fiber `(funcref, sp)` in the driver's
    /// registry and write its handle to `dst`. No switch — handled by the driver.
    ContNew {
        func: u32,
        sp: u32,
        dst: u32,
    },
    /// §12 fiber resume (`cont.resume`): switch into fiber `k`, delivering `arg`; the two results
    /// `(status, value)` land in `dst`, `dst+1` when the fiber suspends or returns. Driver-driven.
    /// `blocking` = the I48 `cont.resume.block` variant: on a still-parked fiber, idle the resumer's
    /// task on the fiber's event instead of returning `FIBER_PARKED` (still advisory — the guest keeps
    /// its loop; `FIBER_PARKED` remains a legal transient on the value-recheck path).
    ContResume {
        k: u32,
        arg: u32,
        dst: u32,
        blocking: bool,
    },
    /// §12 fiber suspend (`suspend`): hand `value` back to the resumer (status SUSPENDED) and park
    /// this fiber; `dst` receives the next resume's `arg`. Driver-driven.
    Suspend {
        value: u32,
        dst: u32,
    },
    /// `<setjmp.h>` `setjmp`: checkpoint this activation's resume point (the op after `setjmp`) keyed
    /// by the guest `jmp_buf` address in `buf`; `dst` receives `i32` 0 (or the long-jump value on
    /// re-entry). Intra-vCPU — handled inline, no scheduler escape.
    SetJmp {
        buf: u32,
        dst: u32,
    },
    /// `<setjmp.h>` `longjmp`: pop the activation stack back to the `setjmp` checkpoint named by `buf`,
    /// re-entering it with the `setjmp` result set to `val` (a `0` becomes `1`, per C). Noreturn.
    LongJmp {
        buf: u32,
        val: u32,
    },
    /// §12 `thread.spawn`: spawn a vCPU running `func` (a direct func index) with `(sp, arg)`; its
    /// handle lands at `dst`. Scheduler-driven.
    ThreadSpawn {
        func: u32,
        sp: u32,
        arg: u32,
        dst: u32,
    },
    /// §12 `thread.join`: park until child `handle` finishes; its result (or trap) lands at `dst`.
    ThreadJoin {
        handle: u32,
        dst: u32,
    },
    /// §14 `Instantiator.instantiate(entry, off, size_log2, quota)` (op 0): spawn a **confined
    /// executor child** running `entry` over `[off, off+2^size_log2)` of the holder's range, with an
    /// attenuated `Instantiator`+`AddressSpace` powerbox over its own window; its handle (or `EINVAL`)
    /// lands at `dst`. `handle` is the Instantiator cap (authority). Scheduler-driven (joinable).
    Instantiate {
        handle: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        quota: u32,
        dst: u32,
        /// §14 `instantiate_named` (op 11, PROCESS.md S2): the `(grants_ptr, grants_n)` register pair
        /// for the child's by-name grant list (op 0 is `None`). Same-module counterpart of op 13 — the
        /// child runs the holder's *own* program at `entry`, but its powerbox additionally carries the
        /// re-granted `grants_n × {name_off, name_len, handle, flags}` caps read from the parent window
        /// (via the shared `Host::spawn_named_child`), so a spawned stage resolves an inherited region
        /// (a ring end) or `stdout` by name — the concurrent-pipeline spawn.
        grants: Option<(u32, u32)>,
    },
    /// §14 `Instantiator.instantiate_module(module, entry, off, size_log2, quota)` (op 5): like
    /// [`Op::Instantiate`], but the child runs a host-granted **separate** `Module` (`module` is its
    /// handle, crossing as the first i64 arg) rather than the holder's own program — the §14
    /// "plugin-in-plugin" story. The driver resolves + compiles the module, materializes its data into
    /// the carve, and runs it as a confined executor child. `handle` is the Instantiator cap.
    InstantiateModule {
        handle: u32,
        module: u32,
        entry: u32,
        off: u32,
        size_log2: u32,
        quota: u32,
        dst: u32,
        /// §14 `instantiate_module_named` (op 13): the `(grants_ptr, grants_n)` register pair for the
        /// child's by-name grant list (op 5 is `None`). The driver reads the `grants_n × {name_off,
        /// name_len, handle, flags}` records from the parent window and re-grants each into the child
        /// powerbox (via the shared `Host::spawn_named_child`), so a spawned command resolves an
        /// inherited `stdout` by name — the shell "exec" primitive.
        grants: Option<(u32, u32)>,
    },
    /// PROCESS.md §5 `Instantiator.instantiate_detached(budget, module, grants_ptr, grants_n, entry,
    /// size_log2, quota[, args_ptr, args_len])` (op 15, #1286): a separate-module child in a **fresh
    /// window** minted through a `Budget` — no carve, no alias; the host owns the window. The
    /// optional trailing `(args_ptr, args_len)` is the spawn-time args payload copied to the child's
    /// `module_args_base()` (the detached twin of the op-13 "parent data segment in the carve"); the
    /// further optional `(region, child_off)` pre-maps a `SharedRegion` whole into the child's window
    /// at `child_off` before it starts (the 11-arg form).
    InstantiateDetached {
        handle: u32,
        budget: u32,
        module: u32,
        grants: Option<(u32, u32)>,
        entry: u32,
        size_log2: u32,
        quota: u32,
        args: Option<(u32, u32)>,
        premap: Option<(u32, u32)>,
        dst: u32,
    },
    /// CONSOLIDATION.md §3d — `instantiate_rec(record_ptr)` (op 17): the config-record spawn.
    /// The 56-byte record is **runtime data** (entry, carve, module, budget, grants — see the
    /// tree-walker's op-17 arm for the layout), so unlike the scalar spawns above the fields are
    /// read from the vCPU's confined window at **exec** time; the op then folds onto the same
    /// [`Outcome::Instantiate`] the drivers already service.
    InstantiateRec {
        handle: u32,
        rec: u32,
        dst: u32,
    },
    /// §14 `Instantiator.join(child)` (op 1): park until executor child `child` finishes; its result
    /// (or trap) lands at `dst`. `handle` is the Instantiator cap (authority). The join itself reuses
    /// the §12 thread machinery — children share one handle namespace (`threads`) with `thread.spawn`.
    InstJoin {
        handle: u32,
        child: u32,
        dst: u32,
    },
    /// §12 `memory.wait`: futex wait (`ty`-wide) on `addr` while it equals `expected`, up to
    /// `timeout` ns; the status (0/1/2) lands at `dst`. Scheduler-driven.
    MemoryWait {
        ty: IntTy,
        addr: u32,
        expected: u32,
        timeout: u32,
        dst: u32,
    },
    /// §12 `memory.notify`: wake up to `count` waiters on `addr`; the woken count lands at `dst`.
    MemoryNotify {
        addr: u32,
        count: u32,
        dst: u32,
    },
    /// §22 `Jit.install(code)` (op 3): compile the unit named by code-handle `code` to bytecode and
    /// install it into the domain's dispatch table; the slot (or `-ENOSPC`) lands at `dst`. `handle`
    /// is the `Jit` domain cap (authority).
    JitInstall {
        handle: u32,
        code: u32,
        dst: u32,
    },
    /// §22 `Jit.uninstall(slot)` (op 4): clear an installed table slot; `0`/`EINVAL` lands at `dst`.
    JitUninstall {
        handle: u32,
        slot: u32,
        dst: u32,
    },
    /// §22 `Jit.invoke(code, args…)` (op 1): run the unit named by `code` synchronously over the
    /// shared window/powerbox; its results land at `dst…`. `params`/`results` are the unit entry's
    /// expected signature (the `call.cap` sig minus the leading code-handle param), used to marshal
    /// args/results through the i64-slot ABI.
    JitInvoke {
        handle: u32,
        code: u32,
        args: Box<[u32]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §GC `gc.roots(heap_lo, heap_hi, mask, buf, cap)`: conservative root enumeration. Escapes to
    /// the driver, which scans every live activation of the vCPU's continuation (the active window,
    /// its call stack, its resume-chain ancestors, parked fibers, and coroutines) for words that —
    /// masked — land in `[lo, hi)`, writes the first `cap` (ascending, deduplicated) to guest memory
    /// at `buf`, and writes the total found to `dst`. Sound (a superset of the genuine roots), not
    /// bit-identical to the tree-walker — the backends over-approximate differently (GC.md §3.2).
    GcRoots {
        lo: u32,
        hi: u32,
        mask: u32,
        buf: u32,
        cap: u32,
        dst: u32,
    },
    Ret {
        srcs: Box<[u32]>,
    },
    /// `return_call`: a direct tail call — reuse the current activation window (no stack growth),
    /// staying in the caller's module; on return the callee returns to *this* activation's caller.
    TailCall {
        callee: u32,
        args: Box<[u32]>,
    },
    /// `return_call.dyn`: an indirect tail call — resolve through the runtime dispatch table
    /// (possibly cross-module), then reuse the current window like [`Op::TailCall`].
    TailCallIndirect {
        idx: u32,
        args: Box<[u32]>,
        want_params: Box<[ValType]>,
        want_results: Box<[ValType]>,
    },
    Unreachable,
    /// Long-tail value/store ops (SIMD, `v128` load/store, fences) delegated to the reference
    /// [`super::eval_inst`] — same semantics, no duplication. The original instruction keeps its
    /// **block-local** operand indices, so it's run against the sub-window `regs[base + block_base
    /// ..]`; `dst` is the frame-relative result slot (unused when `eval_inst` yields no value).
    Eval {
        inst: Box<Inst>,
        block_base: u32,
        dst: u32,
    },
    /// §12.8 4A.5 durable-runtime-internal: push the active context's shadow-SP word address (the
    /// `Vm`'s `durable_region_base`). The reference `eval_inst` can't service it (it needs the running
    /// context), so it gets a dedicated op like `vcpu.tls` would.
    DurableShadowBase {
        dst: u32,
    },
    /// §12 per-vCPU **thread-local register** read (`vcpu.tls.get`): push this `Vm`'s `tls` word to
    /// `dst`. The reference `eval_inst` traps `Malformed` on it (no vCPU context), so — like
    /// [`Op::DurableShadowBase`] — it gets a dedicated op rather than the `Eval` fallback. Seeded to
    /// the dense vCPU id (root = 0) at `Vm` construction; a spawned thread's `Vm` is re-seeded to its
    /// id (see `drive`'s `Spawn` arm). See the tree-walker's `Inst::VcpuTlsGet`.
    VcpuTlsGet {
        dst: u32,
    },
    /// §12 per-vCPU **thread-local register** write (`vcpu.tls.set`): set this `Vm`'s `tls` word to
    /// `val`. No result (like `store`).
    VcpuTlsSet {
        val: u32,
    },
}

/// Marks a [`Program::src`] entry as a **terminator** op's location (OR-ed into the `inst` field).
/// [`vm_trap_bt`] needs terminators distinguished from instructions: a trap at an instruction is
/// reported one past it (the tree-walker's cursor advance), a trap at a terminator (`unreachable`,
/// `return_call.dyn`) at the terminator itself. [`Vm::cur_ir_pc`] just masks the flag off — a
/// terminator is a stop position like any instruction (#1713). The flag is the high bit, never set
/// by a real block/inst count, so masking it off recovers the stored index.
pub const SRC_TERM: u32 = 1 << 31;

struct Program {
    ops: Vec<Op>,
    nslots: u32,
    /// Debug reverse map (Slice 1c-3): the source `(block, inst)` of each op. An instruction op maps
    /// to its `(block, inst)`; a **terminator** op maps to `(block, insts.len() | `[`SRC_TERM`]`)` —
    /// the `insts.len()` is the `inst` the tree-walker's `Vec<Frame>` carries for a terminator (it sits
    /// one past the block's last instruction). The tree-walker's debug seam (`run_inner`'s `before_op`)
    /// fires before every instruction **and** before the terminator (#1713), and [`Vm::cur_ir_pc`]
    /// reports both, keeping the engine's step/breakpoint location trace identical to the
    /// tree-walker's [`crate::IrPc`] sequence op-for-op.
    src: Box<[Option<(u32, u32)>]>,
}

thread_local! {
    /// Debugging aid (#1382): the function index that most recently executed `Op::Unreachable`, so a
    /// caller can name *where* a guest trapped — the trap itself is a unit variant carrying no location.
    /// Best-effort and single-shot (a trap ends the run); `u32::MAX` means "none recorded". Read it right
    /// after a run returns `Err(Trap::Unreachable)`.
    pub static LAST_UNREACHABLE_FUNC: core::cell::Cell<u32> = const { core::cell::Cell::new(u32::MAX) };
}

/// A whole compiled module: one [`Program`] per function plus each function's result types (for
/// reconstructing typed `Value`s at the entry boundary).
pub struct Compiled {
    /// The module's declared durable shadow arena (INVARIANTS.md #16), `None` if it declared none —
    /// a child window built for this program places its contexts here.
    shadow: Option<super::ShadowArena>,
    progs: Vec<Program>,
    result_types: Vec<Vec<ValType>>,
    /// Per-function `(params, results)` for `call.dyn` type-checking — the natural module-0
    /// function table indexes these directly (slot `i` ⇒ func `i`).
    sigs: Vec<(Vec<ValType>, Vec<ValType>)>,
    /// `len - 1` of the natural table (`next_power_of_two(n_funcs)`), used to mask a `ref.func`/fiber
    /// funcref to a module-local slot (the fiber/coroutine dispatch is module-0-natural).
    table_mask: usize,
}

impl Compiled {
    /// Total compiled **bytecode op count** across all functions — the structural-size measure of this
    /// threaded register-VM program (the analogue of the JIT's emitted code bytes / the IR's
    /// instruction count). The engine is a `Vec<Op>` per function, not a serialized byte stream, so op
    /// count, not a byte length, is the meaningful size.
    pub fn op_count(&self) -> usize {
        self.progs.iter().map(|p| p.ops.len()).sum()
    }
}

/// THREADS.md 4c-domain — a domain's `call.dyn` dispatch table, **shareable + installable across
/// parallel vCPUs** (mirrors the tree-walker's [`crate::DomainTable`]). Each slot packs `(module, func)`
/// (`module<<32 | func`, [`super::pack_slot`]); `module == TABLE_EMPTY` is trapping padding. Dispatch is
/// one `Acquire` load; `install` does a `Release` store, so a vCPU that observes a filled slot also
/// observes the unit pushed into the [`ModuleSource`] before it (the install serializes under the
/// source lock). Built once per domain (root / §14 child / coroutine); only the root's is installed into.
pub struct SharedSlots {
    slots: Box<[std::sync::atomic::AtomicU64]>,
}

impl SharedSlots {
    /// `2^table_log2` (at least `next_power_of_two(n_funcs)`) slots: the first `n_funcs` map to
    /// `(module, i)` (module 0 for the primary's natural table; a `k≥1` for a §14 separate-module
    /// child), the rest are trapping padding (fillable by [`Domain::install`]).
    fn new(n_funcs: usize, table_log2: u8, module: u32) -> SharedSlots {
        let len = (1usize << table_log2)
            .max(n_funcs.next_power_of_two())
            .max(1);
        let slots = (0..len)
            .map(|i| {
                std::sync::atomic::AtomicU64::new(if i < n_funcs {
                    super::pack_slot(module, i as u32)
                } else {
                    super::pack_slot(super::TABLE_EMPTY, 0)
                })
            })
            .collect();
        SharedSlots { slots }
    }

    fn len(&self) -> usize {
        self.slots.len()
    }

    /// #1297 — a fork twin's table: a **snapshot** of this one (same size, same slot words). The
    /// units the words name live in the domain's shared, append-only [`ModuleSource`], so the copy
    /// resolves every install made before the fork; installs after it diverge per domain.
    fn fork(&self) -> SharedSlots {
        use std::sync::atomic::{AtomicU64, Ordering};
        SharedSlots {
            slots: self
                .slots
                .iter()
                .map(|s| AtomicU64::new(s.load(Ordering::Acquire)))
                .collect(),
        }
    }

    /// Dispatch-path read: one `Acquire` load, paired with [`Domain::install`]'s `Release` store.
    #[inline]
    fn slot(&self, i: usize) -> super::TableSlot {
        super::unpack_slot(self.slots[i].load(std::sync::atomic::Ordering::Acquire))
    }
}

/// THREADS.md 4c-domain — a domain's compiled modules, **shared (`Arc`) and append-only** so installed
/// §22 units / §14 separate-module children are visible to every parallel vCPU without invalidating
/// references the way a growing `Vec<Compiled>` would: the modules live behind `Arc<Compiled>` (stable
/// address) inside a `Mutex<Vec<_>>` touched only on install or a reader's local-cache miss. `mods[0]`
/// is the primary; `k≥1` is an installed unit. A §14 child / coroutine shares the root's `ModuleSource`
/// (so its table's module indices resolve) but carries its own [`SharedSlots`].
struct ModuleSource {
    mods: std::sync::Mutex<Units>,
}

/// A [`ModuleSource`]'s units, and which of them are `execve`'d commands.
struct Units {
    code: Vec<std::sync::Arc<Compiled>>,
    /// Each unit an `execve` compiled, by its module's content digest ([`super::module_digest`]) —
    /// `(digest, index)`. A command is compiled once per run, however many processes exec it, as a
    /// JIT tree compiles it once per digest (#1825).
    commands: Vec<([u8; 32], usize)>,
}

impl ModuleSource {
    fn new(primary: Compiled) -> ModuleSource {
        ModuleSource::over(std::sync::Arc::new(primary))
    }

    /// [`ModuleSource::new`] over an already-`Arc`'d primary (the cross-run compiled-program cache,
    /// #1144): the shared `Arc<Compiled>` becomes `mods[0]` of a fresh source.
    fn over(primary: std::sync::Arc<Compiled>) -> ModuleSource {
        ModuleSource {
            mods: std::sync::Mutex::new(Units {
                code: vec![primary],
                commands: Vec::new(),
            }),
        }
    }

    /// A fresh clone of the module `Arc`s — a vCPU's lock-free local cache (cheap refcount bumps),
    /// refreshed on a miss. The lock acquire pairs with `install`'s push, so the snapshot sees it.
    fn snapshot(&self) -> Vec<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned().code.clone()
    }

    /// The primary program (module 0).
    fn primary(&self) -> std::sync::Arc<Compiled> {
        std::sync::Arc::clone(&self.mods.lock_unpoisoned().code[0])
    }

    /// Module `i` (`0` = primary, `k≥1` = an installed unit), or `None` if out of range.
    fn get(&self, i: usize) -> Option<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned().code.get(i).cloned()
    }

    /// Append a module (a §14 `instantiate_module` child's program) and return its index. (§22
    /// `Jit.install` instead goes through [`Domain::install`], which also fills a dispatch slot.)
    fn push(&self, unit: Compiled) -> usize {
        let mut mods = self.mods.lock_unpoisoned();
        mods.code.push(std::sync::Arc::new(unit));
        mods.code.len() - 1
    }

    /// The unit of the `execve`'d command whose module has content digest `digest`: the one this
    /// run already compiled, or `compile()`'s, appended. `None` when `compile` refuses. The compile
    /// runs outside the lock; a racing exec of the same command keeps the unit that landed first.
    fn command(
        &self,
        digest: &[u8; 32],
        compile: impl FnOnce() -> Option<Compiled>,
    ) -> Option<usize> {
        let find = |u: &Units| {
            u.commands
                .iter()
                .find(|(d, _)| d == digest)
                .map(|&(_, i)| i)
        };
        if let Some(i) = find(&self.mods.lock_unpoisoned()) {
            return Some(i);
        }
        let unit = compile()?;
        let mut mods = self.mods.lock_unpoisoned();
        if let Some(i) = find(&mods) {
            return Some(i);
        }
        mods.code.push(std::sync::Arc::new(unit));
        let i = mods.code.len() - 1;
        mods.commands.push((*digest, i));
        Some(i)
    }

    /// The **non-primary** units (`mods[1..]`) — a time-travel checkpoint captures these (cheap `Arc`
    /// refcount bumps) so a reverse-`seek` restore can re-push them and a separate-module coroutine/child
    /// frame's `module` index resolves as it did at capture. Paired with [`reset_extra`].
    fn extra_units(&self) -> Vec<std::sync::Arc<Compiled>> {
        self.mods.lock_unpoisoned().code[1..].to_vec()
    }

    /// Reset the pushed units to exactly `units` (keeping the primary at index 0) — the restore inverse
    /// of [`extra_units`]. Idempotent, so restoring twice into the same run is safe. The command index
    /// is dropped with them: a later exec compiles its command again rather than trust an index into
    /// units it did not see pushed.
    fn reset_extra(&self, units: &[std::sync::Arc<Compiled>]) {
        let mut mods = self.mods.lock_unpoisoned();
        mods.code.truncate(1);
        mods.code.extend(units.iter().cloned());
        mods.commands.clear();
    }
}

#[cfg(test)]
mod module_source_tests {
    use super::*;

    fn unit() -> Compiled {
        let m = temen_text::parse_module("func () -> () {\nblock 0 () {\n  return\n  }\n}\n")
            .expect("parse");
        compile_module(&m.funcs, &m.types, None).expect("compile")
    }

    #[test]
    fn an_execd_command_compiles_once_per_run() {
        let src = ModuleSource::new(unit());
        let mut compiles = 0;
        let mut exec = |digest: [u8; 32]| {
            src.command(&digest, || {
                compiles += 1;
                Some(unit())
            })
            .expect("compiles")
        };
        let a = exec([1; 32]);
        assert_eq!(
            exec([1; 32]),
            a,
            "an exec of the same command runs its unit"
        );
        let b = exec([2; 32]);
        assert_ne!(a, b, "another command has its own unit");
        assert_eq!(compiles, 2, "each command compiled once");
        assert_eq!(
            src.snapshot().len(),
            3,
            "the primary and a unit per command"
        );
    }

    #[test]
    fn a_restore_forgets_which_units_are_commands() {
        let src = ModuleSource::new(unit());
        let a = src.command(&[1; 32], || Some(unit())).expect("compiles");
        src.reset_extra(&[]);
        let mut compiled = false;
        let b = src
            .command(&[1; 32], || {
                compiled = true;
                Some(unit())
            })
            .expect("compiles");
        assert!(compiled, "the command is compiled again after a restore");
        assert_eq!((a, b), (1, 1), "into the restored units");
    }
}

/// Build a §14 child / coroutine's natural dispatch table over its `module` in the shared source.
fn build_table_for(n_funcs: usize, table_log2: u8, module: u32) -> SharedSlots {
    SharedSlots::new(n_funcs, table_log2, module)
}

/// Build the primary's natural module-0 dispatch table.
fn build_table(n_funcs: usize, table_log2: u8) -> SharedSlots {
    SharedSlots::new(n_funcs, table_log2, 0)
}

/// The program a **same-module** §14 child (op 0 / op 11) runs: the **spawning frame's** module —
/// the primary for a plain guest, the unit itself for an installed §22 unit — as `(index, program)`.
/// Every driver validates the child's entry against, and builds the child over, this one module, so
/// the two cannot disagree (#1726: they used to validate or build against module 0, so an installed
/// unit's child ran the base program's function of the same index). The module-aware rule
/// `thread.spawn` already follows (`VcpuEvent::Spawn { module }`). `None` only for a module index
/// the source does not hold, which a running frame cannot have.
fn spawner_module(source: &ModuleSource, spawner: &Vm) -> Option<(u32, std::sync::Arc<Compiled>)> {
    source
        .get(spawner.module)
        .map(|c| (spawner.module as u32, c))
}

/// A running domain (THREADS.md 4c-domain): its shared [`ModuleSource`] (`mods[0]` = primary, `k≥1` =
/// installed §22 units / §14 child modules) plus its own [`SharedSlots`] `call.dyn` dispatch table.
/// Both parts are interior-mutable + thread-safe, so a **parallel** driver can share `&Domain` across
/// vCPU threads and still `install`; the cooperative path is single-threaded (uncontended atomics/lock,
/// so dispatch order — hence determinism — is unchanged). A §14 child / coroutine shares the root's
/// `source` (its table's module indices resolve there) but carries its own `table`.
struct Domain {
    source: std::sync::Arc<ModuleSource>,
    /// Shared (`Arc`) so a persistent reactor can keep one table across the frames it runs — a §22
    /// `install` must outlive the frame that made it (#1296, the cross-tier bounce).
    table: std::sync::Arc<SharedSlots>,
}

impl Domain {
    fn new(primary: Compiled, table_log2: u8) -> Domain {
        Domain::over_primary(std::sync::Arc::new(primary), table_log2)
    }

    /// A domain over an existing shared table — the persistent cross-tier reactor's frames
    /// (`SharedProgram::run_over_grown_info`) all dispatch through one table, so a unit installed in
    /// one bounce is reachable from the next.
    fn child_shared(
        source: std::sync::Arc<ModuleSource>,
        table: std::sync::Arc<SharedSlots>,
    ) -> Domain {
        Domain { source, table }
    }

    /// Like [`Domain::new`], but over an **already-`Arc`'d** primary `Compiled` — so a caller that
    /// cached the compiled program across runs (#1144, the browser bash entry) reuses it via a cheap
    /// refcount bump instead of recompiling. A **fresh** `ModuleSource` still wraps it each run (a run
    /// pushes its own §14/exec'd command units into `mods[1..]` and must not inherit a prior run's),
    /// so only the immutable `mods[0]` program is shared.
    fn over_primary(primary: std::sync::Arc<Compiled>, table_log2: u8) -> Domain {
        let table = SharedSlots::new(primary.progs.len(), table_log2, 0);
        Domain {
            source: std::sync::Arc::new(ModuleSource::over(primary)),
            table: std::sync::Arc::new(table),
        }
    }

    /// A §14 confined-child domain over a (cloned `Arc`) **shared** `source` with its own dispatch
    /// `table`. Sharing the source keeps the parent's module archive reachable by index (so an
    /// `instantiate_module` child's pushed program resolves); the fresh `table` is the confinement —
    /// it carries only the child's own natural entries, never the parent's installed §22 unit slots
    /// (matching the tree-walker's `DomainTable::new(&cfuncs, 0)`).
    fn child(source: std::sync::Arc<ModuleSource>, table: SharedSlots) -> Domain {
        Domain {
            source,
            table: std::sync::Arc::new(table),
        }
    }

    /// `Jit.install`: append `unit` to the shared source and fill the first padding slot with
    /// `(module, 0)`, returning the slot — or `None` if the table is full (`-ENOSPC`; the unit is not
    /// appended). `&self` (interior-mutable) so a shared `&Domain` can install. See [`jit_install_into`].
    fn install(&self, unit: Compiled) -> Option<usize> {
        jit_install_into(&self.source, &self.table, unit)
    }

    /// `Jit.uninstall`: clear a filled padding slot (`≥ n_real`) back to trapping. See
    /// [`jit_uninstall_from`].
    fn uninstall(&self, slot: usize, n_real: usize) -> bool {
        jit_uninstall_from(&self.source, &self.table, slot, n_real)
    }
}

/// `Jit.install` over a raw `(source, table)` pair — the shared body of [`Domain::install`] and the
/// debug engines' `dbg_jit_install` (`DebugRun`/`ScheduledDebugRun` hold `source`/`table` as separate
/// fields, not a wrapped [`Domain`]). Append `unit` to the shared source and fill the first padding
/// slot with `(module, 0)`, returning the slot — or `None` if the table is full (`-ENOSPC`; the unit
/// is not appended). The whole op serializes under the source lock, and the slot store is `Release`,
/// so a reader that observes the slot also observes the pushed unit.
fn jit_install_into(source: &ModuleSource, table: &SharedSlots, unit: Compiled) -> Option<usize> {
    use std::sync::atomic::Ordering;
    let mut mods = source.mods.lock_unpoisoned();
    let slot = table
        .slots
        .iter()
        .position(|s| (s.load(Ordering::Relaxed) >> 32) as u32 == super::TABLE_EMPTY)?;
    mods.code.push(std::sync::Arc::new(unit));
    let module = (mods.code.len() - 1) as u32;
    table.slots[slot].store(super::pack_slot(module, 0), Ordering::Release);
    Some(slot)
}

/// `Jit.uninstall` over a raw `(source, table)` pair — the shared body of [`Domain::uninstall`] and
/// the debug engines' `dbg_jit_uninstall`. Clear a filled padding slot (`≥ n_real`) back to trapping,
/// returning success. A real-function slot (`< n_real`), out-of-range, or already-empty slot is
/// rejected. The unit stays in `source` (append-only); only the slot is reclaimed. Serialized under
/// the source lock.
fn jit_uninstall_from(
    source: &ModuleSource,
    table: &SharedSlots,
    slot: usize,
    n_real: usize,
) -> bool {
    use std::sync::atomic::Ordering;
    let _g = source.mods.lock_unpoisoned();
    if slot >= n_real
        && slot < table.slots.len()
        && (table.slots[slot].load(Ordering::Relaxed) >> 32) as u32 != super::TABLE_EMPTY
    {
        table.slots[slot].store(super::pack_slot(super::TABLE_EMPTY, 0), Ordering::Release);
        true
    } else {
        false
    }
}

/// The concurrency/park seams a module's instructions touch — one linear scan feeding both the
/// [`compile_module`] combination vetoes and the cross-backend serve qualification
/// ([`serve_qualifies`]).
#[derive(Default)]
struct Seams {
    has_coro: bool,
    has_fiber: bool,
    has_thread: bool,
    has_instantiate: bool,
    has_gc: bool,
    has_svc: bool,
    has_park_seam: bool,
    /// FORK.md §9 — `clone_caller` (self-op 11) present: the fork-returns-twice servicer primitive.
    /// A distinct seam because the **bytecode** engine now services it natively ([`bytecode_serves_fork`]),
    /// while the Cranelift routing still folds it (`svc_park_veto` keeps it). (`reap`, self-op 12, is
    /// **not** here yet — it stays a `has_park_seam` so fork+wait still folds; that is the next slice.)
    has_fork: bool,
}

impl Seams {
    /// The **serve-qualification veto**: a service point (`svc.poll` / `svc.wait`) coexisting with
    /// any seam that could park or unwind a handler mid-dispatch. This is the single definition of
    /// that disjunction — consulted by both the bytecode compile gate ([`compile_module`]) and the
    /// exported [`serve_qualifies`] that temen-run's JIT routing folds on — so the two backends can
    /// never drift over which modules serve natively vs. decline to the tree-walk oracle. Adding a
    /// new park-capable seam means extending this one list. (INVARIANTS.md §9: one veto predicate,
    /// one definition.)
    ///
    /// **Fork rides this veto for Cranelift** (`has_fork` is in the disjunction), so temen-run's JIT
    /// routing still folds a forking module to the oracle — the Cranelift fork slice is unbuilt
    /// (FORK.md §9.1). The **bytecode** engine, in contrast, services `clone_caller`/`reap` natively;
    /// its compile gate takes the [`bytecode_serves_fork`] escape past this veto (a per-backend split,
    /// the one the two-predicate structure exists to allow — the bytecode gate and `serve_qualifies`
    /// legitimately diverge on fork until Cranelift catches up).
    fn svc_park_veto(&self) -> bool {
        self.has_svc
            && (self.has_park_seam
                || self.has_fiber
                || self.has_thread
                || self.has_coro
                || self.has_instantiate
                || self.has_gc
                || self.has_fork)
    }

    /// FORK.md §9.2 — the **bytecode fork-serving escape**: a serving module the bytecode engine can
    /// run `clone_caller` in natively even though [`svc_park_veto`] folds it (for Cranelift). The
    /// bounded shape: it serves (`has_svc`) and forks (`has_fork`), the manager may spawn children
    /// (`has_instantiate` — the fork topology needs it), and **no other** seam that could park a
    /// *handler* mid-dispatch is present. `clone_caller` itself never parks the handler (it reshapes
    /// the parked caller and returns), so the serve rewind linkage stays intact; the manager's
    /// instantiate/join park ordinary tasks, not handlers. Deliberately narrow (fork-shaped modules
    /// only) to bound the blast radius vs. the general serve+spawn case, which stays folded. A fork
    /// handler that *also* parked (e.g. joined) is out of this shape and is not admitted here.
    fn bytecode_serves_fork(&self) -> bool {
        self.has_svc
            && self.has_fork
            && !self.has_park_seam
            && !self.has_fiber
            && !self.has_thread
            && !self.has_coro
            && !self.has_gc
    }
}

fn scan_seams(funcs: &[Func]) -> Seams {
    let mut s = Seams::default();
    for f in funcs {
        for b in &f.blocks {
            for inst in &b.insts {
                match inst {
                    // ops 0/1 = instantiate/join, op 5 = instantiate_module, op 13 =
                    // instantiate_module_named, op 15 = instantiate_detached, op 17 = instantiate_rec
                    // (all executor children,
                    // scheduler-driven — the grant-carrying spawns re-grant caps but spawn the same
                    // kind of confined task); everything else on INSTANTIATOR is the legacy coroutine
                    // residue. Classifying the named spawns as `has_instantiate` (not `has_coro`) is
                    // load-bearing: a concurrent pipeline mixes them with `memory.wait`/`notify`
                    // (`has_thread`), and the `has_coro && has_thread` veto would otherwise fall the
                    // whole module back to the tree-walker.
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR,
                        op: 0 | 1 | 5 | 13 | 14 | 15 | 17,
                        ..
                    } => s.has_instantiate = true,
                    Inst::CapCall {
                        type_id: super::cap_id::INSTANTIATOR,
                        ..
                    } => s.has_coro = true,
                    // I38's **timed** `svc.wait` (op 10 with the optional timeout arg) needs
                    // the scheduler's deadline machinery — oracle-only; veto like a park seam
                    // so both fast backends decline the module.
                    Inst::CapCall {
                        type_id: temen_ir::CAP_SELF_TYPE_ID,
                        op: 10,
                        args,
                        ..
                    } if !args.is_empty() => {
                        s.has_svc = true;
                        s.has_park_seam = true;
                    }
                    // §3.6 service points (I36 slice 1): svc.poll/svc.wait sites — natively
                    // servable only when nothing in the module could park a handler (below).
                    Inst::CapCall {
                        type_id: temen_ir::CAP_SELF_TYPE_ID,
                        op: 9 | 10,
                        ..
                    } => s.has_svc = true,
                    // FORK.md §9 — `clone_caller` (11) / `reap` (12): the fork servicer primitives.
                    // The bytecode engine services both natively (the [`bytecode_serves_fork`] escape
                    // admits the fork topology; the `VcpuStop::CloneCaller`/`Reap` driver arms build
                    // the twin and reap it), so they are the `has_fork` seam — folded for Cranelift
                    // (`svc_park_veto` keeps `has_fork`) but run natively on bytecode.
                    Inst::CapCall {
                        type_id: temen_ir::CAP_SELF_TYPE_ID,
                        op: 11 | 12,
                        ..
                    } => s.has_fork = true,
                    // A blocking stream `read` (type 0 op 0) can stdin-park, and an import call
                    // can be *bound* to one at spawn — either inside a handler would need the
                    // tree-walker's FIBER_PARKED (completed-but-not-replied) machinery.
                    Inst::CapCall {
                        type_id: super::cap_id::STREAM,
                        op: 0,
                        ..
                    }
                    | Inst::CapCall {
                        type_id: temen_ir::CAP_IMPORT_TYPE_ID,
                        ..
                    }
                    | Inst::CallImport { .. }
                    | Inst::SetJmp { .. }
                    | Inst::LongJmp { .. } => s.has_park_seam = true,
                    Inst::ContNew { .. }
                    | Inst::ContResume { .. } // I48: `block` flag is advisory here
                    | Inst::Suspend { .. } => s.has_fiber = true,
                    Inst::ThreadSpawn { .. }
                    | Inst::ThreadJoin { .. }
                    | Inst::MemoryWait { .. }
                    | Inst::MemoryNotify { .. } => s.has_thread = true,
                    Inst::GcRoots { .. } => s.has_gc = true,
                    _ => {}
                }
            }
        }
    }
    s
}

/// §3.6 (I36): the **serve qualification** — `funcs` contain a service point (`svc.poll` /
/// `svc.wait`) and no seam that could park or unwind a handler mid-dispatch, so a fast backend
/// may run the serve loop natively (every handler runs to completion or traps; the tree-walk
/// oracle's fiber-park machinery is never needed). The veto is module-wide, so it covers
/// handlers' transitive callees for free. This is the same predicate [`compile_module`]'s veto
/// applies — exported so temen-run's JIT routing folds exactly the modules this engine declines
/// (one definition, no drift). A module with no service point returns `false` (it has nothing
/// to serve natively; the caller decides what that means).
pub fn serve_qualifies(funcs: &[Func]) -> bool {
    let s = scan_seams(funcs);
    s.has_svc && !s.svc_park_veto()
}

/// Lower every function (fast path — superinstruction-**fused**, Slice 5a), or `None` if any uses an
/// op outside this slice's subset. This is what every production/runtime path calls.
/// CONSOLIDATION.md §3d — the **module-level** admission for the record spawn (op 17): a module
/// that could build a *pager* record (it has impl exports) declines to the tree-walk oracle, which
/// owns demand paging; with no impl exports every pager record `CapFault`s identically on every
/// tier, so the exec arm's fail-closed pager check is exact. This mirrors temen-run's
/// `module_demand_spawns` fold on the Cranelift tier — one predicate per tier boundary, consulted
/// by every `&Module` compile entry (INVARIANTS.md §9).
fn compile_module_for(m: &Module) -> Option<Compiled> {
    let uses_rec = m.funcs.iter().flat_map(|f| f.blocks.iter()).any(|b| {
        b.insts.iter().any(|i| {
            matches!(
                i,
                Inst::CapCall {
                    type_id: super::cap_id::INSTANTIATOR,
                    op: 17,
                    ..
                }
            )
        })
    });
    // §3d pager guard: an op-17 record-spawn module with impl exports *could* build a pager record
    // (which behaves differently), so it folds to the oracle — **except** a fork-shaped module
    // (FORK.md §9.2): its op-17 records are ordinary executor spawns (the manager spawning the
    // server/guest with by-name grants — the only in-module spawn-with-grants the bytecode tier
    // drives), and its impl export is the fork server. `bytecode_serves_fork` bounds this to the
    // fork shape; other serving record-spawn modules still fold.
    if uses_rec && !m.impl_exports.is_empty() && !scan_seams(&m.funcs).bytecode_serves_fork() {
        return None;
    }
    compile_module(&m.funcs, &m.types, m.memory.and_then(|x| x.shadow))
}

/// §3d — validate + **drain** a spawn record's `Budget` at a driver's commit site, returning the
/// child's funded fuel. `Ok(None)` = refuse the spawn `-EINVAL` with the budget **intact** (mem
/// quota short, or the compiled-tier narrowed gap: a bounded spawn ceiling / bounded-zero fuel,
/// which this tier — like the Cranelift thunk — cannot represent; flip those when child vCPU
/// quotas / zero-fuel children land here). `Err(CapFault)` = the handle vanished since the exec
/// arm's peek (a shared-powerbox race). The fund rule is the tree-walker's: bounded fuel is
/// `min(budget, parent_remaining)`, unbounded inherits the parent's remaining.
fn take_spawn_budget(
    host: &mut Host,
    budget: i32,
    child_size: u64,
    parent_fuel: u64,
) -> Result<Option<u64>, Trap> {
    // Peek + mem-quota gate is the shared `Host::budget_for_spawn` (#911): `None` = dangling handle
    // (CapFault), `Some(Err(()))` = bounded mem quota short (refuse `-EINVAL`, budget intact).
    let (fuel, spawn) = match host.budget_for_spawn(budget, child_size, false) {
        None => return Err(Trap::CapFault),
        Some(Err(())) => return Ok(None), // mem quota short — refuse, budget intact
        Some(Ok(v)) => v,
    };
    if spawn >= 0 || fuel == 0 {
        return Ok(None); // narrowed gap (see doc) — refuse, budget intact
    }
    // Commit: drain to zero. `take_budget` returns the pre-drain state, so the fuel it reports
    // equals the `fuel` peeked above — fund from that. Bounded fuel is `min(budget, parent)`,
    // unbounded inherits the parent's remaining. (#989 slice 1b — the child's `channel` cap is
    // peeked separately by the inline-building spawn arms before this drain; see `set_channel_cap`.)
    host.take_budget(budget).ok_or(Trap::CapFault)?;
    Ok(Some(if fuel >= 0 {
        (fuel as u64).min(parent_fuel)
    } else {
        parent_fuel
    }))
}

/// The §14 carve **geometry** check (D19), the single definition every spawn/instantiate driver and
/// tree-walk arm shares: a child window of `1 << size_log2` bytes at offset `off` fits inside the
/// parent's `[0, isize)` iff the size is a valid power of two (`size_log2` in `0..64`), `off` is
/// size-aligned, and the whole span lies within `isize`. Overflow-free (an out-of-range `size_log2`
/// or a `off + size` past `u64` yields `false`, not a shift/overflow). A future bound tweak lives
/// here, not in ten pasted copies (#911).
///
/// #964: the carve may also not dip into the holder window's reserved NULL region — `ibase + off`
/// (the carve's window-relative base; `ibase` is the holder's own base) must clear `null_guard`
/// (`0` = unguarded, trivially true). The host seeds/copies a carve outside the guarded call, and
/// the reserved region is permanent by design, so a below-guard carve is refused, not admitted.
pub(crate) fn carve_fits(
    off: u64,
    size_log2: i64,
    isize: u64,
    ibase: u64,
    null_guard: u64,
) -> bool {
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    child_size != 0
        && child_size <= isize
        && off & (child_size - 1) == 0
        && off.checked_add(child_size).is_some_and(|e| e <= isize)
        && ibase.checked_add(off).is_some_and(|b| b >= null_guard)
}

/// The §14 entry-shape rule and the starter handles each shape takes live in `temen-ir`, the one
/// crate every tier shares — the Cranelift JIT's spawn arms read the same definition (#911, #1720).
pub use temen_ir::{child_entry_handles, child_entry_ok};

/// [`child_entry_handles`] as the interpreters' entry args.
pub(crate) fn child_entry_args(arity: usize, inst: i32, space: i32) -> Vec<Value> {
    child_entry_handles(arity, inst, space)
        .map(Value::I64)
        .collect()
}

/// A §14 confined spawn — op 0 (`instantiate`), op 5 / op 13 (`instantiate_module[_named]`), or a
/// §3d record (op 17) — as its lowering resolved it. One event for all of them (INVARIANTS #15): a
/// separate-module child is this spawn with a `module`, not a second variant.
#[derive(Clone, Copy)]
struct ConfinedSpawn {
    /// The holder's `Instantiator` range `[ibase, ibase + isize)` in its own window.
    ibase: u64,
    isize: u64,
    /// The granted `Module` handle a separate-module child runs; `None` runs the spawning frame's
    /// own module (#1726).
    module: Option<i32>,
    entry: i64,
    /// The carve: `1 << size_log2` bytes at holder-relative offset `off`.
    off: i64,
    size_log2: i64,
    /// The child's fuel quota (`<= 0` inherits the parent's remaining) when no budget funds it.
    quota: i64,
    /// The by-name grant list's `(ptr, count)` in the holder's window (op 13, a record's); `None`
    /// grants only the starter caps.
    grants: Option<(u64, u64)>,
    /// A record's `Budget` handle (`0` = none), drained at the commit site.
    budget: i32,
}

/// A §5 detached spawn (op 15, #1286) as its lowering resolved it: a separate module in a fresh
/// window of its own, never a carve of the spawner's.
#[derive(Clone, Copy)]
struct DetachedSpawn {
    /// The `Budget` handle whose `mem` quota pays for the window.
    budget: i32,
    /// The granted `Module` handle the child runs.
    module: i32,
    entry: i64,
    /// The window: `1 << size_log2` bytes, the module's declared memory.
    size_log2: i64,
    /// The child's fuel quota (`<= 0` inherits the parent's remaining).
    quota: i64,
    /// The by-name grant list's `(ptr, count)` in the spawner's window; `None` grants only the
    /// starter caps.
    grants: Option<(u64, u64)>,
    /// The spawn-time payload `(ptr, len)` in the spawner's window, seeded at the child's
    /// `module_args_base()`.
    args: Option<(u64, u64)>,
    /// A `SharedRegion` of the spawner's, `(region, child_off)`, aliased into the child's window
    /// before it starts.
    premap: Option<(i32, u64)>,
}

/// What an admitted child runs.
enum ChildProgram {
    /// The spawning frame's own module (#1726): a same-module child (op 0, a module-less op 17).
    Spawner(u32, std::sync::Arc<Compiled>),
    /// A granted separate module, compiled; [`land`](ChildProgram::land) pushes it to the source.
    Granted(Compiled),
}

impl ChildProgram {
    /// Land the program in `source` (a granted module is pushed and gets its own index): the module
    /// index the child runs, and its compiled unit.
    fn land(self, source: &ModuleSource) -> Result<(u32, std::sync::Arc<Compiled>), Trap> {
        match self {
            ChildProgram::Spawner(m, p) => Ok((m, p)),
            ChildProgram::Granted(c) => {
                let m = source.push(c);
                Ok((m as u32, source.get(m).ok_or(Trap::Malformed)?))
            }
        }
    }
}

/// A §14 confined or §5 detached child as every in-process bytecode driver builds it — the **one
/// definition** of admission and of the child powerbox (INVARIANTS #15), made by
/// [`admit_confined_child`] or [`admit_detached_child`] and shared by the cooperative executor
/// (`drive`), the OS-thread parallel driver (`run_vcpu_parallel`) and the debug scheduler. Only *how
/// the child is scheduled* differs per driver: an executor task, an OS thread, a debug task. (Until
/// #1855 each driver admitted a confined child its own way, and two of them trapped on a budget or a
/// grant list the cooperative driver serves.)
struct AdmittedChild {
    /// The child's window: a view of the holder's carve (confined), or a fresh window of its own
    /// (detached) once its [`FreshWindow`] is built over the driver's backing.
    mem: Option<Mem>,
    /// The child powerbox: the starter `Instantiator`/`AddressSpace`, the by-name re-grants, the
    /// module the child serves, its import manifest bound, and a funding budget's channel ceiling.
    host: Host,
    program: ChildProgram,
    /// The entry's arguments: the starter `Instantiator` (plus `AddressSpace` for a two-arg entry).
    args: Vec<Value>,
    /// The child's fuel: a funding budget's, or the op's quota clamped to the parent's remaining.
    fuel: u64,
    /// A detached child's window lease `(budget, bytes)`: `Budget.mem` accounts live windows
    /// (INVARIANTS #3, 2026-09-29), so the driver gives these bytes back to the spawner's budget when
    /// the child ends. `None` for a confined (carve) child, which spends no `Budget.mem`.
    lease: Option<(i32, u64)>,
}

/// Admit a §14 confined spawn and build the child. `host` is the spawning task's own powerbox, where
/// its handles resolve and its budget is charged (#1727); `pm` its window — the holder the carve is
/// cut from, the grant list is read from and a module's data segments land in; `parent_fuel` its
/// remaining fuel; `spawner` the spawning frame, whose module a same-module child runs.
///
/// `Ok(None)` is a **probeable refusal**: the driver lands `-EINVAL`, and the budget is intact. `Err`
/// is a trap: a forged module handle, a module this engine cannot lower, an unreadable grant record,
/// or a handle the parent may not re-grant. The order is the tree-walker's — program, entry and
/// carve; then a module's data segments, the powerbox and its manifest; the budget last, so a refused
/// spawn charges nothing. The vCPU ceiling is the scheduler's to check afterwards, as for op 15.
fn admit_confined_child(
    host: &mut Host,
    pm: Option<&Mem>,
    parent_fuel: u64,
    source: &ModuleSource,
    spawner: &Vm,
    s: ConfinedSpawn,
) -> Result<Option<AdmittedChild>, Trap> {
    // The program, and for a separate module its declared window, data segments and module.
    let (program, granted) = match s.module {
        None => {
            let (m, p) = spawner_module(source, spawner).ok_or(Trap::Malformed)?;
            (ChildProgram::Spawner(m, p), None)
        }
        Some(mh) => {
            let g = host.resolve_module(mh)?;
            // A module this engine cannot lower is the one place a guest-provided program outruns
            // coverage (no tree-walker fallback mid-run) — a `Malformed` trap, as for `Jit.install`.
            let c = compile_module(&g.funcs, &g.types, g.shadow).ok_or(Trap::Malformed)?;
            let granted = (
                g.memory_log2,
                g.data.clone(),
                std::sync::Arc::clone(&g.module),
            );
            (ChildProgram::Granted(c), Some(granted))
        }
    };
    let sig = match &program {
        ChildProgram::Spawner(_, p) => p.sigs.get(s.entry as usize),
        ChildProgram::Granted(c) => c.sigs.get(s.entry as usize),
    };
    let arity = sig.map_or(0, |(p, _)| p.len());
    let ok_entry = sig.is_some_and(|(p, r)| child_entry_ok(p, r));
    let child_size = if (0..64).contains(&s.size_log2) {
        1u64 << s.size_log2
    } else {
        0
    };
    let off = s.off as u64;
    // #1094: the holder's own guard. A nested holder in a sub-guard carve is unguarded, so it may
    // carve below the root's guard.
    let guard = pm.map_or(0, |m| m.null_guard);
    let fits = carve_fits(off, s.size_log2, s.isize, s.ibase, guard);
    // A separate module's carve is at least its declared memory (FORK.md §8.6 / #773: the span above
    // it is heap room an allocating phase grows into via `vm_map`; §2 still confines every access
    // to the carve).
    let mod_ok = granted
        .as_ref()
        .is_none_or(|(ml, _, _)| ml.is_some_and(|ml| ml <= s.size_log2 as u8));
    if !ok_entry || !fits || !mod_ok {
        return Ok(None);
    }
    // Holder-relative → backing-absolute, so nesting composes at any depth.
    let base = pm.map_or(0, |m| m.window.base()) + s.ibase + off;
    // A module's data segments land in the carve now, as if the child wrote them (the verifier
    // bounded them to its declared window). `readonly` is not enforced for a nested child —
    // intra-domain self-corruption is a §1 non-goal — as on the tree-walker.
    if let (Some((_, data, _)), Some(m)) = (&granted, pm) {
        m.write_segments(base, data, child_size);
    }
    let module = granted.as_ref().map(|(_, _, m)| m);
    let Some((mut child_host, cinst, cas)) =
        confined_child_host(host, pm, s.grants, child_size, module)?
    else {
        return Ok(None);
    };
    // §3d: a record's budget funds the child here, after every other refusal. #989 slice 1b: its
    // `channel` ceiling is read before `take_spawn_budget` drains it.
    let fuel = if s.budget != 0 {
        let channel = host.peek_budget(s.budget).map(|b| b.channel);
        let Some(fuel) = take_spawn_budget(host, s.budget, child_size, parent_fuel)? else {
            return Ok(None);
        };
        if let Some(cap) = channel {
            child_host.set_channel_cap(cap);
        }
        fuel
    } else if s.quota <= 0 {
        parent_fuel
    } else {
        (s.quota as u64).min(parent_fuel)
    };
    Ok(Some(AdmittedChild {
        mem: pm.map(|m| m.nested_view(base, s.size_log2 as u8, m.shadow_arena())),
        host: child_host,
        program,
        args: child_entry_args(arity, cinst, cas),
        fuel,
        lease: None,
    }))
}

/// A §14 confined child's powerbox, built from the spawning task's own powerbox `host` (#1570,
/// #1855). The by-name grant list — `grants_n` × 16-byte `{name_off:u32, name_len:u32, handle:i32,
/// _:u32}` records in the holder's window `pm`, the tree-walker's format — re-grants each named cap
/// via [`Host::spawn_named_child`]; with no list the child gets only its starter
/// `Instantiator`/`AddressSpace` over `[0, child_size)`.
///
/// A separate-module child (`module`) serves its *own* offers and binds its own import manifest
/// (§3.6; IMPORTS.md phase 3 — a chibicc child's `write`/`read`/`exit` resolve here, or its first
/// `write` would `CapFault`). A same-module child serves over the parent's registered module and binds
/// the parent's manifest leniently, and only when the spawn handed it caps by name: a grant-less
/// child was given nothing to bind (#1234).
///
/// Returns the host and its starter handles. `Ok(None)` is a **probeable refusal**: a `required` import
/// slot with nothing to bind, which the tree-walker answers `-EINVAL` for. `Err` is a trap: an
/// unreadable record, a non-UTF-8 name, or a handle the parent may not re-grant.
fn confined_child_host(
    host: &mut Host,
    pm: Option<&Mem>,
    grants: Option<(u64, u64)>,
    child_size: u64,
    module: Option<&std::sync::Arc<Module>>,
) -> Result<Option<(Host, i32, i32)>, Trap> {
    let (mut child_host, cinst, cas) = match grants {
        Some((grants_ptr, grants_n)) => {
            let m = pm.ok_or(Trap::Malformed)?;
            let list = super::read_grant_records(grants_ptr, grants_n, |o, l| m.read_window(o, l))?;
            host.spawn_named_child(&list, child_size)
                .ok_or(Trap::CapFault)?
        }
        None => {
            let mut ch = Host::new();
            let (cinst, cas) = ch.grant_starter_caps(child_size);
            (ch, cinst, cas)
        }
    };
    let bound = match module {
        Some(cm) => {
            child_host.set_self_module(cm);
            child_host
                .bind_child_manifest(&cm.imports, &cm.types)
                .is_ok()
        }
        None => {
            child_host.set_self_module_opt(host.self_module.clone());
            let im = child_host.module_imports(super::SELF_MODULE);
            let ty = child_host.module_types(super::SELF_MODULE);
            match (grants, im, ty) {
                (Some(_), Some(im), Some(ty)) => {
                    child_host.bind_same_module_manifest(&im, &ty).is_ok()
                }
                _ => true,
            }
        }
    };
    Ok(bound.then_some((child_host, cinst, cas)))
}

/// A child's entry task in `module` (`prog`, its landed program) and its own natural dispatch table
/// over that module — no installed §22 units; #1296: sized for the install slots its re-granted
/// `Jit` carries.
fn child_task(
    module: u32,
    prog: &Compiled,
    entry: i64,
    args: &[Value],
    jit_table_log2: u8,
) -> Result<(VTask, SharedSlots), Trap> {
    let table = build_table_for(prog.progs.len(), jit_table_log2, module);
    let mut vt = VTask::new(prog, entry as usize, args)?;
    vt.active.module = module as usize;
    vt.active.home = module as usize;
    Ok((vt, table))
}

/// An admitted detached child's window before it exists: a fresh window, never a carve of the
/// spawner's, that starts with its module's data segments under the NULL guard and the spawn-time
/// payload at `module_args_base()`. The driver supplies the backing ([`build`](FreshWindow::build)).
struct FreshWindow {
    size_log2: u8,
    shadow: Option<super::ShadowArena>,
    data: std::sync::Arc<[temen_ir::Data]>,
    payload: Vec<u8>,
}

impl FreshWindow {
    /// Build the window over `back` (`None`: a reservation of the engine's own) the way the
    /// tree-walker builds it ([`Mem::detached`]), and alias in the pre-mapped region the child's
    /// powerbox `host` carries.
    fn build(
        &self,
        back: Option<std::sync::Arc<super::Region>>,
        host: &mut Host,
    ) -> Result<Mem, Trap> {
        let mut mem = Mem::detached(
            DEFAULT_RESERVED_LOG2,
            self.size_log2,
            self.shadow,
            &self.data,
            back,
        );
        if !self.payload.is_empty() {
            let _ = mem.write_bytes(temen_ir::module_args_base(), &self.payload);
        }
        if host.apply_premap(&mut mem) < 0 {
            return Err(Trap::Malformed);
        }
        Ok(mem)
    }
}

/// Admit an op-15 spawn against `host` (the parent powerbox) and build the child, with its window
/// still to be built over a backing the driver supplies. `pm` is the parent's window (the grant list
/// and the args payload are read from it); `parent_fuel` its remaining fuel. `Ok(None)` is a
/// **probeable refusal** — the driver lands `-EINVAL` and nothing was charged; `Err` is a trap (a
/// forged handle, an unreadable payload). The checks — entry shape, the window = the module's
/// declared memory, the payload fitting the args area, `premap_admit`, durability — run before the
/// `Budget.mem` take, so a refusal charges nothing.
///
/// `freezes_detached`: whether a freeze of this run captures a detached child. A durable domain
/// spawns detached only an attested-freezable module, and only while it holds freeze authority over
/// its detached progeny (#1361 step 4, #1440, #1501). The tree-walker's freeze captures the child and
/// `Vcpu` leaves the capture to its embedder, but the in-process drivers' freeze cannot, so there a
/// durable domain's detached spawn refuses (#1893).
fn admit_detached_child(
    host: &mut Host,
    pm: Option<&Mem>,
    parent_fuel: u64,
    s: DetachedSpawn,
    freezes_detached: bool,
) -> Result<Option<(AdmittedChild, FreshWindow)>, Trap> {
    let (cfuncs, cmem_log2, cdata, ctypes, cmodule, cshadow, cdurable) = {
        let g = host.resolve_module(s.module)?;
        (
            g.funcs.clone(),
            g.memory_log2,
            g.data.clone(),
            g.types.clone(),
            std::sync::Arc::clone(&g.module),
            g.shadow,
            g.durable,
        )
    };
    let compiled = compile_module(&cfuncs, &ctypes, cshadow).ok_or(Trap::Malformed)?;
    let sig = compiled.sigs.get(s.entry as usize);
    let arity = sig.map_or(0, |(p, _)| p.len());
    let ok_entry = sig.is_some_and(|(p, r)| child_entry_ok(p, r));
    let size_log2 = temen_ir::detached_size_log2(s.size_log2, cmem_log2);
    let child_size = if (0..64).contains(&size_log2) {
        1u64 << size_log2
    } else {
        0
    };
    let mod_ok = cmem_log2 == Some(size_log2 as u8);
    let payload: Vec<u8> = match s.args {
        Some((ptr, len)) => pm.ok_or(Trap::Malformed)?.read_window(ptr, len as usize)?,
        None => Vec::new(),
    };
    let payload_ok =
        payload.len() as u64 <= temen_ir::module_args_end() - temen_ir::module_args_base();
    // The op-11 record format: `{name_off u32, name_len u32, handle i32, _ u32}`, fail-closed on a
    // handle the parent may not re-grant.
    let glist = match s.grants {
        Some((gptr, gn)) => {
            let m = pm.ok_or(Trap::Malformed)?;
            super::read_grant_records(gptr, gn, |o, l| m.read_window(o, l))?
        }
        None => Vec::new(),
    };
    if !glist.iter().all(|(_, h)| host.can_regrant(*h)) {
        return Err(Trap::CapFault);
    }
    let premap_ok = match s.premap {
        Some((r, o)) => host.premap_admit(r, o, child_size)?,
        None => true,
    };
    let durable = host.is_durable();
    let durable_ok = !durable || (freezes_detached && cdurable);
    if !ok_entry
        || child_size == 0
        || !mod_ok
        || !payload_ok
        || !premap_ok
        || !durable_ok
        || match host.admit_detached_spawn(s.budget, child_size) {
            // D66 — single-spawn lane parity with the tree-walker: this engine's detached children do
            // not yet return a lane at their reap, so the lane is given straight back (#1600). The
            // window's bytes stay spent while the child lives: the driver returns them at its end
            // (`AdmittedChild::lease`).
            Some(lane) => {
                host.give_lane(lane);
                false
            }
            None => true,
        }
    {
        return Ok(None);
    }
    let mut child_host = Host::new();
    // §4: a durable domain's child is durable too, so its own spawns re-apply the rule above.
    child_host.set_durable(durable);
    child_host.set_attestation(host.detached_child_attestation());
    let reservation = 1u64 << DEFAULT_RESERVED_LOG2;
    let (cinst, cas) = child_host.grant_starter_caps(reservation);
    // #1944 — the budget that paid for the window is the child's own.
    host.give_child_budget(s.budget, &mut child_host);
    for (name, gh) in &glist {
        if let Some(cg) = host.regrant_into_child(*gh, &mut child_host) {
            child_host.register_cap_name(name, cg);
        }
    }
    // The pre-mapped region rides the child's powerbox; the window build aliases it in.
    if let Some((r, o)) = s.premap {
        if !host.stage_premap(r, o, &mut child_host) {
            return Err(Trap::Malformed);
        }
    }
    child_host.set_self_module(&cmodule);
    // A child of the running module itself binds leniently (#1234 — its manifest is the parent's
    // whole import surface, not one written for the child).
    let bound = if host.is_self_module(s.module) {
        child_host.bind_same_module_manifest(&cmodule.imports, &cmodule.types)
    } else {
        child_host.bind_child_manifest(&cmodule.imports, &cmodule.types)
    };
    if bound.is_err() {
        return Ok(None);
    }
    let args = child_entry_args(arity, cinst, cas);
    let fuel = if s.quota <= 0 {
        parent_fuel
    } else {
        (s.quota as u64).min(parent_fuel)
    };
    let child = AdmittedChild {
        mem: None,
        host: child_host,
        program: ChildProgram::Granted(compiled),
        args,
        fuel,
        lease: Some((s.budget, child_size)),
    };
    let window = FreshWindow {
        size_log2: size_log2 as u8,
        shadow: cshadow,
        data: cdata,
        payload,
    };
    Ok(Some((child, window)))
}

/// [`admit_detached_child`] for the in-process drivers, which hold the child's window themselves:
/// built at once, over a reservation of the engine's own.
fn admit_detached_in_process(
    host: &mut Host,
    pm: Option<&Mem>,
    parent_fuel: u64,
    s: DetachedSpawn,
) -> Result<Option<AdmittedChild>, Trap> {
    let Some((mut child, window)) = admit_detached_child(host, pm, parent_fuel, s, false)? else {
        return Ok(None);
    };
    child.mem = Some(window.build(None, &mut child.host)?);
    Ok(Some(child))
}

pub fn compile_module(
    funcs: &[Func],
    types: &[temen_ir::TypeEntry],
    shadow: Option<super::ShadowArena>,
) -> Option<Compiled> {
    compile_module_with(funcs, types, true, shadow)
}

/// Unfused lowering — one op per source instruction, so the step/location trace stays
/// tree-walker-identical. The debug/trace entries (`ir_trace`, `ir_window_trace`, `ir_value_trace`,
/// `debug_advance_fiber`, `dbg_pick_runnable`) use this; results and traps are identical to the fused
/// form (fusion only merges a pure compare into its sole-consumer branch).
pub fn compile_module_unfused(
    funcs: &[Func],
    types: &[temen_ir::TypeEntry],
    shadow: Option<super::ShadowArena>,
) -> Option<Compiled> {
    compile_module_with(funcs, types, false, shadow)
}

/// Lower every function, or `None` if any uses an op outside this slice's subset.
fn compile_module_with(
    funcs: &[Func],
    types: &[temen_ir::TypeEntry],
    fuse: bool,
    shadow: Option<super::ShadowArena>,
) -> Option<Compiled> {
    // Coroutines (§14, `spawn_coroutine`/`resume`/`yield`) are driven **inline** as single-vCPU
    // children with a Yielder-only powerbox. A coroutine module that *also* uses fibers or threads
    // would need the child to participate in those seams (a coroutine child can use `cont.*`/`thread.*`
    // in the tree-walker), which the inline coroutine driver here doesn't service — so reject the
    // combination (→ tree-walker fallback). §14 **executor children** (`instantiate`/`join`, ops 0/1)
    // are different: they run on the scheduler like threads, not inline — so they classify as
    // scheduler-driven, not as coroutines — and they combine with `cont.*` fibers too: every driver
    // gives each confined child domain its own fiber registry (the cooperative `ChildEnv::fibers`,
    // the debugger's `DbgEnv::fibers`, the parallel `ParDomain`), as the oracle does. Plain coroutine /
    // fiber / thread / instantiate modules are each fine, as are instantiate+thread,
    // instantiate+coroutine and instantiate+fiber.
    let s = scan_seams(funcs);
    // `gc.roots` (§GC) is per-vCPU **conservative root enumeration**: on this engine it scans the
    // calling vCPU's continuation (`vt.active` + `vt.chain` + `vt.coroutines`) **plus the run-shared
    // fiber registry** (`fibers`, scanned in [`step_vcpu`]'s `Outcome::GcRoots` arm) — the exact
    // scope the tree-walker's op documents ("the caller's own live frames, the parked root, and every
    // registry fiber's frames", `crates/temen/tests/gc_roots.rs`). Neither engine scans a *sibling
    // thread's* own frames, and neither has to: a guest GC that threads coordinates a stop-the-world
    // quiesce and has each vCPU enumerate its own roots (the reference barrier is
    // `crates/temen/tests/gc_quiesce.rs`); JACL's roots live in the migratable fibers the shared
    // registry covers. So `gc.roots` + `thread.*` is **not** vetoed — the criterion for this op is
    // soundness (`tw ⊆ bc`, GC.md §3.2), which holds because the bytecode scope is a superset of the
    // tree-walker's. (`gc.roots` + fibers / coroutines was always fine — those continuations are
    // scanned.)
    //
    // §3.6 (I36 slice 1): a **serving** module is admitted natively only when no handler could
    // park or unwind mid-dispatch ([`serve_qualifies`]) — any park-capable seam anywhere in the
    // module (futex waits / threads, fibers, coroutines, nested instantiate, setjmp/longjmp — a
    // `longjmp` out of a handler would unwind past the serve linkage — blocking stream reads,
    // spawn-bound imports, gc.roots) falls the whole module back to the tree-walk oracle, whose
    // serve arm has the fiber-park machinery (slice 5b).
    // FORK.md §9.2 — the bytecode fork-serving escape: a fork-shaped module (`bytecode_serves_fork`)
    // is admitted natively even though `svc_park_veto` folds it for Cranelift (the per-backend split).
    if (s.has_coro && (s.has_fiber || s.has_thread))
        || (s.svc_park_veto() && !s.bytecode_serves_fork())
    {
        return None;
    }

    let arities: Vec<usize> = funcs.iter().map(|f| f.results.len()).collect();
    let mut progs = Vec::with_capacity(funcs.len());
    for f in funcs {
        progs.push(compile_func(f, &arities, types, fuse)?);
    }
    let table_mask = funcs.len().next_power_of_two().max(1) - 1;
    Some(Compiled {
        shadow,
        progs,
        result_types: funcs.iter().map(|f| f.results.clone()).collect(),
        sigs: funcs
            .iter()
            .map(|f| (f.params.clone(), f.results.clone()))
            .collect(),
        table_mask,
    })
}

fn compile_func(
    f: &Func,
    arities: &[usize],
    types: &[temen_ir::TypeEntry],
    fuse: bool,
) -> Option<Program> {
    // Global slot per value: each block's params then its value-producing insts, in order.
    let mut base = Vec::with_capacity(f.blocks.len());
    let mut nslots = 0u32;
    for b in &f.blocks {
        base.push(nslots);
        nslots += b.params.len() as u32;
        for inst in &b.insts {
            nslots += inst.result_count(arities, types) as u32;
        }
    }
    let mut block_pc = vec![0u32; f.blocks.len()];
    let mut ops: Vec<Op> = Vec::new();
    // Debug reverse map (Slice 1c-3), built **incrementally** alongside `ops` (was a positional
    // post-pass) so fusion — which drops an op — keeps `src` and `ops` in lockstep: each op push is
    // paired with exactly one `src` push. Instruction ops map to their `(block, inst)`; the
    // terminator op maps to `(block, insts.len() | SRC_TERM)` (flagged so `vm_trap_bt` can tell a
    // terminator-trap site from an instruction's). A fused `BrIfCmp` takes the terminator location
    // (it can never trap), and the fused-away `IntCmp`'s entry is dropped — the fused program is
    // never single-stepped (debug/trace compile unfused), so no source location is lost there.
    let mut src: Vec<Option<(u32, u32)>> = Vec::new();
    for (bi, b) in f.blocks.iter().enumerate() {
        block_pc[bi] = ops.len() as u32;
        let g = |local: u32| base[bi] + local; // operand: block-local index -> frame slot
        let mut local = b.params.len() as u32;
        for (i, inst) in b.insts.iter().enumerate() {
            let dst = base[bi] + local;
            local += inst.result_count(arities, types) as u32;
            ops.push(compile_inst(inst, dst, base[bi], types, &g)?);
            src.push(Some((bi as u32, i as u32)));
        }
        // Terminator -> edge copies (block-local src in this block -> first slots of target) + jump.
        let edge = |bidx: usize, args: &[u32]| -> Edge {
            // Slice 5b: drop **identity** self-copies (`src == dst`) at compile time. A loop-invariant
            // block param threaded unchanged across a back-edge lands in the same global slot it came
            // from, so the copy is a no-op — eliding it removes a real `scratch` push+write per such
            // param every iteration. Safe and semantics-transparent: an `x -> x` move changes nothing,
            // and its removal can't affect the gather/scatter of the other (aliasing) copies. Applies
            // uniformly to every terminator's edges (Br/BrIf/BrIfCmp/BrTable), fused or not.
            let pairs: Box<[(u32, u32)]> = args
                .iter()
                .enumerate()
                .map(|(i, a)| (g(*a), base[bidx] + i as u32))
                .filter(|(src, dst)| src != dst)
                .collect();
            (Copies::new(pairs), bidx as u32) // block index; patched to entry pc below
        };
        match &b.term {
            Terminator::Br { target, args } => {
                let (copies, t) = edge(*target as usize, args);
                ops.push(Op::Br { copies, target: t });
            }
            Terminator::BrIf {
                cond,
                then_blk,
                then_args,
                else_blk,
                else_args,
            } => {
                let (then_copies, tt) = edge(*then_blk as usize, then_args);
                let (else_copies, et) = edge(*else_blk as usize, else_args);
                let cond_slot = g(*cond);
                // Slice 5a: fuse a block-final `IntCmp` whose result is this branch's condition and
                // is used nowhere else into a single `BrIfCmp`. Valid because the compare is pure and
                // single-use *here*: it is the last instruction (so no later in-block reader) and the
                // cond slot is not carried to any successor (not an edge-copy source). Fused compile
                // only — `fuse == false` (debug/trace) keeps the compare as its own steppable op.
                let fused = if fuse && !b.insts.is_empty() {
                    match ops.last() {
                        Some(Op::IntCmp {
                            dst,
                            a,
                            b: cb,
                            ty,
                            op,
                        }) if *dst == cond_slot
                            && !then_copies.pairs.iter().any(|(s, _)| *s == cond_slot)
                            && !else_copies.pairs.iter().any(|(s, _)| *s == cond_slot) =>
                        {
                            Some((*a, *cb, *ty, *op))
                        }
                        _ => None,
                    }
                } else {
                    None
                };
                if let Some((a, cb, ty, op)) = fused {
                    ops.pop(); // drop the now-fused IntCmp op ...
                    src.pop(); // ... and its source-map entry (kept in lockstep)
                    ops.push(Op::BrIfCmp {
                        a,
                        b: cb,
                        ty,
                        op,
                        then_copies,
                        then_pc: tt,
                        else_copies,
                        else_pc: et,
                    });
                } else {
                    ops.push(Op::BrIf {
                        cond: cond_slot,
                        then_copies,
                        then_pc: tt,
                        else_copies,
                        else_pc: et,
                    });
                }
            }
            Terminator::BrTable {
                idx,
                targets,
                default,
            } => {
                let arms = targets.iter().map(|(t, a)| edge(*t as usize, a)).collect();
                let default = edge(default.0 as usize, &default.1);
                ops.push(Op::BrTable {
                    idx: g(*idx),
                    arms,
                    default,
                });
            }
            Terminator::Return(vs) => ops.push(Op::Ret {
                srcs: vs.iter().map(|v| g(*v)).collect(),
            }),
            Terminator::Unreachable => ops.push(Op::Unreachable),
            // Tail calls reuse the current activation window (no stack growth): a direct tail call
            // stays in the caller's module; an indirect one dispatches through the runtime table.
            Terminator::ReturnCall { func, args } => ops.push(Op::TailCall {
                callee: *func,
                args: args.iter().map(|a| g(*a)).collect(),
            }),
            Terminator::ReturnCallIndirect { ty, idx, args } => ops.push(Op::TailCallIndirect {
                idx: g(*idx),
                args: args.iter().map(|a| g(*a)).collect(),
                want_params: super::call_sig(types, *ty).params.clone().into(),
                want_results: super::call_sig(types, *ty).results.clone().into(),
            }),
        }
        // Exactly one terminator op was pushed above (fused `BrIfCmp` or a plain terminator); pair it
        // with the terminator source entry so `src` stays the same length as `ops`.
        src.push(Some((bi as u32, b.insts.len() as u32 | SRC_TERM)));
    }
    debug_assert_eq!(
        ops.len(),
        src.len(),
        "src map must stay in lockstep with ops"
    );

    // Patch branch targets from block index to entry pc.
    let patch = |t: &mut u32| *t = block_pc[*t as usize];
    for op in &mut ops {
        match op {
            Op::Br { target, .. } => patch(target),
            Op::BrIf {
                then_pc, else_pc, ..
            } => {
                patch(then_pc);
                patch(else_pc);
            }
            Op::BrIfCmp {
                then_pc, else_pc, ..
            } => {
                patch(then_pc);
                patch(else_pc);
            }
            Op::BrTable { arms, default, .. } => {
                for (_, t) in arms.iter_mut() {
                    patch(t);
                }
                patch(&mut default.1);
            }
            _ => {}
        }
    }
    Some(Program {
        ops,
        nslots,
        src: src.into_boxed_slice(),
    })
}

fn compile_inst(
    inst: &Inst,
    dst: u32,
    block_base: u32,
    types: &[temen_ir::TypeEntry],
    g: &impl Fn(u32) -> u32,
) -> Option<Op> {
    Some(match inst {
        Inst::ConstI32(c) => Op::Const {
            dst,
            val: Reg::from_i32(*c),
        },
        Inst::ConstI64(c) => Op::Const {
            dst,
            val: Reg::from_i64(*c),
        },
        Inst::ConstF32(b) => Op::Const {
            dst,
            val: Reg::from_f32(f32::from_bits(*b)),
        },
        Inst::ConstF64(b) => Op::Const {
            dst,
            val: Reg::from_f64(f64::from_bits(*b)),
        },
        Inst::IntBin { ty, op, a, b } => Op::IntBin {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::IntCmp { ty, op, a, b } => Op::IntCmp {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::IntUn { ty, op, a } => Op::IntUn {
            dst,
            a: g(*a),
            ty: *ty,
            op: *op,
        },
        Inst::Eqz { ty, a } => Op::Eqz {
            dst,
            a: g(*a),
            ty: *ty,
        },
        Inst::Convert { op, a } => Op::Convert {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::Select { cond, a, b } => Op::Select {
            dst,
            cond: g(*cond),
            a: g(*a),
            b: g(*b),
        },
        Inst::FBin { ty, op, a, b } => Op::FBin {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::FUn { ty, op, a } => Op::FUn {
            dst,
            a: g(*a),
            ty: *ty,
            op: *op,
        },
        Inst::FCmp { ty, op, a, b } => Op::FCmp {
            dst,
            a: g(*a),
            b: g(*b),
            ty: *ty,
            op: *op,
        },
        Inst::FToISat { op, a } => Op::FToISat {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::FToITrap { op, a } => Op::FToITrap {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::IToFConv { op, a } => Op::IToFConv {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::Cast { op, a } => Op::Cast {
            dst,
            a: g(*a),
            op: *op,
        },
        Inst::RefFunc { func } => Op::RefFunc { dst, func: *func },
        Inst::Load {
            op, addr, offset, ..
        } => Op::Load {
            dst,
            addr: g(*addr),
            op: *op,
            offset: *offset,
        },
        Inst::Store {
            op,
            addr,
            value,
            offset,
            ..
        } => Op::Store {
            addr: g(*addr),
            value: g(*value),
            op: *op,
            offset: *offset,
        },
        Inst::MemCopy { dst, src, len } => Op::MemCopy {
            dst: g(*dst),
            src: g(*src),
            len: g(*len),
        },
        Inst::MemMove { dst, src, len } => Op::MemMove {
            dst: g(*dst),
            src: g(*src),
            len: g(*len),
        },
        Inst::MemFill { dst, val, len } => Op::MemFill {
            dst: g(*dst),
            val: g(*val),
            len: g(*len),
        },
        Inst::AtomicLoad {
            ty, addr, offset, ..
        } => Op::AtomicLoad {
            dst,
            addr: g(*addr),
            ty: *ty,
            offset: *offset,
        },
        Inst::AtomicStore {
            ty,
            addr,
            value,
            offset,
            ..
        } => Op::AtomicStore {
            addr: g(*addr),
            value: g(*value),
            ty: *ty,
            offset: *offset,
        },
        Inst::AtomicRmw {
            ty,
            op,
            addr,
            value,
            offset,
            ..
        } => Op::AtomicRmw {
            dst,
            addr: g(*addr),
            value: g(*value),
            ty: *ty,
            op: *op,
            offset: *offset,
        },
        Inst::AtomicCmpxchg {
            ty,
            addr,
            expected,
            replacement,
            offset,
            ..
        } => Op::AtomicCmpxchg {
            dst,
            addr: g(*addr),
            expected: g(*expected),
            replacement: g(*replacement),
            ty: *ty,
            offset: *offset,
        },
        Inst::Call { func, args } => Op::Call {
            callee: *func,
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
        },
        // `call.dyn` through module 0's natural table — self-contained (no install/invoke),
        // so the compile-time signature table resolves it. Cross-module units (install/invoke) are
        // still a later slice; here every reachable slot is a module-0 function.
        Inst::CallIndirect { ty, idx, args } => Op::CallIndirect {
            idx: g(*idx),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            want_params: super::call_sig(types, *ty).params.clone().into(),
            want_results: super::call_sig(types, *ty).results.clone().into(),
        },
        // Synchronous capability call: the generic powerbox path (guest suspended, host computes,
        // same activation continues) is driven here via `host.cap_dispatch_slots`. The
        // executor/fiber capability variants — `Instantiator` (child vCPUs), `Yielder` (co-fiber
        // yield), `JIT` (install/uninstall/invoke), and `SharedRegion` op 4 (`grant` into a child) —
        // need seams a later slice drives, so reject those (fall back to the tree-walker). These are
        // exactly the `type_id`/`op` combinations `run_inner` matches in dedicated arms ahead of its
        // generic `CapCall` arm.
        Inst::CapCall {
            type_id,
            op,
            sig,
            handle,
            args,
        } => {
            use super::cap_id;
            match (*type_id, *op) {
                // §14 executor children — instantiate (op 0) spawns a confined child on the scheduler;
                // join (op 1) parks until it finishes, reusing the §12 thread join machinery (children
                // share the `threads` handle namespace). The separate-module / demand variants (5/6/7
                // and op 4) and the JIT / SharedRegion-grant variants need seams this slice doesn't
                // drive: reject (fall back).
                (cap_id::INSTANTIATOR, 0) if args.len() >= 4 => Op::Instantiate {
                    handle: g(*handle),
                    entry: g(args[0]),
                    off: g(args[1]),
                    size_log2: g(args[2]),
                    quota: g(args[3]),
                    dst,
                    grants: None,
                },
                (cap_id::INSTANTIATOR, 1) if !args.is_empty() => Op::InstJoin {
                    handle: g(*handle),
                    child: g(args[0]),
                    dst,
                },
                // op 5 = instantiate_module: the first arg is the granted `Module` handle; the carve
                // args (entry/off/size_log2/quota) follow. (join, op 1, serves both kinds.)
                (cap_id::INSTANTIATOR, 5) if args.len() >= 5 => Op::InstantiateModule {
                    handle: g(*handle),
                    module: g(args[0]),
                    entry: g(args[1]),
                    off: g(args[2]),
                    size_log2: g(args[3]),
                    quota: g(args[4]),
                    dst,
                    grants: None,
                },
                // op 13 = instantiate_module_named: op 5 + a by-name grant list. Args:
                // (module, grants_ptr, grants_n, entry, off, size_log2, quota). The driver reads the
                // grant records from the parent window and re-grants each cap into the child powerbox.
                (cap_id::INSTANTIATOR, 13) if args.len() >= 7 => Op::InstantiateModule {
                    handle: g(*handle),
                    module: g(args[0]),
                    entry: g(args[3]),
                    off: g(args[4]),
                    size_log2: g(args[5]),
                    quota: g(args[6]),
                    dst,
                    grants: Some((g(args[1]), g(args[2]))),
                },
                // op 15 = instantiate_detached (#1286): (budget, module, grants_ptr, grants_n, entry,
                // size_log2, quota[, args_ptr, args_len]) — the fresh-window spawn; the driver mints the
                // window and seeds the payload. A `grants_n` of 0 is the grant-less form.
                (cap_id::INSTANTIATOR, 15) if args.len() >= 7 => Op::InstantiateDetached {
                    handle: g(*handle),
                    budget: g(args[0]),
                    module: g(args[1]),
                    grants: Some((g(args[2]), g(args[3]))),
                    entry: g(args[4]),
                    size_log2: g(args[5]),
                    quota: g(args[6]),
                    args: (args.len() >= 9).then(|| (g(args[7]), g(args[8]))),
                    premap: (args.len() >= 11).then(|| (g(args[9]), g(args[10]))),
                    dst,
                },
                // CONSOLIDATION.md §3d — instantiate_rec (op 17): the record pointer is the one
                // arg; every other spawn parameter is data in the record, read at exec time.
                (cap_id::INSTANTIATOR, 17) if !args.is_empty() => Op::InstantiateRec {
                    handle: g(*handle),
                    rec: g(args[0]),
                    dst,
                },
                // §3.6 (I36 slice 2) — child_offer (op 14): mint a live-callee offer over a running
                // child's export. The mint needs the child's live env, so the op surfaces to the
                // driver; the compile only marshals `(child, export)`.
                (cap_id::INSTANTIATOR, 14) if args.len() >= 2 => Op::ChildOffer {
                    handle: g(*handle),
                    child: g(args[0]),
                    export: g(args[1]),
                    dst,
                },
                // §22 guest-driven JIT units: install/uninstall drive the dispatch table; compile /
                // compile_linked (ops 0/5) are pure host ops, so they fall through to the generic
                // dispatch below. `invoke` (op 1) is the next slice — reject it for now (fall back).
                (cap_id::JIT, 3) if !args.is_empty() => Op::JitInstall {
                    handle: g(*handle),
                    code: g(args[0]),
                    dst,
                },
                (cap_id::JIT, 4) if !args.is_empty() => Op::JitUninstall {
                    handle: g(*handle),
                    slot: g(args[0]),
                    dst,
                },
                (cap_id::JIT, 1) if !args.is_empty() => Op::JitInvoke {
                    handle: g(*handle),
                    code: g(args[0]),
                    args: args[1..].iter().map(|a| g(*a)).collect(),
                    dst,
                    // The call.cap sig is `(i64 code, params…) -> (results…)`; the unit entry's
                    // params are super::call_sig(types, *sig).params without the leading code-handle.
                    params: super::call_sig(types, *sig)
                        .params
                        .get(1..)
                        .unwrap_or(&[])
                        .to_vec()
                        .into(),
                    results: super::call_sig(types, *sig).results.clone().into(),
                },
                (cap_id::INSTANTIATOR, _) => return None,
                (cap_id::SHARED_REGION, 4) => return None,
                // §3.6 service points (I36 slice 1): `svc.poll` with the canonical one-result
                // shape compiles to the native serve-loop-core op — the module-level veto in
                // [`compile_module`] guarantees its handlers cannot park mid-dispatch, so the
                // rewind linkage runs each one to completion (or trap). `svc.wait`'s
                // empty-queue park needs a waker topology (cross-domain callers, timers) the
                // cooperative scheduler doesn't host yet, and a no-result `svc.poll` would
                // leave the op without its result-slot scratch — both still decline, falling
                // the whole module back to the tree-walk oracle, which serves.
                // (The timed `svc.wait` form — op 10 with the optional timeout arg — is
                // oracle-only and declines below; `serve_qualifies` already vetoed the module.)
                (temen_ir::CAP_SELF_TYPE_ID, op @ (9 | 10))
                    if super::call_sig(types, *sig).results.len() == 1 && args.is_empty() =>
                {
                    Op::SvcPoll {
                        dst,
                        wait: op == 10,
                    }
                }
                (temen_ir::CAP_SELF_TYPE_ID, 9 | 10) => return None,
                // FORK.md §9.2 — `clone_caller` (op 11) / `reap` (op 12): compiled to the native fork
                // ops (the module reached here only via the [`Seams::bytecode_serves_fork`] escape).
                // The reply/pid args are register operands, resolved in the driver.
                (temen_ir::CAP_SELF_TYPE_ID, 11) => Op::CloneCaller {
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                    has_result: !super::call_sig(types, *sig).results.is_empty(),
                },
                (temen_ir::CAP_SELF_TYPE_ID, 12) => Op::Reap {
                    pid: args.first().map(|a| g(*a)),
                    dst,
                    has_result: !super::call_sig(types, *sig).results.is_empty(),
                },
                // CALLS.md §10.6 — `fuel.remaining` (op 13) reads the vCPU's live fuel counter, which
                // the host-side `cap_dispatch_slots` can't see; rather than add a native bytecode op,
                // decline the module so it falls back to the tree-walker, which services op 13
                // directly. (The JIT does lower it inline — it owns the fuel cell's address.)
                (temen_ir::CAP_SELF_TYPE_ID, 13) => return None,
                // FORK.md §8.6 — `exec_module` (`execve` image-replace, op 14, #1080): the cooperative
                // driver owns the task/env set + `dom.source`, so the op resolves its five register
                // operands `(module, grants_ptr, grants_n, entry, size_log2)` here and surfaces the
                // image-replace to the driver ([`Outcome::Exec`]). The `dst` result slot receives the
                // `-EINVAL` of a refused exec (a successful exec never returns to this activation).
                // (The tree-walker folds this to `Step::Exec`; this tier does the replace natively so
                // an exec-bearing module — bash — runs on the browser's bytecode engine.)
                (temen_ir::CAP_SELF_TYPE_ID, 14) if args.len() >= 5 => Op::ExecModule {
                    module: g(args[0]),
                    grants_ptr: g(args[1]),
                    grants_n: g(args[2]),
                    entry: g(args[3]),
                    size_log2: g(args[4]),
                    dst,
                },
                // A malformed `exec_module` (< 5 args) is outside the ABI — decline the module so it
                // folds to the tree-walker rather than mis-lowering.
                (temen_ir::CAP_SELF_TYPE_ID, 14) => return None,
                // Generic synchronous powerbox dispatch (Stream/Clock/Memory/host-fn/JIT compile/…).
                _ => Op::CapCall {
                    type_id: *type_id,
                    op: *op,
                    handle: g(*handle),
                    params: super::call_sig(types, *sig).params.clone().into(),
                    args: args.iter().map(|a| g(*a)).collect(),
                    dst,
                    results: super::call_sig(types, *sig).results.clone().into(),
                },
            }
        }
        // §7/§6 reflection `self.count`/`get`/`resolve`/`label`/`attest` reach here as their
        // `call.cap CAP_SELF op 0/1/2/3/4` form and compile via the generic `Op::CapCall` fallthrough
        // in the `Inst::CapCall` arm above.
        // §12 fibers — cooperative continuation switching, driven by the bytecode driver (no M:N
        // pool, no DPOR; single-vCPU). `cont.new` registers a pending fiber, `cont.resume` switches
        // in (two results), `suspend` switches back (one result).
        Inst::ContNew { func, sp } => Op::ContNew {
            func: g(*func),
            sp: g(*sp),
            dst,
        },
        // I48 — the `block: true` form idles the resumer's task on the fiber's event (see the
        // cooperative driver's `Outcome::ContResume` / fiber-park arms + `TaskState::BlockedOnFiber`)
        // instead of spinning the poll. Advisory still holds: `FIBER_PARKED` remains a legal
        // transient (the guest keeps its loop), so the deterministic explorer and any non-idling
        // path stay conforming (invariant 9).
        Inst::ContResume { k, arg, block } => Op::ContResume {
            k: g(*k),
            arg: g(*arg),
            dst,
            blocking: *block,
        },
        Inst::Suspend { value } => Op::Suspend {
            value: g(*value),
            dst,
        },
        // `<setjmp.h>` non-local jump — intra-vCPU (no scheduler escape). `setjmp` checkpoints the
        // activation's resume point (the flat per-function register layout keeps each block's slots
        // distinct, so the `setjmp` block's values survive a deeper call — no window snapshot needed,
        // unlike the tree-walker's per-block `vals`); `longjmp` pops the activation stack back to it.
        Inst::SetJmp { buf } => Op::SetJmp { buf: g(*buf), dst },
        Inst::LongJmp { buf, val } => Op::LongJmp {
            buf: g(*buf),
            val: g(*val),
        },
        // §12 threads / futex — multi-vCPU, serviced by the driver (the cooperative `drive`, the
        // parallel one, or an external `Vcpu` host). Threads and fibers mix: every driver runs a run's
        // vCPUs over one fiber registry (#1761), so a fiber migrates between them.
        Inst::ThreadSpawn { func, sp, arg } => Op::ThreadSpawn {
            func: *func,
            sp: g(*sp),
            arg: g(*arg),
            dst,
        },
        Inst::ThreadJoin { handle } => Op::ThreadJoin {
            handle: g(*handle),
            dst,
        },
        Inst::MemoryWait {
            ty,
            addr,
            expected,
            timeout,
        } => Op::MemoryWait {
            ty: *ty,
            addr: g(*addr),
            expected: g(*expected),
            timeout: g(*timeout),
            dst,
        },
        Inst::MemoryNotify { addr, count } => Op::MemoryNotify {
            addr: g(*addr),
            count: g(*count),
            dst,
        },
        // Cross-module / GC ops this slice doesn't drive (dispatch table / root scan) — fall back.
        // §GC conservative root enumeration — driven by the scheduler (it scans the whole vCPU
        // continuation). `call.import` must already be resolved to a `call.cap`, so it never reaches
        // a backend (a leftover is a fall-back).
        Inst::GcRoots {
            heap_lo,
            heap_hi,
            mask,
            buf,
            cap,
        } => Op::GcRoots {
            lo: g(*heap_lo),
            hi: g(*heap_hi),
            mask: g(*mask),
            buf: g(*buf),
            cap: g(*cap),
            dst,
        },
        // §7 executable named import (IMPORTS.md phase 1): lower to the **generic** cap dispatch
        // with the reserved [`temen_ir::CAP_IMPORT_TYPE_ID`] and the import index as the op — the
        // host's dispatch translates it through the instantiation-time binding table, exactly as
        // the tree-walker and the JIT thunk do (one shared implementation, three backends in
        // lockstep). No handle operand since v8 (the binding carries the granted handle); the
        // dispatch never read one, so the register is simply absent.
        Inst::CallImport {
            import,
            op,
            sig,
            args,
        } => Op::CapCall {
            type_id: temen_ir::CAP_IMPORT_TYPE_ID,
            // §3.5: the reserved import dispatch packs `(slot | consumer_op << 16)`.
            op: *import | (*op << 16),
            handle: u32::MAX, // no operand (v8); the exec passes 0, the dispatch ignores it
            params: super::call_sig(types, *sig).params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: super::call_sig(types, *sig).results.clone().into(),
        },
        // §7/§22 symbolic call: when bound at instantiation it is a flat import dispatch
        // (op 0); the legacy handle operand is a live register the dispatch ignores.
        Inst::CallSym {
            import, sig, args, ..
        } => Op::CapCall {
            type_id: temen_ir::CAP_IMPORT_TYPE_ID,
            op: *import,
            handle: u32::MAX,
            params: super::call_sig(types, *sig).params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: super::call_sig(types, *sig).results.clone().into(),
        },
        // §3.5 dynamic-mode dispatch by type-section reference: the reserved dyn entry packs
        // `(type_idx | op << 16)`; the handle register is live.
        Inst::CallImportDyn {
            ty,
            op,
            sig,
            handle,
            args,
        } => Op::CapCall {
            type_id: temen_ir::CAP_DYN_TYPE_ID,
            op: *ty | (*op << 16),
            handle: g(*handle),
            params: super::call_sig(types, *sig).params.clone().into(),
            args: args.iter().map(|a| g(*a)).collect(),
            dst,
            results: super::call_sig(types, *sig).results.clone().into(),
        },
        // §3.5 self-namespace extensions (see `Op::CapSelfExt`).
        Inst::ExportHandle { export } => Op::CapSelfExt {
            op: 8 | (*export << 8),
            handle: None,
            dst,
        },
        Inst::CapSelfTypeId { ty } => Op::CapSelfExt {
            op: 6 | (*ty << 8),
            handle: None,
            dst,
        },
        Inst::CapSelfCovers { handle, ty } => Op::CapSelfExt {
            op: 7 | (*ty << 8),
            handle: Some(g(*handle)),
            dst,
        },
        // Phase-2 `import.attach` (IMPORTS.md): the attach sentinel with the handle value as the
        // one argument — the same shared host entry as the tree-walker and the JIT.
        Inst::ImportAttach { import, handle } => Op::CapCall {
            type_id: temen_ir::CAP_IMPORT_ATTACH_TYPE_ID,
            op: *import,
            handle: g(*handle),
            params: [].into(),
            args: [g(*handle)].into(),
            dst,
            results: [ValType::I32].into(),
        },
        // §12.8 4A.5: serviced from the running `Vm`'s region base (the reference `eval_inst` has no
        // context), so it gets a dedicated op rather than the `Eval` fallback.
        Inst::DurableShadowBase => Op::DurableShadowBase { dst },
        // §12 per-vCPU TLS register: serviced from the running `Vm`'s `tls` word (the reference
        // `eval_inst` traps on it — no vCPU context), so it gets a dedicated op rather than `Eval`.
        Inst::VcpuTlsGet => Op::VcpuTlsGet { dst },
        Inst::VcpuTlsSet { val } => Op::VcpuTlsSet { val: g(*val) },
        // Everything else is a pure value op or a no-result store that the reference `eval_inst`
        // already implements (the SIMD/`v128`/fence long tail): delegate to it against this block's
        // sub-window, reusing the exact semantics rather than re-inlining ~30 lane ops.
        other => Op::Eval {
            inst: Box::new(other.clone()),
            block_base,
            dst,
        },
    })
}

/// Build the linear-memory window from `m`'s memory declaration + data segments, exactly like
/// [`crate::run`] (a module with no memory yields `None`).
/// `m`'s window: `init_mem` seeded at offset 0 (the §3e args/env blob a host places at
/// `module_args_base()`; empty for none), then `m`'s data segments over it, then the NULL guard.
fn build_mem(m: &Module, init_mem: &[u8]) -> Option<Mem> {
    m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2, mc.shadow);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        mm
    })
}

/// Compile `m`'s function `func` and run it on the bytecode engine, or `None` if it (or any
/// function it can reach by direct call) uses an op outside this slice's subset. Builds a fresh
/// linear-memory window from `m`'s memory declaration + data segments, exactly like
/// [`crate::run`]. Returns typed result `Value`s. The equality harness compares this to `run`.
pub fn compile_and_run(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
) -> Option<Result<Vec<Value>, Trap>> {
    // No capabilities granted: an empty powerbox (any `call.cap` is inert → `CapFault`), exactly
    // like [`crate::run`], so this stays a faithful mirror for the equality harness.
    let mut host = Host::new();
    compile_and_run_with_host(m, func, args, fuel, &mut host)
}

/// Host-carrying [`compile_and_run`]: the powerbox is live, so synchronous capability calls
/// (`call.cap` through the generic dispatch) execute against it. `None` if the module uses an op
/// outside this slice's subset (including the executor/fiber capability variants) — the caller
/// (`crate::run_with_host_fast`) then falls back to the tree-walker.
pub fn compile_and_run_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Option<Result<Vec<Value>, Trap>> {
    compile_and_run_seeded_with_host(m, func, args, fuel, &[], host)
}

/// [`compile_and_run_with_host`] over a window whose low bytes are first seeded with `init_mem` — the
/// §3e args/env blob at `temen_ir::module_args_base()`, laid out as `temen-run`'s `RunConfig` seeds it
/// for the other tiers. Unlike the durable [`compile_and_run_capture_reserved_with_host`] seam, this
/// drives everything the plain run does (`thread.*` included), so a threaded on-ramp guest can be given
/// an environment.
pub fn compile_and_run_seeded_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    host: &mut Host,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    // #799/#1080 — install the personality fork/waitpid park-request door (the cooperative driver
    // services `ParkEvent::ForkSelf`/`TaskExit` after a `call.cap`); without it the op answers `-ENOSYS`.
    host.wire_park_door();
    // Size the dispatch table to the granted `Jit` table reservation (matching the tree-walker's
    // `DomainTable::new(funcs, jit_table_log2)`), so guest-driven `install` returns the same slots.
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = build_mem(m, init_mem);
    super::LAST_CAPTURE_FAULT.with(|c| *c.borrow_mut() = None);
    let r = run(dom, func, args, fuel, &mut mem, host);
    // #1714: the faulting address of a `MemoryFault`, in the same per-run slot the tree-walker's
    // run funnel fills (`last_capture_fault_addr`) — this path dropped the window with it, so an
    // embedder of a plain run could not say *where* a segfault was. Cleared on any other outcome,
    // so a later clean run never reports an earlier run's fault. The scheduler records the
    // trap-origin task's address as it traps (#1720: a joined child's, not the joiner's window).
    let origin = super::last_capture_fault_addr();
    let fault = match &r {
        Err(Trap::MemoryFault) => origin.or_else(|| mem.as_ref().and_then(|m| m.peek_fault_rel())),
        _ => None,
    };
    super::LAST_CAPTURE_FAULT.with(|c| *c.borrow_mut() = fault);
    Some(r)
}

/// What [`compile_and_run_with_host_traced`] returns — the shared traced-run shape (result + trap-time
/// backtrace + trapping fiber). The single-step path is root-only, so its fiber is `-1` (a trap) or
/// `None` (clean); a fibered run is a seam it declines, so the tree-walker reports the real handle.
pub type TracedRun = super::TracedRun;

/// Trap-time-backtrace counterpart of [`compile_and_run_with_host`] — the bytecode mirror of the
/// tree-walker's [`crate::run_with_host_traced`]. Drives the entry **one op at a time** (the proven
/// single-vCPU debug seam, as [`ir_trace`] does — `budget = 1` is bit-identical to run-to-completion,
/// INTERP_PERF.md Slice 1c-2) so that on a trap the `Vm`'s reified continuation still points at the
/// faulting op (the `Err` path never writes the cursor back) and its caller windows are intact; the
/// backtrace is then read off that continuation by [`vm_trap_bt`] — the flat-window analogue of the
/// tree-walker snapshotting `v.frames`. Returns `(result, backtrace)` (innermost frame first, as
/// [`crate::IrPc`]s; empty on a clean finish), resolvable to source with [`crate::source_loc`].
///
/// `None` (caller falls back to [`crate::run_with_host_traced`]) when the module is outside the
/// engine's subset, **or** when a step reaches a concurrency/coroutine seam — backtraces are
/// single-vCPU, seam-free scope (DEBUGGING.md S4), exactly like [`ir_trace`]. Single-stepping is a
/// cold diagnostic path, so the per-op suspend/resume overhead never touches the production
/// `run_fast` loop.
pub fn compile_and_run_with_host_traced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    host: &mut Host,
) -> Option<TracedRun> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new(), None));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = build_mem(m, &[]);
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Err(e), Vec::new(), None)),
    };
    loop {
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut *host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((Ok(vals), Vec::new(), None)),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope (fall back to tree-walker)
            Err(t) => {
                let bt = vm_trap_bt(&vm, &dom.source, &t);
                // This single-step path only ever drives the **root** (a fiber/thread op is a seam →
                // the `Ok(_)` arm above bails to the tree-walker), so a trap here is always the root —
                // attributed `-1`, matching the JIT's root-trap convention.
                return Some((Err(t), bt, Some(-1)));
            }
        }
    }
}

/// The trap-time backtrace of a `Vm` paused (by an `Err` from [`Vm::resume`]) on a faulting op:
/// the [`crate::IrPc`] of every live activation, **innermost frame first** — the flat-window analogue
/// of the tree-walker's [`crate::frames_to_pcs`] over `Vec<Frame>`. The cursor (`module`/`cur`/`pc`)
/// is the trapping op (the `Err` path leaves it as the prior op-boundary persisted it).
///
/// **Cursor-advance parity with the tree-walker** (`run_inner`): the tree-walker charges fuel, then
/// does `inst += 1`, then evaluates the op — so the live frame's recorded `inst` is one *past* the op
/// for any trap raised in evaluation (memory fault, div-by-zero, malformed, …), but the op *itself*
/// for an [`Trap::OutOfFuel`] (caught before the advance). The bytecode loop instead leaves `pc` on
/// the trapping op for *both*, so to report identical `IrPc`s we add `1` to the innermost frame's
/// `inst` unless the trap is `OutOfFuel`. Every suspended caller in `stack` already resumes at
/// `call_pc + 1` (the tree-walker likewise advances a caller's `inst` past the call before
/// descending), so its call op sits at `resume_pc - 1` and we report `inst + 1` for it.
fn vm_trap_bt(vm: &Vm, source: &ModuleSource, trap: &Trap) -> Vec<super::IrPc> {
    let mut bt = Vec::new();
    let Some(c) = source.get(vm.module) else {
        return bt;
    };
    if let Some((block, inst)) = c.progs[vm.cur].src.get(vm.pc).copied().flatten() {
        // An instruction's recorded `inst` advances past the op exactly when the tree-walker's did
        // (it does `inst += 1` before evaluating, so every trap but `OutOfFuel` lands one past); a
        // terminator (`unreachable`, `return_call.dyn`) is already stored as `insts.len()`, the
        // exact `inst` the tree-walker's frame carries there, and gets no bump.
        let inst = if inst & SRC_TERM != 0 {
            (inst & !SRC_TERM) as usize
        } else {
            inst as usize + !matches!(trap, Trap::OutOfFuel) as usize
        };
        bt.push(super::IrPc {
            module: vm.module as u32,
            func: vm.cur as FuncIdx,
            block: block as usize,
            inst,
        });
    }
    // Each suspended caller resumes at `call_pc + 1` (a call is an instruction, never a terminator),
    // so its call op sits at `resume_pc - 1`; report `inst + 1`, mirroring the tree-walker advancing a
    // caller's `inst` past the call before descending.
    for &(module, prog, _base, resume_pc, _ret) in vm.stack.iter().rev() {
        let call_pc = resume_pc.wrapping_sub(1);
        let Some(cm) = source.get(module) else {
            continue;
        };
        if let Some((block, inst)) = cm.progs[prog].src.get(call_pc).copied().flatten() {
            bt.push(super::IrPc {
                module: module as u32,
                func: prog as FuncIdx,
                block: block as usize,
                inst: (inst & !SRC_TERM) as usize + 1,
            });
        }
    }
    bt
}

/// A run result paired with the final window snapshot (the low `init_mem.len()` bytes).
pub type Capture = (Result<Vec<Value>, Trap>, Vec<u8>);

/// Like [`compile_and_run`], but **seeds** the window with `init_mem` first and returns the final
/// window snapshot (the low `init_mem.len()` bytes) alongside the result — the bytecode mirror of
/// [`crate::run_capture_reserved`]. Used by `bytecode_gc_roots.rs` to read back the roots buffer for
/// the §GC soundness check. `None` if the module is outside the engine's subset.
pub fn compile_and_run_capture(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(DEFAULT_RESERVED_LOG2, mc.size_log2, mc.shadow);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

/// Like [`compile_and_run_capture`], but the guest window is backed by a **caller-provided**
/// [`Region`] (a `Region::shared` over host memory) rather than an engine-`mmap`ped one — the
/// substrate→engine bridge for the parallel-wasm backend (THREADS.md step 3). On wasm `back` spans the
/// host's shared linear memory, so the root vCPU here and the per-vCPU Workers a later step spawns all
/// execute over **one shared window**. Today still cooperative (the existing `drive`); only the
/// backing changes from owned to borrowed — so a guest's result + final image are identical to
/// [`compile_and_run_capture`], and its memory effects land in the caller's buffer. (The crate stays
/// `#![forbid(unsafe_code)]`: the `unsafe` of borrowing host memory is in the embedder's
/// `Region::shared` call that built `back`.)
pub fn compile_and_run_capture_over(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(
            DEFAULT_RESERVED_LOG2,
            mc.size_log2,
            std::sync::Arc::clone(&back),
            mc.shadow,
        );
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, &mut host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

/// Run `func(args)` over the caller-provided shared window `back` against a caller-prepared `host`,
/// returning the typed results (`None` if the module is outside the engine's subset). Unlike
/// [`compile_and_run_capture_over`] this carries a live `host` (so `call.cap`s execute) and — when
/// `seed_data` is `false` — it does **not** re-seed or re-apply the module's data segments: the window
/// in `back` is already live, so re-initialising would clobber the guest's globals/heap.
///
/// This is the browser wasm-JIT **reactor** cross-tier seam: the emitted `tick` (run by the host over
/// this same window) bounces a call to a non-emitted function here — the callee runs on the
/// interpreter over the shared window, its memory effects landing in the bytes the emitted code reads.
/// Pass `seed_data = true` exactly once, for the initial `_start`, to data-initialise the window before
/// the first frame; every per-frame cross-tier callee passes `false`.
pub fn compile_and_run_over_shared_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    back: std::sync::Arc<super::Region>,
    host: &mut Host,
    seed_data: bool,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm =
            Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, mc.size_log2, back, mc.shadow);
        if seed_data {
            mm.init_data(&m.data);
            mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        }
        mm
    });
    Some(run(dom, func, args, fuel, &mut mem, host))
}

/// A module compiled **once** for repeated runs over a caller-provided shared window — the cached form
/// of [`compile_and_run_over_shared_with_host`]. The browser wasm-JIT reactor bounces a handful of
/// interpreter helpers per frame through `env.call_interp`; recompiling the whole module on every bounce
/// (as the one-shot does) dominates the frame — for Doom, ~6 ms × 3 calls ≈ 19 ms of a 20 ms frame. This
/// holds the compiled source (a cheap `Arc` clone seeds each run's throwaway [`Domain`]) so a cross-tier
/// run is just build-window + interpret, like [`Reactor`] but over the caller's shared window.
pub struct SharedProgram {
    source: std::sync::Arc<ModuleSource>,
    n_funcs: usize,
    mem_size_log2: Option<u8>,
    /// The module's declared durable shadow arena (INVARIANTS.md #16), `None` if it declared none.
    shadow: Option<super::ShadowArena>,
    data: Vec<super::Data>,
    /// #964: the module's NULL-guard extent (`0` = unmarked/legacy).
    null_guard: u64,
}

impl SharedProgram {
    /// Compile `m` once (`None` if it uses an op outside the engine's subset).
    pub fn compile(m: &Module) -> Option<SharedProgram> {
        let c = compile_module_for(m)?;
        let n_funcs = c.progs.len();
        Some(SharedProgram {
            source: std::sync::Arc::new(ModuleSource::new(c)),
            n_funcs,
            mem_size_log2: m.memory.map(|mc| mc.size_log2),
            shadow: m.memory.and_then(|mc| mc.shadow),
            data: m.data.clone(),
            null_guard: temen_ir::module_null_guard(),
        })
    }

    /// Run `func(args)` over the shared window `back` with `host`, **without recompiling**. `seed_data`
    /// applies the module's data segments first — pass `true` exactly once (the initial `_start`), and
    /// `false` for every per-frame cross-tier callee (the window in `back` is already live). `Err` on a
    /// trap (`Exit` surfaces as `Trap::Exit`), or `Trap::Malformed` if `func` is out of range.
    pub fn run_over(
        &self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        back: std::sync::Arc<super::Region>,
        host: &mut Host,
        seed_data: bool,
    ) -> Result<Vec<Value>, Trap> {
        self.run_over_grown(
            func,
            args,
            fuel,
            back,
            host,
            seed_data,
            DEFAULT_RESERVED_LOG2,
            None,
        )
        .0
    }

    /// [`run_over`](Self::run_over) for a **restorable warm session** (#816): the reservation is
    /// caller-chosen (clamp it to the shared backing's size — a reservation past the backing lets
    /// guest writes silently vanish instead of failing the `map`), and a captured **explicit
    /// page-state map** can be re-established before the run: `prots = Some(entries)` re-inserts
    /// each `(byte offset, kind)` entry (the [`Mem::map_info`] encoding — the on-ramp's
    /// `protect`ed rodata and the `vm_map`-grown tail alike) *without zeroing* the pages the
    /// caller already restored ([`Mem::seed_pages`]) — so a page-managing warm image survives the
    /// fresh-`Mem`-per-call shape. Returns the run's result plus the post-run explicit page map:
    /// `Some(entries)` to seed the next call with (empty for a plain flat window), or `None` if
    /// the guest aliased a §13 `SharedRegion` page — a byte restore cannot reproduce an alias, so
    /// a warm driver must fail closed on it. `Some(vec![])` for a memory-less module.
    ///
    /// The trailing `u64` is the window's **committed scalar extent** (`Mem::map_info`'s `mapped`)
    /// after the run — `0` for a memory-less module. A cross-tier driver that grows a *live* window
    /// across bounces (the single-shot on-ramp JIT path, #1153) reads it to re-sync the emitted
    /// tier's `"mapped"` bound to the guest's real extent instead of pre-sizing a fixed window.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_over_grown(
        &self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        back: std::sync::Arc<super::Region>,
        host: &mut Host,
        seed_data: bool,
        reserved_log2: u8,
        prots: Option<&super::PageMap>,
    ) -> (Result<Vec<Value>, Trap>, Option<Vec<(u64, u8)>>, u64) {
        let (out, info, mapped) = self.run_over_grown_info(
            func,
            args,
            fuel,
            back,
            host,
            seed_data,
            reserved_log2,
            prots,
            None, // no persistent table — the install-less legacy bounce
        );
        (out, info.map(|i| i.3), mapped)
    }

    /// A dispatch table for a **persistent** reactor over this program: `2^table_log2` slots (at least
    /// the natural size), the first `n_funcs` the program's own functions, the rest install padding
    /// (#1296). Pass it to every [`Self::run_over_grown_info`] frame so installs persist across them.
    pub fn dispatch_table(&self, table_log2: u8) -> std::sync::Arc<SharedSlots> {
        std::sync::Arc::new(SharedSlots::new(self.n_funcs, table_log2, 0))
    }

    /// [`run_over_grown`](Self::run_over_grown), but returning the post-run window's whole
    /// [`MemMapInfo`] (page size, committed prefix, reservation, explicit entries) rather than the
    /// entries alone — what a **paged** cross-tier driver feeds [`build_pagestate_table`] after each
    /// bounce (#1201: the single-shot wasm-JIT tier carrying `unmap`/`protect`). `None` under the same
    /// §13 `Backed`-alias condition; `Some((1, 0, 0, vec![]))` for a memory-less module. The trailing
    /// `u64` is the scalar extent, as for `run_over_grown`.
    ///
    /// `mapped` is the committed prefix the previous bounce's `MemMapInfo` reported (`.1`), carried
    /// back so a `vm_map`-grown tail that [`Mem::seed_pages`] folded into the prefix stays committed
    /// across bounces (#1540); `0` for a map captured from a declared-size window.
    ///
    /// `table` is the reactor's **persistent** dispatch table ([`Self::dispatch_table`]): a §22
    /// `install` made in one bounce stays reachable from the next (#1296 — a child holding a
    /// re-granted `Jit` installs into its own table across its emitted run's bounces). `None` builds a
    /// natural, throwaway table for the call (no install state carried).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    pub fn run_over_grown_info(
        &self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        back: std::sync::Arc<super::Region>,
        host: &mut Host,
        seed_data: bool,
        reserved_log2: u8,
        prots: Option<&super::PageMap>,
        table: Option<&std::sync::Arc<SharedSlots>>,
    ) -> (Result<Vec<Value>, Trap>, Option<MemMapInfo>, u64) {
        if func as usize >= self.n_funcs {
            return (Err(Trap::Malformed), None, 0);
        }
        // The domain over the shared compiled source (cheap: an `Arc` clone) — the reactor's persistent
        // table when it keeps one, else a fresh natural table for this call.
        let dom = match table {
            Some(t) => Domain::child_shared(self.source.clone(), std::sync::Arc::clone(t)),
            None => Domain::child(self.source.clone(), SharedSlots::new(self.n_funcs, 0, 0)),
        };
        let mut mem = self.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(reserved_log2, sl, back, self.shadow);
            if seed_data {
                mm.init_data(&self.data);
            }
            mm.seed_null_guard(self.null_guard); // #964
            if let Some(map) = prots {
                // The map carries the prefix its entries are relative to, so the two cannot disagree —
                // they were a `prots` slice and a `mapped` argument that had to be kept in step by
                // every caller (#1456).
                mm.seed_pages(map.mapped(), &map.entries());
            }
            mm
        });
        let out = run(dom, func, args, fuel, &mut mem, host);
        let (pages, mapped) = match mem.as_ref() {
            None => (Some((1, 0, 0, Vec::new())), 0),
            Some(m) => {
                let info = m.map_info();
                // The committed **scalar extent** — not `map_info`'s `window.mapped()`, which counts
                // only the demand-committed prefix and misses a `vm_map`-grown tail (the pages live in
                // the page map). `scalar_extent` folds the contiguous grown tail into the high-water so
                // a cross-tier driver can re-sync the emitted `"mapped"` bound to admit a store into the
                // grown page (#1153). It is `None` for a layout one bound cannot describe over this
                // backing (an `Ro`/`Unmapped` hole, `Rw` past a hole, or pages committed past a fixed
                // backing): then `0`, which admits nothing, so the emitted run's next access faults and
                // the driver declines to the interpreter. (#1919: this fell back to the reservation — the
                // whole backing while the backing was pre-sized to it, before #1153; since, 2^40 bytes
                // of the embedder's memory past a backing that holds far less.)
                let mapped = m.scalar_extent().unwrap_or(0);
                if info.3.iter().any(|&(_, kind)| kind == 3) {
                    (None, mapped) // §13 Backed alias — unrestorable by a byte snapshot; fail closed
                } else {
                    (Some(info), mapped)
                }
            }
        };
        (out, pages, mapped)
    }

    /// #816 item 4 — a **cooperative tier-up run** over the shared compiled source, for a restorable
    /// warm session: the resumable [`CoopRun`] twin of [`run_over_grown`](Self::run_over_grown), so a
    /// page-managing warm guest's `eval_run` can tier its eligible leaves up onto emitted wasm
    /// instead of evaluating interpreter-only. No recompile — the run's `Domain` is a cheap `Arc`
    /// clone of the shared source plus a fresh dispatch table (sized by `host.jit_table_log2()`,
    /// the B2 convention [`CoopRun::new_over`] follows) — so a per-eval run is build-window +
    /// schedule, like `run_over_grown`. The window is **not** data-seeded (the caller already
    /// restored the warm image's bytes into `back`), the reservation is caller-chosen (clamp it to
    /// the backing, as everywhere), and the captured page-state `prots` are re-established without
    /// zeroing ([`Mem::seed_pages`]) — the same restore contract as `run_over_grown`. Read the
    /// post-run page map off the finished run via [`CoopRun::mem_map_info`]. `Err` if `entry` is
    /// out of range or scheduling traps.
    #[allow(clippy::too_many_arguments)] // the warm-restore seam inherently threads more inputs
    pub fn coop_run_over_grown(
        &self,
        entry: FuncIdx,
        args: &[Value],
        mut fuel: u64,
        mut host: Host,
        tierup: Option<TierUpConfig>,
        back: std::sync::Arc<super::Region>,
        reserved_log2: u8,
        prots: &super::PageMap,
    ) -> Result<CoopRun, Trap> {
        if entry as usize >= self.n_funcs {
            return Err(Trap::Malformed);
        }
        let dom = Domain::child(
            self.source.clone(),
            build_table(self.n_funcs, host.jit_table_log2()),
        );
        let mut mem = self.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(reserved_log2, sl, back, self.shadow);
            mm.seed_null_guard(self.null_guard); // #964
            mm.seed_pages(prots.mapped(), &prots.entries());
            mm
        });
        let sched = CoopSched::new(&dom, entry, args, &mut fuel, &mut mem, &mut host, tierup)?;
        Ok(CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        })
    }
}

/// THREADS.md step 4c — the **parallel** sibling of [`compile_and_run_capture_over`]: run the guest's
/// `thread.spawn`ed vCPUs on **separate OS threads** (the native stand-in for per-vCPU wasm Workers)
/// over the **one** caller-owned shared window, instead of cooperatively multiplexing them onto one
/// thread. Every vCPU executes over the same `Region::shared` backing — `thread.spawn`/`join` +
/// hardware `atomic.*` are genuine cross-core operations, not a single-thread interleaving. This is
/// the host-selected `Parallel` mode; the cooperative [`compile_and_run_capture_over`] is its
/// **deterministic oracle** (differential-tested in `bytecode_parallel.rs`).
///
/// Scope: the **full threads model** — `thread.spawn`/`join`, the `memory.wait`/`notify` futex
/// (a genuine cross-thread [`Futex`], not a single-thread park queue), and atomics — plus pure compute.
/// The `Domain` is shared `&`-immutably across threads, so the two events that need a `&mut
/// Domain`/shared powerbox — §14 `instantiate` and §22 JIT install — **fail closed**
/// (`Trap::ThreadFault`) here rather than run wrong; they are the remaining follow-ons. Returns `None`
/// only if the module is outside the engine's subset, same as the cooperative entry.
pub fn compile_and_run_capture_over_parallel(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
) -> Option<Capture> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let mut host = Host::new();
    compile_and_run_capture_over_parallel_with_host(m, func, args, fuel, init_mem, back, &mut host)
}

/// Like [`compile_and_run_capture_over_parallel`], but runs over a **caller-prepared `host`** (the
/// powerbox) shared by every parallel vCPU (THREADS.md 4c-host). A spawned vCPU's `call.cap` dispatches
/// on the **same** host as the root, serialized per call by an internal lock — so host I/O from worker
/// vCPUs works, with compute/atomics/futex still fully parallel. Determinism note: this is the **opt-in
/// parallel** mode, so stateful-cap interleaving (e.g. `Clock.now` values, the order of distinct
/// `stdout` writes) races as real threads do; the **cooperative** entries remain the deterministic
/// oracle. The caller reads the host back (its `stdout`/state) after the run.
pub fn compile_and_run_capture_over_parallel_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    back: std::sync::Arc<super::Region>,
    host: &mut Host,
) -> Option<Capture> {
    // #1694 — the parallel driver keeps no per-fiber shadow-SP swap and has no freeze driver, so a
    // durable host is outside it: `None`, and the caller runs it where durability is kept, rather
    // than here silently non-durable.
    if host.is_durable() {
        return None;
    }
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    let dom = Domain::new(c, host.jit_table_log2());
    // #748 — the root's park-request door: a personality's `fork()`/blocking `waitpid` fires
    // `ParkEvent`s through it (the same wiring the cooperative entries install); without it the ops
    // degrade to `-ENOSYS`/the ECHILD poll and the `ForkSelf`/`ReapWait` arms below never surface.
    host.wire_park_door();
    let mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation_over(
            DEFAULT_RESERVED_LOG2,
            mc.size_log2,
            std::sync::Arc::clone(&back),
            mc.shadow,
        );
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        mm
    });
    let (r, mem) = drive_parallel(dom, func, args, *fuel, mem, host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot(init_mem.len() as u64))
        .unwrap_or_default();
    Some((r, snap))
}

// === THREADS.md step 4c-wasm — the resumable per-vCPU primitive ==================================
// `drive_parallel` runs a guest's vCPUs on native OS threads it spawns itself. The browser can't:
// wasm32 has no `thread::spawn`, so a guest `thread.spawn` must bubble out to JS, which creates a
// Worker that re-enters the engine to run that one vCPU. That needs a *resumable, single-vCPU* entry
// the **host** orchestrates — pausing on each multi-vCPU event (`thread.spawn`/`join`,
// `memory.wait`/`notify`) and resuming once the host has serviced it. `Program` + `Vcpu` are exactly
// that primitive (platform-agnostic, no threads, no FFI): the wasm embedder drives them across Workers
// with the real `memory.atomic.wait`/`notify` futex, and the native orchestration test drives them
// across `std::thread`s as the differential proof.

/// A compiled module, shareable **read-only** across vCPUs / threads / Workers (its [`Domain`] is
/// `Sync`). Built once per run; each [`Vcpu`] borrows it. Also carries the memory declaration + data
/// segments so each vCPU can build its window over the shared backing.
pub struct VcpuProgram {
    dom: Domain,
    mem_size_log2: Option<u8>,
    /// The module's declared durable shadow arena (INVARIANTS.md #16), `None` if it declared none.
    shadow: Option<super::ShadowArena>,
    data: Vec<temen_ir::Data>,
    /// #964: the module's NULL-guard extent (`0` = unmarked/legacy), captured at compile so every
    /// window this program is run over seeds the same guard the module's layout was built for.
    null_guard: u64,
    /// The run's fiber registry (#1761) — run-level state, like `dom`'s §22 install slots: see
    /// [`fibers`](VcpuProgram::fibers).
    fibers: SharedFibers,
    /// The next dense vCPU id a `thread.spawn` hands out (root = 0), in spawn order across the run —
    /// see [`VcpuEvent::Spawn`]'s `vcpu`.
    next_vcpu: std::sync::atomic::AtomicU64,
}

impl VcpuProgram {
    /// Compile `m` for the bytecode engine, or `None` if it uses an op outside the engine's subset.
    /// The dispatch table is natural-sized (no §22 `install` room); use [`compile_with_jit_table`] to
    /// reserve padding slots for guest-driven install.
    ///
    /// [`compile_with_jit_table`]: VcpuProgram::compile_with_jit_table
    pub fn compile(m: &Module) -> Option<VcpuProgram> {
        Self::compile_with_jit_table(m, 0)
    }

    /// Like [`compile`](VcpuProgram::compile), but reserve a `call.dyn` table of `2^table_log2`
    /// slots for §22 `Jit.install` — pass the **same** value the embedder gave `grant_jit_with_table`
    /// (the powerbox's [`Host::jit_table_log2`]), so guest-driven install lands at the same slots the
    /// cooperative oracle uses. `0` ⇒ natural size (no install room).
    pub fn compile_with_jit_table(m: &Module, table_log2: u8) -> Option<VcpuProgram> {
        let c = compile_module_for(m)?;
        let dom = Domain::new(c, table_log2);
        Some(VcpuProgram {
            dom,
            mem_size_log2: m.memory.as_ref().map(|mc| mc.size_log2),
            shadow: m.memory.as_ref().and_then(|mc| mc.shadow),
            data: m.data.clone(),
            null_guard: temen_ir::module_null_guard(),
            fibers: SharedFibers::new(),
            next_vcpu: std::sync::atomic::AtomicU64::new(1),
        })
    }

    /// Reserve the next dense vCPU id (a `thread.spawn` child's), in spawn order across the run.
    fn take_vcpu_id(&self) -> u64 {
        self.next_vcpu
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// The run's **shared fiber registry** (#1761), for [`Vcpu::with_shared_fibers`]: an embedder
    /// that drives one run's vCPUs on separate threads or Workers (the browser's Worker driver)
    /// attaches it to the root and every `thread.spawn` child, so a fiber migrates between them.
    /// Like the domain's §22 install slots it is state of the run this program drives — compile a
    /// fresh program per run.
    pub fn fibers(&self) -> &SharedFibers {
        &self.fibers
    }

    /// Number of functions (a `thread.spawn` target is bounds-checked against this).
    pub fn func_count(&self) -> usize {
        self.dom.source.primary().progs.len()
    }
}

/// A persistent, single-vCPU **reactor instance** — the "instantiate once, call exports many times"
/// shape with **full-memory** fidelity. Unlike the snapshot reactors (`temen-run`'s `Session`, the
/// browser `OnrampReactor`), which round-trip only a fixed low prefix and so lose a `vm_map`-grown
/// heap between calls, a `Reactor` keeps the guest's linear-memory window **live** across calls:
/// globals, BSS, **and** the grown heap all persist frame-to-frame because the window is never torn
/// down. Host capabilities are serviced inline (the [`run`]-to-completion model, identical to
/// [`compile_and_run_with_host`]), so I/O guests work — `stdout`, and the `display`/`keyboard` caps
/// the interactive playground guests (the Doom path) use.
///
/// Single-vCPU: a guest that `thread.spawn`s is out of scope (the live window is not shared with other
/// vCPUs — use the multi-worker `drive` path for those). The usual shape is `open` → `call(0, …)` to
/// run the on-ramp `_start` bootstrap once, then `call(tick, …)` once per frame.
pub struct Reactor {
    /// The compiled program, shared (an `Arc` clone seeds each call's throwaway `Domain`).
    source: std::sync::Arc<ModuleSource>,
    n_funcs: usize,
    /// The live guest window — retained across calls (this is the whole point). `None` for a
    /// memory-less module.
    mem: Option<Mem>,
}

impl Reactor {
    /// Open a reactor over a freshly compiled `m` (`None` if `m` uses an op outside the engine's
    /// subset): build the guest window once (its data segments applied) and keep it live.
    pub fn open(m: &Module) -> Option<Reactor> {
        let c = compile_module_for(m)?;
        let n_funcs = c.progs.len();
        Some(Reactor {
            source: std::sync::Arc::new(ModuleSource::new(c)),
            n_funcs,
            mem: build_mem(m, &[]),
        })
    }

    /// Call `func(args)` on the **live** window, servicing host caps inline; the window (including a
    /// grown heap) persists after the call. `Err` on a trap (an `Exit` surfaces as `Trap::Exit`), or
    /// `Trap::Malformed` if `func` is out of range.
    pub fn call(
        &mut self,
        func: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        host: &mut Host,
    ) -> Result<Vec<Value>, Trap> {
        if func as usize >= self.n_funcs {
            return Err(Trap::Malformed);
        }
        // A fresh natural dispatch table over the shared compiled source (cheap: an `Arc` clone + the
        // slot vector) — there is no §22 install state to carry between frames, so a natural table each
        // call is correct. `run` consumes the `Domain`; the persistent `mem` carries state across calls.
        let dom = Domain::child(self.source.clone(), SharedSlots::new(self.n_funcs, 0, 0));
        run(dom, func, args, fuel, &mut self.mem, host)
    }

    /// Capture the live window as a [`MemLayout`] — the memory half of a **moment** (a reactor
    /// keyframe: time travel, a save-state, a branch point). See [`VcpuReactor::window_layout`] for
    /// why a reactor moment needs no continuation; the two reactors delegate to the one capture.
    pub fn window_layout(&self) -> Option<MemLayout> {
        window_layout_of(self.mem.as_ref())
    }

    /// Reinstate a [`window_layout`](Self::window_layout) capture into the live window. `false` for a
    /// memory-less module (nothing to restore into).
    pub fn restore_window(&mut self, layout: &MemLayout) -> bool {
        restore_window_of(self.mem.as_mut(), layout)
    }

    /// This window's reservation as a log2 — see [`window_reserved_log2_of`].
    pub fn window_reserved_log2(&self) -> Option<u8> {
        window_reserved_log2_of(self.mem.as_ref())
    }
}

/// Capture a reactor's window, or `None` when there is nothing faithfully capturable: a memory-less
/// module, or a window that has aliased a §13 `SharedRegion` — an image cannot reproduce a live alias
/// into shared backing, so the capture **refuses** rather than handing back a fiction that would
/// restore as detached bytes (INVARIANTS #9c; the same `layout_snapshot_safe` gate the checkpoint
/// ladder uses).
fn window_layout_of(mem: Option<&Mem>) -> Option<MemLayout> {
    let m = mem?;
    m.layout_snapshot_safe().then(|| m.layout_snapshot())
}

/// A reactor window's **reservation** as a log2 — the mask domain the guest grew within, which a §12
/// artifact records alongside the committed image so a thaw restores a window that can grow as far
/// again. `None` for a memory-less module.
fn window_reserved_log2_of(mem: Option<&Mem>) -> Option<u8> {
    let (_, _, reserved, _) = mem?.map_info();
    Some(reserved.trailing_zeros() as u8)
}

/// Reinstate `layout` into a reactor's live window (the write half of [`window_layout_of`]).
fn restore_window_of(mem: Option<&mut Mem>, layout: &MemLayout) -> bool {
    match mem {
        Some(m) => {
            m.restore_layout(layout);
            true
        }
        None => false,
    }
}

/// A persistent single-vCPU reactor driven through the **resumable [`Vcpu`]** — the vehicle the
/// browser wasm-JIT **tier-up** rides (BROWSER.md § "wasm-JIT tier"). Like [`Reactor`], it keeps the
/// guest window live across frames (globals, BSS, and the `vm_map`-grown heap, with its address-space
/// commit state), but each frame runs on a `Vcpu` instead of the one-shot [`run`]: a direct `Call` to
/// a [`with_jit_eligible`](Vcpu::with_jit_eligible) function surfaces as a [`VcpuEvent::TierUp`] the
/// caller services (the browser runs the emitted `f{func}` on the raw window; a native driver runs the
/// callee on the interpreter) instead of interpreting it. With no eligibility set it is a faithful,
/// interpreter-only substitute for [`Reactor`] — the differential the reactor tests assert.
///
/// The window lives in the caller-provided `back` [`Region`] (a `Region::shared` over the host's
/// linear memory in the browser; a leaked buffer natively), sized to hold the guest's grown heap. The
/// `Host` is shared (a `Mutex<Host>`) so its capabilities — `display`/`keyboard`/`fs`, stdout —
/// persist across frames and are serviced inline during each frame's `call.cap`s.
pub struct VcpuReactor {
    prog: VcpuProgram,
    /// The live window, carried across per-frame vCPUs via [`Vcpu::take_mem`]. `None` only for a
    /// memory-less module.
    mem: Option<Mem>,
    /// The tier-up eligibility bitmap (`None` ⇒ everything interprets — the pure-substitute mode).
    eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 **paged tier-up**: the eligible set was emitted with the software page-check
    /// (`compile_module_tierup_paged`). Frames then surface tier-up regardless of the window's
    /// scalar representability, and each `TierUp` hands `service` the live [`MemMapInfo`] to build
    /// its page-state table from ([`build_pagestate_table`]).
    page_checked: bool,
}

impl VcpuReactor {
    /// Open over the persistent window `back`: compile `m`, then run `_start` (func 0) once over a
    /// freshly seeded + data-initialised window to bootstrap the guest, keeping the window live for
    /// the per-frame [`frame`](VcpuReactor::frame) calls. `call.cap`s in `_start` (e.g. Doom's WAD
    /// read through `fs`) are serviced inline against `host`. `Err` if `m` is outside the engine's
    /// subset (`Malformed`) or `_start` traps.
    pub fn open(
        m: &Module,
        back: std::sync::Arc<super::Region>,
        host: &std::sync::Mutex<Host>,
        start_args: &[Value],
    ) -> Result<VcpuReactor, Trap> {
        let prog = VcpuProgram::compile(m).ok_or(Trap::Malformed)?;
        let mem;
        {
            let mut vcpu = Vcpu::new_root(&prog, 0, start_args, back, &[])?.with_shared_host(host);
            // `_start` runs to completion in one `run`: `call.cap`s are serviced inline (shared host),
            // and a reactor is single-vCPU with no tier-up during open — so no spawn/join/wait/JIT/
            // tier-up event can occur (a `thread.spawn`ing guest is out of scope).
            match vcpu.run() {
                VcpuEvent::Done(_) => {}
                VcpuEvent::Trapped(t) => return Err(t),
                // Out of a reactor's scope, named rather than `_` (see `VcpuEvent`): a `_start` that
                // spawns, joins, waits, JITs, tiers up, parks on a cap or on stdin is not a reactor.
                VcpuEvent::TierUp { .. }
                | VcpuEvent::Spawn { .. }
                | VcpuEvent::Join { .. }
                | VcpuEvent::Wait { .. }
                | VcpuEvent::Notify { .. }
                | VcpuEvent::JitInstall { .. }
                | VcpuEvent::JitUninstall { .. }
                | VcpuEvent::JitInvoke { .. }
                | VcpuEvent::Instantiate { .. }
                | VcpuEvent::InstantiateDetached { .. }
                | VcpuEvent::CapPending { .. }
                | VcpuEvent::StdinPark => return Err(Trap::Malformed),
            }
            mem = vcpu.take_mem();
        }
        Ok(VcpuReactor {
            prog,
            mem,
            eligible: None,
            page_checked: false,
        })
    }

    /// Enable wasm-JIT tier-up: a direct `Call` to a function `f` with `eligible[f] == true` surfaces
    /// as [`VcpuEvent::TierUp`] for the `frame` caller to service. `None` (the default) interprets
    /// everything — the faithful [`Reactor`] substitute.
    pub fn with_jit_eligible(mut self, eligible: std::sync::Arc<[bool]>) -> VcpuReactor {
        self.eligible = Some(eligible);
        self
    }

    /// #750 paged tier-up: mark the eligible set as page-checked (see [`Vcpu::with_jit_page_checked`]).
    /// Each frame's `TierUp` then carries the live [`MemMapInfo`] to `service` so the driver can
    /// refresh its page-state table ([`build_pagestate_table`]) before running emitted code.
    pub fn with_jit_page_checked(mut self) -> VcpuReactor {
        self.page_checked = true;
        self
    }

    /// Run `func(args)` on the live window for one frame, servicing host caps inline against `host`.
    /// A [`VcpuEvent::TierUp`] is handed to `service(func, argv, mapped, map_info)` — return the
    /// callee's i64 result slots (or an `Err(Trap)` to propagate the emitted region's trap).
    /// `mapped` is the window's scalar committed extent at call entry: `service` MUST write it to
    /// the emitted module's `"mapped"` global before invoking `f{func}` (#717 host sync).
    /// `map_info` is `Some` only on a [`with_jit_page_checked`](Self::with_jit_page_checked)
    /// reactor: the live page map, from which `service` builds its page-state table
    /// ([`build_pagestate_table`]) and writes the returned coverage to `"mapped"` **instead** of
    /// the event's value (#750). With no eligibility set, `service` is never called. The window
    /// persists after the call (reclaimed for the next frame).
    pub fn frame<F>(
        &mut self,
        func: FuncIdx,
        args: &[Value],
        host: &std::sync::Mutex<Host>,
        mut service: F,
    ) -> Result<Vec<Value>, Trap>
    where
        F: FnMut(u32, &[i64], u64, Option<MemMapInfo>) -> Result<Vec<i64>, Trap>,
    {
        let mem = self.mem.take();
        let result;
        let reclaimed;
        {
            let mut vcpu =
                Vcpu::with_mem(&self.prog, func, args, mem, Host::new())?.with_shared_host(host);
            if let Some(e) = &self.eligible {
                vcpu = vcpu.with_jit_eligible(e.clone());
            }
            if self.page_checked {
                vcpu = vcpu.with_jit_page_checked();
            }
            result = loop {
                match vcpu.run() {
                    VcpuEvent::Done(v) => break Ok(v),
                    VcpuEvent::Trapped(t) => break Err(t),
                    VcpuEvent::TierUp { func, argv, mapped } => {
                        // Paged reactors snapshot the live page map for the driver's table build;
                        // computed only here (page state is frozen while emitted code runs).
                        let info = if self.page_checked {
                            vcpu.mem_map_info()
                        } else {
                            None
                        };
                        match service(func, &argv, mapped, info) {
                            Ok(vals) => vcpu.deliver_tierup(&vals),
                            Err(t) => vcpu.deliver_tierup_trap(t),
                        }
                    }
                    // Single-vCPU reactor: no spawn/join/wait/JIT-install events.
                    _ => break Err(Trap::Malformed),
                }
            };
            reclaimed = vcpu.take_mem();
        }
        self.mem = reclaimed;
        result
    }

    /// Capture the live window as a [`MemLayout`] — the memory half of a **moment** (a reactor
    /// keyframe: time travel, a save-state, a branch point).
    ///
    /// A reactor moment needs **no continuation**. `tick` returns to the host every frame, so between
    /// frames there is no guest stack to capture, no shadow stack to unwind, and no handle table to
    /// serialize: the window — plus whatever host-side capability state the embedder holds alongside
    /// it — *is* the state. That is why this costs one image copy and no `temen-durable`
    /// instrumentation, unlike a freeze at an arbitrary safepoint (DURABILITY.md §2).
    ///
    /// `None` when there is nothing faithfully capturable — see [`window_layout_of`].
    pub fn window_layout(&self) -> Option<MemLayout> {
        window_layout_of(self.mem.as_ref())
    }

    /// Reinstate a [`window_layout`](Self::window_layout) capture into the live window — the window
    /// this reactor keeps across frames, so the next `frame` runs over the restored state. `false` for
    /// a memory-less module (nothing to restore into).
    pub fn restore_window(&mut self, layout: &MemLayout) -> bool {
        restore_window_of(self.mem.as_mut(), layout)
    }

    /// This window's reservation as a log2 — see [`window_reserved_log2_of`].
    pub fn window_reserved_log2(&self) -> Option<u8> {
        window_reserved_log2_of(self.mem.as_ref())
    }
}

/// A host-serviced pause point of a [`Vcpu`]. Everything the engine can't do alone on one thread
/// becomes one of these; the host performs the effect (spawn a Worker, futex-wait, …) and resumes the
/// vCPU with the result. Mirrors the cooperative `drive`'s `VcpuStop` arms, but handed to an external
/// orchestrator instead of serviced in-process.
///
/// **A driver matches this exhaustively — no `_` arm** (#1414). A driver that services a subset names
/// the rest, grouped with the reason it declines them, so adding a variant here fails to build in
/// every driver until each has said what it does with it. A `_` would make a new host-facing event
/// compile clean and silently do nothing in that driver — the #1347/#1339 class at its source.
pub enum VcpuEvent {
    /// The vCPU finished with these results.
    Done(Vec<Value>),
    /// The vCPU trapped (a child-join trap propagates here too).
    Trapped(Trap),
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): the interpreter reached a direct `Call`
    /// to the eligible function `func` (see [`Vcpu::with_jit_eligible`]). The host runs the emitted
    /// `f{func}(win, env, ...argv)` region on its Worker — a **top-level** call, so a guest trap is a
    /// catchable `RuntimeError` and never corrupts the engine — then calls [`Vcpu::deliver_tierup`]
    /// with the results, or [`Vcpu::deliver_tierup_trap`] if the region trapped. `argv` is the
    /// marshalled arguments as raw i64 slots (the host reads them per `func`'s signature).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        /// The window's scalar committed extent at call entry ([`Mem::scalar_extent`]) — write it to
        /// the emitted module's `"mapped"` global before invoking `f{func}` (#717 host sync).
        mapped: u64,
    },
    /// `thread.spawn`: start `func(sp, arg)` as a new vCPU, then call [`Vcpu::deliver_child`] with the
    /// host's token for it (the engine issues the handle the guest `join`s it by).
    /// `module` is the spawning frame's module (0 for plain guests; an installed §22 unit's index
    /// when its code spawns) — build the child with [`Vcpu::new_child_in`] so `func` resolves there.
    /// `vcpu` is the child's dense vCPU id, assigned here in spawn order across the run (root = 0) —
    /// seed the child with it via [`Vcpu::with_vcpu_id`], so its `vcpu.tls` starts as the spec says.
    Spawn {
        func: u32,
        sp: i64,
        arg: i64,
        module: u32,
        vcpu: u64,
    },
    /// `thread.join` (or a §14 `join`) of a live child this vCPU spawned: obtain the result of the
    /// child the host gave `child` as its token ([`Vcpu::deliver_child`]), then call
    /// [`Vcpu::deliver_join`]. The engine resolved the guest's handle by the oracle's rule — a
    /// negative, never-issued or already-joined handle traps `ThreadFault` in the vCPU — so a host
    /// keeps no child table of its own, and each token comes back at most once.
    Join { child: u64 },
    /// `memory.wait`: run the futex wait on `addr`, then call [`Vcpu::deliver_code`] with the wasm code
    /// (0 = woken, 1 = not-equal, 2 = timed-out).
    Wait {
        addr: u64,
        expected: u64,
        width: u32,
        /// The guest's timeout in ns, or `None` for an infinite wait (#1638).
        timeout: Option<u64>,
    },
    /// `memory.notify`: wake up to `count` waiters on `addr`, then call [`Vcpu::deliver_code`] with the
    /// number actually woken.
    Notify { addr: u64, count: i32 },
    /// §22 `Jit.install`: the host (which holds the powerbox) resolves authority for `handle` +
    /// code-handle `code`, returning the unit's funcs — then calls [`Vcpu::deliver_jit_install`]. The
    /// vCPU compiles + installs into the **shared** [`Domain`] (visible to every vCPU/Worker via the
    /// interior-mutable table) and writes the slot (or `-ENOSPC`) to the awaiting dst.
    JitInstall { handle: i32, code: i32 },
    /// §22 `Jit.uninstall`: the host checks authority for `handle`, then calls
    /// [`Vcpu::deliver_jit_uninstall`]; the vCPU clears the shared table `slot` (`0`/`EINVAL` → dst).
    JitUninstall { handle: i32, slot: i64 },
    /// §22 `Jit.invoke`: the host resolves the unit's funcs (authority + cross-domain), then calls
    /// [`Vcpu::deliver_jit_invoke`]; the vCPU compiles, arity-checks, and runs the unit synchronously
    /// over its window, writing the results to the awaiting dst.
    ///
    /// A **codegen** host runs the unit on emitted wasm instead ([`Vcpu::deliver_jit_invoke_vals`]).
    /// `mapped` is the window's scalar committed extent at the invoke ([`Mem::scalar_extent`]):
    /// `Some(H)` MUST be written to the emitted unit's `"mapped"` global before running it (#717
    /// host sync — same contract as [`VcpuEvent::TierUp`]); `None` means the window state is not
    /// representable by the single bound, so the host must **decline** emitted execution for this
    /// invoke and use the interpreted delivery — fail-closed, the interpreter honors the full page
    /// map. `Some(0)` for a memory-less module (nothing to bound).
    JitInvoke {
        handle: i32,
        code: i32,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: Option<u64>,
    },
    /// §14 confined child (ops 0, 5, 13, 17; THREADS.md 4c-domain §14-D2): start the admitted child
    /// over the carve, then call [`Vcpu::deliver_child`] with the host's token for it — exactly the
    /// [`VcpuEvent::Spawn`] protocol. All the authority-bearing work already happened in
    /// this vCPU, in the one admission every driver uses ([`admit_confined_child`]): the carve and entry
    /// validated (`-EINVAL` never surfaces), a module child's program resolved, compiled and **pushed to
    /// the shared source** and its data segments materialized into the carve, the child's powerbox
    /// built, its budget charged. The host's job is mechanical: take the child
    /// ([`Vcpu::take_child`]), start it over `[win + carve, win + carve + 2^size_log2)`
    /// ([`PendingChild::start`]) on a Worker/thread, and wire its completion slot into `join` — a
    /// confined child is just a child Worker with a shifted, smaller window (DESIGN.md §14: a
    /// sub-window is indistinguishable from a top-level window).
    Instantiate {
        /// The child's module: the spawning frame's for a same-module child; the pushed
        /// shared-source index for a separate module.
        module: u32,
        entry: u32,
        /// Byte offset of the carve within **this vCPU's window** (the host adds its own window
        /// base/pointer — nesting then composes with no special casing: a confined child's own
        /// `Instantiate` events are relative to *its* window).
        carve: u64,
        size_log2: u8,
    },
    /// §5 `Instantiator.instantiate_detached` (op 15, #1286): start a child in a **fresh window the
    /// host mints** — no carve, nothing of it in the parent's window. The engine already admitted it
    /// with the one admission every driver uses ([`admit_detached_child`]): the entry, the window =
    /// the declared memory, the `Budget` quota; its powerbox built (the starter caps span the
    /// reservation, as a root's do, so `vm_map` grows the window), its module compiled and pushed.
    /// The host takes the child ([`Vcpu::take_child`]), starts it over a backing it mints for the
    /// window ([`PendingChild::start`], which seeds the data segments, the payload and a pre-mapped
    /// region), and delivers its token ([`Vcpu::deliver_child`]) — the [`VcpuEvent::Instantiate`]
    /// protocol minus the carve. A host that runs the child on an emitted tier takes its powerbox
    /// instead ([`PendingChild::into_powerbox`]), seeds the payload ([`PendingChild::payload`]) and,
    /// since its window cannot alias, copies a pre-mapped region ([`Host::take_premap`]) in before
    /// and out after the run.
    InstantiateDetached {
        /// The child's window: its module's declared memory, `1 << size_log2` bytes at start.
        size_log2: u8,
    },
    /// #1366 — a **host-completed cap call** ([`crate::OffloadOutcome::Host`]): the guest's
    /// `call.cap` punted to the embedder, which will supply the scalar asynchronously. The vCPU is
    /// parked on it; the host services the request it recorded under `id` (in its submit hook),
    /// then calls [`Vcpu::deliver_cap`]`(id, value)` and [`run`](Vcpu::run) again. The guest saw a
    /// plain synchronous call. `dst` is the awaiting result slot (informational — `deliver_cap`
    /// writes it). The single-threaded-embedder twin of the pool's inline wait; the
    /// `StdinPark`/`push_stdin` shape generalized to any cap.
    CapPending { id: u64, dst: u32 },
    /// **Blocking stdin park** (a persistent interactive session, e.g. the browser Postgres console):
    /// the guest `read` a `Stream{In}` cap whose buffer is exhausted, under [`Host::set_stdin_blocking`].
    /// The read did **not** complete (nothing written, pc un-advanced); the host pushes more bytes with
    /// [`Vcpu::push_stdin`] and calls [`run`](Vcpu::run) again, which re-issues the same read — now
    /// satisfied. No `deliver_*` is needed (unlike the other events, this one carries no pending dst).
    StdinPark,
}

/// A §22 JIT op awaiting the host's [`VcpuEvent::JitInstall`]/`JitUninstall`/`JitInvoke` reply — the
/// vCPU-side residue (dst + the op's parameters) carried across the host round-trip, so the matching
/// `deliver_jit_*` can finish the op against the shared [`Domain`].
enum PendingJit {
    Install {
        dst: u32,
    },
    Uninstall {
        slot: i64,
        dst: u32,
    },
    Invoke {
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        dst: u32,
    },
}

/// One **resumable** vCPU over a shared window. The host calls [`run`](Vcpu::run) to advance it until a
/// [`VcpuEvent`], services the event, delivers the result (`deliver_*`), and runs again — so the same
/// engine semantics work whether the host orchestrates with native threads or wasm Workers. Scope (as
/// for [`drive_parallel`]): `thread.spawn`/`join` + `memory.wait`/`notify` + atomics + compute, §22
/// guest-JIT (`install`/`uninstall`/`invoke`) serviced as host events against the **shared**
/// [`Domain`], and — for a vCPU carrying a powerbox — the §14 domain ops (`spawn_coroutine_module`
/// serviced internally; the confined and detached spawns surfacing [`VcpuEvent::Instantiate`] and
/// [`VcpuEvent::InstantiateDetached`], whose admitted child the host takes ([`Vcpu::take_child`]) and
/// starts on its own Worker). By default carries a deny-all `Host` (an
/// I/O `call.cap` is an inert `CapFault`); attach the run's shared powerbox with
/// [`with_shared_host`](Vcpu::with_shared_host) (THREADS.md 4d) and `call.cap` host I/O works from
/// every vCPU sharing it, serialized per call — `drive_parallel`'s 4c-host model.
pub struct Vcpu<'p> {
    prog: &'p VcpuProgram,
    vt: VTask,
    fibers: Vec<FiberState>,
    fiber_sp: Vec<u64>,
    fiber_meta: Vec<(i32, i64)>,
    mem: Option<Mem>,
    fuel: u64,
    host: Host,
    /// The run's **shared powerbox** (THREADS.md 4d): when set (see
    /// [`with_shared_host`](Vcpu::with_shared_host)), every host access — `call.cap` dispatch, §14
    /// module/authority resolution, an invoked §22 unit's calls — goes through this `Mutex<Host>`
    /// instead of the owned `host`, exactly [`drive_parallel`]'s 4c-host model: each `call.cap` locks
    /// only for its own dispatch, so compute/atomics between calls stay lock-free, and host I/O
    /// (stream writes, clock) works from every vCPU of the run. `None` ⇒ the owned (default deny-all)
    /// host, as before.
    shared_host: Option<&'p std::sync::Mutex<Host>>,
    /// The run's **shared fiber registry** (#1761, see [`with_shared_fibers`](Vcpu::with_shared_fibers)):
    /// when set, `cont.*` go through it instead of this vCPU's own `fibers`, so a fiber created on one
    /// vCPU of the run can be resumed on another. `None` ⇒ the owned tables, as before.
    shared_fibers: Option<&'p SharedFibers>,
    /// A §14 **confined child**'s own domain (its natural table over the shared source — no parent
    /// §22 install slots); `None` for a root / `thread.spawn` child, which dispatch through
    /// [`VcpuProgram::dom`]'s table (`prog.dom`). The `source` `Arc` is the same either way.
    own_dom: Option<Domain>,
    /// The dst register awaiting a `deliver_*` after a host-serviced event.
    pending: Option<u32>,
    /// A §22 JIT op awaiting its `deliver_jit_*` (carries the op's dst + parameters across the
    /// host round-trip). Distinct from `pending` because the reply payload is richer than one register.
    pending_jit: Option<PendingJit>,
    /// #846 — the **emitted-invoke** fiber registry: while a codegen host services a
    /// [`VcpuEvent::JitInvoke`] on emitted wasm, each cross-tier callback it bounces back here
    /// ([`bounce_call`](Vcpu::bounce_call)) shares this registry, so a fiber parked by one callback
    /// is resumable by a later one — the same one-registry-per-invoke scope the interpreted
    /// `run_invoke` has by construction. Cleared when the invoke resolves (`deliver_jit_invoke_*`).
    invoke_fibers: Vec<FiberState>,
    /// The entry's initial arguments as the constructor built them — a §14 confined child's starter
    /// cap handles (`[Instantiator, AddressSpace?]`, widened to `i64`), so a host that runs the
    /// child's entry on emitted wasm (the browser's codegen path) passes exactly the handles the
    /// interpreter would, and a bounced leaf's `call.cap` resolves them on this vCPU. Empty on the
    /// other constructors.
    entry_args: Vec<Value>,
    /// A trap to surface on the next `run` (a joined child trap propagates to the joiner).
    trap: Option<Trap>,
    /// **wasm-JIT tier-up eligibility** (browser wasm-JIT threads slice). When set, `jit_eligible[f]`
    /// means function `f`'s whole reachable region is JIT-compilable and suspension-free, so a direct
    /// `Call` to it is surfaced as a [`VcpuEvent::TierUp`] — the host runs the emitted `f{f}` on the
    /// Worker (top-level caller, so a guest trap is a catchable `RuntimeError`) and delivers the
    /// result back via [`deliver_tierup`](Vcpu::deliver_tierup). `None` ⇒ everything interprets, as
    /// before this seam existed. The engine stays wasm-agnostic: it consults only this bitmap; the
    /// embedder computes it (e.g. from `temen_wasm_jit::analyze`).
    jit_eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 paged tier-up: see the `Vm` field of the same name; mirrored here for the `JitInvoke`
    /// surfacing (which reads the Vcpu, not the Vm).
    jit_page_checked: bool,
    /// A tier-up call awaiting its [`deliver_tierup`](Vcpu::deliver_tierup): the caller-frame-relative
    /// dst slot the emitted region's results land in, and their types (to re-tag the delivered raw
    /// slots — the caller's window base is the one the spill persisted).
    pending_tierup: Option<(usize, Box<[ValType]>)>,
    /// The child the last [`VcpuEvent::Instantiate`] or [`VcpuEvent::InstantiateDetached`] announced
    /// — admitted, and waiting for the host to take it ([`take_child`](Self::take_child)).
    pending_child: Option<PendingChild>,
    /// The children this vCPU spawned — threads and §14 children — indexed by the handle the guest
    /// joins them by, issued densely (0, 1, …) as the host delivers each one
    /// ([`deliver_child`](Self::deliver_child)). A join resolves its handle by the oracle's rule
    /// ([`super::take_child`]) and spends it, so every host answers a bad join alike (#1728, #1736).
    children: Vec<Option<VcpuChild>>,
    /// The just-admitted detached spawn's window lease `(budget, bytes)`, filed with its child.
    pending_lease: Option<(i32, u64)>,
    /// The lease of the child whose join is in flight, given back on
    /// [`deliver_join`](Self::deliver_join).
    joining: Option<(i32, u64)>,
}

/// A child in a [`Vcpu`]'s table: the host's token for it, and a detached child's window lease.
/// `Budget.mem` accounts live windows (INVARIANTS #3), and the driver runs the child, so this engine
/// sees its end only as the parent's join: that is when the lease's bytes go back to the budget.
struct VcpuChild {
    token: u64,
    lease: Option<(i32, u64)>,
}

/// A child that a [`VcpuEvent::Instantiate`] (§14, confined) or [`VcpuEvent::InstantiateDetached`]
/// (§5, op 15) announced, as the engine admitted it. The one admission every driver uses
/// ([`admit_confined_child`], [`admit_detached_child`]) built its powerbox — the starter
/// `Instantiator`/`AddressSpace`, the by-name re-grants, the module it serves and its import
/// manifest, a funding budget's channel ceiling, a pre-mapped region — charged its budget, and landed
/// its program in the run's shared source. What is left for the host is mechanism: a region for the
/// child's window and a thread or Worker to run it on ([`start`](Self::start)), or, for a host that
/// runs the child on an emitted tier, its powerbox and starter handles
/// ([`into_powerbox`](Self::into_powerbox)). It is `Send`: a host may start it on another thread
/// or Worker than the one that took it.
pub struct PendingChild {
    host: Host,
    module: u32,
    entry: u32,
    args: Vec<Value>,
    fuel: u64,
    window: ChildWindow,
}

const _: fn() = || {
    fn send<T: Send>() {}
    send::<PendingChild>();
};

/// The window a [`PendingChild`] runs in.
enum ChildWindow {
    /// A `1 << log2` carve of the spawner's window, already holding the child's bytes (§14).
    Carve(u8),
    /// A fresh window of its own (§5), seeded when it is built.
    Fresh(FreshWindow),
}

impl PendingChild {
    /// Start the child as a vCPU over `back`. For a confined child, `back` covers exactly its carve —
    /// `[win + carve, +2^size_log2)` of the parent's window, which per DESIGN.md §14 simply *is* the
    /// child's window (anything the parent wrote there, a module child's data segments, is already in
    /// it). For a detached child, `back` is a fresh backing the host minted; the engine seeds it (the
    /// module's data segments under the NULL guard, the args payload, a pre-mapped region), as every
    /// other driver's window is seeded. A carve is committed whole, as on every other driver, unless
    /// `committed_log2` commits less at start and leaves the child to `vm_map` the rest (#1123 slice 4:
    /// the op-13 phase lanes' shape, #1253); a fresh window always starts at its declared size.
    pub fn start(
        mut self,
        prog: &VcpuProgram,
        back: std::sync::Arc<super::Region>,
        committed_log2: Option<u8>,
    ) -> Result<Vcpu<'_>, Trap> {
        let mem = match (&self.window, committed_log2) {
            (ChildWindow::Carve(log2), committed) => {
                let shadow = prog
                    .dom
                    .source
                    .get(self.module as usize)
                    .ok_or(Trap::Malformed)?
                    .shadow;
                carve_window(back, committed.unwrap_or(*log2), *log2, shadow)?
            }
            (ChildWindow::Fresh(w), None) => w.build(Some(back), &mut self.host)?,
            (ChildWindow::Fresh(_), Some(_)) => return Err(Trap::Malformed),
        };
        Vcpu::child_over(
            prog,
            self.module,
            self.entry,
            mem,
            self.fuel,
            self.host,
            self.args,
        )
    }

    /// The child's powerbox and its entry's starter handles — its `Instantiator` and `AddressSpace`,
    /// `0` for one the entry does not take — for a host that runs it on an emitted tier over the same
    /// window rather than as a vCPU. A detached child's pre-mapped region is still staged in the
    /// powerbox ([`Host::take_premap`]).
    pub fn into_powerbox(self) -> (Host, u64, u64) {
        let handle = |i: usize| match self.args.get(i) {
            Some(Value::I64(h)) => *h as u64,
            _ => 0,
        };
        let (inst, space) = (handle(0), handle(1));
        (self.host, inst, space)
    }

    /// The spawn-time args payload a detached child's window starts with at `module_args_base()`:
    /// empty for a confined child, and for a detached spawn that passed none. For a host that runs
    /// the child on an emitted tier and so seeds its window itself
    /// ([`into_powerbox`](Self::into_powerbox)).
    pub fn payload(&self) -> &[u8] {
        match &self.window {
            ChildWindow::Fresh(w) => &w.payload,
            ChildWindow::Carve(_) => &[],
        }
    }
}

/// A confined child's window over `back`, a region covering its `1 << carve_log2` carve: committed
/// to `1 << committed_log2` at start and `vm_map`-growable to the carve (#1123 slice 4), under the
/// NULL guard. #964/#1094/#1206: the guard is the one canonical layout — a carve reserves `[0,
/// POWERBOX_NULL_GUARD)` exactly as a root window does (the tree-walker's nested arm and every
/// cross-tier bounce over the same carve seed it; the emitted tier's guard compare is unconditional).
/// `seed_null_guard` skips a carve smaller than the guard, so a tiny sub-window stays fully usable.
fn carve_window(
    back: std::sync::Arc<super::Region>,
    committed_log2: u8,
    carve_log2: u8,
    shadow: Option<super::ShadowArena>,
) -> Result<Mem, Trap> {
    if carve_log2 >= 64 || committed_log2 > carve_log2 {
        return Err(Trap::Malformed);
    }
    let mut mem = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, committed_log2, back, shadow);
    mem.seed_null_guard(temen_ir::module_null_guard());
    Ok(mem)
}

impl<'p> Vcpu<'p> {
    /// The **root** vCPU: builds its window over `back` and **seeds + data-initialises** it (the once,
    /// before any child shares it).
    pub fn new_root(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        init_mem: &[u8],
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back, prog.shadow);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, Host::new())
    }

    /// Like [`new_root`](Vcpu::new_root), but the vCPU carries a **powerbox** (its own `Host`) instead
    /// of the deny-all default — the seam §14 needs (THREADS.md 4c-domain §14-D). Unlike §22 JIT
    /// (whose ops hand the raw cap handle to the host to resolve), §14 resolves its `Instantiator`
    /// authority **in-Vm** during `resume`, so the grant must live in this vCPU's own host; with it,
    /// `spawn_coroutine_module` is then serviced entirely inside [`run`](Vcpu::run) (no host event).
    /// Grant only the non-I/O caps (`Instantiator`/`Module`) — the resumable path still has no host
    /// I/O, so an I/O `call.cap` remains an inert `CapFault`.
    pub fn new_root_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        init_mem: &[u8],
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(DEFAULT_RESERVED_LOG2, sl, back, prog.shadow);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// Like [`new_root_with_powerbox`](Vcpu::new_root_with_powerbox), but over an **engine-backed
    /// reservation** (`Mem::with_reservation`) instead of an external `Arc<Region>` — the resumable twin
    /// of [`compile_and_run_capture_reserved_with_host`], which reserves the same way. This is the
    /// persistent-backend seam (the browser Postgres console): a single owned-host vCPU that grows its
    /// heap into the `reserved_log2` tail and stays alive across [`run`](Vcpu::run) parks, so blocking
    /// stdin ([`set_stdin_blocking`](Vcpu::set_stdin_blocking)) can suspend it between queries. Uses the
    /// same `DEFAULT_RESERVED_LOG2`-scale window a one-shot `--single` boot uses; pass that.
    pub fn new_root_reserved_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        init_mem: &[u8],
        host: Host,
        reserved_log2: u8,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation(reserved_log2, sl, prog.shadow);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// [`new_root_reserved_with_powerbox`](Self::new_root_reserved_with_powerbox), but the backing
    /// is **caller-provided** (a [`Region::shared`](super::Region) over the host's own window
    /// buffer) rather than engine-owned — the native vehicle for driving wasm-JIT tier-up over a
    /// live, `vm_map`-growable window whose bytes the host must also read (the browser's
    /// shared-linear-memory shape; `tierup_grow_window.rs`). `back` must address the full
    /// `1 << reserved_log2` reservation so grown tail pages land in the caller's buffer.
    pub fn new_root_reserved_over_with_powerbox(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        init_mem: &[u8],
        host: Host,
        reserved_log2: u8,
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = prog.mem_size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(reserved_log2, sl, back, prog.shadow);
            mm.seed(init_mem);
            mm.init_data(&prog.data);
            mm.seed_null_guard(prog.null_guard); // #964
            mm
        });
        Vcpu::with_mem(prog, func, args, mem, host)
    }

    /// A `thread.spawn`ed **child** vCPU: shares `back` but does **not** re-seed (the window is already
    /// live with the root's image + every vCPU's writes). Module-0 shorthand for [`new_child_in`].
    pub fn new_child(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        Vcpu::new_child_in(prog, 0, func, args, back)
    }

    /// [`new_child`], but `func` resolves in `module` of the shared source and the child's root frame
    /// starts there — the constructor for a spawn issued by an **installed §22 unit's** code
    /// ([`VcpuEvent::Spawn`] carries the spawning frame's module; CONSOLIDATION.md §11). The child
    /// keeps `own_dom: None`: a thread shares its spawner's dispatch table, whichever module its
    /// frames start in.
    pub fn new_child_in(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
    ) -> Result<Vcpu<'p>, Trap> {
        let sl = prog.mem_size_log2;
        Self::new_child_with(prog, module, func, args, back, sl)
    }

    /// [`new_child_in`] with the window mask chosen by the caller: a thread shares its **spawner's**
    /// window, which for a §14 confined spawner is its carve — smaller than the guest module's
    /// declared memory — so the driver passes the actual window's `size_log2` (CONSOLIDATION.md §11).
    pub fn new_child_sized(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        size_log2: u8,
    ) -> Result<Vcpu<'p>, Trap> {
        Self::new_child_with(prog, module, func, args, back, Some(size_log2))
    }

    fn new_child_with(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        back: std::sync::Arc<super::Region>,
        size_log2: Option<u8>,
    ) -> Result<Vcpu<'p>, Trap> {
        let mem = size_log2.map(|sl| {
            let mut mm = Mem::with_reservation_over(
                DEFAULT_RESERVED_LOG2,
                sl,
                back,
                prog.dom.source.get(module as usize).and_then(|c| c.shadow),
            );
            // #1206: a spawned thread's `Mem` carries its own page map over the shared window, so it
            // seeds the guard itself — a thread storing at NULL traps exactly as its spawner does.
            mm.seed_null_guard(temen_ir::module_null_guard());
            mm
        });
        Vcpu::with_mem_in(prog, module, func, args, mem, Host::new())
    }

    fn with_mem(
        prog: &'p VcpuProgram,
        func: u32,
        args: &[Value],
        mem: Option<Mem>,
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        Vcpu::with_mem_in(prog, 0, func, args, mem, host)
    }

    fn with_mem_in(
        prog: &'p VcpuProgram,
        module: u32,
        func: u32,
        args: &[Value],
        mem: Option<Mem>,
        host: Host,
    ) -> Result<Vcpu<'p>, Trap> {
        let cm = prog
            .dom
            .source
            .get(module as usize)
            .ok_or(Trap::Malformed)?;
        if func as usize >= cm.progs.len() {
            return Err(Trap::Malformed);
        }
        let mut vt = VTask::new(&cm, func as usize, args)?;
        vt.active.module = module as usize;
        vt.active.home = module as usize;
        Ok(Vcpu {
            vt,
            fibers: Vec::new(),
            fiber_sp: Vec::new(),
            fiber_meta: Vec::new(),
            mem,
            fuel: u64::MAX,
            host,
            shared_host: None,
            shared_fibers: None,
            own_dom: None,
            prog,
            pending: None,
            pending_jit: None,
            invoke_fibers: Vec::new(),
            entry_args: Vec::new(),
            trap: None,
            jit_eligible: None,
            jit_page_checked: false,
            pending_tierup: None,
            pending_child: None,
            children: Vec::new(),
            pending_lease: None,
            joining: None,
        })
    }

    /// A child vCPU over its built window `mem`, with a finished powerbox `host` and entry `args`. It
    /// dispatches through its own natural table over `module` in the shared source (no parent §22
    /// install slots — the fresh table is the confinement).
    fn child_over(
        prog: &'p VcpuProgram,
        module: u32,
        entry: u32,
        mem: Mem,
        fuel: u64,
        host: Host,
        args: Vec<Value>,
    ) -> Result<Vcpu<'p>, Trap> {
        let cunit = prog
            .dom
            .source
            .get(module as usize)
            .ok_or(Trap::Malformed)?;
        let (vt, table) = child_task(
            module,
            &cunit,
            i64::from(entry),
            &args,
            host.jit_table_log2(),
        )?;
        let own_dom = Domain::child(std::sync::Arc::clone(&prog.dom.source), table);
        Ok(Vcpu {
            vt,
            fibers: Vec::new(),
            fiber_sp: Vec::new(),
            fiber_meta: Vec::new(),
            mem: Some(mem),
            fuel,
            host,
            shared_host: None,
            shared_fibers: None,
            own_dom: Some(own_dom),
            prog,
            pending: None,
            pending_jit: None,
            invoke_fibers: Vec::new(),
            entry_args: args,
            trap: None,
            jit_eligible: None,
            jit_page_checked: false,
            pending_tierup: None,
            pending_child: None,
            children: Vec::new(),
            pending_lease: None,
            joining: None,
        })
    }

    /// Attach the run's **shared powerbox** (THREADS.md 4d — builder-style, on any constructor's
    /// result): every host access of this vCPU then goes through `host` under its lock, so `call.cap`
    /// (host I/O), §14 module/authority resolution, and invoked §22 units work from every vCPU of the
    /// run sharing it — the resumable counterpart of [`drive_parallel`]'s 4c-host shared `Mutex<Host>`.
    /// The embedder grants into the `Host` *before* the run (handle order is deterministic) and reads
    /// its state (e.g. `stdout`) after; per-call serialization is the documented 4c-host model.
    pub fn with_shared_host(mut self, host: &'p std::sync::Mutex<Host>) -> Vcpu<'p> {
        self.shared_host = Some(host);
        self
    }

    /// Attach the run's **shared fiber registry** (#1761, builder-style, the fiber twin of
    /// [`with_shared_host`](Vcpu::with_shared_host)): every `cont.new` / `cont.resume` / `suspend` of
    /// this vCPU then goes through `fibers`, so a fiber created on one vCPU of the run can be resumed
    /// on another — a language runtime's worker pool, whose jobs are fibers pinned to whichever
    /// worker owns them. Attach one registry to the root and every `thread.spawn` child of a run;
    /// never to a §14 confined child, whose fibers are its own process's. Every path that touches the
    /// registry — `step_vcpu`, a tier-up bounce's nested drive, a `gc.roots` view beneath an invoke —
    /// locks it per fiber transition or scan, never across execution.
    pub fn with_shared_fibers(mut self, fibers: &'p SharedFibers) -> Vcpu<'p> {
        self.shared_fibers = Some(fibers);
        self
    }

    /// Seed this vCPU's `vcpu.tls` word with its dense vCPU id (builder-style) — a `thread.spawn`
    /// child's, from its [`VcpuEvent::Spawn`]'s `vcpu`. A root is 0, the constructors' default.
    pub fn with_vcpu_id(mut self, id: u64) -> Vcpu<'p> {
        self.vt.active.tls = id as i64;
        self
    }

    /// Attach the **wasm-JIT tier-up bitmap** (browser wasm-JIT threads slice, builder-style). A
    /// direct `Call` to a function `f` with `eligible[f] == true` then surfaces as
    /// [`VcpuEvent::TierUp`] instead of interpreting `f` — the host runs the emitted region and
    /// `deliver_tierup`s the result. `eligible.len()` should cover the primary module's functions;
    /// an out-of-range index is treated as not-eligible (interprets).
    pub fn with_jit_eligible(mut self, eligible: std::sync::Arc<[bool]>) -> Vcpu<'p> {
        self.vt.active.jit_eligible = Some(std::sync::Arc::clone(&eligible));
        self.jit_eligible = Some(eligible);
        self
    }

    /// #750 **paged tier-up**: mark the eligible set as emitted with the software page-check
    /// (`compile_module_tierup_paged`). The dispatch then surfaces tier-up regardless of the
    /// window's scalar representability — the event's `mapped` is the reserved window size, and the
    /// driver's page-state table (refreshed per call from [`mem_map_info`](Vcpu::mem_map_info))
    /// carries the per-page fidelity the scalar cannot.
    pub fn with_jit_page_checked(mut self) -> Vcpu<'p> {
        self.vt.active.jit_page_checked = true;
        self.jit_page_checked = true;
        self
    }

    /// The run's window memory-map introspection ([`MemMapInfo`]) — what a #750 page-checked
    /// driver rebuilds its byte-per-page state table from before each emitted call (page state is
    /// frozen while emitted code runs: page ops are `call.cap`s, which are never emitted and never
    /// reachable through a cross-tier leaf). `None` for a memory-less module.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// The bytes this vCPU's window backing holds, window-relative ([`Mem::win_flat_len`]): the
    /// `backed` a page-checked driver passes to [`build_pagestate_table`], past which no emitted
    /// access may be admitted. `0` for a memory-less module.
    pub fn win_flat_len(&self) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.win_flat_len())
    }

    /// The entry's initial arguments (see [`entry_args`](Self::entry_args) on the struct) — a §14
    /// confined child's starter cap handles, for a host that drives the entry on emitted wasm.
    pub fn entry_args(&self) -> &[Value] {
        &self.entry_args
    }

    /// #1009 paged tier-up: the window's page-map version — a cheap `O(1)` counter bumped on every
    /// `map`/`unmap`/`protect`. A page-checked driver caches the page-state table it built from
    /// [`mem_map_info`](Vcpu::mem_map_info) and rebuilds only when this changes (the table is
    /// identical between two tier-ups with no intervening page-op). `0` for a memory-less module.
    pub fn mem_map_version(&self) -> u64 {
        self.mem.as_ref().map_or(0, |m| m.map_version())
    }

    /// Reclaim this vCPU's live guest window after it finishes — the seam a **reactor** uses to keep
    /// the window (globals, BSS, and the `vm_map`-grown heap, with its address-space commit state)
    /// alive across per-frame vCPUs: build a vCPU over the persistent [`Mem`] with
    /// [`with_mem`](Vcpu::with_mem), run one frame to `Done`, then `take_mem` it back for the next
    /// frame. `None` for a memory-less module (or if already taken).
    pub(crate) fn take_mem(&mut self) -> Option<Mem> {
        self.mem.take()
    }

    /// Enable **blocking stdin** on this vCPU's owned powerbox (a persistent interactive session — the
    /// browser Postgres console). A `read` on an exhausted stdin buffer then surfaces
    /// [`VcpuEvent::StdinPark`] instead of returning EOF; feed more input with [`push_stdin`](Vcpu::push_stdin)
    /// and call [`run`](Vcpu::run) again. Only meaningful for an owned-host vCPU (not `with_shared_host`).
    pub fn set_stdin_blocking(&mut self, on: bool) {
        self.host.set_stdin_blocking(on);
    }

    /// Append bytes to this vCPU's stdin buffer, then [`run`](Vcpu::run) again to satisfy a pending
    /// [`VcpuEvent::StdinPark`] (or to preload input before the first `run`).
    pub fn push_stdin(&mut self, bytes: &[u8]) {
        self.host.push_stdin(bytes);
    }

    /// #1366 — finish a host-completed cap call the vCPU parked on ([`VcpuEvent::CapPending`] with
    /// this `id`): `value` lands in the call's result slot and the completion record settles, then
    /// [`run`](Vcpu::run) again. The twin of [`push_stdin`](Vcpu::push_stdin) for a cap the
    /// embedder services asynchronously (the guest saw a plain synchronous `call.cap`).
    pub fn deliver_cap(&mut self, id: u64, value: i64) {
        let dst = self
            .pending
            .take()
            .expect("deliver_cap with no pending CapPending");
        let comps = match self.shared_host {
            Some(m) => m.lock_unpoisoned().completions(),
            None => self.host.completions(),
        };
        let prefix = comps.complete_host(id, value);
        let r = comps.try_take(id).unwrap_or(value);
        if let Some((type_id, op, handle, args)) = prefix {
            let rec = super::CapRecord {
                type_id,
                op,
                handle,
                args,
                result: Ok(vec![value]),
                mem_writes: Vec::new(),
            };
            match self.shared_host {
                Some(m) => m.lock_unpoisoned().tape_cap_record(rec),
                None => self.host.tape_cap_record(rec),
            }
        }
        self.vt.active.set(dst, Reg::from_i64(r));
    }

    /// Borrow this vCPU's owned powerbox — e.g. to read `stdout` after a [`run`](Vcpu::run) that parked
    /// or finished. `None`-safe only for an owned host; a `with_shared_host` vCPU services I/O through
    /// the shared lock, not here.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// Advance this vCPU until it finishes, traps, or hits a host-serviced event. The host must
    /// `deliver_*` the result of any `Spawn`/`Join`/`Wait`/`Notify` before calling `run` again.
    ///
    ///
    /// A **durable** host runs durable (#1694): each fiber switch keeps the per-context shadow-SP.
    /// This driver has no freeze driver, though, so a run that ends frozen with a parked fiber, whose
    /// continuation only a freeze driver could flatten, fails closed (`FiberFault`) rather than hand
    /// back an artifact missing it. A freeze with no parked fiber is driven wholly by the IR. The
    /// cooperative scheduler ([`compile_and_run_capture_reserved_with_host`], [`SharedProgram`])
    /// flattens them.
    pub fn run(&mut self) -> VcpuEvent {
        let durable = match self.shared_host {
            Some(m) => m.lock_unpoisoned().is_durable(),
            None => self.host.is_durable(),
        };
        // #1366: this driver surfaces cap parks (`VcpuEvent::CapPending`) — admit host-completed
        // punts on its host. A cheap flag store per resume.
        match self.shared_host {
            Some(m) => m.lock_unpoisoned().completions().allow_host_completed(),
            None => self.host.completions().allow_host_completed(),
        }
        if let Some(t) = self.trap.take() {
            return VcpuEvent::Trapped(t);
        }
        debug_assert!(
            self.pending.is_none(),
            "deliver the last event before resuming"
        );
        // Loop so §14 `spawn_coroutine_module` (serviced in-Rust against this vCPU's own powerbox)
        // never surfaces to the orchestrating host — it only ever sees the multi-vCPU events
        // `spawn`/`join`/`wait`/`notify`, the §22 JIT events, and §14 `Instantiate` (+ `done`/`trap`).
        loop {
            // A §14 confined child dispatches through its OWN domain (own natural table, no parent
            // install slots); everything else through the program's shared one. Host access goes
            // through the run's shared powerbox when attached (4d), else the owned host.
            let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
            let mut ctx = RunCtx {
                table: &dom.table,
                fuel: &mut self.fuel,
                mem: &mut self.mem,
                durable,
                host: match self.shared_host {
                    Some(m) => HostCell::Shared(m),
                    None => HostCell::Excl(&mut self.host),
                },
            };
            // The run's shared fiber registry when attached (#1761), else this vCPU's own.
            let mut fibers = match self.shared_fibers {
                Some(s) => FiberCell::Shared(s),
                None => FiberCell::Excl {
                    fibers: &mut self.fibers,
                    sp: &mut self.fiber_sp,
                    meta: &mut self.fiber_meta,
                },
            };
            let stop = step_vcpu(
                &mut self.vt,
                &mut fibers,
                dom,
                &mut ctx,
                u64::MAX,
                false, // single-vCPU `Vcpu::run`: no cooperative waker topology (I48 idle N/A)
                false, // #1157: not preemptible (run-to-completion; budget is u64::MAX anyway)
            );
            match stop {
                // #1732 — `child_offer` mints over a live child's powerbox, which only the
                // cooperative scheduler keeps; this driver has none. "Unavailable" is the `-EINVAL`
                // the oracle gives a child it has nothing to offer over, as on the parallel driver
                // and the Cranelift nursery: a value, not a trap (INVARIANTS #5, #9).
                Ok(VcpuStop::ChildOffer { dst, .. }) => {
                    self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                }
                // #1952 — a fiber's pipe or stdin op that must wait parks the fiber alone, and its
                // resumer runs on (this driver cannot idle a blocking resume: the `FIBER_PARKED`
                // poll). A vanished pipe's op just re-runs, and fails closed.
                Ok(
                    stop @ (VcpuStop::PipeRead { .. }
                    | VcpuStop::PipeWrite { .. }
                    | VcpuStop::StdinPark),
                ) if self.vt.active_id != ROOT_FIBER => {
                    let parked = ctx.host.with(|h| {
                        HostWait::of(&stop, h).map(|on| {
                            let ready = on.ready(h);
                            (on, ready)
                        })
                    });
                    if let Some((on, ready)) = parked {
                        let vt = &mut self.vt;
                        let mem = &mut *ctx.mem;
                        fibers.with(|f, sp, _| {
                            park_fiber_on_host(vt, f, sp, mem, durable, false, on, ready)
                        });
                    }
                }
                // §3.6 (I36 slice 2): live calls / svc.wait need the cooperative scheduler's waker
                // topology (`drive`); on this single-vCPU driver nothing could ever wake them —
                // fail closed rather than hang. They need a hand-wired live cap to arrive here.
                Ok(VcpuStop::LiveCall { .. })
                | Ok(VcpuStop::SvcWait)
                | Ok(VcpuStop::CloneCaller { .. })
                | Ok(VcpuStop::Reap { .. })
                // `exec_module` image-replace needs the cooperative driver's task/env set; this
                // single-vCPU path can't rebuild the activation, so fail closed like its neighbours.
                | Ok(VcpuStop::Exec { .. })
                // #1080 rung 3 — the personality `fork()`/`waitpid()` caller-request parks need the
                // cooperative driver's task set (self-fork a twin, park until a child exits); a
                // single-vCPU path has none, so fail closed like `Exec`/`CloneCaller`.
                | Ok(VcpuStop::ForkSelf { .. })
                | Ok(VcpuStop::SpawnSelf { .. })
                | Ok(VcpuStop::ReapWait { .. })
                | Ok(VcpuStop::PipeRead { .. })
                | Ok(VcpuStop::PipeWrite { .. })
                // I48: `BlockOnFiber` is a cooperative-driver idle (this path passes
                // `cooperative: false`, so it never arises here); fail closed like its neighbours.
                | Ok(VcpuStop::BlockOnFiber { .. })
                // #1157: this path passes `preemptible: false`, so the quantum never yields here.
                | Ok(VcpuStop::Preempted) => return VcpuEvent::Trapped(Trap::ThreadFault),
                Err(t) => return VcpuEvent::Trapped(t),
                Ok(VcpuStop::Done(vals)) => {
                    let froze = durable
                        && self.mem.as_ref().map(|m| m.durable_state())
                            == Some(super::STATE_UNWINDING);
                    let parked = self.fibers.iter().any(|f| {
                        matches!(
                            f,
                            FiberState::Parked { .. }
                                | FiberState::WaitParked { .. }
                                | FiberState::CapParked { .. }
                                | FiberState::HostParked { .. }
                        )
                    });
                    if froze && parked {
                        return VcpuEvent::Trapped(Trap::FiberFault);
                    }
                    return VcpuEvent::Done(vals);
                }
                Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                }) => {
                    self.pending_tierup = Some((dst, results));
                    return VcpuEvent::TierUp { func, argv, mapped };
                }
                Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module,
                }) => {
                    // Bound-check `func` in the SPAWNING FRAME's module (an installed §22 unit spawns
                    // its own functions — CONSOLIDATION.md §11), not module 0.
                    let ok = dom
                        .source
                        .get(module as usize)
                        .is_some_and(|c| (func as usize) < c.progs.len());
                    if !ok {
                        return VcpuEvent::Trapped(Trap::Malformed);
                    }
                    self.pending = Some(dst);
                    return VcpuEvent::Spawn {
                        func,
                        sp,
                        arg,
                        module,
                        vcpu: self.prog.take_vcpu_id(),
                    };
                }
                Ok(VcpuStop::Join { handle, dst }) => {
                    match super::take_child(&mut self.children, handle) {
                        Ok(child) => {
                            self.pending = Some(dst);
                            self.joining = child.lease;
                            return VcpuEvent::Join { child: child.token };
                        }
                        Err(t) => return VcpuEvent::Trapped(t),
                    }
                }
                Ok(VcpuStop::CapPending { id, dst }) => {
                    let comps = match self.shared_host {
                        Some(m) => m.lock_unpoisoned().completions(),
                        None => self.host.completions(),
                    };
                    if comps.is_host_owned(id) {
                        // #1366 host-completed posture: the embedder finishes this one. If it
                        // already did (inside its submit hook), deliver and keep going; else park
                        // the vCPU and surface the id — resume via `deliver_cap` + `run`, the
                        // `StdinPark`/`push_stdin` shape.
                        match comps.try_take(id) {
                            Some(r) => self.vt.active.set(dst, Reg::from_i64(r)),
                            None => {
                                self.pending = Some(dst);
                                return VcpuEvent::CapPending { id, dst };
                            }
                        }
                    } else {
                        // F2: a pool-completed punt keeps the inline completion wait — identical
                        // to the pre-F2 in-op wait (the I45 whole-vCPU posture for this driver).
                        let r = comps.wait(id);
                        self.vt.active.set(dst, Reg::from_i64(r));
                    }
                }
                Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Wait {
                        addr: base,
                        expected,
                        width,
                        timeout,
                    };
                }
                Ok(VcpuStop::Notify { base, count, dst }) => {
                    self.pending = Some(dst);
                    return VcpuEvent::Notify { addr: base, count };
                }
                // §22 guest-JIT — the host resolves the unit (it holds the powerbox), the vCPU
                // installs / invokes it against the **shared** [`Domain`]. The op's residue is parked
                // in `pending_jit` until the matching `deliver_jit_*`.
                Ok(VcpuStop::JitInstall { h, code, dst }) => {
                    self.pending_jit = Some(PendingJit::Install { dst });
                    return VcpuEvent::JitInstall { handle: h, code };
                }
                Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                    self.pending_jit = Some(PendingJit::Uninstall { slot, dst });
                    return VcpuEvent::JitUninstall { handle: h, slot };
                }
                Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                }) => {
                    self.pending_jit = Some(PendingJit::Invoke {
                        argv: argv.clone(),
                        params: params.clone(),
                        results: results.clone(),
                        dst,
                    });
                    // #717 host sync: snapshot the window's scalar committed extent for a codegen
                    // host — `Some(H)` goes into the emitted unit's `"mapped"` global; `None`
                    // (unrepresentable page state) tells it to decline emitted execution and use
                    // the interpreted delivery instead. A memory-less module has nothing to bound.
                    // #750: a page-checked run surfaces the reserved size instead (see the tier-up
                    // dispatch), the table carrying per-page fidelity.
                    let mapped = match self.mem.as_ref() {
                        None => Some(0),
                        Some(m) if self.jit_page_checked => Some(m.reserved_size()),
                        Some(m) => m.scalar_extent(),
                    };
                    return VcpuEvent::JitInvoke {
                        handle: h,
                        code,
                        argv,
                        params,
                        results,
                        mapped,
                    };
                }
                // §14 confined children (ops 0, 5, 13, 17): admitted by `admit_confined_child`, the
                // one admission every driver uses, against this vCPU's own window, fuel and powerbox.
                // A refusal lands `-EINVAL` in place and the run continues; a forged handle traps. The
                // admitted child waits in `pending_child` while [`VcpuEvent::Instantiate`] asks the
                // host for mechanism only — a region over the carve and a thread or Worker to run it
                // on ([`take_child`](Self::take_child)).
                Ok(VcpuStop::Instantiate { spawn, dst }) => match self.admit_confined(spawn) {
                    Ok(Some((child, carve))) => {
                        let (module, entry) = (child.module, child.entry);
                        self.pending = Some(dst);
                        self.pending_child = Some(child);
                        return VcpuEvent::Instantiate {
                            module,
                            entry,
                            carve,
                            size_log2: spawn.size_log2 as u8,
                        };
                    }
                    Ok(None) => self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32)),
                    Err(t) => return VcpuEvent::Trapped(t),
                },
                // op 15 (`instantiate_detached`, #1286): admitted by `admit_detached_child`, the one
                // admission every driver uses, as the confined arm above. The child's window is fresh
                // rather than a carve, so the host mints its backing and `PendingChild::start` seeds
                // it. `true`: this engine leaves the capture of a durable domain's detached child to
                // its embedder's freeze.
                Ok(VcpuStop::InstantiateDetached { spawn, dst }) => {
                    let admitted = match self.shared_host {
                        Some(m) => admit_detached_child(
                            &mut m.lock_unpoisoned(),
                            self.mem.as_ref(),
                            self.fuel,
                            spawn,
                            true,
                        ),
                        None => admit_detached_child(
                            &mut self.host,
                            self.mem.as_ref(),
                            self.fuel,
                            spawn,
                            true,
                        ),
                    };
                    match admitted {
                        Ok(Some((child, window))) => {
                            let (size_log2, lease) = (window.size_log2, child.lease);
                            let window = ChildWindow::Fresh(window);
                            match self.pending_child(child, spawn.entry as u32, window) {
                                Ok(child) => {
                                    // The window lease, filed against the handle the host delivers.
                                    self.pending_lease = lease;
                                    self.pending = Some(dst);
                                    self.pending_child = Some(child);
                                    return VcpuEvent::InstantiateDetached { size_log2 };
                                }
                                Err(t) => return VcpuEvent::Trapped(t),
                            }
                        }
                        Ok(None) => self.vt.active.set(dst, Reg::from_i32(super::EINVAL as i32)),
                        Err(t) => return VcpuEvent::Trapped(t),
                    }
                }
                // Blocking-stdin park: the guest read an exhausted stdin under `set_stdin_blocking`.
                // Nothing to deliver — `pc` was left at the read, so pushing input + `run()` again
                // re-issues it. Surface to the host, which pumps the session.
                Ok(VcpuStop::StdinPark) => return VcpuEvent::StdinPark,
            }
        }
    }

    /// An admitted child as a host takes it: its program lands in the run's shared source.
    fn pending_child(
        &self,
        child: AdmittedChild,
        entry: u32,
        window: ChildWindow,
    ) -> Result<PendingChild, Trap> {
        let source = &self.own_dom.as_ref().unwrap_or(&self.prog.dom).source;
        let (module, _) = child.program.land(source)?;
        Ok(PendingChild {
            host: child.host,
            module,
            entry,
            args: child.args,
            fuel: child.fuel,
            window,
        })
    }

    /// Admit a §14 confined spawn of this vCPU's with [`admit_confined_child`], the one admission
    /// every driver uses, against its own window, fuel and powerbox: the child and where its carve
    /// starts in this vCPU's window — the host adds its window pointer, so nesting composes with no
    /// special casing. `Ok(None)` is a refusal (`-EINVAL`); `Err` a trap (a forged handle).
    fn admit_confined(
        &mut self,
        spawn: ConfinedSpawn,
    ) -> Result<Option<(PendingChild, u64)>, Trap> {
        let source = &self.own_dom.as_ref().unwrap_or(&self.prog.dom).source;
        let admitted = match self.shared_host {
            Some(m) => admit_confined_child(
                &mut m.lock_unpoisoned(),
                self.mem.as_ref(),
                self.fuel,
                source,
                &self.vt.active,
                spawn,
            ),
            None => admit_confined_child(
                &mut self.host,
                self.mem.as_ref(),
                self.fuel,
                source,
                &self.vt.active,
                spawn,
            ),
        }?;
        let Some(child) = admitted else {
            return Ok(None);
        };
        let window = ChildWindow::Carve(spawn.size_log2 as u8);
        let child = self.pending_child(child, spawn.entry as u32, window)?;
        let carve =
            self.mem.as_ref().map_or(0, |m| m.window.base()) + spawn.ibase + spawn.off as u64;
        Ok(Some((child, carve)))
    }

    /// A §14 `instantiate` (op 0) that this vCPU's code issued on an **emitted** tier — the emitted
    /// parent's spawn bounce (the browser's `env.instantiate`) — admitted as the interpreted op is:
    /// the `Instantiator` handle `inst` resolved in this vCPU's powerbox, then
    /// [`admit_confined_child`] against its window and fuel. The child runs this vCPU's module.
    /// `Ok(Some((child, carve)))`: the admitted child and where its carve starts in this vCPU's
    /// window; `Ok(None)`: refused (`-EINVAL`); `Err`: a trap (a forged handle).
    pub fn admit_instantiate(
        &mut self,
        inst: i32,
        entry: i64,
        off: i64,
        size_log2: i64,
        quota: i64,
    ) -> Result<Option<(PendingChild, u64)>, Trap> {
        let (ibase, isize) = match self.shared_host {
            Some(m) => m.lock_unpoisoned().resolve_instantiator(inst)?,
            None => self.host.resolve_instantiator(inst)?,
        };
        self.admit_confined(ConfinedSpawn {
            ibase,
            isize,
            module: None,
            entry,
            off,
            size_log2,
            quota,
            grants: None,
            budget: 0,
        })
    }

    /// Deliver the host's `token` for the child the last [`VcpuEvent::Spawn`],
    /// [`VcpuEvent::Instantiate`] or [`VcpuEvent::InstantiateDetached`] announced, once the host has
    /// started it. The engine files it under the next handle — the spawn's result, which the guest
    /// joins it by — and hands the token back on that join ([`VcpuEvent::Join`]). A token is whatever
    /// finds the child again on the host's side: a completion slot, a thread id, an index.
    pub fn deliver_child(&mut self, token: u64) {
        let handle = self.children.len() as i32;
        self.children.push(Some(VcpuChild {
            token,
            lease: self.pending_lease.take(),
        }));
        self.deliver_code(handle);
    }

    /// Give `bytes` back to `budget` in this vCPU's powerbox.
    fn budget_mem_give(&mut self, budget: i32, bytes: u64) {
        match self.shared_host {
            Some(m) => m.lock_unpoisoned().budget_mem_give(budget, bytes),
            None => self.host.budget_mem_give(budget, bytes),
        }
    }

    /// Take the child the just-surfaced [`VcpuEvent::Instantiate`] or
    /// [`VcpuEvent::InstantiateDetached`] announced — admitted, its powerbox built — to start it
    /// ([`PendingChild::start`]) or run it on an emitted tier ([`PendingChild::into_powerbox`]).
    /// One-shot per event; a host that declines the spawn simply never takes it.
    pub fn take_child(&mut self) -> Option<PendingChild> {
        self.pending_child.take()
    }

    /// Deliver a `Wait` wasm code or a `Notify` woken-count into the pending dst.
    pub fn deliver_code(&mut self, v: i32) {
        let dst = self.pending.take().expect("deliver with no pending event");
        self.vt.active.set(dst, Reg::from_i32(v));
    }

    /// Deliver a joined child's result (after `Join`): its first value lands in the joiner's dst, or a
    /// child trap propagates (the joiner traps on its next `run`).
    pub fn deliver_join(&mut self, res: Result<Vec<Value>, Trap>) {
        if let Some((budget, bytes)) = self.joining.take() {
            self.budget_mem_give(budget, bytes);
        }
        let dst = self.pending.take().expect("deliver with no pending event");
        match res {
            Ok(vals) => {
                let v = vals.first().copied().unwrap_or(Value::I64(0));
                self.vt.active.set(dst, Reg::from_value(v));
            }
            Err(t) => self.trap = Some(t),
        }
    }

    /// Deliver the resolved unit funcs for a `JitInstall` (the host resolved authority + code-handle):
    /// `Err` (forged / cross-domain / wrong-type handle) propagates as a trap; `Ok(funcs)` is compiled
    /// and installed into the **shared** [`Domain`] (so every vCPU/Worker can `call.dyn` it), the
    /// slot — or `-ENOSPC` if the table is full / `Malformed` if the unit is outside engine coverage —
    /// written to the awaiting dst.
    ///
    /// Returns `Some(slot)` iff the unit was actually installed (the slot the guest received), else
    /// `None` (trap / `-ENOSPC`). A wasm-tier host uses this to mirror the shared `Domain` slot into a
    /// per-Worker `WebAssembly.Table` (§22 Model B2 cross-Worker) — funcrefs can't cross Workers, so
    /// each Worker learns *which slot* an install filled and populates its own table. The `Domain`
    /// itself stays wasm-agnostic; the slot→code-handle→emitted-wasm mapping lives in the host.
    pub fn deliver_jit_install(
        &mut self,
        funcs: Result<std::sync::Arc<[Func]>, Trap>,
        types: std::sync::Arc<[temen_ir::TypeEntry]>,
    ) -> Option<usize> {
        let Some(PendingJit::Install { dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_install with no pending install");
        };
        let funcs = match funcs {
            Ok(f) => f,
            Err(t) => {
                self.trap = Some(t);
                return None;
            }
        };
        let (res, slot) = match compile_module(&funcs, &types, None) {
            // Install into THIS vCPU's domain (== the shared one for a root; a §14 confined child —
            // which can't hold a Jit cap anyway — would only ever fill its own table).
            Some(unit) => match self
                .own_dom
                .as_ref()
                .unwrap_or(&self.prog.dom)
                .install(unit)
            {
                Some(slot) => (slot as i64, Some(slot)),
                None => (super::ENOSPC, None),
            },
            None => {
                self.trap = Some(Trap::Malformed); // unit op outside coverage
                return None;
            }
        };
        self.vt.active.set(dst, Reg::from_i64(res));
        slot
    }

    /// Deliver the authority check for a `JitUninstall`: `Err` propagates as a trap; `Ok(())` clears the
    /// shared table `slot` (`0` on success, `EINVAL` for a real-func / out-of-range / already-empty slot).
    ///
    /// Returns `Some(slot)` iff a slot was actually cleared, so a wasm-tier host can null the matching
    /// per-Worker `WebAssembly.Table` slot (the `deliver_jit_install` counterpart) — keeping each
    /// Worker's mirror exact so a stale `call.dyn` traps.
    pub fn deliver_jit_uninstall(&mut self, authorized: Result<(), Trap>) -> Option<usize> {
        let Some(PendingJit::Uninstall { slot, dst }) = self.pending_jit.take() else {
            panic!("deliver_jit_uninstall with no pending uninstall");
        };
        if let Err(t) = authorized {
            self.trap = Some(t);
            return None;
        }
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let n_real = dom.source.primary().progs.len();
        let cleared = dom.uninstall(slot as usize, n_real);
        self.vt
            .active
            .set(dst, Reg::from_i64(if cleared { 0 } else { super::EINVAL }));
        cleared.then_some(slot as usize)
    }

    /// Deliver the resolved unit funcs for a `JitInvoke`: `Err` propagates as a trap; `Ok(funcs)` is
    /// compiled, arity-checked against the call signature (`CapFault` on mismatch), then run
    /// synchronously over this vCPU's window — its results marshalled to the awaiting dst. The invoked
    /// unit runs over this vCPU's (deny-all) powerbox, so a unit that itself makes a `call.cap` faults;
    /// a powerbox-backed unit is the orchestrator's responsibility (see [`Vcpu`]).
    pub fn deliver_jit_invoke(
        &mut self,
        funcs: Result<std::sync::Arc<[Func]>, Trap>,
        types: std::sync::Arc<[temen_ir::TypeEntry]>,
    ) {
        let Some(PendingJit::Invoke {
            argv,
            params,
            results,
            dst,
        }) = self.pending_jit.take()
        else {
            panic!("deliver_jit_invoke with no pending invoke");
        };
        let funcs = match funcs {
            Ok(f) => f,
            Err(t) => {
                self.trap = Some(t);
                return;
            }
        };
        let unit = match compile_module(&funcs, &types, None) {
            Some(u) => u,
            None => {
                self.trap = Some(Trap::Malformed);
                return;
            }
        };
        let arity_ok = unit
            .sigs
            .first()
            .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
        if !arity_ok {
            self.trap = Some(Trap::CapFault);
            return;
        }
        let child_args: Vec<Value> = params
            .iter()
            .zip(argv.iter())
            .map(|(ty, s)| slot_to_val(*ty, *s))
            .collect();
        // The effective domain borrows only `self.own_dom`/`self.prog` (shared) — disjoint from the
        // `&mut self.fuel/mem/host` fields the invoke needs, so the borrows split.
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let umod = dom.source.push(unit);
        // The invoked unit runs over the run's powerbox — the shared one when attached (its
        // `call.cap`s then serialize per-call like every other vCPU's, matching `drive_parallel`),
        // else this vCPU's owned (default deny-all) host.
        let mut cell = match self.shared_host {
            Some(m) => HostCell::Shared(m),
            None => HostCell::Excl(&mut self.host),
        };
        // The unit's `gc.roots` sees the run's parked fibers beneath it (#1660): the run-shared
        // registry when attached — locked only while such a scan reads it — else our own.
        let parked = match self.shared_fibers {
            Some(s) => FiberRegRef::Shared(s),
            None => FiberRegRef::Owned(&self.fibers),
        };
        match run_invoke(
            &dom.source,
            &dom.table,
            umod,
            &child_args,
            &mut self.fuel,
            &mut self.mem,
            &mut cell,
            Some(&Beneath::task(&self.vt, parked)),
        ) {
            Ok(vals) => {
                for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                    let re = slot_to_val(*ty, val_to_slot(*v));
                    self.vt.active.set(dst + i as u32, Reg::from_value(re));
                }
            }
            Err(t) => self.trap = Some(t),
        }
    }

    /// Deliver the **results** of a [`VcpuEvent::JitInvoke`] the host ran on **emitted wasm** (the
    /// browser's real-codegen §22 tier) instead of the engine interpreting the unit. Writes the raw
    /// i64 result slots into the awaiting `dst` and resumes — the invoke then looks exactly like the
    /// interpreted [`deliver_jit_invoke`](Vcpu::deliver_jit_invoke) that ran the unit itself. This is
    /// the alternative to that method: a host that emits wasm for the unit (`f{entry}(win, env,
    /// args)`) calls this with the emitted region's results; a host that interprets calls the other.
    /// Too few results is a `Malformed` trap (a mis-marshalled host reply).
    pub fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        self.invoke_fibers.clear(); // the emitted invoke resolved — its bounce registry dies with it
        let Some(PendingJit::Invoke { results, dst, .. }) = self.pending_jit.take() else {
            panic!("deliver_jit_invoke_vals with no pending invoke");
        };
        if vals.len() < results.len() {
            self.trap = Some(Trap::Malformed);
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.vt
                .active
                .set(dst + i as u32, Reg::from_value(slot_to_val(*ty, vals[i])));
        }
    }

    /// #846 slice 1 — service **one cross-tier bounce** out of an emitted §22 unit: the codegen host
    /// is mid-way through running a [`VcpuEvent::JitInvoke`] on emitted wasm (this vCPU is parked on
    /// the pending invoke), and the unit reached a call the emit routed to `env.call_interp` — a
    /// trampoline'd `call.dyn` target or a cross-tier direct call. `target` resolves through
    /// the **shared dispatch table** exactly as [`Op::CallIndirect`] does (the natural prefix maps a
    /// program function's index to itself; an installed unit sits at its install slot; empty padding
    /// is an `IndirectCallType` trap), and the resolved function runs on a nested interpretation
    /// over this vCPU's **live** window/powerbox/fuel — observably identical to the same call inside
    /// an interpreted invoke. Fibers are serviced against the persistent emitted-invoke registry
    /// ([`invoke_fibers`](Vcpu::invoke_fibers)), so a fiber parked by one callback is resumable by a
    /// later one within the same invoke.
    ///
    /// `io` carries the i64 arg slots in and the result slots out (the `env.call_interp` scratch
    /// ABI — floats by bits, i32s in the low half). Returns the result count. `Err` is the
    /// callback's trap — the host must unwind the emitted unit and deliver it via
    /// [`deliver_jit_invoke_trap`](Vcpu::deliver_jit_invoke_trap) (an `Exit` included: it must
    /// resolve the invoke as the interpreted path would, not be swallowed).
    ///
    /// `mirror` (#1339) is the driver's dispatch-table mirror, lent for the duration of the bounce so
    /// a §22 `Jit.install`/`uninstall` the callback reaches — a guest whose *emitted* frame defines
    /// and dispatches units — moves it before the emitted frame resumes; the driver rebuilds its
    /// table when the generation advances. `None` where there is no shared table to mirror (the
    /// native embedding), and for a §14 child, whose installs stay in its own table (#1296).
    pub fn bounce_call(
        &mut self,
        target: u32,
        io: &mut [i64],
        mirror: Option<JitMirror<'_>>,
    ) -> Result<usize, Trap> {
        step(&mut self.fuel, None)?; // fuel unification: the dispatch-site safepoint
        let dom = self.own_dom.as_ref().unwrap_or(&self.prog.dom);
        let slot = (target as usize) & (dom.table.len() - 1);
        let ts = dom.table.slot(slot);
        if ts.module == super::TABLE_EMPTY {
            return Err(Trap::IndirectCallType);
        }
        let tm = dom.source.get(ts.module as usize).ok_or(Trap::Malformed)?;
        let (cp, cr) = tm.sigs[ts.func as usize].clone();
        if cp.len() > io.len() || cr.len() > io.len() {
            return Err(Trap::Malformed); // scratch too small — a mis-marshalled host call
        }
        // The i64-slot transport carries scalars only; a v128-sig target can never have been given
        // a trampoline (the host gates that at open) — a bounce naming one is a mis-wired host.
        let scalar =
            |t: &ValType| matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64);
        if !cp.iter().all(scalar) || !cr.iter().all(scalar) {
            return Err(Trap::Malformed);
        }
        let args: Vec<Value> = cp
            .iter()
            .zip(io.iter())
            .map(|(ty, s)| slot_to_val(*ty, *s))
            .collect();
        let mut vm = Vm::new(&tm, ts.func as usize, &args)?;
        vm.module = ts.module as usize;
        vm.tls = self.vt.active.tls; // the callback runs on this vCPU: its `vcpu.tls` word
        let mut cell = match self.shared_host {
            Some(m) => HostCell::Shared(m),
            None => HostCell::Excl(&mut self.host),
        };
        // Registry context (#880): during an emitted **invoke**, callbacks share the
        // invoke-confined registry (`run_invoke` parity — fibers die with the invoke). During an
        // emitted **TIERUP region**, the callback is interpreted-inline-call territory: its fibers
        // register in the vCPU's *run-level* registry (parallel arrays mirrored), so one created
        // here persists for the run to resume later — exactly as the same call inline would.
        let vals = if self.pending_jit.is_some() {
            drive_nested(
                &dom.source,
                &dom.table,
                vm,
                &mut self.fuel,
                &mut self.mem,
                &mut cell,
                // Invoke fibers are transient: no shadow-SP / freeze tables to keep aligned.
                &mut FiberCell::Excl {
                    fibers: &mut self.invoke_fibers,
                    sp: &mut Vec::new(),
                    meta: &mut Vec::new(),
                },
                None,
                None, // #1660: emitted frames lie beneath a bounce; opaque until they spill (#1627)
            )?
        } else {
            // The run-level registry: the run-shared one when attached (#1761), else this vCPU's.
            let mut fibers = match self.shared_fibers {
                Some(s) => FiberCell::Shared(s),
                None => FiberCell::Excl {
                    fibers: &mut self.fibers,
                    sp: &mut self.fiber_sp,
                    meta: &mut self.fiber_meta,
                },
            };
            drive_nested(
                &dom.source,
                &dom.table,
                vm,
                &mut self.fuel,
                &mut self.mem,
                &mut cell,
                &mut fibers,
                Some(BounceRunCtx {
                    jit_mirror: mirror,
                    park: None,
                }),
                None, // #1660: emitted frames lie beneath a bounce; opaque until they spill (#1627)
            )?
        };
        for (i, v) in vals.iter().enumerate() {
            io[i] = val_to_slot(*v);
        }
        Ok(cr.len())
    }

    /// The window's committed **scalar extent** right now ([`Mem::scalar_extent`]) — the #717 value
    /// the codegen host re-syncs to every live instance's `"mapped"` global after a
    /// [`bounce_call`](Vcpu::bounce_call) (a bounced callback may have `vm_map`-grown the window
    /// mid-invoke; the fan-out makes the growth visible to the emitted tier exactly when the
    /// interpreted path would see it — after the call returns). `0` when the window state is no
    /// longer scalar-representable — deny-everything, the diverge-toward-refusal posture
    /// (INVARIANTS.md #9), or when there is no window.
    pub fn window_scalar_extent(&self) -> u64 {
        self.mem
            .as_ref()
            .and_then(|m| m.scalar_extent())
            .unwrap_or(0)
    }

    /// Deliver a **trap** from a host-run [`VcpuEvent::JitInvoke`] unit (the emitted region hit a
    /// guest `unreachable` / memory fault / div-by-zero / out-of-fuel, surfaced to the host as a
    /// catchable `RuntimeError`). The vCPU traps on its next `run`, exactly as an interpreted invoke
    /// trap would (`deliver_jit_invoke` sets `self.trap` on the unit's `Err`).
    pub fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        self.invoke_fibers.clear(); // the emitted invoke resolved — its bounce registry dies with it
        self.pending_jit = None;
        self.trap = Some(trap);
    }

    /// Deliver the results of a [`VcpuEvent::TierUp`]: the emitted region returned `vals` (raw i64
    /// result slots, one per the callee's result type). Re-tag each into the awaiting `dst` slot(s) of
    /// the caller's window and resume — the tier-up call then looks exactly like an interpreted call
    /// that returned. Too few results is a `Malformed` trap (a mis-marshalled host reply).
    pub fn deliver_tierup(&mut self, vals: &[i64]) {
        let Some((dst, results)) = self.pending_tierup.take() else {
            panic!("deliver_tierup with no pending tier-up");
        };
        if vals.len() < results.len() {
            self.trap = Some(Trap::Malformed);
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.vt.active.set(
                dst as u32 + i as u32,
                Reg::from_value(slot_to_val(*ty, vals[i])),
            );
        }
    }

    /// Deliver a **trap** from a [`VcpuEvent::TierUp`] region (the emitted `f{func}` hit a guest
    /// `unreachable` / memory fault / div-by-zero / out-of-fuel, surfaced to the host as a catchable
    /// `RuntimeError`). The vCPU traps on its next `run`, exactly as if the interpreted call had.
    pub fn deliver_tierup_trap(&mut self, trap: Trap) {
        self.pending_tierup = None;
        self.trap = Some(trap);
    }

    /// Snapshot this vCPU's window (its `[0, prefix_len)` span) after it finishes — the root's image
    /// for capture. (The bytes also live in the shared backing the host handed in, so a wasm host can
    /// read them straight from the `SharedArrayBuffer` instead.)
    pub fn snapshot(&self, prefix_len: u64) -> Vec<u8> {
        self.mem
            .as_ref()
            .map(|m| m.snapshot(prefix_len))
            .unwrap_or_default()
    }
}

/// Durability seam (Slice 1c-6): the bytecode mirror of [`crate::run_capture_reserved_with_host`] —
/// seed the window with `init_mem` (which for a durable run carries the state word + shadow region),
/// run `m`'s transformed entry over a caller-prepared `host` (the powerbox), and snapshot the window
/// (the `SNAP_CAP` span, matching the tree-walker / JIT durable capture). Single-vCPU, single-fiber
/// freeze/thaw is **driven entirely by the transform's emitted IR** — the engine just runs it; this
/// is the entry the freeze/thaw harness (`bytecode_durable.rs`) and the `super::run_with_host_fast`
/// fast path use. `None` if the module is outside the engine's subset.
pub fn compile_and_run_capture_reserved_with_host(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> Option<Capture> {
    // Multi-vCPU durability (`thread.*`) is out of scope: a durable thread spawn needs the
    // multi-worker freeze the engine doesn't drive, so always refuse it (the caller falls back to the
    // tree-walker), lest it write a silently-wrong artifact. §14 **nesting** (`Instantiator`
    // call.cap calls) is likewise out of scope (DURABILITY.md §4): the tree-walker owns the durable
    // nesting rules — the freezable-module admission check, child durability inheritance, and the
    // fail-closed refusal of a freeze over a live-or-unjoined §14 child; this engine's own
    // instantiate arm has none of them, so driving a durable §14 module here would both skip the
    // admission rule and mint the exact thaw-faulting artifact the tree-walker refuses.
    if outside_reserved_subset(m) {
        return None;
    }
    // `cont.*` durability is fully supported (DURABILITY.md §12.8): the per-fiber shadow-SP swap keeps
    // the active word on the running context (so a freeze poll spills into the right region), the freeze
    // driver flattens idle parked fibers into their regions, and thaw seeding re-creates them from the
    // artifact residue. So a single-vCPU `cont.*` module is driven here in any window state (NORMAL /
    // UNWINDING freeze / REWINDING thaw); only multi-vCPU `thread.*` (above) still falls back.
    let c = std::sync::Arc::new(compile_module_for(m)?);
    run_capture_reserved_over_compiled_with_host(
        m,
        c,
        func,
        args,
        fuel,
        init_mem,
        reserved_log2,
        host,
    )
}

/// The modules the reserved-window entries refuse before compiling: multi-vCPU `thread.*` and §14
/// nesting (`Instantiator` calls) — see [`compile_and_run_capture_reserved_with_host`] for why.
fn outside_reserved_subset(m: &Module) -> bool {
    m.funcs.iter().flat_map(|f| f.blocks.iter()).any(|b| {
        b.insts.iter().any(|i| {
            matches!(i, Inst::ThreadSpawn { .. } | Inst::ThreadJoin { .. })
                || matches!(i, Inst::CapCall { type_id, .. } if *type_id == super::cap_id::INSTANTIATOR)
        })
    })
}

/// Whether this engine runs `m` at all on the reserved-window path — the question a caller that
/// must not silently fall back to the tree-walker has to ask first (the answer
/// [`compile_and_run_capture_reserved_with_host`] gives as `None`, after the fact).
pub fn admits_reserved(m: &Module) -> bool {
    !outside_reserved_subset(m) && compile_module_for(m).is_some()
}

/// #1144 — **compile the reserved-window program without running it**, so a caller (the browser bash
/// entry) can **cache the `Arc<Compiled>` across Runs** and skip the ~200 ms per-Run recompile of a
/// large module. `None` if the module is outside the bytecode subset (same gate as
/// [`compile_and_run_capture_reserved_with_host`]'s compile). Pair with
/// [`run_capture_reserved_over_compiled_with_host`], which takes the cached program.
pub fn compile_reserved(m: &Module) -> Option<std::sync::Arc<Compiled>> {
    compile_module_for(m).map(std::sync::Arc::new)
}

/// #1144 — the run half of [`compile_and_run_capture_reserved_with_host`], over an **already-compiled**
/// (and typically cached) `Arc<Compiled>`. Everything after the compile is identical — the `thread.*`/
/// `Instantiator` out-of-scope refusal, the personality park door, `Mem` reservation + seed +
/// data-init + NULL guard from `m`, run, window snapshot. A fresh `Domain`/`ModuleSource` wraps the
/// shared program each call (so exec'd command units pushed this run don't leak into the next), reusing
/// only the immutable primary program. `m` is still needed for the `Mem` init (data segments, window
/// size, NULL guard) and the out-of-scope scan — cheap reads, no recompile.
#[allow(clippy::too_many_arguments)] // mirrors compile_and_run_capture_reserved_with_host, plus `compiled`
pub fn run_capture_reserved_over_compiled_with_host(
    m: &Module,
    compiled: std::sync::Arc<Compiled>,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    init_mem: &[u8],
    reserved_log2: u8,
    host: &mut Host,
) -> Option<Capture> {
    // Same out-of-scope gate as the compile-and-run entry — a cached program from a caller that also
    // holds the module must still refuse the `thread.*`/§14-nesting shapes the freeze path can't drive.
    if outside_reserved_subset(m) {
        return None;
    }
    if func as usize >= compiled.progs.len() {
        return Some((Err(Trap::Malformed), Vec::new()));
    }
    // #799/#1080 — install the personality fork/waitpid park-request door (see `compile_and_run_with_host`).
    host.wire_park_door();
    let dom = Domain::over_primary(compiled, host.jit_table_log2());
    let mut mem = m.memory.map(|mc| {
        let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2, mc.shadow);
        mm.seed(init_mem);
        mm.init_data(&m.data);
        mm.seed_null_guard(temen_ir::module_null_guard()); // #964
        mm
    });
    let r = run(dom, func, args, fuel, &mut mem, host);
    let snap = mem
        .as_ref()
        .map(|mm| mm.snapshot_window(super::SNAP_CAP))
        .unwrap_or_default();
    Some((r, snap))
}

/// An [`ir_trace`] result: the executed instruction-location sequence plus the run's result.
pub type IrTrace = (Vec<super::IrPc>, Result<Vec<Value>, Trap>);

/// A per-step **window-variable** trace ([`ir_window_trace`]): each executed instruction's [`crate::IrPc`]
/// paired with the watched window range's bytes at that point, plus the run result.
pub type WindowTrace = (Vec<(super::IrPc, Vec<u8>)>, Result<Vec<Value>, Trap>);

/// A per-step **SSA-value** trace ([`ir_value_trace`]): each executed instruction's [`crate::IrPc`]
/// paired with the current frame's typed block-local SSA values, plus the run result.
pub type ValueTrace = (Vec<(super::IrPc, Vec<Value>)>, Result<Vec<Value>, Trap>);

/// Debug seam (Slice 1c-3): single-step `m`'s `func(args)` and record the [`crate::IrPc`] of each
/// op executed — instructions and terminators alike (#1713), matching the tree-walker's `before_op` —
/// returning the location trace plus the result. `None` if the module is
/// outside the engine's subset, or if a step hits a concurrency/coroutine seam (debug is single-vCPU,
/// seam-free — DEBUGGING.md S4). Stepping uses `budget = 1` so each `resume` runs exactly one op.
///
/// The resulting trace is **identical** to driving the tree-walker [`crate::Inspector`] with
/// `seek(0), seek(1), …` — that equality (checked by `bytecode_debug.rs`) is what proves the engine
/// reports tree-walker-identical locations, so breakpoints/stepping at [`crate::IrPc`] granularity
/// land at the same program points on both backends.
pub fn ir_trace(m: &Module, func: FuncIdx, args: &[Value], fuel: &mut u64) -> Option<IrTrace> {
    let c = compile_module_unfused(&m.funcs, &m.types, m.memory.and_then(|x| x.shadow))?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m, &[]);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            trace.push(pc);
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue, // one op done; keep stepping
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Debug-seam **variable-inspection** support (DEBUGGING.md §1b G2). Like [`ir_trace`], but at each
/// instruction step also snapshots `len` window bytes at `addr` — the value a *window-located* source
/// variable (`VarLoc::Window`) holds at that program point. Register-allocated SSA values have no
/// stable cross-engine storage (the bytecode engine packs them into reused slots), but a window
/// variable lives at a shared address in the same `Mem` both engines drive, so its value *is*
/// comparable per step. Paired with the tree-walker `Inspector` driven by `seek(t)` +
/// `read_var`/`read_window`, this proves the two engines hold the **same variable value at every
/// step** — not merely the same locations (`ir_trace`). `None` on the same out-of-subset / seam
/// conditions as [`ir_trace`]. Test surface; not a production entry point.
pub fn ir_window_trace(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    addr: u64,
    len: usize,
) -> Option<WindowTrace> {
    let c = compile_module_unfused(&m.funcs, &m.types, m.memory.and_then(|x| x.shadow))?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m, &[]);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        // Snapshot the window var *before* running the op — the same point `Inspector::seek(t)` pauses
        // at (paused before the op at clock `t`), so the two byte sequences align step-for-step.
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            let bytes = mem
                .as_ref()
                .and_then(|mm| mm.read_window(addr, len).ok())
                .unwrap_or_default();
            trace.push((pc, bytes));
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue,
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Debug-seam **SSA-value inspection** support (DEBUGGING.md §1b G2). Like [`ir_trace`], but at each
/// instruction step also records the current frame's typed block-local SSA values. `compile_func`
/// assigns a **stable, unique slot per value** (no register reuse / coalescing — "global slot per
/// value"), so an SSA value *is* directly inspectable: `regs[base + i]` typed by `func_value_types`,
/// exactly the storage the tree-walker's `read_ir_value` reads. **Single-block functions only**, where
/// the bytecode slot index equals the tree-walker's block-local value index (both `base`-0); `None`
/// for a multi-block function (per-block slot base differs) or the out-of-subset / seam cases
/// [`ir_trace`] declines. Paired with `Inspector::read_ir_value`/`read_var`, this proves SSA-located
/// variables hold the same value on both engines — the bytecode tier is inspectable, not precluded.
/// Test surface; not a production entry point.
pub fn ir_value_trace(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
) -> Option<ValueTrace> {
    // Single-block scope keeps slot index == tree-walker block-local index (see doc).
    if m.funcs.get(func as usize)?.blocks.len() != 1 {
        return None;
    }
    let types0 = temen_verify::func_value_types(
        &m.funcs[func as usize],
        &m.funcs,
        &m.types,
        m.memory.is_some(),
    )
    .into_iter()
    .next()
    .unwrap_or_default();
    let c = compile_module_unfused(&m.funcs, &m.types, m.memory.and_then(|x| x.shadow))?; // unfused: one step per source inst (Slice 5a)
    if func as usize >= c.progs.len() {
        return Some((Vec::new(), Err(Trap::Malformed)));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m, &[]);
    let mut host = Host::new();
    let mut vm = match Vm::new(&dom.source.primary(), func as usize, args) {
        Ok(v) => v,
        Err(e) => return Some((Vec::new(), Err(e))),
    };
    let mut trace = Vec::new();
    loop {
        if let Some(pc) = vm.cur_ir_pc(&dom.source) {
            // The block-0 register window typed per value — the same `(base + i, type)` resolution the
            // tree-walker uses for `read_ir_value`. A not-yet-computed slot reads as its default `Reg`;
            // the caller compares only the defined prefix (where `read_ir_value` returns `Some`).
            let vals: Vec<Value> = types0
                .iter()
                .enumerate()
                .map(|(i, &ty)| vm.regs[vm.base + i].to_value(ty))
                .collect();
            trace.push((pc, vals));
        }
        match vm.resume(
            &dom.source,
            &dom.table,
            fuel,
            &mut mem,
            &mut HostCell::Excl(&mut host),
            1,
        ) {
            Ok(Outcome::Suspended) => continue,
            Ok(Outcome::Done(vals)) => return Some((trace, Ok(vals))),
            Ok(_) => return None, // a seam — out of single-vCPU debug scope
            Err(t) => return Some((trace, Err(t))),
        }
    }
}

/// Per-module §6 debug metadata a [`FrameReader`] resolves a source variable against: the `-g` info plus
/// the per-`(func, block)` slot base and value types needed to read any live frame's values. Module 0's
/// lives directly on the [`DebugRun`]/[`ScheduledDebugRun`]; a §14 **separate-module** child carries its
/// own here (built from the granted `Module` at spawn, keyed by its pushed source index) so
/// `read_var`/`var_addr`/`value_in_frame` resolve inside the child's body, not just module 0's.
#[derive(Clone)]
struct ModuleDebug {
    /// The child's index in the shared [`ModuleSource`] (`>= 1`) — matched against a frame's module so a
    /// frame outside this child (e.g. an installed §22 unit) isn't misread against these tables.
    module: usize,
    debug: Option<DebugInfo>,
    fn_block_base: Vec<Vec<u32>>,
    fn_block_types: Vec<Vec<Vec<ValType>>>,
}

impl ModuleDebug {
    /// Build the per-`(func, block)` slot base + value types + §6 debug info for module `m`, pushed to the
    /// shared source at index `module`. Mirrors the module-0 computation in [`ScheduledDebugRun::new_with_host`], so
    /// a separate-module child is inspected exactly as the primary program is.
    fn build(m: &Module, module: usize) -> ModuleDebug {
        let arities: Vec<usize> = m.funcs.iter().map(|g| g.results.len()).collect();
        let mut fn_block_base = Vec::with_capacity(m.funcs.len());
        let mut fn_block_types = Vec::with_capacity(m.funcs.len());
        for g in &m.funcs {
            let mut base = Vec::with_capacity(g.blocks.len());
            let mut n = 0u32;
            for b in &g.blocks {
                base.push(n);
                n += b.params.len() as u32;
                for inst in &b.insts {
                    n += inst.result_count(&arities, &m.types) as u32;
                }
            }
            fn_block_base.push(base);
            fn_block_types.push(temen_verify::func_value_types(
                g,
                &m.funcs,
                &m.types,
                m.memory.is_some(),
            ));
        }
        ModuleDebug {
            module,
            debug: m.debug_info.clone(),
            fn_block_base,
            fn_block_types,
        }
    }
}

/// Whether a §14 child window (a coroutine or an `instantiate` env) is captured by a checkpoint: its own
/// page map is capturable ([`Mem::layout_snapshot_safe`] — no §13 region aliasing) **and** its extent
/// lies within the parent's snapshotted prefix ([`Mem::nested_within_prefix`], so its bytes ride in the
/// parent reseed rather than a separate copy). A memoryless child is trivially fine. Shared by the
/// single-vCPU and scheduled checkpointable gates, for both coroutines and `instantiate` children.
fn child_checkpointable(child: Option<&Mem>, parent: Option<&Mem>) -> bool {
    child.is_none_or(|m| {
        m.layout_snapshot_safe() && parent.is_some_and(|p| m.nested_within_prefix(p))
    })
}

/// Capture a live `instantiate`-child [`DbgEnv`] into an [`EnvSnapshot`]: its window geometry
/// (`nested_view` base + size) + host replay substate + fuel + its own page-protection map. Its bytes ride
/// in the shared window snapshot (the view shares the root backing region).
fn env_snapshot(e: &DbgEnv, module: usize) -> EnvSnapshot {
    EnvSnapshot {
        win_base: e.mem.as_ref().map_or(0, |m| m.window.base()),
        size_log2: e
            .mem
            .as_ref()
            .map_or(0, |m| m.window.reserved().trailing_zeros() as u8),
        module,
        host: e.host.replay_substate(),
        fuel: e.fuel,
        prot: e.mem.as_ref().map_or_else(Vec::new, |m| m.prot_snapshot()),
        fibers: e.fibers.clone(),
    }
}

/// Rebuild a §14 `instantiate`-child [`DbgEnv`] from an [`EnvSnapshot`] on restore — the inverse of
/// [`env_snapshot`]: recreate its `nested_view` over the (reseeded) `shared_mem` window, its attenuated
/// `Instantiator` + `AddressSpace` powerbox over `[0, child_size)` (deterministic in `child_size`), the
/// captured host replay substate, its natural module-0 table, and its fuel quota.
fn rebuild_env(es: &EnvSnapshot, shared_mem: Option<&Mem>, source: &ModuleSource) -> DbgEnv {
    let child_size = 1u64 << es.size_log2;
    let child_shadow = source
        .get(es.module)
        .and_then(|c| c.shadow)
        .unwrap_or(super::ShadowArena::EMPTY);
    let mem = shared_mem.map(|m| m.nested_view(es.win_base, es.size_log2, child_shadow));
    if let Some(m) = &mem {
        m.install_prot(&es.prot); // its `map`/`unmap`/`protect`ed pages (bytes rode in the shared reseed)
    }
    let progs_len = source.get(es.module).map_or(0, |u| u.progs.len());
    let mut host = Host::new();
    host.grant_instantiator(0, child_size);
    host.grant_address_space(0, child_size);
    host.restore_replay_substate(&es.host);
    let table_log2 = host.jit_table_log2(); // #1296: the child's reserved install slots
    DbgEnv {
        mem,
        host,
        table: build_table_for(progs_len, table_log2, es.module as u32),
        fuel: es.fuel,
        fibers: es.fibers.clone(),
    }
}

/// A resolved value-watch target for the bytecode engine — frame-independent so the DAP backend can
/// re-apply it verbatim after a `seek` rebuild: the owning function, how to find the variable's
/// holding value at any pc ([`ValueSite`]), and the arm-time baseline value. Opaque to the DAP layer
/// (it only stores and hands these back).
#[derive(Clone)]
pub struct ValueWatchTarget {
    func: u32,
    site: super::ValueSite,
    last: super::Reg,
}

/// A value watch live in a [`ScheduledDebugRun`] — a [`ValueWatchTarget`] plus the caller-owned [`WatchId`]
/// and the running `last` value (updated on each fire so the next change re-fires). Kind gates
/// firing (change is a write; a `Read`-only value watch never fires — an SSA value has no read).
struct ValueWatchRun {
    id: super::WatchId,
    func: u32,
    site: super::ValueSite,
    last: super::Reg,
    kind: super::WatchKind,
}

/// The watched range the op at module-0 `(func, block, inst)` would hit, from the live block-local
/// values — the bytecode counterpart of the tree-walker's `access_of` + `watch_hit`. `None` if the op
/// accesses no watched range (or its address can't be resolved). A free fn (not a method) so it borrows
/// only the pieces `run_to` has already split out of `&mut self`.
#[allow(clippy::too_many_arguments)]
fn watch_hit_before(
    vm: &Vm,
    mem: &Option<Mem>,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    watchpoints: &[(u64, u64, super::WatchKind)],
    func: FuncIdx,
    block: usize,
    inst: usize,
) -> Option<(u64, bool)> {
    let ir_inst = funcs
        .get(func as usize)?
        .blocks
        .get(block)?
        .insts
        .get(inst)?;
    let base_off = *fn_block_base.get(func as usize)?.get(block)? as usize;
    let vals = vm.regs.get(vm.base + base_off..)?;
    // `watch_accesses` (not `access_of`): bulk `mem.copy`/`mem.move`/`mem.fill` and v128 ops
    // check both their spans, so a memcpy over a watched byte stops here like a plain store.
    super::watch_accesses(ir_inst, vals, mem)
        .into_iter()
        .find_map(|acc| {
            let super::MemAccess::Range { base, width, write } = acc else {
                return None;
            };
            let end = base.saturating_add(width as u64);
            watchpoints.iter().find_map(|(addr, len, kind)| {
                let w_end = addr.saturating_add(*len);
                (base < w_end && *addr < end && kind.fires_on(write)).then_some((base, write))
            })
        })
}

/// The first value watch whose variable's holding SSA value **changed** since it was last seen, for
/// the top frame at module-0 `(func, block, inst)` — the bytecode counterpart of the tree-walker's
/// `DebugShared::value_watch_hit` (#1229). Reads the holding slot from the top frame's window
/// (`vm.base + block-base + value-index`) and updates the watch's `last` in passing, so the next
/// change re-fires. `Some(())` ⇒ pause; a `Read`-only kind never fires (an SSA value has no read).
/// A free fn so it borrows only the pieces `run_to` has split out of `&mut self`.
fn value_watch_hit_before(
    vm: &Vm,
    fn_block_base: &[Vec<u32>],
    value_watches: &mut [ValueWatchRun],
    func: FuncIdx,
    block: usize,
    inst: usize,
) -> Option<()> {
    let base_off = *fn_block_base.get(func as usize)?.get(block)? as usize;
    for w in value_watches.iter_mut() {
        if w.func != func || !w.kind.fires_on(true) {
            continue;
        }
        let idx = match &w.site {
            super::ValueSite::Single(s) => *s as usize,
            // Not live at this pc (no covering loclist entry) ⇒ don't compare.
            super::ValueSite::List(locs) => match super::loclist_value(locs, block, inst) {
                Some(s) => s as usize,
                None => continue,
            },
        };
        let Some(&cur) = vm.regs.get(vm.base + base_off + idx) else {
            continue;
        };
        if w.last != cur {
            w.last = cur;
            return Some(());
        }
    }
    None
}

/// The watch — window-range or value — the op at `pc` would trip, as the `(addr, write)` the stop
/// reports (a value watch has no window address and reports `(0, true)`; the DAP maps the variant,
/// not the address). `None` off module 0 (coroutine-child / invoked-unit ops over other windows are
/// out of scope), with nothing armed (the common zero-cost case), or when nothing fires. The **one**
/// pre-op watch scan the debug driver runs (`ScheduledDebugRun::drive`, every verb), so a watch kind
/// exists on every verb at once.
#[allow(clippy::too_many_arguments)]
fn watch_stop_before(
    vm: &Vm,
    mem: &Option<Mem>,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    watchpoints: &[(u64, u64, super::WatchKind)],
    value_watches: &mut [ValueWatchRun],
    pc: super::IrPc,
) -> Option<(u64, bool)> {
    if pc.module != 0 || (watchpoints.is_empty() && value_watches.is_empty()) {
        return None;
    }
    watch_hit_before(
        vm,
        mem,
        funcs,
        fn_block_base,
        watchpoints,
        pc.func,
        pc.block,
        pc.inst,
    )
    .or_else(|| {
        value_watch_hit_before(vm, fn_block_base, value_watches, pc.func, pc.block, pc.inst)
            .map(|()| (0, true))
    })
}

/// Replace an engine's armed value watches (#1229) with `watches`: an id already armed keeps its
/// running `last` (so re-applying the set to arm *another* watch doesn't reset a live one's
/// baseline), while an id new to this run seeds from the target's arm-time baseline (a fresh
/// `seek`-rebuilt run starts them all from the target baseline).
fn merge_value_watches(
    prev: &[ValueWatchRun],
    watches: Vec<(super::WatchId, ValueWatchTarget, super::WatchKind)>,
) -> Vec<ValueWatchRun> {
    watches
        .into_iter()
        .map(|(id, target, kind)| {
            let last = prev
                .iter()
                .find(|w| w.id == id)
                .map_or(target.last, |w| w.last);
            ValueWatchRun {
                id,
                func: target.func,
                site: target.site,
                last,
                kind,
            }
        })
        .collect()
}

/// A debug-session **access sink** (INTERACTIVE_EMBEDDING.md slice 3): observes every module-0
/// memory op the session is about to execute — `(clock-or-turn, task, event)`, with **raw
/// pre-confinement addresses** (the W3 hook-pass vocabulary, [`super::MemEvent`]) — with **no
/// module rewrite**, so the machine view, SSA slots, and the op-clock are identical with a sink
/// installed or absent (invariant 9b: observation never perturbs semantics). Zero cost when
/// absent (callers gate on `Some`). Fed to host-side models (cache/paging/shared-state) by the
/// DAP backend.
pub type AccessSinkFn = Box<dyn FnMut(u64, usize, super::MemEvent) + Send>;

/// The window memory-map introspection tuple — `(page_size, mapped, reserved, explicit-state
/// pages)`, the shape `Mem::map_info` returns (INTERACTIVE_EMBEDDING.md slice 5).
pub type MemMapInfo = (u64, u64, u64, Vec<(u64, u8)>);

/// Build the #750 paged-driver **page-state table** from a window's [`MemMapInfo`]: one byte per
/// page over `[0, coverage)` — `0 = Unmapped`, `1 = Rw`, `2 = Ro` (the emitted check's encoding) —
/// where `coverage` (also returned) is `max(mapped prefix, highest explicit entry end)` in bytes,
/// cut at `backed`, the bytes the window's backing holds ([`Vcpu::win_flat_len`],
/// [`CoopRun::pending_win`]): the emitted access addresses the backing directly, so no page past
/// its end may be admitted, whatever the page map says (a `map` past a fixed backing, #1153).
///
/// This is THE per-emitted-call driver contract for a page-checked run (refresh from
/// [`Vcpu::mem_map_info`], write the table where emitted code can read it, its base to the
/// `"pagestate"` global, and **`coverage` — not the reserved mask-domain size — to `"mapped"`**):
/// the bound check then traps everything above the table exactly where the interpreter (no
/// entries above) faults, and the page states refine within. Returning the coverage alongside the
/// table is what makes the contract hard to get wrong. A `Backed` (§13 region-aliased) page is
/// marked `Unmapped` — fail-closed (the emitted tier cannot read a region's bytes), and
/// unreachable for a paged module anyway (SharedRegion gates the whole module off the paged tier).
pub fn build_pagestate_table(info: &MemMapInfo, backed: u64) -> (Vec<u8>, u64) {
    let (page, mapped, _reserved, entries) = info;
    let top = entries
        .iter()
        .map(|(off, _)| off / page + 1)
        .max()
        .unwrap_or(0)
        .max(mapped / page)
        .min(backed / page);
    let mut t = vec![0u8; top as usize];
    for (i, b) in t.iter_mut().enumerate() {
        if (i as u64) * page < *mapped {
            *b = 1; // Rw default inside the mapped prefix
        }
    }
    for (off, kind) in entries {
        let Some(state) = t.get_mut((off / page) as usize) else {
            continue; // past the backing: outside the table, so the bound check traps it
        };
        // `map_info` kinds: 0 = Ro, 1 = Rw, 2 = Unmapped, 3 = Backed (§13 alias).
        *state = match kind {
            0 => 2,
            1 => 1,
            _ => 0, // Unmapped, and Backed fail-closed (see above)
        };
    }
    let coverage = t.len() as u64 * page;
    (t, coverage)
}

/// Decode + report the op the active continuation is about to execute to `sink` (module-0 ops
/// only, like the watchpoint scan; coroutine-child ops over their own confined windows are out of
/// scope). The decode is the same live-SSA lookup as [`watch_hit_before`]; the event vocabulary
/// and address semantics are the instrumentation pass's, pinned by the `access_sink_diff`
/// differential.
fn emit_access(
    vm: &Vm,
    source: &ModuleSource,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    clock: u64,
    task: usize,
    sink: &mut AccessSinkFn,
) {
    let Some(pc) = vm.cur_ir_pc(source) else {
        return;
    };
    if pc.module != 0 {
        return;
    }
    let Some(ir_inst) = funcs
        .get(pc.func as usize)
        .and_then(|f| f.blocks.get(pc.block))
        .and_then(|b| b.insts.get(pc.inst))
    else {
        return;
    };
    let Some(base_off) = fn_block_base
        .get(pc.func as usize)
        .and_then(|v| v.get(pc.block))
    else {
        return;
    };
    let Some(vals) = vm.regs.get(vm.base + *base_off as usize..) else {
        return;
    };
    if let Some(ev) = super::mem_event_of(ir_inst, vals) {
        sink(clock, task, ev);
    }
}

/// Journal the pre-images of every range the op about to run would **write** (#1557), so stepping
/// backward can undo it instead of replaying to it. Mirrors [`emit_access`]'s decode — module-0 ops
/// only — and takes its spans from [`watch_accesses`](super::watch_accesses), the same per-op analysis
/// the watchpoint check runs, so bulk `mem.copy`/`mem.fill` and v128 stores are covered on the one
/// definition and no store path is touched. Inert while the journal is disarmed.
fn journal_op(
    journal: &mut super::journal::Journal,
    vm: &Vm,
    source: &ModuleSource,
    funcs: &[Func],
    fn_block_base: &[Vec<u32>],
    turn: u64,
    mem: &Option<Mem>,
) {
    if !journal.is_armed() {
        return;
    }
    let Some(m) = mem.as_ref() else {
        return;
    };
    let Some(pc) = vm.cur_ir_pc(source) else {
        return;
    };
    if pc.module != 0 {
        return;
    }
    let Some(ir_inst) = funcs
        .get(pc.func as usize)
        .and_then(|f| f.blocks.get(pc.block))
        .and_then(|b| b.insts.get(pc.inst))
    else {
        return;
    };
    let Some(base_off) = fn_block_base
        .get(pc.func as usize)
        .and_then(|v| v.get(pc.block))
    else {
        return;
    };
    let Some(vals) = vm.regs.get(vm.base + *base_off as usize..) else {
        return;
    };
    for acc in super::watch_accesses(ir_inst, vals, mem) {
        if let super::MemAccess::Range { base, width, write } = acc {
            if write {
                journal.record_write(turn, base, width, m);
            }
        }
    }
}

/// Journal the engine state as it stands **before** the op about to run (#1557): the continuation and
/// the compact host cursor, so `undo_to` can put the run back here without replaying.
///
/// Built from the driver's destructured parts rather than through
/// [`ScheduledDebugRun::build_continuation`], which needs `&self`; the shape it produces is identical.
///
/// **Fail-closed.** Nothing is recorded — so `undo_to` will decline this turn — when the state is not
/// invertible in place: a host using the §3.6 serve queue or holding a capability with opaque declared
/// state (`Host::journal_invertible`), a task stepping inside a §22 invoke (its transient `Vm` is not
/// captured, the same exclusion `checkpointable` makes), or an event-parked fiber (a non-deterministic
/// wall-clock deadline). Those runs still time-travel by checkpoint-plus-replay, exactly as today.
#[allow(clippy::too_many_arguments)]
fn journal_state(
    journal: &mut super::journal::Journal,
    tasks: &[DbgTask],
    extra_envs: &[DbgEnv],
    fibers: &[FiberState],
    source: &ModuleSource,
    host: &Host,
    clock: u64,
    turn: u64,
    policy: &super::journal::JournalPolicy,
) {
    if !journal.is_armed() {
        return;
    }
    // **Segment boundaries, not every op.** The clone below is every task's `Vm`, the fiber chain and
    // the task states; its cost scales with frame depth and it dominated the journal on real guests
    // (`JournalPolicy::state_stride` carries the measurements). Turn 0 is always a boundary so a run
    // can always be undone to its start.
    let stride = policy.state_stride.max(1);
    if !turn.is_multiple_of(stride) {
        return;
    }
    let invertible = host.journal_invertible()
        && tasks.iter().all(|t| t.vt.active_invoke.is_none())
        && !fibers.iter().any(|f| {
            matches!(
                f,
                FiberState::WaitParked { .. }
                    | FiberState::CapParked { .. }
                    | FiberState::HostParked { .. }
            )
        });
    if !invertible {
        return;
    }
    let cont = ScheduledContinuation {
        clock,
        tasks: tasks
            .iter()
            .map(|t| DbgTaskSnapshot {
                active: t.vt.active.clone(),
                active_id: t.vt.active_id,
                chain: t.vt.chain.clone(),
                root_shadow_sp: t.vt.root_shadow_sp,
                threads: t.threads.clone(),
                env: t.env,
                state: t.state.clone(),
                at_bp: t.at_bp,
                lease: t.lease,
            })
            .collect(),
        fibers: fibers.to_vec(),
        extra_envs: extra_envs
            .iter()
            .enumerate()
            .map(|(k, e)| {
                let module = tasks
                    .iter()
                    .find(|t| t.env == Some(k))
                    .map_or(0, |t| t.vt.active.module);
                env_snapshot(e, module)
            })
            .collect(),
        extra_units: source.extra_units(),
    };
    journal.record_state(turn, cont, host.journal_cursor());
}

/// The outcome of advancing a debug session's active continuation by one op ([`debug_advance_fiber`]).
enum FiberStep {
    /// One op ran (a normal op, or a `cont.*` / fiber-return switch) — the clock ticks, keep going.
    Stepped,
    /// The **root** activation of this continuation returned (`chain` empty) — its result.
    Finished(Vec<Value>),
    /// A trap (including a `FiberFault`).
    Trapped(Trap),
    /// A non-fiber seam the caller must apply: `thread.spawn`/`join`, `memory.wait`/`notify`,
    /// `instantiate`, coroutine, tier-up. [`ScheduledDebugRun`] dispatches the ones it schedules
    /// (spawn/join/wait/notify/instantiate) and declines the rest.
    Other(Outcome),
    /// #1366 — the op punted to a host-completed cap: the debug run parks on completion `id`
    /// (result slot `dst`); `at` is the call's own pc (captured before the op advanced), the stop
    /// location the backend reports — the call, not whatever op follows it.
    CapParked {
        id: u64,
        dst: u32,
        at: Option<super::IrPc>,
    },
}

/// Run **one op** of a debug session's active continuation (`vt.active`), applying any §12 fiber switch
/// (`cont.new` registers a fiber in the run-shared `fibers`, `cont.resume` switches into one, `suspend`
/// / a fiber's return switches back). Non-fiber seams are handed back as [`FiberStep::Other`]. The debug
/// counterpart of [`step_vcpu`]'s fiber handling, minus durability (debug runs are non-durable, so no
/// `shadow_switch` / `fiber_sp`). `fibers` is run-shared (a fiber created on one vCPU can be resumed on
/// another — D57 migration) and rebuilt deterministically on a reverse `seek` replay.
fn debug_advance_fiber(
    vt: &mut VTask,
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    // #1366: the op's own pc, before it advances — a host-completed park reports it as the stop
    // location (the call, not whatever op follows it).
    let at = vt.debug_active().cur_ir_pc(source);
    // Step-into a §14 coroutine body: while a coroutine child is the
    // its **own** confined `mem`/`host`/`table`, the op-by-op counterpart of `resume_coro`. Surfacing
    // each child op is what makes breakpoints fire inside the body and the child frame inspectable.
    // Stepping *inside* a §22 `Jit.invoke`d unit (single-vCPU step-into): drive it one op over the
    // caller's shared window/table (the §22 counterpart of `active_coro`), so a breakpoint fires inside.
    if vt.active_invoke.is_some() {
        return step_active_invoke(vt, source, table, fuel, mem, host);
    }
    match vt
        .active
        .resume(source, table, fuel, mem, &mut HostCell::Excl(host), 1)
    {
        Ok(Outcome::Suspended) => FiberStep::Stepped,
        Ok(Outcome::Done(vals)) => match vt.chain.pop() {
            // The root activation finished — the run's result.
            None => FiberStep::Finished(vals),
            // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` to its resumer.
            Some((rid, resumer, rdst)) => {
                fibers[vt.active_id] = FiberState::Done;
                let retval = vals.first().copied().unwrap_or(Value::I64(0));
                // `vcpu.tls` is the vCPU's word: it follows execution (as in `step_vcpu`).
                let tls = vt.active.tls;
                vt.active = resumer;
                vt.active.tls = tls;
                vt.active_id = rid;
                vt.active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                vt.active.set(rdst + 1, Reg::from_value(retval));
                FiberStep::Stepped
            }
        },
        Ok(Outcome::ContNew { funcref, sp, dst }) => {
            if fibers.len() + 1 >= super::MAX_FIBERS {
                return FiberStep::Trapped(Trap::FiberFault);
            }
            let h = fibers.len() as i32;
            fibers.push(FiberState::Pending {
                funcref,
                sp,
                consumed: false,
            });
            vt.active.set(dst, Reg::from_i32(h));
            FiberStep::Stepped
        }
        // This stepper runs where fibers never event-park (it traps on `WaitParked`/`CapParked`
        // below), so the I48 `blocking` flag is a no-op here — a blocking resume of a fiber that
        // only ever suspends/returns behaves exactly like `cont.resume`.
        Ok(Outcome::ContResume {
            kh,
            arg,
            dst,
            blocking: _,
            resume_ip: _,
        }) => {
            let k = kh as usize;
            let target = match fibers.get_mut(k) {
                Some(slot @ FiberState::Pending { .. }) => {
                    let (funcref, sp, consumed) = match std::mem::replace(
                        slot,
                        FiberState::Running {
                            blocking_ip: None,
                            pending: None,
                        },
                    ) {
                        FiberState::Pending {
                            funcref,
                            sp,
                            consumed,
                        } => (funcref, sp, consumed),
                        _ => unreachable!(),
                    };
                    if consumed {
                        *slot = FiberState::Running {
                            blocking_ip: None,
                            pending: Some(arg), // #1538
                        };
                    }
                    let Some((tmod, tfunc, tm)) = resolve_fiber_entry(source, table, funcref)
                    else {
                        return FiberStep::Trapped(Trap::FiberFault);
                    };
                    match Vm::new(&tm, tfunc, &[Value::I64(sp), Value::I64(arg)]) {
                        Ok(mut v) => {
                            v.module = tmod;
                            v
                        }
                        Err(t) => return FiberStep::Trapped(t),
                    }
                }
                Some(slot @ FiberState::Parked { .. }) => {
                    match std::mem::replace(
                        slot,
                        FiberState::Running {
                            blocking_ip: None,
                            pending: None,
                        },
                    ) {
                        FiberState::Parked {
                            mut vm,
                            suspend_dst,
                            consumed: _,
                        } => {
                            vm.set(suspend_dst, Reg::from_i64(arg));
                            vm
                        }
                        _ => unreachable!(),
                    }
                }
                _ => return FiberStep::Trapped(Trap::FiberFault), // forged / Running / Done
            };
            let mut target = target;
            target.tls = vt.active.tls; // `vcpu.tls` follows execution (as in `step_vcpu`)
            let resumer = std::mem::replace(&mut vt.active, target);
            vt.chain.push((vt.active_id, resumer, dst));
            vt.active_id = k;
            FiberStep::Stepped
        }
        Ok(Outcome::FiberSuspend { value, dst }) => {
            // #1538: a thawed consumed fiber's rewound `suspend` returns the queued argument.
            if let Some(arg) = take_pending(fibers, vt.active_id) {
                vt.active.set(dst, Reg::from_i64(arg));
                return FiberStep::Stepped;
            }
            // Pop the resumer to switch back to; an empty chain means the root tried to `suspend`.
            let Some((rid, resumer, rdst)) = vt.chain.pop() else {
                return FiberStep::Trapped(Trap::FiberFault);
            };
            let mut resumer = resumer;
            resumer.tls = vt.active.tls; // `vcpu.tls` follows execution (as in `step_vcpu`)
            let suspended = std::mem::replace(&mut vt.active, resumer);
            fibers[vt.active_id] = FiberState::Parked {
                vm: suspended,
                suspend_dst: dst,
                consumed: !is_unwinding(mem),
            };
            vt.active_id = rid;
            vt.active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
            vt.active.set(rdst + 1, Reg::from_i64(value));
            FiberStep::Stepped
        }
        // §22 guest-JIT install / uninstall / invoke: self-contained host-side ops (they mutate only
        // `vt.active` + the shared dispatch table, spawning no scheduler task), so — like the coroutine
        // arms above — they are serviced **inline** here (the `FiberStep::Other` decline sites never
        // see a Jit outcome). `invoke` runs the unit to completion as a seam-free leaf (stepping *over* it, not
        // into it — matching production `run_invoke`). A forged handle / out-of-coverage unit traps the
        // vCPU (`CapFault`/`Malformed`), exactly as the production `drive`. (DESIGN.md §22 debug tier.)
        Ok(Outcome::JitInstall { h, code, dst }) => {
            match dbg_jit_install(vt, host, source, table, h, code, dst) {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        Ok(Outcome::JitUninstall { h, slot, dst }) => {
            match dbg_jit_uninstall(vt, host, source, table, h, slot, dst) {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        Ok(Outcome::JitInvoke {
            h,
            code,
            argv,
            dst,
            params,
            results,
        }) => {
            // Both debug engines step *into* the invoked unit (#1517 slice 3): this arms
            // `active_invoke` and the next advance steps the unit's first op, so a breakpoint fires
            // inside it and the backtrace descends into its module-≥1 frames.
            match dbg_jit_invoke_step_into(vt, host, source, h, code, &argv, dst, &params, &results)
            {
                Ok(()) => FiberStep::Stepped,
                Err(t) => FiberStep::Trapped(t),
            }
        }
        // F2 — a punted host call in a debug advance keeps the pre-F2 inline wait (the debug
        // drivers' whole-vCPU-park shape is sanctioned tiering, invariant 9 observability
        // corollary; checkpointing across one is already excluded by `checkpoint_safe`'s
        // replay-substate rules — the wait happens inside the advance, leaving no parked state).
        Ok(Outcome::CapPending { id, dst }) => {
            match host.completions().wait_unless_host_owned(id) {
                Some(r) => {
                    vt.active.set(dst, Reg::from_i64(r));
                    FiberStep::Stepped
                }
                // #1366: a host-completed punt — the debug run parks on it (`cap_parked`) and the
                // backend surfaces `StopReason::CapPark`; `deliver_cap` resumes.
                None => FiberStep::CapParked { id, dst, at },
            }
        }
        // Threads / wait / notify / instantiate / (scheduled-engine) separate-module coroutine / tier-up
        // — a scheduler seam the caller applies (`service_advance` dispatches its subset, declines the rest).
        Ok(other) => FiberStep::Other(other),
        Err(t) => FiberStep::Trapped(t),
    }
}

/// A read-only inspection view over **one vCPU's** reified state (`vm`) plus the module's §6 debug
/// metadata. This is the frame-reading engine behind [`ScheduledDebugRun`]'s inspection (every task's
/// `Vm`, and a coroutine child's): given any task's `Vm`, it resolves backtrace frames, block-local
/// SSA values, and named source variables identically — so a thread selected mid-stop (`select_task`)
/// reads its own stack through the exact same code the single-vCPU path uses.
struct FrameReader<'a> {
    vm: &'a Vm,
    source: &'a ModuleSource,
    mem: &'a Option<Mem>,
    debug: Option<&'a DebugInfo>,
    fn_block_base: &'a [Vec<u32>],
    fn_block_types: &'a [Vec<Vec<ValType>>],
    /// The active §14 **separate-module** coroutine's own §6 metadata (module index `>= 1`), set while
    /// stepping inside its body — so a frame in the child resolves against the child's funcs. `None` when
    /// the active continuation is module 0 (the parent or a same-module coroutine), where the module-0
    /// fields above apply. See [`FrameReader::md_for`].
    coro_debug: Option<&'a ModuleDebug>,
    /// The thread has returned. Its `Vm` still rests on the final `return` op — a stop position
    /// since #1713 — so without this a finished run would report that `return` as a live frame. (A
    /// thread that *trapped* keeps its frames: they are where it crashed.)
    finished: bool,
}

/// A resolved write destination (slice 8): a typed absolute regs slot (a promoted SSA scalar) or
/// a confined window address (a memory-located variable).
enum WriteTarget {
    Ssa { reg: usize, ty: ValType },
    Win { addr: u64 },
}

/// A **debugger write scheduled at a clock/turn** (INTERACTIVE_EMBEDDING.md slice 8): re-applied
/// whenever execution passes that clock on **any** path — a live resume and a seek replay reach
/// identical states, which is what keeps the history slider truthful after an edit. `task` names
/// the focused vCPU a `Var` write resolves in on the scheduled engine (ignored single-vCPU).
#[derive(Clone, Debug)]
pub enum ScheduledWrite {
    Window {
        addr: u64,
        bytes: Vec<u8>,
    },
    Var {
        task: usize,
        frame: usize,
        name: String,
        value: i64,
        width: usize,
    },
}

/// Write a debugger edit into the window, journaling its pre-image at `turn` first. An edit is not an
/// op, so [`journal_op`] never sees it; without this an `undo_to` past the edit left the edited bytes
/// in place wherever the guest had not itself stored since (#1871). Journaled before the op at
/// `turn` records its own pre-images, so undo — newest-first — unwinds the op, then the edit.
fn journaled_write(
    journal: &mut super::journal::Journal,
    turn: u64,
    m: &mut Mem,
    addr: u64,
    bytes: &[u8],
) -> bool {
    let Ok(width) = u32::try_from(bytes.len()) else {
        return false;
    };
    let Ok(abs) = m.confine_checked(addr, 0, width) else {
        return false;
    };
    journal.record_write(turn, abs, width, m);
    m.write_bytes(addr, bytes).is_some()
}

/// Coerce + store `value` into the typed regs slot / window target. Best-effort like the live
/// write: an unresolvable or float target is skipped.
fn apply_target(
    target: Option<WriteTarget>,
    value: i64,
    width: usize,
    vm_regs: &mut [Reg],
    mem: &mut Option<Mem>,
    journal: &mut super::journal::Journal,
    turn: u64,
) {
    match target {
        Some(WriteTarget::Ssa { reg, ty }) => {
            let v = match ty {
                ValType::I32 => Value::I32(value as i32),
                ValType::I64 => Value::I64(value),
                _ => return,
            };
            if let Some(r) = vm_regs.get_mut(reg) {
                *r = Reg::from_value(v);
            }
        }
        Some(WriteTarget::Win { addr }) => {
            let w = width.clamp(1, 8);
            if let Some(m) = mem.as_mut() {
                journaled_write(journal, turn, m, addr, &value.to_le_bytes()[..w]);
            }
        }
        None => {}
    }
}

/// Apply every scheduled write due at `turn` to the run's pieces (a `Var` write resolves in its
/// recorded `task`'s frame); `cursor` advances past applied and stale entries (entries below `turn`
/// are inside a restored checkpoint already).
#[allow(clippy::too_many_arguments)]
fn apply_due_writes(
    writes: &[(u64, ScheduledWrite)],
    cursor: &mut usize,
    turn: u64,
    tasks: &mut [DbgTask],
    source: &ModuleSource,
    mem: &mut Option<Mem>,
    debug: Option<&DebugInfo>,
    fn_block_base: &[Vec<u32>],
    fn_block_types: &[Vec<Vec<ValType>>],
    journal: &mut super::journal::Journal,
) {
    while *cursor < writes.len() && writes[*cursor].0 < turn {
        *cursor += 1;
    }
    while *cursor < writes.len() && writes[*cursor].0 == turn {
        match &writes[*cursor].1 {
            ScheduledWrite::Window { addr, bytes } => {
                if let Some(m) = mem.as_mut() {
                    journaled_write(journal, turn, m, *addr, bytes);
                }
            }
            ScheduledWrite::Var {
                task,
                frame,
                name,
                value,
                width,
            } => {
                if let Some(t) = tasks.get_mut(*task) {
                    {
                        let target = FrameReader {
                            vm: &t.vt.active,
                            source,
                            mem: &*mem,
                            debug,
                            fn_block_base,
                            fn_block_types,
                            coro_debug: None,
                            finished: matches!(t.state, DbgTaskState::Done(Ok(_))),
                        }
                        .write_target(*frame, name);
                        apply_target(
                            target,
                            *value,
                            *width,
                            &mut t.vt.active.regs,
                            mem,
                            journal,
                            turn,
                        );
                    }
                }
            }
        }
        *cursor += 1;
    }
}

impl<'a> FrameReader<'a> {
    /// Call-stack depth (running activation + suspended callers).
    fn depth(&self) -> usize {
        self.vm.stack.len() + 1
    }

    /// The `(§6 debug, per-block slot base, per-block value types)` to read a frame in `module` against:
    /// module 0's from the session fields, or an active separate-module coroutine's own. `None` for any
    /// other module (e.g. an installed §22 unit whose frames carry no debug tables here) — such a frame is
    /// not source-inspectable, matching the pre-slice module-0 gate.
    #[allow(clippy::type_complexity)]
    fn md_for(
        &self,
        module: usize,
    ) -> Option<(
        Option<&'a DebugInfo>,
        &'a [Vec<u32>],
        &'a [Vec<Vec<ValType>>],
    )> {
        if module == 0 {
            return Some((self.debug, self.fn_block_base, self.fn_block_types));
        }
        let md = self.coro_debug?;
        (md.module == module).then_some((md.debug.as_ref(), &md.fn_block_base, &md.fn_block_types))
    }

    /// The `(module, func, block, inst, window base)` of the frame `depth` levels from the top (0 =
    /// running activation; each caller resolved at its call site, `resume_pc - 1`). `None` past the
    /// stack or when the top is paused on a non-instruction.
    fn frame_at(&self, depth: usize) -> Option<(usize, usize, usize, usize, usize)> {
        if depth == 0 {
            if self.finished {
                return None;
            }
            let pc = self.vm.cur_ir_pc(self.source)?;
            return Some((self.vm.module, self.vm.cur, pc.block, pc.inst, self.vm.base));
        }
        let n = self.vm.stack.len();
        let &(module, f, base, resume_pc, _) = self.vm.stack.get(n.checked_sub(depth)?)?;
        let cm = self.source.get(module)?;
        let (block, inst) = cm
            .progs
            .get(f)?
            .src
            .get(resume_pc.checked_sub(1)?)
            .copied()
            .flatten()?;
        Some((module, f, block as usize, inst as usize, base))
    }

    /// The `IrPc` of the frame `depth` levels from the top.
    fn frame_pc(&self, depth: usize) -> Option<super::IrPc> {
        let (module, func, block, inst, _) = self.frame_at(depth)?;
        Some(super::IrPc {
            module: module as u32,
            func: func as FuncIdx,
            block,
            inst,
        })
    }

    /// Block-local SSA value `idx` in the frame `depth` levels from the top, typed against that frame's
    /// module (module 0's, or an active separate-module coroutine's own).
    fn value_in_frame(&self, depth: usize, idx: usize) -> Option<Value> {
        let (module, func, block, _inst, base) = self.frame_at(depth)?;
        let (_, fn_block_base, fn_block_types) = self.md_for(module)?;
        let off = *fn_block_base.get(func)?.get(block)? as usize;
        let ty = *fn_block_types.get(func)?.get(block)?.get(idx)?;
        Some(self.vm.regs[base + off + idx].to_value(ty))
    }

    /// Where a **write** to source variable `name` in frame `depth` lands (slice 8): the absolute
    /// regs slot + type for a promoted SSA scalar, or the confined window address for a
    /// memory-located var — the write-side mirror of [`FrameReader::read_var`]'s resolution.
    fn write_target(&self, depth: usize, name: &str) -> Option<WriteTarget> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let (di, fn_block_base, fn_block_types) = self.md_for(module)?;
        let var = super::pick_var(di?, func as FuncIdx, name, block, inst)?;
        let off = *fn_block_base.get(func)?.get(block)? as usize;
        let slot = |idx: usize| -> Option<WriteTarget> {
            let ty = *fn_block_types.get(func)?.get(block)?.get(idx)?;
            Some(WriteTarget::Ssa {
                reg: base + off + idx,
                ty,
            })
        };
        match &var.loc {
            VarLoc::Ssa { value } => slot(*value as usize),
            VarLoc::SsaList(locs) => slot(super::loclist_value(locs, block, inst)? as usize),
            VarLoc::Window { off: o } => Some(WriteTarget::Win {
                addr: (self.vm.regs[base].i64() as u64).wrapping_add(*o as u64),
            }),
            VarLoc::WindowVia { base: locs, off: o } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                Some(WriteTarget::Win {
                    addr: addr.wrapping_add(*o as u64),
                })
            }
            VarLoc::Fixed { addr } => Some(WriteTarget::Win { addr: *addr }),
            VarLoc::Tls { off } => Some(WriteTarget::Win {
                addr: di?.tls_addr(self.vm.tls, *off)?,
            }),
        }
    }

    /// Read a source variable by name in the frame `depth` levels from the top, resolving its `VarLoc`
    /// over that frame's module's §6 debug info (SSA slot / window / fixed) — module 0's, or an active
    /// separate-module coroutine's own. `None` if unresolvable here.
    fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let di = self.md_for(module)?.0?;
        let var = super::pick_var(di, func as FuncIdx, name, block, inst)?;
        let window_read = |addr: u64| -> Option<VarValue> {
            Some(VarValue::Bytes(
                self.mem.as_ref()?.read_window(addr, width).ok()?,
            ))
        };
        match &var.loc {
            VarLoc::Ssa { value } => self
                .value_in_frame(depth, *value as usize)
                .map(VarValue::Value),
            VarLoc::SsaList(locs) => {
                let v = super::loclist_value(locs, block, inst)?;
                self.value_in_frame(depth, v as usize).map(VarValue::Value)
            }
            // Address = data-SP (the frame's first value, v0) + off.
            VarLoc::Window { off } => {
                window_read((self.vm.regs[base].i64() as u64).wrapping_add(*off as u64))
            }
            VarLoc::WindowVia { base: locs, off } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                window_read(addr.wrapping_add(*off as u64))
            }
            VarLoc::Fixed { addr } => window_read(*addr),
            // A thread-local: this thread's own block (#1715).
            VarLoc::Tls { off } => window_read(di.tls_addr(self.vm.tls, *off)?),
        }
    }

    /// The window address of a memory-located source variable by name in the frame `depth` from the
    /// top; `None` for a promoted SSA scalar (no address) or an unresolvable name. Resolves against that
    /// frame's module's §6 info (module 0's, or an active separate-module coroutine's own).
    fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let di = self.md_for(module)?.0?;
        let var = super::pick_var(di, func as FuncIdx, name, block, inst)?;
        match &var.loc {
            VarLoc::Ssa { .. } | VarLoc::SsaList(_) => None,
            VarLoc::Window { off } => {
                Some((self.vm.regs[base].i64() as u64).wrapping_add(*off as u64))
            }
            VarLoc::WindowVia { base: locs, off } => {
                let v = super::loclist_value(locs, block, inst)?;
                let addr = match self.value_in_frame(depth, v as usize)? {
                    Value::I32(x) => x as i64 as u64,
                    Value::I64(x) => x as u64,
                    _ => return None,
                };
                Some(addr.wrapping_add(*off as u64))
            }
            VarLoc::Fixed { addr } => Some(*addr),
            VarLoc::Tls { off } => di.tls_addr(self.vm.tls, *off),
        }
    }

    /// Resolve a source variable held in an **SSA value** (no window address) to a value-watch
    /// target (#1229): its owning function, how to find its holding value at any pc, and the current
    /// (arm-time) holding value as the change baseline. `None` for a memory-located var (watch it by
    /// address instead), an unknown name, or a var not live at this frame's pc.
    fn value_watch_target(&self, depth: usize, name: &str) -> Option<(u32, super::ValueSite, Reg)> {
        let (module, func, block, inst, base) = self.frame_at(depth)?;
        let di = self.md_for(module)?.0?;
        let var = super::pick_var(di, func as FuncIdx, name, block, inst)?;
        let (site, idx) = match &var.loc {
            VarLoc::Ssa { value } => (super::ValueSite::Single(*value), *value as usize),
            VarLoc::SsaList(locs) => (
                super::ValueSite::List(locs.clone()),
                super::loclist_value(locs, block, inst)? as usize,
            ),
            // Memory-located: watchable by address (the window-range watch), not by value.
            VarLoc::Window { .. }
            | VarLoc::WindowVia { .. }
            | VarLoc::Fixed { .. }
            | VarLoc::Tls { .. } => return None,
        };
        let off = *self.md_for(module)?.1.get(func)?.get(block)? as usize;
        let last = *self.vm.regs.get(base + off + idx)?;
        Some((func as u32, site, last))
    }
}

/// Whether `m` can spawn a second vCPU — it contains a `thread.spawn` op somewhere. A spawn-free
/// module runs on [`ScheduledDebugRun`] as a one-task schedule (every verb, watch, and checkpoint
/// reads identically); the predicate remains for callers that key a *policy* on it (a schedule seed
/// is meaningless with one vCPU).
pub fn module_spawns_threads(m: &Module) -> bool {
    m.funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .flat_map(|b| b.insts.iter())
        .any(|i| matches!(i, Inst::ThreadSpawn { .. }))
}

/// The outcome of one [`ScheduledDebugRun`] pump — the multi-vCPU counterpart of a `DebugRun` stop.
#[derive(Debug)]
pub enum SchedStop {
    /// A stop fired in some thread; that thread is now the stopped + focused one
    /// ([`stopped_task`](ScheduledDebugRun::stopped_task)). `reason` says why.
    Break { pc: super::IrPc, reason: SchedBreak },
    /// The root vCPU finished — the run's result (or trap).
    Finished(Result<Vec<Value>, Trap>),
    /// No thread is runnable and the root hasn't finished: a `memory.wait`/deadlock the debug
    /// scheduler can't advance (it drives only `thread.spawn`/`join`).
    Blocked,
    /// #1146 (deeper) — every live thread is parked and at least one of them in a **blocking-stdin
    /// `read`** (W4): the run is live and resumable, that thread is now the stopped + focused one
    /// (`pc` = its read), and [`provide_stdin`](ScheduledDebugRun::provide_stdin) + a resume
    /// re-issues the read. Distinct from `Blocked` (a true deadlock) so the backend can show an
    /// input prompt instead of a dead end.
    StdinPark { pc: super::IrPc },
    /// #1366 — a thread is parked on a host-completed cap call `id` (at `pc`): live, resumable once
    /// the embedder [`deliver_cap`](ScheduledDebugRun::deliver_cap)s the value. The cap twin of
    /// `StdinPark`.
    CapPark { id: u64, pc: super::IrPc },
    /// A thread reached an op outside the debug scheduler's subset — only JIT tier-up (never enabled on
    /// this engine). Threads, `wait`/`notify`, fibers, `instantiate`/`instantiate_module`, and §14
    /// coroutines (step-into, with the coroutine's vCPU pinned across the body) are all handled.
    Declined,
}

/// Why a [`SchedStop::Break`] fired — mapped to the DAP stop reason by the backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedBreak {
    /// A pc in the run-shared breakpoint set (in whichever thread reached it).
    Breakpoint,
    /// A window watchpoint: the about-to-run op touches `[addr, addr+len)` with a matching kind. The
    /// stop is *before* the access applies.
    Watchpoint { addr: u64, write: bool },
    /// A single-step / step-over / step-out landed the stepping thread at its target.
    Step,
    /// A budgeted run ([`ScheduledDebugRun::run_until_turn`]) reached its turn without another stop.
    /// A live stop like any other, so an embedder can run a long program in slices — streaming its
    /// output and honoring a Pause between them — and resume where it left off.
    Pause,
}

/// One scheduled vCPU under the multi-vCPU debugger. `Clone` for time-travel checkpointing (W1): a
/// scheduled checkpoint captures each task's state so a reverse `seek` restarts from the nearest
/// snapshot instead of turn 0.
#[derive(Clone)]
enum DbgTaskState {
    Runnable,
    /// Parked on `thread.join` of task `child` (handle `slot`); its result lands at `dst` on wake.
    BlockedJoin {
        child: usize,
        slot: usize,
        dst: u32,
    },
    /// Parked on `memory.wait` at address `addr` of its own window until a `memory.notify` whose
    /// futex key matches, or the logical `clock` reaches `deadline`; the status (`WAIT_WOKEN` /
    /// `WAIT_TIMED_OUT`) lands at `dst`. The key is computed at notify time, from the windows as they
    /// are then, so it survives a `restore` that rebuilds a child's window.
    BlockedWait {
        addr: u64,
        /// Logical-clock deadline, or `None` for an infinite wait — which is therefore not a
        /// clock-advance candidate in `dbg_pick_runnable` (#1638).
        deadline: Option<u64>,
        dst: u32,
    },
    /// #1146 (deeper) — parked in a blocking **`Stream{In}` read** on an exhausted stdin under
    /// [`super::Host::set_stdin_blocking`] ([`Outcome::StdinPark`]); the read op was rewound and the
    /// turn did not tick (the park is not a visible op — the re-issued read is). Woken explicitly by
    /// [`ScheduledDebugRun::provide_stdin`]; a [`restore`](ScheduledDebugRun::restore) re-admits it
    /// (a restored run is not parked — the re-executed read is served from the cap tape, or re-parks
    /// at the frontier). The debug-scheduler twin of the cooperative driver's `TaskState::BlockedStdin`
    /// — invariant 14's debugger axis.
    BlockedStdin,
    /// #1366 — parked on a **host-completed** cap call: the embedder services it asynchronously and
    /// [`ScheduledDebugRun::deliver_cap`] resumes. `dst` is the awaiting result slot; `at` the call's
    /// pc (the stop location). Not runnable until
    /// delivered, and cleared to `Runnable` on a checkpoint restore (a replay serves the call from
    /// the tape, exactly as a restored stdin park re-issues its read).
    CapParked {
        id: u64,
        dst: u32,
        at: Option<super::IrPc>,
    },
    /// Finished — result (or trap) retained for a joiner.
    Done(Result<Vec<Value>, Trap>),
}

struct DbgTask {
    /// The reified continuation of this vCPU: its active `Vm` plus its §12 fiber resume `chain` (a
    /// `cont.resume` switches `vt.active` into a fiber; `suspend` / a fiber return switches back).
    vt: VTask,
    /// This vCPU's `thread.spawn` / `instantiate` children (handle = index → global task index; `None`
    /// = joined). Both seams share one handle namespace, so `Instantiator.join` (→ `ThreadJoin`) joins
    /// an `instantiate` child through the same machinery.
    threads: Vec<Option<usize>>,
    /// The runtime environment this vCPU steps against: `None` = the shared domain (root + its
    /// `thread.spawn` siblings, over `self.mem`/`self.host`/`self.table`); `Some(k)` = the confined
    /// [`extra_envs`](ScheduledDebugRun::extra_envs)`[k]` of a §14 `instantiate` child. The debug-engine
    /// counterpart of the production [`TaskSlot::env`].
    env: Option<usize>,
    state: DbgTaskState,
    /// Paused on a just-reported breakpoint — step one op past it before the next scan makes progress
    /// (so a loop-body breakpoint re-fires each iteration).
    at_bp: bool,
    /// A detached child's window lease `(spawner env, budget, bytes)` — [`TaskSlot::lease`]'s
    /// counterpart, returned by [`dbg_refund_ended_windows`] once this task is `Done`.
    lease: Option<(Option<usize>, i32, u64)>,
}

/// A §14 `instantiate` **confined executor child**'s runtime under the multi-vCPU debug scheduler — the
/// debug-engine counterpart of the production [`ChildEnv`]. Its `mem` is a `nested_view` sub-window
/// sharing the parent's backing (the confinement masking is the production primitive, unchanged — the
/// debugger only drives an already-confined child op-by-op), `host` an attenuated powerbox (an
/// `Instantiator` + `AddressSpace`, each over `[0, child_size)`), `table` a fresh natural dispatch table
/// over module 0 (no installed §22 units), and `fuel` a sub-allocated quota. Plain `Host` (the debug
/// scheduler is single-threaded and the §3.6 live-call/serve machinery is not yet driven here).
/// [`refund_ended_windows`] on the debugger's scheduler: a finished detached child's window goes back
/// to the budget that paid for it, in the spawner's own powerbox.
fn dbg_refund_ended_windows(tasks: &mut [DbgTask], host: &mut Host, envs: &mut [DbgEnv]) {
    for t in tasks.iter_mut() {
        if matches!(t.state, DbgTaskState::Done(_)) {
            if let Some((env, budget, bytes)) = t.lease.take() {
                match env {
                    None => host.budget_mem_give(budget, bytes),
                    Some(k) => envs[k].host.budget_mem_give(budget, bytes),
                }
            }
        }
    }
}

struct DbgEnv {
    mem: Option<Mem>,
    host: Host,
    table: SharedSlots,
    fuel: u64,
    /// The child domain's own §12 fiber registry: each domain numbers its fibers from 0 and cannot
    /// reach another's, as on the oracle (the root's is [`ScheduledDebugRun::fibers`]).
    fibers: Vec<FiberState>,
}

/// One scheduled vCPU's captured state inside a [`ScheduledSnapshot`] — its full `VTask` continuation
/// (active `Vm`, active fiber id, resume `chain`, same-module coroutine children, active-coroutine
/// cursor, durable shadow-SP), the join-handle table, the env index (`Some(k)` for a §14 `instantiate`
/// child — see [`ScheduledSnapshot::extra_envs`]), the run state, and the breakpoint-skip flag. The
/// fiber `Vm`s share the run's window(s), and each coroutine's window is a `nested_view` sharing the
/// parent backing, so their bytes ride in the snapshot's window bytes; `restore` rebuilds the `VTask`.
#[derive(Clone)]
struct DbgTaskSnapshot {
    active: Vm,
    active_id: usize,
    chain: Vec<(usize, Vm, u32)>,
    root_shadow_sp: u64,
    threads: Vec<Option<usize>>,
    env: Option<usize>,
    state: DbgTaskState,
    at_bp: bool,
    lease: Option<(Option<usize>, i32, u64)>,
}

/// A multi-vCPU time-travel **checkpoint** (DEBUGGING.md W1): the re-executable state of a
/// [`ScheduledDebugRun`] at global [`turn`](ScheduledSnapshot::turn), so a reverse `seek`/`step_back`
/// restarts a replay here instead of from turn 0. Captured only for the simple threaded subset (see
/// [`ScheduledDebugRun::checkpointable`]: no §12 fibers, no §14 coroutines/`instantiate` children, a
/// pristine shared window, a restorable host) — where the per-task active `Vm`s + the shared window
/// bytes + the host substate + the scheduler clocks fully determine the continuation. The scheduled
/// Opaque to the DAP backend, which stores the moment in a
/// ladder keyed on the global turn and hands it back to [`ScheduledDebugRun::restore`].
#[derive(Clone)]
pub struct ScheduledContinuation {
    /// The scheduled-mode op clock (visible ops across all vCPUs) — continuation state, unlike the
    /// turn, which is the ladder's key.
    clock: u64,
    tasks: Vec<DbgTaskSnapshot>,
    /// The **run-shared** §12 fiber registry (one handle namespace across all vCPUs — a fiber migrates,
    /// D57), reconstructed verbatim on restore. The parked fiber `Vm`s share the run's window.
    fibers: Vec<FiberState>,
    /// The §14 `instantiate`-child environments (handle = index; a task's [`DbgTask::env`] indexes this).
    /// See [`EnvSnapshot`].
    extra_envs: Vec<EnvSnapshot>,
    /// The non-primary [`ModuleSource`] units (a §14 **separate-module** `instantiate_module` child's or
    /// coroutine's pushed program), re-pushed on restore so a `module >= 1` frame resolves. Empty for a
    /// same-module-only run. Cheap `Arc` clones — the compiled units are immutable.
    extra_units: Vec<std::sync::Arc<Compiled>>,
}

impl ScheduledContinuation {
    /// An empty scheduled continuation — for [`moment`](super::moment)'s continuation-agnostic ladder
    /// test, which needs a `Continuation::Bytecode` payload but never restores it (the ladder treats
    /// the continuation opaquely). Not a valid resume state; only the ladder's variant-independence
    /// rests on it.
    #[cfg(test)]
    pub fn empty_for_test() -> ScheduledContinuation {
        ScheduledContinuation {
            clock: 0,
            tasks: Vec::new(),
            fibers: Vec::new(),
            extra_envs: Vec::new(),
            extra_units: Vec::new(),
        }
    }
}

/// A multi-vCPU checkpoint: [`ScheduledContinuation`] plus the shared window image (all tasks share the
/// one window; capturing its protection map admits a **page-mapping** run) and the host substate.
pub type ScheduledSnapshot = super::moment::Moment;

/// A §14 `instantiate`-child environment ([`DbgEnv`]) inside a [`ScheduledSnapshot`]. Like a coroutine
/// child, its window is a `nested_view` sharing the root backing region — so its bytes ride in the
/// snapshot's window bytes and only the view geometry (`win_base`/`size_log2`) is stored; `restore`
/// rebuilds the view over the reseeded shared window. Its attenuated powerbox (an `Instantiator` +
/// `AddressSpace`, each over `[0, child_size)`) is deterministic in `child_size = 1 << size_log2`, so
/// only the host replay substate is carried; a natural dispatch table over the child's [`module`] (a
/// **separate-module** `instantiate_module` child's pushed unit rides in `extra_units`) and the
/// sub-allocated `fuel` complete it. The child's own page-protection map (`prot`) is captured alongside,
/// admitting a child that `map`/`unmap`/`protect`ed its own window; its bytes still ride in the shared
/// snapshot. A §13 region-aliased child stays outside the checkpointable subset.
#[derive(Clone)]
struct EnvSnapshot {
    win_base: u64,
    size_log2: u8,
    /// The child's module in the shared source (`0` = same-module `instantiate`; `>= 1` = a
    /// **separate-module** `instantiate_module` child, whose pushed unit is captured in the run
    /// snapshot's `extra_units`). Its natural dispatch table is rebuilt over this index.
    module: usize,
    host: super::HostReplaySubstate,
    fuel: u64,
    /// The child's own page-protection map ([`Mem::prot_snapshot`]), reinstalled with
    /// [`Mem::install_prot`] on restore (its bytes ride in the shared snapshot). Empty for a pristine child.
    prot: Vec<(u64, super::PageProt)>,
    /// The child's fiber registry (its `Vm`s' bytes ride in the shared snapshot, like the root's).
    fibers: Vec<FiberState>,
}

/// A **multi-vCPU** debug session on the bytecode engine (DEBUGGING.md Milestone B, bytecode side): a
/// deterministic cooperative debug scheduler over one shared `Mem` for a `thread.spawn`/`join` guest.
/// Mirrors the tree-walker's [`Inspector::attach_scheduled`](crate::Inspector) — a run-shared breakpoint
/// set fires in **whichever** vCPU reaches it (stopping *before* the op), `stopped_task` reports which,
/// and `select_task` focuses read-inspection (backtrace / `read_var` / `read_window`) on any live thread
/// while stopped in another. The schedule is a reproducible lowest-index-runnable, one-op-per-turn pick
/// (the debuggable analogue of the production `drive`), so the interleaving is deterministic — which is
/// what makes **reverse debugging** (`tick`-replay to a global `turn`) and **cross-thread watchpoints**
/// (the per-op seam checks the armed ranges in whichever thread) sound. Stepping is depth-aware
/// (in/over/out), **`memory.wait`/`notify`** park/wake threads (a stuck set advances a logical `clock`
/// to the earliest wait deadline, exactly as the production `drive`), and **§12 fibers** switch each
/// vCPU's active continuation (breakpoints fire inside a resumed fiber; the fiber registry is run-shared
/// so a fiber migrates across vCPUs — D57), and **§14 `instantiate` / `instantiate_module`** spawn a
/// confined executor child as its own scheduled vCPU (its own [`DbgEnv`] — window / powerbox / quota —
/// joinable through the shared thread machinery; a separate-module child runs its own pushed module,
/// and nesting composes to any depth), and **§14 coroutines** are stepped op-by-op (step-into) with the
/// coroutine's vCPU **pinned** across the body so a `resume` stays atomic w.r.t. other vCPUs. The only
/// op outside the subset (→ [`SchedStop::Declined`]) is JIT tier-up (never enabled here).
pub struct ScheduledDebugRun {
    source: std::sync::Arc<ModuleSource>,
    table: SharedSlots,
    mem: Option<Mem>,
    host: Host,
    tasks: Vec<DbgTask>,
    /// §14 `instantiate` confined children's environments (handle = index; a task's [`DbgTask::env`] is
    /// `Some(k)` into this). Grown as children spawn; rebuilt deterministically on a reverse-`seek`
    /// replay. Not torn down mid-run (a finished child's env is inert — revocation semantics deferred).
    extra_envs: Vec<DbgEnv>,
    /// The root domain's §12 fiber registry (one handle namespace across its vCPUs; a fiber created on
    /// one can be resumed on another — D57). Each §14 child has its own ([`DbgEnv::fibers`]). Rebuilt
    /// deterministically on a reverse `seek` replay.
    fibers: Vec<FiberState>,
    fn_block_base: Vec<Vec<u32>>,
    fn_block_types: Vec<Vec<Vec<ValType>>>,
    debug: Option<DebugInfo>,
    /// The IR functions, for computing the effective address of the op about to run when a watchpoint
    /// is armed (`watch_hit_before`). `Arc` so a reverse `seek` rebuild is cheap.
    funcs: std::sync::Arc<[Func]>,
    breakpoints: Vec<super::IrPc>,
    /// Run-shared window watchpoints (DEBUGGING.md W2, cross-thread): `(addr, len, kind)`. Empty in the
    /// common case, so the per-op `access_of` computation is skipped entirely.
    watchpoints: Vec<(u64, u64, super::WatchKind)>,
    /// The **undo journal** (#1556/#1557): pre-images of the window ranges each op overwrites, so
    /// `step_back` can undo rather than restore-and-replay. Disarmed by default and inert then, so a
    /// session that never arms it pays one boolean test per op (INVARIANTS #9b).
    journal: super::journal::Journal,
    /// The journal's static retention policy (#1558) — how much history stays at level 1, and the
    /// ceiling past which the oldest is dropped. Consulted once per op while armed.
    journal_policy: super::journal::JournalPolicy,
    /// Run-shared **value watchpoints** (#1229, #1517 slice 2): stop when a watched SSA-held source
    /// variable's holding value changes, in whichever thread's top frame runs the variable's function
    /// — cross-thread exactly like `watchpoints` (a target is `(func, site)`, not a task). Empty in
    /// the common case.
    value_watches: Vec<ValueWatchRun>,
    /// The session's optional per-op access sink ([`AccessSinkFn`]) — fired before every module-0
    /// op with the global `turn` and the **executing task index** (the vCPU attribution host-side
    /// models key on). `None` (the default) is zero-cost; the DAP backend re-installs it on every
    /// `seek` rebuild, like watchpoints.
    access_sink: Option<AccessSinkFn>,
    /// The optional **scheduler trace tape** ([`SchedTraceEvent`], slice 6) — armed by
    /// [`set_sched_trace`](ScheduledDebugRun::set_sched_trace); `None` (the default) is zero-cost.
    /// Re-armed by the DAP backend on `seek` rebuilds (the replay refills it deterministically).
    sched_trace: Option<Vec<SchedTraceEvent>>,
    /// Scheduled debugger writes ([`ScheduledWrite`], slice 8), sorted by turn, + the next-un-applied
    /// cursor. Empty (the default) is one index compare per advance; the DAP backend re-installs the
    /// list on every rebuild.
    scheduled_writes: Vec<(u64, ScheduledWrite)>,
    write_cursor: usize,
    /// The **seeded pick** (slice 7): `Some(seed)` chooses uniformly among the runnable set via
    /// `splitmix64(seed ^ turn)` — an adversarial-variation knob whose choice is a pure function
    /// of `(seed, turn)`, so replay reproduces it with no captured scheduler state. `None` (the
    /// default) keeps the original lowest-index pick.
    sched_seed: Option<u64>,
    /// Recorded **forced switches** (slice 7): concrete `(turn, task)` overrides, resolved at
    /// record time and re-applied by the DAP backend on rebuilds — so a `seek` replays them at the
    /// identical turns. Empty (the default) is zero-cost.
    forced: Vec<(u64, usize)>,
    /// Recorded **step spans** (#1942): the turns each step drove, its thread, and whether it kept
    /// the other threads frozen. The policy pick would not reproduce a step's schedule, so a replay
    /// picks by the span over those turns. Sorted and disjoint; the DAP backend re-installs the list
    /// on every rebuild, like `forced`.
    step_spans: Vec<StepSpan>,
    /// The scheduler-event sink (#1987), if one is installed ([`set_sched_sink`]).
    ///
    /// [`set_sched_sink`]: Self::set_sched_sink
    sched_sink: Option<SchedSinkFn>,
    /// Whether the next step keeps the other threads frozen ([`set_single_thread`]).
    ///
    /// [`set_single_thread`]: Self::set_single_thread
    single_thread: bool,
    /// Set when `drive` stopped *before* an op that hits a watchpoint (the access hasn't applied yet);
    /// taken by the backend to report `StopReason::Watchpoint`.
    last_watch: Option<(u64, bool)>,
    /// The task index paused on a breakpoint (stepping drives it); `None` while running.
    stopped: Option<usize>,
    /// The task `select_task` focuses read-inspection on; reset to the stopped thread on each stop.
    focus: usize,
    /// Global count of visible ops executed across all vCPUs — the scheduled-mode logical clock and the
    /// reverse-`seek` coordinate.
    turn: u64,
    /// The `memory.wait` deadline clock (advanced only when the whole run is stuck-waiting, to the
    /// earliest deadline). Separate from `turn`: it measures futex timeout time, not ops.
    clock: u64,
}

/// Mark task `ti` done and wake any joiner parked on it (delivering its result / propagating a trap) —
/// the debug-scheduler counterpart of the production [`complete`].
/// One record on the **scheduler trace tape** (INTERACTIVE_EMBEDDING.md slice 6): the cooperative
/// debug scheduler's own decisions — turns, parks, wakes with both identities, spawns —
/// observation-only, derived by diffing task states across each decision (no scheduling logic is
/// touched; invariant 4 holds — the host records, never chooses differently). `turn` is the global
/// turn at the decision. The tape is deterministic: the same run replayed yields the identical
/// tape (the schedule itself is deterministic), which is what makes it a sound timeline source.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SchedTraceEvent {
    /// `task` ran the op at global `turn`.
    Turn { turn: u64, task: usize },
    /// `task` parked joining `child` (`thread.join` on a live child).
    ParkJoin {
        turn: u64,
        task: usize,
        child: usize,
    },
    /// `task` parked on the futex word at window offset `key` (`memory.wait`).
    ParkWait { turn: u64, task: usize, key: u64 },
    /// `waker`'s `memory.notify` woke `wakee` off its futex wait.
    WakeNotify {
        turn: u64,
        waker: usize,
        wakee: usize,
    },
    /// `waker`'s completion (thread exit) woke joiner `wakee`.
    WakeJoin {
        turn: u64,
        waker: usize,
        wakee: usize,
    },
    /// `task` woke from a timed `memory.wait` whose deadline passed (the picker's clock advance).
    WakeTimeout { turn: u64, task: usize },
    /// `parent`'s `thread.spawn` created `task`.
    Spawn {
        turn: u64,
        parent: usize,
        task: usize,
    },
    /// `task` entered function `func` of module `module` (#1981): the op at `turn` deepened its call
    /// stack, or `turn` spawned it (its entry function). One per frame entered.
    Call {
        turn: u64,
        task: usize,
        module: usize,
        func: usize,
    },
    /// `task` left its innermost function at `turn` (#1981): a return, or the task finishing. One per
    /// frame left, so a `longjmp` that drops three frames records three.
    Return { turn: u64, task: usize },
    /// `task`'s `thread.join` of `child` completed at `turn` without parking — `child` had already
    /// finished (#1987). A join that parked completes with `WakeJoin` instead.
    Join {
        turn: u64,
        task: usize,
        child: usize,
    },
}

/// The **scheduler-event sink** (#1987): every [`SchedTraceEvent`] as the engine produces it, live and
/// in order with the access sink — what a model that needs thread lifecycle (a race detector's spawn
/// and join edges) consumes, without arming the trace tape.
pub type SchedSinkFn = Box<dyn FnMut(&SchedTraceEvent) + Send>;

/// Hand scheduler events to whoever is listening: the trace tape when armed, the scheduler-event
/// sink when installed.
fn record_sched(
    trace: &mut Option<Vec<SchedTraceEvent>>,
    sink: &mut Option<SchedSinkFn>,
    events: Vec<SchedTraceEvent>,
) {
    if let Some(sink) = sink.as_mut() {
        for e in &events {
            sink(e);
        }
    }
    if let Some(trace) = trace.as_mut() {
        trace.extend(events);
    }
}

/// How many call frames `task` has open: its active activation plus the suspended callers, or none
/// once it has finished (#1981).
fn trace_depth(task: &DbgTask) -> usize {
    match task.state {
        DbgTaskState::Done(_) => 0,
        _ => task.vt.active.stack.len() + 1,
    }
}

/// The `(module, func)` of `task`'s frame `level` from the bottom (`0` = its entry function).
fn trace_frame(task: &DbgTask, level: usize) -> (usize, usize) {
    let vm = &task.vt.active;
    match vm.stack.get(level) {
        Some(&(module, func, ..)) => (module, func),
        None => (vm.module, vm.cur),
    }
}

/// A compact `(state-tag, aux)` per task for the trace differ: 0 = runnable, 1 = blocked-join
/// (aux = child), 2 = blocked-wait (aux = key), 3 = done, 4 = blocked-stdin (a park/wake the differ
/// records as no timeline edge: the wake is the embedder's `provide_stdin`, not another task's act),
/// 5 = cap-parked (aux = completion id; likewise no edge — the wake is the embedder's `deliver_cap`).
fn trace_tags(tasks: &[DbgTask]) -> Vec<(u8, u64)> {
    tasks
        .iter()
        .map(|t| match t.state {
            DbgTaskState::Runnable => (0, 0),
            DbgTaskState::BlockedJoin { child, .. } => (1, child as u64),
            DbgTaskState::BlockedWait { addr, .. } => (2, addr),
            DbgTaskState::Done(_) => (3, 0),
            DbgTaskState::BlockedStdin => (4, 0),
            DbgTaskState::CapParked { id, .. } => (5, id),
        })
        .collect()
}

/// Diff task states across one advance by `actor` at `turn`, appending the park/wake/spawn events
/// the transition implies. A task beyond `before`'s length is a fresh spawn by the actor.
fn trace_diff(
    before: &[(u8, u64)],
    actor_depth: usize,
    actor_threads: &[Option<usize>],
    tasks: &[DbgTask],
    turn: u64,
    actor: usize,
    out: &mut Vec<SchedTraceEvent>,
) {
    // #1987: a join handle the op consumed without parking — the child had already finished.
    for (slot, child) in actor_threads.iter().enumerate() {
        if let (Some(child), Some(None)) = (child, tasks[actor].threads.get(slot)) {
            out.push(SchedTraceEvent::Join {
                turn,
                task: actor,
                child: *child,
            });
        }
    }
    // #1981: the frames the actor entered or left, so an embedder can build exact per-thread call
    // spans from the tape instead of sampling stacks at stops (a short thread can live and die
    // between two stops).
    let depth = trace_depth(&tasks[actor]);
    for level in actor_depth..depth {
        let (module, func) = trace_frame(&tasks[actor], level);
        out.push(SchedTraceEvent::Call {
            turn,
            task: actor,
            module,
            func,
        });
    }
    for _ in depth..actor_depth {
        out.push(SchedTraceEvent::Return { turn, task: actor });
    }
    let now = trace_tags(tasks);
    for (j, &(nt, naux)) in now.iter().enumerate() {
        match before.get(j) {
            None => {
                out.push(SchedTraceEvent::Spawn {
                    turn,
                    parent: actor,
                    task: j,
                });
                let (module, func) = trace_frame(&tasks[j], 0);
                out.push(SchedTraceEvent::Call {
                    turn,
                    task: j,
                    module,
                    func,
                });
            }
            Some(&(wt, waux)) if wt == nt && waux == naux => {}
            Some(&(wt, _)) => match (wt, nt) {
                (_, 1) => out.push(SchedTraceEvent::ParkJoin {
                    turn,
                    task: j,
                    child: naux as usize,
                }),
                (_, 2) => out.push(SchedTraceEvent::ParkWait {
                    turn,
                    task: j,
                    key: naux,
                }),
                (2, 0) => out.push(SchedTraceEvent::WakeNotify {
                    turn,
                    waker: actor,
                    wakee: j,
                }),
                (1, 0) => out.push(SchedTraceEvent::WakeJoin {
                    turn,
                    waker: actor,
                    wakee: j,
                }),
                _ => {} // a completion or re-key — no timeline edge
            },
        }
    }
}

/// The pick-phase differ: the only transition a pick can cause is a timed-out `memory.wait` waking
/// (`dbg_pick_runnable`'s clock advance), so any blocked-wait → runnable here is a `WakeTimeout`.
fn trace_pick_diff(
    before: &[(u8, u64)],
    tasks: &[DbgTask],
    turn: u64,
    out: &mut Vec<SchedTraceEvent>,
) {
    let now = trace_tags(tasks);
    for (j, &(nt, _)) in now.iter().enumerate() {
        if let Some(&(2, _)) = before.get(j) {
            if nt == 0 {
                out.push(SchedTraceEvent::WakeTimeout { turn, task: j });
            }
        }
    }
}

fn dbg_complete(tasks: &mut [DbgTask], ti: usize, res: Result<Vec<Value>, Trap>) {
    let mut work = vec![(ti, res)];
    while let Some((done, res)) = work.pop() {
        tasks[done].state = DbgTaskState::Done(res.clone());
        for (j, t) in tasks.iter_mut().enumerate() {
            let DbgTaskState::BlockedJoin { child, slot, dst } = t.state else {
                continue;
            };
            if child != done {
                continue;
            }
            t.threads[slot] = None;
            match &res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    t.vt.active.set(dst, Reg::from_value(v));
                    t.state = DbgTaskState::Runnable;
                }
                Err(trap) => work.push((j, Err(*trap))),
            }
        }
    }
}

/// `thread.spawn`: add a child vCPU running `func(sp, arg)` (sharing the domain), write its handle to
/// the spawner's `dst`. Mirrors the production `drive`'s `Spawn` arm for the debuggable subset.
#[allow(clippy::too_many_arguments)]
fn dbg_spawn(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    func: u32,
    sp: i64,
    arg: i64,
    dst: u32,
    module: usize,
    source: &ModuleSource,
) -> Result<(), Trap> {
    // Module-aware, as the drive arms: `func` is the spawning frame's module's index.
    let cm = source.get(module).ok_or(Trap::Malformed)?;
    if func as usize >= cm.progs.len() {
        return Err(Trap::Malformed);
    }
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // thread bomb
    }
    let mut vt = VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
    vt.active.module = module;
    vt.active.home = module;
    let env = tasks[ti].env; // a thread inherits its spawner's environment (shares its window)
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt,
        threads: Vec::new(),
        env,
        state: DbgTaskState::Runnable,
        at_bp: false,
        lease: None,
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// The per-op driver for a scheduled debug task, selecting its runtime context: the shared domain
/// (`env = None`) or its confined [`DbgEnv`] (`env = Some(k)`). Centralizes the split borrow so the
/// unified `drive`/`tick` pumps stay a single call. Mirrors the production `drive`'s `RunCtx` selection.
#[allow(clippy::too_many_arguments)]
fn dbg_advance_task(
    tasks: &mut [DbgTask],
    ti: usize,
    extra_envs: &mut [DbgEnv],
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    match tasks[ti].env {
        None => debug_advance_fiber(&mut tasks[ti].vt, fibers, source, table, fuel, mem, host),
        Some(k) => {
            let e = &mut extra_envs[k];
            debug_advance_fiber(
                &mut tasks[ti].vt,
                &mut e.fibers,
                source,
                &e.table,
                &mut e.fuel,
                &mut e.mem,
                &mut e.host,
            )
        }
    }
}

/// Outcome of [`service_advance`]: `Ran` = a step this engine services (op / thread spawn·join /
/// futex wait·notify / §14 instantiate) was dispatched and `turn` ticked; `Declined` = a coroutine
/// or tier-up op this scheduled engine does not drive (`turn` left untouched, caller bails its own
/// way).
enum Serviced {
    Ran,
    Declined,
}

/// Advance task `ti` one step and service the scheduler seam it produced. This is the shared core of
/// [`ScheduledDebugRun::drive`] and [`ScheduledDebugRun::tick`]: both dispatch the *same* set of
/// [`Outcome`]s with the *same* rejections, so a new seam is added here **once** rather than in two
/// places — a `tick` (the reverse-`seek` replay path) that silently declined what `drive` services
/// would desync replay from live runs (DEBUGGING.md; INVARIANTS #9 observability corollary). `turn`
/// is ticked for every serviced step, matching each engine's per-op clock; on `Declined` it is left
/// untouched so each caller keeps its own bail (`drive` → `SchedStop::Declined` with no tick; `tick`
/// → tick its clock and stop the replay).
#[allow(clippy::too_many_arguments)]
fn service_advance(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    fibers: &mut Vec<FiberState>,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    clock: u64,
    turn: &mut u64,
) -> Serviced {
    let step_res = dbg_advance_task(
        tasks, ti, extra_envs, fibers, source, table, fuel, mem, host,
    );
    tasks[ti].at_bp = false;
    match step_res {
        // A fiber switch (or a plain op) — the vCPU advanced one op, stays runnable.
        FiberStep::Stepped => *turn += 1,
        FiberStep::Finished(vals) => {
            *turn += 1;
            dbg_complete(tasks, ti, Ok(vals));
        }
        FiberStep::Trapped(t) => {
            *turn += 1;
            dbg_complete(tasks, ti, Err(t));
        }
        // #1366 — a host-completed cap park: the call did not run and the turn holds (as a stdin
        // park); `drive` reports `SchedStop::CapPark` once nothing else can run, and `deliver_cap`
        // lands the value and re-admits the task. Until #1517 this engine failed closed here — and
        // could not in fact reach it, since it never admitted host-completed punts at all.
        FiberStep::CapParked { id, dst, at } => {
            tasks[ti].state = DbgTaskState::CapParked { id, dst, at };
        }
        // A scheduler seam: the ones this engine dispatches, else `Declined`.
        FiberStep::Other(outcome) => match outcome {
            Outcome::ThreadSpawn {
                func,
                sp,
                arg,
                dst,
                module,
            } => {
                *turn += 1;
                if let Err(t) = dbg_spawn(tasks, ti, func, sp, arg, dst, module, source) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            Outcome::ThreadJoin { handle, dst } => {
                *turn += 1;
                dbg_join(tasks, ti, handle, dst);
            }
            Outcome::MemoryWait {
                base,
                expected,
                width,
                timeout,
                dst,
            } => {
                *turn += 1;
                let m = dbg_env_mem(mem, extra_envs, tasks[ti].env);
                dbg_wait(tasks, ti, m, clock, base, expected, width, timeout, dst);
            }
            Outcome::MemoryNotify { base, count, dst } => {
                *turn += 1;
                dbg_notify(tasks, ti, mem, extra_envs, base, count, dst);
            }
            // §14 confined children (ops 0, 5, 13, 17): the executor's admission, scheduled as a debug
            // task over its own `DbgEnv`, so a spawning guest stays steppable through its child.
            Outcome::Instantiate { spawn, dst } => {
                *turn += 1;
                if let Err(t) = dbg_instantiate_confined(
                    tasks, ti, extra_envs, source, mem, *fuel, host, spawn, dst,
                ) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            // §5 `instantiate_detached` (op 15): a fresh window of its own as a debug task — the
            // cooperative executor's spawn on this path, so a spawning guest stays steppable through
            // its child (a `DbgEnv` over the child's own `Mem`, joined like a confined child's).
            Outcome::InstantiateDetached { spawn, dst } => {
                *turn += 1;
                if let Err(t) = dbg_instantiate_detached(
                    tasks, ti, extra_envs, source, mem, *fuel, host, spawn, dst,
                ) {
                    dbg_complete(tasks, ti, Err(t));
                }
            }
            // #1146 (deeper) — a blocking-stdin park (W4): the read did not run and the turn holds
            // `drive` reports `SchedStop::StdinPark` once nothing
            // else can run, and `provide_stdin` re-admits the task so the read re-issues.
            Outcome::StdinPark => tasks[ti].state = DbgTaskState::BlockedStdin,
            // Everything this engine does **not** dispatch, named rather than caught by a `_`
            // (#1414). The debugger path is the one native driver whose event match had a wildcard;
            // `pump` and `run_vcpu_parallel` are already exhaustive, so a new `Outcome` variant fails
            // to build there and must be handled. Here it silently joined the declined set — which is
            // the same shape as #1412, where a new answer slipped in without anything going red.
            //
            // Declining is still the right *answer* for all of these (INVARIANTS #9: a tier that
            // cannot service something falls back rather than running it wrong; the caller re-runs on
            // an engine that can). What changes is that it is now a decision per variant instead of a
            // default, so adding an `Outcome` forces someone to say which group it belongs to.

            // Not reachable: a step that completed or suspended is `FiberStep::Stepped`/`Finished`,
            // never `Other` (which is documented as "a non-fiber seam the caller must apply").
            // Declined rather than `unreachable!()` — an impossible-looking arm is a bad place to
            // introduce a panic, and the fallback is correct either way.
            Outcome::Done(_) | Outcome::Suspended => return Serviced::Declined,

            // Coroutines and fibers: this engine schedules tasks, not continuations.
            Outcome::ContNew { .. } | Outcome::ContResume { .. } | Outcome::FiberSuspend { .. } => {
                return Serviced::Declined
            }

            // Tier-up: the debug tier is the interpreter by construction (INVARIANTS #9's
            // observability corollary — stepping wants an interpreter).
            Outcome::TierUp { .. } => return Serviced::Declined,

            // Serving / cap plumbing: needs the host's serve loop and waiter table.
            Outcome::CapPending { .. }
            | Outcome::LiveCall { .. }
            | Outcome::SvcWait
            | Outcome::ChildOffer { .. }
            | Outcome::CloneCaller { .. } => return Serviced::Declined,

            // Process / POSIX seams: fork, exec, reap and pipes are the personality's, driven by the
            // run harness rather than the debug scheduler.
            Outcome::Reap { .. }
            | Outcome::Exec { .. }
            | Outcome::ForkSelf { .. }
            | Outcome::SpawnSelf { .. }
            | Outcome::ReapWait { .. }
            | Outcome::PipeRead { .. }
            | Outcome::PipeWrite { .. } => return Serviced::Declined,

            // §22 guest-JIT: compiling and invoking guest-emitted units needs the JIT tables the
            // debug run does not stand up.
            Outcome::JitInstall { .. }
            | Outcome::JitUninstall { .. }
            | Outcome::JitInvoke { .. } => return Serviced::Declined,

            // §GC root scan: walks the live fiber stacks via the fiber runtime.
            // §GC `gc.roots` (#1563): scanned, not declined. A decline here is not a skip — `drive`
            // turns it into `SchedStop::Declined` without ticking the clock, so a guest that
            // collects could not be stepped past its first collection, i.e. a GC'd language runtime
            // was undebuggable. Everything the op needs is already in hand: the same continuation
            // `step_vcpu` scans (this task's `vt`, the run's `fibers`) and the window every other
            // seam here selects the same way. The scan itself is [`gc_scan`], shared with
            // production, so the debug tier reports the same roots rather than a second answer
            // (INVARIANTS #9's observability corollary).
            Outcome::GcRoots {
                lo,
                hi,
                mask,
                buf,
                cap,
                dst,
            } => {
                *turn += 1;
                let reg = match tasks[ti].env {
                    None => &*fibers,
                    Some(k) => &extra_envs[k].fibers,
                };
                let roots = gc_scan(&tasks[ti].vt, reg, source, lo, hi, mask);
                let m: &mut Option<Mem> = match tasks[ti].env {
                    None => mem,
                    Some(k) => &mut extra_envs[k].mem,
                };
                match gc_write(m, buf, cap, roots) {
                    Ok(total) => tasks[ti].vt.active.set(dst, Reg::from_i64(total)),
                    Err(t) => dbg_complete(tasks, ti, Err(t)),
                }
            }
        },
    }
    Serviced::Ran
}

/// §14 confined children (ops 0, 5, 13, 17) under the debug scheduler: [`admit_confined_child`] —
/// the executor's admission and child powerbox — then [`dbg_start_child`]. Writes the handle (or
/// `EINVAL`) to `dst`; `Err` is a trap the caller completes `ti` with: a forged module handle, an
/// un-lowerable module, an unreadable grant list, or the vCPU-count bomb.
#[allow(clippy::too_many_arguments)]
fn dbg_instantiate_confined(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    source: &ModuleSource,
    shared_mem: &Option<Mem>,
    shared_fuel: u64,
    host: &mut Host,
    spawn: ConfinedSpawn,
    dst: u32,
) -> Result<(), Trap> {
    // The spawning task's window and fuel, and the powerbox its handles resolve in: its own (#1727).
    let (pm, owner, pfuel) = match tasks[ti].env {
        None => (shared_mem.as_ref(), host, shared_fuel),
        Some(k) => {
            let e = &mut extra_envs[k];
            (e.mem.as_ref(), &mut e.host, e.fuel)
        }
    };
    let Some(child) = admit_confined_child(owner, pm, pfuel, source, &tasks[ti].vt.active, spawn)?
    else {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::EINVAL as i32));
        return Ok(());
    };
    dbg_start_child(tasks, ti, extra_envs, source, child, spawn.entry, dst)
}

/// Schedule an admitted §14/§5 child as a debug task over its own [`DbgEnv`], registered as a child
/// handle of `ti` (so `Instantiator.join` → `ThreadJoin` joins it), and land the handle in `dst`.
/// `Err(ThreadFault)` on the vCPU-count bomb (the caller completes `ti`).
fn dbg_start_child(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    source: &ModuleSource,
    child: AdmittedChild,
    entry: i64,
    dst: u32,
) -> Result<(), Trap> {
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, DbgTaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // instantiate bomb
    }
    let AdmittedChild {
        mem,
        host,
        program,
        args,
        fuel,
        lease,
    } = child;
    let (module, prog) = program.land(source)?;
    let (vt, table) = child_task(module, &prog, entry, &args, host.jit_table_log2())?;
    let eidx = extra_envs.len();
    extra_envs.push(DbgEnv {
        mem,
        host,
        table,
        fuel,
        fibers: Vec::new(),
    });
    let cidx = tasks.len();
    tasks.push(DbgTask {
        vt,
        threads: Vec::new(),
        env: Some(eidx),
        state: DbgTaskState::Runnable,
        at_bp: false,
        lease: lease.map(|(budget, bytes)| (tasks[ti].env, budget, bytes)),
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// §5 `instantiate_detached` (op 15) on the debugger path — [`dbg_instantiate_confined`]'s twin over
/// a **fresh window** instead of a carve. Admission and the child powerbox are [`admit_detached_child`]
/// (the executor's); the child is scheduled by [`dbg_start_child`], joined like a confined child.
#[allow(clippy::too_many_arguments)]
fn dbg_instantiate_detached(
    tasks: &mut Vec<DbgTask>,
    ti: usize,
    extra_envs: &mut Vec<DbgEnv>,
    source: &ModuleSource,
    shared_mem: &Option<Mem>,
    shared_fuel: u64,
    host: &mut Host,
    spawn: DetachedSpawn,
    dst: u32,
) -> Result<(), Trap> {
    // The parent's window and fuel, and the powerbox its handles resolve in: its own (#1727).
    let (pm, owner, pfuel) = match tasks[ti].env {
        None => (shared_mem.as_ref(), host, shared_fuel),
        Some(k) => {
            let e = &mut extra_envs[k];
            (e.mem.as_ref(), &mut e.host, e.fuel)
        }
    };
    let Some(child) = admit_detached_in_process(owner, pm, pfuel, spawn)? else {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::EINVAL as i32));
        return Ok(());
    };
    dbg_start_child(tasks, ti, extra_envs, source, child, spawn.entry, dst)
}

/// `thread.join`: deliver a finished child's result now, else park the joiner. Mirrors `drive`'s `Join`.
fn dbg_join(tasks: &mut [DbgTask], ti: usize, handle: i32, dst: u32) {
    let slot = match super::resolve_thread(&tasks[ti].threads, handle) {
        Ok(s) => s,
        Err(t) => {
            dbg_complete(tasks, ti, Err(t));
            return;
        }
    };
    let child = tasks[ti].threads[slot].expect("resolve_thread checked liveness");
    match &tasks[child].state {
        DbgTaskState::Done(res) => {
            let res = res.clone();
            tasks[ti].threads[slot] = None;
            match res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    tasks[ti].vt.active.set(dst, Reg::from_value(v));
                }
                Err(t) => dbg_complete(tasks, ti, Err(t)),
            }
        }
        _ => tasks[ti].state = DbgTaskState::BlockedJoin { child, slot, dst },
    }
}

/// §22 `Jit.install` (op 3) under the debug engine: resolve authority + the unit's funcs from the host
/// (a forged/cross-domain handle is an inert `CapFault` → trap), compile the unit to bytecode, and
/// install it into the debug run's shared `(source, table)` — the debug-engine counterpart of the
/// production `drive`'s `JitInstall` arm. Serviced inline in [`debug_advance_fiber`] (it mutates only
/// `vt.active` + the shared table, spawning no scheduler task). Writes the slot (or `-ENOSPC`, an ordinary value) to `dst`; `Err`
/// traps the vCPU (`CapFault` forged handle, `Malformed` unit outside bytecode coverage — the one place
/// a guest-provided unit can outrun coverage, with no tree-walker fallback mid-run).
fn dbg_jit_install(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    table: &SharedSlots,
    h: i32,
    code: i32,
    dst: u32,
) -> Result<(), Trap> {
    let (funcs, types) = host.resolve_jit_domain(h).and_then(|domain| {
        let (cd, cu) = host.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        host.jit_unit_funcs(cd, cu)
            .ok_or(Trap::CapFault)
            .and_then(|f| {
                host.jit_unit_types(cd, cu)
                    .ok_or(Trap::CapFault)
                    .map(|t| (f, t))
            })
    })?;
    let res = match compile_module(&funcs, &types, None) {
        Some(unit) => match jit_install_into(source, table, unit) {
            Some(slot) => slot as i64,
            None => super::ENOSPC,
        },
        None => return Err(Trap::Malformed), // unit op outside coverage
    };
    vt.active.set(dst, Reg::from_i64(res));
    Ok(())
}

/// §22 `Jit.uninstall` (op 4) under the debug engine: authority-check the domain handle, then clear the
/// installed table slot (`0`/`-EINVAL` to `dst`). Mirrors `drive`'s `JitUninstall` arm; serviced inline
/// in [`debug_advance_fiber`].
fn dbg_jit_uninstall(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    table: &SharedSlots,
    h: i32,
    slot: i64,
    dst: u32,
) -> Result<(), Trap> {
    host.resolve_jit_domain(h)?; // authority (forged handle → CapFault)
    let n_real = source.primary().progs.len();
    let res = if jit_uninstall_from(source, table, slot as usize, n_real) {
        0
    } else {
        super::EINVAL
    };
    vt.active.set(dst, Reg::from_i64(res));
    Ok(())
}

/// Shared prep for §22 `Jit.invoke` on the debug engines (both step into the unit):
/// resolve authority + the unit's funcs from the host (forged/cross-domain → `CapFault`), compile the
/// unit (out-of-coverage → `Malformed`), arity-check its entry (func 0) against the call's
/// (code-stripped) signature (`CapFault` on mismatch), and marshal the args through the i64-slot ABI.
/// Returns the compiled unit + its entry args; the caller pushes it to `source` and runs/steps it.
fn dbg_jit_invoke_unit(
    host: &mut Host,
    h: i32,
    code: i32,
    argv: &[i64],
    params: &[ValType],
    results: &[ValType],
) -> Result<(Compiled, Vec<Value>), Trap> {
    let (funcs, types) = host.resolve_jit_domain(h).and_then(|domain| {
        let (cd, cu) = host.resolve_jit_code(code)?;
        if cd != domain {
            return Err(Trap::CapFault);
        }
        host.jit_unit_funcs(cd, cu)
            .ok_or(Trap::CapFault)
            .and_then(|f| {
                host.jit_unit_types(cd, cu)
                    .ok_or(Trap::CapFault)
                    .map(|t| (f, t))
            })
    })?;
    let unit = compile_module(&funcs, &types, None).ok_or(Trap::Malformed)?;
    let arity_ok = unit
        .sigs
        .first()
        .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
    if !arity_ok {
        return Err(Trap::CapFault);
    }
    let child_args: Vec<Value> = params
        .iter()
        .zip(argv.iter())
        .map(|(ty, s)| slot_to_val(*ty, *s))
        .collect();
    Ok((unit, child_args))
}

/// §22 `Jit.invoke` (op 1) **as a step-into** (both debug engines): compile + push the unit, then
/// arm [`VTask::active_invoke`] so [`debug_advance_fiber`] steps the invoked unit op-by-op (breakpoints
/// fire inside it) instead of running it opaquely — the §22 counterpart of coroutine step-into. The
/// unit runs over the caller's shared window/table; `dst`/`results` marshal its returns back to the
/// caller on completion ([`step_active_invoke`]). `Err` traps the caller (forged handle / bad unit).
#[allow(clippy::too_many_arguments)]
fn dbg_jit_invoke_step_into(
    vt: &mut VTask,
    host: &mut Host,
    source: &ModuleSource,
    h: i32,
    code: i32,
    argv: &[i64],
    dst: u32,
    params: &[ValType],
    results: &[ValType],
) -> Result<(), Trap> {
    let (unit, child_args) = dbg_jit_invoke_unit(host, h, code, argv, params, results)?;
    let umod = source.push(unit);
    let cm = source.get(umod).ok_or(Trap::Malformed)?;
    let mut vm = Vm::new(&cm, 0, &child_args)?;
    vm.module = umod;
    let parent_depth = vt.active.stack.len() + 1;
    vt.active_invoke = Some(Box::new(InvokeStep {
        vm,
        dst,
        results: results.into(),
        parent_depth,
    }));
    Ok(())
}

/// Advance the **active §22 invoked unit** (`vt.active_invoke`) by exactly one op — the op-by-op,
/// debugger-facing counterpart of [`run_invoke`]'s loop. The unit runs over the caller's shared
/// `mem`/`host`/`source`/`table` (a seam-free leaf), so its `call.dyn` reaches installed units and
/// any spawn/park/yield/re-invoke is an inert `CapFault` — exactly `run_invoke`'s `_ => CapFault`, only
/// surfaced one op at a time so a breakpoint can fire inside the unit. On the unit's return the caller's
/// `dst…` slots are filled through the i64-slot ABI and control returns to the caller.
fn step_active_invoke(
    vt: &mut VTask,
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> FiberStep {
    enum InvStep {
        Ran,
        Done(Vec<Value>),
        Trap(Trap),
    }
    let step = {
        let iv = vt.active_invoke.as_mut().expect("active invoke present");
        match iv
            .vm
            .resume(source, table, fuel, mem, &mut HostCell::Excl(host), 1)
        {
            Ok(Outcome::Suspended) => InvStep::Ran, // budget boundary — one op done, keep stepping
            Ok(Outcome::Done(vals)) => InvStep::Done(vals),
            // F2 — a punted host call inside an invoked unit keeps the pre-F2 inline wait
            // (`run_invoke`'s arm; the unit is a seam-free atomic leaf, DESIGN §22).
            Ok(Outcome::CapPending { id, dst }) => {
                let r = host.completions().wait(id);
                iv.vm.set(dst, Reg::from_i64(r));
                InvStep::Ran
            }
            // A seam op (spawn/park/yield/cont.*/re-invoke) inside an invoked unit is an inert CapFault,
            // matching run_invoke's `_ => CapFault`.
            Ok(_) => InvStep::Trap(Trap::CapFault),
            Err(t) => InvStep::Trap(t),
        }
    };
    match step {
        InvStep::Ran => FiberStep::Stepped,
        InvStep::Done(vals) => {
            let iv = vt.active_invoke.take().expect("active invoke present");
            for (i, (v, ty)) in vals.iter().zip(iv.results.iter()).enumerate() {
                let re = slot_to_val(*ty, val_to_slot(*v));
                vt.active.set(iv.dst + i as u32, Reg::from_value(re));
            }
            FiberStep::Stepped
        }
        InvStep::Trap(t) => {
            vt.active_invoke = None;
            FiberStep::Trapped(t)
        }
    }
}

/// The window a task steps against: its §14 env's, or the root's for `env == None`.
fn dbg_env_mem<'a>(
    mem: &'a Option<Mem>,
    envs: &'a [DbgEnv],
    env: Option<usize>,
) -> Option<&'a Mem> {
    match env {
        None => mem.as_ref(),
        Some(k) => envs[k].mem.as_ref(),
    }
}

/// The futex rendezvous key of `addr` in window `m` (#1731): backing identity plus address, as the
/// cooperative driver keys it. Every detached window starts at base 0, so the address alone would
/// rendezvous across windows (#1283).
fn dbg_futex_key(m: Option<&Mem>, addr: u64) -> super::FutexKey {
    m.map_or(super::FutexKey::Anon(0, addr), |m| m.futex_key(addr))
}

/// `memory.wait`: park the caller at `base` until a `notify` or the deadline, unless the value in its
/// own window `mem` already changed (the compare-under-lock analogue). Mirrors `drive`'s `Wait`.
#[allow(clippy::too_many_arguments)]
fn dbg_wait(
    tasks: &mut [DbgTask],
    ti: usize,
    mem: Option<&Mem>,
    clock: u64,
    base: u64,
    expected: u64,
    width: u32,
    timeout: Option<u64>,
    dst: u32,
) {
    let cur = mem.map(|m| m.atomic_value(base, width)).unwrap_or(0);
    if cur != expected {
        tasks[ti]
            .vt
            .active
            .set(dst, Reg::from_i32(super::WAIT_NOT_EQUAL));
    } else {
        tasks[ti].state = DbgTaskState::BlockedWait {
            addr: base,
            // #1638: an infinite wait carries no deadline, so it is not a clock-advance
            // candidate below — it ends by `notify` or not at all (the deadlock exit).
            deadline: timeout.map(|t| clock.saturating_add(t)),
            dst,
        };
    }
}

/// `memory.notify`: wake up to `count` waiters whose futex key matches `base` in the caller's window
/// (lowest task index first, deterministic); the woken count lands at `dst`. Mirrors `drive`'s
/// `Notify`.
fn dbg_notify(
    tasks: &mut [DbgTask],
    ti: usize,
    mem: &Option<Mem>,
    envs: &[DbgEnv],
    base: u64,
    count: i32,
    dst: u32,
) {
    let key = dbg_futex_key(dbg_env_mem(mem, envs, tasks[ti].env), base);
    let want = count as u32;
    let mut woken = 0u32;
    for t in tasks.iter_mut() {
        if woken >= want {
            break;
        }
        if let DbgTaskState::BlockedWait {
            addr, dst: wdst, ..
        } = t.state
        {
            if dbg_futex_key(dbg_env_mem(mem, envs, t.env), addr) == key {
                t.vt.active.set(wdst, Reg::from_i32(super::WAIT_WOKEN));
                t.state = DbgTaskState::Runnable;
                woken += 1;
            }
        }
    }
    tasks[ti].vt.active.set(dst, Reg::from_i32(woken as i32));
}

/// SplitMix64 — the stateless mix behind the **seeded pick** (slice 7): the choice at a turn is a
/// pure function of `(seed, turn)`, so any replay — full or from a checkpoint — reproduces the
/// schedule with zero captured scheduler state (INVARIANTS.md #7: recovery never replays captured
/// scheduler records).
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The forced-switch override recorded for `turn`, if any (slice 7). Entries are concrete
/// `(turn, task)` pairs resolved at record time, so a replay re-applies the identical choice.
fn forced_at(forced: &[(u64, usize)], turn: u64) -> Option<usize> {
    forced
        .iter()
        .find(|(t, _)| *t == turn)
        .map(|(_, task)| *task)
}

/// The task **pinned** to the scheduler because it is mid-`resume` inside a §14 coroutine body
/// (`active_coro` set). A coroutine `resume` is atomic w.r.t. other vCPUs, so while its body is being
/// stepped op-by-op the scheduler must keep running that same vCPU — never interleaving another thread —
/// until the child yields / faults / returns and `active_coro` clears. At most one task is ever pinned
/// (a task can only enter a coroutine while running, and a pinned task runs alone), so the first match
/// is the pin.
fn dbg_pinned_coro(tasks: &[DbgTask]) -> Option<usize> {
    let _ = tasks;
    None
}

/// What a blocked thread waits on ([`ScheduledDebugRun::blocked_on`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockedOn {
    /// A `memory.wait` on the futex word at this window address.
    Futex(u64),
    /// A `thread.join` of this task.
    Join(usize),
}

/// One recorded step (#1942): the turns `[from, to)` it drove, the thread it stepped, and whether it
/// kept the other threads frozen (DAP `singleThread`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepSpan {
    pub from: u64,
    pub to: u64,
    pub task: usize,
    pub single_thread: bool,
}

/// The step a recorded span says was running at `turn`, as `(thread, single_thread)`, if one covers
/// it (#1942).
fn span_at(spans: &[StepSpan], turn: u64) -> Option<(usize, bool)> {
    let i = spans.partition_point(|s| s.from <= turn);
    let s = spans.get(i.checked_sub(1)?)?;
    (turn < s.to).then_some((s.task, s.single_thread))
}

/// Drop the step spans at or past `turn`, clipping one that straddles it — the future a new step
/// from `turn` rewrites (#1942).
fn truncate_spans(spans: &mut Vec<StepSpan>, turn: u64) {
    spans.retain(|s| s.from < turn);
    if let Some(last) = spans.last_mut() {
        last.to = last.to.min(turn);
    }
}

/// Record a step's span, merged into the previous one when it continues it on the same thread in
/// the same mode (a run of steps on one thread — every step of a single-threaded program — stays one
/// span).
fn push_span(spans: &mut Vec<StepSpan>, span: StepSpan) {
    if span.from >= span.to {
        return;
    }
    match spans.last_mut() {
        Some(last)
            if last.to == span.from
                && last.task == span.task
                && last.single_thread == span.single_thread =>
        {
            last.to = span.to
        }
        _ => spans.push(span),
    }
}

/// The pick among the **runnable** tasks, without side effects. Each candidate only while it is
/// runnable; `None` when nothing is.
///
/// 1. A forced switch recorded for `turn` wins.
/// 2. Then `pref`, the step in progress: `(thread, single_thread)` live, or a recorded step span's
///    on replay (#1942).
///    - A **single-thread** step (DAP `singleThread`) runs only its thread, the others frozen.
///    - Any other step shares the turns: the runnable threads take one each, in order. The stepped
///      thread still decides when the step ends, but the others keep running meanwhile, as the DAP
///      spec has them do, so a step over a spin-wait on another thread's flag ends. When a step
///      kept to its thread, it never did.
///    - A blocked stepping thread falls through to the policy (a step over a `join` can't
///      deadlock).
/// 3. Then the policy: a `seed`ed pick chosen uniformly via [`splitmix64`]`(seed ^ turn)`, else the
///    runnable threads in turn, one [`COOP_QUANTUM`] each — the release engine's preemption quantum,
///    so a run through a spin-wait ends on both engines. Within the first quantum that is the
///    lowest-index runnable thread, as it always was.
///
/// Every rule is a function of the arguments alone, so a replay picks exactly as the live run did.
fn dbg_preview_pick(
    tasks: &[DbgTask],
    seed: Option<u64>,
    forced: &[(u64, usize)],
    pref: Option<(usize, bool)>,
    turn: u64,
) -> Option<usize> {
    let runnable =
        |i: usize| matches!(tasks.get(i).map(|t| &t.state), Some(DbgTaskState::Runnable));
    if let Some(f) = forced_at(forced, turn).filter(|&f| runnable(f)) {
        return Some(f);
    }
    let set: Vec<usize> = (0..tasks.len()).filter(|&i| runnable(i)).collect();
    if let Some((p, single_thread)) = pref.filter(|&(p, _)| runnable(p)) {
        return Some(if single_thread {
            p
        } else {
            set[(turn % set.len() as u64) as usize]
        });
    }
    match seed {
        _ if set.is_empty() => None,
        None => Some(set[((turn / COOP_QUANTUM) % set.len() as u64) as usize]),
        Some(s) => Some(set[(splitmix64(s ^ turn) % set.len() as u64) as usize]),
    }
}

/// Pick the next thread to run under the session's **schedule policy** (slice 7) — the
/// [`dbg_preview_pick`] order. If none is runnable, advance the futex `clock` to the earliest
/// `memory.wait` deadline and wake every timed-out waiter (`WAIT_TIMED_OUT`), then retry. `None`
/// only on a true deadlock (no runnable thread and no waiter) — mirrors `drive`. Every path is a
/// pure function of `(seed, forced, pref, turn, task states)`, so replay reproduces it exactly —
/// which is why the live `drive` and the replaying `tick` both pick through here.
fn dbg_pick_runnable(
    tasks: &mut [DbgTask],
    clock: &mut u64,
    seed: Option<u64>,
    forced: &[(u64, usize)],
    pref: Option<(usize, bool)>,
    turn: u64,
) -> Option<usize> {
    loop {
        if let Some(i) = dbg_preview_pick(tasks, seed, forced, pref, turn) {
            return Some(i);
        }
        // #1638 — only a waiter with a REAL deadline is a clock-advance candidate. `flatten`
        // drops the indefinite ones, so "nothing runnable and every remaining waiter is
        // indefinite" yields `None` here and takes the `?` deadlock exit this function already
        // documented, instead of advancing the clock to a `MAX_WAIT` stand-in and handing the
        // guest a `WAIT_TIMED_OUT` it never asked for.
        let next = tasks
            .iter()
            .filter_map(|t| match t.state {
                DbgTaskState::BlockedWait { deadline, .. } => deadline,
                _ => None,
            })
            .min()?;
        *clock = (*clock).max(next);
        for t in tasks.iter_mut() {
            if let DbgTaskState::BlockedWait {
                deadline: Some(deadline),
                dst,
                ..
            } = t.state
            {
                if deadline <= *clock {
                    t.vt.active.set(dst, Reg::from_i32(super::WAIT_TIMED_OUT));
                    t.state = DbgTaskState::Runnable;
                }
            }
        }
    }
}

impl ScheduledDebugRun {
    /// Open a multithreaded debug session on `m`'s `func(args)`. `None` if the module is outside the
    /// bytecode engine's subset (`compile_module` declines it). The powerbox is empty; use
    /// [`new_with_host`](ScheduledDebugRun::new_with_host) to debug a guest needing a granted capability
    /// (e.g. a §14 `Instantiator` for `instantiate`).
    pub fn new(m: &Module, func: FuncIdx, args: &[Value]) -> Option<ScheduledDebugRun> {
        ScheduledDebugRun::new_with_host(m, func, args, Host::new())
    }

    /// [`new`](ScheduledDebugRun::new) carrying a live powerbox `host`, so a granted `Instantiator`
    /// (`host.grant_instantiator(..)`) reaches the guest as an argument and makes an `instantiate`-using
    /// multithreaded guest debuggable.
    pub fn new_with_host(
        m: &Module,
        func: FuncIdx,
        args: &[Value],
        host: Host,
    ) -> Option<ScheduledDebugRun> {
        m.funcs.get(func as usize)?;
        let ModuleDebug {
            fn_block_base,
            fn_block_types,
            ..
        } = ModuleDebug::build(m, 0);
        let c = compile_module_unfused(&m.funcs, &m.types, m.memory.and_then(|x| x.shadow))?; // unfused: debug stepping (Slice 5a)
        let table = SharedSlots::new(c.progs.len(), host.jit_table_log2(), 0);
        let source = std::sync::Arc::new(ModuleSource::over(std::sync::Arc::new(c)));
        let mem = build_mem(m, &[]);
        let vt = VTask::new(&source.primary(), func as usize, args).ok()?;
        Some(ScheduledDebugRun {
            source,
            table,
            mem,
            host,
            tasks: vec![DbgTask {
                vt,
                threads: Vec::new(),
                env: None,
                state: DbgTaskState::Runnable,
                at_bp: false,
                lease: None,
            }],
            extra_envs: Vec::new(),
            fibers: Vec::new(),
            fn_block_base,
            fn_block_types,
            debug: m.debug_info.clone(),
            funcs: std::sync::Arc::from(m.funcs.clone()),
            breakpoints: Vec::new(),
            watchpoints: Vec::new(),
            journal: super::journal::Journal::new(),
            journal_policy: super::journal::JournalPolicy::default(),
            value_watches: Vec::new(),
            access_sink: None,
            sched_trace: None,
            scheduled_writes: Vec::new(),
            write_cursor: 0,
            sched_seed: None,
            forced: Vec::new(),
            step_spans: Vec::new(),
            sched_sink: None,
            single_thread: false,
            last_watch: None,
            // Entry-stopped: the first `step` steps off the entry op (not to completion), and the
            // `stopped`/`focus` reads resolve the root task without an explicit `locate`.
            stopped: Some(0),
            focus: 0,
            turn: 0,
            clock: 0,
        })
    }

    /// Seed the low bytes of the window before stepping — the argv/env blob a powerbox entry expects
    /// at [`module_args_base`](temen_ir::module_args_base), which every non-trivial guest reads on the
    /// way into `main`.
    ///
    /// Without this a debug run of an argv-taking guest starts with an empty args region and takes a
    /// different path than the same guest under `run_with_caps` — so it is the difference between the
    /// debug engine running toy modules and running the real ones. Call it before the first step;
    /// afterwards it would be a write the journal never saw.
    ///
    /// Bytes past the window are ignored, exactly as the production seed does — confinement only
    /// concerns `[0, size)`.
    pub fn seed_mem(&mut self, init: &[u8]) {
        if let Some(m) = self.mem.as_mut() {
            m.seed(init);
        }
    }

    /// Replace the run-shared breakpoint set (fires in whichever thread reaches a pc in it).
    pub fn set_breakpoints(&mut self, bps: Vec<super::IrPc>) {
        self.breakpoints = bps;
    }

    /// Replace the run-shared **watchpoints** (DEBUGGING.md W2, cross-thread): each `(addr, len, kind)`
    /// makes the schedule stop *before* any op — in whichever thread — that accesses `[addr, addr+len)`
    /// with a matching read/write kind.
    pub fn set_watchpoints(&mut self, ranges: Vec<(u64, u64, super::WatchKind)>) {
        self.watchpoints = ranges;
    }

    /// Arm or disarm the **undo journal** (#1556/#1557): while armed, every op's about-to-be-written
    /// window ranges are recorded as pre-images, so [`undo_to`](Self::undo_to) can step backward
    /// without replaying. Disarmed (the default) it is one boolean test per op.
    pub fn set_journal_armed(&mut self, armed: bool) {
        self.journal.set_armed(armed);
    }

    /// Replace the journal's retention policy (#1558). Static: it decides granularity going forward
    /// and never revisits a past compaction, which is what keeps a later adaptive policy a parameter
    /// change rather than a reshape.
    pub fn set_journal_policy(&mut self, policy: super::journal::JournalPolicy) {
        self.journal_policy = policy;
    }

    /// What the journal is holding — the volume numbers #1557/#1558 report.
    pub fn journal_stats(&self) -> super::journal::JournalStats {
        self.journal.stats()
    }

    /// The earliest turn [`undo_to`](Self::undo_to) can reach, or `None` when the journal is empty.
    pub fn journal_earliest(&self) -> Option<u64> {
        self.journal.earliest()
    }

    /// **Level 2 compaction** (#1558): coalesce journal entries before `turn` to one pre-image per
    /// address. The segment can then only be undone to its start, and its size becomes the count of
    /// distinct addresses it touched rather than its write count.
    pub fn coalesce_journal(&mut self, before: u64) {
        self.journal.coalesce(before);
    }

    /// Whether the journal can undo to `turn` — i.e. it holds the engine state recorded before that
    /// turn's op. False for a turn outside the recorded history, one already compacted past, or one in
    /// a run outside the invertible subset (see [`crate::journal`] on fail-closed). The caller falls
    /// back to `seek`, which is always available.
    pub fn can_undo_to(&self, turn: u64) -> bool {
        // Undoing to where the run already stands is vacuously possible, and has no state entry of its
        // own: entries are recorded *before* each op, so the current turn's op has not run.
        if turn == self.turn {
            return true;
        }
        // Undo only goes backward. A turn past the run's own position is not history, and the
        // nearest-at-or-before anchor lookup would otherwise happily answer with the last boundary.
        turn < self.turn && self.journal.can_undo_to(turn)
    }

    /// Undo back to `turn`, putting the run where it stood **before** that turn's op ran: the window
    /// from the journal's pre-images, the continuation and host cursor from the state recorded there.
    /// `false` (changing nothing) when [`can_undo_to`](Self::can_undo_to) is false — undo never
    /// half-rewinds, it declines and leaves `seek` to serve.
    ///
    /// This is the counterpart of `seek(turn)` and must agree with it; `undo_journal.rs` is the
    /// differential that pins that.
    pub fn undo_to(&mut self, turn: u64) -> bool {
        if turn == self.turn {
            return true; // already here; see `can_undo_to`
        }
        let Some(st) = self.journal.state_at(turn) else {
            return false;
        };
        // Continuations ride segment boundaries, so the entry found is the boundary **at or before**
        // the target. Undo lands there and re-executes the remainder forward — bounded by
        // `JournalPolicy::state_stride`, and the window pre-images are per-op either way, so nothing
        // about the *window* is approximated by this.
        let anchor = st.coord;
        let cont = st.cont.clone();
        let cursor = st.cursor;
        // Window first: the pre-images are keyed on the turns being undone, and installing the
        // continuation does not touch guest memory. Undo all the way to the anchor — the forward
        // replay below re-runs the ops between it and the target, which re-writes exactly what was
        // just reverted, so the window ends correct at `turn` either way.
        if let Some(m) = self.mem.as_mut() {
            self.journal.undo_window_to(anchor, m);
        }
        // Verbatim (`readmit_parks = false`): an undo rewinds in place rather than re-executing, so a
        // task parked at that turn comes back parked.
        self.install_continuation(&cont, false);
        // The tape has to serve whatever is re-executed after this. The crossings between `turn` and
        // where the run stood have already happened once, and a *nondeterministic* input capability
        // must not be asked again — the live closure would answer differently and the timeline would
        // fork. Arming replay from the run's own recorded tape is the same pairing `build_run` uses on
        // a seek rebuild, and it is why the cursor belongs in the journal: rewinding `cap_consumed`
        // (below) is what makes the re-execution re-serve exactly the recorded answers.
        let taped = self.host.cap_tape();
        if !taped.records.is_empty() {
            self.host.replay_cap_tape(taped);
        }
        self.host.restore_journal_cursor(&cursor);
        self.turn = anchor;
        self.rewind_write_cursor(); // re-execution re-applies the writes it passes
        self.clock = cont.clock;
        self.locate();
        self.last_watch = None;
        // Re-execute the remainder of the segment. The journal stays armed, so the re-run re-records
        // the same pre-images for those turns and the history is whole again afterwards — an undo to
        // a position inside this stretch, next time, works exactly as this one did.
        //
        // Breakpoints and watchpoints are suppressed across the replay: it is *re-execution of turns
        // that already happened*, not the user continuing, and a stop here would strand the run
        // short of the position it was asked for. The same reasoning the DAP backend's `seek` replay
        // applies to its own drive.
        if anchor < turn {
            // `tick` is the right primitive here and the reason this is sound: it advances the
            // schedule by exactly one visible op, honouring no breakpoint / watch / step checks. This
            // is re-execution of turns that already happened, not the user continuing, so a stop
            // would strand the run short of the position it was asked for — and a `step` would
            // advance by a *step*, which is not the same quantum as a turn.
            let mut fuel = u64::MAX;
            while self.turn < turn && self.tick(&mut fuel) {}
            self.locate();
            self.last_watch = None;
        }
        self.apply_writes_due_now();
        self.turn == turn
    }

    /// Resolve a source variable held in an SSA value to a value-watch target (#1229), in the
    /// **focused** thread's frame `depth` levels from the top. `None` for a memory-located var (watch
    /// it by address via [`var_addr`](ScheduledDebugRun::var_addr)/[`set_watchpoints`](ScheduledDebugRun::set_watchpoints)),
    /// an unknown name, or one not live at the stopped pc. The target is `(func, site, baseline)`,
    /// frame- and thread-independent, so the DAP backend re-applies it verbatim after a `seek` rebuild.
    pub fn resolve_value_watch(&self, depth: usize, name: &str) -> Option<ValueWatchTarget> {
        let (func, site, last) = self.reader().value_watch_target(depth, name)?;
        Some(ValueWatchTarget { func, site, last })
    }

    /// Replace the run-shared **value watchpoints** (#1229) — each `(id, target, kind)` makes the
    /// schedule stop, in whichever thread, when the target variable's holding value changes.
    /// Re-applied by the DAP backend after a `seek` rebuild ([`merge_value_watches`] keeps a live
    /// watch's running baseline).
    pub fn set_value_watches(
        &mut self,
        watches: Vec<(super::WatchId, ValueWatchTarget, super::WatchKind)>,
    ) {
        self.value_watches = merge_value_watches(&self.value_watches, watches);
    }

    /// Install the run-shared per-op **access sink** ([`AccessSinkFn`]) — fired with the global
    /// `turn` and the executing task index. Observation only; zero cost when never installed.
    pub fn set_access_sink(&mut self, sink: AccessSinkFn) {
        self.access_sink = Some(sink);
    }

    /// Install the **scheduler-event sink** ([`SchedSinkFn`], #1987): each scheduler event, live, in
    /// order with the access sink's memory events — replays included, like the access sink.
    pub fn set_sched_sink(&mut self, sink: SchedSinkFn) {
        self.sched_sink = Some(sink);
    }

    /// Install the session's **scheduled debugger writes** ([`ScheduledWrite`], slice 8): each is
    /// applied when execution passes its turn — on the live resume and on every replay — so time
    /// travel stays truthful. The cursor lands past entries at turns already passed.
    pub fn set_scheduled_writes(&mut self, mut writes: Vec<(u64, ScheduledWrite)>) {
        writes.sort_by_key(|(c, _)| *c);
        self.scheduled_writes = writes;
        self.rewind_write_cursor();
    }

    /// Put the scheduled-write cursor at the first write not yet passed — those at turns before the
    /// run's own. Called wherever the run's turn moves other than by a tick.
    fn rewind_write_cursor(&mut self) {
        let turn = self.turn;
        self.write_cursor = self.scheduled_writes.partition_point(|(c, _)| *c < turn);
    }

    /// Apply the scheduled writes due at the run's **current** turn. A write made while stopped at turn
    /// `t` is part of the state at `t` — the live run shows it there — but the landing replay of a
    /// `seek` or `undo_to` stops *before* the op at `t`, where a tick would have applied it. Every
    /// landing calls this so the state at `t` is the same on every path (#1871). A no-op when nothing
    /// is due, and idempotent: the cursor moves past what it applies, so the tick at `t` won't repeat it.
    pub fn apply_writes_due_now(&mut self) {
        let Self {
            source,
            mem,
            tasks,
            turn,
            fn_block_base,
            fn_block_types,
            debug,
            journal,
            scheduled_writes,
            write_cursor,
            ..
        } = self;
        apply_due_writes(
            scheduled_writes,
            write_cursor,
            *turn,
            tasks,
            source,
            mem,
            debug.as_ref(),
            fn_block_base,
            fn_block_types,
            journal,
        );
    }

    /// The focused task index (the one a `write_var` resolves in) — the backend records it on a
    /// scheduled `Var` write so replays resolve in the same task.
    pub fn focus_task(&self) -> usize {
        self.focus
    }

    /// The shared window's memory-map introspection ([`MemMapInfo`]); `None` without a memory.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.mem.as_ref().map(|m| m.map_info())
    }

    /// Arm (or drop) the **scheduler trace tape** — see [`SchedTraceEvent`]. Arming resets the
    /// tape; observation only, zero cost when off.
    pub fn set_sched_trace(&mut self, on: bool) {
        self.sched_trace = on.then(|| {
            // #1981: the frames each task already has open, so every `Return` the tape records
            // pairs with a `Call` on it.
            let mut tape = Vec::new();
            for (task, t) in self.tasks.iter().enumerate() {
                for level in 0..trace_depth(t) {
                    let (module, func) = trace_frame(t, level);
                    tape.push(SchedTraceEvent::Call {
                        turn: self.turn,
                        task,
                        module,
                        func,
                    });
                }
            }
            tape
        });
    }

    /// The trace tape so far (`None` when not armed).
    pub fn sched_trace(&self) -> Option<&[SchedTraceEvent]> {
        self.sched_trace.as_deref()
    }

    /// Set (or clear) the **seeded pick** — see [`ScheduledDebugRun::sched_seed`]. Set before
    /// driving (the DAP backend applies it at construction and on every rebuild).
    pub fn set_sched_seed(&mut self, seed: Option<u64>) {
        self.sched_seed = seed;
    }

    /// Replace the recorded **forced switches** — concrete `(turn, task)` overrides (slice 7).
    pub fn set_forced_switches(&mut self, forced: Vec<(u64, usize)>) {
        self.forced = forced;
    }

    /// The currently-runnable task indices (the forced-switch verb resolves its target from this).
    pub fn runnable_tasks(&self) -> Vec<usize> {
        self.tasks
            .iter()
            .enumerate()
            .filter(|(_, t)| matches!(t.state, DbgTaskState::Runnable))
            .map(|(i, _)| i)
            .collect()
    }

    /// Take the `(addr, write)` of the watchpoint the last stop fired on (cleared by the read), so the
    /// backend can report `StopReason::Watchpoint`. `None` if the last stop was a breakpoint / step.
    pub fn take_watch_hit(&mut self) -> Option<(u64, bool)> {
        self.last_watch.take()
    }

    /// Drive the cooperative schedule until a breakpoint/watchpoint fires (in some thread), the root
    /// finishes, no thread is runnable (`Blocked`), or a thread hits an unsupported op (`Declined`).
    /// Resumable — the previously stopped thread steps one op past its stop before the scan resumes.
    pub fn run_until_stop(&mut self, fuel: &mut u64) -> SchedStop {
        self.drive(fuel, None, None)
    }

    /// [`run_until_stop`](Self::run_until_stop), but stop by turn `until` at the latest — reported as
    /// [`SchedBreak::Pause`] at the next op that can be a stop position. Resumable like any stop.
    pub fn run_until_turn(&mut self, fuel: &mut u64, until: u64) -> SchedStop {
        self.drive(fuel, None, Some(until))
    }

    /// The unified scheduler pump. `step` selects the mode:
    /// - `None` — a plain resume (`continue`/`reverseContinue`): run the policy's pick one op per turn
    ///   ([`dbg_preview_pick`]), stopping on any thread's breakpoint or watchpoint.
    /// - `Some((st, max))` — step thread `st` (sharing the turns with the others, or alone when
    ///   [`single_thread`](Self::set_single_thread); see [`dbg_preview_pick`]), stopping the
    ///   moment `st` reaches a call depth `<= max` at an instruction (`max = None` ⇒ any depth = one
    ///   instruction = step-*in*). Another thread's breakpoint/watchpoint still interrupts a step.
    ///
    /// `pause_at` ends a plain resume at that turn, as a [`SchedBreak::Pause`] stop.
    fn drive(
        &mut self,
        fuel: &mut u64,
        step: Option<(usize, Option<usize>)>,
        pause_at: Option<u64>,
    ) -> SchedStop {
        let Self {
            source,
            table,
            mem,
            host,
            tasks,
            extra_envs,
            fibers,
            funcs,
            breakpoints,
            watchpoints,
            value_watches,
            journal,
            journal_policy,
            access_sink,
            sched_trace,
            sched_sink,
            sched_seed,
            forced,
            step_spans,
            single_thread,
            scheduled_writes,
            write_cursor,
            last_watch,
            fn_block_base,
            fn_block_types,
            debug,
            stopped,
            focus,
            turn,
            clock,
            ..
        } = self;
        *stopped = None;
        // #1366: admit host-completed punts on this run's host, so an offloadable cap the embedder
        // services asynchronously parks the thread (`CapParked`) instead of being waited inline.
        // The single-vCPU engine does the same at the top of every advance; without it
        // `wait_unless_host_owned` never yields `None` and this engine could not park at all.
        host.completions().allow_host_completed();
        loop {
            if let DbgTaskState::Done(res) = &tasks[0].state {
                return SchedStop::Finished(res.clone());
            }
            dbg_refund_ended_windows(tasks, host, extra_envs);
            // A task mid-coroutine is pinned (atomic resume); otherwise prefer the stepping thread while
            // it is runnable (so a step stays on it and a step-over runs its own call), else the
            // lowest-index runnable thread (advancing the futex clock to wake a waiter when the set is
            // stuck; unblocks a stepped `join`/`wait`).
            let pre_pick =
                (sched_trace.is_some() || sched_sink.is_some()).then(|| trace_tags(tasks));
            // Precedence: the coroutine pin (an atomicity constraint) > a forced switch recorded
            // for this turn (explicit user intent) > the stepping thread — this step's, or on a
            // resume past a recorded step, that step's span (#1942) > the policy pick.
            let pref = step
                .map(|(st, _)| (st, *single_thread))
                .or_else(|| span_at(step_spans, *turn));
            let ti = if let Some(p) = dbg_pinned_coro(tasks) {
                p
            } else {
                match dbg_pick_runnable(tasks, clock, *sched_seed, forced, pref, *turn) {
                    Some(i) => i,
                    // Nothing runnable and no timed waiter: a thread parked in a blocking-stdin
                    // read (#1146 deeper) makes this a live `StdinPark` stop on that thread (the
                    // lowest-index one), else a true deadlock.
                    None => {
                        // #1366: a thread parked on a host-completed cap is a live `CapPark`
                        // stop on that thread (lowest-index), resumable via `deliver_cap`.
                        if let Some((p, id, at)) =
                            tasks.iter().enumerate().find_map(|(i, t)| match t.state {
                                DbgTaskState::CapParked { id, at, .. } => Some((i, id, at)),
                                _ => None,
                            })
                        {
                            // The stop location is the call itself, falling back to the live pc.
                            let pc = at.or_else(|| tasks[p].vt.debug_active().cur_ir_pc(source));
                            if let Some(pc) = pc {
                                *stopped = Some(p);
                                *focus = p;
                                return SchedStop::CapPark { id, pc };
                            }
                        }
                        let parked = tasks
                            .iter()
                            .position(|t| matches!(t.state, DbgTaskState::BlockedStdin));
                        let pc = parked.and_then(|p| tasks[p].vt.debug_active().cur_ir_pc(source));
                        return match (parked, pc) {
                            (Some(p), Some(pc)) => {
                                *stopped = Some(p);
                                *focus = p;
                                SchedStop::StdinPark { pc }
                            }
                            _ => SchedStop::Blocked,
                        };
                    }
                }
            };
            // Slice 6: the only transition a pick causes is a timed-out wait waking.
            if let Some(before) = pre_pick.as_ref() {
                let mut events = Vec::new();
                trace_pick_diff(before, tasks, *turn, &mut events);
                record_sched(sched_trace, sched_sink, events);
            }
            // Pre-op stop checks (breakpoint / watchpoint), skipped for a thread that just reported (it
            // must make progress off its current op first, so a loop-body stop re-fires each iteration).
            // Scan the task's *active continuation* — the §14 coroutine child (over its confined window)
            // when this task is mid-`resume`, else its own vCPU — so a breakpoint fires inside a coroutine
            // body on the right thread.
            if !tasks[ti].at_bp {
                let hit = {
                    let cur_vm = tasks[ti].vt.debug_active();
                    let cur_mem: &Option<Mem> = match tasks[ti].env {
                        None => &*mem,
                        Some(k) => &extra_envs[k].mem,
                    };
                    match cur_vm.cur_ir_pc(source) {
                        Some(pc) if breakpoints.contains(&pc) => Some((pc, None)),
                        // A window-range watch (cross-thread) stops *before* the access; a value
                        // watch (#1229) stops when a watched SSA-held variable's value changed.
                        Some(pc) => watch_stop_before(
                            cur_vm,
                            cur_mem,
                            funcs,
                            fn_block_base,
                            watchpoints,
                            value_watches,
                            pc,
                        )
                        .map(|w| (pc, Some(w))),
                        None => None,
                    }
                };
                if let Some((pc, watch)) = hit {
                    let reason = match watch {
                        Some((addr, write)) => {
                            *last_watch = Some((addr, write));
                            SchedBreak::Watchpoint { addr, write }
                        }
                        None => SchedBreak::Breakpoint,
                    };
                    tasks[ti].at_bp = true;
                    *stopped = Some(ti);
                    *focus = ti;
                    return SchedStop::Break { pc, reason };
                }
                // The step target: the stepping thread is at an instruction at a qualifying call
                // depth. Checked *after* the watch scan, so a watch stops a step mid-line (parity
                // with `continue`), and — like the scan — not for the
                // op the thread must first step off. Depth is cumulative across a coroutine / §22
                // invoke boundary (`VTask::debug_depth`), so a step-over of a `resume`/`invoke` runs
                // the child to completion and a step inside its body compares child-local frames.
                // A budgeted run's turn is up: stop here, as a breakpoint would, so resuming steps off
                // this op first.
                if pause_at.is_some_and(|t| *turn >= t) {
                    if let Some(pc) = tasks[ti].vt.debug_active().cur_ir_pc(source) {
                        tasks[ti].at_bp = true;
                        *stopped = Some(ti);
                        *focus = ti;
                        return SchedStop::Break {
                            pc,
                            reason: SchedBreak::Pause,
                        };
                    }
                }
                if let Some((st, max_depth)) = step {
                    if ti == st && max_depth.is_none_or(|m| tasks[st].vt.debug_depth() <= m) {
                        if let Some(pc) = tasks[st].vt.debug_active().cur_ir_pc(source) {
                            *stopped = Some(st);
                            *focus = st;
                            return SchedStop::Break {
                                pc,
                                reason: SchedBreak::Step,
                            };
                        }
                    }
                }
            }
            apply_due_writes(
                scheduled_writes,
                write_cursor,
                *turn,
                tasks,
                source,
                mem,
                debug.as_ref(),
                fn_block_base,
                fn_block_types,
                journal,
            );
            if let Some(sink) = access_sink.as_mut() {
                let cur_vm = tasks[ti].vt.debug_active();
                emit_access(cur_vm, source, funcs, fn_block_base, *turn, ti, sink);
            }
            // #1557: pre-images of what this op is about to overwrite, so a later `undo_to` can put
            // them back without replaying. Must precede the advance — the bytes are gone after it.
            journal_op(
                journal,
                tasks[ti].vt.debug_active(),
                source,
                funcs,
                fn_block_base,
                *turn,
                mem,
            );
            journal_state(
                journal,
                tasks,
                extra_envs,
                fibers,
                source,
                host,
                *clock,
                *turn,
                journal_policy,
            );
            journal.apply_policy(*turn, journal_policy);
            // Slice 6: the turn record + the pre-advance snapshot the park/wake differ compares.
            let trace_turn = *turn;
            let tracing = sched_trace.is_some() || sched_sink.is_some();
            let pre_adv = tracing.then(|| trace_tags(tasks));
            let pre_depth = trace_depth(&tasks[ti]);
            let pre_threads = if tracing {
                tasks[ti].threads.clone()
            } else {
                Vec::new()
            };
            if tracing {
                let event = SchedTraceEvent::Turn {
                    turn: trace_turn,
                    task: ti,
                };
                record_sched(sched_trace, sched_sink, vec![event]);
            }
            if let Serviced::Declined = service_advance(
                tasks, ti, extra_envs, fibers, source, table, fuel, mem, host, *clock, turn,
            ) {
                // A coroutine / tier-up op this engine does not drive — bail, `turn` untouched.
                return SchedStop::Declined;
            }
            // Slice 6: derive the park/wake/spawn edges this advance caused (see `trace_diff`).
            if let Some(before) = pre_adv.as_ref() {
                let mut events = Vec::new();
                trace_diff(
                    before,
                    pre_depth,
                    &pre_threads,
                    tasks,
                    trace_turn,
                    ti,
                    &mut events,
                );
                record_sched(sched_trace, sched_sink, events);
            }
        }
    }

    /// The thread a step drives (#1942): the focused one — the thread a client named with
    /// [`select_task`](Self::select_task) — while it is live, else the stopped one. `None` before the
    /// first stop.
    fn step_thread(&self) -> Option<usize> {
        let st = self.stopped?;
        let live = |i: usize| {
            self.tasks
                .get(i)
                .is_some_and(|t| !matches!(t.state, DbgTaskState::Done(_)))
        };
        Some(if live(self.focus) { self.focus } else { st })
    }

    /// Step [the stepping thread](Self::step_thread) until its call depth is `<= max_depth` (`None` ⇒
    /// any = one instruction). The
    /// shared driver for the stepping verbs (step off the current op first, then seek the next
    /// qualifying stop). The other threads share the turns meanwhile unless
    /// [`set_single_thread`](Self::set_single_thread) froze them.
    ///
    /// The step is recorded as a span (#1942) so a replay runs the same thread over the same turns.
    /// A step taken from an earlier turn (after a step back) rewrites the future, so the spans at or
    /// past this turn go first.
    fn step_to(&mut self, max_depth: Option<usize>, fuel: &mut u64) -> SchedStop {
        let Some(st) = self.step_thread() else {
            return self.run_until_stop(fuel);
        };
        self.tasks[st].at_bp = true; // step *off* the current op first, then seek the next stop
        let from = self.turn;
        truncate_spans(&mut self.step_spans, from);
        let stop = self.drive(fuel, Some((st, max_depth)), None);
        let span = StepSpan {
            from,
            to: self.turn,
            task: st,
            single_thread: self.single_thread,
        };
        push_span(&mut self.step_spans, span);
        stop
    }

    /// Whether the steps that follow keep the other threads frozen (DAP `singleThread`) — the step
    /// moves only its own thread. Off by default: the other runnable threads share the turns while
    /// a step runs (see [`dbg_preview_pick`]).
    pub fn set_single_thread(&mut self, on: bool) {
        self.single_thread = on;
    }

    /// Replace the recorded **step spans** (#1942) — the DAP backend re-installs them on every rebuild.
    pub fn set_step_spans(&mut self, spans: Vec<StepSpan>) {
        self.step_spans = spans;
    }

    /// The recorded step spans, for the backend to carry across rebuilds.
    pub fn step_spans(&self) -> &[StepSpan] {
        &self.step_spans
    }

    /// **Step** one instruction — descends into a call. Drives the stepping thread; other threads
    /// stay frozen.
    pub fn step(&mut self, fuel: &mut u64) -> SchedStop {
        self.step_to(None, fuel)
    }

    /// The stopped thread's **cumulative** call depth (the child's frames count above the parent's resume
    /// frame while it is mid-coroutine or mid-invoke — [`VTask::debug_depth`]). Used by the depth-bounded verbs so
    /// they treat a coroutine `resume` boundary like an ordinary call.
    fn step_depth(&self, s: usize) -> usize {
        self.tasks[s].vt.debug_depth()
    }

    /// **Step over** the next source op: run any call it makes to completion (schedule advances only if
    /// the stepped thread blocks), landing at the next op at the same call depth.
    pub fn step_over(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self.step_thread().map(|s| self.step_depth(s));
        self.step_to(max, fuel)
    }

    /// **Step out** — run until the stepped thread's current function returns (one call depth shallower).
    pub fn step_out(&mut self, fuel: &mut u64) -> SchedStop {
        let max = self
            .step_thread()
            .map(|s| self.step_depth(s).saturating_sub(1));
        self.step_to(max, fuel)
    }

    /// Advance the schedule by exactly one visible op (the raw time quantum for replay-based reverse
    /// `seek` — DEBUGGING.md W1), honoring **no** breakpoint/watch/step checks: the lowest-index runnable
    /// thread runs one op, `turn` ticks. Returns `false` once the root has finished (or the schedule can
    /// no longer advance — blocked/unsupported). Because the debug schedule is deterministic (pure
    /// compute, one-op-per-turn, lowest-index pick), replaying `t` ticks from a fresh session reproduces
    /// the exact state at global turn `t`.
    pub fn tick(&mut self, fuel: &mut u64) -> bool {
        if matches!(self.tasks[0].state, DbgTaskState::Done(_)) {
            return false;
        }
        let Self {
            source,
            table,
            mem,
            host,
            tasks,
            extra_envs,
            fibers,
            turn,
            clock,
            funcs,
            fn_block_base,
            fn_block_types,
            debug,
            journal,
            journal_policy,
            access_sink,
            sched_trace,
            sched_sink,
            sched_seed,
            forced,
            step_spans,
            scheduled_writes,
            write_cursor,
            ..
        } = self;
        host.completions().allow_host_completed(); // #1366: see `drive`
                                                   // A task mid-coroutine is pinned (atomic resume — the same vCPU runs the whole body); the same
                                                   // pin on replay reconstructs the coroutine's op sequence deterministically. The policy pick
                                                   // (seed + forced) matches `drive`'s, so a tick-replay reproduces the interactive schedule.
                                                   // A `CapParked` task is not runnable, so a parked run refuses to tick — as the single engine's.
        let pre_pick = (sched_trace.is_some() || sched_sink.is_some()).then(|| trace_tags(tasks));
        let Some(ti) = dbg_pinned_coro(tasks).or_else(|| {
            let pref = span_at(step_spans, *turn); // #1942: replay a recorded step's thread
            dbg_pick_runnable(tasks, clock, *sched_seed, forced, pref, *turn)
        }) else {
            return false; // no runnable thread and no waiter (deadlock) — can't advance
        };
        if let Some(before) = pre_pick.as_ref() {
            let mut events = Vec::new();
            trace_pick_diff(before, tasks, *turn, &mut events);
            record_sched(sched_trace, sched_sink, events);
        }
        apply_due_writes(
            scheduled_writes,
            write_cursor,
            *turn,
            tasks,
            source,
            mem,
            debug.as_ref(),
            fn_block_base,
            fn_block_types,
            journal,
        );
        if let Some(sink) = access_sink.as_mut() {
            let cur_vm = tasks[ti].vt.debug_active();
            emit_access(cur_vm, source, funcs, fn_block_base, *turn, ti, sink);
        }
        // #1557: the raw replay quantum journals too, so a `tick`-driven replay builds the identical
        // history a `drive`-driven run does (the two must not disagree — DEBUGGING.md, `service_advance`).
        journal_op(
            journal,
            tasks[ti].vt.debug_active(),
            source,
            funcs,
            fn_block_base,
            *turn,
            mem,
        );
        journal_state(
            journal,
            tasks,
            extra_envs,
            fibers,
            source,
            host,
            *clock,
            *turn,
            journal_policy,
        );
        journal.apply_policy(*turn, journal_policy);
        let trace_turn = *turn;
        let tracing = sched_trace.is_some() || sched_sink.is_some();
        let pre_adv = tracing.then(|| trace_tags(tasks));
        let pre_depth = trace_depth(&tasks[ti]);
        let pre_threads = if tracing {
            tasks[ti].threads.clone()
        } else {
            Vec::new()
        };
        if tracing {
            let event = SchedTraceEvent::Turn {
                turn: trace_turn,
                task: ti,
            };
            record_sched(sched_trace, sched_sink, vec![event]);
        }
        if let Serviced::Declined = service_advance(
            tasks, ti, extra_envs, fibers, source, table, fuel, mem, host, *clock, turn,
        ) {
            // An unsupported op — tick this engine's clock (as it did unconditionally before) and
            // stop the replay here.
            *turn += 1;
            return false;
        }
        // Slice 6: the park/wake/spawn edges this replayed op caused (identical to `drive`'s,
        // so a `tick`-replay refills the tape deterministically).
        if let Some(before) = pre_adv.as_ref() {
            let mut events = Vec::new();
            trace_diff(
                before,
                pre_depth,
                &pre_threads,
                tasks,
                trace_turn,
                ti,
                &mut events,
            );
            record_sched(sched_trace, sched_sink, events);
        }
        !matches!(tasks[0].state, DbgTaskState::Done(_))
    }

    /// The current global turn (visible ops replayed so far) — the reverse-`seek` coordinate.
    pub fn op_turn(&self) -> u64 {
        self.turn
    }

    /// The powerbox host backing this run — for reading effects a debugged multithreaded guest
    /// produced (captured stdout) and its [`CapTape`](Host::cap_tape) so a reverse `seek` rebuild
    /// replays identical cap inputs.
    pub fn host(&self) -> &Host {
        &self.host
    }
    /// Mutable powerbox host — e.g. to drain captured stdout between stops.
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// #1146 (deeper) — whether some thread is parked in a blocking-stdin `read` (W4): the run is live,
    /// paused at that read, and resumable once [`provide_stdin`](ScheduledDebugRun::provide_stdin)
    /// supplies bytes.
    pub fn stdin_parked(&self) -> bool {
        self.tasks
            .iter()
            .any(|t| matches!(t.state, DbgTaskState::BlockedStdin))
    }

    /// #1146 (deeper) — append stdin bytes for the parked blocking `read`s ([`Host::push_stdin`]) and
    /// re-admit every stdin-parked thread: the next advance re-issues each read against the new
    /// bytes, and the completed read joins the recorded cap tape so a later `seek` replays it
    /// faithfully. The wake is explicit (no
    /// readiness poll) so the schedule stays a pure function of the recorded inputs.
    pub fn provide_stdin(&mut self, bytes: &[u8]) {
        self.host.push_stdin(bytes);
        for t in self.tasks.iter_mut() {
            if matches!(t.state, DbgTaskState::BlockedStdin) {
                t.state = DbgTaskState::Runnable;
            }
        }
    }

    /// #1366 — the completion id some thread is parked on (a host-completed cap call), if any: the
    /// lowest-index parked thread's. The backend reports it as `StopReason::CapPark { id }`;
    /// [`deliver_cap`](ScheduledDebugRun::deliver_cap) resumes it.
    pub fn cap_parked(&self) -> Option<u64> {
        self.tasks.iter().find_map(|t| match t.state {
            DbgTaskState::CapParked { id, .. } => Some(id),
            _ => None,
        })
    }

    /// #1366 — the pc of the host-completed cap call the parked thread is on (the stop location), if
    /// parked and the call had a source position.
    pub fn cap_park_pc(&self) -> Option<super::IrPc> {
        self.tasks.iter().find_map(|t| match t.state {
            DbgTaskState::CapParked { at, .. } => at,
            _ => None,
        })
    }

    /// #1366 — finish the host-completed cap call a thread is parked on: `value` lands in the call's
    /// result slot, the completion settles, the delivered value joins the cap tape as the call's record
    /// (so a reverse `seek` replays it without re-parking), the thread is re-admitted, and the op
    /// counts on the turn. `false` if no thread is parked on `id`. The twin of `provide_stdin`, and
    /// op-for-op the same delivery the single-vCPU engine performs.
    pub fn deliver_cap(&mut self, id: u64, value: i64) -> bool {
        let Some((ti, dst)) = self
            .tasks
            .iter()
            .enumerate()
            .find_map(|(i, t)| match t.state {
                DbgTaskState::CapParked { id: pid, dst, .. } if pid == id => Some((i, dst)),
                _ => None,
            })
        else {
            return false;
        };
        let comps = self.host.completions();
        let prefix = comps.complete_host(id, value);
        let _ = comps.try_take(id);
        if let Some((type_id, op, handle, args)) = prefix {
            self.host.tape_cap_record(super::CapRecord {
                type_id,
                op,
                handle,
                args,
                result: Ok(vec![value]),
                mem_writes: Vec::new(),
            });
        }
        self.tasks[ti].vt.active.set(dst, Reg::from_i64(value));
        self.tasks[ti].state = DbgTaskState::Runnable;
        self.turn += 1;
        true
    }

    /// Position the session at the current schedule point after a raw `tick`-replay `seek`: the stopped +
    /// focused thread becomes the one the schedule runs next, or none once the run finished.
    pub fn locate(&mut self) {
        // The thread the recorded schedule runs next (#1942) — after a step back, the thread whose
        // step was undone — rather than the lowest-index runnable one.
        let pref = span_at(&self.step_spans, self.turn);
        let next = dbg_preview_pick(&self.tasks, self.sched_seed, &self.forced, pref, self.turn);
        self.stopped = next;
        self.focus = next.unwrap_or(0);
    }

    /// After a `seek` landed exactly on a breakpoint op, arm the stopped thread's skip so a forward
    /// resume steps past it instead of immediately re-reporting the same stop.
    pub fn arm_breakpoint_skip(&mut self) {
        if let Some(st) = self.stopped {
            self.tasks[st].at_bp = true;
        }
    }

    /// Whether the scheduled continuation is fully captured by the per-task active `Vm`s + the shared
    /// window bytes + the host substate + the scheduler clocks — the subset a multi-vCPU time-travel
    /// **checkpoint** (W1) snapshots, over every task:
    /// **§12 fibers** are admitted (the run-shared registry + each task's active fiber / resume chain,
    /// all sharing the run's window) except an event-parked (`memory.wait`) fiber (non-deterministic
    /// wall-clock deadline); **§14 coroutines** are admitted (not demand, pristine `nested_view`),
    /// same-module *or* separate-module (a separate module's pushed unit rides in `extra_units`); and
    /// **§14 `instantiate` / `instantiate_module` children** are admitted (each [`DbgEnv`] a `nested_view`
    /// over the shared backing + a deterministic `Instantiator`/`AddressSpace` powerbox + a natural table
    /// over the child's module, rebuilt on restore) — this is what admits scheduled coroutines, which only
    /// ever arise alongside an `instantiate` sibling (the bytecode engine rejects `coroutine + thread`).
    /// Both coroutines and children may be **demand**/self-page-mapping: each child's own page map is
    /// captured (`child_checkpointable`: `layout_snapshot_safe`, no §13 regions, within the parent's
    /// prefix) and its bytes ride in the shared snapshot. Still excluded (→ replay-from-turn-0): a
    /// region-aliased child, or one carved beyond the parent's captured prefix.
    fn checkpointable(&self) -> bool {
        self.host.checkpoint_safe()
            && self.mem.as_ref().is_none_or(|m| m.layout_snapshot_safe())
            // A task mid-§22-invoke is out-of-subset (CONSOLIDATION.md §11 debug boundary).
            && self.tasks.iter().all(|t| t.vt.active_invoke.is_none())
            && !self
                .fibers
                .iter()
                .chain(self.extra_envs.iter().flat_map(|e| e.fibers.iter()))
                .any(|f| {
                    matches!(
                        f,
                        FiberState::WaitParked { .. }
                            | FiberState::CapParked { .. }
                            | FiberState::HostParked { .. }
                    )
                })
            && self.extra_envs.iter().all(|e| {
                e.host.checkpoint_safe() && child_checkpointable(e.mem.as_ref(), self.mem.as_ref())
            })
    }

    /// Snapshot the scheduled continuation at the current [`turn`](ScheduledDebugRun::op_turn) for the
    /// backend's checkpoint ladder — `None` outside the [`checkpointable`](ScheduledDebugRun::checkpointable)
    /// subset. Captures each task's active `Vm` + join table + state, the shared window bytes, the host
    /// replay substate, and both scheduler clocks; the transient `stopped`/`focus`/`last_watch` are
    /// *not* captured — [`locate`](ScheduledDebugRun::locate) rederives them from the task states.
    pub fn snapshot(&self) -> Option<ScheduledSnapshot> {
        if !self.checkpointable() {
            return None;
        }
        let continuation = self.build_continuation();
        Some(super::moment::Moment::new(
            self.mem.as_ref().map(|m| m.layout_snapshot()),
            &self.host,
            super::moment::Continuation::Bytecode(continuation),
        ))
    }

    /// The continuation half of a capture: every task's `Vm` + join table + state, the run-shared fiber
    /// registry, the §14 child envs, and the pushed source units. Shared by the checkpoint ladder
    /// ([`snapshot`](Self::snapshot), which pairs it with the window image and the host substate) and by
    /// the **undo journal** (#1557), which pairs it with a compact [`HostCursor`](crate::HostCursor)
    /// and the window pre-images instead — one definition of "what the continuation is", two costs.
    fn build_continuation(&self) -> ScheduledContinuation {
        ScheduledContinuation {
            clock: self.clock,
            tasks: self
                .tasks
                .iter()
                .map(|t| DbgTaskSnapshot {
                    active: t.vt.active.clone(),
                    active_id: t.vt.active_id,
                    chain: t.vt.chain.clone(),
                    root_shadow_sp: t.vt.root_shadow_sp,
                    threads: t.threads.clone(),
                    env: t.env,
                    state: t.state.clone(),
                    at_bp: t.at_bp,
                    lease: t.lease,
                })
                .collect(),
            fibers: self.fibers.clone(),
            // Each child env's module = the module its owning task runs (`0` = same-module `instantiate`;
            // `>= 1` = a separate-module `instantiate_module` child), so its table rebuilds correctly.
            extra_envs: self
                .extra_envs
                .iter()
                .enumerate()
                .map(|(k, e)| {
                    let module = self
                        .tasks
                        .iter()
                        .find(|t| t.env == Some(k))
                        .map_or(0, |t| t.vt.active.module);
                    env_snapshot(e, module)
                })
                .collect(),
            extra_units: self.source.extra_units(),
        }
    }

    /// Restore a [`snapshot`](ScheduledDebugRun::snapshot) into this **freshly built** run (its
    /// breakpoints/watchpoints re-armed by the backend before this call), so a subsequent `tick`-replay
    /// resumes exactly at the snapshot's global turn rather than turn 0. Rebuilds the task set (each a
    /// root-only `VTask` around the captured active `Vm`), reseeds the shared window bytes, restores the
    /// host substate and both scheduler clocks, and clears the transient stop state (`locate` rederives
    /// it). A separate-module coroutine/child's pushed source units are re-pushed first (so its `module`
    /// index resolves); the run-shared fibers and each child env are rebuilt from the snapshot.
    /// `turn` is the global turn the snapshot was taken at — the ladder's key, handed back with it.
    pub fn restore(&mut self, turn: u64, snap: &ScheduledSnapshot) {
        let c = snap
            .continuation()
            .as_bytecode()
            .expect("a scheduled seek ladder holds only Bytecode moments");
        if let (Some(m), Some(layout)) = (self.mem.as_mut(), snap.mem()) {
            m.restore_layout(layout);
        }
        self.install_continuation(c, true);
        snap.restore_host(&mut self.host);
        self.turn = turn;
        self.clock = c.clock;
        self.locate(); // stepping-ready: the stop state rederives from the restored task states
        self.last_watch = None;
    }

    /// Install a [`build_continuation`](Self::build_continuation) capture: the fiber registry, the
    /// pushed source units, the §14 child envs, and every task's `VTask`.
    ///
    /// `readmit_parks` is the one place the two callers differ. A **checkpoint restore** rebuilds the
    /// run and replays forward, so a captured blocking-stdin / host-completed park is re-admitted
    /// (`Runnable`) and its read re-executes, served from the cap tape (#1146 deeper). An **undo**
    /// rewinds in place rather than re-executing, so it installs task states **verbatim** — a run that
    /// was parked at that turn must come back parked, not silently runnable.
    fn install_continuation(&mut self, c: &ScheduledContinuation, readmit_parks: bool) {
        self.fibers = c.fibers.clone();
        // Re-push any separate-module units before rebuilding envs/coroutines (their `module` indices
        // resolve against the source).
        self.source.reset_extra(&c.extra_units);
        // Rebuild each task's full `VTask` and each §14 `instantiate`-child env. Coroutine and child
        // windows are `nested_view`s over the shared window (their bytes — shared via the backing
        // region — are already correct); each table is rebuilt over the child's own module.
        let shared_mem = self.mem.as_ref();
        let source = &*self.source;
        self.extra_envs = c
            .extra_envs
            .iter()
            .map(|es| rebuild_env(es, shared_mem, source))
            .collect();
        self.tasks = c
            .tasks
            .iter()
            .map(|ts| DbgTask {
                vt: VTask {
                    active: ts.active.clone(),
                    active_id: ts.active_id,
                    chain: ts.chain.clone(),
                    root_shadow_sp: ts.root_shadow_sp,
                    active_invoke: None, // never captured mid-invoke (`checkpointable`)
                },
                threads: ts.threads.clone(),
                env: ts.env,
                state: match (&ts.state, readmit_parks) {
                    (DbgTaskState::BlockedStdin | DbgTaskState::CapParked { .. }, true) => {
                        DbgTaskState::Runnable
                    }
                    (s, _) => s.clone(),
                },
                at_bp: ts.at_bp,
                lease: ts.lease,
            })
            .collect();
    }

    /// The run's result once the root has finished (`None` while still running).
    pub fn result(&self) -> Option<&Result<Vec<Value>, Trap>> {
        match &self.tasks[0].state {
            DbgTaskState::Done(r) => Some(r),
            _ => None,
        }
    }

    /// Every live (not-yet-finished) vCPU — one DAP thread each. The stopped thread is among them.
    pub fn threads(&self) -> Vec<u64> {
        (0..self.tasks.len())
            .filter(|&i| !matches!(self.tasks[i].state, DbgTaskState::Done(_)))
            .map(|i| i as u64)
            .collect()
    }

    /// What each blocked thread is waiting on (#1986): the futex word a `memory.wait` parked on, or
    /// the thread a `thread.join` waits for. After a deadlock this is the wait-for graph, less the
    /// owners, which only the guest's own sync objects can name (an embedder reads them out of the
    /// words).
    pub fn blocked_on(&self) -> Vec<(usize, BlockedOn)> {
        self.tasks
            .iter()
            .enumerate()
            .filter_map(|(i, t)| match t.state {
                DbgTaskState::BlockedWait { addr, .. } => Some((i, BlockedOn::Futex(addr))),
                DbgTaskState::BlockedJoin { child, .. } => Some((i, BlockedOn::Join(child))),
                _ => None,
            })
            .collect()
    }

    /// The thread index currently paused on a breakpoint (drives stepping); `None` while running.
    pub fn stopped_task(&self) -> Option<u64> {
        self.stopped.map(|i| i as u64)
    }

    /// Focus read-inspection (`backtrace`/`read_var`/`read_window`) on a live thread; `false` if `id`
    /// is not a live task. Resets to the stopped thread on the next `run_until_stop`.
    pub fn select_task(&mut self, id: u64) -> bool {
        let i = id as usize;
        if i < self.tasks.len() && !matches!(self.tasks[i].state, DbgTaskState::Done(_)) {
            self.focus = i;
            true
        } else {
            false
        }
    }

    /// The scheduled-mode logical clock (visible ops across all vCPUs).
    pub fn turn(&self) -> u64 {
        self.turn
    }

    /// The memory window a task steps against: its confined `instantiate` env (`Some(k)`) or the shared
    /// mem — so inspection of a focused child reads its own confined window.
    fn task_mem(&self, ti: usize) -> &Option<Mem> {
        match self.tasks[ti].env {
            None => &self.mem,
            Some(k) => &self.extra_envs[k].mem,
        }
    }

    /// A [`FrameReader`] over the **focused** thread's currently-stepping `Vm` (what `select_task` chose):
    /// the thread's own vCPU over its window (a confined `instantiate` child reads its `nested_view`).
    fn reader(&self) -> FrameReader<'_> {
        let vm = self.tasks[self.focus].vt.debug_active();
        FrameReader {
            vm,
            source: &self.source,
            mem: self.task_mem(self.focus),
            debug: self.debug.as_ref(),
            fn_block_base: &self.fn_block_base,
            fn_block_types: &self.fn_block_types,
            // A separate-module coroutine on the scheduled engine carries its own §6 metadata (built at
            // spawn); a same-module one leaves it `None` (its frames are module 0, read against the fields
            // above).
            coro_debug: None,
            finished: matches!(self.tasks[self.focus].state, DbgTaskState::Done(Ok(_))),
        }
    }

    /// Call-stack depth of the focused thread.
    pub fn depth(&self) -> usize {
        self.reader().depth()
    }

    /// The `IrPc` of the focused thread's frame `depth` levels from the top.
    pub fn frame_pc(&self, depth: usize) -> Option<super::IrPc> {
        self.reader().frame_pc(depth)
    }

    /// Block-local SSA value `idx` in the focused thread's frame `depth` levels from the top, typed —
    /// the bytecode counterpart of `Inspector::read_ir_value`. `None` for a cross-module frame, a bad
    /// `idx`, or past the stack. A not-yet-computed slot reads as its default; the caller compares only
    /// the defined prefix (where `read_ir_value` returns `Some`).
    pub fn value_in_frame(&self, depth: usize, idx: usize) -> Option<Value> {
        self.reader().value_in_frame(depth, idx)
    }

    /// The focused thread's running frame's block-local SSA value `idx` ([`value_in_frame`] at depth 0).
    pub fn value(&self, idx: usize) -> Option<Value> {
        self.value_in_frame(0, idx)
    }

    /// Read a source variable by name in the focused thread's frame `depth` levels from the top.
    pub fn read_var(&self, depth: usize, name: &str, width: usize) -> Option<VarValue> {
        self.reader().read_var(depth, name, width)
    }

    /// The window address of a memory-located source variable in the focused thread's frame `depth`.
    pub fn var_addr(&self, depth: usize, name: &str) -> Option<u64> {
        self.reader().var_addr(depth, name)
    }

    /// **Write a source variable by name** in the **focused** thread's frame (slice 8, the DAP
    /// `setVariable` backend): a promoted SSA scalar takes `value` coerced to its slot type (integers
    /// only); a memory-located var takes `value`'s low `width` bytes little-endian at its resolved
    /// window address. Refused (`false`) for float slots or an unresolvable name — fail-closed, never a
    /// guess. The DAP backend records successful writes and re-applies them at the same turn on every
    /// seek replay.
    pub fn write_var(&mut self, depth: usize, name: &str, value: i64, width: usize) -> bool {
        let focus = self.focus;
        if self.tasks.get(focus).is_none() {
            return false;
        }
        let Some(target) = self.reader().write_target(depth, name) else {
            return false;
        };
        match target {
            WriteTarget::Ssa { reg, ty } => {
                let v = match ty {
                    ValType::I32 => Value::I32(value as i32),
                    ValType::I64 => Value::I64(value),
                    _ => return false,
                };
                match self.tasks[focus].vt.active.regs.get_mut(reg) {
                    Some(r) => {
                        *r = Reg::from_value(v);
                        true
                    }
                    None => false,
                }
            }
            WriteTarget::Win { addr } => {
                let w = width.clamp(1, 8);
                self.write_window(addr, &value.to_le_bytes()[..w])
            }
        }
    }

    /// **Write bytes into the shared guest window** (slice 8, the DAP `writeMemory` backend). `false`
    /// if the range is unmapped or the module has no memory.
    pub fn write_window(&mut self, addr: u64, bytes: &[u8]) -> bool {
        let turn = self.turn;
        self.mem
            .as_mut()
            .is_some_and(|m| journaled_write(&mut self.journal, turn, m, addr, bytes))
    }

    /// Read `len` bytes from the focused thread's guest window at `addr`: the active coroutine child's
    /// confined window when mid-`resume`, else the thread's own window (its confined `instantiate`
    /// window or the shared mem).
    pub fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match self.task_mem(self.focus).as_ref() {
            Some(m) => m.read_window(addr, len),
            None => Err(Trap::Malformed),
        }
    }

    /// The faulting guest address of the focused task's last `MemoryFault` (window-relative; a NULL
    /// deref → `0`), or `None` if it was not an address-recording memory fault. Read by the DAP
    /// layer to report a segfault's address (#1190).
    pub fn fault_addr(&self) -> Option<u64> {
        self.task_mem(self.focus)
            .as_ref()
            .and_then(|m| m.peek_fault_rel())
    }
}

/// Like [`compile_and_run`], but drives the reified [`Vm`] in slices of at most `slice` ops,
/// suspending and resuming at op boundaries until the entry function completes (or traps). The
/// result must be **bit-identical** to [`compile_and_run`] for any `slice ≥ 1` — that equality is
/// what proves the suspend/resume machinery (Slice 1c-2) preserves the continuation exactly. Test
/// surface for the "interrupt-anywhere" harness; not a production entry point.
pub fn compile_and_run_sliced(
    m: &Module,
    func: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    slice: u64,
) -> Option<Result<Vec<Value>, Trap>> {
    let c = compile_module_for(m)?;
    if func as usize >= c.progs.len() {
        return Some(Err(Trap::Malformed));
    }
    let dom = Domain::new(c, 0);
    let mut mem = build_mem(m, &[]);
    let mut host = Host::new();
    Some(drive(
        dom,
        func,
        args,
        fuel,
        &mut mem,
        &mut host,
        slice.max(1),
    ))
}

/// What an exec rebuilt its process into ([`exec_image_build`]).
struct ExecBuilt {
    host: Host,
    table: SharedSlots,
    vt: VTask,
    /// The window the image runs in, which replaces the caller's ([`Mem::exec_window`]).
    mem: Mem,
    /// #1896 — how a leaf image the host emitted tiers up at its entry (its window is flat, where
    /// emitted code can address it). `None`: the image runs interpreted.
    leaf: Option<LeafStart>,
}

/// #1896 — a leaf image tiers up at its entry: the host runs it whole, emitted, and its delivery ends
/// the process. The step that starts task `ti`'s, over its window `win` — or `None` over a window one
/// bound cannot describe, where an image emitted without the page check runs interpreted instead,
/// over the same window.
fn leaf_step(
    start: LeafStart,
    ti: usize,
    win: Option<&Mem>,
    pending_tierup: &mut Option<(usize, TierUpDst, Box<[ValType]>)>,
) -> Option<CoopStep> {
    let mapped = win.and_then(|m| match start.paged {
        true => Some(m.reserved_size()),
        false => m.scalar_extent(),
    })?;
    *pending_tierup = Some((ti, TierUpDst::Entry { parks: start.parks }, start.results));
    Some(CoopStep::TierUp {
        module: start.module,
        func: start.entry,
        argv: start.argv,
        mapped,
    })
}

/// How a leaf image tiers up at its entry ([`CoopStep::TierUp`]).
struct LeafStart {
    module: u32,
    entry: u32,
    argv: Box<[i64]>,
    results: Box<[ValType]>,
    /// The image was emitted with the per-access page check (#750), because it can change its page
    /// state ([`image_pages`]): its `"mapped"` is the window's reservation, and the host keeps the
    /// page-state table.
    paged: bool,
    /// Its host suspends its emitted frames where a call parks ([`LeafOffer::parks`]).
    parks: bool,
}

/// #1896 — how a process running image `m` can **park**: block in an op until another process, or
/// another of its own threads, acts. An emitted frame cannot wait by itself, so an image runs on the
/// emitted tier ([`TierUpConfig::leaf`]) when it cannot park, or when it parks only in a **stream**
/// call and its host can suspend the emitted frames there ([`LeafOffer::parks`]). A stream read
/// parks on a pipe end or a blocking stdin, and a stream write on a pipe end. The process's signal
/// source answers for the ops bound to it ([`super::SignalSource::import_parks`]); the address-space
/// ops cannot park; a §12 concurrency op may (a join or a futex wait parks, and a thread or a fiber
/// needs the interpreter to schedule it); an import `import.attach` may retarget may; any other
/// import or capability call may, a dynamic one included. (A linked program has no `call.sym` left:
/// linking rewrote each to one of these.)
fn image_parks(host: &Host, m: &Module) -> Parks {
    use temen_ir::cap_id::{ADDRESS_SPACE, HOST_PROC, STREAM};
    let source = host.signal_poll().map(|(_, s)| s);
    let import_parks = |(import, b): (&temen_ir::Import, &super::BoundImport)| match b.type_id {
        _ if b.rebindable => true,
        ADDRESS_SPACE => false,
        HOST_PROC => source
            .as_ref()
            .and_then(|s| s.import_parks(&import.name))
            .unwrap_or(true),
        _ => true,
    };
    let holds_pipe = host
        .table
        .iter()
        .any(|s| matches!(s.entry, Some(super::Binding::PipeEnd { .. })));
    // A call that parks other than in a stream call.
    let may_park = |i: &Inst| match *i {
        Inst::CapCall {
            type_id: STREAM,
            op,
            ..
        } => op > 2,
        Inst::CapCall {
            type_id: ADDRESS_SPACE,
            ..
        } => false,
        Inst::CapCall { .. } | Inst::CallImportDyn { .. } => true,
        _ => false,
    };
    let stream_parks = |i: &Inst| match *i {
        Inst::CapCall {
            type_id: STREAM,
            op,
            ..
        } => (op == 0 && (holds_pipe || host.stdin_block)) || (op == 1 && holds_pipe),
        _ => false,
    };
    let insts = || {
        m.funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.insts)
    };
    if m.imports.len() != host.import_bindings.len()
        || m.imports
            .iter()
            .zip(&host.import_bindings)
            .any(import_parks)
        || m.funcs.iter().any(temen_ir::Func::uses_concurrency)
        || insts().any(may_park)
    {
        return Parks::Otherwise;
    }
    match insts().any(stream_parks) {
        true => Parks::InStreams,
        false => Parks::Never,
    }
}

/// How a process running an image can park ([`image_parks`]).
enum Parks {
    Never,
    /// Only in a stream call.
    InStreams,
    Otherwise,
}

/// #1896 — can a process running image `m` change its **page state**: reach an address-space
/// `map`, `unmap` or `protect`, inline or through an import? A map that leaves a hole, or a change
/// of protection, takes its window past what one bound describes ([`Mem::scalar_extent`]), and an
/// image that runs whole on the emitted tier cannot decline there: it checks every access against
/// the page state instead ([`LeafStart::paged`]).
fn image_pages(host: &Host, m: &Module) -> bool {
    let pages = |type_id, op| type_id == temen_ir::cap_id::ADDRESS_SPACE && op <= 2;
    host.import_bindings.iter().any(|b| pages(b.type_id, b.op))
        || m.funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.insts)
            .any(|i| matches!(*i, Inst::CapCall { type_id, op, .. } if pages(type_id, op)))
}

/// FORK.md §8.6 (#1080) — build the `execve` image-replace for the bytecode engine's exec pump arm,
/// given the exec'ing task's current `cur_host` (the old powerbox, drained here) and `cur_mem` (its
/// window). Resolves + compiles the command, admits it (entry sig, command window `<=` the
/// caller's), builds the command powerbox (`spawn_named_child` + [`Host::exec_carry`] — the same
/// personality carry the tree-walker uses), materializes the command image into a fresh window of
/// the caller's geometry ([`Mem::exec_window`]; flat for a leaf image the host emitted), and pushes
/// the compiled command as a new domain unit. Returns the rebuilt process ([`ExecBuilt`]) for the
/// caller to install where the task's `env` points; `Err(())` on any admissibility failure (the
/// caller then writes a probeable `-EINVAL` and lets the task run on — POSIX: execve returns only
/// on failure). The old image's pipe ends are released here ([`Host::release_pipe_ends`]) so the
/// shared counts do not leak; waking any pipe that thereby reached EOF is the tree-walker's job
/// (the cooperative engine has no CorePipe park — pipe-through-exec is a later rung), and is a
/// no-op for a command that inherited none.
#[allow(clippy::too_many_arguments)] // the exec op's operands, plus where the image may run
fn exec_image_build(
    cur_host: &mut Host,
    cur_mem: Option<&Mem>,
    dom: &Domain,
    cmd: super::ExecCmd,
    grants_ptr: u64,
    grants_n: u64,
    entry: u64,
    size_log2: i64,
    personality: bool,
    leaf: Option<&LeafEmitter>,
) -> Result<ExecBuilt, i64> {
    // Resolve and compile the command first: `Host::exec_image` is the commit point (it hands the
    // caller's personality to the new powerbox), so everything that can still refuse must come
    // before it. A command this run already compiled is not compiled again.
    let command = cur_host.exec_module(cmd)?;
    let cm = dom
        .source
        .command(&command.0.digest, || {
            compile_module(&command.0.funcs, &command.0.types, command.0.shadow)
        })
        .ok_or(super::EINVAL)?;
    // Read the by-name grant list (16-byte `{name_off, name_len, handle, flags}` records, the op-13
    // layout) from the caller window, then admit + build through the one rule every engine shares:
    // the command's entry, its fit in the caller's backed prefix, the grants' regrantability, the
    // fresh powerbox and the personality carry.
    let m = cur_mem.ok_or(super::EINVAL)?;
    let grants = super::read_grant_records(grants_ptr, grants_n, |o, l| m.read_window(o, l))
        .map_err(|_| super::EINVAL)?;
    let img = cur_host.exec_image(
        &command,
        &grants,
        entry,
        size_log2,
        m.window.mapped(),
        m.window.reserved(),
    )?;
    let child_args: Vec<Value> = img.entry_args.iter().map(|&h| Value::I64(h)).collect();
    // #1896 — a leaf image the host emitted runs in a flat window, where emitted code can address
    // it. Chosen only after the commit, and only where the image runs: without a flat backing it
    // runs interpreted.
    let leaf = leaf.and_then(|emit| {
        let parks = match image_parks(&img.host, &img.module) {
            Parks::Never => false,
            Parks::InStreams => true,
            Parks::Otherwise => return None,
        };
        let paged = image_pages(&img.host, &img.module);
        let offer = LeafOffer {
            module: cm,
            image: &img.module,
            entry: entry as u32,
            paged,
            parks,
        };
        emit(&offer).then_some((paged, parks))
    });
    let flat = leaf.and_then(|l| Some((super::Region::growable(m.window.mapped(), m.page)?, l)));
    let (back, leaf) = match flat {
        Some((back, l)) => (back, Some(l)),
        None => (m.twin_backing(m.window.reserved()), None),
    };
    // #1768 — a personality exec's staged argv lands only now, at the commit.
    let args = personality.then(|| img.host.exec_commit_args()).flatten();
    let win = m.exec_window(
        back,
        &img.data,
        temen_ir::module_null_guard(),
        args.as_deref(),
    );
    // `exec_image` released the old image's own pipe ends (the fork-inherited ones the exec did not
    // carry). Empty for a command that inherited no CorePipe ends (the rung-1/2a case); non-empty
    // ends need the pipe-EOF wake the cooperative engine does not yet drive — a later rung.
    // Build the command's natural table + activation over its unit.
    let child_host = img.host;
    let cunit = dom.source.get(cm).ok_or(super::EINVAL)?;
    let child_table = build_table_for(cunit.progs.len(), child_host.jit_table_log2(), cm as u32);
    let mut new_vt = VTask::new(&cunit, entry as usize, &child_args).map_err(|_| super::EINVAL)?;
    new_vt.active.module = cm;
    new_vt.active.home = cm;
    let leaf = leaf.map(|(paged, parks)| LeafStart {
        module: cm as u32,
        entry: entry as u32,
        argv: img.entry_args.iter().copied().collect(),
        results: cunit.result_types[entry as usize].clone().into(),
        paged,
        parks,
    });
    Ok(ExecBuilt {
        host: child_host,
        table: child_table,
        vt: new_vt,
        mem: win,
        leaf,
    })
}

fn run(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
) -> Result<Vec<Value>, Trap> {
    // The production path never preempts itself: an unlimited budget makes `resume` run straight to
    // completion, with the per-op budget branch perfectly predicted (so the hot loop is unchanged).
    drive(dom, entry, args, fuel, mem, host, u64::MAX)
}

/// The per-op park/interrupt flags one capability dispatch may raise (#1173). They live on the shared
/// [`Host`], but they describe **one op** — so they must be taken in the same lock scope as the
/// dispatch that set them. On the parallel driver every vCPU of a domain shares one `Host`, and
/// draining them under a later lock let a sibling vCPU's dispatch take another's park: the parked
/// reader kept the placeholder `0` its rewound read had returned, and the sibling completed `-EINTR`
/// into its own `dst`. Grouping them makes "take them all, now, together" the only shape available.
struct DispatchParks {
    /// A `Stream{In}` read that found an empty buffer under `set_stdin_blocking`.
    stdin: bool,
    /// A personality caller-request (`fork` / `execve` / `posix_spawn` / blocking `waitpid`).
    request: Option<super::ParkEvent>,
    /// What a `posix_spawn` request's source staged ([`super::SignalSource::spawn_take`]).
    spawn: Option<super::SpawnPlan>,
    /// A pipe read that found an empty FIFO with writers still open.
    pipe_read: Option<u32>,
    /// A pipe write that found a full FIFO with readers still open.
    pipe_write: Option<u32>,
    /// A signal interrupted a blocking op: the latch a driver's interrupt path set.
    sig_interrupt: bool,
}

impl DispatchParks {
    /// Drain every per-op flag from `p`. Call ONLY inside the dispatch's own lock scope.
    fn take(p: &mut Host) -> Self {
        let request = p.take_park_request();
        let spawn = matches!(request, Some(super::ParkEvent::SpawnSelf { .. }))
            .then(|| p.take_spawn_plan())
            .flatten();
        let parks = Self {
            stdin: p.take_stdin_parked(),
            request,
            spawn,
            pipe_read: p.take_pipe_read_parked(),
            pipe_write: p.take_pipe_write_parked(),
            sig_interrupt: p.take_sig_interrupt(),
        };
        // The wake flags a peer's write/close raised need no action at a dispatch site — both drivers
        // poll pipe readiness at their settle — but they are per-op too, so they drain here as well.
        let _ = (p.take_pipe_wake(), p.take_pipe_wake_writers());
        parks
    }
}

/// Why [`Vm::resume`] returned. `Done`/`Suspended` are the run-to-completion + budget cases; the
/// `Cont*`/`Suspend` cases are §12 fiber switches handled within [`step_vcpu`] (a vCPU's own fiber
/// registry); the `Thread*`/`Memory*` cases are §12 multi-vCPU events handled by the [`drive`]
/// scheduler. A trap is the `Err` arm of `resume`'s `Result` and is terminal, like the tree-walker.
enum Outcome {
    Done(Vec<Value>),
    Suspended,
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): a direct `Call` to an eligible module-0
    /// function. The host runs the emitted `f{func}` region and delivers its `n_results` results to
    /// the absolute register slot `dst`. `argv` is the marshalled arguments (raw i64 slots).
    /// `mapped` is the window's scalar committed extent at call entry ([`Mem::scalar_extent`]) — the
    /// host MUST write it to the emitted module's `"mapped"` global before invoking `f{func}`, so the
    /// emitted bounds check admits exactly what the interpreter would (#717). Surfaced only when the
    /// extent is scalar-representable; otherwise the call is interpreted (fail-closed decline).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
        mapped: u64,
    },
    /// F2 (FIBER_PARK.md) — a punted offloadable dispatch (`Pending(completion_id)`) with an
    /// exactly-`i64` reply, surfaced so the DRIVER decides the wait shape: the cooperative
    /// `drive` parks a punting FIBER (`FiberState::CapParked` — the slice-5a contract) and
    /// blocks inline at root; every other driver keeps the slice-1 inline wait (the I45
    /// posture). The op already advanced `pc`; delivery writes the scalar to `dst`.
    CapPending {
        id: u64,
        dst: u32,
    },
    /// `cont.new`: register a fiber for `(funcref, sp)`, write its handle to `dst`, continue.
    ContNew {
        funcref: i32,
        sp: i64,
        dst: u32,
    },
    /// `cont.resume`: switch into fiber `kh` with `arg`; `(status, value)` land at `dst`/`dst+1`.
    /// `blocking` marks the I48 `cont.resume.block` variant; `resume_ip` is this op's own program
    /// counter, so a blocking park can rewind the resumer's cursor to re-execute the resume on wake.
    ContResume {
        kh: i32,
        arg: i64,
        dst: u32,
        blocking: bool,
        resume_ip: usize,
    },
    /// `suspend`: hand `value` to the resumer; the parked fiber's `dst` receives the next resume arg.
    FiberSuspend {
        value: i64,
        dst: u32,
    },
    /// `thread.spawn`: spawn a vCPU running `func(sp, arg)`; its handle lands at `dst`. `module` is
    /// the **spawning frame's** module — `func` resolves there, so an installed §22 unit's code
    /// spawns the *unit's own* functions (CONSOLIDATION.md §11), and the child's root frame starts
    /// in that module.
    ThreadSpawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
        module: usize,
    },
    /// `thread.join`: park until child `handle` finishes; its result (or trap) lands at `dst`.
    ThreadJoin {
        handle: i32,
        dst: u32,
    },
    /// §3.6 (I36 slice 2) — a caller's `call.cap` through a live-callee offer. The dispatch is
    /// already enqueued on the callee (the op exec holds the callee `Arc`); the driver parks this
    /// task on `ticket` until the callee's serve loop settles the completion cell, then delivers
    /// the reply to `dst`. The cursor is persisted PAST the op (the reply is the call's result).
    LiveCall {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// §3.6 (I36 slice 2) — `svc.wait` with an empty queue and no progress: park this task on its
    /// domain until a caller's enqueue re-admits it. The cursor is persisted AT the op, so the
    /// wake re-executes the whole serve drain (the tree-walker's rewound park).
    SvcWait,
    /// §3.6 (I36 slice 2) — `child_offer`: mint a live offer over child `child`'s export
    /// `export` (driver-side — it owns the child envs); the handle (or `-EINVAL`) lands at `dst`.
    ChildOffer {
        child: i32,
        export: u32,
        dst: u32,
    },
    /// FORK.md §9.2 — `clone_caller`: fork the caller parked on the running handler's dispatch into a
    /// twin. Driver-side (it owns the task/env set + the parked caller). `reply_orig` = `Some` in the
    /// explicit two-reply form, `None` in pid mode (the parent gets the twin's task id). The driver
    /// reads the handler's `serve_ticket` to name the caller; the twin handle (or an errno) lands at
    /// `dst` when `has_result`.
    CloneCaller {
        reply_orig: Option<i64>,
        reply_twin: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap`: reap twin `pid` on behalf of the caller parked on this handler's
    /// dispatch ([`Outcome::Reap`]).
    Reap {
        pid: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §8.6 — `exec_module` (`execve` image-replace, #1080): the running vCPU asks the driver
    /// to replace its image with command module `mh`, granting the by-name cap list at `(grants_ptr,
    /// grants_n)` into the command's fresh powerbox, entering `entry`. `size_log2` is advisory (the
    /// real window is the caller's). On refusal the driver writes `-EINVAL` to `dst`; on success the
    /// task's activation is swapped and it never returns to the caller.
    Exec {
        cmd: super::ExecCmd,
        grants_ptr: u64,
        grants_n: u64,
        entry: u64,
        size_log2: i64,
        dst: u32,
        /// #1768 — a personality `execve` (`ParkEvent::ExecSelf`), whose staged argv the commit
        /// collects; `false` for the guest's own `exec_module` (op 14).
        personality: bool,
    },
    /// #799/#1080 — a personality **`fork()`** caller-request (`ParkEvent::ForkSelf`): the running vCPU
    /// asks the driver to duplicate it (private window copy + forked powerbox) into a twin task. The
    /// parent's `dst` receives the twin's pid; the twin resumes at the same op with `0`. On a failed
    /// fork the parent's `dst` gets `-EAGAIN` (never a hang). Cooperative-driver-only (other drivers
    /// `ThreadFault`) — the port of the tree-walker's `Blocked::ForkSelf` → `fork_vcpu` engine.
    ForkSelf {
        dst: u32,
    },
    /// A personality **`posix_spawn()`** caller-request (`ParkEvent::SpawnSelf`): the driver mints a
    /// new process running `cmd`, with the state the personality staged (`plan`), and writes its pid
    /// to `dst` (`-EAGAIN` when none could be minted). The port of the tree-walker's
    /// `Blocked::SpawnSelf` → `spawn_vcpu`.
    SpawnSelf {
        cmd: super::ExecCmd,
        plan: super::SpawnPlan,
        dst: u32,
    },
    /// #799 — a personality blocking **`waitpid()`** caller-request (`ParkEvent::TaskExit`/`TaskExitAny`):
    /// the running vCPU's op was rewound (it re-executes on wake). The driver parks it until the named
    /// child (`Some(pid)` = a specific twin task, `None` = any child) completes, then re-admits it so the
    /// re-executed `waitpid` finds the twin retired (its exit hooks fired) and serves the exit status.
    ReapWait {
        child: Option<usize>,
    },
    /// #1080 rung 4 — a personality blocking pipe **read** on an empty FIFO with writers still open: the
    /// op was rewound, the driver parks the task on the pipe (`pipe` = the domain-local index) and the
    /// settle scan re-admits it when the FIFO has bytes or every writer closed (EOF). Cooperative-only.
    PipeRead {
        pipe: u32,
    },
    /// #1080 rung 4 (backpressure) — a personality blocking pipe **write** to a full FIFO with readers
    /// still open: rewound + parked, re-admitted when the FIFO has room or every reader closed (`-EPIPE`).
    PipeWrite {
        pipe: u32,
    },
    /// §14 `Instantiator.instantiate` / `instantiate_module[_named]` / a §3d record (ops 0, 5, 13,
    /// 17): the authority `(ibase, isize)` is resolved; the driver builds the **confined executor
    /// child** `spawn` describes with its own attenuated powerbox, registers it (handle = thread
    /// slot), and writes the handle (or `EINVAL`) to `dst`. Unlike a coroutine, the child runs on the
    /// scheduler — joinable via the shared thread machinery (`Instantiator.join` compiles to
    /// [`Outcome::ThreadJoin`]).
    Instantiate {
        spawn: ConfinedSpawn,
        dst: u32,
    },
    /// §5 `Instantiator.instantiate_detached` (op 15, #1286): the driver admits the separate-module
    /// child `spawn` describes, in a fresh window of its own ([`admit_detached_child`]), and writes
    /// the join handle (or `EINVAL`) to `dst`.
    InstantiateDetached {
        spawn: DetachedSpawn,
        dst: u32,
    },
    /// `memory.wait`: futex wait on confined address `base` (already validated); `dst` gets the
    /// status (0 woken / 1 not-equal / 2 timed-out).
    MemoryWait {
        base: u64,
        expected: u64,
        width: u32,
        /// The guest's timeout in ns, or `None` for an infinite wait (#1638) — never a
        /// `MAX_WAIT` stand-in, which is what used to make an infinite wait indistinguishable
        /// from a 10 s one.
        timeout: Option<u64>,
        dst: u32,
    },
    /// `memory.notify`: wake up to `count` waiters on `base`; the woken count lands at `dst`.
    MemoryNotify {
        base: u64,
        count: i32,
        dst: u32,
    },
    /// §22 `install`: the `Jit` cap `h` is authority for code-handle `code`; the driver compiles +
    /// installs the unit and writes the slot (or `-ENOSPC`) to `dst`.
    JitInstall {
        h: i32,
        code: i32,
        dst: u32,
    },
    /// §22 `uninstall`: clear table `slot` (authority `h`); `0`/`EINVAL` → `dst`.
    JitUninstall {
        h: i32,
        slot: i64,
        dst: u32,
    },
    /// §22 `invoke`: run code-handle `code` over the shared window; `argv` are the args as i64 slots,
    /// `params`/`results` type them for the slot ABI; results → `dst…`.
    JitInvoke {
        h: i32,
        code: i32,
        argv: Box<[i64]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// §GC `gc.roots`: operands already resolved + the `mask` validated. The driver does the scan
    /// (it owns the resume chain / fiber registry / coroutines), writes the buffer, and delivers the
    /// total to `dst`.
    GcRoots {
        lo: u64,
        hi: u64,
        mask: u64,
        buf: u64,
        cap: usize,
        dst: u32,
    },
    /// **Blocking stdin park**: a `Stream{In}` `read` found the buffer exhausted under
    /// [`Host::set_stdin_blocking`]. The read did not complete and `pc` was *not* advanced, so the
    /// driver re-issues it after more input arrives. Only the resumable [`Vcpu`] driver honours this
    /// (surfacing [`VcpuEvent::StdinPark`]); the one-shot / scheduler drivers never opt into blocking
    /// stdin, so it never reaches them.
    StdinPark,
}

/// Monotonic clock for a fiber's **real** (busy-poll) `memory.wait` timeout. Returns nanoseconds
/// from an arbitrary process epoch.
///
/// - **Native**: real monotonic wall time (a process-global base [`Instant`]). A timed wait polled
///   in a busy `cont.resume` loop fires after the requested duration elapses, as before.
/// - **Wasm** (`wasm32-unknown-unknown`): there is **no wall clock** — `Instant::now()` panics
///   (`std::time::Instant::now` → `unreachable`). The cdylib is deliberately import-free, so we
///   have no host time either. Instead this returns a monotonic **poll counter** that advances one
///   tick per observation, and [`sched_wall_deadline`] arms the timeout at a small fixed number of
///   ticks ([`WASM_WAIT_POLL_TICKS`]). The busy resume-poll loop therefore still terminates (the
///   sole job of the real deadline; the deterministic *logical* `deadline` remains the idle-time
///   timer). Wall-clock fidelity is meaningless on this target, so counting polls is the honest
///   substitute and keeps the confinement/verifier paths untouched.
#[cfg(not(target_family = "wasm"))]
fn sched_wall_now() -> u64 {
    use std::time::Instant;
    static BASE: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_nanos() as u64
}
#[cfg(target_family = "wasm")]
fn sched_wall_now() -> u64 {
    use core::sync::atomic::{AtomicU64, Ordering};
    static TICKS: AtomicU64 = AtomicU64::new(0);
    TICKS.fetch_add(1, Ordering::Relaxed)
}

/// Poll-ticks a wasm timed `memory.wait` waits before its busy-poll timeout fires (see
/// [`sched_wall_now`]). Small so a `sleep` resolves promptly; non-zero so sibling fibers still get
/// a few turns first rather than the sleeper resolving on its very first poll.
#[cfg(target_family = "wasm")]
const WASM_WAIT_POLL_TICKS: u64 = 8;

/// Arm a real (busy-poll) timeout `timeout` nanoseconds out — native uses the real duration; wasm
/// uses a small fixed tick budget (wall time is meaningless there). See [`sched_wall_now`].
#[cfg(not(target_family = "wasm"))]
fn sched_wall_deadline(timeout: u64) -> u64 {
    sched_wall_now().saturating_add(timeout)
}
#[cfg(target_family = "wasm")]
fn sched_wall_deadline(_timeout: u64) -> u64 {
    sched_wall_now().saturating_add(WASM_WAIT_POLL_TICKS)
}

/// A §12 fiber's state in the driver's per-vCPU registry (handle = index). A durable run maintains the
/// per-context shadow-SP swap ([`shadow_switch`]) and, on freeze, flattens each `Parked` fiber into its
/// shadow region ([`freeze_drive`]); on thaw a flattened fiber is re-seeded as `Pending`. `Clone` for
/// time-travel checkpointing (W1): a fiber-carrying `ScheduledDebugRun` snapshots its whole registry — each
/// fiber `Vm` shares the one window (snapshotted separately), so a clone is a faithful deep copy.
#[derive(Clone)]
enum FiberState {
    /// Created by `cont.new` but never resumed, or unwound by the freeze mid-resume (its spilled
    /// frames in its shadow region, #1835): starts by calling `funcref(sp, arg)`. `consumed`
    /// (#1538) is set only for a **thaw-seeded** fiber whose frozen park was already consumed: its
    /// first resume queues the argument for the rewound `suspend` to return instead of re-parking.
    Pending {
        funcref: i32,
        sp: i64,
        consumed: bool,
    },
    /// Suspended mid-run; resuming delivers the new `arg` into `suspend_dst` and continues `vm`.
    /// `consumed` (#1538): the park happened under `NORMAL`, so its value was taken by a resumer that
    /// ran on — recorded into the freeze residue (see `FrozenFiber::consumed`).
    Parked {
        vm: Vm,
        suspend_dst: u32,
        consumed: bool,
    },
    /// §3.6 slice 5a — **event-parked on a futex wait**: the fiber's `memory.wait` parked the
    /// FIBER, not its vCPU (the tree-walk oracle's fiber-park routing, `fiber_parks.rs`). Not
    /// resumable until an event sets `woken`; a `cont.resume` meanwhile reports `FIBER_PARKED`
    /// to the resumer without switching (the cooperative poll). Woken by `notify`
    /// (`WAIT_WOKEN`), by the park-time value recheck (`WAIT_NOT_EQUAL` — after one transient
    /// `FIBER_PARKED`, matching the oracle's register-then-recheck), or by its timeout — which
    /// fires at driver idle via the **logical** `deadline` (with the whole-vCPU wait timers) or
    /// at a `cont.resume` poll via the **real** `real_deadline`, so a busy resume-poll loop
    /// terminates without depending on driver idle time (the jacl timed-wait shape; see
    /// `temen/tests/fiber_timed_wait.rs`).
    WaitParked {
        vm: Vm,
        /// The wait's status register in `vm`; the waking resume writes the `WAIT_*` result here.
        wait_dst: u32,
        /// The wait's rendezvous key, backing-identity canonical (the same key `TaskState::BlockedWait`
        /// parks on), so a notify from another domain's window on the same `SharedRegion` wakes it.
        key: super::FutexKey,
        /// Logical-clock deadline (`clock + timeout`), fired when no task is runnable;
        /// `None` for an infinite wait, which is never a clock-advance candidate (#1638).
        deadline: Option<u64>,
        /// Real-clock deadline (nanoseconds from [`sched_wall_now`]'s epoch), checked at each
        /// `cont.resume` poll of this fiber. Native: monotonic wall time. Wasm: a monotonic
        /// poll counter (no wall clock on `wasm32-unknown-unknown`), so a busy resume-poll loop
        /// still terminates — see [`sched_wall_now`].
        real_deadline: Option<u64>,
        /// `Some(status)` once the event fired — the fiber is claimable and the next resume
        /// delivers the status; `None` while still blocked.
        woken: Option<i32>,
    },
    /// F2 (FIBER_PARK.md) — **event-parked on a punt completion**: the fiber's blocking host
    /// call punted to the offload pool (`Pending(completion_id)`) and parked the FIBER, not its
    /// vCPU (the oracle's `CapPending` fiber park, `fiber_parks.rs`). Not resumable until the
    /// completion is claimed into `woken`; a `cont.resume` meanwhile reports `FIBER_PARKED`
    /// without switching (the cooperative poll). Claims happen ONLY through the ordered drain
    /// (`drain_cap_parked` — smallest outstanding id first, stop at the first not-yet-arrived),
    /// so a later completion never overtakes an earlier parked fiber: bit-exact with the
    /// oracle's `completion_drain` (the §18 pin). The drain runs at each `cont.resume` poll of
    /// a cap-parked fiber and at driver idle (which blocks on the completion store when only
    /// cap parks remain — a pool completion is pending work, never a deadlock).
    CapParked {
        vm: Vm,
        /// The `call.cap`'s result register in `vm`; the waking resume writes the scalar here.
        dst: u32,
        /// The completion id this fiber waits on (ids are minted monotonically — submission
        /// order — so "smallest outstanding" is the delivery order).
        id: u64,
        /// `Some(result)` once the drain claimed the completion; `None` while still in flight.
        woken: Option<i64>,
    },
    /// #1952 — **event-parked on a host op that must wait**: a pipe read or write, or a blocking
    /// stdin read, parked the FIBER, not its vCPU (the oracle's rewound fiber park). The op was
    /// rewound, so the fiber re-executes it when resumed. A `cont.resume` claims it once the op can
    /// proceed ([`HostWait::ready`]) and reports `FIBER_PARKED` without switching until then.
    HostParked { vm: Vm, on: HostWait },
    /// Currently on the resume chain (active or an ancestor) — not independently resumable.
    /// `blocking_ip` (I48): `Some(ip)` if this fiber's current resume used `cont.resume.block`, so a
    /// park inside it idles the resumer (rewinding the resumer's cursor to `ip`) instead of returning
    /// `FIBER_PARKED`; `None` for a plain `cont.resume`. Set at the claim, read when the fiber parks.
    /// `pending` (#1538): the argument a thaw claim of a seeded, consumed fiber queued for its rewound
    /// `suspend` to return (the fiber runs on instead of re-parking); `None` otherwise.
    Running {
        blocking_ip: Option<usize>,
        pending: Option<i64>,
    },
    /// Returned; resuming again is a `FiberFault`.
    Done,
}

/// The §12 fiber registry's three parallel tables — the slots, each fiber's durable shadow-SP
/// (§12.8) and its freeze re-entry metadata — as one unit, so a shared registry locks them together.
#[derive(Default)]
struct FiberTables {
    fibers: Vec<FiberState>,
    sp: Vec<u64>,
    meta: Vec<(i32, i64)>,
}

/// A **run-shared** §12 fiber registry for vCPUs on separate OS threads or Web Workers (#1761): one
/// handle namespace for the root and all its `thread.spawn` children, so a fiber created on one vCPU
/// can be resumed on another (D57 migration) — the cooperative driver's single registry, and the
/// tree-walker's `FiberRegistry`, made shareable. Attach with [`Vcpu::with_shared_fibers`];
/// [`drive_parallel`] builds one per run. The lock is a leaf, held only for a fiber state transition
/// (`cont.new`, the `cont.resume` claim, `suspend`, a fiber's return) or a `gc.roots` scan, never
/// across execution. The claim is the arbiter: `cont.resume` swaps a parked fiber for a `Running`
/// marker under the lock, so exactly one resumer wins and any other gets `FiberFault`.
#[derive(Default)]
pub struct SharedFibers(std::sync::Mutex<FiberTables>);

impl SharedFibers {
    pub fn new() -> SharedFibers {
        SharedFibers::default()
    }
}

/// How [`step_vcpu`] reaches the run's fiber registry — the fiber twin of [`HostCell`]: exclusively
/// owned tables (the cooperative driver, the freeze unwind, a `Vcpu` with no shared registry), or a
/// [`SharedFibers`] locked per transition (the parallel driver, the browser's Worker driver).
enum FiberCell<'a> {
    Excl {
        fibers: &'a mut Vec<FiberState>,
        sp: &'a mut Vec<u64>,
        meta: &'a mut Vec<(i32, i64)>,
    },
    Shared(&'a SharedFibers),
}

impl FiberCell<'_> {
    /// Run `f` over the three tables: directly (`Excl`) or under the registry lock (`Shared`).
    #[inline]
    fn with<R>(
        &mut self,
        f: impl FnOnce(&mut Vec<FiberState>, &mut Vec<u64>, &mut Vec<(i32, i64)>) -> R,
    ) -> R {
        match self {
            FiberCell::Excl { fibers, sp, meta } => f(fibers, sp, meta),
            FiberCell::Shared(s) => {
                let mut g = s.0.lock_unpoisoned();
                let t = &mut *g;
                f(&mut t.fibers, &mut t.sp, &mut t.meta)
            }
        }
    }

    /// This registry as a `gc.roots` view holds it ([`FiberRegRef`]).
    fn view(&self) -> FiberRegRef<'_> {
        match self {
            FiberCell::Excl { fibers, .. } => FiberRegRef::Owned(fibers),
            FiberCell::Shared(s) => FiberRegRef::Shared(s),
        }
    }

    /// [`shadow_switch`] through the cell. Only a durable run keeps shadow-SP words, so a
    /// non-durable one never takes a shared registry's lock for a switch.
    #[inline]
    fn shadow_switch(&mut self, ctx: &mut RunCtx, vt: &mut VTask, out_ctx: usize, in_ctx: usize) {
        if ctx.durable {
            let (mem, root) = (&mut *ctx.mem, &mut vt.root_shadow_sp);
            self.with(|_, sp, _| shadow_switch(mem, sp, root, true, out_ctx, in_ctx));
        }
    }
}

/// A fiber registry as a `gc.roots` view ([`Beneath`]) holds it: borrowed tables, or a run-shared
/// registry locked only while the scan reads it (#1761) — so a drive paused beneath a nested one
/// never holds the lock across the nested one's execution.
#[derive(Clone, Copy)]
enum FiberRegRef<'a> {
    Owned(&'a [FiberState]),
    Shared(&'a SharedFibers),
}

/// What a `cont.resume` claim decided under the registry lock; [`step_vcpu`] acts on it outside.
// A transient on the stack between the claim and the switch; boxing the `Vm` would add a heap
// allocation to every fiber switch.
#[allow(clippy::large_enum_variant)]
enum Claim {
    /// A pending fiber: start `funcref(sp, arg)`.
    Start { funcref: i32, sp: i64 },
    /// A parked fiber, its resume value already delivered: continue it.
    Continue(Vm),
    /// An event-parked fiber still blocked, under a blocking resume on the cooperative driver: idle.
    Block,
    /// An event-parked fiber still blocked: the resumer gets `(FIBER_PARKED, 0)`.
    Poll,
}

/// #1538 — take the argument a thaw claim queued for fiber `slot`'s rewound `suspend` (see
/// [`FiberState::Running::pending`]); `None` for an ordinary park.
fn take_pending(fibers: &mut [FiberState], slot: usize) -> Option<i64> {
    match fibers.get_mut(slot) {
        Some(FiberState::Running { pending, .. }) => pending.take(),
        _ => None,
    }
}

/// #1538 — whether a durable freeze is in progress (the global freeze word reads `UNWINDING`): a park
/// under it is **fresh** (the resumer's trailing poll unwinds before it observes the value), one under
/// `NORMAL` is consumed by the resumer that runs on.
fn is_unwinding(mem: &Option<Mem>) -> bool {
    mem.as_ref().map(|m| m.durable_state()) == Some(super::STATE_UNWINDING)
}

/// #1952 — what a [`FiberState::HostParked`] fiber waits on.
#[derive(Clone)]
enum HostWait {
    /// A pipe end, read without the `Host` ([`super::PipeProbe`]).
    Pipe(super::PipeProbe),
    /// Its domain's blocking stdin.
    Stdin,
}

impl HostWait {
    /// What a host-park `stop` waits on, over the parker's powerbox: `None` for any other stop, or a
    /// pipe that has vanished (its re-run fails closed).
    fn of(stop: &VcpuStop, host: &Host) -> Option<HostWait> {
        match *stop {
            VcpuStop::PipeRead { pipe } => host.pipe_probe(pipe, false).map(HostWait::Pipe),
            VcpuStop::PipeWrite { pipe } => host.pipe_probe(pipe, true).map(HostWait::Pipe),
            VcpuStop::StdinPark => Some(HostWait::Stdin),
            _ => None,
        }
    }

    /// Whether the parked op can proceed, over the parked fiber's domain powerbox.
    fn ready(&self, host: &Host) -> bool {
        match self {
            HostWait::Pipe(p) => p.ready(),
            HostWait::Stdin => host.stdin_ready(),
        }
    }
}

/// F2 (FIBER_PARK.md) — the ordered completion drain over the fiber registry: claim ready punt
/// completions for [`FiberState::CapParked`] fibers **smallest-id-first, stopping at the first
/// not-yet-arrived** — the bytecode mirror of the oracle scheduler's `completion_drain`
/// (submission-ordered delivery, the §18 pin). Returns whether anything was claimed.
fn drain_cap_parked(fibers: &mut [FiberState], comps: &super::Completions) -> bool {
    let mut claimed = false;
    loop {
        let Some((fi, id)) = fibers
            .iter()
            .enumerate()
            .filter_map(|(i, f)| match f {
                FiberState::CapParked {
                    id, woken: None, ..
                } => Some((i, *id)),
                _ => None,
            })
            .min_by_key(|&(_, id)| id)
        else {
            return claimed;
        };
        let Some(r) = comps.try_take(id) else {
            return claimed;
        };
        if let FiberState::CapParked { woken, .. } = &mut fibers[fi] {
            *woken = Some(r);
        }
        claimed = true;
    }
}

/// The root activation's id in a vCPU's resume chain (it has no fiber handle).
const ROOT_FIBER: usize = usize::MAX;

/// One vCPU's continuation: its active `Vm` and its resume `chain`. A `thread.spawn` creates a fresh
/// `VTask`; the scheduler runs them cooperatively over one shared `Mem` (single-threaded, so shared
/// memory is sequentially consistent — the determinate programs the oracle uses give the same result
/// on any correct schedule). The §12 **fiber registry is run-shared** (one handle namespace per
/// domain, held by [`drive`]), so a fiber created/suspended on one vCPU can be resumed on another
/// (D57 migration) — only the resume `chain` (the ancestor stack) is per-vCPU.
struct VTask {
    active: Vm,
    /// `ROOT_FIBER` or the handle of the fiber currently running in this vCPU.
    active_id: usize,
    /// Parked resumers: `(fiber id, its Vm, the `cont.resume` result slot awaiting (status, value))`.
    chain: Vec<(usize, Vm, u32)>,
    /// DURABILITY.md §12.8 (D-fiber-cont option A): the root computation's (context 0's) saved durable
    /// shadow-stack pointer, swapped with the in-window active word ([`super::SHADOW_SP_OFF`]) on each
    /// fiber switch so a freeze poll spills into the *running* context's region. Only meaningful on a
    /// durable run; `ShadowArena::region_base(0)` (context 0's region base) otherwise.
    root_shadow_sp: u64,
    /// Debug **step-into** of a §22 `Jit.invoke`d unit — `Some` while the debug engine
    /// ([`ScheduledDebugRun`], #1517 slice 3) is stepping inside one, `None` otherwise and always on a
    /// production task (only [`debug_advance_fiber`] arms it). An invoked unit is seam-free (a
    /// `cont.*`/`spawn`/re-invoke inside it `CapFault`s), so no scheduler seam can occur mid-invoke;
    /// a checkpoint mid-invoke is refused on both engines (`checkpointable`), since the unit's
    /// transient `Vm` + `source.push`ed module are not captured.
    active_invoke: Option<Box<InvokeStep>>,
}

/// While a debug engine steps *inside* a §22 `Jit.invoke`d unit, the active continuation
/// is the invoked unit's [`Vm`], not [`VTask::active`]. Unlike a coroutine child (a confined domain with
/// its own `mem`/`host`/`table`), an invoked unit is a **seam-free leaf over the caller's** window /
/// powerbox / dispatch table, so only its `Vm` is held here — the reader resolves its frames against the
/// session `mem`/`source`/`table` (module ≥ 1 for its own funcs, dispatching installed units through the
/// shared table like the production `run_invoke`). On completion the unit's returns marshal into the
/// caller's `dst` slots through the i64-slot ABI; `parent_depth` is the caller's call depth at the
/// invoke, so the stepping predicate sees a *cumulative* depth across the boundary (step-over of the
/// `invoke` runs the unit to completion), exactly like [`VTask::active_coro`].
struct InvokeStep {
    vm: Vm,
    dst: u32,
    results: Box<[ValType]>,
    parent_depth: usize,
}

impl VTask {
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<VTask, Trap> {
        Ok(VTask {
            active: Vm::new(c, entry, args)?,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            root_shadow_sp: c.shadow.unwrap_or(super::ShadowArena::EMPTY).frame_base(0), // §12.8 4A.5: empty root = frame base
            active_invoke: None,
        })
    }

    /// Free a finished task's frames: its register file, call stack and parked fibers. It never
    /// runs again; its result lives in its [`TaskState::Done`].
    fn release(&mut self) {
        let vm = &mut self.active;
        vm.regs = Vec::new();
        vm.stack = Vec::new();
        vm.scratch = Vec::new();
        vm.setjmp_points = std::collections::BTreeMap::new();
        vm.sig_handler_stack = Vec::new();
        self.chain = Vec::new();
        self.active_invoke = None;
    }

    /// The continuation a debug engine is currently stepping: a §22 invoked unit's `Vm`
    /// (which shares the caller's window/table — the reader resolves its frames against the session
    /// `mem`/`source`; its module-≥1 SSA metadata is not plumbed, but its `IrPc`s — hence
    /// breakpoints, stepping, and backtrace — resolve via `source`) or, normally, `active`.
    fn debug_active(&self) -> &Vm {
        match &self.active_invoke {
            Some(iv) => &iv.vm,
            None => &self.active,
        }
    }

    /// The call depth the stepping verbs compare — **cumulative** across a §22 invoke boundary: the
    /// invoked unit's frames sit above the caller's invoke frame (`parent_depth + unit depth`), so a
    /// step-over of the `invoke` runs the unit to completion and a step-out of the unit lands back in
    /// the caller, while a step *within* the unit compares unit-local frames as usual. Equal to the
    /// active `Vm`'s own frame count when no invoke is being stepped.
    fn debug_depth(&self) -> usize {
        match &self.active_invoke {
            Some(iv) => iv.parent_depth + iv.vm.stack.len() + 1,
            None => self.active.stack.len() + 1,
        }
    }
}

/// Re-point the durable active shadow-SP word from the outgoing context's region to the incoming
/// one's, on a fiber switch (DURABILITY.md §12.8, D-fiber-cont option A) — the bytecode-engine mirror
/// of the tree-walker's `shadow_switch`. The running context's live SP is the in-window word the
/// instrumented IR maintains; each *non-running* context's SP lives host-side (the root's in
/// `VTask::root_shadow_sp`, a fiber's in `fiber_sp[slot]`). A no-op unless the run is `durable` with a
/// window. `ctx` is `ROOT_FIBER` for the root or a fiber's registry slot.
fn shadow_switch(
    mem: &mut Option<Mem>,
    fiber_sp: &mut [u64],
    root_shadow_sp: &mut u64,
    durable: bool,
    out_ctx: usize,
    in_ctx: usize,
) {
    if !durable {
        return;
    }
    let Some(m) = mem.as_mut() else { return };
    // §12.8 4A.5: each context's SP word lives in its own region (root = context 0, fiber slot `s` =
    // context `s + 1`). Reached by every durable run on the cooperative scheduler; the drivers that
    // cannot keep it refuse durable hosts (#1694).
    let arena = m.shadow_arena();
    let region_of = |ctx: usize| arena.region_base(if ctx == ROOT_FIBER { 0 } else { ctx + 1 });
    let sp = m.durable_get_sp(region_of(out_ctx));
    if out_ctx == ROOT_FIBER {
        *root_shadow_sp = sp;
    } else {
        fiber_sp[out_ctx] = sp;
    }
    let in_sp = if in_ctx == ROOT_FIBER {
        *root_shadow_sp
    } else {
        fiber_sp[in_ctx]
    };
    m.durable_set_sp(region_of(in_ctx), in_sp);
    // As the tree-walker's `shadow_switch`: carry the active **thaw** phase from the outgoing context
    // to the incoming one (a resumer does not flip its own word; the deepest frame's flip to `NORMAL`
    // propagates back up through the switches), and re-arm an incoming fiber whose restored region
    // still holds a frame (SP above its frame base: seeded frozen residue not yet rewound) to
    // `REWINDING` whatever the carried phase. Without the re-arm, a thawed fiber first claimed by
    // post-rewind `NORMAL` code starts fresh and orphans its spilled frame (#1769: a woken wait then
    // re-parks instead of delivering its wake).
    let ctx_of = |ctx: usize| if ctx == ROOT_FIBER { 0 } else { ctx + 1 };
    let phase = m.durable_thaw_state(ctx_of(out_ctx));
    m.durable_set_thaw_state(ctx_of(in_ctx), phase);
    if in_ctx != ROOT_FIBER && in_sp > arena.frame_base(ctx_of(in_ctx)) {
        m.durable_set_thaw_state(ctx_of(in_ctx), super::STATE_REWINDING);
    }
}

/// **Freeze driver** (DURABILITY.md §12.8 slice 3.1.4) — the bytecode mirror of the tree-walker's
/// `VCpu::freeze_drive`. Called once the root has run to completion under `UNWINDING` (its native
/// stack drained into context 0's shadow region): flatten every still-**parked** fiber into *its own*
/// region so the window snapshot captures it, and return the host-side residue (a [`FrozenFiber`] per
/// flattened fiber) the snapshot records and a thaw re-seeds.
///
/// Each parked fiber is resumed under `UNWINDING` like a standalone root run — a fresh single-frame
/// [`VTask`] whose active `Vm` is the parked continuation with `active_id == ROOT_FIBER` (so its
/// base-frame return ends the sub-run), the active shadow-SP pointed at the fiber's region base, and a
/// placeholder resume value delivered (mimicking `cont.resume`, so the post-suspend continuation is
/// well-formed). The transform places the poll **immediately** after the `suspend`, so the poll fires
/// before any guest code runs: the fiber unwinds with **zero forward progress** and returns. Its
/// flattened shadow-SP extent is saved (into `fiber_sp`, for the snapshot) and recorded in the
/// `FrozenFiber`. The active shadow-SP is left at the **root's** region on return, so the captured
/// window is thaw-ready (the root rewinds first; each fiber's own SP travels in its `FrozenFiber`).
///
/// `generation` is always 0: the bytecode engine is cooperative single-threaded and never recycles a
/// fiber slot, so handles equal slots (matching a non-recycled tree-walker run).
fn freeze_drive(
    fibers: &mut Vec<FiberState>,
    fiber_sp: &mut Vec<u64>,
    fiber_meta: &mut Vec<(i32, i64)>,
    dom: &Domain,
    ctx: &mut RunCtx,
    budget: u64,
) -> Result<Vec<super::FrozenFiber>, Trap> {
    // The root's post-unwind SP (context 0); restored at the end so the window is thaw-ready.
    let arena = ctx
        .mem
        .as_ref()
        .map_or(super::ShadowArena::EMPTY, |m| m.shadow_arena());
    let root_word = arena.region_base(0);
    let root_sp = ctx
        .mem
        .as_ref()
        .map(|m| m.durable_get_sp(root_word))
        .unwrap_or(arena.frame_base(0));
    // The tree-walker's classification, before anything is consumed: an unwoken **cap** park would
    // spill the freeze placeholder as the call's result, which its thaw cannot re-derive, so it
    // fails the whole freeze closed.
    if fibers
        .iter()
        .any(|f| matches!(f, FiberState::CapParked { woken: None, .. }))
    {
        return Err(Trap::FiberFault);
    }
    let mut frozen = Vec::new();
    // Flatten parked fibers in ascending slot order, so the residue's handle namespace is dense from 0
    // (matching the tree-walker's `take_parked_for_freeze`, which always takes the lowest parked slot).
    // Each park's resume value is delivered first, as the tree-walker's flatten does (#1694): a
    // suspend park gets the inert placeholder (the thaw redelivers); a woken event park gets its
    // delivered status or result, which the point's spill reloads at thaw; an unwoken futex park gets
    // an inert status (the point spills without it, and its thaw arm re-issues the wait, which
    // re-checks the restored value). Taking the state also takes the waiter: a notify finds waiters
    // by scanning these states.
    for slot in 0..fibers.len() {
        let (vm, consumed) = match std::mem::replace(&mut fibers[slot], FiberState::Done) {
            FiberState::Parked {
                mut vm,
                suspend_dst,
                consumed,
            } => {
                vm.set(suspend_dst, Reg::from_i64(0));
                (vm, consumed)
            }
            FiberState::WaitParked {
                mut vm,
                wait_dst,
                woken,
                ..
            } => {
                vm.set(
                    wait_dst,
                    Reg::from_i32(woken.unwrap_or(temen_ir::durable_abi::WAIT_FROZEN)),
                );
                (vm, false)
            }
            FiberState::CapParked {
                mut vm,
                dst,
                woken: Some(r),
                ..
            } => {
                vm.set(dst, Reg::from_i64(r));
                (vm, false)
            }
            // A pipe or stdin park's op is rewound (#1952): the unwind re-runs it, and it abandons
            // itself under the landing freeze (`abandon_for_freeze`), for the thaw to re-issue.
            FiberState::HostParked { vm, .. } => (vm, false),
            other => {
                // Not parked: nothing to flatten. A fresh, already-unwound (#1835) or finished slot
                // still rides (#1684), so the thaw rebuilds the table slot for slot.
                match &other {
                    FiberState::Pending { funcref, sp, .. } => frozen.push(super::FrozenFiber {
                        slot,
                        func: *funcref,
                        sp: *sp,
                        shadow_sp: fiber_sp[slot], // its frame base, or the frames the freeze spilled
                        generation: 0,
                        consumed: false,
                    }),
                    FiberState::Done => frozen.push(super::FrozenFiber::free(slot, 0)),
                    _ => {}
                }
                fibers[slot] = other;
                continue;
            }
        };
        let (func, sp) = fiber_meta.get(slot).copied().unwrap_or((0, 0));
        // Point the active shadow-SP at this fiber's region base (an empty shadow stack to unwind into).
        if let Some(m) = ctx.mem.as_mut() {
            m.durable_set_sp(arena.region_base(slot + 1), arena.frame_base(slot + 1));
        }
        // Drive the fiber to its base return under `UNWINDING` (zero forward progress: the poll fires
        // immediately after the park). `step_vcpu` runs the active `Vm` to completion in one call, and
        // the unwind does no fiber/thread ops, so the run-shared registries are untouched and the only
        // stop is `Done`.
        let mut sub = VTask {
            active: vm,
            active_id: ROOT_FIBER,
            chain: Vec::new(),
            root_shadow_sp: root_sp,
            active_invoke: None,
        };
        let mut cell = FiberCell::Excl {
            fibers: &mut *fibers,
            sp: &mut *fiber_sp,
            meta: &mut *fiber_meta,
        };
        match step_vcpu(&mut sub, &mut cell, dom, ctx, budget, false, false)? {
            VcpuStop::Done(_) => {}
            _ => return Err(Trap::FiberFault), // a freeze unwind never spawns / instantiates / blocks
        }
        let shadow_sp = ctx
            .mem
            .as_ref()
            .map(|m| m.durable_get_sp(arena.region_base(slot + 1)))
            .unwrap_or(arena.frame_base(slot + 1));
        fiber_sp[slot] = shadow_sp;
        frozen.push(super::FrozenFiber {
            slot,
            func,
            sp,
            shadow_sp,
            generation: 0,
            consumed,
        });
    }
    // Leave the active shadow-SP at the root's region: the root rewinds first on thaw.
    if let Some(m) = ctx.mem.as_mut() {
        m.durable_set_sp(root_word, root_sp);
    }
    Ok(frozen)
}

/// Scan every live activation of `vm`'s continuation — the active window plus each suspended caller
/// on the call stack — for §GC `gc.roots` candidate words, feeding each 64-bit half (`lo`/`hi`, so a
/// `v128` contributes both) to `consider`. Each activation occupies `regs[base .. base + nslots)` of
/// the function-wide register file (the window model), so this covers exactly that function's live
/// slots — a **sound superset** of the tree-walker's per-block `frame.vals` (it also retains
/// already-dead values from other blocks of the same function, a conservative over-approximation, as
/// the JIT's native-stack scan does — the backends legitimately differ, GC.md §3.2). The register
/// file only ever holds guest words (or default `0`), so `consider`'s mask+range filter keeps any
/// host data out by construction.
fn scan_vm_roots(vm: &Vm, source: &ModuleSource, consider: &mut impl FnMut(u64)) {
    let frames = std::iter::once((vm.module, vm.cur, vm.base))
        .chain(vm.stack.iter().map(|&(m, p, b, _, _)| (m, p, b)));
    for (module, prog, base) in frames {
        let Some(c) = source.get(module) else {
            continue;
        };
        let n = c.progs[prog].nslots as usize;
        let end = (base + n).min(vm.regs.len());
        for r in &vm.regs[base..end] {
            consider(r.lo);
            consider(r.hi);
        }
    }
}

/// Emit a §GC `gc.roots` result: write the first `cap` roots (ascending, already deduplicated by the
/// `BTreeSet`) as little-endian `i64`s into guest memory at `buf` — reusing the confined buffer-write
/// path (a forged/unmapped/RO buffer is a `MemoryFault`) — and return the **total** found.
/// §GC — the candidate root set of one vCPU: its active `Vm` and call stack, every resume-chain
/// ancestor, and every parked fiber in the run's registry, masked and range-filtered to
/// `[lo, hi)`. The **one** definition of "what `gc.roots` scans on this engine", shared by the
/// production drivers (`step_vcpu`) and the debug scheduler (`service_advance`, #1563) — the scope
/// is a property of the op (GC.md §3.1's coverage invariant), not of who is driving, and two copies
/// of it would be two answers to the same question (INVARIANTS #15).
fn gc_scan(
    vt: &VTask,
    fibers: &[FiberState],
    source: &ModuleSource,
    lo: u64,
    hi: u64,
    mask: u64,
) -> std::collections::BTreeSet<u64> {
    gc_scan_beneath(
        &Beneath::task(vt, FiberRegRef::Owned(fibers)),
        source,
        lo,
        hi,
        mask,
    )
}

/// #1660 — the live computations a `gc.roots` must cover, gathered from wherever they are held: the
/// Vms (a task's active one and its parked resumers, a nested drive's own) and the fiber registries,
/// outermost first. A nested drive ([`drive_nested`]) is handed the view of what is paused beneath
/// it; a drive handed none cannot see everything live below it (a bounce out of emitted wasm, whose
/// frames are opaque until they spill — #1627), so a `gc.roots` there fails closed.
#[derive(Default)]
struct Beneath<'a> {
    vms: Vec<&'a Vm>,
    fibers: Vec<FiberRegRef<'a>>,
    /// Raw candidate words: an emitted wasm region's **spill stack** (#1627) — every integer an
    /// emitted frame held live across the host-reaching call that led here, stored before the call.
    words: Vec<&'a [u64]>,
}

impl<'a> Beneath<'a> {
    /// A task continuation `vt` and the registry `fibers` it runs over.
    fn task(vt: &'a VTask, fibers: FiberRegRef<'a>) -> Self {
        let mut vms = vec![&vt.active];
        vms.extend(vt.chain.iter().map(|(_, vm, _)| vm));
        Beneath {
            vms,
            fibers: vec![fibers],
            words: Vec::new(),
        }
    }

    /// This view plus a nested drive's own `active` Vm, resumer `chain` and `fibers` registry — the
    /// view beneath anything that drive enters, and what a `gc.roots` inside it scans.
    fn with_drive<'b>(
        &self,
        active: &'b Vm,
        chain: &'b [(usize, Vm, u32)],
        fibers: FiberRegRef<'b>,
    ) -> Beneath<'b>
    where
        'a: 'b,
    {
        let mut vms: Vec<&'b Vm> = self.vms.clone();
        vms.push(active);
        vms.extend(chain.iter().map(|(_, vm, _)| vm));
        let mut regs: Vec<FiberRegRef<'b>> = self.fibers.clone();
        regs.push(fibers);
        Beneath {
            vms,
            fibers: regs,
            words: self.words.clone(),
        }
    }
}

/// §GC — the candidate root set over everything in `view`: every Vm's frames, and every parked
/// fiber in every registry, masked and range-filtered to `[lo, hi)`. The **one** definition of "what
/// `gc.roots` scans on this engine", shared by the production drivers (`step_vcpu`), the debug
/// scheduler (`service_advance`, #1563) and nested drives (#1660) — the scope is a property of the
/// op (GC.md §3.1's coverage invariant), not of who is driving (INVARIANTS #15).
fn gc_scan_beneath(
    view: &Beneath<'_>,
    source: &ModuleSource,
    lo: u64,
    hi: u64,
    mask: u64,
) -> std::collections::BTreeSet<u64> {
    let mut roots = std::collections::BTreeSet::new();
    {
        let mut consider = |w: u64| {
            let m = w & mask;
            if m >= lo && m < hi {
                roots.insert(m);
            }
        };
        for vm in &view.vms {
            scan_vm_roots(vm, source, &mut consider);
        }
        for &w in view.words.iter().flat_map(|ws| ws.iter()) {
            consider(w);
        }
        // §3.6 slice 5a / F2: an event-parked fiber (`WaitParked` futex, `CapParked` punt
        // completion) holds live frames exactly like a suspended one — scan all three, or a
        // root held across a fiber's blocking point would be missed (unsound for GC.md §3.2).
        let mut scan = |fibers: &[FiberState]| {
            for fib in fibers {
                if let FiberState::Parked { vm, .. }
                | FiberState::WaitParked { vm, .. }
                | FiberState::CapParked { vm, .. }
                | FiberState::HostParked { vm, .. } = fib
                {
                    scan_vm_roots(vm, source, &mut consider);
                }
            }
        };
        for r in &view.fibers {
            match *r {
                FiberRegRef::Owned(f) => scan(f),
                // A run-shared registry is locked only for the scan (#1761).
                FiberRegRef::Shared(s) => scan(&s.0.lock_unpoisoned().fibers),
            }
        }
    }
    roots
}

fn gc_write(
    mem: &mut Option<Mem>,
    buf: u64,
    cap: usize,
    roots: std::collections::BTreeSet<u64>,
) -> Result<i64, Trap> {
    let total = roots.len() as i64;
    let mut bytes = Vec::with_capacity(roots.len().min(cap) * 8);
    for w in roots.into_iter().take(cap) {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    mem.as_mut()
        .ok_or(Trap::Malformed)?
        .write_bytes_impl(buf, &bytes)
        .ok_or(Trap::MemoryFault)?;
    Ok(total)
}

/// Run an invoked §22 unit (`Jit.invoke`) synchronously: a fresh `Vm` for `module`'s entry (func 0)
/// over the shared window/powerbox and the **shared** dispatch table (so the unit's `call.dyn`
/// reaches installed units), to completion. An invoked unit is threads-/seam-free — spawning,
/// event-parking, or re-installing `CapFault`s — but it **may host fibers** (DESIGN.md §22
/// "Concurrency", renegotiated 2026-07-30): `cont.*`/`suspend` are serviced here against an
/// **invoke-confined** registry (a fiber lives and dies within this one invoke — no migration to
/// the run's fibers), with entries resolved through module 0's natural table exactly as
/// [`step_vcpu`]'s arms do (a fiber over an *installed* unit function is the same deferred case).
/// A `suspend` at the invoke root would park the synchronous invoke — `CapFault`, the seam-free
/// half of the contract ("a unit runs its own scheduler to completion"). No durability shadowing:
/// a freeze never lands mid-invoke (snapshot paths carry no invoke state), so unlike `step_vcpu`
/// there is no `fiber_sp`/`shadow_switch` bookkeeping. A trap propagates to the invoker.
/// A resolved unit's `(funcs, types)` — what [`resolve_jit_unit`] hands the driver's Jit arms.
type JitUnitBody = (
    std::sync::Arc<[Func]>,
    std::sync::Arc<[temen_ir::TypeEntry]>,
);

/// Resolve a `Jit.invoke`/`install` `(handle, code)` pair against `host` — authority (a forged handle
/// is a `CapFault`) and the cross-table check (a code handle from another table is one too) — to the
/// unit's funcs + types. The one resolution body the cooperative driver's Jit arms share, so a §14
/// child's units resolve against the **child's** host (#1296: a child holds its own `Jit` table).
/// Also hands back the unit's `(domain, unit)` identity — the install arms mirror it per slot (#1233).
fn resolve_jit_unit(host: &Host, h: i32, code: i32) -> Result<(JitUnitBody, (u32, u32)), Trap> {
    let table = host.resolve_jit_domain(h)?;
    let (cd, cu) = host.resolve_jit_code(code)?;
    if cd != table {
        return Err(Trap::CapFault);
    }
    let funcs = host.jit_unit_funcs(cd, cu).ok_or(Trap::CapFault)?;
    let types = host.jit_unit_types(cd, cu).ok_or(Trap::CapFault)?;
    Ok(((funcs, types), (cd, cu)))
}

#[allow(clippy::too_many_arguments)] // the nested-drive dispatch shim's inputs, as `coop_bounce`'s
fn run_invoke(
    source: &ModuleSource,
    table: &SharedSlots,
    module: usize,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
    beneath: Option<&Beneath<'_>>,
) -> Result<Vec<Value>, Trap> {
    let unit = source.get(module).ok_or(Trap::Malformed)?;
    let mut active = Vm::new(&unit, 0, args)?;
    active.module = module;
    // The interpreted invoke completes synchronously, so its fiber registry is loop-local; the
    // emitted-invoke bounce path threads the vCPU's persistent registry instead (`bounce_call`).
    drive_nested(
        source,
        table,
        active,
        fuel,
        mem,
        host,
        &mut FiberCell::Excl {
            fibers: &mut Vec::new(),
            sp: &mut Vec::new(),
            meta: &mut Vec::new(),
        },
        None,
        beneath,
    )
}

/// The shared nested-drive loop under [`run_invoke`] (an interpreted `Jit.invoke`) and
/// [`Vcpu::bounce_call`] (#846 — one cross-tier callback out of an *emitted* unit): drive `active`
/// to completion over the shared window/powerbox/dispatch-table, servicing fibers against `fibers`.
/// The registry is caller-owned so the bounce path can persist it across the several bounces of one
/// emitted invoke (a fiber parked by one callback is resumable by a later one — exactly the
/// one-registry-per-invoke scope the interpreted loop has by construction).
/// The run-level context a **tier-up region** bounce threads into [`drive_nested`] (`None` for a
/// `Jit.invoke`, whose registry is invoke-confined): its presence marks the drive's registry as the
/// run's (#880 — a `cont.new` then keeps the registry's shadow-SP / freeze-metadata tables
/// index-aligned), and, on the cooperative driver, it lends the B2 slot mirror (#1233) so a
/// `Jit.install`/`uninstall` serviced inside the bounce keeps it exact.
struct BounceRunCtx<'a> {
    /// The coop driver's dispatch-table mirror. `None` on the single-vCPU path (its pump records the
    /// mirror host-side, from surfaced install events — an install serviced inside one of *its*
    /// bounces is not yet mirrored there) and for a §14 child (#1296: a child's installs stay in its
    /// own table, never the root's mirror).
    jit_mirror: Option<JitMirror<'a>>,
    /// #1896 — where a bounce out of a leaf whose host suspends its emitted frames
    /// ([`LeafOffer::parks`]) leaves a call that parks: its continuation, and the state its task
    /// parks in. `None` for any other bounce, in which a park faults.
    park: Option<&'a mut Option<(Vm, TaskState)>>,
}

/// A wasm-JIT driver's dispatch-table mirror — the `slot → (domain, unit)` array and its generation
/// — **lent** to a tier-up-region bounce so a §22 install serviced *inside* the bounce keeps the
/// driver's table exact (#1233 on the cooperative driver, #1339 on the parallel one). Without it the
/// mirror only moves at the pump's own install arm, and an install reached from an emitted frame
/// leaves the driver rebuilding a stale table (the next emitted `call.dyn` traps → decline).
///
/// The cooperative driver lends [`CoopSched`]'s fields; the parallel driver lends its process-global
/// mirror through [`Vcpu::bounce_call`] (its installs are shared across Workers, so the mirror must
/// be too). A §14 child lends nothing: its installs stay in its own table (#1296).
pub struct JitMirror<'a> {
    /// `slot → (domain, unit)`; `None` = empty or natural-prefix.
    pub units: &'a mut Vec<Option<(u32, u32)>>,
    /// Bumped on every install/uninstall, so a driver rebuilds only when the mirror moved.
    pub gen: &'a mut u32,
}

#[allow(clippy::too_many_arguments)] // the nested-drive seam: window + registry halves, all borrowed
fn drive_nested(
    source: &ModuleSource,
    table: &SharedSlots,
    mut active: Vm,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
    // The registry this drive's fibers live in — locked per fiber transition when it is a run-shared
    // one (#1761), never across the drive's execution.
    fibers: &mut FiberCell,
    // #880 — `Some` when `fibers` is the vCPU's **run-level** registry (a bounce out of a TIERUP
    // region: the callback's fibers must persist for the run to resume later, exactly as the same
    // call inline would register them). `ContNew` then mirrors `step_vcpu`'s parallel-array pushes
    // so the run's bookkeeping stays index-aligned; the durability `shadow_switch` is deliberately
    // absent — a bounce host is never a durable run (the pump), and the interpreted invoke path
    // passes `None` (invoke-confined registry).
    mut run_meta: Option<BounceRunCtx<'_>>,
    // #1660 — everything paused beneath this drive, for a `gc.roots` inside it; `None` when that is
    // not fully in view (a bounce out of emitted wasm), which fails the op closed.
    beneath: Option<&Beneath<'_>>,
) -> Result<Vec<Value>, Trap> {
    // The resumer chain (`(resumer's fiber id, resumer, dst)`). Invariant: `chain` is non-empty
    // iff `active` is a fiber (`active_id` then indexes `fibers`).
    let mut chain: Vec<(usize, Vm, u32)> = Vec::new();
    let mut active_id = usize::MAX; // sentinel while the root frame is active
    loop {
        match active.resume(source, table, fuel, mem, host, u64::MAX)? {
            Outcome::Done(vals) => match chain.pop() {
                // The unit entry finished — the invoke's results.
                None => return Ok(vals),
                // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` back.
                Some((rid, resumer, rdst)) => {
                    fibers.with(|f, _, _| f[active_id] = FiberState::Done);
                    let retval = vals.first().copied().unwrap_or(Value::I64(0));
                    // `vcpu.tls` is the vCPU's word: it follows execution (as in `step_vcpu`).
                    let tls = active.tls;
                    active = resumer;
                    active.tls = tls;
                    active_id = rid;
                    active.set(rdst, Reg::from_i32(super::FIBER_RETURNED));
                    active.set(rdst + 1, Reg::from_value(retval));
                }
            },
            Outcome::Suspended => {}
            // F2 — a punted host call inside an invoked unit keeps the pre-F2 inline wait
            // (the unit is a seam-free atomic leaf, DESIGN §22: no park surface exists here,
            // so blocking in place is the contract, not a divergence). Because every cap call
            // completes inline, an invoke fiber is never `CapParked` — no drain arm needed.
            Outcome::CapPending { id, dst } => {
                let r = host.with(|p| p.completions()).wait(id);
                active.set(dst, Reg::from_i64(r));
            }
            Outcome::ContNew { funcref, sp, dst } => {
                // Run-registry mode (#880): keep the parallel arrays index-aligned with the run's
                // (`step_vcpu`'s ContNew arm, minus the durable shadow bookkeeping — see the
                // `run_meta` doc above).
                let run_level = run_meta.is_some();
                let arena = mem
                    .as_ref()
                    .map_or(super::ShadowArena::EMPTY, |m| m.shadow_arena());
                let func_idx = (funcref as u32 as usize & source.primary().table_mask) as i32;
                let h = fibers
                    .with(|f, fsp, meta| {
                        if f.len() + 1 >= super::MAX_FIBERS {
                            return None;
                        }
                        let h = f.len();
                        f.push(FiberState::Pending {
                            funcref,
                            sp,
                            consumed: false,
                        });
                        if run_level {
                            fsp.push(arena.frame_base(h + 1));
                            meta.push((func_idx, sp));
                        }
                        Some(h as i32)
                    })
                    .ok_or(Trap::FiberFault)?;
                active.set(dst, Reg::from_i32(h));
            }
            // Fibers here never event-park (every park surface faults or waits inline above), so
            // the I48 `blocking` flag is a no-op — `cont.resume.block` behaves like `cont.resume`.
            Outcome::ContResume {
                kh,
                arg,
                dst,
                blocking: _,
                resume_ip: _,
            } => {
                let k = kh as usize;
                let running = || FiberState::Running {
                    blocking_ip: None,
                    pending: None,
                };
                // The claim, under a shared registry's lock (#1761); a started fiber's `Vm` is built
                // after it.
                let claim = fibers.with(|f, _, _| match f.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let FiberState::Pending {
                            funcref,
                            sp,
                            consumed,
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        if consumed {
                            *slot = FiberState::Running {
                                blocking_ip: None,
                                pending: Some(arg), // #1538
                            };
                        }
                        Ok(Claim::Start { funcref, sp })
                    }
                    Some(slot @ FiberState::Parked { .. }) => {
                        let FiberState::Parked {
                            mut vm,
                            suspend_dst,
                            consumed: _,
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        vm.set(suspend_dst, Reg::from_i64(arg));
                        Ok(Claim::Continue(vm))
                    }
                    _ => Err(Trap::FiberFault), // forged / Running / Done
                })?;
                let target = match claim {
                    Claim::Continue(vm) => vm,
                    Claim::Start { funcref, sp } => {
                        // Resolve through the shared dispatch table (module-aware, exactly as
                        // `Op::CallIndirect`), so a fiber over an installed §22 unit runs (#1226).
                        let Some((tmod, tfunc, tm)) = resolve_fiber_entry(source, table, funcref)
                        else {
                            return Err(Trap::FiberFault);
                        };
                        let mut fvm = Vm::new(&tm, tfunc, &[Value::I64(sp), Value::I64(arg)])?;
                        fvm.module = tmod;
                        fvm
                    }
                    // Fibers here never event-park (see above), so a claim never polls or blocks.
                    Claim::Block | Claim::Poll => unreachable!(),
                };
                let mut target = target;
                target.tls = active.tls; // `vcpu.tls` follows execution (as in `step_vcpu`)
                let resumer = std::mem::replace(&mut active, target);
                chain.push((active_id, resumer, dst));
                active_id = k;
            }
            Outcome::FiberSuspend { value, dst } => {
                // #1538: a thawed consumed fiber's rewound `suspend` returns the queued argument.
                if let Some(arg) = fibers.with(|f, _, _| take_pending(f, active_id)) {
                    active.set(dst, Reg::from_i64(arg));
                    continue;
                }
                // An empty chain means the unit entry itself tried to `suspend` — that would park
                // the synchronous invoke, the seam the §22 contract forbids.
                let Some((rid, resumer, rdst)) = chain.pop() else {
                    return Err(Trap::CapFault);
                };
                let mut resumer = resumer;
                resumer.tls = active.tls; // `vcpu.tls` follows execution (as in `step_vcpu`)
                let suspended = std::mem::replace(&mut active, resumer);
                let consumed = !is_unwinding(mem);
                fibers.with(|f, _, _| {
                    f[active_id] = FiberState::Parked {
                        vm: suspended,
                        suspend_dst: dst,
                        consumed,
                    }
                });
                active_id = rid;
                active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
                active.set(rdst + 1, Reg::from_i64(value));
            }
            // #1233: a §22 `Jit.install`/`uninstall` reached from a bounced **tier-up region** — the
            // interpreted-inline-call territory (`run_meta` is `Some`) where the pump's own install arm
            // would have serviced the very same call had the region stayed interpreted. Service it here
            // identically (resolve authority + unit → compile → install into the caller's table, the B2
            // mirror kept exact), so a guest whose *emitted* outer loop defines and dispatches units —
            // Forth's `process` — runs on the cooperative tier at all. Inside an invoked unit (`run_meta`
            // `None`) it stays the §22 contract's inert `CapFault`: a unit never re-installs. (A
            // `Jit.invoke` reached the same way is serviced by the arm below, #1334.)
            Outcome::JitInstall { h, code, dst } if run_meta.is_some() => {
                let _ = code; // the mirror keys on the unit identity, not the (revocable) handle
                let ((funcs, types), unit_id) = host.with(|p| resolve_jit_unit(p, h, code))?;
                let res = match compile_module(&funcs, &types, None) {
                    Some(unit) => match jit_install_into(source, table, unit) {
                        Some(slot) => {
                            if let Some(m) = run_meta.as_mut().and_then(|c| c.jit_mirror.as_mut()) {
                                if let Some(e) = m.units.get_mut(slot) {
                                    *e = Some(unit_id);
                                }
                                *m.gen = m.gen.wrapping_add(1); // slot mirror changed → re-sync
                            }
                            slot as i64
                        }
                        None => super::ENOSPC,
                    },
                    None => return Err(Trap::Malformed), // unit op outside coverage
                };
                active.set(dst, Reg::from_i64(res));
            }
            Outcome::JitUninstall { h, slot, dst } if run_meta.is_some() => {
                host.with(|p| p.resolve_jit_domain(h))?; // authority (forged handle → CapFault)
                let n_real = source.primary().progs.len();
                let res = if jit_uninstall_from(source, table, slot as usize, n_real) {
                    if let Some(m) = run_meta.as_mut().and_then(|c| c.jit_mirror.as_mut()) {
                        if let Some(e) = m.units.get_mut(slot as usize) {
                            *e = None; // a freed slot must trap in the driver's table too
                        }
                        *m.gen = m.gen.wrapping_add(1);
                    }
                    0
                } else {
                    super::EINVAL
                };
                active.set(dst, Reg::from_i64(res));
            }
            // #1334: a §22 `Jit.invoke` reached on a nested interpretation — a cross-tier bounce out
            // of an emitted region (the JACL compiler stages a macro from a helper the tiered-up
            // region bounced into), or a unit invoking a unit. Service it interpreted, recursively,
            // over the same window/host/fuel: the task-level interpreted service (`CoopSched::pump`'s
            // fall-through, `Vcpu::deliver_jit_invoke`) verbatim, so the nested unit computes exactly
            // what the oracle does; it gets its own invoke-confined fiber registry (`run_invoke`).
            // It used to fall to the `CapFault` arm below, trapping the bounce and declining the run.
            Outcome::JitInvoke {
                h,
                code,
                argv,
                dst,
                params,
                results,
            } => {
                let ((funcs, types), _) = host.with(|p| resolve_jit_unit(p, h, code))?;
                let unit = compile_module(&funcs, &types, None).ok_or(Trap::Malformed)?;
                let arity_ok = unit
                    .sigs
                    .first()
                    .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
                if !arity_ok {
                    return Err(Trap::CapFault);
                }
                let child_args: Vec<Value> = params
                    .iter()
                    .zip(argv.iter())
                    .map(|(ty, s)| slot_to_val(*ty, *s))
                    .collect();
                let umod = source.push(unit);
                // #1660: this drive is paused beneath the unit — hand it on, if our own view is whole.
                let view = beneath.map(|b| b.with_drive(&active, &chain, fibers.view()));
                let vals = run_invoke(
                    source,
                    table,
                    umod,
                    &child_args,
                    fuel,
                    mem,
                    host,
                    view.as_ref(),
                )?;
                for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                    let re = slot_to_val(*ty, val_to_slot(*v));
                    active.set(dst + i as u32, Reg::from_value(re));
                }
            }
            // §GC `gc.roots` inside a nested drive (#1660): scan what this drive holds — its active
            // Vm, its resumer chain, the registry it runs over — and everything paused beneath it.
            // With no `beneath`, something live below is out of view: fail closed rather than
            // under-report (GC.md §3.2 licenses over-approximation only).
            Outcome::GcRoots {
                lo,
                hi,
                mask,
                buf,
                cap,
                dst,
            } => {
                let roots = {
                    let view =
                        beneath
                            .ok_or(Trap::CapFault)?
                            .with_drive(&active, &chain, fibers.view());
                    gc_scan_beneath(&view, source, lo, hi, mask)
                };
                let total = gc_write(mem, buf, cap, roots)?;
                active.set(dst, Reg::from_i64(total));
            }
            // #1896 — a call parks in a bounce out of a leaf whose host suspends the emitted frames
            // beneath this drive. The drive stops here and hands back its continuation, the op
            // rewound to re-execute, with the state the task parks in; the pump runs the rest once
            // the park clears. A park anywhere else faults, as ever: nothing beneath could wait.
            park @ (Outcome::PipeRead { .. } | Outcome::PipeWrite { .. } | Outcome::StdinPark) => {
                let slot = run_meta
                    .as_mut()
                    .and_then(|c| c.park.as_deref_mut())
                    .filter(|_| chain.is_empty())
                    .ok_or(Trap::CapFault)?;
                let state = match park {
                    Outcome::PipeRead { pipe } => TaskState::BlockedPipeRead { pipe },
                    Outcome::PipeWrite { pipe } => TaskState::BlockedPipeWrite { pipe },
                    _ => TaskState::BlockedStdin,
                };
                *slot = Some((active, state));
                return Ok(Vec::new());
            }
            // #1578 — DESIGN §22: an **invoked** unit (`run_meta` `None`) is a seam-free leaf, so the
            // whole `Instantiator` is unavailable inside it — named here rather than left to the
            // catch-all, because it is the contract, not an unserviced seam. The tree-walker refuses
            // the same way (its `below` guard) and the native tier before it trampolines
            // (`temen_run::invoke_refuses`). An *installed* unit spawns like the base module (#1726).
            // (A tier-up bounce — `run_meta` `Some` — keeps the catch-all's refusal below.)
            Outcome::Instantiate { .. }
            | Outcome::InstantiateDetached { .. }
            | Outcome::ChildOffer { .. }
                if run_meta.is_none() =>
            {
                return Err(Trap::CapFault)
            }
            _ => return Err(Trap::CapFault),
        }
    }
}

/// Why [`step_vcpu`] returned control to the scheduler: the vCPU finished, or it hit a multi-vCPU
/// (`thread.*` / `memory.*`) event the scheduler must service. Fiber switches never reach here —
/// `step_vcpu` handles them against the run's fiber registry ([`FiberCell`]).
enum VcpuStop {
    /// I48 — a `cont.resume.block` whose target fiber is still event-parked: idle this task on the
    /// fiber (`TaskState::BlockedOnFiber`). `step_vcpu` already rewound the resumer's cursor to the
    /// resume op, so the wake re-executes it.
    BlockOnFiber {
        fiber: usize,
    },
    /// §3.6 (I36 slice 2): park this task on a live-call `ticket` against `callee` (see
    /// [`Outcome::LiveCall`] — the enqueue already happened in the op exec).
    LiveCall {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// F2 (FIBER_PARK.md): a punted offloadable dispatch with an exactly-`i64` reply
    /// ([`Outcome::CapPending`]) — the cooperative driver fiber-parks a punting fiber; every
    /// other driver waits inline on the completion (the I45 posture).
    CapPending {
        id: u64,
        dst: u32,
    },
    /// §3.6 (I36 slice 2): park this task in `svc.wait` on its own domain ([`Outcome::SvcWait`]).
    SvcWait,
    /// §3.6 (I36 slice 2): mint a live offer over child `child`'s export ([`Outcome::ChildOffer`]).
    ChildOffer {
        child: i32,
        export: u32,
        dst: u32,
    },
    /// FORK.md §9.2 — `clone_caller`: fork the caller parked on this handler's dispatch into a twin
    /// ([`Outcome::CloneCaller`]). The driver reads the running handler's `serve_ticket` to name it.
    CloneCaller {
        reply_orig: Option<i64>,
        reply_twin: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §9.2 — `reap`: reap twin `pid` on behalf of the caller parked on this handler's
    /// dispatch ([`Outcome::Reap`]).
    Reap {
        pid: i64,
        dst: u32,
        has_result: bool,
    },
    /// FORK.md §8.6 — `exec_module` image-replace (#1080): surfaced to the cooperative driver, which
    /// owns the task/env set + `dom.source`. Cooperative-driver-only (like [`VcpuStop::CloneCaller`]);
    /// other drivers `ThreadFault`.
    Exec {
        cmd: super::ExecCmd,
        grants_ptr: u64,
        grants_n: u64,
        entry: u64,
        size_log2: i64,
        dst: u32,
        /// #1768 — a personality `execve` (`ParkEvent::ExecSelf`), whose staged argv the commit
        /// collects; `false` for the guest's own `exec_module` (op 14).
        personality: bool,
    },
    /// #799/#1080 — personality `fork()` caller-request surfaced to the cooperative driver (which owns
    /// the task/env set). Cooperative-driver-only ([`Outcome::ForkSelf`]).
    ForkSelf {
        dst: u32,
    },
    /// A personality `posix_spawn()` caller-request ([`Outcome::SpawnSelf`]).
    SpawnSelf {
        cmd: super::ExecCmd,
        plan: super::SpawnPlan,
        dst: u32,
    },
    /// #799 — personality blocking `waitpid()` caller-request surfaced to the cooperative driver
    /// ([`Outcome::ReapWait`]). `None` = any child.
    ReapWait {
        child: Option<usize>,
    },
    /// #1080 rung 4 — a personality blocking pipe read/write surfaced to the cooperative driver
    /// ([`Outcome::PipeRead`]/[`Outcome::PipeWrite`]). `pipe` is the parker's domain-local pipe index.
    PipeRead {
        pipe: u32,
    },
    PipeWrite {
        pipe: u32,
    },
    /// #1157 — the cooperative preemption quantum expired at an op boundary. `resume` already persisted
    /// the Vm cursor (its `Outcome::Suspended` path), so the pump re-admits the still-`Runnable` task and
    /// round-robins the pick to a sibling. Produced only when `step_vcpu` is called with `preemptible`
    /// true (the cooperative pump, and only while ≥2 tasks are runnable) — every other caller re-loops
    /// on quantum exhaust, so its behavior is unchanged.
    Preempted,
    Done(Vec<Value>),
    /// **wasm-JIT tier-up** (browser wasm-JIT threads slice): run the emitted `f{func}` region on the
    /// host, delivering its `n_results` results to absolute slot `dst` via `deliver_tierup`.
    /// `mapped` is the entry-snapshot scalar committed extent the host must write to the emitted
    /// `"mapped"` global first (#717 — see [`Outcome::TierUp`]).
    TierUp {
        func: u32,
        argv: Box<[i64]>,
        dst: usize,
        results: Box<[ValType]>,
        mapped: u64,
    },
    Spawn {
        func: u32,
        sp: i64,
        arg: i64,
        dst: u32,
        /// The spawning frame's module — `func` resolves there (CONSOLIDATION.md §11).
        module: u32,
    },
    Join {
        handle: i32,
        dst: u32,
    },
    /// §14 confined child (ops 0, 5, 13, 17): the driver (which owns the task set / extra
    /// environments) admits it with [`admit_confined_child`] and registers it as a joinable thread.
    Instantiate {
        spawn: ConfinedSpawn,
        dst: u32,
    },
    /// §5 `Instantiator.instantiate_detached` (op 15, #1286) — see [`Outcome::InstantiateDetached`].
    InstantiateDetached {
        spawn: DetachedSpawn,
        dst: u32,
    },
    Wait {
        base: u64,
        expected: u64,
        width: u32,
        /// The guest's timeout in ns, or `None` for an infinite wait (#1638).
        timeout: Option<u64>,
        dst: u32,
    },
    Notify {
        base: u64,
        count: i32,
        dst: u32,
    },
    /// §22 `Jit.install` — the driver (which owns the mutable `Domain`) compiles + installs the unit.
    JitInstall {
        h: i32,
        code: i32,
        dst: u32,
    },
    /// §22 `Jit.uninstall` — the driver clears the table slot.
    JitUninstall {
        h: i32,
        slot: i64,
        dst: u32,
    },
    /// §22 `Jit.invoke` — the driver runs the unit synchronously over the shared window.
    JitInvoke {
        h: i32,
        code: i32,
        argv: Box<[i64]>,
        dst: u32,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
    },
    /// Blocking-stdin park (see [`Outcome::StdinPark`]) — the `Vcpu` driver surfaces it as
    /// [`VcpuEvent::StdinPark`]; no residue, since the read re-issues on resume.
    StdinPark,
}

/// How the eval loop reaches the powerbox (THREADS.md 4c-host). The cooperative `drive` owns the host
/// exclusively (`&mut Host`); the **parallel** driver shares one `Arc<Mutex<Host>>` across vCPU threads
/// and takes the lock only for the duration of a single `call.cap` — so compute/atomics/futex between
/// calls stay lock-free (genuine parallelism), exactly the tree-walker's model. Determinism is *not*
/// lost: cooperative is uncontended and dispatches in the same fixed order as before (the oracle);
/// parallel is the opt-in mode whose stateful-cap interleaving races, as real threads do.
enum HostCell<'a> {
    /// Single-owner exclusive access — the cooperative `drive`, the debugger, coroutines, §14 children.
    Excl(&'a mut Host),
    /// Shared behind a lock — the parallel driver's vCPUs; `with` takes the lock per host call.
    Shared(&'a std::sync::Mutex<Host>),
}

impl HostCell<'_> {
    /// Run `f` with exclusive access to the powerbox: directly (`Excl`) or under a brief lock
    /// (`Shared`). `f`'s result is owned (no borrow escapes the lock), so the lock is held only across
    /// the one host call.
    #[inline]
    fn with<R>(&mut self, f: impl FnOnce(&mut Host) -> R) -> R {
        match self {
            HostCell::Excl(h) => f(h),
            HostCell::Shared(m) => f(&mut m.lock_unpoisoned()),
        }
    }
}

/// The per-vCPU execution environment a [`step_vcpu`] runs against: the dispatch `table` it uses
/// (the shared domain table, or a §14 confined child's own natural table), its `fuel` budget, its
/// linear `mem`, and its capability `host`. The root vCPU and its `thread.spawn` siblings share the
/// domain's (env `None`); a §14 `instantiate` child carries its own confined [`ChildEnv`]. Bundled so
/// [`step_vcpu`] takes one ref instead of four (and so the per-task selection has a single type).
struct RunCtx<'a> {
    table: &'a SharedSlots,
    fuel: &'a mut u64,
    mem: &'a mut Option<Mem>,
    host: HostCell<'a>,
    /// DURABILITY.md §12.8: the domain is durable, so each fiber switch maintains the per-context
    /// shadow-SP word ([`shadow_switch`]). Read once from `Host::is_durable` by [`drive`].
    durable: bool,
}

/// #1157 — the cooperative preemption quantum: the op count a runnable task may run before the pump
/// round-robins to a sibling. Armed by [`CoopSched::pump`] **only while ≥2 tasks are runnable**, so a
/// single-runnable run (bash and every non-threaded browser guest) is untouched. Coarse on purpose
/// (~1M ops): fine enough to bound a yield-free spin to sub-second, coarse enough that the interleaving
/// stays close to the old run-to-completion order (fewer differential surprises). Op-count, so the
/// schedule stays deterministic.
const COOP_QUANTUM: u64 = 1 << 20;

/// Run one vCPU (its active `Vm` and any fibers it switches among) until it finishes or hits a
/// multi-vCPU event. Fiber `Outcome`s are serviced here exactly as `run_inner`'s `cont.*` arms switch
/// the active frame stack; `thread.*`/`memory.*` `Outcome`s are handed up to [`drive`]. `budget` only
/// slices *where* the active `Vm` pauses (Slice 1c-2); it never changes results.
#[allow(clippy::too_many_arguments)] // scheduler seam: the vCPU state, registry, domain + the I48 cooperative flag
fn step_vcpu(
    vt: &mut VTask,
    fibers: &mut FiberCell,
    dom: &Domain,
    ctx: &mut RunCtx,
    budget: u64,
    // I48: only the cooperative `drive` scheduler can idle a blocking `cont.resume.block` (park the
    // resumer's task via `VcpuStop::BlockOnFiber`). The OS-thread parallel paths pass `false` and
    // take the advisory `FIBER_PARKED` poll instead — their idle is the follow-up slice (the same
    // OS-thread-block problem as the Cranelift JIT).
    cooperative: bool,
    // #1157: when true, a `budget`-exhausted resume (the cooperative preemption quantum) yields to the
    // pump as `VcpuStop::Preempted` instead of re-looping. Every caller but the cooperative pump passes
    // `false` — the parallel driver and the single-step/debug harness keep the transparent re-loop.
    preemptible: bool,
) -> Result<VcpuStop, Trap> {
    loop {
        match vt.active.resume(
            &dom.source,
            ctx.table,
            &mut *ctx.fuel,
            &mut *ctx.mem,
            &mut ctx.host,
            budget,
        )? {
            // Budget exhausted. Either the transparent sliced-harness re-loop (cursor already
            // persisted), or — with `preemptible` — the #1157 quantum: yield to the pump to round-robin.
            Outcome::Suspended => {
                if preemptible {
                    return Ok(VcpuStop::Preempted);
                }
                // #1198 — the COOPERATIVE pump only: `resume` bailed at a syscall boundary because this
                // domain just STOPPED itself (a background-terminal SIGTTIN/SIGTTOU, or ^Z). Return to the
                // pump so its round-robin pick benches the stopped domain, instead of re-looping straight
                // into another stopped spin. The single-vCPU parallel/debug driver (`cooperative: false`)
                // must NOT bench here: it has no pick to bench into, and its stop is a real concurrent
                // busy-wait — the stopped vCPU spins on its own OS thread until ANOTHER thread SIGCONTs it,
                // exactly as before. `Preempted` on that path is `ThreadFault` (fail-closed), so gate it.
                // Otherwise a normal `Suspended` (the sliced-harness budget boundary) re-loops as ever.
                if cooperative
                    && ctx
                        .host
                        .with(|h| h.signal_poll())
                        .is_some_and(|(_, s)| s.stopped())
                {
                    return Ok(VcpuStop::Preempted);
                }
            }
            Outcome::Done(vals) => match vt.chain.pop() {
                // The vCPU's root activation finished.
                None => return Ok(VcpuStop::Done(vals)),
                // A fiber's function returned: mark it Done, hand `(RETURNED, retval)` to its resumer —
                // unless the freeze unwound it (#1835). An instrumented fiber always unwinds at a poll
                // before a genuine return, so a return under `UNWINDING` that spilled frames (its
                // shadow-SP past its frame base) is the freeze's: the fiber is residue, back to the
                // `Pending` state a thaw seeds it in, and the resumer gets `FIBER_FROZEN`, which its
                // thaw re-issues. Nothing was consumed: with no mid-run trigger, every park in a freeze run
                // was made under `UNWINDING`.
                Some((rid, resumer, rdst)) => {
                    let id = vt.active_id;
                    // Fiber switch (returning fiber → its resumer): re-point the durable shadow-SP.
                    fibers.shadow_switch(ctx, vt, id, rid);
                    let base = ctx
                        .mem
                        .as_ref()
                        .map_or(0, |m| m.shadow_arena().frame_base(id + 1));
                    let frozen = ctx.durable && is_unwinding(ctx.mem);
                    let frozen = fibers.with(|f, sp, meta| {
                        let frozen = frozen && sp[id] > base;
                        f[id] = if frozen {
                            let (funcref, sp) = meta[id];
                            FiberState::Pending {
                                funcref,
                                sp,
                                consumed: false,
                            }
                        } else {
                            FiberState::Done
                        };
                        frozen
                    });
                    let retval = vals.first().copied().unwrap_or(Value::I64(0));
                    // `vcpu.tls` is the vCPU's word, not the fiber's: it goes back with execution.
                    let tls = vt.active.tls;
                    vt.active = resumer;
                    vt.active.tls = tls;
                    vt.active_id = rid;
                    let status = if frozen {
                        super::FIBER_FROZEN
                    } else {
                        super::FIBER_RETURNED
                    };
                    vt.active.set(rdst, Reg::from_i32(status));
                    vt.active.set(rdst + 1, Reg::from_value(retval));
                }
            },
            Outcome::ContNew { funcref, sp, dst } => {
                // A fresh fiber (registry slot `h`) is shadow context `h + 1`; its saved shadow-SP
                // starts at its region base (empty shadow stack) — so a later switch into it points
                // the active word there (DURABILITY.md §12.8; 4A.5: empty = frame base, past the
                // in-region SP + thaw words).
                let arena = ctx
                    .mem
                    .as_ref()
                    .map_or(super::ShadowArena::EMPTY, |m| m.shadow_arena());
                // Freeze residue (DURABILITY.md §12.8): record the fiber's re-entry metadata — its
                // **resolved** entry function index (the natural-table lookup `cont.resume` does, so
                // a `FrozenFiber.func` matches the tree-walker's `Frame::func`) and data-stack base —
                // so the freeze driver can emit a `FrozenFiber` for it even after it parks.
                let func_idx = (funcref as u32 as usize & dom.source.primary().table_mask) as i32;
                let h = fibers
                    .with(|f, fsp, meta| {
                        if f.len() + 1 >= super::MAX_FIBERS {
                            return None;
                        }
                        let h = f.len();
                        f.push(FiberState::Pending {
                            funcref,
                            sp,
                            consumed: false,
                        });
                        fsp.push(arena.frame_base(h + 1));
                        meta.push((func_idx, sp));
                        Some(h as i32)
                    })
                    .ok_or(Trap::FiberFault)?;
                vt.active.set(dst, Reg::from_i32(h));
            }
            Outcome::ContResume {
                kh,
                arg,
                dst,
                blocking,
                resume_ip,
            } => {
                let k = kh as usize;
                // I48: if this resume is `cont.resume.block`, tag the switched-in fiber so a park
                // inside it idles this task (rewinding to `resume_ip`) instead of the FIBER_PARKED
                // poll. `None` for a plain `cont.resume`.
                let blocking_ip = blocking.then_some(resume_ip);
                // #1952 — a poll of a host-parked fiber asks whether its op can proceed now, over
                // this task's powerbox (read here: the claim below runs under the registry).
                let host_ready = fibers
                    .with(|f, _, _| match f.get(k) {
                        Some(FiberState::HostParked { on, .. }) => Some(on.clone()),
                        _ => None,
                    })
                    .is_some_and(|on| ctx.host.with(|h| on.ready(h)));
                // F2 (FIBER_PARK.md) — a poll of a cap-parked fiber runs the ordered drain
                // first (so a busy resume-poll loop observes its completion without waiting
                // for driver idle — the WaitParked `real_deadline` shape, completion form).
                // The drain, not a direct `try_take`, so a poll of a LATER id never lets its
                // ready result overtake an earlier outstanding park (the §18 pin).
                if fibers.with(|f, _, _| {
                    matches!(f.get(k), Some(FiberState::CapParked { woken: None, .. }))
                }) {
                    let comps = ctx.host.with(|p| p.completions());
                    fibers.with(|f, _, _| drain_cap_parked(f, &comps));
                }
                let running = || FiberState::Running {
                    blocking_ip,
                    pending: None,
                };
                // Claim fiber `k` from the **run-shared** registry: a pending fiber starts (call
                // `funcref(sp, arg)`), a parked one continues (the new `arg` becomes its `suspend`'s
                // result) — possibly one suspended on *another* vCPU (D57 migration). Anything else
                // (forged / already running on a vCPU / done) is inert. Only the state transition
                // happens under a shared registry's lock (#1761); a started fiber's `Vm` is built
                // after it.
                let claim = fibers.with(|f, _, _| match f.get_mut(k) {
                    Some(slot @ FiberState::Pending { .. }) => {
                        let FiberState::Pending {
                            funcref,
                            sp,
                            consumed,
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        if consumed {
                            *slot = FiberState::Running {
                                blocking_ip,
                                pending: Some(arg), // #1538
                            };
                        }
                        Ok(Claim::Start { funcref, sp })
                    }
                    Some(slot @ FiberState::Parked { .. }) => {
                        let FiberState::Parked {
                            mut vm,
                            suspend_dst,
                            consumed: _,
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        vm.set(suspend_dst, Reg::from_i64(arg));
                        Ok(Claim::Continue(vm))
                    }
                    // §3.6 slice 5a: an event-parked fiber (blocked in `memory.wait`). Woken —
                    // or with its real deadline passed (the timeout fires at the poll, so a
                    // cooperative resume-poll loop terminates) — the resume delivers the wait's
                    // status into the fiber and continues it (the resume `arg` is deliberately
                    // NOT delivered, matching the oracle's `LiveWoken`); still blocked, the
                    // resumer gets `(FIBER_PARKED, 0)` without a switch (the cooperative poll).
                    Some(slot @ FiberState::WaitParked { .. }) => {
                        let FiberState::WaitParked {
                            woken,
                            real_deadline,
                            ..
                        } = slot
                        else {
                            unreachable!()
                        };
                        let fired = woken.take().or_else(|| {
                            // #1638: an infinite wait has no real deadline to poll against, so
                            // a busy resume-poll loop over one never fabricates a timeout here —
                            // it runs until the poller itself `notify`s the fiber, or until fuel
                            // runs out. That is the answer, not a gap (#1642): the poller is live
                            // guest code, so whether it will ever notify is undecidable and no
                            // engine may call the loop a deadlock. Fuel bounds it like any other
                            // guest loop, at the identical safepoint on all three engines
                            // (`jit_fuel.rs`). Before #1638 this engine alone ended it, with a
                            // `WAIT_TIMED_OUT` at 10 s that the guest never asked for.
                            real_deadline
                                .filter(|dl| sched_wall_now() >= *dl)
                                .map(|_| super::WAIT_TIMED_OUT)
                        });
                        // I48: a blocking resume of a still-parked fiber idles this task on the
                        // fiber (its deadline is already in the idle scan; notify wakes it too).
                        // A plain resume returns the FIBER_PARKED poll (guest loops).
                        let Some(st) = fired else {
                            return Ok(if blocking && cooperative {
                                Claim::Block
                            } else {
                                Claim::Poll
                            });
                        };
                        let FiberState::WaitParked {
                            mut vm, wait_dst, ..
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        vm.set(wait_dst, Reg::from_i32(st));
                        Ok(Claim::Continue(vm))
                    }
                    // F2 — a cap-parked fiber (blocked on its punt completion). Claimed by
                    // the drain above (`woken`), the resume delivers the scalar into the
                    // `call.cap`'s result register and continues it (the resume `arg` is
                    // deliberately NOT delivered — the oracle's `LiveWoken`); still in
                    // flight, the resumer gets `(FIBER_PARKED, 0)` without a switch (I48: a
                    // blocking resume idles until the ordered completion drain wakes it).
                    // #1952 — a host-parked fiber: once its op can proceed, the resume continues it,
                    // and its rewound op re-executes (nothing is delivered); until then the resumer
                    // gets `(FIBER_PARKED, 0)` without a switch, or idles under a blocking resume.
                    Some(slot @ FiberState::HostParked { .. }) => {
                        if !host_ready {
                            return Ok(if blocking && cooperative {
                                Claim::Block
                            } else {
                                Claim::Poll
                            });
                        }
                        let FiberState::HostParked { vm, .. } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        Ok(Claim::Continue(vm))
                    }
                    Some(slot @ FiberState::CapParked { .. }) => {
                        let FiberState::CapParked { woken, .. } = slot else {
                            unreachable!()
                        };
                        let Some(r) = woken.take() else {
                            return Ok(if blocking && cooperative {
                                Claim::Block
                            } else {
                                Claim::Poll
                            });
                        };
                        let FiberState::CapParked {
                            mut vm,
                            dst: cap_dst,
                            ..
                        } = std::mem::replace(slot, running())
                        else {
                            unreachable!()
                        };
                        vm.set(cap_dst, Reg::from_i64(r));
                        Ok(Claim::Continue(vm))
                    }
                    _ => Err(Trap::FiberFault), // forged / Running / Done
                })?;
                let target = match claim {
                    Claim::Continue(vm) => vm,
                    // The oracle's rule: a durable run whose freeze is unwinding takes the
                    // advisory downgrade (the poll status), so the resumer reaches its trailing
                    // poll instead of re-idling (#1904).
                    Claim::Block if !(ctx.durable && is_unwinding(ctx.mem)) => {
                        // Rewind the resumer's cursor so the wake re-executes the resume.
                        vt.active.pc = resume_ip;
                        return Ok(VcpuStop::BlockOnFiber { fiber: k });
                    }
                    Claim::Block | Claim::Poll => {
                        vt.active.set(dst, Reg::from_i32(super::FIBER_PARKED));
                        vt.active.set(dst + 1, Reg::from_i64(0));
                        continue;
                    }
                    Claim::Start { funcref, sp } => {
                        // Resolve the fiber entry through the shared dispatch table (module-aware,
                        // exactly as `Op::CallIndirect` / the tree-walker's `dispatch_indirect` / the
                        // JIT's shared `fn_table`): a fiber may start on an **installed §22 unit**
                        // function (a module ≥ 1 entry), not only a module-0-natural one — a
                        // forged/mistyped funcref is still a `FiberFault` (#1226, DESIGN.md §22
                        // "Concurrency", renegotiated 2026-07-30). Resolve against `ctx.table`, the
                        // same table the fiber's own `call.dyn`s dispatch through (paired with
                        // `dom.source` in the `resume` above).
                        let Some((tmod, tfunc, tm)) =
                            resolve_fiber_entry(&dom.source, ctx.table, funcref)
                        else {
                            return Err(Trap::FiberFault);
                        };
                        let mut fvm = Vm::new(&tm, tfunc, &[Value::I64(sp), Value::I64(arg)])?;
                        fvm.module = tmod;
                        // §12.8 4A.5: this fiber spills into its own region (slot `k` = context `k + 1`).
                        fvm.durable_region_base = ctx
                            .mem
                            .as_ref()
                            .map_or(super::ShadowArena::EMPTY, |m| m.shadow_arena())
                            .region_base(k + 1);
                        fvm
                    }
                };
                // Fiber switch (resumer → fiber `k`): re-point the durable shadow-SP before the swap.
                let out = vt.active_id;
                fibers.shadow_switch(ctx, vt, out, k);
                // `vcpu.tls` is read at the executing vCPU (the op's spec): the fiber runs with this
                // vCPU's word, wherever it last ran.
                let mut target = target;
                target.tls = vt.active.tls;
                let resumer = std::mem::replace(&mut vt.active, target);
                vt.chain.push((vt.active_id, resumer, dst));
                vt.active_id = k;
            }
            Outcome::FiberSuspend { value, dst } => {
                let id = vt.active_id;
                // #1538: a thawed consumed fiber's rewound `suspend` returns the queued argument.
                if let Some(arg) = fibers.with(|f, _, _| take_pending(f, id)) {
                    vt.active.set(dst, Reg::from_i64(arg));
                    continue;
                }
                // Pop the resumer to switch back to; an empty chain means the root tried to
                // `suspend`, which is a `FiberFault` (the root has no resumer).
                let (rid, resumer, rdst) = vt.chain.pop().ok_or(Trap::FiberFault)?;
                // Fiber switch (suspending fiber → its resumer): re-point the durable shadow-SP.
                fibers.shadow_switch(ctx, vt, id, rid);
                // `vcpu.tls` goes back to the resumer with execution (the fiber may have set it).
                let mut resumer = resumer;
                resumer.tls = vt.active.tls;
                let suspended = std::mem::replace(&mut vt.active, resumer);
                let consumed = !is_unwinding(ctx.mem);
                fibers.with(|f, _, _| {
                    f[id] = FiberState::Parked {
                        vm: suspended,
                        suspend_dst: dst,
                        consumed,
                    }
                });
                vt.active_id = rid;
                vt.active.set(rdst, Reg::from_i32(super::FIBER_SUSPENDED));
                vt.active.set(rdst + 1, Reg::from_i64(value));
            }
            Outcome::TierUp {
                func,
                argv,
                dst,
                results,
                mapped,
            } => {
                return Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                })
            }
            Outcome::ThreadSpawn {
                func,
                sp,
                arg,
                dst,
                module,
            } => {
                return Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module: module as u32,
                })
            }
            Outcome::ThreadJoin { handle, dst } => return Ok(VcpuStop::Join { handle, dst }),
            // §3.6 (I36 slice 2): the serve/call/offer trio surface straight to the driver. The
            // qualification veto keeps them out of fiber contexts, so no registry state is live.
            Outcome::LiveCall {
                ticket,
                callee,
                dst,
            } => {
                return Ok(VcpuStop::LiveCall {
                    ticket,
                    callee,
                    dst,
                })
            }
            Outcome::SvcWait => return Ok(VcpuStop::SvcWait),
            Outcome::ChildOffer { child, export, dst } => {
                return Ok(VcpuStop::ChildOffer { child, export, dst })
            }
            Outcome::CloneCaller {
                reply_orig,
                reply_twin,
                dst,
                has_result,
            } => {
                return Ok(VcpuStop::CloneCaller {
                    reply_orig,
                    reply_twin,
                    dst,
                    has_result,
                })
            }
            Outcome::Reap {
                pid,
                dst,
                has_result,
            } => {
                return Ok(VcpuStop::Reap {
                    pid,
                    dst,
                    has_result,
                })
            }
            Outcome::Exec {
                cmd,
                grants_ptr,
                grants_n,
                entry,
                size_log2,
                dst,
                personality,
            } => {
                return Ok(VcpuStop::Exec {
                    cmd,
                    grants_ptr,
                    grants_n,
                    entry,
                    size_log2,
                    dst,
                    personality,
                })
            }
            Outcome::ForkSelf { dst } => return Ok(VcpuStop::ForkSelf { dst }),
            Outcome::SpawnSelf { cmd, plan, dst } => {
                return Ok(VcpuStop::SpawnSelf { cmd, plan, dst })
            }
            Outcome::PipeRead { pipe } => return Ok(VcpuStop::PipeRead { pipe }),
            Outcome::PipeWrite { pipe } => return Ok(VcpuStop::PipeWrite { pipe }),
            Outcome::ReapWait { child } => return Ok(VcpuStop::ReapWait { child }),
            Outcome::Instantiate { spawn, dst } => return Ok(VcpuStop::Instantiate { spawn, dst }),
            Outcome::InstantiateDetached { spawn, dst } => {
                return Ok(VcpuStop::InstantiateDetached { spawn, dst })
            }
            Outcome::CapPending { id, dst } => return Ok(VcpuStop::CapPending { id, dst }),
            Outcome::MemoryWait {
                base,
                expected,
                width,
                timeout,
                dst,
            } => {
                return Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                })
            }
            // Blocking-stdin park (owned-host session): surface it for the `Vcpu` driver to pump.
            Outcome::StdinPark => return Ok(VcpuStop::StdinPark),
            Outcome::MemoryNotify { base, count, dst } => {
                return Ok(VcpuStop::Notify { base, count, dst })
            }
            Outcome::JitInstall { h, code, dst } => {
                return Ok(VcpuStop::JitInstall { h, code, dst })
            }
            Outcome::JitUninstall { h, slot, dst } => {
                return Ok(VcpuStop::JitUninstall { h, slot, dst })
            }
            Outcome::JitInvoke {
                h,
                code,
                argv,
                dst,
                params,
                results,
            } => {
                return Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                })
            }
            // §GC `gc.roots`: scan the whole vCPU continuation — the active window, its call stack
            // (covered by `scan_vm_roots`), every resume-chain ancestor, every parked fiber, and every
            // suspended coroutine — for words that (masked) land in `[lo, hi)`. A **sound superset**
            // of the genuine roots, kept in-window by the range filter (GC.md §3.2).
            Outcome::GcRoots {
                lo,
                hi,
                mask,
                buf,
                cap,
                dst,
            } => {
                let roots = fibers.with(|f, _, _| gc_scan(vt, f, &dom.source, lo, hi, mask));
                let total = gc_write(ctx.mem, buf, cap, roots)?;
                vt.active.set(dst, Reg::from_i64(total));
            }
        }
    }
}

/// `fiber_sig` params/results, inlined so the driver can compare without allocating a `FuncType`.
const FIBER_PARAMS: [ValType; 2] = [ValType::I64, ValType::I64];
const FIBER_RESULTS: [ValType; 1] = [ValType::I64];

/// Resolve a fiber `funcref` through the domain's shared `call.dyn` dispatch table — **module-aware,
/// exactly as [`Op::CallIndirect`]** — so a fiber may start on an **installed §22 unit** function (a
/// module ≥ 1 entry), not only a module-0-natural one (#1226). The install slot the guest passed to
/// `cont.new` is a dispatch-table index just like a `call.dyn` target, so resolving it here matches how
/// the fiber's own calls dispatch and how the tree-walker (`dispatch_indirect`) and Cranelift JIT
/// (shared `fn_table`) resolve the same entry. Returns the target `(module, func, its Compiled)`, or
/// `None` (⇒ `FiberFault`) for an empty padding slot or a signature that is not the fiber-entry type
/// `(i64, i64) -> i64`. Pass the same `(source, table)` pair the fiber's `resume` uses, so entry
/// resolution and in-fiber dispatch agree.
fn resolve_fiber_entry(
    source: &ModuleSource,
    table: &SharedSlots,
    funcref: i32,
) -> Option<(usize, usize, std::sync::Arc<Compiled>)> {
    let ts = table.slot((funcref as u32 as usize) & (table.len() - 1));
    if ts.module == super::TABLE_EMPTY {
        return None;
    }
    let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
    let tm = source.get(tmod)?;
    let (p, r) = tm.sigs.get(tfunc)?;
    if p[..] == FIBER_PARAMS && r[..] == FIBER_RESULTS {
        Some((tmod, tfunc, tm))
    } else {
        None
    }
}

/// A §14 `instantiate` child's confined runtime, owned by [`drive`] alongside the task set. Its `mem`
/// is a `nested_view` sub-window sharing the parent's backing (the §14 shared data plane), its `host`
/// an attenuated powerbox (an `Instantiator` + an `AddressSpace`, each over `[0, child_size)`), its
/// `table` a fresh **natural** dispatch table over module 0 (no access to installed §22 units — like
/// the tree-walker's fresh `DomainTable::new(&cfuncs, 0)`), and `fuel` a sub-allocated quota.
struct ChildEnv {
    mem: Option<Mem>,
    /// The child's live powerbox. `Arc<Mutex<…>>` (single-threaded here, so uncontended) so a
    /// §3.6 live-callee offer can hold the SAME callee the tree-walker's `wire_live_impl`
    /// machinery expects — enqueue, offer-shape, and settle all go through the shared type.
    host: std::sync::Arc<std::sync::Mutex<Host>>,
    table: SharedSlots,
    fuel: u64,
    /// The child domain's own §12 fiber registry (with its durable halves): each domain numbers its
    /// fibers from 0 and cannot reach another's, as on the tree-walk oracle — the parallel driver's
    /// per-domain [`ParDomain`] registry, here. The root domain's is [`CoopSched::fibers`].
    fibers: FiberTables,
}

/// #1727 — the powerbox a task's own handles live in: its §14 [`ChildEnv`]'s, or the domain's for the
/// root and its `thread.spawn` siblings. Every handle a spawn names (the module, the by-name grants,
/// the `Budget`) resolves here. Resolving them in the root's table instead let a nested child spawn
/// with authority only the root held.
fn task_host<'a>(host: &'a mut Host, envs: &'a [ChildEnv], env: Option<usize>) -> HostCell<'a> {
    match env {
        None => HostCell::Excl(host),
        Some(k) => HostCell::Shared(&envs[k].host),
    }
}

/// Schedule an admitted §14/§5 child as a task of the cooperative executor over its own environment,
/// registered as a child handle of task `ti`, and land the handle in `dst`. `tierup` is the run's
/// eligibility bitmap and page-check flag, for a child that inherits them (see the confined arm).
/// `Err(ThreadFault)` on the vCPU-count bomb; the caller completes `ti` with it.
#[allow(clippy::too_many_arguments)]
fn coop_start_child(
    tasks: &mut Vec<TaskSlot>,
    extra_envs: &mut Vec<ChildEnv>,
    ti: usize,
    source: &ModuleSource,
    child: AdmittedChild,
    entry: i64,
    dst: u32,
    tierup: Option<(std::sync::Arc<[bool]>, bool)>,
) -> Result<(), Trap> {
    let live = tasks
        .iter()
        .filter(|t| !matches!(t.state, TaskState::Done(_)))
        .count();
    if live >= super::MAX_VCPUS {
        return Err(Trap::ThreadFault); // instantiate bomb
    }
    let AdmittedChild {
        mem,
        host,
        program,
        args,
        fuel,
        lease,
    } = child;
    let (module, prog) = program.land(source)?;
    let (mut vt, table) = child_task(module, &prog, entry, &args, host.jit_table_log2())?;
    if let Some((eligible, page_checked)) = tierup {
        vt.active.jit_eligible = Some(eligible);
        vt.active.jit_page_checked = page_checked;
    }
    let eidx = extra_envs.len();
    extra_envs.push(ChildEnv {
        mem,
        host: std::sync::Arc::new(std::sync::Mutex::new(host)),
        table,
        fuel,
        fibers: FiberTables::default(),
    });
    let cidx = tasks.len();
    tasks.push(TaskSlot {
        vt,
        threads: Vec::new(),
        env: Some(eidx),
        state: TaskState::Runnable,
        suspended: None,
        lease: lease.map(|(budget, bytes)| (tasks[ti].env, budget, bytes)),
    });
    let handle = tasks[ti].threads.len() as i32;
    tasks[ti].threads.push(Some(cidx));
    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
    Ok(())
}

/// A scheduled vCPU and its blocking state.
struct TaskSlot {
    vt: VTask,
    /// This vCPU's `thread.spawn` / `instantiate` children (handle = index → global task index).
    /// `None` = joined. (Both seams share one handle namespace, matching the tree-walker's `threads`.)
    threads: Vec<Option<usize>>,
    /// The runtime environment this vCPU steps against: `None` = the shared domain (root + its
    /// `thread.spawn` siblings); `Some(k)` = the confined `extra_envs[k]` of a §14 `instantiate` child
    /// (and any threads it spawns, which share its window — they inherit the same env index).
    env: Option<usize>,
    state: TaskState,
    /// #1896 — the rest of a call that parked in a bounce out of this task's emitted leaf, run once
    /// the park clears.
    suspended: Option<Box<Suspended>>,
    /// A detached child's window lease `(spawner env, budget, bytes)`, on the child's root task:
    /// once the task is `Done` the scheduler returns the bytes to the spawner's budget
    /// ([`refund_ended_windows`]).
    lease: Option<(Option<usize>, i32, u64)>,
}

/// Return the window bytes of every detached child whose root task has ended to the budget that paid
/// for them, in the spawner's own powerbox — `Budget.mem` accounts live windows (INVARIANTS #3,
/// 2026-09-29). Run at the top of every scheduling round, so the refund lands before any task runs
/// again.
fn refund_ended_windows(tasks: &mut [TaskSlot], host: &mut Host, envs: &[ChildEnv]) {
    for t in tasks.iter_mut() {
        if matches!(t.state, TaskState::Done(_)) {
            if let Some((env, budget, bytes)) = t.lease.take() {
                task_host(host, envs, env).with(|h| h.budget_mem_give(budget, bytes));
            }
        }
    }
}

enum TaskState {
    Runnable,
    /// Parked on `thread.join` of task `child`; deliver its result to `dst` and wake.
    BlockedJoin {
        child: usize,
        slot: usize,
        dst: u32,
    },
    /// Parked on `memory.wait` at futex `key` until notified or `deadline` (logical clock). The key is
    /// **backing-identity canonical** ([`super::FutexKey`]): two confined `instantiate` children that
    /// mapped the same `SharedRegion` into their separate windows park/wake on the same key (S1c), so a
    /// pipe ring between concurrent stages rendezvous. A plain `thread.spawn` sibling (shared root
    /// window, anonymous page) keys on its confined address — `FutexKey::Anon`, as before.
    BlockedWait {
        key: super::FutexKey,
        /// Logical-clock deadline, or `None` for an infinite wait — which is therefore not a
        /// clock-advance candidate at driver idle (#1638), so a run where every remaining
        /// waiter is indefinite reaches `drive`'s deadlock exit instead of being handed a
        /// `WAIT_TIMED_OUT` the guest never asked for.
        deadline: Option<u64>,
        dst: u32,
    },
    /// §3.6 (I36 slice 2): parked in `svc.wait` on this task's own domain (its env's host); a
    /// caller's enqueue on that host re-admits it (the rewound op re-executes the drain).
    BlockedSvc,
    /// §3.6 (I36 slice 2): parked on a live-call `ticket` against `callee`'s completion cells;
    /// the settle-wake scan delivers the reply to `dst` (the claim — the tree-walker's
    /// `cap_reply` preference, cooperative form).
    BlockedTicket {
        ticket: u64,
        callee: std::sync::Arc<std::sync::Mutex<Host>>,
        dst: u32,
    },
    /// FORK.md §9.2 — parked in `reap` (`wait(pid)`) until fork twin `pid` (a task index) finishes;
    /// the settle scan delivers its exit status ([`super::reap_status`]) to `dst` and wakes. A
    /// trapped twin reaps as a nonzero crash status, never a propagated trap (reap ≠ join).
    BlockedReap {
        pid: usize,
        dst: u32,
    },
    /// #799/#1080 — parked in a personality **`waitpid()`** (`ParkEvent::TaskExit`/`TaskExitAny`, the
    /// bytecode port). Unlike [`Self::BlockedReap`] (the serve-handler `reap` that injects a status into
    /// `dst`), this rewound the guest's `waitpid` op: on `child` completing (`Some(pid)` = a specific
    /// twin, `None` = any child), the settle scan makes this task `Runnable` and the op **re-executes**,
    /// so the personality's own `waitpid` serves the exit status (writing it to guest memory, returning
    /// the pid) against the twin it has by then retired (Live → Zombie via the fired exit hooks).
    BlockedReapPersonality {
        child: Option<usize>,
    },
    /// #1080 rung 4 — parked in a blocking personality pipe **read** on an empty FIFO (writers open). The
    /// read op was rewound; the settle scan polls the shared FIFO via `pipe_read_ready(pipe)` (`pipe` =
    /// this task's domain-local index into its env host's pipe table) and re-admits when bytes arrive or
    /// every writer closed (EOF), so the re-executed read completes / EOFs. The cooperative analogue of
    /// the tree-walker's `Blocked::PipeRead` — polled, not held in a `pipe_waiters` map.
    BlockedPipeRead {
        pipe: u32,
    },
    /// #1080 rung 4 (backpressure) — parked in a blocking personality pipe **write** to a full FIFO
    /// (readers open); re-admitted by `pipe_write_ready(pipe)` (room under `PIPE_CAP`, or all readers gone
    /// → the re-run `-EPIPE`s). The write twin of [`Self::BlockedPipeRead`].
    BlockedPipeWrite {
        pipe: u32,
    },
    /// #1146 (deeper) — parked in a blocking **`Stream{In}` read** on an exhausted stdin under
    /// [`super::Host::set_stdin_blocking`] ([`Outcome::StdinPark`]). The read op was rewound; the
    /// settle scan re-admits when the (root-host) stdin buffer has bytes (`stdin_ready`), and the
    /// all-parked signal sweep interrupts it like a pipe park (the rewound read completes `-EINTR`).
    /// The cooperative analogue of the tree-walker's `Blocked::CapRead` stdin park. Note: `push_stdin`
    /// takes `&mut Host`, which the run exclusively borrows, so no *concurrent* data feed can reach a
    /// coop stdin park today — the only live wake is the signal interrupt; the readiness re-admit is
    /// the correct contract kept for symmetry (and a future shared-Host feed), not a reachable path.
    BlockedStdin,
    /// I48 — parked in a blocking `cont.resume.block` on fiber `fiber` (event-parked, not yet woken).
    /// The resumer's cursor was rewound to the resume op; when `fiber` is woken (idle-timer, notify,
    /// or the cap-completion drain) this task is marked `Runnable` and re-executes the resume, which
    /// now claims the woken fiber and switches in. Burns no fuel while parked (skipped by the runnable
    /// scan) — the idle-not-spin proof.
    BlockedOnFiber {
        fiber: usize,
    },
    /// Finished — its result (or trap) is retained for a joiner.
    Done(Result<Vec<Value>, Trap>),
}

/// Park `vt`'s running fiber (never fiber 0) as `park(vm)` — §3.6 slice 5a, the one route every
/// fiber park takes: unwind one chain link to the fiber's resumer and set its `Vm` aside in `fibers`.
/// The resumer gets `(FIBER_PARKED, 0)`. Under a blocking resume (I48) on a driver that `can_idle`,
/// the resumer instead re-executes its resume: at once when `woken(fibers)` says the park's event
/// already fired (the park-time recheck), else after idling on the fiber — the returned slot, which
/// the caller parks its task on (`BlockedOnFiber`).
#[allow(clippy::too_many_arguments)]
fn park_running_fiber(
    vt: &mut VTask,
    fibers: &mut [FiberState],
    fiber_sp: &mut [u64],
    mem: &mut Option<Mem>,
    durable: bool,
    can_idle: bool,
    park: impl FnOnce(Vm) -> FiberState,
    woken: impl FnOnce(&mut [FiberState]) -> bool,
) -> Option<usize> {
    let k = vt.active_id;
    // I48: the blocking-resume marker, set at the claim, before the park overwrites `Running`.
    let blocking_ip = match fibers.get(k) {
        Some(FiberState::Running { blocking_ip, .. }) => *blocking_ip,
        _ => None,
    };
    let (rid, resumer, rdst) = vt.chain.pop().expect("a running fiber has a resumer");
    shadow_switch(mem, fiber_sp, &mut vt.root_shadow_sp, durable, k, rid);
    fibers[k] = park(std::mem::replace(&mut vt.active, resumer));
    vt.active_id = rid;
    let woken = woken(fibers);
    if let (Some(ip), true) = (blocking_ip, can_idle) {
        vt.active.pc = ip;
        return (!woken).then_some(k);
    }
    vt.active.set(rdst, Reg::from_i32(super::FIBER_PARKED));
    vt.active.set(rdst + 1, Reg::from_i64(0));
    None
}

/// #1952 — park `vt`'s running fiber on a host op that must wait ([`FiberState::HostParked`]):
/// [`park_running_fiber`], with `ready` the park-time recheck.
#[allow(clippy::too_many_arguments)]
fn park_fiber_on_host(
    vt: &mut VTask,
    fibers: &mut [FiberState],
    fiber_sp: &mut [u64],
    mem: &mut Option<Mem>,
    durable: bool,
    can_idle: bool,
    on: HostWait,
    ready: bool,
) -> Option<usize> {
    park_running_fiber(
        vt,
        fibers,
        fiber_sp,
        mem,
        durable,
        can_idle,
        |vm| FiberState::HostParked { vm, on },
        |_| ready,
    )
}

/// #1952 — task `ti`'s running fiber (never fiber 0) parks on the host op `stop` must wait for
/// ([`FiberState::HostParked`]), in its task's domain: the root's registry and window, or its
/// confined `instantiate` env's. Nothing parks for a vanished pipe: its op re-runs, and fails closed.
#[allow(clippy::too_many_arguments)]
fn host_park_fiber(
    tasks: &mut [TaskSlot],
    ti: usize,
    fibers: &mut Vec<FiberState>,
    fiber_sp: &mut Vec<u64>,
    mem: &mut Option<Mem>,
    extra_envs: &mut [ChildEnv],
    host: &mut Host,
    stop: &VcpuStop,
) {
    let durable = host.is_durable();
    let Some((on, ready)) = task_host(host, extra_envs, tasks[ti].env).with(|h| {
        HostWait::of(stop, h).map(|on| {
            let ready = on.ready(h);
            (on, ready)
        })
    }) else {
        return;
    };
    let (fibers, fiber_sp, mem) = match tasks[ti].env {
        None => (fibers, fiber_sp, mem),
        Some(e) => {
            let e = &mut extra_envs[e];
            (&mut e.fibers.fibers, &mut e.fibers.sp, &mut e.mem)
        }
    };
    if let Some(k) = park_fiber_on_host(
        &mut tasks[ti].vt,
        fibers,
        fiber_sp,
        mem,
        durable,
        true,
        on,
        ready,
    ) {
        tasks[ti].state = TaskState::BlockedOnFiber { fiber: k };
    }
}

impl TaskState {
    /// #1904 — the [`super::ParkSite`] this state parks at, so a freeze applies the one rule for it
    /// ([`super::ParkSite::freeze_rule`], DURABILITY §4 "One rule per park site"). Exhaustive, so a
    /// new state cannot be missed.
    fn park_site(&self) -> Option<super::ParkSite> {
        use super::ParkSite;
        match self {
            TaskState::Runnable | TaskState::Done(_) => None,
            // A serve-handler `reap` is a `wait(pid)`, which the oracle files with `thread.join`.
            TaskState::BlockedJoin { .. } | TaskState::BlockedReap { .. } => Some(ParkSite::Join),
            TaskState::BlockedWait { .. } => Some(ParkSite::Futex),
            // A `cont.resume.block` resumer idles beside the serve loop, as on the oracle.
            TaskState::BlockedSvc | TaskState::BlockedOnFiber { .. } => Some(ParkSite::Svc),
            TaskState::BlockedTicket { .. } => Some(ParkSite::Reply),
            TaskState::BlockedReapPersonality { .. } => Some(ParkSite::Reap),
            TaskState::BlockedPipeRead { .. } => Some(ParkSite::PipeRead),
            TaskState::BlockedPipeWrite { .. } => Some(ParkSite::PipeWrite),
            TaskState::BlockedStdin => Some(ParkSite::StreamRead),
        }
    }
}

/// #1904 — the pump's freeze step with nothing runnable, the oracle's `freeze_step`. `true` when it
/// acted, so the pump looks at its tasks again.
///
/// - **Freeze-on-quiesce** fires when every park is one a freeze can end (the oracle's
///   `quiesced_parks_only`: a `svc.wait`, or an indefinite futex wait). The census runs first: a
///   fork twin, or a vCPU parked where the freeze has no rule, declines it on the run's powerbox
///   and spends the arm, and the run carries on as if it was never armed.
/// - **A freeze in flight** (the window reads `UNWINDING`) brings every park through it by its
///   site's rule ([`admit_parks`]): a re-issue site is re-admitted — its re-executed op observes the freeze and is
///   abandoned, a futex wait takes `WAIT_FROZEN` (#1769) — and a phase site stays parked for the
///   completion that wakes it. Nothing else could wake a parked task, so without this the freeze
///   would hang on it.
fn freeze_step(
    tasks: &mut [TaskSlot],
    fibers: &[FiberState],
    forked_twins: &std::collections::BTreeSet<usize>,
    mem: &mut Option<Mem>,
    host: &mut Host,
    freeze_on_quiesce: &mut bool,
) -> bool {
    use super::{DeclineCause, FreezeDeclined, FreezeRule};
    if *freeze_on_quiesce && quiesced_parks_only(tasks, fibers) {
        *freeze_on_quiesce = false;
        let parked_declined = tasks.iter().enumerate().find_map(|(ti, t)| {
            let site = t.state.park_site()?;
            (freeze_rule_here(site) == FreezeRule::Decline).then_some((site, ti))
        });
        let declined = match (forked_twins.first(), parked_declined) {
            (Some(&twin), _) => Some((DeclineCause::ForkTwin, twin)),
            (None, Some((site, ti))) => Some((DeclineCause::Parked(site), ti)),
            (None, None) => None,
        };
        if let Some((cause, task)) = declined {
            host.freeze_declined = Some(FreezeDeclined {
                cause,
                task: task as u64,
                slot: None,
            });
            return true;
        }
        if let Some(m) = mem.as_mut() {
            m.durable_set_state(super::STATE_UNWINDING);
        }
        // Fired: even with nothing to re-admit, a stopped task now sees through its stop.
        admit_parks(tasks);
        return true;
    }
    host.is_durable() && is_unwinding(mem) && admit_parks(tasks)
}

/// The rule this engine applies at `site`: the oracle's, but for a reply wait. The oracle re-issues the
/// wait and the thaw re-parks the call on its ticket (#1901); this engine has no re-park yet, so a
/// re-admitted call would be issued twice, and it declines instead (#1904).
fn freeze_rule_here(site: super::ParkSite) -> super::FreezeRule {
    match site {
        super::ParkSite::Reply => super::FreezeRule::Decline,
        s => s.freeze_rule(),
    }
}

/// A freeze in flight: re-admit every task parked at a [`super::FreezeRule::Reissue`] site (see
/// [`freeze_step`]). `true` when it re-admitted one.
fn admit_parks(tasks: &mut [TaskSlot]) -> bool {
    use super::FreezeRule;
    let mut admitted = false;
    for t in tasks.iter_mut() {
        let Some(site) = t.state.park_site() else {
            continue;
        };
        if freeze_rule_here(site) != FreezeRule::Reissue {
            continue;
        }
        // The freeze ended this wait, not its event: the thaw re-issues it (#1769).
        if let TaskState::BlockedWait { dst, .. } = t.state {
            t.vt.active
                .set(dst, Reg::from_i32(temen_ir::durable_abi::WAIT_FROZEN));
        }
        t.state = TaskState::Runnable;
        admitted = true;
    }
    admitted
}

/// The oracle's `quiesced_parks_only`: something is parked that a freeze could re-admit — a
/// `svc.wait` (or a resumer idling beside it) or a futex wait — and no futex wait has a deadline.
fn quiesced_parks_only(tasks: &[TaskSlot], fibers: &[FiberState]) -> bool {
    let mut any = false;
    for t in tasks {
        match t.state {
            TaskState::BlockedSvc | TaskState::BlockedOnFiber { .. } => any = true,
            TaskState::BlockedWait { deadline: None, .. } => any = true,
            TaskState::BlockedWait { .. } => return false,
            _ => {}
        }
    }
    for f in fibers {
        match f {
            FiberState::WaitParked {
                deadline: None,
                woken: None,
                ..
            } => any = true,
            FiberState::WaitParked { woken: None, .. } => return false,
            _ => {}
        }
    }
    any
}

/// Drive a whole domain — the entry vCPU plus any `thread.spawn` children — to completion on a
/// **cooperative single-threaded scheduler** sharing one `Mem`. The oracle's concurrent programs are
/// interleaving-invariant (verified by the tree-walker via stress / seed-sweep / DPOR), so any
/// correct schedule yields the same result; a deterministic lowest-index-first pick keeps it
/// reproducible. Blocking (`join` / `wait`) parks a task; `notify` / child completion wakes it; a
/// stuck set advances a logical clock to the next `wait` deadline (or deadlocks → `ThreadFault`,
/// matching the deterministic explorer). The run ends when the **root** vCPU completes.
fn drive(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut Host,
    budget: u64,
) -> Result<Vec<Value>, Trap> {
    // The native driver never enables tier-up (no eligibility bitmap), so `pump` runs the whole
    // schedule and returns `Done`; a `TierUp` yield is impossible here.
    let mut sched = CoopSched::new(&dom, entry, args, fuel, mem, host, None)?;
    match sched.pump(&dom, mem, host, fuel, budget)? {
        CoopStep::Done(vals) => Ok(vals),
        CoopStep::Idle => unreachable!("idle suspension not enabled on the native driver"),
        CoopStep::Paused => unreachable!("slicing not enabled on the native driver"),
        CoopStep::TierUp { .. } | CoopStep::Resume { .. } => {
            unreachable!("tier-up not enabled on the native driver")
        }
        CoopStep::JitInvoke { .. } => {
            unreachable!("Jit.invoke surfacing not enabled on the native driver")
        }
    }
}

/// A pause point of [`CoopSched::pump`]: either the run finished (`Done`) or a module-0 task hit an
/// eligible direct `Call` and the emitted region must run on the host (`TierUp`) before the paused
/// task is resumed via [`CoopSched::deliver_tierup`]. `pump` returns `Err(trap)` for a run-fatal trap
/// (the root task trapped, or a driver operation failed) — mirroring the `Result` `drive` returns.
enum CoopStep {
    /// The root task returned; these are the run's results.
    Done(Vec<Value>),
    /// The slice of a [`CoopRun::run_for`] pump is spent; the run is live and resumable.
    Paused,
    /// #1122 route (a) — every task is parked, nothing internal can wake one, and at least one park is
    /// externally wakeable (a terminal/pipe read or a blocking stdin read): with
    /// [`CoopSched::suspend_on_idle`] set the pump RETURNS here instead of blocking on the doorbell,
    /// so an embedder that owns the run (a [`CoopRun`]) can feed input and pump again — the whole
    /// scheduler state stays inside the `CoopSched`. Without the flag this state blocks on the bell
    /// (or is the deadlock it always was).
    Idle,
    /// A task paused to run `func` of program `module` emitted ([`CoopEvent::TierUp`]) with raw i64
    /// arg slots `argv`; `mapped` is the window's committed scalar extent for the emitted `"mapped"`
    /// global (#717 host sync).
    TierUp {
        module: u32,
        func: u32,
        argv: Box<[i64]>,
        mapped: u64,
    },
    /// #1896 — a leaf's call that parked in a bounce returned these raw i64 result slots: the host
    /// resumes the leaf's suspended frames with them ([`CoopEvent::Resume`]).
    Resume { results: Box<[i64]> },
    /// A task paused on a §22 `Jit.invoke` of a runtime-compiled unit that has **emitted wasm**, an
    /// all-scalar signature, and a representable window — the host runs the unit's `f0` and delivers
    /// the results back ([`CoopSched::deliver_jit_invoke_vals`]). `code` is the unit's code handle,
    /// `wasm` its emitted module, `argv` the raw i64 arg slots, `params`/`results` the unit entry's
    /// scalar signature, `mapped` the committed extent. A unit without emitted wasm (or a non-scalar
    /// signature / unrepresentable window) is serviced interpreted inside the pump and never surfaces.
    JitInvoke {
        code: i32,
        wasm: std::sync::Arc<[u8]>,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: u64,
    },
}

/// Where a surfaced tier-up's results land ([`CoopSched::pending_tierup`]).
#[derive(Clone, Copy)]
enum TierUpDst {
    /// The paused caller frame's result slots, from this one on.
    Frame(usize),
    /// No frame: the task tiered up at its entry (#1896), so its results end it. `parks`: its host
    /// suspends its emitted frames where a call parks ([`LeafOffer::parks`]).
    Entry { parks: bool },
}

/// #1896 — a leaf running emitted whose call parked in a bounce ([`LeafOffer::parks`]): its host
/// holds the emitted frames, suspended, and this is the rest of the call, which the pump runs once
/// the park clears ([`CoopStep::Resume`]).
struct Suspended {
    /// The call's interpreted continuation, its parked op rewound.
    vm: Vm,
    /// Where the leaf's results land, and their types ([`CoopSched::pending_tierup`]).
    dst: TierUpDst,
    results: Box<[ValType]>,
}

/// The cooperative multiplex scheduler's run-shared state, extracted from `drive` so that a future
/// resumable-across-FFI tier-up driver (#926 slice 2) can own it between host round-trips. `drive`
/// builds one with [`new`](CoopSched::new) and runs it to completion with [`pump`](CoopSched::pump);
/// the fields are exactly the run-shared locals `drive` used to hold — the task set, the run-shared
/// §12 fiber registry (+ its durable parallel arrays), the §14 confined child environments, the
/// fork/teardown bookkeeping, and the logical clock.
struct CoopSched {
    /// The live vCPUs: the root task (index 0) and its `thread.spawn`/`instantiate` descendants.
    tasks: Vec<TaskSlot>,
    /// §14 `instantiate` children's confined environments (handle = `env` index). The root and its
    /// `thread.spawn` siblings share `mem`/`host`/`dom.table` instead (`env == None`).
    extra_envs: Vec<ChildEnv>,
    /// The root domain's §12 fiber registry, shared by its vCPUs so a fiber created or suspended on
    /// one can be resumed on another (D57 migration). Each §14 child domain has its own
    /// ([`ChildEnv::fibers`]).
    fibers: Vec<FiberState>,
    /// DURABILITY.md §12.8: each fiber's saved durable shadow-SP (run-shared, parallel to `fibers`;
    /// slot `s` is shadow context `s + 1`). Inert on a non-durable run.
    fiber_sp: Vec<u64>,
    /// Freeze residue (DURABILITY.md §12.8): each fiber's `(resolved entry func index, data-stack
    /// base)` — what a [`super::FrozenFiber`] needs after the fiber parks. Parallel to `fibers`.
    fiber_meta: Vec<(i32, i64)>,
    /// §12 teardown: child envs already torn down by a member's trap/exit (D37 death-is-revocation).
    dead_envs: std::collections::BTreeSet<usize>,
    /// FORK.md §9.2 — fork twins minted this run (task index = the pid a `clone_caller` returned).
    forked_twins: std::collections::BTreeSet<usize>,
    /// #799/#1080 — fork twins whose personality **exit hooks have already fired** (Live → Zombie at
    /// their completion). Fired once, in the settle scan, before a personality `waitpid` re-execution
    /// reads the retired status (the bytecode port of the tree-walker's exit-hook-at-death step).
    hooked_twins: std::collections::BTreeSet<usize>,
    /// FORK.md §8.6 / #1807 — child envs (§14 children and fork twins) whose pipe ends were released at
    /// their domain's finish. Released once: the release decrements the shared end counts.
    released_envs: std::collections::BTreeSet<usize>,
    /// The scheduler's logical clock (advanced only when no task is runnable, to the earliest due
    /// `wait` deadline).
    clock: u64,
    /// #926 slice 2: wasm-JIT tier-up eligibility for this run's **module-0** tasks (the root and its
    /// same-module `thread.spawn` descendants). `None` ⇒ everything interprets, exactly the native
    /// `drive` (which never sets it). When set, each qualifying task's `Vm` carries it, so a direct
    /// module-0 `Call` to an eligible function surfaces as [`CoopStep::TierUp`] instead of interpreting
    /// the callee — the host runs the emitted `f{func}` and delivers the results back.
    eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 paged tier-up: the eligible set is page-checked (the emitted region carries a per-access
    /// page check), so an unrepresentable window surfaces with the reserved size instead of declining.
    page_checked: bool,
    /// #1896: the host's emitter for leaf images ([`TierUpConfig::leaf`]); `None` interprets them.
    leaf: Option<LeafEmitter>,
    /// The task currently paused on a surfaced tier-up, awaiting [`deliver_tierup`](Self::deliver_tierup):
    /// `(task index, where its results land, result types)`. At most one is ever outstanding —
    /// the driver services one tier-up round-trip before pumping again — so a single slot suffices.
    /// `None` between round-trips (and always, on the native driver).
    pending_tierup: Option<(usize, TierUpDst, Box<[ValType]>)>,
    /// The task currently paused on a surfaced §22 `Jit.invoke`, awaiting
    /// [`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals): `(task index, dst slot, result
    /// types)` — the same one-outstanding-round-trip discipline as `pending_tierup` (a tier-up and an
    /// invoke are never outstanding at once: each is one `pump` yield). `None` off the browser driver.
    pending_jit: Option<(usize, usize, Box<[ValType]>)>,
    /// #926 slice 2f / #1233 — the driver-table **slot → unit-identity mirror** for the browser B2
    /// coop driver: `slot_units[s]` is the `(domain, unit)` a guest `Jit.install`ed at dispatch slot
    /// `s` (`None` = empty or a natural-prefix slot), recorded wherever an install is serviced so the
    /// JS host can rebuild its `WebAssembly.Table`. Sized `1 << host.jit_table_log2()` — length 1 and
    /// unused on the native `drive` (no shared table).
    ///
    /// The key is the **unit index, not the code handle**: a handle is guest-revocable and
    /// `compile → install → release` is the ordinary pattern (the unit stays installed, only its
    /// handle dies), so a mirror keyed on handles lost every released unit at the next table rebuild
    /// and nulled a live slot (`IndirectCallToNull`, #1233). The unit index is append-only, so this
    /// key never dies while the slot is filled.
    slot_units: Vec<Option<(u32, u32)>>,
    /// #1009: a generation counter bumped on each `Jit.install`/`Jit.uninstall` (the only `slot_units`
    /// mutations) so the browser B2 driver rebuilds its `WebAssembly.Table` only when the mirror
    /// changed — a dispatch-heavy card that never installs syncs the table once, not per tier-up.
    table_gen: u32,
    /// #926 slice 2g — the **invoke-confined** fiber registry for a surfaced emitted `Jit.invoke`'s
    /// cross-tier bounces (the twin of [`Vcpu::invoke_fibers`]). While a `Jit.invoke` unit runs on the
    /// host, its `env.call_interp` callbacks share this registry across the invoke's several bounces (a
    /// fiber one callback parks is resumable by a later bounce of the *same* invoke), then it is cleared
    /// when the invoke resolves ([`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals) / `_trap`) —
    /// so an invoke's fibers die with it, exactly as the interpreted `run_invoke`'s loop-local registry
    /// does. A tier-up region's bounces use the run-level `fibers` instead (a parked fiber persists for
    /// the run to resume). Empty except during an outstanding invoke; always empty on the native `drive`.
    invoke_fibers: Vec<FiberState>,
    /// #1122 route (a) — when set, an all-parked, externally-wakeable settle yields
    /// [`CoopStep::Idle`] to the driver instead of blocking on the #1122 doorbell (see
    /// [`CoopRun::set_suspend_on_idle`]). Off on the native `drive` and the blocking browser session.
    suspend_on_idle: bool,
    /// Ops left in this `pump` call's slice ([`CoopRun::run_for`]), or `None` for an unsliced pump.
    /// At zero the pump returns [`CoopStep::Paused`], every task's cursor persisted — the next pump
    /// resumes exactly where this one stopped, as after a #1157 preemption.
    slice_left: Option<u64>,
    /// DURABILITY.md §13.4 slice 4c-bis — **freeze-on-quiesce**, one-shot: a durable run armed to
    /// freeze the moment it quiesces ([`freeze_step`]). Read once at run setup from the window's
    /// arm-quiesce flag, as the oracle's `Sched::freeze_on_quiesce`.
    freeze_on_quiesce: bool,
}

/// #1262 — wire a domain's personality signal doors to the cooperative pump's `#1122` external-wake
/// bell, so an embedder signal (a terminal `^C`/`^Z`, a `kill(1)`) delivered *while the pump is
/// all-parked* rings the bell and re-runs the settle instead of being slept through. Every door is the
/// same ring — "something changed" — and the pump's settle does the real work (pipe poll, `reap_pending`
/// re-admit, the `#1215` loop-top kill sweep). Two gaps this closes: (1) `set_kill` was never wired, so a
/// default-action TERMINATE (`^C` of a job with no handler) set `term_sig` but woke nothing; (2) a fork
/// twin's doors were never wired at all, so an embedder signal to a foreground/background *twin* (the
/// shape of a real `cat`) could not ring the root bell. The tree-walker points these doors at
/// `interrupt_interruptible_parks`/`wake_stopped`; the cooperative pump needs only the ring.
fn wire_pump_bell(
    source: &std::sync::Arc<dyn super::SignalSource + Send + Sync>,
    bell: &std::sync::Arc<(std::sync::Mutex<u64>, std::sync::Condvar)>,
) {
    fn ring(
        bell: &std::sync::Arc<(std::sync::Mutex<u64>, std::sync::Condvar)>,
    ) -> std::sync::Arc<dyn Fn() + Send + Sync> {
        let bell = std::sync::Arc::clone(bell);
        std::sync::Arc::new(move || {
            let (gen, cv) = &*bell;
            *gen.lock().unwrap_or_else(|e| e.into_inner()) += 1;
            cv.notify_all();
        })
    }
    source.set_wake(ring(bell));
    source.set_kill(ring(bell));
    source.set_chld_wake(ring(bell));
    let bell_pw = std::sync::Arc::clone(bell);
    source.set_pipe_wake(std::sync::Arc::new(move |_pipe| {
        let (gen, cv) = &*bell_pw;
        *gen.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        cv.notify_all();
    }));
}

impl CoopSched {
    /// Build the initial scheduler state: the root task at `entry`, plus any fibers a durable freeze
    /// left to re-seed (taken from `host.frozen_fibers`). This is `drive`'s former preamble verbatim
    /// — including the once-per-run entry-fuel charge — so its behaviour is unchanged. `eligible` /
    /// `page_checked` are the run's #926-slice-2 tier-up config: `None`/`false` (the native `drive`)
    /// leaves everything interpreting; a `Some` bitmap makes the root's — and its same-module
    /// `thread.spawn` descendants' — module-0 direct calls to eligible functions surface as tier-ups.
    fn new(
        dom: &Domain,
        entry: FuncIdx,
        args: &[Value],
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut Host,
        tierup: Option<TierUpConfig>,
    ) -> Result<CoopSched, Trap> {
        // `page_checked` is meaningful only with a bitmap, so it rides in the same `Option`.
        let (eligible, page_checked, leaf) = tierup.map_or((None, false, None), |t| {
            (Some(t.eligible), t.page_checked, t.leaf)
        });
        // Fuel unification (safepoint-anchored): charge one fuel for *entering the top-level entry
        // function*, mirroring the per-callee-entry charge at `Op::Call`/`CallIndirect`/`TailCall*` and
        // the JIT's entry-prologue charge, so the tree-walker, bytecode, and JIT engines burn identically.
        // Gated exactly as the tree-walker's `drive` (`super::drive_arc`): a durable **thaw** re-enters to
        // continue an already-charged run (the root re-enters under `REWINDING`), so it must not re-charge.
        let is_thaw = host.is_durable()
            && mem
                .as_ref()
                .is_some_and(|m| m.durable_thaw_state(0) == super::STATE_REWINDING);
        if !is_thaw {
            *fuel = fuel.checked_sub(1).ok_or(Trap::OutOfFuel)?;
        }
        let mut tasks: Vec<TaskSlot> = vec![TaskSlot {
            vt: VTask::new(&dom.source.primary(), entry as usize, args)?,
            threads: Vec::new(),
            env: None,
            state: TaskState::Runnable,
            suspended: None,
            lease: None,
        }];
        // #926 slice 2: arm the root task's `Vm` for tier-up. The entry runs in module 0, so a direct
        // call to an eligible function surfaces (`Vm::resume`'s `module == 0 && jit_eligible[callee]`
        // gate). `thread.spawn` children inherit the bitmap in `pump`'s `Spawn` arm (same-module only).
        if let Some(e) = &eligible {
            tasks[0].vt.active.jit_eligible = Some(std::sync::Arc::clone(e));
            tasks[0].vt.active.jit_page_checked = page_checked;
        }
        // §14 `instantiate` children's confined environments (handle = `env` index). The root and its
        // `thread.spawn` siblings use the shared `mem`/`host`/`dom.table` instead (`env == None`).
        let extra_envs: Vec<ChildEnv> = Vec::new();
        // The §12 fiber registry is **run-shared** (one handle namespace per domain) so a fiber created
        // or suspended on one vCPU can be resumed on another (D57 migration).
        let mut fibers: Vec<FiberState> = Vec::new();
        // DURABILITY.md §12.8: each fiber's saved durable shadow-SP (run-shared, parallel to `fibers`;
        // slot `s` is shadow context `s + 1`). Inert on a non-durable run.
        let mut fiber_sp: Vec<u64> = Vec::new();
        // Freeze residue (DURABILITY.md §12.8): each fiber's `(resolved entry func index, data-stack base)`
        // — what a [`super::FrozenFiber`] needs after the fiber parks (when its `Pending` `funcref`/`sp` are
        // gone). Parallel to `fibers`. Inert on a non-durable run.
        let mut fiber_meta: Vec<(i32, i64)> = Vec::new();
        // Thaw seeding (DURABILITY.md §12.8 slice 3.1.5): a `REWINDING` run re-creates the fibers a freeze
        // flattened *before* the root re-enters, so the root's re-issued `cont.resume` names the same dense
        // handles (0, 1, …) and each fiber's saved shadow-SP is back in `fiber_sp` for the swap to re-point
        // to. Taken (cleared) from the host; empty for a freeze or ordinary run.
        {
            let mut seed = std::mem::take(&mut host.frozen_fibers);
            seed.sort_by_key(|f| f.slot);
            for (expected, ff) in seed.into_iter().enumerate() {
                debug_assert_eq!(
                    expected,
                    fibers.len(),
                    "frozen fibers re-seed densely from slot 0"
                );
                debug_assert_eq!(
                    ff.slot,
                    fibers.len(),
                    "re-seeded slot matches the recorded handle"
                );
                fibers.push(if ff.is_free() {
                    FiberState::Done // #1684: a finished slot keeps its place
                } else {
                    FiberState::Pending {
                        funcref: ff.func,
                        sp: ff.sp,
                        consumed: ff.consumed, // #1538: its first resume delivers at the rewound suspend
                    }
                });
                fiber_sp.push(ff.shadow_sp);
                fiber_meta.push((ff.func, ff.sp));
            }
        }
        let clock: u64 = 0;
        // §12 "Domain lifetime & teardown" (owner 2026-07-24): child envs already torn down by a
        // member's trap/exit — a later live call through one completes with an errno instead of
        // parking forever (D37 death-is-revocation; the tree-walker's dead-callee park probe).
        let dead_envs: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        // FORK.md §9.2 — fork twins minted this run (task index = the pid a `clone_caller` returned). The
        // servicer-side `reap` (`wait`) acts only on ids in this allow-set (a foreign/bogus pid is
        // `-ECHILD`, never a park that hangs); an id is retired when reaped.
        let forked_twins: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let hooked_twins: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        let released_envs: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

        // #1122 — an armed external-wake doorbell: wire the personality's pipe-wake door to ring
        // it (the cooperative twin of the tree-walker's `set_pipe_wake` → scheduler wiring). The
        // pump then BLOCKS on the bell at its would-be all-parked deadlock and re-polls pipe
        // readiness on each ring — an embedder's `feed_terminal` (another OS thread, or another
        // wasm-thread instantiation) is the wake source. The pipe id is unused: the pump's settle
        // already polls every parked pipe, so the ring only needs to say "something changed".
        // #1122/#1146/#1262 — wire the root's personality signal doors to ring the external-wake bell
        // (pipe-wake for a `feed_terminal` byte arrival; `set_wake` for an EINTR-bearing deliverable
        // signal; `set_chld_wake` for a child stop/continue re-scan; `set_kill` for a default-action
        // TERMINATE), so an embedder signal delivered while the pump is all-parked re-runs the settle
        // instead of being slept through. The pump's settle does the work (pipe poll, `reap_pending`,
        // the #1215 kill sweep). See [`wire_pump_bell`].
        if let Some(bell) = host.external_wake() {
            if let Some((_, source)) = host.signal_poll() {
                wire_pump_bell(&source, &bell);
            }
        }

        Ok(CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            dead_envs,
            forked_twins,
            hooked_twins,
            released_envs,
            clock,
            eligible,
            page_checked,
            leaf,
            pending_tierup: None,
            pending_jit: None,
            // Sized to the domain table (`Domain::new(_, host.jit_table_log2())`), so a `Jit.install`'s
            // returned slot always indexes it. `1 << 0 == 1` and unused on the native `drive`.
            slot_units: vec![None; 1usize << host.jit_table_log2()],
            table_gen: 0,
            // Empty until a surfaced `Jit.invoke` bounces; populated only across that invoke's bounces.
            invoke_fibers: Vec::new(),
            suspend_on_idle: false,
            slice_left: None,
            freeze_on_quiesce: host.is_durable()
                && mem.as_ref().is_some_and(|m| m.durable_freeze_on_quiesce()),
        })
    }

    /// Run the scheduler until it must pause — either the run finished ([`CoopStep::Done`]) or a task
    /// hit an eligible module-0 call and its emitted region must run on the host ([`CoopStep::TierUp`],
    /// resumed via [`deliver_tierup`](Self::deliver_tierup)). On the native `drive` (no eligibility),
    /// tier-up never fires, so this runs the whole schedule and returns `Done` — behaviourally the
    /// inline loop `drive` used to run. Each iteration services one runnable vCPU via `step_vcpu` and
    /// settles wakes/teardown; a run-fatal trap is `Err(trap)`.
    fn pump(
        &mut self,
        dom: &Domain,
        mem: &mut Option<Mem>,
        host: &mut Host,
        fuel: &mut u64,
        budget: u64,
    ) -> Result<CoopStep, Trap> {
        let CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            dead_envs,
            forked_twins,
            hooked_twins,
            released_envs,
            clock,
            eligible,
            page_checked,
            leaf,
            pending_tierup,
            pending_jit,
            slot_units,
            table_gen,
            // The invoke-confined registry is threaded only by `CoopRun::bounce` (an emitted invoke's
            // callbacks), never touched by the scheduler loop itself.
            invoke_fibers: _,
            suspend_on_idle,
            slice_left,
            freeze_on_quiesce,
        } = self;
        // #1157 — the round-robin pick cursor (the last task index run). Scanning from `last_pick + 1`
        // (rather than always lowest-index) is what lets the preemption quantum actually rotate: a
        // just-preempted task is re-admitted `Runnable`, and the next pick advances past it to a sibling
        // instead of re-winning. Also subsumes #1115 (a re-woken low-index task can't starve a
        // long-Runnable high-index one).
        let mut last_pick: usize = 0;
        loop {
            // #1122 — the external-wake generation, snapshotted FIRST, before any of this settle's
            // reads: the all-parked block below waits for a ring newer than this, so every fact an
            // embedder door raises and then rings for — bytes fed to a pipe, a deliverable signal,
            // a child transition, a kill — is either seen by the settle or wakes the block. Taken
            // after the kill sweep, a terminate landing between the two had its ring folded into
            // the snapshot and the pump slept through a dead domain until the next unrelated ring.
            // `0` when unarmed.
            let bell_gen = host
                .external_wake()
                .map_or(0, |bell| *bell.0.lock().unwrap_or_else(|e| e.into_inner()));
            // Domain lifetime & teardown (DESIGN.md §12 / ISSUES.md I37, owner 2026-07-24): a
            // member's trap/exit is terminal for its whole DOMAIN — run the teardown fixpoint
            // before reading the root's state, so a sibling's trap that killed the root domain
            // surfaces as the run's result (previously the root's timed wait simply outlived it).
            teardown_domains(tasks, extra_envs, dead_envs);
            // The root's result is the run's result (other vCPUs' effects are already reflected in it).
            if let TaskState::Done(res) = &tasks[0].state {
                let res = res.clone();
                // Freeze driver (DURABILITY.md §12.8 slice 3.1.4): a durable run left in `UNWINDING` has
                // drained the root's native stack into context 0's region; now flatten the still-parked
                // fibers into theirs, while the registry is alive, before the window is snapshotted. A drive
                // trap (out-of-scope fiber) surfaces as the run's result. `cont.*` durability is single-vCPU
                // (the entry guard refuses `thread.*`), so only the root task owns fibers.
                if res.is_ok()
                    && host.is_durable()
                    && mem.as_ref().map(|m| m.durable_state()) == Some(super::STATE_UNWINDING)
                {
                    let mut ctx = RunCtx {
                        table: &dom.table,
                        fuel: &mut *fuel,
                        mem: &mut *mem,
                        durable: true,
                        host: HostCell::Excl(&mut *host),
                    };
                    // Only the root domain's registry is flattened: a child's live fibers would be
                    // lost, so they refuse the freeze (fail closed) rather than vanish.
                    if extra_envs.iter().any(|e| {
                        e.fibers
                            .fibers
                            .iter()
                            .any(|f| !matches!(f, FiberState::Done))
                    }) {
                        return Err(Trap::FiberFault);
                    }
                    host.frozen_fibers =
                        freeze_drive(fibers, fiber_sp, fiber_meta, dom, &mut ctx, budget)?;
                }
                // `res` is the root's `Result<Vec<Value>, Trap>`: `Ok(vals)` → `Done(vals)`; a root
                // trap stays `Err(trap)` (the run's fatal trap), exactly as `drive` returned it.
                return res.map(CoopStep::Done);
            }
            // §3.6 (I36 slice 2) — settle wakes: a task parked on a live-call ticket wakes when the
            // callee's serve loop completed its dispatch; claiming the completion cell delivers the
            // reply (the tree-walker's cap_reply preference — a parked caller beats the cell).
            for t in tasks.iter_mut() {
                let hit = match &t.state {
                    TaskState::BlockedTicket {
                        ticket,
                        callee,
                        dst,
                    } => callee
                        .lock_unpoisoned()
                        .svc_results
                        .remove(ticket)
                        .map(|v| (v, *dst)),
                    _ => None,
                };
                if let Some((v, dst)) = hit {
                    t.vt.active.set(dst, Reg::from_i64(v));
                    t.state = TaskState::Runnable;
                }
            }
            // #1215 — the default-action TERMINATE, cooperative form (invariant 14). A domain the
            // personality has terminated (a `SIG_DFL` SIGKILL/SIGTERM/SIGINT delivered through the gate,
            // `term_sig` set) must DIE. The tree-walker benches every vCPU of the domain at its per-op
            // `term_flag` safepoint and traps it; the cooperative driver has no per-op poll, so finalize
            // each task of a killed domain HERE — running, stopped (#1198-benched), or parked — as a
            // fatal completion. Because the driver is single-threaded round-robin, a killed task is never
            // mid-step when its signaller ran, so finalizing at the loop top loses no work; and it runs
            // before the exit-hook step below, so the twin retires (WIFSIGNALED via `term_sig`) and the
            // signaller's `waitpid` reaps it in the same settle. Domain-scoped (invariant 12).
            let killed: Vec<usize> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ti2, t)| {
                    if matches!(t.state, TaskState::Done(_)) {
                        return None;
                    }
                    let dead = match t.env {
                        Some(k) => extra_envs[k]
                            .host
                            .lock_unpoisoned()
                            .signal_poll()
                            .is_some_and(|(_, s)| s.killed()),
                        None => host.signal_poll().is_some_and(|(_, s)| s.killed()),
                    };
                    dead.then_some(ti2)
                })
                .collect();
            for ti2 in killed {
                complete(tasks, ti2, Err(Trap::ThreadFault));
            }
            // FORK.md §8.6 / #1807 — a child domain that has finished (every task on its env done, cleanly
            // or not) releases its pipe ends once: the tree-walker's domain-finish `drop_all_pipe_*`. A
            // producer that exits lets its consumer see EOF; a consumer that exits (e.g. `head`) wakes a
            // parked producer to `-EPIPE`.
            let mut live = vec![false; extra_envs.len()];
            let mut seen = vec![false; extra_envs.len()];
            for t in tasks.iter() {
                if let Some(k) = t.env {
                    seen[k] = true;
                    live[k] |= !matches!(t.state, TaskState::Done(_));
                }
            }
            let mut finished: Vec<usize> = Vec::new();
            for k in 0..extra_envs.len() {
                if seen[k] && !live[k] && released_envs.insert(k) {
                    extra_envs[k].host.lock_unpoisoned().release_pipe_ends();
                    finished.push(k);
                }
            }
            // #799/#1080 — a **fork twin** finishing fires its personality exit hooks ONCE (Live →
            // Zombie in the process table), the bytecode port of the tree-walker's death-hook step. It
            // runs BEFORE the reap wakes below so a personality `waitpid` re-execution finds the twin
            // already retired. Gated on the twin registry + a not-yet-hooked marker; a twin is always
            // its own env, so its host is `extra_envs[k]`. Two-phase (gather host + status, then fire).
            let to_hook: Vec<(usize, i64, usize)> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ti2, t)| match (&t.state, t.env) {
                    (TaskState::Done(res), Some(k))
                        if forked_twins.contains(&ti2) && !hooked_twins.contains(&ti2) =>
                    {
                        Some((ti2, super::reap_status(res), k))
                    }
                    _ => None,
                })
                .collect();
            for (ti2, status, k) in to_hook {
                // Its pipe ends were released above, with every finished domain's.
                let hooks = extra_envs[k].host.lock_unpoisoned().exit_hooks.clone();
                for h in hooks {
                    h(status);
                }
                hooked_twins.insert(ti2);
            }
            // A finished domain gives back what it held — its window, its powerbox, its frames — now
            // that its pipe ends are released and its exit hooks have fired. What a reaper reads is its
            // tasks' `Done` results, which stay. Without this a run holds every process it ever ran: a
            // build that forks and execs a compiler per module grew by a window per process.
            for k in finished {
                let env = &mut extra_envs[k];
                env.mem = None;
                env.host = std::sync::Arc::new(std::sync::Mutex::new(Host::new()));
                env.fibers = FiberTables::default();
                for t in tasks.iter_mut().filter(|t| t.env == Some(k)) {
                    t.vt.release();
                    t.suspended = None;
                }
            }
            // FORK.md §9.2 — reap wakes: a caller parked in `wait(pid)` wakes when fork twin `pid`
            // finishes, with the twin's exit status ([`super::reap_status`]; a trapped twin reaps as a
            // crash status, never a propagated trap — reap ≠ join). Two-phase (read the twin's outcome,
            // then deliver) so the caller and the twin task are not borrowed at once.
            let reap_wakes: Vec<(usize, u32, i64, usize)> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ci, t)| match &t.state {
                    TaskState::BlockedReap { pid, dst } => match &tasks[*pid].state {
                        TaskState::Done(res) => Some((ci, *dst, super::reap_status(res), *pid)),
                        _ => None,
                    },
                    _ => None,
                })
                .collect();
            for (ci, dst, status, pid) in reap_wakes {
                tasks[ci].vt.active.set(dst, Reg::from_i64(status));
                tasks[ci].state = TaskState::Runnable;
                forked_twins.remove(&pid);
            }
            // #799/#1080 — personality `waitpid` wakes: a task parked in [`TaskState::BlockedReapPersonality`]
            // re-admits (its rewound `waitpid` re-executes) once the named child completes — `Some(pid)`
            // that specific twin, `None` any forked child. No status is delivered here: the re-executed op
            // asks the personality, which serves the exit of the twin its exit hooks (fired above) retired.
            let reap_p_wakes: Vec<usize> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ci, t)| match &t.state {
                    TaskState::BlockedReapPersonality { child } => {
                        let done = match child {
                            // Personality pid → task index is `pid - 1` (the fork convention above).
                            Some(pid) => matches!(
                                pid.checked_sub(1)
                                    .and_then(|i| tasks.get(i))
                                    .map(|c| &c.state),
                                Some(TaskState::Done(_))
                            ),
                            None => tasks.iter().enumerate().any(|(j, c)| {
                                forked_twins.contains(&j) && matches!(c.state, TaskState::Done(_))
                            }),
                        };
                        done.then_some(ci)
                    }
                    _ => None,
                })
                .collect();
            for ci in reap_p_wakes {
                tasks[ci].state = TaskState::Runnable;
            }
            // #1080 rung 4 — pipe wakes: the cooperative driver has no scheduler-side `pipe_waiters`, so
            // it POLLS each parked reader/writer's shared FIFO here and re-admits (the rewound read/write
            // re-executes) once ready. A reader/writer is usually a forked command (`env: Some`); a root
            // task (`env: None`, a bash builtin over a pipe) checks the driver host.
            let pipe_wakes: Vec<usize> = tasks
                .iter()
                .enumerate()
                .filter_map(|(ci, t)| {
                    // #1171 — a STOPPED reader/writer is not re-admitted by an input/room wake: a
                    // stopped process makes no progress, so a suspended foreground `cat` must not steal
                    // the bytes the shell should read (it re-admits at its `SIGCONT`). Domain-scoped.
                    let stopped = match t.env {
                        Some(k) => extra_envs[k]
                            .host
                            .lock_unpoisoned()
                            .signal_poll()
                            .is_some_and(|(_, s)| s.stopped()),
                        None => host.signal_poll().is_some_and(|(_, s)| s.stopped()),
                    };
                    if stopped {
                        return None;
                    }
                    let ready = match &t.state {
                        TaskState::BlockedPipeRead { pipe } => match t.env {
                            Some(k) => extra_envs[k].host.lock_unpoisoned().pipe_read_ready(*pipe),
                            None => host.pipe_read_ready(*pipe),
                        },
                        TaskState::BlockedPipeWrite { pipe } => match t.env {
                            Some(k) => extra_envs[k].host.lock_unpoisoned().pipe_write_ready(*pipe),
                            None => host.pipe_write_ready(*pipe),
                        },
                        // #1146 (deeper) — a blocking stdin park re-admits once its own host's stdin
                        // buffer has bytes (the stdin twin of the pipe poll; domain-scoped like the rest).
                        TaskState::BlockedStdin => match t.env {
                            Some(k) => extra_envs[k].host.lock_unpoisoned().stdin_ready(),
                            None => host.stdin_ready(),
                        },
                        _ => return None,
                    };
                    ready.then_some(ci)
                })
                .collect();
            for ci in pipe_wakes {
                tasks[ci].state = TaskState::Runnable;
            }
            refund_ended_windows(tasks, host, extra_envs);
            // I48 — wake blocking-resume idlers: a `TaskState::BlockedOnFiber { fiber }` becomes
            // runnable once its fiber is woken (the idle-timer's `WAIT_TIMED_OUT`, a `notify`'s
            // `WAIT_WOKEN`, or the cap-completion drain). Its cursor was rewound to the resume op, so
            // the next step re-executes it and claims the now-woken fiber. Centralized here so every
            // wake source feeds it uniformly (no per-site wiring). Runs before the pick so a fiber
            // woken during the previous step is seen this iteration.
            for t in tasks.iter_mut() {
                if let TaskState::BlockedOnFiber { fiber } = t.state {
                    let reg = match t.env {
                        None => &*fibers,
                        Some(k) => &extra_envs[k].fibers.fibers,
                    };
                    let woken = match reg.get(fiber) {
                        Some(
                            FiberState::WaitParked { woken: Some(_), .. }
                            | FiberState::CapParked { woken: Some(_), .. },
                        ) => true,
                        Some(FiberState::HostParked { on, .. }) => {
                            task_host(host, extra_envs, t.env).with(|h| on.ready(h))
                        }
                        _ => false,
                    };
                    if woken {
                        t.state = TaskState::Runnable;
                    }
                }
            }
            // #1198 — a Runnable task whose DOMAIN is stopped (SIGTSTP/SIGTTIN/SIGTTOU, before its
            // SIGCONT) must not be stepped: a stopped process makes no progress. The tree-walker won't
            // run a stopped process; the coop pick must skip it too, or a background job whose read
            // returns `-ERESTART` on the SIGTTIN keeps re-issuing (its libc retries) and spins forever
            // instead of benching. It re-runs when SIGCONT clears the stop. Domain-scoped (invariant 12).
            // #1904 — the oracle's see-through (#1672): under a landing freeze a stopped domain runs to
            // its next freeze point, and `Op::CapCall` abandons each host call on the way.
            let see_through = host.is_durable() && is_unwinding(mem);
            let domain_stopped = |i: usize| -> bool {
                !see_through
                    && match tasks[i].env {
                        Some(k) => extra_envs[k]
                            .host
                            .lock_unpoisoned()
                            .signal_poll()
                            .is_some_and(|(_, s)| s.stopped()),
                        None => host.signal_poll().is_some_and(|(_, s)| s.stopped()),
                    }
            };
            // #1157 — round-robin from `last_pick + 1` (wrapping) rather than lowest-index-first.
            let n = tasks.len();
            let Some(ti) = (1..=n)
                .map(|k| (last_pick + k) % n)
                .find(|&i| matches!(tasks[i].state, TaskState::Runnable) && !domain_stopped(i))
            else {
                // #1904 — bring the parks through a freeze before any timer fires or the run is
                // called deadlocked.
                if freeze_step(tasks, fibers, forked_twins, mem, host, freeze_on_quiesce) {
                    continue;
                }
                // F2 (FIBER_PARK.md) — no runnable task with punt completions outstanding: that is
                // pending work on the offload pool, never a deadlock and never a reason to jump the
                // logical clock. Block on the store for the smallest outstanding id (submission
                // order), deliver through the ordered drain, and loop — the woken fibers'
                // resumers observe the wake at their next poll (or their own timers fire below on
                // a later pass).
                let min_cap = fibers
                    .iter()
                    .filter_map(|f| match f {
                        FiberState::CapParked {
                            id, woken: None, ..
                        } => Some(*id),
                        _ => None,
                    })
                    .min();
                if let Some(id) = min_cap {
                    let comps = host.completions();
                    let r = comps.wait(id);
                    for f in fibers.iter_mut() {
                        if let FiberState::CapParked {
                            id: fid,
                            woken: w @ None,
                            ..
                        } = f
                        {
                            if *fid == id {
                                *w = Some(r);
                            }
                        }
                    }
                    drain_cap_parked(fibers, &comps);
                    continue;
                }
                // No runnable task: fire the earliest `wait` timeout — whole-vCPU waiters and
                // event-parked fiber waiters alike (§3.6 slice 5a) — else it is a deadlock.
                //
                // #1638 — only a waiter with a REAL deadline is a candidate. An infinite wait
                // carries `None` and is dropped here, so a run whose every remaining waiter is
                // indefinite falls through to the `None` arm and reaches the deadlock this
                // function's own doc already promised ("or deadlocks → `ThreadFault`, matching
                // the deterministic explorer"). That path was unreachable while the `MAX_WAIT`
                // clamp gave every infinite wait a deadline to advance to.
                let next = tasks
                    .iter()
                    .filter_map(|t| match t.state {
                        TaskState::BlockedWait { deadline, .. } => deadline,
                        _ => None,
                    })
                    .chain(
                        fibers
                            .iter()
                            .chain(extra_envs.iter().flat_map(|e| e.fibers.fibers.iter()))
                            .filter_map(|f| match f {
                                FiberState::WaitParked {
                                    deadline,
                                    woken: None,
                                    ..
                                } => *deadline,
                                _ => None,
                            }),
                    )
                    .min();
                match next {
                    Some(d) => {
                        *clock = (*clock).max(d);
                        for t in tasks.iter_mut() {
                            if let TaskState::BlockedWait {
                                deadline: Some(deadline),
                                dst,
                                ..
                            } = t.state
                            {
                                if deadline <= *clock {
                                    t.vt.active.set(dst, Reg::from_i32(super::WAIT_TIMED_OUT));
                                    t.state = TaskState::Runnable;
                                }
                            }
                        }
                        // §3.6 slice 5a: a due fiber wait completes with `WAIT_TIMED_OUT` — the
                        // fiber becomes claimable (leaving the pending set, so this loop makes
                        // progress); its resumer's next `cont.resume` delivers the status. Every
                        // domain's registry: a child's fibers wait on the same clock.
                        for f in fibers.iter_mut().chain(
                            extra_envs
                                .iter_mut()
                                .flat_map(|e| e.fibers.fibers.iter_mut()),
                        ) {
                            if let FiberState::WaitParked {
                                deadline: Some(deadline),
                                woken: w @ None,
                                ..
                            } = f
                            {
                                if *deadline <= *clock {
                                    *w = Some(super::WAIT_TIMED_OUT);
                                }
                            }
                        }
                    }
                    None => {
                        // #1146 slice 2 — before blocking, interrupt the parks if a deliverable
                        // signal reached this all-parked run (e.g. a `^C` the terminal line
                        // discipline raised, which rang the doorbell but deposited no bytes, so the
                        // readiness poll above found nothing runnable). Set each pipe-parked task's
                        // host EINTR flag and re-admit it: the rewound read/write re-runs and
                        // completes `-EINTR` at the park site, and the caught handler is delivered
                        // at that task's next safepoint (slice 1). The tree-walker drives this from
                        // its `set_wake` closure; the cooperative pump polls it here, at the would-be
                        // block. `interrupt_pending` is a non-consuming peek — delivery still fires.
                        // #1171 — DOMAIN-SCOPED (invariant 12): a parked task is interrupted only when
                        // ITS OWN domain has a deliverable signal pending, never because some other
                        // domain does. Before this a single root-host pending signal swept EVERY
                        // pipe-parked task across all domains — so a shell's `SIGCHLD` (raised when a
                        // foreground job stopped) wrongly `-EINTR`'d that job's own blocked read,
                        // running it off the end instead of leaving it stopped (the browser `^Z` gap).
                        // A pipe read/write re-runs `-EINTR`; a personality `waitpid` (BlockedReap-
                        // Personality) re-runs and serves whatever its table now reports — a fresh
                        // `WUNTRACED` stop, an exit, or re-parks if still nothing — so the shell's
                        // `SIGCHLD` on a child's stop/continue transition wakes its blocked `waitpid`,
                        // matching the tree-walker (whose `Blocked::Stopped` insert drains the reap
                        // waiters). The caught handler itself is delivered at that task's next safepoint.
                        let mut woke = false;
                        for t in tasks.iter_mut() {
                            // A pipe read/write OR a blocking stdin read (#1146 deeper) — the interruptible
                            // blocking-I/O parks whose rewound op completes `-EINTR` on a signal.
                            let is_pipe = matches!(
                                t.state,
                                TaskState::BlockedPipeRead { .. }
                                    | TaskState::BlockedPipeWrite { .. }
                                    | TaskState::BlockedStdin
                            );
                            let is_reap =
                                matches!(t.state, TaskState::BlockedReapPersonality { .. });
                            if !is_pipe && !is_reap {
                                continue;
                            }
                            // A pipe/reap park is interrupted by a deliverable (async) signal on its
                            // OWN domain. A **reap** park is ALSO re-admitted by the one-shot
                            // child-transition edge (#1171 `reap_pending`, read-and-clear) — so a
                            // blocking `waitpid(WUNTRACED/WCONTINUED)` wakes when a child stops/continues
                            // even with no async SIGCHLD delivery (bash: no sigaltstack). The re-run
                            // `waitpid` reports the fresh stop/continue (report-once) and returns.
                            let interrupt = match t.env {
                                Some(k) => extra_envs[k]
                                    .host
                                    .lock_unpoisoned()
                                    .signal_poll()
                                    .is_some_and(|(_, s)| s.interrupt_pending()),
                                None => host
                                    .signal_poll()
                                    .is_some_and(|(_, s)| s.interrupt_pending()),
                            };
                            let reap = is_reap
                                && match t.env {
                                    Some(k) => extra_envs[k]
                                        .host
                                        .lock_unpoisoned()
                                        .signal_poll()
                                        .is_some_and(|(_, s)| s.reap_pending()),
                                    None => {
                                        host.signal_poll().is_some_and(|(_, s)| s.reap_pending())
                                    }
                                };
                            if !interrupt && !reap {
                                continue;
                            }
                            // Only a pipe park interrupted by a signal needs the EINTR latch (its
                            // rewound read/write completes `-EINTR`); a re-run `waitpid` re-consults the
                            // personality with no flag.
                            if is_pipe && interrupt {
                                match t.env {
                                    Some(k) => {
                                        extra_envs[k].host.lock_unpoisoned().set_sig_interrupt()
                                    }
                                    None => host.set_sig_interrupt(),
                                }
                            }
                            t.state = TaskState::Runnable;
                            woke = true;
                        }
                        if woke {
                            continue;
                        }
                        // #1122 — every task is parked and no internal wake can come. With an
                        // armed doorbell and at least one task parked on a PIPE — the state an
                        // embedder can feed from outside the run (the interactive terminal) —
                        // BLOCK for the next ring instead of declaring deadlock, then re-settle
                        // (the readiness poll above sees the deposit). The `bell_gen` snapshot
                        // predates that poll, so a feed racing this park is never lost. Without
                        // a doorbell, or with only internally-wakeable parks (a join cycle),
                        // this is the deadlock it always was.
                        let external = tasks.iter().any(|t| {
                            matches!(
                                t.state,
                                TaskState::BlockedPipeRead { .. }
                                    | TaskState::BlockedPipeWrite { .. }
                                    // #1146 (deeper) — a blocking stdin park is externally wakeable
                                    // too (an embedder signal rings the bell); without this an
                                    // all-parked stdin run would fault as a deadlock instead of
                                    // blocking for the `^C`.
                                    | TaskState::BlockedStdin
                            )
                        });
                        // #1122 route (a) — a suspend/resume session: hand the idle state back to the
                        // driver (it feeds the terminal and pumps again; the loop-top settle then sees
                        // the deposit / the signal) instead of sleeping this thread on the bell.
                        if *suspend_on_idle && external {
                            return Ok(CoopStep::Idle);
                        }
                        match host.external_wake().filter(|_| external) {
                            Some(bell) => {
                                let (gen, cv) = &*bell;
                                let mut g = gen.lock().unwrap_or_else(|e| e.into_inner());
                                while *g == bell_gen {
                                    g = cv.wait(g).unwrap_or_else(|e| e.into_inner());
                                }
                            }
                            None => return Err(Trap::ThreadFault), // deadlock (no runnable, no waiters)
                        }
                    }
                }
                continue;
            };
            last_pick = ti;
            // #1157 — arm the preemption quantum ONLY when ≥2 tasks are runnable (genuinely concurrent).
            // A single runnable task (the common case — bash, every non-threaded browser guest) keeps
            // `budget = u64::MAX` / run-to-completion, so the hot path takes zero extra pump round-trips
            // and its interleaving is unchanged. With a concurrent sibling, an op-count quantum bounds a
            // yield-free spinner so the sibling gets the thread — deterministically (op-count, not
            // wall-clock). `COOP_QUANTUM` is coarse (~1M ops) to keep interleaving close to the old
            // run-to-completion order while still bounding a spin to sub-second.
            let (quantum, preemptible) = if tasks
                .iter()
                .filter(|t| matches!(t.state, TaskState::Runnable))
                .count()
                >= 2
            {
                (COOP_QUANTUM, true)
            } else {
                (budget, false)
            };
            // A sliced pump caps the quantum at what is left of the slice, preemptibly, so the task
            // yields at the slice's end even when it runs alone.
            let (quantum, preemptible) = match *slice_left {
                Some(left) => (quantum.min(left.max(1)), true),
                None => (quantum, preemptible),
            };

            // Select this vCPU's environment and fiber registry: the shared ones (root + thread
            // siblings), or its own confined `instantiate` env's. `tasks[ti].vt` and the chosen env
            // borrow disjoint storage (`tasks` vs `extra_envs` / the `mem`/`host`/`fuel` params), so the
            // split borrow is sound.
            let (mut ctx, mut fcell) = match tasks[ti].env {
                None => (
                    RunCtx {
                        table: &dom.table,
                        fuel: &mut *fuel,
                        mem: &mut *mem,
                        durable: host.is_durable(),
                        host: HostCell::Excl(&mut *host),
                    },
                    FiberCell::Excl {
                        fibers: &mut *fibers,
                        sp: &mut *fiber_sp,
                        meta: &mut *fiber_meta,
                    },
                ),
                Some(k) => {
                    let e = &mut extra_envs[k];
                    let durable = e
                        .host
                        .lock()
                        .unwrap_or_else(|er| er.into_inner())
                        .is_durable();
                    (
                        RunCtx {
                            table: &e.table,
                            fuel: &mut e.fuel,
                            mem: &mut e.mem,
                            durable,
                            host: HostCell::Shared(&e.host),
                        },
                        FiberCell::Excl {
                            fibers: &mut e.fibers.fibers,
                            sp: &mut e.fibers.sp,
                            meta: &mut e.fibers.meta,
                        },
                    )
                }
            };
            let fuel_before = *ctx.fuel;
            // Fuel is charged once per op, so what it dropped by is the ops this step ran.
            let charge_slice = |slice_left: &mut Option<u64>, fuel_after: u64| {
                if let Some(left) = slice_left.as_mut() {
                    *left = left.saturating_sub(fuel_before.saturating_sub(fuel_after));
                }
            };
            // #1896 — a leaf's call that parked in a bounce: run the rest of it in the task's env.
            // When it returns, its host resumes the leaf's suspended frames with its results; when
            // it parks again, the task waits again. A trap ends the process, whose frames the host
            // never resumes.
            if let Some(s) = tasks[ti].suspended.take() {
                let Suspended { vm, dst, results } = *s;
                let mut parked = None;
                let meta = BounceRunCtx {
                    jit_mirror: None, // a leaf's env is its own
                    park: Some(&mut parked),
                };
                let done = drive_nested(
                    &dom.source,
                    ctx.table,
                    vm,
                    ctx.fuel,
                    ctx.mem,
                    &mut ctx.host,
                    &mut fcell,
                    Some(meta),
                    None,
                );
                charge_slice(slice_left, *ctx.fuel);
                match (done, parked) {
                    (Err(trap), _) => complete(tasks, ti, Err(trap)),
                    (Ok(_), Some((vm, state))) => {
                        tasks[ti].state = state;
                        tasks[ti].suspended = Some(Box::new(Suspended { vm, dst, results }));
                    }
                    (Ok(vals), None) => {
                        *pending_tierup = Some((ti, dst, results));
                        let results = vals.iter().map(|v| val_to_slot(*v)).collect();
                        return Ok(CoopStep::Resume { results });
                    }
                }
                continue;
            }
            let stop = step_vcpu(
                &mut tasks[ti].vt,
                &mut fcell,
                dom,
                &mut ctx,
                quantum,
                true, // the cooperative scheduler: idle blocking `cont.resume.block` (I48)
                preemptible, // #1157: yield at the op-count quantum when ≥2 tasks are runnable
            );
            charge_slice(slice_left, *ctx.fuel);
            match stop {
                Err(trap) => {
                    // #1720 — the **trap-origin** fault address (the tree-walker's rule): a child's
                    // trap re-raises at its joiner, whose window is not the one that faulted, so the
                    // run's `last_capture_fault_addr` must be read here, from the trapping task's own.
                    // The first fault recorded wins; the run entry clears the slot beforehand.
                    if trap == Trap::MemoryFault {
                        if let Some(a) = ctx.mem.as_ref().and_then(|m| m.peek_fault_rel()) {
                            super::LAST_CAPTURE_FAULT.with(|c| _ = c.borrow_mut().get_or_insert(a));
                        }
                    }
                    complete(tasks, ti, Err(trap))
                }
                // #1157 — the quantum expired: the task is still `Runnable` (its cursor persisted), so
                // just loop. The round-robin `last_pick` advance picks a sibling next, giving it the
                // thread; this task resumes on a later turn.
                // A sliced pump whose slice is spent returns here instead, to its embedder.
                Ok(VcpuStop::Preempted) => {
                    if *slice_left == Some(0) {
                        return Ok(CoopStep::Paused);
                    }
                }
                Ok(VcpuStop::Done(vals)) => complete(tasks, ti, Ok(vals)),
                // #926 slice 2 — wasm-JIT tier-up: this module-0 task hit a direct call to an eligible
                // function (its `Vm` carries the run's bitmap). `step_vcpu` has already spilled the frame
                // past the call, so the task is resumable with just the result slots filled. Stash the
                // delivery target — `(task, dst, result types)` — and surface the region to the driver;
                // `deliver_tierup` writes the emitted results back into `tasks[ti].vt.active` and the next
                // `pump` resumes the task. At most one tier-up is outstanding (we return here), so the
                // single `pending_tierup` slot suffices — no per-task field needed. On the native driver
                // no task is eligible, so this arm is never reached (the bitmap is `None`).
                Ok(VcpuStop::TierUp {
                    func,
                    argv,
                    dst,
                    results,
                    mapped,
                }) => {
                    debug_assert!(
                        pending_tierup.is_none(),
                        "a tier-up is already outstanding — deliver_tierup was skipped"
                    );
                    *pending_tierup = Some((ti, TierUpDst::Frame(dst), results));
                    return Ok(CoopStep::TierUp {
                        module: 0,
                        func,
                        argv,
                        mapped,
                    });
                }
                // #1146 (deeper) — park this task on a blocking `Stream{In}` read (the op was rewound);
                // the settle scan re-admits it on `stdin_ready` and the all-parked signal sweep
                // interrupts it (the re-run completes `-EINTR`), exactly like a pipe park. Invariant 14:
                // the tree-walker's stdin park, carried to the cooperative driver.
                // #1952 — a fiber's pipe or stdin op that must wait parks the fiber alone, and its
                // resumer runs on (or idles on it, under a blocking resume). Fiber 0's park is its
                // task's (the arms below). A vanished pipe's op just re-runs, and fails closed.
                Ok(
                    stop @ (VcpuStop::PipeRead { .. }
                    | VcpuStop::PipeWrite { .. }
                    | VcpuStop::StdinPark),
                ) if tasks[ti].vt.active_id != ROOT_FIBER => {
                    host_park_fiber(tasks, ti, fibers, fiber_sp, mem, extra_envs, host, &stop);
                }
                Ok(VcpuStop::StdinPark) => {
                    tasks[ti].state = TaskState::BlockedStdin;
                }
                // I48 — a blocking `cont.resume.block` of a still-parked fiber: idle this task on the
                // fiber (`step_vcpu` already rewound the resumer's cursor to the resume op). The
                // top-of-loop scan re-marks it `Runnable` once the fiber wakes.
                Ok(VcpuStop::BlockOnFiber { fiber }) => {
                    tasks[ti].state = TaskState::BlockedOnFiber { fiber };
                }
                // §3.6 (I36 slice 2) — the serve/call/offer trio, cooperative form.
                Ok(VcpuStop::SvcWait) => {
                    tasks[ti].state = TaskState::BlockedSvc;
                }
                Ok(VcpuStop::LiveCall {
                    ticket,
                    callee,
                    dst,
                }) => {
                    // The enqueue already happened in the op exec (holding only the callee's lock).
                    // Wake any svc.wait-parked task of the callee's domain — the tree-walker's
                    // `svc_wake` — then park the caller on its ticket.
                    let k = extra_envs
                        .iter()
                        .position(|e| std::sync::Arc::ptr_eq(&e.host, &callee));
                    // §12 teardown / D37 death-is-revocation (owner 2026-07-24): a call through an
                    // already-torn-down callee can never be replied — complete with the probeable
                    // errno instead of parking forever (the tree-walker's dead-callee park probe).
                    if k.is_some_and(|k| dead_envs.contains(&k)) {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i64(super::CAP_REVOKED));
                        continue;
                    }
                    if let Some(k) = k {
                        for t in tasks.iter_mut() {
                            if t.env == Some(k) && matches!(t.state, TaskState::BlockedSvc) {
                                t.state = TaskState::Runnable;
                            }
                        }
                    }
                    tasks[ti].state = TaskState::BlockedTicket {
                        ticket,
                        callee,
                        dst,
                    };
                }
                Ok(VcpuStop::CapPending { id, dst }) => {
                    // F2 (FIBER_PARK.md) — a punted dispatch, cooperative form. A FIBER parks
                    // (`CapParked` — the slice-5a contract; the ordered drain right after the
                    // park is the register-then-recheck closing the completion-raced-the-park
                    // window). The root keeps the inline wait (guest-invisible — the oracle
                    // parks the vCPU here instead; the cooperative driver has nothing else to
                    // run on this task anyway). Two more inline cases, both mirroring the
                    // oracle's predicate: a durable run (`freeze_drive` has no cap-park
                    // re-derivation — the freeze must never meet one) and a confined
                    // `instantiate` child (its completions live on ITS host; keeping the child
                    // inline keeps the drain single-store — recorded FIBER_PARK.md residue).
                    let durable = host.is_durable();
                    if tasks[ti].vt.active_id != ROOT_FIBER && !durable && tasks[ti].env.is_none() {
                        let comps = host.completions();
                        let k = tasks[ti].vt.active_id;
                        // The drain right after the park is the register-then-recheck: a completion
                        // that raced the park wakes the fiber at once.
                        if let Some(k) = park_running_fiber(
                            &mut tasks[ti].vt,
                            fibers,
                            fiber_sp,
                            mem,
                            durable,
                            true,
                            |vm| FiberState::CapParked {
                                vm,
                                dst,
                                id,
                                woken: None,
                            },
                            |fibers| {
                                drain_cap_parked(fibers, &comps);
                                matches!(
                                    fibers.get(k),
                                    Some(FiberState::CapParked { woken: Some(_), .. })
                                )
                            },
                        ) {
                            tasks[ti].state = TaskState::BlockedOnFiber { fiber: k };
                        }
                    } else {
                        let comps = match tasks[ti].env {
                            None => host.completions(),
                            Some(k) => extra_envs[k].host.lock_unpoisoned().completions(),
                        };
                        // #1366: a host-completed punt has no completer on this driver — decline.
                        match comps.wait_unless_host_owned(id) {
                            Some(r) => tasks[ti].vt.active.set(dst, Reg::from_i64(r)),
                            None => complete(tasks, ti, Err(Trap::CapFault)),
                        }
                    }
                }
                Ok(VcpuStop::ChildOffer { child, export, dst }) => {
                    // Mint a live-callee offer over a running child's export: shape from the
                    // CALLEE's module (fetched before the wirer's lock — the tree-walker's lock
                    // order), interned structurally into the wirer's table. A bad child handle /
                    // no such export is a probeable -EINVAL, matching the oracle.
                    let callee = usize::try_from(child)
                        .ok()
                        .and_then(|h| tasks[ti].threads.get(h).copied().flatten())
                        .and_then(|cidx| tasks[cidx].env)
                        .map(|k| std::sync::Arc::clone(&extra_envs[k].host));
                    let cap = callee.and_then(|callee: std::sync::Arc<std::sync::Mutex<Host>>| {
                        let (names, sigs) = callee.lock_unpoisoned().offer_shape(export)?;
                        match tasks[ti].env {
                            None => host.wire_live_impl(&callee, export, &names, &sigs).ok(),
                            Some(pk) => extra_envs[pk]
                                .host
                                .lock_unpoisoned()
                                .wire_live_impl(&callee, export, &names, &sigs)
                                .ok(),
                        }
                    });
                    tasks[ti]
                        .vt
                        .active
                        .set(dst, Reg::from_i32(cap.unwrap_or(super::EINVAL as i32)));
                }
                Ok(VcpuStop::CloneCaller {
                    reply_orig,
                    reply_twin,
                    dst,
                    has_result,
                }) => {
                    // FORK.md §9.2 — fork-returns-twice on the cooperative driver. Duplicate the caller
                    // parked on this handler's dispatch into a live **twin** (private window +
                    // duplicated powerbox, its own env), deliver `reply_twin` to the twin and
                    // `reply_orig` (pid mode: the twin's task id) to the original; both resume past the
                    // same fork `call.cap`. Fail-closed to a single reply on any shape the driver can't
                    // duplicate — never a hang, mirroring the oracle's degrade (temen-interp
                    // `fork_parked_caller`). `reap` (fork+wait) is a later slice — such modules fold.
                    let result: i64 = 'fork: {
                        // The running handler's dispatch ticket names the parked caller; outside a
                        // handler there is none → `-EINVAL`, exactly as the oracle.
                        let Some(ticket) = tasks[ti].vt.active.serve_ticket else {
                            break 'fork super::EINVAL;
                        };
                        // The server (this task) is the callee the caller parked on. In the fork
                        // topology the server is a spawned child with an `Arc` host (never the root).
                        let Some(server_env) = tasks[ti].env else {
                            break 'fork super::EINVAL;
                        };
                        let server_host = std::sync::Arc::clone(&extra_envs[server_env].host);
                        // Locate the parked caller on `(ticket, this server)`. In the cooperative driver
                        // the caller has already parked (it enqueued + woke us before we ran), so a miss
                        // is defensive → degrade.
                        let caller_ti = tasks.iter().position(|t| {
                            matches!(&t.state,
                            TaskState::BlockedTicket { ticket: tk, callee, .. }
                                if *tk == ticket && std::sync::Arc::ptr_eq(callee, &server_host))
                        });
                        let degrade = |tasks: &mut Vec<TaskSlot>,
                                       caller_ti: Option<usize>|
                         -> i64 {
                            // One reply to the caller, no twin. Explicit mode delivers `reply_orig`; pid
                            // mode delivers `-EAGAIN` (POSIX fork failure). Returns the handler's result.
                            let fallback = reply_orig.unwrap_or(super::EAGAIN);
                            if let Some(cti) = caller_ti {
                                if let TaskState::BlockedTicket { dst: cdst, .. } = tasks[cti].state
                                {
                                    tasks[cti].vt.active.set(cdst, Reg::from_i64(fallback));
                                    tasks[cti].state = TaskState::Runnable;
                                }
                            }
                            reply_orig.map_or(super::EAGAIN, |_| 0)
                        };
                        let Some(caller_ti) = caller_ti else {
                            break 'fork degrade(tasks, None);
                        };
                        let TaskState::BlockedTicket {
                            dst: caller_dst, ..
                        } = tasks[caller_ti].state
                        else {
                            break 'fork super::EINVAL;
                        };
                        let caller_env = tasks[caller_ti].env;
                        // Only a bare root caller (no spawned children/threads, no live fiber chain)
                        // forks faithfully — the oracle's `bare` gate. Anything else degrades.
                        let bare = tasks[caller_ti].threads.iter().all(|t| t.is_none())
                            && tasks[caller_ti].vt.active_id == ROOT_FIBER
                            && tasks[caller_ti].vt.chain.is_empty();
                        // Duplicate the caller's window (private copy — fork does not share memory) and
                        // powerbox (own handle namespace, shared `Arc` backings). A root caller (no env)
                        // or a non-forkable window/powerbox fails closed to a single reply.
                        // The twin's pid is its task index (`twin_ti` below); nothing is pushed between
                        // here and that push, so the fork factories learn it up front (#863 slice 2).
                        let twin_pid = tasks.len() as u64;
                        let forked = if bare {
                            caller_env.and_then(|ck| {
                                let twin_mem = match &extra_envs[ck].mem {
                                    Some(m) => Some(m.fork_private()?),
                                    None => None,
                                };
                                let twin_host = extra_envs[ck]
                                    .host
                                    .lock_unpoisoned()
                                    .fork_powerbox(twin_pid)?;
                                Some((ck, twin_mem, twin_host))
                            })
                        } else {
                            None
                        };
                        let Some((ck, twin_mem, twin_host)) = forked else {
                            break 'fork degrade(tasks, Some(caller_ti));
                        };
                        // The twin's continuation is the caller's — a bare root `Vm` cloned at its
                        // post-call resume point (`Vm` derives `Clone`; a bare caller carries no resume
                        // chain / invoke) — with `reply_twin` injected at the caller's reply slot.
                        let mut twin_active = tasks[caller_ti].vt.active.clone();
                        twin_active.set(caller_dst, Reg::from_i64(reply_twin));
                        // #816 env-routed tier-up: same rule as the `ForkSelf` twin — the clone may
                        // keep an inherited bitmap only if its private `fork_private` window is
                        // servable (flat; the owned-flat twin backing makes the tier-up shapes so
                        // on every target); a `Paged` twin window strips it and interprets,
                        // fail-closed.
                        if !tierup_servable(twin_mem.as_ref(), mem.as_ref()) {
                            twin_active.jit_eligible = None;
                            twin_active.jit_page_checked = false;
                        }
                        let twin_root_sp =
                            twin_active.durable_region_base + super::REGION_HEADER_LEN; // its context's empty frame base
                        let twin_vt = VTask {
                            active: twin_active,
                            active_id: ROOT_FIBER,
                            chain: Vec::new(),
                            root_shadow_sp: twin_root_sp,
                            active_invoke: None,
                        };
                        // The twin is its own domain: a fresh env over the private window + duplicated
                        // powerbox, its own dispatch table seeded with the caller's installs (#1297),
                        // the caller's env fuel.
                        let twin_table = extra_envs[ck].table.fork();
                        let twin_eidx = extra_envs.len();
                        extra_envs.push(ChildEnv {
                            mem: twin_mem,
                            host: std::sync::Arc::new(std::sync::Mutex::new(twin_host)),
                            table: twin_table,
                            fuel: extra_envs[ck].fuel,
                            fibers: FiberTables::default(),
                        });
                        let twin_ti = tasks.len();
                        tasks.push(TaskSlot {
                            vt: twin_vt,
                            threads: Vec::new(),
                            env: Some(twin_eidx),
                            state: TaskState::Runnable,
                            suspended: None,
                            lease: None,
                        });
                        // Mark the twin reapable so a later servicer-side `wait()` (`reap`) can deliver
                        // its exit status to the parent (FORK.md §8.6); retired when reaped.
                        forked_twins.insert(twin_ti);
                        // Deliver the original's reply and re-run it: explicit `reply_orig`, or pid mode
                        // = the twin's task id (parent-sees-pid). The handler's own return still writes
                        // the ticket's completion cell, but the caller is now `Runnable` (not
                        // `BlockedTicket`), so the settle scan never claims it — harmless, no flag needed.
                        let orig_reply = reply_orig.unwrap_or(twin_ti as i64);
                        tasks[caller_ti]
                            .vt
                            .active
                            .set(caller_dst, Reg::from_i64(orig_reply));
                        tasks[caller_ti].state = TaskState::Runnable;
                        twin_ti as i64
                    };
                    if has_result {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(result));
                    }
                }
                Ok(VcpuStop::Reap {
                    pid,
                    dst,
                    has_result,
                }) => {
                    // FORK.md §9.2 — the servicer side of `wait(pid)`. Reap fork twin `pid` on behalf of
                    // the caller parked on this handler's dispatch: deliver the twin's exit status now (if
                    // it finished) or park the caller until it does. `-ECHILD` for a pid this run did not
                    // mint (the handler's own reply carries it); `-EINVAL` outside a handler. Never a hang.
                    let result: i64 = 'reap: {
                        let Some(ticket) = tasks[ti].vt.active.serve_ticket else {
                            break 'reap super::EINVAL;
                        };
                        let Some(server_env) = tasks[ti].env else {
                            break 'reap super::EINVAL;
                        };
                        let server_host = std::sync::Arc::clone(&extra_envs[server_env].host);
                        let caller_ti = tasks.iter().position(|t| {
                            matches!(&t.state,
                            TaskState::BlockedTicket { ticket: tk, callee, .. }
                                if *tk == ticket && std::sync::Arc::ptr_eq(callee, &server_host))
                        });
                        // The pid must be a twin this run minted; otherwise a genuine `-ECHILD`, which the
                        // handler's own return delivers to the still-parked caller (the normal serve path).
                        let Some(pid_us) = usize::try_from(pid)
                            .ok()
                            .filter(|p| forked_twins.contains(p))
                        else {
                            break 'reap super::ECHILD;
                        };
                        // The cooperative driver parks the caller before the handler runs, so a miss is
                        // defensive → `-EAGAIN` (retryable, never a false `-ECHILD` — the twin is real).
                        let Some(caller_ti) = caller_ti else {
                            break 'reap super::EAGAIN;
                        };
                        let TaskState::BlockedTicket {
                            dst: caller_dst, ..
                        } = tasks[caller_ti].state
                        else {
                            break 'reap super::EINVAL;
                        };
                        // Twin finished → deliver its status now and retire it; else park the caller on it
                        // (the settle scan wakes it on twin-exit). Either way the caller's reply is handled
                        // here — the handler's own return lands on a caller no longer `BlockedTicket`, so
                        // the settle scan never claims it (harmless, mirroring `clone_caller`).
                        if let TaskState::Done(res) = &tasks[pid_us].state {
                            let status = super::reap_status(res);
                            forked_twins.remove(&pid_us);
                            tasks[caller_ti]
                                .vt
                                .active
                                .set(caller_dst, Reg::from_i64(status));
                            tasks[caller_ti].state = TaskState::Runnable;
                            status
                        } else {
                            tasks[caller_ti].state = TaskState::BlockedReap {
                                pid: pid_us,
                                dst: caller_dst,
                            };
                            0
                        }
                    };
                    if has_result {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(result));
                    }
                }
                Ok(VcpuStop::Exec {
                    cmd,
                    grants_ptr,
                    grants_n,
                    entry,
                    size_log2,
                    dst,
                    personality,
                }) => {
                    // FORK.md §8.6 — `execve` image-replace (#1080). Every refusal writes a probeable
                    // errno to `dst` and lets the caller run on (POSIX: `execve` returns only on
                    // failure); a success swaps this task's activation to the command and never returns.
                    macro_rules! refuse {
                        ($e:expr) => {{
                            tasks[ti].vt.active.set(dst, Reg::from_i32($e as i32));
                            continue;
                        }};
                    }
                    // Admissible from a clean root computation only (no serve handler, root fiber, a
                    // non-durable domain) — the tree-walker's `clean_root`. Both a **root-context** exec
                    // (`env: None` — the shell / `bash -c` replacing itself) and a **fork-twin** exec
                    // (`env: Some` — bash forking then exec'ing an external command) are serviced; they
                    // differ only in where the rebuilt activation's window/host/table live.
                    let clean = tasks[ti].vt.active.serve_ticket.is_none()
                        && tasks[ti].vt.active_id == ROOT_FIBER
                        && !host.is_durable();
                    if !clean {
                        refuse!(super::EINVAL);
                    }
                    // The build (resolve + compile + admit + powerbox + personality carry + image
                    // materialize) runs against the exec'ing task's own window + powerbox, then the
                    // rebuilt activation is installed where its `env` points.
                    // The rebuilt activation lands where the task's `env` points, with the window the
                    // image was built in, which replaces the caller's.
                    let start = match tasks[ti].env {
                        None => {
                            // Root: build against the driver window/host, then migrate the task into a
                            // confined env holding the command powerbox + its module table (the shared
                            // `dom.table` maps module 0, not the pushed command).
                            let built = exec_image_build(
                                host,
                                mem.as_ref(),
                                dom,
                                cmd,
                                grants_ptr,
                                grants_n,
                                entry,
                                size_log2,
                                personality,
                                leaf.as_ref(),
                            );
                            match built {
                                Err(e) => refuse!(e),
                                Ok(built) => {
                                    tasks[ti].vt = built.vt;
                                    // The driver window is released for the image's own.
                                    drop(mem.take());
                                    let eidx = extra_envs.len();
                                    extra_envs.push(ChildEnv {
                                        mem: Some(built.mem),
                                        host: std::sync::Arc::new(std::sync::Mutex::new(
                                            built.host,
                                        )),
                                        table: built.table,
                                        fuel: *fuel,
                                        fibers: FiberTables::default(),
                                    });
                                    tasks[ti].env = Some(eidx);
                                    built.leaf
                                }
                            }
                        }
                        Some(k) => {
                            // Fork twin: build against its confined env (its powerbox carries the
                            // personality), then overwrite the env's window, host and table — the fuel
                            // and task id are kept.
                            let host_arc = std::sync::Arc::clone(&extra_envs[k].host);
                            let built = {
                                let mut g = host_arc.lock_unpoisoned();
                                exec_image_build(
                                    &mut g,
                                    extra_envs[k].mem.as_ref(),
                                    dom,
                                    cmd,
                                    grants_ptr,
                                    grants_n,
                                    entry,
                                    size_log2,
                                    personality,
                                    leaf.as_ref(),
                                )
                            };
                            match built {
                                Err(e) => refuse!(e),
                                Ok(built) => {
                                    tasks[ti].vt = built.vt;
                                    extra_envs[k].host =
                                        std::sync::Arc::new(std::sync::Mutex::new(built.host));
                                    extra_envs[k].table = built.table;
                                    extra_envs[k].mem = Some(built.mem);
                                    built.leaf
                                }
                            }
                        }
                    };
                    if let Some(start) = start {
                        let win = tasks[ti].env.and_then(|k| extra_envs[k].mem.as_ref());
                        if let Some(step) = leaf_step(start, ti, win, pending_tierup) {
                            return Ok(step);
                        }
                    }
                }
                Ok(VcpuStop::SpawnSelf { cmd, plan, dst }) => {
                    // A personality `posix_spawn`: mint a process as the fork arm mints a twin (its
                    // pid is its task index + 1, and its parent reaps it) and build its image as the
                    // exec arm builds one, in an env of its own: a fresh window of the caller's
                    // geometry, with nothing of the caller copied. The caller runs on with the pid,
                    // or `-EAGAIN` when no process could be minted. A process whose image cannot be
                    // built still takes its task, done before it ran, so the settle retires it as it
                    // retires any process and no later process is handed its pid.
                    // The tree-walker's gate: a request from a fiber keeps its placeholder, and a
                    // serve handler is not a clean root.
                    if tasks[ti].vt.active_id != ROOT_FIBER {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i64(temen_ir::errno::ENOSYS));
                        continue;
                    }
                    if tasks[ti].vt.active.serve_ticket.is_some() {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(super::EINVAL));
                        continue;
                    }
                    let child_ti = tasks.len();
                    let pid = child_ti as u64 + 1;
                    let (twin, child_fuel) = match tasks[ti].env {
                        Some(k) => (
                            extra_envs[k]
                                .host
                                .lock_unpoisoned()
                                .spawn_powerbox(pid, plan),
                            extra_envs[k].fuel,
                        ),
                        None => (host.spawn_powerbox(pid, plan), *fuel),
                    };
                    let Some(mut twin) = twin else {
                        tasks[ti].vt.active.set(dst, Reg::from_i64(super::EAGAIN));
                        continue;
                    };
                    let caller = match tasks[ti].env {
                        Some(k) => extra_envs[k].mem.as_ref(),
                        None => mem.as_ref(),
                    };
                    let built = exec_image_build(
                        &mut twin,
                        caller,
                        dom,
                        cmd,
                        0,
                        0,
                        0,
                        0,
                        true,
                        leaf.as_ref(),
                    );
                    let (child_mem, child_host, table, vt, state, start) = match built {
                        Ok(built) => {
                            // Its own park door and pump bell, as the fork arm wires a twin's.
                            built.host.wire_park_door();
                            if let Some(bell) = host.external_wake() {
                                if let Some((_, source)) = built.host.signal_poll() {
                                    wire_pump_bell(&source, &bell);
                                }
                            }
                            let state = TaskState::Runnable;
                            let (m, h) = (Some(built.mem), built.host);
                            (m, h, built.table, built.vt, state, built.leaf)
                        }
                        Err(_) => {
                            let mut vt = VTask {
                                active: tasks[ti].vt.active.clone(),
                                active_id: ROOT_FIBER,
                                chain: Vec::new(),
                                root_shadow_sp: 0,
                                active_invoke: None,
                            };
                            vt.release();
                            let failed = Err(Trap::Exit(super::SPAWN_EXEC_FAILED as i32));
                            let state = TaskState::Done(failed);
                            (None, twin, dom.table.fork(), vt, state, None)
                        }
                    };
                    let eidx = extra_envs.len();
                    extra_envs.push(ChildEnv {
                        mem: child_mem,
                        host: std::sync::Arc::new(std::sync::Mutex::new(child_host)),
                        table,
                        fuel: child_fuel,
                        fibers: FiberTables::default(),
                    });
                    tasks.push(TaskSlot {
                        vt,
                        threads: Vec::new(),
                        env: Some(eidx),
                        state,
                        suspended: None,
                        lease: None,
                    });
                    forked_twins.insert(child_ti);
                    tasks[ti].vt.active.set(dst, Reg::from_i64(pid as i64));
                    if let Some(start) = start {
                        let win = extra_envs[eidx].mem.as_ref();
                        if let Some(step) = leaf_step(start, child_ti, win, pending_tierup) {
                            return Ok(step);
                        }
                    }
                }
                Ok(VcpuStop::ForkSelf { dst }) => {
                    // #799/#1080 — personality `fork()`: duplicate THIS task into a twin (private window
                    // copy + forked powerbox). The parent (`ti`) keeps running with the twin's pid; the
                    // twin resumes at the same op (pc already advanced) with `0`. A non-bare or
                    // non-forkable caller fails closed to `-EAGAIN` (a value, never a hang — invariant 5).
                    // The bytecode port of the tree-walker's `Blocked::ForkSelf` → `fork_vcpu` engine.
                    // The twin's **personality pid** is its task index + 1: the process table's root is
                    // pid 1 (grant) at task index 0, so `pid = index + 1` aligns the two spaces (twins
                    // get 2, 3, … — never colliding with the root's 1), and a `waitpid(pid)` maps back to
                    // task index `pid - 1` (the reap settle below). `twin_ti` stays the raw task index.
                    let twin_ti = tasks.len();
                    let twin_pid = twin_ti as u64 + 1;
                    let bare = tasks[ti].threads.iter().all(|t| t.is_none())
                        && tasks[ti].vt.active_id == ROOT_FIBER
                        && tasks[ti].vt.chain.is_empty();
                    // The window + powerbox to duplicate: a confined child from its `extra_env`, or a
                    // **root** caller (`env: None`, e.g. `bash -c` forking) from the driver window/host.
                    let forked: Option<(u64, Option<Mem>, Host)> = if bare {
                        (|| match tasks[ti].env {
                            Some(k) => {
                                let tm = match &extra_envs[k].mem {
                                    Some(m) => Some(m.fork_private()?),
                                    None => None,
                                };
                                let th = extra_envs[k]
                                    .host
                                    .lock_unpoisoned()
                                    .fork_powerbox(twin_pid)?;
                                Some((extra_envs[k].fuel, tm, th))
                            }
                            None => {
                                let tm = match mem.as_ref() {
                                    Some(m) => Some(m.fork_private()?),
                                    None => None,
                                };
                                let th = host.fork_powerbox(twin_pid)?;
                                Some((*fuel, tm, th))
                            }
                        })()
                    } else {
                        None
                    };
                    match forked {
                        None => {
                            tasks[ti].vt.active.set(dst, Reg::from_i64(super::EAGAIN));
                        }
                        Some((twin_fuel, twin_mem, twin_host)) => {
                            // #1080 pipeline rung — wire the twin's OWN park door. `fork_powerbox` mints
                            // the twin's personality with `park_req: None` (the door "lands at mint");
                            // the driver must install it, exactly as the tree-walker wires each child's
                            // door at fork and as the root got it at run start. Without this a forked
                            // twin's own `fork()`/`waitpid()` finds no park delegate and fails closed
                            // (`-ENOSYS`/`-ECHILD`) — so a bash pipeline SUBSHELL (itself a twin) cannot
                            // fork+wait its command, wedging `echo | cat` in a waitpid busy-loop.
                            twin_host.wire_park_door();
                            // #1262 — wire the twin's personality signal doors to the SAME external-wake
                            // bell as the root. A foreground/background job is a twin, and an embedder
                            // signal to it (a terminal `^C`/`^Z`, a `kill(1)`) while the pump is
                            // all-parked must ring the bell so the settle re-runs — the #1215 kill sweep
                            // then finalizes a terminated twin parked on its terminal read, and the reap
                            // wakes the shell. Without it the twin's doors were unwired and the embedder
                            // signal was slept through: interactive `^C` of a parked `cat` deadlocked.
                            if let Some(bell) = host.external_wake() {
                                if let Some((_, tsource)) = twin_host.signal_poll() {
                                    wire_pump_bell(&tsource, &bell);
                                }
                            }
                            // The twin's continuation is the parent's, at the post-fork resume point,
                            // with the return-twice `0` (the parent keeps the twin's pid, set below). A
                            // bare root carries no resume chain / invoke (`Vm` derives `Clone`).
                            let mut twin_active = tasks[ti].vt.active.clone();
                            twin_active.set(dst, Reg::from_i64(0));
                            // #816 env-routed tier-up: the clone carries the parent's bitmap; the twin
                            // may keep it only if its OWN private window (`fork_private`, pushed into
                            // `extra_envs` below) is servable — the driver then answers each of the
                            // twin's events over that window via the pending-env routing. The
                            // owned-flat twin backing (#816 item 3) makes the tier-up shapes servable
                            // on every target; a twin still on the `Paged` fallback (unbounded
                            // reservation, allocation failure) is not: strip the bitmap so it
                            // interprets, fail-closed.
                            if !tierup_servable(twin_mem.as_ref(), mem.as_ref()) {
                                twin_active.jit_eligible = None;
                                twin_active.jit_page_checked = false;
                            }
                            let twin_root_sp =
                                twin_active.durable_region_base + super::REGION_HEADER_LEN; // its context's empty frame base
                            let twin_vt = VTask {
                                active: twin_active,
                                active_id: ROOT_FIBER,
                                chain: Vec::new(),
                                root_shadow_sp: twin_root_sp,
                                active_invoke: None,
                            };
                            // #1297: the twin's own dispatch table, seeded with the caller's installs.
                            let twin_table = match tasks[ti].env {
                                Some(k) => extra_envs[k].table.fork(),
                                None => dom.table.fork(),
                            };
                            let twin_eidx = extra_envs.len();
                            extra_envs.push(ChildEnv {
                                mem: twin_mem,
                                host: std::sync::Arc::new(std::sync::Mutex::new(twin_host)),
                                table: twin_table,
                                fuel: twin_fuel,
                                fibers: FiberTables::default(),
                            });
                            debug_assert_eq!(
                                tasks.len(),
                                twin_ti,
                                "twin lands at its pre-read index"
                            );
                            tasks.push(TaskSlot {
                                vt: twin_vt,
                                threads: Vec::new(),
                                env: Some(twin_eidx),
                                state: TaskState::Runnable,
                                suspended: None,
                                lease: None,
                            });
                            forked_twins.insert(twin_ti);
                            tasks[ti].vt.active.set(dst, Reg::from_i64(twin_pid as i64));
                        }
                    }
                }
                Ok(VcpuStop::ReapWait { child }) => {
                    // #799 — personality blocking `waitpid()`: the op was rewound (it re-executes on
                    // wake). Park until the named child completes; the settle scan re-admits it after
                    // firing the twin's exit hooks, so the re-executed op finds the twin retired.
                    //
                    // #1080 pipeline rung — an ANY-child parker (`child: None`) that re-parks has just
                    // re-run its `waitpid(-1)` and found nothing reapable, so every already-hooked Done
                    // twin's exit has been CONSUMED (reaped by this parker, or owned by another parent
                    // who reaps straight from the personality table without an engine wake). Such twins
                    // must stop satisfying the any-child wake criterion, or the settle re-wakes this
                    // parker forever on the same stale Done twin — a livelock in which the woken parker
                    // (lowest task index, e.g. root bash) is picked every pump iteration and a Runnable
                    // later task (the pipeline's exec stage) is NEVER scheduled: the `echo | cat` wedge
                    // (root re-parked ~2M times while `cat`'s twin starved). A Done-but-NOT-yet-hooked
                    // twin is kept: its exit hooks (and so its reapable zombie) fire at the next settle,
                    // and this parker must wake for it. `Some(pid)` parks are prune-immune (their wake
                    // keys on `tasks[pid-1]` directly) and self-limiting (the woken re-run reaps + returns).
                    if child.is_none() {
                        forked_twins.retain(|&j| {
                            !(hooked_twins.contains(&j)
                                && matches!(tasks[j].state, TaskState::Done(_)))
                        });
                    }
                    tasks[ti].state = TaskState::BlockedReapPersonality { child };
                }
                Ok(VcpuStop::PipeRead { pipe }) => {
                    // #1080 rung 4 — park this task on a blocking pipe read; the settle scan polls
                    // `pipe_read_ready` and re-admits (the rewound read re-executes) when ready.
                    tasks[ti].state = TaskState::BlockedPipeRead { pipe };
                }
                Ok(VcpuStop::PipeWrite { pipe }) => {
                    tasks[ti].state = TaskState::BlockedPipeWrite { pipe };
                }
                Ok(VcpuStop::Spawn {
                    func,
                    sp,
                    arg,
                    dst,
                    module,
                }) => {
                    // `func` resolves in the SPAWNING FRAME's module (an installed §22 unit spawns its
                    // own functions — CONSOLIDATION.md §11); the child's root frame starts there too.
                    let Some(cm) = dom.source.get(module as usize) else {
                        complete(tasks, ti, Err(Trap::Malformed));
                        continue;
                    };
                    if func as usize >= cm.progs.len() {
                        complete(tasks, ti, Err(Trap::Malformed));
                        continue;
                    }
                    let live = tasks
                        .iter()
                        .filter(|t| !matches!(t.state, TaskState::Done(_)))
                        .count();
                    if live >= super::MAX_VCPUS {
                        complete(tasks, ti, Err(Trap::ThreadFault)); // thread bomb
                        continue;
                    }
                    let mut child =
                        VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)])?;
                    child.active.module = module as usize;
                    child.active.home = module as usize;
                    // #926 slice 2: a `thread.spawn` child running in **module 0** tiers up its own
                    // eligible calls too (the run's bitmap is per-module-0-function, shared across the
                    // root and its same-module threads). A child spawned in another module (a confined
                    // §22 unit spawning its own function) runs a different module, where the module-0
                    // bitmap does not apply — leave it interpreting.
                    //
                    // #816 env-routed tier-up: the thread shares its spawner's env (below), so it may
                    // inherit the bitmap exactly when that env's window is tier-up-servable — the
                    // driver answers every per-event read (`win` base, mapped extent, page state) for
                    // the pending task's env ([`CoopRun::pending_win`] and friends), so a §14
                    // confined child's thread now tiers up over its own carve. A window the driver
                    // cannot flatly address (a fork twin's `Paged` private copy on wasm) fails
                    // closed to the interpreter, which is always right.
                    if module == 0
                        && tierup_servable(
                            match tasks[ti].env {
                                None => mem.as_ref(),
                                Some(k) => extra_envs[k].mem.as_ref(),
                            },
                            mem.as_ref(),
                        )
                    {
                        if let Some(e) = eligible.as_ref() {
                            child.active.jit_eligible = Some(std::sync::Arc::clone(e));
                            child.active.jit_page_checked = *page_checked;
                        }
                    }
                    let cidx = tasks.len();
                    // §12 seed the child vCPU's TLS register to its dense id (root is task 0), so
                    // `vcpu.tls.get` returns the worker index — the tree-walker's `tls: id` seeding.
                    child.active.tls = cidx as i64;
                    // A thread shares its spawner's window/powerbox — so it inherits the spawner's env
                    // (the shared domain for a root-spawned thread, or the same confined `instantiate`
                    // env for one spawned by a confined child).
                    let env = tasks[ti].env;
                    tasks.push(TaskSlot {
                        vt: child,
                        threads: Vec::new(),
                        env,
                        state: TaskState::Runnable,
                        suspended: None,
                        lease: None,
                    });
                    let handle = tasks[ti].threads.len() as i32;
                    tasks[ti].threads.push(Some(cidx));
                    tasks[ti].vt.active.set(dst, Reg::from_i32(handle));
                }
                // §14 confined children (ops 0, 5, 13, 17): admitted by the one shared
                // `admit_confined_child` against the spawning task's own window, fuel and powerbox
                // (#1727), then a task of this executor over its own environment — a nested child is
                // its own domain (a fresh natural table over the module it runs, no installed §22
                // units). `join` is the shared seam below.
                Ok(VcpuStop::Instantiate { spawn, dst }) => {
                    let pm: Option<&Mem> = match tasks[ti].env {
                        None => mem.as_ref(),
                        Some(k) => extra_envs[k].mem.as_ref(),
                    };
                    let pfuel = match tasks[ti].env {
                        None => *fuel,
                        Some(k) => extra_envs[k].fuel,
                    };
                    let spawner = &tasks[ti].vt.active;
                    let admitted = task_host(host, extra_envs, tasks[ti].env)
                        .with(|h| admit_confined_child(h, pm, pfuel, &dom.source, spawner, spawn));
                    let child = match admitted {
                        Ok(Some(c)) => c,
                        Ok(None) => {
                            tasks[ti]
                                .vt
                                .active
                                .set(dst, Reg::from_i32(super::EINVAL as i32));
                            continue;
                        }
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    // #816 env-routed tier-up: a same-module child runs the spawner's module, so the
                    // run's bitmap applies to it too — inherited when the child's window is servable
                    // (`nested_view` shares the parent backing, so a root-lineage carve always is; a
                    // fork twin's descendant is gated by its backing like the twin). The driver serves
                    // each of the child's tier-ups over its own carve via the pending-env routing
                    // (`CoopRun::pending_win`/`mem_map_info`); the emitted module's per-access live
                    // bound (elision off for instantiator-bearing modules, temen-wasm-jit
                    // `elide_bound`) confines them to the carve.
                    let tierup = match (&child.program, eligible.as_ref()) {
                        (ChildProgram::Spawner(..), Some(e))
                            if tierup_servable(child.mem.as_ref(), mem.as_ref()) =>
                        {
                            Some((std::sync::Arc::clone(e), *page_checked))
                        }
                        _ => None,
                    };
                    let started = coop_start_child(
                        tasks,
                        extra_envs,
                        ti,
                        &dom.source,
                        child,
                        spawn.entry,
                        dst,
                        tierup,
                    );
                    if let Err(t) = started {
                        complete(tasks, ti, Err(t));
                    }
                }
                // §5 `instantiate_detached` (op 15): the child is a task of this executor over a
                // **fresh window of its own** (`Mem::detached`, its own guard) — not a carve — the
                // tree-walk oracle's spawn (`run_with_host`'s op-15 arm) on the cooperative driver.
                // Admission first (entry shape, the window = the module's declared memory, the args
                // payload, `premap_admit`, no durable domain, then the `Budget.mem` take — a refused
                // spawn lands `-EINVAL` and charges nothing); then the child powerbox: starter
                // `Instantiator`/`AddressSpace` over the reservation, the by-name re-grants (the op-11
                // record format, fail-closed), and a pre-mapped `SharedRegion` staged and applied to
                // the window before the child runs an op. `join` is the shared seam below.
                Ok(VcpuStop::InstantiateDetached { spawn, dst }) => {
                    let pm: Option<&Mem> = match tasks[ti].env {
                        None => mem.as_ref(),
                        Some(k) => extra_envs[k].mem.as_ref(),
                    };
                    let pfuel = match tasks[ti].env {
                        None => *fuel,
                        Some(k) => extra_envs[k].fuel,
                    };
                    let admitted = task_host(host, extra_envs, tasks[ti].env)
                        .with(|h| admit_detached_in_process(h, pm, pfuel, spawn));
                    let child = match admitted {
                        Ok(Some(c)) => c,
                        Ok(None) => {
                            tasks[ti]
                                .vt
                                .active
                                .set(dst, Reg::from_i32(super::EINVAL as i32));
                            continue;
                        }
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let started = coop_start_child(
                        tasks,
                        extra_envs,
                        ti,
                        &dom.source,
                        child,
                        spawn.entry,
                        dst,
                        None,
                    );
                    if let Err(t) = started {
                        complete(tasks, ti, Err(t));
                    }
                }
                Ok(VcpuStop::Join { handle, dst }) => {
                    let slot = match super::resolve_thread(&tasks[ti].threads, handle) {
                        Ok(s) => s,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let child = tasks[ti].threads[slot].expect("resolve_thread checked liveness");
                    match &tasks[child].state {
                        TaskState::Done(res) => {
                            // The child already finished: deliver now (a child trap propagates here).
                            let res = res.clone();
                            tasks[ti].threads[slot] = None;
                            match res {
                                Ok(vals) => {
                                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                                    tasks[ti].vt.active.set(dst, Reg::from_value(v));
                                }
                                Err(t) => complete(tasks, ti, Err(t)),
                            }
                        }
                        _ => {
                            tasks[ti].state = TaskState::BlockedJoin { child, slot, dst };
                        }
                    }
                }
                Ok(VcpuStop::Wait {
                    base,
                    expected,
                    width,
                    timeout,
                    dst,
                }) => {
                    // §3.6 slice 5a: a wait issued INSIDE a fiber parks the FIBER, not this vCPU
                    // (the tree-walk oracle's fiber-park routing — DESIGN.md "blocks the fiber,
                    // never the domain"; `fiber_parks.rs`). Unwind one chain link to the resumer
                    // with `(FIBER_PARKED, 0)` and set the fiber aside; the park-time value recheck
                    // closes the park-vs-store race (a store that already landed wakes it with
                    // `WAIT_NOT_EQUAL` — after the one transient `FIBER_PARKED`, like the oracle).
                    if tasks[ti].vt.active_id != ROOT_FIBER {
                        let durable = host.is_durable();
                        // The fiber lives in its task's domain: the root's registry and window, or its
                        // confined `instantiate` env's.
                        let (fibers, fiber_sp, mem) = match tasks[ti].env {
                            None => (&mut *fibers, &mut *fiber_sp, &mut *mem),
                            Some(e) => {
                                let e = &mut extra_envs[e];
                                (&mut e.fibers.fibers, &mut e.fibers.sp, &mut e.mem)
                            }
                        };
                        let (cur, key) = mem
                            .as_ref()
                            .map_or((0, super::FutexKey::Anon(0, base)), |m| {
                                (m.atomic_value(base, width), m.futex_key(base))
                            });
                        let woken = (cur != expected).then_some(super::WAIT_NOT_EQUAL);
                        let park = |vm| FiberState::WaitParked {
                            vm,
                            wait_dst: dst,
                            key,
                            // #1638: an infinite wait arms neither clock. It ends by `notify`,
                            // by the park-time recheck, or not at all — and "not at all" is the
                            // driver's deadlock exit, not a fabricated `WAIT_TIMED_OUT`.
                            deadline: timeout.map(|t| clock.saturating_add(t)),
                            real_deadline: timeout.map(sched_wall_deadline),
                            woken,
                        };
                        if let Some(k) = park_running_fiber(
                            &mut tasks[ti].vt,
                            fibers,
                            fiber_sp,
                            mem,
                            durable,
                            true,
                            park,
                            |_| woken.is_some(),
                        ) {
                            tasks[ti].state = TaskState::BlockedOnFiber { fiber: k };
                        }
                        continue;
                    }
                    // Re-read the value (the cooperative analogue of the futex compare-under-lock): if it
                    // already changed, return not-equal; else park until notified or timed out. Both the
                    // value re-read and the rendezvous key are taken against THIS task's own memory: a
                    // confined `instantiate` child steps against its `extra_envs` window, not the root
                    // `mem`, and the key is backing-identity canonical (`futex_key`) so two children that
                    // mapped the same `SharedRegion` into separate windows rendezvous (S1c). Reading the
                    // root `mem` here instead would make a child's `wait` on its mapped ring flag re-read
                    // an unrelated root byte and spin forever.
                    let (cur, key) = {
                        let tmem: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        (
                            tmem.map(|m| m.atomic_value(base, width)).unwrap_or(0),
                            tmem.map(|m| m.futex_key(base))
                                .unwrap_or(super::FutexKey::Anon(0, base)),
                        )
                    };
                    if cur != expected {
                        tasks[ti]
                            .vt
                            .active
                            .set(dst, Reg::from_i32(super::WAIT_NOT_EQUAL));
                    } else {
                        tasks[ti].state = TaskState::BlockedWait {
                            key,
                            // #1638: `None` for an infinite wait — see the fiber park above.
                            deadline: timeout.map(|t| clock.saturating_add(t)),
                            dst,
                        };
                    }
                }
                Ok(VcpuStop::Notify { base, count, dst }) => {
                    // Wake up to `count` waiters, lowest task index first (deterministic). Key on the
                    // notifying task's own memory + backing identity (mirrors the wait arm), so a notify
                    // from one child's window matches a waiter parked from another child's window on the
                    // same `SharedRegion` byte.
                    let key = {
                        let tmem: Option<&Mem> = match tasks[ti].env {
                            None => mem.as_ref(),
                            Some(k) => extra_envs[k].mem.as_ref(),
                        };
                        tmem.map(|m| m.futex_key(base))
                            .unwrap_or(super::FutexKey::Anon(0, base))
                    };
                    let want = count as u32;
                    let mut woken = 0u32;
                    for t in tasks.iter_mut() {
                        if woken >= want {
                            break;
                        }
                        if let TaskState::BlockedWait {
                            key: wkey,
                            dst: wdst,
                            ..
                        } = t.state
                        {
                            if wkey == key {
                                t.vt.active.set(wdst, Reg::from_i32(super::WAIT_WOKEN));
                                t.state = TaskState::Runnable;
                                woken += 1;
                            }
                        }
                    }
                    // §3.6 slice 5a: also wake event-parked FIBER waiters, in every domain (the root's
                    // registry, then each child env's, lowest slot first — deterministic like the task
                    // scan), on the same canonical key. The status is delivered when a `cont.resume`
                    // claims the fiber.
                    for f in fibers.iter_mut().chain(
                        extra_envs
                            .iter_mut()
                            .flat_map(|e| e.fibers.fibers.iter_mut()),
                    ) {
                        if woken >= want {
                            break;
                        }
                        if let FiberState::WaitParked {
                            key: fkey,
                            woken: w @ None,
                            ..
                        } = f
                        {
                            if *fkey == key {
                                *w = Some(super::WAIT_WOKEN);
                                woken += 1;
                            }
                        }
                    }
                    tasks[ti].vt.active.set(dst, Reg::from_i32(woken as i32));
                }
                Ok(VcpuStop::JitInstall { h, code, dst }) => {
                    // Resolve authority + the unit's funcs from the TASK's host (a forged/cross-table
                    // handle is an inert CapFault → trap), compile the unit to bytecode, and install it
                    // into the task's own dispatch table (a §14 child's `extra_envs[k].table` — its
                    // installs are invisible to the parent's `call.dyn`, #1296). Compiling the unit can
                    // fail only if it uses an op the bytecode engine doesn't lower yet — the one place a
                    // guest-provided unit can outrun coverage (no tree-walker fallback mid-run).
                    let resolved = match tasks[ti].env {
                        None => resolve_jit_unit(host, h, code),
                        Some(k) => resolve_jit_unit(&extra_envs[k].host.lock_unpoisoned(), h, code),
                    };
                    let ((funcs, types), unit_id) = match resolved {
                        Ok(f) => f,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let res = match compile_module(&funcs, &types, None) {
                        Some(unit) => match match tasks[ti].env {
                            None => dom.install(unit),
                            Some(k) => jit_install_into(&dom.source, &extra_envs[k].table, unit),
                        } {
                            Some(slot) => {
                                // #926 slice 2f: mirror `slot → (domain, unit)` so the browser B2
                                // driver can rebuild its `WebAssembly.Table` when the mirror moves.
                                // The install itself always runs interpreted (a unit with a `call.cap`
                                // never emits) — here between host events, or (#1233) inside a tier-up
                                // region's bounce via `drive_nested`'s twin of this arm, after which
                                // the driver re-syncs before the emitted frame resumes. Inert on the
                                // native drive (no shared table).
                                // The mirror is the ROOT's emitted dispatch table: a §14 child's
                                // install stays in the child's own table (#1296) — mirroring it here
                                // would publish the child's unit into the parent's `call.dyn` slots.
                                if tasks[ti].env.is_none() {
                                    if let Some(e) = slot_units.get_mut(slot) {
                                        *e = Some(unit_id); // #1233: survives the guest's `release`
                                    }
                                    *table_gen = table_gen.wrapping_add(1); // slot mirror changed → re-sync
                                }
                                slot as i64
                            }
                            None => super::ENOSPC,
                        },
                        None => {
                            complete(tasks, ti, Err(Trap::Malformed)); // unit op outside coverage
                            continue;
                        }
                    };
                    tasks[ti].vt.active.set(dst, Reg::from_i64(res));
                }
                Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                    let authority = match tasks[ti].env {
                        None => host.resolve_jit_domain(h),
                        Some(k) => extra_envs[k].host.lock_unpoisoned().resolve_jit_domain(h),
                    };
                    if let Err(t) = authority {
                        complete(tasks, ti, Err(t)); // authority check
                        continue;
                    }
                    let n_real = dom.source.primary().progs.len();
                    let cleared = match tasks[ti].env {
                        None => dom.uninstall(slot as usize, n_real),
                        Some(k) => jit_uninstall_from(
                            &dom.source,
                            &extra_envs[k].table,
                            slot as usize,
                            n_real,
                        ),
                    };
                    let res = if cleared {
                        // Keep the B2 mirror exact — a freed slot must trap in the JS table too.
                        if let Some(e) = slot_units.get_mut(slot as usize) {
                            *e = None;
                        }
                        *table_gen = table_gen.wrapping_add(1); // slot mirror changed → re-sync
                        0
                    } else {
                        super::EINVAL
                    };
                    tasks[ti].vt.active.set(dst, Reg::from_i64(res));
                }
                Ok(VcpuStop::JitInvoke {
                    h,
                    code,
                    argv,
                    dst,
                    params,
                    results,
                }) => {
                    // #926 slice 2e — surface to the browser host when this run drives tier-up
                    // (`eligible` set) and the unit has **emitted wasm** with an all-scalar signature
                    // over a representable window: the host runs the emitted `f0` and delivers the
                    // results back ([`deliver_jit_invoke_vals`]). Otherwise — the native `drive` (no
                    // `eligible`), an interpreter-only unit (no emitted wasm), a non-scalar signature,
                    // or an unrepresentable window — fall through to the interpreted service below,
                    // which is always correct (the fail-closed default the single-vCPU pump also uses).
                    let scalar = |t: &ValType| {
                        matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64)
                    };
                    // A §14 child's invoke is serviced interpreted below, against its own window
                    // and host (#1296); only the root's units surface to the emitted tier.
                    let emittable = eligible.is_some()
                        && tasks[ti].env.is_none()
                        && params.iter().all(scalar)
                        && results.iter().all(scalar);
                    let surfaced = if emittable {
                        // Resolve the unit's emitted wasm exactly as the browser FFI's resolver does
                        // (`jit_unit_wasm`), and the #717 committed-extent bound over the run's window
                        // (a `Jit.invoke` runs against the shared root powerbox/window).
                        let wasm = host.resolve_jit_domain(h).ok().and_then(|domain| {
                            let (cd, cu) = host.resolve_jit_code(code).ok()?;
                            (cd == domain).then(|| host.jit_unit_wasm_or_emit(cd, cu))?
                            // #1301
                        });
                        let mapped = match mem.as_ref() {
                            None => Some(0),
                            Some(m) if *page_checked => Some(m.reserved_size()),
                            Some(m) => m.scalar_extent(),
                        };
                        wasm.zip(mapped)
                    } else {
                        None
                    };
                    if let Some((wasm, mapped)) = surfaced {
                        *pending_jit = Some((ti, dst as usize, results.clone()));
                        return Ok(CoopStep::JitInvoke {
                            code,
                            wasm,
                            argv,
                            params,
                            results,
                            mapped,
                        });
                    }
                    // Resolve unit funcs (authority + cross-table) against the task's host, as for
                    // install, and compile.
                    let resolved = match tasks[ti].env {
                        None => resolve_jit_unit(host, h, code),
                        Some(k) => resolve_jit_unit(&extra_envs[k].host.lock_unpoisoned(), h, code),
                    };
                    let (funcs, types) = match resolved {
                        Ok((f, _)) => f,
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    };
                    let unit = match compile_module(&funcs, &types, None) {
                        Some(u) => u,
                        None => {
                            complete(tasks, ti, Err(Trap::Malformed));
                            continue;
                        }
                    };
                    // Arity-check the unit entry (func 0) against the call's (code-stripped) signature.
                    let arity_ok = unit.sigs.first().is_some_and(|(ep, er)| {
                        ep.len() == params.len() && er.len() == results.len()
                    });
                    if !arity_ok {
                        complete(tasks, ti, Err(Trap::CapFault));
                        continue;
                    }
                    // Marshal args via the slot ABI, push the unit as a transient module, run it.
                    let child_args: Vec<Value> = params
                        .iter()
                        .zip(argv.iter())
                        .map(|(ty, s)| slot_to_val(*ty, *s))
                        .collect();
                    let umod = dom.source.push(unit);
                    // The unit runs over the TASK's window, table and host: the root's, or a §14
                    // child's own (`extra_envs[k]`) — a child's unit never sees the parent's window.
                    let ran = match tasks[ti].env {
                        None => run_invoke(
                            &dom.source,
                            &dom.table,
                            umod,
                            &child_args,
                            fuel,
                            mem,
                            &mut HostCell::Excl(host),
                            Some(&Beneath::task(&tasks[ti].vt, FiberRegRef::Owned(fibers))),
                        ),
                        Some(k) => {
                            let ChildEnv {
                                mem: cmem,
                                host: chost,
                                table: ctable,
                                fibers: cfibers,
                                ..
                            } = &mut extra_envs[k];
                            run_invoke(
                                &dom.source,
                                ctable,
                                umod,
                                &child_args,
                                fuel,
                                cmem,
                                &mut HostCell::Shared(chost),
                                Some(&Beneath::task(
                                    &tasks[ti].vt,
                                    FiberRegRef::Owned(&cfibers.fibers),
                                )),
                            )
                        }
                    };
                    match ran {
                        Ok(vals) => {
                            for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                                let re = slot_to_val(*ty, val_to_slot(*v));
                                tasks[ti].vt.active.set(dst + i as u32, Reg::from_value(re));
                            }
                        }
                        Err(t) => {
                            complete(tasks, ti, Err(t));
                            continue;
                        }
                    }
                }
            }
        }
    }

    /// #926 slice 2 — deliver an emitted tier-up region's results into the paused task and clear the
    /// pending slot, so the next [`pump`](Self::pump) resumes it. `vals` are the raw i64 result slots
    /// the host read out of the emitted `f{func}` run; they are re-tagged into the caller frame's `dst`
    /// slots per the recorded result types (mirroring [`Vcpu::deliver_tierup`]). A short reply is a
    /// malformed host reply, which traps the task (its domain tears down, surfacing as the run result).
    fn deliver_tierup(&mut self, vals: &[i64]) {
        let (ti, dst, results) = self
            .pending_tierup
            .take()
            .expect("deliver_tierup with no pending tier-up");
        if vals.len() < results.len() {
            complete(&mut self.tasks, ti, Err(Trap::Malformed));
            return;
        }
        let vals = results.iter().zip(vals).map(|(ty, v)| slot_to_val(*ty, *v));
        match dst {
            TierUpDst::Frame(dst) => {
                for (i, v) in vals.enumerate() {
                    self.tasks[ti]
                        .vt
                        .active
                        .set(dst as u32 + i as u32, Reg::from_value(v));
                }
            }
            // #1896: the process ran whole: it returned from its entry.
            TierUpDst::Entry { .. } => complete(&mut self.tasks, ti, Ok(vals.collect())),
        }
    }

    /// #926 slice 2 — the emitted tier-up region trapped: surface it exactly where the interpreter
    /// would by trapping the paused task (mirroring [`Vcpu::deliver_tierup_trap`]).
    fn deliver_tierup_trap(&mut self, trap: Trap) {
        let (ti, _dst, _results) = self
            .pending_tierup
            .take()
            .expect("deliver_tierup_trap with no pending tier-up");
        complete(&mut self.tasks, ti, Err(trap));
    }

    /// #926 slice 2e — deliver a surfaced `Jit.invoke`'s emitted `f0` result slots into the paused
    /// task and clear the pending slot (mirroring [`deliver_tierup`](Self::deliver_tierup), routed to
    /// the invoking task's frame via `pending_jit`). A short reply traps the task.
    fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        // #926 slice 2g: the emitted invoke resolved — its bounce registry dies with it (Vcpu parity).
        self.invoke_fibers.clear();
        let (ti, dst, results) = self
            .pending_jit
            .take()
            .expect("deliver_jit_invoke_vals with no pending invoke");
        if vals.len() < results.len() {
            complete(&mut self.tasks, ti, Err(Trap::Malformed));
            return;
        }
        for (i, ty) in results.iter().enumerate() {
            self.tasks[ti].vt.active.set(
                dst as u32 + i as u32,
                Reg::from_value(slot_to_val(*ty, vals[i])),
            );
        }
    }

    /// #926 slice 2e — the emitted `Jit.invoke` unit trapped: trap the invoking task (mirroring
    /// [`deliver_tierup_trap`](Self::deliver_tierup_trap)).
    fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        // #926 slice 2g: the emitted invoke resolved (trapped) — its bounce registry dies with it.
        self.invoke_fibers.clear();
        let (ti, _dst, _results) = self
            .pending_jit
            .take()
            .expect("deliver_jit_invoke_trap with no pending invoke");
        complete(&mut self.tasks, ti, Err(trap));
    }
}

/// #926 slice 2 tier-up configuration for a cooperative run: the wasm-JIT eligibility bitmap plus
/// whether it is page-checked (#750). Bundled rather than passed as two loose parameters because
/// `page_checked` is meaningful only alongside a bitmap — a `None` config is "no tier-up", exactly the
/// native `drive`. The bitmap is per **module-0** function; a direct call to an `eligible[f] == true`
/// function surfaces as a tier-up on the root and any same-module `thread.spawn` descendant.
pub struct TierUpConfig {
    /// Per-module-0-function eligibility (index = function). `true` ⇒ a direct call tiers up.
    pub eligible: std::sync::Arc<[bool]>,
    /// #750 paged tier-up: the emitted region carries a per-access page check, so an unrepresentable
    /// window surfaces with the reserved size instead of declining.
    pub page_checked: bool,
    /// #1896 — the host's emitter for **leaf images**: an image a process `execve`s that cannot park,
    /// or parks only in a stream call ([`image_parks`]), is offered to it, and runs whole on the
    /// emitted tier if it emits it. `None`: every exec'd image interprets.
    pub leaf: Option<LeafEmitter>,
}

/// The host's emitter for leaf images ([`TierUpConfig::leaf`]). Offered one ([`LeafOffer`]), it
/// emits the image to run whole from its entry, and answers whether it did.
pub type LeafEmitter = std::sync::Arc<dyn Fn(&LeafOffer) -> bool + Send + Sync>;

/// A leaf image the engine offers the host's emitter ([`LeafEmitter`]). `paged` and `parks` are the
/// engine's to decide: they are facts of the imports the image is bound to and of what its process
/// holds, which the image alone does not show.
pub struct LeafOffer<'a> {
    /// The program index the image's [`CoopEvent::TierUp`] names.
    pub module: usize,
    pub image: &'a Module,
    /// The function it runs whole from.
    pub entry: u32,
    /// It can change its page state: emit it with the per-access page check (#750).
    pub paged: bool,
    /// It can park, in a stream call (a read of an empty pipe, say). Emit it only if the host can
    /// **suspend** its emitted frames there: the call's bounce ([`CoopRun::bounce`]) then answers
    /// that it parked, and the host holds the frames until [`CoopEvent::Resume`] hands it the call's
    /// results.
    pub parks: bool,
}

/// What a [`CoopRun`] holds ([`CoopRun::footprint`]): the memory a process tree costs its embedder.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Footprint {
    /// Process windows: the root's, and each child's that has not finished. A finished process
    /// gives its window back.
    pub windows: usize,
    /// Compiled programs: the root's, one per command the run's processes exec'd, and any unit
    /// installed or spawned. A command exec'd again runs the program its first exec compiled.
    pub units: usize,
}

/// A pause of the cooperative tier-up driver [`CoopRun`], mirroring the single-vCPU [`VcpuEvent`]'s
/// tier-up-relevant subset. The cooperative driver services concurrency (`thread.spawn`, join, futex
/// wait/notify) **internally** — multiplexing every vCPU on the one host thread — so, unlike the
/// per-Worker parallel driver, those never surface; only the run's end and tier-up round-trips do.
pub enum CoopEvent {
    /// A [`CoopRun::run_for`] slice is spent: the run is live, every task's cursor persisted — call
    /// `run`/`run_for` again to continue exactly where it stopped. This is how an embedder that owns
    /// its thread (a browser Worker) runs a long program in slices, streaming its output and honoring
    /// a Pause between them. Never surfaced by an unsliced `run`.
    Paused,
    /// #1122 route (a) — the run is **idle**: every task is parked on something only the embedder can
    /// satisfy (a terminal/pipe read, a blocking stdin read) and [`CoopRun::set_suspend_on_idle`] is
    /// on. The run is live and resumable: feed input (e.g. `Posix::feed_terminal`, or a signal) and
    /// call [`run`](CoopRun::run) again. Never surfaced with the flag off (the pump blocks on the
    /// doorbell instead, or faults as a deadlock).
    Idle,
    /// The run finished; these are the root task's results.
    Done(Vec<Value>),
    /// The run trapped (the root task, or a fatal driver fault).
    Trapped(Trap),
    /// A task paused to run `func` of program `module` emitted, with raw i64 arg slots `argv`: a
    /// module-0 task on an eligible direct `Call` (`module` 0), or a process whose exec'd image is a
    /// leaf the host emitted ([`TierUpConfig::leaf`]), at its entry. `mapped` is the committed scalar
    /// window extent for the emitted `"mapped"` global. The host runs the emitted `f{func}` and calls
    /// [`CoopRun::deliver_tierup`] / [`CoopRun::deliver_tierup_trap`]; for an entry, the delivery ends
    /// the process as a return from its entry, or a trap in it, would.
    TierUp {
        module: u32,
        func: u32,
        argv: Box<[i64]>,
        mapped: u64,
    },
    /// #1896 — a call that parked in a bounce out of a leaf running emitted ([`LeafOffer::parks`],
    /// [`CoopRun::bounce`]) has returned these raw i64 result slots. The host resumes that leaf's
    /// suspended frames (the round-trip is its task's again: [`CoopRun::pending_task`]) as the
    /// bounce's return, then serves the leaf's further bounces and delivers its end as for a
    /// [`CoopEvent::TierUp`].
    Resume { results: Box<[i64]> },
    /// A task paused on a §22 `Jit.invoke` of a runtime-compiled unit with emitted `wasm`: the host
    /// runs the unit's `f0(win, env, ...argv)` (marshalling by `params`/`results`, `mapped` into its
    /// `"mapped"` global) and calls [`CoopRun::deliver_jit_invoke_vals`] /
    /// [`CoopRun::deliver_jit_invoke_trap`]. A non-emittable unit runs interpreted and never surfaces.
    JitInvoke {
        code: i32,
        wasm: std::sync::Arc<[u8]>,
        argv: Box<[i64]>,
        params: Box<[ValType]>,
        results: Box<[ValType]>,
        mapped: u64,
    },
}

/// A **resumable cooperative tier-up run**: the single-thread, no-Worker analogue of the parallel
/// `temen_par_*` driver. It owns the run's `Domain`/window/powerbox/fuel and a [`CoopSched`] that
/// multiplexes every vCPU (root + `thread.spawn` descendants) on this one thread, and pauses to the
/// host on each wasm-JIT tier-up ([`run`](Self::run) → [`CoopEvent::TierUp`] → run the emitted region
/// → [`deliver_tierup`](Self::deliver_tierup) → `run` again), exactly the loop the native tests and
/// the browser cdylib drive it with. With no eligibility bitmap it behaves as `drive`: `run` returns
/// `Done`/`Trapped` in one call. (#926 slice 2.)
pub struct CoopRun {
    dom: Domain,
    mem: Option<Mem>,
    host: Host,
    fuel: u64,
    sched: CoopSched,
}

impl CoopRun {
    /// Build a run over module `m`, entering `entry(args)` with `fuel` and the granted `host` powerbox.
    /// `tierup` is the tier-up config ([`TierUpConfig`]); pass `None` for a pure-interpreter multiplex.
    /// `None` if `m` uses an op outside the bytecode engine's subset (the caller falls back to the
    /// tree-walker), `Some(Err)` if `entry` is out of range.
    pub fn new(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        tierup: Option<TierUpConfig>,
    ) -> Option<Result<CoopRun, Trap>> {
        // A fresh engine-sized window built from `m`'s declaration + data (the native/test path).
        Self::assemble(m, entry, args, fuel, host, tierup, build_mem(m, &[]))
    }

    /// The resumable twin of [`compile_and_run_seeded_with_host`]: the same window (`init_mem` seeded
    /// under `m`'s data) and the same park-request door, so a run pumped in [`run_for`](Self::run_for)
    /// slices behaves exactly as that one-shot run.
    pub fn new_seeded(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        init_mem: &[u8],
    ) -> Option<Result<CoopRun, Trap>> {
        host.wire_park_door();
        super::LAST_CAPTURE_FAULT.with(|c| *c.borrow_mut() = None);
        Self::assemble(m, entry, args, fuel, host, None, build_mem(m, init_mem))
    }

    /// Like [`new`](Self::new), but the linear-memory window is built **over a caller-provided
    /// backing** `back` (with `init_mem` seeded first), the resumable twin of
    /// [`Vcpu::new_root_reserved_over_with_powerbox`]. This is the browser cdylib seam: the window
    /// lives in the host's own linear memory, so every emitted `f{func}(win, env, …)` addresses it
    /// directly through the one shared `env.memory`. `back` is dropped if `m` is unsupported.
    #[allow(clippy::too_many_arguments)] // the window-backing seam inherently threads more inputs
    pub fn new_over(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        tierup: Option<TierUpConfig>,
        init_mem: &[u8],
        reserved_log2: u8,
        back: std::sync::Arc<super::Region>,
    ) -> Option<Result<CoopRun, Trap>> {
        let mem = m.memory.map(|mc| {
            let mut mm = Mem::with_reservation_over(reserved_log2, mc.size_log2, back, mc.shadow);
            mm.seed(init_mem);
            mm.init_data(&m.data);
            mm.seed_null_guard(temen_ir::module_null_guard()); // #964
            mm
        });
        Self::assemble(m, entry, args, fuel, host, tierup, mem)
    }

    /// Shared constructor tail: compile `m`, range-check `entry`, and build the `CoopSched` over the
    /// caller-chosen `mem`. `None` if `m` is outside the bytecode engine's subset (fall back to the
    /// tree-walker); `Some(Err)` if `entry` is out of range or seeding traps.
    /// #1122 route (a) — the **suspend/resume session** constructor: like [`new_over`](Self::new_over)
    /// but over an engine-backed reservation (`Mem::with_reservation` — the shape every temen-run
    /// bytecode run uses, the `vm_map`-grown heap living in the reserved tail), seeded with the
    /// embedder's window image (`init_mem`: the powerbox argv/env blob) and the module's data, with
    /// the personality fork/`waitpid` park-request door wired (as the one-shot entries do) so a
    /// `bash -i` session forks and waits. Pair with [`set_suspend_on_idle`](Self::set_suspend_on_idle).
    /// `None` if `m` is outside the engine's subset.
    #[allow(clippy::too_many_arguments)] // the window-seeding seam inherently threads more inputs
    pub fn new_reserved(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        fuel: u64,
        host: Host,
        tierup: Option<TierUpConfig>,
        init_mem: &[u8],
        reserved_log2: u8,
    ) -> Option<Result<CoopRun, Trap>> {
        let compiled = compile_reserved(m)?;
        Some(Self::new_reserved_over_compiled(
            m,
            compiled,
            entry,
            args,
            fuel,
            host,
            tierup,
            init_mem,
            reserved_log2,
        ))
    }

    /// [`new_reserved`](Self::new_reserved) over an already-compiled program (the browser's cached
    /// `bash.temen` compile, #1144): the `Domain` is `over_primary` on the shared `compiled`, so a
    /// session open costs a window build + schedule, not a recompile of the whole shell.
    #[allow(clippy::too_many_arguments)]
    pub fn new_reserved_over_compiled(
        m: &Module,
        compiled: std::sync::Arc<Compiled>,
        entry: FuncIdx,
        args: &[Value],
        mut fuel: u64,
        mut host: Host,
        tierup: Option<TierUpConfig>,
        init_mem: &[u8],
        reserved_log2: u8,
    ) -> Result<CoopRun, Trap> {
        if entry as usize >= compiled.progs.len() {
            return Err(Trap::Malformed);
        }
        host.wire_park_door();
        let dom = Domain::over_primary(compiled, host.jit_table_log2());
        let mut mem = m.memory.map(|mc| {
            let mut mm = Mem::with_reservation(reserved_log2, mc.size_log2, mc.shadow);
            mm.seed(init_mem);
            mm.init_data(&m.data);
            mm.seed_null_guard(temen_ir::module_null_guard()); // #964
            mm
        });
        let sched = CoopSched::new(&dom, entry, args, &mut fuel, &mut mem, &mut host, tierup)?;
        Ok(CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        })
    }

    fn assemble(
        m: &Module,
        entry: FuncIdx,
        args: &[Value],
        mut fuel: u64,
        mut host: Host,
        tierup: Option<TierUpConfig>,
        mut mem: Option<Mem>,
    ) -> Option<Result<CoopRun, Trap>> {
        let c = compile_module_for(m)?;
        if entry as usize >= c.progs.len() {
            return Some(Err(Trap::Malformed));
        }
        let dom = Domain::new(c, host.jit_table_log2());
        let sched = match CoopSched::new(&dom, entry, args, &mut fuel, &mut mem, &mut host, tierup)
        {
            Ok(s) => s,
            Err(e) => return Some(Err(e)),
        };
        Some(Ok(CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        }))
    }

    /// #816 env-routed tier-up: the task index of the outstanding tier-up / `Jit.invoke` round-trip
    /// (the one the driver is currently serving), if any. The same resolution [`bounce`](Self::bounce)
    /// performs — at most one round-trip is outstanding, so this names *the* task every per-event
    /// driver read (`win`, mapped extent, page state) must be answered for.
    /// #1896 — the task the outstanding round-trip is for. A host that suspends leaves' frames
    /// ([`LeafOffer::parks`]) keys them by it: a [`CoopEvent::TierUp`] starts that task's leaf, and a
    /// [`CoopEvent::Resume`] resumes it.
    pub fn pending_task(&self) -> Option<usize> {
        self.pending_ti()
    }

    fn pending_ti(&self) -> Option<usize> {
        self.sched
            .pending_tierup
            .as_ref()
            .map(|(ti, ..)| *ti)
            .or_else(|| self.sched.pending_jit.as_ref().map(|(ti, ..)| *ti))
    }

    /// #816: the window the outstanding round-trip's task runs over — `extra_envs[k].mem` for an
    /// env-carrying task (a §14 confined child or one of its threads), the root window otherwise
    /// (including when no round-trip is pending — the pre-open / post-done reads). Mirrors
    /// [`bounce`](Self::bounce)'s env dispatch, so the driver's reads and the bounced cap calls
    /// always agree on which window is live.
    fn pending_mem(&self) -> Option<&Mem> {
        match self.pending_ti().and_then(|ti| self.sched.tasks[ti].env) {
            Some(k) => self.sched.extra_envs[k].mem.as_ref(),
            None => self.mem.as_ref(),
        }
    }

    /// #816: the pending round-trip task's env identity — `-1` for the root env, the `extra_envs`
    /// index for a confined child. Part of the driver's page-state cache key: env windows have their
    /// **own** `map_version` counters (a fresh Arc per `nested_view`), so a version compare is only
    /// meaningful within one env — the driver rebuilds whenever (env, version) changes.
    pub fn pending_env(&self) -> i64 {
        self.pending_ti()
            .and_then(|ti| self.sched.tasks[ti].env)
            .map_or(-1, |k| k as i64)
    }

    /// #816: the pending task's **flat window view** for the emitted call — `(base ptr, addressable
    /// len)`. For the root this is the run backing itself; for a §14 child it is the backing plus
    /// the carve offset, so the emitted `win + addr` accesses land exactly where the interpreter's
    /// confined accesses do. `None` when the pending window has no flat address (a `Paged`-backed
    /// window, e.g. a fork twin's private copy on wasm) — such tasks never carry the tier-up bitmap
    /// (the eligibility gates check [`Mem::flat_win_base`] at task creation), so a pending tier-up
    /// always resolves `Some`; the `None` arm is the fail-closed default for defensive callers.
    ///
    /// #1312: the length is the backing-clamped [`Mem::win_flat_len`], not the reservation — a
    /// cooperative run now reserves the oracle's `DEFAULT_RESERVED_LOG2` over a growable backing, and
    /// a driver must mirror what exists, not the mask domain. **Re-read both per event and after any
    /// bounce**: a `vm_map` grow can extend *and relocate* the backing.
    pub fn pending_win(&self) -> Option<(*const u8, u64)> {
        let m = self.pending_mem()?;
        Some((m.flat_win_base()?, m.win_flat_len()))
    }

    /// The pending task's window committed **scalar extent** right now — the #717 value the cdylib
    /// re-syncs to every emitted instance's `"mapped"` global after a [`bounce`](Self::bounce) (a
    /// bounced callback may have grown the window). `0` when there is no window or its state is not
    /// representable by one bound. #816: routed to the pending round-trip's task env, so a confined
    /// child's own extent (its fully-mapped carve) bounds its emitted accesses — the confinement
    /// bound — rather than the root's.
    pub fn window_scalar_extent(&self) -> u64 {
        self.pending_mem()
            .and_then(|m| m.scalar_extent())
            .unwrap_or(0)
    }

    /// The run's **root** powerbox — where the root task and its `thread.spawn` threads' host I/O
    /// lands (stdout/stderr, the framebuffer). The cdylib drains it into its capture slots at the end
    /// of a run. (§14 confined children keep their own `host` in `extra_envs`, not exposed here.)
    pub fn host_mut(&mut self) -> &mut Host {
        &mut self.host
    }

    /// #926 slice 2f / #1233 — the `(domain, unit)` identity installed at dispatch-table `slot`
    /// (`None` = empty or natural-prefix): the browser B2 driver's slot mirror, from which it
    /// rebuilds its `WebAssembly.Table` when the generation moves (a slot at or past the program's
    /// `f{i}` prefix holds an installed unit's `f0`). Keyed on the unit index rather than the §22
    /// code handle so it survives the guest's `Jit.release` of that handle — see
    /// [`CoopSched::slot_units`].
    pub fn slot_unit(&self, slot: u32) -> Option<(u32, u32)> {
        self.sched.slot_units.get(slot as usize).copied().flatten()
    }

    /// #1009: the dispatch-table generation — bumped on each `Jit.install`/`Jit.uninstall` the
    /// scheduler services. The browser B2 driver caches the generation it last synced its
    /// `WebAssembly.Table` at and rebuilds only when this advances (the single-shot pump's
    /// `temen_onramp_tierup_table_gen` twin).
    pub fn table_gen(&self) -> u32 {
        self.sched.table_gen
    }

    /// #1009 paged tier-up: the pending task's window memory-map introspection ([`MemMapInfo`]) — a
    /// paged coop driver rebuilds its page-state table from this. `None` for a memory-less run.
    /// #816: routed to the pending round-trip's task env, so a confined child's own page map (its
    /// carve geometry, its `protect`/`unmap` state) is what the emitted page check enforces —
    /// window-relative on both sides, no translation needed.
    pub fn mem_map_info(&self) -> Option<MemMapInfo> {
        self.pending_mem().map(|m| m.map_info())
    }

    /// #1009 paged tier-up: the pending task's window page-map version (bumped on every `map`/
    /// `unmap`/`protect`) — the cheap `O(1)` counter a paged coop driver compares to skip an
    /// unchanged page-state rebuild. `0` for a memory-less run. #816: env windows carry their own
    /// counters, so the driver's cache key is ([`pending_env`](Self::pending_env), this) — a bare
    /// version compare across different envs would alias.
    pub fn mem_map_version(&self) -> u64 {
        self.pending_mem().map_or(0, |m| m.map_version())
    }

    /// #1122 route (a) — make an all-parked, externally-wakeable settle surface as
    /// [`CoopEvent::Idle`] instead of blocking the calling thread on the #1122 doorbell. The
    /// suspend/resume session shape: the embedder owns this `CoopRun`, pumps until `Idle`, feeds the
    /// terminal (`Posix::feed_terminal` — the line discipline runs at feed time, host-side), and pumps
    /// again; the loop-top settle re-admits the reader on the deposit. Works on any host (no threads,
    /// no SharedArrayBuffer), keeps the deterministic cooperative schedule.
    pub fn set_suspend_on_idle(&mut self, on: bool) {
        self.sched.suspend_on_idle = on;
    }

    /// What the run holds now ([`Footprint`]).
    pub fn footprint(&self) -> Footprint {
        let children = self.sched.extra_envs.iter().filter(|e| e.mem.is_some());
        Footprint {
            windows: usize::from(self.mem.is_some()) + children.count(),
            units: self.dom.source.snapshot().len(),
        }
    }

    /// [`run`](Self::run), but return after about `ops` ops with [`CoopEvent::Paused`] if nothing else
    /// stopped the run first. The slice is counted in ops (fuel), so it is deterministic.
    pub fn run_for(&mut self, ops: u64) -> CoopEvent {
        self.sched.slice_left = Some(ops.max(1));
        let ev = self.run();
        self.sched.slice_left = None;
        ev
    }

    /// Pump the schedule to its next pause: [`CoopEvent::Done`]/[`CoopEvent::Trapped`] end the run,
    /// [`CoopEvent::TierUp`] hands an emitted region to the host (resume with `deliver_tierup*`).
    pub fn run(&mut self) -> CoopEvent {
        // `budget` is the per-step op budget `step_vcpu` hands `Vm::resume` (a `0` would run zero ops
        // and spin on `Outcome::Suspended`); the native `drive` runs unsliced, so match it with
        // `u64::MAX` — run each vCPU to its next stop. It doubles as the §3d spawn record budget
        // (`u64::MAX` ⇒ the no-record default `take_spawn_budget` already uses for a normal run).
        match self.sched.pump(
            &self.dom,
            &mut self.mem,
            &mut self.host,
            &mut self.fuel,
            u64::MAX,
        ) {
            Ok(CoopStep::Done(vals)) => CoopEvent::Done(vals),
            Ok(CoopStep::Idle) => CoopEvent::Idle,
            Ok(CoopStep::Paused) => CoopEvent::Paused,
            Ok(CoopStep::TierUp {
                module,
                func,
                argv,
                mapped,
            }) => CoopEvent::TierUp {
                module,
                func,
                argv,
                mapped,
            },
            Ok(CoopStep::Resume { results }) => CoopEvent::Resume { results },
            Ok(CoopStep::JitInvoke {
                code,
                wasm,
                argv,
                params,
                results,
                mapped,
            }) => CoopEvent::JitInvoke {
                code,
                wasm,
                argv,
                params,
                results,
                mapped,
            },
            Err(t) => CoopEvent::Trapped(t),
        }
    }

    /// Deliver the emitted tier-up region's raw i64 result slots and resume (see
    /// [`CoopSched::deliver_tierup`]). Call exactly once after a [`CoopEvent::TierUp`], before `run`.
    pub fn deliver_tierup(&mut self, vals: &[i64]) {
        self.sched.deliver_tierup(vals);
    }

    /// Surface an emitted tier-up region's trap and resume (the paused task traps). Call once after a
    /// [`CoopEvent::TierUp`] in lieu of [`deliver_tierup`](Self::deliver_tierup).
    pub fn deliver_tierup_trap(&mut self, trap: Trap) {
        self.sched.deliver_tierup_trap(trap);
    }

    /// Deliver a surfaced `Jit.invoke` unit's emitted `f0` result slots and resume (see
    /// [`CoopSched::deliver_jit_invoke_vals`]). Call once after a [`CoopEvent::JitInvoke`], before `run`.
    pub fn deliver_jit_invoke_vals(&mut self, vals: &[i64]) {
        self.sched.deliver_jit_invoke_vals(vals);
    }

    /// Surface a `Jit.invoke` unit's trap and resume (the invoking task traps). Call once after a
    /// [`CoopEvent::JitInvoke`] in lieu of [`deliver_jit_invoke_vals`](Self::deliver_jit_invoke_vals).
    pub fn deliver_jit_invoke_trap(&mut self, trap: Trap) {
        self.sched.deliver_jit_invoke_trap(trap);
    }

    /// #926 slice 2 — service an emitted tier-up region's **or** a surfaced `Jit.invoke` unit's cross-tier
    /// `call_interp(target, io)` while it is mid-run for the **currently paused task**. Routes the bounce
    /// to *that task's* env — a §14 confined child steps its own window/powerbox/dispatch table
    /// (`env == Some`), the root and its `thread.spawn` threads the shared ones (`env == None`) — so a
    /// confined leaf's callback can never reach outside its window (the confinement hinge). The fiber
    /// registry is picked by which round-trip is outstanding (#926 slice 2g, `Vcpu::bounce_call` parity):
    /// a tier-up region uses the **run-level** registry (a parked fiber persists for the run to resume),
    /// while a surfaced `Jit.invoke` uses the **invoke-confined** registry (`invoke_fibers` — its fibers
    /// die when the invoke resolves). Marshals results back into `io` and returns the result count. Call
    /// only between a [`CoopEvent::TierUp`]/[`CoopEvent::JitInvoke`] and its delivery; `Err(Malformed)` if
    /// nothing is outstanding.
    ///
    /// `spill` (#1627) is the emitted region's **spill stack** — the candidate words its frames stored
    /// before each host-reaching call. Pass `Some` **only** when every emitted frame live beneath this
    /// bounce has spilled into it (the region was emitted with spill instrumentation, and no native §22
    /// unit frame can be live). Then a `gc.roots` inside the bounce is serviced over the paused task's
    /// frames, the run's fibers, and those words. With `None` — or during a `Jit.invoke`, whose emitted
    /// unit frames never spill — something live below is out of view and the op fails closed
    /// (`CapFault`, #1660).
    ///
    /// #1896 — `Ok(None)`: the call **parked**, in a leaf whose host suspends its emitted frames
    /// ([`LeafOffer::parks`]). The task parks as an interpreted one would, the round-trip is no longer
    /// outstanding, and the host holds the frames and runs on; [`CoopEvent::Resume`] hands them the
    /// call's results once it returns.
    pub fn bounce(
        &mut self,
        target: u32,
        io: &mut [i64],
        spill: Option<&[u64]>,
    ) -> Result<Option<usize>, Trap> {
        // The paused task is whichever host round-trip is outstanding — a tier-up region or a
        // surfaced `Jit.invoke` unit; both bounce cross-tier the same way. (#926 slice 2e)
        let ti = self.pending_ti().ok_or(Trap::Malformed)?;
        // Mid-invoke iff a `Jit.invoke` (not a tier-up) is the outstanding round-trip — never both at
        // once (the one-round-trip discipline). Selects the invoke-confined registry below.
        let in_invoke = self.sched.pending_jit.is_some();
        let CoopRun {
            dom,
            mem,
            host,
            fuel,
            sched,
        } = self;
        let CoopSched {
            tasks,
            extra_envs,
            fibers,
            fiber_sp,
            fiber_meta,
            invoke_fibers,
            slot_units,
            table_gen,
            pending_tierup,
            ..
        } = sched;
        // #1896: a leaf whose host suspends its frames can park in a bounce.
        let parks = !in_invoke
            && matches!(
                pending_tierup,
                Some((_, TierUpDst::Entry { parks: true }, _))
            );
        let mut parked = None;
        // The registry `coop_bounce` threads into `drive_nested`: invoke-confined (`invoke_fibers`, no
        // shadow-SP/freeze halves — invoke fibers are transient) during an emitted `Jit.invoke`, else the
        // run-level registry with its parallel arrays. One of the two `match` arms below moves it.
        let (mut scratch_sp, mut scratch_meta) = (Vec::new(), Vec::new());
        let (mut bounce_fibers, bounce_meta): (FiberCell, Option<BounceRunCtx<'_>>) = if in_invoke {
            (
                FiberCell::Excl {
                    fibers: invoke_fibers,
                    sp: &mut scratch_sp,
                    meta: &mut scratch_meta,
                },
                None,
            )
        } else {
            (
                FiberCell::Excl {
                    fibers,
                    sp: fiber_sp,
                    meta: fiber_meta,
                },
                Some(BounceRunCtx {
                    // #1233: an install serviced inside the bounce updates the ROOT's mirror.
                    jit_mirror: Some(JitMirror {
                        units: slot_units,
                        gen: table_gen,
                    }),
                    park: parks.then_some(&mut parked),
                }),
            )
        };
        // #1627: the view beneath this bounce — the paused task's Vms and the spilled words. The
        // run's fiber registry is the bounce's own (`bounce_fibers`), which the drive scans itself.
        let beneath = spill.filter(|_| !in_invoke).map(|words| {
            let mut b = Beneath::task(&tasks[ti].vt, FiberRegRef::Owned(&[]));
            b.words.push(words);
            b
        });
        let n = match tasks[ti].env {
            // Root / `thread.spawn` thread: the run's shared window, powerbox, and domain table.
            None => {
                let mut cell = HostCell::Excl(host);
                coop_bounce(
                    &dom.source,
                    &dom.table,
                    fuel,
                    mem,
                    &mut cell,
                    &mut bounce_fibers,
                    bounce_meta,
                    target,
                    io,
                    beneath.as_ref(),
                )
            }
            // §14 confined child: its OWN window, powerbox, table, and fuel — never the root's.
            Some(k) => {
                let e = &mut extra_envs[k];
                let mut cell = HostCell::Shared(&e.host);
                coop_bounce(
                    &dom.source,
                    &e.table,
                    &mut e.fuel,
                    &mut e.mem,
                    &mut cell,
                    &mut bounce_fibers,
                    // The child's installs land in its own table (#1296), never the root's mirror.
                    bounce_meta.map(|mut c| {
                        c.jit_mirror = None;
                        c
                    }),
                    target,
                    io,
                    beneath.as_ref(),
                )
            }
        }?;
        drop(beneath);
        let Some((vm, state)) = parked else {
            return Ok(Some(n));
        };
        // #1896: the call parked. Its task waits with the rest of the call, and its host with the
        // emitted frames.
        let (ti, dst, results) = pending_tierup.take().expect("a leaf's call parked");
        let t = &mut tasks[ti];
        t.state = state;
        t.suspended = Some(Box::new(Suspended { vm, dst, results }));
        Ok(None)
    }
}

/// #816 env-routed tier-up: may a task running over `cand` carry the tier-up bitmap? Either arm
/// keeps the driver contract satisfiable — [`CoopRun::pending_win`] must resolve a flat view of the
/// pending window:
/// - a window sharing the **root backing** (a root-env thread, a §14 child carve via `nested_view`)
///   is servable wherever the root is (the per-event `win` is the backing base plus the window's
///   carve offset);
/// - any other window (a fork twin's private `fork_private` copy) qualifies only if its own region
///   is flat-addressable ([`Mem::flat_win_base`]). `fork_private` now picks an owned flat buffer
///   for a bounded twin of a flat parent (#816 item 3), so tier-up-shaped twins qualify on every
///   target; a twin that still lands on the `Paged` fallback (unbounded reservation, allocation
///   failure) stays interpreted, fail-closed.
///
/// A memory-less run has no window to serve — no bitmap.
fn tierup_servable(cand: Option<&Mem>, root: Option<&Mem>) -> bool {
    match (cand, root) {
        (Some(c), Some(r)) => {
            std::sync::Arc::ptr_eq(&c.back, &r.back) || c.flat_win_base().is_some()
        }
        _ => false,
    }
}

/// #926 slice 2 — the cooperative driver's cross-tier **bounce** body: an emitted tier-up region, mid
/// run, calls back to an interp-resident leaf `target`. Resolves `target` through the caller's dispatch
/// `table` (masked, exactly as `Op::CallIndirect`), marshals the i64 scratch `io` per the callee's
/// signature, and drives it to completion on a nested interpretation over the caller's `mem`/`host`/
/// `fuel`. The `fibers`/`fiber_meta` the caller passes select the registry (#926 slice 2g): a **tier-up
/// region**'s callbacks run against the **run-level** registry (`fiber_meta = Some(shadow-SP/freeze
/// halves)` — a parked fiber persists for the run, the same registry `step_vcpu` mirrors its parallel
/// arrays into), while a surfaced **`Jit.invoke`** unit's callbacks run against an **invoke-confined**
/// registry (`fiber_meta = None`, the transient loop-local scope `run_invoke` and `Vcpu::bounce_call`'s
/// invoke branch use — its fibers die when the invoke resolves). The multi-task analogue of
/// [`Vcpu::bounce_call`], differing only in that the window/powerbox/table come from the *paused task's*
/// env (resolved by [`CoopRun::bounce`]).
#[allow(clippy::too_many_arguments)] // an inherently many-input dispatch shim; a config struct would obscure it
fn coop_bounce(
    source: &ModuleSource,
    table: &SharedSlots,
    fuel: &mut u64,
    mem: &mut Option<Mem>,
    host: &mut HostCell,
    fibers: &mut FiberCell,
    fiber_meta: Option<BounceRunCtx<'_>>,
    target: u32,
    io: &mut [i64],
    beneath: Option<&Beneath<'_>>,
) -> Result<usize, Trap> {
    step(fuel, None)?; // fuel unification: the dispatch-site safepoint
    let slot = (target as usize) & (table.len() - 1);
    let ts = table.slot(slot);
    if ts.module == super::TABLE_EMPTY {
        return Err(Trap::IndirectCallType);
    }
    let tm = source.get(ts.module as usize).ok_or(Trap::Malformed)?;
    let (cp, cr) = tm.sigs[ts.func as usize].clone();
    if cp.len() > io.len() || cr.len() > io.len() {
        return Err(Trap::Malformed); // scratch too small — a mis-marshalled host call
    }
    // The i64-slot transport carries scalars only; a v128-sig target can never have been given a
    // trampoline (the host gates that at open) — a bounce naming one is a mis-wired host.
    let scalar =
        |t: &ValType| matches!(t, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64);
    if !cp.iter().all(scalar) || !cr.iter().all(scalar) {
        return Err(Trap::Malformed);
    }
    let args: Vec<Value> = cp
        .iter()
        .zip(io.iter())
        .map(|(ty, s)| slot_to_val(*ty, *s))
        .collect();
    let mut vm = Vm::new(&tm, ts.func as usize, &args)?;
    vm.module = ts.module as usize;
    // #1660/#1627: `beneath` is `Some` only when the emitted frames under this bounce have spilled.
    let vals = drive_nested(
        source, table, vm, fuel, mem, host, fibers, fiber_meta, beneath,
    )?;
    for (i, v) in vals.iter().enumerate() {
        io[i] = val_to_slot(*v);
    }
    Ok(cr.len())
}

/// THREADS.md step 4c — a **native futex**, the parallel driver's stand-in for wasm
/// `memory.atomic.wait`/`notify`. A parked waiter enqueues a token (its own `woken` flag + `Condvar`)
/// under its address key; `notify` wakes up to `count` of them FIFO. The compare-and-park runs under
/// `buckets`, so a concurrent `notify` cannot slip between a waiter reading the futex word and parking
/// (the std-sync analogue of the kernel's per-bucket futex lock) — no lost wakeups. In real wasm this
/// role is played by `memory.atomic.wait`/`notify` directly; here it serves the cooperative oracle's
/// same `wait`/`notify` semantics for genuinely parallel vCPUs.
#[derive(Default)]
struct Futex {
    buckets: std::sync::Mutex<
        std::collections::HashMap<u64, std::collections::VecDeque<std::sync::Arc<Waiter>>>,
    >,
}

struct Waiter {
    woken: std::sync::Mutex<bool>,
    cv: std::sync::Condvar,
    /// The waiter's domain ([`ParDomain`], by address), so a domain's death wakes only its members.
    domain: usize,
}

impl Futex {
    /// `memory.wait`: compare the futex word at `base` to `expected` under the bucket lock; if it
    /// already differs, return `WAIT_NOT_EQUAL` without parking (the fast path). Otherwise enqueue a
    /// token and park on it until `notify` wakes it (`WAIT_WOKEN`) or `timeout` ns elapse
    /// (`WAIT_TIMED_OUT`). Mirrors the cooperative `BlockedWait` arm; the per-token flag absorbs
    /// spurious condvar wakeups.
    /// `timeout` is the guest's own, **unclamped** (#1641) — two waiters asking 30 s and 20 s
    /// must not tie at a 10 s cap and wake in the wrong order.
    ///
    /// `None` (an infinite wait) is the one case this driver still backstops with [`MAX_WAIT`],
    /// and it is a **known divergence** from the oracle in both directions (#1652): a satisfiable
    /// wait longer than the cap returns a spurious `WAIT_TIMED_OUT` where the oracle waits it out,
    /// and an unsatisfiable one returns `WAIT_TIMED_OUT` at 10 s where the oracle faults. It stays
    /// only because this driver runs each vCPU on its own OS thread with no cross-thread park
    /// census, so dropping the backstop outright would turn the second case into a hang.
    fn wait(
        &self,
        domain: &ParDomain,
        mem: &Mem,
        base: u64,
        expected: u64,
        width: u32,
        timeout: Option<u64>,
    ) -> i32 {
        let waiter = {
            let mut buckets = self.buckets.lock().unwrap();
            // A domain that died since this vCPU's last safepoint: its kill scanned the buckets
            // before this waiter was in them ([`ParDomain::kill`] records the death, *then* takes
            // this lock to wake), so ask here, under the lock, and return as its wake would have.
            if domain.dead().is_some() {
                return super::WAIT_WOKEN;
            }
            // Compare-under-lock: the futex word lives in the shared backing (`atomic_value` reads it).
            if mem.atomic_value(base, width) != expected {
                return super::WAIT_NOT_EQUAL;
            }
            let w = std::sync::Arc::new(Waiter {
                woken: std::sync::Mutex::new(false),
                cv: std::sync::Condvar::new(),
                domain: domain as *const ParDomain as usize,
            });
            buckets
                .entry(base)
                .or_default()
                .push_back(std::sync::Arc::clone(&w));
            w
        };
        // Park on our own token (the bucket lock is released): woken by `notify`, or timed out.
        let timeout = timeout.map_or(super::MAX_WAIT, std::time::Duration::from_nanos);
        let (flag, res) = waiter
            .cv
            .wait_timeout_while(waiter.woken.lock().unwrap(), timeout, |w| !*w)
            .unwrap();
        let woken = *flag;
        drop(flag);
        if woken {
            super::WAIT_WOKEN
        } else {
            debug_assert!(res.timed_out());
            // Timed out: de-enqueue our (possibly still-parked) token so a later `notify` skips it.
            let mut buckets = self.buckets.lock().unwrap();
            if let Some(q) = buckets.get_mut(&base) {
                q.retain(|x| !std::sync::Arc::ptr_eq(x, &waiter));
            }
            super::WAIT_TIMED_OUT
        }
    }

    /// `memory.notify`: wake up to `count` waiters parked on `base`, FIFO, and return how many were
    /// woken (mirrors the cooperative `Notify` arm's count; the guest typically ignores it).
    fn notify(&self, base: u64, count: i32) -> i32 {
        let want = count as u32;
        let mut buckets = self.buckets.lock().unwrap();
        let mut woken = 0u32;
        if let Some(q) = buckets.get_mut(&base) {
            while woken < want {
                let Some(w) = q.pop_front() else { break };
                *w.woken.lock().unwrap() = true;
                w.cv.notify_one();
                woken += 1;
            }
        }
        woken as i32
    }

    /// Wake every waiter of `domain` (it died — see [`ParDomain::kill`]); each returns `WAIT_WOKEN`
    /// and its vCPU observes the death at its next safepoint.
    fn wake_domain(&self, domain: &ParDomain) {
        let id = domain as *const ParDomain as usize;
        let mut buckets = self.buckets.lock().unwrap();
        for q in buckets.values_mut() {
            q.retain(|w| {
                if w.domain != id {
                    return true;
                }
                *w.woken.lock().unwrap() = true;
                w.cv.notify_one();
                false
            });
        }
    }
}

#[cfg(test)]
mod par_futex_tests {
    use super::*;

    /// A domain that died between a member's loop-top `dead()` check and its futex enqueue: the
    /// kill's wake scanned the buckets before the waiter was in them. The waiter must see the death
    /// under the bucket lock, not sleep out its timeout (`MAX_WAIT`, 10 s, for an infinite wait).
    #[test]
    fn a_wait_in_a_domain_that_died_before_it_enqueued_returns_at_once() {
        let mem = Mem::with_reservation(temen_ir::DEFAULT_RESERVED_LOG2, 16, None);
        let base = mem.prepare_wait(16384, IntTy::I32).expect("in bounds");
        let reg = ThreadRegistry::new();
        let dom = ParDomain::default();
        dom.kill(&Trap::ThreadFault, &reg);
        let t0 = std::time::Instant::now();
        let r = reg.futex.wait(&dom, &mem, base, 0, 4, None);
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(2),
            "slept {:?} in a dead domain",
            t0.elapsed()
        );
        assert_eq!(r, super::super::WAIT_WOKEN);
    }
}

/// One **domain** of the parallel driver (DESIGN.md §12): the root and its `thread.spawn` threads,
/// or a §14 confined child or fork twin and its threads — the world that shares one window and
/// powerbox. It holds the domain's fiber registry (#1761) and its death: a member's trap is
/// terminal for the whole domain (I37, the cooperative `teardown_domains` rule), so the first trap
/// is recorded here and every other member dies with it at its next safepoint — the per-quantum
/// check, or a futex wait / join this kill wakes. A sibling's trap thereby becomes the root's result.
#[derive(Default)]
struct ParDomain {
    fibers: SharedFibers,
    dead: std::sync::Mutex<Option<Trap>>,
}

impl ParDomain {
    /// This domain's trap, once a member has died.
    fn dead(&self) -> Option<Trap> {
        *self.dead.lock_unpoisoned()
    }

    /// A member trapped with `t`: record it (the first trap wins) and wake every member blocked in a
    /// futex wait or a join, so each observes the death.
    fn kill(&self, t: &Trap, reg: &ThreadRegistry) {
        {
            let mut d = self.dead.lock_unpoisoned();
            if d.is_some() {
                return;
            }
            *d = Some(*t);
        }
        reg.futex.wake_domain(self);
        {
            let _g = reg.done.lock().unwrap_or_else(|e| e.into_inner());
            reg.woken.notify_all();
        }
        reg.wake_fork_waiters();
    }
}

/// THREADS.md step 4c — the cross-thread `thread.spawn`/`join` rendezvous for the parallel driver.
/// The cooperative `drive` keeps its child vCPUs in one `tasks` vec and wakes joiners inline; the
/// parallel driver runs each vCPU on its **own OS thread**, so a joiner blocks here on a `Condvar`
/// until the child it named publishes its result. One `id` namespace across the whole run (handed out
/// by `next_id`); a child's result (value-or-trap) is delivered to the lowest-index waiter via the
/// `done` map. `live` mirrors the cooperative `MAX_VCPUS` anti-bomb gate across threads. `futex` serves
/// the guest's `memory.wait`/`notify` across threads.
struct ThreadRegistry {
    done: std::sync::Mutex<std::collections::HashMap<u64, Result<Vec<Value>, Trap>>>,
    woken: std::sync::Condvar,
    next_id: std::sync::atomic::AtomicU64,
    live: std::sync::atomic::AtomicUsize,
    futex: Futex,
    /// #748 — the personality **fork-twin table** for the parallel driver's `ForkSelf`/`ReapWait`
    /// arms: `(exited pids, generation)`. Exited pids are permanent (never removed — pids are
    /// per-run unique), so a `waitpid(pid)` waiter can never miss its wake. The generation bumps
    /// once per twin exit so an **any-child** waiter waits for "an exit newer than the ones my
    /// re-issued waitpid already consumed" — the condvar analogue of the cooperative driver's
    /// consumed-Done-twin prune: a stale exit can neither re-wake forever nor be lost (an exit
    /// before the op's table check left a zombie the re-issue reaps; one after bumps past the
    /// waiter's recorded generation).
    fork_exits: std::sync::Mutex<(std::collections::HashSet<i64>, u64)>,
    fork_woken: std::sync::Condvar,
    /// The next personality twin pid. Starts at 2: the root personality is pid 1 (the same
    /// no-collision shape as the cooperative driver's `task index + 1`).
    next_fork_pid: std::sync::atomic::AtomicI64,
    /// This registry, for the personality doors [`wire_parallel_doors`] installs: they outlive the
    /// run on the host's signal source, so they hold it weakly.
    me: std::sync::Weak<ThreadRegistry>,
}

impl ThreadRegistry {
    fn new() -> std::sync::Arc<ThreadRegistry> {
        std::sync::Arc::new_cyclic(|me| ThreadRegistry {
            done: std::sync::Mutex::new(std::collections::HashMap::new()),
            woken: std::sync::Condvar::new(),
            next_id: std::sync::atomic::AtomicU64::new(0),
            live: std::sync::atomic::AtomicUsize::new(0),
            futex: Futex::default(),
            fork_exits: std::sync::Mutex::new((std::collections::HashSet::new(), 0)),
            fork_woken: std::sync::Condvar::new(),
            next_fork_pid: std::sync::atomic::AtomicI64::new(2),
            me: me.clone(),
        })
    }

    /// #748 — a fork twin's OS thread finished (its exit hooks have already fired, so the
    /// personality table shows the zombie): record the exit and wake every reap waiter.
    fn publish_fork_exit(&self, pid: i64) {
        let mut g = self.fork_exits.lock().unwrap_or_else(|e| e.into_inner());
        g.0.insert(pid);
        g.1 += 1;
        self.fork_woken.notify_all();
    }

    /// Something other than an exit a reap waiter must re-check (a kill, a raise, a child
    /// stop/continue): wake them all under the table lock, so a waiter between its check and its
    /// wait cannot miss it.
    fn wake_fork_waiters(&self) {
        let _g = self.fork_exits.lock().unwrap_or_else(|e| e.into_inner());
        self.fork_woken.notify_all();
    }

    /// #748 — block this vCPU's OS thread until `child` (`Some(pid)`) has published its exit, or
    /// (`None`, the any-child wait) until the exit generation exceeds `last_gen`. Returns the
    /// current generation for the caller to carry into its next wait; either way the caller's
    /// rewound `waitpid` re-executes against the updated personality table.
    ///
    /// `interrupted` asks, under the table lock, whether anything else this park must not sleep
    /// through has happened (see the `ReapWait` arm); each such event rings
    /// [`Self::wake_fork_waiters`] after raising its fact, so the pair cannot lose it.
    fn wait_fork_exit(
        &self,
        child: Option<i64>,
        last_gen: u64,
        interrupted: impl Fn() -> bool,
    ) -> u64 {
        let mut g = self.fork_exits.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            let ready = match child {
                Some(pid) => g.0.contains(&pid),
                None => g.1 > last_gen,
            };
            if ready || interrupted() {
                return g.1;
            }

            g = self.fork_woken.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// A spawned vCPU finished: publish its result and wake any joiner parked on it.
    fn publish(&self, id: u64, res: Result<Vec<Value>, Trap>) {
        self.done.lock().unwrap().insert(id, res);
        self.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        self.woken.notify_all();
    }

    /// Block until vCPU `id` has published, then take (consume) its result — the parallel analogue of
    /// the cooperative `BlockedJoin` wakeup. A child trap is returned to propagate to the joiner.
    /// A joiner whose own `domain` dies while it waits completes with the domain's trap.
    fn join(&self, id: u64, domain: &ParDomain) -> Result<Vec<Value>, Trap> {
        let mut g = self.done.lock().unwrap();
        loop {
            if let Some(r) = g.remove(&id) {
                return r;
            }
            if let Some(t) = domain.dead() {
                return Err(t);
            }
            g = self.woken.wait(g).unwrap();
        }
    }
}

/// #1246 — wire a parallel domain's default-action TERMINATE door: the personality's `set_kill`
/// closure stores into this host's `term_flag`, the atomic every vCPU of the domain polls per op in
/// [`Vm::resume`] and traps on. Called on the root host and on each fork twin's freshly-minted host;
/// a host with no signal personality is a no-op.
///
/// A running vCPU needs nothing more, but a **blocked** one polls nothing: one parked in a futex
/// `wait` slept out its timeout (10 s for an infinite wait), and one parked in a blocking `waitpid`
/// until a child exited — forever, if that child was parked too. So the deferred doors wake them:
/// a terminate is this domain's death, the same `ThreadFault` the per-op poll raises
/// ([`ParDomain::kill`] wakes its futex waiters, joiners and reap waiters), and a deliverable raise
/// or a child's stop/continue re-checks every reap waiter (the `ReapWait` arm's predicate).
fn wire_parallel_doors(
    host: &std::sync::Arc<std::sync::Mutex<Host>>,
    reg: &ThreadRegistry,
    domain: &std::sync::Arc<ParDomain>,
) {
    let (term_flag, source) = {
        let hg = host.lock_unpoisoned();
        (hg.term_flag.clone(), hg.signal_poll().map(|(_, s)| s))
    };
    if let Some(source) = source {
        // #1259 — the `term_flag` write is the INLINE apply (fired synchronously by the personality),
        // not a deferred wake: a per-op-polling parallel vCPU needs no scheduler notify, and applying it
        // in program order keeps it reorder-free by construction.
        source.set_kill_apply(std::sync::Arc::new(move || {
            term_flag.store(true, std::sync::atomic::Ordering::SeqCst);
        }));
        let (r, d) = (reg.me.clone(), std::sync::Arc::downgrade(domain));
        source.set_kill(std::sync::Arc::new(move || {
            if let (Some(r), Some(d)) = (r.upgrade(), d.upgrade()) {
                d.kill(&Trap::ThreadFault, &r);
            }
        }));
        let ring = |r: std::sync::Weak<ThreadRegistry>| -> std::sync::Arc<dyn Fn() + Send + Sync> {
            std::sync::Arc::new(move || {
                if let Some(r) = r.upgrade() {
                    r.wake_fork_waiters();
                }
            })
        };
        source.set_wake(ring(reg.me.clone()));
        source.set_chld_wake(ring(reg.me.clone()));
    }
}

/// THREADS.md step 4c — the **parallel** driver (the host-selected `Parallel` mode). One guest's vCPUs
/// run on **separate OS threads** sharing **one** `Region::shared` window, instead of the cooperative
/// `drive`'s single-thread `tasks` loop. `std::thread::scope` borrows the `&Domain` (which is `Sync`)
/// and the `&ThreadRegistry` into each child and joins every still-running thread before returning, so
/// the window is quiescent for the snapshot. The root runs on the calling thread (it never
/// `atomic.wait`s — `join` blocks on a `Condvar`, sidestepping the browser main-thread-wait wrinkle).
/// Returns the root's result and its (now-quiescent) `Mem` for capture. Scope: the pure-threads subset
/// (`thread.spawn`/`join` + atomics); other multi-vCPU events fail closed (see
/// [`compile_and_run_capture_over_parallel`]).
fn drive_parallel(
    dom: Domain,
    entry: FuncIdx,
    args: &[Value],
    fuel: u64,
    mem: Option<Mem>,
    host: &mut Host,
) -> (Result<Vec<Value>, Trap>, Option<Mem>) {
    let root_vt = match VTask::new(&dom.source.primary(), entry as usize, args) {
        Ok(v) => v,
        Err(t) => return (Err(t), mem),
    };
    let reg = ThreadRegistry::new();
    // Share the caller's powerbox across every vCPU thread, then hand it back (so the caller reads its
    // stdout / final state). `Arc` (not a scope-borrowed `&Mutex`) because a personality fork twin
    // (#748) carries its OWN host cell minted at runtime — per-process powerboxes, the parallel twin
    // of the cooperative driver's `extra_envs`. `scope` joins all vCPUs before returning, so every
    // clone is dropped and the unwrap below is the sole owner.
    let shared = std::sync::Arc::new(std::sync::Mutex::new(std::mem::take(host)));
    // #1246 — wire the root domain's terminate door (a guest that kills its own group, or is killed
    // by an embedder, dies at its next per-op poll). Fork twins get theirs at mint (the `ForkSelf` arm).
    let root_domain = std::sync::Arc::new(ParDomain::default());
    wire_parallel_doors(&shared, &reg, &root_domain);
    let out = std::thread::scope(|scope| {
        run_vcpu_parallel(
            scope,
            &dom,
            &reg,
            std::sync::Arc::clone(&shared),
            root_domain,
            None,
            root_vt,
            mem,
            fuel,
        )
    });
    *host = match std::sync::Arc::try_unwrap(shared) {
        Ok(m) => m.into_inner().unwrap_or_else(|e| e.into_inner()),
        // Unreachable in practice (the scope joined every holder), but never lose the powerbox.
        Err(a) => std::mem::take(&mut *a.lock().unwrap_or_else(|e| e.into_inner())),
    };
    out
}

/// Run process `pid` of a parallel run on its own OS thread, over its own powerbox cell, window and
/// table — a fork twin continuing its parent's image, or a spawned process starting a new one — and
/// retire it when it ends: its pipe ends released (EOF/`-EPIPE` for peers) and its exit hooks fired
/// once with the reap-encoded status (Live → Zombie in the personality table), THEN its exit
/// published, so a woken waiter's re-issued `waitpid` finds the zombie already there.
#[allow(clippy::too_many_arguments)]
fn start_process<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    dom: &'env Domain,
    reg: &'env ThreadRegistry,
    host: std::sync::Arc<std::sync::Mutex<Host>>,
    tbl: Option<std::sync::Arc<SharedSlots>>,
    vt: VTask,
    mem: Option<Mem>,
    fuel: u64,
    pid: i64,
) {
    // #1246 — the process's own terminate door, so a SIGKILL/SIGTERM to it sets its `term_flag` and its
    // resume loop traps at the next op (its parent's `waitpid` then reaps the WIFSIGNALED death).
    let domain = std::sync::Arc::new(ParDomain::default());
    wire_parallel_doors(&host, reg, &domain);
    let hooks_host = std::sync::Arc::clone(&host);
    scope.spawn(move || {
        let (r, _m) = run_vcpu_parallel(scope, dom, reg, host, domain, tbl, vt, mem, fuel);
        let status = super::reap_status(&r);
        let hooks = {
            let g = hooks_host.lock_unpoisoned();
            g.release_pipe_ends();
            g.exit_hooks.clone()
        };
        for h in hooks {
            h(status);
        }
        reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        reg.publish_fork_exit(pid);
    });
}

/// Run one vCPU of the parallel driver to completion on **this** OS thread, fanning each
/// `thread.spawn` onto a fresh scoped thread (over a `fork_for_thread` view of the shared window) and
/// blocking each `thread.join` on the [`ThreadRegistry`]. Mirrors the cooperative `drive`'s `Spawn` /
/// `Join` / `Done` arms, one vCPU at a time. Returns this vCPU's result and the `Mem` it owned (the
/// root's is the one captured; a child's is dropped, its bytes already live in the shared backing).
#[allow(clippy::too_many_arguments)] // an internal driver entry: the args ARE the vCPU's identity
fn run_vcpu_parallel<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    dom: &'env Domain,
    reg: &'env ThreadRegistry,
    host: std::sync::Arc<std::sync::Mutex<Host>>,
    domain: std::sync::Arc<ParDomain>,
    tbl: Option<std::sync::Arc<SharedSlots>>,
    vt: VTask,
    mem: Option<Mem>,
    fuel: u64,
) -> (Result<Vec<Value>, Trap>, Option<Mem>) {
    let d = std::sync::Arc::clone(&domain);
    let out = run_vcpu_parallel_body(scope, dom, reg, host, domain, tbl, vt, mem, fuel);
    // A member's trap is terminal for its domain (I37): the others die with it, so the scope that
    // joins them — and the run — ends instead of waiting on vCPUs that would never finish.
    if let Err(t) = &out.0 {
        d.kill(t, reg);
    }
    out
}

/// [`run_vcpu_parallel`]'s loop, without the domain kill on a trap.
#[allow(clippy::too_many_arguments)] // an internal driver entry: the args ARE the vCPU's identity
fn run_vcpu_parallel_body<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    dom: &'env Domain,
    reg: &'env ThreadRegistry,
    // This vCPU's powerbox cell: the run's shared host for the root and its `thread.spawn` siblings
    // (4c-host), or a fork twin's OWN forked powerbox (#748) — its own spawned threads then share
    // *that*. Owned `Arc` so a twin's runtime-minted cell moves into its scoped thread cleanly.
    host: std::sync::Arc<std::sync::Mutex<Host>>,
    // This vCPU's domain (its fiber registry, #1761, and its death), shared like the powerbox: by
    // the root and its `thread.spawn` siblings. A fork twin or a §14 confined child is its own
    // process and starts a fresh one.
    domain: std::sync::Arc<ParDomain>,
    // This vCPU's dispatch table when it is not the domain's shared `dom.table`: an `exec_module`
    // image-replace (#748 rung 2) installs the command's natural table here, and a fork twin (same
    // image as its parent) or `thread.spawn` child (same image as its spawner) inherits its
    // parent's. `Arc` so inheritance is a clone, not a rebuild.
    mut tbl: Option<std::sync::Arc<SharedSlots>>,
    mut vt: VTask,
    mut mem: Option<Mem>,
    mut fuel: u64,
) -> (Result<Vec<Value>, Trap>, Option<Mem>) {
    // handle (index) → global vCPU id of a `thread.spawn` child (shares the cooperative handle scheme).
    let mut threads: Vec<Option<u64>> = Vec::new();
    // #748 — the exit generation this vCPU's any-child `waitpid` has consumed up to (see
    // [`ThreadRegistry::wait_fork_exit`]).
    let mut fork_gen: u64 = 0;
    loop {
        // A sibling's trap killed this domain (I37): die with it.
        if let Some(t) = domain.dead() {
            return (Err(t), mem);
        }
        let mut ctx = RunCtx {
            table: tbl.as_deref().unwrap_or(&dom.table),
            fuel: &mut fuel,
            mem: &mut mem,
            durable: false,
            // The powerbox is **shared** by every vCPU of the run (4c-host): `call.cap` takes the lock
            // only for its own dispatch, so compute/atomics/futex between calls stay lock-free.
            host: HostCell::Shared(&host),
        };
        // NLL ends `ctx`'s borrows of `mem`/`fuel` at this call, so the arms below may touch them.
        let stop = step_vcpu(
            &mut vt,
            &mut FiberCell::Shared(&domain.fibers),
            dom,
            &mut ctx,
            COOP_QUANTUM,
            false, // OS-thread parallel driver: blocking `cont.resume.block` idle is a follow-up (I48)
            // The OS preempts real threads; the quantum is only this vCPU's safepoint for observing a
            // sibling's trap (the loop-top check) when it never blocks.
            true,
        );
        match stop {
            // §3.6 (I36 slice 2): the serve/call pair runs only on the cooperative driver
            // (`drive`); a serving module never reaches the parallel driver (the qualification veto
            // refuses svc + threads together) — fail closed if it somehow does, rather than park
            // unwakeably. (`child_offer` was grouped here until #1566; see the arm above.) I48 `BlockOnFiber` is likewise cooperative-only
            // (this path passes `cooperative: false`), so it never arises here — grouped in.
            // §3.6 `child_offer` (op 14) — #1566. Unlike its neighbours below, this one **is**
            // reachable here: the qualification veto that keeps a serving module off this driver
            // covers the svc ops, and `child_offer` is an `Instantiator` op, so a single-vCPU guest
            // that never spawns a thread reaches it with nothing refusing first. The capability
            // itself is genuinely unavailable — minting a live offer needs the CALLEE's powerbox,
            // and this driver moves each child's `Host` into that child's own OS thread, publishing
            // only its result through `reg`, so the parent has no path to it (the same reason the
            // browser's per-Worker driver fails closed on a stashed powerbox).
            //
            // But "unavailable" is a value, not a trap. The cooperative driver answers `-EINVAL`
            // for a child it cannot resolve, so this answers the same: one op, one answer, whichever
            // loop is driving (INVARIANTS #9), and a guest probing a stale child handle is not
            // killed for the driver it happened to land on (#5 — errors are values, traps are for
            // forgery). It used to be grouped into the fail-closed trap below on the premise that it
            // could not arrive.
            Ok(VcpuStop::ChildOffer { dst, .. }) => {
                vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
            }
            // #1952 — a fiber's pipe or stdin op that must wait parks the fiber alone, and this
            // vCPU's resumer runs on (the driver cannot idle a blocking resume: the `FIBER_PARKED`
            // poll). A vanished pipe's op just re-runs, and fails closed.
            Ok(
                stop @ (VcpuStop::PipeRead { .. }
                | VcpuStop::PipeWrite { .. }
                | VcpuStop::StdinPark),
            ) if vt.active_id != ROOT_FIBER => {
                let parked = {
                    let g = host.lock_unpoisoned();
                    HostWait::of(&stop, &g).map(|on| {
                        let ready = on.ready(&g);
                        (on, ready)
                    })
                };
                if let Some((on, ready)) = parked {
                    FiberCell::Shared(&domain.fibers).with(|f, sp, _| {
                        park_fiber_on_host(&mut vt, f, sp, &mut mem, false, false, on, ready)
                    });
                }
            }
            Ok(VcpuStop::LiveCall { .. })
            | Ok(VcpuStop::SvcWait)
            | Ok(VcpuStop::CloneCaller { .. })
            | Ok(VcpuStop::Reap { .. })
            | Ok(VcpuStop::BlockOnFiber { .. }) => return (Err(Trap::ThreadFault), mem),
            // The safepoint quantum elapsed: back to the loop top's domain check, then resume.
            Ok(VcpuStop::Preempted) => {}
            Err(trap) => return (Err(trap), mem),
            Ok(VcpuStop::Done(vals)) => return (Ok(vals), mem),
            // Tier-up is only enabled on the browser `Vcpu::run` path (`with_jit_eligible`).
            Ok(VcpuStop::TierUp { .. }) => unreachable!("tier-up not enabled on the native driver"),
            // #1146 (deeper) — blocking `Stream{In}` read on an exhausted stdin (the op was rewound):
            // block this OS thread until bytes arrive, a default-action TERMINATE flips `term_flag`
            // (the re-run then dies at its per-op safepoint — invariant 14's terminate axis), or a
            // deliverable non-`SA_RESTART` signal interrupts it (latch `set_sig_interrupt` and break so
            // the re-run completes `-EINTR` at the stdin park site in `resume`). The stdin twin of the
            // pipe poll below — each blocked OS thread observes its own interrupt, no central sweep.
            Ok(VcpuStop::StdinPark) => {
                let term_flag = host.lock_unpoisoned().term_flag.clone();
                while !host.lock_unpoisoned().stdin_ready() {
                    // A default-action TERMINATE, or a sibling's trap killing this domain (I37): the
                    // rewound op re-runs into the safepoint that ends this vCPU.
                    if term_flag.load(std::sync::atomic::Ordering::SeqCst)
                        || domain.dead().is_some()
                    {
                        break;
                    }
                    if host.lock_unpoisoned().park_interrupted() {
                        host.lock_unpoisoned().set_sig_interrupt();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
            Ok(VcpuStop::Spawn {
                func,
                sp,
                arg,
                dst,
                module,
            }) => {
                // Module-aware, as the cooperative arm: `func` is the spawning frame's module's index.
                let Some(cm) = dom.source.get(module as usize) else {
                    return (Err(Trap::Malformed), mem);
                };
                if func as usize >= cm.progs.len() {
                    return (Err(Trap::Malformed), mem);
                }
                // Cross-thread anti-bomb gate (mirrors the cooperative `live >= MAX_VCPUS`).
                if reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1
                    > super::MAX_VCPUS
                {
                    reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    return (Err(Trap::ThreadFault), mem);
                }
                let id = reg
                    .next_id
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let child_vt =
                    match VTask::new(&cm, func as usize, &[Value::I64(sp), Value::I64(arg)]) {
                        Ok(mut v) => {
                            v.active.module = module as usize;
                            v.active.home = module as usize;
                            // §12 seed the child's `vcpu.tls` to its dense id (root = 0; ids start
                            // at 0 for the first child) — the cooperative `Spawn` arm's seeding.
                            v.active.tls = id as i64 + 1;
                            v
                        }
                        Err(t) => return (Err(t), mem),
                    };
                // The child runs over its own `Mem` view of the **same** shared backing (real atomics)
                // and SHARES this vCPU's powerbox cell (a thread, not a process — cf. `ForkSelf`).
                let child_mem = mem.as_ref().map(|m| m.fork_for_thread());
                let child_host = std::sync::Arc::clone(&host);
                let child_tbl = tbl.clone();
                let child_domain = std::sync::Arc::clone(&domain);
                scope.spawn(move || {
                    let (r, _m) = run_vcpu_parallel(
                        scope,
                        dom,
                        reg,
                        child_host,
                        child_domain,
                        child_tbl,
                        child_vt,
                        child_mem,
                        fuel,
                    );
                    reg.publish(id, r);
                });
                let handle = threads.len() as i32;
                threads.push(Some(id));
                vt.active.set(dst, Reg::from_i32(handle));
            }
            Ok(VcpuStop::Join { handle, dst }) => {
                // Single join: the handle is now spent.
                let id = match super::take_child(&mut threads, handle) {
                    Ok(id) => id,
                    Err(t) => return (Err(t), mem),
                };
                match reg.join(id, &domain) {
                    // A joined child's first result value lands in the joiner's `dst`.
                    Ok(vals) => {
                        let v = vals.first().copied().unwrap_or(Value::I64(0));
                        vt.active.set(dst, Reg::from_value(v));
                    }
                    // A child trap propagates: the joiner completes with the same trap.
                    Err(t) => return (Err(t), mem),
                }
            }
            Ok(VcpuStop::ForkSelf { dst }) => {
                // #748 rung 0 — personality `fork()` on the parallel driver: duplicate THIS vCPU
                // into a twin **OS thread** over a PRIVATE window copy with its OWN forked powerbox
                // — a process, not a 4c thread (cf. `Spawn`'s `fork_for_thread` shared view). The
                // parent keeps running with the twin's pid; the twin resumes at the same op (pc
                // already advanced) with the return-twice `0`. Bare gate + `-EAGAIN` refusal mirror
                // the cooperative arm (invariant 5: a value, never a hang).
                let bare = threads.iter().all(|t| t.is_none())
                    && vt.active_id == ROOT_FIBER
                    && vt.chain.is_empty();
                // Cross-thread anti-bomb gate (the `Spawn` arm's), released on any refusal.
                let admitted = bare
                    && reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                        < super::MAX_VCPUS;
                if bare && !admitted {
                    reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                }
                let forked: Option<(Option<Mem>, Host, i64)> = if admitted {
                    let twin_pid = reg
                        .next_fork_pid
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let built = (|| {
                        let tm = match mem.as_ref() {
                            Some(m) => Some(m.fork_private()?),
                            None => None,
                        };
                        let th = host.lock_unpoisoned().fork_powerbox(twin_pid as u64)?;
                        Some((tm, th, twin_pid))
                    })();
                    if built.is_none() {
                        reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    built
                } else {
                    None
                };
                match forked {
                    None => vt.active.set(dst, Reg::from_i64(super::EAGAIN)),
                    Some((twin_mem, twin_host, twin_pid)) => {
                        // The twin's OWN park door (the #1112 lesson): `fork_powerbox` mints its
                        // personality with no park delegate, and without one the twin's own
                        // `fork()`/`waitpid()` cannot park (`-ENOSYS`/the ECHILD poll).
                        twin_host.wire_park_door();
                        let mut twin_active = vt.active.clone();
                        twin_active.set(dst, Reg::from_i64(0));
                        let twin_root_sp =
                            twin_active.durable_region_base + super::REGION_HEADER_LEN; // its context's empty frame base
                        let twin_vt = VTask {
                            active: twin_active,
                            active_id: ROOT_FIBER,
                            chain: Vec::new(),
                            root_shadow_sp: twin_root_sp,
                            active_invoke: None,
                        };
                        let twin_host = std::sync::Arc::new(std::sync::Mutex::new(twin_host));
                        // The twin continues the SAME image as its parent, so it dispatches over the
                        // same table (post-exec parents included — cf. the coop arm's fresh primary
                        // table, which a bare pre-exec caller also resolves to).
                        let twin_tbl = tbl.clone();
                        start_process(
                            scope, dom, reg, twin_host, twin_tbl, twin_vt, twin_mem, fuel, twin_pid,
                        );
                        vt.active.set(dst, Reg::from_i64(twin_pid));
                    }
                }
            }
            Ok(VcpuStop::SpawnSelf { cmd, plan, dst }) => {
                // A personality `posix_spawn` on the parallel driver: mint a process as the fork
                // arm does (its pid from the run's counter, its own OS thread, retired through its
                // exit hooks) and build its image as the exec arm does, with nothing of the caller
                // copied. The caller runs on with the pid, `-EAGAIN` when none could be minted; a
                // process whose image cannot be built exits as it is born. The tree-walker's gate
                // first: a request from a fiber keeps its placeholder, and a serve handler is not a
                // clean root.
                if vt.active_id != ROOT_FIBER {
                    vt.active.set(dst, Reg::from_i64(temen_ir::errno::ENOSYS));
                    continue;
                }
                if vt.active.serve_ticket.is_some() {
                    vt.active.set(dst, Reg::from_i64(super::EINVAL));
                    continue;
                }
                let admitted =
                    reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < super::MAX_VCPUS;
                let twin = if admitted {
                    let pid = reg
                        .next_fork_pid
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let twin = host.lock_unpoisoned().spawn_powerbox(pid as u64, plan);
                    twin.map(|t| (pid, t))
                } else {
                    None
                };
                match twin {
                    None => {
                        reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                        vt.active.set(dst, Reg::from_i64(super::EAGAIN));
                    }
                    Some((pid, mut twin)) => {
                        let built = exec_image_build(
                            &mut twin,
                            mem.as_ref(),
                            dom,
                            cmd,
                            0,
                            0,
                            0,
                            0,
                            true,
                            None,
                        );
                        match built {
                            Ok(built) => {
                                built.host.wire_park_door();
                                let child_host =
                                    std::sync::Arc::new(std::sync::Mutex::new(built.host));
                                let table = Some(std::sync::Arc::new(built.table));
                                let child_mem = Some(built.mem);
                                start_process(
                                    scope, dom, reg, child_host, table, built.vt, child_mem, fuel,
                                    pid,
                                );
                            }
                            Err(_) => {
                                let _ = twin.spawn_failed(super::SPAWN_EXEC_FAILED);
                                reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
                                reg.publish_fork_exit(pid);
                            }
                        }
                        vt.active.set(dst, Reg::from_i64(pid));
                    }
                }
            }
            Ok(VcpuStop::ReapWait { child }) => {
                // #748 rung 1 — blocking personality `waitpid` on the parallel driver: the op was
                // REWOUND, so after this (real, condvar) block the loop re-executes it against the
                // now-updated personality table. `Some(pid)` waits on that twin's permanent exit
                // record; `None` (any-child) waits for an exit generation newer than the last one
                // this vCPU consumed — see [`ThreadRegistry::wait_fork_exit`] for why neither can
                // livelock on a stale exit nor lose a wake.
                //
                // Not only an exit ends the wait: a kill of this domain (its `term_flag`, or a
                // sibling's trap), a deliverable signal (`-EINTR`, as the stdin park does), or a
                // child's stop/continue (the personality's one-shot `reap_pending` edge — what the
                // other two drivers wake a `WUNTRACED` wait on). Each is raised before its door
                // rings the table (`wire_parallel_doors`), so asking under the table lock cannot miss
                // one. The rewound `waitpid` then re-runs into the right outcome.
                let term_flag = host.lock_unpoisoned().term_flag.clone();
                fork_gen = reg.wait_fork_exit(child.map(|p| p as i64), fork_gen, || {
                    if term_flag.load(std::sync::atomic::Ordering::SeqCst)
                        || domain.dead().is_some()
                    {
                        return true;
                    }
                    let mut h = host.lock_unpoisoned();
                    if h.park_interrupted() {
                        h.set_sig_interrupt();
                        return true;
                    }
                    h.signal_poll().is_some_and(|(_, s)| s.reap_pending())
                });
            }
            Ok(VcpuStop::Exec {
                cmd,
                grants_ptr,
                grants_n,
                entry,
                size_log2,
                dst,
                personality,
            }) => {
                // #748 rung 2 — FORK.md §8.6 `execve` image-replace on the parallel driver. Every
                // refusal writes a probeable errno to `dst` and lets the caller run on (POSIX:
                // `execve` returns only on failure). Admissible from a clean root computation only
                // (no serve handler, root fiber, a non-durable domain) — the cooperative arm's
                // gate. Root and fork-twin execs take the SAME path here: each vCPU already owns
                // its window/host cell, so the cooperative arm's env split does not arise.
                let clean = vt.active.serve_ticket.is_none()
                    && vt.active_id == ROOT_FIBER
                    && !host.lock_unpoisoned().is_durable();
                let built = if clean {
                    let mut g = host.lock_unpoisoned();
                    exec_image_build(
                        &mut g,
                        mem.as_ref(),
                        dom,
                        cmd,
                        grants_ptr,
                        grants_n,
                        entry,
                        size_log2,
                        personality,
                        None,
                    )
                } else {
                    Err(super::EINVAL)
                };
                match built {
                    Err(e) => vt.active.set(dst, Reg::from_i32(e as i32)),
                    Ok(ExecBuilt {
                        host: child_host,
                        table: child_table,
                        vt: new_vt,
                        mem: win,
                        leaf: _,
                    }) => {
                        // Replace the powerbox INSIDE this vCPU's cell, not the `Arc` itself: a
                        // fork twin's exit-hook holder kept a clone of the cell at spawn, so the
                        // post-exec exit must find the CARRIED hooks (`exec_carry`) behind the
                        // same cell — the parallel analogue of the cooperative arm overwriting
                        // `extra_envs[k].host`. The image's window and the command's natural table
                        // replace this vCPU's.
                        *host.lock_unpoisoned() = child_host;
                        mem = Some(win);
                        tbl = Some(std::sync::Arc::new(child_table));
                        vt = new_vt;
                    }
                }
            }
            Ok(VcpuStop::PipeRead { pipe }) => {
                // #748 rung 3 (#1080 rung 4) — blocking CorePipe read: the op was rewound, so
                // block this OS thread until level-triggered readiness (bytes buffered, or every
                // writer closed) and let the loop re-execute it (a read, or EOF). A short-sleep
                // poll rather than a condvar door: the peer end lives on another powerbox (a fork
                // twin's, or even another driver's engine) with no cross-thread doorbell into this
                // cell yet, and a level-triggered poll cannot lose a wake. Condvar doors on the
                // shared pipe backing are the follow-up.
                // #1262 (parallel) — a default-action TERMINATE (this domain's `term_flag`, set by a
                // group `^C`/SIGKILL) must break a twin blocked here even with NO caught handler: the
                // re-run then hits the per-op `term_flag` safepoint in `resume` and the vCPU dies
                // (WIFSIGNALED). Without this a pipeline stage parked on an empty pipe when its group is
                // `^C`'d slept forever and the shell's reap never woke. Cheap to clone the flag once.
                let term_flag = host.lock_unpoisoned().term_flag.clone();
                while !host.lock_unpoisoned().pipe_read_ready(pipe) {
                    // A default-action TERMINATE, or a sibling's trap killing this domain (I37): the
                    // rewound op re-runs into the safepoint that ends this vCPU.
                    if term_flag.load(std::sync::atomic::Ordering::SeqCst)
                        || domain.dead().is_some()
                    {
                        break;
                    }
                    // #1146 slice 2 (parallel) — a signal reaching this OS thread while it blocks on
                    // the pipe interrupts the read: when a deliverable, non-`SA_RESTART` signal is
                    // pending, set this host's EINTR flag and break, so the re-run completes `-EINTR`
                    // at the slice-2a park site (the caught handler is delivered at the vCPU's next
                    // safepoint). Unlike the cooperative pump there is no central sweep — each blocked
                    // OS thread observes the interrupt itself. Setting the flag latches the interrupt
                    // across the break so the re-run surfaces EINTR even if that safepoint delivery
                    // consumes the pending signal first. `SA_RESTART` leaves `park_interrupted` false,
                    // so the poll keeps waiting for data (the restarted read).
                    if host.lock_unpoisoned().park_interrupted() {
                        host.lock_unpoisoned().set_sig_interrupt();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
            Ok(VcpuStop::PipeWrite { pipe }) => {
                // The write twin: ready when the FIFO has room under `PIPE_CAP` (backpressure
                // drained) or every reader closed (the re-run completes `-EPIPE`).
                // #1262 (parallel) — same terminate break as the read poll below.
                let term_flag = host.lock_unpoisoned().term_flag.clone();
                while !host.lock_unpoisoned().pipe_write_ready(pipe) {
                    // A default-action TERMINATE, or a sibling's trap killing this domain (I37): the
                    // rewound op re-runs into the safepoint that ends this vCPU.
                    if term_flag.load(std::sync::atomic::Ordering::SeqCst)
                        || domain.dead().is_some()
                    {
                        break;
                    }
                    // #1146 slice 2 (parallel) — same interruptible break as the read poll above.
                    if host.lock_unpoisoned().park_interrupted() {
                        host.lock_unpoisoned().set_sig_interrupt();
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
            Ok(VcpuStop::CapPending { id, dst }) => {
                // F2: the parallel driver keeps the inline completion wait — it blocks only
                // this OS thread while the pool works (the §5b overlap across sibling vCPUs
                // is the lock-release, already landed); fiber-waiter delivery through the
                // real cross-thread futex is the I45/I73 residue, its own slice.
                let comps = host.lock_unpoisoned().completions();
                // #1366: a host-completed punt has no completer on this driver — decline.
                match comps.wait_unless_host_owned(id) {
                    Some(r) => vt.active.set(dst, Reg::from_i64(r)),
                    None => return (Err(Trap::CapFault), mem),
                }
            }
            Ok(VcpuStop::Wait {
                base,
                expected,
                width,
                timeout,
                dst,
            }) => {
                // Genuine cross-thread futex: park on the shared address until another vCPU `notify`s
                // (or the timeout fires). No memory ⇒ can't park ⇒ vacuously not-equal.
                let r = match mem.as_ref() {
                    Some(m) => reg.futex.wait(&domain, m, base, expected, width, timeout),
                    None => super::WAIT_NOT_EQUAL,
                };
                vt.active.set(dst, Reg::from_i32(r));
            }
            Ok(VcpuStop::Notify { base, count, dst }) => {
                let woken = reg.futex.notify(base, count);
                vt.active.set(dst, Reg::from_i32(woken));
            }
            // §22 guest-JIT (THREADS.md 4c-domain): install/uninstall/invoke against the **shared**
            // [`Domain`] — `install`/`uninstall`/`push` are interior-mutable (Release/Acquire-paired
            // with the dispatch reads), so a worker vCPU drives them on `&Domain` while compute/atomics
            // on the other vCPUs stay lock-free. The result (slot / `-ENOSPC` / value) is
            // schedule-independent for the disciplined guest the oracle is differentially run against.
            Ok(VcpuStop::JitInstall { h, code, dst }) => {
                // Resolve authority + the unit's funcs under the host lock (a forged/cross-domain
                // handle is an inert CapFault → trap), then compile + install. Compiling can fail only
                // if the unit uses an op the engine doesn't lower yet (the one place a guest unit can
                // outrun coverage — no tree-walker fallback mid-run).
                let (funcs, types) = {
                    let g = host.lock_unpoisoned();
                    match g.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = g.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        g.jit_unit_funcs(cd, cu)
                            .ok_or(Trap::CapFault)
                            .and_then(|f| {
                                g.jit_unit_types(cd, cu)
                                    .ok_or(Trap::CapFault)
                                    .map(|t| (f, t))
                            })
                    }) {
                        Ok(f) => f,
                        Err(t) => return (Err(t), mem),
                    }
                };
                let res = match compile_module(&funcs, &types, None) {
                    Some(unit) => match dom.install(unit) {
                        Some(slot) => slot as i64,
                        None => super::ENOSPC,
                    },
                    None => return (Err(Trap::Malformed), mem), // unit op outside coverage
                };
                vt.active.set(dst, Reg::from_i64(res));
            }
            Ok(VcpuStop::JitUninstall { h, slot, dst }) => {
                {
                    let g = host.lock_unpoisoned();
                    if let Err(t) = g.resolve_jit_domain(h) {
                        return (Err(t), mem); // authority check
                    }
                }
                let n_real = dom.source.primary().progs.len();
                let res = if dom.uninstall(slot as usize, n_real) {
                    0
                } else {
                    super::EINVAL
                };
                vt.active.set(dst, Reg::from_i64(res));
            }
            Ok(VcpuStop::JitInvoke {
                h,
                code,
                argv,
                dst,
                params,
                results,
            }) => {
                // Resolve unit funcs (authority + cross-domain) and compile, as for install.
                let (funcs, types) = {
                    let g = host.lock_unpoisoned();
                    match g.resolve_jit_domain(h).and_then(|domain| {
                        let (cd, cu) = g.resolve_jit_code(code)?;
                        if cd != domain {
                            return Err(Trap::CapFault);
                        }
                        g.jit_unit_funcs(cd, cu)
                            .ok_or(Trap::CapFault)
                            .and_then(|f| {
                                g.jit_unit_types(cd, cu)
                                    .ok_or(Trap::CapFault)
                                    .map(|t| (f, t))
                            })
                    }) {
                        Ok(f) => f,
                        Err(t) => return (Err(t), mem),
                    }
                };
                let unit = match compile_module(&funcs, &types, None) {
                    Some(u) => u,
                    None => return (Err(Trap::Malformed), mem),
                };
                // Arity-check the unit entry (func 0) against the call's (code-stripped) signature.
                let arity_ok = unit
                    .sigs
                    .first()
                    .is_some_and(|(ep, er)| ep.len() == params.len() && er.len() == results.len());
                if !arity_ok {
                    return (Err(Trap::CapFault), mem);
                }
                // Marshal args via the slot ABI, push the unit as a transient module, run it over the
                // **shared** powerbox (its `call.cap`s serialize per-call, like every other vCPU's).
                let child_args: Vec<Value> = params
                    .iter()
                    .zip(argv.iter())
                    .map(|(ty, s)| slot_to_val(*ty, *s))
                    .collect();
                let umod = dom.source.push(unit);
                // The unit's `gc.roots` sees the run's parked fibers beneath it (#1660) — the shared
                // registry, locked only while such a scan reads it.
                match run_invoke(
                    &dom.source,
                    &dom.table,
                    umod,
                    &child_args,
                    &mut fuel,
                    &mut mem,
                    &mut HostCell::Shared(&host),
                    Some(&Beneath::task(&vt, FiberRegRef::Shared(&domain.fibers))),
                ) {
                    Ok(vals) => {
                        for (i, (v, ty)) in vals.iter().zip(results.iter()).enumerate() {
                            let re = slot_to_val(*ty, val_to_slot(*v));
                            vt.active.set(dst + i as u32, Reg::from_value(re));
                        }
                    }
                    Err(t) => return (Err(t), mem),
                }
            }
            // §14 `Instantiator.instantiate` (THREADS.md 4c-domain) — a **same-module** confined
            // executor child: its own power-of-two sub-window (`nested_view` of the shared backing,
            // own page-prot map), its own attenuated powerbox (`Instantiator` + `AddressSpace` over
            // `[0, child_size)`), its own natural dispatch table (no parent install slots), and a
            // quota sub-allocated from the parent's fuel. The child is a **nested confined parallel
            // run** on its own scoped thread — joinable through the parent's registry exactly like a
            // `thread.spawn` child. Unlike a `thread.spawn` child (which shares this vCPU's `Mem`
            // view + the shared powerbox), it owns all of these — the §14 confinement.
            // §14 confined children (ops 0, 5, 13, 17): the executor's admission, under the host lock —
            // so a budget or a grant list is served here as there (#1855) — then a scoped OS thread
            // over the carve. This vCPU's own `mem`/`fuel` *are* its environment: a confined parent
            // already runs on its own thread with its own confined view.
            Ok(VcpuStop::Instantiate { spawn, dst }) => {
                let admitted = {
                    let mut hg = host.lock_unpoisoned();
                    admit_confined_child(
                        &mut hg,
                        mem.as_ref(),
                        fuel,
                        &dom.source,
                        &vt.active,
                        spawn,
                    )
                };
                let child = match admitted {
                    Ok(Some(c)) => c,
                    Ok(None) => {
                        vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                    Err(t) => return (Err(t), mem),
                };
                match par_start_child(scope, dom, reg, &host, &mut threads, child, spawn.entry) {
                    Ok(handle) => vt.active.set(dst, Reg::from_i32(handle)),
                    Err(t) => return (Err(t), mem),
                }
            }
            // §5 `instantiate_detached` (op 15): a fresh window of its own on its own OS thread — the
            // cooperative executor's spawn (`admit_detached_child`, the same admission and child
            // powerbox) on this driver, then `run_vcpu_parallel` over the child's `Mem` exactly as a
            // confined child, its result published to the parent's `reg` for `join`.
            Ok(VcpuStop::InstantiateDetached { spawn, dst }) => {
                let admitted = admit_detached_in_process(
                    &mut host.lock_unpoisoned(),
                    mem.as_ref(),
                    fuel,
                    spawn,
                );
                let child = match admitted {
                    Ok(Some(c)) => c,
                    Ok(None) => {
                        vt.active.set(dst, Reg::from_i32(super::EINVAL as i32));
                        continue;
                    }
                    Err(t) => return (Err(t), mem),
                };
                match par_start_child(scope, dom, reg, &host, &mut threads, child, spawn.entry) {
                    Ok(handle) => vt.active.set(dst, Reg::from_i32(handle)),
                    Err(t) => return (Err(t), mem),
                }
            }
        }
    }
}

/// Start an admitted §14/§5 child on its own scoped OS thread — its own domain (a natural table over
/// the shared source), attenuated powerbox, window, fuel and thread registry (for the threads and
/// children *it* spawns) — publishing its result to this vCPU's `reg`, where `join` finds it. Returns
/// the join handle; `Err(ThreadFault)` on the cross-thread vCPU-count bomb (the cooperative driver's
/// `live >= MAX_VCPUS`).
fn par_start_child<'scope, 'env>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    dom: &'env Domain,
    reg: &'env ThreadRegistry,
    parent_host: &std::sync::Arc<std::sync::Mutex<Host>>,
    threads: &mut Vec<Option<u64>>,
    child: AdmittedChild,
    entry: i64,
) -> Result<i32, Trap> {
    if reg.live.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1 > super::MAX_VCPUS {
        reg.live.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        return Err(Trap::ThreadFault);
    }
    let AdmittedChild {
        mem,
        host,
        program,
        args,
        fuel,
        lease,
    } = child;
    let (module, prog) = program.land(&dom.source)?;
    let (vt, table) = child_task(module, &prog, entry, &args, host.jit_table_log2())?;
    let child_dom = Domain::child(std::sync::Arc::clone(&dom.source), table);
    let lease = lease.map(|(budget, bytes)| (std::sync::Arc::clone(parent_host), budget, bytes));
    let id = reg
        .next_id
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    scope.spawn(move || {
        let child_reg = ThreadRegistry::new();
        let child_host = std::sync::Arc::new(std::sync::Mutex::new(host));
        let (r, _m) = std::thread::scope(|cscope| {
            run_vcpu_parallel(
                cscope,
                &child_dom,
                &child_reg,
                std::sync::Arc::clone(&child_host),
                std::sync::Arc::new(ParDomain::default()),
                None,
                vt,
                mem,
                fuel,
            )
        });
        // FORK.md §8.6 / #1807 — the child's domain finished: release its pipe ends.
        child_host.lock_unpoisoned().release_pipe_ends();
        // A detached child's window goes back to the budget that paid for it (INVARIANTS #3), before
        // the result is published, so a joiner sees the refund.
        if let Some((parent, budget, bytes)) = lease {
            parent.lock_unpoisoned().budget_mem_give(budget, bytes);
        }
        reg.publish(id, r);
    });
    let handle = threads.len() as i32;
    threads.push(Some(id));
    Ok(handle)
}

/// Mark task `ti` finished with `res`, then wake any vCPU parked on `thread.join` of it: an `Ok`
/// result is delivered into the joiner's `dst` (it becomes runnable); a trap propagates — the joiner
/// completes with the same trap (transitively, via the worklist).
fn complete(tasks: &mut [TaskSlot], ti: usize, res: Result<Vec<Value>, Trap>) {
    let mut work = vec![(ti, res)];
    while let Some((done, res)) = work.pop() {
        tasks[done].state = TaskState::Done(res.clone());
        for (j, t) in tasks.iter_mut().enumerate() {
            let TaskState::BlockedJoin { child, slot, dst } = t.state else {
                continue;
            };
            if child != done {
                continue;
            }
            t.threads[slot] = None;
            match &res {
                Ok(vals) => {
                    let v = vals.first().copied().unwrap_or(Value::I64(0));
                    t.vt.active.set(dst, Reg::from_value(v));
                    t.state = TaskState::Runnable;
                }
                Err(trap) => work.push((j, Err(*trap))),
            }
        }
    }
}

/// Domain lifetime & teardown, cooperative-bytecode form (DESIGN.md §12, owner 2026-07-24;
/// ISSUES.md I37): a member's trap/exit is terminal for its whole **domain** — the shared-window
/// world its `env` names (`None` = the root + its `thread.spawn` threads; `Some(k)` = a §14
/// child + its threads). Fixpoint: find a domain with a `Done(Err)` member and a still-live
/// member, kill every live member with the same trap (via [`complete`], so a cross-domain joiner
/// re-raises and a `poll` reports status 2 — the I37 supervision mechanics), then errno-wake
/// cross-domain callers parked through the dying child (D37 death-is-revocation: cancellation is
/// a value, never a hang); repeat until no such domain remains (a kill that propagates into a
/// joiner may fell *its* domain next). The root domain's death is read by the caller's loop-top
/// root check — a sibling's trap becomes the run's result. Runs before anything is scheduled, so
/// teardown is "next safepoint" prompt in the cooperative model.
fn teardown_domains(
    tasks: &mut [TaskSlot],
    extra_envs: &[ChildEnv],
    dead_envs: &mut std::collections::BTreeSet<usize>,
) {
    loop {
        let hit = tasks.iter().find_map(|t| {
            if let TaskState::Done(Err(trap)) = &t.state {
                match t.env {
                    // A child domain is processed exactly once (`dead_envs` is the marker), even
                    // when the trapping member was its only vCPU — a later call through it must
                    // still find it dead (errno, not a deadlock).
                    Some(k) if !dead_envs.contains(&k) => {
                        return Some((Some(k), *trap));
                    }
                    // The root domain: a live member left means the sweep hasn't run yet (once
                    // every member is Done the caller's root check ends the run).
                    None if tasks
                        .iter()
                        .any(|u| u.env.is_none() && !matches!(u.state, TaskState::Done(_))) =>
                    {
                        return Some((None, *trap));
                    }
                    _ => {}
                }
            }
            None
        });
        let Some((env, trap)) = hit else { return };
        if let Some(k) = env {
            dead_envs.insert(k);
        }
        for i in 0..tasks.len() {
            if tasks[i].env == env && !matches!(tasks[i].state, TaskState::Done(_)) {
                complete(tasks, i, Err(trap));
            }
        }
        // The dying child's undelivered dispatches: wake every caller parked on a ticket
        // against its host with the probeable errno (queued or admitted — no reply will ever
        // come), and drop the queue. Later calls are refused at the LiveCall arm (`dead_envs`).
        if let Some(k) = env {
            let dying = &extra_envs[k].host;
            for t in tasks.iter_mut() {
                if let TaskState::BlockedTicket { callee, dst, .. } = &t.state {
                    if std::sync::Arc::ptr_eq(callee, dying) {
                        let dst = *dst;
                        t.vt.active.set(dst, Reg::from_i64(super::CAP_REVOKED));
                        t.state = TaskState::Runnable;
                    }
                }
            }
            dying.lock_unpoisoned().svc_queue.clear();
        }
    }
}

/// The reified bytecode continuation — everything a suspended activation needs to resume, held as
/// an explicit value rather than on the host Rust call stack. The register file (`regs`), the stack
/// of suspended caller activations (`stack`), and the `(cur, base, pc)` cursor together fully
/// describe a paused vCPU: the flat analogue of the tree-walker's `Vec<Frame>`.
///
/// Holding the continuation as data (not as live host-stack frames) is the structural prerequisite
/// for the scheduler / fiber / thread / debug seams (INTERP_PERF.md Slice 1c): a later slice breaks
/// [`Vm::resume`]'s loop at suspension points (preemption budget, blocking op, debug stop), persists
/// the cursor back into `self`, and hands this struct to the caller to park / hash / resume — exactly
/// what `park_suspended(frames)` does for the tree-walker today.
/// #1062 — the monotonic **`setjmp` token** source for the bytecode engine (its own counter; a run
/// uses one engine tier, and `setjmp_points` is per-`Vm`, so it never collides with the tree-walk
/// counter). Each `setjmp` writes a fresh token into the guest `jmp_buf`'s opaque first 8 bytes and
/// keys its checkpoint by that token, so a guest that *copies* the `jmp_buf` (bash's `COPY_PROCENV`
/// memcpy of `top_level`, on every `bash -c`) carries the checkpoint identity with the bytes —
/// address-keying could not, and mis-resolved a restored copy into an infinite `longjmp` loop. The
/// value is opaque: never guest-observable, so only token *equality* (write `T`, read it back)
/// matters — the exact value and any cross-thread interleaving of this counter cannot affect a run.
static BYTE_SETJMP_TOKEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// A `<setjmp.h>` checkpoint (see [`Vm::setjmp_points`]): everything needed to re-enter a `setjmp`
/// activation. `longjmp` truncates [`Vm::stack`] to `depth` (the intervening activations discarded —
/// C has no cleanups), restores the `(module, cur, base, pc)` cursor, and sets the `dst` register to
/// the long-jump value. The activation's register window survives in place, so it is not snapshotted.
#[derive(Clone, Copy)]
struct ByteSetJmp {
    /// `Vm::stack` length at `setjmp` (the `setjmp` activation is the current one, not yet pushed).
    depth: usize,
    module: usize,
    cur: usize,
    base: usize,
    /// The op index just after the `setjmp`.
    pc: usize,
    /// The `setjmp` result's window slot (relative to `base`) — set to the long-jump value on re-entry.
    dst: u32,
}

// `Clone` for time-travel checkpointing (DEBUGGING.md W1): a `ScheduledDebugRun` snapshots each task's
// active `Vm` into the `seek` checkpoint ladder. Every field is a plain value or read-only `Arc`
// (`jit_eligible` — the tier-up bitmap, shared not mutated), so this is a faithful deep copy; the
// guest page store is **not** here (it lives in `ScheduledDebugRun::mem`, snapshotted separately via
// `Mem::window_snapshot`), so cloning a `Vm` never aliases another run's memory.
#[derive(Clone)]
struct Vm {
    /// Function-wide register file, shared across activations by register windows (`[base, base +
    /// nslots)` per activation). Grows on demand as calls open deeper windows.
    regs: Vec<Reg>,
    /// Suspended caller activations: `(module, prog, base, resume pc, absolute first result slot)`.
    /// `module` is carried so a cross-module `call.dyn` (into an installed §22 unit) returns to
    /// the caller's module.
    stack: Vec<(usize, usize, usize, usize, usize)>,
    /// The running activation's module (index into `Domain::mods`; 0 = primary), function index,
    /// window base, and op cursor.
    module: usize,
    cur: usize,
    base: usize,
    pc: usize,
    /// Edge-copy staging buffer (parallel-copy safety); kept here so it is reused across resumes.
    scratch: Vec<Reg>,
    /// `<setjmp.h>` checkpoints — `setjmp` records its activation's resume point here, keyed by a
    /// per-`setjmp` **token** it writes into the guest `jmp_buf`'s opaque first 8 bytes (#1062);
    /// `longjmp` reads the token back from the (possibly-copied) buffer and looks it up. No register
    /// snapshot is needed (unlike the tree-walker): the flat per-function register layout gives each
    /// block its own slots, so the `setjmp` block's values survive a deeper call in place.
    /// Token-keying (vs the old address key) is what lets a guest *copy* a `jmp_buf` — bash's
    /// `COPY_PROCENV` memcpy of `top_level` — carry the checkpoint identity with the bytes; bounded
    /// by pruning checkpoints whose `setjmp` activation has returned (`SetJmp` op).
    setjmp_points: std::collections::BTreeMap<u64, ByteSetJmp>,
    /// §12.8 4A.5: the window offset of this context's shadow-SP **word** — the base of its own region
    /// (`shadow_region_base`), which `durable.shadow_base` returns so the instrumented IR addresses its
    /// per-context SP word. The root's is context 0 (`ShadowArena::region_base(0)`); a fiber's its `slot + 1`. Set when
    /// the Vm is created (fiber) / activated; unused on a non-durable run.
    durable_region_base: u64,
    /// **wasm-JIT tier-up bitmap** (browser wasm-JIT threads slice), for module-0 functions only. Set
    /// on the root Vm via [`Vcpu::with_jit_eligible`]; a direct `Call` in module 0 to an eligible
    /// function surfaces [`Outcome::TierUp`] instead of interpreting. `None` (fibers, invoked units,
    /// non-JIT runs) ⇒ everything interprets — tier-up is a pure acceleration, never a correctness gate.
    jit_eligible: Option<std::sync::Arc<[bool]>>,
    /// #750 **paged tier-up**: the eligible functions were emitted with the software page-check
    /// (`compile_module_tierup_paged`), so the dispatch must NOT decline tier-up on a
    /// scalar-unrepresentable window — the host-maintained page table carries per-page fidelity,
    /// and the event's `mapped` becomes the reserved window size (the bound must never under-admit
    /// a table-admitted page). Set only via [`Vcpu::with_jit_page_checked`].
    jit_page_checked: bool,
    /// §3.6 serve-loop core (I36 slice 1): the in-flight handler's completion ticket — `Some`
    /// between admitting a handler activation (whose return linkage rewinds into the `SvcPoll`
    /// op) and the re-execution that settles its result — and the count of dispatches completed
    /// by the current `svc.poll` activation.
    serve_ticket: Option<u64>,
    serve_count: i64,
    /// §12 per-vCPU **thread-local register** (`vcpu.tls.get`/`set`). One i64 of per-vCPU state,
    /// seeded to this vCPU's dense id at construction (root = 0; a spawned thread's `Vm` is re-seeded
    /// to its id in `drive`'s `Spawn` arm), guest-overwritable. Read at the op's execution point.
    /// Mirrors the tree-walker's `Vm::tls`. (Multi-OS-thread fiber *migration* re-seeding — a fiber
    /// resumed on a different worker reading that worker's word — is a follow-up; the browser tier is
    /// single-OS-thread cooperative, where the sole worker is 0, so every read is a faithful `0`.)
    tls: i64,
    /// The domain's **home module** — the unit whose functions are its service handlers (0 for the
    /// primary; a separate-module child's pushed unit index). `svc.poll`/`svc.wait` only dispatch
    /// handlers while executing in this module: `svc_handler_func` resolves indices against the
    /// domain's registered `self_module`, so serving from any *other* unit (an installed §22 unit
    /// running in the root domain) would index the wrong program table — fail closed instead.
    home: usize,
    /// #1146 async signal delivery — the guard stack, the bytecode twin of the tree-walker's
    /// `Vm::sig_handler_stack` (lib.rs). Each entry records `self.stack.len()` at the point an async
    /// signal handler activation was injected (a `(i64 sp, i32 signum) -> ()` window opened like a
    /// `call.dyn` at a per-op safepoint); `Op::Ret` pops the matching entry and calls
    /// `handler_returned()` to restore the block-during-handler mask, and a `longjmp` OUT of a handler
    /// (bash's `throw_to_top_level`) pops every crossed entry the same way. Bounded to
    /// `MAX_SIG_HANDLER_NEST` nested deliveries. Empty on any run without a signal personality.
    sig_handler_stack: Vec<usize>,
}

impl Vm {
    /// Open the entry activation: a zero-based window sized to the entry function, seeded with the
    /// call arguments. Total — an out-of-range entry or arg overflow is a clean `Malformed` trap.
    /// Every entry (root, fiber, thread, coroutine) starts in module 0.
    fn new(c: &Compiled, entry: usize, args: &[Value]) -> Result<Vm, Trap> {
        let prog = c.progs.get(entry).ok_or(Trap::Malformed)?;
        let mut regs: Vec<Reg> = vec![Reg::default(); prog.nslots as usize];
        for (i, a) in args.iter().enumerate() {
            *regs.get_mut(i).ok_or(Trap::Malformed)? = Reg::from_value(*a);
        }
        Ok(Vm {
            regs,
            stack: Vec::new(),
            module: 0,
            cur: entry,
            base: 0,
            pc: 0,
            scratch: Vec::new(),
            setjmp_points: std::collections::BTreeMap::new(),
            durable_region_base: c.shadow.unwrap_or(super::ShadowArena::EMPTY).region_base(0), // root context (overwritten for fibers)
            jit_eligible: None, // set only on the root Vm via `Vcpu::with_jit_eligible`
            jit_page_checked: false,
            serve_ticket: None,
            serve_count: 0,
            tls: 0, // §12 per-vCPU TLS seed: dense vCPU id (root = 0; a spawned thread re-seeds to its id)
            home: 0,
            sig_handler_stack: Vec::new(), // #1146 — no delivery in flight at entry
        })
    }

    /// Write a value to a frame-relative slot of the *current* (persisted) activation window. Used
    /// by [`drive`] to deliver fiber results (`cont.new` handle, `cont.resume` `(status, value)`,
    /// the next `arg` into a `suspend`) into a `Vm` paused at a fiber op — `base` is the cursor the
    /// last `resume` persisted, so this targets the same window the op's `dst` was resolved against.
    fn set(&mut self, slot: u32, v: Reg) {
        self.regs[self.base + slot as usize] = v;
    }

    /// The [`crate::IrPc`] of the op the cursor is on, or `None` if that op is a terminator (which the
    /// debug seam never stops at — see [`Program::src`]). Used by [`ir_trace`] to record the same
    /// instruction-location sequence the tree-walker's `Inspector` reports.
    fn cur_ir_pc(&self, source: &ModuleSource) -> Option<super::IrPc> {
        let cm = source.get(self.module)?;
        let (block, inst) = cm.progs[self.cur].src.get(self.pc).copied().flatten()?;
        // A terminator is a stop position too (#1713), at `inst == insts.len()` — see `Program::src`.
        Some(super::IrPc {
            module: self.module as u32,
            func: self.cur as FuncIdx,
            block: block as usize,
            inst: (inst & !SRC_TERM) as usize,
        })
    }

    /// Run the continuation for at most `budget` ops, then return [`Outcome::Suspended`] at the next
    /// op boundary with the cursor persisted into `self` (resume by calling again); return
    /// [`Outcome::Done`] when the entry activation returns, or `Err` on a trap. Per-op fuel is
    /// charged here, one charge per op, exactly as the run-to-completion form did — slicing only
    /// chooses *where* to pause, never *what* runs, so the result is independent of `budget`.
    ///
    /// The cursor (`cur`/`base`/`pc`) lives in locals for the duration of the loop so the optimizer
    /// keeps it in registers; it is written back to `self` only when the loop exits (suspend), which
    /// is also what a future blocking-op / debug-stop seam will do before yielding.
    /// Write a host op's result slots into the registers from `at`.
    fn set_results(&mut self, at: usize, res: &[i64], results: &[ValType]) {
        for (i, (s, ty)) in res.iter().zip(results.iter()).enumerate() {
            self.regs[at + i] = Reg::from_value(slot_to_val(*ty, *s));
        }
    }

    /// #1904 — under a landing freeze a durable domain never parks (the oracle's `decide`, #1672):
    /// an op that would park took no effect, so it is **abandoned**. `true` when it is: the running
    /// context's re-issue word is set, the op's placeholder results stand, and the call's trailing
    /// poll unwinds; the thaw re-issues the call.
    fn abandon_for_freeze(&self, mem: &mut Option<Mem>, host: &mut HostCell) -> bool {
        let freezing = is_unwinding(mem) && host.with(|p| p.is_durable());
        if freezing {
            if let Some(m) = mem.as_mut() {
                m.durable_set_reissue_at(self.durable_region_base);
            }
        }
        freezing
    }

    fn resume(
        &mut self,
        source: &ModuleSource,
        table: &SharedSlots,
        fuel: &mut u64,
        mem: &mut Option<Mem>,
        host: &mut HostCell,
        mut budget: u64,
    ) -> Result<Outcome, Trap> {
        let mut module = self.module;
        let mut cur = self.cur;
        let mut base = self.base;
        let mut pc = self.pc;
        // THREADS.md 4c-domain: the shared module source is read through a per-vCPU **lock-free local
        // cache** (`Arc` clones), refreshed only on a miss (a unit installed since the last sync). The
        // active module is held as an owned `Arc<Compiled>` (`c`) — independent of `local`, so a refresh
        // can't invalidate it — re-resolved only when an activation crosses modules (so the per-op hot
        // path, `c.*` via `Arc` deref, is unchanged). `resolve!` returns the `Arc` for a module index.
        let mut local: Vec<std::sync::Arc<Compiled>> = source.snapshot();
        macro_rules! resolve {
            ($m:expr) => {{
                let m = $m as usize;
                if m >= local.len() {
                    local = source.snapshot(); // miss: a module installed since last sync
                }
                match local.get(m) {
                    Some(a) => std::sync::Arc::clone(a),
                    None => return Err(Trap::Malformed), // forged/stale module index (defensive)
                }
            }};
        }
        let mut c: std::sync::Arc<Compiled> = resolve!(module);

        // #1146 async signal delivery (the bytecode twin of the tree-walker's #796 L2 safepoint
        // redirect). Fetch the personality's `(armed, source)` poll pair once per resume — two cheap
        // `Arc` clones per quantum, not per op. `None` on any run without a signal personality (JIT
        // bench, pure compute), so the per-op check in the loop is a single predictable branch there.
        let signal_poll = host.with(|h| h.signal_poll());
        // #1246 — the per-op default-action TERMINATE poll (the bytecode twin of the tree-walker's
        // `term_flag` safepoint). Fetched once per resume, and only when a signal personality is present
        // (`None` on a pure-compute / JIT-bench run, so the per-op check is skipped entirely). This is
        // the GENUINELY-PARALLEL driver's kill mechanism: its OS-thread vCPUs run concurrently, so a
        // killed thread must observe the kill mid-execution (it can't wait for a scheduler sweep). The
        // cooperative driver never sets this flag — it finalizes a killed domain at its loop top (#1215)
        // — so the load is dead (always `false`) there.
        let term_flag = signal_poll
            .as_ref()
            .map(|_| host.with(|h| h.term_flag.clone()));

        macro_rules! r {
            ($i:expr) => {
                self.regs[base + $i as usize]
            };
        }
        // Apply edge copies parallel-safely (a self-loop can alias src/dst): gather then scatter.
        macro_rules! edge {
            ($copies:expr) => {{
                let cp = $copies;
                if cp.aliasing {
                    // A destination is re-read as a source: gather all sources, then scatter, so a
                    // value isn't clobbered before it is read (a param swap/rotation).
                    self.scratch.clear();
                    for &(s, _) in cp.pairs.iter() {
                        self.scratch.push(self.regs[base + s as usize]);
                    }
                    for (k, &(_, d)) in cp.pairs.iter().enumerate() {
                        self.regs[base + d as usize] = self.scratch[k];
                    }
                } else {
                    // Non-aliasing (the common induction/accumulator edge): copy directly in one pass,
                    // no `scratch` traffic — sources and destinations are disjoint slot sets.
                    for &(s, d) in cp.pairs.iter() {
                        let v = self.regs[base + s as usize];
                        self.regs[base + d as usize] = v;
                    }
                }
            }};
        }
        // Fuel unification: fuel is metered at **IR safepoints** — a taken back-edge and each function
        // entry — not per op. A back-edge is a *backward* jump in the flat op array: blocks are laid
        // out in index order, so a terminator (the last op of its block, at index `pc`) taking a target
        // whose entry `$t <= pc` is exactly a branch to an earlier-or-same block. This bounds every
        // loop/recursion (an infinite loop must cross its back-edge unboundedly) while leaving
        // straight-line code free — the one unit the tree-walker and JIT can meter identically. The
        // per-op `budget` (suspension / single-step) is unchanged.
        macro_rules! backedge {
            ($t:expr) => {
                if ($t as usize) <= pc {
                    step(fuel, None)?;
                }
            };
        }

        loop {
            if budget == 0 {
                // Pause at this op boundary: persist the cursor so a later `resume` continues here.
                self.module = module;
                self.cur = cur;
                self.base = base;
                self.pc = pc;
                return Ok(Outcome::Suspended);
            }
            budget -= 1;
            // #1246 default-action terminate: a `SIG_DFL` SIGKILL/SIGTERM/SIGINT delivered to this
            // domain set its `term_flag` (via the personality's `set_kill` closure, wired on the
            // parallel driver by `wire_parallel_doors`). Die at this op — the vCPU's thread returns the trap,
            // and the driver's exit hook reports term-by-signal from the personality's `term_sig`
            // bookkeeping (WIFSIGNALED), exactly like the tree-walker. Checked before the async-signal
            // redirect (death beats a caught delivery).
            if let Some(tf) = &term_flag {
                if tf.load(std::sync::atomic::Ordering::Relaxed) {
                    return Err(Trap::ThreadFault);
                }
            }
            // #1146 async signal delivery: at this per-op safepoint, if the personality has a caught,
            // unmasked signal pending (its cheap `armed` flag) and we are not already
            // `MAX_SIG_HANDLER_NEST` deep, redirect this vCPU into the guest handler
            // `(i64 sp, i32 signum) -> ()` — a window opened exactly like a `call.dyn`, resolved and
            // type-checked through the dispatch table. The interrupted op is NOT advanced (the return
            // entry resumes at `pc`, re-running it once the handler returns), matching the tree-walker.
            // Same interrupt-at-safepoint shape as `kill`, but non-lethal. (Interruptible-park `-EINTR`
            // is a follow-up: no interruptible parks reach this synchronous safepoint path yet.)
            if self.sig_handler_stack.len() < super::MAX_SIG_HANDLER_NEST {
                if let Some((armed, source)) = &signal_poll {
                    if armed.load(std::sync::atomic::Ordering::Relaxed) {
                        // The source owns its own locking (an `Arc<Mutex<Proc>>`); we hold no host lock
                        // here. A `None` means nothing deliverable (ignored/masked/stopped) — it also
                        // disarms, so the check idles until the next `raise`/`kill`.
                        if let Some((fref, signum, sp)) = source.take_deliverable() {
                            let slot = (fref as u32 as usize) & (table.len() - 1);
                            let ts = table.slot(slot);
                            if ts.module != super::TABLE_EMPTY {
                                let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
                                let tm = resolve!(tmod);
                                let (cp, cr) = &tm.sigs[tfunc];
                                // `void handler(int)` = `(i64 sp, i32 signum) -> ()` — chibicc threads
                                // the data-SP as v0. A mis-typed handler is dropped (the signal was
                                // already consumed by `take_deliverable`), never fatal.
                                if matches!(cp.as_slice(), [ValType::I64, ValType::I32])
                                    && cr.is_empty()
                                {
                                    // Open a fresh window past the current activation (like `Op::Call`),
                                    // seeded with `(sp, signum)`. `nb`/`need` use the *current* `c`
                                    // before any cross-module reassignment below.
                                    let nb = base + c.progs[cur].nslots as usize;
                                    let need = nb + tm.progs[tfunc].nslots as usize;
                                    if self.regs.len() < need {
                                        self.regs.resize(need, Reg::default());
                                    }
                                    self.regs[nb] = Reg::from_i64(sp as i64);
                                    self.regs[nb + 1] = Reg::from_i32(signum);
                                    // Return linkage resumes at the SAME `pc` so the interrupted op
                                    // re-runs; the void handler writes no results (`ret_abs` unused —
                                    // `base` is a harmless placeholder).
                                    self.stack.push((module, cur, base, pc, base));
                                    if tmod != module {
                                        module = tmod;
                                        c = tm;
                                    }
                                    cur = tfunc;
                                    base = nb;
                                    pc = 0;
                                    self.sig_handler_stack.push(self.stack.len());
                                    continue;
                                }
                            }
                        }
                    }
                }
            }
            #[cfg(feature = "callprof")]
            if module == 0 {
                if let Some(loc) = c.progs[cur].src.get(pc).copied().flatten() {
                    callprof::op(cur, loc);
                }
            }
            match &c.progs[cur].ops[pc] {
                Op::Const { dst, val } => {
                    r!(*dst) = *val;
                    pc += 1;
                }
                Op::IntBin { dst, a, b, ty, op } => {
                    let v = match ty {
                        IntTy::I32 => Reg::from_i32(bin32(*op, r!(*a).i32(), r!(*b).i32())?),
                        IntTy::I64 => Reg::from_i64(bin64(*op, r!(*a).i64(), r!(*b).i64())?),
                    };
                    r!(*dst) = v;
                    pc += 1;
                }
                Op::IntCmp { dst, a, b, ty, op } => {
                    let res = match ty {
                        IntTy::I32 => cmp32(*op, r!(*a).i32(), r!(*b).i32()),
                        IntTy::I64 => cmp64(*op, r!(*a).i64(), r!(*b).i64()),
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::IntUn { dst, a, ty, op } => {
                    r!(*dst) = match ty {
                        IntTy::I32 => Reg::from_i32(intun32(*op, r!(*a).i32())),
                        IntTy::I64 => Reg::from_i64(intun64(*op, r!(*a).i64())),
                    };
                    pc += 1;
                }
                Op::Eqz { dst, a, ty } => {
                    let res = match ty {
                        IntTy::I32 => r!(*a).i32() == 0,
                        IntTy::I64 => r!(*a).i64() == 0,
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::Convert { dst, a, op } => {
                    r!(*dst) = match op {
                        ConvOp::ExtendI32S => Reg::from_i64(r!(*a).i32() as i64),
                        ConvOp::ExtendI32U => Reg::from_i64(r!(*a).i32() as u32 as i64),
                        ConvOp::WrapI64 => Reg::from_i32(r!(*a).i64() as i32),
                    };
                    pc += 1;
                }
                Op::Select { dst, cond, a, b } => {
                    r!(*dst) = if r!(*cond).i32() != 0 { r!(*a) } else { r!(*b) };
                    pc += 1;
                }
                Op::FBin { dst, a, b, ty, op } => {
                    r!(*dst) = match ty {
                        FloatTy::F32 => Reg::from_f32(fbin32(*op, r!(*a).f32(), r!(*b).f32())),
                        FloatTy::F64 => Reg::from_f64(fbin64(*op, r!(*a).f64(), r!(*b).f64())),
                    };
                    pc += 1;
                }
                Op::FUn { dst, a, ty, op } => {
                    r!(*dst) = match ty {
                        FloatTy::F32 => Reg::from_f32(fun32(*op, r!(*a).f32())),
                        FloatTy::F64 => Reg::from_f64(fun64(*op, r!(*a).f64())),
                    };
                    pc += 1;
                }
                Op::FCmp { dst, a, b, ty, op } => {
                    let res = match ty {
                        FloatTy::F32 => fcmp32(*op, r!(*a).f32(), r!(*b).f32()),
                        FloatTy::F64 => fcmp64(*op, r!(*a).f64(), r!(*b).f64()),
                    };
                    r!(*dst) = Reg::from_i32(res as i32);
                    pc += 1;
                }
                Op::FToISat { dst, a, op } => {
                    r!(*dst) = fto_i(*op, r!(*a));
                    pc += 1;
                }
                Op::FToITrap { dst, a, op } => {
                    r!(*dst) = trunc_trap(*op, r!(*a))?;
                    pc += 1;
                }
                Op::IToFConv { dst, a, op } => {
                    r!(*dst) = i_to_f(*op, r!(*a));
                    pc += 1;
                }
                Op::Cast { dst, a, op } => {
                    r!(*dst) = cast(*op, r!(*a));
                    pc += 1;
                }
                Op::RefFunc { dst, func } => {
                    r!(*dst) = Reg::from_i32(*func as i32);
                    pc += 1;
                }
                Op::Load {
                    dst,
                    addr,
                    op,
                    offset,
                } => {
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let a = r!(*addr).i64() as u64;
                    r!(*dst) = m.load_scalar(a, *offset, *op)?;
                    pc += 1;
                }
                Op::Store {
                    addr,
                    value,
                    op,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let lo = r!(*value).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .store_scalar(a, *offset, *op, lo)?;
                    pc += 1;
                }
                // Bulk-memory ops (D62): both `MemCopy` and `MemMove` use the overlap-safe fast path
                // (bulk `memmove` on the backing behind the same whole-span confinement; the tree-walk
                // oracle keeps the scalar `mem_copy`).
                Op::MemCopy { dst, src, len } | Op::MemMove { dst, src, len } => {
                    let d = r!(*dst).i64() as u64;
                    let s = r!(*src).i64() as u64;
                    let n = r!(*len).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .mem_copy_fast(d, s, n)?;
                    pc += 1;
                }
                Op::MemFill { dst, val, len } => {
                    let d = r!(*dst).i64() as u64;
                    let v = r!(*val).i32() as u8;
                    let n = r!(*len).i64() as u64;
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .mem_fill_fast(d, v, n)?;
                    pc += 1;
                }
                Op::AtomicLoad {
                    dst,
                    addr,
                    ty,
                    offset,
                } => {
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let a = r!(*addr).i64() as u64;
                    r!(*dst) = Reg::from_value(m.atomic_load(a, *offset, *ty)?);
                    pc += 1;
                }
                Op::AtomicStore {
                    addr,
                    value,
                    ty,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let v = Value::I64(r!(*value).i64());
                    mem.as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_store(a, *offset, *ty, v)?;
                    pc += 1;
                }
                Op::AtomicRmw {
                    dst,
                    addr,
                    value,
                    ty,
                    op,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let v = Value::I64(r!(*value).i64());
                    let res = mem
                        .as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_rmw(a, *offset, *ty, *op, v)?;
                    r!(*dst) = Reg::from_value(res);
                    pc += 1;
                }
                Op::AtomicCmpxchg {
                    dst,
                    addr,
                    expected,
                    replacement,
                    ty,
                    offset,
                } => {
                    let a = r!(*addr).i64() as u64;
                    let exp = Value::I64(r!(*expected).i64());
                    let rep = Value::I64(r!(*replacement).i64());
                    let res = mem
                        .as_mut()
                        .ok_or(Trap::Malformed)?
                        .atomic_cmpxchg(a, *offset, *ty, exp, rep)?;
                    r!(*dst) = Reg::from_value(res);
                    pc += 1;
                }
                Op::Br { copies, target } => {
                    backedge!(*target);
                    edge!(copies);
                    pc = *target as usize;
                }
                Op::BrIf {
                    cond,
                    then_copies,
                    then_pc,
                    else_copies,
                    else_pc,
                } => {
                    if r!(*cond).i32() != 0 {
                        backedge!(*then_pc);
                        edge!(then_copies);
                        pc = *then_pc as usize;
                    } else {
                        backedge!(*else_pc);
                        edge!(else_copies);
                        pc = *else_pc as usize;
                    }
                }
                // Slice 5a fused compare+branch. Fuel is charged only if the taken edge is a
                // back-edge (fuel unification) — the fused-away `IntCmp` no longer needs its own
                // per-op charge, so fusion now saves a full op's work on the loop back-edge.
                Op::BrIfCmp {
                    a,
                    b,
                    ty,
                    op,
                    then_copies,
                    then_pc,
                    else_copies,
                    else_pc,
                } => {
                    let taken = match ty {
                        IntTy::I32 => cmp32(*op, r!(*a).i32(), r!(*b).i32()),
                        IntTy::I64 => cmp64(*op, r!(*a).i64(), r!(*b).i64()),
                    };
                    if taken {
                        backedge!(*then_pc);
                        edge!(then_copies);
                        pc = *then_pc as usize;
                    } else {
                        backedge!(*else_pc);
                        edge!(else_copies);
                        pc = *else_pc as usize;
                    }
                }
                Op::BrTable { idx, arms, default } => {
                    let i = r!(*idx).i32() as u32 as usize;
                    let (copies, target) = arms.get(i).unwrap_or(default);
                    backedge!(*target);
                    edge!(copies);
                    pc = *target as usize;
                }
                // `<setjmp.h>` `setjmp`: checkpoint the resume point (the op after this, in this
                // activation) keyed by the guest `jmp_buf` address, and return 0. The register window
                // survives in place (per-block slots are distinct), so no snapshot is taken.
                Op::SetJmp { buf, dst } => {
                    // #1062 — key the checkpoint by a fresh TOKEN written into the guest `jmp_buf`'s
                    // opaque first 8 bytes (not the buffer address), so a guest that memcpy-copies
                    // the buffer (bash's `COPY_PROCENV`) carries the identity with the bytes. The
                    // write goes through the confinement mask like any guest store; a `jmp_buf` is
                    // always a committed object the guest just passed, so a fault is a broken guest.
                    let buf_addr = r!(*buf).i64() as u64;
                    let token =
                        BYTE_SETJMP_TOKEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    mem.as_mut().ok_or(Trap::Malformed)?.store_scalar(
                        buf_addr,
                        0,
                        StoreOp::I64,
                        token,
                    )?;
                    // Bound the token map (address-keying was bounded by overwrite; tokens are not):
                    // drop checkpoints whose `setjmp` activation has returned — a `longjmp` to one
                    // would trap regardless. A copied-but-live checkpoint sits at a still-live
                    // activation (`depth <= stack.len()`) and is retained.
                    let live = self.stack.len();
                    self.setjmp_points.retain(|_, p| p.depth <= live);
                    self.setjmp_points.insert(
                        token,
                        ByteSetJmp {
                            depth: self.stack.len(),
                            module,
                            cur,
                            base,
                            pc: pc + 1,
                            dst: *dst,
                        },
                    );
                    r!(*dst) = Reg::from_i32(0);
                    pc += 1;
                }
                // `<setjmp.h>` `longjmp`: pop the activation stack back to the checkpoint (intervening
                // activations discarded — C has no cleanups), restore its cursor, and re-enter with the
                // `setjmp` result set to `val` (a `0` becomes `1`, per C). A missing checkpoint or one
                // whose activation already returned traps in-sandbox (§3b totality).
                Op::LongJmp { buf, val } => {
                    // #1062 — read the token back from the (possibly-copied) `jmp_buf` and look it up.
                    let buf_addr = r!(*buf).i64() as u64;
                    let v = r!(*val).i32();
                    let resume = if v == 0 { 1 } else { v };
                    let token = mem
                        .as_ref()
                        .ok_or(Trap::Malformed)?
                        .load_scalar(buf_addr, 0, LoadOp::I64)?
                        .i64() as u64;
                    let point = *self.setjmp_points.get(&token).ok_or(Trap::Malformed)?;
                    if point.depth > self.stack.len() {
                        return Err(Trap::Malformed); // the setjmp activation already returned
                    }
                    self.stack.truncate(point.depth);
                    // #1146 — a longjmp OUT of an injected signal handler leaves it exactly like a
                    // return: pop each crossed guard and fire `handler_returned()` so the personality
                    // restores its block-during-handler mask and a LATER instance of the signal can
                    // deliver (bash's `throw_to_top_level` siglongjmps out of `sigint_sighandler` —
                    // without this, one ^C would silence the signal for the session). Guards record the
                    // post-push stack depth, so `g > self.stack.len()` after the truncate is exactly the
                    // crossed handlers; a longjmp WITHIN a handler crosses none and is untouched.
                    while self
                        .sig_handler_stack
                        .last()
                        .is_some_and(|&g| g > self.stack.len())
                    {
                        self.sig_handler_stack.pop();
                        if let Some((_, source)) = &signal_poll {
                            source.handler_returned();
                        }
                    }
                    module = point.module;
                    cur = point.cur;
                    base = point.base;
                    pc = point.pc;
                    c = resolve!(module);
                    self.regs[base + point.dst as usize] = Reg::from_i32(resume);
                }
                Op::Call { callee, args, dst } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let callee = *callee as usize;
                    #[cfg(feature = "callprof")]
                    if module == 0 {
                        callprof::hit(callee);
                    }
                    // wasm-JIT tier-up: a module-0 direct call to an eligible function surfaces to the
                    // host, which runs the emitted region and delivers the results. `argv` is the raw
                    // i64 arg slots; the host reads them per the callee's signature. Suspension-free by
                    // construction (`mixed_ok`), so this is a plain "fast call": spill past the op and
                    // resume with the results in `dst` (`deliver_tierup`), exactly like an interp call.
                    if module == 0
                        && self
                            .jit_eligible
                            .as_ref()
                            .is_some_and(|e| e.get(callee).copied().unwrap_or(false))
                    {
                        // #717 host sync: snapshot the window's scalar committed extent for the host
                        // to write into the emitted `"mapped"` global. A window whose page state is
                        // not representable by one bound (sparse grow, `Ro`/`Unmapped`/aliased pages)
                        // declines tier-up — fall through to the interpreted call below, which honors
                        // the full per-page map (fail-closed; the interpreter is always right). A
                        // memory-less module has nothing to bound (no emitted access): sync `0`.
                        //
                        // #750 paged tier: the emitted code carries a per-access page check, so an
                        // unrepresentable window must NOT decline — surface with the reserved size
                        // (the bound must never under-admit a page the driver's table admits).
                        let extent = match mem.as_ref() {
                            None => Some(0),
                            Some(m) if self.jit_page_checked => Some(m.reserved_size()),
                            Some(m) => m.scalar_extent(),
                        };
                        if let Some(mapped) = extent {
                            let argv: Box<[i64]> = args.iter().map(|a| r!(*a).i64()).collect();
                            let results: Box<[ValType]> = c.result_types[callee].clone().into();
                            // Spill past the call with the caller's window intact (no callee frame
                            // pushed); `deliver_tierup` writes the results into `dst` relative to
                            // this base.
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::TierUp {
                                func: callee as u32,
                                argv,
                                dst: *dst as usize,
                                results,
                                mapped,
                            });
                        }
                    }
                    // A direct call stays in the current module.
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack
                        .push((module, cur, base, pc + 1, base + *dst as usize));
                    cur = callee;
                    base = nb;
                    pc = 0;
                }
                Op::CallIndirect {
                    idx,
                    args,
                    dst,
                    want_params,
                    want_results,
                } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                                       // Resolve through the **runtime dispatch table** (slot ⇒ (module, func)); an empty
                                       // padding slot or a signature mismatch is an inert IndirectCallType trap. The
                                       // target may be an installed §22 unit (a different module) — a cross-module call.
                    let slot = (r!(*idx).i32() as u32 as usize) & (table.len() - 1);
                    let ts = table.slot(slot);
                    if ts.module == super::TABLE_EMPTY {
                        return Err(Trap::IndirectCallType);
                    }
                    let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
                    let tm = resolve!(tmod);
                    let (cp, cr) = &tm.sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let nb = base + c.progs[cur].nslots as usize;
                    let need = nb + tm.progs[tfunc].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    for (i, a) in args.iter().enumerate() {
                        self.regs[nb + i] = self.regs[base + *a as usize];
                    }
                    self.stack
                        .push((module, cur, base, pc + 1, base + *dst as usize));
                    if tmod != module {
                        module = tmod;
                        c = tm;
                    }
                    cur = tfunc;
                    base = nb;
                    pc = 0;
                }
                Op::Ret { srcs } => {
                    // #1146 — if this returns from an injected async signal handler (its guard entry
                    // records the stack depth at injection, which equals the current depth when the
                    // handler's own `Ret` fires), restore the personality's block-during-handler mask
                    // before unwinding, exactly as the tree-walker does on the handler's `Return`. A
                    // held fatal signal exposed by the unmask fires its default action inside
                    // `handler_returned` (a no-op here unless the driver wired `set_kill`).
                    if self.sig_handler_stack.last() == Some(&self.stack.len()) {
                        self.sig_handler_stack.pop();
                        if let Some((_, source)) = &signal_poll {
                            source.handler_returned();
                        }
                    }
                    match self.stack.pop() {
                        None => {
                            let tys = &c.result_types[cur];
                            return Ok(Outcome::Done(
                                srcs.iter()
                                    .zip(tys)
                                    .map(|(s, ty)| self.regs[base + *s as usize].to_value(*ty))
                                    .collect(),
                            ));
                        }
                        Some((cmod, cprog, cbase, cpc, ret_abs)) => {
                            for (i, s) in srcs.iter().enumerate() {
                                self.regs[ret_abs + i] = self.regs[base + *s as usize];
                            }
                            if cmod != module {
                                module = cmod;
                                c = resolve!(cmod);
                            }
                            cur = cprog;
                            base = cbase;
                            pc = cpc;
                        }
                    }
                }
                // Tail calls reuse the *current* window (`base` unchanged) instead of pushing a
                // return entry, so the callee returns to this activation's caller. Args may alias the
                // destination prefix, so gather into `scratch` then scatter (like edge copies).
                Op::TailCall { callee, args } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let callee = *callee as usize;
                    #[cfg(feature = "callprof")]
                    if module == 0 {
                        callprof::hit(callee);
                    }
                    let need = base + c.progs[callee].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    self.scratch.clear();
                    for a in args.iter() {
                        self.scratch.push(self.regs[base + *a as usize]);
                    }
                    for (i, &v) in self.scratch.iter().enumerate() {
                        self.regs[base + i] = v;
                    }
                    cur = callee;
                    pc = 0;
                }
                Op::TailCallIndirect {
                    idx,
                    args,
                    want_params,
                    want_results,
                } => {
                    step(fuel, None)?; // fuel unification: function-entry safepoint
                    let slot = (r!(*idx).i32() as u32 as usize) & (table.len() - 1);
                    let ts = table.slot(slot);
                    if ts.module == super::TABLE_EMPTY {
                        return Err(Trap::IndirectCallType);
                    }
                    let (tmod, tfunc) = (ts.module as usize, ts.func as usize);
                    let tm = resolve!(tmod);
                    let (cp, cr) = &tm.sigs[tfunc];
                    if cp.as_slice() != &want_params[..] || cr.as_slice() != &want_results[..] {
                        return Err(Trap::IndirectCallType);
                    }
                    let need = base + tm.progs[tfunc].nslots as usize;
                    if self.regs.len() < need {
                        self.regs.resize(need, Reg::default());
                    }
                    self.scratch.clear();
                    for a in args.iter() {
                        self.scratch.push(self.regs[base + *a as usize]);
                    }
                    for (i, &v) in self.scratch.iter().enumerate() {
                        self.regs[base + i] = v;
                    }
                    if tmod != module {
                        module = tmod;
                        c = tm;
                    }
                    cur = tfunc;
                    pc = 0;
                }
                Op::CapCall {
                    type_id,
                    op,
                    handle,
                    params,
                    args,
                    dst,
                    results,
                } => {
                    // Generic synchronous powerbox dispatch — the same path and ABI the tree-walker's
                    // generic `CapCall` arm uses (`cap_dispatch_slots`): handle as an i32, args/results
                    // as i64 slots, results re-typed by the call's `sig.results`. Via [`HostCell`] so a
                    // parallel vCPU takes the shared-host lock only for this one call (4c-host); the
                    // cooperative path is exclusive (uncontended), so order is unchanged.
                    // `u32::MAX` = no handle operand (a v8 `call.import` — the slot binding
                    // identifies the capability; the dispatch ignores the value).
                    let h = if *handle == u32::MAX {
                        0
                    } else {
                        r!(*handle).i32()
                    };
                    let mut argv: Vec<i64> = Vec::with_capacity(args.len());
                    for a in args.iter() {
                        argv.push(r!(*a).i64());
                    }
                    // #1904 — the oracle's stop see-through (#1672): a stopped domain runs under a
                    // landing freeze only to reach its next freeze point, so nothing may leave it on the
                    // way — a host call is abandoned rather than performed, for the thaw to re-issue.
                    // (A serve op is `Op::SvcPoll`, which the freeze already makes inert.)
                    if signal_poll.as_ref().is_some_and(|(_, s)| s.stopped())
                        && self.abandon_for_freeze(mem, host)
                    {
                        for i in 0..results.len() {
                            self.regs[base + *dst as usize + i] = Reg::default();
                        }
                        pc += 1;
                        continue;
                    }
                    // FORK.md §8.6 / #1080 rung 4 — `pipe(fds)` (CAP_SELF op 16): the one mint
                    // ([`super::Host::mint_pipe`]). The generic dispatch below declines it (an engine
                    // serves it only where its reads and writes can park), so it is serviced here.
                    if *type_id == temen_ir::CAP_SELF_TYPE_ID && *op == super::CAP_SELF_PIPE {
                        let fds = argv.first().copied().unwrap_or(0) as u64;
                        let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                        let r = host.with(|p| p.mint_pipe(fds, gm));
                        if !results.is_empty() {
                            self.regs[base + *dst as usize] = Reg::from_i64(r);
                        }
                        pc += 1;
                        continue;
                    }
                    // §22: an **import-bound** `Jit` driver op (`invoke`/`install`/`uninstall`) can't be
                    // serviced by the generic `cap_dispatch_slots` — it needs the scheduler-owning
                    // driver, exactly like a *static* `call.cap (JIT, op)` (which lowers straight to
                    // `Op::JitInvoke`). Resolve the binding and surface the same driver `Outcome`, so a
                    // self-hosted guest whose `__vm_jit_*` are lowered to `call.import` (temen-llvm) drives
                    // the Jit cap on the bytecode engine too (the pure-host `compile`/`compile_linked`
                    // ops 0/5 stay on `cap_dispatch_slots` below). The submitted unit is re-verified by
                    // the embedder's `jit_validator` before it runs — the security hinge is unchanged.
                    if *type_id == temen_ir::CAP_IMPORT_TYPE_ID {
                        if let Some(b) = host.with(|p| p.import_binding(*op)) {
                            if b.bound
                                && b.type_id == super::cap_id::JIT
                                && matches!(b.op, 1 | 3 | 4)
                            {
                                if argv.is_empty() {
                                    return Err(Trap::CapFault); // invoke/install/uninstall need arg0
                                }
                                self.module = module;
                                self.cur = cur;
                                self.base = base;
                                self.pc = pc + 1;
                                return Ok(match b.op {
                                    3 => Outcome::JitInstall {
                                        h: b.handle,
                                        code: argv[0] as i32,
                                        dst: *dst,
                                    },
                                    4 => Outcome::JitUninstall {
                                        h: b.handle,
                                        slot: argv[0],
                                        dst: *dst,
                                    },
                                    _ => Outcome::JitInvoke {
                                        h: b.handle,
                                        code: argv[0] as i32,
                                        argv: argv[1..].to_vec().into_boxed_slice(),
                                        dst: *dst,
                                        // The unit entry's params are the call's params minus arg0 (code).
                                        params: params
                                            .get(1..)
                                            .unwrap_or(&[])
                                            .to_vec()
                                            .into_boxed_slice(),
                                        results: results.clone(),
                                    },
                                });
                            }
                        }
                    }
                    // §3.6 (I36 slice 2) — caller-side parking: a call through a live-callee
                    // offer never reaches the generic dispatch. It enqueues on the callee's
                    // inbound queue and parks this task until the handler's reply (the
                    // tree-walker's caller-parking arm, task-level). A full callee queue is
                    // probeable backpressure (`EAGAIN` as the call's result), never a trap.
                    if let Some((callee, export)) = host.with(|p| p.live_impl_of(h, *type_id)) {
                        let t = callee.lock_unpoisoned().svc_enqueue(export, *op, argv);
                        if let Some(ticket) = t {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::LiveCall {
                                ticket,
                                callee,
                                dst: *dst,
                            });
                        }
                        if !results.is_empty() {
                            self.regs[base + *dst as usize] = Reg::from_i64(super::EAGAIN);
                        }
                        pc += 1;
                        continue;
                    }
                    let gm = mem.as_mut().map(|m| m as &mut dyn GuestMem);
                    let mut pending_id = None;
                    // #1173 — take this dispatch's PER-OP park/interrupt flags in the SAME lock scope
                    // as the dispatch that sets them. They live on the shared `Host`, and on the
                    // PARALLEL driver every vCPU of a domain shares one, so draining them under a
                    // later, separate `with` let a sibling thread's unrelated dispatch — landing in
                    // the gap — take THIS op's flags: the parked reader then fell through to the
                    // non-park arm and kept the placeholder `0` its rewound read had returned (a pipe
                    // read answering EOF that never happened), while the sibling wrote the `-EINTR`
                    // into its own `dst`. That is the `EINTR(42)` → `0` flake, at ~20% per attempt.
                    // Under one lock the flags are what they always meant to be: extra return values
                    // of this dispatch, unreachable by any other vCPU.
                    let (res, parks) = host.with(|p| {
                        let r = p.cap_dispatch_slots_pending(
                            *type_id,
                            *op,
                            h,
                            &argv,
                            gm,
                            &mut pending_id,
                        );
                        // Drain unconditionally (the tree-walker's park site does the same): a flag
                        // only ever means "the op *this* dispatch just ran wants to park", so one
                        // left set would misfire on a later, unrelated op. The wake flags need no
                        // action here — the drivers poll readiness at their settle.
                        (r, DispatchParks::take(p))
                    });
                    let res = res?;
                    // §12 parking-on-blocking: a punted offloadable dispatch. The `with` scope
                    // above already released the shared-host lock. The exactly-`i64` case is
                    // surfaced as [`Outcome::CapPending`] so the DRIVER chooses the wait shape
                    // (F2: the cooperative `drive` fiber-parks a punting fiber; every other
                    // driver waits inline — the I45 posture). Other reply shapes keep the
                    // slice-1 inline wait right here; the placeholder `res` is discarded.
                    if let Some(id) = pending_id {
                        if results.len() > 1 {
                            // Parkable ops carry a single-slot scalar reply (invariant 8) —
                            // a wider declared signature is a registration bug, fail-closed
                            // BEFORE any park or wait.
                            return Err(Trap::CapFault);
                        }
                        if let [ValType::I64] = &results[..] {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc + 1;
                            return Ok(Outcome::CapPending { id, dst: *dst });
                        }
                        let comps = host.with(|p| p.completions());
                        let r = comps.wait(id);
                        if let Some(ty) = results.first() {
                            self.regs[base + *dst as usize] = Reg::from_value(slot_to_val(*ty, r));
                        }
                        pc += 1;
                        continue;
                    }
                    // Blocking-stdin park: a `Stream{In}` `read` (type 0, op 0) whose buffer was empty
                    // under `Host::set_stdin_blocking` yields here instead of completing. Do NOT write
                    // results or advance `pc`: persist state at *this* instruction so the driver, after
                    // pushing more input, re-issues the read on resume. Gated on the stream-read op so
                    // no other call.cap pays the flag check.
                    // A `call.import` dispatch carries the `CAP_IMPORT_TYPE_ID` sentinel — read
                    // its bound `(type_id, op)` so an imported stdin `read` parks exactly like the
                    // resolved `call.cap` form (IMPORTS.md phase 3).
                    let (eff_tid, eff_op) = if *type_id == temen_ir::CAP_IMPORT_TYPE_ID {
                        host.with(|p| p.import_binding(*op))
                            .map(|b| (b.type_id, b.op))
                            .unwrap_or((*type_id, *op))
                    } else {
                        (*type_id, *op)
                    };
                    if eff_tid == super::cap_id::STREAM && eff_op == 0 && parks.stdin {
                        // #1146 (deeper) — the stdin park's EINTR leg. The scheduler drivers' interrupt
                        // paths (the cooperative all-parked sweep / the parallel poll break) latch
                        // `set_sig_interrupt` and re-admit this task; the rewound read re-executes and
                        // lands HERE. Without this leg the re-run would re-park unconditionally and the
                        // latched interrupt would be silently dropped (a livelock, not a visible hang) —
                        // the EINTR completion for a stdin park lives in `resume`, not only in the pump.
                        // Mirror the pipe park site below: drain the transient flag unconditionally (so a
                        // stale interrupt can never leak into a later read), and on a deliverable
                        // non-`SA_RESTART` signal — the latched flag, or one already pending at the park
                        // insert (the pre-park race) — complete `-EINTR` in `dst` and advance instead of
                        // parking; the caught handler is delivered at the next safepoint. `SA_RESTART`
                        // leaves it to re-park (data resumes it).
                        let interrupted = (parks.sig_interrupt
                            && !host.with(|p| p.signal_restart()))
                            || host.with(|p| p.park_interrupted());
                        if interrupted {
                            self.regs[base + *dst as usize] = Reg::from_i64(temen_ir::errno::EINTR);
                            pc += 1;
                            continue;
                        }
                        if self.abandon_for_freeze(mem, host) {
                            self.set_results(base + *dst as usize, &res, results);
                            pc += 1;
                            continue;
                        }
                        self.module = module;
                        self.cur = cur;
                        self.base = base;
                        self.pc = pc;
                        return Ok(Outcome::StdinPark);
                    }
                    // #799/#1080 — a personality caller-request (`fork` / blocking `waitpid`) rides this
                    // dispatch: bash's `fork`/`waitpid` are host-proc ops that set a park request during
                    // their own dispatch. Always DRAIN it (so it never leaks into the next op); act on it
                    // only from the ROOT fiber (a serve handler / coroutine keeps the placeholder answer,
                    // as the tree-walker keeps it off the Real scheduler / root fiber). `fork` advances
                    // past the op — the driver writes both return-twice replies; `waitpid` REWINDS it, so
                    // it re-executes on wake after the driver parks the task. The cooperative driver owns
                    // both ([`Outcome::ForkSelf`]/[`Outcome::ReapWait`]); other drivers `ThreadFault`.
                    if let Some(ev) = parks.request {
                        // A blocking `waitpid` would park: under a landing freeze it is abandoned.
                        if matches!(
                            ev,
                            super::ParkEvent::TaskExit(_) | super::ParkEvent::TaskExitAny
                        ) && self.abandon_for_freeze(mem, host)
                        {
                            self.set_results(base + *dst as usize, &res, results);
                            pc += 1;
                            continue;
                        }
                        self.module = module;
                        self.cur = cur;
                        self.base = base;
                        match ev {
                            // `fork` advances past the op; the driver writes both return-twice replies
                            // (a non-root/non-bare caller degrades to `-EAGAIN` at the driver's `bare`
                            // gate — never the wrong image). `waitpid` REWINDS the op so it re-executes
                            // on wake after the driver parks the task. Cooperative-driver-only; a
                            // non-cooperative driver `ThreadFault`s these (like `Exec`).
                            super::ParkEvent::ForkSelf => {
                                self.pc = pc + 1;
                                return Ok(Outcome::ForkSelf { dst: *dst });
                            }
                            super::ParkEvent::TaskExit(id) => {
                                self.pc = pc;
                                return Ok(Outcome::ReapWait {
                                    child: Some(id as usize),
                                });
                            }
                            super::ParkEvent::TaskExitAny => {
                                self.pc = pc;
                                return Ok(Outcome::ReapWait { child: None });
                            }
                            // #1609 — `execve` through the personality. The personality already
                            // resolved the path against its command registry and packed argv/envp
                            // into the powerbox args region, so all that is left is the image
                            // -replace the op-14 route already surfaces: reuse [`Outcome::Exec`]
                            // verbatim rather than growing a parallel variant through the driver.
                            // Empty grant list + entry 0 = the self-contained-command shape
                            // `exec.c` passes; `size_log2` has been advisory since #773 (the real
                            // bound is the caller's own window, checked in the builder). Advances
                            // past the op like `fork`: a refused exec lands its errno in `dst`, a
                            // successful one never returns to this activation.
                            super::ParkEvent::ExecSelf { cmd } => {
                                self.pc = pc + 1;
                                return Ok(Outcome::Exec {
                                    cmd,
                                    grants_ptr: 0,
                                    grants_n: 0,
                                    entry: 0,
                                    size_log2: 0,
                                    dst: *dst,
                                    personality: true,
                                });
                            }
                            // `posix_spawn` advances past the op: the driver writes the new pid
                            // (or the errno) into `dst`. A source that asked without staging has
                            // nothing to start, and the op's placeholder stands.
                            super::ParkEvent::SpawnSelf { cmd } => {
                                if let Some(plan) = parks.spawn {
                                    self.pc = pc + 1;
                                    return Ok(Outcome::SpawnSelf {
                                        cmd,
                                        plan,
                                        dst: *dst,
                                    });
                                }
                            }
                        }
                    }
                    // #1080 rung 4 — a personality blocking pipe read/write rides this dispatch (a
                    // command's `read`/`write` are `call.sym` imports bound to pipe ends). DRAIN all four
                    // pipe flags (so none leaks); a read on an empty FIFO with writers open, or a write to
                    // a full FIFO with readers open, REWINDS the op and parks the task on the pipe — the
                    // settle scan re-admits it when ready. The wake flags a write/close set need no action
                    // here: the cooperative driver POLLS pipe readiness at the settle (no `pipe_waiters`).
                    // #1146 slice 2 — drain the transient EINTR flag ([`Host::set_sig_interrupt`], set by
                    // the all-parked sweep) alongside the park flags, unconditionally: consuming it even on
                    // the non-parking path keeps a mixed feed (bytes AND a signal in one `feed_terminal`)
                    // from leaving it set to spuriously interrupt a *later* read. It only *acts* below.
                    let (pipe_read_park, pipe_write_park, sig_flag) =
                        (parks.pipe_read, parks.pipe_write, parks.sig_interrupt);
                    // A signal interrupted this blocking pipe read/write: either the sweep set the flag
                    // above, or a deliverable signal is already pending at the park insert (the slice-D
                    // pre-park race). When so — and the delivery does not carry `SA_RESTART` — complete
                    // `-EINTR` in `dst` and advance, exactly as the tree-walker's eval-loop park site does,
                    // instead of rewinding+parking; the caught handler itself is delivered at the next
                    // safepoint (the slice-1 redirect). `SA_RESTART` leaves the op to re-park (data resumes
                    // it). The restart / pre-park peek is taken only when a park is actually pending.
                    if pipe_read_park.is_some() || pipe_write_park.is_some() {
                        let interrupted = (sig_flag && !host.with(|p| p.signal_restart()))
                            || host.with(|p| p.park_interrupted());
                        if interrupted {
                            self.regs[base + *dst as usize] = Reg::from_i64(temen_ir::errno::EINTR);
                            pc += 1;
                        } else if self.abandon_for_freeze(mem, host) {
                            self.set_results(base + *dst as usize, &res, results);
                            pc += 1;
                        } else if let Some(pipe) = pipe_read_park {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc; // rewind: the read re-executes on wake
                            return Ok(Outcome::PipeRead { pipe });
                        } else {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc; // rewind: the write re-executes on wake
                            return Ok(Outcome::PipeWrite {
                                pipe: pipe_write_park.expect("write park present"),
                            });
                        }
                    } else {
                        self.set_results(base + *dst as usize, &res, results);
                        pc += 1;
                        // #1198 — this syscall STOPPED its own domain (a background terminal read/write
                        // hit `tty_background_check` → SIGTTIN/SIGTTOU default-action stop; or a `^Z`
                        // SIGTSTP). The tree-walker benches a stopped domain at its per-op `stop_flag`
                        // safepoint, so a libc restart loop (`while (r == -ERESTART) read();`) parks after
                        // one turn. The bytecode engine has no per-op poll, so that loop would spin here
                        // forever. Yield to the pump at THIS syscall boundary — the result is already
                        // written and `pc` is past the op, so the resume re-executes the *next* op on
                        // wake — and let the round-robin pick bench the stopped domain (it re-admits at
                        // SIGCONT). Gated on a signal personality (`signal_poll` is `None` for a
                        // pure-compute guest), and only reached when a syscall completed inline (no park),
                        // so the common syscall path pays a single already-cached predicate.
                        if signal_poll.as_ref().is_some_and(|(_, s)| s.stopped()) {
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc;
                            return Ok(Outcome::Suspended);
                        }
                    }
                }
                Op::SvcPoll { dst, wait } => {
                    // §3.6 serve-loop core (I36 slice 1), the tree-walk serve arm's rewind state
                    // machine in register-window form. A handler that just returned re-entered
                    // this op via its rewound linkage with its result in `dst` — settle it into
                    // the ticket's completion cell. No cross-domain caller can be parked on the
                    // ticket in this engine yet (caller-side parking is a later I36 slice), so
                    // the reply always rides the cell — the tree-walker's unclaimed-result path.
                    if let Some(t) = self.serve_ticket.take() {
                        let v = self.regs[base + *dst as usize].i64();
                        host.with(|p| p.svc_results.insert(t, v));
                        self.serve_count += 1;
                    }
                    // DURABILITY.md §13.4 slice 4b, the oracle's rule: under `UNWINDING` a serve op
                    // makes **no progress** — it delivers an inert sentinel so its trailing poll
                    // spills with the queue untouched, and the transform's `SvcServe` re-issue arm
                    // re-executes the drain on thaw (#1904).
                    if is_unwinding(mem) && host.with(|p| p.is_durable()) {
                        self.regs[base + *dst as usize] = Reg::from_i64(0);
                        self.serve_count = 0;
                        pc += 1;
                        continue;
                    }
                    // Admit queued dispatches: un-servable ones settle inline with a probeable
                    // errno (the dispatch's fault, never the domain's — it keeps serving); the
                    // first servable one switches into a handler activation whose return linkage
                    // re-executes this op (pc deliberately NOT advanced).
                    let mut admitted = false;
                    loop {
                        let d = host.with(|p| p.svc_queue.pop_front());
                        let Some(d) = d else { break };
                        // The queue only holds servable dispatches (checked at enqueue), so a
                        // missing handler here is host-state corruption: fail closed. Handlers
                        // are the domain's home-module functions (`self.home` — the primary, or
                        // a separate-module child's own unit); serving from any other unit
                        // would resolve indices against the wrong program table.
                        let fidx = host
                            .with(|p| p.svc_handler_func(d.export, d.op))
                            .ok_or(Trap::CapFault)? as usize;
                        if module != self.home {
                            return Err(Trap::CapFault);
                        }
                        let (params, _) = c.sigs.get(fidx).ok_or(Trap::CapFault)?;
                        if d.args.len() != params.len() {
                            host.with(|p| p.svc_results.insert(d.ticket, super::EINVAL));
                            continue;
                        }
                        let nb = base + c.progs[cur].nslots as usize;
                        let need = nb + c.progs[fidx].nslots as usize;
                        if self.regs.len() < need {
                            self.regs.resize(need, Reg::default());
                        }
                        for (i, (s, ty)) in d.args.iter().zip(params.iter()).enumerate() {
                            self.regs[nb + i] = Reg::from_value(slot_to_val(*ty, *s));
                        }
                        self.stack
                            .push((module, cur, base, pc, base + *dst as usize));
                        self.serve_ticket = Some(d.ticket);
                        cur = fidx;
                        base = nb;
                        pc = 0;
                        admitted = true;
                        break;
                    }
                    if !admitted {
                        if *wait && self.serve_count == 0 {
                            // svc.wait with no progress: persist the cursor AT this op (a wake
                            // re-executes the whole drain) and park the task on its domain.
                            self.module = module;
                            self.cur = cur;
                            self.base = base;
                            self.pc = pc;
                            return Ok(Outcome::SvcWait);
                        }
                        // Queue drained: deliver the completed count and close the activation.
                        self.regs[base + *dst as usize] = Reg::from_i64(self.serve_count);
                        self.serve_count = 0;
                        pc += 1;
                    }
                }
                Op::CapSelfExt { op, handle, dst } => {
                    // §3.5 self-namespace extensions — through the shared &mut dispatch entry
                    // (interning / reification mutate host state), same as the tree-walker.
                    let argv: Vec<i64> = match handle {
                        Some(h) => vec![r!(*h).i32() as i64],
                        None => Vec::new(),
                    };
                    let res = host.with(|p| {
                        p.cap_dispatch_slots(temen_ir::CAP_SELF_TYPE_ID, *op, 0, &argv, None)
                    })?;
                    r!(*dst) = Reg::from_i32(*res.first().ok_or(Trap::CapFault)? as i32);
                    pc += 1;
                }
                // §12 fiber ops escape to `drive` (which owns the registry / resume chain). Each
                // advances past itself and persists the cursor, so the driver — after creating the
                // fiber, switching in, or switching back — resumes this activation right after the op
                // (with the op's `dst` slot(s) filled in by the driver).
                Op::ContNew { func, sp, dst } => {
                    let funcref = r!(*func).i32();
                    let spv = r!(*sp).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContNew {
                        funcref,
                        sp: spv,
                        dst,
                    });
                }
                Op::ContResume {
                    k,
                    arg,
                    dst,
                    blocking,
                } => {
                    // Fuel unification: charge one fuel per `cont.resume` op — the tree-walker charges
                    // the same at its `Inst::ContResume` arm. Resuming a fiber is a control transfer
                    // per-op fuel used to meter; without this, a long fiber-resume chain runs unmetered.
                    step(fuel, None)?;
                    let kh = r!(*k).i32();
                    let arg = r!(*arg).i64();
                    let dst = *dst;
                    let blocking = *blocking;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    // I48: a blocking park rewinds the resumer's cursor to THIS op (via `resume_ip`)
                    // so the wake re-executes it; `pc` here is this op's index (the cursor is written
                    // back as `pc + 1` for the ordinary switch/poll continuation).
                    let resume_ip = pc;
                    self.pc = pc + 1;
                    return Ok(Outcome::ContResume {
                        kh,
                        arg,
                        dst,
                        blocking,
                        resume_ip,
                    });
                }
                Op::Suspend { value, dst } => {
                    let value = r!(*value).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::FiberSuspend { value, dst });
                }
                // §12 multi-vCPU ops escape to the `drive` scheduler (which owns the task set). Each
                // advances past itself and persists the cursor, so the scheduler resumes this
                // activation right after the op with the op's `dst` filled in.
                Op::ThreadSpawn { func, sp, arg, dst } => {
                    let sp = r!(*sp).i64();
                    let arg = r!(*arg).i64();
                    let (func, dst) = (*func, *dst);
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    // `module` is the executing frame's module: an installed unit's spawn resolves
                    // `func` in the unit, not module 0 (the verifier checked it there).
                    return Ok(Outcome::ThreadSpawn {
                        func,
                        sp,
                        arg,
                        dst,
                        module,
                    });
                }
                Op::ThreadJoin { handle, dst } => {
                    let handle = r!(*handle).i32();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ThreadJoin { handle, dst });
                }
                // §14 executor children — the Instantiator authority `(ibase, isize)` is resolved here
                // (a forged/ungranted cap is an inert CapFault in place), then the driver builds the
                // confined child (it owns the task set + the per-child environments).
                Op::ChildOffer {
                    handle,
                    child,
                    export,
                    dst,
                } => {
                    // The family-level authority check (as the tree-walker's Instantiator arm):
                    // a forged/wrong-type handle is a CapFault before the op logic runs.
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?;
                    let child = r!(*child).i32();
                    let export = r!(*export).i64() as u32;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ChildOffer { child, export, dst });
                }
                Op::CloneCaller {
                    args,
                    dst,
                    has_result,
                } => {
                    // Arity picks the mode (mirrors the oracle, `temen-interp` clone_caller arm):
                    // 2 args = explicit `(reply_orig, reply_twin)`; 0/1 args = pid mode
                    // (`reply_orig = None` → the parent gets the twin's task id).
                    let (reply_orig, reply_twin) = if args.len() >= 2 {
                        (Some(r!(args[0]).i64()), r!(args[1]).i64())
                    } else {
                        let twin = args.first().map(|a| r!(*a).i64()).unwrap_or(0);
                        (None, twin)
                    };
                    let dst = *dst;
                    let has_result = *has_result;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::CloneCaller {
                        reply_orig,
                        reply_twin,
                        dst,
                        has_result,
                    });
                }
                Op::Reap {
                    pid,
                    dst,
                    has_result,
                } => {
                    // The pid to reap (an out-of-range default → the driver answers -ECHILD).
                    let pid = pid.map(|p| r!(p).i64()).unwrap_or(-1);
                    let dst = *dst;
                    let has_result = *has_result;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Reap {
                        pid,
                        dst,
                        has_result,
                    });
                }
                // FORK.md §8.6 — `exec_module` (#1080): resolve the five register operands and surface
                // the image-replace to the driver. The cursor is saved (pc+1) so a *refused* exec
                // (driver writes `-EINVAL` to `dst`) resumes at the next op — the caller survives, POSIX.
                Op::ExecModule {
                    module: module_reg,
                    grants_ptr,
                    grants_n,
                    entry,
                    size_log2,
                    dst,
                } => {
                    let mh = r!(*module_reg).i64() as i32;
                    let grants_ptr = r!(*grants_ptr).i64() as u64;
                    let grants_n = r!(*grants_n).i64() as u64;
                    let entry = r!(*entry).i64() as u64;
                    let size_log2 = r!(*size_log2).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Exec {
                        cmd: super::ExecCmd::Granted(mh),
                        grants_ptr,
                        grants_n,
                        entry,
                        size_log2,
                        dst,
                        personality: false,
                    });
                }
                Op::Instantiate {
                    handle,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
                    // op 11: resolve the grant-list `(ptr, count)` from their registers (op 0 is None).
                    let grants = grants.map(|(pr, nr)| (r!(pr).i64() as u64, r!(nr).i64() as u64));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Instantiate {
                        spawn: ConfinedSpawn {
                            ibase,
                            isize: isz,
                            module: None,
                            entry,
                            off,
                            size_log2,
                            quota,
                            grants,
                            budget: 0,
                        },
                        dst,
                    });
                }
                // §14 separate-module executor child — like `Instantiate`, but the first arg is a
                // granted `Module` handle (the slot ABI crosses it as an i64; low 32 bits) whose
                // program the driver resolves + compiles + runs.
                Op::InstantiateModule {
                    handle,
                    module: module_reg,
                    entry,
                    off,
                    size_log2,
                    quota,
                    dst,
                    grants,
                } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let mh = r!(*module_reg).i64() as i32;
                    let entry = r!(*entry).i64();
                    let off = r!(*off).i64();
                    let size_log2 = r!(*size_log2).i64();
                    let quota = r!(*quota).i64();
                    // op 13: resolve the grant-list `(ptr, count)` from their registers (op 5 is None).
                    let grants = grants.map(|(pr, nr)| (r!(pr).i64() as u64, r!(nr).i64() as u64));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Instantiate {
                        spawn: ConfinedSpawn {
                            ibase,
                            isize: isz,
                            module: Some(mh),
                            entry,
                            off,
                            size_log2,
                            quota,
                            grants,
                            budget: 0,
                        },
                        dst,
                    });
                }
                // §5 detached spawn (op 15, #1286): the Instantiator is the authority (a forged one is a
                // CapFault, as every op above); the budget's quota take and the module resolve happen at
                // the driver's commit site (`admit_detached_child`), peek-then-drain like op 13.
                Op::InstantiateDetached {
                    handle,
                    budget,
                    module: module_reg,
                    grants,
                    entry,
                    size_log2,
                    quota,
                    args,
                    premap,
                    dst,
                } => {
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?;
                    let spawn = DetachedSpawn {
                        budget: r!(*budget).i64() as i32,
                        module: r!(*module_reg).i64() as i32,
                        entry: r!(*entry).i64(),
                        size_log2: r!(*size_log2).i64(),
                        quota: r!(*quota).i64(),
                        grants: grants
                            .map(|(pr, nr)| (r!(pr).i64() as u64, r!(nr).i64() as u64))
                            .filter(|(_, n)| *n != 0),
                        args: args.map(|(pr, lr)| (r!(pr).i64() as u64, r!(lr).i64() as u64)),
                        premap: premap.map(|(rr, or)| (r!(rr).i64() as i32, r!(or).i64() as u64)),
                    };
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::InstantiateDetached { spawn, dst });
                }
                // CONSOLIDATION.md §3d — `instantiate_rec` (op 17): read the 56-byte record from
                // this vCPU's confined window and fail closed exactly as the tree-walker's arm does
                // (bad version / pager / budget-quota mix / dangling budget handle → `CapFault`).
                // This tier never natively demand-pages: the module-level entries decline any
                // op-17 module with impl exports ([`compile_module_for`]), so a surviving pager
                // field can only come from an export-less module — which `CapFault`s identically
                // on the tree-walker. Geometry validation and the budget **drain** stay in the one
                // admission (`admit_confined_child`), after every refusal, so a refused spawn
                // leaves the budget intact — the tree-walker's peek-then-drain discipline. (One
                // known error-order seam, shared with ops 11/13: an invalid grant list *plus* bad
                // geometry lands `-EINVAL` here but `CapFault` on the tree-walker, because the
                // admission parses grant records after the geometry.)
                Op::InstantiateRec { handle, rec, dst } => {
                    let ih = r!(*handle).i32();
                    let (ibase, isz) = host.with(|p| p.resolve_instantiator(ih))?;
                    let rp = r!(*rec).i64() as u64;
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let head = m.read_window(rp, 56)?;
                    let head: &[u8; 56] =
                        head.as_slice().try_into().map_err(|_| Trap::Malformed)?;
                    // The version word says how long the record is (v0 carve, v1 detached).
                    let len = SpawnRec::len_for(head).ok_or(Trap::CapFault)?;
                    let raw = m.read_window(rp, len)?;
                    // Shared layout decode (#911); pager/budget handling stays tier-local.
                    let sr = SpawnRec::parse(&raw).ok_or(Trap::CapFault)?; // version / reserved — fail closed
                    if sr.detached {
                        // #1863: a v1 record is op 15 as data — the same outcome op 15 produces,
                        // served by every driver's detached arm. No pager on this tier (see above).
                        if sr.pager != u32::MAX {
                            return Err(Trap::CapFault);
                        }
                        let dst = *dst;
                        self.module = module;
                        self.cur = cur;
                        self.base = base;
                        self.pc = pc + 1;
                        let spawn = DetachedSpawn {
                            budget: sr.budget,
                            module: sr.modh,
                            entry: sr.entry as i64,
                            size_log2: sr.size_log2,
                            quota: sr.quota,
                            grants: (sr.grants_n > 0).then_some((sr.grants_ptr, sr.grants_n)),
                            args: (sr.args.1 > 0).then_some(sr.args),
                            premap: (sr.region >= 0).then_some((sr.region, sr.child_off)),
                        };
                        return Ok(Outcome::InstantiateDetached { spawn, dst });
                    }
                    let entry = sr.entry as i64;
                    let off = sr.off as i64;
                    let size_log2 = sr.size_log2;
                    let modh = sr.modh;
                    let budget = sr.budget;
                    let quota = sr.quota;
                    if sr.pager != u32::MAX {
                        return Err(Trap::CapFault); // no impl exports here (see above) — fail closed
                    }
                    if budget != 0 {
                        if quota != 0 {
                            return Err(Trap::CapFault); // budget + raw quota is ambiguous
                        }
                        // Validate the handle now (dangling → CapFault, as the tree-walker);
                        // the drain waits for the drivers' commit.
                        host.with(|p| p.peek_budget(budget).map(|_| ()).ok_or(Trap::CapFault))?;
                    }
                    let grants = (sr.grants_n > 0).then_some((sr.grants_ptr, sr.grants_n));
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::Instantiate {
                        spawn: ConfinedSpawn {
                            ibase,
                            isize: isz,
                            module: (modh >= 0).then_some(modh),
                            entry,
                            off,
                            size_log2,
                            quota,
                            grants,
                            budget,
                        },
                        dst,
                    });
                }
                // §14 `join` — check the Instantiator authority, then reuse the thread join machinery
                // (executor children live in the same `threads` handle namespace as `thread.spawn`).
                Op::InstJoin { handle, child, dst } => {
                    let ih = r!(*handle).i32();
                    host.with(|p| p.resolve_instantiator(ih))?; // authority
                    let handle = r!(*child).i32();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::ThreadJoin { handle, dst });
                }
                Op::MemoryWait {
                    ty,
                    addr,
                    expected,
                    timeout,
                    dst,
                } => {
                    // Validate the address (confine/align/prot — traps surface here), mirroring
                    // `Inst::MemoryWait`; the scheduler does the value compare + park/wake.
                    let width = super::atomic_width(*ty);
                    let a = r!(*addr).i64() as u64;
                    let expected = r!(*expected).lo & super::width_mask(width);
                    let to_ns = r!(*timeout).i64();
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base_addr = m.prepare_wait(a, *ty)?;
                    // #1638 / #1641 — the guest's timeout, **unclamped**, and `None` for an
                    // infinite wait. `MAX_WAIT` used to collapse both of those into an ordinary
                    // deadline, which is two bugs in one line: an infinite wait got a deadline
                    // nobody asked for (the logical clock then advanced to it and delivered
                    // `WAIT_TIMED_OUT` — the #1638 divergence from the oracle), and a finite wait
                    // longer than the cap was silently truncated to it (two waiters asking 30 s
                    // and 20 s tie at 10 s and wake in the wrong order — the #1641 shape, which
                    // a logical clock does not excuse: it reorders the wakes either way).
                    //
                    // `None` is not "no deadline" as a special case for the schedulers to test —
                    // it is the absence of a clock-advance candidate, so "nothing runnable and
                    // every remaining waiter is indefinite" falls out of `.flatten().min()`
                    // returning `None` and takes each scheduler's existing deadlock exit.
                    let timeout = (to_ns >= 0).then_some(to_ns as u64);
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::MemoryWait {
                        base: base_addr,
                        expected,
                        width,
                        timeout,
                        dst,
                    });
                }
                Op::MemoryNotify { addr, count, dst } => {
                    let a = r!(*addr).i64() as u64;
                    let count = r!(*count).i32();
                    let m = mem.as_ref().ok_or(Trap::Malformed)?;
                    let base_addr = m.confine_for_notify(a)?;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::MemoryNotify {
                        base: base_addr,
                        count,
                        dst,
                    });
                }
                // §22 install/uninstall escape to the driver, which owns the (mutable) dispatch table
                // and module set. Authority is resolved there (a forged handle is an inert CapFault).
                Op::JitInstall { handle, code, dst } => {
                    let h = r!(*handle).i32();
                    let code = r!(*code).i64() as i32;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitInstall { h, code, dst });
                }
                Op::JitUninstall { handle, slot, dst } => {
                    let h = r!(*handle).i32();
                    let slot = r!(*slot).i64();
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitUninstall { h, slot, dst });
                }
                Op::JitInvoke {
                    handle,
                    code,
                    args,
                    dst,
                    params,
                    results,
                } => {
                    let h = r!(*handle).i32();
                    let code = r!(*code).i64() as i32;
                    let argv: Box<[i64]> = args.iter().map(|a| r!(*a).i64()).collect();
                    // `params`/`results` live in this op (in `mods`), which the driver may reallocate
                    // when it pushes the invoked unit — so hand owned copies up.
                    let (dst, params, results) = (*dst, params.clone(), results.clone());
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::JitInvoke {
                        h,
                        code,
                        argv,
                        dst,
                        params,
                        results,
                    });
                }
                Op::GcRoots {
                    lo,
                    hi,
                    mask,
                    buf,
                    cap,
                    dst,
                } => {
                    let lo = r!(*lo).i64() as u64;
                    let hi = r!(*hi).i64() as u64;
                    let mask = r!(*mask).i64() as u64;
                    // Security (GC.md §3/§6): the payload mask may only clear the top byte, else a host
                    // word could be folded into the guest window past the range filter. (The verifier
                    // rejects a constant fold-down mask; this defends an unverified / non-constant mask.)
                    if mask | 0xFF00_0000_0000_0000 != u64::MAX {
                        return Err(Trap::Malformed);
                    }
                    let buf = r!(*buf).i64() as u64;
                    let cap = r!(*cap).i64().max(0) as usize;
                    let dst = *dst;
                    self.module = module;
                    self.cur = cur;
                    self.base = base;
                    self.pc = pc + 1;
                    return Ok(Outcome::GcRoots {
                        lo,
                        hi,
                        mask,
                        buf,
                        cap,
                        dst,
                    });
                }
                Op::Unreachable => {
                    LAST_UNREACHABLE_FUNC.with(|c| c.set(cur as u32)); // #1382: record where we trapped
                    return Err(Trap::Unreachable);
                }
                Op::Eval {
                    inst,
                    block_base,
                    dst,
                } => {
                    // Run the op against this block's sub-window with its original block-local operand
                    // indices; reuse the reference semantics. `eval_inst` borrows the window immutably
                    // and `mem` mutably (disjoint), so we read the result before writing it back.
                    let win_lo = base + *block_base as usize;
                    let win_hi = base + c.progs[cur].nslots as usize;
                    let r = super::eval_inst(inst, &self.regs[win_lo..win_hi], mem)?;
                    if let Some(v) = r {
                        self.regs[base + *dst as usize] = v;
                    }
                    pc += 1;
                }
                Op::DurableShadowBase { dst } => {
                    // §12.8 4A.5: this context's shadow-SP word address (its own region base).
                    self.regs[base + *dst as usize] =
                        Reg::from_i64(self.durable_region_base as i64);
                    pc += 1;
                }
                Op::VcpuTlsGet { dst } => {
                    // §12 per-vCPU TLS read: this vCPU's word (seeded to its dense id, guest-overwritable).
                    self.regs[base + *dst as usize] = Reg::from_i64(self.tls);
                    pc += 1;
                }
                Op::VcpuTlsSet { val } => {
                    self.tls = r!(*val).i64();
                    pc += 1;
                }
            }
        }
    }
}
