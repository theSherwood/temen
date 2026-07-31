//! The **debuggee backend seam** (DEBUGGING.md G3). `DapServer` was hard-wired to the tree-walking
//! `svm_interp::Inspector`; this trait lets the same server drive *either* engine — the tree-walker
//! (the reference oracle, full feature set) or the **bytecode VM** the browser playground actually
//! runs, over `svm_interp::bytecode::DebugRun`.
//!
//! The bytecode engine covers breakpoints, stepping, backtrace, scalar/aggregate inspection,
//! **reverse debugging** (`seek`/`step_back`/`reverseContinue`, by deterministic replay — the debug
//! run is pure compute, so seeking to an earlier op clock rebuilds a fresh `DebugRun` and replays to
//! that many ops, restarting from the nearest **checkpoint** in a single-vCPU ladder so the replay is
//! bounded by the checkpoint stride rather than O(t) from clock 0), and **data breakpoints**
//! (`set_watchpoint` — a per-op check of the effective
//! address, computed like the interpreter's `access_of`, against the watched ranges; the run stops
//! *before* an op that touches one). For a spawn-free guest `supports_reverse`/`supports_watch` are
//! both `true`.
//!
//! **Multithreading** lands on its *own* engine — a `thread.spawn` guest launches the
//! `svm_interp::bytecode::ScheduledDebugRun`, a deterministic cooperative multi-vCPU **debug
//! scheduler**: breakpoints fire in whichever thread reaches them, `threads`/`stopped_task`/
//! `select_task` serve per-thread stacks, stepping (in/over/out) drives the stopped thread,
//! **cross-thread watchpoints** fire in whichever thread touches the range, and **reverse debugging**
//! works by deterministic replay to a global scheduler `turn` — *not* delegated to the tree-walker
//! (the differential oracle only, far too slow for any user-facing path). Both engines report
//! `supports_reverse`/`supports_watch` = `true`; the single-vCPU coordinate is the op `clock`, the
//! multithreaded one the global `turn`. Correctness is guaranteed by `crates/svm/tests/debug_parity.rs`
//! and `bytecode_debug_threads.rs` (engine level, vs the tree-walker oracle) and `dap_over_bytecode_*`
//! (server level).

use svm_interp::bytecode::{
    self, AccessSinkFn, DebugRun, DebugRunSnapshot, SchedBreak, SchedStop, ScheduledDebugRun,
    ScheduledSnapshot, ScheduledWrite,
};
use svm_interp::MemEvent;

use crate::json::Json;
use svm_interp::{
    cap_id, BoundImport, CapTape, FrameInfo, Host, Inspector, IrPc, SourceLoc, Stop, StopReason,
    StreamRole, Trap, Value, VarValue, WatchId, WatchKind,
};
use svm_ir::{FuncIdx, Module};

/// The on-ramp import-name → `(cap type_id, op)` policy for the I/O + memory caps a manifest program
/// (a chibicc `_start`) imports — a twin of the browser's `onramp_cap_resolver` and svm-run's
/// `default_cap_resolver`, kept small and self-contained here so `svm-dap` needn't depend on the
/// browser/runtime crates. Covers the caps a debugged C program actually reaches (`write`/`read`/`exit`
/// + the `vm_*` memory ops); anything else leaves its slot unbound (fail-closed at dispatch).
fn io_cap(name: &str) -> Option<(u32, u32)> {
    Some(match name {
        "write" => (cap_id::STREAM, 1),
        "read" => (cap_id::STREAM, 0),
        "exit" => (cap_id::EXIT, 0),
        "vm_map" => (cap_id::MEMORY, 0),
        "vm_unmap" => (cap_id::MEMORY, 1),
        "vm_protect" => (cap_id::MEMORY, 2),
        "vm_page_size" => (cap_id::MEMORY, 3),
        "vm_region_create" => (cap_id::ADDRESS_SPACE, 5),
        _ => return None,
    })
}

/// Grant the **on-ramp I/O powerbox** on `host` for module `m`: the §3e prefix (stdout/stdin/exit/
/// memory/addrspace), each registered under its `cap.self.resolve` name, plus the module's manifest
/// imports bound to it (IMPORTS.md phase 4 — [`io_cap`] maps each import name to the granted handle).
/// This is the powerbox a chibicc `_start` expects (minus the browser's graphical caps), so a debugged C
/// program that `printf`s (→ a `write` cap) runs instead of `CapFault`ing; its output lands in
/// `host.stdout`. The browser's `grant_onramp_caps` is the twin used for the non-debug Run path.
fn grant_io_powerbox(host: &mut Host, m: &Module, stdin: &[u8]) {
    host.stdin = stdin.to_vec();
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    let handles = [
        host.grant_stream(StreamRole::Out),
        host.grant_stream(StreamRole::In),
        host.grant_exit(),
        host.grant_memory(),
        host.grant_address_space(0, win),
    ];
    for (name, h) in ["stdout", "stdin", "exit", "memory", "addrspace"]
        .iter()
        .zip(&handles)
    {
        host.register_cap_name(name, *h);
    }
    if !m.imports.is_empty() {
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                let Some((type_id, op)) = io_cap(&im.name) else {
                    return BoundImport::rebindable(0, 0, None);
                };
                let handle = match (type_id, op) {
                    (cap_id::STREAM, 1) => handles[0],
                    (cap_id::STREAM, _) => handles[1],
                    (cap_id::EXIT, _) => handles[2],
                    (cap_id::MEMORY, _) => handles[3],
                    (cap_id::ADDRESS_SPACE, _) => handles[4],
                    _ => return BoundImport::rebindable(0, 0, None),
                };
                BoundImport::required(type_id, op, handle)
            })
            .collect();
        host.set_import_bindings(bindings);
    }
}

/// Build a single-vCPU [`DebugRun`] for `module`'s `func(args)`: under the on-ramp I/O powerbox when
/// `powerbox` (recording cap inputs, and replaying `tape` from a prior forward run so a reverse-`seek`
/// rebuild re-executes with identical clock/stdin inputs), else deny-all (`DebugRun::new`). `None` if the
/// module is outside the engine's subset.
#[allow(clippy::too_many_arguments)]
fn build_single_run(
    module: &Module,
    func: FuncIdx,
    args: &[Value],
    powerbox: bool,
    stdin: &[u8],
    block_stdin: bool,
    mem_limit: Option<u64>,
    tape: &CapTape,
) -> Option<DebugRun> {
    if !powerbox {
        return DebugRun::new(module, func, args);
    }
    let mut host = Host::new();
    grant_io_powerbox(&mut host, module, stdin);
    // Slice 5: the Memory-capability growth cap — a `vm_map` past the limit returns -ENOMEM, so a
    // guest malloc observes NULL (the OOM-teaching knob). Re-armed on every seek rebuild.
    host.set_mem_map_limit(mem_limit);
    // W4 blocking stdin: a `read` on an exhausted buffer parks the run (`StopReason::StdinPark`)
    // instead of returning EOF; `provideStdin` appends bytes and a resume re-issues the read.
    // Re-armed on every `seek` rebuild so a read past the replay frontier parks again.
    if block_stdin {
        host.set_stdin_blocking(true);
    }
    host.record_caps(); // tape this run's cap inputs so a later reverse seek can replay them
    if !tape.records.is_empty() {
        host.replay_cap_tape(tape.clone()); // serve the furthest-forward inputs on a rebuild
    }
    DebugRun::new_with_host(module, func, args, host)
}

/// Build a multi-vCPU [`ScheduledDebugRun`] for `module`'s `func(args)`: the scheduled-engine twin of
/// [`build_single_run`]. Under the on-ramp I/O powerbox when `powerbox` (so a threaded C guest's
/// `malloc`/`printf` reach the `memory`/`write` caps instead of `CapFault`ing, and `main`'s return
/// becomes an `exit` code), else deny-all. The `seed` is the slice-7 schedule variation. `None` if the
/// module is outside the scheduled engine's subset. Blocking stdin is single-vCPU only, so a threaded
/// session never sets it — the caller declines a `blockStdin` threaded launch upstream.
#[allow(clippy::too_many_arguments)]
fn build_scheduled_run(
    module: &Module,
    func: FuncIdx,
    args: &[Value],
    powerbox: bool,
    stdin: &[u8],
    mem_limit: Option<u64>,
    seed: Option<u64>,
    tape: &CapTape,
) -> Option<ScheduledDebugRun> {
    let mut run = if powerbox {
        let mut host = Host::new();
        grant_io_powerbox(&mut host, module, stdin);
        host.set_mem_map_limit(mem_limit);
        host.record_caps();
        if !tape.records.is_empty() {
            host.replay_cap_tape(tape.clone());
        }
        ScheduledDebugRun::new_with_host(module, func, args, host)?
    } else {
        ScheduledDebugRun::new(module, func, args)?
    };
    run.set_sched_seed(seed);
    Some(run)
}

/// The ~20 `Inspector` operations `DapServer` drives, abstracted so a bytecode-backed session can
/// serve the same requests. Methods a backend can't honor (reverse/watch) are gated by
/// [`supports_reverse`](Debuggee::supports_reverse) / [`supports_watch`](Debuggee::supports_watch);
/// the server checks the gate before calling, so their bodies are dormant on such a backend.
pub trait Debuggee {
    // --- execution -------------------------------------------------------------------------------
    fn run_until_stop(&mut self) -> Stop;
    fn step(&mut self) -> Stop;
    fn step_over(&mut self) -> Stop;
    fn step_out(&mut self) -> Stop;
    fn step_back(&mut self) -> Stop;
    fn seek(&mut self, t: u64) -> Stop;

    // --- breakpoints / watchpoints ---------------------------------------------------------------
    fn set_breakpoint(&mut self, pc: IrPc);
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool;
    /// `None` if the backend has no watchpoints (bytecode) — the server reports the data breakpoint
    /// unverified.
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId>;
    fn clear_watchpoint(&mut self, id: WatchId) -> bool;

    // --- inspection ------------------------------------------------------------------------------
    fn backtrace(&self) -> Vec<FrameInfo>;
    fn func_name(&self, func: FuncIdx) -> Option<&str>;
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc>;
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue>;
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64>;
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap>;

    // --- threads / time coordinate ---------------------------------------------------------------
    fn threads(&self) -> Vec<u64>;
    fn select_task(&mut self, id: u64) -> bool;
    fn stopped_task(&self) -> Option<u64>;
    fn turn(&self) -> u64;
    fn clock(&self) -> u64;

    // --- capability gates ------------------------------------------------------------------------
    /// Reverse debugging (`stepBack` / `reverseContinue`). Default `true` (the tree-walker).
    fn supports_reverse(&self) -> bool {
        true
    }
    /// Data breakpoints (`setDataBreakpoints` watchpoints). Default `true` (the tree-walker).
    fn supports_watch(&self) -> bool {
        true
    }

    // --- blocking stdin (W4) ---------------------------------------------------------------------
    /// Append stdin bytes for a session parked at a blocking `read` (`StopReason::StdinPark`) —
    /// the next resume re-issues the read against them. `false` when this backend/session has no
    /// blocking stdin (the `provideStdin` request fails cleanly). Default: unsupported.
    fn provide_stdin(&mut self, _bytes: &[u8]) -> bool {
        false
    }

    // --- memory map (slice 5) --------------------------------------------------------------------
    /// The window's memory-map introspection as JSON (geometry, data segments, explicit-state
    /// pages, powerbox stack/heap regions). `None` when this backend doesn't expose it (the
    /// tree-walker) or the module has no memory — the `memoryMap` request fails cleanly.
    fn memory_map(&self) -> Option<Json> {
        None
    }

    // --- scheduler trace (slice 6) ---------------------------------------------------------------
    /// Arm the scheduler trace tape ([`bytecode::SchedTraceEvent`]) — turns, parks, wakes with
    /// both identities, spawns. `false` when this backend/session has no schedule to trace (the
    /// tree-walker, a single-vCPU session): a `schedTrace` launch fails cleanly. Default:
    /// unsupported.
    fn set_sched_trace(&mut self, _on: bool) -> bool {
        false
    }
    /// The trace tape so far as a JSON array (`None` when unarmed/unsupported).
    fn sched_trace_json(&self) -> Option<Json> {
        None
    }
    // --- state writes (slice 8) ------------------------------------------------------------------
    /// **Write bytes into the guest window** (the DAP `writeMemory` backend). Recorded and
    /// re-applied on every seek replay so time travel stays truthful. `false` when unsupported
    /// (tree-walker) or the range is unwritable.
    fn write_window(&mut self, _addr: u64, _bytes: &[u8]) -> bool {
        false
    }
    /// **Write a source variable by name** in the frame `frame_from_top` (the DAP `setVariable`
    /// backend): integers only; a memory-located var takes the low `width` bytes. Recorded and
    /// re-applied like [`write_window`](Debuggee::write_window). `false` when unsupported or
    /// unresolvable — fail-closed, never a guess.
    fn write_var(
        &mut self,
        _frame_from_top: usize,
        _name: &str,
        _value: i64,
        _width: usize,
    ) -> bool {
        false
    }

    /// **Force a context switch** (slice 7): override the schedule's next pick with `target` (a
    /// task index; `None` = the lowest-index runnable task other than the default choice). The
    /// override is recorded as a concrete `(turn, task)` and re-applied on every rebuild, so a
    /// `seek` replays it at the identical turn. Returns the resolved task, `None` when
    /// unsupported (tree-walker, single-vCPU) or nothing is runnable.
    fn force_switch(&mut self, _target: Option<usize>) -> Option<usize> {
        None
    }

    // --- access sink / models --------------------------------------------------------------------
    /// Install the session's access-sink consumer (INTERACTIVE_EMBEDDING.md slice 3): every
    /// module-0 memory op reaches it, `seek` replays included. `false` when this backend has no
    /// sink (the tree-walker) — a `memModel` launch fails cleanly instead of silently observing
    /// nothing. Default: unsupported.
    fn set_access_sink(&mut self, _sink: SharedSink) -> bool {
        false
    }

    // --- powerbox output -------------------------------------------------------------------------
    /// The guest's captured stdout at the current stop, if this session runs under a powerbox (else
    /// empty). The server surfaces it as DAP `output` events; on a reverse `seek` it reflects exactly
    /// the output produced up to *here* (the run is rebuilt + replayed), so it rewinds with the program.
    fn stdout(&self) -> &[u8] {
        &[]
    }
}

/// The tree-walker backend — the original, full-featured engine. Every method delegates to the
/// inherent `Inspector` method (an inherent method shadows the trait one, so no recursion).
impl Debuggee for Inspector {
    fn run_until_stop(&mut self) -> Stop {
        Inspector::run_until_stop(self)
    }
    fn step(&mut self) -> Stop {
        Inspector::step(self)
    }
    fn step_over(&mut self) -> Stop {
        Inspector::step_over(self)
    }
    fn step_out(&mut self) -> Stop {
        Inspector::step_out(self)
    }
    fn step_back(&mut self) -> Stop {
        Inspector::step_back(self)
    }
    fn seek(&mut self, t: u64) -> Stop {
        Inspector::seek(self, t)
    }
    fn set_breakpoint(&mut self, pc: IrPc) {
        Inspector::set_breakpoint(self, pc)
    }
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool {
        Inspector::clear_breakpoint(self, pc)
    }
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId> {
        Some(Inspector::set_watchpoint(self, addr, len, kind))
    }
    fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        Inspector::clear_watchpoint(self, id)
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        Inspector::backtrace(self)
    }
    fn func_name(&self, func: FuncIdx) -> Option<&str> {
        Inspector::func_name(self, func)
    }
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        Inspector::source_loc(self, pc)
    }
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        Inspector::read_var(self, frame_from_top, name, width)
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        Inspector::var_addr(self, frame_from_top, name)
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        Inspector::read_window(self, addr, len)
    }
    fn threads(&self) -> Vec<u64> {
        Inspector::threads(self)
    }
    fn select_task(&mut self, id: u64) -> bool {
        Inspector::select_task(self, id)
    }
    fn stopped_task(&self) -> Option<u64> {
        Inspector::stopped_task(self)
    }
    fn turn(&self) -> u64 {
        Inspector::turn(self)
    }
    fn clock(&self) -> u64 {
        Inspector::clock(self)
    }
}

/// The bytecode engine behind a [`BytecodeBackend`]: the single-vCPU [`DebugRun`] for a spawn-free
/// guest, or the multi-vCPU [`ScheduledDebugRun`] (a cooperative debug scheduler with per-thread
/// breakpoints, `select_task`, in/over/out stepping, cross-thread watchpoints, and reverse debugging)
/// for a `thread.spawn` guest. Chosen at launch by [`bytecode::module_spawns_threads`]; both are fully
/// forward + reverse + watch capable.
enum Engine {
    // Both `DebugRun` and `ScheduledDebugRun` are large (their `VTask`s carry the reified continuation),
    // so box each variant to keep `Engine` — and the `BytecodeBackend` holding it — pointer-sized
    // (`clippy::large_enum_variant`).
    Single(Box<DebugRun>),
    Threaded(Box<ScheduledDebugRun>),
}

/// A cached map of the run's **stoppable positions** — `(clock, depth)` for every op that sits at a
/// real IR instruction — over `[0, high_water]`. The op timeline of a deterministic replay is fixed
/// for the whole session (breakpoints never change *which* ops run, and cap-input `tape` records are
/// append-only + positional, so an earlier op's behavior can't change as the run goes further
/// forward), so this is built once by a single fresh-run scan and reused by every `step_back` target
/// search — replacing the per-`step_back` probe scan that re-derived it. Rebuilt only when a forward
/// step advances the current position past `high_water`.
struct RevTrace {
    /// The furthest clock/turn the scan covered; the cache is valid for any position `<= high_water`.
    high_water: u64,
    /// `(clock, depth)` of each stoppable op in `[0, high_water)`, ascending — a `step_back` target is
    /// the last entry strictly before the current position at call depth `<=` the current frame count.
    stoppable: Vec<(u64, usize)>,
}

/// The **bytecode backend** — the resumable bytecode debug session ([`Engine`]) plus the persistent
/// breakpoint set `DapServer` expects, the module (for `source_loc`/`func_name`, which are
/// engine-neutral free functions keyed on the `IrPc`), and the launch `func`/`args` so reverse
/// debugging can rebuild a fresh run and replay to an earlier op clock.
pub struct BytecodeBackend {
    engine: Engine,
    module: Module,
    func: FuncIdx,
    args: Vec<Value>,
    breakpoints: Vec<IrPc>,
    /// Armed watchpoints with backend-owned stable ids (re-applied to the run after a `seek` rebuild).
    /// Single-vCPU only — the scheduled engine reports `supports_watch = false` this slice.
    watch_specs: Vec<(WatchId, u64, u64, WatchKind)>,
    next_watch: u32,
    fuel: u64,
    /// This session runs its guest under the **on-ramp I/O powerbox** ([`grant_io_powerbox`]) instead of
    /// deny-all — so a manifest program that calls a host capability (a chibicc `printf` → `write`) runs
    /// and its output is captured, rather than `CapFault`ing. Single-vCPU only (a `thread.spawn` guest
    /// stays deny-all this slice). Off ⇒ the compute-only path, unchanged.
    powerbox: bool,
    /// Preloaded stdin for the powerbox (`read(0, …)`); empty for a pure-output program.
    stdin: Vec<u8>,
    /// W4 blocking stdin: a `read` on an exhausted buffer parks the session
    /// (`StopReason::StdinPark`, resumed by `provideStdin`) instead of returning EOF. Single-vCPU
    /// powerbox sessions only — `new` declines a threaded module with this set (fail-closed).
    block_stdin: bool,
    /// Slice 5: the session's Memory-capability growth cap ([`Host::set_mem_map_limit`]) — set on
    /// the powerbox at build and on every seek rebuild. `None` = unbounded.
    mem_limit: Option<u64>,
    /// Slice 6: whether the scheduler trace tape is armed (threaded engine only) — re-armed on
    /// every seek rebuild so the replay refills the tape deterministically.
    sched_trace: bool,
    /// Slice 7: the seeded-pick schedule policy (threaded engine only) — applied at construction
    /// and re-applied on every rebuild (seek and rev-trace probes: the seed is *semantic* schedule
    /// policy, unlike the observation-only sink/trace, so every replay must carry it).
    seed: Option<u64>,
    /// Slice 7: the recorded forced switches, concrete `(turn, task)` — re-applied on every
    /// rebuild for the same reason.
    forced: Vec<(u64, usize)>,
    /// Slice 8: recorded **debugger state writes** ([`ScheduledWrite`]), keyed by the clock/turn
    /// they were made at. The engine re-applies each whenever execution passes its clock — on the
    /// live resume *and* on every seek replay / rev-trace probe (the list is re-installed on each
    /// rebuild) — so time travel stays truthful: `seek` back before a write shows the original
    /// state, any path forward past it re-observes the write. The write-side `CapTape`.
    writes: Vec<(u64, ScheduledWrite)>,
    /// The recorded [`CapTape`] of nondeterministic cap **inputs** (clock / stdin `read` / host-fn) from
    /// the furthest-forward execution — replayed on a reverse `seek` rebuild so re-execution sees
    /// identical inputs (DEBUGGING.md W1). A pure-output program (`write` only) records nothing, so its
    /// reverse replay is deterministic by re-execution alone; the tape covers the input-reading case.
    tape: CapTape,
    /// Cached stoppable-position timeline for `step_back` target search (see [`RevTrace`]). `None` until
    /// the first `step_back`; rebuilt when a forward step moves the position past its `high_water`.
    rev_trace: Option<RevTrace>,
    /// Time-travel **checkpoint ladder** (DEBUGGING.md W1), single-vCPU only: snapshots of the run at
    /// ascending op clocks (kept sorted) so a reverse `seek`/`step_back` restarts from the nearest one
    /// (`clock <= t`) instead of clock 0, bounding the replay to [`CHECKPOINT_STRIDE`]. Populated lazily
    /// as `seek` drives past stride boundaries — the bytecode port of the tree-walker `Inspector`'s
    /// ladder.
    checkpoints: Vec<DebugRunSnapshot>,
    /// Multi-vCPU checkpoint ladder (same role as `checkpoints`, keyed on the global scheduler `turn`)
    /// for a threaded session's `ScheduledDebugRun`. Only one of the two ladders is ever populated —
    /// a session is single-vCPU xor threaded for its whole life.
    sched_checkpoints: Vec<ScheduledSnapshot>,
    /// Whether checkpointing is still active. Cleared (and the ladder dropped) the first time a stride
    /// boundary falls outside the [`DebugRun::snapshot`] / [`ScheduledDebugRun::snapshot`] subset (a
    /// fiber/coroutine/§14-child seam, a non-pristine memory layout, or a host that grew unrestorable
    /// state), after which `seek` reverts to replay-from-0 for the rest of the session — mirroring
    /// `Inspector::maybe_checkpoint`.
    checkpointing: bool,
    /// The session's shared access-sink consumer (INTERACTIVE_EMBEDDING.md slice 3), re-installed
    /// into the live engine on every `seek` rebuild (the `watch_specs` pattern) — so a model fed by
    /// it observes the replay too and can re-derive its state (`seek(t)` ≡ a from-0 run to `t`).
    /// The rev-trace probes stay silent (they build raw runs, no sink). `None` = no consumer.
    access_sink: Option<SharedSink>,
}

/// A shared, re-installable access-sink consumer: `(clock-or-turn, task, event)`. `Arc<Mutex<…>>`
/// so the backend can hand a fresh boxed wrapper to every rebuilt run while one consumer (a
/// host-side model) accumulates.
pub type SharedSink = std::sync::Arc<std::sync::Mutex<dyn FnMut(u64, usize, MemEvent) + Send>>;

/// Wrap the shared consumer as the engine's boxed sink ([`AccessSinkFn`]).
fn wrap_sink(sink: &SharedSink) -> AccessSinkFn {
    let s = std::sync::Arc::clone(sink);
    Box::new(move |clock, task, ev| {
        let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
        (*g)(clock, task, ev)
    })
}

/// The op-clock stride between time-travel checkpoints (DEBUGGING.md W1). Matches the tree-walker
/// `Inspector`'s `SEEK_CHECKPOINT_STRIDE`, so a reverse `seek`/`step_back` replays at most this many
/// ops past the nearest snapshot instead of O(t) from clock 0. `pub(crate)`: the host-side models
/// (`models.rs`) snapshot their own state at the same boundaries, so any clock the engine can
/// restore to has a matching model snapshot (their seek-consistency hinges on the strides agreeing).
pub(crate) const CHECKPOINT_STRIDE: u64 = 1024;

impl BytecodeBackend {
    /// Open a bytecode debug session on `module`'s `func(args)`. A `thread.spawn` guest gets the
    /// multithreaded scheduled engine; a spawn-free one the single-vCPU engine. `None` if the module
    /// is outside the bytecode engine's subset (`compile_module` declines it). `powerbox` runs a
    /// spawn-free guest under the on-ramp I/O powerbox (`stdin` preloads `read`); a threaded guest ignores
    /// it (deny-all).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        module: Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        powerbox: bool,
        stdin: Vec<u8>,
        block_stdin: bool,
        mem_limit: Option<u64>,
        seed: Option<u64>,
    ) -> Option<BytecodeBackend> {
        let tape = CapTape::default();
        let engine = if bytecode::module_spawns_threads(&module) {
            // Blocking stdin is single-vCPU only this slice — decline a threaded module rather
            // than silently keeping EOF semantics (fail-closed).
            if block_stdin {
                return None;
            }
            // The powerbox rides the scheduled engine too now, so a threaded C guest's
            // `malloc`/`printf` work under the debugger (the seed is the slice-7 variation).
            let run = build_scheduled_run(
                &module, func, args, powerbox, &stdin, mem_limit, seed, &tape,
            )?;
            Engine::Threaded(Box::new(run))
        } else {
            // A seed is meaningless with one vCPU — decline rather than silently ignore it.
            if seed.is_some() {
                return None;
            }
            Engine::Single(Box::new(build_single_run(
                &module,
                func,
                args,
                powerbox,
                &stdin,
                block_stdin,
                mem_limit,
                &tape,
            )?))
        };
        Some(BytecodeBackend {
            engine,
            module,
            func,
            args: args.to_vec(),
            breakpoints: Vec::new(),
            watch_specs: Vec::new(),
            next_watch: 0,
            fuel,
            powerbox,
            stdin,
            block_stdin,
            mem_limit,
            tape,
            rev_trace: None,
            checkpoints: Vec::new(),
            sched_checkpoints: Vec::new(),
            checkpointing: true,
            access_sink: None,
            sched_trace: false,
            seed,
            forced: Vec::new(),
            writes: Vec::new(),
        })
    }

    /// Install the session's **access-sink consumer** (INTERACTIVE_EMBEDDING.md slice 3): every
    /// module-0 memory op the session executes — including `seek`-replay re-execution — reaches
    /// `sink` as `(clock-or-turn, task, MemEvent)`. Re-installed transparently across `seek`
    /// rebuilds; the rev-trace probes never fire it.
    pub fn set_access_sink(&mut self, sink: SharedSink) {
        match &mut self.engine {
            Engine::Single(run) => run.set_access_sink(wrap_sink(&sink)),
            Engine::Threaded(run) => run.set_access_sink(wrap_sink(&sink)),
        }
        self.access_sink = Some(sink);
    }

    /// Number of time-travel checkpoints currently in the ladder (single-vCPU or scheduled — only one
    /// is ever populated) — test/introspection hook (mirrors `Inspector::checkpoint_count`). `0` for a
    /// run not yet seeked far enough to lay one down, or one outside the checkpointable subset (fibers,
    /// coroutines, §14 children, stateful host), which replays from clock 0 / turn 0.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len() + self.sched_checkpoints.len()
    }

    /// Build a fresh single-vCPU [`DebugRun`] with this session's engine config — under the on-ramp I/O
    /// powerbox (recording + replaying `tape`) when `self.powerbox`, else deny-all. Used at launch and on
    /// every reverse `seek`/`step_back` rebuild, so a re-executed powerbox run sees identical cap inputs.
    fn fresh_single(&self) -> Option<DebugRun> {
        build_single_run(
            &self.module,
            self.func,
            &self.args,
            self.powerbox,
            &self.stdin,
            self.block_stdin,
            self.mem_limit,
            &self.tape,
        )
    }

    /// Rebuild a fresh scheduled run with this session's powerbox + seed (the threaded twin of
    /// [`fresh_single`]) — the base a reverse `seek`/rev-trace probe re-drives from.
    fn fresh_scheduled(&self) -> Option<ScheduledDebugRun> {
        build_scheduled_run(
            &self.module,
            self.func,
            &self.args,
            self.powerbox,
            &self.stdin,
            self.mem_limit,
            self.seed,
            &self.tape,
        )
    }

    /// Drive a freshly rebuilt (and possibly checkpoint-restored) single-vCPU `run` forward to op clock
    /// `t`, laying down a checkpoint at each [`CHECKPOINT_STRIDE`] boundary along the way (so a later
    /// reverse `seek`/`step_back` restarts nearby). With checkpointing off it is a straight tick-to-`t`.
    /// The stride capture is transparent — ticking is the same raw replay quantum regardless — so the
    /// state at `t` is identical to a single run from the restore point to `t`.
    fn drive_single_to(&mut self, run: &mut DebugRun, t: u64, fuel: &mut u64) {
        loop {
            let clock = run.op_clock();
            if clock >= t {
                break;
            }
            // At a positive stride boundary short of `t`, snapshot before executing the op (so the
            // checkpoint's clock is exactly the boundary). Deduped + subset-guarded by `maybe_checkpoint`.
            if self.checkpointing && clock > 0 && clock.is_multiple_of(CHECKPOINT_STRIDE) {
                self.maybe_checkpoint(run);
            }
            if !run.tick(fuel) {
                break;
            }
        }
    }

    /// Snapshot `run` into the checkpoint ladder at its current clock, if checkpointing is still on and
    /// the continuation is [`DebugRun::snapshot`]-able; otherwise disable checkpointing and drop the
    /// ladder (the run left the snapshottable subset, so `seek` reverts to replay-from-0). A clock
    /// already in the ladder is not duplicated. Mirrors `Inspector::maybe_checkpoint`.
    fn maybe_checkpoint(&mut self, run: &DebugRun) {
        if !self.checkpointing {
            return;
        }
        let Some(snap) = run.snapshot() else {
            self.checkpointing = false;
            self.checkpoints.clear();
            return;
        };
        let clock = snap.clock();
        if self.checkpoints.iter().any(|c| c.clock() == clock) {
            return;
        }
        // Keep the ladder sorted by clock (boundaries usually append in order, but a fresh replay-from-0
        // can fill a gap below an existing entry).
        let at = self.checkpoints.partition_point(|c| c.clock() < clock);
        self.checkpoints.insert(at, snap);
    }

    /// Scheduled-engine counterpart of [`drive_single_to`](BytecodeBackend::drive_single_to): drive a
    /// freshly rebuilt (and possibly checkpoint-restored) `ScheduledDebugRun` forward to global turn `t`,
    /// laying down a checkpoint at each [`CHECKPOINT_STRIDE`] boundary. With checkpointing off it is a
    /// straight tick-to-`t`. Ticking is the same raw quantum regardless, so the state at `t` equals a
    /// single run from the restore point to `t`.
    fn drive_scheduled_to(&mut self, run: &mut ScheduledDebugRun, t: u64, fuel: &mut u64) {
        loop {
            let turn = run.op_turn();
            if turn >= t {
                break;
            }
            if self.checkpointing && turn > 0 && turn.is_multiple_of(CHECKPOINT_STRIDE) {
                self.maybe_sched_checkpoint(run);
            }
            if !run.tick(fuel) {
                break;
            }
        }
    }

    /// Scheduled-engine counterpart of [`maybe_checkpoint`](BytecodeBackend::maybe_checkpoint): snapshot
    /// `run` into the scheduled ladder at its current turn, or disable checkpointing + drop the ladder if
    /// the continuation left the [`ScheduledDebugRun::snapshot`]-able subset.
    fn maybe_sched_checkpoint(&mut self, run: &ScheduledDebugRun) {
        if !self.checkpointing {
            return;
        }
        let Some(snap) = run.snapshot() else {
            self.checkpointing = false;
            self.sched_checkpoints.clear();
            return;
        };
        let turn = snap.turn();
        if self.sched_checkpoints.iter().any(|c| c.turn() == turn) {
            return;
        }
        let at = self.sched_checkpoints.partition_point(|c| c.turn() < turn);
        self.sched_checkpoints.insert(at, snap);
    }

    /// Push the recorded writes into the live engine (slice 8) — its cursor lands past entries at
    /// clocks already passed, so the just-applied live write isn't double-counted going forward.
    fn sync_writes(&mut self) {
        match &mut self.engine {
            Engine::Single(run) => run.set_scheduled_writes(self.writes.clone()),
            Engine::Threaded(run) => run.set_scheduled_writes(self.writes.clone()),
        }
    }

    /// Absorb the live single-vCPU run's recorded cap-input tape if it now reaches further than the one
    /// held — so a later reverse `seek` replays the furthest-forward inputs. Cheap no-op for a
    /// pure-output (`write`-only) program (its tape stays empty) and for deny-all sessions.
    fn capture_tape(&mut self) {
        if !self.powerbox {
            return;
        }
        let live = match &self.engine {
            Engine::Single(run) => run.host().cap_tape(),
            Engine::Threaded(run) => run.host().cap_tape(),
        };
        if live.records.len() > self.tape.records.len() {
            self.tape = live;
        }
    }

    /// Whether this session runs on the multithreaded scheduled engine (so the DAP server uses the
    /// global `turn` as the reverse time coordinate, not the single-vCPU op `clock`).
    pub fn is_threaded(&self) -> bool {
        matches!(self.engine, Engine::Threaded(_))
    }

    /// Push the current watchpoint ranges into the live engine (after arming/clearing one, or re-arming
    /// a fresh single-vCPU run built by `seek`). Cross-thread on the scheduled engine.
    fn apply_watches(&mut self) {
        let ranges: Vec<_> = self
            .watch_specs
            .iter()
            .map(|(_, a, l, k)| (*a, *l, *k))
            .collect();
        match &mut self.engine {
            Engine::Single(run) => run.set_watchpoints(ranges),
            Engine::Threaded(run) => run.set_watchpoints(ranges),
        }
    }

    /// Ensure [`Self::rev_trace`] covers `[0, now]`, (re)building it with a single fresh-run scan when
    /// absent or stale (a forward step moved the position past the cached `high_water`). This is the one
    /// replay `step_back` used to pay *in addition to* the `seek`; caching it means a `step_back` at or
    /// below a position already scanned pays only the `seek`. The op timeline is deterministic, so a
    /// cached entry stays valid across later forward progress and `tape` growth (see [`RevTrace`]).
    /// Returns `false` if the run couldn't be rebuilt (module outside the engine's subset) — the caller
    /// reports `Stop::Blocked`, matching the pre-cache behavior.
    fn ensure_rev_trace(&mut self, now: u64) -> bool {
        if let Some(t) = &self.rev_trace {
            if now <= t.high_water {
                return true;
            }
        }
        let mut fuel = self.fuel;
        let mut stoppable = Vec::new();
        match &self.engine {
            Engine::Single(_) => {
                let Some(mut probe) = self.fresh_single() else {
                    return false;
                };
                // Slice 8: the probe replays the debugger writes too, or its timeline diverges.
                probe.set_scheduled_writes(self.writes.clone());
                loop {
                    let c = probe.op_clock();
                    if c >= now {
                        break;
                    }
                    if probe.frame_pc(0).is_some() {
                        stoppable.push((c, probe.depth()));
                    }
                    if !probe.tick(&mut fuel) {
                        break;
                    }
                }
            }
            Engine::Threaded(_) => {
                let Some(mut probe) = self.fresh_scheduled() else {
                    return false;
                };
                // The probe must replay the *same schedule* as the session (slice 7 policy; the
                // seed rides `fresh_scheduled`) and the same debugger writes (slice 8), or its
                // timeline diverges.
                probe.set_forced_switches(self.forced.clone());
                probe.set_scheduled_writes(self.writes.clone());
                loop {
                    let c = probe.op_turn();
                    if c >= now {
                        break;
                    }
                    probe.locate();
                    if probe.frame_pc(0).is_some() {
                        stoppable.push((c, probe.depth()));
                    }
                    if !probe.tick(&mut fuel) {
                        break;
                    }
                }
            }
        }
        self.rev_trace = Some(RevTrace {
            high_water: now,
            stoppable,
        });
        true
    }

    /// Map an engine completion (a resume/step returned no pc) to a `Stop`: a blocking-stdin park
    /// first (the run is live, paused at the read, resumable once `provideStdin` supplies bytes),
    /// then the finished result (or trap) if the root is done, else `Blocked` (a concurrency seam
    /// that engine can't follow).
    fn finish_stop(&self) -> Stop {
        if let Engine::Single(run) = &self.engine {
            if run.stdin_parked() {
                if let Some(pc) = run.frame_pc(0) {
                    return Stop::Break {
                        reason: StopReason::StdinPark,
                        pc,
                    };
                }
            }
        }
        let result = match &self.engine {
            Engine::Single(run) => run.result().cloned(),
            Engine::Threaded(run) => run.result().cloned(),
        };
        match result {
            Some(r) => Stop::Finished(r),
            None => Stop::Blocked,
        }
    }

    /// A `Step` stop at the focused thread's current pc (or the finished result), for a resume/seek that
    /// didn't hit a breakpoint.
    fn step_stop(&self) -> Stop {
        let pc = match &self.engine {
            Engine::Single(run) => run.frame_pc(0),
            Engine::Threaded(run) => run.frame_pc(0),
        };
        match pc {
            Some(pc) => Stop::Break {
                reason: StopReason::Step,
                pc,
            },
            None => self.finish_stop(),
        }
    }

    /// Map a multithreaded [`SchedStop`] to the DAP [`Stop`] — the `SchedBreak` reason carries whether
    /// it was a breakpoint, a data breakpoint (with the confined address + read/write), or a step.
    fn sched_stop(s: SchedStop) -> Stop {
        match s {
            SchedStop::Break { pc, reason } => {
                let reason = match reason {
                    SchedBreak::Breakpoint => StopReason::Breakpoint,
                    SchedBreak::Watchpoint { addr, write } => {
                        StopReason::Watchpoint { addr, write }
                    }
                    SchedBreak::Step => StopReason::Step,
                };
                Stop::Break { reason, pc }
            }
            SchedStop::Finished(r) => Stop::Finished(r),
            // No runnable thread (deadlock/`wait`), or an op outside the scheduler's subset.
            SchedStop::Blocked | SchedStop::Declined => Stop::Blocked,
        }
    }
}

impl Debuggee for BytecodeBackend {
    fn run_until_stop(&mut self) -> Stop {
        // A fresh fuel budget per resume (debugging is interactive; the run replays from scratch on a
        // seek, so a shared decrementing counter would be inconsistent).
        let mut fuel = self.fuel;
        let stop = match &mut self.engine {
            Engine::Single(run) => match run.run_to(&self.breakpoints, &mut fuel) {
                // A stop is a watchpoint hit if the run flagged one before this op, else a breakpoint.
                Some(pc) => {
                    let reason = match run.take_watch_hit() {
                        Some((addr, write)) => StopReason::Watchpoint { addr, write },
                        None => StopReason::Breakpoint,
                    };
                    Stop::Break { reason, pc }
                }
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => {
                run.set_breakpoints(self.breakpoints.clone());
                Self::sched_stop(run.run_until_stop(&mut fuel))
            }
        };
        self.capture_tape(); // absorb any new cap inputs this advance recorded (for a later reverse seek)
        stop
    }
    fn step(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = match &mut self.engine {
            Engine::Single(run) => match run.step(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step(&mut fuel)),
        };
        self.capture_tape();
        stop
    }
    fn step_over(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = match &mut self.engine {
            Engine::Single(run) => match run.step_over(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step_over(&mut fuel)),
        };
        self.capture_tape();
        stop
    }
    fn step_out(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = match &mut self.engine {
            Engine::Single(run) => match run.step_out(&mut fuel) {
                Some(pc) => Stop::Break {
                    reason: StopReason::Step,
                    pc,
                },
                None => self.finish_stop(),
            },
            Engine::Threaded(run) => Self::sched_stop(run.step_out(&mut fuel)),
        };
        self.capture_tape();
        stop
    }
    // Reverse debugging by **deterministic replay** (DEBUGGING.md W1): the debug run is pure compute
    // (single-vCPU, no capabilities), so seeking to an earlier op clock = rebuild a fresh run and
    // replay to that many ops. `step_back` = one op earlier. The `seek` replay is bounded by the
    // single-vCPU **checkpoint ladder** (see `drive_single_to`/`maybe_checkpoint`): a restart from the
    // nearest snapshot replays at most `CHECKPOINT_STRIDE` ops instead of O(t) from clock 0.
    fn step_back(&mut self) -> Stop {
        // Rewind to the previous op that sits at a real IR instruction (a stoppable position — not a
        // terminator slot, where there's nothing to inspect) strictly before now, then seek there. The
        // single-vCPU coordinate is the op `clock`; the multithreaded one is the global scheduler `turn`.
        //
        // **Depth-aware** (the reverse of `next`, not `stepIn`): only ops at call depth ≤ the current
        // frame count are candidates, so a step-back from a line that *called* something (a chibicc
        // `printf` → the guest libc) rewinds to the previous op **in the caller's frame**, not down into
        // the callee's last op. Without this, stepping back from a `printf` descends into the libc
        // internals (`__pf_flush`, …) instead of the previous source line — the reverse of stepping over.
        //
        // The candidate positions come from the cached [`RevTrace`] (the run's fixed stoppable-op
        // timeline), so the target search is a lookup rather than a second full replay — only the `seek`
        // below re-executes. The first `step_back` past a new high-water builds the trace with one scan.
        let (now, now_depth) = match &self.engine {
            Engine::Single(run) => (run.op_clock(), run.depth()),
            Engine::Threaded(run) => (run.op_turn(), run.depth()),
        };
        if !self.ensure_rev_trace(now) {
            return Stop::Blocked;
        }
        let target = self
            .rev_trace
            .as_ref()
            .expect("ensure_rev_trace populated the cache")
            .stoppable
            .iter()
            .rev()
            .find(|(c, d)| *c < now && *d <= now_depth)
            .map_or(0, |(c, _)| *c);
        self.seek(target)
    }
    fn seek(&mut self, t: u64) -> Stop {
        let mut fuel = self.fuel;
        if self.is_threaded() {
            // Rebuild a fresh scheduled run and replay `t` turns — the schedule is deterministic, so
            // this reproduces the exact state at global turn `t` (DEBUGGING.md W1, multithreaded).
            let Some(mut run) = self.fresh_scheduled() else {
                return Stop::Blocked;
            };
            run.set_breakpoints(self.breakpoints.clone());
            run.set_watchpoints(
                self.watch_specs
                    .iter()
                    .map(|(_, a, l, k)| (*a, *l, *k))
                    .collect(),
            );
            // Re-install the access sink *before* the replay drive, so a model consumer observes
            // the re-execution and can re-derive its state (`seek(t)` ≡ a from-0 run to `t`).
            if let Some(sink) = &self.access_sink {
                run.set_access_sink(wrap_sink(sink));
            }
            // Re-arm the trace tape: the replay refills it deterministically from the restore point.
            if self.sched_trace {
                run.set_sched_trace(true);
            }
            // Re-apply the schedule policy (slice 7) — semantic, so the replay must carry it (the
            // seed rides `fresh_scheduled`; forced switches are applied here).
            run.set_forced_switches(self.forced.clone());
            // And the recorded debugger writes (slice 8) — the replay re-applies them at their turns.
            run.set_scheduled_writes(self.writes.clone());
            // Restart from the nearest scheduled checkpoint at or before `t` (ladder kept sorted by
            // turn) instead of turn 0, when still checkpointable — bounding the replay to the stride.
            if self.checkpointing {
                if let Some(cp) = self.sched_checkpoints.iter().rev().find(|c| c.turn() <= t) {
                    run.restore(cp);
                }
            }
            self.drive_scheduled_to(&mut run, t, &mut fuel);
            run.locate();
            if let Some(pc) = run.frame_pc(0) {
                if self.breakpoints.contains(&pc) {
                    run.arm_breakpoint_skip();
                }
            }
            self.engine = Engine::Threaded(Box::new(run));
            return self.step_stop();
        }
        let Some(mut run) = self.fresh_single() else {
            return Stop::Blocked;
        };
        // Re-install the access sink *before* the replay drive (see the threaded path above).
        if let Some(sink) = &self.access_sink {
            run.set_access_sink(wrap_sink(sink));
        }
        // And the recorded debugger writes (slice 8) — the replay re-applies them at their clocks.
        run.set_scheduled_writes(self.writes.clone());
        // Restart from the nearest checkpoint at or before `t` (the ladder is kept sorted by clock)
        // instead of clock 0, when this run is still checkpointable — bounding the replay to the stride.
        if self.checkpointing {
            if let Some(cp) = self.checkpoints.iter().rev().find(|c| c.clock() <= t) {
                run.restore(cp);
            }
        }
        self.drive_single_to(&mut run, t, &mut fuel);
        self.engine = Engine::Single(Box::new(run));
        self.apply_watches(); // re-arm the watchpoints on the fresh (replayed) run
                              // If the replay landed exactly on a breakpoint op, arm the skip so a forward `continue` from
                              // here makes progress instead of immediately re-reporting this stop.
        if let Engine::Single(run) = &mut self.engine {
            if let Some(pc) = run.frame_pc(0) {
                if self.breakpoints.contains(&pc) {
                    run.arm_breakpoint_skip();
                }
            }
        }
        self.step_stop()
    }
    fn set_breakpoint(&mut self, pc: IrPc) {
        if !self.breakpoints.contains(&pc) {
            self.breakpoints.push(pc);
        }
    }
    fn clear_breakpoint(&mut self, pc: IrPc) -> bool {
        let before = self.breakpoints.len();
        self.breakpoints.retain(|&b| b != pc);
        self.breakpoints.len() != before
    }
    // Data breakpoints: arm a window watchpoint (a backend-owned stable id, so it survives a `seek`).
    // Cross-thread on the scheduled engine (fires in whichever thread touches the range).
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId> {
        let id = WatchId::from_raw(self.next_watch);
        self.next_watch += 1;
        self.watch_specs.push((id, addr, len, kind));
        self.apply_watches();
        Some(id)
    }
    fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        let before = self.watch_specs.len();
        self.watch_specs.retain(|(w, ..)| *w != id);
        let removed = self.watch_specs.len() != before;
        if removed {
            self.apply_watches();
        }
        removed
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        let mut out = Vec::new();
        // The focused thread's stack (single-vCPU: the sole thread; scheduled: `select_task`'s pick).
        let depth = match &self.engine {
            Engine::Single(run) => run.depth(),
            Engine::Threaded(run) => run.depth(),
        };
        for d in 0..depth {
            let pc = match &self.engine {
                Engine::Single(run) => run.frame_pc(d),
                Engine::Threaded(run) => run.frame_pc(d),
            };
            if let Some(pc) = pc {
                let source = svm_interp::source_loc(&self.module, pc);
                out.push(FrameInfo {
                    pc,
                    vals: Vec::new(), // unused by DapServer (it reads via read_var)
                    source,
                });
            }
        }
        out
    }
    fn func_name(&self, func: FuncIdx) -> Option<&str> {
        svm_interp::func_name(&self.module, func)
    }
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        svm_interp::source_loc(&self.module, pc)
    }
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        match &self.engine {
            Engine::Single(run) => run.read_var(frame_from_top, name, width),
            Engine::Threaded(run) => run.read_var(frame_from_top, name, width),
        }
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        match &self.engine {
            Engine::Single(run) => run.var_addr(frame_from_top, name),
            Engine::Threaded(run) => run.var_addr(frame_from_top, name),
        }
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        match &self.engine {
            Engine::Single(run) => run.read_window(addr, len),
            Engine::Threaded(run) => run.read_window(addr, len),
        }
    }
    fn threads(&self) -> Vec<u64> {
        match &self.engine {
            // Single-vCPU: the sole task is 0 (DAP thread 1).
            Engine::Single(_) => vec![0],
            Engine::Threaded(run) => run.threads(),
        }
    }
    fn select_task(&mut self, id: u64) -> bool {
        match &mut self.engine {
            Engine::Single(_) => id == 0,
            Engine::Threaded(run) => run.select_task(id),
        }
    }
    fn stopped_task(&self) -> Option<u64> {
        match &self.engine {
            Engine::Single(_) => Some(0),
            Engine::Threaded(run) => run.stopped_task(),
        }
    }
    fn turn(&self) -> u64 {
        match &self.engine {
            // Single-vCPU: no scheduler turns; the op `clock` is the time coordinate.
            Engine::Single(_) => 0,
            Engine::Threaded(run) => run.turn(),
        }
    }
    fn clock(&self) -> u64 {
        match &self.engine {
            Engine::Single(run) => run.op_clock(),
            Engine::Threaded(run) => run.turn(),
        }
    }
    // Both engines are fully reversible (deterministic replay) and watch-capable: the single-vCPU op
    // `clock` and the multithreaded global `turn` are the two time coordinates.
    fn supports_reverse(&self) -> bool {
        true
    }
    fn supports_watch(&self) -> bool {
        true
    }
    /// The memory-map JSON: window geometry + explicit-state pages from the engine
    /// ([`Mem::map_info`]), data-segment placements from the module, the powerbox stack layout
    /// constants, and — under the powerbox — the guest heap cursor (window words at
    /// `POWERBOX_HEAP_BRK`/`_TOP`; the heap base is the mapped end, §1a growth into the tail).
    fn memory_map(&self) -> Option<Json> {
        let (page, mapped, reserved, pages) = match &self.engine {
            Engine::Single(run) => run.mem_map_info(),
            Engine::Threaded(run) => run.mem_map_info(),
        }?;
        let kinds = ["ro", "rw", "unmapped", "region"];
        let mut fields = vec![
            ("pageSize", Json::i(page as i64)),
            ("mapped", Json::i(mapped as i64)),
            ("reserved", Json::i(reserved as i64)),
            (
                "segments",
                Json::Arr(
                    self.module
                        .data
                        .iter()
                        .map(|d| {
                            Json::obj(vec![
                                ("offset", Json::i(d.offset as i64)),
                                ("len", Json::i(d.bytes.len() as i64)),
                                ("readonly", Json::Bool(d.readonly)),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "pages",
                Json::Arr(
                    pages
                        .iter()
                        .map(|(base, k)| {
                            Json::obj(vec![
                                ("base", Json::i(*base as i64)),
                                (
                                    "kind",
                                    Json::s(kinds.get(*k as usize).copied().unwrap_or("?")),
                                ),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "stack",
                Json::obj(vec![
                    ("argsBase", Json::i(svm_ir::POWERBOX_ARGS_BASE as i64)),
                    ("argsEnd", Json::i(svm_ir::POWERBOX_ARGS_END as i64)),
                    ("stackPage", Json::i(svm_ir::POWERBOX_STACK_PAGE as i64)),
                    (
                        "stackReserve",
                        Json::i(svm_ir::POWERBOX_STACK_RESERVE as i64),
                    ),
                ]),
            ),
        ];
        if self.powerbox {
            let word = |off: u64| -> Option<i64> {
                let b = self.read_window(off, 8).ok()?;
                Some(i64::from_le_bytes(b.try_into().ok()?))
            };
            if let (Some(brk), Some(top)) = (
                word(svm_ir::POWERBOX_HEAP_BRK),
                word(svm_ir::POWERBOX_HEAP_TOP),
            ) {
                fields.push((
                    "heap",
                    Json::obj(vec![
                        ("base", Json::i(mapped as i64)),
                        ("brk", Json::i(brk)),
                        ("top", Json::i(top)),
                    ]),
                ));
            }
        }
        Some(Json::obj(fields))
    }
    /// The trace tape is a threaded-engine feature (a single vCPU has no schedule).
    fn set_sched_trace(&mut self, on: bool) -> bool {
        match &mut self.engine {
            Engine::Threaded(run) => {
                run.set_sched_trace(on);
                self.sched_trace = on;
                true
            }
            Engine::Single(_) => false,
        }
    }
    fn sched_trace_json(&self) -> Option<Json> {
        let Engine::Threaded(run) = &self.engine else {
            return None;
        };
        let tape = run.sched_trace()?;
        use bytecode::SchedTraceEvent as E;
        Some(Json::Arr(
            tape.iter()
                .map(|e| match e {
                    E::Turn { turn, task } => Json::obj(vec![
                        ("kind", Json::s("turn")),
                        ("turn", Json::i(*turn as i64)),
                        ("task", Json::i(*task as i64)),
                    ]),
                    E::ParkJoin { turn, task, child } => Json::obj(vec![
                        ("kind", Json::s("parkJoin")),
                        ("turn", Json::i(*turn as i64)),
                        ("task", Json::i(*task as i64)),
                        ("child", Json::i(*child as i64)),
                    ]),
                    E::ParkWait { turn, task, key } => Json::obj(vec![
                        ("kind", Json::s("parkWait")),
                        ("turn", Json::i(*turn as i64)),
                        ("task", Json::i(*task as i64)),
                        ("key", Json::i(*key as i64)),
                    ]),
                    E::WakeNotify { turn, waker, wakee } => Json::obj(vec![
                        ("kind", Json::s("wakeNotify")),
                        ("turn", Json::i(*turn as i64)),
                        ("waker", Json::i(*waker as i64)),
                        ("wakee", Json::i(*wakee as i64)),
                    ]),
                    E::WakeJoin { turn, waker, wakee } => Json::obj(vec![
                        ("kind", Json::s("wakeJoin")),
                        ("turn", Json::i(*turn as i64)),
                        ("waker", Json::i(*waker as i64)),
                        ("wakee", Json::i(*wakee as i64)),
                    ]),
                    E::WakeTimeout { turn, task } => Json::obj(vec![
                        ("kind", Json::s("wakeTimeout")),
                        ("turn", Json::i(*turn as i64)),
                        ("task", Json::i(*task as i64)),
                    ]),
                    E::Spawn { turn, parent, task } => Json::obj(vec![
                        ("kind", Json::s("spawn")),
                        ("turn", Json::i(*turn as i64)),
                        ("parent", Json::i(*parent as i64)),
                        ("task", Json::i(*task as i64)),
                    ]),
                })
                .collect(),
        ))
    }
    /// Slice 8: apply + record a window write. The engine re-applies it at this clock on every
    /// path that passes it — live resume and seek replay alike.
    fn write_window(&mut self, addr: u64, bytes: &[u8]) -> bool {
        let (ok, now) = match &mut self.engine {
            Engine::Single(run) => (run.write_window(addr, bytes), run.op_clock()),
            Engine::Threaded(run) => (run.write_window(addr, bytes), run.op_turn()),
        };
        if ok {
            self.writes.push((
                now,
                ScheduledWrite::Window {
                    addr,
                    bytes: bytes.to_vec(),
                },
            ));
            self.sync_writes();
        }
        ok
    }
    /// Slice 8: apply + record a variable write (with the focused task on the scheduled engine,
    /// so replays resolve it in the same thread).
    fn write_var(&mut self, frame_from_top: usize, name: &str, value: i64, width: usize) -> bool {
        let (ok, now, task) = match &mut self.engine {
            Engine::Single(run) => (
                run.write_var(frame_from_top, name, value, width),
                run.op_clock(),
                0,
            ),
            Engine::Threaded(run) => (
                run.write_var(frame_from_top, name, value, width),
                run.op_turn(),
                run.focus_task(),
            ),
        };
        if ok {
            self.writes.push((
                now,
                ScheduledWrite::Var {
                    task,
                    frame: frame_from_top,
                    name: name.to_string(),
                    value,
                    width,
                },
            ));
            self.sync_writes();
        }
        ok
    }
    /// Slice 7: resolve + record a forced switch on the threaded engine.
    fn force_switch(&mut self, target: Option<usize>) -> Option<usize> {
        let Engine::Threaded(run) = &mut self.engine else {
            return None;
        };
        let runnable = run.runnable_tasks();
        let chosen = match target {
            Some(t) if runnable.contains(&t) => t,
            Some(_) => return None, // the named task isn't runnable — refuse, don't guess
            // Default: the lowest-index runnable that is *not* the schedule's default choice —
            // "switch away"; with a single runnable task there is nothing to switch to.
            None => *runnable.get(1).or_else(|| runnable.first())?,
        };
        let turn = run.op_turn();
        self.forced.push((turn, chosen));
        run.set_forced_switches(self.forced.clone());
        Some(chosen)
    }
    /// The engine-level sink installer (both bytecode engines support it).
    fn set_access_sink(&mut self, sink: SharedSink) -> bool {
        BytecodeBackend::set_access_sink(self, sink);
        true
    }
    /// W4 blocking stdin: append the provided bytes to the parked single-vCPU run's stdin — the
    /// next resume re-issues the parked read against them (and the completed read joins the
    /// session's cap tape, so a later reverse `seek` replays it faithfully).
    fn provide_stdin(&mut self, bytes: &[u8]) -> bool {
        if !self.block_stdin {
            return false;
        }
        match &mut self.engine {
            Engine::Single(run) => {
                run.provide_stdin(bytes);
                true
            }
            Engine::Threaded(_) => false, // declined at construction; unreachable in practice
        }
    }
    /// The guest's captured stdout at the current stop (the on-ramp powerbox's `write` output). On a
    /// reverse `seek` the run is rebuilt and replayed to the earlier point, so this reflects exactly the
    /// output produced up to *here* — it rewinds with the program. Empty for a deny-all session.
    fn stdout(&self) -> &[u8] {
        match &self.engine {
            Engine::Single(run) => &run.host().stdout,
            Engine::Threaded(run) => &run.host().stdout,
        }
    }
}
