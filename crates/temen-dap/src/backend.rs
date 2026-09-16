//! The **debuggee backend seam** (DEBUGGING.md G3). `DapServer` was hard-wired to the tree-walking
//! `temen_interp::Inspector`; this trait lets the same server drive *either* engine — the tree-walker
//! (the reference oracle, full feature set) or the **bytecode VM** the browser playground actually
//! runs, over `temen_interp::bytecode::ScheduledDebugRun`.
//!
//! The bytecode engine is one deterministic cooperative multi-vCPU **debug scheduler** — a spawn-free
//! guest is simply a one-task schedule (#1517 slice 4 collapsed the former single-vCPU `DebugRun` into
//! it). It covers breakpoints (firing in whichever thread reaches them), `threads`/`stopped_task`/
//! `select_task` per-thread stacks, stepping (in/over/out) of the stopped thread, scalar/aggregate
//! inspection, **cross-thread data breakpoints** (`set_watchpoint` — a per-op check of the effective
//! address, computed like the interpreter's `access_of`, against the watched ranges; the schedule
//! stops *before* an op that touches one — and #1229 value watches), and **reverse debugging**
//! (`seek`/`step_back`/`reverseContinue`, by deterministic replay to a global scheduler `turn`: the
//! debug run is pure compute plus a recorded cap tape, so seeking to an earlier turn rebuilds a fresh
//! run and replays to it, restarting from the nearest **checkpoint** in a ladder so the replay is
//! bounded by the checkpoint stride rather than O(t) from turn 0) — *not* delegated to the
//! tree-walker (the differential oracle only, far too slow for any user-facing path).
//! `supports_reverse`/`supports_watch` are both `true`. Correctness is guaranteed by
//! `crates/temen/tests/debug_parity.rs` and `bytecode_debug_threads.rs` (engine level, vs the
//! tree-walker oracle) and `dap_over_bytecode_*` (server level).

use temen_interp::bytecode::{
    self, AccessSinkFn, SchedBreak, SchedStop, ScheduledContinuation, ScheduledDebugRun,
    ScheduledWrite, ValueWatchTarget,
};
use temen_interp::moment::Ladder;
use temen_interp::MemEvent;

use crate::json::Json;
use temen_interp::{
    cap_id, BoundImport, CapTape, FrameInfo, Host, Inspector, IrPc, SourceLoc, Stop, StopReason,
    Trap, Value, VarValue, WatchId, WatchKind,
};
use temen_ir::{FuncIdx, Module};

/// #1366 slice (c) — a **declared host-completed cap** the launch named (`hostCaps`) is parked on:
/// the completion id the run parked with, the cap's name, and the guest's call arguments (a flat
/// `call.sym "<name>"` — the guest's op rides in `args[0]` by convention, like `vm_fs`). Filled by
/// the proc's submit hook, read back by the DAP `stopped` event, cleared by `provideCap`.
#[derive(Clone, Debug, PartialEq)]
pub struct CapRequest {
    pub id: u64,
    pub name: String,
    pub args: Vec<i64>,
}

/// The parked-request cell one session's declared procs share with its backend (`Arc` so a rebuilt
/// run's procs write the same cell).
pub type SharedCapRequest = std::sync::Arc<std::sync::Mutex<Option<CapRequest>>>;

/// Grant the **on-ramp I/O powerbox** on `host` for module `m`: the §3e prefix (stdout/stdin/exit/
/// memory/addrspace), each registered under its `self.resolve` name, plus the module's manifest
/// imports bound to it (IMPORTS.md phase 4 — [`io_cap`] maps each import name to the granted handle).
/// This is the powerbox a chibicc `_start` expects (minus the browser's graphical caps), so a debugged C
/// program that `printf`s (→ a `write` cap) runs instead of `CapFault`ing; its output lands in
/// `host.stdout`. The browser's `grant_onramp_caps` is the twin used for the non-debug Run path.
fn grant_io_powerbox(
    host: &mut Host,
    m: &Module,
    stdin: &[u8],
    fs_seed: Option<&temen_fs::FsSeed>,
    host_caps: &[String],
    parked: &SharedCapRequest,
) {
    host.stdin = stdin.to_vec();
    let win = m.memory.map_or(0, |mc| 1u64 << mc.size_log2);
    // The §3e prefix + its canonical-name registration — the shared sequence every powerbox host
    // performs (#912), so a debugged guest sees the same handles in the same order the Run path gives
    // it. This session's own capabilities (`vm_fs`, the declared host-completed ones) follow.
    let granted = temen_ir::PowerboxHandles::prefix(host.grant_powerbox_prefix(win));
    // #1323 (c_interpret #16, file I/O): a debugged program that does file I/O reaches a private,
    // in-memory **read-write** scratch filesystem through the `vm_fs` seam (chibicc `__vm_fs` builtin
    // → `call.sym "vm_fs"`, a flat call with base op 0 and the fs op in arg0). Mirror the browser Run
    // path (`grant_onramp_caps`): grant the same `temen-fs` memfs, wrapped to forward `args[0]` as the
    // op, and bind the `vm_fs` slot to it below. Granted only when the module imports it (a plain
    // stdout-only program is unaffected); guest-private, no host disk, dropped at session end.
    let vm_fs_h: Option<i32> = if m.imports.iter().any(|im| im.name == "vm_fs") {
        // #1323 slice 3: seed the memfs with the launch's fs-image when one was supplied (a lesson's
        // pre-seeded input files, e.g. a `colors.txt` the guest `fopen`s for read), else an empty
        // scratch store. Each build clones the seed fresh, so a reverse-`seek` rebuild re-seeds
        // identically (deterministic replay).
        let mut inner = match fs_seed {
            Some((files, dirs)) => temen_fs::mem_fs_seeded_handler(files.clone(), dirs.clone())(),
            None => temen_fs::mem_fs_handler(false)(),
        };
        let h = host.grant_host_proc(Box::new(
            move |_slot_op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn temen_interp::GuestMem>,
                  minter: Option<&mut dyn temen_interp::RegionMinter>| {
                let (op, rest) = args
                    .split_first()
                    .map(|(o, r)| (*o as u32, r))
                    .unwrap_or((0, &[][..]));
                inner(op, rest, mem, minter)
            },
        ));
        host.register_cap_name("vm_fs", h);
        Some(h)
    } else {
        None
    };
    // #1366 slice (c): the launch's **declared host-completed caps** (`hostCaps`). Each name the
    // module imports gets an offloadable proc that always punts to the host
    // (`OffloadOutcome::Host`): the guest's flat `call.sym "<name>"` parks the run, the request
    // lands in `parked` (read by the DAP `stopped{reason:"cap"}` event), and `provideCap` resumes
    // it. Granted in launch order on every rebuild, so a reverse `seek`'s replay finds the same
    // handles the tape recorded. Nothing here is graphics- or embedder-specific: temen never
    // learns what a name means.
    let mut declared: Vec<(String, i32)> = Vec::new();
    for name in host_caps {
        if !m.imports.iter().any(|im| &im.name == name) {
            continue;
        }
        let cell = std::sync::Arc::clone(parked);
        let cap_name = name.clone();
        let h = host.grant_host_proc_offloadable(Box::new(move |_op: u32, args: &[i64]| {
            let cell = std::sync::Arc::clone(&cell);
            let name = cap_name.clone();
            let args = args.to_vec();
            temen_interp::OffloadOutcome::Host(Box::new(move |id| {
                *cell.lock().unwrap_or_else(|e| e.into_inner()) =
                    Some(CapRequest { id, name, args });
            }))
        }));
        host.register_cap_name(name, h);
        declared.push((name.clone(), h));
    }
    if !m.imports.is_empty() {
        let bindings = m
            .imports
            .iter()
            .map(|im| {
                // #1366 slice (c): a declared host-completed cap — a flat `call.sym` (base op 0)
                // on the proc granted above, like `vm_fs`.
                if let Some((_, h)) = declared.iter().find(|(n, _)| *n == im.name) {
                    return BoundImport::required(cap_id::HOST_PROC, 0, *h);
                }
                // #1323: the `vm_fs` file-I/O seam — a flat `call.sym` (base op 0) on the memfs
                // HostProc granted above; the guest's fs op rides in arg0.
                if im.name == "vm_fs" {
                    return match vm_fs_h {
                        Some(h) => BoundImport::required(cap_id::HOST_PROC, 0, h),
                        None => BoundImport::rebindable(0, 0, None),
                    };
                }
                // The shared powerbox ABI (#912): the name's capability and the handle this
                // session granted for it. A name it did not grant (the `Jit` cap, `stderr`) or a
                // dynamic-only interface leaves its slot unbound — fail-closed at dispatch.
                match granted.bind(&im.name) {
                    Some((cap, handle)) => BoundImport::required(cap.type_id, cap.op, handle),
                    None => BoundImport::rebindable(0, 0, None),
                }
            })
            .collect();
        host.set_import_bindings(bindings);
    }
}

/// Build a [`ScheduledDebugRun`] for `module`'s `func(args)`: under the on-ramp I/O powerbox when
/// `powerbox` (recording cap inputs, and replaying `tape` from a prior forward run so a reverse-`seek`
/// rebuild re-executes with identical clock/stdin inputs — so a C guest's `malloc`/`printf` reach the
/// `memory`/`write` caps instead of `CapFault`ing, and `main`'s return becomes an `exit` code), else
/// deny-all. The `seed` is the slice-7 schedule variation. `None` if the module is outside the
/// engine's subset. `block_stdin` (W4) arms the blocking-stdin park on the powerbox host: a thread's
/// `read` on an exhausted buffer parks it (`SchedStop::StdinPark`) instead of returning EOF, re-armed
/// on every `seek` rebuild so a read past the replay frontier parks again. `mem_limit` (slice 5) is
/// the Memory-capability growth cap — a `vm_map` past it returns -ENOMEM, so a guest malloc observes
/// NULL (the OOM-teaching knob) — likewise re-armed on every rebuild.
#[allow(clippy::too_many_arguments)]
fn build_run(
    module: &Module,
    func: FuncIdx,
    args: &[Value],
    powerbox: bool,
    stdin: &[u8],
    block_stdin: bool,
    mem_limit: Option<u64>,
    seed: Option<u64>,
    fs_seed: Option<&temen_fs::FsSeed>,
    host_caps: &[String],
    parked: &SharedCapRequest,
    tape: &CapTape,
) -> Option<ScheduledDebugRun> {
    let mut run = if powerbox {
        let mut host = Host::new();
        grant_io_powerbox(&mut host, module, stdin, fs_seed, host_caps, parked);
        host.set_mem_map_limit(mem_limit);
        if block_stdin {
            host.set_stdin_blocking(true);
        }
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
    /// Arm a **value watchpoint** on an SSA-held (address-less) source variable `name` in the frame
    /// `frame_from_top` levels up — stop when its value changes (#1229). `None` when the variable has
    /// no SSA location there (it's memory-located — watch it with [`set_watchpoint`] via
    /// [`var_addr`](Debuggee::var_addr) — or isn't live/known), or the backend doesn't serve value
    /// watches. Default `None`.
    fn set_value_watchpoint(
        &mut self,
        frame_from_top: usize,
        name: &str,
        kind: WatchKind,
    ) -> Option<WatchId> {
        let _ = (frame_from_top, name, kind);
        None
    }

    // --- inspection ------------------------------------------------------------------------------
    fn backtrace(&self) -> Vec<FrameInfo>;
    fn func_name(&self, func: FuncIdx) -> Option<&str>;
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc>;
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue>;
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64>;
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap>;
    /// The faulting guest address of the run's last `MemoryFault`, window-relative (a NULL deref →
    /// `0`); `None` if unknown or the last termination was not an address-recording memory fault.
    /// Read by `stop_events` to attach `faultAddr` to the `exited` event (#1190). Default `None` for
    /// engines that don't track it.
    fn fault_addr(&self) -> Option<u64> {
        None
    }

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

    /// #1366 — deliver the embedder's value for the host-completed cap call the session is parked
    /// on (`StopReason::CapPark { id }`); the next resume continues past the call. `false` when
    /// the session isn't parked on `id` (the `provideCap` request fails cleanly). Default:
    /// unsupported.
    fn provide_cap(&mut self, _id: u64, _value: i64) -> bool {
        false
    }

    /// #1366 slice (c) — the declared host-completed cap the session is parked on: `(name, args)`
    /// for the `stopped{reason:"cap"}` event's `capName`/`args`. `None` when not parked on a
    /// declared cap (or on a backend without declared caps). Default: none.
    fn cap_park_request(&self) -> Option<(String, Vec<i64>)> {
        None
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
    fn set_value_watchpoint(
        &mut self,
        frame_from_top: usize,
        name: &str,
        kind: WatchKind,
    ) -> Option<WatchId> {
        Inspector::set_value_watchpoint(self, frame_from_top, name, kind)
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
    fn fault_addr(&self) -> Option<u64> {
        Inspector::fault_addr(self)
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

/// The **bytecode backend** — the resumable bytecode debug session ([`ScheduledDebugRun`]) plus the
/// persistent breakpoint set `DapServer` expects, the module (for `source_loc`/`func_name`, which are
/// engine-neutral free functions keyed on the `IrPc`), and the launch `func`/`args` so reverse
/// debugging can rebuild a fresh run and replay to an earlier turn.
pub struct BytecodeBackend {
    /// #1366 slice (c): the launch's declared host-completed cap names (`hostCaps`), re-granted on
    /// every rebuild (see `grant_io_powerbox`).
    host_caps: Vec<String>,
    /// #1366 slice (c): the request the run is currently parked on (filled by a declared proc's
    /// submit hook; read by `cap_park_request`; cleared by `provide_cap`).
    parked_cap: SharedCapRequest,
    run: ScheduledDebugRun,
    module: Module,
    func: FuncIdx,
    args: Vec<Value>,
    breakpoints: Vec<IrPc>,
    /// Armed watchpoints with backend-owned stable ids (re-applied to the run after a `seek` rebuild).
    /// Cross-thread: a range fires in whichever thread touches it.
    watch_specs: Vec<(WatchId, u64, u64, WatchKind)>,
    /// Armed **value** watchpoints on SSA-held source variables (#1229), backend-owned stable ids,
    /// re-applied after a `seek` rebuild like `watch_specs` (the target is frame-independent so
    /// re-application is verbatim).
    value_specs: Vec<(WatchId, ValueWatchTarget, WatchKind)>,
    next_watch: u32,
    fuel: u64,
    /// This session runs its guest under the **on-ramp I/O powerbox** ([`grant_io_powerbox`]) instead of
    /// deny-all — so a manifest program that calls a host capability (a chibicc `printf` → `write`) runs
    /// and its output is captured, rather than `CapFault`ing. Off ⇒ the compute-only path, unchanged.
    powerbox: bool,
    /// Preloaded stdin for the powerbox (`read(0, …)`); empty for a pure-output program.
    stdin: Vec<u8>,
    /// #1323 slice 3: the launch's fs-image seed (a lesson's pre-seeded files + dirs) mounted on the
    /// `vm_fs` memfs; cloned fresh on every (re)build so a reverse-`seek` re-seeds identically.
    /// `None` = an empty scratch store.
    fs_seed: Option<temen_fs::FsSeed>,
    /// W4 blocking stdin: a `read` on an exhausted buffer parks the session
    /// (`StopReason::StdinPark`, resumed by `provideStdin`) instead of returning EOF (powerbox
    /// sessions).
    block_stdin: bool,
    /// Slice 5: the session's Memory-capability growth cap ([`Host::set_mem_map_limit`]) — set on
    /// the powerbox at build and on every seek rebuild. `None` = unbounded.
    mem_limit: Option<u64>,
    /// Slice 6: whether the scheduler trace tape is armed — re-armed on every seek rebuild so the
    /// replay refills the tape deterministically.
    sched_trace: bool,
    /// Slice 7: the seeded-pick schedule policy — applied at construction
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
    /// Time-travel **checkpoint ladder** (DEBUGGING.md W1): snapshots of the run at ascending global
    /// turns (kept sorted) so a reverse `seek`/`step_back` restarts from the nearest one (`turn <= t`)
    /// instead of turn 0, bounding the replay to [`CHECKPOINT_STRIDE`]. Populated lazily as `seek`
    /// drives past stride boundaries — the bytecode port of the tree-walker `Inspector`'s ladder. One
    /// `Ladder` (unbounded) keyed on the turn; the ladder itself is the same type the tree-walker and
    /// the reactor timeline use (`temen_interp::moment`, #1460).
    checkpoints: Ladder<ScheduledContinuation>,
    /// Whether checkpointing is still active. Cleared (and the ladder dropped) the first time a stride
    /// boundary falls outside the [`ScheduledDebugRun::snapshot`] subset (a
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
    /// Open a bytecode debug session on `module`'s `func(args)`. `None` if the module is outside the
    /// bytecode engine's subset (`compile_module` declines it), or a schedule `seed` is given for a
    /// spawn-free guest (meaningless with one vCPU — declined rather than silently ignored).
    /// `powerbox` runs the guest under the on-ramp I/O powerbox (`stdin` preloads `read`).
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
        Self::new_with_fs_seed(
            module,
            func,
            args,
            fuel,
            powerbox,
            stdin,
            block_stdin,
            mem_limit,
            seed,
            None,
            Vec::new(),
        )
    }

    /// #1323 slice 3: [`new`](Self::new) plus a `vm_fs` **memfs seed** — the launch's fs-image
    /// (a lesson's pre-seeded input files) mounted so the guest can `fopen` them for read. `new`
    /// is the unseeded (empty scratch store) shorthand.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_fs_seed(
        module: Module,
        func: FuncIdx,
        args: &[Value],
        fuel: u64,
        powerbox: bool,
        stdin: Vec<u8>,
        block_stdin: bool,
        mem_limit: Option<u64>,
        seed: Option<u64>,
        fs_seed: Option<temen_fs::FsSeed>,
        host_caps: Vec<String>,
    ) -> Option<BytecodeBackend> {
        let tape = CapTape::default();
        let parked_cap: SharedCapRequest = SharedCapRequest::default();
        // A seed is meaningless with one vCPU — decline rather than silently ignore it.
        if seed.is_some() && !bytecode::module_spawns_threads(&module) {
            return None;
        }
        let run = build_run(
            &module,
            func,
            args,
            powerbox,
            &stdin,
            block_stdin,
            mem_limit,
            seed,
            fs_seed.as_ref(),
            &host_caps,
            &parked_cap,
            &tape,
        )?;
        Some(BytecodeBackend {
            run,
            module,
            func,
            args: args.to_vec(),
            breakpoints: Vec::new(),
            watch_specs: Vec::new(),
            value_specs: Vec::new(),
            next_watch: 0,
            fuel,
            powerbox,
            stdin,
            fs_seed,
            host_caps,
            parked_cap,
            block_stdin,
            mem_limit,
            tape,
            rev_trace: None,
            checkpoints: Ladder::new(CHECKPOINT_STRIDE, 0, 0),
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
        self.run.set_access_sink(wrap_sink(&sink));
        self.access_sink = Some(sink);
    }

    /// Number of time-travel checkpoints currently in the ladder — test/introspection hook (mirrors
    /// `Inspector::checkpoint_count`). `0` for a run not yet seeked far enough to lay one down, or one
    /// outside the checkpointable subset (an event-parked fiber, a mid-invoke task, a non-pristine
    /// layout, a stateful host), which replays from turn 0.
    pub fn checkpoint_count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Rebuild a fresh run with this session's engine config — under the on-ramp I/O powerbox
    /// (recording + replaying `tape`) when `self.powerbox`, else deny-all, with the schedule seed —
    /// the base a reverse `seek`/rev-trace probe re-drives from, so a re-executed powerbox run sees
    /// identical cap inputs.
    fn fresh(&self) -> Option<ScheduledDebugRun> {
        build_run(
            &self.module,
            self.func,
            &self.args,
            self.powerbox,
            &self.stdin,
            self.block_stdin,
            self.mem_limit,
            self.seed,
            self.fs_seed.as_ref(),
            &self.host_caps,
            &self.parked_cap,
            &self.tape,
        )
    }

    /// Drive a freshly rebuilt (and possibly checkpoint-restored) `run` forward to global turn `t`,
    /// laying down a checkpoint at each [`CHECKPOINT_STRIDE`] boundary along the way (so a later
    /// reverse `seek`/`step_back` restarts nearby). With checkpointing off it is a straight tick-to-`t`.
    /// The stride capture is transparent — ticking is the same raw replay quantum regardless — so the
    /// state at `t` is identical to a single run from the restore point to `t`.
    fn drive_to(&mut self, run: &mut ScheduledDebugRun, t: u64, fuel: &mut u64) {
        loop {
            let turn = run.op_turn();
            if turn >= t {
                break;
            }
            // At a positive stride boundary short of `t`, snapshot before executing the op (so the
            // checkpoint's turn is exactly the boundary). Deduped + subset-guarded by `maybe_checkpoint`.
            if self.checkpointing && turn > 0 && turn.is_multiple_of(CHECKPOINT_STRIDE) {
                self.maybe_checkpoint(run);
            }
            if !run.tick(fuel) {
                break;
            }
        }
    }

    /// Snapshot `run` into the checkpoint ladder at its current turn, if checkpointing is still on and
    /// the continuation is [`ScheduledDebugRun::snapshot`]-able; otherwise disable checkpointing and
    /// drop the ladder (the run left the snapshottable subset, so `seek` reverts to replay-from-0). A
    /// turn already in the ladder is not duplicated. Mirrors `Inspector::maybe_checkpoint`.
    fn maybe_checkpoint(&mut self, run: &ScheduledDebugRun) {
        if !self.checkpointing {
            return;
        }
        let Some(snap) = run.snapshot() else {
            self.checkpointing = false;
            self.checkpoints.clear();
            return;
        };
        // Sorted insert, deduped, is the ladder's own contract now.
        self.checkpoints.take(run.op_turn(), snap);
    }

    /// Push the recorded writes into the live engine (slice 8) — its cursor lands past entries at
    /// clocks already passed, so the just-applied live write isn't double-counted going forward.
    fn sync_writes(&mut self) {
        self.run.set_scheduled_writes(self.writes.clone());
    }

    /// Absorb the live run's recorded cap-input tape if it now reaches further than the one held — so
    /// a later reverse `seek` replays the furthest-forward inputs. Cheap no-op for a pure-output
    /// (`write`-only) program (its tape stays empty) and for deny-all sessions.
    fn capture_tape(&mut self) {
        if !self.powerbox {
            return;
        }
        let live = self.run.host().cap_tape();
        if live.records.len() > self.tape.records.len() {
            self.tape = live;
        }
    }

    /// Push the current watchpoints — window ranges and #1229 value watches — into the live run
    /// (after arming/clearing one, or re-arming a fresh run built by `seek`; a value target is
    /// frame-independent, so re-application is verbatim). Both are cross-thread.
    fn apply_watches(&mut self) {
        let ranges: Vec<_> = self
            .watch_specs
            .iter()
            .map(|(_, a, l, k)| (*a, *l, *k))
            .collect();
        self.run.set_watchpoints(ranges);
        self.run.set_value_watches(self.value_specs.clone());
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
        let Some(mut probe) = self.fresh() else {
            return false;
        };
        // The probe must replay the *same schedule* as the session (slice 7 policy; the seed rides
        // `fresh`) and the same debugger writes (slice 8), or its timeline diverges.
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
        let (parked, pc) = (self.run.stdin_parked(), self.run.frame_pc(0));
        // #1366: parked on a host-completed cap call — live, paused past the call, resumable once
        // `provideCap` delivers the value.
        if let Some((id, at)) = self.run.cap_parked().map(|id| (id, self.run.cap_park_pc())) {
            // The stop location is the call itself (the position after it may be a terminator).
            if let Some(pc) = at.or(pc) {
                return Stop::Break {
                    reason: StopReason::CapPark { id },
                    pc,
                };
            }
        }
        if let (true, Some(pc)) = (parked, pc) {
            return Stop::Break {
                reason: StopReason::StdinPark,
                pc,
            };
        }
        match self.run.result() {
            Some(r) => Stop::Finished(r.clone()),
            None => Stop::Blocked,
        }
    }

    /// A `Step` stop at the focused thread's current pc (or the finished result), for a resume/seek that
    /// didn't hit a breakpoint.
    fn step_stop(&self) -> Stop {
        match self.run.frame_pc(0) {
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
            // W4 (#1146 deeper): every thread parked and one of them in a blocking-stdin read — a live
            // stop at that read, resumable after `provideStdin`.
            SchedStop::StdinPark { pc } => Stop::Break {
                reason: StopReason::StdinPark,
                pc,
            },
            // No runnable thread (deadlock/`wait`), or an op outside the scheduler's subset.
            // #1366: parked on a host-completed cap — live, paused at the call, resumable once
            // `provideCap` delivers (the scheduled twin of the single engine's `CapPark` stop).
            SchedStop::CapPark { id, pc } => Stop::Break {
                reason: StopReason::CapPark { id },
                pc,
            },
            SchedStop::Blocked | SchedStop::Declined => Stop::Blocked,
        }
    }
}

impl Debuggee for BytecodeBackend {
    fn run_until_stop(&mut self) -> Stop {
        // A fresh fuel budget per resume (debugging is interactive; the run replays from scratch on a
        // seek, so a shared decrementing counter would be inconsistent).
        let mut fuel = self.fuel;
        self.run.set_breakpoints(self.breakpoints.clone());
        let stop = Self::sched_stop(self.run.run_until_stop(&mut fuel));
        self.capture_tape(); // absorb any new cap inputs this advance recorded (for a later reverse seek)
        stop
    }
    fn step(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = Self::sched_stop(self.run.step(&mut fuel));
        self.capture_tape();
        stop
    }
    fn step_over(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = Self::sched_stop(self.run.step_over(&mut fuel));
        self.capture_tape();
        stop
    }
    fn step_out(&mut self) -> Stop {
        let mut fuel = self.fuel;
        let stop = Self::sched_stop(self.run.step_out(&mut fuel));
        self.capture_tape();
        stop
    }
    // Reverse debugging by **deterministic replay** (DEBUGGING.md W1): the debug run is pure compute
    // plus a recorded cap tape, so seeking to an earlier turn = rebuild a fresh run and replay to that
    // many turns. `step_back` = one stoppable op earlier. The `seek` replay is bounded by the
    // **checkpoint ladder** (see `drive_to`/`maybe_checkpoint`): a restart from the nearest snapshot
    // replays at most `CHECKPOINT_STRIDE` turns instead of O(t) from turn 0.
    fn step_back(&mut self) -> Stop {
        // Rewind to the previous op that sits at a real IR instruction (a stoppable position — not a
        // terminator slot, where there's nothing to inspect) strictly before now, then seek there.
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
        let (now, now_depth) = (self.run.op_turn(), self.run.depth());
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
        // Rebuild a fresh run and replay `t` turns — the schedule is deterministic, so this reproduces
        // the exact state at global turn `t` (DEBUGGING.md W1).
        let Some(mut run) = self.fresh() else {
            return Stop::Blocked;
        };
        run.set_breakpoints(self.breakpoints.clone());
        // Re-install the access sink *before* the replay drive, so a model consumer observes the
        // re-execution and can re-derive its state (`seek(t)` ≡ a from-0 run to `t`).
        if let Some(sink) = &self.access_sink {
            run.set_access_sink(wrap_sink(sink));
        }
        // Re-arm the trace tape: the replay refills it deterministically from the restore point.
        if self.sched_trace {
            run.set_sched_trace(true);
        }
        // Re-apply the schedule policy (slice 7) — semantic, so the replay must carry it (the seed
        // rides `fresh`; forced switches are applied here).
        run.set_forced_switches(self.forced.clone());
        // And the recorded debugger writes (slice 8) — the replay re-applies them at their turns.
        run.set_scheduled_writes(self.writes.clone());
        // Restart from the nearest checkpoint at or before `t` (ladder kept sorted by turn) instead
        // of turn 0, when still checkpointable — bounding the replay to the stride.
        if self.checkpointing {
            if let Some((turn, cp)) = self.checkpoints.nearest_at_or_before(t) {
                run.restore(turn, cp);
            }
        }
        self.drive_to(&mut run, t, &mut fuel);
        run.locate();
        // If the replay landed exactly on a breakpoint op, arm the skip so a forward `continue` from
        // here makes progress instead of immediately re-reporting this stop.
        if let Some(pc) = run.frame_pc(0) {
            if self.breakpoints.contains(&pc) {
                run.arm_breakpoint_skip();
            }
        }
        self.run = run;
        // A `seek` past the furthest point reached so far runs new ground **live**, recording cap
        // crossings a later reverse seek would otherwise have to re-run live against a rebuilt
        // powerbox. Absorb them here, exactly as every forward advance does — a host capability's
        // closure is gone on a rebuild, so only the tape reproduces it (`is_recorded_input`).
        self.capture_tape();
        self.apply_watches(); // re-arm the range + value watches on the fresh (replayed) run
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
    // Cross-thread: fires in whichever thread touches the range.
    fn set_watchpoint(&mut self, addr: u64, len: u64, kind: WatchKind) -> Option<WatchId> {
        let id = WatchId::from_raw(self.next_watch);
        self.next_watch += 1;
        self.watch_specs.push((id, addr, len, kind));
        self.apply_watches();
        Some(id)
    }
    fn clear_watchpoint(&mut self, id: WatchId) -> bool {
        let before = self.watch_specs.len() + self.value_specs.len();
        self.watch_specs.retain(|(w, ..)| *w != id);
        self.value_specs.retain(|(w, ..)| *w != id);
        let removed = self.watch_specs.len() + self.value_specs.len() != before;
        if removed {
            self.apply_watches();
        }
        removed
    }
    // Value watches (#1229): stop when an SSA-held source variable's value changes — resolved in the
    // focused thread's frame, armed run-wide (cross-thread, like a range).
    fn set_value_watchpoint(
        &mut self,
        frame_from_top: usize,
        name: &str,
        kind: WatchKind,
    ) -> Option<WatchId> {
        let target = self.run.resolve_value_watch(frame_from_top, name)?;
        let id = WatchId::from_raw(self.next_watch);
        self.next_watch += 1;
        self.value_specs.push((id, target, kind));
        self.apply_watches();
        Some(id)
    }
    fn backtrace(&self) -> Vec<FrameInfo> {
        let mut out = Vec::new();
        // The focused thread's stack (`select_task`'s pick; the sole thread of a spawn-free guest).
        for d in 0..self.run.depth() {
            if let Some(pc) = self.run.frame_pc(d) {
                let source = temen_interp::source_loc(&self.module, pc);
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
        temen_interp::func_name(&self.module, func)
    }
    fn source_loc(&self, pc: IrPc) -> Option<SourceLoc> {
        temen_interp::source_loc(&self.module, pc)
    }
    fn read_var(&self, frame_from_top: usize, name: &str, width: usize) -> Option<VarValue> {
        self.run.read_var(frame_from_top, name, width)
    }
    fn var_addr(&self, frame_from_top: usize, name: &str) -> Option<u64> {
        self.run.var_addr(frame_from_top, name)
    }
    fn read_window(&self, addr: u64, len: usize) -> Result<Vec<u8>, Trap> {
        self.run.read_window(addr, len)
    }
    fn fault_addr(&self) -> Option<u64> {
        self.run.fault_addr()
    }
    fn threads(&self) -> Vec<u64> {
        self.run.threads()
    }
    fn select_task(&mut self, id: u64) -> bool {
        self.run.select_task(id)
    }
    fn stopped_task(&self) -> Option<u64> {
        self.run.stopped_task()
    }
    // One time coordinate: the global scheduler `turn` (for a spawn-free guest, its op count).
    fn turn(&self) -> u64 {
        self.run.turn()
    }
    fn clock(&self) -> u64 {
        self.run.turn()
    }
    // Fully reversible (deterministic replay) and watch-capable.
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
        let (page, mapped, reserved, pages) = self.run.mem_map_info()?;
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
                    // #964/#1094: a module's args blob (and low scratch) sits one guard up (the
                    // unconditional guarded layout) — report where THIS module actually reads it.
                    ("argsBase", Json::i(temen_ir::module_args_base() as i64)),
                    ("argsEnd", Json::i(temen_ir::module_args_end() as i64)),
                    ("stackPage", Json::i(temen_ir::POWERBOX_STACK_PAGE as i64)),
                    (
                        "stackReserve",
                        Json::i(temen_ir::POWERBOX_STACK_RESERVE as i64),
                    ),
                ]),
            ),
        ];
        if self.powerbox {
            let word = |off: u64| -> Option<i64> {
                let b = self.read_window(off, 8).ok()?;
                Some(i64::from_le_bytes(b.try_into().ok()?))
            };
            // #964: the heap words ride the (possibly guard-shifted) low scratch.
            let scratch = temen_ir::module_null_guard();
            if let (Some(brk), Some(top)) = (
                word(scratch + temen_ir::POWERBOX_HEAP_BRK),
                word(scratch + temen_ir::POWERBOX_HEAP_TOP),
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
    /// The scheduler trace tape (a one-task schedule traces its turns alone).
    fn set_sched_trace(&mut self, on: bool) -> bool {
        self.run.set_sched_trace(on);
        self.sched_trace = on;
        true
    }
    fn sched_trace_json(&self) -> Option<Json> {
        let tape = self.run.sched_trace()?;
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
    /// Slice 8: apply + record a window write. The engine re-applies it at this turn on every
    /// path that passes it — live resume and seek replay alike.
    fn write_window(&mut self, addr: u64, bytes: &[u8]) -> bool {
        let ok = self.run.write_window(addr, bytes);
        if ok {
            self.writes.push((
                self.run.op_turn(),
                ScheduledWrite::Window {
                    addr,
                    bytes: bytes.to_vec(),
                },
            ));
            self.sync_writes();
        }
        ok
    }
    /// Slice 8: apply + record a variable write (with the focused task, so replays resolve it in
    /// the same thread).
    fn write_var(&mut self, frame_from_top: usize, name: &str, value: i64, width: usize) -> bool {
        let ok = self.run.write_var(frame_from_top, name, value, width);
        if ok {
            self.writes.push((
                self.run.op_turn(),
                ScheduledWrite::Var {
                    task: self.run.focus_task(),
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
    /// Slice 7: resolve + record a forced switch.
    fn force_switch(&mut self, target: Option<usize>) -> Option<usize> {
        let runnable = self.run.runnable_tasks();
        let chosen = match target {
            Some(t) if runnable.contains(&t) => t,
            Some(_) => return None, // the named task isn't runnable — refuse, don't guess
            // Default: the lowest-index runnable that is *not* the schedule's default choice —
            // "switch away". With a single runnable task (a spawn-free guest always) there is
            // nothing to switch to, so `get(1)` is `None` and the request fails cleanly.
            None => *runnable.get(1)?,
        };
        let turn = self.run.op_turn();
        self.forced.push((turn, chosen));
        self.run.set_forced_switches(self.forced.clone());
        Some(chosen)
    }
    /// The engine-level sink installer.
    fn set_access_sink(&mut self, sink: SharedSink) -> bool {
        BytecodeBackend::set_access_sink(self, sink);
        true
    }
    /// W4 blocking stdin: append the provided bytes to the parked run's stdin — the next resume
    /// re-issues the parked read against them (and the completed read joins the session's cap
    /// tape, so a later reverse `seek` replays it faithfully).
    fn provide_stdin(&mut self, bytes: &[u8]) -> bool {
        if !self.block_stdin {
            return false;
        }
        self.run.provide_stdin(bytes);
        true
    }
    /// #1366 — deliver the value for the host-completed cap call the session is parked on.
    fn provide_cap(&mut self, id: u64, value: i64) -> bool {
        let ok = self.run.deliver_cap(id, value);
        if ok {
            *self.parked_cap.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        ok
    }
    fn cap_park_request(&self) -> Option<(String, Vec<i64>)> {
        let parked = self.run.cap_parked()?;
        let g = self.parked_cap.lock().unwrap_or_else(|e| e.into_inner());
        g.as_ref()
            .filter(|r| r.id == parked)
            .map(|r| (r.name.clone(), r.args.clone()))
    }
    /// The guest's captured stdout at the current stop (the on-ramp powerbox's `write` output). On a
    /// reverse `seek` the run is rebuilt and replayed to the earlier point, so this reflects exactly the
    /// output produced up to *here* — it rewinds with the program. Empty for a deny-all session.
    fn stdout(&self) -> &[u8] {
        &self.run.host().stdout
    }
}
