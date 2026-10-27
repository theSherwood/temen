# Feature × run-mode matrix

**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity`. The runners and the exceptions live in `crates/temen-parity/src/run_modes.rs`; this file is its human-readable view.

Can a program that does something run in a given mode, and which modes combine (#2545)? A run is one **executor** plus any **properties** it adds. The columns are both kinds; the rows are what a program does, then the properties again.

Every cell comes from **runs**: each feature's probe on each configuration that `crates/temen-parity/tests/run_mode_conformance.rs` has a runner for (listed below). A run gives the oracle's answer unless the manifest records an exception, and the test fails when a run and the manifest disagree. What a cell folds depends on its row and column:

| row | column | the runs behind the cell |
|---|---|---|
| feature | executor | the feature's probe on the executor alone |
| feature | property | the probe on each executor that has a runner with the property |
| property | executor | every probe on the executor's runner with the property |
| property | property | every probe, on each executor that has a runner with both |

On each executor a cell uses the runner that adds its properties and the fewest others. A cell is ✅ when every run gives the oracle's answer, 🚧 when any run is a known gap, ⛔ when every run declines by design, 🔶 for any other mix, and ❔ when no runner covers it. Each pair of properties appears once, in the earlier property's row.

**124 of 192 cells audited.** Every 🚧 cell cites the issue tracking its gap.

## Legend

- ✅ **Full** — holds
- ⛔ **Declines** — deliberately does not, and the note says why
- 🚧 **Not yet** — a real gap, and the note cites its issue
- 🔶 **Conditional** — holds where the note's condition does
- ❔ **Unaudited** — nobody has established this cell

## Columns

- **tree-walk interpreter** (executor) — the tree-walk interpreter, the oracle every other mode is held to
- **bytecode, one thread** (executor) — the bytecode tier's cooperative executor (the pump): every task takes turns on one thread
- **bytecode, thread per task** (executor) — the bytecode tier's parallel executor (executor 2): every task on a thread of its own, under the pump's rules
- **Cranelift JIT** (executor) — native code from the Cranelift JIT, run through temen-run
- **under a debugger** (property) — a debugger steps the run: the tree-walk Inspector (the DAP's default engine) on the tree-walker, the bytecode debug scheduler (ScheduledDebugRun) on the pump
- **frozen and thawed** (property) — the run is frozen part-way and thawed, and finishes as if it had never stopped
- **held by the embedder** (property) — the embedder holds the run and resumes it (CoopRun, CoopSession, the debug scheduler): the run hands control back at each call the embedder answers, when it can only wait on the embedder, and at its end
- **from a warm snapshot** (property) — the run starts from a snapshot of its window taken after its setup
- **re-entered per frame** (property) — a reactor: the entry is called again, a frame at a time, over state kept between frames
- **in the browser** (property) — the wasm32 build, on a web page (the playground)
- **with the wasm-JIT** (property) — hot code is compiled to wasm by the wasm-JIT and run by the browser's engine (the browser's emitted tier)

## Rows

- **spawns threads** — spawns two threads (thread.spawn) and joins them
- **runs fibers** — runs a generator fiber (cont.new, cont.resume, suspend) to its end
- **spawns a child domain (§14)** — spawns a detached §14 child from a spawn record (Instantiator op 17) and joins it
- **forks itself (clone_caller)** — a served call forks its caller (clone_caller), and the caller reaps the twin (FORK.md §9)
- **execs another image (exec_module)** — replaces its image with another module (exec_module, FORK.md §8.6)
- **makes a call the embedder answers** — makes capability calls the embedder answers later, from outside the run (OffloadOutcome::Host)
- **makes a call a pool thread answers** — makes capability calls a runtime pool thread answers (OffloadOutcome::Offload)
- **reads input that arrives later** — reads stdin before its input has arrived, and gets it when the embedder supplies it
- **is stopped and continued (job control)** — stops its own process, and its embedder continues it. A run the embedder holds waits for it. A run nobody holds faults when its embedder may not act from outside (its external-wake bell unarmed), and waits for the continue when it may (#2529)
- **grows its window** — maps pages past its declared window through its AddressSpace and uses them
- **maps a shared region (§13)** — maps a §13 shared region at two offsets and sees a store through one in the other
- **compiles code at runtime (§22)** — compiles a §22 unit, installs it and calls it through call.dyn, and invokes it
- **serves calls on its offer** — serves the calls queued on its offer (svc.poll)

The property rows are the columns' properties again.

## Runners

- **tree-walk interpreter** — temen_interp::run_with_host
- **bytecode, one thread** — bytecode::compile_and_run_with_host
- **bytecode, thread per task** — bytecode::compile_and_run_capture_over_parallel_with_host
- **Cranelift JIT** — temen_run::jit_cap_run
- **tree-walk interpreter, under a debugger** — the tree-walk Inspector, the DAP's default engine
- **bytecode, one thread, under a debugger, held by the embedder** — the bytecode debug scheduler (ScheduledDebugRun), the DAP's bytecode engine
- **bytecode, one thread, held by the embedder** — CoopRun on the pump
- **bytecode, thread per task, held by the embedder** — CoopRun on executor 2
- **tree-walk interpreter, frozen and thawed** — run_capture_reserved_with_host_layout over an instrumented module: frozen at its first safepoint, serialized by the §12 codec, restored into a fresh powerbox and thawed
- **bytecode, one thread, frozen and thawed** — bytecode::compile_and_run_capture_reserved_with_host, frozen and thawed the same way
- **Cranelift JIT, frozen and thawed** — temen_run::jit_cap_run over a durable host, frozen and thawed the same way
- **bytecode, one thread, re-entered per frame** — bytecode::Reactor, the engine under the browser's reactors: the entry called as two frames over one window and one powerbox, each run to its end, and held to the tree-walker's capture entry called the same way
- **bytecode, one thread, held by the embedder, re-entered per frame** — bytecode::Reactor::coop_call, each frame a CoopRun the embedder holds

## Features

| a program that… | tree-walk interpreter | bytecode, one thread | bytecode, thread per task | Cranelift JIT | under a debugger | frozen and thawed | held by the embedder | from a warm snapshot | re-entered per frame | in the browser | with the wasm-JIT |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| spawns threads | ✅ | ✅ | ✅ | 🔶 | ✅ | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| runs fibers | ✅ | ✅ | ✅ | 🔶 | ✅ | 🔶 | ✅ | ❔ | ✅ | ❔ | ❔ |
| spawns a child domain (§14) | ✅ | ✅ | ✅ | 🔶 | 🚧 | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| forks itself (clone_caller) | ✅ | ✅ | ✅ | 🚧 | 🚧 | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| execs another image (exec_module) | ✅ | ✅ | ✅ | 🚧 | 🚧 | 🚧 | ✅ | ❔ | 🚧 | ❔ | ❔ |
| makes a call the embedder answers | ✅ | ✅ | ✅ | ✅ | 🚧 | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| makes a call a pool thread answers | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ |
| reads input that arrives later | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ |
| is stopped and continued (job control) | ✅ | ✅ | ✅ | 🚧 | 🚧 | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| grows its window | ✅ | ✅ | ✅ | ✅ | ✅ | 🚧 | ✅ | ❔ | ✅ | ❔ | ❔ |
| maps a shared region (§13) | ✅ | ✅ | ✅ | 🔶 | ✅ | 🔶 | ✅ | ❔ | ✅ | ❔ | ❔ |
| compiles code at runtime (§22) | ✅ | ✅ | ✅ | ✅ | 🚧 | ✅ | ✅ | ❔ | 🚧 | ❔ | ❔ |
| serves calls on its offer | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ |

## Properties

| a run… | tree-walk interpreter | bytecode, one thread | bytecode, thread per task | Cranelift JIT | under a debugger | frozen and thawed | held by the embedder | from a warm snapshot | re-entered per frame | in the browser | with the wasm-JIT |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| under a debugger | 🚧 | 🚧 | ⛔ | ⛔ | · | 🚧 | 🚧 | ❔ | ❔ | ❔ | ❔ |
| frozen and thawed | 🚧 | 🚧 | 🚧 | 🚧 | · | · | 🚧 | ❔ | ❔ | ❔ | ❔ |
| held by the embedder | ⛔ | ✅ | ✅ | 🚧 | · | · | · | ❔ | 🚧 | ❔ | ❔ |
| from a warm snapshot | ❔ | ❔ | ❔ | ❔ | · | · | · | · | ❔ | ❔ | ❔ |
| re-entered per frame | 🚧 | 🚧 | 🚧 | 🚧 | · | · | · | · | · | ❔ | ❔ |
| in the browser | ❔ | ❔ | ❔ | ❔ | · | · | · | · | · | · | ❔ |
| with the wasm-JIT | ❔ | ❔ | ❔ | ❔ | · | · | · | · | · | · | · |

## Notes

**spawns threads**
- *Cranelift JIT* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module
- *frozen and thawed* 🚧 — bytecode, one thread: the bytecode tier's durable entry declines a module with threads or §14 children, and the pump hands it to the tree-walker (#2533)
- *frozen and thawed* 🔶 — Cranelift JIT: on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module

**runs fibers**
- *Cranelift JIT* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module
- *frozen and thawed* 🔶 — Cranelift JIT: on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module

**spawns a child domain (§14)**
- *Cranelift JIT* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector runs one vCPU over the powerbox, or every vCPU over none, so a child's join blocks (#2560)
- *frozen and thawed* 🚧 — bytecode, one thread: the bytecode tier's durable entry declines a module with threads or §14 children, and the pump hands it to the tree-walker (#2533)
- *frozen and thawed* 🔶 — Cranelift JIT: on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module

**forks itself (clone_caller)**
- *Cranelift JIT* 🚧 — answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md §9.3 plans them natively. The POSIX personality's own fork runs there (§9.5)
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector blocks on a fork (#2560)
- *under a debugger* 🚧 — bytecode, one thread: the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554)
- *frozen and thawed* 🚧 — bytecode, one thread: the bytecode tier's durable entry declines a module with threads or §14 children, and the pump hands it to the tree-walker (#2533)
- *frozen and thawed* 🚧 — Cranelift JIT: answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md §9.3 plans them natively. The POSIX personality's own fork runs there (§9.5)

**execs another image (exec_module)**
- *Cranelift JIT* 🚧 — declines exec_module, the op C's execve replaces its image with (#2555); the POSIX personality's OP_EXECVE runs there (FORK.md §9.4)
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector traps FiberFault on an exec (#2560)
- *under a debugger* 🚧 — bytecode, one thread: the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554)
- *frozen and thawed* 🚧 — tree-walk interpreter; bytecode, one thread: a durable domain refuses exec_module with -EINVAL (#1696)
- *frozen and thawed* 🚧 — Cranelift JIT: declines exec_module, the op C's execve replaces its image with (#2555); the POSIX personality's OP_EXECVE runs there (FORK.md §9.4)
- *re-entered per frame* 🚧 — bytecode, one thread: the frame that execs drops the reactor's window, so the next frame traps Malformed (#2660)

**makes a call the embedder answers**
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector traps CapFault on a host-answered call (#2560)
- *frozen and thawed* 🚧 — tree-walk interpreter; bytecode, one thread; Cranelift JIT: a durable run declines a host-answered call with CapFault before its embedder sees it: no caller in a durable run parks on a call (#2504)

**is stopped and continued (job control)**
- *Cranelift JIT* 🚧 — serves no job control, so a stopped process runs on (#1826)
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector runs a stopped process on (#2560)
- *under a debugger* 🚧 — bytecode, one thread: the bytecode debug scheduler runs a stopped process on: its pick never asks whether the process is stopped (#2615)
- *frozen and thawed* 🚧 — Cranelift JIT: serves no job control, so a stopped process runs on (#1826)

**grows its window**
- *frozen and thawed* 🚧 — bytecode, one thread: the bytecode tier's durable entry takes a flat window, so a thaw traps MemoryFault on a page the run mapped past its declared memory (#2628)

**maps a shared region (§13)**
- *Cranelift JIT* 🔶 — for a region over an OS shared-memory object (temen_run::new_shared_region), which the JIT maps by its descriptor; a software-only one (Host::grant_shared_region) has none, so its map answers -EINVAL there
- *frozen and thawed* 🔶 — Cranelift JIT: for a region over an OS shared-memory object (temen_run::new_shared_region), which the JIT maps by its descriptor; a software-only one (Host::grant_shared_region) has none, so its map answers -EINVAL there

**compiles code at runtime (§22)**
- *under a debugger* 🚧 — tree-walk interpreter: the tree-walk Inspector traps IndirectCallType at the installed unit's call.dyn (#2560)
- *re-entered per frame* 🚧 — bytecode, one thread: Reactor::call gives a frame no room for a §22 install, so the unit's call.dyn traps IndirectCallType; coop_call sizes it by the Jit grant (#2661)

**under a debugger**
- *tree-walk interpreter* 🚧 — spawns a child domain (§14): the tree-walk Inspector runs one vCPU over the powerbox, or every vCPU over none, so a child's join blocks (#2560)
- *tree-walk interpreter* 🚧 — forks itself (clone_caller): the tree-walk Inspector blocks on a fork (#2560)
- *tree-walk interpreter* 🚧 — execs another image (exec_module): the tree-walk Inspector traps FiberFault on an exec (#2560)
- *tree-walk interpreter* 🚧 — makes a call the embedder answers: the tree-walk Inspector traps CapFault on a host-answered call (#2560)
- *tree-walk interpreter* 🚧 — is stopped and continued (job control): the tree-walk Inspector runs a stopped process on (#2560)
- *tree-walk interpreter* 🚧 — compiles code at runtime (§22): the tree-walk Inspector traps IndirectCallType at the installed unit's call.dyn (#2560)
- *bytecode, one thread* 🚧 — forks itself (clone_caller); execs another image (exec_module): the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554)
- *bytecode, one thread* 🚧 — is stopped and continued (job control): the bytecode debug scheduler runs a stopped process on: its pick never asks whether the process is stopped (#2615)
- *bytecode, thread per task* ⛔ — the debug scheduler owns the schedule, every task on one thread, which is what makes a replay see the same interleaving (DEBUGGING.md); executor 2 runs its tasks at once
- *Cranelift JIT* ⛔ — the interpreter is the debug engine and the JIT the production engine, differential-checked against it (DEBUGGING.md)
- *frozen and thawed* 🚧 — tree-walk interpreter; bytecode, one thread: a debugger runs no durable run: its checkpoints refuse a durable host, and nothing freezes a run under it (#2658)
- *held by the embedder* 🚧 — forks itself (clone_caller); execs another image (exec_module): the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554)
- *held by the embedder* 🚧 — is stopped and continued (job control): the bytecode debug scheduler runs a stopped process on: its pick never asks whether the process is stopped (#2615)

**frozen and thawed**
- *tree-walk interpreter* 🚧 — execs another image (exec_module): a durable domain refuses exec_module with -EINVAL (#1696)
- *tree-walk interpreter* 🚧 — makes a call the embedder answers: a durable run declines a host-answered call with CapFault before its embedder sees it: no caller in a durable run parks on a call (#2504)
- *bytecode, one thread* 🚧 — spawns threads; spawns a child domain (§14); forks itself (clone_caller): the bytecode tier's durable entry declines a module with threads or §14 children, and the pump hands it to the tree-walker (#2533)
- *bytecode, one thread* 🚧 — execs another image (exec_module): a durable domain refuses exec_module with -EINVAL (#1696)
- *bytecode, one thread* 🚧 — makes a call the embedder answers: a durable run declines a host-answered call with CapFault before its embedder sees it: no caller in a durable run parks on a call (#2504)
- *bytecode, one thread* 🚧 — grows its window: the bytecode tier's durable entry takes a flat window, so a thaw traps MemoryFault on a page the run mapped past its declared memory (#2628)
- *bytecode, thread per task* 🚧 — executor 2 has no freeze across its threads yet; a durable run asked to run on it runs on the pump (#2541)
- *Cranelift JIT* 🔶 — spawns threads; runs fibers; spawns a child domain (§14): on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module
- *Cranelift JIT* 🚧 — forks itself (clone_caller): answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md §9.3 plans them natively. The POSIX personality's own fork runs there (§9.5)
- *Cranelift JIT* 🚧 — execs another image (exec_module): declines exec_module, the op C's execve replaces its image with (#2555); the POSIX personality's OP_EXECVE runs there (FORK.md §9.4)
- *Cranelift JIT* 🚧 — makes a call the embedder answers: a durable run declines a host-answered call with CapFault before its embedder sees it: no caller in a durable run parks on a call (#2504)
- *Cranelift JIT* 🚧 — is stopped and continued (job control): serves no job control, so a stopped process runs on (#1826)
- *Cranelift JIT* 🔶 — maps a shared region (§13): for a region over an OS shared-memory object (temen_run::new_shared_region), which the JIT maps by its descriptor; a software-only one (Host::grant_shared_region) has none, so its map answers -EINVAL there
- *held by the embedder* 🚧 — bytecode, one thread: a held run cannot be frozen: CoopRun takes no freeze (#2659)

**held by the embedder**
- *tree-walk interpreter* ⛔ — the tree-walker runs every run to its end: the oracle's only run an embedder holds is its debugger's, the Inspector
- *Cranelift JIT* 🚧 — an embedder holds and resumes a run only on the bytecode tier; whether a Cranelift run can be held, and at what cost, is #2561
- *re-entered per frame* 🚧 — execs another image (exec_module): the frame that execs drops the reactor's window, so the next frame traps Malformed (#2660)

**re-entered per frame**
- *tree-walk interpreter* 🚧 — only the pump's reactor runs an embedder's powerbox: the tree-walker's and the JIT's are temen_run::Session, which runs a single-threaded on-ramp module, and executor 2 has none (#2662)
- *bytecode, one thread* 🚧 — execs another image (exec_module): the frame that execs drops the reactor's window, so the next frame traps Malformed (#2660)
- *bytecode, one thread* 🚧 — compiles code at runtime (§22): Reactor::call gives a frame no room for a §22 install, so the unit's call.dyn traps IndirectCallType; coop_call sizes it by the Jit grant (#2661)
- *bytecode, thread per task* 🚧 — only the pump's reactor runs an embedder's powerbox: the tree-walker's and the JIT's are temen_run::Session, which runs a single-threaded on-ramp module, and executor 2 has none (#2662)
- *Cranelift JIT* 🚧 — only the pump's reactor runs an embedder's powerbox: the tree-walker's and the JIT's are temen_run::Session, which runs a single-threaded on-ramp module, and executor 2 has none (#2662)

