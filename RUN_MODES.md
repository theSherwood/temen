# Feature × run-mode matrix

**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity`. The classification lives in `crates/temen-parity/src/run_modes.rs`; this file is its human-readable view.

Can a program that uses a feature run in a given mode, and which modes combine (#2545)? A run is one **executor** (tree-walk, pump, parallel, Cranelift) plus any **properties** it adds. The columns are both kinds; the rows are guest features, then the properties again. What a cell asks depends on its row and column:

| row | column | the question |
|---|---|---|
| feature | executor | does the executor run a guest that uses the feature, to the oracle's answer? |
| feature | property | does the property hold for such a guest, on every executor that runs the property? |
| property | executor | does the executor run with the property? |
| property | property | do the two combine? |

**85 of 220 cells audited** (20 rows × 11 columns). A cell says anything but ❔ only where `crates/temen-parity/tests/run_mode_conformance.rs` runs the case and checks it, and every 🚧 cell cites the issue tracking its gap.

## Legend

- ✅ **Full** — holds
- ⛔ **Declines** — deliberately does not, and the note says why
- 🚧 **Not yet** — a real gap, and the note cites its issue
- 🔶 **Conditional** — holds where the note's condition does
- ❔ **Unaudited** — nobody has established this cell

## Columns

- **tree-walk** — the tree-walk interpreter, the oracle every other mode is held to
- **pump** — the bytecode tier's cooperative executor: every task in turn on one thread
- **parallel** — executor 2: a thread per task over the pump's rules
- **Cranelift** — the Cranelift JIT, run through temen-run
- **debugger** — under a debugger: the tree-walk Inspector (the DAP's default engine) or the bytecode debug scheduler (ScheduledDebugRun, its bytecode engine)
- **durable** — the run can freeze and thaw
- **pumped** — the embedder holds the run and pumps it (CoopRun, CoopSession), told of each call it answers and of the run's idle point
- **warm** — the run starts from a warm snapshot taken after its setup
- **reactor** — the entry is re-entered a frame at a time over kept state
- **browser** — in the wasm build, on a page (the playground)
- **emitted** — hot code runs as emitted wasm (the browser's emitted tier)

## Features

| feature | tree-walk | pump | parallel | Cranelift | debugger | durable | pumped | warm | reactor | browser | emitted |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| threads | ✅ | ✅ | ✅ | 🔶 | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| fibers | ✅ | ✅ | ✅ | 🔶 | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| §14 children | ✅ | ✅ | ✅ | 🔶 | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| fork (clone_caller) | ✅ | ✅ | ✅ | 🚧 | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| exec (exec_module) | ✅ | ✅ | ✅ | 🚧 | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| host-answered calls | ✅ | ✅ | ✅ | ✅ | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| offloaded calls | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| stdin fed later | ❔ | ✅ | ✅ | ❔ | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| job-control signals | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ |
| a growing window | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| §13 shared regions | ✅ | ✅ | ✅ | 🔶 | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| §22 guest JIT | ✅ | ✅ | ✅ | ✅ | 🚧 | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| serving | ✅ | ✅ | ✅ | ✅ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |

## Properties

| property | tree-walk | pump | parallel | Cranelift | debugger | durable | pumped | warm | reactor | browser | emitted |
|----|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
| debugger | ✅ | ✅ | ⛔ | ⛔ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| durable | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ | ❔ |
| pumped | ❔ | ✅ | ✅ | ❔ | ✅ | ❔ | ✅ | ❔ | ❔ | ❔ | ❔ |
| warm | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ | ❔ | ❔ |
| reactor | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ | ❔ |
| browser | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ | ❔ |
| emitted | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ❔ | ✅ |

## Notes

**threads**
- *Cranelift* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module

**fibers**
- *Cranelift* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module

**§14 children**
- *Cranelift* 🔶 — on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); elsewhere the JIT declines the module
- *debugger* 🚧 — the tree-walk Inspector runs one vCPU over the powerbox, or every vCPU over none, so a child's join blocks (#2560)

**fork (clone_caller)**
- *Cranelift* 🚧 — answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md §9.3 plans them natively. The POSIX personality's own fork runs there (§9.5)
- *debugger* 🚧 — the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554); the tree-walk Inspector blocks on a fork and traps on an exec (#2560)

**exec (exec_module)**
- *Cranelift* 🚧 — declines exec_module, the op C's execve replaces its image with (#2555); the POSIX personality's OP_EXECVE runs there (FORK.md §9.4)
- *debugger* 🚧 — the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait and pipes (#2554); the tree-walk Inspector blocks on a fork and traps on an exec (#2560)

**host-answered calls**
- *debugger* 🚧 — the tree-walk Inspector traps CapFault on a host-answered call (#2560)

**stdin fed later**
- *debugger* 🚧 — the tree-walk Inspector has no way to feed a read (#2560)

**§13 shared regions**
- *Cranelift* 🔶 — for a region over an OS shared-memory object (temen_run::new_shared_region), which the JIT maps by its descriptor; a software-only one (Host::grant_shared_region) has none, so its map answers -EINVAL there

**§22 guest JIT**
- *debugger* 🚧 — the tree-walk Inspector traps IndirectCallType at the installed unit's call.dyn (#2560)

**debugger**
- *parallel* ⛔ — the debug scheduler owns the schedule, every task on one thread, which is what makes a replay see the same interleaving (DEBUGGING.md); executor 2 runs its tasks at once
- *Cranelift* ⛔ — the interpreter is the debug engine and the JIT the production engine, differential-checked against it (DEBUGGING.md)

