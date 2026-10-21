//! # The feature × run-mode matrix
//!
//! The third matrix (#2545), beside the op × backend one ([`crate::parity`]) and the capability ×
//! axis frontier ([`crate::frontier`]). It answers the question an embedder asks before it picks how
//! to run a program: **can a program that uses this run that way, and do these two ways combine?**
//!
//! A run is one **executor** — what steps the guest's instructions and schedules its tasks — plus
//! any number of **properties** it adds: a debugger, durability, an embedder that pumps it, a warm
//! start, a reactor's re-entry, the browser, the browser's emitted tier. The columns are both kinds,
//! [`Mode`]. The rows are what a guest program uses, [`Feature`], and the properties again, so the
//! same table says which properties combine with which executors and with each other.
//!
//! What a cell asks depends on its row and column:
//!
//! | row | column | the question |
//! |---|---|---|
//! | feature | executor | does the executor run a guest that uses the feature, to the oracle's answer? |
//! | feature | property | does the property hold for such a guest, on every executor that runs the property? |
//! | property | executor | does the executor run with the property? |
//! | property | property | do the two combine? |
//!
//! ## The same three properties as the other two
//!
//! 1. **Exhaustive.** [`feature_cells`] and [`property_cells`] match with no wildcard arm, so a new
//!    feature or property fails to compile until it is classified on every column.
//! 2. **Driven.** A cell says anything but [`Status::Unaudited`] only where
//!    `tests/run_mode_conformance.rs` runs the case and checks the claim, and a [`Status::NotYet`]
//!    cell must cite the issue tracking the gap. The test fails when a claim and the run disagree,
//!    and when a claimed cell has nothing driving it.
//! 3. **One vocabulary.** The cells are the shared [`Cell`]/[`Status`], so the renderer, the JSON
//!    and the playground's parity page read this matrix as they read the other two.
//!
//! ## Why the rows and columns are these
//!
//! The executors are the four a native embedder can pick. Neither of the DAP's debug engines is a
//! fifth: the tree-walk `Inspector` debugs the tree-walker, and the bytecode debug scheduler steps
//! the pump's task set one task at a time, so each is the `debugger` property on its executor. The
//! browser's executors are the pump and executor 2 again, built for wasm32, so the browser is a
//! property rather than two more executors.
//!
//! The features are the ones whose support differs between modes: the concurrency, process, host
//! and memory features where an executor or a property has a decline or a gap. Plain computation
//! is the op matrix's question, and a capability's own behaviour is the frontier's.

use crate::{Cell, Status};

/// What steps a run's instructions and schedules its tasks: exactly one per run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Executor {
    /// The tree-walk interpreter (`temen_interp::run_with_host`): the oracle, whose answer every
    /// other mode is held to (INVARIANTS #9).
    TreeWalk,
    /// The bytecode tier's cooperative executor, every task in turn on one thread
    /// (`bytecode::compile_and_run_with_host`, `CoopRun::new`).
    Pump,
    /// Executor 2: a thread per task over the pump's rules
    /// (`bytecode::compile_and_run_capture_over_parallel_with_host`, `CoopRun::new_parallel`).
    Parallel,
    /// The Cranelift JIT (`temen-jit`, run through `temen_run`).
    Cranelift,
}

impl Executor {
    pub const ALL: [Executor; 4] = [
        Executor::TreeWalk,
        Executor::Pump,
        Executor::Parallel,
        Executor::Cranelift,
    ];
}

/// What a run adds to its executor: any number per run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Property {
    /// Under a debugger: the tree-walk `Inspector`, the DAP's default engine, or the bytecode debug
    /// scheduler (`ScheduledDebugRun`, its `bytecode` engine), which steps the pump's task set one
    /// task at a time so a replay sees the same schedule.
    Debugger,
    /// The run can freeze and thaw (DURABILITY.md).
    Durable,
    /// The embedder holds the run and pumps it (`CoopRun`, `temen_run::CoopSession`): each pump
    /// returns at a call the embedder answers, when the run idles on it, and at the end.
    Pumped,
    /// The run starts from a warm snapshot taken after its setup.
    Warm,
    /// The run is a reactor: its entry is re-entered, a frame at a time, over state kept between.
    Reactor,
    /// The run is in the wasm build, on a page (the playground).
    Browser,
    /// Hot code runs as emitted wasm (the browser's emitted tier).
    Emitted,
}

impl Property {
    pub const ALL: [Property; 7] = [
        Property::Debugger,
        Property::Durable,
        Property::Pumped,
        Property::Warm,
        Property::Reactor,
        Property::Browser,
        Property::Emitted,
    ];
}

/// What a guest program uses.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Feature {
    /// `thread.spawn` / `thread.join`.
    Threads,
    /// `cont.new` / `cont.resume` / `suspend` (§12 fibers).
    Fibers,
    /// A §14 detached child: `instantiate_rec` (Instantiator op 17), then a join.
    Children,
    /// Fork: the self-namespace `clone_caller` / `reap` (FORK.md §9).
    Fork,
    /// Exec: the self-namespace `exec_module`, which replaces the running image (FORK.md §8.6).
    Exec,
    /// A capability call the embedder answers later (`OffloadOutcome::Host`).
    HostAnswered,
    /// A capability call a pool thread answers (`OffloadOutcome::Offload`).
    Offloaded,
    /// A blocking-stdin `read` that waits for input the embedder feeds later.
    StdinLater,
    /// Job-control signals: a process stops and continues.
    Signals,
    /// A window that grows past its initial size.
    GrowingWindow,
    /// A §13 shared region mapped into the window.
    SharedRegion,
    /// §22 guest JIT: a unit the guest compiles, installs and calls.
    GuestJit,
    /// Serving: `svc.poll` / `svc.wait` on a live offer.
    Serving,
}

impl Feature {
    pub const ALL: [Feature; 13] = [
        Feature::Threads,
        Feature::Fibers,
        Feature::Children,
        Feature::Fork,
        Feature::Exec,
        Feature::HostAnswered,
        Feature::Offloaded,
        Feature::StdinLater,
        Feature::Signals,
        Feature::GrowingWindow,
        Feature::SharedRegion,
        Feature::GuestJit,
        Feature::Serving,
    ];

    /// The row label.
    pub fn name(self) -> &'static str {
        match self {
            Feature::Threads => "threads",
            Feature::Fibers => "fibers",
            Feature::Children => "§14 children",
            Feature::Fork => "fork (clone_caller)",
            Feature::Exec => "exec (exec_module)",
            Feature::HostAnswered => "host-answered calls",
            Feature::Offloaded => "offloaded calls",
            Feature::StdinLater => "stdin fed later",
            Feature::Signals => "job-control signals",
            Feature::GrowingWindow => "a growing window",
            Feature::SharedRegion => "§13 shared regions",
            Feature::GuestJit => "§22 guest JIT",
            Feature::Serving => "serving",
        }
    }
}

/// A column: an executor or a property.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Executor(Executor),
    Property(Property),
}

impl Mode {
    /// Every column, in matrix order: the executors, then the properties.
    pub const ALL: [Mode; 11] = [
        Mode::Executor(Executor::TreeWalk),
        Mode::Executor(Executor::Pump),
        Mode::Executor(Executor::Parallel),
        Mode::Executor(Executor::Cranelift),
        Mode::Property(Property::Debugger),
        Mode::Property(Property::Durable),
        Mode::Property(Property::Pumped),
        Mode::Property(Property::Warm),
        Mode::Property(Property::Reactor),
        Mode::Property(Property::Browser),
        Mode::Property(Property::Emitted),
    ];

    /// The column header.
    pub fn short(self) -> &'static str {
        match self {
            Mode::Executor(Executor::TreeWalk) => "tree-walk",
            Mode::Executor(Executor::Pump) => "pump",
            Mode::Executor(Executor::Parallel) => "parallel",
            Mode::Executor(Executor::Cranelift) => "Cranelift",
            Mode::Property(p) => p.short(),
        }
    }

    /// What the column is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Mode::Executor(Executor::TreeWalk) => {
                "the tree-walk interpreter, the oracle every other mode is held to"
            }
            Mode::Executor(Executor::Pump) => {
                "the bytecode tier's cooperative executor: every task in turn on one thread"
            }
            Mode::Executor(Executor::Parallel) => {
                "executor 2: a thread per task over the pump's rules"
            }
            Mode::Executor(Executor::Cranelift) => "the Cranelift JIT, run through temen-run",
            Mode::Property(p) => p.question(),
        }
    }

    /// The column's index in a row's cells.
    pub fn index(self) -> usize {
        Mode::ALL
            .iter()
            .position(|&m| m == self)
            .expect("every mode is a column")
    }
}

impl Property {
    /// The row label and column header.
    pub fn short(self) -> &'static str {
        match self {
            Property::Debugger => "debugger",
            Property::Durable => "durable",
            Property::Pumped => "pumped",
            Property::Warm => "warm",
            Property::Reactor => "reactor",
            Property::Browser => "browser",
            Property::Emitted => "emitted",
        }
    }

    /// What the property is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Property::Debugger => {
                "under a debugger: the tree-walk Inspector (the DAP's default engine) or the \
                 bytecode debug scheduler (ScheduledDebugRun, its bytecode engine)"
            }
            Property::Durable => "the run can freeze and thaw",
            Property::Pumped => {
                "the embedder holds the run and pumps it (CoopRun, CoopSession), told of each call \
                 it answers and of the run's idle point"
            }
            Property::Warm => "the run starts from a warm snapshot taken after its setup",
            Property::Reactor => "the entry is re-entered a frame at a time over kept state",
            Property::Browser => "in the wasm build, on a page (the playground)",
            Property::Emitted => "hot code runs as emitted wasm (the browser's emitted tier)",
        }
    }
}

/// A row: a feature, or a property.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Row {
    Feature(Feature),
    Property(Property),
}

impl Row {
    /// Every row, in matrix order: the features, then the properties.
    pub fn all() -> impl Iterator<Item = Row> {
        Feature::ALL
            .into_iter()
            .map(Row::Feature)
            .chain(Property::ALL.into_iter().map(Row::Property))
    }

    /// The row label.
    pub fn name(self) -> &'static str {
        match self {
            Row::Feature(f) => f.name(),
            Row::Property(p) => p.short(),
        }
    }

    /// Which half of the table the row is in.
    pub fn group(self) -> &'static str {
        match self {
            Row::Feature(_) => "feature",
            Row::Property(_) => "property",
        }
    }

    /// The row's cells, in [`Mode::ALL`] order.
    pub fn cells(self) -> [Cell; 11] {
        match self {
            Row::Feature(f) => feature_cells(f),
            Row::Property(p) => property_cells(p),
        }
    }
}

/// `Full`, no note.
const F: Cell = Cell {
    status: Status::Full,
    note: "",
};
/// Not driven yet. Distinct from `NotYet`, which is a known gap with an issue.
const U: Cell = Cell {
    status: Status::Unaudited,
    note: "",
};

const fn declines(note: &'static str) -> Cell {
    Cell {
        status: Status::Declines,
        note,
    }
}

const fn not_yet(note: &'static str) -> Cell {
    Cell {
        status: Status::NotYet,
        note,
    }
}

const fn conditional(note: &'static str) -> Cell {
    Cell {
        status: Status::Conditional,
        note,
    }
}

/// Why neither debugger runs fork or exec.
const PROCESS_SEAMS: &str =
    "the bytecode debug scheduler declines every process seam: fork, exec, \
                             spawn, wait and pipes (#2554); the tree-walk Inspector blocks on a \
                             fork and traps on an exec (#2560)";

/// The `fiber_rt` targets, where the Cranelift JIT has its stack-switch substrate.
const FIBER_RT: &str = "on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); \
                        elsewhere the JIT declines the module";

/// **The feature rows.** Exhaustive over [`Feature`], no wildcard arm. Cell order matches
/// [`Mode::ALL`]: tree-walk, pump, parallel, Cranelift, then debugger, durable, pumped, warm,
/// reactor, browser, emitted.
pub fn feature_cells(f: Feature) -> [Cell; 11] {
    match f {
        Feature::Threads => [F, F, F, conditional(FIBER_RT), F, U, F, U, U, U, U],
        Feature::Fibers => [F, F, F, conditional(FIBER_RT), F, U, F, U, U, U, U],
        Feature::Children => [
            F,
            F,
            F,
            conditional(FIBER_RT),
            not_yet(
                "the tree-walk Inspector runs one vCPU over the powerbox, or every vCPU over none, \
                 so a child's join blocks (#2560)",
            ),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::Fork => [
            F,
            F,
            F,
            not_yet(
                "answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md \
                 §9.3 plans them natively. The POSIX personality's own fork runs there (§9.5)",
            ),
            not_yet(PROCESS_SEAMS),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::Exec => [
            F,
            F,
            F,
            not_yet(
                "declines exec_module, the op C's execve replaces its image with (#2555); the \
                 POSIX personality's OP_EXECVE runs there (FORK.md §9.4)",
            ),
            not_yet(PROCESS_SEAMS),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::HostAnswered => [
            F,
            F,
            F,
            F,
            not_yet("the tree-walk Inspector traps CapFault on a host-answered call (#2560)"),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::Offloaded => [F, F, F, F, F, U, F, U, U, U, U],
        Feature::StdinLater => [
            U,
            F,
            F,
            U,
            not_yet("the tree-walk Inspector has no way to feed a read (#2560)"),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::Signals => [U, U, U, U, U, U, U, U, U, U, U],
        Feature::GrowingWindow => [F, F, F, F, F, U, F, U, U, U, U],
        Feature::SharedRegion => [
            F,
            F,
            F,
            conditional(
                "for a region over an OS shared-memory object (temen_run::new_shared_region), \
                 which the JIT maps by its descriptor; a software-only one \
                 (Host::grant_shared_region) has none, so its map answers -EINVAL there",
            ),
            F,
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::GuestJit => [
            F,
            F,
            F,
            F,
            not_yet(
                "the tree-walk Inspector traps IndirectCallType at the installed unit's call.dyn \
                 (#2560)",
            ),
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Feature::Serving => [F, F, F, F, F, U, F, U, U, U, U],
    }
}

/// **The property rows.** Exhaustive over [`Property`], no wildcard arm, in [`Mode::ALL`] order. A
/// property's own column is `Full`, and the property × property half is symmetric: the
/// conformance test holds both.
pub fn property_cells(p: Property) -> [Cell; 11] {
    match p {
        Property::Debugger => [
            F,
            F,
            declines(
                "the debug scheduler owns the schedule, every task on one thread, which is what \
                 makes a replay see the same interleaving (DEBUGGING.md); executor 2 runs its \
                 tasks at once",
            ),
            declines(
                "the interpreter is the debug engine and the JIT the production engine, \
                 differential-checked against it (DEBUGGING.md)",
            ),
            F,
            U,
            F,
            U,
            U,
            U,
            U,
        ],
        Property::Durable => [U, U, U, U, U, F, U, U, U, U, U],
        Property::Pumped => [U, F, F, U, F, U, F, U, U, U, U],
        Property::Warm => [U, U, U, U, U, U, U, F, U, U, U],
        Property::Reactor => [U, U, U, U, U, U, U, U, F, U, U],
        Property::Browser => [U, U, U, U, U, U, U, U, U, F, U],
        Property::Emitted => [U, U, U, U, U, U, U, U, U, U, F],
    }
}
