//! # The feature × run-mode matrix
//!
//! The third matrix (#2545), beside the op × backend one ([`crate::parity`]) and the capability ×
//! axis frontier ([`crate::frontier`]). It answers the question an embedder asks before it picks how
//! to run a program: **can a program that does this run that way, and do these ways of running
//! combine?**
//!
//! A run is one **executor** — what steps the guest's instructions and schedules its tasks — plus
//! any number of **properties** it adds: a debugger, a freeze and thaw, an embedder that holds it, a
//! warm start, a reactor's re-entry, the browser, the wasm-JIT. The columns are both kinds, [`Mode`].
//! The rows are what a program does, [`Feature`], and the properties again, so the same table says
//! which executors each property runs on and which pairs of properties combine.
//!
//! ## Cells come from runs
//!
//! The unit of evidence is a **run**: one feature's probe on one [`Config`], an executor and a set of
//! properties that `tests/run_mode_conformance.rs` has a runner for ([`RUNNERS`]). A run gives the
//! oracle's answer unless [`exception`] says otherwise, and the conformance test holds every run to
//! that: a run that differs with no exception fails it, and so does an exception whose run now
//! gives the oracle's answer. Nothing else is written by hand. A cell is the fold of the runs behind
//! it:
//!
//! | row | column | the runs behind the cell |
//! |---|---|---|
//! | feature | executor | the feature's probe on the executor alone |
//! | feature | property | the probe on each executor that has a runner with the property |
//! | property | executor | every probe on the executor's runner with the property |
//! | property | property | every probe, on each executor that has a runner with both |
//!
//! On each executor a cell uses the runner that adds its properties and the fewest others
//! ([`runner_for`]). The bytecode debug scheduler is held by its embedder as well as debugging, so it
//! is the `debugger` run on the pump, while `held` there is `CoopRun`'s.
//!
//! The fold ([`Summary`]) is ✅ when every run gives the oracle's answer, 🚧 when any run is a known
//! gap, ⛔ when every run declines by design, and 🔶 for any other mix. Its note lists the runs that
//! are not ✅, so a 🚧 always names what failed and where. Where no runner exists, a cell is ❔
//! unless [`config_exception`] says why there is none.
//!
//! The property × property half is a triangle: each pair appears once, in the earlier property's
//! row, and a property is not paired with itself.
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
    /// The bytecode tier's cooperative executor, the pump: every task takes turns on one thread
    /// (`bytecode::compile_and_run_with_host`).
    Pump,
    /// Executor 2: every task on a thread of its own, under the pump's rules
    /// (`bytecode::compile_and_run_capture_over_parallel_with_host`).
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

    /// The column header.
    pub fn label(self) -> &'static str {
        match self {
            Executor::TreeWalk => "tree-walk interpreter",
            Executor::Pump => "bytecode, one thread",
            Executor::Parallel => "bytecode, thread per task",
            Executor::Cranelift => "Cranelift JIT",
        }
    }

    /// What the column is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Executor::TreeWalk => "the tree-walk interpreter, the oracle every other mode is held to",
            Executor::Pump => {
                "the bytecode tier's cooperative executor (the pump): every task takes turns on one \
                 thread"
            }
            Executor::Parallel => {
                "the bytecode tier's parallel executor (executor 2): every task on a thread of its \
                 own, under the pump's rules"
            }
            Executor::Cranelift => "native code from the Cranelift JIT, run through temen-run",
        }
    }
}

/// What a run adds to its executor: any number per run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Property {
    /// A debugger steps the run: the tree-walk `Inspector`, the DAP's default engine, or the bytecode
    /// debug scheduler (`ScheduledDebugRun`, its `bytecode` engine), which steps the pump's task set
    /// one task at a time so a replay sees the same schedule.
    Debugger,
    /// The run is frozen part-way and thawed (DURABILITY.md).
    Durable,
    /// The embedder holds the run and resumes it (`CoopRun`, `temen_run::CoopSession`, the debug
    /// scheduler): the run hands control back at each call the embedder answers, when it can only
    /// wait on the embedder, and at its end (#1414 D4).
    Held,
    /// The run starts from a warm snapshot taken after its setup.
    Warm,
    /// The run is a reactor: its entry is re-entered, a frame at a time, over state kept between.
    Reactor,
    /// The run is in the wasm build, on a page (the playground).
    Browser,
    /// Hot code runs as wasm the wasm-JIT emits (the browser's emitted tier).
    Emitted,
}

impl Property {
    pub const ALL: [Property; 7] = [
        Property::Debugger,
        Property::Durable,
        Property::Held,
        Property::Warm,
        Property::Reactor,
        Property::Browser,
        Property::Emitted,
    ];

    /// The row label and column header.
    pub fn label(self) -> &'static str {
        match self {
            Property::Debugger => "under a debugger",
            Property::Durable => "frozen and thawed",
            Property::Held => "held by the embedder",
            Property::Warm => "from a warm snapshot",
            Property::Reactor => "re-entered per frame",
            Property::Browser => "in the browser",
            Property::Emitted => "with the wasm-JIT",
        }
    }

    /// What the property is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Property::Debugger => {
                "a debugger steps the run: the tree-walk Inspector (the DAP's default engine) on the \
                 tree-walker, the bytecode debug scheduler (ScheduledDebugRun) on the pump"
            }
            Property::Durable => {
                "the run is frozen part-way and thawed, and finishes as if it had never stopped"
            }
            Property::Held => {
                "the embedder holds the run and resumes it (CoopRun, CoopSession, the debug \
                 scheduler): the run hands control back at each call the embedder answers, when it \
                 can only wait on the embedder, and at its end"
            }
            Property::Warm => "the run starts from a snapshot of its window taken after its setup",
            Property::Reactor => {
                "a reactor: the entry is called again, a frame at a time, over state kept between \
                 frames"
            }
            Property::Browser => "the wasm32 build, on a web page (the playground)",
            Property::Emitted => {
                "hot code is compiled to wasm by the wasm-JIT and run by the browser's engine (the \
                 browser's emitted tier)"
            }
        }
    }

    fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// What a guest program does.
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
    /// A stdin `read` made before its input has arrived, which waits for it.
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

    /// The row label: what a program that uses the feature does.
    pub fn label(self) -> &'static str {
        match self {
            Feature::Threads => "spawns threads",
            Feature::Fibers => "runs fibers",
            Feature::Children => "spawns a child domain (§14)",
            Feature::Fork => "forks itself (clone_caller)",
            Feature::Exec => "execs another image (exec_module)",
            Feature::HostAnswered => "makes a call the embedder answers",
            Feature::Offloaded => "makes a call a pool thread answers",
            Feature::StdinLater => "reads input that arrives later",
            Feature::Signals => "is stopped and continued (job control)",
            Feature::GrowingWindow => "grows its window",
            Feature::SharedRegion => "maps a shared region (§13)",
            Feature::GuestJit => "compiles code at runtime (§22)",
            Feature::Serving => "serves calls on its offer",
        }
    }

    /// What the feature's probe does, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Feature::Threads => "spawns two threads (thread.spawn) and joins them",
            Feature::Fibers => "runs a generator fiber (cont.new, cont.resume, suspend) to its end",
            Feature::Children => {
                "spawns a detached §14 child from a spawn record (Instantiator op 17) and joins it"
            }
            Feature::Fork => {
                "a served call forks its caller (clone_caller), and the caller reaps the twin (FORK.md \
                 §9)"
            }
            Feature::Exec => "replaces its image with another module (exec_module, FORK.md §8.6)",
            Feature::HostAnswered => {
                "makes capability calls the embedder answers later, from outside the run \
                 (OffloadOutcome::Host)"
            }
            Feature::Offloaded => {
                "makes capability calls a runtime pool thread answers (OffloadOutcome::Offload)"
            }
            Feature::StdinLater => {
                "reads stdin before its input has arrived, and gets it when the embedder supplies it"
            }
            Feature::Signals => "is stopped by job control and continued later",
            Feature::GrowingWindow => {
                "maps pages past its declared window through its AddressSpace and uses them"
            }
            Feature::SharedRegion => {
                "maps a §13 shared region at two offsets and sees a store through one in the other"
            }
            Feature::GuestJit => {
                "compiles a §22 unit, installs it and calls it through call.dyn, and invokes it"
            }
            Feature::Serving => "serves the calls queued on its offer (svc.poll)",
        }
    }
}

/// The features no probe drives yet. Every run of one is ❔, and its row says nothing else.
pub const UNPROBED: &[Feature] = &[Feature::Signals];

/// A set of properties.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Props(u8);

impl Props {
    /// No properties: the executor alone.
    pub const NONE: Props = Props(0);

    pub const fn of(ps: &[Property]) -> Props {
        let mut bits = 0;
        let mut i = 0;
        while i < ps.len() {
            bits |= 1 << ps[i] as u8;
            i += 1;
        }
        Props(bits)
    }

    pub fn has(self, p: Property) -> bool {
        self.0 & p.bit() != 0
    }

    /// Whether every property of `other` is in `self`.
    pub fn contains(self, other: Props) -> bool {
        self.0 & other.0 == other.0
    }

    pub fn len(self) -> u32 {
        self.0.count_ones()
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Every set of properties, for checks that must cover them all.
    pub fn every() -> impl Iterator<Item = Props> {
        (0..1u8 << Property::ALL.len()).map(Props)
    }
}

/// A way to run a program: one executor, and the properties it adds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Config {
    pub executor: Executor,
    pub props: Props,
}

impl Config {
    pub const fn new(executor: Executor, props: &[Property]) -> Config {
        Config {
            executor,
            props: Props::of(props),
        }
    }

    /// The executor's label, then each property's.
    pub fn label(self) -> String {
        let mut s = self.executor.label().to_string();
        for p in Property::ALL.into_iter().filter(|&p| self.props.has(p)) {
            s.push_str(", ");
            s.push_str(p.label());
        }
        s
    }
}

/// A runner `tests/run_mode_conformance.rs` has: the configuration it runs, and what runs it.
#[derive(Clone, Copy, Debug)]
pub struct Runner {
    pub config: Config,
    pub what: &'static str,
}

/// **The runners**, the matrix's evidence: every configuration the conformance test runs each probe
/// on. The test holds its own runners to this list exactly. A configuration missing here is ❔
/// wherever a cell needs it, unless [`config_exception`] says why it does not exist.
pub const RUNNERS: [Runner; 8] = {
    use Executor::*;
    use Property::*;
    [
        Runner {
            config: Config::new(TreeWalk, &[]),
            what: "temen_interp::run_with_host",
        },
        Runner {
            config: Config::new(Pump, &[]),
            what: "bytecode::compile_and_run_with_host",
        },
        Runner {
            config: Config::new(Parallel, &[]),
            what: "bytecode::compile_and_run_capture_over_parallel_with_host",
        },
        Runner {
            config: Config::new(Cranelift, &[]),
            what: "temen_run::jit_cap_run",
        },
        Runner {
            config: Config::new(TreeWalk, &[Debugger]),
            what: "the tree-walk Inspector, the DAP's default engine",
        },
        Runner {
            config: Config::new(Pump, &[Debugger, Held]),
            what: "the bytecode debug scheduler (ScheduledDebugRun), the DAP's bytecode engine",
        },
        Runner {
            config: Config::new(Pump, &[Held]),
            what: "CoopRun on the pump",
        },
        Runner {
            config: Config::new(Parallel, &[Held]),
            what: "CoopRun on executor 2",
        },
    ]
};

/// The runner a cell uses on `e` for the properties `need`: the one that adds them and the fewest
/// others. The conformance test checks that no two runners tie.
pub fn runner_for(e: Executor, need: Props) -> Option<Config> {
    RUNNERS
        .iter()
        .map(|r| r.config)
        .filter(|c| c.executor == e && c.props.contains(need))
        .min_by_key(|c| c.props.len())
}

const FULL: Cell = Cell {
    status: Status::Full,
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

/// The `fiber_rt` targets, where the Cranelift JIT has its stack-switch substrate.
const FIBER_RT: &str = "on the fiber_rt targets (unix x86-64, unix aarch64, Windows x86-64); \
                        elsewhere the JIT declines the module";

/// **What a run does that is not the oracle's answer**: ⛔ by design, 🚧 a gap with its issue, or 🔶
/// with the condition it holds under. `None` means the run gives the oracle's answer, and the
/// conformance test holds every run of every runner to what this says.
///
/// A rule may cover configurations no runner has yet. It applies as soon as one does, and the test
/// then checks it.
pub fn exception(f: Feature, c: Config) -> Option<Cell> {
    use Executor::*;
    use Feature::*;
    let debugged = c.props.has(Property::Debugger);
    Some(match (f, c.executor) {
        // The Cranelift JIT, whatever the run adds.
        (Threads | Fibers | Children, Cranelift) => conditional(FIBER_RT),
        (Fork, Cranelift) => not_yet(
            "answers clone_caller and reap -EINVAL where the oracle forks (#2553); FORK.md §9.3 \
             plans them natively. The POSIX personality's own fork runs there (§9.5)",
        ),
        (Exec, Cranelift) => not_yet(
            "declines exec_module, the op C's execve replaces its image with (#2555); the POSIX \
             personality's OP_EXECVE runs there (FORK.md §9.4)",
        ),
        (SharedRegion, Cranelift) => conditional(
            "for a region over an OS shared-memory object (temen_run::new_shared_region), which \
             the JIT maps by its descriptor; a software-only one (Host::grant_shared_region) has \
             none, so its map answers -EINVAL there",
        ),
        // The tree-walk Inspector runs less than the tree-walker it debugs (#2560).
        (Children, TreeWalk) if debugged => not_yet(
            "the tree-walk Inspector runs one vCPU over the powerbox, or every vCPU over none, so \
             a child's join blocks (#2560)",
        ),
        (Fork, TreeWalk) if debugged => not_yet("the tree-walk Inspector blocks on a fork (#2560)"),
        (Exec, TreeWalk) if debugged => {
            not_yet("the tree-walk Inspector traps FiberFault on an exec (#2560)")
        }
        (HostAnswered, TreeWalk) if debugged => {
            not_yet("the tree-walk Inspector traps CapFault on a host-answered call (#2560)")
        }
        (GuestJit, TreeWalk) if debugged => not_yet(
            "the tree-walk Inspector traps IndirectCallType at the installed unit's call.dyn \
             (#2560)",
        ),
        // The bytecode debug scheduler.
        (Fork | Exec, Pump) if debugged => not_yet(
            "the bytecode debug scheduler declines every process seam: fork, exec, spawn, wait \
             and pipes (#2554)",
        ),
        _ => return None,
    })
}

/// **Why no run exists** for the properties `props` on `e`, where that is known: ⛔ by design or 🚧
/// with its issue. Consulted only where no runner covers the combination; a runner for one this
/// names makes the entry stale, which the conformance test reports.
pub fn config_exception(e: Executor, props: Props) -> Option<Cell> {
    use Executor::*;
    use Property::*;
    if props.has(Debugger) {
        match e {
            Parallel => {
                return Some(declines(
                    "the debug scheduler owns the schedule, every task on one thread, which is \
                     what makes a replay see the same interleaving (DEBUGGING.md); executor 2 runs \
                     its tasks at once",
                ))
            }
            Cranelift => {
                return Some(declines(
                    "the interpreter is the debug engine and the JIT the production engine, \
                     differential-checked against it (DEBUGGING.md)",
                ))
            }
            TreeWalk if props.has(Held) => {
                return Some(not_yet(
                    "the tree-walk Inspector hands its embedder no call to answer and no read to \
                     feed (#2560)",
                ))
            }
            TreeWalk | Pump => {}
        }
    }
    None
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
        Mode::Property(Property::Held),
        Mode::Property(Property::Warm),
        Mode::Property(Property::Reactor),
        Mode::Property(Property::Browser),
        Mode::Property(Property::Emitted),
    ];

    /// The column header.
    pub fn label(self) -> &'static str {
        match self {
            Mode::Executor(e) => e.label(),
            Mode::Property(p) => p.label(),
        }
    }

    /// What the column is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Mode::Executor(e) => e.question(),
            Mode::Property(p) => p.question(),
        }
    }

    /// Which kind of column it is.
    pub fn kind(self) -> &'static str {
        match self {
            Mode::Executor(_) => "executor",
            Mode::Property(_) => "property",
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
    pub fn label(self) -> &'static str {
        match self {
            Row::Feature(f) => f.label(),
            Row::Property(p) => p.label(),
        }
    }

    /// What the row is, for the legend.
    pub fn question(self) -> &'static str {
        match self {
            Row::Feature(f) => f.question(),
            Row::Property(p) => p.question(),
        }
    }

    /// Which half of the table the row is in.
    pub fn group(self) -> &'static str {
        match self {
            Row::Feature(_) => "feature",
            Row::Property(_) => "property",
        }
    }

    /// The row's cells, in [`Mode::ALL`] order. `None` where a property's row meets itself or an
    /// earlier property: that pair is in the earlier property's row.
    pub fn cells(self) -> Vec<Option<Summary>> {
        Mode::ALL
            .into_iter()
            .map(|m| match (self, m) {
                (Row::Feature(f), m) => Some(feature_cell(f, m)),
                (Row::Property(p), Mode::Executor(e)) => {
                    Some(on_executor(e, Props::of(&[p]), |c| {
                        probed()
                            .map(|f| (f.label().to_string(), run(f, c)))
                            .collect()
                    }))
                }
                (Row::Property(p), Mode::Property(q)) if q as u8 <= p as u8 => None,
                (Row::Property(p), Mode::Property(q)) => Some(pair_cell(p, q)),
            })
            .collect()
    }
}

/// One cell: the fold of the runs behind it.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Summary {
    pub status: Status,
    /// The runs that are not plain ✅, one entry per distinct reason.
    pub notes: Vec<Note>,
}

/// Why some of a cell's runs are not plain ✅.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Note {
    /// Which of the cell's runs this is about, where the cell has more than one ("" where it has
    /// one).
    pub runs: String,
    pub status: Status,
    pub text: &'static str,
}

const UNAUDITED: Summary = Summary {
    status: Status::Unaudited,
    notes: Vec::new(),
};

/// What `f`'s probe does on `c`.
fn run(f: Feature, c: Config) -> Cell {
    exception(f, c).unwrap_or(FULL)
}

/// The features a probe drives.
fn probed() -> impl Iterator<Item = Feature> {
    Feature::ALL.into_iter().filter(|f| !UNPROBED.contains(f))
}

fn feature_cell(f: Feature, m: Mode) -> Summary {
    if UNPROBED.contains(&f) {
        return UNAUDITED;
    }
    match m {
        Mode::Executor(e) => on_executor(e, Props::NONE, |c| vec![(String::new(), run(f, c))]),
        // An executor with no runner for the property is left to the property's own row.
        Mode::Property(p) => fold(
            Executor::ALL
                .into_iter()
                .filter_map(|e| {
                    let c = runner_for(e, Props::of(&[p]))?;
                    Some((e.label().to_string(), run(f, c)))
                })
                .collect(),
        ),
    }
}

/// The cell for the properties `need` on `e`: its runner's `runs`, or why there is no runner, or ❔.
fn on_executor(e: Executor, need: Props, runs: impl Fn(Config) -> Vec<(String, Cell)>) -> Summary {
    match runner_for(e, need) {
        Some(c) => fold(runs(c)),
        None => config_exception(e, need).map_or(UNAUDITED, |x| fold(vec![(String::new(), x)])),
    }
}

/// The pair `p` × `q`: every probe on each executor with a runner for both, and the executors whose
/// own reason rules the pair out. An executor that rules out `p` or `q` alone is left to that
/// property's row.
fn pair_cell(p: Property, q: Property) -> Summary {
    let need = Props::of(&[p, q]);
    let alone = |e, x| config_exception(e, Props::of(&[x])).is_some();
    // What each executor says about the pair: its runner, or its reason there is none.
    let says: Vec<(Executor, Result<Config, Cell>)> = Executor::ALL
        .into_iter()
        .filter_map(|e| match runner_for(e, need) {
            Some(c) => Some((e, Ok(c))),
            None if alone(e, p) || alone(e, q) => None,
            None => Some((e, Err(config_exception(e, need)?))),
        })
        .collect();
    let several = says.len() > 1;
    let mut runs = Vec::new();
    for (e, say) in says {
        match say {
            Ok(c) => runs.extend(probed().map(|f| {
                let who = if several {
                    format!("{} on {}", f.label(), e.label())
                } else {
                    f.label().to_string()
                };
                (who, run(f, c))
            })),
            Err(x) => runs.push((e.label().to_string(), x)),
        }
    }
    fold(runs)
}

/// Fold a cell's runs, each named by what varies across the cell. No runs is ❔.
fn fold(runs: Vec<(String, Cell)>) -> Summary {
    if runs.is_empty() {
        return UNAUDITED;
    }
    let all = |s| runs.iter().all(|(_, c)| c.status == s);
    let status = if all(Status::Full) {
        Status::Full
    } else if runs.iter().any(|(_, c)| c.status == Status::NotYet) {
        Status::NotYet
    } else if all(Status::Declines) {
        Status::Declines
    } else {
        Status::Conditional
    };
    let mut notes: Vec<Note> = Vec::new();
    for (who, c) in runs.into_iter().filter(|(_, c)| c.status != Status::Full) {
        match notes
            .iter_mut()
            .find(|n| n.status == c.status && n.text == c.note)
        {
            Some(n) => {
                n.runs.push_str("; ");
                n.runs.push_str(&who);
            }
            None => notes.push(Note {
                runs: who,
                status: c.status,
                text: c.note,
            }),
        }
    }
    Summary { status, notes }
}
