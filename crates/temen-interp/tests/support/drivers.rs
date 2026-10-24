//! Every bytecode-tier driver that runs a guest, as data (#1414): one place that knows how to run a
//! module on each of them, so a cross-driver test lists its cases and asks for [`ALL`], instead of
//! hand-rolling its own `enum Driver` (five test files do).
//!
//! The tree-walk oracle is here too, as the reference every driver is compared against
//! (INVARIANTS #9). Include with `#[path = "support/drivers.rs"] mod drivers;`.
#![allow(dead_code)] // each test binary uses a different subset

use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_with_host, Host, Trap, Value};
use temen_ir::Module;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Driver {
    /// The tree-walk oracle (`run_with_host`).
    Oracle,
    /// The cooperative executor (`bytecode::compile_and_run_with_host`).
    Coop,
    /// The OS-thread parallel driver (`compile_and_run_capture_over_parallel_with_host`): executor 2,
    /// an OS thread per task on the pump's rules (#1414 3e), over a window it reserves itself.
    Parallel,
    /// The debug scheduler (`ScheduledDebugRun`), run to completion through its breakpoints.
    Debug,
}

pub const ALL: [Driver; 4] = [
    Driver::Oracle,
    Driver::Coop,
    Driver::Parallel,
    Driver::Debug,
];

/// The executors a run its embedder pumps ([`bytecode::CoopRun`]) runs on (#1414 5b).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pumped {
    /// The pump: every task in turn on the embedder's thread (`CoopRun::new_reserved`).
    Pump,
    /// Executor 2: every task on a thread of its own (`CoopRun::new_parallel`).
    Threads,
}

pub const PUMPED: [Pumped; 2] = [Pumped::Pump, Pumped::Threads];

impl Pumped {
    /// A run of `m`'s function 0 on `args`, with `fuel` and `host`, over a reservation of
    /// `2^reserved_log2` bytes. Panics if the executor declines `m` or its entry.
    pub fn open(
        self,
        m: &Module,
        args: &[Value],
        fuel: u64,
        host: Host,
        reserved_log2: u8,
    ) -> bytecode::CoopRun {
        let run = match self {
            Pumped::Pump => {
                bytecode::CoopRun::new_reserved(m, 0, args, fuel, host, None, &[], reserved_log2)
            }
            Pumped::Threads => bytecode::CoopRun::new_parallel(
                m,
                0,
                args,
                fuel,
                host,
                &[],
                reserved_log2,
                None,
                true,
            ),
        };
        run.expect("the module is in the engine's subset")
            .expect("the run starts")
    }
}

/// What one run produced: the entry's result and what reached the root powerbox's streams.
#[derive(Debug, PartialEq)]
pub struct Ran {
    pub result: Result<Vec<Value>, Trap>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

impl Ran {
    fn of(result: Result<Vec<Value>, Trap>, host: &Host) -> Ran {
        Ran {
            result,
            stdout: host.stdout_bytes(),
            stderr: host.stderr_bytes(),
        }
    }
}

/// The fuel each driver is handed.
pub const FUEL: u64 = 50_000_000;

/// How often a driver its embedder pumps (the debug scheduler) is pumped again while it waits on a
/// host-answered call (#2495) that a thread of the test answers.
const ANSWER_POLL: std::time::Duration = std::time::Duration::from_millis(1);

/// Run `m`'s function 0 on `driver`. `setup` builds a fresh powerbox and the entry's arguments (it
/// is called once per run, so handles are minted in the same order on every driver). `None` means
/// the driver declined the module — it does not compile on that engine, or the debug tier fell back.
pub fn run_on(driver: Driver, m: &Module, setup: &dyn Fn() -> (Host, Vec<Value>)) -> Option<Ran> {
    run_on_then(driver, m, setup, &|_| ()).map(|(ran, ())| ran)
}

/// [`run_on`], and what `after` reads from the root powerbox once the run is over (a budget's use,
/// say).
pub fn run_on_then<R>(
    driver: Driver,
    m: &Module,
    setup: &dyn Fn() -> (Host, Vec<Value>),
    after: &dyn Fn(&Host) -> R,
) -> Option<(Ran, R)> {
    let (mut host, args) = setup();
    let done = |r, host: &Host| Some((Ran::of(r, host), after(host)));
    let mut fuel = FUEL;
    match driver {
        Driver::Oracle => {
            let r = run_with_host(m, 0, &args, &mut fuel, &mut host);
            done(r, &host)
        }
        Driver::Coop => {
            let r = bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host)?;
            done(r, &host)
        }
        Driver::Parallel => {
            let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
                m,
                0,
                &args,
                &mut fuel,
                &[],
                None,
                &mut host,
            )?;
            done(r, &host)
        }
        Driver::Debug => {
            let mut d = ScheduledDebugRun::new_with_host(m, 0, &args, host)?;
            let r = loop {
                match d.run_until_stop(&mut fuel) {
                    SchedStop::Finished(r) => break r,
                    SchedStop::Break { .. } => continue,
                    SchedStop::Declined => return None,
                    // #2495, #1414 D4 — a host-answered call: told as the thread makes it, and the
                    // run idles on it once nothing else can run. This harness answers none itself;
                    // a thread of the embedder's answers it, so look again shortly. It feeds no
                    // input either, so a run idle on a read stopped early. Ask the run that, not
                    // whether a call is still owed: its answer can land between the stop and the
                    // question.
                    SchedStop::CapPark { .. } => continue,
                    SchedStop::Idle { .. } if !d.stdin_parked() => std::thread::sleep(ANSWER_POLL),
                    other => panic!("the debug scheduler stopped early: {other:?}"),
                }
            };
            done(r, d.host())
        }
    }
}

/// Run `m` on every driver and assert each one gives `want` — result and streams. Every driver must
/// run the module (none may decline), so a case cannot pass by falling back. A failure lists every
/// driver that disagreed, not just the first.
pub fn agree_on_every_driver(
    what: &str,
    m: &Module,
    setup: &dyn Fn() -> (Host, Vec<Value>),
    want: &Ran,
) {
    agree_on(&ALL, what, m, setup, want);
}

/// [`agree_on_every_driver`] over `drivers` only.
pub fn agree_on(
    drivers: &[Driver],
    what: &str,
    m: &Module,
    setup: &dyn Fn() -> (Host, Vec<Value>),
    want: &Ran,
) {
    let wrong: Vec<String> = drivers
        .iter()
        .filter_map(|&d| match run_on(d, m, setup) {
            Some(got) if &got == want => None,
            Some(got) => Some(format!("  {d:?}: {got:?}")),
            None => Some(format!("  {d:?}: declined the module")),
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "{what}: want {want:?} on every driver, but\n{}",
        wrong.join("\n")
    );
}
