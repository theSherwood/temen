//! Every bytecode-tier driver that runs a guest, as data (#1414): one place that knows how to run a
//! module on each of them, so a cross-driver test lists its cases and asks for [`ALL`], instead of
//! hand-rolling its own `enum Driver` (five test files do).
//!
//! The tree-walk oracle is here too, as the reference every driver is compared against
//! (INVARIANTS #9). Include with `#[path = "support/drivers.rs"] mod drivers;`.
#![allow(dead_code)] // each test binary uses a different subset

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};

use temen_interp::bytecode::{self, SchedStop, ScheduledDebugRun};
use temen_interp::{run_with_host, Host, Region, Trap, Value};
use temen_ir::Module;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Driver {
    /// The tree-walk oracle (`run_with_host`).
    Oracle,
    /// The cooperative executor (`bytecode::compile_and_run_with_host`).
    Coop,
    /// The OS-thread parallel driver (`compile_and_run_capture_over_parallel_with_host`).
    Parallel,
    /// The debug scheduler (`ScheduledDebugRun`), run to completion through its breakpoints.
    Debug,
    /// The resumable single vCPU (`Vcpu::run`) under a native orchestrator: the engine the browser's
    /// per-Worker driver and the op-13 drivers wrap.
    Vcpu,
}

pub const ALL: [Driver; 5] = [
    Driver::Oracle,
    Driver::Coop,
    Driver::Parallel,
    Driver::Debug,
    Driver::Vcpu,
];

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

const FUEL: u64 = 50_000_000;

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
            let (back, base, layout) = window(m);
            let r = bytecode::compile_and_run_capture_over_parallel_with_host(
                m,
                0,
                &args,
                &mut fuel,
                &[],
                Arc::clone(&back),
                &mut host,
            );
            drop(back);
            // SAFETY: the layout `window` allocated; the run joined every vCPU and dropped every view.
            unsafe { std::alloc::dealloc(base, layout) };
            let (r, _image) = r?;
            done(r, &host)
        }
        Driver::Debug => {
            let mut d = ScheduledDebugRun::new_with_host(m, 0, &args, host)?;
            let r = loop {
                match d.run_until_stop(&mut fuel) {
                    SchedStop::Finished(r) => break r,
                    SchedStop::Break { .. } => continue,
                    SchedStop::Declined => return None,
                    other => panic!("the debug scheduler stopped early: {other:?}"),
                }
            };
            done(r, d.host())
        }
        Driver::Vcpu => {
            let prog = bytecode::VcpuProgram::compile(m)?;
            let (back, base, layout) = window(m);
            let shared = Mutex::new(host);
            let orch = Orch::default();
            let r = {
                let root = bytecode::Vcpu::new_root(&prog, 0, &args, Arc::clone(&back), &[])
                    .map(|v| v.with_shared_host(&shared));
                match root {
                    Err(t) => Err(t),
                    Ok(root) => {
                        let win = Win {
                            base: WinPtr(base),
                            back: Arc::clone(&back),
                            size_log2: size_log2(m),
                            host: Some(&shared),
                        };
                        std::thread::scope(|s| drive(s, &prog, &win, &orch, root))
                    }
                }
            };
            drop(back);
            // SAFETY: as for `Parallel` — every vCPU thread joined inside the scope above.
            unsafe { std::alloc::dealloc(base, layout) };
            let host = shared.into_inner().unwrap_or_else(|e| e.into_inner());
            done(r, &host)
        }
    }
}

/// The `log2` of `m`'s declared window.
fn size_log2(m: &Module) -> u8 {
    m.memory.map_or(16, |mc| mc.size_log2)
}

/// A zeroed window of the module's declared size, shareable across vCPU threads. The `unsafe` of
/// lending host memory stays here in the test embedder, as in every parallel harness.
fn window(m: &Module) -> (Arc<Region>, *mut u8, std::alloc::Layout) {
    let size = 1usize << size_log2(m);
    let layout = std::alloc::Layout::from_size_align(size, 8).expect("layout");
    // SAFETY: a non-zero, 8-aligned layout; the caller frees it once the run has joined.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null(), "window allocation");
    // SAFETY: `base` owns `size` zeroed bytes and outlives every vCPU of the run.
    (
        Arc::new(unsafe { Region::shared(base, size as u64) }),
        base,
        layout,
    )
}

/// The completion slots a `Vcpu` run's joins wait on.
#[derive(Default)]
struct Orch {
    next: Mutex<u64>,
    done: Mutex<HashMap<u64, Result<Vec<Value>, Trap>>>,
    cv: Condvar,
}

/// A window pointer that crosses the scoped-thread hand-off (raw pointers are not `Send`). Null for
/// a detached child, whose window is not addressable from here.
#[derive(Clone, Copy)]
struct WinPtr(*mut u8);
// SAFETY: it only ever names the one live window allocation, which outlives the scope, or is null.
unsafe impl Send for WinPtr {}

/// The window a vCPU runs over, which the threads it spawns share: its bytes (at `base`), its backing
/// and size, and the powerbox its threads share (the root's; a child's own is not shareable here, so
/// its threads get an empty one, as the browser's per-Worker threads of a detached child do).
struct Win<'e> {
    base: WinPtr,
    back: Arc<Region>,
    size_log2: u8,
    host: Option<&'e Mutex<Host>>,
}

/// Drive one `Vcpu` to completion on this thread, starting each child on its own scoped thread — a
/// confined one over its carve, a detached one over a fresh backing — as the browser's per-Worker
/// driver does with Workers. An event this harness does not orchestrate fails the test rather than
/// guessing an answer.
fn drive<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    win: &Win<'e>,
    orch: &'e Orch,
    mut vcpu: bytecode::Vcpu<'e>,
) -> Result<Vec<Value>, Trap> {
    loop {
        match vcpu.run() {
            bytecode::VcpuEvent::Done(v) => return Ok(v),
            bytecode::VcpuEvent::Trapped(t) => return Err(t),
            bytecode::VcpuEvent::Instantiate {
                carve, size_log2, ..
            } => {
                if win.base.0.is_null() {
                    unorchestrated("a confined spawn inside a detached child");
                }
                // SAFETY: the engine validated the carve inside this vCPU's window.
                let base = WinPtr(unsafe { win.base.0.add(carve as usize) });
                // SAFETY: `2^size_log2` valid bytes at the validated carve, alive for the scope.
                let back = Arc::new(unsafe { Region::shared(base.0, 1u64 << size_log2) });
                let child_win = Win {
                    base,
                    back,
                    size_log2,
                    host: None,
                };
                let child = take_child(&mut vcpu, prog, &child_win)?;
                start(scope, prog, orch, &mut vcpu, child, child_win);
            }
            bytecode::VcpuEvent::InstantiateDetached { size_log2 } => {
                // A fresh reservation, as every driver's detached window has; the engine seeds it.
                // Not flat on every host (no `mmap` on Windows), so its bytes are not addressed here.
                let back = Arc::new(Region::new(
                    1u64 << temen_ir::DEFAULT_RESERVED_LOG2,
                    temen_interp::host_page_size(),
                ));
                let child_win = Win {
                    base: WinPtr(std::ptr::null_mut()),
                    back,
                    size_log2,
                    host: None,
                };
                let child = take_child(&mut vcpu, prog, &child_win)?;
                start(scope, prog, orch, &mut vcpu, child, child_win);
            }
            bytecode::VcpuEvent::Spawn {
                func,
                sp,
                arg,
                module,
                vcpu: id,
            } => {
                // A thread over its spawner's window, as the browser's per-Worker driver starts one.
                let args = [Value::I64(sp), Value::I64(arg)];
                let back = Arc::clone(&win.back);
                let child = bytecode::Vcpu::new_child_sized(
                    prog,
                    module,
                    func,
                    &args,
                    back,
                    win.size_log2,
                )?
                .with_vcpu_id(id);
                let child = match win.host {
                    Some(h) => child.with_shared_host(h),
                    None => child,
                };
                let thread_win = Win {
                    base: win.base,
                    back: Arc::clone(&win.back),
                    size_log2: win.size_log2,
                    host: win.host,
                };
                start(scope, prog, orch, &mut vcpu, child, thread_win);
            }
            bytecode::VcpuEvent::Join { child } => {
                let mut g = orch.done.lock().unwrap();
                let r = loop {
                    if let Some(r) = g.remove(&child) {
                        break r;
                    }
                    g = orch.cv.wait(g).unwrap();
                };
                drop(g);
                vcpu.deliver_join(r);
            }
            // Named, not `_`: a new event fails to build here as in every driver (#1414).
            bytecode::VcpuEvent::TierUp { .. } => unorchestrated("TierUp"),
            bytecode::VcpuEvent::Wait { .. } => unorchestrated("Wait"),
            bytecode::VcpuEvent::Notify { .. } => unorchestrated("Notify"),
            bytecode::VcpuEvent::JitInstall { .. } => unorchestrated("JitInstall"),
            bytecode::VcpuEvent::JitUninstall { .. } => unorchestrated("JitUninstall"),
            bytecode::VcpuEvent::JitInvoke { .. } => unorchestrated("JitInvoke"),
            bytecode::VcpuEvent::CapPending { .. } => unorchestrated("CapPending"),
            bytecode::VcpuEvent::StdinPark => unorchestrated("StdinPark"),
        }
    }
}

/// The child the last event announced, started over `win`.
fn take_child<'e>(
    vcpu: &mut bytecode::Vcpu<'e>,
    prog: &'e bytecode::VcpuProgram,
    win: &Win<'e>,
) -> Result<bytecode::Vcpu<'e>, Trap> {
    let Some(pending) = vcpu.take_child() else {
        panic!("a spawn event carries its admitted child");
    };
    pending.start(prog, Arc::clone(&win.back), None)
}

/// Run `child` over `win` on its own scoped thread, and give `vcpu` its completion id as the token a
/// join hands back.
fn start<'s, 'e>(
    scope: &'s std::thread::Scope<'s, 'e>,
    prog: &'e bytecode::VcpuProgram,
    orch: &'e Orch,
    vcpu: &mut bytecode::Vcpu<'e>,
    child: bytecode::Vcpu<'e>,
    win: Win<'e>,
) {
    let id = {
        let mut n = orch.next.lock().unwrap();
        *n += 1;
        *n
    };
    scope.spawn(move || {
        let r = drive(scope, prog, &win, orch, child);
        orch.done.lock().unwrap().insert(id, r);
        orch.cv.notify_all();
    });
    vcpu.deliver_child(id);
}

fn unorchestrated(event: &str) -> ! {
    panic!("the Vcpu harness does not orchestrate {event}; extend `drivers::drive` to cover it")
}

/// The drivers that schedule a spawned child themselves. The [`Driver::Vcpu`]'s host runs its
/// children and has no surface to answer `poll`, `detach` or `kill` yet, so each traps there when it
/// runs (temen#2083).
pub const SCHEDULING: [Driver; 4] = [
    Driver::Oracle,
    Driver::Coop,
    Driver::Parallel,
    Driver::Debug,
];

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
