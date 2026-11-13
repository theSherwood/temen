//! **#2958 — a Cranelift JIT program's vCPUs as tasks of executor 2's scheduling core.** For a
//! program [`admit`] admits, the run is executor 2's (`CoopRun::new_parallel`), with the Cranelift JIT
//! as its emitted tier ([`ThreadTier`]): the root runs whole as a leaf, on the caller's thread, and
//! each `thread.spawn` child tiers up at its entry, so every vCPU is a kernel task whose own thread
//! runs Cranelift code from its entry. A §12 op the code makes goes to the kernel
//! ([`EmittedCall::park`]) through the vCPU's hook ([`temen_jit::kernel_rt`]), and the thread parks
//! there with its frames on its stack. Everything [`admit`] declines runs on the JIT's own thread
//! runtime, unchanged: `jit_cap_run` and `jit_run` ask it first, and nothing else chooses.
//!
//! The window is the JIT's, started as the JIT's own runs start one (`seed_window`, then the
//! host's page map over it), and it has one page map: the domain `Host`'s, which its capability
//! calls check and the kernel's rules read it through ([`temen_interp::FutexWindow`], #2958 slice 3).
//! The kernel builds no window of its own over it.
//!
//! **Fuel.** Each vCPU charges a fuel cell of its own that draws from its task's budget node
//! ([`EmittedCall::fuel_node`]), as today's JIT's vCPUs draw from their domain's: the node is the
//! one source of truth, and what a cell drew and did not burn goes back to it when its vCPU ends. A
//! refill draws and runs on: a native task does not yield to the kernel at a refill (quantum
//! preemption is #2958 slice 6).

use std::sync::Arc;
use temen_interp::bytecode::{
    CallEntry, CoopEvent, CoopRun, EmittedCall, EmittedTarget, NativeStop, NativeTier, Opener,
    ProgramEmit, ThreadPlatform, ThreadTier, TierWindow,
};
use temen_interp::{Host, ShadowArena, Trap, Value};
use temen_ir::Module;
use temen_jit::kernel_rt::{KernelPark, KernelRun, VcpuEntry};
use temen_jit::{CompiledModule, FuelCell, JitOutcome, TrapKind};

/// A program [`admit`] admitted: only a [`run`] of it takes the kernel's route.
pub struct Admitted<'m>(&'m Module);

/// Whether `m`, started from `init` under `host`, runs its vCPUs as the kernel's tasks:
/// `temen_interp::bytecode::native_scope`, plus the start's own clauses:
/// - **`region start`**: a window that would start with a §13 region aliased into it: a captured
///   layout that aliases one, or a host with a pre-map to apply (an op-17 pre-map an embedder
///   staged, or a thaw's §13 pages, `Host::has_premap`). A native window keys a futex word on its
///   own backing, so a word on an alias would not rendezvous with the region's other mappings. With
///   this clause and `native_scope`'s `cap` (which declines every op that maps a region), no page of
///   a native window is ever an alias. The phase that brings §13 regions onto this route removes
///   both.
/// - **`kept code`**: the embedder keeps the run's compile for later runs (a `JitCodeCache`,
///   #2389), whose shared code the kernel route does not instance yet (#2981).
///
/// `Err` names the clause that keeps it on the JIT's own runtime, with nothing taken from `host`.
pub fn admit<'m>(
    m: &'m Module,
    host: &Host,
    init: Option<&temen_interp::MemLayout>,
    kept_code: bool,
) -> Result<Admitted<'m>, &'static str> {
    temen_interp::bytecode::native_scope(m, host)?;
    if host.has_premap() || init.is_some_and(temen_interp::MemLayout::aliases_regions) {
        return Err("region start");
    }
    if kept_code {
        return Err("kept code");
    }
    Ok(Admitted(m))
}

/// How a kernel run ended.
pub struct KernelEnd {
    /// The entry's results, or the trap the run ended with.
    pub result: Result<Vec<i64>, Trap>,
    /// The window's bytes, when the run asked for a snapshot: `None` when it did not, or the OS
    /// refused to make a page readable again.
    pub snapshot: Option<Vec<u8>>,
    /// Whether the run asked for a snapshot ([`Start::snapshot_cap`]).
    pub asked: bool,
    /// How many vCPUs ran as kernel tasks, the root's included.
    pub vcpus: u64,
}

impl KernelEnd {
    /// A run the OS would not give its window to (#2305, #2286): it ends as one the process cannot
    /// hold, before its first instruction.
    fn refused() -> KernelEnd {
        KernelEnd {
            result: Err(Trap::ThreadFault),
            snapshot: None,
            asked: false,
            vcpus: 0,
        }
    }

    /// The end as the JIT's own runs report it: its outcome, and the window's bytes it copied back.
    /// A snapshot the OS refused (a page it would not make readable again) ends the run as a window
    /// it cannot hold does, `ThreadFault`, as the JIT's own snapshot does (#2286).
    pub fn outcome(self) -> (JitOutcome, Vec<u8>) {
        let outcome = match self.result {
            Ok(v) => JitOutcome::Returned(v),
            Err(Trap::Exit(code)) => JitOutcome::Exited(code),
            Err(t) => JitOutcome::Trapped(
                TrapKind::from_code(t.code() as u32).unwrap_or(TrapKind::ThreadFault),
            ),
        };
        match (self.snapshot, self.asked) {
            (Some(bytes), _) => (outcome, bytes),
            (None, false) => (outcome, Vec::new()),
            (None, true) => (JitOutcome::Trapped(TrapKind::ThreadFault), Vec::new()),
        }
    }
}

/// A run of `m`'s `entry(args)` on the kernel's route over a fresh host, with the default fuel, as
/// the JIT's own entries answer it, and how many vCPUs ran as kernel tasks. `Err` names the clause
/// that declines it, and nothing ran. For the suites that run each case on both routes while both
/// exist (#2958): today's route is the direct `temen_jit` entry they always used.
pub fn run_fresh(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Result<(JitOutcome, u64), &'static str> {
    let mut host = Host::new();
    let admitted = admit(m, &host, None, false)?;
    let start = Start {
        entry,
        args,
        init_mem: &[],
        init: None,
        reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
        fuel: temen_interp::DEFAULT_FUEL,
        snapshot_cap: None,
        kill,
    };
    let end = run(admitted, start, &mut host).map_err(|_| "the JIT's compile")?;
    let vcpus = end.vcpus;
    Ok((end.outcome().0, vcpus))
}

/// The two routes a JIT program's threads run on while both exist (#2958): the JIT's own thread
/// runtime, and the kernel's. The suites run each case on both; today's arm goes with the code it
/// tests, in the phase-2 PR that deletes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Today,
    Kernel,
}

impl Route {
    pub const BOTH: [Route; 2] = [Route::Today, Route::Kernel];
}

/// Test-facing: run `m`'s `entry(args)` on the kernel route ([`run_fresh`]) and check the route a
/// case expects of it: `declined` names the clause that must decline it, `None` that it must run
/// as kernel tasks. `None` back when it was declined, as expected.
#[doc(hidden)]
pub fn expect_kernel(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
    declined: Option<&'static str>,
) -> Option<JitOutcome> {
    match (run_fresh(m, entry, args, kill), declined) {
        (Ok((o, vcpus)), None) => {
            assert!(vcpus >= 1, "the run's vCPUs ran as kernel tasks");
            Some(o)
        }
        (Err(clause), Some(want)) => {
            assert_eq!(clause, want, "the clause that declines it");
            None
        }
        (Ok(_), Some(want)) => panic!("expected `{want}` to decline it; it ran on the kernel"),
        (Err(clause), None) => panic!("expected the kernel route; `{clause}` declined it"),
    }
}

/// What a run of `entry(args)` starts with and hands back.
pub struct Start<'a> {
    pub entry: u32,
    pub args: &'a [i64],
    /// The window's low bytes before the data segments land (argv and env).
    pub init_mem: &'a [u8],
    /// The layout the window starts from, in place of `init_mem` and the data segments: a thaw's, or
    /// a fresh image's (`jit_cap_run`'s). `None` for a fresh run.
    pub init: Option<&'a temen_interp::MemLayout>,
    pub reserved_log2: u8,
    /// The fuel the run's activation is granted ([`temen_interp::DEFAULT_FUEL`] by default).
    pub fuel: u64,
    /// What of the window to copy back ([`temen_jit::CompiledModule::run`]'s `snapshot_cap`).
    pub snapshot_cap: Option<usize>,
    /// The JIT kill door the run's code polls; its firing also stops the kernel's run.
    pub kill: Option<&'a Arc<temen_jit::Interrupt>>,
}

/// Run the admitted program as `start` says, its vCPUs as the kernel's tasks, under `host`. `Err` is
/// the JIT's compile refusing it, with nothing run and `host` as it was.
pub fn run(
    admitted: Admitted<'_>,
    start: Start<'_>,
    host: &mut Host,
) -> Result<KernelEnd, temen_jit::JitError> {
    let m = admitted.0;
    let f = m
        .funcs
        .get(start.entry as usize)
        .ok_or(temen_jit::JitError::Unsupported("no such entry"))?;
    if f.params.len() != start.args.len() {
        return Err(temen_jit::JitError::Unsupported("entry arity"));
    }
    // Fuel checks are compiled in when the run's chain meters: its own grant, or a level above it.
    let metered = start.fuel != u64::MAX || crate::run_node(host).is_some_and(|n| n.room() >= 0);
    // The compiled instance's cell, which no vCPU charges (each has its own, `tier_run`): it only
    // arms the compile's fuel checks, and lives as long as the code (`KernelRun::new`).
    let mut armed = metered.then(|| FuelCell::fixed(0));
    let cm = CompiledModule::compile(
        m,
        start.entry,
        crate::cap_thunk_locked,
        std::ptr::null_mut(), // each vCPU's context names its domain's host cell
        start.reserved_log2,
        None, // no §14 children
        start.kill,
        armed.as_deref_mut().map(|c| c as *mut FuelCell),
        None,
        None,
        0,
    )?;
    // The JIT's own window start, as a fresh run or a thaw of today's route starts it: the window
    // seeded and protected (`seed_window`), then the host's page map started over it with the
    // read-only pages and any pre-map (`start_window_view`). That page map is the window's one: the
    // `Host`'s capability calls check it, and the kernel's rules read the window through it.
    let layout = start.init;
    if let Some(l) = layout {
        host.reset_cap_pages(l.page_map());
    }
    let mut cm = cm;
    cm.set_restore(layout.and_then(crate::jit_restore));
    let init_mem = layout.map_or(start.init_mem, |l| l.bytes());
    let (kr, ro) = match KernelRun::new(cm, armed, init_mem) {
        Some(Some(started)) => started,
        Some(None) => return Ok(KernelEnd::refused()),
        None => return Err(temen_jit::JitError::Unsupported("kernel window")),
    };
    let kr = Arc::new(kr);
    let (base, reserved) = kr.window();
    // SAFETY: `[base, base + reserved)` is the window `kr` owns, which the reader keeps.
    let keep: Arc<dyn std::any::Any + Send + Sync> = Arc::clone(&kr) as _;
    let Some((read, runs)) = (unsafe {
        crate::start_window_view(
            host,
            base,
            kr.mapped(),
            reserved,
            &ro,
            ShadowArena::EMPTY,
            keep,
        )
    }) else {
        return Err(temen_jit::JitError::Unsupported("kernel window pre-map"));
    };
    let tier_window = TierWindow {
        read,
        charge: crate::jit_root_window(m, layout, start.reserved_log2),
    };
    let vcpus = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let state: Arc<dyn std::any::Any + Send + Sync> = Arc::new(TierState {
        kr: Arc::clone(&kr),
        vcpus: Arc::clone(&vcpus),
    });
    let n = m.funcs.len();
    let platform = ThreadPlatform {
        tier: Some(ThreadTier {
            root: ProgramEmit {
                eligible: std::iter::repeat_n(true, n).collect(),
                ..ProgramEmit::default()
            },
            spill: 0,
            modules: None,
            leaf: Some(Arc::new(|offer| offer.module == 0)),
            run: tier_run,
            native: Some(NativeTier {
                state,
                window: Some(tier_window),
            }),
        }),
        // The root runs on the caller's thread, as the JIT's own runtime runs it.
        opener: Opener::RunsRoot,
        ..ThreadPlatform::OS
    };
    host.set_thread_platform(platform);
    // The kill door stops the kernel's run too, so its parked tasks unwind: the embedder's stop, as
    // the JIT's kill is (`OutOfFuel`).
    let door = Arc::new(temen_interp::bytecode::RunDoor::default());
    let embedders = host.replace_run_door(Some(Arc::clone(&door)));
    let _kill = start.kill.map(|k| {
        let door = Arc::clone(&door);
        // A busy run lock answers `false`: the stop is asked again (`RunDoor::stop`).
        temen_jit::kernel_rt::on_kill(
            k,
            Box::new(move || {
                while !door.stop() {
                    std::thread::yield_now();
                }
            }),
        )
    });
    let vals: Vec<Value> = f
        .params
        .iter()
        .zip(start.args)
        .map(|(t, a)| slot_value(*t, *a))
        .collect();
    let opened = CoopRun::open_parallel(
        m,
        start.entry,
        &vals,
        start.fuel,
        std::mem::take(host),
        &[], // the window is the tier's, started above
        reserved.trailing_zeros() as u8,
        None,
        false,
    );
    let mut co = match opened {
        Ok(co) => co,
        // No run started: the caller has its host back, as on the JIT's own runtime. A run its
        // activation refused (the process's pools) ends as an ordinary run does; one the engine
        // declines (outside its subset) is the JIT's compile refusing it.
        Err(refused) => {
            let (t, back) = *refused;
            *host = back;
            host.set_thread_platform(ThreadPlatform::OS);
            host.replace_run_door(embedders);
            let t = t.ok_or(temen_jit::JitError::Unsupported(
                "executor 2 declines the module",
            ))?;
            return Ok(KernelEnd {
                result: Err(t),
                snapshot: None,
                asked: false,
                vcpus: 0,
            });
        }
    };
    // The activation charged the window's bound; it holds what its start made of it, as on the
    // JIT's own runtime (`window_aliases`).
    co.host_mut().start_window(runs);
    let result = match co.run() {
        CoopEvent::Done(vals) => Ok(vals.iter().map(|v| value_slot(*v)).collect()),
        CoopEvent::Trapped(t) => Err(t),
        _ => Err(Trap::Malformed),
    };
    *host = std::mem::take(&mut *co.host_mut());
    host.set_thread_platform(ThreadPlatform::OS);
    host.replace_run_door(embedders);
    drop(co);
    drop(_kill);
    // Every thread has counted itself out, so no guest code runs: the window holds what the run
    // left in it.
    // The window is unmapped as the last holder of `kr` drops it, the reader's included (a thread
    // that has counted itself out may still hold the run, and with it both, for a moment).
    let snapshot = start.snapshot_cap.and_then(|cap| kr.snapshot(cap));
    Ok(KernelEnd {
        result,
        snapshot,
        asked: start.snapshot_cap.is_some(),
        vcpus: vcpus.load(std::sync::atomic::Ordering::Relaxed),
    })
}

/// A scalar argument slot as the value of its declared type.
fn slot_value(t: temen_ir::ValType, s: i64) -> Value {
    use temen_ir::ValType as T;
    match t {
        T::I32 | T::Cap => Value::I32(s as i32),
        T::F32 => Value::F32(f32::from_bits(s as u32)),
        T::F64 => Value::F64(f64::from_bits(s as u64)),
        T::Ref => Value::Ref(s as u64),
        _ => Value::I64(s),
    }
}

/// A scalar result as its slot.
fn value_slot(v: Value) -> i64 {
    match v {
        Value::I32(x) => x as i64,
        Value::I64(x) => x,
        Value::F32(x) => x.to_bits() as i64,
        Value::F64(x) => x.to_bits() as i64,
        Value::Ref(x) => x as i64,
        Value::V128(_) => 0,
    }
}

/// The tier's run state ([`NativeTier::state`]): the JIT's half of the run, and how many vCPUs ran.
struct TierState {
    kr: Arc<KernelRun>,
    vcpus: Arc<std::sync::atomic::AtomicU64>,
}

/// [`ThreadTier::run`] for the Cranelift JIT: run the task's entry on this thread as a vCPU.
fn tier_run(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    let st = call
        .native()
        .and_then(|n| n.state.downcast_ref::<TierState>())
        .ok_or(Trap::Malformed)?;
    st.vcpus.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let kr = Arc::clone(&st.kr);
    let EmittedTarget::Region { module: 0, func } = call.target else {
        return Err(Trap::Malformed);
    };
    let argv: Vec<i64> = call.argv.to_vec();
    let entry = match call.entry {
        CallEntry::Root => VcpuEntry::Root(&argv),
        CallEntry::Thread => VcpuEntry::Thread {
            func,
            arg0: argv.first().copied().unwrap_or(0),
            arg1: argv.get(1).copied().unwrap_or(0),
        },
        // A task's code is the JIT's from its entry, so no interpreted call tiers up here: every
        // vCPU of an admitted program is native, or the run traps.
        CallEntry::Call => return Err(Trap::Malformed),
    };
    // Its own cell, drawing from its task's budget node, so siblings never share one; what it drew
    // and did not burn goes back to the node as it drops, when the vCPU ends however it ends.
    let mut fuel = match call.fuel_node.take() {
        Some(node) => FuelCell::drawing(Arc::new(crate::HostNode(node))),
        None => FuelCell::fixed(u64::MAX),
    };
    let (tls, host, end) = (call.tls, call.host_cell(), call.end);
    let ran = {
        let mut hook = Hook(call);
        // SAFETY: `host` is the task's domain's live host cell, what `cap_thunk_locked` takes; `run`
        // started the window before any task ran.
        unsafe {
            kr.run_vcpu(
                entry,
                tls,
                end,
                &mut hook,
                host as *mut core::ffi::c_void,
                &mut fuel,
            )
        }
    };
    match ran {
        Ok(vals) => {
            call.results = vals;
            Ok(())
        }
        Err(code) => Err(Trap::from_code(code).unwrap_or(Trap::ThreadFault)),
    }
}

/// A vCPU's §12 ops, applied by its task's rules.
struct Hook<'c, 'a>(&'c mut EmittedCall<'a>);

impl KernelPark for Hook<'_, '_> {
    fn park(&mut self, stop: NativeStop) -> Result<[i64; 2], i64> {
        self.0.park(stop).map_err(|t| t.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program whose root does not run whole on the JIT: an import its host does not bind keeps
    /// the root interpreted (`image_parks`), so its call to a function the tier emits tiers up as an
    /// interpreted call.
    const INTERPRETED_ROOT: &str = r#"import 0 "vm_page_size" () -> (i64)
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vst, vv = thread.join vh
  vr = call 1(vz, vv)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vone = i64.const 1
  vr = i64.add vb vone
  return vr
  }
}
"#;

    /// The JIT's tier traps an interpreted call (`CallEntry::Call`): what makes every vCPU of an
    /// admitted program native. `admit` declines this one (`imports`); run past it, it traps.
    #[test]
    fn an_interpreted_call_traps_on_the_jits_tier_and_admit_keeps_it_away() {
        let m = temen_text::parse_module(INTERPRETED_ROOT).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let mut host = Host::new();
        assert_eq!(admit(&m, &host, None, false).err(), Some("imports"));
        let start = Start {
            entry: 0,
            args: &[],
            init_mem: &[],
            init: None,
            reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
            fuel: u64::MAX,
            snapshot_cap: None,
            kill: None,
        };
        let end = run(Admitted(&m), start, &mut host).expect("compiles");
        assert_eq!(end.result, Err(Trap::Malformed));
    }
}
