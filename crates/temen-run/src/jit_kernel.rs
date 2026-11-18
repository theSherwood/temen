//! **#2958 — a Cranelift JIT program's vCPUs as tasks of executor 2's scheduling core.** For a
//! program [`admit`] admits, the run is executor 2's (`CoopRun::new_parallel`), with the Cranelift JIT
//! as its emitted tier ([`ThreadTier`]): the root runs whole as a leaf, on the caller's thread, and
//! each `thread.spawn` child tiers up at its entry, so every vCPU is a kernel task whose own thread
//! runs Cranelift code from its entry. A §12 op the code makes goes to the kernel
//! ([`EmittedCall::park`]) through the vCPU's hook ([`temen_jit::kernel_rt`]), and the thread parks
//! there with its frames on its stack. A capability call that parks (a pipe or stdin read, a pipe
//! write, a punted call) or wakes another's park (a region post's `signal`) is dispatched under the
//! domain's `Host` lock, decided there as an interpreted op's is ([`temen_interp::bytecode::call_park`]),
//! and carried out by the kernel's rules once the lock is let go ([`park_call`], #2958 slice 5): the
//! lock order is the run lock, then the `Host` lock. Everything [`admit`] declines runs on the JIT's
//! own thread runtime, unchanged: `jit_cap_run` and `jit_run` ask it first, and nothing else chooses.
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
    CallEntry, CallPark, CoopEvent, CoopRun, EmittedCall, EmittedTarget, NativeStop, NativeTier,
    Opener, ProgramEmit, ThreadPlatform, ThreadTier, TierWindow,
};
use temen_interp::{Host, ShadowArena, Trap, Value};
use temen_ir::Module;
use temen_jit::kernel_rt::{KernelPark, KernelRun, VcpuEntry};
use temen_jit::{CompiledModule, FuelCell, JitOutcome, TrapKind};

/// A program [`admit`] admitted: only a [`run`] of it takes the kernel's route.
pub struct Admitted<'m>(&'m Module);

/// Whether `m`, under `host`, runs its vCPUs as the kernel's tasks, on a run granted `fuel`:
/// `temen_interp::bytecode::native_scope`, plus the start's own clauses:
/// - **`fiber fuel`**: the run is metered ([`metered`]) and `m` uses fibers. A fiber's code charges
///   its creator vCPU's fuel cell wherever it runs, so a fiber resumed on another vCPU loses charges
///   to its creator's at once, where the oracle charges the vCPU that runs it. It is a memory-safety
///   gate too: a vCPU's cell is a local of its thread (`tier_run`), so a metered fiber resumed after
///   its creator ended would write into a dead thread's stack; `KernelRun::new` refuses such a
///   compile whatever admitted it. #3080's fix removes the clause, and must first move a vCPU's cell
///   off its thread's stack;
/// - **`kept code`**: the embedder keeps the run's compile for later runs (a `JitCodeCache`,
///   #2389), whose shared code the kernel route does not instance yet (#2981).
///
/// A window that starts with a §13 region aliased into it (a restored layout's, a pre-map) is
/// admitted: its words key on their region (#2958 slice 4). `Err` names the clause that keeps it on the JIT's own runtime, with nothing taken from
/// `host`.
pub fn admit<'m>(
    m: &'m Module,
    host: &Host,
    kept_code: bool,
    fuel: u64,
) -> Result<Admitted<'m>, &'static str> {
    temen_interp::bytecode::native_scope(m, host)?;
    if metered(host, fuel) && m.funcs.iter().any(temen_ir::Func::uses_fibers) {
        return Err("fiber fuel");
    }
    if kept_code {
        return Err("kept code");
    }
    Ok(Admitted(m))
}

/// Whether a run granted `fuel` under `host` is metered: its own grant, or a level of its chain
/// above it, bounds it. Its code is then compiled with fuel checks ([`run`]).
fn metered(host: &Host, fuel: u64) -> bool {
    fuel != u64::MAX || crate::run_node(host).is_some_and(|n| n.room() >= 0)
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

/// A run of `m`'s `entry(args)` on the kernel's route over a fresh host, granted `fuel`
/// ([`temen_interp::DEFAULT_FUEL`] as the JIT's own entries grant; `u64::MAX` unmetered), as the
/// JIT's own entries answer it, and how many vCPUs ran as kernel tasks. `Err` names the clause
/// that declines it, and nothing ran. For the suites that run each case on both routes while both
/// exist (#2958): today's route is the direct `temen_jit` entry they always used.
pub fn run_fresh(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
    fuel: u64,
) -> Result<(JitOutcome, u64), &'static str> {
    let mut host = Host::new();
    let admitted = admit(m, &host, false, fuel)?;
    let start = Start {
        entry,
        args,
        init_mem: &[],
        init: None,
        reserved_log2: temen_ir::DEFAULT_RESERVED_LOG2,
        fuel,
        snapshot_cap: None,
        kill,
    };
    let end = run(admitted, start, &mut host).map_err(|_| "the JIT's compile")?;
    let vcpus = end.vcpus;
    Ok((end.outcome().0, vcpus))
}

/// How long a suite's kernel-lane run ([`expect_kernel`]) may take before the case fails: a lost
/// wake must be a failure that says what ran, not a job's timeout.
const LANE_BOUND: std::time::Duration = std::time::Duration::from_secs(60);

/// [`run_fresh`] on a thread of its own, failing the case if it does not end within
/// [`LANE_BOUND`]: the thread is left behind, and the panic names the entry, its arguments and the
/// module's size.
fn bounded_fresh(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
    fuel: u64,
) -> Result<(JitOutcome, u64), &'static str> {
    let (m2, args2, kill2) = (m.clone(), args.to_vec(), kill.cloned());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run_fresh(&m2, entry, &args2, kill2.as_ref(), fuel));
    });
    match rx.recv_timeout(LANE_BOUND) {
        Ok(r) => r,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
            "the kernel route's run of entry {entry}{args:?} ({} functions, fuel {fuel}) did not end \
             within {LANE_BOUND:?}",
            m.funcs.len()
        ),
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
            panic!("the kernel route's run of entry {entry}{args:?} panicked")
        }
    }
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

/// Test-facing: run `m`'s `entry(args)` on the kernel route ([`run_fresh`], granted `fuel`, bounded
/// by [`LANE_BOUND`]) and check the route a case expects of it: `declined` names the clause that must decline it, `None` that it must run
/// as kernel tasks. `None` back when it was declined, as expected.
#[doc(hidden)]
pub fn expect_kernel(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
    fuel: u64,
    declined: Option<&'static str>,
) -> Option<JitOutcome> {
    match (bounded_fresh(m, entry, args, kill, fuel), declined) {
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

/// Test-facing: the fuel a suite's kernel lane grants `m`: the default, or unmetered (`u64::MAX`) for
/// a module that uses fibers, whose metered run `fiber fuel` declines until #3080. The one place the suites
/// read it, so #3080's fix gives every fiber row its metered lane here.
#[doc(hidden)]
pub fn lane_fuel(m: &Module) -> u64 {
    match m.funcs.iter().any(temen_ir::Func::uses_fibers) {
        true => u64::MAX,
        false => temen_interp::DEFAULT_FUEL,
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

std::thread_local! {
    /// How many kernel runs this thread has opened ([`kernel_runs`]).
    static RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Test-facing: how many runs this thread has taken the kernel route for. The kernel runs its root on
/// the caller's thread, so a test reads, before and after an embedder's run, which route that run
/// took (`jit_run` falls back to the JIT's own runtime when [`admit`] declines, and says nothing).
/// Correct only while the run's root runs on the caller's thread: a change that moves it elsewhere
/// must move this count with it.
#[doc(hidden)]
pub fn kernel_runs() -> u64 {
    RUNS.with(std::cell::Cell::get)
}

/// Run the admitted program as `start` says, its vCPUs as the kernel's tasks, under `host`. `Err` is
/// the JIT's compile refusing it, with nothing run and `host` as it was.
pub fn run(
    admitted: Admitted<'_>,
    start: Start<'_>,
    host: &mut Host,
) -> Result<KernelEnd, temen_jit::JitError> {
    let m = admitted.0;
    RUNS.with(|r| r.set(r.get() + 1));
    let f = m
        .funcs
        .get(start.entry as usize)
        .ok_or(temen_jit::JitError::Unsupported("no such entry"))?;
    if f.params.len() != start.args.len() {
        return Err(temen_jit::JitError::Unsupported("entry arity"));
    }
    // Fuel checks are compiled in when the run's chain meters: its own grant, or a level above it.
    let metered = metered(host, start.fuel);
    // The compiled instance's cell, which no vCPU charges (each has its own, `tier_run`): it only
    // arms the compile's fuel checks, and lives as long as the code (`KernelRun::new`).
    let mut armed = metered.then(|| FuelCell::fixed(0));
    // Every kernel run is concurrent (`native_scope`'s `no §12 op`), so its calls take the locked
    // thunk alone (#2958 condition 3, met by routing in phase 1).
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
    let (kr, ro) = match KernelRun::new(cm, armed, init_mem, crate::run_node(host)) {
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
    // No thread runs guest code now: the run's fibers go, before its host goes back to the caller.
    kr.end_fibers();
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

    fn park_call(&mut self, call: &mut dyn std::any::Any) -> Result<[i64; 2], i64> {
        // What [`park_call`] sent; anything else is no call of this embedder's: the backstop.
        let park = (call.downcast_mut::<CallPayload>())
            .and_then(Option::take)
            .ok_or(Trap::Malformed.code())?;
        self.0.park_call(park).map_err(|t| t.code())
    }
}

/// #2958 slice 5 — what a capability call's park carries from the cap thunk to [`Hook`], through
/// `temen_jit`, which cannot name it: [`park_call`] sends it, and [`Hook::park_call`] takes it.
type CallPayload = Option<CallPark>;

/// #2958 slice 5 — hand `park`, what a call the vCPU on `vm` just made left it to do
/// ([`temen_interp::bytecode::call_park`]), to its task's rules ([`EmittedCall::park_call`]): `None`
/// when `vm` is no kernel vCPU's. A park in a fiber traps `Malformed` (`fiber parks`).
///
/// # Safety
/// As [`temen_jit::kernel_rt::park_call`]'s: the caller is the cap thunk of code running on `vm`, on
/// `vm`'s own thread, inside the guarded call its vCPU runs (`KernelRun::run_vcpu`), and holds no
/// other reference to the hook and no `Host` lock.
pub(crate) unsafe fn park_call(vm: &temen_jit::VmCtx, park: CallPark) -> Option<[i64; 2]> {
    // A rewound park or a punted call's parks its caller; a region post's wake does not.
    let parks = !matches!(park, CallPark::Done(_));
    let mut payload: CallPayload = Some(park);
    // SAFETY: this function's own contract is the callee's.
    unsafe { temen_jit::kernel_rt::park_call(vm, parks, &mut payload) }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A program `native_scope` declines (`jit cap`): a §22 `Jit.invoke`, which only the JIT's own
    /// runtime serves, beside a §12 op.
    const SERVES: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vq = i64.const 16400
  vone = i32.const 1
  vk = atomic.notify vq vone
  vz = i32.const 0
  vc = i64.const 0
  vn = call.cap 11 1 (i64) -> (i64) vz (vc)
  return vn
  }
}
"#;

    /// Run past `admit`, a declined program meets the kernel route's two guards: the native tier's
    /// root offer asserts `native_scope` (`NativeTier`'s contract) in a build with debug assertions,
    /// and in any other the call's thunk traps the domain `Malformed` (`kernel_backstop`) instead of
    /// reaching the JIT's own runtime.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "a native tier's image"))]
    fn a_declined_program_run_past_admit_meets_the_offers_assertion_or_the_backstop() {
        let m = temen_text::parse_module(SERVES).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let mut host = Host::new();
        assert_eq!(admit(&m, &host, false, u64::MAX).err(), Some("jit cap"));
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

    /// A threaded module whose fiber makes `park` (a body defining `vr`), run past `admit` granted
    /// no fuel: the kernel route's answer.
    fn a_fiber_parks_past_admit(park: &str, mut host: Host) -> Result<Vec<i64>, Trap> {
        let src = format!(
            "memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vts, vtv = thread.join vt
  vf = ref.func 2
  vk = cont.new vf vz
  vs, vv = cont.resume vk vz
  return vv
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  return va
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vz = i64.const 0
{park}
  return vr
  }}
}}
"
        );
        let m = temen_text::parse_module(&src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        assert_eq!(admit(&m, &host, false, u64::MAX).err(), Some("fiber parks"));
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
        run(Admitted(&m), start, &mut host)
            .expect("compiles")
            .result
    }

    /// #3078 slice 2 — the three backstops `fiber parks` keeps unreachable: a futex wait, a rewound
    /// call park (an empty pipe's read) and a punted call's park, each made in a fiber, trap the
    /// domain `Malformed` instead of parking the fiber's task. Run past `admit`, each meets the native
    /// tier's assertion first in a build with debug assertions.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "a native tier's image"))]
    fn a_futex_wait_in_a_fiber_run_past_admit_meets_the_backstop() {
        let wait = "  vq = i64.const 16400\n  vzw = i32.const 0\n  vto = i64.const -1\n  vw = i32.atomic.wait vq vzw vto\n  vr = i64.extend_i32_u vw";
        assert_eq!(
            a_fiber_parks_past_admit(wait, Host::new()),
            Err(Trap::Malformed)
        );
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "a native tier's image"))]
    fn a_pipe_read_in_a_fiber_run_past_admit_meets_the_backstop() {
        let mut host = Host::new();
        let (_, read) = host.grant_pipe();
        let park = format!("  vc = i32.const {read}\n  vr = call.cap 0 0 (i64) -> (i64) vc(vz)");
        assert_eq!(a_fiber_parks_past_admit(&park, host), Err(Trap::Malformed));
    }

    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "a native tier's image"))]
    fn a_punted_call_in_a_fiber_run_past_admit_meets_the_backstop() {
        let mut host = Host::new();
        let requests = temen_interp::CapRequests::default();
        let proc = host.grant_host_proc(temen_interp::HostProcGrant::declared("later", &requests));
        let park = format!("  vc = i32.const {proc}\n  vr = call.cap 13 0 (i64) -> (i64) vc(vz)");
        assert_eq!(a_fiber_parks_past_admit(&park, host), Err(Trap::Malformed));
    }
}
