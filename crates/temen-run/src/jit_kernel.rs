//! **#2948 spike — a Cranelift JIT program's vCPUs as tasks of executor 2's scheduling core.**
//! Scaffolding (INVARIANTS #13, #15), never a supported second route: it either becomes phase 1 or is
//! deleted. Off unless `TEMEN_JIT_KERNEL=1` ([`enabled`]); Linux x86-64 only.
//!
//! The run is executor 2's (`CoopRun::new_parallel`), with the Cranelift JIT as its emitted tier
//! ([`ThreadTier`]): the root runs whole as a leaf, and each `thread.spawn` child tiers up at its
//! entry, so every vCPU is a kernel task whose own thread runs Cranelift code from its entry. A §12 op
//! the code makes goes to the kernel ([`EmittedCall::park`]) through the JIT's per-thread hook
//! ([`temen_jit::kernel_rt`]), and the thread parks there with its frames on its stack.
//!
//! The window is the JIT's (guard-paged, reserved by [`KernelRun::new`]), handed to the kernel as a
//! shared region: the kernel seeds it and keys its futexes on it, then the JIT protects it.
//!
//! Scope: one root domain, one module, no fibers, no §14 children, not durable, no fuel metering.
//! Anything else is `None`, and the caller runs today's path.

use std::sync::Arc;
use temen_interp::bytecode::{
    CoopEvent, CoopRun, EmittedCall, EmittedTarget, NativeStop, NativeTier, ProgramEmit,
    ThreadPlatform, ThreadTier,
};
use temen_interp::{Host, Trap, Value};
use temen_ir::Module;
use temen_jit::kernel_rt::{KStop, KernelPark, KernelRun, VcpuEntry};
use temen_jit::CompiledModule;

/// How many vCPUs this process has run as kernel tasks: what a test reads to see that a run took the
/// spike path rather than today's.
static VCPUS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`VCPUS`].
pub fn vcpus() -> u64 {
    VCPUS.load(std::sync::atomic::Ordering::Relaxed)
}

/// #2868 prototype, for attribution: whether the runs this process opens next queue their native
/// futex waits by key ([`set_keyed`]), as `TEMEN_SPIKE_KEYED=1` does.
static KEYED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// [`KEYED`].
pub fn set_keyed(on: bool) {
    KEYED.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Whether the spike path is switched on (`TEMEN_JIT_KERNEL=1`).
pub fn enabled() -> bool {
    std::env::var_os("TEMEN_JIT_KERNEL").is_some_and(|v| v == "1")
}

/// Run `m`'s `entry(args)` on the Cranelift JIT with its vCPUs as executor 2's tasks, under `host`.
/// `None` when the program is outside the spike's scope: nothing ran, and `host` is as it was.
pub fn run(
    m: &Module,
    entry: u32,
    args: &[i64],
    reserved_log2: u8,
    host: &mut Host,
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Option<Result<Vec<i64>, Trap>> {
    temen_interp::bytecode::native_scope(m, host).ok()?;
    let cm = compile(m, entry, reserved_log2, kill)?;
    run_compiled(cm, m, entry, args, host, kill)
}

/// The spike's compile of `m` for [`run_compiled`]: `None` when `m` is outside its scope.
pub fn compile(
    m: &Module,
    entry: u32,
    reserved_log2: u8,
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Option<CompiledModule> {
    if m.funcs.iter().any(temen_ir::Func::uses_fibers) {
        return None;
    }
    CompiledModule::compile(
        m,
        entry,
        crate::cap_thunk_locked,
        std::ptr::null_mut(), // each vCPU's context names its domain's host cell
        reserved_log2,
        None, // no §14 children
        kill, // its running code polls the kill door; the run's end raises the domain's end
        None, // no fuel metering
        None,
        None,
        0,
    )
    .ok()
}

/// [`run`] over `cm`, `m`'s [`compile`] for `entry` with `kill`: one run, which takes the compile
/// (its window is the run's).
pub fn run_compiled(
    cm: CompiledModule,
    m: &Module,
    entry: u32,
    args: &[i64],
    host: &mut Host,
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Option<Result<Vec<i64>, Trap>> {
    // Every refusal is decided here, before anything is taken from `host`.
    let f = m.funcs.get(entry as usize)?;
    if f.params.len() != args.len() {
        return None;
    }
    temen_interp::bytecode::native_scope(m, host).ok()?;
    let kr = Arc::new(KernelRun::new(cm)?);
    let (base, reserved) = kr.window();
    // SAFETY: `kr` owns the reservation and is dropped at the end of this function, after the run
    // (whose threads have all gone by then, `ThreadRun::finish`) and its host's tier are dropped;
    // `back`'s strong count is checked to be 1 before then, and the window is kept mapped if it is
    // not, so no clone of the region outlives its memory.
    let back = Arc::new(unsafe { temen_interp::Region::shared(base, reserved) });
    let back_held = Arc::clone(&back);
    let ran = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let state: Arc<dyn std::any::Any + Send + Sync> = Arc::new(TierState {
        kr: Arc::clone(&kr),
        vcpus: Arc::clone(&ran),
    });
    let on_end = {
        let kr = Arc::clone(&kr);
        Arc::new(move || kr.end())
    };
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
                on_end,
                state,
                // #2868 prototype, for attribution only: `TEMEN_SPIKE_KEYED=1`.
                keyed_futex: KEYED.load(std::sync::atomic::Ordering::Relaxed)
                    || std::env::var_os("TEMEN_SPIKE_KEYED").is_some_and(|v| v == "1"),
            }),
        }),
        // The root runs on the caller's thread, as the JIT's does.
        root_on_opener: true,
        ..ThreadPlatform::OS
    };
    host.set_thread_platform(platform);
    // The kill door stops the kernel's run too, so its parked tasks unwind: the embedder's stop, as
    // the JIT's kill is (`OutOfFuel`).
    let door = Arc::new(temen_interp::bytecode::RunDoor::default());
    host.set_run_door(Arc::clone(&door));
    let _kill = kill.map(|k| {
        let door = Arc::clone(&door);
        // A busy run lock answers `false`: the stop is asked again (`RunDoor::stop`).
        temen_jit::kernel_rt::on_kill(k, Box::new(move || while !door.stop() {}))
    });
    let vals: Vec<Value> = f
        .params
        .iter()
        .zip(args)
        .map(|(t, a)| slot_value(*t, *a))
        .collect();
    let reserved_log2 = reserved.trailing_zeros() as u8;
    let opened = CoopRun::new_parallel(
        m,
        entry,
        &vals,
        temen_interp::DEFAULT_FUEL,
        std::mem::take(host),
        &[],
        reserved_log2,
        Some(back),
        false,
    );
    let mut run = match opened {
        Some(Ok(run)) => run,
        // A run its activation refused (the process's pools): an ordinary run ends so too. Its host
        // went with it, as `new_parallel` keeps it.
        Some(Err(t)) => return Some(Err(t)),
        // `native_scope` admitted the module, so the bytecode compile cannot refuse it.
        None => unreachable!("native_scope admits only what the bytecode engine compiles"),
    };
    // The kernel seeded the window; nothing has run. A protection the OS refuses ends the run as a
    // window it cannot hold does (`ThreadFault`), with its host given back.
    let end = if kr.protect() {
        match run.run() {
            CoopEvent::Done(vals) => Ok(vals.iter().map(|v| value_slot(*v)).collect()),
            CoopEvent::Trapped(t) => Err(t),
            _ => Err(Trap::Malformed),
        }
    } else {
        Err(Trap::ThreadFault)
    };
    *host = std::mem::take(&mut *run.host_mut());
    host.set_thread_platform(ThreadPlatform::OS);
    drop(run);
    LAST_VCPUS.with(|c| c.set(ran.load(std::sync::atomic::Ordering::Relaxed)));
    // Nothing the run handed back may still reach the window, which `kr` unmaps as it drops.
    if Arc::strong_count(&back_held) != 1 {
        debug_assert!(false, "a clone of the JIT window outlived its run");
        std::mem::forget(kr); // keep it mapped
    }
    Some(end)
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

/// [`outcome`]'s run, with a fresh host: `None` when [`run`] declines.
fn outcome_unchecked(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Option<temen_jit::JitOutcome> {
    use temen_jit::{JitOutcome, TrapKind};
    let mut host = Host::new();
    Some(
        match run(
            m,
            entry,
            args,
            temen_ir::DEFAULT_RESERVED_LOG2,
            &mut host,
            kill,
        )? {
            Ok(v) => JitOutcome::Returned(v),
            Err(Trap::Exit(code)) => JitOutcome::Exited(code),
            Err(t) => JitOutcome::Trapped(
                TrapKind::from_code(t.code() as u32).unwrap_or(TrapKind::ThreadFault),
            ),
        },
    )
}

/// The tier's run state ([`NativeTier::state`]): the JIT's half of the run, and how many vCPUs ran.
struct TierState {
    kr: Arc<KernelRun>,
    vcpus: Arc<std::sync::atomic::AtomicU64>,
}

thread_local! {
    /// How many vCPUs the last run this thread made ran as kernel tasks ([`last_vcpus`]).
    static LAST_VCPUS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// How many vCPUs the last [`run`] this thread made ran as kernel tasks.
pub fn last_vcpus() -> u64 {
    LAST_VCPUS.with(std::cell::Cell::get)
}

/// #2948 spike, for tests: under [`enabled`], run `m`'s `entry(args)` on the spike path with a fresh
/// host and answer as the JIT's own entries do. `None` when off, or when `m` is outside the scope
/// ([`temen_interp::bytecode::native_scope`], whose clause it names on stderr): the caller runs
/// today's path. An in-scope program is checked to have run on the kernel path, every vCPU native:
/// it never falls back silently.
pub fn outcome(
    m: &Module,
    entry: u32,
    args: &[i64],
    kill: Option<&Arc<temen_jit::Interrupt>>,
) -> Option<temen_jit::JitOutcome> {
    if !enabled() {
        return None;
    }
    if let Err(clause) = temen_interp::bytecode::native_scope(m, &Host::new()) {
        eprintln!("#2948 spike: today's path ({clause})");
        return None;
    }
    let o =
        outcome_unchecked(m, entry, args, kill).expect("an in-scope program runs as kernel tasks");
    assert!(last_vcpus() >= 1, "the run's vCPUs ran as kernel tasks");
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "no interpreted op on the spike path"
    );
    eprintln!(
        "#2948 spike: kernel path ({} vCPUs, all native)",
        last_vcpus()
    );
    Some(o)
}

/// [`ThreadTier::run`] for the Cranelift JIT: run the task's entry on this thread as a vCPU.
fn tier_run(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    VCPUS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    let entry = if call.root {
        VcpuEntry::Root(&argv)
    } else {
        VcpuEntry::Thread {
            func,
            arg0: argv.first().copied().unwrap_or(0),
            arg1: argv.get(1).copied().unwrap_or(0),
        }
    };
    let (tls, host) = (call.tls, call.host_cell());
    let ran = {
        let mut park = Park(call);
        // SAFETY: `host` is the task's domain's live host cell, which `cap_thunk_locked` takes; the
        // kernel seeded the window and `run` protected it before any task ran.
        unsafe { kr.run_vcpu(entry, tls, host as *mut core::ffi::c_void, &mut park) }
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
struct Park<'c, 'a>(&'c mut EmittedCall<'a>);

impl KernelPark for Park<'_, '_> {
    fn park(&mut self, stop: KStop) -> Result<[i64; 2], i64> {
        let stop = match stop {
            KStop::Spawn { func, arg0, arg1 } => NativeStop::Spawn { func, arg0, arg1 },
            KStop::Join { handle } => NativeStop::Join { handle },
            KStop::Wait {
                off,
                expected,
                wide,
                timeout,
            } => NativeStop::Wait {
                addr: off,
                expected,
                wide,
                timeout,
            },
            KStop::Notify { off, count } => NativeStop::Notify { addr: off, count },
        };
        self.0.park(stop).map_err(|t| t.code())
    }
}
