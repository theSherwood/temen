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
) -> Option<Result<Vec<i64>, Trap>> {
    if host.is_durable() || m.funcs.iter().any(temen_ir::Func::uses_fibers) {
        return None;
    }
    let f = m.funcs.get(entry as usize)?;
    if f.params.len() != args.len() {
        return None;
    }
    let cm = CompiledModule::compile(
        m,
        entry,
        crate::cap_thunk_locked,
        std::ptr::null_mut(), // each vCPU's context names its domain's host cell
        reserved_log2,
        None, // no §14 children
        None, // no interrupt: the run's end raises the domain's end instead
        None, // no fuel metering
        None,
        None,
        0,
    )
    .ok()?;
    let kr = Arc::new(KernelRun::new(cm)?);
    let (base, reserved) = kr.window();
    // SAFETY: the JIT window outlives the run: `kr` is held by the platform's tier, which the host
    // holds until the run has ended and every thread has gone.
    let back = Arc::new(unsafe { temen_interp::Region::shared(base, reserved) });
    let state: Arc<dyn std::any::Any + Send + Sync> = Arc::new(Arc::clone(&kr));
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
            native: Some(NativeTier { on_end, state }),
        }),
        // The root runs on the caller's thread, as the JIT's does.
        root_on_opener: true,
        ..ThreadPlatform::OS
    };
    host.set_thread_platform(platform);
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
        // `new_parallel` consumed the host: an out-of-subset module is not one this spike can give
        // back, so it is refused as a run, never silently re-run.
        Some(Err(t)) => return Some(Err(t)),
        None => return Some(Err(Trap::Malformed)),
    };
    if !kr.protect() {
        return Some(Err(Trap::ThreadFault));
    }
    let end = match run.run() {
        CoopEvent::Done(vals) => Ok(vals.iter().map(|v| value_slot(*v)).collect()),
        CoopEvent::Trapped(t) => Err(t),
        _ => Err(Trap::Malformed),
    };
    *host = std::mem::take(&mut *run.host_mut());
    // The tier holds the run's window: it goes with the run.
    host.set_thread_platform(ThreadPlatform::OS);
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

/// [`ThreadTier::run`] for the Cranelift JIT: run the task's entry on this thread as a vCPU.
fn tier_run(call: &mut EmittedCall<'_>) -> Result<(), Trap> {
    VCPUS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let kr = call
        .native()
        .and_then(|n| n.state.downcast_ref::<Arc<KernelRun>>())
        .ok_or(Trap::Malformed)?
        .clone();
    let EmittedTarget::Region { module: 0, func } = call.target else {
        return Err(Trap::Malformed);
    };
    let argv: Vec<i64> = call.argv.to_vec();
    // The root is the kernel's task 0, whose `vcpu.tls` is 0; a thread's starts at 1 (#1775).
    let entry = if call.tls == 0 {
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
