//! Unified Temen runtime assembly crate (`#3187` / `#3196`).
//!
//! `temen-runtime` sits at the top of the acyclic runtime DAG:
//! `temen-kernel` (`#![forbid(unsafe_code)]`) -> engine crates (`temen-bytecode`, future Stage 1d
//! `temen-jit` / `temen-wasm-jit` / `OpOracle`) -> `temen-runtime`.
//!
//! It owns:
//! - The closed [`Engine`] dispatch enum (zero monomorphization bloat, mixed-engine domains).
//! - One [`driver::DomainWorld`] shared by both [`RunMode::Cooperative`] (single-threaded /
//!   wasm64-ready) and [`RunMode::Parallel`] (1:1 OS threads) Carriers (Invariant 15).
//! - Carrier-owned [`FuelCell`] chunking/refunding (Decision 30) and uniform `Context` execution
//!   across `Root`, `Thread`, and migratable `GuestFiber` (`FIBER_PARKED` status 3).

#![forbid(unsafe_code)]

pub mod driver;

use std::sync::Arc;
use temen_bytecode::{compile_module, Code, CompileError};
use temen_ir::Module;
use temen_kernel::{
    CapOutcome, ContextExit, ContextId, DomainId, LifecycleHooks, Limits, NoopLifecycleHooks, Trap,
};
use temen_mem::Region;
use temen_verify::{verify_module, VerifyError};

pub use driver::{DomainWorld, RunOutput};
pub use temen_kernel;

/// Closed set of execution engines (Decision 5).
///
/// Stage 1c wires `Engine::Bytecode`; Stage 1d (`#3197`) adds `#[cfg(feature = "cranelift")]
/// Cranelift(...)` on this same enum and `DomainWorld`.
#[derive(Clone)]
pub enum Engine {
    Bytecode(Arc<Code>),
}

/// How Carriers execute concurrent contexts in a domain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RunMode {
    /// Single-threaded cooperative round-robin Carrier (works identically on native and wasm64).
    Cooperative,
    /// Multi-threaded 1:1 OS-thread Carriers for `thread.spawn` sharing one `Domain` & `WaitIndex`.
    #[cfg(not(target_arch = "wasm32"))]
    Parallel,
}

/// Host capability dispatch hook for `call.cap` (Stage 2 plugs in the full powerbox & offer wire).
pub trait CapHandler: Send + Sync {
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        domain: DomainId,
        ctx: ContextId,
        handle: i32,
        type_id: u32,
        op: u32,
        args: &[i64],
        mem: Option<&Region>,
    ) -> CapOutcome;
}

/// Default capability handler when no capabilities are granted: fails closed with `CapFault`.
pub struct NoopCapHandler;

impl CapHandler for NoopCapHandler {
    fn call(
        &self,
        _domain: DomainId,
        _ctx: ContextId,
        _handle: i32,
        _type_id: u32,
        _op: u32,
        _args: &[i64],
        _mem: Option<&Region>,
    ) -> CapOutcome {
        CapOutcome::Trap(Trap::CapFault)
    }
}

/// Configuration for running a compiled [`Engine`] module in a [`DomainWorld`].
pub struct RunConfig {
    /// 3-dimension durable budget ceiling (`mem`, `fuel`, `tasks`) + D66 `lanes`.
    pub limits: Limits,
    /// Fuel chunk drawn per Carrier from `BudgetTree` (refunded on yield/park/exit — Decision 30).
    pub fuel_chunk: u64,
    /// Cooperative safepoint quantum before round-robin handoff (`RunMode::Cooperative`).
    pub quantum: u32,
    /// Execution Carrier mode (`Cooperative` or `Parallel`).
    pub mode: RunMode,
    /// 4 OS-neutral lifecycle hooks (default: [`NoopLifecycleHooks`]).
    pub hooks: Arc<dyn LifecycleHooks>,
    /// Capability handler for `call.cap` (default: [`NoopCapHandler`]).
    pub caps: Arc<dyn CapHandler>,
}

impl Default for RunConfig {
    fn default() -> RunConfig {
        RunConfig {
            limits: Limits::default(),
            fuel_chunk: 4096,
            quantum: 256,
            mode: RunMode::Cooperative,
            hooks: Arc::new(NoopLifecycleHooks),
            caps: Arc::new(NoopCapHandler),
        }
    }
}

/// Failure when verifying or compiling a [`Module`] into an [`Engine`].
#[derive(Debug)]
pub enum BuildError {
    Verify(VerifyError),
    Compile(CompileError),
}

impl std::fmt::Display for BuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuildError::Verify(e) => write!(f, "verify error: {e:?}"),
            BuildError::Compile(e) => write!(f, "compile error: {e}"),
        }
    }
}

impl std::error::Error for BuildError {}

/// Verify `module` and compile it to `Engine::Bytecode`.
pub fn compile_bytecode(module: &Module) -> Result<Engine, BuildError> {
    verify_module(module).map_err(BuildError::Verify)?;
    let code = compile_module(module).map_err(BuildError::Compile)?;
    Ok(Engine::Bytecode(Arc::new(code)))
}

/// Run `engine`'s function `entry_func` with `args` under `cfg` and return its [`RunOutput`].
pub fn run(engine: Engine, entry_func: u32, args: &[i64], cfg: RunConfig) -> RunOutput {
    driver::run_entry(engine, entry_func, args, cfg)
}

/// Verify, compile to `Engine::Bytecode`, and run `entry_func(args)` with `cfg`.
pub fn run_module(
    module: &Module,
    entry_func: u32,
    args: &[i64],
    cfg: RunConfig,
) -> Result<RunOutput, BuildError> {
    let engine = compile_bytecode(module)?;
    Ok(run(engine, entry_func, args, cfg))
}

/// Convenience helper: check if a [`ContextExit`] returned `vals`.
pub fn exit_values(exit: &ContextExit) -> Option<&[i64]> {
    match exit {
        ContextExit::Returned(v) => Some(v),
        _ => None,
    }
}
