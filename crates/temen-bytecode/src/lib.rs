//! Pure stackless bytecode compiler and execution engine (`Engine::Bytecode`, `#3187` / `#3195`).
//!
//! `temen-bytecode` compiles a verified [`temen_ir::Module`] into a flat register-window opcode
//! stream and steps any [`temen_kernel::Context`] over a shared [`temen_mem::Region`] +
//! [`temen_mask::Window`], returning [`temen_kernel::Step`] at every kernel boundary
//! (futex wait/notify, thread spawn/join, fiber new/resume/suspend, capability call, fuel refill,
//! poll safepoint, or completion).
//!
//! It is `#![forbid(unsafe_code)]`, wasm32/wasm64-clean, and carries no scheduler, host table,
//! or OS-thread code — all of those live once in `temen-kernel` + `temen-runtime` (Invariant 15).

#![forbid(unsafe_code)]

pub mod compile;
pub mod ops;
pub mod vm;

pub use compile::{compile_module, Code, CompileError, CompiledFunc};
pub use ops::Reg;
pub use vm::{step, ExecState, MemView, StepConfig};
