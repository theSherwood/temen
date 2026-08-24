//! Umbrella crate: re-exports the Phase-1 components and ties them into the
//! end-to-end pipeline `text -> binary -> verify -> interp` (`DESIGN.md` §18).
//!
//! `load` mirrors the start of the instantiation contract (§3b): decode, then
//! verify, **fail-closed** — only a module that passes both is runnable.
#![forbid(unsafe_code)]

pub use temen_encode as encode;
pub use temen_interp as interp;
pub use temen_ir as ir;
pub use temen_text as text;
pub use temen_verify as verify;

/// The embedding runtime — instantiate a verified module with the powerbox and run it. Behind the
/// off-by-default `run` feature (#918) so the umbrella crate's default build stays the dependency-free
/// pipeline core; `temen_run` itself re-exports `parse_module`/`decode_module`/`verify_module`, so an
/// embedder that enables `run` needs only this one crate.
#[cfg(feature = "run")]
pub use temen_run as run;

use temen_interp::Value;
use temen_ir::{FuncIdx, Module, ValType};

/// Any failure along the pipeline — a compile-time reject (parse/decode/verify) **or** a runtime
/// [`Trap`](temen_interp::Trap) from running the verified module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    Parse(temen_text::ParseError),
    Decode(temen_encode::DecodeError),
    Verify(temen_verify::VerifyError),
    Trap(temen_interp::Trap),
}

impl From<temen_text::ParseError> for Error {
    fn from(e: temen_text::ParseError) -> Self {
        Error::Parse(e)
    }
}
impl From<temen_encode::DecodeError> for Error {
    fn from(e: temen_encode::DecodeError) -> Self {
        Error::Decode(e)
    }
}
impl From<temen_verify::VerifyError> for Error {
    fn from(e: temen_verify::VerifyError) -> Self {
        Error::Verify(e)
    }
}
impl From<temen_interp::Trap> for Error {
    fn from(t: temen_interp::Trap) -> Self {
        Error::Trap(t)
    }
}

/// Parse text and encode it to the binary form.
pub fn assemble(src: &str) -> Result<Vec<u8>, Error> {
    let m = temen_text::parse_module(src)?;
    Ok(temen_encode::encode_module(&m))
}

/// Decode **and verify** a binary module (the runnable precondition, §3b).
pub fn load(bytes: &[u8]) -> Result<Module, Error> {
    let m = temen_encode::decode_module(bytes)?;
    temen_verify::verify_module(&m)?;
    Ok(m)
}

/// Convenience: assemble, load (decode+verify), and run a function.
pub fn run_text(src: &str, func: FuncIdx, args: &[Value], fuel: u64) -> Result<Vec<Value>, Error> {
    let bytes = assemble(src)?;
    let m = load(&bytes)?;
    let mut fuel = fuel;
    // A verified module that traps is a real outcome — surface it as `Error::Trap` rather than
    // swallowing it into an empty result (a trap-swallowing example API teaches the wrong contract).
    Ok(temen_interp::run_fast(&m, func, args, &mut fuel)?)
}

/// A zeroed value of each parameter type — handy for fuzzing/driving arbitrary funcs.
pub fn default_args(params: &[ValType]) -> Vec<Value> {
    params
        .iter()
        .map(|t| match t {
            ValType::I32 => Value::I32(0),
            ValType::I64 => Value::I64(0),
            ValType::F32 => Value::F32(0.0),
            ValType::F64 => Value::F64(0.0),
            ValType::V128 => Value::V128([0; 16]),
            ValType::Ref => Value::Ref(0),
            ValType::Cap => Value::I32(0),
        })
        .collect()
}
