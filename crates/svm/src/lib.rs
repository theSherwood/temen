//! Umbrella crate: re-exports the Phase-1 components and ties them into the
//! end-to-end pipeline `text -> binary -> verify -> interp` (`DESIGN.md` §18).
//!
//! `load` mirrors the start of the instantiation contract (§3b): decode, then
//! verify, **fail-closed** — only a module that passes both is runnable.
#![forbid(unsafe_code)]

pub use svm_encode as encode;
pub use svm_interp as interp;
pub use svm_ir as ir;
pub use svm_text as text;
pub use svm_verify as verify;

/// The embedding runtime — instantiate a verified module with the powerbox and run it. Behind the
/// off-by-default `run` feature (#918) so the umbrella crate's default build stays the dependency-free
/// pipeline core; `svm_run` itself re-exports `parse_module`/`decode_module`/`verify_module`, so an
/// embedder that enables `run` needs only this one crate.
#[cfg(feature = "run")]
pub use svm_run as run;

use svm_interp::Value;
use svm_ir::{FuncIdx, Module, ValType};

/// Any failure along the pipeline — a compile-time reject (parse/decode/verify) **or** a runtime
/// [`Trap`](svm_interp::Trap) from running the verified module.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Error {
    Parse(svm_text::ParseError),
    Decode(svm_encode::DecodeError),
    Verify(svm_verify::VerifyError),
    Trap(svm_interp::Trap),
}

impl From<svm_text::ParseError> for Error {
    fn from(e: svm_text::ParseError) -> Self {
        Error::Parse(e)
    }
}
impl From<svm_encode::DecodeError> for Error {
    fn from(e: svm_encode::DecodeError) -> Self {
        Error::Decode(e)
    }
}
impl From<svm_verify::VerifyError> for Error {
    fn from(e: svm_verify::VerifyError) -> Self {
        Error::Verify(e)
    }
}
impl From<svm_interp::Trap> for Error {
    fn from(t: svm_interp::Trap) -> Self {
        Error::Trap(t)
    }
}

/// Parse text and encode it to the binary form.
pub fn assemble(src: &str) -> Result<Vec<u8>, Error> {
    let m = svm_text::parse_module(src)?;
    Ok(svm_encode::encode_module(&m))
}

/// Decode **and verify** a binary module (the runnable precondition, §3b).
pub fn load(bytes: &[u8]) -> Result<Module, Error> {
    let m = svm_encode::decode_module(bytes)?;
    svm_verify::verify_module(&m)?;
    Ok(m)
}

/// Convenience: assemble, load (decode+verify), and run a function.
pub fn run_text(src: &str, func: FuncIdx, args: &[Value], fuel: u64) -> Result<Vec<Value>, Error> {
    let bytes = assemble(src)?;
    let m = load(&bytes)?;
    let mut fuel = fuel;
    // A verified module that traps is a real outcome — surface it as `Error::Trap` rather than
    // swallowing it into an empty result (a trap-swallowing example API teaches the wrong contract).
    Ok(svm_interp::run_fast(&m, func, args, &mut fuel)?)
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
