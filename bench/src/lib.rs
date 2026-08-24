//! Shared partial-evaluation harness for the peval benches: the interpreter-agnostic loop-cell
//! discovery driver (`capture`) and the zero-config Lua auto-rolled residual builder (`futamura`).
//! Copied from the proven `crates/temen-llvm/tests/{peval_capture,lua_rolled}` drivers so the
//! out-of-workspace bench (which links Wasmtime) can reuse them; the tests remain the correctness
//! source of truth. TODO: dedup into a shared `crates/temen-futamura` crate.

pub mod capture;
pub mod futamura;
