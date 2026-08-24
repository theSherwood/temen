//! libFuzzer target: fork **oracle-vs-bytecode** differential (FORK.md §9 / OPS_PARITY.md).
//!
//! Native fork runs on the bytecode cooperative `drive`; the input bytes pick the `clone_caller`
//! reply / reaped-status values, and the harness builds the `SRC_TWIN` fork topology with them, runs
//! it on the tree-walk oracle and the bytecode engine, and asserts they agree on the return value +
//! the sorted shared-stdout multiset. A crash here is a bytecode-fork miscompile (or a harness bug).
//! Hardens the single hand-written SRC_TWIN pin across a value range.
//!
//! Run: `cargo +nightly fuzz run fork_diff`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/fork_fuzz.rs"]
mod fork_fuzz;

fuzz_target!(|data: &[u8]| {
    fork_fuzz::fuzz_one(data);
});
