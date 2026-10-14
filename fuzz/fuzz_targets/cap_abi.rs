//! libFuzzer target for the **capability ABI** (#2353): coverage-guided bytes become a guest that
//! makes a random sequence of `call.cap`s against a powerbox holding each capability a default one
//! grants, run on the tree-walker and the bytecode engine. A host panic, an abort, or one host
//! allocation past `ALLOC_LIMIT` is a find: no argument a guest chooses may crash its host
//! (INVARIANTS #3, #5). The stable `cap_abi` test drives the same `fuzz_one` from seeds, and
//! `cap_abi::guest` prints the program an input becomes.
//!
//! Run: `cargo +nightly fuzz run cap_abi`
#![no_main]

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/cap_abi.rs"]
mod cap_abi;

#[global_allocator]
static ALLOC: cap_abi::Bounded = cap_abi::Bounded;

fuzz_target!(|data: &[u8]| {
    let _ = cap_abi::fuzz_one(data);
});
