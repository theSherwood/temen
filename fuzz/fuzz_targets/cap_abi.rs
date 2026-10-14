//! libFuzzer target for the **capability ABI** (#2353): coverage-guided bytes become a guest that
//! makes a random sequence of `call.cap`s against a powerbox holding each capability a default one
//! grants, run on the tree-walker and the bytecode engine. A host panic, an abort, or one host
//! allocation past `ALLOC_LIMIT` is a find: no argument a guest chooses may crash its host
//! (INVARIANTS #3, #5). The stable `cap_abi` test drives the same `fuzz_one` from seeds, and
//! `cap_abi::guest` prints the program an input becomes.
//!
//! Run: `cargo +nightly fuzz run cap_abi`
#![no_main]

use std::time::{Duration, Instant};

use libfuzzer_sys::fuzz_target;

#[path = "../../crates/temen/tests/support/cap_abi.rs"]
mod cap_abi;

#[global_allocator]
static ALLOC: cap_abi::Bounded = cap_abi::Bounded;

fuzz_target!(|data: &[u8]| {
    let threads = live_threads();
    let _ = cap_abi::fuzz_one(data);
    // The POSIX personality fires a cross-process wake from a thread it detaches. libFuzzer checks
    // each input for leaks as it returns, and a thread still tearing down its thread-locals then
    // reads as one, so the input waits for every thread its run started to exit.
    let deadline = Instant::now() + Duration::from_secs(10);
    while live_threads() > threads {
        assert!(
            Instant::now() < deadline,
            "{} threads the run started outlived it by 10 s",
            live_threads() - threads
        );
        std::thread::yield_now();
    }
});

/// The process's threads: each has an entry until it has exited, its thread-locals torn down.
fn live_threads() -> usize {
    std::fs::read_dir("/proc/self/task").map_or(0, Iterator::count)
}
