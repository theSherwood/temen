//! #2353 — the capability-ABI fuzz harness (`support/cap_abi.rs`) driven from seeds on stable: the
//! same `fuzz_one` the `cap_abi` fuzz target runs, so the test suite keeps what the fuzzer finds
//! fixed, and checks that the generated guests reach every capability in the powerbox.

#[path = "support/cap_abi.rs"]
mod cap_abi;

#[global_allocator]
static ALLOC: cap_abi::Bounded = cap_abi::Bounded;

/// `n` inputs from a fixed xorshift64* stream, each 64 to 1 KiB long.
fn seeds(n: usize) -> impl Iterator<Item = Vec<u8>> {
    let mut x: u64 = 0x2353_cafe_f00d_d00d;
    let mut next = move || {
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        x.wrapping_mul(0x2545_f491_4f6c_dd1d)
    };
    (0..n).map(move |_| {
        let len = 64 + next() as usize % 960;
        (0..len).map(|_| next() as u8).collect()
    })
}

/// Every seed's guest runs to an answer, a trap or a refusal on both engines, and the host survives
/// it. A panic names the seed and prints its guest; an abort (a refused allocation) does not unwind,
/// so `CAP_ABI_TRACE=1` prints each seed's number before it runs.
#[test]
fn random_capability_calls_never_crash_the_host() {
    let (mut completed, mut runs) = (0, 0);
    let mut answered = std::collections::BTreeSet::new();
    for (i, seed) in seeds(1500).enumerate() {
        if std::env::var_os("CAP_ABI_TRACE").is_some() {
            eprintln!("seed {i}");
        }
        let r = std::panic::catch_unwind(|| cap_abi::fuzz_one(&seed)).unwrap_or_else(|e| {
            eprintln!("seed {i} panicked; its guest:\n{}", cap_abi::guest(&seed));
            std::panic::resume_unwind(e)
        });
        runs += r.completed + r.trapped;
        completed += r.completed;
        // A result at all is an answer: a call the host refused would have trapped the run.
        answered.extend(r.results.iter().map(|(t, _)| *t));
    }
    assert!(
        completed * 10 >= runs,
        "{completed} of {runs} runs completed"
    );
    // What random seeds reach. A spawn takes a record naming live handles, which the coverage-guided
    // corpus builds and random bytes rarely do.
    for t in [
        temen_ir::cap_id::STREAM,
        temen_ir::cap_id::ADDRESS_SPACE,
        temen_ir::cap_id::BUDGET,
        temen_ir::cap_id::CLOCK,
        temen_ir::cap_id::MODULE_LOADER,
        temen_ir::cap_id::HOST_PROC,
        temen_ir::cap_id::JIT,
        temen_ir::CAP_SELF_TYPE_ID,
    ] {
        assert!(
            answered.contains(&t),
            "no call of type {t} answered: {answered:?}"
        );
    }
}
