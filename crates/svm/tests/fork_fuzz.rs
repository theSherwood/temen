//! Stable seed-sweep counterpart to the `fork_diff` libFuzzer target (FORK.md §9 / OPS_PARITY.md).
//!
//! The `fork_diff` fuzz target only runs on the nightly `cargo-fuzz` job; this drives the *same*
//! differential harness ([`fork_fuzz::fuzz_one`]) over a deterministic sweep of reply/status values so
//! the bytecode-fork==oracle property is gated on **every** CI run, not just nightly — the discipline
//! `jit_fuzz`/`fiber_fuzz` follow. Each iteration builds the `SRC_TWIN` fork topology with the chosen
//! replies and asserts the bytecode cooperative-drive run matches the tree-walk oracle (return value +
//! sorted stdout multiset). A regression in the bytecode twin/reap machinery surfaces here.

#[path = "support/fork_fuzz.rs"]
mod fork_fuzz;

/// Tiny deterministic PRNG (xorshift64*) — mirrors `fiber_fuzz`/`fuzz_smoke`, no external deps.
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
}

#[test]
fn bytecode_fork_matches_the_oracle_over_a_reply_sweep() {
    // A handful of adversarial fixed pairs (sentinels, zero, negatives, equal-before-sanitize) plus a
    // deterministic random sweep. `fuzz_one` sanitizes replies to be distinct and clear of the reap
    // sentinels, so every pair drives a real fork+wait round-trip.
    let fixed: &[(i64, i64)] = &[
        (0, 1),
        (100, 200),
        (-10, -11),
        (-11, -10),
        (5, 5),
        (i64::MIN, i64::MAX),
        (-1, 0),
        (42, -10),
    ];
    for (ro, rt) in fixed {
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&ro.to_le_bytes());
        bytes.extend_from_slice(&rt.to_le_bytes());
        fork_fuzz::fuzz_one(&bytes);
    }

    let mut rng = Rng(0x5eed_f00d_1c0f_fee5);
    for _ in 0..96 {
        let mut bytes = [0u8; 16];
        bytes[0..8].copy_from_slice(&rng.next_u64().to_le_bytes());
        bytes[8..16].copy_from_slice(&rng.next_u64().to_le_bytes());
        fork_fuzz::fuzz_one(&bytes);
    }
}
