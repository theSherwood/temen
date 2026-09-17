//! libFuzzer target for the **trap-time backtrace walk's fault recovery** (DEBUGGING.md §5 W3, #1487).
//!
//! The walk follows the frame-pointer chain at a trap to collect return addresses. Its input is the
//! **guest's own stack**, so the guest chooses every byte of it — which is exactly why the loop's
//! bounds (aligned, non-null, strictly increasing, an 8 MiB span, a 64-frame cap) constrain the
//! arithmetic and can say nothing about whether a link is *mapped*. It used to be possible to steer
//! the walk onto an unmapped page and, because on unix it runs from the SIGSEGV handler after that
//! handler has disarmed, take the whole host process down with it.
//!
//! The recovery bracket makes that survivable. This fuzzes the property the bracket exists for,
//! against arbitrary chain contents:
//!
//! 1. **The host survives.** Any fault inside the walk is recovered; nothing reaches `SIG_DFL`.
//! 2. **It terminates.** Strictly-increasing links plus the span and frame bounds mean no cycle and
//!    no unbounded run, whatever the fuzzer writes.
//! 3. **The capture stays bounded and consistent** — never more than `TEMEN_TRAP_MAXFRAMES`, and
//!    always at least the trap site the helper records before walking anything.
//!
//! The chain lives in a plain heap buffer and the fuzzer supplies the words, so most links point at
//! addresses that are not mapped at all — which is the case that matters. libFuzzer installs its own
//! SIGSEGV handler at startup, so ours captures it as the previous disposition and chains to it for a
//! *genuine* fault; a fault the walk itself raises stands down before that, and never reaches it.
//!
//! Run: `cargo +nightly fuzz run trap_walk`
#![no_main]

use libfuzzer_sys::fuzz_target;

/// Matches `TEMEN_TRAP_MAXFRAMES` in `crates/temen-jit/src/trap_capture.c`.
const TRAP_MAXFRAMES: usize = 64;

fuzz_target!(|data: &[u8]| {
    // Words the fuzzer controls, laid out as `{ saved_fp, ret_addr }` records. Two words minimum so
    // there is a frame to read at all; the cap keeps an input from being mostly memset.
    if data.len() < 16 {
        return;
    }
    let n_words = (data.len() / 8).min(4096);
    let mut chain: Vec<usize> = data[..n_words * 8]
        .chunks_exact(8)
        .map(|c| usize::from_le_bytes(c.try_into().unwrap()))
        .collect();

    // Rebase a *prefix* of the links onto the buffer itself, so some inputs produce a genuinely
    // walkable chain (and exercise the ordinary path and the frame cap) instead of every input
    // faulting on its first link. The rest stay whatever the fuzzer chose — usually unmapped.
    let base = chain.as_ptr() as usize;
    let walkable = (chain[0] % chain.len().max(1)) / 2;
    for i in 0..walkable.min(chain.len() / 2) {
        let target = base + (i + 1) * 16;
        chain[i * 2] = target;
    }

    let fp = match chain[1] % 4 {
        // From the buffer (the realistic case: a live frame pointer).
        0 | 1 => base,
        // Mid-buffer, possibly unaligned — the alignment guard should end it immediately.
        2 => base + (chain[0] % (n_words * 8)),
        // Wholly arbitrary, usually unmapped: the walk must fault on the very first deref and
        // recover, yielding just the trap site.
        _ => chain[0],
    };

    // SAFETY: the walk's faults are recovered by its own bracket, which is what this target gates.
    let rets = unsafe { temen_jit::walk_trap_frame_chain(fp) };

    assert!(
        !rets.is_empty(),
        "the capture always records the trap site before walking"
    );
    assert!(
        rets.len() <= TRAP_MAXFRAMES,
        "walk exceeded the frame cap: {} > {TRAP_MAXFRAMES}",
        rets.len()
    );
});
