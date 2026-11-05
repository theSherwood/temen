//! Indexed work shared among threads (#2596): the bodies a module's decode and its verify take one
//! function at a time. One loop serves one thread or many, so a build without threads, or a host
//! with one core, runs the same code in order.
//!
//! The threads come with the `std` feature. Without it the crate stays `no_std`, as the in-sandbox
//! builds of the crates over it need, and the calling thread runs every item in order.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering::Relaxed};

/// `each(state, i)` for every `i` in `0..n`, shared among up to `threads` threads, each thread with
/// its own `state` from `init`. The results come back in index order. On a failure, the error is
/// that of the lowest failing `i`, the one an in-order loop meets first: each thread takes the next
/// `i` in order and stops at its first failure, and none takes an `i` past a failure already
/// recorded. Every recorded failure is at or above the lowest, so every `i` below it runs.
pub fn try_each<S, T: Send, E: Send>(
    n: usize,
    threads: usize,
    init: impl Fn() -> S + Sync,
    each: impl Fn(&mut S, usize) -> Result<T, E> + Sync,
) -> Result<Vec<T>, E> {
    let (next, failed) = (AtomicUsize::new(0), AtomicUsize::new(usize::MAX));
    // One thread's share: the items it took, in order, and its first failure.
    let share = || {
        let (mut state, mut done) = (init(), Vec::new());
        loop {
            let i = next.fetch_add(1, Relaxed);
            if i >= n || i > failed.load(Relaxed) {
                return (done, None);
            }
            match each(&mut state, i) {
                Ok(t) => done.push((i, t)),
                Err(e) => {
                    // A store, not `fetch_min`: the LLVM on-ramp, which builds this crate into
                    // guests (temen-link), has no `atomicrmw umin`. A higher failure stored over a
                    // lower one only lets the other threads run further.
                    failed.store(i, Relaxed);
                    return (done, Some((i, e)));
                }
            }
        }
    };
    let shares = run(threads.min(n), &share);
    let mut error: Option<(usize, E)> = None;
    let mut results = Vec::new();
    for (done, failure) in shares {
        match (failure, &error) {
            (Some((i, _)), Some((j, _))) if i > *j => {}
            (Some(failure), _) => error = Some(failure),
            (None, _) => {}
        }
        results.extend(done);
    }
    if let Some((_, e)) = error {
        return Err(e);
    }
    // Each share is in order already, and the stable sort merges sorted runs: one per thread. The
    // list is exactly its length.
    results.sort_by_key(|&(i, _)| i);
    let mut ordered = Vec::with_capacity(results.len());
    ordered.extend(results.into_iter().map(|(_, t)| t));
    Ok(ordered)
}

/// The threads a job gets: the host's, when `enough` work repays starting them, else one. A host
/// that reports no parallelism (wasm), or a build without threads, gets one.
pub fn threads(enough: bool) -> usize {
    #[cfg(feature = "std")]
    if enough {
        // Asked once: on Linux the answer reads the process's cgroup limits from files.
        static HOST: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        return *HOST.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    }
    let _ = enough;
    1
}

/// `share` run on the calling thread and on up to `threads - 1` more. A thread the host cannot start
/// leaves its share to the others.
#[cfg(feature = "std")]
fn run<R: Send>(threads: usize, share: &(impl Fn() -> R + Sync)) -> Vec<R> {
    if threads <= 1 {
        return alloc::vec![share()];
    }
    std::thread::scope(|scope| {
        let helpers: Vec<_> = (1..threads)
            .filter_map(|_| std::thread::Builder::new().spawn_scoped(scope, share).ok())
            .collect();
        let mut shares = alloc::vec![share()];
        for h in helpers {
            shares.push(h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)));
        }
        shares
    })
}

/// `share`, on the calling thread: a build without the `std` feature has no others.
#[cfg(not(feature = "std"))]
fn run<R>(_: usize, share: &impl Fn() -> R) -> Vec<R> {
    alloc::vec![share()]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Items 5 and 9 of 16 fail; 0 to 5 are slow, so a thread reaches 9 while another is still on
    /// 5. Every thread count reports 5, and a run with no failure gives every result in order.
    #[test]
    fn the_lowest_failure_wins_and_results_keep_their_order() {
        let each = |_: &mut (), i: usize| {
            if i <= 5 {
                (0..200_000u64).fold(0u64, |a, x| a.wrapping_add(core::hint::black_box(x)));
            }
            match i {
                5 | 9 => Err(i),
                _ => Ok(i * 10),
            }
        };
        for threads in [1, 2, 3, 4, 8, 16] {
            for _ in 0..10 {
                assert_eq!(
                    try_each(16, threads, || (), each),
                    Err(5),
                    "{threads} threads"
                );
            }
            let ok = try_each(16, threads, || (), |_: &mut (), i| Ok::<_, ()>(i * 10));
            assert_eq!(
                ok,
                Ok((0..16).map(|i| i * 10).collect()),
                "{threads} threads"
            );
        }
        assert_eq!(
            try_each(0, 4, || (), |_: &mut (), i| Ok::<_, ()>(i)),
            Ok(Vec::new())
        );
    }
}
