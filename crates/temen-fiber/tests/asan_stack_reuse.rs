//! #3032 — under ASan, an arena stack is poisoned once its fiber is dropped, and clean when the next
//! fiber starts on it.
//!
//! A fiber's entry frame never returns (it leaves by its last `jump`), and a fiber dropped while
//! suspended leaves every frame of its body unreturned, so none of them clears the redzones its
//! prologue poisoned. An arena slot keeps that shadow when it is freed, and the slot's next fiber,
//! whose frames lie differently, read their own locals through it: a stack-buffer-overflow report
//! in `fiber_entry` with no overflow in it. This binary holds one test, so no other thread takes the
//! freed slot before the next fiber here does (the arena's free list is LIFO).
// The arena backend only: a guard-paged stack is unmapped on drop, which clears its shadow.
#![cfg(all(feature = "asan", not(feature = "guard-page-stacks")))]

use std::hint::black_box;
use temen_fiber::{Fiber, State, Yielder};

extern "C" {
    fn __asan_region_is_poisoned(beg: *const u8, size: usize) -> *const u8;
    fn __asan_address_is_poisoned(addr: *const u8) -> i32;
}

/// Suspend from a frame with an instrumented local, so its redzones are poisoned while it waits.
#[inline(never)]
fn suspend_in_a_frame(y: &Yielder) -> u64 {
    let local = [7u64; 4];
    black_box(&local);
    y.suspend(1) + local[0]
}

/// A fiber that runs `suspend_in_a_frame`.
fn fiber() -> Fiber {
    Fiber::new(64 * 1024, |y, _| suspend_in_a_frame(y)).unwrap()
}

/// Drop `f` and check its stack: every 8-byte granule of it poisoned, so a stale reach into a
/// dropped fiber's stack reports. Returns the stack's `[low, high)` frame extent.
fn drop_and_check(f: Fiber, what: &str) -> (usize, usize) {
    let (low, high) = f.full_extent();
    drop(f);
    let unpoisoned = (low as usize..high as usize)
        .step_by(8)
        // SAFETY: reads ASan's shadow only; nothing runs on the freed stack.
        .find(|&a| unsafe { __asan_address_is_poisoned(a as *const u8) } == 0);
    assert_eq!(
        unpoisoned.map(|a| high as usize - a),
        None,
        "{what}: its stack is not poisoned at top-N once it is dropped"
    );
    (low as usize, high as usize)
}

/// A fresh fiber on the stack the last one left: same slot, and no shadow left on it.
fn check_reuse(extent: (usize, usize), what: &str) {
    let f = fiber();
    let (low, high) = f.full_extent();
    assert_eq!(
        extent.0, low as usize,
        "{what}: the next fiber got another slot"
    );
    // SAFETY: reads ASan's shadow only, over the stack `f` owns and has not run on.
    let p = unsafe { __asan_region_is_poisoned(low, high as usize - low as usize) };
    assert!(
        p.is_null(),
        "{what}: the next fiber starts with shadow left at top-{}",
        high as usize - p as usize
    );
}

#[test]
fn a_fiber_stack_is_poisoned_when_dropped_and_clean_when_reused() {
    let mut done = fiber();
    assert_eq!(done.resume(0), State::Yielded(1));
    assert_eq!(done.resume(0), State::Complete(7));
    let extent = drop_and_check(done, "a completed fiber");
    check_reuse(extent, "after a completed fiber");

    let mut parked = fiber();
    assert_eq!(parked.resume(0), State::Yielded(1));
    let extent = drop_and_check(parked, "a fiber dropped while suspended");
    check_reuse(extent, "after a fiber dropped while suspended");
}
