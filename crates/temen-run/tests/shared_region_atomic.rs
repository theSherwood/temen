//! #2127 — `SharedBacking::atomic` on the OS-mapped region backing (`new_shared_region`), the one
//! the interpreter's atomics reach a JIT-shared region through: real hardware atomics, so two
//! threads' fetch-adds lose nothing and a 64-bit load never sees half of a store.

use std::sync::Arc;
use temen_interp::SharedBacking;

const N: u64 = 100_000;

#[test]
fn concurrent_fetch_adds_lose_nothing() {
    let r = temen_run::new_shared_region(4096);
    let add = |r: Arc<dyn SharedBacking>| {
        std::thread::spawn(move || {
            for _ in 0..N {
                r.atomic(8, 8, &mut |v| Some(v + 1));
            }
        })
    };
    let (a, b) = (add(r.clone()), add(r.clone()));
    a.join().expect("thread a");
    b.join().expect("thread b");
    assert_eq!(r.atomic(8, 8, &mut |_| None), 2 * N);
}

#[test]
fn a_load_never_sees_half_of_a_store() {
    let r = temen_run::new_shared_region(4096);
    let w = {
        let r = r.clone();
        std::thread::spawn(move || {
            for i in 0..N {
                let v = if i % 2 == 0 { u64::MAX } else { 0 };
                r.atomic(0, 8, &mut |_| Some(v));
            }
        })
    };
    let torn = (0..N)
        .filter(|_| !matches!(r.atomic(0, 8, &mut |_| None), 0 | u64::MAX))
        .count();
    w.join().expect("writer");
    assert_eq!(torn, 0);
}

#[test]
fn narrow_widths_and_refusals() {
    let r = temen_run::new_shared_region(4096);
    // A 4-byte store keeps to its 4 bytes, and its old value comes back zero-extended.
    r.atomic(16, 8, &mut |_| Some(u64::MAX));
    assert_eq!(r.atomic(16, 4, &mut |_| Some(7)), 0xffff_ffff);
    assert_eq!(r.atomic(16, 8, &mut |_| None), 0xffff_ffff_0000_0007);
    // Out of range, misaligned, or an odd width: 0, and nothing stored.
    assert_eq!(r.atomic(4096, 4, &mut |_| Some(1)), 0);
    assert_eq!(r.atomic(17, 4, &mut |_| Some(1)), 0);
    assert_eq!(r.atomic(16, 3, &mut |_| Some(1)), 0);
    assert_eq!(r.atomic(16, 8, &mut |_| None), 0xffff_ffff_0000_0007);
}
