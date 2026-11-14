//! #2958 slice 3 — `MprotectWindow`'s futex face (`temen_interp::FutexWindow`), the host code that
//! reads a guest word at a guest-chosen address on the kernel route, tested as its own unit:
//!
//! - **The guarded read against a model.** Random page ops and §13 region maps, then random
//!   addresses and widths: the read is refused exactly when a byte lies outside the reservation, the
//!   address is misaligned, the page is below the NULL guard, or the page is not readable, and
//!   otherwise answers the word. A read the model refuses that the window admitted would fault this
//!   process. And the keys (#2958 slice 4): two aliases of one region byte key alike, and a word on
//!   an alias and a private word at the same offset key apart.
//! - **The race.** Threads read words of a page while another unmaps, maps and protects it, for as
//!   long as it takes: the process never faults, and each read answers a value or a refusal.
#![cfg(unix)]

use temen_interp::{FutexWindow, GuestMem};
use temen_run::MprotectWindow;

/// A window laid out like the JIT's: an inaccessible reservation with the backed prefix committed
/// read-write.
struct Reservation {
    base: *mut u8,
    len: usize,
}

// SAFETY: the reservation is only reached through raw pointers by the test's own threads, which it
// outlives (scoped threads).
unsafe impl Send for Reservation {}
unsafe impl Sync for Reservation {}

impl Reservation {
    fn new(len: usize, prefix: usize) -> Reservation {
        // SAFETY: a fresh anonymous reservation, checked below.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANON,
                -1,
                0,
            )
        };
        assert_ne!(base, libc::MAP_FAILED);
        // SAFETY: `[base, base+prefix)` lies inside the reservation just made.
        let rc = unsafe { libc::mprotect(base, prefix, libc::PROT_READ | libc::PROT_WRITE) };
        assert_eq!(rc, 0);
        Reservation {
            base: base as *mut u8,
            len,
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // SAFETY: releasing exactly the reservation `new` made; nothing borrows it any more.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
    }
}

/// Sets its flag as it drops.
struct Stop<'a>(&'a std::sync::atomic::AtomicBool);

impl Drop for Stop<'_> {
    fn drop(&mut self) {
        self.0.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// A small deterministic generator (xorshift64*), so a failure names its seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The model of one page: whether a load may read it.
#[derive(Clone, Copy)]
enum Model {
    Readable,
    Not,
}

#[test]
fn the_guarded_read_refuses_exactly_what_the_model_refuses() {
    let page = temen_interp::host_page_size();
    // A reservation of 32 pages, 8 of them the backed prefix, the first 2 a NULL guard.
    let (pages, prefix, guard) = (32u64, 8u64, 2u64);
    let reserved = pages * page;
    for seed in 1..=64u64 {
        let r = Reservation::new(reserved as usize, (prefix * page) as usize);
        let map = temen_interp::CapPageMap::default();
        let mut w = MprotectWindow::new_shared(r.base, prefix * page, reserved, map.clone());
        // A region of two pages, which a map aliases one page of at a time.
        let region = temen_run::new_shared_region((2 * page) as usize).expect("a region");
        // Which region page each window page aliases, if any.
        let mut alias: Vec<Option<u64>> = vec![None; pages as usize];
        // A region map has succeeded: the window's `aliased` is set from then on, never cleared.
        let mut ever = false;
        w.set_null_guard(guard * page);
        // The NULL guard is the window start's to protect (`seed_window`), not a page op's.
        // SAFETY: the guard's pages lie inside the reservation.
        assert_eq!(
            unsafe { libc::mprotect(r.base.cast(), (guard * page) as usize, libc::PROT_NONE) },
            0
        );
        let mut model: Vec<Model> = (0..pages)
            .map(|p| match p >= guard && p < prefix {
                true => Model::Readable,
                false => Model::Not,
            })
            .collect();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        for _ in 0..200 {
            // A page op on a run of pages, or none.
            let first = rng.below(pages);
            let n = 1 + rng.below(3).min(pages - first - 1);
            let (off, len) = (first * page, n * page);
            let prot = [0, 1, 3][rng.below(3) as usize];
            let rp = rng.below(2);
            let which = rng.below(5);
            let (rc, readable) = match which {
                0 => (w.map(off, len, prot), prot & 1 != 0),
                1 => (w.unmap(off, len), false),
                2 => (w.protect(off, len, prot), prot & 1 != 0),
                3 => (
                    w.map_region(off, rp * page, page, 3, 7, region.clone()),
                    true,
                ),
                _ => (1, false),
            };
            // A region map aliases one page.
            let n = if which == 3 { 1 } else { n };
            if rc == 0 {
                for p in first..first + n {
                    model[p as usize] = if readable {
                        Model::Readable
                    } else {
                        Model::Not
                    };
                    // A region map makes the page an alias; a protect that leaves access keeps one;
                    // any other op ends one.
                    alias[p as usize] = match which {
                        3 => Some(rp),
                        2 if readable => alias[p as usize],
                        _ => None,
                    };
                }
                ever |= which == 3;
            } else if rc < 0 {
                assert!(first < guard, "seed {seed}: only the guard refuses: {rc}");
                // A refused op changes nothing, a region map's record of its region included.
                if which == 3 && !ever {
                    assert!(
                        !map.aliased(),
                        "seed {seed}: a refused map records no alias"
                    );
                    assert_eq!(map.region_ident(7), None, "seed {seed}");
                }
            }
            // The keys: at one offset, the words of two readable pages key alike exactly when they
            // are one page or alias one region page.
            let o = 4 * rng.below(page / 4);
            let readable: Vec<u64> = (0..pages)
                .filter(|&p| matches!(model[p as usize], Model::Readable))
                .collect();
            let key = |p: u64| {
                let base = w.prepare_wait(p * page + o, 4).expect("readable");
                w.site_key(w.futex_site(base))
            };
            for &p in &readable {
                for &q in &readable {
                    let same = p == q
                        || matches!((alias[p as usize], alias[q as usize]), (Some(a), Some(b)) if a == b);
                    assert_eq!(
                        key(p) == key(q),
                        same,
                        "seed {seed}: pages {p} and {q} at {o:#x}, aliases {:?} and {:?}",
                        alias[p as usize],
                        alias[q as usize]
                    );
                }
            }
            // Reads at random addresses and widths, including past the reservation.
            for _ in 0..16 {
                let width = [4u32, 8][rng.below(2) as usize];
                let addr = match rng.below(8) {
                    0 => reserved - u64::from(width) + rng.below(16),
                    1 => u64::MAX - rng.below(16),
                    _ => rng.below(reserved),
                };
                let aligned = addr.is_multiple_of(u64::from(width));
                let inside = addr
                    .checked_add(u64::from(width))
                    .is_some_and(|e| e <= reserved);
                let want =
                    aligned && inside && matches!(model[(addr / page) as usize], Model::Readable);
                let got = w.wait_word(addr, width);
                assert_eq!(
                    got.is_some(),
                    want,
                    "seed {seed}: a {width}-byte read at {addr:#x}"
                );
                // Every refusal is the map check's: no read faulted and was recovered.
                assert_eq!(
                    map.recovered(),
                    0,
                    "seed {seed}: a recovered read at {addr:#x}"
                );
                if let Some(v) = got {
                    // SAFETY: the model, and so the read just made, says the word is readable.
                    let raw = unsafe {
                        let mut b = [0u8; 8];
                        std::ptr::copy_nonoverlapping(
                            r.base.add(addr as usize),
                            b.as_mut_ptr(),
                            width as usize,
                        );
                        u64::from_le_bytes(b)
                    };
                    assert_eq!(v, raw, "seed {seed}: the word at {addr:#x}");
                }
                // A wait's check refuses what the read refuses.
                let checked = w.prepare_wait(addr, width);
                assert_eq!(checked.is_ok(), want, "seed {seed}: the check at {addr:#x}");
            }
        }
    }
}

#[test]
fn a_read_racing_an_unmap_or_a_protect_never_faults_the_host() {
    let page = temen_interp::host_page_size();
    let (prefix, reserved) = (8 * page, 32 * page);
    let r = Reservation::new(reserved as usize, prefix as usize);
    let pages = temen_interp::CapPageMap::default();
    let target = 4 * page;
    let stop = std::sync::atomic::AtomicBool::new(false);
    let (mut answered, mut refused) = (0u64, 0u64);
    std::thread::scope(|s| {
        // Every thread below stops as this drops, however the scope ends: a failed assertion fails
        // the test instead of leaving the scope joining threads that never stop.
        let _stop = Stop(&stop);
        // Busy threads, so the readers and the page op are descheduled at arbitrary points.
        let busy: Vec<_> = (0..2)
            .map(|_| {
                s.spawn(|| {
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        std::hint::spin_loop();
                    }
                })
            })
            .collect();
        let readers: Vec<_> = (0..3)
            .map(|i| {
                let (base, pages, stop) = (r.base as usize, pages.clone(), &stop);
                s.spawn(move || {
                    let w = MprotectWindow::new_shared(base as *mut u8, prefix, reserved, pages);
                    let (mut some, mut none) = (0u64, 0u64);
                    let at = target + 8 * i;
                    while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                        match w.wait_word(at, 4) {
                            Some(_) => some += 1,
                            None => none += 1,
                        }
                    }
                    (some, none)
                })
            })
            .collect();
        let mut w = MprotectWindow::new_shared(r.base, prefix, reserved, pages.clone());
        for i in 0..20_000 {
            let rc = match i % 4 {
                0 => w.unmap(target, page),
                1 => w.map(target, page, 3),
                2 => w.protect(target, page, 0),
                _ => w.protect(target, page, 3),
            };
            assert_eq!(rc, 0, "page op {i}");
        }
        stop.store(true, std::sync::atomic::Ordering::Relaxed);
        for b in busy {
            b.join().expect("busy");
        }
        for t in readers {
            let (some, none) = t.join().expect("a reader");
            answered += some;
            refused += none;
        }
    });
    // Both answers happened: the reads raced the page ops, and none faulted the process. The
    // refusals are all the map check's: a read meeting a page op's hardware change midway would be
    // recovered (`wait_word`'s load installs the fault handler), and counted.
    assert!(
        answered > 0 && refused > 0,
        "{answered} read, {refused} refused"
    );
    assert_eq!(pages.recovered(), 0, "a read recovered a fault");
}

/// #2958 slice 4 review, finding 4 — a region's identity is held as the interpreter holds it, by a
/// weak hold on its backing: once an alias of region 7 is unmapped and the region dropped, no region
/// made after it has its identity while the page map still names it, so a waiter that parked on the
/// old region meets no new one at its address. This pins only the half that holds: once the host
/// reuses the region's id slot the old identity can recur (#3036, parity on every engine).
#[test]
fn a_dropped_regions_identity_is_not_reused_while_the_map_names_it() {
    let page = temen_interp::host_page_size();
    let (prefix, reserved) = (8 * page, 32 * page);
    let r = Reservation::new(reserved as usize, prefix as usize);
    let map = temen_interp::CapPageMap::default();
    let mut w = MprotectWindow::new_shared(r.base, prefix, reserved, map.clone());
    let at = 16 * page;
    let old = temen_run::new_shared_region(page as usize).expect("a region");
    assert_eq!(w.map_region(at, 0, page, 3, 7, old.clone()), 0);
    let ident = map.region_ident(7).expect("recorded");
    assert_eq!(ident, std::sync::Arc::as_ptr(&old) as *const u8 as u64);
    assert_eq!(w.unmap(at, page), 0);
    drop(old);
    let later: Vec<_> = (0..64)
        .map(|_| temen_run::new_shared_region(page as usize).expect("a region"))
        .collect();
    for b in &later {
        assert_ne!(std::sync::Arc::as_ptr(b) as *const u8 as u64, ident);
    }
    assert_eq!(map.region_ident(7), Some(ident), "the entry still names it");
}
