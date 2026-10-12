//! #2267 — the JIT window's `map`/`unmap` drop pages instead of writing them (Linux), so a large
//! commit stays unbacked until the guest touches it, while every committed page still reads zero.
//!
//! What `map` must not do is *touch* the pages it commits, and that is measured exactly: this
//! thread's own minor faults across the call (writing zeros over a commit faults every page in).
//! Whether the kernel backs an untouched range anyway is its own policy, not `map`'s — the Azure CI
//! kernel backs a few percent of a freshly `mprotect`ed range on its own — so the commit's
//! residency is reported, not asserted. `unmap` is checked with `mincore` on the one page it
//! dropped and left inaccessible, which reads only this test's mapping.
#![cfg(target_os = "linux")]

use temen_interp::GuestMem;
use temen_run::MprotectWindow;

const PROT_RW: i32 = 3;

/// A window laid out like the JIT's (`temen-jit`'s `GuestWindow`): an inaccessible, lazily backed
/// reservation with the backed prefix committed read/write.
struct Reservation {
    base: *mut u8,
    len: usize,
}

impl Reservation {
    fn new(len: usize, prefix: usize) -> Reservation {
        // SAFETY: a fresh anonymous reservation, checked below.
        let base = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_NONE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
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

    /// How many pages of `[off, off+len)` are resident.
    fn resident(&self, off: usize, len: usize, page: usize) -> usize {
        let mut v = vec![0u8; len / page];
        // SAFETY: the range lies inside the live reservation; `v` holds one byte per page.
        let rc =
            unsafe { libc::mincore(self.base.add(off) as *mut libc::c_void, len, v.as_mut_ptr()) };
        assert_eq!(rc, 0);
        v.iter().filter(|b| *b & 1 == 1).count()
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // SAFETY: releasing exactly the reservation `new` made; nothing borrows it any more.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
    }
}

/// This thread's minor page faults so far: one per page it touched for the first time.
fn minor_faults() -> i64 {
    // SAFETY: an all-zero `rusage` is a valid value for the out-parameter `getrusage` fills.
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    // SAFETY: `usage` is a valid out-parameter; `RUSAGE_THREAD` names the calling thread.
    assert_eq!(
        unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) },
        0
    );
    usage.ru_minflt
}

#[test]
fn map_does_not_touch_a_large_commit_and_every_commit_reads_zero() {
    let page = temen_interp::host_page_size() as usize;
    let (prefix, reserved) = (64 << 10, 1 << 30);
    let r = Reservation::new(reserved, prefix);
    let mut w = MprotectWindow::new(r.base, prefix as u64, reserved as u64);
    let (tail, big) = (1usize << 20, 256usize << 20);
    let pages = big / page;

    // A 256 MiB commit into the reserved tail. Writing zeros over it would fault in all of its
    // pages; dropping them faults in none, leaving only the page map's own bookkeeping.
    let before = minor_faults();
    assert_eq!(w.map(tail as u64, big as u64, PROT_RW), 0);
    let faults = minor_faults() - before;
    assert!(
        (faults as usize) < pages / 8,
        "map touched the commit: {faults} faults for {pages} pages ({} resident)",
        r.resident(tail, big, page)
    );

    // SAFETY: the page at `tail` is committed read/write for every access below; between the
    // `unmap` and the `map` that follows it, nothing touches it.
    let p = unsafe { r.base.add(tail + 123) };
    unsafe { p.write_volatile(0x5A) };
    assert_eq!(unsafe { p.read_volatile() }, 0x5A);

    // Re-mapping a live page is a fresh commit: it reads zero.
    assert_eq!(w.map(tail as u64, page as u64, PROT_RW), 0);
    assert_eq!(unsafe { p.read_volatile() }, 0);

    // `unmap` hands the page back; `map` after it reads zero again.
    unsafe { p.write_volatile(0x6B) };
    assert_eq!(w.unmap(tail as u64, page as u64), 0);
    assert_eq!(
        r.resident(tail, page, page),
        0,
        "unmap kept the page backed"
    );
    assert_eq!(w.map(tail as u64, page as u64, PROT_RW), 0);
    assert_eq!(unsafe { p.read_volatile() }, 0);
}
