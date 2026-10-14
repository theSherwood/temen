//! The JIT window's page ops (`MprotectWindow`) against the real hardware (Linux).
//!
//! #2267 — `map`/`unmap` drop pages instead of writing them, so a large commit stays unbacked until
//! the guest touches it, while every committed page still reads zero. What `map` must not do is
//! *touch* the pages it commits, and that is measured exactly: this thread's own minor faults across
//! the call (writing zeros over a commit faults every page in). Whether the kernel backs an untouched
//! range anyway is its own policy, not `map`'s — the Azure CI kernel backs a few percent of a freshly
//! `mprotect`ed range on its own — so the commit's residency is reported, not asserted. `unmap` is
//! checked with `mincore` on the one page it dropped and left inaccessible, which reads only this
//! test's mapping.
//!
//! #1707 — an op that ends a §13 alias ends it in the hardware too.
//!
//! #2268 — the OS can refuse a protection change: `ENOMEM` once the process holds
//! `vm.max_map_count` mappings, `EACCES` on a mapping it may not make writable. A refused op returns
//! `-ENOMEM`, and the page map never grants the host more than the hardware might, since host code
//! copies into any page the map says it may.
#![cfg(target_os = "linux")]

use temen_interp::GuestMem;
use temen_run::MprotectWindow;

const ENOMEM: i64 = -12;
const PROT_R: i32 = 1;
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

    /// Replace `[off, off+len)` with a read-only shared mapping of a file opened read-only, which the
    /// OS refuses to make writable (`EACCES`) however few mappings the process holds.
    fn read_only_file_over(&self, off: usize, len: usize) {
        // SAFETY: plain fd calls on a fresh memfd, each checked; the read-only reopen goes through
        // `/proc/self/fd`, so the second fd lacks write access; the `MAP_FIXED` lies inside the
        // reservation, which this test owns.
        unsafe {
            let fd = libc::memfd_create(c"refuses_writes".as_ptr(), libc::MFD_CLOEXEC);
            assert!(fd >= 0);
            assert_eq!(libc::ftruncate(fd, len as libc::off_t), 0);
            let path = std::ffi::CString::new(format!("/proc/self/fd/{fd}")).unwrap();
            let ro = libc::open(path.as_ptr(), libc::O_RDONLY | libc::O_CLOEXEC);
            assert!(ro >= 0);
            let p = libc::mmap(
                self.base.add(off) as *mut libc::c_void,
                len,
                libc::PROT_READ,
                libc::MAP_SHARED | libc::MAP_FIXED,
                ro,
                0,
            );
            assert_ne!(p, libc::MAP_FAILED);
            libc::close(ro);
            libc::close(fd);
        }
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

#[test]
fn a_refused_protection_change_fails_the_op_and_grants_the_host_nothing() {
    let page = temen_interp::host_page_size() as usize;
    let r = Reservation::new(64 * page, 16 * page);
    let mut w = MprotectWindow::new(r.base, 16 * page as u64, 64 * page as u64);
    let (ro, fixed) = (20 * page as u64, 21 * page as u64);
    r.read_only_file_over(21 * page, page);

    // Committing the page read-write is refused, by `map` and by `protect` alike. The page map must
    // not say otherwise: a host copy into the page would fault outside any guarded call.
    assert_eq!(w.map(fixed, page as u64, PROT_RW), ENOMEM);
    assert!(w.write_bytes(fixed, &[1]).is_none(), "map granted a write");
    assert_eq!(w.protect(fixed, page as u64, PROT_RW), ENOMEM);
    assert!(
        w.write_bytes(fixed, &[1]).is_none(),
        "protect granted a write"
    );

    // Refused partway: the window's own read-only page before it is made writable, then the OS
    // refuses the next mapping. The page map keeps both as they were.
    assert_eq!(w.map(ro, page as u64, PROT_R), 0);
    assert_eq!(w.protect(ro, 2 * page as u64, PROT_RW), ENOMEM);
    assert!(
        w.write_bytes(ro, &[1]).is_none(),
        "a partial protect granted a write"
    );
    assert_eq!(w.read_bytes(ro, 1), Some(vec![0]));
}

/// #1707 — an op that ends a §13 alias (a `map` over it, a `protect(none)`) ends it in the hardware
/// too, as `unmap` does, so the page is the window's own from then on, as on the interpreter: it
/// reads zero and stops writing through, and the region keeps its bytes for every other mapping.
#[test]
fn ending_an_alias_leaves_the_region_alone() {
    let page = temen_interp::host_page_size() as usize;
    let r = Reservation::new(64 * page, 16 * page);
    let mut w = MprotectWindow::new(r.base, 16 * page as u64, 64 * page as u64);
    let at = 20 * page;
    // SAFETY: the page at `at` is committed read/write at every access below.
    let p = unsafe { r.base.add(at) };
    for (how, by_protect) in [("map over it", false), ("protect(none)", true)] {
        let region = temen_run::new_shared_region(page).expect("a region");
        assert_eq!(
            w.map_region(at as u64, 0, page as u64, PROT_RW, 1, region.clone()),
            0
        );
        unsafe { p.write_volatile(0x11) };
        assert_eq!(region.read_byte(0), 0x11, "the alias writes the region");
        if by_protect {
            assert_eq!(w.protect(at as u64, page as u64, 0), 0);
        }
        assert_eq!(w.map(at as u64, page as u64, PROT_RW), 0);
        assert_eq!(
            unsafe { p.read_volatile() },
            0,
            "{how}: the page is not fresh"
        );
        unsafe { p.write_volatile(0x22) };
        assert_eq!(
            region.read_byte(0),
            0x11,
            "{how}: the window reached the region"
        );
        assert_eq!(w.unmap(at as u64, page as u64), 0);
    }
}

/// Whether the hardware lets this process read the byte at `p`: a `write` from it fails `EFAULT`
/// where it may not, instead of faulting.
fn readable(p: *const u8) -> bool {
    let mut fds = [0; 2];
    // SAFETY: plain fd calls on a fresh pipe, each checked; the kernel validates `p` itself.
    unsafe {
        assert_eq!(libc::pipe(fds.as_mut_ptr()), 0);
        let n = libc::write(fds[1], p as *const libc::c_void, 1);
        libc::close(fds[0]);
        libc::close(fds[1]);
        n == 1
    }
}

/// #2370 — a page op that grants no access changes only what the window holds, in the page map and
/// in the hardware, so its host work is bounded by that, not by the span it names. Over a 128 GiB
/// reservation's 2^25 pages, a walk of the span took seconds per op; this one costs microseconds.
/// Measured in this thread's CPU time, which a loaded machine does not stretch.
#[test]
fn a_page_op_over_the_whole_reservation_costs_what_the_window_holds() {
    let cpu_us = || {
        // SAFETY: as in `minor_faults`.
        let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut usage) },
            0
        );
        let us = |t: libc::timeval| t.tv_sec as u64 * 1_000_000 + t.tv_usec as u64;
        us(usage.ru_utime) + us(usage.ru_stime)
    };
    let page = temen_interp::host_page_size();
    let (mapped, reserved) = (64u64 << 10, 1u64 << 37);
    let r = Reservation::new(reserved as usize, mapped as usize);
    let mut w = MprotectWindow::new(r.base, mapped, reserved);
    // Two adjacent tail pages and one far up, each holding a value.
    let held = [mapped, mapped + page, reserved - page];
    // SAFETY: each offset lies inside the reservation; `readable` only probes it.
    let hw_readable = |at: u64| readable(unsafe { r.base.add(at as usize) });
    let fill = |w: &mut MprotectWindow| {
        for at in held {
            assert_eq!(w.map(at, page, PROT_RW), 0);
            assert_eq!(w.write_bytes(at, &[7]), Some(()));
        }
    };
    let read = |w: &mut MprotectWindow, at| {
        assert_eq!(w.protect(at, page, PROT_R), 0);
        w.read_bytes(at, 1).map(|b| b[0])
    };
    let start = cpu_us();
    for _ in 0..4 {
        fill(&mut w);
        // To no access: the pages keep their contents, which a later `protect` shows.
        assert_eq!(w.protect(0, reserved, 0), 0);
        assert!(held.iter().chain(&[0]).all(|&at| !hw_readable(at)));
        assert_eq!(read(&mut w, held[2]), Some(7));
        // A `map` to no access and an `unmap` zero them, the declared window's included.
        assert_eq!(w.map(0, reserved, 0), 0);
        assert_eq!(read(&mut w, held[0]), Some(0));
        fill(&mut w);
        assert_eq!(w.unmap(0, reserved), 0);
        assert!(held.iter().chain(&[0]).all(|&at| !hw_readable(at)));
        for at in [0, held[1], held[2]] {
            assert_eq!(read(&mut w, at), Some(0));
        }
        assert_eq!(w.unmap(0, reserved), 0);
        assert_eq!(w.map(0, mapped, PROT_RW), 0);
    }
    let spent = cpu_us() - start;
    assert!(spent < 500_000, "16 ops took {spent} µs of CPU");
}

/// What the child below found wrong, one bit per check.
const WRONG_PROTECT: i32 = 1;
const WRONG_PROTECT_HOST: i32 = 2;
const WRONG_MAP: i32 = 4;
const WRONG_MAP_HOST: i32 = 8;

/// At the process's mapping limit — reached by a guest that splits its window page by page — an op
/// that needs another mapping is refused. Run in a forked child, which fills its own mapping table
/// and leaves this process's alone.
#[test]
fn page_ops_fail_enomem_at_the_mapping_limit() {
    let page = temen_interp::host_page_size() as usize;
    let max: usize = std::fs::read_to_string("/proc/sys/vm/max_map_count")
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    if max > 1 << 22 {
        eprintln!("skipped: vm.max_map_count = {max} is too large to fill");
        return;
    }
    let r = Reservation::new(64 * page, 16 * page);
    let mut w = MprotectWindow::new(r.base, 16 * page as u64, 64 * page as u64);
    // An explicit entry now, so the page-map updates at the limit need no new allocation.
    assert_eq!(w.protect(15 * page as u64, page as u64, PROT_R), 0);
    // SAFETY: the child runs only the code below, allocating nothing, and leaves with `_exit`.
    match unsafe { libc::fork() } {
        0 => {
            // SAFETY: a scratch reservation of the child's own, every other page made readable
            // until the OS refuses: each one splits the reservation into two more mappings.
            let wrong = unsafe {
                let len = 2 * (max + 16) * page;
                let scratch = libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_NONE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                    -1,
                    0,
                ) as *mut u8;
                let mut i = 0;
                while scratch as *mut libc::c_void != libc::MAP_FAILED
                    && 2 * i * page < len
                    && libc::mprotect(
                        scratch.add(2 * i * page) as *mut libc::c_void,
                        page,
                        libc::PROT_READ,
                    ) == 0
                {
                    i += 1;
                }
                let mut wrong = 0;
                // A read-only page inside the read-write prefix splits it.
                if w.protect(2 * page as u64, page as u64, PROT_R) != ENOMEM {
                    wrong |= WRONG_PROTECT;
                }
                if w.write_bytes(2 * page as u64, &[1]).is_some() {
                    wrong |= WRONG_PROTECT_HOST;
                }
                // So does a read-write page inside the inaccessible tail.
                if w.map(30 * page as u64, page as u64, PROT_RW) != ENOMEM {
                    wrong |= WRONG_MAP;
                }
                if w.write_bytes(30 * page as u64, &[1]).is_some() {
                    wrong |= WRONG_MAP_HOST;
                }
                wrong
            };
            // SAFETY: leaves the child without running the parent's exit handlers.
            unsafe { libc::_exit(wrong) }
        }
        pid => {
            assert!(pid > 0, "fork failed");
            let mut status = 0;
            // SAFETY: waits for the child just forked; `status` is a valid out-parameter.
            assert_eq!(unsafe { libc::waitpid(pid, &mut status, 0) }, pid);
            assert!(
                libc::WIFEXITED(status),
                "the child died of signal {} (a host copy into a page the hardware refused)",
                libc::WTERMSIG(status)
            );
            let wrong = libc::WEXITSTATUS(status);
            assert_eq!(
                wrong,
                0,
                "at the mapping limit: protect {} {}, map {} {}",
                if wrong & WRONG_PROTECT != 0 {
                    "returned success"
                } else {
                    "was refused"
                },
                if wrong & WRONG_PROTECT_HOST != 0 {
                    "and granted a write"
                } else {
                    ""
                },
                if wrong & WRONG_MAP != 0 {
                    "returned success"
                } else {
                    "was refused"
                },
                if wrong & WRONG_MAP_HOST != 0 {
                    "and granted a write"
                } else {
                    ""
                },
            );
        }
    }
}
