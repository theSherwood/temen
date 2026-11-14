//! #2958 slice 4 review — `temen_jit::try_load_in`, the recoverable load host code reads a guest
//! word through (the kernel route's `wait_word`, today's route's futex read). Both callers check the
//! word before they call it, so its own refusals, and its recovery off any guarded call, are pinned
//! here:
//!
//! - a width other than 1, 2, 4 or 8, a word not aligned to its width, and a word not wholly inside
//!   the window answer `None` without a load (each is set on a readable page, where a load would
//!   answer a value);
//! - a load of an unmapped page of a real reservation, on a fresh thread that never entered a
//!   guarded call, answers `None`, and the process lives: the function installs the handler.
#![cfg(all(unix, any(target_arch = "x86_64", target_arch = "aarch64")))]

/// A reservation of `pages` pages, the first `readable` of them read-write, the rest inaccessible.
struct Reservation {
    base: *mut u8,
    len: usize,
}

impl Reservation {
    fn new(pages: usize, readable: usize) -> Reservation {
        let page = page_size();
        let len = pages * page;
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
        // SAFETY: `[base, base + readable pages)` lies inside the reservation just made.
        let rc =
            unsafe { libc::mprotect(base, readable * page, libc::PROT_READ | libc::PROT_WRITE) };
        assert_eq!(rc, 0);
        Reservation {
            base: base as *mut u8,
            len,
        }
    }
}

impl Drop for Reservation {
    fn drop(&mut self) {
        // SAFETY: releasing exactly the reservation `new` made.
        unsafe { libc::munmap(self.base as *mut libc::c_void, self.len) };
    }
}

fn page_size() -> usize {
    // SAFETY: sysconf is always safe to call.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[test]
fn a_bad_width_alignment_or_span_is_refused_without_a_load() {
    let r = Reservation::new(4, 4);
    let page = page_size();
    // A readable word, where a load answers its value.
    // SAFETY: the first page is mapped read-write.
    unsafe { r.base.add(64).cast::<u64>().write(0x1122_3344_5566_7788) };
    let load = |base: *mut u8, reserved: usize, at: usize, width: u32| {
        // SAFETY: `[base, base + reserved)` lies inside the live reservation `r`.
        unsafe { temen_jit::try_load_in(base, reserved, r.base.add(at), width) }
    };
    assert_eq!(load(r.base, r.len, 64, 8), Some(0x1122_3344_5566_7788));
    assert_eq!(load(r.base, r.len, 64, 4), Some(0x5566_7788));
    for width in [0, 3, 5, 16] {
        assert_eq!(load(r.base, r.len, 64, width), None, "width {width}");
    }
    assert_eq!(load(r.base, r.len, 66, 4), None, "misaligned");
    // A window that is the reservation's second page only: a word on its first page lies below
    // the window, and one at its last word's end past it, though both are readable.
    let (lo, len) = (unsafe { r.base.add(page) }, page);
    assert_eq!(load(lo, len, 64, 8), None, "below the window");
    assert_eq!(load(lo, len, 2 * page - 4, 8), None, "past the window");
    assert_eq!(
        load(lo, len, 2 * page - 8, 8),
        Some(0),
        "the window's last word"
    );
}

#[test]
fn an_unmapped_page_answers_none_on_a_thread_that_never_entered_a_guarded_call() {
    let r = Reservation::new(4, 1);
    let page = page_size();
    let (base, len) = (r.base as usize, r.len);
    let answer = std::thread::spawn(move || {
        let base = base as *mut u8;
        // SAFETY: `[base, base + len)` is the live reservation, which outlives the thread (joined
        // below); its second page is inaccessible.
        unsafe { temen_jit::try_load_in(base, len, base.add(page), 8) }
    })
    .join()
    .expect("the thread lives");
    assert_eq!(answer, None);
}
