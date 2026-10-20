//! Guest-window allocation + detect-and-kill trap recovery (`DESIGN.md` §4/§5), over a small
//! **Platform Abstraction Layer** (PAL, §16/D51).
//!
//! The window is a large **reserved** virtual range whose backed prefix `[0, mapped)` is committed
//! read/write and whose tail `[mapped, reserved)` plus a trailing guard page are inaccessible. The
//! masking lowering confines every access into `[0, reserved)` (the JIT mask const = `reserved−1`),
//! so an access lands either in the backed prefix or in the inaccessible tail/guard — where it
//! **faults**, and the fault is caught and turned into a clean [`TrapKind::MemoryFault`] instead of
//! corrupting the host (§5 detect-and-kill). This fires on a width-overrun at the very top of the
//! window or — defense-in-depth — a masking/elision bug the guard caught.
//!
//! Everything platform-specific is the small [`pal`] module: reserve / commit / protect / release
//! virtual address space, and run a call under a guard that converts an in-window fault into a
//! caught return. The window *model* (page rounding, the `mapped`/`total` bookkeeping, the RW
//! slice) is portable and shared. There is a **unix** PAL (`mmap` + `mprotect(PROT_NONE)` + a
//! SIGSEGV/SIGBUS handler via `trap_shim.c`) and a **windows** PAL (`VirtualAlloc2` placeholder
//! reservation + `VirtualProtect(PAGE_NOACCESS)` + a Vectored Exception Handler). The confinement
//! arithmetic is identical on every target, so the interpreter↔JIT differential (§18) is the
//! cross-platform conformance oracle.

use crate::TrapKind;
use core::ffi::c_void;
use std::collections::BTreeMap;

/// The compiled entry trampoline ABI (see `build_trampoline`): `(args, results, mem_base,
/// fn_table_base, vmctx)`. The 4th pointer is opaque here (`FnEntry*` to the JIT); the 5th is the
/// instance's [`crate::VmCtx`], spelled as the trap cell it starts with.
type Entry = extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64);

/// Protection the PAL applies to a committed range of the window.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Prot {
    /// Readable + writable (the backed prefix; restored before a snapshot read).
    Rw,
    /// Read-only (the D40 const data segment; a later write faults).
    Ro,
    /// Inaccessible — any access faults into the guard (a durable-restored `Unmapped` page).
    None,
}

#[inline]
fn round_up(n: usize, align: usize) -> usize {
    (n + align - 1) & !(align - 1)
}

// ============================ portable window model (over the PAL) ============================

/// A guest window: `mapped` readable/writable bytes (the backed prefix `[0, mapped)`) inside a
/// larger `reserved` virtual range whose tail `[mapped, reserved)` plus one trailing guard page are
/// inaccessible (so an access past `mapped` faults → §4/§5 detect-and-kill). A fully-mapped window
/// is just `reserved == mapped` (the historical single-extent case). Built on the [`pal`] below.
pub(crate) struct GuestWindow {
    base: *mut u8,
    mapped: usize, // backed RW bytes `[0, mapped)` (the logical window the guest declared)
    total: usize,  // full reservation length: reserved (page-rounded) + one guard page
}

// D66 — a child-domain task owns its window and is resumed by whichever worker claims it, so the
// window moves between OS threads. Sound: an mmap reservation has no thread affinity; every access
// goes through the resumed child's own frames on the thread running it, and `Drop` releases from
// whichever thread finishes the task. The raw pointer is the only reason `Send` is not derived.
unsafe impl Send for GuestWindow {}

impl GuestWindow {
    /// Reserve `reserved` bytes (page-rounded) + a guard page as inaccessible, then commit the
    /// `mapped` backed prefix read/write. `reserved` is raised to at least `mapped`. `None`, holding
    /// nothing, when the OS gives neither: the address space has no room for the reservation, or the
    /// commit is refused. Every caller refuses then (#2305) — a run ends with `ThreadFault` before it
    /// starts, a spawn answers `-EINVAL`, a fork `-EAGAIN` — and none fails the host.
    ///
    /// For the inaccessible tail's fault boundary to agree with the interpreter's byte-exact
    /// `mapped` bound, `mapped` must be page-aligned whenever `reserved > mapped` — true for any
    /// `size_log2 >= 12`, which every caller of the decoupled form satisfies.
    pub(crate) fn try_new(mapped: usize, reserved: usize) -> Option<GuestWindow> {
        if mapped == 0 {
            return Some(GuestWindow {
                base: std::ptr::null_mut(),
                mapped: 0,
                total: 0,
            });
        }
        let page = pal::page_size();
        let reserved = reserved.max(mapped);
        let rw = round_up(mapped, page);
        let total = round_up(reserved, page) + trailing_guard();
        // SAFETY: a fresh inaccessible reservation (a huge `reserved` costs only virtual address
        // space until pages are committed/touched). Checked non-null below.
        let base = unsafe { pal::reserve(total) };
        if base.is_null() {
            return None;
        }
        // Owned from here: a refused commit drops it, releasing the reservation.
        let window = GuestWindow {
            base,
            mapped,
            total,
        };
        // SAFETY: commit the backed prefix `[0, rw)` read/write; the tail + guard stay inaccessible
        // so any access past `mapped` faults.
        unsafe { pal::commit_rw(base, rw) }.then_some(window)
    }

    /// The logical (backed) window `[0, mapped)`, readable/writable (freshly committed pages are
    /// zeroed). Bytes in the reserved-but-inaccessible tail are not part of this slice.
    pub(crate) fn rw_mut(&mut self) -> &mut [u8] {
        if self.mapped == 0 {
            return &mut [];
        }
        // SAFETY: `[base, base+mapped)` is committed RW for the window's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.base, self.mapped) }
    }

    pub(crate) fn base(&self) -> *mut u8 {
        self.base
    }

    /// The backed prefix's length (`[0, mapped)`).
    pub(crate) fn mapped(&self) -> usize {
        self.mapped
    }

    /// Re-enable read+write on the whole backed region `[0, mapped)`. The guest may have changed
    /// page protections through the `Memory` capability (`unmap`→inaccessible, `protect`→read-only),
    /// so a snapshot read of the window could otherwise fault *outside* the guarded call and crash
    /// the host. Idempotent; a no-op cost when nothing changed. `false` when the OS refuses (#2286:
    /// at `vm.max_map_count`, a boundary split): the caller must not read the window then.
    #[must_use]
    pub(crate) fn restore_rw(&self) -> bool {
        if self.mapped == 0 {
            return true;
        }
        let rw = round_up(self.mapped, pal::page_size());
        // SAFETY: `[base, base+rw)` is the window's backed region, owned for its lifetime.
        unsafe { pal::protect(self.base, rw, Prot::Rw) }
    }

    /// Commit + read the low `snap` bytes of the window for the escape-oracle snapshot — including
    /// reserved-**tail** pages the guest grew (or left `unmap`-ed / RO) via the Memory cap. `commit_rw`
    /// makes an uncommitted reserved page readable-as-zero and re-asserts RW on a `NOACCESS`/RO page,
    /// so the host read can't fault outside the guarded call. `snap` is clamped to the reservation
    /// (excluding the trailing guard page). Used only by the `_with_host` capture; the common path
    /// reads `[0, mapped)` directly. `None`, nothing read, when the OS refuses the commit (#2318).
    pub(crate) fn read_low(&self, snap: usize) -> Option<Vec<u8>> {
        if self.mapped == 0 || snap == 0 {
            return Some(Vec::new());
        }
        let max = self.total - trailing_guard(); // everything but the trailing guard page
        let snap = snap.min(max);
        let commit = round_up(snap, pal::page_size()).min(max);
        // SAFETY: `[base, base+commit)` lies in the reservation (≤ total − guard); once `commit_rw`
        // makes it committed + RW, the `[0, snap)` read is in-bounds and cannot fault.
        unsafe {
            pal::commit_rw(self.base, commit)
                .then(|| std::slice::from_raw_parts(self.base, snap).to_vec())
        }
    }

    /// Give the whole pages touched by `[offset, offset+len)` the protection `prot`: `Ro` for the D40
    /// const data segment (§3a/§4), whose later guest write faults into the guarded range; `None`
    /// for a durable-restored `Unmapped` page (DURABILITY.md §12.3), whose every access does, as on
    /// the frozen guest. The data must already be written (this only changes protection). A
    /// producer keeps RO data on its own pages; protection is page-granular, so a shared page would
    /// over-protect. `false` when the OS refuses (#2286): the window then lacks a protection its
    /// page map has, so the caller must not run a guest on it.
    #[must_use]
    pub(crate) fn protect(&self, offset: u64, len: u64, prot: Prot) -> bool {
        if self.mapped == 0 || len == 0 {
            return true;
        }
        let page = pal::page_size();
        let start = (offset as usize / page) * page;
        let end = round_up((offset + len) as usize, page);
        // SAFETY: `[base+start, base+end)` lies within the reservation (the caller bounds
        // `offset+len` by the backed region, or by a tail page it committed).
        unsafe { pal::protect(self.base.add(start), end - start, prot) }
    }

    /// Re-establish a captured page map on a freshly seeded window (DURABILITY.md §12.3): one
    /// [`crate::WindowProt`] per [`crate::DURABLE_SNAPSHOT_PAGE`] of `init` from the window's start,
    /// then each page of `tail` (#2427: those the window holds past `init`). In the first `mapped`
    /// bytes `Rw` entries keep the default; past them (the reserved tail) every page listed is one the
    /// window holds — grown, or protected to nothing with its contents kept (#1909) — so it is
    /// committed and seeded with its bytes (#1834). A thawed guest then faults on a restored
    /// `Ro`/`Unmapped` page exactly as the frozen one would — matching `temen-interp`'s
    /// `apply_layout`. The root's thaw and a detached child's (#1733) both apply theirs here. `false`
    /// when the OS refuses a commit or a protection (#2318): the thaw is refused then, as the window
    /// cannot hold the page map it captured.
    #[must_use]
    pub(crate) fn apply_prots(
        &self,
        prots: &[crate::WindowProt],
        mapped: u64,
        init: &[u8],
        tail: &[crate::TailPage],
    ) -> bool {
        if self.mapped == 0 {
            return true;
        }
        let page = crate::DURABLE_SNAPSHOT_PAGE as u64;
        let head = prots.iter().enumerate().map(|(i, &p)| {
            let off = i as u64 * page;
            (off, p, init.get(off as usize..).unwrap_or(&[]))
        });
        let tail = tail
            .iter()
            .map(|t| (t.off, t.prot, t.bytes.as_deref().unwrap_or(&[])));
        head.chain(tail)
            .all(|(off, p, bytes)| self.apply_page(off, p, mapped, bytes))
    }

    /// [`Self::apply_prots`] for the one codec page at `off`, given its bytes (from its start; a
    /// short or empty slice leaves the rest zero).
    fn apply_page(&self, off: u64, p: crate::WindowProt, mapped: u64, bytes: &[u8]) -> bool {
        let page = crate::DURABLE_SNAPSHOT_PAGE as u64;
        // Everything but the trailing guard page, as `read_low` bounds it.
        let limit = (self.total - trailing_guard()) as u64;
        if off >= mapped {
            if off + page > limit {
                return true;
            }
            let host = pal::page_size();
            let start = (off as usize / host) * host;
            let end = round_up((off + page) as usize, host);
            let n = bytes.len().min(page as usize);
            // SAFETY: `[start, end)` lies in the reservation below its guard page (checked above);
            // committed, it takes the `n` image bytes at its offset.
            unsafe {
                if !pal::commit_rw(self.base.add(start), end - start) {
                    return false;
                }
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.base.add(off as usize), n);
            }
        }
        match p {
            crate::WindowProt::Ro => self.protect(off, page, Prot::Ro),
            crate::WindowProt::Unmapped => self.protect(off, page, Prot::None),
            crate::WindowProt::Rw => true,
        }
    }

    /// #2427 — read the host pages at byte offsets `offs`, ascending (the pages a window holds past
    /// its declared size, [`crate::HeldTail`]), each run of consecutive ones committed read/write first
    /// so the read cannot fault outside the guarded call: what a durable capture carries past the
    /// window's prefix, read apart, never the span between them. A page whose bytes are all zero is
    /// left out (its offset says it is held). `None`, nothing read, when the OS refuses a commit
    /// (#2286).
    pub(crate) fn read_pages(&self, offs: &[u64]) -> Option<Vec<(u64, Vec<u8>)>> {
        let page = pal::page_size();
        let limit = self.total - trailing_guard();
        let offs: Vec<usize> = offs
            .iter()
            .map(|&off| off as usize)
            .filter(|&off| off + page <= limit)
            .collect();
        for run in offs.chunk_by(|a, b| b - a == page) {
            // SAFETY: the run lies in the reservation below its guard page (filtered above).
            if !unsafe { pal::commit_rw(self.base.add(run[0]), run.len() * page) } {
                return None;
            }
        }
        let pages = offs.into_iter().filter_map(|off| {
            // SAFETY: committed read/write above, so the read cannot fault.
            let bytes = unsafe { std::slice::from_raw_parts(self.base.add(off), page) };
            bytes
                .iter()
                .any(|&b| b != 0)
                .then(|| (off as u64, bytes.to_vec()))
        });
        Some(pages.collect())
    }

    /// #964/#1094: reserve `[0, guard)` inaccessible — the unconditional NULL guard: a NULL
    /// dereference faults into the guard handler (hardware enforcement at zero per-access cost),
    /// matching the interpreter oracle's `[0, guard)` `Unmapped` seeding.
    /// Page-granular, so it engages only when `guard` is host-page-exact (4 KiB / 16 KiB hosts;
    /// on a larger-page host protecting would swallow live scratch above the guard, so it skips —
    /// and the interpreter skips identically, keeping the tiers trap-parity per platform).
    /// `false` when the OS refuses (#2286): a NULL dereference would then read the window where
    /// the interpreter traps, so the caller must not run a guest on it.
    #[must_use]
    pub(crate) fn seed_null_guard(&self, guard: u64) -> bool {
        if self.mapped == 0 || guard == 0 || !(guard as usize).is_multiple_of(pal::page_size()) {
            return true;
        }
        // The marked layout puts the args blob and stack above the guard, so a marked window is
        // always larger than it.
        self.protect(0, guard, Prot::None)
    }

    /// #1768 — a **private duplicate** of this window for a fork twin (the JIT's
    /// `Mem::fork_private`): the same geometry, the bytes of every committed page, and the same
    /// protection on every page. `prots` names each host page (by index) whose state is not the
    /// default — the backed prefix defaults to read-write, the reserved tail to uncommitted — so it
    /// lists every committed tail page and every read-only or inaccessible prefix page.
    ///
    /// An inaccessible page may still hold bytes (a `protect(none)` keeps them; only an `unmap`
    /// zeroes), so each is read through a brief read-write window on `self` and put straight back:
    /// the caller guarantees nothing runs on this window meanwhile (its only vCPU is unwound). Every
    /// other page of `self` is left exactly as it was. `None`, `self` as it was, when the OS cannot
    /// give the duplicate (#2305): no room for its reservation, a page it cannot commit or protect, or
    /// an inaccessible page of `self` it cannot open for the read.
    ///
    /// # Safety
    /// No guest code runs on `self` during the call, and `prots` indexes only pages inside its
    /// reservation.
    pub(crate) unsafe fn fork_copy(&self, prots: &BTreeMap<usize, Prot>) -> Option<GuestWindow> {
        let twin = GuestWindow::try_new(self.mapped, self.total - trailing_guard())?;
        if self.mapped == 0 {
            return Some(twin);
        }
        let page = pal::page_size();
        let prefix = round_up(self.mapped, page);
        let at = |w: &GuestWindow, p: usize| w.base.add(p * page);
        // Open the inaccessible pages (prefix and committed tail) for the read — every one, or none
        // is read: a page still closed would fault the host.
        let mut opened = Vec::new();
        let open = prots
            .iter()
            .filter(|&(_, &prot)| prot == Prot::None)
            .all(|(&p, _)| {
                let ok = pal::protect(at(self, p), page, Prot::Rw);
                opened.extend(ok.then_some(p));
                ok
            });
        // The backed prefix in one copy; then each committed tail page, committed in the twin first.
        let copied = open && {
            std::ptr::copy_nonoverlapping(self.base, twin.base, prefix);
            prots.range(prefix / page..).all(|(&p, _)| {
                let ok = pal::commit_rw(at(&twin, p), page);
                if ok {
                    std::ptr::copy_nonoverlapping(at(self, p), at(&twin, p), page);
                }
                ok
            })
        };
        // Put `self` back. Closing a page it opened only merges what the opening split, so the OS
        // has no cause to refuse it.
        for p in opened {
            pal::protect(at(self, p), page, Prot::None);
        }
        // Give the twin the same protections: one it cannot take refuses the fork, as a twin that
        // could write a page its page map forbids would be wrong.
        let protected = || {
            prots
                .iter()
                .filter(|&(_, &prot)| prot != Prot::Rw)
                .all(|(&p, &prot)| pal::protect(at(&twin, p), page, prot))
        };
        (copied && protected()).then_some(twin)
    }

    /// The address range a fault must land in to be attributed to this window (the whole
    /// reservation, so the inaccessible tail + guard page are covered). `(0, 0)` when there is no
    /// window.
    pub(crate) fn fault_range(&self) -> (usize, usize) {
        if self.mapped == 0 {
            (0, 0)
        } else {
            (self.base as usize, self.base as usize + self.total)
        }
    }
}

/// The fault range a window of `reserved` bytes at `base` has — `[base, base + reserved + guard)`,
/// exactly what [`GuestWindow::fault_range`] reports for a live window built with that reservation —
/// recomputed from the base alone. A §14 child's module never records its live window (one compiled
/// child may run in several windows at once — the plain-child cache), so its `Jit.invoke` derives the
/// range from the window the thunk was handed.
pub(crate) fn fault_range_of(base: *mut u8, reserved: usize) -> (usize, usize) {
    let page = pal::page_size();
    let lo = base as usize;
    (lo, lo + round_up(reserved, page) + trailing_guard())
}

impl Drop for GuestWindow {
    fn drop(&mut self) {
        if !self.base.is_null() {
            // SAFETY: releasing exactly the reservation we created.
            unsafe { pal::release(self.base, self.total) };
        }
    }
}

/// Run the JIT entry `code` under the guard. Returns `true` if a fault in the window's guarded
/// range was caught and unwound (→ the caller reports `MemoryFault`).
///
/// # Safety
/// `code` must be the finalized trampoline with the [`Entry`] signature, and the pointers must
/// satisfy its contract (valid for the call, outliving it).
pub(crate) unsafe fn run_guarded(
    window: &GuestWindow,
    code: *const u8,
    args: *const i64,
    results: *mut i64,
    mem_base: *mut u8,
    fn_table: *const c_void,
    vmctx: *const crate::VmCtx,
) -> bool {
    pal::install_guard();
    let (lo, hi) = window.fault_range();
    let f: Entry = std::mem::transmute(code);
    // The entry ABI spells the context pointer as the trap cell it starts with (`VmCtx` field 0).
    pal::run_guarded(
        f,
        args,
        results,
        mem_base,
        fn_table,
        vmctx as *mut i64,
        lo,
        hi,
    )
}

/// Install the guard on the calling (worker) thread. Idempotent; the handler is process-wide but its
/// recovery state is thread-local, so each worker arms independently.
#[cfg(fiber_rt)]
pub(crate) fn install_guard() {
    pal::install_guard();
}

/// The trap stack the guard handler captured at the most recent caught memory fault on this thread,
/// read-and-cleared (§5 W3 trap-time backtrace): the faulting `pc` and the frame-pointer chain's raw
/// return addresses (the handler walked them while the guest stack was intact). `None` if nothing
/// was captured (no fault since the last take, or an arch the handler doesn't decode). The host
/// symbolizes them via [`CompiledModule::trap_backtrace`].
/// Returns `(faulting pc, return-address chain, fiber handle)` — the fiber handle is the guest fiber
/// that was running when the trap fired (per-fiber attribution, §5 W3 / §23-D57), or `-1` for the root
/// computation (no fiber).
pub(crate) fn take_trap_frame() -> Option<(usize, Vec<usize>, i64)> {
    pal::take_trap_frame()
}

/// Walk a frame-pointer chain under the trap-time capture's own fault recovery (§5 W3, #1487),
/// returning the return addresses it collected: `[0]` is this call's own return site, then one per
/// frame the chain yielded before it ended or ran off a mapped page.
///
/// Exposed for the `trap_walk` fuzz target, which gates the property the recovery exists for: the
/// walk reads the **guest's** stack, so its input is whatever the guest chose to put there, and for
/// *any* contents it must terminate, stay within the frame cap, and not take the host with it. Not a
/// runtime entry point — in production the trap detectors call the capture directly.
///
/// # Safety
/// Reads memory reachable from `fp` by the walk's own rules. A fault on any link is recovered, so an
/// arbitrary `fp` is sound; this is the whole point of the entry.
#[doc(hidden)]
#[cfg(unix)]
pub unsafe fn walk_trap_frame_chain(fp: usize) -> Vec<usize> {
    extern "C" {
        fn temen_capture_explicit_trap(guest_fp: usize);
    }
    pal::install_guard();
    temen_capture_explicit_trap(fp);
    take_trap_frame()
        .map(|(_, rets, _)| rets)
        .unwrap_or_default()
}

/// Run an `Entry`-shaped `code` with faults in `[lo, hi)` caught (detect-and-kill), for a window
/// fault range obtained from [`GuestWindow::fault_range`]. Used to run a fiber resume on a worker (a
/// guest memory fault inside the fiber unwinds back here — the fiber stack is abandoned, the domain is
/// being killed) and by `CompiledModule::invoke_extra` to nest a recovery inside an in-flight run
/// (re-entrant, like the §14 child path). Returns `true` if a guarded fault was caught.
///
/// # Safety
/// `code` must honour the [`Entry`] ABI and its pointer args must be valid for the call.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_guarded_range(
    code: *const u8,
    args: *const i64,
    results: *mut i64,
    mem_base: *mut u8,
    fn_table: *const c_void,
    vmctx: *const crate::VmCtx,
    lo: usize,
    hi: usize,
) -> bool {
    let f: Entry = std::mem::transmute(code);
    pal::run_guarded(
        f,
        args,
        results,
        mem_base,
        fn_table,
        vmctx as *mut i64,
        lo,
        hi,
    )
}

/// The trap code a caught guard fault reports.
pub(crate) const FAULT_TRAP: i64 = TrapKind::MemoryFault as i64;

/// The host page size — the §14 demand-paging granularity (what one fault supplies), and a fork
/// copy's (#1768).
pub(crate) fn page_size() -> usize {
    pal::page_size()
}

/// The inaccessible guard a window reserves past `round_up(reserved, page)`: one page. An access
/// that reaches up to this far past the reservation still faults, which the JIT leans on to let
/// accesses near a checked base skip their own check (`trusted_reach ≤ guard_size`, DESIGN.md
/// D61/D67): it reads its reach from here, so the guard cannot shrink under it.
pub(crate) fn trailing_guard() -> usize {
    pal::page_size()
}

/// Windows-only: make `[base, base+len)` committed read/write across this window's **placeholder**
/// reservation — idempotent (see the windows `pal::commit_rw`). Exposed so `temen-run`'s production
/// `MprotectWindow` Memory-cap backend, which operates on this *same* window, can commit/grow tail
/// pages without re-implementing the placeholder split/replace dance (a plain `VirtualAlloc(
/// MEM_COMMIT)` fails on a placeholder). On unix the equivalent is a plain `mprotect`, done inline.
/// `false` if the OS refused (#2268).
///
/// # Safety
/// `[base, base+len)` must lie within the JIT guest-window reservation that produced `base`.
#[cfg(windows)]
pub unsafe fn win_commit_rw(base: *mut u8, len: usize) -> bool {
    pal::commit_rw(base, len)
}

/// Windows-only: give the committed `[base, base+len)` the cap `prot` bits (`0` none, `1` read,
/// `2`/`3` read-write), one `VirtualProtect` per allocation the range spans (see the windows
/// `pal::protect`: a §13 view splits the reservation, and one call cannot span two). Exposed for
/// `temen-run`'s `MprotectWindow`, beside [`win_commit_rw`]. `false` if the OS refused (#2268).
///
/// # Safety
/// `[base, base+len)` must lie within the JIT guest-window reservation that produced `base`.
#[cfg(windows)]
pub unsafe fn win_protect(base: *mut u8, len: usize, prot: i32) -> bool {
    let prot = if prot & 2 != 0 {
        Prot::Rw
    } else if prot & 1 != 0 {
        Prot::Ro
    } else {
        Prot::None
    };
    pal::protect(base, len, prot)
}

// ================================= PAL: unix ==================================================
// `mmap` reservation + `mprotect` (PROT_NONE tail/guard) + a SIGSEGV/SIGBUS handler that
// `siglongjmp`s out of an in-window fault (the C shim `trap_shim.c`, for sound setjmp/longjmp).
#[cfg(unix)]
mod pal {
    use super::{Entry, Prot};
    use core::ffi::c_void;
    use std::sync::Once;

    extern "C" {
        fn temen_install_trap_handler();
        #[allow(clippy::too_many_arguments)]
        fn temen_run_guarded(
            f: Entry,
            a: *const i64,
            r: *mut i64,
            m: *mut u8,
            t: *const c_void,
            tc: *mut i64,
            lo: usize,
            hi: usize,
        ) -> i32;
        fn temen_take_trap_frame(pc: *mut usize, rets: *mut usize, max: i32) -> i32;
        fn temen_take_trap_fiber() -> i64;
    }

    /// Read and clear the trap stack the SIGSEGV/SIGBUS handler captured at the most recent caught
    /// guard fault (§5 W3 trap-time backtrace): the faulting `pc`, the frame-pointer chain's raw
    /// return addresses, and the fiber running at the trap (§23-D57; `-1` = root). `None` if none was
    /// captured. Must match `TEMEN_TRAP_MAXFRAMES` in the shim.
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>, i64)> {
        const MAX: usize = 64;
        let mut pc = 0usize;
        let mut rets = [0usize; MAX];
        // SAFETY: reads+clears the shim's thread-locals into `pc` + the `MAX`-slot `rets` buffer; the
        // fiber pairs with the same capture (read right after, before any other trap can overwrite it).
        let n = unsafe { temen_take_trap_frame(&mut pc, rets.as_mut_ptr(), MAX as i32) };
        let fiber = unsafe { temen_take_trap_fiber() };
        (n >= 0).then(|| (pc, rets[..n as usize].to_vec(), fiber))
    }

    pub(super) fn page_size() -> usize {
        // SAFETY: sysconf is always safe to call; _SC_PAGESIZE returns a positive size.
        let p = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
        if p > 0 {
            p as usize
        } else {
            4096
        }
    }

    /// Reserve `total` bytes of inaccessible virtual address space (no commit; `MAP_NORESERVE` so a
    /// huge reservation costs only VA). Returns the base, or null on failure.
    ///
    /// # Safety
    /// Returns a fresh mapping owned by the caller until [`release`].
    pub(super) unsafe fn reserve(total: usize) -> *mut u8 {
        let p = libc::mmap(
            std::ptr::null_mut(),
            total,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        );
        if p == libc::MAP_FAILED {
            std::ptr::null_mut()
        } else {
            p as *mut u8
        }
    }

    /// Commit `[base, base+len)` read/write.
    ///
    /// # Safety
    /// `[base, base+len)` must lie within a reservation from [`reserve`].
    pub(super) unsafe fn commit_rw(base: *mut u8, len: usize) -> bool {
        libc::mprotect(base as *mut c_void, len, libc::PROT_READ | libc::PROT_WRITE) == 0
    }

    /// Set the protection of a committed range `[base, base+len)`.
    ///
    /// # Safety
    /// `[base, base+len)` must be a committed sub-range of a window reservation.
    pub(super) unsafe fn protect(base: *mut u8, len: usize, prot: Prot) -> bool {
        let p = match prot {
            Prot::Rw => libc::PROT_READ | libc::PROT_WRITE,
            Prot::Ro => libc::PROT_READ,
            Prot::None => libc::PROT_NONE,
        };
        libc::mprotect(base as *mut c_void, len, p) == 0
    }

    /// Release a whole reservation from [`reserve`].
    ///
    /// # Safety
    /// `base`/`total` must be exactly a mapping returned by [`reserve`].
    pub(super) unsafe fn release(base: *mut u8, total: usize) {
        libc::munmap(base as *mut c_void, total);
    }

    static INSTALL: Once = Once::new();

    pub(super) fn install_guard() {
        // SAFETY: installs the process-wide SIGSEGV/SIGBUS handler exactly once.
        INSTALL.call_once(|| unsafe { temen_install_trap_handler() });
    }

    /// Run `f` under the installed handler; `true` if an in-`[lo,hi)` fault was caught.
    ///
    /// # Safety
    /// `f` and its pointer args must honour the [`Entry`] contract for the call.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_guarded(
        f: Entry,
        a: *const i64,
        r: *mut i64,
        m: *mut u8,
        t: *const c_void,
        tc: *mut i64,
        lo: usize,
        hi: usize,
    ) -> bool {
        temen_run_guarded(f, a, r, m, t, tc, lo, hi) != 0
    }
}

// ================================= PAL: windows ===============================================
// `VirtualAlloc2(MEM_RESERVE_PLACEHOLDER)` reservation + `VirtualProtect` (PAGE_NOACCESS tail/guard)
// + a **Vectored Exception Handler** that, on an in-window access violation, resumes at the guarded
// call's landing pad to unwind out of the fault — the windows analogue of unix's signal +
// siglongjmp. A **placeholder** reservation (rather than a plain `MEM_RESERVE`) is what lets `temen-run`'s §13
// `SharedRegion` path alias a shared section into a fixed window sub-range via
// `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` (issue #1). Pure Rust via `windows-sys` (no C shim), so
// the path stays `cargo check/clippy --target *-windows-*`-able on a non-windows host.
//
// Validated locally by cross-compiling to `x86_64-pc-windows-msvc` (`cargo-xwin`) and running the
// suite under **wine** (the PAL conformance test + the interp↔JIT differential, incl. §13
// `SharedRegion` aliasing); the gating runtime check remains the `cross-os` `windows-latest` CI run.
#[cfg(windows)]
mod pal {
    use super::{Entry, Prot};
    use core::cell::Cell;
    use core::ffi::c_void;
    use std::sync::Once;
    use windows_sys::Win32::Foundation::{GetLastError, HANDLE};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, CONTEXT, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{
        UnmapViewOfFile2, VirtualAlloc2, VirtualFree, VirtualProtect, VirtualQuery,
        MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MEM_COMMIT, MEM_MAPPED,
        MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
        MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, PAGE_PROTECTION_FLAGS, PAGE_READONLY,
        PAGE_READWRITE,
    };
    use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    /// `GetLastError()` as an `i64`, for thread-into-panic-messages debuggability (CI has no debugger,
    /// so a failing Win32 call names its error code in the test output).
    fn last_error() -> i64 {
        // SAFETY: GetLastError reads thread-local state; always safe.
        unsafe { GetLastError() as i64 }
    }

    /// `ERROR_COMMITMENT_LIMIT` — the system-wide commit charge (RAM + page file) is momentarily
    /// exhausted. Unlike unix `mmap` (overcommit), Windows charges every committed page up front, so a
    /// tight CI page file under churn (e.g. the 4000-window fuzz loop) can transiently hit this. It is
    /// **transient** (clears as other allocations free / the page file grows), so the commit path
    /// retries it with backoff rather than failing the run.
    const ERROR_COMMITMENT_LIMIT: i64 = 1455;
    /// Bounded retries on a transient commit-limit (backoff 5·2^k ms ⇒ ~0.3 s worst case, then give up
    /// loudly). Enough to ride out a momentary spike without ever hanging on a genuine exhaustion.
    const COMMIT_RETRIES: u32 = 6;

    // VEH return codes + the access-violation status (kept local to avoid version-specific paths).
    const EXCEPTION_CONTINUE_EXECUTION: i32 = -1;
    const EXCEPTION_CONTINUE_SEARCH: i32 = 0;
    const STATUS_ACCESS_VIOLATION: i32 = 0xC000_0005u32 as i32;
    // The Cranelift `trapnz` confinement trap surfaces as an illegal-instruction exception (the
    // windows analogue of unix SIGILL); see `veh` and DESIGN.md §4/§5.
    const STATUS_ILLEGAL_INSTRUCTION: i32 = 0xC000_001Du32 as i32;

    pub(super) fn page_size() -> usize {
        // SAFETY: GetSystemInfo only writes the out-param; always safe.
        let mut si: SYSTEM_INFO = unsafe { core::mem::zeroed() };
        unsafe { GetSystemInfo(&mut si) };
        let p = si.dwPageSize as usize;
        if p == 0 {
            4096
        } else {
            p
        }
    }

    /// Reserve `total` bytes of inaccessible VA as a **placeholder** (`VirtualAlloc2` with
    /// `MEM_RESERVE_PLACEHOLDER`). A placeholder — rather than a plain `MEM_RESERVE` — is what lets a
    /// later `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` alias a shared section into a *fixed* sub-range
    /// of this window (the §13 `SharedRegion` path in `temen-run`); a plain reservation cannot host a
    /// fixed-address view. The reservation costs only address space until a sub-range is committed.
    ///
    /// # Safety
    /// Returns a fresh reservation owned by the caller until [`release`].
    pub(super) unsafe fn reserve(total: usize) -> *mut u8 {
        VirtualAlloc2(
            0 as HANDLE,
            core::ptr::null(),
            total,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            core::ptr::null_mut(),
            0,
        ) as *mut u8
    }

    /// Make `[base, base+len)` committed + read/write — **idempotently**, across a placeholder
    /// reservation. A *placeholder* sub-range is split out (`MEM_PRESERVE_PLACEHOLDER`) to its exact
    /// bounds and replace-committed (`MEM_REPLACE_PLACEHOLDER`, RW, OS-zero-filled); an already-
    /// *committed* sub-range (private memory or a mapped §13 view) is merely re-asserted read/write
    /// with `VirtualProtect`, so its live contents are **never re-zeroed**. Walking the region map
    /// with `VirtualQuery` is what makes this safe to call on overlapping / growing ranges — the
    /// snapshot path ([`read_low`](super::GuestWindow::read_low)) re-commits a prefix that is already
    /// live, and a plain `VirtualAlloc(MEM_COMMIT)` cannot commit a placeholder at all (it fails).
    /// `false` if the OS refused a step; the sub-ranges before it stay committed.
    ///
    /// # Safety
    /// `[base, base+len)` must lie within a reservation from [`reserve`].
    pub(super) unsafe fn commit_rw(base: *mut u8, len: usize) -> bool {
        let end = base as usize + len;
        let mut addr = base as usize;
        while addr < end {
            let mut mbi: MEMORY_BASIC_INFORMATION = core::mem::zeroed();
            let n = VirtualQuery(
                addr as *const c_void,
                &mut mbi,
                core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            );
            if n == 0 {
                return false;
            }
            let region_base = mbi.BaseAddress as usize;
            let region_end = region_base + mbi.RegionSize;
            let a = addr.max(region_base);
            let b = end.min(region_end);
            if mbi.State == MEM_COMMIT {
                // Already committed (private or a mapped view): re-assert RW, keep contents.
                let mut old: PAGE_PROTECTION_FLAGS = 0;
                if VirtualProtect((a as *const c_void) as _, b - a, PAGE_READWRITE, &mut old) == 0 {
                    return false;
                }
            } else {
                // Placeholder (`MEM_RESERVE`): carve the exact sub-range, then replace-commit it RW.
                if (a != region_base || b != region_end)
                    && VirtualFree(
                        a as *mut c_void,
                        b - a,
                        MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                    ) == 0
                {
                    return false;
                }
                // Replace-commit the placeholder sub-range RW. `MEM_REPLACE_PLACEHOLDER` is atomic
                // (success → ptr, failure → null leaving the placeholder intact), so retrying a
                // transient `ERROR_COMMITMENT_LIMIT` is safe.
                let mut p = core::ptr::null_mut();
                for attempt in 0..COMMIT_RETRIES {
                    p = VirtualAlloc2(
                        0 as HANDLE,
                        a as *const c_void,
                        b - a,
                        MEM_RESERVE | MEM_COMMIT | MEM_REPLACE_PLACEHOLDER,
                        PAGE_READWRITE,
                        core::ptr::null_mut(),
                        0,
                    );
                    if !p.is_null() || last_error() != ERROR_COMMITMENT_LIMIT {
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(5u64 << attempt));
                }
                if p.is_null() {
                    return false;
                }
            }
            addr = b;
        }
        true
    }

    /// Set the protection of a committed range `[base, base+len)`.
    ///
    /// Walks the range with `VirtualQuery` and protects each committed region's intersection
    /// separately: unlike unix `mprotect`, `VirtualProtect` **cannot span distinct allocations**,
    /// and a §13 region map splits the window reservation into several (committed-private
    /// fragments + mapped section views). A single spanning call then fails wholesale — leaving
    /// e.g. the #1094 NULL-guard pages `NOACCESS` after `restore_rw`, so the post-run snapshot's
    /// raw window read AVs in host code (the windows `jit_diff` STATUS_ACCESS_VIOLATION).
    /// Non-committed fragments (placeholders) are skipped — they can't be protected. `false` if the
    /// OS refused a step; the regions before it keep the new protection.
    ///
    /// # Safety
    /// `[base, base+len)` must lie within a window reservation.
    pub(super) unsafe fn protect(base: *mut u8, len: usize, prot: Prot) -> bool {
        let flags: PAGE_PROTECTION_FLAGS = match prot {
            Prot::Rw => PAGE_READWRITE,
            Prot::Ro => PAGE_READONLY,
            Prot::None => PAGE_NOACCESS,
        };
        let end = base as usize + len;
        let mut addr = base as usize;
        while addr < end {
            let mut mbi: MEMORY_BASIC_INFORMATION = core::mem::zeroed();
            if VirtualQuery(
                addr as *const c_void,
                &mut mbi,
                core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            ) == 0
            {
                return false;
            }
            let region_end = mbi.BaseAddress as usize + mbi.RegionSize;
            let b = end.min(region_end);
            let mut old: PAGE_PROTECTION_FLAGS = 0;
            if mbi.State == MEM_COMMIT
                && b > addr
                && VirtualProtect(addr as *const c_void, b - addr, flags, &mut old) == 0
            {
                return false;
            }
            addr = if region_end > addr {
                region_end
            } else {
                addr + page_size()
            };
        }
        true
    }

    /// Release a whole reservation from [`reserve`] — **every fragment of it**. After [`commit_rw`]
    /// (and the §13 `map_region` view path in `temen-run`) have split the original placeholder into
    /// independent sub-allocations — committed-private regions, leftover placeholders, mapped views —
    /// a single `VirtualFree(base, 0, MEM_RELEASE)` frees only the *first* fragment and **leaks the
    /// rest**: grown-tail committed pages accumulate as commit charge across many window teardowns
    /// until an allocation fails (the intermittent `windows-latest` `jit_fuzz` OOM/fastfail). So walk
    /// `[base, base+total)` with `VirtualQuery` and tear down each region — unmap shared-section views,
    /// release placeholders / private commits. Defensive: teardown must never fault, so individual
    /// Win32 failures are ignored (the address space is reclaimed wholesale at process exit anyway).
    ///
    /// # Safety
    /// `base`/`total` must be exactly a reservation returned by [`reserve`].
    pub(super) unsafe fn release(base: *mut u8, total: usize) {
        let end = base as usize + total;
        let mut addr = base as usize;
        let proc = GetCurrentProcess();
        while addr < end {
            let mut mbi: MEMORY_BASIC_INFORMATION = core::mem::zeroed();
            if VirtualQuery(
                addr as *const c_void,
                &mut mbi,
                core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
            ) == 0
            {
                break;
            }
            let region_base = mbi.BaseAddress as usize;
            let next = region_base + mbi.RegionSize;
            if mbi.Type == MEM_MAPPED {
                // A §13 shared-section view: unmap it back to a placeholder, then release that.
                UnmapViewOfFile2(
                    proc,
                    MEMORY_MAPPED_VIEW_ADDRESS {
                        Value: region_base as *mut c_void,
                    },
                    MEM_PRESERVE_PLACEHOLDER,
                );
                VirtualFree(region_base as *mut c_void, 0, MEM_RELEASE);
            } else if mbi.State == MEM_COMMIT {
                // A committed-private fragment. Free by its **exact queried range**, never by
                // `mbi.AllocationBase`: after §13 interior view maps split the original commit, the
                // committed remainders can still report the window base as their allocation base —
                // a range this walk already released, which a concurrent thread may have reused, so
                // a `VirtualFree(AllocationBase, 0, MEM_RELEASE)` here frees *another thread's live
                // allocation* (the parallel-test STATUS_ACCESS_VIOLATION). Shrink the fragment to a
                // placeholder (exact bounds), then release that placeholder.
                VirtualFree(
                    region_base as *mut c_void,
                    mbi.RegionSize,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                VirtualFree(region_base as *mut c_void, 0, MEM_RELEASE);
            } else if mbi.State == MEM_RESERVE {
                // A leftover placeholder fragment: its region base is its allocation base; release it.
                VirtualFree(region_base as *mut c_void, 0, MEM_RELEASE);
            }
            addr = if next > addr {
                next
            } else {
                addr + page_size()
            };
        }
    }

    // ---- the guard: AddVectoredExceptionHandler + an asm guarded-call trampoline ---------------
    //
    // #1792 — the recovery point is the **trampoline's own return**, never a point inside a Rust
    // function. `run_guarded` used to capture a `CONTEXT` with `RtlCaptureContext` and have the VEH
    // restore it, resuming *inside* `run_guarded` a second time: a call returning twice, which LLVM
    // was never told (C's `setjmp` is `returns_twice`; `RtlCaptureContext` is not). What the resumed
    // code found in its registers and stack slots then depended on the build's register allocation
    // and frame layout, so some builds crashed on every guarded fault and others never did.
    //
    // `temen_win_guarded_call(f, a, r, m, t, tc, jb)` saves the Win64 non-volatile registers (`rbx`,
    // `rbp`, `rdi`, `rsi`, `r12`–`r15`, `xmm6`–`xmm15`) and its entry `rsp` into `jb`, calls
    // `f(a, r, m, t, tc)` and returns 0. On a guarded fault the VEH loads those registers from `jb`
    // into the faulting context and resumes at `temen_win_guarded_landing`, which returns 1 from the
    // trampoline's frame. Either way Rust sees one ordinary call that returns once, with its
    // non-volatile registers intact. The trampoline carries SEH unwind info, so a stack walk through
    // it (an exception dispatched from inside `f`) unwinds correctly; the landing pad is a leaf.
    core::arch::global_asm!(
        ".globl temen_win_guarded_call",
        ".def temen_win_guarded_call",
        ".scl 2",
        ".type 32",
        ".endef",
        ".seh_proc temen_win_guarded_call",
        "temen_win_guarded_call:",
        "sub rsp, 0x38",
        ".seh_stackalloc 0x38",
        ".seh_endprologue",
        // `jb`, the 7th argument, sits at entry `rsp + 0x38` — `rsp + 0x70` after the prologue.
        "mov rax, [rsp + 0x70]",
        "mov [rax + 0x00], rbx",
        "mov [rax + 0x08], rbp",
        "mov [rax + 0x10], rdi",
        "mov [rax + 0x18], rsi",
        "mov [rax + 0x20], r12",
        "mov [rax + 0x28], r13",
        "mov [rax + 0x30], r14",
        "mov [rax + 0x38], r15",
        "lea r10, [rsp + 0x38]", // entry `rsp`: it points at our return address
        "mov [rax + 0x40], r10",
        "movdqu [rax + 0x50], xmm6",
        "movdqu [rax + 0x60], xmm7",
        "movdqu [rax + 0x70], xmm8",
        "movdqu [rax + 0x80], xmm9",
        "movdqu [rax + 0x90], xmm10",
        "movdqu [rax + 0xa0], xmm11",
        "movdqu [rax + 0xb0], xmm12",
        "movdqu [rax + 0xc0], xmm13",
        "movdqu [rax + 0xd0], xmm14",
        "movdqu [rax + 0xe0], xmm15",
        // f(a, r, m, t, tc): four register arguments, the fifth above the 32-byte home area.
        "mov r10, rcx",
        "mov rcx, rdx",
        "mov rdx, r8",
        "mov r8, r9",
        "mov r9, [rsp + 0x60]",  // `t`, entry `rsp + 0x28`
        "mov rax, [rsp + 0x68]", // `tc`, entry `rsp + 0x30`
        "mov [rsp + 0x20], rax",
        "call r10",
        "xor eax, eax",
        "add rsp, 0x38",
        "ret",
        ".seh_endproc",
        ".globl temen_win_guarded_landing",
        "temen_win_guarded_landing:",
        "mov eax, 1",
        "ret",
    );

    /// The registers [`temen_win_guarded_call`] saves at entry, in the layout its asm writes.
    #[repr(C)]
    struct JmpBuf {
        /// `rbx`, `rbp`, `rdi`, `rsi`, `r12`, `r13`, `r14`, `r15`, then the entry `rsp`.
        gpr: [u64; 9],
        _pad: u64,
        /// `xmm6`–`xmm15`, low then high quadword.
        xmm: [[u64; 2]; 10],
    }

    extern "C" {
        /// See the asm above: `1` if a guarded fault resumed at the landing pad, else `0`.
        fn temen_win_guarded_call(
            f: Entry,
            a: *const i64,
            r: *mut i64,
            m: *mut u8,
            t: *const c_void,
            tc: *mut i64,
            jb: *mut JmpBuf,
        ) -> u32;
        fn temen_win_guarded_landing();
    }

    /// Point the faulting `ctx` at the landing pad with the registers `jb` saved: execution resumes
    /// as `temen_win_guarded_call` returning 1, in its caller's frame.
    fn resume_at_landing(ctx: &mut CONTEXT, jb: &JmpBuf) {
        let [rbx, rbp, rdi, rsi, r12, r13, r14, r15, rsp] = jb.gpr;
        ctx.Rbx = rbx;
        ctx.Rbp = rbp;
        ctx.Rdi = rdi;
        ctx.Rsi = rsi;
        ctx.R12 = r12;
        ctx.R13 = r13;
        ctx.R14 = r14;
        ctx.R15 = r15;
        ctx.Rsp = rsp;
        ctx.Rip = temen_win_guarded_landing as *const () as usize as u64;
        let m = |i: usize| windows_sys::Win32::System::Diagnostics::Debug::M128A {
            Low: jb.xmm[i][0],
            High: jb.xmm[i][1] as i64,
        };
        // SAFETY: the x64 `CONTEXT`'s float area is always the legacy/XMM view on this arch; the
        // union's two members alias the same bytes.
        let x = unsafe { &mut ctx.Anonymous.Anonymous };
        x.Xmm6 = m(0);
        x.Xmm7 = m(1);
        x.Xmm8 = m(2);
        x.Xmm9 = m(3);
        x.Xmm10 = m(4);
        x.Xmm11 = m(5);
        x.Xmm12 = m(6);
        x.Xmm13 = m(7);
        x.Xmm14 = m(8);
        x.Xmm15 = m(9);
    }

    #[derive(Clone, Copy)]
    struct Frame {
        /// The active trampoline's saved registers (a stack local of `run_guarded`).
        jb: *const JmpBuf,
        lo: usize,
        hi: usize,
    }
    thread_local! {
        // The active guarded call's window range + saved registers (None ⇒ no guarded call).
        static GUARD: Cell<Option<Frame>> = const { Cell::new(None) };
        // §5 W3 trap-time backtrace: the faulting `(pc, frame-pointer-chain return addresses)` the VEH
        // captures from the access-violation `CONTEXT` *before* it overwrites that context with the
        // recovery one. Read + cleared by `take_trap_frame`. A fixed buffer (no allocation in the VEH).
        static TRAP_VALID: Cell<bool> = const { Cell::new(false) };
        static TRAP_PC: Cell<usize> = const { Cell::new(0) };
        static TRAP_RETS: Cell<[usize; TRAP_MAXFRAMES]> = const { Cell::new([0; TRAP_MAXFRAMES]) };
        static TRAP_N: Cell<usize> = const { Cell::new(0) };
        // Per-fiber attribution (§5 W3 / §23-D57): the fiber running when the fault fired, snapshotted
        // in the VEH (the C current-fiber TLS the fiber runtime maintains). `-1` = root (no fiber).
        static TRAP_FIBER: Cell<i64> = const { Cell::new(-1) };
    }

    /// Max trap-backtrace frames captured (matches the unix shim's `TEMEN_TRAP_MAXFRAMES`).
    const TRAP_MAXFRAMES: usize = 64;

    /// Walk the frame-pointer chain from `fp` into `out`, returning the count — the Rust analog of the
    /// unix shim's `temen_walk_fp_chain` (DEBUGGING.md §5/W3). The JIT's `preserve_frame_pointers` gives
    /// every guest frame a `{ saved_fp, ret_addr }` record (`*fp` = caller's saved fp, `*(fp+1)` = its
    /// return address). Bounded — aligned, strictly-increasing links inside `[lo, hi)`, the frame
    /// cap — so a corrupt chain terminates instead of looping or reading wild memory.
    ///
    /// `[lo, hi)` is the running thread's committed stack. The walk runs inside the VEH, where a fault
    /// of its own has no recovery and kills the process, and `fp` is whatever the faulting `Rbp` held:
    /// code without frame pointers leaves any value there. So every record read must lie in memory
    /// known to be mapped (#1487, #1575).
    ///
    /// # Safety
    /// `[lo, hi)` must be committed, readable memory (the thread's stack, from its TEB).
    unsafe fn walk_fp_chain(
        fp: usize,
        lo: usize,
        hi: usize,
        out: &mut [usize; TRAP_MAXFRAMES],
    ) -> usize {
        const RECORD: usize = 2 * core::mem::size_of::<usize>(); // { saved_fp, ret_addr }
        let align = core::mem::size_of::<usize>() - 1;
        let mut cur = fp;
        let mut n = 0;
        while n < TRAP_MAXFRAMES
            && cur & align == 0
            && cur >= lo
            && cur.checked_add(RECORD).is_some_and(|end| end <= hi)
        {
            let next = *(cur as *const usize);
            let ret = *((cur + core::mem::size_of::<usize>()) as *const usize);
            out[n] = ret;
            n += 1;
            if next <= cur {
                break; // frame pointers grow toward the base; a non-increasing link is the end
            }
            cur = next;
        }
        n
    }

    /// Capture the trap-time backtrace from the faulting `CONTEXT` (§5/W3): the faulting `Rip` is the
    /// innermost frame (symbolized directly by the host), and the `Rbp` chain gives the callers. Called
    /// from the VEH while the guest stack is still intact, before the context is pointed at the landing
    /// pad.
    ///
    /// # Safety
    /// `ctx` is the live faulting context for an in-window access violation in guest JIT code.
    unsafe fn capture_trap_frame(ctx: &CONTEXT) {
        let mut rets = [0usize; TRAP_MAXFRAMES];
        // The TEB's `StackLimit`/`StackBase` name the committed stack the fault ran on; the fiber
        // switch keeps them current, so a fault on a fiber bounds the walk to that fiber's stack.
        let (base, limit, _) = teb_stack_fields();
        let n = walk_fp_chain(ctx.Rbp as usize, limit as usize, base as usize, &mut rets);
        TRAP_PC.with(|c| c.set(ctx.Rip as usize));
        TRAP_RETS.with(|c| c.set(rets));
        TRAP_N.with(|c| c.set(n));
        // Snapshot the running fiber at fault time (the C trap-capture TLS the fiber runtime maintains).
        extern "C" {
            fn temen_current_fiber() -> i64;
        }
        TRAP_FIBER.with(|c| c.set(unsafe { temen_current_fiber() }));
        TRAP_VALID.with(|c| c.set(true));
    }

    unsafe extern "system" fn veh(ep: *mut EXCEPTION_POINTERS) -> i32 {
        let ep = &*ep;
        let rec = &*ep.ExceptionRecord;
        if rec.ExceptionCode == STATUS_ACCESS_VIOLATION {
            // ExceptionInformation[1] is the faulting address for an access violation.
            let addr = rec.ExceptionInformation[1];
            if let Some(f) = GUARD.with(|g| g.get()) {
                if addr >= f.lo && addr < f.hi {
                    // §5 W3: capture the trap-time backtrace from the faulting context *before*
                    // `resume_at_landing` below rewrites it.
                    capture_trap_frame(&*ep.ContextRecord);
                    // Resume as the trampoline returning 1 (the unix siglongjmp analogue). The
                    // abandoned JIT frames hold no Rust destructors.
                    resume_at_landing(&mut *ep.ContextRecord, &*f.jb);
                    return EXCEPTION_CONTINUE_EXECUTION;
                }
            }
        } else if rec.ExceptionCode == STATUS_ILLEGAL_INSTRUCTION {
            // The Cranelift `trapnz` confinement bounds-check trap (memory-fault lowering, §4/§5) —
            // the windows analogue of unix SIGILL. There is no faulting data address (the fault is
            // the instruction itself), so unlike an access violation the [lo,hi) test does not apply:
            // an illegal instruction inside an armed guarded guest call is our emitted memory-fault
            // trap. Recover exactly like a guarded access violation (unwind to `run_guarded`).
            if let Some(f) = GUARD.with(|g| g.get()) {
                capture_trap_frame(&*ep.ContextRecord);
                resume_at_landing(&mut *ep.ContextRecord, &*f.jb);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }

    static INSTALL: Once = Once::new();
    pub(super) fn install_guard() {
        INSTALL.call_once(|| {
            // first = 1 → our handler runs before any previously-registered one.
            let h = unsafe { AddVectoredExceptionHandler(1, Some(veh)) };
            assert!(
                !h.is_null(),
                "temen-jit: AddVectoredExceptionHandler failed"
            );
        });
    }

    /// Read and clear the most recent caught trap's stack (§5/W3): the faulting `pc` + the
    /// frame-pointer chain's return addresses. Two capture paths feed it on Windows — a **memory
    /// fault** lands in the VEH (Rust thread-local, above), and an **explicit-check trap** (div-by-zero
    /// etc.) lands in the shared `trap_capture.c` helper (its C thread-local, `temen_take_trap_frame`).
    /// A run trips at most one, so check the Rust capture first and fall back to the C one. `None` if
    /// neither captured.
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>, i64)> {
        if TRAP_VALID.with(|c| c.get()) {
            TRAP_VALID.with(|c| c.set(false));
            let pc = TRAP_PC.with(|c| c.get());
            let n = TRAP_N.with(|c| c.get());
            let rets = TRAP_RETS.with(|c| c.get());
            let fiber = TRAP_FIBER.with(|c| c.get());
            return Some((pc, rets[..n].to_vec(), fiber));
        }
        // Explicit-check trap: the C helper (`trap_capture.c`) stashed it in its own thread-local.
        extern "C" {
            fn temen_take_trap_frame(pc: *mut usize, rets: *mut usize, max: i32) -> i32;
            fn temen_take_trap_fiber() -> i64;
        }
        const MAX: usize = 64;
        let (mut pc, mut rets) = (0usize, [0usize; MAX]);
        // SAFETY: reads+clears the shim's thread-locals into `pc` + the `MAX`-slot buffer.
        let n = unsafe { temen_take_trap_frame(&mut pc, rets.as_mut_ptr(), MAX as i32) };
        let fiber = unsafe { temen_take_trap_fiber() };
        (n >= 0).then(|| (pc, rets[..n as usize].to_vec(), fiber))
    }

    /// The TEB stack-bound fields (`StackBase`/`StackLimit`/`DeallocationStack` — the same trio the
    /// fiber switch swaps, at the same x64 offsets as `temen-fiber`'s `switch_x86_64_windows.rs`).
    /// A trap inside a *fiber* unwinds to `run_guarded`'s recovery point on the root stack via a
    /// `CONTEXT` restore, which does **not** touch the TEB — so without an explicit restore the
    /// thread keeps the fiber's stack fields, and at thread exit the kernel frees
    /// `DeallocationStack` = the fiber's arena slot. When that slot is an arena **allocation base**
    /// the whole control-stack arena is released and the next fiber bootstrap write AVs (the
    /// windows `jit_fibers` STATUS_ACCESS_VIOLATION, reproduced + verified under Wine). Non-x86-64
    /// windows has no fiber switch (fiber_rt is x86-64-gated), so nothing to restore there.
    #[cfg(target_arch = "x86_64")]
    unsafe fn teb_stack_fields() -> (u64, u64, u64) {
        let (b, l, d): (u64, u64, u64);
        core::arch::asm!(
            "mov {t}, gs:[0x30]",
            "mov {b}, [{t} + 0x08]",
            "mov {l}, [{t} + 0x10]",
            "mov {d}, [{t} + 0x1478]",
            t = out(reg) _, b = out(reg) b, l = out(reg) l, d = out(reg) d,
            options(nostack, preserves_flags),
        );
        (b, l, d)
    }
    #[cfg(target_arch = "x86_64")]
    unsafe fn restore_teb_stack_fields(fields: (u64, u64, u64)) {
        core::arch::asm!(
            "mov {t}, gs:[0x30]",
            "mov [{t} + 0x08], {b}",
            "mov [{t} + 0x10], {l}",
            "mov [{t} + 0x1478], {d}",
            t = out(reg) _, b = in(reg) fields.0, l = in(reg) fields.1, d = in(reg) fields.2,
            options(nostack, preserves_flags),
        );
    }
    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn teb_stack_fields() {}
    #[cfg(not(target_arch = "x86_64"))]
    unsafe fn restore_teb_stack_fields(_fields: ()) {}

    /// Run `f` under the handler; `true` if an in-`[lo,hi)` access violation was caught.
    ///
    /// # Safety
    /// `f` and its pointer args must honour the [`Entry`] contract for the call.
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_guarded(
        f: Entry,
        a: *const i64,
        r: *mut i64,
        m: *mut u8,
        t: *const c_void,
        tc: *mut i64,
        lo: usize,
        hi: usize,
    ) -> bool {
        // Save the caller's (possibly-armed parent) guard frame to restore on exit — re-entrant so a
        // §14 child guest can run (in its own window) inside the parent's guarded call. A child fault
        // resumes at the child's landing pad; the parent's frame is restored afterwards intact.
        let prev = GUARD.with(|g| g.replace(None));
        // Snapshot the TEB stack fields: a fiber-side fault resumes at the landing pad without the
        // fiber switch-back, and the context rewrite leaves the fiber's TEB values in place (see
        // `teb_stack_fields`) — restore them on the tripped path.
        let teb = teb_stack_fields();
        let mut jb = JmpBuf {
            gpr: [0; 9],
            _pad: 0,
            xmm: [[0; 2]; 10],
        };
        let jb_ptr: *mut JmpBuf = &mut jb;
        GUARD.with(|g| g.set(Some(Frame { jb: jb_ptr, lo, hi })));
        // SAFETY: `f` honours the `Entry` ABI (the caller's contract); `jb` outlives the call, and
        // the VEH reads it only while this frame is armed.
        let tripped = temen_win_guarded_call(f, a, r, m, t, tc, jb_ptr) != 0;
        if tripped {
            restore_teb_stack_fields(teb); // undo a fiber-side fault's leftover TEB stack fields
        }
        GUARD.with(|g| g.set(prev)); // restore the parent's frame
        tripped
    }
}

// ============================ PAL: unsupported targets ========================================
// The escape guarantee (and the guard the §4 elision leans on) needs a guard-page + fault-recovery
// PAL. unix and windows have one (above); any *other* target (no-MMU / wasm) is unsupported and
// refuses to build rather than weaken the guarantee.
#[cfg(not(any(unix, windows)))]
mod pal {
    use super::{Entry, Prot};
    use core::ffi::c_void;

    compile_error!(
        "temen-jit needs a guard-page Platform Abstraction Layer: unix (mmap + mprotect(PROT_NONE) + \
         a SIGSEGV/SIGBUS handler) and windows (VirtualAlloc + VirtualProtect + a Vectored \
         Exception Handler) exist; other targets (no-MMU / wasm) are unsupported. See DESIGN.md §4 \
         \"Platform support\"."
    );

    // Stubs so the portable code's references resolve to that single, clear error rather than a
    // pile of "unresolved name" follow-ons.
    pub(super) fn page_size() -> usize {
        4096
    }
    pub(super) unsafe fn reserve(_total: usize) -> *mut u8 {
        core::ptr::null_mut()
    }
    pub(super) unsafe fn commit_rw(_base: *mut u8, _len: usize) -> bool {
        false
    }
    pub(super) unsafe fn protect(_base: *mut u8, _len: usize, _prot: Prot) -> bool {
        false
    }
    pub(super) unsafe fn release(_base: *mut u8, _total: usize) {}
    pub(super) fn install_guard() {}
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn run_guarded(
        _f: Entry,
        _a: *const i64,
        _r: *mut i64,
        _m: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
        _lo: usize,
        _hi: usize,
    ) -> bool {
        false
    }
    pub(super) fn take_trap_frame() -> Option<(usize, Vec<usize>, i64)> {
        None
    }
}

// ============================ PAL conformance test ============================================
// Platform-agnostic: it drives the window model + the guard PAL directly (no JIT), so the *same*
// test validates each platform's PAL — unix now, windows on CI once that leg lands.
#[cfg(test)]
mod tests {
    use super::*;

    /// `vm.max_map_count`, when it is small enough to fill in a test (#2286).
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(crate) fn max_map_count() -> Option<usize> {
        let max: usize = std::fs::read_to_string("/proc/sys/vm/max_map_count")
            .ok()?
            .trim()
            .parse()
            .ok()?;
        (max <= 1 << 22).then_some(max)
    }

    /// Fill this process's mapping table up to `max` (`vm.max_map_count`), then free `spare` mappings:
    /// a scratch reservation, every other page made readable until the OS refuses, each one splitting
    /// it into two more mappings. Allocates nothing.
    ///
    /// # Safety
    /// Only in a forked child that leaves with `_exit`: nothing else in the process may need a mapping
    /// afterwards.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    pub(crate) unsafe fn fill_mapping_table(max: usize, spare: usize) {
        let page = pal::page_size();
        let len = 2 * (max + 16) * page;
        let scratch = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_NONE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
            -1,
            0,
        ) as *mut u8;
        if scratch as *mut c_void == libc::MAP_FAILED {
            return;
        }
        let mut made = 0;
        while 2 * made * page < len
            && libc::mprotect(
                scratch.add(2 * made * page) as *mut c_void,
                page,
                libc::PROT_READ,
            ) == 0
        {
            made += 1;
        }
        // A readable page between two inaccessible ones is a mapping of its own (the first may have
        // merged with whatever lies below the scratch, so it is not one of them).
        for i in (1..made).take(spare) {
            libc::munmap(scratch.add(2 * i * page) as *mut c_void, page);
        }
    }

    // Entry-shaped probes (the trampoline ABI). They do a single volatile read at a fixed window
    // offset; if it faults inside the guarded range the guard unwinds and `run_guarded` reports it.
    extern "C" fn read_at_0(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // SAFETY: offset 0 is in the committed prefix for the test's window.
        unsafe {
            let _ = core::ptr::read_volatile(mem);
        }
    }
    extern "C" fn read_in_tail(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // 512 KiB lands in the reserved-but-inaccessible tail of the test's window → faults.
        // SAFETY: the read is expected to fault; the guard catches it (or the test fails loudly).
        unsafe {
            let _ = core::ptr::read_volatile(mem.add(512 << 10));
        }
    }

    fn guarded(
        win: &GuestWindow,
        f: extern "C" fn(*const i64, *mut i64, *mut u8, *const c_void, *mut i64),
    ) -> bool {
        let vm = crate::VmCtx::new(crate::InstanceAddrs::NONE);
        // SAFETY: `f` honours the Entry ABI; `mem_base` is this window's base.
        unsafe {
            run_guarded(
                win,
                f as *const u8,
                std::ptr::null(),
                std::ptr::null_mut(),
                win.base(),
                std::ptr::null(),
                &vm,
            )
        }
    }

    #[test]
    fn pal_guard_catches_tail_fault_not_in_window() {
        // 64 KiB committed inside a 1 MiB reservation: offset 0 is live, 512 KiB is in the tail.
        let win = GuestWindow::try_new(64 << 10, 1 << 20).expect("window");
        assert!(
            !guarded(&win, read_at_0),
            "an in-window read must complete without a guard fault"
        );
        assert!(
            guarded(&win, read_in_tail),
            "a read in the reserved-but-inaccessible tail must be caught by the guard"
        );
    }

    /// Where `read_in_tail_with_a_wild_rbp` points `rbp`: inaccessible memory *outside* the guarded
    /// window, so a fault there is not a guard fault the handler would recover.
    #[cfg(all(windows, target_arch = "x86_64"))]
    static WILD_RBP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    /// As `read_in_tail`, but first points `rbp` at [`WILD_RBP`]. The guard's
    /// trap-backtrace walk starts from the faulting `rbp`, and code without frame pointers (this
    /// probe, or any host code a fault lands in) leaves whatever value there. The walk runs inside the
    /// fault handler, where a fault of its own has no recovery, so it must stop rather than read.
    #[cfg(all(windows, target_arch = "x86_64"))]
    extern "C" fn read_in_tail_with_a_wild_rbp(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // SAFETY: the read is expected to fault in-window; the guard resumes at its landing pad with
        // the caller's registers (`rbp` included), so the `pop` never runs and the stack is not left
        // unbalanced.
        unsafe {
            core::arch::asm!(
                "push rbp",
                "mov rbp, {wild}",
                "mov {tmp}, byte ptr [{addr}]",
                "pop rbp",
                wild = in(reg) WILD_RBP.load(std::sync::atomic::Ordering::Relaxed),
                addr = in(reg) mem.add(512 << 10),
                tmp = out(reg_byte) _,
            );
        }
    }

    /// #1575: the windows memory-fault handler's backtrace walk dereferenced any aligned `rbp` within
    /// 8 MiB of itself, so a non-frame `rbp` naming unmapped memory faulted inside the handler and
    /// killed the process. It now reads only inside the thread's committed stack.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_guard_fault_with_a_wild_frame_pointer_is_caught_not_fatal() {
        let win = GuestWindow::try_new(64 << 10, 1 << 20).expect("window");
        let elsewhere = GuestWindow::try_new(64 << 10, 1 << 20).expect("window");
        WILD_RBP.store(
            elsewhere.base() as usize + (512 << 10),
            std::sync::atomic::Ordering::Relaxed,
        );
        assert!(
            guarded(&win, read_in_tail_with_a_wild_rbp),
            "the tail fault is caught although rbp names no frame"
        );
    }

    /// As `read_in_tail`, but first overwrites every Win64 non-volatile register — `rbx`, `rbp`,
    /// `rsi`, `rdi`, `r12`–`r15`, `xmm6`–`xmm15` — so only the guard's landing can give the caller
    /// its values back.
    #[cfg(all(windows, target_arch = "x86_64"))]
    extern "C" fn clobber_then_fault(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const c_void,
        _tc: *mut i64,
    ) {
        // SAFETY: the read faults in-window and the guard resumes at its landing pad, so the pops
        // never run; the declared outputs cover every register the asm writes.
        unsafe {
            core::arch::asm!(
                "push rbx",
                "push rbp",
                "mov rbx, {junk}",
                "mov rbp, {junk}",
                "mov rsi, {junk}",
                "mov rdi, {junk}",
                "mov r12, {junk}",
                "mov r13, {junk}",
                "mov r14, {junk}",
                "mov r15, {junk}",
                "movq xmm6, {junk}",
                "movq xmm7, {junk}",
                "movq xmm8, {junk}",
                "movq xmm9, {junk}",
                "movq xmm10, {junk}",
                "movq xmm11, {junk}",
                "movq xmm12, {junk}",
                "movq xmm13, {junk}",
                "movq xmm14, {junk}",
                "movq xmm15, {junk}",
                "mov {tmp}, byte ptr [{addr}]",
                "pop rbp",
                "pop rbx",
                junk = in(reg) 0xdead_beef_dead_beef_u64,
                addr = in(reg) mem.add(512 << 10),
                tmp = out(reg_byte) _,
                out("rsi") _, out("rdi") _,
                out("r12") _, out("r13") _, out("r14") _, out("r15") _,
                out("xmm6") _, out("xmm7") _, out("xmm8") _, out("xmm9") _, out("xmm10") _,
                out("xmm11") _, out("xmm12") _, out("xmm13") _, out("xmm14") _, out("xmm15") _,
            );
        }
    }

    #[cfg(all(windows, target_arch = "x86_64"))]
    static CLOBBER_CAUGHT: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);

    /// A plain Rust caller of a guarded call, entered from the test's asm below.
    #[cfg(all(windows, target_arch = "x86_64"))]
    extern "C" fn guarded_clobber() {
        let win = GuestWindow::try_new(64 << 10, 1 << 20).expect("window");
        CLOBBER_CAUGHT.store(
            guarded(&win, clobber_then_fault),
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// #1792 — a caught guard fault returns to the guarded call's caller with every Win64
    /// non-volatile register it had, although the faulting code overwrote all of them. The caller is
    /// asm, so the values sit in the registers themselves rather than wherever a Rust frame might
    /// spill them: only the guard's landing (restoring what the trampoline saved) can put them back.
    #[cfg(all(windows, target_arch = "x86_64"))]
    #[test]
    fn a_guarded_fault_returns_the_callers_nonvolatile_registers() {
        const S: u64 = 0x5a5a_0000_0000_0000;
        let (mut r12, mut r13, mut r14, mut r15, mut rsi, mut rdi) =
            (S | 12, S | 13, S | 14, S | 15, S | 6, S | 7);
        let mut x: [f64; 10] = core::array::from_fn(|i| f64::from_bits(S | (0x60 + i as u64)));
        let rbx_kept: u8;
        let rbp_kept: u8;
        // SAFETY: `guarded_clobber` is an `extern "C"` fn of no arguments; the call runs on a
        // 16-byte-aligned stack with its 32-byte home area, and `rsp`, `rbx` and `rbp` are restored
        // before the asm ends. Every volatile register is declared clobbered (`clobber_abi`).
        unsafe {
            core::arch::asm!(
                "push rbx",
                "push rbp",
                "mov rbx, rax",
                "mov rbp, rsp", // `rbp` must still hold this after the call
                "and rsp, -16",
                "sub rsp, 48",
                "mov [rsp + 40], rbp", // …so keep a copy above the callee's home area
                "call {f}",
                "mov rax, [rsp + 40]",
                "cmp rbp, rax",
                "sete dl",
                "mov rsp, rax",
                "mov rax, {sentinel}",
                "cmp rbx, rax",
                "sete cl",
                "pop rbp",
                "pop rbx",
                f = sym guarded_clobber,
                sentinel = const 0x5a5a_0000_0000_00bb_u64,
                inout("rax") 0x5a5a_0000_0000_00bb_u64 => _,
                out("cl") rbx_kept,
                out("dl") rbp_kept,
                inout("r12") r12,
                inout("r13") r13,
                inout("r14") r14,
                inout("r15") r15,
                inout("rsi") rsi,
                inout("rdi") rdi,
                inout("xmm6") x[0],
                inout("xmm7") x[1],
                inout("xmm8") x[2],
                inout("xmm9") x[3],
                inout("xmm10") x[4],
                inout("xmm11") x[5],
                inout("xmm12") x[6],
                inout("xmm13") x[7],
                inout("xmm14") x[8],
                inout("xmm15") x[9],
                clobber_abi("C"),
            );
        }
        assert!(
            CLOBBER_CAUGHT.load(std::sync::atomic::Ordering::Relaxed),
            "the fault was caught"
        );
        assert_eq!(rbx_kept, 1, "rbx");
        assert_eq!(rbp_kept, 1, "rbp");
        assert_eq!(
            [r12, r13, r14, r15, rsi, rdi],
            [S | 12, S | 13, S | 14, S | 15, S | 6, S | 7]
        );
        for (i, v) in x.iter().enumerate() {
            assert_eq!(v.to_bits(), S | (0x60 + i as u64), "xmm{}", i + 6);
        }
    }

    #[test]
    fn pal_window_is_writable_and_released() {
        let mut win = GuestWindow::try_new(8 << 10, 64 << 10).expect("window");
        let w = win.rw_mut();
        w[0] = 0xAB;
        w[(8 << 10) - 1] = 0xCD;
        assert_eq!(win.rw_mut()[0], 0xAB);
        assert_eq!(win.rw_mut()[(8 << 10) - 1], 0xCD);
        // dropped here — `release` must not fault.
    }

    // The windows placeholder allocator splits its reservation into independent fragments —
    // committed-private regions (the prefix and grown-tail pages) and leftover placeholders (plus
    // shared-section views in `temen-run`'s §13 path). Teardown (`pal::release`) must free **every**
    // fragment: a single `VirtualFree(base, 0, MEM_RELEASE)` frees only the *first* and leaks the
    // rest, whose commit charge accumulated across `jit_fuzz`'s teardowns until an allocation failed
    // (the intermittent `windows-latest` crash). This pins the no-leak contract: fragment a
    // reservation the way production does, release it, then `VirtualQuery` the original range and
    // assert not one byte remains mapped/committed/reserved. (Non-vacuous: the pre-fix single-release
    // leaks all-but-the-first fragment here.)
    //
    // That walk reads the address map of the whole process. Any allocation another thread makes
    // after the release can land in the freed range and read as a leak, and `cargo test` runs the
    // other tests on other threads meanwhile (#1793). So the check runs alone: the test re-runs
    // itself as the only test in a child process of this binary.
    #[cfg(windows)]
    #[test]
    fn pal_release_frees_all_placeholder_fragments_no_leak() {
        use windows_sys::Win32::System::Memory::{
            VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_FREE,
        };

        const ALONE: &str = "TEMEN_PAL_RELEASE_ALONE";
        if std::env::var_os(ALONE).is_none() {
            let name = "mem::tests::pal_release_frees_all_placeholder_fragments_no_leak";
            let out = std::process::Command::new(std::env::current_exe().expect("test binary"))
                .args(["--exact", name, "--test-threads=1"])
                .env(ALONE, "1")
                .output()
                .expect("re-run the test binary");
            let stdout = String::from_utf8_lossy(&out.stdout);
            // "1 passed", not just a zero exit: a filter that matched nothing would exit 0 too.
            assert!(
                out.status.success() && stdout.contains("1 passed"),
                "the no-leak check failed in its own process:\n{stdout}{}",
                String::from_utf8_lossy(&out.stderr)
            );
            return;
        }

        let page = pal::page_size();
        let total = 64 * page; // room for several disjoint fragments
                               // SAFETY: a fresh placeholder reservation, released below.
        let base = unsafe { pal::reserve(total) };
        assert!(!base.is_null(), "reserve failed");
        // Fragment it the way production does: commit the prefix, then two *non-adjacent* tail pages
        // (each an interior split + replace-commit out of the big placeholder), so the reservation is
        // now a mix of committed-private regions and leftover placeholders.
        // SAFETY: every range lies within `[base, base+total)`.
        unsafe {
            assert!(pal::commit_rw(base, page)); // prefix
            assert!(pal::commit_rw(base.add(20 * page), page)); // a "grown" tail page
            assert!(pal::commit_rw(base.add(40 * page), page)); // another, non-adjacent
        }
        // SAFETY: release exactly the reservation created above.
        unsafe { pal::release(base, total) };

        // Walk the original range: every region must now be `MEM_FREE` (nothing leaked). This process
        // runs no other test, so nothing allocates into the freed VA before the walk reads it.
        let lo = base as usize;
        let hi = lo + total;
        let mut addr = lo;
        let mut leaked = 0usize;
        while addr < hi {
            let mut mbi: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
            // SAFETY: `VirtualQuery` accepts any address and only writes `mbi`.
            let n = unsafe {
                VirtualQuery(
                    addr as *const c_void,
                    &mut mbi,
                    core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                )
            };
            if n == 0 {
                break;
            }
            let region_end = mbi.BaseAddress as usize + mbi.RegionSize;
            let overlap = region_end
                .min(hi)
                .saturating_sub(addr.max(mbi.BaseAddress as usize));
            if mbi.State != MEM_FREE {
                leaked += overlap;
            }
            addr = if region_end > addr {
                region_end
            } else {
                addr + page
            };
        }
        assert_eq!(
            leaked, 0,
            "pal::release leaked {leaked} bytes of the placeholder reservation \
             (fragments past the first not freed)"
        );
    }

    /// #2286 — a run whose window the OS will not seed ends with `ThreadFault` before its first
    /// instruction, as one with no room for its window does (#2305), instead of running with its
    /// read-only data writable and its NULL guard open. In a forked child that fills its own mapping
    /// table, then frees just enough for the window itself: the window alone is granted (checked);
    /// its read-only page and its NULL guard each need a split the table has no room for.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn a_run_whose_window_cannot_be_seeded_ends_with_thread_fault() {
        const WRONG_RAN: i32 = 1;
        const WRONG_OTHER: i32 = 2;
        const INCONCLUSIVE: i32 = 3;
        let Some(max) = max_map_count() else {
            eprintln!("skipped: vm.max_map_count is too large to fill");
            return;
        };
        // Its read-only page lies inside the backed prefix, above the NULL guard: protecting it
        // splits the prefix's mapping in two more.
        let m = temen_text::parse_module(
            "memory 16\ndata ro 32768 \"const\"\nfunc () -> (i64) {\nblock 0 () {\n  v = i64.const 7\n  return v\n  }\n}\n",
        )
        .expect("parse");
        let mut cm = crate::compile(&m, 0).expect("compile");
        assert_eq!(
            cm.run(&[], None, None).expect("run").0,
            crate::JitOutcome::Returned(vec![7])
        );
        let (mapped, reserved) = (cm.win_mapped, cm.win_reserved);
        // SAFETY: the child runs only the code below, allocating nothing once the table is full, and
        // leaves with `_exit`.
        match unsafe { libc::fork() } {
            0 => {
                // SAFETY: the child's own alarm: a child that hangs dies of SIGALRM, so the test fails
                // rather than hangs.
                unsafe { libc::alarm(60) };
                // SAFETY: the forked child's own table; two spare mappings, the window's own two.
                unsafe { fill_mapping_table(max, 2) };
                let wrong = if GuestWindow::try_new(mapped, reserved).is_none() {
                    INCONCLUSIVE
                } else {
                    match cm.run(&[], None, None) {
                        Ok((crate::JitOutcome::Trapped(crate::TrapKind::ThreadFault), _)) => 0,
                        Ok((crate::JitOutcome::Returned(_), _)) => WRONG_RAN,
                        _ => WRONG_OTHER,
                    }
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
                    "the child died of signal {}: it faulted, or (SIGALRM) hung",
                    libc::WTERMSIG(status)
                );
                match libc::WEXITSTATUS(status) {
                    0 => {}
                    INCONCLUSIVE => eprintln!("inconclusive: the window alone was refused"),
                    WRONG_RAN => panic!("the run ran on a window missing its protections"),
                    _ => panic!("the run did not end with ThreadFault"),
                }
            }
        }
    }

    /// #2286/#2318 — at the process's mapping limit, each window operation the host makes that needs
    /// another mapping reports the refusal, where it used to panic or leave a protection missing. Run
    /// in a forked child, which fills its own mapping table and leaves this process's alone; once the
    /// table is full the child allocates nothing.
    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn window_operations_report_a_refusal_at_the_mapping_limit() {
        const WRONG_PROTECT: i32 = 1;
        const WRONG_THAW_TAIL: i32 = 4;
        const WRONG_THAW_RO: i32 = 8;
        const WRONG_RESTORE: i32 = 16;
        const WRONG_READ: i32 = 32;
        let page = pal::page_size();
        let Some(max) = max_map_count() else {
            eprintln!("skipped: vm.max_map_count is too large to fill");
            return;
        };
        let at = |pages: usize| (pages * page) as u64;
        // 16 backed pages in a 64-page reservation: one read-write, one with its prefix closed, so
        // that re-opening it splits the mapping it shares with the inaccessible tail.
        let open = GuestWindow::try_new(16 * page, 64 * page).expect("window");
        let closed = GuestWindow::try_new(16 * page, 64 * page).expect("window");
        assert!(closed.protect(0, at(16), Prot::None));
        // A thaw's page maps, built now: a page the guest grew in the tail; a read-only prefix page.
        let snap = crate::DURABLE_SNAPSHOT_PAGE;
        let mut grown = vec![crate::WindowProt::Unmapped; 20 * page / snap + 1];
        grown[..16 * page / snap].fill(crate::WindowProt::Rw);
        grown[20 * page / snap] = crate::WindowProt::Rw;
        let mut ro = vec![crate::WindowProt::Rw; 2 * page / snap + 1];
        ro[2 * page / snap] = crate::WindowProt::Ro;
        // SAFETY: the child runs only the code below, allocating nothing once the table is full, and
        // leaves with `_exit`.
        match unsafe { libc::fork() } {
            0 => {
                // SAFETY: the child's own alarm: a child that hangs dies of SIGALRM, so the test fails
                // rather than hangs.
                unsafe { libc::alarm(60) };
                // SAFETY: the forked child's own table.
                unsafe { fill_mapping_table(max, 0) };
                let mut wrong = 0;
                // A read-only data page inside the read-write prefix splits it. (A NULL guard is the
                // same `protect` at the window's start, where it may merge with the mapping below
                // instead of splitting, so it can succeed here.)
                if open.protect(at(2), at(1), Prot::Ro) {
                    wrong |= WRONG_PROTECT;
                }
                // A thaw commits the tail page the guest grew inside the inaccessible tail, and
                // protects a read-only page inside the prefix.
                if open.apply_prots(&grown, at(16), &[], &[]) {
                    wrong |= WRONG_THAW_TAIL;
                }
                if open.apply_prots(&ro, at(16), &[], &[]) {
                    wrong |= WRONG_THAW_RO;
                }
                // Re-opening the closed prefix for a snapshot, or committing past it for one, splits
                // the mapping it shares with the tail.
                if closed.restore_rw() {
                    wrong |= WRONG_RESTORE;
                }
                if closed.read_low(20 * page).is_some() {
                    wrong |= WRONG_READ;
                }
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
                    "the child died of signal {}: a window operation panicked, faulted or (SIGALRM) hung",
                    libc::WTERMSIG(status)
                );
                let wrong = libc::WEXITSTATUS(status);
                let names = [
                    (WRONG_PROTECT, "protect"),
                    (WRONG_THAW_TAIL, "apply_prots (a grown tail page)"),
                    (WRONG_THAW_RO, "apply_prots (a read-only page)"),
                    (WRONG_RESTORE, "restore_rw"),
                    (WRONG_READ, "read_low"),
                ];
                let passed: Vec<&str> = names
                    .iter()
                    .filter(|(bit, _)| wrong & bit != 0)
                    .map(|(_, name)| *name)
                    .collect();
                assert!(
                    passed.is_empty(),
                    "at the mapping limit these reported success: {passed:?}"
                );
            }
        }
        // Away from the limit, each one applies.
        assert!(open.protect(at(2), at(1), Prot::Ro));
        assert!(open.seed_null_guard(at(1)));
        assert!(open.apply_prots(&grown, at(16), &[], &[]));
        assert!(closed.restore_rw());
        assert_eq!(closed.read_low(20 * page).map(|b| b.len()), Some(20 * page));
    }
}

// ======================= trap-walk fault recovery (#1487) =====================================
// A fault *inside* the trap-time backtrace walk must cost the backtrace, not the host.
//
// `temen_walk_fp_chain` (DEBUGGING.md §5 W3) follows the guest's frame-pointer chain at a trap. Its
// loop bounds the *arithmetic* — aligned, non-null, strictly increasing, inside an 8 MiB span — but
// nothing there establishes that a link is **mapped**, and nothing can: the walk reads the guest's
// own stack, and the guest chooses what is in it. On unix the walk runs from the SIGSEGV handler
// *after* it has disarmed, so a fault on a bad link used to fall through to `SIG_DFL` and kill the
// host — turning a recoverable guest `MemoryFault` into a dead host process.
//
// The repros named in #1487 (`null_guard_native`, `jit_code_memory`) only crash under a lucky mmap
// layout — the bad slot's value has to land above the faulting frame pointer and inside the span — so
// they are not a gate. These drive the walk directly with a chain built to fault, which is.
//
// Unit tests rather than an integration test: the `trap_capture.c` symbols are linked into this
// crate, not re-exported from its rlib. Arch-independent — the chain is synthetic, so no register or
// frame layout is assumed — so `fiber_rt` (where `install_guard`/`run_guarded` exist) also puts this
// under macOS/aarch64 in the `cross-os` lane. **Unix only**: the windows capture keeps its own walk,
// which this recovery does not cover — see #1575.
#[cfg(all(test, unix, fiber_rt))]
mod trap_walk_tests {
    use core::ptr;

    extern "C" {
        fn temen_capture_explicit_trap(guest_fp: usize);
        fn temen_take_trap_frame(pc: *mut usize, rets: *mut usize, max: i32) -> i32;
    }

    /// Two readable frames followed by a link into an inaccessible page, all inside the walk's span.
    ///
    /// Three consecutive pages, the third `PROT_NONE`/`PAGE_NOACCESS`. Frame 0 sits at the base and
    /// frame 1 at `+16`, each a `{ saved_fp, ret_addr }` record; frame 1's `saved_fp` points at page
    /// 2. Every one of the loop's tests passes on that link — non-null, pointer-aligned, greater than
    /// the current frame, ~8 KiB into the span — so the walk dereferences it and faults. Same shape as
    /// the real failure, where the link is the `mem_base` the JIT entry trampoline spilled and the
    /// page it lands on is the window's null guard.
    struct FaultingChain {
        base: *mut u8,
        len: usize,
        page: usize,
    }

    impl FaultingChain {
        fn new() -> Self {
            let page = page_size();
            let len = page * 3;
            let base = reserve_rw(len);
            let wall = unsafe { base.add(page * 2) }; // the wall the walk runs into
            make_inaccessible(wall, page);

            let w = base.cast::<usize>();
            unsafe {
                w.write(base as usize + 16); // frame 0: saved_fp -> frame 1
                w.add(1).write(0xf00d_0001); //          ret
                w.add(2).write(wall as usize); // frame 1: saved_fp -> the inaccessible page
                w.add(3).write(0xf00d_0002); //          ret
            }
            FaultingChain { base, len, page }
        }

        fn fp(&self) -> usize {
            self.base as usize
        }

        /// The guarded range that *contains the wall* — i.e. the walk's fault will look to the trap
        /// detector exactly like a guest window fault. This is #1487's observed shape.
        fn wall_range(&self) -> (usize, usize) {
            let wall = self.base as usize + self.page * 2;
            (wall, wall + self.page)
        }
    }

    impl Drop for FaultingChain {
        fn drop(&mut self) {
            release(self.base, self.len);
        }
    }

    fn page_size() -> usize {
        unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
    }
    fn reserve_rw(len: usize) -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(p, libc::MAP_FAILED, "mmap");
        p.cast()
    }
    fn make_inaccessible(at: *mut u8, len: usize) {
        assert_eq!(
            unsafe { libc::mprotect(at.cast(), len, libc::PROT_NONE) },
            0,
            "mprotect"
        );
    }
    fn release(base: *mut u8, len: usize) {
        unsafe { libc::munmap(base.cast(), len) };
    }

    fn take() -> Option<(usize, Vec<usize>)> {
        const MAX: usize = 64;
        let mut pc = 0usize;
        let mut rets = [0usize; MAX];
        // SAFETY: reads and clears the shim's thread-locals into the `MAX`-slot buffer.
        let n = unsafe { temen_take_trap_frame(&mut pc, rets.as_mut_ptr(), MAX as i32) };
        (n >= 0).then(|| (pc, rets[..n as usize].to_vec()))
    }

    /// The gate: a walk that runs off a mapped page returns, and returns what it collected.
    ///
    /// Reaching the assertions at all is most of the point — without the recovery bracket this dies
    /// with SIGSEGV inside the handler rather than failing.
    #[test]
    fn a_walk_that_faults_truncates_the_backtrace_instead_of_killing_the_host() {
        super::install_guard();
        let chain = FaultingChain::new();

        unsafe { temen_capture_explicit_trap(chain.fp()) };

        let (pc, rets) = take().expect("the capture stands even though the walk was cut short");
        assert_eq!(
            pc, 0,
            "an explicit trap has no faulting pc; the trap site is rets[0]"
        );
        // rets[0] is the capture helper's own return address (inside this test), then the two frames
        // read before the bad link. The third link faults and contributes nothing.
        assert_eq!(
            rets.len(),
            3,
            "trap site + the two readable frames: {rets:x?}"
        );
        assert_eq!(rets[1], 0xf00d_0001);
        assert_eq!(rets[2], 0xf00d_0002);
    }

    /// Entry-shaped probe: walk the chain whose base arrives as the window pointer.
    extern "C" fn walk_from_mem(
        _a: *const i64,
        _r: *mut i64,
        mem: *mut u8,
        _t: *const core::ffi::c_void,
        _tc: *mut i64,
    ) {
        // SAFETY: `mem` is the `FaultingChain` base the caller passed; the walk only reads.
        unsafe { temen_capture_explicit_trap(mem as usize) };
    }

    /// **The ordering.** `temen_walk_in_progress()` is checked *before* the armed-range test, and
    /// this is what says so: the wall the walk faults on is inside the guarded `[lo, hi)`, which is
    /// #1487's observed shape (its gdb trace has both faults at exactly `g_lo`, the window's null
    /// guard). A detector that tested the range first would treat the walk's own fault as a fresh
    /// guest window fault — reporting a `MemoryFault` the guest never took, and on unix re-entering
    /// the capture from a half-written one.
    ///
    /// So the assertion is that the guarded call reports **no** fault: the walk stood down, its
    /// backtrace truncated, and the guest's run was left alone.
    #[test]
    fn the_walk_stands_down_before_the_armed_range_test() {
        super::install_guard();
        let chain = FaultingChain::new();
        let (lo, hi) = chain.wall_range();

        // SAFETY: `walk_from_mem` honours the `Entry` ABI and ignores every argument but `mem`, which
        // carries the chain base; `[lo, hi)` is the chain's own inaccessible page.
        let tripped = unsafe {
            super::pal::run_guarded(
                walk_from_mem,
                ptr::null(),
                ptr::null_mut(),
                chain.base,
                ptr::null(),
                ptr::null_mut(),
                lo,
                hi,
            )
        };

        assert!(
            !tripped,
            "the walk's own fault was reported to the host as a guest memory fault"
        );
        let (_, rets) = take().expect("the capture stands");
        assert_eq!(rets.len(), 3, "{rets:x?}");
        assert_eq!(rets[1], 0xf00d_0001);
        assert_eq!(rets[2], 0xf00d_0002);
    }

    /// A truncated walk disarms itself, so a later fault still reaches its normal disposition instead
    /// of being swallowed as "the walk faulted". Checked by walking the faulting chain twice: the
    /// second call only behaves identically if the first cleared the in-progress flag on its way out.
    #[test]
    fn a_truncated_walk_disarms_itself() {
        super::install_guard();
        let chain = FaultingChain::new();

        unsafe { temen_capture_explicit_trap(chain.fp()) };
        let first = take().expect("first capture");
        unsafe { temen_capture_explicit_trap(chain.fp()) };
        let second = take().expect("second capture: the flag was cleared, so this walked again");

        assert_eq!(first.1.len(), second.1.len());
        assert_eq!(&first.1[1..], &second.1[1..]);
    }

    /// A chain that ends cleanly is unaffected — the bracket costs the normal path nothing.
    #[test]
    fn a_well_formed_chain_still_walks_to_its_end() {
        super::install_guard();
        let mut buf = vec![0usize; 512];
        let base = buf.as_mut_ptr();
        unsafe {
            base.write(base as usize + 16);
            base.add(1).write(0xbeef_0001);
            base.add(2).write(0); // a null saved_fp terminates the walk
            base.add(3).write(0xbeef_0002);
        }

        unsafe { temen_capture_explicit_trap(base as usize) };

        let (_, rets) = take().expect("capture");
        assert_eq!(rets.len(), 3, "{rets:x?}");
        assert_eq!(rets[1], 0xbeef_0001);
        assert_eq!(rets[2], 0xbeef_0002);
    }
}
