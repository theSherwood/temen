//! #2996 — the address range of every live JIT code arena, which the fault handler reads to tell a
//! fault in emitted code (a guest fault) from one in host code (a host bug).
//!
//! Every [`JITModule`] this crate makes takes its code and its read-only data from one contiguous
//! [`ArenaMemoryProvider`] reservation of [`ARENA`] bytes ([`arena_module`]), so one `[base, base +
//! ARENA)` per module covers everything it will ever emit: the module's functions, the lazily
//! compiled ones, extras, §22 units, the entry and hosting trampolines. The range is registered
//! before the module has any code and released after its code has stopped ([`CodeSlot`]), so it
//! never changes while code from it can run.
//!
//! ## The slot protocol
//!
//! The table is fixed (no allocation, no pointer to swap): `N` slots of `{ seq, lo, hi }`, scanned
//! up to `used`. Writers — [`Table::claim`] and [`Table::release`] — serialise on `writers`, which
//! the handler never takes, and write `lo`/`hi` inside a seqlock: `seq` odd, the two words, `seq`
//! even (release). The handler only loads ([`Table::read`]). A reader that sees `seq` odd, or a
//! different `seq` after reading the words, **skips the slot**: it is being claimed or released, so
//! no code from it is running, and no fault the reader is classifying can be in it. A reader
//! therefore never pairs one module's `lo` with another module's `hi`, nor with an empty slot's zero.
//!
//! Ordering: a slot is published (the release increment of `seq`) before the module's first
//! function exists, and every function reaches a vCPU through a later release (its `fn` table slot,
//! a lazily compiled function's publication, or the thread start that hands over the code), which
//! that vCPU acquires before it jumps. So the fault, and the handler on that vCPU, happen after the
//! slot's publication and see it.
//!
//! When the table is full a new module is refused (`JitError`), so code is never left out of it.

#[cfg(loom)]
use loom::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(loom)]
use loom::sync::{Mutex, MutexGuard};
#[cfg(not(loom))]
use std::sync::atomic::{fence, AtomicUsize, Ordering};
#[cfg(not(loom))]
use std::sync::{Mutex, MutexGuard};

/// How many JIT modules may be live at once: 2^16, so the OS's own limit binds first. On default
/// Linux a process holds at most 65,530 mappings (`vm.max_map_count`), and each arena is at least
/// one; on Windows each arena charges 256 MiB of commit. A raised map count, or macOS, can reach
/// the table first; what should bound live code arenas in any case is the spawning guest's
/// accounting (#3006). Costs ~1.6 MB of untouched BSS; the claim scan stops at the first free
/// slot, and the handler's scan, which runs only on a fault, stops at the high-water mark.
const SLOTS: usize = 1 << 16;

/// The size of each module's code arena.
pub(crate) const ARENA: usize = 256 << 20;

struct Slot {
    seq: AtomicUsize,
    lo: AtomicUsize,
    hi: AtomicUsize,
}

impl Slot {
    #[cfg(not(loom))]
    #[allow(clippy::declare_interior_mutable_const)]
    const EMPTY: Slot = Slot {
        seq: AtomicUsize::new(0),
        lo: AtomicUsize::new(0),
        hi: AtomicUsize::new(0),
    };

    /// Write `[lo, hi)` inside the seqlock. Under `writers`.
    fn write(&self, lo: usize, hi: usize) {
        self.seq.fetch_add(1, Ordering::Relaxed); // odd: being written
        fence(Ordering::Release);
        self.lo.store(lo, Ordering::Relaxed);
        self.hi.store(hi, Ordering::Relaxed);
        self.seq.fetch_add(1, Ordering::Release); // even: published
    }
}

/// The table: `N` slots, the scan bound, and the writers' lock with which slots are taken.
pub(crate) struct Table<const N: usize> {
    slots: [Slot; N],
    used: AtomicUsize,
    writers: Mutex<[bool; N]>,
}

impl<const N: usize> Table<N> {
    #[cfg(not(loom))]
    const fn new() -> Self {
        Table {
            slots: [Slot::EMPTY; N],
            used: AtomicUsize::new(0),
            writers: Mutex::new([false; N]),
        }
    }

    #[cfg(loom)]
    pub(crate) fn new() -> Self {
        Table {
            slots: std::array::from_fn(|_| Slot {
                seq: AtomicUsize::new(0),
                lo: AtomicUsize::new(0),
                hi: AtomicUsize::new(0),
            }),
            used: AtomicUsize::new(0),
            writers: Mutex::new([false; N]),
        }
    }

    fn writers(&self) -> MutexGuard<'_, [bool; N]> {
        self.writers.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Publish `[lo, hi)` in a free slot. `None` when every slot is taken.
    pub(crate) fn claim(&self, lo: usize, hi: usize) -> Option<usize> {
        let mut taken = self.writers();
        let i = taken.iter().position(|t| !t)?;
        taken[i] = true;
        self.slots[i].write(lo, hi);
        self.used.fetch_max(i + 1, Ordering::Release);
        Some(i)
    }

    /// Empty slot `i` and free it. Called once no code in its range runs.
    pub(crate) fn release(&self, i: usize) {
        let mut taken = self.writers();
        self.slots[i].write(0, 0);
        taken[i] = false;
    }

    /// The range slot `i` holds, if a reader can trust it now (not mid-write, not empty).
    /// Loads only: async-signal-safe.
    pub(crate) fn read(&self, i: usize) -> Option<(usize, usize)> {
        let s = &self.slots[i];
        let seq = s.seq.load(Ordering::Acquire);
        if seq & 1 != 0 {
            return None;
        }
        let lo = s.lo.load(Ordering::Relaxed);
        let hi = s.hi.load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        (s.seq.load(Ordering::Relaxed) == seq && lo < hi).then_some((lo, hi))
    }

    /// Whether `pc` lies in a published range. Loads only: async-signal-safe.
    pub(crate) fn contains(&self, pc: usize) -> bool {
        let used = self.used.load(Ordering::Acquire).min(N);
        (0..used).any(|i| self.read(i).is_some_and(|(lo, hi)| lo <= pc && pc < hi))
    }
}

pub(crate) use live::*;

mod live {
    use super::{Table, ARENA, SLOTS};
    use cranelift_jit::{ArenaMemoryProvider, JITBuilder, JITModule};
    use cranelift_module::{DataDescription, Module};

    #[cfg(not(loom))]
    static TABLE: Table<SLOTS> = Table::new();
    // A `--cfg loom` build runs only the loom models, which build tables of their own; this one is
    // never touched there (loom's atomics exist only inside a model).
    #[cfg(loom)]
    static TABLE: std::sync::LazyLock<Table<SLOTS>> = std::sync::LazyLock::new(Table::new);

    /// Whether `pc` is in JIT-emitted code. For the fault handlers (`trap_shim.c`, the Windows VEH):
    /// loads only, no lock, no allocation, no panic.
    pub(crate) fn pc_is_jit(pc: usize) -> bool {
        TABLE.contains(pc)
    }

    /// The C handler's entry to [`pc_is_jit`].
    #[cfg(unix)]
    #[no_mangle]
    extern "C" fn temen_pc_is_jit(pc: usize) -> i32 {
        pc_is_jit(pc) as i32
    }

    /// A module's arena, in the table until this drops. Held beside the module's code
    /// (`OwnedJit`), and dropped before the arena is freed.
    pub(crate) struct CodeSlot(usize);

    impl Drop for CodeSlot {
        fn drop(&mut self) {
            TABLE.release(self.0);
        }
    }

    /// A `JITModule` from `builder` whose memory is one [`ARENA`] reservation, with that reservation
    /// in the table. Fails closed: no arena, or no free slot, is no module.
    ///
    /// The arena's provider does not say where its reservation is, so the module's first
    /// allocation asks it: a one-byte data object, which a bump arena places at offset 0, finalized
    /// so its address can be read. Everything the module allocates after it lies above it, inside
    /// the reservation, so `[base, base + ARENA)` holds all of the module's code and nothing else.
    ///
    /// **"The first allocation is at the reservation's base" is cranelift-jit's layout, not its
    /// API** (`ArenaMemoryProvider` in 0.132 bumps from offset 0, a segment per protection). If it
    /// stops being true — a header, or anything else, placed ahead of the probe — the registered
    /// range shifts up: code below the probe is unregistered, so every guest fault there kills the
    /// process (the suites catch that loudly), and the range's top runs past the reservation into
    /// whatever is mapped next, where a host fault would be unwound as the guest's (nothing catches
    /// that). So: the probe must be page-aligned, which a header ahead of it would almost always
    /// break, and is checked in [`arena_base`]; and a cranelift-jit upgrade must re-run `code_ranges`' tests,
    /// which check the range against the OS's own record of the reservation where one is cheap
    /// (Windows' `VirtualQuery`).
    pub(crate) fn arena_module(mut builder: JITBuilder) -> Result<NewModule, crate::JitError> {
        let arena = ArenaMemoryProvider::new_with_size(ARENA)
            .map_err(|e| crate::JitError::Backend(format!("JIT code arena: {e}")))?;
        builder.memory_provider(Box::new(arena));
        let mut new = NewModule {
            module: Some(JITModule::new(builder)),
            slot: None,
        };
        let base = arena_base(&mut new)?;
        let slot = TABLE
            .claim(base, base + ARENA)
            .ok_or(crate::JitError::Unsupported("too many live JIT modules"))?;
        new.slot = Some(CodeSlot(slot));
        Ok(new)
    }

    /// A module being compiled: its code arena, in the table, until it is handed to its owner
    /// ([`NewModule::into_parts`], then `OwnedJit`, which frees it when its code is done). Dropped
    /// before then — a compile that failed or declined — it gives the whole arena back: a
    /// `JITModule` frees nothing when it drops, and cranelift's arena frees its reservation on drop
    /// only while nothing in it is finalized, which the probe ([`arena_base`]) always is.
    pub(crate) struct NewModule {
        module: Option<JITModule>,
        slot: Option<CodeSlot>,
    }

    impl NewModule {
        /// The module and its slot, for the owner that frees them.
        pub(crate) fn into_parts(mut self) -> (JITModule, CodeSlot) {
            let module = self.module.take().expect("a module until handed over");
            let slot = self.slot.take().expect("a slot until handed over");
            (module, slot)
        }
    }

    impl std::ops::Deref for NewModule {
        type Target = JITModule;
        fn deref(&self) -> &JITModule {
            self.module.as_ref().expect("a module until handed over")
        }
    }

    impl std::ops::DerefMut for NewModule {
        fn deref_mut(&mut self) -> &mut JITModule {
            self.module.as_mut().expect("a module until handed over")
        }
    }

    impl Drop for NewModule {
        fn drop(&mut self) {
            // Out of the table first, then the pages back (`Arena::drop`'s order).
            drop(self.slot.take());
            if let Some(m) = self.module.take() {
                // SAFETY: a module is handed over before any of its code is published to run, so
                // nothing from one dropped here ever ran or will.
                unsafe { m.free_memory() };
            }
        }
    }

    /// The base of `module`'s arena reservation, from its first allocation (see [`arena_module`]).
    fn arena_base(module: &mut JITModule) -> Result<usize, crate::JitError> {
        let backend = |e: cranelift_module::ModuleError| crate::JitError::Backend(e.to_string());
        let probe = module
            .declare_anonymous_data(false, false)
            .map_err(backend)?;
        let mut d = DataDescription::new();
        d.define_zeroinit(1);
        module.define_data(probe, &d).map_err(backend)?;
        module.finalize_definitions().map_err(backend)?;
        let base = module.get_finalized_data(probe).0 as usize;
        if !base.is_multiple_of(crate::mem::page_size()) {
            // Not the reservation's base: decline rather than register a range that is not the
            // arena's (see above).
            return Err(crate::JitError::Unsupported(
                "the JIT code arena's base could not be found",
            ));
        }
        Ok(base)
    }

    /// A test's stand-in for emitted code: the page at `f`, in the table until the slot drops.
    #[cfg(test)]
    pub(crate) fn test_code(f: usize) -> CodeSlot {
        CodeSlot(TABLE.claim(f, f + 4096).expect("a free slot"))
    }

    #[cfg(test)]
    mod tests {
        use super::super::Table;
        use super::*;

        #[test]
        fn a_slot_covers_its_range_and_nothing_once_released() {
            let t: Table<4> = Table::new();
            assert!(!t.contains(0x1000));
            let i = t.claim(0x1000, 0x2000).unwrap();
            assert!(t.contains(0x1000) && t.contains(0x1fff));
            assert!(!t.contains(0xfff) && !t.contains(0x2000));
            t.release(i);
            assert!(!t.contains(0x1000));
            assert_eq!(
                t.claim(0x5000, 0x6000),
                Some(i),
                "a released slot is reused"
            );
        }

        #[test]
        fn a_full_table_refuses_a_claim() {
            let t: Table<2> = Table::new();
            assert!(t.claim(1, 2).is_some() && t.claim(3, 4).is_some());
            assert_eq!(t.claim(5, 6), None);
        }

        /// The arena probe finds the reservation: a module's code lies in its registered range,
        /// which is released with the slot.
        #[test]
        fn a_modules_code_is_in_its_registered_range() {
            let new = crate::new_jit_module(crate::Imports::NONE, &crate::FastClaims::new())
                .expect("module");
            let (m, slot) = new.into_parts();
            let (lo, hi) = TABLE.read(slot.0).expect("published");
            assert_eq!(hi - lo, ARENA);
            let here = crate::fuel::refill_trampoline();
            assert!(!pc_is_jit(
                a_modules_code_is_in_its_registered_range as *const () as usize
            ));
            assert!(pc_is_jit(here), "the fuel trampoline is JIT code");
            drop(slot);
            assert!(!pc_is_jit(lo));
            // SAFETY: nothing from `m` ever ran.
            unsafe { m.free_memory() };
        }

        /// A module dropped before it is handed over (a compile that failed or declined) gives its
        /// slot back, and its arena with it (`NewModule::drop`; the arena's own drop would not free
        /// it, since the probe is finalized). The unmap itself is not asserted: tests run in
        /// parallel, and another test's mapping can take the freed range at once.
        #[test]
        fn a_module_that_never_ran_gives_its_arena_back() {
            let new = crate::new_jit_module(crate::Imports::NONE, &crate::FastClaims::new())
                .expect("module");
            let base = new
                .slot
                .as_ref()
                .and_then(|s| TABLE.read(s.0))
                .expect("published")
                .0;
            assert!(pc_is_jit(base));
            drop(new);
            assert!(!pc_is_jit(base), "the slot is released");
        }

        /// The probe's layout assumption, checked against the OS's own record of the reservation:
        /// the registered range is exactly one whole allocation, starting at its base. Windows only:
        /// `VirtualQuery` names a reservation's `AllocationBase`. Linux and macOS keep no such record
        /// (`/proc/self/maps` merges an adjacent mapping with the same protection into one entry),
        /// so there the page-alignment check in `arena_module` and the suites' guest faults stand
        /// in for it. Re-run on every cranelift-jit upgrade: the layout is not an API.
        #[cfg(windows)]
        #[test]
        fn the_registered_range_is_the_arenas_reservation() {
            use windows_sys::Win32::System::Memory::{VirtualQuery, MEMORY_BASIC_INFORMATION};
            let (m, slot) = crate::new_jit_module(crate::Imports::NONE, &crate::FastClaims::new())
                .expect("module")
                .into_parts();
            let (lo, hi) = TABLE.read(slot.0).expect("published");
            let query = |at: usize| {
                // SAFETY: `VirtualQuery` only reads the process's address-space bookkeeping.
                let mut info: MEMORY_BASIC_INFORMATION = unsafe { core::mem::zeroed() };
                let n = unsafe {
                    VirtualQuery(
                        at as *const core::ffi::c_void,
                        &mut info,
                        core::mem::size_of::<MEMORY_BASIC_INFORMATION>(),
                    )
                };
                assert_ne!(n, 0, "VirtualQuery({at:#x})");
                info
            };
            assert_eq!(
                query(lo).AllocationBase as usize,
                lo,
                "lo is the reservation's base"
            );
            // Walk the reservation's regions: they end exactly at `hi`.
            let mut at = lo;
            while at < hi {
                let info = query(at);
                assert_eq!(
                    info.AllocationBase as usize, lo,
                    "{at:#x} is in the reservation"
                );
                at = info.BaseAddress as usize + info.RegionSize;
            }
            assert_eq!(at, hi, "the reservation ends at the range's top");
            assert_ne!(
                query(hi).AllocationBase as usize,
                lo,
                "nothing past hi is the arena's"
            );
            drop(slot);
            // SAFETY: nothing from `m` ever ran.
            unsafe { m.free_memory() };
        }
    }
}

// The slot protocol, model-checked (#2996): a reader racing a slot's release and reuse never sees
// one module's `lo` with another's `hi`, nor a range that was never published; and a reader that
// acquired code published after its range finds the range. Run with
// `RUSTFLAGS="--cfg loom" cargo test -p temen-jit --lib code_ranges::loom`.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use loom::sync::Arc;

    const X: (usize, usize) = (0x1000, 0x2000);
    const Y: (usize, usize) = (0x9000, 0xa000);

    #[test]
    fn loom_reader_never_mixes_two_modules() {
        loom::model(|| {
            let t: Arc<Table<1>> = Arc::new(Table::new());
            let i = t.claim(X.0, X.1).unwrap();
            let w = {
                let t = t.clone();
                loom::thread::spawn(move || {
                    t.release(i);
                    t.claim(Y.0, Y.1).unwrap();
                })
            };
            if let Some(r) = t.read(0) {
                assert!(r == X || r == Y, "a torn or unpublished range: {r:x?}");
            }
            w.join().unwrap();
        });
    }

    #[test]
    fn loom_code_published_after_its_range_is_covered() {
        loom::model(|| {
            let t: Arc<Table<1>> = Arc::new(Table::new());
            // The `fn` pointer a module's code reaches a vCPU through.
            let published = Arc::new(AtomicUsize::new(0));
            let w = {
                let (t, published) = (t.clone(), published.clone());
                loom::thread::spawn(move || {
                    t.claim(X.0, X.1).unwrap();
                    published.store(X.0 + 0x80, Ordering::Release);
                })
            };
            let f = published.load(Ordering::Acquire);
            if f != 0 {
                assert!(
                    t.contains(f),
                    "code reached through the publication is covered"
                );
            }
            w.join().unwrap();
        });
    }
}
