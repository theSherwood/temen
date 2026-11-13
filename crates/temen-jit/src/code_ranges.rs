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

/// How many JIT modules may be live at once. Each holds an [`ARENA`] reservation, so this is a
/// terabyte of code VA: the OS's own limits (Linux's map count, Windows's commit charge for each
/// arena) bind long before it.
#[cfg(not(loom))]
const SLOTS: usize = 4096;

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

#[cfg(not(loom))]
pub(crate) use live::*;

#[cfg(not(loom))]
mod live {
    use super::{Table, ARENA, SLOTS};
    use cranelift_jit::{ArenaMemoryProvider, JITBuilder, JITModule};
    use cranelift_module::{DataDescription, Module};

    static TABLE: Table<SLOTS> = Table::new();

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
    pub(crate) fn arena_module(
        mut builder: JITBuilder,
    ) -> Result<(JITModule, CodeSlot), crate::JitError> {
        let backend = |e: cranelift_module::ModuleError| crate::JitError::Backend(e.to_string());
        let arena = ArenaMemoryProvider::new_with_size(ARENA)
            .map_err(|e| crate::JitError::Backend(format!("JIT code arena: {e}")))?;
        builder.memory_provider(Box::new(arena));
        let mut module = JITModule::new(builder);
        let probe = module
            .declare_anonymous_data(false, false)
            .map_err(backend)?;
        let mut d = DataDescription::new();
        d.define_zeroinit(1);
        module.define_data(probe, &d).map_err(backend)?;
        module.finalize_definitions().map_err(backend)?;
        let base = module.get_finalized_data(probe).0 as usize;
        let slot = TABLE
            .claim(base, base + ARENA)
            .ok_or(crate::JitError::Unsupported("too many live JIT modules"))?;
        Ok((module, CodeSlot(slot)))
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
            let (m, slot) = crate::new_jit_module(crate::Imports::NONE, &crate::FastClaims::new())
                .expect("module");
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
