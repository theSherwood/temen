//! The shared guest-memory substrate (`DESIGN.md` §12/§13).
//!
//! A [`Region`] is the backing store for a guest window's **anonymous** pages: a flat,
//! demand-zeroed, raw-addressable byte range of the window's *reserved* size. Confinement/masking
//! (§4) is the caller's job ([`temen_mask::Window`]); a `Region` only sees an already-confined offset
//! and bounds-checks it against its own length as defense-in-depth.
//!
//! Why a separate crate: the parallel interpreter (and later the JIT) must run **multiple OS-thread
//! vCPUs over one shared region with real hardware atomics** (§12). A shared raw region + real
//! width-4/8 atomics inherently need `unsafe`, but the interpreter is the `#![forbid(unsafe_code)]`
//! reference oracle. So all the `unsafe` lives *here*, behind a safe API, and is audited/fuzzed in
//! isolation — exactly the role [`temen_mask`] plays for masking.
//!
//! ## Sharing across vCPUs
//!
//! A `Region` is [`Send`] + [`Sync`]: several vCPU threads hold `&Region` and run over the *one*
//! guest memory image — that shared image is what makes them threads of one guest rather than
//! isolated programs. Every accessor therefore takes `&self`. `Region` itself adds **no** locking or
//! ordering policy beyond what each op needs to be language-level sound; the concurrency *semantics*
//! (the memory model, scheduling, `wait`/`notify`) live above it. What each op guarantees:
//! - **atomic ops** (`atomic_*`) — real seq-cst hardware atomics; the sound primitive for concurrent
//!   access to a *shared* location.
//! - **single-byte plain ops** (`byte`/`set_byte`) — relaxed atomics, so even a same-byte race is
//!   *defined* (no UB), just unordered (the guest's responsibility, per the §12 C11-style model).
//! - **bulk ops** (`zero`/`read_into`) — control-plane (`map`/`unmap`/snapshot); they assume no
//!   concurrent access to *their own range*, which holds for steady-state guest execution.
//!
//! Beyond that, a guest data race corrupts only the guest's own confined memory and can never escape
//! the window (§12) — masking + bounds still gate every access.
//!
//! Two backings:
//! - **`Mapped`** (unix): one anonymous `mmap` of the reserved size (lazy: pages cost nothing until
//!   touched, then the kernel zero-fills). Page-aligned, so **real** `AtomicU32`/`AtomicU64` ops
//!   (the §12 hardware atomics the JIT already emits) are sound on it. The substrate parallel
//!   execution runs on.
//! - **`Paged`** (non-unix, or a reservation too large to `mmap`): a `BTreeMap` of zeroed pages
//!   behind a `Mutex`. Correct but serialized — the portable fallback, not the parallel substrate.
//!
//! Plus two flat wrappers around the same raw accessor bodies: **`Shared`** (borrowed,
//! embedder-owned memory — the browser's window-in-linear-memory shape) and **`Owned`** (an
//! eagerly-allocated heap buffer — the portable flat backing a fork twin's private window needs
//! where `Paged` would apply, #816).
//!
//! And one **proxied** backing: **`Foreign`** (#1284, `DETACHED_JIT.md` §3.3) — a region whose bytes
//! live in memory this process cannot address (a detached child's own `WebAssembly.Memory` on
//! wasm32, where the engine's pointers are offsets into *its* one linear memory). Every accessor
//! calls through a caller-supplied [`ForeignOps`] table; the browser cdylib fills it with JS host
//! imports, a native test fills it with a mock. Not flat-addressable (`raw_base` is `None`), so a
//! consumer that needs a `base + off` pointer view declines it, as for `Paged`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

/// The six read-modify-write operations (§12), mirrored from `temen_ir::AtomicRmwOp` without taking a
/// dependency on the IR crate (this crate sits below it). Each returns the **old** value.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RmwOp {
    Add,
    Sub,
    And,
    Or,
    Xor,
    Xchg,
}

impl RmwOp {
    /// The wire code a [`ForeignOps::atomic`] call carries: `2 + discriminant` (0/1 are load/store,
    /// 8 is compare-exchange — see [`ForeignOps`]).
    pub fn code(self) -> u32 {
        2 + self as u32
    }
}

/// A guest window's anonymous-page backing store. See the crate docs for the two variants and the
/// sharing contract. All accessors take `&self`: a `Region` is shared by reference across vCPUs.
pub enum Region {
    /// Unix: one demand-zeroed anonymous `mmap` of `[0, len)` (page-rounded). Real atomics.
    #[cfg(unix)]
    Mapped(Mapped),
    /// A borrowed, **externally-owned** region (e.g. a span of wasm shared linear memory): the same
    /// raw-pointer hardware atomics as `Mapped`, but it does not own its backing (no `mmap`, no
    /// `Drop`). The parallel-wasm path — every instance's `Region::Shared` over the same shared-memory
    /// address is one shared cell. Built via the `unsafe` [`Region::shared`].
    Shared(Shared),
    /// An **owned, eagerly-allocated** flat heap buffer: the same raw-pointer hardware atomics as
    /// `Shared`, with the backing's lifecycle owned here (like `Mapped`, but plain `alloc`, so it
    /// exists on every target). Flat-addressable where [`Region::new`]'s non-unix fallback is
    /// `Paged` — the #816 twin-backing seam: a fork twin's private window must be a single
    /// contiguous span for emitted `win + addr` code to serve it. Built via
    /// [`Region::owned_zeroed`]; eager allocation is the cost, so callers bound the size.
    Owned(Owned),
    /// Portable fallback: zeroed pages in a `Mutex`-guarded map (serialized, not the parallel path).
    Paged(Paged),
    /// **Proxied**: bytes in memory this process cannot address, reached through a [`ForeignOps`]
    /// table (a detached child's own `WebAssembly.Memory`, #1284). Every accessor is one call into the
    /// table — bulk where the API is bulk (`read_into`/`write_from`/`fill`/`copy_within`), one call
    /// per word or byte otherwise. Not flat-addressable. Built via [`Region::foreign`].
    Foreign(Foreign),
}

/// The accessor table behind [`Region::Foreign`]: how to reach the bytes of foreign backing `id`.
/// Offsets are region-relative and already bounded to `[0, len)` by `Region` before the call, so an
/// implementation may trust them. `atomic` is the one seq-cst primitive: `kind` 0 = load, 1 = store
/// (`a` = value), 2..=7 = read-modify-write ([`RmwOp::code`], `a` = operand), 8 = compare-exchange
/// (`a` = expected, `b` = replacement); it returns the **old** value (`width` 4 or 8, `off` aligned).
pub struct ForeignOps {
    pub read: fn(id: u32, off: u64, out: &mut [u8]),
    pub write: fn(id: u32, off: u64, data: &[u8]),
    pub fill: fn(id: u32, off: u64, len: u64, b: u8),
    pub copy_within: fn(id: u32, dst: u64, src: u64, len: u64),
    pub atomic: fn(id: u32, kind: u32, off: u64, width: u32, a: u64, b: u64) -> u64,
}

/// The [`ForeignOps::atomic`] kind for a plain load / store / compare-exchange.
pub const FOREIGN_ATOMIC_LOAD: u32 = 0;
pub const FOREIGN_ATOMIC_STORE: u32 = 1;
pub const FOREIGN_ATOMIC_CMPXCHG: u32 = 8;

/// A proxied backing (see [`Region::Foreign`]): the foreign memory's id in the embedder's registry,
/// its current addressable length (the foreign memory can grow; the owner bumps it with
/// [`Region::set_foreign_len`]), and the accessor table.
pub struct Foreign {
    id: u32,
    len: AtomicU64,
    ops: &'static ForeignOps,
}

impl Foreign {
    fn word(&self, off: u64, width: u32) -> u64 {
        let mut b = [0u8; 8];
        (self.ops.read)(self.id, off, &mut b[..width as usize]);
        u64::from_le_bytes(b)
    }
    fn set_word(&self, off: u64, width: u32, val: u64) {
        let b = val.to_le_bytes();
        (self.ops.write)(self.id, off, &b[..width as usize]);
    }
}

/// Which accessor body serves a [`Region`]: the one raw-pointer body (`Shared`, which `Mapped` and
/// `Owned` wrap), the safe `Paged` reference, or the proxied `Foreign` table.
enum Backing<'a> {
    Raw(&'a Shared),
    Paged(&'a Paged),
    Foreign(&'a Foreign),
}

impl Region {
    /// A region addressing `[0, size)` bytes, all reading as zero until written. `page` is the
    /// host page granularity (the unit [`Region::zero`] re-zeroes and the `Paged` chunk size).
    ///
    /// On unix a feasible `size` is `mmap`-backed (the shared substrate); a `size` too large to map
    /// — or any non-unix target — falls back to the paged backing.
    pub fn new(size: u64, page: u64) -> Region {
        #[cfg(unix)]
        {
            if size > 0 {
                if let Some(m) = Mapped::new(size, page) {
                    return Region::Mapped(m);
                }
            }
        }
        Region::Paged(Paged::new(size, page))
    }

    /// Build a region over **caller-owned** memory `[base, base+size)` — real hardware atomics like
    /// `Mapped`, but **non-owning** (the embedder owns the backing). For the parallel-wasm backend,
    /// `base` is an address in the shared linear memory, so each instance's region over the same
    /// address shares one cell. Single-threaded today (the cooperative path); genuinely parallel once
    /// the threads build wires `thread.spawn` to a Worker over this shared backing.
    ///
    /// # Safety
    /// `base` must point to ≥ `size` valid bytes that stay live and are exclusively managed through
    /// this `Region` (and its clones across threads) for its whole lifetime, and be 8-aligned so a
    /// naturally-aligned 4/8-byte atomic at any in-bounds offset is a valid `AtomicU32`/`U64`.
    pub unsafe fn shared(base: *mut u8, size: u64) -> Region {
        Region::Shared(Shared::new(base, size))
    }

    /// The portable `Paged` fallback, **forced** — what [`Region::new`] returns on non-unix
    /// targets (or an un-`mmap`-able reservation), constructible directly so a unix host can
    /// exercise the non-flat arm of a backing decision (e.g. the #816 fork-twin seam's tests).
    pub fn paged(size: u64, page: u64) -> Region {
        Region::Paged(Paged::new(size, page))
    }

    /// A region addressing `[0, size)` over an **owned, zero-initialized flat buffer** — flat-
    /// addressable ([`Region::raw_base`]) on every target, where [`Region::new`] falls back to the
    /// non-flat `Paged` on non-unix. The buffer is allocated **eagerly** (a `new` reservation is
    /// lazy), so callers bound `size` — the #816 fork-twin seam bounds it by the parent backing's
    /// length (the run window size). `None` when `size` is 0 or the allocation fails, so callers
    /// fall back (`Region::new`) rather than abort.
    pub fn owned_zeroed(size: u64, page: u64) -> Option<Region> {
        Owned::new(size, page).map(Region::Owned)
    }

    /// A proxied region over foreign backing `id` with `len` addressable bytes, reached through `ops`
    /// (#1284). `len` may later grow ([`Region::set_foreign_len`]); it never shrinks.
    pub fn foreign(id: u32, len: u64, ops: &'static ForeignOps) -> Region {
        Region::Foreign(Foreign {
            id,
            len: AtomicU64::new(len),
            ops,
        })
    }

    /// Record that a `Foreign` backing grew to `len` bytes (the embedder grew the foreign memory).
    /// A no-op on every other variant, and never shrinks.
    pub fn set_foreign_len(&self, len: u64) {
        if let Region::Foreign(f) = self {
            f.len.fetch_max(len, Ordering::AcqRel);
        }
    }

    /// The raw-backed variants (`Mapped`, `Shared`, `Owned`) all dispatch to the one accessor body in
    /// [`Shared`] — `Mapped` *is* a `Shared` plus an owned `mmap` — so a single `Raw` arm serves
    /// them; `Paged` (the safe reference) and `Foreign` (the proxied table) are the other two. No
    /// `Mapped`/`Shared` duplication to keep in step.
    #[inline]
    fn backing(&self) -> Backing<'_> {
        match self {
            #[cfg(unix)]
            Region::Mapped(m) => Backing::Raw(&m.raw),
            Region::Shared(s) => Backing::Raw(s),
            Region::Owned(o) => Backing::Raw(&o.raw),
            Region::Paged(p) => Backing::Paged(p),
            Region::Foreign(f) => Backing::Foreign(f),
        }
    }

    /// The flat raw body, if this region has one (`Paged`/`Foreign` do not).
    #[inline]
    fn flat(&self) -> Option<&Shared> {
        match self.backing() {
            Backing::Raw(s) => Some(s),
            _ => None,
        }
    }

    /// The addressable length `[0, size)`.
    pub fn len(&self) -> u64 {
        match self.backing() {
            Backing::Raw(s) => s.size,
            Backing::Paged(p) => p.size,
            Backing::Foreign(f) => f.len.load(Ordering::Acquire),
        }
    }

    /// #816: the raw base address of a **flat** (raw-backed `Shared`/`Mapped`) region — the pointer
    /// `[0, len)` maps to contiguously, for an embedder that must hand emitted code a
    /// `base + offset` view of a window (the browser tier-up driver's per-event `win`). `None` for
    /// the `Paged` fallback, which has no single flat address — callers treat that as
    /// "not flat-addressable" and fail closed (the window then runs on the interpreter only).
    /// Reading or writing through the returned pointer is subject to the same safety contract as
    /// the region's construction; bounds are the caller's to keep.
    pub fn raw_base(&self) -> Option<*mut u8> {
        self.flat().map(|s| s.base_ptr())
    }

    /// #816: the raw address of flat offset `off` — [`raw_base`](Region::raw_base)` + off`,
    /// bounds-checked to the region (`off <= len`; the offset itself, not an access through it).
    /// `None` for `Paged` or an out-of-region offset. Lives here (not in the `unsafe`-free callers)
    /// because the in-allocation pointer add is `unsafe`; the bound makes it sound.
    pub fn raw_base_at(&self, off: u64) -> Option<*mut u8> {
        let s = self.flat()?;
        if off > s.size {
            return None;
        }
        // SAFETY: `off <= size`, so the result stays within (or one past) the allocation the
        // region's construction contract covers.
        Some(unsafe { s.base_ptr().add(off as usize) })
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Read one byte; an untouched address reads zero. Out of range reads zero (the caller already
    /// confined into `[0, len)`; this is belt-and-suspenders).
    pub fn byte(&self, off: u64) -> u8 {
        if off >= self.len() {
            return 0;
        }
        match self.backing() {
            Backing::Raw(s) => s.byte(off),
            Backing::Paged(p) => p.byte(off),
            Backing::Foreign(f) => f.word(off, 1) as u8,
        }
    }

    /// Write one byte. Out-of-range writes are dropped (the caller confines first).
    pub fn set_byte(&self, off: u64, b: u8) {
        if off >= self.len() {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.set_byte(off, b),
            Backing::Paged(p) => p.set_byte(off, b),
            Backing::Foreign(f) => f.set_word(off, 1, b as u64),
        }
    }

    /// Reset `[off, off+len)` to zero (the `map`/`unmap` "fresh page" semantics). Range is clamped
    /// to `[0, size)`. Control-plane: assumes no concurrent access to the range.
    pub fn zero(&self, off: u64, len: u64) {
        let len = clamp_len(off, len, self.len());
        if len == 0 {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.zero(off, len),
            Backing::Paged(p) => p.zero(off, len),
            Backing::Foreign(f) => (f.ops.fill)(f.id, off, len, 0),
        }
    }

    /// Set `[off, off+len)` to byte `b` — the bulk `memory.fill` primitive (a generalized
    /// [`Region::zero`]). Range clamped to `[0, size)`. Bulk/non-atomic: sound for the single-threaded
    /// cooperative caller (the bytecode interpreter's `memory.fill` fast path), the same contract as
    /// [`Region::read_word`]; the concurrent tree-walker keeps the per-byte [`Region::set_byte`] path.
    pub fn fill(&self, off: u64, len: u64, b: u8) {
        let len = clamp_len(off, len, self.len());
        if len == 0 {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.fill(off, len, b),
            Backing::Paged(p) => p.fill(off, len, b),
            Backing::Foreign(f) => (f.ops.fill)(f.id, off, len, b),
        }
    }

    /// Copy `len` bytes `src`→`dst` within the region — the bulk `memory.copy`/`memory.move` primitive.
    /// Overlap-safe (a `memmove`, so it serves both the non-overlapping and overlapping cases). Both
    /// spans clamped to their in-range prefix (defense-in-depth; the caller confined both into
    /// `[0, size)`). Bulk/non-atomic: same single-threaded contract as [`Region::fill`].
    pub fn copy_within(&self, dst: u64, src: u64, len: u64) {
        let size = self.len();
        let len = clamp_len(dst, len, size).min(clamp_len(src, len, size));
        if len == 0 {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.copy_within(dst, src, len),
            Backing::Paged(p) => p.copy_within(dst, src, len),
            Backing::Foreign(f) => (f.ops.copy_within)(f.id, dst, src, len),
        }
    }

    /// Copy `[off, off+out.len())` into `out` (zero past the touched extent / region end). Used for
    /// the escape-oracle window snapshot, which can span the whole mapped extent — so the mmap
    /// backing bulk-copies rather than dispatching per byte.
    pub fn read_into(&self, off: u64, out: &mut [u8]) {
        match self.backing() {
            Backing::Raw(s) => s.read_into(off, out),
            Backing::Paged(p) => p.read_into(off, out),
            Backing::Foreign(f) => {
                // Zero-fill past the end (the flat body's contract), read the in-range prefix.
                let n = clamp_len(off, out.len() as u64, self.len()) as usize;
                out[n..].fill(0);
                if n != 0 {
                    (f.ops.read)(f.id, off, &mut out[..n]);
                }
            }
        }
    }

    /// Copy `data` into `[off, off+data.len())` — the bulk slice-store, mirror of [`Region::read_into`].
    /// The mmap/shared backing does one `memcpy`; the `Paged` fallback locks its map once and writes
    /// page-aligned chunks (vs the per-byte [`Region::set_byte`]). Bytes past the region end are
    /// dropped (the caller confined `[off, off+len) ⊆ [0, size)`). Bulk/non-atomic: same
    /// single-threaded contract as [`Region::fill`].
    pub fn write_from(&self, off: u64, data: &[u8]) {
        match self.backing() {
            Backing::Raw(s) => s.write_from(off, data),
            Backing::Paged(p) => p.write_from(off, data),
            Backing::Foreign(f) => {
                let n = clamp_len(off, data.len() as u64, self.len()) as usize;
                if n != 0 {
                    (f.ops.write)(f.id, off, &data[..n]);
                }
            }
        }
    }

    /// Whether `[off, off+width)` lies inside the region — the one bound every word/atomic accessor
    /// enforces (#1191). The interpreter confines an access to the window's *reservation* and its
    /// page map, not to this backing, and a caller-provided backing (`Region::shared` over a browser
    /// window slice) can be narrower than the reservation: an admitted page past the backing's end
    /// must read as zero / drop the write here, never touch the host memory behind the buffer.
    #[inline]
    fn in_range(&self, off: u64, width: u32) -> bool {
        off.checked_add(width as u64)
            .is_some_and(|end| end <= self.len())
    }

    /// **Non-atomic** width-specialized (1/2/4/8) little-endian read — one (possibly unaligned)
    /// machine load instead of `width` per-byte atomic loads. Sound **only for a single-threaded
    /// caller**: the cooperative bytecode interpreter has exactly one vCPU touching the backing at a
    /// time (no race), so it can take this path; the genuinely concurrent tree-walker / §12 atomics
    /// keep the per-byte [`Region::byte`] / [`Region::atomic_load`] paths. Out-of-range reads
    /// return 0 (like [`Region::byte`]); the caller confines to the window, this bounds to the backing.
    #[inline]
    pub fn read_word(&self, off: u64, width: u32) -> u64 {
        if !self.in_range(off, width) {
            return 0;
        }
        match self.backing() {
            Backing::Raw(s) => s.read_word(off, width),
            Backing::Paged(p) => p.read_word(off, width),
            Backing::Foreign(f) => f.word(off, width),
        }
    }

    /// **Non-atomic** width-specialized little-endian write — the store counterpart of
    /// [`Region::read_word`] (same single-threaded contract). Keeps only the low `width` bytes.
    #[inline]
    pub fn write_word(&self, off: u64, width: u32, val: u64) {
        if !self.in_range(off, width) {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.write_word(off, width, val),
            Backing::Paged(p) => p.write_word(off, width, val),
            Backing::Foreign(f) => f.set_word(off, width, val),
        }
    }

    /// `width`-byte (4 or 8) sequentially-consistent atomic load (§12). The caller guarantees
    /// natural alignment and in-window bounds.
    pub fn atomic_load(&self, off: u64, width: u32) -> u64 {
        if !self.in_range(off, width) {
            return 0;
        }
        match self.backing() {
            Backing::Raw(s) => s.atomic_load(off, width),
            Backing::Paged(p) => p.atomic_load(off, width),
            Backing::Foreign(f) => (f.ops.atomic)(f.id, FOREIGN_ATOMIC_LOAD, off, width, 0, 0),
        }
    }

    /// `width`-byte seq-cst atomic store.
    pub fn atomic_store(&self, off: u64, width: u32, val: u64) {
        if !self.in_range(off, width) {
            return;
        }
        match self.backing() {
            Backing::Raw(s) => s.atomic_store(off, width, val),
            Backing::Paged(p) => p.atomic_store(off, width, val),
            Backing::Foreign(f) => {
                (f.ops.atomic)(f.id, FOREIGN_ATOMIC_STORE, off, width, val, 0);
            }
        }
    }

    /// `width`-byte seq-cst read-modify-write; returns the **old** value.
    pub fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        if !self.in_range(off, width) {
            return 0;
        }
        match self.backing() {
            Backing::Raw(s) => s.atomic_rmw(off, width, op, val),
            Backing::Paged(p) => p.atomic_rmw(off, width, op, val),
            Backing::Foreign(f) => (f.ops.atomic)(f.id, op.code(), off, width, val, 0),
        }
    }

    /// `width`-byte seq-cst compare-exchange: store `replacement` iff the current value equals
    /// `expected`; always return the **old** value.
    pub fn atomic_cmpxchg(&self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        if !self.in_range(off, width) {
            return 0;
        }
        match self.backing() {
            Backing::Raw(s) => s.atomic_cmpxchg(off, width, expected, replacement),
            Backing::Paged(p) => p.atomic_cmpxchg(off, width, expected, replacement),
            Backing::Foreign(f) => (f.ops.atomic)(
                f.id,
                FOREIGN_ATOMIC_CMPXCHG,
                off,
                width,
                expected,
                replacement,
            ),
        }
    }
}

/// Apply an [`RmwOp`] to `(old, v)` truncated to `width` bytes — the value math the `Paged` backing
/// uses (the `Mapped` path uses the hardware `fetch_*` instead).
fn rmw_apply(op: RmwOp, old: u64, v: u64, width: u32) -> u64 {
    let m = width_mask(width);
    let r = match op {
        RmwOp::Add => old.wrapping_add(v),
        RmwOp::Sub => old.wrapping_sub(v),
        RmwOp::And => old & v,
        RmwOp::Or => old | v,
        RmwOp::Xor => old ^ v,
        RmwOp::Xchg => v,
    };
    r & m
}

fn width_mask(width: u32) -> u64 {
    if width >= 8 {
        u64::MAX
    } else {
        (1u64 << (width * 8)) - 1
    }
}

/// Bytes of `[off, off+len)` that lie within `[0, size)`.
fn clamp_len(off: u64, len: u64, size: u64) -> u64 {
    if off >= size {
        0
    } else {
        len.min(size - off)
    }
}

/// **Differential check of one backing against another** (the §18 interp-as-oracle discipline applied
/// to the memory substrate): `ops` seeded-random operations — mixed atomic / non-atomic, 4- and 8-byte
/// widths, cross-page offsets, out-of-range bytes (which both must confine inertly), bulk zero/fill/
/// overlapping copy — applied to both regions, every result compared, then the final images compared
/// byte-for-byte. `Err` names the first divergence. Deterministic in `seed`, so a failure replays.
/// Used by the unit tests (`Shared` vs `Paged`, mock `Foreign` vs `Paged`) and by the browser cdylib's
/// real-Chromium self-test of `Foreign` over a JS-owned `WebAssembly.Memory` (#1284).
pub fn differential(
    a: &Region,
    b: &Region,
    size: u64,
    page: u64,
    ops: usize,
    seed: u64,
) -> Result<(), String> {
    macro_rules! check {
        ($x:expr, $y:expr, $($msg:tt)+) => {{
            let (x, y) = ($x, $y);
            if x != y {
                return Err(format!("{}: {:?} vs {:?}", format!($($msg)+), x, y));
            }
        }};
    }
    fn xs(s: &mut u64) -> u64 {
        let mut x = *s;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *s = x;
        x
    }
    let mut s = seed;
    // Atomic location: a `width`-aligned (4 or 8) in-bounds offset (the caller's contract).
    let aligned = |s: &mut u64| -> (u64, u32) {
        let width: u32 = if xs(s) & 1 == 0 { 4 } else { 8 };
        let slots = size / width as u64;
        let off = (xs(s) % slots) * width as u64;
        (off, width)
    };
    for _ in 0..ops {
        match xs(&mut s) % 9 {
            0 => {
                // byte read, sometimes out of range (must read 0 / confine on both).
                let off = xs(&mut s) % (size + 64);
                check!(a.byte(off), b.byte(off), "byte({off}) diverged");
            }
            1 => {
                // byte write, sometimes out of range (must drop on both).
                let off = xs(&mut s) % (size + 64);
                let v = xs(&mut s) as u8;
                a.set_byte(off, v);
                b.set_byte(off, v);
            }
            2 => {
                let (off, w) = aligned(&mut s);
                check!(
                    a.atomic_load(off, w),
                    b.atomic_load(off, w),
                    "atomic_load({off},{w}) diverged"
                );
            }
            3 => {
                let (off, w) = aligned(&mut s);
                let v = xs(&mut s);
                a.atomic_store(off, w, v);
                b.atomic_store(off, w, v);
            }
            4 => {
                let (off, w) = aligned(&mut s);
                let op = [
                    RmwOp::Add,
                    RmwOp::Sub,
                    RmwOp::And,
                    RmwOp::Or,
                    RmwOp::Xor,
                    RmwOp::Xchg,
                ][(xs(&mut s) % 6) as usize];
                let v = xs(&mut s);
                check!(
                    a.atomic_rmw(off, w, op, v),
                    b.atomic_rmw(off, w, op, v),
                    "atomic_rmw({off},{w},{op:?}) diverged"
                );
            }
            5 => {
                let (off, w) = aligned(&mut s);
                // Bias `expected` toward a hit half the time by reading the current value first
                // (both backings agree on it, so this stays a pure differential).
                let expected = if xs(&mut s) & 1 == 0 {
                    a.atomic_load(off, w)
                } else {
                    xs(&mut s)
                };
                let rep = xs(&mut s);
                check!(
                    a.atomic_cmpxchg(off, w, expected, rep),
                    b.atomic_cmpxchg(off, w, expected, rep),
                    "atomic_cmpxchg({off},{w}) diverged"
                );
            }
            6 => {
                // zero a random (clamped) range.
                let off = xs(&mut s) % size;
                let len = xs(&mut s) % (2 * page);
                a.zero(off, len);
                b.zero(off, len);
            }
            7 => {
                // fill a random (clamped) range with an arbitrary byte.
                let off = xs(&mut s) % size;
                let len = xs(&mut s) % (2 * page);
                let byte = xs(&mut s) as u8;
                a.fill(off, len, byte);
                b.fill(off, len, byte);
            }
            _ => {
                // overlap-safe copy of a random (clamped) range — dst/src freely overlap.
                let src = xs(&mut s) % size;
                let dst = xs(&mut s) % size;
                let len = xs(&mut s) % (2 * page);
                a.copy_within(dst, src, len);
                b.copy_within(dst, src, len);
            }
        }
    }
    let mut ai = vec![0u8; size as usize];
    let mut bi = vec![0u8; size as usize];
    a.read_into(0, &mut ai);
    b.read_into(0, &mut bi);
    check!(ai, bi, "final region images diverge");
    Ok(())
}

// ============================== unix: the mmap-backed shared region ==============================

pub use shared::Shared;

/// A region over **caller-owned** memory — the parallel-wasm backing. Available on every target
/// (unlike `Mapped`, which is unix-only): on wasm it spans the shared linear memory. This holds the
/// tree's raw-pointer hardware-atomic accessor bodies **once**: `Mapped` is a thin `mmap`-owning
/// wrapper around a `Shared` and delegates here (so the two cannot drift — it is not a gated
/// invariant, it is unrepresentable). `differential_shared_vs_paged_fuzz` gates these bodies against
/// the safe `Paged` reference.
mod shared {
    use super::RmwOp;
    use core::sync::atomic::{
        AtomicU32, AtomicU64, AtomicU8,
        Ordering::{Relaxed, SeqCst},
    };

    /// Borrowed backing over `[base, base+size)`. `base` must be 8-aligned (so a naturally-aligned
    /// 4/8-byte access is a valid atomic) and outlive every `Region::Shared` over it.
    pub struct Shared {
        base: *mut u8,
        pub(super) size: u64,
    }

    // SAFETY: as `Mapped` — a raw `*mut u8` whose every access is a real seq-cst atomic (`atomic_*`)
    // or a relaxed single byte (`byte`/`set_byte`), both defined under races; bulk `zero`/`read_into`
    // are control-plane. The embedder guarantees the backing is genuinely shared across the threads
    // that hold this region and outlives it; `Region` bounds every access to `[0, size)` before
    // dispatching (`byte`/`set_byte` per byte, `in_range` for the word/atomic paths, `clamp_len` for
    // the bulk paths).
    unsafe impl Send for Shared {}
    unsafe impl Sync for Shared {}

    impl Shared {
        /// Wrap `[base, base+size)` as the raw backing. The caller guarantees `base` points to ≥ `size`
        /// valid, 8-aligned bytes that **outlive** this `Shared` and are touched *only* through it (and
        /// its clones across threads) — so a naturally-aligned 4/8-byte atomic at any in-bounds offset
        /// is a valid `AtomicU32`/`U64`. Both callers meet it: `Region::shared` (its `unsafe` contract)
        /// over embedder memory, and `Mapped` over an `mmap` reservation it owns for the map's lifetime
        /// (no aliasing beyond `map_len`).
        pub(super) fn new(base: *mut u8, size: u64) -> Shared {
            Shared { base, size }
        }

        /// The raw base pointer — for the owner (`Mapped`) to `munmap` exactly the reservation it
        /// wrapped, and for [`Region::raw_base`] (#816: the browser tier-up driver's flat `win`
        /// view). Not for byte access (that goes through the bounds-checked accessors below).
        pub(super) fn base_ptr(&self) -> *mut u8 {
            self.base
        }

        #[inline]
        fn ptr(&self, off: u64) -> *mut u8 {
            // SAFETY: callers go through `Region`, which bounds `off < size`.
            unsafe { self.base.add(off as usize) }
        }

        pub(super) fn byte(&self, off: u64) -> u8 {
            // SAFETY: `off < size`; `*mut u8` is 1-aligned for `AtomicU8`. Relaxed → defined under races.
            unsafe { AtomicU8::from_ptr(self.ptr(off)).load(Relaxed) }
        }

        pub(super) fn set_byte(&self, off: u64, b: u8) {
            // SAFETY: as `byte`.
            unsafe { AtomicU8::from_ptr(self.ptr(off)).store(b, Relaxed) }
        }

        pub(super) fn read_word(&self, off: u64, width: u32) -> u64 {
            let p = self.ptr(off);
            // SAFETY: caller confined `[off, off+width) ⊆ [0, size)`; `read_unaligned` needs no align.
            // **Non-atomic** — sound only for the single-threaded (cooperative) caller of this path.
            unsafe {
                match width {
                    1 => p.read() as u64,
                    2 => p.cast::<u16>().read_unaligned() as u64,
                    4 => p.cast::<u32>().read_unaligned() as u64,
                    _ => p.cast::<u64>().read_unaligned(),
                }
            }
        }

        pub(super) fn write_word(&self, off: u64, width: u32, val: u64) {
            let p = self.ptr(off);
            // SAFETY: as `read_word`.
            unsafe {
                match width {
                    1 => p.write(val as u8),
                    2 => p.cast::<u16>().write_unaligned(val as u16),
                    4 => p.cast::<u32>().write_unaligned(val as u32),
                    _ => p.cast::<u64>().write_unaligned(val),
                }
            }
        }

        pub(super) fn zero(&self, off: u64, len: u64) {
            // SAFETY: `[off, off+len) ⊆ [0, size)` (clamped by caller). Control-plane, not raced.
            unsafe { core::ptr::write_bytes(self.ptr(off), 0, len as usize) }
        }

        pub(super) fn fill(&self, off: u64, len: u64, b: u8) {
            // SAFETY: as `zero`, with an arbitrary fill byte. Non-atomic bulk — single-threaded caller.
            unsafe { core::ptr::write_bytes(self.ptr(off), b, len as usize) }
        }

        pub(super) fn copy_within(&self, dst: u64, src: u64, len: u64) {
            // SAFETY: both spans `⊆ [0, size)` (clamped by caller); `ptr::copy` is overlap-safe
            // (`memmove`). Non-atomic bulk — sound only for the single-threaded cooperative caller.
            unsafe { core::ptr::copy(self.ptr(src), self.ptr(dst), len as usize) }
        }

        pub(super) fn read_into(&self, off: u64, out: &mut [u8]) {
            let avail = self.size.saturating_sub(off) as usize;
            let n = avail.min(out.len());
            if n == 0 {
                return;
            }
            // SAFETY: `[off, off+n) ⊆ [0, size)`; `out[..n]` is a distinct caller buffer.
            unsafe { core::ptr::copy_nonoverlapping(self.ptr(off), out.as_mut_ptr(), n) }
        }

        pub(super) fn write_from(&self, off: u64, data: &[u8]) {
            let avail = self.size.saturating_sub(off) as usize;
            let n = avail.min(data.len());
            if n == 0 {
                return;
            }
            // SAFETY: `[off, off+n) ⊆ [0, size)`; `data[..n]` is a distinct caller buffer. The bulk
            // store counterpart of `read_into` (same single-threaded, non-atomic contract).
            unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr(off), n) }
        }

        pub(super) fn atomic_load(&self, off: u64, width: u32) -> u64 {
            // SAFETY: caller guarantees `off` is `width`-aligned + in-bounds → a valid atomic location.
            unsafe {
                match width {
                    4 => AtomicU32::from_ptr(self.ptr(off) as *mut u32).load(SeqCst) as u64,
                    _ => AtomicU64::from_ptr(self.ptr(off) as *mut u64).load(SeqCst),
                }
            }
        }

        pub(super) fn atomic_store(&self, off: u64, width: u32, val: u64) {
            // SAFETY: aligned + in-bounds as in `atomic_load`.
            unsafe {
                match width {
                    4 => AtomicU32::from_ptr(self.ptr(off) as *mut u32).store(val as u32, SeqCst),
                    _ => AtomicU64::from_ptr(self.ptr(off) as *mut u64).store(val, SeqCst),
                }
            }
        }

        pub(super) fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
            // SAFETY: aligned + in-bounds as in `atomic_load`.
            unsafe {
                match width {
                    4 => {
                        let a = AtomicU32::from_ptr(self.ptr(off) as *mut u32);
                        let v = val as u32;
                        let old = match op {
                            RmwOp::Add => a.fetch_add(v, SeqCst),
                            RmwOp::Sub => a.fetch_sub(v, SeqCst),
                            RmwOp::And => a.fetch_and(v, SeqCst),
                            RmwOp::Or => a.fetch_or(v, SeqCst),
                            RmwOp::Xor => a.fetch_xor(v, SeqCst),
                            RmwOp::Xchg => a.swap(v, SeqCst),
                        };
                        old as u64
                    }
                    _ => {
                        let a = AtomicU64::from_ptr(self.ptr(off) as *mut u64);
                        match op {
                            RmwOp::Add => a.fetch_add(val, SeqCst),
                            RmwOp::Sub => a.fetch_sub(val, SeqCst),
                            RmwOp::And => a.fetch_and(val, SeqCst),
                            RmwOp::Or => a.fetch_or(val, SeqCst),
                            RmwOp::Xor => a.fetch_xor(val, SeqCst),
                            RmwOp::Xchg => a.swap(val, SeqCst),
                        }
                    }
                }
            }
        }

        pub(super) fn atomic_cmpxchg(
            &self,
            off: u64,
            width: u32,
            expected: u64,
            replacement: u64,
        ) -> u64 {
            // SAFETY: aligned + in-bounds as in `atomic_load`. `compare_exchange` returns the prior
            // value in both arms.
            unsafe {
                match width {
                    4 => {
                        let a = AtomicU32::from_ptr(self.ptr(off) as *mut u32);
                        match a.compare_exchange(
                            expected as u32,
                            replacement as u32,
                            SeqCst,
                            SeqCst,
                        ) {
                            Ok(old) | Err(old) => old as u64,
                        }
                    }
                    _ => {
                        let a = AtomicU64::from_ptr(self.ptr(off) as *mut u64);
                        match a.compare_exchange(expected, replacement, SeqCst, SeqCst) {
                            Ok(old) | Err(old) => old,
                        }
                    }
                }
            }
        }
    }
}

#[cfg(unix)]
pub use mapped::Mapped;

#[cfg(unix)]
mod mapped {
    use super::Shared;

    /// One anonymous `mmap` of `[0, size)` (rounded up to `map_len`), owned here — the unix parallel
    /// backing. A **thin owner**: every raw-pointer accessor body lives once in [`Shared`], which this
    /// wraps; `Mapped` adds only the reservation's lifecycle (`mmap` in `new`, `munmap` in `Drop`).
    /// The base is page-aligned, so any naturally-aligned 4/8-byte access is hardware-atomic-able.
    /// `Send`/`Sync` are **automatic** — a `Shared` (itself declared shareable under the crate's raced-
    /// atomics contract) plus a `usize` — so `Mapped` needs no `unsafe impl` of its own.
    pub struct Mapped {
        pub(super) raw: Shared,
        map_len: usize,
    }

    impl Mapped {
        pub(super) fn new(size: u64, page: u64) -> Option<Mapped> {
            let page = (page as usize).max(1);
            let map_len = round_up(size as usize, page);

            // Under **miri** (which can't execute the `mmap` FFI), back the region with a heap
            // allocation so the raw-pointer atomic accessors below run *unchanged* — miri then checks
            // their provenance and the data-race freedom of the concurrent atomics, complementing
            // ThreadSanitizer. The production path is the anonymous lazy `mmap`.
            #[cfg(miri)]
            let base = {
                // 8-aligned for the widest (`U64`) atomic; `map_len >= page >= 1` so the layout is
                // non-zero. Freed with the same layout in `Drop`.
                let layout = std::alloc::Layout::from_size_align(map_len, 8).ok()?;
                // SAFETY: non-zero layout.
                let p = unsafe { std::alloc::alloc_zeroed(layout) };
                if p.is_null() {
                    return None;
                }
                p
            };
            #[cfg(not(miri))]
            let base = {
                // SAFETY: a fresh anonymous lazy reservation; `MAP_NORESERVE` so a large `size` costs
                // only virtual address space until pages are touched (then kernel-zeroed).
                // Null/MAP_FAILED → caller falls back to the paged backing.
                let base = unsafe {
                    libc::mmap(
                        core::ptr::null_mut(),
                        map_len,
                        libc::PROT_READ | libc::PROT_WRITE,
                        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                        -1,
                        0,
                    )
                };
                if base == libc::MAP_FAILED || base.is_null() {
                    return None;
                }
                base as *mut u8
            };

            // Wrap the owned reservation as the raw backing — the accessor bodies live in `Shared`.
            Some(Mapped {
                raw: Shared::new(base, size),
                map_len,
            })
        }
    }

    // The mapping outlives every use of the inner `Shared`: nothing hands out the base pointer except
    // this `Drop`, which runs once, after all `&self` accesses have ended (structural via ownership).
    impl Drop for Mapped {
        fn drop(&mut self) {
            let base = self.raw.base_ptr();
            #[cfg(miri)]
            // SAFETY: the exact layout `new` allocated under miri (8-aligned, `map_len` bytes).
            unsafe {
                std::alloc::dealloc(
                    base,
                    std::alloc::Layout::from_size_align_unchecked(self.map_len, 8),
                );
            }
            #[cfg(not(miri))]
            // SAFETY: releasing exactly the reservation created in `new`.
            unsafe {
                libc::munmap(base as *mut libc::c_void, self.map_len);
            }
        }
    }

    fn round_up(n: usize, align: usize) -> usize {
        (n + align - 1) & !(align - 1)
    }
}

pub use owned::Owned;

mod owned {
    use super::Shared;

    /// An eagerly-allocated, zero-initialized flat heap buffer of `[0, size)`, owned here — the
    /// portable flat backing ([`Region::owned_zeroed`](super::Region::owned_zeroed)). A **thin
    /// owner** exactly like `Mapped`: every accessor body lives once in [`Shared`]; `Owned` adds
    /// only the buffer's lifecycle (`alloc_zeroed` in `new`, `dealloc` in `Drop`). 8-aligned so a
    /// naturally-aligned 4/8-byte access is hardware-atomic-able. `Send`/`Sync` are automatic —
    /// a `Shared` plus a `usize`.
    pub struct Owned {
        pub(super) raw: Shared,
        alloc_len: usize,
    }

    impl Owned {
        pub(super) fn new(size: u64, page: u64) -> Option<Owned> {
            let page = (page as usize).max(1);
            let alloc_len = usize::try_from(size)
                .ok()
                .filter(|&s| s > 0)?
                .checked_next_multiple_of(page)?;
            // 8-aligned for the widest (`U64`) atomic; `alloc_len > 0` so the layout is non-zero.
            // Freed with the same layout in `Drop`. A failed allocation is `None` — the caller
            // falls back to a lazy/paged backing instead of aborting.
            let layout = std::alloc::Layout::from_size_align(alloc_len, 8).ok()?;
            // SAFETY: non-zero layout.
            let base = unsafe { std::alloc::alloc_zeroed(layout) };
            if base.is_null() {
                return None;
            }
            Some(Owned {
                raw: Shared::new(base, size),
                alloc_len,
            })
        }
    }

    // The buffer outlives every use of the inner `Shared`: nothing hands out the base pointer
    // except through `&self` accessors, and this `Drop` runs once, after those end (structural
    // via ownership).
    impl Drop for Owned {
        fn drop(&mut self) {
            // SAFETY: the exact layout `new` allocated (8-aligned, `alloc_len` bytes).
            unsafe {
                std::alloc::dealloc(
                    self.raw.base_ptr(),
                    std::alloc::Layout::from_size_align_unchecked(self.alloc_len, 8),
                );
            }
        }
    }
}

// ========================= portable fallback: paged, Mutex-serialized =========================

/// The portable backing: zeroed `page`-sized chunks in a `BTreeMap`, committed on first write, all
/// behind a `Mutex`. Used on non-unix targets and for reservations too large to `mmap`. Correct
/// under sharing but fully serialized — the fallback, not the parallel substrate (which is `Mapped`).
pub struct Paged {
    size: u64,
    page: u64,
    pages: Mutex<BTreeMap<u64, Vec<u8>>>,
}

impl Paged {
    fn new(size: u64, page: u64) -> Paged {
        Paged {
            size,
            page: page.max(1),
            pages: Mutex::new(BTreeMap::new()),
        }
    }

    /// Lock the page map, recovering from a poisoned lock (our ops never panic while holding it, so
    /// the data is always consistent) rather than propagating the panic.
    fn lock(&self) -> std::sync::MutexGuard<'_, BTreeMap<u64, Vec<u8>>> {
        self.pages.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn byte(&self, off: u64) -> u8 {
        let idx = (off % self.page) as usize;
        self.lock().get(&(off / self.page)).map_or(0, |p| p[idx])
    }

    // The non-mmap fallback has no contiguous backing, so the width-specialized word ops just reuse
    // the per-byte path (this backend is the rare case where `mmap` was unavailable).
    fn read_word(&self, off: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            raw |= (self.byte(off + k) as u64) << (8 * k);
        }
        raw
    }

    fn write_word(&self, off: u64, width: u32, val: u64) {
        for k in 0..width as u64 {
            self.set_byte(off + k, (val >> (8 * k)) as u8);
        }
    }

    fn set_byte(&self, off: u64, b: u8) {
        let page = self.page as usize;
        let idx = (off % self.page) as usize;
        let key = off / self.page;
        self.lock().entry(key).or_insert_with(|| vec![0u8; page])[idx] = b;
    }

    fn zero(&self, off: u64, len: u64) {
        let mut map = self.lock();
        // Whole pages of the range are dropped (an absent page reads zero); partial edges are
        // overwritten byte-wise.
        let mut o = off;
        let end = off + len;
        let page_sz = self.page as usize;
        while o < end {
            let key = o / self.page;
            let page_start = key * self.page;
            let page_end = page_start + self.page;
            if o == page_start && end >= page_end {
                map.remove(&key);
                o = page_end;
            } else {
                let stop = end.min(page_end);
                let p = map.entry(key).or_insert_with(|| vec![0u8; page_sz]);
                for b in o..stop {
                    p[(b % self.page) as usize] = 0;
                }
                o = stop;
            }
        }
    }

    fn fill(&self, off: u64, len: u64, b: u8) {
        if b == 0 {
            return self.zero(off, len); // a zero fill is exactly the page-dropping `zero`
        }
        let page_sz = self.page as usize;
        let mut map = self.lock();
        for o in off..off + len {
            let key = o / self.page;
            map.entry(key).or_insert_with(|| vec![0u8; page_sz])[(o % self.page) as usize] = b;
        }
    }

    fn copy_within(&self, dst: u64, src: u64, len: u64) {
        // Snapshot the source first (overlap-safe, mirroring the interpreter oracle), then write it
        // back at `dst`. The portable path has no contiguous backing to `memmove`, so this is the
        // rare-fallback cost.
        let mut buf = vec![0u8; len as usize];
        self.read_into(src, &mut buf);
        let page_sz = self.page as usize;
        let mut map = self.lock();
        for (k, &byte) in buf.iter().enumerate() {
            let o = dst + k as u64;
            let key = o / self.page;
            map.entry(key).or_insert_with(|| vec![0u8; page_sz])[(o % self.page) as usize] = byte;
        }
    }

    fn read_into(&self, off: u64, out: &mut [u8]) {
        let map = self.lock();
        for (k, slot) in out.iter_mut().enumerate() {
            let o = off.saturating_add(k as u64); // audit #6: inert past range, no overflow
            if o >= self.size {
                break;
            }
            let idx = (o % self.page) as usize;
            *slot = map.get(&(o / self.page)).map_or(0, |p| p[idx]);
        }
    }

    // The bulk slice-store (the write counterpart of `read_into`): locks the page map ONCE and
    // copies whole page-aligned chunks with `copy_from_slice`, instead of the per-byte `set_byte`
    // (a lock + `BTreeMap` entry per byte). The hot fork/checkpoint `seed` copies the whole 2 MB
    // window through here, so on the non-mmap `Paged` path this is the difference between ~2 M
    // locked map ops and ~32 page inserts (#1080 browser bash-fork perf).
    fn write_from(&self, off: u64, data: &[u8]) {
        let page_sz = self.page as usize;
        let mut map = self.lock();
        let mut i = 0usize;
        while i < data.len() {
            let o = off.saturating_add(i as u64);
            if o >= self.size {
                break;
            }
            let idx = (o % self.page) as usize;
            let take = (page_sz - idx).min(data.len() - i);
            let p = map
                .entry(o / self.page)
                .or_insert_with(|| vec![0u8; page_sz]);
            p[idx..idx + take].copy_from_slice(&data[i..i + take]);
            i += take;
        }
    }

    // The atomic ops hold the lock across the whole read-modify-write, so they are atomic with
    // respect to one another (true atomicity vs. other backings comes from `Mapped`).
    fn load_locked(map: &BTreeMap<u64, Vec<u8>>, page: u64, off: u64, width: u32) -> u64 {
        let mut raw = 0u64;
        for k in 0..width as u64 {
            let o = off + k;
            let idx = (o % page) as usize;
            let b = map.get(&(o / page)).map_or(0, |p| p[idx]);
            raw |= (b as u64) << (8 * k);
        }
        raw
    }

    fn store_locked(map: &mut BTreeMap<u64, Vec<u8>>, page: u64, off: u64, width: u32, val: u64) {
        let page_sz = page as usize;
        for k in 0..width as u64 {
            let o = off + k;
            let idx = (o % page) as usize;
            map.entry(o / page).or_insert_with(|| vec![0u8; page_sz])[idx] = (val >> (8 * k)) as u8;
        }
    }

    fn atomic_load(&self, off: u64, width: u32) -> u64 {
        Self::load_locked(&self.lock(), self.page, off, width)
    }

    fn atomic_store(&self, off: u64, width: u32, val: u64) {
        Self::store_locked(&mut self.lock(), self.page, off, width, val);
    }

    fn atomic_rmw(&self, off: u64, width: u32, op: RmwOp, val: u64) -> u64 {
        let mut map = self.lock();
        let old = Self::load_locked(&map, self.page, off, width);
        Self::store_locked(
            &mut map,
            self.page,
            off,
            width,
            rmw_apply(op, old, val, width),
        );
        old
    }

    fn atomic_cmpxchg(&self, off: u64, width: u32, expected: u64, replacement: u64) -> u64 {
        let mut map = self.lock();
        let old = Self::load_locked(&map, self.page, off, width);
        if old == (expected & width_mask(width)) {
            Self::store_locked(&mut map, self.page, off, width, replacement);
        }
        old
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn each_region(size: u64, page: u64, mut f: impl FnMut(Region)) {
        f(Region::new(size, page)); // platform default (mmap on unix)
        f(Region::Paged(Paged::new(size, page))); // force the portable path
        f(Region::owned_zeroed(size, page).expect("test sizes allocate")); // the owned flat buffer
    }

    /// #816 item 3 — the owned flat backing's contract: flat-addressable (`raw_base`) on every
    /// target with an 8-aligned base (valid naturally-aligned atomics), zero-initialized, sized
    /// as asked; a zero-size ask is `None` (fail-soft), not a zero-size allocation.
    #[test]
    fn owned_region_is_flat_aligned_zeroed() {
        let r = Region::owned_zeroed(1 << 16, 4096).expect("64 KiB allocates");
        let base = r.raw_base().expect("an owned region is flat-addressable");
        assert_eq!(base as usize % 8, 0, "8-aligned for the widest atomic");
        assert_eq!(r.len(), 1 << 16);
        assert_eq!(r.byte(0), 0, "zero-initialized");
        assert_eq!(r.byte((1 << 16) - 1), 0, "to the last byte");
        assert!(Region::owned_zeroed(0, 4096).is_none(), "zero-size is None");
    }

    #[test]
    fn byte_rw_and_zero_default() {
        each_region(1 << 16, 4096, |r| {
            assert_eq!(r.byte(10), 0);
            r.set_byte(10, 0xAB);
            assert_eq!(r.byte(10), 0xAB);
            r.zero(0, 4096);
            assert_eq!(r.byte(10), 0);
        });
    }

    #[test]
    fn out_of_range_is_inert() {
        let r = Region::new(4096, 4096);
        r.set_byte(1 << 20, 1); // ignored
        assert_eq!(r.byte(1 << 20), 0);
    }

    #[test]
    fn atomics_value_semantics() {
        each_region(1 << 16, 4096, |r| {
            r.atomic_store(8, 8, 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_rmw(8, 8, RmwOp::Add, 1), 0x1122_3344_5566_7788);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7789);
            // cmpxchg miss leaves it; hit swaps it.
            assert_eq!(r.atomic_cmpxchg(8, 8, 0, 7), 0x1122_3344_5566_7789);
            assert_eq!(r.atomic_load(8, 8), 0x1122_3344_5566_7789);
            assert_eq!(
                r.atomic_cmpxchg(8, 8, 0x1122_3344_5566_7789, 7),
                0x1122_3344_5566_7789
            );
            assert_eq!(r.atomic_load(8, 8), 7);
            // 32-bit width truncates.
            r.atomic_store(16, 4, 0xDEAD_BEEF);
            assert_eq!(r.atomic_load(16, 4), 0xDEAD_BEEF);
            assert_eq!(r.atomic_rmw(16, 4, RmwOp::Xchg, 1), 0xDEAD_BEEF);
        });
    }

    #[test]
    fn read_into_spans_pages() {
        each_region(1 << 16, 4096, |r| {
            r.set_byte(4095, 1);
            r.set_byte(4096, 2);
            let mut out = [0u8; 4];
            r.read_into(4094, &mut out);
            assert_eq!(out, [0, 1, 2, 0]);
        });
    }

    #[test]
    fn write_from_round_trips_across_pages_and_backings() {
        each_region(1 << 16, 4096, |r| {
            // A slice that spans a page boundary (partial head + whole-page interior + partial tail):
            // the bulk store must land byte-identically to a per-byte `set_byte` loop.
            let data: Vec<u8> = (0u32..8200).map(|i| (i % 251) as u8).collect();
            r.write_from(4090, &data);
            let mut out = vec![0u8; data.len()];
            r.read_into(4090, &mut out);
            assert_eq!(out, data, "write_from → read_into round-trips");
            assert_eq!(r.byte(4089), 0, "the byte before the span is untouched");
            assert_eq!(
                r.byte(4090 + data.len() as u64),
                0,
                "the byte after is untouched"
            );
            // A second write over part of it overwrites exactly that sub-span, nothing more.
            r.write_from(4090, &[0xEE; 3]);
            assert_eq!([r.byte(4090), r.byte(4091), r.byte(4092)], [0xEE; 3]);
            assert_eq!(r.byte(4093), data[3], "past the overwrite is unchanged");
            // An out-of-range tail is dropped (the caller confined; belt-and-suspenders), no panic.
            r.write_from((1 << 16) - 2, &[1, 2, 3, 4]);
            assert_eq!([r.byte((1 << 16) - 2), r.byte((1 << 16) - 1)], [1, 2]);
        });
    }

    #[test]
    fn fill_sets_span_and_clamps() {
        each_region(1 << 16, 4096, |r| {
            r.fill(10, 5, 0xAB);
            assert_eq!(r.byte(9), 0);
            for o in 10..15 {
                assert_eq!(r.byte(o), 0xAB);
            }
            assert_eq!(r.byte(15), 0);
            // A zero fill clears it again (the `zero` fast path).
            r.fill(10, 5, 0);
            assert_eq!(r.byte(12), 0);
            // Length past the region end is clamped, not out-of-bounds.
            r.fill((1 << 16) - 2, 100, 0xCD);
            assert_eq!(r.byte((1 << 16) - 1), 0xCD);
        });
    }

    #[test]
    fn copy_within_is_overlap_safe() {
        // Reference: an overlap-safe (memmove) copy against a plain Vec.
        let scalar = |src_off: u64, dst_off: u64, len: u64, seed: &[u8]| -> Vec<u8> {
            let mut v = seed.to_vec();
            let window: Vec<u8> = (0..len).map(|k| v[(src_off + k) as usize]).collect();
            for (k, b) in window.iter().enumerate() {
                v[dst_off as usize + k] = *b;
            }
            v
        };
        let cases = [
            (0u64, 8u64, 8u64), // disjoint forward
            (8, 0, 8),          // disjoint backward
            (0, 4, 8),          // overlap, dst > src (backward memmove)
            (4, 0, 8),          // overlap, dst < src (forward memmove)
            (0, 0, 8),          // self-copy no-op
        ];
        for (src, dst, len) in cases {
            let seed: Vec<u8> = (0..64u8).collect();
            each_region(1 << 16, 4096, |r| {
                for (i, b) in seed.iter().enumerate() {
                    r.set_byte(i as u64, *b);
                }
                r.copy_within(dst, src, len);
                let want = scalar(src, dst, len, &seed);
                for (i, b) in want.iter().enumerate() {
                    assert_eq!(
                        r.byte(i as u64),
                        *b,
                        "case src={src} dst={dst} len={len} at {i}"
                    );
                }
            });
        }
    }

    #[test]
    fn region_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Region>();
    }

    /// The headline Phase-2 capability: many OS threads sharing `&Region` and racing on one atomic
    /// counter still land on the exact total — i.e. the atomic RMWs are genuinely atomic across
    /// threads over the shared substrate, not just value-correct single-threaded.
    #[test]
    fn shared_atomic_counter_across_threads() {
        let threads: u64 = 8;
        let iters: u64 = if cfg!(miri) { 200 } else { 20_000 }; // miri's race detector is slow
        let r = Region::new(1 << 16, 4096);
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| {
                    for _ in 0..iters {
                        r.atomic_rmw(0, 8, RmwOp::Add, 1);
                    }
                });
            }
        });
        assert_eq!(r.atomic_load(0, 8), threads * iters);
    }

    /// Non-atomic sharing too: threads writing *disjoint* byte ranges through one `&Region` all land
    /// (no data race — distinct addresses), proving the shared image is one backing, not per-thread.
    #[test]
    fn shared_disjoint_plain_writes() {
        let threads: u64 = 8;
        let span: u64 = if cfg!(miri) { 128 } else { 1024 };
        let r = Region::new(1 << 16, 4096);
        std::thread::scope(|s| {
            for t in 0..threads {
                let r = &r;
                s.spawn(move || {
                    let v = (t as u8).wrapping_add(1);
                    for i in 0..span {
                        r.set_byte(t * span + i, v);
                    }
                });
            }
        });
        for t in 0..threads {
            let v = (t as u8).wrapping_add(1);
            assert_eq!(r.byte(t * span), v);
            assert_eq!(r.byte(t * span + span - 1), v);
        }
    }

    /// The parallel-wasm backing carries the same genuine cross-thread atomics as the `mmap` path,
    /// over **caller-owned** memory: 8 OS threads racing one counter through `&Region::Shared` land on
    /// the exact total. This is the native stand-in for the wasm Worker pool — identical
    /// `core::sync::atomic` lowering (`i32`/`i64.atomic.rmw` under `+atomics`), here over a heap
    /// allocation the threads share — so a green run here means the substrate is parallel-ready.
    #[test]
    fn shared_backing_atomic_counter_across_threads() {
        let threads: u64 = 8;
        let iters: u64 = if cfg!(miri) { 200 } else { 20_000 };
        let size = 1u64 << 16;
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        // SAFETY: 8-aligned (widest atomic) zeroed backing; freed after the region + threads finish.
        // Raw alloc (not `Vec`) so miri checks the raw-pointer atomics with no Rust-reference aliasing
        // — the same discipline `Mapped` uses under miri.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        {
            // SAFETY: `base` is `size` valid 8-aligned bytes living to the end of this block, touched
            // only through `r` (and `&r` shared across the scoped threads).
            let r = unsafe { Region::shared(base, size) };
            std::thread::scope(|s| {
                for _ in 0..threads {
                    s.spawn(|| {
                        for _ in 0..iters {
                            r.atomic_rmw(0, 8, RmwOp::Add, 1);
                        }
                    });
                }
            });
            assert_eq!(r.atomic_load(0, 8), threads * iters);
        }
        // SAFETY: same layout; `r` (and all borrows) dropped above.
        unsafe { std::alloc::dealloc(base, layout) };
    }

    /// The in-tree differential ([`differential`]) with the test seed and op count (miri runs every op
    /// through its interpreter + provenance/race checkers, so far fewer there).
    fn fuzz_against(a: &Region, b: &Region, size: u64, page: u64) {
        let ops = if cfg!(miri) { 400 } else { 20_000 };
        differential(a, b, size, page, ops, 0x9e37_79b9_7f4a_7c15).unwrap();
    }

    /// The **one** raw-pointer accessor body (`Shared`, which `Mapped` now merely `mmap`-owns and
    /// delegates to) vs the `Paged` safe reference — gates that the `unsafe` atomics/bulk ops agree
    /// with the safe model byte-for-byte across 20k ops. The former `Mapped`-vs-`Paged` twin was a
    /// drift gate between two copies of this body; with the copies merged, drift is unrepresentable,
    /// so that test could no longer fail and was deleted (invariant 1). The unix `mmap` lifecycle
    /// stays covered by `Region::new` in the unit tests + `shared_atomic_counter_across_threads`.
    #[test]
    fn differential_shared_vs_paged_fuzz() {
        let (size, page) = (3 * 4096, 4096);
        let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
        // SAFETY: an 8-aligned zeroed backing, freed below once `a` (no borrows escape) is dropped.
        let base = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!base.is_null());
        // SAFETY: `base` is `size` valid 8-aligned bytes used only through `a` within this scope.
        let a = unsafe { Region::shared(base, size) };
        fuzz_against(&a, &Region::Paged(Paged::new(size, page)), size, page);
        drop(a);
        // SAFETY: same layout; `a` dropped above.
        unsafe { std::alloc::dealloc(base, layout) };
    }
}

/// A native stand-in for the browser's JS-backed [`ForeignOps`]: each foreign id is a zeroed `Vec<u8>`
/// behind a `Mutex`, atomics done with the same value math the `Paged` reference uses. Lets the
/// `Foreign` dispatch (bounds, clamping, word assembly, kind codes) be gated natively; the browser's
/// real-Chromium self-test gates the JS half over a real `WebAssembly.Memory`.
#[cfg(test)]
mod mock_foreign {
    use super::{rmw_apply, width_mask, ForeignOps, RmwOp};
    use std::sync::Mutex;

    static MEMS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

    pub fn new_mem(size: u64) -> u32 {
        let mut g = MEMS.lock().unwrap();
        g.push(vec![0u8; size as usize]);
        (g.len() - 1) as u32
    }

    fn read(id: u32, off: u64, out: &mut [u8]) {
        let g = MEMS.lock().unwrap();
        let o = off as usize;
        out.copy_from_slice(&g[id as usize][o..o + out.len()]);
    }
    fn write(id: u32, off: u64, data: &[u8]) {
        let mut g = MEMS.lock().unwrap();
        let o = off as usize;
        g[id as usize][o..o + data.len()].copy_from_slice(data);
    }
    fn fill(id: u32, off: u64, len: u64, b: u8) {
        let mut g = MEMS.lock().unwrap();
        let o = off as usize;
        g[id as usize][o..o + len as usize].fill(b);
    }
    fn copy_within(id: u32, dst: u64, src: u64, len: u64) {
        let mut g = MEMS.lock().unwrap();
        let (d, s, n) = (dst as usize, src as usize, len as usize);
        g[id as usize].copy_within(s..s + n, d);
    }
    fn atomic(id: u32, kind: u32, off: u64, width: u32, a: u64, b: u64) -> u64 {
        let mut g = MEMS.lock().unwrap();
        let m = &mut g[id as usize];
        let o = off as usize;
        let w = width as usize;
        let mut raw = [0u8; 8];
        raw[..w].copy_from_slice(&m[o..o + w]);
        let old = u64::from_le_bytes(raw) & width_mask(width);
        let new = match kind {
            0 => return old,
            1 => a & width_mask(width),
            8 => {
                if old == (a & width_mask(width)) {
                    b & width_mask(width)
                } else {
                    return old;
                }
            }
            k => {
                let op = [
                    RmwOp::Add,
                    RmwOp::Sub,
                    RmwOp::And,
                    RmwOp::Or,
                    RmwOp::Xor,
                    RmwOp::Xchg,
                ][(k - 2) as usize];
                rmw_apply(op, old, a, width)
            }
        };
        m[o..o + w].copy_from_slice(&new.to_le_bytes()[..w]);
        old
    }

    pub static OPS: ForeignOps = ForeignOps {
        read,
        write,
        fill,
        copy_within,
        atomic,
    };
}

#[cfg(test)]
mod foreign_tests {
    use super::*;

    /// The proxied `Foreign` dispatch vs the `Paged` safe reference — the same 20k-op differential the
    /// raw body is gated by. Covers bounds/clamping before the call, word assembly, and the atomic
    /// kind codes the JS side must honour ([`ForeignOps`] contract).
    #[test]
    fn differential_foreign_vs_paged_fuzz() {
        let (size, page) = (3 * 4096, 4096);
        let id = mock_foreign::new_mem(size);
        let a = Region::foreign(id, size, &mock_foreign::OPS);
        differential(
            &a,
            &Region::paged(size, page),
            size,
            page,
            20_000,
            0x1234_5678_9abc_def1,
        )
        .unwrap();
    }

    /// A `Foreign` is not flat-addressable, its length follows the foreign memory's growth (never
    /// shrinking), and accesses past the current length are inert — exactly `Paged`'s contract.
    #[test]
    fn foreign_length_grows_and_is_not_flat() {
        let id = mock_foreign::new_mem(8192);
        let r = Region::foreign(id, 4096, &mock_foreign::OPS);
        assert!(r.raw_base().is_none());
        assert!(r.raw_base_at(0).is_none());
        r.set_byte(5000, 7);
        assert_eq!(
            r.byte(5000),
            0,
            "past the current length: dropped / reads zero"
        );
        r.set_foreign_len(8192);
        assert_eq!(r.len(), 8192);
        r.set_foreign_len(100);
        assert_eq!(r.len(), 8192, "never shrinks");
        r.write_word(5000, 4, 0xdead_beef);
        assert_eq!(r.read_word(5000, 4), 0xdead_beef);
        assert_eq!(r.atomic_rmw(5000, 4, RmwOp::Add, 1), 0xdead_beef);
        assert_eq!(r.atomic_load(5000, 4), 0xdead_bef0);
    }
}

#[cfg(test)]
mod backing_bound_tests {
    //! #1191 — the word/atomic accessors are bounded to the backing: an interpreter that admitted a
    //! page past a caller-provided backing (the reservation is wider than a `Region::shared` window
    //! slice) must read zero / drop the write, never touch the host memory behind the buffer.
    use super::*;

    #[test]
    fn word_and_atomic_accessors_stop_at_the_backing_end() {
        // A 32-byte buffer; the region covers only the first 16 — the last 16 are the canary.
        let mut buf = [0u8; 32];
        for (i, b) in buf.iter_mut().enumerate().skip(16) {
            *b = 0xA5 ^ i as u8;
        }
        let canary = buf[16..].to_vec();
        // SAFETY: `buf` outlives the region and is touched only through it below.
        let r = unsafe { Region::shared(buf.as_mut_ptr(), 16) };
        r.write_word(8, 8, 0x1122_3344_5566_7788);
        assert_eq!(
            r.read_word(8, 8),
            0x1122_3344_5566_7788,
            "in-range round-trips"
        );
        // Exactly past the end, and straddling it: reads zero, writes dropped, rmw/cmpxchg inert.
        for (off, width) in [(16u64, 8u32), (12, 8), (15, 2), (u64::MAX - 3, 8)] {
            r.write_word(off, width, u64::MAX);
            assert_eq!(
                r.read_word(off, width),
                0,
                "read past the backing at {off}+{width}"
            );
            r.atomic_store(off, width.max(4), u64::MAX);
            assert_eq!(r.atomic_load(off, width.max(4)), 0);
            assert_eq!(r.atomic_rmw(off, width.max(4), RmwOp::Add, 1), 0);
            assert_eq!(r.atomic_cmpxchg(off, width.max(4), 0, 1), 0);
        }
        assert_eq!(
            r.read_word(8, 8),
            0x1122_3344_5566_7788,
            "a straddling write is dropped whole, not truncated into the in-range prefix"
        );
        drop(r);
        assert_eq!(
            &buf[16..],
            &canary[..],
            "the bytes behind the backing are untouched"
        );
    }
}
