//! **The undo journal** — incremental guest history, so stepping backward is *undo* rather than
//! *restore the nearest checkpoint and replay forward* (#1556, #1557).
//!
//! Time travel today is snapshot-at-intervals plus deterministic replay to bridge the gap: `seek(t)`
//! restores the nearest [`Ladder`](crate::moment::Ladder) rung and re-executes, and `step_back` is
//! `seek(t−1)`. That is what INTERACTIVE_EMBEDDING.md's direction item 2 specified for v1, which also
//! said an undo log could come later "if replay-cost ever matters; it changes nothing observable".
//! This is that log.
//!
//! # What it records
//!
//! A **pre-image**: for each byte range an op is about to overwrite, the bytes it held first. Undoing
//! to a coordinate is then re-applying those pre-images in reverse order. The spans come from
//! [`watch_accesses`](crate::watch_accesses), the same per-op analysis the watchpoint check already
//! runs, so bulk `mem.copy`/`mem.fill`/`mem.move` and v128 stores are covered on the one definition
//! and no store path is touched. The hook is **pre-op, in the debug driver only**: nothing in `Mem`,
//! nothing in the confinement lowering, and nothing in emitted code — the debug engines are
//! interpreters (DEBUGGING.md: JIT tier-up is never enabled there), so this is not the write-barrier
//! surface #1454 declined (INVARIANTS #2).
//!
//! # Granularity, and the bound the design rests on
//!
//! Entries age through levels, and a compaction is **never revisited** — which is what keeps the
//! policy a pure parameter (a later dynamic policy swaps fixed values for computed ones at the same
//! call sites, rather than changing the shape).
//!
//! - **Level 1, fine**: one entry per write, so undo can stop anywhere. Size grows with write *count*.
//! - **Level 2, coalesced**: per address, only the **earliest** pre-image in the segment — all that
//!   "undo to the segment start" needs. Size is the count of **distinct addresses touched**, so a loop
//!   hammering one cell a million times collapses to that cell, and a segment is bounded above by the
//!   window itself. **A compacted segment is therefore never worse than a snapshot**, which is what
//!   makes journaling safe to age; it costs the ability to stop *inside* the segment, where the
//!   anchor-plus-replay path still serves.
//!
//! Level 3 is the existing [`Moment`](crate::moment::Moment) rung and is unchanged by this module.
//!
//! # Scope of this slice
//!
//! Window bytes only. The continuation (registers, frames, fiber chain, task states) is small next to
//! the window and is not inverted here; until it is journaled too, [`Journal::undo_window_to`] restores
//! *guest memory* and the caller supplies the continuation, which is what the differential against the
//! replay path checks. Cap-input entries and continuation deltas are the remainder of #1557.

use crate::Mem;

/// One journaled mutation, tagged with the coordinate (the scheduler turn) of the op that made it.
///
/// Only the pre-image is kept: undo re-applies it, and a pre-image for a write that ended up not
/// happening (the op trapped) restores bytes that never changed, which is a no-op. That makes the
/// pre-op hook fail-safe rather than requiring the write to be confirmed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// The turn of the op that was about to perform this write.
    pub coord: u64,
    /// Confined window base of the range (post-bounds-check, as `watch_accesses` reports it).
    pub base: u64,
    /// The bytes `[base, base + pre.len())` held before the op ran.
    pub pre: Vec<u8>,
}

/// How much a journal is holding — the numbers #1557 and #1558 report, and the inputs to the bail
/// criteria on #1556.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct JournalStats {
    /// Entries currently held.
    pub entries: usize,
    /// Pre-image bytes currently held.
    pub bytes: usize,
    /// Entries ever appended, including those since coalesced away.
    pub appended: usize,
    /// Pre-image bytes ever appended, including those since coalesced away. The ratio of this to
    /// [`bytes`](Self::bytes) after [`coalesce`](Journal::coalesce) is the coalescing ratio.
    pub appended_bytes: usize,
}

/// An ordered journal of window pre-images (see the module docs).
///
/// Disarmed by default and inert when disarmed: [`record_write`](Self::record_write) returns immediately, so
/// a run that never arms one pays a single boolean test per op (INVARIANTS #9b — observation never
/// perturbs semantics, and an unarmed observer costs nothing).
#[derive(Default)]
pub struct Journal {
    entries: Vec<Entry>,
    armed: bool,
    appended: usize,
    appended_bytes: usize,
}

impl Journal {
    /// A disarmed journal.
    pub fn new() -> Journal {
        Journal::default()
    }

    /// Arm or disarm recording. Disarming keeps what is already held, so a session can stop extending
    /// history without losing the ability to undo into it.
    pub fn set_armed(&mut self, armed: bool) {
        self.armed = armed;
    }

    /// Whether recording is on.
    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// What this journal is holding.
    pub fn stats(&self) -> JournalStats {
        JournalStats {
            entries: self.entries.len(),
            bytes: self.entries.iter().map(|e| e.pre.len()).sum(),
            appended: self.appended,
            appended_bytes: self.appended_bytes,
        }
    }

    /// The coordinate of the oldest entry held, or `None` when empty — the earliest turn
    /// [`undo_window_to`](Self::undo_window_to) can reach without an anchor.
    pub fn earliest(&self) -> Option<u64> {
        self.entries.first().map(|e| e.coord)
    }

    /// Record the pre-image of one range the op at `coord` is about to **write**. `abs_base` is the
    /// already-confined absolute address the caller took from
    /// [`watch_accesses`](crate::watch_accesses), the same per-op analysis the watchpoint check runs,
    /// so bulk `mem.copy`/`mem.fill` and v128 spans arrive here on the one definition. A no-op while
    /// disarmed, and for a zero-width span.
    pub(crate) fn record_write(&mut self, coord: u64, abs_base: u64, width: u32, mem: &Mem) {
        if !self.armed || width == 0 {
            return;
        }
        let pre = mem.read_abs(abs_base, width as usize);
        self.appended += 1;
        self.appended_bytes += pre.len();
        self.entries.push(Entry {
            coord,
            base: abs_base,
            pre,
        });
    }

    /// Undo every entry at a coordinate `>= coord`, restoring the window to the state it held *before*
    /// the op at `coord` ran. Entries are re-applied newest-first, so an address written several times
    /// ends at its oldest pre-image, and the undone entries are dropped.
    ///
    /// Returns the number of entries applied. Restores **guest memory only**; the caller supplies the
    /// continuation (see the module docs on this slice's scope).
    pub(crate) fn undo_window_to(&mut self, coord: u64, mem: &mut Mem) -> usize {
        let first = self.entries.partition_point(|e| e.coord < coord);
        let mut applied = 0;
        for e in self.entries[first..].iter().rev() {
            // A pre-image goes back through the same pure byte view it was read through, so undo is
            // symmetric with capture and a protection change made after the write cannot block it.
            mem.write_abs(e.base, &e.pre);
            applied += 1;
        }
        self.entries.truncate(first);
        applied
    }

    /// Drop everything, keeping the cumulative counters (so a session that discards history still
    /// reports what it recorded).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// **Level 2**: coalesce every entry with a coordinate `< before` down to one pre-image per
    /// address — the *earliest*, which is all "undo to the segment start" needs.
    ///
    /// This is the compaction whose bound the design rests on: the result is sized by the number of
    /// distinct addresses touched in the segment, not by how many times they were written, and is
    /// bounded above by the window. Entries at or after `before` (the fine tail) are untouched, so
    /// recent history keeps per-op granularity.
    ///
    /// The coalesced entries are re-tagged with the segment's earliest coordinate, because after this
    /// the segment can only be undone as a unit: positions inside it are no longer reachable by undo
    /// and come from the anchor-plus-replay path instead. That loss is the trade, and it is why a
    /// compaction is never revisited.
    pub fn coalesce(&mut self, before: u64) {
        let split = self.entries.partition_point(|e| e.coord < before);
        if split == 0 {
            return;
        }
        let tail = self.entries.split_off(split);
        let segment_coord = self.entries.first().map_or(0, |e| e.coord);
        // Earliest-wins per byte: walk oldest-first and only fill a byte not already carried. A
        // byte-keyed map keeps this exact under overlapping and differently-sized spans, which an
        // address-range-keyed one would not.
        let mut earliest: std::collections::BTreeMap<u64, u8> = std::collections::BTreeMap::new();
        for e in self.entries.iter() {
            for (i, b) in e.pre.iter().enumerate() {
                earliest.entry(e.base + i as u64).or_insert(*b);
            }
        }
        // Re-emit as maximal contiguous runs, so the coalesced form is a handful of spans rather than
        // one entry per byte.
        let mut out: Vec<Entry> = Vec::new();
        for (addr, b) in earliest {
            match out.last_mut() {
                Some(last) if last.base + last.pre.len() as u64 == addr => last.pre.push(b),
                _ => out.push(Entry {
                    coord: segment_coord,
                    base: addr,
                    pre: vec![b],
                }),
            }
        }
        out.extend(tail);
        self.entries = out;
    }
}
