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
//! # The three things an undo must put back
//!
//! - **Window bytes** — the pre-images above, keyed by the op's turn.
//! - **The continuation** — the task set's `Vm`s, fiber chain and task states, journaled per op as a
//!   [`ScheduledContinuation`], the same shape the checkpoint ladder already captures and
//!   `ScheduledDebugRun::restore` already installs. It is small next to the window, which is the whole
//!   reason a journal beats a snapshot: the window is what a snapshot copies and a journal does not.
//! - **The host's run-mutable state** — a compact [`HostCursor`], not a substate clone. Every field an
//!   ordinary op moves is a scalar or the length of an **append-only** buffer, so lengths plus
//!   truncation invert it in O(1). Taking a full `HostReplaySubstate` every op would clone `stdout` and
//!   the cap tape, which is quadratic on a `printf` guest.
//!
//! Because the cursor carries `cap_consumed`, **the cap tape rewinds with the journal by
//! construction** — there is no second structure to keep in step, which is what an undo log carrying
//! the tape was supposed to buy.
//!
//! # Fail-closed
//!
//! Two kinds of host state cannot be inverted by a cursor: the §3.6 serve queue (it drains, so it is
//! not append-only) and a capability's **opaque declared state** (an embedder capture/restore blob has
//! no inverse). A run using either records no state entries, so [`Journal::state_at`] finds nothing and
//! the engine declines to undo, leaving the checkpoint-plus-replay path to serve it exactly as it does
//! today. Refusing is the point: restoring such a run from a cursor would be quietly wrong.

use crate::bytecode::ScheduledContinuation;
use crate::{HostCursor, Mem};

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
    /// Window pre-image entries currently held.
    pub entries: usize,
    /// Per-op state entries (continuation + host cursor) currently held.
    pub states: usize,
    /// Pre-image bytes currently held.
    pub bytes: usize,
    /// Entries ever appended, including those since coalesced away.
    pub appended: usize,
    /// Pre-image bytes ever appended, including those since coalesced away. The ratio of this to
    /// [`bytes`](Self::bytes) after [`coalesce`](Journal::coalesce) is the coalescing ratio.
    pub appended_bytes: usize,
}

/// The engine state as it stood **before** the op at `coord` ran: the continuation plus the compact
/// host cursor. Journaled once per op while armed, and coalesced to the earliest in a segment (see
/// [`Journal::coalesce`]) because that is the only one "undo to the segment start" needs.
pub(crate) struct StateEntry {
    pub(crate) coord: u64,
    pub(crate) cont: ScheduledContinuation,
    pub(crate) cursor: HostCursor,
}

/// **The retention policy** (#1558) — static, and deliberately a parameter struct rather than
/// constants at the call sites, so a later adaptive policy is the same shape with computed values.
///
/// The rule that keeps that true: a policy decides granularity **going forward only** and never
/// revisits a past compaction. Level 2 has discarded the intra-segment positions, so un-compacting is
/// impossible anyway; holding the rule is what makes static-to-dynamic a parameter change instead of an
/// architectural one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JournalPolicy {
    /// Turns behind the live position kept at **level 1** — the window in which `step_back` can stop
    /// anywhere. Older history is coalesced to level 2 and can then only be undone to a segment
    /// boundary.
    pub fine_turns: u64,
    /// Ceiling on retained pre-image bytes. Past it the **oldest** history is dropped, which bounds how
    /// far back undo reaches without ever making a reachable position wrong: a dropped turn simply has
    /// no state entry, so [`Journal::can_undo_to`] declines it and the checkpoint-plus-replay path
    /// serves. `0` means unbounded.
    pub byte_budget: usize,
}

impl Default for JournalPolicy {
    /// A fine window wide enough that ordinary interactive stepping never leaves it, and no byte
    /// ceiling — the conservative default, since dropping history is a capability loss and should be
    /// something an embedder opts into.
    fn default() -> JournalPolicy {
        JournalPolicy {
            fine_turns: 4096,
            byte_budget: 0,
        }
    }
}

/// An ordered journal of window pre-images and per-op engine state (see the module docs).
///
/// Disarmed by default and inert when disarmed: [`record_write`](Self::record_write) returns immediately, so
/// a run that never arms one pays a single boolean test per op (INVARIANTS #9b — observation never
/// perturbs semantics, and an unarmed observer costs nothing).
#[derive(Default)]
pub struct Journal {
    entries: Vec<Entry>,
    states: Vec<StateEntry>,
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

    /// What this journal is holding. `bytes` counts window pre-images only — the per-op continuations
    /// are counted by [`states`](JournalStats::states), since their cost is a `Vm` clone rather than a
    /// byte count and the two do not usefully add up.
    pub fn stats(&self) -> JournalStats {
        JournalStats {
            entries: self.entries.len(),
            bytes: self.entries.iter().map(|e| e.pre.len()).sum(),
            states: self.states.len(),
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

    /// Record the engine state as it stands **before** the op at `coord` — the continuation and the
    /// compact host cursor. A no-op while disarmed. The caller is responsible for only calling this
    /// when the state is invertible (`Host::journal_invertible` and the checkpointable subset); a turn
    /// with no state entry is one [`state_at`](Self::state_at) will decline, so undo fails closed.
    pub(crate) fn record_state(
        &mut self,
        coord: u64,
        cont: ScheduledContinuation,
        cursor: HostCursor,
    ) {
        if !self.armed {
            return;
        }
        self.states.push(StateEntry {
            coord,
            cont,
            cursor,
        });
    }

    /// The state entry recorded exactly at `coord`, or `None` if that turn has none — which is how a
    /// run outside the invertible subset, or one whose history has been compacted past `coord`,
    /// declines to be undone there.
    pub(crate) fn state_at(&self, coord: u64) -> Option<&StateEntry> {
        let i = self.states.partition_point(|s| s.coord < coord);
        self.states.get(i).filter(|s| s.coord == coord)
    }

    /// Whether [`state_at`](Self::state_at) can serve `coord` — the engine's precondition for undoing
    /// there at all.
    pub(crate) fn can_undo_to(&self, coord: u64) -> bool {
        self.state_at(coord).is_some()
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
        // The state entries from `coord` forward describe turns that no longer happened.
        let s_first = self.states.partition_point(|s| s.coord <= coord);
        self.states.truncate(s_first);
        applied
    }

    /// Drop everything, keeping the cumulative counters (so a session that discards history still
    /// reports what it recorded).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.states.clear();
    }

    /// Apply `policy` at live position `now`: coalesce history that has aged out of the fine window,
    /// then drop the oldest segments until the byte budget is met.
    ///
    /// Cheap to call every op — the common case is a bounds check and no work. Dropping is fail-closed
    /// by construction: a turn whose state entry is gone is one [`can_undo_to`](Self::can_undo_to)
    /// declines, never one it answers wrongly.
    pub(crate) fn apply_policy(&mut self, now: u64, policy: &JournalPolicy) {
        if !self.armed {
            return;
        }
        if let Some(cut) = now.checked_sub(policy.fine_turns) {
            // Only worth walking if something has actually aged out of the fine window.
            if self.entries.first().is_some_and(|e| e.coord < cut) {
                self.coalesce(cut);
            }
        }
        if policy.byte_budget == 0 {
            return;
        }
        // Drop oldest-first. Undo reaching a given turn needs only the entries from that turn forward,
        // so shedding the tail of history shortens the reach without corrupting what remains.
        let mut held: usize = self.entries.iter().map(|e| e.pre.len()).sum();
        while held > policy.byte_budget && !self.entries.is_empty() {
            let dropped = self.entries.remove(0);
            held -= dropped.pre.len();
            // Any state entry no longer backed by a full pre-image history is unusable.
            let keep_from = self.entries.first().map_or(u64::MAX, |e| e.coord);
            self.states.retain(|s| s.coord >= keep_from);
        }
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
        // The same earliest-wins rule for state: a coalesced segment can only be undone to its start,
        // so exactly one continuation — the earliest — survives it. The fine tail keeps per-op state.
        let s_split = self.states.partition_point(|s| s.coord < before);
        if s_split > 1 {
            self.states.drain(1..s_split);
        }
    }
}
