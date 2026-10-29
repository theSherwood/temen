//! Generation-checked handle tables (DESIGN.md §3c): the one layout and the one rule a domain's
//! handles are issued and resolved by. A domain holds two tables of them, each shared by all its
//! vCPUs: its capabilities (the powerbox), and its children, where `thread.spawn` threads and §14
//! children share one table (#2043, D1–D3).
//!
//! A handle is a positive `i32` packing `(generation << SLOT_BITS) | slot`. A slot's counter
//! advances when the slot issues a handle, never when the handle is retired, so every generation in
//! `1..=counter` was once live. That splits every handle a table does not hold live in two
//! (INVARIANTS #5, #2667):
//! - **spent**: a generation the slot issued and has since retired (a child joined or detached, a
//!   capability closed), or reissued past. Its use answers an errno.
//! - **forged**: anything else (negative, past the table, generation 0, or a generation above the
//!   counter). Its use traps.

use alloc::vec::Vec;

/// `log2` of a table's capacity: 1,024 live entries at once, the POSIX personality's `FD_MAX`
/// (#2365).
pub const SLOT_BITS: u32 = 10;
/// How many slots a table holds at most.
pub const SLOTS: usize = 1 << SLOT_BITS;
/// The bits a handle carries for its generation: the 31 bits of a **positive** `i32` above the slot
/// bits. Bit 31 stays clear, so a handle is never negative, and a guest can tell it from a `-errno`
/// (#2365).
pub const GEN_BITS: u32 = 31 - SLOT_BITS;
/// A slot's counter is a full `u32`, compared **masked** to [`GEN_BITS`]: a slot keeps recycling when
/// its counter wraps, instead of dying once no handle can carry its generation. ABA resistance is
/// `2^GEN_BITS` (~2M) issues of one slot; a handle that aliases after a full wrap names one of the
/// same domain's own entries, never another domain's.
pub const GEN_MASK: u32 = (1 << GEN_BITS) - 1;
const _: () = assert!(SLOT_BITS + GEN_BITS == 31);

/// The handle of `slot` at `generation`. Every handle is packed here, so the layout lives in one
/// place.
pub const fn pack(slot: u32, generation: u32) -> i32 {
    (((generation & GEN_MASK) << SLOT_BITS) | slot) as i32
}

/// The slot a handle names, masked into the table's capacity.
pub const fn slot_of(handle: i32) -> usize {
    (handle as u32 as usize) & (SLOTS - 1)
}

/// The generation a handle carries.
pub const fn generation_of(handle: i32) -> u32 {
    (handle as u32) >> SLOT_BITS
}

/// How a [`Table`] names a handle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Named {
    /// It holds a live entry at this slot.
    Live(usize),
    /// It was issued, and its entry has since been retired (or its slot reissued): an errno.
    Spent,
    /// It was never issued: a trap.
    Forged,
}

/// One slot: its counter, the generation of the last handle it issued, and its entry while that
/// handle is live.
#[derive(Clone, Debug)]
pub struct Slot<T> {
    pub generation: u32,
    pub entry: Option<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Slot {
            generation: 0,
            entry: None,
        }
    }
}

/// A generation-checked handle table, grown as its slots are first used, to at most [`SLOTS`].
#[derive(Clone, Debug)]
pub struct Table<T> {
    slots: Vec<Slot<T>>,
}

impl<T> Default for Table<T> {
    fn default() -> Self {
        Table::new()
    }
}

impl<T> Table<T> {
    pub const fn new() -> Self {
        Table { slots: Vec::new() }
    }

    /// A table whose every slot is free at the counter `generations` gives it, as a freeze left it
    /// ([`Table::generations`]). Its live entries go back with [`Table::pin`].
    pub fn from_generations(generations: &[u32]) -> Self {
        Table {
            slots: generations
                .iter()
                .map(|&generation| Slot {
                    generation,
                    entry: None,
                })
                .collect(),
        }
    }

    /// Issue `entry` into the lowest free slot, advancing its counter: the handle that names it.
    /// `Err(entry)` when every slot is live.
    pub fn issue(&mut self, entry: T) -> Result<i32, T> {
        let slot = match self.slots.iter().position(|s| s.entry.is_none()) {
            Some(slot) => slot,
            None if self.slots.len() < SLOTS => {
                self.slots.push(Slot::default());
                self.slots.len() - 1
            }
            None => return Err(entry),
        };
        let s = &mut self.slots[slot];
        s.generation = s.generation.wrapping_add(1);
        // A wrap skips generation 0, so no handle is `0`, which a spawn record reads as none
        // (#2365).
        if s.generation & GEN_MASK == 0 {
            s.generation = s.generation.wrapping_add(1);
        }
        s.entry = Some(entry);
        Ok(pack(slot as u32, s.generation))
    }

    /// How this table names `handle`.
    pub fn name(&self, handle: i32) -> Named {
        if handle < 0 {
            return Named::Forged;
        }
        let slot = slot_of(handle);
        let gen = generation_of(handle);
        let Some(s) = self.slots.get(slot) else {
            return Named::Forged;
        };
        if s.entry.is_some() && s.generation & GEN_MASK == gen {
            Named::Live(slot)
        } else if gen >= 1 && (s.generation > GEN_MASK || gen <= s.generation) {
            Named::Spent
        } else {
            Named::Forged
        }
    }

    /// The live entry `handle` names, if it names one.
    pub fn get_live(&self, handle: i32) -> Option<(usize, &T)> {
        match self.name(handle) {
            Named::Live(slot) => self.slots[slot].entry.as_ref().map(|e| (slot, e)),
            _ => None,
        }
    }

    /// The handle `slot` issued last.
    pub fn handle(&self, slot: usize) -> i32 {
        pack(slot as u32, self.slots[slot].generation)
    }

    pub fn get(&self, slot: usize) -> Option<&T> {
        self.slots.get(slot).and_then(|s| s.entry.as_ref())
    }

    pub fn get_mut(&mut self, slot: usize) -> Option<&mut T> {
        self.slots.get_mut(slot).and_then(|s| s.entry.as_mut())
    }

    /// Retire `slot`'s entry: its handle is spent from now on, and the slot is free to reissue.
    pub fn retire(&mut self, slot: usize) -> Option<T> {
        self.slots.get_mut(slot).and_then(|s| s.entry.take())
    }

    /// Pin `entry` live at `slot`, under the counter the slot already has: a thaw re-filing an entry
    /// a freeze recorded there, so the handle the guest holds resolves as before the cut. Grows the
    /// table to reach `slot`. Panics past [`SLOTS`] (a codec bug, not guest-reachable).
    pub fn pin(&mut self, slot: usize, entry: T) {
        self.slot_mut(slot).entry = Some(entry);
    }

    /// `slot` itself, the table grown to reach it. Panics past [`SLOTS`].
    pub fn slot_mut(&mut self, slot: usize) -> &mut Slot<T> {
        assert!(slot < SLOTS, "handle slot {slot} past the table");
        if self.slots.len() <= slot {
            self.slots.resize_with(slot + 1, Slot::default);
        }
        &mut self.slots[slot]
    }

    /// Every slot the table has grown to.
    pub fn slots(&self) -> &[Slot<T>] {
        &self.slots
    }

    /// The live entries, with their slots, in slot order.
    pub fn live(&self) -> impl Iterator<Item = (usize, &T)> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(slot, s)| s.entry.as_ref().map(|e| (slot, e)))
    }

    /// Whether no entry is live.
    pub fn is_idle(&self) -> bool {
        self.slots.iter().all(|s| s.entry.is_none())
    }

    /// Each slot's counter, in slot order up to the last slot that ever issued: the half of the table
    /// its live entries do not carry. A freeze carries it, so after a thaw a spent handle stays spent
    /// rather than reading as a forgery, and the next issue into a slot gives the handle the
    /// uninterrupted run would have (#2360, #2619).
    pub fn generations(&self) -> Vec<u32> {
        let len = self
            .slots
            .iter()
            .rposition(|s| s.generation != 0)
            .map_or(0, |i| i + 1);
        self.slots[..len].iter().map(|s| s.generation).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn issue_takes_the_lowest_free_slot_and_advances_its_counter() {
        let mut t = Table::new();
        let a = t.issue('a').unwrap();
        let b = t.issue('b').unwrap();
        assert_eq!((a, b), (pack(0, 1), pack(1, 1)));
        assert_eq!(t.retire(0), Some('a'));
        // Slot 0 is reused at the next generation; its old handle is spent, never the new entry.
        let c = t.issue('c').unwrap();
        assert_eq!(c, pack(0, 2));
        assert_eq!(t.name(a), Named::Spent);
        assert_eq!(t.name(c), Named::Live(0));
        assert_eq!(t.get_live(c), Some((0, &'c')));
        assert_eq!(t.get_live(a), None);
    }

    #[test]
    fn a_retired_handle_is_spent_and_a_never_issued_one_is_forged() {
        let mut t = Table::new();
        let h = t.issue(()).unwrap();
        t.retire(0);
        assert_eq!(t.name(h), Named::Spent);
        for forged in [
            -1,
            0,
            pack(0, 2),   // a generation the slot has not reached
            pack(1, 1),   // a slot past the table
            pack(5, 1),   // a slot past the table
            i32::MIN,     // negative
            pack(0, 0x7), // above the counter
        ] {
            assert_eq!(t.name(forged), Named::Forged, "{forged:#x}");
        }
    }

    #[test]
    fn a_full_table_refuses_and_hands_the_entry_back() {
        let mut t = Table::new();
        for i in 0..SLOTS {
            t.issue(i).unwrap();
        }
        assert_eq!(t.issue(SLOTS), Err(SLOTS));
        t.retire(7);
        assert_eq!(t.issue(SLOTS), Ok(pack(7, 2)));
    }

    #[test]
    fn a_wrapped_counter_skips_generation_zero_and_keeps_recycling() {
        let mut t: Table<()> = Table::from_generations(&[GEN_MASK]);
        let h = t.issue(()).unwrap();
        assert_ne!(h, 0);
        assert_eq!(generation_of(h), 1);
        assert_eq!(t.name(h), Named::Live(0));
        t.retire(0);
        // Past the wrap every masked generation was once issued, so none reads as a forgery.
        assert_eq!(t.name(pack(0, 5)), Named::Spent);
    }

    #[test]
    fn generations_round_trip_a_freeze() {
        let mut t = Table::new();
        let a = t.issue('a').unwrap();
        let b = t.issue('b').unwrap();
        let c = t.issue('c').unwrap();
        t.retire(1);
        t.retire(2);
        let gens = t.generations();
        assert_eq!(gens, vec![1, 1, 1]);
        // The thaw: counters back, the live entry pinned at its slot.
        let mut thawed = Table::from_generations(&gens);
        thawed.pin(0, 'a');
        assert_eq!(thawed.name(a), Named::Live(0));
        assert_eq!(thawed.name(b), Named::Spent);
        assert_eq!(thawed.name(c), Named::Spent);
        // The next issue gives the handle the uninterrupted run would have.
        assert_eq!(thawed.issue('d'), t.issue('d'));
    }

    #[test]
    fn a_trailing_free_slot_with_no_counter_is_not_carried() {
        let mut t: Table<()> = Table::new();
        t.slot_mut(3);
        assert!(t.generations().is_empty());
        assert!(t.is_idle());
    }
}
