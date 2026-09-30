//! A **happens-before data-race detector** over the debug session's access sink (#1987) — pure
//! tooling beside the cache model (`models.rs`): it observes the engine and never touches it.
//!
//! **What it reports.** Two accesses to the same 4-byte word by different threads, at least one a
//! write, with nothing ordering them: neither happens-before the other. Each such pair is reported
//! once per word and pair of threads. Four bytes, not eight, so two `int`s side by side that
//! different threads write aren't taken for one shared variable.
//!
//! **What orders accesses.** Each thread carries a vector clock, and so does each address an atomic
//! op touched:
//! - an atomic load **acquires** its address's clock, an atomic store **releases** into it, and an
//!   RMW or compare-exchange does both, so a mutex built on them orders what it protects;
//! - a spawn hands the child the parent's clock;
//! - a join hands the joiner the child's (a blocking join, `WakeJoin`, and one of a thread that had
//!   already finished, `Join`);
//! - a futex wake orders the waker before the wakee.
//!
//! The atomics arrive on the access sink; the lifecycle events arrive on the engine's scheduler-event
//! sink, in order with them.
//!
//! **Time travel.** A reverse `seek` rebuilds the run and replays it, and both sinks see the replay.
//! Races are found on the furthest run: an event at a turn the model has already passed is a replay,
//! and the model skips it. (A step back that then takes a different schedule keeps the races the old
//! one found.)

use std::collections::HashMap;
use temen_interp::bytecode::SchedTraceEvent;
use temen_interp::MemEvent;

/// A vector clock, indexed by thread (task) index; a missing entry is 0.
#[derive(Clone, Default, Debug)]
struct Vc(Vec<u64>);

impl Vc {
    fn get(&self, t: usize) -> u64 {
        self.0.get(t).copied().unwrap_or(0)
    }
    fn set(&mut self, t: usize, v: u64) {
        if self.0.len() <= t {
            self.0.resize(t + 1, 0);
        }
        self.0[t] = v;
    }
    fn join(&mut self, other: &Vc) {
        if self.0.len() < other.0.len() {
            self.0.resize(other.0.len(), 0);
        }
        for (a, b) in self.0.iter_mut().zip(&other.0) {
            *a = (*a).max(*b);
        }
    }
}

/// One access, as the word's shadow remembers it: who, at which of their own clock ticks, and when.
#[derive(Clone, Copy, Debug)]
struct Access {
    task: usize,
    clock: u64,
    turn: u64,
}

/// A word's shadow: its last write, and the reads since it (one per thread).
#[derive(Default, Debug)]
struct Shadow {
    write: Option<Access>,
    reads: Vec<Access>,
}

/// One side of a reported race.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceSide {
    pub task: usize,
    pub turn: u64,
    pub write: bool,
}

/// A reported race on the 4-byte word at `addr`: `first` came before `second` in the run, and nothing
/// ordered them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Race {
    pub addr: u64,
    pub first: RaceSide,
    pub second: RaceSide,
}

/// The shadow's granularity in bytes.
const WORD: u64 = 4;

/// Races reported past this many are dropped: a racy loop would otherwise report one per word.
const MAX_RACES: usize = 1024;

/// The detector's state (see the module docs).
#[derive(Default)]
pub struct RaceModel {
    threads: Vec<Vc>,
    sync: HashMap<u64, Vc>,
    words: HashMap<u64, Shadow>,
    races: Vec<Race>,
    /// `(word, earlier task, later task)` already reported.
    seen: std::collections::HashSet<(u64, usize, usize)>,
    /// The furthest turn observed; events at or before it are a replay (see the module docs).
    high: Option<u64>,
    /// The turn of the event being observed, so a later event at the same turn isn't a replay.
    at: u64,
}

impl RaceModel {
    pub fn new() -> RaceModel {
        RaceModel::default()
    }

    /// The races found so far, in the order found.
    pub fn races(&self) -> &[Race] {
        &self.races
    }

    /// Whether an event at `turn` is new ground, advancing the frontier when it is.
    fn fresh(&mut self, turn: u64) -> bool {
        match self.high {
            Some(h) if turn < h || (turn == h && turn != self.at) => false,
            _ => {
                self.high = Some(turn);
                self.at = turn;
                true
            }
        }
    }

    fn vc(&mut self, task: usize) -> &mut Vc {
        if self.threads.len() <= task {
            self.threads.resize_with(task + 1, Vc::default);
        }
        let vc = &mut self.threads[task];
        if vc.get(task) == 0 {
            vc.set(task, 1); // a thread's own clock starts at 1, so its first access has an epoch
        }
        vc
    }

    /// Tick `task`'s own clock — after it released, so what it does next is a new epoch.
    fn tick(&mut self, task: usize) {
        let vc = self.vc(task);
        let c = vc.get(task);
        vc.set(task, c + 1);
    }

    /// An access-sink event: `task` ran a memory op at `turn`.
    pub fn observe(&mut self, turn: u64, task: usize, ev: MemEvent) {
        if !self.fresh(turn) {
            return;
        }
        match ev {
            MemEvent::Load { addr, width } => self.access(turn, task, addr, width as u64, false),
            MemEvent::Store { addr, width } => self.access(turn, task, addr, width as u64, true),
            MemEvent::Copy { dst, src, len } => {
                self.access(turn, task, src, len, false);
                self.access(turn, task, dst, len, true);
            }
            MemEvent::Fill { dst, len } => self.access(turn, task, dst, len, true),
            MemEvent::AtomicLoad { addr, .. } => self.acquire(task, addr),
            MemEvent::AtomicStore { addr, .. } => self.release(task, addr),
            MemEvent::AtomicRmw { addr, .. } | MemEvent::AtomicCmpxchg { addr, .. } => {
                self.acquire(task, addr);
                self.release(task, addr);
            }
        }
    }

    /// A scheduler-sink event: the lifecycle edges (see the module docs).
    pub fn observe_sched(&mut self, ev: &SchedTraceEvent) {
        use SchedTraceEvent as E;
        let turn = match *ev {
            E::Spawn { turn, .. }
            | E::Join { turn, .. }
            | E::WakeJoin { turn, .. }
            | E::WakeNotify { turn, .. } => turn,
            _ => return,
        };
        if !self.fresh(turn) {
            return;
        }
        match *ev {
            E::Spawn { parent, task, .. } => {
                let from = self.vc(parent).clone();
                self.vc(task).join(&from);
                self.tick(parent);
            }
            E::Join { task, child, .. } => self.order(child, task),
            E::WakeJoin { waker, wakee, .. } | E::WakeNotify { waker, wakee, .. } => {
                self.order(waker, wakee)
            }
            _ => {}
        }
    }

    /// Everything `before` has done happens-before what `after` does next.
    fn order(&mut self, before: usize, after: usize) {
        let from = self.vc(before).clone();
        self.vc(after).join(&from);
        self.tick(before);
    }

    fn acquire(&mut self, task: usize, addr: u64) {
        if let Some(l) = self.sync.get(&addr).cloned() {
            self.vc(task).join(&l);
        }
    }

    fn release(&mut self, task: usize, addr: u64) {
        let vc = self.vc(task).clone();
        self.sync.entry(addr).or_default().join(&vc);
        self.tick(task);
    }

    /// A plain access of `[addr, addr + len)`: check it against each word's shadow, then record it.
    fn access(&mut self, turn: u64, task: usize, addr: u64, len: u64, write: bool) {
        if len == 0 {
            return;
        }
        let now = self.vc(task).clone();
        let me = Access {
            task,
            clock: now.get(task),
            turn,
        };
        // Whether an earlier access happens-before this one: the thread's clock has seen its epoch.
        let ordered = |a: &Access| a.task == task || a.clock <= now.get(a.task);
        for word in addr / WORD..=(addr + len - 1) / WORD {
            let shadow = self.words.entry(word).or_default();
            let mut found = Vec::new();
            if let Some(w) = shadow.write.filter(|w| !ordered(w)) {
                found.push((w, true));
            }
            if write {
                for r in shadow.reads.iter().filter(|r| !ordered(r)) {
                    found.push((*r, false));
                }
                shadow.write = Some(me);
                shadow.reads.clear();
            } else {
                shadow.reads.retain(|r| r.task != task);
                shadow.reads.push(me);
            }
            for (earlier, earlier_write) in found {
                self.report(word * WORD, earlier, earlier_write, me, write);
            }
        }
    }

    fn report(&mut self, addr: u64, a: Access, a_write: bool, b: Access, b_write: bool) {
        if self.races.len() >= MAX_RACES || !self.seen.insert((addr, a.task, b.task)) {
            return;
        }
        self.races.push(Race {
            addr,
            first: RaceSide {
                task: a.task,
                turn: a.turn,
                write: a_write,
            },
            second: RaceSide {
                task: b.task,
                turn: b.turn,
                write: b_write,
            },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(addr: u64) -> MemEvent {
        MemEvent::Store { addr, width: 4 }
    }
    fn ld(addr: u64) -> MemEvent {
        MemEvent::Load { addr, width: 4 }
    }

    /// Two threads writing the same word with nothing between them race; the second write reports it.
    #[test]
    fn unordered_writes_race() {
        let mut m = RaceModel::new();
        m.observe_sched(&SchedTraceEvent::Spawn {
            turn: 1,
            parent: 0,
            task: 1,
        });
        m.observe(2, 0, st(64));
        m.observe(3, 1, st(64));
        assert_eq!(
            m.races(),
            &[Race {
                addr: 64,
                first: RaceSide {
                    task: 0,
                    turn: 2,
                    write: true
                },
                second: RaceSide {
                    task: 1,
                    turn: 3,
                    write: true
                },
            }]
        );
    }

    /// A write before the spawn, read by the child, is ordered by the spawn: no race.
    #[test]
    fn a_spawn_orders_what_came_before_it() {
        let mut m = RaceModel::new();
        m.observe(1, 0, st(64));
        m.observe_sched(&SchedTraceEvent::Spawn {
            turn: 2,
            parent: 0,
            task: 1,
        });
        m.observe(3, 1, ld(64));
        assert!(m.races().is_empty());
    }

    /// The child's write, read after joining it, is ordered by the join — whether the join blocked
    /// (`WakeJoin`) or found the child already done (`Join`).
    #[test]
    fn a_join_orders_what_the_child_did() {
        for blocking in [true, false] {
            let mut m = RaceModel::new();
            m.observe_sched(&SchedTraceEvent::Spawn {
                turn: 1,
                parent: 0,
                task: 1,
            });
            m.observe(2, 1, st(64));
            m.observe_sched(&if blocking {
                SchedTraceEvent::WakeJoin {
                    turn: 3,
                    waker: 1,
                    wakee: 0,
                }
            } else {
                SchedTraceEvent::Join {
                    turn: 3,
                    task: 0,
                    child: 1,
                }
            });
            m.observe(4, 0, ld(64));
            assert!(m.races().is_empty(), "blocking join: {blocking}");
        }
    }

    /// A counter both threads increment under a lock (an atomic CAS to take it, an atomic store to
    /// release it) is ordered by the lock word: no race, where the same increments without it race.
    #[test]
    fn a_lock_orders_what_it_protects() {
        for locked in [true, false] {
            let mut m = RaceModel::new();
            m.observe_sched(&SchedTraceEvent::Spawn {
                turn: 1,
                parent: 0,
                task: 1,
            });
            let mut turn = 2;
            for task in [0, 1] {
                if locked {
                    m.observe(turn, task, MemEvent::AtomicCmpxchg { addr: 8, width: 4 });
                }
                m.observe(turn + 1, task, ld(64));
                m.observe(turn + 2, task, st(64));
                if locked {
                    m.observe(turn + 3, task, MemEvent::AtomicStore { addr: 8, width: 4 });
                }
                turn += 4;
            }
            assert_eq!(m.races().is_empty(), locked, "locked: {locked}");
        }
    }

    /// A replay re-delivers events the model has passed; it skips them rather than reporting twice.
    #[test]
    fn a_replay_reports_nothing_new() {
        let mut m = RaceModel::new();
        m.observe_sched(&SchedTraceEvent::Spawn {
            turn: 1,
            parent: 0,
            task: 1,
        });
        m.observe(2, 0, st(64));
        m.observe(3, 1, st(64));
        m.observe(2, 0, st(64));
        m.observe(3, 1, st(64));
        assert_eq!(m.races().len(), 1);
    }
}
