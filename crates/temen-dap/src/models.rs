//! Host-side **memory models** over the debug-session access sink (INTERACTIVE_EMBEDDING.md
//! slice 4, the X2/X4 consumer): a configurable two-level cache model with per-vCPU L1s +
//! coherence states, and a first-touch demand-paging fault counter. Pure tooling — the models
//! observe the sink's `(clock, task, MemEvent)` stream and never touch the engine, the module, or
//! the TCB; a session with no model installed pays nothing.
//!
//! **Time-travel consistency.** A reverse `seek` rebuilds the run and replays it, and the backend
//! re-installs the sink before the replay drive — so the model *observes the replay*. The model
//! keeps its own snapshot ladder at the engine's [`CHECKPOINT_STRIDE`] boundaries (state *before*
//! the first event at-or-past each boundary): the engine only ever restores to such a boundary (or
//! to clock 0), so when the stream's clock jumps backward the model restores its latest snapshot
//! at-or-before the first replayed event and re-derives from there. Net effect, pinned by tests:
//! after `seek(t)`, model state ≡ a fresh model observing a from-0 run to `t`.

use crate::backend::CHECKPOINT_STRIDE;
use crate::json::Json;
use temen_interp::MemEvent;

/// One cache level's geometry. `line` is the line size in bytes; `sets` × `ways` lines total.
/// Teaching-scale defaults keep the line-state dump small enough to render as a grid.
#[derive(Clone, Copy, Debug)]
pub struct CacheCfg {
    pub sets: u64,
    pub ways: usize,
    pub line: u64,
}

/// The model's configuration: per-vCPU L1s, one shared L2, and the paging model's page size.
#[derive(Clone, Copy, Debug)]
pub struct MemModelCfg {
    pub l1: CacheCfg,
    pub l2: CacheCfg,
    pub page_size: u64,
}

impl Default for MemModelCfg {
    fn default() -> Self {
        MemModelCfg {
            l1: CacheCfg {
                sets: 4,
                ways: 2,
                line: 64,
            },
            l2: CacheCfg {
                sets: 16,
                ways: 4,
                line: 64,
            },
            page_size: 4096,
        }
    }
}

/// An L1 line's coherence state (MESI). The L2 is a plain shared victim of the L1s — its lines
/// carry validity only.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mesi {
    Modified,
    Exclusive,
    Shared,
    Invalid,
}

impl Mesi {
    fn letter(self) -> &'static str {
        match self {
            Mesi::Modified => "M",
            Mesi::Exclusive => "E",
            Mesi::Shared => "S",
            Mesi::Invalid => "I",
        }
    }
}

/// One cache line slot: the cached line index (`addr / line`), its state, and a last-touch tick
/// for LRU. `tag: None` = never filled.
#[derive(Clone, Copy, Debug)]
struct Slot {
    tag: Option<u64>,
    state: Mesi,
    lru: u64,
}

const EMPTY: Slot = Slot {
    tag: None,
    state: Mesi::Invalid,
    lru: 0,
};

/// A set-associative array: `sets × ways` slots, direct-mapped by `line_idx % sets`.
#[derive(Clone, Debug)]
struct SetArray {
    cfg: CacheCfg,
    slots: Vec<Slot>, // sets * ways, row-major by set
}

impl SetArray {
    fn new(cfg: CacheCfg) -> SetArray {
        SetArray {
            cfg,
            slots: vec![EMPTY; cfg.sets as usize * cfg.ways],
        }
    }
    fn set_of(&self, line_idx: u64) -> &[Slot] {
        let s = (line_idx % self.cfg.sets) as usize * self.cfg.ways;
        &self.slots[s..s + self.cfg.ways]
    }
    fn set_of_mut(&mut self, line_idx: u64) -> &mut [Slot] {
        let s = (line_idx % self.cfg.sets) as usize * self.cfg.ways;
        &mut self.slots[s..s + self.cfg.ways]
    }
    /// The way holding `line_idx` (any non-Invalid state), if cached.
    fn find(&self, line_idx: u64) -> Option<usize> {
        self.set_of(line_idx)
            .iter()
            .position(|s| s.tag == Some(line_idx) && s.state != Mesi::Invalid)
    }
    /// Fill `line_idx` into its set, evicting the LRU way; returns the way filled.
    fn fill(&mut self, line_idx: u64, state: Mesi, tick: u64) -> usize {
        let set = self.set_of_mut(line_idx);
        let way = (0..set.len())
            .min_by_key(|&w| match set[w].tag {
                None => (0, 0), // an empty way first
                Some(_) if set[w].state == Mesi::Invalid => (1, set[w].lru),
                Some(_) => (2, set[w].lru), // else the least recently touched
            })
            .expect("ways >= 1");
        set[way] = Slot {
            tag: Some(line_idx),
            state,
            lru: tick,
        };
        way
    }
}

/// The rollback-able model state — everything a snapshot clones. Teaching-scale geometries keep
/// clones cheap (a few hundred slots).
#[derive(Clone, Debug)]
struct State {
    /// Per-task L1s, indexed by the sink's task attribution (grown on first touch).
    l1s: Vec<SetArray>,
    l2: SetArray,
    l1_hits: u64,
    l1_misses: u64,
    l2_hits: u64,
    l2_misses: u64,
    /// First-touch page set (page index = `addr / page_size`); `len()` = the fault count.
    pages: std::collections::BTreeSet<u64>,
    /// The **shared-state consumer** (slice 6, the X1 contested tracker): per 8-byte word, the
    /// last task to write it and whether it is *contested* — written by one task and then touched
    /// (read or written) by another. `(last_writer, contested)`; a never-written word is absent.
    words: std::collections::BTreeMap<u64, (usize, bool)>,
    /// LRU tick, monotonically increasing per line touch.
    tick: u64,
}

/// The combined cache + paging model. Feed it the sink stream via [`MemModel::observe`]; read
/// counters and the line-state grid back via [`MemModel::stats_json`].
#[derive(Clone, Debug)]
pub struct MemModel {
    cfg: MemModelCfg,
    state: State,
    /// `(boundary_clock, state-before-that-clock)` at [`CHECKPOINT_STRIDE`] boundaries, ascending.
    snaps: Vec<(u64, State)>,
    /// The clock of the last observed event (`u64::MAX` = none yet).
    last_clock: u64,
    /// Bound on span expansion (a runaway `memset` len), so one event can't spin the model.
    max_lines_per_event: u64,
    /// Whether to lay boundary snapshots (the time-travel ladder). On by default for debug
    /// sessions; a run-mode profiler (the browser W3 entry) turns it off — its clock only moves
    /// forward and a long run would otherwise accumulate snapshots for nothing.
    snapshots: bool,
}

impl MemModel {
    pub fn new(cfg: MemModelCfg) -> MemModel {
        MemModel {
            cfg,
            state: State {
                l1s: Vec::new(),
                l2: SetArray::new(cfg.l2),
                l1_hits: 0,
                l1_misses: 0,
                l2_hits: 0,
                l2_misses: 0,
                pages: std::collections::BTreeSet::new(),
                words: std::collections::BTreeMap::new(),
                tick: 0,
            },
            snaps: Vec::new(),
            last_clock: u64::MAX,
            max_lines_per_event: 1 << 20,
            snapshots: true,
        }
    }

    /// Disable the boundary-snapshot ladder (run-mode profiling: forward-only clock, no seeks).
    pub fn set_snapshots(&mut self, on: bool) {
        self.snapshots = on;
    }

    /// Observe one sink event. Handles time travel transparently: a backward clock jump restores
    /// the snapshot ladder (see the module docs), a forward pass lays boundary snapshots down.
    pub fn observe(&mut self, clock: u64, task: usize, ev: MemEvent) {
        if self.last_clock != u64::MAX && clock < self.last_clock {
            // A seek replay: restore the latest snapshot at-or-before this event's clock. The
            // engine restored at a stride boundary (or 0) ≤ clock with no events in between, so
            // that snapshot *is* the state at the engine's restore point.
            while self.snaps.last().is_some_and(|(b, _)| *b > clock) {
                self.snaps.pop();
            }
            match self.snaps.last() {
                Some((_, s)) => self.state = s.clone(),
                None => self.state = MemModel::new(self.cfg).state,
            }
        } else {
            // Forward: snapshot each stride boundary crossed since the last event — the state
            // *before* any event at-or-past the boundary (model state is constant between events).
            let from = if self.last_clock == u64::MAX {
                0
            } else {
                self.last_clock / CHECKPOINT_STRIDE + 1
            };
            if self.snapshots {
                for b in from..=clock / CHECKPOINT_STRIDE {
                    let boundary = b * CHECKPOINT_STRIDE;
                    if boundary > 0 {
                        self.snaps.push((boundary, self.state.clone()));
                    }
                }
            }
        }
        self.last_clock = clock;
        self.apply(task, ev);
    }

    fn apply(&mut self, task: usize, ev: MemEvent) {
        match ev {
            MemEvent::Load { addr, width }
            | MemEvent::AtomicLoad { addr, width }
            | MemEvent::AtomicRmw { addr, width } // an RMW reads then writes: model as the write
            | MemEvent::AtomicCmpxchg { addr, width } => {
                let write = matches!(
                    ev,
                    MemEvent::AtomicRmw { .. } | MemEvent::AtomicCmpxchg { .. }
                );
                self.span(task, addr, width as u64, write);
            }
            MemEvent::Store { addr, width } | MemEvent::AtomicStore { addr, width } => {
                self.span(task, addr, width as u64, true);
            }
            MemEvent::Copy { dst, src, len } => {
                self.span(task, src, len, false);
                self.span(task, dst, len, true);
            }
            MemEvent::Fill { dst, len } => {
                self.span(task, dst, len, true);
            }
        }
    }

    /// Touch every line and page in `[addr, addr+len)`.
    fn span(&mut self, task: usize, addr: u64, len: u64, write: bool) {
        if len == 0 {
            return;
        }
        let line = self.cfg.l1.line.max(1);
        let first = addr / line;
        let last = addr.saturating_add(len - 1) / line;
        let count = (last - first + 1).min(self.max_lines_per_event);
        for li in first..first + count {
            self.touch_line(task, li, write);
        }
        let page = self.cfg.page_size.max(1);
        let (pf, pl) = (addr / page, addr.saturating_add(len - 1) / page);
        for pi in pf..=pl {
            self.state.pages.insert(pi);
        }
        // The shared-state tracker (8-byte words): a write records the writer; any touch by a
        // task other than the last writer marks the word contested.
        let (wf, wl) = (addr / 8, addr.saturating_add(len - 1) / 8);
        let wcount = (wl - wf + 1).min(self.max_lines_per_event);
        for wi in wf..wf + wcount {
            match self.state.words.get_mut(&wi) {
                Some((writer, contested)) => {
                    if *writer != task {
                        *contested = true;
                    }
                    if write {
                        *writer = task;
                    }
                }
                None if write => {
                    self.state.words.insert(wi, (task, false));
                }
                None => {}
            }
        }
    }

    /// One line access by `task`: L1 lookup (MESI upgrade/invalidation on write), L2 on a miss.
    fn touch_line(&mut self, task: usize, line_idx: u64, write: bool) {
        let st = &mut self.state;
        st.tick += 1;
        let tick = st.tick;
        while st.l1s.len() <= task {
            let cfg = self.cfg.l1;
            st.l1s.push(SetArray::new(cfg));
        }
        if let Some(way) = st.l1s[task].find(line_idx) {
            st.l1_hits += 1;
            let set_base = (line_idx % self.cfg.l1.sets) as usize * self.cfg.l1.ways;
            st.l1s[task].slots[set_base + way].lru = tick;
            if write {
                st.l1s[task].slots[set_base + way].state = Mesi::Modified;
                invalidate_others(&mut st.l1s, task, line_idx);
            }
            return;
        }
        st.l1_misses += 1;
        // The shared L2: a plain lookup/fill (validity only).
        if st.l2.find(line_idx).is_some() {
            st.l2_hits += 1;
            let set_base = (line_idx % self.cfg.l2.sets) as usize * self.cfg.l2.ways;
            if let Some(way) = st.l2.find(line_idx) {
                st.l2.slots[set_base + way].lru = tick;
            }
        } else {
            st.l2_misses += 1;
            st.l2.fill(line_idx, Mesi::Exclusive, tick);
        }
        // Fill the L1: Modified on a write (invalidating other copies), else Shared when another
        // L1 holds the line, Exclusive otherwise.
        let others_hold = st
            .l1s
            .iter()
            .enumerate()
            .any(|(t, l1)| t != task && l1.find(line_idx).is_some());
        let new_state = if write {
            invalidate_others(&mut st.l1s, task, line_idx);
            Mesi::Modified
        } else if others_hold {
            // Downgrade the other holders to Shared (an M/E copy is now shared).
            for (t, l1) in st.l1s.iter_mut().enumerate() {
                if t != task {
                    if let Some(w) = l1.find(line_idx) {
                        let sb = (line_idx % l1.cfg.sets) as usize * l1.cfg.ways;
                        l1.slots[sb + w].state = Mesi::Shared;
                    }
                }
            }
            Mesi::Shared
        } else {
            Mesi::Exclusive
        };
        st.l1s[task].fill(line_idx, new_state, tick);
    }

    /// Counters + the full line-state grid, as the `memModelStats` reply body.
    pub fn stats_json(&self) -> Json {
        let st = &self.state;
        let grid = |arr: &SetArray| -> Json {
            let mut sets = Vec::new();
            for s in 0..arr.cfg.sets as usize {
                let mut ways = Vec::new();
                for w in 0..arr.cfg.ways {
                    let slot = arr.slots[s * arr.cfg.ways + w];
                    ways.push(Json::obj(vec![
                        (
                            "line",
                            slot.tag.map(|t| Json::i(t as i64)).unwrap_or(Json::Null),
                        ),
                        ("state", Json::s(slot.state.letter())),
                        ("lru", Json::i(slot.lru as i64)),
                    ]));
                }
                sets.push(Json::Arr(ways));
            }
            Json::Arr(sets)
        };
        Json::obj(vec![
            (
                "l1",
                Json::obj(vec![
                    ("hits", Json::i(st.l1_hits as i64)),
                    ("misses", Json::i(st.l1_misses as i64)),
                ]),
            ),
            (
                "l2",
                Json::obj(vec![
                    ("hits", Json::i(st.l2_hits as i64)),
                    ("misses", Json::i(st.l2_misses as i64)),
                ]),
            ),
            ("pageFaults", Json::i(st.pages.len() as i64)),
            (
                "sharedState",
                Json::obj(vec![
                    (
                        "contested",
                        Json::Arr(
                            st.words
                                .iter()
                                .filter(|(_, (_, c))| *c)
                                .map(|(w, (writer, _))| {
                                    Json::obj(vec![
                                        ("addr", Json::i((w * 8) as i64)),
                                        ("lastWriter", Json::i(*writer as i64)),
                                    ])
                                })
                                .collect(),
                        ),
                    ),
                    ("trackedWords", Json::i(st.words.len() as i64)),
                ]),
            ),
            (
                "l1Grids",
                Json::Arr(st.l1s.iter().map(&grid).collect::<Vec<_>>()),
            ),
            ("l2Grid", grid(&st.l2)),
        ])
    }

    /// `(l1_hits, l1_misses, l2_hits, l2_misses, page_faults)` — the counter tuple tests compare.
    pub fn counters(&self) -> (u64, u64, u64, u64, u64) {
        let st = &self.state;
        (
            st.l1_hits,
            st.l1_misses,
            st.l2_hits,
            st.l2_misses,
            st.pages.len() as u64,
        )
    }
}

/// Invalidate every other task's copy of `line_idx` (a write by `task` — MESI invalidation).
fn invalidate_others(l1s: &mut [SetArray], task: usize, line_idx: u64) {
    for (t, l1) in l1s.iter_mut().enumerate() {
        if t != task {
            if let Some(w) = l1.find(line_idx) {
                let sb = (line_idx % l1.cfg.sets) as usize * l1.cfg.ways;
                l1.slots[sb + w].state = Mesi::Invalid;
            }
        }
    }
}
