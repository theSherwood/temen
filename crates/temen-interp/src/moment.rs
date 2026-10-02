//! **Moments and ladders** — capture a run at a boundary, hold a ring of captures, scrub between them
//! (#1454, #1460).
//!
//! Three time-travel routes grew up here separately: the tree-walk `Inspector`'s seek checkpoints, the
//! bytecode engine's `ScheduledDebugRun` snapshots behind the DAP backend, and the reactor
//! moments behind the playground's scrub bar. Each captured the same two things — a window image and
//! the host's run-mutable substate — plus its own idea of a continuation, and each kept its own sorted
//! ladder of them with its own stride, dedupe, nearest-at-or-before and drop-past logic. Four copies
//! of one data structure is four things to keep correct (INVARIANTS #15), and the two that had a
//! memory bound disagreed about it.
//!
//! This module is the one shape. [`Moment`] is the shared halves plus a [`Continuation`] the engine
//! chooses; [`Ladder`] is the sorted, stride-gated, bounded ring of them; every driver keys it on its
//! own monotonic coordinate (an op clock, a scheduler turn, a frame tick) and never has to say which.
//! It is deliberately **driver-light**: it owns no reactor and runs no tick, so it lands before the run
//! loops it serves converge (#1414) rather than waiting on them.
//!
//! The continuation was a type parameter here until #1517 slice 5, one instantiation per engine, so
//! that collapsing the three into one restore path could wait on collapsing the engines' run loops
//! (#1460's second half). With `DebugRun` folded into `ScheduledDebugRun` (slice 4), the three shapes
//! are now the three values of one [`Continuation`] enum on one moment type: `None` (a reactor, which
//! resumes nothing between frames), `Bytecode` (the scheduled debug engine's task set + fibers + child
//! envs), and `ShadowStack` (the tree-walk oracle's call stack + fuel). One `Moment`, one `Ladder`, one
//! serialization ([`temen-snapshot`] writes a `ShadowStack` moment as a §12 artifact).

use std::borrow::Cow;
use std::sync::Arc;

use crate::{Host, HostReplaySubstate, MemLayout, PageMap, RegionTable};

/// The engine's half of a [`Moment`] — whatever it needs to resume execution from this boundary. The
/// shared halves (window image + host substate) live on the `Moment`; this is the part that differs by
/// engine, one variant each (#1517 slice 5, closing #1460).
///
/// The variants are opaque payloads: this module holds and hands them back, and each engine matches the
/// one it captured (a `Ladder` a given driver owns only ever holds that driver's variant). The tree-walk
/// oracle's [`ShadowStack`] stays a distinct implementation on purpose — it is the differential the
/// bytecode engine is checked against (INVARIANTS #15).
pub enum Continuation {
    /// A reactor: its `tick` returns to the host every frame, so at a frame boundary there is no guest
    /// stack, shadow stack or handle table to serialize — the moment is a memcpy and no more (which is
    /// why it works identically on every tier, without the `temen-durable` instrumentation an
    /// arbitrary-safepoint freeze needs, DURABILITY.md §2).
    None,
    /// The **scheduled bytecode debug engine**'s continuation (the sole bytecode debug engine since
    /// slice 4): every task's `Vm`, the run-shared fiber registry, and any §14 child environments.
    Bytecode(crate::bytecode::ScheduledContinuation),
    /// The **tree-walk oracle**'s continuation: the sole vCPU's call stack and fuel.
    ShadowStack(ShadowStack),
}

impl Continuation {
    /// The [`ShadowStack`] this holds, or `None` for another variant — for the tree-walker's restore.
    pub fn as_shadow_stack(&self) -> Option<&ShadowStack> {
        match self {
            Continuation::ShadowStack(s) => Some(s),
            _ => None,
        }
    }

    /// The [`ScheduledContinuation`](crate::bytecode::ScheduledContinuation) this holds, or `None` for
    /// another variant — for the bytecode engine's restore.
    pub fn as_bytecode(&self) -> Option<&crate::bytecode::ScheduledContinuation> {
        match self {
            Continuation::Bytecode(c) => Some(c),
            _ => None,
        }
    }
}

/// The tree-walk oracle's half of a single-threaded time-travel **checkpoint** (W1): the sole vCPU's
/// call stack and fuel, so [`Inspector::seek`](crate::Inspector) can restart a replay at the
/// checkpoint's clock rather than from clock 0. Captured only for the root-only / non-fiber /
/// non-durable / simple-memory subset (`VCpu::checkpointable`), where these plus the window image fully
/// determine the continuation. The window image and the host substate are the [`Moment`]'s own halves,
/// shared with every other engine's checkpoint; the clock is the ladder's key, not the checkpoint's.
///
/// This is [`temen-snapshot`]'s natural §12 artifact in all but name (slice 5): a quiesced tree-walk
/// domain *is* a shadow-stack moment, so the crate serializes one directly.
pub struct ShadowStack {
    pub(crate) frames: Vec<crate::Frame>,
    pub(crate) fuel: u64,
}

impl ShadowStack {
    /// Build a shadow-stack continuation from the root vCPU's `frames` and `fuel`.
    pub(crate) fn new(frames: Vec<crate::Frame>, fuel: u64) -> ShadowStack {
        ShadowStack { frames, fuel }
    }

    /// The captured call stack.
    pub(crate) fn frames(&self) -> &[crate::Frame] {
        &self.frames
    }

    /// The captured fuel.
    pub(crate) fn fuel(&self) -> u64 {
        self.fuel
    }
}

/// A **moment** of a run: the window image, the host's run-mutable substate, and a [`Continuation`].
///
/// The first two are the same for every engine and are captured and restored here, once. `mem` is
/// `None` for a memoryless run. `host` is [`Host::replay_substate`] — streams, the deterministic clock,
/// the cap-tape cursor, serve state, growth accounting, and each named capability's own declared state
/// (#1455) — the identical set a §12 freeze writes into an artifact, so a moment and a save-state
/// cannot disagree about what "the host's state" is (INVARIANTS #13).
///
/// A debugger checkpoint ([`checkpoint`](Self::checkpoint)) also carries the §13 regions the host
/// holds (#2026): each backing's bytes, held in pages like the window image, and the
/// [`RegionTable`] naming them.
pub struct Moment {
    mem: Option<Image>,
    host: HostReplaySubstate,
    /// `None` for a host holding no region, and for a reactor moment, whose restore lands on the
    /// live host and keeps its regions.
    regions: Option<(Vec<Image>, RegionTable)>,
    continuation: Continuation,
}

/// The unit a [`Ladder`] shares window bytes in: the §12 codec's page.
const SHARED_PAGE: usize = crate::DURABLE_SNAPSHOT_PAGE as usize;

/// A moment's window image: flat as captured, or paged once a [`Ladder`] holds it (#1459).
///
/// A ladder compares each page of a rung it takes with the same page of the rung below, and the two
/// share one copy where they agree. So a run that writes a few pages between rungs holds a few pages
/// per rung, not a window. Every rung still owns its whole image (the sharing is reference counted),
/// so a rung restores and evicts on its own and no rung is a delta another one depends on.
enum Image {
    Flat(MemLayout),
    Paged {
        /// The image in [`SHARED_PAGE`] pages; the last one holds the remainder.
        pages: Vec<Arc<[u8]>>,
        map: PageMap,
    },
}

impl Image {
    /// The image as a [`MemLayout`]: borrowed if flat, assembled from its pages if paged.
    fn layout(&self) -> Cow<'_, MemLayout> {
        match self {
            Image::Flat(l) => Cow::Borrowed(l),
            Image::Paged { pages, map } => Cow::Owned(MemLayout {
                bytes: pages.concat(),
                map: map.clone(),
            }),
        }
    }

    fn byte_len(&self) -> usize {
        match self {
            Image::Flat(l) => l.byte_len(),
            Image::Paged { pages, .. } => pages.iter().map(|p| p.len()).sum(),
        }
    }

    /// This image in pages, each one shared with `below`'s page at the same index when the bytes
    /// agree. An image already paged is left as it is.
    fn paged(self, below: Option<&Image>) -> Image {
        let Image::Flat(MemLayout { bytes, map }) = self else {
            return self;
        };
        let below: &[Arc<[u8]>] = match below {
            Some(Image::Paged { pages, .. }) => pages,
            _ => &[],
        };
        let pages = bytes
            .chunks(SHARED_PAGE)
            .enumerate()
            .map(|(i, page)| match below.get(i) {
                Some(b) if **b == *page => Arc::clone(b),
                _ => Arc::from(page),
            })
            .collect();
        Image::Paged { pages, map }
    }

    /// The pages this image holds that `below` does not share — what holding it adds to a ladder.
    fn bytes_beyond(&self, below: Option<&Image>) -> usize {
        let Image::Paged { pages, .. } = self else {
            return self.byte_len();
        };
        let below: &[Arc<[u8]>] = match below {
            Some(Image::Paged { pages, .. }) => pages,
            _ => &[],
        };
        pages
            .iter()
            .enumerate()
            .filter(|(i, p)| !below.get(*i).is_some_and(|b| Arc::ptr_eq(b, p)))
            .map(|(_, p)| p.len())
            .sum()
    }
}

impl Moment {
    /// Capture `mem` and `host`'s substate together with `continuation`, at one instant so the three
    /// halves describe the same one.
    pub fn new(mem: Option<MemLayout>, host: &Host, continuation: Continuation) -> Moment {
        Moment {
            mem: mem.map(Image::Flat),
            host: host.replay_substate(),
            regions: None,
            continuation,
        }
    }

    /// A debugger checkpoint: [`new`](Self::new), plus the §13 regions `host` holds, which its
    /// [`restore_host`](Self::restore_host) rebuilds on the host a seek starts from (#2026). `None`
    /// when a region cannot ride ([`Host::capture_regions`]).
    pub fn checkpoint(
        mem: Option<MemLayout>,
        host: &Host,
        continuation: Continuation,
    ) -> Option<Moment> {
        let regions = match host.region_count() {
            0 => None,
            _ => {
                let (bytes, table) = host.capture_regions()?;
                let images = bytes
                    .into_iter()
                    .map(|b| Image::Flat(MemLayout::image(b)))
                    .collect();
                Some((images, table))
            }
        };
        Some(Moment {
            regions,
            ..Moment::new(mem, host, continuation)
        })
    }

    /// The window image, for the engine to seed back into whatever holds its window. A moment a
    /// [`Ladder`] holds assembles it from the pages it shares with its neighbours.
    pub fn mem(&self) -> Option<Cow<'_, MemLayout>> {
        self.mem.as_ref().map(Image::layout)
    }

    /// The engine's half.
    pub fn continuation(&self) -> &Continuation {
        &self.continuation
    }

    /// Put the host half back — the other side of [`new`](Self::new) — and a checkpoint's regions.
    /// Before the window: its aliases are re-mapped onto the regions this rebuilds.
    pub fn restore_host(&self, host: &mut Host) {
        host.restore_replay_substate(&self.host);
        if let Some((images, table)) = &self.regions {
            let backings: Vec<_> = images
                .iter()
                .map(|i| host.rebuild_region(&i.layout().bytes))
                .collect();
            host.restore_regions(&backings, table);
        }
    }

    /// The window image's and the regions' byte length (the other halves are a handful of words, or
    /// the engine's frames). On its own this is what holding the moment costs; in a [`Ladder`] the
    /// pages it shares with its neighbours are counted once ([`Ladder::held_bytes`]).
    pub fn byte_len(&self) -> usize {
        self.images().map(Image::byte_len).sum()
    }

    /// The images this moment holds: the window's, then each region's.
    fn images(&self) -> impl Iterator<Item = &Image> {
        self.mem
            .iter()
            .chain(self.regions.iter().flat_map(|(r, _)| r))
    }

    /// The region images, or none.
    fn region_images(&self) -> &[Image] {
        self.regions.as_ref().map_or(&[], |(r, _)| r)
    }

    /// This moment with each image in pages, sharing every page that equals `below`'s at the same
    /// place: the window's with the window's, a region's with the region's at the same index.
    fn paged(self, below: Option<&Moment>) -> Moment {
        let mem = self
            .mem
            .map(|m| m.paged(below.and_then(|b| b.mem.as_ref())));
        let regions = self.regions.map(|(images, table)| {
            let under = below.map_or(&[][..], Moment::region_images);
            let images = images
                .into_iter()
                .enumerate()
                .map(|(i, im)| im.paged(under.get(i)))
                .collect();
            (images, table)
        });
        Moment {
            mem,
            regions,
            ..self
        }
    }

    /// The pages this moment holds that `below` does not share — what holding it adds to a ladder.
    fn bytes_beyond(&self, below: Option<&Moment>) -> usize {
        let under = below.map_or(&[][..], Moment::region_images);
        self.mem
            .as_ref()
            .map_or(0, |m| m.bytes_beyond(below.and_then(|b| b.mem.as_ref())))
            + self
                .region_images()
                .iter()
                .enumerate()
                .map(|(i, im)| im.bytes_beyond(under.get(i)))
                .sum::<usize>()
    }

    /// Capture `layout` with `host`'s substate as a reactor moment (continuation [`Continuation::None`]).
    /// A reactor moment always carries a window image (this takes it by value); restoring one gives
    /// **rewind**, a [`Ladder`] of them a keyframe ladder, re-running forward over recorded input the
    /// frames between rungs. A moment restored into a fresh reactor is a save-state; restored twice, a
    /// branch. `None` for a window that aliases a §13 region: a reactor moment carries no region's
    /// bytes, so a restore would show the region as it is then, not as it was (#2051) — a §12
    /// artifact, which carries them, is the save-state for such a window.
    pub fn capture(layout: MemLayout, host: &Host) -> Option<ReactorMoment> {
        (!layout.aliases_regions()).then(|| Moment::new(Some(layout), host, Continuation::None))
    }

    /// The window image. A reactor moment always carries one — it is only ever built by
    /// [`capture`](Self::capture), which takes it by value.
    pub fn layout(&self) -> Cow<'_, MemLayout> {
        self.mem()
            .expect("a reactor moment is built from a window image and always carries it")
    }
}

/// A reactor's moment: a window image and the host substate, its continuation [`Continuation::None`].
/// The name marks intent at reactor call sites; it is the one [`Moment`] type.
pub type ReactorMoment = Moment;

/// A sorted ring of [`Moment`]s keyed on a monotonic coordinate — a keyframe ladder.
///
/// Every time-travel driver needs the same five operations over its captures: is a rung due at this
/// coordinate; take one; find the nearest at or before a target; drop everything past a position
/// (a branch); and keep the set bounded. The coordinate's *meaning* — an op clock, a scheduler turn, a
/// frame tick — is the driver's, and the ladder never has to know it.
///
/// Bounds are a ring size and a byte budget, whichever bites first, and both are optional (`0`). The
/// **lowest rung is pinned** and never evicted, so every position from the first capture onward stays
/// reachable — at worst by replaying the whole tail. A ring that could evict the start would leave a
/// stretch of a scrub track the viewer can see and cannot drag to. Of the rest, eviction drops the rung
/// furthest from the working position, so the ladder stays dense around wherever the driver is —
/// oldest-first would evict a rung just re-taken while scrubbing back.
pub struct Ladder {
    /// Ascending by coordinate; no two share one.
    rungs: Vec<(u64, Moment)>,
    stride: u64,
    ring: usize,
    budget: usize,
}

impl Ladder {
    /// A ladder with a rung due every `stride` coordinates, holding at most `ring` rungs and
    /// `budget` bytes of window image ([`held_bytes`](Self::held_bytes)) — `0` for either bound means
    /// unbounded. `stride` is clamped to at least 1.
    pub fn new(stride: u64, ring: usize, budget: usize) -> Ladder {
        Ladder {
            rungs: Vec::new(),
            stride: stride.max(1),
            ring,
            budget,
        }
    }

    /// Whether a rung is held at exactly `coord`.
    pub fn holds(&self, coord: u64) -> bool {
        self.rungs.binary_search_by_key(&coord, |(c, _)| *c).is_ok()
    }

    /// Whether `coord` is a rung boundary the ladder does not yet hold.
    pub fn is_due(&self, coord: u64) -> bool {
        coord.is_multiple_of(self.stride) && !self.holds(coord)
    }

    /// Whether a driver should **capture** a rung at `coord`: a positive stride boundary not already
    /// held. This is the one admission gate the three time-travel drivers share (#1517 slice 5), in
    /// place of each open-coding `coord > 0 && coord.is_multiple_of(stride)`; the live "is the run still
    /// checkpointable" half stays with the engine, which reads its own state. `coord == 0` is never a
    /// rung — the from-scratch replay start needs no checkpoint to reach.
    pub fn admits(&self, coord: u64) -> bool {
        coord > 0 && self.is_due(coord)
    }

    /// Hold `moment` at `coord`, keeping the ladder sorted and bounded. A coordinate already held is
    /// left as it was: a replay re-crossing a rung it took on the way out captures the same state, so
    /// there is nothing to replace.
    ///
    /// The window image is held in pages, sharing every page that equals the rung below's (#1459), and
    /// so is each region's (#2026).
    /// That is one compare of the window per rung taken, and no cost at all between rungs, on any
    /// tier: the run's writes are never tracked.
    pub fn take(&mut self, coord: u64, moment: Moment) {
        let Err(at) = self.rungs.binary_search_by_key(&coord, |(c, _)| *c) else {
            return;
        };
        let below = at.checked_sub(1).map(|i| &self.rungs[i].1);
        let moment = moment.paged(below);
        self.rungs.insert(at, (coord, moment));
        self.evict(coord);
    }

    /// The nearest rung at or before `coord`, with its coordinate — where a seek to `coord` restarts.
    /// `None` if nothing is held that early.
    pub fn nearest_at_or_before(&self, coord: u64) -> Option<(u64, &Moment)> {
        let n = self.rungs.partition_point(|(c, _)| *c <= coord);
        self.rungs[..n].last().map(|(c, m)| (*c, m))
    }

    /// Drop every rung past `coord` — the recorded future is being abandoned.
    pub fn truncate_after(&mut self, coord: u64) {
        let n = self.rungs.partition_point(|(c, _)| *c <= coord);
        self.rungs.truncate(n);
    }

    /// Drop every rung.
    pub fn clear(&mut self) {
        self.rungs.clear();
    }

    /// How many rungs are held.
    pub fn len(&self) -> usize {
        self.rungs.len()
    }

    /// Whether none are.
    pub fn is_empty(&self) -> bool {
        self.rungs.is_empty()
    }

    /// The coordinates held, ascending.
    pub fn coords(&self) -> Vec<u64> {
        self.rungs.iter().map(|(c, _)| *c).collect()
    }

    /// What the ladder is holding, in bytes: its rungs' window and region images, a page shared by
    /// neighbouring rungs counted once.
    ///
    /// Rungs share a page only with the rung below at the time they were taken, so a shared page is
    /// held by a run of adjacent rungs. A rung taken later between two that share a page breaks that
    /// run, and the page is then counted on both sides of it: an overcount, so a byte budget errs
    /// towards holding less.
    pub fn held_bytes(&self) -> usize {
        let mut below = None;
        let mut held = 0;
        for (_, m) in &self.rungs {
            held += m.bytes_beyond(below);
            below = Some(m);
        }
        held
    }

    /// Bring the ladder back inside both bounds, never below the pinned lowest rung. `here` is the
    /// working position — the coordinate just taken — which the survivors stay dense around.
    fn evict(&mut self, here: u64) {
        while self.rungs.len() > 1
            && ((self.ring > 0 && self.rungs.len() > self.ring)
                || (self.budget > 0 && self.held_bytes() > self.budget))
        {
            let victim = (1..self.rungs.len())
                .max_by_key(|&i| self.rungs[i].0.abs_diff(here))
                .expect("len > 1, so there is a rung after the pinned one");
            self.rungs.remove(victim);
        }
    }
}

/// One frame's worth of driver input, as it rides a [`ReactorTimeline`]'s tape.
///
/// This records what the driver *offered* — the key/pointer event — not the packed queue word a
/// particular reactor encodes it into, so the tape stays independent of that encoding and a replay
/// goes back in through the same front door the live driver used rather than reaching into the queues
/// behind it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReactorInput {
    /// A key transition: `pressed` is 1 (down) / 0 (up), `keycode` the platform key id.
    Key { keycode: i32, pressed: i32 },
    /// A pointer event: `kind` 0 = pointer, 1 = wheel; `payload` per the reactor's mouse encoding.
    Mouse { kind: i32, payload: i32 },
}

/// A reactor a [`ReactorTimeline`] can record: take input, capture and restore a [`ReactorMoment`].
///
/// Deliberately **not** "and run a tick". A wasm-JIT reactor's `tick` is *emitted wasm*, compiled and
/// called by whoever emitted it — a browser page's JS host in production — so for that tier running a
/// tick is not something Rust can do at all. Such a reactor implements this trait and is driven with
/// [`begin_tick`](ReactorTimeline::begin_tick) / [`end_tick`](ReactorTimeline::end_tick), running its
/// own tick in between; a reactor that *can* step itself also implements [`SteppableReactor`] and
/// gets [`frame`](ReactorTimeline::frame) and [`seek`](ReactorTimeline::seek) as well.
///
/// Splitting it this way is what keeps one ladder for both. The alternative — one trait with a `step`
/// the emitted tier answers with a refusal — would make `frame` compile for a reactor that cannot run
/// it and fail at runtime, halfway through a tick it had already opened.
pub trait MomentReactor {
    /// Enqueue a key event for the guest to poll next tick.
    fn push_key(&self, keycode: i32, pressed: i32);
    /// Enqueue a pointer event for the guest to poll next tick.
    fn push_mouse(&self, kind: i32, payload: i32);
    /// Capture this reactor's state at the current frame boundary, or `None` if it cannot be imaged
    /// faithfully (a §13 region alias — refuse rather than hand back a fiction, INVARIANTS #9c).
    fn moment(&self) -> Option<ReactorMoment>;
    /// Put the reactor back at `m`. `false` if it has no window to restore into.
    fn restore(&mut self, m: &ReactorMoment) -> bool;

    /// Hand one recorded input back to the reactor — the replay side of the tape.
    fn feed(&self, ev: ReactorInput) {
        match ev {
            ReactorInput::Key { keycode, pressed } => self.push_key(keycode, pressed),
            ReactorInput::Mouse { kind, payload } => self.push_mouse(kind, payload),
        }
    }
}

/// A [`MomentReactor`] that can also run its own tick — every tier whose `tick` executes inside this
/// process. Implementing it is what unlocks the self-driving [`ReactorTimeline::frame`] and
/// [`ReactorTimeline::seek`]; a reactor whose tick belongs to the embedder simply does not, and the
/// embedder uses the split form instead.
pub trait SteppableReactor: MomentReactor {
    /// Run one tick to the next frame boundary, returning the reactor's own status word.
    fn step(&mut self) -> i32;
}

/// A [`Ladder`] of reactor moments plus a tick-indexed input tape — the two things that turn a rewind
/// into a **scrub** (#1457).
///
/// A [`ReactorMoment`] on its own only goes *back* to a point someone thought to save. A timeline
/// records what the driver fed the guest each tick, so any recorded position can be reconstructed —
/// restore the nearest rung at or before it, then re-feed the tape forward.
///
/// It does not own the reactor. Two reasons, both load-bearing: an embedder keeps its reactors in
/// their own storage (the browser cdylib has one static per tier), and the wasm-JIT tier's tick is run
/// by the embedder rather than by Rust, so the timeline has to be drivable from outside.
///
/// ## Why the tape records the driver, not the guest
///
/// The obvious move is [`Host::record_caps`], which tapes every `HOST_PROC` crossing so a replay can
/// serve them without a live powerbox — the seam the debug checkpoint ladder rides. A reactor needs
/// none of it: it replays against its **live** powerbox — the same capabilities, still granted — and
/// everything those capabilities read from is already inside the moment (the input queues and any
/// `fs` cursors are captured cap state; the window is the image). Given the same starting moment and
/// the same driver input, the guest's crossings *recompute* identically. The only thing outside the
/// moment is what the host injects from the outside world, which is exactly what this tape holds. For
/// Doom, taping the crossings instead would have carried every `display.present` and the `mem_writes`
/// of every `fs` read — roughly the WAD — per keyframe interval.
///
/// So what a tape must hold is decided by whether the powerbox survives the restore: rebuilt (the
/// debug ladder) ⇒ tape the guest's crossings; kept (a reactor) ⇒ tape the host's injections. A
/// reactor granted a genuinely nondeterministic capability — a wall clock, entropy, a socket — would
/// need its crossings taped as well, and the honest move then is a recorded-input predicate on
/// `record_caps`, not a second tape.
///
/// ## Positions
///
/// [`tick`](Self::tick) is where the reactor stands (frames presented so far) and [`len`](Self::len)
/// is how far the recording goes. They are equal at the live end, where [`frame`](Self::frame)
/// *extends* the recording with whatever input has been pushed; when `tick < len` the timeline is
/// parked inside its own history and `frame` *replays* the recorded input for that tick instead.
/// Pushing input while parked in the past abandons the rest of the recording — that is a branch, and
/// the tape and rungs past that point describe a run that will not happen.
pub struct ReactorTimeline {
    /// Frames presented so far — the position on the timeline.
    tick: usize,
    /// `tape[t]` is the input the driver offered for tick `t`, in the order it offered it.
    tape: Vec<Vec<ReactorInput>>,
    /// Input offered for a tick that has not run yet.
    pending: Vec<ReactorInput>,
    /// The keyframes, keyed on tick.
    ladder: Ladder,
}

impl ReactorTimeline {
    /// A timeline over a reactor standing at tick 0.
    ///
    /// `stride` is how often a rung is taken; `ring` how many are held; `budget` a ceiling on their
    /// total bytes. The last one matters because a rung's cost depends on the guest: the first rung
    /// holds the whole window (a few KiB for `bounce`, 16 MiB for Doom) and each later one the pages
    /// the guest changed since the rung below, so a count alone could mean almost nothing or most of a
    /// window per rung. Whichever bound bites first wins, and neither can take the ring
    /// below the pinned tick-0 rung. `stride` and `ring` are clamped to at least 1; a `budget` of 0
    /// means no byte ceiling.
    pub fn new(stride: usize, ring: usize, budget: usize) -> ReactorTimeline {
        ReactorTimeline {
            tick: 0,
            tape: Vec::new(),
            pending: Vec::new(),
            ladder: Ladder::new(stride as u64, ring.max(1), budget),
        }
    }

    /// Where the reactor stands: the number of frames presented, so the next [`frame`](Self::frame)
    /// produces frame `tick`.
    pub fn tick(&self) -> usize {
        self.tick
    }

    /// How far the recording goes — the highest tick this timeline can [`seek`](Self::seek) to.
    pub fn len(&self) -> usize {
        self.tape.len()
    }

    /// Whether nothing has been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.tape.is_empty()
    }

    /// The ticks currently held as keyframes, ascending — the ladder's rungs.
    pub fn keyframe_ticks(&self) -> Vec<usize> {
        self.ladder
            .coords()
            .into_iter()
            .map(|c| c as usize)
            .collect()
    }

    /// What the ladder is holding, in bytes: the sum of its rungs' window images.
    pub fn held_bytes(&self) -> usize {
        self.ladder.held_bytes()
    }

    /// The ticks whose recorded input is non-empty, ascending — where the driver actually did
    /// something, for a scrub track that marks it.
    pub fn taped_ticks(&self) -> Vec<usize> {
        self.tape
            .iter()
            .enumerate()
            .filter(|(_, evs)| !evs.is_empty())
            .map(|(t, _)| t)
            .collect()
    }

    /// Offer a key event for the next tick. Offered while parked in the past, this **branches**: the
    /// recorded future is dropped (see the type's docs).
    pub fn push_key(&mut self, keycode: i32, pressed: i32) {
        self.offer(ReactorInput::Key { keycode, pressed });
    }

    /// Offer a pointer event for the next tick, branching as [`push_key`](Self::push_key) does.
    pub fn push_mouse(&mut self, kind: i32, payload: i32) {
        self.offer(ReactorInput::Mouse { kind, payload });
    }

    fn offer(&mut self, ev: ReactorInput) {
        // New input from inside the recording: from here the run diverges, so the tape beyond this
        // point — and every rung past it — describes a future that will not happen.
        self.truncate();
        self.pending.push(ev);
    }

    /// Abandon the recorded future: drop the tape past this position and every rung beyond it. A
    /// no-op at the live end.
    ///
    /// New input does this on its own, because a run that is steered differently *has* diverged. This
    /// is the same act asked for outright, for a driver whose "play on from here" is a deliberate
    /// choice by the person scrubbing rather than something inferred from the next keypress.
    pub fn truncate(&mut self) {
        if self.tick >= self.tape.len() {
            return;
        }
        self.tape.truncate(self.tick);
        self.ladder.truncate_after(self.tick as u64);
    }

    /// Open the tick at this position: take a rung if one is due, then hand the reactor its input —
    /// either the input pushed since the last frame (at the live end, which also records it) or the
    /// input recorded for this tick (parked in the past).
    ///
    /// The caller then runs the tick and calls [`end_tick`](Self::end_tick). A reactor that can step
    /// itself should use [`frame`](Self::frame), which is these three in order.
    ///
    /// The rung is taken **before** the input reaches the reactor, and that ordering is the whole
    /// reason the tape and the moment compose: the input for tick `t` lives in the tape alone, so a
    /// replay from the rung at `t` feeds it exactly once. Were the rung taken after, that input would
    /// sit in both and every replay would double it.
    pub fn begin_tick<R: MomentReactor + ?Sized>(&mut self, r: &mut R) {
        let coord = self.tick as u64;
        if self.ladder.is_due(coord) {
            // An uncapturable window leaves the ladder as it was and `seek` refuses rather than
            // holding a partial image that would restore a fiction (#9c).
            if let Some(m) = r.moment() {
                self.ladder.take(coord, m);
            }
        }
        if self.tick == self.tape.len() {
            self.tape.push(std::mem::take(&mut self.pending));
        }
        for ev in self.tape[self.tick].clone() {
            r.feed(ev);
        }
    }

    /// Close the tick the caller just ran, advancing the position.
    pub fn end_tick(&mut self) {
        self.tick += 1;
    }

    /// Run one tick on a reactor that can step itself: [`begin_tick`](Self::begin_tick), the
    /// reactor's own `step`, then [`end_tick`](Self::end_tick). Returns the reactor's status word.
    pub fn frame<R: SteppableReactor + ?Sized>(&mut self, r: &mut R) -> i32 {
        self.begin_tick(r);
        let status = r.step();
        self.end_tick();
        status
    }

    /// Reposition to the nearest rung at or before `target`, without replaying the tail — the half of
    /// a [`seek`](Self::seek) that does not need to run ticks, so an embedder that runs its own can
    /// call this and then loop [`begin_tick`](Self::begin_tick) / its tick / [`end_tick`](Self::end_tick)
    /// until [`tick`](Self::tick) reaches `target`.
    ///
    /// `false` for a position past the recording, or if there is no rung at or before `target`, or if
    /// the reactor refuses the restore. Seeking *forward* from where the reactor already stands needs
    /// no restore at all and reports `true` having done nothing.
    pub fn seek_begin<R: MomentReactor + ?Sized>(&mut self, r: &mut R, target: usize) -> bool {
        if target > self.tape.len() {
            return false;
        }
        // Input aimed at the tick being left, exactly as a `restore` drops the queue it sat in.
        self.pending.clear();
        if target >= self.tick {
            return true; // replay forward from here; no rewind needed
        }
        let Some((at, m)) = self.ladder.nearest_at_or_before(target as u64) else {
            return false; // nothing captured at or before `target` — refuse, never guess (#9c)
        };
        if !r.restore(m) {
            return false;
        }
        self.tick = at as usize;
        true
    }

    /// Move to tick `target`, replaying from the nearest rung at or before it. The frames the replay
    /// presents on the way are discarded: this positions the guest, and the caller renders forward
    /// with [`frame`](Self::frame).
    ///
    /// Seeking *forward* inside the recording needs no rewind — the replay continues from where the
    /// reactor already is — so dragging a scrub bar forward costs the frames crossed rather than a
    /// rewind and a full re-run.
    pub fn seek<R: SteppableReactor + ?Sized>(&mut self, r: &mut R, target: usize) -> bool {
        if !self.seek_begin(r, target) {
            return false;
        }
        while self.tick < target {
            self.frame(r);
        }
        true
    }
}

#[cfg(test)]
mod ladder_tests {
    //! The ladder's contract, on its own — no engine, no reactor. Every driver's seek/scrub/rewind
    //! correctness rests on these five operations behaving exactly so, and the engine suites gate
    //! them only through a whole seek; this pins them directly.

    use super::*;

    /// A moment whose window is `pages` 4 KiB pages, so byte bounds are testable. Its continuation is
    /// [`Continuation::None`] — the ladder never reads the continuation (only [`Moment::byte_len`], which
    /// is `mem`-only), so the reactor variant stands in for all three here; [`ladder_is_continuation_agnostic`]
    /// pins that the payload variant does not change the ladder's behaviour.
    fn moment(pages: usize) -> Moment {
        moment_with(pages, Continuation::None)
    }

    fn moment_with(pages: usize, continuation: Continuation) -> Moment {
        image_moment(vec![0; pages * 4096], continuation)
    }

    fn image_moment(bytes: Vec<u8>, continuation: Continuation) -> Moment {
        let n = bytes.len() as u64;
        let layout =
            MemLayout::from_parts(bytes, 4096, n, &[]).expect("a window is a valid layout");
        Moment::new(Some(layout), &Host::new(), continuation)
    }

    /// A one-page moment filled with `fill`, so neighbouring rungs share nothing.
    fn filled(fill: u8) -> Moment {
        image_moment(vec![fill; 4096], Continuation::None)
    }

    /// **One parameterised warm≡cold property over the three continuation variants** (#1517 slice 5).
    /// The [`Ladder`] is the shared spine of all three time-travel drivers; it must treat the
    /// continuation as an opaque payload — take/nearest/evict/pin decided only by coordinate and window
    /// bytes, never by which engine's continuation a rung carries. Running the identical rung sequence
    /// under each variant and asserting the same held coordinates and eviction is what lets the three
    /// engine harnesses (`debug_checkpoints`, `dap_checkpoints`, `native_reactor_timeline`) trust the
    /// spine and only differential their own capture/restore.
    #[test]
    fn ladder_is_continuation_agnostic() {
        // A distinct continuation per variant, all with the same one-page window so byte accounting is
        // identical. `Bytecode` is built from an empty scheduled continuation via its test constructor.
        let variants: [fn() -> Continuation; 3] = [
            || Continuation::None,
            || Continuation::ShadowStack(ShadowStack::new(Vec::new(), 0)),
            || Continuation::Bytecode(crate::bytecode::ScheduledContinuation::empty_for_test()),
        ];
        let mut coords_seen: Option<Vec<u64>> = None;
        for make in variants {
            // ring of 3, so eviction (drop furthest from the working position, pin the lowest) fires.
            let mut l = Ladder::new(1, 3, 0);
            for c in [0u64, 1, 2, 3, 4] {
                l.take(c, moment_with(1, make()));
            }
            let coords = l.coords();
            match &coords_seen {
                None => coords_seen = Some(coords),
                Some(prev) => assert_eq!(
                    *prev, coords,
                    "the ladder's held set must not depend on the continuation variant"
                ),
            }
        }
    }

    #[test]
    fn rungs_stay_sorted_and_a_coordinate_is_held_once() {
        let mut l: Ladder = Ladder::new(4, 0, 0);
        for c in [8u64, 0, 16, 4, 8] {
            l.take(c, moment(1));
        }
        assert_eq!(
            l.coords(),
            vec![0, 4, 8, 16],
            "sorted on insert, duplicates ignored"
        );
        assert!(l.holds(8) && !l.holds(12));
        assert!(
            l.is_due(12) && !l.is_due(8) && !l.is_due(13),
            "due = on-stride and not held"
        );
    }

    #[test]
    fn nearest_at_or_before_is_where_a_seek_restarts() {
        let mut l: Ladder = Ladder::new(1, 0, 0);
        assert!(
            l.nearest_at_or_before(5).is_none(),
            "an empty ladder has nowhere to restart"
        );
        for c in [4u64, 8, 12] {
            l.take(c, moment(1));
        }
        assert_eq!(
            l.nearest_at_or_before(3).map(|(c, _)| c),
            None,
            "nothing that early"
        );
        assert_eq!(
            l.nearest_at_or_before(4).map(|(c, _)| c),
            Some(4),
            "exact hit"
        );
        assert_eq!(
            l.nearest_at_or_before(11).map(|(c, _)| c),
            Some(8),
            "between rungs"
        );
        assert_eq!(
            l.nearest_at_or_before(100).map(|(c, _)| c),
            Some(12),
            "past the end"
        );
    }

    #[test]
    fn truncate_after_drops_the_abandoned_future_only() {
        let mut l: Ladder = Ladder::new(1, 0, 0);
        for c in [0u64, 4, 8, 12] {
            l.take(c, moment(1));
        }
        l.truncate_after(8);
        assert_eq!(
            l.coords(),
            vec![0, 4, 8],
            "a rung at the position itself survives"
        );
        l.truncate_after(100);
        assert_eq!(l.len(), 3, "a no-op past the end");
    }

    #[test]
    fn the_ring_pins_the_lowest_rung_and_drops_the_one_furthest_from_here() {
        let mut l: Ladder = Ladder::new(1, 3, 0);
        for c in [0u64, 10, 20, 30] {
            l.take(c, moment(1));
        }
        // Taking 30 with here=30: of {10, 20, 30}, 10 is furthest from 30 and goes; 0 is pinned.
        assert_eq!(l.coords(), vec![0, 20, 30]);
        // Scrubbing back: re-take 10 with here=10 — now 30 is furthest and goes, not the rung just
        // taken (oldest-first would have evicted 10 immediately).
        l.take(10, moment(1));
        assert_eq!(l.coords(), vec![0, 10, 20]);
        assert!(
            l.holds(0),
            "the start of the run stays reachable however long it goes on"
        );
    }

    #[test]
    fn the_byte_budget_binds_before_the_count_does() {
        // Room for two one-page rungs against a count that would allow sixteen.
        let mut l: Ladder = Ladder::new(1, 16, 2 * 4096 + 100);
        for c in [0u64, 1, 2, 3] {
            l.take(c, filled(c as u8 + 1));
        }
        assert_eq!(l.len(), 2, "two rungs fit the budget: {:?}", l.coords());
        assert!(l.held_bytes() <= 2 * 4096 + 100);
        assert!(l.holds(0), "and the pin survives a budget that tight");
    }

    #[test]
    fn rungs_share_the_pages_they_agree_on() {
        // Four pages; each rung after the first rewrites one of them.
        let mut image = vec![0u8; 4 * 4096];
        let mut l: Ladder = Ladder::new(1, 0, 0);
        l.take(0, image_moment(image.clone(), Continuation::None));
        assert_eq!(
            l.held_bytes(),
            4 * 4096,
            "the first rung holds its whole window"
        );
        image[2 * 4096 + 7] = 1;
        l.take(1, image_moment(image.clone(), Continuation::None));
        image[4096] = 2;
        l.take(2, image_moment(image.clone(), Continuation::None));
        assert_eq!(
            l.held_bytes(),
            6 * 4096,
            "each later rung adds only the page that changed"
        );
        // A rung with nothing new adds nothing.
        l.take(3, image_moment(image.clone(), Continuation::None));
        assert_eq!(l.held_bytes(), 6 * 4096);
        // Dropping a rung in the middle keeps every other rung's image whole.
        let mut l2: Ladder = Ladder::new(1, 3, 0);
        let mut img = vec![0u8; 4 * 4096];
        let mut want = Vec::new();
        for c in 0..4u64 {
            img[(c as usize % 4) * 4096] = c as u8 + 1;
            want.push(img.clone());
            l2.take(c, image_moment(img.clone(), Continuation::None));
        }
        assert_eq!(l2.coords(), vec![0, 2, 3], "1 is furthest from 3 and goes");
        for c in [0u64, 2, 3] {
            let (_, m) = l2.nearest_at_or_before(c).unwrap();
            assert_eq!(m.mem().unwrap().bytes(), &want[c as usize][..]);
        }
        assert_eq!(
            l2.held_bytes(),
            7 * 4096,
            "4, then 2 for rung 2 (its own page and the one 1 changed, now shared with nothing \
             below), then 1 for rung 3"
        );
    }

    /// The page sharing is invisible to a restore: under a random workload of writes, takes at any
    /// coordinate (in order, out of order and repeated), truncation and both bounds, every rung a
    /// ladder holds gives back exactly the image it was taken with, and the ladder never holds more
    /// than the images would flat.
    #[test]
    fn a_held_rung_restores_the_image_it_was_taken_with() {
        let mut seed = 0x1459_u64;
        let mut rand = move |n: u64| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 33) % n
        };
        for case in 0..64 {
            let pages = 1 + rand(6) as usize;
            let tail = rand(4096) as usize; // a window need not be whole pages
            let mut image = vec![0u8; pages * 4096 + tail];
            let budget = if case % 3 == 0 { image.len() * 3 } else { 0 };
            let mut l = Ladder::new(1, rand(8) as usize, budget);
            let mut taken = std::collections::BTreeMap::new();
            for _ in 0..200 {
                match rand(10) {
                    0..=5 => {
                        // A few writes, of a byte or a run, anywhere in the window.
                        for _ in 0..1 + rand(4) {
                            let at = rand(image.len() as u64) as usize;
                            let len = (1 + rand(300) as usize).min(image.len() - at);
                            let v = rand(256) as u8;
                            image[at..at + len].fill(v);
                        }
                    }
                    6..=8 => {
                        let c = rand(64);
                        if !l.holds(c) {
                            taken.insert(c, image.clone());
                        }
                        l.take(c, image_moment(image.clone(), Continuation::None));
                    }
                    _ => {
                        let c = rand(64);
                        l.truncate_after(c);
                        taken.retain(|&k, _| k <= c);
                    }
                }
                let flat: usize = l.coords().len() * image.len();
                assert!(l.held_bytes() <= flat, "never more than the flat images");
                for c in l.coords() {
                    let (at, m) = l.nearest_at_or_before(c).unwrap();
                    assert_eq!(at, c);
                    assert_eq!(
                        m.mem().unwrap().bytes(),
                        &taken[&c][..],
                        "case {case}: rung {c} restores what it was taken with"
                    );
                }
            }
        }
    }

    #[test]
    fn zero_bounds_mean_unbounded() {
        let mut l: Ladder = Ladder::new(1, 0, 0);
        for c in 0..64u64 {
            l.take(c, moment(1));
        }
        assert_eq!(
            l.len(),
            64,
            "the debug ladders' shape: nothing is ever evicted"
        );
    }

    /// **A checkpoint carries the regions its host holds** (#2026): the ladder holds their bytes in
    /// pages, sharing with the rung below what did not change, and counts them in
    /// [`Ladder::held_bytes`]; a restore rebuilds each region on a fresh host under the same id, its
    /// handle at the same value. A region something else also holds does not ride.
    #[test]
    fn a_checkpoint_carries_its_regions_paged_and_counted() {
        let mut host = Host::new();
        let handle = host.grant_shared_region(4 * SHARED_PAGE);
        let take = |host: &Host| {
            Moment::checkpoint(None, host, Continuation::None)
                .expect("a region its host alone holds rides")
        };
        let mut ladder = Ladder::new(1, 0, 0);
        ladder.take(1, take(&host));
        assert_eq!(
            ladder.held_bytes(),
            4 * SHARED_PAGE,
            "the region's bytes are held"
        );
        host.regions[0].write_byte(SHARED_PAGE as u64, 7);
        ladder.take(2, take(&host));
        assert_eq!(
            ladder.held_bytes(),
            5 * SHARED_PAGE,
            "the second rung holds only the page that changed"
        );

        let mut fresh = Host::new();
        let (_, rung) = ladder.nearest_at_or_before(2).expect("held");
        rung.restore_host(&mut fresh);
        let region = fresh
            .resolve_region(handle)
            .expect("the handle names the region again");
        assert_eq!(
            region.read_byte(SHARED_PAGE as u64),
            7,
            "with the bytes it had"
        );
        assert!(Arc::ptr_eq(&region, &fresh.regions[0]), "under the same id");

        let mut shared = Host::new();
        shared.grant_shared_region_backed(Arc::clone(&host.regions[0]));
        assert!(
            Moment::checkpoint(None, &shared, Continuation::None).is_none(),
            "a region another holder can write declines"
        );
    }
}
