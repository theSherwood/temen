//! **Reactor moments and timelines** — capture a reactor at a frame boundary, restore it, and scrub
//! (#1454's first cell; #1457).
//!
//! A reactor's `tick` returns to the host every frame, so between frames there is no guest stack, no
//! shadow stack and no handle table: a *moment* is one window image plus a few host-side words. That
//! is why this costs a memcpy rather than the `temen-durable` instrumentation an arbitrary-safepoint
//! freeze needs (DURABILITY.md §2), and why it works identically on every tier.
//!
//! This lives beside [`MemLayout`] and [`Host`] rather than in an embedder because **one** ladder
//! serves every reactor there is — the engine-backed interpreter reactor, one over a caller-owned
//! region, the wasm-JIT reactor whose tick the embedder runs, and a native driver. A ladder that an
//! embedder could not reach would be answered by that embedder growing its own, which is how a
//! behaviour ends up with two implementations that drift (INVARIANTS #15).

use crate::{Host, MemLayout};

/// A **moment** of a reactor: everything needed to put the guest back exactly where it was at a frame
/// boundary — the window image (bytes + page-protection map) and each capability's own declared state.
///
/// The capability half is exactly what [`Host::capture_cap_states`] yields and
/// [`Host::restore_cap_states`] takes back — the same bytes a §12 freeze writes into the artifact's
/// named-capability section, so an in-session rewind and a save-state cannot drift apart (#1455).
///
/// Restoring a moment gives **rewind**; keeping several gives a keyframe ladder; re-running the guest
/// forward over recorded input gives the frames between two rungs. A moment restored into a fresh
/// reactor is a save-state; restored twice, a branch.
pub struct ReactorMoment {
    layout: MemLayout,
    /// Each host capability's own state, positional over the host's capability table.
    caps: Vec<Option<Vec<u8>>>,
}

impl ReactorMoment {
    /// Capture `layout` together with `host`'s capability state — the two halves of a moment, taken
    /// at one instant so they describe the same one.
    pub fn capture(layout: MemLayout, host: &Host) -> ReactorMoment {
        ReactorMoment {
            layout,
            caps: host.capture_cap_states(),
        }
    }

    /// The window image, for the reactor to seed back into whatever holds its window.
    pub fn layout(&self) -> &MemLayout {
        &self.layout
    }

    /// Put the capability half back into `host` — the other side of [`capture`](Self::capture).
    pub fn restore_caps(&self, host: &mut Host) {
        host.restore_cap_states(&self.caps);
    }

    /// The window image's byte length — what holding this moment costs (the capability half is a
    /// handful of words). A ladder sizes its ring against this.
    pub fn byte_len(&self) -> usize {
        self.layout.byte_len()
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

/// The two things that turn a rewind into a **scrub**: a tick-indexed input tape and a keyframe
/// ladder (#1457 items 3–4).
///
/// A [`ReactorMoment`] on its own only goes *back* to a point someone thought to save. A timeline
/// records what the driver fed the guest each tick, so any recorded position can be reconstructed —
/// restore the nearest rung at or before it, then re-feed the tape forward. Keyframe stride trades
/// memory for seek latency; the tape is a few words per tick either way.
///
/// It does not own the reactor. Two reasons, both load-bearing: an embedder keeps its reactors in
/// their own storage (the browser cdylib has one static per tier), and the wasm-JIT tier's tick is run
/// by the embedder rather than by Rust, so the timeline has to be drivable from outside. One ladder
/// covers all of it, parameterized by stride, ring size, byte budget and who runs the tick, rather
/// than each driver growing its own (INVARIANTS #15).
///
/// ## Why the tape records the driver, not the guest
///
/// The obvious move is [`Host::record_caps`], which tapes every `HOST_PROC` crossing so a replay can
/// serve them without a live powerbox — the seam the debug checkpoint ladder rides. A reactor needs
/// none of it, and it is worth saying why, because the cost of getting this wrong is large: for Doom
/// that tape would carry every `display.present` and the `mem_writes` of every `fs` read, which is
/// approximately the whole WAD, per keyframe interval.
///
/// The reason it is unnecessary is that a reactor replays against its **live** powerbox — the same
/// capabilities, still granted — and everything those capabilities read from is already inside the
/// moment: the input queues and any `fs` cursors are captured cap state, and the window is the image.
/// So given the same starting moment and the same driver input, the guest's crossings *recompute*
/// identically instead of needing to be served from a recording. The only thing outside the moment is
/// what the host injects from the outside world, which is exactly what this tape holds.
///
/// That is also the boundary of the claim: a reactor granted a genuinely nondeterministic capability
/// — a wall clock, entropy, a socket — would need its crossings taped as well, and the honest move
/// then is the recorded-input predicate on `record_caps` that #1457 sketched, not a second tape.
///
/// ## Positions
///
/// [`tick`](Self::tick) is where the reactor stands (frames presented so far) and [`len`](Self::len)
/// is how far the recording goes. They are equal at the live end, where [`frame`](Self::frame)
/// *extends* the recording with whatever input has been pushed; when `tick < len` the timeline is
/// parked inside its own history and `frame` *replays* the recorded input for that tick instead.
/// Pushing input while parked in the past abandons the rest of the recording — that is a branch, and
/// the tape and keyframes past that point describe a run that will not happen.
pub struct ReactorTimeline {
    /// Frames presented so far — the position on the timeline.
    tick: usize,
    /// `tape[t]` is the input the driver offered for tick `t`, in the order it offered it.
    tape: Vec<Vec<ReactorInput>>,
    /// Input offered for a tick that has not run yet.
    pending: Vec<ReactorInput>,
    /// Keyframes, ascending by tick. `keyframes[0]` is tick 0 and is never evicted, so **every**
    /// recorded position stays reachable — at worst by replaying the whole tape. A ring that could
    /// evict it would leave the start of a run unreachable, which for a scrub bar means a region of
    /// the track the viewer can see and cannot drag to.
    keyframes: Vec<(usize, ReactorMoment)>,
    stride: usize,
    ring: usize,
    budget: usize,
}

impl ReactorTimeline {
    /// A timeline over a reactor standing at tick 0.
    ///
    /// `stride` is how often a rung is taken; `ring` how many are held; `budget` a ceiling on their
    /// total bytes. The last one matters because a rung's cost is the guest's window: a `bounce` rung
    /// is a few KiB and a Doom one is 16 MiB, so a count alone would mean 8 rungs is either nothing or
    /// 128 MiB depending on the guest. Whichever bound bites first wins, and neither can take the ring
    /// below the pinned tick-0 rung. `stride` and `ring` are clamped to at least 1; a `budget` of 0
    /// means no byte ceiling.
    pub fn new(stride: usize, ring: usize, budget: usize) -> ReactorTimeline {
        ReactorTimeline {
            tick: 0,
            tape: Vec::new(),
            pending: Vec::new(),
            keyframes: Vec::new(),
            stride: stride.max(1),
            ring: ring.max(1),
            budget,
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
        self.keyframes.iter().map(|(t, _)| *t).collect()
    }

    /// What the ladder is holding, in bytes: the sum of its rungs' window images.
    pub fn held_bytes(&self) -> usize {
        self.keyframes.iter().map(|(_, m)| m.byte_len()).sum()
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
        let here = self.tick;
        self.keyframes.retain(|(t, _)| *t <= here);
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
        self.keyframe_here(r);
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
        let Some(i) = self.keyframes.iter().rposition(|(t, _)| *t <= target) else {
            return false; // nothing captured at or before `target` — refuse, never guess (#9c)
        };
        let at = self.keyframes[i].0;
        if !r.restore(&self.keyframes[i].1) {
            return false;
        }
        self.tick = at;
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

    /// Take a rung if this boundary is one and we are not already holding it. A replay re-takes a rung
    /// it crossed on the way out and has since evicted, which is what keeps the ladder dense around
    /// wherever the driver is working.
    fn keyframe_here<R: MomentReactor + ?Sized>(&mut self, r: &mut R) {
        if !self.tick.is_multiple_of(self.stride) {
            return;
        }
        let Err(at) = self.keyframes.binary_search_by_key(&self.tick, |(t, _)| *t) else {
            return; // already held for this tick
        };
        let Some(m) = r.moment() else {
            return; // an uncapturable window: the ladder stays empty and `seek` refuses, rather than
                    // holding a partial image that would restore a fiction (#9c)
        };
        self.keyframes.insert(at, (self.tick, m));
        self.evict();
    }

    /// Bring the ladder back inside both bounds, never below the pinned tick-0 rung.
    fn evict(&mut self) {
        while self.keyframes.len() > 1
            && (self.keyframes.len() > self.ring
                || (self.budget > 0 && self.held_bytes() > self.budget))
        {
            // Tick 0 is pinned; of the rest, drop whichever rung is furthest from where we are, so the
            // ladder stays dense around the working position instead of around where the run started.
            // Dropping the *oldest* would evict a rung we just re-took while scrubbing back.
            let here = self.tick;
            let victim = (1..self.keyframes.len())
                .max_by_key(|&i| self.keyframes[i].0.abs_diff(here))
                .expect("len > 1, so there is a rung after the pinned one");
            self.keyframes.remove(victim);
        }
    }
}
