//! **Reactor moments** — capture a frame boundary, restore it, and prove the guest replays the same
//! future (#1457, the first cell of the #1454 moment design).
//!
//! A reactor's `tick` returns to the host every frame, so between frames there is no guest stack to
//! capture: a *moment* is the window image (bytes + page-protection map) plus the host-side capability
//! state (undrained input queues, `fs` cursors). That is the whole of it — no `temen-durable`
//! instrumentation, no handle-table codec.
//!
//! The gate these tests are: **warm ≡ cold**, in reactor form. Run a guest forward over a scripted
//! input sequence, recording each presented frame; take a moment part-way; rewind to it and replay the
//! *same* input; the frames after the moment must be identical, frame for frame. That is the same
//! oracle `crates/temen/tests/debug_checkpoints.rs` uses for the time-travel checkpoint ladder, and
//! `crates/temen-run/demos/doom/doom_diff.c` uses for the Doom renderer — a frame hash stream.
//!
//! Both interpreter reactors run every case through one harness (INVARIANTS #15): `OnrampReactor`
//! (engine-backed window) and `SharedOnrampReactor` (window in a caller-owned region — the shape the
//! wasm-JIT tier drives). Fixtures are the reactor suite's own, so no built asset is needed:
//! `bounce` (state in globals), `life` (state in a **malloc heap above the mapped window** — the
//! grown-page case Doom's zone heap is), and `fsread` (state behind the `fs` capability).

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use temen_browser::{
    Frame, MomentReactor, OnrampReactor, ReactorMoment, ReactorTimeline, SharedOnrampReactor,
    SteppableReactor, STATUS_OK,
};

/// Keyframe stride / ring / byte budget for the timeline cases. The budget is off (0) except where a
/// case is about it — these fixtures' windows are small enough that a count bound is the only one that
/// would ever bite.
const STRIDE: usize = 8;
const RING: usize = 4;

/// The shared window the `SharedOnrampReactor` cases run over (matches `shared_reactor.rs`).
const WIN_LOG2: u8 = 25;
// JS keyCodes the `bounce` guest steers on.
const LEFT: i32 = 37;
const RIGHT: i32 = 39;

/// Opening one reactor under test from a fixture — the only part of the surface these cases need
/// that the library's [`MomentReactor`] does not carry (that trait is about *driving* a reactor, not
/// building one). Every case below is written once against this and run against each interpreter
/// reactor.
trait Fixture: SteppableReactor + Sized {
    fn open_fixture(bytes: &[u8]) -> Self;
    /// The frame the last tick presented. Not on [`MomentReactor`]: that trait drives ticks, and what
    /// a reactor *presents* is its own business (the wasm-JIT one hands its framebuffer to a JS host).
    fn presented(&self) -> Option<Frame>;
}

impl Fixture for OnrampReactor {
    fn open_fixture(bytes: &[u8]) -> Self {
        let m = temen_encode::decode_module(bytes).expect("decode fixture");
        OnrampReactor::open(&m).expect("open the reactor")
    }
    fn presented(&self) -> Option<Frame> {
        self.take_frame()
    }
}

impl Fixture for SharedOnrampReactor {
    fn open_fixture(bytes: &[u8]) -> Self {
        let m = temen_encode::decode_module(bytes).expect("decode fixture");
        SharedOnrampReactor::open_owned(&m, WIN_LOG2).expect("open the shared reactor")
    }
    fn presented(&self) -> Option<Frame> {
        self.take_frame()
    }
}

/// Run one tick and take the frame it presented, asserting the reactor kept going.
fn stepped<R: Fixture>(r: &mut R) -> Frame {
    assert_eq!(r.step(), STATUS_OK, "tick should keep going");
    r.presented().expect("tick presented a frame")
}

const BOUNCE: &[u8] = include_bytes!("fixtures/bounce.temen");
const LIFE: &[u8] = include_bytes!("fixtures/life.temen");
const FSREAD: &[u8] = include_bytes!("fixtures/fsread.temen");

/// A stable hash of a presented frame (dims + pixels) — the per-frame equality unit, as in the Doom
/// differential and `jit_reactor.rs`.
fn frame_hash(f: &Frame) -> u64 {
    let mut h = DefaultHasher::new();
    f.width.hash(&mut h);
    f.height.hash(&mut h);
    f.rgba.hash(&mut h);
    h.finish()
}

/// The scripted input for frame `i` of a run: a key press/release schedule that actually steers the
/// guest, so the recorded frames differ from an idle run (a rewind that replayed *nothing* would pass
/// a test whose input never mattered).
fn drive<R: Fixture>(r: &R, i: usize) {
    match i % 6 {
        0 => r.push_key(RIGHT, 1),
        2 => r.push_key(RIGHT, 0),
        3 => r.push_key(LEFT, 1),
        5 => r.push_key(LEFT, 0),
        _ => {}
    }
}

/// Run `frames` frames from the current state, feeding the scripted input for each (offset by
/// `from` so a replay presents the *same* schedule the recorded run saw), returning the frame hashes.
fn run_scripted<R: Fixture>(r: &mut R, from: usize, frames: usize) -> Vec<u64> {
    (from..from + frames)
        .map(|i| {
            drive(r, i);
            frame_hash(&stepped(r))
        })
        .collect()
}

/// The core gate, run against one fixture on one reactor: a rewound reactor replays the recorded
/// future frame for frame.
fn rewind_replays_the_recorded_future<R: Fixture>(fixture: &[u8]) {
    const WARMUP: usize = 7; // frames before the moment (so it is mid-animation, not at init)
    const TAIL: usize = 11; // frames recorded after it

    let mut r = R::open_fixture(fixture);
    let warm = run_scripted(&mut r, 0, WARMUP);
    let moment = r.moment().expect("a reactor window is capturable");
    let recorded = run_scripted(&mut r, WARMUP, TAIL);

    assert!(r.restore(&moment), "restore into a live reactor");
    let replayed = run_scripted(&mut r, WARMUP, TAIL);
    assert_eq!(
        recorded, replayed,
        "a rewound reactor must replay the recorded frames exactly"
    );

    // And the moment is *reusable* — restoring twice replays the same future twice, so a ladder can
    // seek back to the same keyframe repeatedly rather than consuming it.
    assert!(r.restore(&moment));
    assert_eq!(recorded, run_scripted(&mut r, WARMUP, TAIL));

    // The recorded tail must actually depend on the run — a fixture whose frames never change would
    // make every assertion above vacuous.
    assert!(
        recorded
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "the fixture animates, so the gate is not vacuous"
    );
    assert_ne!(
        warm.last(),
        recorded.last(),
        "the tail differs from the warm-up, so the moment is mid-run"
    );
}

#[test]
fn rewind_replays_the_recorded_future_bounce() {
    rewind_replays_the_recorded_future::<OnrampReactor>(BOUNCE);
    rewind_replays_the_recorded_future::<SharedOnrampReactor>(BOUNCE);
}

/// The Doom-shaped case: `life` keeps its grids in a **malloc heap above the mapped window**, so its
/// state lives in `vm_map`-grown pages. Those ride the moment only because the capture carries the
/// page-protection map alongside the bytes (`MemLayout`) — a bytes-only image would restore a window
/// whose grown pages are no longer addressable.
#[test]
fn rewind_carries_a_heap_grown_above_the_window() {
    rewind_replays_the_recorded_future::<OnrampReactor>(LIFE);
    rewind_replays_the_recorded_future::<SharedOnrampReactor>(LIFE);
}

/// A moment is a **branch point**: two different input sequences from the same moment produce two
/// different futures, and each is reproducible by restoring again. This is the property a scrub UI
/// needs when the user rewinds and then plays differently — and the one `fork` would build on.
#[test]
fn a_moment_branches_into_reproducible_timelines() {
    let mut r = OnrampReactor::open_fixture(BOUNCE);
    let _ = run_scripted(&mut r, 0, 5);
    let moment = r.moment().expect("capturable");

    // Timeline A: hold RIGHT. Timeline B: hold LEFT. The box steers opposite ways, so the frames differ.
    let timeline = |r: &mut OnrampReactor, key: i32| -> Vec<u64> {
        r.push_key(key, 1);
        (0..6).map(|_| frame_hash(&stepped(r))).collect()
    };

    let a = timeline(&mut r, RIGHT);
    assert!(r.restore(&moment));
    let b = timeline(&mut r, LEFT);
    assert_ne!(
        a, b,
        "different input from one moment ⇒ different timelines"
    );

    assert!(r.restore(&moment));
    assert_eq!(a, timeline(&mut r, RIGHT), "timeline A reproduces");
    assert!(r.restore(&moment));
    assert_eq!(b, timeline(&mut r, LEFT), "timeline B reproduces");
}

/// **Inertness pin** (INVARIANTS #9b — observation never perturbs semantics): a run that captures a
/// moment at every frame boundary presents exactly the frames a run that captures none does. Capture
/// is a read of the window; nothing about it is visible to the guest.
#[test]
fn capturing_moments_does_not_perturb_the_run() {
    let mut plain = OnrampReactor::open_fixture(LIFE);
    let mut captured = OnrampReactor::open_fixture(LIFE);

    let mut held = Vec::new();
    for i in 0..12 {
        drive(&plain, i);
        drive(&captured, i);
        let a = frame_hash(&stepped(&mut plain));
        held.push(captured.moment().expect("capturable"));
        let b = frame_hash(&stepped(&mut captured));
        assert_eq!(a, b, "frame {i} is unchanged by capturing moments");
    }
    assert!(held[0].byte_len() > 0, "a moment carries the window image");
}

/// The `fs` capability's cursors are **host-side guest state**: they live in the capability, not the
/// window, so a moment that carried only the window would rewind the guest's memory while leaving its
/// open file mid-stream. Capture them and a restored guest re-reads exactly what it read before.
#[test]
fn restore_rewinds_the_fs_cursors() {
    let blob: Vec<u8> = (0..=255u8).collect();
    let m = temen_encode::decode_module(FSREAD).expect("decode fsread.temen");
    let mut r = OnrampReactor::open_with_fs(&m, "data.bin".to_string(), blob)
        .expect("open the fsread reactor");

    // `_start` has opened and read the file, so the cursors are live state by the first frame.
    let first = frame_hash(&stepped(&mut r));
    let moment = r.moment().expect("capturable");
    let after = frame_hash(&stepped(&mut r));
    assert!(r.restore(&moment));
    assert_eq!(after, frame_hash(&stepped(&mut r)), "the file view replays");
    assert_eq!(
        first, after,
        "fsread renders the same bytes each frame (the fixture's own property)"
    );

    // The cursors really are in the moment: a restore reinstates the captured vector wholesale, so a
    // reactor whose guest had re-opened the file (growing the cursor list) is rewound to the
    // captured list — asserted through the frame, which is what the guest can see.
    assert!(r.restore(&moment));
    assert_eq!(after, frame_hash(&stepped(&mut r)));
}

/// Input the host has accepted but the guest has not yet polled is part of the moment. Without it a
/// rewind would replay a *different* input stream than the one it recorded — the queue would be empty
/// where the recording had a keypress pending.
#[test]
fn undrained_input_rides_the_moment() {
    let mut r = OnrampReactor::open_fixture(BOUNCE);
    let _ = run_scripted(&mut r, 0, 3);

    // Enqueue a steer the guest has NOT drained yet, then capture. LEFT, because the scripted warm-up
    // leaves the box heading +x — a RIGHT here would be a no-op and the test would prove nothing.
    r.push_key(LEFT, 1);
    let moment = r.moment().expect("capturable");
    let with_pending: Vec<u64> = (0..5).map(|_| frame_hash(&stepped(&mut r))).collect();

    // Rewind: the pending keypress must be back in the queue, so the same frames follow without the
    // driver re-sending it.
    assert!(r.restore(&moment));
    let replayed: Vec<u64> = (0..5).map(|_| frame_hash(&stepped(&mut r))).collect();
    assert_eq!(
        with_pending, replayed,
        "the undrained keypress rides the moment"
    );

    // And the converse: a moment taken *without* a pending key does not resurrect one. Rewinding to
    // it and stepping must differ from the sequence that had the steer queued.
    let mut fresh = OnrampReactor::open_fixture(BOUNCE);
    let _ = run_scripted(&mut fresh, 0, 3);
    let idle = fresh.moment().expect("capturable");
    assert!(fresh.restore(&idle));
    let without: Vec<u64> = (0..5).map(|_| frame_hash(&stepped(&mut fresh))).collect();
    assert_ne!(
        with_pending, without,
        "the queued steer changed the future, so the queue capture is load-bearing"
    );
}

// ---- #1458: a reactor freezes to a §12 artifact and thaws ------------------------------------
//
// A moment lives in this process; an **artifact** does not. It is the same state — window image, page
// map, capability state — written into the durability codec's container, so it can be persisted,
// reloaded into a fresh instance, and played on: a save-state.
//
// Two properties are load-bearing and neither is machinery, they are both consequences of freezing at
// a frame boundary. There is no continuation to capture, so the module needs none of the
// `temen-durable` instrumentation an arbitrary-safepoint freeze does. And the image is post-`_start`,
// so a thaw does not re-run init — for Doom that is seconds of WAD parsing skipped, which is most of
// what a save-state is for.

/// The gate: freeze mid-run, thaw into a **fresh** reactor, and the frames that follow are the frames
/// that would have followed. Same warm ≡ cold shape as the in-process rewind, across a serialization.
#[test]
fn a_reactor_freezes_to_an_artifact_and_thaws_playing() {
    let m = temen_encode::decode_module(BOUNCE).expect("decode bounce.temen");
    let mut live = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut live, 0, 9);

    let artifact = live
        .freeze(&m)
        .expect("a reactor at a frame boundary is freezable");
    let expected = run_scripted(&mut live, 9, 7);

    let mut thawed = OnrampReactor::thaw(&artifact, &m, None).expect("thaw");
    assert_eq!(
        expected,
        run_scripted(&mut thawed, 9, 7),
        "a thawed reactor plays on exactly where the frozen one did"
    );

    // And the artifact is reusable — a save-state is loaded more than once.
    let mut again = OnrampReactor::thaw(&artifact, &m, None).expect("thaw again");
    assert_eq!(expected, run_scripted(&mut again, 9, 7));
}

/// The Doom-shaped case, as a save-state: `life` keeps its grids in a **malloc heap above the mapped
/// window**, so its state is in `vm_map`-grown reserved-tail pages. Those ride the artifact only if
/// the codec's page map marks them committed — a thaw that brought them back unmapped would fault on
/// the first tick.
#[test]
fn a_grown_heap_rides_the_artifact() {
    let m = temen_encode::decode_module(LIFE).expect("decode life.temen");
    let mut live = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut live, 0, 7);

    let artifact = live.freeze(&m).expect("freeze a grown window");
    let expected = run_scripted(&mut live, 7, 5);

    let mut thawed = OnrampReactor::thaw(&artifact, &m, None).expect("thaw");
    assert_eq!(
        expected,
        run_scripted(&mut thawed, 7, 5),
        "the grown heap came back committed and the frames that follow are the same"
    );
}

/// The `fs` capability's cursors ride the artifact, through the capability's own declared state
/// (#1455). This is the case a window-only save-state gets wrong: the guest's memory comes back but
/// its open file is at whatever offset a freshly-granted server starts at.
#[test]
fn a_thawed_reactor_keeps_its_fs_cursors() {
    let blob: Vec<u8> = (0..=255u8).collect();
    let m = temen_encode::decode_module(FSREAD).expect("decode fsread.temen");
    let mut live = OnrampReactor::open_with_fs(&m, "data.bin".to_string(), blob.clone())
        .expect("open the fsread reactor");
    let before = frame_hash(&stepped(&mut live));

    let artifact = live.freeze(&m).expect("freeze a cap-using reactor");
    // The capability table is what used to make this impossible: `display` + `keyboard` + `fs` are all
    // host capabilities, and before #1455 any one of them refused the freeze outright.
    let mut thawed = OnrampReactor::thaw(&artifact, &m, Some(("data.bin".to_string(), blob)))
        .expect("thaw re-grants display/keyboard/fs by name");
    assert_eq!(
        before,
        frame_hash(&stepped(&mut thawed)),
        "the thawed guest renders the same file view"
    );
}

/// Input the guest has been handed but not yet drained rides the artifact too — the same property the
/// in-process moment has, across a serialization.
#[test]
fn a_thawed_reactor_keeps_undrained_input() {
    let m = temen_encode::decode_module(BOUNCE).expect("decode bounce.temen");
    let mut live = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut live, 0, 3);
    live.push_key(LEFT, 1); // queued, not yet polled

    let artifact = live.freeze(&m).expect("freeze");
    let steered: Vec<u64> = (0..5).map(|_| frame_hash(&stepped(&mut live))).collect();

    let mut thawed = OnrampReactor::thaw(&artifact, &m, None).expect("thaw");
    assert_eq!(
        steered,
        (0..5)
            .map(|_| frame_hash(&stepped(&mut thawed)))
            .collect::<Vec<_>>(),
        "the undrained keypress rode the artifact"
    );
}

/// The digest binding: an artifact is bound to the module it was frozen over, so a save-state cannot
/// be loaded under a different guest's code.
#[test]
fn a_save_state_refuses_a_different_module() {
    let bounce = temen_encode::decode_module(BOUNCE).expect("decode");
    let life = temen_encode::decode_module(LIFE).expect("decode");
    let mut r = OnrampReactor::open(&bounce).expect("open");
    let _ = run_scripted(&mut r, 0, 3);
    let artifact = r.freeze(&bounce).expect("freeze");

    assert!(
        OnrampReactor::thaw(&artifact, &life, None).is_err(),
        "one guest's save-state must not restore under another's code"
    );
}

/// A thawed reactor is itself freezable — the state hooks are re-declared after the thaw re-grants the
/// handlers, so a save-state can be taken, loaded, and taken again rather than degrading after one
/// round trip. The second freeze is taken with input queued but undrained, so a thaw that had lost
/// the hooks (an empty state riding the second artifact) is observable: the queued press steers the
/// live reactor and not the twice-thawed one.
#[test]
fn a_thawed_reactor_can_be_frozen_again() {
    let m = temen_encode::decode_module(BOUNCE).expect("decode bounce.temen");
    let mut live = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut live, 0, 5);
    let first = live.freeze(&m).expect("freeze");

    let mut thawed = OnrampReactor::thaw(&first, &m, None).expect("thaw");
    let expected = run_scripted(&mut thawed, 5, 4);
    thawed.push_key(LEFT, 1); // queued, not yet polled — rides the second artifact only via the hooks
    let second = thawed.freeze(&m).expect("a thawed reactor freezes again");

    let mut twice = OnrampReactor::thaw(&second, &m, None).expect("thaw the second artifact");
    assert_eq!(
        run_scripted(&mut twice, 9, 4),
        run_scripted(&mut thawed, 9, 4),
        "the second round trip is as faithful as the first"
    );
    assert!(!expected.is_empty());
}

// ---- #1457 items 3–4: the input tape and the keyframe ladder ---------------------------------
//
// A moment alone only goes *back* to a point you thought to save. A `ReactorTimeline` adds the two
// things that make that a scrub: it records what the driver fed the guest each tick, and it keeps a
// ring of keyframes, so **any** recorded tick can be reconstructed — restore the nearest rung at or
// before it, re-feed the tape forward.
//
// The tape records the *driver*, not the guest. A reactor replays against its live powerbox, and
// everything those capabilities read from is already inside the moment (the input queues, the `fs`
// cursors, the window) — the on-ramp powerbox grants no wall clock and no entropy — so the guest's
// crossings recompute rather than needing to be served from a recording. What is left outside the
// moment is exactly what the host injects from the outside world, and that is what rides the tape.
//
// The timeline does not own the reactor: an embedder keeps its reactors in its own storage, and the
// wasm-JIT tier's tick is run by the embedder rather than by Rust. So every helper here takes both.

/// A reactor that counts the ticks run through it. Landing on the right frame does not by itself
/// prove the ladder did anything — a `seek` that silently re-ran from tick 0 lands there too. This is
/// the instrument that tells the two apart.
struct Counting<R> {
    inner: R,
    steps: usize,
}

impl<R: SteppableReactor> SteppableReactor for Counting<R> {
    fn step(&mut self) -> i32 {
        self.steps += 1;
        self.inner.step()
    }
}

impl<R: MomentReactor> MomentReactor for Counting<R> {
    fn push_key(&self, keycode: i32, pressed: i32) {
        self.inner.push_key(keycode, pressed);
    }
    fn push_mouse(&self, kind: i32, payload: i32) {
        self.inner.push_mouse(kind, payload);
    }
    fn moment(&self) -> Option<ReactorMoment> {
        self.inner.moment()
    }
    fn restore(&mut self, m: &ReactorMoment) -> bool {
        self.inner.restore(m)
    }
}

impl<R: Fixture> Fixture for Counting<R> {
    fn open_fixture(bytes: &[u8]) -> Self {
        Counting {
            inner: R::open_fixture(bytes),
            steps: 0,
        }
    }
    fn presented(&self) -> Option<Frame> {
        self.inner.presented()
    }
}

/// [`drive`], addressed to a timeline so the input lands on its tape. Keyed on the timeline's own
/// position, so a recording carries the same schedule the reactor-level cases use.
fn drive_timeline(t: &mut ReactorTimeline, i: usize) {
    match i % 6 {
        0 => t.push_key(RIGHT, 1),
        2 => t.push_key(RIGHT, 0),
        3 => t.push_key(LEFT, 1),
        5 => t.push_key(LEFT, 0),
        _ => {}
    }
}

/// Run one tick through the timeline and take the frame it presented.
fn tl_frame<R: Fixture>(t: &mut ReactorTimeline, r: &mut R) -> Frame {
    assert_eq!(t.frame(r), STATUS_OK, "tick should keep going");
    r.presented().expect("tick presented a frame")
}

/// Extend the recording by `frames` frames, feeding the scripted input for each.
fn record<R: Fixture>(t: &mut ReactorTimeline, r: &mut R, frames: usize) -> Vec<u64> {
    (0..frames)
        .map(|_| {
            let i = t.tick();
            drive_timeline(t, i);
            frame_hash(&tl_frame(t, r))
        })
        .collect()
}

/// Play `frames` frames from wherever the timeline stands, replaying the tape — no new input, so the
/// recording is followed rather than branched.
fn play<R: Fixture>(t: &mut ReactorTimeline, r: &mut R, frames: usize) -> Vec<u64> {
    (0..frames).map(|_| frame_hash(&tl_frame(t, r))).collect()
}

/// The gate for the pair: **every** recorded tick is reachable, in any order, repeatedly, and each
/// one replays the frame it originally produced. This is warm ≡ cold again, now over a whole
/// recording rather than a single saved point.
fn a_timeline_seeks_to_any_recorded_tick<R: Fixture>(fixture: &[u8]) {
    const N: usize = 40;
    let mut r = R::open_fixture(fixture);
    let mut t = ReactorTimeline::new(STRIDE, RING, 0);
    let recorded = record(&mut t, &mut r, N);
    assert_eq!((t.tick(), t.len()), (N, N));

    for target in [37usize, 3, 22, 0, 39, 22, 8] {
        assert!(t.seek(&mut r, target), "tick {target} is on the recording");
        assert_eq!(t.tick(), target);
        assert_eq!(
            play(&mut t, &mut r, 1),
            vec![recorded[target]],
            "frame {target} replays identically from a seek"
        );
    }

    assert!(
        !t.seek(&mut r, N + 1),
        "a position past the recording is not a position (#9c: refuse, never guess)"
    );
    assert!(
        recorded
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "the fixture animates, so the gate is not vacuous"
    );
}

#[test]
fn a_timeline_seeks_to_any_recorded_tick_bounce() {
    a_timeline_seeks_to_any_recorded_tick::<OnrampReactor>(BOUNCE);
    a_timeline_seeks_to_any_recorded_tick::<SharedOnrampReactor>(BOUNCE);
}

/// The grown-heap case again, now through the ladder: `life`'s grids live in `vm_map`-grown pages
/// above the mapped window, so every rung is imaging that tail and every replay depends on it.
#[test]
fn a_timeline_seeks_over_a_heap_grown_above_the_window() {
    a_timeline_seeks_to_any_recorded_tick::<OnrampReactor>(LIFE);
    a_timeline_seeks_to_any_recorded_tick::<SharedOnrampReactor>(LIFE);
}

/// What the ladder is *for*: a backward seek costs the tail after the nearest rung, not the whole run.
/// Without this the tape alone would be correct and O(t) — exactly the replay-from-0 the debug
/// checkpoint ladder exists to avoid.
#[test]
fn the_ladder_bounds_a_backward_seek() {
    const N: usize = 40;
    let mut r = Counting::<OnrampReactor>::open_fixture(BOUNCE);
    let mut t = ReactorTimeline::new(STRIDE, 8, 0);
    let recorded = record(&mut t, &mut r, N);
    assert_eq!(
        t.keyframe_ticks(),
        vec![0, 8, 16, 24, 32],
        "a rung every stride"
    );

    r.steps = 0;
    assert!(t.seek(&mut r, 39));
    assert!(
        r.steps <= STRIDE,
        "a seek to 39 replays from the rung at 32, not from 0 (ran {} ticks)",
        r.steps
    );
    assert_eq!(play(&mut t, &mut r, 1), vec![recorded[39]]);

    // And a seek *forward* inside the recording needs no rewind at all — it just keeps replaying from
    // where the reactor already stands, so dragging a scrub bar forward costs the frames it crosses.
    assert!(t.seek(&mut r, 4));
    r.steps = 0;
    assert!(t.seek(&mut r, 20));
    assert_eq!(
        r.steps, 16,
        "a forward seek replays only the frames between here and there"
    );
    assert_eq!(play(&mut t, &mut r, 1), vec![recorded[20]]);
}

/// Steering differently from a rewound position is a **branch**: the recorded future is gone, and so
/// are the rungs that described it. This is the property a scrub bar needs the moment the user
/// rewinds and then plays — without it the timeline would claim a future that cannot happen.
#[test]
fn new_input_in_the_past_branches_the_timeline() {
    let mut r = OnrampReactor::open_fixture(BOUNCE);
    let mut t = ReactorTimeline::new(4, 8, 0);
    let recorded = record(&mut t, &mut r, 24);
    assert_eq!(t.len(), 24);

    assert!(t.seek(&mut r, 10));
    // `bounce` steers on key-*downs* and ignores releases, so branch with the opposite direction: the
    // recorded run is heading left here (LEFT↓ at tick 9), and this turns it right on tick 10 itself
    // rather than only diverging later when the truncated RIGHT↓ at tick 12 fails to arrive.
    t.push_key(RIGHT, 1);
    assert_eq!(t.len(), 10, "the abandoned future leaves the tape");
    assert!(
        t.keyframe_ticks().iter().all(|&k| k <= 10),
        "and its keyframes go with it: {:?}",
        t.keyframe_ticks()
    );

    let branch = play(&mut t, &mut r, 6);
    assert_eq!(t.len(), 16, "the branch is the recording now");
    assert_ne!(
        branch,
        recorded[10..16].to_vec(),
        "steering differently really does diverge"
    );

    // The branch is a recording like any other — seekable, replayable. A scrub bar keeps working
    // after the user plays their own way.
    assert!(t.seek(&mut r, 10));
    assert_eq!(
        play(&mut t, &mut r, 6),
        branch,
        "the new timeline replays like any other"
    );
}

/// The ring is a memory budget, and tick 0 is pinned inside it. A ring that could evict the start
/// would leave a region of the scrub track the user can see and cannot drag to; pinning it means the
/// worst case is a long replay rather than a refusal.
#[test]
fn the_ring_is_bounded_and_pins_the_start() {
    const SMALL: usize = 3;
    let mut r = OnrampReactor::open_fixture(LIFE);
    let mut t = ReactorTimeline::new(4, SMALL, 0);
    let recorded = record(&mut t, &mut r, 40);

    let rungs = t.keyframe_ticks();
    assert!(
        rungs.len() <= SMALL,
        "the ring is a budget, not a suggestion: {rungs:?}"
    );
    assert_eq!(
        rungs[0], 0,
        "tick 0 is pinned, so the start of the run stays reachable"
    );
    let one = r.moment().expect("capturable").byte_len();
    assert!(t.held_bytes() > 0 && t.held_bytes() <= SMALL * one);

    // The pin earning its keep: a tick the ring passed long ago is still reachable, by replaying from
    // 0 rather than refusing.
    assert!(t.seek(&mut r, 17));
    assert_eq!(play(&mut t, &mut r, 1), vec![recorded[17]]);
    assert!(t.seek(&mut r, 0));
    assert_eq!(play(&mut t, &mut r, 1), vec![recorded[0]]);
}

/// The **byte** bound, which is the one that matters for a real guest: a rung's cost is the guest's
/// window, so a count alone means "8 rungs" is a few KiB for `bounce` and 128 MiB for Doom. A budget
/// that admits two rungs holds two however big the count allows — and never drops below the pin.
#[test]
fn the_ring_honours_a_byte_budget_under_the_count() {
    let mut r = OnrampReactor::open_fixture(LIFE);
    let one = r.moment().expect("capturable").byte_len();
    // Room for two rungs, against a count that would allow sixteen.
    let mut t = ReactorTimeline::new(4, 16, one * 2 + one / 2);
    let recorded = record(&mut t, &mut r, 40);

    let rungs = t.keyframe_ticks();
    assert!(
        rungs.len() <= 2 && t.held_bytes() <= one * 2 + one / 2,
        "the byte ceiling bit before the count did: {rungs:?} holding {} of {}",
        t.held_bytes(),
        one * 2 + one / 2
    );
    assert_eq!(rungs[0], 0, "the pin survives a budget that tight");
    assert!(
        t.seek(&mut r, 21),
        "and the run is still fully seekable through it"
    );
    assert_eq!(play(&mut t, &mut r, 1), vec![recorded[21]]);
}

/// **Inertness pin** (INVARIANTS #9b): a run recorded and keyframed presents exactly the frames a
/// plain run does. Taping is a read of the driver's own input and keyframing a read of the window;
/// neither is visible to the guest.
#[test]
fn recording_a_timeline_does_not_perturb_the_run() {
    let mut plain = OnrampReactor::open_fixture(LIFE);
    let mut taped = OnrampReactor::open_fixture(LIFE);
    let mut t = ReactorTimeline::new(3, 4, 0);
    for i in 0..15 {
        drive(&plain, i);
        drive_timeline(&mut t, i);
        let a = frame_hash(&stepped(&mut plain));
        let b = frame_hash(&tl_frame(&mut t, &mut taped));
        assert_eq!(a, b, "frame {i} is unchanged by taping and keyframing it");
    }
}

/// The seam where the tape and the moment meet, and the one place they could overlap: a keyframe is
/// taken **before** the tick's input reaches the queues. Taken after, that input would sit in both the
/// rung *and* the tape, and every replay from that rung would feed it twice.
///
/// The fixtures cannot witness that through their frames. `bounce` and `life` drain their whole queue
/// every tick and fold it into a direction, so a duplicated event is invisible to them — whereas the
/// `keyboard` ABI explicitly allows a guest to take one event per tick (`poll` "dequeues one packed
/// event, or -1", which is how Doom's `DG_GetKey` is pumped), and such a guest would drift a frame per
/// rung. So this asserts it where it *is* visible: a rung, restored and frozen, must be byte-identical
/// to a reactor driven the same way that was never handed that tick's input — and the first assertion
/// pins that the two states are distinguishable at all, so passing means something.
#[test]
fn a_keyframe_holds_the_state_before_its_tick_of_input() {
    const AT: usize = 6; // `drive` offers input at tick 6, so the rung there has something to hold
    let m = temen_encode::decode_module(BOUNCE).expect("decode bounce.temen");

    // The two states the rung at tick 6 could be in: driven to that boundary, frozen once before tick
    // 6's input is offered and once after.
    let mut plain = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut plain, 0, AT);
    let before = plain.freeze(&m).expect("freeze at the boundary");
    drive(&plain, AT); // tick 6's input: queued, not yet polled
    let after = plain.freeze(&m).expect("freeze with it queued");
    assert_ne!(
        before, after,
        "queued input is visible in a frozen reactor, so this gate can tell the two apart"
    );

    let mut r = OnrampReactor::open(&m).expect("open");
    let mut t = ReactorTimeline::new(AT, 8, 0);
    let _ = record(&mut t, &mut r, 18);
    assert!(t.keyframe_ticks().contains(&AT), "there is a rung at {AT}");
    assert!(
        t.seek(&mut r, AT),
        "seek to the rung itself, so it is restored and nothing is replayed"
    );
    assert_eq!(
        r.freeze(&m).expect("freeze the restored rung"),
        before,
        "a rung holds the state *before* its tick's input, so a replay feeds that input exactly once"
    );
}

/// The **split** form the wasm-JIT tier needs: `begin_tick` / run the tick yourself / `end_tick`,
/// with `seek_begin` for the reposition half. An embedder whose tick is emitted wasm run by a JS host
/// cannot hand Rust a `step`, so if this did not exist it would grow its own ladder — which is exactly
/// what the playground page had before this (INVARIANTS #15). Driving the split form by hand must
/// produce the same recording the self-driving `frame` does.
#[test]
fn the_split_form_records_and_seeks_like_the_self_driving_one() {
    let mut whole = OnrampReactor::open_fixture(BOUNCE);
    let mut tw = ReactorTimeline::new(STRIDE, RING, 0);
    let expected = record(&mut tw, &mut whole, 24);

    let mut split = OnrampReactor::open_fixture(BOUNCE);
    let mut ts = ReactorTimeline::new(STRIDE, RING, 0);
    let mut got = Vec::new();
    for _ in 0..24 {
        let i = ts.tick();
        drive_timeline(&mut ts, i);
        ts.begin_tick(&mut split);
        assert_eq!(split.step(), STATUS_OK); // the embedder's own tick
        ts.end_tick();
        got.push(frame_hash(&split.presented().expect("a frame")));
    }
    assert_eq!(got, expected, "hand-driven ticks record the same run");
    assert_eq!(
        ts.keyframe_ticks(),
        tw.keyframe_ticks(),
        "and the same ladder"
    );

    // And the split seek: reposition, then run the tail yourself.
    assert!(ts.seek_begin(&mut split, 9));
    while ts.tick() < 9 {
        ts.begin_tick(&mut split);
        assert_eq!(split.step(), STATUS_OK);
        ts.end_tick();
    }
    assert_eq!(ts.tick(), 9);
    assert_eq!(
        play(&mut ts, &mut split, 4),
        expected[9..13].to_vec(),
        "a hand-driven seek lands where the self-driving one does"
    );
}
