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
use temen_browser::{Frame, OnrampReactor, ReactorMoment, SharedOnrampReactor, STATUS_OK};

/// The shared window the `SharedOnrampReactor` cases run over (matches `shared_reactor.rs`).
const WIN_LOG2: u8 = 25;
// JS keyCodes the `bounce` guest steers on.
const LEFT: i32 = 37;
const RIGHT: i32 = 39;

/// One reactor under test. Both interpreter reactors expose the same frame/input/moment surface, so
/// every case below is written once against this and run against each.
trait MomentReactor: Sized {
    fn open_fixture(bytes: &[u8]) -> Self;
    fn step(&mut self) -> Frame;
    fn key(&self, keycode: i32, pressed: i32);
    fn moment(&self) -> Option<ReactorMoment>;
    fn restore(&mut self, m: &ReactorMoment) -> bool;
}

impl MomentReactor for OnrampReactor {
    fn open_fixture(bytes: &[u8]) -> Self {
        let m = temen_encode::decode_module(bytes).expect("decode fixture");
        OnrampReactor::open(&m).expect("open the reactor")
    }
    fn step(&mut self) -> Frame {
        let (status, _stdout) = self.frame();
        assert_eq!(status, STATUS_OK, "tick should keep going");
        self.take_frame().expect("tick presented a frame")
    }
    fn key(&self, keycode: i32, pressed: i32) {
        self.push_key(keycode, pressed);
    }
    fn moment(&self) -> Option<ReactorMoment> {
        OnrampReactor::moment(self)
    }
    fn restore(&mut self, m: &ReactorMoment) -> bool {
        OnrampReactor::restore(self, m)
    }
}

impl MomentReactor for SharedOnrampReactor {
    fn open_fixture(bytes: &[u8]) -> Self {
        let m = temen_encode::decode_module(bytes).expect("decode fixture");
        SharedOnrampReactor::open_owned(&m, WIN_LOG2).expect("open the shared reactor")
    }
    fn step(&mut self) -> Frame {
        let (status, _stdout) = self.frame();
        assert_eq!(status, STATUS_OK, "tick should keep going");
        self.take_frame().expect("tick presented a frame")
    }
    fn key(&self, keycode: i32, pressed: i32) {
        self.push_key(keycode, pressed);
    }
    fn moment(&self) -> Option<ReactorMoment> {
        SharedOnrampReactor::moment(self)
    }
    fn restore(&mut self, m: &ReactorMoment) -> bool {
        SharedOnrampReactor::restore(self, m)
    }
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
fn drive<R: MomentReactor>(r: &R, i: usize) {
    match i % 6 {
        0 => r.key(RIGHT, 1),
        2 => r.key(RIGHT, 0),
        3 => r.key(LEFT, 1),
        5 => r.key(LEFT, 0),
        _ => {}
    }
}

/// Run `frames` frames from the current state, feeding the scripted input for each (offset by
/// `from` so a replay presents the *same* schedule the recorded run saw), returning the frame hashes.
fn run_scripted<R: MomentReactor>(r: &mut R, from: usize, frames: usize) -> Vec<u64> {
    (from..from + frames)
        .map(|i| {
            drive(r, i);
            frame_hash(&r.step())
        })
        .collect()
}

/// The core gate, run against one fixture on one reactor: a rewound reactor replays the recorded
/// future frame for frame.
fn rewind_replays_the_recorded_future<R: MomentReactor>(fixture: &[u8]) {
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
        r.key(key, 1);
        (0..6).map(|_| frame_hash(&r.step())).collect()
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
        let a = frame_hash(&plain.step());
        held.push(captured.moment().expect("capturable"));
        let b = frame_hash(&captured.step());
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
    let first = frame_hash(&r.step());
    let moment = r.moment().expect("capturable");
    let after = frame_hash(&r.step());
    assert!(r.restore(&moment));
    assert_eq!(after, frame_hash(&r.step()), "the file view replays");
    assert_eq!(
        first, after,
        "fsread renders the same bytes each frame (the fixture's own property)"
    );

    // The cursors really are in the moment: a restore reinstates the captured vector wholesale, so a
    // reactor whose guest had re-opened the file (growing the cursor list) is rewound to the
    // captured list — asserted through the frame, which is what the guest can see.
    assert!(r.restore(&moment));
    assert_eq!(after, frame_hash(&r.step()));
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
    r.key(LEFT, 1);
    let moment = r.moment().expect("capturable");
    let with_pending: Vec<u64> = (0..5).map(|_| frame_hash(&r.step())).collect();

    // Rewind: the pending keypress must be back in the queue, so the same frames follow without the
    // driver re-sending it.
    assert!(r.restore(&moment));
    let replayed: Vec<u64> = (0..5).map(|_| frame_hash(&r.step())).collect();
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
    let without: Vec<u64> = (0..5).map(|_| frame_hash(&fresh.step())).collect();
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

/// The `fs` capability's cursors ride the artifact, through the capability's own declared state
/// (#1455). This is the case a window-only save-state gets wrong: the guest's memory comes back but
/// its open file is at whatever offset a freshly-granted server starts at.
#[test]
fn a_thawed_reactor_keeps_its_fs_cursors() {
    let blob: Vec<u8> = (0..=255u8).collect();
    let m = temen_encode::decode_module(FSREAD).expect("decode fsread.temen");
    let mut live = OnrampReactor::open_with_fs(&m, "data.bin".to_string(), blob.clone())
        .expect("open the fsread reactor");
    let before = frame_hash(&live.step());

    let artifact = live.freeze(&m).expect("freeze a cap-using reactor");
    // The capability table is what used to make this impossible: `display` + `keyboard` + `fs` are all
    // host capabilities, and before #1455 any one of them refused the freeze outright.
    let mut thawed = OnrampReactor::thaw(&artifact, &m, Some(("data.bin".to_string(), blob)))
        .expect("thaw re-grants display/keyboard/fs by name");
    assert_eq!(
        before,
        frame_hash(&thawed.step()),
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
    let steered: Vec<u64> = (0..5).map(|_| frame_hash(&live.step())).collect();

    let mut thawed = OnrampReactor::thaw(&artifact, &m, None).expect("thaw");
    assert_eq!(
        steered,
        (0..5)
            .map(|_| frame_hash(&thawed.step()))
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
/// round trip.
#[test]
fn a_thawed_reactor_can_be_frozen_again() {
    let m = temen_encode::decode_module(BOUNCE).expect("decode bounce.temen");
    let mut live = OnrampReactor::open(&m).expect("open");
    let _ = run_scripted(&mut live, 0, 5);
    let first = live.freeze(&m).expect("freeze");

    let mut thawed = OnrampReactor::thaw(&first, &m, None).expect("thaw");
    let expected = run_scripted(&mut thawed, 5, 4);
    let second = thawed.freeze(&m).expect("a thawed reactor freezes again");

    let mut twice = OnrampReactor::thaw(&second, &m, None).expect("thaw the second artifact");
    assert_eq!(
        run_scripted(&mut twice, 9, 4),
        run_scripted(&mut thawed, 9, 4),
        "the second round trip is as faithful as the first"
    );
    assert!(!expected.is_empty());
}
