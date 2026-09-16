//! **A reactor timeline off the browser** (#1457 item 6) — the host-target axis of INVARIANTS #14.
//!
//! The moment/ladder machinery lives in `temen_interp::moment` rather than in the browser cdylib
//! precisely so this file can exist: a native embedder builds a reactor out of the same
//! `bytecode::Reactor` + `Host` the playground does, and gets the same capture, the same tape and the
//! same keyframe ladder — not a second implementation that has to be kept in step (INVARIANTS #15).
//! Nothing here is browser-shaped: a hand-granted powerbox, a hand-written guest, a plain `cargo test`.
//!
//! The guest is also one the playground's fixtures cannot be. `bounce` and `life` drain their whole
//! input queue every tick and fold it into a direction, which makes a *duplicated* event invisible in
//! their output — so they cannot witness the one ordering rule where the tape and the moment could
//! overlap. This guest consumes exactly **one** event per tick, which the `keyboard` capability's
//! `poll` ABI explicitly allows and is how Doom's `DG_GetKey` is pumped, and folds the sequence into a
//! rolling hash. Feed a tick's input twice and every later value differs.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use temen_interp::moment::{MomentReactor, ReactorMoment, ReactorTimeline, SteppableReactor};
use temen_interp::{bytecode, Host, Value};
use temen_text::parse_module;

/// `tick()`: resolve the `kbd` capability by name (cap 13 is `HOST_PROC`), take **one** event from it (`-1` when the queue is
/// empty), fold it into a rolling hash held in the window above the old snapshot cap, and return the
/// hash. The fold is order- and multiplicity-sensitive on purpose: it is what makes a mis-fed replay
/// visible. The name bytes are written each tick (idempotent) so the guest needs no separate init, and
/// they live up beside the accumulator because the window's low pages carry the module's data segments
/// and are write-protected.
const SRC: &str = r#"
memory 19
func () -> (i64) {
block 0 () {
  vnp = i64.const 310000
  vk = i32.const 107
  i32.store8 vnp vk
  vn1 = i64.const 310001
  vb = i32.const 98
  i32.store8 vn1 vb
  vn2 = i64.const 310002
  vd = i32.const 100
  i32.store8 vn2 vd
  vnl = i64.const 3
  vh = self.resolve vnp vnl
  vzero = i64.const 0
  vev = call.cap 13 0 (i64) -> (i64) vh (vzero)
  vaddr = i64.const 300000
  vacc = i64.load vaddr
  v31 = i64.const 31
  vmul = i64.mul vacc v31
  vtwo = i64.const 2
  vev2 = i64.add vev vtwo
  vsum = i64.add vmul vev2
  i64.store vaddr vsum
  return vsum
  }
}
"#;

/// The host-side event queue the `kbd` capability serves — the native twin of the playground's
/// `KeyQueue`, and (as there) the part of the guest's state that lives in a capability rather than in
/// the window, so a moment has to carry it or a rewind replays a differently-steered run.
type Queue = Arc<Mutex<VecDeque<i64>>>;

/// A native reactor driver: a persistent `bytecode::Reactor` over one window, one stateful host
/// capability, and whatever the embedder feeds it. This is the shape an embedder outside the browser
/// writes, and the whole point of the file — it reaches the same ladder with no cdylib in sight.
struct NativeReactor {
    inst: bytecode::Reactor,
    host: Host,
    queue: Queue,
    /// What the last tick returned — this reactor's equivalent of a presented frame.
    last: i64,
}

impl NativeReactor {
    fn open() -> NativeReactor {
        let m = parse_module(SRC).expect("parse the tick guest");
        let inst = bytecode::Reactor::open(&m).expect("open the reactor");
        let mut host = Host::new();
        // §3.5: register the running module, as every powerbox host does — `self.resolve` is part of
        // that self-referential surface.
        host.set_self_module(&std::sync::Arc::new(m.clone()));
        let queue: Queue = Arc::new(Mutex::new(VecDeque::new()));

        // One event per call, `-1` when empty — the `poll` ABI the playground's `keyboard` cap serves.
        let q = Arc::clone(&queue);
        let kbd = host.grant_host_proc(Box::new(move |_op, _args, _mem, _minter| {
            Ok(vec![q.lock().unwrap().pop_front().unwrap_or(-1)])
        }));
        host.register_cap_name("kbd", kbd);
        // The capability declares its own state — the undrained queue — through the pair a moment
        // captures and a §12 freeze writes into its named-capability section (#1455). One definition,
        // read two ways.
        let q = Arc::clone(&queue);
        host.set_cap_state_capture(
            kbd,
            Box::new(move || {
                q.lock()
                    .unwrap()
                    .iter()
                    .flat_map(|e| e.to_le_bytes())
                    .collect()
            }),
        );
        let q = Arc::clone(&queue);
        host.set_cap_state_restore(
            kbd,
            Box::new(move |bytes: &[u8]| {
                *q.lock().unwrap() = bytes
                    .chunks_exact(8)
                    .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
                    .collect();
            }),
        );

        NativeReactor {
            inst,
            host,
            queue,
            last: 0,
        }
    }
}

impl SteppableReactor for NativeReactor {
    fn step(&mut self) -> i32 {
        let mut fuel = u64::MAX;
        match self.inst.call(0, &[], &mut fuel, &mut self.host) {
            Ok(v) => {
                self.last = match v.first() {
                    Some(Value::I64(x)) => *x,
                    _ => 0,
                };
                0
            }
            Err(_) => -1,
        }
    }
}

impl MomentReactor for NativeReactor {
    fn push_key(&self, keycode: i32, pressed: i32) {
        self.queue
            .lock()
            .unwrap()
            .push_back((((pressed & 1) << 16) | (keycode & 0xffff)) as i64);
    }
    fn push_mouse(&self, kind: i32, payload: i32) {
        // This reactor grants one input capability, so a pointer event rides the same queue under its
        // own encoding rather than being silently dropped.
        self.queue
            .lock()
            .unwrap()
            .push_back(((kind as i64) << 32) | (payload as i64 & 0xffff_ffff));
    }
    fn moment(&self) -> Option<ReactorMoment> {
        Some(ReactorMoment::capture(
            self.inst.window_layout()?,
            &self.host,
        ))
    }
    fn restore(&mut self, m: &ReactorMoment) -> bool {
        if !self.inst.restore_window(m.layout()) {
            return false;
        }
        m.restore_host(&mut self.host);
        true
    }
}

/// A scripted input schedule with gaps, so some ticks poll an empty queue and some do not, and both
/// have to replay in the right places.
fn drive(t: &mut ReactorTimeline, i: usize) {
    match i % 5 {
        0 => t.push_key(37, 1),
        1 => t.push_key(39, 0),
        3 => t.push_mouse(1, 0x1234),
        _ => {}
    }
}

fn record(t: &mut ReactorTimeline, r: &mut NativeReactor, frames: usize) -> Vec<i64> {
    (0..frames)
        .map(|_| {
            let i = t.tick();
            drive(t, i);
            assert_eq!(t.frame(r), 0, "the guest keeps going");
            r.last
        })
        .collect()
}

fn play(t: &mut ReactorTimeline, r: &mut NativeReactor, frames: usize) -> Vec<i64> {
    (0..frames)
        .map(|_| {
            assert_eq!(t.frame(r), 0, "the guest keeps going");
            r.last
        })
        .collect()
}

/// The #14 host-target gate: every recorded tick is reachable off the browser, in any order,
/// repeatedly, and replays exactly what it produced.
#[test]
fn a_native_reactor_scrubs_its_own_recording() {
    const N: usize = 40;
    let mut r = NativeReactor::open();
    let mut t = ReactorTimeline::new(8, 4, 0);
    let recorded = record(&mut t, &mut r, N);

    assert!(
        recorded
            .iter()
            .collect::<std::collections::HashSet<_>>()
            .len()
            > 1,
        "the guest's output varies, so the gate is not vacuous"
    );
    // Three frames per probe, not one. This guest takes a single event per tick, so an input
    // delivered twice leaves the *first* value right and every later one an event behind — a
    // one-frame probe would miss exactly the defect this file exists to catch.
    for target in [36usize, 3, 22, 0, 37, 22, 8] {
        assert!(t.seek(&mut r, target), "tick {target} is on the recording");
        assert_eq!(
            play(&mut t, &mut r, 3),
            recorded[target..target + 3].to_vec(),
            "ticks {target}.. replay identically"
        );
    }
    assert!(
        !t.seek(&mut r, N + 1),
        "a position past the recording is refused"
    );
}

/// The gate the playground's fixtures cannot give, on the guest shape that can see it: input is fed
/// **exactly once** per replayed tick.
///
/// A keyframe is taken before the tick's input reaches the capability, so that input lives in the tape
/// alone. Were it taken after, the input would sit in the rung *and* the tape, a replay would deliver
/// it twice, and this guest — which takes one event per tick — would be an event behind from there on.
/// Stride 1 puts a rung at every boundary, so every tick's input is pushed at one.
#[test]
fn a_replay_feeds_each_tick_its_input_exactly_once() {
    let mut r = NativeReactor::open();
    let mut t = ReactorTimeline::new(1, 40, 0);
    let recorded = record(&mut t, &mut r, 20);

    // The probe has to outlast the duplicate: a doubled event is consumed *as* the original on the
    // tick it lands, and only shifts the sequence from the tick after.
    for target in [15usize, 11, 6, 2, 1] {
        assert!(t.seek(&mut r, target));
        assert_eq!(
            play(&mut t, &mut r, 4),
            recorded[target..target + 4].to_vec(),
            "tick {target}'s input is delivered once, not twice"
        );
    }

    // And the queue is genuinely load-bearing: a run with the same guest and no input at all produces
    // a different sequence, so the assertions above are about the input and not just about the window.
    let mut bare = NativeReactor::open();
    let mut bt = ReactorTimeline::new(1, 32, 0);
    let idle: Vec<i64> = (0..16)
        .map(|_| {
            assert_eq!(bt.frame(&mut bare), 0);
            bare.last
        })
        .collect();
    assert_ne!(
        idle, recorded,
        "the scripted input changed what the guest computed"
    );
}

/// The capability half of a native moment: undrained input rides the rung. Without it a rewind would
/// restore the guest's memory but leave the queue wherever the abandoned timeline had left it.
#[test]
fn undrained_input_rides_a_native_moment() {
    let mut r = NativeReactor::open();
    let mut t = ReactorTimeline::new(1, 40, 0);
    let _ = record(&mut t, &mut r, 8);

    // Six events in one tick against a guest that takes one per tick, so the queue stays non-empty
    // for five ticks afterwards — which is what puts *undrained* input inside the rungs at 9..13. A
    // moment taken where the queue happens to be empty could not tell whether it carried one.
    for k in [37, 39, 40, 38, 65, 66] {
        t.push_key(k, 1);
    }
    let onward = play(&mut t, &mut r, 8);

    // Rewind into the middle of that backlog: rung 10 holds four still-queued events, and replaying
    // from it has to see them again.
    assert!(t.seek(&mut r, 10));
    assert_eq!(
        play(&mut t, &mut r, 6),
        onward[2..8].to_vec(),
        "the queue the guest had not drained is part of the moment"
    );
}
