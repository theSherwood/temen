//! **Reactor moments on the wasm-JIT tier** (#1457) — the tier a playable Doom actually runs on, so
//! INVARIANTS #14 requires the capability here and not only on the interpreter reactors
//! (`reactor_moment.rs`).
//!
//! The question this answers that the interpreter cases cannot: emitted wasm writes the window
//! *directly*, and it also reads host-maintained wasm globals out of an **env cell that lives outside
//! the window** (fuel, `mapped`, the dispatch table). If any guest-visible state lived there, a moment
//! — which images the window and nothing else — would silently miss it, and a rewound guest would
//! drift. It does not: the env cell is host configuration the driver rewrites every frame, so the
//! window really is the whole of the guest's state on this tier too.
//!
//! The harness plays the browser's JS host with `wasmi`, as `jit_reactor.rs` does for Doom: one linear
//! memory holding the env cell plus the window, the emitted `f{tick}` called per frame, and
//! `env.call_interp` serviced by running the cross-tier callee on the interpreter over the same
//! window. Unlike that file this one needs no built asset — it drives the reactor suite's own
//! `bounce`/`life` fixtures, which the cap-call outlining pass made emittable — so it gates in CI.
//!
//! (`jit_reactor.rs` carries its own copy of this wasmi glue, written for Doom. The two want to
//! converge onto one parameterised driver; that is noted on #1457 rather than done blind here, since
//! this tree cannot run the Doom-gated one to check a refactor of it.)

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use temen_browser::{
    Frame, JitOnrampReactor, JitStart, MomentReactor, ReactorMoment, ReactorTimeline, TickOutcome,
    STATUS_OK,
};
use temen_interp::Value;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 24;
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // the window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;
const LEFT: i32 = 37;
const RIGHT: i32 = 39;

fn frame_hash(f: &Frame) -> u64 {
    let mut h = DefaultHasher::new();
    f.width.hash(&mut h);
    f.height.hash(&mut h);
    f.rgba.hash(&mut h);
    h.finish()
}

/// A live wasm-JIT reactor on `wasmi`: the emitted `tick` plus everything needed to call it. Frames
/// are driven with [`JitDriver::step`]; the reactor itself (and so its `moment`/`restore`) is reached
/// with [`JitDriver::reactor`].
struct JitDriver {
    store: Store<Option<JitOnrampReactor>>,
    memory: Memory,
    f_tick: wasmi::Func,
    entry_sp: u64,
    results: Vec<Val>,
}

impl JitDriver {
    fn open(fixture: &[u8]) -> JitDriver {
        JitDriver::start(fixture, JitStart::Entry)
    }

    /// The same driver over a reactor thawed from a §12 save-state instead of booted (#1458).
    fn thaw(fixture: &[u8], artifact: &[u8]) -> JitDriver {
        JitDriver::start(fixture, JitStart::Thaw(artifact))
    }

    fn start(fixture: &[u8], start: JitStart<'_>) -> JitDriver {
        let m = temen_encode::decode_module(fixture).expect("decode fixture");
        let engine = Engine::default();
        let total_bytes = WIN_BASE as u64 + WIN_SIZE;
        let pages = (total_bytes / (64 * 1024)) as u32;
        let mut store: Store<Option<JitOnrampReactor>> = Store::new(&engine, None);
        let memory =
            Memory::new(&mut store, MemoryType::new(pages, Some(pages))).expect("wasmi memory");

        let win_ptr = unsafe {
            memory
                .data_mut(&mut store)
                .as_mut_ptr()
                .add(WIN_BASE as usize)
        };
        // SAFETY: `memory` is fixed-size (min == max), so its data pointer is stable for the run; the
        // window `[win_ptr, WIN_SIZE)` lives inside it and is used solely as this reactor's window.
        let reactor = unsafe {
            JitOnrampReactor::open_shared_jit(&m, win_ptr, WIN_SIZE, WIN_LOG2, false, None, start)
        }
        .expect("the fixture's tick is wasm-JIT-emittable");

        let emitted_wasm = reactor.emitted_wasm().to_vec();
        let entry_sp = reactor.entry_sp();
        let tick = reactor.tick();
        let module = WModule::new(&engine, &emitted_wasm).expect("emitted tick validates");
        let rtys: Vec<temen_ir::ValType> = reactor.func_sig(tick).1.to_vec();
        *store.data_mut() = Some(reactor);

        let mut linker: Linker<Option<JitOnrampReactor>> = Linker::new(&engine);
        linker.define("env", "memory", memory).unwrap();
        linker
            .func_wrap("env", "trap", |_caller: Caller<'_, _>, _code: i32| {})
            .unwrap();
        linker
            .func_wrap(
                "env",
                "call_interp",
                move |mut caller: Caller<'_, Option<JitOnrampReactor>>,
                      func: i32,
                      args_ptr: i32|
                      -> Result<(), wasmi::Error> {
                    let (params, results) = {
                        let r = caller.data().as_ref().unwrap();
                        let (p, rs) = r.func_sig(func as u32);
                        (p.to_vec(), rs.to_vec())
                    };
                    let args: Vec<Value> = {
                        let data = memory.data(&caller);
                        params
                            .iter()
                            .enumerate()
                            .map(|(i, t)| {
                                let o = args_ptr as usize + i * 8;
                                let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                                match t {
                                    temen_ir::ValType::I32 => Value::I32(raw as i32),
                                    _ => Value::I64(raw as i64),
                                }
                            })
                            .collect()
                    };
                    let outcome = caller
                        .data_mut()
                        .as_mut()
                        .unwrap()
                        .run_cross_tier(func as u32, &args);
                    match outcome {
                        Ok(vals) => {
                            let data = memory.data_mut(&mut caller);
                            for (i, v) in vals.iter().enumerate() {
                                if i >= results.len() {
                                    break;
                                }
                                let raw = match v {
                                    Value::I32(x) => *x as u32 as u64,
                                    Value::I64(x) => *x as u64,
                                    _ => 0,
                                };
                                let o = args_ptr as usize + i * 8;
                                data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                            }
                            Ok(())
                        }
                        Err(t) => {
                            caller
                                .data_mut()
                                .as_mut()
                                .unwrap()
                                .set_last_trap(format!("{t:?}"));
                            Err(wasmi::Error::from(
                                wasmi::core::TrapCode::UnreachableCodeReached,
                            ))
                        }
                    }
                },
            )
            .unwrap();

        let instance = linker
            .instantiate(&mut store, &module)
            .unwrap()
            .start(&mut store)
            .unwrap();
        let f_tick = instance
            .get_func(&store, &format!("f{tick}"))
            .expect("emitted f{tick} export");
        let results: Vec<Val> = rtys
            .iter()
            .map(|t| match t {
                temen_ir::ValType::I64 => Val::I64(0),
                temen_ir::ValType::F32 => Val::F32(0.0f32.into()),
                temen_ir::ValType::F64 => Val::F64(0.0f64.into()),
                _ => Val::I32(0),
            })
            .collect();

        JitDriver {
            store,
            memory,
            f_tick,
            entry_sp,
            results,
        }
    }

    fn reactor(&self) -> &JitOnrampReactor {
        self.store.data().as_ref().unwrap()
    }

    fn reactor_mut(&mut self) -> &mut JitOnrampReactor {
        self.store.data_mut().as_mut().unwrap()
    }

    /// Run one frame on the emitted tier and hash what it presented — the per-frame equality unit
    /// these cases compare.
    fn hashed(&mut self) -> u64 {
        frame_hash(&self.step().frame.expect("a frame was presented"))
    }
}

/// The wasm-JIT reactor joins the moment/timeline surface by **implementing** the library's trait
/// rather than by a second ladder of its own (INVARIANTS #15). It has to be here rather than in the
/// library because this tier's `tick` is emitted wasm, run by whoever compiled it — the browser's JS
/// host in production, `wasmi` here — so "run one tick" is the driver's to define, not the reactor's.
impl MomentReactor for JitDriver {
    fn step(&mut self) -> TickOutcome {
        // Refill fuel (the emitted code debits an i64 counter at env[0] and traps when it goes < 0).
        self.memory
            .write(
                &mut self.store,
                ENV_PTR as usize,
                &(1i64 << 52).to_le_bytes(),
            )
            .unwrap();
        let r = self.f_tick.call(
            &mut self.store,
            &[
                Val::I32(WIN_BASE as i32),
                Val::I32(ENV_PTR as i32),
                Val::I64(self.entry_sp as i64),
            ],
            &mut self.results,
        );
        if let Err(e) = r {
            let why = self.reactor().last_trap().to_string();
            panic!("emitted tick trapped: {e} ({why})");
        }
        TickOutcome {
            status: STATUS_OK,
            // The emitted tier writes stdout through the shared `Host`, which this driver does not
            // drain per frame; the reactor cases here compare frames, so it stays empty.
            stdout: Vec::new(),
            frame: self.reactor().take_frame(),
        }
    }
    fn push_key(&self, keycode: i32, pressed: i32) {
        self.reactor().push_key(keycode, pressed);
    }
    fn push_mouse(&self, kind: i32, payload: i32) {
        self.reactor().push_mouse(kind, payload);
    }
    fn moment(&self) -> Option<ReactorMoment> {
        self.reactor().moment()
    }
    fn restore(&mut self, m: &ReactorMoment) -> bool {
        self.reactor_mut().restore(m)
    }
}

/// The same warm ≡ cold gate `reactor_moment.rs` applies to the interpreter reactors, with every frame
/// produced by **emitted wasm**: rewind to a moment, replay, and the frames must match exactly.
fn rewind_replays_the_recorded_future(fixture: &[u8]) {
    let mut d = JitDriver::open(fixture);
    for _ in 0..4 {
        d.hashed();
    }
    let moment: ReactorMoment = d.reactor().moment().expect("the JIT window is capturable");
    let recorded: Vec<u64> = (0..8).map(|_| d.hashed()).collect();

    assert!(d.reactor_mut().restore(&moment), "restore the JIT reactor");
    let replayed: Vec<u64> = (0..8).map(|_| d.hashed()).collect();
    assert_eq!(
        recorded, replayed,
        "a rewound wasm-JIT reactor replays the emitted frames exactly"
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
fn jit_rewind_replays_the_recorded_future() {
    rewind_replays_the_recorded_future(include_bytes!("fixtures/bounce.temen"));
}

/// The grown-heap case on the emitted tier: `life`'s grids live in a malloc heap above the mapped
/// window, written by emitted code.
#[test]
fn jit_rewind_carries_a_heap_grown_above_the_window() {
    rewind_replays_the_recorded_future(include_bytes!("fixtures/life.temen"));
}

/// Input queued through the capability (not the window) rides a JIT-tier moment too — the emitted
/// `tick` drains it through a cross-tier `keyboard.poll` bounce, so the queue is guest state on this
/// tier exactly as on the interpreter.
#[test]
fn jit_moment_carries_queued_input() {
    let mut d = JitDriver::open(include_bytes!("fixtures/bounce.temen"));
    for _ in 0..3 {
        d.hashed();
    }
    d.reactor().push_key(LEFT, 1);
    let moment = d.reactor().moment().expect("capturable");
    let steered: Vec<u64> = (0..4).map(|_| d.hashed()).collect();

    assert!(d.reactor_mut().restore(&moment));
    assert_eq!(
        steered,
        (0..4).map(|_| d.hashed()).collect::<Vec<_>>(),
        "the undrained keypress rides the moment on the emitted tier"
    );
}

// ---- save-states on the emitted tier (#1458) -----------------------------------------------------
//
// A moment lives in engine memory; a save-state has to leave, so it is frozen to a §12 artifact and
// thawed back into a **fresh** reactor. The interpreter's half is gated in `reactor_moment.rs`; these
// are the same properties with every frame produced by emitted wasm, because a save-state the playable
// tier cannot take is not the feature (INVARIANTS #14).

/// Freeze a running JIT reactor, thaw a new one from the artifact, and the frames it goes on to
/// present are the frames the original would have — without `_start` ever running again.
fn a_jit_reactor_freezes_and_thaws_playing(fixture: &[u8]) {
    let m = temen_encode::decode_module(fixture).expect("decode fixture");
    let mut d = JitDriver::open(fixture);
    for _ in 0..5 {
        d.hashed();
    }
    let artifact = d
        .reactor()
        .freeze(&m)
        .expect("freeze the emitted-tier window");
    let recorded: Vec<u64> = (0..8).map(|_| d.hashed()).collect();

    let mut thawed = JitDriver::thaw(fixture, &artifact);
    let replayed: Vec<u64> = (0..8).map(|_| thawed.hashed()).collect();
    assert_eq!(
        recorded, replayed,
        "a thawed wasm-JIT reactor resumes the frozen instant frame for frame"
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
fn a_jit_reactor_freezes_to_an_artifact_and_thaws_playing() {
    a_jit_reactor_freezes_and_thaws_playing(include_bytes!("fixtures/bounce.temen"));
}

/// The grown-heap case: `life`'s grids sit in a malloc heap above the declared window, so the artifact
/// has to carry the emitted tier's whole 16 MiB reservation's committed extent, not the declared prefix.
#[test]
fn a_jit_grown_heap_rides_the_artifact() {
    a_jit_reactor_freezes_and_thaws_playing(include_bytes!("fixtures/life.temen"));
}

/// Input queued but not yet drained rides the artifact, as it rides a moment: the queue is capability
/// state, and #1455's named re-grant seeds it back into the capability the thaw's host mints.
#[test]
fn a_thawed_jit_reactor_keeps_undrained_input() {
    let fixture = include_bytes!("fixtures/bounce.temen");
    let m = temen_encode::decode_module(fixture).expect("decode fixture");
    let mut d = JitDriver::open(fixture);
    for _ in 0..3 {
        d.hashed();
    }
    d.reactor().push_key(LEFT, 1);
    let artifact = d.reactor().freeze(&m).expect("freeze");
    let steered: Vec<u64> = (0..4).map(|_| d.hashed()).collect();

    let mut thawed = JitDriver::thaw(fixture, &artifact);
    assert_eq!(
        steered,
        (0..4).map(|_| thawed.hashed()).collect::<Vec<_>>(),
        "the undrained keypress rides the artifact onto the emitted tier"
    );
}

/// The artifact binds the module it was frozen over, so a thaw against a different guest refuses
/// rather than opening one guest's memory under another's code (INVARIANTS #9c).
#[test]
fn a_jit_save_state_refuses_a_different_module() {
    let bounce = include_bytes!("fixtures/bounce.temen");
    let m = temen_encode::decode_module(bounce).expect("decode fixture");
    let mut d = JitDriver::open(bounce);
    d.hashed();
    let artifact = d.reactor().freeze(&m).expect("freeze");

    let life = temen_encode::decode_module(include_bytes!("fixtures/life.temen")).expect("decode");
    let mut backing = vec![0u8; WIN_SIZE as usize].into_boxed_slice();
    let ptr = backing.as_mut_ptr();
    // SAFETY: `backing` outlives the call; the window is used solely as this reactor's window.
    let refused = unsafe {
        JitOnrampReactor::open_shared_jit(
            &life,
            ptr,
            WIN_SIZE,
            WIN_LOG2,
            false,
            None,
            JitStart::Thaw(&artifact),
        )
    };
    assert!(
        refused.is_err(),
        "a bounce save-state must not thaw under life's code"
    );
}

/// A thawed reactor is an ordinary reactor: it can be frozen again, so save-states chain rather than
/// being a one-way door out of a run.
#[test]
fn a_thawed_jit_reactor_can_be_frozen_again() {
    let fixture = include_bytes!("fixtures/bounce.temen");
    let m = temen_encode::decode_module(fixture).expect("decode fixture");
    let mut d = JitDriver::open(fixture);
    for _ in 0..4 {
        d.hashed();
    }
    let first = d.reactor().freeze(&m).expect("freeze");

    let mut thawed = JitDriver::thaw(fixture, &first);
    for _ in 0..3 {
        thawed.hashed();
    }
    let second = thawed
        .reactor()
        .freeze(&m)
        .expect("re-freeze a thawed reactor");
    let recorded: Vec<u64> = (0..5).map(|_| thawed.hashed()).collect();

    let mut again = JitDriver::thaw(fixture, &second);
    assert_eq!(
        recorded,
        (0..5).map(|_| again.hashed()).collect::<Vec<_>>(),
        "a save-state taken from a thawed reactor is as good as the first"
    );
}

// ---- the tape and the ladder on the emitted tier (#1457 items 3–4) -------------------------------

/// The scripted input for frame `i`, addressed to a timeline so it lands on the tape. Key-*downs*
/// only: `bounce` steers on `(e >> 16) & 1` and ignores releases, so a schedule built from press/release
/// pairs would be one event's worth of input pretending to be four (and a branch off it would not
/// branch).
fn drive(t: &mut ReactorTimeline<JitDriver>, i: usize) {
    match i % 6 {
        0 => t.push_key(RIGHT, 1),
        3 => t.push_key(LEFT, 1),
        _ => {}
    }
}

/// Extend the recording by `frames` frames, feeding the scripted input for each.
fn record(t: &mut ReactorTimeline<JitDriver>, frames: usize) -> Vec<u64> {
    (0..frames)
        .map(|_| {
            let i = t.tick();
            drive(t, i);
            frame_hash(&t.frame().frame.expect("a frame was presented"))
        })
        .collect()
}

/// Play `frames` frames from wherever the timeline stands, replaying the tape rather than branching.
fn play(t: &mut ReactorTimeline<JitDriver>, frames: usize) -> Vec<u64> {
    (0..frames)
        .map(|_| frame_hash(&t.frame().frame.expect("a frame was presented")))
        .collect()
}

/// The scrub gate, with every frame produced by **emitted wasm**: any recorded tick is reachable and
/// replays the frame it originally produced.
///
/// This is INVARIANTS #14 rather than a repeat — a scrub bar the playable tier cannot serve is not the
/// feature. It also answers the question the interpreter cases cannot: emitted code writes the window
/// directly and reads host-maintained globals from an env cell *outside* it, so if any guest-visible
/// state lived in that cell a ladder would rewind the window and leave it behind. It does not.
#[test]
fn a_jit_timeline_seeks_to_any_recorded_tick() {
    const N: usize = 24;
    let mut t = ReactorTimeline::new(
        JitDriver::open(include_bytes!("fixtures/bounce.temen")),
        6,
        4,
    );
    let recorded = record(&mut t, N);
    assert_eq!((t.tick(), t.len()), (N, N));
    assert_eq!(
        t.keyframe_ticks(),
        vec![0, 6, 12, 18],
        "a rung every stride"
    );

    for target in [21usize, 2, 13, 0, 23, 13] {
        assert!(t.seek(target), "tick {target} is on the recording");
        assert_eq!(
            play(&mut t, 1),
            vec![recorded[target]],
            "frame {target} replays identically from a seek on the emitted tier"
        );
    }
    assert!(
        !t.seek(N + 1),
        "a position past the recording is not a position"
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

/// Branching on the emitted tier: steering differently from a rewound position drops the recorded
/// future, and the new one is a recording like any other.
#[test]
fn a_jit_timeline_branches_on_new_input_in_the_past() {
    let mut t = ReactorTimeline::new(
        JitDriver::open(include_bytes!("fixtures/bounce.temen")),
        4,
        4,
    );
    let recorded = record(&mut t, 20);

    assert!(t.seek(8));
    t.push_key(LEFT, 1); // recorded is heading right at tick 8 (RIGHT↓ at 6), so this really turns
    assert_eq!(t.len(), 8, "the abandoned future leaves the tape");
    assert!(t.keyframe_ticks().iter().all(|&k| k <= 8));

    let branch = play(&mut t, 6);
    assert_ne!(branch, recorded[8..14].to_vec(), "the branch diverges");
    assert!(t.seek(8));
    assert_eq!(
        play(&mut t, 6),
        branch,
        "and replays like any other recording"
    );
}
