//! #1366 — the **host-completed cap posture** on the resumable [`bytecode::Vcpu`]: a cap call the
//! *embedder* services asynchronously while the guest sees a plain synchronous `call.cap`.
//!
//! An offloadable handler returns [`OffloadOutcome::Host`] for the slow case; the dispatch mints a
//! host-owned completion id, hands it to the submit hook (the host records the request), and the
//! vCPU parks, surfacing [`VcpuEvent::CapPending`]. The host later calls [`Vcpu::deliver_cap`] and
//! `run`s again — the W4 `StdinPark`/`push_stdin` shape generalized to any cap. This is how a
//! single-threaded embedder (the `wasm32` cdylib, where no offload pool can exist) composes an
//! asynchronous I/O powerbox. Pinned here: the park/deliver round-trip, byte-parity with the pool
//! posture (decline-never-diverge, INVARIANTS 9), the sync face's fail-closed decline (it has no
//! completer — it must not hang), and the §12 "sync ops never pay" pin.
//!
//! #1953 carries the posture to the cooperative multiplex driver ([`bytecode::CoopRun`], the engine
//! behind the browser's release session): a task or fiber parks on the call, the pump surfaces
//! [`CoopEvent::CapPark`] once nothing else can run, and [`CoopRun::deliver_cap`] resumes it —
//! pinned below against the same pool-completed twin, unsliced and sliced, on the root, a spawned
//! thread, and a fiber.
//!
//! #1954 runs the **root program as an emitted leaf**: a root that parks only in declared
//! host-completed caps is offered to the host's leaf emitter, and each call parks in a bounce out of
//! the emitted frames, which the host holds until [`CoopEvent::Resume`]. Served here, as
//! `leaf_tierup.rs` does, by bouncing the entry — the nested interpretation the emitted image's own
//! calls bounce into.

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{self, CoopEvent, LeafOffer, TierUpConfig, VcpuEvent};
use temen_interp::{run_with_host, BoundImport, CapRequests, Host, OffloadOutcome, Trap, Value};

/// Two `HOST_PROC` (iface 13) calls on one handle: op 0 with 5, op 1 with 7; the composite
/// `r0 * 1000 + r1` proves both results landed in the right slots and in order.
const TWO_CALLS: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vfive = i64.const 5
  vr0 = call.cap 13 0 (i64) -> (i64) vh (vfive)
  vseven = i64.const 7
  vr1 = call.cap 13 1 (i64) -> (i64) vh (vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
"#;

/// Both ops answer `arg + 100`: 105 and 107 → 105_107.
const WANT: i64 = 105_107;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module(TWO_CALLS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A recorded host-completed request: `(completion id, the op's argument)`.
type Recorded = Arc<Mutex<Vec<(u64, i64)>>>;

/// The host-completed handler: op 0 answers inline (`Done`); op 1 punts to the *host*
/// (`Host`), whose submit hook records `(id, arg)` so the driver can service it later.
fn host_completed(recorded: &Recorded) -> (Host, i32) {
    let mut host = Host::new();
    let rec = Arc::clone(recorded);
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            let rec = Arc::clone(&rec);
            OffloadOutcome::Host(Box::new(move |id| rec.lock().unwrap().push((id, a))))
        }),
        temen_interp::CapState::Stateless,
    );
    (host, h)
}

/// The pool-completed twin: identical semantics, but op 1 punts a self-contained job to the
/// offload pool (`Offload`) — the posture the parity test compares against.
fn pool_completed() -> (Host, i32) {
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            OffloadOutcome::Offload(Box::new(move || a + 100))
        }),
        temen_interp::CapState::Stateless,
    );
    (host, h)
}

fn program(m: &temen_ir::Module) -> bytecode::VcpuProgram {
    bytecode::VcpuProgram::compile(m).expect("compile")
}

/// A root vCPU over `prog` (borrowed — the test body owns the program) with `host` as its powerbox.
fn vcpu<'p>(prog: &'p bytecode::VcpuProgram, host: Host, h: i32) -> bytecode::Vcpu<'p> {
    bytecode::Vcpu::new_root_reserved_with_powerbox(prog, 0, &[Value::I32(h)], &[], host, 20)
        .expect("build root vcpu")
}

/// Drive a vCPU to completion, servicing every surfaced host-completed request from what the
/// submit hook recorded (`arg + 100`). Returns the results and how many times it parked.
fn drive(vcpu: &mut bytecode::Vcpu, recorded: &Recorded) -> (Vec<Value>, usize) {
    let mut parks = 0;
    loop {
        match vcpu.run() {
            VcpuEvent::Done(v) => return (v, parks),
            VcpuEvent::Trapped(t) => panic!("guest trapped: {t:?}"),
            VcpuEvent::CapPending { id, .. } => {
                parks += 1;
                let a = recorded
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(rid, _)| *rid == id)
                    .map(|(_, a)| *a)
                    .expect("the submit hook recorded this id before the park surfaced");
                vcpu.deliver_cap(id, a + 100);
            }
            _ => panic!("unexpected vcpu event"),
        }
    }
}

/// The round-trip: op 1 parks the vCPU with a surfaced id that the submit hook had already
/// recorded; `deliver_cap` + `run` resumes it with the value in the right slot; the completion
/// store settles (nothing outstanding).
#[test]
fn host_completed_cap_parks_and_resumes() {
    let m = module();
    let prog = program(&m);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let comps = host.completions();
    let mut v = vcpu(&prog, host, h);
    let (r, parks) = drive(&mut v, &recorded);
    assert_eq!(r, vec![Value::I64(WANT)], "both results landed, in order");
    assert_eq!(parks, 1, "exactly the host-completed op parked");
    let rec = recorded.lock().unwrap();
    assert_eq!(rec.len(), 1, "one request recorded");
    assert_eq!(rec[0].1, 7, "the request carried op 1's argument");
    assert_eq!(
        comps.minted(),
        1,
        "one completion id minted (the Done op paid nothing)"
    );
    assert_eq!(comps.outstanding(), 0, "the delivered completion settled");
}

/// A host that answers *inside* its submit hook (the request was serviceable synchronously after
/// all) never parks: the driver finds the result already posted and continues.
#[test]
fn host_completing_inside_submit_does_not_park() {
    let m = module();
    let prog = program(&m);
    let mut host = Host::new();
    let comps = host.completions();
    let comps_for_hook = Arc::clone(&comps);
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            let c = Arc::clone(&comps_for_hook);
            OffloadOutcome::Host(Box::new(move |id| {
                c.complete_host(id, a + 100);
            }))
        }),
        temen_interp::CapState::Stateless,
    );
    let mut v = vcpu(&prog, host, h);
    match v.run() {
        VcpuEvent::Done(r) => assert_eq!(r, vec![Value::I64(WANT)]),
        VcpuEvent::CapPending { .. } => panic!("an already-completed request must not park"),
        _ => panic!("unexpected vcpu event"),
    }
    assert_eq!(comps.outstanding(), 0);
}

/// **Host-completed ≡ pool-completed** (decline-never-diverge): the same guest and semantics
/// produce the identical composite whether the slow op is finished by the embedder (surfaced
/// park + `deliver_cap`) or by the offload pool (the inline wait) — on the `Vcpu`, on the one-shot
/// bytecode face, and on the tree-walk oracle.
#[test]
fn host_completed_matches_pool_completed() {
    let m = module();
    let prog = program(&m);

    // Host-completed on the resumable driver.
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let mut v = vcpu(&prog, host, h);
    let (hosted, _) = drive(&mut v, &recorded);

    // Pool-completed on the same driver (inline wait, the pre-#1366 posture).
    let (host, h) = pool_completed();
    let mut v = vcpu(&prog, host, h);
    let pooled = match v.run() {
        VcpuEvent::Done(r) => r,
        _ => panic!("pool posture never parks the session driver"),
    };

    // Pool-completed on the one-shot bytecode face (the job runs inline) and the oracle.
    let (mut host, h) = pool_completed();
    let mut fuel = 2_000_000_000u64;
    let oneshot =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
            .expect("in subset")
            .expect("no trap");
    let (mut host, h) = pool_completed();
    let mut fuel = 2_000_000_000u64;
    let oracle = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    host.quiesce_pool();

    assert_eq!(hosted, vec![Value::I64(WANT)]);
    assert_eq!(
        hosted, pooled,
        "host-completed ≡ pool-completed on the Vcpu"
    );
    assert_eq!(hosted, oneshot, "≡ the one-shot bytecode face");
    assert_eq!(hosted, oracle, "≡ the tree-walk oracle");
}

/// The one-shot (sync) face has no completer for a host-completed punt: it must **decline**
/// with `CapFault`, fail-closed — never block forever.
#[test]
fn host_completed_declines_on_the_sync_face() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let mut fuel = 2_000_000_000u64;
    let r = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
        .expect("in subset");
    assert!(
        matches!(r, Err(Trap::CapFault)),
        "a host-completed punt on the sync face declines with CapFault"
    );
    assert!(
        recorded.lock().unwrap().is_empty(),
        "the declined call never reached the host's submit hook"
    );
}

/// §12 pin, host-completed form: a handler that always answers inline touches none of the
/// parking machinery — no id minted, nothing outstanding.
#[test]
fn sync_ops_never_pay_for_the_host_posture() {
    let m = module();
    let prog = program(&m);
    let mut host = Host::new();
    let comps = host.completions();
    let h = host.grant_host_proc_offloadable(
        Box::new(|_op, args| OffloadOutcome::Done(Ok(vec![*args.first().unwrap_or(&0) + 100]))),
        temen_interp::CapState::Stateless,
    );
    let mut v = vcpu(&prog, host, h);
    match v.run() {
        VcpuEvent::Done(r) => assert_eq!(r, vec![Value::I64(WANT)]),
        _ => panic!("an all-inline handler never parks"),
    }
    assert_eq!(comps.minted(), 0, "no completion id was ever minted");
    assert_eq!(comps.outstanding(), 0);
}

/// The tree-walk oracle has no driver that could surface a host-completed park to an embedder,
/// so it too must **decline** with `CapFault` — bounded fuel turns a regression into a loud
/// OutOfFuel rather than a hang.
#[test]
fn host_completed_declines_on_the_oracle() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let mut fuel = 2_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
    assert!(
        matches!(r, Err(Trap::CapFault)),
        "the oracle declines a host-completed punt with CapFault, got {r:?}"
    );
    assert!(recorded.lock().unwrap().is_empty());
}

// --- #1953: the cooperative multiplex driver (`CoopRun`) -----------------------------------------

/// Fuel for the cooperative runs: bounded, so a lost wake is a loud `OutOfFuel`, never a hang.
const FUEL: u64 = 2_000_000_000;

/// The root spawns a thread that makes the host-completed call (op 1, arg 7) while the root makes
/// the inline one (op 0, arg 5), then joins it: the same composite, `105 * 1000 + 107`, with the
/// park on a spawned thread. The thread gets the handle as its argument.
const THREADED: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vh: i32) {
  vz = i64.const 0
  vh64 = i64.extend_i32_u vh
  vt = thread.spawn 1 vz vh64
  vfive = i64.const 5
  vr0 = call.cap 13 0 (i64) -> (i64) vh (vfive)
  vr1 = thread.join vt
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  return vr
  }
}
"#;

/// The root runs a fiber that makes the host-completed call (op 1, arg 7) and `cont.resume.block`s
/// it, so the root idles on the parked fiber: status `1` (returned) and value 107 → `10_107`.
const FIBER: &str = r#"memory 16 shadow 16448 65536
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vh64 = i64.extend_i32_u v0
  vs, vv = cont.resume.block vk vh64
  vk4 = i64.const 10000
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk4
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vseven = i64.const 7
  vr = call.cap 13 1 (i64) -> (i64) vh (vseven)
  return vr
  }
}
"#;

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn coop(m: &temen_ir::Module, host: Host, h: i32) -> bytecode::CoopRun {
    bytecode::CoopRun::new(m, 0, &[Value::I32(h)], FUEL, host, None)
        .expect("in subset")
        .expect("entry in range")
}

/// Pump a `CoopRun` to the end — unsliced, or in `slice`-op slices — answering every surfaced park
/// from what the submit hook recorded (`arg + 100`). Returns the results and the parks surfaced.
fn drive_coop(
    run: &mut bytecode::CoopRun,
    recorded: &Recorded,
    slice: Option<u64>,
) -> (Vec<Value>, usize) {
    let mut parks = 0;
    loop {
        let ev = match slice {
            Some(n) => run.run_for(n),
            None => run.run(),
        };
        match ev {
            CoopEvent::Done(v) => return (v, parks),
            CoopEvent::Paused => {}
            CoopEvent::CapPark { id } => {
                parks += 1;
                let a = recorded
                    .lock()
                    .unwrap()
                    .iter()
                    .find(|(rid, _)| *rid == id)
                    .map(|(_, a)| *a)
                    .expect("the submit hook recorded this id before the park surfaced");
                assert!(
                    run.deliver_cap(id, a + 100),
                    "the surfaced id is deliverable"
                );
            }
            CoopEvent::Trapped(t) => panic!("guest trapped: {t:?}"),
            _ => panic!("unexpected coop event"),
        }
    }
}

/// The cooperative round-trip: the root parks on op 1, the pump surfaces the id the submit hook
/// recorded, a foreign id is refused, and the delivered value lands in the call's slot.
#[test]
fn coop_run_parks_on_a_host_completed_cap_and_resumes() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let comps = host.completions();
    let mut run = coop(&m, host, h);
    let id = match run.run() {
        CoopEvent::CapPark { id } => id,
        _ => panic!("the host-completed call parks the run"),
    };
    assert_eq!(recorded.lock().unwrap().clone(), vec![(id, 7)]);
    assert!(
        matches!(run.run(), CoopEvent::CapPark { id: again } if again == id),
        "pumping while parked surfaces the same call again"
    );
    assert!(!run.deliver_cap(id + 1, 0), "a foreign id is refused");
    assert!(run.deliver_cap(id, 107));
    assert!(
        !run.deliver_cap(id, 107),
        "a delivered call is no longer outstanding"
    );
    assert!(matches!(run.run(), CoopEvent::Done(ref v) if *v == vec![Value::I64(WANT)]));
    assert_eq!(comps.outstanding(), 0, "the delivered completion settled");
}

/// **Host-completed ≡ pool-completed on `CoopRun`**, for each shape of parked caller — the root, a
/// spawned thread, and a fiber idled on by a blocking resume — unsliced and pumped in 1-op slices,
/// against the pool posture on the same driver and on the tree-walk oracle.
#[test]
fn coop_host_completed_matches_pool_completed() {
    for (name, m, want) in [
        ("root", module(), WANT),
        ("thread", parse(THREADED), WANT),
        ("fiber", parse(FIBER), 10_107),
    ] {
        for slice in [None, Some(1)] {
            let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
            let (host, h) = host_completed(&recorded);
            let comps = host.completions();
            let mut run = coop(&m, host, h);
            let (hosted, parks) = drive_coop(&mut run, &recorded, slice);
            assert_eq!(hosted, vec![Value::I64(want)], "{name} {slice:?}");
            assert_eq!(
                parks, 1,
                "{name} {slice:?}: exactly the host-completed call parked"
            );
            assert_eq!(comps.outstanding(), 0, "{name} {slice:?}");
        }
        let (host, h) = pool_completed();
        let mut run = coop(&m, host, h);
        let pooled = match run.run() {
            CoopEvent::Done(r) => r,
            _ => panic!("{name}: the pool posture never parks the run"),
        };
        let (mut host, h) = pool_completed();
        let mut fuel = FUEL;
        let oracle = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
        host.quiesce_pool();
        assert_eq!(
            pooled,
            vec![Value::I64(want)],
            "{name}: pool ≡ host on CoopRun"
        );
        assert_eq!(
            oracle,
            vec![Value::I64(want)],
            "{name}: ≡ the tree-walk oracle"
        );
    }
}

/// Only an embedder-driven `CoopRun` admits host-completed calls: the native cooperative driver
/// (the one-shot face with threads) has no one to answer one, so it declines with `CapFault`.
#[test]
fn host_completed_declines_on_the_native_coop_driver() {
    let m = parse(THREADED);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let mut fuel = FUEL;
    let r = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
        .expect("in subset");
    assert!(matches!(r, Err(Trap::CapFault)), "declines, got {r:?}");
    assert!(recorded.lock().unwrap().is_empty());
}

// --- #1954: the root program as an emitted leaf ----------------------------------------------------

/// The root calls the declared cap `ping` twice, through an import: `ping(5) * 1000 + ping(7)`.
const ROOT_PINGS: &str = r#"memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vfive = i64.const 5
  vr0 = call.import 0 (vfive)
  vseven = i64.const 7
  vr1 = call.import 0 (vseven)
  vk = i64.const 1000
  vm = i64.mul vr0 vk
  vsum = i64.add vm vr1
  return vsum
  }
}
"#;

/// The same `ping`, reached from function 1, which tiers up as a region (not a leaf): `ping(7)`.
const REGION_PING: &str = r#"memory 16
import 0 "ping" (i64) -> (i64)
func () -> (i64) {
block 0 () {
  vr = call 1 ()
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vseven = i64.const 7
  vr = call.import 0 (vseven)
  return vr
  }
}
"#;

/// A host granting `ping` as a declared host-completed cap, bound to import 0.
fn declared_ping(m: &temen_ir::Module) -> (Host, CapRequests) {
    let mut host = Host::new();
    let requests = CapRequests::default();
    let seams = host.grant_declared_host_caps(&m.imports, &["ping".to_string()], &requests);
    assert_eq!(seams.len(), 1, "ping is imported, so granted");
    host.set_import_bindings(vec![BoundImport::required(
        temen_ir::cap_id::HOST_PROC,
        0,
        seams[0].1,
    )]);
    (host, requests)
}

/// Offers seen by a leaf emitter: `(module, entry, parks)`.
type Offers = Arc<Mutex<Vec<(usize, u32, bool)>>>;

fn leaf_config(offers: &Offers, take: bool) -> TierUpConfig {
    let offers = Arc::clone(offers);
    TierUpConfig {
        eligible: Arc::from([]),
        page_checked: false,
        leaf: Some(Arc::new(move |o: &LeafOffer| {
            offers.lock().unwrap().push((o.module, o.entry, o.parks));
            take
        })),
    }
}

/// What a run of `m` did: its result, the parks it surfaced, its tier-ups, and its resumes.
#[derive(Debug, PartialEq)]
struct Ran {
    end: Result<Vec<Value>, Trap>,
    parks: usize,
    tierups: usize,
    resumes: usize,
}

/// Pump `run` to the end, answering each park from the request its proc filed (`arg + 100`) and
/// serving each tier-up by bouncing its function, as `leaf_tierup.rs` does.
fn drive_leaf(mut run: bytecode::CoopRun, requests: &CapRequests) -> Ran {
    let (mut parks, mut tierups, mut resumes) = (0, 0, 0);
    let end = loop {
        match run.run() {
            CoopEvent::TierUp { func, argv, .. } => {
                tierups += 1;
                let mut io = argv.to_vec();
                io.resize(io.len().max(1), 0);
                match run.bounce(func, &mut io, None) {
                    Ok(Some(n)) => run.deliver_tierup(&io[..n]),
                    Ok(None) => {}
                    Err(t) => run.deliver_tierup_trap(t),
                }
            }
            CoopEvent::Resume { results } => {
                resumes += 1;
                run.deliver_tierup(&results);
            }
            CoopEvent::CapPark { id } => {
                parks += 1;
                let arg = requests.lock().unwrap().remove(&id).expect("filed").args[0];
                assert!(run.deliver_cap(id, arg + 100));
            }
            CoopEvent::Done(v) => break Ok(v),
            CoopEvent::Trapped(t) => break Err(t),
            _ => panic!("unexpected coop event"),
        }
    };
    Ran {
        end,
        parks,
        tierups,
        resumes,
    }
}

/// A run of `m` over a window **reserved at its declared size**. A root leaf needs a window emitted
/// code can address flat: unix's lazy `mmap` gives the default 1-TiB reservation one, but elsewhere
/// (Windows, wasm) that reservation is a sparse table, and only a reservation within the flat-buffer
/// cap is backed flat — so a run at the default runs its root interpreted there, by design.
fn coop_with(m: &temen_ir::Module, host: Host, tierup: Option<TierUpConfig>) -> bytecode::CoopRun {
    let declared = m.memory.expect("a window").size_log2;
    bytecode::CoopRun::new_reserved(m, 0, &[], FUEL, host, tierup, &[], declared)
        .expect("in subset")
        .expect("entry in range")
}

/// **The root runs as an emitted leaf and parks in it.** A root whose only parking calls are declared
/// host-completed caps is offered at its entry (module 0, parks = true), tiers up there, and each
/// call parks in a bounce out of the emitted frames: the park surfaces, the answer is delivered, and
/// `Resume` hands the frames the entry's results. The run ends exactly as the interpreted run does,
/// with the same parks.
#[test]
fn the_root_runs_as_an_emitted_leaf_and_parks_on_declared_caps() {
    let m = parse(ROOT_PINGS);
    let want = Ok(vec![Value::I64(WANT)]);

    let (host, requests) = declared_ping(&m);
    let interpreted = drive_leaf(coop_with(&m, host, None), &requests);
    assert_eq!(
        interpreted,
        Ran {
            end: want.clone(),
            parks: 2,
            tierups: 0,
            resumes: 0
        },
        "interpreted"
    );

    let offers: Offers = Arc::default();
    let (host, requests) = declared_ping(&m);
    let leaf = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, true))),
        &requests,
    );
    assert_eq!(
        offers.lock().unwrap().clone(),
        vec![(0, 0, true)],
        "the root, at its entry"
    );
    assert_eq!(
        leaf,
        Ran {
            end: want.clone(),
            parks: 2,
            tierups: 1,
            resumes: 1
        },
        "one tier-up at the entry, two parks inside it, one resume of its frames"
    );

    // Every constructor that holds the image offers it: `CoopRun::new` too, whose default 1-TiB
    // reservation only unix's lazy `mmap` backs flat (elsewhere that root interprets, by design).
    #[cfg(unix)]
    {
        let offers: Offers = Arc::default();
        let (host, requests) = declared_ping(&m);
        let run = bytecode::CoopRun::new(&m, 0, &[], FUEL, host, Some(leaf_config(&offers, true)))
            .expect("in subset")
            .expect("entry in range");
        let ran = drive_leaf(run, &requests);
        assert_eq!(
            offers.lock().unwrap().clone(),
            vec![(0, 0, true)],
            "CoopRun::new"
        );
        assert_eq!(
            (ran.end, ran.tierups, ran.resumes),
            (want, 1, 1),
            "CoopRun::new"
        );
    }
}

/// An emitter that declines the root (a host that cannot suspend its frames) leaves it interpreted,
/// with the same ending.
#[test]
fn a_declined_root_leaf_runs_interpreted() {
    let m = parse(ROOT_PINGS);
    let offers: Offers = Arc::default();
    let (host, requests) = declared_ping(&m);
    let ran = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, false))),
        &requests,
    );
    assert_eq!(offers.lock().unwrap().len(), 1, "offered once");
    assert_eq!(
        ran,
        Ran {
            end: Ok(vec![Value::I64(WANT)]),
            parks: 2,
            tierups: 0,
            resumes: 0
        }
    );
}

/// A tier-up **region** (not a leaf: nothing holds its frames) that reaches a host-completed call
/// declines with `CapFault` — it never waits on an id only the embedder could complete.
#[test]
fn a_host_completed_call_in_a_tierup_region_declines() {
    let m = parse(REGION_PING);
    let (host, requests) = declared_ping(&m);
    let tierup = TierUpConfig {
        eligible: Arc::from([false, true]),
        page_checked: false,
        leaf: None,
    };
    let ran = drive_leaf(coop_with(&m, host, Some(tierup)), &requests);
    assert_eq!(
        ran,
        Ran {
            end: Err(Trap::CapFault),
            parks: 0,
            tierups: 1,
            resumes: 0
        }
    );
}

// --- #1954: what the root-leaf predicate admits -------------------------------------------------

/// A root that writes to stdout and exits through its imports — what every on-ramp C program
/// does — and imports a stdin read it never makes.
const ROOT_IO: &str = r#"memory 16
import 0 "write" (i64, i64) -> (i64)
import 1 "read" (i64, i64) -> (i64)
import 2 "exit" (i32) -> ()
data 16384 "hi\n"
func () -> (i64) {
block 0 () {
  vbuf = i64.const 16384
  vn = i64.const 3
  vw = call.import 0 (vbuf, vn)
  vseven = i32.const 7
  call.import 2 (vseven)
  unreachable
  }
}
"#;

/// A root that serves through `svc.wait` (self op 10), which parks until a caller arrives.
const ROOT_SVC_WAIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

/// The §3e powerbox prefix bound to `write`/`read`/`exit`, as an on-ramp host binds them.
fn io_host(stdin_blocking: bool) -> Host {
    let mut host = Host::new();
    let [stdout, stdin, exit, _, _] = host.grant_powerbox_prefix(1 << 16);
    host.set_import_bindings(vec![
        BoundImport::required(temen_ir::cap_id::STREAM, 1, stdout),
        BoundImport::required(temen_ir::cap_id::STREAM, 0, stdin),
        BoundImport::required(temen_ir::cap_id::EXIT, 0, exit),
    ]);
    host.set_stdin_blocking(stdin_blocking);
    host
}

/// An `exit` never parks, and a stream import parks exactly as the same op inline: without a pipe
/// or a blocking stdin the root cannot park, so it is offered (`parks = false`) and runs whole; with
/// a blocking stdin its read can park, so it is offered only to a host that suspends.
#[test]
fn a_root_writing_and_exiting_through_its_imports_is_offered() {
    let m = parse(ROOT_IO);
    let none = CapRequests::default();

    let offers: Offers = Arc::default();
    let ran = drive_leaf(
        coop_with(&m, io_host(false), Some(leaf_config(&offers, true))),
        &none,
    );
    assert_eq!(offers.lock().unwrap().clone(), vec![(0, 0, false)]);
    assert_eq!((ran.end, ran.tierups), (Err(Trap::Exit(7)), 1));

    let offers: Offers = Arc::default();
    let _ = coop_with(&m, io_host(true), Some(leaf_config(&offers, false)));
    assert_eq!(
        offers.lock().unwrap().clone(),
        vec![(0, 0, true)],
        "a blocking stdin read can park"
    );
}

/// A self op that parks (`svc.wait`) keeps the root off the emitted tier: it is never offered.
#[test]
fn a_root_that_can_park_in_a_self_op_is_not_offered() {
    let m = parse(ROOT_SVC_WAIT);
    let offers: Offers = Arc::default();
    let _ = coop_with(&m, Host::new(), Some(leaf_config(&offers, true)));
    assert!(offers.lock().unwrap().is_empty());
}
