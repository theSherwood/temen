//! #1366 — the **host-answered cap posture** on the resumable [`bytecode::Vcpu`]: a cap call the
//! *embedder* answers later while the guest sees a plain synchronous `call.cap`.
//!
//! An offloadable handler returns [`OffloadOutcome::Host`] for the slow case; the dispatch mints a
//! completion id, hands the call's [`Completer`] to the submit hook (the host records the request),
//! and the vCPU parks, surfacing [`VcpuEvent::CapPending`]. The host answers it with the completer
//! and `run`s again — the W4 `StdinPark`/`push_stdin` shape generalized to any cap. This is how a
//! single-threaded embedder (the `wasm32` cdylib, where no offload pool can exist) composes an
//! asynchronous I/O powerbox. Pinned here: the park/answer round-trip, byte-parity with the pool
//! posture (decline-never-diverge, INVARIANTS 9), the sync face's fail-closed decline (no caller
//! there can park — it must not hang), and the §12 "sync ops never pay" pin.
//!
//! #1953 carries the posture to the cooperative multiplex driver ([`bytecode::CoopRun`], the engine
//! behind the browser's release session): a task or fiber parks on the call, the pump surfaces
//! [`CoopEvent::CapPark`] once nothing else can run, and the next pump after the answer resumes it
//! — pinned below against the same pool-completed twin, unsliced and sliced, on the root, a spawned
//! thread, and a fiber. #2495 carries it to the drivers no embedder pumps — the tree-walker and the
//! native cooperative driver — where the embedder answers from a thread of its own.
//!
//! #1954 runs the **root program as an emitted leaf**: a root that parks only in declared
//! host-completed caps is offered to the host's leaf emitter, and each call parks in a bounce out of
//! the emitted frames, which the host holds until [`CoopEvent::Resume`]. Served here, as
//! `leaf_tierup.rs` does, by bouncing the entry — the nested interpretation the emitted image's own
//! calls bounce into.

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{self, CoopEvent, LeafOffer, ProgramEmit, TierUpConfig, VcpuEvent};
use temen_interp::{
    run_with_host, BoundImport, CapRequests, Completer, Host, OffloadOutcome, Trap, Value,
};

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

/// The host-answered requests the submit hook filed and nobody has answered yet: each call's
/// completer and the op's argument.
type Recorded = Arc<Mutex<Vec<(Completer, i64)>>>;

/// The filed requests, as `(completion id, argument)`.
fn filed(recorded: &Recorded) -> Vec<(u64, i64)> {
    let g = recorded.lock().unwrap();
    g.iter().map(|(c, a)| (c.id(), *a)).collect()
}

/// Answer the filed request `id` with its argument + 100: the argument, or `None` if none is
/// filed under `id`.
fn answer(recorded: &Recorded, id: u64) -> Option<i64> {
    let mut g = recorded.lock().unwrap();
    let at = g.iter().position(|(c, _)| c.id() == id)?;
    let (c, a) = g.remove(at);
    drop(g);
    c.complete(a + 100);
    Some(a)
}

/// Answer the next `n` requests from a thread of the embedder's own, each as it is filed: what a
/// driver no embedder pumps needs, since it returns only when the run ends.
fn answer_from_a_thread(recorded: &Recorded, n: usize) -> std::thread::JoinHandle<()> {
    let recorded = Arc::clone(recorded);
    std::thread::spawn(move || {
        for _ in 0..n {
            let id = loop {
                if let Some(&(id, _)) = filed(&recorded).first() {
                    break id;
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            answer(&recorded, id);
        }
    })
}

/// The host-answered handler: op 0 answers inline (`Done`); op 1 punts to the *host*
/// (`Host`), whose submit hook files the call's completer and argument for the driver's embedder.
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
            OffloadOutcome::Host(Box::new(move |c| rec.lock().unwrap().push((c, a))))
        }),
        None,
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
        None,
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

/// Drive a vCPU to completion, answering every surfaced host-answered request from what the
/// submit hook filed (`arg + 100`). Returns the results and the argument of each call it answered.
fn drive(vcpu: &mut bytecode::Vcpu, recorded: &Recorded) -> (Vec<Value>, Vec<i64>) {
    let mut answered = Vec::new();
    loop {
        match vcpu.run() {
            VcpuEvent::Done(v) => return (v, answered),
            VcpuEvent::Trapped(t) => panic!("guest trapped: {t:?}"),
            VcpuEvent::CapPending { id, .. } => {
                let a = answer(recorded, id)
                    .expect("the submit hook filed this id before the park surfaced");
                answered.push(a);
            }
            _ => panic!("unexpected vcpu event"),
        }
    }
}

/// The round-trip: op 1 parks the vCPU with a surfaced id that the submit hook had already
/// filed; answering it and `run` resumes it with the value in the right slot; the completion
/// store settles (nothing outstanding).
#[test]
fn host_completed_cap_parks_and_resumes() {
    let m = module();
    let prog = program(&m);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let comps = host.completions();
    let mut v = vcpu(&prog, host, h);
    let (r, answered) = drive(&mut v, &recorded);
    assert_eq!(r, vec![Value::I64(WANT)], "both results landed, in order");
    assert_eq!(
        answered,
        vec![7],
        "exactly the host-answered op parked, with op 1's argument"
    );
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
    let h = host.grant_host_proc_offloadable(
        Box::new(move |op, args| {
            let a = *args.first().unwrap_or(&0);
            if op == 0 {
                return OffloadOutcome::Done(Ok(vec![a + 100]));
            }
            OffloadOutcome::Host(Box::new(move |c: Completer| c.complete(a + 100)))
        }),
        None,
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

/// **Host-answered ≡ pool-completed** (decline-never-diverge): the same guest and semantics
/// produce the identical composite whether the slow op is answered by the embedder (surfaced
/// park + completer) or by the offload pool (the inline wait) — on the `Vcpu`, on the one-shot
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

/// #2495 — where its caller could never park on it, a host-answered call **declines** with
/// `CapFault` before its embedder sees it, fail-closed — never blocking forever: in a durable run,
/// whose freeze has no cap park to re-derive, and on the tree-walker under a recording host, which
/// tapes no delivered value.
#[test]
fn a_host_answered_call_declines_where_its_caller_cannot_park() {
    declines(
        "a durable run on the tree-walker",
        |h| h.set_durable(true),
        true,
    );
    declines(
        "a durable run on the native coop driver",
        |h| h.set_durable(true),
        false,
    );
    declines(
        "a recording host on the tree-walker",
        |h| h.record_caps(),
        true,
    );
}

/// A run of [`module`] over a host `prepare`d so, on the tree-walker (`oracle`) or the native coop
/// driver, declines the host-answered call before the submit hook sees it.
fn declines(what: &str, prepare: fn(&mut Host), oracle: bool) {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    prepare(&mut host);
    let r = bounded(move || {
        let (args, mut fuel) = ([Value::I32(h)], 2_000_000_000u64);
        if oracle {
            run_with_host(&m, 0, &args, &mut fuel, &mut host)
        } else {
            bytecode::compile_and_run_with_host(&m, 0, &args, &mut fuel, &mut host)
                .expect("in subset")
        }
    });
    assert_eq!(r, Err(Trap::CapFault), "{what}");
    assert!(
        filed(&recorded).is_empty(),
        "{what}: the declined call never reached the host's submit hook"
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
        None,
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

/// #2495 — the tree-walk oracle, which no embedder pumps, parks the caller of a host-answered call
/// as it parks a pool call's, and the embedder answers from a thread of its own.
#[test]
fn the_oracle_parks_a_host_answered_call_until_its_embedder_answers() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let comps = host.completions();
    let answering = answer_from_a_thread(&recorded, 1);
    let r = bounded(move || {
        let mut fuel = 2_000_000u64;
        run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
    });
    answering.join().unwrap();
    assert_eq!(r, Ok(vec![Value::I64(WANT)]));
    assert_eq!(comps.outstanding(), 0, "the answered call settled");
}

/// #2495 — a completer dropped unanswered answers its call `CAP_REVOKED` (`-9`): a call its host
/// can no longer answer never hangs. The Vcpu surfaces the park; dropping the request
/// lets the run end, the call having answered `-9` (`105 * 1000 - 9`).
#[test]
fn a_completer_dropped_unanswered_answers_cap_revoked() {
    let m = module();
    let prog = program(&m);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let mut v = vcpu(&prog, host, h);
    assert!(matches!(v.run(), VcpuEvent::CapPending { .. }));
    recorded.lock().unwrap().clear();
    match v.run() {
        VcpuEvent::Done(r) => assert_eq!(r, vec![Value::I64(105 * 1000 - 9)]),
        _ => panic!("the revoked call answers"),
    }
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
/// from what the submit hook filed (`arg + 100`). Returns the results and the parks surfaced.
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
                answer(recorded, id)
                    .expect("the submit hook filed this id before the park surfaced");
            }
            CoopEvent::Trapped(t) => panic!("guest trapped: {t:?}"),
            _ => panic!("unexpected coop event"),
        }
    }
}

/// The cooperative round-trip: the root parks on op 1, the pump surfaces the id the submit hook
/// filed, and the answered value lands in the call's slot.
#[test]
fn coop_run_parks_on_a_host_completed_cap_and_resumes() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = host_completed(&recorded);
    let comps = host.completions();
    let mut run = coop(&m, host, h);
    let id = match run.run() {
        CoopEvent::CapPark { id } => id,
        _ => panic!("the host-answered call parks the run"),
    };
    assert_eq!(filed(&recorded), vec![(id, 7)]);
    assert!(
        matches!(run.run(), CoopEvent::CapPark { id: again } if again == id),
        "pumping while parked surfaces the same call again"
    );
    answer(&recorded, id);
    assert!(matches!(run.run(), CoopEvent::Done(ref v) if *v == vec![Value::I64(WANT)]));
    assert_eq!(comps.outstanding(), 0, "the answered completion settled");
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

/// #2495 — the native cooperative driver (the one-shot face with threads), which no embedder
/// pumps, parks a thread on a host-answered call as on a pool call, and the embedder answers from a
/// thread of its own.
#[test]
fn the_native_coop_driver_parks_a_host_answered_call_until_its_embedder_answers() {
    let m = parse(THREADED);
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = host_completed(&recorded);
    let answering = answer_from_a_thread(&recorded, 1);
    let r = bounded(move || {
        let mut fuel = FUEL;
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host)
            .expect("in subset")
    });
    answering.join().unwrap();
    assert_eq!(r, Ok(vec![Value::I64(WANT)]));
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
        root: ProgramEmit::default(),
        leaf: Some(Arc::new(move |o: &LeafOffer| {
            offers.lock().unwrap().push((o.module, o.entry, o.parks));
            take
        })),
        modules: None,
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
                let request = requests.lock().unwrap().remove(&id).expect("filed");
                request.completer.complete(request.args[0] + 100);
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
/// code can address flat: unix's lazy `mmap` gives the default 128 GiB reservation one, but elsewhere
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

    // Every constructor that holds the image offers it: `CoopRun::new` too, whose default 128 GiB
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
        root: ProgramEmit {
            eligible: Arc::from([false, true]),
            page_checked: false,
        },
        leaf: None,
        modules: None,
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

/// A root that reads its stdin with an inline `call.cap` (stream op 0) rather than an import.
const ROOT_INLINE_READ: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vh = i32.const 1
  vbuf = i64.const 16384
  vn = i64.const 8
  vr = call.cap 0 0 (i64, i64) -> (i64) vh (vbuf, vn)
  return vr
  }
}
"#;

/// #1414 S4-4a — a kept program reads its image's code once, and each offer still asks the run's
/// host. One program, compiled once and run over a host whose stdin cannot block, then over one
/// whose can, then the first kind again, is offered `parks = false`, `true`, `false`: the inline
/// read parks only on the blocking stdin.
#[test]
fn a_kept_programs_offers_ask_each_runs_host() {
    let m = parse(ROOT_INLINE_READ);
    let program = bytecode::compile_reserved(&m).expect("in subset");
    let declared = m.memory.expect("a window").size_log2;
    let offer = |stdin_blocking: bool| {
        let offers: Offers = Arc::default();
        let mut host = Host::new();
        host.set_stdin_blocking(stdin_blocking);
        let _ = bytecode::CoopRun::new_reserved_over_compiled(
            &m,
            Arc::clone(&program),
            0,
            &[],
            FUEL,
            host,
            Some(leaf_config(&offers, false)),
            &[],
            declared,
            None,
        )
        .expect("entry in range");
        let seen = offers.lock().unwrap().clone();
        seen
    };
    assert_eq!(offer(false), vec![(0, 0, false)]);
    assert_eq!(
        offer(true),
        vec![(0, 0, true)],
        "a blocking stdin read can park"
    );
    assert_eq!(offer(false), vec![(0, 0, false)], "and this host's cannot");
}

/// A self op that parks (`svc.wait`) keeps the root off the emitted tier: it is never offered.
#[test]
fn a_root_that_can_park_in_a_self_op_is_not_offered() {
    let m = parse(ROOT_SVC_WAIT);
    let offers: Offers = Arc::default();
    let _ = coop_with(&m, Host::new(), Some(leaf_config(&offers, true)));
    assert!(offers.lock().unwrap().is_empty());
}

/// A root that calls a host capability inline: a `call.cap` whose handle `h` is an operand, as
/// chibicc reaches its in-memory file system.
fn root_host_call(h: i32) -> temen_ir::Module {
    parse(&format!(
        "memory 16
func () -> (i64) {{
block 0 () {{
  vh = i32.const {h}
  va = i64.const 5
  vr = call.cap 13 0 (i64) -> (i64) vh (va)
  return vr
  }}
}}
"
    ))
}

/// An inline host-capability call parks as any of the host's host capabilities could (#1414 S4-3b).
/// When each answers synchronously and no personality can ask one to park, the root cannot park:
/// it is offered (`parks = false`) and runs whole. An offloadable capability's call can punt, so a
/// host holding one keeps the root off the emitted tier.
#[test]
fn a_root_calling_a_host_capability_inline_is_offered_when_none_can_park() {
    let mut host = Host::new();
    let h = host.grant_host_proc(
        Box::new(|_op, args, _mem, _| Ok(vec![args[0] + 100])),
        temen_interp::CapState::Stateless,
    );
    let m = root_host_call(h);
    let offers: Offers = Arc::default();
    let ran = drive_leaf(
        coop_with(&m, host, Some(leaf_config(&offers, true))),
        &CapRequests::default(),
    );
    assert_eq!(offers.lock().unwrap().clone(), vec![(0, 0, false)]);
    assert_eq!((ran.end, ran.tierups), (Ok(vec![Value::I64(105)]), 1));

    let (host, h) = pool_completed();
    let m = root_host_call(h);
    let offers: Offers = Arc::default();
    let _ = coop_with(&m, host, Some(leaf_config(&offers, true)));
    assert!(
        offers.lock().unwrap().is_empty(),
        "an offloadable capability's call can punt"
    );
}

// --- #1722: a host-completed call whose reply cannot park -----------------------------------------

/// One host-completed call (op 1) declared with a reply that is not a single `i64`: C's `int`, as a
/// program declares an extern its embedder grants as a host cap, or no reply at all. A parked call
/// is delivered into one `i64` slot, so neither can park; each declines, never waiting on an answer
/// only the host could give.
fn narrow_call(reply: &str) -> temen_ir::Module {
    let (ret, sig) = match reply {
        "i32" => ("  vr64 = i64.extend_i32_s vr\n  return vr64\n", "(i32)"),
        _ => ("  vz = i64.const 0\n  return vz\n", "()"),
    };
    let call = if reply == "i32" {
        format!("  vr = call.cap 13 1 (i64) -> {sig} vh (vseven)\n")
    } else {
        format!("  call.cap 13 1 (i64) -> {sig} vh (vseven)\n")
    };
    parse(&format!(
        "memory 16\nfunc (i32) -> (i64) {{\nblock 0 (vh: i32) {{\n  vseven = i64.const 7\n{call}{ret}  }}\n}}\n"
    ))
}

/// Run `f` on its own thread, for at most 30 s: a call that waits on a completion only its host could
/// give would hang the test rather than fail it.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .expect("the run waited on a host-completed call that cannot park")
}

/// The trap each driver that parks host-answered calls ends a run of `m` with.
fn on_each_parking_driver(m: &temen_ir::Module) -> Vec<(&'static str, Option<Trap>)> {
    ["vcpu", "coop", "debug", "oracle"]
        .into_iter()
        .map(|driver| {
            let m = m.clone();
            let end = bounded(move || {
                let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
                let (mut host, h) = host_completed(&recorded);
                match driver {
                    "oracle" => {
                        run_with_host(&m, 0, &[Value::I32(h)], &mut 2_000_000, &mut host).err()
                    }
                    "vcpu" => match vcpu(&program(&m), host, h).run() {
                        VcpuEvent::Trapped(t) => Some(t),
                        _ => None,
                    },
                    "coop" => match coop(&m, host, h).run() {
                        CoopEvent::Trapped(t) => Some(t),
                        _ => None,
                    },
                    _ => {
                        let args = [Value::I32(h)];
                        let mut run =
                            bytecode::ScheduledDebugRun::new_with_host(&m, 0, &args, host)
                                .expect("in subset");
                        match run.run_until_stop(&mut 2_000_000) {
                            bytecode::SchedStop::Finished(Err(t)) => Some(t),
                            _ => None,
                        }
                    }
                }
            });
            (driver, end)
        })
        .collect()
}

/// A host-answered call declared with an `int` reply declines with `CapFault` on every driver
/// that parks host-answered calls.
#[test]
fn a_host_completed_call_with_an_int_reply_declines() {
    for (driver, end) in on_each_parking_driver(&narrow_call("i32")) {
        assert_eq!(end, Some(Trap::CapFault), "{driver}");
    }
}

/// The same, declared with no reply.
#[test]
fn a_host_completed_call_with_no_reply_declines() {
    for (driver, end) in on_each_parking_driver(&narrow_call("none")) {
        assert_eq!(end, Some(Trap::CapFault), "{driver}");
    }
}
