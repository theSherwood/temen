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

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{self, VcpuEvent};
use temen_interp::{run_with_host, Host, OffloadOutcome, Trap, Value};

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
    let h = host.grant_host_proc_offloadable(Box::new(move |op, args| {
        let a = *args.first().unwrap_or(&0);
        if op == 0 {
            return OffloadOutcome::Done(Ok(vec![a + 100]));
        }
        let rec = Arc::clone(&rec);
        OffloadOutcome::Host(Box::new(move |id| rec.lock().unwrap().push((id, a))))
    }));
    (host, h)
}

/// The pool-completed twin: identical semantics, but op 1 punts a self-contained job to the
/// offload pool (`Offload`) — the posture the parity test compares against.
fn pool_completed() -> (Host, i32) {
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(Box::new(move |op, args| {
        let a = *args.first().unwrap_or(&0);
        if op == 0 {
            return OffloadOutcome::Done(Ok(vec![a + 100]));
        }
        OffloadOutcome::Offload(Box::new(move || a + 100))
    }));
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
    let h = host.grant_host_proc_offloadable(Box::new(move |op, args| {
        let a = *args.first().unwrap_or(&0);
        if op == 0 {
            return OffloadOutcome::Done(Ok(vec![a + 100]));
        }
        let c = Arc::clone(&comps_for_hook);
        OffloadOutcome::Host(Box::new(move |id| {
            c.complete_host(id, a + 100);
        }))
    }));
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
    let h = host.grant_host_proc_offloadable(Box::new(|_op, args| {
        OffloadOutcome::Done(Ok(vec![*args.first().unwrap_or(&0) + 100]))
    }));
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
