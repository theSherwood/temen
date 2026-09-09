//! #1366 slice (b) — the **host-completed cap posture on the debug run** (`DebugRun`, the engine
//! behind the DAP bytecode backend): a `call.cap` the embedder services asynchronously parks the
//! debug run (`cap_parked`), `deliver_cap` resumes it, and — the W4 inertness pin generalized —
//! the *delivered* value joins the cap tape as the call's record, so a rebuilt run that replays
//! the tape (a reverse `seek`) serves the call from the tape and **never re-parks**.

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::DebugRun;
use temen_interp::{Host, OffloadOutcome, Value};

/// Two `HOST_PROC` (iface 13) calls: op 0 answers inline, op 1 is host-completed; composite
/// `r0 * 1000 + r1` = 105_107 when both answer `arg + 100`.
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
const WANT: i64 = 105_107;

type Recorded = Arc<Mutex<Vec<(u64, i64)>>>;

fn module() -> temen_ir::Module {
    let m = temen_text::parse_module(TWO_CALLS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A recording host (the DAP backend's shape: `record_caps` on) with the host-completed handler.
fn recording_host(recorded: &Recorded) -> (Host, i32) {
    let mut host = Host::new();
    host.record_caps();
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

/// Park → refuse-to-advance → `deliver_cap` → finish, with the tape carrying the delivered value.
#[test]
fn debug_run_parks_on_a_host_completed_cap_and_resumes() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = recording_host(&recorded);
    let mut run = DebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;

    // The first advance runs op 0 inline and parks on op 1: no breakpoint pc, not finished.
    assert_eq!(run.run_to(&[], &mut fuel), None);
    assert!(
        run.result().is_none(),
        "parked, not finished — got {:?}; recorded {:?}; minted {}; allowed {}",
        run.result(),
        recorded.lock().unwrap(),
        run.host().completions().minted(),
        run.host().completions().host_completed_allowed()
    );
    let id = run.cap_parked().expect("parked on the host-completed call");
    let rec = recorded.lock().unwrap().clone();
    assert_eq!(
        rec,
        vec![(id, 7)],
        "the submit hook recorded the request under the surfaced id"
    );

    // Resuming without delivering must not run anything: the op's slot is still empty.
    assert_eq!(run.run_to(&[], &mut fuel), None);
    assert_eq!(run.cap_parked(), Some(id), "still parked on the same call");

    // A wrong id is refused; the right one resumes.
    assert!(!run.deliver_cap(id + 1, 0), "a foreign id is refused");
    assert!(run.deliver_cap(id, 107));
    assert_eq!(run.cap_parked(), None);
    assert_eq!(run.run_to(&[], &mut fuel), None);
    assert_eq!(
        run.result().cloned(),
        Some(Ok(vec![Value::I64(WANT)])),
        "the delivered value landed in the call's result slot"
    );

    // The tape holds both host-proc calls, and op 1's record is the *delivered* value — not the
    // dispatch-time placeholder.
    let tape = run.host().cap_tape();
    let host_procs: Vec<_> = tape.records.iter().filter(|r| r.type_id == 13).collect();
    assert_eq!(host_procs.len(), 2, "both HOST_PROC calls taped: {tape:?}");
    assert_eq!(host_procs[0].op, 0);
    assert_eq!(host_procs[0].result, Ok(vec![105]));
    assert_eq!(host_procs[1].op, 1);
    assert_eq!(host_procs[1].args, vec![7]);
    assert_eq!(
        host_procs[1].result,
        Ok(vec![107]),
        "the delivered value, taped at delivery"
    );
}

/// The replay pin: a fresh run over the same host shape that **replays the tape** (what a reverse
/// `seek` rebuild does) serves the host-completed call from the tape — it finishes without ever
/// parking, and the handler's submit hook is never invoked again.
#[test]
fn replaying_the_tape_serves_the_delivered_value_without_re_parking() {
    let m = module();
    // Furthest-forward run: park, deliver, finish — producing the tape.
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = recording_host(&recorded);
    let mut run = DebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;
    assert_eq!(run.run_to(&[], &mut fuel), None);
    let id = run
        .cap_parked()
        .unwrap_or_else(|| panic!("parked — got {:?}", run.result()));
    assert!(run.deliver_cap(id, 107));
    assert_eq!(run.run_to(&[], &mut fuel), None);
    assert_eq!(run.result().cloned(), Some(Ok(vec![Value::I64(WANT)])));
    let tape = run.host().cap_tape();

    // The rebuild: same powerbox shape, replaying the tape (and recording past it, as the backend
    // does). The guest re-executes both calls; both are served from the tape.
    let replayed: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = recording_host(&replayed);
    host.replay_cap_tape(tape);
    let mut run = DebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;
    assert_eq!(run.run_to(&[], &mut fuel), None);
    assert_eq!(
        run.cap_parked(),
        None,
        "a replayed host-completed call never re-parks"
    );
    assert_eq!(
        run.result().cloned(),
        Some(Ok(vec![Value::I64(WANT)])),
        "the replay reproduces the delivered value"
    );
    assert!(
        replayed.lock().unwrap().is_empty(),
        "the tape served the call — the handler's submit hook never ran on the replay"
    );
}
