//! #1366 slice (b) / #1517 slice 1 — the **host-completed cap posture on the debug run**
//! (`ScheduledDebugRun`, the engine behind the DAP bytecode backend): a `call.cap` the embedder
//! services asynchronously parks the thread (`cap_parked`), `deliver_cap` resumes it, and — the W4
//! inertness pin generalized —
//! the *delivered* value joins the cap tape as the call's record, so a rebuilt run that replays
//! the tape (a reverse `seek`) serves the call from the tape and **never re-parks**.

use std::sync::{Arc, Mutex};
use temen_interp::bytecode::{SchedStop, ScheduledDebugRun};
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

#[test]
fn scheduled_run_parks_on_a_host_completed_cap_and_resumes() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = recording_host(&recorded);
    let mut run =
        ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;

    let stop = run.run_until_stop(&mut fuel);
    let id = run.cap_parked().expect("parked on the host-completed call");
    assert!(
        matches!(stop, SchedStop::CapPark { id: sid, .. } if sid == id),
        "the drive reports a live CapPark stop on the parked thread"
    );
    assert!(run.result().is_none(), "parked, not finished");
    assert_eq!(
        recorded.lock().unwrap().clone(),
        vec![(id, 7)],
        "the submit hook recorded the request under the surfaced id"
    );

    // Advancing while parked refuses (nothing is runnable), and the park is stable.
    assert!(matches!(
        run.run_until_stop(&mut fuel),
        SchedStop::CapPark { .. }
    ));
    assert_eq!(run.cap_parked(), Some(id), "still parked on the same call");
    assert!(
        !run.tick(&mut fuel),
        "a raw tick cannot advance a parked run"
    );
    assert!(!run.deliver_cap(id + 1, 0), "a foreign id is refused");

    assert!(run.deliver_cap(id, 107));
    assert_eq!(run.cap_parked(), None);
    assert!(matches!(
        run.run_until_stop(&mut fuel),
        SchedStop::Finished(Ok(ref v)) if *v == vec![Value::I64(WANT)]
    ));
    assert_eq!(
        run.result().cloned(),
        Some(Ok(vec![Value::I64(WANT)])),
        "the delivered value landed in the call's result slot"
    );

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

#[test]
fn scheduled_replay_serves_the_delivered_value_without_re_parking() {
    let m = module();
    let recorded: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (host, h) = recording_host(&recorded);
    let mut run =
        ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;
    assert!(matches!(
        run.run_until_stop(&mut fuel),
        SchedStop::CapPark { .. }
    ));
    let id = run.cap_parked().expect("parked");
    assert!(run.deliver_cap(id, 107));
    assert!(matches!(
        run.run_until_stop(&mut fuel),
        SchedStop::Finished(_)
    ));
    let tape = run.host().cap_tape();

    let replayed: Recorded = Arc::new(Mutex::new(Vec::new()));
    let (mut host, h) = recording_host(&replayed);
    host.replay_cap_tape(tape);
    let mut run =
        ScheduledDebugRun::new_with_host(&m, 0, &[Value::I32(h)], host).expect("in subset");
    let mut fuel = 2_000_000u64;
    assert!(
        matches!(run.run_until_stop(&mut fuel), SchedStop::Finished(_)),
        "a replayed host-completed call never re-parks"
    );
    assert_eq!(run.cap_parked(), None);
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
