//! #1122 route (a), #2528 — a [`temen_run::CoopSession`] tells its embedder of a host-answered call
//! ([`SessionStep::CapPark`]): a capability whose proc answers with
//! [`OffloadOutcome::Host`] parks its caller, the embedder answers through the call's
//! [`Completer`], and the session pumps on. On both executors (#1414 5b): the pump, and executor 2
//! with [`RunConfig::parallel`].

use std::sync::{Arc, Mutex};
use temen_interp::{
    cap_id, CapState, Completer, HostProcGrant, OffloadOutcome, ProcHandler, Value,
};
use temen_run::{instantiate, HostCap, Outcome, RunConfig, SessionStep};
use temen_text::parse_module;

/// `_start` resolves the capability named `ask` and calls its op 1 with 5: the answer is its result.
const ASKS: &str = "\
memory 16
data 16384 \"ask\"
func 0 () -> (i32) {
block 0 () {
  vp = i64.const 16384
  vl = i64.const 3
  vh = self.resolve vp vl
  vfive = i64.const 5
  vr = call.cap 13 1 (i64) -> (i64) vh (vfive)
  vc = i32.wrap_i64 vr
  return vc
  }
}
export 0 func \"_start\" 0
";

/// The calls `ask`'s submit hook filed and nobody has answered: each call's completer and argument.
type Filed = Arc<Mutex<Vec<(Completer, i64)>>>;

/// `ask`: every call punts to its embedder, who answers `argument + 100`.
fn ask(filed: &Filed) -> HostCap {
    let filed = Arc::clone(filed);
    HostCap::custom(cap_id::HOST_PROC, 1, move |h, _| {
        let filed = Arc::clone(&filed);
        h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Offloadable(Box::new(move |_op, args| {
                let a = *args.first().unwrap_or(&0);
                let filed = Arc::clone(&filed);
                OffloadOutcome::Host(Box::new(move |c| filed.lock().unwrap().push((c, a))))
            })),
            CapState::Stateless,
        ))
    })
}

/// Run `f` on its own thread, for at most 30 s: a session that waits for good fails, never hangs.
fn bounded<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    rx.recv_timeout(std::time::Duration::from_secs(30))
        .expect("the session neither paused nor ended")
}

#[test]
fn a_session_tells_of_a_host_answered_call_and_pumps_on_once_it_is_answered() {
    for parallel in [false, true] {
        bounded(move || session_asks(parallel));
    }
}

fn session_asks(parallel: bool) {
    let inst = instantiate(parse_module(ASKS).expect("parse")).expect("instantiate");
    {
        let filed: Filed = Arc::default();
        let config = RunConfig {
            parallel,
            ..RunConfig::default()
        };
        let mut session = inst
            .open_coop_session(&config, &[("ask", ask(&filed))])
            .expect("open the session");
        let id = match session.pump().expect("pump") {
            SessionStep::CapPark { id } => id,
            other => panic!("parallel {parallel}: the call waits for its embedder, not {other:?}"),
        };
        let (c, a) = filed
            .lock()
            .unwrap()
            .pop()
            .expect("the submit hook filed the call");
        assert_eq!((c.id(), a), (id, 5), "parallel {parallel}");
        c.complete(a + 100);
        match session.pump().expect("pump") {
            SessionStep::Done(outcome) => assert_eq!(
                outcome,
                Outcome::Returned(vec![Value::I32(105)]),
                "parallel {parallel}"
            ),
            other => panic!("parallel {parallel}: the answered run ends, not {other:?}"),
        }
    }
}
