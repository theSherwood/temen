//! The cooperative bytecode pump must not sleep through a kill that lands **during its settle**.
//!
//! The pump's loop top finalizes every task of a killed domain (#1215), and when everything is parked
//! on something an embedder can wake it blocks on the external-wake doorbell (#1122) until the next
//! ring. A kill rings that bell. The pump snapshotted the bell generation *after* its kill sweep, so
//! a kill landing between the two had its ring folded into the snapshot: the pump then blocked until
//! some unrelated ring (the next keystroke) although the domain was already dead. The source below
//! lands the kill at exactly that point — right after the sweep reads `killed()` — so the ordering
//! is deterministic.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use temen_interp::{bytecode, Host, SignalSource, StreamRole, Value};

/// Answers `killed()` truthfully, but on the `trigger`th query goes dead *after* answering and rings
/// the kill door — a terminate arriving from another thread just after the sweep looked.
struct KilledAfterTheSweep {
    queries: AtomicUsize,
    trigger: usize,
    dead: AtomicBool,
    ring: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
}

impl SignalSource for KilledAfterTheSweep {
    fn take_deliverable(&self) -> Option<(i32, i32, u64)> {
        None
    }
    fn set_kill(&self, kill: Arc<dyn Fn() + Send + Sync>) {
        *self.ring.lock().unwrap() = Some(kill);
    }
    fn killed(&self) -> bool {
        let was = self.dead.load(Ordering::SeqCst);
        if self.queries.fetch_add(1, Ordering::SeqCst) + 1 == self.trigger {
            self.dead.store(true, Ordering::SeqCst);
            if let Some(ring) = self.ring.lock().unwrap().clone() {
                ring();
            }
        }
        was
    }
}

/// The root reads the (empty, blocking) stdin stream: it parks, and the pump — nothing runnable, an
/// externally wakeable park, an armed bell — blocks for a ring.
const READ_STDIN: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vbuf = i64.const 16392
  vcap = i64.const 4
  vr = call.cap 0 0 (i64, i64) -> (i64) v0 (vbuf, vcap)
  return vr
  }
}
"#;

#[test]
fn a_kill_that_lands_after_the_kill_sweep_is_not_slept_through() {
    let m = temen_text::parse_module(READ_STDIN).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut host = Host::new();
        let h = host.grant_stream(StreamRole::In);
        host.set_stdin_blocking(true);
        host.arm_external_wake();
        // Query 1: the root is still runnable. Query 2: the root is parked on stdin — the kill lands
        // right after this sweep, and the pump's next move is the all-parked block.
        let source = Arc::new(KilledAfterTheSweep {
            queries: AtomicUsize::new(0),
            trigger: 2,
            dead: AtomicBool::new(false),
            ring: Mutex::new(None),
        });
        host.set_signal_source(source, Arc::new(AtomicBool::new(false)));
        let mut fuel = u64::MAX;
        let r = bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host);
        let _ = tx.send(r);
    });
    let r = rx
        .recv_timeout(Duration::from_secs(10))
        .expect("the pump slept through a kill that rang its bell");
    let r = r.expect("the bytecode engine runs this module");
    assert!(
        r.is_err(),
        "the killed root is finalized, not returned: {r:?}"
    );
}
