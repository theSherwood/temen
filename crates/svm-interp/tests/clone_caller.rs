//! FORK.md PR 1, increment 1 — the **handler→caller linkage**. `clone_caller` (self-namespace op 11)
//! is fork's servicer-side primitive: from within a serve handler it acts on the caller parked on the
//! dispatch this handler is serving. This first increment proves only the linkage — the handler can read
//! its own dispatch ticket via `serve_run` — before the capture / twin / dual-reply increments.
//!
//! Harness (the `svc_serve_loop` shape): the root spawns a server child, mints an offer over its `svc`
//! export, and calls it — parking on `CapReply`. The server's handler (func 2) calls `clone_caller` and
//! returns its result; the reply threads back to the root. A result `>= 0` (the served ticket) proves the
//! handler was correctly identified as serving a dispatch; `-EINVAL` would mean the linkage is broken.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// func 0 (root/caller): spawn a server running func 1, mint an offer over export 0 (`svc` → func 2),
/// call it with `7`, return the reply. func 1 (server entry): serve one dispatch via `svc.wait`
/// (op 10). func 2 (the handler, bound to the offer): `clone_caller()` (op 11) and return it. The
/// child-entry signature is `(i64) -> (i64)` — the spawn ABI hands the entry an `(i64)` starter arg.
const SRC: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "svc" 1 { op: 2 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ventry = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vc = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) v0 (vc, vexp)
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vh (varg)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 11 () -> (i64) vz ()
  return vt
  }
}
"#;

/// The negative half of the linkage: `clone_caller` called from **outside** a handler (the root's
/// own `main`, no `serve_run`) is a probeable `-EINVAL`, never a trap and never a bogus ticket — the
/// op only acts when a handler is genuinely serving a parked caller.
const SRC_NO_HANDLER: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vt = cap.call 4294967295 11 () -> (i64) vz ()
  return vt
  }
}
"#;

/// The self-namespace op number is part of the public surface (the personality's `fork` provider
/// will emit it): pin it, as `svc_serve_loop` pins svc.poll/svc.wait.
#[test]
fn the_clone_caller_op_number_is_pinned() {
    assert_eq!(svm_interp::CAP_SELF_CLONE_CALLER, 11);
}

/// A backend tier **without** the eval-loop servicing arm (the JIT's route) answers `clone_caller`
/// with a probeable `-EINVAL` from the one shared host dispatch — refusal, never a trap — exactly as
/// it does for svc.poll/svc.wait. Pinned directly at the shared entry.
#[test]
fn a_non_serving_tier_refuses_clone_caller_probeably() {
    let mut host = Host::new();
    let r = host
        .cap_dispatch_slots(
            svm_ir::CAP_SELF_TYPE_ID,
            svm_interp::CAP_SELF_CLONE_CALLER,
            0,
            &[],
            None,
        )
        .expect("refusal, not a trap");
    assert_eq!(r, vec![-22], "probeable -EINVAL from a non-serving tier");
}

#[test]
fn clone_caller_outside_a_handler_is_probeable_einval() {
    let m = module(SRC_NO_HANDLER);
    let mut host = Host::new();
    host.set_self_module(&m);
    let mut fuel = 1_000_000u64;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(-22)],
        "clone_caller from main (no serve_run) refuses with -EINVAL, not a trap or a bogus ticket",
    );
}

#[test]
fn clone_caller_reads_its_dispatch_ticket_from_within_a_handler() {
    let m = module(SRC);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let mut fuel = 20_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &mut host).expect("run");
    // The handler returned `clone_caller()`'s result, threaded back as the offer call's reply. The first
    // dispatch's ticket is 0 (per-callee tickets start at 0); a broken linkage would be -EINVAL (-22).
    assert_eq!(
        r,
        vec![Value::I64(0)],
        "clone_caller from within a handler returns the served ticket (0), not -EINVAL",
    );
}
