//! FORK.md PR 1 — `clone_caller(reply_orig, reply_twin)` (self-namespace op 11), fork's servicer-side
//! primitive, built across increments:
//!
//! - **1 — handler→caller linkage:** the handler can name the caller parked on the dispatch it serves.
//! - **2 — reply-injection nucleus:** from within the handler, deliver a reply to that parked caller
//!   *out-of-band* and suppress the handler's own auto-reply, so the caller reloads the **injected**
//!   value, not the handler's return — fork's core insight, "return-twice is a reply value" (FORK.md §3).
//! - **3 — the twin:** duplicate the parked caller into a second live domain (private window +
//!   duplicated powerbox), deliver `reply_orig` to the original and `reply_twin` to the twin. Both
//!   resume past the same fork `cap.call` — return-twice, one live run.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, StreamRole, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// func 0 (root/caller): spawn a server running func 1, mint an offer over export 0 (`svc` → func 2),
/// call it with `7`, return the reply. func 1 (server entry): serve one dispatch via `svc.wait`
/// (op 10). func 2 (the handler, bound to the offer): `clone_caller(999, 0)` — the explicit two-reply
/// form; the caller here is the ROOT (it spawned the server, so it is not a bare park and the fork
/// degrades to the single-reply injection of `reply_orig` = 999) — then return a *different* value
/// `5`, which must NOT reach the caller. The child-entry signature is `(i64) -> (i64)` — the spawn
/// ABI hands the entry an `(i64)` starter arg.
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
  v999 = i64.const 999
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64, i64) -> (i64) vz (v999, vzero)
  v5 = i64.const 5
  return v5
  }
}
"#;

/// The negative half of the linkage: `clone_caller` called from **outside** a handler (the root's
/// own `main`, no `serve_run`) is a probeable `-EINVAL` (it injects nothing), never a trap — the op
/// only acts when a handler is genuinely serving a parked caller.
const SRC_NO_HANDLER: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  varg = i64.const 999
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (varg)
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
            &[0],
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
        "clone_caller from main (no serve_run) refuses with -EINVAL, not a trap",
    );
}

#[test]
fn clone_caller_injects_the_callers_reply_out_of_band() {
    let m = module(SRC);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let mut fuel = 20_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &mut host).expect("run");
    // The root observes the value `clone_caller` INJECTED (999), not the handler's own return (5) —
    // proving the servicer supplied the caller's reply out-of-band and the auto-reply was suppressed.
    assert_eq!(
        r,
        vec![Value::I64(999)],
        "the caller reloads the injected reply (999), not the handler's return (5)",
    );
}

/// Increment 3 — the **twin**: `clone_caller(reply_orig, reply_twin)` from within the handler
/// duplicates the parked caller into a second live domain. Both copies resume **past** the fork
/// `cap.call` from the same call site — the original with `reply_orig`, the twin with `reply_twin` —
/// each over its own private window + duplicated powerbox, and each writes its reply to the **shared**
/// stdout sink (fork shares stdout) before returning.
///
/// The **correct fork topology**: the *caller* is a spawned child with no children of its own (the
/// only shape the targeted clone duplicates faithfully). func 0 (root) spawns server `S` (func 1),
/// mints an offer over its `svc` export, then spawns caller `C` (func 3) re-granting that offer (as
/// `"svc"`) and stdout (as `"o"`) into it; root joins `C` and returns its result. `C` resolves both
/// caps by name, calls the fork offer, writes its reply to stdout, and returns it. func 2 (S's
/// handler): `clone_caller(100, 200)`. Root returns `C`'s `reply_orig` (100); the shared stdout
/// carries BOTH 100 and 200 — the twin ran `C`'s continuation with `reply_twin`.
const SRC_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "svc" 1 { op: 2 }
data 300 "svc"
data 310 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  ve1 = i64.const 1
  voffs = i64.const 131072
  vlog = i64.const 12
  vq = i64.const 0
  vs = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve1, voffs, vlog, vq)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp = i32.const 300
  i32.store va0 vnp
  va1 = i64.const 260
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 264
  i32.store va2 vcap
  va3 = i64.const 272
  vnp2 = i32.const 310
  i32.store va3 vnp2
  va4 = i64.const 276
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 280
  i32.store va5 vout
  vgp = i64.const 256
  vgn = i64.const 2
  ve3 = i64.const 3
  voffc = i64.const 135168
  vc = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve3, voffc, vlog, vq)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
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
  vro = i64.const 100
  vrt = i64.const 200
  vt = cap.call 4294967295 11 (i64, i64) -> (i64) vz (vro, vrt)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vhsvc (varg)
  vp16 = i64.const 16
  i64.store vp16 vr
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }
}
"#;

#[test]
fn clone_caller_forks_the_caller_into_a_twin_that_returns_the_second_reply() {
    let m = module(SRC_TWIN);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout(); // promote stdout to a shared sink the twin inherits
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    // The run's result is the ORIGINAL caller's reply (reply_orig = 100), joined back through C.
    assert_eq!(
        r,
        vec![Value::I64(100)],
        "the original caller resumes past the fork call with reply_orig (100)"
    );
    // The shared stdout carries BOTH replies — the twin resumed the same continuation with reply_twin
    // (200) over its own private window and duplicated powerbox. Order is scheduler-dependent.
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(bytes.len(), 16, "two i64 writes reached the shared sink");
    let mut vals: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    vals.sort();
    assert_eq!(
        vals,
        vec![100, 200],
        "both the original (100) and the twin (200) wrote their reply — return-twice, one live run"
    );
}

/// FORK.md PR 5 — **pid mode**, the exact `fork()` shape: `clone_caller(0)` (one arg) replies the
/// twin's `TaskId` to the original (parent sees pid) and `0` to the twin (child sees 0). Same
/// topology as the two-reply test; only the handler changes. Task ids are deterministic here
/// (root = 0, server S = 1, caller C = 2, twin = 3), so the original C returns 3 and the twin
/// returns 0 — both observed on the shared stdout sink, and C's 3 joined back through the root.
const SRC_FORK_PID: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { op: 0 }
export 0 interface "svc" 1 { op: 2 }
data 300 "svc"
data 310 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  ve1 = i64.const 1
  voffs = i64.const 131072
  vlog = i64.const 12
  vq = i64.const 0
  vs = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve1, voffs, vlog, vq)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 256
  vnp = i32.const 300
  i32.store va0 vnp
  va1 = i64.const 260
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 264
  i32.store va2 vcap
  va3 = i64.const 272
  vnp2 = i32.const 310
  i32.store va3 vnp2
  va4 = i64.const 276
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 280
  i32.store va5 vout
  vgp = i64.const 256
  vgn = i64.const 2
  ve3 = i64.const 3
  voffc = i64.const 135168
  vc = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) v0 (vgp, vgn, ve3, voffc, vlog, vq)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
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
  vzero = i64.const 0
  vt = cap.call 4294967295 11 (i64) -> (i64) vz (vzero)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vhsvc (varg)
  vp16 = i64.const 16
  i64.store vp16 vr
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vr
  }
}
"#;

#[test]
fn pid_mode_replies_the_twins_task_id_to_the_parent_and_zero_to_the_child() {
    let m = module(SRC_FORK_PID);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 40_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    // The original C's fork() returned the twin's TaskId (3) — the POSIX parent-sees-pid — joined
    // back through the root.
    assert_eq!(
        r,
        vec![Value::I64(3)],
        "the original caller's fork() returns the twin's pid (TaskId 3)"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(bytes.len(), 16, "both copies wrote their fork() return");
    let mut vals: Vec<i64> = bytes
        .chunks_exact(8)
        .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    vals.sort();
    assert_eq!(
        vals,
        vec![0, 3],
        "child sees 0, parent sees the pid — fork()'s return-twice contract"
    );
}
