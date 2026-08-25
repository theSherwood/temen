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
use temen_interp::{run_with_host, Host, StreamRole, Value};

fn module(text: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// FORK.md §9 / OPS_PARITY.md (`clone_caller`/`reap`) — `serve_qualifies` (temen-run's **Cranelift**
/// routing predicate) still folds a forking module to the oracle: the Cranelift fork slice is
/// unbuilt, so it must never compile fork there and answer `-EINVAL` where the oracle forks (a
/// silent divergence, INVARIANTS.md #9). This pins the *Cranelift* side of the per-backend split —
/// the **bytecode** engine now serves fork natively (see
/// `bytecode_forks_the_twin_identically_to_the_oracle`), via the separate `bytecode_serves_fork`
/// escape, so the two predicates deliberately diverge on fork until Cranelift catches up.
#[test]
fn serve_qualifies_still_folds_fork_for_cranelift() {
    use temen_interp::bytecode::serve_qualifies;
    use temen_ir::{Block, Func, Inst, Terminator, ValType, CAP_SELF_TYPE_ID};

    // #922: `serve_qualifies` inspects only the cap-op, never the signature, so any interned
    // type index (0) is fine here — `results` still shapes the surrounding func's declared results.
    let self_cap = |op: u32, _results: Vec<ValType>| Inst::CapCall {
        type_id: CAP_SELF_TYPE_ID,
        op,
        sig: 0,
        handle: 0,
        args: vec![],
    };
    // A bare `svc.poll` service point is serve-qualified (a fast backend runs the serve loop).
    let serve_only = vec![Func {
        params: vec![],
        results: vec![ValType::I32],
        blocks: vec![Block {
            params: vec![],
            insts: vec![self_cap(
                temen_interp::CAP_SELF_SVC_POLL,
                vec![ValType::I32],
            )],
            term: Terminator::Return(vec![0]),
        }],
    }];
    assert!(
        serve_qualifies(&serve_only),
        "a bare serve point should qualify for native serving"
    );

    // Adding a handler that forks (`clone_caller`) or reaps (`reap`) makes `serve_qualifies` (the
    // Cranelift routing predicate) fold the module — fork is not lowered on Cranelift yet. (The
    // bytecode gate takes the `bytecode_serves_fork` escape instead; that path is proven separately.)
    for fork_op in [
        temen_interp::CAP_SELF_CLONE_CALLER,
        temen_interp::CAP_SELF_REAP,
    ] {
        let mut funcs = serve_only.clone();
        funcs.push(Func {
            params: vec![],
            results: vec![],
            blocks: vec![Block {
                params: vec![],
                insts: vec![self_cap(fork_op, vec![])],
                term: Terminator::Return(vec![]),
            }],
        });
        assert!(
            !serve_qualifies(&funcs),
            "serve_qualifies must fold a forking module (self-op {fork_op}) for Cranelift"
        );
    }
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
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967284
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 17536
  i64.store q0a0 q0v0
  q0a1 = i64.const 17544
  i64.store q0a1 q0v1
  q0a2 = i64.const 17552
  i64.store q0a2 q0v2
  q0a3 = i64.const 17560
  i64.store q0a3 q0v3
  q0a4 = i64.const 17568
  i64.store q0a4 q0v4
  q0a5 = i64.const 17576
  i64.store q0a5 q0v4
  q0a6 = i64.const 17584
  i64.store q0a6 q0v4
  vc = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
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
    assert_eq!(temen_interp::CAP_SELF_CLONE_CALLER, 11);
}

/// A backend tier **without** the eval-loop servicing arm (the JIT's route) answers `clone_caller`
/// with a probeable `-EINVAL` from the one shared host dispatch — refusal, never a trap — exactly as
/// it does for svc.poll/svc.wait. Pinned directly at the shared entry.
#[test]
fn a_non_serving_tier_refuses_clone_caller_probeably() {
    let mut host = Host::new();
    let r = host
        .cap_dispatch_slots(
            temen_ir::CAP_SELF_TYPE_ID,
            temen_interp::CAP_SELF_CLONE_CALLER,
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
/// mints an offer over its `svc` export, then spawns caller `C` (func 4) re-granting that offer (as
/// `"svc"`) and stdout (as `"o"`) into it; root joins `C` and returns its result. `C` resolves both
/// caps by name, calls the `fork` verb, writes its reply to stdout, and returns it. func 2 (S's
/// `fork` handler): `clone_caller(100, 200)`. Root returns `C`'s `reply_orig` (100); the shared stdout
/// carries BOTH 100 and 200 — the twin ran `C`'s continuation with `reply_twin`.
///
/// **Determinism / ISSUES.md I53.** Two serve/park races made the one-shot form flaky, and this test
/// closes both with the FORK.md §8.6 `reap` idiom (which is what makes `fork_then_wait` stable). The
/// parent reaps the twin's deterministic `TaskId` (root = 0, S = 1, C = 2, twin = 3 — the fixed
/// assignment failed forks never consume, so the winning fork always mints id 3):
///   1. **Teardown-drain.** A fork twin is minted straight onto `runnable` with no handle in anyone's
///      table — a *detached daemon*, which INVARIANT #6 says root completion abandons (post-teardown
///      effects unspecified). A bare `join C`-only fixture thus lets the twin's stdout write lose to
///      teardown (`left: 8, right: 16`). The parent reaping the twin (`wait`, op 1 → `reap`, func 3)
///      keeps the run alive until the twin has finished — and thus written.
///   2. **Fork-park.** The two-reply `clone_caller(reply_orig, reply_twin)` **silently degrades to a
///      single reply with no twin** when the server drains the dispatch before `C` registers its
///      `CapReply` waiter — unlike pid mode there is no `-EAGAIN` on the return to retry. `C` instead
///      detects it: after a `reply_orig` (100) return, `reap(3)` answers `-ECHILD` (no such twin), and
///      `C` **re-forks**. A live twin makes `reap` retryable-`-EAGAIN` (serve/park race) or deliver its
///      status, never `-ECHILD`, so the re-fork fires only on a genuinely lost fork. Interp only.
const SRC_TWIN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=131072 sl=12 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967284
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 17600
  i64.store q1a0 q1v0
  q1a1 = i64.const 17608
  i64.store q1a1 q1v1
  q1a2 = i64.const 17616
  i64.store q1a2 q1v2
  q1a3 = i64.const 17624
  i64.store q1a3 q1v3
  q1a4 = i64.const 17632
  i64.store q1a4 q1v4
  q1a5 = i64.const 17640
  i64.store q1a5 q1v4
  q1a6 = i64.const 17648
  i64.store q1a6 q1v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  ; spawn via record (op 17): entry=4 off=135168 sl=12 quota=0
  q2v0 = i64.const 17179869184
  q2v1 = i64.const 135168
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
  q2v5 = i64.const 16640
  q2v6 = i64.const 2
  q2a0 = i64.const 17664
  i64.store q2a0 q2v0
  q2a1 = i64.const 17672
  i64.store q2a1 q2v1
  q2a2 = i64.const 17680
  i64.store q2a2 q2v2
  q2a3 = i64.const 17688
  i64.store q2a3 q2v3
  q2a4 = i64.const 17696
  i64.store q2a4 q2v4
  q2a5 = i64.const 17704
  i64.store q2a5 q2v5
  q2a6 = i64.const 17712
  i64.store q2a6 q2v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q2a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
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
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vhsvc (varg)
  v200 = i64.const 200
  vistwin = i64.eq vr v200
  br_if vistwin 4(vr, vho) 2(vr, vhsvc, vho)
  }
block 2 (vr: i64, vhsvc: i32, vho: i32) {
  vpid3 = i64.const 3
  vstatus = cap.call 268435456 1 (i64) -> (i64) vhsvc (vpid3)
  veagain = i64.const -11
  viseagain = i64.eq vstatus veagain
  br_if viseagain 2(vr, vhsvc, vho) 3(vr, vstatus, vhsvc, vho)
  }
block 3 (vr: i64, vstatus: i64, vhsvc: i32, vho: i32) {
  vechild = i64.const -10
  visechild = i64.eq vstatus vechild
  br_if visechild 1(vhsvc, vho) 4(vr, vho)
  }
block 4 (vr: i64, vho: i32) {
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
    // (200) over its own private window and duplicated powerbox. The original reaped the twin before
    // returning (I53), so both writes are on the sink deterministically; order is scheduler-dependent.
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

/// FORK.md §9 — **native bytecode fork**: the `SRC_TWIN` fork topology now runs on the *bytecode*
/// engine (not folded to the tree-walk oracle), producing the identical observable result — the run
/// value plus both replies on the shared stdout sink. This is the differential pin the parity
/// matrix's bytecode `clone_caller`/`reap` cells rest on (INVARIANTS.md #9; OPS_PARITY.md). The
/// twin's continuation is the caller's `Vm` cloned at its post-call resume point over a private
/// window (`Mem::fork_private`) + duplicated powerbox (`Host::fork_powerbox`); the parent `wait`s
/// the twin (`reap`) so its stdout write is observed deterministically before teardown.
fn run_src_twin_oracle() -> (Vec<Value>, Vec<u8>) {
    let m = module(SRC_TWIN);
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
    .expect("oracle run");
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    (r, bytes)
}

#[test]
fn bytecode_forks_the_twin_identically_to_the_oracle() {
    let (oracle_r, oracle_bytes) = run_src_twin_oracle();

    // The bytecode engine must run the fork module NATIVELY (`Some`) — a fold would return `None`.
    let m = module(SRC_TWIN);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 40_000_000u64;
    let bc_r = temen_interp::bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("the fork module runs natively on the bytecode engine (not folded to the oracle)")
    .expect("bytecode run");
    let bc_bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();

    assert_eq!(
        bc_r, oracle_r,
        "bytecode fork returns the oracle's value (reply_orig = 100, joined through C)"
    );
    let sorted = |b: &[u8]| {
        let mut v: Vec<i64> = b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        v.sort();
        v
    };
    assert_eq!(
        sorted(&bc_bytes),
        sorted(&oracle_bytes),
        "bytecode and oracle agree on the shared sink (both replies, order-independent)"
    );
    assert_eq!(
        sorted(&bc_bytes),
        vec![100, 200],
        "return-twice on bytecode: original (100) + twin (200) both wrote their reply"
    );
}

/// FORK.md PR 5 — **pid mode**, the exact `fork()` shape: `clone_caller(0)` (one arg) replies the
/// twin's `TaskId` to the original (parent sees pid) and `0` to the twin (child sees 0). Same
/// topology as the two-reply test; only the `fork` handler changes. Task ids are deterministic here
/// (root = 0, server S = 1, caller C = 2, twin = 3), so the original C returns 3 and the twin
/// returns 0 — both observed on the shared stdout sink, and C's 3 joined back through the root.
///
/// **Determinism / ISSUES.md I53.** Two serve/park races made the one-shot form flaky. (1) `fork`
/// itself can lose the enqueue-before-park window — the server drains the dispatch before C registers
/// its `CapReply` waiter — and pid mode answers that with `-EAGAIN` (POSIX fork-failure); C **retries**
/// `while ((pid = fork()) < 0)`, the realistic shell idiom (this is the historical `left: [-11]`
/// flake). (2) The twin is a detached daemon whose stdout write races the run's teardown (INVARIANT
/// #6), so C **reaps** it (`wait`, op 1 → `reap`, func 3) before returning, forcing the run to outlive
/// the twin's write. Both are the `fork_then_wait` idioms that make that test stable. Interp only.
const SRC_FORK_PID: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=131072 sl=12 quota=0
  q3v0 = i64.const 4294967296
  q3v1 = i64.const 131072
  q3v2 = i64.const -4294967284
  q3v3 = i64.const 4294967295
  q3v4 = i64.const 0
  q3a0 = i64.const 17728
  i64.store q3a0 q3v0
  q3a1 = i64.const 17736
  i64.store q3a1 q3v1
  q3a2 = i64.const 17744
  i64.store q3a2 q3v2
  q3a3 = i64.const 17752
  i64.store q3a3 q3v3
  q3a4 = i64.const 17760
  i64.store q3a4 q3v4
  q3a5 = i64.const 17768
  i64.store q3a5 q3v4
  q3a6 = i64.const 17776
  i64.store q3a6 q3v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q3a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  ; spawn via record (op 17): entry=4 off=135168 sl=12 quota=0
  q4v0 = i64.const 17179869184
  q4v1 = i64.const 135168
  q4v2 = i64.const -4294967284
  q4v3 = i64.const 4294967295
  q4v4 = i64.const 0
  q4v5 = i64.const 16640
  q4v6 = i64.const 2
  q4a0 = i64.const 17792
  i64.store q4a0 q4v0
  q4a1 = i64.const 17800
  i64.store q4a1 q4v1
  q4a2 = i64.const 17808
  i64.store q4a2 q4v2
  q4a3 = i64.const 17816
  i64.store q4a3 q4v3
  q4a4 = i64.const 17824
  i64.store q4a4 q4v4
  q4a5 = i64.const 17832
  i64.store q4a5 q4v5
  q4a6 = i64.const 17840
  i64.store q4a6 q4v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q4a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
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
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  br 1(vhsvc, vho)
  }
block 1 (vhsvc: i32, vho: i32) {
  vc0 = i64.const 0
  vpid = cap.call 268435456 0 (i64) -> (i64) vhsvc (vc0)
  vz1 = i64.const 0
  vforkfail = i64.lt_s vpid vz1
  br_if vforkfail 1(vhsvc, vho) 2(vpid, vhsvc, vho)
  }
block 2 (vpid: i64, vhsvc: i32, vho: i32) {
  vp16 = i64.const 16
  i64.store vp16 vpid
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  vz2 = i64.const 0
  vparent = i64.ne vpid vz2
  br_if vparent 3(vpid, vhsvc) 5(vpid)
  }
block 3 (vpid: i64, vhsvc: i32) {
  vstatus = cap.call 268435456 1 (i64) -> (i64) vhsvc (vpid)
  vz3 = i64.const 0
  vwaitfail = i64.lt_s vstatus vz3
  br_if vwaitfail 3(vpid, vhsvc) 4(vpid)
  }
block 4 (vpid: i64) {
  return vpid
  }
block 5 (vpid: i64) {
  return vpid
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
    // Both copies wrote their fork() return (parent 3, child 0). The parent reaped the twin before
    // returning (I53), so the child's write is on the sink deterministically before teardown.
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

/// FORK.md §8.6 — **fork + wait**, the shell's command loop in miniature: `while ((pid = fork()) <
/// 0); if (pid) { while ((s = wait(pid)) < 0); } else exit(42);`. One server serves **two** verbs
/// over one offer — `fork` (op 0 → `clone_caller`, func 2) and `wait` (op 1 → `reap`, func 3). The
/// caller C (func 4) forks, then **branches on the return**: the twin (pid 0) exits `42`; the
/// original (pid 3) `wait(3)`s and writes the reaped status to stdout. Because C waits the instant it
/// forks, the twin is usually still running — so `reap` takes the **deferred** path (park the parent
/// in `join_waiters[3]`, resumed by the twin's completion with `Pending::ReapPid`). The reaped status
/// is the twin's `exit(42)`, and the run (root joins C, the original) returns it.
///
/// **Both `fork` and `wait` retry on `-EAGAIN`** — the realistic shell idiom. Either call can lose
/// the serve/park race under load (the servicer drains the dispatch before the caller registers its
/// `CapReply` waiter): `fork` then fails `-EAGAIN` (the existing pid-mode contract) and `reap`
/// returns `-EAGAIN` too (never `-ECHILD` — the twin is real). The retry loops make the outcome
/// deterministic regardless of interleaving, which is why the test is stable under a full parallel
/// suite where the raw one-shot form flakes (ISSUES.md I53 family). Interp only, like every fork test.
const SRC_FORK_WAIT: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 0 }
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=131072 sl=12 quota=0
  q5v0 = i64.const 4294967296
  q5v1 = i64.const 131072
  q5v2 = i64.const -4294967284
  q5v3 = i64.const 4294967295
  q5v4 = i64.const 0
  q5a0 = i64.const 17856
  i64.store q5a0 q5v0
  q5a1 = i64.const 17864
  i64.store q5a1 q5v1
  q5a2 = i64.const 17872
  i64.store q5a2 q5v2
  q5a3 = i64.const 17880
  i64.store q5a3 q5v3
  q5a4 = i64.const 17888
  i64.store q5a4 q5v4
  q5a5 = i64.const 17896
  i64.store q5a5 q5v4
  q5a6 = i64.const 17904
  i64.store q5a6 q5v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q5a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  ; spawn via record (op 17): entry=4 off=135168 sl=12 quota=0
  q6v0 = i64.const 17179869184
  q6v1 = i64.const 135168
  q6v2 = i64.const -4294967284
  q6v3 = i64.const 4294967295
  q6v4 = i64.const 0
  q6v5 = i64.const 16640
  q6v6 = i64.const 2
  q6a0 = i64.const 17920
  i64.store q6a0 q6v0
  q6a1 = i64.const 17928
  i64.store q6a1 q6v1
  q6a2 = i64.const 17936
  i64.store q6a2 q6v2
  q6a3 = i64.const 17944
  i64.store q6a3 q6v3
  q6a4 = i64.const 17952
  i64.store q6a4 q6v4
  q6a5 = i64.const 17960
  i64.store q6a5 q6v5
  q6a6 = i64.const 17968
  i64.store q6a6 q6v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q6a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
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
block 0 (vpid: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64) -> (i64) vz (vpid)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  br 1()
  }
block 1 () {
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vc0 = i64.const 0
  vpid = cap.call 268435456 0 (i64) -> (i64) vhsvc (vc0)
  vz1 = i64.const 0
  vforkfail = i64.lt_s vpid vz1
  br_if vforkfail 1() 2(vpid)
  }
block 2 (vpid: i64) {
  vz2 = i64.const 0
  vparent = i64.ne vpid vz2
  br_if vparent 3(vpid) 5()
  }
block 3 (vpid: i64) {
  vp0b = i64.const 0
  vl3b = i64.const 3
  vhsvc2 = cap.self.resolve vp0b vl3b
  vstatus = cap.call 268435456 1 (i64) -> (i64) vhsvc2 (vpid)
  vz3 = i64.const 0
  vwaitfail = i64.lt_s vstatus vz3
  br_if vwaitfail 3(vpid) 4(vstatus)
  }
block 4 (vstatus: i64) {
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  vp16 = i64.const 16
  i64.store vp16 vstatus
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vstatus
  }
block 5 () {
  v42 = i64.const 42
  return v42
  }
}
"#;

#[test]
fn fork_then_wait_reaps_the_twins_exit_status_through_the_shared_offer() {
    let m = module(SRC_FORK_WAIT);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 60_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    // The original C forked (pid 3), waited on it, and returned the reaped status — joined to root.
    assert_eq!(
        r,
        vec![Value::I64(42)],
        "the parent's wait(pid) reaped the twin's exit(42)"
    );
    // Exactly ONE write reached stdout: the parent's reaped status. The twin took the `else` branch
    // (exit 42) and never wrote — proving the branch-on-fork-return split parent from child.
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(
        bytes.len(),
        8,
        "only the parent wrote — the child exited without writing"
    );
    let status = i64::from_le_bytes(bytes[..8].try_into().unwrap());
    assert_eq!(status, 42, "the reaped status is the twin's exit code");
}

/// FORK.md §8.6 — **`waitpid(pid, WNOHANG)`**: the non-blocking poll. Same fork/wait server as
/// `SRC_FORK_WAIT`, but the `wait` offer op now carries `(pid, flags)` and the twin **parks forever**
/// (`atomic.wait` on an unchanging futex, timeout `-1`) so it *never* exits. The parent's
/// `waitpid(pid, WNOHANG)` therefore **must** return `0` — POSIX's "no child changed state" — *without
/// blocking* (a blocking `wait` would hang the run against a never-exiting twin). The parent retries
/// only the serve/park-race `-EAGAIN`, accepts `0`, writes it, and returns it; the run ends when the
/// parent finishes and the parked twin is abandoned at teardown.
const SRC_FORK_WAITPID: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { fork: 0, wait: 2 }
type 2 func (i64, i64) -> (i64)
export 0 interface "svc" 1 { fork: 2, wait: 3 }
data 16684 "svc"
data 16694 "o"
func (i32, i32) -> (i64) {
block 0 (v0: i32, vout: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  q5v0 = i64.const 4294967296
  q5v1 = i64.const 131072
  q5v2 = i64.const -4294967284
  q5v3 = i64.const 4294967295
  q5v4 = i64.const 0
  q5a0 = i64.const 17856
  i64.store q5a0 q5v0
  q5a1 = i64.const 17864
  i64.store q5a1 q5v1
  q5a2 = i64.const 17872
  i64.store q5a2 q5v2
  q5a3 = i64.const 17880
  i64.store q5a3 q5v3
  q5a4 = i64.const 17888
  i64.store q5a4 q5v4
  q5a5 = i64.const 17896
  i64.store q5a5 q5v4
  q5a6 = i64.const 17904
  i64.store q5a6 q5v4
  vs = cap.call 6 17 (i64) -> (i32) v0 (q5a0)
  vz0 = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vs, vz0)
  va0 = i64.const 16640
  vnp = i32.const 16684
  i32.store va0 vnp
  va1 = i64.const 16644
  vnl = i32.const 3
  i32.store va1 vnl
  va2 = i64.const 16648
  i32.store va2 vcap
  va3 = i64.const 16656
  vnp2 = i32.const 16694
  i32.store va3 vnp2
  va4 = i64.const 16660
  vnl2 = i32.const 1
  i32.store va4 vnl2
  va5 = i64.const 16664
  i32.store va5 vout
  q6v0 = i64.const 17179869184
  q6v1 = i64.const 135168
  q6v2 = i64.const -4294967284
  q6v3 = i64.const 4294967295
  q6v4 = i64.const 0
  q6v5 = i64.const 16640
  q6v6 = i64.const 2
  q6a0 = i64.const 17920
  i64.store q6a0 q6v0
  q6a1 = i64.const 17928
  i64.store q6a1 q6v1
  q6a2 = i64.const 17936
  i64.store q6a2 q6v2
  q6a3 = i64.const 17944
  i64.store q6a3 q6v3
  q6a4 = i64.const 17952
  i64.store q6a4 q6v4
  q6a5 = i64.const 17960
  i64.store q6a5 q6v5
  q6a6 = i64.const 17968
  i64.store q6a6 q6v6
  vc = cap.call 6 17 (i64) -> (i32) v0 (q6a0)
  vjc = cap.call 6 1 (i32) -> (i64) v0 (vc)
  return vjc
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  br 1()
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
func (i64, i64) -> (i64) {
block 0 (vpid: i64, vflags: i64) {
  vz = i32.const 0
  vt = cap.call 4294967295 12 (i64, i64) -> (i64) vz (vpid, vflags)
  return vt
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsvc = i64.const 6518387
  vzero = i64.const 0
  i64.store vzero vsvc
  voname = i64.const 111
  va8 = i64.const 8
  i64.store va8 voname
  br 1()
  }
block 1 () {
  vp0 = i64.const 0
  vl3 = i64.const 3
  vhsvc = cap.self.resolve vp0 vl3
  vc0 = i64.const 0
  vpid = cap.call 268435456 0 (i64) -> (i64) vhsvc (vc0)
  vz1 = i64.const 0
  vforkfail = i64.lt_s vpid vz1
  br_if vforkfail 1() 2(vpid)
  }
block 2 (vpid: i64) {
  vz2 = i64.const 0
  vparent = i64.ne vpid vz2
  br_if vparent 3(vpid) 5()
  }
block 3 (vpid: i64) {
  vp0b = i64.const 0
  vl3b = i64.const 3
  vhsvc2 = cap.self.resolve vp0b vl3b
  vnohang = i64.const 1
  vstatus = cap.call 268435456 1 (i64, i64) -> (i64) vhsvc2 (vpid, vnohang)
  vz3 = i64.const 0
  vraced = i64.lt_s vstatus vz3
  br_if vraced 3(vpid) 4(vstatus)
  }
block 4 (vstatus: i64) {
  vp8 = i64.const 8
  vl1 = i64.const 1
  vho = cap.self.resolve vp8 vl1
  vp16 = i64.const 16
  i64.store vp16 vstatus
  vlen = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vho (vp16, vlen)
  return vstatus
  }
block 5 () {
  vfa = i64.const 24
  vval = i32.const 7
  i32.store vfa vval
  vexp = i32.const 7
  vto = i64.const -1
  vst = i32.atomic.wait vfa vexp vto
  v99 = i64.const 99
  return v99
  }
}
"#;

#[test]
fn waitpid_wnohang_returns_zero_for_a_still_running_twin_without_blocking() {
    let m = module(SRC_FORK_WAITPID);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 18);
    let sink = host.shared_stdout();
    let out_h = host.grant_stream(StreamRole::Out);
    let mut fuel = 60_000_000u64;
    let r = run_with_host(
        &m,
        0,
        &[Value::I32(ih), Value::I32(out_h)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    // The twin parks forever, so `waitpid(pid, WNOHANG)` returned 0 — non-blocking, no hang.
    assert_eq!(
        r,
        vec![Value::I64(0)],
        "WNOHANG returned 0 for a still-running (never-exiting) twin"
    );
    let bytes = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
    assert_eq!(bytes.len(), 8, "the parent wrote its WNOHANG result once");
    let status = i64::from_le_bytes(bytes[..8].try_into().unwrap());
    assert_eq!(status, 0, "WNOHANG on a running child is 0");
}
