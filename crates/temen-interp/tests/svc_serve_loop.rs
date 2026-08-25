//! §3.6 slice 2 — the **serve-loop core**: a domain's offers served as handlers over its
//! **one world** (IMPORTS.md §3.6). Dispatches queue on the domain's bounded inbound queue
//! (embedder-enqueued this slice; the cross-domain caller is the caller-parking slice) and are
//! admitted at the guest's `svc.poll` service point (`cap.call CAP_SELF_TYPE_ID 9` — riding
//! the reserved self-namespace dispatch, no wire change). A handler runs over the SAME live
//! window and powerbox as `main` — what `main` writes, handlers read, and vice versa. There
//! is no second state: the passive instance's two-world split is what §3.6 dissolves.

use std::sync::Arc;
use temen_interp::{run_with_host, Host, Value, CAP_SELF_SVC_POLL, SVC_QUEUE_CAP};

/// The serving domain. Offer "counter" op 0 = func 1 `bump(x) -> old + x`, where `old` is the
/// LIVE value at mem[16384] (above the #1094 NULL guard) — the handler both reads and writes
/// main's memory (one world).
/// `main` (func 0) seeds mem[16384] = 7, `svc.poll`s (serving everything queued), then returns
/// `served * 1000 + mem[16384]` — so the return value proves both the served count and that the
/// handlers' writes landed in main's own window.
const SERVER: &str = r#"
memory 16
type 0 func (i64) -> (i64)
type 1 interface { bump: 0 }
export 0 interface "counter" 1 { bump: 1 }

func () -> (i64) {
block 0 () {
  va = i64.const 16384
  vseed = i64.const 7
  i64.store va vseed
  vz = i32.const 0
  vn = cap.call 4294967295 9 () -> (i64) vz ()
  vafter = i64.load va
  vk = i64.const 1000
  vm = i64.mul vn vk
  vr = i64.add vm vafter
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  va = i64.const 16384
  vold = i64.load va
  vnew = i64.add vold vx
  i64.store va vnew
  return vold
  }
}
"#;

fn server_module() -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(SERVER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

#[test]
fn queued_dispatches_run_as_handlers_over_the_one_world() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    // Two dispatches queued before the run: bump(5) then bump(30).
    let t1 = host.svc_enqueue(0, 0, vec![5]).expect("enqueue 1");
    let t2 = host.svc_enqueue(0, 0, vec![30]).expect("enqueue 2");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    // served=2; mem[16384] went 7 → 12 → 42 (handlers mutated MAIN's window: one world).
    assert_eq!(
        r,
        vec![Value::I64(2042)],
        "served*1000 + final live counter"
    );
    // Completion cells: each handler returned the counter's value BEFORE its bump — 7 and 12 —
    // proving the handlers observed main's seed and each other's effects, serialized in order.
    assert_eq!(host.svc_result(t1), Some(7));
    assert_eq!(host.svc_result(t2), Some(12));
    assert_eq!(host.svc_result(t1), None, "a completion cell drains once");
}

#[test]
fn an_empty_poll_serves_nothing_and_returns_zero() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(r, vec![Value::I64(7)], "0 served, seed untouched");
}

#[test]
fn the_queue_is_bounded_and_refuses_fail_closed() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    for i in 0..SVC_QUEUE_CAP {
        assert!(
            host.svc_enqueue(0, 0, vec![i as i64]).is_some(),
            "under cap"
        );
    }
    assert_eq!(
        host.svc_enqueue(0, 0, vec![0]),
        None,
        "a full queue refuses the enqueue — backpressure at the enqueuer, never buffering"
    );
    // An unservable target (no such offer/op) also refuses at enqueue: the queue only ever
    // holds dispatches the domain can actually serve.
    assert_eq!(host.svc_enqueue(9, 0, vec![0]), None, "unknown export");
    assert_eq!(host.svc_enqueue(0, 7, vec![0]), None, "unknown op");
}

#[test]
fn an_arity_mismatched_dispatch_gets_a_probeable_errno_and_serving_continues() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let bad = host.svc_enqueue(0, 0, vec![1, 2, 3]).expect("enqueues"); // bump takes 1 arg
    let good = host.svc_enqueue(0, 0, vec![5]).expect("enqueues");
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("run");
    // Only the good dispatch served (7 → 12); the bad one errnos in its cell.
    assert_eq!(r, vec![Value::I64(1012)]);
    assert_eq!(host.svc_result(bad), Some(-22), "-EINVAL, probeable");
    assert_eq!(host.svc_result(good), Some(7));
}

/// The self-namespace op number is part of the public surface (jacl will emit it): pin it.
#[test]
fn the_svc_poll_op_number_is_pinned() {
    assert_eq!(CAP_SELF_SVC_POLL, 9);
    assert_eq!(temen_interp::CAP_SELF_SVC_WAIT, 10);
}

/// §3.6 parity — **the bytecode entry serves via the oracle fallback**: `run_with_host_fast`
/// (the `Backend::Bytecode` entry) declines to compile the svc ops and runs the whole module
/// on the tree-walker, so a serving domain behaves **identically** through either entry.
/// Fallback is the same free-correctness path the Instantiator ops already ride.
#[test]
fn the_bytecode_entry_serves_identically_via_the_oracle_fallback() {
    let m = server_module();
    let mut host = Host::new();
    host.set_self_module(&m);
    let t1 = host.svc_enqueue(0, 0, vec![5]).expect("enqueue 1");
    let t2 = host.svc_enqueue(0, 0, vec![30]).expect("enqueue 2");
    let mut fuel = u64::MAX;
    let r = temen_interp::run_with_host_fast(&m, 0, &[], &mut fuel, &mut host).expect("run");
    assert_eq!(r, vec![Value::I64(2042)], "identical to the tree-walk run");
    assert_eq!(host.svc_result(t1), Some(7));
    assert_eq!(host.svc_result(t2), Some(12));
}

/// §3.6 parity — a backend tier **without** eval-loop servicing answers both svc ops with a
/// probeable `-EINVAL` from the one shared host dispatch (the JIT's route): refusal, never a
/// trap, never a wrong answer — pinned directly at the shared entry.
#[test]
fn a_non_serving_tier_refuses_both_svc_ops_probeably() {
    let mut host = Host::new();
    for op in [CAP_SELF_SVC_POLL, temen_interp::CAP_SELF_SVC_WAIT] {
        let r = host
            .cap_dispatch_slots(temen_ir::CAP_SELF_TYPE_ID, op, 0, &[], None)
            .expect("refusal, not a trap");
        assert_eq!(r, vec![-22], "probeable -EINVAL for svc op {op}");
    }
}

/// §3.6 slice 3 — **caller-side parking, end to end**: a parent spawns a serving child
/// (§14 same-module), mints a live-callee offer over the child's export
/// (`Instantiator.child_offer`, op 14), and calls through it. The call enqueues on the
/// child's queue and parks the parent; the child's `svc.wait` (op 10) wakes on the enqueue
/// (or finds the work already queued — both orders are correct), serves `add(40, 2)` as a
/// handler, and the reply wakes the parent with 42. The child returns its served count,
/// which the parent reads back through `join` — proving the whole caller ↔ servicer
/// round-trip parked and woke rather than deadlocked. The offer's structural type id is
/// the first guest intern (`GUEST_IMPL_BASE` = 268435456), pinned by D59 determinism.
const CALLER_PARKING: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

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
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

/// §3.6 slice 4 — the **slot route**: the same round-trip as the direct form, but the caller
/// attaches the live-callee cap into a rebindable import slot and calls `call.import 0` — the
/// discovery-then-attach pattern over a live domain. Same enqueue/park/reply machinery.
const SLOT_CALLER: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }
import 0 "svc.add" (i64, i64) -> (i64) rebindable

func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 65536
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
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  vst = import.attach 0 v7
  va = i64.const 40
  vb = i64.const 2
  vr = call.import 0 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

#[test]
fn a_slot_attached_live_call_parks_and_wakes_like_the_direct_form() {
    let m = Arc::new({
        let m = temen_text::parse_module(SLOT_CALLER).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    });
    let mut host = Host::new();
    host.set_self_module(&m);
    // The rebindable slot's template: typed to the (first-interned) offer interface, unbound.
    host.set_import_bindings(vec![temen_interp::BoundImport {
        type_id: 268435456, // GUEST_IMPL_BASE — the offer's structural intern (D59-deterministic)
        op: 0,
        handle: 0,
        bound: false,
        rebindable: true,
    }]);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "attach → call.import through a live callee: park, serve (svc.wait sugar), reply, join"
    );
}

/// §3.6 slice 4 — the `svc.*` sugar round-trips: `svc.wait v0` in SLOT_CALLER above already
/// proves parse; this pins print→parse stability and the desugared identity.
#[test]
fn svc_sugar_round_trips_and_desugars_to_the_reserved_dispatch() {
    let m = temen_text::parse_module(SLOT_CALLER).expect("parse");
    let printed = temen_text::print_module(&m);
    assert!(
        printed.contains("svc.wait v"),
        "the printer emits the greppable sugar"
    );
    let m2 = temen_text::parse_module(&printed).expect("reparse");
    assert_eq!(m, m2, "text round-trip");
    let m3 = temen_encode::decode_module(&temen_encode::encode_module(&m)).expect("decode");
    assert_eq!(
        m, m3,
        "wire round-trip (sugar is pure spelling — no wire change)"
    );
}

/// §3.6 — **separate-module serving children**: the child domain runs its OWN module
/// (`instantiate_module`, op 5) with its own offers, and the parent wires a live offer over
/// the child's export via the same `child_offer` (op 14). The offer's shape is the CHILD
/// module's export — the parent registers no self module at all, pinning that the wirer's
/// own program is irrelevant to the wire — interned structurally into the parent's table
/// (D59: first guest intern = `GUEST_IMPL_BASE`, same as the same-module form). Same
/// enqueue/park/`svc.wait`-serve/reply/join round-trip: join(served=1)*100 + add(40,2) = 142.
const SEPARATE_MODULE_CALLER: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vlog, vq)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vk = i64.const 100
  vm = i64.mul vj vk
  vs = i64.add vm vr
  return vs
  }
}
"#;

/// The child's own program: its own memory declaration (the carve must equal it — §14
/// transparency), its own offer, its own serve loop. Entry = func 0 (`svc.wait`, return the
/// served count to the joiner); `add` = func 1.
const SEPARATE_MODULE_SERVER: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

#[test]
fn a_separate_module_child_serves_its_own_offers() {
    let a = Arc::new({
        let m = temen_text::parse_module(SEPARATE_MODULE_CALLER).expect("parse caller");
        temen_verify::verify_module(&m).expect("verify caller");
        m
    });
    let b = temen_text::parse_module(SEPARATE_MODULE_SERVER).expect("parse server");
    temen_verify::verify_module(&b).expect("verify server");
    let mut host = Host::new();
    // Deliberately NO set_self_module on the parent: the offer's shape is the child's.
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "join(served=1)*100 + add(40,2) — a foreign program served the parent's live call"
    );
    let _ = a;
}

/// A `child_offer` naming an export the child's module doesn't have refuses with a probeable
/// `-EINVAL` — resolved against the CHILD's module (which has export 0 only), never the
/// wirer's. The child here polls-and-returns (nothing to serve), so the parent's `join`
/// completes the run cleanly after the refused wire.
const SEPARATE_MODULE_BAD_EXPORT: &str = r#"
memory 17

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  ventry = i64.const 0
  voff = i64.const 65536
  vlog = i64.const 12
  vq = i64.const 0
  v5 = cap.call 6 5 (i64, i64, i64, i64, i64) -> (i32) v0 (vmh, ventry, voff, vlog, vq)
  v6 = i64.const 9
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  vj = cap.call 6 1 (i32) -> (i64) v0 (v5)
  vr = i64.extend_i32_s v7
  vs = i64.add vr vj
  return vs
  }
}
"#;

/// The bad-export test's child: same offer surface, but the entry `svc.poll`s (serving the
/// nothing that's queued) and returns 0 — so it completes without a caller.
const SEPARATE_MODULE_POLL_SERVER: &str = r#"
memory 12
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 1 }

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.poll vz
  return vn
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

#[test]
fn a_bad_export_on_a_separate_module_child_refuses_probeably() {
    let a = temen_text::parse_module(SEPARATE_MODULE_BAD_EXPORT).expect("parse");
    temen_verify::verify_module(&a).expect("verify");
    let b = temen_text::parse_module(SEPARATE_MODULE_POLL_SERVER).expect("parse server");
    temen_verify::verify_module(&b).expect("verify server");
    let mut host = Host::new();
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hm = host.grant_module(&b);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(
        &a,
        0,
        &[Value::I32(hi), Value::I32(hm)],
        &mut fuel,
        &mut host,
    )
    .expect("run");
    assert_eq!(
        r,
        vec![Value::I64(-22)],
        "-EINVAL (plus join(0)), probeable — the wire refused, the run completed"
    );
}

/// §3.6 — **sibling-as-service**: the parent spawns serving child A, takes a live offer over
/// it (`child_offer`), and re-grants that cap into child B at spawn (`instantiate_named`,
/// op 11 — the grant record's handle field is stored at runtime). B discovers it by
/// `cap.self.resolve("adder")` in its OWN powerbox (the re-grant interned the shape there —
/// B's first guest intern, `GUEST_IMPL_BASE`) and calls through it: the call enqueues on A,
/// parks B's vCPU, A's `svc.wait` serves `add(40, 2)`, and the reply wakes B — two siblings
/// coordinating through a live peer their parent introduced, no shared memory, no parent
/// relay. Composite: join(A=1)*100 + join(B=42) = 142.
const SIBLING_AS_SERVICE: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 3 }
data 16584 "adder"

func (i32) -> (i64) {
block 0 (v0: i32) {
  vlog = i64.const 12
  vq = i64.const 0
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q2v0 = i64.const 4294967296
  q2v1 = i64.const 65536
  q2v2 = i64.const -4294967284
  q2v3 = i64.const 4294967295
  q2v4 = i64.const 0
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
  i64.store q2a5 q2v4
  q2a6 = i64.const 17712
  i64.store q2a6 q2v4
  vA = cap.call 6 17 (i64) -> (i32) v0 (q2a0)
  vz = i64.const 0
  vcap = cap.call 6 14 (i32, i64) -> (i32) v0 (vA, vz)
  va1 = i64.const 16640
  vv1 = i32.const 16584
  i32.store va1 vv1
  va2 = i64.const 16644
  vv2 = i32.const 5
  i32.store va2 vv2
  va3 = i64.const 16648
  i32.store va3 vcap
  ; spawn via record (op 17): entry=2 off=69632 sl=12 quota=0
  q3v0 = i64.const 8589934592
  q3v1 = i64.const 69632
  q3v2 = i64.const -4294967284
  q3v3 = i64.const 4294967295
  q3v4 = i64.const 0
  q3v5 = i64.const 16640
  q3v6 = i64.const 1
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
  i64.store q3a5 q3v5
  q3a6 = i64.const 17776
  i64.store q3a6 q3v6
  vB = cap.call 6 17 (i64) -> (i32) v0 (q3a0)
  vjB = cap.call 6 1 (i32) -> (i64) v0 (vB)
  vjA = cap.call 6 1 (i32) -> (i64) v0 (vA)
  vk = i64.const 100
  vm = i64.mul vjA vk
  vs = i64.add vm vjB
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = svc.wait vz
  return vn
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vnm = i64.const 491327349857
  vza = i64.const 0
  i64.store vza vnm
  vp = i64.const 0
  vl = i64.const 5
  vh = cap.self.resolve vp vl
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) vh (va, vb)
  return vr
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

#[test]
fn a_sibling_calls_a_sibling_through_a_regranted_live_offer() {
    let m = Arc::new({
        let m = temen_text::parse_module(SIBLING_AS_SERVICE).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    });
    let mut host = Host::new();
    host.set_self_module(&m);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "B's call parked, A served, the reply woke B — through a parent-regranted live offer"
    );
}

#[test]
fn a_caller_parks_on_a_live_child_and_wakes_with_the_reply() {
    let m = Arc::new({
        let m = temen_text::parse_module(CALLER_PARKING).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    });
    let mut host = Host::new();
    host.set_self_module(&m);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(142)],
        "join(served=1)*100 + add(40,2) — the parked caller woke with the handler's reply"
    );
}
