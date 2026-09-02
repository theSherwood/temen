//! #1025 Gap-2 (guest-serves-exec-via-grandchild), increment 1 — **a serve handler services a
//! dispatch by nesting a §14 spawn+join of a grandchild.** This is the "no precedent" core of the
//! keystone: today an `exec`-shaped cap always bottoms out in a host closure (`make_exec` inline;
//! `domain_exec_with_fs` top-level). The driver guest instead answers a served call by *instantiating*
//! a phase (nifler) as its own §14 grandchild — a guest providing a spawn-backed service.
//!
//! This increment isolates the novel control-flow with the smallest possible topology: the host
//! enqueues one dispatch onto the servicer's inbound queue (the embedder-enqueue path proven in
//! `svc_serve_loop.rs`), the servicer `svc.poll`s, and its **handler op-5-spawns a toy grandchild and
//! joins it**, returning the grandchild's status. The caller-parking layer (a child *calls* the
//! offer, re-granted a self-offer over the driver's export) is increment 2; a real nifler grandchild
//! over a shared memfs is increment 3.
//!
//! Because the module both serves (`svc.poll`, `has_svc`) and instantiates (`has_instantiate`), the
//! §9 serve-qualification veto (`svc_park_veto`) folds it to the tree-walk oracle — exactly right:
//! the driver is cheap orchestration, and the *phases* (the grandchildren) are what tier up. This
//! test pins that the oracle's serve arm admits a handler that nests a spawn+join.

use std::sync::Arc;
use temen_interp::bytecode::serve_qualifies;
use temen_interp::{run_with_host, Host, Value};

/// The toy grandchild (a stand-in for a nifler phase): a separate module whose child entry (func 0,
/// `(i64)->(i64)`, the §14 starter cap ignored) returns a sentinel `99` — the "phase status" the
/// servicer's handler reads back via `join` and hands to the served caller.
const GRANDCHILD: &str = r#"
memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 99
  return vr
  }
}
"#;

/// The driver-analog servicer. It offers "svc" op 0 = the handler func 1. `main` (func 0) receives
/// its Instantiator handle (`v0`) and the grandchild module handle (`v1`), stashes both into its
/// window (mem[16384]/mem[16392], above the #1094 NULL guard) so the **handler** — which runs over
/// main's one world but can't see main's locals — can reach them, then `svc.poll`s (serving whatever
/// the host queued) and returns the served count.
///
/// The handler (func 1) is the crux: it loads the stashed Instantiator + grandchild module, op-5
/// `instantiate_module`s the grandchild into a `[65536, 65536+4096)` carve, `join`s it, and returns
/// the grandchild's status — a serve handler that services its dispatch by nesting a §14 spawn.
const SERVICER: &str = r#"
memory 17
type 0 func (i64) -> (i64)
type 1 interface { go: 0 }
export 0 interface "svc" 1 { go: 1 }

func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  a0 = i64.const 16384
  i32.store a0 v0
  a1 = i64.const 16392
  i32.store a1 v1
  vz = i32.const 0
  vn = call.cap 4294967295 9 () -> (i64) vz ()
  return vn
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  a0 = i64.const 16384
  vi = i32.load a0
  a1 = i64.const 16392
  vg32 = i32.load a1
  vgh = i64.extend_i32_u vg32
  ventry = i64.const 0
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vc = call.cap 6 5 (i64, i64, i64, i64, i64) -> (i32) vi (vgh, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) vi (vc)
  return vr
  }
}
"#;

fn module(src: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// Increment 2 — the **caller-parking round-trip**, in the topology a driver actually uses. A finding
/// from increment 1's follow-on: a domain cannot both inline-spawn a caller *and* serve that caller's
/// calls from the same task — it would have to reach its `svc.wait` while still executing the spawn.
/// The servicer must be a **separate task already parked at a service point** before the caller runs
/// (the `svc_serve_loop::a_sibling_calls_a_sibling` shape). So a conductor (func 0) spawns two
/// children over one module:
/// - **S, the servicer** (func 1 entry): stashes its own §14 Instantiator (its starter cap `v0`) for
///   the handler, then `svc.wait`s — parked, ready, before C exists.
/// - **C, the caller** (func 2 entry): writes "svc" into its own low memory then `self.resolve`s it
///   (a §14 child's small carve does not carry the module's data segment at its authored offset, so
///   the name is written locally — the `svc_serve_loop` sibling caller does the same) — the conductor
///   re-granted S's offer into C via `child_offer` + a grant record — and calls it, parking C.
/// - the **handler** (func 3, "svc.go") loads S's Instantiator and op-17-spawns a grandchild (func 4,
///   returns 99) into a sub-carve of S's own window, joins it, and returns 99 — S's serve handler
///   answering C's cap call by nesting a §14 spawn. The reply wakes C.
///
/// Composite return: `join(C) * 1000 + join(S)` = 99*1000 + 1 (S served exactly one) = 99001. This is
/// the full guest-serves-via-grandchild shape (toy grandchild): nimsem(C) calls exec(the re-granted
/// offer), the driver(S) services it by spawning nifler(the grandchild). Increment 3 swaps the toy
/// grandchild for a real nifler_ce over a shared memfs.
const SIBLING_DRIVEN: &str = r#"
memory 18
type 0 func (i64) -> (i64)
type 1 interface { go: 0 }
export 0 interface "svc" 1 { go: 3 }
data 16584 "svc"

func (i32) -> (i64) {
block 0 (v0: i32) {
  s0 = i64.const 4294967296
  a0 = i64.const 17664
  i64.store a0 s0
  s1 = i64.const 65536
  a1 = i64.const 17672
  i64.store a1 s1
  s2 = i64.const -4294967280
  a2 = i64.const 17680
  i64.store a2 s2
  s3 = i64.const 4294967295
  a3 = i64.const 17688
  i64.store a3 s3
  s4 = i64.const 0
  a4 = i64.const 17696
  i64.store a4 s4
  a5 = i64.const 17704
  i64.store a5 s4
  a6 = i64.const 17712
  i64.store a6 s4
  sp = i64.const 17664
  vS = call.cap 6 17 (i64) -> (i32) v0 (sp)
  vzero = i64.const 0
  vcap = call.cap 6 14 (i32, i64) -> (i32) v0 (vS, vzero)
  g0 = i64.const 16640
  gn = i32.const 16584
  i32.store g0 gn
  g1 = i64.const 16644
  gl = i32.const 3
  i32.store g1 gl
  g2 = i64.const 16648
  i32.store g2 vcap
  d0 = i64.const 8589934592
  b0 = i64.const 17728
  i64.store b0 d0
  d1 = i64.const 131072
  b1 = i64.const 17736
  i64.store b1 d1
  d2 = i64.const -4294967284
  b2 = i64.const 17744
  i64.store b2 d2
  d3 = i64.const 4294967295
  b3 = i64.const 17752
  i64.store b3 d3
  d4 = i64.const 0
  b4 = i64.const 17760
  i64.store b4 d4
  d5 = i64.const 16640
  b5 = i64.const 17768
  i64.store b5 d5
  d6 = i64.const 1
  b6 = i64.const 17776
  i64.store b6 d6
  bp = i64.const 17728
  vC = call.cap 6 17 (i64) -> (i32) v0 (bp)
  vjC = call.cap 6 1 (i32) -> (i64) v0 (vC)
  vjS = call.cap 6 1 (i32) -> (i64) v0 (vS)
  vk = i64.const 1000
  vm = i64.mul vjC vk
  vr = i64.add vm vjS
  return vr
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vi = i32.wrap_i64 v0
  sa = i64.const 16512
  i32.store sa vi
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  nm = i64.const 6518387
  za = i64.const 0
  i64.store za nm
  vp = i64.const 0
  vl = i64.const 3
  vh = self.resolve vp vl
  va = i64.const 7
  vr = call.cap 268435456 0 (i64) -> (i64) vh (va)
  return vr
  }
}

func (i64) -> (i64) {
block 0 (vx: i64) {
  sa = i64.const 16512
  vi = i32.load sa
  e0 = i64.const 17179869184
  f0 = i64.const 18432
  i64.store f0 e0
  e1 = i64.const 32768
  f1 = i64.const 18440
  i64.store f1 e1
  e2 = i64.const -4294967284
  f2 = i64.const 18448
  i64.store f2 e2
  e3 = i64.const 4294967295
  f3 = i64.const 18456
  i64.store f3 e3
  e4 = i64.const 0
  f4 = i64.const 18464
  i64.store f4 e4
  f5 = i64.const 18472
  i64.store f5 e4
  f6 = i64.const 18480
  i64.store f6 e4
  fp = i64.const 18432
  vG = call.cap 6 17 (i64) -> (i32) vi (fp)
  vr = call.cap 6 1 (i32) -> (i64) vi (vG)
  return vr
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 99
  return vr
  }
}
"#;

#[test]
fn a_child_cap_call_is_serviced_by_a_handler_that_nests_a_grandchild_spawn() {
    let m = module(SIBLING_DRIVEN);
    // Serves and instantiates → folded to the tree-walk oracle, same as increment 1.
    assert!(
        !serve_qualifies(&m.funcs),
        "serve+instantiate folds to the oracle"
    );
    let mut host = Host::new();
    host.set_self_module(&m);
    let hi = host.grant_instantiator(0, 1u64 << 18);
    let mut fuel = 50_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(hi)], &mut fuel, &mut host).expect("run");
    assert_eq!(
        r,
        vec![Value::I64(99001)],
        "C called S's re-granted offer → parked → S's handler op-17-spawned a grandchild and joined \
         it (99) → reply woke C → conductor joined C(99) and S(served 1): 99*1000 + 1"
    );
}

#[test]
fn a_serve_handler_services_a_dispatch_by_nesting_a_spawn_and_join() {
    let servicer = module(SERVICER);
    let grandchild = module(GRANDCHILD);

    // The servicer both serves and instantiates, so the §9 veto folds it to the oracle — pin that.
    assert!(
        !serve_qualifies(&servicer.funcs),
        "serve+instantiate is the general serve+spawn case: folded to the tree-walk oracle"
    );

    let mut host = Host::new();
    host.set_self_module(&servicer);
    let hi = host.grant_instantiator(0, 1u64 << 17);
    let hg = host.grant_module(&grandchild);
    // The host enqueues one dispatch to "svc".go (embedder-enqueue path); its result cell is filled
    // by the handler that services it.
    let ticket = host.svc_enqueue(0, 0, vec![0]).expect("enqueue go");

    let mut fuel = 50_000_000u64;
    let r = run_with_host(
        &servicer,
        0,
        &[Value::I32(hi), Value::I32(hg)],
        &mut fuel,
        &mut host,
    )
    .expect("run");

    assert_eq!(r, vec![Value::I64(1)], "main served exactly one dispatch");
    assert_eq!(
        host.svc_result(ticket),
        Some(99),
        "the handler spawned the grandchild, joined it, and returned its status — \
         a served call answered by nesting a §14 spawn"
    );
}
