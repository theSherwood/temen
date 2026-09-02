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
