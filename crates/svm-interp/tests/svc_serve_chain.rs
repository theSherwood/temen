//! I49 — a **serve chain**: a serving domain's handler that itself calls *another* server through
//! a live offer. `ticket_waiters` is keyed by `(callee domain, dispatch ticket)`, not the ticket
//! alone, because tickets are per-callee-domain (each host's `svc_next_ticket` starts at 0). With
//! the bare-ticket key, `root -> C1.fwd -> C2.leaf` deadlocked: root parked on C1's ticket 0, then
//! C1's handler parked on C2's ticket 0 — **clobbering root's waiter** — so when C1's handler
//! finally returned, its reply to ticket 0 found no waiter and stashed, and root hung forever. This
//! is a general serving-correctness bug (a front-end server delegating to a back-end is the common
//! jacl shape), reproduced here with zero durability.

use std::sync::Arc;
use svm_interp::{run_with_host, Host, Value};

fn module(text: &str) -> Arc<svm_ir::Module> {
    let m = svm_text::parse_module(text).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    Arc::new(m)
}

/// root spawns C1 (a server) and mints a cap over its `fwd` export; C1 spawns C2 and mints a cap
/// over its `leaf` export, stashing that cap's handle in memory. root calls `fwd(7)` through its
/// cap; C1's handler loads its stashed cap and calls `leaf(7)` through it; C2 returns `7 + 100`;
/// the reply threads back C2 -> C1's handler -> root, which returns 107. Child entries take the
/// `(i64)` starter arg the spawn-ABI enforces; C1 uses `i32.wrap_i64` on it (no durable transform
/// here, so conversions are fine) to get the `i32` instantiator handle for spawning C2.
const CHAIN: &str = r#"
memory 19
type 0 func (i64) -> (i64)
type 1 interface { call: 0 }
export 0 interface "leaf" 1 { call: 1 }
export 1 interface "fwd" 1 { call: 3 }
func (i32) -> (i64) {
block 0 (v0: i32) {
  ventry = i64.const 2
  voff = i64.const 262144
  vsl = i64.const 18
  vq = i64.const 0
  vc1 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v0 (ventry, voff, vsl, vq)
  vexp = i64.const 1
  vh1 = cap.call 6 14 (i32, i64) -> (i32) v0 (vc1, vexp)
  varg = i64.const 7
  vr = cap.call 268435456 0 (i64) -> (i64) vh1 (varg)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v1 = i64.const 100
  vr = i64.add vx v1
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vh = i32.wrap_i64 v0
  ventry = i64.const 4
  voff = i64.const 131072
  vsl = i64.const 17
  vq = i64.const 0
  vc2 = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vh (ventry, voff, vsl, vq)
  vexp = i64.const 0
  vh2 = cap.call 6 14 (i32, i64) -> (i32) vh (vc2, vexp)
  vk = i64.const 65600
  vh2w = i64.extend_i32_u vh2
  i64.store vk vh2w
  vz = i32.const 0
  vn = cap.call 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 65600
  vh2l = i64.load vk
  vh2 = i32.wrap_i64 vh2l
  vr = cap.call 268435456 0 (i64) -> (i64) vh2 (vx)
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
"#;

#[test]
fn a_handler_forwarding_to_another_server_completes() {
    let m = module(CHAIN);
    let mut host = Host::new();
    host.set_self_module(&m);
    let ih = host.grant_instantiator(0, 1u64 << 19);
    let mut fuel = 20_000_000u64;
    let r = run_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &mut host);
    assert_eq!(
        r,
        Ok(vec![Value::I64(107)]),
        "root -> C1.fwd -> C2.leaf(7) -> 107 threads back through the chain \
         (the two hops' ticket 0 no longer collide)"
    );
}
