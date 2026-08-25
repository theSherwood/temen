//! I49 — a **serve chain**: a serving domain's handler that itself calls *another* server through
//! a live offer. `ticket_waiters` is keyed by `(callee domain, dispatch ticket)`, not the ticket
//! alone, because tickets are per-callee-domain (each host's `svc_next_ticket` starts at 0). With
//! the bare-ticket key, `root -> C1.fwd -> C2.leaf` deadlocked: root parked on C1's ticket 0, then
//! C1's handler parked on C2's ticket 0 — **clobbering root's waiter** — so when C1's handler
//! finally returned, its reply to ticket 0 found no waiter and stashed, and root hung forever. This
//! is a general serving-correctness bug (a front-end server delegating to a back-end is the common
//! jacl shape), reproduced here with zero durability.

use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;
use temen_interp::{run_with_host, Host, Trap, Value};

fn module(text: &str) -> Arc<temen_ir::Module> {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
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
  ; spawn via record (op 17): entry=2 off=262144 sl=18 quota=0
  q0v0 = i64.const 8589934592
  q0v1 = i64.const 262144
  q0v2 = i64.const -4294967278
  q0v3 = i64.const 4294967295
  q0v4 = i64.const 0
  q0a0 = i64.const 1152
  i64.store q0a0 q0v0
  q0a1 = i64.const 1160
  i64.store q0a1 q0v1
  q0a2 = i64.const 1168
  i64.store q0a2 q0v2
  q0a3 = i64.const 1176
  i64.store q0a3 q0v3
  q0a4 = i64.const 1184
  i64.store q0a4 q0v4
  q0a5 = i64.const 1192
  i64.store q0a5 q0v4
  q0a6 = i64.const 1200
  i64.store q0a6 q0v4
  vc1 = call.cap 6 17 (i64) -> (i32) v0 (q0a0)
  vexp = i64.const 1
  vh1 = call.cap 6 14 (i32, i64) -> (i32) v0 (vc1, vexp)
  varg = i64.const 7
  vr = call.cap 268435456 0 (i64) -> (i64) vh1 (varg)
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
  ; spawn via record (op 17): entry=4 off=131072 sl=17 quota=0
  q1v0 = i64.const 17179869184
  q1v1 = i64.const 131072
  q1v2 = i64.const -4294967279
  q1v3 = i64.const 4294967295
  q1v4 = i64.const 0
  q1a0 = i64.const 1216
  i64.store q1a0 q1v0
  q1a1 = i64.const 1224
  i64.store q1a1 q1v1
  q1a2 = i64.const 1232
  i64.store q1a2 q1v2
  q1a3 = i64.const 1240
  i64.store q1a3 q1v3
  q1a4 = i64.const 1248
  i64.store q1a4 q1v4
  q1a5 = i64.const 1256
  i64.store q1a5 q1v4
  q1a6 = i64.const 1264
  i64.store q1a6 q1v4
  vc2 = call.cap 6 17 (i64) -> (i32) vh (q1a0)
  vexp = i64.const 0
  vh2 = call.cap 6 14 (i32, i64) -> (i32) vh (vc2, vexp)
  vk = i64.const 65600
  vh2w = i64.extend_i32_u vh2
  i64.store vk vh2w
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vk = i64.const 65600
  vh2l = i64.load vk
  vh2 = i32.wrap_i64 vh2l
  vr = call.cap 268435456 0 (i64) -> (i64) vh2 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
  return vn
  }
}
"#;

/// Drive the chain on a worker thread under a wall-clock **watchdog** (ISSUES.md I52).
///
/// The forwarding-chain rendezvous can, on the slower/serialized macOS + Windows CI runners, hit a
/// wake-ordering window that strands a vCPU parked forever (a `BlockedTicket`/`BlockedSvc` with no
/// wake left to fire). The cooperative scheduler's `worker_loop` then blocks on its idle condvar
/// with `live > 0` — the multi-worker `drive` path shuts down only when `live == 0`, so it has no
/// quiescence-deadlock detector (unlike the deterministic explorer) and the run never returns. That
/// hangs the whole `svc_serve_chain` test binary to the job's `timeout-minutes` ceiling (cancelled
/// after ~45 min, taking sibling jobs and runner budget with it).
///
/// This run is tiny — it finishes in milliseconds on any runner — so a generous wall-clock bound
/// converts that intermittent *hang* into an intermittent *fast failure*: CI goes red in seconds and
/// a re-run clears it, instead of burning the runner. **Fail-fast mitigation, not a root-cause fix**
/// (the wake-ordering window is still open; see ISSUES.md I52). On the timeout the run's worker
/// threads are left parked and reaped at process exit — they hold no cross-test locks, so leaking
/// them cannot wedge the rest of the binary.
fn run_chain_with_watchdog() -> Result<Vec<Value>, Trap> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let m = module(CHAIN);
        let mut host = Host::new();
        host.set_self_module(&m);
        let ih = host.grant_instantiator(0, 1u64 << 19);
        let mut fuel = 20_000_000u64;
        let r = run_with_host(&m, 0, &[Value::I32(ih)], &mut fuel, &mut host);
        // The receiver is gone if the watchdog already fired; the send error is expected there.
        let _ = tx.send(r);
    });
    rx.recv_timeout(Duration::from_secs(60)).unwrap_or_else(|_| {
        panic!(
            "I52: root -> C1.fwd -> C2.leaf serve chain did not complete within 60s (it normally \
             finishes in milliseconds) — the forwarding-chain rendezvous stranded a parked vCPU and \
             the scheduler's worker_loop blocked forever on its idle condvar. Fail-fast watchdog \
             fired instead of hanging the runner to the job's timeout ceiling; a re-run typically \
             clears it. See ISSUES.md I52."
        )
    })
}

#[test]
fn a_handler_forwarding_to_another_server_completes() {
    let r = run_chain_with_watchdog();
    assert_eq!(
        r,
        Ok(vec![Value::I64(107)]),
        "root -> C1.fwd -> C2.leaf(7) -> 107 threads back through the chain \
         (the two hops' ticket 0 no longer collide)"
    );
}
