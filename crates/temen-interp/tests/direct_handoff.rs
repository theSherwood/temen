//! CALLS.md 4d — **direct handoff** into a `single` process provider parked at `svc.wait`.
//!
//! A cross-domain call to a live-child process provider is served today by enqueue + `svc_wake` +
//! caller-park-on-reply. With handoff enabled (`Host::set_handoff`), a caller finding the provider's
//! serve loop parked at `svc.wait` **donates its thread** to serve the dispatch inline instead — no
//! worker wake, no reply round-trip. The invariant is **handoff-on ≡ handoff-off** on observable
//! results and served counts (transport choice may never change semantics, §9/§10.2).

use std::sync::Arc;
use temen_interp::{run_with_host, Host, Value};

/// One module, three functions: func 0 is the root caller, func 1 is a child **serve loop** (parks
/// at `svc.wait`, serves, loops), func 2 is the `add` handler behind the child's `adder` offer. The
/// root spawns the child, takes a live offer over it, and calls it **twice**: the first call parks
/// the root (the child hasn't reached `svc.wait` yet), the child serves it and re-parks — so the
/// **second** call finds the serve loop parked and takes the handoff path when it is enabled. Both
/// results are the same with handoff on or off: add(40,2) + add(10,3) = 55.
const HANDOFF_CALLER: &str = r#"
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
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q0a0)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr1 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vc = i64.const 10
  vd = i64.const 3
  vr2 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (vc, vd)
  vs = i64.add vr1 vr2
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = svc.wait vz
  br 1()
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vs = i64.add va vb
  return vs
  }
}
"#;

/// Like [`HANDOFF_CALLER`] but the `add` handler **parks mid-serve** (a 2 ms timed `atomic.wait`
/// that times out) before returning. Under handoff the second call donates its thread, the handler
/// parks mid-handoff — so there is no inline reply and the caller falls back to parking on the
/// ticket exactly as the enqueue path would (4d.2), completing when the handler's timer resumes it.
/// Same result, on or off.
const HANDOFF_PARKING_CALLER: &str = r#"
memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (v0: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 65536
  q1v2 = i64.const -4294967284
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
  v5 = cap.call 6 17 (i64) -> (i32) v0 (q1a0)
  v6 = i64.const 0
  v7 = cap.call 6 14 (i32, i64) -> (i32) v0 (v5, v6)
  va = i64.const 40
  vb = i64.const 2
  vr1 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (va, vb)
  vc = i64.const 10
  vd = i64.const 3
  vr2 = cap.call 268435456 0 (i64, i64) -> (i64) v7 (vc, vd)
  vs = i64.add vr1 vr2
  return vs
  }
}

func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1()
  }
block 1 () {
  vz = i32.const 0
  vn = svc.wait vz
  br 1()
  }
}

func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vaddr = i64.const 8
  vexp = i32.const 0
  vto = i64.const 2000000
  vst = i32.atomic.wait vaddr vexp vto
  vs = i64.add va vb
  return vs
  }
}
"#;

fn run_handoff(
    module: &Arc<temen_ir::Module>,
    handoff: bool,
) -> Result<Vec<Value>, temen_interp::Trap> {
    let mut host = Host::new();
    host.set_self_module(module);
    host.set_handoff(handoff);
    let h = host.grant_instantiator(0, 1u64 << 17);
    let mut fuel = 5_000_000u64;
    run_with_host(module, 0, &[Value::I32(h)], &mut fuel, &mut host)
}

/// **4d.1 — run-to-completion handoff ≡ enqueue+park.** The same program serves its second live
/// call by direct handoff (handoff on) or by enqueue+park (handoff off); the observable result is
/// identical, the transport-equivalence pin.
#[test]
fn direct_handoff_matches_enqueue_park_run_to_completion() {
    let m = Arc::new({
        let m = temen_text::parse_module(HANDOFF_CALLER).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    });
    let off = run_handoff(&m, false);
    let on = run_handoff(&m, true);
    assert_eq!(
        off,
        Ok(vec![Value::I64(55)]),
        "enqueue+park: add(40,2) + add(10,3) = 55"
    );
    assert_eq!(
        on, off,
        "handoff-on ≡ handoff-off: transport choice must not change the observable result"
    );
}

/// **4d.2 — mid-handoff park ≡ enqueue+park.** The handler parks mid-serve; under handoff the
/// donating caller gets no inline reply and falls back to parking on the ticket, the handler
/// resuming on its timer and replying later. Observably identical to the enqueue path.
#[test]
fn direct_handoff_matches_enqueue_park_with_a_parking_handler() {
    let m = Arc::new({
        let m = temen_text::parse_module(HANDOFF_PARKING_CALLER).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        m
    });
    let off = run_handoff(&m, false);
    let on = run_handoff(&m, true);
    assert_eq!(
        off,
        Ok(vec![Value::I64(55)]),
        "enqueue+park with a parking handler: add(40,2) + add(10,3) = 55"
    );
    assert_eq!(
        on, off,
        "handoff-on ≡ handoff-off: a mid-handoff park falls back to the ticket, same result"
    );
}
