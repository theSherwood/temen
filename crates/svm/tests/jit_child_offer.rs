//! CALLS.md 5c.0 — `child_offer` (Instantiator op 14) on the **JIT**: minting a live-callee offer
//! over a spawned granted child's shared powerbox. Before this slice the JIT answered a blanket
//! probeable `-EINVAL` ("the JIT runtime has neither [scheduler nor child registry]"); now the
//! nursery retains a counted ref to each granted child's `Arc<Mutex<Host>>` and op 14 mints the
//! same live-impl handle the interp's op-14 arm mints (shape from the child's `self_module`,
//! structurally interned — D59 gives both backends the identical type id).
//!
//! With 5c.1b the call **through** the minted handle completes on both backends (the JIT via the
//! parked transport: enqueue on the child's shared cell, the child's blocking `svc.wait` serves,
//! the thread-blocked caller wakes with the reply) — the equality flip 5c.0 promised.

use std::sync::Arc;
use svm_interp::{run_capture_reserved_with_host, Host, Value};
use svm_jit::{compile_and_run_capture_reserved_with_host_ex, GrantChildHooks, JitOutcome};
use svm_text::parse_module;
use svm_verify::verify_module;

fn grant_hooks() -> GrantChildHooks {
    GrantChildHooks {
        build: svm_run::grant_child_build,
        build_named: svm_run::grant_named_child_build,
        bind_imports: svm_run::child_bind_imports,
        release: svm_run::grant_child_release,
        mint: svm_run::child_offer_mint,
        thunk: svm_run::cap_thunk_locked,
        register_serve: svm_run::child_register_serve,
    }
}

/// Parent (func 0, `(Instantiator)`): spawn a same-module named-grant child (empty grant list,
/// entry 1, 64 KiB carve) via op 11, mint `child_offer(child, export 0)` via op 14, return the
/// minted handle widened to i64 (negative = the errno). Child (func 1): returns 0. Func 2 is the
/// `add` handler behind `export 0 interface "adder"`.
const MINT_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  gp = i64.const 0
  gn = i64.const 0
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  r = i64.extend_i32_s vh
  return r
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  z = i64.const 0
  return z
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  s = i64.add va vb
  return s
  }
}
"#;

/// Like [`MINT_SRC`] but the parent **calls through** the minted handle (`add(40, 2)`, type id
/// `268435456` = the first guest intern, identical on both backends by D59) and returns that.
const CALL_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  gp = i64.const 0
  gn = i64.const 0
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) vh (va, vb)
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
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  s = i64.add va vb
  return s
  }
}
"#;

/// A parent that spawns a **plain** (op 0) child — no grant hooks involved, no shared powerbox —
/// then tries op 14 on it: must refuse `-EINVAL`, fail-closed (nothing retained to offer).
const PLAIN_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ent, off, sl, q)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  r = i64.extend_i32_s vh
  return r
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  z = i64.const 0
  return z
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  s = i64.add va vb
  return s
  }
}
"#;

fn run_jit_i64_knob(src: &str, handoff: bool) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let am = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&am);
    host.set_handoff(handoff);
    let ih = host.grant_instantiator(0, 128 << 10);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &am,
        0,
        &[ih as i64],
        &[0u8; 128 << 10],
        0,
        svm_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        Some(grant_hooks()),
    )
    .expect("jit");
    match jo {
        JitOutcome::Returned(v) => v.first().copied().unwrap_or(i64::MIN),
        other => panic!("jit did not return cleanly: {other:?}"),
    }
}

fn run_jit_i64(src: &str) -> i64 {
    run_jit_i64_knob(src, false)
}

fn run_interp_i64(src: &str) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let am = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&am);
    let ih = host.grant_instantiator(0, 128 << 10);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &am,
        0,
        &[Value::I32(ih)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        &mut host,
    );
    match res.expect("interp ok").as_slice() {
        [Value::I64(v)] => *v,
        other => panic!("unexpected interp result shape: {other:?}"),
    }
}

/// **The 5c.0 pin.** Op 14 on the JIT mints a real handle (≥ 0) over a granted child — no longer
/// the blanket `-EINVAL` — and the interp mints on the same program too (mint parity).
#[test]
fn child_offer_mints_on_the_jit_and_matches_interp() {
    let jit = run_jit_i64(MINT_SRC);
    let interp = run_interp_i64(MINT_SRC);
    assert!(jit >= 0, "JIT op 14 mints a live-impl handle, got {jit}");
    assert!(interp >= 0, "interp op 14 mints, got {interp}");
}

/// **The 5c.1 equality flip** (promised in the 5c.0 PR): a call **through** the minted handle now
/// completes on BOTH backends — the interp via its eval-loop transport, the JIT via the 5c.1b
/// parked transport (enqueue on the child's shared cell → the child's `svc.wait` block-waits,
/// serves, settles → the thread-blocked caller wakes with the reply).
#[test]
fn call_through_minted_offer_completes_on_both_backends() {
    assert_eq!(
        run_interp_i64(CALL_SRC),
        42,
        "interp: the live call enqueues, the child serves, add(40,2)"
    );
    assert_eq!(
        run_jit_i64(CALL_SRC),
        42,
        "JIT: the parked transport — enqueue, child serves, thread-blocked caller wakes"
    );
}

/// Op 14 on a **plain** (op 0) child stays `-EINVAL` fail-closed on the JIT: nothing was shared,
/// nothing is offered.
#[test]
fn child_offer_on_a_plain_child_refuses() {
    assert_eq!(run_jit_i64(PLAIN_SRC), -22);
}

/// CALLS.md 5c.2 — the **settlement** module: the parent calls `add(40,2)` through the minted
/// offer AND joins the child, returning `call*100 + join`. The child's `svc.wait` returns its
/// served count — which, under the §10.2 settlement rule, must observe the dispatch **whichever
/// transport served it** (enqueue+park or a claimed inline handoff): 42*100 + 1 = 4201, always.
const JOIN_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  gp = i64.const 0
  gn = i64.const 0
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  vspin = i32.const 2000000
  br 1(vspin, vh, vch, vinst)
}
block 1 (vk0: i32, vh1: i32, vch1: i32, vin1: i32) {
  vone = i32.const 1
  vk1 = i32.sub vk0 vone
  br_if vk1 1(vk1, vh1, vch1, vin1) 2(vh1, vch1, vin1)
}
block 2 (vh2: i32, vch2: i32, vin2: i32) {
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) vh2 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) vin2 (vch2)
  vk = i64.const 100
  vm = i64.mul vr vk
  vs = i64.add vm vj
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
  s = i64.add va vb
  return s
  }
}
"#;

/// Like [`JOIN_SRC`] but the handler **parks mid-serve** (a 2ms timed `atomic.wait` that times
/// out) before returning — CALLS.md 5c.4: under handoff the *claimer's* thread blocks inside the
/// inline invoke (the §10.2 arm-6 "thread-blocks (JIT)" flavor); under the parked transport the
/// child's thread blocks. Same observables either way.
const PARK_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  gp = i64.const 0
  gn = i64.const 0
  ent = i64.const 1
  off = i64.const 65536
  sl = i64.const 16
  q = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vinst (gp, gn, ent, off, sl, q)
  vexp = i64.const 0
  vh = cap.call 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  vspin = i32.const 2000000
  br 1(vspin, vh, vch, vinst)
}
block 1 (vk0: i32, vh1: i32, vch1: i32, vin1: i32) {
  vone = i32.const 1
  vk1 = i32.sub vk0 vone
  br_if vk1 1(vk1, vh1, vch1, vin1) 2(vh1, vch1, vin1)
}
block 2 (vh2: i32, vch2: i32, vin2: i32) {
  va = i64.const 40
  vb = i64.const 2
  vr = cap.call 268435456 0 (i64, i64) -> (i64) vh2 (va, vb)
  vj = cap.call 6 1 (i32) -> (i64) vin2 (vch2)
  vk = i64.const 100
  vm = i64.mul vr vk
  vs = i64.add vm vj
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
  vaddr = i64.const 8
  vexp = i32.const 0
  vto = i64.const 2000000
  vst = i32.atomic.wait vaddr vexp vto
  s = i64.add va vb
  return s
  }
}
"#;

/// **The 5c.2 pin**: handoff-on ≡ handoff-off ≡ interp, on the result AND the callee's
/// served-count observation (the §10.2 settlement rule). With the knob on, whether a given run
/// claims (child parked in time) or falls back to enqueue is a race — and the pin's point is
/// that the observables are identical either way.
#[test]
fn direct_handoff_matches_parked_and_settles() {
    assert_eq!(run_interp_i64(JOIN_SRC), 4201, "interp: 42*100 + served(1)");
    assert_eq!(
        run_jit_i64_knob(JOIN_SRC, false),
        4201,
        "JIT parked transport"
    );
    assert_eq!(run_jit_i64_knob(JOIN_SRC, true), 4201, "JIT direct handoff");
}

/// **The 5c.4 pin**: a handler that parks mid-serve (timed futex wait) completes identically
/// under handoff (the claimer's thread blocks inline — the arm-6 thread-block flavor) and under
/// the parked transport (the child's thread blocks).
#[test]
fn direct_handoff_with_parking_handler_matches() {
    assert_eq!(run_interp_i64(PARK_SRC), 4201, "interp");
    assert_eq!(
        run_jit_i64_knob(PARK_SRC, false),
        4201,
        "JIT parked transport"
    );
    assert_eq!(run_jit_i64_knob(PARK_SRC, true), 4201, "JIT direct handoff");
}
