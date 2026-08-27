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

#[path = "support/grant_hooks.rs"]
mod grant_hooks_mod;
use grant_hooks_mod::grant_hooks;

use std::sync::Arc;
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

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
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q0v0 = i64.const 4294967296
  q0v1 = i64.const 65536
  q0v2 = i64.const -4294967280
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
  vch = call.cap 6 17 (i64) -> (i32) vinst (q0a0)
  vexp = i64.const 0
  vh = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
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
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q1v0 = i64.const 4294967296
  q1v1 = i64.const 65536
  q1v2 = i64.const -4294967280
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
  vch = call.cap 6 17 (i64) -> (i32) vinst (q1a0)
  vexp = i64.const 0
  vh = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
  va = i64.const 40
  vb = i64.const 2
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vh (va, vb)
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
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  s = i64.add va vb
  return s
  }
}
"#;

/// A parent that spawns a grant-free record child, then tries op 14 on it. With the grant hooks
/// installed the record routes through the named path (retained → mints, interp parity); on a
/// hookless harness nothing is retained and op 14 refuses `-EINVAL` fail-closed.
const PLAIN_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32) -> (i64) {
block 0 (vinst: i32) {
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q2v0 = i64.const 4294967296
  q2v1 = i64.const 65536
  q2v2 = i64.const -4294967280
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
  vch = call.cap 6 17 (i64) -> (i32) vinst (q2a0)
  vexp = i64.const 0
  vh = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
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
        temen_run::cap_thunk,
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

/// Like [`run_jit_i64`] but with NO grant hooks installed — the bare-embedder shape: record
/// spawns take the plain thunk, children are not retained, op 14 refuses.
fn run_jit_i64_hookless(src: &str) -> i64 {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let am = Arc::new(m);
    let mut host = Host::new();
    host.set_self_module(&am);
    let ih = host.grant_instantiator(0, 128 << 10);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &am,
        0,
        &[ih as i64],
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        None,
    )
    .expect("jit");
    match jo {
        JitOutcome::Returned(v) => v.first().copied().unwrap_or(i64::MIN),
        other => panic!("jit did not return cleanly: {other:?}"),
    }
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

/// §3d flipped this pin to **interp parity**: the record spawn routes through the named path
/// whenever the grant hooks are installed (this harness installs them), so the child is retained
/// and op 14 mints — exactly the interpreter, which retains every child's powerbox. The
/// fail-closed edge that remains is the **hookless** embedder: with no grant hooks there is no
/// shared powerbox, nothing is retained, and op 14 stays `-EINVAL`.
#[test]
fn child_offer_on_a_hooked_record_child_mints_hookless_refuses() {
    assert!(
        run_jit_i64(PLAIN_SRC) >= 0,
        "hooked harness: record children are retained (interp parity)"
    );
    assert_eq!(
        run_jit_i64_hookless(PLAIN_SRC),
        -22,
        "hookless: fail closed"
    );
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
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q3v0 = i64.const 4294967296
  q3v1 = i64.const 65536
  q3v2 = i64.const -4294967280
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
  vch = call.cap 6 17 (i64) -> (i32) vinst (q3a0)
  vexp = i64.const 0
  vh = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
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
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vh2 (va, vb)
  vj = call.cap 6 1 (i32) -> (i64) vin2 (vch2)
  vk = i64.const 100
  vm = i64.mul vr vk
  vs = i64.add vm vj
  return vs
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
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
  ; spawn via record (op 17): entry=1 off=65536 sl=16 quota=0
  q4v0 = i64.const 4294967296
  q4v1 = i64.const 65536
  q4v2 = i64.const -4294967280
  q4v3 = i64.const 4294967295
  q4v4 = i64.const 0
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
  i64.store q4a5 q4v4
  q4a6 = i64.const 17840
  i64.store q4a6 q4v4
  vch = call.cap 6 17 (i64) -> (i32) vinst (q4a0)
  vexp = i64.const 0
  vh = call.cap 6 14 (i32, i64) -> (i32) vinst (vch, vexp)
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
  vr = call.cap 268435456 0 (i64, i64) -> (i64) vh2 (va, vb)
  vj = call.cap 6 1 (i32) -> (i64) vin2 (vch2)
  vk = i64.const 100
  vm = i64.mul vr vk
  vs = i64.add vm vj
  return vs
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i32.const 0
  vn = call.cap 4294967295 10 () -> (i64) vz ()
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
