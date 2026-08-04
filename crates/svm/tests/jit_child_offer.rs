//! CALLS.md 5c.0 — `child_offer` (Instantiator op 14) on the **JIT**: minting a live-callee offer
//! over a spawned granted child's shared powerbox. Before this slice the JIT answered a blanket
//! probeable `-EINVAL` ("the JIT runtime has neither [scheduler nor child registry]"); now the
//! nursery retains a counted ref to each granted child's `Arc<Mutex<Host>>` and op 14 mints the
//! same live-impl handle the interp's op-14 arm mints (shape from the child's `self_module`,
//! structurally interned — D59 gives both backends the identical type id).
//!
//! Scope pin: a call **through** the minted handle still answers the host dispatch's probeable
//! `-EINVAL` on the JIT — the cross-domain transport is 5c.1; minting is this slice. The interp
//! call-through completes (42), asserted here as each backend's *documented* value, not equality —
//! the equality pin flips in 5c.1.

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

fn run_jit_i64(src: &str) -> i64 {
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

/// A call **through** the minted handle: completes on the interp (the eval-loop transport), and
/// answers the host dispatch's probeable `-EINVAL` on the JIT until the 5c.1 transport lands.
/// Each backend's documented value — the equality pin is 5c.1's flip.
#[test]
fn call_through_minted_offer_documented_per_backend() {
    assert_eq!(
        run_interp_i64(CALL_SRC),
        42,
        "interp: the live call enqueues, the child serves, add(40,2)"
    );
    assert_eq!(
        run_jit_i64(CALL_SRC),
        -22,
        "JIT: minted but untransportable until 5c.1 — probeable, never a trap"
    );
}

/// Op 14 on a **plain** (op 0) child stays `-EINVAL` fail-closed on the JIT: nothing was shared,
/// nothing is offered.
#[test]
fn child_offer_on_a_plain_child_refuses() {
    assert_eq!(run_jit_i64(PLAIN_SRC), -22);
}
