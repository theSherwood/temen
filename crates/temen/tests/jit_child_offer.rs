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

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;
use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Value};
use temen_ir::{Module, SpawnRec};
use temen_jit::{compile_and_run_capture_reserved_with_host_ex, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Parent (func 0, `(Instantiator, Budget)`): spawn a same-module child (entry 1) detached through
/// a v1 record paid from the `Budget`, mint `child_offer(child, export 0)` via op 14, return the
/// minted handle widened to i64 (negative = the errno). Child (func 1): returns 0. Func 2 is the
/// `add` handler behind `export 0 interface "adder"`.
const MINT_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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

/// `src` with its spawn record (at 17536), and a host for it: the parent's three args, an
/// `Instantiator`, the `Budget` that pays for the child, and the child, func 1's child image (#2219).
fn setup(src: &str) -> (Arc<Module>, Host, [i32; 3]) {
    let src = format!("{src}{}", rec::segment(17536, &SpawnRec::v1(0)));
    let m = parse_module(&src).expect("parse");
    verify_module(&m).expect("verify");
    let mut host = Host::new();
    let ih = host.grant_instantiator(0, 128 << 10);
    let bh = host.grant_budget(-1, 1 << 20, -1);
    let ch = host.grant_module(&temen_ir::child_image_at(&m, 1).expect("child image"));
    (Arc::new(m), host, [ih, bh, ch])
}

fn run_jit_i64_knob(src: &str, handoff: bool) -> i64 {
    let src = src.to_string();
    let (tx, rx) = std::sync::mpsc::channel();
    // The run is on its own thread so a hang (a call the child never serves, a join on it) fails
    // this test instead of stalling the suite.
    std::thread::spawn(move || {
        let (am, mut host, args) = setup(&src);
        host.set_handoff(handoff);
        let (jo, _) = temen_run::jit_cap_run(
            &am,
            0,
            &args.map(i64::from),
            &MemLayout::image(vec![0u8; 128 << 10]),
            0,
            0,
            &mut host,
            None,
        )
        .expect("jit");
        let _ = tx.send(jo);
    });
    let jo = rx
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("the JIT run hung");
    match jo {
        JitOutcome::Returned(v) => v.first().copied().unwrap_or(i64::MIN),
        other => panic!("jit did not return cleanly: {other:?}"),
    }
}

fn run_jit_i64(src: &str) -> i64 {
    run_jit_i64_knob(src, false)
}

/// Like [`run_jit_i64`] but with NO grant hooks installed — the bare-embedder shape: nothing can
/// build a detached child's powerbox, so the spawn fails closed.
fn run_jit_hookless(src: &str) -> JitOutcome {
    let (am, mut host, args) = setup(src);
    let (jo, _jmem) = compile_and_run_capture_reserved_with_host_ex(
        &am,
        0,
        &args.map(i64::from),
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
        None,
        None,
    )
    .expect("jit");
    jo
}

fn run_interp_i64(src: &str) -> i64 {
    let (am, mut host, args) = setup(src);
    let mut fuel = 5_000_000u64;
    let (res, _snap) = run_capture_reserved_with_host(
        &am,
        0,
        &args.map(Value::I32),
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

/// §3d flipped this pin to **interp parity**: with the grant hooks installed (this harness installs
/// them) the record child is retained and op 14 mints — exactly the interpreter, which retains every
/// child's powerbox. The fail-closed edge that remains is the **hookless** embedder: with no grant
/// hooks nothing can build a detached child's powerbox, so the spawn itself is a `CapFault`.
#[test]
fn child_offer_on_a_hooked_record_child_mints_hookless_refuses() {
    assert!(
        run_jit_i64(PLAIN_SRC) >= 0,
        "hooked harness: record children are retained (interp parity)"
    );
    assert_eq!(
        run_jit_hookless(PLAIN_SRC),
        JitOutcome::Trapped(TrapKind::CapFault),
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

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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

/// #2160 — like [`JOIN_SRC`] but the handler first calls `svc.poll` itself, a serve nested under
/// the running handler, and returns what that answered. The oracle refuses it with `-EINVAL`, so the
/// parent sees `-22*100 + served(1)`.
const NESTED_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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
  vz = i32.const 0
  vinner = call.cap 4294967295 9 () -> (i64) vz ()
  return vinner
  }
}
"#;

/// Like [`JOIN_SRC`] but the handler **parks mid-serve** (a 2ms timed `atomic.wait` that times
/// out) before returning — CALLS.md 5c.4: under handoff the *claimer's* thread blocks inside the
/// inline invoke (the §10.2 arm-6 "thread-blocks (JIT)" flavor); under the parked transport the
/// child's thread blocks. Same observables either way. The wait makes the module concurrent, so the
/// parent calls from a locked domain (#2139).
const PARK_SRC: &str = r#"memory 17
type 0 func (i64, i64) -> (i64)
type 1 interface { add: 0 }
export 0 interface "adder" 1 { add: 2 }

func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vbud: i32, vmod: i32) {
  vrm = i64.const 17560
  i32.store vrm vmod
  vrb = i64.const 17564
  i32.store vrb vbud
  vrp = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
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
  vaddr = i64.const 17408
  vexp = i32.const 0
  vto = i64.const 2000000
  vst = i32.atomic.wait vaddr vexp vto
  s = i64.add va vb
  return s
  }
}
"#;
// (The parking handler's futex word sits at 17408 — above the NULL guard and the control words a
// detached window keeps above it, #1206.)

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

/// #2160 — a serving child's handler that calls `svc.poll` gets `-EINVAL` on every transport: the
/// child's own serve loop (the parked transport) and the caller's thread running the handler inline
/// (direct handoff). Under handoff the nested poll used to wait on the claim its own thread held.
#[test]
fn a_childs_serve_nested_under_its_handler_is_refused() {
    assert_eq!(
        run_interp_i64(NESTED_SRC),
        -2199,
        "interp: -22*100 + served(1)"
    );
    assert_eq!(
        run_jit_i64_knob(NESTED_SRC, false),
        -2199,
        "JIT parked transport"
    );
    assert_eq!(
        run_jit_i64_knob(NESTED_SRC, true),
        -2199,
        "JIT direct handoff"
    );
}

/// #2166 — like [`JOIN_SRC`], but the handler stores to `addr` before it answers, and the parent
/// asks how the child ended with `Instantiator.wait` (op 18) rather than joining it. The parent
/// returns `call * 100000 + wait`.
fn faulting_handler_src(addr: u64) -> String {
    JOIN_SRC
        .replace(
            "vj = call.cap 6 1 (i32) -> (i64) vin2 (vch2)\n  vk = i64.const 100\n",
            "vj = call.cap 6 18 (i32) -> (i64) vin2 (vch2)\n  vk = i64.const 100000\n",
        )
        .replace(
            "block 0 (va: i64, vb: i64) {\n  s = i64.add va vb\n",
            &format!(
                "block 0 (va: i64, vb: i64) {{\n  vbad = i64.const {addr}\n  i64.store vbad va\n  s = i64.add va vb\n"
            ),
        )
}

/// #2166 — a serving child whose handler faults dies of it on every transport, as the oracle's does:
/// its caller's call answers the dead-callee errno (`CAP_REVOKED`, -9) and `wait` reports the
/// child's `MemoryFault`. Both addresses fault: the NULL guard, and the first word past the child's
/// mapped window (inside its reservation). The JIT's child routes used to lose the first fault (the
/// child kept running) and not recover the second at all (the process died of SIGSEGV).
#[test]
fn a_childs_faulting_handler_kills_the_child_on_every_transport() {
    let want = -9 * 100_000 + temen_interp::Trap::MemoryFault.code();
    for addr in [8u64, (128 << 10) + 8] {
        let src = faulting_handler_src(addr);
        assert!(
            src.contains("call.cap 6 18") && src.contains("i64.store vbad va"),
            "the rewrites must apply"
        );
        assert_eq!(run_interp_i64(&src), want, "interp, store at {addr}");
        assert_eq!(
            run_jit_i64_knob(&src, false),
            want,
            "JIT parked transport, store at {addr}"
        );
        assert_eq!(
            run_jit_i64_knob(&src, true),
            want,
            "JIT direct handoff, store at {addr}"
        );
    }
}
