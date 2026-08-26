//! PROCESS.md S1 — the JIT **per-carve compile cache**. The JIT re-compiles a §14 child as a
//! top-level guest over its own window; `compile_child` bakes only the *size* mask and the window
//! **base is a runtime arg** to `run_guarded`, so one compiled child runs at any carve offset. This
//! test pins that repeat spawns of the same `(module, entry, size)` — even at *different* offsets —
//! compile the child **once** (`temen_jit::child_compiles()` advances by 1, not 2), while each spawn
//! still runs correctly confined to its own carve. This attacks the F6 / gap-4 cost the design
//! flagged: without the cache, a shell spawning the same applet N times would run Cranelift N times.
//!
//! Sole test in its own binary so the process-wide compile counter is stable (cargo runs each test
//! binary in its own process; a lone test rules out interference from siblings in the same binary).

use temen_interp::Host;
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Parent (func 0) instantiates the child (func 1) in a 4 KiB carve at `off_a`, joins it, then does
/// the **same** at `off_b`, and returns the sum of the two results. The child stores `42` at its own
/// offset 0 (→ the carve base in the parent window) and returns `7`, so a correct confined run leaves
/// `42` at each carve base and the parent returns `14`.
const PARENT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  v1 = i64.const 1\n\
  v3 = i64.const 12\n\
  v4 = i64.const 0\n\
  ; spawn via record (op 17): entry=1 off=65536 sl=12 quota=0\n\
  q0v0 = i64.const 4294967296\n\
  q0v1 = i64.const 65536\n\
  q0v2 = i64.const -4294967284\n\
  q0v3 = i64.const 4294967295\n\
  q0v4 = i64.const 0\n\
  q0a0 = i64.const 17536\n\
  i64.store q0a0 q0v0\n\
  q0a1 = i64.const 17544\n\
  i64.store q0a1 q0v1\n\
  q0a2 = i64.const 17552\n\
  i64.store q0a2 q0v2\n\
  q0a3 = i64.const 17560\n\
  i64.store q0a3 q0v3\n\
  q0a4 = i64.const 17568\n\
  i64.store q0a4 q0v4\n\
  q0a5 = i64.const 17576\n\
  i64.store q0a5 q0v4\n\
  q0a6 = i64.const 17584\n\
  i64.store q0a6 q0v4\n\
  v5 = call.cap 6 17 (i64) -> (i32) v0 (q0a0)\n\
  v6 = call.cap 6 1 (i32) -> (i64) v0 (v5)\n\
  ; spawn via record (op 17): entry=1 off=69632 sl=12 quota=0\n\
  q1v0 = i64.const 4294967296\n\
  q1v1 = i64.const 69632\n\
  q1v2 = i64.const -4294967284\n\
  q1v3 = i64.const 4294967295\n\
  q1v4 = i64.const 0\n\
  q1a0 = i64.const 17600\n\
  i64.store q1a0 q1v0\n\
  q1a1 = i64.const 17608\n\
  i64.store q1a1 q1v1\n\
  q1a2 = i64.const 17616\n\
  i64.store q1a2 q1v2\n\
  q1a3 = i64.const 17624\n\
  i64.store q1a3 q1v3\n\
  q1a4 = i64.const 17632\n\
  i64.store q1a4 q1v4\n\
  q1a5 = i64.const 17640\n\
  i64.store q1a5 q1v4\n\
  q1a6 = i64.const 17648\n\
  i64.store q1a6 q1v4\n\
  v8 = call.cap 6 17 (i64) -> (i32) v0 (q1a0)\n\
  v9 = call.cap 6 1 (i32) -> (i64) v0 (v8)\n\
  v10 = i64.add v6 v9\n\
  return v10\n\
  }\n\
}\n\
func (i64) -> (i64) {\n\
block 0 (v0: i64) {\n\
  v1 = i64.const 0\n\
  v2 = i32.const 42\n\
  i32.store8 v1 v2\n\
  v3 = i64.const 7\n\
  return v3\n\
  }\n\
}\n";

#[test]
fn same_child_at_two_offsets_compiles_once() {
    let m = parse_module(PARENT).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << 17;
    let init = vec![0u8; win as usize];

    let mut host = Host::new();
    let h = host.grant_instantiator(0, win);

    let before = temen_jit::child_compiles();
    let (jo, jmem) = compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[h as i64],
        &init,
        0,
        temen_run::cap_thunk,
        &mut host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit run");
    let compiled = temen_jit::child_compiles() - before;

    // Both children ran and returned 7 → sum 14.
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[14]),
        "expected both children to run (sum 14), got {jo:?}"
    );
    // Each ran confined to its own carve: the `42` marker landed at each carve base.
    assert_eq!(jmem[65536], 42, "child A store missing at carve A base");
    assert_eq!(jmem[69632], 42, "child B store missing at carve B base");
    // Confinement: nothing outside the two carves' first byte was touched (both carves start zeroed;
    // the child writes only offset 0). Spot-check the byte just past carve A's marker and a byte in
    // between stays zero.
    assert_eq!(jmem[65537], 0, "child A escaped past its marker");
    assert_eq!(jmem[0], 0, "a child escaped into the parent's low window");

    // The load-bearing assertion: the child compiled **once**, then the second spawn (a different
    // offset) reused the cached code — the base is a runtime arg, so position-independent reuse.
    assert_eq!(
        compiled, 1,
        "same (module, entry, size) spawned twice must JIT-compile once, saw {compiled}"
    );
}
