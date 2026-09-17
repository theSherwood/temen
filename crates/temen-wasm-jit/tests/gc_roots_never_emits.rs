//! **#1546 §2 — a `gc.roots`-bearing module gets no emitted wasm.**
//!
//! `gc.roots` enumerates the live candidate words of the *interpreter's* frames and parked fibers. On
//! this tier an emitted frame's locals and operand stack are not addressable by the guest and not
//! introspectable by the host, so nothing can see them — a root held only in an emitted local is
//! **missed**, and a non-moving collector then frees a live object. Silent heap corruption, not a
//! decline.
//!
//! BROWSER.md sets the posture ("`gc.roots` bails unconditionally on this tier", module-granular
//! fallback), but what shipped was function granularity with no module veto: `temen-wasm-jit` had no
//! `GcRoots` arm anywhere, so a `gc.roots`-bearing function was merely "out of subset" — i.e. a
//! **cross-tier callee**, exactly the reachable-from-emitted position that is unsound. Two routes
//! reached it:
//!
//! 1. the **#888 widened** cross set (B2 / shared reserved table) admits *any* `marshallable_sig`
//!    non-in-subset function, so an emitted caller could bounce straight into a collecting helper;
//! 2. `interp_leaf` — the strict set — pattern-matches memory ops and calls and had **no `GcRoots`
//!    arm**, so a bare `gc.roots` wrapper (no load, no store, no call) passed it and became a
//!    cross-tier leaf even under a local table.
//!
//! Excluding it from the cross sets is *not* sufficient on the B2 path: the host writes a bounce shim
//! into **every** program slot of the shared table (`syncTable` in `wasmjit-module.js` —
//! `emitted['f'+slot] ?? shimFor(slot)`), so an emitted `call.dyn` reaches any non-emitted function
//! whatever the leaf set says. Module granularity is the sound answer, and it is what the docs
//! already claimed. These pins fail on the pre-#1546 emitter, where `emitted` carries a `true`.

use temen_wasm_jit::{
    compile_jit, compile_jit_paged, compile_module_reactor, compile_module_reactor_keep,
    compile_module_reactor_paged, compile_module_tierup, compile_module_tierup_b2, compile_nested,
    DriveMode, Shape,
};

/// A guest shaped like the hazard, on the **widened** route: `f0` is pure integer compute (in-subset,
/// the natural emit candidate) and calls `f1`, a realistic collector — it writes its result back to
/// the heap, so the strict `interp_leaf` set already excludes it and only the #888 widened cross set
/// admits it. `f1` is out of subset only because of `gc.roots`, which is exactly what made it a
/// cross-tier callee rather than a decline.
const COLLECTING_GUEST: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vone = i64.const 1
  vx = i64.add v0 vone
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vlo = i64.const 16384
  vhi = i64.const 32768
  vmask = i64.const -1
  vbuf = i64.const 20480
  vcap = i64.const 64
  vn = gc.roots vlo vhi vmask vbuf vcap
  vslot = i64.const 24576
  i64.store vslot vn
  return vn
  }
}
"#;

/// The §2.2 shape the strict `interp_leaf` set missed: a **bare** `gc.roots` wrapper — no load, no
/// store, no call, marshallable signature. It passed `interp_leaf` and became a cross-tier leaf even
/// under a local table, so function granularity was unsound on both tier-up modes, not just B2.
const BARE_WRAPPER_GUEST: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vone = i64.const 1
  vx = i64.add v0 vone
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vlo = i64.const 16384
  vhi = i64.const 32768
  vmask = i64.const -1
  vbuf = i64.const 20480
  vcap = i64.const 64
  vn = gc.roots vlo vhi vmask vbuf vcap
  return vn
  }
}
"#;

/// A control: the same shape with the collecting helper replaced by pure arithmetic. It must keep
/// emitting, so the veto is proven to be about `gc.roots` and not about this module shape.
const PLAIN_GUEST: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vone = i64.const 1
  vx = i64.add v0 vone
  vr = call 1 (vx)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vtwo = i64.const 2
  vr = i64.mul v0 vtwo
  return vr
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Every wasm front door, for one guest: `(label, emitted bitmap)`.
fn emitted_everywhere(m: &temen_ir::Module) -> Vec<(&'static str, Vec<bool>)> {
    let mut out = Vec::new();
    out.push((
        "tierup (local table)",
        compile_module_tierup(m, false).expect("tierup emits").1,
    ));
    out.push((
        "tierup b2 (#888 widened)",
        compile_module_tierup_b2(m, false, 8).expect("b2 emits").1,
    ));
    let a = compile_jit(m, Shape::Batch { entry: 0 }, false).expect("compile_jit");
    out.push(("compile_jit batch", a.emitted));
    let a = compile_jit(m, Shape::Threaded, false).expect("compile_jit threaded");
    out.push(("compile_jit threaded", a.emitted));
    let a = compile_jit_paged(m, Shape::Batch { entry: 0 }, false, 16).expect("compile_jit_paged");
    out.push(("compile_jit_paged batch", a.emitted));
    let a = compile_nested(m, false).expect("compile_nested");
    out.push(("compile_nested", a.emitted));
    out
}

#[test]
fn a_collecting_guest_emits_nothing_on_any_front_door() {
    let m = module(COLLECTING_GUEST);
    for (label, emitted) in emitted_everywhere(&m) {
        assert!(
            emitted.iter().all(|&e| !e),
            "{label}: a gc.roots-bearing module must emit nothing — an emitted frame live across a \
             bounce into the collector loses every root it holds. Got {emitted:?}"
        );
    }
}

/// The `interp_leaf` half: the bare wrapper must be vetoed too, on the local-table path where the
/// widened cross set is not even in play.
#[test]
fn a_bare_gc_roots_wrapper_is_vetoed_too() {
    let m = module(BARE_WRAPPER_GUEST);
    for (label, emitted) in emitted_everywhere(&m) {
        assert!(
            emitted.iter().all(|&e| !e),
            "{label}: a bare gc.roots wrapper is a cross-tier leaf by signature — it must not be \
             reachable from emitted code. Got {emitted:?}"
        );
    }
}

/// No collateral damage: the same module shape without `gc.roots` still tiers up, and a rooted
/// front door still reports `WasmDriven`.
#[test]
fn a_guest_without_gc_roots_still_emits() {
    let m = module(PLAIN_GUEST);
    for (label, emitted) in emitted_everywhere(&m) {
        assert!(
            emitted.iter().any(|&e| e),
            "{label}: this guest is pure integer compute and must still emit"
        );
    }
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("compile_jit");
    assert!(
        matches!(a.drive, DriveMode::WasmDriven { entry: 0 }),
        "a rooted, suspension-free guest is wasm-driven: {:?}",
        a.drive
    );
}

/// The direct reactor entries are not shape-deriving front doors — a caller asks for a rooted emit
/// explicitly — so they refuse rather than decline, and the property ("no wasm emit of a collecting
/// module, from any entry in this crate") holds without the caller having to know about it.
#[test]
fn the_reactor_entries_refuse_a_collecting_guest() {
    let m = module(COLLECTING_GUEST);
    let keep = vec![true; m.funcs.len()];
    assert!(compile_module_reactor(&m, 0, false).is_err(), "reactor");
    assert!(
        compile_module_reactor_paged(&m, 0, false, 16).is_err(),
        "reactor paged"
    );
    assert!(
        compile_module_reactor_keep(&m, 0, &keep, false).is_err(),
        "reactor keep — the one entry with its own emit computation"
    );
    // The control still emits through the same entries.
    let plain = module(PLAIN_GUEST);
    assert!(compile_module_reactor(&plain, 0, false).is_ok(), "control");
}

/// And the collecting guest stays *runnable* — the veto is a decline to the interpreter, never an
/// `Err`: every front door still returns a valid artifact, just with nothing emitted.
#[test]
fn the_veto_declines_rather_than_failing_the_compile() {
    let m = module(COLLECTING_GUEST);
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("still compiles");
    assert!(
        matches!(a.drive, DriveMode::InterpDriven),
        "a collecting guest runs wholly on the interpreter, which alone can see its roots: {:?}",
        a.drive
    );
    assert!(
        !a.wasm.is_empty(),
        "still a valid (import-only) wasm module"
    );
}
