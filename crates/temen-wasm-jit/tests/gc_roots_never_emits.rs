//! **#1546 §2, the residual route — a `gc.roots`-bearing module emits no wasm at all.**
//!
//! `gc.roots` enumerates the live candidate words of the *interpreter's* frames and parked fibers.
//! An emitted frame's locals and operand stack are addressable by neither the guest nor the host, so
//! a root held only there is **missed** and a non-moving collector frees a live object — an
//! *under*-approximation, which GC.md §3.2's per-backend licence does not cover.
//!
//! Excluding the op from both cross-tier sets (`interp_leaf`, and `bounce_serviceable`'s fourth
//! seed) closes the **direct** route: the emit fixpoint cascades a direct caller off, which
//! `tierup::gc_roots_callee_is_not_a_cross_tier_leaf_on_either_table` and
//! `analysis::gc_roots_callee_is_not_an_interp_leaf` pin.
//!
//! It does not close the **indirect** route, and that is what these pins are for. In B2 mode an
//! in-subset function that makes a `call.dyn` is emitted regardless of any cross set, and the host
//! fills *every* program slot of the shared table with a bounce shim built from that slot's
//! signature alone (`temen_coop_shim_wasm` reads `s.sigs[slot]` and emits a trampoline; it never
//! consults the leaf set). So the emitted `call.dyn` still lands on the collector, with the caller's
//! frame live. Measured on the pre-veto source: the dispatcher below emits as `[true, false]`.
//!
//! Hence module granularity — the posture BROWSER.md documented all along. `module_uses_gc_roots`
//! vetoes the emit at every entry in this crate, so no emitted frame of such a module can exist for
//! the op to fail to scan, by any route.

use temen_wasm_jit::{
    compile_jit, compile_jit_paged, compile_module_reactor, compile_module_reactor_keep,
    compile_module_reactor_paged, compile_module_tierup, compile_module_tierup_b2, compile_nested,
    DriveMode, Shape,
};

/// **The residual route.** `f0` is in-subset and makes a `call.dyn`, so B2 emits it whatever the
/// cross sets say; `f1` carries `gc.roots` and therefore owns a table slot the host fills with a
/// bounce shim. Nothing on the direct-call axis connects them — the hazard is entirely in the
/// dispatch table. Pre-veto this emits `[true, false]`.
const INDIRECT_DISPATCH_GUEST: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vi = i32.wrap_i64 v0
  vs = i64.const 7
  vr = call.dyn (i64) -> (i64) vi (vs)
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

/// The direct shape, kept here only to prove the **front doors** decline it too: `compile_jit`,
/// `compile_jit_paged`, `compile_nested` and the `compile_module_reactor*` family are outside what
/// the cross-set fix reaches, since they compute their own emit sets.
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

/// A control: the same dispatch shape with the collector replaced by arithmetic. It must keep
/// emitting, so the veto is proven specific to `gc.roots` rather than to this module shape.
const PLAIN_GUEST: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vi = i32.wrap_i64 v0
  vs = i64.const 7
  vr = call.dyn (i64) -> (i64) vi (vs)
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

/// The residual route, pinned: an emitted `call.dyn` can reach the collector through a host-written
/// shim no cross set gates, so the dispatcher must not be emitted either.
#[test]
fn an_indirect_dispatcher_beside_a_collector_emits_nothing() {
    let m = module(INDIRECT_DISPATCH_GUEST);
    for (label, emitted) in emitted_everywhere(&m) {
        assert!(
            emitted.iter().all(|&e| !e),
            "{label}: the dispatcher is in-subset and its `call.dyn` reaches the collector through \
             the table, so emitting it leaves a live wasm frame the scan cannot see. Got {emitted:?}"
        );
    }
}

/// The front doors compute their own emit sets, so each carries the veto rather than inheriting it.
#[test]
fn a_collecting_guest_emits_nothing_on_any_front_door() {
    let m = module(COLLECTING_GUEST);
    for (label, emitted) in emitted_everywhere(&m) {
        assert!(
            emitted.iter().all(|&e| !e),
            "{label}: a gc.roots-bearing module must emit nothing. Got {emitted:?}"
        );
    }
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
    let plain = module(PLAIN_GUEST);
    assert!(compile_module_reactor(&plain, 0, false).is_ok(), "control");
}

/// No collateral damage: the same dispatch shape without `gc.roots` still tiers up, and a rooted
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

/// And the collecting guest stays *runnable* — the veto is a decline to the interpreter, never an
/// `Err`: the shape-deriving front doors still return a valid artifact, just with nothing emitted.
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
