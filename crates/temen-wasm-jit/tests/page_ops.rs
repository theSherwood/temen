//! **Window-remapping gating** (DESIGN.md §14 "wasm-JIT tier coverage"). The wasm-JIT tier confines
//! memory by a mask plus one live bound (the `"mapped"` global) — it cannot honor per-page state a
//! shrinking/aliasing op establishes (`unmap`/`protect`, ADDRESS_SPACE iface 5 ops 1/2; SharedRegion
//! `map`/`unmap`, iface 4 ops 0/1), so an emitted access would sail through a page the interpreter
//! (the oracle, which enforces `Mem`'s page-protection map) would trap on. A module that reaches one
//! is therefore compiled to **nothing** and run wholly on the interpreter — correct by construction.
//!
//! `ADDRESS_SPACE.map` (5,0) — a guest *grow* — no longer gates the module (#717): growth only adds
//! committed pages and rides the live `"mapped"` global the driver syncs per emitted call
//! (`tierup_grow_window.rs` is the differential proof). The map-containing function itself is still
//! never emitted; only its pure siblings are.
//!
//! These tests pin the routing: a gated-op module yields an `InterpDriven` artifact with an
//! all-`false` `emitted` bitmap (so even a memory-accessing function in it is *not* emitted), while
//! an otherwise identical module without the op still emits — the gate is specific, not a blanket
//! disable — and a map(5,0)-only module keeps its pure leaves eligible.

use temen_wasm_jit::{compile_jit, compile_module_tierup, compile_nested, DriveMode, Shape};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store};

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

// func 0 stores to the window (a would-be-emitted access) and then `protect`s a page (ADDRESS_SPACE
// op 2) through its handle — so the module reaches a page op.
const PROTECT_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.wrap_i64 v0
  va = i64.const 8
  i64.store va v0
  voff = i64.const 0
  vlen = i64.const 4096
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
"#;

// The same shape without the page op: a pure store+load. This one must still emit — proving the gate
// keys on the page op, not on memory access in general.
const PLAIN_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 8
  i64.store va v0
  vr = i64.load va
  return vr
  }
}
"#;

#[test]
fn compile_jit_page_op_module_emits_nothing() {
    let m = parse(PROTECT_UNIT);
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    // The store-bearing func 0 is NOT emitted — the whole guest runs on the interpreter.
    assert_eq!(a.emitted, vec![false]);
}

// A §13 SharedRegion `map` (iface 4 op 0) aliases host backing into window pages — the same hazard
// class as ADDRESS_SPACE, so it must gate too.
const REGION_MAP_UNIT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vh = i32.wrap_i64 v0
  vwo = i64.const 0
  vro = i64.const 0
  vlen = i64.const 4096
  vprot = i32.const 3
  vr = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vh (vwo, vro, vlen, vprot)
  return vr
  }
}
"#;

#[test]
fn compile_jit_shared_region_map_emits_nothing() {
    let m = parse(REGION_MAP_UNIT);
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    assert_eq!(a.emitted, vec![false]);
}

#[test]
fn compile_jit_plain_module_still_emits() {
    // Control: identical but for the page op → the store/load func emits and is wasm-driven.
    let m = parse(PLAIN_UNIT);
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("artifact");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    assert_eq!(a.emitted, vec![true]);
}

#[test]
fn compile_nested_page_op_module_emits_nothing() {
    let m = parse(PROTECT_UNIT);
    let a = compile_nested(&m, false).expect("artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    assert_eq!(a.emitted, vec![false]);
}

// A page-op orchestrator (func 0: `protect`, out of subset — the interpreter drives it) plus a
// **memory-accessing** compute leaf (func 1: a window load) it calls. This is the JACL tier-up shape
// (`TEMEN_BROWSER_TIERUP_FINDINGS.md`): mainline InterpDriven code with a hot leaf, in a guest that
// manages its own pages. The leaf's emitted load is mask-only, so tiering it up would ignore the page
// state func 0 established — an emitted access over a page-managed window. The gate must therefore
// emit **nothing** here too, even though the leaf is in-subset in isolation.
const PAGE_OP_WITH_LEAF: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.wrap_i64 v0
  voff = i64.const 0
  vlen = i64.const 4096
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  v1 = call 1 (vr)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 8
  vld = i64.load va
  v2 = i64.add vld v0
  return v2
  }
}
"#;

// The same two functions WITHOUT the page op (func 0 just calls the leaf) — the control: here the
// leaf must still tier up, proving the gate keys on the page op, not on having a memory leaf.
const PLAIN_WITH_LEAF: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 8
  vld = i64.load va
  v2 = i64.add vld v0
  return v2
  }
}
"#;

#[test]
fn tierup_page_op_module_emits_nothing() {
    // The regression pin for `TEMEN_BROWSER_TIERUP_FINDINGS.md`: calling the public `compile_module_tierup`
    // entry directly (as the JACL spike did, bypassing `compile_jit`'s gate) must still emit nothing on
    // a page-op module — the leaf is NOT tiered up. Before the fix this returned `[false, true]`.
    let m = parse(PAGE_OP_WITH_LEAF);
    let (_wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(
        eligible,
        vec![false, false],
        "a page-op module must emit no tier-up functions (the leaf stays on the interpreter)"
    );
}

// The grow split (#717): the same orchestrator+leaf shape but with `map` (op 0) instead of
// `protect` — a guest growing its heap. The map-containing func 0 stays on the interpreter (a
// remapping call.cap is not in-subset), but the pure leaf must now be eligible: growth is carried
// to the emitted bounds check by the live `"mapped"` global, synced per call by the driver.
const MAP_WITH_LEAF: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vas = i32.wrap_i64 v0
  voff = i64.const 0
  vlen = i64.const 4096
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  v1 = call 1 (vr)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 8
  vld = i64.load va
  v2 = i64.add vld v0
  return v2
  }
}
"#;

#[test]
fn tierup_map_module_keeps_leaf_eligible() {
    // #717 gate split: a pure-grow (`map`, 5/0) module is NOT interp-only — the mapping func 0
    // stays on the interpreter, the pure memory leaf tiers up (its emitted access confines against
    // the live `"mapped"` global the driver syncs).
    let m = parse(MAP_WITH_LEAF);
    let (_wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(
        eligible,
        vec![false, true],
        "a map-only module keeps its pure leaf tier-up eligible; the mapping func stays interpreted"
    );
}

#[test]
fn tierup_plain_module_still_emits_leaf() {
    // Control: identical minus the page op → the memory leaf (func 1) tiers up as before.
    let m = parse(PLAIN_WITH_LEAF);
    let (_wasm, eligible) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(
        eligible,
        vec![true, true],
        "without a page op the leaf (and its pure caller) must still tier up"
    );
}

#[test]
fn tierup_matches_compile_jit_on_page_op_module() {
    // The two public entries must agree on a page-op module: `compile_jit(Threaded)` already gates
    // (routes through `compile_module_tierup`), and the direct entry now gates too.
    let m = parse(PAGE_OP_WITH_LEAF);
    let via_jit = compile_jit(&m, Shape::Threaded, false).expect("artifact");
    let (_wasm, via_tierup) = compile_module_tierup(&m, false).expect("tier-up emit");
    assert_eq!(
        via_jit.emitted, via_tierup,
        "gate must agree across entries"
    );
    assert!(via_tierup.iter().all(|&e| !e), "both emit nothing");
}

/// The emit-nothing artifact must still be a **valid** wasm module (imports only, no `f{i}` exports),
/// so a host can instantiate it even though it never calls in. Instantiate the `compile_jit`
/// interp-only output under wasmi with its three imports stubbed.
#[test]
fn interp_only_wasm_validates_and_exports_no_functions() {
    let m = parse(PROTECT_UNIT);
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("artifact");

    let engine = Engine::default();
    let module = WModule::new(&engine, &a.wasm).expect("interp-only wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, i32>, _c: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {},
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    assert!(
        instance.get_func(&store, "f0").is_none(),
        "interp-only module must export no emitted function"
    );
}
