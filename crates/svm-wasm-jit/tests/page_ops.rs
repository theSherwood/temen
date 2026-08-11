//! **Window-remapping gating** (DESIGN.md §14 "wasm-JIT tier coverage"). The wasm-JIT tier confines memory by
//! masking to `[0, size)` — it cannot honor per-page state (`map`/`unmap`/`protect`, ADDRESS_SPACE
//! iface 5 ops 0/1/2), so an emitted access would sail through a page the interpreter (the oracle,
//! which enforces `Mem`'s page-protection map) would trap on. A module that reaches a page op is
//! therefore compiled to **nothing** and run wholly on the interpreter — correct by construction.
//!
//! These tests pin the routing: a page-op module yields an `InterpDriven` artifact with an all-`false`
//! `emitted` bitmap (so even a memory-accessing function in it is *not* emitted), while an otherwise
//! identical module without the page op still emits — the gate is specific, not a blanket disable.

use svm_wasm_jit::{compile_jit, compile_nested, DriveMode, Shape};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store};

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
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
  vr = cap.call 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
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
  vr = cap.call 4 0 (i64, i64, i64, i32) -> (i64) vh (vwo, vro, vlen, vprot)
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
