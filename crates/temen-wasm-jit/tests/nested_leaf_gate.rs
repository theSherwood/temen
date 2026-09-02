//! **#1151 — `Artifact::leaves_pure`, the throwaway-servicer gate.** The nested emit admits any
//! marshallable-signature out-of-subset function as an `env.call_interp` leaf, under the contract that
//! the servicer carries the powerbox over the **live window** (the op-13 grant path relies on a leaf
//! that stores and `call.cap`s). The browser's §14 codegen Worker instead services leaves with
//! `temen_wasmjit_call_interp` — a throwaway window and an empty powerbox — which is faithful only
//! for **pure** leaves: a `map`-calling allocator helper `CapFault`ed / wrote a fresh window there
//! while the interpreter succeeded (a trap-parity divergence, INVARIANTS #9). `leaves_pure` is the
//! predicate such a host gates on; `temen_par_enable_inst_codegen` declines codegen when it is false.

use temen_wasm_jit::{
    compile_module_nested, compile_nested, compile_nested_paged, outline_nested_cap_calls,
    DriveMode,
};

const PAGE_LOG2: u8 = 12;

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Entry `(address_space) -> i64` calls a helper that **grows** (`map`, op 0) and stores into the
/// fresh page — the allocator shape of a malloc child. The helper is out of the nested subset (a
/// non-lowerable `call.cap`) and impure.
const MAP_HELPER: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 4096
  vr = call 1 (v0, va)
  return vr
  }
}
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  vlen = i64.const 16384
  vm = call.cap 5 0 (i64, i64) -> (i64) v0 (v1, vlen)
  vk = i64.const 7
  i64.store v1 vk
  return vm
  }
}
"#;

/// A cap-free but memory-touching out-of-subset helper: `f64.fma` (no core-wasm op) plus a store.
const STORE_HELPER: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = call 1 (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = f64.reinterpret_i64 v0
  vg = f64.fma vf vf vf
  vi = i64.reinterpret_f64 vg
  va = i64.const 64
  i64.store va vi
  return vi
  }
}
"#;

/// A **pure** out-of-subset helper (`f64.fma` only) — the leaf shape a throwaway-window servicer
/// computes exactly like the interpreter.
const PURE_HELPER: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = call 1 (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = f64.reinterpret_i64 v0
  vg = f64.fma vf vf vf
  vi = i64.reinterpret_f64 vg
  return vi
  }
}
"#;

/// Entry `(address_space) -> i64` that queries `page_size` — after the outline the `call.cap` lives
/// in an appended wrapper: an impure leaf that needs the powerbox.
const PAGE_SIZE_ENTRY: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vps = call.cap 5 3 () -> (i64) v0 ()
  return vps
  }
}
"#;

#[test]
fn map_helper_is_an_impure_leaf() {
    let m = parse(MAP_HELPER);
    // The nested emit admits it (live-window servicer contract) — the entry emits, the helper is a
    // leaf — but a throwaway-window host must not drive it.
    let a = compile_nested(&m, true).expect("front door");
    assert!(matches!(a.drive, DriveMode::WasmDriven { entry: 0 }));
    assert_eq!(a.emitted, vec![true, false]);
    assert!(!a.leaves_pure(&m), "a map+store helper is not a pure leaf");
    // Same on the paged front door (outlining touches only `unmap`/`protect`/`page_size`/`sub`).
    let mut mo = m.clone();
    outline_nested_cap_calls(&mut mo);
    assert_eq!(mo.funcs.len(), 2, "map is not outlined");
    let p = compile_nested_paged(&mo, true, PAGE_LOG2).expect("paged front door");
    assert!(matches!(p.drive, DriveMode::WasmDriven { entry: 0 }));
    assert!(!p.leaves_pure(&mo));
}

#[test]
fn store_helper_is_an_impure_leaf() {
    let m = parse(STORE_HELPER);
    let a = compile_nested(&m, true).expect("front door");
    assert!(matches!(a.drive, DriveMode::WasmDriven { entry: 0 }));
    assert_eq!(a.emitted, vec![true, false]);
    assert!(
        !a.leaves_pure(&m),
        "a memory-touching leaf needs the live window"
    );
}

#[test]
fn pure_helper_is_a_pure_leaf() {
    let m = parse(PURE_HELPER);
    let a = compile_nested(&m, true).expect("front door");
    assert!(matches!(a.drive, DriveMode::WasmDriven { entry: 0 }));
    assert_eq!(
        a.emitted,
        vec![true, false],
        "entry emitted, fma helper a leaf"
    );
    assert!(
        a.leaves_pure(&m),
        "an fma-only leaf is faithful on a throwaway window"
    );
}

#[test]
fn outlined_wrapper_is_an_impure_leaf_and_interp_driven_is_pure() {
    let mut m = parse(PAGE_SIZE_ENTRY);
    outline_nested_cap_calls(&mut m);
    assert_eq!(m.funcs.len(), 2, "one wrapper appended");
    let a = compile_nested(&m, true).expect("front door");
    assert!(matches!(a.drive, DriveMode::WasmDriven { entry: 0 }));
    assert_eq!(a.emitted, vec![true, false]);
    assert!(
        !a.leaves_pure(&m),
        "the `call.cap` wrapper needs the powerbox"
    );
    // Un-outlined, the `call.cap` sits in the entry itself: out of subset, the unit falls to the
    // interp-driven mode — whose tier-up fixpoint bounces only pure leaves, so it is always pure.
    let raw = parse(PAGE_SIZE_ENTRY);
    assert!(compile_module_nested(&raw, true).is_err());
    let b = compile_nested(&raw, true).expect("front door");
    assert!(matches!(b.drive, DriveMode::InterpDriven));
    assert!(b.leaves_pure(&raw));
}
