//! The unified front door [`svm_wasm_jit::compile_jit`]: pins the **IR-driven strategy selection**
//! (which [`DriveMode`] and which functions emit) for each invocation [`Shape`]. Execution
//! correctness of the chosen strategy is already covered by `differential.rs` / `cross_tier.rs`
//! (wasm-driven) and `tierup.rs` (interp-driven); this suite covers only the new decision layer —
//! that the embedder no longer has to pick, and the pick is right.

use svm_wasm_jit::{compile_jit, DriveMode, Shape};

fn build(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// A rooted, suspension-free, fully-in-subset guest → **wasm-driven**, every function emitted.
#[test]
fn pure_compute_is_wasm_driven() {
    let m = build(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 2\n  v2 = i64.mul v0 v1\n  return v2\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("compile");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    assert_eq!(a.emitted, vec![true]);
    assert!(!a.wasm.is_empty());
}

/// A `Reactor` shape is wasm-driven on the same terms as `Batch` (the difference is host-side
/// re-entry, not codegen); the entry may be a non-zero `tick`.
#[test]
fn reactor_entry_is_wasm_driven_at_its_tick() {
    // func 0 = a helper, func 1 = the `tick` the host re-enters.
    let m = build(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n\
         func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 1\n  v2 = i64.add v0 v1\n  return v2\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Reactor { entry: 1 }, false).expect("compile");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 1 });
    // Rooted at func 1: func 1 emits; the unreachable func 0 does not.
    assert_eq!(a.emitted, vec![false, true]);
}

/// An in-subset caller of a **memory-free, integer-signature, non-subset leaf** (scalar `fma`, which
/// has no core-wasm opcode, behind an i64 signature) → still wasm-driven, with the leaf left as a
/// cross-tier interpreter call. (Note scalar floats *are* in the wasm subset — `f64.add` emits — so
/// the leaf must use an op wasm lacks, like `fma`, to actually stay interpreter-tier.)
#[test]
fn integer_caller_with_nonsubset_leaf_keeps_wasm_driver() {
    let m = build(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = call 1 (v0)\n  return v1\n  }\n}\n\
         func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = f64.reinterpret_i64 v0\n  v2 = f64.fma v1 v1 v1\n  v3 = i64.reinterpret_f64 v2\n  return v3\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("compile");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    // func 0 emits; the float leaf (func 1) is cross-tier (interpreter-serviced), not emitted.
    assert_eq!(a.emitted, vec![true, false]);
}

/// A guest whose reachable set **suspends** (`cont.*`) cannot be wasm-driven (a wasm frame can't
/// unwind across a stack switch) → the front door routes it to the interpreter driver, *not* the raw
/// `compile_module_reactor` path (which would emit a cross-tier callee it can't safely unwind).
#[test]
fn reachable_suspension_forces_interp_driver() {
    let m = build(
        "func () -> (i64) {\nblock 0 () {\n  v0 = ref.func 1\n  v1 = i64.const 0\n  v2 = cont.new v0 v1\n  v3 = i64.const 7\n  v4, v5 = cont.resume v2 v3\n  return v5\n  }\n}\n\
         func (i64, i64) -> (i64) {\nblock 0 (vsp: i64, varg: i64) {\n  return varg\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("compile");
    assert_eq!(a.drive, DriveMode::InterpDriven);
}

/// A guest whose **entry** is outside the emit subset for a non-concurrency reason (scalar `fma`, no
/// core-wasm opcode) can't be rooted for a wasm driver → falls back to the interpreter driver, which
/// simply emits nothing for that function.
#[test]
fn non_emittable_entry_falls_back_to_interp_driver() {
    let m = build(
        "func (f64) -> (f64) {\nblock 0 (v0: f64) {\n  v1 = f64.fma v0 v0 v0\n  return v1\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Batch { entry: 0 }, false).expect("compile");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    assert_eq!(a.emitted, vec![false]);
}

/// The `Threaded` shape is always interpreter-driven — there is no single top-level frame the host
/// can own — even for otherwise pure-compute IR; hot regions still tier up.
#[test]
fn threaded_shape_is_always_interp_driven() {
    let m = build(
        "func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  v1 = i64.const 3\n  v2 = i64.mul v0 v1\n  return v2\n  }\n}\n",
    );
    let a = compile_jit(&m, Shape::Threaded, true).expect("compile");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    // Pure compute still tiers up: the one function is emit-eligible.
    assert_eq!(a.emitted, vec![true]);
}
