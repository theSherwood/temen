//! **The emitted-function size valve** (#1004): wasm engines reject a single function body over a
//! hard limit (V8: ~7.65 MB), which fails `WebAssembly.compile` for the *whole* module. A rare huge
//! SVM function — the SQLite VDBE dispatcher in the shipped cards, whose bulk-memory body rejoined
//! the emit subset under #1004 — is kept on the interpreter (a cross-tier leaf), so the module still
//! emits WasmDriven with every other function on wasm. This pins the valve through the public
//! `compile_module_reactor` entry: an over-cap callee is excluded from the emitted set while the
//! module compiles, and an under-cap twin emits.

/// A module whose `_start` (func 0) calls func 1, a pure `(i64) -> (i64)` leaf padded with `n`
/// redundant loads so its estimated emitted body scales with `n` (each scalar load carries the fat
/// confine sequence). `n` large ⇒ over the valve's cap.
fn module_with_padded_callee(n: usize) -> svm_ir::Module {
    let mut loads = String::new();
    for i in 0..n {
        loads.push_str(&format!("  vp{i} = i64.load v0\n"));
    }
    let src = format!(
        r#"memory 17
func () -> (i64) {{
block 0 () {{
  v0 = i64.const 16384
  vr = call 1 (v0)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{loads}  return v0
  }}
}}
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// An over-cap callee is kept off the wasm tier (a cross-tier leaf), the entry still emits, and the
/// module compiles — the SQLite-dispatcher shape. An under-cap twin emits both functions.
#[test]
fn oversized_function_stays_interpreter_resident_and_the_module_still_emits() {
    // ~55k loads ⇒ estimate ≈ 55k × 128 B ≈ 7 MB, over the 6.5 MB cap.
    let big = module_with_padded_callee(55_000);
    let (wasm, emitted) = svm_wasm_jit::compile_module_reactor(&big, 0, false).expect(
        "module compiles — the over-cap callee falls to a cross-tier leaf, not a hard fail",
    );
    assert!(emitted[0], "the entry emits");
    assert!(
        !emitted[1],
        "the over-cap callee is excluded from the wasm tier (kept interpreter-resident)"
    );
    // It really is a compilable module (a valid wasm binary; wasmi validates on `Module::new`).
    wasmi::Module::new(&wasmi::Engine::default(), &wasm[..]).expect("emitted wasm validates");

    // Control: a small callee emits like any in-subset function.
    let small = module_with_padded_callee(100);
    let (_, emitted) = svm_wasm_jit::compile_module_reactor(&small, 0, false).expect("emits");
    assert!(
        emitted[0] && emitted[1],
        "an under-cap callee emits on the wasm tier"
    );
}
