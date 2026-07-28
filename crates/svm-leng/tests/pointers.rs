//! Pointer tests (NIM.md Phase 2 broadening): pointer-typed params, `deref` (load) and store
//! through a pointer, over a window. No data-stack frame yet — the pointer is supplied by the
//! caller as a window offset — so these exercise the memory machinery (memory decl, loads, stores)
//! without the address-of-local ABI (the next slice). Run on both engines.

use svm_interp::Value;

/// Run func `idx` with i64 args on interp + JIT (both must agree); return the i64 result.
fn run(module: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match svm_jit::compile_and_run(module, idx, args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

#[test]
fn store_then_load_through_pointer() {
    // roundtrip(p: ptr int, x: int): int = { *p = x; return *p }  — p is a window offset.
    let leng = "\
(stmts
 (proc :roundtrip.0 (params (param :p.0 . (ptr (i +64))) (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (deref p.0) x.0)
   (ret (deref p.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // The IR must declare a window for the load/store.
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.starts_with("memory "),
        "module must declare memory:\n{text}"
    );
    // Pass p = a valid 8-aligned window offset; x is the value to round-trip.
    assert_eq!(run(&m, 0, &[64, 42]), 42);
    assert_eq!(run(&m, 0, &[128, -7]), -7);
}

#[test]
fn store_stmt_reversed_operands() {
    // Leng StoreStmt `(store Value Lvalue)` = asgn with reversed operands: *p = v.
    // addto(p, d): int = { *p = *p + d; return *p }
    let leng = "\
(stmts
 (proc :addto.0 (params (param :p.0 . (ptr (i +64))) (param :d.0 . (i +64))) (i +64) .
  (stmts .
   (store (add (i +64) (deref p.0) d.0) (deref p.0))
   (ret (deref p.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // The window starts zeroed, so *p == 0 initially; addto(p, 5) → 5, and no `d` doubling.
    assert_eq!(run(&m, 0, &[64, 5]), 5);
    assert_eq!(run(&m, 0, &[256, 100]), 100);
}

#[test]
fn no_memory_decl_without_pointers() {
    // A pure-integer proc must NOT declare a window (memory only appears when actually used).
    let leng = "(stmts (proc :id.0 (params (param :x.0 . (i +64))) (i +64) . (stmts . (ret x.0))))";
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(!text.contains("memory "), "no memory expected:\n{text}");
}
