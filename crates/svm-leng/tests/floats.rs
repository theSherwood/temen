//! Float tests (NIM.md Phase 2): `f32`/`f64` types, arithmetic (`fN.add/sub/mul/div`), `neg`,
//! comparisons, int↔float `conv`, and float literals. Run on both engines with an f64-aware harness
//! (the JIT passes/returns floats as raw bits). Hand-written fixtures + real nimony float bytes.

use svm_interp::Value;

/// Run func `idx` returning an f64, with mixed i64/f64 args (given as `Value`s). Interp and JIT
/// (bits) must agree; returns the f64 result.
fn run_f64(module: &svm_ir::Module, idx: u32, args: &[Value]) -> f64 {
    svm_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(module, idx, args, &mut fuel).expect("interp run");
    let interp_f = match interp.as_slice() {
        [Value::F64(f)] => *f,
        other => panic!("expected f64, got {other:?}"),
    };
    // JIT args/results are raw i64 words: an f64 arg is its bit pattern; the result likewise.
    let jargs: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::F64(f) => f.to_bits() as i64,
            Value::I64(n) => *n,
            Value::I32(n) => *n as i64,
            other => panic!("unsupported arg {other:?}"),
        })
        .collect();
    let jit = match svm_jit::compile_and_run(module, idx, &jargs).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    let jit_f = f64::from_bits(jit[0] as u64);
    assert_eq!(interp_f.to_bits(), jit_f.to_bits(), "§9 interp/JIT parity");
    interp_f
}

#[test]
fn float_arithmetic_and_literals() {
    // favg(a, b) = (a + b) / 2.0
    let leng = "\
(stmts
 (proc :favg.0 (params (param :a.0 . (f +64)) (param :b.0 . (f +64))) (f +64) .
  (stmts .
   (ret (div (f +64) (add (f +64) a.0 b.0) 2.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("f64.add") && text.contains("f64.div"),
        "float ops:\n{text}"
    );
    assert_eq!(run_f64(&m, 0, &[Value::F64(3.0), Value::F64(5.0)]), 4.0);
    assert_eq!(run_f64(&m, 0, &[Value::F64(1.5), Value::F64(2.5)]), 2.0);
}

#[test]
fn int_to_float_conversion() {
    // toF(n) = float(n) * 1.5
    let leng = "\
(stmts
 (proc :toF.0 (params (param :n.0 . (i +64))) (f +64) .
  (stmts .
   (ret (mul (f +64) (conv (f +64) n.0) 1.5)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("f64.convert_i64_s"),
        "int->float conv:\n{text}"
    );
    assert_eq!(run_f64(&m, 0, &[Value::I64(10)]), 15.0);
    assert_eq!(run_f64(&m, 0, &[Value::I64(-4)]), -6.0);
}

#[test]
fn local_const_float_array_materializes_and_indexes() {
    // A **local `const`** float table — `parseutils.parseFloat`'s `powtens = [1e0, 1e1, …]`, an
    // `array[.., float64]` declared inside the proc. `collect_local_aggregate_consts` materializes it
    // into a data segment via `const_aggregate_bytes`; before the `aconstr` folder handled float
    // elements it returned `None`, so the const stayed unmaterialized and lowering errored
    // "non-scalar local const `powtens`". Now each element folds to its f64 bits; `(at pt idx)` reads
    // one back on both engines.
    let leng = "\
(stmts
 (type :FTbl.0. . (array (f +64) 4))
 (proc :pick.0. (params (param :i.0 . (i +64))) (f +64) .
  (stmts .
   (const :pt.0. . FTbl.0. (aconstr FTbl.0. 1.0 10.0 100.0 1000.0))
   (ret (at pt.0. i.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run_f64(&m, 0, &[Value::I64(0)]), 1.0, "pt[0]");
    assert_eq!(run_f64(&m, 0, &[Value::I64(2)]), 100.0, "pt[2]");
    assert_eq!(run_f64(&m, 0, &[Value::I64(3)]), 1000.0, "pt[3]");
}

#[test]
fn float_to_int_and_neg() {
    // trunc_neg(x): int = int(-x)   — f64.neg then i64.trunc_f64_s.
    let leng = "\
(stmts
 (proc :trunc_neg.0 (params (param :x.0 . (f +64))) (i +64) .
  (stmts .
   (ret (conv (i +64) (neg (f +64) x.0))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("f64.neg") && text.contains("i64.trunc_f64_s"),
        "neg+trunc:\n{text}"
    );
    // Returns i64 — run via the integer path.
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(&m, 0, &[Value::F64(3.7)], &mut fuel).unwrap();
    assert_eq!(
        interp.as_slice(),
        &[Value::I64(-3)],
        "int(-3.7) = -3 (trunc toward zero)"
    );
    let jit = match svm_jit::compile_and_run(&m, 0, &[3.7f64.to_bits() as i64]).unwrap() {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[-3], "§9 parity");
}

#[test]
fn float_comparison() {
    // fmax(a, b): float = if a < b: b else: a  — exercises f64.lt in a condition.
    let leng = "\
(stmts
 (proc :fmax.0 (params (param :a.0 . (f +64)) (param :b.0 . (f +64))) (f +64) .
  (stmts .
   (var :r.0 . (f +64) .)
   (if
    (elif (lt a.0 b.0) (stmts . (asgn r.0 b.0)))
    (else (stmts . (asgn r.0 a.0))))
   (ret r.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert!(svm_leng::translate_to_text(leng)
        .unwrap()
        .contains("f64.lt"));
    assert_eq!(run_f64(&m, 0, &[Value::F64(3.0), Value::F64(9.0)]), 9.0);
    assert_eq!(run_f64(&m, 0, &[Value::F64(9.0), Value::F64(3.0)]), 9.0);
}

/// Real nimony `hexer` output for `favg(a,b)=(a+b)/2.0` — genuine `(add (f 64) …)` / `(div (f 64)
/// …)` bytes with a `2.0` float literal.
#[test]
fn real_nimony_floats() {
    const REAL: &str = include_str!("fixtures/real_floats.leng.nif");
    let m = svm_leng::translate_proc(REAL, "favg.0.")
        .unwrap_or_else(|e| panic!("translate real favg: {e}"));
    assert_eq!(run_f64(&m, 0, &[Value::F64(3.0), Value::F64(5.0)]), 4.0);
    assert_eq!(run_f64(&m, 0, &[Value::F64(10.0), Value::F64(20.0)]), 15.0);
}
