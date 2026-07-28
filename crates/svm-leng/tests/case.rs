//! `case` → `br_table` tests (NIM.md Phase 2): a dense-integer `case` lowers to a normalized
//! `br_table` (out-of-range → default). Hand-written fixtures (single values, a multi-value `of`,
//! a `range`, and the `else`) run on both engines; the real nimony `classify` case verifies + runs.

use svm_interp::Value;

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
fn dense_case_with_multivalue_and_else() {
    // classify(n) = case n: 0→100, 1|2→200, 3→300, else→999.
    let leng = "\
(stmts
 (proc :classify.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (case n.0
    (of (ranges 0) (stmts . (asgn r.0 100)))
    (of (ranges 1 2) (stmts . (asgn r.0 200)))
    (of (ranges 3) (stmts . (asgn r.0 300)))
    (else (stmts . (asgn r.0 999))))
   (ret r.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("br_table"),
        "case should lower to br_table:\n{text}"
    );
    assert_eq!(run(&m, 0, &[0]), 100);
    assert_eq!(run(&m, 0, &[1]), 200);
    assert_eq!(run(&m, 0, &[2]), 200);
    assert_eq!(run(&m, 0, &[3]), 300);
    assert_eq!(run(&m, 0, &[7]), 999); // out of range → default (else)
    assert_eq!(run(&m, 0, &[-5]), 999); // negative → default
}

#[test]
fn case_with_range() {
    // grade(n): 0..59 → 0 (F), 60..100 → 1 (pass), else → -1.
    let leng = "\
(stmts
 (proc :grade.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (case n.0
    (of (ranges (range 0 59)) (stmts . (asgn r.0 0)))
    (of (ranges (range 60 100)) (stmts . (asgn r.0 1)))
    (else (stmts . (asgn r.0 (neg (i +64) 1)))))
   (ret r.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[0]), 0);
    assert_eq!(run(&m, 0, &[59]), 0);
    assert_eq!(run(&m, 0, &[60]), 1);
    assert_eq!(run(&m, 0, &[100]), 1);
    assert_eq!(run(&m, 0, &[101]), -1);
}

#[test]
fn sparse_case_is_fail_closed() {
    // A huge span is out of br_table range → clean Unsupported, never a giant table.
    let leng = "\
(stmts
 (proc :f.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (case n.0
    (of (ranges 0) (stmts . (asgn r.0 1)))
    (of (ranges 1000000) (stmts . (asgn r.0 2)))
    (else (stmts . (asgn r.0 0))))
   (ret r.0))))";
    match svm_leng::translate(leng) {
        Err(svm_leng::LengError::Unsupported(_)) => {}
        other => panic!("expected Unsupported for sparse case, got {other:?}"),
    }
}

/// Real nimony `hexer` output for `classify(n): case n of 0/1,2/3/else`. Translate it out of the
/// module and run — the real `(case … (of (ranges …) …) (else …))` bytes.
#[test]
fn real_nimony_case() {
    const REAL: &str = include_str!("fixtures/real_case.leng.nif");
    let m = svm_leng::translate_proc(REAL, "classify.0.")
        .unwrap_or_else(|e| panic!("translate real classify: {e}"));
    assert_eq!(run(&m, 0, &[0]), 100);
    assert_eq!(run(&m, 0, &[1]), 200);
    assert_eq!(run(&m, 0, &[2]), 200);
    assert_eq!(run(&m, 0, &[3]), 300);
    assert_eq!(run(&m, 0, &[42]), 999);
}
