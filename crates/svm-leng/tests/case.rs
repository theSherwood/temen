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
fn sparse_case_lowers_to_comparison_chain() {
    // A span too large/sparse for a br_table (0 vs 1_000_000) now lowers to an **if-else
    // comparison chain** — each `of` tests `disc == v` and br_ifs to its body — instead of
    // fail-closing or emitting a million-entry table. Real nimony emits these for scattered
    // enum/tag ordinals and string-dispatch case statements. Both engines; no br_table.
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
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        !text.contains("br_table"),
        "sparse case must use a comparison chain, not a br_table:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[0]), 1);
    assert_eq!(run(&m, 0, &[1000000]), 2);
    assert_eq!(run(&m, 0, &[5]), 0); // no match → else
    assert_eq!(run(&m, 0, &[999999]), 0);
}

#[test]
fn sparse_case_with_range_and_multivalue() {
    // A comparison chain with a **multi-value** `of` and a **range** `of` over a sparse span:
    // 0|5 → 10, 100..200 → 20, else → 0. Exercises the OR-of-equalities and the `l <= disc <= h`
    // range predicate on the chain path.
    let leng = "\
(stmts
 (proc :g.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (case n.0
    (of (ranges 0 5) (stmts . (asgn r.0 10)))
    (of (ranges (range 100 200)) (stmts . (asgn r.0 20)))
    (else (stmts . (asgn r.0 0))))
   (ret r.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[0]), 10);
    assert_eq!(run(&m, 0, &[5]), 10);
    assert_eq!(run(&m, 0, &[3]), 0); // between the two multi-values, no match
    assert_eq!(run(&m, 0, &[150]), 20);
    assert_eq!(run(&m, 0, &[100]), 20);
    assert_eq!(run(&m, 0, &[200]), 20);
    assert_eq!(run(&m, 0, &[201]), 0);
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
