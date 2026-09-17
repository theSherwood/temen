//! Boolean `and`/`or` in expression position (#763) — v0.6.2's hexer emits `leng_tags.AndC`/`OrC`
//! there, and the lowering must **short-circuit**: the right operand cannot be evaluated once the
//! left decides the answer.

use temen_interp::Value;

/// Both engines, asserting §9 parity — the same helper the sibling construct tests use.
fn run(module: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    let jit = match temen_jit::compile_and_run(module, idx, args).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 interp/JIT parity");
    interp_n
}

/// `f(a, b) = if a > 0 or b > 0: 1 else: 0` — the plain two-operand `or`.
#[test]
fn or_in_expression_position() {
    let leng = "\
(stmts
 (proc :f.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (if (elif (or (lt 0 a.0) (lt 0 b.0)) (stmts . (asgn r.0 1)))
       (else (stmts . (asgn r.0 0))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[1, 0]), 1);
    assert_eq!(run(&m, 0, &[0, 1]), 1);
    assert_eq!(run(&m, 0, &[0, 0]), 0);
    assert_eq!(run(&m, 0, &[-5, -5]), 0);
}

/// `(or A (and B C))` — the nested shape the real stdlib Leng has, which needs one live temp slot
/// per nesting level.
#[test]
fn nested_and_inside_or() {
    let leng = "\
(stmts
 (proc :g.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64)) (param :c.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (if (elif (or (lt 0 a.0)
                 (and (lt 0 b.0) (lt 0 c.0)))
             (stmts . (asgn r.0 1)))
       (else (stmts . (asgn r.0 0))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[1, 0, 0]), 1); // left decides
    assert_eq!(run(&m, 0, &[0, 1, 1]), 1); // both of the inner and
    assert_eq!(run(&m, 0, &[0, 1, 0]), 0); // inner and is false
    assert_eq!(run(&m, 0, &[0, 0, 1]), 0);
    assert_eq!(run(&m, 0, &[0, 0, 0]), 0);
}

/// `and` alone, and the mirror-image short circuit.
#[test]
fn and_in_expression_position() {
    let leng = "\
(stmts
 (proc :h.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (if (elif (and (lt 0 a.0) (lt 0 b.0)) (stmts . (asgn r.0 1)))
       (else (stmts . (asgn r.0 0))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[1, 1]), 1);
    assert_eq!(run(&m, 0, &[1, 0]), 0);
    assert_eq!(run(&m, 0, &[0, 1]), 0);
}
