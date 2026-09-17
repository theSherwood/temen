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

/// A short-circuit in a **`while` header** — the shape `if` tests do not reach. The `while` lowering
/// allocates its header block before evaluating the condition and points the back-edge at it, while
/// the `and`/`or` lowering *splits* that block, so the loop must still re-test the whole condition
/// on every iteration. Fuel is bounded: a wrong back-edge spins, and this reports it instead of
/// hanging the suite.
#[test]
fn short_circuit_in_a_while_header() {
    // count(n, flag) { i = 0; while i < n and flag != 0: i += 1; return i }
    let leng = "\
(stmts
 (proc :count.0 (params (param :n.0 . (i +64)) (param :flag.0 . (i +64))) (i +64) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (and (lt i.0 n.0) (lt 0 flag.0))
     (stmts . (asgn i.0 (add (i +64) i.0 1))))
   (ret i.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    for (n, flag, want) in [(0i64, 1i64, 0i64), (5, 1, 5), (5, 0, 0), (1, 1, 1)] {
        let mut fuel = 5_000_000u64;
        let got = temen_interp::run(&m, 0, &[Value::I64(n), Value::I64(flag)], &mut fuel)
            .unwrap_or_else(|t| panic!("count({n},{flag}) trapped: {t:?} (fuel left {fuel})"));
        assert!(
            fuel > 0,
            "count({n},{flag}) exhausted fuel — the loop does not terminate"
        );
        assert_eq!(got.as_slice(), &[Value::I64(want)], "count({n},{flag})");
    }
}

/// The same, with `or` in the header.
#[test]
fn or_in_a_while_header() {
    // count(n, m) { i = 0; while i < n or i < m: i += 1; return i }  -> max(n, m), floored at 0
    let leng = "\
(stmts
 (proc :count2.0 (params (param :n.0 . (i +64)) (param :m.0 . (i +64))) (i +64) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (or (lt i.0 n.0) (lt i.0 m.0))
     (stmts . (asgn i.0 (add (i +64) i.0 1))))
   (ret i.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    for (a, b, want) in [(0i64, 0i64, 0i64), (3, 1, 3), (1, 4, 4), (2, 2, 2)] {
        let mut fuel = 5_000_000u64;
        let got = temen_interp::run(&m, 0, &[Value::I64(a), Value::I64(b)], &mut fuel)
            .unwrap_or_else(|t| panic!("count2({a},{b}) trapped: {t:?} (fuel left {fuel})"));
        assert!(
            fuel > 0,
            "count2({a},{b}) exhausted fuel — the loop does not terminate"
        );
        assert_eq!(got.as_slice(), &[Value::I64(want)], "count2({a},{b})");
    }
}
