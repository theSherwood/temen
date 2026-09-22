//! Control-flow tests (NIM.md Phase 2 broadening): `if`/`elif`/`else`, `while`, and comparisons,
//! lowered to multi-block TEMEN-IR with locals threaded as block parameters (the chibicc φ model).
//! Hand-written fixtures plus a **real** nimony proc (`maxi`, an `if`/`else`), all run on both
//! the interpreter and the JIT with matching results (§9 parity).

use temen_interp::Value;

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

#[test]
fn if_else_max() {
    // maxi(a, b) = if a < b: b else: a
    let leng = "\
(stmts
 (proc :maxi.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (if
    (elif (lt a.0 b.0)
     (stmts . (asgn r.0 b.0)))
    (else
     (stmts . (asgn r.0 a.0))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[3, 9]), 9);
    assert_eq!(run(&m, 0, &[9, 3]), 9);
    assert_eq!(run(&m, 0, &[5, 5]), 5);
}

#[test]
fn while_sum_to_n() {
    // sumto(n) = { r=0; i=1; while i <= n: { r += i; i += 1 } ; r }  = n(n+1)/2
    let leng = "\
(stmts
 (proc :sumto.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) 0)
   (var :i.0 . (i +64) 1)
   (while (le i.0 n.0)
    (stmts .
     (asgn r.0 (add (i +64) r.0 i.0))
     (asgn i.0 (add (i +64) i.0 1))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[10]), 55);
    assert_eq!(run(&m, 0, &[100]), 5050);
    assert_eq!(run(&m, 0, &[0]), 0);
}

#[test]
fn elif_chain_sign() {
    // sign(x) = if x < 0: -1 elif 0 < x: 1 else: 0
    let leng = "\
(stmts
 (proc :sign.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (var :s.0 . (i +64) .)
   (if
    (elif (lt x.0 0) (stmts . (asgn s.0 (neg (i +64) 1))))
    (elif (lt 0 x.0) (stmts . (asgn s.0 1)))
    (else (stmts . (asgn s.0 0))))
   (ret s.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[-7]), -1);
    assert_eq!(run(&m, 0, &[42]), 1);
    assert_eq!(run(&m, 0, &[0]), 0);
}

/// The real thing: nimony's own hexer output for `proc maxi(a,b: int): int = (if a > b: a else: b)`,
/// which lowers `a > b` to `(lt b a)` with an `if`/`else`. Translate it out of the full module.
#[test]
fn real_nimony_if_else() {
    const REAL: &str = include_str!("fixtures/real_controlflow.leng.nif");
    let m = temen_leng::translate_proc(REAL, "maxi.0.")
        .unwrap_or_else(|e| panic!("translate real maxi: {e}"));
    assert_eq!(run(&m, 0, &[3, 9]), 9);
    assert_eq!(run(&m, 0, &[9, 3]), 9);
    assert_eq!(run(&m, 0, &[-4, -10]), -4);
}

/// **A `scope`'s children are its statements, wrapped in `stmts` or not** (the v0.6.2 bump's
/// no-output blocker).
///
/// hexer wraps an inlined proc body in `(scope (stmts …))` most of the time, but when the body is a
/// single statement plus its return label it emits them **bare**: `+=` on a `var` parameter comes out
/// as `(scope (scope (asgn …) (lab returnLabel)))`. Recursing only into `stmts`-tagged children
/// dropped every such statement *silently* — `wbuf.setLen`, `copyMem` and every loop increment
/// vanished, so the whole nim corpus compiled, verified, ran to completion and printed nothing, and
/// `rawWriteAll`'s `off += k` left a loop that never advanced. A dropped statement must never be a
/// quiet success, so this pins both shapes against the same expected value.
#[test]
fn bare_statements_inside_a_scope_are_not_dropped() {
    // sumto(n) with the loop's two updates buried in bare `scope`s, the way hexer inlines them.
    let leng = "\
(stmts
 (proc :sumto.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) 0)
   (var :i.0 . (i +64) 1)
   (while (le i.0 n.0)
    (stmts .
     (scope (scope (asgn r.0 (add (i +64) r.0 i.0)) (lab :rl.0)))
     (scope (asgn i.0 (add (i +64) i.0 1)))))
   (ret r.0))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[10]), 55, "bare `scope` statements must run");
    assert_eq!(run(&m, 0, &[1]), 1);
    assert_eq!(run(&m, 0, &[0]), 0);
}
