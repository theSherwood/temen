//! End-to-end tests for the `svm-leng` walking skeleton (NIM.md Phase 2): hand-written Leng-NIF
//! (faithful to `doc/leng-spec.md`) → SVM-IR → verify → run on **both** the interpreter and the
//! JIT, asserting the computed values agree with the expected result. The two-backend shape is the
//! §9 parity discipline the other frontends follow (`svm-wasm`'s `run_both`).

use svm_interp::Value;
use svm_leng::{translate, LengError};

/// Translate, verify, then run function `idx` with i64 `args` on interp + JIT; assert both return
/// the same single i64 and return it. (JIT uses raw i64 words for args/results; interp uses `Value`.)
fn run_i64(leng: &str, idx: usize, args: &[i64]) -> i64 {
    let module = translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    svm_verify::verify_module(&module)
        .unwrap_or_else(|e| panic!("verify (untrusted frontend): {e:?}"));

    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(&module, idx as u32, &ivals, &mut fuel).expect("interp run");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected one i64 result, got {other:?}"),
    };

    let jit = match svm_jit::compile_and_run(&module, idx as u32, args).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    assert_eq!(
        jit.as_slice(),
        &[interp_n],
        "interp/JIT divergence (§9 parity)"
    );
    interp_n
}

#[test]
fn constant_arithmetic() {
    // main(): int = 3 + 4*2 = 11  (the doc-comment example).
    let leng =
        "(stmts (proc :main.0 . (i +64) . (stmts . (ret (add (i +64) 3 (mul (i +64) 4 2))))))";
    assert_eq!(run_i64(leng, 0, &[]), 11);
}

#[test]
fn suffixed_literals() {
    // nimony emits typed literals as `(suf <value> "<type>")` — e.g. `255'i64`. They evaluate to the
    // value at the suffix's width. `(add (i +64) (suf 200 "i64") (suf 55 "u8")) = 255`.
    let leng = "\
(stmts
 (proc :main.0 . (i +64) .
  (stmts . (ret (add (i +64) (suf 200 \"i64\") (suf 55 \"u8\"))))))";
    assert_eq!(run_i64(leng, 0, &[]), 255);
}

#[test]
fn bitwise_and_shift_ops() {
    // nimony's arithmetics: bitand/bitor/bitxor/shl/shr/bitnot, same (Type a b) shape as arith.
    // ((0xF0 & 0x3C) | 0x01) = 0x30 | 1 = 0x31; (0x31 ^ 0x0F) = 0x3E; (0x3E << 2) = 0xF8;
    // (0xF8 >> 3) = 0x1F = 31.
    let leng = "\
(stmts
 (proc :main.0 . (i +64) .
  (stmts .
   (ret (shr (u +64)
     (shl (i +64)
       (bitxor (i +64)
         (bitor (i +64) (bitand (i +64) 240 60) 1)
         15)
       2)
     3)))))";
    assert_eq!(run_i64(leng, 0, &[]), 31);
}

#[test]
fn narrow_int_local() {
    // A sub-word (`i8`) local — like `system`'s `min`/`max` result — now gets its own (i32) SSA slot
    // and can be assigned in both `if` branches, then widened on return. min(3,5)=3, min(200,5)=5.
    let leng = "\
(stmts
 (proc :m.0 (params (param :x.0 . (i +64)) (param :y.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i 8) .)
   (if (elif (le x.0 y.0) (stmts . (asgn r.0 x.0)))
       (else (stmts . (asgn r.0 y.0))))
   (ret (conv (i +64) r.0)))))";
    assert_eq!(run_i64(leng, 0, &[3, 5]), 3);
    assert_eq!(run_i64(leng, 0, &[100, 5]), 5);
}

#[test]
fn bool_case_values() {
    // `case` over a bool branches on `(true)`/`(false)` literals (system's `$` for bool).
    let leng = "\
(stmts
 (proc :b.0 (params (param :e.0 . (bool))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (case e.0
    (of (ranges (false)) (stmts . (asgn r.0 10)))
    (of (ranges (true)) (stmts . (asgn r.0 20))))
   (ret r.0))))";
    assert_eq!(run_i64(leng, 0, &[0]), 10);
    assert_eq!(run_i64(leng, 0, &[1]), 20);
}

#[test]
fn char_literals() {
    // nimony emits char literals as `'c'` / `'\HH'`. `'0'` = 48, `'\0A'` = 10; 48 + 10 = 58.
    let leng = "\
(stmts
 (proc :m.0 . (i +64) . (stmts . (ret (add (i +64) '0' '\\0A')))))";
    assert_eq!(run_i64(leng, 0, &[]), 58);
}

#[test]
fn bitnot_and_logical_not() {
    // bitnot: ~0 = -1. not: logical negation of a bool (0/1).
    assert_eq!(
        run_i64(
            "(stmts (proc :m.0 . (i +64) . (stmts . (ret (bitnot (i +64) 0)))))",
            0,
            &[]
        ),
        -1
    );
    // not(3 == 4) = not(false) = true = 1.
    assert_eq!(
        run_i64(
            "(stmts (proc :m.0 . (i +64) . (stmts . (ret (not (eq 3 4))))))",
            0,
            &[]
        ),
        1
    );
    // not(2 == 2) = not(true) = false = 0.
    assert_eq!(
        run_i64(
            "(stmts (proc :m.0 . (i +64) . (stmts . (ret (not (eq 2 2))))))",
            0,
            &[]
        ),
        0
    );
}

#[test]
fn params_and_locals() {
    // addup(a, b): int = (var t = b*2; a + t)
    let leng = "\
(stmts
 (proc :addup.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :t.0 . (i +64) (mul (i +64) b.0 2))
   (ret (add (i +64) a.0 t.0)))))";
    assert_eq!(run_i64(leng, 0, &[3, 4]), 11);
    assert_eq!(run_i64(leng, 0, &[10, 20]), 50);
}

#[test]
fn direct_call_between_procs() {
    // addup is func 0; main() = addup(3,4) + addup(10,20) = 11 + 50 = 61.
    let leng = "\
(stmts
 (proc :addup.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) a.0 (mul (i +64) b.0 2)))))
 (proc :main.0 . (i +64) .
  (stmts .
   (ret (add (i +64)
             (call addup.0 3 4)
             (call addup.0 10 20))))))";
    assert_eq!(run_i64(leng, 1, &[]), 61);
}

#[test]
fn signed_division_and_mod() {
    // f(a, b): int = (a div b) + (a mod b);  f(17,5) = 3 + 2 = 5.
    let leng = "\
(stmts
 (proc :f.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (div (i +64) a.0 b.0) (mod (i +64) a.0 b.0))))))";
    assert_eq!(run_i64(leng, 0, &[17, 5]), 5);
}

#[test]
fn narrow_int_width() {
    // i32 arithmetic threaded through, widened on return: g(x:i32) = (x + 1) as i64.
    let leng = "\
(stmts
 (proc :g.0 (params (param :x.0 . (i +32))) (i +64) .
  (stmts .
   (var :y.0 . (i +32) (add (i +32) x.0 1))
   (ret (conv (i +64) y.0)))))";
    // arg is passed as i64 but the param is i32; the interp/JIT narrow it. 41+1 = 42.
    // (We pass via i64 harness; value fits i32.)
    assert_eq!(run_i64_i32arg(leng, 0, 41), 42);
}

/// Variant harness for a single i32 parameter (the value still fits, passed as the engine's i32).
fn run_i64_i32arg(leng: &str, idx: usize, arg: i32) -> i64 {
    let module = translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    svm_verify::verify_module(&module).expect("verify");
    let mut fuel = u64::MAX;
    let interp =
        svm_interp::run(&module, idx as u32, &[Value::I32(arg)], &mut fuel).expect("interp");
    let interp_n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        other => panic!("expected i64, got {other:?}"),
    };
    // JIT: an i32 param is passed in a raw i64 word (low 32 bits).
    let jit = match svm_jit::compile_and_run(&module, idx as u32, &[arg as i64]).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    assert_eq!(jit.as_slice(), &[interp_n], "§9 parity");
    interp_n
}

#[test]
fn unsupported_is_fail_closed() {
    // A `try`/exception construct is outside the subset → a clean Unsupported error, never a bad
    // module. (Floats, once unsupported here, are now handled — see tests/floats.rs.)
    let leng = "(stmts (proc :h.0 . (i +64) . (stmts . (try (stmts .) (stmts .) (stmts .)))))";
    match translate(leng) {
        Err(LengError::Unsupported(_)) | Err(LengError::Malformed(_)) => {}
        other => panic!("expected a fail-closed error, got {other:?}"),
    }
}
