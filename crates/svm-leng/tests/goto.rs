//! General `goto` tests (NIM.md Phase 2, W1 Leng totality). hexer keeps `if`/`while` structured and
//! emits the low-level jump family only for `break`/`block`-`break`: `(jmp L)` an unconditional
//! branch, `(lab :L)` a label definition. Both lower onto the block-parameter (slot-threading)
//! model — a label is just another successor block carrying all slots; a jump passes the live slot
//! set — so forward jumps out of a loop/block Just Work. Validated on both engines (§9 parity),
//! hand-written and against genuine nimony `firstHit`/`labeled` bytes.

use svm_interp::Value;

/// Translate, verify, run func `idx` with i64 `args` on interp + JIT; assert both agree, return it.
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
fn forward_jmp_over_code() {
    // pick(a): int = ( result = a; jmp skip; result = 0; lab skip; result )  — the `jmp` skips the
    // dead `result = 0`, so the answer is always `a`.
    let leng = "\
(stmts
 (proc :pick.0 (params (param :a.0 . (i +64))) (i +64) .
  (stmts .
   (var :result.0 . (i +64) .)
   (asgn result.0 a.0)
   (jmp skip.0)
   (asgn result.0 0)
   (lab :skip.0)
   (ret result.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(text.contains("br "), "jmp lowers to a branch:\n{text}");
    assert_eq!(run(&m, 0, &[42]), 42);
    assert_eq!(run(&m, 0, &[-7]), -7);
}

#[test]
fn break_out_of_while() {
    // countTo(n, k): int = ( result = 0; i = 0; while i < n: { if i == k: jmp done; result = result
    // + 1; inc i }; lab done; result )  — count iterations until i hits k, else n. `jmp done` breaks.
    let leng = "\
(stmts
 (proc :countTo.0 (params (param :n.0 . (i +64)) (param :k.0 . (i +64))) (i +64) .
  (stmts .
   (var :result.0 . (i +64) .)
   (var :i.0 . (i +64) .)
   (asgn result.0 0)
   (asgn i.0 0)
   (while (lt i.0 n.0)
    (stmts .
     (if (elif (eq i.0 k.0) (stmts . (jmp done.0))))
     (asgn result.0 (add (i +64) result.0 1))
     (asgn i.0 (add (i +64) i.0 1))))
   (lab :done.0)
   (ret result.0))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[10, 3]), 3, "breaks at i==3");
    assert_eq!(run(&m, 0, &[10, 0]), 0, "breaks immediately");
    assert_eq!(run(&m, 0, &[4, 99]), 4, "k never hit → full count");
}

#[test]
fn jmp_to_unknown_label_is_fail_closed() {
    let leng = "\
(stmts
 (proc :bad.0 . (i +64) .
  (stmts .
   (jmp nowhere.0)
   (ret 0))))";
    match svm_leng::translate(leng) {
        Err(svm_leng::LengError::Unsupported(_)) | Err(svm_leng::LengError::Malformed(_)) => {}
        other => panic!("expected fail-closed, got {other:?}"),
    }
}

/// Real nimony `hexer` output for `firstHit(n,k)` — `while` with a `break` that hexer lowers to
/// `(jmp whileStmtLabel.0)` and a trailing `(lab :whileStmtLabel.0)`. All-SSA (no frame), so it runs
/// with no bound imports.
#[test]
fn real_nimony_break() {
    const REAL: &str = include_str!("fixtures/real_goto.leng.nif");
    // firstHit(n,k): result = -1; i = 0; while i<n: if i==k: result = i; break; i = i+1.
    let m = svm_leng::translate_proc(REAL, "firstHit.0.")
        .unwrap_or_else(|e| panic!("translate real firstHit: {e}"));
    assert_eq!(run(&m, 0, &[10, 3]), 3, "k=3 found at index 3");
    assert_eq!(run(&m, 0, &[10, 0]), 0, "k=0 found at index 0");
    assert_eq!(run(&m, 0, &[5, 99]), -1, "k not present → -1");
}

/// Real nimony `hexer` output for `labeled(n)` — a `block done:`/`break done` lowered to
/// `(jmp done.0)` out of a nested `while`, with the `(lab :done.0)` after the block's scope.
#[test]
fn real_nimony_labeled_block() {
    const REAL: &str = include_str!("fixtures/real_goto.leng.nif");
    // labeled(n): result = 0; block done: (i=0; while i<n: if i==3: break done; result += i; i=i+1).
    let m = svm_leng::translate_proc(REAL, "labeled.0.")
        .unwrap_or_else(|e| panic!("translate real labeled: {e}"));
    assert_eq!(run(&m, 0, &[5]), 3, "0+1+2 then break at i==3");
    assert_eq!(run(&m, 0, &[2]), 1, "0+1, loop exits before i==3");
    assert_eq!(run(&m, 0, &[0]), 0, "empty loop");
}
