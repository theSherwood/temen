//! Exception tests (NIM.md Phase 2, W1 Leng totality). nimony lowers exceptions to an **error-flag
//! ABI** entirely out of constructs the frontend already has — no new node type:
//!
//!   * a `.raises` proc returns a tuple `(object (fld :fld.0 ErrorCode) (fld :fld.1 result))` by
//!     **sret** — `fld.0` is the error code (an `enum`, i.e. a scalar), `fld.1` the real result;
//!   * `raise E` is `ret (oconstr tuple (kv fld.0 <code>) (kv fld.1 <default>))` — a non-zero code;
//!   * the normal return sets `fld.0 = 0`;
//!   * `try/except` is `var canRaise = call; if canRaise.fld.0: jmp exlab; result = canRaise.fld.1`
//!     with the handler under an `if (false) { lab exlab; … }` guard reached only via the `jmp`.
//!
//! So exceptions fall out of sret + objects + `if` + `jmp`/`lab` + enum-scalar error codes. Both
//! engines, hand-written and against genuine nimony `mayFail`/`guarded` bytes.

use svm_interp::Value;

/// Run func `idx` (which needs a frame `$sp`) with `sp` + i64 `args` on both engines; assert parity.
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

/// A hand-written model of nimony's error-flag ABI: an `Err` tuple `(code, value)`, a `mayFail` that
/// returns it by sret (code != 0 on the error path), and a `guarded` that checks the code, jumps to
/// a handler on error, else takes the value.
const HANDWRITTEN: &str = "\
(stmts
 (type :Err.0. . (object . (fld :code.0 . (i +32)) (fld :val.0 . (i +64))))
 (proc :mayFail.0 (params (param :x.0 . (i +64))) Err.0. .
  (stmts .
   (var :result.0 . Err.0. .)
   (asgn (dot result.0 code.0 0) 0)
   (if (elif (lt x.0 0)
        (stmts . (ret (oconstr Err.0. (kv code.0 1) (kv val.0 0))))))
   (asgn (dot result.0 val.0 0) (mul (i +64) x.0 2))
   (ret result.0)))
 (proc :guarded.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts .
   (var :r.0 . (i +64) .)
   (stmts .
    (var :canRaise.0 . Err.0. (call mayFail.0 x.0))
    (if (elif (dot canRaise.0 code.0 0) (stmts . (jmp exlab.0))))
    (asgn r.0 (dot canRaise.0 val.0 0)))
   (if (elif (false) (stmts .
        (lab :exlab.0)
        (asgn r.0 -1))))
   (ret r.0))))";

#[test]
fn error_flag_happy_and_error_paths() {
    let m = svm_leng::translate(HANDWRITTEN).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 8192;
    // guarded is func 1 and needs a frame (canRaise is an aggregate var).
    assert_eq!(run(&m, 1, &[sp, 5]), 10, "no raise → 5*2");
    assert_eq!(run(&m, 1, &[sp, 21]), 42, "no raise → 21*2");
    assert_eq!(run(&m, 1, &[sp, -3]), -1, "x<0 raises → handler returns -1");
}

#[test]
fn aggregate_returning_call_of_raiser_directly() {
    // A `.raises` proc returned straight back out (propagation) — the sret tuple flows through.
    let leng = "\
(stmts
 (type :Err.0. . (object . (fld :code.0 . (i +32)) (fld :val.0 . (i +64))))
 (proc :mayFail.0 (params (param :x.0 . (i +64))) Err.0. .
  (stmts .
   (if (elif (lt x.0 0)
        (stmts . (ret (oconstr Err.0. (kv code.0 1) (kv val.0 0))))))
   (ret (oconstr Err.0. (kv code.0 0) (kv val.0 x.0)))))
 (proc :propagate.0 (params (param :x.0 . (i +64))) Err.0. .
  (stmts .
   (ret (call mayFail.0 x.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // propagate returns the tuple by sret: (i64 $sret, i64 x) -> ().  Run it and read code/val back.
    let seed = vec![0u8; 8192];
    let dst = 4096usize;
    let mut fuel = u64::MAX;
    let (ir, imem) = svm_interp::run_capture(
        &m,
        1,
        &[Value::I64(dst as i64), Value::I64(-4)],
        &mut fuel,
        &seed,
    );
    ir.expect("interp");
    let code = i32::from_le_bytes(imem[dst..dst + 4].try_into().unwrap());
    assert_eq!(code, 1, "propagated error code");
}

/// Real nimony `hexer` output for a `distinct`-int-raising `mayFail` and its `try/except` `guarded`
/// caller. `mayFail(x): int {.raises: MyError.}` returns the `(ErrorCode, int)` tuple by sret;
/// `guarded` runs it, and on a non-zero `fld.0` jumps to the `except` handler (returning -1),
/// otherwise takes `fld.1`. The genuine error-flag ABI, end-to-end on both engines.
#[test]
fn real_nimony_try_except() {
    const REAL: &str = include_str!("fixtures/real_exception.leng.nif");
    let m = svm_leng::translate_procs(REAL, &["mayFail.0.", "guarded.0."])
        .unwrap_or_else(|e| panic!("translate real mayFail/guarded: {e}"));
    let sp = 8192;
    // guarded is func 1 (needs a frame for its `canRaise` tuple).
    assert_eq!(run(&m, 1, &[sp, 5]), 10, "no raise → 5*2");
    assert_eq!(run(&m, 1, &[sp, 7]), 14, "no raise → 7*2");
    assert_eq!(run(&m, 1, &[sp, -2]), -1, "x<0 raises → except returns -1");
    assert_eq!(run(&m, 1, &[sp, 0]), 0, "0 not < 0 → 0*2");
}
