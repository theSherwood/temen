//! Indirect calls through function pointers (`proctype`) — the nimony backend's dynamic-dispatch
//! path. A `proctype` value is an `i32` **function index** (`ref.func`); calling through it lowers to
//! `call_indirect`, whose masked table dispatch + runtime signature check (§3c) is the security
//! hinge — the verifier and engine carry it, svm-leng only emits the op. Both engines, §9 parity.

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
fn store_funcref_then_call_through_field() {
    // A `Box{fn: proc(int):int, v: int}`. `run(b)` stores `ref.func dbl` into `b.fn`, then calls
    // `b.fn(b.v)`. dbl doubles, so run over v=21 → 42 — proving ref.func + call_indirect round-trip.
    let leng = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (type :Box.0. . (object . (fld :fn.0 . IntFn.0.) (fld :v.0 . (i +64))))
 (proc :dbl.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 x.0))))
 (proc :run.0 (params (param :b.0 . (ptr Box.0.))) (i +64) .
  (stmts .
   (asgn (dot (deref b.0) fn.0 0) dbl.0)
   (ret (call (dot (deref b.0) fn.0 0) (dot (deref b.0) v.0 0))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(text.contains("ref.func 0"), "run stores ref.func dbl:\n{text}");
    assert!(text.contains("call_indirect"), "run dispatches indirectly:\n{text}");
    // dbl = func 0, run = func 1. Box at offset 128: fn@0 (overwritten by the store), v@8 = 21.
    // run(128) stores ref.func dbl into b.fn, then b.fn(b.v) = dbl(21) = 21*21 = 441.
    let b = 128usize;
    assert_eq!(run_with_seed(&m, 1, &[b as i64], b + 8, 21), 441);
}

/// Like `run`, but pre-seed an i64 at `off` in the window before running (both engines, parity).
fn run_with_seed(m: &svm_ir::Module, idx: u32, args: &[i64], off: usize, val: i64) -> i64 {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let mut seed = vec![0u8; 4096];
    seed[off..off + 8].copy_from_slice(&val.to_le_bytes());
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let (ir, _) = svm_interp::run_capture(m, idx, &ivals, &mut fuel, &seed);
    let iout = ir.expect("interp");
    let iword = match iout.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("unexpected {o:?}"),
    };
    let (jout, _) = svm_jit::compile_and_run_capture(m, idx, args, &seed).expect("jit");
    let jword = match jout {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(vec![iword], jword, "§9 interp/JIT parity");
    iword
}

#[test]
fn call_through_funcref_param() {
    // `apply(f: proc(int):int, x): int = f(x)` — an indirect call through a scalar funcref *param*.
    // A driver stores ref.func into a Box and passes it; here we test the param path directly by
    // having `runner(b)` load the funcref from memory and dispatch. dbl(x)=x*x → 7 → 49.
    let leng = "\
(stmts
 (type :IntFn.0. . (proctype . (params (param :x.0 . (i +64))) (i +64) (pragmas (nimcall))))
 (proc :dbl.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 x.0))))
 (proc :apply.0 (params (param :f.0 . IntFn.0.) (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (call f.0 x.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(text.contains("call_indirect"), "apply dispatches indirectly:\n{text}");
    // apply = func 1; f is the i32 funcref index of dbl (func 0), x = 7 → 49.
    assert_eq!(run(&m, 1, &[0, 7]), 49);
}
