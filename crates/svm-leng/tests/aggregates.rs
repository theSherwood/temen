//! Aggregate access tests (NIM.md Phase 2): `dot` (object fields), `at` (array elements), and
//! `pat` (pointer indexing), lowered to window address arithmetic + load/store via `lvalue_addr`.
//! Objects/arrays are supplied by the caller as window offsets (no aggregate frame vars needed to
//! exercise the addressing). Run on both engines.

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
fn object_field_dot() {
    // Point { x: int; y: int }.  set_get(p: Point, vx, vy): int = { p.x=vx; p.y=vy; p.x + p.y }
    // p is an object param → passed by address; fields x@0, y@8.
    let leng = "\
(stmts
 (type :Point.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :set_get.0 (params (param :p.0 . Point.0.) (param :vx.0 . (i +64)) (param :vy.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (dot p.0 x.0 0) vx.0)
   (asgn (dot p.0 y.0 0) vy.0)
   (ret (add (i +64) (dot p.0 x.0 0) (dot p.0 y.0 0))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // p points at a 16-byte object in the window; write x=vx, y=vy, read back the sum.
    assert_eq!(run(&m, 0, &[128, 3, 4]), 7);
    assert_eq!(run(&m, 0, &[256, -10, 25]), 15);
}

#[test]
fn array_index_at_and_pat() {
    // set_get_at(a: ptr Arr4, i, v): int = { (deref a)[i] = v; (deref a)[i] }  — array via a pointer.
    let leng = "\
(stmts
 (type :Arr4.0. . (array (i +64) 4))
 (proc :set_get_at.0 (params (param :a.0 . (ptr Arr4.0.)) (param :i.0 . (i +64)) (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (at (deref a.0) i.0) v.0)
   (ret (at (deref a.0) i.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // a → an array in the window; write a[i]=v, read back.
    assert_eq!(run(&m, 0, &[512, 0, 11]), 11);
    assert_eq!(run(&m, 0, &[512, 3, 99]), 99);
}

#[test]
fn pointer_index_pat() {
    // set_get_pat(p: aptr int, i, v): int = { p[i] = v; p[i] }
    let leng = "\
(stmts
 (proc :set_get_pat.0 (params (param :p.0 . (aptr (i +64))) (param :i.0 . (i +64)) (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn (pat p.0 i.0) v.0)
   (ret (pat p.0 i.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[1024, 0, 7]), 7);
    assert_eq!(run(&m, 0, &[1024, 5, 42]), 42);
    // distinct indices write distinct slots (no aliasing): sum after two writes handled by caller.
}

/// Real nimony hexer output for `proc dot2(p: Point): int = p.x*p.x + p.y*p.y`, translated out of
/// the full module — real `(type … (object …))` + `(dot p.0 x.0 0)` bytes. The object param is a
/// window address; with a zeroed window it returns 0, but the point is that the real `dot` lowering
/// translates, verifies, and agrees interp == JIT.
#[test]
fn real_nimony_object_dot() {
    const REAL: &str = include_str!("fixtures/real_aggregates.leng.nif");
    let m = svm_leng::translate_proc(REAL, "dot2.0.")
        .unwrap_or_else(|e| panic!("translate real dot2: {e}"));
    svm_verify::verify_module(&m).expect("verify real dot2");
    // p → a zeroed 16-byte object; x=y=0 → 0. Consistency across engines is the real assertion.
    assert_eq!(run(&m, 0, &[256]), 0);
}

#[test]
fn aggregate_local_in_frame() {
    // A local array lives in the frame: fill5(): int = { var a: Arr3; a[0]=10; a[1]=20; a[2]=30; a[0]+a[1]+a[2] }
    let leng = "\
(stmts
 (type :Arr3.0. . (array (i +64) 3))
 (proc :fill.0 . (i +64) .
  (stmts .
   (var :a.0 . Arr3.0. .)
   (asgn (at a.0 0) 10)
   (asgn (at a.0 1) 20)
   (asgn (at a.0 2) 30)
   (ret (add (i +64) (add (i +64) (at a.0 0) (at a.0 1)) (at a.0 2))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = svm_leng::translate_to_text(leng).unwrap();
    // The aggregate local forces a frame → the proc gains a stack-pointer param.
    assert!(
        text.contains("func (i64) -> (i64)"),
        "aggregate-local proc gains a stack pointer:\n{text}"
    );
    let sp = 4096;
    assert_eq!(run(&m, 0, &[sp]), 60);
}

#[test]
fn oconstr_aggregate_argument() {
    // An aggregate literal in *argument* position: `caller` builds `{a:x, b:7}` inline and passes it
    // to `addp` by value. The `(oconstr …)` is constructed into a scratch frame temp and passed
    // by-address. `addp` reads a+b through its by-address param. caller(5) = addp({5,7}) = 12.
    let leng = "\
(stmts
 (type :P.0. . (object . (fld :a.0 . (i +64)) (fld :b.0 . (i +64))))
 (proc :addp.0 (params (param :p.0 . P.0.)) (i +64) .
  (stmts . (ret (add (i +64) (dot p.0 a.0 0) (dot p.0 b.0 0)))))
 (proc :caller.0 (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (call addp.0 (oconstr P.0. (kv a.0 x.0) (kv b.0 7)))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // caller is frame-needing (holds the temp): ($sp, x) -> i64.
    assert_eq!(run(&m, 1, &[4096, 5]), 12, "addp of a+b with a=5, b=7");
    assert_eq!(run(&m, 1, &[4096, 100]), 107);
}

#[test]
fn oconstr_in_expression_position() {
    // An `(oconstr …)` in general **expression** position (a `discard`ed object literal), not a call
    // argument — the case a stdlib `result = mk(); discard mkPair()` hits, which svm-leng fail-closed
    // on before #990. `expr()` now materializes it into a scratch temp (the same lowering the
    // call-argument case uses), and the position-aware `agg_temp_bytes` reserves that temp's bytes so
    // it can't stomp the frame. `acc` is address-taken (`(addr acc)`), so it is frame-resident; the
    // discarded 16-byte `Pair` temp must not corrupt it — reading `acc` back must still see `n`.
    let leng = "\
(stmts
 (type :Pair.0. . (object . (fld :a.0 . (i +64)) (fld :b.0 . (i +64))))
 (proc :f.0 (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (var :acc.0 . (i +64) n.0)
   (var :p.0 . (ptr (i +64)) (addr acc.0))
   (discard (oconstr Pair.0. (kv a.0 111) (kv b.0 222)))
   (ret (deref p.0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // f is frame-needing (acc is address-taken + the oconstr temp): ($sp, n) -> i64.
    assert_eq!(
        run(&m, 0, &[8192, 42]),
        42,
        "a discarded oconstr temp in expression position leaves the frame local intact"
    );
    assert_eq!(run(&m, 0, &[8192, 7]), 7);
}
