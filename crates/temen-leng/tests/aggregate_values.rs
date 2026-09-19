//! Whole-aggregate value tests (NIM.md Phase 2): object constructors (`oconstr`), array
//! constructors (`aconstr`), and whole-aggregate copy (`mem.copy`) into a destination. Aggregate
//! locals live in the frame; construct in place, then read back. Both engines.

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
fn object_constructor_then_read() {
    // mkSum(a,b): var p = Pt(x:a, y:b); p.x + p.y
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :mkSum.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . Pt.0. (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))
   (ret (add (i +64) (dot p.0 x.0 0) (dot p.0 y.0 0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 20480;
    assert_eq!(run(&m, 0, &[sp, 3, 4]), 7);
    assert_eq!(run(&m, 0, &[sp, 10, -25]), -15);
}

#[test]
fn whole_aggregate_copy() {
    // copySum(a,b): var p = Pt(x:a,y:b); var q: Pt = p; q.x + q.y   (q is a byte-copy of p)
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :copySum.0 (params (param :a.0 . (i +64)) (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (var :p.0 . Pt.0. (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))
   (var :q.0 . Pt.0. p.0)
   (ret (add (i +64) (dot q.0 x.0 0) (dot q.0 y.0 0))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let text = temen_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("mem.copy"),
        "copy should use mem.copy:\n{text}"
    );
    assert_eq!(run(&m, 0, &[24576, 5, 6]), 11);
}

#[test]
fn array_constructor_then_index() {
    // sum3(): var a: Arr3 = [10, 20, 30]; a[0] + a[1] + a[2]
    let leng = "\
(stmts
 (type :Arr3.0. . (array (i +64) 3))
 (proc :sum3.0 . (i +64) .
  (stmts .
   (var :a.0 . Arr3.0. (aconstr Arr3.0. 10 20 30))
   (ret (add (i +64) (add (i +64) (at a.0 0) (at a.0 1)) (at a.0 2))))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[20480]), 60);
}

/// Real nimony `hexer` output for `mkSum(a,b): var p = Pt(x:a,y:b); p.x + p.y` — genuine
/// `(var :p.0 . Pt.0. (oconstr Pt.0. (kv x.0 a.0) (kv y.0 b.0)))` bytes.
#[test]
fn real_nimony_oconstr() {
    const REAL: &str = include_str!("fixtures/real_oconstr.leng.nif");
    let m = temen_leng::translate_proc(REAL, "mkSum.0.")
        .unwrap_or_else(|e| panic!("translate real mkSum: {e}"));
    let sp = 20480;
    assert_eq!(run(&m, 0, &[sp, 3, 4]), 7);
    assert_eq!(run(&m, 0, &[sp, 100, 200]), 300);
}

#[test]
fn array_literal_indexed_by_a_runtime_value() {
    // #760: `[10, 20, 30][i]` — a constant array literal in *lvalue* position, indexed by a
    // runtime value. hexer emits exactly this for table lookups (`lifter.nim` does
    // `addParLe(c.dest, [ParLe, ParRi][k])`), and it was the first construct standing between
    // hexer and the pure no-C path: `lvalue_addr`'s `at` arm recursed into a base that is not an
    // lvalue. There is no object to address, so the literal materializes into a frame temp.
    //
    // Indexing is by a *parameter*, not a constant, so constant-folding cannot turn this back
    // into the `var a = [...]` case `array_constructor_then_index` already covers.
    let leng = "\
(stmts
 (type :Arr3.0. . (array (i +64) 3))
 (proc :pick.0 (params (param :i.0 . (i +64))) (i +64) .
  (stmts .
   (ret (at (aconstr Arr3.0. 10 20 30) i.0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 20480;
    assert_eq!(run(&m, 0, &[sp, 0]), 10);
    assert_eq!(run(&m, 0, &[sp, 1]), 20);
    assert_eq!(run(&m, 0, &[sp, 2]), 30);
}

#[test]
fn object_literal_field_read_without_a_local() {
    // The `dot` sibling of the above: an object constructor in lvalue position, never bound to a
    // local. Same materialization path, different accessor — pinned so a future narrowing of the
    // `aconstr` arm to arrays alone fails here rather than silently regressing `oconstr`.
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64)) (fld :y.0 . (i +64))))
 (proc :gety.0 (params (param :b.0 . (i +64))) (i +64) .
  (stmts .
   (ret (dot (oconstr Pt.0. (kv x.0 7) (kv y.0 b.0)) y.0 0)))))";
    let m = temen_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 20480;
    assert_eq!(run(&m, 0, &[sp, 42]), 42);
    assert_eq!(run(&m, 0, &[sp, -5]), -5);
}
