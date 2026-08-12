//! Whole-module tests (NIM.md Phase 2): a module with `gvar` globals, a `const`, and several procs
//! that call each other (intra-module) — translated and run as one module, entry chosen by func
//! index. Globals live at fixed window offsets and are shared across calls. Both engines.

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

/// A global counter, a `const` step, and a 3-proc call chain: main → bumpN → bump.
const MODULE: &str = "\
(stmts
 (gvar :counter.0. . (i +64) .)
 (const :step.0. . (i +64) 1)
 (proc :bump.0. . (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. step.0.))))
 (proc :bumpN.0. (params (param :n.0 . (i +64))) (void) .
  (stmts .
   (var :i.0 . (i +64) 0)
   (while (lt i.0 n.0)
    (stmts .
     (call bump.0.)
     (asgn i.0 (add (i +64) i.0 1))))))
 (proc :main.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (call bumpN.0. n.0)
   (ret counter.0.))))";

#[test]
fn globals_const_and_intramodule_calls() {
    let text = svm_leng::translate_to_text(MODULE).unwrap();
    assert!(
        text.starts_with("memory "),
        "globals need a window:\n{text}"
    );
    // three funcs emitted (bump=0, bumpN=1, main=2), no stack pointers (no frames here).
    assert_eq!(text.matches("func (").count(), 3, "3 procs:\n{text}");

    let m = svm_leng::translate(MODULE).unwrap_or_else(|e| panic!("translate: {e}"));
    // main is func 2; counter starts 0 and is bumped n times → returns n.
    assert_eq!(run(&m, 2, &[0]), 0);
    assert_eq!(run(&m, 2, &[7]), 7);
    // The global persists across calls within a run, but each fresh run re-zeroes the window,
    // so a second independent run also starts from 0.
    assert_eq!(run(&m, 2, &[100]), 100);
}

#[test]
fn nonzero_global_initializer_via_data() {
    // A non-zero scalar global initializer becomes a `data` segment seeding the window; the global
    // reads back its initial value. (i32 and i64 widths both.)
    let leng = "\
(stmts
 (gvar :g.0. . (i +64) 5)
 (gvar :h.0. . (i +32) 7)
 (proc :sum.0. . (i +64) . (stmts . (ret (add (i +64) g.0. (conv (i +64) h.0.))))))";
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("data "),
        "non-zero init emits a data segment:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 12, "5 + 7");
}

#[test]
fn aggregate_global_initializer_is_materialized() {
    // A **mutable** global with an aggregate initializer (`var g = Pt(x: 7)`) materializes into a
    // data segment at the global's offset — the same treatment an aggregate `const` gets, but the
    // global stays writable. `readG` reads the field back; both engines see 7. (Real nimony emits
    // these constantly — module-level `var`s of object/string type — so this closes a #760 gap.)
    let leng = "\
(stmts
 (type :Pt.0. . (object . (fld :x.0 . (i +64))))
 (gvar :g.0. . Pt.0. (oconstr Pt.0. (kv x.0 7)))
 (proc :readG.0. . (i +64) . (stmts . (ret (dot g.0. x.0 0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(
        run(&m, 0, &[]),
        7,
        "aggregate gvar init materialized: g.x = 7"
    );
}

#[test]
fn float_global_initializer_is_materialized() {
    // A **float scalar global** — nimony emits `let x = foo[float]()` as `(gvar :x (f 64)
    // (conv (f 64) 123))` (an int→float const conversion), and `var pi = 3.14` as a bare float
    // literal. `collect_globals` only folded *integer* scalars, so any float module-level `var`/
    // `let` fail-closed ("non-scalar-int global initializer"). It now folds the three constant
    // float forms (bare literal, `(suf N "f32")`, `(conv (f T) int)`) into window bytes. `sumf`
    // reads `a` (via a conv-of-int init) plus `b` (a bare literal) back; both engines see 3.5.
    let leng = "\
(stmts
 (gvar :a.0. . (f +64) (conv (f +64) 1))
 (gvar :b.0. . (f +64) 2.5)
 (proc :sumf.0. . (i +64) .
  (stmts .
   (ret (conv (i +64) (add (f +64) a.0. b.0.))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 3, "1.0 + 2.5 = 3.5, truncated to 3");
}

#[test]
fn conv_wrapped_and_narrow_int_global_initializers() {
    // Two int-scalar global forms `collect_globals` previously fail-closed on:
    //  - a **conv-wrapped** int (`let x: distinct int = 4` → `(gvar :a (i 64) (conv (i 64) 4))`) —
    //    `int_literal` doesn't see through the `conv`, so it was "non-scalar-int global";
    //  - a **narrow** (`u8`) global (`var b: uint8 = 3` → `(gvar :b (u 8) (suf 3u "u8"))`) — the
    //    old branch matched only `Scalar`, so a `Narrow` type fell through to the error.
    // Both now fold via `const_scalar_int` + `int_store_width`. `sumi` reads them back: 4 + 3 = 7.
    let leng = "\
(stmts
 (gvar :a.0. . (i +64) (conv (i +64) 4))
 (gvar :b.0. . (u 8) (suf 3u \"u8\"))
 (proc :sumi.0. . (i +64) .
  (stmts .
   (ret (add (i +64) a.0. (conv (i +64) b.0.))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 7, "conv-wrapped 4 + narrow 3 = 7");
}

#[test]
fn aggregate_const_with_float_fields() {
    // An aggregate global/const with **float members** — `var c = Shape(x: 1.5, y: 2.0)` →
    // `(oconstr Shape (kv x 1.5) (kv y 2.0))`. `const_aggregate_bytes` folded int, string, and
    // pointer field values but not floats, so any object const with a float field fail-closed
    // (a placeholder, then a read fault). It now folds float field values to their width's bits.
    // `readY` reads the `y` member (offset 8) back; both engines see 2.0 (truncated to 2).
    let leng = "\
(stmts
 (type :Shape.0. . (object . (fld :x.0 . (f +64)) (fld :y.0 . (f +64))))
 (gvar :c.0. . Shape.0. (oconstr Shape.0. (kv x.0 1.5) (kv y.0 2.0)))
 (proc :readY.0. . (i +64) . (stmts . (ret (conv (i +64) (dot c.0. y.0 1))))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 2, "c.y = 2.0, truncated to 2");
}

#[test]
fn global_pointer_deref() {
    // A module-level **pointer global** (`var x: ptr Node` — nimony's `ref NodeObj`, lowered to a
    // pointer) is dereferenced: `roundtrip` builds a frame object, points the global at it, then reads
    // a field back through the global pointer. `pointer_operand` fail-closed on a global pointer
    // ("`x.0.` is not a known pointer" — it tracked only *local* pointers); it now falls through to
    // load the global's value. Real nimony emits these for every module-level `ref` var (binary
    // trees, linked lists — the #760 pointer-tracking gap).
    let leng = "\
(stmts
 (type :Node.0. . (object . (fld :data.0 . (i +64))))
 (gvar :x.0. . (ptr Node.0.) .)
 (proc :roundtrip.0. . (i +64) .
  (stmts .
   (var :o.0 . Node.0. (oconstr Node.0. (kv data.0 42)))
   (asgn x.0. (addr o.0))
   (ret (dot (deref x.0.) data.0 0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[4096]), 42, "(deref global x).data = 42");
}

#[test]
fn store_aggregate_through_lvalue() {
    // Assigning an aggregate **through a `dot`/`deref` lvalue** — `(*p) = Inner(v: 42)` — is a
    // whole-aggregate construct into the target address, not a scalar store. It previously
    // fail-closed ("assigning to aggregate lvalue"); real nimony assigns object/string fields this
    // way (`x.left = node`, string fields). `setit` writes the aggregate through the pointer and
    // reads a field back.
    let leng = "\
(stmts
 (type :Inner.0. . (object . (fld :v.0 . (i +64))))
 (proc :setit.0. (params (param :p.0 . (ptr Inner.0.))) (i +64) .
  (stmts .
   (asgn (deref p.0) (oconstr Inner.0. (kv v.0 42)))
   (ret (dot (deref p.0) v.0 0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // A frame is needed for the constructor temp, so slot 0 = $sp; p points into the window at 2048.
    assert_eq!(
        run(&m, 0, &[4096, 2048]),
        42,
        "(*p).v after aggregate store = 42"
    );
}

#[test]
fn read_aggregate_field_in_value_position() {
    // Reading an aggregate field **in value position** — `discard o.inner`, an object-typed field
    // evaluated as an expression — now yields the field's address (aggregates are by-address), rather
    // than fail-closing ("reading aggregate as a value"). Real nimony emits these when an object/
    // string field flows through an expression. The read translates and re-verifies; the scalar
    // `tag` field still reads normally alongside it.
    let leng = "\
(stmts
 (type :Inner.0. . (object . (fld :v.0 . (i +64))))
 (type :Outer.0. . (object . (fld :inner.0 . Inner.0.) (fld :tag.0 . (i +64))))
 (proc :f.0. (params (param :o.0 . Outer.0.)) (i +64) .
  (stmts .
   (discard (dot o.0 inner.0 0))
   (ret (dot o.0 tag.0 1)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
}

/// Real nimony `hexer` output for `var counter: int = 42` plus `getCounter`/`addCounter` — the
/// static non-zero initializer must seed the window so `getCounter()` reads 42.
#[test]
fn real_nimony_global_init() {
    const REAL: &str = include_str!("fixtures/real_global.leng.nif");
    let m = svm_leng::translate_procs(REAL, &["getCounter.0.", "addCounter.0."])
        .unwrap_or_else(|e| panic!("translate real getCounter/addCounter: {e}"));
    assert_eq!(run(&m, 0, &[]), 42, "counter initialized to 42");
    assert_eq!(run(&m, 1, &[8]), 50, "42 + 8");
    assert_eq!(run(&m, 1, &[-42]), 0, "42 - 42");
}
