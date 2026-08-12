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
