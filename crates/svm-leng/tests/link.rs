//! Cross-module linking tests (NIM.md W2 — the linker). A real program is many modules; nimony emits
//! one Leng file per module and references a proc `P.` *defined* in module `stem` from *other*
//! modules as `P.<stem>`. `svm_leng::link_units` translates each module's procs, exports them under
//! those global names, and resolves every unit's cross-module calls (named imports) against the
//! exports via `svm_ir::link` — one merged, import-free svm module. Both engines, §9 parity.

use svm_interp::Value;
use svm_leng::LengModule;

/// Run func `idx` of a linked module on both engines; assert agreement; return the i64 result.
fn run(m: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(m, idx, &ivals, &mut fuel).expect("interp");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match svm_jit::compile_and_run(m, idx, args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

/// Real nimony: `moda` imports `pkg/modb` and calls `helper`. `hexer` emits `moda`'s call as
/// `helper.0.<modb-stem>`; `modb` defines `helper.0.` locally. Linking the two translated modules
/// resolves the cross-module call to compiled Nim, and `useit(5) = helper(5) + 1 = 16`.
#[test]
fn real_two_module_program() {
    const MODA: &str = include_str!("fixtures/real_moda.leng.nif");
    const MODB: &str = include_str!("fixtures/real_modb.leng.nif");
    // modb's file stem — the suffix moda's cross-module call to `helper` carries.
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modywjwgs",
            src: MODA,
            names: &["useit.0."],
        },
        LengModule {
            stem: "modwru7vt1",
            src: MODB,
            names: &["helper.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // useit is module A's first proc → func 0.
    assert_eq!(run(&linked, 0, &[5]), 16);
    assert_eq!(run(&linked, 0, &[10]), 31);
}

/// A transitive chain A → B → C across three (hand-written) modules: `top` calls `mid.<B>`, `mid`
/// calls `leaf.<C>`. The linker composes them into one module; `top(n) = leaf(n)*2 + 10 + 1`.
#[test]
fn transitive_three_module_chain() {
    let mod_a = "\
(stmts
 (proc :top.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call mid.0.modb n.0) 1)))))";
    let mod_b = "\
(stmts
 (proc :mid.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (add (i +64) (call leaf.0.modc n.0) 10)))))";
    let mod_c = "\
(stmts
 (proc :leaf.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts .
   (ret (mul (i +64) n.0 2)))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "moda",
            src: mod_a,
            names: &["top.0."],
        },
        LengModule {
            stem: "modb",
            src: mod_b,
            names: &["mid.0."],
        },
        LengModule {
            stem: "modc",
            src: mod_c,
            names: &["leaf.0."],
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // top(n) = (leaf(n)*... ) : leaf(n)=2n; mid=2n+10; top=2n+11.
    assert_eq!(run(&linked, 0, &[4]), 19, "2*4 + 10 + 1");
    assert_eq!(run(&linked, 0, &[0]), 11);
}

#[test]
fn unresolved_cross_module_call_is_fail_closed() {
    // `top` calls `missing.0.modx`, which no linked unit exports → a clean link error.
    let mod_a = "\
(stmts
 (proc :top.0. . (i +64) . (stmts . (ret (call missing.0.modx)))))";
    match svm_leng::link_units(&[LengModule {
        stem: "moda",
        src: mod_a,
        names: &["top.0."],
    }]) {
        Err(svm_leng::LengError::Malformed(_)) => {}
        other => panic!("expected a fail-closed link error, got {other:?}"),
    }
}
