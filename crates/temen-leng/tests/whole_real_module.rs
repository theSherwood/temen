//! The capstone (NIM.md Phase 2): translate an **entire** real nimony module — every proc, plus
//! its globals, type declarations, and cross-module imports — from raw `hexer` output, and check it
//! **verifies**. Then run the user `main` (which only makes an intra-module call, no unbound
//! imports) end-to-end on both engines.
//!
//! These three fixtures are verbatim `hexer c --isMain` output for real Nim programs — `real_module`
//! (`addTwo(a,b)=a+b` + `main`, 4 funcs), `real_controlflow` (`maxi` if/else + `sumto` while/`inc`,
//! 5 funcs), and `real_aggregates` (`dot2` object fields + `idx` array `at`, 4 funcs). Each carries
//! the full module scaffolding nimony emits: `gvar`s (`cmdCount`, `iniGuard`, …), a `` `cchar ``
//! type, and the `` `ini ``/`` `main `` C entry procs that call the stdlib via `…sysvq0asl` imports.

fn verify_whole(src: &str) -> temen_ir::Module {
    let text = temen_leng::translate_to_text(src)
        .unwrap_or_else(|e| panic!("translate whole module: {e}"));
    let m = temen_text::parse_module(&text)
        .unwrap_or_else(|e| panic!("parse emitted IR: {e:?}\n{text}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}\n{text}"));
    m
}

#[test]
fn whole_addtwo_module_verifies() {
    let m = verify_whole(include_str!("fixtures/real_module.leng.nif"));
    // addTwo, user main, `ini`, C `main`.
    assert_eq!(m.funcs.len(), 4);
}

#[test]
fn whole_controlflow_module_verifies() {
    // maxi, sumto, `ini`, C `main`, + a compiler helper.
    let m = verify_whole(include_str!("fixtures/real_controlflow.leng.nif"));
    assert_eq!(m.funcs.len(), 5);
}

#[test]
fn whole_aggregates_module_verifies() {
    // dot2, idx, `ini`, C `main`.
    let m = verify_whole(include_str!("fixtures/real_aggregates.leng.nif"));
    assert_eq!(m.funcs.len(), 4);
}

/// Run the **user `main`** of the real addTwo module end-to-end: it calls `addTwo(20, 22)` (an
/// intra-module call — no unbound imports on this path), discards the result, and returns. Both
/// engines must complete cleanly (the `ini`/C-`main` procs, which call stdlib imports, aren't
/// entered here).
#[test]
fn whole_addtwo_user_main_runs() {
    let m = verify_whole(include_str!("fixtures/real_module.leng.nif"));
    // func 1 is the user `main` (func 0 is addTwo); it takes no params and returns void.
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(&m, 1, &[], &mut fuel).expect("interp run of user main");
    let jit = match temen_jit::compile_and_run(&m, 1, &[]).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit: {other:?}"),
    };
    // Void proc → both return no values, cleanly (the point is that the whole module runs).
    assert!(interp.is_empty(), "user main is void: {interp:?}");
    assert!(jit.is_empty(), "user main is void: {jit:?}");
}
