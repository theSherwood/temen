//! Whole-module linking (NIM.md W2, **Path A**). A real program links *whole compiled modules* —
//! every proc a module defines, plus the scaffolding nimony emits around them: an `ini`
//! module-initializer (a run-once guard that chains sub-module inits), and, in the main module, a C
//! `main` and the exportc gvars. `link_whole_units` compiles each module in full to a `.temeno` object
//! (all procs exported under their global names) and links them — the analog of `lld` over object
//! files, versus `link_units`' "lift a named subset" go-deep mode. Both engines, §9 parity.

use temen_interp::Value;
use temen_leng::WholeModule;

/// A stand-in `system` module (stem `sysvq0asl`, the real one nimony assigns `std/system`): whole
/// modules chain their `ini` to the system `ini` and flush via `nimFlushStdStreams`, both compiled
/// Nim in a real build. Here they're no-op void procs — enough to *resolve the link* and run code
/// that doesn't depend on their effects (the real compiled `system` object replaces this, W2/Path A).
const SYSTEM_STUB: &str = "\
(stmts
 (proc :`ini.0. . . . (stmts . (ret .)))
 (proc :nimFlushStdStreams.0. . . . (stmts . (ret .))))";

/// Run func `idx` of a linked module on both engines; assert agreement; return the i64 result.
fn run(m: &temen_ir::Module, idx: u32, args: &[i64]) -> i64 {
    temen_verify::verify_module(m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = temen_interp::run(m, idx, &ivals, &mut fuel).expect("interp");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match temen_jit::compile_and_run(m, idx, args).expect("jit") {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

/// Two whole modules with the shape nimony emits: each defines a real proc *and* an `ini` (the
/// run-once guard — `if iniGuard: return; iniGuard = true; <chain sub-inits>`). Linking them in
/// full brings **every** proc (not a hand-picked subset): `mod_a`'s cross-module call to `mod_b`'s
/// `helper` resolves, and `mod_a`'s `ini` chains to `mod_b`'s `ini`. `useit(5) = helper(5)+1 = 16`.
#[test]
fn whole_modules_link_with_ini_scaffolding() {
    // Leaf module: a real proc + a self-contained `ini` (guard only, no sub-inits to chain).
    let mod_b = "\
(stmts
 (proc :helper.0. (params (param :x.0 . (i +64))) (i +64) .
  (stmts . (ret (mul (i +64) x.0 3))))
 (gvar :iniGuard.0. . (bool) .)
 (proc :ini.0. . . .
  (stmts .
   (if (elif iniGuard.0. (stmts . (ret .))))
   (asgn iniGuard.0. (true)))))";
    // Main-ish module: a real proc that calls across, and an `ini` that chains to mod_b's `ini`.
    let mod_a = "\
(stmts
 (proc :useit.0. (params (param :n.0 . (i +64))) (i +64) .
  (stmts . (ret (add (i +64) (call helper.0.modb n.0) 1))))
 (gvar :iniGuard.0. . (bool) .)
 (proc :ini.0. . . .
  (stmts .
   (if (elif iniGuard.0. (stmts . (ret .))))
   (asgn iniGuard.0. (true))
   (call ini.0.modb))))";
    let linked = temen_leng::link_whole_units(&[
        WholeModule {
            stem: "moda",
            src: mod_a,
        },
        WholeModule {
            stem: "modb",
            src: mod_b,
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // Linked func order: [useit, mod_a.ini, helper, mod_b.ini].
    assert_eq!(run(&linked, 0, &[5]), 16, "useit(5) = helper(5)+1");
    assert_eq!(run(&linked, 0, &[10]), 31);
    // The `ini` chain runs: mod_a.ini (func 1) sets its guard and calls mod_b.ini — void, and each
    // module reads its *own* (relocated) guard, so both engines complete cleanly.
    let ivals = [];
    let mut fuel = u64::MAX;
    temen_verify::verify_module(&linked).unwrap();
    let interp = temen_interp::run(&linked, 1, &ivals, &mut fuel).expect("interp ini");
    assert!(interp.is_empty(), "ini is void: {interp:?}");
    match temen_jit::compile_and_run(&linked, 1, &[]).expect("jit ini") {
        temen_jit::JitOutcome::Returned(v) => assert!(v.is_empty(), "ini is void: {v:?}"),
        o => panic!("jit ini: {o:?}"),
    }
}

/// The real thing: **two genuine `hexer` modules linked in full** (Path A). `moda` (the main module:
/// `useit`, its `ini`, and the C `main` + exportc gvars) and `modb` (`helper` + `ini`) each compile
/// as a whole `.temeno` — no hand-picked procs. Their cross-module edges resolve against each other
/// (`moda`'s `useit` → `modb`'s `helper`; `moda`'s `ini` → `modb`'s `ini`), and the remaining
/// `sysvq0asl` system edges (`ini`, `nimFlushStdStreams`) resolve against the [`SYSTEM_STUB`]. The
/// whole program links, verifies, and `useit(5) = helper(5)+1 = 16` runs on both engines.
#[test]
fn real_two_modules_link_in_full() {
    const MODA: &str = include_str!("fixtures/real_moda.leng.nif");
    const MODB: &str = include_str!("fixtures/real_modb.leng.nif");
    let linked = temen_leng::link_whole_units(&[
        WholeModule {
            stem: "modywjwgs",
            src: MODA,
        },
        WholeModule {
            stem: "modwru7vt1", // the stem moda's `helper.0.<stem>` / `ini.0.<stem>` calls carry
            src: MODB,
        },
        WholeModule {
            stem: "sysvq0asl",
            src: SYSTEM_STUB,
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    // moda's first proc is `useit` → func 0.
    assert_eq!(run(&linked, 0, &[5]), 16, "useit(5) through the full link");
    assert_eq!(run(&linked, 0, &[10]), 31);

    // Capstone: run moda's real C `main` (func 2: (argc, argv, envp) -> i32) end-to-end. It stores
    // argc/argv/envp into the exportc gvars, drives the **whole init chain** (moda.ini → the
    // sysvq0asl `ini` stub + modb.ini, each guarding its own relocated `iniGuard`), calls the flush
    // stub, and returns 0 — identically on both engines. The module scaffolding *executes*, not just
    // links.
    let mut fuel = u64::MAX;
    let iargs = [Value::I32(0), Value::I64(0), Value::I64(0)];
    let interp = temen_interp::run(&linked, 2, &iargs, &mut fuel).expect("interp C main");
    let ret = match interp.as_slice() {
        [Value::I32(n)] => *n as i64,
        [Value::I64(n)] => *n,
        o => panic!("C main return: {o:?}"),
    };
    assert_eq!(ret, 0, "C main returns 0");
    let jit = match temen_jit::compile_and_run(&linked, 2, &[0, 0, 0]).expect("jit C main") {
        temen_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit C main: {o:?}"),
    };
    assert_eq!(jit, vec![0], "§9 interp/JIT parity on C main");
}
