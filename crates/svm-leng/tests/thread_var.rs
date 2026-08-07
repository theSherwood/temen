//! Thread-var (`tvar`) lowering — the single-threaded TLS model (NIM.md §3d).
//!
//! Leng marks a thread-local `tvar` (nimony's `__thread`; the allocator and exception state in the
//! real `system` module are thread-vars). svm-leng lowers a `tvar` **identically to a `gvar`**: one
//! plain, zero-initialized global at a fixed window offset. This is sound because every guest we
//! target — each nimony compiler phase, each svm domain — runs single-threaded, so a thread-local has
//! exactly one instance and a plain global *is* that instance. It mirrors the C on-ramp, which strips
//! `__thread` before clang (`demos/nimony/build_nimony.sh`). These tests pin that model: a `tvar`
//! must behave as a persistent global — writes survive across calls, non-zero initializers seed it,
//! and it links across modules the same as a `gvar` — on both engines. (A real multi-threaded
//! `__thread` lowering over `vcpu.tls` is NIM.md §3d Tier 2, deferred.)

use svm_interp::Value;
use svm_leng::LengModule;

/// Run func `idx` on both engines; assert §9 parity; return the i64 result.
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

/// A `tvar` is real, persistent backing store: `bump` writes it, later calls read the accumulated
/// value back. Same shape as `whole_module::globals_const_and_intramodule_calls`, but the counter is
/// a **thread-var** — so this proves `tvar` lowers to a plain global that persists across calls.
#[test]
fn thread_var_persists_across_calls_like_a_global() {
    let leng = "\
(stmts
 (tvar :counter.0. . (i +64) .)
 (proc :bump.0. . (void) .
  (stmts .
   (asgn counter.0. (add (i +64) counter.0. 1))))
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
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.starts_with("memory "),
        "a tvar needs a window:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    // main is func 2; the thread-var counter starts 0 and is bumped n times → returns n.
    assert_eq!(run(&m, 2, &[0]), 0);
    assert_eq!(run(&m, 2, &[7]), 7);
    assert_eq!(run(&m, 2, &[100]), 100);
}

/// A `tvar` with a non-zero initializer seeds the window like a `gvar` initializer does.
#[test]
fn thread_var_nonzero_initializer_seeds_the_window() {
    let leng = "\
(stmts
 (tvar :g.0. . (i +64) 5)
 (tvar :h.0. . (i +32) 7)
 (proc :sum.0. . (i +64) . (stmts . (ret (add (i +64) g.0. (conv (i +64) h.0.))))))";
    let text = svm_leng::translate_to_text(leng).unwrap();
    assert!(
        text.contains("data "),
        "a non-zero tvar init emits a data segment:\n{text}"
    );
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(
        run(&m, 0, &[]),
        12,
        "5 + 7 read from the seeded thread-vars"
    );
}

/// A `tvar` links across modules exactly like a `gvar`: module `w`'s proc reads and writes a
/// thread-var `g` **defined in module `s`** (a `data.sym` the linker binds to `s`'s exported global).
/// This is the shape of the real allocator's thread-vars in `system`, referenced from user code —
/// write-then-read-back through the cross-module symbol must reach `s`'s storage, so `rw(v) = v`.
/// (Mirrors `link::cross_module_global_read_write`, with `g` a `tvar`.)
#[test]
fn thread_var_links_cross_module_like_a_global() {
    let mod_s = "\
(stmts
 (tvar :g.0. . (i +64) 0))";
    let mod_w = "\
(stmts
 (proc :rw.0. (params (param :v.0 . (i +64))) (i +64) .
  (stmts .
   (asgn g.0.mods v.0)
   (ret g.0.mods))))";
    let linked = svm_leng::link_units(&[
        LengModule {
            stem: "modw",
            src: mod_w,
            names: &["rw.0."],
        },
        LengModule {
            stem: "mods",
            src: mod_s,
            names: &[], // data-only unit: it just defines (and exports) the thread-var `g`
        },
    ])
    .unwrap_or_else(|e| panic!("link: {e}"));
    assert_eq!(
        run(&linked, 0, &[42]),
        42,
        "write then read the external tvar"
    );
    assert_eq!(run(&linked, 0, &[-5]), -5);
}
