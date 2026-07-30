//! String-literal tests (NIM.md Phase 2 / W2). nimony's `string` is a small-string-optimized object
//! `{bytes: u64@0, more: ptr@8}` (confirmed from the system module's `basic_types.nim`): a **short**
//! literal packs its chars into the inline `bytes` word with a nil `more`, so it's an ordinary
//! `(oconstr string (kv bytes.0 <packed-u64>) (kv more.0 (nil)))` — no data segment at all.
//!
//! Two pieces make that lower: unsigned literals (`122511465736197u`) parse, and the external
//! `string` type's layout is available. `link_units` resolves the layout automatically from the
//! defining unit (see `link.rs::real_string_type_resolves_across_link`); this file exercises the
//! *manual* escape hatch, `translate_proc_with_types`, which supplies it as a prelude — here, the
//! *real* `string` def.

use svm_interp::Value;

fn run(module: &svm_ir::Module, idx: u32, args: &[i64]) -> i64 {
    svm_verify::verify_module(module).unwrap_or_else(|e| panic!("verify: {e:?}"));
    let ivals: Vec<Value> = args.iter().map(|&n| Value::I64(n)).collect();
    let mut fuel = u64::MAX;
    let interp = svm_interp::run(module, idx, &ivals, &mut fuel).expect("interp run");
    let n = match interp.as_slice() {
        [Value::I64(n)] => *n,
        o => panic!("expected i64, got {o:?}"),
    };
    let jit = match svm_jit::compile_and_run(module, idx, args).expect("jit") {
        svm_jit::JitOutcome::Returned(v) => v,
        o => panic!("jit: {o:?}"),
    };
    assert_eq!(jit.as_slice(), &[n], "§9 interp/JIT parity");
    n
}

#[test]
fn sso_string_construct_and_read() {
    // Build an SSO string exactly as nimony does — pack "hello" (0x6f6c6c6568 = 478560413032) into
    // `bytes`, nil `more` — then read the `bytes` word back. Runs (no runtime: it's a plain object).
    let leng = "\
(stmts
 (type :Str.0. . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64)))))
 (proc :mk.0 . (i +64) .
  (stmts .
   (var :s.0 . Str.0. (oconstr Str.0. (kv bytes.0 478560413032u) (kv more.0 (nil))))
   (ret (dot s.0 bytes.0 0)))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    let sp = 4096;
    assert_eq!(
        run(&m, 0, &[sp]),
        478560413032,
        "packed \"hello\" bytes read back"
    );
}

#[test]
fn unsigned_literal_over_i64_max() {
    // An SSO word can exceed i64::MAX; the `u` literal keeps the bit pattern.
    let leng = "\
(stmts
 (proc :big.0 . (i +64) . (stmts . (ret 18000000000000000000u))))";
    let m = svm_leng::translate(leng).unwrap_or_else(|e| panic!("translate: {e}"));
    assert_eq!(run(&m, 0, &[]), 18000000000000000000u64 as i64);
}

/// Real nimony `hexer` output for `greet(): string = "hello"` — the genuine SSO oconstr
/// `(oconstr string (kv bytes.0 122511465736197u) (kv more.0 (nil)))` returned by sret. Supplying
/// the real `string` layout (its `system`-module def, prepended), it **translates and verifies**;
/// running it end-to-end needs the ARC `=wasMoved`/`=destroy` imports bound (the W3 runtime edge).
#[test]
fn real_nimony_short_string() {
    const REAL: &str = include_str!("fixtures/real_string.leng.nif");
    // The `string` type as the system module defines it (bytes: u64, more: ptr), under the global
    // name the module references.
    let string_ty =
        "(stmts (type :string.0.sysvq0asl . (object . (fld :bytes.0 . (u 64)) (fld :more.0 . (ptr (i +64))))))";
    let m = svm_leng::translate_proc_with_types(REAL, "greet.0.", string_ty)
        .unwrap_or_else(|e| panic!("translate real greet: {e}"));
    svm_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify greet: {e:?}"));
}
