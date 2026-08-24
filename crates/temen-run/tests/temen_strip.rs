//! Tests for [`temen_run::demote_exports`] — the `temen-strip` packaging step (#907): the
//! `.dynsym`/`.symtab` split. Demoting a non-entry-point export from the semantic exports table into
//! the strippable `debug_info.func_names` diagnostic table must be **inert**: byte-identical run
//! output, the module still verifies, the demoted name survives an encode/decode round-trip (it
//! ships), and only what [`Module::resolve_export`](temen_ir::Module::resolve_export) can reach
//! changes. `_start` is always kept — it is the powerbox-entry *marker*
//! ([`temen_run::is_named_powerbox_entry`]), semantics, not observability.

use temen_ir::{FuncName, Module};
use temen_run::{demote_exports, is_named_powerbox_entry, run_powerbox};
use temen_text::parse_module;
use temen_verify::verify_module;

/// `_start` (func 0) calls `emit` (func 1), which writes a constant `data` segment to stdout — two
/// defined functions, both exported by symbol name, translate-style.
fn two_func_module() -> Module {
    let m = parse_module(
        "memory 16\n\
         data 16 \"hi\\n\"\n\
         export 0 func \"_start\" 0\n\
         export 1 func \"emit\" 1\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = call 1 ()\n\
         \x20 return v0\n\
           }\n\
         }\n\
         func () -> (i32) {\n\
         block 0 () {\n\
         \x20 v0 = i32.const 0\n\
         \x20 v1 = i64.const 16\n\
         \x20 v2 = i64.const 3\n\
         \x20 v3 = call.sym \"write\" (i64, i64) -> (i64) v0(v1, v2)\n\
         \x20 v4 = i32.const 7\n\
         \x20 return v4\n\
           }\n\
         }\n",
    )
    .expect("parse text IR");
    verify_module(&m).expect("verify");
    m
}

/// The core guarantee: demotion is **inert** — same stdout, same outcome, still verifies — while the
/// semantic surface genuinely shrinks (`resolve_export` no longer reaches the demoted name).
#[test]
fn demotion_is_inert_and_slims_resolve_surface() {
    let full = two_func_module();
    let before = run_powerbox(&full, b"").expect("run full");

    let mut slim = full.clone();
    let n = demote_exports(&mut slim, &[]);
    assert_eq!(n, 1, "one non-_start export demoted");
    verify_module(&slim).expect("demoted module still verifies");
    assert!(is_named_powerbox_entry(&slim), "_start marker survives");
    assert_eq!(slim.resolve_export("_start"), Some(0));
    assert_eq!(
        slim.resolve_export("emit"),
        None,
        "demoted name is no longer semantic"
    );
    let di = slim.debug_info.as_ref().expect("diagnostic table created");
    assert_eq!(
        di.func_names,
        vec![FuncName {
            func: 1,
            name: "emit".into()
        }],
        "the demoted name lands in func_names"
    );

    let after = run_powerbox(&slim, b"").expect("run demoted");
    assert_eq!(after.stdout, before.stdout, "byte-identical stdout");
    assert_eq!(after.stdout, b"hi\n");
    assert_eq!(
        format!("{:?}", after.outcome),
        format!("{:?}", before.outcome)
    );
}

/// The demoted name **ships**: it survives `encode_module` → `decode_module`, so the wasm-JIT
/// emitter's name section and interp backtraces still see `emit` in the shipped artifact.
#[test]
fn demoted_name_survives_encode_roundtrip() {
    let mut m = two_func_module();
    demote_exports(&mut m, &[]);
    let rt = temen_encode::decode_module(&temen_encode::encode_module(&m)).expect("roundtrip");
    assert_eq!(rt.exports.len(), 1);
    assert_eq!(
        rt.debug_info.expect("debug_info shipped").func_names,
        vec![FuncName {
            func: 1,
            name: "emit".into()
        }]
    );
}

/// A `--keep` name stays semantic; a pre-existing `func_names` entry (a `-g` source name) is never
/// overwritten by a demoted symbol name — source names win, same precedence as the emitter's.
#[test]
fn keep_list_and_source_name_precedence() {
    let mut m = two_func_module();
    assert_eq!(
        demote_exports(&mut m, &["emit"]),
        0,
        "kept ⇒ nothing demoted"
    );
    assert_eq!(m.resolve_export("emit"), Some(1));
    assert!(m.debug_info.is_none(), "nothing demoted ⇒ no table created");

    let mut m = two_func_module();
    m.debug_info = Some(temen_ir::DebugInfo {
        func_names: vec![FuncName {
            func: 1,
            name: "emit_src".into(),
        }],
        ..Default::default()
    });
    demote_exports(&mut m, &[]);
    assert_eq!(
        m.debug_info.unwrap().func_names,
        vec![FuncName {
            func: 1,
            name: "emit_src".into()
        }],
        "existing source name is not overwritten"
    );
}
