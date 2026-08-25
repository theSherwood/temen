//! **The #964 guarded layout** (`TranslateOptions::null_guard`) — PR-1 slice: the *relocation*, not
//! yet the enforcement. A `--null-guard` translate shifts the powerbox low scratch (heap words,
//! format buffer, args blob) up by one `POWERBOX_NULL_GUARD` and marks the module with the
//! `__null_guard` function export; a marker-aware host (`RunConfig::init_mem` →
//! `temen_ir::module_args_base`) seeds the args blob where the shifted `_start` reads it. Nothing is
//! guarded yet — `[0, guard)` is merely *unused* on the new layout — so the two layouts must be
//! **behaviorally identical**: same stdout, same exit, for the same program and args. That parity is
//! this test. The flag-off path is pinned byte-identical by the entire existing suite (every other
//! test runs the legacy layout).

use std::path::{Path, PathBuf};
use std::process::Command;

use temen_run::run_powerbox_cfg;

/// A program exercising every relocated scratch consumer: argv/envp parsing (the args blob),
/// `getenv` (the independent blob reader), `malloc`/`realloc` (the heap words), and `printf` with
/// integer conversions (the format buffer).
const SRC: &str = r#"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
int main(int argc, char **argv) {
  int sum = 0;
  for (int i = 1; i < argc; i++) sum += (int)strlen(argv[i]);
  char *p = malloc(64);
  for (int i = 0; i < 64; i++) p[i] = (char)(i * 3);
  int acc = 0;
  for (int i = 0; i < 64; i++) acc += p[i];
  const char *home = getenv("GUEST_HOME");
  printf("argc=%d sum=%d acc=%d home=%s\n", argc, sum, acc, home ? home : "(none)");
  return argc == 4 ? 0 : 1;
}
"#;

fn compile_to_ll(name: &str) -> Option<PathBuf> {
    let dir = std::env::temp_dir();
    let c = dir.join(format!("temen_nullguard_{}_{}.c", std::process::id(), name));
    let ll = dir.join(format!(
        "temen_nullguard_{}_{}.ll",
        std::process::id(),
        name
    ));
    std::fs::write(&c, SRC).expect("write C source");
    let status = Command::new("clang")
        .args(["-O2", "-emit-llvm", "-S"])
        .arg(&c)
        .arg("-o")
        .arg(&ll)
        .status();
    match status {
        Ok(s) if s.success() => Some(ll),
        _ => {
            eprintln!("note: skipping {name} (clang unavailable)");
            None
        }
    }
}

fn translate(ll: &Path, null_guard: bool) -> temen_ir::Module {
    let opts = temen_llvm::TranslateOptions {
        null_guard,
        ..Default::default()
    };
    let t =
        temen_llvm::translate_ll_path_with_options(ll.to_str().unwrap(), opts).expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
}

/// The guarded layout is marked, reads its args one guard up, and behaves **identically** to the
/// legacy layout for the same program/args/env — argv, getenv, malloc, and printf all land on the
/// relocated scratch.
#[test]
fn guarded_layout_matches_legacy_end_to_end() {
    let Some(ll) = compile_to_ll("parity") else {
        return;
    };
    let legacy = translate(&ll, false);
    let guarded = translate(&ll, true);

    assert_eq!(
        temen_ir::module_null_guard(&legacy),
        None,
        "legacy: no marker"
    );
    assert_eq!(
        temen_ir::module_null_guard(&guarded),
        Some(temen_ir::POWERBOX_NULL_GUARD),
        "guarded: marker present"
    );
    assert_eq!(
        temen_ir::module_args_base(&guarded),
        temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_ARGS_BASE,
        "guarded args base is one guard up"
    );

    let args: [&[u8]; 4] = [b"prog", b"alpha", b"be", b"c"];
    let env: [&[u8]; 1] = [b"GUEST_HOME=/warm"];
    let a = run_powerbox_cfg(&legacy, b"", &args, &env, None, temen_run::Quota::default())
        .expect("legacy runs");
    let b = run_powerbox_cfg(
        &guarded,
        b"",
        &args,
        &env,
        None,
        temen_run::Quota::default(),
    )
    .expect("guarded runs");
    assert_eq!(
        String::from_utf8_lossy(&a.stdout),
        "argc=4 sum=8 acc=672 home=/warm\n"
    );
    assert_eq!(a.stdout, b.stdout, "byte-identical stdout across layouts");
    assert_eq!(
        format!("{:?}", a.outcome),
        format!("{:?}", b.outcome),
        "same outcome across layouts"
    );
}

/// The marker survives the wire: encode → decode keeps `__null_guard`, and `temen-strip`'s
/// `demote_exports` treats it as semantics (never demoted), like `_start`.
#[test]
fn marker_survives_wire_and_strip() {
    let Some(ll) = compile_to_ll("wire") else {
        return;
    };
    let guarded = translate(&ll, true);
    let mut rt =
        temen_encode::decode_module(&temen_encode::encode_module(&guarded)).expect("roundtrip");
    assert_eq!(
        temen_ir::module_null_guard(&rt),
        Some(temen_ir::POWERBOX_NULL_GUARD),
        "marker rides the .temen wire"
    );
    temen_run::demote_exports(&mut rt, &[]);
    assert_eq!(
        temen_ir::module_null_guard(&rt),
        Some(temen_ir::POWERBOX_NULL_GUARD),
        "temen-strip keeps the marker (semantics, not observability)"
    );
    assert_eq!(rt.resolve_export("_start"), Some(0), "_start kept too");
}
