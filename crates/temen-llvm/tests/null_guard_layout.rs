//! **The #964/#1094 guarded layout** — the powerbox low scratch (heap words, format buffer, args
//! blob) lives one `POWERBOX_NULL_GUARD` up, so `[0, guard)` is empty and a host seeds it `Unmapped`
//! (the guard is **unconditional** now — #1094, the one canonical layout; the `__null_guard` marker
//! export is retired). A host reads the args blob one guard up (`temen_ir::module_args_base` →
//! `RunConfig::init_mem`), where the shifted `_start` reads it. This test pins that the guarded layout
//! runs a scratch-heavy program correctly end to end, and that it survives the wire + `temen-strip`.

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

fn translate(ll: &Path) -> temen_ir::Module {
    let t = temen_llvm::translate_ll_path_with_options(
        ll.to_str().unwrap(),
        temen_llvm::TranslateOptions::default(),
    )
    .expect("translate");
    temen_verify::verify_module(&t.module).expect("verify");
    t.module
}

/// The guarded layout reads its args one guard up and runs a scratch-heavy program correctly —
/// argv, getenv, malloc, and printf all land on the relocated scratch.
#[test]
fn guarded_layout_runs_end_to_end() {
    let Some(ll) = compile_to_ll("parity") else {
        return;
    };
    let guarded = translate(&ll);

    assert_eq!(
        temen_ir::module_null_guard(),
        temen_ir::POWERBOX_NULL_GUARD,
        "the guard is unconditional (#1094)"
    );
    assert_eq!(
        temen_ir::module_args_base(),
        temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_ARGS_BASE,
        "guarded args base is one guard up"
    );

    let args: [&[u8]; 4] = [b"prog", b"alpha", b"be", b"c"];
    let env: [&[u8]; 1] = [b"GUEST_HOME=/warm"];
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
        String::from_utf8_lossy(&b.stdout),
        "argc=4 sum=8 acc=672 home=/warm\n"
    );
}

/// The guarded layout survives the wire: encode → decode → `temen-strip`'s `demote_exports` keeps
/// `_start` and the module still runs under the guard. (#1094: no `__null_guard` marker export is
/// emitted any more, so none should appear.)
#[test]
fn guarded_layout_survives_wire_and_strip() {
    let Some(ll) = compile_to_ll("wire") else {
        return;
    };
    let guarded = translate(&ll);
    let mut rt =
        temen_encode::decode_module(&temen_encode::encode_module(&guarded)).expect("roundtrip");
    assert_eq!(
        temen_ir::module_null_guard(),
        temen_ir::POWERBOX_NULL_GUARD,
        "runs under the unconditional guard after the wire roundtrip"
    );
    temen_run::demote_exports(&mut rt, &[]);
    assert_eq!(rt.resolve_export("_start"), Some(0), "_start kept by strip");
    assert_eq!(
        rt.resolve_export("__null_guard"),
        None,
        "the retired marker export is not emitted (#1094)"
    );
}
