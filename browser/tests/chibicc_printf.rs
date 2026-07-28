//! The playground C-compiler card, end to end in Rust (SELFHOST_C.md §7): compile a C program that
//! `#include <stdio.h>` and `printf`s — using the built-in playground headers — then run the emitted
//! module and assert its stdout. This is the "text-emitting programs" follow-up: a `printf` used to
//! compile to an unresolved `call.sym "printf"` and trap; now `<stdio.h>` carries a guest-C `printf`
//! that formats over the powerbox's ambient `write`, so the program actually prints.
//!
//! Fail-soft: `chibicc.svmb` is a code-coupled asset (`browser/web/assets/`) that CI regenerates. If
//! it is absent (a fresh tree without the build), the test SKIPs rather than failing — the Lua/Doom
//! pattern.

use svm_browser::{onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn chibicc_svmb() -> Option<Vec<u8>> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.svmb");
    std::fs::read(p).ok()
}

/// Compile `src` with the seeded playground headers, run the result, return its captured stdout.
fn compile_and_run(chibicc: &svm_ir::Module, src: &str) -> (i32, String) {
    // The memfs the card seeds: built-in headers under `include/` + the user's source at `in.c`.
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    let image = svm_fs::encode_image(&files, &dirs);

    // Pass 1 — chibicc.svmb emits SVM-IR text on stdout. `--data-page 65536` mirrors the browser card
    // (D40 isolation at the 64 KiB wasm host page), so this exercises exactly the shipped path.
    let compiled = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {}",
        compiled.status
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    assert!(ir.contains("func"), "expected SVM IR, got: {ir:.200}");

    // Pass 2 — parse the IR into a module and run it under the powerbox.
    let m = svm_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}\n{ir}"));
    let run = onramp_exec(&m, b"");
    (
        run.status,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    )
}

#[test]
fn printf_program_prints_through_the_powerbox() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: browser/web/assets/chibicc.svmb absent (run build-onramp-assets.mjs)");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode chibicc.svmb");

    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
int main(void) {
  printf("hello, %s! %d + %d = %d\n", "playground", 2, 40, 42);
  for (int i = 1; i <= 3; i++) printf("line %d\n", i);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "hello, playground! 2 + 40 = 42\nline 1\nline 2\nline 3\n"
    );
}

#[test]
fn string_and_stdlib_headers_resolve_and_run() {
    let Some(bytes) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let chibicc = svm_encode::decode_module(&bytes).expect("decode chibicc.svmb");

    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(void) {
  char *p = malloc(32);
  strcpy(p, "chibicc");
  printf("%s has %lu letters; atoi(\"41\")+1=%d\n", p, (unsigned long)strlen(p), atoi("41") + 1);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(out, "chibicc has 7 letters; atoi(\"41\")+1=42\n");
}
