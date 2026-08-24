//! The playground C-compiler card, end to end in Rust (SELFHOST_C.md §7): compile a C program that
//! `#include <stdio.h>` and `printf`s — using the built-in playground headers — then run the emitted
//! module and assert its stdout. This is the "text-emitting programs" follow-up: a `printf` used to
//! compile to an unresolved `call.sym "printf"` and trap; now `<stdio.h>` carries a guest-C `printf`
//! that formats over the powerbox's ambient `write`, so the program actually prints.
//!
//! Fail-soft: `chibicc.temen` is a code-coupled asset (`browser/web/assets/`) that CI regenerates. If
//! it is absent (a fresh tree without the build), the test SKIPs rather than failing — the Lua/Doom
//! pattern.

use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK,
};

fn chibicc_temen() -> Option<Vec<u8>> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    std::fs::read(p).ok()
}

/// Compile `src` with the seeded playground headers, run the result, return its captured stdout.
fn compile_and_run(chibicc: &temen_ir::Module, src: &str) -> (i32, String) {
    // The memfs the card seeds: built-in headers under `include/` + the user's source at `in.c`.
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);

    // Pass 1 — chibicc.temen emits TEMEN-IR text on stdout. `--data-page 65536` mirrors the browser card
    // (D40 isolation at the 64 KiB wasm host page), so this exercises exactly the shipped path.
    let compiled = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"-g0", b"/in.c"],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {}",
        compiled.status
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    assert!(ir.contains("func"), "expected Temen IR, got: {ir:.200}");

    // Pass 2 — parse the IR into a module and run it under the powerbox.
    let m = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}\n{ir}"));
    let run = onramp_exec(&m, b"");
    (
        run.status,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    )
}

#[test]
fn printf_program_prints_through_the_powerbox() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: browser/web/assets/chibicc.temen absent (run build-onramp-assets.mjs)");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");

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
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");

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

#[test]
fn float_formatting_matches_glibc_for_typical_values() {
    let Some(bytes) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let chibicc = temen_encode::decode_module(&bytes).expect("decode chibicc.temen");

    // The seeded <stdio.h> formats %f/%e/%g in guest C, correctly rounded to the requested precision.
    // These are all cases where that matches glibc byte-for-byte (the bignum-only ties — an exact 0.5
    // at %.0f, or 0.015 at %.2f — are deliberately not asserted; see the header's float helpers).
    let (status, out) = compile_and_run(
        &chibicc,
        r#"
#include <stdio.h>
int main(void) {
  printf("%f|%.2f|%.0f|%+.1f\n", 3.14159, 2.71828, 2.7, 42.0);
  printf("[%8.2f][%-8.2f][%08.2f]\n", 3.14, 3.14, 3.14);
  printf("%e|%.2e|%E\n", 12345.678, 0.000123, 6.022e23);
  printf("%g|%g|%g|%g\n", 3.14159, 1000000.0, 0.0001, 0.00001);
  return 0;
}
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(
        out,
        "3.141590|2.72|3|+42.0\n\
         [    3.14][3.14    ][00003.14]\n\
         1.234568e+04|1.23e-04|6.022000E+23\n\
         3.14159|1e+06|0.0001|1e-05\n"
    );
}
