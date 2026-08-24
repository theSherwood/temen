//! Multi-file guest compile for the playground C compiler (SELFHOST_C.md §7, the stage-2 lever). The
//! card's editor can hold a small **multi-file project**: `//// file: NAME` marker lines split the
//! buffer into memfs files, and the entry (`/in.c`) `#include`s the sibling `.h`/`.c` files —
//! chibicc-the-guest resolves quote-includes against the source's own directory (`/`), so a program
//! split across a header + a second translation unit compiles (unity-build style) and runs, entirely
//! in-sandbox. These tests exercise the split (`split_multifile_source`) and the end-to-end compile.
//! Fail-soft: SKIPs if `chibicc.temen` isn't built.

use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, split_multifile_source, STATUS_EXIT,
    STATUS_OK,
};

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// Assemble the card's memfs exactly as `chibicc_card_image` does — built-in headers under `/include`
/// plus the multi-file split of the editor buffer — then compile `/in.c` and run the result.
fn compile_and_run(chibicc: &temen_ir::Module, editor: &str) -> (i32, String) {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    let mut dirs = vec!["include".to_string()];
    for (key, bytes) in split_multifile_source(editor.as_bytes()) {
        if let Some((dir, _)) = key.rsplit_once('/') {
            if !dirs.iter().any(|d| d == dir) {
                dirs.push(dir.to_string());
            }
        }
        files.retain(|(k, _)| *k != key);
        files.push((key, bytes));
    }
    let image = temen_fs::encode_image(&files, &dirs);
    let compiled = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "compile status {} — stderr: {}",
        compiled.status,
        String::from_utf8_lossy(&compiled.stderr)
    );
    let ir = String::from_utf8(compiled.stdout).expect("IR is utf8");
    assert!(ir.contains("func"), "expected Temen IR, got: {ir:.200}");
    let m = temen_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    let run = onramp_exec(&m, b"");
    (
        run.status,
        String::from_utf8_lossy(&run.stdout).into_owned(),
    )
}

/// The pure splitter: no marker ⇒ one `in.c`; markers seed siblings (with nesting + leading-`/` trim).
#[test]
fn split_marks_files() {
    let one = split_multifile_source(b"int main(void){return 0;}\n");
    assert_eq!(one.len(), 1);
    assert_eq!(one[0].0, "in.c");

    let many = split_multifile_source(
        b"#include \"u.h\"\nint main(void){return 0;}\n//// file: u.h\nint f(void);\n//// file: /sub/g.h\nint g(void);\n",
    );
    let keys: Vec<&str> = many.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(keys, ["in.c", "u.h", "sub/g.h"]); // leading `/` trimmed, nesting preserved
    assert!(String::from_utf8_lossy(&many[0].1).contains("int main"));
    assert_eq!(many[1].1, b"int f(void);\n");
}

/// A real 3-file project — an entry that pulls in a header (declarations) and a second `.c` (defs) via
/// quote-includes — compiles and runs (the unity-build path the memfs already resolves).
#[test]
fn three_file_project_compiles_and_runs() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"#include <stdio.h>
#include "mathx.h"
#include "mathx.c"
int main(void) {
  printf("gcd(48,36)=%d fib(10)=%d\n", gcd(48, 36), fib(10));
  return 0;
}
//// file: mathx.h
int gcd(int a, int b);
int fib(int n);
//// file: mathx.c
int gcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }
int fib(int n) { int x = 0, y = 1; for (int i = 0; i < n; i++) { int t = x + y; x = y; y = t; } return x; }
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(out, "gcd(48,36)=12 fib(10)=55\n");
}

/// A nested-directory include (`//// file: inc/vec.h`) resolves via `#include "inc/vec.h"`.
#[test]
fn nested_directory_include() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent");
        return;
    };
    let (status, out) = compile_and_run(
        &chibicc,
        r#"#include <stdio.h>
#include "inc/vec.h"
int main(void) { printf("dot=%d\n", dot3(1, 2, 3, 4, 5, 6)); return 0; }
//// file: inc/vec.h
static inline int dot3(int a, int b, int c, int d, int e, int f) { return a*d + b*e + c*f; }
"#,
    );
    assert_eq!(status, STATUS_OK, "run status");
    assert_eq!(out, "dot=32\n");
}
