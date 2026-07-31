//! **Stage-2: chibicc compiles its own source in the sandbox** (SELFHOST_C.md §7 step E). The
//! playground libc grew the surface chibicc's own translation units need — a real buffered `FILE*`
//! with `open_memstream` (chibicc's `format()` builds every string through a memory stream), `strtold`
//! (long double = double here), `strerror`, and the system-header stubs `chibicc.h` `#include`s
//! (`<sys/stat.h>`, `<unistd.h>`, `<time.h>`, `<glob.h>`, …). With those seeded, chibicc-the-guest
//! **compiles real chibicc `-cc1` translation units to valid SVM IR**, entirely in-sandbox — the
//! self-host lever. These tests compile several of chibicc's own `.c` files (tokenize/strings/hashmap/
//! unicode — chosen to exercise the tokenizer, the `open_memstream` string builder, the allocator, and
//! the UTF-8 path) and assert each parses + verifies as a module. Fail-soft: SKIPs without `chibicc.svmb`.
//!
//! Not yet covered (the remaining self-host slice): a single **unity** build of the whole compiler
//! (all TUs + the `cc1_main` entry) into one runnable module — cross-TU references (e.g. parse.c's
//! `struct_type`) only resolve when every TU is compiled together, and that whole-compiler amalgamation
//! has its own tail. This gate pins that the per-TU compile — the hard libc/header work — is done.

use svm_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn chibicc_svmb() -> Option<svm_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.svmb");
    let bytes = std::fs::read(p).ok()?;
    Some(svm_encode::decode_module(&bytes).expect("decode chibicc.svmb"))
}

/// Seed the playground libc + `chibicc.h` (a sibling of the source, quote-included), compile the
/// chibicc translation unit `tu` (read from `frontend/chibicc/`), and return `(status, ir_len,
/// parses_ok)`. The seeded headers are exactly what the card ships (`playground_include_files`), so
/// this exercises the *shipped* libc — no test-only headers.
fn compile_tu(chibicc: &svm_ir::Module, tu: &str) -> (i32, usize, bool) {
    let frontend = concat!(env!("CARGO_MANIFEST_DIR"), "/../frontend/chibicc");
    let mut files = playground_include_files();
    // chibicc.h at `/chibicc.h` — the TU `#include "chibicc.h"`s it (quote-include against `/`).
    files.push((
        "chibicc.h".to_string(),
        std::fs::read(format!("{frontend}/chibicc.h")).expect("read chibicc.h"),
    ));
    files.push((
        "in.c".to_string(),
        std::fs::read(format!("{frontend}/{tu}")).unwrap_or_else(|e| panic!("read {tu}: {e}")),
    ));
    let dirs = vec!["include".to_string(), "include/sys".to_string()];
    let image = svm_fs::encode_image(&files, &dirs);

    let out = onramp_fs_exec(
        chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    let ir = String::from_utf8_lossy(&out.stdout).into_owned();
    let parses = (out.status == STATUS_OK || out.status == STATUS_EXIT)
        && ir.contains("func")
        && svm_text::parse_module(&ir).is_ok();
    (out.status, ir.len(), parses)
}

/// Real chibicc `-cc1` translation units compile to valid SVM IR in the sandbox. `strings.c` is the
/// libc-surface load-bearer (chibicc's `format()` builds strings through `open_memstream` +
/// `vfprintf` + `fclose`). Also pins the **`vm_map`-growable heap**: removing chibicc's fixed 24 MiB
/// static arena (so `malloc` grows into the reserved tail on demand) dropped chibicc.svmb's declared
/// memory — the arena was the bloat — which the `size_log2` check gates structurally.
#[test]
fn chibicc_translation_units_compile_to_valid_ir() {
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    assert!(
        chibicc.memory.map_or(0, |m| m.size_log2) <= 24,
        "growable heap ⇒ chibicc.svmb no longer carries a multi-MiB static arena in its window"
    );
    for tu in ["tokenize.c", "strings.c", "hashmap.c", "unicode.c"] {
        let (status, ir_len, parses) = compile_tu(&chibicc, tu);
        assert!(
            parses,
            "chibicc TU {tu} should compile to valid SVM IR (status {status}, {ir_len} B IR)"
        );
        eprintln!("chibicc {tu} → {ir_len} B of SVM IR (parses + verifies)");
    }
}

/// The **growable-heap** proof, on chibicc's largest translation unit: `codegen_ir.c` emits ~1.4 MB of
/// IR, a compiler working set that overran the old fixed 24 MiB guest arena (silent corruption →
/// spurious type errors). With `malloc` now growing into the reserved tail on demand, it compiles
/// cleanly. `#[ignore]`d because it takes ~70 s on the interpreter — run with `--ignored` on demand.
#[test]
#[ignore = "heavy (~70s): chibicc's largest TU — the growable-heap stress proof"]
fn largest_tu_compiles_with_the_growable_heap() {
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let (status, ir_len, parses) = compile_tu(&chibicc, "codegen_ir.c");
    assert!(
        parses,
        "chibicc codegen_ir.c should compile to valid SVM IR (status {status}, {ir_len} B IR)"
    );
    eprintln!("chibicc codegen_ir.c → {ir_len} B of SVM IR (parses + verifies)");
}

/// The load-bearing new capability, exercised at **runtime**: `open_memstream` → `fprintf` into the
/// memory `FILE*` → `fclose` → read the buffer back. This is exactly chibicc's `format()` shape, so a
/// program using it compiles *and runs* correctly under the powerbox.
#[test]
fn open_memstream_round_trips_at_runtime() {
    use svm_browser::onramp_exec;
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: chibicc.svmb absent");
        return;
    };
    let mut files = playground_include_files();
    files.push((
        "in.c".to_string(),
        br#"#include <stdio.h>
int main(void) {
  char *buf; size_t len;
  FILE *m = open_memstream(&buf, &len);
  for (int i = 0; i < 3; i++) fprintf(m, "[%d]", i * i);
  fclose(m);
  printf("%s len=%lu\n", buf, (unsigned long)len);
  return 0;
}
"#
        .to_vec(),
    ));
    let image = svm_fs::encode_image(&files, &["include".to_string()]);
    let compiled = onramp_fs_exec(
        &chibicc,
        &image,
        &[b"chibicc", b"--data-page", b"65536", b"/in.c"],
        b"",
    );
    assert!(compiled.status == STATUS_OK || compiled.status == STATUS_EXIT);
    let ir = String::from_utf8(compiled.stdout).expect("IR utf8");
    let m = svm_text::parse_module(&ir).expect("parse IR");
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_OK);
    assert_eq!(String::from_utf8_lossy(&run.stdout), "[0][1][4] len=9\n");
}
