//! **The chibicc card as separate compilation** (#1392): compile a libc unit once, compile the
//! user's translation unit against *declarations only*, and link the two — instead of compiling the
//! source-level libc into every program. The compile win is ~66x (a `printf` program: 9.1s of
//! header bodies vs 138ms of prototypes), and this is the path that has to work for it.
//!
//! Fail-soft: SKIPs if `chibicc.temen` isn't built.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// Compile one TU to a **linkable object** (`cc -c`) with the seeded headers plus `extra`, and
/// return its IR text.
fn emit_object(
    chibicc: &temen_ir::Module,
    path: &str,
    extra: &[(&str, &str)],
    debug: bool,
) -> String {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    for (name, body) in extra {
        files.push((name.to_string(), body.as_bytes().to_vec()));
    }
    let dirs = vec!["include".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);
    let tu = format!("/{path}");
    let mut argv: Vec<&[u8]> = vec![b"chibicc", b"--emit-object", b"-Iinclude"];
    if debug {
        argv.push(b"-g");
    }
    argv.push(tu.as_bytes());
    let out = onramp_fs_exec(chibicc, &image, &argv, b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "emit-object {path}: status {} — stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).expect("IR is utf8")
}

/// A miniature "libc" unit: a variadic `printf` over the ambient `write`, with external linkage so
/// `--emit-object` exports it. Stands in for the real seeded libc while the header split lands.
const LIBC_C: &str = r#"#include <stdarg.h>
int write(int fd, char *buf, long n);
int printf(char *fmt, ...) {
  va_list ap; va_start(ap, fmt);
  char buf[256]; int bi = 0;
  for (int i = 0; fmt[i]; i++) {
    if (fmt[i] == '%' && fmt[i+1] == 'd') {
      int v = va_arg(ap, int); char tmp[16]; int ti = 0; int neg = 0;
      if (v < 0) { neg = 1; v = -v; }
      if (v == 0) tmp[ti++] = '0';
      while (v) { tmp[ti++] = '0' + (v % 10); v = v / 10; }
      if (neg) buf[bi++] = '-';
      while (ti > 0) { ti = ti - 1; buf[bi++] = tmp[ti]; }
      i++;
    } else { buf[bi++] = fmt[i]; }
  }
  va_end(ap);
  write(1, buf, bi);
  return bi;
}
"#;

/// The user's TU: it sees only a **prototype**, so its own compile emits no libc at all.
const USER_C: &str = r#"int printf(char *fmt, ...);
int main(void) { printf("hi %d and %d\n", 42, -7); return 0; }
"#;

#[test]
fn a_user_tu_links_against_a_separately_compiled_libc_and_runs() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    let lib_ir = emit_object(&chibicc, "pglibc.c", &[("pglibc.c", LIBC_C)], false);
    let prog_ir = emit_object(&chibicc, "in.c", &[("in.c", USER_C)], false);

    // The user's unit must be tiny: no libc bodies in it at all.
    assert!(
        prog_ir.len() < lib_ir.len(),
        "the user unit ({} B) should be far smaller than the libc unit ({} B)",
        prog_ir.len(),
        lib_ir.len()
    );

    let lib = temen_text::parse_module(&lib_ir).expect("libc unit parses");
    let prog = temen_text::parse_module(&prog_ir).expect("user unit parses");
    let out = temen_browser::link_run_units(&lib, &prog, "main", b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run status {} — stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi 42 and -7\n");
}

/// The debugger half (#1392): with `-g`, each unit carries its own debug info, and the linked
/// program's **IR text** — what c_interpret's debug session launches from — carries the *merged*
/// tables. Without the linker's merge this text would have no `debug.*` directives at all, and
/// stepping a separately-compiled program would be impossible.
#[test]
fn the_linked_programs_ir_text_carries_both_units_debug_info() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    let lib_ir = emit_object(&chibicc, "pglibc.c", &[("pglibc.c", LIBC_C)], true);
    let prog_ir = emit_object(&chibicc, "in.c", &[("in.c", USER_C)], true);
    assert!(
        lib_ir.contains("debug.file"),
        "the libc unit was built with -g"
    );
    assert!(
        prog_ir.contains("debug.file"),
        "the user unit was built with -g"
    );

    let lib = temen_text::parse_module_debug(&lib_ir).expect("libc unit parses");
    let prog = temen_text::parse_module_debug(&prog_ir).expect("user unit parses");
    let lib_exports: Vec<(String, temen_ir::FuncIdx)> = lib
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();
    let linked = temen_browser::link_program(
        temen_ir::LinkUnitRef {
            module: &lib,
            exports: &lib_exports,
            data_exports: &[],
        },
        &prog,
        "main",
    )
    .expect("links and verifies");

    let di = linked
        .debug_info
        .as_ref()
        .expect("the linked module carries merged debug info");
    // Both units' source files survive, so a stop in either resolves to a real file.
    assert!(
        di.files.iter().any(|f| f.contains("pglibc.c")),
        "libc file missing from {:?}",
        di.files
    );
    assert!(
        di.files.iter().any(|f| f.contains("in.c")),
        "user file missing from {:?}",
        di.files
    );
    // Every loc points at a function that exists and a file that exists — the property a stepper
    // dereferences on every stop.
    for l in &di.locs {
        assert!(
            (l.func as usize) < linked.funcs.len(),
            "loc func {} out of range ({} funcs)",
            l.func,
            linked.funcs.len()
        );
        assert!(
            (l.file as usize) < di.files.len(),
            "loc file {} out of range ({} files)",
            l.file,
            di.files.len()
        );
    }
    // And it round-trips through the text waist the DAP launches from.
    let text = temen_text::print_module(&linked);
    assert!(text.contains("debug.loc"), "printed IR carries locations");
    let reparsed = temen_text::parse_module_debug(&text).expect("re-parse printed IR");
    assert_eq!(
        reparsed.debug_info, linked.debug_info,
        "the debug info survives print → parse"
    );
}

/// Cross-unit **data**, which the header split depends on: `<stdio.h>`'s `stdout` is a macro for
/// `&__pg_std[1]`, so once the libc's bodies move into their own unit that array must be one shared
/// exported global rather than a private copy per translation unit. Pinned here on the real
/// frontend: the libc unit defines the array, the user's TU declares it `extern` and reads it, and
/// the linker resolves the data symbol.
#[test]
fn a_user_tu_reads_a_libc_global_across_the_link() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    const LIB: &str = r#"int write(int fd, char *buf, long n);
int __shared_counters[3] = { 7, 11, 13 };
int emit(char *buf, long n) { return write(1, buf, n); }
"#;
    const USER: &str = r#"extern int __shared_counters[3];
int emit(char *buf, long n);
int main(void) {
  char out[4];
  int sum = __shared_counters[0] + __shared_counters[1] + __shared_counters[2]; /* 31 */
  out[0] = '0' + (sum / 10);
  out[1] = '0' + (sum % 10);
  out[2] = 10;
  emit(out, 3);
  return 0;
}
"#;
    let lib_ir = emit_object(&chibicc, "pglibc.c", &[("pglibc.c", LIB)], false);
    let prog_ir = emit_object(&chibicc, "in.c", &[("in.c", USER)], false);
    assert!(
        lib_ir.contains("__shared_counters"),
        "the libc unit publishes its global as a data symbol"
    );
    let lib = temen_text::parse_module(&lib_ir).expect("libc unit parses");
    let prog = temen_text::parse_module(&prog_ir).expect("user unit parses");
    let out = temen_browser::link_run_units(&lib, &prog, "main", b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run status {} — stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "31\n",
        "the user TU read the libc unit's global through the resolved data symbol"
    );
}
