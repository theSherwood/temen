//! **The chibicc card as separate compilation** (#1392): compile a libc unit once, compile the
//! user's translation unit against *declarations only*, and link the two — instead of compiling the
//! source-level libc into every program. This is the path that has to work for it, and
//! `the_real_seeded_libc_links_as_a_prebuilt_unit` prints what it buys. Measured there on a debug
//! build (a three-call `printf`/`fprintf`/`puts` program): **12.4 s** with the libc's bodies compiled
//! in, **1.0 s** compiled decls-only against a prebuilt libc unit — ~12x, with the emitted IR
//! 353 KB → 1.2 KB. The libc unit itself costs 13.6 s, paid *once* (shippable beside `chibicc.temen`).
//! The floor — a program with no headers at all — is 71 ms, so what is left in the 1.0 s is
//! preprocessing the *declarations*, which no amount of splitting removes.
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
    emit_object_flags(chibicc, path, extra, debug, &[])
}

/// [`emit_object`] with extra compiler flags (e.g. [`temen_browser::PG_DECLS_ONLY_ARGV`]).
fn emit_object_flags(
    chibicc: &temen_ir::Module,
    path: &str,
    extra: &[(&str, &str)],
    debug: bool,
    flags: &[&[u8]],
) -> String {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    for (name, body) in extra {
        files.push((name.to_string(), body.as_bytes().to_vec()));
    }
    let dirs = vec!["include".to_string()];
    let image = temen_fs::encode_image(&files, &dirs);
    let tu = format!("/{path}");
    let mut argv: Vec<&[u8]> = vec![b"chibicc", b"--emit-object", b"-Iinclude"];
    argv.extend_from_slice(flags);
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

/// **The payoff** (#1392): the *real* seeded libc as a prebuilt unit. The libc's bodies are compiled
/// once (`__pg_libc.c`), the user's program is compiled against declarations only (it force-includes
/// `__pg_decls_only.h`) so it sees prototypes only, and the two link and run — including `fprintf(stdout, …)`, which reaches the
/// libc unit's `__pg_std` array through a resolved data symbol rather than a private copy.
#[test]
fn the_real_seeded_libc_links_as_a_prebuilt_unit() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    const USER: &str = r#"#include <stdio.h>
int main(void) {
  printf("printf %d\n", 42);
  fprintf(stdout, "fprintf %s\n", "shared-stdout");
  puts("puts");
  return 0;
}
"#;
    let t0 = std::time::Instant::now();
    let lib_ir = emit_object(
        &chibicc,
        "__pg_libc.c",
        &[("__pg_libc.c", temen_browser::playground_libc_tu())],
        false,
    );
    let lib_ms = t0.elapsed().as_millis();

    let t1 = std::time::Instant::now();
    let prog_ir = emit_object_flags(
        &chibicc,
        "in.c",
        &[("in.c", USER)],
        false,
        temen_browser::PG_DECLS_ONLY_ARGV,
    );
    let prog_ms = t1.elapsed().as_millis();

    // The same program the old way: the libc's bodies compiled into it.
    let t2 = std::time::Instant::now();
    let whole_ir = emit_object(&chibicc, "in.c", &[("in.c", USER)], false);
    let whole_ms = t2.elapsed().as_millis();

    // The floor: a program with no headers at all. What's left above it in the decls-only compile is
    // tokenizing + preprocessing the header text, which no amount of splitting removes.
    let t3 = std::time::Instant::now();
    let bare_ir = emit_object(
        &chibicc,
        "bare.c",
        &[("bare.c", "int main(void) { return 0; }\n")],
        false,
    );
    let bare_ms = t3.elapsed().as_millis();

    eprintln!(
        "#1392 compile: libc unit {lib_ms} ms (once) | program decls-only {prog_ms} ms | \
         program with libc bodies inline {whole_ms} ms | floor (no headers) {bare_ms} ms | \
         IR {} B decls-only vs {} B whole vs {} B floor",
        prog_ir.len(),
        whole_ir.len(),
        bare_ir.len()
    );
    assert!(
        prog_ir.len() * 4 < whole_ir.len(),
        "a decls-only program unit should be far smaller: {} B vs {} B",
        prog_ir.len(),
        whole_ir.len()
    );

    let lib = temen_text::parse_module(&lib_ir).expect("libc unit parses");
    let prog = temen_text::parse_module(&prog_ir).expect("program unit parses");
    let out = temen_browser::link_run_units(&lib, &prog, "main", b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run status {} — stderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "printf 42\nfprintf shared-stdout\nputs\n"
    );
}

/// **The card's shape** (#1392): the prebuilt libc unit goes *resident* once
/// ([`temen_browser::temen_link_lib_open`]), and each compile then hands over only the user's small
/// program unit — to [`temen_browser::temen_link_run_lib`] to run it, or to
/// [`temen_browser::temen_link_text_lib`] to hand a debugger the linked program's IR text. Pins the
/// data-symbol half of that, too: `fprintf(stdout, …)` resolves `__pg_std` out of the *resident*
/// library, which the resident table used to drop on the floor (it linked with no data symbols at
/// all, so the whole seeded-libc path would have failed the moment it went through a handle).
#[test]
fn the_resident_libc_unit_serves_both_running_and_stepping() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen not built");
        return;
    };
    const USER: &str = r#"#include <stdio.h>
int main(void) {
  fprintf(stdout, "resident %d\n", 7);
  return 0;
}
"#;
    let lib_ir = emit_object(
        &chibicc,
        "__pg_libc.c",
        &[("__pg_libc.c", temen_browser::playground_libc_tu())],
        true,
    );
    let prog_ir = emit_object_flags(
        &chibicc,
        "in.c",
        &[("in.c", USER)],
        true,
        temen_browser::PG_DECLS_ONLY_ARGV,
    );

    let h = temen_browser::temen_link_lib_open(lib_ir.as_ptr(), lib_ir.len());
    assert!(
        h >= 0,
        "the libc unit goes resident (status {})",
        temen_browser::temen_status()
    );

    // (a) run it: one resident library, the program unit by handle.
    let ret = temen_browser::temen_link_run_lib(
        h,
        prog_ir.as_ptr(),
        prog_ir.len(),
        b"main".as_ptr(),
        4,
        core::ptr::null(),
        0,
    );
    assert_eq!(
        temen_browser::temen_status(),
        STATUS_OK,
        "link+run against the resident libc (ret={ret})"
    );
    assert_eq!(
        read_out(),
        b"resident 7\n",
        "`fprintf(stdout, …)` reached the resident libc's __pg_std"
    );

    // (b) step it: the same handle, the same program unit, the linked program's IR text.
    assert_eq!(
        temen_browser::temen_link_text_lib(h, prog_ir.as_ptr(), prog_ir.len(), b"main".as_ptr(), 4),
        0,
        "link-to-text against the resident libc"
    );
    let text = String::from_utf8(read_out()).expect("IR text is utf8");
    assert!(
        text.contains("debug.loc"),
        "the linked text carries merged debug info"
    );
    // Both units' files survive, so a stop on either side of the link resolves. The libc unit's
    // functions are attributed to the header the bodies live in (`__pg_stdio_impl.h`) rather than to
    // the one-line `__pg_libc.c` that includes it — which is what a step *into* `fprintf` should show.
    assert!(
        text.contains("\"/in.c\"") && text.contains("__pg_stdio_impl.h"),
        "both units' source files survive into the debug session:\n{}",
        text.lines()
            .filter(|l| l.contains("debug.file"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // A closed handle declines instead of linking against whatever is left in the slot.
    temen_browser::temen_link_lib_close(h);
    assert!(
        temen_browser::temen_link_text_lib(h, prog_ir.as_ptr(), prog_ir.len(), b"main".as_ptr(), 4)
            < 0,
        "a closed handle is not linkable"
    );
}

/// The bytes the cdylib accessors currently hold (`temen_stdout_ptr`/`_len`).
fn read_out() -> Vec<u8> {
    let n = temen_browser::temen_stdout_len();
    if n == 0 {
        return Vec::new();
    }
    // SAFETY: the accessor pair describes a live stash owned by the cdylib.
    unsafe { core::slice::from_raw_parts(temen_browser::temen_stdout_ptr(), n) }.to_vec()
}
