//! **The committed prebuilt libc unit** (`web/assets/pg_libc.temeno`, #1392) — the asset gate.
//!
//! The card no longer compiles the seeded libc into every program: the bodies are compiled once into
//! this unit (`src/genlibc.rs`, driven by `scripts/rebuild-assets.sh`) and a user's program is
//! compiled decls-only against the headers' prototypes and linked against it. That makes the asset
//! **doubly wire-coupled** — it was produced by the committed `chibicc.temen` and it is itself an
//! encoded unit — so an IR / encoder / wire change invalidates it. These tests are what turns that
//! drift red instead of letting the card fail in a browser: the asset must decode, still export the
//! libc, and still *link and run* against a freshly compiled program unit.
//!
//! Fail-soft: SKIPs if the assets aren't built.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

fn asset(name: &str) -> Option<Vec<u8>> {
    std::fs::read(format!("{}/web/assets/{name}", env!("CARGO_MANIFEST_DIR"))).ok()
}

/// The committed unit, decoded through the same path the cdylib's `temen_link_lib_open` takes.
fn pg_libc() -> Option<temen_ir::Module> {
    let bytes = asset("pg_libc.temeno")?;
    Some(
        temen_encode::decode_unit(&bytes)
            .expect("decode pg_libc.temeno — stale asset? see AGENTS.md"),
    )
}

/// Compile `src` as a **program unit**: decls-only against the seeded headers, so it carries no libc
/// bodies and resolves them against the prebuilt unit at link time.
fn program_unit(src: &str) -> Option<temen_ir::Module> {
    let chibicc =
        temen_encode::decode_module(&asset("chibicc.temen")?).expect("decode chibicc.temen");
    let mut files = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let image = temen_fs::encode_image(&files, &["include".to_string()]);
    let mut argv: Vec<&[u8]> = vec![
        b"chibicc",
        b"--emit-object",
        b"--data-page",
        b"65536",
        b"-Iinclude",
    ];
    argv.extend_from_slice(temen_browser::PG_DECLS_ONLY_ARGV);
    argv.push(b"-g");
    argv.push(b"/in.c");
    let out = onramp_fs_exec(&chibicc, &image, &argv, b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "compiling the program unit: status {} — {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let ir = String::from_utf8(out.stdout).expect("IR is utf8");
    Some(temen_text::parse_module(&ir).expect("the program unit parses"))
}

/// The shape the asset exists for: it decodes, it publishes the libc, and it carries the debug info a
/// debug session steps into. Cheap — no compile at all, so this is the first thing to go red on drift.
#[test]
fn the_committed_unit_decodes_and_publishes_the_libc() {
    let Some(lib) = pg_libc() else {
        eprintln!("SKIP: pg_libc.temeno not built");
        return;
    };
    for name in [
        "printf", "fprintf", "puts", "snprintf", "malloc", "qsort", "strtod",
    ] {
        assert!(
            lib.exports.iter().any(|e| e.name == name),
            "the unit must export `{name}`; got {:?}",
            lib.exports.iter().map(|e| &e.name).collect::<Vec<_>>()
        );
    }
    // `<stdio.h>`'s `stdout` is `&__pg_std[1]`, so the streams must be a *data* symbol a program unit
    // resolves across the link rather than a private copy per translation unit.
    assert!(
        lib.data_exports.iter().any(|d| d.name == "__pg_std"),
        "the unit must publish `__pg_std` as a data symbol; got {:?}",
        lib.data_exports.iter().map(|d| &d.name).collect::<Vec<_>>()
    );
    let di = lib
        .debug_info
        .as_ref()
        .expect("built with -g, so a debug session can step into the libc");
    assert!(
        di.files.iter().any(|f| f.contains("__pg_stdio_impl.h")),
        "the bodies' own file is where a step into `printf` lands; got {:?}",
        di.files
    );
}

/// End to end, the way the card runs: a freshly compiled program unit links against the **committed**
/// asset and prints. This is the one that catches a chibicc/IR change that leaves the asset decodable
/// but no longer linkable.
#[test]
fn a_program_unit_links_against_the_committed_asset_and_runs() {
    let (Some(lib), Some(prog)) = (
        pg_libc(),
        program_unit(
            r#"#include <stdio.h>
#include <stdlib.h>
int main(void) {
  printf("printf %d\n", 42);
  fprintf(stdout, "fprintf %s\n", "shared-stdout");
  char *buf = malloc(32);
  snprintf(buf, 32, "snprintf %.2f", 1.5);
  puts(buf);
  return 0;
}
"#,
        ),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    let out = temen_browser::link_run_units(&lib, &prog, "main", b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run against the committed unit: status {} — {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        String::from_utf8_lossy(&out.stdout),
        "printf 42\nfprintf shared-stdout\nsnprintf 1.50\n"
    );
}

/// And the debugger half: the linked program's IR text carries **both** units' debug info, so a DAP
/// session can step the user's lines and step *into* the prebuilt libc.
#[test]
fn the_linked_program_carries_both_units_debug_info() {
    let (Some(lib), Some(prog)) = (
        pg_libc(),
        program_unit("#include <stdio.h>\nint main(void) { puts(\"hi\"); return 0; }\n"),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    let lib_exports: Vec<(String, temen_ir::FuncIdx)> = lib
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();
    let lib_data: Vec<(String, u64)> = lib
        .data_exports
        .iter()
        .map(|d| (d.name.clone(), d.offset))
        .collect();
    let linked = temen_browser::link_program(
        temen_ir::LinkUnitRef {
            module: &lib,
            exports: &lib_exports,
            data_exports: &lib_data,
        },
        &prog,
        "main",
    )
    .expect("links and verifies");
    let di = linked.debug_info.as_ref().expect("merged debug info");
    assert!(
        di.files.iter().any(|f| f.contains("/in.c")),
        "the user's file: {:?}",
        di.files
    );
    assert!(
        di.files.iter().any(|f| f.contains("__pg_stdio_impl.h")),
        "the libc's file: {:?}",
        di.files
    );
    // Every stop a stepper can reach resolves to a real function and a real file.
    for l in &di.locs {
        assert!(
            (l.func as usize) < linked.funcs.len(),
            "loc func {}",
            l.func
        );
        assert!((l.file as usize) < di.files.len(), "loc file {}", l.file);
    }
}
