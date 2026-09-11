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

/// **The card's own path**, end to end through the cdylib exports the browser calls (#1392): the
/// committed unit goes resident, `temen_run_onramp_fs` compiles the user's C as a *program unit*
/// (`CHIBICC_PROGRAM_UNIT` — `--emit-object` against declarations only), `temen_link_encode_lib` links
/// it against the resident unit and hands back **runnable module bytes**, and those run. No IR text
/// round trip for the linked libc, and the card's existing run passes need no change.
#[test]
fn the_card_path_compiles_a_program_unit_and_links_it_through_the_cdylib() {
    let (Some(chibicc), Some(lib_bytes)) = (asset("chibicc.temen"), asset("pg_libc.temeno")) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    const USER: &str = r#"#include <stdio.h>
int main(void) {
  printf("card %d\n", 3);
  fprintf(stdout, "and stdout\n");
  return 7;
}
"#;
    let h = temen_browser::temen_link_lib_open(lib_bytes.as_ptr(), lib_bytes.len());
    assert!(h >= 0, "the committed unit goes resident");

    // Pass 1 — compile, exactly as the card does (empty caller image; flags pick the program unit).
    let flags = temen_browser::CHIBICC_DEBUG_INFO | temen_browser::CHIBICC_PROGRAM_UNIT;
    temen_browser::temen_run_onramp_fs(
        chibicc.as_ptr(),
        chibicc.len(),
        core::ptr::null(),
        0,
        USER.as_ptr(),
        USER.len(),
        flags,
    );
    assert_eq!(
        temen_browser::temen_status(),
        STATUS_OK,
        "compile status; stderr: {}",
        String::from_utf8_lossy(&read_err())
    );
    let unit = read_out();
    // The payoff, visible on the card: the emitted IR is the user's program, not a libc.
    assert!(
        unit.len() < 8 * 1024,
        "a program unit should be small; got {} B",
        unit.len()
    );

    // Pass 1b — link against the resident unit, straight to runnable module bytes.
    assert_eq!(
        temen_browser::temen_link_encode_lib(h, unit.as_ptr(), unit.len(), b"main".as_ptr(), 4),
        0,
        "link+encode against the resident unit"
    );
    let module = read_out();
    let m = temen_encode::decode_module(&module).expect("the linked module decodes as runnable");
    assert!(
        temen_verify::verify_module(&m).is_ok(),
        "and verifies — it is what the card hands to either tier"
    );

    // Pass 2 — run it, the card's oracle tier.
    let run = temen_browser::onramp_exec(&m, b"");
    assert!(
        run.status == STATUS_OK || run.status == STATUS_EXIT,
        "run status {}",
        run.status
    );
    assert_eq!(String::from_utf8_lossy(&run.stdout), "card 3\nand stdout\n");
    assert_eq!(run.value, 7, "the card shows `main`'s return value");

    temen_browser::temen_link_lib_close(h);
}

/// The bytes the cdylib's stdout / stderr accessors currently hold.
fn read_out() -> Vec<u8> {
    let n = temen_browser::temen_stdout_len();
    if n == 0 {
        return Vec::new();
    }
    // SAFETY: the accessor pair describes a live stash owned by the cdylib.
    unsafe { core::slice::from_raw_parts(temen_browser::temen_stdout_ptr(), n) }.to_vec()
}

fn read_err() -> Vec<u8> {
    let n = temen_browser::temen_stderr_len();
    if n == 0 {
        return Vec::new();
    }
    // SAFETY: as above.
    unsafe { core::slice::from_raw_parts(temen_browser::temen_stderr_ptr(), n) }.to_vec()
}
