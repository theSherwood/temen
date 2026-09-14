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
    // One name from each seeded header the unit carries: <stdio.h>, <stdlib.h>, <string.h>, <math.h>.
    for name in [
        "printf", "fprintf", "puts", "snprintf", "malloc", "qsort", "strtod", "strlen", "memcpy",
        "strtok", "strdup", "sqrt", "pow", "sin", "fabs",
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
    for f in [
        "__pg_stdio_impl.h",
        "__pg_stdlib_impl.h",
        "__pg_string_impl.h",
        "__pg_math_impl.h",
    ] {
        assert!(
            di.files.iter().any(|n| n.contains(f)),
            "{f} is where a step into its functions lands; got {:?}",
            di.files
        );
    }
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
#include <string.h>
#include <math.h>
int main(void) {
  printf("printf %d\n", 42);
  fprintf(stdout, "fprintf %s\n", "shared-stdout");
  char *buf = malloc(32);
  snprintf(buf, 32, "snprintf %.2f", 1.5);
  puts(buf);
  char *dup = strdup("strdup");
  printf("%s len=%d\n", dup, (int)strlen(dup));
  printf("sqrt=%g pow=%g\n", sqrt(169.0), pow(2.0, 10.0));
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
        "printf 42\nfprintf shared-stdout\nsnprintf 1.50\nstrdup len=6\nsqrt=13 pow=1024\n"
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

/// **Dead-code elimination across the link** (#1407): a program that calls one libc function must not
/// carry the other 118. Measured against the same program linked with the collection declined, so this
/// pins the *effect*, not a hand-written number that drifts with the libc's size.
#[test]
fn the_linked_program_drops_what_it_cannot_reach() {
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
    let unit = temen_ir::LinkUnitRef {
        module: &lib,
        exports: &lib_exports,
        data_exports: &lib_data,
    };
    let linked = temen_browser::link_program(unit, &prog, "main").expect("links");

    // The uncollected shape, for comparison: the same units linked with every library export still a
    // root. The program's exports come from its own table (minus chibicc's whole-program `_start`
    // bootstrap at func 0, whose paramless signature `synth_manifest_start` rejects) — the same
    // resolution `link_program` does, so the only difference between the two modules is the collection.
    let mut prog_exports: Vec<(String, temen_ir::FuncIdx)> = prog
        .exports
        .iter()
        .filter(|e| e.name != "_start")
        .map(|e| (e.name.clone(), e.func))
        .collect();
    if !prog_exports.iter().any(|(n, _)| n == "main") {
        prog_exports.push(("main".to_string(), 0));
    }
    let whole = temen_ir::link_with_manifest_ref(&[
        unit,
        temen_ir::LinkUnitRef {
            module: &prog,
            exports: &prog_exports,
            data_exports: &[],
        },
    ])
    .expect("links");
    let entry = whole.resolve_export("main").expect("entry");
    let whole = temen_ir::synth_manifest_start(whole, entry, false).expect("synth");

    // The index space is deliberately unchanged (a funcidx is observable through `call.indirect`), so
    // the win shows up as *bodies*: count functions that still have instructions, and compare encoded
    // size.
    let with_bodies = |m: &temen_ir::Module| {
        m.funcs
            .iter()
            .filter(|f| f.blocks.iter().any(|b| !b.insts.is_empty()))
            .count()
    };
    let (kept, total) = (with_bodies(&linked), with_bodies(&whole));
    let (small, big) = (
        temen_encode::encode_module(&linked).len(),
        temen_encode::encode_module(&whole).len(),
    );
    assert_eq!(
        linked.funcs.len(),
        whole.funcs.len(),
        "the index space must not move: `call.indirect` masks into a table whose slot i is funcidx i"
    );
    assert!(
        kept * 2 < total,
        "a `puts`-only program should keep a small fraction of the libc's bodies: {kept} of {total}"
    );
    assert!(
        small * 2 < big,
        "and the encoded module should shrink with them: {small} B vs {big} B"
    );
    eprintln!("#1407 gc: {kept} bodies kept of {total} ({small} B vs {big} B encoded)");

    // Still correct, still steppable: it runs, and the debug info that survived is in range.
    let out = temen_browser::onramp_exec(&linked, b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "run status {}",
        out.status
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), "hi\n");
    let di = linked
        .debug_info
        .as_ref()
        .expect("debug info survives the gc");
    for l in &di.locs {
        assert!(
            (l.func as usize) < linked.funcs.len(),
            "loc func {} out of range after the gc ({} funcs)",
            l.func,
            linked.funcs.len()
        );
    }
    assert!(
        di.files.iter().any(|f| f.contains("/in.c")),
        "the user's own file is still there: {:?}",
        di.files
    );
}

/// **Several resident libraries at once** (#1408): the same program links against the prebuilt libc
/// passed as a one-entry handle list, and an unknown handle anywhere in the list declines the whole
/// call rather than linking against whatever occupies that slot.
#[test]
fn the_multi_handle_entries_link_and_fail_closed() {
    let (Some(chibicc), Some(lib_bytes)) = (asset("chibicc.temen"), asset("pg_libc.temeno")) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    const USER: &str = "#include <stdio.h>\nint main(void) { puts(\"multi\"); return 3; }\n";
    let h = temen_browser::temen_link_lib_open(lib_bytes.as_ptr(), lib_bytes.len());
    assert!(h >= 0, "resident");

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
    assert_eq!(temen_browser::temen_status(), STATUS_OK, "compile");
    let unit = read_out();

    let handles = [h];
    let rv = temen_browser::temen_link_run_libs(
        handles.as_ptr(),
        handles.len(),
        unit.as_ptr(),
        unit.len(),
        b"main".as_ptr(),
        4,
        core::ptr::null(),
        0,
    );
    assert!(
        temen_browser::temen_status() == STATUS_OK || temen_browser::temen_status() == STATUS_EXIT,
        "run status {}",
        temen_browser::temen_status()
    );
    assert_eq!(String::from_utf8_lossy(&read_out()), "multi\n");
    assert_eq!(rv, 3, "`main`'s return value");

    // An empty list is legal and links: `link_with_manifest_ref` *retains* an unresolved name as a
    // host-bound manifest import rather than failing, so a library-less program is a well-formed
    // module whose `puts` is simply unbound. (Running it is what would fault — not this call's job.)
    assert_eq!(
        temen_browser::temen_link_encode_libs(
            core::ptr::null(),
            0,
            unit.as_ptr(),
            unit.len(),
            b"main".as_ptr(),
            4
        ),
        0,
        "an empty handle list is a legal link, not an error"
    );

    // A bad handle beside a good one declines the whole call.
    let bogus = [h, 9999];
    assert!(
        temen_browser::temen_link_text_libs(
            bogus.as_ptr(),
            bogus.len(),
            unit.as_ptr(),
            unit.len(),
            b"main".as_ptr(),
            4
        ) < 0,
        "one unknown handle fails the list closed"
    );

    temen_browser::temen_link_lib_close(h);
}
