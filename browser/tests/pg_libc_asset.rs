//! **The committed prebuilt C units** — the libc (`web/assets/pg_libc.temeno`, #1392) and the heap
//! (`web/assets/pg_heap.temeno`, #2172) — the asset gate.
//!
//! The card no longer compiles the seeded libc into every program: the bodies are compiled once into
//! the libc unit (`src/genlibc.rs`, driven by `scripts/rebuild-assets.sh`) and a user's program is
//! compiled decls-only against the headers' prototypes and linked against it and the heap unit, which
//! alone defines `malloc`/`free`/`calloc`/`realloc`. That makes the libc **doubly wire-coupled** — it
//! was produced by the committed `chibicc.temen` and it is itself an encoded unit — and the heap is an
//! encoded unit too, so an IR / encoder / wire change invalidates both. These tests are what turns that
//! drift red instead of letting the card fail in a browser: the assets must decode, still export the
//! libc and the heap, and still *link and run* against a freshly compiled program unit.
//!
//! Fail-soft: SKIPs if the assets aren't built.

use temen_browser::{onramp_fs_exec, playground_include_files, STATUS_EXIT, STATUS_OK};

#[path = "support/card_children.rs"]
mod card_children;
#[path = "support/ffi.rs"]
mod ffi;
#[path = "support/pg_heap.rs"]
mod pg_heap;

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

/// `unit` lays no data in the scratch page below `guard + POWERBOX_ARGS_END`. The unit linked first
/// keeps its offsets, and that page is where the program's `_start` seeds the heap words and the host
/// seeds the args, so a global there would be overwritten.
fn assert_above_the_scratch_page(unit: &temen_ir::Module, what: &str) {
    let end = temen_ir::module_args_end();
    for d in &unit.data {
        assert!(
            d.offset >= end,
            "{what}: a data segment at {} lies in the scratch page, below {end}",
            d.offset
        );
    }
}

/// Compile `src` as a **program unit**: decls-only against the seeded headers, so it carries no libc
/// bodies and resolves them against the prebuilt unit at link time.
fn program_unit(src: &str) -> Option<temen_ir::Module> {
    let ir = program_unit_text(src)?;
    Some(temen_text::parse_module(&ir).expect("the program unit parses"))
}

/// [`program_unit`]'s IR text, as chibicc emits it: what the card hands the link entries.
fn program_unit_text(src: &str) -> Option<String> {
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
    Some(String::from_utf8(out.stdout).expect("IR is utf8"))
}

/// A program that reaches much of the libc and the heap — formatted output, strings, math — with a
/// dispatch table of function pointers in its static data.
const TABLE_PROGRAM: &str = r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int add(int a, int b) { return a + b; }
int sub(int a, int b) { return a - b; }
int (*ops[2])(int, int) = { add, sub };
int main(void) {
  printf("printf %d\n", 42);
  fprintf(stdout, "fprintf %s\n", "shared-stdout");
  char *buf = malloc(32);
  snprintf(buf, 32, "snprintf %.2f", 1.5);
  puts(buf);
  char *dup = strdup("strdup");
  printf("%s len=%d\n", dup, (int)strlen(dup));
  printf("sqrt=%g pow=%g\n", sqrt(169.0), pow(2.0, 10.0));
  printf("ops %d %d\n", ops[0](2, 3), ops[1](9, 4));
  return 0;
}
"#;
/// What [`TABLE_PROGRAM`] prints.
const TABLE_OUT: &str =
    "printf 42\nfprintf shared-stdout\nsnprintf 1.50\nstrdup len=6\nsqrt=13 pow=1024\nops 5 5\n";

/// And one that reaches little of either.
const SMALL_PROGRAM: &str = r#"#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void) {
  char *s = malloc(8);
  strcpy(s, "small");
  puts(s);
  free(s);
  return 3;
}
"#;

/// And one that passes libc functions as pointers (#2203): it takes `sqrt` and `fabs` by name in
/// code (`ref.sym`), and nothing else in it calls them.
const APPLY_PROGRAM: &str = r#"#include <stdio.h>
#include <math.h>
static double apply(double (*f)(double), double x) { return f(x); }
int main(void) {
  printf("%g %g\n", apply(sqrt, 16.0), apply(fabs, -2.5));
  return (int)apply(sqrt, 49.0);
}
"#;
/// What [`APPLY_PROGRAM`] prints.
const APPLY_OUT: &str = "4 2.5\n";

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
        "printf", "fprintf", "puts", "snprintf", "qsort", "strtod", "strlen", "memcpy", "strtok",
        "strdup", "sqrt", "pow", "sin", "fabs",
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
    assert_above_the_scratch_page(&lib, "pg_libc.temeno");
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

/// The heap unit (#2172) decodes the way the cdylib's `temen_link_lib_open` takes it, and is exactly
/// the allocator: it exports the four names the headers declare and nothing else (dlmalloc's own
/// entry points stay inside the unit), and asks the powerbox for only what it grows and aborts with.
#[test]
fn the_committed_heap_unit_is_the_allocator_and_nothing_more() {
    let Some(bytes) = asset("pg_heap.temeno") else {
        eprintln!("SKIP: pg_heap.temeno not built");
        return;
    };
    let heap = temen_encode::decode_unit(&bytes)
        .expect("decode pg_heap.temeno — stale asset? see AGENTS.md");
    let mut exports: Vec<&str> = heap.exports.iter().map(|e| e.name.as_str()).collect();
    exports.sort_unstable();
    assert_eq!(exports, ["calloc", "free", "malloc", "realloc"]);
    assert!(heap.data_exports.is_empty(), "{:?}", heap.data_exports);
    let mut imports: Vec<&str> = heap.imports.iter().map(|i| i.name.as_str()).collect();
    imports.sort_unstable();
    imports.dedup();
    assert_eq!(imports, ["exit", "stderr", "vm_map", "vm_page_size"]);
    assert_above_the_scratch_page(&heap, "pg_heap.temeno");
}

/// End to end, the way the card runs: a freshly compiled program unit links against the **committed**
/// asset and prints. This is the one that catches a chibicc/IR change that leaves the asset decodable
/// but no longer linkable. The program's dispatch table holds function pointers in static data, which
/// a program unit could not until #2194.
#[test]
fn a_program_unit_links_against_the_committed_asset_and_runs() {
    let (Some(lib), Some(prog)) = (pg_libc(), program_unit(TABLE_PROGRAM)) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    let out = temen_browser::onramp_exec(&pg_heap::link(&[&lib], &prog), b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "link+run against the committed unit: status {} — {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(String::from_utf8_lossy(&out.stdout), TABLE_OUT);
}

/// **A libc function passed as a pointer** (#2203): since every card program is a unit, `sqrt` lives
/// in another unit, so taking its address is a `ref.sym` the link resolves to the libc's `sqrt`.
#[test]
fn a_program_unit_passes_a_libc_function_as_a_pointer() {
    let (Some(lib), Some(prog)) = (pg_libc(), program_unit(APPLY_PROGRAM)) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    let mut taken: Vec<&[u8]> = prog
        .funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter_map(|i| match i {
            temen_ir::Inst::RefSym { name } => Some(name.as_slice()),
            _ => None,
        })
        .collect();
    taken.sort_unstable();
    taken.dedup();
    assert_eq!(
        taken,
        [&b"fabs"[..], b"sqrt"],
        "the libc functions it takes by name"
    );
    let out = temen_browser::onramp_exec(&pg_heap::link(&[&lib], &prog), b"");
    assert_eq!(out.status, STATUS_OK, "trap: {:?}", out.trap);
    assert_eq!(String::from_utf8_lossy(&out.stdout), APPLY_OUT);
    assert_eq!(out.value, 7);
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
    let linked = pg_heap::link(&[&lib], &prog);
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

/// **The card's own path**, end to end through the cdylib exports the browser calls (#1392, #2172):
/// the committed libc and heap units go resident, `temen_run_onramp_fs` compiles the user's C as a
/// *program unit* (`CHIBICC_PROGRAM_UNIT` — `--emit-object` against declarations only),
/// `temen_link_encode_libs` links it against the resident units and hands back **runnable module
/// bytes**, and those run. No IR text round trip for the linked libc.
#[test]
fn the_card_path_compiles_a_program_unit_and_links_it_through_the_cdylib() {
    let _exports = ffi::lock();
    let (Some(chibicc), Some(lib_bytes), Some(heap_bytes)) = (
        asset("chibicc.temen"),
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
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
    assert!(h >= 0, "the committed libc unit goes resident");
    let hh = temen_browser::temen_link_lib_open(heap_bytes.as_ptr(), heap_bytes.len());
    assert!(hh >= 0, "the committed heap unit goes resident");
    let handles = [h, hh];

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

    // Pass 1b — link against the resident units, straight to runnable module bytes.
    assert_eq!(
        temen_browser::temen_link_encode_libs(
            handles.as_ptr(),
            handles.len(),
            unit.as_ptr(),
            unit.len(),
            b"main".as_ptr(),
            4
        ),
        0,
        "link+encode against the resident units"
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
    temen_browser::temen_link_lib_close(hh);
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
    let linked = pg_heap::link(&[&lib], &prog);
    let heap = asset("pg_heap.temeno")
        .map(|b| temen_encode::decode_unit(&b).expect("decode pg_heap.temeno"))
        .expect("pg_heap.temeno");
    let exports_of = |m: &temen_ir::Module| -> Vec<(String, temen_ir::FuncIdx)> {
        m.exports.iter().map(|e| (e.name.clone(), e.func)).collect()
    };
    let (lib_exports, heap_exports) = (exports_of(&lib), exports_of(&heap));

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
        temen_ir::LinkUnitRef {
            module: &lib,
            exports: &lib_exports,
            data_exports: &lib.data_exports,
            live: None,
        },
        temen_ir::LinkUnitRef {
            module: &heap,
            exports: &heap_exports,
            data_exports: &heap.data_exports,
            live: None,
        },
        temen_ir::LinkUnitRef {
            module: &prog,
            exports: &prog_exports,
            data_exports: &[],
            live: None,
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

/// **Several resident libraries at once** (#1408): the program links against the prebuilt libc and
/// heap passed as a handle list, and an unknown handle anywhere in the list declines the whole call
/// rather than linking against whatever occupies that slot.
#[test]
fn the_multi_handle_entries_link_and_fail_closed() {
    let _exports = ffi::lock();
    let (Some(chibicc), Some(lib_bytes), Some(heap_bytes)) = (
        asset("chibicc.temen"),
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
        return;
    };
    const USER: &str = "#include <stdio.h>\nint main(void) { puts(\"multi\"); return 3; }\n";
    let h = temen_browser::temen_link_lib_open(lib_bytes.as_ptr(), lib_bytes.len());
    assert!(h >= 0, "resident");
    let hh = temen_browser::temen_link_lib_open(heap_bytes.as_ptr(), heap_bytes.len());
    assert!(hh >= 0, "resident");

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

    let handles = [h, hh];
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
    temen_browser::temen_link_lib_close(hh);
}

/// A library as [`temen_browser::temen_link_lib_open`] keeps it resident: without the variables its
/// debug info scopes globally, which stay out of a linked program's debug vars.
fn resident(mut lib: temen_ir::Module) -> temen_ir::Module {
    if let Some(di) = lib.debug_info.as_mut() {
        di.vars.retain(|v| v.func != temen_ir::GLOBAL_SCOPE);
    }
    lib
}

/// The resident libc and heap ([`temen_browser::temen_link_lib_open`]), in the card's link order.
fn open_pg_units(libc: &[u8], heap: &[u8]) -> [i32; 2] {
    [libc, heap].map(|bytes| {
        let h = temen_browser::temen_link_lib_open(bytes.as_ptr(), bytes.len());
        assert!(h >= 0, "the committed unit goes resident");
        h
    })
}

/// The program unit `ir` linked against `handles` at `main`, as the module bytes
/// [`temen_browser::temen_link_encode_libs`] hands back.
fn link_encode_libs(handles: &[i32], ir: &str) -> Vec<u8> {
    let rc = temen_browser::temen_link_encode_libs(
        handles.as_ptr(),
        handles.len(),
        ir.as_ptr(),
        ir.len(),
        b"main".as_ptr(),
        4,
    );
    assert_eq!(
        rc,
        0,
        "link+encode: status {}",
        temen_browser::temen_status()
    );
    read_out()
}

/// The program unit `ir` linked against `handles` and run ([`temen_browser::temen_link_run_libs`]):
/// `main`'s value, the status, and what it printed.
fn link_run_libs(handles: &[i32], ir: &str) -> (i64, i32, String) {
    let rv = temen_browser::temen_link_run_libs(
        handles.as_ptr(),
        handles.len(),
        ir.as_ptr(),
        ir.len(),
        b"main".as_ptr(),
        4,
        core::ptr::null(),
        0,
    );
    let out = String::from_utf8_lossy(&read_out()).into_owned();
    (rv, temen_browser::temen_status(), out)
}

/// **The libraries laid out once** (#1373). The link entries start from the resident libraries'
/// laid-out base and copy only the library functions the program reaches, and the module they hand
/// back is, byte for byte, the one linking against the libraries themselves gives. Checked for a
/// program that reaches much of the libc, one that reaches little, and one that reaches two libc
/// functions only by taking their addresses, in turn, against one base.
#[test]
fn a_program_links_against_the_laid_out_libraries_as_against_the_libraries() {
    let _exports = ffi::lock();
    let (Some(libc), Some(heap), Some(table), Some(small), Some(apply)) = (
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
        program_unit_text(TABLE_PROGRAM),
        program_unit_text(SMALL_PROGRAM),
        program_unit_text(APPLY_PROGRAM),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
        return;
    };
    let libs = [&libc, &heap].map(|b| resident(temen_encode::decode_unit(b).expect("decode")));
    let exports: Vec<Vec<(String, temen_ir::FuncIdx)>> = libs
        .iter()
        .map(|m| m.exports.iter().map(|e| (e.name.clone(), e.func)).collect())
        .collect();
    let units: Vec<temen_ir::LinkUnitRef<'_>> = libs
        .iter()
        .zip(&exports)
        .map(|(m, exports)| temen_ir::LinkUnitRef {
            module: m,
            exports,
            data_exports: &m.data_exports,
            live: None,
        })
        .collect();
    let handles = open_pg_units(&libc, &heap);
    for ir in [&table, &small, &apply, &table] {
        let prog = temen_text::parse_module(ir).expect("the program unit parses");
        let plain = temen_browser::link_program_multi(&units, &prog, "main").expect("links");
        let plain = temen_encode::encode_module(&plain);
        let laid_out = link_encode_libs(&handles, ir);
        assert!(
            laid_out == plain,
            "linked against the base: {} B; against the libraries: {} B",
            laid_out.len(),
            plain.len()
        );
    }
    for h in handles {
        temen_browser::temen_link_lib_close(h);
    }
}

/// **The libraries compiled once per page** (#2168). Programs run against the same resident libraries
/// compile through one memo, which hands a run the functions an earlier run compiled, and only while
/// everything the compile read is unchanged: a library function the same, a program function at the
/// same position compiled anew. Three programs in turn, twice over, each printing what it prints
/// alone.
#[test]
fn programs_run_in_turn_against_the_resident_libraries_print_their_own_output() {
    let _exports = ffi::lock();
    let (Some(libc), Some(heap), Some(table), Some(small), Some(apply)) = (
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
        program_unit_text(TABLE_PROGRAM),
        program_unit_text(SMALL_PROGRAM),
        program_unit_text(APPLY_PROGRAM),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
        return;
    };
    let handles = open_pg_units(&libc, &heap);
    for _ in 0..2 {
        assert_eq!(
            link_run_libs(&handles, &table),
            (0, STATUS_OK, TABLE_OUT.to_string())
        );
        assert_eq!(
            link_run_libs(&handles, &small),
            (3, STATUS_OK, "small\n".to_string())
        );
        assert_eq!(
            link_run_libs(&handles, &apply),
            (7, STATUS_OK, APPLY_OUT.to_string())
        );
    }
    for h in handles {
        temen_browser::temen_link_lib_close(h);
    }
}

/// What the engine keeps for a list of resident libraries goes when one of them closes. The list
/// declines, and a library opened into the freed handle links as itself: here the libc a second time,
/// whose symbols collide with the first's, so the link declines where the heap's base would have run.
#[test]
fn a_closed_library_takes_its_laid_out_base_with_it() {
    let _exports = ffi::lock();
    let (Some(libc), Some(heap), Some(small)) = (
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
        program_unit_text(SMALL_PROGRAM),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
        return;
    };
    let handles = open_pg_units(&libc, &heap);
    assert_eq!(
        link_run_libs(&handles, &small),
        (3, STATUS_OK, "small\n".to_string())
    );
    temen_browser::temen_link_lib_close(handles[1]);
    assert_eq!(
        link_run_libs(&handles, &small).1,
        temen_browser::STATUS_UNSUPPORTED,
        "a closed handle declines the list"
    );
    let again = temen_browser::temen_link_lib_open(libc.as_ptr(), libc.len());
    assert_eq!(again, handles[1], "the freed handle is the next one opened");
    assert_eq!(
        link_run_libs(&handles, &small).1,
        temen_browser::STATUS_UNSUPPORTED,
        "the libc twice is a duplicate-symbol link"
    );
    for h in handles {
        temen_browser::temen_link_lib_close(h);
    }
}

/// The playground's C `detached` card, down the page's own path: compiled as a program unit, linked
/// against the committed libc and heap, run by `onramp_exec`. Its `//// child: square.c` program is
/// compiled and linked to run as a child, and staged for the run as the page stages it (#2219). It
/// spawns detached, whose `"budget"` allowance cannot cross into a §14 child yet, so the on-ramp runs it
/// at the root (#1720).
#[test]
fn the_c_detached_card_links_and_squares_through_its_region() {
    const PLAY_JS: &str = include_str!("../web/play.js");
    let key = "'detached child over a pre-mapped region (chibicc → Temen)'";
    let i = PLAY_JS.find(key).expect("card in play.js");
    let j = PLAY_JS[i..].find("src: `").expect("card src") + i + 6;
    let k = PLAY_JS[j..].find("`,\n  },").expect("card src end") + j;
    let src = PLAY_JS[j..k].replace("\\\\", "\\");
    let (parent, children) = card_children::split(&src);
    let (Some(lib), Some(prog)) = (pg_libc(), program_unit(&parent)) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno not built");
        return;
    };
    assert_eq!(children.len(), 1, "the card's square.c");
    for (name, child) in children {
        let child = program_unit(&child).expect("compile square.c");
        temen_browser::stage_child_program(&name, pg_heap::link_child(&[&lib], &child));
    }
    let out = temen_browser::onramp_exec(&pg_heap::link(&[&lib], &prog), b"");
    assert_eq!(out.status, STATUS_OK, "trap: {:?}", out.trap);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("child returned 8; the region now holds: 1 4 9 16 25 36 49 64"),
        "{stdout}"
    );
}

/// A child that prints: what a `//// child: talker.c` program is (#2219).
const TALKER: &str = r#"#include <stdio.h>
int main(void) {
  printf("talker\n");
  return 7;
}
"#;

/// Spawns `talker` with what its `printf` needs — its parent's stdout and the libc's `vm_fs` — and the
/// rest of its imports left empty on purpose (`*`), then with nothing, and prints both results.
const SPAWNS_TALKER: &str = r#"#include <stdio.h>
#include <temen.h>
#include <temen/spawn.h>

static long scratch[24];

int main(void) {
  long talker = __vm_resolve("talker", 6);
  if (talker < 0) {
    printf("no talker\n");
    return 1;
  }
  vm_grant g[3];
  g[0].name = "stdout";
  g[0].handle = (int)__vm_resolve("stdout", 6);
  g[1].name = "vm_fs";
  g[1].handle = (int)__vm_resolve("vm_fs", 5);
  g[2].name = "*";
  g[2].handle = VM_EMPTY;
  long a = vm_spawn(talker, 0, g, 3, 0, 0, scratch);
  long ra = a < 0 ? a : vm_join(a);
  long b = vm_spawn(talker, 0, g, 0, 0, 0, scratch);
  printf("granted: %ld, refused: %ld\n", ra, b);
  return 0;
}
"#;

/// The last run's notes, through the exports the page reads them by.
fn read_notes() -> String {
    let len = temen_browser::temen_notes_len();
    // SAFETY: the stash holds `len` bytes until the next `temen_notes_len`.
    let bytes = unsafe { core::slice::from_raw_parts(temen_browser::temen_notes_ptr(), len) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// #2219 — a card's child program through the exports the page calls: `temen_link_child_libs` links it
/// against the resident libraries to run as a child and stages it, and the next `temen_link_run_libs`
/// grants it by name. Granted what its `printf` imports, the child prints; granted nothing, it is
/// refused, because it binds what it imports strictly, and the run's notes name the import.
/// The run takes what was staged: a second run without staging has no `talker` to spawn.
#[test]
fn a_card_child_links_through_the_exports_and_a_refusal_is_noted() {
    let _exports = ffi::lock();
    let (Some(libc), Some(heap), Some(parent), Some(child)) = (
        asset("pg_libc.temeno"),
        asset("pg_heap.temeno"),
        program_unit_text(SPAWNS_TALKER),
        program_unit_text(TALKER),
    ) else {
        eprintln!("SKIP: chibicc.temen / pg_libc.temeno / pg_heap.temeno not built");
        return;
    };
    let handles = open_pg_units(&libc, &heap);
    let name = "talker";
    let rc = temen_browser::temen_link_child_libs(
        handles.as_ptr(),
        handles.len(),
        child.as_ptr(),
        child.len(),
        name.as_ptr(),
        name.len(),
    );
    assert_eq!(
        rc,
        0,
        "link the child: status {}",
        temen_browser::temen_status()
    );
    let first = link_run_libs(&handles, &parent);
    let notes = read_notes();
    assert_eq!(
        first,
        (
            0,
            STATUS_OK,
            "talker\ngranted: 7, refused: -22\n".to_string()
        ),
        "notes: {notes}"
    );
    assert!(
        notes.starts_with("refused a child: no grant satisfies its import `vm_fs`"),
        "{notes}"
    );
    assert_eq!(
        link_run_libs(&handles, &parent),
        (1, STATUS_OK, "no talker\n".to_string()),
        "the first run took the staged child"
    );
    for h in handles {
        temen_browser::temen_link_lib_close(h);
    }
}
