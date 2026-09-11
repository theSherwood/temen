//! **The linker merges per-unit debug info** (#1392): `temen_ir::link`/`link_with_manifest` used to
//! drop it (`debug_info: None`), which made separate compilation unusable for a *debugger* — the
//! motivating consumer being the browser C card, where compiling the source-level libc into every
//! translation unit costs ~9–12 s and moving it into a prebuilt library unit is a 66× win, but only
//! if stepping survives the link.
//!
//! Merging is the same reindexing the linker already does for code, applied to the debug tables:
//! function indices shift by the unit's function base, file/type indices by the running length of the
//! merged tables (nested type references with them), and a global's absolute `Fixed` window address by
//! the unit's data base — while frame-relative (`Window`) and value-indexed (`Ssa`) locations are
//! relocation-invariant and must *not* move. The load-bearing invariant a debugger depends on, pinned
//! below: a global's debug address equals its relocated data-symbol address.

use temen_ir::{link, synth_manifest_start, DebugInfo, LinkUnit, TypeDef, VarLoc, GLOBAL_SCOPE};

fn unit(src: &str, exports: &[(&str, u32)], data_exports: &[(&str, u64)]) -> LinkUnit {
    LinkUnit {
        module: temen_text::parse_module_debug(src).expect("parse unit"),
        exports: exports.iter().map(|(n, i)| (n.to_string(), *i)).collect(),
        data_exports: data_exports
            .iter()
            .map(|(n, o)| (n.to_string(), *o))
            .collect(),
    }
}

/// Unit A (the "library"): two functions, a writable global at 1024, and debug info exercising every
/// reindexed shape — a file, two function names, locs in both functions, a 3-entry type table with a
/// nested pointer/aggregate reference, a window local, an SSA local, and a `Fixed` global.
const A: &str = r#"memory 16

data 1024 "\x07\x00\x00\x00"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
export 0 func "a_one" 0
export 1 func "a_two" 1

debug.file 0 "a.c"
debug.fname 0 "a_one"
debug.fname 1 "a_two"
debug.loc 0 0 0 0 10 1
debug.loc 1 0 0 0 20 2
debug.type 0 base "int" signed 4
debug.type 1 ptr "int *" 0 8
debug.type 2 agg "struct P" 8
debug.field 2 "x" 0 0
debug.var 0 "loc_a" win -8 "int" 0
debug.var 1 "sc_a" ssa 0 "int" 0
debug.var global "g_a" fixed 1024 "int" 0
"#;

/// Unit B (the "program"): one function that calls into A by symbol, its own global at 2048, and its
/// own single-file debug info whose file index (0) and type ids (0, 1) must both shift on merge.
const B: &str = r#"memory 16

data 2048 "\x01\x00\x00\x00"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 0
  v2 = call.sym "a_one" (i64) -> (i64) v1 (v0)
  return v2
  }
}
export 0 func "b_main" 0

debug.file 0 "b.c"
debug.fname 0 "b_main"
debug.loc 0 0 1 0 30 3
debug.type 0 base "char" signed 1
debug.type 1 array "char [4]" 0 4
debug.var 0 "loc_b" win -16 "char [4]" 1
debug.var global "g_b" fixed 2048 "char [4]" 1
"#;

fn linked() -> temen_ir::Module {
    let units = [
        unit(A, &[("a_one", 0), ("a_two", 1)], &[("g_a", 1024)]),
        unit(B, &[("b_main", 0)], &[("g_b", 2048)]),
    ];
    link(&units).expect("units link")
}

/// Function, file and type indices all shift by their unit's base; nested type references shift with
/// the table; `GLOBAL_SCOPE` is a sentinel and stays put.
#[test]
fn merged_debug_info_reindexes_every_table() {
    let m = linked();
    let di: &DebugInfo = m
        .debug_info
        .as_ref()
        .expect("linked module carries debug info");

    // Files append in unit order; B's single file lands at index 1.
    assert_eq!(di.files, vec!["a.c".to_string(), "b.c".to_string()]);

    // A has 2 funcs, so every B funcidx shifts by 2; B's file index shifts by 1.
    let loc = |func: u32, line: u32| di.locs.iter().find(|l| l.func == func && l.line == line);
    assert!(loc(0, 10).is_some(), "A's func 0 loc keeps its index");
    assert!(loc(1, 20).is_some(), "A's func 1 loc keeps its index");
    let b_loc = loc(2, 30).expect("B's func 0 loc shifted to 2");
    assert_eq!(b_loc.file, 1, "B's file index shifted past A's table");
    assert_eq!(
        (b_loc.block, b_loc.inst, b_loc.col),
        (0, 1, 3),
        "the rest of a loc is untouched"
    );

    // Function names shift identically.
    let fname = |f: u32| {
        di.func_names
            .iter()
            .find(|n| n.func == f)
            .map(|n| n.name.as_str())
    };
    assert_eq!(fname(0), Some("a_one"));
    assert_eq!(fname(1), Some("a_two"));
    assert_eq!(
        fname(2),
        Some("b_main"),
        "B's function name shifted with its function"
    );

    // Types append; A's 3 entries keep ids 0..2, B's two land at 3..4 with `elem` reindexed.
    assert_eq!(di.types.len(), 5, "both type tables merged: {:?}", di.types);
    match &di.types[1] {
        TypeDef::Pointer { pointee, .. } => {
            assert_eq!(*pointee, 0, "A's pointee is already correct")
        }
        other => panic!("expected A's pointer at 1, got {other:?}"),
    }
    match &di.types[2] {
        TypeDef::Aggregate { fields, .. } => {
            assert_eq!(fields[0].ty, 0, "A's field type unchanged")
        }
        other => panic!("expected A's aggregate at 2, got {other:?}"),
    }
    match &di.types[4] {
        TypeDef::Array { elem, count, .. } => {
            assert_eq!(*elem, 3, "B's array element type shifted past A's table");
            assert_eq!(*count, 4);
        }
        other => panic!("expected B's array at 4, got {other:?}"),
    }

    // Vars: locals follow their function, globals stay GLOBAL_SCOPE, type ids shift.
    let var = |name: &str| di.vars.iter().find(|v| v.name == name).expect(name);
    assert_eq!(var("loc_a").func, 0);
    assert_eq!(var("sc_a").func, 1);
    assert_eq!(var("loc_b").func, 2, "B's local followed its function");
    assert_eq!(var("g_a").func, GLOBAL_SCOPE, "a global is not a funcidx");
    assert_eq!(var("g_b").func, GLOBAL_SCOPE);
    assert_eq!(var("loc_b").type_id, Some(4), "B's var type id shifted");
    assert_eq!(var("g_a").type_id, Some(0));

    // Frame-relative and value-indexed locations are relocation-invariant.
    assert_eq!(var("loc_a").loc, VarLoc::Window { off: -8 });
    assert_eq!(var("loc_b").loc, VarLoc::Window { off: -16 });
    assert_eq!(var("sc_a").loc, VarLoc::Ssa { value: 0 });
}

/// The invariant a debugger actually depends on: a global's `Fixed` debug address is relocated by the
/// same data base as its data segment, so it equals the merged address of its own data symbol.
#[test]
fn a_globals_debug_address_tracks_its_relocated_data_symbol() {
    let m = linked();
    let di = m.debug_info.as_ref().expect("debug info");
    let sym = |name: &str| {
        m.data_exports
            .iter()
            .find(|d| d.name == name)
            .unwrap_or_else(|| panic!("data export {name}"))
            .offset
    };
    let fixed = |name: &str| match di.vars.iter().find(|v| v.name == name).expect(name).loc {
        VarLoc::Fixed { addr } => addr,
        ref other => panic!("{name} should be Fixed, got {other:?}"),
    };
    assert_eq!(fixed("g_a"), sym("g_a"), "unit 0's global");
    assert_eq!(
        fixed("g_b"),
        sym("g_b"),
        "unit 1's global, past the page-aligned base"
    );
    assert!(
        fixed("g_b") > fixed("g_a"),
        "the second unit's data was relocated upward: {} vs {}",
        fixed("g_b"),
        fixed("g_a")
    );
}

/// A release link produces no debug info at all — the merge adds nothing to the stripped path, so an
/// unstripped `link` stays exactly what it was before this merge existed. Note this uses
/// `parse_module` rather than `parse_module_debug`: the latter *synthesizes* debug info for the text
/// itself (one loc per op, one var per SSA value), so a module parsed that way is never debug-free.
#[test]
fn no_debug_info_in_no_debug_info_out() {
    let plain = |src: &str, exports: &[(&str, u32)]| LinkUnit {
        // Strip the `debug.*` directives, and parse with `parse_module` — both halves matter.
        module: temen_text::parse_module(
            &src.lines()
                .filter(|l| !l.starts_with("debug."))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("parse unit"),
        exports: exports.iter().map(|(n, i)| (n.to_string(), *i)).collect(),
        data_exports: Vec::new(),
    };
    let units = [
        plain(A, &[("a_one", 0), ("a_two", 1)]),
        plain(B, &[("b_main", 0)]),
    ];
    let m = link(&units).expect("link");
    assert!(
        m.debug_info.is_none(),
        "nothing in, nothing out; got {:?}",
        m.debug_info
    );
}

/// The merged info survives the text waist (`print_module` → `parse_module_debug`) — the path
/// c_interpret's debug session takes, since it launches from IR text — and survives
/// `synth_manifest_start`, which shifts every funcidx by one for the prepended `_start` while leaving
/// `GLOBAL_SCOPE` alone.
#[test]
fn merged_debug_info_survives_the_text_waist_and_start_synthesis() {
    let m = linked();
    let round = temen_text::parse_module_debug(&temen_text::print_module(&m)).expect("re-parse");
    assert_eq!(
        round.debug_info, m.debug_info,
        "printing and re-parsing the linked module preserves the merged debug info"
    );

    let entry = m
        .exports
        .iter()
        .find(|e| e.name == "b_main")
        .expect("entry")
        .func;
    let started = synth_manifest_start(m, entry, false).expect("synth _start");
    let di = started
        .debug_info
        .as_ref()
        .expect("debug info survives synthesis");
    let var = |name: &str| di.vars.iter().find(|v| v.name == name).expect(name);
    assert_eq!(
        var("loc_a").func,
        1,
        "every funcidx shifted by the prepended _start"
    );
    assert_eq!(var("loc_b").func, 3);
    assert_eq!(
        var("g_a").func,
        GLOBAL_SCOPE,
        "the global sentinel is not shifted"
    );
    assert!(
        di.locs.iter().any(|l| l.func == 3 && l.line == 30),
        "B's loc shifted too"
    );
}
