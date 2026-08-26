//! In-window dynamic linking, milestone 0: **compile-time (static) linking of a function symbol.**
//!
//! A unit `caller` references another unit `add` by *name* (`call.sym "add"`). The loader resolves
//! the name to `add`'s function index and `temen_ir::resolve_imports_with` rewrites the `CallImport`
//! into a **direct `call`** — exactly what a static linker does (symbol → concrete call). By the time
//! the verifier and both backends see the module, it's an ordinary closed module; "linking" was a
//! source-to-source rewrite, above the TCB, re-verified like everything else. (Dynamic, separately-
//! compiled linking — `call.dyn` through a `Jit.install` slot — is the next milestone.)

use temen_interp::Value;
use temen_ir::{Resolved, ResolvedCap};

/// Two "units" in one module: `add(a,b)=a+b` at index 0, and `caller(a,b)` (index 1) that calls
/// `add` **by name**. The dummy `v2` is the (unused) capability-handle operand `call.import` carries;
/// resolving to a `Func` drops it.
const TWO_UNITS: &str = "\
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.add v0 v1
  return v2
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.const 0
  v3 = call.sym \"add\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
  }
}
";

/// Resolve + verify, then run `caller` (entry 1) on interp + JIT with `args`; assert they agree and
/// return the i32 result.
fn link_and_run(resolver: impl FnMut(&str) -> Option<Resolved>, args: &[i32]) -> i64 {
    let m = temen_text::parse_module(TWO_UNITS).expect("parse");
    assert_eq!(m.imports.len(), 1, "one named import: \"add\"");
    // The compile-time link step: rewrite call.sym "add" → a direct call to add's index.
    let linked = temen_ir::resolve_imports_with(&m, resolver).expect("resolve");
    assert!(linked.imports.is_empty(), "imports lowered away");
    // No CallImport survives; it became a direct Call.
    assert!(
        linked.funcs[1].blocks[0]
            .insts
            .iter()
            .all(|i| !matches!(i, temen_ir::Inst::CallImport { .. })),
        "the import must be lowered to a direct call"
    );
    temen_verify::verify_module(&linked).expect("verify linked module");

    let ivals: Vec<Value> = args.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let interp = temen_interp::run(&linked, 1, &ivals, &mut fuel).expect("interp run");
    let jargs: Vec<i64> = args.iter().map(|&x| x as i64).collect();
    let jit = match temen_jit::compile_and_run(&linked, 1, &jargs).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

/// The core: `caller` reaches `add` purely by name, resolved at link time to a direct call.
#[test]
fn caller_links_to_add_by_name() {
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[3, 4]),
        7
    );
    assert_eq!(
        link_and_run(|n| (n == "add").then_some(Resolved::Func(0)), &[100, -1]),
        99
    );
}

/// An **unresolved** symbol is fail-closed (the loader can't find `add`).
#[test]
fn unresolved_symbol_fails_closed() {
    let m = temen_text::parse_module(TWO_UNITS).expect("parse");
    let err = temen_ir::resolve_imports_with(&m, |_| None).expect_err("must fail closed");
    assert_eq!(err, temen_ir::ImportError::Unresolved("add".into()));
}

/// A **signature mismatch** can't produce a type-unsafe call: linking feeds the re-verifier, never
/// bypasses it. `sym` is declared `(i32,i32)->i32` but resolved to a `(i64)->i64` function, so the
/// rewritten direct call has the wrong arg count/types — and `verify_module` rejects the linked
/// module. (This is the link-time symbol-signature check, enforced by re-verification, not trust.)
#[test]
fn signature_mismatch_is_caught_by_reverify() {
    let src = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.const 0
  v3 = call.sym \"sym\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
  }
}
";
    let m = temen_text::parse_module(src).expect("parse");
    let linked = temen_ir::resolve_imports_with(&m, |_| Some(Resolved::Func(0))).expect("resolve");
    assert!(
        temen_verify::verify_module(&linked).is_err(),
        "a signature-mismatched link must be rejected by re-verification"
    );
}

/// The generalized pass still does the §7 capability case (`Resolved::Cap`) — a sanity check that the
/// `resolve_imports` (cap-only) path is unchanged by delegating through `resolve_imports_with`.
#[test]
fn capability_resolution_still_works() {
    let src = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = call.sym \"write\" (i32) -> (i32) v0 (v1)
  return v2
  }
}
";
    let m = temen_text::parse_module(src).expect("parse");
    let linked = temen_ir::resolve_imports_with(&m, |_| {
        Some(Resolved::Cap(ResolvedCap { type_id: 0, op: 1 }))
    })
    .expect("resolve");
    // The import lowered to a call.cap (not a direct call).
    assert!(linked.funcs[0].blocks[0].insts.iter().any(|i| matches!(
        i,
        temen_ir::Inst::CapCall {
            type_id: 0,
            op: 1,
            ..
        }
    )));
}

// ---------------------------------------------------------------------------------------------
// Milestone 1: the static linker — concatenate *separate* units into one program (temen_ir::link).
// ---------------------------------------------------------------------------------------------

use temen_ir::{link, LinkUnit};

/// Run entry `idx` of an already-verified module on interp + JIT, assert they agree, return the i32.
fn run_entry(m: &temen_ir::Module, idx: u32, args: &[i32]) -> i64 {
    let ivals: Vec<Value> = args.iter().map(|&x| Value::I32(x)).collect();
    let mut fuel = 10_000_000u64;
    let interp = temen_interp::run(m, idx, &ivals, &mut fuel).expect("interp run");
    let jargs: Vec<i64> = args.iter().map(|&x| x as i64).collect();
    let jit = match temen_jit::compile_and_run(m, idx, &jargs).expect("jit compile") {
        temen_jit::JitOutcome::Returned(v) => v,
        other => panic!("jit did not return: {other:?}"),
    };
    let iv = match interp[0] {
        Value::I32(x) => x as i64,
        other => panic!("unexpected interp value {other:?}"),
    };
    assert_eq!(iv as u32 as u64, jit[0] as u32 as u64, "interp != jit");
    iv
}

fn unit(src: &str, exports: &[(&str, u32)]) -> LinkUnit {
    LinkUnit {
        module: temen_text::parse_module(src).expect("parse unit"),
        exports: exports.iter().map(|(n, i)| (n.to_string(), *i)).collect(),
        ..Default::default()
    }
}

const MATH_UNIT: &str = "\
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.add v0 v1
  return v2
  }
}
";

/// `app` calls `add` by name; it lives in a **separate** unit (`math`). The linker concatenates them
/// (app's functions reindexed after math's) and resolves the import to a direct call.
const APP_UNIT: &str = "\
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.const 0
  v3 = call.sym \"add\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
  }
}
";

#[test]
fn links_two_separate_units_into_one_program() {
    let linked = link(&[unit(MATH_UNIT, &[("add", 0)]), unit(APP_UNIT, &[])]).expect("link");
    // math's `add` is function 0; app's `main` is function 1 (reindexed after math).
    assert_eq!(linked.funcs.len(), 2);
    assert!(linked.imports.is_empty(), "all imports resolved");
    temen_verify::verify_module(&linked).expect("verify linked program");
    // app's main (entry 1) calls into math's add across the unit boundary.
    assert_eq!(run_entry(&linked, 1, &[3, 4]), 7);
    assert_eq!(run_entry(&linked, 1, &[40, 2]), 42);
}

/// A three-unit chain proves reindexing across more than two units: `app` → `add`, where `add` itself
/// lives after an unrelated `pad` unit, so its global index is shifted and the import still resolves.
#[test]
fn links_across_a_reindexing_offset() {
    let pad = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  return v0
  }
}
"; // an unrelated unit so `math` lands at a non-zero base
    let linked = link(&[
        unit(pad, &[("pad", 0)]),
        unit(MATH_UNIT, &[("add", 0)]), // global index 1
        unit(APP_UNIT, &[]),            // global index 2; its "add" → 1
    ])
    .expect("link");
    temen_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 2, &[10, 5]),
        15,
        "app(entry 2) → add at global index 1"
    );
}

/// An import no unit exports is fail-closed.
#[test]
fn link_unresolved_symbol_fails_closed() {
    let err = link(&[unit(APP_UNIT, &[])]).expect_err("nothing exports add");
    assert_eq!(err, temen_ir::LinkError::Unresolved("add".into()));
}

/// Two units exporting the same symbol is fail-closed.
#[test]
fn link_duplicate_symbol_fails_closed() {
    let err = link(&[
        unit(MATH_UNIT, &[("add", 0)]),
        unit(MATH_UNIT, &[("add", 0)]),
    ])
    .expect_err("two `add`s");
    assert_eq!(err, temen_ir::LinkError::DuplicateSymbol("add".into()));
}

// ---------------------------------------------------------------------------------------------
// Milestone 2: cross-unit data symbols via the **self-describing** link forms — `export … data`
// (provider), `data.sym "name" <addend>` / `data.self <offset>` (consumer). No relocation
// side-table: the symbol rides in the instruction, so the linker rewrites it 1:1 to `i64.const`.
// ---------------------------------------------------------------------------------------------

/// 16 bytes of padding data so the unit that follows lands at a **non-zero** data base — making the
/// relocation observable (a coincidental base of 0 would prove nothing).
const PAD16: &str = "\
memory 16
data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"
func (i32) -> (i32) {
block 0 (v0: i32) {
  return v0
  }
}
";

/// Build a link unit from text, taking its function and data exports from the module's own
/// `export`/`export … data` directives — so this exercises the text round-trip of both.
fn text_unit(src: &str) -> LinkUnit {
    let module = temen_text::parse_module(src).expect("parse unit");
    let exports = module
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();
    let data_exports = module
        .data_exports
        .iter()
        .map(|e| (e.name.clone(), e.offset))
        .collect();
    LinkUnit {
        module,
        exports,
        data_exports,
    }
}

/// A **cross-unit data symbol**: `store` exports the byte 42 as data symbol "answer" (`export … data`);
/// `load` reads it with `data.sym "answer"`. The linker places `store`'s data at a non-zero base
/// (after `pad`), records "answer" at that window address, and rewrites `load`'s `data.sym` to an
/// `i64.const` of it — so `load` reads the byte wherever the linker put it. Proves data moved + ref
/// followed, with the symbol carried in the instruction (no relocation table).
#[test]
fn cross_unit_data_symbol_resolves() {
    let store = text_unit(
        "memory 16\ndata 0 \"\\x2a\"\nexport 0 data \"answer\" 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    let load = text_unit(
        "memory 16\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.sym \"answer\" 0\n\
         \x20 v2 = i32.load8_u v1\n\
         \x20 return v2\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), store, load]).expect("link");
    assert!(
        !linked
            .funcs
            .iter()
            .flat_map(|f| &f.blocks)
            .flat_map(|b| &b.insts)
            .any(|i| matches!(i, temen_ir::Inst::DataSym { .. })),
        "every data.sym is rewritten to a const at link"
    );
    temen_verify::verify_module(&linked).expect("verify");
    // `load` is the 3rd unit's function → global index 2.
    assert_eq!(
        run_entry(&linked, 2, &[0]),
        42,
        "read the relocated cross-unit datum"
    );
}

/// The **addend** rides through: `store` exports a 4-byte array as "arr"; `load` reads element 2 with
/// `data.sym "arr" 8` (offset 8 = index 2 of `i32`). The linker resolves to `addr(arr) + 8`.
#[test]
fn cross_unit_data_symbol_addend() {
    let store = text_unit(
        // arr = {10, 20, 30, 40} as i32 LE.
        "memory 16\ndata 0 \"\\x0a\\x00\\x00\\x00\\x14\\x00\\x00\\x00\\x1e\\x00\\x00\\x00\\x28\\x00\\x00\\x00\"\n\
         export 0 data \"arr\" 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    let load = text_unit(
        "memory 16\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.sym \"arr\" 8\n\
         \x20 v2 = i32.load v1\n\
         \x20 return v2\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), store, load]).expect("link");
    temen_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 2, &[0]),
        30,
        "arr[2] via the data.sym addend"
    );
}

/// **Own-data address**: a unit references its *own* data with `data.self <offset>`; linked after
/// `pad`, its data moves to a non-zero base and the `data.self` resolves to `base + offset`, so the
/// reference still lands on its data — proving the base was applied to segment and reference alike.
#[test]
fn self_data_address_resolves() {
    let me = text_unit(
        "memory 16\n\
         data 0 \"\\x07\"\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.self 0\n\
         \x20 v2 = i32.load8_u v1\n\
         \x20 return v2\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), me]).expect("link");
    temen_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 1, &[0]),
        7,
        "own data ref follows the relocation"
    );
}

/// A `data.sym` naming a symbol no unit exports is fail-closed — the linker's `Unresolved`, the same
/// guarantee a missing function symbol gets, now for data.
#[test]
fn unresolved_data_symbol_fails_closed() {
    let u = text_unit(
        "memory 16\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.sym \"nowhere\" 0\n\
         \x20 v2 = i32.load8_u v1\n\
         \x20 return v2\n\
           }\n\
         }\n",
    );
    assert_eq!(
        link(&[u]),
        Err(temen_ir::LinkError::Unresolved("nowhere".into()))
    );
}

/// The text round-trips: `data.sym`/`data.self`/`export … data` print and re-parse identically.
#[test]
fn data_link_forms_round_trip() {
    let src = "memory 16\n\
               data 0 \"\\x2a\"\n\n\
               export 0 data \"answer\" 0\n\n\
               func (i64) -> (i64) {\n\
               block 0 (v0: i64) {\n\
               \x20 v1 = data.sym \"answer\" 8\n\
               \x20 v2 = data.self 4\n\
               \x20 v3 = i64.add v1 v2\n\
               \x20 return v3\n\
                 }\n\
               }\n";
    let m = temen_text::parse_module(src).expect("parse");
    let printed = temen_text::print_module(&m);
    let m2 = temen_text::parse_module(&printed).expect("re-parse");
    assert_eq!(m, m2, "print ∘ parse is identity for the data link forms");
    assert_eq!(m.data_exports.len(), 1);
    assert_eq!(m.data_exports[0].name, "answer");
}

// ---------------------------------------------------------------------------------------------
// data → data: a pointer baked into a global's *own* initializer (`int *p = &g;`). The pointer
// lives in static data, not in an instruction, so it rides a `data.ptr` slot the linker patches
// into the data image — the data→data twin of `data.self`/`data.sym`.
// ---------------------------------------------------------------------------------------------

/// **Own-data pointer** (`int *p = &g;`, same TU): a unit stores a pointer *to its own datum* in its
/// data image. `g` (byte 99) sits at offset 0; an 8-byte pointer slot at offset 8 is fixed up by
/// `data.ptr 8 self 0`. Linked behind `pad`, the datum and the slot both move to base 16, and the
/// linker writes `base+0` into the slot — so loading the pointer (`i64.load`) and dereferencing it
/// yields 99 wherever the data landed. Proves the *stored bytes* were relocated, not just an instr.
#[test]
fn data_ptr_self_pointer_resolves() {
    let me = text_unit(
        "memory 16\n\
         data 0 \"\\x63\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 8 self 0\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.self 8\n\
         \x20 v2 = i64.load v1\n\
         \x20 v3 = i32.load8_u v2\n\
         \x20 return v3\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), me]).expect("link");
    // The relocation was applied and consumed — a runnable module carries no `data.ptr`.
    assert!(linked.data_ptrs.is_empty(), "data.ptr resolved and cleared");
    temen_verify::verify_module(&linked).expect("verify");
    assert_eq!(
        run_entry(&linked, 1, &[0]),
        99,
        "load the stored self-pointer and deref it"
    );
}

/// **Cross-unit data pointer** (`extern int g; int *p = &g;`): unit `store` exports datum "answer"
/// (byte 55); unit `hold` keeps an 8-byte pointer to it, fixed up by `data.ptr 0 sym "answer" 0`.
/// The linker writes `addr(answer)` into `hold`'s slot, so `hold`'s function loads the pointer and
/// dereferences the *other unit's* datum. The data-image twin of `cross_unit_data_symbol_resolves`.
#[test]
fn data_ptr_cross_unit_pointer_resolves() {
    let store = text_unit(
        "memory 16\ndata 0 \"\\x37\"\nexport 0 data \"answer\" 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    let hold = text_unit(
        "memory 16\n\
         data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 0 sym \"answer\" 0\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.self 0\n\
         \x20 v2 = i64.load v1\n\
         \x20 v3 = i32.load8_u v2\n\
         \x20 return v3\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), store, hold]).expect("link");
    assert!(linked.data_ptrs.is_empty());
    temen_verify::verify_module(&linked).expect("verify");
    // hold's function is the 3rd unit → global index 2.
    assert_eq!(
        run_entry(&linked, 2, &[0]),
        55,
        "deref a cross-unit pointer stored in data"
    );
}

/// The **addend** rides through the data pointer: `store` exports `arr = {10,20,30,40}` as "arr";
/// `hold` keeps a pointer to `arr[2]` via `data.ptr 0 sym "arr" 8`. The linker writes `addr(arr)+8`
/// into the slot, so dereferencing the stored pointer reads `30`.
#[test]
fn data_ptr_addend_rides() {
    let store = text_unit(
        "memory 16\ndata 0 \"\\x0a\\x00\\x00\\x00\\x14\\x00\\x00\\x00\\x1e\\x00\\x00\\x00\\x28\\x00\\x00\\x00\"\n\
         export 0 data \"arr\" 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    let hold = text_unit(
        "memory 16\n\
         data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 0 sym \"arr\" 8\n\
         func (i32) -> (i32) {\n\
         block 0 (v0: i32) {\n\
         \x20 v1 = data.self 0\n\
         \x20 v2 = i64.load v1\n\
         \x20 v3 = i32.load v2\n\
         \x20 return v3\n\
           }\n\
         }\n",
    );
    let linked = link(&[unit(PAD16, &[]), store, hold]).expect("link");
    temen_verify::verify_module(&linked).expect("verify");
    assert_eq!(run_entry(&linked, 2, &[0]), 30, "&arr[2] baked into data");
}

/// The text round-trips: both `data.ptr <at> self <off>` and `data.ptr <at> sym "<name>" <addend>`
/// print and re-parse identically (print ∘ parse is identity, including the `data_ptrs` table).
#[test]
fn data_ptr_forms_round_trip() {
    let src = "memory 16\n\
               data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
               data.ptr 0 self 8\n\
               data.ptr 8 sym \"g\" 4\n\
               func (i32) -> (i32) {\n\
               block 0 (v0: i32) {\n  return v0\n  }\n}\n";
    let m = temen_text::parse_module(src).expect("parse");
    assert_eq!(m.data_ptrs.len(), 2);
    let printed = temen_text::print_module(&m);
    let m2 = temen_text::parse_module(&printed).expect("re-parse");
    assert_eq!(m, m2, "print ∘ parse is identity for data.ptr");
}

/// A `data.ptr … sym` naming a symbol no unit exports is fail-closed — the linker's `Unresolved`,
/// the same guarantee `data.sym` gets, now for a pointer stored in data.
#[test]
fn unresolved_data_ptr_fails_closed() {
    let u = text_unit(
        "memory 16\n\
         data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 0 sym \"nowhere\" 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    assert_eq!(
        link(&[u]),
        Err(temen_ir::LinkError::Unresolved("nowhere".into()))
    );
}

/// A `data.ptr` slot with no covering data segment is a malformed unit — fail-closed with
/// `BadDataPtr` (the frontend must emit an 8-byte placeholder covering `[at, at+8)`).
#[test]
fn data_ptr_outside_segment_fails_closed() {
    let u = text_unit(
        "memory 16\n\
         data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 100 self 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    );
    assert_eq!(link(&[u]), Err(temen_ir::LinkError::BadDataPtr { at: 100 }));
}

/// A `data.ptr` **surviving** into a would-be-runnable module fails verification: unlike the
/// instruction link forms (which trap at execution), a data pointer has no execution site, so its
/// placeholder bytes would be read unpatched. `verify_module` is the fail-closed gate.
#[test]
fn surviving_data_ptr_fails_verify() {
    let m = temen_text::parse_module(
        "memory 16\n\
         data 0 \"\\x00\\x00\\x00\\x00\\x00\\x00\\x00\\x00\"\n\
         data.ptr 0 self 0\n\
         func (i32) -> (i32) {\nblock 0 (v0: i32) {\n  return v0\n  }\n}\n",
    )
    .expect("parse");
    assert_eq!(
        temen_verify::verify_module(&m),
        Err(temen_verify::VerifyError::UnlinkedDataPtr { at: 0 })
    );
}

// ---------------------------------------------------------------------------------------------
// Milestone 3: dynamic linking — resolve a symbol to a call.dyn TABLE SLOT (Resolved::Slot).
// A separately-compiled unit reaches a function it doesn't share an index space with, by slot.
// ---------------------------------------------------------------------------------------------

/// `main` imports `F` by name and the loader resolves it to **table slot 1** — not a direct call but a
/// `call.dyn` through the shared function table (how a separately-compiled unit reaches another).
/// `F` (slot 1) is `a*2 + b`; a decoy `G` sits at slot 0. The handle placeholder const (`i32.const 0`)
/// is patched to `1` and reused as the index, so a passing `F(10,3)=23` (not `G`'s 7) proves the slot.
#[test]
fn import_resolves_to_a_call_indirect_slot() {
    let src = "\
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.sub v0 v1
  return v2
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.const 2
  v3 = i32.mul v0 v2
  v4 = i32.add v3 v1
  return v4
  }
}

func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = i32.const 0
  v3 = call.sym \"F\" (i32, i32) -> (i32) v2 (v0, v1)
  return v3
  }
}
";
    let m = temen_text::parse_module(src).expect("parse");
    let linked =
        temen_ir::resolve_imports_with(&m, |n| (n == "F").then_some(temen_ir::Resolved::Slot(1)))
            .expect("resolve to slot");
    // The import became a `call.dyn`, and the handle const was patched to the slot (1).
    let insts = &linked.funcs[2].blocks[0].insts;
    assert!(
        matches!(insts[0], temen_ir::Inst::ConstI32(1)),
        "handle const patched to slot 1"
    );
    assert!(
        matches!(insts[1], temen_ir::Inst::CallIndirect { .. }),
        "import lowered to call.dyn, not a direct call"
    );
    temen_verify::verify_module(&linked).expect("verify");
    // main (entry 2) dispatches to slot 1 = F(a,b) = a*2+b; F(10,3) = 23 (G would give 7).
    assert_eq!(
        run_entry(&linked, 2, &[10, 3]),
        23,
        "reached F via the resolved slot"
    );
}

/// A `Slot` import whose handle operand isn't a `ConstI32` placeholder is fail-closed (the frontend
/// must emit one — it's patched to the slot and reused as the index).
#[test]
fn slot_import_requires_a_const_handle() {
    // The handle here is a block *parameter* (v0), not a const → SlotHandleNotConst.
    let src = "\
func (i32, i32) -> (i32) {
block 0 (v0: i32, v1: i32) {
  v2 = call.sym \"F\" (i32) -> (i32) v0 (v1)
  return v2
  }
}
";
    let m = temen_text::parse_module(src).expect("parse");
    let err = temen_ir::resolve_imports_with(&m, |_| Some(temen_ir::Resolved::Slot(0)))
        .expect_err("non-const handle must fail closed");
    assert_eq!(err, temen_ir::ImportError::SlotHandleNotConst);
}

/// The linker merges the units' impl surfaces (IMPORTS.md §3.2/OQ3): interfaces concatenate
/// with a per-unit index offset, offers reindex their interface reference and op funcidxs
/// through the same offsets as function exports, and an offer name colliding with any symbol
/// fails closed like a duplicate export.
#[test]
fn link_merges_impl_surfaces_across_units() {
    let provider = unit(
        "type 0 func (i32, i32) -> (i32)\n\
         type 1 interface { add: 0 }\n\
         export 0 interface \"adder\" 1 { add: 0 }\n\n\
         func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
           v2 = i32.add v0 v1\n\
           return v2\n\
           }\n\
         }\n",
        &[],
    );
    let other = unit(MATH_UNIT, &[("add", 0)]);
    // `other` first, so the provider's funcs and interface reindex across a nonzero offset.
    let m = temen_ir::link(&[other, provider]).expect("links");
    temen_verify::verify_module(&m).expect("merged module verifies");
    assert_eq!(
        m.types.len(),
        2,
        "type section merged (one Func + one Interface)"
    );
    let offer = m
        .resolve_impl_export("adder")
        .expect("offer survives the merge");
    assert_eq!(offer.interface, 1);
    assert_eq!(
        offer.ops,
        vec![1],
        "op funcidx reindexed past the first unit"
    );

    // An offer name colliding with a function export symbol fails closed.
    let clash = unit(
        "type 0 func (i32, i32) -> (i32)\n\
         type 1 interface { add: 0 }\n\
         export 0 interface \"add\" 1 { add: 0 }\n\n\
         func (i32, i32) -> (i32) {\n\
         block 0 (v0: i32, v1: i32) {\n\
           v2 = i32.add v0 v1\n\
           return v2\n\
           }\n\
         }\n",
        &[],
    );
    let named = unit(MATH_UNIT, &[("add", 0)]);
    assert!(
        matches!(
            temen_ir::link(&[named, clash]),
            Err(temen_ir::LinkError::DuplicateSymbol(n)) if n == "add"
        ),
        "offer/export name collision fails the link closed"
    );
}
