//! **#1746 — LLVM-translated link units.** `TranslateOptions::link_unit` translates a library the
//! linker can place anywhere among other units: its globals are addressed relative to its own data
//! (`data.self`, `data.ptr`), a call to a function it does not define is a `call.sym` import, a global
//! it only declares is a `data.sym`, and only external-linkage names are exported.
//!
//! Before, a translated unit baked every global to a fixed window address, so it ran correctly only
//! as the first unit of a link, and a call to an undefined function failed translation: two
//! LLVM-translated units could not link each other at all. Here two of them do, in either order, on
//! the interpreter and the JIT (`run_diff`). No clang: the units are inline textual LLVM IR, so this
//! runs in every job.

use temen_interp::Value;
use temen_ir::LinkUnit;

/// The library: a counter it increments on every `lib_add`, a table holding a pointer to that
/// counter and to `lib_add` itself, and a `static` helper of its own.
const LIB: &str = r#"
@lib_counter = global i64 0
@lib_table = global [2 x ptr] [ptr @lib_counter, ptr @lib_add]

define i64 @lib_add(i64 %a, i64 %b) {
entry:
  %c = load i64, ptr @lib_counter
  %c1 = add i64 %c, 1
  store i64 %c1, ptr @lib_counter
  %r = add i64 %a, %b
  ret i64 %r
}

define internal i64 @helper(i64 %x) {
entry:
  %r = add i64 %x, 1000
  ret i64 %r
}

define i64 @lib_helped(i64 %x) {
entry:
  %r = call i64 @helper(i64 %x)
  ret i64 %r
}
"#;

/// The program side: calls into the library, reads its counter directly and through a pointer stored
/// in its own data, calls through the library's function-pointer table, and has a `static` `helper`
/// of the same name as the library's.
const APP: &str = r#"
@lib_counter = external global i64
@lib_table = external global [2 x ptr]
@app_state = global i64 5
@app_counter_ptr = global ptr @lib_counter

declare i64 @lib_add(i64, i64)
declare i64 @lib_helped(i64)

define internal i64 @helper(i64 %x) {
entry:
  %r = mul i64 %x, 3
  ret i64 %r
}

define i64 @run() {
entry:
  %a = call i64 @lib_add(i64 2, i64 3)
  %b = call i64 @lib_add(i64 %a, i64 10)
  %c = load i64, ptr @lib_counter
  %p = load ptr, ptr @app_counter_ptr
  %c2 = load i64, ptr %p
  %fslot = getelementptr [2 x ptr], ptr @lib_table, i64 0, i64 1
  %f = load ptr, ptr %fslot
  %d = call i64 %f(i64 100, i64 1)
  %cslot = getelementptr [2 x ptr], ptr @lib_table, i64 0, i64 0
  %cp = load ptr, ptr %cslot
  %c3 = load i64, ptr %cp
  %h = call i64 @helper(i64 7)
  %hl = call i64 @lib_helped(i64 4)
  %s = load i64, ptr @app_state
  %t1 = mul i64 %c, 100
  %t2 = mul i64 %c2, 1000
  %t3 = mul i64 %c3, 10000
  %t4 = mul i64 %h, 100000
  %t5 = mul i64 %s, 10000000
  %t6 = mul i64 %d, 100000000
  %u1 = add i64 %b, %t1
  %u2 = add i64 %u1, %t2
  %u3 = add i64 %u2, %t3
  %u4 = add i64 %u3, %t4
  %u5 = add i64 %u4, %t5
  %u6 = add i64 %u5, %t6
  %u7 = add i64 %u6, %hl
  ret i64 %u7
}
"#;

/// `run`'s value: `b = 15`; the counter is 2 after two `lib_add`s, read directly (`c`) and through the
/// app's pointer (`c2`); 3 after the call through the table (`c3`, which reads it through the
/// library's own pointer); `d = 101`; the app's own `helper(7) = 21`; `lib_helped(4)` reaches the
/// library's `helper` (1004); `app_state = 5`.
const EXPECT: i64 =
    15 + 2 * 100 + 2 * 1000 + 3 * 10000 + 21 * 100000 + 5 * 10000000 + 101 * 100000000 + 1004;

fn link_unit(src: &str) -> temen_ir::Module {
    let opts = temen_llvm::TranslateOptions {
        link_unit: true,
        ..Default::default()
    };
    temen_llvm::translate_ll_str_with_options(src, opts)
        .expect("translate link unit")
        .module
}

fn unit(module: temen_ir::Module) -> LinkUnit {
    LinkUnit {
        exports: module
            .exports
            .iter()
            .map(|e| (e.name.clone(), e.func))
            .collect(),
        data_exports: module.data_exports.clone(),
        module,
    }
}

/// Link `units`, enter at `entry`, and run it on the interpreter and the JIT (which must agree).
fn link_and_run(units: Vec<LinkUnit>, entry: &str) -> temen_run::Outcome {
    let linked = temen_ir::link_with_manifest(&units).expect("link");
    let idx = linked.resolve_export(entry).expect("entry export");
    let program = temen_ir::synth_manifest_start(linked, idx, false).expect("powerbox wrap");
    temen_run::instantiate(program)
        .expect("instantiate")
        .run_diff(&temen_run::RunConfig::default())
        .expect("run (interp == JIT)")
        .outcome
}

/// The #1746 regression: two LLVM-translated units link each other, in either order, so each unit
/// runs both at data base 0 and relocated above the other.
#[test]
fn two_translated_units_link_in_either_order() {
    for order in [[LIB, APP], [APP, LIB]] {
        let units = order.iter().map(|src| unit(link_unit(src))).collect();
        assert_eq!(
            link_and_run(units, "run"),
            temen_run::Outcome::Returned(vec![Value::I64(EXPECT)]),
            "first unit: {}",
            if order[0] == LIB { "library" } else { "app" }
        );
    }
}

/// A link unit exports its external-linkage functions and globals; a C `static` (and a private
/// string literal) stays inside the unit, so two units with a same-named `static` link.
#[test]
fn a_link_unit_exports_only_external_names() {
    let lib = link_unit(LIB);
    let funcs: Vec<&str> = lib.exports.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(funcs, ["lib_add", "lib_helped"]);
    let data: Vec<&str> = lib.data_exports.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(data, ["lib_counter", "lib_table"]);

    // The table's two slots are link-form: a pointer to the unit's own counter, and `lib_add`'s index.
    assert_eq!(lib.data_ptrs.len(), 1);
    assert!(matches!(
        lib.data_ptrs[0].target,
        temen_ir::DataPtrTarget::SelfOff(_)
    ));
    assert_eq!(lib.data_funcrefs.len(), 1);
    assert_eq!(lib.data_funcrefs[0].name, "lib_add");

    // The app imports what it calls and reads, rather than failing on it.
    let app = link_unit(APP);
    let imports: Vec<&str> = app.imports.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(imports, ["lib_add", "lib_helped"]);
    assert!(app.data_ptrs.iter().any(|p| matches!(
        &p.target,
        temen_ir::DataPtrTarget::Sym { name, .. } if name == "lib_counter"
    )));
}

/// Without the option, the same library keeps today's whole-module form: every defined function
/// exported, no data symbols, and its addresses baked (no link forms left for a linker to fill in).
#[test]
fn without_the_option_a_unit_is_unchanged() {
    let lib = temen_llvm::translate_ll_str_with_options(LIB, Default::default())
        .expect("translate")
        .module;
    temen_verify::verify_module(&lib).expect("a whole module verifies");
    let funcs: Vec<&str> = lib.exports.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(funcs, ["lib_add", "helper", "lib_helped"]);
    assert!(
        lib.data_exports.is_empty() && lib.data_ptrs.is_empty() && lib.data_funcrefs.is_empty()
    );
}

/// A link unit reaches the powerbox like any unit: its `vm_page_size` import is bound when the
/// linked program is instantiated.
#[test]
fn a_link_unit_keeps_its_capability_imports() {
    const PAGES: &str = r#"
declare i64 @__vm_page_size()

define i64 @page_size() {
entry:
  %p = call i64 @__vm_page_size()
  ret i64 %p
}
"#;
    let outcome = link_and_run(vec![unit(link_unit(PAGES))], "page_size");
    let temen_run::Outcome::Returned(vals) = outcome else {
        panic!("page_size exited: {outcome:?}");
    };
    assert!(
        matches!(vals.as_slice(), [Value::I64(p)] if *p > 0),
        "the host page size: {vals:?}"
    );
}

/// Each refusal is a clean translate-time error, never a unit that runs misplaced.
#[test]
fn a_link_unit_refuses_what_it_cannot_relocate() {
    let refusal = |src: &str| match temen_llvm::translate_ll_str_with_options(
        src,
        temen_llvm::TranslateOptions {
            link_unit: true,
            ..Default::default()
        },
    ) {
        Ok(_) => panic!("translated:\n{src}"),
        Err(e) => format!("{e:?}"),
    };

    // A pointer to a `static` function in static data: a `data.funcref` names an exported function.
    let e = refusal(
        r#"
@table = global [1 x ptr] [ptr @f]
define internal i64 @f() {
entry:
  ret i64 1
}
"#,
    );
    assert!(e.contains("static function `@f`"), "{e}");

    // A program: a link unit is a library.
    let e = refusal(
        r#"
define i32 @main() {
entry:
  ret i32 0
}
"#,
    );
    assert!(e.contains("`main`"), "{e}");

    // The address of a function the unit does not define (it can call it, not take its address).
    let e = refusal(
        r#"
declare i64 @elsewhere()
@slot = global ptr null
define void @take() {
entry:
  store ptr @elsewhere, ptr @slot
  ret void
}
"#,
    );
    assert!(
        e.contains("address of undefined function `@elsewhere`"),
        "{e}"
    );
}
