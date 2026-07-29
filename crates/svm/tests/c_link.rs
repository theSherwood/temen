//! **Separate compilation for the C frontend** (the native-`cc` model): compile each translation
//! unit on its own with `chibicc --emit-object`, then `svm_ir::link` the units into one program.
//!
//! This is the mechanism the whole-compiler self-host build stands on (SELFHOST_C.md §7): chibicc's
//! own source is ~9 TUs with cross-TU calls, and a whole-program-per-invocation backend can't see
//! across them. `--emit-object` makes each TU a **linkable unit** — every non-`static` function is
//! `export`ed, and a call to a function *declared but not defined* in the TU lowers to a
//! function-symbol import (`call.sym "name"`) instead of the generic capability import. The already
//! battle-tested static linker (`dynlink.rs`) then resolves each import to a direct cross-unit call.
//!
//! Two properties this pins that the unity-build alternative could not:
//!   * **`static` = internal linkage per unit.** A file-local `static` helper is never exported, so
//!     same-named statics in different TUs (chibicc has several, e.g. `eval2` in both parse.c and
//!     codegen_ir.c) never collide — no renaming, unlike a single amalgamated TU.
//!   * **the import signature matches the definition.** The `call.sym` carries the callee's real
//!     SVM signature (leading data-SP, then the C params), so the linker's direct call type-checks.
//!
//! Scope: cross-TU *functions* **and cross-TU *data*** — a global defined in one unit and read in
//! another. `--emit-object` publishes each non-`static` global as a data symbol (`export … data`),
//! materializes a global's own address as `data.self` and a cross-TU global's as `data.sym`, and
//! lowers a pointer initializer to a `data.ptr … self`/`sym` relocation (the data→data twin). The
//! linker places each unit's data in a non-overlapping window, resolves every symbol, and grows the
//! merged window to hold the stacked data. Here the linked functions are still run directly by index
//! (as `dynlink.rs` does); running a linked program through `_start` under the powerbox is the next
//! slice.
//!
//! `#![cfg(unix)]` — like `c_frontend.rs`, this builds the chibicc fork with `make`, and chibicc's
//! headers (`<glob.h>`, …) are POSIX; Windows lacks the toolchain, so the whole suite is Unix-only.
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_interp::Value;
use svm_ir::{link, LinkUnit};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

/// Build the chibicc fork once per test binary, returning the path to its binary.
fn chibicc() -> &'static Path {
    static CC: OnceLock<PathBuf> = OnceLock::new();
    CC.get_or_init(|| {
        let dir = repo_root().join("frontend/chibicc");
        let status = Command::new("make")
            .arg("-s")
            .current_dir(&dir)
            .status()
            .expect("run `make` to build the chibicc fork");
        assert!(status.success(), "chibicc build failed");
        dir.join("chibicc")
    })
    .as_path()
}

/// Compile one C source string to a **linkable unit** (`--emit-object`) and parse it, turning the
/// emitted `export` directives into the `LinkUnit::exports` the linker resolves against.
fn object_unit(tag: &str, src: &str) -> LinkUnit {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_clink_{tag}_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();

    let status = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-object",
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc --emit-object failed on:\n{src}");

    let ir = std::fs::read_to_string(&irfile).unwrap();
    let m = svm_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse {tag} unit: {e:?}\n{ir}"));
    let exports = m.exports.iter().map(|e| (e.name.clone(), e.func)).collect();
    // Cross-TU *data* symbols: a non-`static` global defined here (`export … data`) that another
    // unit reads via `data.sym`/`data.ptr … sym`. The linker places this unit's data at its window
    // base and records each symbol's absolute address for the consumers.
    let data_exports = m
        .data_exports
        .iter()
        .map(|e| (e.name.clone(), e.offset))
        .collect();
    LinkUnit {
        module: m,
        exports,
        data_exports,
    }
}

/// Run function `idx` of an already-verified module on interp **and** JIT with a leading data-SP
/// (`i64`) followed by `i32` args, assert the backends agree, and return the `i32` result. The SP is
/// above the module's data window; these functions do only SP arithmetic (no load/store), so any
/// in-range SP is inert — the point is the cross-unit *calls*, not memory.
fn run_i32(m: &svm_ir::Module, idx: u32, sp: i64, args: &[i32]) -> i32 {
    let mut ivals = vec![Value::I64(sp)];
    ivals.extend(args.iter().map(|&x| Value::I32(x)));
    let mut fuel = 10_000_000u64;
    let interp = svm_interp::run(m, idx, &ivals, &mut fuel).expect("interp run");
    let iv = match interp[0] {
        Value::I32(x) => x,
        other => panic!("unexpected interp value {other:?}"),
    };

    let mut jargs = vec![sp];
    jargs.extend(args.iter().map(|&x| x as i64));
    let jit = match svm_jit::compile_and_run(m, idx, &jargs).expect("jit compile") {
        svm_jit::JitOutcome::Returned(v) => v[0] as i32,
        other => panic!("jit did not return: {other:?}"),
    };
    assert_eq!(iv, jit, "interp != jit for entry {idx}");
    iv
}

/// The load-bearing proof: two translation units compiled **separately**, then linked. `app`'s
/// `compute` calls `math`'s `add3` across the unit boundary; `add3` in turn calls its own unit's
/// `helper` (twice) and file-local `static secret`. The linker resolves the `call.sym "add3"`
/// import to a direct call and reindexes `app`'s functions after `math`'s.
#[test]
fn links_two_separately_compiled_c_units() {
    let math = object_unit(
        "math",
        "int helper(int a, int b) { return a + b; }\n\
         static int secret(int x) { return x * 3; }\n\
         int add3(int a, int b, int c) { return helper(helper(a, b), c) + secret(0); }\n",
    );
    let app = object_unit(
        "app",
        "extern int add3(int a, int b, int c);\n\
         int compute(int a, int b) { return add3(a, b, 10) * 2; }\n",
    );

    // `math` exports `add3` and `helper`; `secret` is `static`, so it is NOT exported.
    let math_names: Vec<&str> = math.exports.iter().map(|(n, _)| n.as_str()).collect();
    assert!(
        math_names.contains(&"add3"),
        "add3 exported: {math_names:?}"
    );
    assert!(
        math_names.contains(&"helper"),
        "helper exported: {math_names:?}"
    );
    assert!(
        !math_names.contains(&"secret"),
        "static secret must stay internal: {math_names:?}"
    );
    // `app` imports `add3` (a function-symbol import), it does not define it.
    assert!(
        app.module.imports.iter().any(|i| i.name == "add3"),
        "app should carry an `add3` function-symbol import: {:?}",
        app.module
            .imports
            .iter()
            .map(|i| &i.name)
            .collect::<Vec<_>>()
    );

    let n_math = math.module.funcs.len() as u32;
    let linked = link(&[math, app]).expect("link the two units");
    assert!(
        linked.imports.is_empty(),
        "all imports resolved to direct calls"
    );
    svm_verify::verify_module(&linked).expect("verify the linked program");

    // `compute` is `app`'s only function, reindexed just past `math`'s functions.
    let compute = n_math; // math: [add3, secret, helper] → compute at index 3
                          // compute(a,b) = add3(a,b,10)*2 = (a+b+10)*2  (secret(0)=0 contributes nothing).
    let sp = 0x1_0000; // above the 64 KiB data window; unused (no load/store)
    assert_eq!(run_i32(&linked, compute, sp, &[3, 4]), 34);
    assert_eq!(run_i32(&linked, compute, sp, &[10, 5]), 50);
    assert_eq!(run_i32(&linked, compute, sp, &[-10, 0]), 0);
}

/// A three-unit chain proves reindexing across more than two units and that an import binds to the
/// *right* unit when an unrelated unit shifts everyone's base: `app` → `add3` (in `math`), with a
/// `pad` unit compiled first so `math` lands at a non-zero function base.
#[test]
fn links_a_three_unit_chain() {
    let pad = object_unit("pad", "int pad(int x) { return x; }\n");
    let math = object_unit(
        "math2",
        "int add3(int a, int b, int c) { return a + b + c; }\n",
    );
    let app = object_unit(
        "app2",
        "extern int add3(int a, int b, int c);\n\
         int compute(int a, int b) { return add3(a, b, 100); }\n",
    );
    let base = pad.module.funcs.len() as u32 + math.module.funcs.len() as u32;
    let linked = link(&[pad, math, app]).expect("link three units");
    svm_verify::verify_module(&linked).expect("verify");
    // app's compute is the last function; add3 lives after pad, so its global index is shifted.
    assert_eq!(run_i32(&linked, base, 0x1_0000, &[10, 5]), 115);
}

/// Fail-closed: a unit that calls an undefined, un-exported function fails the link (the linker's
/// existing `Unresolved`), not a silent miscompile — the same guarantee `dynlink.rs` pins, reached
/// now through the C frontend.
#[test]
fn unresolved_cross_unit_call_fails_closed() {
    let app = object_unit(
        "orphan",
        "extern int nowhere(int x);\n\
         int compute(int a) { return nowhere(a); }\n",
    );
    let err = link(&[app]).expect_err("nothing exports `nowhere`");
    assert_eq!(err, svm_ir::LinkError::Unresolved("nowhere".into()));
}

/// The data-stack pointer for a by-index run of a **data-touching** linked program: just above the
/// linker-placed data (16-byte aligned). Unlike the function tests (SP-only arithmetic), these
/// functions `load` from the window, so `svm_interp::run` materializes the data segments and the SP
/// must sit above them; the linker grew the window's power-of-two size to cover the data, leaving
/// slack for the leaf frame (asserted, so a future TU that fills the window fails loudly).
fn data_sp(linked: &svm_ir::Module) -> i64 {
    let data_top = linked
        .data
        .iter()
        .map(|d| d.offset + d.bytes.len() as u64)
        .max()
        .unwrap_or(0);
    let sp = (data_top + 15) & !15;
    let win = 1u64 << linked.memory.expect("linked data window").size_log2;
    assert!(
        sp + 4096 <= win,
        "leaf frame needs headroom above data (sp={sp}, win={win})"
    );
    sp as i64
}

/// **Cross-TU data symbol.** `provider` defines a non-`static` `int counter = 42`; `consumer` reads
/// it. The provider publishes `counter` as a data symbol (`export … data`) and the consumer's read
/// lowers to `data.sym "counter"` — no local storage for the `extern`. The linker places the
/// provider's data at its window base, records `counter`'s address, and rewrites the consumer's
/// `data.sym` to that constant, so the read lands on the relocated datum.
#[test]
fn cross_tu_data_symbol() {
    let provider = object_unit("dprov", "int counter = 42;\n");
    let consumer = object_unit(
        "dcons",
        "extern int counter;\n\
         int read_counter(void) { return counter; }\n",
    );

    // The provider exports `counter` as *data*; the consumer keeps no storage for the `extern`.
    assert!(
        provider.data_exports.iter().any(|(n, _)| n == "counter"),
        "counter exported as data: {:?}",
        provider.data_exports
    );
    assert!(
        consumer.module.data.is_empty()
            || consumer
                .module
                .data
                .iter()
                .all(|d| d.bytes.iter().all(|&b| b != 42)),
        "consumer must not carry its own copy of `counter`"
    );

    let n_prov = provider.module.funcs.len() as u32;
    let linked = link(&[provider, consumer]).expect("link data provider+consumer");
    assert!(linked.imports.is_empty(), "all symbols resolved");
    svm_verify::verify_module(&linked).expect("verify linked data program");

    // `read_counter` is the consumer's only function, reindexed past the provider's.
    assert_eq!(run_i32(&linked, n_prov, data_sp(&linked), &[]), 42);
}

/// **Cross-TU data → data pointer.** `provider` holds `struct T anon = {3,4}` and a pointer
/// `struct T *shared = &anon` — a `data.ptr … self` relocation (the pointer's bytes are patched to
/// `anon`'s window address at link). `consumer` reads `shared->tag` across the unit boundary
/// (`data.sym "shared"`, then a load-and-deref). Exercises both #507's `data.ptr self` (in the
/// provider) and the cross-TU `data.sym` (in the consumer) in one linked program.
#[test]
fn cross_tu_data_pointer() {
    let provider = object_unit(
        "pprov",
        "struct T { int tag; int size; };\n\
         struct T anon = { 3, 4 };\n\
         struct T *shared = &anon;\n",
    );
    let consumer = object_unit(
        "pcons",
        "struct T { int tag; int size; };\n\
         extern struct T *shared;\n\
         int read_tag(void) { return shared->tag; }\n",
    );

    // The provider's `shared` initializer is a data→data pointer: a `data.ptr` slot the linker
    // resolves and clears, so the linked module carries none.
    assert!(
        !provider.module.data_ptrs.is_empty(),
        "provider's `shared = &anon` is a data.ptr relocation"
    );

    let n_prov = provider.module.funcs.len() as u32;
    let linked = link(&[provider, consumer]).expect("link pointer provider+consumer");
    svm_verify::verify_module(&linked).expect("verify");
    assert!(
        linked.data_ptrs.is_empty(),
        "data.ptr resolved and cleared at link"
    );

    assert_eq!(run_i32(&linked, n_prov, data_sp(&linked), &[]), 3);
}

/// A **cross-TU `data.ptr … sym`**: `holder` keeps `int *p = &target` where `target` lives in
/// another unit — so the pointer's target is a *cross-unit symbol*, emitted as `data.ptr <at> sym
/// "target"` (#507's sym path), which the linker patches to `target`'s window address. `owner`
/// defines `int target = 55` and a reader that dereferences `p`.
#[test]
fn cross_tu_data_ptr_sym() {
    let owner = object_unit(
        "owner",
        "int target = 55;\n\
         extern int *p;\n\
         int read_via_p(void) { return *p; }\n",
    );
    let holder = object_unit(
        "holder",
        "extern int target;\n\
         int *p = &target;\n",
    );

    // `holder`'s `p = &target` names a symbol it does not define → a `data.ptr … sym` relocation.
    assert!(
        holder.module.data_ptrs.iter().any(|dp| matches!(
            &dp.target,
            svm_ir::DataPtrTarget::Sym { name, .. } if name == "target"
        )),
        "holder carries a `data.ptr … sym \"target\"`: {:?}",
        holder.module.data_ptrs
    );

    let n_owner = owner.module.funcs.len() as u32;
    let linked = link(&[owner, holder]).expect("link owner+holder");
    svm_verify::verify_module(&linked).expect("verify");
    // `read_via_p` is `owner`'s first function (owner linked first, index 0).
    assert_eq!(run_i32(&linked, 0, data_sp(&linked), &[]), 55);
    let _ = n_owner;
}

/// Fail-closed for **data**: a unit reading an `extern` global that no unit exports fails the link
/// with `Unresolved` (the same guarantee as an unresolved call), not a read of uninitialized memory.
#[test]
fn unresolved_cross_unit_data_fails_closed() {
    let app = object_unit(
        "dorphan",
        "extern int nowhere_data;\n\
         int compute(void) { return nowhere_data; }\n",
    );
    let err = link(&[app]).expect_err("nothing exports `nowhere_data`");
    assert_eq!(err, svm_ir::LinkError::Unresolved("nowhere_data".into()));
}

/// A file-local `static` global is **internal**: it is not published as a data symbol, so a
/// same-named `static` in another TU never collides — the data-side twin of the `static` function
/// property, and what lets chibicc's per-TU statics link without renaming.
#[test]
fn static_global_stays_internal() {
    let unit = object_unit(
        "dstatic",
        "static int secret = 7;\n\
         int visible = 9;\n\
         int get_secret(void) { return secret; }\n",
    );
    let names: Vec<&str> = unit.data_exports.iter().map(|(n, _)| n.as_str()).collect();
    assert!(names.contains(&"visible"), "visible exported: {names:?}");
    assert!(
        !names.contains(&"secret"),
        "static secret must stay internal: {names:?}"
    );
}
