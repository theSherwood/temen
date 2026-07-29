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

/// The self-host demo directory (`cc1_main.c`, the emit-object libc, and the self-host prelude).
#[cfg(target_os = "linux")]
fn selfhost_dir() -> PathBuf {
    repo_root().join("crates/svm-run/demos/chibicc_selfhost")
}

/// Compile one real self-host `.c` file to a linkable unit via `--emit-object`, force-including the
/// self-host prelude (`selfhost_prelude.h`) that closes chibicc's own parser gaps against modern
/// glibc's `<stdlib.h>` (the `strtoul`/`atoi`/… implicit-declaration wall — see the prelude header).
/// `include_dirs` are passed as chibicc's **joined** `-I<dir>` form (its parser reads `argv+2`, so a
/// spaced `-I dir` is a no-op — a real cc1 TU that isn't a sibling of `chibicc.h` needs this).
#[cfg(target_os = "linux")]
fn emit_object_real(cfile: &Path, include_dirs: &[PathBuf], tag: &str) -> LinkUnit {
    let irfile = std::env::temp_dir().join(format!(
        "svm_clink_real_{}_{}.svm",
        tag.replace(['.', '/'], "_"),
        std::process::id()
    ));
    let prelude = selfhost_dir().join("selfhost_prelude.h");
    let mut args: Vec<String> = vec![
        "-cc1".into(),
        "-include".into(),
        prelude.to_str().unwrap().into(),
    ];
    for dir in include_dirs {
        args.push(format!("-I{}", dir.to_str().unwrap())); // joined form — chibicc reads argv+2
    }
    args.extend([
        "--emit-object".into(),
        "-cc1-input".into(),
        cfile.to_str().unwrap().into(),
        "-cc1-output".into(),
        irfile.to_str().unwrap().into(),
        cfile.to_str().unwrap().into(),
    ]);
    let status = Command::new(chibicc())
        .args(&args)
        .status()
        .expect("run chibicc");
    assert!(status.success(), "chibicc --emit-object failed on {tag}");
    let ir = std::fs::read_to_string(&irfile).unwrap();
    let m = svm_text::parse_module(&ir).unwrap_or_else(|e| panic!("parse {tag}: {e:?}"));
    let exports = m.exports.iter().map(|e| (e.name.clone(), e.func)).collect();
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

/// Compile one of chibicc's **own** upstream source files (`frontend/chibicc/<name>`) to a linkable
/// unit, on real self-host source rather than a crafted string, so the tests exercise the actual
/// shared globals (`ty_int`, …) and cross-TU call graph the compiler is written with.
///
/// **Linux-only** (like `svm-llvm` and the self-host build scripts): `chibicc.h` pulls system headers
/// (`<assert.h>`, `<stdio.h>`, …) and chibicc's default include search path is Linux-hardcoded
/// (`/usr/include/x86_64-linux-gnu`, main.c) — macOS/Windows runners have no such path, so compiling
/// chibicc's own source is a Linux operation here. The crafted-string tests carry no `#include` and
/// stay cross-platform.
#[cfg(target_os = "linux")]
fn object_unit_real(name: &str) -> LinkUnit {
    let src_dir = repo_root().join("frontend/chibicc");
    let cfile = src_dir.join(name);
    emit_object_real(&cfile, &[src_dir.clone()], name)
}

/// Compile the cc1 **entry** TU — `crates/svm-run/demos/chibicc_selfhost/cc1_main.c`, which replaces
/// `main.c` for the `-cc1` slice (defines `main` + the globals `main.c` owned, drives
/// tokenize→preprocess→parse→codegen_ir). It lives outside `frontend/chibicc`, so it needs both the
/// chibicc source dir (for `chibicc.h`) and its own dir on the include path.
#[cfg(target_os = "linux")]
fn object_unit_cc1_entry() -> LinkUnit {
    let src_dir = repo_root().join("frontend/chibicc");
    let cfile = selfhost_dir().join("cc1_main.c");
    emit_object_real(&cfile, &[src_dir, selfhost_dir()], "cc1_main.c")
}

/// Compile the **emit-object guest libc** (`emit_libc.c`) — the reusable, intrinsic-free core of the
/// libc the linked cc1 runs on (task #20, increment 2b). Needs the chibicc, self-host, and Postgres-
/// shim dirs on the include path.
#[cfg(target_os = "linux")]
fn object_unit_emit_libc() -> LinkUnit {
    let cfile = selfhost_dir().join("emit_libc.c");
    let dirs = [
        repo_root().join("frontend/chibicc"),
        selfhost_dir(),
        repo_root().join("crates/svm-run/demos/postgres"),
    ];
    emit_object_real(&cfile, &dirs, "emit_libc.c")
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

/// Compile a crafted fixture `.c` file (repo-relative path) to a linkable unit via `--emit-object`.
/// Like [`object_unit`] but sourced from a file so larger units (the emit-object libc) live as real C
/// on disk. Fixtures carry no system `#include` (only the compiler-provided `<stdarg.h>`), so this is
/// cross-platform — unlike [`object_unit_real`], which compiles chibicc's own Linux-header source.
fn object_unit_file(rel_path: &str) -> LinkUnit {
    let full = repo_root().join(rel_path);
    let src =
        std::fs::read_to_string(&full).unwrap_or_else(|e| panic!("read fixture {rel_path}: {e}"));
    let tag = full
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fixture")
        .to_string();
    object_unit(&tag, &src)
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

/// **Run a linked multi-TU program through `_start` under the powerbox** — the slice this suite was
/// building toward. The entry unit's `main` reads a global defined in another unit, and its own
/// stack frame (an address-taken local array) must sit **above every unit's data**. A whole-program
/// build bakes the data-SP as `i64.const data_end` (this unit's own top), but after linking the
/// provider's data is stacked *above* the entry unit — so that baked SP would drop `main`'s frame
/// on top of `provider`'s `base_val`. `--emit-object` instead emits `data.top`, which the linker
/// resolves to the post-link top of all data; the frame clears everything and the read is correct.
#[test]
fn links_and_runs_under_powerbox() {
    use svm_run::{instantiate, Outcome, RunConfig, Value};

    // main sums a stack array (0²+1²+…+7² = 140) and adds a cross-TU global (1234) → 1374. The
    // array is address-taken, so it lives on the data stack at the `_start` SP; a wrong SP (inside
    // the provider's data) would corrupt `base_val` and change the result.
    let main_unit = object_unit(
        "pbmain",
        "extern int base_val;\n\
         int main(void) {\n\
         \x20 int a[8];\n\
         \x20 for (int i = 0; i < 8; i++) a[i] = i * i;\n\
         \x20 int s = 0;\n\
         \x20 for (int i = 0; i < 8; i++) s += a[i];\n\
         \x20 return s + base_val;\n\
         }\n",
    );
    let provider = object_unit("pbprov", "int base_val = 1234;\n");

    // The entry unit is linked **first** so its `_start` is merged function 0 (the powerbox entry).
    let linked = link(&[main_unit, provider]).expect("link entry + provider");
    svm_verify::verify_module(&linked).expect("verify linked powerbox program");

    // Sanity: the `data.top` SP really is above the entry unit's own data — i.e. the provider's data
    // was stacked on top, so a baked `data_end` SP would have collided. `powerbox_entry_sp` is the
    // value the linker rewrote `data.top` to.
    let sp = svm_ir::powerbox_entry_sp(&linked);
    let data_top = linked
        .data
        .iter()
        .map(|d| d.offset + d.bytes.len() as u64)
        .max()
        .unwrap_or(0);
    assert!(sp >= data_top, "data-stack base clears all data");

    // Runs identically on the tree-walk oracle and the JIT (§18), and returns main's 1374.
    let inst = instantiate(linked).expect("instantiate");
    let run = inst.run_diff(&RunConfig::default()).expect("run _start");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(1374)]));
}

/// The **argv path** through a linked `_start`: `main(int, char**)` builds its `argv[]` array at the
/// data-stack base (`data.top` in emit-object mode — the blocks that a wrong SP would scribble over
/// the linked data). Here `main` writes a cross-TU greeting to `stdout` `argc` times, proving the
/// argv scaffold, a cross-TU data read, and a powerbox cap (`write`) all compose in one linked run.
#[test]
fn links_and_runs_under_powerbox_with_argv() {
    use svm_run::{instantiate, Backend, Outcome, RunConfig, Value};

    let main_unit = object_unit(
        "avmain",
        "extern char *msg;\n\
         extern long msglen;\n\
         int write(int fd, void *buf, unsigned long n);\n\
         int main(int argc, char **argv) {\n\
         \x20 (void)argv;\n\
         \x20 for (int i = 0; i < argc; i++) write(1, msg, msglen);\n\
         \x20 return argc;\n\
         }\n",
    );
    let provider = object_unit(
        "avprov",
        "char *msg = \"hi\\n\";\n\
         long msglen = 3;\n",
    );

    // `write` is a powerbox capability, not a cross-TU function: no unit defines it, so it lowers to
    // a `call.sym "write"` the *host* binds to the stdout stream at instantiation. `link_with_manifest`
    // retains such unresolved names as a merged import manifest (vs plain `link`, which fails closed).
    let linked =
        svm_ir::link_with_manifest(&[main_unit, provider]).expect("link argv entry + provider");
    svm_verify::verify_module(&linked).expect("verify");
    assert!(
        linked.imports.iter().any(|i| i.name == "write"),
        "write retained as a host-bound manifest import: {:?}",
        linked.imports.iter().map(|i| &i.name).collect::<Vec<_>>()
    );

    // Two argv entries → the loop runs twice → "hi\n" twice; `main` returns argc (2).
    let cfg = RunConfig {
        args: vec![b"prog".to_vec(), b"x".to_vec()],
        ..RunConfig::default()
    };
    let inst = instantiate(linked).expect("instantiate");
    let run = inst
        .run_with_caps(Backend::TreeWalk, &cfg, &[])
        .expect("run _start with argv");
    assert_eq!(run.stdout, b"hi\nhi\n", "wrote the cross-TU msg argc times");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(2)]));
}

/// **A self-host libc TU for the emit-object path, run under the powerbox** (task #20, increment 1).
/// The whole point of #20 is that chibicc's `--emit-object` backend — unlike the LLVM on-ramp — does
/// *not* synthesize `malloc`/`memcpy`/`printf`, so running a compiled-by-chibicc program needs a libc
/// chibicc itself can compile. This links two emit-object units: `mini_libc.c` (a from-scratch libc:
/// bump `malloc`, the `mem`/`str` family, and a **varargs** `printf` that writes through the powerbox
/// `write` cap) and `demo1.c` (a consumer whose `main` malloc's an array, copies a string, and prints
/// a deterministic line). Every libc call crosses the unit boundary and resolves at link; `write`
/// stays a host-bound manifest import. The linked program then runs through `_start` on interp==JIT,
/// so this pins that the libc *mechanism* the full self-host needs — varargs printf + heap + cross-TU
/// linking + a powerbox cap — composes end-to-end through emit-object. Later increments grow the libc
/// toward the surface that runs all ~9 cc1 TUs (the clang-tuned aggregator does not compile here).
#[test]
fn links_and_runs_emit_libc_under_powerbox() {
    use svm_run::{instantiate, Outcome, RunConfig, Value};

    // Entry unit first so its `_start` is merged function 0 (the powerbox entry); the libc unit has
    // no `main`, so it contributes no `_start`.
    let demo = object_unit_file("crates/svm/tests/fixtures/emit_libc/demo1.c");
    let libc = object_unit_file("crates/svm/tests/fixtures/emit_libc/mini_libc.c");

    // `demo` imports `malloc`/`printf`/`strcpy`/`strlen` (cross-TU, resolved to direct calls); `libc`
    // imports only `write` (a powerbox cap). `link_with_manifest` resolves the cross-TU calls and
    // retains `write` as a host-bound manifest import.
    let linked = svm_ir::link_with_manifest(&[demo, libc]).expect("link demo + emit-object libc");
    assert!(
        linked.imports.iter().any(|i| i.name == "write"),
        "write retained as a host-bound manifest import: {:?}",
        linked.imports.iter().map(|i| &i.name).collect::<Vec<_>>()
    );
    assert!(
        !linked
            .imports
            .iter()
            .any(|i| i.name == "malloc" || i.name == "printf"),
        "libc calls resolved to direct cross-TU calls, not left as imports"
    );
    svm_verify::verify_module(&linked).expect("verify linked emit-object libc program");

    // Runs `_start` on the tree-walk oracle and the JIT (§18); `run_diff` asserts their stdout/stderr
    // agree. `main` returns the array sum (0²+…+7² = 140); the printf line is a fixed oracle.
    let inst = instantiate(linked).expect("instantiate");
    let run = inst
        .run_diff(&RunConfig::default())
        .expect("run _start on interp+jit");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(140)]));
    assert_eq!(
        run.stdout, b"sum=140 first=0 last=49 len=5 str=hello hex=ff char=!\n",
        "malloc + strcpy/strlen + varargs printf composed through the emit-object libc"
    );
}

/// The mechanism on **unmodified upstream chibicc source**: `type.c` is the TU that defines the
/// shared `Type *` globals (`ty_int`/`ty_void`/…) as pointers to anonymous `Type` compound literals.
/// Under `--emit-object` it publishes each as a data symbol and lowers each pointer to a `data.ptr …
/// self` relocation — the exact cross-TU data the self-host build shares. It links + verifies on its
/// own (its libc calls — `strcmp`, `calloc`, … — stay retained manifest imports via
/// `link_with_manifest`), so this pins that real source flows through emit-object → link, while the
/// crafted tests above pin the runtime read behavior.
///
/// Linux-only: compiling chibicc's own source needs its Linux system-header path (see
/// [`object_unit_real`]).
#[cfg(target_os = "linux")]
#[test]
fn real_chibicc_type_tu_emits_and_links() {
    let type_tu = object_unit_real("type.c");
    let data_names: Vec<&str> = type_tu
        .data_exports
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    for g in ["ty_int", "ty_void", "ty_long", "ty_char", "ty_bool"] {
        assert!(
            data_names.contains(&g),
            "{g} exported as a data symbol: {data_names:?}"
        );
    }
    // Each `ty_* = &(Type){…}` is a data→data pointer the linker resolves and clears.
    assert!(
        !type_tu.module.data_ptrs.is_empty(),
        "type.c's Type structs carry data.ptr relocations"
    );
    let linked = svm_ir::link_with_manifest(&[type_tu]).expect("link real type.c");
    assert!(
        linked.data_ptrs.is_empty(),
        "data.ptr relocations resolved and cleared at link"
    );
    svm_verify::verify_module(&linked).expect("verify real type.c");
}

/// **Multiple real cc1 TUs linked together.** Four of chibicc's own translation units —
/// `type` (defines `ty_int`/…), `hashmap`, `unicode`, `strings` — compiled separately with
/// `--emit-object` and linked into one module. `type`'s data symbols and each unit's file-local
/// `static`s coexist without collision, cross-TU calls resolve to direct calls, and the compiler's
/// libc (`calloc`, `memcpy`, …) stays a retained host-bound manifest — the module links and verifies.
///
/// This is the "drive the chibicc TUs through emit-object + link" result on real source. Running the
/// *whole* cc1 additionally needs a self-host libc for the emit-object path (`tokenize.c` wants
/// `strtoul` declared; the runtime needs `malloc`/`printf`/file I/O) — a separate slice; every
/// linking mechanism it stands on (cross-TU functions, cross-TU data, the `data.top` stack base) is
/// proven here and above.
///
/// Linux-only: compiling chibicc's own source needs its Linux system-header path (see
/// [`object_unit_real`]).
#[cfg(target_os = "linux")]
#[test]
fn links_multiple_real_chibicc_tus() {
    let units = [
        object_unit_real("type.c"),
        object_unit_real("hashmap.c"),
        object_unit_real("unicode.c"),
        object_unit_real("strings.c"),
    ];
    // `type.c` publishes the shared `Type` globals the others link against.
    let type_data: Vec<&str> = units[0]
        .data_exports
        .iter()
        .map(|(n, _)| n.as_str())
        .collect();
    assert!(
        type_data.contains(&"ty_int"),
        "type.c exports ty_int: {type_data:?}"
    );

    let linked = svm_ir::link_with_manifest(&units).expect("link four real cc1 TUs");
    svm_verify::verify_module(&linked).expect("verify the linked cc1 subset");

    // The shared globals survive into the merged module's data-export table at their window offsets.
    for g in ["ty_int", "ty_void"] {
        assert!(
            linked.data_exports.iter().any(|e| e.name == g),
            "{g} present in the linked module's data exports"
        );
    }
    // Any unresolved names left are host-bound libc (a manifest), never a dangling cross-TU call.
    for imp in &linked.imports {
        assert!(
            !imp.name.is_empty(),
            "retained manifest import has a name (host-bound libc)"
        );
    }
}

/// **All ~9 cc1 translation units compile, link, and verify under `--emit-object`** (task #20,
/// increment 2a). The whole `-cc1` slice — the eight upstream TUs (`tokenize`, `preprocess`, `parse`,
/// `type`, `codegen_ir`, `hashmap`, `unicode`, `strings`) plus the `cc1_main.c` entry that replaces
/// `main.c` — each compiled separately by chibicc itself and linked into one module. This is the
/// compile-and-link half of "chibicc compiles its own source": every cross-TU function and data symbol
/// resolves, and only libc names survive as host-bound manifest imports (the increment-2b runtime).
///
/// Getting here needed the self-host prelude (`selfhost_prelude.h`): modern glibc's `<stdlib.h>` hides
/// `strtoul`/`atoi`/`strtold`/… behind ISO C23 `__isoc23_*` redirect + attribute forms chibicc's own
/// parser doesn't resolve, so those calls were implicitly declared (a hard error) — the prelude
/// force-includes plain prototypes. (The LLVM on-ramp never hit this: clang parses the redirects.)
///
/// Linux-only, like the other real-source tests (chibicc's system-header path).
#[cfg(target_os = "linux")]
#[test]
fn all_cc1_tus_compile_link_and_verify_under_emit_object() {
    // Entry unit first, so its `main`-derived `_start` is merged function 0 (the powerbox entry the
    // increment-2b run will use); the order is otherwise immaterial to linking.
    let mut units = vec![object_unit_cc1_entry()];
    for tu in [
        "tokenize.c",
        "preprocess.c",
        "parse.c",
        "type.c",
        "codegen_ir.c",
        "hashmap.c",
        "unicode.c",
        "strings.c",
    ] {
        units.push(object_unit_real(tu));
    }
    // The three stdio streams are the *only* data symbols no cc1 TU defines (they belong to the libc's
    // stdio) — and a cross-TU **data** reference must resolve to a concrete address, so unlike a libc
    // *function* it can't be left as a manifest import. A minimal unit defines them, standing in for
    // the increment-2b libc's stdio globals; with it, the whole cc1 links. (Verified: `stdin`/`stdout`/
    // `stderr` are the complete unresolved-data set — every `ty_*`/`opt_*`/`include_paths`/`base_file`
    // resolves cross-TU.)
    units.push(object_unit(
        "stdio_globals",
        "void *stdin;\nvoid *stdout;\nvoid *stderr;\n",
    ));

    let linked = svm_ir::link_with_manifest(&units).expect("link all cc1 TUs");
    svm_verify::verify_module(&linked).expect("verify the whole linked cc1");

    // The entry unit contributes a `_start` powerbox entry at function 0 (its `main` wrapped by the
    // emit-object `_start` that carries `data.top`), so the linked module is a runnable shape.
    assert!(
        linked
            .exports
            .iter()
            .any(|e| e.name == "_start" && e.func == 0),
        "cc1_main's _start is merged function 0: {:?}",
        linked
            .exports
            .iter()
            .map(|e| (&e.name, e.func))
            .collect::<Vec<_>>()
    );
    // The shared `Type *` globals from type.c survived cross-TU into the merged data exports.
    assert!(
        linked.data_exports.iter().any(|e| e.name == "ty_int"),
        "ty_int present in the linked cc1's data exports"
    );
    // Everything left unresolved is host-bound libc (a manifest), never a dangling cross-TU symbol.
    // This is exactly the surface the increment-2b emit-object libc must supply: the stdio + string
    // helpers as guest C (`fopen`/`vfprintf`/`strlen`/…) over the powerbox syscall + memory caps
    // (`exit`/`vm_map`). The heap allocator is NOT here: each TU that `#include <stdlib.h>` gets
    // chibicc's *bundled* header, whose `static malloc`/`calloc`/`free`/`realloc` grow the window via
    // `__vm_map` inline — so `malloc` never appears as a cross-unit symbol at all.
    let imports: Vec<&str> = linked.imports.iter().map(|i| i.name.as_str()).collect();
    for name in &imports {
        assert!(!name.is_empty(), "retained manifest import has a name");
    }
    for expected in ["fopen", "vfprintf", "strlen", "exit", "vm_map"] {
        assert!(
            imports.contains(&expected),
            "libc/cap `{expected}` retained as a manifest import (increment-2b target); imports: {imports:?}"
        );
    }
}

/// **The emit-object guest libc's reusable core compiles under `--emit-object` and is intrinsic-free**
/// (task #20, increment 2b). `emit_libc.c` aggregates the pieces the linked cc1 runs on that carry no
/// dependency on the LLVM on-ramp: the bundled-`<stdlib.h>` allocator, `mem`/`str`, ctype
/// (`__ctype_b_loc`), errno, `strtod`, and the fd-backed + `open_memstream` stdio (`chibicc_extra.c`,
/// with its allocator now guarded out under the bundled header). It defines the real `stdin`/`stdout`/
/// `stderr` the 2a link stubbed.
///
/// The guard this pins: **no svm-llvm intrinsic survives.** The on-ramp's `os_shim.c`/`printf_shim.c`
/// reach the host via `__vm_stream_write`/`__vm_host_call`/`__vm_cap_resolve` and format floats via
/// `__vm_fmt_*` — names chibicc's own `--emit-object` codegen does not lower (it knows `__vm_map`/
/// `__vm_jit_`/… but not those). So the emit-object libc must replace both bottom-edge shims (an os
/// layer over plain `call.sym "write"`/`"read"` + an fs cap, and a `__vm_fmt_*`-free `vfprintf`) — the
/// next slice. This test compiles the intrinsic-free core today and fails if an on-ramp intrinsic ever
/// leaks back in, or if the shared `chibicc_extra.c` allocator guard regresses (a redefinition).
///
/// Linux-only, like the other real-source tests.
#[cfg(target_os = "linux")]
#[test]
fn emit_object_libc_core_compiles_and_is_intrinsic_free() {
    let libc = object_unit_emit_libc();

    // The reusable surface is exported as guest C.
    for f in [
        "memcpy",
        "memset",
        "strlen",
        "strchr",
        "__ctype_b_loc",
        "__errno_location",
        "strtoul",
        "strtod",
        "fopen",
        "fwrite",
        "fread",
        "fputc",
        "fclose",
        "open_memstream",
    ] {
        assert!(
            libc.exports.iter().any(|(n, _)| n == f),
            "emit-object libc exports `{f}`"
        );
    }
    // The real stdio streams (fd 0/1/2), defined here rather than stubbed as in the 2a link.
    for g in ["stdin", "stdout", "stderr"] {
        assert!(
            libc.data_exports.iter().any(|(n, _)| n == g),
            "emit-object libc defines the `{g}` global"
        );
    }
    // The load-bearing guard: not one on-ramp svm-llvm intrinsic is referenced. If `os_shim.c`/
    // `printf_shim.c` were pulled in (or the bundled-header allocator guard regressed), one of these
    // would appear as an unbindable import and the linked cc1 could never instantiate.
    for imp in &libc.module.imports {
        let n = imp.name.as_str();
        assert!(
            !n.starts_with("__vm_fmt")
                && n != "__vm_stream_write"
                && n != "__vm_stream_read"
                && n != "__vm_host_call"
                && n != "__vm_cap_resolve",
            "emit-object libc must not reference on-ramp intrinsic `{n}`"
        );
    }
    // What it *does* leave undefined is only host-bindable caps + the two documented bottom-edge
    // seams (fs `open`/`close`, the printf formatters) — never an intrinsic.
    let imports: Vec<&str> = libc
        .module
        .imports
        .iter()
        .map(|i| i.name.as_str())
        .collect();
    for cap in ["write", "read"] {
        assert!(
            imports.contains(&cap),
            "emit-object libc reaches stdout/stdin via the powerbox `{cap}` cap; imports: {imports:?}"
        );
    }
}
