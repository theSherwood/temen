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

#[cfg(target_os = "linux")]
#[path = "support/repo_root.rs"]
mod repo_root_mod;
use repo_root_mod::repo_root;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
#[cfg(target_os = "linux")]
use std::process::Stdio;
use std::sync::OnceLock;

use svm_interp::Value;
use svm_ir::{link, LinkUnit};

/// Turn an emit-object module into the `LinkUnit` the linker consumes — **via the binary object
/// wire** (`.svmo`, the v9 object dialect): encode with `encode_unit`, re-decode with
/// `decode_unit` (asserting round-trip identity on the real compiler output), and build the
/// export maps from the re-decoded module's in-band tables. Every chibicc-emitted unit in these
/// tests therefore travels the same bytes an on-disk object would — a unit the object dialect
/// can't carry fails here, not in a downstream consumer.
fn unit_via_svmo(m: svm_ir::Module, tag: &str) -> LinkUnit {
    let bytes = svm_encode::encode_unit(&m);
    let re =
        svm_encode::decode_unit(&bytes).unwrap_or_else(|e| panic!("decode {tag} object: {e:?}"));
    assert_eq!(m, re, "object wire changed the {tag} unit");
    let exports = re
        .exports
        .iter()
        .map(|e| (e.name.clone(), e.func))
        .collect();
    let data_exports = re
        .data_exports
        .iter()
        .map(|e| (e.name.clone(), e.offset))
        .collect();
    LinkUnit {
        module: re,
        exports,
        data_exports,
    }
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
    // Unique per call: the test binary runs tests in parallel threads (one process ⇒ one pid), and
    // several compile the same `tag` (e.g. `emit_libc.c` from three different tests). A pid+tag-only
    // name would let those concurrent compiles clobber each other's IR file mid-write — a real race
    // that surfaced as a libc unit with missing exports. The atomic id makes every output distinct,
    // matching what `object_unit` already does for the string-sourced units.
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let irfile = std::env::temp_dir().join(format!(
        "svm_clink_real_{}_{}_{id}.svm",
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
    unit_via_svmo(m, tag)
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
    emit_object_real(&cfile, std::slice::from_ref(&src_dir), name)
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
    // Cross-TU *data* symbols (`export … data` → `data.sym`/`data.ptr … sym`) ride the unit's
    // in-band tables through the object wire like everything else.
    unit_via_svmo(m, tag)
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
/// A stand-in **owner** for the shared bump-allocator state (`__svm_brk`/`__svm_committed`/
/// `__svm_page`/`__svm_grow_lock`). The emit-object cc1 TUs now *import* these four globals from the
/// libc (`stdlib.h`'s `__SVM_LIBC_EXTERN` sharing — one bump pointer across all units, so the per-TU
/// allocators don't hand out overlapping addresses), so a link of a *subset* that omits `emit_libc.c`
/// must still resolve them. Compiled without the self-host prelude, so these are plain exported data
/// symbols (not `extern`); the values mirror the libc's initializers (heap base 256 MiB). Subset tests
/// link+verify only, so only the *symbols* need to resolve.
#[cfg(target_os = "linux")]
fn alloc_state_stub() -> LinkUnit {
    object_unit(
        "alloc_state",
        "long __svm_brk = 268435456L;\n\
         long __svm_committed = 268435456L;\n\
         long __svm_page = 0;\n\
         int __svm_grow_lock = 0;\n",
    )
}

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
    let linked =
        svm_ir::link_with_manifest(&[type_tu, alloc_state_stub()]).expect("link real type.c");
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
        // These TUs' `calloc`/`malloc` share the libc's bump-allocator state; without `emit_libc.c`
        // in this subset, a stand-in owns those four globals so the cross-TU import resolves.
        alloc_state_stub(),
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
    // Two families of data symbols no cc1 TU defines on its own, both belonging to the libc — and a
    // cross-TU **data** reference must resolve to a concrete address (unlike a libc *function*, it
    // can't be a manifest import). Minimal units stand in for them so the whole cc1 links:
    //   * the three stdio streams (`stdin`/`stdout`/`stderr`);
    //   * the shared bump-allocator state (`__svm_brk`/…) — each TU's `#include <stdlib.h>` allocator
    //     now imports one shared bump pointer (see `stdlib.h`'s `__SVM_LIBC_EXTERN`) instead of minting
    //     its own, so the state is a cross-TU symbol the libc owns.
    // (Verified: these are the complete unresolved-data set — every `ty_*`/`opt_*`/`include_paths`/
    // `base_file` resolves cross-TU.)
    units.push(object_unit(
        "stdio_globals",
        "void *stdin;\nvoid *stdout;\nvoid *stderr;\n",
    ));
    units.push(alloc_state_stub());

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
    // (`exit`/`vm_map`). The heap allocator *functions* are NOT here: each TU that `#include
    // <stdlib.h>` gets chibicc's *bundled* header, whose `static malloc`/`calloc`/`free`/`realloc` grow
    // the window via `__vm_map` inline — so `malloc` never appears as a cross-unit *function* symbol.
    // Its *state* (`__svm_brk`/…) is shared cross-TU (one bump pointer, `__SVM_LIBC_EXTERN`), stubbed
    // above alongside the stdio globals.
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
    for cap in ["stream_write", "stream_read"] {
        assert!(
            imports.contains(&cap),
            "emit-object libc reaches stdout/stdin via the powerbox `{cap}` cap; imports: {imports:?}"
        );
    }
}

/// **The emit-object libc runs** (task #20, increment 2b): `demo2.c` drives the `printf_emit.c` engine
/// — integer/hex/octal radices, width/precision/flags, the `%*d`/`%.*s` star forms, and the `FILE*`
/// path (`fprintf`, `open_memstream`) — linked against the full `emit_libc.c` and run through `_start`
/// on **interp==JIT** under the powerbox. The stdout bytes are a fixed oracle **matching native glibc**
/// (captured from a `cc`-built reference of the same directives), so this pins that the varargs printf
/// the whole self-host stands on formats byte-for-byte correctly through emit-object → link → run.
///
/// This is the proof for the two 2b bottom-edge shims: `os_emit.c` (stdout via the powerbox `write`
/// cap) and `printf_emit.c` (the `__vm_fmt_*`-free formatter). Float directives and fd≥3 file I/O are
/// out of scope here (later increments). Linux-only (links the system-header libc).
#[cfg(target_os = "linux")]
#[test]
fn emit_object_libc_printf_runs_byte_exact_under_powerbox() {
    use svm_run::{instantiate_with_imports, Outcome, RunConfig, Value};

    // Entry unit first so `demo2`'s `_start` is merged function 0; the libc has no `main`.
    let demo = object_unit_file("crates/svm/tests/fixtures/emit_libc/demo2.c");
    let libc = object_unit_emit_libc();

    let linked = svm_ir::link_with_manifest(&[demo, libc]).expect("link demo2 + emit-object libc");
    // Every printf/stdio call resolved cross-TU; only powerbox-bindable caps survive as imports.
    let imports: Vec<&str> = linked.imports.iter().map(|i| i.name.as_str()).collect();
    for resolved in [
        "printf",
        "snprintf",
        "fprintf",
        "open_memstream",
        "vfprintf",
    ] {
        assert!(
            !imports.contains(&resolved),
            "libc `{resolved}` resolved to a direct call, not left an import; imports: {imports:?}"
        );
    }
    for cap in &imports {
        assert!(
            matches!(
                *cap,
                "stream_write" | "stream_read" | "vm_fs" | "exit" | "vm_map" | "vm_page_size"
            ),
            "only powerbox + fs caps remain; unexpected import `{cap}` in {imports:?}"
        );
    }
    svm_verify::verify_module(&linked).expect("verify linked emit-object libc program");

    // Run `_start` on the tree-walk oracle and the JIT (§18); `run_diff` asserts they agree. Bound to
    // the fs+powerbox registry (`demo2` writes only to stdout / an in-memory stream, so its `vm_fs`
    // slot, if present, is never exercised — the empty memfs suffices).
    let inst = instantiate_with_imports(linked, cc1_imports(vec![], vec![])).expect("instantiate");
    let run = inst
        .run_diff(&RunConfig::default())
        .expect("run _start on interp+jit");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(5)]));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "-42 4000000000 ff FF 100\n\
         -1234567890123 9876543210\n\
         [   42][42   ][00042][+42][ 42]\n\
         [     007][007     ][]\n\
         [    99][tru]\n\
         abc str % (null)\n\
         snlen=11 buf=111-222\n\
         stream=mem<7,x>! len=9\n\
         fd=stdout\n",
        "emit-object printf/stdio formats byte-for-byte like glibc"
    );
}

/// The **correctly-rounded float path** (task #22): `demo_float.c` prints a battery of `%.17g`/`%g`/
/// `%e`/`%f` conversions through the emit-object libc's Steele & White scaled-bignum dtoa
/// (`printf_emit.c`), run on interp==JIT under the powerbox. The headline is `%.17g` — the format
/// chibicc itself emits for float literals — round-tripping the values that a lossy formatter gets
/// wrong (0.1, 2/3, 1e-7). The oracle is glibc's own output for the identical calls (verified
/// natively): a `double` printed with `%.17g` must read back bit-identical. Linux-only (mirrors the
/// sibling printf gate — `object_unit_emit_libc`/`cc1_imports` compile the system-header libc).
#[cfg(target_os = "linux")]
#[test]
fn emit_object_libc_float_runs_byte_exact_under_powerbox() {
    use svm_run::{instantiate_with_imports, Outcome, RunConfig, Value};

    // Entry unit first so `demo_float`'s `_start` is merged function 0; the libc has no `main`.
    let demo = object_unit_file("crates/svm/tests/fixtures/emit_libc/demo_float.c");
    let libc = object_unit_emit_libc();

    let linked =
        svm_ir::link_with_manifest(&[demo, libc]).expect("link demo_float + emit-object libc");
    svm_verify::verify_module(&linked).expect("verify linked emit-object float program");

    let inst = instantiate_with_imports(linked, cc1_imports(vec![], vec![])).expect("instantiate");
    let run = inst
        .run_diff(&RunConfig::default())
        .expect("run _start on interp+jit");
    assert_eq!(run.outcome, Outcome::Returned(vec![Value::I32(0)]));
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "0.10000000000000001\n\
         1\n\
         3.14159265358979\n\
         0.66666666666666663\n\
         1e+20\n\
         -9.9999999999999995e-08\n\
         123456789 6.0220000000000003e+23\n\
         [0.5][100][0.0001][1e-05]\n\
         [1.234568e+04][4.200000E-04]\n\
         [3.14][2][0.007]\n\
         [+4.0][-5.0e+01][-0]\n",
        "emit-object %.17g (and %g/%e/%f) is correctly rounded — byte-for-byte like glibc"
    );
}

/// Link the whole `-cc1` slice — the eight upstream chibicc TUs, the `cc1_main.c` entry, and the
/// emit-object guest libc (`emit_libc.c`) — into one verified module. The shared machinery behind
/// both the "links with only powerbox caps" gate and the 2c self-compile run below. Linux-only.
#[cfg(target_os = "linux")]
fn link_whole_cc1() -> svm_ir::Module {
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
    units.push(object_unit_emit_libc());

    let linked = svm_ir::link_with_manifest(&units).expect("link whole cc1 + emit-object libc");
    svm_verify::verify_module(&linked).expect("verify the whole linked cc1 + libc");
    linked
}

/// **The whole cc1 links against the real emit-object libc** (task #20, increment 2b): the nine cc1
/// TUs + `emit_libc.c`, linked and verified into one module whose *every* remaining import is a
/// default-powerbox-bindable cap — so the linked compiler instantiates, not just verifies. This is the
/// 2a link (which stood the libc's stdio globals in with a stub) now closed against the actual libc:
/// `stdin`/`stdout`/`stderr`, the `str`/`mem`/ctype/stdio/printf surface, and `strtod` all resolve
/// cross-unit, and what is left — `write`/`read`/`exit`/`vm_map`/`vm_page_size` — the powerbox binds.
#[cfg(target_os = "linux")]
#[test]
fn whole_cc1_links_against_emit_libc_with_only_powerbox_caps() {
    let linked = link_whole_cc1();

    // `_start` at function 0 and the shared `ty_int` global both survived (as in 2a).
    assert!(
        linked
            .exports
            .iter()
            .any(|e| e.name == "_start" && e.func == 0),
        "cc1_main's _start is merged function 0"
    );
    // The decisive close: no libc symbol is left dangling — every import is a powerbox or fs cap. (The
    // 2a link still had `fopen`/`vfprintf`/`strlen`/… as manifest imports; the libc now supplies them.)
    // The stdio edge is `stream_write`/`stream_read` (fd-dispatch owns `write`/`read`), and the fs edge
    // is the single `vm_fs` seam (2c) — both bound by [`cc1_imports`].
    let imports: Vec<&str> = linked.imports.iter().map(|i| i.name.as_str()).collect();
    for imp in &imports {
        assert!(
            matches!(
                *imp,
                "stream_write" | "stream_read" | "vm_fs" | "exit" | "vm_map" | "vm_page_size"
            ),
            "every remaining import is a powerbox or fs cap; unexpected `{imp}` in {imports:?}"
        );
    }
    // And the stdio streams are real definitions now, not the 2a stub.
    for g in ["stdin", "stdout", "stderr"] {
        assert!(
            linked.data_exports.iter().any(|e| e.name == g),
            "the linked cc1 carries a real `{g}` from the libc"
        );
    }
}

/// Build the **native reference compiler** `chibicc_ref` — the *same* `-cc1` slice the guest is made
/// of (the eight upstream TUs + `cc1_main.c`), but built with the system clang + libc instead of
/// chibicc-`--emit-object` + the guest libc. This is the apples-to-apples oracle for the self-host
/// differential (`run_selfhost_diff.sh`): identical frontend both sides, so the only variables are the
/// substrate (guest libc + SVM engine vs system libc + native CPU). Built with `-mlong-double-64` (F3:
/// the guest bounds `long double` to `double`) + `native_ref_shims.c` (`strtold`→`strtod`, so the
/// 64-bit `long double` ABI does not clash with glibc's 80-bit `strtold` — see that file). Cached.
#[cfg(target_os = "linux")]
fn chibicc_ref() -> &'static Path {
    static REF: OnceLock<PathBuf> = OnceLock::new();
    REF.get_or_init(|| {
        let src = repo_root().join("frontend/chibicc");
        let here = selfhost_dir();
        let out = std::env::temp_dir().join(format!("svm_chibicc_ref_{}", std::process::id()));
        // Prefer clang-18 (the pinned version the on-ramp uses), else whatever `clang` is on PATH.
        let clang = if Command::new("clang-18")
            .arg("--version")
            .status()
            .is_ok_and(|s| s.success())
        {
            "clang-18"
        } else {
            "clang"
        };
        let mut cmd = Command::new(clang);
        cmd.args(["-O2", "-mlong-double-64", "-Wno-switch"])
            .arg(format!("-I{}", src.display()));
        for t in [
            "tokenize",
            "preprocess",
            "parse",
            "type",
            "codegen_ir",
            "strings",
            "hashmap",
            "unicode",
        ] {
            cmd.arg(src.join(format!("{t}.c")));
        }
        cmd.arg(here.join("cc1_main.c"))
            .arg(here.join("native_ref_shims.c"))
            .arg("-o")
            .arg(&out);
        let status = cmd.status().expect("run clang to build chibicc_ref");
        assert!(status.success(), "native chibicc_ref build failed");
        out
    })
    .as_path()
}

/// The name-keyed capability registry the linked cc1 runs against (SELFHOST_C.md §7, 2c). The
/// emit-object compiler reaches the host only through the recognized `__vm_*` builtins the frontend
/// lowers (`codegen_ir.c`); this binds each to a concrete cap:
///   * `stream_write`/`stream_read` → the powerbox `Stream` out/in (stdout/stderr, stdin);
///   * `exit` → the `Exit` cap; `vm_map`/`vm_page_size` → the `Memory` cap (window growth);
///   * `vm_fs` → a **seeded in-memory filesystem**. The guest passes the fs op in arg0
///     (`__vm_fs(op, …)`), so one manifest slot carries the whole protocol: a thin wrapper forwards
///     `args[0]` as the op and `args[1..]` as its arguments to the shared `mem_fs` handler
///     (`crates/svm-run/src/fs.rs`). `files`/`dirs` seed the memfs — a `#include` header lives here.
#[cfg(target_os = "linux")]
fn cc1_imports(files: Vec<(String, Vec<u8>)>, dirs: Vec<String>) -> svm_run::Imports {
    use svm_run::HostCap;
    // The fs seam: op-in-arg0 over one shared mem_fs store (fresh per host grant, deterministic seed).
    let fs = HostCap::host_proc(0, move || {
        let (mut inner, _handle) = svm_run::fs::mem_fs_seeded_shared(files.clone(), dirs.clone());
        Box::new(
            move |_slot_op: u32,
                  args: &[i64],
                  mem: Option<&mut dyn svm_interp::GuestMem>,
                  minter: Option<&mut dyn svm_interp::RegionMinter>| {
                inner(args[0] as u32, &args[1..], mem, minter)
            },
        )
    });
    svm_run::Imports::new()
        .provide("stream_write", HostCap::stdout())
        .provide("stream_read", HostCap::stdin())
        .provide("exit", HostCap::exit())
        .provide("vm_map", HostCap::memory(0))
        .provide("vm_page_size", HostCap::memory(3))
        .provide("vm_fs", fs)
}

/// Run the linked cc1 (`args` = argv, `stdin` = the source for `chibicc -`) on the interpreter **and**
/// the JIT under [`cc1_imports`] (memfs seeded with `files`/`dirs`), assert the two engines emit
/// byte-identical IR (§18) that parses, and return that IR. The self-compile pipeline, both engines.
#[cfg(target_os = "linux")]
fn cc1_compile(
    linked: &svm_ir::Module,
    args: &[&[u8]],
    stdin: &[u8],
    files: Vec<(String, Vec<u8>)>,
    dirs: Vec<String>,
) -> Vec<u8> {
    use svm_run::{instantiate_with_imports, Backend, RunConfig};
    let cfg = RunConfig {
        args: args.iter().map(|a| a.to_vec()).collect(),
        stdin: stdin.to_vec(),
        ..RunConfig::default()
    };
    let run = |backend| {
        instantiate_with_imports(linked.clone(), cc1_imports(files.clone(), dirs.clone()))
            .expect("instantiate the linked cc1 against the fs+powerbox registry")
            .run_with_caps(backend, &cfg, &[])
            .expect("run the linked cc1")
    };
    let interp = run(Backend::TreeWalk);
    let jit = run(Backend::Jit);
    assert!(
        !interp.stdout.is_empty(),
        "the compiler emitted SVM IR (interp stderr: {})",
        String::from_utf8_lossy(&interp.stderr)
    );
    assert_eq!(
        interp.stdout, jit.stdout,
        "the linked compiler emits byte-identical IR on the interpreter and the JIT (§18)"
    );
    let ir = String::from_utf8(interp.stdout.clone()).expect("emitted IR is UTF-8");
    svm_text::parse_module(&ir).unwrap_or_else(|e| panic!("emitted IR should parse: {e:?}\n{ir}"));
    interp.stdout
}

/// **Benchmark: chibicc self-compile on native vs SVM bytecode vs SVM JIT.** The tree-walk interpreter
/// is the correctness oracle, not a consumer engine, so it's excluded here — this measures the two
/// engines a consumer actually runs. For each TU it times: native `chibicc_ref`; the SVM **bytecode**
/// engine (instantiate = IR→bytecode compile; run = execute); and the SVM **JIT** (instantiate =
/// compile the ~300-function compiler to native; run = execute). Instantiate is timed separately so the
/// JIT's one-time compile-the-compiler cost (amortized in any real deployment — compile once, run many)
/// is visible apart from the per-invocation run. Opt-in (`SVM_SELFHOST_BENCH=1`); not a CI gate.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "benchmark: chibicc self-compile native vs bytecode vs jit; opt-in via SVM_SELFHOST_BENCH=1"]
fn bench_self_compile_native_vs_bytecode_vs_jit() {
    use std::time::Instant;
    use svm_run::{instantiate_with_imports, Backend, RunConfig};
    if std::env::var_os("SVM_SELFHOST_BENCH").is_none() {
        eprintln!("skipping self-compile benchmark; set SVM_SELFHOST_BENCH=1 to run");
        return;
    }
    let linked = link_whole_cc1();
    // The linked cc1 module is IDENTICAL for every TU below — only the *input file* (fed via the fs
    // cap) changes. So the JIT recompiles the exact same module on every run, and its compile cost is
    // a genuine constant. Measure it in isolation (pure `svm_jit::compile`, no execution) so the table
    // can be read as `jit-run ≈ compile-the-compiler + execution`, with execution the only variable.
    let nfuncs = linked.funcs.len();
    let ninsns: usize = linked
        .funcs
        .iter()
        .flat_map(|f| f.blocks.iter())
        .map(|b| b.insts.len())
        .sum();
    let mut jit_compile_ms = u128::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let _ = svm_jit::compile(&linked, 0).expect("compile whole cc1");
        jit_compile_ms = jit_compile_ms.min(t.elapsed().as_millis());
    }
    let tus = [
        "hashmap.c",
        "type.c",
        "tokenize.c",
        "preprocess.c",
        "parse.c",
        "codegen_ir.c",
    ];
    let inc: &[&[u8]] = &[
        b"-Ifrontend/chibicc",
        b"-Ifrontend/chibicc/include",
        b"-Iusr/include/x86_64-linux-gnu",
        b"-Iusr/include",
    ];
    // Time (instantiate, run) for a backend, best-of-`reps` on each phase; returns (ms, ms, out_len).
    let bench =
        |backend: Backend, args: &[Vec<u8>], files: &[(String, Vec<u8>)], dirs: &[String], reps| {
            let cfg = RunConfig {
                args: args.to_vec(),
                ..RunConfig::default()
            };
            let (mut inst_ms, mut run_ms, mut out) = (u128::MAX, u128::MAX, 0usize);
            for _ in 0..reps {
                let t0 = Instant::now();
                let instance = instantiate_with_imports(
                    linked.clone(),
                    cc1_imports(files.to_vec(), dirs.to_vec()),
                )
                .expect("instantiate");
                inst_ms = inst_ms.min(t0.elapsed().as_millis());
                let t1 = Instant::now();
                let r = instance
                    .run_with_caps(backend, &cfg, &[])
                    .expect("run backend");
                run_ms = run_ms.min(t1.elapsed().as_millis());
                out = r.stdout.len();
            }
            (inst_ms, run_ms, out)
        };
    println!("\n=== chibicc self-compile: native vs SVM bytecode vs SVM JIT (best-of-3, ms) ===");
    println!(
        "{:<14} {:>8} | {:>10} {:>8} | {:>10} {:>8} {:>8}   out",
        "TU", "native", "bc-inst", "bc-run", "jit-inst", "jit-run", "jit-tot"
    );
    // Floor: a trivial no-`#include` compile isolates the fixed per-run cost (JIT = compile the ~300-fn
    // compiler to native; bytecode = engine startup) from the actual front-end/codegen work. Subtract
    // this from a TU's `*-run` to get the marginal compile time a real deployment amortizes to.
    {
        let files = vec![("t.c".to_string(), b"int main(void){return 42;}\n".to_vec())];
        let dirs: Vec<String> = vec![];
        let args: Vec<Vec<u8>> = vec![
            b"chibicc".to_vec(),
            b"--emit-object".to_vec(),
            b"t.c".to_vec(),
        ];
        let (bi, br, _) = bench(Backend::Bytecode, &args, &files, &dirs, 3);
        let (ji, jr, _) = bench(Backend::Jit, &args, &files, &dirs, 3);
        println!(
            "{:<14} {:>8} | {bi:>10} {br:>8} | {ji:>10} {jr:>8} {:>8}   (fixed floor)",
            "(trivial)",
            "-",
            ji + jr
        );
    }
    for tu in tus {
        let path = format!("frontend/chibicc/{tu}");
        let (mut files, mut dirs) = chibicc_tu_closure(&path, &["-Ifrontend/chibicc"]);
        files.push((
            SELFHOST_PRELUDE.to_string(),
            std::fs::read(repo_root().join(SELFHOST_PRELUDE)).unwrap(),
        ));
        let mut anc = Path::new(SELFHOST_PRELUDE).parent();
        while let Some(d) = anc {
            if d.as_os_str().is_empty() {
                break;
            }
            dirs.push(d.to_string_lossy().into_owned());
            anc = d.parent();
        }
        dirs.sort();
        dirs.dedup();
        let mut args: Vec<Vec<u8>> = vec![b"chibicc".to_vec(), b"--emit-object".to_vec()];
        args.push(b"-include".to_vec());
        args.push(SELFHOST_PRELUDE.as_bytes().to_vec());
        for i in inc {
            args.push(i.to_vec());
        }
        args.push(path.as_bytes().to_vec());

        // native: time chibicc_ref on the identical relative flags.
        let native_args = [
            "--emit-object",
            "-include",
            SELFHOST_PRELUDE,
            "-Ifrontend/chibicc",
            "-Ifrontend/chibicc/include",
            "-I/usr/include/x86_64-linux-gnu",
            "-I/usr/include",
            &path,
        ];
        let mut nat = u128::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            let _ = native_cc1_file(&native_args);
            nat = nat.min(t.elapsed().as_millis());
        }
        let (bi, br, _) = bench(Backend::Bytecode, &args, &files, &dirs, 3);
        let (ji, jr, _out) = bench(Backend::Jit, &args, &files, &dirs, 3);
        println!(
            "{tu:<14} {nat:>8} | {bi:>10} {br:>8} | {ji:>10} {jr:>8} {:>8}   {_out}",
            ji + jr
        );
    }
    let profile = if cfg!(debug_assertions) {
        "DEBUG"
    } else {
        "release"
    };
    println!(
        "\nlinked cc1: {nfuncs} funcs, {ninsns} IR insns.  \
         pure svm_jit::compile (Cranelift, opt_level=speed, no execution, best-of-3): {jit_compile_ms} ms [{profile} build].\n\
         *** PROFILE CAVEAT: `cargo test` builds Cranelift ITSELF unoptimized (no [profile.test] dep \
         override), so in a DEBUG build every engine column here is ~10-15x inflated. The SAME compile \
         in a release build is ~0.7s, not ~10s — that ~15x is the debug-Cranelift penalty, NOT the \
         production JIT cost. Run with `cargo test --release` for representative engine numbers; native \
         (an external clang-built process) is unaffected either way. ***\n\
         `jit-run` is still a genuine cost in kind: the powerbox entry is run-once (`_start`), so it \
         Cranelift-compiles the whole {nfuncs}-fn guest AND executes it every call — a `compile one file, \
         exit` invocation really does pay the full compile (release: ~0.7s here, not ~10s).\n\
         The compile so dominates that this bench cannot isolate execution (compile ≈ run; re-running \
         `_start` on a dirtied window isn't valid), so per-file execution sits below the compile's own \
         run-to-run variance — measured here only as 'small vs compile', not precisely.\n\
         When the fixed compile amortizes (long-lived process reusing a persistent CompiledModule / \
         JitSession) it drops out; a run-once self-host does not get that.\n\
         (bc-inst / jit-inst = per-call instantiate — negligible; the cost lives in `run`.)\n"
    );
}

/// The native oracle: `chibicc_ref` (same `cc1_main` entry, system clang + libc) fed `stdin` for
/// `chibicc [extra_args...] -`, returning the emitted IR. `-g` is off by default on both sides, so
/// there is no `debug.file <path>` line — a true byte-for-byte compare of the emitted IR.
#[cfg(target_os = "linux")]
fn native_cc1(extra_args: &[&str], stdin: &[u8]) -> Vec<u8> {
    let mut cmd = Command::new(chibicc_ref());
    cmd.args(extra_args).arg("-");
    let mut child = cmd
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn chibicc_ref");
    child
        .stdin
        .take()
        .expect("chibicc_ref stdin")
        .write_all(stdin)
        .expect("write source to chibicc_ref");
    let out = child.wait_with_output().expect("wait for chibicc_ref");
    assert!(
        out.status.success(),
        "native chibicc_ref failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// The **transitive header/source closure** a chibicc TU pulls, computed by asking native chibicc for
/// its own dependency list (`-M`). Returns `(files, dirs)` for seeding the guest memfs: `files` maps
/// each closure member to its bytes at a **relative** key (the fs cap refuses absolute paths, §fs
/// `read_path`), `dirs` names every ancestor directory. The TU itself is in the closure, so it seeds
/// too. Native reads the identical files from the real tree; header paths never reach the emitted IR
/// (only the TU's own `__FILE__`), so the two sides stay byte-comparable despite the `/usr` vs `usr`
/// `-I` difference. Linux-only (real glibc header tree). Returns `None` if `chibicc -M` can't run.
#[cfg(target_os = "linux")]
fn chibicc_tu_closure(tu_rel: &str, m_incs: &[&str]) -> (Vec<(String, Vec<u8>)>, Vec<String>) {
    let root = repo_root();
    // Run `-M` on the **relative** TU path (cwd = repo root) so repo-local deps come back relative
    // (`frontend/chibicc/hashmap.c`, `frontend/chibicc/chibicc.h`) — matching the guest's base_file and
    // `-I` keys. An absolute input would emit absolute deps and the memfs keys wouldn't line up.
    // `m_incs` are extra `-I` dirs the `#include` search needs (e.g. `-Ifrontend/chibicc` for cc1_main.c,
    // whose `#include "chibicc.h"` isn't a sibling) — IR/closure-neutral for the frontend TUs.
    let out = Command::new(chibicc())
        .args(["-M", "-c", tu_rel])
        .args(m_incs)
        .current_dir(&root)
        .output()
        .expect("run chibicc -M");
    assert!(
        out.status.success(),
        "chibicc -M failed on {tu_rel}: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8(out.stdout).expect("-M output is UTF-8");
    // `-M` prints `target.o: dep dep \<newline> dep ...`; everything after the first ':' is the deps.
    let deps = text.split_once(':').map(|(_, d)| d).unwrap_or("");
    let mut files = Vec::new();
    let mut dirs = std::collections::BTreeSet::new();
    let mut seen = std::collections::BTreeSet::new();
    for tok in deps.split_whitespace() {
        if tok == "\\" {
            continue; // line-continuation marker
        }
        // Real host path: absolute (`/usr/...`, or `/<repo>/frontend/...` when chibicc is invoked by
        // absolute argv0) or repo-relative (`./frontend/...`, `frontend/...`).
        let real = if tok.starts_with('/') {
            PathBuf::from(tok)
        } else {
            root.join(tok.trim_start_matches("./"))
        };
        // Memfs key: repo-relative for repo files (so it matches the guest's relative `-I` and
        // base_file), else the system path with its leading '/' stripped (`/usr/... → usr/...`).
        let rel = match real.strip_prefix(&root) {
            Ok(r) => r.to_string_lossy().into_owned(),
            Err(_) => real.to_string_lossy().trim_start_matches('/').to_string(),
        };
        if !seen.insert(rel.clone()) {
            continue;
        }
        let rel = rel.as_str();
        let bytes =
            std::fs::read(&real).unwrap_or_else(|e| panic!("read closure file {real:?}: {e}"));
        let mut anc = Path::new(rel).parent();
        while let Some(d) = anc {
            if d.as_os_str().is_empty() {
                break;
            }
            dirs.insert(d.to_string_lossy().into_owned());
            anc = d.parent();
        }
        files.push((rel.to_string(), bytes));
    }
    (files, dirs.into_iter().collect())
}

/// The native oracle for a **file** compile (vs [`native_cc1`]'s stdin form): run `chibicc_ref` with
/// `args` from the repo root so repo-relative sources/`-I`s resolve against the real tree, returning
/// the emitted IR. Used to diff the guest compiling a real chibicc TU against native.
#[cfg(target_os = "linux")]
fn native_cc1_file(args: &[&str]) -> Vec<u8> {
    let out = Command::new(chibicc_ref())
        .args(args)
        .current_dir(repo_root())
        .output()
        .expect("run chibicc_ref on a file");
    assert!(
        out.status.success(),
        "native chibicc_ref (file) failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

/// Compile one real chibicc TU through the linked guest cc1 and assert the emitted IR byte-matches
/// native `chibicc_ref` on the identical source. Guest `-I`s are relative (memfs, no absolute paths)
/// in native's default search order — chibicc's bundled `include/` first, then the system roots; the
/// TU is a memfs file (not stdin) so its quoted `#include "chibicc.h"` resolves against its own dir.
/// Native reads the real header tree (absolute `-I`); header paths never reach the IR, so the only
/// variables are the substrate (guest libc + SVM vs system libc + native CPU).
/// The self-host prelude, force-included on every guest/native emit-object compile of a chibicc TU (as
/// `emit_object_real` does when building the guest): chibicc's parser can't ingest modern glibc's
/// ISO-C23 `strtoul`/`atoi`/… redirects, so the prelude supplies clean prototypes. Declarations-only ⇒
/// emits no IR ⇒ byte-match holds; TUs that don't call those functions are unaffected.
#[cfg(target_os = "linux")]
const SELFHOST_PRELUDE: &str = "crates/svm-run/demos/chibicc_selfhost/selfhost_prelude.h";

/// Guest-compile one chibicc TU to emitted-object IR bytes: seed its `-M` header closure + the prelude
/// into the memfs, then run the linked guest cc1 on it (interp==JIT, via [`cc1_compile`]). The `-I`s
/// are relative (the fs cap refuses absolute paths); `-Ifrontend/chibicc` lets cc1_main.c's quoted
/// `#include "chibicc.h"` (not a sibling) resolve and is IR-neutral for the frontend TUs. Shared by the
/// per-TU differential and the relink test.
#[cfg(target_os = "linux")]
fn guest_emit_object(linked: &svm_ir::Module, tu: &str) -> Vec<u8> {
    let (mut files, mut dirs) = chibicc_tu_closure(tu, &["-Ifrontend/chibicc"]);
    assert!(
        files.len() > 20,
        "{tu}: the closure should pull chibicc.h's system headers, got {} files",
        files.len()
    );
    files.push((
        SELFHOST_PRELUDE.to_string(),
        std::fs::read(repo_root().join(SELFHOST_PRELUDE)).expect("read selfhost prelude"),
    ));
    let mut anc = Path::new(SELFHOST_PRELUDE).parent();
    while let Some(d) = anc {
        if d.as_os_str().is_empty() {
            break;
        }
        dirs.push(d.to_string_lossy().into_owned());
        anc = d.parent();
    }
    dirs.sort();
    dirs.dedup();
    cc1_compile(
        linked,
        &[
            b"chibicc",
            b"--emit-object",
            b"-include",
            SELFHOST_PRELUDE.as_bytes(),
            b"-Ifrontend/chibicc",
            b"-Ifrontend/chibicc/include",
            b"-Iusr/include/x86_64-linux-gnu",
            b"-Iusr/include",
            tu.as_bytes(),
        ],
        b"",
        files,
        dirs,
    )
}

#[cfg(target_os = "linux")]
fn assert_guest_tu_matches_native(linked: &svm_ir::Module, tu: &str) {
    let guest = guest_emit_object(linked, tu);
    // Native reads the real header tree (absolute `-I`); header paths never reach the IR, so the only
    // variables are the substrate (guest libc + SVM vs system libc + native CPU). `-Ifrontend/chibicc`
    // mirrors the guest side (IR-neutral; both stay symmetric so the diff is only substrate).
    let native = native_cc1_file(&[
        "--emit-object",
        "-include",
        SELFHOST_PRELUDE,
        "-Ifrontend/chibicc",
        "-Ifrontend/chibicc/include",
        "-I/usr/include/x86_64-linux-gnu",
        "-I/usr/include",
        tu,
    ]);
    assert_eq!(
        String::from_utf8_lossy(&guest),
        String::from_utf8_lossy(&native),
        "guest-compiled IR for {tu} is byte-identical to native chibicc_ref"
    );
}

/// Guest-compile a chibicc TU into a **linkable unit** — the guest's *own* emitted object, parsed and
/// wire-round-tripped like [`object_unit_real`]'s native ones. The building block of `chibicc2`.
#[cfg(target_os = "linux")]
fn guest_cc1_unit(linked: &svm_ir::Module, tu: &str, tag: &str) -> LinkUnit {
    let ir = guest_emit_object(linked, tu);
    let m = svm_text::parse_module(&String::from_utf8(ir).expect("guest IR is UTF-8"))
        .unwrap_or_else(|e| panic!("parse guest object {tag}: {e:?}"));
    unit_via_svmo(m, tag)
}

/// The **native** counterpart of [`guest_cc1_unit`]: compile the same TU with `chibicc_ref` using the
/// *identical relative* flags/paths the guest uses (repo-relative `-I`s and TU path), so the two units
/// differ **only** by substrate (system libc + native CPU vs guest libc + SVM). This is what makes the
/// relink comparison apples-to-apples — unlike `link_whole_cc1`, whose units embed *absolute* `__FILE__`
/// paths (`emit_object_real` passes an absolute cfile), which the memfs-relative guest can't reproduce.
#[cfg(target_os = "linux")]
fn native_cc1_unit(tu: &str, tag: &str) -> LinkUnit {
    let ir = native_cc1_file(&[
        "--emit-object",
        "-include",
        SELFHOST_PRELUDE,
        "-Ifrontend/chibicc",
        "-Ifrontend/chibicc/include",
        "-I/usr/include/x86_64-linux-gnu",
        "-I/usr/include",
        tu,
    ]);
    let m = svm_text::parse_module(&String::from_utf8(ir).expect("native IR is UTF-8"))
        .unwrap_or_else(|e| panic!("parse native object {tag}: {e:?}"));
    unit_via_svmo(m, tag)
}

/// **Bootstrap-fixpoint slice #1 (SELFHOST_C.md §5 E, task #7): the guest compiles chibicc's *own*
/// source.** The linked whole cc1 ([`link_whole_cc1`]) compiles real upstream chibicc TUs — each
/// pulling chibicc.h's full system-header closure (~95 files) from the seeded memfs — and the emitted
/// IR must byte-match native `chibicc_ref` on the identical source. This is the stage-2 conformance
/// check: the SVM-executed compiler is faithful to native not just on crafted programs but on the
/// compiler's own code, the hardest input on the path to `chibicc2 == chibicc3`.
///
/// Covers the **tractable** TUs (≤ ~800 lines): the guest runs under the tree-walk interpreter, so
/// runtime scales with TU size. The three giants (`preprocess.c`, `codegen_ir.c`, `parse.c` —
/// 1.2k–3.4k lines) are left to a future slow lane; they exercise no new *frontend* surface, only more
/// of the same. Heavy (`#[ignore]`d): builds/links the whole cc1, runs each TU on two engines over
/// glibc-scale headers; the fixed link cost is shared across the corpus.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "heavy: the guest compiles real chibicc TUs (full glibc header closure) on two engines"]
fn whole_cc1_compiles_its_own_tu_matching_native() {
    let linked = link_whole_cc1();
    for tu in [
        "frontend/chibicc/strings.c",
        "frontend/chibicc/hashmap.c",
        "frontend/chibicc/unicode.c",
        "frontend/chibicc/type.c",
        "frontend/chibicc/tokenize.c",
    ] {
        assert_guest_tu_matches_native(&linked, tu);
    }
}

/// The **giant** chibicc TUs (`preprocess.c`/`parse.c`/`codegen_ir.c`, 1.2k–3.4k lines) on the same
/// guest-vs-native differential as [`whole_cc1_compiles_its_own_tu_matching_native`]. Compiling these
/// under the tree-walk interpreter is minutes each (~1h for all three on two engines), far past the
/// per-PR CI budget — so this is **opt-in**: it no-ops fast unless `SVM_SELFHOST_GIANTS=1` is set
/// (a nightly lane sets it). Proving these three (on top of the five tractable TUs) closes per-TU
/// byte-identity across all nine cc1 TUs — the sufficient condition for `chibicc2 == chibicc3` (linking
/// the guest's objects then reproduces the native-built guest, which reproduces its own output).
#[cfg(target_os = "linux")]
#[test]
#[ignore = "very heavy (~1h): giant chibicc TUs on two engines; opt-in via SVM_SELFHOST_GIANTS=1"]
fn whole_cc1_compiles_giant_tus_matching_native() {
    if std::env::var_os("SVM_SELFHOST_GIANTS").is_none() {
        eprintln!("skipping giant-TU self-compile; set SVM_SELFHOST_GIANTS=1 to run (~1h)");
        return;
    }
    let linked = link_whole_cc1();
    for tu in [
        "frontend/chibicc/preprocess.c",
        "frontend/chibicc/parse.c",
        "frontend/chibicc/codegen_ir.c",
    ] {
        assert_guest_tu_matches_native(&linked, tu);
    }
}

/// **Bootstrap fixpoint, mechanized (SELFHOST_C.md §5 E, task #7): `chibicc2 == chibicc1`.** Where the
/// per-TU differential proves the guest emits byte-identical *object IR* to native for each cc1 TU,
/// this closes the loop at the **module** level: relink a whole cc1 from the guest's *own* emitted
/// objects and assert it is byte-identical to `link_whole_cc1` (the native-built guest). Each of the
/// nine cc1 TUs (the `cc1_main` entry + eight upstream) is compiled by the running guest cc1 into a
/// unit; linked with the same native `emit_libc` (the libc is substrate, not the compiler under test)
/// the result is `chibicc2`. Byte-equality with `chibicc1` means the guest-built compiler *is* the
/// native-built one, so running it reproduces its own output (`chibicc2 == chibicc3`). ∎ Very heavy
/// (guest-compiles all nine, incl. the giants); opt-in via `SVM_SELFHOST_GIANTS=1`, on the nightly lane.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "very heavy (~10 min): relink the whole cc1 from the guest's own objects; opt-in via SVM_SELFHOST_GIANTS=1"]
fn whole_cc1_relinks_from_guest_objects_equals_native() {
    if std::env::var_os("SVM_SELFHOST_GIANTS").is_none() {
        eprintln!("skipping guest-object relink; set SVM_SELFHOST_GIANTS=1 to run (~10 min)");
        return;
    }
    let entry = "crates/svm-run/demos/chibicc_selfhost/cc1_main.c";
    let upstream = [
        "tokenize.c",
        "preprocess.c",
        "parse.c",
        "type.c",
        "codegen_ir.c",
        "hashmap.c",
        "unicode.c",
        "strings.c",
    ];
    // The running native-built guest that *does* the compiling. How it was itself built doesn't affect
    // the IR it emits — only the input's base_file does.
    let running = link_whole_cc1();

    // chibicc1 — the native reference, built from `chibicc_ref` units compiled with the **same relative**
    // flags/paths as the guest, so `__FILE__` and every other byte match and the only variable is the
    // substrate. (Not `link_whole_cc1` itself: its units embed *absolute* `__FILE__` paths —
    // `emit_object_real` passes an absolute cfile — which the memfs-relative guest can't reproduce.)
    let mut native_units = vec![native_cc1_unit(entry, "cc1_main.c")];
    for tu in upstream {
        native_units.push(native_cc1_unit(&format!("frontend/chibicc/{tu}"), tu));
    }
    native_units.push(object_unit_emit_libc());
    let chibicc1 =
        svm_ir::link_with_manifest(&native_units).expect("link native chibicc1 (relative paths)");

    // chibicc2 — the same units, but each cc1 TU compiled by the **running guest** into its own object;
    // same native emit_libc (the libc is substrate, not the compiler under test), same order.
    let mut guest_units = vec![guest_cc1_unit(&running, entry, "cc1_main.c")];
    for tu in upstream {
        guest_units.push(guest_cc1_unit(
            &running,
            &format!("frontend/chibicc/{tu}"),
            tu,
        ));
    }
    guest_units.push(object_unit_emit_libc());
    let chibicc2 = svm_ir::link_with_manifest(&guest_units)
        .expect("link chibicc2 from the guest's own objects");
    svm_verify::verify_module(&chibicc2).expect("verify chibicc2");

    assert_eq!(
        svm_text::print_module(&chibicc2),
        svm_text::print_module(&chibicc1),
        "chibicc2 (relinked from the guest's own emitted objects) is byte-identical to chibicc1 (native-built, same flags)"
    );
}

/// **2c — the linked whole compiler *runs* and self-hosts a real compile** (SELFHOST_C.md §7). The
/// nine cc1 TUs + `emit_libc.c` link into one module ([`link_whole_cc1`]); here that module runs
/// through `_start` under the powerbox and **compiles C to SVM IR inside the sandbox** — the self-host
/// lever, executed rather than merely linked. Two programs, each proven three ways at once: byte-
/// identical on the **interpreter and the JIT** (§18), a **parseable** module, and byte-identical to the
/// **native reference** (`chibicc_ref`, the same `cc1_main` entry built with system clang + libc).
///
/// (a) A no-`#include` program on stdin (`chibicc -`) — the base self-compile.
/// (b) A `#include`-bearing program whose header the running compiler **reads from a seeded in-sandbox
/// filesystem** (`vm_fs` cap → `mem_fs`) — the 2c fs slice: `fopen`/`fread` on fd≥3 reach the memfs
/// through `os_emit.c`'s `__vm_fs` seam, while stdout/stdin stay on the `Stream` cap. The native side
/// reads the identical header from a real `-I` directory; the emitted IR matches (headers aren't named
/// in the output, `-g` off), so the guest fs path reproduces the native include exactly.
/// (c) A **corpus** of feature-diverse standalone programs (SELFHOST_C.md §5 E, task #7) — widening the
/// differential toward the bootstrap fixpoint by exercising a broad C surface (structs/arrays, enums +
/// function pointers + switch, unions + bitfields, goto/ternary/comma, globals + static locals + string
/// literals, `long long` + shifts + 2-D arrays, designated initializers + compound literals); each is
/// byte-identical guest-vs-native, catching any construct where the SVM-executed compiler would diverge.
///
/// Heavy (`make`s chibicc, compiles ten TUs `--emit-object`, links + verifies, runs on two engines, and
/// clang-builds the reference), so `#[ignore]`d — run explicitly:
/// `cargo test -p svm --test c_link -- --ignored self_compiles`.
#[cfg(target_os = "linux")]
#[test]
#[ignore = "heavy: builds + links + runs the whole cc1 on two engines and clang-builds the reference"]
fn whole_cc1_self_compiles_a_program_matching_native_on_interp_and_jit() {
    let linked = link_whole_cc1();

    // (a) A no-`#include` program (its result is `main`'s exit code): recursion, an array + loop, a
    // couple of calls — enough that the emitted IR exercises the codegen, not a one-liner. On stdin.
    let src: &[u8] = b"int add(int a, int b) { return a + b; }\n\
        int fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n\
        int main(void) {\n\
        \x20 int xs[4];\n\
        \x20 int s = 0;\n\
        \x20 for (int i = 0; i < 4; i++) { xs[i] = i * i; s += xs[i]; }\n\
        \x20 return add(s, fib(10));\n\
        }\n";
    let guest = cc1_compile(&linked, &[b"chibicc", b"-"], src, vec![], vec![]);
    assert_eq!(
        guest,
        native_cc1(&[], src),
        "no-#include: guest-emitted IR is byte-identical to the native reference"
    );

    // (b) A `#include`-bearing program: the header is read from the seeded memfs through the `vm_fs`
    // cap. The source is on stdin; `#include <vec.h>` resolves against the default `/include` search
    // path (cc1_main), which the memfs serves at `include/vec.h`. The native side reads the identical
    // header from a real `-I` directory. Same source + header both ways ⇒ identical IR.
    let hdr: &[u8] = b"#ifndef VEC_H\n#define VEC_H\n\
        static int sq(int x) { return x * x; }\n\
        static int cube(int x) { return x * sq(x); }\n\
        #endif\n";
    let src_inc: &[u8] = b"#include <vec.h>\n\
        int main(void) {\n\
        \x20 int acc = 0;\n\
        \x20 for (int i = 1; i <= 5; i++) acc += sq(i) + cube(i);\n\
        \x20 return acc & 0xff;\n\
        }\n";
    let guest_inc = cc1_compile(
        &linked,
        &[b"chibicc", b"-"],
        src_inc,
        vec![("include/vec.h".to_string(), hdr.to_vec())],
        vec!["include".to_string()],
    );
    // Native reads the same header from a real directory on the `-I` path (cc1_main takes one `-I`).
    let incdir = std::env::temp_dir().join(format!("svm_cc1_inc_{}", std::process::id()));
    std::fs::create_dir_all(&incdir).expect("create native include dir");
    std::fs::write(incdir.join("vec.h"), hdr).expect("write native header");
    let iarg = format!("-I{}", incdir.display());
    assert_eq!(
        guest_inc,
        native_cc1(&[&iarg], src_inc),
        "#include: guest IR (header read from the in-sandbox memfs) == native (header from disk)"
    );

    // (c) Corpus widening toward the bootstrap fixpoint (SELFHOST_C.md §5 E, task #7): feature-diverse
    // standalone programs, each proving the guest-executed compiler emits byte-identical IR to native
    // across a broad C surface — structs + arrays of structs, enums + function pointers + switch,
    // unions + bitfields + casts, goto/ternary/comma + do-while, globals + static locals + string
    // literals, `long long` + shifts + 2-D arrays, and designated initializers + compound literals.
    // No `#include` (pure language surface ⇒ no header seeding); each result is `main`'s exit code, but
    // it is the *emitted IR* that is diffed, so the diff exercises codegen breadth, not runtime.
    let corpus: &[(&str, &[u8])] = &[
        (
            "structs+arrays",
            br#"struct P { int x, y; };
int dist2(struct P *a, struct P *b) { int dx = a->x - b->x, dy = a->y - b->y; return dx * dx + dy * dy; }
int main(void) { struct P ps[3] = {{0, 0}, {3, 4}, {6, 8}}; int s = 0; for (int i = 0; i < 2; i++) s += dist2(&ps[i], &ps[i + 1]); return s & 0xff; }
"#,
        ),
        (
            "enum+fnptr+switch",
            br#"enum Op { ADD, SUB, MUL };
int apply(enum Op o, int a, int b) { switch (o) { case ADD: return a + b; case SUB: return a - b; case MUL: return a * b; } return 0; }
int main(void) { int (*f)(enum Op, int, int) = apply; int s = 0; for (enum Op o = ADD; o <= MUL; o++) s += f(o, 7, 3); return s & 0xff; }
"#,
        ),
        (
            "union+bitfield+cast",
            br#"union U { unsigned int u; struct { unsigned a : 4, b : 4, c : 24; } f; };
int main(void) { union U x; x.u = 0xABCDEF12u; unsigned t = (unsigned)x.f.a + x.f.b; return (int)(t + (x.u >> 28)) & 0xff; }
"#,
        ),
        (
            "goto+ternary+comma",
            br#"int main(void) { int i = 0, s = 0; start: s += i; i++; if (i < 5) goto start; int t = (s > 10) ? (s * 2, s + 1) : 0; do { t--; } while (t > 20); return t & 0xff; }
"#,
        ),
        (
            "globals+static+strlit",
            br#"static const char *msg = "hello";
int counter(void) { static int n = 0; return ++n; }
int main(void) { int s = 0; for (const char *p = msg; *p; p++) s += *p; for (int i = 0; i < 3; i++) s += counter(); return s & 0xff; }
"#,
        ),
        (
            "longlong+shifts+2d",
            br#"int main(void) { long long acc = 0; int m[3][3]; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) m[i][j] = (i << 4) | j; for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) acc += m[i][j] * 0x10LL; return (int)(acc >> 3) & 0xff; }
"#,
        ),
        (
            "designated+compound",
            br#"struct R { int a, b, c; };
int main(void) { struct R r = {.a = 1, .c = 3}; int arr[5] = {[2] = 9, [4] = 1}; int n = sizeof arr / sizeof arr[0]; struct R q = (struct R){.b = 5}; return (r.a + r.c + arr[2] + arr[4] + n + q.b) & 0xff; }
"#,
        ),
    ];
    for (name, src) in corpus {
        assert_eq!(
            cc1_compile(&linked, &[b"chibicc", b"-"], src, vec![], vec![]),
            native_cc1(&[], src),
            "corpus `{name}`: guest-executed compiler emits byte-identical IR to native"
        );
    }
}
