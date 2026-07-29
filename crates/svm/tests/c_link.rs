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
//! Scope: cross-TU *functions*. Cross-TU *data* (a global defined in one unit, referenced in
//! another → `LinkUnit::data_exports` + relocations) and running a linked program through `_start`
//! under the powerbox are the next slices; here the linked functions are run directly by index
//! (as `dynlink.rs` does), which is all the function-linking mechanism needs to prove.

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
    LinkUnit {
        module: m,
        exports,
        ..Default::default()
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
