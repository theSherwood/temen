//! The browser playground's `<stdatomic.h>` compiles and runs on the IR backends. Regression for
//! the `atomic_compare_exchange_strong` macro: it used a GCC statement-expression, which the
//! browser IR-codegen subset rejects (`unsupported expression`), so any lesson using CAS failed to
//! compile. Rewritten as a small helper, it now compiles here (via the seeded playground header)
//! and produces C11 semantics on the tree-walk oracle, the bytecode engine, and the Cranelift JIT.
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like the other frontend suites).
#![cfg(unix)]

#[path = "support/repo_root.rs"]
mod repo_root_mod;
use repo_root_mod::repo_root;

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_run::{Backend, Outcome, RunConfig, Value};

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

/// Compile with the seeded playground include dir (attached `-I` — cc1 only accepts `-Ipath`).
fn c_to_ir(src: &str) -> String {
    let base = std::env::temp_dir().join(format!("svm_pgatomic_{}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    let inc = repo_root().join("browser/playground-include");
    let ok = Command::new(chibicc())
        .args([
            "-cc1",
            "--emit-ir",
            &format!("-I{}", inc.to_str().unwrap()),
            "-cc1-input",
            cfile.to_str().unwrap(),
            "-cc1-output",
            irfile.to_str().unwrap(),
            cfile.to_str().unwrap(),
        ])
        .status()
        .expect("run chibicc")
        .success();
    assert!(ok, "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

fn run(backend: Backend, ir: &str) -> i64 {
    let module = svm_text::parse_module(ir).expect("parse chibicc IR");
    let inst = svm_run::instantiate(module).expect("instantiate (resolve + verify)");
    match inst
        .run(backend, &RunConfig::default())
        .expect("run")
        .outcome
    {
        Outcome::Exited(c) => c as i64,
        Outcome::Returned(ref v) => match v.first() {
            Some(Value::I32(x)) => *x as i64,
            Some(Value::I64(x)) => *x,
            _ => 0,
        },
    }
}

/// `atomic_compare_exchange_strong` C11 semantics on all three backends:
/// - success swaps and returns 1;
/// - failure leaves the target, returns 0, and writes the current value back through `expected`.
#[test]
fn compare_exchange_strong_semantics() {
    let ir = c_to_ir(
        r#"#include <stdatomic.h>
int main() {
  int v = 5, exp = 5;
  int ok1 = atomic_compare_exchange_strong(&v, &exp, 9);  /* swaps: v=9, ok1=1 */
  int exp2 = 5;                                            /* stale expected */
  int ok2 = atomic_compare_exchange_strong(&v, &exp2, 100);/* fails: v=9, ok2=0, exp2<-9 */
  return ok1 * 1000 + ok2 * 100 + v + exp2;               /* 1000 + 0 + 9 + 9 = 1018 */
}"#,
    );
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        assert_eq!(run(backend, &ir), 1018, "{backend:?}\nIR:\n{ir}");
    }
}
