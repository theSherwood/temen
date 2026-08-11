//! SSE `_ps` intrinsics (xmmintrin.h) end-to-end through the chibicc frontend: `__m128` is a
//! 16-byte SIMD value type that lowers to the IR `v128`, and the `_mm_*_ps` builtins lower to
//! `f32x4`/`v128` ops. Each program is compiled `#include <xmmintrin.h>` (the seeded playground
//! header) and run on all three backends — the tree-walk oracle, the bytecode engine (what the
//! browser runs), and the Cranelift JIT — which must agree on the result (the §3 parity invariant).
//!
//! Gated `#![cfg(unix)]` (needs the chibicc toolchain, like the other frontend suites).
#![cfg(unix)]

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use svm_run::{Backend, Outcome, RunConfig, Value};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

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

fn c_to_ir(src: &str) -> String {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let id = N.fetch_add(1, Ordering::Relaxed);
    let base = std::env::temp_dir().join(format!("svm_simd_{}_{id}", std::process::id()));
    let cfile = base.with_extension("c");
    let irfile = base.with_extension("svm");
    std::fs::write(&cfile, src).unwrap();
    // `-I<playground-include>` (attached form — cc1 mode only accepts `-Ipath`, not `-I path`) so
    // `#include <xmmintrin.h>` resolves to the seeded guest header.
    let inc = repo_root().join("browser/playground-include");
    let status = Command::new(chibicc())
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
        .expect("run chibicc");
    assert!(status.success(), "chibicc failed on:\n{src}");
    std::fs::read_to_string(&irfile).unwrap()
}

fn run(backend: Backend, ir: &str) -> i64 {
    let module = svm_text::parse_module(ir).expect("parse chibicc IR");
    let inst = svm_run::instantiate(module).expect("instantiate (resolve + verify)");
    let cfg = RunConfig::default();
    let run = inst.run(backend, &cfg).expect("run");
    match run.outcome {
        Outcome::Exited(c) => c as i64,
        Outcome::Returned(ref vals) => match vals.first() {
            Some(Value::I32(x)) => *x as i64,
            Some(Value::I64(x)) => *x,
            _ => 0,
        },
    }
}

/// Compile `src`, run on all three backends, assert they agree and equal `expect`.
fn check(src: &str, expect: i64) {
    let ir = c_to_ir(src);
    let tw = run(Backend::TreeWalk, &ir);
    let bc = run(Backend::Bytecode, &ir);
    let jit = run(Backend::Jit, &ir);
    assert_eq!(tw, expect, "tree-walk result\nIR:\n{ir}");
    assert_eq!(bc, tw, "bytecode disagrees with oracle\nIR:\n{ir}");
    assert_eq!(jit, tw, "cranelift disagrees with oracle\nIR:\n{ir}");
}

/// `_mm_set_ps` (lane order), `_mm_set1_ps` (broadcast), `_mm_mul_ps`, `_mm_storeu_ps`:
/// {1,2,3,4} * 10 = {10,20,30,40} → 100.
#[test]
fn set_broadcast_mul_store() {
    check(
        r#"#include <xmmintrin.h>
int main() {
  __m128 a = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);
  __m128 b = _mm_set1_ps(10.0f);
  __m128 c = _mm_mul_ps(a, b);
  float out[4];
  _mm_storeu_ps(out, c);
  return (int)(out[0] + out[1] + out[2] + out[3]);
}"#,
        100,
    );
}

/// `_mm_load_ps`, `_mm_min_ps`/`_mm_max_ps` (pmin/pmax = SSE minps/maxps), `_mm_add_ps`,
/// `_mm_sub_ps`, `_mm_setzero_ps`: min({1,5,3,8},4)+max({1,5,3,8},4) = {5,9,7,12} → 33.
#[test]
fn load_minmax_add_sub_zero() {
    check(
        r#"#include <xmmintrin.h>
int main() {
  float in[4] = {1.0f, 5.0f, 3.0f, 8.0f};
  __m128 v = _mm_load_ps(in);
  __m128 lo = _mm_set1_ps(4.0f);
  __m128 s = _mm_add_ps(_mm_min_ps(v, lo), _mm_max_ps(v, lo));
  s = _mm_sub_ps(s, _mm_setzero_ps());
  float out[4];
  _mm_storeu_ps(out, s);
  return (int)(out[0] + out[1] + out[2] + out[3]);
}"#,
        33,
    );
}

/// The lesson pattern: a SIMD loop over a `float` array with `_mm_loadu_ps`/`_mm_storeu_ps`,
/// doubling four lanes at a time. {1,2,3,4}*2 summed → 20.
#[test]
fn loadu_storeu_loop() {
    check(
        r#"#include <xmmintrin.h>
int main() {
  float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  __m128 two = _mm_set1_ps(2.0f);
  __m128 v = _mm_loadu_ps(&data[0]);
  v = _mm_mul_ps(v, two);
  _mm_storeu_ps(&data[0], v);
  int sum = 0;
  for (int i = 0; i < 4; i = i + 1) sum = sum + (int)data[i];
  return sum;
}"#,
        20,
    );
}

/// `sizeof(__m128)` is 16 and a `__m128` local can be reassigned through a variable (window slot
/// load/store), exercising the non-promoted memory path.
#[test]
fn sizeof_and_reassign() {
    check(
        r#"#include <xmmintrin.h>
int main() {
  __m128 v = _mm_set1_ps(1.0f);
  v = _mm_add_ps(v, _mm_set1_ps(2.0f));
  v = _mm_add_ps(v, v);
  float out[4];
  _mm_storeu_ps(out, v);
  return (int)sizeof(__m128) + (int)out[0]; /* 16 + 6 = 22 */
}"#,
        22,
    );
}
