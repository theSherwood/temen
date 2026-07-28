//! §12 per-vCPU TLS register (`vcpu.tls.get`/`set`) on the **bytecode engine**.
//!
//! The fused bytecode engine previously had no `Op` for `vcpu.tls`, so it fell through to the
//! reference `eval_inst`, which traps `Malformed` (no vCPU context) — folding any `vcpu.tls`-using
//! module back to the tree-walker. It now lowers to dedicated `Op::VcpuTlsGet`/`Op::VcpuTlsSet`,
//! reading/writing the running `Vm`'s `tls` word (seeded to the dense vCPU id, root = 0). These
//! tests pin the bytecode engine's behaviour to the tree-walker oracle: identical results, and the
//! bytecode engine actually *runs* the module (its `compile_and_run` returns `Some`, not a decline).

use svm_interp::{bytecode, run_with_host, run_with_host_fast, Host, Value};

/// Read the root vCPU's TLS seed (0), overwrite it with 42, read it back, return the sum (0 + 42).
const GET_SET: &str = r#"
func () -> (i64) {
block 0 () {
  a = vcpu.tls.get
  v42 = i64.const 42
  vcpu.tls.set v42
  b = vcpu.tls.get
  r = i64.add a b
  return r
  }
}
"#;

/// Bare read of the root seed — must be 0 (the dense vCPU id of the root).
const GET_ROOT_SEED: &str = r#"
func () -> (i64) {
block 0 () {
  a = vcpu.tls.get
  return a
  }
}
"#;

fn tree_walk(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = svm_text::parse_module(src).expect("parse");
    let mut fuel = u64::MAX;
    run_with_host(&m, 0, &[], &mut fuel, &mut Host::new())
}

fn bytecode_fast(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = svm_text::parse_module(src).expect("parse");
    let mut fuel = u64::MAX;
    run_with_host_fast(&m, 0, &[], &mut fuel, &mut Host::new())
}

/// The bytecode engine must *natively* run the module (not decline to the tree-walker), else the op
/// coverage this test asserts would be vacuously provided by the fallback.
fn bytecode_native(src: &str) -> Result<Vec<Value>, svm_interp::Trap> {
    let m = svm_text::parse_module(src).expect("parse");
    let mut fuel = u64::MAX;
    bytecode::compile_and_run(&m, 0, &[], &mut fuel)
        .expect("bytecode engine must lower vcpu.tls natively (not decline)")
}

#[test]
fn vcpu_tls_get_set_matches_tree_walker() {
    assert_eq!(bytecode_fast(GET_SET), tree_walk(GET_SET));
    assert_eq!(bytecode_native(GET_SET), Ok(vec![Value::I64(42)]));
}

#[test]
fn vcpu_tls_root_seed_is_zero() {
    assert_eq!(bytecode_fast(GET_ROOT_SEED), tree_walk(GET_ROOT_SEED));
    assert_eq!(bytecode_native(GET_ROOT_SEED), Ok(vec![Value::I64(0)]));
}
