//! §12 per-vCPU TLS register (`vcpu.tls.get`/`set`) on the **bytecode engine**.
//!
//! The fused bytecode engine previously had no `Op` for `vcpu.tls`, so it fell through to the
//! reference `eval_inst`, which traps `Malformed` (no vCPU context) — folding any `vcpu.tls`-using
//! module back to the tree-walker. It now lowers to dedicated `Op::VcpuTlsGet`/`Op::VcpuTlsSet`,
//! reading/writing the running `Vm`'s `tls` word (seeded to the dense vCPU id, root = 0). These
//! tests pin the bytecode engine's behaviour to the tree-walker oracle: identical results, and the
//! bytecode engine actually *runs* the module (its `compile_and_run` returns `Some`, not a decline).

use temen_interp::{bytecode, run_with_host, run_with_host_fast, Host, Value};

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

fn tree_walk(src: &str) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = temen_text::parse_module(src).expect("parse");
    let mut fuel = u64::MAX;
    run_with_host(&m, 0, &[], &mut fuel, &mut Host::new())
}

fn bytecode_fast(src: &str) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = temen_text::parse_module(src).expect("parse");
    let mut fuel = u64::MAX;
    run_with_host_fast(&m, 0, &[], &mut fuel, &mut Host::new())
}

/// The bytecode engine must *natively* run the module (not decline to the tree-walker), else the op
/// coverage this test asserts would be vacuously provided by the fallback.
fn bytecode_native(src: &str) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = temen_text::parse_module(src).expect("parse");
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

/// Two `thread.spawn`ed threads, each returning its own `vcpu.tls` seed; the root joins both and
/// returns `first * 100 + second`.
const THREAD_SEEDS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  va = i64.const 0
  vh = thread.spawn 1 vsp va
  vh2 = thread.spawn 1 vsp va
  vj = thread.join vh
  vj2 = thread.join vh2
  vk = i64.const 100
  vm = i64.mul vj vk
  vr = i64.add vm vj2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  t = vcpu.tls.get
  return t
  }
}
"#;

/// A spawned thread's seed is its dense vCPU id (the root is 0, its threads 1 and 2) on every
/// driver: the tree-walker oracle, the cooperative pump, the parallel driver and the debugger. The
/// debugger's scheduler used to leave every thread at 0.
#[test]
fn a_spawned_thread_is_seeded_with_its_vcpu_id_on_every_driver() {
    let m = temen_text::parse_module(THREAD_SEEDS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let want = Ok(vec![Value::I64(102)]);
    assert_eq!(tree_walk(THREAD_SEEDS), want, "tree-walker");
    assert_eq!(bytecode_native(THREAD_SEEDS), want, "cooperative pump");

    let back = std::sync::Arc::new(temen_interp::Region::new(1 << 16, 4096));
    let mut fuel = u64::MAX;
    let parallel =
        bytecode::compile_and_run_capture_over_parallel(&m, 0, &[], &mut fuel, &[], back)
            .expect("parallel driver supports the module")
            .0;
    assert_eq!(parallel, want, "parallel driver");

    let mut run = bytecode::ScheduledDebugRun::new_with_host(&m, 0, &[], Host::new())
        .expect("debug engine supports the module");
    let mut fuel = 10_000_000u64;
    let debugged = loop {
        match run.run_until_stop(&mut fuel) {
            bytecode::SchedStop::Finished(r) => break r,
            bytecode::SchedStop::Break { .. } => continue,
            other => panic!("the debug run ends, got {other:?}"),
        }
    };
    assert_eq!(debugged, want, "debugger");
}
