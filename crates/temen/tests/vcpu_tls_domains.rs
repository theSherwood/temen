//! #1775 — a vCPU's `vcpu.tls` word starts at its id **in its domain**, on every engine: each
//! interpreter driver and the Cranelift JIT. A domain's root starts at 0, a §14 child's root among
//! them, and its threads at 1, 2, … in the order the domain spawns them, whatever other domains
//! spawned in between.
//!
//! Before, only the JIT numbered within the domain. The tree-walker seeded each vCPU with its
//! run-wide task id, so a child's root started nonzero; the cooperative and debug drivers numbered
//! threads across the run; and the parallel driver's count took in the child's spawn too.
//!
//! The `Vcpu` driver, the browser's per-Worker path, still numbers threads across the run: its
//! threads hold no handle on their domain (#2236).

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use drivers::{agree_on, Driver, Ran};
use temen_interp::{Host, MemLayout, Value};
use temen_ir::{Module, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitError, JitOutcome};

/// The root's window: `memory 18`.
const WIN: usize = 256 << 10;

/// The root (args `(instantiator, CHILD, budget)`): a thread, then a detached `CHILD`, then another
/// thread, each joined before the next starts. Returns the first thread's seed, the child's
/// result and the second thread's seed.
const ROOT: &str = "memory 18
func (i32, i32, i32) -> (i64, i64, i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32) {
  z = i64.const 0
  ha = thread.spawn 1 z z
  ta = thread.join ha
  vm = i64.extend_i32_s vmod
  vb = i64.extend_i32_s vbud
  sl = i64.const 16
  hc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, vm, z, z, z, sl, z)
  rc = call.cap 6 1 (i32) -> (i64) vinst (hc)
  hb = thread.spawn 1 z z
  tb = thread.join hb
  return ta, rc, tb
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  t = vcpu.tls.get
  return t
  }
}
";

/// The child: its own seed, then two threads', as the digits `root·100 + first·10 + second`.
const CHILD: &str = "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  r0 = vcpu.tls.get
  z = i64.const 0
  h1 = thread.spawn 1 z z
  r1 = thread.join h1
  h2 = thread.spawn 1 z z
  r2 = thread.join h2
  k100 = i64.const 100
  k10 = i64.const 10
  a = i64.mul r0 k100
  b = i64.mul r1 k10
  s = i64.add a b
  r = i64.add s r2
  return r
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  t = vcpu.tls.get
  return t
  }
}
";

/// The root's threads are its 1 and 2, the child's root is 0 and its threads 1 and 2.
const WANT: [i64; 3] = [1, 12, 2];

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

fn setup() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, WIN as u64);
    let child = host.grant_module(&module(CHILD));
    let budget = host.grant_budget(-1, 1 << 20, -1);
    (host, [inst, child, budget].map(Value::I32).to_vec())
}

#[test]
fn a_domains_vcpus_are_numbered_within_it() {
    let m = module(ROOT);
    let want = Ran {
        result: Ok(WANT.map(Value::I64).to_vec()),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    let drivers = [
        Driver::Oracle,
        Driver::Coop,
        Driver::Parallel,
        Driver::Debug,
    ];
    agree_on(&drivers, "the vcpu.tls seeds", &m, &setup, &want);
    let (mut host, args) = setup();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(h) => i64::from(*h),
            other => panic!("a handle, not {other:?}"),
        })
        .collect();
    let init = MemLayout::image(vec![0u8; WIN]);
    match temen_run::jit_cap_run(
        &m,
        0,
        &args,
        &init,
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    ) {
        Ok((o, _)) => assert_eq!(o, JitOutcome::Returned(WANT.to_vec()), "Cranelift"),
        Err(JitError::Unsupported(_)) => {} // a target without the thread runtime
        Err(e) => panic!("the JIT run failed: {e:?}"),
    }
}
