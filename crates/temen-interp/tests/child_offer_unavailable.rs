//! #1732 — a driver that cannot mint a `child_offer` (op 14) answers `-EINVAL`, the value the oracle
//! gives for a child it has nothing to offer over. Minting needs a live child's powerbox, which only
//! the cooperative scheduler keeps; the parallel driver and the Cranelift nursery without a mint
//! hook already answered `-EINVAL`, while the single-vCPU `Vcpu::run` trapped `ThreadFault`. An
//! unavailable seam is a value, not a trap (INVARIANTS #5), and one op gets one answer whichever
//! loop drives it (#9).
//!
//! (The debug scheduler's `Declined` is not in this set: it refuses the whole debug run rather than
//! answering the guest — the observability corollary's "a tool that can't see something refuses".)

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

/// `child_offer(child = 0, export = 0)` on a domain that never spawned — no child to offer over.
const GUEST: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vi: i32) {
  vc = i32.const 0
  vx = i64.const 0
  vo = call.cap 6 14 (i32, i64) -> (i32) vi (vc, vx)
  vr = i64.extend_i32_s vo
  return vr
  }
}
"#;

fn guest() -> temen_ir::Module {
    let m = parse_module(GUEST).expect("parse");
    verify_module(&m).expect("verify");
    m
}

fn oracle() -> Result<Vec<Value>, Trap> {
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 16);
    let mut fuel = 1_000_000u64;
    temen_interp::run_with_host(&guest(), 0, &[Value::I32(inst)], &mut fuel, &mut host)
}

/// The single-vCPU driver an embedder steps event by event (the browser's op-13 loops).
fn single_vcpu() -> Result<Vec<Value>, Trap> {
    let m = guest();
    let prog = bytecode::VcpuProgram::compile(&m).expect("compile");
    let size = 1usize << 16;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked for the test's lifetime — never freed.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes, owned here and never freed.
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });
    let mut host = Host::new();
    let inst = host.grant_instantiator(0, 1 << 16);
    let mut vcpu =
        bytecode::Vcpu::new_root_with_powerbox(&prog, 0, &[Value::I32(inst)], back, &[], host)
            .expect("root");
    match vcpu.run() {
        bytecode::VcpuEvent::Done(v) => Ok(v),
        bytecode::VcpuEvent::Trapped(t) => Err(t),
        _ => panic!("an event this guest cannot raise"),
    }
}

#[test]
fn child_offer_with_nothing_to_offer_is_einval_on_the_single_vcpu_driver() {
    assert_eq!(oracle(), Ok(vec![Value::I64(-22)]), "oracle");
    assert_eq!(single_vcpu(), Ok(vec![Value::I64(-22)]), "Vcpu::run");
}
