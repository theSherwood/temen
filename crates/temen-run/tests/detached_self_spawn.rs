//! The playground's **detached-child demo** (`browser/web/play.js`, the `detached` card), pinned on
//! every engine: a plain powerbox guest resolves `instantiator`, `addrspace`, `budget` and `module` by
//! name, mints a `SharedRegion`, maps it into its own window, and spawns **its own func 1** as a §5
//! detached child with the region pre-mapped (`Instantiator` op 15, the 11-arg form) — no separate
//! child module, no grant list, no `map` in the child. The child reads the parent's word through the
//! alias, writes its answer beside it, returns; the parent reads the answer back after `join`.
//!
//! Three things this pins: the by-name spawn grants both reference powerboxes make
//! (`Host::grant_detached_spawn_caps` — `"module"` is the running module, `"budget"` a window's worth
//! of detached memory); the cooperative executor servicing op 15 as a fresh-window task (the bytecode
//! engine's `drive`, what the browser on-ramp runs on); and the CLI's JIT resolving a `Module` grant
//! for the spawn (`module_resolver`). The tree-walk oracle is the reference.

use std::sync::Arc;
use temen_interp::{bytecode, run_with_host, Host, Value};

/// The card's source, verbatim.
const DEMO: &str = r#"memory 17
data 20480 "instantiator"
data 20496 "addrspace"
data 20512 "budget"
data 20528 "module"
export 0 func "_start" 0
func () -> (i64) {
block 0 () {
  vp0 = i64.const 20480
  vl0 = i64.const 12
  vinst = self.resolve vp0 vl0
  vp1 = i64.const 20496
  vl1 = i64.const 9
  vas = self.resolve vp1 vl1
  vp2 = i64.const 20512
  vl2 = i64.const 6
  vbud = self.resolve vp2 vl2
  vp3 = i64.const 20528
  vl3 = i64.const 6
  vmod = self.resolve vp3 vl3
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vb = i64.extend_i32_u vbud
  vmh = i64.extend_i32_u vmod
  vz = i64.const 0
  vent = i64.const 1
  vlog = i64.const 17
  vreg = i64.extend_i32_u vrh
  vc = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vb, vmh, vz, vz, vent, vlog, vz, vz, vz, vreg, vwo)
  vj = call.cap 6 1 (i32) -> (i64) vinst (vc)
  vk = i64.const 1000
  vm = i64.mul vj vk
  vob = i64.const 65544
  vo = i64.load vob
  vr = i64.add vm vo
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 65536
  vin = i64.load va
  vtwo = i64.const 2
  vout = i64.mul vin vtwo
  vb = i64.const 65544
  i64.store vb vout
  vone = i64.const 1
  vr = i64.add vin vone
  return vr
  }
}
"#;

/// The powerbox the browser on-ramp and the CLI both present (`grant_onramp_caps` /
/// `grant_powerbox_prefix`): the §3e prefix, the by-name `instantiator`, and the spawn companions.
fn powerbox(m: &temen_ir::Module) -> Host {
    let win = 1u64 << m.memory.expect("declares memory").size_log2;
    let mut h = Host::new();
    h.set_self_module(&Arc::new(m.clone()));
    h.set_region_factory(temen_run::new_shared_region);
    h.grant_powerbox_prefix(win);
    let inst = h.grant_instantiator(0, win);
    h.register_cap_name("instantiator", inst);
    h.grant_detached_spawn_caps(win);
    h
}

const EXPECT: i64 = 42 * 1000 + 82;

#[test]
fn self_spawned_detached_child_over_a_premapped_region_on_every_engine() {
    let m = temen_text::parse_module(DEMO).expect("parse");
    temen_verify::verify_module(&m).expect("verify");

    let mut h = powerbox(&m);
    let mut fuel = 50_000_000u64;
    let oracle = run_with_host(&m, 0, &[], &mut fuel, &mut h).expect("tree-walk runs");
    assert_eq!(oracle, vec![Value::I64(EXPECT)], "tree-walk oracle");

    let mut h = powerbox(&m);
    let mut fuel = 50_000_000u64;
    let bc = bytecode::compile_and_run_with_host(&m, 0, &[], &mut fuel, &mut h)
        .expect("bytecode compiles")
        .expect("bytecode runs");
    assert_eq!(bc, vec![Value::I64(EXPECT)], "bytecode executor");

    let run = temen_run::run_powerbox(&m, b"").expect("CLI powerbox (JIT) runs");
    assert_eq!(
        run.outcome,
        temen_run::Outcome::Returned(vec![Value::I64(EXPECT)]),
        "CLI powerbox on the JIT"
    );
}
