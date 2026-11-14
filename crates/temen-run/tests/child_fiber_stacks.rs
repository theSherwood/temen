//! #3020 — a §14 child that ends with fibers parked on an event gives their stacks back, as a run's
//! root does: its domain's fiber table ends with it ([`temen_jit`]'s `finish_child`). Alone in its
//! binary: the count of live stacks is the process's.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use core::ffi::c_void;
use temen_interp::Host;
use temen_ir::SpawnRec;
use temen_jit::{CompiledModule, JitOutcome};

/// `v0` Instantiator, `v1` the child's module, `v2` its Budget: spawn the child detached from the
/// record at 17408 and join it; answer its result.
fn parent() -> String {
    let child = SpawnRec {
        size_log2: 16,
        ..SpawnRec::default()
    };
    format!(
        r#"memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32) {{
  vmod = i64.const {module_at}
  i32.store vmod v1
  vbud = i64.const {budget_at}
  i32.store vbud v2
  vrec = i64.const 17408
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrec)
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)
  return vj
  }}
}}
{rec}"#,
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    )
}

/// The child: make 8 fibers, resume each once: it waits on 16384, which nothing notifies, and parks.
/// Answer 8.
const CHILD: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vn = i64.const 8
  vz = i64.const 0
  br 1(vn, vz)
}
block 1 (vn1: i64, vi1: i64) {
  vlt = i64.lt_u vi1 vn1
  br_if vlt 2(vn1, vi1) 3(vn1)
}
block 2 (vn2: i64, vi2: i64) {
  vf = ref.func 1
  vsp = i64.const 4096
  vh = cont.new vf vsp
  va = i64.const 0
  vs, vr = cont.resume vh va
  vone = i64.const 1
  vin = i64.add vi2 vone
  br 1(vn2, vin)
}
block 3 (vn3: i64) {
  return vn3
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  vw = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.const 0
  return vr
  }
}
"#;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

#[test]
fn a_child_gives_back_the_stacks_of_fibers_it_left_parked() {
    if !temen_fiber::supported() {
        return;
    }
    let Some(_) = temen_fiber::live_stacks() else {
        return; // the guard-paged backend counts nothing
    };
    let p = module(&parent());
    let child = module(CHILD);
    let mut host = Box::new(Host::new());
    let inst = host.grant_instantiator();
    let cm_child = host.grant_module(&child);
    let budget = host.grant_budget(-1, 8i64 << 20, -1);
    let hp = &mut *host as *mut Host;
    let mut cm = CompiledModule::compile(
        &p,
        0,
        temen_run::cap_thunk,
        hp as *mut c_void,
        temen_ir::DEFAULT_RESERVED_LOG2,
        Some(temen_run::module_resolver),
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile");
    cm.set_grant_child_hooks(Some(temen_run::production_grant_hooks(
        temen_run::CapCtx::Raw(hp),
    )));
    let args = [inst as i64, cm_child as i64, budget as i64];
    // The first run warms whatever the pool keeps; every later one must leave the count where the
    // first did.
    let (out, _) = cm.run(&args, None, None).expect("run 1");
    assert_eq!(out, JitOutcome::Returned(vec![8]), "run 1");
    let after_one = temen_fiber::live_stacks();
    for run in 2..=5 {
        let (out, _) = cm.run(&args, None, None).expect("run");
        assert_eq!(out, JitOutcome::Returned(vec![8]), "run {run}");
        assert_eq!(
            temen_fiber::live_stacks(),
            after_one,
            "run {run}'s child left its parked fibers' stacks live"
        );
    }
}
