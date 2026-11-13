//! #2961 — a child's **grouped import** of its parent's self-served interface reaches the op each
//! call names, on every driver. The parent serves `calc { add, sub, neg }` through the live
//! self-serve grant (#744) and loops `svc.wait` until its child is gone, then joins. The child is its
//! own module and imports the interface twice, through two slots bound to the same grant:
//!
//! - slot 0 as `calc { neg, sub, add }`, the provider's ops in another order, so each call goes
//!   through the slot's bind-time remap, and `add` and `sub`, which share a shape, bind by name;
//! - slot 1 as `calc2 { add }`, so a call there is op 0 of slot 1, whose packed `call.import`
//!   operand (`slot | op << 16`) is `1`.
//!
//! Each of those once reached the wrong op: the bytecode engine and the Cranelift JIT read the packed
//! operand as the op, the tree-walker added the call's op to the slot's first, and the binder matched
//! a live grant by shape alone. The child answers `neg(5) + 100·sub(10, 3) + 10⁴·add(1, 2) +
//! 10⁶·add(100, 1)` and the parent `join · 10 + served`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::sync::Arc;

use temen_interp::Value;
use temen_ir::SpawnRec;
use temen_run::{instantiate, Backend, HostCap, Outcome, RunConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

/// The parent: spawn the child with `calc` and `calc2` both naming export 0 as a live offer, serve
/// until `svc.wait` answers 0, join.
fn parent() -> String {
    let child = SpawnRec {
        grants_ptr: 16640,
        grants_n: 2,
        ..SpawnRec::default()
    };
    format!(
        "memory 17
data 16384 \"vm\"
data 16400 \"budget\"
data 16448 \"child\"
data 16684 \"calc\"
data 16692 \"calc2\"
{rec}type 0 func (i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 interface {{ add: 0, sub: 0, neg: 1 }}
export 0 interface \"calc\" 2 {{ add: 1, sub: 2, neg: 3 }}
func 0 () -> (i32) {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vba = i64.const {budget_at}
  i32.store vba vb
  vcp = i64.const 16448
  vcl = i64.const 5
  vchild = self.resolve vcp vcl
  vma = i64.const {module_at}
  i32.store vma vchild
  vgh = i32.const {tag}
{g0}{g1}  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vh (vrp)
  vn0 = i64.const 0
  vz = i32.const 0
  vneg = i32.lt_s vch vz
  br_if vneg 3(vch) 1(vh, vch, vn0)
}}
block 1 (vh1: i32, vch1: i32, vn: i64) {{
  vz1 = i32.const 0
  vw = svc.wait vz1
  vzero = i64.const 0
  vdone = i64.eq vw vzero
  vn2 = i64.add vn vw
  br_if vdone 2(vh1, vch1, vn2) 1(vh1, vch1, vn2)
}}
block 2 (vh2: i32, vch2: i32, vn3: i64) {{
  vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) vh2 (vch2)
  vk = i64.const 10
  vt = i64.mul vj vk
  vs = i64.add vt vn3
  vc = i32.wrap_i64 vs
  return vc
}}
block 3 (verr: i32) {{
  return verr
}}
}}
func 1 (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  s = i64.add va vb
  return s
  }}
}}
func 2 (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  s = i64.sub va vb
  return s
  }}
}}
func 3 (i64) -> (i64) {{
block 0 (va: i64) {{
  vz = i64.const 0
  s = i64.sub vz va
  return s
  }}
}}
",
        rec = rec::segment(17408, &child),
        g0 = rec::grant("ga", 16640, 16684, 4, "vgh"),
        g1 = rec::grant("gb", 16656, 16692, 5, "vgh"),
        tag = temen_interp::GRANT_SERVE_LIVE_TAG as i32,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        module_at = 17408 + SpawnRec::MODULE_AT,
    )
}

const CHILD: &str = "memory 17
type 0 func (i64, i64) -> (i64)
type 1 func (i64) -> (i64)
type 2 interface { neg: 1, sub: 0, add: 0 }
type 3 interface { add: 0 }
import 0 interface \"calc\" 2
import 1 interface \"calc2\" 3
func 0 (i64) -> (i64) {
block 0 (v0: i64) {
  v5 = i64.const 5
  vn = call.import 0.neg (v5)
  v10 = i64.const 10
  v3 = i64.const 3
  vs = call.import 0.sub (v10, v3)
  v1 = i64.const 1
  v2 = i64.const 2
  va = call.import 0.add (v1, v2)
  v100 = i64.const 100
  vb = call.import 1.add (v100, v1)
  vk2 = i64.const 100
  vk4 = i64.const 10000
  vk6 = i64.const 1000000
  ms = i64.mul vs vk2
  ma = i64.mul va vk4
  mb = i64.mul vb vk6
  r1 = i64.add vn ms
  r2 = i64.add r1 ma
  r3 = i64.add r2 mb
  return r3
  }
}
";

fn run(backend: Option<Backend>) -> Result<i32, String> {
    let pm = parse_module(&parent()).expect("parse parent");
    verify_module(&pm).expect("verify parent");
    let cm = parse_module(CHILD).expect("parse child");
    verify_module(&cm).expect("verify child");
    let cm = Arc::new(cm);
    let inst = instantiate(pm).expect("instantiate");
    let caps = [
        ("vm", HostCap::custom(6, 0, |h, _| h.grant_instantiator())),
        ("budget", HostCap::detached_budget(1 << 20)),
        (
            "child",
            HostCap::custom(temen_ir::cap_id::MODULE, 0, move |h, _| h.grant_module(&cm)),
        ),
    ];
    let cfg = RunConfig::default();
    let r = match backend {
        Some(b) => inst.run_with_caps(b, &cfg, &caps),
        None => inst.run_with_caps_parallel(&cfg, &caps),
    }
    .map_err(|e| e.to_string())?;
    match r.outcome {
        Outcome::Returned(v) => match v[..] {
            [Value::I32(x)] => Ok(x),
            _ => Err(format!("result {v:?}")),
        },
        o => Err(format!("outcome {o:?}")),
    }
}

#[test]
fn each_import_call_reaches_the_op_it_names() {
    let child = -5 + 100 * 7 + 10_000 * 3 + 1_000_000 * 101;
    for b in [
        Some(Backend::TreeWalk),
        Some(Backend::Bytecode),
        Some(Backend::Jit),
        None,
    ] {
        assert_eq!(run(b), Ok(child * 10 + 4), "{b:?}");
    }
}
