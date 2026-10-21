//! The §14 spawn kernel the two **Miri** tests share (`vcpu_instantiate_miri.rs` and
//! `parallel_instantiate_miri.rs`). The root maps two `SharedRegion`s into its window and spawns two
//! detached children of a granted module, each with one of the regions pre-mapped into its own window
//! (the §13 data plane between windows). Each child reads the byte its data segment put in its window
//! (`K` = 75) and writes the marker 30 through its region; the root joins both (the happens-before that
//! publishes the writes) and reads both markers through its own mappings: 75 + 75 + 30 + 30. Real
//! cross-thread non-atomic access through two views of one backing. A test includes [`super::rec`]
//! beside it.
#![allow(dead_code)] // each test binary uses a different subset

use super::rec;
use temen_interp::{Host, Value};
use temen_ir::{Module, SpawnRec};

/// What the root returns.
pub const RESULT: i64 = 75 + 75 + 30 + 30;

/// The child: its window is its declared 128 KiB, and its region is pre-mapped at 64 KiB.
const CHILD: &str = r#"memory 17
data 17408 "K"
func (i64) -> (i64) {
block 0 (v0: i64) {
  vd = i64.const 17408
  vk = i32.load8_u vd
  vm = i64.const 65536
  v30 = i32.const 30
  i32.store8 vm v30
  vr = i64.extend_i32_u vk
  return vr
  }
}
"#;

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The root `(inst, aspace, child module, budget) -> i64`, over a 256 KiB window: region A mapped at
/// 64 KiB and region B at 128 KiB, each spawn's record at 17408 and 17536 with its handles stored in
/// at run time.
pub fn root() -> Module {
    let rec = SpawnRec {
        child_off: 65536,
        ..SpawnRec::default()
    };
    let (m, b, r) = (
        SpawnRec::MODULE_AT,
        SpawnRec::BUDGET_AT,
        SpawnRec::REGION_AT,
    );
    module(&format!(
        r#"memory 18
func (i32, i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vas: i32, vmod: i32, vbud: i32) {{
  vlen = i64.const 65536
  vra64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vra = i32.wrap_i64 vra64
  vrb64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrb = i32.wrap_i64 vrb64
  vwa = i64.const 65536
  vwb = i64.const 131072
  vro = i64.const 0
  vprot = i32.const 3
  vma = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vra (vwa, vro, vlen, vprot)
  vmb = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrb (vwb, vro, vlen, vprot)
  vpa = i64.const 17408
  i32.store vpa vmod offset={m}
  i32.store vpa vbud offset={b}
  i32.store vpa vra offset={r}
  vpb = i64.const 17536
  i32.store vpb vmod offset={m}
  i32.store vpb vbud offset={b}
  i32.store vpb vrb offset={r}
  vha = call.cap 6 17 (i64) -> (i32) vinst (vpa)
  vhb = call.cap 6 17 (i64) -> (i32) vinst (vpb)
  vja_st, vja = call.cap 6 1 (i32) -> (i64, i64) vinst (vha)
  vjb_st, vjb = call.cap 6 1 (i32) -> (i64, i64) vinst (vhb)
  vxa = i32.load8_u vwa
  vxb = i32.load8_u vwb
  vya = i64.extend_i32_u vxa
  vyb = i64.extend_i32_u vxb
  vs = i64.add vja vjb
  vt = i64.add vs vya
  vr = i64.add vt vyb
  return vr
  }}
}}
{}{}"#,
        rec::segment(17408, &rec),
        rec::segment(17536, &rec),
    ))
}

/// The root's powerbox and entry args: an `Instantiator` and `AddressSpace` over its window, the
/// child module, and a `Budget` that pays for both children's windows.
pub fn host() -> (Host, Vec<Value>) {
    let mut host = Host::new();
    let inst = host.grant_instantiator();
    let aspace = host.grant_address_space(0, 1 << 18);
    let child = host.grant_module(&module(CHILD));
    let budget = host.grant_budget(-1, 1 << 18, -1);
    let args = [inst, aspace, child, budget].map(Value::I32).to_vec();
    (host, args)
}
