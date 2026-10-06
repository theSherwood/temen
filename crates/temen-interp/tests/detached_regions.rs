//! #2111 — **a guest-minted region spends `Budget.channel`**, on every driver.
//!
//! A detached child's starter `AddressSpace` mints §13 regions (`create_region`, op 5). The region's
//! bytes are charged to the child's own node (the budget that paid for its window) and every ancestor,
//! as a pipe's FIFO is: past the ceiling the mint refuses with `-ENOMEM` and charges nothing. A region
//! goes back once no domain holds it, and a domain lets go of its regions when it ends.
//!
//! The root splits its budget into a node whose `channel` ceiling is one 64 KiB region, spawns the
//! child paid from it, joins it, and reads the node's `channel` room. The child mints a 64 KiB region
//! (it fits) and asks for a second (refused). It ends holding the first, which its end hands back: the
//! room is the whole ceiling again.

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/rec.rs"]
mod rec;

use std::sync::Arc;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, Value};
use temen_ir::{Module, SpawnRec};

/// The node's `channel` ceiling: one 64 KiB region.
const CEILING: i64 = 1 << 16;

/// A root that pays for func 1 from a node whose `channel` ceiling is [`CEILING`], and returns
/// `child_result * 1_000_000 + room`, the node's `channel` room once the child has ended. The child
/// returns a bit per expected answer: 1 the first region is minted, 2 the second is `-ENOMEM`.
fn src() -> String {
    format!(
        "memory 16
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32) {{
  f = i64.const -1
  c = i64.const {CEILING}
  vsub = call.cap 14 0 (i64, i64, i64, i64) -> (i64) vbud (f, f, f, c)
  vnode = i32.wrap_i64 vsub
  rb = i64.const 17436
  i32.store rb vnode
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  three = i64.const 3
  vroom = call.cap 14 1 (i64) -> (i64) vnode (three)
  k = i64.const 1000000
  vhi = i64.mul vr k
  vsum = i64.add vhi vroom
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas64: i64) {{
  vas = i32.wrap_i64 vas64
  n = i64.const {CEILING}
  z = i64.const 0
  enomem = i64.const -12
  r1 = call.cap 5 5 (i64) -> (i64) vas (n)
  r2 = call.cap 5 5 (i64) -> (i64) vas (n)
  b1 = i64.ge_s r1 z
  b2 = i64.eq r2 enomem
  w1 = i64.extend_i32_u b1
  w2 = i64.extend_i32_u b2
  two = i64.const 2
  x2 = i64.mul w2 two
  s = i64.add w1 x2
  return s
  }}
}}
{rec}",
        rec = rec::segment(17408, &SpawnRec::v1(1)),
    )
}

#[test]
fn a_minted_region_spends_the_childs_channel_until_it_ends_on_every_driver() {
    let m: Module = temen_text::parse_module(&src()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let setup = || {
        let mut h = Host::new();
        h.set_self_module(&Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 16);
        let b = h.grant_budget(-1, -1, -1);
        (h, vec![Value::I32(i), Value::I32(b)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(3 * 1_000_000 + CEILING)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver("create_region", &m, &setup, &want);
}
