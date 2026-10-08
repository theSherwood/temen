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

use drivers::{agree_on, agree_on_every_driver, Ran, SCHEDULING};
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

/// #2189 — a root that maps a 64 KiB region at 65536 and spawns func 1 detached with the region
/// pre-mapped at the same offset. The child raises a flag at region byte 0 and waits on the word at
/// byte 8 (for at most 2 s). The root spins until the flag is up, notifies the word until the notify
/// wakes someone (or gives up), joins, and returns `woken * 10 + the child's wait status`.
fn shared_futex_src() -> String {
    let child = SpawnRec {
        child_off: 65536,
        ..SpawnRec::v1(1)
    };
    format!(
        "memory 17
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vas: i32, vbud: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) vas (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vba = i64.const {budget_at}
  i32.store vba vbud
  vra = i64.const {region_at}
  i32.store vra vrh
  vrp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vspin = i64.const 100000000
  br 1(vspin, vinst, vch)
}}
block 1 (vs: i64, vi1: i32, vc1: i32) {{
  vfa = i64.const 65536
  vf = i64.atomic.load vfa
  vz = i64.const 0
  vset = i64.ne vf vz
  vone = i64.const 1
  vs2 = i64.sub vs vone
  vgone = i64.eq vs2 vz
  vstop = i32.or vset vgone
  vn = i64.const 1000000
  br_if vstop 2(vn, vi1, vc1) 1(vs2, vi1, vc1)
}}
block 2 (vi: i64, vi2a: i32, vc2a: i32) {{
  vwa = i64.const 65544
  vc = i32.const 1
  vk = atomic.notify vwa vc
  vz2 = i32.const 0
  vhit = i32.ne vk vz2
  vone2 = i64.const 1
  vi2 = i64.sub vi vone2
  vz64 = i64.const 0
  vout = i64.eq vi2 vz64
  vend = i32.or vhit vout
  vk64 = i64.extend_i32_s vk
  br_if vend 3(vk64, vi2a, vc2a) 2(vi2, vi2a, vc2a)
}}
block 3 (vcount: i64, vi3: i32, vc3: i32) {{
  vj = call.cap 6 1 (i32) -> (i64) vi3 (vc3)
  v10 = i64.const 10
  vm = i64.mul vcount v10
  vr = i64.add vm vj
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vfa = i64.const 65536
  vone = i64.const 1
  i64.atomic.store vfa vone
  vwa = i64.const 65544
  vexp = i32.const 0
  vto = i64.const 2000000000
  vw = i32.atomic.wait vwa vexp vto
  vw64 = i64.extend_i32_s vw
  return vw64
  }}
}}
{rec}",
        rec = rec::segment(17408, &child),
        budget_at = 17408 + rec::BUDGET_AT,
        region_at = 17408 + 72,
    )
}

/// #2189 — a region word that a parent and its child both map is one futex word: the parent's
/// notify wakes the child's wait on every scheduling driver (woken once, the child's wait `0`).
/// The parallel driver kept a futex table per domain, keyed on the raw address, so the notify
/// found no waiter and the child timed out.
#[test]
fn a_notify_on_a_region_word_wakes_the_childs_wait_on_every_scheduling_driver() {
    let m: Module = temen_text::parse_module(&shared_futex_src()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let setup = || {
        let mut h = Host::new();
        h.set_self_module(&Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 17);
        let a = h.grant_address_space(0, 1 << 17);
        let b = h.grant_budget(-1, 1 << 20, -1);
        (h, vec![Value::I32(i), Value::I32(a), Value::I32(b)])
    };
    let want = Ran {
        result: Ok(vec![Value::I64(10)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on(&SCHEDULING, "a shared region futex", &m, &setup, &want);
}
