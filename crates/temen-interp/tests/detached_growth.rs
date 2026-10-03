//! #1909, INVARIANTS #3 R2 — **growing a detached window spends `Budget.mem`**, on every driver.
//!
//! A detached child's starter `AddressSpace` spans its whole reservation, so it can commit pages past
//! its declared window. That growth is charged to the budget that paid for the window (the child's own
//! node) and every ancestor, all or nothing: past the ceiling the op refuses with `-ENOMEM` and
//! changes no page. A page given back (unmapped) is refunded at once, and whatever the child still
//! holds goes back when it ends, as the window's own bytes do. A page protected to nothing is not given
//! back: it keeps its contents, so it stays charged.
//!
//! The root splits its budget into a node capped at the child's 64 KiB window plus 64 KiB of growth,
//! spawns the child paid from it, joins it, and reads the node's room. The child grows 64 KiB (it
//! fits), asks for 64 KiB more (refused), gives the first 64 KiB back, and grows the second (it fits
//! again). It ends holding 64 KiB of growth, which its end refunds: the room is the whole ceiling again.

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/rec.rs"]
mod rec;

use std::sync::Arc;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, Value};
use temen_ir::{Module, SpawnRec};

/// The node's `mem` ceiling: the child's 64 KiB window and 64 KiB of growth.
const CEILING: i64 = 2 << 16;

/// A root that pays for func 1 from a node capped at [`CEILING`], and returns
/// `child_result * 1_000_000 + room`, the node's `mem` room once the child has ended. The child, given
/// `commit` and `release` as `AddressSpace` op lines over `(at, len)`, returns a bit per expected
/// answer: 1 the first 64 KiB fits, 2 the next 64 KiB is `-ENOMEM`, 4 giving the first back succeeds,
/// 8 the next 64 KiB then fits.
fn src(commit: &str, release: &str) -> String {
    let op = |dst: &str, line: &str, at: u64| {
        line.replace("DST", dst)
            .replace("AT", &format!("a{at}"))
            .replace("LEN", "n")
    };
    format!(
        "memory 16
func (i32, i32) -> (i64) {{
block 0 (vinst: i32, vbud: i32) {{
  f = i64.const -1
  m = i64.const {CEILING}
  vsub = call.cap 14 0 (i64, i64, i64) -> (i64) vbud (f, m, f)
  vnode = i32.wrap_i64 vsub
  rb = i64.const 17436
  i32.store rb vnode
  rp = i64.const 17408
  vch = call.cap 6 17 (i64) -> (i32) vinst (rp)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  one = i64.const 1
  vroom = call.cap 14 1 (i64) -> (i64) vnode (one)
  k = i64.const 1000000
  vhi = i64.mul vr k
  vsum = i64.add vhi vroom
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas64: i64) {{
  vas = i32.wrap_i64 vas64
  a1 = i64.const 65536
  a2 = i64.const 131072
  n = i64.const 65536
  rw = i64.const 3
  z = i64.const 0
  enomem = i64.const -12
{c1}
{c2}
{r3}
{c4}
  b1 = i64.eq r1 z
  b2 = i64.eq r2 enomem
  b3 = i64.eq r3 z
  b4 = i64.eq r4 z
  w1 = i64.extend_i32_u b1
  w2 = i64.extend_i32_u b2
  w3 = i64.extend_i32_u b3
  w4 = i64.extend_i32_u b4
  two = i64.const 2
  four = i64.const 4
  eight = i64.const 8
  x2 = i64.mul w2 two
  x3 = i64.mul w3 four
  x4 = i64.mul w4 eight
  s1 = i64.add w1 x2
  s2 = i64.add s1 x3
  s3 = i64.add s2 x4
  return s3
  }}
}}
{rec}",
        c1 = op("r1", commit, 1),
        c2 = op("r2", commit, 2),
        r3 = op("r3", release, 1),
        c4 = op("r4", commit, 2),
        rec = rec::segment(17408, &SpawnRec::v1(1)),
    )
}

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The root's powerbox: its program registered (a self-spawn), an `Instantiator`, and an unbounded
/// `Budget` to split.
fn setup(m: &Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut h = Host::new();
        h.set_self_module(&Arc::new(m.clone()));
        let i = h.grant_instantiator(0, 1 << 16);
        let b = h.grant_budget(-1, -1, -1);
        (h, vec![Value::I32(i), Value::I32(b)])
    }
}

fn want(child: i64) -> Ran {
    Ran {
        result: Ok(vec![Value::I64(child * 1_000_000 + CEILING)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    }
}

const MAP: &str = "  DST = call.cap 5 0 (i64, i64, i64) -> (i64) vas (AT, LEN, rw)";
const UNMAP: &str = "  DST = call.cap 5 1 (i64, i64) -> (i64) vas (AT, LEN)";
const PROTECT_RW: &str = "  DST = call.cap 5 2 (i64, i64, i64) -> (i64) vas (AT, LEN, rw)";
const PROTECT_NONE: &str = "  DST = call.cap 5 2 (i64, i64, i64) -> (i64) vas (AT, LEN, z)";

#[test]
fn map_past_the_declared_window_spends_the_childs_budget_on_every_driver() {
    let m = module(&src(MAP, UNMAP));
    agree_on_every_driver("map / unmap", &m, &setup(&m), &want(15));
}

/// `protect` to readable or writable commits a tail page as `map` does, so it is metered the same way.
#[test]
fn protect_past_the_declared_window_spends_the_childs_budget_on_every_driver() {
    let m = module(&src(PROTECT_RW, UNMAP));
    agree_on_every_driver("protect", &m, &setup(&m), &want(15));
}

/// A page `protect`ed to nothing keeps its contents (a later `protect` reveals them), so it is still
/// the child's memory and stays charged: the protect succeeds (4) but frees no room, and the next 64 KiB
/// stays refused (no 8). The child's end hands it back with the rest.
#[test]
fn a_page_protected_to_nothing_stays_charged_on_every_driver() {
    let m = module(&src(MAP, PROTECT_NONE));
    agree_on_every_driver("protect none", &m, &setup(&m), &want(7));
}
