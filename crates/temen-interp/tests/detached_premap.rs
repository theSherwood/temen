//! A spawn record's **pre-mapped region** — the `region` and `child_off` fields of a v1
//! `instantiate_rec` record (op 17): a `SharedRegion` of the parent aliased whole, read-write, into
//! the detached child's private window at `child_off` before it starts. The child sees plain memory
//! at a fixed offset — no `self.resolve`, no `page_size` query, no `map` call — and the parent
//! reads the child's output back through its own mapping of the same region after `join`. This is
//! the bulk data plane between a parent and a child whose memory it cannot otherwise address
//! (DESIGN.md §13 / DETACHED_JIT.md §3.4), minus the ceremony.
//!
//! Pinned on every driver (`support/drivers.rs`), each giving the tree-walk oracle's answer: the
//! alias round-trip, the probeable `-EINVAL` refusals (an offset inside the NULL guard, a region
//! overrunning the declared window), and the `CapFault` a wrong-kind handle raises (invariant 5:
//! forgery traps, geometry refuses).

#[path = "support/drivers.rs"]
mod drivers;
#[path = "support/rec.rs"]
mod rec;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{Host, Trap, Value};
use temen_ir::SpawnRec;

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The child (`memory 17`, child entry): reads the parent's input word at the pre-mapped offset
/// (region byte 0), writes `2 × input` at region byte 8, returns `input + 1`. It holds no region
/// handle and calls no cap — the pre-mapped pages are just its memory.
const CHILD: &str = r#"memory 17
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

/// The offset the child's window aliases the region at: region-granularity aligned on every host
/// (64 KiB — Windows' allocation granularity), above the NULL guard, and `+ 64 KiB` exactly fills the
/// child's 128 KiB window.
const CHILD_OFF: u64 = 65536;

/// The parent: `v0` Instantiator, `v1` AddressSpace, `v2` the child `Module`, `v3` the
/// detached-spawn `Budget`. Mints a 64 KiB region, maps it at 65536 in its own window, stores the
/// input word 41 at region byte 0, spawns the child detached from its spawn record at 17408 with the
/// region pre-mapped at `off` (`reg` names the handle the parent stores into the record — the
/// region, or a wrong-kind handle for the forgery probe). `join` then also joins the child and
/// returns `1000 × result + the word the child wrote at region byte 8`; otherwise it returns the
/// spawn's own result (the refusal probes).
fn parent(off: u64, reg: &str, join: bool) -> String {
    let child = SpawnRec {
        size_log2: 17,
        child_off: off,
        ..SpawnRec::default()
    };
    let tail = if join {
        "vj_st, vj = call.cap 6 1 (i32) -> (i64, i64) v0 (vc)\n  vk = i64.const 1000\n  vm = i64.mul vj vk\n  vob = i64.const 65544\n  vo = i64.load vob\n  vr = i64.add vm vo\n  return vr"
    } else {
        "vr = i64.extend_i32_s vc\n  return vr"
    };
    format!(
        r#"memory 17
func (i32, i32, i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32, v2: i32, v3: i32) {{
  vlen = i64.const 65536
  vrh64 = call.cap 5 5 (i64) -> (i64) v1 (vlen)
  vrh = i32.wrap_i64 vrh64
  vwo = i64.const 65536
  vro = i64.const 0
  vprot = i32.const 3
  vm0 = call.cap 4 0 (i64, i64, i64, i32) -> (i64) vrh (vwo, vro, vlen, vprot)
  vin = i64.const 41
  i64.store vwo vin
  vma = i64.const 17416
  i32.store vma v2
  vba = i64.const 17420
  i32.store vba v3
  vra = i64.const 17456
  i32.store vra {reg}
  vrp = i64.const 17408
  vc = call.cap 6 17 (i64) -> (i32) v0 (vrp)
  {tail}
  }}
}}
{rec}"#,
        rec = rec::segment(17408, &child),
    )
}

/// `1000 × (41 + 1) + 2 × 41`.
const ROUND_TRIP: i64 = 42_082;

/// The parent's powerbox: its `Instantiator`, an `AddressSpace`, `child` as a `Module`, and a
/// detached-spawn `Budget` that holds exactly one 2^17 window.
fn setup(child: &temen_ir::Module) -> impl Fn() -> (Host, Vec<Value>) + '_ {
    move || {
        let mut host = Host::new();
        let inst = host.grant_instantiator();
        let aspace = host.grant_address_space(0, 1u64 << 17);
        let modh = host.grant_module(child);
        let budget = host.grant_budget(-1, 1i64 << 17, -1);
        let args = [inst, aspace, modh, budget].map(Value::I32).to_vec();
        (host, args)
    }
}

/// `parent_src` gives `want` on every driver.
fn agree(what: &str, parent_src: &str, want: Result<Vec<Value>, Trap>) {
    let child = module(CHILD);
    let want = Ran {
        result: want,
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(what, &module(parent_src), &setup(&child), &want);
}

#[test]
fn a_pre_mapped_region_carries_bulk_data_both_ways() {
    // The child read the parent's word through the alias and the parent read the child's back.
    agree(
        "the round trip",
        &parent(CHILD_OFF, "vrh", true),
        Ok(vec![Value::I64(ROUND_TRIP)]),
    );
}

#[test]
fn an_offset_inside_the_null_guard_refuses_probeably() {
    agree(
        "an offset inside the NULL guard",
        &parent(0, "vrh", false),
        Ok(vec![Value::I64(-22)]),
    );
}

#[test]
fn a_region_overrunning_the_child_window_refuses_probeably() {
    // 128 KiB + 64 KiB > the child's 128 KiB declared window.
    agree(
        "a region overrunning the window",
        &parent(1 << 17, "vrh", false),
        Ok(vec![Value::I64(-22)]),
    );
}

#[test]
fn a_wrong_kind_handle_traps() {
    // The AddressSpace handle where a SharedRegion is expected: a typing violation on a live handle.
    agree(
        "a wrong-kind handle",
        &parent(CHILD_OFF, "v1", false),
        Err(Trap::CapFault),
    );
}
