//! PROCESS.md §5 / §15 — `Budget` (iface 14): a node of a run's **tree of ceilings** (#1944,
//! INVARIANTS #3 ruling 2026-09-30) over `(fuel, mem, spawn, channel, lane)`. A domain can `split` a
//! child node (its ceilings clamped to the holder's, nothing deducted) and `read` the room left along
//! the node's chain. `transfer`, R2's top-up, retired with the tree.
//!
//! `Budget` is an ordinary capability (it dispatches through the generic `call.cap` path, not the
//! eval-loop-serviced `Instantiator`), so the interpreter and the JIT service it through the **same**
//! `Host::cap_dispatch_slots` — these tests run each program on both backends and assert identical
//! results (parity for free, like `Stream`/`Clock`). Charging a node's chain is pinned where windows
//! are minted (`instantiate_record.rs`, `temen-interp`'s `budget_tree.rs`).

use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Run `src`'s func 0 on the interpreter with `bh` (a Budget handle) as its single `i32` arg.
fn run_interp(src: &str, host: &mut Host, bh: i32) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(bh)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        host,
    )
    .0
}

/// Same program + Budget handle on the JIT (`bh` widened into the entry's `i64` arg slot).
fn run_jit(src: &str, host: &mut Host, bh: i32) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[bh as i64],
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// `split(300, 200, 3)` out of a `(1000, 500, 10)` budget, then read the parent's fuel and the child's
/// whole vector; encode all four as `((parent_fuel*1000 + child_fuel)*1000 + child_mem)*1000 +
/// child_spawn` = `((1000*1000 + 300)*1000 + 200)*1000 + 3` = `1000300200003` — the split deducted
/// nothing from the parent.
const SPLIT_AND_READ: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  f300 = i64.const 300\n\
  m200 = i64.const 200\n\
  s3 = i64.const 3\n\
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vb (f300, m200, s3)\n\
  fld0 = i64.const 0\n\
  fld1 = i64.const 1\n\
  fld2 = i64.const 2\n\
  vpf = call.cap 14 1 (i64) -> (i64) vb (fld0)\n\
  vcf = call.cap 14 1 (i64) -> (i64) vsub (fld0)\n\
  vcm = call.cap 14 1 (i64) -> (i64) vsub (fld1)\n\
  vcs = call.cap 14 1 (i64) -> (i64) vsub (fld2)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vpf k1000\n\
  t1 = i64.add t0 vcf\n\
  t2 = i64.mul t1 k1000\n\
  t3 = i64.add t2 vcm\n\
  t4 = i64.mul t3 k1000\n\
  t5 = i64.add t4 vcs\n\
  return t5\n\
  }\n\
}\n";

/// `split(2000, 0, 0)` out of a `(1000, …)` budget asks past the holder's fuel ceiling, so the
/// child's is clamped to it: the child reads `1000`.
const OVER_SPLIT: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  big = i64.const 2000\n\
  z = i64.const 0\n\
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vb (big, z, z)\n\
  vr = call.cap 14 1 (i64) -> (i64) vsub (z)\n\
  return vr\n\
  }\n\
}\n";

/// `split(-1, -1, -1)` inherits every ceiling; the parent keeps its own. Encode `((child_fuel*1000 +
/// parent_fuel)*1000 + parent_spawn)` = `((1000*1000 + 1000)*1000 + 10)` = `1001000010`.
const SPLIT_ALL: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  all = i64.const -1\n\
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vb (all, all, all)\n\
  fld0 = i64.const 0\n\
  fld2 = i64.const 2\n\
  vcf = call.cap 14 1 (i64) -> (i64) vsub (fld0)\n\
  vpf = call.cap 14 1 (i64) -> (i64) vb (fld0)\n\
  vps = call.cap 14 1 (i64) -> (i64) vb (fld2)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vcf k1000\n\
  t1 = i64.add t0 vpf\n\
  t2 = i64.mul t1 k1000\n\
  t3 = i64.add t2 vps\n\
  return t3\n\
  }\n\
}\n";

fn both(
    src: &str,
    budget: (i64, i64, i64),
) -> (Result<Vec<Value>, temen_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    let ibh = ih.grant_budget(budget.0, budget.1, budget.2);
    let ir = run_interp(src, &mut ih, ibh);
    let mut jh = Host::new();
    let jbh = jh.grant_budget(budget.0, budget.1, budget.2);
    let jo = run_jit(src, &mut jh, jbh);
    (ir, jo)
}

#[test]
fn split_and_read_matches_across_backends() {
    let (ir, jo) = both(SPLIT_AND_READ, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(1_000_300_200_003)]),
        "interp: the parent keeps its 1000 fuel ceiling; the child's is (300, 200, 3)"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1_000_300_200_003]),
        "jit: must match interp, got {jo:?}"
    );
}

#[test]
fn an_over_split_is_clamped_to_the_holder_on_both() {
    let (ir, jo) = both(OVER_SPLIT, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(1000)]),
        "interp: the child's fuel ceiling is clamped to the parent's"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1000]),
        "jit: must match interp, got {jo:?}"
    );
}

#[test]
fn split_all_inherits_and_the_parent_keeps_its_ceilings_on_both() {
    let (ir, jo) = both(SPLIT_ALL, (1000, 500, 10));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(1_001_000_010)]),
        "interp: the child inherits 1000 fuel; the parent still holds 1000 fuel and 10 spawn"
    );
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[1_001_000_010]),
        "jit: must match interp, got {jo:?}"
    );
}

/// Op 2 (`transfer`, #1289 R2's top-up) retired with the tree (#1944): it answers `-EINVAL`.
const TRANSFER: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  z = i64.const 0\n\
  m200 = i64.const 200\n\
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vb (z, m200, z)\n\
  m100 = i64.const 100\n\
  vt = call.cap 14 2 (i32, i64, i64, i64) -> (i32) vb (vsub, z, m100, z)\n\
  vr = i64.extend_i32_s vt\n\
  return vr\n\
  }\n\
}\n";

#[test]
fn transfer_is_retired_on_both() {
    let (ir, jo) = both(TRANSFER, (1000, 500, 10));
    assert_eq!(ir, Ok(vec![Value::I64(-22)]), "interp: transfer -> -EINVAL");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[-22]),
        "jit: must match interp, got {jo:?}"
    );
}

// ---- #989: the channel (4th) dimension ---------------------------------------------------------

/// `split(300, 200, 3, 500)` out of a `(1000, 500, 10, channel=800)` budget, then read the parent's
/// and child's `channel` (field 3). Encode `child_channel*1000 + parent_channel` = `500*1000 + 800`
/// = `500800` (the child's ceiling 500, the parent's still 800).
const SPLIT_CHANNEL: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  f300 = i64.const 300\n\
  m200 = i64.const 200\n\
  s3 = i64.const 3\n\
  c500 = i64.const 500\n\
  vsub = call.cap 14 0 (i64, i64, i64, i64) -> (i32) vb (f300, m200, s3, c500)\n\
  fld3 = i64.const 3\n\
  vcc = call.cap 14 1 (i64) -> (i64) vsub (fld3)\n\
  vpc = call.cap 14 1 (i64) -> (i64) vb (fld3)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vcc k1000\n\
  t1 = i64.add t0 vpc\n\
  return t1\n\
  }\n\
}\n";

/// A 3-arg `split(300, 200, 3)` (no channel arg) of a `channel=800` budget: the omitted channel means
/// "inherit", so the child's ceiling is 800 too — which keeps a §14 child's channel UNBOUNDED when the
/// parent's was. Encode `child_channel*1000 + parent_channel` = `800*1000 + 800` = `800800`.
const SPLIT_CHANNEL_LEGACY_3ARG: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (vb: i32) {\n\
  f300 = i64.const 300\n\
  m200 = i64.const 200\n\
  s3 = i64.const 3\n\
  vsub = call.cap 14 0 (i64, i64, i64) -> (i32) vb (f300, m200, s3)\n\
  fld3 = i64.const 3\n\
  vcc = call.cap 14 1 (i64) -> (i64) vsub (fld3)\n\
  vpc = call.cap 14 1 (i64) -> (i64) vb (fld3)\n\
  k1000 = i64.const 1000\n\
  t0 = i64.mul vcc k1000\n\
  t1 = i64.add t0 vpc\n\
  return t1\n\
  }\n\
}\n";

/// [`both`] with an explicit `channel` (4th) dimension on the granted budget.
fn both_channel(
    src: &str,
    budget: (i64, i64, i64, i64),
) -> (Result<Vec<Value>, temen_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    let ibh = ih.grant_budget_channel(budget.0, budget.1, budget.2, budget.3);
    let ir = run_interp(src, &mut ih, ibh);
    let mut jh = Host::new();
    let jbh = jh.grant_budget_channel(budget.0, budget.1, budget.2, budget.3);
    let jo = run_jit(src, &mut jh, jbh);
    (ir, jo)
}

#[test]
fn channel_split_and_read_matches_across_backends() {
    let (ir, jo) = both_channel(SPLIT_CHANNEL, (1000, 500, 10, 800));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(500_800)]),
        "interp: channel split(500) → child 500, parent still 800"
    );
    assert!(
        matches!(&jo, JitOutcome::Returned(s) if s == &[500_800]),
        "jit ≡ interp on the channel dimension: {jo:?}"
    );
}

#[test]
fn channel_omitted_arg_inherits_the_ceiling() {
    let (ir, jo) = both_channel(SPLIT_CHANNEL_LEGACY_3ARG, (1000, 500, 10, 800));
    assert_eq!(
        ir,
        Ok(vec![Value::I64(800_800)]),
        "interp: a 3-arg split inherits the channel ceiling (child 800, parent 800)"
    );
    assert!(
        matches!(&jo, JitOutcome::Returned(s) if s == &[800_800]),
        "jit ≡ interp: {jo:?}"
    );
}
