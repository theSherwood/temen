//! D66 — **lanes on the interpreter** (INVARIANTS #3 ruling 2026-09-21, #1586, #1600 slice 4).
//!
//! Parallelism is a granted resource, bounded at **dispatch**: a domain's lane cap is how many tasks
//! of its subtree may be running at once, checked when a worker picks a task and released the moment
//! it parks, yields or finishes. A lane is a **ceiling**, not a stock — `split` hands a child a lane
//! bounded by the holder's own cap and the holder keeps its cap; the grantor's Σ-of-granted-lanes ≤
//! its cap is enforced at spawn and a reaped child returns its lane.
//!
//! The default is unbounded, so every run that sets no cap is untouched — the hot path skips both
//! scheduler locks. These tests set one.

use temen_interp::{cap_id, run_with_host, Host, Value};

fn module(text: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(text).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Four `thread.spawn` siblings in the root domain, each: bump a shared **running** counter, note a
/// violation if it was already ≥ 1, spin, drop the counter. Root joins all four and returns the
/// violation count. Above the #1094 NULL guard: RUN at 16392, VIOL at 16400, handles at 16408+.
const FOUR_SPINNERS: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1(vz)
}
block 1 (vi: i64) {
  vfour = i64.const 4
  vz2 = i64.const 0
  vlt = i64.lt_u vi vfour
  br_if vlt 2(vi) 3(vz2)
}
block 2 (vi2: i64) {
  vsp = i64.const 0
  vt = thread.spawn 1 vsp vi2
  vh = i64.const 16408
  v8 = i64.const 4
  vm = i64.mul vi2 v8
  va = i64.add vh vm
  i32.store va vt
  vone = i64.const 1
  vn = i64.add vi2 vone
  br 1(vn)
}
block 3 (vj: i64) {
  vfour2 = i64.const 4
  vlt2 = i64.lt_u vj vfour2
  br_if vlt2 4(vj) 5()
}
block 4 (vj2: i64) {
  vh2 = i64.const 16408
  v82 = i64.const 4
  vm2 = i64.mul vj2 v82
  va2 = i64.add vh2 vm2
  vt2 = i32.load va2
  vr = thread.join vt2
  vone2 = i64.const 1
  vn2 = i64.add vj2 vone2
  br 3(vn2)
}
block 5 () {
  vviol = i64.const 16400
  vv = i64.atomic.load vviol
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vrun = i64.const 16392
  vone = i64.const 1
  vold = i64.atomic.rmw.add vrun vone
  vge = i64.ge_s vold vone
  br_if vge 1() 2()
}
block 1 () {
  vviol = i64.const 16400
  vone1 = i64.const 1
  vx = i64.atomic.rmw.add vviol vone1
  br 2()
}
block 2 () {
  vz = i64.const 0
  br 3(vz)
}
block 3 (vi: i64) {
  vlim = i64.const 20000
  vlt = i64.lt_u vi vlim
  vone3 = i64.const 1
  vi1 = i64.add vi vone3
  br_if vlt 3(vi1) 4()
}
block 4 () {
  vrun2 = i64.const 16392
  vneg = i64.const -1
  vd = i64.atomic.rmw.add vrun2 vneg
  vz4 = i64.const 0
  return vz4
  }
}
"#;

/// **The dispatch-time bound.** With the root's lane cap at 1, four siblings on the real worker pool
/// never overlap: each holds the lane only while on a worker, and a worker admits a task only while
/// every lane in its chain has room. (The counter is a real cross-thread atomic; with no cap and ≥2
/// cores this program reports violations — that is the mutation the pin was checked against.)
#[test]
fn a_lane_cap_of_one_serializes_four_spinning_siblings() {
    let m = module(FOUR_SPINNERS);
    let mut host = Host::new();
    host.set_lane_cap(1);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "no two siblings were ever on a worker at once"
    );
}

/// The default: no cap, nothing changes — the same program completes (its violation count is
/// whatever the hardware allowed, deliberately not asserted).
#[test]
fn an_unbounded_lane_is_the_default_and_runs_untouched() {
    let m = module(FOUR_SPINNERS);
    let mut host = Host::new();
    assert_eq!(host.lane_cap(), -1);
    let mut fuel = u64::MAX;
    assert!(run_with_host(&m, 0, &[], &mut fuel, &mut host).is_ok());
}

/// A detached child (`memory 12`, entry returns 0).
const CHILD: &str = r#"memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  vz = i64.const 0
  return vz
  }
}
"#;

/// `v0` Instantiator, `v1` Module, `v2` Budget. Three sub-budgets split off `v2` — lanes 2, 1, 1 —
/// then: spawn with lane 2 (fits a cap of 2), spawn with lane 1 (Σ 3 > 2: refused, charges nothing),
/// join the first (its lane returns), spawn with lane 1 (fits). Returns
/// `100·(c1 ≥ 0) + 10·(c2 == -EINVAL) + (c3 ≥ 0)`, so the expected answer is `111`.
const SIGMA: &str = r#"memory 16
func (i32, i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32, v2: i32) {
  vf = i64.const -1
  vm = i64.const 4096
  vs = i64.const 0
  vc = i64.const -1
  vl2 = i64.const 2
  vl1 = i64.const 1
  vb1 = call.cap 14 0 (i64, i64, i64, i64, i64) -> (i32) v2 (vf, vm, vs, vc, vl2)
  vb2 = call.cap 14 0 (i64, i64, i64, i64, i64) -> (i32) v2 (vf, vm, vs, vc, vl1)
  vb3 = call.cap 14 0 (i64, i64, i64, i64, i64) -> (i32) v2 (vf, vm, vs, vc, vl1)
  vmh = i64.extend_i32_u v1
  vz = i64.const 0
  ve = i64.const 0
  vlog = i64.const 12
  vq = i64.const 0
  vb1w = i64.extend_i32_u vb1
  vc1 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb1w, vmh, vz, vz, ve, vlog, vq)
  vb2w = i64.extend_i32_u vb2
  vc2 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb2w, vmh, vz, vz, ve, vlog, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vc1)
  vb3w = i64.extend_i32_u vb3
  vc3 = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vb3w, vmh, vz, vz, ve, vlog, vq)
  vzero = i32.const 0
  veinval = i32.const -22
  vok1 = i32.ge_s vc1 vzero
  vref2 = i32.eq vc2 veinval
  vok3 = i32.ge_s vc3 vzero
  v100 = i32.const 100
  v10 = i32.const 10
  va = i32.mul vok1 v100
  vb = i32.mul vref2 v10
  vab = i32.add va vb
  vabc = i32.add vab vok3
  vr = i64.extend_i32_s vabc
  return vr
  }
}
"#;

/// **The grant-time Σ and the reap credit**, on the tree-walk oracle: a parent with cap 2 can hold
/// children whose lanes sum to 2, is refused a third that would make 3, and gets the lane back when a
/// child is joined. The refusal is a value (`-EINVAL`) and charges nothing — the refused sub-budget's
/// `mem` is intact, so the third spawn can use a fresh one.
#[test]
fn a_parents_granted_lanes_may_not_exceed_its_cap_and_a_reaped_child_returns_its_lane() {
    let parent = module(SIGMA);
    let child = module(CHILD);
    let mut host = Host::new();
    host.set_lane_cap(2);
    let inst = host.grant_instantiator(0, 1 << 16);
    let modh = host.grant_module(&child);
    let budget = host.grant_budget(-1, 3 << 12, 0);
    let args = vec![Value::I32(inst), Value::I32(modh), Value::I32(budget)];
    let mut fuel = u64::MAX;
    let r = run_with_host(&parent, 0, &args, &mut fuel, &mut host);
    assert_eq!(r, Ok(vec![Value::I64(111)]));
    assert_eq!(
        host.granted_lanes(),
        1,
        "after the run: the joined child's 2 came back, the third child's 1 is still out (unjoined)"
    );
}

/// **`split` treats the lane as a ceiling.** Omitted ⇒ inherit the holder's cap; a request within the
/// cap is granted as asked; a request above it refuses the whole split; and the holder's *own* cap
/// is untouched by any of them — no draw-down, unlike the four stocks.
#[test]
fn split_bounds_a_childs_lane_by_the_holders_cap_and_never_draws_the_cap_down() {
    let mut host = Host::new();
    host.set_lane_cap(2);
    let b = host.grant_budget(-1, 1 << 20, 0);
    let split = |host: &mut Host, lane: i64| -> i64 {
        host.cap_dispatch_slots(cap_id::BUDGET, 0, b, &[-1, 4096, 0, -1, lane], None)
            .expect("split dispatches")[0]
    };
    let read_lane = |host: &mut Host, h: i32| -> i64 {
        host.cap_dispatch_slots(cap_id::BUDGET, 1, h, &[4], None)
            .expect("read dispatches")[0]
    };
    let inherit = split(&mut host, -1) as i32;
    assert_eq!(
        read_lane(&mut host, inherit),
        2,
        "omitted ⇒ the holder's cap"
    );
    let one = split(&mut host, 1) as i32;
    assert_eq!(read_lane(&mut host, one), 1);
    assert_eq!(
        split(&mut host, 3),
        -22,
        "a lane wider than the holder's cap refuses the split"
    );
    assert_eq!(
        host.lane_cap(),
        2,
        "the holder's cap is a ceiling: nothing was drawn down"
    );
    assert_eq!(
        read_lane(&mut host, b),
        -1,
        "the budget's own lane field is what *its* child gets"
    );
}

/// #1587 — the undo of a `budget_mem_take` whose spawn then failed after the commit: exact, and inert
/// on an unbounded budget.
#[test]
fn budget_mem_give_is_the_exact_undo_of_a_take() {
    let mut host = Host::new();
    let b = host.grant_budget(0, 8192, 0);
    let read_mem = |host: &mut Host| -> i64 {
        host.cap_dispatch_slots(cap_id::BUDGET, 1, b, &[1], None)
            .unwrap()[0]
    };
    assert!(host.budget_mem_take(b, 4096));
    assert_eq!(read_mem(&mut host), 4096);
    host.budget_mem_give(b, 4096);
    assert_eq!(read_mem(&mut host), 8192);
    let u = host.grant_budget(0, -1, 0);
    host.budget_mem_give(u, 4096);
    assert_eq!(
        host.cap_dispatch_slots(cap_id::BUDGET, 1, u, &[1], None)
            .unwrap()[0],
        -1,
        "an unbounded field stays unbounded"
    );
}
