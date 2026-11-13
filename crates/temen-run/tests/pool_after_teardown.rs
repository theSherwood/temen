//! #2993 (review A) — a run's pool is joined before the run returns, even for a task filed into it
//! after its first join at teardown. One test in this binary: it counts the process's pool workers.
#![cfg(target_os = "linux")]

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::Host;
use temen_ir::{Module, SpawnRec};
use temen_jit::JitOutcome;

const PARENT_LOG2: u8 = 23;
const CHILD_LOG2: u8 = 16;

fn parse(src: &str) -> Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The process's pool workers alive now (`ChildExec`'s threads are named `temen-child`).
fn pool_workers() -> usize {
    std::fs::read_dir("/proc/self/task")
        .expect("/proc/self/task")
        .filter_map(|e| std::fs::read_to_string(e.ok()?.path().join("comm")).ok())
        .filter(|c| c.trim() == "temen-child")
        .count()
}

/// #2993 (review A) — the root's spawned thread spawns and joins long-running §14 children in a loop
/// while the root returns: a spawn of the thread's may land after the root's teardown has joined the
/// run's pool, and must not get a worker that outlives the run. Its pool is joined again once the
/// last vCPU is (`Nursery::join_late`); before, a stray worker read the freed `Domain` (a host
/// SIGSEGV in 7 of 10 runs on main, ASan: heap-use-after-free in `Domain::hub`). Here, at each run's
/// return, no worker of a pool is left: the binary runs this one test, so every `temen-child` thread
/// would be this run's.
#[test]
fn a_child_a_thread_spawns_as_its_root_returns_ends_with_the_run() {
    let child = SpawnRec {
        size_log2: CHILD_LOG2.into(),
        ..SpawnRec::default()
    };
    let parent = parse(&format!(
        "memory {PARENT_LOG2}
func (i32, i32, i32) -> (i64) {{
block 0 (vinst: i32, vmod: i32, vbud: i32) {{
  vma = i64.const {module_at}
  i32.store vma vmod
  vba = i64.const {budget_at}
  i32.store vba vbud
  vsi = i64.const 16444
  i32.store vsi vinst
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  br 1(vz)
}}
block 1 (wn: i64) {{
  vlim = i64.const 20000000
  vgo = i64.lt_u wn vlim
  br_if vgo 2(wn) 3()
}}
block 2 (xn: i64) {{
  vone = i64.const 1
  xm = i64.add xn vone
  br 1(xm)
}}
block 3 () {{
  vr = i64.const 1
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vn = i64.const 0
  br 1(vn)
}}
block 1 (wn: i64) {{
  vlim = i64.const 100
  vgo = i64.lt_u wn vlim
  br_if vgo 2(wn) 3(wn)
}}
block 2 (xn: i64) {{
  vsi = i64.const 16444
  vinst = i32.load vsi
  vrp = i64.const 17408
  ch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vr_st, vr = call.cap 6 1 (i32) -> (i64, i64) vinst (ch)
  vone = i64.const 1
  xm = i64.add xn vone
  br 1(xm)
}}
block 3 (yn: i64) {{
  return yn
  }}
}}
{rec}",
        module_at = 17408 + SpawnRec::MODULE_AT,
        budget_at = 17408 + SpawnRec::BUDGET_AT,
        rec = rec::segment(17408, &child),
    ));
    let spinner = parse(
        "memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vn = i64.const 0
  br 1(vn)
}
block 1 (wn: i64) {
  vlim = i64.const 300000000
  vgo = i64.lt_u wn vlim
  br_if vgo 2(wn) 3(wn)
}
block 2 (xn: i64) {
  vone = i64.const 1
  xm = i64.add xn vone
  br 1(xm)
}
block 3 (yn: i64) {
  return yn
  }
}
",
    );
    for _ in 0..3 {
        let mut host = Host::new();
        host.set_lane_cap(4);
        let (inst, module, budget) = (
            host.grant_instantiator(),
            host.grant_module(&spinner),
            host.grant_budget(-1, -1, -1),
        );
        let args = [inst as i64, module as i64, budget as i64];
        let (jo, _) = temen_run::jit_cap_run(
            &parent,
            0,
            &args,
            &temen_interp::MemLayout::image(Vec::new()),
            PARENT_LOG2,
            4,
            &mut host,
            None,
        )
        .expect("the JIT compiles the parent");
        assert!(
            matches!(jo, JitOutcome::Returned(ref v) if v == &[1]),
            "{jo:?}"
        );
        assert_eq!(
            pool_workers(),
            0,
            "every worker of the run's pool is joined before it returns"
        );
    }
}
