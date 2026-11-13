//! #2953 — a reused `CompiledModule`'s run answers as a fresh compile's first run does, and as the
//! oracle does. Before, the instance's thread domain carried its run's state into the next one: the
//! child handle table (handles a run never joined filled its 1,024 slots, and `thread.spawn` then
//! trapped `ThreadFault`), the slot generations and the vCPU id seed (a joined-only program answered
//! different handles and `vcpu.tls` ids on each later run).
//!
//! Each program is compiled once and run many times; every run's answer must equal the tree-walker's
//! single run. The run counts are fixed past the run at which the old code trapped.

use temen_interp::Value;
use temen_jit::{CompiledModule, JitOutcome, INERT_CAP_THUNK};

/// `func 0 (k)`: spawn `k` threads of `func 1`, which end at once, join none, and return `k`.
const UNJOINED: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vk: i64) {
  vz = i64.const 0
  br 1(vk, vz)
}
block 1 (vk1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vk1, vi1) 3(vk1)
}
block 2 (vk2: i64, vi2: i64) {
  vh2 = thread.spawn 1 vi2 vi2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vk2, vin2)
}
block 3 (vk3: i64) {
  return vk3
  }
}
func (i64, i64) -> (i64) {
block 0 (vi: i64, vj: i64) {
  return vi
  }
}
"#;

/// `func 0 (k)`: spawn `k` threads of `func 1`, each of which counts itself in at 16384 and then
/// parks for good on 16392, which nothing notifies; wait until all `k` have counted in, and return
/// the count, joining none. The run's end ends them (#2868's idlers).
const PARKED: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vk: i64) {
  vz = i64.const 0
  br 1(vk, vz)
}
block 1 (vk1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vk1, vi1) 3(vk1)
}
block 2 (vk2: i64, vi2: i64) {
  vh2 = thread.spawn 1 vi2 vi2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vk2, vin2)
}
block 3 (vk3: i64) {
  va3 = i64.const 16384
  vc3 = i64.atomic.load va3
  vlt3 = i64.lt_u vc3 vk3
  br_if vlt3 3(vk3) 4(vc3)
}
block 4 (vc4: i64) {
  return vc4
  }
}
func (i64, i64) -> (i64) {
block 0 (vi: i64, vj: i64) {
  va = i64.const 16384
  vone = i64.const 1
  vold = i64.atomic.rmw.add va vone
  vw = i64.const 16392
  vz = i32.const 0
  vinf = i64.const -1
  vs = i32.atomic.wait vw vz vinf
  vr = i64.const 0
  return vr
  }
}
"#;

/// Spawn one thread and join it: `(its handle, its vcpu.tls id)`. Leaks nothing, yet a run that
/// inherits the last one's slot generations and id seed answers differently.
const JOINED: &str = r#"memory 16
func () -> (i32, i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vst, vr = thread.join vh
  return vh vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vi: i64, vj: i64) {
  vt = vcpu.tls.get
  return vt
  }
}
"#;

fn module(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn compile(m: &temen_ir::Module) -> CompiledModule {
    CompiledModule::compile(
        m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("jit compile")
}

/// The oracle's single run of `func 0`.
fn oracle(m: &temen_ir::Module, args: &[i64]) -> Vec<i64> {
    let args: Vec<Value> = args.iter().map(|&a| Value::I64(a)).collect();
    let mut fuel = u64::MAX;
    let out = temen_interp::run(m, 0, &args, &mut fuel).expect("oracle run");
    out.into_iter()
        .map(|v| match v {
            Value::I64(x) => x,
            Value::I32(x) => x as i64,
            other => panic!("unexpected oracle value {other:?}"),
        })
        .collect()
}

/// Compile `src` once and run it `runs` times on `args`: every run answers the oracle's answer.
fn every_run_answers_the_oracle(src: &str, args: &[i64], runs: usize) {
    let m = module(src);
    let want = oracle(&m, args);
    let mut cm = compile(&m);
    for run in 1..=runs {
        let (out, _) = cm.run(args, None, None).expect("jit run");
        match out {
            JitOutcome::Returned(v) => assert_eq!(
                v, want,
                "run {run} of one compile at {args:?} answered {v:?}; the oracle answers {want:?}"
            ),
            other => panic!("run {run} of one compile at {args:?} ended {other:?}; want {want:?}"),
        }
    }
}

/// `bench --threads`' shape: 64 workers a run never joins. The table used to fill at run 17.
#[test]
fn many_unjoined_workers_on_one_compile() {
    every_run_answers_the_oracle(UNJOINED, &[64], 40);
}

/// 100 and 300 unjoined workers a run: the table used to fill at runs 11 and 4.
#[test]
fn hundreds_of_unjoined_workers_on_one_compile() {
    every_run_answers_the_oracle(UNJOINED, &[100], 15);
    every_run_answers_the_oracle(UNJOINED, &[300], 8);
}

/// #2868's idlers: K threads parked for good beside a run that returns. The run's end wakes and
/// ends them; their handles retire with it. K = 100 used to trap at run 11, K = 300 at run 4.
#[test]
fn k_parked_threads_on_one_compile() {
    every_run_answers_the_oracle(PARKED, &[100], 15);
    every_run_answers_the_oracle(PARKED, &[300], 8);
}

/// A joined-only program: the same handle and the same vCPU id on every run (it used to answer
/// `[1024, 1]`, `[2048, 2]`, `[3072, 3]`, …).
#[test]
fn a_joined_thread_has_the_same_handle_and_id_on_every_run() {
    every_run_answers_the_oracle(JOINED, &[], 5);
}
