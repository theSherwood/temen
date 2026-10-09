//! `self.parallelism` (self-namespace op 19): how many of a domain's vCPUs may run at once. That is
//! the workers its driver has, an input the embedder sets (`Host::set_workers`; by default the real
//! pool's size, `pool_workers`), bounded by its lane cap (D66). The host answers it, so each
//! interpreter driver and the Cranelift JIT agree on a host, and an embedder that runs a one-thread
//! driver says 1 (#2254).

#[path = "../../temen-interp/tests/support/drivers.rs"]
mod drivers;

use drivers::{agree_on_every_driver, Ran};
use temen_interp::{pool_workers, run_with_host, Host, MemLayout, Value};
use temen_ir::{Module, DEFAULT_RESERVED_LOG2};
use temen_jit::{JitError, JitOutcome};

/// The root's window: `memory 16`.
const WIN: usize = 64 << 10;

/// `self.parallelism`, returned.
const ASK: &str = "memory 16
func () -> (i64) {
block 0 () {
  v0 = i32.const 0
  v1 = call.cap 4294967295 19 () -> (i64) v0 ()
  return v1
  }
}
";

/// A detached child that returns its own `self.parallelism`.
const CHILD: &str = "memory 12
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i32.const 0
  v2 = call.cap 4294967295 19 () -> (i64) v1 ()
  return v2
  }
}
";

/// The root (args `(instantiator, CHILD, budget)`): split a budget with a lane of 2 from `budget`,
/// spawn `CHILD` detached on it, and return what the child answers.
const SPAWN: &str = "memory 16
func (i32, i32, i32) -> (i64) {
block 0 (vinst: i32, vmod: i32, vbud: i32) {
  vf = i64.const -1
  vm = i64.const 4096
  vs = i64.const -1
  vc = i64.const -1
  vl = i64.const 2
  vb = call.cap 14 0 (i64, i64, i64, i64, i64) -> (i32) vbud (vf, vm, vs, vc, vl)
  vbw = i64.extend_i32_u vb
  vmh = i64.extend_i32_u vmod
  vz = i64.const 0
  vlog = i64.const 12
  vh = call.cap 6 15 (i64, i64, i64, i64, i64, i64, i64) -> (i32) vinst (vbw, vmh, vz, vz, vz, vlog, vz)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vh)
  return vr
  }
}
";

fn module(src: &str) -> Module {
    let m = temen_text::parse_module(src).unwrap_or_else(|e| panic!("parse: {e:?}"));
    temen_verify::verify_module(&m).unwrap_or_else(|e| panic!("verify: {e:?}"));
    m
}

/// Run `src`'s function 0 on every interpreter driver and on the JIT, and assert each answers `want`.
fn every_engine(what: &str, src: &str, setup: &dyn Fn() -> (Host, Vec<Value>), want: i64) {
    let m = module(src);
    let ran = Ran {
        result: Ok(vec![Value::I64(want)]),
        stdout: Vec::new(),
        stderr: Vec::new(),
    };
    agree_on_every_driver(what, &m, setup, &ran);
    let (mut host, args) = setup();
    let args: Vec<i64> = args
        .iter()
        .map(|v| match v {
            Value::I32(h) => i64::from(*h),
            other => panic!("a handle, not {other:?}"),
        })
        .collect();
    let init = MemLayout::image(vec![0u8; WIN]);
    match temen_run::jit_cap_run(
        &m,
        0,
        &args,
        &init,
        DEFAULT_RESERVED_LOG2,
        0,
        &mut host,
        None,
    ) {
        Ok((o, _)) => assert_eq!(o, JitOutcome::Returned(vec![want]), "{what}: Cranelift"),
        Err(JitError::Unsupported(_)) => {} // a target without the JIT's runtime
        Err(e) => panic!("{what}: the JIT run failed: {e:?}"),
    }
}

/// A host with `workers` set, and lane cap `lane` when it is not negative.
fn host(workers: Option<usize>, lane: i64) -> Host {
    let mut host = Host::new();
    if let Some(n) = workers {
        host.set_workers(n);
    }
    if lane >= 0 {
        host.set_lane_cap(lane);
    }
    host
}

#[test]
fn a_domain_reports_its_drivers_workers() {
    let unset = || (host(None, -1), Vec::new());
    every_engine("workers unset", ASK, &unset, pool_workers() as i64);
    let six = || (host(Some(6), -1), Vec::new());
    every_engine("six workers", ASK, &six, 6);
}

#[test]
fn the_lane_cap_bounds_the_workers() {
    let wide = || (host(Some(8), 3), Vec::new());
    every_engine("lane 3 of 8 workers", ASK, &wide, 3);
    let narrow = || (host(Some(2), 3), Vec::new());
    every_engine("lane 3 of 2 workers", ASK, &narrow, 2);
}

/// A child runs on its parent's driver, so it has its parent's workers, within its own lane.
#[test]
fn a_child_has_its_parents_workers_within_its_own_lane() {
    for (workers, want) in [(8, 2), (1, 1)] {
        let setup = || {
            let mut host = host(Some(workers), -1);
            let inst = host.grant_instantiator(0, WIN as u64);
            let child = host.grant_module(&module(CHILD));
            let budget = host.grant_budget(-1, 1 << 20, -1);
            (host, [inst, child, budget].map(Value::I32).to_vec())
        };
        let what = format!("a child's lane of 2, {workers} workers");
        every_engine(&what, SPAWN, &setup, want);
    }
}

/// The worker count is the machine's and the driver's, so a replay answers what the recording saw,
/// as it does a clock read (DEBUGGING.md W1), whatever the replaying host would say.
#[test]
fn a_replay_answers_what_the_recording_saw() {
    let m = module(ASK);
    let run = |host: &mut Host| run_with_host(&m, 0, &[], &mut 1_000, host);
    let mut recording = host(Some(3), -1);
    recording.record_caps();
    assert_eq!(run(&mut recording), Ok(vec![Value::I64(3)]));
    let mut replaying = host(Some(5), -1);
    replaying.replay_cap_tape(recording.cap_tape());
    assert_eq!(run(&mut replaying), Ok(vec![Value::I64(3)]));
}

/// temen-run sets each backend's workers: the bytecode engine runs every vCPU on one thread, and
/// the tree-walker's pool and the JIT run the host's workers.
#[test]
fn each_temen_run_backend_reports_its_drivers_workers() {
    use temen_run::{Backend, Outcome, RunConfig};
    let workers = pool_workers() as i64;
    for (backend, want) in [
        (Backend::TreeWalk, workers),
        (Backend::Bytecode, 1),
        (Backend::Jit, workers),
    ] {
        let run = temen_run::instantiate(module(ASK))
            .expect("instantiates")
            .run(backend, &RunConfig::default());
        match run {
            Ok(r) => assert_eq!(
                r.outcome,
                Outcome::Returned(vec![Value::I64(want)]),
                "{backend:?}"
            ),
            // A target without the JIT's runtime.
            Err(e) if backend == Backend::Jit && e.contains("nsupported") => {}
            Err(e) => panic!("{backend:?}: {e}"),
        }
    }
}
