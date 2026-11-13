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

/// #3029 — `func 0 (k)`: spawn `func 1` and return 7, joining nothing. `func 1` spawns `func 2` and
/// returns; `func 2` adds `k` at 16384. `func 1` can still be running when the run's end begins.
const SPAWNS_IN_TEARDOWN: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vk: i64) {
  vh = thread.spawn 1 vk vk
  vr = i64.const 7
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vk: i64, vj: i64) {
  vh = thread.spawn 2 vk vk
  vz = i64.const 0
  return vz
  }
}
func (i64, i64) -> (i64) {
block 0 (vk: i64, vj: i64) {
  va = i64.const 16384
  vold = i64.atomic.rmw.add va vk
  vz = i64.const 0
  return vz
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

/// #3029 — a vCPU still running when its run's end begins cannot leave one behind: a spawn into a
/// domain whose vCPUs are being joined is refused. Before, the orphan ran on after its run's window
/// and context were gone, and wrote into the next run's window, mapped again at the same address
/// (in debug, `end_run` caught it: "a vCPU outlived its run"). Each run's window holds its own
/// write, or none: never an earlier run's.
#[test]
fn a_vcpu_spawned_during_teardown_does_not_outlive_its_run() {
    let m = module(SPAWNS_IN_TEARDOWN);
    let mut cm = compile(&m);
    for run in 1..=2000i64 {
        let (out, window) = cm.run(&[run], None, Some(1 << 15)).expect("jit run");
        assert_eq!(out, JitOutcome::Returned(vec![7]), "run {run}");
        let word = i64::from_le_bytes(window[16384..16392].try_into().unwrap());
        assert!(
            word == 0 || word == run,
            "run {run}'s window holds {word}: an earlier run's vCPU wrote it"
        );
    }
}

// ---- #3020 — a run's fibers end with it ----

/// `func 0 (mode, h)`:
/// - mode 0: make a fiber and resume it once (it stores its argument, 10, at 16384 and suspends);
///   answer `(handle, status, value)`.
/// - mode 1: resume `h` with 7, making nothing (were it a fiber, it would store 7 at 16392 and
///   return 107); answer `(h, status, value)`.
/// - mode 2: make a fiber and never resume it; answer `(handle, 0, 0)`.
const FIBERS: &str = r#"memory 16
func (i64, i64) -> (i64, i64, i64) {
block 0 (vm: i64, vh: i64) {
  vz = i64.const 0
  vis0 = i64.eq vm vz
  br_if vis0 1() 2(vm, vh)
}
block 1 () {
  v0 = ref.func 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  v6 = i64.extend_i32_u v4
  return v2 v6 v5
}
block 2 (vm2: i64, vh2: i64) {
  vone = i64.const 1
  vis1 = i64.eq vm2 vone
  br_if vis1 3(vh2) 4()
}
block 3 (vh3: i64) {
  v7 = i64.const 7
  v8, v9 = cont.resume vh3 v7
  v10 = i64.extend_i32_u v8
  return vh3 v10 v9
}
block 4 () {
  w0 = ref.func 1
  w1 = i64.const 4096
  w2 = cont.new w0 w1
  wz = i64.const 0
  return w2 wz wz
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  p = i64.const 16384
  i64.store p b
  x = suspend b
  q = i64.const 16392
  i64.store q x
  c = i64.const 100
  r = i64.add x c
  return r
  }
}
"#;

/// Make a fiber.
const LEAVE_SUSPENDED: i64 = 0;
/// Resume a handle, making nothing.
const RESUME: i64 = 1;
/// Make a fiber and never start it.
const LEAVE_UNSTARTED: i64 = 2;

/// The two words a fiber of [`FIBERS`] writes.
fn fiber_words(window: &[u8]) -> (i64, i64) {
    let word = |o: usize| i64::from_le_bytes(window[o..o + 8].try_into().unwrap());
    (word(16384), word(16392))
}

/// What the oracle answers for a run that resumes a handle it never made: a trap.
fn oracle_resume_traps(m: &temen_ir::Module) -> temen_interp::Trap {
    let mut fuel = u64::MAX;
    let r = temen_interp::run(m, 0, &[Value::I64(RESUME), Value::I64(0)], &mut fuel);
    r.expect_err("the oracle's resume of a handle no fiber has traps")
}

/// Run 1 leaves a fiber (`first`); run 2, on the same compile, resumes its handle. Run 2 answers
/// the oracle's trap, as a fresh compile's does, and its window holds nothing the fiber would write.
fn a_later_run_cannot_resume(first: i64) {
    let m = module(FIBERS);
    let want = oracle_resume_traps(&m);
    assert_eq!(
        temen_jit::TrapKind::FiberFault.code(),
        want.code(),
        "the oracle's trap is FiberFault's"
    );
    let mut fresh = compile(&m);
    let (out, _) = fresh.run(&[RESUME, 0], None, Some(1 << 15)).unwrap();
    assert!(
        matches!(out, JitOutcome::Trapped(temen_jit::TrapKind::FiberFault)),
        "a fresh compile: {out:?}"
    );

    let mut cm = compile(&m);
    let (out, _) = cm.run(&[first, 0], None, Some(1 << 15)).unwrap();
    let JitOutcome::Returned(v) = out else {
        panic!("run 1: {out:?}")
    };
    let (out, window) = cm.run(&[RESUME, v[0]], None, Some(1 << 15)).unwrap();
    assert!(
        matches!(out, JitOutcome::Trapped(temen_jit::TrapKind::FiberFault)),
        "run 2 resumed run 1's fiber {}: {out:?}",
        v[0]
    );
    assert_eq!(fiber_words(&window), (0, 0), "run 1's fiber ran in run 2");
}

#[test]
fn a_later_run_cannot_start_an_earlier_runs_fiber() {
    a_later_run_cannot_resume(LEAVE_UNSTARTED);
}

#[test]
fn a_later_run_cannot_resume_an_earlier_runs_suspended_fiber() {
    a_later_run_cannot_resume(LEAVE_SUSPENDED);
}

/// The compile moved to another thread, so run 2's stack is not run 1's.
struct OnAnotherThread(CompiledModule);
// SAFETY: one thread uses the module at a time: the spawning thread hands it over and waits.
unsafe impl Send for OnAnotherThread {}

/// The SIGSEGV shape: run 2 on another thread, deeper in its stack. Run 1's fiber held run 1's
/// context, a local of a stack run 2 is not on.
#[test]
fn a_later_run_on_another_thread_cannot_resume_an_earlier_runs_fiber() {
    let m = module(FIBERS);
    let mut cm = compile(&m);
    let (out, _) = cm.run(&[LEAVE_SUSPENDED, 0], None, None).unwrap();
    let JitOutcome::Returned(v) = out else {
        panic!("run 1: {out:?}")
    };
    let h = v[0];
    let moved = OnAnotherThread(cm);
    let out = std::thread::spawn(move || {
        let mut moved = moved;
        fn deeper(n: u32, cm: &mut CompiledModule, h: i64) -> JitOutcome {
            let pad = [n as u8; 4096];
            let out = match n {
                0 => cm.run(&[RESUME, h], None, None).unwrap().0,
                _ => deeper(n - 1, cm, h),
            };
            std::hint::black_box(&pad);
            out
        }
        deeper(8, &mut moved.0, h)
    })
    .join()
    .expect("run 2's thread");
    assert!(
        matches!(out, JitOutcome::Trapped(temen_jit::TrapKind::FiberFault)),
        "run 2: {out:?}"
    );
}

/// A budget node that counts the fibers charged to it.
#[derive(Default)]
struct CountsFibers(std::sync::atomic::AtomicI64);

impl temen_jit::BudgetNode for CountsFibers {
    fn draw(&self) -> Option<u64> {
        Some(u64::MAX)
    }
    fn give_back(&self, _unspent: u64) {}
    fn room(&self) -> i64 {
        i64::MAX
    }
    fn charge_vcpu(&self) -> bool {
        true
    }
    fn rebuild_vcpu(&self) -> bool {
        true
    }
    fn vcpu_ended(&self) {}
    fn charge_fiber(&self) -> bool {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        true
    }
    fn force_fiber(&self) {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    fn fiber_ended(&self) {
        self.0.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

/// Runs that each leave a fiber suspended leave their budget node where it started: each run's
/// fibers hand their charge back as it ends.
#[test]
fn a_run_s_fibers_hand_their_charges_back_when_it_ends() {
    let m = module(FIBERS);
    let node = std::sync::Arc::new(CountsFibers::default());
    let mut cm = CompiledModule::compile(
        &m,
        0,
        INERT_CAP_THUNK,
        std::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        Some(node.clone()),
        0,
    )
    .expect("jit compile");
    for run in 1..=5 {
        let (out, _) = cm.run(&[LEAVE_SUSPENDED, 0], None, None).unwrap();
        assert!(matches!(out, JitOutcome::Returned(_)), "run {run}: {out:?}");
        assert_eq!(
            node.0.load(std::sync::atomic::Ordering::SeqCst),
            0,
            "run {run}'s fiber still holds its charge"
        );
    }
}

/// `func 0 (mode, h)`: mode 0 spawns a vCPU that makes a fiber, resumes it once (it suspends) and
/// answers its handle, which the root joins and answers; mode 1 resumes `h` with 7.
const SPAWNED_FIBER: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vm: i64, vh: i64) {
  vz = i64.const 0
  vis0 = i64.eq vm vz
  br_if vis0 1() 2(vh)
}
block 1 () {
  va = i64.const 0
  vt = thread.spawn 1 va va
  vst, vr = thread.join vt
  return vr
}
block 2 (vh2: i64) {
  v7 = i64.const 7
  vs, vr2 = cont.resume vh2 v7
  return vr2
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  vf = ref.func 2
  vsp = i64.const 4096
  vh = cont.new vf vsp
  v10 = i64.const 10
  vs, vr = cont.resume vh v10
  return vh
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  x = suspend b
  c = i64.const 100
  r = i64.add x c
  return r
  }
}
"#;

/// #3020 — a fiber a spawned vCPU made and left is its run's too: a later run cannot resume it.
#[test]
fn a_later_run_cannot_resume_a_fiber_an_earlier_runs_vcpu_made() {
    let m = module(SPAWNED_FIBER);
    let mut cm = compile(&m);
    let (out, _) = cm.run(&[0, 0], None, None).unwrap();
    let JitOutcome::Returned(v) = out else {
        panic!("run 1: {out:?}")
    };
    let (out, _) = cm.run(&[1, v[0]], None, None).unwrap();
    assert_eq!(
        out,
        JitOutcome::Trapped(temen_jit::TrapKind::FiberFault),
        "run 2 resumed run 1's vCPU's fiber {}",
        v[0]
    );
}

/// `func 0 (mode)`: mode 0 makes a fiber and resumes it, and the fiber traps `unreachable`; any
/// other mode answers 7.
const FAULTS_IN_A_FIBER: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vm: i64) {
  vz = i64.const 0
  vis0 = i64.eq vm vz
  br_if vis0 1() 2()
}
block 1 () {
  vf = ref.func 1
  vsp = i64.const 4096
  vh = cont.new vf vsp
  va = i64.const 0
  vs, vr = cont.resume vh va
  return vr
}
block 2 () {
  v7 = i64.const 7
  return v7
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  unreachable
  }
}
"#;

/// #3020 — a run that ends in a trap inside a fiber leaves nothing for the next: the next run
/// answers as a fresh compile's does, and a run that traps the same way traps the same way.
#[test]
fn a_trap_inside_a_fiber_leaves_nothing_for_the_next_run() {
    let m = module(FAULTS_IN_A_FIBER);
    let mut cm = compile(&m);
    let (first, _) = cm.run(&[0], None, None).unwrap();
    assert!(matches!(first, JitOutcome::Trapped(_)), "run 1: {first:?}");
    let (out, _) = cm.run(&[1], None, None).unwrap();
    assert_eq!(out, JitOutcome::Returned(vec![7]), "run 2");
    let (again, _) = cm.run(&[0], None, None).unwrap();
    assert_eq!(again, first, "run 3 traps as run 1 did");
}

/// #3020 — a hosted unit's fiber is its run's too. The parent's code makes no fiber; a unit
/// defined into it under fiber hosting ([`CompiledModule::enable_fiber_hosting`]) does, over the
/// parent's body at table slot 1, and leaves it suspended; the unit's next run cannot resume it.
#[test]
fn a_later_run_cannot_resume_a_hosted_units_fiber() {
    const PARENT: &str = "memory 16
func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  return v0
  }
}
func (i64, i64) -> (i64) {
block 0 (a: i64, b: i64) {
  x = suspend b
  c = i64.const 100
  r = i64.add x c
  return r
  }
}
";
    // `(mode, h)`: mode 0 makes a fiber over the parent's slot 1, resumes it once and answers its
    // handle; mode 1 resumes `h` with 7.
    const UNIT: &str = "memory 16
func (i64, i64) -> (i64) {
block 0 (vm: i64, vh: i64) {
  vz = i64.const 0
  vis0 = i64.eq vm vz
  br_if vis0 1() 2(vh)
}
block 1 () {
  v0 = i32.const 1
  v1 = i64.const 4096
  v2 = cont.new v0 v1
  v3 = i64.const 10
  v4, v5 = cont.resume v2 v3
  return v2
}
block 2 (vh2: i64) {
  v7 = i64.const 7
  v8, v9 = cont.resume vh2 v7
  return v9
  }
}
";
    let parent = module(PARENT);
    let mut cm = compile(&parent);
    if cm.enable_fiber_hosting().is_err() {
        return;
    }
    let unit = temen_text::parse_module(UNIT).expect("parse unit");
    let defined = cm
        .define_extra(&unit.funcs, &unit.types)
        .expect("define the unit");
    let tramp = defined[0].tramp;
    // SAFETY: `tramp` is the unit's finalized buffer-ABI trampoline, `(i64, i64) -> (i64)`.
    let (out, _) = unsafe { cm.run_extra(tramp, 2, 1, &[0, 0], None) }.unwrap();
    let JitOutcome::Returned(v) = out else {
        panic!("run 1: {out:?}")
    };
    // SAFETY: as above.
    let (out, _) = unsafe { cm.run_extra(tramp, 2, 1, &[1, v[0]], None) }.unwrap();
    assert_eq!(
        out,
        JitOutcome::Trapped(temen_jit::TrapKind::FiberFault),
        "run 2 resumed run 1's hosted fiber {}",
        v[0]
    );
}
