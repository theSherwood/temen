//! End-to-end and differential tests for `temen-runtime` (`Engine::Bytecode` on Cooperative &
//! Parallel Carriers) against `temen-interp` (`#3187` / `#3196`).

use std::sync::Arc;
use temen_interp::Value;
use temen_ir::Module;
use temen_kernel::{CapOutcome, ContextExit, ContextId, DomainId, Limits, Trap};
use temen_mem::Region;
use temen_runtime::{run_module, CapHandler, RunConfig, RunMode};
use temen_text::parse_module;

fn parse(src: &str) -> Module {
    parse_module(src).unwrap_or_else(|e| panic!("parse failed: {e:?}\n{src}"))
}

fn value_as_i64(v: &Value) -> i64 {
    match *v {
        Value::I32(x) => i64::from(x as u32),
        Value::I64(x) => x,
        Value::F32(f) => i64::from(f.to_bits()),
        Value::F64(f) => f.to_bits() as i64,
        Value::Ref(x) => x as i64,
        Value::V128(_) => panic!("unexpected v128 in scalar helper"),
    }
}

fn diff_against_interp(src: &str, args: &[i64], want_fuel_match: bool) {
    let m = parse(src);
    let interp_args: Vec<Value> = m.funcs[0]
        .params
        .iter()
        .zip(args.iter())
        .map(|(ty, &a)| match ty {
            temen_ir::ValType::I32 | temen_ir::ValType::Ref | temen_ir::ValType::Cap => {
                Value::I32(a as i32)
            }
            temen_ir::ValType::I64 => Value::I64(a),
            temen_ir::ValType::F32 => Value::F32(f32::from_bits(a as u32)),
            temen_ir::ValType::F64 => Value::F64(f64::from_bits(a as u64)),
            temen_ir::ValType::V128 => Value::V128([0; 16]),
        })
        .collect();

    let mut interp_fuel = 100_000u64;
    let oracle = temen_interp::run(&m, 0, &interp_args, &mut interp_fuel);
    let oracle_used = 100_000 - interp_fuel;

    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let cfg = RunConfig {
            mode,
            limits: Limits {
                fuel: 100_000,
                ..Limits::default()
            },
            ..RunConfig::default()
        };
        let out = run_module(&m, 0, args, cfg).expect("compile & run");
        match (&oracle, &out.exit) {
            (Ok(ov), ContextExit::Returned(rv)) => {
                let o_i64: Vec<i64> = ov.iter().map(value_as_i64).collect();
                assert_eq!(&o_i64, rv, "result mismatch in {mode:?}");
                if want_fuel_match {
                    assert_eq!(
                        oracle_used, out.fuel_used,
                        "exact IR safepoint fuel mismatch in {mode:?}"
                    );
                }
            }
            (Err(ot), ContextExit::Trapped(rt)) => {
                assert_eq!(ot.name(), rt.name(), "trap mismatch in {mode:?}");
            }
            other => panic!("oracle vs runtime diverged in {mode:?}: {other:?}"),
        }
    }
}

#[test]
fn diff_compute_loops_memory_bulk_and_atomics() {
    // Fibonacci loop with memory scratch + bulk copy + atomic RMW.
    let src = r#"
memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vz = i64.const 0
  vo = i64.const 1
  br 1 (vz, vo, vz)
}
block 1 (va: i64, vb: i64, vi: i64, vn = 0.vn) {
  vlt = i64.lt_s vi vn
  br_if vlt 2 () 3 ()
}
block 2 in 1 (va = 1.va, vb = 1.vb, vi = 1.vi) {
  vnext = i64.add va vb
  vone = i64.const 1
  vi1 = i64.add vi vone
  br 1 (vb, vnext, vi1)
}
block 3 in 1 (va = 1.va) {
  vp0 = i64.const 16448
  i64.store vp0 va
  vp1 = i64.const 16512
  vlen = i64.const 8
  mem.copy vp1 vp0 vlen
  vinc = i64.const 10
  vold = i64.atomic.rmw.add vp1 vinc
  vnew = i64.atomic.load vp1
  vsum = i64.add vold vnew
  return vsum
}
}
"#;
    // fib(10) = 55; vold = 55, vnew = 65 => 120.
    diff_against_interp(src, &[10], true);
}

#[test]
fn diff_null_guard_readonly_data_and_oob_trap_memory_fault() {
    // Deref inside [0, POWERBOX_NULL_GUARD) traps MemoryFault (Invariant 13).
    let null_src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vp = i64.const 64
  vx = i64.load vp
  return vx
}
}
"#;
    diff_against_interp(null_src, &[], false);

    // Store to read-only data segment above NULL guard traps MemoryFault.
    let ro_src = r#"
memory 16
data ro 16384 "hello"
func () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vv = i32.const 42
  i32.store8 vp vv
  vz = i64.const 0
  return vz
}
}
"#;
    diff_against_interp(ro_src, &[], false);

    // Access past window top traps MemoryFault.
    let oob_src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vp = i64.const 65536
  vx = i64.load vp
  return vx
}
}
"#;
    diff_against_interp(oob_src, &[], false);
}

#[test]
fn diff_guest_fibers_generator_and_spent_handle_ebadf() {
    // Fiber yields (arg0 + arg1), then yields (resumed_val * 2), then returns (resumed_val + 5).
    // Root resumes 3 times, then resumes spent handle -> (-9, 0), and tests Invariant 11 tagged handle.
    let src = r#"
func () -> (i64) {
block 0 () {
  vf = ref.func 1
  va0 = i64.const 10
  vk = cont.new vf va0
  vtag = i64.const -6124895493223874560
  vkt = i64.or vk vtag
  v1 = i64.const 3
  vs1, vr1 = cont.resume vkt v1
  v2 = i64.const 7
  vs2, vr2 = cont.resume vk v2
  v3 = i64.const 20
  vs3, vr3 = cont.resume vkt v3
  vs4, vr4 = cont.resume vk v3
  vs1_64 = i64.extend_i32_s vs1
  vs2_64 = i64.extend_i32_s vs2
  vs3_64 = i64.extend_i32_s vs3
  vs4_64 = i64.extend_i32_s vs4
  va = i64.add vr1 vr2
  vb = i64.add va vr3
  vc = i64.add vb vr4
  vd = i64.add vc vs1_64
  ve = i64.add vd vs2_64
  vg = i64.add ve vs3_64
  vh = i64.add vg vs4_64
  return vh
}
}
func (i64, i64) -> (i64) {
block 0 (va0: i64, va1: i64) {
  vy1 = i64.add va0 va1
  vr1 = suspend vy1
  vt = i64.const 2
  vy2 = i64.mul vr1 vt
  vr2 = suspend vy2
  vf = i64.const 5
  vret = i64.add vr2 vf
  return vret
}
}
"#;
    // vr1 = 13 (s=0), vr2 = 14 (s=0), vr3 = 25 (s=1), vr4 = 0 (s=-9) => 13+14+25+0+0+0+1-9 = 44.
    diff_against_interp(src, &[], true);
}

#[test]
fn threads_futex_ping_pong_and_tls_on_cooperative_and_parallel() {
    // Root spawns worker thread (func 1).
    // Worker stores 1 at [16384], notifies [16384], waits for [16392] == 0 to become 2, checks its
    // TLS != root TLS, and returns 100 + arg0 + arg1 + tls.
    // Root waits for [16384] == 0 to become 1, stores 2 at [16392], notifies [16392], joins worker,
    // and tries a second join (spent -> -9).
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vtls = i64.const 777
  vcpu.tls.set vtls
  va0 = i64.const 20
  va1 = i64.const 22
  vh = thread.spawn 1 va0 va1
  vp0 = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vw0 = i32.atomic.wait vp0 vz vinf
  vp8 = i64.const 16392
  vtwo = i32.const 2
  i32.atomic.store vp8 vtwo
  vone = i32.const 1
  vn = atomic.notify vp8 vone
  vj_st, vj_val = thread.join vh
  vj2_st, vj2_val = thread.join vh
  vmy_tls = vcpu.tls.get
  vs1 = i64.add vj_val vmy_tls
  vs2 = i64.add vs1 vj_st
  vs3 = i64.add vs2 vj2_st
  return vs3
}
}
func (i64, i64) -> (i64) {
block 0 (va0: i64, va1: i64) {
  vp0 = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vp0 vone
  vn = atomic.notify vp0 vone
  vp8 = i64.const 16392
  vz = i32.const 0
  vinf = i64.const -1
  vw8 = i32.atomic.wait vp8 vz vinf
  vwtls = vcpu.tls.get
  v100 = i64.const 100
  va = i64.add va0 va1
  vb = i64.add va v100
  vc = i64.add vb vwtls
  return vc
}
}
"#;
    let m = parse(src);
    // Worker dense TLS starts at 1; worker returns 20+22+100+1 = 143.
    // Root TLS is 777; vj_st = 0; vj2_st = -9 => 143 + 777 + 0 - 9 = 911.
    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let out = run_module(
            &m,
            0,
            &[],
            RunConfig {
                mode,
                ..RunConfig::default()
            },
        )
        .expect("run");
        assert_eq!(out.exit, ContextExit::Returned(vec![911]), "mode={mode:?}");
    }
}

#[test]
fn guest_fiber_parks_non_blockingly_and_migrates_across_carriers() {
    // Root creates GuestFiber (func 2) which calls `i32.atomic.wait` on [16384] == 0.
    // Root resumes it -> receives FIBER_PARKED (status 3) immediately without blocking!
    // Root spawns a worker thread (func 1), passing the fiber handle as arg0.
    // Worker stores 1 at [16384], calls `atomic.notify`, then ITSELF resumes the migrated fiber
    // (on a different Carrier OS thread in Parallel mode!) and returns (status * 1000 + fiber_val).
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 2
  va0 = i64.const 40
  vk = cont.new vf va0
  vz = i64.const 0
  vs, vr = cont.resume vk vz
  vth = thread.spawn 1 vk vz
  vjst, vjval = thread.join vth
  vs64 = i64.extend_i32_s vs
  vm = i64.const 10000
  vhi = i64.mul vs64 vm
  vout = i64.add vhi vjval
  return vout
}
}
func (i64, i64) -> (i64) {
block 0 (vk: i64, vunused: i64) {
  vwtls = i64.const 2
  vcpu.tls.set vwtls
  vp0 = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vp0 vone
  vn = atomic.notify vp0 vone
  vz = i64.const 0
  vs, vr = cont.resume vk vz
  vs64 = i64.extend_i32_s vs
  vk1000 = i64.const 1000
  vhi = i64.mul vs64 vk1000
  vout = i64.add vhi vr
  return vout
}
}
func (i64, i64) -> (i64) {
block 0 (va0: i64, va1: i64) {
  vp0 = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vp0 vz vinf
  vtls = vcpu.tls.get
  vret = i64.add va0 vtls
  return vret
}
}
"#;
    let m = parse(src);
    // Root gets vs = 3 (FIBER_PARKED).
    // Worker wakes fiber, migrates & resumes it; fiber sees worker's TLS = 2 and returns 40 + 2 = 42
    // with vs = 1 (FIBER_RETURNED) => worker returns 1042 => root returns 30000 + 1042 = 31042!
    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let out = run_module(
            &m,
            0,
            &[],
            RunConfig {
                mode,
                ..RunConfig::default()
            },
        )
        .expect("run");
        assert_eq!(
            out.exit,
            ContextExit::Returned(vec![31042]),
            "mode={mode:?}"
        );
    }
}

#[test]
fn cooperative_deadlock_settles_to_thread_fault_and_budget_caps_enforce() {
    // Lone waiter with infinite timeout deadlocks deterministically to ThreadFault (#2967).
    let deadlock = parse(
        r#"
memory 16
func () -> (i64) {
block 0 () {
  vp = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vp vz vinf
  vr = i64.extend_i32_s vw
  return vr
}
}
"#,
    );
    let out = run_module(
        &deadlock,
        0,
        &[],
        RunConfig {
            mode: RunMode::Cooperative,
            ..RunConfig::default()
        },
    )
    .expect("run");
    assert_eq!(out.exit, ContextExit::Trapped(Trap::ThreadFault));

    // Memory ceiling smaller than declared 64 KiB window traps MemoryFault.
    let out_mem = run_module(
        &deadlock,
        0,
        &[],
        RunConfig {
            limits: Limits {
                mem: 4096,
                ..Limits::default()
            },
            ..RunConfig::default()
        },
    )
    .expect("run");
    assert_eq!(out_mem.exit, ContextExit::Trapped(Trap::MemoryFault));
}

#[test]
fn sibling_thread_trap_terminates_parked_root_invariant_6() {
    // Root spawns a thread that immediately hits `unreachable`, then parks forever on a futex.
    // Invariant 6: the sibling trap ends the domain and wakes the parked root cleanly.
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vh = thread.spawn 1 vz vz
  vp = i64.const 16384
  vz32 = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vp vz32 vinf
  vr = i64.extend_i32_s vw
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  unreachable
}
}
"#;
    let m = parse(src);
    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let out = run_module(
            &m,
            0,
            &[],
            RunConfig {
                mode,
                ..RunConfig::default()
            },
        )
        .expect("run");
        assert_eq!(
            out.exit,
            ContextExit::Trapped(Trap::Unreachable),
            "mode={mode:?}"
        );
    }
}

struct EchoCap;

impl CapHandler for EchoCap {
    fn call(
        &self,
        _domain: DomainId,
        _ctx: ContextId,
        handle: i32,
        type_id: u32,
        op: u32,
        args: &[i64],
        _mem: Option<&Region>,
    ) -> CapOutcome {
        if handle == 1 && type_id == 7 && op == 0 {
            CapOutcome::Ready(vec![args[0] + 100])
        } else {
            CapOutcome::Trap(Trap::CapFault)
        }
    }
}

#[test]
fn cap_call_dispatches_and_forged_handle_traps_cap_fault() {
    let src = r#"
func (i32, i64) -> (i64) {
block 0 (vh: i32, vx: i64) {
  vr = call.cap 7 0 (i64) -> (i64) vh (vx)
  return vr
}
}
"#;
    let m = parse(src);
    let ok = run_module(
        &m,
        0,
        &[1, 23],
        RunConfig {
            caps: Arc::new(EchoCap),
            ..RunConfig::default()
        },
    )
    .expect("run");
    assert_eq!(ok.exit, ContextExit::Returned(vec![123]));

    let forged = run_module(
        &m,
        0,
        &[99, 23],
        RunConfig {
            caps: Arc::new(EchoCap),
            ..RunConfig::default()
        },
    )
    .expect("run");
    assert_eq!(forged.exit, ContextExit::Trapped(Trap::CapFault));
}

#[test]
fn cont_resume_block_idles_until_sibling_wakes_parked_fiber() {
    // Root spawns a worker thread that notifies [16384], then Root creates a fiber that waits on
    // [16384] and resumes it with `cont.resume.block` so Root idles cleanly until the fiber completes.
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 2
  va0 = i64.const 50
  vk = cont.new vf va0
  vz = i64.const 0
  vth = thread.spawn 1 vz vz
  vs, vr = cont.resume.block vk vz
  vjst, vjval = thread.join vth
  vs64 = i64.extend_i32_s vs
  vm = i64.const 1000
  vhi = i64.mul vs64 vm
  vout = i64.add vhi vr
  return vout
}
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vp0 = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vp0 vone
  vn = atomic.notify vp0 vone
  return va
}
}
func (i64, i64) -> (i64) {
block 0 (va0: i64, va1: i64) {
  vp0 = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vp0 vz vinf
  vseven = i64.const 7
  vret = i64.add va0 vseven
  return vret
}
}
"#;
    let m = parse(src);
    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let out = run_module(
            &m,
            0,
            &[],
            RunConfig {
                mode,
                ..RunConfig::default()
            },
        )
        .expect("run");
        assert_eq!(out.exit, ContextExit::Returned(vec![1057]), "mode={mode:?}");
    }
}
