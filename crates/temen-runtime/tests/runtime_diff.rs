//! End-to-end and differential tests for `temen-runtime` (`Engine::Bytecode` on Cooperative &
//! Parallel Carriers AND `Engine::Cranelift` on native OS threads + `temen-fiber`) against
//! `temen-interp` (`#3187` / `#3196` / `#3197`).

use std::sync::Arc;
use temen_interp::Value;
use temen_ir::Module;
use temen_kernel::{CapOutcome, ContextExit, ContextId, DomainId, Limits, Trap};
use temen_mem::Region;
use temen_runtime::{run_module, CapHandler, RunConfig, RunMode};
use temen_text::parse_module;

#[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
use temen_runtime::{compile_cranelift_with_mode, run, run_cranelift};

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

fn assert_matches_oracle(
    label: &str,
    oracle: &Result<Vec<Value>, temen_interp::Trap>,
    oracle_used: u64,
    out: &temen_runtime::RunOutput,
    want_fuel_match: bool,
) {
    match (oracle, &out.exit) {
        (Ok(ov), ContextExit::Returned(rv)) => {
            let o_i64: Vec<i64> = ov.iter().map(value_as_i64).collect();
            assert_eq!(&o_i64, rv, "result mismatch in {label}");
            if want_fuel_match {
                assert_eq!(
                    oracle_used, out.fuel_used,
                    "exact IR safepoint fuel mismatch in {label}"
                );
            }
        }
        (Err(ot), ContextExit::Trapped(rt)) => {
            assert_eq!(ot.name(), rt.name(), "trap mismatch in {label}");
        }
        other => panic!("oracle vs runtime diverged in {label}: {other:?}"),
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
        let out = run_module(&m, 0, args, cfg).expect("compile & run bytecode");
        assert_matches_oracle(
            &format!("Bytecode({mode:?})"),
            &oracle,
            oracle_used,
            &out,
            want_fuel_match,
        );
    }

    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    {
        for bmode in [
            temen_jit::kernel_engine::BoundMode::DynamicVmCtx,
            temen_jit::kernel_engine::BoundMode::BakedImmediate,
        ] {
            let eng = compile_cranelift_with_mode(&m, bmode).expect("compile cranelift");
            let cfg = RunConfig {
                mode: RunMode::Parallel,
                limits: Limits {
                    fuel: 100_000,
                    ..Limits::default()
                },
                ..RunConfig::default()
            };
            let out = run(eng, 0, args, cfg);
            assert_matches_oracle(
                &format!("Cranelift({bmode:?})"),
                &oracle,
                oracle_used,
                &out,
                want_fuel_match,
            );
        }
    }
}

/// Run `m` across Bytecode(Cooperative), Bytecode(Parallel), and Cranelift (when enabled),
/// asserting `check` on every engine's `RunOutput`.
fn run_all_engines(
    m: &Module,
    args: &[i64],
    make_cfg: impl Fn(RunMode) -> RunConfig,
    check: impl Fn(&str, &temen_runtime::RunOutput),
) {
    for mode in [RunMode::Cooperative, RunMode::Parallel] {
        let out = run_module(m, 0, args, make_cfg(mode)).expect("run bytecode");
        check(&format!("Bytecode({mode:?})"), &out);
    }
    #[cfg(all(feature = "cranelift", not(target_arch = "wasm32")))]
    {
        let out = run_cranelift(m, 0, args, make_cfg(RunMode::Parallel)).expect("run cranelift");
        check("Cranelift", &out);
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
fn diff_arithmetic_conversion_indirect_and_tail_calls() {
    // DivByZero
    let div0 = r#"
func (i64) -> (i64) {
block 0 (vx: i64) {
  vz = i64.const 0
  vr = i64.div_s vx vz
  return vr
}
}
"#;
    diff_against_interp(div0, &[42], false);

    // IntOverflow (i64::MIN / -1)
    let div_ov = r#"
func () -> (i64) {
block 0 () {
  vmin = i64.const -9223372036854775808
  vm1 = i64.const -1
  vr = i64.div_s vmin vm1
  return vr
}
}
"#;
    diff_against_interp(div_ov, &[], false);

    // BadConversion (NaN -> i64 trap)
    let bad_conv = r#"
func () -> (i64) {
block 0 () {
  vz = f64.const 0.0
  vnan = f64.div vz vz
  vr = i64.trunc_f64_s vnan
  return vr
}
}
"#;
    diff_against_interp(bad_conv, &[], false);

    // Indirect call (`call.dyn`) + tail call (`return_call` / `return_call.dyn`) + canonical
    // 64-bit `TypeId` mismatch -> `IndirectCallType`.
    let call_src = r#"
func (i32, i64) -> (i64) {
block 0 (vidx: i32, vx: i64) {
  vr = call.dyn (i64) -> (i64) vidx (vx)
  return vr
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v2 = i64.const 2
  vy = i64.mul vx v2
  return_call 2 (vy)
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v7 = i64.const 7
  vr = i64.add vx v7
  return vr
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vfn = ref.func 2
  return_call.dyn (i64) -> (i64) vfn (vx)
}
}
"#;
    // Valid indirect call -> func 1 tail-calls func 2 => 10 * 2 + 7 = 27.
    diff_against_interp(call_src, &[1, 10], true);
    // Valid indirect call -> func 3 indirect-tail-calls func 2 => 10 + 7 = 17.
    diff_against_interp(call_src, &[3, 10], true);
    // Calling func 0 through `(i64) -> (i64)` (wrong signature!) traps IndirectCallType.
    diff_against_interp(call_src, &[0, 10], false);
}

#[test]
fn diff_out_of_fuel_traps_identically() {
    let spin_src = r#"
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  br 1 (vz)
}
block 1 (vi: i64) {
  v1 = i64.const 1
  vi1 = i64.add vi v1
  br 1 (vi1)
}
}
"#;
    let m = parse(spin_src);
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            limits: Limits {
                fuel: 50,
                ..Limits::default()
            },
            fuel_chunk: 16,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::OutOfFuel), "{label}");
            assert_eq!(out.fuel_used, 50, "exact fuel consumed in {label}");
        },
    );
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
fn diff_forged_fiber_and_thread_handles_trap() {
    let forged_fiber = parse(
        r#"
func () -> (i64) {
block 0 () {
  vbad = i64.const 123456
  vz = i64.const 0
  vs, vr = cont.resume vbad vz
  return vr
}
}
"#,
    );
    run_all_engines(
        &forged_fiber,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::FiberFault), "{label}");
        },
    );

    let forged_thread = parse(
        r#"
func () -> (i64) {
block 0 () {
  vbad = i32.const 999999
  vst, vval = thread.join vbad
  return vval
}
}
"#,
    );
    run_all_engines(
        &forged_thread,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::ThreadFault), "{label}");
        },
    );
}

#[test]
fn threads_futex_ping_pong_and_tls_on_all_engines() {
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
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Returned(vec![911]), "{label}");
        },
    );
}

#[test]
fn guest_fiber_parks_non_blockingly_and_migrates_across_carriers() {
    // Root creates GuestFiber (func 2) which calls `i32.atomic.wait` on [16384] == 0.
    // Root resumes it -> receives FIBER_PARKED (status 3) immediately without blocking!
    // Root spawns a worker thread (func 1), passing the fiber handle as arg0.
    // Worker stores 1 at [16384], calls `atomic.notify`, then ITSELF resumes the migrated fiber
    // (on a different Carrier OS thread in Parallel & Cranelift modes!) and returns
    // (status * 1000 + fiber_val).
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
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Returned(vec![31042]), "{label}");
        },
    );
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

    // Memory ceiling smaller than declared 64 KiB window traps MemoryFault on ALL engines.
    run_all_engines(
        &deadlock,
        &[],
        |mode| RunConfig {
            mode,
            limits: Limits {
                mem: 4096,
                ..Limits::default()
            },
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::MemoryFault), "{label}");
        },
    );
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
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::Unreachable), "{label}");
        },
    );
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
    run_all_engines(
        &m,
        &[1, 23],
        |mode| RunConfig {
            mode,
            caps: Arc::new(EchoCap),
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Returned(vec![123]), "{label}");
        },
    );

    run_all_engines(
        &m,
        &[99, 23],
        |mode| RunConfig {
            mode,
            caps: Arc::new(EchoCap),
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Trapped(Trap::CapFault), "{label}");
        },
    );
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
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Returned(vec![1057]), "{label}");
        },
    );
}

#[test]
fn guest_fiber_hardware_fault_cleans_up_and_traps_memory_fault() {
    // Fiber dereferences NULL inside [0, POWERBOX_NULL_GUARD) -> SIGSEGV inside native `temen-fiber`
    // stack unwinds cleanly back to Carrier's outermost `run_guarded` with zero leaked TLS state.
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vs, vr = cont.resume vk vz
  return vr
}
}
func (i64, i64) -> (i64) {
block 0 (va0: i64, va1: i64) {
  vp = i64.const 64
  vx = i64.load vp
  return vx
}
}
"#;
    let m = parse(src);
    // Run twice sequentially on the same thread to prove TLS fiber stacks were cleanly restored.
    for _ in 0..2 {
        run_all_engines(
            &m,
            &[],
            |mode| RunConfig {
                mode,
                ..RunConfig::default()
            },
            |label, out| {
                assert_eq!(out.exit, ContextExit::Trapped(Trap::MemoryFault), "{label}");
            },
        );
    }
}

#[test]
fn guest_fiber_parks_on_thread_join_and_resumes_cleanly() {
    // Root spawns a worker thread that waits on [16384] == 0 before returning 88.
    // Root creates a fiber that calls `thread.join` on that worker -> parks non-blockingly
    // (`FIBER_PARKED` = 3)! Root then notifies [16384] and resumes with `cont.resume.block`,
    // getting `(FIBER_RETURNED = 1, 88)`!
    let src = r#"
memory 16
func () -> (i64) {
block 0 () {
  vz = i64.const 0
  vth = thread.spawn 1 vz vz
  vth64 = i64.extend_i32_u vth
  vf = ref.func 2
  vk = cont.new vf vth64
  vs1, vr1 = cont.resume vk vz
  vp0 = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vp0 vone
  vn = atomic.notify vp0 vone
  vs2, vr2 = cont.resume.block vk vz
  vs1_64 = i64.extend_i32_s vs1
  vs2_64 = i64.extend_i32_s vs2
  vm1 = i64.const 10000
  vm2 = i64.const 1000
  va = i64.mul vs1_64 vm1
  vb = i64.mul vs2_64 vm2
  vc = i64.add va vb
  vout = i64.add vc vr2
  return vout
}
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  vp0 = i64.const 16384
  vz = i32.const 0
  vinf = i64.const -1
  vw = i32.atomic.wait vp0 vz vinf
  v88 = i64.const 88
  return v88
}
}
func (i64, i64) -> (i64) {
block 0 (vth64: i64, vunused: i64) {
  vth = i32.wrap_i64 vth64
  vst, vval = thread.join vth
  vret = i64.add vst vval
  return vret
}
}
"#;
    let m = parse(src);
    // vs1 = 3 (FIBER_PARKED), vs2 = 1 (FIBER_RETURNED), vr2 = 0 + 88 = 88 => 31088.
    run_all_engines(
        &m,
        &[],
        |mode| RunConfig {
            mode,
            ..RunConfig::default()
        },
        |label, out| {
            assert_eq!(out.exit, ContextExit::Returned(vec![31088]), "{label}");
        },
    );
}
