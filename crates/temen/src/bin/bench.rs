//! Dependency-free benchmark harness (`DESIGN.md` §18, AGENTS.md "benchmark early").
//!
//! Three jobs:
//!   1. **escape-TCB hot paths** — decode / verify throughput, watched over time for regressions.
//!   2. **interpreter A/B** — the *same* compute kernels run through the **tree-walker** (`run`) and
//!      the **bytecode engine** (`bytecode::compile_and_run`), so we can see where the bytecode
//!      engine stands and measure Phase-2 (memory-op specialization) work against a real baseline
//!      rather than guessing (INTERP_PERF.md "Benchmark first").
//!   3. **concurrency** — threaded and fiber kernels on the tree-walker's scheduler, the cooperative
//!      pump and the parallel driver, the baseline a scheduler change is measured against (#1414
//!      slice 3e, #2695). `--threads` runs this section alone.
//!
//! Each compute kernel takes its **loop count `n`** as the entry argument, so per-iteration compute
//! is isolated by **subtraction** — `(time(large_n) − time(small_n)) / (large_n − small_n)` — which
//! cancels the fixed per-run cost each engine pays (the tree-walker's frame setup, the bytecode
//! engine's per-run *compile*), leaving steady-state op cost. Times are the **min** over repetitions
//! (robust to a noisy box). Uses only `std`.
//!
//! Run: `cargo run --release --bin temen-bench` (`-- --threads` for the concurrency section only,
//! `-- --threads pingpong mutex` for the kernels named)

use std::time::Instant;

use temen::{encode, ir, verify};
use temen_interp::{bytecode, Value};

fn main() {
    // `--threads`: only the concurrency section (the slow, noisy one), for A/B runs of a scheduler
    // change.
    // `--threads name…`: only the kernels named.
    if std::env::args().any(|a| a == "--threads") {
        let only: Vec<String> = std::env::args()
            .skip_while(|a| a != "--threads")
            .skip(1)
            .collect();
        concurrency(&only);
        return;
    }
    // ---- escape-TCB hot paths (decode/verify) ----------------------------------------------
    let module = ir_from_text(ALU);
    let bytes = encode::encode_module(&module);
    println!(
        "module: {} funcs, {} encoded bytes\n",
        module.funcs.len(),
        bytes.len()
    );

    bench("decode", 200_000, || {
        let m = encode::decode_module(&bytes).expect("decode");
        std::hint::black_box(&m);
    });
    bench("verify", 200_000, || {
        let m = encode::decode_module(&bytes).unwrap();
        verify::verify_module(&m).expect("verify");
        std::hint::black_box(&m);
    });

    // ---- interpreter A/B: tree-walker vs bytecode, per-iteration compute --------------------
    println!("\ninterpreter A/B (ns per loop iteration, compute-isolated by subtraction):");
    println!(
        "{:>12}  {:>12}  {:>12}  {:>8}",
        "kernel", "tree-walker", "bytecode", "tw/bc"
    );
    // (name, source, small_n, large_n). Each kernel loops exactly `n` times.
    let kernels = [
        ("alu", ALU, 1_000, 201_000),
        ("call", CALL, 1_000, 201_000),
        ("call_indirect", CALL_INDIRECT, 1_000, 201_000),
        ("mem", MEM, 1_000, 201_000),
    ];
    for (name, src, small, large) in kernels {
        let m = ir_from_text(src);
        // A kernel that traps measures nothing: it would read as 0 ns.
        let tw = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = temen_interp::run(m, 0, &[Value::I32(n)], &mut fuel);
            assert!(r.is_ok(), "{name} trapped on the tree-walker: {r:?}");
            std::hint::black_box(&r);
        });
        let bc = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode engine drives the kernel");
            assert!(r.is_ok(), "{name} trapped on the bytecode engine: {r:?}");
            std::hint::black_box(&r);
        });
        println!("{name:>12}  {tw:>10.2}ns  {bc:>10.2}ns  {:>7.2}×", tw / bc);
    }
    concurrency(&[]);
}

/// **Bytecode concurrency** (#1414 slice 3e): the same threaded and fiber kernels on the cooperative
/// pump (one OS thread, the op-count quantum) and on the parallel driver (executor 2: one OS thread
/// per task, on the pump's rules). Per-iteration cost by subtraction, as the A/B above, so each
/// run's compile, window and thread start-up cancel out. The parallel column is real OS threads, so
/// it is noisier: compare runs of two builds on the same box, interleaved.
fn concurrency(only: &[String]) {
    println!("\nconcurrency (ns per iteration, compute-isolated by subtraction):");
    println!(
        "{:>14}  {:>12}  {:>12}  {:>12}",
        "kernel", "tree-walker", "pump", "parallel"
    );
    // (name, source, small_n, large_n).
    let kernels = [
        ("par_compute", PAR_COMPUTE.into(), 1_000, 101_000),
        ("mutex", MUTEX.into(), 100, 10_100),
        ("pingpong", PINGPONG.into(), 100, 5_100),
        ("pingpong_idle", pingpong_idle(IDLE_PARKED), 100, 2_100),
        ("spawn_join", SPAWN_JOIN.into(), 10, 1_010),
        ("fiber_switch", FIBER_SWITCH.into(), 1_000, 101_000),
        ("fiber_wait", FIBER_WAIT.into(), 1_000, 101_000),
        (
            "fiber_wait_idle",
            fiber_wait_idle(IDLE_PARKED),
            1_000,
            21_000,
        ),
    ];
    for (name, src, small, large) in kernels {
        if !only.is_empty() && !only.iter().any(|o| o == name) {
            continue;
        }
        let m = ir_from_text(&src);
        let tw = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = temen_interp::run(m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(r.expect("the kernel runs to completion"));
        });
        let pump = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode engine drives the kernel");
            std::hint::black_box(r.expect("the kernel runs to completion"));
        });
        let parallel = per_iter(&m, small, large, |m, n| {
            std::hint::black_box(run_parallel(m, n));
        });
        println!("{name:>14}  {tw:>10.1}ns  {pump:>10.1}ns  {parallel:>10.1}ns");
    }
}

/// `m`'s function 0 on the parallel driver, as `temen-run`'s `run_with_caps_parallel` drives it.
fn run_parallel(m: &ir::Module, n: i32) -> Vec<Value> {
    let mut host = temen_interp::Host::new();
    let mut fuel = u64::MAX;
    let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        &[Value::I32(n)],
        &mut fuel,
        &[],
        None,
        &mut host,
    )
    .expect("the parallel driver drives the kernel");
    r.expect("the kernel runs to completion")
}

/// Four threads each run an `n`-step add loop; the root joins them. No scheduler events past the
/// spawns and joins, so this is the parallel driver's per-op cost with four vCPUs live.
const PAR_COMPUTE: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vz = i64.const 0
  vn64 = i64.extend_i32_u vn
  vt0 = thread.spawn 1 vz vn64
  vt1 = thread.spawn 1 vz vn64
  vt2 = thread.spawn 1 vz vn64
  vt3 = thread.spawn 1 vz vn64
  vr0_st, vr0 = thread.join vt0
  vr1_st, vr1 = thread.join vt1
  vr2_st, vr2 = thread.join vt2
  vr3_st, vr3 = thread.join vt3
  va = i64.add vr0 vr1
  vb = i64.add vr2 vr3
  vs = i64.add va vb
  return vs
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  vacc = i64.const 0
  br 1(vn, vacc)
  }
block 1 (vi: i64, vacc: i64) {
  vacc2 = i64.add vacc vi
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2, vacc2) 2(vacc2)
  }
block 2 (vr: i64) {
  return vr
  }
}
"#;

/// Four threads each take a futex mutex `n` times (word 16384: 0 free, 1 held, 2 held with
/// waiters), bump a plain counter (16392) inside it, and release it, waking one waiter when there
/// were any. The root joins them and returns the counter, `4n`. Contended: most rounds wait and
/// notify.
const MUTEX: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vz = i64.const 0
  vn64 = i64.extend_i32_u vn
  vt0 = thread.spawn 1 vz vn64
  vt1 = thread.spawn 1 vz vn64
  vt2 = thread.spawn 1 vz vn64
  vt3 = thread.spawn 1 vz vn64
  vr0_st, vr0 = thread.join vt0
  vr1_st, vr1 = thread.join vt1
  vr2_st, vr2 = thread.join vt2
  vr3_st, vr3 = thread.join vt3
  vc = i64.const 16392
  vcount = i64.load vc
  return vcount
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  br 1(vn)
  }
block 1 (vi: i64) {
  vm = i64.const 16384
  vzero = i32.const 0
  vone = i32.const 1
  vc = i32.atomic.cmpxchg vm vzero vone
  vfree = i32.eq vc vzero
  br_if vfree 4(vi) 2(vi)
  }
block 2 (vi: i64) {
  vm = i64.const 16384
  vtwo = i32.const 2
  vc = i32.atomic.rmw.xchg vm vtwo
  vzero = i32.const 0
  vgot = i32.eq vc vzero
  br_if vgot 4(vi) 3(vi)
  }
block 3 (vi: i64) {
  vm = i64.const 16384
  vtwo = i32.const 2
  vto = i64.const -1
  vst = i32.atomic.wait vm vtwo vto
  br 2(vi)
  }
block 4 (vi: i64) {
  vc = i64.const 16392
  vx = i64.load vc
  vone = i64.const 1
  vx2 = i64.add vx vone
  i64.store vc vx2
  vm = i64.const 16384
  vzero = i32.const 0
  vold = i32.atomic.rmw.xchg vm vzero
  vtwo = i32.const 2
  vwaiters = i32.eq vold vtwo
  br_if vwaiters 5(vi) 6(vi)
  }
block 5 (vi: i64) {
  vm = i64.const 16384
  vcnt = i32.const 1
  vw = atomic.notify vm vcnt
  br 6(vi)
  }
block 6 (vi: i64) {
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2) 7()
  }
block 7 () {
  vz = i64.const 0
  return vz
  }
}
"#;

/// The root and one thread hand a turn word (16384) back and forth `n` times: each waits for its
/// value, stores the other's, and notifies. Every hand-off is a wait and a notify.
const PINGPONG: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vz = i64.const 0
  vn64 = i64.extend_i32_u vn
  vt = thread.spawn 1 vz vn64
  br 1(vn64, vt)
  }
block 1 (vi: i64, vt: i32) {
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vzero = i32.const 0
  vmine = i32.eq vcur vzero
  br_if vmine 3(vi, vt) 2(vi, vt, vcur)
  }
block 2 (vi: i64, vt: i32, vcur: i32) {
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  br 1(vi, vt)
  }
block 3 (vi: i64, vt: i32) {
  vw = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vw vone
  vcnt = i32.const 1
  vk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2, vt) 4(vt)
  }
block 4 (vt: i32) {
  vr_st, vr = thread.join vt
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, vn: i64) {
  br 1(vn)
  }
block 1 (vi: i64) {
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vone = i32.const 1
  vmine = i32.eq vcur vone
  br_if vmine 3(vi) 2(vi, vcur)
  }
block 2 (vi: i64, vcur: i32) {
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  br 1(vi)
  }
block 3 (vi: i64) {
  vw = i64.const 16384
  vzero = i32.const 0
  i32.atomic.store vw vzero
  vcnt = i32.const 1
  vk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2) 4()
  }
block 4 () {
  vz = i64.const 0
  return vz
  }
}
"#;

/// How many threads [`pingpong_idle`] parks for good, and fibers [`fiber_wait_idle`].
const IDLE_PARKED: u32 = 1_000;

/// #2868 — [`PINGPONG`] beside `k` threads parked for good on a word nothing notifies: what a
/// hand-off costs while idle threads wait, as a thread per idle connection does. A scheduler whose
/// rounds or notifies walk every park pays for each of them on every hand-off. The root returns
/// without them: its return ends the run.
fn pingpong_idle(k: u32) -> String {
    let partner = &PINGPONG[PINGPONG.find("func (i64, i64)").expect("the partner")..];
    format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (vn: i32) {{
  vk = i64.const {k}
  br 1(vn, vk)
  }}
block 1 (vn: i32, vk: i64) {{
  vz = i64.const 0
  vmore = i64.ne vk vz
  br_if vmore 2(vn, vk) 3(vn)
  }}
block 2 (vn: i32, vk: i64) {{
  vz = i64.const 0
  vh = thread.spawn 2 vz vz
  vone = i64.const 1
  vk2 = i64.sub vk vone
  br 1(vn, vk2)
  }}
block 3 (vn: i32) {{
  vz = i64.const 0
  vn64 = i64.extend_i32_u vn
  vt = thread.spawn 1 vz vn64
  br 4(vn64, vt)
  }}
block 4 (vi: i64, vt: i32) {{
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vzero = i32.const 0
  vmine = i32.eq vcur vzero
  br_if vmine 6(vi, vt) 5(vi, vt, vcur)
  }}
block 5 (vi: i64, vt: i32, vcur: i32) {{
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  br 4(vi, vt)
  }}
block 6 (vi: i64, vt: i32) {{
  vw = i64.const 16384
  vone = i32.const 1
  i32.atomic.store vw vone
  vcnt = i32.const 1
  vk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 4(vi2, vt) 7(vt)
  }}
block 7 (vt: i32) {{
  vr_st, vr = thread.join vt
  return vr
  }}
}}
{partner}func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vw = i64.const 16400
  vzero = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vw vzero vto
  vr = i64.const 0
  return vr
  }}
}}
"#
    )
}

/// The root spawns a thread that returns its argument and joins it, `n` times in a row.
const SPAWN_JOIN: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vn64 = i64.extend_i32_u vn
  vacc = i64.const 0
  br 1(vn64, vacc)
  }
block 1 (vi: i64, vacc: i64) {
  vz = i64.const 0
  vt = thread.spawn 1 vz vi
  vr_st, vr = thread.join vt
  vacc2 = i64.add vacc vr
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2, vacc2) 2(vacc2)
  }
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

/// One vCPU resumes a fiber `n` times; the fiber counts and suspends each time. A resume and a
/// suspend per iteration: the fiber switch, which no scheduler event touches.
const FIBER_SWITCH: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vn64 = i64.extend_i32_u vn
  br 1(vk, vn64)
  }
block 1 (vk: i64, vi: i64) {
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vmore = i64.ne vi2 vz
  br_if vmore 1(vk, vi2) 2(vv)
  }
block 2 (vv: i64) {
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vz = i64.const 0
  br 1(vz)
  }
block 1 (vc: i64) {
  vone = i64.const 1
  vc2 = i64.add vc vone
  vx = suspend vc2
  br 1(vc2)
  }
}
"#;

/// One vCPU and one fiber that futex-waits forever on word 16384, in a loop. The root notifies the
/// word and resumes the fiber `n` times: each round wakes the fiber, which loops and parks again.
/// A guest M:N scheduler's blocking path: the wait parks the fiber, never the thread.
const FIBER_WAIT: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vn: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vs0, vv0 = cont.resume vk vz
  vn64 = i64.extend_i32_u vn
  br 1(vk, vn64)
  }
block 1 (vk: i64, vi: i64) {
  vw = i64.const 16384
  vcnt = i32.const 1
  vwoken = atomic.notify vw vcnt
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vmore = i64.ne vi2 vz
  br_if vmore 1(vk, vi2) 2(vv)
  }
block 2 (vv: i64) {
  return vv
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  br 1()
  }
block 1 () {
  vw = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vw vexp vto
  br 1()
  }
}
"#;

/// #2868 — [`FIBER_WAIT`] beside `k` fibers parked for good on a word nothing notifies (16400), as
/// a guest M:N scheduler parks a fiber per idle connection. A notify that walks every parked fiber
/// for its waiters pays for each of them on every round.
fn fiber_wait_idle(k: u32) -> String {
    let waiter = &FIBER_WAIT[FIBER_WAIT.find("func (i64, i64)").expect("the waiter")..];
    format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (vn: i32) {{
  vk = i64.const {k}
  br 1(vn, vk)
  }}
block 1 (vn: i32, vk: i64) {{
  vz = i64.const 0
  vmore = i64.ne vk vz
  br_if vmore 2(vn, vk) 3(vn)
  }}
block 2 (vn: i32, vk: i64) {{
  vf = ref.func 2
  vz = i64.const 0
  vc = cont.new vf vz
  vs, vv = cont.resume vc vz
  vone = i64.const 1
  vk2 = i64.sub vk vone
  br 1(vn, vk2)
  }}
block 3 (vn: i32) {{
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vs0, vv0 = cont.resume vk vz
  vn64 = i64.extend_i32_u vn
  br 4(vk, vn64)
  }}
block 4 (vk: i64, vi: i64) {{
  vw = i64.const 16384
  vcnt = i32.const 1
  vwoken = atomic.notify vw vcnt
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i64.const 1
  vi2 = i64.sub vi vone
  vmore = i64.ne vi2 vz
  br_if vmore 4(vk, vi2) 5(vv)
  }}
block 5 (vv: i64) {{
  return vv
  }}
}}
{waiter}func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vw = i64.const 16400
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vw vexp vto
  vr = i64.const 0
  return vr
  }}
}}
"#
    )
}

/// `acc += n; n -= 1` until zero — a pure scalar/branch recurrence (the ALU kernel).
const ALU: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.add v3 v2
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
"#;

/// Each iteration calls a leaf `+1` function — the call/return kernel (window open/close cost).
const CALL: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = call 1(v3)
  v5 = i32.const 1
  v6 = i32.sub v2 v5
  br_if v6 1(v6, v4) 2(v4)
}
block 2 (v7: i32) {
  return v7
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration dispatches through the `call.dyn` table — mask + slot read + type-check.
const CALL_INDIRECT: &str = r#"
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i32.const 1
  v5 = call.dyn (i32) -> (i32) v4 (v3)
  v6 = i32.const 1
  v7 = i32.sub v2 v6
  br_if v7 1(v7, v5) 2(v5)
}
block 2 (v8: i32) {
  return v8
  }
}
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 1
  v2 = i32.add v0 v1
  return v2
  }
}
"#;

/// Each iteration does one `i32.store` + one `i32.load` at a fixed address — the memory kernel that
/// Phase 2 (width-specialized load/store + inlined confinement) targets. The address sits above the
/// NULL guard (#1094), which would trap the first store.
const MEM: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i64.const 16384
  i32.store v4 v3
  v5 = i32.load v4
  v6 = i32.const 1
  v7 = i32.add v5 v6
  v8 = i32.const 1
  v9 = i32.sub v2 v8
  br_if v9 1(v9, v7) 2(v7)
}
block 2 (v10: i32) {
  return v10
  }
}
"#;

fn ir_from_text(src: &str) -> ir::Module {
    temen::text::parse_module(src).expect("corpus program must parse")
}

/// Per-iteration compute (ns) for `run_one(module, n)`, isolated by large/small-`n` subtraction and
/// taken as the min over repetitions (robust to a noisy box).
fn per_iter(m: &ir::Module, small: i32, large: i32, run_one: impl Fn(&ir::Module, i32)) -> f64 {
    let t_small = min_run(m, small, &run_one);
    let t_large = min_run(m, large, &run_one);
    (t_large - t_small) / (large - small) as f64
}

fn min_run(m: &ir::Module, n: i32, run_one: &impl Fn(&ir::Module, i32)) -> f64 {
    // Warm up, then take the fastest of several reps (compute is deterministic; min rejects noise).
    run_one(m, n);
    let reps = 25;
    let mut best = f64::MAX;
    for _ in 0..reps {
        let start = Instant::now();
        run_one(m, n);
        best = best.min(start.elapsed().as_nanos() as f64);
    }
    best
}

fn bench(name: &str, iters: u64, mut f: impl FnMut()) {
    for _ in 0..(iters / 10).max(1) {
        f();
    }
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    let elapsed = start.elapsed();
    let per = elapsed.as_nanos() as f64 / iters as f64;
    println!("{name:>8}: {iters} iters in {elapsed:?}  ({per:.1} ns/iter)");
}
