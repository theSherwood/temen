//! Dependency-free benchmark harness (`DESIGN.md` §18, AGENTS.md "benchmark early").
//!
//! Three jobs:
//!   1. **escape-TCB hot paths** — decode / verify throughput, watched over time for regressions.
//!   2. **interpreter A/B** — the *same* compute kernels run through the **tree-walker** (`run`) and
//!      the **bytecode engine** (`bytecode::compile_and_run`), so we can see where the bytecode
//!      engine stands and measure Phase-2 (memory-op specialization) work against a real baseline
//!      rather than guessing (INTERP_PERF.md "Benchmark first").
//!   3. **bytecode concurrency** — threaded and fiber kernels on the cooperative pump and on the
//!      parallel driver, the baseline a scheduler change is measured against (#1414 slice 3e).
//!      `--threads` runs this section alone.
//!
//! Each compute kernel takes its **loop count `n`** as the entry argument, so per-iteration compute
//! is isolated by **subtraction** — `(time(large_n) − time(small_n)) / (large_n − small_n)` — which
//! cancels the fixed per-run cost each engine pays (the tree-walker's frame setup, the bytecode
//! engine's per-run *compile*), leaving steady-state op cost. Times are the **min** over repetitions
//! (robust to a noisy box). Uses only `std`.
//!
//! Run: `cargo run --release --bin temen-bench` (`-- --threads` for the concurrency section only)

use std::time::Instant;

use temen::{encode, ir, verify};
use temen_interp::{bytecode, Value};

fn main() {
    // `--threads`: only the bytecode concurrency section (the slow, noisy one), for A/B runs of a
    // scheduler change.
    if std::env::args().any(|a| a == "--threads") {
        concurrency();
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
        let tw = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = temen_interp::run(m, 0, &[Value::I32(n)], &mut fuel);
            std::hint::black_box(&r);
        });
        let bc = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode engine drives the kernel");
            std::hint::black_box(&r);
        });
        println!("{name:>12}  {tw:>10.2}ns  {bc:>10.2}ns  {:>7.2}×", tw / bc);
    }
    concurrency();
}

/// **Bytecode concurrency** (#1414 slice 3e): the same threaded and fiber kernels on the cooperative
/// pump (one OS thread, the op-count quantum) and on the parallel driver (one OS thread per vCPU).
/// Per-iteration cost by subtraction, as the A/B above, so each run's compile, window and thread
/// start-up cancel out. The parallel column is real OS threads, so it is noisier: compare runs of
/// two builds on the same box, interleaved.
fn concurrency() {
    println!("\nbytecode concurrency (ns per iteration, compute-isolated by subtraction):");
    println!("{:>14}  {:>12}  {:>12}", "kernel", "pump", "parallel");
    // (name, source, small_n, large_n, runs on the parallel driver).
    let kernels = [
        ("par_compute", PAR_COMPUTE, 1_000, 101_000, true),
        ("mutex", MUTEX, 100, 10_100, true),
        ("pingpong", PINGPONG, 100, 5_100, true),
        ("spawn_join", SPAWN_JOIN, 10, 1_010, true),
        ("fiber_switch", FIBER_SWITCH, 1_000, 101_000, true),
        // #2215: a futex wait inside a fiber blocks the whole thread on the parallel driver.
        ("fiber_wait", FIBER_WAIT, 1_000, 101_000, false),
    ];
    for (name, src, small, large, on_parallel) in kernels {
        let m = ir_from_text(src);
        let pump = per_iter(&m, small, large, |m, n| {
            let mut fuel = u64::MAX;
            let r = bytecode::compile_and_run(m, 0, &[Value::I32(n)], &mut fuel)
                .expect("bytecode engine drives the kernel");
            std::hint::black_box(r.expect("the kernel runs to completion"));
        });
        let parallel = if on_parallel {
            let t = per_iter(&m, small, large, |m, n| {
                std::hint::black_box(run_parallel(m, n));
            });
            format!("{t:>10.1}ns")
        } else {
            format!("{:>12}", "-")
        };
        println!("{name:>14}  {pump:>10.1}ns  {parallel}");
    }
}

/// `m`'s function 0 on the parallel driver over a fresh zeroed window, as `temen-run`'s
/// `run_with_caps_parallel` drives it.
fn run_parallel(m: &ir::Module, n: i32) -> Vec<Value> {
    let size = 1usize
        << m.memory
            .expect("a threaded kernel declares memory")
            .size_log2;
    let layout = std::alloc::Layout::from_size_align(size, 4096).expect("layout");
    // SAFETY: a non-zero, page-aligned layout, freed below once the run has joined every vCPU.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null(), "window allocation");
    // SAFETY: `base` owns `size` zeroed bytes and outlives the run.
    let back = std::sync::Arc::new(unsafe { temen_interp::Region::shared(base, size as u64) });
    let mut host = temen_interp::Host::new();
    let mut fuel = u64::MAX;
    let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        &[Value::I32(n)],
        &mut fuel,
        &[],
        std::sync::Arc::clone(&back),
        &mut host,
    )
    .expect("the parallel driver drives the kernel");
    drop(back);
    // SAFETY: the same layout; every vCPU joined and dropped its view of the region.
    unsafe { std::alloc::dealloc(base, layout) };
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
  vr0 = thread.join vt0
  vr1 = thread.join vt1
  vr2 = thread.join vt2
  vr3 = thread.join vt3
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
  vr0 = thread.join vt0
  vr1 = thread.join vt1
  vr2 = thread.join vt2
  vr3 = thread.join vt3
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
  vr = thread.join vt
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
  vr = thread.join vt
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
/// Phase 2 (width-specialized load/store + inlined confinement) targets.
const MEM: &str = r#"memory 16
func (i32) -> (i32) {
block 0 (v0: i32) {
  v1 = i32.const 0
  br 1(v0, v1)
}
block 1 (v2: i32, v3: i32) {
  v4 = i64.const 0
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
