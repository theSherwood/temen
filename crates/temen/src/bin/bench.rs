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
//! `-- --threads pingpong mutex` for the kernels named). With `--features run`, the pingpong kernels
//! also run on today's Cranelift JIT, the reference a bytecode scheduler change is measured against.

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
///
/// `blocks` is how many times per iteration a parallel-driver thread's park blocked in the OS
/// ([`bytecode::thread_blocks`]): a pingpong iteration is two hand-offs, and most of its cost is
/// its blocks. `jit` is `temen-run`'s JIT backend, for the pingpong kernels, with `--features run`:
/// these modules use §12 ops, so it runs them on the kernel route, as executor 2 tasks (#2958).
fn concurrency(only: &[String]) {
    println!("\nconcurrency (ns per iteration, compute-isolated by subtraction):");
    println!(
        "{:>18}  {:>12}  {:>12}  {:>12}  {:>7}  {:>12}",
        "kernel", "tree-walker", "pump", "parallel", "blocks", "jit"
    );
    // (name, source, small_n, large_n).
    let kernels = [
        ("par_compute", PAR_COMPUTE.into(), 1_000, 101_000),
        ("mutex", MUTEX.into(), 100, 10_100),
        ("pingpong", pingpong_plain(), 100, 5_100),
        ("pingpong_spin100", pingpong(100, 0), 100, 5_100),
        ("pingpong_spin1000", pingpong(1_000, 0), 100, 5_100),
        ("pingpong_spin10000", pingpong(10_000, 0), 100, 5_100),
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
        let blocks = blocks_per_iter(&m, small, large);
        let jit = (name.starts_with("pingpong")).then(|| jit_per_iter(&src, small, large));
        let jit = match jit.flatten() {
            Some(ns) => format!("{ns:>10.1}ns"),
            None => format!("{:>12}", "-"),
        };
        println!(
            "{name:>18}  {tw:>10.1}ns  {pump:>10.1}ns  {parallel:>10.1}ns  {blocks:>7.3}  {jit}"
        );
    }
    if only.is_empty() || only.iter().any(|o| o == "pingpong_work") {
        pingpong_work();
    }
}

/// #2963 — what a park and its wake cost when the waiter always sleeps: [`pingpong`] with a turn of
/// [`WORK_NS`] of work, so each partner has gone to sleep long before its notify comes. Each lane's
/// count is calibrated to that time from [`work_alone`], whose time is then taken out of the
/// hand-off: what is `left` is the park, the wake and the stops. The waiting policy is the guest's
/// (INVARIANTS #4), so this is the mechanism's own cost, with no race between a notify and a sleep.
fn pingpong_work() {
    println!(
        "\npingpong_work (µs per hand-off, the waiter always sleeps; {WORK_NS} ns of work a turn):"
    );
    println!(
        "{:>12}  {:>9}  {:>10}  {:>8}  {:>8}  {:>9}  {:>8}",
        "lane", "count", "hand-off", "work", "left", "blocks", "cpu"
    );
    // (lane, its run of a kernel, whether it counts executor 2's blocks). The `jit` lane runs on the
    // kernel route, whose tasks park on executor 2's parks, so its blocks are counted too (#2985).
    type Lane = (&'static str, fn(&ir::Module), bool);
    #[cfg(feature = "run")]
    let jit: Option<Lane> = Some(("jit", lane_jit, true));
    #[cfg(not(feature = "run"))]
    let jit: Option<Lane> = None;
    let lanes: [Lane; 3] = [
        ("tree-walker", lane_tree_walker, false),
        ("pump", lane_pump, false),
        ("parallel", lane_parallel, true),
    ];
    for (name, lane, counts_blocks) in lanes.into_iter().chain(jit) {
        let step = baked_per_iter(&work_alone(10_000), 20, 220, lane) / 10_000.0;
        let count = ((WORK_NS as f64 / step) as u64).max(1);
        let work = baked_per_iter(&work_alone(count), 20, 120, lane);
        let src = pingpong(0, count);
        let hand_off = baked_per_iter(&src, 20, 120, lane) / 2.0;
        // One more pair of runs for the counts: the blocks, and the CPU every thread spent.
        let (small, large) = (
            ir_from_text(&baked(&src, 20)),
            ir_from_text(&baked(&src, 120)),
        );
        let (b0, c0) = (bytecode::thread_blocks(), cpu_ns());
        lane(&small);
        let (b1, c1) = (bytecode::thread_blocks(), cpu_ns());
        lane(&large);
        let (b2, c2) = (bytecode::thread_blocks(), cpu_ns());
        let per = |a: u64, b: u64, c: u64| ((c - b) as f64 - (b - a) as f64) / 200.0;
        let blocks = if counts_blocks {
            format!("{:>9.3}", per(b0, b1, b2))
        } else {
            format!("{:>9}", "-")
        };
        let cpu = match (c0, c1, c2) {
            (Some(a), Some(b), Some(c)) => format!("{:>7.1}µs", per(a, b, c) / 1_000.0),
            _ => format!("{:>8}", "-"),
        };
        println!(
            "{name:>12}  {count:>9}  {:>8.1}µs  {:>6.1}µs  {:>6.1}µs  {blocks}  {cpu}",
            hand_off / 1_000.0,
            work / 1_000.0,
            (hand_off - work) / 1_000.0,
        );
    }
}

/// [`pingpong_work`]'s turn of work, in ns.
const WORK_NS: u64 = 200_000;

/// Per-iteration ns of `src` with its count baked in ([`baked`]) on `lane`, as [`per_iter`] has it.
fn baked_per_iter(src: &str, small: i32, large: i32, lane: fn(&ir::Module)) -> f64 {
    let (s, l) = (
        ir_from_text(&baked(src, small)),
        ir_from_text(&baked(src, large)),
    );
    let t = |m: &ir::Module| min_run(m, 0, &|m, _| lane(m));
    (t(&l) - t(&s)) / (large - small) as f64
}

/// `src`, whose function 0 takes its count, with the count `n` baked in: function 0 takes no
/// argument, as a lane that cannot pass one (today's JIT, through `temen-run`) runs it.
fn baked(src: &str, n: i32) -> String {
    let entry = "func (i32) -> (i64) {\nblock 0 (vn: i32) {\n";
    assert!(src.contains(entry), "a kernel's entry takes its count");
    src.replacen(
        entry,
        &format!("func () -> (i64) {{\nblock 0 () {{\n  vn = i32.const {n}\n"),
        1,
    )
}

fn lane_tree_walker(m: &ir::Module) {
    let mut fuel = u64::MAX;
    let r = temen_interp::run(m, 0, &[], &mut fuel);
    std::hint::black_box(r.expect("the kernel runs to completion"));
}

fn lane_pump(m: &ir::Module) {
    let mut fuel = u64::MAX;
    let r = bytecode::compile_and_run(m, 0, &[], &mut fuel).expect("bytecode engine drives it");
    std::hint::black_box(r.expect("the kernel runs to completion"));
}

fn lane_parallel(m: &ir::Module) {
    std::hint::black_box(run_parallel_args(m, &[]));
}

/// `temen-run`'s JIT backend, with fuel unmetered: the kernel route for a module with a §12 op (#2958).
#[cfg(feature = "run")]
fn lane_jit(m: &ir::Module) {
    let inst = temen_run::instantiate(m.clone()).expect("the JIT instantiates the kernel");
    let mut config = temen_run::RunConfig::default();
    config.limits.fuel = Some(u64::MAX);
    let run = inst.run_with_caps(temen_run::Backend::Jit, &config, &[]);
    std::hint::black_box(run.expect("the JIT runs the kernel").outcome);
}

/// Per-iteration ns of `src` on today's Cranelift JIT, with `--features run`.
fn jit_per_iter(src: &str, small: i32, large: i32) -> Option<f64> {
    #[cfg(feature = "run")]
    return Some(baked_per_iter(src, small, large, lane_jit));
    #[cfg(not(feature = "run"))]
    {
        let _ = (src, small, large);
        None
    }
}

/// How many times per iteration a parallel-driver thread's park blocked in the OS
/// ([`bytecode::thread_blocks`]), over one run at each count.
fn blocks_per_iter(m: &ir::Module, small: i32, large: i32) -> f64 {
    let b0 = bytecode::thread_blocks();
    run_parallel(m, small);
    let b1 = bytecode::thread_blocks();
    run_parallel(m, large);
    let b2 = bytecode::thread_blocks();
    ((b2 - b1) as f64 - (b1 - b0) as f64) / (large - small) as f64
}

/// The CPU time every thread of this process has spent, ended ones included, in ns: Linux's
/// `/proc/self/stat`, at its 100 Hz tick. `None` elsewhere.
fn cpu_ns() -> Option<u64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields: Vec<&str> = stat[stat.rfind(')')? + 2..].split_whitespace().collect();
    let ticks = fields.get(11)?.parse::<u64>().ok()? + fields.get(12)?.parse::<u64>().ok()?;
    Some(ticks * 10_000_000)
}

/// `m`'s function 0 on the parallel driver, as `temen-run`'s `run_with_caps_parallel` drives it.
fn run_parallel(m: &ir::Module, n: i32) -> Vec<Value> {
    run_parallel_args(m, &[Value::I32(n)])
}

/// [`run_parallel`] with the entry's arguments `args`.
fn run_parallel_args(m: &ir::Module, args: &[Value]) -> Vec<Value> {
    let mut host = temen_interp::Host::new();
    let mut fuel = u64::MAX;
    let (r, _image) = bytecode::compile_and_run_capture_over_parallel_with_host(
        m,
        0,
        args,
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
/// value, stores the other's, and notifies. Every hand-off is a wait and a notify. [`pingpong`]
/// with no spin and no work.
fn pingpong_plain() -> String {
    pingpong(0, 0)
}

/// #2963 — [`pingpong_plain`]'s hand-off, with what a guest does around it. Before each wait a side
/// re-reads the word up to `spin` times, as a guest's lock spins before it parks; on its turn it
/// counts down `work` before it flips the word and notifies, so its partner has long since gone to
/// sleep. How often a partner's notify beats the waiter to its sleep depends on both sides' path
/// lengths, so the plain pingpong's cost is partly an accident of them: a spin takes the accident
/// out one way (no side sleeps), and work the other (every waiter sleeps).
fn pingpong(spin: u64, work: u64) -> String {
    // One side: `me` is the word's value on its turn, `next` the other side's. `head` is its block 0,
    // which branches to block 1 with a full spin; `end` is its block 7. The root carries its
    // thread's handle `vt` through every block (`ca` declares it, `c` passes it), to join it at
    // the end.
    let side = |me: u32, next: u32, head: &str, ca: &str, end: &str| {
        let (c, c0) = if ca.is_empty() {
            ("", "")
        } else {
            (", vt", "vt")
        };
        format!(
            r#"{head}
block 1 (vi: i64{ca}, vs: i64) {{
  vw = i64.const 16384
  vcur = i32.atomic.load vw
  vme = i32.const {me}
  vmine = i32.eq vcur vme
  br_if vmine 4(vi{c}) 2(vi{c}, vs, vcur)
  }}
block 2 (vi: i64{ca}, vs: i64, vcur: i32) {{
  vz = i64.const 0
  vout = i64.eq vs vz
  vone = i64.const 1
  vs2 = i64.sub vs vone
  br_if vout 3(vi{c}, vcur) 1(vi{c}, vs2)
  }}
block 3 (vi: i64{ca}, vcur: i32) {{
  vw = i64.const 16384
  vto = i64.const -1
  vst = i32.atomic.wait vw vcur vto
  vs = i64.const {spin}
  br 1(vi{c}, vs)
  }}
block 4 (vi: i64{ca}) {{
  vk = i64.const {work}
  br 5(vi{c}, vk)
  }}
block 5 (vi: i64{ca}, vk: i64) {{
  vz = i64.const 0
  vdone = i64.eq vk vz
  vone = i64.const 1
  vk2 = i64.sub vk vone
  br_if vdone 6(vi{c}) 5(vi{c}, vk2)
  }}
block 6 (vi: i64{ca}) {{
  vw = i64.const 16384
  vnext = i32.const {next}
  i32.atomic.store vw vnext
  vcnt = i32.const 1
  vkk = atomic.notify vw vcnt
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  vs = i64.const {spin}
  br_if vmore 1(vi2{c}, vs) 7({c0})
  }}
{end}
}}
"#
        )
    };
    let root = side(
        0,
        1,
        &format!(
            "func (i32) -> (i64) {{\nblock 0 (vn: i32) {{\n  vz = i64.const 0\n  vn64 = i64.extend_i32_u vn\n  vt = thread.spawn 1 vz vn64\n  vs = i64.const {spin}\n  br 1(vn64, vt, vs)\n  }}"
        ),
        ", vt: i32",
        "block 7 (vt: i32) {\n  vr_st, vr = thread.join vt\n  return vr\n  }",
    );
    let partner = side(
        1,
        0,
        &format!("func (i64, i64) -> (i64) {{\nblock 0 (vsp: i64, vn: i64) {{\n  vs = i64.const {spin}\n  br 1(vn, vs)\n  }}"),
        "",
        "block 7 () {\n  vz = i64.const 0\n  return vz\n  }",
    );
    format!("memory 16\n{root}{partner}")
}

/// #2963 — one thread counts down `work`, `n` times: a [`pingpong`] turn's work alone, which
/// [`pingpong_work`] calibrates its count by and takes out of the hand-off.
fn work_alone(work: u64) -> String {
    format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (vn: i32) {{
  vn64 = i64.extend_i32_u vn
  br 1(vn64)
  }}
block 1 (vi: i64) {{
  vk = i64.const {work}
  br 2(vi, vk)
  }}
block 2 (vi: i64, vk: i64) {{
  vz = i64.const 0
  vdone = i64.eq vk vz
  vone = i64.const 1
  vk2 = i64.sub vk vone
  br_if vdone 3(vi) 2(vi, vk2)
  }}
block 3 (vi: i64) {{
  vone64 = i64.const 1
  vi2 = i64.sub vi vone64
  vz = i64.const 0
  vmore = i64.ne vi2 vz
  br_if vmore 1(vi2) 4()
  }}
block 4 () {{
  vz = i64.const 0
  return vz
  }}
}}
"#
    )
}

/// How many threads [`pingpong_idle`] parks for good, and fibers [`fiber_wait_idle`].
const IDLE_PARKED: u32 = 1_000;

/// #2868 — [`pingpong_plain`] beside `k` threads parked for good on a word nothing notifies: what a
/// hand-off costs while idle threads wait, as a thread per idle connection does. A scheduler whose
/// rounds or notifies walk every park pays for each of them on every hand-off. The root returns
/// without them: its return ends the run.
fn pingpong_idle(k: u32) -> String {
    let plain = pingpong_plain();
    let partner = &plain[plain.find("func (i64, i64)").expect("the partner")..];
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
