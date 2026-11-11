//! **#2948 spike measurements** (the `--kernel-spike` mode): today's Cranelift JIT against the spike
//! path, where its vCPUs are tasks of executor 2's scheduling core (`temen_run::jit_kernel`), and,
//! for the park-and-wake probe, executor 2 interpreting the same program, for reference.
//!
//! Every lane is a release build on this machine, and the lanes run interleaved: each rep runs every
//! lane once before the next rep starts. A row reports the median over the reps and the spread
//! (min..max). Per-iteration costs are by subtraction of a small-`n` run from a large-`n` one, so
//! each lane's fixed setup (compiles, the window, the run's open and teardown) cancels; the setup
//! itself is reported apart.
//!
//! Scaffolding for the spike, deleted with it.

use std::time::{Duration, Instant};
use temen_interp::{Host, Value};
use temen_ir::Module;
use temen_jit::{CompiledModule, JitOutcome};

const RESERVED: u8 = temen_ir::DEFAULT_RESERVED_LOG2;

fn parse(ir: &str) -> Module {
    let m = temen_text::parse_module(ir).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// How a lane runs a program once: `(result, elapsed)`.
type Lane<'a> = Box<dyn FnMut(&[i64]) -> (i64, Duration) + 'a>;

/// Today's JIT, compiled once over `host`'s powerbox; `locked` is the serialized thunk a concurrent
/// program runs (`jit_cap_run`'s choice), else the raw one.
fn today<'a>(m: &'a Module, host: Host, locked: bool) -> Lane<'a> {
    let cell = Box::leak(Box::new(std::sync::Mutex::new(host)));
    let (thunk, ctx): (temen_jit::CapThunk, *mut core::ffi::c_void) = if locked {
        (temen_run::cap_thunk_locked, cell as *const _ as *mut _)
    } else {
        let h = cell.get_mut().unwrap() as *mut Host;
        (temen_run::cap_thunk, h as *mut _)
    };
    let mut cm =
        CompiledModule::compile(m, 0, thunk, ctx, RESERVED, None, None, None, None, None, 0)
            .expect("jit compile");
    // A fresh, untimed compile per run, as the spike lane's: a module run again after a run that
    // left 200 or more vCPUs parked traps `ThreadFault` on today's JIT (`SPIKE_FRESH=0` shows it).
    let fresh = std::env::var_os("SPIKE_FRESH").is_none_or(|v| v != "0");
    Box::new(move |args| {
        if fresh {
            cm = CompiledModule::compile(
                m, 0, thunk, ctx, RESERVED, None, None, None, None, None, 0,
            )
            .expect("jit compile");
        }
        let t0 = Instant::now();
        let (out, _) = cm.run(args, None, None).expect("jit run");
        let dt = t0.elapsed();
        match out {
            JitOutcome::Returned(v) => (v[0], dt),
            o => panic!("today's JIT at {args:?}: {o:?}"),
        }
    })
}

/// The spike path. Each run takes a fresh compile (its window is the run's), made untimed. `keyed`
/// runs it with the #2868 keyed-queue prototype.
fn spike<'a>(m: &'a Module, mut host: impl FnMut() -> Host + 'a, keyed: bool) -> Lane<'a> {
    Box::new(move |args| {
        temen_run::jit_kernel::set_keyed(keyed);
        let cm = temen_run::jit_kernel::compile(m, 0, RESERVED, None).expect("in scope");
        let mut h = host();
        let t0 = Instant::now();
        let r =
            temen_run::jit_kernel::run_compiled(cm, m, 0, args, &mut h, None).expect("in scope");
        let dt = t0.elapsed();
        (r.expect("spike run")[0], dt)
    })
}

/// Executor 2 interpreting `m`: no emitted tier.
fn x2_interp(m: &Module) -> Lane<'_> {
    Box::new(move |args| {
        let vals: Vec<Value> = args.iter().map(|a| Value::I64(*a)).collect();
        let t0 = Instant::now();
        let mut run = temen_interp::bytecode::CoopRun::new_parallel(
            m,
            0,
            &vals,
            u64::MAX,
            Host::new(),
            &[],
            RESERVED,
            None,
            false,
        )
        .expect("subset")
        .expect("open");
        let out = run.run();
        let dt = t0.elapsed();
        match out {
            temen_interp::bytecode::CoopEvent::Done(v) => match v[0] {
                Value::I64(x) => (x, dt),
                ref o => panic!("x2: {o:?}"),
            },
            _ => panic!("x2: the run did not end with a value"),
        }
    })
}

/// Median and spread of `xs`, as `median [min..max]`.
fn stats(xs: &mut [f64]) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (xs[xs.len() / 2], xs[0], xs[xs.len() - 1])
}

fn fmt(x: (f64, f64, f64), unit: &str) -> String {
    format!("{:.3}{unit} [{:.3}..{:.3}]", x.0, x.1, x.2)
}

/// Per-iteration cost of each lane by subtraction, interleaved over `reps`: lane `i` runs
/// `args(n_small)` then `args(n_big)` in each rep. `per` divides the difference (hand-offs per `n`).
fn per_iter(
    label: &str,
    lanes: &mut [(&str, Lane<'_>)],
    args: impl Fn(i64) -> Vec<i64>,
    (n_small, n_big): (i64, i64),
    per: f64,
    reps: usize,
    unit_ns: f64,
    unit: &str,
) {
    let mut per_lane: Vec<Vec<f64>> = vec![Vec::new(); lanes.len()];
    let mut setup: Vec<Vec<f64>> = vec![Vec::new(); lanes.len()];
    for _ in 0..reps {
        for (i, (_, lane)) in lanes.iter_mut().enumerate() {
            let (r0, t0) = lane(&args(n_small));
            let (r1, t1) = lane(&args(n_big));
            std::hint::black_box((r0, r1));
            let d = t1.as_secs_f64() - t0.as_secs_f64();
            per_lane[i].push(d * 1e9 / ((n_big - n_small) as f64 * per) / unit_ns);
            setup[i].push(t0.as_secs_f64() * 1e3);
        }
    }
    for (i, (name, _)) in lanes.iter().enumerate() {
        println!(
            "  {label:<22} {name:<10} {:>36}   run at n={n_small}: {}",
            fmt(stats(&mut per_lane[i]), unit),
            fmt(stats(&mut setup[i]), "ms"),
        );
    }
}

/// The #2868 probe: a pingpong, two vCPUs handing a turn word back and forth through
/// `memory.wait`/`memory.notify`, beside `k` threads parked for good on a word nothing notifies.
/// `func 0 (n, k)`: spawn the `k` idlers, wait until each has counted itself in, spawn the partner,
/// then take `n` turns and join it. Each turn is two hand-offs.
pub fn pingpong_ir() -> &'static str {
    r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vn: i64, vk: i64) {
  vz = i64.const 0
  br 1(vn, vk, vz)
}
block 1 (vn1: i64, vk1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vk1
  br_if vlt1 2(vn1, vk1, vi1) 3(vn1, vk1)
}
block 2 (vn2: i64, vk2: i64, vi2: i64) {
  vh2 = thread.spawn 1 vi2 vi2
  vone2 = i64.const 1
  vin2 = i64.add vi2 vone2
  br 1(vn2, vk2, vin2)
}
block 3 (vn3: i64, vk3: i64) {
  va3 = i64.const 16384
  vc3 = i64.atomic.load va3
  vlt3 = i64.lt_u vc3 vk3
  br_if vlt3 3(vn3, vk3) 4(vn3)
}
block 4 (vn4: i64) {
  vb4 = thread.spawn 2 vn4 vn4
  vz4 = i64.const 0
  br 5(vn4, vb4, vz4)
}
block 5 (vn5: i64, vb5: i32, vi5: i64) {
  vlt5 = i64.lt_u vi5 vn5
  br_if vlt5 6(vn5, vb5, vi5) 8(vb5)
}
block 6 (vn6: i64, vb6: i32, vi6: i64) {
  vt6 = i64.const 16400
  vcur6 = i32.atomic.load vt6
  vone6 = i32.const 1
  veq6 = i32.eq vcur6 vone6
  br_if veq6 7(vn6, vb6, vi6) 9(vn6, vb6, vi6)
}
block 7 (vn7: i64, vb7: i32, vi7: i64) {
  vt7 = i64.const 16400
  vone7 = i32.const 1
  vinf7 = i64.const -1
  vs7 = i32.atomic.wait vt7 vone7 vinf7
  br 6(vn7, vb7, vi7)
}
block 8 (vb8: i32) {
  vst8, vr8 = thread.join vb8
  return vr8
}
block 9 (vn9: i64, vb9: i32, vi9: i64) {
  vt9 = i64.const 16400
  vone9 = i32.const 1
  i32.atomic.store vt9 vone9
  vw9 = atomic.notify vt9 vone9
  vinc9 = i64.const 1
  vin9 = i64.add vi9 vinc9
  br 5(vn9, vb9, vin9)
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
func (i64, i64) -> (i64) {
block 0 (vn: i64, vm: i64) {
  vz = i64.const 0
  br 1(vn, vz)
}
block 1 (vn1: i64, vi1: i64) {
  vlt1 = i64.lt_u vi1 vn1
  br_if vlt1 2(vn1, vi1) 4(vi1)
}
block 2 (vn2: i64, vi2: i64) {
  vt2 = i64.const 16400
  vcur2 = i32.atomic.load vt2
  vz2 = i32.const 0
  veq2 = i32.eq vcur2 vz2
  br_if veq2 3(vn2, vi2) 5(vn2, vi2)
}
block 3 (vn3: i64, vi3: i64) {
  vt3 = i64.const 16400
  vz3 = i32.const 0
  vinf3 = i64.const -1
  vs3 = i32.atomic.wait vt3 vz3 vinf3
  br 2(vn3, vi3)
}
block 4 (vr4: i64) {
  return vr4
}
block 5 (vn5: i64, vi5: i64) {
  vt5 = i64.const 16400
  vz5 = i32.const 0
  i32.atomic.store vt5 vz5
  vone5 = i32.const 1
  vw5 = atomic.notify vt5 vone5
  vinc5 = i64.const 1
  vin5 = i64.add vi5 vinc5
  br 1(vn5, vin5)
  }
}
"#
}

pub fn run(reps: usize) {
    let reps = reps.max(5);
    println!("#2948 spike: today's Cranelift JIT vs the spike path (vCPUs as executor 2 tasks)");
    println!("  median [min..max] over {reps} interleaved reps; release build\n");

    // `SPIKE_SECTIONS=compute,spawn,park` picks sections (all by default).
    let only = std::env::var("SPIKE_SECTIONS").unwrap_or_default();
    let on = |s: &str| only.is_empty() || only.split(',').any(|x| x == s);
    if on("compute") {
        println!("compute and synchronous call.cap (ns per iteration):");
        for k in [&crate::ALU, &crate::MEMSUM] {
            let m = parse(k.ir);
            let mut lanes = [
                ("today", today(&m, Host::new(), false)),
                ("spike", spike(&m, Host::new, false)),
            ];
            per_iter(
                k.name,
                &mut lanes,
                |n| vec![n],
                (1_000, 20_000_000),
                1.0,
                reps,
                1.0,
                "ns",
            );
        }
        {
            let k = crate::hostproc_kernel();
            let m = parse(&k.ir);
            let lead = k.lead_args.clone();
            let args = move |n| {
                let mut a = lead.clone();
                a.push(n);
                a
            };
            let mut lanes = [
                ("today", today(&m, crate::dispatch_host().0, false)),
                ("spike", spike(&m, || crate::dispatch_host().0, false)),
            ];
            per_iter(
                "hostproc",
                &mut lanes,
                args,
                (1_000, 2_000_000),
                1.0,
                reps,
                1.0,
                "ns",
            );
        }
    }
    if on("spawn") {
        println!("\nspawn and join (bench --threads, ms per run; spawn + work + join):");
        for (label, nw, steps) in [
            ("spawn-heavy (64x1k)", 64u32, 1_000u32),
            ("balanced (8x100k)", 8, 100_000),
            ("compute-heavy (4x2M)", 4, 2_000_000),
        ] {
            let wasm = wat::parse_str(crate::threads::parallel_wat(nw, steps)).expect("wat");
            let t = temen_wasm::transpile(&wasm).expect("transpile");
            let entry = t.exports.iter().find(|(n, _)| n == "run").unwrap().1;
            let m = t.module;
            let mut a: Vec<f64> = Vec::new();
            let mut b: Vec<f64> = Vec::new();
            let mut cm = CompiledModule::compile(
                &m,
                entry,
                temen_jit::INERT_CAP_THUNK,
                std::ptr::null_mut(),
                RESERVED,
                None,
                None,
                None,
                None,
                None,
                0,
            )
            .expect("compile");
            let want = (nw as i64) * (steps as i64);
            for _ in 0..reps {
                let t0 = Instant::now();
                let (out, _) = cm.run(&[], None, None).expect("run");
                a.push(t0.elapsed().as_secs_f64() * 1e3);
                assert!(
                    matches!(out, JitOutcome::Returned(ref v) if v[0] == want),
                    "today: {out:?}"
                );
                let c =
                    temen_run::jit_kernel::compile(&m, entry, RESERVED, None).expect("in scope");
                let mut h = Host::new();
                let t0 = Instant::now();
                let r = temen_run::jit_kernel::run_compiled(c, &m, entry, &[], &mut h, None)
                    .expect("scope");
                b.push(t0.elapsed().as_secs_f64() * 1e3);
                assert_eq!(r.expect("spike run")[0], want, "spike result");
            }
            let (sa, sb) = (stats(&mut a), stats(&mut b));
            println!(
                "  {label:<22} today {:>28}   spike {:>28}   ratio {:.2}",
                fmt(sa, "ms"),
                fmt(sb, "ms"),
                sb.0 / sa.0
            );
        }
    }
    if !on("park") {
        return;
    }
    println!("\npark and wake: #2868 pingpong beside K idle parked threads (us per hand-off):");
    let m = parse(pingpong_ir());
    // `SPIKE_K=0,100` picks the K values.
    let ks: Vec<i64> = std::env::var("SPIKE_K")
        .map(|v| v.split(',').map(|x| x.parse().expect("SPIKE_K")).collect())
        .unwrap_or_else(|_| vec![0, 100, 1_000]);
    for k in ks {
        let lanes = vec![
            ("today", today(&m, Host::new(), true)),
            ("spike", spike(&m, Host::new, false)),
            ("spike+kq", spike(&m, Host::new, true)),
            ("x2-interp", x2_interp(&m)),
        ];
        // `SPIKE_LANES=today,spike` picks lanes.
        let pick = std::env::var("SPIKE_LANES").unwrap_or_default();
        let mut lanes: Vec<_> = lanes
            .into_iter()
            .filter(|(n, _)| pick.is_empty() || pick.split(',').any(|x| x == *n))
            .collect();
        let span = if k >= 1_000 {
            (100, 2_100)
        } else {
            (100, 5_100)
        };
        per_iter(
            &format!("K = {k}"),
            &mut lanes,
            move |n| vec![n, k],
            span,
            2.0,
            reps,
            1e3,
            "us",
        );
    }
}
