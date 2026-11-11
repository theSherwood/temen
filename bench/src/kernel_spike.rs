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
    crate::kernel_spike_ir::PINGPONG_IR
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
                // What today's JIT pays for the same call in a program that spawns.
                ("today-lock", today(&m, crate::dispatch_host().0, true)),
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
            let compile = || {
                CompiledModule::compile(
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
                .expect("compile")
            };
            let want = (nw as i64) * (steps as i64);
            for _ in 0..reps {
                // A fresh, untimed compile per run, as the spike's: a module run again traps
                // `ThreadFault` in `thread.spawn` once enough earlier runs spawned on it, on
                // today's JIT.
                let mut cm = compile();
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
    // Off unless asked for: `SPIKE_SECTIONS=pairs`, `nowake`, `k1000`.
    if on("pairs") && !only.is_empty() {
        run_pairs(reps);
    }
    if on("nowake") && !only.is_empty() {
        run_nowake(reps);
    }
    if on("k1000") && !only.is_empty() {
        run_k1000(reps);
    }
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
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
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
}

/// The no-wake probe ([`crate::kernel_spike_ir::NOWAKE_IR`], `SPIKE_SECTIONS=nowake`): ns per
/// `memory.notify` with no waiter and per `memory.wait` that answers not-equal at once, beside `k`
/// threads parked for good. No thread wakes, so the OS's wake cost is out of it: what is left is the
/// runtime's own per-op cost, which a futex-based guest mutex pays on each contended unlock.
pub fn run_nowake(reps: usize) {
    println!("\nno wake: a notify with no waiter / a wait that answers not-equal (ns per op):");
    let m = parse(crate::kernel_spike_ir::NOWAKE_IR);
    for (mode, what) in [(0i64, "notify, none"), (1, "wait, not-equal")] {
        for k in [0i64, 100, 1_000] {
            let label = format!("{what}, K = {k}");
            let args = move |n| vec![n, k, mode];
            // A span long enough that each lane's fixed cost (spawning the K threads, about 0.6 s
            // at K = 1,000 on today's lane) subtracts to a usable range; the spike without the
            // queue costs ~0.1 us per parked task per op, so it takes a short one.
            let long = if k >= 1_000 { 20_001_000 } else { 2_001_000 };
            let mut fast = vec![
                ("today", today(&m, Host::new(), true)),
                ("spike+kq", spike(&m, Host::new, true)),
            ];
            per_iter(&label, &mut fast, args, (1_000, long), 1.0, reps, 1.0, "ns");
            let mut slow = vec![("spike", spike(&m, Host::new, false))];
            per_iter(
                &label,
                &mut slow,
                args,
                (1_000, 101_000),
                1.0,
                reps,
                1.0,
                "ns",
            );
        }
    }
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
}

/// K = 1,000 of the pingpong with a span long enough that today's run, whose fixed cost is spawning
/// the 1,000 threads (about 0.6 s, give or take 0.2 s), still subtracts to a usable range:
/// `SPIKE_SECTIONS=k1000`. The spike without the queue keeps the short span (its cost is 100x).
pub fn run_k1000(reps: usize) {
    println!("\npark and wake, K = 1000, long span (us per hand-off):");
    let m = parse(crate::kernel_spike_ir::PINGPONG_IR);
    let mut lanes = vec![
        ("today", today(&m, Host::new(), true)),
        ("spike+kq", spike(&m, Host::new, true)),
    ];
    let args = |n| vec![n, 1_000];
    per_iter(
        "K = 1000",
        &mut lanes,
        args,
        (100, 200_100),
        2.0,
        reps,
        1e3,
        "us",
    );
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
}

/// The run-lock probe ([`crate::kernel_spike_ir::PAIRS_IR`], `SPIKE_SECTIONS=pairs`): `p` independent pingpong pairs, each
/// on a turn word of its own a cache line apart, so nothing but the runtime's own locks couples
/// them. Per pair, a hand-off costs what it does alone if the runtime runs unrelated hand-offs in
/// parallel, and up to `p` times that if one lock serializes them.
pub fn run_pairs(reps: usize) {
    println!("\nrun lock: P independent pingpong pairs at once (us per hand-off, per pair):");
    let m = parse(crate::kernel_spike_ir::PAIRS_IR);
    for p in [1i64, 2, 4] {
        let mut lanes = vec![
            ("today", today(&m, Host::new(), true)),
            ("spike+kq", spike(&m, Host::new, true)),
        ];
        let args = move |n| vec![n, p];
        per_iter(
            &format!("P = {p}"),
            &mut lanes,
            args,
            (100, 5_100),
            2.0,
            reps,
            1e3,
            "us",
        );
    }
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
}
