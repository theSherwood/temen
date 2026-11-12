//! **#2958 — the kernel-route probe** (the `--kernel-spike` mode): today's Cranelift JIT, whose
//! threads run on its own runtime, against the kernel route, where they are tasks of executor 2's
//! scheduling core (`temen_run::jit_kernel`). Phase 1's exit condition is read from it (#2958); its
//! park-and-wake sections land in slice 1b.
//!
//! Every lane is a release build on this machine, and the lanes run interleaved: each rep runs every
//! lane once before the next rep starts. A row reports the median over the reps and the spread
//! (min..max). Per-iteration costs are by subtraction of a small-`n` run from a large-`n` one, so
//! each lane's fixed setup (compiles, the window, the run's open and teardown) cancels; the setup
//! itself is reported apart. Every lane compiles afresh, untimed, before each run (#2953). No lane
//! meters fuel.
//!
//! `SPIKE_SECTIONS` picks sections, comma-separated: `compute` and `spawn` (both by default).

use std::time::{Duration, Instant};
use temen_interp::Host;
use temen_ir::Module;
use temen_jit::{CompiledModule, JitOutcome};
use temen_run::jit_kernel;

const RESERVED: u8 = temen_ir::DEFAULT_RESERVED_LOG2;

fn parse(ir: &str) -> Module {
    let m = temen_text::parse_module(ir).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// How a lane runs a program once: `(result, elapsed)`.
type Lane<'a> = Box<dyn FnMut(&[i64]) -> (i64, Duration) + 'a>;

/// Today's JIT over `host`'s powerbox; `locked` is the serialized thunk a concurrent program runs
/// (`jit_cap_run`'s choice), else the raw one.
fn today<'a>(m: &'a Module, entry: u32, host: Host, locked: bool) -> Lane<'a> {
    let cell = Box::leak(Box::new(std::sync::Mutex::new(host)));
    let (thunk, ctx): (temen_jit::CapThunk, *mut core::ffi::c_void) = if locked {
        (temen_run::cap_thunk_locked, cell as *const _ as *mut _)
    } else {
        let h = cell.get_mut().unwrap() as *mut Host;
        (temen_run::cap_thunk, h as *mut _)
    };
    Box::new(move |args| {
        let mut cm = CompiledModule::compile(
            m, entry, thunk, ctx, RESERVED, None, None, None, None, None, 0,
        )
        .expect("jit compile");
        let t0 = Instant::now();
        let (out, _) = cm.run(args, None, None).expect("jit run");
        let dt = t0.elapsed();
        match out {
            JitOutcome::Returned(v) => (v[0], dt),
            o => panic!("today's JIT at {args:?}: {o:?}"),
        }
    })
}

/// The kernel route over a fresh host, unmetered (no fuel checks compiled, as today's lane). Its
/// compile is inside the timed run, so the per-iteration rows subtract it and the run-at-`n` column
/// carries it: [`setup`] reports the compile apart.
fn kernel(m: &Module, entry: u32) -> Lane<'_> {
    Box::new(move |args| {
        let mut h = Host::new();
        let admitted = jit_kernel::admit(m, &h, None, false).expect("in the kernel's scope");
        let start = jit_kernel::Start {
            entry,
            args,
            init_mem: &[],
            reserved_log2: RESERVED,
            fuel: u64::MAX,
            snapshot_cap: None,
            kill: None,
        };
        let t0 = Instant::now();
        let end = jit_kernel::run(admitted, start, &mut h).expect("compiles");
        let dt = t0.elapsed();
        assert!(end.vcpus >= 1, "it ran as kernel tasks");
        (end.result.expect("kernel run")[0], dt)
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
#[allow(clippy::too_many_arguments)]
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

pub fn run(reps: usize) {
    let reps = reps.max(5);
    println!("#2958: today's Cranelift JIT vs the kernel route (vCPUs as executor 2 tasks)");
    println!("  median [min..max] over {reps} interleaved reps; release build\n");
    let only = std::env::var("SPIKE_SECTIONS").unwrap_or_default();
    let on = |s: &str| only.is_empty() || only.split(',').any(|x| x == s);
    if on("compute") {
        // A module with no §12 op keeps today's path (the `no §12 op` clause): these rows are the
        // declined path's own cost, to compare against main's build of this probe.
        println!("compute and synchronous call.cap, declined to today's path (ns per iteration):");
        for k in [&crate::ALU, &crate::MEMSUM] {
            let m = parse(k.ir);
            assert_eq!(
                jit_kernel::admit(&m, &Host::new(), None, false).err(),
                Some("no §12 op")
            );
            let mut lanes = [("today", today(&m, 0, Host::new(), false))];
            let range = (1_000, 20_000_000);
            per_iter(k.name, &mut lanes, |n| vec![n], range, 1.0, reps, 1.0, "ns");
        }
        let k = crate::hostproc_kernel();
        let m = parse(&k.ir);
        let lead = k.lead_args.clone();
        let args = move |n| {
            let mut a = lead.clone();
            a.push(n);
            a
        };
        let mut lanes = [
            ("today", today(&m, 0, crate::dispatch_host().0, false)),
            ("today-lock", today(&m, 0, crate::dispatch_host().0, true)),
        ];
        let range = (1_000, 2_000_000);
        per_iter("hostproc", &mut lanes, args, range, 1.0, reps, 1.0, "ns");
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
            let want = (nw as i64) * (steps as i64);
            let mut lanes = [
                ("today", today(&m, entry, Host::new(), true)),
                ("kernel", kernel(&m, entry)),
            ];
            let mut ms: Vec<Vec<f64>> = vec![Vec::new(); lanes.len()];
            for _ in 0..reps {
                for (i, (_, lane)) in lanes.iter_mut().enumerate() {
                    let (r, dt) = lane(&[]);
                    assert_eq!(r, want, "{label}");
                    ms[i].push(dt.as_secs_f64() * 1e3);
                }
            }
            let (sa, sb) = (stats(&mut ms[0]), stats(&mut ms[1]));
            println!(
                "  {label:<22} today {:>28}   kernel {:>28}   ratio {:.2}",
                fmt(sa, "ms"),
                fmt(sb, "ms"),
                sb.0 / sa.0
            );
        }
    }
}
