//! **#2958 — the kernel-route probe** (the `--kernel-spike` mode): today's Cranelift JIT, whose
//! threads run on its own runtime, against the kernel route, where they are tasks of executor 2's
//! scheduling core (`temen_run::jit_kernel`), and, for the park-and-wake sections, executor 2
//! interpreting the same program, for reference. Phase 1's exit condition is read from it (#2958).
//!
//! Every lane is a release build on this machine, and the lanes run interleaved: each rep runs every
//! lane once before the next rep starts. A row reports the median over the reps and the spread
//! (min..max). Per-iteration costs are by subtraction of a small-`n` run from a large-`n` one, so
//! each lane's fixed setup (compiles, the window, the run's open and teardown) cancels; the setup
//! itself is reported apart. Every lane compiles afresh, untimed, before each run (#2953). No lane
//! meters fuel.
//!
//! `SPIKE_SECTIONS` picks sections, comma-separated: `compute`, `spawn`, `park` (the default three),
//! `k1000`, `nowake`, `alias` (#2958 slice 4), `pairs`, `always` (#2985). `SPIKE_K` picks the park section's K values (`0,100,1000` by
//! default), `SPIKE_LANES` its lanes.

use std::time::{Duration, Instant};
use temen_interp::{Host, Value};
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

/// The kernel route over a fresh host `host` makes, unmetered (no fuel checks compiled, as today's
/// lane). Its compile is inside the timed run, so the per-iteration rows subtract it and the
/// run-at-`n` column carries it: [`setup`] reports the compile apart.
fn kernel(m: &Module, entry: u32, host: fn() -> Host) -> Lane<'_> {
    Box::new(move |args| {
        let mut h = host();
        let admitted = jit_kernel::admit(m, &h, false).expect("in the kernel's scope");
        let start = jit_kernel::Start {
            entry,
            args,
            init_mem: &[],
            init: None,
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

/// #2985 — what one run cost the OS's scheduler, for every lane alike: the voluntary context switches
/// of every thread of this process (a thread that blocks and sleeps makes one), the involuntary
/// ones, and executor 2's own count of its parks that blocked ([`temen_interp::bytecode::thread_blocks`]),
/// which the voluntary count is checked against.
#[derive(Clone, Copy, Default)]
struct Sw {
    vol: f64,
    invol: f64,
    blocks: f64,
}

impl Sw {
    /// The counts now. `getrusage(RUSAGE_SELF)` keeps the counts of threads that have exited. Off
    /// unix the switch counts read 0.
    fn now() -> Sw {
        #[cfg(unix)]
        // SAFETY: `getrusage` writes the struct it is handed and reads nothing else.
        let (vol, invol) = unsafe {
            let mut ru = std::mem::zeroed::<libc::rusage>();
            libc::getrusage(libc::RUSAGE_SELF, &mut ru);
            (ru.ru_nvcsw as f64, ru.ru_nivcsw as f64)
        };
        #[cfg(not(unix))]
        let (vol, invol) = (0.0, 0.0);
        Sw {
            vol,
            invol,
            blocks: temen_interp::bytecode::thread_blocks() as f64,
        }
    }

    fn minus(self, o: Sw) -> Sw {
        Sw {
            vol: self.vol - o.vol,
            invol: self.invol - o.invol,
            blocks: self.blocks - o.blocks,
        }
    }

    fn per(self, d: f64) -> Sw {
        Sw {
            vol: self.vol / d,
            invol: self.invol / d,
            blocks: self.blocks / d,
        }
    }
}

/// `SPIKE_RAW=1`: print each rep's value and counts as well, to fit a lane's time against its block
/// rate (#2985).
fn raw() -> bool {
    std::env::var_os("SPIKE_RAW").is_some()
}

/// How many threads this process has (`/proc/self/status`), `None` where there is no such file.
fn threads_now() -> Option<usize> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let l = s.lines().find(|l| l.starts_with("Threads:"))?;
    l[8..].trim().parse().ok()
}

/// Run `lane` on `args` and count what it cost the scheduler ([`Sw`]). A thread's counts join the
/// process's only once the OS has reaped it, which can be after the run that joined it returns, so
/// the count waits (untimed, and bounded) until the threads the run started are gone.
fn counted(lane: &mut Lane<'_>, args: &[i64]) -> ((i64, Duration), Sw) {
    let before = threads_now();
    let s0 = Sw::now();
    let out = lane(args);
    let t0 = Instant::now();
    while threads_now() > before {
        assert!(
            t0.elapsed() < Duration::from_secs(5),
            "the run's threads did not exit: {:?} against {before:?} before it",
            threads_now()
        );
        std::thread::yield_now();
    }
    (out, Sw::now().minus(s0))
}

/// Median and spread of `xs`, as `median [min..max]`.
fn stats(xs: &mut [f64]) -> (f64, f64, f64) {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (xs[xs.len() / 2], xs[0], xs[xs.len() - 1])
}

/// Per-op scheduler counts over the reps, as one row: voluntary switches, involuntary ones, and
/// executor 2's blocks.
fn sw_row(sws: &[Sw]) -> String {
    let col = |f: fn(&Sw) -> f64| {
        let mut xs: Vec<f64> = sws.iter().map(f).collect();
        let (m, lo, hi) = stats(&mut xs);
        format!("{m:.3} [{lo:.3}..{hi:.3}]")
    };
    format!(
        "per op: vol cs {}  invol cs {}  x2 blocks {}",
        col(|s| s.vol),
        col(|s| s.invol),
        col(|s| s.blocks),
    )
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
    let mut sws: Vec<Vec<Sw>> = vec![Vec::new(); lanes.len()];
    for _ in 0..reps {
        for (i, (name, lane)) in lanes.iter_mut().enumerate() {
            let ((r0, t0), s0) = counted(lane, &args(n_small));
            let ((r1, t1), s1) = counted(lane, &args(n_big));
            std::hint::black_box((r0, r1));
            let d = t1.as_secs_f64() - t0.as_secs_f64();
            let ops = (n_big - n_small) as f64 * per;
            per_lane[i].push(d * 1e9 / ops / unit_ns);
            setup[i].push(t0.as_secs_f64() * 1e3);
            let sw = s1.minus(s0).per(ops);
            if raw() {
                let v = d * 1e9 / ops / unit_ns;
                println!(
                    "    raw {label} {name} {v:.3} vol {:.3} blocks {:.3}",
                    sw.vol, sw.blocks
                );
            }
            sws[i].push(sw);
        }
    }
    for (i, (name, _)) in lanes.iter().enumerate() {
        println!(
            "  {label:<22} {name:<10} {:>36}   run at n={n_small}: {}",
            fmt(stats(&mut per_lane[i]), unit),
            fmt(stats(&mut setup[i]), "ms"),
        );
        println!("  {:<33} {}", "", sw_row(&sws[i]));
    }
}

pub fn run(reps: usize) {
    let reps = reps.max(5);
    println!("#2958: today's Cranelift JIT vs the kernel route (vCPUs as executor 2 tasks)");
    println!("  median [min..max] over {reps} interleaved reps; release build\n");
    let only = std::env::var("SPIKE_SECTIONS").unwrap_or_default();
    let on = |s: &str| {
        if only.is_empty() {
            ["compute", "spawn", "park"].contains(&s)
        } else {
            only.split(',').any(|x| x == s)
        }
    };
    if on("compute") {
        // A module with no §12 op keeps today's path (the `no §12 op` clause): these rows are the
        // declined path's own cost, to compare against main's build of this probe.
        println!("compute and synchronous call.cap, declined to today's path (ns per iteration):");
        for k in [&crate::ALU, &crate::MEMSUM] {
            let m = parse(k.ir);
            assert_eq!(
                jit_kernel::admit(&m, &Host::new(), false).err(),
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
        per_iter(
            "hostproc",
            &mut lanes,
            args.clone(),
            range,
            1.0,
            reps,
            1.0,
            "ns",
        );
        // #2958 slice 4 — the same procedure on the kernel route: a twin with one §12 op before the
        // loop, which the kernel admits, called inline and through a `call.import` bound to it.
        println!("\nsynchronous call.cap on the kernel route (ns per call):");
        let notified = k.ir.replacen(
            "block 0 (v0: i32, v1: i64) {\n",
            "block 0 (v0: i32, v1: i64) {\n  vq = i64.const 16400\n  vqo = i32.const 1\n  vqk = atomic.notify vq vqo\n",
            1,
        );
        let imported = format!(
            "import 0 \"inc\" (i64) -> (i64)\n{}",
            notified.replacen(
                "call.cap 13 0 (i64) -> (i64) v9(v12)",
                "call.import 0 (v12)",
                1
            )
        );
        fn import_host() -> Host {
            let (mut h, handle) = crate::dispatch_host();
            h.set_import_bindings(vec![temen_interp::BoundImport::required(
                temen_ir::cap_id::HOST_PROC,
                0,
                handle,
            )]);
            h
        }
        for (name, ir, host) in [
            (
                "hostproc",
                &notified,
                (|| crate::dispatch_host().0) as fn() -> Host,
            ),
            ("hostimport", &imported, import_host),
        ] {
            let m = parse(ir);
            let mut lanes = [
                ("today-lock", today(&m, 0, host(), true)),
                ("kernel", kernel(&m, 0, host)),
            ];
            per_iter(name, &mut lanes, args.clone(), range, 1.0, reps, 1.0, "ns");
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
            let want = (nw as i64) * (steps as i64);
            let mut lanes = [
                ("today", today(&m, entry, Host::new(), true)),
                ("kernel", kernel(&m, entry, Host::new)),
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
    if on("park") {
        run_park(reps);
    }
    if on("k1000") {
        run_k1000(reps);
    }
    if on("nowake") {
        run_nowake(reps);
    }
    #[cfg(any(unix, windows))]
    if on("alias") {
        run_nowake_alias(reps);
    }
    if on("pairs") {
        run_pairs(reps);
    }
    if on("always") {
        run_always(reps);
    }
}

/// #2985 — the always-block hand-off ([`crate::kernel_spike_ir::always_ir`]): the pingpong with a
/// turn of work before each notify, long enough that the waiter is asleep before every notify comes.
/// The zero-delay pingpong is a race between a notify and its waiter reaching its sleep (#2963), so
/// its cost is the lane's block rate as much as its path; here every lane blocks on every hand-off
/// (the `vol cs` and `x2 blocks` columns show whether it did), and what is left once the work is
/// taken out is the path: the stops, the park and the wake.
///
/// Each lane's work count is calibrated to `SPIKE_WORK_US` (`20,100` by default) from the work alone
/// on that lane, and the work alone ([`crate::kernel_spike_ir::always_work_ir`]) is measured again
/// in every rep, interleaved with the pingpong, and taken out.
fn run_always(reps: usize) {
    println!(
        "\nalways block: the pingpong with a turn of work before each notify (us per hand-off):"
    );
    let pp = parse(&crate::kernel_spike_ir::always_ir());
    let alone = parse(&crate::kernel_spike_ir::always_work_ir());
    let targets: Vec<f64> = std::env::var("SPIKE_WORK_US")
        .map(|v| {
            v.split(',')
                .map(|x| x.parse().expect("SPIKE_WORK_US"))
                .collect()
        })
        .unwrap_or_else(|_| vec![20.0, 100.0]);
    let pick = std::env::var("SPIKE_LANES").unwrap_or_default();
    let lanes = |m| {
        let all: Vec<(&str, Lane<'_>)> = vec![
            ("today", today(m, 0, Host::new(), true)),
            ("kernel", kernel(m, 0, Host::new)),
            ("x2-interp", x2_interp(m)),
        ];
        (all.into_iter())
            .filter(|(n, _)| pick.is_empty() || pick.split(',').any(|x| x == *n))
            .collect::<Vec<_>>()
    };
    let (mut pps, mut alones) = (lanes(&pp), lanes(&alone));
    // ns per step of work on each lane: the fastest of a few runs.
    let calib = 20_000i64;
    let steps: Vec<f64> = (alones.iter_mut())
        .map(|(_, lane)| {
            (0..3)
                .map(|_| {
                    let (_, a) = lane(&[10, calib]);
                    let (_, b) = lane(&[110, calib]);
                    (b.as_secs_f64() - a.as_secs_f64()) * 1e9 / (100.0 * calib as f64)
                })
                .fold(f64::INFINITY, f64::min)
        })
        .collect();
    let (n_small, n_big) = (20i64, 520i64);
    let turns = (n_big - n_small) as f64;
    for target in targets {
        // Both native lanes run the same Cranelift code for the work, so they take the same count
        // (today's), and their hand-offs compare rep by rep with no work taken out.
        let native = (pps.iter().position(|(n, _)| *n == "today")).map(|t| steps[t]);
        let ws: Vec<i64> = (pps.iter().zip(&steps))
            .map(|((n, _), s)| {
                let s = if *n == "x2-interp" {
                    *s
                } else {
                    native.unwrap_or(*s)
                };
                ((target * 1e3 / s) as i64).max(1)
            })
            .collect();
        println!("  work {target} us a turn:");
        let k = pps.len();
        let (mut hand, mut work, mut left) = (vec![vec![]; k], vec![vec![]; k], vec![vec![]; k]);
        let mut sws: Vec<Vec<Sw>> = vec![Vec::new(); k];
        for _ in 0..reps {
            for i in 0..k {
                let w = ws[i];
                let ((_, a0), _) = counted(&mut alones[i].1, &[n_small, w]);
                let ((_, a1), _) = counted(&mut alones[i].1, &[n_big, w]);
                let ((_, p0), s0) = counted(&mut pps[i].1, &[n_small, w]);
                let ((_, p1), s1) = counted(&mut pps[i].1, &[n_big, w]);
                // The pingpong's root and partner each take `n` turns: 2n hand-offs, 2n turns of work
                // in series.
                let h = (p1.as_secs_f64() - p0.as_secs_f64()) * 1e6 / (2.0 * turns);
                let wk = (a1.as_secs_f64() - a0.as_secs_f64()) * 1e6 / turns;
                hand[i].push(h);
                work[i].push(wk);
                left[i].push(h - wk);
                let sw = s1.minus(s0).per(2.0 * turns);
                if raw() {
                    let name = pps[i].0;
                    println!(
                        "    raw always{target} {name} {h:.3} work {wk:.3} vol {:.3}",
                        sw.vol
                    );
                }
                sws[i].push(sw);
            }
        }
        // Paired by rep, before any column is sorted: each lane's hand-off less today's, over the
        // same work count.
        let t = pps.iter().position(|(n, _)| *n == "today");
        let vs_today: Vec<Option<String>> = (0..k)
            .map(|i| {
                t.filter(|&t| t != i && ws[t] == ws[i]).map(|t| {
                    let mut d: Vec<f64> =
                        (hand[i].iter().zip(&hand[t])).map(|(a, b)| a - b).collect();
                    format!("  vs today {}", fmt(stats(&mut d), "us"))
                })
            })
            .collect();
        for i in 0..k {
            println!(
                "    {:<10} steps {:>8}  hand-off {:>26}  work {:>26}  left {:>26}",
                pps[i].0,
                ws[i],
                fmt(stats(&mut hand[i]), "us"),
                fmt(stats(&mut work[i]), "us"),
                fmt(stats(&mut left[i]), "us"),
            );
            if let Some(d) = &vs_today[i] {
                println!("    {:<10} {d}", "");
            }
            println!("    {:<10} {}", "", sw_row(&sws[i]));
        }
    }
}

/// The #2868 probe: a pingpong, two vCPUs handing a turn word back and forth through
/// `memory.wait`/`memory.notify`, beside `k` threads parked for good on a word nothing notifies.
/// `func 0 (n, k)`: spawn the `k` idlers, wait until each has counted itself in, spawn the partner,
/// then take `n` turns and join it. Each turn is two hand-offs.
pub fn pingpong_ir() -> &'static str {
    crate::kernel_spike_ir::PINGPONG_IR
}

/// The park-and-wake section: the pingpong beside K idle parked threads, per hand-off. The short
/// span's K = 1,000 row is not usable for today's lane (its fixed cost of spawning 1,000 threads
/// swamps the span): read K = 1,000 from [`run_k1000`].
fn run_park(reps: usize) {
    println!("\npark and wake: #2868 pingpong beside K idle parked threads (us per hand-off):");
    let m = parse(pingpong_ir());
    let ks: Vec<i64> = std::env::var("SPIKE_K")
        .map(|v| v.split(',').map(|x| x.parse().expect("SPIKE_K")).collect())
        .unwrap_or_else(|_| vec![0, 100, 1_000]);
    let pick = std::env::var("SPIKE_LANES").unwrap_or_default();
    for k in ks {
        let lanes = vec![
            ("today", today(&m, 0, Host::new(), true)),
            ("kernel", kernel(&m, 0, Host::new)),
            ("x2-interp", x2_interp(&m)),
        ];
        let mut lanes: Vec<_> = lanes
            .into_iter()
            .filter(|(n, _)| pick.is_empty() || pick.split(',').any(|x| x == *n))
            .collect();
        let span = if k >= 1_000 {
            (100, 2_100)
        } else {
            (100, 5_100)
        };
        let args = move |n| vec![n, k];
        per_iter(
            &format!("K = {k}"),
            &mut lanes,
            args,
            span,
            2.0,
            reps,
            1e3,
            "us",
        );
    }
}

/// K = 1,000 of the pingpong with a span long enough that today's run, whose fixed cost is spawning
/// the 1,000 threads (about 0.6 s, give or take 0.2 s), still subtracts to a usable range.
fn run_k1000(reps: usize) {
    println!("\npark and wake, K = 1000, long span (us per hand-off):");
    let m = parse(pingpong_ir());
    let mut lanes = [
        ("today", today(&m, 0, Host::new(), true)),
        ("kernel", kernel(&m, 0, Host::new)),
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
}

/// The no-wake probe ([`crate::kernel_spike_ir::NOWAKE_IR`]): ns per `memory.notify` with no
/// waiter and per `memory.wait` that answers not-equal at once, beside `k` threads parked for good.
/// No thread wakes, so the OS's wake cost is out of it: what is left is the runtime's own per-op
/// cost, which a futex-based guest mutex pays on each contended unlock.
fn run_nowake(reps: usize) {
    println!("\nno wake: a notify with no waiter / a wait that answers not-equal (ns per op):");
    let m = parse(crate::kernel_spike_ir::NOWAKE_IR);
    for (mode, what) in [(0i64, "notify, none"), (1, "wait, not-equal")] {
        for k in [0i64, 100, 1_000] {
            let args = move |n| vec![n, k, mode];
            // A span long enough that each lane's fixed cost (spawning the K threads, about 0.6 s
            // at K = 1,000 on today's lane) subtracts to a usable range.
            let long = if k >= 1_000 { 20_001_000 } else { 2_001_000 };
            // Executor 2 interpreting the program too: its interpreted futex ops take the same
            // fast paths as the kernel's native ones (#2958 slice 2).
            let mut lanes = [
                ("today", today(&m, 0, Host::new(), true)),
                ("kernel", kernel(&m, 0, Host::new)),
                ("x2-interp", x2_interp(&m)),
            ];
            let label = format!("{what}, K = {k}");
            per_iter(
                &label,
                &mut lanes,
                args,
                (1_000, long),
                1.0,
                reps,
                1.0,
                "ns",
            );
        }
    }
}

/// #2958 slice 4 — the no-wake costs in a window that holds a §13 alias: [`run_nowake`]'s program,
/// which first maps one granule of a region into its reservation (past the words it uses). A
/// notify's key then reads the page map once the window was ever aliased (`CapPages::aliased`).
#[cfg(any(unix, windows))]
fn run_nowake_alias(reps: usize) {
    println!("\nno wake, a §13 alias in the window (ns per op):");
    fn host() -> Host {
        let mut h = Host::new();
        let len = temen_interp::host_region_granularity() as usize;
        h.grant_shared_region_backed(temen_run::new_shared_region(len).expect("a region"));
        h
    }
    let handle = {
        let mut h = Host::new();
        let len = temen_interp::host_region_granularity() as usize;
        h.grant_shared_region_backed(temen_run::new_shared_region(len).expect("a region"))
    };
    let gran = temen_interp::host_region_granularity();
    let ir = crate::kernel_spike_ir::NOWAKE_IR.replacen(
        "block 0 (vn: i64, vk: i64, vm: i64) {\n",
        &format!(
            "block 0 (vn: i64, vk: i64, vm: i64) {{\n  vrc = i32.const {handle}\n  vra = i64.const 1048576\n  vro = i64.const 0\n  vrl = i64.const {gran}\n  vrp = i64.const 3\n  vrm = call.cap 4 0 (i64, i64, i64, i64) -> (i64) vrc(vra, vro, vrl, vrp)\n"
        ),
        1,
    );
    let m = parse(&ir);
    for (mode, what) in [(0i64, "notify, none"), (1, "wait, not-equal")] {
        for k in [0i64, 100] {
            let args = move |n| vec![n, k, mode];
            let mut lanes = [
                ("today", today(&m, 0, host(), true)),
                ("kernel", kernel(&m, 0, host)),
            ];
            let label = format!("{what}, K = {k}");
            per_iter(
                &label,
                &mut lanes,
                args,
                (1_000, 2_001_000),
                1.0,
                reps,
                1.0,
                "ns",
            );
        }
    }
}

/// The run-lock probe ([`crate::kernel_spike_ir::PAIRS_IR`]): `p` independent pingpong pairs, each
/// on a turn word of its own a cache line apart, so nothing but the runtime's own locks couples
/// them. Per pair, a hand-off costs what it does alone if the runtime runs unrelated hand-offs in
/// parallel, and up to `p` times that if one lock serializes them.
fn run_pairs(reps: usize) {
    println!("\nrun lock: P independent pingpong pairs at once (us per hand-off, per pair):");
    let m = parse(crate::kernel_spike_ir::PAIRS_IR);
    for p in [1i64, 2, 4] {
        let mut lanes = [
            ("today", today(&m, 0, Host::new(), true)),
            ("kernel", kernel(&m, 0, Host::new)),
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
}
