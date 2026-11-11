//! #2948 spike — cold start and memory of the spike path against today's Cranelift JIT. A binary of
//! its own, because it counts every allocation (a global allocator), which `--kernel-spike`'s timings
//! must not pay. Median [min..max] over interleaved reps; release build. Scaffolding, deleted with
//! the spike.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use temen_interp::Host;
use temen_jit::CompiledModule;

#[allow(dead_code)] // its pingpong is `--kernel-spike`'s
#[path = "../kernel_spike_ir.rs"]
mod kernel_spike_ir;

/// Counts the bytes it hands out.
struct Counting;
static BYTES: AtomicU64 = AtomicU64::new(0);
// SAFETY: forwards to `System`, adding a count.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        BYTES.fetch_add(l.size() as u64, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
        BYTES.fetch_add(n.saturating_sub(l.size()) as u64, Ordering::Relaxed);
        System.realloc(p, l, n)
    }
}
#[global_allocator]
static A: Counting = Counting;

const RESERVED: u8 = temen_ir::DEFAULT_RESERVED_LOG2;

/// `(µs, bytes allocated)` of `f`.
fn measure<T>(f: impl FnOnce() -> T) -> (f64, u64, T) {
    let b0 = BYTES.load(Ordering::Relaxed);
    let t0 = Instant::now();
    let r = f();
    let dt = t0.elapsed().as_secs_f64() * 1e6;
    (dt, BYTES.load(Ordering::Relaxed) - b0, r)
}

fn stats(mut xs: Vec<f64>) -> String {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "{:.1} [{:.1}..{:.1}]",
        xs[xs.len() / 2],
        xs[0],
        xs[xs.len() - 1]
    )
}

fn jit(m: &temen_ir::Module) -> CompiledModule {
    let cell = Box::leak(Box::new(std::sync::Mutex::new(Host::new())));
    CompiledModule::compile(
        m,
        0,
        temen_run::cap_thunk_locked,
        cell as *const _ as *mut _,
        RESERVED,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("compile")
}

fn main() {
    let reps = 21;
    let m = temen_text::parse_module(kernel_spike_ir::PAIRS_IR).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    // One pair taking one turn each, and 32 pairs: the spawns' share by subtraction.
    let (small, big) = ([1i64, 1], [1i64, 32]);
    let mut rows: Vec<(&str, Vec<f64>, Vec<f64>)> = Vec::new();
    let mut row = |name: &'static str, t: f64, b: u64| {
        if let Some(r) = rows.iter_mut().find(|r| r.0 == name) {
            r.1.push(t);
            r.2.push(b as f64);
        } else {
            rows.push((name, vec![t], vec![b as f64]));
        }
    };
    for _ in 0..reps {
        let (t, b, _) = measure(|| jit(&m));
        row("JIT compile (both lanes)", t, b);
        let (t, b, _) = measure(|| {
            temen_interp::bytecode::compile_module(&m.funcs, &m.types, None).expect("bytecode")
        });
        row("bytecode compile (spike only)", t, b);
        for (label, args) in [("1 pair", small), ("32 pairs", big)] {
            let mut cm = jit(&m);
            let (t, b, _) = measure(|| cm.run(&args, None, None).expect("run"));
            row(
                if label == "1 pair" {
                    "today: run, 1 pair"
                } else {
                    "today: run, 32 pairs"
                },
                t,
                b,
            );
            let c = temen_run::jit_kernel::compile(&m, 0, RESERVED, None).expect("scope");
            let mut h = Host::new();
            let (t, b, r) = measure(|| {
                temen_run::jit_kernel::run_compiled(c, &m, 0, &args, &mut h, None).expect("scope")
            });
            r.expect("spike run");
            row(
                if label == "1 pair" {
                    "spike: run, 1 pair"
                } else {
                    "spike: run, 32 pairs"
                },
                t,
                b,
            );
        }
    }
    assert_eq!(
        temen_interp::bytecode::native_interp_stops(),
        0,
        "all native"
    );
    println!("#2948 spike: cold start and allocation, median [min..max] over {reps} reps");
    println!("  {:<32} {:>26} {:>30}", "what", "µs", "bytes allocated");
    for (name, t, b) in &rows {
        println!(
            "  {name:<32} {:>26} {:>30}",
            stats(t.clone()),
            stats(b.clone())
        );
    }
    let per = |a: &str, b: &str| {
        let get = |n: &str| {
            let mut v = rows.iter().find(|r| r.0 == n).unwrap().2.clone();
            v.sort_by(|x, y| x.partial_cmp(y).unwrap());
            v[v.len() / 2]
        };
        (get(b) - get(a)) / 62.0
    };
    println!(
        "  bytes allocated per spawned vCPU (32 pairs - 1 pair, / 62): today {:.0}, spike {:.0}",
        per("today: run, 1 pair", "today: run, 32 pairs"),
        per("spike: run, 1 pair", "spike: run, 32 pairs")
    );
}
