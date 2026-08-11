//! **Per-iteration perf for the control-flow residuals** (while / repeat / for), rolled vs interpreter.
//!
//! Same differential-N methodology as `lua_futamura_bench.rs`: the interpreter JIT-compiles the whole
//! ~690-function module every call, so absolute times are compile-dominated; measuring at two trip
//! counts and taking `(T(N₂) − T(N₁)) / (N₂ − N₁)` cancels compile + parse and isolates the per-iteration
//! cost. The rolled residual is N-independent (one module, swept limit/counter), so its differential is a
//! true execution-vs-execution number too. `#[ignore]`d report, no timing assertions.
//!
//! **Reference numbers** (2026-08-11, release, CI-class container; re-run to refresh). Per-iteration,
//! 20M-iteration differential on both sides:
//!
//! | script                              | interpreter   | rolled residual | speedup |
//! |-------------------------------------|---------------|-----------------|---------|
//! | `for i=1,n do x=x+3 end`            | ≈ 18–19 ns    | ≈ 1.6 ns        | ≈ 11–12× |
//! | `while i<=n do s=s+i; i=i+1 end`    | ≈ 27–29 ns    | ≈ 1.9 ns        | ≈ 15×    |
//! | `repeat s=s+i; i=i+1 until i>n`     | ≈ 24–27 ns    | ≈ 1.5–2.0 ns    | ≈ 14–16× |
//!
//! The residual holds ≈ 1.5–2.0 ns/iter regardless of loop form — its dispatch is fully folded. The
//! interpreter costs *more* per iteration for `while`/`repeat` because their bodies carry more bytecode
//! (an `LE` + two `ADD`s + `MMBIN`s + a `JMP`) than the `for` body's single `ADD` + `FORLOOP`, so folding
//! that heavier dispatch yields an even larger speedup. Residual numbers are stable; the interpreter side
//! is noisier on shared runners.
//!
//! Run: `cargo test -p svm-llvm --release --test lua_cf_bench -- --ignored --nocapture`

mod lua_rolled;
mod peval_capture;

use lua_rolled::{auto_rolled, lua_module, while_rolled, with_readback, AutoRolled};
use std::hint::black_box;
use std::time::{Duration, Instant};
use svm_ir::Module;

fn best_dur(reps: usize, mut f: impl FnMut()) -> Duration {
    for _ in 0..2 {
        f();
    } // warmup
    let mut best = Duration::MAX;
    for _ in 0..reps {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed());
    }
    best
}

fn jit_run(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

/// Interpreter per-iteration ns via differential-N: run `mk(n)` (a `print(result)` script) at two trip
/// counts and cancel compile. `mk` must place `n` as the loop bound.
fn interp_ns(m: &Module, reps: usize, mk: &dyn Fn(u64) -> String, n1: u64, n2: u64) -> f64 {
    let run = |n: u64| {
        let s = mk(n);
        move || {
            black_box(svm_run::run_powerbox(m, s.as_bytes()).expect("run").stdout);
        }
    };
    let t1 = best_dur(reps, run(n1));
    let t2 = best_dur(reps, run(n2));
    (t2.saturating_sub(t1)).as_nanos() as f64 / (n2 - n1) as f64
}

/// Residual per-iteration ns via differential over the swept cell (the same rolled module at both trip
/// counts, so compile cancels). `sweep_ix` is the dynamic-cell index that sets the trip count.
fn residual_ns(
    r: &AutoRolled,
    reps: usize,
    sweep_ix: usize,
    n1: u64,
    n2: u64,
    to_cell: impl Fn(u64) -> i64,
) -> f64 {
    let (wm, we) = with_readback(&r.residual, r.entry, r.acc_addr, r.dyn_cells.len());
    svm_verify::verify_module(&wm).expect("verifies");
    let time_at = |n: u64| {
        let mut a = r.captured.clone();
        a[sweep_ix] = to_cell(n);
        best_dur(reps, || {
            black_box(jit_run(&wm, we, &a));
        })
    };
    let t1 = time_at(n1);
    let t2 = time_at(n2);
    (t2.saturating_sub(t1)).as_nanos() as f64 / (n2 - n1) as f64
}

#[test]
#[ignore = "benchmark; run explicitly with --release -- --ignored --nocapture"]
fn control_flow_residual_speedups() {
    let reps = 3;
    let m = lua_module();
    let (n1, n2) = (1_000_000u64, 21_000_000u64);

    // ---- for (reference): `for i=1,n do x = x + 3 end` — counter is the swept cell (counts down). ----
    {
        let script = "local x = 0\nfor i = 1, 8 do x = x + 3 end\nreturn x\n";
        let r = auto_rolled(&m, script);
        let mk = |n: u64| format!("local x = 0\nfor i = 1, {n} do x = x + 3 end\nprint(x)\n");
        let ins = interp_ns(&m, reps, &mk, n1, n2);
        // counter c ⇒ c+1 body executions, so sweep to n-1.
        let res = residual_ns(&r, reps, r.counter_ix, n1, n2, |n| n as i64 - 1);
        println!(
            "for    (x=x+3):   interp {ins:6.1} ns/iter | residual {res:5.2} ns/iter | {:.1}x",
            ins / res
        );
    }

    // ---- while: `while i<=n do s=s+i; i=i+1 end` — limit n is the swept cell (counts up to n). ----
    {
        let script = "local n = 8\nlocal s = 0\nlocal i = 1\nwhile i <= n do s = s + i i = i + 1 end\nreturn s\n";
        let r = while_rolled(&m, script);
        let mk = |n: u64| {
            format!("local n = {n}\nlocal s = 0\nlocal i = 1\nwhile i <= n do s = s + i i = i + 1 end\nprint(s)\n")
        };
        let ins = interp_ns(&m, reps, &mk, n1, n2);
        let res = residual_ns(&r, reps, r.counter_ix, n1, n2, |n| n as i64);
        println!(
            "while  (s=s+i):   interp {ins:6.1} ns/iter | residual {res:5.2} ns/iter | {:.1}x",
            ins / res
        );
    }

    // ---- repeat: `repeat s=s+i; i=i+1 until i>n` — limit n is the swept cell. ----
    {
        let script = "local n = 8\nlocal s = 0\nlocal i = 1\nrepeat s = s + i i = i + 1 until i > n\nreturn s\n";
        let r = while_rolled(&m, script);
        let mk = |n: u64| {
            format!("local n = {n}\nlocal s = 0\nlocal i = 1\nrepeat s = s + i i = i + 1 until i > n\nprint(s)\n")
        };
        let ins = interp_ns(&m, reps, &mk, n1, n2);
        let res = residual_ns(&r, reps, r.counter_ix, n1, n2, |n| n as i64);
        println!(
            "repeat (s=s+i):   interp {ins:6.1} ns/iter | residual {res:5.2} ns/iter | {:.1}x",
            ins / res
        );
    }
}
