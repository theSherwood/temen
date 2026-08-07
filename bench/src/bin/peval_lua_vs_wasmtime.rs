//! **Real Lua: interpreter vs partial-evaluation residual vs native wasm (Wasmtime).**
//!
//! The Futamura arc (`crates/svm-llvm/tests/lua_futamura_*.rs`) measures the peval *residual* of the
//! real `luaV_execute` against the interpreter and reports an ~11.5× per-iteration win — but only
//! *within our own engine*. This bench adds the missing anchor: an **independent native baseline**,
//! the same per-iteration kernel compiled Rust → wasm32 → Wasmtime (Cranelift). Three lanes, one
//! kernel, all timed the same way, so the question the whole arc points at gets a number:
//!
//!   **how much of the interpreted-Lua → native gap does specializing the interpreter actually close?**
//!
//! Kernel: `for i = 1, n do x = x + 3 end` — a deliberately trivial loop body. A trivial body is the
//! *right* microbenchmark here: it maximizes the share of time spent in interpreter decode+dispatch,
//! which is exactly what the residual removes and what native never pays. The native lane adds 3 per
//! iteration behind a `black_box` so the optimizer can't close-form the loop to `3*n`.
//!
//!   - **interpreter**: real `luaV_execute` run through svm-jit on the Lua chunk (whole program,
//!     differential-N over the trip count cancels compile+parse).
//!   - **residual**: the dispatch-folded, loop-rolled specialization of `luaV_execute` for this chunk
//!     (the arc's payoff), run through svm-jit; the module is N-independent so the same differential
//!     cancels its compile exactly.
//!   - **native (wasm32/Wasmtime)**: `run(n){ x=0; for i<n { x += black_box(3) } }` in Rust, built to
//!     wasm32 and run on the same Wasmtime this crate already links.
//!
//! All three: min-over-reps + large/small-N differential (`(t(N₂)−t(N₁))/(N₂−N₁)`), the repo-standard
//! per-iteration methodology (rustbench.rs, embench_one.rs). The native lane skips gracefully if
//! `rustc`/the wasm32 target is unavailable (like rustbench), so the interpreter-vs-residual half
//! always runs.
//!
//! Run: `cargo run --release --manifest-path bench/Cargo.toml --bin peval_lua_vs_wasmtime`
//! CSV:  `SVM_BENCH_CSV=1 cargo run --release … --bin peval_lua_vs_wasmtime`

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use svm_interp::{IrPc, Stop, StopReason, Value, WatchKind};
use svm_ir::{Block, Func, Inst, LoadOp, Module, Terminator, ValType};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use wasmtime::{Config, Engine, Instance, Module as WtModule, Store, Val};

// ---- Lua 5.4.7 layout offsets (identical to lua_futamura_bench.rs — the proven capture). ----
const CAPTURE_LEN: usize = 8 << 20;
const L_STACK_LAST: u64 = 40;
const L_STACK: u64 = 48;
const LUA_STATE_SIZE: usize = 200;
const CI_SIZE: usize = 104;
const CI_FUNC: u64 = 0;
const LCLOSURE_P: u64 = 24;
const PROTO_SIZECODE: u64 = 24;
const PROTO_CODE: u64 = 64;
const SV: u64 = 16;

const ROLL_SCRIPT: &str = "local x = 0\nfor i = 1, 50 do x = x + 3 end\nreturn x\n";

// The differential trip counts (a 20M-iteration delta keeps the per-iteration number stable against
// shared-runner noise; the interpreter side especially).
const N1: u64 = 1_000_000;
const N2: u64 = 21_000_000;
const REPS: usize = 5;

fn rd_u64(w: &[u8], a: u64) -> u64 {
    u64::from_le_bytes(w[a as usize..a as usize + 8].try_into().unwrap())
}
fn rd_i32(w: &[u8], a: u64) -> i32 {
    i32::from_le_bytes(w[a as usize..a as usize + 4].try_into().unwrap())
}

fn lua_module() -> Module {
    let p = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../crates/svm-llvm/tests/fixtures/lua/lua_eval.ll"
    );
    svm_llvm::translate_ll_path(p)
        .expect("translate lua_eval.ll")
        .module
}

fn luav_execute(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "luaV_execute")
        .expect("luaV_execute export")
        .func
}

/// The loop-body safepoint capture: (sp, L, ci, base, counter, window). Watch the FORLOOP counter cell
/// and stop once it holds a small positive VNUMINT while the accumulator is still 0 — a clean mid-loop
/// resume image. (Ported verbatim from `lua_futamura_bench.rs`.)
struct LoopEntry {
    sp: i64,
    l: u64,
    ci: u64,
    base: u64,
    counter: u64,
    window: Vec<u8>,
}

fn capture_loop_entry(m: &Module, luav: u32) -> LoopEntry {
    let inst = svm_run::instantiate(m.clone()).expect("instantiate");
    let mut insp = inst.debug_attach(ROLL_SCRIPT.as_bytes().to_vec(), u64::MAX);
    let entry_pc = IrPc {
        module: 0,
        func: luav,
        block: 0,
        inst: 0,
    };
    insp.set_breakpoint(entry_pc);
    let (sp, l, ci) = match insp.run_until_stop() {
        Stop::Break {
            reason: StopReason::Breakpoint,
            ..
        } => {
            let g = |i| match insp.read_ir_value(0, i) {
                Some(Value::I64(v)) => v,
                o => panic!("{o:?}"),
            };
            (g(0), g(1) as u64, g(2) as u64)
        }
        o => panic!("no entry break: {o:?}"),
    };
    insp.clear_breakpoint(entry_pc);
    let w0 = insp.read_window(0, CAPTURE_LEN).expect("window");
    let func = rd_u64(&w0, ci + CI_FUNC);
    let base = func + SV;
    let counter = base + 3 * SV;
    let acc = base + SV;
    insp.set_watchpoint(counter, 8, WatchKind::Write);
    let window = loop {
        match insp.run_until_stop() {
            Stop::Break {
                reason: StopReason::Watchpoint { write: true, .. },
                ..
            } => {
                insp.step();
                let w = insp.read_window(0, CAPTURE_LEN).expect("window");
                let ctag = w[(counter + 8) as usize];
                let atag = w[(acc + 8) as usize];
                if ctag == 0x03 && atag == 0x03 && rd_u64(&w, acc) == 0 && rd_u64(&w, counter) > 0 {
                    break w;
                }
            }
            o => panic!("did not reach loop body: {o:?}"),
        }
    };
    LoopEntry {
        sp,
        l,
        ci,
        base,
        counter,
        window,
    }
}

/// Build the rolled residual for the captured `x = x + 3` loop: dispatch folds, the loop rolls over a
/// dynamic trip counter. (The SpecConfig recipe from `lua_futamura_bench.rs`.)
fn build_residual(m: &Module, luav: u32, cap: &LoopEntry) -> Module {
    let (l, ci, w) = (cap.l, cap.ci, &cap.window);
    let stack_lo = rd_u64(w, l + L_STACK);
    let stack_hi = rd_u64(w, l + L_STACK_LAST);
    let func = rd_u64(w, ci + CI_FUNC);
    let closure = rd_u64(w, func);
    let proto = rd_u64(w, closure + LCLOSURE_P);
    let code = rd_u64(w, proto + PROTO_CODE);
    let sizecode = rd_i32(w, proto + PROTO_SIZECODE) as usize;
    let slice = |a: u64, n: usize| w[a as usize..a as usize + n].to_vec();
    let cfg = SpecConfig {
        const_overlays: vec![
            (l, slice(l, LUA_STATE_SIZE)),
            (stack_lo, slice(stack_lo, (stack_hi - stack_lo) as usize)),
            (ci, slice(ci, CI_SIZE)),
            (code, slice(code, 4 * sizecode)),
            (closure, slice(closure, 48)),
            (proto, slice(proto, 128)),
        ],
        rename: Some((l, l + LUA_STATE_SIZE as u64)),
        rename_extra: vec![(stack_lo, stack_hi), (ci, ci + CI_SIZE as u64)],
        rename_is_private: true,
        rename_seed_from_image: true,
        dynamic_cells: vec![(cap.base + SV, 8), (cap.base + 2 * SV, 8), (cap.counter, 8)],
        indirect_targets_cap: Some(16),
        ..SpecConfig::default()
    };
    let args = [
        SpecArg::ConstI64(cap.sp),
        SpecArg::ConstI64(l as i64),
        SpecArg::ConstI64(ci as i64),
    ];
    specialize_with_config(m, luav, &args, &cfg).expect("residual rolls")
}

/// Append a `(x0, i0, counter) -> i64` wrapper that calls the rolled residual then loads the
/// accumulator cell it wrote back.
fn with_readback(residual: &Module, read_addr: u64) -> (Module, u32) {
    let mut m = residual.clone();
    let wrapper = m.funcs.len() as u32;
    m.funcs.push(Func {
        params: vec![ValType::I64, ValType::I64, ValType::I64],
        results: vec![ValType::I64],
        blocks: vec![Block {
            params: vec![ValType::I64, ValType::I64, ValType::I64],
            insts: vec![
                Inst::ConstI64(read_addr as i64),
                Inst::Call {
                    func: 0,
                    args: vec![0, 1, 2],
                },
                Inst::Load {
                    op: LoadOp::I64,
                    addr: 3,
                    offset: 0,
                    align: 8,
                },
            ],
            term: Terminator::Return(vec![4]),
        }],
    });
    (m, wrapper)
}

fn jit_run(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

fn pure_loop_script(n: u64) -> String {
    format!("local x = 0\nfor i = 1, {n} do x = x + 3 end\nprint(x)\n")
}

// ---- The native lane: the same per-iteration kernel, Rust → wasm32 → Wasmtime. ----

/// A bare `no_std` cdylib exporting `run(n) -> i64` that adds 3 per iteration behind a `black_box`
/// (so the optimizer can't close-form the loop to `3*n`). Matches the Lua `x = x + 3` body exactly.
const NATIVE_SRC: &str = r#"#![no_std]
#[panic_handler]
fn ph(_: &core::panic::PanicInfo) -> ! { loop {} }
#[no_mangle]
pub extern "C" fn run(n: i64) -> i64 {
    let mut x: i64 = 0;
    let mut i: i64 = 0;
    while i < n {
        x = x.wrapping_add(core::hint::black_box(3));
        i += 1;
    }
    x
}
"#;

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("peval_lua_vs_wt_{name}"))
}

/// `rustc -O --target wasm32-unknown-unknown` → module path; `None` if the toolchain/target is absent.
fn build_wasm32() -> Option<PathBuf> {
    let src = tmp("native.rs");
    std::fs::write(&src, NATIVE_SRC).ok()?;
    let wasm = tmp("native.wasm");
    let rustc = std::env::var("SVM_RUSTBENCH_RUSTC").unwrap_or_else(|_| "rustc".into());
    let ok = Command::new(rustc)
        .args([
            "--edition",
            "2021",
            "-O",
            "-Cpanic=abort",
            "--crate-type=cdylib",
            "--target=wasm32-unknown-unknown",
        ])
        .arg(&src)
        .arg("-o")
        .arg(&wasm)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    (ok && wasm.exists()).then_some(wasm)
}

/// Instantiate the prebuilt wasm on Wasmtime and return a `run(n)` closure. (rustbench's `wt_runner`.)
fn wt_runner(wasm: &Path) -> Option<impl FnMut(i64) -> i64> {
    let engine = Engine::new(&Config::new()).ok()?;
    let module = WtModule::from_file(&engine, wasm).ok()?;
    let mut store = Store::new(&engine, ());
    let inst = Instance::new(&mut store, &module, &[]).ok()?;
    let f = inst.get_func(&mut store, "run")?;
    let mut out = [Val::I64(0)];
    Some(move |n: i64| -> i64 {
        f.call(&mut store, &[Val::I64(n)], &mut out)
            .expect("wt run");
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected wt return"),
        }
    })
}

fn best(mut f: impl FnMut()) -> Duration {
    f(); // warmup
    let mut b = Duration::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        f();
        b = b.min(t.elapsed());
    }
    b
}

/// Per-iteration ns via the large/small-N differential (compile/parse/warmup cancel).
fn per_iter(t1: Duration, t2: Duration) -> f64 {
    (t2.saturating_sub(t1)).as_nanos() as f64 / (N2 - N1) as f64
}

fn main() {
    let csv = std::env::var("SVM_BENCH_CSV").is_ok();
    let m = lua_module();
    let luav = luav_execute(&m);

    // ---- Residual: build once (N-independent). ----
    let cap = capture_loop_entry(&m, luav);
    let (wm, we) = with_readback(&build_residual(&m, luav, &cap), cap.base + SV);
    svm_verify::verify_module(&wm).expect("residual verifies");
    // Body-entered: counter = c ⇒ x = 3·(c+1). Seed counter = n-1 to reproduce n loop iterations.
    let resid_at = |n: u64| jit_run(&wm, we, &[0, 1, n as i64 - 1]);

    // ---- Correctness: all three lanes must agree on 3·n before we trust any timing. ----
    let want = 3 * N1 as i64;
    assert_eq!(resid_at(N1), want, "residual == 3·n");
    let interp_out = |n: u64| -> i64 {
        let out = svm_run::run_powerbox(&m, pure_loop_script(n).as_bytes())
            .expect("interp")
            .stdout;
        String::from_utf8_lossy(&out)
            .trim()
            .parse()
            .expect("int out")
    };
    assert_eq!(interp_out(N1), want, "interpreter == 3·n");

    // ---- Interpreter per-iteration (whole-program differential-N). ----
    let interp_ns = {
        let (s1, s2) = (pure_loop_script(N1), pure_loop_script(N2));
        let t1 = best(|| {
            black_box(
                svm_run::run_powerbox(&m, s1.as_bytes())
                    .expect("run")
                    .stdout,
            );
        });
        let t2 = best(|| {
            black_box(
                svm_run::run_powerbox(&m, s2.as_bytes())
                    .expect("run")
                    .stdout,
            );
        });
        per_iter(t1, t2)
    };

    // ---- Residual per-iteration (same module, only the counter arg changes). ----
    let resid_ns = {
        let (a1, a2) = ([0i64, 1, N1 as i64 - 1], [0i64, 1, N2 as i64 - 1]);
        let t1 = best(|| {
            black_box(jit_run(&wm, we, &a1));
        });
        let t2 = best(|| {
            black_box(jit_run(&wm, we, &a2));
        });
        per_iter(t1, t2)
    };

    // ---- Native per-iteration (Rust → wasm32 → Wasmtime), or a skip note. ----
    let native_ns = build_wasm32().and_then(|w| wt_runner(&w)).map(|mut run| {
        assert_eq!(run(N1 as i64), want, "native == 3·n");
        let t1 = best(|| {
            black_box(run(N1 as i64));
        });
        let t2 = best(|| {
            black_box(run(N2 as i64));
        });
        per_iter(t1, t2)
    });

    // ---- Report. ----
    if csv {
        println!("lane,ns_per_iter");
        println!("interpreter,{interp_ns:.4}");
        println!("residual,{resid_ns:.4}");
        match native_ns {
            Some(n) => println!("native_wasm32,{n:.4}"),
            None => println!("native_wasm32,NA"),
        }
        return;
    }

    println!("\nReal Lua `x = x + 3` per iteration — interpreter vs peval residual vs native wasm");
    println!("(differential-N: N₁={N1}, N₂={N2}, min of {REPS} reps)\n");
    println!("{:<24} {:>14}", "lane", "ns/iter");
    println!("{}", "-".repeat(40));
    println!("{:<24} {:>11.2} ns", "Lua interpreter", interp_ns);
    println!("{:<24} {:>11.2} ns", "Lua peval residual", resid_ns);
    match native_ns {
        Some(n) => println!("{:<24} {:>11.2} ns", "native wasm (Wasmtime)", n),
        None => println!(
            "{:<24} {:>14}",
            "native wasm (Wasmtime)", "skipped (no rustc/wasm32)"
        ),
    }
    println!("\nspeedups:");
    println!("  residual vs interpreter: {:>6.1}x", interp_ns / resid_ns);
    if let Some(n) = native_ns {
        println!("  interpreter vs native:   {:>6.1}x", interp_ns / n);
        println!("  residual  vs native:     {:>6.2}x", resid_ns / n);
        let closed = (interp_ns - resid_ns) / (interp_ns - n) * 100.0;
        println!(
            "\nthe residual closes {closed:.0}% of the interpreted→native gap \
             (interp {interp_ns:.1} → residual {resid_ns:.2} → native {n:.2} ns/iter)"
        );
    }
}
