//! **Real Lua: interpreter vs partial-evaluation residual vs native wasm (Wasmtime).**
//!
//! The Futamura arc (`crates/svm-llvm/tests/lua_futamura_*.rs`) measures the peval *residual* of the
//! real `luaV_execute` against the interpreter and reports an ~11–19× per-iteration win — but only
//! *within our own engine*. This bench adds the missing anchor: an **independent native baseline**,
//! the same per-iteration kernel compiled Rust → wasm → Wasmtime (Cranelift). Three lanes, a gallery
//! of kernels, all timed the same way, so the question the whole arc points at gets a number:
//!
//!   **how much of the interpreted-Lua → native gap does specializing the interpreter actually close?**
//!
//! Kernels are real numeric loops whose body uses the loop variable and does genuine arithmetic
//! (`s+i`, `s+i*i`, a fibonacci recurrence, `i%d` / `i//d` — the divisor a local, so a register
//! operand). Each lane:
//!   - **interpreter**: real `luaV_execute` through svm-jit on the Lua chunk (whole program).
//!   - **residual**: the dispatch-folded, loop-rolled specialization built by the shared
//!     `futamura::auto_rolled` driver (the arc's payoff), through svm-jit.
//!   - **native**: the same per-iteration kernel in Rust, built to wasm32 and run on the same Wasmtime
//!     this crate already links. All kernels live in one wasm module (built once) and the body reads
//!     its inputs through `black_box`, so the optimizer can't close-form the loop.
//!
//! All three: min-over-reps + large/small-N differential (`(t(N₂)−t(N₁))/(N₂−N₁)`), the repo-standard
//! per-iteration methodology (rustbench.rs, embench_one.rs). Correctness (residual == interpreter ==
//! native at a small sample trip count) is asserted before any timing is trusted. The native lane
//! skips gracefully if `rustc`/the wasm32 target is unavailable, so the interpreter-vs-residual half
//! always runs.
//!
//! Run: `cargo run --release --manifest-path bench/Cargo.toml --bin peval_lua_vs_wasmtime`
//! CSV:  `SVM_BENCH_CSV=1 cargo run --release … --bin peval_lua_vs_wasmtime`

use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use svm_bench_vs_wasmtime::futamura::{auto_rolled, has_br_table, lua_module, with_readback};
use svm_ir::Module;
use wasmtime::{Config, Engine, Instance, Module as WtModule, Store, Val};

// Differential trip counts (a 20M-iteration delta keeps the per-iteration number stable against
// shared-runner noise), a small sample count for the correctness cross-check, and the rep count.
const N1: u64 = 1_000_000;
const N2: u64 = 21_000_000;
const SAMPLE: u64 = 50;
const REPS: usize = 5;

/// A gallery kernel: the Lua chunk parts (`decls`, `body`, `ret`) and the matching native Rust — the
/// per-iteration statement(s), the accumulator init, and the returned expression. The native loop is
/// `for i in 1..=n`, mirroring Lua's `for i = 1, n`.
struct Kernel {
    name: &'static str,
    decls: &'static str,
    body: &'static str,
    ret: &'static str,
    native_init: &'static str,
    native_body: &'static str,
    native_ret: &'static str,
}

fn kernels() -> Vec<Kernel> {
    vec![
        Kernel {
            name: "add const (x+3)",
            decls: "local x = 0",
            body: "x = x + 3",
            ret: "x",
            native_init: "let mut x: i64 = 0;",
            native_body: "x = x.wrapping_add(bb(3));",
            native_ret: "x",
        },
        Kernel {
            name: "sum (s+i)",
            decls: "local s = 0",
            body: "s = s + i",
            ret: "s",
            native_init: "let mut s: i64 = 0;",
            native_body: "s = s.wrapping_add(bb(i));",
            native_ret: "s",
        },
        Kernel {
            name: "sum squares (s+i*i)",
            decls: "local s = 0",
            body: "s = s + i * i",
            ret: "s",
            native_init: "let mut s: i64 = 0;",
            native_body: "s = s.wrapping_add(bb(i).wrapping_mul(i));",
            native_ret: "s",
        },
        Kernel {
            name: "fibonacci (a,b=b,a+b)",
            decls: "local a = 0\nlocal b = 1",
            body: "local t = a + b a = b b = t",
            ret: "b",
            native_init: "let mut a: i64 = 0; let mut b: i64 = 1;",
            native_body: "let t = a.wrapping_add(bb(b)); a = b; b = t;",
            native_ret: "b",
        },
        Kernel {
            name: "modulo (s + i%d)",
            decls: "local d = 2\nlocal s = 0",
            body: "s = s + i % d",
            ret: "s",
            native_init: "let mut s: i64 = 0; let d: i64 = bb(2);",
            native_body: "s = s.wrapping_add(i % d);",
            native_ret: "s",
        },
        Kernel {
            name: "floordiv (s + i//d)",
            decls: "local d = 3\nlocal s = 0",
            body: "s = s + i // d",
            ret: "s",
            native_init: "let mut s: i64 = 0; let d: i64 = bb(3);",
            native_body: "s = s.wrapping_add(i / d);", // i,d>0 ⇒ Rust trunc-div == Lua floor-div
            native_ret: "s",
        },
    ]
}

fn return_script(k: &Kernel, n: u64) -> String {
    format!(
        "{}\nfor i = 1, {n} do {} end\nreturn {}\n",
        k.decls, k.body, k.ret
    )
}
fn print_script(k: &Kernel, n: u64) -> String {
    format!(
        "{}\nfor i = 1, {n} do {} end\nprint({})\n",
        k.decls, k.body, k.ret
    )
}

fn jit_run(m: &Module, e: u32, a: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, a) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}

fn interp_out(m: &Module, k: &Kernel, n: u64) -> i64 {
    let out = svm_run::run_powerbox(m, print_script(k, n).as_bytes())
        .expect("interp")
        .stdout;
    String::from_utf8_lossy(&out)
        .trim()
        .parse()
        .expect("int out")
}

// ---- The native lane: all kernels in one wasm module, built once, run on Wasmtime. ----

/// One `no_std` wasm module exporting `run_0 … run_{K-1}`, each `run_k(n) = for i in 1..=n { body }`.
/// The body reads its inputs through `bb` (`core::hint::black_box`) so the optimizer can't close-form
/// the loop to a formula.
fn native_source(ks: &[Kernel]) -> String {
    let mut s = String::from(
        "#![no_std]\nuse core::hint::black_box as bb;\n\
         #[panic_handler]\nfn ph(_: &core::panic::PanicInfo) -> ! { loop {} }\n",
    );
    for (i, k) in ks.iter().enumerate() {
        s.push_str(&format!(
            "#[no_mangle]\npub extern \"C\" fn run_{i}(n: i64) -> i64 {{\n    \
             {}\n    let mut i: i64 = 1;\n    while i <= n {{\n        {}\n        i += 1;\n    }}\n    \
             {}\n}}\n",
            k.native_init, k.native_body, k.native_ret
        ));
    }
    s
}

fn tmp(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("peval_lua_vs_wt_{name}"))
}

/// `rustc -O --target wasm32-unknown-unknown` on the combined source → module path; `None` if the
/// toolchain/target is absent. wasm32 (ILP32) is the fast default that needs no nightly.
fn build_wasm32(src_text: &str) -> Option<PathBuf> {
    let src = tmp("native.rs");
    std::fs::write(&src, src_text).ok()?;
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

/// `cargo +nightly build -Z build-std=core --target wasm64-unknown-unknown` in a throwaway project
/// (wasm64 is tier-3, so it needs build-std). The LP64-matched native baseline DESIGN.md §1a cares
/// about; opt in with `SVM_BENCH_W64=1`. `None` if nightly / rust-src / the target is absent.
fn build_wasm64(src_text: &str) -> Option<PathBuf> {
    let dir = tmp("w64proj");
    std::fs::create_dir_all(dir.join("src")).ok()?;
    std::fs::write(
        dir.join("Cargo.toml"),
        "[package]\nname = \"peval_lua_w64\"\nversion = \"0.0.0\"\nedition = \"2021\"\n\
         [lib]\ncrate-type = [\"cdylib\"]\n[profile.release]\npanic = \"abort\"\n",
    )
    .ok()?;
    std::fs::write(dir.join("src/lib.rs"), src_text).ok()?;
    let ok = Command::new("cargo")
        .args([
            "+nightly",
            "build",
            "-Zbuild-std=core",
            "--target=wasm64-unknown-unknown",
            "--release",
        ])
        .current_dir(&dir)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    let wasm = dir.join("target/wasm64-unknown-unknown/release/peval_lua_w64.wasm");
    (ok && wasm.exists()).then_some(wasm)
}

/// A reusable Wasmtime instance over the prebuilt module; `run(k, n)` calls `run_k(n)`.
struct Native {
    store: Store<()>,
    inst: Instance,
}
impl Native {
    fn load(wasm: &Path, w64: bool) -> Option<Native> {
        let mut cfg = Config::new();
        cfg.wasm_memory64(w64);
        let engine = Engine::new(&cfg).ok()?;
        let module = WtModule::from_file(&engine, wasm).ok()?;
        let mut store = Store::new(&engine, ());
        let inst = Instance::new(&mut store, &module, &[]).ok()?;
        Some(Native { store, inst })
    }
    fn run(&mut self, k: usize, n: i64) -> i64 {
        let f = self
            .inst
            .get_func(&mut self.store, &format!("run_{k}"))
            .expect("export run_k");
        let mut out = [Val::I64(0)];
        f.call(&mut self.store, &[Val::I64(n)], &mut out)
            .expect("wt run");
        match out[0] {
            Val::I64(x) => x,
            Val::I32(x) => x as i64,
            _ => panic!("unexpected wt return"),
        }
    }
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
    let w64 = std::env::var("SVM_BENCH_W64").is_ok();
    let m = lua_module();
    let ks = kernels();

    // Native module: build all kernels once (or note the skip). wasm32 by default; wasm64 (the
    // LP64-matched §1a baseline, needs nightly + build-std) when SVM_BENCH_W64=1.
    let src = native_source(&ks);
    let native_kind = if w64 { "wasm64" } else { "wasm32" };
    let mut native = if w64 {
        build_wasm64(&src)
    } else {
        build_wasm32(&src)
    }
    .and_then(|w| Native::load(&w, w64));

    if !csv {
        println!(
            "\nReal Lua per iteration — interpreter vs peval residual vs native wasm (Wasmtime)"
        );
        println!("(differential-N: N₁={N1}, N₂={N2}, min of {REPS} reps)\n");
        println!(
            "{:<22} {:>13} {:>13} {:>13} {:>9}",
            "kernel", "interp ns", "residual ns", "native ns", "resid/nat"
        );
        println!("{}", "-".repeat(76));
    } else {
        println!("kernel,interp_ns,residual_ns,native_ns");
    }

    for (ki, k) in ks.iter().enumerate() {
        // ---- Residual: build once (N-independent), verify, correctness-check at SAMPLE. ----
        let ar = auto_rolled(&m, &return_script(k, SAMPLE));
        assert!(
            !has_br_table(&ar.residual.funcs[ar.entry as usize]),
            "{}: dispatch folds",
            k.name
        );
        let np = ar.dyn_cells.len();
        let (wm, we) = with_readback(&ar.residual, ar.entry, ar.acc_addr, np);
        svm_verify::verify_module(&wm).expect("residual verifies");

        let want = interp_out(&m, k, SAMPLE);
        assert_eq!(
            jit_run(&wm, we, &ar.captured),
            want,
            "{}: residual == interp",
            k.name
        );
        if let Some(nat) = native.as_mut() {
            assert_eq!(
                nat.run(ki, SAMPLE as i64),
                want,
                "{}: native == interp",
                k.name
            );
        }

        // ---- Interpreter per-iteration (whole-program differential-N). ----
        let (s1, s2) = (print_script(k, N1), print_script(k, N2));
        let interp_ns = {
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
        let seed_at = |c: u64| {
            let mut a = ar.captured.clone();
            a[ar.counter_ix] = c as i64;
            a
        };
        let resid_ns = {
            let (a1, a2) = (seed_at(N1), seed_at(N2));
            let t1 = best(|| {
                black_box(jit_run(&wm, we, &a1));
            });
            let t2 = best(|| {
                black_box(jit_run(&wm, we, &a2));
            });
            per_iter(t1, t2)
        };

        // ---- Native per-iteration (wasm32/Wasmtime), if available. ----
        let native_ns = native.as_mut().map(|nat| {
            let t1 = best(|| {
                black_box(nat.run(ki, N1 as i64));
            });
            let t2 = best(|| {
                black_box(nat.run(ki, N2 as i64));
            });
            per_iter(t1, t2)
        });

        if csv {
            match native_ns {
                Some(n) => println!("{},{interp_ns:.4},{resid_ns:.4},{n:.4}", k.name),
                None => println!("{},{interp_ns:.4},{resid_ns:.4},NA", k.name),
            }
            continue;
        }
        match native_ns {
            Some(n) => println!(
                "{:<22} {:>10.1} ns {:>10.2} ns {:>10.2} ns {:>8.2}x",
                k.name,
                interp_ns,
                resid_ns,
                n,
                resid_ns / n
            ),
            None => println!(
                "{:<22} {:>10.1} ns {:>10.2} ns {:>13} {:>9}",
                k.name, interp_ns, resid_ns, "skipped", "-"
            ),
        }
    }

    if !csv {
        if native.is_some() {
            println!(
                "\n(interp = luaV_execute via svm-jit; residual = folded auto_rolled via svm-jit; \
                 native = Rust→{native_kind}→Wasmtime.\n resid/nat = the residual's remaining gap to \
                 native — the Lua-ness that survives dispatch folding.{})",
                if w64 {
                    ""
                } else {
                    " Set SVM_BENCH_W64=1 for the LP64-matched wasm64 baseline."
                }
            );
        } else {
            println!(
                "\n(native {native_kind} skipped: no rustc / {native_kind}-unknown-unknown target.)"
            );
        }
    }
}
