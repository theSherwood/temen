//! **wasm-JIT fuel/dispatch ROI bench** — emits each kernel to wasm under three fuel placements
//! (`Memory` = shipped, `Global` = register-allocatable, `None` = upper bound), writes the bytes +
//! a manifest to `$OUT` (default: a scratch dir), and times the **native** backends (tree-walk,
//! bytecode, Cranelift JIT) on the same IR for the cross-engine table. A sibling Node harness
//! (`fuelbench.mjs`) times the emitted wasm on V8. Run:
//!   cargo run --release -p svm-wasm-jit --example fuelbench -- <out_dir>

use std::hint::black_box;
use std::path::PathBuf;
use std::time::Instant;

use svm_interp::{bytecode, Value};
use svm_wasm_jit::{compile_module_fuel, FuelMode};

/// (name, description, svm-text). Each is `func (i64 n) -> (i64 acc)` at func 0 — an `n`-iteration
/// hot loop. Per-iteration cost is isolated by a large/small-`n` subtraction (the fixed entry/exit
/// cancels), and the loop bodies are fold-resistant recurrences so no engine closed-forms them.
const KERNELS: &[(&str, &str, &str)] = &[
    // Pure-ALU single-block loop: an LCG recurrence. No memory — in Memory mode V8 CAN keep the
    // fuel cell in a register (no aliasing store), so this is the null hypothesis for fuel-in-global.
    (
        "alu_lcg",
        "pure i64 LCG, single hot block, no memory",
        r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v3: i64, v4: i64) {
  v6 = i64.eqz v3
  br_if v6 3(v4) 2(v3, v4)
}
block 2 (v7: i64, v8: i64) {
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  v12 = i64.const 1442695040888963407
  v13 = i64.add v11 v12
  v15 = i64.const 1
  v16 = i64.sub v7 v15
  br 1(v16, v13)
}
block 3 (v17: i64) {
  return v17
  }
}
"#,
    ),
    // Serial xorshift hash — clang can't strength-reduce it; the representative scalar kernel.
    (
        "xorshift",
        "serial xorshift* scalar hash, no memory",
        r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 88172645463325252
  br 1(v0, v1)
}
block 1 (v3: i64, v4: i64) {
  v6 = i64.eqz v3
  br_if v6 3(v4) 2(v3, v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.const 13
  v10 = i64.shl v8 v9
  v11 = i64.xor v8 v10
  v12 = i64.const 7
  v13 = i64.shr_u v11 v12
  v14 = i64.xor v11 v13
  v15 = i64.const 17
  v16 = i64.shl v14 v15
  v17 = i64.xor v14 v16
  v18 = i64.const 1
  v19 = i64.sub v7 v18
  br 1(v19, v17)
}
block 3 (v20: i64) {
  return v20
  }
}
"#,
    ),
    // Store→load forwarding to a FIXED cell each iter. In Memory mode the guest store to
    // `win+(addr&mask)` may-alias the fuel cell at `env`, so V8 must reload fuel from memory every
    // iteration; Global mode removes the aliasing. The load-bearing A/B for the hypothesis.
    (
        "mem_fwd",
        "store→load a fixed window cell each iter (fuel-cell aliasing probe)",
        r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v3: i64, v4: i64) {
  v6 = i64.eqz v3
  br_if v6 3(v4) 2(v3, v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.const 64
  v10 = i64.const 6364136223846793005
  v11 = i64.mul v8 v10
  i64.store v9 v11
  v12 = i64.load v9
  v13 = i64.const 1442695040888963407
  v14 = i64.add v12 v13
  v15 = i64.const 1
  v16 = i64.sub v7 v15
  br 1(v16, v14)
}
block 3 (v17: i64) {
  return v17
  }
}
"#,
    ),
    // Scatter store to a rolling window address (definitely not register-allocatable), then read
    // back — a real memory-write loop, the strongest case for the aliasing hypothesis.
    (
        "mem_scatter",
        "scatter store to a rolling window slot each iter",
        r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1, v1)
}
block 1 (v3: i64, v4: i64, v5: i64) {
  v6 = i64.eqz v3
  br_if v6 3(v4) 2(v3, v4, v5)
}
block 2 (v7: i64, v8: i64, v9: i64) {
  v10 = i64.const 4088
  v11 = i64.and v9 v10
  v12 = i64.const 6364136223846793005
  v13 = i64.mul v8 v12
  i64.store v11 v13
  v14 = i64.load v11
  v15 = i64.const 1442695040888963407
  v16 = i64.add v14 v15
  v17 = i64.const 8
  v18 = i64.add v9 v17
  v19 = i64.const 1
  v20 = i64.sub v7 v19
  br 1(v20, v16, v18)
}
block 3 (v21: i64) {
  return v21
  }
}
"#,
    ),
    // Branchy loop: an inner if/else splits each iteration across extra blocks, so each loop trip
    // round-trips the dispatcher more than once — amplifies both per-dispatch fuel cost AND the
    // dispatch round-trip a relooper would remove. No memory (isolate control-flow cost).
    (
        "branchy",
        "loop with an inner if/else (multi-block per trip)",
        r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 0
  br 1(v0, v1)
}
block 1 (v3: i64, v4: i64) {
  v6 = i64.eqz v3
  br_if v6 6(v4) 2(v3, v4)
}
block 2 (v7: i64, v8: i64) {
  v9 = i64.const 1
  v10 = i64.and v7 v9
  v10b = i64.eqz v10
  br_if v10b 4(v7, v8) 3(v7, v8)
}
block 3 (v11: i64, v12: i64) {
  v13 = i64.const 6364136223846793005
  v14 = i64.mul v12 v13
  br 5(v11, v14)
}
block 4 (v15: i64, v16: i64) {
  v17 = i64.const 1442695040888963407
  v18 = i64.add v16 v17
  br 5(v15, v18)
}
block 5 (v19: i64, v20: i64) {
  v21 = i64.const 1
  v22 = i64.sub v19 v21
  br 1(v22, v20)
}
block 6 (v23: i64) {
  return v23
  }
}
"#,
    ),
];

const SMALL: i64 = 1_000;
const LARGE: i64 = 5_000_000;
const REPS: usize = 9;
const FUEL: u64 = 1u64 << 61;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Min-over-reps per-iteration ns via large/small subtraction, running `f` once per call.
fn per_iter_ns(mut f: impl FnMut(i64) -> i64) -> (f64, i64) {
    let mut best = f64::MAX;
    let mut result = 0i64;
    for _ in 0..REPS {
        let t = Instant::now();
        result = black_box(f(black_box(LARGE)));
        let tl = t.elapsed().as_nanos() as f64;
        let t = Instant::now();
        black_box(f(black_box(SMALL)));
        let ts = t.elapsed().as_nanos() as f64;
        let per = (tl - ts) / (LARGE - SMALL) as f64;
        if per < best {
            best = per;
        }
    }
    (best, result)
}

fn main() {
    let out: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/tmp/fuelbench"));
    std::fs::create_dir_all(&out).unwrap();

    let mut manifest = String::from("[\n");
    println!("engine,kernel,ns_per_iter,result");

    for (i, (name, _desc, src)) in KERNELS.iter().enumerate() {
        let m = parse(src);

        // Emit the three fuel variants to disk for the V8 harness.
        for (mode, tag) in [
            (FuelMode::Memory, "mem"),
            (FuelMode::Global, "global"),
            (FuelMode::None, "nofuel"),
        ] {
            let wasm =
                compile_module_fuel(&m, mode).unwrap_or_else(|e| panic!("{name}/{tag}: emit: {e}"));
            std::fs::write(out.join(format!("{name}.{tag}.wasm")), &wasm).unwrap();
        }

        // Tree-walk oracle.
        let (twns, twr) = per_iter_ns(|n| {
            let mut fuel = FUEL;
            match svm_interp::run(&m, 0, &[Value::I64(n)], &mut fuel) {
                Ok(v) => match v[0] {
                    Value::I64(x) => x,
                    _ => 0,
                },
                Err(_) => -1,
            }
        });
        println!("tree-walk,{name},{twns:.3},{twr}");

        // Bytecode engine.
        let prog = bytecode::SharedProgram::compile(&m).expect("bytecode compile");
        let (bcns, bcr) = per_iter_ns(|n| {
            let mut fuel = FUEL;
            match bytecode::compile_and_run(&m, 0, &[Value::I64(n)], &mut fuel) {
                Some(Ok(v)) => match v[0] {
                    Value::I64(x) => x,
                    _ => 0,
                },
                _ => -1,
            }
        });
        let _ = prog;
        println!("bytecode,{name},{bcns:.3},{bcr}");

        // Cranelift JIT (native), compiled once, run per call.
        let mut cm = svm_jit::compile(&m, 0).expect("cranelift compile");
        let (jitns, jitr) = per_iter_ns(|n| match cm.run(&[n], None, None, None) {
            Ok((svm_jit::JitOutcome::Returned(r), _)) => r[0],
            _ => -1,
        });
        println!("cranelift,{name},{jitns:.3},{jitr}");

        manifest.push_str(&format!(
            "  {{\"name\":\"{name}\",\"result\":\"{twr}\",\"small\":{SMALL},\"large\":{LARGE}}}{}\n",
            if i + 1 < KERNELS.len() { "," } else { "" }
        ));

        // Sanity: all native backends agree.
        assert_eq!(twr, bcr, "{name}: tree-walk vs bytecode mismatch");
        assert_eq!(twr, jitr, "{name}: tree-walk vs cranelift mismatch");
    }
    manifest.push_str("]\n");
    std::fs::write(out.join("manifest.json"), manifest).unwrap();
    eprintln!("wrote wasm variants + manifest to {}", out.display());
}
