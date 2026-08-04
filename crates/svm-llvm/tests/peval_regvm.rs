//! **Futamura projection of a register-machine bytecode VM** — the Lua / QuickJS execution model
//! (a register file + a constant program + a decode/dispatch loop), one rung up from the
//! tape-machine Brainfuck demo (`peval_demo.rs`). A real register VM in C is compiled
//! `clang -O2 → LLVM → svm-llvm → svm-IR`, then partial-evaluated against a fixed bytecode program
//! (declared constant — the §20c caller contract). The residual is the *compiled* program: decode
//! and dispatch fold away, leaving the data-dependent loop over the register file.
//!
//! This is the correctness gate (residual ≡ interpreter on every input, on the tree-walk oracle, the
//! bytecode engine, and the Cranelift JIT). The four-backend ROI number — including svm-wasm-jit on
//! Wasmtime — lives in `bench/src/bin/peval_regvm.rs`.
//!
//! Run: `cargo test -p svm-llvm --test peval_regvm -- --nocapture`  (svm-llvm is workspace-excluded).

use std::process::Command;

use svm_interp::Value;
use svm_ir::Module;
use svm_peval::{optimize_module, specialize, SpecArg};
use svm_verify::verify_module;

/// A register-machine bytecode interpreter. Fixed-width `(op,a,b,c)` instructions in a flat `int`
/// array; a 16-slot `long` register file (zero-initialized at load, so each register is set before
/// it is read); `prog` is an **opaque pointer** clang cannot fold. The bundled program computes
/// `3 * input` via a runtime-count loop, so the residual is a loop, not a full unroll.
const VM_C: &str = r#"
enum { HALT, LOADI, LOADIN, ADD, SUB, ADDK, JLE, JMP };
static long reg[16];
long run(const int *prog, long input) {
    long pc = 0;
    for (;;) {
        int op = prog[pc], a = prog[pc+1], b = prog[pc+2], c = prog[pc+3];
        if (op == HALT) return reg[a];
        else if (op == LOADI)  reg[a] = b;
        else if (op == LOADIN) reg[a] = input;
        else if (op == ADD)    reg[a] = reg[b] + reg[c];
        else if (op == SUB)    reg[a] = reg[b] - reg[c];
        else if (op == ADDK)   reg[a] = reg[b] + c;
        else if (op == JLE)  { if (reg[a] <= reg[b]) { pc = (long)c * 4; continue; } }
        else if (op == JMP)  { pc = (long)a * 4; continue; }
        pc += 4;
    }
}
static const int prog[] = {
    LOADIN, 0, 0,  0,   /* r0 = input          */
    LOADI,  1, 0,  0,   /* r1 = 0              */
    LOADI,  2, 0,  0,   /* r2 = 0              */
    JLE,    0, 2,  7,   /* if r0<=0 goto HALT   (loop header) */
    ADDK,   1, 1,  3,   /* r1 += 3             */
    ADDK,   0, 0, -1,   /* r0 -= 1             */
    JMP,    3, 0,  0,   /* goto header         */
    HALT,   1, 0,  0,   /* return r1           */
};
int main(void) { return (int)run(prog, 0); }
"#;

/// The program bytes as they appear in the readonly segment (little-endian `int`s), for locating it.
fn prog_bytes() -> Vec<u8> {
    let words: [i32; 32] = [
        2, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 6, 0, 2, 7, 5, 1, 1, 3, 5, 0, 0, -1, 7, 3, 0, 0, 0, 1,
        0, 0,
    ];
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// Compile C → LLVM (`clang -O2`) → svm-IR. `None` if clang is unavailable (skip).
fn compile_c() -> Option<(Module, i64)> {
    let base = std::env::temp_dir().join("peval_regvm_test");
    let cf = base.with_extension("c");
    let ll = base.with_extension("ll");
    std::fs::write(&cf, VM_C).ok()?;
    let ok = Command::new("clang")
        .args([
            "-O2",
            "-emit-llvm",
            "-S",
            "-fno-vectorize",
            "-fno-slp-vectorize",
        ])
        .arg(&cf)
        .arg("-o")
        .arg(&ll)
        .status()
        .ok()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        return None;
    }
    let t = svm_llvm::translate_ll_path(&ll).expect("svm-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&ll);
    Some((t.module, t.entry_sp as i64))
}

fn entry(m: &Module) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == "run")
        .expect("run export")
        .func
}

fn prog_addr(m: &Module) -> i64 {
    let pb = prog_bytes();
    let d = m
        .data
        .iter()
        .find(|d| d.readonly && d.bytes.windows(pb.len()).any(|w| w == pb.as_slice()))
        .expect("readonly program segment");
    let off = d
        .bytes
        .windows(pb.len())
        .position(|w| w == pb.as_slice())
        .unwrap();
    d.offset as i64 + off as i64
}

fn tw(m: &Module, e: u32, args: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = args.iter().map(|&a| Value::I64(a)).collect();
    match svm_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad result {o:?}"),
    }
}

fn bc(m: &Module, e: u32, args: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = args.iter().map(|&a| Value::I64(a)).collect();
    match svm_interp::bytecode::compile_and_run(m, e, &vs, &mut fuel)
        .expect("in subset")
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad result {o:?}"),
    }
}

fn jit(m: &Module, e: u32, args: &[i64]) -> i64 {
    match svm_jit::compile_and_run(m, e, args) {
        Ok(svm_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

#[test]
fn regvm_specializes_and_matches_interpreter() {
    let Some((m, sp)) = compile_c() else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    verify_module(&m).expect("interpreter verifies");
    let e = entry(&m);
    let pa = prog_addr(&m);

    let residual = optimize_module(
        &specialize(
            &m,
            e,
            &[
                SpecArg::ConstI64(sp),
                SpecArg::ConstI64(pa),
                SpecArg::Dynamic,
            ],
        )
        .expect("specializes the register VM"),
    );
    verify_module(&residual).expect("residual re-verifies");

    // The dispatch folded: the residual is far smaller than the interpreter's run().
    assert!(
        residual.funcs[0].blocks.len() < m.funcs[e as usize].blocks.len(),
        "residual should be smaller than the interpreter loop"
    );

    for input in [0i64, 1, 2, 7, 100, 1000, 100_000] {
        let want = 3 * input;
        assert_eq!(
            tw(&m, e, &[sp, pa, input]),
            want,
            "interpreter itself wrong @ {input}"
        );
        assert_eq!(
            tw(&residual, 0, &[input]),
            want,
            "tree-walk residual @ {input}"
        );
        assert_eq!(
            bc(&residual, 0, &[input]),
            want,
            "bytecode residual @ {input}"
        );
        assert_eq!(
            jit(&residual, 0, &[input]),
            want,
            "svm-jit residual @ {input}"
        );
    }
}
