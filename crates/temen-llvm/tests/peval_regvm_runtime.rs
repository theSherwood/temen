//! **Wiring the cut set and guard-and-deopt against a real ported interpreter.**
//!
//! `peval_regvm.rs` shows the Futamura projection of a C register-machine VM (`clang -O2 → LLVM →
//! temen-llvm → temen-IR`): specialized against a constant bytecode program, decode + dispatch fold and
//! the residual is the compiled program. That VM is pure compute. A *real* interpreter also (a) makes
//! **runtime call-outs** — a boxed op, an allocator, a write barrier, a GC safepoint — you don't want
//! the specializer to *project* (they're stateful or reach GC/`setjmp` machinery that can't fold); and
//! (b) has **cold paths** — a type-check failure, an overflow slow path, an error raise — you want to
//! bail out of rather than compile. #617/#619 added exactly these controls; this test drives them
//! against genuinely ported C, not synthetic temen-IR (the register VM is the Lua/QuickJS execution
//! model — the mechanisms are interpreter-agnostic).
//!
//! - `cut_keeps_the_runtime_builtin_opaque_in_the_rolled_residual` — a `noinline` builtin `ext` kept
//!   opaque with [`SpecConfig::cut_calls`]: dispatch folds, the loop rolls, and `ext` survives as one
//!   carried call, byte-identical across all three backends.
//! - `a_cold_guard_deopts_to_the_resume_handler` — a `GUARD` opcode whose cold branch (`return
//!   slow(reg[a])`) is a [`SpecConfig::deopt_targets`] block: the specializer bails to a `resume`
//!   handler (sharing `run`'s signature, so the entry's dynamic `input` is *threaded* to the deopt
//!   edge) instead of projecting the cold path. Fast path folds and rolls; cold path deopts.
//!
//! Run: `cargo test -p temen-llvm --test peval_regvm_runtime -- --nocapture`

use std::process::Command;

use temen_interp::Value;
use temen_ir::{Inst, Module, Terminator};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};
use temen_verify::verify_module;

/// Compile a C source `src` → LLVM (`clang -O2`) → temen-IR. `None` if clang is unavailable (skip).
fn compile(name: &str, src: &str) -> Option<(Module, i64)> {
    let base = std::env::temp_dir().join(name);
    let cf = base.with_extension("c");
    let ll = base.with_extension("ll");
    std::fs::write(&cf, src).ok()?;
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
    let t = temen_llvm::translate_ll_path(&ll).expect("temen-llvm translate");
    let _ = std::fs::remove_file(&cf);
    let _ = std::fs::remove_file(&ll);
    Some((t.module, t.entry_sp as i64))
}

fn export(m: &Module, name: &str) -> u32 {
    m.exports
        .iter()
        .find(|e| e.name == name)
        .unwrap_or_else(|| panic!("{name} export"))
        .func
}

/// The readonly data address of `prog` (located by its little-endian `int` bytes).
fn prog_addr(m: &Module, words: &[i32]) -> i64 {
    let pb: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
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
    match temen_interp::run(m, e, &vs, &mut fuel)
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
    match temen_interp::bytecode::compile_and_run(m, e, &vs, &mut fuel)
        .expect("in subset")
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad result {o:?}"),
    }
}
fn jit(m: &Module, e: u32, args: &[i64]) -> i64 {
    match temen_jit::compile_and_run(m, e, args) {
        Ok(temen_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit outcome {o:?}"),
    }
}

fn n_calls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count()
}
fn n_return_calls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { .. }))
        .count()
}
fn has_br_table(f: &temen_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
/// The first block of `f` that calls `target` — used to locate the cold-path handler block.
fn block_calling(f: &temen_ir::Func, target: u32) -> u32 {
    f.blocks
        .iter()
        .position(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == target))
        })
        .expect("a block calling the target") as u32
}

// =============================================================================================
// Cut set: a runtime builtin kept opaque.
// =============================================================================================

// `CALL` opcode: `reg[a] = ext(reg[b])`, `ext` a `noinline` builtin (`x + 7`). The program applies it
// `input` times to a 0-seeded accumulator via a runtime-count loop → residual is a rolled loop with
// one opaque call per iteration. Result = 7·input.
const CUT_VM: &str = r#"
enum { HALT, LOADI, LOADIN, ADD, ADDK, JLE, JMP, CALL };
static long reg[16];
__attribute__((noinline)) long ext(long x) { return x + 7; }
long run(const int *prog, long input) {
    long pc = 0;
    for (;;) {
        int op = prog[pc], a = prog[pc+1], b = prog[pc+2], c = prog[pc+3];
        if (op == HALT) return reg[a];
        else if (op == LOADI)  reg[a] = b;
        else if (op == LOADIN) reg[a] = input;
        else if (op == ADD)    reg[a] = reg[b] + reg[c];
        else if (op == ADDK)   reg[a] = reg[b] + c;
        else if (op == CALL)   reg[a] = ext(reg[b]);
        else if (op == JLE)  { if (reg[a] <= reg[b]) { pc = (long)c * 4; continue; } }
        else if (op == JMP)  { pc = (long)a * 4; continue; }
        pc += 4;
    }
}
static const int prog[] = {
    LOADIN, 0, 0,  0,   LOADI,  1, 0,  0,   LOADI,  2, 0,  0,   JLE,    0, 2,  7,
    CALL,   1, 1,  0,   ADDK,   0, 0, -1,   JMP,    3, 0,  0,   HALT,   1, 0,  0,
};
int main(void) { return (int)run(prog, 0); }
"#;
// Opcodes: HALT=0 LOADI=1 LOADIN=2 ADD=3 ADDK=4 JLE=5 JMP=6 CALL=7.
const CUT_PROG: [i32; 32] = [
    2, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 5, 0, 2, 7, 7, 1, 1, 0, 4, 0, 0, -1, 6, 3, 0, 0, 0, 1, 0, 0,
];

#[test]
fn cut_keeps_the_runtime_builtin_opaque_in_the_rolled_residual() {
    let Some((m, sp)) = compile("peval_regvm_cut", CUT_VM) else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    verify_module(&m).expect("interpreter verifies");
    let run = export(&m, "run");
    let ext = export(&m, "ext");
    let pa = prog_addr(&m, &CUT_PROG);

    assert!(
        n_calls(&m.funcs[run as usize]) >= 1,
        "the VM should call ext"
    );
    assert!(
        has_br_table(&m.funcs[run as usize]),
        "the VM dispatches through a table"
    );

    let residual = specialize_with_config(
        &m,
        run,
        &[
            SpecArg::ConstI64(sp),
            SpecArg::ConstI64(pa),
            SpecArg::Dynamic,
        ],
        &SpecConfig {
            cut_calls: vec![ext],
            ..SpecConfig::default()
        },
    )
    .expect("specializes the register VM with a cut call-out");
    verify_module(&residual).expect("residual re-verifies");

    // Dispatch folded (no residual `br_table`), the loop rolled, and the builtin survives as a single
    // carried opaque call — not projected, not inlined, not unrolled. (A downstream optimizer may later
    // inline a trivial `+7` leaf; the property under test is that the *specializer* kept it opaque —
    // what the GC/`setjmp` wall makes essential for a real call-out.)
    assert_eq!(residual.funcs.len(), 2, "entry + the carried builtin");
    assert!(
        !has_br_table(&residual.funcs[0]),
        "dispatch folded (no residual br_table)"
    );
    assert_eq!(
        n_calls(&residual.funcs[0]),
        1,
        "one opaque builtin call survives the rolled loop"
    );

    for input in [0i64, 1, 2, 7, 100, 1000, 100_000] {
        let want = 7 * input;
        assert_eq!(tw(&m, run, &[sp, pa, input]), want, "interpreter @ {input}");
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
            "temen-jit residual @ {input}"
        );
    }
}

// =============================================================================================
// Guard-and-deopt: a cold path bailed out of rather than projected.
// =============================================================================================

// `GUARD a _ c`: if `reg[a] > c`, take the cold path `return slow(reg[a])`; else fall through. The
// program accumulates `r1 += 7` per iteration under `GUARD r1 > 1000`; once `r1` crosses 1000 the
// guard fires and the interpreter returns `slow(r1)`. We mark that cold block a deopt target: the
// residual keeps the fast path (folded, rolled) and, on the cold edge, tail-calls `resume` — which
// shares `run`'s signature, so the entry's dynamic `input` is threaded to the deopt edge. `resume`
// finishes from the register state (`slow(reg[1])`), exactly as the cold block would.
const DEOPT_VM: &str = r#"
enum { HALT, LOADI, LOADIN, ADDK, JLE, JMP, GUARD };
static long reg[16];
__attribute__((noinline)) long slow(long x) { return x * 100; }
long resume(const int *prog, long input) { (void)prog; (void)input; return slow(reg[1]); }
long run(const int *prog, long input) {
    long pc = 0;
    for (;;) {
        int op = prog[pc], a = prog[pc+1], b = prog[pc+2], c = prog[pc+3];
        if (op == HALT) return reg[a];
        else if (op == LOADI)  reg[a] = b;
        else if (op == LOADIN) reg[a] = input;
        else if (op == ADDK)   reg[a] = reg[b] + c;
        else if (op == GUARD)  { if (reg[a] > c) return slow(reg[a]); }
        else if (op == JLE)  { if (reg[a] <= reg[b]) { pc = (long)c * 4; continue; } }
        else if (op == JMP)  { pc = (long)a * 4; continue; }
        pc += 4;
    }
}
static const int prog[] = {
    LOADIN, 0, 0,    0,   LOADI, 1, 0, 0,   LOADI, 2, 0, 0,   JLE,  0, 2, 8,
    GUARD,  1, 0, 1000,   ADDK,  1, 1, 7,   ADDK,  0, 0, -1,   JMP,  3, 0, 0,
    HALT,   1, 0,    0,
};
int main(void) { return (int)run(prog, 0); }
"#;
// Opcodes: HALT=0 LOADI=1 LOADIN=2 ADDK=3 JLE=4 JMP=5 GUARD=6.
const DEOPT_PROG: [i32; 36] = [
    2, 0, 0, 0, 1, 1, 0, 0, 1, 2, 0, 0, 4, 0, 2, 8, 6, 1, 0, 1000, 3, 1, 1, 7, 3, 0, 0, -1, 5, 3,
    0, 0, 0, 1, 0, 0,
];

/// The interpreter's own answer: accumulate 7 per step; once the accumulator would exceed 1000 while
/// the counter is still positive, the guard fires and returns `slow(accumulator)`.
fn expected(input: i64) -> i64 {
    let mut acc = 0i64;
    for _ in 0..input {
        if acc > 1000 {
            return acc * 100; // slow(acc)
        }
        acc += 7;
    }
    acc
}

#[test]
fn a_cold_guard_deopts_to_the_resume_handler() {
    let Some((m, sp)) = compile("peval_regvm_deopt", DEOPT_VM) else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    verify_module(&m).expect("interpreter verifies");
    let run = export(&m, "run");
    let slow = export(&m, "slow");
    let resume = export(&m, "resume");
    let pa = prog_addr(&m, &DEOPT_PROG);
    // The cold-path block is the one that calls `slow` (the `return slow(reg[a])` handler).
    let cold = block_calling(&m.funcs[run as usize], slow);

    // Sanity: the interpreter matches our oracle (guard fires for large inputs).
    for input in [0i64, 10, 143, 144, 500] {
        assert_eq!(
            tw(&m, run, &[sp, pa, input]),
            expected(input),
            "interpreter @ {input}"
        );
    }

    let residual = specialize_with_config(
        &m,
        run,
        &[
            SpecArg::ConstI64(sp),
            SpecArg::ConstI64(pa),
            SpecArg::Dynamic,
        ],
        &SpecConfig {
            deopt_targets: vec![(run, cold)],
            deopt_handler: Some(resume),
            ..SpecConfig::default()
        },
    )
    .expect("specializes the register VM with a deopt guard");
    verify_module(&residual).expect("residual re-verifies");

    // Dispatch folded, the loop rolled, and the cold guard branch deopts (a tail-call to the carried
    // resume handler) instead of projecting `slow`. `slow` never appears in the entry — it lives only
    // behind the resume handler.
    assert!(
        !has_br_table(&residual.funcs[0]),
        "dispatch folded (no residual br_table)"
    );
    assert_eq!(
        n_return_calls(&residual.funcs[0]),
        1,
        "the cold guard edge deopts to the handler"
    );

    for input in [0i64, 1, 100, 143, 144, 200, 1000, 100_000] {
        let want = expected(input);
        assert_eq!(tw(&m, run, &[sp, pa, input]), want, "interpreter @ {input}");
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
            "temen-jit residual @ {input}"
        );
    }
}
