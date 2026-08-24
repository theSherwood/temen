//! **Mid-loop deopt resume against a ported C `savedpc`-style VM.**
//!
//! `savedpc_resume.rs` (temen-peval) showed mid-loop resume on a VM authored in temen-IR, where the
//! program counter's window address was hand-chosen so `savedpc` could be renamed. The blocker to
//! doing it on **ported C** was that the on-ramp bakes mutable-global addresses into the IR but did
//! not expose them, so a caller couldn't find where a C `static long savedpc` lives to rename it.
//! [`temen_llvm::Translated::data_symbols`] now exposes every global's name → window address; this test
//! uses it to rename a real C interpreter's `savedpc` and drive a mid-loop deopt/resume end to end.
//!
//! The VM (`clang -O2 → LLVM → temen-llvm → temen-IR`): a per-opcode `pc` lives in `savedpc`; the loop
//! reloads it, dispatches, and (like the register VM) uses `continue`-based conditional jumps so each
//! path stores a *constant* `pc` — which lets the specializer fold the dispatch once `savedpc` is
//! renamed. `run` seeds `savedpc = 0` and tail-calls `interp`; a `GUARD` opcode's cold branch is a
//! deopt target; the deopt spills `savedpc` and tail-calls `resume` (the interpreter, carried
//! unspecialized), which re-reads `savedpc` and finishes the loop from where the fast path stopped.
//!
//! The oracle is `out`, accumulated across the deopt: it is correct only if `resume` continued from
//! the checkpointed `savedpc` (a restart at `pc = 0` would re-run the emitted iterations and
//! double-count). Byte-identical `out` vs the interpreter on all three backends is a positive test of
//! genuine mid-loop resume — now on ported C, not synthetic temen-IR.
//!
//! Run: `cargo test -p temen-llvm --test peval_savedpc_ported -- --nocapture`

use std::process::Command;

use temen_interp::Value;
use temen_ir::{Inst, Module, Terminator};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig};
use temen_verify::verify_module;

// Per-opcode `pc` in `savedpc`; `reg[0]` is the countdown, `out` the accumulator. pc states:
//   0 LOADIN (reg0=input)   1 LOOPCHK (reg0==0 → HALT/pc=5, else fall to pc=2)   2 EMIT (out+=reg0)
//   3 GUARD  (reg0==5 → bonus(); a deopt target)   4 DEC (reg0-=1; jump pc=1)   5.. HALT (return out)
// `continue`-based jumps keep each stored `pc` a constant, so the dispatch folds once savedpc renames.
const VM: &str = r#"
static long reg[16];
static long savedpc;
static long out;
__attribute__((noinline)) void emit(long v) { out += v; }
__attribute__((noinline)) void bonus(void) { out += 1000; }
__attribute__((noinline)) long interp(long input) {
    for (;;) {
        long pc = savedpc;
        if (pc == 0) { reg[0] = input; }
        else if (pc == 1) { if (reg[0] == 0) { savedpc = 5; continue; } }
        else if (pc == 2) { emit(reg[0]); }
        else if (pc == 3) { if (reg[0] == 5) bonus(); }
        else if (pc == 4) { reg[0] -= 1; savedpc = 1; continue; }
        else return out;
        savedpc = pc + 1;
    }
}
long run(long input)    { savedpc = 0; return interp(input); }
long resume(long input) { return interp(input); }
int main(void) { return (int)run(0); }
"#;

fn compile(name: &str, src: &str) -> Option<temen_llvm::Translated> {
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
    Some(t)
}

fn fn_idx(t: &temen_llvm::Translated, name: &str) -> u32 {
    t.exports
        .iter()
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("{name} fn export"))
        .1
}
fn data_addr(t: &temen_llvm::Translated, name: &str) -> (u64, u64) {
    let s = t
        .data_symbols
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("{name} data symbol"));
    (s.addr, s.size)
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

fn has_br_table(f: &temen_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
fn n_return_calls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { .. }))
        .count()
}

/// sum(1..=input) + (1000 if input >= 5 else 0) — the interpreter's answer (the guard's `bonus`
/// fires exactly once, when the countdown passes 5).
fn expected(input: i64) -> i64 {
    let sum: i64 = (1..=input.max(0)).sum();
    sum + if input >= 5 { 1000 } else { 0 }
}

#[test]
fn deopt_resumes_a_ported_interpreter_from_the_saved_pc() {
    let Some(t) = compile("peval_savedpc_ported", VM) else {
        eprintln!("clang unavailable — skipping");
        return;
    };
    let m = &t.module;
    verify_module(m).expect("interpreter verifies");
    let run = fn_idx(&t, "run");
    let interp = fn_idx(&t, "interp");
    let resume = fn_idx(&t, "resume");
    let bonus = fn_idx(&t, "bonus");
    let (spc_addr, spc_size) = data_addr(&t, "savedpc"); // the address we could not find before

    // Sanity: the interpreter matches our oracle.
    for input in [0i64, 4, 5, 6, 20] {
        assert_eq!(
            tw(m, run, &[t.entry_sp as i64, input]),
            expected(input),
            "interpreter @ {input}"
        );
    }

    // The GUARD cold block (the one that calls `bonus`) is the deopt target.
    let cold = m.funcs[interp as usize]
        .blocks
        .iter()
        .position(|b| {
            b.insts
                .iter()
                .any(|i| matches!(i, Inst::Call { func, .. } if *func == bonus))
        })
        .expect("the guard cold block calls bonus") as u32;

    // Rename the ported interpreter's `savedpc` (found via `data_symbols`) so `pc` folds, and mark the
    // guard cold block a deopt target that bails to the carried `resume`.
    let cfg = SpecConfig {
        rename: Some((spc_addr, spc_addr + spc_size)),
        rename_is_private: true,
        deopt_targets: vec![(interp, cold)],
        deopt_handler: Some(resume),
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(
        m,
        run,
        &[SpecArg::ConstI64(t.entry_sp as i64), SpecArg::Dynamic],
        &cfg,
    )
    .expect("specializes the ported savedpc VM");
    verify_module(&residual).expect("residual re-verifies");

    // The dispatch folded (renaming `savedpc` collapsed the pc-switch), and the cold guard edge deopts
    // to the carried resume handler.
    assert!(
        !has_br_table(&residual.funcs[0]),
        "dispatch folded (no residual br_table)"
    );
    assert_eq!(
        n_return_calls(&residual.funcs[0]),
        1,
        "the cold guard edge deopts to resume"
    );
    // Rolled: block count is fixed (one specialization), independent of the dynamic input.
    assert!(
        residual.funcs[0].blocks.len() <= 60,
        "the loop rolled (got {} blocks)",
        residual.funcs[0].blocks.len()
    );

    // Byte-identical to the interpreter on all three backends. Correct `out` proves resume continued
    // from the checkpointed `savedpc` — a restart at pc=0 would double-count the emitted iterations.
    for input in [0i64, 1, 4, 5, 6, 7, 20, 100, 1000] {
        let want = expected(input);
        assert_eq!(
            tw(m, run, &[t.entry_sp as i64, input]),
            want,
            "interpreter @ {input}"
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
            "temen-jit residual @ {input}"
        );
    }
}
