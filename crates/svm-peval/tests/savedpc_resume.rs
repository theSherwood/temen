//! **Mid-loop resume via a `savedpc`-style VM.**
//!
//! The deopt tests in `cut_state.rs` bail to a handler that computes a *terminal* value. A real tiered
//! JIT deopts **mid-loop**: it checkpoints the interpreter's program counter and hands control to the
//! baseline interpreter, which picks up *where the fast path left off* and runs the remaining
//! iterations. This test demonstrates exactly that on a small bytecode-dispatch VM whose `pc` lives in
//! memory — a `savedpc` — so the deopt can write it back and the handler can resume from it.
//!
//! Structure (a tiered shape): `run` (the entry) seeds `savedpc = 0` and tail-calls `interp`; `interp`
//! is the dispatch loop, reading `pc` from the `savedpc` **rename cell** (so `pc` folds and the
//! dispatch collapses) and driving a countdown that sums `1..input` into `out`. A `GUARD` opcode's
//! cold branch (fires once, when the counter hits 5) is a **deopt target**: the specializer keeps the
//! fast path (folded, rolled) and, on that edge, spills `savedpc` to memory and tail-calls `resume`
//! — the baseline interpreter, carried unspecialized, which re-reads `savedpc` and finishes the loop.
//! `resume` shares `run`'s signature, so the dynamic `input` is threaded to the deopt edge.
//!
//! The oracle is `out`, an observable side effect accumulated *across* the deopt. It is right (= the
//! full sum) **only if `resume` continued from the checkpointed `savedpc`** — a resume that restarted
//! at `pc = 0` would re-run the already-emitted iterations and double-count. So byte-identical `out`
//! against the interpreter is a positive test that the resume was genuinely mid-loop.
//!
//! `savedpc` is at a fixed, renameable window address here (authored in svm-IR) to control its layout;
//! wiring this against on-ramp-compiled C additionally needs the on-ramp to expose mutable-global
//! addresses so `savedpc` can be renamed — a separate integration task.
//!
//! Run: `cargo test -p svm-peval --test savedpc_resume -- --nocapture`

use svm_interp::{Trap, Value};
use svm_ir::{Module, Terminator};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use svm_text::parse_module;
use svm_verify::verify_module;

// savedpc @0 (rename cell, 4 bytes), counter @8 (plain), out @16 (plain). Opcode pc values:
//   0 LOADIN (counter=input; pc=1)   1 LOOPCHK (counter==0 ? pc=5 : pc=2)   2 EMIT (out+=counter; pc=3)
//   3 GUARD  (counter==5 ? cold : ok; pc=4)   4 DEC (counter-=1; pc=1)   5.. HALT (return out)
// interp is func 1; block 10 is the GUARD cold branch (the deopt target). `run` (func 0) and `resume`
// (func 2) both tail-call interp; run seeds savedpc=0, resume relies on the spilled savedpc.
const VM: &str = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsp = i64.const 0
  vpc0 = i32.const 0
  i32.store vsp vpc0
  return_call 1 (v0)
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (vin: i64) {
  vsp = i64.const 0
  vpc = i32.load vsp
  br_table vpc [ 2(vin), 3(vin), 4(vin), 5(vin), 6(vin) ] 7(vin)
}
block 2 (vin: i64) {
  v8 = i64.const 8
  i64.store v8 vin
  vsp = i64.const 0
  v1 = i32.const 1
  i32.store vsp v1
  br 1(vin)
}
block 3 (vin: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  vz = i64.const 0
  visz = i64.eq vc vz
  br_if visz 8(vin) 9(vin)
}
block 4 (vin: i64) {
  v16 = i64.const 16
  vout = i64.load v16
  v8 = i64.const 8
  vc = i64.load v8
  vo2 = i64.add vout vc
  i64.store v16 vo2
  vsp = i64.const 0
  v3 = i32.const 3
  i32.store vsp v3
  br 1(vin)
}
block 5 (vin: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  v5 = i64.const 5
  visg = i64.eq vc v5
  br_if visg 10(vin) 11(vin)
}
block 6 (vin: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  vsp = i64.const 0
  v1p = i32.const 1
  i32.store vsp v1p
  br 1(vin)
}
block 7 (vin: i64) {
  v16 = i64.const 16
  vr = i64.load v16
  return vr
}
block 8 (vin: i64) {
  vsp = i64.const 0
  v5 = i32.const 5
  i32.store vsp v5
  br 1(vin)
}
block 9 (vin: i64) {
  vsp = i64.const 0
  v2 = i32.const 2
  i32.store vsp v2
  br 1(vin)
}
block 10 (vin: i64) {
  vsp = i64.const 0
  v4 = i32.const 4
  i32.store vsp v4
  br 1(vin)
}
block 11 (vin: i64) {
  vsp = i64.const 0
  v4 = i32.const 4
  i32.store vsp v4
  br 1(vin)
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return_call 1 (v0)
}
}
"#;

fn run(m: &Module, x: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    svm_interp::run(m, 0, &[Value::I64(x)], &mut fuel)
}
fn n_return_calls(f: &svm_ir::Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { .. }))
        .count()
}
fn has_br_table(f: &svm_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
/// sum(1..=input), clamped at 0 for non-positive input — the VM's answer.
fn expected(input: i64) -> i64 {
    (1..=input.max(0)).sum()
}

#[test]
fn deopt_resumes_the_loop_from_the_saved_pc() {
    let m = parse_module(VM).expect("parse");
    verify_module(&m).expect("source verifies");

    // Sanity: the interpreter itself sums 1..input (the GUARD is transparent in the baseline).
    for input in [0i64, 1, 4, 5, 6, 20] {
        assert_eq!(
            run(&m, input),
            Ok(vec![Value::I64(expected(input))]),
            "interpreter @ {input}"
        );
    }

    let cfg = SpecConfig {
        rename: Some((0, 4)),
        rename_is_private: true,
        deopt_targets: vec![(1, 10)], // interp's GUARD cold branch
        deopt_handler: Some(2),       // resume: (i64) -> i64, shares run's signature
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    // The dispatch folded (no residual br_table — pc folds via the savedpc cell) and the cold guard
    // edge deopts (a tail-call to the resume handler), while the loop rolled.
    assert!(
        !has_br_table(&r.funcs[0]),
        "dispatch folded away (no residual br_table)"
    );
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "the cold guard edge deopts to resume"
    );
    // Rolled: the residual block count is fixed and small (one specialization, independent of the
    // dynamic trip count) — not one unrolled copy per iteration. (Pre-optimization, so a bit verbose.)
    assert!(
        r.funcs[0].blocks.len() <= 40,
        "the loop rolled (got {})",
        r.funcs[0].blocks.len()
    );

    // Byte-identical to the interpreter across inputs that stay on the fast path (< 5) and inputs that
    // cross the guard and deopt (>= 5). Correct `out` proves resume continued from the saved pc.
    for input in [0i64, 1, 4, 5, 6, 7, 20, 100, 1000] {
        assert_eq!(run(&r, input), run(&m, input), "diverged at input={input}");
        assert_eq!(
            run(&r, input),
            Ok(vec![Value::I64(expected(input))]),
            "input={input}"
        );
    }
}
