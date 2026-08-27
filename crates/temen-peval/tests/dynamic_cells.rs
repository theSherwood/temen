//! **Runtime-input cells: the Futamura dynamic input that makes a captured loop roll.**
//!
//! `savedpc_resume.rs` rolls a dispatch loop whose runtime input arrives as a **threaded argument**
//! (`run(input)` passes it to `interp`). But a *capture-based* projection has no such wrapper: the
//! specializer is pointed straight at the interpreter's entry (`luaV_execute`) with the whole VM state
//! seeded from a live snapshot, and the program's runtime input is already sitting in a **register** in
//! that snapshot. If every seeded cell folds to its captured constant, the loop unrolls to a constant
//! (that is exactly what `lua_futamura_vmstate.rs` shows). To get a *rolled, input-taking* residual we
//! must declare which captured cells are runtime input — [`SpecConfig::dynamic_cells`] does that: the
//! named cells become dynamic residual parameters instead of folding to their seed.
//!
//! This models it minimally. `interp` (func 0) is a `savedpc`-style dispatch VM read entirely from
//! memory: `savedpc@16384` (the renamed pc), `reg0@16392`, `out@16400`, and the loop bound `N@16408`. Seeding the
//! whole `[16384,16416)` region folds `savedpc` (so the dispatch collapses) — but marking `N@16408` a
//! **dynamic cell** keeps the countdown's initial value dynamic, so the loop condition stays dynamic
//! and the loop **rolls**. The residual is `(N) -> out` computing `N + (N-1) + … + 1 = N(N+1)/2`, the
//! same shape a rolled `luaV_execute` residual would have. `run` (func 1) is the oracle wrapper: it
//! writes `N`/`savedpc` into memory and tail-calls `interp`, so the interpreter can be run for any `N`.
//!
//! Byte-identical output vs the interpreter on all three backends, across `N` that exercises the rolled
//! loop, is the positive test: the dispatch folded, the loop rolled, and `N` genuinely entered as a
//! residual parameter (not a baked constant).
//!
//! Run: `cargo test -p temen-peval --test dynamic_cells -- --nocapture`

use temen_interp::{Trap, Value};
use temen_ir::{Module, Terminator};
use temen_peval::{specialize_with_config, SpecConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

// Cells sit one guard up (above the #1094 NULL guard): savedpc@16384 (i32 rename cell),
// reg0@16392 (i64), out@16400 (i64), N@16408 (i64, the runtime-input cell).
// pc states:  0 LOADN (reg0=N; pc=1)  1 LOOPCHK (reg0==0 ? pc=4 : pc=2)  2 EMIT (out+=reg0; pc=3)
//             3 DEC (reg0-=1; pc=1)    4.. HALT (return out)
// `interp` (func 0) reads all state from memory; `run` (func 1) seeds it and tail-calls interp.
const VM: &str = r#"
memory 17
func () -> (i64) {
block 0 () {
  br 1()
}
block 1 () {
  vsp = i64.const 16384
  vpc = i32.load vsp
  br_table vpc [ 2(), 3(), 4(), 5() ] 6()
}
block 2 () {
  v24 = i64.const 16408
  vn = i64.load v24
  v8 = i64.const 16392
  i64.store v8 vn
  vsp = i64.const 16384
  v1 = i32.const 1
  i32.store vsp v1
  br 1()
}
block 3 () {
  v8 = i64.const 16392
  vc = i64.load v8
  vz = i64.const 0
  visz = i64.eq vc vz
  br_if visz 7() 8()
}
block 4 () {
  v16 = i64.const 16400
  vout = i64.load v16
  v8 = i64.const 16392
  vc = i64.load v8
  vo2 = i64.add vout vc
  i64.store v16 vo2
  vsp = i64.const 16384
  v3 = i32.const 3
  i32.store vsp v3
  br 1()
}
block 5 () {
  v8 = i64.const 16392
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  vsp = i64.const 16384
  v1p = i32.const 1
  i32.store vsp v1p
  br 1()
}
block 6 () {
  v16 = i64.const 16400
  vr = i64.load v16
  return vr
}
block 7 () {
  vsp = i64.const 16384
  v4 = i32.const 4
  i32.store vsp v4
  br 1()
}
block 8 () {
  vsp = i64.const 16384
  v2 = i32.const 2
  i32.store vsp v2
  br 1()
}
}
func (i64) -> (i64) {
block 0 (vn: i64) {
  v24 = i64.const 16408
  i64.store v24 vn
  vsp = i64.const 16384
  vz = i32.const 0
  i32.store vsp vz
  return_call 0 ()
}
}
"#;

fn run_oracle(m: &Module, n: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    temen_interp::run(m, 1, &[Value::I64(n)], &mut fuel)
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
/// N + (N-1) + … + 1 = N(N+1)/2 (clamped at 0 for non-positive N) — the VM's answer.
fn expected(n: i64) -> i64 {
    (1..=n.max(0)).sum()
}

#[test]
fn a_dynamic_cell_rolls_a_captured_loop() {
    let m = parse_module(VM).expect("parse");
    verify_module(&m).expect("source verifies");

    // Sanity: the interpreter (via the `run` wrapper) computes N(N+1)/2.
    for n in [0i64, 1, 4, 5, 20] {
        assert_eq!(
            run_oracle(&m, n),
            Ok(vec![Value::I64(expected(n))]),
            "interpreter @ {n}"
        );
    }

    // Point the specializer straight at `interp` (func 0), no wrapper. Seed the whole [16384,16416) VM-state
    // region from a zero overlay (savedpc=0, reg0=0, out=0) so `savedpc` folds and the dispatch
    // collapses — but mark `N@16408` a runtime-input cell so it enters as a dynamic residual parameter.
    let cfg = SpecConfig {
        rename: Some((16384, 16416)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_overlays: vec![(16384, vec![0u8; 32])],
        dynamic_cells: vec![(16408, 8)],
        ..SpecConfig::default()
    };
    let r =
        specialize_with_config(&m, 0, &[], &cfg).expect("specializes interp with a dynamic cell");
    verify_module(&r).expect("residual verifies");

    // The residual takes exactly one runtime parameter — `N` — and the dispatch folded.
    assert_eq!(
        r.funcs[0].params.len(),
        1,
        "residual takes N as its one parameter"
    );
    assert!(
        !has_br_table(&r.funcs[0]),
        "dispatch folded (savedpc renamed) — no residual br_table"
    );
    // Rolled: block count is fixed (one specialization), not one unrolled copy per iteration.
    assert!(
        r.funcs[0].blocks.len() <= 40,
        "the loop rolled (got {} blocks)",
        r.funcs[0].blocks.len()
    );

    // Byte-identical to the interpreter across N, on all three backends. Correct output proves `N`
    // entered as a genuine dynamic parameter (a baked constant would give one fixed answer).
    for n in [0i64, 1, 2, 4, 5, 6, 20, 100, 1000, 100_000] {
        let want = expected(n);
        assert_eq!(
            run_oracle(&m, n),
            Ok(vec![Value::I64(want)]),
            "interp @ {n}"
        );
        assert_eq!(tw(&r, 0, &[n]), want, "tree-walk residual @ {n}");
        assert_eq!(bc(&r, 0, &[n]), want, "bytecode residual @ {n}");
        assert_eq!(jit(&r, 0, &[n]), want, "temen-jit residual @ {n}");
    }
}
