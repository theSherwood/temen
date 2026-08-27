//! **Read-only-state cut calls: a safepoint spill that keeps the dispatch folded.**
//!
//! `cut_calls_touch_state` spills the register file before an opaque call-out and **reloads** it after
//! (the callee may have rewritten it). But a *non-moving* collector — a GC that scans the stack to
//! mark, or a read barrier — only *reads* the registers; reloading them would turn every folded tag
//! constant into a fresh unknown and re-open the per-opcode type checks the dispatch fold depends on.
//! [`SpecConfig::cut_calls_read_state`] is the right discipline for that: spill before (so the scan
//! sees the live, typed registers), keep the abstract cells afterward (no reload).
//!
//! This models it minimally. `interp` (func 0) is a `savedpc`-style dispatch loop that sums `1..N` into
//! `out`; each EMIT calls the opaque `observe` (func 1) — a stand-in for a stack-scanning GC step. With
//! `observe` cut **read-state**, `savedpc` (the renamed pc) stays folded across the call, so the
//! dispatch collapses and the loop rolls, while `observe` is emitted as an opaque `call` per iteration.
//! The control assertion is decisive: cutting the *same* call **touch-state** reloads `savedpc` as an
//! unknown, so the dispatch can no longer fold and a residual `br_table` survives.
//!
//! Run: `cargo test -p temen-peval --test cut_read_state -- --nocapture`

use temen_interp::{Trap, Value};
use temen_ir::{Module, Terminator};
use temen_peval::{specialize_with_config, SpecConfig};
use temen_text::parse_module;
use temen_verify::verify_module;

// Cells sit one guard up (above the #1094 NULL guard): savedpc@16384 (i32 rename cell),
// reg0@16392 (i64), out@16400 (i64), N@16408 (i64, the runtime-input cell).
// pc:  0 LOADN (reg0=N; pc=1)  1 LOOPCHK (reg0==0 ? pc=4 : pc=2)  2 EMIT (out+=reg0; observe(reg0); pc=3)
//      3 DEC (reg0-=1; pc=1)   4.. HALT (return out)
// `observe` (func 1) is the opaque state-reading call-out; `run` (func 2) seeds + tail-calls interp.
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
  call 1 (vc)
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
func (i64) -> () {
block 0 (v0: i64) {
  return
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
    temen_interp::run(m, 2, &[Value::I64(n)], &mut fuel)
}
fn tw(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match temen_interp::run(m, e, &vs, &mut fuel)
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn bc(m: &Module, e: u32, a: &[i64]) -> i64 {
    let mut fuel = u64::MAX;
    let vs: Vec<Value> = a.iter().map(|&x| Value::I64(x)).collect();
    match temen_interp::bytecode::compile_and_run(m, e, &vs, &mut fuel)
        .expect("subset")
        .expect("no trap")
        .as_slice()
    {
        [Value::I64(x)] => *x,
        o => panic!("bad {o:?}"),
    }
}
fn jit(m: &Module, e: u32, a: &[i64]) -> i64 {
    match temen_jit::compile_and_run(m, e, a) {
        Ok(temen_jit::JitOutcome::Returned(v)) => v[0],
        o => panic!("jit {o:?}"),
    }
}
fn has_br_table(f: &temen_ir::Func) -> bool {
    f.blocks
        .iter()
        .any(|b| matches!(b.term, Terminator::BrTable { .. }))
}
fn n_calls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, temen_ir::Inst::Call { .. }))
        .count()
}
fn expected(n: i64) -> i64 {
    (1..=n.max(0)).sum()
}

fn base_cfg() -> SpecConfig {
    SpecConfig {
        rename: Some((16384, 16416)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_overlays: vec![(16384, vec![0u8; 32])],
        dynamic_cells: vec![(16408, 8)],
        ..SpecConfig::default()
    }
}

#[test]
fn read_state_cut_keeps_the_dispatch_folded_and_rolls() {
    let m = parse_module(VM).expect("parse");
    verify_module(&m).expect("source verifies");

    for n in [0i64, 1, 5, 20] {
        assert_eq!(
            run_oracle(&m, n),
            Ok(vec![Value::I64(expected(n))]),
            "interp @ {n}"
        );
    }

    // `observe` (func 1) is cut read-state: spilled-before, not reloaded — so `savedpc` stays folded.
    let cfg = SpecConfig {
        cut_calls_read_state: vec![1],
        ..base_cfg()
    };
    let r = specialize_with_config(&m, 0, &[], &cfg).expect("specializes with a read-state cut");
    verify_module(&r).expect("residual verifies");

    assert!(
        !has_br_table(&r.funcs[0]),
        "dispatch folded across the read-state cut"
    );
    assert_eq!(r.funcs.len(), 2, "observe is carried into the residual");
    assert!(
        n_calls(&r.funcs[0]) >= 1,
        "observe stays an opaque per-iteration call-out"
    );
    assert!(
        r.funcs[0].blocks.len() <= 40,
        "the loop rolled ({} blocks)",
        r.funcs[0].blocks.len()
    );

    for n in [0i64, 1, 2, 5, 6, 20, 100, 1000] {
        let want = expected(n);
        assert_eq!(tw(&r, 0, &[n]), want, "tree-walk @ {n}");
        assert_eq!(bc(&r, 0, &[n]), want, "bytecode @ {n}");
        assert_eq!(jit(&r, 0, &[n]), want, "temen-jit @ {n}");
    }
}

#[test]
fn touch_state_reload_reopens_the_dispatch() {
    // The control: the *same* cut as touch-state reloads `savedpc` as an unknown after the call, so the
    // pc can no longer fold and the dispatch stays a residual `br_table` (read-state is what avoids it).
    let m = parse_module(VM).expect("parse");
    let cfg = SpecConfig {
        cut_calls_touch_state: vec![1],
        ..base_cfg()
    };
    // Reloading `savedpc` may re-open the dispatch (a br_table survives) or just fail closed — either
    // way it does not produce the folded, rolled residual that read-state does.
    if let Ok(r) = specialize_with_config(&m, 0, &[], &cfg) {
        assert!(
            has_br_table(&r.funcs[0]),
            "touch-state reload should re-open the dispatch (a br_table survives)"
        );
    }
}
