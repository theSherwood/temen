//! **The cut set: opaque runtime call-outs** ([`SpecConfig::cut_calls`]).
//!
//! Projecting the real `luaV_execute` for an input-taking program is blocked at one wall: to *roll*
//! its dispatch loop (rather than unroll an input-less one) the loop-carried values must be dynamic,
//! which forces a residual branch at the loop exit — and the specializer then has to *project the
//! exit continuation*, which reaches Lua's incremental GC and its `setjmp` error handling. That
//! machinery is deeply stateful and can't fold under a rolled loop.
//!
//! The cut set is the escape hatch: mark those runtime functions (a GC step, an allocator, an error
//! raiser) as **opaque** so the specializer *calls* them instead of *projecting* them. Each cut call
//! is emitted verbatim as a residual `call`, its results are fresh unknowns, and the callee is carried
//! into the residual unspecialized. The dispatch and arithmetic *around* the call still fold, so a
//! hot loop keeps its safepoint call-out as one opaque instruction while everything else collapses —
//! which is what a real fast-path JIT does (guarded fast path + call-outs to the runtime for the cold,
//! stateful work).
//!
//! This is a **prototype**. It models runtime call-outs that don't mutate the specialized register
//! file (a cut callee reads renamed state only via explicit arguments; mutating it back would need the
//! cell threaded out as a result, as outlining does). These tests pin the mechanism against the
//! interpreter oracle: the opaque call survives a rolled loop, the loop still rolls, a private rename
//! cell is preserved across the boundary, results are unknowns, the direct-call closure is carried,
//! and the unsupportable configurations are rejected.
//!
//! Run: `cargo test -p temen-peval --test cut_calls -- --nocapture`

use temen_interp::{Trap, Value};
use temen_ir::{Inst, Module};
use temen_peval::{specialize_with_config, SpecArg, SpecConfig, SpecError};
use temen_text::parse_module;
use temen_verify::verify_module;

fn run(m: &Module, x: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    temen_interp::run(m, 0, &[Value::I64(x)], &mut fuel)
}

fn cut(calls: Vec<u32>) -> SpecConfig {
    SpecConfig {
        cut_calls: calls,
        ..SpecConfig::default()
    }
}

fn n_calls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count()
}
fn n_mem_ops(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Load { .. } | Inst::Store { .. }))
        .count()
}
fn n_muls(f: &temen_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| {
            matches!(
                i,
                Inst::IntBin {
                    op: temen_ir::BinOp::Mul,
                    ..
                }
            )
        })
        .count()
}
/// Assert the residual agrees with the source interpreter over a spread of trip counts.
fn agrees(residual: &Module, source: &Module, trips: &[i64]) {
    for &t in trips {
        assert_eq!(run(residual, t), run(source, t), "diverged at trip={t}");
    }
}

/// A rolled interpreter loop with a per-iteration runtime call-out. `func 0`: `acc = input`, then
/// `for (n = input; n != 0; n -= 1) { gc_step(n); acc += 3 }`, `return acc`. The counter lives in
/// **plain** window memory (addr 8) so it is a dynamic residual load — that is what makes the loop
/// *roll* (constant `pc`, dynamic exit test), exactly as the real dispatch loop does. `func 1` is the
/// opaque safepoint: it does dynamic work on its argument (so inlining it would land real ops in the
/// hot body) and returns a value the loop ignores — the side-effect-only shape of a GC step.
const LOOP_WITH_CALLOUT: &str = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v8 = i64.const 8
  i64.store v8 v0
  v16 = i64.const 16
  vz = i64.const 0
  i64.store v16 vz
  br 1()
}
block 1 () {
  v8 = i64.const 8
  vc = i64.load v8
  vzero = i64.const 0
  vcond = i64.ne vc vzero
  br_if vcond 2() 3()
}
block 2 () {
  v8 = i64.const 8
  vc = i64.load v8
  vign = call 1 (vc)
  v16 = i64.const 16
  vacc = i64.load v16
  v3 = i64.const 3
  vacc2 = i64.add vacc v3
  i64.store v16 vacc2
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  br 1()
}
block 3 () {
  v16 = i64.const 16
  vr = i64.load v16
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  vr = i64.mul v0 v7
  return vr
}
}
"#;

#[test]
fn opaque_call_out_lets_a_rolled_loop_keep_its_runtime_call() {
    let m = parse_module(LOOP_WITH_CALLOUT).expect("parse");
    verify_module(&m).expect("source verifies");
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])).expect("specializes");
    verify_module(&r).expect("residual verifies");

    // The callee is carried into the residual, and the loop kept exactly one opaque call to it — the
    // body was neither inlined (which would land the `mul` in `func 0`) nor unrolled.
    assert_eq!(r.funcs.len(), 2, "entry + the carried cut callee");
    assert_eq!(
        n_calls(&r.funcs[0]),
        1,
        "one opaque call-out survives in the hot loop"
    );
    assert_eq!(
        n_muls(&r.funcs[0]),
        0,
        "the callee's body is NOT inlined into the loop"
    );
    assert_eq!(
        n_muls(&r.funcs[1]),
        1,
        "it lives in the carried callee instead"
    );

    // The loop rolled: the residual's block count is fixed, independent of the (dynamic) trip count.
    assert!(
        r.funcs[0].blocks.len() <= 5,
        "a rolled loop is a handful of blocks, not one per trip (got {})",
        r.funcs[0].blocks.len()
    );

    agrees(&r, &m, &[0, 1, 2, 5, 50, 1000]);
}

#[test]
fn without_the_cut_set_the_call_out_is_inlined() {
    // Same program, no cut set: the specializer inlines `gc_step` into every iteration of the rolled
    // loop — so the `mul` lands in the hot body and no `call` remains. Both residuals are correct and
    // agree; the cut is purely about *where the runtime work lives* (one shared opaque call vs. copied
    // into the loop) — which for the real GC/`setjmp` callee is the difference between a call and an
    // un-projectable body.
    let m = parse_module(LOOP_WITH_CALLOUT).expect("parse");
    verify_module(&m).expect("source verifies");

    let inlined =
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &SpecConfig::default()).expect("spec");
    verify_module(&inlined).expect("inlined residual verifies");
    assert_eq!(
        inlined.funcs.len(),
        1,
        "inlining yields a single residual function"
    );
    assert_eq!(n_calls(&inlined.funcs[0]), 0, "the call-out is absorbed");
    assert_eq!(
        n_muls(&inlined.funcs[0]),
        1,
        "its body is now in the hot loop"
    );

    let opaque = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])).expect("spec");
    // Same observable behavior, different residual shape.
    agrees(&inlined, &m, &[0, 3, 7, 40]);
    agrees(&opaque, &m, &[0, 3, 7, 40]);
}

#[test]
fn a_private_rename_cell_survives_the_cut_call() {
    // The crux for the real use case: the loop-carried **register file** stays in SSA across the
    // opaque call-out. Here the accumulator lives in a *private rename region* (addr 16..24), seeded
    // dynamic from the input; the counter stays in plain memory (addr 8) to drive the roll. Under the
    // private promise the cut call preserves the renamed cell, so it never spills to the window — it
    // threads through the loop as a block parameter, straight across the opaque call.
    let src = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v8 = i64.const 8
  i64.store v8 v0
  v16 = i64.const 16
  i64.store v16 v0
  br 1()
}
block 1 () {
  v8 = i64.const 8
  vc = i64.load v8
  vzero = i64.const 0
  vcond = i64.ne vc vzero
  br_if vcond 2() 3()
}
block 2 () {
  v8 = i64.const 8
  vc = i64.load v8
  vign = call 1 (vc)
  v16 = i64.const 16
  vacc = i64.load v16
  v3 = i64.const 3
  vacc2 = i64.add vacc v3
  i64.store v16 vacc2
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  br 1()
}
block 3 () {
  v16 = i64.const 16
  vr = i64.load v16
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  vr = i64.mul v0 v7
  return vr
}
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        cut_calls: vec![1],
        rename: Some((16, 24)),
        rename_is_private: true,
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    assert_eq!(n_calls(&r.funcs[0]), 1, "one opaque call-out survives");
    // The accumulator region is SSA-lifted: *every* residual memory op in the entry belongs to the
    // plain counter (addr 8) — two stores (init, decrement) and two loads (the header test, the body
    // reload). The renamed accumulator never touches the window, not even to spill around the opaque
    // call: it threads through the loop (and straight across the cut) as a block parameter.
    assert_eq!(
        n_mem_ops(&r.funcs[0]),
        4,
        "only the plain counter's 4 mem ops remain; the renamed accumulator is fully SSA-lifted \
         across the opaque call"
    );

    agrees(&r, &m, &[0, 1, 4, 25, 250]);
}

#[test]
fn a_cut_result_is_an_opaque_unknown() {
    // `func 0(x) = { y = opaque(x); return y * y }`. The cut result is a fresh unknown, so the
    // dependent `y*y` cannot fold to a constant even though `x` is... dynamic here — the point is the
    // residual computes `opaque(x)` then squares *that value*, never peeking inside the callee.
    let src = r#"
memory 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  vy = call 1 (v0)
  vr = i64.mul vy vy
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v100 = i64.const 100
  vr = i64.add v0 v100
  return vr
}
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert_eq!(n_calls(&r.funcs[0]), 1, "the opaque call is kept");
    assert_eq!(
        n_muls(&r.funcs[0]),
        1,
        "the square is computed on the call's (unknown) result"
    );
    // (x+100)^2 for a spread of inputs.
    for x in [0i64, 1, 3, -5, 12] {
        assert_eq!(run(&r, x), run(&m, x));
        assert_eq!(run(&r, x), Ok(vec![Value::I64((x + 100) * (x + 100))]));
    }
}

#[test]
fn cut_carries_the_whole_direct_call_closure() {
    // A cut callee (`func 1`) that itself calls `func 2`: both must be carried, with `func 1`'s call
    // remapped to `func 2`'s residual index. `func 0(x) = opaque1(x)`, `opaque1(x) = opaque2(x) + 1`,
    // `opaque2(x) = x * 3`.
    let src = r#"
memory 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = call 1 (v0)
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vt = call 2 (v0)
  v1 = i64.const 1
  vr = i64.add vt v1
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v3 = i64.const 3
  vr = i64.mul v0 v3
  return vr
}
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    // Only `func 1` is named as a cut root; `func 2` is pulled in by the closure.
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert_eq!(r.funcs.len(), 3, "entry + the two carried functions");
    for x in [0i64, 1, 7, -4] {
        assert_eq!(run(&r, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&r, x), Ok(vec![Value::I64(x * 3 + 1)]));
    }
}

#[test]
fn cut_set_rejects_unsupportable_configs() {
    // (1) A cut callee that dispatches through a `call.dyn` can't be carried: its runtime target
    // index refers to the *source* function numbering, which the residual renumbers — so carrying it
    // would silently misdispatch. Rejected rather than miscompiled.
    let indirect = r#"
memory 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = call 1 (v0)
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vsel = i32.const 2
  vr = call.dyn (i64) -> (i64) vsel (v0)
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
}
}
"#;
    let m = parse_module(indirect).expect("parse");
    verify_module(&m).expect("source verifies");
    assert_eq!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])),
        Err(SpecError::Unsupported),
        "a cut callee using call.dyn is rejected"
    );

    // (2) Cut + outlining is rejected: the outline driver numbers residual functions differently than
    // the inline carry (entry 0, carried at 1..).
    let m2 = parse_module(LOOP_WITH_CALLOUT).expect("parse");
    let both = SpecConfig {
        cut_calls: vec![1],
        outline_calls: true,
        ..SpecConfig::default()
    };
    assert_eq!(
        specialize_with_config(&m2, 0, &[SpecArg::Dynamic], &both),
        Err(SpecError::Unsupported),
        "cut + outline is rejected"
    );

    // (3) A cut call reached with live rename cells but *without* the private promise is rejected:
    // an opaque callee might alias the region, and neither preserving nor dropping the cell is sound
    // (a dropped in-region cell would later read the seed, losing the write). Same program as the
    // private-cell test but with `rename_is_private` off.
    let unpriv = SpecConfig {
        cut_calls: vec![1],
        rename: Some((16, 24)),
        rename_is_private: false,
        ..SpecConfig::default()
    };
    let m3 = parse_module(LOOP_WITH_CALLOUT).expect("parse");
    assert_eq!(
        specialize_with_config(&m3, 0, &[SpecArg::Dynamic], &unpriv),
        Err(SpecError::Unsupported),
        "a cut call with live non-private region cells is rejected"
    );
}

#[test]
fn cut_keeps_a_tail_call_opaque() {
    // A `return runtime_bail(...)` tail-call kept opaque: emitted as a residual `return_call` to the
    // carried callee, not inlined. Models an interpreter that checkpoints its state and tail-calls a
    // resume/slow-path routine on a cold edge. entry(x) = tail-call rt(x + 10); rt(y) = y * 2.
    let src = r#"
memory 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  v10 = i64.const 10
  va = i64.add v0 v10
  return_call 1 (va)
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v2 = i64.const 2
  vr = i64.mul v0 v2
  return vr
}
}
"#;
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cut(vec![1])).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert_eq!(r.funcs.len(), 2, "entry + the carried tail callee");
    assert_eq!(
        r.funcs[0]
            .blocks
            .iter()
            .filter(|b| matches!(b.term, temen_ir::Terminator::ReturnCall { .. }))
            .count(),
        1,
        "the tail call stays opaque as a residual return_call"
    );
    assert_eq!(n_muls(&r.funcs[0]), 0, "the callee body is not inlined");
    for x in [0i64, 1, 5, -3, 100] {
        assert_eq!(run(&r, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&r, x), Ok(vec![Value::I64((x + 10) * 2)]));
    }
}
