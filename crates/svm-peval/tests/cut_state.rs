//! **Cut calls that touch VM state (slice 1) + guard-and-deopt (slice 2).**
//!
//! These two mechanisms are what let a partial-evaluated interpreter keep a *rolled* fast path while
//! its stateful, cold, un-projectable machinery stays behind a call boundary — the guarded-fast-path
//! JIT discipline. They are deliberately **interpreter-agnostic**: everything below is expressed over
//! the generic rename region and generic `(func, block)` control-flow points, with nothing specific to
//! Lua (or any one interpreter). The tests use small synthetic interpreter-shaped modules.
//!
//! - **Slice 1 — state-touching cut calls** ([`SpecConfig::cut_calls_touch_state`]). A runtime call-out
//!   that reads/writes the interpreter's register file (a GC scanning the stack, a `poscall` rewriting
//!   registers) is kept opaque, but the live rename cells are **spilled to the window before the call
//!   and reloaded after** — the safepoint-spill discipline. The callee observes the current state
//!   through memory and its writes become visible to later renamed accesses.
//!
//! - **Slice 2 — guard-and-deopt** ([`SpecConfig::deopt_targets`] / [`SpecConfig::deopt_handler`]). A
//!   cold-path block the specializer must not project (a type-check failure, an error raise, a GC-
//!   needed slow path) becomes a **deopt**: when a dynamic branch would enter it, the residual spills
//!   all live state to the window (a valid interpreter resume image) and tail-calls a resume handler,
//!   which finishes from that state. Fast paths fold; cold paths bail to the runtime.
//!
//! - **Slice 3 — threading dynamic entry args to a deopt edge**. When the resume handler takes the
//!   entry's arguments (`<entry params> -> R`), the specializer threads the entry's dynamic parameters
//!   through the CFG to every deopt edge and passes them to the handler — so an interpreter whose input
//!   arrives as a parameter (not only via the window) can resume. Both handler shapes coexist.
//!
//! Run: `cargo test -p svm-peval --test cut_state -- --nocapture`

use svm_interp::{Trap, Value};
use svm_ir::{Inst, Module, Terminator};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig, SpecError};
use svm_text::parse_module;
use svm_verify::verify_module;

fn run(m: &Module, x: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 10_000_000u64;
    svm_interp::run(m, 0, &[Value::I64(x)], &mut fuel)
}
fn n_calls(f: &svm_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| matches!(i, Inst::Call { .. }))
        .count()
}
fn n_return_calls(f: &svm_ir::Func) -> usize {
    f.blocks
        .iter()
        .filter(|b| matches!(b.term, Terminator::ReturnCall { .. }))
        .count()
}
fn agrees(residual: &Module, source: &Module, xs: &[i64]) {
    for &x in xs {
        assert_eq!(run(residual, x), run(source, x), "diverged at x={x}");
    }
}

// =============================================================================================
// Slice 1: a runtime call-out that reads and mutates the register file, kept opaque.
// =============================================================================================

// counter@8 (plain memory → dynamic, drives the roll), accumulator@16 (private rename, seeded
// dynamic from the input). Each iteration an opaque *state-touching* callee `bump()` does `*16 += 5`
// through memory; the specializer spills @16 before the call and reloads it after, so the mutation is
// seen. return *16 = x + 5*x = 6*x.
const LOOP_WITH_SAFEPOINT: &str = r#"
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
  call 1 ()
  v8 = i64.const 8
  vc = i64.load v8
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
func () -> () {
block 0 () {
  v16 = i64.const 16
  vacc = i64.load v16
  v5 = i64.const 5
  vacc2 = i64.add vacc v5
  i64.store v16 vacc2
  return
}
}
"#;

#[test]
fn a_state_touching_cut_call_reads_and_mutates_the_register_file() {
    let m = parse_module(LOOP_WITH_SAFEPOINT).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        cut_calls_touch_state: vec![1],
        rename: Some((16, 24)),
        rename_is_private: true,
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    assert_eq!(r.funcs.len(), 2, "entry + the carried safepoint callee");
    assert_eq!(
        n_calls(&r.funcs[0]),
        1,
        "one opaque safepoint call survives the rolled loop"
    );
    // The renamed accumulator is spilled before the opaque call and reloaded after — so the residual's
    // hot loop has a store/load pair around the call (the safepoint spill), where the pure-opaque cut
    // of PR #617 kept the cell entirely in SSA. That spill is what lets the callee mutate the register.
    let (loads, stores) = r.funcs[0].blocks.iter().flat_map(|b| &b.insts).fold(
        (0usize, 0usize),
        |(l, s), i| match i {
            Inst::Load { .. } => (l + 1, s),
            Inst::Store { .. } => (l, s + 1),
            _ => (l, s),
        },
    );
    assert!(
        stores >= 1 && loads >= 1,
        "the register file spills and reloads around the safepoint"
    );

    for x in [0i64, 1, 3, 10, 100, 1000] {
        assert_eq!(run(&r, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&r, x), Ok(vec![Value::I64(6 * x)]), "x={x}");
    }
}

// =============================================================================================
// Slice 2: guard-and-deopt — a cold branch bails to the resume handler instead of being projected.
// =============================================================================================

// A mini-interpreter: input x is stored into the VM window@16 (renamed, dynamic), then read back and
// dispatched. Fast path (x<100): result = x*2, folds to a residual multiply. Cold path (block 3, a
// DEOPT TARGET): the specializer must NOT project it — it spills state to the window and tail-calls
// the `() -> i64` resume handler (func 1), which reads x@16 and finishes (x+7), exactly as block 3
// would. This models "guard passes → stay in the fast path; guard fails → deopt to the interpreter."
const FAST_PATH_WITH_DEOPT: &str = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v16 = i64.const 16
  i64.store v16 v0
  br 1()
}
block 1 () {
  v16 = i64.const 16
  vx = i64.load v16
  v100 = i64.const 100
  vlt = i64.lt_s vx v100
  br_if vlt 2() 3()
}
block 2 () {
  v16 = i64.const 16
  vx = i64.load v16
  v2 = i64.const 2
  vr = i64.mul vx v2
  return vr
}
block 3 () {
  v16 = i64.const 16
  vx = i64.load v16
  v7 = i64.const 7
  vr = i64.add vx v7
  return vr
}
}
func () -> (i64) {
block 0 () {
  v16 = i64.const 16
  vx = i64.load v16
  v7 = i64.const 7
  vr = i64.add vx v7
  return vr
}
}
"#;

#[test]
fn a_cold_branch_deopts_to_the_resume_handler() {
    let m = parse_module(FAST_PATH_WITH_DEOPT).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_targets: vec![(0, 3)],
        deopt_handler: Some(1),
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    assert_eq!(r.funcs.len(), 2, "entry + the carried resume handler");
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "the cold edge deopts via a tail-call to the handler"
    );
    // The cold path's body (`x + 7`) is NOT projected into the entry — it stays in the carried handler.
    let adds_in_entry = r.funcs[0]
        .blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| {
            matches!(
                i,
                Inst::IntBin {
                    op: svm_ir::BinOp::Add,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        adds_in_entry, 0,
        "the cold path is not projected — only the fast path folds"
    );

    for x in [0i64, 1, 50, 99, 100, 101, 250, -5] {
        assert_eq!(run(&r, x), run(&m, x), "diverged at x={x}");
        let want = if x < 100 { x * 2 } else { x + 7 };
        assert_eq!(run(&r, x), Ok(vec![Value::I64(want)]), "x={x}");
    }
}

#[test]
fn deopt_requires_a_well_typed_handler() {
    let m = parse_module(FAST_PATH_WITH_DEOPT).expect("parse");
    verify_module(&m).expect("source verifies");
    // A deopt target with no handler is rejected.
    let no_handler = SpecConfig {
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_targets: vec![(0, 3)],
        deopt_handler: None,
        ..SpecConfig::default()
    };
    assert_eq!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &no_handler),
        Err(SpecError::Unsupported),
        "deopt targets without a handler are rejected"
    );
    // A handler whose signature isn't `() -> <entry results>` is rejected (func 0 takes an i64 param).
    let bad_sig = SpecConfig {
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_targets: vec![(0, 3)],
        deopt_handler: Some(0),
        ..SpecConfig::default()
    };
    assert_eq!(
        specialize_with_config(&m, 0, &[SpecArg::Dynamic], &bad_sig),
        Err(SpecError::Unsupported),
        "a handler that isn't `() -> <entry results>` is rejected"
    );
}

// =============================================================================================
// Composition: a rolled fast path with BOTH a per-iteration safepoint call-out AND a deopt guard.
// This is the shape a partial-evaluated interpreter fast path actually wants.
// =============================================================================================

// counter@8 (plain, dynamic → rolls), accumulator@16 (private rename, seeded dynamic = input). The
// loop runs a state-touching safepoint `bump()` (`*16 += 3`) each iteration; on exit a guard checks
// the accumulator and, when it is "too large" (>= 1000, the cold case), DEOPTS to the resume handler
// (func 2, `*16 + 1`) instead of projecting the cold block. With input v0: accumulator = v0 + 3*v0 =
// 4*v0, so v0 < 250 stays on the fast path (return 4*v0) and v0 >= 250 deopts (return 4*v0 + 1).
const ROLLED_FASTPATH: &str = r#"
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
  call 1 ()
  v8 = i64.const 8
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  br 1()
}
block 3 () {
  v16 = i64.const 16
  vacc = i64.load v16
  v1000 = i64.const 1000
  vlt = i64.lt_s vacc v1000
  br_if vlt 4() 5()
}
block 4 () {
  v16 = i64.const 16
  vr = i64.load v16
  return vr
}
block 5 () {
  v16 = i64.const 16
  vacc = i64.load v16
  v1 = i64.const 1
  vr = i64.add vacc v1
  return vr
}
}
func () -> () {
block 0 () {
  v16 = i64.const 16
  vacc = i64.load v16
  v3 = i64.const 3
  vacc2 = i64.add vacc v3
  i64.store v16 vacc2
  return
}
}
func () -> (i64) {
block 0 () {
  v16 = i64.const 16
  vacc = i64.load v16
  v1 = i64.const 1
  vr = i64.add vacc v1
  return vr
}
}
"#;

#[test]
fn a_rolled_fast_path_composes_a_safepoint_call_out_and_a_deopt_guard() {
    let m = parse_module(ROLLED_FASTPATH).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        cut_calls_touch_state: vec![1],
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_targets: vec![(0, 5)],
        deopt_handler: Some(2),
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    assert_eq!(
        r.funcs.len(),
        3,
        "entry + the safepoint callee + the resume handler"
    );
    assert_eq!(
        n_calls(&r.funcs[0]),
        1,
        "the per-iteration safepoint call-out survives the rolled loop"
    );
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "the cold guard edge deopts to the handler"
    );
    // The loop rolled: block count is fixed, independent of the dynamic trip count.
    assert!(
        r.funcs[0].blocks.len() <= 7,
        "a rolled fast path is a handful of blocks (got {})",
        r.funcs[0].blocks.len()
    );

    agrees(&r, &m, &[0, 1, 100, 249, 250, 251, 400]);
    for x in [0i64, 1, 249, 250, 400] {
        let want = if 4 * x < 1000 { 4 * x } else { 4 * x + 1 };
        assert_eq!(run(&r, x), Ok(vec![Value::I64(want)]), "x={x}");
    }
}

// =============================================================================================
// Slice 3: threading the entry's *dynamic argument* to a deopt edge.
//
// The interpreter's input `n` arrives as a function PARAMETER, not pre-seeded in the window, and the
// resume handler is `(i64) -> i64` — it must receive `n` directly. `n` is the loop trip count (plain
// memory@8, dynamic → the loop rolls) and is also threaded as a live value `vn` through the loop to
// the exit block, which is a DEOPT TARGET reached only after the rolled loop. So `n` must survive the
// whole rolled loop and land on the deopt edge. Source: loop `n` times, then `return n*100`; handler:
// `n*100`. Both equal, so the residual (which replaces the exit block with `handler(n)`) matches the
// interpreter — and only if `n` was correctly threaded to the deopt.
// =============================================================================================
const THREADED_ARG_DEOPT: &str = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v8 = i64.const 8
  i64.store v8 v0
  br 1(v0)
}
block 1 (vn: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  vzero = i64.const 0
  vcond = i64.ne vc vzero
  br_if vcond 2(vn) 3(vn)
}
block 2 (vn: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  br 1(vn)
}
block 3 (vn: i64) {
  v100 = i64.const 100
  vr = i64.mul vn v100
  return vr
}
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v100 = i64.const 100
  vr = i64.mul v0 v100
  return vr
}
}
"#;

#[test]
fn a_deopt_edge_receives_the_threaded_entry_argument() {
    let m = parse_module(THREADED_ARG_DEOPT).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        deopt_targets: vec![(0, 3)],
        deopt_handler: Some(1), // (i64) -> i64 — receives the threaded entry arg `n`
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert_eq!(r.funcs.len(), 2, "entry + the carried arg-taking handler");
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "the exit deopts, tail-calling handler(n) — n threaded through the rolled loop"
    );
    // The loop rolled: fixed block count despite the dynamic trip count.
    assert!(
        r.funcs[0].blocks.len() <= 5,
        "the loop rolled with the arg threaded (got {} blocks)",
        r.funcs[0].blocks.len()
    );

    // Differential: the residual deopts to handler(n) = n*100, matching the source (loop then n*100),
    // for every input — proving n reached the deopt edge after surviving the whole rolled loop.
    for n in [0i64, 1, 2, 5, 37, 250] {
        assert_eq!(run(&r, n), run(&m, n), "diverged at n={n}");
        assert_eq!(run(&r, n), Ok(vec![Value::I64(n * 100)]), "n={n}");
    }
}

// =============================================================================================
// Slice 4: edge-level deopt — deopt ONE incoming edge of a SHARED block, leaving the block
// projected for its other predecessors.
//
// This is the shape the real `OP_CALL` divergence needs: an interpreter's dispatch block is
// reached from many opcodes; one cold opcode arm re-enters dispatch in a way that diverges
// (the Lua-function re-dispatch after a C-function call), but every other edge into dispatch is
// a live fast path that must still be projected. `deopt_targets` can't express that — it would
// deopt the whole block, killing the fast paths too. `deopt_edges` deopts the single
// `(from_block -> to_block)` CFG edge.
//
// Toy: input x → window@16. Block 1 splits (dynamic) into a fast pred (block 2) and a cold pred
// (block 3); BOTH branch to the shared block 4, which returns `x*3`. We mark ONLY the cold edge
// `(3 -> 4)` as a deopt edge. So x<50 flows 2→4 and PROJECTS block 4 (a residual multiply
// survives), while x>=50 flows 3→4 and DEOPTS to the handler (`() -> i64`, also `x*3`). Both
// arms compute x*3, so the residual matches the source for every input — and the projected
// multiply proves the shared block was NOT deopted wholesale, only the one edge.
const SHARED_BLOCK_EDGE_DEOPT: &str = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v16 = i64.const 16
  i64.store v16 v0
  br 1()
}
block 1 () {
  v16 = i64.const 16
  vx = i64.load v16
  v50 = i64.const 50
  vlt = i64.lt_s vx v50
  br_if vlt 2() 3()
}
block 2 () {
  br 4()
}
block 3 () {
  br 4()
}
block 4 () {
  v16 = i64.const 16
  vx = i64.load v16
  v3 = i64.const 3
  vr = i64.mul vx v3
  return vr
}
}
func () -> (i64) {
block 0 () {
  v16 = i64.const 16
  vx = i64.load v16
  v3 = i64.const 3
  vr = i64.mul vx v3
  return vr
}
}
"#;

fn n_muls(f: &svm_ir::Func) -> usize {
    f.blocks
        .iter()
        .flat_map(|b| &b.insts)
        .filter(|i| {
            matches!(
                i,
                Inst::IntBin {
                    op: svm_ir::BinOp::Mul,
                    ..
                }
            )
        })
        .count()
}

#[test]
fn one_edge_of_a_shared_block_deopts_while_the_block_still_projects() {
    let m = parse_module(SHARED_BLOCK_EDGE_DEOPT).expect("parse");
    verify_module(&m).expect("source verifies");

    // Edge deopt: only the cold edge (3 -> 4) bails; the fast edge (2 -> 4) projects block 4.
    let cfg = SpecConfig {
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_edges: vec![(0, 3, 4)],
        deopt_handler: Some(1),
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");

    assert_eq!(r.funcs.len(), 2, "entry + the carried resume handler");
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "exactly the one cold edge deopts via a tail-call to the handler"
    );
    assert_eq!(
        n_muls(&r.funcs[0]),
        1,
        "the shared block is still PROJECTED via the fast edge (its multiply survives) — \
         edge-deopt did not kill the whole block"
    );

    for x in [0i64, 1, 49, 50, 51, 100, -7] {
        assert_eq!(run(&r, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&r, x), Ok(vec![Value::I64(x * 3)]), "x={x}");
    }

    // Contrast: deopt_targets on the same block (0, 4) deopts BOTH edges — the multiply is gone.
    let cfg_target = SpecConfig {
        rename: Some((16, 24)),
        rename_is_private: true,
        deopt_targets: vec![(0, 4)],
        deopt_handler: Some(1),
        ..SpecConfig::default()
    };
    let rt = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg_target).expect("specializes");
    verify_module(&rt).expect("residual verifies");
    assert_eq!(
        n_muls(&rt.funcs[0]),
        0,
        "target-deopt kills the whole block — no projected multiply survives (the edge vs target distinction)"
    );
    for x in [0i64, 49, 50, 100] {
        assert_eq!(run(&rt, x), run(&m, x), "target-deopt diverged at x={x}");
    }
}

#[test]
fn a_no_arg_handler_still_resumes_from_the_window() {
    // The `() -> R` shape is unchanged by arg threading: same program, but the handler ignores args
    // and returns a constant. (Guards the two handler shapes don't interfere.)
    let m = parse_module(THREADED_ARG_DEOPT).expect("parse");
    let src_noarg = r#"
memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  v8 = i64.const 8
  i64.store v8 v0
  br 1(v0)
}
block 1 (vn: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  vzero = i64.const 0
  vcond = i64.ne vc vzero
  br_if vcond 2(vn) 3(vn)
}
block 2 (vn: i64) {
  v8 = i64.const 8
  vc = i64.load v8
  v1 = i64.const 1
  vc2 = i64.sub vc v1
  i64.store v8 vc2
  br 1(vn)
}
block 3 (vn: i64) {
  v42 = i64.const 42
  return v42
}
}
func () -> (i64) {
block 0 () {
  v42 = i64.const 42
  return v42
}
}
"#;
    let _ = m;
    let mn = parse_module(src_noarg).expect("parse");
    verify_module(&mn).expect("source verifies");
    let cfg = SpecConfig {
        deopt_targets: vec![(0, 3)],
        deopt_handler: Some(1), // () -> i64
        ..SpecConfig::default()
    };
    let r = specialize_with_config(&mn, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&r).expect("residual verifies");
    assert_eq!(
        n_return_calls(&r.funcs[0]),
        1,
        "still deopts via the no-arg handler"
    );
    for n in [0i64, 1, 5, 100] {
        assert_eq!(run(&r, n), run(&mn, n), "diverged at n={n}");
        assert_eq!(run(&r, n), Ok(vec![Value::I64(42)]), "n={n}");
    }
}
