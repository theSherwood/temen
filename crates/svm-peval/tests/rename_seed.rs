//! The mutable-VM-state rename feature: **seed rename cells from the constant image** (slice A) and
//! **write the live cells back on exit** (slice B).
//!
//! The plain `rename` region is zero-init scratch — an untouched renamed cell reads 0, and writes are
//! elided entirely. That is wrong for a region that aliases *real* captured memory (a `lua_State`, a
//! `CallInfo`): its fields must fold to their captured values (not zero) for the guards they drive to
//! fold, and any field the interpreter mutates must reach memory so a later reader (Lua's `poscall`)
//! sees it. With `rename_seed_from_image` an untouched cell reads its seed from the constant-memory
//! sources (here a writable data segment declared constant via `const_regions`) while writes shadow
//! it, and every written cell is spilled back to the window before the residual returns. These tests
//! pin all of it against the interpreter oracle — seeded reads, write-shadows-seed, and post-call
//! memory — and check the residual keeps only the loads/stores it must.

use svm_interp::{Trap, Value};
use svm_ir::{Data, Module};
use svm_peval::{specialize_with_config, SpecArg, SpecConfig};
use svm_text::parse_module;
use svm_verify::verify_module;

fn run(m: &Module, x: i64) -> Result<Vec<Value>, Trap> {
    let mut fuel = 1_000_000u64;
    svm_interp::run(m, 0, &[Value::I64(x)], &mut fuel)
}

/// Attach a **writable** data segment holding `val` (little-endian i64) at window address `at`. It is
/// writable (not RO-protected) because a live VM-state region is mutated in place; the bytes are
/// declared constant *at specialization time* via `const_regions`, which is the actual seed source.
fn with_seed(mut m: Module, at: u64, val: i64) -> Module {
    m.data.push(Data {
        offset: at,
        readonly: false,
        bytes: val.to_le_bytes().to_vec(),
    });
    m
}

fn count<F: Fn(&svm_ir::Inst) -> bool>(m: &Module, pred: F) -> usize {
    m.funcs
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.insts)
        .filter(|i| pred(i))
        .count()
}
fn n_loads(m: &Module) -> usize {
    count(m, |i| matches!(i, svm_ir::Inst::Load { .. }))
}
fn n_stores(m: &Module) -> usize {
    count(m, |i| matches!(i, svm_ir::Inst::Store { .. }))
}

const SEED: i64 = 0x1234_5678;
const AT: u64 = 128;

#[test]
fn untouched_seeded_cell_reads_the_image_not_zero() {
    // f(x) = *S + x, where S is a renamed cell seeded from the constant image. Renaming it must fold to the
    // seed (0x12345678), matching the interpreter — not to the scratch zero.
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  vs = i64.load va
  vr = i64.add vs v0
  return vr
}
}
";
    let m = with_seed(parse_module(src).expect("parse"), AT, SEED);
    verify_module(&m).expect("source verifies");

    let cfg = SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_regions: vec![(AT, AT + 8)],
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert_eq!(
        n_loads(&residual) + n_stores(&residual),
        0,
        "a read-only seeded cell is SSA-lifted, leaving no residual load/store (nothing to write back)"
    );

    for x in [0i64, 1, -9, 1000, i64::MAX - SEED] {
        assert_eq!(
            run(&residual, x),
            run(&m, x),
            "diverged from interpreter at x={x}"
        );
        assert_eq!(
            run(&residual, x),
            Ok(vec![Value::I64(x.wrapping_add(SEED))])
        );
    }

    // Without the flag the region is zero-init scratch: the residual would read 0, *not* the seed —
    // i.e. it would diverge from the interpreter. This is exactly the unsoundness the flag fixes.
    let scratch = specialize_with_config(
        &m,
        0,
        &[SpecArg::Dynamic],
        &SpecConfig {
            rename: Some((AT, AT + 8)),
            rename_is_private: true,
            ..SpecConfig::default()
        },
    )
    .expect("specializes");
    assert_eq!(
        run(&scratch, 5),
        Ok(vec![Value::I64(5)]),
        "zero-init scratch reads 0, so f(5) = 5 — the wrong answer the seed fixes"
    );
}

#[test]
fn a_write_shadows_the_seed() {
    // f(x) = { *S = 999; return *S + x }. The store shadows the seed, so the read-back is 999, not
    // the seed — matching the interpreter. (The region is only observed via its return here, so no
    // write-back is needed yet; that is Slice B.)
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  vw = i64.const 999
  i64.store va vw
  vs = i64.load va
  vr = i64.add vs v0
  return vr
}
}
";
    let m = with_seed(parse_module(src).expect("parse"), AT, SEED);
    verify_module(&m).expect("source verifies");

    let cfg = SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        rename_seed_from_image: true,
        const_regions: vec![(AT, AT + 8)],
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    // The read folds (shadowed by the store) → no residual load. The store, into a live-backed
    // region, is spilled once on exit (write-back, slice B) rather than elided.
    assert_eq!(
        n_loads(&residual),
        0,
        "the read should fold to the shadowing store's value"
    );
    assert_eq!(
        n_stores(&residual),
        1,
        "the written live cell is written back once on exit"
    );

    for x in [0i64, 3, -1, 12345] {
        assert_eq!(run(&residual, x), run(&m, x), "diverged at x={x}");
        assert_eq!(run(&residual, x), Ok(vec![Value::I64(x.wrapping_add(999))]));
    }
}

#[test]
fn write_back_makes_post_call_memory_correct() {
    // f(x) = { *S = x; return 7 }. The write into the live-backed region must reach the window so a
    // caller reading S afterward sees x — the property Lua's `poscall` needs (it reads L->top and the
    // return registers luaV_execute leaves behind). We observe it via the captured final window.
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  i64.store va v0
  vc = i64.const 7
  return vc
}
}
";
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("source verifies");
    let cfg = SpecConfig {
        rename: Some((AT, AT + 8)),
        rename_is_private: true,
        rename_seed_from_image: true,
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert_eq!(
        n_stores(&residual),
        1,
        "the write must be spilled back to the window"
    );

    let init = vec![0u8; (AT + 8) as usize];
    for x in [1i64, -5, 0x0BAD_F00D, i64::MIN] {
        let mut f1 = 1_000_000u64;
        let (r_ref, w_ref) = svm_interp::run_capture(&m, 0, &[Value::I64(x)], &mut f1, &init);
        let mut f2 = 1_000_000u64;
        let (r_res, w_res) =
            svm_interp::run_capture(&residual, 0, &[Value::I64(x)], &mut f2, &init);
        assert_eq!(r_res, r_ref, "result diverged at x={x}");
        assert_eq!(w_res, w_ref, "post-call window diverged at x={x}");
        assert_eq!(
            &w_res[AT as usize..AT as usize + 8],
            &x.to_le_bytes(),
            "the written value must be visible in memory after the call (x={x})"
        );
    }

    // Contrast: plain scratch rename (no seed/write-back) elides the store, so the window stays at
    // its pre-call bytes — stale, diverging from the interpreter. This is what write-back fixes.
    let scratch = specialize_with_config(
        &m,
        0,
        &[SpecArg::Dynamic],
        &SpecConfig {
            rename: Some((AT, AT + 8)),
            rename_is_private: true,
            ..SpecConfig::default()
        },
    )
    .expect("specializes");
    assert_eq!(
        n_stores(&scratch),
        0,
        "scratch rename elides the store (no write-back)"
    );
    let mut f = 1_000_000u64;
    let (_, w_scratch) = svm_interp::run_capture(&scratch, 0, &[Value::I64(42)], &mut f, &init);
    assert_eq!(
        &w_scratch[AT as usize..AT as usize + 8],
        &[0u8; 8],
        "without write-back the window keeps its stale pre-call bytes"
    );
}

#[test]
fn two_disjoint_seeded_regions_fold_and_write_back() {
    // Disjoint regions A@128 and B@256 (as a lua_State / stack / CallInfo would be), both seeded and
    // both mutated: f(x) = { a=*A; b=*B; *A=b; *B=a; return a+b+x } — a swap. Reads fold to the two
    // seeds, and the swapped values must be written back to their windows.
    let src = "\
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  va = i64.const 128
  vb = i64.const 256
  a = i64.load va
  b = i64.load vb
  i64.store va b
  i64.store vb a
  s0 = i64.add a b
  s1 = i64.add s0 v0
  return s1
}
}
";
    let (a0, b0) = (100i64, 200i64);
    let mut m = parse_module(src).expect("parse");
    m.data.push(Data {
        offset: 128,
        readonly: false,
        bytes: a0.to_le_bytes().to_vec(),
    });
    m.data.push(Data {
        offset: 256,
        readonly: false,
        bytes: b0.to_le_bytes().to_vec(),
    });
    verify_module(&m).expect("source verifies");

    let cfg = SpecConfig {
        rename: Some((128, 136)),
        rename_extra: vec![(256, 264)],
        rename_is_private: true,
        rename_seed_from_image: true,
        const_regions: vec![(128, 136), (256, 264)],
        ..SpecConfig::default()
    };
    let residual = specialize_with_config(&m, 0, &[SpecArg::Dynamic], &cfg).expect("specializes");
    verify_module(&residual).expect("residual verifies");
    assert_eq!(n_loads(&residual), 0, "both seeded reads fold");
    assert_eq!(n_stores(&residual), 2, "both swapped cells write back");

    let init = vec![0u8; 264];
    for x in [0i64, 7, -3] {
        let mut f1 = 1_000_000u64;
        let (r_ref, w_ref) = svm_interp::run_capture(&m, 0, &[Value::I64(x)], &mut f1, &init);
        let mut f2 = 1_000_000u64;
        let (r_res, w_res) =
            svm_interp::run_capture(&residual, 0, &[Value::I64(x)], &mut f2, &init);
        assert_eq!(r_res, r_ref, "result diverged at x={x}");
        assert_eq!(w_res, w_ref, "post-call window diverged at x={x}");
        assert_eq!(r_res, Ok(vec![Value::I64(a0 + b0 + x)]));
        assert_eq!(&w_res[128..136], &b0.to_le_bytes(), "A holds the swapped B");
        assert_eq!(&w_res[256..264], &a0.to_le_bytes(), "B holds the swapped A");
    }
}
