//! **General (non-rotate) vector funnel shift** (`llvm.fshl.v4i32` with distinct operands), run on svm.
//!
//! The on-ramp already lowered the *rotate idiom* (`fshl(a, a, s)` → per-lane `Rotl`) and a *scalar*
//! general funnel shift. The **vector** general form — `fshl(a, b, s)` with `a != b` — stayed
//! fail-closed. Rust's `std` float formatter (`flt2dec`/dragon4, u128 bignum) emits exactly this as
//! `llvm.fshl.v4i32`, which is what blocked a *float-capable* `svm-leng` from running on svm (W5 §3e
//! path (a)). This pins the fix directly with a hand-written `.ll`: a `<4 x i32>` funnel shift with
//! constant `a`/`b` vectors and a runtime splat amount, differentially checked interp == JIT == native
//! across amounts that drive both the general path and its `s == 0` width edge.
//!
//! Gated only on the in-house textual `.ll` reader (no on-ramp toolchain needed).

use std::path::PathBuf;
use svm_ir::ValType;
use svm_jit::JitOutcome;

fn assemble(name: &str, ll: &str) -> PathBuf {
    let dir = std::env::temp_dir();
    let llp = dir.join(format!("svm_vfsh_{}_{}.ll", std::process::id(), name));
    std::fs::write(&llp, ll).expect("write .ll");
    llp
}

// The constant lane vectors the guest funnel-shifts. Distinct `a`/`b` (so this is the general path,
// not a rotate), spanning sign bits and zero.
const A: [u32; 4] = [0x0000_0001, 0xffff_ffff, 0x1234_5678, 0xcafe_babe];
const B: [u32; 4] = [0xdead_beef, 0x0000_0002, 0x0000_0000, 0xffff_ffff];

/// `run(sp, s) -> i32`: `fshl.v4i32(A, B, splat s)`, then xor-reduce the four lanes.
const FSHL_LL: &str = r#"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

declare <4 x i32> @llvm.fshl.v4i32(<4 x i32>, <4 x i32>, <4 x i32>)

define i32 @run(i32 %s) {
entry:
  %s0 = insertelement <4 x i32> undef, i32 %s, i32 0
  %sv = shufflevector <4 x i32> %s0, <4 x i32> undef, <4 x i32> zeroinitializer
  %r = call <4 x i32> @llvm.fshl.v4i32(
         <4 x i32> <i32 1, i32 -1, i32 305419896, i32 -889275714>,
         <4 x i32> <i32 -559038737, i32 2, i32 0, i32 -1>,
         <4 x i32> %sv)
  %r0 = extractelement <4 x i32> %r, i32 0
  %r1 = extractelement <4 x i32> %r, i32 1
  %r2 = extractelement <4 x i32> %r, i32 2
  %r3 = extractelement <4 x i32> %r, i32 3
  %x0 = xor i32 %r0, %r1
  %x1 = xor i32 %x0, %r2
  %x2 = xor i32 %x1, %r3
  ret i32 %x2
}
"#;

/// The native oracle — a per-lane general funnel shift, xor-reduced (matches the guest exactly).
fn oracle(s: u32) -> i32 {
    let sl = s & 31;
    let mut x = 0u32;
    for i in 0..4 {
        let r = if sl == 0 {
            A[i]
        } else {
            (A[i] << sl) | (B[i] >> (32 - sl))
        };
        x ^= r;
    }
    x as i32
}

#[test]
fn general_vector_funnel_shift_runs_on_svm() {
    let bc = assemble("fshl", FSHL_LL);
    let t = svm_llvm::translate_ll_path(&bc).expect("translate general vector funnel shift");
    let module = t.module;
    svm_verify::verify_module(&module).expect("verify translated IR");

    let entry = module
        .funcs
        .iter()
        .position(|f| f.params == [ValType::I64, ValType::I32] && f.results == [ValType::I32])
        .expect("run(sp, i32) present") as u32;

    // Amounts chosen to drive both the general body and the `s == 0` width edge (and mod-32 wrap).
    for s in [0u32, 1, 7, 16, 31, 32, 63, 0xffff_ffff] {
        let full = vec![
            svm_interp::Value::I64(t.entry_sp as i64),
            svm_interp::Value::I32(s as i32),
        ];
        let mut fuel = 10_000_000u64;
        let interp = match svm_interp::run(&module, entry, &full, &mut fuel)
            .expect("interp run")
            .as_slice()
        {
            [svm_interp::Value::I32(x)] => *x,
            other => panic!("expected one i32, got {other:?}"),
        };
        let jit =
            match svm_jit::compile_and_run(&module, entry, &[t.entry_sp as i64, s as i32 as i64])
                .expect("jit run")
            {
                JitOutcome::Returned(v) => v[0] as i32,
                other => panic!("unexpected JIT outcome {other:?}"),
            };
        assert_eq!(interp, jit, "§9 interp/JIT parity for s={s:#x}");
        assert_eq!(interp, oracle(s), "§18 svm == native for s={s:#x}");
    }
}
