//! #850 — a **coverage + correctness lock** over the on-ramp's integer min/max + bit + saturating
//! intrinsic family (the enumerated `match` in `lower_int_minmax_or_bit_intrinsic`).
//!
//! Motivation: the on-ramp's intrinsic support was discovered reactively, one `std` program at a time
//! (#822/#827/#849). This is the "enumerate + assert coverage" half of #850: a curated corpus of every
//! name in that family, each lowered on a minimal module and **differentialled against the interpreter**
//! (DESIGN §18) with a host-computed expected value. Unlike a gap-*finder*, this is a regression-*lock*:
//! deleting a handler (or breaking its lowering) turns a green corpus red, so the enumeration only
//! changes deliberately. A bogus `llvm.*` name is asserted to **fail closed** (never silently accepted).
//!
//! Scope: this family only — the one with a single self-contained name list. Other intrinsic families
//! (memory, float math, fptosi.sat, va_start, eh.typeid, load.relative) have their own unit tests.

use svm_interp::Value;

const HEADER: &str = "target datalayout = \"e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128\"\n\
     target triple = \"x86_64-unknown-none\"\n";

/// Build a module: a `declare` for the intrinsic, and `@f() -> i64` that calls it (on constant args)
/// and zero-extends the `i32` result to `i64` (what the interpreter hands back).
fn module(decl: &str, call: &str) -> String {
    format!(
        "{HEADER}{decl}\n\
         define i64 @f() {{\nentry:\n  %r = {call}\n  %z = zext i32 %r to i64\n  ret i64 %z\n}}\n"
    )
}

/// Translate + verify + run a single-`i64`-returning module, returning its result or an error string.
fn run_i64(ir: &str) -> Result<i64, String> {
    let t = svm_llvm::translate_ll_str(ir).map_err(|e| format!("translate: {e:?}"))?;
    svm_verify::verify_module(&t.module).map_err(|e| format!("verify: {e:?}"))?;
    let f = t
        .exports
        .iter()
        .find(|(n, _)| n == "f")
        .map(|x| x.1)
        .ok_or("no export `f`")?;
    let mut fuel = 1_000_000u64;
    match svm_interp::run(&t.module, f, &[Value::I64(t.entry_sp as i64)], &mut fuel) {
        Ok(v) => match v[0] {
            Value::I64(x) => Ok(x),
            other => Err(format!("non-i64 result {other:?}")),
        },
        Err(e) => Err(format!("trap: {e:?}")),
    }
}

/// The zero-extended `i32` result the interpreter returns for a host-computed `u32`.
fn zext(v: u32) -> i64 {
    v as i64
}

// The min/max references behind fns so the operand values aren't literal comparisons at the call site
// (clippy's `const_comparisons` would otherwise flag `(-5).max(7)` as "never greater").
fn smax(a: i32, b: i32) -> i32 {
    a.max(b)
}
fn smin(a: i32, b: i32) -> i32 {
    a.min(b)
}
fn umax(a: u32, b: u32) -> u32 {
    a.max(b)
}
fn umin(a: u32, b: u32) -> u32 {
    a.min(b)
}

#[test]
fn integer_minmax_bit_intrinsic_family_lowers_and_computes() {
    // (name, declare, call, expected zext-i64 result). Every name in the family's enumeration; the
    // expected value is computed host-side with Rust's own operation, so this locks correctness too.
    let cases: Vec<(&str, String, String, i64)> = vec![
        (
            "llvm.smax.i32",
            "declare i32 @llvm.smax.i32(i32, i32)".into(),
            "call i32 @llvm.smax.i32(i32 -5, i32 7)".into(),
            zext(smax(-5, 7) as u32),
        ),
        (
            "llvm.smin.i32",
            "declare i32 @llvm.smin.i32(i32, i32)".into(),
            "call i32 @llvm.smin.i32(i32 -5, i32 7)".into(),
            zext(smin(-5, 7) as u32),
        ),
        (
            "llvm.umax.i32",
            "declare i32 @llvm.umax.i32(i32, i32)".into(),
            "call i32 @llvm.umax.i32(i32 5, i32 7)".into(),
            zext(umax(5, 7)),
        ),
        (
            "llvm.umin.i32",
            "declare i32 @llvm.umin.i32(i32, i32)".into(),
            "call i32 @llvm.umin.i32(i32 5, i32 7)".into(),
            zext(umin(5, 7)),
        ),
        (
            "llvm.ctlz.i32",
            "declare i32 @llvm.ctlz.i32(i32, i1)".into(),
            "call i32 @llvm.ctlz.i32(i32 16, i1 false)".into(),
            zext(16u32.leading_zeros()),
        ),
        (
            "llvm.cttz.i32",
            "declare i32 @llvm.cttz.i32(i32, i1)".into(),
            "call i32 @llvm.cttz.i32(i32 16, i1 false)".into(),
            zext(16u32.trailing_zeros()),
        ),
        (
            "llvm.ctpop.i32",
            "declare i32 @llvm.ctpop.i32(i32)".into(),
            "call i32 @llvm.ctpop.i32(i32 61680)".into(), // 0xF0F0
            zext(0xF0F0u32.count_ones()),
        ),
        (
            "llvm.abs.i32",
            "declare i32 @llvm.abs.i32(i32, i1)".into(),
            "call i32 @llvm.abs.i32(i32 -5, i1 false)".into(),
            zext((-5i32).unsigned_abs()),
        ),
        (
            "llvm.fshl.i32",
            "declare i32 @llvm.fshl.i32(i32, i32, i32)".into(),
            // Rotate idiom (a == b): fshl(x, x, c) == x.rotate_left(c).
            "call i32 @llvm.fshl.i32(i32 305419896, i32 305419896, i32 4)".into(), // 0x12345678
            zext(0x12345678u32.rotate_left(4)),
        ),
        (
            "llvm.fshr.i32",
            "declare i32 @llvm.fshr.i32(i32, i32, i32)".into(),
            "call i32 @llvm.fshr.i32(i32 305419896, i32 305419896, i32 4)".into(),
            zext(0x12345678u32.rotate_right(4)),
        ),
        (
            "llvm.bswap.i32",
            "declare i32 @llvm.bswap.i32(i32)".into(),
            "call i32 @llvm.bswap.i32(i32 287454020)".into(), // 0x11223344
            zext(0x11223344u32.swap_bytes()),
        ),
        (
            "llvm.bitreverse.i32",
            "declare i32 @llvm.bitreverse.i32(i32)".into(),
            "call i32 @llvm.bitreverse.i32(i32 1)".into(),
            zext(1u32.reverse_bits()),
        ),
        (
            "llvm.uadd.sat.i32",
            "declare i32 @llvm.uadd.sat.i32(i32, i32)".into(),
            "call i32 @llvm.uadd.sat.i32(i32 -1, i32 2)".into(), // 0xFFFFFFFF + 2 saturates
            zext(u32::MAX.saturating_add(2)),
        ),
        (
            "llvm.usub.sat.i32",
            "declare i32 @llvm.usub.sat.i32(i32, i32)".into(),
            "call i32 @llvm.usub.sat.i32(i32 1, i32 5)".into(),
            zext(1u32.saturating_sub(5)),
        ),
        (
            "llvm.sadd.sat.i32",
            "declare i32 @llvm.sadd.sat.i32(i32, i32)".into(),
            "call i32 @llvm.sadd.sat.i32(i32 2147483647, i32 1)".into(), // i32::MAX + 1 saturates
            zext(i32::MAX.saturating_add(1) as u32),
        ),
        (
            "llvm.ssub.sat.i32",
            "declare i32 @llvm.ssub.sat.i32(i32, i32)".into(),
            "call i32 @llvm.ssub.sat.i32(i32 -2147483648, i32 1)".into(), // i32::MIN - 1 saturates
            zext(i32::MIN.saturating_sub(1) as u32),
        ),
    ];

    let mut failures: Vec<String> = Vec::new();
    for (name, decl, call, expected) in &cases {
        let ir = module(decl, call);
        match run_i64(&ir) {
            Ok(got) if got == *expected => {}
            Ok(got) => failures.push(format!("{name}: got {got}, want {expected}")),
            Err(e) => failures.push(format!("{name}: {e}\n{ir}")),
        }
    }
    assert!(
        failures.is_empty(),
        "{}/{} intrinsic corpus case(s) failed:\n{}",
        failures.len(),
        cases.len(),
        failures.join("\n---\n")
    );
}

#[test]
fn unknown_llvm_intrinsic_fails_closed() {
    // A made-up `llvm.*` name must NOT be silently accepted (translated + verified into a call to a
    // nonexistent body). Fail-closed = an error at some stage; a clean `Ok` here would mean the on-ramp
    // emitted a bogus module. (This is the §2a "reject what you don't understand" posture.)
    let ir = module(
        "declare i32 @llvm.definitely.not.real.i32(i32, i32)",
        "call i32 @llvm.definitely.not.real.i32(i32 1, i32 2)",
    );
    assert!(
        run_i64(&ir).is_err(),
        "an unknown llvm.* intrinsic must fail closed, not translate+verify+run"
    );
}
