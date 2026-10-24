//! Dominance form (#2454, DESIGN.md §3a): a block captures values of the blocks that dominate it,
//! certified by a parent per block. Pins the verifier's two rules, the wire and text forms, and the
//! passes between the forms ([`Func::raise`], [`Func::localize`]) over generated programs.

#[path = "support/irgen.rs"]
mod irgen;

use irgen::{gen_args, gen_module, run_differential, Gen};
use temen::interp::{bytecode, run, Trap, Value};
use temen::ir::{Func, Module};
use temen::verify::{func_value_types, verify_module, VerifyError};

/// Every function of `m` raised into dominance form.
fn raised(m: &Module) -> Module {
    let mut r = m.clone();
    for f in &mut r.funcs {
        f.raise();
    }
    r
}

/// Every function of `m` localized back into block-local form.
fn localized(m: &Module) -> Module {
    let mut r = m.clone();
    let types: Vec<_> = r
        .funcs
        .iter()
        .map(|f| func_value_types(f, &r.funcs, &r.types, r.memory.is_some()))
        .collect();
    for (f, t) in r.funcs.iter_mut().zip(&types) {
        f.localize(t);
    }
    r
}

fn captures(f: &Func) -> usize {
    f.blocks.iter().map(|b| b.captures.len()).sum()
}

fn params(f: &Func) -> usize {
    f.blocks.iter().map(|b| b.params.len()).sum()
}

/// Bit-exact result equality (NaN payloads included).
fn same(a: &Result<Vec<Value>, Trap>, b: &Result<Vec<Value>, Trap>) -> bool {
    let bits = |v: &Value| match v {
        Value::I32(x) => *x as u32 as u128,
        Value::I64(x) => *x as u64 as u128,
        Value::F32(x) => x.to_bits() as u128,
        Value::F64(x) => x.to_bits() as u128,
        Value::Ref(x) => *x as u128,
        Value::V128(b) => u128::from_le_bytes(*b),
    };
    match (a, b) {
        (Ok(x), Ok(y)) => x.len() == y.len() && x.iter().zip(y).all(|(p, q)| bits(p) == bits(q)),
        (Err(x), Err(y)) => x == y,
        _ => false,
    }
}

/// Raising a generated program verifies, survives the wire and text forms unchanged, computes what
/// the original did on every engine, and localizes back to a block-local program that does too.
#[test]
fn raise_and_localize_generated_programs() {
    let (mut funcs_with_captures, mut params_before, mut params_after) = (0, 0, 0);
    for seed in 0..3000u64 {
        let mut g = Gen::from_seed(seed);
        let m = gen_module(&mut g);
        verify_module(&m).unwrap();
        let args = gen_args(&mut g, &m.funcs[0].params);
        let r = raised(&m);
        verify_module(&r).unwrap_or_else(|e| {
            panic!(
                "seed {seed}: raised module fails to verify: {e:?}\n{}\nraised:\n{}",
                temen::text::print_module(&m),
                temen::text::print_module(&r)
            )
        });
        funcs_with_captures += r.funcs.iter().filter(|f| captures(f) > 0).count();
        params_before += m.funcs.iter().map(params).sum::<usize>();
        params_after += r.funcs.iter().map(params).sum::<usize>();
        let bytes = temen::encode::encode_module(&r);
        assert_eq!(
            temen::encode::decode_module(&bytes).unwrap(),
            r,
            "seed {seed}: wire"
        );
        // Text is compared printed: it does not spell everything a module can hold (an `i64x2`
        // lane extract's sign flag, which means nothing at that width).
        let text = temen::text::print_module(&r);
        let back = temen::text::print_module(&temen::text::parse_module(&text).unwrap());
        assert_eq!(back, text, "seed {seed}: text");
        let l = localized(&r);
        assert!(l.funcs.iter().all(|f| captures(f) == 0));
        verify_module(&l).unwrap_or_else(|e| {
            panic!(
                "seed {seed}: localized module fails to verify: {e:?}\nraised:\n{}\nlocalized:\n{}",
                temen::text::print_module(&r),
                temen::text::print_module(&l)
            )
        });
        // Renumbering moves which edges count as back edges, and so where fuel is charged: compare
        // the original only with runs that finish on both.
        let fuel = || 2_000_000u64;
        let before = run(&m, 0, &args, &mut fuel());
        let native = run(&r, 0, &args, &mut fuel());
        let back = run(&l, 0, &args, &mut fuel());
        for (what, after) in [("raised", &native), ("localized", &back)] {
            if matches!(before, Err(Trap::OutOfFuel)) || matches!(after, Err(Trap::OutOfFuel)) {
                continue;
            }
            assert!(
                same(&before, after),
                "seed {seed}: {before:?} became {after:?} {what}\n{}\n{what}:\n{}",
                temen::text::print_module(&m),
                temen::text::print_module(if what == "raised" { &r } else { &l })
            );
        }
        // The raised program on the other engines: the bytecode engine bit-exact with the oracle
        // (fuel included: the same IR charges the same safepoints), the Cranelift JIT through the
        // shared differential and its escape oracle.
        if let Some(bc) = bytecode::compile_and_run(&r, 0, &args, &mut fuel()) {
            assert!(
                same(&native, &bc),
                "seed {seed}: bytecode {bc:?}, oracle {native:?}\n{}",
                temen::text::print_module(&r)
            );
        }
        if seed % 4 == 0 {
            run_differential(&r, &args);
        }
    }
    println!(
        "{funcs_with_captures} functions captured; block params {params_before} -> {params_after}"
    );
    assert!(
        funcs_with_captures > 100,
        "raise rarely captured: the test is near-vacuous"
    );
}

fn verify_text(src: &str) -> Result<(), VerifyError> {
    verify_module(&temen::text::parse_module(src).unwrap())
}

/// A capture names a value of a proper ancestor, under parents that rule 1 makes dominators.
#[test]
fn verifier_accepts_a_capture_of_a_dominating_value() {
    let src = "\
func (i64, i32) -> (i64) {
block 0 (x: i64, c: i32) {
  one = i64.const 1
  y = i64.add x one
  br_if c 1() 2()
}
block 1 in 0 (y1 = 0.y) {
  z = i64.add y1 y1
  br 3(z)
}
block 2 in 0 (x2 = 0.x) {
  br 3(x2)
}
block 3 in 0 (r: i64, y3 = 0.y) {
  s = i64.add r y3
  return s
}
}
";
    verify_text(src).unwrap();
    let m = temen::text::parse_module(src).unwrap();
    let mut fuel = 1_000u64;
    let l = localized(&m);
    verify_module(&l).unwrap();
    assert_eq!(
        run(&l, 0, &[Value::I64(5), Value::I32(1)], &mut fuel),
        Ok(vec![Value::I64(18)])
    );
    assert_eq!(
        run(&l, 0, &[Value::I64(5), Value::I32(0)], &mut fuel),
        Ok(vec![Value::I64(11)])
    );
}

#[test]
fn verifier_rejects_a_parent_that_is_not_earlier() {
    let src = "\
func () -> () {
block 0 () {
  br 1()
}
block 1 in 1 () {
  return
}
}
";
    assert_eq!(
        verify_text(src),
        Err(VerifyError::ParentNotEarlier {
            func: 0,
            block: 1,
            parent: 1
        })
    );
}

/// Rule 1: block 2's parent is block 1, but block 2 is also entered from the entry, which block 1
/// does not dominate.
#[test]
fn verifier_rejects_an_edge_its_target_parent_does_not_dominate() {
    let src = "\
func (i32) -> () {
block 0 (c: i32) {
  br_if c 1() 2()
}
block 1 in 0 () {
  br 2()
}
block 2 in 1 () {
  return
}
}
";
    assert_eq!(
        verify_text(src),
        Err(VerifyError::ParentNotDominating {
            func: 0,
            block: 0,
            target: 2
        })
    );
}

/// Rule 2: block 2 captures a value of block 1, its sibling under the entry, which does not dominate it.
#[test]
fn verifier_rejects_a_capture_of_a_sibling() {
    let src = "\
func (i32) -> (i32) {
block 0 (c: i32) {
  br_if c 1() 2()
}
block 1 in 0 () {
  k = i32.const 7
  return k
}
block 2 in 0 (k2 = 1.k) {
  return k2
}
}
";
    assert_eq!(
        verify_text(src),
        Err(VerifyError::CaptureNotAncestor {
            func: 0,
            block: 2,
            captured: 1
        })
    );
}

#[test]
fn verifier_rejects_a_captured_value_out_of_range() {
    let mut m = temen::text::parse_module(
        "\
func () -> (i32) {
block 0 () {
  k = i32.const 7
  br 1()
}
block 1 in 0 (k1 = 0.k) {
  return k1
}
}
",
    )
    .unwrap();
    m.funcs[0].blocks[1].captures[0].1 = 5;
    assert_eq!(
        verify_module(&m),
        Err(VerifyError::CaptureOutOfRange {
            func: 0,
            block: 1,
            captured: 0,
            value: 5
        })
    );
}

/// A capture carries its value's type: using an `i64` capture as an `i32` is a type error.
#[test]
fn verifier_types_a_capture_as_its_value() {
    let src = "\
func () -> (i32) {
block 0 () {
  k = i64.const 7
  br 1()
}
block 1 in 0 (k1 = 0.k) {
  return k1
}
}
";
    assert!(matches!(
        verify_text(src),
        Err(VerifyError::TypeMismatch { block: 1, .. })
    ));
}

/// The entry is the root: it can name no parent and capture nothing.
#[test]
fn verifier_rejects_a_parent_or_capture_on_the_entry() {
    let mut m = temen::text::parse_module(
        "\
func () -> () {
block 0 () {
  return
}
}
",
    )
    .unwrap();
    m.funcs[0].blocks[0].parent = 3;
    assert_eq!(
        verify_module(&m),
        Err(VerifyError::ParentNotEarlier {
            func: 0,
            block: 0,
            parent: 3
        })
    );
    m.funcs[0].blocks[0].parent = 0;
    m.funcs[0].blocks[0].captures.push((0, 0));
    assert_eq!(
        verify_module(&m),
        Err(VerifyError::CaptureNotAncestor {
            func: 0,
            block: 0,
            captured: 0
        })
    );
}
