//! #2474 — a function's OSR entry ([`temen_ir::Func::osr_entry`]): the function entered at one of
//! its loop entries, where a call that has run long in the interpreter goes on in emitted code.
//! The interpreter is the oracle for it as for every function: entered with the state the function
//! has on first reaching the loop, it returns what the function returns, burns the same fuel, and
//! runs out of fuel at the same safepoint (INVARIANTS #9).

use temen_interp::{bytecode, Value};
use temen_ir::{append_osr_entries, osr_entries, Func, Module};
use temen_text::parse_module;

fn module(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// `m` with its OSR entries appended, verified.
fn with_osr(m: &Module) -> Module {
    let mut osr = m.clone();
    append_osr_entries(&mut osr);
    temen_verify::verify_module(&osr).expect("the OSR entries verify");
    osr
}

/// Run function `f` of `m` from `args` with `fuel`: its results and the fuel left.
fn run(
    m: &Module,
    f: u32,
    args: &[Value],
    fuel: u64,
) -> (Result<Vec<Value>, temen_interp::Trap>, u64) {
    let mut left = fuel;
    let r = bytecode::compile_and_run(m, f, args, &mut left).expect("in subset");
    (r, left)
}

/// Function 0 counts to its argument in a loop whose header carries an `i64` count, an `i32` that
/// steps by 2, an `f64` that steps by 3 and an `f32` that steps by 1. It returns the count plus the
/// `i32` and the bits of both floats. Its entry block only starts the loop.
const MIXED: &str = "memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  v0 = i64.const 0
  w0 = i32.const 0
  d0 = f64.const 0.0
  s0 = f32.const 0.0
  br 1(vn, v0, w0, d0, s0)
}
block 1 (vm: i64, vk: i64, vw: i32, vd: f64, vs: f32) {
  vlt = i64.lt_s vk vm
  br_if vlt 2(vm, vk, vw, vd, vs) 3(vk, vw, vd, vs)
}
block 2 (am: i64, ak: i64, aw: i32, ad: f64, as: f32) {
  v1 = i64.const 1
  ak1 = i64.add ak v1
  w2 = i32.const 2
  aw1 = i32.add aw w2
  d3 = f64.const 3.0
  ad1 = f64.add ad d3
  s1 = f32.const 1.0
  as1 = f32.add as s1
  br 1(am, ak1, aw1, ad1, as1)
}
block 3 (rk: i64, rw: i32, rd: f64, rs: f32) {
  rw64 = i64.extend_i32_u rw
  rd64 = i64.reinterpret_f64 rd
  rs32 = i32.reinterpret_f32 rs
  rs64 = i64.extend_i32_u rs32
  r1 = i64.add rk rw64
  r2 = i64.add r1 rd64
  r3 = i64.add r2 rs64
  return r3
  }
}
";

#[test]
fn an_osr_entry_goes_on_from_its_loop_entry_as_the_function_does() {
    let m = module(MIXED);
    let osr = with_osr(&m);
    assert_eq!(osr_entries(&m.funcs), [Some(1)]);
    // The state the function has on first reaching its loop: the count, then the zero slots.
    let at_entry = |n: i64| {
        let zeros = [0i64; 4];
        let mut a = vec![Value::I64(0), Value::I64(n)];
        a.extend(zeros.iter().map(|z| Value::I64(*z)));
        a
    };
    for n in [0, 1, 7, 100] {
        let fuel = 1_000_000;
        let direct = run(&m, 0, &[Value::I64(n)], fuel);
        assert_eq!(run(&osr, 1, &at_entry(n), fuel), direct, "n = {n}");
    }
    // A budget that runs out mid-loop runs out at the same safepoint.
    for fuel in [1, 2, 3, 50] {
        let direct = run(&m, 0, &[Value::I64(100)], fuel);
        assert_eq!(direct.0, Err(temen_interp::Trap::OutOfFuel), "fuel {fuel}");
        assert_eq!(run(&osr, 1, &at_entry(100), fuel), direct, "fuel {fuel}");
    }
}

/// Function 0's results are an `i32` and an `f32`, which its OSR entry returns widened to `i64`
/// slots holding their bits.
const NARROW_RESULTS: &str = "memory 16
func (i32) -> (i32, f32) {
block 0 (vn: i32) {
  v0 = i32.const 0
  br 1(vn, v0)
}
block 1 (vm: i32, vk: i32) {
  vlt = i32.lt_s vk vm
  br_if vlt 2(vm, vk) 3(vk)
}
block 2 (am: i32, ak: i32) {
  v1 = i32.const 1
  ak1 = i32.add ak v1
  br 1(am, ak1)
}
block 3 (rk: i32) {
  half = f32.const 0.5
  return rk, half
  }
}
";

#[test]
fn an_osr_entry_returns_narrow_results_in_slots() {
    let m = module(NARROW_RESULTS);
    let osr = with_osr(&m);
    let direct = run(&m, 0, &[Value::I32(9)], 10_000).0.expect("runs");
    let entered = run(
        &osr,
        1,
        &[Value::I64(0), Value::I64(9), Value::I64(0)],
        10_000,
    )
    .0
    .expect("runs");
    let bits: Vec<i64> = direct
        .iter()
        .map(|v| match v {
            Value::I32(x) => *x as u32 as i64,
            Value::F32(x) => x.to_bits() as i64,
            v => panic!("unexpected {v:?}"),
        })
        .collect();
    assert_eq!(
        entered,
        bits.into_iter().map(Value::I64).collect::<Vec<_>>()
    );
}

/// Two loops in a row, the second holding a loop of its own, then the function returns.
const LOOPS: &str = "memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  v0 = i64.const 0
  br 1(vn, v0)
}
block 1 (vm: i64, vk: i64) {
  vlt = i64.lt_s vk vm
  br_if vlt 2(vm, vk) 3(vm, vk)
}
block 2 (am: i64, ak: i64) {
  v1 = i64.const 1
  ak1 = i64.add ak v1
  br 1(am, ak1)
}
block 3 (bm: i64, bk: i64) {
  v0 = i64.const 0
  br 4(bm, bk, v0)
}
block 4 (om: i64, ok: i64, oi: i64) {
  vlt = i64.lt_s oi om
  br_if vlt 5(om, ok, oi, oi) 7(ok)
}
block 5 (im: i64, ik: i64, ii: i64, ij: i64) {
  vlt = i64.lt_s ij im
  br_if vlt 6(im, ik, ii, ij) 8(im, ik, ii)
}
block 6 (jm: i64, jk: i64, ji: i64, jj: i64) {
  v1 = i64.const 1
  jk1 = i64.add jk v1
  jj1 = i64.add jj v1
  br 5(jm, jk1, ji, jj1)
}
block 7 (rk: i64) {
  return rk
}
block 8 (nm: i64, nk: i64, ni: i64) {
  v1 = i64.const 1
  ni1 = i64.add ni v1
  br 4(nm, nk, ni1)
  }
}
";

#[test]
fn a_loop_inside_another_is_no_entry() {
    let m = module(LOOPS);
    assert_eq!(
        m.funcs[0].loop_entries(),
        [1, 4],
        "the two outer headers, not the inner loop's (block 5)"
    );
    let osr = with_osr(&m);
    // Selector 1 enters the second loop as the function reaches it: the first loop has counted
    // to `n`, and the outer count is 0. (It skips the first loop's fuel.)
    for n in [0, 3, 10] {
        let direct = run(&m, 0, &[Value::I64(n)], 1_000_000).0;
        let args = [1, n, n, 0].map(Value::I64);
        assert_eq!(run(&osr, 1, &args, 1_000_000).0, direct, "n = {n}");
    }
}

#[test]
fn a_function_without_a_loop_has_no_osr_entry() {
    let m = module(
        "memory 16
func (i64) -> (i64) {
block 0 (v: i64) {
  return v
  }
}
",
    );
    assert_eq!(osr_entries(&m.funcs), [None]);
    assert!(m.funcs[0].osr_entry().is_none());
    let osr = with_osr(&m);
    assert_eq!(osr.funcs.len(), 1);
}

#[test]
fn a_header_with_a_vector_param_is_no_entry() {
    let f = &module(
        "memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  z = v128.const 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  v0 = i64.const 0
  br 1(vn, v0, z)
}
block 1 (vm: i64, vk: i64, vz: v128) {
  vlt = i64.lt_s vk vm
  br_if vlt 2(vm, vk, vz) 3(vk)
}
block 2 (am: i64, ak: i64, az: v128) {
  v1 = i64.const 1
  ak1 = i64.add ak v1
  br 1(am, ak1, az)
}
block 3 (rk: i64) {
  return rk
  }
}
",
    )
    .funcs[0];
    assert!(f.loop_entries().is_empty());
    assert!(Func::osr_entry(f).is_none());
}
