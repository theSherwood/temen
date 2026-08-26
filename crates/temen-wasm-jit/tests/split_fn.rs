//! **#1120 Slice 1 — intra-function split, correctness differential.** A single Temen function emitted
//! as K block-group wasm functions (via [`temen_wasm_jit::compile_split_fn`]) must return exactly what
//! the tree-walk interpreter (the oracle, INVARIANTS.md #9) returns, for every K. The split's new
//! machinery is the inter-group control-flow edge: a branch to a block in a sibling group becomes a
//! frame-reusing `return_call` that marshals the target block's params through the env scratch. So the
//! guests here have rich intra-function control flow — loops (back-edges), `br_if`, `br_table`, and
//! memory accesses — cut at every K so those edges actually cross group boundaries.
//!
//! K=1 is the degenerate single-group case (proves the wrapper + entry-dispatch + group-dispatch scaffold
//! reproduces the monolithic result); K up to the block count forces many cross-group edges.

use temen_interp::{run, Value};
use temen_wasm_jit::{compile_module_with, compile_split_fn};
use wasmi::{Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024; // env cell (fuel scratch) — outside the window

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Interpreter oracle: run func 0 with the given i64 args.
fn oracle(m: &temen_ir::Module, args: &[i64]) -> i64 {
    let vals: Vec<Value> = args.iter().map(|&a| Value::I64(a)).collect();
    let mut fuel = u64::MAX;
    match run(m, 0, &vals, &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("oracle result: {other:?}"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Emit `m`'s func 0 split into `k` groups, instantiate over a private memory, and call `f0`.
fn run_split(m: &temen_ir::Module, k: usize, args: &[i64]) -> i64 {
    let wasm = compile_split_fn(m, 0, k, false).expect("split emits");
    let engine = Engine::default();
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
    // Seed the env cell's fuel slot high (the emitted fuel global self-inits, but keep parity with the
    // other harnesses); harmless — the group fuel checks read the global, not this cell.
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: wasmi::Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: wasmi::Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("split-fn guest has no leaves")
            },
        )
        .unwrap();

    let module = WModule::new(&engine, &wasm).unwrap_or_else(|e| panic!("split wasm (k={k}): {e}"));
    let inst = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = inst.get_func(&store, "f0").expect("f0 export");

    let mut params = vec![Val::I32(WIN_BASE), Val::I32(ENV_PTR)];
    params.extend(args.iter().map(|&a| Val::I64(a)));
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .unwrap_or_else(|e| panic!("f0 call (k={k}): {e}"));
    results[0].i64().expect("i64 result")
}

/// Split `m`'s func 0 at every K from 1..=block_count and assert parity with the oracle over `inputs`.
fn assert_split_parity(src: &str, inputs: &[&[i64]]) {
    let m = parse(src);
    let nblocks = m.funcs[0].blocks.len();
    for args in inputs {
        let want = oracle(&m, args);
        for k in 1..=nblocks {
            let got = run_split(&m, k, args);
            assert_eq!(
                got, want,
                "split k={k} args={args:?}: got {got} != oracle {want}"
            );
        }
    }
}

/// A self-contained countdown-sum loop: `f0(n) = n + (n-1) + ... + 1`. Four blocks, a `br_if`, and a
/// back-edge — every K≥2 cuts the loop across groups.
const LOOP_SUM: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vacc = i64.const 0
  br 1(vn, vacc)
}
block 1 (vi: i64, vacc: i64) {
  vz = i64.const 0
  vcmp = i64.eq vi vz
  br_if vcmp 2(vacc) 3(vi, vacc)
}
block 2 (vr: i64) {
  return vr
}
block 3 (vi3: i64, vacc3: i64) {
  vacc4 = i64.add vacc3 vi3
  vone = i64.const 1
  vim1 = i64.sub vi3 vone
  br 1(vim1, vacc4)
}
}
"#;

/// A `br_table` dispatch: `f0(sel, x)` routes on `sel` to one of three arms, each returning a different
/// function of `x`. Exercises multi-target cross-group edges.
const BR_TABLE: &str = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vsel: i64, vx: i64) {
  vs32 = i32.wrap_i64 vsel
  br_table vs32 [1(vx), 2(vx), 3(vx)] 4(vx)
}
block 1 (va: i64) {
  vc = i64.const 10
  vr = i64.add va vc
  return vr
}
block 2 (vb: i64) {
  vc = i64.const 2
  vr = i64.mul vb vc
  return vr
}
block 3 (vd: i64) {
  vr = i64.sub vd vd
  return vr
}
block 4 (ve: i64) {
  return ve
}
}
"#;

/// A memory round-trip inside the loop: store the accumulator each iteration, load it back — exercises
/// `emit_confine` across group boundaries (the mask/bounds lowering must be identical per block).
const MEM_LOOP: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  vacc = i64.const 0
  br 1(vn, vacc)
}
block 1 (vi: i64, vacc: i64) {
  vz = i64.const 0
  vcmp = i64.eq vi vz
  br_if vcmp 2(vacc) 3(vi, vacc)
}
block 2 (vr: i64) {
  return vr
}
block 3 (vi3: i64, vacc3: i64) {
  vaddr = i64.const 64
  vsum = i64.add vacc3 vi3
  i64.store vaddr vsum
  vld = i64.load vaddr
  vone = i64.const 1
  vim1 = i64.sub vi3 vone
  br 1(vim1, vld)
}
}
"#;

/// Instantiate `wasm` (exporting `f0` + the `fuel` global), seed `fuel = budget`, call `f0(args)`, and
/// return `Some(value)` or `None` on a trap (OutOfFuel). Used to compare the split's fuel-trap point to
/// the monolithic emit's — both use the identical safepoint convention (entry + taken back-edges).
fn run_fuel(wasm: &[u8], args: &[i64], budget: i64) -> Option<i64> {
    let engine = Engine::default();
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |mut c: wasmi::Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: wasmi::Caller<'_, i32>, _f: i32, _a: i32| unreachable!(),
        )
        .unwrap();
    let module = WModule::new(&engine, wasm).expect("wasm validates");
    let inst = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let fuel = inst.get_global(&store, "fuel").expect("fuel global");
    fuel.set(&mut store, Val::I64(budget)).unwrap();
    let f0 = inst.get_func(&store, "f0").expect("f0");
    let mut params = vec![Val::I32(WIN_BASE), Val::I32(ENV_PTR)];
    params.extend(args.iter().map(|&a| Val::I64(a)));
    let mut results = [Val::I64(0)];
    match f0.call(&mut store, &params, &mut results) {
        Ok(()) => Some(results[0].i64().expect("i64")),
        Err(_) => None,
    }
}

/// The split traps `OutOfFuel` at the *same* budget as the monolithic emit (same safepoint charges), and
/// returns the same value when it doesn't trap — across a budget sweep straddling the trap threshold.
#[test]
fn split_fuel_trap_parity_with_monolithic() {
    let m = parse(LOOP_SUM);
    let mono = compile_module_with(&m, false).expect("monolithic emits");
    let nblocks = m.funcs[0].blocks.len();
    for &n in &[0i64, 1, 3, 7] {
        for budget in 0..=(n + 4) {
            let want = run_fuel(&mono, &[n], budget);
            for k in 1..=nblocks {
                let split = compile_split_fn(&m, 0, k, false).expect("split emits");
                assert_eq!(
                    run_fuel(&split, &[n], budget),
                    want,
                    "fuel parity k={k} n={n} budget={budget}"
                );
            }
        }
    }
}

#[test]
fn split_loop_sum_matches_interp() {
    assert_split_parity(
        LOOP_SUM,
        &[&[0], &[1], &[2], &[3], &[5], &[10], &[100], &[1000]],
    );
}

#[test]
fn split_br_table_matches_interp() {
    assert_split_parity(
        BR_TABLE,
        &[
            &[0, 7],
            &[1, 7],
            &[2, 7],
            &[3, 7],
            &[9, 7],
            &[0, 0],
            &[2, 100],
        ],
    );
}

#[test]
fn split_mem_loop_matches_interp() {
    assert_split_parity(MEM_LOOP, &[&[0], &[1], &[4], &[10], &[50]]);
}
