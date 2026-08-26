//! **#1110 emit-split — correctness differential.** A whole-program guest emitted as *two* partitions
//! sharing one reserved funcref table must return exactly what the single-module tier and the tree-walk
//! interpreter (the oracle, INVARIANTS.md #9) return. The split's new lowering is the cross-module call:
//! a `Call`/`ReturnCall` to a function emitted by the *sibling* partition dispatches through the shared
//! table (`env.__indirect_function_table`) at the callee's index — so the whole point of this test is a
//! guest whose hot function calls a helper that lands in the other module.
//!
//! The host plays the domain exactly as Model B2 does (`b2_install.rs`): it instantiates every partition
//! against one shared memory + one shared table, and writes each function's `f{i}` funcref into slot `i`
//! (`table.set`). The entry is then invoked on whichever partition emitted it. Splitting the *same* IR at
//! *different* cut points (and single-module, and interpreter) must all agree — parity across the cut is
//! the property the emit-split feasibility rests on.

use temen_interp::{run, Value};
use temen_wasm_jit::{compile_module_b2, compile_module_split};
use wasmi::core::ValType;
use wasmi::{
    Engine, FuncRef, Linker, Memory, MemoryType, Module as WModule, Store, Table, TableType, Val,
};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;
const LOG2: u32 = 4; // 16 slots — ample for these tiny guests

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// The interpreter oracle: run `m`'s entry (func 0) with a single i64 arg (INVARIANTS.md #9).
fn oracle(m: &temen_ir::Module, arg: i64) -> i64 {
    let mut fuel = u64::MAX;
    match run(m, 0, &[Value::I64(arg)], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("oracle result: {other:?}"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Emit `m` as one whole-program B2 module, instantiate it against a shared memory + table, populate
/// every slot with its own `f{i}`, and call `f0(WIN_BASE, ENV_PTR, arg)`. The single-module baseline.
fn run_single(m: &temen_ir::Module, arg: i64) -> i64 {
    run_partitions(m, &[full_mask(m)], arg)
}

/// A mask with every function in the one (and only) partition.
fn full_mask(m: &temen_ir::Module) -> Vec<bool> {
    vec![true; m.funcs.len()]
}

/// Emit `m` split across `masks` partitions (each `masks[p][i]` ⇒ partition `p` emits func `i`; the
/// masks must be a partition — exactly one true per `i`), link them over one shared memory + table,
/// populate every slot `i` with the emitting partition's `f{i}`, and invoke the entry `f0`.
fn run_partitions(m: &temen_ir::Module, masks: &[Vec<bool>], arg: i64) -> i64 {
    let n = m.funcs.len();
    // Sanity: the masks partition [0, n) exactly once (a slot's owner must be unambiguous).
    for i in 0..n {
        let owners = masks.iter().filter(|mk| mk[i]).count();
        assert_eq!(
            owners, 1,
            "func {i} must be emitted by exactly one partition"
        );
    }

    let engine = Engine::default();
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    let table = Table::new(
        &mut store,
        TableType::new(ValType::FuncRef, 1 << LOG2, Some(1 << LOG2)),
        Val::FuncRef(FuncRef::null()),
    )
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
            |_: wasmi::Caller<'_, i32>, _f: i32, _a: i32| unreachable!("no cross-tier leaf here"),
        )
        .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();

    // Compile + instantiate each partition; remember which instance owns `f0` (the entry lives in
    // whichever partition emitted func 0).
    let mut entry_f0 = None;
    for (p, mask) in masks.iter().enumerate() {
        let wasm = if masks.len() == 1 {
            // Single-partition = the ordinary whole-program B2 emit (proves the split path degenerates
            // to it, and exercises the plain B2 module as the baseline).
            compile_module_b2(m, false, LOG2).expect("whole-program B2 emits")
        } else {
            compile_module_split(m, false, LOG2, mask).expect("partition emits")
        };
        let module =
            WModule::new(&engine, &wasm).unwrap_or_else(|e| panic!("partition {p} wasm: {e}"));
        let inst = linker
            .instantiate(&mut store, &module)
            .unwrap()
            .start(&mut store)
            .unwrap();
        // Populate the shared table: every func this partition emits is exported `f{i}` → slot `i`.
        for (i, &owns) in mask.iter().enumerate() {
            if owns {
                let f = inst
                    .get_func(&store, &format!("f{i}"))
                    .expect("f{i} export");
                table
                    .set(&mut store, i as u64, Val::FuncRef(FuncRef::new(f)))
                    .unwrap();
                if i == 0 {
                    entry_f0 = Some(f);
                }
            }
        }
    }

    let f0 = entry_f0.expect("some partition emitted f0");
    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(arg)];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .expect("entry call");
    results[0].i64().expect("i64 result")
}

/// A hot loop `f0(n)` that calls helper `f1` each iteration (`f1(x) = 3x + 1`), summing the results —
/// the shape whose per-iteration call becomes a **cross-module call_indirect** when `f1` is split off.
const LOOP_CALLS_HELPER: &str = r#"memory 16
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
  vd = call 1 (vi3)
  vsum = i64.add vacc3 vd
  vone = i64.const 1
  vim1 = i64.sub vi3 vone
  br 1(vim1, vsum)
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v3 = i64.const 3
  vm = i64.mul vx v3
  v1 = i64.const 1
  vr = i64.add vm v1
  return vr
}
}
"#;

/// `f0(n)` tail-calls helper `f1` (`f1(x) = x + 7`) — exercises the **cross-module return_call_indirect**
/// lowering when `f1` is split off.
const TAILCALL_HELPER: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vn: i64) {
  return_call 1 (vn)
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v7 = i64.const 7
  vr = i64.add vx v7
  return vr
}
}
"#;

/// Two partitionings of `LOOP_CALLS_HELPER` — helper in the sibling module (cross-module call), and the
/// mirror cut — must both match the single-module tier and the interpreter, across a sweep of inputs.
#[test]
fn split_loop_calls_helper_matches_interp() {
    let m = parse(LOOP_CALLS_HELPER);
    // A={f0}, B={f1}: f0's per-iteration `call 1` is a cross-module dispatch through the shared table.
    let a_lo = vec![vec![true, false], vec![false, true]];
    // A={f1}, B={f0}: the entry itself lands in the second partition; the cut is mirrored.
    let a_hi = vec![vec![false, true], vec![true, false]];
    for arg in [0i64, 1, 2, 3, 5, 10, 100] {
        let want = oracle(&m, arg);
        assert_eq!(
            run_single(&m, arg),
            want,
            "single-module != interp (n={arg})"
        );
        assert_eq!(
            run_partitions(&m, &a_lo, arg),
            want,
            "split A={{f0}} != interp (n={arg})"
        );
        assert_eq!(
            run_partitions(&m, &a_hi, arg),
            want,
            "split A={{f1}} != interp (n={arg})"
        );
    }
}

/// The cross-module **tail** call (`return_call_indirect`) matches the oracle.
#[test]
fn split_tailcall_helper_matches_interp() {
    let m = parse(TAILCALL_HELPER);
    let masks = vec![vec![true, false], vec![false, true]]; // f0 tail-calls f1 cross-module
    for arg in [0i64, 1, 7, 41, 1000] {
        let want = oracle(&m, arg);
        assert_eq!(
            run_single(&m, arg),
            want,
            "single-module != interp (n={arg})"
        );
        assert_eq!(
            run_partitions(&m, &masks, arg),
            want,
            "split tail-call != interp (n={arg})"
        );
    }
}
