//! **#1120 Slice 2b — intra-function split inside a real (Model B2) module, correctness differential.**
//! [`compile_module_b2_split`] emits a whole reserved-table module with one function split into K
//! block-group wasm functions. Unlike the standalone `compile_split_fn`, this is the production
//! `emit_module` path, so the split function may use `call_indirect` and interp leaves — `emit_module`
//! wires the shared table and `env.call_interp`, and the `SplitCtx` only rewrites intra-function edges.
//!
//! The guest here proves exactly that: the hot function (func 0) does a **`call_indirect`** each loop
//! iteration, dispatching to one of two monolithic helpers through the shared table. Splitting func 0 at
//! every K must equal both the interpreter oracle (INVARIANTS.md #9) and the unsplit `compile_module_b2`.

use temen_interp::{run, Value};
use temen_wasm_jit::{compile_module_b2, compile_module_b2_split};
use wasmi::core::ValType;
use wasmi::{
    Engine, FuncRef, Linker, Memory, MemoryType, Module as WModule, Store, Table, TableType, Val,
};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;
const LOG2: u32 = 4; // 16 reserved slots

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

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

/// Instantiate a B2 module over one shared memory + one shared reserved table, populate every slot `i`
/// with the module's own `f{i}` funcref (the host-owns-the-table contract), and call `f0(WIN, ENV, arg)`.
fn run_b2(m: &temen_ir::Module, wasm: &[u8], arg: i64) -> i64 {
    let n = m.funcs.len();
    let engine = Engine::default();
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
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
            |_: wasmi::Caller<'_, i32>, _f: i32, _a: i32| unreachable!("no interp leaves here"),
        )
        .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();

    let module = WModule::new(&engine, wasm).unwrap_or_else(|e| panic!("b2 wasm: {e}"));
    let inst = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // Populate the shared table: slot i = f{i} (for the split function, f{fidx} is its wrapper).
    for i in 0..n {
        let f = inst
            .get_func(&store, &format!("f{i}"))
            .expect("f{i} export");
        table
            .set(&mut store, i as u64, Val::FuncRef(FuncRef::new(f)))
            .unwrap();
    }
    let f0 = inst.get_func(&store, "f0").expect("f0 export");
    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(arg)];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results).expect("f0 call");
    results[0].i64().expect("i64 result")
}

/// Func 0 loops; each iteration `call_indirect`s slot `1 + (i & 1)` — helper func 1 (`x+10`) or func 2
/// (`x*2`) — and sums the results. Splitting func 0 must keep the indirect dispatch correct.
const DISPATCH_LOOP: &str = r#"memory 16
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
  vone = i64.const 1
  vsel = i64.and vi3 vone
  vbase = i64.const 1
  vslot = i64.add vbase vsel
  vslot32 = i32.wrap_i64 vslot
  vd = call.dyn (i64) -> (i64) vslot32 (vi3)
  vsum = i64.add vacc3 vd
  vim1 = i64.sub vi3 vone
  br 1(vim1, vsum)
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 10
  vr = i64.add vx vc
  return vr
}
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  vc = i64.const 2
  vr = i64.mul vx vc
  return vr
}
}
"#;

#[test]
fn b2_split_dispatch_loop_matches_interp() {
    let m = parse(DISPATCH_LOOP);
    let nblocks = m.funcs[0].blocks.len();
    let unsplit = compile_module_b2(&m, false, LOG2).expect("b2 emits");
    for &n in &[0i64, 1, 2, 3, 7, 20, 100] {
        let want = oracle(&m, n);
        // Unsplit B2 is the baseline (also proves the harness).
        assert_eq!(
            run_b2(&m, &unsplit, n),
            want,
            "unsplit b2 != interp (n={n})"
        );
        // Split func 0 (the hot loop with the call_indirect) at every K.
        for k in 2..=nblocks {
            let split = compile_module_b2_split(&m, false, LOG2, 0, k).expect("b2 split emits");
            assert_eq!(
                run_b2(&m, &split, n),
                want,
                "b2 split k={k} != interp (n={n})"
            );
        }
    }
}
