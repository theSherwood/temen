//! **§14 VM-in-VM on the wasm-JIT tier** — a unit whose entry *uses* its `Instantiator` capability
//! (spawns a nested confined VM and `join`s it) now runs on emitted wasm, where before any `cap.call`
//! forced the whole entry onto the interpreter (`DESIGN.md` §14, BROWSER.md "wasm-JIT tier").
//!
//! The mechanism under test is [`svm_wasm_jit::compile_module_nested`]: a `cap.call` to INSTANTIATOR
//! `instantiate` (op 0) / `join` (op 1) lowers to a host-driver bounce — `env.instantiate` /
//! `env.join` imports, the funcref-table-free analog of `env.call_interp` — so the child vCPU spawn +
//! join happen host-side, exactly as the interpreter surfaces `VcpuStop::Instantiate` to its driver.
//! The emitted parent does no confinement itself.
//!
//! Oracle (INVARIANTS.md #9): the child runs on the tree-walk interpreter (`env.instantiate` services
//! it there), so the emitted parent's `instantiate → join → return` must yield exactly the child's
//! interpreter result. The `saw_bounce` flag is the non-vacuity guard (the emitted code really called
//! the host imports, not a silent fallback).

use svm_interp::{run, Value};
use svm_wasm_jit::compile_module_nested;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;

/// A §14 guest: func 0's entry takes its `Instantiator` handle, `instantiate`s func 1 into a
/// `2^10`-byte sub-window at offset 0, `join`s the child, and returns its result. Func 1 (the child)
/// is pure compute — `9`.
const NESTED: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  ventry = i64.const 1
  voff = i64.const 0
  vslog = i64.const 10
  vquota = i64.const 0
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vr = cap.call 6 1 (i32) -> (i64) vinst (vch)
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vr = i64.const 9
  return vr
  }
}
"#;

fn parse(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// The interpreter oracle for the child unit (func `entry`), run directly.
fn oracle_child(m: &svm_ir::Module, entry: u32) -> i64 {
    let mut fuel = u64::MAX;
    match run(m, entry, &[], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("child oracle: {other:?}"),
        },
        other => panic!("child oracle run: {other:?}"),
    }
}

/// Host state threaded through the wasmi `Store`: the module (to run children on the interpreter),
/// each spawned child's result (indexed by the handle `env.instantiate` returns), and the non-vacuity
/// flag.
struct HostState {
    module: svm_ir::Module,
    children: Vec<i64>,
    saw_bounce: bool,
}

#[test]
fn nested_instantiate_join_matches_interp() {
    let m = parse(NESTED);
    let want = oracle_child(&m, 1);
    assert_eq!(want, 9, "child oracle");

    let wasm = compile_module_nested(&m, false).expect("§14 entry emits (nested)");

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("nested wasm validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            module: m.clone(),
            children: Vec::new(),
            saw_bounce: false,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, HostState>, _f: i32, _a: i32| {
                unreachable!("no interp leaf in this unit")
            },
        )
        .unwrap();
    // env.instantiate: spawn the child (run func `entry` on the interpreter — the oracle) and return
    // its handle. The confinement (window carve at `off`/`size_log2`) is the host's job; this proof
    // exercises the wasm→host marshalling + the join round-trip, so it runs the child directly.
    linker
        .func_wrap(
            "env",
            "instantiate",
            |mut caller: Caller<'_, HostState>,
             _win: i32,
             _inst: i32,
             entry: i64,
             _off: i64,
             _slog: i64,
             _quota: i64|
             -> i32 {
                caller.data_mut().saw_bounce = true;
                let m = caller.data().module.clone();
                let r = oracle_child(&m, entry as u32);
                let st = caller.data_mut();
                st.children.push(r);
                (st.children.len() - 1) as i32
            },
        )
        .unwrap();
    // env.join: return the spawned child's result.
    linker
        .func_wrap(
            "env",
            "join",
            |caller: Caller<'_, HostState>, _inst: i32, child: i32| -> i64 {
                caller.data().children[child as usize]
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("f0 exported");

    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(7)]; // arg0 = instantiator handle
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results).expect("f0 runs");

    assert!(
        store.data().saw_bounce,
        "entry never bounced to env.instantiate (silent fallback)"
    );
    assert_eq!(
        results[0].i64(),
        Some(want),
        "emitted §14 entry (instantiate→join) != interpreter child result"
    );
}
