//! **CONSOLIDATION.md §3c.3 — the config-record spawn on the wasm-JIT tier**: `Instantiator`
//! op 17 (`instantiate_rec(record_ptr)`) lowers to an `env.instantiate_rec` host bounce
//! `(win, inst, record_ptr) -> child`, emitted as a **conditional** import — func import 8,
//! present only when the module actually uses op 17 — so every existing nested-mode module
//! keeps its exact import set and no driver changes until it loads an op-17 module.
//!
//! The proof mirrors `nested_vm.rs`: the emitted parent builds the 56-byte record in its window
//! with ordinary stores, bounces, and the servicer reads the record back out of linear memory
//! (the full marshalling round-trip), validates it exactly as the other tiers do, runs the child
//! on the interpreter (the oracle), and `env.join` returns its result. The conditional-import
//! property is asserted from both sides: the op-17 module refuses to instantiate without the
//! import defined, and the op-0 module from `nested_vm.rs`'s shape instantiates with a linker
//! that never defines it.

use temen_interp::{run, Value};
use temen_wasm_jit::compile_module_nested;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;

/// Parent (func 0, `(i64 inst) -> (i64)`): build the record at window offset 2048 — version 0,
/// entry 1, off 0, size_log2 10, pager `u32::MAX`, module -1, budget 0, quota 0, no grants —
/// `instantiate_rec`, `join`, return. Child (func 1): pure compute, returns 9.
const REC_PARENT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  va0 = i64.const 2048
  vf0 = i64.const 4294967296
  i64.store va0 vf0
  va1 = i64.const 2056
  vz = i64.const 0
  i64.store va1 vz
  va2 = i64.const 2064
  vf16 = i64.const -4294967286
  i64.store va2 vf16
  va3 = i64.const 2072
  vf24 = i64.const 4294967295
  i64.store va3 vf24
  va4 = i64.const 2080
  i64.store va4 vz
  va5 = i64.const 2088
  i64.store va5 vz
  va6 = i64.const 2096
  i64.store va6 vz
  vrp = i64.const 2048
  vch = call.cap 6 17 (i64) -> (i32) vinst (vrp)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
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

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn oracle_child(m: &temen_ir::Module, entry: u32) -> i64 {
    let mut fuel = u64::MAX;
    match run(m, entry, &[], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("child oracle: {other:?}"),
        },
        other => panic!("child oracle run: {other:?}"),
    }
}

struct HostState {
    module: temen_ir::Module,
    children: Vec<i64>,
    saw_rec_bounce: bool,
}

/// Define every nested-mode import EXCEPT `env.instantiate_rec` (the callers add that one, or
/// deliberately don't — the conditional-import property is proven by which modules instantiate).
fn base_linker(engine: &Engine, memory: Memory) -> Linker<HostState> {
    let mut linker: Linker<HostState> = Linker::new(engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, HostState>, _f: i32, _a: i32| unreachable!("no interp leaf"),
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, HostState>,
             _win: i32,
             _inst: i32,
             _entry: i64,
             _off: i64,
             _slog: i64,
             _quota: i64|
             -> i32 { unreachable!("no op-0 spawn in these units") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |caller: Caller<'_, HostState>, _inst: i32, child: i32| -> i64 {
                caller.data().children[child as usize]
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, HostState>, _f: i32, _sp: i64, _a: i64| -> i32 {
                unreachable!("no thread op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, HostState>, _h: i32| -> i64 { unreachable!("no thread op") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 {
                unreachable!("no futex op")
            },
        )
        .unwrap();
    linker
}

/// The record round-trip: emitted stores → linear memory → `env.instantiate_rec` reads + validates
/// the record host-side → interpreter child → `env.join`. Result must equal the child oracle.
#[test]
fn record_spawn_bounces_and_matches_interp() {
    let m = parse(REC_PARENT);
    let want = oracle_child(&m, 1);
    assert_eq!(want, 9, "child oracle");

    let wasm = compile_module_nested(&m, false).expect("op-17 entry emits (nested)");
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            module: m.clone(),
            children: Vec::new(),
            saw_rec_bounce: false,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    let mut linker = base_linker(&engine, memory);
    linker
        .func_wrap(
            "env",
            "instantiate_rec",
            move |mut caller: Caller<'_, HostState>,
                  win: i32,
                  _inst: i32,
                  record_ptr: i64|
                  -> i32 {
                // Read the 56-byte record back out of linear memory — the marshalling proof —
                // and validate it exactly as the tree-walker / Cranelift thunk do.
                let mut rec = [0u8; 56];
                memory
                    .read(&caller, (win as u64 + record_ptr as u64) as usize, &mut rec)
                    .expect("record in-window");
                let u32_at =
                    |o: usize| u32::from_le_bytes([rec[o], rec[o + 1], rec[o + 2], rec[o + 3]]);
                assert_eq!(u32_at(0), 0, "version");
                assert_eq!(u32_at(20), u32::MAX, "no pager");
                assert_eq!(u32_at(24) as i32, -1, "self module");
                assert_eq!(u32_at(28), 0, "no budget");
                let entry = u32_at(4);
                caller.data_mut().saw_rec_bounce = true;
                let m = caller.data().module.clone();
                let r = oracle_child(&m, entry);
                let st = caller.data_mut();
                st.children.push(r);
                (st.children.len() - 1) as i32
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("f0 exported");
    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(7)];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results).expect("f0 runs");

    assert!(
        store.data().saw_rec_bounce,
        "entry never bounced to env.instantiate_rec (silent fallback)"
    );
    assert_eq!(results[0].i64(), Some(want), "record spawn != oracle");
}

/// The conditional-import property, from both sides: the op-17 module refuses to instantiate
/// when `env.instantiate_rec` is undefined (the import is real), while an op-less pure-compute
/// module instantiates against the very same rec-less linker (no import was added for it).
#[test]
fn instantiate_rec_import_is_conditional() {
    let engine = Engine::default();

    // Side 1: the op-17 module NEEDS the import.
    let m = parse(REC_PARENT);
    let wasm = compile_module_nested(&m, false).expect("emits");
    let module = WModule::new(&engine, &wasm).expect("validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            module: m,
            children: Vec::new(),
            saw_rec_bounce: false,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let linker = base_linker(&engine, memory);
    assert!(
        linker.instantiate(&mut store, &module).is_err(),
        "op-17 module must demand env.instantiate_rec"
    );

    // Side 2: a module with no op 17 instantiates against the SAME import set (no rec import
    // was emitted for it — existing drivers keep working unchanged).
    let plain = parse(
        r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = i64.const 3
  return vr
  }
}
"#,
    );
    let wasm = compile_module_nested(&plain, false).expect("emits");
    let module = WModule::new(&engine, &wasm).expect("validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            module: plain,
            children: Vec::new(),
            saw_rec_bounce: false,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let linker = base_linker(&engine, memory);
    let inst = linker.instantiate(&mut store, &module);
    assert!(
        inst.is_ok(),
        "an op-less module must NOT demand env.instantiate_rec: {:?}",
        inst.err()
    );
}
