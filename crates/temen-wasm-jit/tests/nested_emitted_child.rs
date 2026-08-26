//! **§14 nested child on the *emitted* tier** (issue #1123 slice 1) — the wasm-JIT twin of the native
//! `compile_child` (`temen-run/tests/nifler_child_jit.rs`, `temen-llvm/tests/rust_guest_op13.rs`). Where
//! `nested_vm.rs` services the `env.instantiate` bounce by running the child on the tree-walk
//! interpreter, this test services it by **emitting a *separate* child module and running it as its own
//! `wasmi` instance over a sub-window carve** — so both the parent and the confined child execute on
//! emitted wasm, exactly as the native Cranelift path runs a nimony phase child on emitted code.
//!
//! The mechanism: the emitted parent's `call.cap 6 0`/`6 1` lower to `env.instantiate`/`env.join`
//! ([`compile_module_nested`]). The host callback treats the bounce as "spawn the resolved child module
//! into the carve `[win+off, win+off + (1<<size_log2))`" — the step a real op-13 driver performs after
//! resolving a module capability — by [`compile_module_nested`]-emitting that *distinct* module and
//! `call`ing its `f{entry}(carve, env)` on a second instance sharing the one linear memory. The child's
//! confinement window base is `win+off`; its accesses mask to its own `1<<size_log2` (invariant I2).
//!
//! Oracle (INVARIANTS.md #9): the child run on the interpreter yields `WANT`; the emitted child must
//! return the same value **and** leave the same bytes in its carve. `saw_emit` is the non-vacuity guard
//! (the handler really emitted + ran the child instance, not a silent interpreter fallback).

use temen_interp::{run, Value};
use temen_wasm_jit::compile_module_nested;
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000; // parent window base (the env cell lives below it)
const ENV_PTR: i32 = 1024; // parent dispatcher-fuel cell
const CHILD_ENV_PTR: i32 = 512; // child dispatcher-fuel cell (below the parent's scratch)
const CARVE_OFF: i64 = 2048; // the child's carve offset (mirrors PARENT's `voff`); 1 KiB = child `memory 10`

/// The **separate** child module: pure compute over its own carve. It stores the sentinel `42` at
/// carve-relative offset 0, loads it back, doubles it, and returns `84`. The store proves the emitted
/// child ran over its carve (the test reads `mem[win+off] == 42`); the doubled load proves real compute.
const CHILD: &str = r#"memory 10
func () -> (i64) {
block 0 () {
  vaddr = i64.const 0
  vsent = i64.const 42
  i64.store vaddr vsent
  vgot = i64.load vaddr
  vtwo = i64.const 2
  vr = i64.mul vgot vtwo
  return vr
  }
}
"#;

/// The parent: its entry takes its `Instantiator` handle (`v0`), `instantiate`s a child into a
/// 1-KiB carve at `CARVE_OFF` (`vslog = 10`), `join`s it, and returns the child's result. Func 0 is the
/// only func — the child it spawns is a *separate* module the host resolves (op-0 is the transport; the
/// host decides which module to emit, as a real driver resolves a module capability).
const PARENT: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  ventry = i64.const 0
  voff = i64.const 2048
  vslog = i64.const 10
  vquota = i64.const 0
  vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  return vr
  }
}
"#;

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// Interpreter oracle for the child module's entry (func 0), run directly.
fn oracle_child(child: &temen_ir::Module) -> i64 {
    let mut fuel = u64::MAX;
    match run(child, 0, &[], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("child oracle: {other:?}"),
        },
        other => panic!("child oracle run: {other:?}"),
    }
}

/// Host state threaded through the wasmi `Store`: the pre-instantiated child's emitted entry `f0`, each
/// spawned child's result (indexed by the handle `env.instantiate` returns), and the non-vacuity flag.
struct HostState {
    child_entry: Option<Func>,
    children: Vec<i64>,
    saw_emit: bool,
}

/// Define the eight §14 nested imports on `linker`, sharing `memory`. `instantiate` emits + runs the
/// separate child over the carve; `join` returns its banked result; the rest are unreachable here.
fn wire_nested_imports(linker: &mut Linker<HostState>, memory: Memory) {
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
    // env.instantiate: EMIT the resolved child (already compiled + instantiated as its own wasmi
    // instance in `child_entry`) over the carve `[win+off, ...)` and run it. This is the wasm-JIT twin of
    // the native `compile_child`: the confined child executes on emitted wasm, not the interpreter.
    linker
        .func_wrap(
            "env",
            "instantiate",
            |mut caller: Caller<'_, HostState>,
             win: i32,
             _inst: i32,
             _entry: i64,
             off: i64,
             _slog: i64,
             _quota: i64|
             -> i32 {
                caller.data_mut().saw_emit = true;
                let child_entry = caller.data().child_entry.expect("child instance pre-built");
                let carve = win + off as i32;
                let mut r = [Val::I64(0)];
                child_entry
                    .call(
                        &mut caller,
                        &[Val::I32(carve), Val::I32(CHILD_ENV_PTR)],
                        &mut r,
                    )
                    .expect("emitted child runs");
                let res = r[0].i64().expect("child returns i64");
                let st = caller.data_mut();
                st.children.push(res);
                (st.children.len() - 1) as i32
            },
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
                unreachable!("no thread op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, HostState>, _h: i32| -> i64 {
                unreachable!("no thread op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!("no futex op in this unit")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 {
                unreachable!("no futex op in this unit")
            },
        )
        .unwrap();
}

#[test]
fn nested_child_runs_on_the_emitted_tier() {
    let parent = parse(PARENT);
    let child = parse(CHILD);
    let want = oracle_child(&child);
    assert_eq!(want, 84, "child oracle: (42 stored, loaded) * 2");

    let parent_wasm =
        compile_module_nested(&parent, false).expect("parent §14 entry emits (nested)");
    let child_wasm = compile_module_nested(&child, false).expect("child module emits (nested)");

    let engine = Engine::default();
    let parent_module = WModule::new(&engine, &parent_wasm).expect("parent wasm validates");
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");

    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_entry: None,
            children: Vec::new(),
            saw_emit: false,
        },
    );
    // Two pages (128 KiB) hold the parent window `[WIN_BASE, WIN_BASE + 64 KiB)`; the carve lives inside.
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    memory
        .write(&mut store, CHILD_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    wire_nested_imports(&mut linker, memory);

    // Pre-build the child as its own wasmi instance over the shared memory, and stash its emitted entry
    // so `env.instantiate` can run it. Both instances share the one linear memory ⇒ the carve is a real
    // sub-window of the parent's window.
    let child_instance = linker
        .instantiate(&mut store, &child_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let child_entry = child_instance
        .get_func(&store, "f0")
        .expect("child emitted f0 export");
    store.data_mut().child_entry = Some(child_entry);

    let parent_instance = linker
        .instantiate(&mut store, &parent_module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = parent_instance
        .get_func(&store, "f0")
        .expect("parent f0 export");

    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(7)]; // arg0 = instantiator handle
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .expect("parent f0 runs");

    assert!(
        store.data().saw_emit,
        "entry never bounced to env.instantiate (silent fallback)"
    );
    assert_eq!(
        results[0].i64(),
        Some(want),
        "emitted §14 child (instantiate→join) != interpreter child result"
    );

    // The emitted child ran over its carve: its sentinel store landed at `WIN_BASE + CARVE_OFF`.
    let mut sentinel = [0u8; 8];
    memory
        .read(
            &store,
            (WIN_BASE as i64 + CARVE_OFF) as usize,
            &mut sentinel,
        )
        .unwrap();
    assert_eq!(
        i64::from_le_bytes(sentinel),
        42,
        "the emitted child wrote its sentinel into its carve (proof it ran over `win+off`)"
    );
}
