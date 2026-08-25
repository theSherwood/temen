//! **§14 VM-in-VM on the wasm-JIT tier** — a unit whose entry *uses* its `Instantiator` capability
//! (spawns a nested confined VM and `join`s it) now runs on emitted wasm, where before any `call.cap`
//! forced the whole entry onto the interpreter (`DESIGN.md` §14, BROWSER.md "wasm-JIT tier").
//!
//! The mechanism under test is [`temen_wasm_jit::compile_module_nested`]: a `call.cap` to INSTANTIATOR
//! `instantiate` (op 0) / `join` (op 1) lowers to a host-driver bounce — `env.instantiate` /
//! `env.join` imports, the funcref-table-free analog of `env.call_interp` — so the child vCPU spawn +
//! join happen host-side, exactly as the interpreter surfaces `VcpuStop::Instantiate` to its driver.
//! The emitted parent does no confinement itself.
//!
//! Oracle (INVARIANTS.md #9): the child runs on the tree-walk interpreter (`env.instantiate` services
//! it there), so the emitted parent's `instantiate → join → return` must yield exactly the child's
//! interpreter result. The `saw_bounce` flag is the non-vacuity guard (the emitted code really called
//! the host imports, not a silent fallback).

use temen_interp::{run, Value};
use temen_wasm_jit::{
    compile_module_nested, compile_module_nested_with_eligibility, compile_nested, DriveMode,
};
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
  vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
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

/// The interpreter oracle for the child unit (func `entry`), run directly.
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

/// Host state threaded through the wasmi `Store`: the module (to run children on the interpreter),
/// each spawned child's result (indexed by the handle `env.instantiate` returns), and the non-vacuity
/// flag.
struct HostState {
    module: temen_ir::Module,
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

/// **§11 slice 3 — thread/futex ops in an emitted unit** (CONSOLIDATION.md §11): the unit's entry
/// `thread.spawn`s its OWN `f1` (→ 7), `join`s it, and does a mismatching `i32.atomic.wait`
/// (mem[64] = 0 ≠ 99 → status 1) — returning 7·10 + 1 = 71. On the emitted tier the four ops arrive
/// as the `env.thread_spawn`/`env.thread_join`/`env.mem_wait`/`env.mem_notify` imports; the servicer
/// supplies the module context (it runs `f1` of THIS unit), mirroring the Worker host. Oracle: the
/// bytecode cooperative driver runs the same unit whole (module-aware spawn, PR #593).
const THREADED_UNIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vsp = i64.const 0
  varg = i64.const 0
  vt = thread.spawn 1 vsp varg
  vj = thread.join vt
  vaddr = i64.const 64
  vexp = i32.const 99
  vtmo = i64.const 0
  vw = i32.atomic.wait vaddr vexp vtmo
  vw64 = i64.extend_i32_u vw
  vten = i64.const 10
  vjs = i64.mul vj vten
  vr = i64.add vjs vw64
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (va: i64, vb: i64) {
  v7 = i64.const 7
  return v7
  }
}
"#;

#[test]
fn threaded_unit_matches_oracle() {
    let m = parse(THREADED_UNIT);
    // Oracle: the whole unit on the bytecode cooperative driver (spawn + join + wait serviced there).
    let want = {
        let mut fuel = 50_000_000u64;
        match temen_interp::bytecode::compile_and_run(&m, 0, &[], &mut fuel) {
            Some(Ok(v)) => match v.first() {
                Some(Value::I64(x)) => *x,
                other => panic!("oracle result: {other:?}"),
            },
            other => panic!("oracle: {other:?}"),
        }
    };
    assert_eq!(want, 71, "oracle: join(7)*10 + wait-mismatch(1)");

    let wasm = compile_module_nested(&m, false).expect("threaded unit emits (nested)");
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("wasm validates");
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
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _c: i32| {})
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
             _w: i32,
             _i: i32,
             _e: i64,
             _o: i64,
             _s: i64,
             _q: i64|
             -> i32 { unreachable!("no instantiate in this unit") },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, HostState>, _i: i32, _c: i32| -> i64 {
                unreachable!("no instantiate join in this unit")
            },
        )
        .unwrap();
    // env.thread_spawn: run the unit's own `func` on the tree-walker (the servicer knows the module —
    // exactly the Worker host's position) and bank the result under a dense handle.
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |mut caller: Caller<'_, HostState>, func: i32, sp: i64, arg: i64| -> i32 {
                caller.data_mut().saw_bounce = true;
                let m = caller.data().module.clone();
                let mut fuel = u64::MAX;
                let r = match run(
                    &m,
                    func as u32,
                    &[Value::I64(sp), Value::I64(arg)],
                    &mut fuel,
                ) {
                    Ok(v) => match v.first() {
                        Some(Value::I64(x)) => *x,
                        other => panic!("spawned result: {other:?}"),
                    },
                    other => panic!("spawned run: {other:?}"),
                };
                let st = caller.data_mut();
                st.children.push(r);
                (st.children.len() - 1) as i32
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |caller: Caller<'_, HostState>, h: i32| -> i64 { caller.data().children[h as usize] },
        )
        .unwrap();
    // env.mem_wait: the futex compare against linear memory (win + masked addr) — a mismatch returns
    // 1 without blocking, exactly the engine's semantics; an equal value would block, which this
    // deterministic test never requests.
    let mem = memory;
    linker
        .func_wrap(
            "env",
            "mem_wait",
            move |caller: Caller<'_, HostState>,
                  win: i32,
                  addr: i64,
                  expected: i64,
                  _t: i64,
                  is64: i32|
                  -> i32 {
                let o = win as usize + (addr as usize & 0xFFFF);
                let data = mem.data(&caller);
                let cur = if is64 != 0 {
                    i64::from_le_bytes(data[o..o + 8].try_into().unwrap())
                } else {
                    i32::from_le_bytes(data[o..o + 4].try_into().unwrap()) as i64
                };
                if cur == expected {
                    panic!("deterministic test must not block");
                }
                1 // not-equal
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 { 0 },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("f0 exported");
    let params = [Val::I32(WIN_BASE), Val::I32(ENV_PTR)];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results).expect("f0 runs");

    assert!(
        store.data().saw_bounce,
        "spawn never bounced (silent fallback)"
    );
    assert_eq!(
        results[0].i64(),
        Some(want),
        "emitted threaded unit != cooperative oracle"
    );
}

/// **Track 1 — a `f64`-signature cross-tier leaf in a nested unit.** The entry (func 0) instantiates +
/// joins a child (so it is nested-emittable) and also calls a float helper `f2: (f64)->(f64)` that does
/// a scalar `f64.fma` (no core-wasm opcode → not emittable). Before the cross-tier ABI widened to
/// scalar floats, `f2`'s non-integer signature failed `compile_module_nested` closed
/// (`v128`/float couldn't be marshalled); now it rides `env.call_interp` as a cross-tier leaf, so the
/// unit compiles with `f2` unemitted. Compile-level regression — the emit is what gap-1 was blocking.
const NESTED_FLOAT_LEAF: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vinst = i32.wrap_i64 v0
  ventry = i64.const 1
  voff = i64.const 0
  vslog = i64.const 10
  vquota = i64.const 0
  vch = call.cap 6 0 (i64, i64, i64, i64) -> (i32) vinst (ventry, voff, vslog, vquota)
  vr = call.cap 6 1 (i32) -> (i64) vinst (vch)
  vf = f64.convert_i64_s vr
  vg = call 2 (vf)
  vi = i64.trunc_sat_f64_s vg
  return vi
  }
}
func () -> (i64) {
block 0 () {
  vr = i64.const 9
  return vr
  }
}
func (f64) -> (f64) {
block 0 (v0: f64) {
  vout = f64.fma v0 v0 v0
  return vout
  }
}
"#;

#[test]
fn nested_unit_with_float_signature_leaf_compiles() {
    let m = parse(NESTED_FLOAT_LEAF);
    let (_wasm, eligible) =
        compile_module_nested_with_eligibility(&m, false).expect("float-leaf nested unit compiles");
    // Entry (0) and the child (1) emit; the `f64.fma` helper (2) is a cross-tier leaf, not emitted.
    assert_eq!(eligible, vec![true, true, false]);
}

// ---- Track 2: the `compile_nested` two-mode front door -------------------------------------------

/// A pure `instantiate`/`join` unit (no fiber) takes the **wasm-driven** nested emit — the host calls
/// `f0` directly and the child spawn/join bounce to the imports. (`NESTED` reused from above.)
#[test]
fn compile_nested_pure_instantiator_is_wasm_driven() {
    let m = parse(NESTED);
    let a = compile_nested(&m, false).expect("nested compiles");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    assert_eq!(a.emitted, vec![true, true], "entry + child both emit");
}

/// A **threads/futex** unit is still wasm-driven — only fibers force the interpreter to own the frame.
/// `thread.spawn`/`join`/`wait`/`notify` lower to host bounces on the emit path. (`THREADED_UNIT`
/// reused from above.)
#[test]
fn compile_nested_threaded_unit_is_wasm_driven() {
    let m = parse(THREADED_UNIT);
    let a = compile_nested(&m, false).expect("nested compiles");
    assert_eq!(a.drive, DriveMode::WasmDriven { entry: 0 });
    assert_eq!(
        a.emitted,
        vec![true, true],
        "spawn entry + spawned func both emit"
    );
}

/// A nested unit whose **entry uses a fiber** (`cont.new`/`cont.resume`) can't be wasm-driven — a wasm
/// frame can't unwind for a stack switch — so `compile_nested` routes it **interpreter-driven** with a
/// `nested_caps`-aware tier-up: the fiber entry (func 0) runs on the interpreter (which also services
/// the child spawn/join natively), while the pure compute helper (func 1) and the fiber body (func 2,
/// also pure) tier up onto emitted wasm. This is the gap-2 close: before, such a unit hard-`Err`ed.
const FIBER_ENTRY_UNIT: &str = r#"memory 16
func () -> (i64) {
block 0 () {
  vf = ref.func 2
  varg = i64.const 0
  vk = cont.new vf varg
  vin = i64.const 5
  vtag, vres = cont.resume vk vin
  vh = call 1 (vres)
  return vh
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v3 = i64.const 3
  vr = i64.mul v0 v3
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  return varg
  }
}
"#;

#[test]
fn compile_nested_fiber_entry_is_interp_driven() {
    let m = parse(FIBER_ENTRY_UNIT);
    // The nested emit fails closed on the fiber, and the front door must not surface that as an error.
    let a = compile_nested(&m, false).expect("fiber-bearing nested unit still yields an artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);
    // Fiber entry stays on the interpreter; the pure helper and the (pure) fiber body tier up.
    assert_eq!(a.emitted, vec![false, true, true]);
}

/// The interpreter-driven fallback's tier-up wasm must be **well-formed** (the `nested_caps` import
/// layout + emitted-function base offset) and its emitted region correct. Instantiate the artifact
/// with the full eight-import nested linker and run the emitted pure helper `f1` — `v0*3`, matching the
/// interpreter — proving the `nested_caps`-aware tier-up emits a valid, correct module.
#[test]
fn compile_nested_fiber_entry_tierup_emit_runs() {
    let m = parse(FIBER_ENTRY_UNIT);
    let a = compile_nested(&m, false).expect("artifact");
    assert_eq!(a.drive, DriveMode::InterpDriven);

    let engine = Engine::default();
    let module =
        WModule::new(&engine, &a.wasm).expect("interp-driven nested tier-up wasm must validate");
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
    // All eight nested imports present; the emitted `f1`/`f2` are pure, so none fire.
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _c: i32| {})
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, HostState>, _f: i32, _a: i32| {
                unreachable!("pure emit — no cross-tier call")
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "instantiate",
            |_: Caller<'_, HostState>,
             _w: i32,
             _i: i32,
             _e: i64,
             _o: i64,
             _s: i64,
             _q: i64|
             -> i32 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "join",
            |_: Caller<'_, HostState>, _i: i32, _c: i32| -> i64 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_spawn",
            |_: Caller<'_, HostState>, _f: i32, _sp: i64, _a: i64| -> i32 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "thread_join",
            |_: Caller<'_, HostState>, _h: i32| -> i64 { unreachable!() },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_wait",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _e: i64, _t: i64, _is64: i32| -> i32 {
                unreachable!()
            },
        )
        .unwrap();
    linker
        .func_wrap(
            "env",
            "mem_notify",
            |_: Caller<'_, HostState>, _w: i32, _a: i64, _c: i32| -> i32 { unreachable!() },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f1 = instance.get_func(&store, "f1").expect("f1 emitted");
    let mut r = [Val::I64(0)];
    f1.call(
        &mut store,
        &[Val::I32(WIN_BASE), Val::I32(ENV_PTR), Val::I64(5)],
        &mut r,
    )
    .expect("emitted f1 runs");
    assert_eq!(
        r[0].i64(),
        Some(15),
        "emitted f1 = v0*3 (interpreter parity)"
    );
}
