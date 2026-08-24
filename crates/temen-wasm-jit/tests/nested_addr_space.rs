//! **§14 ADDRESS_SPACE on the wasm-JIT tier** — a nested unit's entry that *carves its own window*
//! (`sub`, iface 5 op 4) or queries `page_size` (op 3) now emits, riding the **one existing**
//! cross-tier transport: [`temen_wasm_jit::outline_nested_cap_calls`] hoists each such `cap.call` into
//! an int-signature wrapper leaf and [`temen_wasm_jit::compile_module_nested`] routes it through
//! `env.call_interp`, whose callback here carries the run's **powerbox** (the reactor-path contract).
//! No new imports — the CONSOLIDATION.md §0 yardstick.
//!
//! `map`/`unmap`/`protect` (ops 0/1/2) stay out-of-subset by design: they change page state that
//! subsequent *emitted* accesses can't honor under mask-only confinement (deferred with D40/§13).
//!
//! Oracle (INVARIANTS.md #9): the same entry run whole on the bytecode cooperative driver with an
//! identically-granted `Host` — grants encode deterministically, so `sub`'s minted handle and every
//! result must be byte-identical across tiers. The bounce counter is the non-vacuity guard.

use temen_interp::{bytecode, Host, Value};
use temen_wasm_jit::{compile_module_nested, outline_nested_cap_calls};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000;
const ENV_PTR: i32 = 1024;
const ENV_SCRATCH_OFF: usize = 16; // temen_wasm_jit's cross-tier scratch offset in the env cell
const WIN: u64 = 1 << 17; // `memory 17`

/// Entry `(address_space) -> i64`: `page_size` on the root, `sub`-carve a 64 KiB child at 64 KiB,
/// `page_size` again **through the minted child handle** (the attenuation is a live `AddressSpace`),
/// fold. Pure AddressSpace — pins the outlined bounce alone.
const AS_ONLY: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  vps = cap.call 5 3 () -> (i64) v0 ()
  voff = i64.const 65536
  vslog = i64.const 16
  vsub = cap.call 5 4 (i64, i64) -> (i32) v0 (voff, vslog)
  vps2 = cap.call 5 3 () -> (i64) vsub ()
  v3 = i64.const 3
  vt = i64.mul vps2 v3
  vr = i64.add vps vt
  return vr
  }
}
"#;

/// Entry `(address_space, instantiator) -> i64`: the §14 story in one function — `sub`-carve the
/// window (outlined bounce), then `instantiate` a child + `join` it (the dedicated import bounce),
/// folding both results. The child (func 1) mirrors the driver contract (`(i64) -> (i64)`, ignores
/// its attenuated-handle arg) and returns 9.
const COMPOSED: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vps = cap.call 5 3 () -> (i64) v0 ()
  voff = i64.const 65536
  vslog = i64.const 16
  vsub = cap.call 5 4 (i64, i64) -> (i32) v0 (voff, vslog)
  vps2 = cap.call 5 3 () -> (i64) vsub ()
  ventry = i64.const 1
  vzoff = i64.const 0
  vzslog = i64.const 10
  vquota = i64.const 0
  vch = cap.call 6 0 (i64, i64, i64, i64) -> (i32) v1 (ventry, vzoff, vzslog, vquota)
  vjr = cap.call 6 1 (i32) -> (i64) v1 (vch)
  vk = i64.const 100000
  vjs = i64.mul vjr vk
  v3 = i64.const 3
  vt = i64.mul vps2 v3
  vt2 = i64.add vps vt
  vr = i64.add vjs vt2
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
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

/// Grant the host exactly as both tiers must: AddressSpace over the whole window, then (iff the
/// entry takes it) an Instantiator. Same order ⇒ identical handle encodings across hosts.
fn granted_host(with_inst: bool) -> (Host, Vec<Value>) {
    let mut h = Host::new();
    let a = h.grant_address_space(0, WIN);
    let mut args = vec![Value::I32(a)];
    if with_inst {
        let i = h.grant_instantiator(0, WIN);
        args.push(Value::I32(i));
    }
    (h, args)
}

/// The whole-entry oracle: the bytecode cooperative driver over an identically-granted host — it
/// services the generic ADDRESS_SPACE dispatch *and* `instantiate`/`join` in one run.
fn oracle(m: &temen_ir::Module, with_inst: bool) -> i64 {
    let (mut host, args) = granted_host(with_inst);
    let mut fuel = 50_000_000u64;
    match bytecode::compile_and_run_with_host(m, 0, &args, &mut fuel, &mut host) {
        Some(Ok(v)) => match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("oracle result: {other:?}"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Host state behind the wasmi `Store`: the **outlined** module (wrappers live only there), the one
/// persistent powerbox-carrying `Host` (so `sub`'s mint lands in the same table the entry's handles
/// came from), spawned-child results, and the bounce counters (non-vacuity).
struct HostState {
    m: temen_ir::Module,
    host: Host,
    children: Vec<i64>,
    leaf_bounces: u32,
    inst_bounces: u32,
}

/// Run the emitted §14 entry under wasmi: outlined ADDRESS_SPACE wrappers bounce via
/// `env.call_interp` (serviced on the bytecode interp against the persistent host);
/// `instantiate`/`join` bounce via the dedicated imports (the child runs on the tree-walker).
fn emitted_run(src: &str, with_inst: bool) -> (i64, u32, u32) {
    let mut m = parse(src);
    outline_nested_cap_calls(&mut m);
    let wasm = compile_module_nested(&m, false).expect("nested entry emits");

    let (host, args) = granted_host(with_inst);
    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("nested wasm validates");
    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            m,
            host,
            children: Vec::new(),
            leaf_bounces: 0,
            inst_bounces: 0,
        },
    );
    let memory = Memory::new(&mut store, MemoryType::new(4, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();

    let mut linker: Linker<HostState> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _c: i32| {})
        .unwrap();
    // The outlined-wrapper bounce: decode the wrapper's args from the env scratch, run it on the
    // bytecode interpreter against the **persistent** host, write the results back to the same slots.
    let mem = memory;
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, HostState>, func: i32, args_ptr: i32| {
                let params: Vec<temen_ir::ValType> =
                    caller.data().m.funcs[func as usize].params.clone();
                let vals: Vec<Value> = {
                    let data = mem.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                temen_ir::ValType::I32 => Value::I32(raw as i32),
                                _ => Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                let st = caller.data_mut();
                st.leaf_bounces += 1;
                let mut fuel = 1_000_000u64;
                let m = st.m.clone();
                let out = bytecode::compile_and_run_with_host(
                    &m,
                    func as u32,
                    &vals,
                    &mut fuel,
                    &mut st.host,
                )
                .expect("wrapper in engine coverage")
                .expect("wrapper must not trap");
                let data = mem.data_mut(&mut caller);
                for (i, v) in out.iter().enumerate() {
                    let raw = match v {
                        Value::I32(x) => *x as u32 as u64,
                        Value::I64(x) => *x as u64,
                        _ => panic!("non-integer wrapper result"),
                    };
                    let o = args_ptr as usize + i * 8;
                    data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                }
            },
        )
        .unwrap();
    // The dedicated §14 spawn/join bounce (as in nested_vm.rs): run the child entry on the
    // tree-walker (the confinement carve is the host's job; the child here is pure).
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
                let st = caller.data_mut();
                st.inst_bounces += 1;
                let m = st.m.clone();
                let mut fuel = u64::MAX;
                let r = match temen_interp::run(&m, entry as u32, &[Value::I64(0)], &mut fuel) {
                    Ok(v) => match v.first() {
                        Some(Value::I64(x)) => *x,
                        other => panic!("child result: {other:?}"),
                    },
                    other => panic!("child run: {other:?}"),
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

    let mut params = vec![Val::I32(WIN_BASE), Val::I32(ENV_PTR)];
    for a in &args {
        match a {
            Value::I32(x) => params.push(Val::I32(*x)),
            Value::I64(x) => params.push(Val::I64(*x)),
            _ => unreachable!(),
        }
    }
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results).expect("f0 runs");
    let st = store.data();
    (
        results[0].i64().expect("i64"),
        st.leaf_bounces,
        st.inst_bounces,
    )
}

/// The outlined ADDRESS_SPACE bounce alone: `page_size` + `sub` + `page_size`-through-the-mint on
/// emitted wasm ≡ the whole-entry interpreter oracle (identical minted handle, identical values).
#[test]
fn addr_space_bounce_matches_interp() {
    let m = parse(AS_ONLY);
    let want = oracle(&m, false);
    let (got, leaves, insts) = emitted_run(AS_ONLY, false);
    assert!(leaves >= 3, "expected 3 wrapper bounces, saw {leaves}");
    assert_eq!(insts, 0);
    assert_eq!(got, want, "emitted AddressSpace entry != interpreter");
}

/// The composed §14 entry — `sub`-carve (outlined bounce) + `instantiate`/`join` (dedicated bounce)
/// in one function ≡ the whole-entry cooperative oracle.
#[test]
fn sub_carve_then_instantiate_matches_interp() {
    let m = parse(COMPOSED);
    let want = oracle(&m, true);
    let (got, leaves, insts) = emitted_run(COMPOSED, true);
    assert!(leaves >= 3, "expected 3 wrapper bounces, saw {leaves}");
    assert_eq!(insts, 1, "expected exactly one instantiate bounce");
    assert_eq!(got, want, "emitted composed §14 entry != interpreter");
}

/// `map`/`unmap`/`protect` stay fail-closed: an entry using `unmap` (op 1) must not emit — the whole
/// unit falls back to the interpreter (mask-only confinement can't honor page state; D40/§13).
#[test]
fn page_state_ops_stay_out_of_subset() {
    let src = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  voff = i64.const 0
  vlen = i64.const 16384
  vr = cap.call 5 1 (i64, i64) -> (i64) v0 (voff, vlen)
  return vr
  }
}
"#;
    let mut m = parse(src);
    outline_nested_cap_calls(&mut m); // must NOT outline op 1
    assert!(
        compile_module_nested(&m, false).is_err(),
        "an unmap-using entry must fail closed (interp fallback)"
    );
}

/// Unused scratch-offset sanity: the harness's arg decoding assumes the emitter's documented layout.
#[test]
fn scratch_layout_assumption() {
    assert_eq!(ENV_SCRATCH_OFF, 16);
    assert!(ENV_PTR as usize + temen_wasm_jit::ENV_CELL_BYTES < WIN_BASE as usize);
}
