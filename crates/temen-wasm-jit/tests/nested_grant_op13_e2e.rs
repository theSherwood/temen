//! **§14 op-13 end-to-end on the emitted tier** (issue #1123, slice B) — the wasmi twin of the native
//! `rust_guest_op13` / `nifler_child_jit`: an **emitted op-13 parent** spawns a **separate-module child
//! that does real granted work** (a `call.cap` on a re-granted `"fs"`), and the whole thing runs on
//! emitted wasm, byte-identical to the interpreter.
//!
//! This composes the two halves proven separately elsewhere:
//!   - the op-13 emitted parent (`nested_emitted_child.rs`, PR #1129): `call.cap 6 13` lowers to the
//!     `env.instantiate_module` bounce, so the driver runs emitted;
//!   - the granted child cross-tier `call.cap` (`nested_grant_jit.rs`, #1011 3a): the child's `f0`
//!     emits, its `call.cap` leaf `f1` bounces via `env.call_interp` and runs on the interpreter **over
//!     the child's carve** against the granted host, so the grant resolves exactly as a nimony phase
//!     child's `fs` call will.
//!
//! The child returns `40 + granted_counter()` = `41`; a correct run returns `41` and the shared counter
//! ticks exactly once — the observable proof the re-granted authority ran *inside the confined, emitted
//! child*. Confinement (§2/§4) is unchanged: the grant is a `call.cap` (authority, §3), not a window
//! access; the child's window base is its carve.

use std::sync::{Arc, Mutex};

use temen_interp::{bytecode, ForkedProc, Host, HostProc, Region, Value};
use temen_wasm_jit::{compile_module_nested, compile_module_nested_with_eligibility};
use wasmi::{Caller, Engine, Func, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: i32 = 0x1_0000; // parent window base (the env cell lives below it)
const ENV_PTR: i32 = 1024; // parent dispatcher-fuel cell
const CHILD_ENV_PTR: i32 = 512; // child dispatcher-fuel cell (below the parent's scratch)
const CARVE_OFF: i64 = 4096; // the child's carve, offset into the parent window
const CHILD_WIN_SIZE: u64 = 1 << 12; // 4 KiB — matches the child's `memory 12`

/// The **separate** granted child (from `nested_grant_jit.rs`, adapted to a paramless entry). `f0`
/// (emitted): `40 + f1()`. `f1` (a cross-tier `call.cap` leaf): seed `"fs"` (`0x7366` LE) into its
/// window, resolve it, call the granted `HOST_PROC` counter (post-increment `1`), return its result.
const CHILD: &str = r#"memory 12
func () -> (i64) {
block 0 () {
  v40 = i64.const 40
  vcap = call 1 ()
  vr = i64.add v40 vcap
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vname = i64.const 29542
  vzero = i64.const 0
  i64.store vzero vname
  vp0 = i64.const 0
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

/// The op-13 parent driver: `v0` = `Instantiator` handle, `v1` = child `Module` handle. Spawns the
/// module into the 4-KiB carve at `CARVE_OFF` (empty grant list — the host binds `"fs"`), `join`s it,
/// and returns the child's result. Emittable via the `env.instantiate_module` bounce (PR #1129).
const PARENT_OP13: &str = r#"memory 16
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  vmh = i64.extend_i32_u v1
  vgptr = i64.const 0
  vgn = i64.const 0
  ventry = i64.const 0
  voff = i64.const 4096
  vsl = i64.const 12
  vq = i64.const 0
  vh = call.cap 6 13 (i64, i64, i64, i64, i64, i64, i64) -> (i32) v0 (vmh, vgptr, vgn, ventry, voff, vsl, vq)
  vr = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vr
  }
}
"#;

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// A fresh host holding the granted counter cap registered as `"fs"` (the re-grantable form a shared
/// memfs takes), plus the shared counter so a call from inside the confined child is observable here.
fn granted_host() -> (Host, Arc<Mutex<i64>>) {
    let counter = Arc::new(Mutex::new(0i64));
    let mut host = Host::new();
    let c1 = Arc::clone(&counter);
    let handler: HostProc = Box::new(move |_op, _args, _mem, _| {
        let mut c = c1.lock().unwrap();
        *c += 1;
        Ok(vec![*c])
    });
    let c2 = Arc::clone(&counter);
    let fork = Arc::new(move |_pid: u64| {
        let c = Arc::clone(&c2);
        ForkedProc::shared(Box::new(move |_op, _args, _mem, _| {
            let mut c = c.lock().unwrap();
            *c += 1;
            Ok(vec![*c])
        }))
    });
    let h = host.grant_host_proc_forkable(handler, fork);
    host.register_cap_name("fs", h);
    (host, counter)
}

/// Interpreter oracle: run the child's `f0` over a flat window with the granted host — `f1`'s `call.cap`
/// resolves `"fs"` inline. Returns `(f0_result, counter)`.
fn oracle_child(child: &temen_ir::Module) -> (i64, i64) {
    let (mut host, counter) = granted_host();
    let prog = bytecode::SharedProgram::compile(child).expect("compile child");
    let mut backing = vec![0u8; CHILD_WIN_SIZE as usize].into_boxed_slice();
    // SAFETY: `backing` outlives the run; the region is this call's exclusive window.
    let back = Arc::new(unsafe { Region::shared(backing.as_mut_ptr(), CHILD_WIN_SIZE) });
    let mut fuel = u64::MAX;
    let r = prog
        .run_over(0, &[], &mut fuel, back, &mut host, true)
        .expect("oracle run");
    let out = match r.first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        other => panic!("oracle result: {other:?}"),
    };
    let cval = *counter.lock().unwrap();
    (out, cval)
}

/// Host state: the pre-built child's emitted entry `f0`, and the banked child results the parent joins.
struct HostState {
    child_entry: Option<Func>,
    children: Vec<i64>,
}

#[test]
fn op13_emitted_parent_spawns_granted_child_e2e() {
    let parent = parse(PARENT_OP13);
    let child = parse(CHILD);

    let (want, want_counter) = oracle_child(&child);
    assert_eq!(want, 41, "interpreter oracle: 40 + granted counter (1)");
    assert_eq!(
        want_counter, 1,
        "the granted handler ran once on the interpreter"
    );

    // The child's `f0` emits; its `call.cap` leaf `f1` stays a cross-tier `env.call_interp` bounce.
    let (child_wasm, eligible) =
        compile_module_nested_with_eligibility(&child, false).expect("child emits (nested)");
    assert_eq!(
        eligible,
        vec![true, false],
        "child f0 emits; the call.cap leaf f1 is cross-tier"
    );
    let parent_wasm = compile_module_nested(&parent, false).expect("op-13 parent emits (nested)");

    let engine = Engine::default();
    let parent_module = WModule::new(&engine, &parent_wasm).expect("op-13 parent wasm validates");
    let child_module = WModule::new(&engine, &child_wasm).expect("child wasm validates");

    let mut store: Store<HostState> = Store::new(
        &engine,
        HostState {
            child_entry: None,
            children: Vec::new(),
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

    // The child's granted host + a bytecode program for its cross-tier leaf. Shared into `env.call_interp`.
    let (host, counter) = granted_host();
    let host = Arc::new(Mutex::new(host));
    let child_prog =
        Arc::new(bytecode::SharedProgram::compile(&child).expect("compile child prog"));
    let child_mod = Arc::new(child.clone());
    let carve_base = (WIN_BASE as i64 + CARVE_OFF) as usize;

    let mut linker: Linker<HostState> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_: Caller<'_, HostState>, _code: i32| {})
        .unwrap();

    // env.call_interp: the child's `f1` leaf runs on the interpreter **over the child's carve** against
    // the granted host, so its `call.cap` resolves `"fs"` — the browser cross-tier path, at the carve.
    {
        let host_cb = Arc::clone(&host);
        let prog_cb = Arc::clone(&child_prog);
        let mod_cb = Arc::clone(&child_mod);
        linker
            .func_wrap(
                "env",
                "call_interp",
                move |mut caller: Caller<'_, HostState>,
                      func: i32,
                      args_ptr: i32|
                      -> Result<(), wasmi::Error> {
                    let callee = &mod_cb.funcs[func as usize];
                    let args: Vec<Value> = {
                        let data = memory.data(&caller);
                        let mut off = args_ptr as usize;
                        callee
                            .params
                            .iter()
                            .map(|t| {
                                let o = off;
                                off += 8;
                                let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                                match t {
                                    temen_ir::ValType::I32 => Value::I32(raw as i32),
                                    _ => Value::I64(raw as i64),
                                }
                            })
                            .collect()
                    };
                    let base = memory.data_mut(&mut caller).as_mut_ptr();
                    // SAFETY: single-threaded; the 2-page wasm memory outlives the call and never grows.
                    // The child's window IS its carve, so the interp runs `f1` over `[carve, carve+size)`.
                    let back =
                        Arc::new(unsafe { Region::shared(base.add(carve_base), CHILD_WIN_SIZE) });
                    let mut fuel = u64::MAX;
                    let r = prog_cb
                        .run_over(
                            func as u32,
                            &args,
                            &mut fuel,
                            back,
                            &mut host_cb.lock().unwrap(),
                            false,
                        )
                        .map_err(|_| {
                            wasmi::Error::from(wasmi::core::TrapCode::UnreachableCodeReached)
                        })?;
                    let data = memory.data_mut(&mut caller);
                    let mut off = args_ptr as usize;
                    for v in &r {
                        let raw = match v {
                            Value::I32(x) => *x as u32 as u64,
                            Value::I64(x) => *x as u64,
                            _ => 0,
                        };
                        data[off..off + 8].copy_from_slice(&raw.to_le_bytes());
                        off += 8;
                    }
                    Ok(())
                },
            )
            .unwrap();
    }

    // env.instantiate_module (op 13): emit + run the resolved child over the carve and bank its result.
    linker
        .func_wrap(
            "env",
            "instantiate_module",
            |mut caller: Caller<'_, HostState>,
             win: i32,
             _inst: i32,
             _module: i64,
             _grants_ptr: i64,
             _grants_n: i64,
             _entry: i64,
             off: i64,
             _slog: i64,
             _quota: i64|
             -> i32 {
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
    // The remaining nested imports are present (uniform layout) but never fire in this unit.
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
             -> i32 { unreachable!("no op-0 instantiate in this unit") },
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

    // Pre-build the child as its own wasmi instance over the shared memory; stash its emitted entry.
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

    // f0(win, env, inst_handle, module_handle).
    let params = [
        Val::I32(WIN_BASE),
        Val::I32(ENV_PTR),
        Val::I32(7),
        Val::I32(99),
    ];
    let mut results = [Val::I64(0)];
    f0.call(&mut store, &params, &mut results)
        .expect("op-13 parent f0 runs");

    assert_eq!(
        results[0].i64(),
        Some(want),
        "emitted op-13 parent + emitted granted child != interpreter oracle (41)"
    );
    assert_eq!(
        *counter.lock().unwrap(),
        1,
        "the re-granted `fs` ran exactly once inside the confined, emitted child"
    );
}
