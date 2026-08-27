//! **#1011 slice 3a — a granted child runs on the wasm-JIT tier (differential).** A §14 phase child
//! carries a re-granted powerbox (a shared `fs`), and its `call.cap` on that grant must produce the
//! same result on the **emitted** tier as on the interpreter oracle (INVARIANTS #9). The child here is
//! `f0` (emitted integer compute) + `f1` (a cross-tier leaf that resolves the granted cap `"fs"` by
//! name and calls it): `f0(x) = 40 + f1()`, and `f1()` calls a granted counter (post-increment `1`),
//! so a correct run returns `41`. On the emitted tier `f0`'s call to `f1` bounces cross-tier
//! (`env.call_interp`) and runs on the interpreter **over the granted host** — so the grant resolves
//! there exactly as a JIT'd nim phase child's `fs` call will. Window confinement (§2) is unchanged:
//! the grant is a `call.cap`, not a window access.

use std::sync::{Arc, Mutex};

use temen_interp::{bytecode, ForkedProc, Host, HostProc, Region, Value};
use temen_wasm_jit::compile_module_reactor;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000; // guest window at wasm offset 64 KiB (`memory 16`)
const WIN_SIZE: u64 = 1 << 16;
const ENV_PTR: u32 = 1024;

// f0 (emitted): 40 + f1(v0). f1 (cross-tier: it makes a `call.cap`): seed "fs" (0x7366 LE) into the
// window, resolve it, call the granted HOST_PROC counter, return its result.
const SRC: &str = r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v40 = i64.const 40
  vcap = call 1 (v0)
  vr = i64.add v40 vcap
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vname = i64.const 29542
  vzero = i64.const 16384
  i64.store vzero vname
  vp0 = i64.const 16384
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vr = call.cap 13 0 (i64) -> (i64) vh (vp0)
  return vr
  }
}
"#;

fn parse() -> temen_ir::Module {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

// A fresh host holding the granted counter cap registered as `"fs"` (the shape a shared memfs takes),
// plus the shared counter so a call from inside the child is observable.
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

/// Interpreter oracle: run `f0` over a flat window with the granted host — `f1`'s `call.cap` resolves
/// `"fs"` inline. Returns `f0(arg)`.
fn oracle(m: &temen_ir::Module, arg: i64) -> (i64, i64) {
    let (mut host, counter) = granted_host();
    let prog = bytecode::SharedProgram::compile(m).expect("compile");
    let mut backing = vec![0u8; WIN_SIZE as usize].into_boxed_slice();
    // SAFETY: `backing` outlives the run; the region is this call's exclusive window.
    let back = Arc::new(unsafe { Region::shared(backing.as_mut_ptr(), WIN_SIZE) });
    let mut fuel = u64::MAX;
    let r = prog
        .run_over(0, &[Value::I64(arg)], &mut fuel, back, &mut host, true)
        .expect("oracle run");
    let out = match r.first() {
        Some(Value::I64(x)) => *x,
        Some(Value::I32(x)) => *x as i64,
        _ => panic!("oracle result"),
    };
    let cval = *counter.lock().unwrap();
    (out, cval)
}

/// JIT run: `f0` on wasmi; `f1` bounces cross-tier and runs on the interpreter **over the granted
/// host** (the browser cross-tier path), so its `call.cap` resolves `"fs"`. Returns `f0(arg)`.
fn jit_run(m: &temen_ir::Module, arg: i64) -> (i64, i64) {
    let (wasm, emitted) = compile_module_reactor(m, 0, false).expect("reactor emittable");
    assert_eq!(
        emitted,
        vec![true, false],
        "f0 emits; the call.cap leaf f1 is cross-tier"
    );

    let (host, counter) = granted_host();
    let host = Arc::new(Mutex::new(host));
    let prog = Arc::new(bytecode::SharedProgram::compile(m).expect("compile"));

    let engine = Engine::default();
    let module = WModule::new(&engine, &wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap::<_, ()>("env", "trap", |_caller: Caller<'_, i32>, _code: i32| {})
        .unwrap();

    let mem = memory;
    let prog_cb = Arc::clone(&prog);
    let host_cb = Arc::clone(&host);
    let mod_cb = Arc::new(m.clone());
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, i32>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let callee = &mod_cb.funcs[func as usize];
                let args: Vec<Value> = {
                    let data = mem.data(&caller);
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
                let base = mem.data_mut(&mut caller).as_mut_ptr();
                // SAFETY: single-threaded; the 2-page wasm memory outlives the call and does not grow.
                let back =
                    Arc::new(unsafe { Region::shared(base.add(WIN_BASE as usize), WIN_SIZE) });
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
                let data = mem.data_mut(&mut caller);
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

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("f0");
    let mut results = [Val::I64(0)];
    f0.call(
        &mut store,
        &[
            Val::I32(WIN_BASE as i32),
            Val::I32(ENV_PTR as i32),
            Val::I64(arg),
        ],
        &mut results,
    )
    .expect("f0 runs");
    let out = match results[0] {
        Val::I64(x) => x,
        Val::I32(x) => x as i64,
        _ => panic!("result type"),
    };
    let cval = *counter.lock().unwrap();
    (out, cval)
}

#[test]
fn granted_child_cap_call_matches_on_both_tiers() {
    let m = parse();
    let (oi, oc) = oracle(&m, 0);
    let (ji, jc) = jit_run(&m, 0);
    assert_eq!(oi, 41, "interpreter: 40 + granted counter (1)");
    assert_eq!(ji, 41, "wasm-JIT: 40 + the cross-tier granted counter (1)");
    assert_eq!(oi, ji, "the granted child's call.cap agrees on both tiers");
    assert_eq!(
        (oc, jc),
        (1, 1),
        "the granted handler ran once on each tier"
    );
}
