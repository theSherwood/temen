//! **Over-large-function exclusion** (#1011). V8 refuses to load any single wasm function body larger
//! than `kV8MaxWasmFunctionSize` (~7.65 MB), and one over-large body makes the *whole* module
//! unloadable — the wall nifler hit, whose linked-in-but-never-run Nim VM `rawExecute` lowers to an
//! ~8.5 MB body. `compile_module_reactor` measures each emitted body and pulls any that exceeds the
//! cap out of the emitted set, running it as a **cross-tier interpreter leaf** instead so the rest of
//! the module still JITs.
//!
//! The exclusion re-uses the existing cross-tier machinery (a direct `Call` to an excluded, marshallable
//! function is routed through `env.call_interp`, run on the interpreter over the *shared* window). This
//! differential proves that: `f0` (emitted) writes `mem[8]=x+7`, calls the over-large `f1` which reads
//! `mem[8]` and writes `mem[100]`, then `f0` reads `mem[100]`. With the shared window the round trip
//! yields `x+7`; the interpreter oracle over the same module must agree. Rather than build a genuine
//! multi-megabyte function, the test drives the real exclusion loop with a small cap via
//! `compile_module_reactor_capped` and a modest `f1`.

use std::sync::Arc;

use svm_interp::{bytecode, Region, Value};
use svm_wasm_jit::compile_module_reactor_capped;
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_BASE: u32 = 0x1_0000; // guest window at wasm offset 64 KiB (`memory 16`)
const WIN_SIZE: u64 = 1 << 16;
const ENV_PTR: u32 = 1024;

/// Build the two-function module. `f1` carries `n_dummy` no-effect stores so its emitted body clears a
/// small test cap (each masked `i64.store` lowers to ~27 bytes). `f0(x)` = `mem[100]` after calling
/// `f1`, which copies `mem[8]` (= x+7) into `mem[100]` — so a correct run yields `x+7` whether `f1`
/// ran as emitted wasm or as a cross-tier interpreter leaf over the shared window.
fn build_module(n_dummy: usize) -> svm_ir::Module {
    let mut src = String::from(
        r#"
memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  v7 = i64.const 7
  vpre = i64.add v0 v7
  va8 = i64.const 8
  i64.store va8 vpre
  vr1 = call 1 (v0)
  vaddr = i64.const 100
  vr = i64.load vaddr
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  va8 = i64.const 8
  vread = i64.load va8
  vaddr = i64.const 100
  i64.store vaddr vread
  vscr = i64.const 16
"#,
    );
    for _ in 0..n_dummy {
        src.push_str("  i64.store vscr v0\n");
    }
    src.push_str("  return vread\n  }\n}\n");
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    m
}

/// Largest function body in the code section (id 10) of an assembled module — the quantity V8 caps.
fn max_body_size(wasm: &[u8]) -> usize {
    fn uleb(p: &[u8], mut i: usize) -> (u64, usize) {
        let (mut v, mut s) = (0u64, 0u32);
        loop {
            let b = p[i];
            i += 1;
            v |= u64::from(b & 0x7f) << s;
            if b & 0x80 == 0 {
                return (v, i);
            }
            s += 7;
        }
    }
    let mut i = 8;
    while i < wasm.len() {
        let id = wasm[i];
        i += 1;
        let (sz, after) = uleb(wasm, i);
        i = after;
        if id == 10 {
            let (count, mut j) = uleb(wasm, i);
            let mut m = 0;
            for _ in 0..count {
                let (blen, a) = uleb(wasm, j);
                m = m.max(blen as usize);
                j = a + blen as usize;
            }
            return m;
        }
        i += sz as usize;
    }
    0
}

/// Full-interpreter oracle over `f0` (the root oracle, INVARIANTS.md #9).
fn oracle(m: &svm_ir::Module, arg: i64) -> i64 {
    let mut fuel = u64::MAX;
    match svm_interp::run(m, 0, &[Value::I64(arg)], &mut fuel) {
        Ok(v) => match v.first() {
            Some(Value::I64(x)) => *x,
            Some(Value::I32(x)) => *x as i64,
            _ => panic!("oracle result"),
        },
        other => panic!("oracle: {other:?}"),
    }
}

/// Run `f0` on wasmi with the given emitted wasm; any non-emitted `f1` runs on the interpreter over the
/// shared window through the `env.call_interp` servicer (mirrors the browser).
fn run(m: &svm_ir::Module, wasm: &[u8], arg: i64) -> i64 {
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm validates");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let memory = Memory::new(&mut store, MemoryType::new(2, None)).unwrap();
    memory
        .write(
            &mut store,
            ENV_PTR as usize,
            &1_000_000_000i64.to_le_bytes(),
        )
        .unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap::<_, ()>("env", "trap", |mut caller: Caller<'_, i32>, code: i32| {
            *caller.data_mut() = code;
        })
        .unwrap();

    let mod_cb = Arc::new(m.clone());
    let mem = memory;
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
                            off += if *t == svm_ir::ValType::V128 { 16 } else { 8 };
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                svm_ir::ValType::I32 => Value::I32(raw as i32),
                                svm_ir::ValType::I64 => Value::I64(raw as i64),
                                svm_ir::ValType::F32 => Value::F32(f32::from_bits(raw as u32)),
                                svm_ir::ValType::F64 => Value::F64(f64::from_bits(raw)),
                                _ => panic!("test callee is all-i64"),
                            }
                        })
                        .collect()
                };
                let base = mem.data_mut(&mut caller).as_mut_ptr();
                // SAFETY: single-threaded; the 2-page wasm memory outlives this call and does not grow.
                let back =
                    Arc::new(unsafe { Region::shared(base.add(WIN_BASE as usize), WIN_SIZE) });
                let mut fuel = u64::MAX;
                match bytecode::compile_and_run_capture_over(
                    &mod_cb,
                    func as u32,
                    &args,
                    &mut fuel,
                    &[],
                    back,
                ) {
                    Some((Ok(vals), _)) => {
                        let data = mem.data_mut(&mut caller);
                        let mut off = args_ptr as usize;
                        for v in vals.iter() {
                            let raw = match v {
                                Value::I32(x) => *x as u32 as u64,
                                Value::I64(x) => *x as u64,
                                Value::F32(x) => x.to_bits() as u64,
                                Value::F64(x) => x.to_bits(),
                                _ => panic!("test callee is all-i64"),
                            };
                            data[off..off + 8].copy_from_slice(&raw.to_le_bytes());
                            off += 8;
                        }
                        Ok(())
                    }
                    Some((Err(_), _)) => Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    )),
                    None => panic!("cross-tier callee unsupported"),
                }
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
    match results[0] {
        Val::I64(x) => x,
        Val::I32(x) => x as i64,
        _ => panic!("result type"),
    }
}

/// A small cap under which `f1`'s body lands: 3000 dummy stores (~81 KB body) over a 40 KB cap.
const SMALL_CAP: usize = 40_000;
const N_DUMMY: usize = 3000;

#[test]
fn oversized_marshallable_func_is_excluded_and_still_runs() {
    let m = build_module(N_DUMMY);

    // With the small cap, `f1` exceeds it → excluded from the emitted set, kept as a cross-tier leaf.
    let (wasm, emitted) = compile_module_reactor_capped(&m, 0, false, SMALL_CAP).expect("reactor");
    assert_eq!(
        emitted,
        vec![true, false],
        "f0 emits; over-large f1 is excluded"
    );
    assert!(
        max_body_size(&wasm) <= SMALL_CAP,
        "no emitted body exceeds the cap ({} > {SMALL_CAP})",
        max_body_size(&wasm)
    );
    // The excluded f1 still runs, on the interpreter over the shared window: f0(x) == x+7.
    for x in [0i64, 1, 35, -9] {
        assert_eq!(run(&m, &wasm, x), x + 7, "excluded-leaf run must match");
        assert_eq!(oracle(&m, x), x + 7, "oracle sanity");
    }
}

#[test]
fn no_exclusion_when_under_cap() {
    let m = build_module(N_DUMMY);
    // usize::MAX cap disables the exclusion — f1 emits like any in-subset function (default behavior).
    let (wasm, emitted) = compile_module_reactor_capped(&m, 0, false, usize::MAX).expect("reactor");
    assert_eq!(
        emitted,
        vec![true, true],
        "both functions emit when nothing is over-cap"
    );
    for x in [0i64, 1, 35, -9] {
        assert_eq!(run(&m, &wasm, x), x + 7, "all-emitted run must match");
    }
}
