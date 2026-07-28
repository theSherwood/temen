//! **chibicc single-shot wasm-JIT differential** — the fs+argv twin of `jit_module.rs`. The
//! playground C-compiler card runs `chibicc.svmb` (whose whole program is `_start` = func 0) to emit
//! SVM-IR text; this test emits that `_start` as wasm and runs `f0(win, env)` once on `wasmi`
//! (playing the browser's JS host), with chibicc's `fopen`/`read`/`write`/`exit` bouncing to the
//! interpreter over the shared window through `env.call_interp` (so the seeded memfs `/in.c` +
//! `/include/*.h` and the powerbox `write` resolve). The emitted IR must be **byte-identical** to the
//! interpreter path ([`onramp_fs_exec`], the oracle the shipped card is gated against in
//! `chibicc_printf.rs`) — the JIT correctness contract for the compiler tier. Then it parses + runs the
//! emitted IR and checks the program's own stdout, so the whole card pipeline is covered on the JIT.
//!
//! Fail-soft: `chibicc.svmb` is a code-coupled asset CI regenerates; absent, the test SKIPs.

use svm_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, JitOnrampRun, STATUS_EXIT, STATUS_OK,
};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 25; // 32 MiB — chibicc.svmb declares size_log2=25; the emitted run can't grow it
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;

fn chibicc_svmb() -> Option<svm_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.svmb");
    let bytes = std::fs::read(p).ok()?;
    Some(svm_encode::decode_module(&bytes).expect("decode chibicc.svmb"))
}

/// The memfs the card seeds: built-in headers under `include/` + the user's source at `in.c`.
fn card_image(src: &str) -> Vec<u8> {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    svm_fs::encode_image(&files, &dirs)
}

/// The card's argv (mirrors `chibicc_printf.rs` + the shipped browser card).
const ARGV: [&[u8]; 5] = [b"chibicc", b"--data-page", b"65536", b"-g0", b"/in.c"];

/// Compile `src` with chibicc on the **wasm-JIT** (emitted `f0` on `wasmi`, cross-tier helpers on the
/// interpreter over the shared window). Returns the emitted SVM-IR text (chibicc's stdout).
fn jit_compile(chibicc: &svm_ir::Module, src: &str) -> String {
    let image = card_image(src);
    let engine = Engine::default();
    let pages = ((WIN_BASE as u64 + WIN_SIZE) / (64 * 1024)) as u32;
    let mut store: Store<Option<JitOnrampRun>> = Store::new(&engine, None);
    let memory =
        Memory::new(&mut store, MemoryType::new(pages, Some(pages))).expect("wasmi memory");

    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(WIN_BASE as usize)
    };
    // SAFETY: fixed-size memory ⇒ stable data pointer; the window lives inside it for the run.
    let run = unsafe {
        JitOnrampRun::open_shared_run_fs(
            chibicc,
            win_ptr,
            WIN_SIZE,
            WIN_LOG2,
            false,
            &image,
            &ARGV,
            Vec::new(),
        )
    }
    .expect("chibicc emittable as a single-shot JIT run");

    let emitted_wasm = run.emitted_wasm().to_vec();
    let rtys: Vec<svm_ir::ValType> = run.func_sig(0).1.to_vec();
    let module = WModule::new(&engine, &emitted_wasm).unwrap_or_else(|e| {
        panic!(
            "emitted _start ({} B) failed to validate: {e}",
            emitted_wasm.len()
        )
    });
    *store.data_mut() = Some(run);

    let mut linker: Linker<Option<JitOnrampRun>> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .func_wrap("env", "trap", |_c: Caller<'_, _>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut caller: Caller<'_, Option<JitOnrampRun>>,
                  func: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                let (params, results) = {
                    let r = caller.data().as_ref().unwrap();
                    let (p, rs) = r.func_sig(func as u32);
                    (p.to_vec(), rs.to_vec())
                };
                let args: Vec<svm_interp::Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                svm_ir::ValType::I32 => svm_interp::Value::I32(raw as i32),
                                _ => svm_interp::Value::I64(raw as i64),
                            }
                        })
                        .collect()
                };
                let outcome = caller
                    .data_mut()
                    .as_mut()
                    .unwrap()
                    .run_cross_tier(func as u32, &args);
                match outcome {
                    Ok(vals) => {
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            if i >= results.len() {
                                break;
                            }
                            let raw = match v {
                                svm_interp::Value::I32(x) => *x as u32 as u64,
                                svm_interp::Value::I64(x) => *x as u64,
                                _ => 0,
                            };
                            let o = args_ptr as usize + i * 8;
                            data[o..o + 8].copy_from_slice(&raw.to_le_bytes());
                        }
                        Ok(())
                    }
                    Err(t) => {
                        caller
                            .data_mut()
                            .as_mut()
                            .unwrap()
                            .set_last_trap(format!("{t:?}"));
                        Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ))
                    }
                }
            },
        )
        .unwrap();

    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    let f0 = instance.get_func(&store, "f0").expect("emitted f0 export");

    let args: Vec<Val> = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    let mut results: Vec<Val> = rtys
        .iter()
        .map(|t| match t {
            svm_ir::ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();

    // Huge dispatcher-fuel budget (debited per emitted dispatcher iteration).
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 60).to_le_bytes())
        .unwrap();
    let call = f0.call(&mut store, &args, &mut results);

    let run = store.data().as_ref().unwrap();
    // chibicc's `_start` calls `exit(0)` — that unwinds `f0` through `call_interp` and is expected; a
    // trap without an `exit` recorded is a real fault.
    if call.is_err() && !run.exited() {
        panic!(
            "emitted f0 trapped (not an exit): {} ({})",
            call.unwrap_err(),
            run.last_trap()
        );
    }
    String::from_utf8(run.stdout().to_vec()).expect("emitted IR is utf8")
}

/// Compile `src` on the **interpreter** (the oracle), returning the emitted SVM-IR text.
fn interp_compile(chibicc: &svm_ir::Module, src: &str) -> String {
    let image = card_image(src);
    let out = onramp_fs_exec(chibicc, &image, &ARGV, b"");
    assert!(
        out.status == STATUS_OK || out.status == STATUS_EXIT,
        "interp compile status {}",
        out.status
    );
    String::from_utf8(out.stdout).expect("IR is utf8")
}

#[test]
fn chibicc_jit_emits_identical_ir_and_runs() {
    let Some(chibicc) = chibicc_svmb() else {
        eprintln!("SKIP: browser/web/assets/chibicc.svmb absent (run build-onramp-assets.mjs)");
        return;
    };

    let src = r#"
#include <stdio.h>
int main(void) {
  printf("hello, %s! %d + %d = %d\n", "jit", 2, 40, 42);
  for (int i = 1; i <= 3; i++) printf("line %d\n", i);
  return 0;
}
"#;

    // 1. The JIT-emitted IR must byte-match the interpreter oracle.
    let jit_ir = jit_compile(&chibicc, src);
    let interp_ir = interp_compile(&chibicc, src);
    assert!(
        jit_ir.contains("func"),
        "expected SVM IR, got: {jit_ir:.200}"
    );
    assert_eq!(
        jit_ir, interp_ir,
        "JIT-emitted IR must match the interpreter"
    );

    // 2. The whole card pipeline on the JIT: parse the emitted IR + run it, check the program's stdout.
    let m = svm_text::parse_module(&jit_ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_OK, "compiled program run status");
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "hello, jit! 2 + 40 = 42\nline 1\nline 2\nline 3\n"
    );
}
