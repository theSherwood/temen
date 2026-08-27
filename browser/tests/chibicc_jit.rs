//! **chibicc single-shot wasm-JIT differential** — the fs+argv twin of `jit_module.rs`. The
//! playground C-compiler card runs `chibicc.temen` (whose whole program is `_start` = func 0) to emit
//! TEMEN-IR text; this test emits that `_start` as wasm and runs `f0(win, env)` once on `wasmi`
//! (playing the browser's JS host), with chibicc's `fopen`/`read`/`write`/`exit` bouncing to the
//! interpreter over the shared window through `env.call_interp` (so the seeded memfs `/in.c` +
//! `/include/*.h` and the powerbox `write` resolve). The emitted IR must be **byte-identical** to the
//! interpreter path ([`onramp_fs_exec`], the oracle the shipped card is gated against in
//! `chibicc_printf.rs`) — the JIT correctness contract for the compiler tier. Then it parses + runs the
//! emitted IR and checks the program's own stdout, so the whole card pipeline is covered on the JIT.
//!
//! A second test covers the **RUN path** (#1153): a chibicc-compiled program whose `malloc` `vm_map`s
//! its heap arena past the run window must, on the JIT, either match the interpreter or decline (trap →
//! browser fallback) — never complete with divergent output.
//!
//! Fail-soft: `chibicc.temen` is a code-coupled asset CI regenerates; absent, the test SKIPs.

use temen_browser::{
    onramp_exec, onramp_fs_exec, playground_include_files, JitOnrampRun, STATUS_EXIT, STATUS_OK,
};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 25; // 32 MiB run window (JIT_RUN_WIN_LOG2); chibicc declares size_log2=21 and grows into it
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;

fn chibicc_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/chibicc.temen");
    let bytes = std::fs::read(p).ok()?;
    Some(temen_encode::decode_module(&bytes).expect("decode chibicc.temen"))
}

/// The memfs the card seeds: built-in headers under `include/` + the user's source at `in.c`.
fn card_image(src: &str) -> Vec<u8> {
    let mut files: Vec<(String, Vec<u8>)> = playground_include_files();
    files.push(("in.c".to_string(), src.as_bytes().to_vec()));
    let dirs = vec!["include".to_string()];
    temen_fs::encode_image(&files, &dirs)
}

/// The card's argv (mirrors `chibicc_printf.rs` + the shipped browser card).
const ARGV: [&[u8]; 5] = [b"chibicc", b"--data-page", b"65536", b"-g0", b"/in.c"];

/// Compile `src` with chibicc on the **wasm-JIT** (emitted `f0` on `wasmi`, cross-tier helpers on the
/// interpreter over the shared window). Returns the emitted TEMEN-IR text (chibicc's stdout).
fn jit_compile(chibicc: &temen_ir::Module, src: &str) -> String {
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
    let rtys: Vec<temen_ir::ValType> = run.func_sig(0).1.to_vec();
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
                let args: Vec<temen_interp::Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                temen_ir::ValType::I32 => temen_interp::Value::I32(raw as i32),
                                _ => temen_interp::Value::I64(raw as i64),
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
                        // #1153: re-sync the emitted `"mapped"` global to the (possibly `vm_map`-grown)
                        // live extent after each bounce, exactly as `driveJitRun` does in the browser —
                        // chibicc declares a 2-MiB window and grows its heap past it, so without this the
                        // emitted store into a grown page traps against the stale declared bound.
                        let mapped = caller.data().as_ref().unwrap().mapped() as i64;
                        if let Some(wasmi::Extern::Global(g)) = caller.get_export("mapped") {
                            g.set(&mut caller, Val::I64(mapped)).ok();
                        }
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            if i >= results.len() {
                                break;
                            }
                            let raw = match v {
                                temen_interp::Value::I32(x) => *x as u32 as u64,
                                temen_interp::Value::I64(x) => *x as u64,
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
            temen_ir::ValType::I32 => Val::I32(0),
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
    if let Err(e) = &call {
        if !run.exited() {
            panic!(
                "emitted f0 trapped (not an exit): {} ({})",
                e,
                run.last_trap()
            );
        }
    }
    String::from_utf8(run.stdout().to_vec()).expect("emitted IR is utf8")
}

/// Compile `src` on the **interpreter** (the oracle), returning the emitted TEMEN-IR text.
fn interp_compile(chibicc: &temen_ir::Module, src: &str) -> String {
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
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: browser/web/assets/chibicc.temen absent (run build-onramp-assets.mjs)");
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
        "expected Temen IR, got: {jit_ir:.200}"
    );
    assert_eq!(
        jit_ir, interp_ir,
        "JIT-emitted IR must match the interpreter"
    );

    // 2. The whole card pipeline on the JIT: parse the emitted IR + run it, check the program's stdout.
    let m = temen_text::parse_module(&jit_ir).unwrap_or_else(|e| panic!("parse IR: {e:?}"));
    let run = onramp_exec(&m, b"");
    assert_eq!(run.status, STATUS_OK, "compiled program run status");
    assert_eq!(
        String::from_utf8_lossy(&run.stdout),
        "hello, jit! 2 + 40 = 42\nline 1\nline 2\nline 3\n"
    );
}

/// #1153 — the **RUN path** (the fn above is the COMPILE path): run a produced on-ramp module's
/// `_start` on the single-shot JIT (emitted `f0` on `wasmi`, cross-tier on the interpreter with the
/// `"mapped"` re-sync), returning its stdout and whether it **declined** (trapped without `exit` — an
/// access past the linear-memory window the browser resolves by falling back to the interpreter).
fn jit_run_module(m: &temen_ir::Module) -> (Vec<u8>, bool) {
    let engine = Engine::default();
    let pages = ((WIN_BASE as u64 + WIN_SIZE) / (64 * 1024)) as u32;
    let mut store: Store<Option<JitOnrampRun>> = Store::new(&engine, None);
    let memory = Memory::new(&mut store, MemoryType::new(pages, Some(pages))).unwrap();
    let win_ptr = unsafe { memory.data_mut(&mut store).as_mut_ptr().add(WIN_BASE as usize) };
    let run =
        unsafe { JitOnrampRun::open_shared_run(m, win_ptr, WIN_SIZE, WIN_LOG2, false, Vec::new()) }
            .expect("emittable");
    let emitted_wasm = run.emitted_wasm().to_vec();
    let rtys: Vec<temen_ir::ValType> = run.func_sig(0).1.to_vec();
    let module = WModule::new(&engine, &emitted_wasm).expect("validate");
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
                let args: Vec<temen_interp::Value> = {
                    let data = memory.data(&caller);
                    params
                        .iter()
                        .enumerate()
                        .map(|(i, t)| {
                            let o = args_ptr as usize + i * 8;
                            let raw = u64::from_le_bytes(data[o..o + 8].try_into().unwrap());
                            match t {
                                temen_ir::ValType::I32 => temen_interp::Value::I32(raw as i32),
                                _ => temen_interp::Value::I64(raw as i64),
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
                        // #1153: re-sync the emitted `"mapped"` bound after each bounce (as `driveJitRun`).
                        let mapped = caller.data().as_ref().unwrap().mapped() as i64;
                        if let Some(wasmi::Extern::Global(g)) = caller.get_export("mapped") {
                            g.set(&mut caller, Val::I64(mapped)).ok();
                        }
                        let data = memory.data_mut(&mut caller);
                        for (i, v) in vals.iter().enumerate() {
                            if i >= results.len() {
                                break;
                            }
                            let raw = match v {
                                temen_interp::Value::I32(x) => *x as u32 as u64,
                                temen_interp::Value::I64(x) => *x as u64,
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
    let f0 = instance.get_func(&store, "f0").unwrap();
    let args = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    let mut results: Vec<Val> = rtys
        .iter()
        .map(|t| match t {
            temen_ir::ValType::I32 => Val::I32(0),
            _ => Val::I64(0),
        })
        .collect();
    memory
        .write(&mut store, ENV_PTR as usize, &(1i64 << 60).to_le_bytes())
        .unwrap();
    let call = f0.call(&mut store, &args, &mut results);
    let run = store.data().as_ref().unwrap();
    let declined = call.is_err() && !run.exited();
    (run.stdout().to_vec(), declined)
}

/// #1153 run-path guard (the `chibicc libc`/`memstream` play cards, `browser-play-editor-test.mjs`):
/// chibicc compiles a program whose `malloc` `vm_map`s its heap arena at a high address (256 MiB, well
/// past the 32-MiB run window); running the produced IR on the JIT must either MATCH the interpreter or
/// DECLINE (trap → the browser falls back), never *complete* with divergent output. The regression this
/// pins: a cross-tier reservation clamped to the window turned that high `vm_map` into `-EINVAL`, so
/// `malloc` returned null and the run finished printing `(null)` instead of declining.
#[test]
fn chibicc_compiled_malloc_program_matches_or_declines() {
    let Some(chibicc) = chibicc_temen() else {
        eprintln!("SKIP: chibicc.temen absent (run build-onramp-assets.mjs)");
        return;
    };
    let src = "#include <stdio.h>\n#include <string.h>\n\
        int main(void){ char*d=strdup(\"libc\"); printf(\"%s\\n\", d); return 0; }\n";
    let image = card_image(src);
    let compiled = onramp_fs_exec(&chibicc, &image, &ARGV, b"");
    assert!(
        compiled.status == STATUS_OK || compiled.status == STATUS_EXIT,
        "chibicc compile status {}",
        compiled.status
    );
    let m = temen_text::parse_module(&String::from_utf8(compiled.stdout).expect("ir utf8"))
        .expect("parse produced IR");

    let interp_out = String::from_utf8_lossy(&onramp_exec(&m, b"").stdout).to_string();
    assert_eq!(interp_out, "libc\n", "interp oracle prints the strdup'd string");
    let (jit_bytes, declined) = jit_run_module(&m);
    let jit_out = String::from_utf8_lossy(&jit_bytes).to_string();
    assert!(
        declined || jit_out == interp_out,
        "JIT run must match the interpreter or decline, not diverge (#1153): got {jit_out:?} declined={declined}"
    );
}
