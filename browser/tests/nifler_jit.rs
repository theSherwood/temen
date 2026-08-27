//! **nifler single-shot wasm-JIT differential** — the nimony-front-end twin of `chibicc_jit.rs`, and
//! slice 1 of wiring the multi-phase "compile Nim in the browser" card onto the wasm-JIT tier. The
//! playground runs `nifler.temen` (the first real nimony phase: Nim source → parsed `.p.nif`) whose
//! whole program is `_start` = func 0; this test emits that `_start` as wasm and runs `f0(win, env)`
//! once on `wasmi` (playing the browser's JS host), with nifler's `fopen`/`read`/`write`/`exit`
//! bouncing to the interpreter over the shared window through `env.call_interp` (so the seeded memfs
//! `/in.nim` and the powerbox `write` of `/out.p.nif` resolve). nifler's real output is a **file it
//! writes to the memfs**, so the run is opened with a readback key and its emitted `.p.nif` must be
//! **byte-identical** to the interpreter path ([`onramp_fs_exec_readback`], the oracle the shipped
//! single-phase nifler card is gated against) — the JIT correctness contract for the compiler tier.
//!
//! Fail-soft: `nifler.temen.gz` is a code-coupled asset CI regenerates; absent (or `gzip` unavailable),
//! the test SKIPs.

use std::io::Write;
use std::process::{Command, Stdio};

use temen_browser::{onramp_fs_exec_readback, JitOnrampRun, STATUS_EXIT, STATUS_OK};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

const WIN_LOG2: u8 = 25; // 32 MiB — the window the shipped nifler-on-wasm-JIT card uses (JIT_RUN_WIN_LOG2)
const WIN_SIZE: u64 = 1 << WIN_LOG2;
const WIN_BASE: u32 = 0x1_0000; // the window starts at 64 KiB (the env cell lives below it)
const ENV_PTR: u32 = 1024;

/// A small Nim program to parse; the differential holds for any input (the two tiers see the same memfs).
const IN_NIM: &str = "let x = 5\n";
/// The card's argv (mirrors `temen_run_nifler_fs` + the shipped browser card).
const ARGV: [&[u8]; 4] = [b"nifler", b"p", b"/in.nim", b"/out.p.nif"];

/// Decompress + decode the committed `nifler.temen.gz` asset (via `gzip`, as the JS host does before
/// handing raw module bytes to the FFI). `None` if the asset or `gzip` is missing (fail-soft SKIP).
fn nifler_temen() -> Option<temen_ir::Module> {
    let p = concat!(env!("CARGO_MANIFEST_DIR"), "/web/assets/nifler.temen.gz");
    let gz = std::fs::read(p).ok()?;
    let mut c = Command::new("gzip")
        .args(["-dc"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let mut stdin = c.stdin.take()?;
    let w = std::thread::spawn(move || {
        let _ = stdin.write_all(&gz);
    });
    let out = c.wait_with_output().ok()?;
    w.join().ok()?;
    out.status.success().then_some(())?;
    Some(temen_encode::decode_module(&out.stdout).expect("decode nifler.temen"))
}

/// The memfs the card seeds: the user's Nim source at `in.nim`.
fn card_image(src: &str) -> Vec<u8> {
    temen_fs::encode_image(&[("in.nim".to_string(), src.as_bytes().to_vec())], &[])
}

/// Parse `src` with nifler on the **wasm-JIT** (emitted `f0` on `wasmi`, cross-tier helpers on the
/// interpreter over the shared window). Returns the `.p.nif` nifler wrote to the memfs.
fn jit_parse(nifler: &temen_ir::Module, src: &str) -> Vec<u8> {
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
        JitOnrampRun::open_shared_run_fs_readback(
            nifler,
            win_ptr,
            WIN_SIZE,
            WIN_LOG2,
            false,
            &image,
            &ARGV,
            "out.p.nif".to_string(),
        )
    }
    .expect("nifler emittable as a single-shot JIT run");

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
                        // nifler grows its heap past its declared window, so without this the emitted
                        // store into a grown page traps against the stale declared bound.
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

    // nifler's `_start` is paramless (IMPORTS.md phase 4), so the emitted `f0` takes just (win, env).
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
    // nifler's `_start` returns (status 0) or calls `exit(0)` — either unwinds `f0`; a trap without an
    // `exit` recorded is a real fault.
    if let Err(e) = &call {
        if !run.exited() {
            panic!(
                "emitted f0 trapped (not an exit): {} ({})",
                e,
                run.last_trap()
            );
        }
    }
    run.output()
}

#[test]
fn nifler_jit_emits_identical_pnif() {
    let Some(nifler) = nifler_temen() else {
        eprintln!("SKIP: browser/web/assets/nifler.temen.gz absent or gzip unavailable");
        return;
    };

    // Interpreter oracle: the shipped single-phase nifler card path.
    let image = card_image(IN_NIM);
    let (oracle, expected) = onramp_fs_exec_readback(&nifler, &image, &ARGV, b"", "out.p.nif");
    assert!(
        oracle.status == STATUS_OK || oracle.status == STATUS_EXIT,
        "interp nifler status {}",
        oracle.status
    );
    assert!(
        !expected.is_empty(),
        "interp nifler produced a non-empty .p.nif"
    );

    // wasm-JIT: the emitted `_start` on wasmi, cross-tier I/O on the interpreter over the shared window.
    let jit = jit_parse(&nifler, IN_NIM);
    assert_eq!(
        jit, expected,
        "nifler on the wasm-JIT parses byte-identically to the interpreter (the compiler-tier contract)"
    );
}
