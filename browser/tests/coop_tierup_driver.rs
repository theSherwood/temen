//! **Cooperative on-ramp tier-up** (#926 slice 2) — the native differential for the `svm_coop_*`
//! event-pump FFI. A genuinely threaded on-ramp guest (its `_start` `thread.spawn`s a worker, so the
//! single-vCPU `svm_onramp_tierup_*` driver would *decline* it at the spawn event) must run observably
//! identical to the plain bytecode path (`onramp_exec`, INVARIANTS.md #9) when its hot leaf is serviced
//! on the emitted wasm — with wasmi playing the browser's JS host (`driveCoopTierupRun`). The
//! cooperative driver multiplexes the root and the worker on one wasm thread and tiers up **both**
//! their calls to the eligible leaf; the non-vacuity count pins that two tier-ups (one per vCPU) fire.
//!
//! This proves the cooperative-driver FFI wiring end-to-end: open → pump → service → deliver → capture,
//! over a thread topology the single-vCPU pump cannot run at all. The engine seam itself (multi-task
//! resumable tier-up + `deliver` routing) is pinned natively in `svm-interp/tests/coop_tierup.rs`.

use std::sync::Mutex;
use svm_browser::{
    onramp_exec, svm_coop_argv_len, svm_coop_argv_ptr, svm_coop_close, svm_coop_deliver,
    svm_coop_deliver_jit, svm_coop_deliver_jit_trap, svm_coop_deliver_trap, svm_coop_func,
    svm_coop_jit_result_types_len, svm_coop_jit_wasm_len, svm_coop_jit_wasm_ptr, svm_coop_mapped,
    svm_coop_open, svm_coop_run, svm_coop_value, svm_coop_wasm_len, svm_coop_wasm_ptr,
    svm_coop_win_len, svm_coop_win_ptr, svm_run_value, svm_status, svm_stdout_len, svm_stdout_ptr,
    COOP_RUN_DONE, COOP_RUN_JIT_INVOKE, COOP_RUN_TIERUP, STATUS_OK,
};
use svm_interp::{Host, StreamRole};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

/// The coop session statics are process-global (single-threaded wasm by design) — serialize the tests
/// in this binary across them.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// Where the wasmi harness places the mirrored window / env cell in the emitted module's memory.
const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
/// Declared-prefix cell `_start` stages the summed result in before streaming it to stdout.
const SLOT: i64 = 2048;

/// The on-ramp powerbox's stdout handle, replicated from `grant_onramp_caps`'s grant order (stdout,
/// stdin, exit, memory, addrspace, …) against a fresh `Host` — deterministic per session, so the
/// import-free guest text can `cap.call` it directly.
fn onramp_out_handle() -> i32 {
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    let _ = h.grant_memory();
    out
}

/// The guest: `_start` (interp-driven — it `thread.spawn`s, so it stays on the interpreter) spawns a
/// worker (func 1), calls the pure all-i64 leaf (func 2) itself with `3`, joins the worker (which
/// called the leaf with `5`), sums the two results, stages the sum at `SLOT`, writes those 8 bytes to
/// stdout, and returns the sum. The leaf `f(x) = 3x + 7` is the tier-up target — **both** the root's
/// `call 2` and the worker's tier up. Sum = f(3) + f(5) = 16 + 22 = 38.
fn coop_guest_text() -> String {
    let out_h = onramp_out_handle();
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  v3 = i64.const 3
  vlocal = call 2 (v3)
  vj = thread.join vt
  vsum = i64.add vj vlocal
  vsl = i64.const {SLOT}
  i64.store vsl vsum
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  v5 = i64.const 5
  vr = call 2 (v5)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  v3 = i64.const 3
  vm = i64.mul vx v3
  v7 = i64.const 7
  va = i64.add vm v7
  return va
  }}
}}
export 0 func "_start" 0
"#
    )
}

/// Service one cooperative TIERUP on wasmi (the browser JS host's `driveCoopTierupRun` role): mirror
/// the live window into a fresh instance's memory, write the event's `"mapped"` sync (#717), call
/// `f{func}(win, env, ...argv)`, copy the window back, and return the i64 results (or `None` on a wasm
/// trap — delivered as a trap to the paused task). The pure leaf never bounces, so `env.call_interp`
/// is a trap-stub here.
fn service_coop_on_wasmi(n_results: usize) -> Option<Vec<i64>> {
    // SAFETY: the paused task is parked inside the TIERUP event; the session stash (wasm, argv,
    // window) is stable until the deliver call, and this thread is the only accessor (FFI_LOCK).
    let wasm = unsafe { std::slice::from_raw_parts(svm_coop_wasm_ptr(), svm_coop_wasm_len()) };
    let func = svm_coop_func();
    let argv = unsafe { std::slice::from_raw_parts(svm_coop_argv_ptr(), svm_coop_argv_len()) };
    let win_len = svm_coop_win_len();
    let win_ptr = svm_coop_win_ptr() as *mut u8;
    let mapped = svm_coop_mapped();

    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = ((WIN_BASE as usize + win_len) as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    // SAFETY: see above — exclusive mirror of the parked window.
    let live = unsafe { std::slice::from_raw_parts(win_ptr, win_len) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();

    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    // A closed leaf never `call_indirect`s; an all-null table satisfies the B2 import (unused here).
    let table = wasmi::Table::new(
        &mut store,
        wasmi::TableType::new(wasmi::core::ValType::FuncRef, 1 << 10, Some(1 << 10)),
        Val::FuncRef(wasmi::FuncRef::null()),
    )
    .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call expected from this pure leaf");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // #717 driver contract: the event's committed-extent snapshot lands in the emitted `"mapped"`
    // global before the call (the fuel global self-initializes to the standard budget).
    instance
        .get_global(&store, "mapped")
        .expect("emitted module exports the live-mapped global")
        .set(&mut store, Val::I64(mapped))
        .unwrap();
    let entry = format!("f{func}");
    let f = instance
        .get_func(&store, &entry)
        .unwrap_or_else(|| panic!("{entry} not exported"));

    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(argv.iter().map(|a| Val::I64(*a)));
    let mut results: Vec<Val> = (0..n_results).map(|_| Val::I64(0)).collect();
    let ran = f.call(&mut store, &params, &mut results);

    // Copy emitted writes back into the live window before the task resumes.
    let mut buf = vec![0u8; win_len];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    // SAFETY: see above — exclusive mirror of the parked window.
    unsafe { std::slice::from_raw_parts_mut(win_ptr, win_len) }.copy_from_slice(&buf);

    match ran {
        Ok(()) => Some(
            results
                .iter()
                .map(|v| match v {
                    Val::I64(x) => *x,
                    Val::I32(x) => *x as i64,
                    _ => panic!("non-integer result"),
                })
                .collect(),
        ),
        Err(_) => None,
    }
}

// ---- #926 slice 2e: the §22 `Jit.invoke` half — a threaded `vm_jit_*` guest ----------------------

/// Leaf probe, blob-staging base, and the unit's added constant — mirrors the single-vCPU jit test.
const PROBE: i64 = 65536 + 16;
const BLOB_BASE: i64 = 4096;
const UNIT_K: i64 = 90909;

/// The on-ramp stdout + memory (`AddressSpace`) handles, for a guest that both `vm_map`-grows and
/// streams — replicated from `grant_onramp_caps`'s grant order.
fn onramp_out_mem_handles() -> (i32, i32) {
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    let mem = h.grant_memory();
    (out, mem)
}

/// The unit `vm_jit_compile` compiles: `f(x) = x + UNIT_K`, storing to the grown page `[x+8]` and
/// loading it back (so its emitted run exercises the grown-window `"mapped"` bound).
fn unit_blob() -> Vec<u8> {
    let src = format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {UNIT_K}
  vsum = i64.add v0 vk
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr vsum
  vld = i64.load vaddr
  return vld
  }}
}}
"#
    );
    let m = svm_text::parse_module(&src).expect("unit parse");
    svm_verify::verify_module(&m).expect("unit verify");
    svm_encode::encode_module(&m)
}

/// A **threaded** `vm_jit_*` guest (the JACL-with-runtime-threads shape): `_start` (the root vCPU)
/// spawns a worker, `vm_map`-grows `[64 KiB, 80 KiB)`, stages `blob`, `vm_jit_compile`s it,
/// `vm_jit_invoke2`s the unit with the grown-page probe, joins the worker (which just returns 0), and
/// streams the sum. The `thread.spawn` is what routes it to the cooperative driver (the single-vCPU
/// pump declines it); the `vm_jit_invoke2` is what surfaces `COOP_RUN_JIT_INVOKE`.
fn coop_jit_guest_text(blob: &[u8]) -> String {
    let (out_h, mem_h) = onramp_out_mem_handles();
    let blob_len = blob.len();
    let mut stores = String::new();
    for (i, chunk) in blob.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let val = i64::from_le_bytes(word);
        let addr = BLOB_BASE + (i as i64) * 8;
        stores.push_str(&format!(
            "  va{i} = i64.const {addr}\n  vv{i} = i64.const {val}\n  i64.store va{i} vv{i}\n"
        ));
    }
    format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
import 1 "vm_jit_invoke2" (i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 1 vz vz
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
{stores}  vbp = i64.const {BLOB_BASE}
  vbl = i64.const {blob_len}
  vcode = call.import 0 (vbp, vbl)
  vprobe = i64.const {PROBE}
  vres = call.import 1 (vcode, vprobe)
  vj = thread.join vt
  vsum = i64.add vres vj
  vsl = i64.const {SLOT}
  i64.store vsl vsum
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vz = i64.const 0
  return vz
  }}
}}
export 0 func "_start" 0
"#
    )
}

/// Instantiate `wasm` over a mirrored window and run `entry(win, env, ...argv)` with the event's
/// `"mapped"` sync — the shared wasmi half reused by the JIT_INVOKE service (`f0`). Returns the i64
/// results, or `None` on a wasm trap.
fn run_emitted_coop(
    wasm: &[u8],
    entry: &str,
    argv: &[i64],
    mapped: i64,
    n_results: usize,
) -> Option<Vec<i64>> {
    let win_len = svm_coop_win_len();
    let win_ptr = svm_coop_win_ptr() as *mut u8;
    let engine = Engine::default();
    let module = WModule::new(&engine, wasm).expect("emitted wasm must validate");
    let mut store: Store<i32> = Store::new(&engine, 0);
    let pages = ((WIN_BASE as usize + win_len) as u32).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
    memory
        .write(&mut store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    // SAFETY: the paused task is parked inside the event; this thread is the only accessor (FFI_LOCK).
    let live = unsafe { std::slice::from_raw_parts(win_ptr, win_len) };
    memory.write(&mut store, WIN_BASE as usize, live).unwrap();
    let mut linker: Linker<i32> = Linker::new(&engine);
    linker.define("env", "memory", memory).unwrap();
    let table = wasmi::Table::new(
        &mut store,
        wasmi::TableType::new(wasmi::core::ValType::FuncRef, 1 << 10, Some(1 << 10)),
        Val::FuncRef(wasmi::FuncRef::null()),
    )
    .unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |mut c: Caller<'_, i32>, code: i32| {
            *c.data_mut() = code;
        })
        .unwrap();
    linker
        .func_wrap::<_, ()>(
            "env",
            "call_interp",
            |_: Caller<'_, i32>, _f: i32, _a: i32| {
                unreachable!("no cross-tier call expected from this unit");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    instance
        .get_global(&store, "mapped")
        .expect("emitted module exports the live-mapped global")
        .set(&mut store, Val::I64(mapped))
        .unwrap();
    let f = instance
        .get_func(&store, entry)
        .unwrap_or_else(|| panic!("{entry} not exported"));
    let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
    params.extend(argv.iter().map(|a| Val::I64(*a)));
    let mut results: Vec<Val> = (0..n_results).map(|_| Val::I64(0)).collect();
    let ran = f.call(&mut store, &params, &mut results);
    let mut buf = vec![0u8; win_len];
    memory.read(&store, WIN_BASE as usize, &mut buf).unwrap();
    // SAFETY: exclusive mirror of the parked window.
    unsafe { std::slice::from_raw_parts_mut(win_ptr, win_len) }.copy_from_slice(&buf);
    match ran {
        Ok(()) => Some(
            results
                .iter()
                .map(|v| match v {
                    Val::I64(x) => *x,
                    Val::I32(x) => *x as i64,
                    _ => panic!("non-integer result"),
                })
                .collect(),
        ),
        Err(_) => None,
    }
}

/// Service one cooperative JIT_INVOKE on wasmi: run the invoked unit's `f0`. All-i64 in this harness.
fn service_coop_jit_on_wasmi(n_results: usize) -> Option<Vec<i64>> {
    // SAFETY: the JIT_INVOKE operand stash is stable until deliver; only accessor (FFI_LOCK).
    let wasm =
        unsafe { std::slice::from_raw_parts(svm_coop_jit_wasm_ptr(), svm_coop_jit_wasm_len()) };
    let argv =
        unsafe { std::slice::from_raw_parts(svm_coop_argv_ptr(), svm_coop_argv_len()) }.to_vec();
    run_emitted_coop(wasm, "f0", &argv, svm_coop_mapped(), n_results)
}

#[test]
fn coop_jit_invoke_pump_matches_the_bytecode_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&coop_jit_guest_text(&unit_blob())).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // The oracle services the invoke interpreted; the pump runs the unit on emitted wasm.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + UNIT_K,
        "oracle: worker 0 + unit(probe) = probe + UNIT_K, through the grown page"
    );

    let opened = svm_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded vm_jit_* guest (no eligible leaf, jit importer) (status {})",
        svm_status()
    );

    let mut jit_invokes = 0u32;
    loop {
        match svm_coop_run() {
            COOP_RUN_JIT_INVOKE => {
                jit_invokes += 1;
                assert!(jit_invokes < 50, "runaway invokes");
                // #717: the event's committed extent is the grown window (a declared-only bound would
                // refuse the unit's probe store and diverge from the oracle).
                assert_eq!(
                    svm_coop_mapped(),
                    65536 + 16384,
                    "the JIT_INVOKE mapped operand carries the grown extent"
                );
                let n = svm_coop_jit_result_types_len();
                match service_coop_jit_on_wasmi(n) {
                    Some(res) => svm_coop_deliver_jit(res.as_ptr(), res.len()),
                    None => svm_coop_deliver_jit_trap(),
                }
            }
            COOP_RUN_TIERUP => panic!("unexpected TIERUP from the leafless vm_jit_* guest"),
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    // Non-vacuity: the unit ran on its emitted wasm on the cooperative driver (which also multiplexed
    // the spawned worker — a topology the single-vCPU vm_jit pump declines).
    assert_eq!(
        jit_invokes, 1,
        "expected exactly one emitted Jit.invoke, got {jit_invokes}"
    );

    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_coop_value(), want.value, "value parity with the oracle");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_coop_close();
}

#[test]
fn coop_tierup_pump_matches_the_bytecode_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&coop_guest_text()).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // The oracle: the plain bytecode path the page falls back to today (multiplexes the threads
    // cooperatively on the interpreter — INVARIANTS.md #9).
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 38, "oracle: f(3) + f(5) = 16 + 22 = 38");
    assert_eq!(
        want.stdout,
        38i64.to_le_bytes().to_vec(),
        "oracle streams the staged sum"
    );

    // The cooperative tier-up pump under test.
    let opened = svm_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must accept the threaded eligible-leaf guest (status {})",
        svm_status()
    );

    let n_results = m.funcs[2].results.len();
    let mut tierups = 0u32;
    loop {
        match svm_coop_run() {
            COOP_RUN_TIERUP => {
                tierups += 1;
                assert!(tierups < 50, "runaway tier-ups");
                assert_eq!(svm_coop_func(), 2, "only the leaf (func 2) tiers up");
                match service_coop_on_wasmi(n_results) {
                    Some(res) => svm_coop_deliver(res.as_ptr(), res.len()),
                    None => svm_coop_deliver_trap(),
                }
            }
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    // Non-vacuity: exactly two tier-ups — the root's `call 2` and the worker's — proving the
    // cooperative driver tiered up across both vCPUs (the single-vCPU pump would have declined the
    // spawn entirely).
    assert_eq!(
        tierups, 2,
        "expected 2 tier-ups (root + worker), got {tierups}"
    );

    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_coop_value(), want.value, "value parity with the oracle");
    assert_eq!(
        svm_run_value(),
        want.value,
        "the page-facing `svm_run_value` slot is staged too"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_coop_close();
}
