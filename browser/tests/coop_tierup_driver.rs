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

use std::collections::HashMap;
use std::sync::Mutex;
use svm_browser::{
    onramp_exec, svm_coop_argv_len, svm_coop_argv_ptr, svm_coop_call_interp, svm_coop_close,
    svm_coop_deliver, svm_coop_deliver_jit, svm_coop_deliver_jit_trap, svm_coop_deliver_trap,
    svm_coop_func, svm_coop_jit_code, svm_coop_jit_param_types_ptr, svm_coop_jit_result_types_len,
    svm_coop_jit_result_types_ptr, svm_coop_jit_wasm_by_handle_len,
    svm_coop_jit_wasm_by_handle_ptr, svm_coop_jit_wasm_len, svm_coop_jit_wasm_ptr, svm_coop_mapped,
    svm_coop_mapped_now, svm_coop_nfuncs, svm_coop_open, svm_coop_run, svm_coop_shim_ptr,
    svm_coop_shim_wasm, svm_coop_slot_code, svm_coop_table_log2, svm_coop_value, svm_coop_wasm_len,
    svm_coop_wasm_ptr, svm_coop_win_len, svm_coop_win_ptr, svm_run_value, svm_status,
    svm_stdout_len, svm_stdout_ptr, COOP_RUN_DONE, COOP_RUN_JIT_INVOKE, COOP_RUN_TIERUP, STATUS_OK,
};
use svm_interp::{Host, StreamRole};
use wasmi::{
    Caller, Engine, Func, FuncRef, Instance, Linker, Memory, MemoryType, Module as WModule, Store,
    Table, TableType, Val,
};

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

// ============================================================================================
// #926 slice 2f: the B2 driver-table half — `call_indirect` tiers up on the cooperative path.
// A persistent driver (the Rust twin of `driveCoopTierupRun`) with **one shared funcref table**
// resynced from the engine's slot mirror (`svm_coop_nfuncs`/`svm_coop_slot_code`/…) at each event, so
// an emitted `call_indirect` dispatches natively (to a program `f{i}` or an installed unit's `f0`) or
// through a bounce shim, exactly as the browser would. The single-shot analogue is `tierup_driver.rs`.
// ============================================================================================

/// Per-store host state: the shared memory, every live instance's `"mapped"`/`"fuel"` globals (the
/// #717 fan-out set), and the bounce log (the edges that went through `env.call_interp`).
#[derive(Default)]
struct DriverData {
    mem: Option<Memory>,
    mapped_globals: Vec<wasmi::Global>,
    fuel_globals: Vec<wasmi::Global>,
    bounces: Vec<u32>,
}

struct CoopB2Driver {
    store: Store<DriverData>,
    engine: Engine,
    memory: Memory,
    table: Table,
    main: Instance,
    unit_insts: HashMap<i32, Instance>,
    /// Bounce shims keyed by `(slot, occupant code)` (`-2` = a program-function slot) so an
    /// uninstall/reinstall regenerates against the new occupant's signature.
    shims: HashMap<(u32, i32), Func>,
    win_len: usize,
}

impl CoopB2Driver {
    /// Build the driver for the freshly opened coop session: memory sized for the mirrored window, the
    /// shared table (sized to the engine's `call_indirect` mask), and the main emitted module.
    fn new() -> CoopB2Driver {
        let engine = Engine::default();
        let mut store: Store<DriverData> = Store::new(&engine, DriverData::default());
        let win_len = svm_coop_win_len();
        let pages = ((WIN_BASE as usize + win_len) as u32).div_ceil(1 << 16) + 1;
        let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
        store.data_mut().mem = Some(memory);
        let tsize = 1u32 << svm_coop_table_log2();
        let table = Table::new(
            &mut store,
            TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
            Val::FuncRef(FuncRef::null()),
        )
        .unwrap();
        let main_wasm =
            unsafe { std::slice::from_raw_parts(svm_coop_wasm_ptr(), svm_coop_wasm_len()) }
                .to_vec();
        let mut d = CoopB2Driver {
            store,
            engine,
            memory,
            table,
            main: unsafe { std::mem::zeroed() },
            unit_insts: HashMap::new(),
            shims: HashMap::new(),
            win_len,
        };
        d.main = d.instantiate(&main_wasm);
        d
    }

    /// Instantiate an emitted module (main, unit, or shim) against the shared memory/table and the
    /// cooperative live-state bounce, registering its `"mapped"`/`"fuel"` globals for the fan-out set.
    fn instantiate(&mut self, wasm: &[u8]) -> Instance {
        let module = WModule::new(&self.engine, wasm).expect("emitted wasm must validate");
        let mut linker: Linker<DriverData> = Linker::new(&self.engine);
        linker.define("env", "memory", self.memory).unwrap();
        linker
            .define("env", "__indirect_function_table", self.table)
            .unwrap();
        linker
            .func_wrap("env", "trap", |_c: Caller<'_, DriverData>, _code: i32| {})
            .unwrap();
        let win_len = self.win_len;
        linker
            .func_wrap(
                "env",
                "call_interp",
                move |mut c: Caller<'_, DriverData>,
                      target: i32,
                      args_ptr: i32|
                      -> Result<(), wasmi::Error> {
                    let mem = c.data().mem.unwrap();
                    let win_ptr = svm_coop_win_ptr() as *mut u8;
                    // Make the emitted frames' window writes visible to the engine before the callback.
                    let mut w = vec![0u8; win_len];
                    mem.read(&c, WIN_BASE as usize, &mut w).unwrap();
                    // SAFETY: the paused task is parked on the pending event; the window is exclusive.
                    unsafe { std::slice::from_raw_parts_mut(win_ptr, win_len) }.copy_from_slice(&w);
                    let mut slots = [0u8; 512];
                    mem.read(&c, args_ptr as usize, &mut slots).unwrap();
                    let rc = svm_coop_call_interp(target as u32, slots.as_mut_ptr());
                    let live = unsafe { std::slice::from_raw_parts(win_ptr, win_len) };
                    mem.write(&mut c, WIN_BASE as usize, live).unwrap();
                    mem.write(&mut c, args_ptr as usize, &slots).unwrap();
                    // #717 fan-out: a bounced callback may have `vm_map`-grown the window.
                    let now = svm_coop_mapped_now();
                    for g in c.data().mapped_globals.clone() {
                        g.set(&mut c, Val::I64(now)).unwrap();
                    }
                    c.data_mut().bounces.push(target as u32);
                    if rc != 0 {
                        return Err(wasmi::Error::from(
                            wasmi::core::TrapCode::UnreachableCodeReached,
                        ));
                    }
                    Ok(())
                },
            )
            .unwrap();
        let instance = linker
            .instantiate(&mut self.store, &module)
            .unwrap()
            .start(&mut self.store)
            .unwrap();
        if let Some(g) = instance.get_global(&self.store, "mapped") {
            self.store.data_mut().mapped_globals.push(g);
        }
        if let Some(g) = instance.get_global(&self.store, "fuel") {
            self.store.data_mut().fuel_globals.push(g);
        }
        instance
    }

    /// Get-or-build the bounce shim for `slot` (occupant `code`, `-2` = program function).
    fn shim(&mut self, slot: u32, code: i32) -> Option<Func> {
        if let Some(f) = self.shims.get(&(slot, code)) {
            return Some(*f);
        }
        let len = svm_coop_shim_wasm(slot);
        if len == 0 {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(svm_coop_shim_ptr(), len) }.to_vec();
        let inst = self.instantiate(&bytes);
        let f = inst.get_func(&self.store, "t").expect("shim exports t");
        self.shims.insert((slot, code), f);
        Some(f)
    }

    /// Rebuild the shared table from the engine's slot mirror — the per-event sync (installs only
    /// happen between events, so a synced table is exact for the whole event).
    fn sync_table(&mut self) {
        let nfuncs = svm_coop_nfuncs();
        let tsize = 1usize << svm_coop_table_log2();
        for slot in 0..tsize {
            let entry: Option<Func> = if slot < nfuncs {
                match self.main.get_func(&self.store, &format!("f{slot}")) {
                    Some(f) => Some(f),
                    None => self.shim(slot as u32, -2),
                }
            } else {
                let code = svm_coop_slot_code(slot as u32);
                if code < 0 {
                    None
                } else if svm_coop_jit_wasm_by_handle_len(code) > 0 {
                    let inst = match self.unit_insts.get(&code) {
                        Some(i) => *i,
                        None => {
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    svm_coop_jit_wasm_by_handle_ptr(),
                                    svm_coop_jit_wasm_by_handle_len(code),
                                )
                            }
                            .to_vec();
                            let i = self.instantiate(&bytes);
                            self.unit_insts.insert(code, i);
                            i
                        }
                    };
                    inst.get_func(&self.store, "f0")
                } else {
                    self.shim(slot as u32, code)
                }
            };
            let fr = match entry {
                Some(f) => FuncRef::new(f),
                None => FuncRef::null(),
            };
            self.table
                .set(&mut self.store, slot as u64, Val::FuncRef(fr))
                .unwrap();
        }
    }

    /// Sync window + globals into the shared instances before running an emitted entry.
    fn prime(&mut self, mapped: i64) {
        let win_ptr = svm_coop_win_ptr() as *mut u8;
        // SAFETY: the paused task is parked on the pending event; the window is exclusive.
        let live = unsafe { std::slice::from_raw_parts(win_ptr, self.win_len) };
        self.memory
            .write(&mut self.store, WIN_BASE as usize, live)
            .unwrap();
        self.memory
            .write(&mut self.store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
            .unwrap();
        for g in self.store.data().mapped_globals.clone() {
            g.set(&mut self.store, Val::I64(mapped)).unwrap();
        }
        for g in self.store.data().fuel_globals.clone() {
            g.set(&mut self.store, Val::I64(1 << 61)).unwrap();
        }
    }

    /// Mirror the emitted writes back into the live window before the vCPU resumes.
    fn writeback(&mut self) {
        let win_ptr = svm_coop_win_ptr() as *mut u8;
        let mut buf = vec![0u8; self.win_len];
        self.memory
            .read(&self.store, WIN_BASE as usize, &mut buf)
            .unwrap();
        // SAFETY: see above.
        unsafe { std::slice::from_raw_parts_mut(win_ptr, self.win_len) }.copy_from_slice(&buf);
    }

    /// Service the pending TIERUP through the shared table (#880): sync window/table/globals, run the
    /// main module's `f{func}` (whose `call_indirect` now dispatches through the table), deliver.
    fn service_tierup(&mut self, n_results: usize) {
        self.sync_table();
        self.prime(svm_coop_mapped());
        let func = svm_coop_func();
        let f = self
            .main
            .get_func(&self.store, &format!("f{func}"))
            .unwrap_or_else(|| panic!("f{func} not exported"));
        let n = svm_coop_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(svm_coop_argv_ptr(), n) };
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        params.extend(argv.iter().map(|a| Val::I64(*a)));
        let mut results: Vec<Val> = (0..n_results).map(|_| Val::I64(0)).collect();
        let ran = f.call(&mut self.store, &params, &mut results);
        self.writeback();
        match ran {
            Ok(()) => {
                let slots: Vec<i64> = results
                    .iter()
                    .map(|v| match v {
                        Val::I64(x) => *x,
                        Val::I32(x) => *x as i64,
                        _ => panic!("non-integer TIERUP result"),
                    })
                    .collect();
                svm_coop_deliver(slots.as_ptr(), slots.len());
            }
            Err(_) => svm_coop_deliver_trap(),
        }
    }

    /// Service the pending JIT_INVOKE: sync window/table/globals, run the invoked unit's `f0` (whose
    /// `call_indirect` dispatches through the shared table), deliver results or the trap.
    fn service_jit_invoke(&mut self) {
        self.sync_table();
        self.prime(svm_coop_mapped());
        let code = svm_coop_jit_code();
        let inst = match self.unit_insts.get(&code) {
            Some(i) => *i,
            None => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(svm_coop_jit_wasm_ptr(), svm_coop_jit_wasm_len())
                }
                .to_vec();
                let i = self.instantiate(&bytes);
                self.unit_insts.insert(code, i);
                i
            }
        };
        let f0 = inst.get_func(&self.store, "f0").expect("unit exports f0");
        let n = svm_coop_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(svm_coop_argv_ptr(), n) };
        let ptypes = unsafe { std::slice::from_raw_parts(svm_coop_jit_param_types_ptr(), n) };
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        for (a, tc) in argv.iter().zip(ptypes) {
            params.push(match tc {
                0 => Val::I32(*a as i32),
                1 => Val::I64(*a),
                2 => Val::F32(f32::from_bits(*a as u32).into()),
                _ => Val::F64(f64::from_bits(*a as u64).into()),
            });
        }
        let rn = svm_coop_jit_result_types_len();
        let rtypes = unsafe { std::slice::from_raw_parts(svm_coop_jit_result_types_ptr(), rn) };
        let mut results: Vec<Val> = rtypes
            .iter()
            .map(|tc| match tc {
                0 => Val::I32(0),
                1 => Val::I64(0),
                2 => Val::F32(0f32.into()),
                _ => Val::F64(0f64.into()),
            })
            .collect();
        let ran = f0.call(&mut self.store, &params, &mut results);
        self.writeback();
        match ran {
            Ok(()) => {
                let slots: Vec<i64> = results
                    .iter()
                    .zip(rtypes)
                    .map(|(v, _)| match v {
                        Val::I32(x) => *x as i64,
                        Val::I64(x) => *x,
                        Val::F32(x) => f32::from(*x).to_bits() as i64,
                        Val::F64(x) => f64::from(*x).to_bits() as i64,
                        _ => panic!("non-scalar result"),
                    })
                    .collect();
                svm_coop_deliver_jit(slots.as_ptr(), slots.len());
            }
            Err(_) => svm_coop_deliver_jit_trap(),
        }
    }

    fn bounces(&self) -> &[u32] {
        &self.store.data().bounces
    }
}

/// Drive an opened coop session to DONE with the full B2 driver. Returns the driver (for its bounce
/// log) and the `(tierups, invokes)` counters.
fn drive_coop_b2_session(m: &svm_ir::Module) -> (CoopB2Driver, u32, u32) {
    let mut d = CoopB2Driver::new();
    let (mut tierups, mut invokes) = (0u32, 0u32);
    loop {
        match svm_coop_run() {
            COOP_RUN_JIT_INVOKE => {
                invokes += 1;
                assert!(invokes < 50, "runaway invokes");
                d.service_jit_invoke();
            }
            COOP_RUN_TIERUP => {
                tierups += 1;
                assert!(tierups < 50, "runaway tier-ups");
                let f = svm_coop_func() as usize;
                d.service_tierup(m.funcs[f].results.len());
            }
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    (d, tierups, invokes)
}

/// The added constant of the `call_indirect`-reached leaf (`f2`), distinct from the unit's `UNIT_K`.
const LEAF_K: i64 = 424242;

/// A **threaded** guest whose tiered-up leaf `call_indirect`s another **emitted** leaf (#880, the
/// native edge, on the cooperative path). `_start` (the root vCPU) spawns a worker (`f3`, returns 0),
/// `vm_map`-grows `[64 KiB, 80 KiB)`, calls `f1` (which tiers up and `call_indirect`s `f2` at table
/// slot 2 — dispatched natively through the synced shared table), joins the worker, sums, streams, and
/// returns. `f2` stores into the grown page, so the tier-up's `"mapped"` bound must admit the growth.
/// The `thread.spawn` routes it to the cooperative driver; the `call_indirect` is the B2 edge.
fn coop_indirect_guest_text() -> String {
    let (out_h, mem_h) = onramp_out_mem_handles();
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 3 vz vz
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
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
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  vr = call_indirect (i64) -> (i64) vs2 (v0)
  vone = i64.const 1
  vsum = i64.add vr vone
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {LEAF_K}
  vsum = i64.add v0 vk
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr vsum
  vld = i64.load vaddr
  return vld
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

#[test]
fn coop_indirect_leaf_tiers_up_natively() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&coop_indirect_guest_text()).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // Oracle: the plain bytecode path. `f1(probe) = f2(probe) + 1 = (probe + LEAF_K) + 1`; worker 0.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + LEAF_K + 1,
        "oracle: call_indirect(f2)(probe) + 1, through the grown page"
    );

    let opened = svm_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded call_indirect guest (status {})",
        svm_status()
    );

    let (d, tierups, invokes) = drive_coop_b2_session(&m);
    // Non-vacuity: the `call_indirect`-bearing leaf `f1` tiered up (#880), and the indirect edge to
    // the emitted `f2` was **native** — never a bounce (an empty bounce log proves the shared-table
    // dispatch, not the interpreter, serviced it).
    assert!(tierups >= 1, "the call_indirect leaf must tier up (#880)");
    assert_eq!(invokes, 0, "no vm_jit units in this guest");
    assert!(
        d.bounces().is_empty(),
        "the indirect edge targets an emitted function — native, never bounced: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_coop_value(), want.value, "value parity with the oracle");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_coop_close();
}

/// The on-ramp powerbox's stdout + **jit** handles (grant order incl. the conditional `Jit` grant a
/// `vm_jit_*` importer gets — `grant_onramp_caps`), for the install-slot guest below.
fn onramp_out_jit_handles() -> (i32, i32) {
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    let _ = h.grant_memory();
    let _ = h.grant_address_space(0, 1 << 16);
    let jit = h.grant_jit_with_table(Some(16), 10);
    (out, jit)
}

/// A **threaded** `vm_jit_*` guest whose tiered-up leaf `call_indirect`s an **installed unit's**
/// emitted `f0` (#880 old→new native, on the cooperative path). `_start` (the root vCPU) spawns a
/// worker (`f2`, returns 0), `vm_jit_compile`s `f(x)=x+7`, installs it (getting a runtime slot past the
/// program's `f{i}` prefix), calls `f1(slot, X)` — which tiers up and `call_indirect`s the install slot
/// — joins the worker, sums, streams, returns. The install-slot edge exercises the coop `slot_code`
/// mirror + the by-handle unit wasm the driver syncs into the shared table.
fn coop_installed_unit_guest_text() -> String {
    let (out_h, jit_h) = onramp_out_jit_handles();
    let unit_src = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 7
  vsum = i64.add v0 vk
  return vsum
  }
}
"#;
    let blob = {
        let u = svm_text::parse_module(unit_src).expect("parse unit");
        svm_verify::verify_module(&u).expect("verify unit");
        svm_encode::encode_module(&u)
    };
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
    const X: i64 = 4000;
    format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
  vz = i64.const 0
  vt = thread.spawn 2 vz vz
{stores}  vbp = i64.const {BLOB_BASE}
  vbl = i64.const {blob_len}
  vcode = call.import 0 (vbp, vbl)
  vjit = i32.const {jit_h}
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vx = i64.const {X}
  vres = call 1 (vslot, vx)
  vj = thread.join vt
  vfin = i64.add vres vj
  vsl = i64.const {SLOT}
  i64.store vsl vfin
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vfin
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vslot: i64, vx: i64) {{
  vs = i32.wrap_i64 vslot
  vr = call_indirect (i64) -> (i64) vs (vx)
  vk = i64.const 100
  vsum = i64.add vr vk
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

#[test]
fn coop_leaf_reaches_installed_unit_natively() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&coop_installed_unit_guest_text()).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // Oracle: `f1(slot, X) = unit(X) + 100 = (X + 7) + 100`; worker 0. (X = 4000.)
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        4000 + 7 + 100,
        "oracle: leaf → installed unit → +100"
    );

    let opened = svm_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded install-slot guest (status {})",
        svm_status()
    );

    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    // Non-vacuity: the dispatching leaf `f1` tiered up, and the install-slot edge reached the unit's
    // **emitted** `f0` natively (old→new) — never a bounce (the empty log proves the shared-table slot
    // the coop `slot_code` mirror populated dispatched, not an interpreter fallback).
    assert!(
        tierups >= 1,
        "the dispatching leaf must tier up (#880 old→new)"
    );
    assert!(
        d.bounces().is_empty(),
        "the install-slot edge reaches the unit's emitted f0 — native: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_coop_value(), want.value, "value parity with the oracle");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_coop_close();
}

// ---- #926 slice 2g: the invoke-confined bounce — an emitted `Jit.invoke` unit bounces cross-tier ----

/// What the bounce helper adds, and where the unit probes the page a bounced callback grew mid-invoke.
const BOUNCE_K: i64 = 1000;
const MID_GROW_PROBE: i64 = 81920 + 8;

/// A **threaded** `vm_jit_*` guest whose surfaced `Jit.invoke` unit **bounces** cross-tier (#926 slice
/// 2g — the invoke-confined registry path; the #846 `linked_unit_bounces` shape, on the cooperative
/// driver). `_start` (root vCPU) spawns a worker (`f3`, returns 0), `vm_map`-grows `[64 KiB, 80 KiB)`,
/// compiles + `invoke2`s a unit whose emitted `f0`: (1) `call_indirect`s slot 1 → the program's
/// cap-calling helper `f1` — interpreter-resident, so the edge **bounces** via `env.call_interp` (the
/// callback grows `[80 KiB, 96 KiB)` and streams); (2) stores into that just-grown page (correct only
/// if the post-bounce `"mapped"` fan-out admits the growth); (3) `call_indirect`s slot 2 → the pure leaf
/// `f2` — emitted, **native**. The bounce runs through the invoke-confined fiber registry (`Vcpu`
/// parity), and must match the interpreted oracle bit-for-bit.
fn coop_invoke_bounce_guest_text() -> String {
    let (out_h, mem_h) = onramp_out_mem_handles();
    let unit_src = format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs1 = i32.const 1
  va = call_indirect (i64) -> (i64) vs1 (v0)
  vsum = i64.add v0 va
  vaddr = i64.const {MID_GROW_PROBE}
  i64.store vaddr vsum
  vld = i64.load vaddr
  vs2 = i32.const 2
  vc = call_indirect (i64) -> (i64) vs2 (vld)
  return vc
  }}
}}
"#
    );
    let unit = svm_text::parse_module(&unit_src).expect("unit parse");
    svm_verify::verify_module(&unit).expect("unit verify");
    let blob = svm_encode::encode_module(&unit);
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
  vt = thread.spawn 3 vz vz
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
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vas = i32.const {mem_h}
  voff = i64.const 81920
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
  vk = i64.const {BOUNCE_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const 3
  vmul = i64.mul v0 vk
  return vmul
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

#[test]
fn coop_invoked_unit_bounces_and_native_edges_match_the_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&coop_invoke_bounce_guest_text()).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // Oracle: bounce (+K, grow), grown-page store/load, native ×3; worker 0.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        (2 * PROBE + BOUNCE_K) * 3,
        "oracle: bounce (+K, grow), grown-page store/load, native ×3"
    );

    let opened = svm_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded bouncing-invoke guest (status {})",
        svm_status()
    );

    let (d, _tierups, invokes) = drive_coop_b2_session(&m);
    // Non-vacuity: the unit ran emitted (invoke), its cap-calling edge **bounced** through the
    // invoke-confined registry (slot 1 in the bounce log), and the pure leaf dispatched **natively**
    // (slot 2 absent) — the exact edge split #846 pins, now on the cooperative driver.
    assert!(
        invokes >= 1,
        "the linked unit must run emitted (invoke non-vacuity)"
    );
    assert!(
        d.bounces().contains(&1),
        "the cap-calling helper must have bounced during the invoke: {:?}",
        d.bounces()
    );
    assert!(
        !d.bounces().contains(&2),
        "the eligible leaf must dispatch natively, never bounce: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_coop_value(),
        want.value,
        "value parity (growth-mid-invoke visible post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity (bounce ordering included)"
    );
    svm_coop_close();
}
