//! **Cooperative on-ramp tier-up** (#926 slice 2) — the native differential for the `temen_coop_*`
//! event-pump FFI. A genuinely threaded on-ramp guest (its `_start` `thread.spawn`s a worker, so the
//! single-vCPU `temen_onramp_tierup_*` driver would *decline* it at the spawn event) must run observably
//! identical to the plain bytecode path (`onramp_exec`, INVARIANTS.md #9) when its hot leaf is serviced
//! on the emitted wasm — with wasmi playing the browser's JS host (`driveCoopTierupRun`). The
//! cooperative driver multiplexes the root and the worker on one wasm thread and tiers up **both**
//! their calls to the eligible leaf; the non-vacuity count pins that two tier-ups (one per vCPU) fire.
//!
//! This proves the cooperative-driver FFI wiring end-to-end: open → pump → service → deliver → capture,
//! over a thread topology the single-vCPU pump cannot run at all. The engine seam itself (multi-task
//! resumable tier-up + `deliver` routing) is pinned natively in `temen-interp/tests/coop_tierup.rs`.

use std::collections::HashMap;
use std::sync::Mutex;
use temen_browser::{
    onramp_exec, temen_coop_argv_len, temen_coop_argv_ptr, temen_coop_call_interp,
    temen_coop_close, temen_coop_deliver, temen_coop_deliver_jit, temen_coop_deliver_jit_trap,
    temen_coop_deliver_trap, temen_coop_func, temen_coop_jit_code, temen_coop_jit_param_types_ptr,
    temen_coop_jit_result_types_len, temen_coop_jit_result_types_ptr,
    temen_coop_jit_wasm_by_handle_len, temen_coop_jit_wasm_by_handle_ptr, temen_coop_jit_wasm_len,
    temen_coop_jit_wasm_ptr, temen_coop_mapped, temen_coop_mapped_now, temen_coop_nfuncs,
    temen_coop_open, temen_coop_paged, temen_coop_pagestate_len, temen_coop_pagestate_ptr,
    temen_coop_run, temen_coop_set_tierup_floor, temen_coop_shim_ptr, temen_coop_shim_wasm,
    temen_coop_slot_code, temen_coop_table_gen, temen_coop_table_log2, temen_coop_tierup_win_len,
    temen_coop_tierup_win_ptr, temen_coop_value, temen_coop_wasm_len, temen_coop_wasm_ptr,
    temen_coop_win_len, temen_coop_win_ptr,
    temen_run_value, temen_status, temen_stdout_len, temen_stdout_ptr, COOP_RUN_DONE,
    COOP_RUN_JIT_INVOKE, COOP_RUN_TIERUP, COOP_RUN_TRAP, STATUS_OK, STATUS_TRAP,
    STATUS_UNSUPPORTED,
};
use temen_interp::{Host, StreamRole};
use wasmi::{
    Caller, Engine, Func, FuncRef, Instance, Linker, Memory, MemoryType, Module as WModule, Store,
    Table, TableType, Val,
};

/// The coop session statics are process-global (single-threaded wasm by design) — serialize the tests
/// in this binary across them.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// Acquire [`FFI_LOCK`] and disable the #1026 leaf tier-up **size floor** for this test. These
/// differentials exercise the emitted-tier **mechanism** with deliberately tiny synthetic leaves; the
/// size **policy** (a leaf below the floor stays on the interpreter) is pinned separately in
/// [`leaf_tierup_size_floor_gates_tiny_and_admits_heavy`]. Floor 0 ⇒ any emittable leaf tiers up, the
/// behavior these tests were written against. The guard resets to production default on drop so a
/// test that does *not* call this (the policy test) sees the real default even after one that did.
fn ffi_guard() -> FloorGuard {
    let g = FFI_LOCK.lock().unwrap();
    temen_coop_set_tierup_floor(0);
    FloorGuard(g)
}
struct FloorGuard(std::sync::MutexGuard<'static, ()>);
impl Drop for FloorGuard {
    fn drop(&mut self) {
        temen_coop_set_tierup_floor(temen_wasm_jit::MIN_TIERUP_EMITTED_FN_BYTES);
    }
}

/// Where the wasmi harness places the mirrored window / env cell in the emitted module's memory.
const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;
/// Declared-prefix cell `_start` stages the summed result in before streaming it to stdout.
const SLOT: i64 = 2048;

/// The on-ramp powerbox's stdout handle, replicated from `grant_onramp_caps`'s grant order (stdout,
/// stdin, exit, memory, addrspace, …) against a fresh `Host` — deterministic per session, so the
/// import-free guest text can `call.cap` it directly.
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
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
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
    let wasm = unsafe { std::slice::from_raw_parts(temen_coop_wasm_ptr(), temen_coop_wasm_len()) };
    let func = temen_coop_func();
    let argv = unsafe { std::slice::from_raw_parts(temen_coop_argv_ptr(), temen_coop_argv_len()) };
    // #816 env-routed tier-up: the PENDING task's window (base + span) — the browser JS driver's
    // per-event `win`. A §14 confined child's event mirrors just its carve.
    let win_len = temen_coop_tierup_win_len();
    let win_ptr = temen_coop_tierup_win_ptr() as *mut u8;
    let mapped = temen_coop_mapped();

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
    // A closed leaf never `call.dyn`s; an all-null table satisfies the B2 import (unused here).
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
    let m = temen_text::parse_module(&src).expect("unit parse");
    temen_verify::verify_module(&m).expect("unit verify");
    temen_encode::encode_module(&m)
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
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
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
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
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
    // #816: the pending task's window, per event (see `service_coop_on_wasmi`).
    let win_len = temen_coop_tierup_win_len();
    let win_ptr = temen_coop_tierup_win_ptr() as *mut u8;
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
        unsafe { std::slice::from_raw_parts(temen_coop_jit_wasm_ptr(), temen_coop_jit_wasm_len()) };
    let argv = unsafe { std::slice::from_raw_parts(temen_coop_argv_ptr(), temen_coop_argv_len()) }
        .to_vec();
    run_emitted_coop(wasm, "f0", &argv, temen_coop_mapped(), n_results)
}

#[test]
fn coop_jit_invoke_pump_matches_the_bytecode_oracle() {
    let _g = ffi_guard();
    let m = temen_text::parse_module(&coop_jit_guest_text(&unit_blob())).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // The oracle services the invoke interpreted; the pump runs the unit on emitted wasm.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + UNIT_K,
        "oracle: worker 0 + unit(probe) = probe + UNIT_K, through the grown page"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded vm_jit_* guest (no eligible leaf, jit importer) (status {})",
        temen_status()
    );

    let mut jit_invokes = 0u32;
    loop {
        match temen_coop_run() {
            COOP_RUN_JIT_INVOKE => {
                jit_invokes += 1;
                assert!(jit_invokes < 50, "runaway invokes");
                // #717: the event's committed extent is the grown window (a declared-only bound would
                // refuse the unit's probe store and diverge from the oracle).
                assert_eq!(
                    temen_coop_mapped(),
                    65536 + 16384,
                    "the JIT_INVOKE mapped operand carries the grown extent"
                );
                let n = temen_coop_jit_result_types_len();
                match service_coop_jit_on_wasmi(n) {
                    Some(res) => temen_coop_deliver_jit(res.as_ptr(), res.len()),
                    None => temen_coop_deliver_jit_trap(),
                }
            }
            COOP_RUN_TIERUP => panic!("unexpected TIERUP from the leafless vm_jit_* guest"),
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", temen_status()),
        }
    }
    // Non-vacuity: the unit ran on its emitted wasm on the cooperative driver (which also multiplexed
    // the spawned worker — a topology the single-vCPU vm_jit pump declines).
    assert_eq!(
        jit_invokes, 1,
        "expected exactly one emitted Jit.invoke, got {jit_invokes}"
    );

    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    temen_coop_close();
}

#[test]
fn coop_tierup_pump_matches_the_bytecode_oracle() {
    let _g = ffi_guard();
    let m = temen_text::parse_module(&coop_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

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
    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must accept the threaded eligible-leaf guest (status {})",
        temen_status()
    );

    let n_results = m.funcs[2].results.len();
    let mut tierups = 0u32;
    loop {
        match temen_coop_run() {
            COOP_RUN_TIERUP => {
                tierups += 1;
                assert!(tierups < 50, "runaway tier-ups");
                assert_eq!(temen_coop_func(), 2, "only the leaf (func 2) tiers up");
                match service_coop_on_wasmi(n_results) {
                    Some(res) => temen_coop_deliver(res.as_ptr(), res.len()),
                    None => temen_coop_deliver_trap(),
                }
            }
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", temen_status()),
        }
    }
    // Non-vacuity: exactly two tier-ups — the root's `call 2` and the worker's — proving the
    // cooperative driver tiered up across both vCPUs (the single-vCPU pump would have declined the
    // spawn entirely).
    assert_eq!(
        tierups, 2,
        "expected 2 tier-ups (root + worker), got {tierups}"
    );

    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    assert_eq!(
        temen_run_value(),
        want.value,
        "the page-facing `temen_run_value` slot is staged too"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    temen_coop_close();
}

/// #1026 slice 4 — the leaf tier-up **size floor** gates strictly by [`temen_wasm_jit::est_emitted_size`]:
/// the same guest tiers up when the floor is at-or-below its leaf's estimated body and stays on the
/// interpreter when the floor is one byte above — with **identical** guest-observable results either
/// way (INVARIANTS.md #9: the gate changes which tier runs, never the semantics). Pins that a
/// dispatch-heavy card's tiny leaves (far below the production default) will not tier up, while a
/// genuinely heavy leaf still does — the "decision in one place" #1026 asks for.
#[test]
fn leaf_tierup_size_floor_gates_tiny_and_admits_heavy() {
    let _g = ffi_guard(); // resets the floor to the production default on drop
    let m = temen_text::parse_module(&coop_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);
    // The floor is compared against the *outlined* module the open builds its eligibility from
    // (`outline_cap_calls` runs first), so compute the leaf's estimate on the same shape.
    let leaf_sz = {
        let mut outlined = m.clone();
        temen_wasm_jit::outline_cap_calls(&mut outlined);
        temen_wasm_jit::est_emitted_size(&outlined.funcs[2])
    };
    assert!(
        leaf_sz < temen_wasm_jit::MIN_TIERUP_EMITTED_FN_BYTES,
        "the synthetic leaf ({leaf_sz}B) must sit below the production default \
         ({}B) — so the default gates it, as it gates every dispatch-heavy card's leaves",
        temen_wasm_jit::MIN_TIERUP_EMITTED_FN_BYTES
    );

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 38, "oracle: f(3) + f(5) = 16 + 22 = 38");

    // Drive the guest to completion at a given floor, returning the tier-up count. Parity with the
    // oracle is asserted on every arm — the gate must never change the result.
    let drive_at_floor = |floor: usize| -> u32 {
        temen_coop_set_tierup_floor(floor);
        let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
        assert_eq!(opened, 0, "open (floor {floor}) status {}", temen_status());
        let n_results = m.funcs[2].results.len();
        let mut tierups = 0u32;
        loop {
            match temen_coop_run() {
                COOP_RUN_TIERUP => {
                    tierups += 1;
                    assert!(tierups < 50, "runaway tier-ups");
                    assert_eq!(temen_coop_func(), 2, "only the leaf (func 2) can tier up");
                    match service_coop_on_wasmi(n_results) {
                        Some(res) => temen_coop_deliver(res.as_ptr(), res.len()),
                        None => temen_coop_deliver_trap(),
                    }
                }
                COOP_RUN_DONE => break,
                ev => panic!(
                    "unexpected event {ev} (floor {floor}, status {})",
                    temen_status()
                ),
            }
        }
        assert_eq!(temen_status(), want.status, "status parity (floor {floor})");
        assert_eq!(
            temen_coop_value(),
            want.value,
            "value parity (floor {floor})"
        );
        // SAFETY: DONE-arm capture slots; sole accessor under FFI_LOCK.
        let got_out =
            unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
        assert_eq!(got_out, want.stdout, "stdout parity (floor {floor})");
        temen_coop_close();
        tierups
    };

    // A floor comfortably below the leaf's estimate ⇒ eligible (the predicate is `>=`) ⇒ both the
    // root's and the worker's `call 2` tier up. (Floors are held off the exact estimate to stay
    // robust to a byte or two of encode/outline drift between here and the open.)
    assert_eq!(
        drive_at_floor(leaf_sz / 2),
        2,
        "a floor below the leaf est size admits it (root + worker tier up)"
    );
    // A floor a few times the leaf ⇒ gated ⇒ the interpreter runs the leaf inline, zero crossings,
    // same result — the size comparison, not an on/off switch.
    assert_eq!(
        drive_at_floor(leaf_sz * 4),
        0,
        "a floor above the leaf est size suppresses its tier-up"
    );
    // And the production default (well above any dispatch leaf) gates it too.
    assert_eq!(
        drive_at_floor(temen_wasm_jit::MIN_TIERUP_EMITTED_FN_BYTES),
        0,
        "the production default gates this dispatch-leaf-sized function"
    );
}

// ============================================================================================
// #926 slice 2f: the B2 driver-table half — `call.dyn` tiers up on the cooperative path.
// A persistent driver (the Rust twin of `driveCoopTierupRun`) with **one shared funcref table**
// resynced from the engine's slot mirror (`temen_coop_nfuncs`/`temen_coop_slot_code`/…) at each event, so
// an emitted `call.dyn` dispatches natively (to a program `f{i}` or an installed unit's `f0`) or
// through a bounce shim, exactly as the browser would. The single-shot analogue is `tierup_driver.rs`.
// ============================================================================================

/// Per-store host state: the shared memory, every live instance's `"mapped"`/`"fuel"` globals (the
/// #717 fan-out set), and the bounce log (the edges that went through `env.call_interp`).
#[derive(Default)]
struct DriverData {
    mem: Option<Memory>,
    mapped_globals: Vec<wasmi::Global>,
    fuel_globals: Vec<wasmi::Global>,
    /// #1009 paged: the emitted `"pagestate"` globals (the coop twin of the pump driver's).
    pagestate_globals: Vec<wasmi::Global>,
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
    /// #1009: the dispatch-table generation the shared table was last synced at (mirrors the JS
    /// driver's cache so an install re-syncs and a no-install run syncs once).
    synced_gen: i64,
}

impl CoopB2Driver {
    /// Build the driver for the freshly opened coop session: memory sized for the mirrored window, the
    /// shared table (sized to the engine's `call.dyn` mask), and the main emitted module.
    fn new() -> CoopB2Driver {
        let engine = Engine::default();
        let mut store: Store<DriverData> = Store::new(&engine, DriverData::default());
        let win_len = temen_coop_win_len();
        let pages = ((WIN_BASE as usize + win_len) as u32).div_ceil(1 << 16) + 1;
        let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
        store.data_mut().mem = Some(memory);
        let tsize = 1u32 << temen_coop_table_log2();
        let table = Table::new(
            &mut store,
            TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
            Val::FuncRef(FuncRef::null()),
        )
        .unwrap();
        let main_wasm =
            unsafe { std::slice::from_raw_parts(temen_coop_wasm_ptr(), temen_coop_wasm_len()) }
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
            synced_gen: -1,
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
                    let win_ptr = temen_coop_win_ptr() as *mut u8;
                    // Make the emitted frames' window writes visible to the engine before the callback.
                    let mut w = vec![0u8; win_len];
                    mem.read(&c, WIN_BASE as usize, &mut w).unwrap();
                    // SAFETY: the paused task is parked on the pending event; the window is exclusive.
                    unsafe { std::slice::from_raw_parts_mut(win_ptr, win_len) }.copy_from_slice(&w);
                    let mut slots = [0u8; 512];
                    mem.read(&c, args_ptr as usize, &mut slots).unwrap();
                    let rc = temen_coop_call_interp(target as u32, slots.as_mut_ptr());
                    let live = unsafe { std::slice::from_raw_parts(win_ptr, win_len) };
                    mem.write(&mut c, WIN_BASE as usize, live).unwrap();
                    mem.write(&mut c, args_ptr as usize, &slots).unwrap();
                    // #717 fan-out: a bounced callback may have `vm_map`-grown the window. #1009
                    // paged: the grow refreshed the page-state table (`call_interp` rebuilt it) —
                    // fan the fresh coverage to `"mapped"`, re-copy the table, re-point `"pagestate"`.
                    if temen_coop_paged() != 0 {
                        let cover = temen_coop_mapped();
                        for g in c.data().mapped_globals.clone() {
                            g.set(&mut c, Val::I64(cover)).unwrap();
                        }
                        let plen = temen_coop_pagestate_len();
                        // SAFETY: pending-event page-state table, stable until the deliver.
                        let table =
                            unsafe { std::slice::from_raw_parts(temen_coop_pagestate_ptr(), plen) }
                                .to_vec();
                        let table_base = WIN_BASE as usize + win_len;
                        let need = (table_base + plen).div_ceil(1 << 16) as u32;
                        let have = mem.size(&c) as u32;
                        if need > have {
                            mem.grow(&mut c, (need - have) as u64).unwrap();
                        }
                        mem.write(&mut c, table_base, &table).unwrap();
                        for g in c.data().pagestate_globals.clone() {
                            g.set(&mut c, Val::I32(table_base as i32)).unwrap();
                        }
                    } else {
                        let now = temen_coop_mapped_now();
                        for g in c.data().mapped_globals.clone() {
                            g.set(&mut c, Val::I64(now)).unwrap();
                        }
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
        if let Some(g) = instance.get_global(&self.store, "pagestate") {
            self.store.data_mut().pagestate_globals.push(g);
        }
        instance
    }

    /// Get-or-build the bounce shim for `slot` (occupant `code`, `-2` = program function).
    fn shim(&mut self, slot: u32, code: i32) -> Option<Func> {
        if let Some(f) = self.shims.get(&(slot, code)) {
            return Some(*f);
        }
        let len = temen_coop_shim_wasm(slot);
        if len == 0 {
            return None;
        }
        let bytes = unsafe { std::slice::from_raw_parts(temen_coop_shim_ptr(), len) }.to_vec();
        let inst = self.instantiate(&bytes);
        let f = inst.get_func(&self.store, "t").expect("shim exports t");
        self.shims.insert((slot, code), f);
        Some(f)
    }

    /// Rebuild the shared table from the engine's slot mirror — the per-event sync (installs only
    /// happen between events, so a synced table is exact for the whole event).
    fn sync_table(&mut self) {
        let gen = temen_coop_table_gen() as i64;
        if gen == self.synced_gen {
            return;
        }
        let nfuncs = temen_coop_nfuncs();
        let tsize = 1usize << temen_coop_table_log2();
        for slot in 0..tsize {
            let entry: Option<Func> = if slot < nfuncs {
                match self.main.get_func(&self.store, &format!("f{slot}")) {
                    Some(f) => Some(f),
                    None => self.shim(slot as u32, -2),
                }
            } else {
                let code = temen_coop_slot_code(slot as u32);
                if code < 0 {
                    None
                } else if temen_coop_jit_wasm_by_handle_len(code) > 0 {
                    let inst = match self.unit_insts.get(&code) {
                        Some(i) => *i,
                        None => {
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    temen_coop_jit_wasm_by_handle_ptr(),
                                    temen_coop_jit_wasm_by_handle_len(code),
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
        self.synced_gen = gen;
    }

    /// Sync window + globals into the shared instances before running an emitted entry.
    fn prime(&mut self, mapped: i64) {
        let win_ptr = temen_coop_win_ptr() as *mut u8;
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
        // #1009 paged: copy the page-state table in after the window and point `"pagestate"` at it
        // (the browser shares memory, zero-copy; here the emitted module has its own).
        if temen_coop_paged() != 0 {
            let plen = temen_coop_pagestate_len();
            // SAFETY: pending-event page-state table, stable until the deliver.
            let table = unsafe { std::slice::from_raw_parts(temen_coop_pagestate_ptr(), plen) };
            let table_base = WIN_BASE as usize + self.win_len;
            let need = (table_base + plen).div_ceil(1 << 16) as u32;
            let have = self.memory.size(&self.store) as u32;
            if need > have {
                self.memory
                    .grow(&mut self.store, (need - have) as u64)
                    .unwrap();
            }
            self.memory
                .write(&mut self.store, table_base, table)
                .unwrap();
            for g in self.store.data().pagestate_globals.clone() {
                g.set(&mut self.store, Val::I32(table_base as i32)).unwrap();
            }
        }
    }

    /// Mirror the emitted writes back into the live window before the vCPU resumes.
    fn writeback(&mut self) {
        let win_ptr = temen_coop_win_ptr() as *mut u8;
        let mut buf = vec![0u8; self.win_len];
        self.memory
            .read(&self.store, WIN_BASE as usize, &mut buf)
            .unwrap();
        // SAFETY: see above.
        unsafe { std::slice::from_raw_parts_mut(win_ptr, self.win_len) }.copy_from_slice(&buf);
    }

    /// Service the pending TIERUP through the shared table (#880): sync window/table/globals, run the
    /// main module's `f{func}` (whose `call.dyn` now dispatches through the table), deliver.
    fn service_tierup(&mut self, n_results: usize) {
        self.sync_table();
        self.prime(temen_coop_mapped());
        let func = temen_coop_func();
        let f = self
            .main
            .get_func(&self.store, &format!("f{func}"))
            .unwrap_or_else(|| panic!("f{func} not exported"));
        let n = temen_coop_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(temen_coop_argv_ptr(), n) };
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
                temen_coop_deliver(slots.as_ptr(), slots.len());
            }
            Err(_) => temen_coop_deliver_trap(),
        }
    }

    /// Service the pending JIT_INVOKE: sync window/table/globals, run the invoked unit's `f0` (whose
    /// `call.dyn` dispatches through the shared table), deliver results or the trap.
    fn service_jit_invoke(&mut self) {
        self.sync_table();
        self.prime(temen_coop_mapped());
        let code = temen_coop_jit_code();
        let inst = match self.unit_insts.get(&code) {
            Some(i) => *i,
            None => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(temen_coop_jit_wasm_ptr(), temen_coop_jit_wasm_len())
                }
                .to_vec();
                let i = self.instantiate(&bytes);
                self.unit_insts.insert(code, i);
                i
            }
        };
        let f0 = inst.get_func(&self.store, "f0").expect("unit exports f0");
        let n = temen_coop_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(temen_coop_argv_ptr(), n) };
        let ptypes = unsafe { std::slice::from_raw_parts(temen_coop_jit_param_types_ptr(), n) };
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        for (a, tc) in argv.iter().zip(ptypes) {
            params.push(match tc {
                0 => Val::I32(*a as i32),
                1 => Val::I64(*a),
                2 => Val::F32(f32::from_bits(*a as u32).into()),
                _ => Val::F64(f64::from_bits(*a as u64).into()),
            });
        }
        let rn = temen_coop_jit_result_types_len();
        let rtypes = unsafe { std::slice::from_raw_parts(temen_coop_jit_result_types_ptr(), rn) };
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
                temen_coop_deliver_jit(slots.as_ptr(), slots.len());
            }
            Err(_) => temen_coop_deliver_jit_trap(),
        }
    }

    fn bounces(&self) -> &[u32] {
        &self.store.data().bounces
    }
}

/// Drive an opened coop session to DONE with the full B2 driver. Returns the driver (for its bounce
/// log) and the `(tierups, invokes)` counters.
fn drive_coop_b2_session(m: &temen_ir::Module) -> (CoopB2Driver, u32, u32) {
    let mut d = CoopB2Driver::new();
    let (mut tierups, mut invokes) = (0u32, 0u32);
    loop {
        match temen_coop_run() {
            COOP_RUN_JIT_INVOKE => {
                invokes += 1;
                assert!(invokes < 50, "runaway invokes");
                d.service_jit_invoke();
            }
            COOP_RUN_TIERUP => {
                tierups += 1;
                assert!(tierups < 50, "runaway tier-ups");
                let f = temen_coop_func() as usize;
                d.service_tierup(m.funcs[f].results.len());
            }
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", temen_status()),
        }
    }
    (d, tierups, invokes)
}

/// [`drive_coop_b2_session`] that tolerates a trap (an expected fault, e.g. an `Ro` store) instead of
/// panicking — drive to DONE or TRAP; the caller reads [`temen_status`] to assert trap-parity.
fn drive_coop_b2_session_allow_trap(m: &temen_ir::Module) -> (CoopB2Driver, u32) {
    let mut d = CoopB2Driver::new();
    let mut tierups = 0u32;
    loop {
        match temen_coop_run() {
            COOP_RUN_JIT_INVOKE => d.service_jit_invoke(),
            COOP_RUN_TIERUP => {
                tierups += 1;
                assert!(tierups < 100, "runaway tier-ups");
                let f = temen_coop_func() as usize;
                d.service_tierup(m.funcs[f].results.len());
            }
            COOP_RUN_DONE | COOP_RUN_TRAP => break,
            ev => panic!("unexpected pump event {ev} (status {})", temen_status()),
        }
    }
    (d, tierups)
}

/// #1009 paged, on the cooperative path: a rodata guest whose eligible leaf accesses its `Ro` page
/// (load succeeds, store traps) and, in a second guest, a page grown mid-invoke through a bounce —
/// both matching the `onramp_exec` oracle. The coop twin of `tierup_driver`'s paged differentials:
/// exercises `temen_coop_open`'s paged flip, `CoopRun`'s `page_checked` + `mem_map_*`, and the coop
/// driver's `"pagestate"` wiring + mid-invoke refresh.
#[test]
fn coop_rodata_and_midinvoke_grow_match_the_oracle() {
    let _g = ffi_guard();
    let (_out_h, mem_h) = onramp_out_mem_handles();

    // Ro load succeeds; Ro store traps — through the paged coop tier.
    for (store, want_trap) in [(false, false), (true, true)] {
        let body = if store {
            "  i64.store v0 v0\n  vl = i64.load v0\n  return vl\n"
        } else {
            "  vl = i64.load v0\n  return vl\n"
        };
        let src = format!(
            "memory 17\ndata ro 65536 \"temen-coop-rodata-payload!!\"\nfunc () -> (i64) {{\nblock 0 () {{\n  vp = i64.const 65536\n  vr = call 1 (vp)\n  return vr\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n{body}  }}\n}}\nexport 0 func \"_start\" 0\n"
        );
        let m = temen_text::parse_module(&src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let bytes = temen_encode::encode_module(&m);
        let want = onramp_exec(&m, b"");
        assert_eq!(
            want.status,
            if want_trap { STATUS_TRAP } else { STATUS_OK },
            "oracle sanity (store={store})"
        );
        let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
        assert_eq!(opened, 0, "coop open (status {})", temen_status());
        assert_ne!(
            temen_coop_paged(),
            0,
            "the rodata guest opens the coop run paged"
        );
        let (_d, tierups) = drive_coop_b2_session_allow_trap(&m);
        assert!(tierups >= 1, "the leaf tiers up on the coop path");
        assert_eq!(
            temen_status(),
            want.status,
            "coop status parity (store={store})"
        );
        if !want_trap {
            assert_eq!(
                temen_coop_value(),
                want.value,
                "coop value parity (Ro load)"
            );
        }
        temen_coop_close();
    }

    // A page grown mid-invoke through a bounce round-trips (the pagestate is refreshed in the bounce).
    const X: i64 = 4321;
    let src = format!(
        "memory 16\ndata ro 32768 \"temen-coop-rodata-flip!!\"\nfunc () -> (i64) {{\nblock 0 () {{\n  vx = i64.const {X}\n  vr = call 1 (vx)\n  return vr\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vg = call 2 (v0)\n  va = i64.const 65552\n  i64.store va v0\n  vl = i64.load va\n  return vl\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vas = i32.const {mem_h}\n  voff = i64.const 65536\n  vlen = i64.const 16384\n  vprot = i32.const 3\n  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)\n  return v0\n  }}\n}}\nexport 0 func \"_start\" 0\n"
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity (mid-invoke grow)");
    assert_eq!(want.value, X, "oracle value");
    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "coop open (status {})", temen_status());
    assert_ne!(temen_coop_paged(), 0, "paged");
    let (d, tierups) = drive_coop_b2_session_allow_trap(&m);
    assert!(tierups >= 1, "the leaf tiers up");
    assert!(
        !d.bounces().is_empty(),
        "the grow helper bounces: {:?}",
        d.bounces()
    );
    assert_eq!(
        temen_status(),
        want.status,
        "coop status parity (mid-invoke grow)"
    );
    assert_eq!(
        temen_coop_value(),
        want.value,
        "coop value parity through the grown page"
    );
    temen_coop_close();
}

/// The paged-flip gap: a guest that `protect`s its own pages (`call.cap 5 2`) but carries **no**
/// `readonly` data segment. Keyed on rodata alone, `temen_coop_open` emitted it non-paged, the
/// emitter's window-remapping gate module-gated it to emit-nothing, and the open **declined** —
/// interpreter-only for a guest paged mode carries fine. Now the flip also keys on
/// `module_uses_unmap_protect`: the guest opens paged, the leaf tiers up, and the post-`protect`
/// access matches the oracle (Ro load succeeds, Ro store traps) through the per-event
/// `sync_pagestate` refresh.
#[test]
fn coop_unmap_protect_guest_without_rodata_opens_paged() {
    let _g = ffi_guard();
    let (_out_h, mem_h) = onramp_out_mem_handles();

    for (store, want_trap) in [(false, false), (true, true)] {
        let body = if store {
            "  i64.store v0 v0\n  vl = i64.load v0\n  return vl\n"
        } else {
            "  vl = i64.load v0\n  return vl\n"
        };
        // `_start` seeds the page while it is still Rw, `protect`s [64 KiB, 80 KiB) read-only
        // (interp-serviced — a call.cap-bearing function never emits), then calls the leaf.
        let src = format!(
            "memory 17\nfunc () -> (i64) {{\nblock 0 () {{\n  vp = i64.const 65536\n  i64.store vp vp\n  vas = i32.const {mem_h}\n  voff = i64.const 65536\n  vlen = i64.const 16384\n  vprot = i32.const 1\n  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)\n  vres = call 1 (vp)\n  return vres\n  }}\n}}\nfunc (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n{body}  }}\n}}\nexport 0 func \"_start\" 0\n"
        );
        let m = temen_text::parse_module(&src).expect("parse");
        temen_verify::verify_module(&m).expect("verify");
        let bytes = temen_encode::encode_module(&m);
        let want = onramp_exec(&m, b"");
        assert_eq!(
            want.status,
            if want_trap { STATUS_TRAP } else { STATUS_OK },
            "oracle sanity (store={store})"
        );
        let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
        assert_eq!(opened, 0, "coop open (status {})", temen_status());
        assert_ne!(
            temen_coop_paged(),
            0,
            "the unmap/protect guest opens the coop run paged (no rodata needed)"
        );
        let (_d, tierups) = drive_coop_b2_session_allow_trap(&m);
        assert!(tierups >= 1, "the leaf tiers up on the coop path");
        assert_eq!(
            temen_status(),
            want.status,
            "coop status parity (store={store})"
        );
        if !want_trap {
            assert_eq!(
                temen_coop_value(),
                want.value,
                "coop value parity (Ro load)"
            );
        }
        temen_coop_close();
    }
}

/// The added constant of the `call.dyn`-reached leaf (`f2`), distinct from the unit's `UNIT_K`.
const LEAF_K: i64 = 424242;

/// A **threaded** guest whose tiered-up leaf `call.dyn`s another **emitted** leaf (#880, the
/// native edge, on the cooperative path). `_start` (the root vCPU) spawns a worker (`f3`, returns 0),
/// `vm_map`-grows `[64 KiB, 80 KiB)`, calls `f1` (which tiers up and `call.dyn`s `f2` at table
/// slot 2 — dispatched natively through the synced shared table), joins the worker, sums, streams, and
/// returns. `f2` stores into the grown page, so the tier-up's `"mapped"` bound must admit the growth.
/// The `thread.spawn` routes it to the cooperative driver; the `call.dyn` is the B2 edge.
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
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
  vj = thread.join vt
  vsum = i64.add vres vj
  vsl = i64.const {SLOT}
  i64.store vsl vsum
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  vr = call.dyn (i64) -> (i64) vs2 (v0)
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
    let _g = ffi_guard();
    let m = temen_text::parse_module(&coop_indirect_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // Oracle: the plain bytecode path. `f1(probe) = f2(probe) + 1 = (probe + LEAF_K) + 1`; worker 0.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + LEAF_K + 1,
        "oracle: call_indirect(f2)(probe) + 1, through the grown page"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded call.dyn guest (status {})",
        temen_status()
    );

    let (d, tierups, invokes) = drive_coop_b2_session(&m);
    // Non-vacuity: the `call.dyn`-bearing leaf `f1` tiered up (#880), and the indirect edge to
    // the emitted `f2` was **native** — never a bounce (an empty bounce log proves the shared-table
    // dispatch, not the interpreter, serviced it).
    assert!(tierups >= 1, "the call.dyn leaf must tier up (#880)");
    assert_eq!(invokes, 0, "no vm_jit units in this guest");
    assert!(
        d.bounces().is_empty(),
        "the indirect edge targets an emitted function — native, never bounced: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    temen_coop_close();
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

/// A **threaded** `vm_jit_*` guest whose tiered-up leaf `call.dyn`s an **installed unit's**
/// emitted `f0` (#880 old→new native, on the cooperative path). `_start` (the root vCPU) spawns a
/// worker (`f2`, returns 0), `vm_jit_compile`s `f(x)=x+7`, installs it (getting a runtime slot past the
/// program's `f{i}` prefix), calls `f1(slot, X)` — which tiers up and `call.dyn`s the install slot
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
        let u = temen_text::parse_module(unit_src).expect("parse unit");
        temen_verify::verify_module(&u).expect("verify unit");
        temen_encode::encode_module(&u)
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
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vcode)
  vx = i64.const {X}
  vres = call 1 (vslot, vx)
  vj = thread.join vt
  vfin = i64.add vres vj
  vsl = i64.const {SLOT}
  i64.store vsl vfin
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vfin
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vslot: i64, vx: i64) {{
  vs = i32.wrap_i64 vslot
  vr = call.dyn (i64) -> (i64) vs (vx)
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
    let _g = ffi_guard();
    let m = temen_text::parse_module(&coop_installed_unit_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // Oracle: `f1(slot, X) = unit(X) + 100 = (X + 7) + 100`; worker 0. (X = 4000.)
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        4000 + 7 + 100,
        "oracle: leaf → installed unit → +100"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded install-slot guest (status {})",
        temen_status()
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
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    temen_coop_close();
}

// ---- #926 slice 2g: the invoke-confined bounce — an emitted `Jit.invoke` unit bounces cross-tier ----

/// What the bounce helper adds, and where the unit probes the page a bounced callback grew mid-invoke.
const BOUNCE_K: i64 = 1000;
const MID_GROW_PROBE: i64 = 81920 + 8;

/// A **threaded** `vm_jit_*` guest whose surfaced `Jit.invoke` unit **bounces** cross-tier (#926 slice
/// 2g — the invoke-confined registry path; the #846 `linked_unit_bounces` shape, on the cooperative
/// driver). `_start` (root vCPU) spawns a worker (`f3`, returns 0), `vm_map`-grows `[64 KiB, 80 KiB)`,
/// compiles + `invoke2`s a unit whose emitted `f0`: (1) `call.dyn`s slot 1 → the program's
/// cap-calling helper `f1` — interpreter-resident, so the edge **bounces** via `env.call_interp` (the
/// callback grows `[80 KiB, 96 KiB)` and streams); (2) stores into that just-grown page (correct only
/// if the post-bounce `"mapped"` fan-out admits the growth); (3) `call.dyn`s slot 2 → the pure leaf
/// `f2` — emitted, **native**. The bounce runs through the invoke-confined fiber registry (`Vcpu`
/// parity), and must match the interpreted oracle bit-for-bit.
fn coop_invoke_bounce_guest_text() -> String {
    let (out_h, mem_h) = onramp_out_mem_handles();
    let unit_src = format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs1 = i32.const 1
  va = call.dyn (i64) -> (i64) vs1 (v0)
  vsum = i64.add v0 va
  vaddr = i64.const {MID_GROW_PROBE}
  i64.store vaddr vsum
  vld = i64.load vaddr
  vs2 = i32.const 2
  vc = call.dyn (i64) -> (i64) vs2 (vld)
  return vc
  }}
}}
"#
    );
    let unit = temen_text::parse_module(&unit_src).expect("unit parse");
    temen_verify::verify_module(&unit).expect("unit verify");
    let blob = temen_encode::encode_module(&unit);
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
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
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
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vsum
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vas = i32.const {mem_h}
  voff = i64.const 81920
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
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
    let _g = ffi_guard();
    let m = temen_text::parse_module(&coop_invoke_bounce_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // Oracle: bounce (+K, grow), grown-page store/load, native ×3; worker 0.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        (2 * PROBE + BOUNCE_K) * 3,
        "oracle: bounce (+K, grow), grown-page store/load, native ×3"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the threaded bouncing-invoke guest (status {})",
        temen_status()
    );

    let (d, _tierups, invokes) = drive_coop_b2_session(&m);
    // Non-vacuity: the unit ran emitted (invoke), and — post-#889 outlining (#1026 slice 1: this
    // driver now outlines cap sites exactly like the pump) — the helper `f1` itself **emits** (its
    // cap sites hoist to wrappers), so BOTH program slots dispatch natively and the live-window
    // bounces are f1's outlined wrappers 8/9 (append order: f0's map/compile/invoke/write = 4–7,
    // f1's grow/write = 8/9). The mid-invoke growth now happens inside wrapper 8's bounce — same
    // contract, the pump's `linked_unit_bounces` shape on the cooperative driver.
    assert!(
        invokes >= 1,
        "the linked unit must run emitted (invoke non-vacuity)"
    );
    assert!(
        d.bounces().contains(&8) && d.bounces().contains(&9),
        "the helper's outlined wrappers must bounce during the invoke: {:?}",
        d.bounces()
    );
    assert!(
        !d.bounces().contains(&1) && !d.bounces().contains(&2),
        "the helper and the leaf both emit — program slots dispatch natively, never bounce: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity (growth-mid-invoke visible post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity (bounce ordering included)"
    );
    temen_coop_close();
}

// ================================================================================================
// #1026 slice 1 — the single-vCPU pump's differentials, ported onto the cooperative driver.
//
// The pump (`temen_onramp_tierup_*`) is being collapsed into this driver (coop subsumes its admission
// set and is faster — see the issue). These ports establish equivalent coverage here BEFORE the
// pump and its harness are deleted: every pump differential without a coop twin gets one, run
// through the full `CoopB2Driver` against the same `onramp_exec` oracle. Guests are verbatim from
// `tierup_driver.rs` unless a comment says otherwise; the one semantic adjustment is the
// reachable-concurrency test — the pump DECLINES a runtime-reached `atomic.notify`, while this
// driver SERVICES it, so the port asserts full parity instead of a clean trap (strictly stronger).
// ================================================================================================

/// The ported guests' leaf constant (the pump file's `LEAF_K`; this file's own is 424242).
const PLEAF_K: i64 = 40404;
/// What the fiber-hosting unit adds inside its fiber (distinct from `UNIT_K`/`PLEAF_K`).
const FIBER_UNIT_K: i64 = 777;
/// What the direct-cross-tier helper adds.
const XT_K: i64 = 222;
/// Hot-loop constants: per-iteration multiplier, staging cell, iteration count (== bounce count).
const HOT_K: i64 = 77;
const SLOT2: i64 = 2064;
const HOT_N: i64 = 4;

/// Stage `blob` into guest memory at `base` as i64 stores (`prefix` uniquifies the SSA names).
fn stage_blob(prefix: &str, base: i64, blob: &[u8]) -> String {
    let mut s = String::new();
    for (i, chunk) in blob.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let val = i64::from_le_bytes(word);
        let addr = base + (i as i64) * 8;
        s.push_str(&format!(
            "  v{prefix}a{i} = i64.const {addr}\n  v{prefix}v{i} = i64.const {val}\n  i64.store v{prefix}a{i} v{prefix}v{i}\n"
        ));
    }
    s
}

/// The pump's `jit_guest_text_with`, verbatim (single-vCPU — no `thread.spawn`; this driver admits
/// it the same way): `_start` grows `[64 KiB, 80 KiB)`, stages `blob`, `vm_jit_compile`s it,
/// `vm_jit_invoke2`s with the grown-page probe, streams. `extra_funcs` appends program functions
/// (e.g. a fiber body a unit names by raw slot — #845).
fn coop_jit_guest_text_with(blob: &[u8], extra_funcs: &str) -> String {
    let (out_h, mem_h) = onramp_out_mem_handles();
    let blob_len = blob.len();
    let stores = stage_blob("s", BLOB_BASE, blob);
    format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
import 1 "vm_jit_invoke2" (i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
{stores}  vbp = i64.const {BLOB_BASE}
  vbl = i64.const {blob_len}
  vcode = call.import 0 (vbp, vbl)
  vprobe = i64.const {PROBE}
  vres = call.import 1 (vcode, vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
{extra_funcs}export 0 func "_start" 0
"#
    )
}

/// Pump port (#835): a **fiber**-using guest (`cont.new`/`cont.resume`/`suspend` — the JACL
/// scheduler shape) is admitted, its fibers serviced in-engine, and its eligible leaf still tiers
/// up — parity with the oracle.
#[test]
fn coop_fiber_guest_is_admitted_and_matches_the_oracle() {
    let _g = ffi_guard();
    let src = r#"memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 2
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 7
  vs1, vv1 = cont.resume v2 v3
  vs2, vv2 = cont.resume v2 v3
  vprobe = i64.const 1234
  vres = call 1 (vprobe)
  va = i64.add vv1 vres
  vb = i64.add va vv2
  vs1e = i64.extend_i32_s vs1
  vk1 = i64.const 1000000
  vc = i64.mul vs1e vk1
  vs2e = i64.extend_i32_s vs2
  vk2 = i64.const 10000000
  vd = i64.mul vs2e vk2
  ve = i64.add vb vc
  vf = i64.add ve vd
  return vf
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 40404
  vsum = i64.add v0 vk
  return vsum
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vk = i64.const 100
  vs = i64.add varg vk
  vv = suspend vs
  return vv
  }
}
export 0 func "_start" 0
"#;
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "a fiber guest must be admitted (status {})",
        temen_status()
    );
    let (_d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the leaf still tiers up beside the fibers");
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    temen_coop_close();
}

/// Pump port (#845): a guest-compiled **fiber-hosting** unit is admitted by the validator and runs
/// observably identical — on the **interpreter** on both paths: a fiber unit never emits, so the
/// driver surfaces **zero** JIT_INVOKE events (emitting one would run fiber ops on a wasm frame).
#[test]
fn coop_fiber_hosting_unit_is_admitted_and_matches_the_oracle() {
    let _g = ffi_guard();
    let fiber_unit_blob = {
        let src = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vs1, vv1 = cont.resume vk v0
  vs2, vv2 = cont.resume vk v0
  vr = i64.add vv1 vv2
  return vr
  }
}
"#;
        let m = temen_text::parse_module(src).expect("parse fiber unit");
        temen_verify::verify_module(&m).expect("verify fiber unit");
        temen_encode::encode_module(&m)
    };
    let fiber_body = format!(
        r#"func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vk = i64.const {FIBER_UNIT_K}
  vs = i64.add varg vk
  vv = suspend vs
  return vv
  }}
}}
"#
    );
    let m = temen_text::parse_module(&coop_jit_guest_text_with(&fiber_unit_blob, &fiber_body))
        .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(
        want.status, STATUS_OK,
        "the fiber-hosting unit compiles + invokes interpreted"
    );
    assert_eq!(
        want.value,
        2 * PROBE + FIBER_UNIT_K,
        "both yielded values arrive"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", temen_status());
    let (_d, _tierups, invokes) = drive_coop_b2_session(&m);
    assert_eq!(
        invokes, 0,
        "a fiber unit never runs emitted (compile_jit declines it) — the invoke stays interpreted"
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    temen_coop_close();
}

/// Pump port (#845's closed half, driver-independent): a **futex**-using unit (`atomic.notify`) is
/// still refused by the validator (`-EINVAL` from `vm_jit_compile`), so the guest's invoke of the
/// bogus code handle traps — pinned on the oracle (both drivers inherit it).
#[test]
fn coop_futex_unit_is_still_refused() {
    let _g = ffi_guard();
    let unit_src = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vaddr = i64.const 0
  vcnt = i32.const 0
  vw = atomic.notify vaddr vcnt
  return v0
  }
}
"#;
    let unit = temen_text::parse_module(unit_src).expect("parse futex unit");
    temen_verify::verify_module(&unit).expect("verify futex unit");
    let blob = temen_encode::encode_module(&unit);
    let m = temen_text::parse_module(&coop_jit_guest_text_with(&blob, "")).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let want = onramp_exec(&m, b"");
    assert_ne!(
        want.status, STATUS_OK,
        "a futex unit must fail compile (-EINVAL) → the invoke of the bogus handle traps"
    );
}

/// Pump port: a guest with **no** eligible leaf (all fiber ops, nothing emittable) must refuse the
/// open (`STATUS_UNSUPPORTED`) so the page runs the plain bytecode path — same fail-closed gate.
#[test]
fn coop_open_fails_closed_without_an_eligible_leaf() {
    let _g = ffi_guard();
    let src = r#"memory 16
func () -> (i64) {
block 0 () {
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  varg = i64.const 1
  vs1, vv1 = cont.resume vk varg
  return vv1
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vv = suspend varg
  return vv
  }
}
export 0 func "_start" 0
"#;
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);
    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened, -STATUS_UNSUPPORTED,
        "nothing for the emitted tier to run → clean refusal (bytecode fallback)"
    );
    temen_coop_close();
}

/// Pump port (#846, unit→**unit** native): the guest compiles + `install`s pure unit A, then
/// compiles unit B whose `call.dyn` reaches A's install slot — both emitted, the edge
/// dispatches natively (zero bounces), matching the oracle.
#[test]
fn coop_installed_unit_edge_dispatches_natively() {
    let _g = ffi_guard();
    let (out_h, jit_h) = onramp_out_jit_handles();
    let unit_a_src = r#"memory 16
func (i64) -> (i64) {
block 0 (v0: i64) {
  vk = i64.const 7
  vsum = i64.add v0 vk
  return vsum
  }
}
"#;
    let unit_b_src = r#"memory 16
func (i64, i64) -> (i64) {
block 0 (vslot: i64, vx: i64) {
  vs = i32.wrap_i64 vslot
  vr = call.dyn (i64) -> (i64) vs (vx)
  vk = i64.const 100
  vsum = i64.add vr vk
  return vsum
  }
}
"#;
    let blob_a = {
        let u = temen_text::parse_module(unit_a_src).expect("parse A");
        temen_verify::verify_module(&u).expect("verify A");
        temen_encode::encode_module(&u)
    };
    let blob_b = {
        let u = temen_text::parse_module(unit_b_src).expect("parse B");
        temen_verify::verify_module(&u).expect("verify B");
        temen_encode::encode_module(&u)
    };
    const A_BASE: i64 = 4096;
    const B_BASE: i64 = 8192;
    let (sa, sb) = (
        stage_blob("a", A_BASE, &blob_a),
        stage_blob("c", B_BASE, &blob_b),
    );
    let (la, lb) = (blob_a.len(), blob_b.len());
    const X: i64 = 5000;
    let src = format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
import 1 "vm_jit_invoke2" (i64, i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
{sa}{sb}  vap = i64.const {A_BASE}
  val = i64.const {la}
  vca = call.import 0 (vap, val)
  vjit = i32.const {jit_h}
  vslot = call.cap 11 3 (i64) -> (i64) vjit (vca)
  vbp = i64.const {B_BASE}
  vbl = i64.const {lb}
  vcb = call.import 0 (vbp, vbl)
  vx = i64.const {X}
  vres = call.import 1 (vcb, vslot, vx)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, X + 7 + 100, "oracle: B → installed A → +100");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", temen_status());
    let (d, _tierups, invokes) = drive_coop_b2_session(&m);
    assert!(invokes >= 1, "unit B must run emitted (non-vacuity)");
    assert!(
        d.bounces().is_empty(),
        "both units are emitted — the installed edge must dispatch natively: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    temen_coop_close();
}

/// Pump port (#1009 Mechanism 1): a guest with **more than 1024 functions** whose tier-up-eligible
/// dispatch leaf `call.dyn`s a slot beyond the 1024-slot floor — the emitted table and the
/// interpreter's `SharedSlots` must both size to `next_power_of_two(n_funcs)` so the two tiers mask
/// identically (a fixed-1024 mask silently reached a wrong, identically-typed function).
#[test]
fn coop_high_index_dispatch_beyond_the_table_floor_matches_the_oracle() {
    let _g = ffi_guard();
    let out_h = onramp_out_handle();
    const TARGET: usize = 1050;
    const INPUT: i64 = 12345;
    const DISTINCT: i64 = 777;
    assert_eq!(
        TARGET & 1023,
        26,
        "the fixed-1024 mask lands on an identity slot"
    );
    let mut fns = format!(
        r#"func () -> (i64) {{
block 0 () {{
  vin = i64.const {INPUT}
  vres = call 1 (vin)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
"#
    );
    fns.push_str(&format!(
        r#"func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs = i32.const {TARGET}
  vr = call.dyn (i64) -> (i64) vs (v0)
  return vr
  }}
}}
"#
    ));
    for i in 2..=TARGET {
        if i == TARGET {
            fns.push_str(&format!(
                "func (i64) -> (i64) {{\nblock 0 (v0: i64) {{\n  vk = i64.const {DISTINCT}\n  vr = i64.add v0 vk\n  return vr\n  }}\n}}\n"
            ));
        } else {
            fns.push_str("func (i64) -> (i64) {\nblock 0 (v0: i64) {\n  return v0\n  }\n}\n");
        }
    }
    let src = format!("memory 16\n{fns}export 0 func \"_start\" 0\n");
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        m.funcs.len() > (1usize << 10),
        "the guest must exceed the 1024-slot floor"
    );
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        INPUT + DISTINCT,
        "oracle: the dispatch reaches slot {TARGET}"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", temen_status());
    assert!(
        (1u32 << temen_coop_table_log2()) >= m.funcs.len() as u32,
        "the table must cover every function: 1<<{} < {}",
        temen_coop_table_log2(),
        m.funcs.len()
    );
    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the dispatch leaf must tier up");
    assert!(
        d.bounces().is_empty(),
        "the indirect edge reaches an emitted function natively: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity — the emitted high-index dispatch must reach slot {TARGET}, not {}",
        TARGET & 1023
    );
    temen_coop_close();
}

/// Pump port (#880, TIERUP-region bounce + growth): a tiered-up leaf's `call.dyn` lands on a
/// **shim** (the target hosts a fiber, so it stays interpreter-resident), whose callback grows the
/// window and streams — the leaf then stores into the just-grown page, correct only through the
/// post-bounce `"mapped"` fan-out; the bounce's stdout interleaves exactly as interpreted.
#[test]
fn coop_tierup_region_bounce_grows_and_streams() {
    let _g = ffi_guard();
    let (out_h, mem_h) = onramp_out_mem_handles();
    const X: i64 = 1234;
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vx = i64.const {X}
  vres = call 1 (vx)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  va = call.dyn (i64) -> (i64) vs2 (v0)
  vsum = i64.add v0 va
  vaddr = i64.const 65552
  i64.store vaddr vsum
  vld = i64.load vaddr
  return vld
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
  vf = i32.const 3
  vk2 = cont.new vf vzero
  vs1, vv1 = cont.resume vk2 vzero
  vk = i64.const {BOUNCE_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  return varg
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 2 * X + BOUNCE_K, "oracle value");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", temen_status());
    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the leaf must tier up");
    assert!(
        d.bounces().contains(&2),
        "the fiber-hosting target must bounce through the slot shim: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity (mid-region growth admitted post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity (bounce ordering)");
    temen_coop_close();
}

/// Pump port (#880, run-registry fiber persistence): a fiber created inside a TIERUP-region bounce
/// registers in the **run-level** registry — `_start` (interpreted) resumes it *after* the emitted
/// region returned. An invoke-confined registry would `FiberFault` where the interpreter succeeds;
/// pins the context split in the driver's bounce.
#[test]
fn coop_fiber_from_tierup_bounce_persists() {
    let _g = ffi_guard();
    let (out_h, _mem_h) = onramp_out_mem_handles();
    const X: i64 = 500;
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vx = i64.const {X}
  vr1 = call 1 (vx)
  vha = i64.const 2104
  vh = i64.load vha
  varg = i64.const 77
  vs2, vv2 = cont.resume vh varg
  vres = i64.add vr1 vv2
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  vr = call.dyn (i64) -> (i64) vs2 (v0)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vf = i32.const 3
  vsp = i64.const 0
  vk = cont.new vf vsp
  vs1, vv1 = cont.resume vk v0
  vha = i64.const 2104
  i64.store vha vk
  return vv1
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vk = i64.const 11
  vs = i64.add varg vk
  vv = suspend vs
  vk2 = i64.const 22
  vr = i64.add vv vk2
  return vr
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        (X + 11) + (77 + 22),
        "oracle: bounce-created fiber + later resume"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", temen_status());
    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the dispatching leaf must tier up");
    assert!(
        d.bounces().contains(&2),
        "the fiber-creating target must bounce: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity — the bounce-created fiber persisted into the run registry"
    );
    temen_coop_close();
}

/// Pump port (#888/#889, direct cross-tier over the live window): an eligible leaf's **direct
/// `Call`** to a cap-calling helper — post-#889 the helper's cap sites are outlined, so the helper
/// itself emits and the live-window bounces are its wrappers 4/5 (grow + stream), with the leaf
/// storing into the just-grown page after the bounce.
#[test]
fn coop_direct_cross_tier_call_bounces_over_the_live_window() {
    let _g = ffi_guard();
    let (out_h, mem_h) = onramp_out_mem_handles();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vx = i64.const 7
  vres = call 1 (vx)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {PLEAF_K}
  vsum = i64.add v0 vk
  vg = call 2 (vsum)
  vaddr = i64.const {PROBE}
  i64.store vaddr vg
  vld = i64.load vaddr
  return vld
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
  vk = i64.const {XT_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 7 + PLEAF_K + XT_K, "oracle value");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "#888: the leaf with a direct cross-tier call must be eligible (status {})",
        temen_status()
    );
    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the widened leaf must tier up (non-vacuity)");
    assert!(
        d.bounces().contains(&4) && d.bounces().contains(&5),
        "the cap-calling helper's outlined wrappers must bounce over the live window: {:?}",
        d.bounces()
    );
    assert!(
        !d.bounces().contains(&2),
        "#889: the helper itself must emit (its cap sites are outlined), not bounce: {:?}",
        d.bounces()
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(temen_coop_value(), want.value, "value parity");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity (bounce ordering included)"
    );
    temen_coop_close();
}

/// Pump port (#926 slice 1): a guest whose concurrency op is **linked but dead** (unreachable) is
/// admitted, its dead op never reached, and its pure leaf tiers up — parity with the oracle.
#[test]
fn coop_dead_concurrency_op_is_admitted_and_tiers_up() {
    let _g = ffi_guard();
    let (out_h, mem_h) = onramp_out_mem_handles();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {PLEAF_K}
  vsum = i64.add v0 vk
  vp = i64.const 8
  vaddr = i64.add v0 vp
  i64.store vaddr vsum
  vld = i64.load vaddr
  return vld
  }}
}}
func () -> (i64) {{
block 0 () {{
  vaddr = i64.const 0
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, PROBE + PLEAF_K, "oracle value");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "a dead concurrency op must not refuse (status {})",
        temen_status()
    );
    let (_d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(
        tierups >= 1,
        "the pure leaf must tier up beside the dead op"
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(temen_coop_value(), want.value, "value parity");
    // SAFETY: capture slots staged at DONE; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity");
    temen_coop_close();
}

/// Pump port (#926 slice 1), **adjusted**: a guest that actually reaches an `atomic.notify` — the
/// pump DECLINES it (`TIERUP_RUN_TRAP` → interpreter re-run); this driver **services** it (the
/// point of the cooperative scheduler), so the strictly stronger property holds: the guest runs to
/// DONE with full parity, its leaf having tiered up first.
#[test]
fn coop_reachable_concurrency_op_is_serviced_and_matches_the_oracle() {
    let _g = ffi_guard();
    let out_h = onramp_out_handle();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vprobe = i64.const 7
  vres = call 1 (vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  vaddr = i64.const 0
  vcnt = i32.const 1
  vn = atomic.notify vaddr vcnt
  vn64 = i64.extend_i32_s vn
  return vn64
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {PLEAF_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(
        want.status, STATUS_OK,
        "oracle sanity (notify with no waiters returns 0)"
    );

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "admitted (status {})", temen_status());
    let (_d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the leaf tiered up before the concurrency op");
    assert_eq!(
        temen_status(),
        want.status,
        "the reachable atomic.notify is SERVICED here (the pump declined it) — full parity"
    );
    assert_eq!(temen_coop_value(), want.value, "value parity");
    temen_coop_close();
}

/// Pump port (#889, the card shape): a hot loop with **one inline stdout `call.cap` per iteration**
/// — the site outlines, the loop emits and tiers up, and each iteration's cap write bounces to the
/// outlined wrapper (index 3), interleaving stdout exactly as interpreted.
#[test]
fn coop_hot_loop_with_inline_cap_write_emits_and_bounces_per_iteration() {
    let _g = ffi_guard();
    let out_h = onramp_out_handle();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vn = i64.const {HOT_N}
  vres = call 1 (vn)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vacc0 = i64.const 0
  br 1(v0, vacc0)
}}
block 1 (vi: i64, vacc: i64) {{
  vk = i64.const {HOT_K}
  vmul = i64.mul vi vk
  vacc2 = i64.add vacc vmul
  vsl2 = i64.const {SLOT2}
  i64.store vsl2 vacc2
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = call.cap 0 1 (i64, i64) -> (i64) vout (vsl2, vlen8)
  vone = i64.const -1
  vnext = i64.add vi vone
  vz = i64.const 0
  vgo = i64.ne vnext vz
  br_if vgo 1(vnext, vacc2) 2(vacc2)
}}
block 2 (vr: i64) {{
  return vr
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    let expect = HOT_K * (1..=HOT_N).sum::<i64>();
    assert_eq!(want.value, expect, "oracle value");

    let opened = temen_coop_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "#889: the cap-bearing hot loop must be admitted (status {})",
        temen_status()
    );
    let (d, tierups, _invokes) = drive_coop_b2_session(&m);
    assert!(tierups >= 1, "the hot loop must tier up (non-vacuity)");
    assert_eq!(
        d.bounces(),
        &vec![3u32; HOT_N as usize][..],
        "each iteration's cap write bounces to f1's outlined wrapper, nothing else bounces"
    );
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(temen_coop_value(), want.value, "value parity");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity — per-iteration writes interleave as interpreted"
    );
    temen_coop_close();
}

/// Pump port (#835 capstone, asset-gated): the real JACL self-hosted compiler-guest — `vm_jit_*`
/// imports + fiber scheduler — runs through this driver observably identical to the interpreter
/// oracle, emitted events serviced by the full B2 driver as they surface. Absent asset ⇒ SKIP.
#[test]
fn coop_jacl_compiler_runs_through_the_driver() {
    let _g = ffi_guard();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../codegen/selfhost/build/jacl_compiler.temen"
    );
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!(
            "SKIP: jacl_compiler.temen absent (run codegen/selfhost/build_compiler_temen.sh)"
        );
        return;
    };
    let compiler = temen_encode::decode_module(&bytes).expect("decode jacl_compiler.temen");
    const MACRO_SRC: &[u8] = b"defmacro unless {cond body} { syntax-quote [if ~cond {} ~body] }\n\
                               mut hit 0\nunless [== 1 2] { set hit 5 }\nhit\n";

    let want = onramp_exec(&compiler, MACRO_SRC);
    let opened = temen_coop_open(
        bytes.as_ptr(),
        bytes.len(),
        MACRO_SRC.as_ptr(),
        MACRO_SRC.len(),
        0,
    );
    if opened != 0 {
        // The driver may still refuse the giant module (fail-closed); pin the refusal is clean.
        assert_eq!(opened, -STATUS_UNSUPPORTED, "refusal must be clean");
        eprintln!("SKIP: coop refused the compiler-guest (clean bytecode fallback)");
        return;
    }
    // Uncapped drive (the compiler fires many events — the session driver's runaway caps are for
    // the small synthetic guests).
    let mut d = CoopB2Driver::new();
    loop {
        match temen_coop_run() {
            COOP_RUN_JIT_INVOKE => d.service_jit_invoke(),
            COOP_RUN_TIERUP => {
                let f = temen_coop_func() as usize;
                d.service_tierup(compiler.funcs[f].results.len());
            }
            COOP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", temen_status()),
        }
    }
    assert_eq!(temen_status(), want.status, "status parity with the oracle");
    assert_eq!(
        temen_coop_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(temen_stdout_ptr(), temen_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    temen_coop_close();
}
