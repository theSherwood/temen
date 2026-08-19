//! **Single-shot on-ramp leaf tier-up** (#809) — the native differential for the
//! `svm_onramp_tierup_*` event-pump FFI: an `InterpDriven` on-ramp guest (it `vm_map`-grows, so its
//! `_start` stays on the interpreter) with a tier-up-eligible pure leaf must run observably
//! identical to the plain bytecode path (`onramp_exec`, INVARIANTS.md #9) when the leaf is serviced
//! on the emitted wasm — with wasmi playing the browser's JS host (`driveTierupRun`), including the
//! #717 per-call `"mapped"` sync from [`svm_browser::svm_onramp_tierup_mapped`].
//!
//! The engine-seam halves are pinned elsewhere: `svm-wasm-jit/tests/tierup_grow_window.rs` proves
//! the event/sync contract over a raw `Vcpu`, and `page_ops.rs` the eligibility split. This test
//! proves the *browser FFI* wiring end-to-end: open → pump → service → deliver → capture slots.

use std::collections::HashMap;
use std::sync::Mutex;
use svm_browser::{
    onramp_exec, svm_onramp_tierup_argv_len, svm_onramp_tierup_argv_ptr,
    svm_onramp_tierup_call_interp, svm_onramp_tierup_close, svm_onramp_tierup_deliver,
    svm_onramp_tierup_deliver_jit, svm_onramp_tierup_deliver_jit_trap,
    svm_onramp_tierup_deliver_trap, svm_onramp_tierup_func, svm_onramp_tierup_jit_code,
    svm_onramp_tierup_jit_param_types_ptr, svm_onramp_tierup_jit_result_types_len,
    svm_onramp_tierup_jit_result_types_ptr, svm_onramp_tierup_jit_wasm_by_handle_len,
    svm_onramp_tierup_jit_wasm_by_handle_ptr, svm_onramp_tierup_jit_wasm_len,
    svm_onramp_tierup_jit_wasm_ptr, svm_onramp_tierup_mapped, svm_onramp_tierup_mapped_now,
    svm_onramp_tierup_nfuncs, svm_onramp_tierup_open, svm_onramp_tierup_run,
    svm_onramp_tierup_shim_ptr, svm_onramp_tierup_shim_wasm, svm_onramp_tierup_slot_code,
    svm_onramp_tierup_table_log2, svm_onramp_tierup_value, svm_onramp_tierup_wasm_len,
    svm_onramp_tierup_wasm_ptr, svm_onramp_tierup_win_len, svm_onramp_tierup_win_ptr,
    svm_run_value, svm_status, svm_stdout_len, svm_stdout_ptr, STATUS_OK, STATUS_TRAP,
    STATUS_UNSUPPORTED, TIERUP_RUN_DONE, TIERUP_RUN_JIT_INVOKE, TIERUP_RUN_TIERUP, TIERUP_RUN_TRAP,
};
use svm_interp::{Host, StreamRole};
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

/// The tier-up session statics are process-global (single-threaded wasm by design) — serialize the
/// tests in this binary across them.
static FFI_LOCK: Mutex<()> = Mutex::new(());

/// Where the wasmi harness places the mirrored window / env cell in the emitted module's memory.
const WIN_BASE: u32 = 0x4_0000;
const ENV_PTR: u32 = 1024;

/// Leaf probe: just inside the `vm_map`-grown page `[64 KiB, 80 KiB)` (whole 16-KiB page — a page
/// on every host page size), so the emitted leaf's store/load runs over grown-window bytes that the
/// synced `"mapped"` bound must admit.
const PROBE: i64 = 65536 + 16;
const LEAF_K: i64 = 40404;
/// Declared-prefix cell `_start` stages the leaf result in before streaming it to stdout.
const SLOT: i64 = 2048;

/// The on-ramp powerbox's stdout + memory handles, replicated from `grant_onramp_caps`'s grant
/// order (stdout, stdin, exit, memory, addrspace, …) against a fresh `Host` — deterministic per
/// session, so the import-free guest text can `cap.call` them directly.
fn onramp_handles() -> (i32, i32) {
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    let mem = h.grant_memory();
    (out, mem)
}

/// The guest: `_start` (interp-driven — it `vm_map`s and streams) grows `[64 KiB, 80 KiB)`, calls
/// the pure all-i64 leaf with the probe, stages the result at `SLOT`, writes those 8 bytes to
/// stdout, and returns the result. The leaf stores `probe + LEAF_K` at `probe + 8` (grown-page
/// bytes), loads it back, and returns it.
fn guest_text() -> String {
    let (out_h, mem_h) = onramp_handles();
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
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
export 0 func "_start" 0
"#
    )
}

/// Service one TIERUP on wasmi (the browser JS host's `driveTierupRun` role): mirror the live
/// window into a fresh instance's memory, write the event's `"mapped"` sync (#717), call
/// `f{func}(win, env, ...argv)`, copy the window back, and return the i64 results (or `None` on a
/// wasm trap — delivered as a trap to the vCPU).
fn service_on_wasmi(n_results: usize) -> Option<Vec<i64>> {
    // SAFETY: the vCPU is parked inside the TIERUP event; the session stash (wasm, argv, window)
    // is stable until the deliver call, and this thread is the only accessor (FFI_LOCK).
    let wasm = unsafe {
        std::slice::from_raw_parts(svm_onramp_tierup_wasm_ptr(), svm_onramp_tierup_wasm_len())
    };
    let func = svm_onramp_tierup_func();
    run_emitted_on_wasmi(wasm, &format!("f{func}"), n_results)
}

/// Service one JIT_INVOKE on wasmi (#835 — `driveTierupRun`'s §22 unit arm): same window-mirror /
/// `"mapped"`-sync mechanics, but the wasm is the *invoked unit*'s emit and the entry is its `f0`.
/// The result count comes from the event's own result-type operand (the JS host reads it the same
/// way). All-i64 operands only in this harness — asserted against the event's type codes.
fn service_jit_on_wasmi() -> Option<Vec<i64>> {
    // SAFETY: the vCPU is parked inside the JIT_INVOKE event; the session stash is stable until
    // the deliver call, and this thread is the only accessor (FFI_LOCK).
    let wasm = unsafe {
        std::slice::from_raw_parts(
            svm_onramp_tierup_jit_wasm_ptr(),
            svm_onramp_tierup_jit_wasm_len(),
        )
    };
    let ptypes = unsafe {
        std::slice::from_raw_parts(
            svm_onramp_tierup_jit_param_types_ptr(),
            svm_onramp_tierup_argv_len(),
        )
    };
    assert!(
        ptypes.iter().all(|&t| t == 1),
        "this harness marshals i64 slots only (type codes {ptypes:?})"
    );
    run_emitted_on_wasmi(wasm, "f0", svm_onramp_tierup_jit_result_types_len())
}

/// The shared wasmi half of both service arms: instantiate `wasm` over a mirrored window, write the
/// pending event's `"mapped"` sync (#717), call `entry(win, env, ...argv)`, copy the window back.
fn run_emitted_on_wasmi(wasm: &[u8], entry: &str, n_results: usize) -> Option<Vec<i64>> {
    // SAFETY: as the callers' — the pending event's operand stash is stable until the deliver.
    let argv = unsafe {
        std::slice::from_raw_parts(svm_onramp_tierup_argv_ptr(), svm_onramp_tierup_argv_len())
    };
    let win_len = svm_onramp_tierup_win_len();
    let win_ptr = svm_onramp_tierup_win_ptr() as *mut u8;
    let mapped = svm_onramp_tierup_mapped();

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
    // #846: B2-emitted units import the shared funcref table; a closed unit never call_indirects,
    // so an all-null table satisfies the import without changing behavior. (Linked-unit dispatch is
    // the full driver's job — `TierupDriver` below.)
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
                unreachable!("no cross-tier call expected from this leaf");
            },
        )
        .unwrap();
    let instance = linker
        .instantiate(&mut store, &module)
        .unwrap()
        .start(&mut store)
        .unwrap();
    // The #717 driver contract: the event's committed-extent snapshot lands in the emitted
    // `"mapped"` global before the call (the fuel global self-initializes to the standard budget).
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

    // Copy emitted writes back into the live window before the vCPU resumes.
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

#[test]
fn tierup_pump_matches_the_bytecode_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&guest_text()).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    // The oracle: the exact bytecode path the page falls back to today (`svm_run_onramp`'s core).
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + LEAF_K,
        "oracle computes through the grown page"
    );
    assert_eq!(
        want.stdout,
        (PROBE + LEAF_K).to_le_bytes().to_vec(),
        "oracle streams the staged result"
    );

    // The tier-up pump under test.
    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must accept the eligible-leaf guest (status {})",
        svm_status()
    );

    let n_results = m.funcs[1].results.len();
    let mut tierups = 0u32;
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_TIERUP => {
                tierups += 1;
                match service_on_wasmi(n_results) {
                    Some(res) => svm_onramp_tierup_deliver(res.as_ptr(), res.len()),
                    None => svm_onramp_tierup_deliver_trap(),
                }
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    assert!(
        tierups >= 1,
        "the leaf must actually run on the emitted tier (non-vacuity)"
    );

    assert_eq!(
        svm_status(),
        want.status,
        "status parity with the bytecode oracle"
    );
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the bytecode oracle"
    );
    assert_eq!(
        svm_run_value(),
        want.value,
        "the page-facing `svm_run_value` slot is staged too"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity with the bytecode oracle"
    );
    svm_onramp_tierup_close();
}

// ---- #835: the §22 vm_jit_* half of the pump -------------------------------------------------

/// What the guest-compiled unit adds to its probe (distinct from `LEAF_K` so a cross-wire shows).
const UNIT_K: i64 = 90909;
/// Where `_start` stages the unit blob bytes before `vm_jit_compile`.
const BLOB_BASE: i64 = 4096;

/// The unit the guest compiles at runtime: all-i64 `f(x) = *(x+8) after *(x+8) = x + UNIT_K` — a
/// store/load over the `vm_map`-grown page, so the emitted unit's masked window (bumped past the
/// declared 2^16 by the pump's emitter) and the per-invoke `"mapped"` sync are both load-bearing.
/// Declares the guest's memory (16) — the validator's memory-match precondition.
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
    let m = svm_text::parse_module(&src).expect("parse unit");
    svm_verify::verify_module(&m).expect("verify unit");
    svm_encode::encode_module(&m)
}

/// The `vm_jit_*`-importing guest (#835 — the JACL compiler shape in miniature): `_start` grows
/// `[64 KiB, 80 KiB)`, stages `blob` into memory, `vm_jit_compile`s it, `vm_jit_invoke2`s the
/// compiled unit with the grown-page probe, and streams the result — no tier-up-eligible leaf at
/// all, so the open admits it purely on the `Jit` import (the widened #835 gate). `extra_funcs`
/// appends program functions after `_start` (e.g. a fiber body a unit names by raw slot — #845).
fn jit_guest_text_with(blob: &[u8], extra_funcs: &str) -> String {
    let (out_h, mem_h) = onramp_handles();
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
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
{extra_funcs}export 0 func "_start" 0
"#
    )
}

fn jit_guest_text_for(blob: &[u8]) -> String {
    jit_guest_text_with(blob, "")
}

/// #835 differential: the `vm_jit_*` guest through the pump — its runtime-compiled unit serviced
/// on wasmi via the JIT_INVOKE event — must match the bytecode oracle (`onramp_exec`, which
/// services the invoke on the interpreter), with the emitted-invoke non-vacuity counter and the
/// grown-extent `"mapped"` operand pinned.
#[test]
fn jit_invoke_pump_matches_the_bytecode_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&jit_guest_text_for(&unit_blob())).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        PROBE + UNIT_K,
        "oracle invokes the guest-compiled unit through the grown page"
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "open must admit the vm_jit_* guest (#835) even with no eligible leaf (status {})",
        svm_status()
    );

    let mut jit_invokes = 0u32;
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_JIT_INVOKE => {
                jit_invokes += 1;
                // The event's committed extent is the grown window — the value the JS host writes
                // to the unit's `"mapped"` global (#717; a declared-only bound would refuse the
                // probe and diverge).
                assert_eq!(
                    svm_onramp_tierup_mapped(),
                    65536 + 16384,
                    "the JIT_INVOKE mapped operand carries the grown extent"
                );
                match service_jit_on_wasmi() {
                    Some(res) => svm_onramp_tierup_deliver_jit(res.as_ptr(), res.len()),
                    None => svm_onramp_tierup_deliver_jit_trap(),
                }
            }
            TIERUP_RUN_TIERUP => {
                // No eligible leaf exists in this guest; reaching here is a wiring bug.
                panic!("unexpected TIERUP from the leafless vm_jit_* guest");
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    assert!(
        jit_invokes >= 1,
        "the unit must actually run on its emitted wasm (non-vacuity)"
    );

    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_onramp_tierup_close();
}

/// What the fiber-hosting unit adds inside its fiber (distinct from `UNIT_K`/`LEAF_K`).
const FIBER_UNIT_K: i64 = 777;

/// A **fiber-hosting** unit (#845): `f0(x)` spins up a fiber over the *program's* fiber body
/// (raw natural-table slot 1 — the module-0-entry shape `step_vcpu`'s renegotiated arms resolve;
/// a fiber over the unit's own function is the engine's documented deferred case), resumes it
/// twice (the second runs it to completion — the §22 "runs its own scheduler to completion"
/// contract), and returns the two yielded values summed: `2x + FIBER_UNIT_K`. No threads/futex,
/// no data — admissible under the renegotiated gate, rejectable only by the pre-#845 coarse
/// `uses_concurrency` sweep.
fn fiber_unit_blob() -> Vec<u8> {
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
    let m = svm_text::parse_module(src).expect("parse fiber unit");
    svm_verify::verify_module(&m).expect("verify fiber unit");
    svm_encode::encode_module(&m)
}

/// The program-side fiber body the unit names (guest func 1): suspends `arg + FIBER_UNIT_K`, then
/// returns whatever the second resume passes.
fn fiber_body_func() -> String {
    format!(
        r#"func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vk = i64.const {FIBER_UNIT_K}
  vs = i64.add varg vk
  vv = suspend vs
  return vv
  }}
}}
"#
    )
}

/// #845 differential: a guest-compiled **fiber-hosting** unit is admitted by the browser validator
/// (canonical §22 renegotiated gate — pre-fix, `vm_jit_compile` returned `-EINVAL` and the guest
/// trapped on the bogus code handle) and runs observably identical through the oracle and the pump.
/// It must run on the **interpreter** on both paths: `compile_jit`'s `reachable_concurrency` guard
/// never yields `WasmDriven` for a fiber unit, so the pump surfaces **zero** JIT_INVOKE events —
/// pinned, since emitting one would run fiber ops on a wasm frame (fail-closed stays closed).
#[test]
fn fiber_hosting_unit_is_admitted_and_matches_the_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let m = svm_text::parse_module(&jit_guest_text_with(&fiber_unit_blob(), &fiber_body_func()))
        .expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(
        want.status, STATUS_OK,
        "the fiber-hosting unit must compile + invoke on the interpreter (pre-#845 this was -EINVAL)"
    );
    assert_eq!(
        want.value,
        2 * PROBE + FIBER_UNIT_K,
        "both yielded values arrive (the unit ran its fiber to completion)"
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let mut jit_invokes = 0u32;
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_JIT_INVOKE => {
                jit_invokes += 1;
                match service_jit_on_wasmi() {
                    Some(res) => svm_onramp_tierup_deliver_jit(res.as_ptr(), res.len()),
                    None => svm_onramp_tierup_deliver_jit_trap(),
                }
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    assert_eq!(
        jit_invokes, 0,
        "a fiber unit never runs emitted (compile_jit declines it) — the invoke stays interpreted"
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle"
    );
    svm_onramp_tierup_close();
}

/// #845's other half stays closed: a **futex**-using unit (`atomic.notify`) is still refused by
/// the validator (`-EINVAL` from `vm_jit_compile`), so the guest's invoke of the bogus code handle
/// traps — on the oracle and the pump identically. (Threads/futex need multi-vCPU orchestration a
/// synchronous invoke can never host — the un-renegotiated half of the §22 gate.)
#[test]
fn futex_unit_is_still_refused() {
    let _g = FFI_LOCK.lock().unwrap();
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
    let unit = svm_text::parse_module(unit_src).expect("parse futex unit");
    svm_verify::verify_module(&unit).expect("verify futex unit");
    let blob = svm_encode::encode_module(&unit);
    let m = svm_text::parse_module(&jit_guest_text_for(&blob)).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let want = onramp_exec(&m, b"");
    assert_ne!(
        want.status, STATUS_OK,
        "a futex unit must fail compile (-EINVAL) → the invoke of the bogus handle traps"
    );
}

/// #835 gate pin: a **fiber**-using guest (`cont.new`/`cont.resume`/`suspend` — the JACL scheduler
/// shape) is admitted now (`step_vcpu` services fibers in-engine; only threads/futex refuse) and
/// runs through the pump observably identical to the oracle, its eligible leaf still tiering up.
#[test]
fn fiber_guest_is_admitted_and_matches_the_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
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
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "a fiber guest must be admitted now (#835 gate — status {})",
        svm_status()
    );
    let n_results = m.funcs[1].results.len();
    let mut tierups = 0u32;
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_TIERUP => {
                tierups += 1;
                match service_on_wasmi(n_results) {
                    Some(res) => svm_onramp_tierup_deliver(res.as_ptr(), res.len()),
                    None => svm_onramp_tierup_deliver_trap(),
                }
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    assert!(tierups >= 1, "the leaf still tiers up beside the fibers");
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle"
    );
    svm_onramp_tierup_close();
}

/// #835 capstone (asset-gated): the real JACL self-hosted compiler-guest — `vm_jit_*` imports +
/// fiber scheduler — runs through the pump observably identical to the interpreter oracle. Emitted
/// events are serviced when they surface (a symtab-linked macro unit may stay interpreted — the
/// emitter admits closed units only — so no non-vacuity is asserted here; the synthetic tests above
/// pin that). Cross-repo asset (see `jacl_selfhost_jit.rs`); absent ⇒ SKIP.
#[test]
fn jacl_compiler_runs_through_the_pump() {
    let _g = FFI_LOCK.lock().unwrap();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../../codegen/selfhost/build/jacl_compiler.svmb"
    );
    let Ok(bytes) = std::fs::read(path) else {
        eprintln!("SKIP: jacl_compiler.svmb absent (run codegen/selfhost/build_compiler_svmb.sh)");
        return;
    };
    let compiler = svm_encode::decode_module(&bytes).expect("decode jacl_compiler.svmb");
    const MACRO_SRC: &[u8] = b"defmacro unless {cond body} { syntax-quote [if ~cond {} ~body] }\n\
                               mut hit 0\nunless [== 1 2] { set hit 5 }\nhit\n";

    let want = onramp_exec(&compiler, MACRO_SRC);
    let opened = svm_onramp_tierup_open(
        bytes.as_ptr(),
        bytes.len(),
        MACRO_SRC.as_ptr(),
        MACRO_SRC.len(),
        0,
    );
    if opened != 0 {
        // The pump may still refuse the giant module (e.g. its tier-up emit declines) — the
        // fail-closed contract; the page then runs the bytecode path. Pin the refusal is clean.
        assert_eq!(opened, -STATUS_UNSUPPORTED, "refusal must be clean");
        eprintln!("SKIP: pump refused the compiler-guest (clean bytecode fallback)");
        return;
    }
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_TIERUP => {
                let f = svm_onramp_tierup_func() as usize;
                let n_results = compiler.funcs[f].results.len();
                match service_on_wasmi(n_results) {
                    Some(res) => svm_onramp_tierup_deliver(res.as_ptr(), res.len()),
                    None => svm_onramp_tierup_deliver_trap(),
                }
            }
            TIERUP_RUN_JIT_INVOKE => match service_jit_on_wasmi() {
                Some(res) => svm_onramp_tierup_deliver_jit(res.as_ptr(), res.len()),
                None => svm_onramp_tierup_deliver_jit_trap(),
            },
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity with the oracle");
    svm_onramp_tierup_close();
}

/// Fail-closed open: a guest with **no** eligible leaf (its only function `cap.call`s) must refuse
/// the tier-up session (`STATUS_UNSUPPORTED`) so the page runs the plain bytecode path — the pump
/// never opens a run that could only ever interpret.
#[test]
fn open_fails_closed_without_an_eligible_leaf() {
    let _g = FFI_LOCK.lock().unwrap();
    // Every function must stay non-emittable THROUGH the #889 outlining pass — a lone `cap.call`
    // `_start` no longer qualifies (its site outlines and the function becomes eligible), so this
    // guest is all fiber ops: `_start` hosts a fiber (out of subset), the body suspends (out of
    // subset) — nothing for the emitted tier to ever run.
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vf = i32.const 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  varg = i64.const 1
  vs1, vv1 = cont.resume vk varg
  return vv1
  }}
}}
{}export 0 func "_start" 0
"#,
        fiber_body_func()
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);
    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened, -STATUS_UNSUPPORTED,
        "no eligible leaf → refuse the session"
    );
    assert_eq!(svm_status(), STATUS_UNSUPPORTED);
    // A refused open leaves no session: the pump reports TRAP immediately, not a phantom run.
    assert_eq!(svm_onramp_tierup_run(), TIERUP_RUN_TRAP);
    svm_onramp_tierup_close();
}

// ---- #846: the linked-unit driver (B2 table + live-state bounce), wasmi as the JS host --------

/// Shared state the bounce closure reaches through the wasmi store: the one memory, every live
/// instance's `"mapped"`/`"fuel"` globals (the post-bounce fan-out set), and the bounce log the
/// tests read for per-edge non-vacuity.
#[derive(Default)]
struct DriverData {
    mem: Option<Memory>,
    mapped_globals: Vec<wasmi::Global>,
    fuel_globals: Vec<wasmi::Global>,
    bounces: Vec<u32>,
}

/// `driveTierupRun`'s #846 shape with wasmi playing the JS host: one funcref table shared by every
/// instance (main `f{i}`s, unit `f0`s, bounce shims), a live-state `call_interp` that mirrors the
/// window across the wasm/engine boundary around each bounce and fans the fresh `mapped` extent out
/// to every instance, and a per-event table sync from the engine's slot mirror. The browser needs
/// none of the window mirroring (its instances share the cdylib's memory); everything else is the
/// JS driver, line for line.
struct TierupDriver {
    store: Store<DriverData>,
    engine: Engine,
    memory: Memory,
    table: wasmi::Table,
    main: wasmi::Instance,
    unit_insts: HashMap<i32, wasmi::Instance>,
    /// Bounce shims keyed by `(slot, occupant code)` (`-2` = a program-function slot) so an
    /// uninstall/reinstall regenerates against the new occupant's signature.
    shims: HashMap<(u32, i32), wasmi::Func>,
    win_len: usize,
}

impl TierupDriver {
    /// Build the driver for the freshly opened pump session: memory sized for the mirrored window,
    /// the shared table, and the main emitted module instantiated against both.
    fn new() -> TierupDriver {
        let engine = Engine::default();
        let mut store: Store<DriverData> = Store::new(&engine, DriverData::default());
        let win_len = svm_onramp_tierup_win_len();
        let pages = ((WIN_BASE as usize + win_len) as u32).div_ceil(1 << 16) + 1;
        let memory = Memory::new(&mut store, MemoryType::new(pages, None)).unwrap();
        store.data_mut().mem = Some(memory);
        let tsize = 1u32 << svm_onramp_tierup_table_log2();
        let table = wasmi::Table::new(
            &mut store,
            wasmi::TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
            Val::FuncRef(wasmi::FuncRef::null()),
        )
        .unwrap();
        let main_wasm = unsafe {
            std::slice::from_raw_parts(svm_onramp_tierup_wasm_ptr(), svm_onramp_tierup_wasm_len())
        }
        .to_vec();
        let mut d = TierupDriver {
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
    /// live-state bounce, registering its `"mapped"`/`"fuel"` globals for the fan-out set.
    fn instantiate(&mut self, wasm: &[u8]) -> wasmi::Instance {
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
                    let win_ptr = svm_onramp_tierup_win_ptr() as *mut u8;
                    // The emitted frames may have written the window since the last sync — make
                    // those writes visible to the engine before the callback runs. (The browser
                    // skips both copies: its instances share the engine's memory.)
                    let mut w = vec![0u8; win_len];
                    mem.read(&c, WIN_BASE as usize, &mut w).unwrap();
                    // SAFETY: the vCPU is parked on the pending invoke; the window is exclusive.
                    unsafe { std::slice::from_raw_parts_mut(win_ptr, win_len) }.copy_from_slice(&w);
                    // Marshal the scratch out, bounce, marshal back.
                    let mut slots = [0u8; 512];
                    mem.read(&c, args_ptr as usize, &mut slots).unwrap();
                    let rc = svm_onramp_tierup_call_interp(target as u32, slots.as_mut_ptr());
                    let live = unsafe { std::slice::from_raw_parts(win_ptr, win_len) };
                    mem.write(&mut c, WIN_BASE as usize, live).unwrap();
                    mem.write(&mut c, args_ptr as usize, &slots).unwrap();
                    // #717 fan-out: the callback may have vm_map-grown the window — every live
                    // instance's bound must admit the growth from the next instruction on.
                    let now = svm_onramp_tierup_mapped_now();
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
    fn shim(&mut self, slot: u32, code: i32) -> Option<wasmi::Func> {
        if let Some(f) = self.shims.get(&(slot, code)) {
            return Some(*f);
        }
        let len = svm_onramp_tierup_shim_wasm(slot);
        if len == 0 {
            return None;
        }
        let bytes =
            unsafe { std::slice::from_raw_parts(svm_onramp_tierup_shim_ptr(), len) }.to_vec();
        let inst = self.instantiate(&bytes);
        let f = inst.get_func(&self.store, "t").expect("shim exports t");
        self.shims.insert((slot, code), f);
        Some(f)
    }

    /// Rebuild the shared table from the engine's slot mirror — the per-event sync (installs only
    /// happen between events, so a synced table is exact for the whole event).
    fn sync_table(&mut self) {
        let nfuncs = svm_onramp_tierup_nfuncs();
        let tsize = 1usize << svm_onramp_tierup_table_log2();
        for slot in 0..tsize {
            let entry: Option<wasmi::Func> = if slot < nfuncs {
                match self.main.get_func(&self.store, &format!("f{slot}")) {
                    Some(f) => Some(f),
                    None => self.shim(slot as u32, -2),
                }
            } else {
                let code = svm_onramp_tierup_slot_code(slot as u32);
                if code < 0 {
                    None
                } else if svm_onramp_tierup_jit_wasm_by_handle_len(code) > 0 {
                    let inst = match self.unit_insts.get(&code) {
                        Some(i) => *i,
                        None => {
                            let bytes = unsafe {
                                std::slice::from_raw_parts(
                                    svm_onramp_tierup_jit_wasm_by_handle_ptr(),
                                    svm_onramp_tierup_jit_wasm_by_handle_len(code),
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
                Some(f) => wasmi::FuncRef::new(f),
                None => wasmi::FuncRef::null(),
            };
            self.table
                .set(&mut self.store, slot as u64, Val::FuncRef(fr))
                .unwrap();
        }
    }

    /// Service the pending JIT_INVOKE: sync window + table + globals, run the invoked unit's `f0`,
    /// mirror the window back, deliver results or the trap.
    fn service_jit_invoke(&mut self) {
        self.sync_table();
        let win_ptr = svm_onramp_tierup_win_ptr() as *mut u8;
        // SAFETY: the vCPU is parked on the pending invoke; the window is exclusive.
        let live = unsafe { std::slice::from_raw_parts(win_ptr, self.win_len) };
        self.memory
            .write(&mut self.store, WIN_BASE as usize, live)
            .unwrap();
        self.memory
            .write(&mut self.store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
            .unwrap();
        let mapped = svm_onramp_tierup_mapped();
        for g in self.store.data().mapped_globals.clone() {
            g.set(&mut self.store, Val::I64(mapped)).unwrap();
        }
        for g in self.store.data().fuel_globals.clone() {
            g.set(&mut self.store, Val::I64(1 << 61)).unwrap();
        }
        let code = svm_onramp_tierup_jit_code();
        let inst = match self.unit_insts.get(&code) {
            Some(i) => *i,
            None => {
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        svm_onramp_tierup_jit_wasm_ptr(),
                        svm_onramp_tierup_jit_wasm_len(),
                    )
                }
                .to_vec();
                let i = self.instantiate(&bytes);
                self.unit_insts.insert(code, i);
                i
            }
        };
        let f0 = inst.get_func(&self.store, "f0").expect("unit exports f0");
        let n = svm_onramp_tierup_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(svm_onramp_tierup_argv_ptr(), n) };
        let ptypes =
            unsafe { std::slice::from_raw_parts(svm_onramp_tierup_jit_param_types_ptr(), n) };
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        for (a, tc) in argv.iter().zip(ptypes) {
            params.push(match tc {
                0 => Val::I32(*a as i32),
                1 => Val::I64(*a),
                2 => Val::F32(f32::from_bits(*a as u32).into()),
                _ => Val::F64(f64::from_bits(*a as u64).into()),
            });
        }
        let rn = svm_onramp_tierup_jit_result_types_len();
        let rtypes =
            unsafe { std::slice::from_raw_parts(svm_onramp_tierup_jit_result_types_ptr(), rn) };
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
        // Mirror the emitted writes back before the vCPU resumes (also after a trap — partial
        // writes are visible interpreted too).
        let mut buf = vec![0u8; self.win_len];
        self.memory
            .read(&self.store, WIN_BASE as usize, &mut buf)
            .unwrap();
        // SAFETY: see above.
        unsafe { std::slice::from_raw_parts_mut(win_ptr, self.win_len) }.copy_from_slice(&buf);
        match ran {
            Ok(()) => {
                let slots: Vec<i64> = results
                    .iter()
                    .zip(rtypes)
                    .map(|(v, tc)| match (v, tc) {
                        (Val::I32(x), _) => *x as i64,
                        (Val::I64(x), _) => *x,
                        (Val::F32(x), _) => f32::from(*x).to_bits() as i64,
                        (Val::F64(x), _) => f64::from(*x).to_bits() as i64,
                        _ => panic!("non-scalar result (type code {tc})"),
                    })
                    .collect();
                svm_onramp_tierup_deliver_jit(slots.as_ptr(), slots.len());
            }
            Err(_) => svm_onramp_tierup_deliver_jit_trap(),
        }
    }

    fn bounces(&self) -> &[u32] {
        &self.store.data().bounces
    }

    /// Service the pending TIERUP through the shared table (#880): sync window + table + globals,
    /// run the main module's `f{func}` (whose `call_indirect` now dispatches through the table —
    /// natively or via bounce shims), mirror back, deliver results or the trap.
    fn service_tierup(&mut self, n_results: usize) {
        self.sync_table();
        let win_ptr = svm_onramp_tierup_win_ptr() as *mut u8;
        // SAFETY: the vCPU is parked on the pending event; the window is exclusive.
        let live = unsafe { std::slice::from_raw_parts(win_ptr, self.win_len) };
        self.memory
            .write(&mut self.store, WIN_BASE as usize, live)
            .unwrap();
        self.memory
            .write(&mut self.store, ENV_PTR as usize, &i64::MAX.to_le_bytes())
            .unwrap();
        let mapped = svm_onramp_tierup_mapped();
        for g in self.store.data().mapped_globals.clone() {
            g.set(&mut self.store, Val::I64(mapped)).unwrap();
        }
        for g in self.store.data().fuel_globals.clone() {
            g.set(&mut self.store, Val::I64(1 << 61)).unwrap();
        }
        let func = svm_onramp_tierup_func();
        let f = self
            .main
            .get_func(&self.store, &format!("f{func}"))
            .unwrap_or_else(|| panic!("f{func} not exported"));
        let n = svm_onramp_tierup_argv_len();
        // SAFETY: pending-event operand stash, stable until the deliver.
        let argv = unsafe { std::slice::from_raw_parts(svm_onramp_tierup_argv_ptr(), n) };
        let mut params = vec![Val::I32(WIN_BASE as i32), Val::I32(ENV_PTR as i32)];
        params.extend(argv.iter().map(|a| Val::I64(*a)));
        let mut results: Vec<Val> = (0..n_results).map(|_| Val::I64(0)).collect();
        let ran = f.call(&mut self.store, &params, &mut results);
        let mut buf = vec![0u8; self.win_len];
        self.memory
            .read(&self.store, WIN_BASE as usize, &mut buf)
            .unwrap();
        // SAFETY: see above.
        unsafe { std::slice::from_raw_parts_mut(win_ptr, self.win_len) }.copy_from_slice(&buf);
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
                svm_onramp_tierup_deliver(slots.as_ptr(), slots.len());
            }
            Err(_) => svm_onramp_tierup_deliver_trap(),
        }
    }
}

/// Drive an opened pump session to DONE with the full driver. `sigs` supplies each function's
/// result count for TIERUP servicing. Returns the driver (for its bounce log) and the
/// `(tierups, invokes)` counters.
fn drive_full_session(m: &svm_ir::Module) -> (TierupDriver, u32, u32) {
    let mut d = TierupDriver::new();
    let (mut tierups, mut invokes) = (0u32, 0u32);
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_JIT_INVOKE => {
                invokes += 1;
                d.service_jit_invoke();
            }
            TIERUP_RUN_TIERUP => {
                tierups += 1;
                let f = svm_onramp_tierup_func() as usize;
                d.service_tierup(m.funcs[f].results.len());
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    (d, tierups, invokes)
}

/// [`drive_full_session`] for guests with no eligible leaves (every emitted entry a unit invoke).
fn drive_jit_session() -> (TierupDriver, u32) {
    let mut d = TierupDriver::new();
    let mut invokes = 0u32;
    loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_JIT_INVOKE => {
                invokes += 1;
                d.service_jit_invoke();
            }
            TIERUP_RUN_DONE => break,
            ev => panic!("unexpected pump event {ev} (status {})", svm_status()),
        }
    }
    (d, invokes)
}

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

/// The on-ramp powerbox's stdout/memory/**jit** handles (grant order incl. the conditional `Jit`
/// grant a `vm_jit_*` importer gets — `grant_onramp_caps`).
fn onramp_handles_jit() -> (i32, i32, i32) {
    let mut h = Host::new();
    let out = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    let mem = h.grant_memory();
    let _ = h.grant_address_space(0, 1 << 16);
    let jit = h.grant_jit_with_table(Some(16), 10);
    (out, mem, jit)
}

/// What the bounce helper adds, and where the unit probes the mid-invoke-grown page.
const BOUNCE_K: i64 = 1000;
const MID_GROW_PROBE: i64 = 81920 + 8;

/// #846 differential (unit→program edges + growth-mid-invoke): a **linked** unit whose
/// `call_indirect` slots split across edge kinds, run through the full driver, must match
/// `onramp_exec` — including a store into a page grown mid-invoke (only correct if the post-bounce
/// `"mapped"` fan-out admits the growth). Post-#889 the cap-calling helper's sites are outlined, so
/// **both** program slots dispatch natively (the helper emits) and the live-window bounces are its
/// wrappers 7/8 (append order: f0's vm_map/compile/invoke/write = 3–6, f1's grow/write = 7/8) —
/// the growth now happens inside wrapper 7, mid-invoke, same contract. The interp-resident-slot
/// shim bounce this test used to carry is pinned by `tierup_region_bounce_grows_and_streams`
/// (whose target hosts a fiber, the shape that stays interpreter-resident post-#889).
#[test]
fn linked_unit_bounces_and_native_edges_match_the_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, mem_h, _jit_h) = onramp_handles_jit();
    // The unit: slot 1 → the bounce helper (grows [80 KiB, 96 KiB), returns x + BOUNCE_K); a
    // store/load in the just-grown page; slot 2 → the emitted pure program leaf (×3), native.
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
    let unit = svm_text::parse_module(&unit_src).expect("parse unit");
    svm_verify::verify_module(&unit).expect("verify unit");
    let blob = svm_encode::encode_module(&unit);
    let blob_len = blob.len();
    let stores = stage_blob("b", BLOB_BASE, &blob);
    // Program: f0 = _start (grow + compile + invoke2 + stdout), f1 = the cap-calling bounce helper
    // (never emitted), f2 = the pure eligible leaf (emitted — the unit reaches it natively).
    let src = format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
import 1 "vm_jit_invoke2" (i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
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
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
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
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        (2 * PROBE + BOUNCE_K) * 3,
        "oracle: bounce (+K, grow), grown-page store/load, native ×3"
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, invokes) = drive_jit_session();
    assert!(
        invokes >= 1,
        "the linked unit must run emitted (non-vacuity)"
    );
    assert!(
        d.bounces().contains(&7) && d.bounces().contains(&8),
        "the helper's outlined wrappers must bounce (edge non-vacuity): {:?}",
        d.bounces()
    );
    assert!(
        !d.bounces().contains(&1),
        "#889: the cap-calling helper itself must emit and dispatch natively: {:?}",
        d.bounces()
    );
    assert!(
        !d.bounces().contains(&2),
        "the eligible leaf must dispatch natively, never bounce: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle (growth-mid-invoke visible post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity (bounce ordering included)"
    );
    svm_onramp_tierup_close();
}

/// #846 differential (unit→**unit** native): the guest compiles + `install`s a pure unit A, then
/// compiles unit B whose `call_indirect` reaches A's install slot — with both emitted, the edge
/// dispatches natively through the shared table (zero bounces), matching the oracle.
#[test]
fn installed_unit_edge_dispatches_natively() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h, jit_h) = onramp_handles_jit();
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
  vr = call_indirect (i64) -> (i64) vs (vx)
  vk = i64.const 100
  vsum = i64.add vr vk
  return vsum
  }
}
"#;
    let blob_a = {
        let u = svm_text::parse_module(unit_a_src).expect("parse A");
        svm_verify::verify_module(&u).expect("verify A");
        svm_encode::encode_module(&u)
    };
    let blob_b = {
        let u = svm_text::parse_module(unit_b_src).expect("parse B");
        svm_verify::verify_module(&u).expect("verify B");
        svm_encode::encode_module(&u)
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
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vca)
  vbp = i64.const {B_BASE}
  vbl = i64.const {lb}
  vcb = call.import 0 (vbp, vbl)
  vx = i64.const {X}
  vres = call.import 1 (vcb, vslot, vx)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, X + 7 + 100, "oracle: B → installed A → +100");

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, invokes) = drive_jit_session();
    assert!(invokes >= 1, "unit B must run emitted (non-vacuity)");
    assert!(
        d.bounces().is_empty(),
        "both units are emitted — the installed edge must dispatch natively: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity with the oracle"
    );
    svm_onramp_tierup_close();
}

// ---- #880: call_indirect-bearing functions tier up (B2 main module) ---------------------------

/// #880 differential (native indirect dispatch inside a tiered-up leaf): an eligible leaf whose
/// body `call_indirect`s an emitted program function — impossible to emit before the shared-table
/// main module — runs on the emitted tier with the indirect edge dispatching natively.
#[test]
fn indirect_leaf_tiers_up_natively() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, mem_h) = onramp_handles();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
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
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, PROBE + LEAF_K + 1, "oracle value");

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(
        tierups >= 1,
        "the call_indirect-bearing leaf must tier up (#880 non-vacuity)"
    );
    assert!(
        d.bounces().is_empty(),
        "the indirect edge targets an emitted function — native, never bounced: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_onramp_tierup_value(), want.value, "value parity");
    svm_onramp_tierup_close();
}

/// #1009 Mechanism 1 (regression): a guest with **more than 1024 functions** whose tier-up-eligible
/// dispatch leaf `call_indirect`s a slot **beyond the 1024-slot floor**. The emitted B2 dispatch
/// table and the interpreter's `SharedSlots` table must both size to `next_power_of_two(n_funcs)`;
/// before the fix the emitted leaf masked the index against a fixed `1 << 10` (`1050 & 1023 = 26`)
/// while the interpreter reached slot 1050 — a wrong-but-identically-typed function, so the tiered-up
/// call returned a **silently wrong value** with no trap. With the table sized to the guest both
/// tiers mask identically and the pump matches the oracle.
#[test]
fn high_index_dispatch_beyond_the_table_floor_matches_the_oracle() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h) = onramp_handles();

    // Slots: 0 = `_start`, 1 = the dispatch leaf, 2..=TARGET = identity padding, TARGET = the
    // distinctive target (returns v0 + DISTINCT). TARGET > 1024, and next_power_of_two(1051) = 2048,
    // so a fixed-1024 mask sends TARGET to `TARGET & 1023` = 26 — an *identity* function.
    const TARGET: usize = 1050;
    const INPUT: i64 = 12345;
    const DISTINCT: i64 = 777;
    assert_eq!(
        TARGET & 1023,
        26,
        "the fixed-1024 mask lands on an identity slot"
    );

    // func 0: `_start` (interp-driven — it streams) calls the dispatch leaf, stages + writes the
    // result to stdout, returns it.
    let mut fns = format!(
        r#"func () -> (i64) {{
block 0 () {{
  vin = i64.const {INPUT}
  vres = call 1 (vin)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
"#
    );
    // func 1: the dispatch leaf — `call_indirect` the high slot TARGET.
    fns.push_str(&format!(
        r#"func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs = i32.const {TARGET}
  vr = call_indirect (i64) -> (i64) vs (v0)
  return vr
  }}
}}
"#
    ));
    // funcs 2..=TARGET: identity padding, except func TARGET returns v0 + DISTINCT.
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

    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    assert!(
        m.funcs.len() > (1usize << 10),
        "the guest must exceed the 1024-slot table floor to exercise M1 (got {})",
        m.funcs.len()
    );
    let bytes = svm_encode::encode_module(&m);

    // Oracle: dispatch reaches slot TARGET → INPUT + DISTINCT.
    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        INPUT + DISTINCT,
        "oracle: the dispatch reaches slot {TARGET}, not {}",
        TARGET & 1023
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());

    // The dispatch table (emitted mask + interpreter) is sized to the guest, not the 1024 floor.
    assert!(
        (1u32 << svm_onramp_tierup_table_log2()) >= m.funcs.len() as u32,
        "the table must cover every function (fixed at 1024 before #1009 M1): 1<<{} < {}",
        svm_onramp_tierup_table_log2(),
        m.funcs.len()
    );

    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(tierups >= 1, "the dispatch leaf must tier up");
    assert!(
        d.bounces().is_empty(),
        "the indirect edge reaches an emitted function natively, never bounced: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity — the emitted high-index dispatch must reach slot {TARGET}, not {}",
        TARGET & 1023
    );
    svm_onramp_tierup_close();
}

/// #880 differential (**old→new native**): an eligible leaf `call_indirect`s an **install slot** —
/// the program reaching a guest-compiled unit's *emitted* `f0` through the shared table, the edge
/// that previously always executed the unit's bytecode.
#[test]
fn leaf_reaches_installed_unit_natively() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h, jit_h) = onramp_handles_jit();
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
    let stores = stage_blob("a", BLOB_BASE, &blob);
    let blob_len = blob.len();
    const X: i64 = 4000;
    let src = format!(
        r#"memory 16
import 0 "vm_jit_compile" (i64, i64) -> (i64)
func () -> (i64) {{
block 0 () {{
{stores}  vbp = i64.const {BLOB_BASE}
  vbl = i64.const {blob_len}
  vcode = call.import 0 (vbp, vbl)
  vjit = i32.const {jit_h}
  vslot = cap.call 11 3 (i64) -> (i64) vjit (vcode)
  vx = i64.const {X}
  vres = call 1 (vslot, vx)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
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
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        X + 7 + 100,
        "oracle: leaf → installed unit → +100"
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(tierups >= 1, "the dispatching leaf must tier up");
    assert!(
        d.bounces().is_empty(),
        "the install-slot edge reaches the unit's emitted f0 — native (old→new): {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_onramp_tierup_value(), want.value, "value parity");
    svm_onramp_tierup_close();
}

/// #880 differential (TIERUP-region bounce + growth): a tiered-up leaf's `call_indirect` lands on
/// a **shim** (the target is interpreter-resident), whose callback grows the window and streams —
/// the leaf then stores into the just-grown page, correct only through the post-bounce `"mapped"`
/// fan-out; the bounce's stdout interleaves exactly as the interpreted call would. Post-#889 a
/// merely cap-calling target would emit (its sites outline), so the target here also **hosts a
/// fiber** (`cont.new`/`resume` — out of subset by design, §22): the one shape that keeps a
/// marshallable slot interpreter-resident, which is exactly what this shim edge exists for.
#[test]
fn tierup_region_bounce_grows_and_streams() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, mem_h) = onramp_handles();
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  va = call_indirect (i64) -> (i64) vs2 (v0)
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
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
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
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 2 * X + BOUNCE_K, "oracle value");

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(tierups >= 1, "the leaf must tier up");
    assert!(
        d.bounces().contains(&2),
        "the fiber-hosting target must bounce through the slot shim: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity (mid-region growth admitted post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity (bounce ordering)");
    svm_onramp_tierup_close();
}

/// #880 differential (**run-registry fiber persistence**): a fiber created inside a TIERUP-region
/// bounce must register in the vCPU's *run-level* registry — `_start` (interpreted) resumes it
/// **after** the emitted region returned. An invoke-confined registry here would `FiberFault`
/// where the interpreter succeeds; this pins the context split in `Vcpu::bounce_call`.
#[test]
fn fiber_from_tierup_bounce_persists() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h) = onramp_handles();
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vs2 = i32.const 2
  vr = call_indirect (i64) -> (i64) vs2 (v0)
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
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(
        want.value,
        (X + 11) + (77 + 22),
        "oracle: yield from the bounce-created fiber + its completion under _start's later resume"
    );

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(opened, 0, "open must admit (status {})", svm_status());
    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(tierups >= 1, "the dispatching leaf must tier up");
    assert!(
        d.bounces().contains(&2),
        "the fiber-creating target must bounce: {:?}",
        d.bounces()
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity — the bounce-created fiber persisted into the run registry"
    );
    svm_onramp_tierup_close();
}

// ---- #888: widened cross-tier — a DIRECT Call from an emitted leaf to a cap-calling helper ------

/// What the direct-cross-tier helper adds.
const XT_K: i64 = 222;

/// #888 differential: `_start` (cap-calling, interpreter-driven) calls `f1`, a pure all-i64
/// compute leaf whose *only* non-emittable feature is a **direct `Call`** to `f2`, a `cap.call`-ing
/// helper (grows the window + streams). Before #888 `f1` cascades to the interpreter — the tier-up
/// open would refuse the guest (no eligible leaf). After #888 `f1` emits (the reactor `cross` set),
/// so the open succeeds and `f1` tiers up. Post-#889 (the pump outlines cap sites) `f2` **also**
/// emits — its two `cap.call`s become wrappers 4/5 (appended after the 3 originals; 3 is
/// `_start`'s) — so the direct cross-tier calls over the **live** window are now the wrapper
/// bounces: wrapper 4 grows `[64K, 80K)` mid-`f2`, wrapper 5 streams, then `f1` stores into the
/// just-grown page — correct only through the post-bounce `"mapped"` fan-out. Distinct from the
/// #880 `call_indirect` bounce: these are direct `env.call_interp`s, not table shims.
#[test]
fn direct_cross_tier_call_bounces_over_the_live_window() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, mem_h) = onramp_handles();
    // f0 = _start (cap-calls → interp-driven); f1 = the eligible leaf (compute + direct call to f2 +
    // a store into f2's grown page); f2 = the cap-calling helper (grow + stream), out of subset,
    // reached as a direct cross-tier call.
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {LEAF_K}
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
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vout = i32.const {out_h}
  vzero = i64.const 0
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vzero, vlen8)
  vk = i64.const {XT_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, 7 + LEAF_K + XT_K, "oracle value");

    // The open itself is the #888 pin: pre-#888 f1 cascades (no eligible leaf) → UNSUPPORTED.
    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "#888: the leaf with a direct cross-tier call must now be eligible (status {})",
        svm_status()
    );
    let (d, tierups, _invokes) = drive_full_session(&m);
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
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(
        svm_onramp_tierup_value(),
        want.value,
        "value parity (direct cross-tier over the live window; mid-call growth admitted post-bounce)"
    );
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity (bounce ordering included)"
    );
    svm_onramp_tierup_close();
}

// ---- #926 slice 1: no static concurrency gate — the runtime event is the real gate ------------

/// #926 slice 1: a guest whose concurrency op is **linked but dead** (unreachable) — jaclrt's
/// shape (#839): scheduler/GC futex/thread ops that never run with `POOL_WORKERS=1`. Pre-#926 the
/// whole-module `any(uses_threads || uses_futex)` gate refused it at open; now it is admitted, its
/// dead op is never reached, and its pure leaf tiers up — parity with `onramp_exec`.
#[test]
fn dead_concurrency_op_is_admitted_and_tiers_up() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, mem_h) = onramp_handles();
    // f0 = _start (grow + call the leaf + stream); f1 = the pure all-i64 leaf; f2 = DEAD, never
    // called, carrying a futex op (`atomic.wait`) — the whole-module refusal used to catch it.
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vprobe = i64.const {PROBE}
  vres = call 1 (vprobe)
  vsl = i64.const {SLOT}
  i64.store vsl vres
  vout = i32.const {out_h}
  vlen8 = i64.const 8
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  return vres
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
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    assert_eq!(want.value, PROBE + LEAF_K, "oracle value");

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "#926: a dead (unreachable) concurrency op must no longer refuse the pump (status {})",
        svm_status()
    );
    let (_d, tierups, _invokes) = drive_full_session(&m);
    assert!(
        tierups >= 1,
        "the pure leaf must tier up beside the dead op"
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_onramp_tierup_value(), want.value, "value parity");
    // SAFETY: capture slots staged at DONE; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(got_out, want.stdout, "stdout parity");
    svm_onramp_tierup_close();
}

/// #926 slice 1: a guest that **actually reaches** a concurrency op (an `atomic.notify` in
/// `_start`'s path, after a leaf tiers up) is admitted, tiers up the leaf, then **declines cleanly**
/// at the op — `TIERUP_RUN_TRAP`, not a crash or hang — so the page re-runs on the interpreter
/// (INVARIANT 9). The runtime event, not a static scan, is the gate.
#[test]
fn reachable_concurrency_op_declines_cleanly() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h) = onramp_handles();
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
  vaddr = i64.const 0
  vcnt = i32.const 1
  vn = atomic.notify vaddr vcnt
  vn64 = i64.extend_i32_s vn
  return vn64
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  vk = i64.const {LEAF_K}
  vsum = i64.add v0 vk
  return vsum
  }}
}}
export 0 func "_start" 0
"#
    );
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "admitted (no static concurrency gate) — status {}",
        svm_status()
    );
    let mut d = TierupDriver::new();
    let mut tierups = 0u32;
    let ev = loop {
        match svm_onramp_tierup_run() {
            TIERUP_RUN_TIERUP => {
                tierups += 1;
                let f = svm_onramp_tierup_func() as usize;
                d.service_tierup(m.funcs[f].results.len());
            }
            other => break other,
        }
    };
    assert_eq!(
        ev, TIERUP_RUN_TRAP,
        "the reachable atomic.notify must decline cleanly to TRAP (→ interpreter), not DONE/crash"
    );
    assert!(
        tierups >= 1,
        "the leaf tiered up before the concurrency op was reached"
    );
    assert_eq!(
        svm_status(),
        STATUS_TRAP,
        "status is TRAP (the page re-runs on the interpreter)"
    );
    svm_onramp_tierup_close();
}

/// What each hot-loop iteration adds (distinct from every other K so a cross-wire shows).
const HOT_K: i64 = 77;
/// Where the hot loop stages each iteration's running total before streaming it.
const SLOT2: i64 = 2064;
/// Loop iterations — also the exact expected wrapper-bounce count.
const HOT_N: i64 = 4;

/// #889 differential (the card shape): `f1` is a hot loop — compute plus **one inline stdout
/// `cap.call` per iteration**. Pre-#889 the inline cap site pins the whole function to the
/// interpreter (the open even refuses: nothing eligible). Post-#889 the pump outlines the site, so
/// `f1` emits and tiers up; each iteration's cap write bounces to the outlined wrapper (index 3 —
/// wrapper 2 is `_start`'s write, never bounced: the root runs interpreted). Pinned: parity with
/// the bytecode oracle (status/value/stdout — the per-iteration writes interleave identically),
/// the tier-up non-vacuity, and bounce-count == the loop's cap-site executions, all to the wrapper.
#[test]
fn hot_loop_with_inline_cap_write_emits_and_bounces_per_iteration() {
    let _g = FFI_LOCK.lock().unwrap();
    let (out_h, _mem_h) = onramp_handles();
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl, vlen8)
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
  vw = cap.call 0 1 (i64, i64) -> (i64) vout (vsl2, vlen8)
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
    let m = svm_text::parse_module(&src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let bytes = svm_encode::encode_module(&m);

    let want = onramp_exec(&m, b"");
    assert_eq!(want.status, STATUS_OK, "oracle sanity");
    let expect = HOT_K * (1..=HOT_N).sum::<i64>();
    assert_eq!(want.value, expect, "oracle value");

    // Pre-#889 this open REFUSED (the inline cap site cascades f1, and _start holds its own) —
    // admitting it at all is half the pin.
    let opened = svm_onramp_tierup_open(bytes.as_ptr(), bytes.len(), core::ptr::null(), 0, 0);
    assert_eq!(
        opened,
        0,
        "#889: the cap-bearing hot loop must be admitted (status {})",
        svm_status()
    );
    let (d, tierups, _invokes) = drive_full_session(&m);
    assert!(tierups >= 1, "the hot loop must tier up (non-vacuity)");
    assert_eq!(
        d.bounces(),
        &vec![3u32; HOT_N as usize][..],
        "each iteration's cap write bounces to f1's outlined wrapper, nothing else bounces"
    );
    assert_eq!(svm_status(), want.status, "status parity with the oracle");
    assert_eq!(svm_onramp_tierup_value(), want.value, "value parity");
    // SAFETY: capture slots staged by the DONE arm; this thread is the only accessor (FFI_LOCK).
    let got_out =
        unsafe { std::slice::from_raw_parts(svm_stdout_ptr(), svm_stdout_len()) }.to_vec();
    assert_eq!(
        got_out, want.stdout,
        "stdout parity — the per-iteration writes interleave exactly as interpreted"
    );
    svm_onramp_tierup_close();
}
