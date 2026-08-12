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

use std::sync::Mutex;
use svm_browser::{
    onramp_exec, svm_onramp_tierup_argv_len, svm_onramp_tierup_argv_ptr, svm_onramp_tierup_close,
    svm_onramp_tierup_deliver, svm_onramp_tierup_deliver_jit, svm_onramp_tierup_deliver_jit_trap,
    svm_onramp_tierup_deliver_trap, svm_onramp_tierup_func, svm_onramp_tierup_jit_param_types_ptr,
    svm_onramp_tierup_jit_result_types_len, svm_onramp_tierup_jit_wasm_len,
    svm_onramp_tierup_jit_wasm_ptr, svm_onramp_tierup_mapped, svm_onramp_tierup_open,
    svm_onramp_tierup_run, svm_onramp_tierup_value, svm_onramp_tierup_wasm_len,
    svm_onramp_tierup_wasm_ptr, svm_onramp_tierup_win_len, svm_onramp_tierup_win_ptr,
    svm_run_value, svm_status, svm_stdout_len, svm_stdout_ptr, STATUS_OK, STATUS_UNSUPPORTED,
    TIERUP_RUN_DONE, TIERUP_RUN_JIT_INVOKE, TIERUP_RUN_TIERUP, TIERUP_RUN_TRAP,
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
    let (_, mem_h) = onramp_handles();
    let src = format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vas = i32.const {mem_h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
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
        opened, -STATUS_UNSUPPORTED,
        "no eligible leaf → refuse the session"
    );
    assert_eq!(svm_status(), STATUS_UNSUPPORTED);
    // A refused open leaves no session: the pump reports TRAP immediately, not a phantom run.
    assert_eq!(svm_onramp_tierup_run(), TIERUP_RUN_TRAP);
    svm_onramp_tierup_close();
}
