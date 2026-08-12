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
    svm_onramp_tierup_deliver, svm_onramp_tierup_deliver_trap, svm_onramp_tierup_func,
    svm_onramp_tierup_mapped, svm_onramp_tierup_open, svm_onramp_tierup_run,
    svm_onramp_tierup_value, svm_onramp_tierup_wasm_len, svm_onramp_tierup_wasm_ptr,
    svm_onramp_tierup_win_len, svm_onramp_tierup_win_ptr, svm_run_value, svm_status,
    svm_stdout_len, svm_stdout_ptr, STATUS_OK, STATUS_UNSUPPORTED, TIERUP_RUN_DONE,
    TIERUP_RUN_TIERUP, TIERUP_RUN_TRAP,
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
    let argv = unsafe {
        std::slice::from_raw_parts(svm_onramp_tierup_argv_ptr(), svm_onramp_tierup_argv_len())
    };
    let win_len = svm_onramp_tierup_win_len();
    let win_ptr = svm_onramp_tierup_win_ptr() as *mut u8;
    let func = svm_onramp_tierup_func();
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
        .get_func(&store, &format!("f{func}"))
        .unwrap_or_else(|| panic!("f{func} not exported"));

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
