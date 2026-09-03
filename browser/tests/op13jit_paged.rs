//! **A page-op op-13 child on the browser's JS-orchestrated op-13 loop** (`temen_op13jit_*`, #1025
//! Path 1). The loop runs each separate-module child on the single-shot JIT tier (`JitOnrampRun`).
//! #1199 made a child the emit *declines* run on the interpreter inside the step instead of trapping
//! the driver; #1201 makes a page-op child not decline at all: it emits **paged**, is staged in
//! `JIT_RUN` for the JS driver (`OP13JIT_CHILD`), and its `env.call_interp` bounces rebuild the
//! page-state table the emitted accesses consult. The emitted execution itself is pinned on wasmi by
//! `jit_paged_onramp.rs` (the same run type over a wasmi memory); here the loop's own seam is pinned:
//! the child is staged emitted and paged, with its **real** starter `AddressSpace` handle as the second
//! entry slot (it was `0` before #1201), and the staged run's bounce services the leaf — the marshaled
//! `fs` ticks, the `protect` lands in the table (page 1 → `Ro`), and `"mapped"` is the table's coverage.
//!
//! The child: `f0(sp, as)` = `40 + f1(as) + K`; `f1` resolves the marshaled `"fs"` (the counter → 1),
//! then `protect`s the page holding "K" = 75 read-only. The built-in emittable child still yields
//! `OP13JIT_CHILD`; a child the emit genuinely declines (a `SharedRegion` op anywhere in the module
//! gates the paged emit off) runs on the interpreter inline — the #1199 fallback, still there.

use std::sync::Mutex;

use temen_browser::{
    temen_onramp_jit_run_call_interp, temen_onramp_jit_run_mapped,
    temen_onramp_jit_run_pagestate_len, temen_onramp_jit_run_pagestate_ptr,
    temen_onramp_jit_run_slot, temen_onramp_jit_run_slot_count, temen_op13jit_close,
    temen_op13jit_counter, temen_op13jit_open, temen_op13jit_open_child, temen_op13jit_result,
    temen_op13jit_step, OP13JIT_CHILD, OP13JIT_DONE,
};
use temen_interp::host_page_size;

// The op-13 loop state is process-global (`OP13_JIT`, `JIT_RUN`): serialize the tests.
static LOCK: Mutex<()> = Mutex::new(());

/// The page-op child (`memory 15` — the mini driver's 32-KiB buddy-half carve). "K" sits at 16 KiB —
/// just above the NULL guard the single-shot bounce seeds, page-aligned on a 4 KiB or 16 KiB host —
/// on the page `f1` protects (after storing the `fs` name on it). `region_op` appends an
/// **unreachable** function with a §13 `SharedRegion` `map` (iface 4 op 0): the paged emit gates on
/// it anywhere in the module, so the child declines and runs on the interpreter inline instead.
fn child(region_op: bool) -> Vec<u8> {
    let page = host_page_size();
    let extra = if region_op {
        "func () -> (i64) {\nblock 0 () {\n  vh = i32.const 0\n  va = i64.const 0\n  vr = call.cap 4 0 (i64, i64) -> (i64) vh (va, va)\n  return vr\n  }\n}\n"
    } else {
        ""
    };
    let src = format!(
        r#"memory 15
data 16384 "K"
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vas: i64) {{
  vc = call 1 (vas)
  vq = i64.const 16384
  vk = i64.load8_u vq
  v40 = i64.const 40
  vs = i64.add v40 vc
  vr = i64.add vs vk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vas: i64) {{
  vname = i64.const 29542
  vnp = i64.const 16392
  i64.store vnp vname
  vl2 = i64.const 2
  vh = self.resolve vnp vl2
  vc = call.cap 13 0 (i64) -> (i64) vh (vnp)
  vas32 = i32.wrap_i64 vas
  vq = i64.const 16384
  vlen = i64.const {page}
  vro = i32.const 1
  vpr = call.cap 5 2 (i64, i64, i32) -> (i64) vas32 (vq, vlen, vro)
  vsum = i64.add vc vpr
  return vsum
  }}
}}
{extra}"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    assert!(
        temen_wasm_jit::module_uses_unmap_protect(&m),
        "the child reaches protect"
    );
    temen_encode::encode_module(&m)
}

fn open(bytes: &[u8]) {
    // SAFETY: a live byte slice for the duration of the call.
    let st = unsafe { temen_op13jit_open_child(bytes.as_ptr(), bytes.len()) };
    assert_eq!(st, 0, "op-13 loop opens over the child");
}

#[test]
fn page_op_child_is_staged_emitted_and_paged_with_its_real_handles() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(false));
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_CHILD,
        "the page-op child emits (paged) and is staged for the JS driver"
    );
    // The staged run is paged: a table over the child's declared window, `"mapped"` = its coverage.
    let page = host_page_size();
    let len = temen_onramp_jit_run_pagestate_len();
    assert_eq!(len as u64 * page, 1 << 15, "table over the 32-KiB carve");
    assert_eq!(temen_onramp_jit_run_mapped(), len as u64 * page);
    // The entry slots: `[Instantiator, AddressSpace]` — the real handles the interpreter passes.
    assert_eq!(temen_onramp_jit_run_slot_count(), 2);
    let cas = temen_onramp_jit_run_slot(1);
    assert_ne!(
        cas, 0,
        "the starter AddressSpace handle reaches the emitted entry"
    );
    // Service the leaf's bounce the way the emitted `f0` would (`env.call_interp(1, args)`): the
    // marshaled fs ticks, the protect lands, and the table is rebuilt from the live map.
    let mut scratch = [0u8; 16];
    scratch[..8].copy_from_slice(&(cas as u64).to_le_bytes());
    assert_eq!(
        temen_onramp_jit_run_call_interp(1, scratch.as_mut_ptr()),
        0,
        "the leaf runs over the marshaled host + carve"
    );
    let ret = i64::from_le_bytes(scratch[..8].try_into().unwrap());
    assert_eq!(ret, 1, "fs() = 1, protect = 0");
    assert_eq!(temen_op13jit_counter(), 1);
    // SAFETY: the table is live until the next bounce (none — the loop is closed below).
    let table = unsafe {
        std::slice::from_raw_parts(
            temen_onramp_jit_run_pagestate_ptr(),
            temen_onramp_jit_run_pagestate_len(),
        )
    };
    assert_eq!(
        table[(16384 / page) as usize],
        2,
        "the K page is read-only after the bounce"
    );
    assert_eq!(table[0], 0, "the NULL guard page is unmapped");
    assert_eq!(
        temen_onramp_jit_run_mapped(),
        table.len() as u64 * page,
        "mapped is the table's coverage"
    );
    temen_op13jit_close();
}

#[test]
fn declined_child_runs_on_the_interpreter_inline() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(true));
    // No `OP13JIT_CHILD` yield: the declined child ran on the interpreter inside the step, and the
    // driver's join was serviced inline — the loop completes in one step (the #1199 fallback).
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_DONE,
        "driver runs to completion"
    );
    assert_eq!(
        temen_op13jit_result(),
        40 + 1 + 75,
        "40 + fs() + K through the Ro page"
    );
    assert_eq!(
        temen_op13jit_counter(),
        1,
        "the marshaled fs ran once inside the child"
    );
    temen_op13jit_close();
}

#[test]
fn emittable_child_still_yields_to_the_emitted_tier() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert_eq!(temen_op13jit_open(), 0);
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_CHILD,
        "the built-in child emits and is staged for the JS driver"
    );
    assert_eq!(
        temen_onramp_jit_run_pagestate_len(),
        0,
        "a page-op-free child emits mask-only (no table)"
    );
    assert_eq!(
        temen_op13jit_counter(),
        0,
        "nothing ran yet — the child awaits its drive"
    );
    temen_op13jit_close();
}
