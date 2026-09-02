//! **#1151 — a page-op op-13 child on the browser's JS-orchestrated op-13 loop** (`temen_op13jit_*`,
//! #1025 Path 1). The loop runs each separate-module child on the single-shot JIT tier
//! (`JitOnrampRun`, `compile_jit`); that emit **declines** a module that `unmap`s/`protect`s its own
//! pages (the mask-only emit can't honor page state), and the loop used to answer the decline with
//! `OP13JIT_TRAP` — the driver died for want of an emit, where the interpreter runs the child fine.
//! Now the decline runs the child on the interpreter over the same carve and marshaled powerbox, inside
//! the step, and banks its result for the driver's `join` — fail-closed, byte-identical.
//!
//! The child: `f0(sp, as)` (child-entry shape) = `40 + f1(as) + K`, where `f1` resolves the marshaled
//! `"fs"` (the counter → 1), then `protect`s the page holding the data byte "K" = 75 read-only; `f0`
//! reads K back through the now-`Ro` page (a read passes) → 116. The trap twin stores on that page
//! after the protect: the store faults, the driver's `join` propagates the trap (`OP13JIT_TRAP`) — but
//! only **after** the child ran (the counter ticked), which is what distinguishes it from the old
//! decline-trap. The built-in emittable child still yields `OP13JIT_CHILD` (the emitted path is not
//! stolen by the fallback).

use std::sync::Mutex;

use temen_browser::{
    temen_op13jit_close, temen_op13jit_counter, temen_op13jit_open, temen_op13jit_open_child,
    temen_op13jit_result, temen_op13jit_step, OP13JIT_CHILD, OP13JIT_DONE, OP13JIT_TRAP,
};
use temen_interp::host_page_size;

// The op-13 loop state is process-global (`OP13_JIT`, `JIT_RUN`): serialize the tests.
static LOCK: Mutex<()> = Mutex::new(());

/// The page-op child (`memory 15` — the mini driver's 32-KiB buddy-half carve). "K" sits at the start
/// of page 1 (whatever the host page size), the page `f1` protects.
fn child(store_to_ro: bool) -> Vec<u8> {
    let page = host_page_size();
    let store = if store_to_ro {
        format!(
            "  vt = i64.const {}\n  vseven = i64.const 7\n  i64.store vt vseven\n",
            page + 8
        )
    } else {
        String::new()
    };
    let src = format!(
        r#"memory 15
data {page} "K"
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, vas: i64) {{
  vc = call 1 (vas)
  vq = i64.const {page}
  vk = i64.load8_u vq
{store}  v40 = i64.const 40
  vs = i64.add v40 vc
  vr = i64.add vs vk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vas: i64) {{
  vname = i64.const 29542
  vzero = i64.const 0
  i64.store vzero vname
  vp0 = i64.const 0
  vl2 = i64.const 2
  vh = self.resolve vp0 vl2
  vc = call.cap 13 0 (i64) -> (i64) vh (vp0)
  vas32 = i32.wrap_i64 vas
  vq = i64.const {page}
  vlen = i64.const {page}
  vro = i32.const 1
  vpr = call.cap 5 2 (i64, i64, i32) -> (i64) vas32 (vq, vlen, vro)
  vsum = i64.add vc vpr
  return vsum
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    // The single-shot emit declines this child: it manages its own pages.
    assert!(
        temen_wasm_jit::module_uses_unmap_protect(&m),
        "the child reaches protect"
    );
    temen_encode::encode_module(&m)
}

fn open(bytes: &[u8]) {
    // SAFETY: a live byte slice for the duration of the call.
    let st = unsafe { temen_op13jit_open_child(bytes.as_ptr(), bytes.len()) };
    assert_eq!(st, 0, "op-13 loop opens over the page-op child");
}

#[test]
fn page_op_child_runs_on_the_interpreter_inline() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(false));
    // No `OP13JIT_CHILD` yield: the declined child ran on the interpreter inside the step, and the
    // driver's join was serviced inline — the loop completes in one step.
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
fn page_op_child_store_to_protected_page_traps() {
    let _g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    open(&child(true));
    assert_eq!(
        temen_op13jit_step(),
        OP13JIT_TRAP,
        "the child's store to its protected page faults; the join propagates it"
    );
    assert_eq!(
        temen_op13jit_counter(),
        1,
        "the child ran up to the fault (not the driver refusing the spawn)"
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
        temen_op13jit_counter(),
        0,
        "nothing ran yet — the child awaits its drive"
    );
    temen_op13jit_close();
}
