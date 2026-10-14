//! **Warm-runtime snapshot of a `vm_map`-growing guest** (#816) — end-to-end through the real
//! `temen_warm_open`/`temen_warm_eval` FFI: a guest whose `warmup` grows the window past its declared
//! size must restore per eval to the same mapped geometry (the committed-extent round-trip), with
//! fresh-per-Run isolation intact. Before #816 this shape required the `WARM_MAPPED_LOG2` over-size
//! workaround (declared ≥ heap) — this guest deliberately declares a 64-KiB window and heaps past it.
//!
//! The negative half of the contract (an unseeded fresh `Mem` faults on the grown page) is pinned
//! at the engine seam in `temen-interp/tests/run_over_grown.rs`.

use std::sync::Mutex;
use temen_browser::{temen_status, temen_warm_close, temen_warm_eval, temen_warm_open, STATUS_OK};
use temen_interp::{Host, StreamRole};

/// The warm session is one process-global static: serialize the tests in this binary across it.
static WARM_LOCK: Mutex<()> = Mutex::new(());

/// Marker the warmup plants in the grown page; scratch cell the evals write (must NOT persist).
const MARKER: i64 = 424242;
const MARKER_ADDR: u64 = 65552; // 64 KiB + 16 — inside the vm_map-grown page
const SCRATCH_ADDR: u64 = 65560;

/// The memory (whole-window AddressSpace) handle `grant_onramp_caps` mints — replicated from its
/// grant order (stdout, stdin, exit, memory, …) against a fresh `Host`, so the guest text can
/// `call.cap` it directly (the on-ramp powerbox mints deterministic handles per session).
fn memory_handle() -> i32 {
    let mut h = Host::new();
    let _ = h.grant_stream(StreamRole::Out);
    let _ = h.grant_stream(StreamRole::In);
    let _ = h.grant_exit();
    h.grant_memory()
}

/// The two-phase growing guest. `warmup(sp)`: `vm_map` a 16-KiB page above the declared 64-KiB
/// window (whole page on every host page size), plant the marker there, and advance the on-ramp
/// brk word so the image capture covers the grown bytes. `eval_run(sp)`: read the marker plus the
/// scratch cell (fresh-per-Run: must be 0 every eval), write the scratch, and return
/// `marker + scratch`.
fn guest_text() -> String {
    let h = memory_handle();
    // The on-ramp brk word sits one guard up on the #1094 marked layout (`warm_read_brk`:
    // `scratch + POWERBOX_HEAP_BRK`, scratch = the module guard base) — where the warm host seeds it.
    let brk = temen_ir::POWERBOX_NULL_GUARD + temen_ir::POWERBOX_HEAP_BRK;
    format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (vsp: i64) {{
  vas = i32.const {h}
  voff = i64.const 65536
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vaddr = i64.const {MARKER_ADDR}
  vmark = i64.const {MARKER}
  i64.store vaddr vmark
  vbrkaddr = i64.const {brk}
  vbrk = i64.const 81920
  i64.store vbrkaddr vbrk
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vsp: i64) {{
  vaddr = i64.const {MARKER_ADDR}
  vm = i64.load vaddr
  vsaddr = i64.const {SCRATCH_ADDR}
  vs = i64.load vsaddr
  vseven = i64.const 7
  i64.store vsaddr vseven
  vsum = i64.add vm vs
  return vsum
  }}
}}
export 0 func "warmup" 0
export 1 func "eval_run" 1
"#
    )
}

#[test]
fn warm_session_restores_a_grown_heap_across_evals() {
    let _g = WARM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let mut m = temen_text::parse_module(&guest_text()).expect("parse");
    // Belt-and-braces: the text `export` lines must have resolved (temen_warm_open requires both).
    assert!(m.resolve_export("warmup").is_some() && m.resolve_export("eval_run").is_some());
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    let live = temen_warm_open(bytes.as_ptr(), bytes.len());
    assert!(
        live > 0,
        "warm open must succeed for a growing guest (status {})",
        temen_status()
    );
    // The image must cover the grown marker (brk was advanced past it by warmup).
    assert!(
        live as u64 >= MARKER_ADDR + 8,
        "image covers the grown page (live {live})"
    );

    // Eval 1: marker restored from the grown page, scratch fresh (0) → MARKER + 0.
    let v1 = temen_warm_eval(core::ptr::null(), 0);
    assert_eq!(temen_status(), 0, "eval 1 status");
    assert_eq!(
        v1, MARKER,
        "eval 1 must read the vm_map-grown marker over a restored extent"
    );

    // Eval 2: identical — the restore re-establishes the SAME committed extent and the scratch the
    // prior eval wrote is wiped (fresh-per-Run isolation, INVARIANT #6).
    let v2 = temen_warm_eval(core::ptr::null(), 0);
    assert_eq!(temen_status(), 0, "eval 2 status");
    assert_eq!(
        v2, MARKER,
        "eval 2 must see byte-identical warm state (no scratch leak)"
    );

    temen_warm_close();
    // Ensure the module the test built stays alive to here (the FFI copied what it needed).
    m.exports.clear();
}

/// The #1734 guest. `warmup` `vm_map`s 16 KiB at [`TAIL`], above its brk, so the warm image's page
/// map holds a committed page past the captured bytes. `eval_run` returns the `i64` it finds at
/// `TAIL + 16`, then stores [`MARKER`] there.
fn tail_guest_text() -> String {
    let h = memory_handle();
    format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (vsp: i64) {{
  vas = i32.const {h}
  vt = i64.const {TAIL}
  vlen = i64.const 16384
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (vt, vlen, vprot)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vsp: i64) {{
  vaddr = i64.const {cell}
  vv = i64.load vaddr
  vmark = i64.const {MARKER}
  i64.store vaddr vmark
  return vv
  }}
}}
export 0 func "warmup" 0
export 1 func "eval_run" 1
"#,
        cell = TAIL + 16,
    )
}

/// Where the #1734 guest's warmup commits its page: above the brk.
const TAIL: i64 = 1 << 20;

/// #1734 — an eval leaves nothing for the next one. An eval may write a page warmup left committed
/// above its brk, and the next eval may read it without mapping it, so every restore zeroes it.
/// The restore is one body for every warm tier (`restore_warm_window`).
#[test]
fn an_eval_leaves_nothing_above_the_brk_for_the_next() {
    let _g = WARM_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let m = temen_text::parse_module(&tail_guest_text()).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);
    assert!(temen_warm_open(bytes.as_ptr(), bytes.len()) > 0, "open");
    for eval in 1..=2 {
        assert_eq!(
            (temen_warm_eval(core::ptr::null(), 0), temen_status()),
            (0, STATUS_OK),
            "eval {eval} must find the page warmup committed zeroed"
        );
    }
    temen_warm_close();
}
