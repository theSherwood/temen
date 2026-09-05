//! #816 item 5 — **parallel-driver tier-up for §14 confined children**: the `temen_par_*` FFI's
//! former "plain compute paths only" gate is lifted, so a same-module confined child's eligible
//! leaves tier up over the child's OWN carve. This is the first native harness for the par FFI at
//! all (it was real-browser-only): the test plays `par.js` + `worker.js` single-threaded — it
//! services `PAR_INSTANTIATE` by building the child vCPU itself (`temen_par_child_confined`) and
//! driving it to completion before delivering the join, and services each `PAR_TIERUP` the way the
//! Worker's emitted region would: write the leaf's store through the **serving vCPU's window**
//! (the root window for the root's event, the carve for the child's) and deliver the computed
//! results. The per-vCPU routing pins are direct: the child event's `ev_b` (the `"mapped"` value)
//! must be the CHILD's carve size, not the root window's, and each leaf's store must land in its
//! own task's window (the root reads the child's marker back through the carve offset).
//! Differential against the same guest on the cooperative interpreter (no bitmap).
//!
//! The emitted-wasm execution half over a confined carve is pinned by the coop browser gate
//! (`coop_tierup_driver.rs::coop_tierup_serves_a_confined_child_over_its_own_carve`); this file
//! pins the parallel FFI's event plumbing, which no other native test reaches.

use temen_browser::{
    temen_par_child_confined, temen_par_compile, temen_par_deliver_handle, temen_par_deliver_join,
    temen_par_deliver_tierup, temen_par_enable_jit, temen_par_enable_jit_paged, temen_par_ev_a,
    temen_par_ev_b, temen_par_ev_c, temen_par_ev_d, temen_par_free, temen_par_powerbox_inst,
    temen_par_root, temen_par_run, temen_par_tierup_argv_len, temen_par_tierup_argv_ptr,
    temen_par_tierup_pagestate_len, temen_par_tierup_pagestate_ptr, PAR_DONE, PAR_INSTANTIATE,
    PAR_JOIN, PAR_TIERUP,
};
use temen_interp::{bytecode, host_page_size, Host, Value};

const FUEL: u64 = 10_000_000;

// #1182 — serialize the tests that drive the `temen_par_*` codegen path. The emitted-JIT stash
// (`WASMJIT`, `PAR_JIT_ELIGIBLE`, `PAR_JIT_PAGED`) and its once-per-run memoization
// (`PAR_RUN_GEN` → `TIERUP_DONE_GEN`) are process-GLOBAL and single-run by design: in production one
// page runs one program, and the page-side publisher bumps the generation serially before any Worker
// is alive (see `CodegenGuard` in `browser/src/lib.rs`). `cargo test` breaks that assumption by
// running these two tests concurrently: their `powerbox_inst` gen-bumps and `enable_jit*` emits
// interleave, so the paged test can observe `TIERUP_DONE_GEN == generation` already set by the
// sibling's NON-paged emit and early-return with `PAR_JIT_PAGED` still false — then its child either
// tiers up with no pagestate table (`plen == 0`) or, on a module mismatch, never tiers up at all.
// Holding this lock across each test body restores the serial single-run contract the globals assume.
// Poison-tolerant (`into_inner`) so one test's panic still lets the other run and report on its own.
static JIT_STATE_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Window-relative offset the leaf writes its marker to. It must be **above the #1094 unconditional
/// NULL guard** (`[0, POWERBOX_NULL_GUARD)` = `[0, 16 KiB)` faults on any guest access) yet still
/// inside the child's carve — so the same offset is valid in both the root window and the carve.
/// That forces the carve to exceed the guard: a sub-16-KiB carve would have no writable byte the
/// root's guarded window also admits. `16384` is the first writable byte above the guard.
const MARKER_OFF: u64 = 16384;
/// The child's carve: 32 KiB (`> POWERBOX_NULL_GUARD`) at 64 KiB in the 128 KiB root window.
const CARVE_OFF: u64 = 65536;
const CARVE_LOG2: u32 = 15;

/// The guest. f0 (root; arg = its granted `Instantiator` handle): §14-instantiates a same-module
/// confined child at f1 (32 KiB carve at 64 KiB), joins it, calls the eligible leaf f2 itself with
/// 3, reads back both leaf markers — its own at `[MARKER_OFF]` and the child's at
/// `[CARVE_OFF + MARKER_OFF]` (the carve interior, visible through the parent window) — and sums.
/// f1 (child entry): calls the leaf with 5. f2 (the leaf `f(x) = x*3 + 7`): stores its result at
/// window-relative `[MARKER_OFF]` — the store that must land in each CALLER's own window (root vs
/// carve), the routing pin. Total: child f(5)=22 + local f(3)=16 + root marker 16 + child marker 22 = 76.
const SRC: &str = r#"
memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 15
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  v3 = i64.const 3
  vlocal = call 2 (v3)
  vma = i64.const 16384
  vm0 = i64.load vma
  vca = i64.const 81920
  vm1 = i64.load vca
  vs1 = i64.add vj vlocal
  vs2 = i64.add vs1 vm0
  vs3 = i64.add vs2 vm1
  return vs3
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v5 = i64.const 5
  vr = call 2 (v5)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (vx: i64) {
  v3 = i64.const 3
  vm = i64.mul vx v3
  v7 = i64.const 7
  va = i64.add vm v7
  vaddr = i64.const 16384
  i64.store vaddr va
  return va
  }
}
"#;

/// Service one `PAR_TIERUP` the way the Worker's emitted `f{func}` would, over the serving vCPU's
/// window `[win, win + len)`: assert the event's `"mapped"` operand is that window's own extent
/// (the per-vCPU routing pin), emulate the leaf's effect (store the result at window-relative 8),
/// and deliver the computed result. Only the leaf (f2) is ever eligible here.
fn service_tierup(v: *mut temen_browser::ParVcpu, win: *mut u8, len: u64) {
    assert_eq!(temen_par_ev_a(v), 2, "only the leaf is eligible");
    assert_eq!(
        temen_par_ev_b(v),
        len as i64,
        "the event's mapped bound must be the serving vCPU's OWN window extent (#816 item 5)"
    );
    // SAFETY: the paused vCPU is parked inside the event; the argv stash is stable until deliver.
    let argv = unsafe {
        std::slice::from_raw_parts(temen_par_tierup_argv_ptr(v), temen_par_tierup_argv_len(v))
    };
    assert_eq!(argv.len(), 1);
    let r = argv[0] * 3 + 7;
    // The emitted leaf's store, emulated over the serving window: the marker at window-relative
    // `[MARKER_OFF]` — the write that must land in each caller's OWN window (root vs carve).
    // SAFETY: the paused vCPU is parked; `[win, win+len)` is exclusively ours until deliver, and
    // `MARKER_OFF + 8 <= len` (the guard-clearing offset fits both the root window and the carve).
    unsafe {
        std::ptr::copy_nonoverlapping(r.to_le_bytes().as_ptr(), win.add(MARKER_OFF as usize), 8);
    }
    temen_par_deliver_tierup(v, [r].as_ptr(), 1);
}

#[test]
fn par_confined_child_tiers_up_over_its_own_carve() {
    let _jit = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // #1182 — serial single-run
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // Oracle: the same guest + grant on the cooperative interpreter (no bitmap) — in-engine §14
    // spawn/join, every leaf interpreted in its caller's window.
    let want = {
        let mut host = Host::new();
        let inst = host.grant_instantiator(0, 1 << 17);
        let mut run = bytecode::CoopRun::new(&m, 0, &[Value::I32(inst)], FUEL, host, None)
            .expect("supported")
            .expect("entry in range");
        match run.run() {
            bytecode::CoopEvent::Done(vals) => match vals.first() {
                Some(Value::I64(x)) => *x,
                other => panic!("non-i64 oracle result {other:?}"),
            },
            bytecode::CoopEvent::Trapped(t) => panic!("oracle trapped: {t:?}"),
            other => panic!(
                "oracle did not run to completion: {:?}",
                core::mem::discriminant(&other)
            ),
        }
    };
    assert_eq!(want, 76, "oracle value");

    // The parallel drive, this test playing par.js + worker.js single-threaded.
    assert_eq!(
        temen_par_powerbox_inst(1 << 17, core::ptr::null(), 0, 0),
        1,
        "publish the §14 run recipe"
    );
    assert_eq!(
        temen_par_enable_jit(bytes.as_ptr(), bytes.len()),
        1,
        "the leaf must be tier-up eligible"
    );
    let prog = temen_par_compile(bytes.as_ptr(), bytes.len());
    assert!(!prog.is_null(), "program compiles");
    let mut win = vec![0u8; 1 << 17].into_boxed_slice();
    let win_ptr = win.as_mut_ptr();

    let root = temen_par_root(prog, win_ptr, 1 << 17, 0);
    assert!(!root.is_null(), "root vCPU builds");

    let mut tierups = 0u32;
    let mut child_value: Option<i64> = None;
    let result = loop {
        match temen_par_run(root) {
            PAR_DONE => break temen_par_ev_a(root),
            PAR_TIERUP => {
                tierups += 1;
                assert!(tierups < 10, "runaway tier-ups");
                // The root's event serves over the full root window.
                service_tierup(root, win_ptr, 1 << 17);
            }
            PAR_INSTANTIATE => {
                // (module << 32) | entry, the carve offset, its size log2, the child's fuel —
                // shuttled verbatim into the child constructor, exactly as worker.js does.
                let am = temen_par_ev_a(root);
                let (smod, entry) = ((am >> 32) as u32, am as u32);
                assert_eq!((smod, entry), (0, 1), "same-module child at f1");
                let carve = temen_par_ev_b(root) as usize;
                let slog = temen_par_ev_c(root) as u32;
                assert_eq!(
                    (carve as u64, slog),
                    (CARVE_OFF, CARVE_LOG2),
                    "32 KiB carve at 64 KiB"
                );
                let cfuel = temen_par_ev_d(root);
                // SAFETY: the engine validated the carve lies inside the root window before
                // surfacing the event (worker.js relies on the same contract).
                let carve_ptr = unsafe { win_ptr.add(carve) };
                let child = temen_par_child_confined(prog, carve_ptr, slog, smod, entry, cfuel);
                assert!(!child.is_null(), "confined child vCPU builds");
                // Drive the child to completion (single-threaded stand-in for its Worker): its
                // tier-up events serve over ITS OWN CARVE — the #816 item 5 behavior under test.
                let v = loop {
                    match temen_par_run(child) {
                        PAR_DONE => break temen_par_ev_a(child),
                        PAR_TIERUP => {
                            tierups += 1;
                            assert!(tierups < 10, "runaway tier-ups");
                            service_tierup(child, carve_ptr, 1 << slog);
                        }
                        ev => panic!("unexpected child event {ev}"),
                    }
                };
                temen_par_free(child);
                child_value = Some(v);
                temen_par_deliver_handle(root, 0);
            }
            PAR_JOIN => {
                assert_eq!(temen_par_ev_a(root), 0, "join of the one child");
                temen_par_deliver_join(root, child_value.expect("child ran before join"), 0);
            }
            ev => panic!("unexpected root event {ev}"),
        }
    };
    temen_par_free(root);

    assert_eq!(child_value, Some(22), "child leaf f(5) over its carve");
    assert_eq!(
        result, want,
        "parallel drive with per-vCPU tier-up diverged from the interpreter oracle"
    );
    // Non-vacuity + the item-5 pin: BOTH the root's and the confined child's leaf calls tiered up
    // (before this slice the child's interpreted — this was 1).
    assert_eq!(
        tierups, 2,
        "root + confined child must each tier up exactly once (#816 item 5)"
    );
}

// ---- #1151: a §14 child that manages its OWN pages, paged, over its carve -------------------------

/// The child (entry `(inst, as)`, f1) `unmap`s the page at `unmap_off` in its carve through the
/// AddressSpace it was granted, then calls the eligible leaf f2 over a **different, still-mapped**
/// page — so f2 tiers up paged over the carve while a page it does *not* touch is now unmapped. The
/// leaf f2 (`f(x)=x*3+7`, storing at `marker_off`) stays byte-identical to the plain test. Offsets
/// are page-aligned above the 16 KiB NULL guard and generated for the host page size.
fn page_op_child_src(unmap_off: u64, marker_off: u64, page: u64) -> String {
    format!(
        r#"
memory 17
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  ve = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vh = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (ve, voff, vsl, vq)
  vj = call.cap 6 1 (i32) -> (i64) v0 (vh)
  return vj
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vinst: i64, vas: i64) {{
  vasi = i32.wrap_i64 vas
  voff = i64.const {unmap_off}
  vlen = i64.const {page}
  vu = call.cap 5 1 (i64, i64) -> (i64) vasi (voff, vlen)
  vx = i64.const 5
  vr = call 2 (vx)
  return vr
  }}
}}
func (i64) -> (i64) {{
block 0 (vx: i64) {{
  v3 = i64.const 3
  vm = i64.mul vx v3
  v7 = i64.const 7
  va = i64.add vm v7
  vaddr = i64.const {marker_off}
  i64.store vaddr va
  return va
  }}
}}
"#
    )
}

/// #1151 — the **par leaf-tier-up paged** path carries a §14 confined child's own `unmap` over its
/// carve: the child unmaps a page (on its interpreter), then a pure leaf tiers up over the carve, and
/// the pagestate table the driver hands the emitted leaf (`temen_par_tierup_pagestate_ptr`, built
/// from the CHILD's `map_info`) reflects that unmap. Composed with the emitted per-access page
/// check's window-independence — proven to trap `Unmapped`/`Ro` in `temen-wasm-jit`'s
/// `nested_paged.rs` and fuzzed by `crates/temen/tests/support/paged.rs` — this closes the
/// "a §14 child that touches page-ops runs confined on the emitted tier" acceptance (#1151) for the
/// leaf-tier-up path. The child's leaf here touches only a still-mapped page (so the interpreter
/// oracle does not trap and the emulated leaf is honest); the unmapped page is the one the pagestate
/// assertion inspects.
#[test]
fn par_confined_child_paged_reflects_its_own_unmap() {
    let _jit = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // #1182 — serial single-run
    let page = host_page_size();
    // Page-aligned offsets above the 16 KiB NULL guard, inside the 64 KiB carve.
    let unmap_off = 16384u64; // the first usable page (guard is a multiple of every host page size)
    let marker_off = 16384 + page; // a distinct, still-mapped page the leaf actually touches
    let src = page_op_child_src(unmap_off, marker_off, page);
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let bytes = temen_encode::encode_module(&m);

    // Oracle: the whole guest on the cooperative interpreter. The child unmaps a page it never
    // accesses, then f2 stores at the (mapped) marker and returns 22; the root joins → 22. No trap.
    let want = {
        let mut host = Host::new();
        let inst = host.grant_instantiator(0, 1 << 17);
        let mut run = bytecode::CoopRun::new(&m, 0, &[Value::I32(inst)], FUEL, host, None)
            .expect("supported")
            .expect("entry in range");
        match run.run() {
            bytecode::CoopEvent::Done(vals) => match vals.first() {
                Some(Value::I64(x)) => *x,
                other => panic!("non-i64 oracle result {other:?}"),
            },
            other => panic!(
                "oracle did not run to completion: {:?}",
                core::mem::discriminant(&other)
            ),
        }
    };
    assert_eq!(want, 22, "oracle: child f(5)=22, root joins");

    assert_eq!(
        temen_par_powerbox_inst(1 << 17, core::ptr::null(), 0, 0),
        1,
        "publish the §14 run recipe"
    );
    assert_eq!(
        temen_par_enable_jit_paged(bytes.as_ptr(), bytes.len()),
        1,
        "the leaf must be tier-up eligible under the paged tier"
    );
    let prog = temen_par_compile(bytes.as_ptr(), bytes.len());
    assert!(!prog.is_null(), "program compiles");
    let mut win = vec![0u8; 1 << 17].into_boxed_slice();
    let win_ptr = win.as_mut_ptr();
    let root = temen_par_root(prog, win_ptr, 1 << 17, 0);
    assert!(!root.is_null(), "root vCPU builds");

    let mut child_value: Option<i64> = None;
    let mut saw_unmap_reflected = false;
    let result = loop {
        match temen_par_run(root) {
            PAR_DONE => break temen_par_ev_a(root),
            PAR_INSTANTIATE => {
                let am = temen_par_ev_a(root);
                let (smod, entry) = ((am >> 32) as u32, am as u32);
                assert_eq!((smod, entry), (0, 1), "same-module child at f1");
                let carve = temen_par_ev_b(root) as usize;
                let slog = temen_par_ev_c(root) as u32;
                let cfuel = temen_par_ev_d(root);
                // SAFETY: the engine validated the carve lies inside the root window.
                let carve_ptr = unsafe { win_ptr.add(carve) };
                let child = temen_par_child_confined(prog, carve_ptr, slog, smod, entry, cfuel);
                assert!(!child.is_null(), "confined child vCPU builds");
                let v = loop {
                    match temen_par_run(child) {
                        PAR_DONE => break temen_par_ev_a(child),
                        PAR_TIERUP => {
                            assert_eq!(temen_par_ev_a(child), 2, "only the leaf is eligible");
                            // The #1151 assertion: the pagestate table the driver hands the emitted
                            // leaf (built from the CHILD's own live map) marks the page the child just
                            // unmapped as Unmapped (0), while the marker page it stores to stays Rw (1).
                            let plen = temen_par_tierup_pagestate_len(child);
                            assert!(
                                plen > 0,
                                "paged run must expose a page-state table for the child"
                            );
                            // SAFETY: the pending-event table is stable until deliver; this thread is
                            // the only accessor (single-threaded stand-in for the child's Worker).
                            let table = unsafe {
                                std::slice::from_raw_parts(
                                    temen_par_tierup_pagestate_ptr(child),
                                    plen,
                                )
                            };
                            let upage = (unmap_off / page) as usize;
                            let mpage = (marker_off / page) as usize;
                            assert!(
                                upage < plen && mpage < plen,
                                "both pages within the child's table coverage"
                            );
                            assert_eq!(
                                table[upage], 0,
                                "the page the child unmapped must read Unmapped in ITS OWN pagestate"
                            );
                            assert_eq!(
                                table[mpage], 1,
                                "the still-mapped marker page must read Rw"
                            );
                            saw_unmap_reflected = true;
                            // `ev_b` is the paged coverage (table bytes × page), the child's own bound.
                            assert_eq!(
                                temen_par_ev_b(child) as u64,
                                plen as u64 * page,
                                "the paged 'mapped' bound is the child's own table coverage"
                            );
                            // Emulate f2 over the carve (it touches only the mapped marker page).
                            let argv = unsafe {
                                std::slice::from_raw_parts(
                                    temen_par_tierup_argv_ptr(child),
                                    temen_par_tierup_argv_len(child),
                                )
                            };
                            let r = argv[0] * 3 + 7;
                            // SAFETY: the paused child is parked; the carve is exclusively ours, and
                            // `marker_off + 8` is inside the 64 KiB carve.
                            unsafe {
                                std::ptr::copy_nonoverlapping(
                                    r.to_le_bytes().as_ptr(),
                                    carve_ptr.add(marker_off as usize),
                                    8,
                                );
                            }
                            temen_par_deliver_tierup(child, [r].as_ptr(), 1);
                        }
                        ev => panic!("unexpected child event {ev}"),
                    }
                };
                temen_par_free(child);
                child_value = Some(v);
                temen_par_deliver_handle(root, 0);
            }
            PAR_JOIN => {
                assert_eq!(temen_par_ev_a(root), 0, "join of the one child");
                temen_par_deliver_join(root, child_value.expect("child ran before join"), 0);
            }
            ev => panic!("unexpected root event {ev}"),
        }
    };
    temen_par_free(root);

    assert_eq!(child_value, Some(22), "child leaf f(5) over its carve");
    assert_eq!(
        result, want,
        "parallel paged drive diverged from the interpreter oracle"
    );
    assert!(
        saw_unmap_reflected,
        "non-vacuity: the child's leaf tiered up paged and its unmap was reflected in the pagestate"
    );
}
