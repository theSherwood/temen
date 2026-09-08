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
use wasmi::{Caller, Engine, Linker, Memory, MemoryType, Module as WModule, Store, Val};

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

// ---- #1347: a runtime unit's `call.dyn` into the dispatch table's NATURAL PREFIX on the par driver --
//
// A `compile_linked` unit's Slot import names a *program function* (the JACL macro-staging shape). On
// the interpreter and the coop tier that dispatches through the shared table's natural prefix; the
// parallel Worker's B2 table mirror used to fill only installed-unit slots, leaving every program slot a
// null funcref. This plays `worker.js` under wasmi over the real `temen_par_*` FFI: the root
// `compile_linked`s + `invoke`s a unit whose `call.dyn` lands on program function 1 — once
// interpreter-resident (a bounce shim → `temen_par_inst_call_interp` on the root vCPU), once emitted by
// the tier-up module (`emitted.f1`, natively) — differential against the interpreted service.

const LK_WIN_BASE: u32 = 0x4_0000;
const LK_ENV_PTR: u32 = 1024;
const LK_BLOB_OFF: usize = 0x6000;
const LK_SYMTAB_OFF: usize = 0x7000;
const LK_PROBE: i64 = 21;
const LK_K: i64 = 90909;
/// Where the emitted `F` stores + reloads its result (above the NULL guard, below the staged blob).
const LK_MARK: i64 = 0x5000;

/// `unit(x) = F(x) + LK_K`, `F` an unresolved `call.sym` the guest binds to Slot 1 at link time.
fn lk_unit_module() -> temen_ir::Module {
    let src = format!(
        r#"memory 16
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i32.const 0
  v2 = call.sym "F" (i64) -> (i64) v1 (v0)
  vk = i64.const {LK_K}
  v3 = i64.add v2 vk
  return v3
  }}
}}
"#
    );
    temen_text::parse_module(&src).expect("unit parse")
}

/// The guest: f0 (root, arg = its `Jit` handle) `compile_linked`s the staged unit against the staged
/// symbol table (`"F"` → Slot 1) and `invoke`s it with `LK_PROBE`. f1 = `F(x) = 2x`: with
/// `emitted_callee` a pure leaf that stores + reloads through `[LK_MARK]` (so a mis-primed `"mapped"`
/// would fault it), else a `call.dyn` through slot 2 to f2 — off the tier-up emit, so the unit's
/// `call.dyn` to slot 1 must bounce. f2 = the helper `2x`.
fn lk_guest_module(blob_len: usize, st_len: usize, emitted_callee: bool) -> temen_ir::Module {
    let f1 = if emitted_callee {
        format!(
            "  v1 = i64.const 2\n  vm = i64.mul v0 v1\n  va = i64.const {LK_MARK}\n  i64.store va vm\n  vr = i64.load va\n  return vr\n"
        )
    } else {
        "  vs2 = i32.const 2\n  vr = call.dyn (i64) -> (i64) vs2 (v0)\n  return vr\n".to_string()
    };
    let src = format!(
        r#"memory 16
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vbp = i64.const {LK_BLOB_OFF}
  vbl = i64.const {blob_len}
  vsp = i64.const {LK_SYMTAB_OFF}
  vsn = i64.const {st_len}
  vcode = call.cap 11 5 (i64, i64, i64, i64) -> (i64) v0 (vbp, vbl, vsp, vsn)
  vprobe = i64.const {LK_PROBE}
  vres = call.cap 11 1 (i64, i64) -> (i64) v0 (vcode, vprobe)
  return vres
  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
{f1}  }}
}}
func (i64) -> (i64) {{
block 0 (v0: i64) {{
  v1 = i64.const 2
  vr = i64.mul v0 v1
  return vr
  }}
}}
"#
    );
    let m = temen_text::parse_module(&src).expect("guest parse");
    temen_verify::verify_module(&m).expect("guest verify");
    m
}

struct LkDrv {
    v: usize,
    bounces: u32,
    mapped_globals: Vec<wasmi::Global>,
    fuel_globals: Vec<wasmi::Global>,
}

/// Instantiate an emitted module (unit / shim / tier-up module) against the harness's memory + table,
/// with `env.call_interp` routed to the root vCPU's live-state bounce — `worker.js`'s import objects.
fn lk_instantiate(
    store: &mut Store<LkDrv>,
    engine: &Engine,
    memory: Memory,
    table: wasmi::Table,
    wasm: &[u8],
) -> wasmi::Instance {
    use temen_browser::temen_par_inst_call_interp;
    let module = WModule::new(engine, wasm).expect("emitted wasm validates");
    let mut linker: Linker<LkDrv> = Linker::new(engine);
    linker.define("env", "memory", memory).unwrap();
    linker
        .define("env", "__indirect_function_table", table)
        .unwrap();
    linker
        .func_wrap("env", "trap", |_c: Caller<'_, LkDrv>, _code: i32| {})
        .unwrap();
    linker
        .func_wrap(
            "env",
            "call_interp",
            move |mut c: Caller<'_, LkDrv>,
                  target: i32,
                  args_ptr: i32|
                  -> Result<(), wasmi::Error> {
                c.data_mut().bounces += 1;
                let v = c.data().v as *mut temen_browser::ParVcpu;
                // SAFETY: `args_ptr` is the env scratch inside the fixed-size wasmi memory.
                let ap = unsafe { memory.data_mut(&mut c).as_mut_ptr().add(args_ptr as usize) };
                if temen_par_inst_call_interp(v, target as u32, ap) != 0 {
                    return Err(wasmi::Error::from(
                        wasmi::core::TrapCode::UnreachableCodeReached,
                    ));
                }
                Ok(())
            },
        )
        .unwrap();
    let inst = linker
        .instantiate(&mut *store, &module)
        .unwrap()
        .start(&mut *store)
        .unwrap();
    if let Some(g) = inst.get_global(&*store, "mapped") {
        store.data_mut().mapped_globals.push(g);
    }
    if let Some(g) = inst.get_global(&*store, "fuel") {
        store.data_mut().fuel_globals.push(g);
    }
    inst
}

/// One run: `(value, surfaced invokes, bounces)`. `codegen == false` is the oracle (the engine services
/// the invoke interpreted, no events); `codegen == true` plays the Worker.
fn lk_run(codegen: bool, emitted_callee: bool) -> (i64, u32, u32) {
    use temen_browser::{
        temen_par_compile_jit, temen_par_deliver_jit_invoke, temen_par_deliver_jit_invoke_trap,
        temen_par_jit_argv_len, temen_par_jit_argv_ptr, temen_par_jit_code,
        temen_par_jit_code_wasm_len, temen_par_jit_param_types_ptr, temen_par_jit_result_types_len,
        temen_par_jit_result_types_ptr, temen_par_jit_set_b2, temen_par_jit_set_codegen,
        temen_par_jit_slot_code, temen_par_jit_table_log2, temen_par_nfuncs,
        temen_par_powerbox_jit_runtime, temen_par_shim_wasm_len, temen_wasmjit_len,
        temen_wasmjit_ptr, PAR_JIT_INVOKE,
    };
    let unit_m = lk_unit_module();
    let blob = temen_encode::encode_module(&unit_m); // imports unresolved — the guest links it
    let symtab: Vec<u8> = vec![1, 1, b'F', 0, 1]; // `"F"` → Slot(1) (canonical wire form)
    let guest = lk_guest_module(blob.len(), symtab.len(), emitted_callee);
    let guest_bytes = temen_encode::encode_module(&guest);

    assert_eq!(
        temen_par_powerbox_jit_runtime(guest_bytes.as_ptr(), guest_bytes.len()),
        1,
        "runtime-compile powerbox"
    );
    temen_par_jit_set_b2(1);
    temen_par_jit_set_codegen(if codegen { 1 } else { 0 });
    let prog = temen_par_compile_jit(guest_bytes.as_ptr(), guest_bytes.len());
    assert!(!prog.is_null(), "guest compiles");
    // The tier-up module: `F` (pure leaf, all-i64) emits + is eligible in the emitted-callee shape;
    // in the shim shape `F` calls f2 so it is interpreter-resident (f2 itself emits, unreached).
    let tiers = temen_par_enable_jit(guest_bytes.as_ptr(), guest_bytes.len());
    assert_eq!(tiers, 1, "some leaf emits in either shape");

    let engine = Engine::default();
    let mut store: Store<LkDrv> = Store::new(
        &engine,
        LkDrv {
            v: 0,
            bounces: 0,
            mapped_globals: Vec::new(),
            fuel_globals: Vec::new(),
        },
    );
    let pages = (LK_WIN_BASE + (1 << 16)).div_ceil(1 << 16) + 1;
    let memory = Memory::new(&mut store, MemoryType::new(pages, Some(pages))).unwrap();
    memory
        .write(&mut store, LK_ENV_PTR as usize, &i64::MAX.to_le_bytes())
        .unwrap();
    // Stage the unit blob + symbol table where the guest reads them (before the root seeds data).
    memory
        .write(&mut store, LK_WIN_BASE as usize + LK_BLOB_OFF, &blob)
        .unwrap();
    memory
        .write(&mut store, LK_WIN_BASE as usize + LK_SYMTAB_OFF, &symtab)
        .unwrap();
    // SAFETY: fixed-size memory ⇒ a stable data pointer; the window lives inside it (the browser's
    // shared-linear-memory shape) and is used solely as this run's window.
    let win_ptr = unsafe {
        memory
            .data_mut(&mut store)
            .as_mut_ptr()
            .add(LK_WIN_BASE as usize)
    };
    let v = temen_par_root(prog, win_ptr, 1 << 16, 0);
    assert!(!v.is_null(), "root vCPU builds");
    store.data_mut().v = v as usize;

    let tsize = 1u32 << temen_par_jit_table_log2();
    let table = wasmi::Table::new(
        &mut store,
        wasmi::TableType::new(wasmi::core::ValType::FuncRef, tsize, Some(tsize)),
        Val::FuncRef(wasmi::FuncRef::null()),
    )
    .unwrap();
    // wasmi validates no threads proposal: run non-shared twins of the FFI's shared-memory emits (the
    // shared import only adds a max limit — a few LEB bytes; `inst_codegen_paged.rs` pins the same).
    let emitted = {
        let stashed =
            unsafe { std::slice::from_raw_parts(temen_wasmjit_ptr(), temen_wasmjit_len()) };
        let art = temen_wasm_jit::compile_jit(&guest, temen_wasm_jit::Shape::Threaded, false)
            .expect("tier-up emit");
        assert!(
            stashed.len() > art.wasm.len() && stashed.len() - art.wasm.len() <= 8,
            "tier-up stash {} vs unshared {}",
            stashed.len(),
            art.wasm.len()
        );
        assert_eq!(
            art.emitted[1], emitted_callee,
            "`F` emits iff it is the pure-leaf shape"
        );
        lk_instantiate(&mut store, &engine, memory, table, &art.wasm)
    };
    let linked = temen_ir::resolve_imports_with(&unit_m, |_| Some(temen_ir::Resolved::Slot(1)))
        .expect("link");
    temen_verify::verify_module(&linked).expect("linked unit verifies");
    let unit_wasm = temen_wasm_jit::compile_module_b2(&linked, false, temen_par_jit_table_log2())
        .expect("B2 unit emit");
    let mut unit_inst: Option<wasmi::Instance> = None;
    let mut shims: Vec<Option<wasmi::Instance>> = vec![None; tsize as usize];

    let mut invokes = 0u32;
    let value = loop {
        match temen_par_run(v) {
            PAR_DONE => break temen_par_ev_a(v),
            PAR_JIT_INVOKE => {
                invokes += 1;
                assert!(invokes < 8, "runaway invokes");
                // `worker.js::jitSyncTable` (#1347): the natural prefix from the tier-up module or
                // a shim, installed slots from their units (none here), the rest null.
                let nfuncs = temen_par_nfuncs();
                assert_eq!(
                    nfuncs, 3,
                    "the natural prefix is the program's function count"
                );
                for slot in 0..tsize {
                    let entry = if (slot as usize) < nfuncs {
                        match emitted.get_func(&store, &format!("f{slot}")) {
                            Some(f) => Some(f),
                            None => {
                                if shims[slot as usize].is_none() {
                                    let ffi_len = temen_par_shim_wasm_len(slot);
                                    let (p, r) = (
                                        &guest.funcs[slot as usize].params,
                                        &guest.funcs[slot as usize].results,
                                    );
                                    let w = temen_wasm_jit::emit_slot_trampoline(p, r, slot, false)
                                        .expect("shim emit");
                                    assert!(
                                        ffi_len > w.len() && ffi_len - w.len() <= 8,
                                        "FFI shim {ffi_len} vs unshared {}",
                                        w.len()
                                    );
                                    shims[slot as usize] = Some(lk_instantiate(
                                        &mut store, &engine, memory, table, &w,
                                    ));
                                }
                                shims[slot as usize].unwrap().get_func(&store, "t")
                            }
                        }
                    } else {
                        assert_eq!(temen_par_jit_slot_code(slot), -1, "nothing installed");
                        None
                    };
                    let fr = match entry {
                        Some(f) => wasmi::FuncRef::new(f),
                        None => wasmi::FuncRef::null(),
                    };
                    table
                        .set(&mut store, slot as u64, Val::FuncRef(fr))
                        .unwrap();
                }
                // The invoked unit (cached per code handle): the FFI emitted it (shared) on demand.
                assert!(temen_par_jit_code_wasm_len(v) > 0, "the linked unit emits");
                let _code = temen_par_jit_code(v);
                let inst = *unit_inst.get_or_insert_with(|| {
                    lk_instantiate(&mut store, &engine, memory, table, &unit_wasm)
                });
                // Prime every instance's `"mapped"` (the event's extent) + fuel, as the Worker does.
                let mapped = temen_par_ev_b(v);
                for g in store.data().mapped_globals.clone() {
                    g.set(&mut store, Val::I64(mapped)).unwrap();
                }
                for g in store.data().fuel_globals.clone() {
                    g.set(&mut store, Val::I64(1 << 61)).unwrap();
                }
                memory
                    .write(&mut store, LK_ENV_PTR as usize, &(1i64 << 61).to_le_bytes())
                    .unwrap();
                let n = temen_par_jit_argv_len(v);
                // SAFETY: pending-event operand stash, stable until the deliver.
                let argv = unsafe { std::slice::from_raw_parts(temen_par_jit_argv_ptr(v), n) };
                let ptypes =
                    unsafe { std::slice::from_raw_parts(temen_par_jit_param_types_ptr(v), n) };
                let mut params = vec![Val::I32(LK_WIN_BASE as i32), Val::I32(LK_ENV_PTR as i32)];
                for (a, tc) in argv.iter().zip(ptypes) {
                    params.push(match tc {
                        0 => Val::I32(*a as i32),
                        1 => Val::I64(*a),
                        _ => panic!("non-integer arg in this guest"),
                    });
                }
                let rn = temen_par_jit_result_types_len(v);
                let rtypes =
                    unsafe { std::slice::from_raw_parts(temen_par_jit_result_types_ptr(v), rn) };
                let mut results: Vec<Val> = rtypes
                    .iter()
                    .map(|tc| if *tc == 0 { Val::I32(0) } else { Val::I64(0) })
                    .collect();
                let f0 = inst.get_func(&store, "f0").expect("unit exports f0");
                match f0.call(&mut store, &params, &mut results) {
                    Ok(()) => {
                        let slots: Vec<i64> = results
                            .iter()
                            .map(|r| match r {
                                Val::I32(x) => *x as i64,
                                Val::I64(x) => *x,
                                _ => unreachable!(),
                            })
                            .collect();
                        temen_par_deliver_jit_invoke(v, slots.as_ptr(), slots.len());
                    }
                    Err(_) => temen_par_deliver_jit_invoke_trap(v),
                }
            }
            ev => panic!(
                "unexpected par event {ev} (codegen={codegen}, emitted_callee={emitted_callee})"
            ),
        }
    };
    let bounces = store.data().bounces;
    temen_par_free(v);
    (value, invokes, bounces)
}

#[test]
fn par_linked_unit_dispatches_program_functions_through_the_b2_mirror() {
    let _jit = JIT_STATE_LOCK.lock().unwrap_or_else(|e| e.into_inner()); // #1182 — serial single-run
    for emitted_callee in [false, true] {
        let (want, i0, b0) = lk_run(false, emitted_callee);
        assert_eq!(
            want,
            2 * LK_PROBE + LK_K,
            "oracle value (emitted_callee={emitted_callee})"
        );
        assert_eq!(
            (i0, b0),
            (0, 0),
            "the oracle services the invoke interpreted, no events"
        );
        let (got, invokes, bounces) = lk_run(true, emitted_callee);
        assert_eq!(
            got, want,
            "B2 codegen ≡ interpreter (emitted_callee={emitted_callee})"
        );
        assert_eq!(invokes, 1, "the linked unit ran on emitted wasm");
        // Non-vacuity: the unit's `call.dyn` reached program function 1 through the natural prefix —
        // via the bounce shim when interpreter-resident, natively (no bounce) when emitted.
        assert_eq!(
            bounces,
            u32::from(!emitted_callee),
            "bounce iff `F` is interpreter-resident"
        );
    }
}
