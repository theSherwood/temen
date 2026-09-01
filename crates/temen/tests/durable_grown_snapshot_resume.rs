//! #816 Slice C oracle (invariant 14, **durability axis**, cross-host leg): a durability-instrumented
//! guest that **`vm_map`-grows** its window past its declared size, **freezes mid-run** at a clock
//! unwind point, is serialized through the §12 SVMD codec, **restored on a fresh `Host`**, and
//! **resumes to completion** — with the grown-page content surviving the whole freeze→serialize→
//! restore→thaw cross-host round-trip.
//!
//! This is the native oracle for the browser "persist a warmed/grown guest across a reload"
//! consumer (`browser/src/lib.rs` `temen_durable_freeze` / `temen_durable_thaw_resume`): the browser
//! FFI drives exactly this sequence, storing the SVMD bytes in IndexedDB between the freeze and the
//! thaw. The codec-only round-trip is already pinned by `durable_prot_capture.rs`
//! (`a_vm_map_grown_window_survives_the_codec`); what this adds is the *running* two-phase shape —
//! freeze-at-unwind → SVMD → restore → **resume** — of a grown guest, the behavior the reload
//! consumer depends on and which nothing exercised before.
//!
//! Freeze point: the guest self-flips the state word to `UNWINDING` just before the **clock** call
//! (the `multipoint.rs` idiom), so the grow + marker store complete first and the unwind lands at the
//! clock. The grow (`vm_map`) is an earlier resume point, so on thaw the growth is **not** re-issued —
//! the grown pages can only be present because they rode the SVMD artifact. Handle continuity across
//! the codec mirrors `durable_nesting.rs`: the restored caps are recovered from the thawed handle
//! table, never re-granted.

use std::sync::Arc;
use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module_assume_confined,
};
use temen_interp::{
    bytecode, run_capture_reserved_with_host_prots, CapturedProt, Host, Region, Value,
};
use temen_ir::{Memory, Module};
use temen_snapshot::{freeze_with_prots, restore_with_prots, PageProt, PAGE};

const SIZE_LOG2: u8 = 17; // 128 KiB declared window (64 KiB durable reserve + guest-usable above it)
const RESERVED_LOG2: u8 = 20; // 1 MiB mask domain the guest grows within
const WINDOW: usize = 1 << SIZE_LOG2;
const GROW_OFF: u64 = 1 << SIZE_LOG2; // grow starts exactly at the declared window's end (128 KiB)
const GROW_LEN: u64 = 64 * 1024; // [128 KiB, 192 KiB) — the grown tail, above the declared window
const MARK_ADDR: u64 = GROW_OFF + 3 * PAGE as u64 + 7; // a byte inside a grown page
const MARK_VAL: i64 = 77;
const STATE_ADDR: i64 = 16384; // the window's freeze state word (STATE_OFF), guest-addressable in Phase 1
const STATE_UNWINDING: i32 = 1;
const FREEZE_CLOCK: i64 = 42; // captured at freeze; replayed (not re-issued) on thaw
const THAW_CLOCK: i64 = 9999; // deliberately different — a re-issue instead of a replay would show it

fn to_codec_prots(caps: &[CapturedProt]) -> Vec<PageProt> {
    caps.iter()
        .map(|c| match c {
            CapturedProt::Rw => PageProt::Rw,
            CapturedProt::Ro => PageProt::Ro,
            CapturedProt::Unmapped => PageProt::Unmapped,
            CapturedProt::Backed => unreachable!("guest holds no §13 backed regions"),
        })
        .collect()
}

// A confined durable guest taking two caps: v0 = AddressSpace (vm_map), v1 = Clock. It grows its
// window into the reserved tail, writes a marker into a grown page, reads the clock (the durable
// unwind point), then reloads the marker *after* the call and returns clock + marker. Baseline
// (clock 42): 42 + 77 = 119. The reloaded marker lives in a grown page, so it can only survive a
// freeze/thaw if the grown extent rides the SVMD artifact. `flip` inserts the self-requested freeze
// (store UNWINDING into the state word) just before the clock — the mid-run trigger.
fn guest_src(flip: bool) -> String {
    let flip_ir = if flip {
        format!(
            "  vsa = i64.const {STATE_ADDR}\n  vsu = i32.const {STATE_UNWINDING}\n  i32.store vsa vsu\n"
        )
    } else {
        String::new()
    };
    format!(
        "func (i32, i32) -> (i64) {{\n\
block 0 (v0: i32, v1: i32) {{\n\
  voff = i64.const {grow_off}\n\
  vlen = i64.const {grow_len}\n\
  vprot = i32.const 3\n\
  vg = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (voff, vlen, vprot)\n\
  vaddr = i64.const {mark_addr}\n\
  vmark = i64.const {mark_val}\n\
  i64.store vaddr vmark\n\
{flip_ir}\
  vz = i32.const 0\n\
  vc = call.cap 2 0 (i32) -> (i64) v1 (vz)\n\
  vld = i64.load vaddr\n\
  vsum = i64.add vc vld\n\
  return vsum\n\
  }}\n\
}}\n",
        grow_off = GROW_OFF,
        grow_len = GROW_LEN,
        mark_addr = MARK_ADDR,
        mark_val = MARK_VAL,
    )
}

fn instrument(flip: bool) -> Module {
    let mut m = temen_text::parse_module(&guest_src(flip)).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module_assume_confined(&m).expect("confined transform");
    temen_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

// Grant a fresh AddressSpace + Clock (the two entry caps) and run the instrumented guest over
// `window`/`prots` at the grown reservation. Returns (result, captured window, captured prots).
fn run_fresh(
    inst: &Module,
    clock_v: i64,
    window: &[u8],
    prots: Option<&[CapturedProt]>,
    host: &mut Host,
) -> (
    Result<Vec<Value>, temen_interp::Trap>,
    Vec<u8>,
    Vec<CapturedProt>,
) {
    let space = host.grant_memory();
    let clk = host.grant_clock();
    run_at(inst, clock_v, window, prots, host, space, clk)
}

// Run the instrumented guest with explicit cap handles (for the thaw phase, where the handles come
// from the restored table rather than fresh grants).
fn run_at(
    inst: &Module,
    clock_v: i64,
    window: &[u8],
    prots: Option<&[CapturedProt]>,
    host: &mut Host,
    space: i32,
    clk: i32,
) -> (
    Result<Vec<Value>, temen_interp::Trap>,
    Vec<u8>,
    Vec<CapturedProt>,
) {
    host.clock_ns = clock_v;
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host_prots(
        inst,
        0,
        &[Value::I32(space), Value::I32(clk)],
        &mut fuel,
        window,
        prots,
        RESERVED_LOG2,
        host,
    )
}

#[test]
fn a_grown_durable_guest_survives_freeze_serialize_restore_resume() {
    let oracle = instrument(false);
    let freezable = instrument(true);

    // Baseline: the uninterrupted oracle run (no self-flip). 42 + grown-page marker 77 = 119.
    let mut base_host = Host::new();
    base_host.set_durable(true);
    let (baseline, _, _) = run_fresh(
        &oracle,
        FREEZE_CLOCK,
        &init_durable_window(WINDOW),
        None,
        &mut base_host,
    );
    let baseline = baseline.expect("baseline runs to completion");
    assert_eq!(baseline, vec![Value::I64(119)], "42 + grown-page marker 77");

    // Phase 1 — freeze mid-run: grow + store run NORMAL, then the self-flip unwinds at the clock.
    let mut fhost = Host::new();
    fhost.set_durable(true);
    let (froze, fsnap, fprots) = run_fresh(
        &freezable,
        FREEZE_CLOCK,
        &init_durable_window(WINDOW),
        None,
        &mut fhost,
    );
    froze.expect("freeze run does not trap");
    assert_eq!(
        read_state(&fsnap),
        STATE_UNWINDING,
        "froze, did not complete"
    );
    let mark_page = MARK_ADDR as usize / PAGE;
    assert!(mark_page >= WINDOW / PAGE, "marker is in the grown tail");
    assert!(
        fsnap.len() > WINDOW,
        "the grown extent was captured (len {} > declared {})",
        fsnap.len(),
        WINDOW
    );
    assert_eq!(
        fsnap[MARK_ADDR as usize],
        MARK_VAL.to_le_bytes()[0],
        "marker captured in the grown page at freeze"
    );
    assert_eq!(
        fprots[mark_page],
        CapturedProt::Rw,
        "grown page captured Rw"
    );

    // Serialize the frozen (grown) domain through the §12 SVMD codec at its real reservation, then
    // restore it on a FRESH host — the cross-host boundary the browser crosses via IndexedDB.
    let art = freeze_with_prots(
        &freezable,
        &fsnap,
        &to_codec_prots(&fprots),
        RESERVED_LOG2,
        &fhost,
    )
    .expect("freeze grown durable domain to SVMD");
    let mut thost = Host::new();
    thost.set_durable(true);
    let (rwin, rprots, rreserved) =
        restore_with_prots(&art, &freezable, &mut thost).expect("restore grown durable domain");
    assert_eq!(
        rreserved, RESERVED_LOG2,
        "the mask domain survives the codec"
    );
    assert_eq!(
        rwin[MARK_ADDR as usize],
        MARK_VAL.to_le_bytes()[0],
        "grown-page marker survives serialize/restore"
    );
    assert_eq!(rprots[mark_page], PageProt::Rw, "grown page restored Rw");

    // Recover the restored cap handles from the thawed table (never re-grant — mirrors
    // durable_nesting.rs). The guest granted AddressSpace then Clock, so the caps come back in that
    // order; the i32 handle is `(generation << 8) | slot`.
    let caps = thost
        .capture_durable_handles()
        .expect("only durable handles restored");
    assert_eq!(caps.len(), 2, "AddressSpace + Clock restored");
    let handle = |i: usize| ((caps[i].generation << 8) | caps[i].slot) as i32;
    let (t_space, t_clk) = (handle(0), handle(1));

    // Phase 2 — thaw + resume on the restored window, through the real grown-restore seam
    // (`SharedProgram::run_over_grown`, #828/#1127 — the primitive the browser thaw FFI uses): fill a
    // fresh backing with the restored (grown) window image, set it REWINDING, and re-establish the
    // captured page map via `seed_pages`. The clock result (42) is REPLAYED (THAW_CLOCK would surface
    // if it were re-issued), the grown-page marker is reloaded, and the guest runs to completion with
    // the same result as the uninterrupted baseline. `vm_map` (an earlier resume point) is NOT
    // re-issued, so the grown pages are present only because they rode the SVMD artifact.
    let prog = bytecode::SharedProgram::compile(&freezable).expect("compile for resume");
    let mut rwin = rwin;
    begin_thaw(&mut rwin, 0); // clear the freeze word, set context 0 REWINDING

    // A fresh backing sized to the restored reservation, pre-filled with the restored window image
    // (grown pages included). Leaked for the run's life; `Region::shared` borrows it.
    let size = 1usize << rreserved;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked, so the Region borrow is sound for the test.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `base` owns `size` bytes and `rwin.len() <= size` (mapped extent ≤ reservation).
    unsafe { core::ptr::copy_nonoverlapping(rwin.as_ptr(), base, rwin.len()) };
    let back = Arc::new(unsafe { Region::shared(base, size as u64) });

    // The restored per-page protection map, as `seed_pages` entries (kind: 0=Ro, 1=Rw, 2=Unmapped).
    let entries: Vec<(u64, u8)> = rprots
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let kind = match p {
                PageProt::Ro => 0u8,
                PageProt::Rw => 1,
                PageProt::Unmapped => 2,
            };
            (i as u64 * PAGE as u64, kind)
        })
        .collect();

    // Seed the thaw host's clock to a DIFFERENT value: a correct replay reproduces 42; a re-issue
    // would consume THAW_CLOCK and shift the result, so the equality below is load-bearing.
    thost.clock_ns = THAW_CLOCK;
    let mut fuel = 5_000_000u64;
    let (thawed, _, _) = prog.run_over_grown(
        0,
        &[Value::I32(t_space), Value::I32(t_clk)],
        &mut fuel,
        back,
        &mut thost,
        false, // bytes already restored into the backing — do not re-init data segments
        rreserved,
        Some(&entries),
    );
    assert_eq!(
        thawed.expect("resume runs to completion"),
        baseline,
        "clock replayed (42, not 9999) + grown-page marker 77 reloaded across the reload"
    );
}
