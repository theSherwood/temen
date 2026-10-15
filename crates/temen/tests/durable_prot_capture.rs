//! Phase-2 page protections, end to end on the interpreter:
//!
//! * **capture** — the interpreter reports real per-page protections, so a D40 `readonly` data
//!   segment is captured as `Ro` and survives the §12 codec (byte image *and* protection), where
//!   Phase-1's flat all-`Rw` image would have lost it;
//! * **re-establish** — restoring that protection map and seeding a thawed run with it makes a
//!   write to a restored `Ro` page fault exactly as the frozen guest would (vs. succeeding when
//!   the page comes back `Rw`) — on **both** the interpreter and the JIT (real `mprotect` /
//!   `VirtualProtect`), so the two backends agree on a thawed protected window.

use temen_interp::{run_capture_reserved_with_host_layout, CapturedProt, Host, MemLayout, Value};
use temen_ir::{Memory, Module};
use temen_snapshot::{freeze_with_prots, restore_with_prots, PageProt, PAGE};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

const SIZE_LOG2: u8 = 17;
const WINDOW: usize = 1 << SIZE_LOG2;
const RO_OFF: usize = 5 * PAGE; // a read-only data segment lands on page 5
                                // The "ordinary committed page" reference: page 8 (offset 32768). It must be Rw ABOVE the
                                // #1094 NULL guard `[0, 16384)` (pages 0..3, now seeded Unmapped) *and* live in a different
                                // host page from `RO_OFF` — protection is host-page granular, so on a 16 KiB-page host
                                // (macOS) page 4 shares host page 1 with the page-5 RO segment and captures Ro; page 8 is a
                                // clean host page above it.
const RW_OFF: usize = 8 * PAGE;

/// `m`'s `func` on the interpreter over `init`, in a reservation of `1 << reserved_log2`: how it
/// ended, and the window it left.
fn interp_run(
    m: &Module,
    func: u32,
    args: &[Value],
    init: &MemLayout,
    reserved_log2: u8,
    host: &mut Host,
) -> (Result<Vec<Value>, temen_interp::Trap>, MemLayout) {
    let mut fuel = 1_000_000u64;
    let (r, left) =
        run_capture_reserved_with_host_layout(m, func, args, &mut fuel, init, reserved_log2, host);
    (r, left.expect("the guest has a window"))
}

/// What a window layout holds, page for page: its head, the head's protections, and each page it
/// holds past the head (#2427).
type Held<'a> = (
    &'a [u8],
    Vec<CapturedProt>,
    Vec<(u64, CapturedProt, Option<&'a [u8]>)>,
);

fn held(l: &MemLayout) -> Held<'_> {
    (l.bytes(), l.dense_prots(), l.tail_pages())
}

// A read-only data segment + a trivial entry (it doesn't touch memory; the segment alone marks
// its page `Ro` at instantiation, D40).
const SRC: &str = r#"
data ro 20480 "ABCD"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  return v1
  }
}
"#;

#[test]
fn readonly_data_segment_is_captured_and_survives_the_codec() {
    assert_eq!(RO_OFF, 20480);
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });

    let mut host = Host::new();
    let _ = host.grant_clock(); // a durable handle so the freeze has a non-empty table

    let init = MemLayout::image(vec![0u8; WINDOW]);
    let (r, left) = interp_run(&m, 0, &[Value::I32(0)], &init, SIZE_LOG2, &mut host);
    assert_eq!(r, Ok(vec![Value::I64(7)]));
    let (window, caps) = (left.bytes(), left.dense_prots());

    // Capture: the readonly segment's page is `Ro`; an ordinary committed page is `Rw`.
    let ro_page = RO_OFF / PAGE;
    assert_eq!(
        caps[ro_page],
        CapturedProt::Ro,
        "readonly segment page captured Ro"
    );
    assert_eq!(
        caps[RW_OFF / PAGE],
        CapturedProt::Rw,
        "an ordinary committed page (above the #1094 guard) is Rw"
    );
    assert_eq!(
        &window[RO_OFF..RO_OFF + 4],
        b"ABCD",
        "segment bytes landed in the window"
    );

    // Through the §12 codec: the protection is recorded and recovered (Phase-1 would have lost it).
    let art = freeze_with_prots(&m, window, &caps, SIZE_LOG2, &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots, _) = restore_with_prots(&art, &m, &mut rhost).expect("restore");
    assert_eq!(
        rprots[ro_page],
        PageProt::Ro,
        "Ro survives serialize/restore"
    );
    assert_eq!(rprots[RW_OFF / PAGE], PageProt::Rw);
    assert_eq!(&rwin[RO_OFF..RO_OFF + 4], b"ABCD", "Ro page bytes survive");
}

/// #1154 end-to-end (invariant 14, durability axis): a guest that **`vm_map`-grows** its window past
/// its declared size, captured through the real interpreter and round-tripped through the §12 codec.
/// Pre-v18 the captured window (larger than `1 << size_log2`) hit `GeometryMismatch`; v18 carries the
/// grown extent + reservation, so the grown page's content and protection survive serialize/restore —
/// the durable-artifact analogue of the in-process warm-snapshot restore (#828/#1127). v46 (#2427)
/// carries them as pages held past the declared window.
#[test]
fn a_vm_map_grown_window_survives_the_codec() {
    // Declares 128 KiB; `_start` `vm_map`s [128 KiB, 192 KiB) Rw and writes a marker into the grown
    // region. The capture reserves 512 KiB (`GROW_RESERVED_LOG2`), so the guest grows within a mask
    // domain larger than its declared window.
    const GROW_SIZE_LOG2: u8 = 17; // 128 KiB declared
    const GROW_RESERVED_LOG2: u8 = 19; // 512 KiB reservation
    const GROWN_MARK_OFF: usize = (1 << GROW_SIZE_LOG2) + 3 * PAGE + 7; // a byte in a grown page
    let src = r#"memory 17 shadow 16448 65536
func (i32) -> (i64) {
block 0 (v0: i32) {
  voff = i64.const 131072
  vlen = i64.const 65536
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (voff, vlen, vprot)
  vaddr = i64.const 143367
  vmark = i64.const 424242
  i64.store vaddr vmark
  return vr
  }
}
"#;
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: GROW_SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });

    let mut host = Host::new();
    let mem_h = host.grant_memory(); // the AddressSpace cap the guest `vm_map`s through (durable)
    let init = MemLayout::image(vec![0u8; 1 << GROW_SIZE_LOG2]);
    let args = [Value::I32(mem_h)];
    let (r, left) = interp_run(&m, 0, &args, &init, GROW_RESERVED_LOG2, &mut host);
    assert_eq!(r, Ok(vec![Value::I64(0)]), "the vm_map grow succeeds");
    // The capture holds the grown region past its declared window: the marker's page is Rw.
    let mark_page = (GROWN_MARK_OFF / PAGE * PAGE) as u64;
    assert_eq!(
        left.bytes().len(),
        1 << GROW_SIZE_LOG2,
        "the head is the declared window"
    );
    let marked = |l: &MemLayout| {
        l.tail_pages()
            .into_iter()
            .find(|&(off, ..)| off == mark_page)
            .map(|(_, prot, bytes)| (prot, bytes.map(|b| b[GROWN_MARK_OFF % PAGE])))
    };
    let mark = Some((CapturedProt::Rw, Some(424242i64.to_le_bytes()[0])));
    assert_eq!(
        marked(&left),
        mark,
        "grown page captured Rw, its marker with it"
    );

    // Through the codec at the guest's real reservation: pre-v18 this was GeometryMismatch.
    let art =
        temen_snapshot::freeze_layout(&m, &left, GROW_RESERVED_LOG2, &host).expect("freeze grown");
    let mut rhost = Host::new();
    let (restored, rreserved) =
        temen_snapshot::restore_layout(&art, &m, &mut rhost).expect("restore grown");
    assert_eq!(
        rreserved, GROW_RESERVED_LOG2,
        "the mask domain survives the codec"
    );
    assert_eq!(
        held(&restored),
        held(&left),
        "the grown window survives the codec"
    );
    assert_eq!(
        marked(&restored),
        mark,
        "the grown-region marker survives serialize/restore"
    );
}

// Stores to page 5 (`RO_OFF`), then returns 0. With that page restored `Ro` the store faults;
// with it `Rw` the store succeeds.
const STORE_SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 20480
  v2 = i64.const 99
  i64.store v1 v2
  v3 = i64.const 0
  return v3
  }
}
"#;

#[test]
fn restore_re_establishes_ro_so_a_thawed_write_faults() {
    let mut m = temen_text::parse_module(STORE_SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let mut host = Host::new();
    let _ = host.grant_clock();

    // Freeze a window whose page 5 is read-only, then restore it.
    let window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; WINDOW / PAGE];
    prots[RO_OFF / PAGE] = PageProt::Ro;
    let art = freeze_with_prots(&m, &window, &prots, SIZE_LOG2, &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots, _) = restore_with_prots(&art, &m, &mut rhost).expect("restore");

    // Thaw with the restored protections: the store into the Ro page faults.
    let thawed = MemLayout::from_dense(rwin.clone(), &rprots, WINDOW as u64);
    let (faulted, _) = interp_run(&m, 0, &[Value::I32(0)], &thawed, SIZE_LOG2, &mut rhost);
    assert!(faulted.is_err(), "a store to a restored Ro page must fault");

    // Control: the same window/guest with no protections re-established — the store succeeds.
    let mut host2 = Host::new();
    let bare = MemLayout::image(rwin);
    let (ok, _) = interp_run(&m, 0, &[Value::I32(0)], &bare, SIZE_LOG2, &mut host2);
    assert_eq!(
        ok,
        Ok(vec![Value::I64(0)]),
        "without the Ro protection the store succeeds"
    );
}

#[test]
fn jit_re_establishes_ro_so_a_thawed_write_faults() {
    use core::ffi::c_void;
    use temen_jit::{
        compile_and_run_capture_reserved_with_host_prots, JitOutcome, TrapKind, WindowProt,
    };

    let mut m = temen_text::parse_module(STORE_SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let mut host = Host::new();
    let _ = host.grant_clock();

    // Freeze a window whose page 5 is read-only, then restore it.
    let window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; WINDOW / PAGE];
    prots[RO_OFF / PAGE] = PageProt::Ro;
    let art = freeze_with_prots(&m, &window, &prots, SIZE_LOG2, &host).expect("freeze");
    let mut rhost = Host::new();
    let (rwin, rprots, _) = restore_with_prots(&art, &m, &mut rhost).expect("restore");
    let jit_prots: Vec<WindowProt> = rprots
        .iter()
        .map(|p| match p {
            PageProt::Ro => WindowProt::Ro,
            PageProt::Unmapped => WindowProt::Unmapped,
            PageProt::Rw | PageProt::Backed { .. } => WindowProt::Rw,
        })
        .collect();

    // Thaw on the JIT with the restored protections: the store into the Ro page faults (real
    // mprotect/VirtualProtect → guard → MemoryFault), exactly as on the interpreter above.
    let mut h = Host::new();
    let (faulted, _) = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &rwin,
        &jit_prots,
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut h as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert!(
        matches!(faulted, JitOutcome::Trapped(TrapKind::MemoryFault)),
        "a JIT store to a restored Ro page must fault, got {faulted:?}"
    );

    // Control: no protections re-established — the same store succeeds on the JIT.
    let mut h2 = Host::new();
    let (ok, _) = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &rwin,
        &[], // all Rw
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut h2 as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    assert!(
        matches!(ok, JitOutcome::Returned(_)),
        "without the Ro protection the JIT store succeeds, got {ok:?}"
    );
}

#[test]
fn jit_capture_matches_interp_for_a_readonly_segment() {
    use core::ffi::c_void;
    use temen_jit::compile_and_run_capture_reserved_with_host_prots;

    let mut m = temen_text::parse_module(SRC).expect("parse"); // `data ro 20480 "ABCD"`
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let init = vec![0u8; WINDOW];

    // Interpreter capture (reads its software page map).
    let mut hi = Host::new();
    let _ = hi.grant_clock();
    let image = MemLayout::image(init.clone());
    let caps_i = interp_run(&m, 0, &[Value::I32(0)], &image, SIZE_LOG2, &mut hi)
        .1
        .dense_prots();

    // JIT capture: run, then reconstruct the protections from the host (data segments + cap map).
    let mut hj = Host::new();
    let _ = hj.grant_clock();
    let _ = compile_and_run_capture_reserved_with_host_prots(
        &m,
        0,
        &[0i64],
        &init,
        &[],
        SIZE_LOG2,
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut c_void,
    )
    .expect("jit compiles");
    let caps_j = hj
        .capture_window_layout(
            &m.data,
            WINDOW as u64,
            temen_ir::module_null_guard(),
            init,
            Vec::new(),
        )
        .dense_prots();

    assert_eq!(caps_i[RO_OFF / PAGE], CapturedProt::Ro);
    assert_eq!(
        caps_j[RO_OFF / PAGE],
        CapturedProt::Ro,
        "JIT capture reports the readonly segment as Ro"
    );
    assert_eq!(
        caps_i, caps_j,
        "interp and JIT capture the same protection map"
    );
}

#[test]
fn jit_capture_overlays_runtime_protect_over_the_default() {
    // The runtime page-state map (`cap_pages`, populated by Memory-cap map/unmap/protect)
    // overrides the default — exercised directly here (page 0 → Unmapped) so the merge is
    // covered without a Memory-cap-using guest. Page 0 maps to host page 0 on any host page size.
    let mut h = Host::new();
    let map = h.cap_window_pages(0);
    map.lock().unwrap().insert(0, temen_ir::PageState::Unmapped);
    let caps = h
        .capture_window_layout(&[], WINDOW as u64, 0, vec![0; WINDOW], Vec::new())
        .dense_prots();
    assert_eq!(
        caps[0],
        CapturedProt::Unmapped,
        "a runtime cap_pages entry overrides the page default"
    );
    assert_eq!(
        caps[WINDOW / PAGE - 1],
        CapturedProt::Rw,
        "untouched pages stay Rw"
    );
}

/// #1700: a page grown past the 256 KiB escape-oracle span still rides the capture. The capture ran
/// to `max(mapped, 256 KiB)`, so a guest that `vm_map`ped at 512 KiB lost the page (bytes and
/// protection) and a freeze of it silently dropped live memory. #2427: and the capture holds that
/// page alone past the declared window, not the span below it.
#[test]
fn a_page_grown_past_the_oracle_span_is_captured() {
    const RESERVED_LOG2: u8 = 20; // 1 MiB reservation
    const GROWN: usize = 512 << 10; // the grown page, past 256 KiB
    const MARK_OFF: usize = GROWN + 7;
    let src = format!(
        "memory 17 shadow 16448 65536
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  voff = i64.const {GROWN}
  vlen = i64.const {PAGE}
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (voff, vlen, vprot)
  vaddr = i64.const {MARK_OFF}
  vmark = i64.const 424242
  i64.store vaddr vmark
  return vr
  }}
}}
"
    );
    let m = temen_text::parse_module(&src).expect("parse");
    let mut host = Host::new();
    let mem_h = host.grant_memory();
    let init = MemLayout::image(vec![0u8; 1 << 17]);
    let (r, left) = interp_run(&m, 0, &[Value::I32(mem_h)], &init, RESERVED_LOG2, &mut host);
    assert_eq!(r, Ok(vec![Value::I64(0)]), "the vm_map grow succeeds");
    let (head, _, tail) = held(&left);
    assert_eq!(head.len(), 1 << 17, "the head is the declared window");
    // The host page the grow committed, in codec pages: the grown one first.
    let host_page = temen_interp::host_page_size();
    assert_eq!(tail.len() as u64, host_page / PAGE as u64, "{tail:?}");
    let (off, prot, bytes) = tail[0];
    assert_eq!(
        (off, prot),
        (GROWN as u64, CapturedProt::Rw),
        "grown page captured Rw"
    );
    let bytes = bytes.expect("the marker's page has contents");
    assert_eq!(&bytes[MARK_OFF - GROWN..][..8], &424242i64.to_le_bytes());
    assert_eq!(
        left.byte_len() as u64,
        (1 << 17) + host_page,
        "nothing of the gap"
    );
}

/// #2427: a guest that maps one page 64 GiB into its 128 GiB reservation and writes a marker there
/// (entry 0), and one that loads the marker back (entry 1).
const FAR_SRC: &str = "memory 17 shadow 16448 65536
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 68719476736
  vlen = i64.const 16384
  vrw = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (va, vlen, vrw)
  vm = i64.const 424242
  i64.store va vm
  return vr
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 68719476736
  vl = i64.load va
  return vl
  }
}
";

/// #2427 — a capture holds the pages a window holds past its declared size, apart, on both engines:
/// a page mapped 64 GiB up is one page of the capture and of the artifact, not the span below it
/// (which the capture used to allocate whole), and a window thawed from it on either engine holds
/// that page alone, its marker and all.
#[test]
fn a_far_page_rides_the_artifact_alone_on_both_engines() {
    if !temen_jit::fiber_supported() {
        return;
    }
    let m = temen_text::parse_module(FAR_SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let reserved = temen_ir::DEFAULT_RESERVED_LOG2;
    let image = MemLayout::image(vec![0u8; WINDOW]);

    let mut hi = Host::new();
    let ih = hi.grant_memory();
    let (ir, ilayout) = interp_run(&m, 0, &[Value::I32(ih)], &image, reserved, &mut hi);
    assert_eq!(ir, Ok(vec![Value::I64(0)]), "the far map succeeds");
    assert_eq!(
        ilayout.page_map().grown_bytes(),
        16384,
        "the capture holds the page alone"
    );
    assert!(
        ilayout.byte_len() <= WINDOW + 16384,
        "{} bytes",
        ilayout.byte_len()
    );

    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_memory();
    let (jo, jlayout) =
        temen_run::jit_cap_run(&m, 0, &[jh as i64], &image, reserved, 0, &mut hj, None)
            .expect("jit");
    assert_eq!(jo, temen_jit::JitOutcome::Returned(vec![0]));
    assert!(
        held(&jlayout) == held(&ilayout),
        "the JIT captures what the interpreter does"
    );

    let art = temen_snapshot::freeze_layout(&m, &jlayout, reserved, &Host::new()).expect("freeze");
    assert!(
        art.len() < 1 << 20,
        "the artifact holds the page alone: {} bytes",
        art.len()
    );
    let (restored, _) =
        temen_snapshot::restore_layout(&art, &m, &mut Host::new()).expect("restore");
    assert_eq!(
        restored.page_map().grown_bytes(),
        16384,
        "the thaw holds what the guest did"
    );

    let mut hi = Host::new();
    let ih = hi.grant_memory();
    let (ir, _) = interp_run(&m, 1, &[Value::I32(ih)], &restored, reserved, &mut hi);
    assert_eq!(
        ir,
        Ok(vec![Value::I64(424242)]),
        "the interpreter's thaw reads the marker"
    );
    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_memory();
    let (jo, _) =
        temen_run::jit_cap_run(&m, 1, &[jh as i64], &restored, reserved, 0, &mut hj, None)
            .expect("jit");
    assert_eq!(
        jo,
        temen_jit::JitOutcome::Returned(vec![424242]),
        "the JIT's thaw reads it"
    );
}

/// #1810: a guest that changes its own pages through its `AddressSpace` — a page mapped and then
/// `protect`ed read-only, one mapped and `unmap`ped, and one grown past the 256 KiB escape-oracle
/// span, all in the reserved tail, and a page of its declared memory `protect`ed read-only — plus a
/// `readonly` data segment.
const PAGE_OPS: &str = "memory 17
data ro 81920 \"abcd\"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vro = i64.const 196608
  vlen = i64.const 16384
  vrw = i32.const 3
  vrd = i32.const 1
  v1 = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (vro, vlen, vrw)
  vm1 = i64.const 55
  i64.store vro vm1
  v2 = call.cap 5 2 (i64, i64, i32) -> (i64) v0 (vro, vlen, vrd)
  vun = i64.const 229376
  v3 = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (vun, vlen, vrw)
  v4 = call.cap 5 1 (i64, i64) -> (i64) v0 (vun, vlen)
  vhi = i64.const 524288
  v5 = call.cap 5 0 (i64, i64, i32) -> (i64) v0 (vhi, vlen, vrw)
  vm2 = i64.const 424242
  i64.store vhi vm2
  vlo = i64.const 98304
  v6 = call.cap 5 2 (i64, i64, i32) -> (i64) v0 (vlo, vlen, vrd)
  va = i64.add v1 v2
  vb = i64.add v3 v4
  vc = i64.add va vb
  vd = i64.add vc v5
  ve = i64.add vd v6
  return ve
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  vz = i64.const 0
  return vz
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 196608
  vm = i64.const 7
  i64.store va vm
  vz = i64.const 0
  return vz
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 229376
  vl = i64.load va
  return vl
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 524288
  vl = i64.load va
  return vl
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 524288
  vn = i64.const 16384
  vrd = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) v0 (va, vn, vrd)
  return vr
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 98304
  vm = i64.const 7
  i64.store va vm
  vz = i64.const 0
  return vz
  }
}
";
/// `PAGE_OPS`' probes of the window its entry 0 left (#1834): 1 does nothing, 2 stores to the
/// read-only tail page, 3 loads from the unmapped one, 4 loads the grown page's marker, 5 `protect`s
/// the grown page read-only through the Memory capability (refused on a page it does not see
/// mapped), and 6 stores to the read-only page of the declared memory.
const PROBES: [u32; 6] = [1, 2, 3, 4, 5, 6];
const PAGE_OPS_RESERVED_LOG2: u8 = 20; // 1 MiB

/// #1810: the JIT's durable capture is the interpreter's — its declared window, each page held past
/// it (#2427), and the page map. It stopped at 256 KiB, so the page at 512 KiB never reached a JIT
/// freeze; and it was bytes alone, so a JIT freeze recorded every page `Rw`, losing the guest's
/// `protect` and `unmap`.
#[test]
fn jit_durable_capture_matches_interp_past_the_oracle_span() {
    if !temen_jit::fiber_supported() {
        return;
    }
    let m = temen_text::parse_module(PAGE_OPS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let init = vec![0u8; 1 << 17];

    let mut hi = Host::new();
    let ih = hi.grant_memory();
    let image = MemLayout::image(init.clone());
    let (ir, ilayout) = interp_run(
        &m,
        0,
        &[Value::I32(ih)],
        &image,
        PAGE_OPS_RESERVED_LOG2,
        &mut hi,
    );
    assert_eq!(ir, Ok(vec![Value::I64(0)]), "every page op succeeds");

    let mut hj = Host::new();
    hj.set_durable(true);
    let jh = hj.grant_memory();
    let (jo, jlayout) = temen_run::jit_cap_run(
        &m,
        0,
        &[jh as i64],
        &MemLayout::image(init.clone()),
        PAGE_OPS_RESERVED_LOG2,
        0,
        &mut hj,
        None,
    )
    .expect("jit");
    assert_eq!(jo, temen_jit::JitOutcome::Returned(vec![0]));
    assert!(held(&jlayout) == held(&ilayout), "capture");
    // The cases the page map must carry, spelled out: in the declared window, and past it, where an
    // `unmap`ped page is one the window no longer holds.
    let (_, iprots, itail) = held(&ilayout);
    for (off, want) in [
        (81920, CapturedProt::Ro), // the readonly segment
        (98304, CapturedProt::Ro), // `protect`ed read-only in the declared memory
    ] {
        assert_eq!(iprots[off / PAGE], want, "interp page at {off}");
    }
    let tail = |off: u64| itail.iter().find(|t| t.0 == off).map(|t| t.1);
    assert_eq!(
        tail(196608),
        Some(CapturedProt::Ro),
        "`protect`ed read-only"
    );
    assert_eq!(tail(229376), None, "`unmap`ped");
    assert_eq!(tail(524288), Some(CapturedProt::Rw), "grown past 256 KiB");

    // Through the codec and back: the JIT's artifact keeps what the interpreter's does.
    let art = temen_snapshot::freeze_layout(&m, &jlayout, PAGE_OPS_RESERVED_LOG2, &Host::new())
        .expect("freeze");
    let (restored, _) =
        temen_snapshot::restore_layout(&art, &m, &mut Host::new()).expect("restore");
    assert!(held(&restored) == held(&ilayout), "restored window");
}

/// #1834: a thaw on the JIT re-applies the page map the artifact carries, as the interpreter's does.
/// The JIT thaw took window bytes alone: the `protect`ed page came back writable, the `unmap`ped one
/// readable, the page grown past the declared memory was never committed (its marker lost), and the
/// host's page map started empty, so the next freeze recorded none of it.
#[test]
fn a_jit_thaw_keeps_the_page_map_the_artifact_carries() {
    if !temen_jit::fiber_supported() {
        return;
    }
    let m = temen_text::parse_module(PAGE_OPS).expect("parse");
    temen_verify::verify_module(&m).expect("verify");

    // The window entry 0 leaves, through the codec.
    let mut h = Host::new();
    h.set_durable(true);
    let mh = h.grant_memory();
    let init = MemLayout::image(vec![0u8; 1 << 17]);
    let (o, frozen) = temen_run::jit_cap_run(
        &m,
        0,
        &[mh as i64],
        &init,
        PAGE_OPS_RESERVED_LOG2,
        0,
        &mut h,
        None,
    )
    .expect("jit");
    assert_eq!(o, temen_jit::JitOutcome::Returned(vec![0]));
    let art = temen_snapshot::freeze_layout(&m, &frozen, PAGE_OPS_RESERVED_LOG2, &Host::new())
        .expect("freeze");

    let mut wrong = Vec::new();
    for entry in PROBES {
        // The interpreter's thaw: the restored window, page map and all.
        let mut hi = Host::new();
        let (rlayout, reserved) =
            temen_snapshot::restore_layout(&art, &m, &mut hi).expect("restore");
        let ih = hi.grant_memory();
        let (ir, ilayout) = interp_run(&m, entry, &[Value::I32(ih)], &rlayout, reserved, &mut hi);
        let iout = match &ir {
            Ok(v) => format!("{v:?}"),
            Err(t) => format!("{t:?}"),
        };

        // The JIT's: the restored layout.
        let mut hj = Host::new();
        hj.set_durable(true);
        let (layout, reserved) =
            temen_snapshot::restore_layout(&art, &m, &mut hj).expect("restore");
        let jh = hj.grant_memory();
        let (jo, jlayout) =
            temen_run::jit_cap_run(&m, entry, &[jh as i64], &layout, reserved, 0, &mut hj, None)
                .expect("jit");
        let jout = match jo {
            temen_jit::JitOutcome::Returned(v) => format!("{:?}", [Value::I64(v[0])]),
            temen_jit::JitOutcome::Trapped(t) => format!("{t:?}"),
            other => format!("{other:?}"),
        };
        if jout != iout {
            wrong.push(format!("entry {entry}: interp {iout}, JIT {jout}"));
        }
        // What the run leaves, as the next freeze would record it.
        if ir.is_ok() && held(&jlayout) != held(&ilayout) {
            wrong.push(format!("entry {entry}: the window the run leaves differs"));
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n"));
}
