//! Per-page protections round-trip through the §12.3 window image (Phase-2 slice 1): `Ro` and
//! `Unmapped` pages are carried in the artifact and recovered on restore, while zero `Rw` pages
//! stay elided. The next slice feeds these from / applies them to a running backend's window.

use temen_interp::{Host, StreamRole};
use temen_ir::{Memory, Module};
use temen_snapshot::{freeze, freeze_with_prots, restore_with_prots, FreezeError, PageProt};

const SIZE_LOG2: u8 = 17; // 128 KiB
const WINDOW: usize = 1 << SIZE_LOG2;
const RESERVED_LOG2: u8 = 19; // 512 KiB reservation — the mask domain a grown window lives in
const PAGE: usize = 4096;
const NPAGES: usize = WINDOW / PAGE; // 32

// A minimal module that just declares the window: freeze digests its encoded bytes and restore
// checks the geometry against `memory.size_log2`.
const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 7
  return v1
  }
}
"#;

fn module() -> Module {
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    m
}

fn host_with_durable_handles() -> Host {
    let mut h = Host::new();
    h.grant_clock();
    let _ = h.grant_stream(StreamRole::Out);
    h
}

#[test]
fn page_protections_round_trip_through_the_window_image() {
    let m = module();
    let host = host_with_durable_handles();

    let mut window = vec![0u8; WINDOW];
    let mut prots = vec![PageProt::Rw; NPAGES];
    window[0] = 0xAB; // page 0: Rw, non-zero
    window[PAGE - 1] = 0xCD;
    // page 3 left zero `Rw` → elided, restores as zero.
    window[5 * PAGE + 10] = 0x11; // page 5: Ro, non-zero
    prots[5] = PageProt::Ro;
    prots[6] = PageProt::Ro; // page 6: Ro, all-zero → must still come back Ro
    window[9 * PAGE + 2] = 0x99; // page 9: Unmapped — content is NOT stored
    prots[9] = PageProt::Unmapped;

    let art = freeze_with_prots(&m, &window, &prots, SIZE_LOG2, &host).expect("freeze");

    let mut rhost = Host::new();
    let (rwin, rprots, rreserved) = restore_with_prots(&art, &m, &mut rhost).expect("restore");
    assert_eq!(rreserved, SIZE_LOG2, "flat window: reserved == declared");

    // Protections recovered exactly.
    assert_eq!(rprots.len(), NPAGES);
    assert_eq!(rprots[0], PageProt::Rw);
    assert_eq!(rprots[3], PageProt::Rw, "elided page defaults to Rw");
    assert_eq!(rprots[5], PageProt::Ro);
    assert_eq!(rprots[6], PageProt::Ro, "a zero Ro page is still preserved");
    assert_eq!(rprots[9], PageProt::Unmapped);

    // Bytes recovered for Rw/Ro; an Unmapped page restores zero (its content is never stored).
    assert_eq!(rwin[0], 0xAB);
    assert_eq!(rwin[PAGE - 1], 0xCD);
    assert_eq!(rwin[5 * PAGE + 10], 0x11);
    assert!(
        rwin[6 * PAGE..7 * PAGE].iter().all(|&b| b == 0),
        "zero Ro page restores zero"
    );
    assert!(
        rwin[9 * PAGE..10 * PAGE].iter().all(|&b| b == 0),
        "Unmapped page drops its pre-freeze content"
    );

    // §12.6 canonical: re-serializing the restored image at the same safepoint is byte-identical.
    assert_eq!(
        freeze_with_prots(&m, &rwin, &rprots, SIZE_LOG2, &host).expect("re-freeze"),
        art,
        "restore → re-serialize reproduces the artifact"
    );
}

#[test]
fn a_vm_map_grown_extent_round_trips() {
    // #1154 (invariant 14, durability axis): a guest that `vm_map`-grew its window past its declared
    // size (`SIZE_LOG2` = 128 KiB) to a 192-KiB committed extent, inside a 512-KiB reservation. Pre-v18
    // this was `GeometryMismatch` (the codec required `mapped == 1 << declared`); v18 carries the grown
    // `mapped` + `reserved_log2` and round-trips it — the grown pages ride the image as `Rw` above the
    // declared window, exactly as an in-process `snapshot_window`/`seed_pages` restore does.
    let m = module();
    let host = host_with_durable_handles();

    const MAPPED: usize = WINDOW + 16 * PAGE; // 128 KiB + 64 KiB = 192 KiB committed (48 pages)
    const GROWN0: usize = WINDOW / PAGE; // first grown page (page 32)
    let mut window = vec![0u8; MAPPED];
    let mut prots = vec![PageProt::Rw; MAPPED / PAGE]; // committed prefix default Rw
                                                       // A marker in a grown page proves grown content survives the codec.
    window[(GROWN0 + 3) * PAGE + 7] = 0x5A;
    // A read-only grown page (a `protect`ed grown allocation) and an Unmapped hole inside the grown
    // region (a `vm_unmap` between two grows) — both must survive.
    window[(GROWN0 + 5) * PAGE + 1] = 0x99;
    prots[GROWN0 + 5] = PageProt::Ro;
    prots[GROWN0 + 8] = PageProt::Unmapped;
    // A declared-prefix Ro page too (rodata), so the mix spans both regions.
    prots[2] = PageProt::Ro;

    let art = freeze_with_prots(&m, &window, &prots, RESERVED_LOG2, &host).expect("freeze grown");

    let mut rhost = Host::new();
    let (rwin, rprots, rreserved) =
        restore_with_prots(&art, &m, &mut rhost).expect("restore grown");

    assert_eq!(rreserved, RESERVED_LOG2, "the mask domain survives");
    assert_eq!(rwin.len(), MAPPED, "the committed extent survives");
    assert_eq!(rprots.len(), MAPPED / PAGE);
    assert_eq!(
        rwin[(GROWN0 + 3) * PAGE + 7],
        0x5A,
        "grown-page content restored"
    );
    assert_eq!(rprots[GROWN0 + 5], PageProt::Ro, "grown Ro page restored");
    assert_eq!(
        rwin[(GROWN0 + 5) * PAGE + 1],
        0x99,
        "grown Ro page bytes restored"
    );
    assert_eq!(
        rprots[GROWN0 + 8],
        PageProt::Unmapped,
        "grown Unmapped hole restored"
    );
    assert!(
        rwin[(GROWN0 + 8) * PAGE..(GROWN0 + 9) * PAGE]
            .iter()
            .all(|&b| b == 0),
        "an Unmapped grown page carries no bytes"
    );
    assert_eq!(rprots[2], PageProt::Ro, "declared-prefix Ro still restored");

    // §12.6 canonical: restore → re-serialize at the same reservation is byte-identical.
    assert_eq!(
        freeze_with_prots(&m, &rwin, &rprots, RESERVED_LOG2, &host).expect("re-freeze"),
        art,
        "a grown artifact re-serializes byte-identically"
    );
}

#[test]
fn freeze_rejects_a_committed_extent_past_the_reservation() {
    // The nested-chain geometry gate: `mapped` (the window image) must fit the mask domain it grew
    // within. A window longer than `1 << reserved_log2` is corrupt — freeze fails closed.
    let m = module();
    let host = host_with_durable_handles();
    let window = vec![0u8; 1 << (RESERVED_LOG2 + 1)]; // twice the reservation
    let prots = vec![PageProt::Rw; window.len() / PAGE];
    assert!(matches!(
        freeze_with_prots(&m, &window, &prots, RESERVED_LOG2, &host),
        Err(FreezeError::WindowGeometry(_))
    ));
}

#[test]
fn freeze_rejects_a_wrong_length_prot_map() {
    let m = module();
    let host = host_with_durable_handles();
    let window = vec![0u8; WINDOW];
    let prots = vec![PageProt::Rw; NPAGES - 1]; // one short
    assert!(matches!(
        freeze_with_prots(&m, &window, &prots, SIZE_LOG2, &host),
        Err(FreezeError::ProtCount { pages: NPAGES, prots: p }) if p == NPAGES - 1
    ));
}

#[test]
fn flat_freeze_equals_an_all_rw_prot_map() {
    // The flat convenience must equal an explicit all-`Rw` map — the back-compat the
    // cross-backend (`durable_jit`) path relies on.
    let m = module();
    let host = host_with_durable_handles();
    let mut window = vec![0u8; WINDOW];
    window[100] = 0x42;
    let all_rw = [PageProt::Rw; NPAGES];
    assert_eq!(
        freeze(&m, &window, &host).expect("flat"),
        freeze_with_prots(&m, &window, &all_rw, SIZE_LOG2, &host).expect("explicit"),
    );
}
