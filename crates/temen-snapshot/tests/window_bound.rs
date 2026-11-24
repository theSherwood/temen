//! #3100, #3137 — what a restore of an untrusted artifact allocates, and what it accepts: a window's
//! head past its declared memory is held as one span (nothing is allocated for a page the artifact
//! carries no bytes for), every window is charged against the memory its thaw holds it under, and a
//! reservation is bounded by the one the restoring host builds.

use temen_interp::{Host, MemLayout, StreamRole};
use temen_ir::{Memory, Module};
use temen_snapshot::{
    freeze_layout, freeze_with_prots, restore_into, restore_layout, restore_with_prots, PageProt,
    RestoreError, Restoring,
};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

const SIZE_LOG2: u8 = 17; // 128 KiB
const WINDOW: usize = 1 << SIZE_LOG2;
const PAGE: usize = 4096;
const W: u64 = WINDOW as u64;
const P: u64 = PAGE as u64;

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
        shadow: Some(TEST_ARENA),
    });
    m
}

fn host() -> Host {
    let mut h = Host::new();
    h.grant_clock();
    let _ = h.grant_stream(StreamRole::Out);
    h
}

/// The default reservation a restore of [`module`] builds its window in.
fn default_reserve() -> u8 {
    temen_ir::window_reserved_log2(SIZE_LOG2)
}

/// A dense image of `pages` pages over the declared window: page 3 nonzero and, when it runs past the
/// declared window, the first page past it nonzero and the second `Ro`.
fn dense(pages: usize, reserved_log2: u8) -> Vec<u8> {
    let mut window = vec![0u8; pages * PAGE];
    let mut prots = vec![PageProt::Rw; pages];
    window[3 * PAGE] = 0x33;
    if pages > WINDOW / PAGE + 1 {
        window[WINDOW + 7] = 0x77;
        prots[WINDOW / PAGE + 1] = PageProt::Ro;
    }
    freeze_with_prots(&module(), &window, &prots, reserved_log2, &host()).expect("freeze")
}

/// A layout holding the declared window and one page far past it (with a byte), frozen under
/// `reserved_log2`: about six bytes of artifact name the page.
fn far_page(at: u64, reserved_log2: u8) -> Vec<u8> {
    let mut l = MemLayout::from_parts(vec![0u8; WINDOW], P, W, &[(at, 1)]).expect("a page list");
    let mut bytes = vec![0u8; PAGE];
    bytes[0] = 0x5A;
    l.hold(at, PageProt::Rw, Some(bytes));
    freeze_layout(&module(), &l, reserved_log2, &host()).expect("freeze")
}

/// `art` with its header's reservation and head (`mapped`) rewritten, every other byte kept: what a
/// hostile artifact can say in a few bytes. The wire header is 16 bytes; Section 0 follows it.
fn rehead(art: &[u8], reserved_log2: u8, mapped: u64) -> Vec<u8> {
    let (tag, len) = (art[16], art[17] as usize);
    assert_eq!(tag, 0, "Section 0 first");
    let body = &art[18..18 + len];
    let mut at = 33; // digest, reservation
    let mut uleb = |b: &[u8]| {
        let mut n = 0;
        while b[at + n] & 0x80 != 0 {
            n += 1;
        }
        at += n + 1;
    };
    uleb(body); // the old head
    let mut nb = body[..32].to_vec();
    nb.push(reserved_log2);
    let mut v = mapped;
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        nb.push(if v == 0 { b } else { b | 0x80 });
        if v == 0 {
            break;
        }
    }
    nb.extend_from_slice(&body[at..]);
    assert!(nb.len() < 128);
    let mut out = art[..17].to_vec();
    out.push(nb.len() as u8);
    out.extend_from_slice(&nb);
    out.extend_from_slice(&art[18 + len..]);
    out
}

/// A refused restore leaves `h` as a fresh host is.
fn untouched(h: &Host, what: &str) {
    assert_eq!(h.restore_footprint(), host().restore_footprint(), "{what}");
}

/// #3100 — a ~100-byte artifact asking for a 2^63 reservation and a 2^62 head. On `257f553` it
/// asked the allocator for 2^62 bytes before any section was checked (an abort, or an OOM kill).
#[test]
fn a_header_asking_for_2_to_the_63_is_refused_before_anything_is_allocated() {
    let m = module();
    let zero = vec![0u8; WINDOW];
    let prots = vec![PageProt::Rw; WINDOW / PAGE];
    let flat = freeze_with_prots(&m, &zero, &prots, SIZE_LOG2, &host()).expect("freeze");
    let art = rehead(&flat, 63, 1 << 62);
    assert!(art.len() < 200, "a short artifact: {}", art.len());
    let want = RestoreError::ReservationRefused {
        asked: 63,
        granted: default_reserve(),
    };
    let mut h = host();
    assert_eq!(restore_layout(&art, &m, &mut h).err(), Some(want.clone()));
    untouched(&h, "the layout entry");
    assert_eq!(
        restore_with_prots(&art, &m, &mut h).map(|_| ()).err(),
        Some(want),
        "the dense entry"
    );
    untouched(&h, "the dense entry");
    // An entry that does build a 2^63 reservation still holds the window to its budget: 2^62 bytes
    // past the declared window are not what its 4 GiB default lets a guest commit.
    let wide = Restoring {
        reserved_log2: 63,
        flat: false,
    };
    assert!(matches!(
        restore_into(&art, &m, &mut h, wide),
        Err(RestoreError::WindowOverBudget { .. })
    ));
    untouched(&h, "a wide reservation");
}

/// #3100 — one page at the top of a reservation the restoring host builds. On `257f553` the dense
/// entry resized its image to the page (128 GiB here) after `apply` had written the host; the layout
/// entry holds it as one page, charged one page.
#[test]
fn a_far_page_is_refused_by_span_for_a_flat_holder_and_held_alone_for_a_sparse_one() {
    let m = module();
    let r = default_reserve();
    let at = (1u64 << r) - P;
    let art = far_page(at, r);
    let mut h = host();
    assert!(matches!(
        restore_with_prots(&art, &m, &mut h),
        Err(RestoreError::WindowOverBudget { need, .. }) if need == at + P,
    ));
    untouched(&h, "the dense entry");
    let flat = Restoring {
        reserved_log2: r,
        flat: true,
    };
    assert!(matches!(
        restore_into(&art, &m, &mut h, flat),
        Err(RestoreError::WindowOverBudget { .. })
    ));
    untouched(&h, "a flat holder");
    let (layout, _) = restore_layout(&art, &m, &mut h).expect("the sparse entry holds one page");
    assert_eq!(layout.byte_len(), WINDOW + PAGE, "the head and the page");
    assert_eq!(
        layout.page_map().grown_bytes(),
        temen_interp::host_page_size()
    );
}

/// #3137 — a head of 2^30 with four stored pages past the declared window. On `257f553` the layout
/// held 2^30 bytes and one map entry per page past the declared window (2^18 of them). Here it
/// holds the declared window, the four pages, and the span as one range; and it re-freezes to the
/// artifact it came from.
#[test]
fn a_sparse_large_head_is_held_as_a_span_and_re_freezes_byte_identically() {
    let m = module();
    let reserved = 31;
    let mut l = MemLayout::from_parts(vec![0u8; WINDOW], P, W, &[]).expect("a page list");
    for k in 1..=4u64 {
        let mut page = vec![0u8; PAGE];
        page[1] = k as u8;
        l.hold(k << 27, PageProt::Rw, Some(page));
    }
    let art = rehead(
        &freeze_layout(&m, &l, reserved, &host()).expect("freeze"),
        reserved,
        1 << 30,
    );
    let mut h = host();
    let (restored, got) = restore_layout(&art, &m, &mut h).expect("restore");
    assert_eq!(got, reserved);
    assert_eq!(
        restored.bytes().len(),
        WINDOW,
        "the head is the declared window"
    );
    assert_eq!(restored.byte_len(), WINDOW + 4 * PAGE, "and the four pages");
    assert_eq!(restored.page_map().held_span(), W..1 << 30);
    assert_eq!(
        restored.tail_pages().len(),
        4,
        "no entry per page of the span"
    );
    assert_eq!(restored.span(), 1 << 30);
    let back = freeze_layout(&m, &restored, reserved, &h).expect("re-freeze");
    assert_eq!(back, art, "the canonical round trip");
}

/// #3137 — a dense artifact whose head runs past the declared window (a flat grown window's freeze)
/// round-trips through both forms: the layout holds the span, its dense form is the frozen window,
/// and a re-freeze of either is byte-identical.
#[test]
fn a_dense_head_past_the_declared_window_round_trips_through_both_forms() {
    let m = module();
    let pages = WINDOW / PAGE + 16;
    let art = dense(pages, 20);
    let mut h = host();
    let (layout, _) = restore_layout(&art, &m, &mut h).expect("restore");
    assert_eq!(layout.bytes().len(), WINDOW);
    assert_eq!(layout.page_map().held_span(), W..(pages * PAGE) as u64);
    assert_eq!(freeze_layout(&m, &layout, 20, &h).expect("re-freeze"), art);
    let (window, prots) = layout.to_dense();
    assert_eq!(window.len(), pages * PAGE);
    assert_eq!((window[3 * PAGE], window[WINDOW + 7]), (0x33, 0x77));
    assert_eq!(
        prots[WINDOW / PAGE + 1],
        PageProt::Ro,
        "a listed page keeps its protection"
    );
    assert_eq!(
        prots[WINDOW / PAGE + 2],
        PageProt::Rw,
        "an unlisted one is held read-write"
    );
    let (dw, dp, _) = restore_with_prots(&art, &m, &mut host()).expect("dense restore");
    assert_eq!((dw, dp), (window.clone(), prots.clone()));
    assert_eq!(
        freeze_with_prots(&m, &window, &prots, 20, &h).expect("re-freeze"),
        art
    );
}

/// #3100 — the calibration: an artifact restores under exactly the memory its window holds (its
/// declared window, and the 16 pages past it: zero, listed `Ro`, or with bytes, each one a page a
/// running guest's `map` was charged for) and is refused one page under it.
#[test]
fn a_window_restores_under_exactly_its_budget_and_not_a_page_less() {
    let m = module();
    let need = (WINDOW + 16 * PAGE) as i64;
    let art = dense(WINDOW / PAGE + 16, 20);
    let mut h = host();
    h.set_mem_grant(need - PAGE as i64);
    assert_eq!(
        restore_layout(&art, &m, &mut h).err(),
        Some(RestoreError::WindowOverBudget {
            need: need as u64,
            room: (need - PAGE as i64) as u64,
        })
    );
    let mut h = host();
    h.set_mem_grant(need);
    restore_layout(&art, &m, &mut h).expect("restores under exactly its budget");
}

/// #3100 — a window reserved above the default still freezes (the shared geometry rule has no cap)
/// and restores where the restoring entry builds that reservation; the default entry refuses it.
#[test]
fn a_window_reserved_at_2_to_the_41_freezes_and_restores_where_its_entry_builds_it() {
    let m = module();
    let art = dense(WINDOW / PAGE, 41);
    let mut h = host();
    assert_eq!(
        restore_layout(&art, &m, &mut h).err(),
        Some(RestoreError::ReservationRefused {
            asked: 41,
            granted: default_reserve(),
        })
    );
    let wide = Restoring {
        reserved_log2: 41,
        flat: false,
    };
    let (_, got) = restore_into(&art, &m, &mut h, wide).expect("restores at 2^41");
    assert_eq!(got, 41);
}
