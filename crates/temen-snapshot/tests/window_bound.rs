//! #3100, #3137 — what a restore of an untrusted artifact allocates, and what it accepts: a window's
//! head past its declared memory is held as one span (nothing is allocated for a page the artifact
//! carries no bytes for), every window is charged against the memory its thaw holds it under, and a
//! reservation is bounded by the one the restoring host builds.

use temen_interp::{Host, MemLayout, StreamRole};
use temen_ir::{Memory, Module};
use temen_snapshot::{
    freeze_layout, freeze_with_prots, restore_into, restore_layout, restore_with_prots, Holder,
    PageProt, RestoreError, Restoring,
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

/// The refusal of a root window of `window` bytes (its declared memory and its growth) under a `mem`
/// grant of `grant`. The root's declared memory is what the embedder chose to run, charged past any
/// ceiling at its activation (INVARIANTS #3), so the refusal names the growth against what the grant
/// has left past it.
fn over(grant: u64, window: u64) -> RestoreError {
    RestoreError::WindowOverBudget {
        need: window - W,
        room: grant - W,
    }
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
        holder: Holder::Sparse,
    };
    let hp = temen_interp::host_page_size();
    assert_eq!(
        restore_into(&art, &m, &mut h, wide).map(|_| ()).err(),
        Some(over(
            temen_interp::DEFAULT_MEM,
            ((1u64 << 62) - W).div_ceil(hp) * hp + W
        ))
    );
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
    let over = over(temen_interp::DEFAULT_MEM, at + P);
    assert_eq!(
        restore_with_prots(&art, &m, &mut h).map(|_| ()).err(),
        Some(over.clone()),
        "the dense entry: refused by the budget, not the allocator"
    );
    untouched(&h, "the dense entry");
    let flat = Restoring::of(&m, Holder::Flat);
    assert_eq!(
        restore_into(&art, &m, &mut h, flat).map(|_| ()).err(),
        Some(over)
    );
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

/// `art` restored on a host whose `mem` grant is `mem`: what the restore answered.
fn under(art: &[u8], mem: u64) -> Result<(), RestoreError> {
    let mut h = host();
    h.set_mem_grant(mem as i64);
    let got = restore_layout(art, &module(), &mut h).map(|_| ());
    if got.is_err() {
        untouched(&h, "refused");
    }
    got
}

/// #3100 — the calibration: an artifact restores under exactly the memory its window holds, in this
/// host's pages (its declared window, and the 64 KiB of head span past it: zero, listed `Ro`, or with
/// bytes, each one a page a running guest's `map` was charged for), and is refused one host page
/// under it.
#[test]
fn a_window_restores_under_exactly_its_budget_and_not_a_page_less() {
    let hp = temen_interp::host_page_size();
    let need = W + 16 * P;
    let art = dense(WINDOW / PAGE + 16, 20);
    assert_eq!(under(&art, need - hp), Err(over(need - hp, need)));
    assert_eq!(under(&art, need), Ok(()));
}

/// The id a fresh host gives the one region it holds, and that host.
fn with_region() -> (Host, u32) {
    let mut h = host();
    h.grant_shared_region(PAGE);
    let region = h
        .capture_durable_handles()
        .expect("durable")
        .iter()
        .find_map(|x| x.binding.region())
        .expect("the region's handle");
    (h, region)
}

/// #3100 — growth past the held span is the pages a running guest was charged for, by the rule the
/// thaw charges by (`held_tail`): an `Rw` page with bytes, and an `Unmapped` one that keeps its
/// contents (#1909), are each the window's own memory; a `Backed` page is its region's, charged to
/// `channel`. The window restores under exactly its declared memory and those two host pages, and is
/// refused one host page under.
#[test]
fn growth_past_the_span_counts_the_pages_a_running_guest_was_charged_for() {
    let hp = temen_interp::host_page_size();
    let (fh, region) = with_region();
    let mut l = MemLayout::from_parts(vec![0u8; WINDOW], P, W, &[]).expect("a page list");
    let mut bytes = vec![0u8; PAGE];
    bytes[0] = 1;
    l.hold(W + 4 * hp, PageProt::Rw, Some(bytes.clone()));
    l.hold(W + 8 * hp, PageProt::Unmapped, Some(bytes));
    let backed = PageProt::Backed {
        region,
        off: 0,
        writable: true,
    };
    l.hold(W + 12 * hp, backed, None);
    let art = freeze_layout(&module(), &l, 20, &fh).expect("freeze");
    let need = W + 2 * hp;
    assert_eq!(under(&art, need - hp), Err(over(need - hp, need)));
    assert_eq!(under(&art, need), Ok(()));
}

/// #3100 — a flat holder is charged its span whatever its pages' state: 256 `Backed` pages from the
/// declared window up, every one aliasing the same page of one carried 4 KiB region, cost the
/// artifact about 7 bytes each and a flat holder 1 MiB of buffer. Sparse, they are the region's (it
/// restores under a grant of the declared window alone); dense or flat, they are refused under that
/// grant, the host as it was.
#[test]
fn a_flat_holder_is_charged_its_span_whatever_its_pages() {
    const N: u64 = 256;
    let (fh, region) = with_region();
    let mut l = MemLayout::from_parts(vec![0u8; WINDOW], P, W, &[]).expect("a page list");
    let backed = PageProt::Backed {
        region,
        off: 0,
        writable: true,
    };
    for k in 0..N {
        l.hold(W + k * P, backed, None);
    }
    let m = module();
    let art = freeze_layout(&m, &l, 24, &fh).expect("freeze");
    // The region's 4 KiB, and about 6 bytes per page.
    assert!(
        art.len() < PAGE + 8 * N as usize,
        "{} bytes of artifact",
        art.len()
    );
    assert_eq!(under(&art, W), Ok(()), "sparse: the pages are the region's");
    let want = over(W, W + N * P);
    let mut h = host();
    h.set_mem_grant(W as i64);
    assert_eq!(
        restore_with_prots(&art, &m, &mut h).map(|_| ()).err(),
        Some(want.clone()),
        "dense"
    );
    untouched(&h, "dense");
    let flat = Restoring::of(&m, Holder::Flat);
    assert_eq!(
        restore_into(&art, &m, &mut h, flat).map(|_| ()).err(),
        Some(want),
        "flat"
    );
    untouched(&h, "flat");
}

/// #3100 — a dense image the budget admits and the allocator cannot supply is its own refusal,
/// before anything is written to the host: one page at 2^55 under an unbounded grant.
#[test]
fn a_dense_image_no_allocator_serves_is_refused_before_the_host_is_written() {
    let m = module();
    let at = (1u64 << 55) - P;
    let art = far_page(at, 56);
    let mut h = host();
    h.set_mem_grant(-1);
    let dense = Restoring {
        reserved_log2: 56,
        holder: Holder::Dense,
    };
    assert_eq!(
        restore_into(&art, &m, &mut h, dense).map(|_| ()).err(),
        Some(RestoreError::DenseUnallocated { span: at + P })
    );
    let mut fresh = host();
    fresh.set_mem_grant(-1);
    assert_eq!(h.restore_footprint(), fresh.restore_footprint());
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
        holder: Holder::Sparse,
    };
    let (_, got) = restore_into(&art, &m, &mut h, wide).expect("restores at 2^41");
    assert_eq!(got, 41);
}

/// A [`module`] whose declared memory is `1 << size_log2`.
fn module_of(size_log2: u8) -> Module {
    let mut m = module();
    m.memory = Some(Memory {
        size_log2,
        shadow: None,
    });
    m
}

/// #3100 — a module that declares less than a page (`memory 10`: 1 KiB). Its head is one page, as
/// the geometry rule lets it be (`declared <= mapped`): an honest dense artifact with bytes in page 0
/// round-trips through both forms, and a hostile one listing page 0 is restored or refused, never a
/// panic. (On `2d67be1` both indexed past a head sized by the declared 1 KiB.)
#[test]
fn a_module_declaring_less_than_a_page_round_trips() {
    let m = module_of(10);
    let mut window = vec![0u8; PAGE];
    window[7] = 9;
    let art = freeze_with_prots(&m, &window, &[PageProt::Rw], 12, &host()).expect("freeze");
    let (w, p, _) = restore_with_prots(&art, &m, &mut host()).expect("dense restore");
    assert_eq!((w, p), (window.clone(), vec![PageProt::Rw]));
    let (layout, _) = restore_layout(&art, &m, &mut host()).expect("layout restore");
    assert_eq!(layout.bytes(), &window[..], "the head is the page");
    assert_eq!(
        freeze_layout(&m, &layout, 12, &host()).expect("re-freeze"),
        art
    );
    // A head of two pages, the second listed `Ro`: past the declared KiB, in the head span.
    let mut window = vec![0u8; 2 * PAGE];
    window[PAGE + 1] = 1;
    let art =
        freeze_with_prots(&m, &window, &[PageProt::Rw, PageProt::Ro], 13, &host()).expect("freeze");
    let (w, p, _) = restore_with_prots(&art, &m, &mut host()).expect("restore");
    assert_eq!((w, p), (window, vec![PageProt::Rw, PageProt::Ro]));
    // Hostile: the honest one-page artifact reheaded to a reservation under a page, and to heads
    // past it. Each answers, and none panics.
    let one = freeze_with_prots(&m, &[0u8; PAGE], &[PageProt::Rw], 12, &host()).expect("freeze");
    for (reserved, mapped) in [(11, P), (12, 2 * P), (12, 0), (13, 3 * P)] {
        let bent = rehead(&one, reserved, mapped);
        let _ = restore_layout(&bent, &m, &mut host());
        let _ = restore_with_prots(&bent, &m, &mut host());
    }
    assert_eq!(
        restore_layout(&rehead(&one, 11, P), &m, &mut host()).err(),
        Some(RestoreError::GeometryMismatch),
        "a head past its reservation"
    );
}

/// #3100 — a sub-page module's own first page is not growth, on any holder: the honest un-grown
/// `memory 10` artifact restores under a grant of one page through the sparse, flat and dense entries
/// (its 1 KiB declared window is charged past any ceiling, and the rest of its page is its head).
#[test]
fn a_sub_page_modules_first_page_is_not_growth_on_any_holder() {
    let m = module_of(10);
    let mut window = vec![0u8; PAGE];
    window[7] = 9;
    let art = freeze_with_prots(&m, &window, &[PageProt::Rw], 12, &host()).expect("freeze");
    for holder in [Holder::Sparse, Holder::Flat, Holder::Dense] {
        let mut h = host();
        h.set_mem_grant(PAGE as i64);
        let got = restore_into(&art, &m, &mut h, Restoring::of(&m, holder)).map(|_| ());
        assert_eq!(got, Ok(()), "{holder:?}");
    }
}

/// #3100 — the root's declared window is what the embedder chose to run: an activation charges it
/// past any ceiling (INVARIANTS #3), so a module declaring 1 MiB runs, freezes, and restores under a
/// grant of 64 KiB, as on main. Its growth is what the grant bounds, by what the grant has left past
/// the declared window, as a running guest's `map` is: none here, so one host page of growth is
/// refused; under a grant of the declared window and 64 KiB, 64 KiB of growth fits and a host page
/// more does not.
#[test]
fn the_roots_declared_window_is_charged_past_the_ceiling_and_its_growth_is_not() {
    let hp = temen_interp::host_page_size();
    let m = module_of(20);
    let declared = 1u64 << 20;
    let freeze = |growth: u64| {
        let window = vec![0u8; (declared + growth) as usize];
        let prots = vec![PageProt::Rw; window.len() / PAGE];
        freeze_with_prots(&m, &window, &prots, 24, &host()).expect("freeze")
    };
    let restore = |art: &[u8], grant: u64| {
        let mut h = host();
        h.set_mem_grant(grant as i64);
        let got = restore_layout(art, &m, &mut h).map(|_| ());
        if got.is_err() {
            untouched(&h, "refused");
        }
        got
    };
    let small = 64 * 1024;
    assert_eq!(
        restore(&freeze(0), small),
        Ok(()),
        "the declared window alone"
    );
    for holder in [Holder::Flat, Holder::Dense] {
        let mut h = host();
        h.set_mem_grant(small as i64);
        let got = restore_into(&freeze(0), &m, &mut h, Restoring::of(&m, holder)).map(|_| ());
        assert_eq!(got, Ok(()), "the declared window alone, {holder:?}");
    }
    assert_eq!(
        restore(&freeze(hp), small),
        Err(RestoreError::WindowOverBudget { need: hp, room: 0 }),
        "a page of growth past a grant the declared window already passes"
    );
    let grant = declared + small;
    assert_eq!(
        restore(&freeze(small), grant),
        Ok(()),
        "growth under the grant"
    );
    assert_eq!(
        restore(&freeze(small + hp), grant),
        Err(RestoreError::WindowOverBudget {
            need: small + hp,
            room: small
        })
    );
}

/// #3137 — a held span with entries that are not plain: a `Backed` page in it is its region's, not
/// held (`held_tail`), and a bare `Unmapped` one (no bytes) is held with its contents kept (#1909).
/// The span's charge is its host pages less the `Backed` one; the window restores under exactly that
/// and is refused one host page under.
#[test]
fn a_held_span_charges_its_pages_less_the_ones_a_region_holds() {
    let hp = temen_interp::host_page_size();
    let (fh, region) = with_region();
    let mut l = MemLayout::from_parts(vec![0u8; WINDOW], P, W, &[]).expect("a page list");
    let backed = PageProt::Backed {
        region,
        off: 0,
        writable: true,
    };
    for k in 0..hp / P {
        l.hold(W + 4 * hp + k * P, backed, None);
        l.hold(W + 8 * hp + k * P, PageProt::Unmapped, None);
    }
    let art = rehead(
        &freeze_layout(&module(), &l, 20, &fh).expect("freeze"),
        20,
        W + 16 * hp,
    );
    let (layout, _) = restore_layout(&art, &module(), &mut host()).expect("restore");
    assert_eq!(layout.page_map().held_span(), W..W + 16 * hp);
    let need = W + 15 * hp;
    assert_eq!(under(&art, need - hp), Err(over(need - hp, need)));
    assert_eq!(under(&art, need), Ok(()));
}
