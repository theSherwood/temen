//! #3100 — the window image a restore allocates is bounded before anything is allocated for it. On
//! `11ada7d` the header's `mapped` was bounded only by the artifact's own `reserved_log2` (up to 63),
//! and `restore` allocated `mapped` bytes before it read a page: a ~100-byte artifact asked for up to
//! 2^63 bytes, an abort or an OOM kill. Its own test binary, for the counting allocator.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};

use temen_encode::wire;
use temen_interp::Host;
use temen_snapshot::{freeze, restore, RestoreError};

/// The largest single allocation this binary made since the last [`reset`].
static PEAK: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        PEAK.fetch_max(l.size(), SeqCst);
        System.alloc(l)
    }
    unsafe fn alloc_zeroed(&self, l: Layout) -> *mut u8 {
        PEAK.fetch_max(l.size(), SeqCst);
        System.alloc_zeroed(l)
    }
    unsafe fn realloc(&self, p: *mut u8, l: Layout, new: usize) -> *mut u8 {
        PEAK.fetch_max(new, SeqCst);
        System.realloc(p, l, new)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}

#[global_allocator]
static A: Counting = Counting;

const ARENA: temen_ir::durable_abi::ShadowArena =
    temen_ir::durable_abi::ShadowArena::new(16448, 65536);

fn module() -> temen_ir::Module {
    let mut m = temen_text::parse_module(
        "func () -> (i64) {\nblock 0 () {\n  v0 = i64.const 1\n  return v0\n  }\n}\n",
    )
    .expect("parse");
    m.memory = Some(temen_ir::Memory {
        size_log2: 17,
        shadow: Some(ARENA),
    });
    temen_durable::transform_module(&m).expect("transform")
}

fn uleb(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let b = (v & 0x7f) as u8;
        v >>= 7;
        out.push(if v == 0 { b } else { b | 0x80 });
        if v == 0 {
            return;
        }
    }
}

/// `art` with its header's `reserved_log2` and `mapped` replaced.
fn with_geometry(art: &[u8], reserved_log2: u8, mapped: u64) -> Vec<u8> {
    let (hdr, payload) = wire::read_header(art).expect("a header");
    let mut out = Vec::new();
    wire::write_header(&mut out, hdr.kind, hdr.version, hdr.flags);
    // Section 0 is first: its tag and length are one byte each for a header this small.
    let (tag, len) = (payload[0], payload[1] as usize);
    assert_eq!(tag, 0, "the header section leads");
    let body = &payload[2..2 + len];
    let mut nb = body[..32].to_vec(); // the digest
    nb.push(reserved_log2);
    // Skip the old reserved_log2 and mapped, keep the rest.
    let mut at = 33;
    while body[at] & 0x80 != 0 {
        at += 1;
    }
    at += 1;
    uleb(&mut nb, mapped);
    nb.extend_from_slice(&body[at..]);
    out.push(0);
    uleb(&mut out, nb.len() as u64);
    out.extend_from_slice(&nb);
    out.extend_from_slice(&payload[2 + len..]);
    out
}

#[test]
fn a_window_no_host_would_hold_is_refused_before_it_is_allocated() {
    let m = module();
    let win = temen_durable::init_durable_window(1 << 17, ARENA);
    let art = freeze(&m, &win, &Host::new()).expect("freeze");
    // A real artifact's header, bent: the image it asks for is set by two fields of its first
    // section, whatever else it holds.
    assert!(restore(&art, &m, &mut Host::new()).is_ok(), "the control");

    for (what, reserved_log2, mapped) in [
        // #3100's own artifact: a reservation no engine builds, and an image of 2^62 bytes.
        ("a 2^63 reservation", 63, 1u64 << 62),
        // A reservation an engine builds (a window frozen under the old 2^40 default), and an image
        // past the 4 GiB a default host's budget lets a domain hold (`DEFAULT_MEM`).
        ("an image past the domain's memory", 40, 1u64 << 39),
    ] {
        let bent = with_geometry(&art, reserved_log2, mapped);
        PEAK.store(0, SeqCst);
        let got = restore(&bent, &m, &mut Host::new()).err();
        let peak = PEAK.load(SeqCst);
        assert_eq!(got, Some(RestoreError::GeometryMismatch), "{what}");
        assert!(
            peak < 1 << 20,
            "{what}: allocated {peak} bytes before refusing"
        );
    }

    // The bound is the restoring host's: a domain granted less memory holds a smaller window.
    let grown = with_geometry(&art, 17, 1 << 17);
    let mut tight = Host::new();
    tight.set_mem_grant(0);
    assert!(
        restore(&grown, &m, &mut tight).is_ok(),
        "its declared window is not growth"
    );
}
