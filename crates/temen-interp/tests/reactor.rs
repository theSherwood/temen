//! The persistent `bytecode::Reactor` — instantiate once, call exports many times, with the **whole**
//! guest window (globals/BSS **and** a `vm_map`-grown heap) persisting across calls. This is what the
//! snapshot reactors (`temen-run`'s `Session`, the browser `OnrampReactor`) could not do: they
//! round-trip only the low `SNAP_CAP` (256 KiB) prefix, so a guest's state above that — a grown heap —
//! was lost every call. Keeping the `Mem` live fixes it, and is what lets a heavy-heap guest (the
//! playground's Game of Life, eventually Doom) hold state frame to frame.

use temen_interp::{bytecode, Host, Value};
use temen_text::parse_module;

// A counter at a HIGH address (300000 ≈ 293 KiB — above the 256 KiB `SNAP_CAP` the snapshot reactors
// captured) is loaded, incremented, stored, and returned. Over repeated `call`s it must climb
// 1, 2, 3, … which holds only if the reactor persists memory beyond that 256 KiB prefix.
const SRC: &str = r#"
memory 19
func () -> (i64) {
block 0 () {
  v0 = i64.const 300000
  v1 = i64.load v0
  v2 = i64.const 1
  v3 = i64.add v1 v2
  i64.store v0 v3
  return v3
  }
}
"#;

#[test]
fn reactor_persists_high_memory_across_calls() {
    let m = parse_module(SRC).expect("parse the counter module");
    let mut r = bytecode::Reactor::open(&m).expect("open the reactor");
    let mut host = Host::new();
    for expect in 1..=5i64 {
        let mut fuel = u64::MAX;
        let out = r
            .call(0, &[], &mut fuel, &mut host)
            .expect("call the counter");
        assert_eq!(
            out,
            vec![Value::I64(expect)],
            "the counter at 293 KiB climbs — memory above the 256 KiB prefix persisted across calls",
        );
    }
}

// A fresh reactor starts from a zeroed window (persistence is per-instance, not global): the counter
// begins at 1 again, proving `open` seeds a clean window rather than leaking the previous instance's.
#[test]
fn a_fresh_reactor_starts_clean() {
    let m = parse_module(SRC).expect("parse");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let mut r1 = bytecode::Reactor::open(&m).expect("open r1");
    assert_eq!(
        r1.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(1)])
    );
    assert_eq!(
        r1.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(2)])
    );
    let mut r2 = bytecode::Reactor::open(&m).expect("open r2");
    assert_eq!(
        r2.call(0, &[], &mut fuel, &mut host),
        Ok(vec![Value::I64(1)]),
        "a new reactor's window is fresh, not inherited from r1",
    );
}

// ---- #1458: the §12 codec's dense page map round-trips through a live reactor window ------------
//
// A save-state carries the window's protection map in the codec's fixed 4 KiB unit; the reactor's
// window is in the host's (16 KiB on an arm64 Mac). `MemLayout::from_dense` / `dense_prots` are the
// one conversion, and `restore_window` re-expresses the map in the window's own unit — so a thawed
// guest faults exactly where the frozen one did: on an uncommitted hole *below* the grown high-water
// as much as above it, and never on a page it had committed.

/// `memory 16` (64 KiB declared); func 0 `peek(addr)` loads the i64 at `addr`.
const PEEK_SRC: &str = r#"memory 16
func (i64) -> (i64) {
block 0 (vaddr: i64) {
  vl = i64.load vaddr
  return vl
  }
}
"#;

const K16: u64 = 16 * 1024;

#[test]
fn a_dense_page_map_restores_into_the_reactor_window_in_its_own_page_unit() {
    use temen_interp::{CapturedProt, MemLayout};
    let m = parse_module(PEEK_SRC).expect("parse");
    let mut r = bytecode::Reactor::open(&m).expect("open");

    // 128 KiB image over a 64 KiB window, in 16 KiB runs so it is representable on either host page:
    // [0,16K) Ro (the null guard's shape), [16K,64K) Rw prefix, [64K,112K) an uncommitted hole under
    // the high-water, [112K,128K) a grown Rw page carrying a marker.
    let mut bytes = vec![0u8; 8 * K16 as usize];
    bytes[7 * K16 as usize..7 * K16 as usize + 8].copy_from_slice(&0x5eed_u64.to_le_bytes());
    let mut dense = vec![CapturedProt::Rw; 32];
    dense[0..4].fill(CapturedProt::Ro);
    dense[16..28].fill(CapturedProt::Unmapped);
    let layout = MemLayout::from_dense(bytes, &dense, 1 << 16);
    assert_eq!(
        layout.dense_prots(),
        dense,
        "from_dense ⇄ dense_prots is exact"
    );

    assert!(r.restore_window(&layout));
    let back = r.window_layout().expect("capturable");
    assert_eq!(
        back.dense_prots(),
        dense,
        "the live window's map, read back in the codec's unit, is the map that was restored"
    );
    assert_eq!(
        back.bytes(),
        layout.bytes(),
        "the image round-trips, hole included"
    );

    // And the map is really installed: the grown page reads its marker, the hole faults.
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    assert_eq!(
        r.call(0, &[Value::I64(7 * K16 as i64)], &mut fuel, &mut host),
        Ok(vec![Value::I64(0x5eed)]),
        "a grown reserved-tail page came back committed"
    );
    assert!(
        r.call(0, &[Value::I64(5 * K16 as i64)], &mut fuel, &mut host)
            .is_err(),
        "an uncommitted hole under the high-water stays a hole"
    );
}

#[test]
fn a_coarse_page_entry_expands_over_every_codec_page_it_covers() {
    use temen_interp::{CapturedProt, MemLayout};
    // A 16 KiB-page capture with one `Ro` entry at page 1: in the codec's 4 KiB unit that is pages
    // 4..8, not page 4 alone (which would leave the rest of the protected page writable on restore).
    let layout = MemLayout::from_parts(vec![0u8; 4 * K16 as usize], K16, 4 * K16, &[(K16, 0)])
        .expect("a valid page list");
    let mut want = vec![CapturedProt::Rw; 16];
    want[4..8].fill(CapturedProt::Ro);
    assert_eq!(layout.dense_prots(), want);
}
