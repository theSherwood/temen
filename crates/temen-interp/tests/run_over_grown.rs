//! **`SharedProgram::run_over_grown`** (#816) — the warm-snapshot restore seam: a `vm_map`-grown
//! (and `protect`ed) page-state map survives the fresh-`Mem`-per-call shape by round-tripping the
//! explicit `map_info` entries (`Mem::seed_pages` back in, no zeroing).
//!
//! The bytecode differential contract: call 1 grows the window and writes a marker into the grown
//! page; call 2 over the SAME shared backing must read the marker back **iff** the returned extent
//! is seeded — unseeded, the fresh `Mem`'s empty page map faults exactly where a cold window would
//! (the pre-#816 warm-restore bug, kept as the negative pin). Also pins that the reservation clamp
//! makes an over-grow fail with `-EINVAL` instead of minting silently-dropped pages.

use std::sync::Arc;
use temen_interp::{bytecode, Host, Region, Trap, Value};

/// The declared window: 64 KiB; the backing (and clamped reservation): 128 KiB.
const DECLARED_LOG2: u8 = 16;
const BACKING_LOG2: u8 = 17;

/// func 0 `grow(as, len)`: `vm_map` `[64 KiB, 64 KiB + len)`, store a marker at `64 KiB + 16`,
/// return the map's result (`0` ok, negative errno). func 1 `probe()`: load the marker back.
const SRC: &str = r#"memory 16
func (i32, i64) -> (i64) {
block 0 (vas: i32, vlen: i64) {
  voff = i64.const 65536
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  vaddr = i64.const 65552
  vmark = i64.const 424242
  i64.store vaddr vmark
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vaddr = i64.const 65552
  vl = i64.load vaddr
  return vl
  }
}
func (i32, i64) -> (i64) {
block 0 (vas: i32, vlen: i64) {
  voff = i64.const 65536
  vprot = i32.const 3
  vr = call.cap 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
"#;

fn build() -> bytecode::SharedProgram {
    let m = temen_text::parse_module(SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    bytecode::SharedProgram::compile(&m).expect("compile")
}

/// An 8-aligned zeroed backing + `Region::shared` over it, leaked for the test's life.
fn backing() -> Arc<Region> {
    let size = 1usize << BACKING_LOG2;
    let layout = std::alloc::Layout::from_size_align(size, 8).unwrap();
    // SAFETY: non-zero 8-aligned layout; leaked, so the Region borrow is sound for the process.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null());
    // SAFETY: `size` valid 8-aligned bytes, owned and never freed.
    Arc::new(unsafe { Region::shared(base, size as u64) })
}

type ProbeOutcome = (Result<Vec<Value>, Trap>, Option<Vec<(u64, u8)>>);

fn grow_then(seed: bool) -> ProbeOutcome {
    let prog = build();
    let back = backing();
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut fuel = u64::MAX;
    // Call 1: grow one host page and write the marker (a whole 16-KiB page covers macOS too).
    let (ran, pages, _) = prog.run_over_grown(
        0,
        &[Value::I32(asl), Value::I64(16384)],
        &mut fuel,
        back.clone(),
        &mut host,
        true,
        BACKING_LOG2,
        None,
    );
    assert_eq!(
        ran.expect("grow call runs"),
        vec![Value::I64(0)],
        "the map itself must succeed"
    );
    let pages = pages.expect("no region alias — the map must be restorable");
    assert!(
        pages
            .iter()
            .any(|&(off, kind)| kind == 1 && off >= 1 << DECLARED_LOG2),
        "entries must cover the grown page (got {pages:?})"
    );
    // Call 2: fresh Mem over the same backing — the seam under test.
    let seed = seed.then_some(pages);
    let (ran, pages, _) = prog.run_over_grown(
        1,
        &[],
        &mut fuel,
        back,
        &mut host,
        false,
        BACKING_LOG2,
        seed.as_deref(),
    );
    (ran, pages)
}

#[test]
fn seeded_extent_restores_the_grown_page() {
    let (ran, pages) = grow_then(true);
    assert_eq!(
        ran.expect("seeded probe reads the grown page"),
        vec![Value::I64(424242)],
        "the marker must survive the re-commit (seed_pages must NOT zero)"
    );
    assert!(pages.is_some(), "probe leaves the state restorable");
}

#[test]
fn unseeded_probe_faults_like_a_cold_window() {
    // The pre-#816 warm-restore bug as the negative pin: without the seed, the fresh Mem's empty
    // page map treats the grown page as uncommitted reserved tail — the load faults.
    let (ran, _) = grow_then(false);
    assert!(
        matches!(ran, Err(Trap::MemoryFault)),
        "unseeded fresh Mem must fault on the grown page (got {ran:?})"
    );
}

#[test]
fn overgrow_past_the_clamped_reservation_fails_probeably() {
    // The backing clamp (#816): with the reservation clamped to the 128-KiB backing, a map that
    // would land pages past it fails with a negative errno the guest can observe — instead of
    // minting page-map entries whose writes the backing silently drops.
    let prog = build();
    let back = backing();
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut fuel = u64::MAX;
    let (ran, _, _) = prog.run_over_grown(
        2, // the store-free grow func: the errno itself is the observable
        &[Value::I32(asl), Value::I64(1 << 20)], // 1 MiB — far past the 128-KiB backing
        &mut fuel,
        back,
        &mut host,
        true,
        BACKING_LOG2,
        None,
    );
    let vals = ran.expect("the guest observes the errno, no trap");
    assert!(
        matches!(vals.first(), Some(Value::I64(e)) if *e < 0),
        "over-grow must fail probeably (got {vals:?})"
    );
}

/// The real-consumer shape that broke the scalar design (the on-ramp `protect`s its rodata):
/// func 0 writes a marker INSIDE the declared prefix then protects its page read-only; the seeded
/// follow-up must read it back (Ro restore keeps bytes + readability) and a seeded store into it
/// must fault (the protection itself is restored, not just the bytes).
const PROTECT_SRC: &str = r#"memory 16
func (i32) -> (i64) {
block 0 (vas: i32) {
  vaddr = i64.const 16400
  vmark = i64.const 777
  i64.store vaddr vmark
  voff = i64.const 16384
  vlen = i64.const 16384
  vprot = i32.const 1
  vr = call.cap 5 2 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16400
  vl = i64.load vaddr
  return vl
  }
}
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16400
  vx = i64.const 1
  i64.store vaddr vx
  return vx
  }
}
"#;

#[test]
fn protected_rodata_round_trips_readable_and_write_protected() {
    let m = temen_text::parse_module(PROTECT_SRC).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let prog = bytecode::SharedProgram::compile(&m).expect("compile");
    let back = backing();
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut fuel = u64::MAX;
    let (ran, pages, _) = prog.run_over_grown(
        0,
        &[Value::I32(asl)],
        &mut fuel,
        back.clone(),
        &mut host,
        true,
        BACKING_LOG2,
        None,
    );
    assert_eq!(ran.expect("protect runs"), vec![Value::I64(0)]);
    let pages = pages.expect("restorable");
    assert!(
        pages.iter().any(|&(_, kind)| kind == 0),
        "the protect must appear as an Ro entry (got {pages:?})"
    );

    // Seeded read: bytes AND readability restored.
    let (ran, _, _) = prog.run_over_grown(
        1,
        &[],
        &mut fuel,
        back.clone(),
        &mut host,
        false,
        BACKING_LOG2,
        Some(&pages),
    );
    assert_eq!(
        ran.expect("seeded Ro read"),
        vec![Value::I64(777)],
        "the marker under the restored Ro page must read back"
    );

    // Seeded store: the PROTECTION is restored too — the write faults.
    let (ran, _, _) = prog.run_over_grown(
        2,
        &[],
        &mut fuel,
        back,
        &mut host,
        false,
        BACKING_LOG2,
        Some(&pages),
    );
    assert!(
        matches!(ran, Err(Trap::MemoryFault)),
        "a store into the restored Ro page must fault (got {ran:?})"
    );
}
