//! **`SharedProgram::run_over_grown`** (#816) — the warm-snapshot growth seam: a `vm_map`-grown
//! committed extent survives the fresh-`Mem`-per-call shape by round-tripping the scalar extent
//! (`Mem::scalar_extent` out, `Mem::seed_committed` back in, no zeroing).
//!
//! The bytecode differential contract: call 1 grows the window and writes a marker into the grown
//! page; call 2 over the SAME shared backing must read the marker back **iff** the returned extent
//! is seeded — unseeded, the fresh `Mem`'s empty page map faults exactly where a cold window would
//! (the pre-#816 warm-restore bug, kept as the negative pin). Also pins that the reservation clamp
//! makes an over-grow fail with `-EINVAL` instead of minting silently-dropped pages.

use std::sync::Arc;
use svm_interp::{bytecode, Host, Region, Trap, Value};

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
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
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
  vr = cap.call 5 0 (i64, i64, i32) -> (i64) vas (voff, vlen, vprot)
  return vr
  }
}
"#;

fn build() -> bytecode::SharedProgram {
    let m = svm_text::parse_module(SRC).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
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

fn grow_then(seed: Option<u64>) -> (Result<Vec<Value>, Trap>, Option<u64>) {
    let prog = build();
    let back = backing();
    let mut host = Host::new();
    let asl = host.grant_memory();
    let mut fuel = u64::MAX;
    // Call 1: grow one host page and write the marker (a whole 16-KiB page covers macOS too).
    let (ran, extent) = prog.run_over_grown(
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
    let extent = extent.expect("contiguous grow stays scalar-representable");
    assert!(
        extent >= (1 << DECLARED_LOG2) + 16384,
        "extent must cover the grown page (got {extent})"
    );
    // Call 2: fresh Mem over the same backing — the seam under test.
    let seed = seed.map(|_| extent);
    prog.run_over_grown(
        1,
        &[],
        &mut fuel,
        back,
        &mut host,
        false,
        BACKING_LOG2,
        seed,
    )
}

#[test]
fn seeded_extent_restores_the_grown_page() {
    let (ran, extent) = grow_then(Some(0));
    assert_eq!(
        ran.expect("seeded probe reads the grown page"),
        vec![Value::I64(424242)],
        "the marker must survive the re-commit (seed_committed must NOT zero)"
    );
    assert!(extent.is_some(), "probe leaves the state scalar");
}

#[test]
fn unseeded_probe_faults_like_a_cold_window() {
    // The pre-#816 warm-restore bug as the negative pin: without the seed, the fresh Mem's empty
    // page map treats the grown page as uncommitted reserved tail — the load faults.
    let (ran, _) = grow_then(None);
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
    let (ran, _) = prog.run_over_grown(
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
