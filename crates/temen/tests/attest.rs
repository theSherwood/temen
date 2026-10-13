//! PROCESS.md §6 — `self.attest`: the non-interposable **trust anchor**. A domain reads its own
//! platform-vouched provenance (isolation tier + whether an ancestor can read/snapshot it) as a packed
//! `i32` — `tier | (window_exposed << 8) | (freeze_exposed << 9)`. Being a D46 `cap.self` intrinsic
//! (runtime-resolved, never a handle), no nested host can interpose and forge it.
//!
//! `attest` routes through the same `Host::self_dispatch` on every backend (like `self.count`), so
//! the **root** report is a cross-backend differential. A detached child's report is
//! `spawn_every_driver.rs`'s `a_detached_child_attests_its_spawners_tier`.

use temen_interp::{run_capture_reserved_with_host, Attestation, Host, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host, JitOutcome};
use temen_text::parse_module;
use temen_verify::verify_module;

/// func 0 `(i32) -> (i64)` (arg ignored): return `self.attest` zero-extended to `i64`.
const ATTEST_SELF: &str = "memory 17\n\
func (i32) -> (i64) {\n\
block 0 (v0: i32) {\n\
  va = self.attest\n\
  vr = i64.extend_i32_u va\n\
  return vr\n\
  }\n\
}\n";

fn run_interp(src: &str, host: &mut Host) -> Result<Vec<Value>, temen_interp::Trap> {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut fuel = 5_000_000u64;
    run_capture_reserved_with_host(
        &m,
        0,
        &[Value::I32(0)],
        &mut fuel,
        &[0u8; 128 << 10],
        0,
        host,
    )
    .0
}

fn run_jit(src: &str, host: &mut Host) -> JitOutcome {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    compile_and_run_capture_reserved_with_host(
        &m,
        0,
        &[0i64],
        &[0u8; 128 << 10],
        0,
        temen_run::cap_thunk,
        host as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit")
    .0
}

/// The root's `attest` report packs `tier | (exposed << 8) | (freeze << 9)` and is byte-identical on
/// both backends (`self_dispatch` is shared). Checked across a few provenance configs.
fn both_root(att: Attestation) -> (Result<Vec<Value>, temen_interp::Trap>, JitOutcome) {
    let mut ih = Host::new();
    ih.set_attestation(att);
    let ir = run_interp(ATTEST_SELF, &mut ih);
    let mut jh = Host::new();
    jh.set_attestation(att);
    let jo = run_jit(ATTEST_SELF, &mut jh);
    (ir, jo)
}

fn expect_both(att: Attestation, packed: i64) {
    let (ir, jo) = both_root(att);
    assert_eq!(ir, Ok(vec![Value::I64(packed)]), "interp: attest packing");
    assert!(
        matches!(jo, JitOutcome::Returned(ref s) if s == &[packed]),
        "jit: attest must match interp ({packed}), got {jo:?}"
    );
}

#[test]
fn root_attest_default_is_tier1_unexposed() {
    // A fresh Host is a root domain: tier 1, no ancestor read, not ancestor-freezable → packed 1.
    expect_both(Attestation::default(), 1);
}

#[test]
fn root_attest_packs_tier_and_exposure_bits() {
    // tier 3 (separate-process): packed 3.
    expect_both(
        Attestation {
            tier: 3,
            window_exposed: false,
            freeze_exposed: false,
        },
        3,
    );
    // window-exposed sets bit 8; freeze-exposed sets bit 9. tier 1 + both → 1 | 256 | 512 = 769.
    expect_both(
        Attestation {
            tier: 1,
            window_exposed: true,
            freeze_exposed: true,
        },
        769,
    );
}
