//! **The canonical futex-key registry keeps no entry past its window** (#1608, #3059).
//!
//! Window virtual addresses are recycled between runs. An entry left behind at a dead window's
//! address sits under whichever window is handed it next, and a waiter and a notifier there can then
//! compute different keys for one address: the notify reaches nobody and the run wedges, or its
//! deadlock verdict fires, with no error anywhere. A window therefore forgets every entry in its
//! reservation as it releases it.
//!
//! These live in their own integration binary rather than in `temen-jit`'s unit tests **on purpose**:
//! added there, registry tests perturbed the Windows `fiber-scaling` lane's `mem::tests` (#1575). The
//! registry is public API, so this is where they belong.

use std::sync::atomic::{AtomicU64, Ordering};
use temen_jit::{region_canon_forget_window, region_canon_lookup, CompiledModule, JitOutcome};

/// 4 KiB everywhere temen runs; the registry pages with its own granule, so a test that stays a
/// whole page away from its neighbours does not need to know it exactly.
const PAGE: u64 = 4096;

/// A forget clears every entry in its range, and nothing outside it.
#[test]
fn a_forget_clears_its_range_alone() {
    let abs = 0x5a5a_0000_0000u64;
    temen_jit::region_canon_record(abs, PAGE, 42, 0);
    temen_jit::region_canon_record(abs + 4 * PAGE, PAGE, 43, 0);
    region_canon_forget_window(abs - PAGE, PAGE * 4);
    assert_eq!(region_canon_lookup(abs), None);
    assert_eq!(region_canon_lookup(abs + 4 * PAGE), Some((43, 0)));
    region_canon_forget_window(abs + 4 * PAGE, PAGE);
}

/// The page of the window the hook below records at, and what the lookup saw while the window lived.
static RECORDED_AT: AtomicU64 = AtomicU64::new(0);
static SEEN_LIVE: AtomicU64 = AtomicU64::new(0);

/// A window-ready hook that records a region identity for the window's second page, as a host's
/// region recorder would at a §13 `map`, and reads it back.
unsafe extern "C" fn record_a_page(
    _ctx: *mut core::ffi::c_void,
    base: *mut u8,
    _mapped: u64,
    _reserved: u64,
    _ro: *const [u64; 2],
    _n_ro: usize,
    _shadow: temen_ir::durable_abi::ShadowArena,
) -> i32 {
    let at = base as u64 + PAGE;
    temen_jit::region_canon_record(at, PAGE, 77, 0);
    RECORDED_AT.store(at, Ordering::SeqCst);
    SEEN_LIVE.store(
        region_canon_lookup(at).map_or(0, |(backing, _)| backing),
        Ordering::SeqCst,
    );
    1
}

/// #3059 — a window forgets what was recorded for it as it is released: once the run is over, the
/// page is nobody's in the registry, whatever recorded it and however the host that did lives on.
#[test]
fn a_window_forgets_its_entries_as_it_is_released() {
    let m = temen_text::parse_module(
        "memory 16
func (i64) -> (i64) {
block 0 (v: i64) {
  return v
  }
}",
    )
    .expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut cm = CompiledModule::compile(
        &m,
        0,
        temen_jit::INERT_CAP_THUNK,
        std::ptr::null_mut(),
        temen_ir::DEFAULT_RESERVED_LOG2,
        None,
        None,
        None,
        None,
        None,
        0,
    )
    .expect("jit compile");
    cm.set_window_ready(Some((record_a_page, std::ptr::null_mut())));
    let (out, _) = cm.run(&[5], None, None).expect("jit run");
    assert_eq!(out, JitOutcome::Returned(vec![5]));
    let at = RECORDED_AT.load(Ordering::SeqCst);
    assert_ne!(at, 0, "the window-ready hook did not run");
    assert_eq!(
        SEEN_LIVE.load(Ordering::SeqCst),
        77,
        "the record did not land"
    );
    assert_eq!(
        region_canon_lookup(at),
        None,
        "a released window's page kept its region identity"
    );
}
