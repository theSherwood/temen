//! **The honesty pin for the frontier matrix's `concurrency` column (#1413 slice 3).**
//!
//! INVARIANTS #14's third axis asks whether a capability is "carried by both the cooperative
//! multiplex driver and the genuinely-parallel driver". Natively those are `bytecode::drive` (one
//! thread multiplexing every vCPU) and `bytecode::run_vcpu_parallel` (a real OS thread per vCPU,
//! the `temen_par_*` per-Worker driver's native twin). This test drives **both** with the same
//! generated call and fails when the manifest disagrees with what they do.
//!
//! ## What the predicate is, and why it is not "does it trap"
//!
//! The first probe I wrote asked whether each driver *refused* the op, and found nothing: both
//! refuse the same things. That predicate is too weak, and provably so — the op-15 gap fixed in
//! #1531 was a driver **answering differently** (`-EINVAL` on one, a spawned child on the other),
//! not refusing. So the question here is the stronger one INVARIANTS #9 actually asks: given the
//! *same* call, do the two drivers give the *same* answer? A value on one and a domain-killing trap
//! on the other is the divergence this column exists to surface, whichever direction it points.
//!
//! ## The comparison is per call shape, not per op
//!
//! The lowering's arms are arity- and signature-guarded, so a sweep that folds each driver's *best*
//! result across arities can compare an arity one driver reached against a different one the other
//! did — which manufactures divergences that are really the harness miscalling. Each `(op, argc,
//! result type)` triple is therefore run on both drivers and compared to its own twin. A shape that
//! cap-faults on both is the harness missing the arm, and is skipped.
//!
//! ## The audit rule
//!
//! As on the other columns, a row is scored only when the harness could actually mint its handle and
//! drive its ops. `Jit`, `JitCode`, `Offer` and `LiveImpl` need a live unit or a live peer, so they
//! stay `Unaudited` here exactly as they do on the debugger column, rather than being scored on a
//! sweep that never reached them.

mod support;
use std::sync::Arc;
use support::capability_probe::{probe_module_with, rows, Row, MAX_ARGC, PROBE_BEYOND};
use temen_interp::{bytecode, Host, Region, Trap, Value};
use temen_parity::frontier::{capability_axes, Axis, Capability};
use temen_parity::Status;

/// What one driver did with one call shape. Compared for equality between the two drivers, so the
/// answer's *content* matters: `Answered(-22)` and `Trapped(ThreadFault)` are different answers to
/// the same question, which is the whole point.
#[derive(Clone, PartialEq, Eq, Debug)]
enum Verdict {
    /// The module did not compile for this driver at all.
    NotInSubset,
    /// The call reached no dispatch arm — the harness miscalled it. Skipped, never compared.
    CapFault,
    /// The driver ran the op and it trapped, killing the domain.
    Trapped(String),
    /// The driver ran the op and it returned. A `-errno` counts: that is an answer.
    Answered(i64),
}

fn verdict(out: Option<Result<Vec<Value>, Trap>>) -> Verdict {
    match out {
        None => Verdict::NotInSubset,
        Some(Err(Trap::CapFault)) => Verdict::CapFault,
        Some(Err(t)) => Verdict::Trapped(format!("{t:?}")),
        Some(Ok(v)) => Verdict::Answered(match v.first() {
            Some(Value::I64(x)) => *x,
            other => panic!("the probe entry returns i64, got {other:?}"),
        }),
    }
}

/// The cooperative driver: one thread, every vCPU multiplexed onto it.
fn coop(
    mut host: Host,
    handle: i32,
    iface: u32,
    op: u32,
    argc: usize,
    res: &str,
    real: &[(usize, i64)],
) -> Verdict {
    let m = probe_module_with(iface, op, argc, res, real);
    let mut fuel = 1_000_000u64;
    verdict(bytecode::compile_and_run_with_host(
        &m,
        0,
        &[Value::I32(handle)],
        &mut fuel,
        &mut host,
    ))
}

/// The OS-thread parallel driver, over a shared window it can hand to a second vCPU. The `unsafe` of
/// borrowing host memory lives here in the test embedder, as it does in every parallel harness — the
/// engine stays `#![forbid(unsafe_code)]` and just takes the `Arc<Region>`.
fn parallel(
    mut host: Host,
    handle: i32,
    iface: u32,
    op: u32,
    argc: usize,
    res: &str,
    real: &[(usize, i64)],
) -> Verdict {
    let m = probe_module_with(iface, op, argc, res, real);
    let size = 1u64 << 16;
    let layout = std::alloc::Layout::from_size_align(size as usize, 8).unwrap();
    // SAFETY: non-zero, 8-aligned layout; freed below, after the run has joined every vCPU.
    let base = unsafe { std::alloc::alloc_zeroed(layout) };
    assert!(!base.is_null(), "window allocation");
    // SAFETY: `base` owns `size` zeroed bytes and outlives the run (which joins before dealloc).
    let back = Arc::new(unsafe { Region::shared(base, size) });
    let mut fuel = 1_000_000u64;
    let out = bytecode::compile_and_run_capture_over_parallel_with_host(
        &m,
        0,
        &[Value::I32(handle)],
        &mut fuel,
        &[],
        Arc::clone(&back),
        &mut host,
    );
    drop(back);
    // SAFETY: same layout; the region and every borrow of `base` are gone.
    unsafe { std::alloc::dealloc(base, layout) };
    verdict(out.map(|(r, _image)| r))
}

/// Every `(argc, result type)` shape the sweep tries.
fn shapes() -> impl Iterator<Item = (usize, &'static str)> {
    (0..MAX_ARGC).flat_map(|a| ["i64", "i32"].map(move |r| (a, r)))
}

/// The call shapes of `op` on which the two drivers disagree, as `(shape, coop, parallel)`.
fn divergences(row: &Row, op: u32) -> Vec<(String, Verdict, Verdict)> {
    // Real argument values this row declares for `op` (see `Row::real_args`): a handle the sweep's
    // zeros cannot stand in for, because a zero handle is forged and never reaches the code the two
    // drivers actually differ in.
    let mut out = Vec::new();
    for (argc, res) in shapes() {
        let (h, x, real) = (row.mint)();
        let real: Vec<(usize, i64)> = real
            .iter()
            .filter(|(o, _, _)| *o == op)
            .map(|(_, idx, v)| (*idx, *v))
            .collect();
        let c = coop(h, x, row.iface, op, argc, res, &real);
        let (h, x, _) = (row.mint)();
        let p = parallel(h, x, row.iface, op, argc, res, &real);
        // Neither driver found an arm: the harness miscalled this shape, so there is nothing to
        // compare. A cap fault on exactly one side *is* a divergence and is kept.
        if c == Verdict::CapFault && p == Verdict::CapFault {
            continue;
        }
        if c != p {
            out.push((format!("op {op}, argc {argc}, -> {res}"), c, p));
        }
    }
    out
}

fn col(a: Axis) -> usize {
    Axis::ALL.iter().position(|x| *x == a).expect("axis listed")
}

/// **The pin.** For every drivable capability, the manifest's `concurrency` cell must match whether
/// the two drivers actually agree on that capability's ops.
#[test]
fn the_concurrency_column_matches_what_the_two_drivers_actually_do() {
    let k = col(Axis::ConcurrencyModel);
    for row in rows() {
        let ops: Vec<u32> = if row.ops.is_empty() {
            // A capability with no callable ops: sweep a little way past its (empty) interface, so
            // "nothing to diverge about" is checked rather than assumed.
            (0..PROBE_BEYOND).collect()
        } else {
            row.ops.to_vec()
        };
        let found: Vec<(String, Verdict, Verdict)> =
            ops.iter().flat_map(|op| divergences(&row, *op)).collect();
        let observed = if found.is_empty() {
            Status::Full
        } else {
            Status::NotYet
        };
        let cell = capability_axes(row.cap)[k];
        assert_eq!(
            observed,
            cell.status,
            "{}: the frontier manifest says `{:?}` on the concurrency axis, but the drivers {}.\n\
             Divergences found:\n{}",
            row.cap.name(),
            cell.status,
            if found.is_empty() {
                "agree on every call shape"
            } else {
                "disagree"
            },
            found
                .iter()
                .map(|(shape, c, p)| format!("  {shape}: coop={c:?} parallel={p:?}\n"))
                .collect::<String>(),
        );
    }
}

/// The two divergences the column's first rendering found, now both closed — pinned as *specific*
/// facts so a regression names itself instead of just reddening a cell.
///
/// `child_offer` (op 14) answered `-EINVAL` on the cooperative driver and **trapped** on the
/// parallel one, so a guest probing a stale child handle survived on one driver and had its domain
/// killed on the other — INVARIANTS #5 (errors are values, traps are for forgery) and #9 (refuse
/// probeably, never diverge). Fixed in #1566.
///
/// `instantiate_module_named` (op 13) was worse than the issue that filed it claimed. Its lowering
/// passes `grants: Some((ptr, n))` unconditionally, so the parallel driver's `grants.is_some()`
/// guard declined **every** op-13 spawn there with `Trap::Malformed`, grant-free ones included; the
/// guard also preceded the module resolve, so a forged handle got that trap instead of the
/// `CapFault` the cooperative driver gives; and that arm never called `bind_child_manifest`, so a
/// child's imports went unbound where the cooperative driver bound them. Fixed in #1570 by giving
/// both drivers one `named_child_host`.
///
/// Reaching op 13's admission at all needs a real `Module` handle and a matching `size_log2` — see
/// `Row::mint`. With the sweep's zeros it dies at the handle resolve, which is why the original
/// probe saw the two drivers "agree" on this op while they disagreed one step further in.
#[test]
fn the_two_instantiator_divergences_stay_closed() {
    let row = rows()
        .into_iter()
        .find(|r| r.cap == Capability::Instantiator)
        .expect("the Instantiator row");
    for op in [13, 14] {
        let found = divergences(&row, op);
        assert!(
            found.is_empty(),
            "op {op} diverges again — the drivers must give one answer per call shape: {found:?}"
        );
    }
    // Non-vacuity: op 13 really does reach admission, rather than agreeing by faulting early on
    // both sides. `-EINVAL` is the answer for a carve that cannot hold the child module.
    let (h, x, real) = (row.mint)();
    let real: Vec<(usize, i64)> = real
        .iter()
        .filter(|(o, _, _)| *o == 13)
        .map(|(_, i, v)| (*i, *v))
        .collect();
    assert_eq!(
        coop(h, x, row.iface, 13, 7, "i32", &real),
        Verdict::Answered(-22),
        "the op-13 sweep must get past the handle resolve into admission"
    );
}
