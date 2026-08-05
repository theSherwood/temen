//! Fault-service latency probe (run on demand, not in CI):
//!
//!   cargo test -p svm --release --test paging_bench -- --ignored --nocapture
//!
//! **What it measures and why.** The §14 parent-as-pager round-trip — a demand coroutine child
//! (`spawn_demand_coroutine`, Instantiator op 4) touches an unsupplied page, faults to the parent
//! (`resume` returns `CORO_FAULTED` + the address), the parent writes the page's bytes into the
//! shared window with ordinary stores and resumes, and the child's rewound access re-executes —
//! timed per-fault across the three backends. This is **the CONSOLIDATION.md §2 gate**: the
//! coroutine family's collapse onto the unified offer (child = process-provider, `resume` =
//! `cap.call`, fault-service = the §10.2 direct-handoff shape) must not regress this number. Pin
//! the bespoke ops' latency here first; the offer-backed twin is benchmarked against it.
//!
//! Shape notes: each child touches `P` pages at a 64 KiB stride, so the fault count is exactly
//! `P` on any host page size ≤ 64 KiB (supply maps only the host page containing the fault, and
//! the next touch is a stride away) — the checksum is host-independent. The parent respawns `S`
//! children over the same carve; a fresh demand child re-faults every page, so total round-trips
//! are `S·P` with spawn cost amortized 1/P per fault, identically in the bespoke and offer
//! variants (same S, P) — the comparison this file exists for is unaffected. As in
//! `serving_bench`, all backends must agree on the checksum *before* timing (never benchmark a
//! miscompile), so this file doubles as a CI pin for N sequential fault-service rounds — the
//! existing demand-coroutine tests only fault once or twice.
//!
//! Observed shape at pin time (ratios matter, not the absolute ns): TreeWalk and Bytecode sit
//! within ~20% of each other — the round-trip is a direct in-Rust call on a shared backing, so
//! there is little for the bytecode tier to recover. The **Jit lane runs an order of magnitude
//! slower than TreeWalk** (which also proves it runs the native coroutine path — a fold would
//! match TreeWalk): every fault switch pays `sync_committed` window mirroring in both directions
//! across the whole carve (`instantiator_rt.rs`). Read offer-variant comparisons per-tier: the
//! interp tiers set the tight bar; on the JIT the offer path (shared cell, no mirroring) may
//! legitimately *beat* the bespoke ops.

use std::time::Instant;

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// Child carve: a 2^23 window at offset 2^23 inside a 2^24 parent — 128 strides of 64 KiB.
const CARVE_OFF: u64 = 1 << 23;
const CARVE_LOG2: u64 = 23;
/// The byte the pager supplies at each fault address; the child sums what it reads.
const PAGE_BYTE: i64 = 7;
/// Per-fault marker the parent folds into the checksum, so a silently skipped fault (a page that
/// never unmapped, a supply that didn't stick) breaks the expected value rather than hiding.
const FAULT_MARK: i64 = 1000;

/// The status `resume` (op 3) reports for a child suspended at a fault; any other status exits
/// the resume loop (`RETURNED` = 1 folds the child's sum into the checksum implicitly).
const FAULTED: i64 = 2;

/// A parent `_start` that runs `s` pager cycles: spawn a demand coroutine over the carve, then
/// resume-loop — on `FAULTED`, store `PAGE_BYTE` at the reported (parent-coordinate) address and
/// count `FAULT_MARK`; on `RETURNED`, fold the child's sum in. Exits with the wrapped total.
/// The child (func 1) reads one byte per 64 KiB stride across its whole window and returns the sum.
fn paging_program(s: u64, p: u64) -> String {
    format!(
        "\
memory 24
data 0 \"vm\"
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 0
  vl = i64.const 2
  vh = cap.self.resolve vp vl
  vs0 = i64.const 0
  vacc0 = i64.const 0
  br 1(vh, vs0, vacc0)
}}
block 1 (vh1: i32, vs: i64, vacc: i64) {{
  ventry = i64.const 1
  voff = i64.const {off}
  vsl = i64.const {sl}
  vq = i64.const 0
  vch = cap.call 6 4 (i64, i64, i64, i64) -> (i32) vh1 (ventry, voff, vsl, vq)
  br 2(vh1, vs, vacc, vch)
}}
block 2 (vh2: i32, vs2: i64, vacc2: i64, vch2: i32) {{
  vz = i64.const 0
  vst, vval = cap.call 6 3 (i32, i64) -> (i32, i64) vh2 (vch2, vz)
  vstw = i64.extend_i32_s vst
  vf = i64.const {faulted}
  visf = i64.eq vstw vf
  br_if visf 3(vh2, vs2, vacc2, vch2, vval) 4(vh2, vs2, vacc2, vval)
}}
block 3 (vh3: i32, vs3: i64, vacc3: i64, vch3: i32, vaddr: i64) {{
  vb = i32.const {byte}
  i32.store8 vaddr vb
  vm = i64.const {mark}
  vacc4 = i64.add vacc3 vm
  br 2(vh3, vs3, vacc4, vch3)
}}
block 4 (vh4: i32, vs4: i64, vacc5: i64, vret: i64) {{
  vacc6 = i64.add vacc5 vret
  vone = i64.const 1
  vs5 = i64.add vs4 vone
  vsn = i64.const {s}
  vcmp = i64.lt_s vs5 vsn
  br_if vcmp 1(vh4, vs5, vacc6) 5(vacc6)
}}
block 5 (vaccf: i64) {{
  vc = i32.wrap_i64 vaccf
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vi0 = i64.const 0
  va0 = i64.const 0
  br 1(vi0, va0)
}}
block 1 (vi: i64, va: i64) {{
  vstride = i64.const 65536
  vaddr = i64.mul vi vstride
  vb = i32.load8_u vaddr
  vbw = i64.extend_i32_u vb
  va2 = i64.add va vbw
  vone = i64.const 1
  vi2 = i64.add vi vone
  vpn = i64.const {p}
  vcmp = i64.lt_s vi2 vpn
  br_if vcmp 1(vi2, va2) 2(va2)
}}
block 2 (vaf: i64) {{
  return vaf
  }}
}}
",
        off = CARVE_OFF,
        sl = CARVE_LOG2,
        faulted = FAULTED,
        byte = PAGE_BYTE,
        mark = FAULT_MARK,
        s = s,
        p = p,
    )
}

/// Every touch faults exactly once and reads `PAGE_BYTE`, so each cycle contributes
/// `p·(PAGE_BYTE + FAULT_MARK)`; `RETURNED` is implicitly asserted because a non-fault status
/// exits the resume loop with the child's sum, and a wrong sum breaks this expectation.
fn expected_exit(s: u64, p: u64) -> i32 {
    (s as i64)
        .wrapping_mul(p as i64)
        .wrapping_mul(PAGE_BYTE + FAULT_MARK) as i32
}

fn run(backend: Backend, src: &str) -> i32 {
    let m = parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    let r = inst
        .run_with_caps(
            backend,
            &RunConfig::default(),
            &[(
                "vm",
                HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
            )],
        )
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
    match r.outcome {
        Outcome::Exited(code) => code,
        other => panic!("{backend:?}: unexpected outcome {other:?}"),
    }
}

const BACKENDS: [Backend; 3] = [Backend::TreeWalk, Backend::Bytecode, Backend::Jit];

/// Correctness pin (runs in CI): all three backends agree on the checksum for `s` pager cycles of
/// `p` sequential fault-service rounds each.
#[test]
fn demand_fault_rounds_agree_across_backends() {
    let (s, p) = (2, 128);
    let src = paging_program(s, p);
    let want = expected_exit(s, p);
    for b in BACKENDS {
        assert_eq!(
            run(b, &src),
            want,
            "{b:?}: {s}x{p} fault-service rounds must checksum to s*p*(byte+mark)"
        );
    }
}

/// The latency probe. Ignored (timing is machine-dependent and not a CI gate); run manually and
/// compare the offer-backed variant against these numbers before deleting the bespoke ops.
#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn fault_service_latency() {
    let (s, p) = (64u64, 128u64);
    let faults = s * p;
    let src = paging_program(s, p);
    let want = expected_exit(s, p);

    // Correctness first — never benchmark a miscompile.
    for b in BACKENDS {
        assert_eq!(run(b, &src), want, "{b:?}: checksum before timing");
    }

    println!("\nfault-service round-trip — {s} spawns x {p} faults = {faults}, per-fault ns:");
    for b in BACKENDS {
        // Warm, then best-of-5 to shed scheduler noise.
        run(b, &src);
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            let t = Instant::now();
            let got = run(b, &src);
            let elapsed = t.elapsed().as_nanos() as f64;
            assert_eq!(got, want);
            best = best.min(elapsed);
        }
        println!(
            "  {b:?}: {:.0} ns/fault  (total {:.2} ms)",
            best / faults as f64,
            best / 1e6
        );
    }
}
