//! Fault-service latency probe (run on demand, not in CI):
//!
//!   cargo test -p temen --release --test paging_bench -- --ignored --nocapture
//!
//! **What it measures and why.** The fault-service round-trip on the unified offer transport —
//! a detached demand child (a v1 record naming the parent's pager export) touches an unsupplied
//! page, the fault becomes a `page(addr)` call on the parent's pager export (direct handoff runs the
//! handler inline on the child's thread), the handler stores the page's bytes, the substrate supplies
//! the page, and the rewound access re-executes — timed per-fault across the backends. Until §2.3
//! this file raced the offer lane against the bespoke coroutine ops (`spawn_demand_coroutine` +
//! resume-loop paging) they replaced; the collapse deleted those ops, so the offer lane now holds
//! the absolute pin, with the deletion-time record below as the reference point.
//!
//! Shape notes: each child touches `P` pages at a 64 KiB stride, so the fault count is exactly
//! `P` on any host page size ≤ 64 KiB (supply maps only the host page containing the fault, and
//! the next touch is a stride away) — the checksum is host-independent. The parent respawns `S`
//! children, each into a fresh window; a fresh demand child faults every page, so total round-trips
//! are `S·P` with spawn cost amortized 1/P per fault. As in `serving_bench`, all backends must
//! agree on the checksum *before* timing (never benchmark a miscompile), so this file doubles as
//! the CI pin for N sequential fault-service rounds.
//!
//! Deletion-time record (the §2.3 evidence, measured on the last commit carrying both lanes,
//! after the §2.2b dispatch fast lane): bespoke TreeWalk 1430 ns / Bytecode 1828 ns / Jit
//! 28760 ns per fault (the JIT paid `sync_committed` window mirroring per switch); offer
//! TreeWalk 2364 ns [1.65x] / Bytecode 2409 ns [1.32x] / **Jit 2325 ns [0.08x — 12.4x faster,
//! the mirroring cost deleted by construction]**. The interp residual is the price of a real
//! concurrent child; the priced-but-not-queued parked-provider cache (~300-450 ns floor)
//! remains the lever if it ever matters.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use std::time::Instant;

use temen_ir::SpawnRec;
use temen_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use temen_text::parse_module;

/// Where the pager stages a page in its own window: this far above the fault address, past the
/// 128 strides of 64 KiB the child touches and inside the parent's 2^24 window.
const BUFFER_OFF: u64 = 1 << 23;
/// The byte the pager supplies at each fault address; the child sums what it reads.
const PAGE_BYTE: i64 = 7;
/// Per-fault marker the parent folds into the checksum, so a silently skipped fault (a page that
/// never unmapped, a supply that didn't stick) breaks the expected value rather than hiding.
const FAULT_MARK: i64 = 1000;

/// CONSOLIDATION.md §2.2 — the **offer-transport twin**: the same S x P shape, driven the
/// collapsed way. The parent spawns its own func 1 as a **detached demand child**, through a v1
/// record paid from its `"budget"` that names its own impl export 0 as the pager; the child walks
/// the strides of its own window concurrently, each first touch faulting into a `page(addr)` call
/// the parent serves from `svc.wait` (direct handoff runs the handler inline on the child's thread);
/// the parent counts serves (`FAULT_MARK` each), joins the child (its stride sum), and repeats. The
/// handler gets the fault address in the child's coordinates, stores the page's byte `BUFFER_OFF`
/// above it in its own window, and replies with that address (#1862). The checksum is identical to
/// the bespoke lane's by construction: the comparison this file exists for.
fn offer_paging_program(s: u64, p: u64) -> String {
    format!(
        "\
memory 24
data 16384 \"vm\"
data 16400 \"budget\"
{rec}type 0 func (i64) -> (i64)
type 1 interface {{ page: 0 }}
export 0 interface \"pager\" 1 {{ page: 2 }}
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 16384
  vl = i64.const 2
  vh = self.resolve vp vl
  vbp = i64.const 16400
  vbl = i64.const 6
  vb = self.resolve vbp vbl
  vrb = i64.const 17564
  i32.store vrb vb
  vs0 = i64.const 0
  vacc0 = i64.const 0
  br 1(vh, vs0, vacc0)
}}
block 1 (vh1: i32, vs: i64, vacc: i64) {{
  rra0 = i64.const 17536
  vch = call.cap 6 17 (i64) -> (i32) vh1 (rra0)
  vsrv0 = i64.const 0
  br 2(vh1, vs, vacc, vch, vsrv0)
}}
block 2 (vh2: i32, vs2: i64, vacc2: i64, vch2: i32, vsrv: i64) {{
  vz = i32.const 0
  vw = svc.wait vz
  vsrv2 = i64.add vsrv vw
  vpn = i64.const {p}
  vcmp = i64.lt_s vsrv2 vpn
  br_if vcmp 2(vh2, vs2, vacc2, vch2, vsrv2) 3(vh2, vs2, vacc2, vch2, vsrv2)
}}
block 3 (vh3: i32, vs3: i64, vacc3: i64, vch3: i32, vsrvf: i64) {{
  vj = call.cap 6 1 (i32) -> (i64) vh3 (vch3)
  vmk = i64.const {mark}
  vm = i64.mul vsrvf vmk
  vacc4 = i64.add vacc3 vj
  vacc5 = i64.add vacc4 vm
  vone = i64.const 1
  vs4 = i64.add vs3 vone
  vsn = i64.const {s}
  vcmp2 = i64.lt_s vs4 vsn
  br_if vcmp2 1(vh3, vs4, vacc5) 4(vacc5)
}}
block 4 (vaccf: i64) {{
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
  vaddrm = i64.mul vi vstride
  vguard = i64.const 16384
  vaddr = i64.add vaddrm vguard
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

func 2 (i64) -> (i64) {{
block 0 (vaddr: i64) {{
  vbuf = i64.const {off}
  vsrc = i64.add vaddr vbuf
  vb = i32.const {byte}
  i32.store8 vsrc vb
  return vsrc
  }}
}}
",
        rec = rec::segment(
            17536,
            &SpawnRec {
                pager: 0,
                ..SpawnRec::v1(1)
            }
        ),
        off = BUFFER_OFF,
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
    temen_verify::verify_module(&m).expect("verify");
    let registry = Imports::new().provide("exit", HostCap::exit());
    let inst = instantiate_with_imports(m, registry).expect("instantiate");
    let r = inst
        .run_with_caps(
            backend,
            &RunConfig::default(),
            &[
                (
                    "vm",
                    HostCap::custom(6, 0, |h, win| h.grant_instantiator(0, win)),
                ),
                // One live child at a time, each a 2^24 window.
                ("budget", HostCap::detached_budget(1 << 25)),
            ],
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
    let want = expected_exit(s, p);
    let offer = offer_paging_program(s, p);
    for b in BACKENDS {
        assert_eq!(
            run(b, &offer),
            want,
            "{b:?}: offer (record pager) {s}x{p} fault-service rounds"
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
    let want = expected_exit(s, p);
    let src = offer_paging_program(s, p);

    // Correctness first — never benchmark a miscompile.
    for b in BACKENDS {
        assert_eq!(run(b, &src), want, "{b:?}: checksum before timing");
    }

    println!("\nfault-service round-trip — {s} spawns x {p} faults = {faults}, per-fault ns");
    println!("(deletion-time record in the header is the reference point;");
    println!(" 4th backend, wasm-JIT: folds to bytecode — see wasm_jit_lane_folds_closed_today):");
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
        println!("  {b:?}: {:.0} ns/fault", best / faults as f64);
    }
}

/// The **fourth backend** — the wasm-JIT tier (`temen-wasm-jit`) — pinned so it is never silently
/// forgotten in this comparison. Today the pager parent sits outside the emitter's nested
/// subset (`svc.wait` serve loops have no bounce arms), so the tier **fails closed**
/// and the entry folds to the bytecode interpreter: the wasm-JIT row of the fault-service table
/// *is* the Bytecode row. This test asserts exactly that state.
/// When the emitter grows the arms, the assert flips — replace it with a real timed lane
/// (emit + run under `wasmi`, the `nested_vm.rs` harness shape) instead of deleting the test.
#[test]
fn wasm_jit_lane_folds_closed_today() {
    let src = offer_paging_program(1, 1);
    let m = parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let r = temen_wasm_jit::compile_module_nested(&m, false);
    assert!(
        r.is_err(),
        "wasm-jit: the emitter now accepts the pager shape — add the timed wasmi lane to the \
         probe (and to the fault-service table) instead of relying on the fold"
    );
}
