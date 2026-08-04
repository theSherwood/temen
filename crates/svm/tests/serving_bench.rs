//! Serving throughput probe (run on demand, not in CI):
//!
//!   cargo test -p svm --test serving_bench -- --ignored --nocapture
//!
//! **What it measures and why.** A §3.6 serving round-trip — a caller with a *live offer* over a
//! spawned child calls `add(i, 1)` through it, parking until the child's `svc.wait` serves the
//! request, wakes, and loops — is timed per-request across the three backends. The point is to
//! turn the **JIT-caller-side deferral** (IMPORTS.md §3.6: "Native fast-backend serving … waits
//! for benchmark evidence") into a *measured* decision rather than a guess.
//!
//! The fold this exposes (documented in `svm_run::Instance::run_with_caps`): a module that mints
//! an op-14 offer and makes a **parking** live call does not `serve_qualifies`, so on `Backend::Jit`
//! it folds to the tree-walk oracle. This benchmark therefore reads as three lanes:
//!   - **TreeWalk** — the reference oracle serve path.
//!   - **Bytecode** — the native bytecode serve loop + caller park/wake driver.
//!   - **Jit** — folds to TreeWalk for this shape (op-14 mint + parking live call), so its time
//!     should track TreeWalk's. That equality *is* the deferral cost: the JIT recovers nothing on
//!     the serving round-trip today.
//!
//! The Bytecode ÷ TreeWalk ratio is the quantity that informs the deferral: a large native win
//! there is evidence a native **JIT** caller side is worth building; a wash says the round-trip is
//! dominated by park/wake scheduling, not per-op codegen, and the deferral is well-placed. We
//! assert all three backends agree on the checksum *before* timing (never benchmark a miscompile),
//! so this file doubles as an N-round serial-serve correctness pin — the existing svc parity tests
//! only serve a single request.

use std::time::Instant;

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// A caller `_start` that spawns a serving child (func 1), mints a live offer over its `adder`
/// export (op 14), then calls `add(i, 1)` through the offer `n` times — parking on each call until
/// the child serves it — accumulating the replies, joins the child, and exits with the low 32 bits
/// of the sum. (CALLS.md 5c.1c: the spawn is op 11 with an empty grant list — a plain op-0
/// child is destitute by design on the JIT, and a serving child needs the shared granted
/// powerbox the parked transport rides; interp/bytecode semantics unchanged.) The child serves exactly `n` requests in a `svc.wait` loop, then returns. The sum is
/// `Σ_{i=0}^{n-1} (i + 1) = n(n+1)/2` (wrapped to i32 at exit) — deterministic across backends.
fn serving_program(n: u64) -> String {
    format!(
        "\
memory 17
data 0 \"vm\"
type 0 func (i64, i64) -> (i64)
type 1 interface {{ add: 0 }}
export 0 interface \"adder\" 1 {{ add: 2 }}
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 0
  vl = i64.const 2
  vh = cap.self.resolve vp vl
  vgp = i64.const 0
  vgn = i64.const 0
  ventry = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 12
  vq = i64.const 0
  vch = cap.call 6 11 (i64, i64, i64, i64, i64, i64) -> (i32) vh (vgp, vgn, ventry, voff, vsl, vq)
  vexp = i64.const 0
  voffer = cap.call 6 14 (i32, i64) -> (i32) vh (vch, vexp)
  vi0 = i64.const 0
  vacc0 = i64.const 0
  br 1(vh, vch, voffer, vi0, vacc0)
}}
block 1 (vh1: i32, vch1: i32, voffer1: i32, vi: i64, vacc: i64) {{
  vone = i64.const 1
  vr = cap.call 268435456 0 (i64, i64) -> (i64) voffer1 (vi, vone)
  vacc2 = i64.add vacc vr
  vi2 = i64.add vi vone
  vn = i64.const {n}
  vcmp = i64.lt_s vi2 vn
  br_if vcmp 1(vh1, vch1, voffer1, vi2, vacc2) 2(vh1, vch1, vacc2)
}}
block 2 (vh2: i32, vch2: i32, vaccf: i64) {{
  vj = cap.call 6 1 (i32) -> (i64) vh2 (vch2)
  vc = i32.wrap_i64 vaccf
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vc0 = i64.const 0
  br 1(vc0)
}}
block 1 (vc: i64) {{
  vz = i32.const 0
  vs = svc.wait vz
  vc2 = i64.add vc vs
  vn = i64.const {n}
  vcmp = i64.lt_s vc2 vn
  br_if vcmp 1(vc2) 2(vc2)
}}
block 2 (vcf: i64) {{
  return vcf
  }}
}}

func 2 (i64, i64) -> (i64) {{
block 0 (va: i64, vb: i64) {{
  vs = i64.add va vb
  return vs
  }}
}}
"
    )
}

/// Sum `Σ_{i=0}^{n-1}(i+1)` wrapped to i32 (the exit code the program returns).
fn expected_exit(n: u64) -> i32 {
    let mut acc = 0i64;
    for i in 0..n {
        acc = acc.wrapping_add(i as i64 + 1);
    }
    acc as i32
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

/// Correctness pin (runs in CI): all three backends agree on the checksum for a serial N-round
/// serve, and it equals `n(n+1)/2` wrapped. This is the piece the existing single-request svc
/// parity tests do not cover — a serve loop driven through N sequential park/wake rounds.
#[test]
fn serial_serve_rounds_agree_across_backends() {
    let n = 64;
    let src = serving_program(n);
    let want = expected_exit(n);
    for b in BACKENDS {
        assert_eq!(
            run(b, &src),
            want,
            "{b:?}: {n} serial serve rounds must checksum to n(n+1)/2"
        );
    }
}

/// The throughput probe. Ignored (timing is machine-dependent and not a CI gate); run manually.
#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn serving_roundtrip_throughput() {
    let n = 20_000u64;
    let src = serving_program(n);
    let want = expected_exit(n);

    // Correctness first — never benchmark a miscompile.
    for b in BACKENDS {
        assert_eq!(run(b, &src), want, "{b:?}: checksum before timing");
    }

    println!("\nserving round-trip — {n} requests, per-request ns:");
    let mut treewalk_ns = 0f64;
    let mut bytecode_ns = 0f64;
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
        let per_req = best / n as f64;
        match b {
            Backend::TreeWalk => treewalk_ns = per_req,
            Backend::Bytecode => bytecode_ns = per_req,
            _ => {}
        }
        println!("  {b:?}: {per_req:.0} ns/req  (total {:.2} ms)", best / 1e6);
    }
    // The number that informs the §3.6 JIT-caller-side deferral: how much the native bytecode serve
    // loop beats the folded oracle. Jit ≈ TreeWalk by the fold, so this ratio bounds what a native
    // JIT caller side could recover on this shape.
    if treewalk_ns.is_finite() && bytecode_ns > 0.0 {
        println!(
            "  native serve win (TreeWalk ÷ Bytecode): {:.2}x",
            treewalk_ns / bytecode_ns
        );
    }
}
