//! CONSOLIDATION.md §2.1/§2.3 — the **value round-trip** pin and latency probe on the unified
//! offer transport (probe is run on demand, not in CI):
//!
//!   cargo test -p svm --release --test coroutine_offer_bench -- --ignored --nocapture
//!
//! Until §2.3 this file raced the offer lane against the bespoke coroutine ops (Instantiator
//! ops 2/3 + `Yielder`) it replaced; the collapse deleted those ops, so the offer lane is now the
//! only transport and this pin holds the absolute per-round shape: the parent spawns a serving
//! child (op 11), mints a live offer over its `adder` export (op 14), and calls `add(i, 1)`
//! through it n times, the child replying from a `svc.wait` loop — checksum
//! `Σ_{i=0}^{n-1}(i+1) = n(n+1)/2`. Known asymmetry, same as `serving_bench` documents: the
//! op-14 mint + parking live call does not `serve_qualifies` on `Backend::Jit`, so that lane
//! folds to the tree-walk oracle — read the JIT row accordingly.
//!
//! Deletion-time record (the §2.3 evidence, measured on the last commit carrying both lanes):
//! bespoke resume/yield ~250 ns/round interp; offer ~740 ns interp (the residual = admission
//! word, per-call revocation check, provider dispatch, serve accounting — the price of a real
//! concurrent child); on the JIT the offer lane **beat** the bespoke coroutine ~3x (~1.1 us vs
//! ~3.5 us of per-switch committed-page window mirroring). The first 9-10x reading predated
//! `RunConfig::handoff` (2.1b) — it measured the queued transport. A parked-provider cache
//! (~300-450 ns floor) remains the priced-but-not-queued option if ~740 ns ever matters.

use std::time::Instant;

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// The offer lane — `serving_bench::serving_program` duplicated verbatim (test targets cannot
/// share code without a fixture crate; the duplication is deliberate so the two probes stay
/// independently runnable). Keep in sync with `serving_bench.rs` if the serving shape changes.
fn offer_program(n: u64) -> String {
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

/// Both lanes checksum to `Σ_{i=0}^{n-1}(i+1)` wrapped to i32.
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

/// Correctness pin (runs in CI): n offer serve rounds compute the expected checksum on every
/// backend — the §2 collapse statement (`resume` = `cap.call`, `yield` = the handler replying)
/// with the offer transport as the only transport.
#[test]
fn offer_rounds_agree_across_backends() {
    let n = 32;
    let want = expected_exit(n);
    let offer = offer_program(n);
    for b in BACKENDS {
        assert_eq!(run(b, &offer), want, "{b:?}: offer serve lane");
    }
}

/// The latency probe. Ignored (timing is machine-dependent and not a CI gate); the per-round
/// numbers track the offer transport against the deletion-time record in the header.
#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn value_roundtrip_latency() {
    let n = 20_000u64;
    let want = expected_exit(n);
    let src = offer_program(n);

    // Correctness first — never benchmark a miscompile.
    for b in BACKENDS {
        assert_eq!(run(b, &src), want, "{b:?}: checksum before timing");
    }

    println!("\nvalue round-trip — {n} rounds, per-round ns:");
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
        println!("  {b:?}: {:.0} ns/round", best / n as f64);
    }
}
