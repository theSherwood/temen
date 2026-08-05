//! CONSOLIDATION.md §2.1 — the **value-coroutine ⇄ offer equivalence** pin and latency probe
//! (probe is run on demand, not in CI):
//!
//!   cargo test -p svm --release --test coroutine_offer_bench -- --ignored --nocapture
//!
//! The §2 collapse says a coroutine child *is* a process-provider child: `resume` is a `cap.call`
//! through a live offer, `yield` is the handler replying. This file makes that claim executable
//! **before** anything is deleted, as two programs computing the identical checksum
//! `Σ_{i=0}^{n-1}(i+1) = n(n+1)/2`:
//!
//!   - **bespoke** — the parent `spawn_coroutine`s a child (Instantiator op 2) and drives n
//!     resume/yield rounds: resume k delivers `k`, the child yields `k+1` (returning on the last
//!     round so the parent never abandons a suspended coroutine).
//!   - **offer** — `serving_bench`'s shape verbatim: the parent spawns a serving child (op 11),
//!     mints a live offer over its `adder` export (op 14), and calls `add(i, 1)` through it n
//!     times, the child replying from a `svc.wait` loop.
//!
//! The CI pin asserts both programs agree with each other and across all three backends — the
//! semantic half of the §2 gate. The `#[ignore]`d probe times both transports per-round; the
//! offer ÷ bespoke ratio per tier is the migration cost §2.3 must accept (or drive down) on the
//! plain value-ping shape, complementing `paging_bench`'s fault-service pin. Known asymmetry,
//! same as `serving_bench` documents: the bespoke lane runs natively on all three backends, while
//! the offer lane's op-14 mint + parking live call does not `serve_qualifies` on `Backend::Jit`,
//! so that lane folds to the tree-walk oracle — read the JIT column accordingly.

use std::time::Instant;

use svm_run::{instantiate_with_imports, Backend, HostCap, Imports, Outcome, RunConfig};
use svm_text::parse_module;

/// The bespoke lane: n resume/yield rounds through Instantiator op 2 / op 3 and the child's
/// `Yielder` (iface 7), whose handle arrives as the child's entry argument (the spawn arm passes
/// `Value::I64(yielder_handle)`). Child carve: 64 KiB window at offset 64 KiB in a 128 KiB parent.
fn bespoke_program(n: u64) -> String {
    format!(
        "\
memory 17
data 0 \"vm\"
import 0 \"exit\" (i32) -> ()

func 0 () -> () {{
block 0 () {{
  vp = i64.const 0
  vl = i64.const 2
  vh = cap.self.resolve vp vl
  ventry = i64.const 1
  voff = i64.const 65536
  vsl = i64.const 16
  vq = i64.const 0
  vch = cap.call 6 2 (i64, i64, i64, i64) -> (i32) vh (ventry, voff, vsl, vq)
  vi0 = i64.const 0
  vacc0 = i64.const 0
  br 1(vh, vch, vi0, vacc0)
}}
block 1 (vh1: i32, vch1: i32, vi: i64, vacc: i64) {{
  vst, vy = cap.call 6 3 (i32, i64) -> (i32, i64) vh1 (vch1, vi)
  vacc2 = i64.add vacc vy
  vone = i64.const 1
  vi2 = i64.add vi vone
  vn = i64.const {n}
  vcmp = i64.lt_s vi2 vn
  br_if vcmp 1(vh1, vch1, vi2, vacc2) 2(vacc2)
}}
block 2 (vaccf: i64) {{
  vc = i32.wrap_i64 vaccf
  call.import 0 (vc)
  unreachable
  }}
}}

func 1 (i64) -> (i64) {{
block 0 (v0: i64) {{
  vd0 = i64.const 0
  vk0 = i64.const 1
  br 1(v0, vd0, vk0)
}}
block 1 (vyh: i64, vd: i64, vk: i64) {{
  vone = i64.const 1
  vy = i64.add vd vone
  vn = i64.const {n}
  vcmp = i64.lt_s vk vn
  br_if vcmp 2(vyh, vk, vy) 3(vy)
}}
block 2 (vyh2: i64, vk2: i64, vy2: i64) {{
  vhh = i32.wrap_i64 vyh2
  vr = cap.call 7 0 (i64) -> (i64) vhh (vy2)
  vone2 = i64.const 1
  vk3 = i64.add vk2 vone2
  br 1(vyh2, vr, vk3)
}}
block 3 (vyf: i64) {{
  return vyf
  }}
}}
",
    )
}

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

/// Correctness pin (runs in CI): the bespoke resume/yield rounds and the offer serve rounds
/// compute the identical checksum on every backend — the CONSOLIDATION.md §2 collapse statement
/// (`resume` = `cap.call`, `yield` = the handler replying) as an executable equivalence.
#[test]
fn bespoke_and_offer_rounds_agree_across_backends() {
    let n = 32;
    let want = expected_exit(n);
    let bespoke = bespoke_program(n);
    let offer = offer_program(n);
    for b in BACKENDS {
        assert_eq!(run(b, &bespoke), want, "{b:?}: bespoke resume/yield lane");
        assert_eq!(run(b, &offer), want, "{b:?}: offer serve lane");
    }
}

/// The latency probe. Ignored (timing is machine-dependent and not a CI gate); the offer ÷
/// bespoke per-round ratio per backend is the §2.3 migration cost on the value-ping shape.
#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn value_roundtrip_latency() {
    let n = 20_000u64;
    let want = expected_exit(n);
    let lanes = [("bespoke", bespoke_program(n)), ("offer", offer_program(n))];

    // Correctness first — never benchmark a miscompile.
    for (name, src) in &lanes {
        for b in BACKENDS {
            assert_eq!(run(b, src), want, "{b:?}/{name}: checksum before timing");
        }
    }

    println!("\nvalue round-trip — {n} rounds, per-round ns (offer ÷ bespoke in brackets):");
    for b in BACKENDS {
        let mut per: [f64; 2] = [0.0; 2];
        for (li, (_, src)) in lanes.iter().enumerate() {
            // Warm, then best-of-5 to shed scheduler noise.
            run(b, src);
            let mut best = f64::INFINITY;
            for _ in 0..5 {
                let t = Instant::now();
                let got = run(b, src);
                let elapsed = t.elapsed().as_nanos() as f64;
                assert_eq!(got, want);
                best = best.min(elapsed);
            }
            per[li] = best / n as f64;
        }
        println!(
            "  {b:?}: bespoke {:.0} ns  offer {:.0} ns  [{:.2}x]",
            per[0],
            per[1],
            per[1] / per[0]
        );
    }
}
