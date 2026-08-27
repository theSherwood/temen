//! Single-vCPU fiber-overlap probe (run on demand, not in CI):
//!
//!   cargo test -p temen --test fiber_overlap_bench -- --ignored --nocapture
//!
//! **What it measures and why.** The IoRing retirement (DESIGN.md §12) recorded one known loss:
//! `submit_async`/`reap` was the only *single-vCPU* compute/host overlap. Fiber-park promotion
//! (the FIBER_PARK arc) is its replacement: one vCPU resumes `n` fibers, each punting one
//! blocking `call.cap` (block = 2 ms) — every punt's pool job is dispatched at park time, so all
//! `n` co-reside on the K=4 offload pool and the batch completes in ~ceil(n/K)·2 ms instead of
//! the serialized n·2 ms. The **root lane** (the same `n` punts made sequentially from the root,
//! each vCPU-parking alone) is the in-file serialized baseline; the retirement record's anchors
//! (ring ~5.2–5.8 ms/batch, threaded lane ~5.1–5.5 ms at n=8/block=2 ms/`max_active` 4;
//! 18.5 ms serialized) are the historical comparison.
//!
//! Checksums are asserted across all three engines *before* timing (never benchmark a
//! miscompile); the timing itself is print-only — the rendezvous-pinned correctness lives in
//! `fiber_punt_jit.rs` / `fiber_punt_diff.rs`, never here.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::time::{Duration, Instant};

use temen_interp::{bytecode, run_with_host, Host, Value};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

/// The `Blocking` cap's deterministic transform (mirrors `AsyncState::mix`).
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// Fiber lane `(i32 h) -> (i64)`: create `n` fibers (handles stored at window 16400+8i, above the #1094 NULL guard), resume
/// each once (each punts `Blocking.work(i)` and parks — its pool job is now in flight), then
/// poll each to completion in order, summing `mix(i)`. The resume arg packs `h | (i << 16)`.
fn fiber_lane(n: u64) -> String {
    format!(
        "memory 16
func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vh64 = i64.extend_i32_u v0
  vi0 = i64.const 0
  br 1(vi0, vh64)
}}
block 1 (vi: i64, vh: i64) {{
  vn = i64.const {n}
  vlt = i64.lt_u vi vn
  br_if vlt 2(vi, vh) 3(vh)
}}
block 2 (vi2: i64, vh2: i64) {{
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vsh = i64.const 16
  vsw = i64.shl vi2 vsh
  varg = i64.or vh2 vsw
  vs, vv = cont.resume vk varg
  veight = i64.const 8
  voff = i64.mul vi2 veight
  vb = i64.const 16400
  va = i64.add vb voff
  i64.store va vk
  vone = i64.const 1
  vnext = i64.add vi2 vone
  br 1(vnext, vh2)
}}
block 3 (vh3: i64) {{
  vj0 = i64.const 0
  vsum0 = i64.const 0
  br 4(vj0, vsum0, vh3)
}}
block 4 (vj: i64, vsum: i64, vh4: i64) {{
  vn2 = i64.const {n}
  vlt2 = i64.lt_u vj vn2
  br_if vlt2 5(vj, vsum, vh4) 8(vsum)
}}
block 5 (vj2: i64, vsum2: i64, vh5: i64) {{
  veight2 = i64.const 8
  voff2 = i64.mul vj2 veight2
  vb2 = i64.const 16400
  va2 = i64.add vb2 voff2
  vk2 = i64.load va2
  vsh2 = i64.const 16
  vsw2 = i64.shl vj2 vsh2
  varg2 = i64.or vh5 vsw2
  br 6(vj2, vsum2, vh5, vk2, varg2)
}}
block 6 (vj3: i64, vsum3: i64, vh6: i64, vk3: i64, varg3: i64) {{
  vs3, vv3 = cont.resume vk3 varg3
  vone3 = i32.const 1
  vdone = i32.eq vs3 vone3
  br_if vdone 7(vj3, vsum3, vh6, vv3) 6(vj3, vsum3, vh6, vk3, varg3)
}}
block 7 (vj4: i64, vsum4: i64, vh7: i64, vv4: i64) {{
  vsum5 = i64.add vsum4 vv4
  vone4 = i64.const 1
  vjn = i64.add vj4 vone4
  br 4(vjn, vsum5, vh7)
}}
block 8 (vr: i64) {{
  return vr
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vmask = i64.const 65535
  vh = i64.and varg vmask
  vhi = i32.wrap_i64 vh
  vsh = i64.const 16
  vi = i64.shr_u varg vsh
  vr = call.cap 10 0 (i64) -> (i64) vhi (vi)
  return vr
  }}
}}
"
    )
}

/// Root lane `(i32 h) -> (i64)`: the same `n` punts made sequentially from the root — each
/// vCPU-parks alone, so the batch serializes at n·block (the pre-arc shape, the baseline).
fn root_lane(n: u64) -> String {
    format!(
        "func (i32) -> (i64) {{
block 0 (v0: i32) {{
  vi0 = i64.const 0
  vs0 = i64.const 0
  br 1(v0, vi0, vs0)
}}
block 1 (vh: i32, vi: i64, vsum: i64) {{
  vn = i64.const {n}
  vlt = i64.lt_u vi vn
  br_if vlt 2(vh, vi, vsum) 3(vsum)
}}
block 2 (vh2: i32, vi2: i64, vsum2: i64) {{
  vr = call.cap 10 0 (i64) -> (i64) vh2 (vi2)
  vsum3 = i64.add vsum2 vr
  vone = i64.const 1
  vnext = i64.add vi2 vone
  br 1(vh2, vnext, vsum3)
}}
block 3 (vr2: i64) {{
  return vr2
  }}
}}
"
    )
}

fn parse(src: &str) -> temen_ir::Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

const N: u64 = 8;
const BLOCK: Duration = Duration::from_millis(2);
const WARM: usize = 2;
const TIMED: usize = 10;

fn want(n: u64) -> i64 {
    (0..n as i64).map(mix).fold(0i64, |a, b| a.wrapping_add(b))
}

fn time_lane(label: &str, w: i64, mut run_once: impl FnMut() -> i64) {
    for _ in 0..WARM {
        assert_eq!(
            run_once(),
            w,
            "{label}: checksum (never benchmark a miscompile)"
        );
    }
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    for _ in 0..TIMED {
        let t0 = Instant::now();
        let got = run_once();
        let dt = t0.elapsed();
        assert_eq!(got, w, "{label}: checksum");
        best = best.min(dt);
        total += dt;
    }
    println!(
        "  {label:<28} best {:>7.2} ms   avg {:>7.2} ms",
        best.as_secs_f64() * 1e3,
        total.as_secs_f64() * 1e3 / TIMED as f64
    );
}

#[test]
#[ignore = "perf probe — run with --ignored --nocapture"]
fn single_vcpu_fiber_overlap_vs_serialized_root() {
    let fiber_m = parse(&fiber_lane(N));
    let root_m = parse(&root_lane(N));
    let floor_m = parse(&fiber_lane(0));
    println!(
        "\nn={N} punts, block={:?}, pool K=4 — anchors (DESIGN §12): ring ~5.2–5.8 ms, \
         threaded ~5.1–5.5 ms, serialized 18.5 ms",
        BLOCK
    );

    println!("fiber lane (one vCPU, n fibers — the promotion overlap; jit lanes include a per-run compile):");
    time_lane("tree-walk", want(N), || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        let mut fuel = u64::MAX;
        match run_with_host(&fiber_m, 0, &[Value::I32(hh)], &mut fuel, &mut h)
            .expect("oracle ok")
            .pop()
        {
            Some(Value::I64(x)) => x,
            o => panic!("unexpected {o:?}"),
        }
    });
    time_lane("bytecode", want(N), || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        let mut fuel = u64::MAX;
        match bytecode::compile_and_run_with_host(&fiber_m, 0, &[Value::I32(hh)], &mut fuel, &mut h)
            .expect("in subset")
            .expect("bytecode ok")
            .pop()
        {
            Some(Value::I64(x)) => x,
            o => panic!("unexpected {o:?}"),
        }
    });
    time_lane("cranelift-jit", want(N), || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        match temen_jit::compile_and_run_with_host(
            &fiber_m,
            0,
            &[hh as i64],
            temen_run::cap_thunk,
            &mut h as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit compiles")
        {
            JitOutcome::Returned(vals) => vals[0],
            o => panic!("unexpected jit outcome {o:?}"),
        }
    });

    println!("root lane (sequential punts — the serialized baseline):");
    time_lane("tree-walk (root)", want(N), || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        let mut fuel = u64::MAX;
        match run_with_host(&root_m, 0, &[Value::I32(hh)], &mut fuel, &mut h)
            .expect("oracle ok")
            .pop()
        {
            Some(Value::I64(x)) => x,
            o => panic!("unexpected {o:?}"),
        }
    });
    // The JIT entry bakes the thunk/ctx into the code (compile, run once, discard), so both JIT
    // lanes pay a per-run Cranelift compile (~10–15 ms); the FIBER-vs-ROOT delta on this same
    // engine is the overlap signal, not the absolute number. The compile-floor lane below (the
    // fiber module at n=0 — identical shape, zero iterations, zero punts) isolates the
    // compile+overhead constant the two lanes share.
    time_lane("cranelift-jit (floor)", 0, || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        match temen_jit::compile_and_run_with_host(
            &floor_m,
            0,
            &[hh as i64],
            temen_run::cap_thunk,
            &mut h as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit compiles")
        {
            JitOutcome::Returned(vals) => vals[0],
            o => panic!("unexpected jit outcome {o:?}"),
        }
    });
    time_lane("cranelift-jit (root)", want(N), || {
        let mut h = Host::new();
        let hh = h.grant_blocking(BLOCK, None);
        match temen_jit::compile_and_run_with_host(
            &root_m,
            0,
            &[hh as i64],
            temen_run::cap_thunk,
            &mut h as *mut Host as *mut core::ffi::c_void,
        )
        .expect("jit compiles")
        {
            JitOutcome::Returned(vals) => vals[0],
            o => panic!("unexpected jit outcome {o:?}"),
        }
    });
}

/// Diagnostic (gating): the fiber lane must actually PUNT on every engine — a lane that ran its
/// blocking calls inline would silently benchmark the wrong thing.
#[test]
fn the_fiber_lane_punts_on_every_engine() {
    let m = parse(&fiber_lane(N));
    let w = want(N);

    let mut h = Host::new();
    let hh = h.grant_blocking(BLOCK, None);
    let comps = h.completions();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(hh)], &mut fuel, &mut h).expect("oracle ok");
    assert_eq!(r, vec![Value::I64(w)]);
    assert_eq!(comps.minted(), N, "oracle: one punt per fiber");

    let mut h = Host::new();
    let hh = h.grant_blocking(BLOCK, None);
    let comps = h.completions();
    let jo = temen_jit::compile_and_run_with_host(
        &m,
        0,
        &[hh as i64],
        temen_run::cap_thunk,
        &mut h as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit compiles");
    match jo {
        JitOutcome::Returned(vals) => assert_eq!(vals[0], w),
        o => panic!("unexpected jit outcome {o:?}"),
    }
    assert_eq!(comps.minted(), N, "jit: one punt per fiber");
}
