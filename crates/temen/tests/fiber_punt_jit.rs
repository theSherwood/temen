//! F3 (FIBER_PARK.md) — **punt-inside-a-fiber, TreeWalk ≡ Bytecode ≡ Cranelift JIT** — the pin
//! that closes ISSUES.md I73. The JIT's `cap_thunk`/`cap_thunk_locked` now route a fiber's punt
//! through the pending face and park the FIBER (`fiber_cap_wait` over the `Completions` fiber
//! cells + the `fiber_rt` event-park seam), so every F1 kernel produces the identical composite
//! on all three engines — statuses, values, and delivery alike (all-integer kernels, so
//! the NaN-insensitive JIT contract is bit-exact here). Kernels duplicated verbatim from
//! `temen-interp/tests/fiber_punt_diff.rs` (no fixture crate — the `host_park.rs` precedent).
//!
//! Stack switching exists on x86-64 unix, aarch64 unix, and x86-64 Windows today
//! (`temen_fiber::supported()`); elsewhere the JIT bails `Unsupported`, so gated like
//! `jit_fibers.rs`.
#![cfg(any(
    all(unix, target_arch = "x86_64"),
    all(unix, target_arch = "aarch64"),
    all(windows, target_arch = "x86_64")
))]

use std::sync::Arc;
use std::time::Duration;

use temen_interp::{
    bytecode, run_with_host, Host, HostProcGrant, OffloadHostProc, OffloadOutcome, ProcHandler,
    Value,
};
use temen_jit::JitOutcome;
use temen_text::parse_module;
use temen_verify::verify_module;

/// The deterministic transform the `Blocking` cap's jobs apply (mirrors `AsyncState::mix`).
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

fn module(text: &str) -> temen_ir::Module {
    let m = parse_module(text).expect("parse");
    verify_module(&m).expect("verify");
    m
}

/// Run `m` on all three engines against per-engine hosts from `mk_host`, asserting the identical
/// `want` composite; each engine must deliver every completion it mints, and the punt counts
/// must agree everywhere (the JIT fiber path mints exactly like the parking-aware tiers).
fn pin_three(m: &temen_ir::Module, mk_host: &dyn Fn() -> (Host, i32), want: i64, label: &str) {
    // Tree-walk oracle.
    let (mut hi, h) = mk_host();
    let comps_i = hi.completions();
    let mut fuel = 2_000_000_000u64;
    let r = run_with_host(m, 0, &[Value::I32(h)], &mut fuel, &mut hi).expect("oracle: no trap");
    assert_eq!(r, vec![Value::I64(want)], "{label}: tree-walk composite");
    hi.quiesce_pool();

    // Bytecode cooperative driver.
    let (mut hb, hbh) = mk_host();
    let comps_b = hb.completions();
    let mut fuel = 2_000_000_000u64;
    let r = bytecode::compile_and_run_with_host(m, 0, &[Value::I32(hbh)], &mut fuel, &mut hb)
        .expect("bytecode: in subset")
        .expect("bytecode: no trap");
    assert_eq!(r, vec![Value::I64(want)], "{label}: bytecode composite");
    hb.quiesce_pool();

    // Cranelift JIT — the F3 fiber path through `cap_thunk`.
    let (mut hj, hjh) = mk_host();
    let comps_j = hj.completions();
    let jo = temen_jit::compile_and_run_with_host(
        m,
        0,
        &[hjh as i64],
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
    )
    .expect("jit compiles");
    let jv = match jo {
        JitOutcome::Returned(vals) => vals[0],
        o => panic!("{label}: unexpected jit outcome {o:?}"),
    };
    assert_eq!(jv, want, "{label}: Cranelift JIT composite");
    hj.quiesce_pool();

    assert_eq!(comps_i.outstanding(), 0, "{label}: oracle delivered all");
    assert_eq!(comps_b.outstanding(), 0, "{label}: bytecode delivered all");
    assert_eq!(comps_j.outstanding(), 0, "{label}: jit delivered all");
    assert_eq!(
        (comps_i.minted(), comps_b.minted()),
        (comps_j.minted(), comps_j.minted()),
        "{label}: same punt count on all three engines"
    );
}

const PUNT_IN_FIBER: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vh64 = i64.extend_i32_u v0
  vs1, vv1 = cont.resume vk vh64
  br 1(vk, vh64, vs1)
}
block 1 (vk1: i64, vh1: i64, vfirst: i32) {
  vs, vv = cont.resume vk1 vh1
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vfirst, vv) 1(vk1, vh1, vfirst)
}
block 2 (vf2: i32, vres: i64) {
  vk4 = i64.const 10000
  vfe = i64.extend_i32_s vf2
  va = i64.mul vfe vk4
  vr = i64.add va vres
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vfive = i64.const 5
  vr = call.cap 13 0 (i64) -> (i64) vh (vfive)
  return vr
  }
}
"#;

fn punting_handler() -> OffloadHostProc {
    Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        OffloadOutcome::Offload(Box::new(move || a + 100))
    })
}

#[test]
fn a_punt_in_a_fiber_parks_identically_on_all_three_engines() {
    let m = module(PUNT_IN_FIBER);
    pin_three(
        &m,
        &|| {
            let mut h = Host::new();
            let hh = h.grant_host_proc(HostProcGrant::new(
                ProcHandler::Offloadable(punting_handler()),
                temen_interp::CapState::Stateless,
            ));
            (h, hh)
        },
        30_105,
        "punt-in-fiber",
    );
}

const TWO_FIBER_OVERLAP: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh64 = i64.extend_i32_u v0
  vfa = ref.func 1
  vz = i64.const 0
  vka = cont.new vfa vz
  vfb = ref.func 2
  vkb = cont.new vfb vz
  vsa, vva = cont.resume vka vh64
  vsb, vvb = cont.resume vkb vh64
  br 1(vka, vkb, vh64, vsa, vsb)
}
block 1 (k1a: i64, k1b: i64, h1: i64, s1a: i32, s1b: i32) {
  vs, vv = cont.resume k1a h1
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(k1b, h1, s1a, s1b, vv) 1(k1a, k1b, h1, s1a, s1b)
}
block 2 (k2b: i64, h2: i64, s2a: i32, s2b: i32, va: i64) {
  vs2, vv2 = cont.resume k2b h2
  vone2 = i32.const 1
  vdone2 = i32.eq vs2 vone2
  br_if vdone2 3(s2a, s2b, va, vv2) 2(k2b, h2, s2a, s2b, va)
}
block 3 (s3a: i32, s3b: i32, va3: i64, vb3: i64) {
  vten = i64.const 10
  vae = i64.extend_i32_s s3a
  vbe = i64.extend_i32_s s3b
  vp = i64.mul vae vten
  vcomp = i64.add vp vbe
  vsum = i64.add va3 vb3
  vr = i64.add vsum vcomp
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 0
  vr = call.cap 10 0 (i64) -> (i64) vh (va)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 1
  vr = call.cap 10 0 (i64) -> (i64) vh (va)
  return vr
  }
}
"#;

/// **The I73-closing overlap**: one OS thread, two fibers, a rendezvous-2 `Blocking` handle —
/// on a pre-F3 JIT the first fiber's punt ran inline on the vCPU thread and blocked at the
/// width-2 barrier forever (the second job could never be submitted). Completing at all — with
/// `max_active == 2` — proves the fiber parked and the vCPU kept running.
#[test]
fn the_single_vcpu_overlap_holds_identically_on_all_three_engines() {
    let m = module(TWO_FIBER_OVERLAP);
    pin_three(
        &m,
        &|| {
            let mut h = Host::new();
            let hh = h.grant_blocking(Duration::ZERO, Some(2));
            (h, hh)
        },
        mix(0).wrapping_add(mix(1)).wrapping_add(33),
        "two-fiber overlap",
    );
}

/// `fiber_parks.rs::LATER_ANSWER_FIRST` — a later call's answer reaches its fiber while an
/// earlier call is still held (#2477), identically on all three engines: fiber 2 returns within its
/// 1,000 polls, fiber 1 is still parked, then both deliver — 13_111_222.
const LATER_ANSWER_FIRST: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vh64 = i64.extend_i32_u v0
  vf1 = ref.func 1
  vz = i64.const 0
  vk1 = cont.new vf1 vz
  vf2 = ref.func 2
  vk2 = cont.new vf2 vz
  vsa, vva = cont.resume vk1 vh64
  vsb, vvb = cont.resume vk2 vh64
  vi0 = i64.const 0
  br 1(vk1, vk2, vh64, vi0)
}
block 1 (ka: i64, kb: i64, h1: i64, vi: i64) {
  vlim = i64.const 1000
  vlt = i64.lt_s vi vlim
  vno = i64.const 0
  br_if vlt 2(ka, kb, h1, vi) 3(ka, h1, vno, vno)
}
block 2 (ka2: i64, kb2: i64, h2: i64, vi2: i64) {
  vpa = i64.const 16384
  vpe = i32.const 0
  vpt = i64.const 1000000
  vps = i32.atomic.wait vpa vpe vpt
  vs, vv = cont.resume kb2 h2
  vone = i32.const 1
  vdone = i32.eq vs vone
  vstep = i64.const 1
  vi3 = i64.add vi2 vstep
  vyes = i64.const 1
  br_if vdone 3(ka2, h2, vyes, vv) 1(ka2, kb2, h2, vi3)
}
block 3 (ka3: i64, h3: i64, vok: i64, v2: i64) {
  vs1, vv1 = cont.resume ka3 h3
  vheld = i64.extend_i32_u vs1
  vh = i32.wrap_i64 h3
  vz3 = i64.const 0
  vrel = call.cap 13 1 (i64) -> (i64) vh (vz3)
  br 4(ka3, h3, vok, v2, vheld)
}
block 4 (ka4: i64, h4: i64, vok4: i64, v24: i64, vheld4: i64) {
  vs5, vv5 = cont.resume ka4 h4
  vone5 = i32.const 1
  vdone5 = i32.eq vs5 vone5
  br_if vdone5 5(vok4, v24, vheld4, vv5) 4(ka4, h4, vok4, v24, vheld4)
}
block 5 (vok5: i64, v25: i64, vheld5: i64, v1: i64) {
  vm7 = i64.const 10000000
  va = i64.mul vok5 vm7
  vm6 = i64.const 1000000
  vb = i64.mul vheld5 vm6
  vk = i64.const 1000
  vc = i64.mul v1 vk
  vab = i64.add va vb
  vabc = i64.add vab vc
  vr = i64.add vabc v25
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 0
  vr = call.cap 13 0 (i64) -> (i64) vh (va)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 1
  vr = call.cap 13 0 (i64) -> (i64) vh (va)
  return vr
  }
}
"#;

fn gated_handler() -> OffloadHostProc {
    let gate = Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    Box::new(move |op, args| {
        if op == 1 {
            let (mx, cv) = &*gate;
            *mx.lock().unwrap() = true;
            cv.notify_all();
            return OffloadOutcome::Done(Ok(vec![0]));
        }
        let a = *args.first().unwrap_or(&0);
        if a == 0 {
            let g = Arc::clone(&gate);
            OffloadOutcome::Offload(Box::new(move || {
                let (mx, cv) = &*g;
                let mut open = mx.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
                111
            }))
        } else {
            OffloadOutcome::Offload(Box::new(move || 222))
        }
    })
}

#[test]
fn a_later_answer_reaches_its_fiber_first_on_all_three_engines() {
    let m = module(LATER_ANSWER_FIRST);
    pin_three(
        &m,
        &|| {
            let mut h = Host::new();
            let hh = h.grant_host_proc(HostProcGrant::new(
                ProcHandler::Offloadable(gated_handler()),
                temen_interp::CapState::Stateless,
            ));
            (h, hh)
        },
        13_111_222,
        "a later answer first",
    );
}

const RESUME_ONCE: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vz = i64.const 0
  vk = cont.new vf vz
  vh64 = i64.extend_i32_u v0
  vs, vv = cont.resume vk vh64
  vk4 = i64.const 10000
  vse = i64.extend_i32_s vs
  va = i64.mul vse vk4
  vr = i64.add va vv
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vfive = i64.const 5
  vr = call.cap 13 0 (i64) -> (i64) vh (vfive)
  return vr
  }
}
"#;

#[test]
fn root_return_abandons_a_cap_parked_fiber_identically_on_all_three() {
    let m = module(RESUME_ONCE);
    let mk = || {
        let mut h = Host::new();
        let hh = h.grant_host_proc(HostProcGrant::new(
            ProcHandler::Offloadable(Box::new(|_op, args| {
                let a = *args.first().unwrap_or(&0);
                OffloadOutcome::Offload(Box::new(move || {
                    std::thread::sleep(Duration::from_millis(100));
                    a + 100
                }))
            })),
            temen_interp::CapState::Stateless,
        ));
        (h, hh)
    };
    let t0 = std::time::Instant::now();
    pin_three(&m, &mk, 30_000, "abandoned cap park");
    assert!(
        t0.elapsed() < Duration::from_secs(15),
        "teardown must not wait out the abandoned fibers beyond the pool quiesces"
    );
}
