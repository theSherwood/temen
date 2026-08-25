//! §3.6 slice 5a — **fiber-level park routing**: a park inside a fiber parks the FIBER, not
//! the vCPU (DESIGN.md "blocks the fiber, never the domain"). The resumer observes a third
//! `cont.resume` status, `FIBER_PARKED` (3) — the suspend the guest didn't write — and keeps
//! running; re-resuming while blocked reports it again (the cooperative poll); after the
//! event fires, a resume continues the fiber past its park with the event's result delivered.
//! All single-vCPU: the whole point is that the domain never stops.

use std::time::Duration;

use temen_interp::{run_with_host, Host, OffloadOutcome, StreamRole, Value};

/// Futex park: the fiber `atomic.wait`s on a zero cell (forever). The root sees
/// `(FIBER_PARKED, 0)` twice (park, then poll), `atomic.notify`s the cell (waking exactly 1
/// waiter — the fiber), and the third resume runs the fiber to completion with the wait's
/// `WAIT_WOKEN` (0) status as its return. Composite: s1*100_000 + s2*10_000 + woken*1_000 +
/// s3*100 + value = 3*100_000 + 3*10_000 + 1*1_000 + 1*100 + 0 = 331_100.
const FUTEX_FIBER_PARK: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs1, vv1 = cont.resume v2 v3
  vs2, vv2 = cont.resume v2 v3
  vaddr = i64.const 16384
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  vs3, vv3 = cont.resume v2 v3
  vk1 = i64.const 100000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 10000
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vk3 = i64.const 1000
  vwe = i64.extend_i32_s vw
  vc = i64.mul vwe vk3
  vk4 = i64.const 100
  vs3e = i64.extend_i32_s vs3
  vd = i64.mul vs3e vk4
  vab = i64.add va vb
  vcd = i64.add vc vd
  vabcd = i64.add vab vcd
  vr = i64.add vabcd vv3
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_fiber_futex_park_parks_the_fiber_not_the_vcpu() {
    let m = temen_text::parse_module(FUTEX_FIBER_PARK).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(331_100)],
        "park(3) → poll(3) → notify wakes 1 → resume returns (1, WAIT_WOKEN=0)"
    );
}

/// The racing-fibers pattern on ONE vCPU: the fiber blocks reading empty blocking stdin
/// (fiber-parks; root sees `FIBER_PARKED`), the ROOT — still running, that's the point —
/// closes the handle, and the next resume completes the fiber's read with the revocation
/// errno (`-EBADF`), which it returns. Composite: s1*10_000 + s2*100 + (-value) =
/// 3*10_000 + 1*100 + 9 = 30_109.
const REVOKE_INTO_FIBER: &str = r#"
memory 16
func (i32) -> (i64) {
block 0 (v0: i32) {
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vh64 = i64.extend_i32_u v0
  vs1, vv1 = cont.resume vk vh64
  vc = cap.call 0 2 () -> (i64) v0 ()
  vs2, vv2 = cont.resume vk vh64
  vk1 = i64.const 10000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 100
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vz = i64.const 0
  vneg = i64.sub vz vv2
  vab = i64.add va vb
  vr = i64.add vab vneg
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  vbuf = i64.const 16392
  vcap = i64.const 4
  vr = cap.call 0 0 (i64, i64) -> (i64) vh (vbuf, vcap)
  return vr
  }
}
"#;

#[test]
fn the_root_revokes_a_read_its_own_fiber_is_parked_in() {
    let m = temen_text::parse_module(REVOKE_INTO_FIBER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_stream(StreamRole::In);
    host.set_stdin_blocking(true);
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(30_109)],
        "fiber parked (3), root closed, fiber returned -EBADF (status 1, value -9)"
    );
}

/// The park-vs-event race, closed: the cell already differs from `expected` when the fiber
/// waits, so the register-then-recheck wakes it immediately — the resumer sees one
/// `FIBER_PARKED` (the transient set-aside), and the next resume completes the wait with
/// `WAIT_NOT_EQUAL` (1). Composite: 3*10_000 + 1*100 + 1 = 30_101.
const NOT_EQUAL_INSTA_WAKE: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  vaddr = i64.const 16392
  vfive = i64.const 5
  i64.store vaddr vfive
  vf = ref.func 1
  vsp = i64.const 0
  vk = cont.new vf vsp
  vz = i64.const 0
  vs1, vv1 = cont.resume vk vz
  vs2, vv2 = cont.resume vk vz
  vk1 = i64.const 10000
  vs1e = i64.extend_i32_s vs1
  va = i64.mul vs1e vk1
  vk2 = i64.const 100
  vs2e = i64.extend_i32_s vs2
  vb = i64.mul vs2e vk2
  vab = i64.add va vb
  vr = i64.add vab vv2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16392
  vexp = i32.const 0
  vto = i64.const -1
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_prechanged_cell_wakes_the_parking_fiber_immediately() {
    let m = temen_text::parse_module(NOT_EQUAL_INSTA_WAKE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(30_101)],
        "one transient FIBER_PARKED, then WAIT_NOT_EQUAL delivered"
    );
}

/// The TIMEOUT wake (the jacl timed-wait regression, 2026-07-24): the fiber parks on a 10ms
/// **timed** wait on a never-notified cell; the root sees `FIBER_PARKED`, does its own 100ms
/// timed wait (the scheduler fires the fiber's deadline meanwhile), and the re-resume completes
/// the fiber with `WAIT_TIMED_OUT` (2) delivered as the wait's result. Composite: s1*10_000 +
/// s2*100 + v2 = 3*10_000 + 1*100 + 2 = 30_102. The cross-backend pinning (and the poll-fires-
/// the-deadline rule) lives in `temen/tests/fiber_timed_wait.rs`.
const TIMED_WAIT_TIMES_OUT: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs1, vv1 = cont.resume v2 v3
  va = i64.const 16392
  ve = i32.const 0
  vt = i64.const 100000000
  vw = i32.atomic.wait va ve vt
  vs2, vv2 = cont.resume v2 v3
  vk1 = i64.const 10000
  vs1e = i64.extend_i32_s vs1
  vp1 = i64.mul vs1e vk1
  vk2 = i64.const 100
  vs2e = i64.extend_i32_s vs2
  vp2 = i64.mul vs2e vk2
  vp = i64.add vp1 vp2
  vr = i64.add vp vv2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_timed_fiber_wait_fires_its_deadline_and_completes() {
    let m = temen_text::parse_module(TIMED_WAIT_TIMES_OUT).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let mut fuel = u64::MAX;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no hang");
    assert_eq!(
        r,
        vec![Value::I64(30_102)],
        "park (3), root sleeps across the deadline, resume delivers WAIT_TIMED_OUT (2)"
    );
}

/// The **poll fires a passed deadline**: no root sleep at all — a busy `cont.resume` poll loop
/// over the parked fiber (jacl's harness shape). The loop's polls are the only scheduler
/// touchpoints, so this pins that a poll of a still-blocked fiber whose deadline has passed
/// completes the wait right there (rather than starving until a worker idles — the regression's
/// tree-walker leg). Result: the fiber's `WAIT_TIMED_OUT` status (2).
const POLL_LOOP_TIMES_OUT: &str = r#"
memory 16
func () -> (i64) {
block 0 () {
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  br 1(v2)
}
block 1 (vk: i64) {
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vv) 1(vk)
}
block 2 (vr: i64) {
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const 10000000
  vst = i32.atomic.wait vaddr vexp vto
  vst64 = i64.extend_i32_s vst
  return vst64
  }
}
"#;

#[test]
fn a_resume_poll_loop_observes_the_passed_deadline() {
    let m = temen_text::parse_module(POLL_LOOP_TIMES_OUT).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    // Bounded fuel: the pre-fix failure mode is the loop spinning forever (the deadline never
    // fired without an idle worker), which this converts into a loud OutOfFuel.
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[], &mut fuel, &mut host).expect("no trap, no spin-forever");
    assert_eq!(
        r,
        vec![Value::I64(2)],
        "the poll itself fires the due timeout: WAIT_TIMED_OUT delivered"
    );
}

// ----- F1 (FIBER_PARK.md) — a punted host call parks the FIBER, not the vCPU ------------------

/// The deterministic transform the `Blocking` cap's jobs apply (mirrors `AsyncState::mix`).
fn mix(arg: i64) -> i64 {
    arg.wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

/// A punted offloadable dispatch inside a fiber unwinds `FIBER_PARKED` to the resumer (the
/// slice-5a contract, extended to `Pending`), the pool completion wakes the fiber, and the next
/// resume delivers the scalar. The fiber calls `cap.call 13 0` (HOST_PROC) with arg 5 on a
/// handler that always punts `arg + 100`; the root records the first resume status (must be the
/// park, 3) then polls to completion. Composite: s1*10_000 + value = 3*10_000 + 105 = 30_105.
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
  vr = cap.call 13 0 (i64) -> (i64) vh (vfive)
  return vr
  }
}
"#;

#[test]
fn a_punted_host_call_parks_the_fiber_not_the_vcpu() {
    let m = temen_text::parse_module(PUNT_IN_FIBER).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        OffloadOutcome::Offload(Box::new(move || a + 100))
    }));
    let comps = host.completions();
    // Bounded fuel: a lost completion wake leaves the poll loop spinning forever — convert
    // that into a loud OutOfFuel instead of a hang.
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(30_105)],
        "first resume parked (3), completion woke the fiber, resume delivered 105"
    );
    assert_eq!(comps.minted(), 1, "one completion per punt");
    assert_eq!(comps.outstanding(), 0, "the completion was delivered");
}

/// **The single-vCPU overlap the IoRing retirement recorded as its known loss** — pinned by
/// rendezvous, never timing. ONE vCPU, two fibers, each punting one `cap.call 10 0` on a
/// rendezvous-2 `Blocking` handle: fiber A's pool job waits at the width-2 barrier until fiber
/// B's job arrives, which can only happen if A's park freed the vCPU (the pool job is dispatched
/// at punt time, before the park). Under the pre-F1 degenerate wait, resuming A deadlocks the
/// run. Both first statuses are deterministically the park (even an instantly-raced completion
/// unwinds the transient `FIBER_PARKED` once). Composite: mix(0) + mix(1) + (sA*10 + sB) =
/// mix-sum + 33, wrapping.
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
  vr = cap.call 10 0 (i64) -> (i64) vh (va)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 1
  vr = cap.call 10 0 (i64) -> (i64) vh (va)
  return vr
  }
}
"#;

#[test]
fn two_fibers_on_one_vcpu_overlap_their_punts() {
    let m = temen_text::parse_module(TWO_FIBER_OVERLAP).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_blocking(Duration::ZERO, Some(2));
    let state = host.blocking_state(h).expect("blocking state");
    let comps = host.completions();
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    let want = mix(0).wrapping_add(mix(1)).wrapping_add(33);
    assert_eq!(
        r,
        vec![Value::I64(want)],
        "both fibers parked (3,3) and both punts completed on one vCPU"
    );
    assert_eq!(
        state.max_active(),
        2,
        "both punted jobs co-resided on the pool — the single-vCPU overlap"
    );
    assert_eq!(comps.minted(), 2, "one completion per punt");
    assert_eq!(comps.outstanding(), 0, "all completions delivered");
}

/// **Submission-ordered delivery holds for fiber parks (the §18 pin).** Fiber 1 punts a job
/// gated on a host-side latch; fiber 2 punts a job that completes immediately. The drain stops
/// at the first not-yet-arrived id, so fiber 2's ready result must NOT be delivered while fiber
/// 1's earlier id is outstanding: 50 polls of fiber 2 all observe `FIBER_PARKED`. The root then
/// releases the latch (op 1, inline `Done`) and polls both to completion. Composite:
/// parked_polls*1_000_000 + v1*1_000 + v2 = 50*1_000_000 + 111_000 + 222 = 50_111_222.
const ORDERED_FIBER_DELIVERY: &str = r#"
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
  vcnt0 = i64.const 0
  vi0 = i64.const 0
  br 1(vk1, vk2, vh64, vcnt0, vi0)
}
block 1 (ka: i64, kb: i64, h1: i64, cnt: i64, vi: i64) {
  vlim = i64.const 50
  vlt = i64.lt_s vi vlim
  br_if vlt 2(ka, kb, h1, cnt, vi) 3(ka, kb, h1, cnt)
}
block 2 (ka2: i64, kb2: i64, h2: i64, cnt2: i64, vi2: i64) {
  vs, vv = cont.resume kb2 h2
  vthree = i32.const 3
  veq = i32.eq vs vthree
  veq64 = i64.extend_i32_u veq
  vcnt3 = i64.add cnt2 veq64
  vone = i64.const 1
  vi3 = i64.add vi2 vone
  br 1(ka2, kb2, h2, vcnt3, vi3)
}
block 3 (ka4: i64, kb4: i64, h4: i64, cnt4: i64) {
  vh = i32.wrap_i64 h4
  vz4 = i64.const 0
  vrel = cap.call 13 1 (i64) -> (i64) vh (vz4)
  br 4(ka4, kb4, h4, cnt4)
}
block 4 (ka5: i64, kb5: i64, h5: i64, cnt5: i64) {
  vs5, vv5 = cont.resume ka5 h5
  vone5 = i32.const 1
  vdone5 = i32.eq vs5 vone5
  br_if vdone5 5(kb5, h5, cnt5, vv5) 4(ka5, kb5, h5, cnt5)
}
block 5 (kb6: i64, h6: i64, cnt6: i64, v1: i64) {
  vs6, vv6 = cont.resume kb6 h6
  vone6 = i32.const 1
  vdone6 = i32.eq vs6 vone6
  br_if vdone6 6(cnt6, v1, vv6) 5(kb6, h6, cnt6, v1)
}
block 6 (cnt7: i64, v1b: i64, v2: i64) {
  vm = i64.const 1000000
  vp = i64.mul cnt7 vm
  vk = i64.const 1000
  vq = i64.mul v1b vk
  vpq = i64.add vp vq
  vr = i64.add vpq v2
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 0
  vr = cap.call 13 0 (i64) -> (i64) vh (va)
  return vr
  }
}
func (i64, i64) -> (i64) {
block 0 (vsp: i64, varg: i64) {
  vh = i32.wrap_i64 varg
  va = i64.const 1
  vr = cap.call 13 0 (i64) -> (i64) vh (va)
  return vr
  }
}
"#;

/// Resume the punting fiber once and return `status*10_000 + value` immediately — the shared
/// kernel for the teardown and durable pins below. Non-durable: the fiber parks → 3*10_000 + 0.
/// Durable: the F1 predicate excludes durable callers, so the punt takes the slice-1 degenerate
/// blocking wait INSIDE the resume and the fiber completes → 1*10_000 + 105.
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
  vr = cap.call 13 0 (i64) -> (i64) vh (vfive)
  return vr
  }
}
"#;

/// **Teardown abandons a cap-parked fiber** (the batch rule, invariant 6): the root returns with
/// the fiber still parked on its punt. The run must end promptly — the pool job (a bounded
/// 100 ms sleep) is drained by the run-end pool quiesce, its result lands unclaimed, and the
/// parked fiber dies with the domain instead of holding the run open.
#[test]
fn root_return_abandons_a_cap_parked_fiber() {
    let m = temen_text::parse_module(RESUME_ONCE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        OffloadOutcome::Offload(Box::new(move || {
            std::thread::sleep(Duration::from_millis(100));
            a + 100
        }))
    }));
    let comps = host.completions();
    let t0 = std::time::Instant::now();
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    let took = t0.elapsed();
    assert_eq!(
        r,
        vec![Value::I64(30_000)],
        "the resume observed the park (3, 0) and the root returned over it"
    );
    assert!(
        took < Duration::from_secs(10),
        "teardown must not wait out the abandoned fiber beyond the pool quiesce (took {took:?})"
    );
    assert_eq!(comps.minted(), 1, "the punt minted before the abandon");
}

/// **Durable callers keep the degenerate wait** (the F1 predicate's `!durable` conjunct):
/// `freeze_drive` fails closed on any unwoken cap park, so a durable run must never create one —
/// the punt blocks inside the resume and the fiber RETURNS on the first poll (status 1, not 3).
#[test]
fn a_durable_caller_never_fiber_parks_on_a_punt() {
    let m = temen_text::parse_module(RESUME_ONCE).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let mut host = Host::new();
    host.set_durable(true);
    let h = host.grant_host_proc_offloadable(Box::new(|_op, args| {
        let a = *args.first().unwrap_or(&0);
        OffloadOutcome::Offload(Box::new(move || a + 100))
    }));
    let comps = host.completions();
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(10_105)],
        "durable: the punt completed inside the resume (FIBER_RETURNED, 105) — no ParkedOn state"
    );
    assert_eq!(
        comps.minted(),
        1,
        "the punt still minted (degenerate wait, not inline)"
    );
    assert_eq!(
        comps.outstanding(),
        0,
        "the completion was consumed by the wait"
    );
}

#[test]
fn ordered_delivery_holds_a_ready_later_completion_for_an_earlier_park() {
    let m = temen_text::parse_module(ORDERED_FIBER_DELIVERY).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    let gate = std::sync::Arc::new((std::sync::Mutex::new(false), std::sync::Condvar::new()));
    let gate_job = std::sync::Arc::clone(&gate);
    let gate_rel = std::sync::Arc::clone(&gate);
    let mut host = Host::new();
    let h = host.grant_host_proc_offloadable(Box::new(move |op, args| {
        if op == 1 {
            // The release (inline `Done` — never punts): open the latch.
            let (m, cv) = &*gate_rel;
            *m.lock().unwrap() = true;
            cv.notify_all();
            return OffloadOutcome::Done(Ok(vec![0]));
        }
        let a = *args.first().unwrap_or(&0);
        if a == 0 {
            let g = std::sync::Arc::clone(&gate_job);
            OffloadOutcome::Offload(Box::new(move || {
                let (m, cv) = &*g;
                let mut open = m.lock().unwrap();
                while !*open {
                    open = cv.wait(open).unwrap();
                }
                111
            }))
        } else {
            OffloadOutcome::Offload(Box::new(move || 222))
        }
    }));
    let comps = host.completions();
    let mut fuel = 2_000_000_000;
    let r = run_with_host(&m, 0, &[Value::I32(h)], &mut fuel, &mut host).expect("no trap");
    assert_eq!(
        r,
        vec![Value::I64(50_111_222)],
        "fiber 2 stayed parked (50/50 polls) while its ready result waited on fiber 1's id"
    );
    assert_eq!(
        comps.minted(),
        2,
        "the release is inline; only the punts minted"
    );
    assert_eq!(comps.outstanding(), 0, "all completions delivered");
}
