//! §3.6 slice 5a — **timed fiber waits**, pinned across TreeWalk / Bytecode / Cranelift JIT
//! (DESIGN.md §12 futexes, §3.6 5a fiber-level park routing; INVARIANTS.md invariant 9: the
//! tree-walk interpreter is the oracle — the fast backends run these identically or decline,
//! never diverge). Consumer: the jacl timed-wait regression (2026-07-24) — jacl runs every job
//! as a stackful fiber, so `sleep` is an `i32.atomic.wait` with a finite timeout inside a
//! `cont.new`/`cont.resume` fiber, driven by a cooperative resume-poll loop.
//!
//! The contract these kernels pin (extending `temen-interp/tests/fiber_parks.rs`, whose
//! notify-/revocation-/insta-wake trio never covered the TIMEOUT wake):
//!
//! - a `memory.wait` inside a fiber parks the FIBER, not the vCPU — the resumer observes
//!   `cont.resume` status `FIBER_PARKED` (3) and keeps running;
//! - re-resuming while still blocked (and before the deadline) reports `FIBER_PARKED` again —
//!   the cooperative poll never blocks the vCPU and never runs guest code in the fiber;
//! - a timed wait whose deadline has passed completes with `WAIT_TIMED_OUT` (2), **fired at
//!   the latest at the next `cont.resume` poll of the parked fiber** — the deadline is checked
//!   at the poll, so a busy resume-poll loop terminates without depending on scheduler idle
//!   time. Engines may also fire it earlier when their scheduler is idle (the tree-walker's
//!   worker-loop timers; the bytecode cooperative driver's logical clock).
//!
//! The deterministic explorer (`run_scheduled`) deliberately keeps **whole-vCPU** parks for
//! fiber waits (the `fiber_park!` gate on `SchedRef::Real`, so interleavings stay explorable);
//! under that subset a fiber's wait resolves inside the first resume. The poll-LOOP kernel is
//! agnostic to which semantics drives it (it stops on status `FIBER_RETURNED` whenever that
//! arrives), so it is additionally pinned on the explorer; the poll-shaped kernels
//! (`IMMEDIATE_REPOLL` / `ONE_RESUME`) are Real-scheduler contracts and are not run there.

use temen_run::{instantiate, Backend, Outcome, RunConfig, Value};

/// Run `src` on `backend` and return the single i64 result.
fn run_on(src: &str, backend: Backend) -> i64 {
    let module = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&module).expect("verify");
    let instance = instantiate(module).expect("instantiate");
    let run = instance
        .run(backend, &RunConfig::default())
        .unwrap_or_else(|e| panic!("{backend:?}: {e}"));
    match run.outcome {
        Outcome::Returned(vals) => match vals.as_slice() {
            [Value::I64(x)] => *x,
            other => panic!("{backend:?}: expected one i64, got {other:?}"),
        },
        other => panic!("{backend:?}: did not return: {other:?}"),
    }
}

/// Assert all three backends agree on `want` for `src`.
fn pin_all(src: &str, want: i64) {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        assert_eq!(run_on(src, backend), want, "{backend:?}");
    }
}

/// The fiber body every kernel shares: a timed `i32.atomic.wait` on a never-notified zero cell
/// (`mem[16384]`, above the #1094 NULL guard, timeout `{TO}` ns), returning `wait_status * 1000 + 42` — so the result pins
/// both that the fiber completed (42) and which `WAIT_*` status its park delivered.
fn fiber_body(timeout_ns: u64) -> String {
    format!(
        r#"func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vaddr = i64.const 16384
  vexp = i32.const 0
  vto = i64.const {timeout_ns}
  vst = i32.atomic.wait vaddr vexp vto
  vk = i64.const 1000
  vst64 = i64.extend_i32_s vst
  va = i64.mul vst64 vk
  v42 = i64.const 42
  vr = i64.add va v42
  return vr
  }}
}}
"#
    )
}

/// jacl's `sleep` shape with a sleeping root: the fiber parks on a 10ms timed wait (root sees
/// `FIBER_PARKED`), the root does its own 100ms timed wait, then re-resumes — by then the
/// fiber's deadline has passed, so the resume completes it with `WAIT_TIMED_OUT` delivered.
/// Composite: s1*100_000 + s2*10_000 + v2 = 3*100_000 + 1*10_000 + (2*1000 + 42) = 312_042.
fn sleep_between() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
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
  vk1 = i64.const 100000
  vs1e = i64.extend_i32_s vs1
  vp1 = i64.mul vs1e vk1
  vk2 = i64.const 10000
  vs2e = i64.extend_i32_s vs2
  vp2 = i64.mul vs2e vk2
  vp = i64.add vp1 vp2
  vr = i64.add vp vv2
  return vr
  }}
}}
{}"#,
        fiber_body(10_000_000)
    )
}

/// The cooperative poll before the deadline: the root re-resumes IMMEDIATELY (no sleep), well
/// inside the fiber's 2s timeout — the poll must report `FIBER_PARKED` without blocking the
/// vCPU and without completing the wait. The root then returns, abandoning the parked fiber
/// (invariant 6: root completion ends the activation; nothing waits for parked daemons) — so
/// this also pins that an abandoned timed fiber wait never wedges teardown on any backend.
/// Composite: s1*10_000 + s2*100 + v2 = 3*10_000 + 3*100 + 0 = 30_300.
fn immediate_repoll() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs1, vv1 = cont.resume v2 v3
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
  }}
}}
{}"#,
        // 2s: wide margin so a loaded CI machine can't reach the deadline between the park and
        // the poll (the run never waits it out — the root returns with the fiber parked).
        fiber_body(2_000_000_000)
    )
}

/// jacl's exact harness shape: a cooperative resume-poll LOOP, no root sleep — resume until
/// status `FIBER_RETURNED` (1), then return the fiber's value. Terminates on every backend
/// because the poll itself fires the passed deadline (and on the explorer because the wait
/// resolves inside the first resume). Expect 2*1000 + 42 = 2_042.
fn poll_loop() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  br 1(v2)
}}
block 1 (vk: i64) {{
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 2(vv) 1(vk)
}}
block 2 (vr: i64) {{
  return vr
  }}
}}
{}"#,
        fiber_body(10_000_000)
    )
}

/// The notify wake on the fast backends — `fiber_parks.rs`'s futex kernel through the temen-run
/// path, so the whole 5a park contract (not just the timeout) is pinned on Bytecode and the
/// Cranelift JIT too: park (3), poll (3), `atomic.notify` wakes exactly 1 (the fiber), resume
/// completes with `WAIT_WOKEN` delivered. Composite: s1*100_000 + s2*10_000 + woken*1_000 +
/// s3*100 + v3 = 3*100_000 + 3*10_000 + 1*1_000 + 1*100 + (0*1000+42) = 331_142.
fn notify_wake() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
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
  }}
}}
{}"#,
        // 2s: long enough that the notify (microseconds after the park) always beats it.
        fiber_body(2_000_000_000)
    )
}

/// The insta-wake on the fast backends — `fiber_parks.rs`'s not-equal kernel through the
/// temen-run path: the cell (`mem[0]`) already differs when the fiber waits, so the
/// register-then-recheck wakes it immediately — one transient `FIBER_PARKED`, then the next
/// resume delivers `WAIT_NOT_EQUAL`. Composite: s1*10_000 + s2*100 + v2 = 3*10_000 + 1*100 +
/// (1*1000 + 42) = 31_142. (`mem[16384]`, above the #1094 NULL guard.)
fn insta_wake() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
  vaddr = i64.const 16384
  vfive = i32.const 5
  i32.store vaddr vfive
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  v3 = i64.const 0
  vs1, vv1 = cont.resume v2 v3
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
  }}
}}
{}"#,
        fiber_body(2_000_000_000)
    )
}

/// A timed-out fiber wait completes with 42 on every backend — jacl's acceptance criterion,
/// with the root sleeping across the fiber's deadline.
#[test]
fn a_timed_fiber_wait_woken_by_timeout_completes_everywhere() {
    pin_all(&sleep_between(), 312_042);
}

/// An immediate re-poll before the deadline reports `FIBER_PARKED` — the cooperative poll
/// never blocks the vCPU, on any backend.
#[test]
fn b_a_poll_before_the_deadline_reports_fiber_parked_everywhere() {
    pin_all(&immediate_repoll(), 30_300);
}

/// A cooperative resume-poll loop terminates once the deadline passes — the timeout fires at
/// the poll, on every backend (the jacl harness shape).
#[test]
fn c_a_cooperative_poll_loop_terminates_at_the_deadline_everywhere() {
    pin_all(&poll_loop(), 2_042);
}

/// The notify wake of a fiber park, on the fast backends too (previously the Cranelift JIT
/// blocked the whole vCPU inside the fiber's wait — a root-side notify could never run).
#[test]
fn d_a_notify_wakes_a_parked_fiber_everywhere() {
    pin_all(&notify_wake(), 331_142);
}

/// The insta-wake (`WAIT_NOT_EQUAL`) register-then-recheck race close, on the fast backends
/// too — one transient `FIBER_PARKED`, then the status delivered.
#[test]
fn e_a_prechanged_cell_insta_wakes_the_parking_fiber_everywhere() {
    pin_all(&insta_wake(), 31_142);
}

/// Explorer coverage: `run_scheduled` keeps whole-vCPU parks for fiber waits (its documented
/// subset — see the module doc), and the semantics-agnostic poll loop must still finish with
/// the fiber's 42 on every seed, via the explorer's logical clock.
#[test]
fn f_the_explorer_finishes_the_poll_loop_on_its_logical_clock() {
    let src = poll_loop();
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    for seed in 0..20 {
        assert_eq!(
            temen_interp::run_scheduled(&m, 0, &[], 50_000_000, seed),
            Ok(vec![temen_interp::Value::I64(2_042)]),
            "explorer seed {seed}"
        );
    }
}
