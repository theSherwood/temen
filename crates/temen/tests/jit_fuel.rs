//! Safepoint-anchored **counted fuel on the JIT** (INTERP_PERF.md "Fuel unification", step 3). Unlike
//! the §5 kill-path (`jit_killpath.rs` — a host-written cell polled asynchronously, so interp and JIT
//! stop at *different* points and only the *outcome* matches), counted fuel is a **deterministic guest
//! budget**: the lowering decrements a host-owned `u64` at every function entry, taken back-edge and
//! `cont.resume`, and traps `OutOfFuel` when it would underflow. So a runaway traps at a fixed point
//! with no watchdog, and a finite run charges a countable amount — the same unit the tree-walker and
//! bytecode engines now charge, so fuel becomes a checked cross-engine quantity rather than an
//! excluded difference.
//!
//! Exact interp/JIT fuel *count* parity now holds: the interpreters were reconciled to charge the
//! top-level entry function too (`super::drive_arc` / bytecode `drive`), matching the JIT's entry-
//! prologue charge, so all three engines consume the identical amount on the same run and exhaust at
//! the identical safepoint. These tests assert strict equality (`== interp`), and the differential
//! harnesses assert `OutOfFuel` parity rather than excluding it (`bytecode_diff`, `jit_fuzz`).

use temen_interp::{bytecode, run, Trap, Value};
use temen_jit::{compile_and_run, compile_and_run_with_host_fuel, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// A non-terminating **intra-function loop** (block 1 branches to itself forever) — caught by the
/// per-back-edge fuel charge.
const INFINITE_LOOP: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  br 1(v0)
}
block 1 (v1: i64) {
  v2 = i64.const 1
  v3 = i64.add v1 v2
  br 1(v3)
  }
}
";

/// A non-terminating **tail-recursion** (function 0 tail-calls itself forever) — runs in O(1) native
/// stack, so only the *function-entry* fuel charge (the callee prologue) can stop it.
const INFINITE_TAIL_RECURSION: &str = "\
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 1
  v2 = i64.add v0 v1
  return_call 0(v2)
  }
}
";

/// A **finite** countdown N → 0: block 1 loops back to itself while the counter is non-zero (a taken
/// back-edge each iteration), then exits forward to block 2 (no charge). The back-edge is taken `N-1`
/// times; both engines also charge the single top-level entry, so each charges `N` on the same run.
const FINITE_COUNTDOWN: &str = "\
func (i32) -> (i32) {
block 0 (v0: i32) {
  br 1(v0)
}
block 1 (v1: i32) {
  v2 = i32.const -1
  v3 = i32.add v1 v2
  br_if v3 1(v3) 2(v3)
}
block 2 (v4: i32) {
  return v4
  }
}
";

/// Run `src` on the JIT with a counted-fuel budget of `budget`; returns the outcome and the *remaining*
/// fuel the guest wrote back into the cell.
fn jit_fuel(src: &str, args: &[i64], budget: u64) -> (JitOutcome, u64) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    // The guest decrements this cell in place through the baked address; after the run returns we read
    // the remainder back (no concurrent access — a single guest thread owns the budget).
    let mut cell: u64 = budget;
    let outcome = compile_and_run_with_host_fuel(
        &m,
        0,
        args,
        temen_run::cap_thunk,
        core::ptr::null_mut(),
        &mut cell as *mut u64,
    )
    .expect("jit compiles");
    (outcome, cell)
}

#[test]
fn jit_fuel_stops_infinite_loop() {
    // Deterministic: a small budget bounds the loop; the per-back-edge charge trips at exactly zero.
    let (outcome, remaining) = jit_fuel(INFINITE_LOOP, &[0i64], 1_000);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "counted fuel must stop the runaway loop"
    );
    assert_eq!(remaining, 0, "an exhausted budget is drained to zero");
}

#[test]
fn jit_fuel_stops_infinite_tail_recursion() {
    // The function-entry charge (the callee prologue), not the back-edge one, is what catches this —
    // tail calls never grow the stack, so without it the guest would spin forever.
    let (outcome, remaining) = jit_fuel(INFINITE_TAIL_RECURSION, &[0i64], 1_000);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "counted fuel must stop the runaway tail recursion"
    );
    assert_eq!(remaining, 0, "an exhausted budget is drained to zero");
}

#[test]
fn jit_fuel_finite_run_completes_and_charges() {
    const N: i32 = 1000;
    const BUDGET: u64 = 10_000_000;

    // Interpreter reference: run with the same generous budget and read how much it consumed.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let mut interp_fuel = BUDGET;
    let interp = run(&m, 0, &[Value::I32(N)], &mut interp_fuel).expect("interp ok");
    assert_eq!(interp, vec![Value::I32(0)], "countdown returns 0");
    let interp_consumed = BUDGET - interp_fuel;
    assert!(interp_consumed > 0, "the loop charges some fuel");

    // JIT: same program, same budget — must return the same result and charge the *exact* same amount.
    // Both engines now charge the top-level entry (fuel unification), so this is strict equality.
    let (outcome, remaining) = jit_fuel(FINITE_COUNTDOWN, &[N as i64], BUDGET);
    assert_eq!(
        outcome,
        JitOutcome::Returned(vec![0]),
        "an armed-but-sufficient finite run completes normally"
    );
    let jit_consumed = BUDGET - remaining;
    assert_eq!(
        jit_consumed, interp_consumed,
        "JIT fuel must exactly match interp after top-level-entry reconciliation"
    );
}

#[test]
fn jit_fuel_exhausts_one_short() {
    // A budget one below the exact need must trap `OutOfFuel` rather than complete — proving the charge
    // is exact at the boundary, not approximate. After fuel unification both engines charge the
    // top-level entry, so they consume the *same* amount and exhaust at the *same* point: budget
    // `interp_consumed` completes on both; `interp_consumed - 1` traps on both.
    const N: i32 = 1000;
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let mut interp_fuel = 10_000_000u64;
    run(&m, 0, &[Value::I32(N)], &mut interp_fuel).expect("interp ok");
    let interp_consumed = 10_000_000u64 - interp_fuel;
    let short = interp_consumed - 1;

    // Interp one short: traps.
    let mut interp_short = short;
    assert_eq!(
        run(&m, 0, &[Value::I32(N)], &mut interp_short),
        Err(Trap::OutOfFuel),
        "interp one short of its exact need must trap OutOfFuel"
    );
    // JIT one short: traps at the same boundary (exact interp/JIT fuel parity).
    let (outcome, remaining) = jit_fuel(FINITE_COUNTDOWN, &[N as i64], short);
    assert_eq!(
        outcome,
        JitOutcome::Trapped(TrapKind::OutOfFuel),
        "JIT one short of the exact need must trap OutOfFuel, not complete"
    );
    assert_eq!(remaining, 0);
}

#[test]
fn jit_unarmed_path_is_unchanged() {
    // Arming is strictly opt-in: the ordinary (fuel-un-armed) entry runs the same finite program to
    // completion, byte-identical to before the feature existed.
    let m = parse_module(FINITE_COUNTDOWN).expect("parse");
    verify_module(&m).expect("verify");
    let jit = compile_and_run(&m, 0, &[1000i64]).expect("jit");
    assert_eq!(jit, JitOutcome::Returned(vec![0]));
}

// #1642 — `cont.resume` is a fuel safepoint on every engine, the JIT included.

/// How a run ended, normalized across the three engines' result types.
#[derive(Debug, PartialEq)]
enum Ended {
    Returned(i64),
    OutOfFuel,
}

/// Run `src`'s function 0 (no arguments, one `i64` result) on the tree-walker, the bytecode engine and
/// the JIT, each against its own `budget`-fuel cell. Returns how each ended and the fuel it charged,
/// oracle first. Any other trap is a test failure, not a result.
fn on_all_three(src: &str, budget: u64) -> [(Ended, u64); 3] {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let mut tw_left = budget;
    let tw = match run(&m, 0, &[], &mut tw_left) {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Ended::Returned(*x),
            other => panic!("TreeWalk returned {other:?}"),
        },
        Err(Trap::OutOfFuel) => Ended::OutOfFuel,
        Err(t) => panic!("TreeWalk trapped {t:?}"),
    };
    let mut bc_left = budget;
    let bc = match bytecode::compile_and_run(&m, 0, &[] as &[Value], &mut bc_left)
        .expect("the kernel is in the bytecode engine's subset")
    {
        Ok(v) => match v.as_slice() {
            [Value::I64(x)] => Ended::Returned(*x),
            other => panic!("Bytecode returned {other:?}"),
        },
        Err(Trap::OutOfFuel) => Ended::OutOfFuel,
        Err(t) => panic!("Bytecode trapped {t:?}"),
    };
    let (jo, jit_left) = jit_fuel(src, &[], budget);
    let jit = match jo {
        JitOutcome::Returned(v) => match v.as_slice() {
            [x] => Ended::Returned(*x),
            other => panic!("JIT returned {other:?}"),
        },
        JitOutcome::Trapped(TrapKind::OutOfFuel) => Ended::OutOfFuel,
        other => panic!("JIT ended {other:?}"),
    };
    [
        (tw, budget - tw_left),
        (bc, budget - bc_left),
        (jit, budget - jit_left),
    ]
}

/// Assert the bytecode engine and the JIT end `src` exactly as the oracle does **and charge exactly
/// what it charges** (INVARIANTS #9: fuel is a checked cross-engine quantity). Returns the oracle's row.
fn assert_three_way(src: &str, budget: u64) -> (Ended, u64) {
    let [tw, bc, jit] = on_all_three(src, budget);
    assert_eq!(
        bc, tw,
        "Bytecode vs the oracle: (how it ended, fuel charged)"
    );
    assert_eq!(jit, tw, "JIT vs the oracle: (how it ended, fuel charged)");
    tw
}

/// `f` fibers, each `suspend`ing `s` times (yielding 0, 1, …, s−1) and then returning 7, drained one
/// at a time by a plain `cont.resume` loop that sums everything they yield and return — so the result
/// is `f · (s(s−1)/2 + 7)`.
fn fibers_suspending(f: i64, s: i64) -> String {
    let body = if s == 0 {
        "func (i64, i64) -> (i64) {\nblock 0 (vsp: i64, varg: i64) {\n  v7 = i64.const 7\n  return v7\n  }\n}\n"
            .to_string()
    } else {
        format!(
            r#"func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vc0 = i64.const 0
  br 1(vc0)
}}
block 1 (vc: i64) {{
  vx = suspend vc
  vinc = i64.const 1
  vc2 = i64.add vc vinc
  vlim = i64.const {s}
  vmore = i64.ne vc2 vlim
  br_if vmore 1(vc2) 2(vc2)
}}
block 2 (vd: i64) {{
  v7 = i64.const 7
  return v7
  }}
}}
"#
        )
    };
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  vf0 = i64.const 0
  vacc0 = i64.const 0
  br 1(vf0, vacc0)
}}
block 1 (vf: i64, vacc: i64) {{
  v0 = ref.func 1
  vz = i64.const 0
  vk = cont.new v0 vz
  br 2(vk, vf, vacc)
}}
block 2 (vk2: i64, vf2: i64, vacc2: i64) {{
  vz2 = i64.const 0
  vs, vv = cont.resume vk2 vz2
  vone = i32.const 1
  vdone = i32.eq vs vone
  vacc3 = i64.add vacc2 vv
  br_if vdone 3(vf2, vacc3) 2(vk2, vf2, vacc3)
}}
block 3 (vf3: i64, vacc4: i64) {{
  vinc = i64.const 1
  vf4 = i64.add vf3 vinc
  vlim = i64.const {f}
  vmore = i64.ne vf4 vlim
  br_if vmore 1(vf4, vacc4) 4(vacc4)
}}
block 4 (vr: i64) {{
  return vr
  }}
}}
{body}"#
    )
}

/// **Every `cont.resume` is charged, on every engine.** The interpreters charge one fuel per resume
/// (INVARIANTS #9), so `F` fibers of `S` suspends each cost `F·(3S+1)`: `F·(S+1)` resumes, `F·S`
/// resume-loop back-edges, `F·(S−1)` back-edges inside the fibers, `F−1` outer back-edges and the
/// top-level entry. The JIT charged only each fiber's entry prologue — `F·(2S+1)`, every re-resume
/// unmetered. The two models agree exactly when each fiber is resumed once (`S = 0`: `2F` either
/// way), which is why no existing test saw it. That row also pins the start refund: charging the
/// resume without refunding the fiber's entry prologue would make a start cost two, `3F`.
#[test]
fn jit_fuel_charges_every_cont_resume_like_the_interpreters() {
    for (f, s) in [(10, 5), (1, 100), (10, 0)] {
        let (ended, fuel) = assert_three_way(&fibers_suspending(f, s), 10_000_000);
        assert_eq!(
            ended,
            Ended::Returned(f * (s * (s - 1) / 2 + 7)),
            "F={f} S={s}"
        );
        let unit = if s == 0 { 2 * f } else { f * (3 * s + 1) };
        assert_eq!(fuel, unit as u64, "F={f} S={s}: one fuel per resume");
    }
}

/// A plain-`cont.resume` poll loop over a fiber parked in an **infinite** `atomic.wait`, counting the
/// `FIBER_PARKED` polls. With `notify_after = Some(k)` the poller itself notifies the fiber's word
/// after `k` polls; with `None` it never does. Returns `polls · 1_000_000 + (wait status · 1000 + 42)`.
fn busy_poll(notify_after: Option<i64>) -> String {
    let k = notify_after.unwrap_or(-1); // `-1`: a poll count never reached
    format!(
        r#"memory 16
func () -> (i64) {{
block 0 () {{
  v0 = ref.func 1
  v1 = i64.const 0
  v2 = cont.new v0 v1
  vi0 = i64.const 0
  br 1(v2, vi0)
}}
block 1 (vk: i64, vi: i64) {{
  vz = i64.const 0
  vs, vv = cont.resume vk vz
  vone = i32.const 1
  vdone = i32.eq vs vone
  br_if vdone 4(vv, vi) 2(vk, vi)
}}
block 2 (vk2: i64, vi2: i64) {{
  vinc = i64.const 1
  vi3 = i64.add vi2 vinc
  vlim = i64.const {k}
  vhit = i64.eq vi3 vlim
  br_if vhit 3(vk2, vi3) 1(vk2, vi3)
}}
block 3 (vk3: i64, vi4: i64) {{
  vaddr = i64.const 16384
  vcnt = i32.const 1
  vw = atomic.notify vaddr vcnt
  br 1(vk3, vi4)
}}
block 4 (vr: i64, vn: i64) {{
  vm = i64.const 1000000
  vnm = i64.mul vn vm
  vout = i64.add vnm vr
  return vout
  }}
}}
func (i64, i64) -> (i64) {{
block 0 (vsp: i64, varg: i64) {{
  vaddr = i64.const 16384
  vexp = i32.const 0
  vinf = i64.const -1
  vst = i32.atomic.wait vaddr vexp vinf
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

/// **A busy poll is a guest loop, not a deadlock (#1642) — and it is metered per poll.** The fiber
/// waits forever and nothing but the poller can wake it; the poller does, after 1000 polls, so every
/// engine completes with exactly 1000 parked polls seen and `WAIT_WOKEN` delivered. That is why no
/// engine may ask the deadlock predicate from a resume poll, as #1642 proposed: the poller is live
/// guest code, and whether it will ever `notify` is undecidable.
///
/// It charges identically too — 2002 on every engine, where the JIT charged 1002 before `cont.resume`
/// became one of its safepoints — and one fuel short, every engine runs out at the same safepoint.
#[test]
fn jit_fuel_meters_a_busy_resume_poll_that_wakes_its_own_fiber() {
    let src = busy_poll(Some(1000));
    let (ended, need) = assert_three_way(&src, 10_000_000);
    assert_eq!(
        ended,
        Ended::Returned(1000 * 1_000_000 + 42),
        "1000 polls, then WAIT_WOKEN"
    );
    assert_eq!(need, 2002, "1 entry + 1001 resumes + 1000 back-edges");
    let (short, spent) = assert_three_way(&src, need - 1);
    assert_eq!(short, Ended::OutOfFuel, "one short of the exact need");
    assert_eq!(spent, need - 1, "an exhausted budget is drained to zero");
}

/// **A busy poll that never wakes its fiber is bounded by fuel, identically.** The same loop with the
/// notify never reached is an infinite loop like `INFINITE_LOOP`, and ends the same way on every
/// engine: `OutOfFuel`, the whole budget spent. (#1642 recorded it as a hang on all three because it
/// ran through `temen_run`'s defaults: 2^34 fuel on the interpreters, and on the JIT no fuel at all —
/// only an embedder's `deadline` — as for any runaway guest.)
#[test]
fn jit_fuel_bounds_a_busy_resume_poll_that_never_wakes_its_fiber() {
    let (ended, spent) = assert_three_way(&busy_poll(None), 20_000);
    assert_eq!(ended, Ended::OutOfFuel);
    assert_eq!(spent, 20_000, "an exhausted budget is drained to zero");
}
