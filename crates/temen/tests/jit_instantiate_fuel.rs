//! §14 nested-child **counted fuel** on the JIT, differentially vs. the interpreter (INTERP_PERF.md
//! "Fuel unification" step 5). Before this slice a nested JIT child ran **un-metered** — a runaway
//! child only stopped via the §5 async kill cell (`jit_killpath.rs`), never via counted fuel, while
//! the interpreter metered it. Now the child is compiled with its own budget cell seeded to
//! `min(quota, parent_remaining)` (the interpreter's `child_fuel` contract), so a runaway child traps
//! `OutOfFuel` at the same point on both backends. These tests are differential: the tree-walk
//! interpreter is the oracle, and the JIT must agree.
//!
//! Gated on `fiber_supported()` — elsewhere an `instantiate` is an inert `CapFault`, no child runs.

use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_jit::{compile_and_run_capture_reserved_with_host_fuel, JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

/// Parent (func 0) `instantiate`s a child (func 1) in a 4 KiB carve with quota `q` (the 4th operand),
/// `join`s it, returns its result. The child loops forever (a taken back-edge each iteration), so with
/// counted fuel it exhausts rather than running unbounded.
fn runaway_child_src(quota: i64) -> String {
    format!(
        "memory 17\n\
         func (i32) -> (i64) {{\n\
         block 0 (v0: i32) {{\n\
         \x20 v1 = i64.const 1\n\
         \x20 v2 = i64.const 0\n\
         \x20 v3 = i64.const 12\n\
         \x20 v4 = i64.const {quota}\n\
         \x20 v5 = call.cap 6 0 (i64, i64, i64, i64) -> (i32) v0 (v1, v2, v3, v4)\n\
         \x20 v6 = call.cap 6 1 (i32) -> (i64) v0 (v5)\n\
         \x20 return v6\n\
           }}\n\
         }}\n\
         func (i64) -> (i64) {{\n\
         block 0 (v0: i64) {{\n\
         \x20 br 1(v0)\n\
         }}\n\
         block 1 (v1: i64) {{\n\
         \x20 v2 = i64.const 1\n\
         \x20 v3 = i64.add v1 v2\n\
         \x20 br 1(v3)\n\
           }}\n\
         }}\n"
    )
}

/// Run the entry on both backends with the **same** fuel budget (the JIT armed via the counted-fuel
/// capture entry), an `Instantiator` granted over the whole window. Returns both outcomes.
fn both_fuel(src: &str, fuel: u64) -> (Result<Vec<Value>, Trap>, JitOutcome) {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    let win = 1u64 << 17;
    let init: Vec<u8> = Vec::new();

    let mut hi = Host::new();
    let ih = hi.grant_instantiator(0, win);
    let mut hj = Host::new();
    let jh = hj.grant_instantiator(0, win);
    assert_eq!(ih, jh, "the Instantiator grant must encode identically");

    let mut ifuel = fuel;
    let (ir, _) =
        run_capture_reserved_with_host(&m, 0, &[Value::I32(ih)], &mut ifuel, &init, 0, &mut hi);
    let mut jfuel = fuel;
    let (jo, _) = compile_and_run_capture_reserved_with_host_fuel(
        &m,
        0,
        &[jh as i64],
        &init,
        0,
        temen_run::cap_thunk,
        &mut hj as *mut Host as *mut core::ffi::c_void,
        &mut jfuel,
    )
    .expect("jit compiles");
    (ir, jo)
}

/// Reduce both backends' outcomes to a comparable shape: `OutOfFuel` trap, some-other trap, or the
/// returned i64 values. Lets a differential assert agreement without hard-coding trap propagation.
#[derive(Debug, PartialEq)]
enum Shape {
    OutOfFuel,
    OtherTrap,
    Returned(Vec<i64>),
}
fn interp_shape(r: &Result<Vec<Value>, Trap>) -> Shape {
    match r {
        Err(Trap::OutOfFuel) => Shape::OutOfFuel,
        Err(_) => Shape::OtherTrap,
        Ok(vs) => Shape::Returned(
            vs.iter()
                .map(|v| match v {
                    Value::I32(x) => *x as i64,
                    Value::I64(x) => *x,
                    _ => panic!("unexpected result type"),
                })
                .collect(),
        ),
    }
}
fn jit_shape(o: &JitOutcome) -> Shape {
    match o {
        JitOutcome::Trapped(TrapKind::OutOfFuel) => Shape::OutOfFuel,
        JitOutcome::Trapped(_) => Shape::OtherTrap,
        JitOutcome::Exited(_) => Shape::OtherTrap, // guest `exit` — these programs never do
        JitOutcome::Returned(vs) => Shape::Returned(vs.clone()),
    }
}

#[test]
fn jit_nested_child_fuel_stops_runaway_matches_interp() {
    if !temen_jit::fiber_supported() {
        return;
    }
    // quota = 0 ⇒ the child inherits the parent's whole remaining budget; the runaway drains it. With
    // the child now fuel-metered on the JIT, it traps `OutOfFuel` instead of hanging (no watchdog).
    let (ir, jo) = both_fuel(&runaway_child_src(0), 50_000);
    assert_eq!(
        interp_shape(&ir),
        Shape::OutOfFuel,
        "interp: a runaway nested child must exhaust counted fuel, got {ir:?}"
    );
    assert_eq!(
        jit_shape(&jo),
        Shape::OutOfFuel,
        "JIT: the runaway nested child must trap OutOfFuel like the interp, got {jo:?}"
    );
}

#[test]
fn jit_nested_child_fuel_quota_cap_matches_interp() {
    if !temen_jit::fiber_supported() {
        return;
    }
    // quota = 500 caps the child far below the parent's 5M budget — pinning `min(quota, parent)`: the
    // child exhausts at ~500 while the parent has plenty. Whatever the interp does with that (whether
    // the child trap propagates or the parent survives), the JIT must agree.
    let (ir, jo) = both_fuel(&runaway_child_src(500), 5_000_000);
    assert_eq!(
        interp_shape(&ir),
        jit_shape(&jo),
        "quota-capped child: interp {ir:?} vs JIT {jo:?} disagree"
    );
    // And it must actually be an exhaustion (the cap bit), not a clean return — else the test is
    // vacuous (a bug that left the child un-metered would let both simply run to some other outcome).
    assert_eq!(
        interp_shape(&ir),
        Shape::OutOfFuel,
        "the quota cap must make the runaway child exhaust, got {ir:?}"
    );
}
