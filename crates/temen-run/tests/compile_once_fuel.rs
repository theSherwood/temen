//! #2113 — the JIT's **compile-once** entries meter the default fuel, as the per-call ones do.
//!
//! [`PowerboxProgram`] and [`JitSession`] compile their code once and run it many times, so their
//! code charges a fuel cell that outlives each run: every run re-arms it from that run's own node,
//! which [`Host::begin_activation`] grants [`DEFAULT_FUEL`], and hands back what it did not burn. A
//! guest reads its fuel with `fuel.remaining` (self op 13): `i64::MAX` when its code meters nothing,
//! else what is left of the activation's grant.

use temen_interp::{Host, DEFAULT_FUEL};
use temen_ir::{Module, DEFAULT_RESERVED_LOG2};
use temen_jit::JitOutcome;
use temen_run::{grant_jit, run_powerbox, JitSession, Outcome, PowerboxProgram, Value};
use temen_text::parse_module;
use temen_verify::verify_module;

/// `_start` returns what `fuel.remaining` reads at its entry.
const FUEL_READBACK: &str = "\
memory 16
export 0 func \"_start\" 0
func () -> (i64) {
block 0 () {
  vz = i32.const 0
  vr = call.cap 4294967295 13 () -> (i64) vz ()
  return vr
  }
}
";

fn load(src: &str) -> Module {
    let m = parse_module(src).expect("parse");
    verify_module(&m).expect("verify");
    m
}

fn readback(outcome: &Outcome) -> i64 {
    match outcome {
        Outcome::Returned(v) => match v.as_slice() {
            [Value::I64(r)] => *r,
            other => panic!("one i64, not {other:?}"),
        },
        other => panic!("a return, not {other:?}"),
    }
}

/// A [`PowerboxProgram`] run reads the default grant, less what it has burned, exactly as the same
/// program run once through [`run_powerbox`] does — and every run starts from a fresh grant.
#[test]
fn a_powerbox_program_meters_the_default_fuel_on_every_run() {
    let m = load(FUEL_READBACK);
    let once = readback(&run_powerbox(&m, b"").expect("run_powerbox").outcome);
    assert!(
        once > 0 && once <= DEFAULT_FUEL as i64,
        "the one-shot run meters the default fuel: {once}"
    );
    let mut prog = PowerboxProgram::compile(m).expect("compile");
    for run in 0..3 {
        let r = readback(&prog.run(b"").expect("run").outcome);
        assert_eq!(
            r, once,
            "run {run}: the cached code meters as the one-shot code does"
        );
    }
}

/// Each [`JitSession`] prompt is an activation with the default grant, and a recompacted module
/// charges the same cell, so a prompt after [`JitSession::compact`] reads what one before it did.
#[test]
fn a_jit_session_meters_the_default_fuel_across_prompts_and_compaction() {
    let base = load(FUEL_READBACK);
    let mut host = Host::new();
    let jit = grant_jit(&mut host, &base, 0);
    let domain = host.resolve_jit_domain(jit).expect("domain");
    let mut session =
        JitSession::new(&base, 0, DEFAULT_RESERVED_LOG2, 0, domain, 0, host).expect("session");
    let prompt = |s: &mut JitSession| match s.run_prompt(&[]).expect("prompt") {
        JitOutcome::Returned(v) => v[0],
        other => panic!("a return, not {other:?}"),
    };
    let first = prompt(&mut session);
    assert!(
        first > 0 && first <= DEFAULT_FUEL as i64,
        "a prompt meters the default fuel: {first}"
    );
    assert_eq!(prompt(&mut session), first, "the next prompt starts afresh");
    session.compact().expect("compact");
    assert_eq!(
        prompt(&mut session),
        first,
        "the recompacted code charges the same cell"
    );
}
