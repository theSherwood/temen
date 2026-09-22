//! #1641 — **a guest's `atomic.wait` timeout is the guest's**, pinned across TreeWalk / Bytecode /
//! Cranelift JIT (INVARIANTS.md invariant 9: the tree-walk interpreter is the oracle; the fast
//! backends run these identically or decline, never diverge).
//!
//! `MAX_WAIT` used to cap the wait itself — "regardless of the guest's requested timeout" — while
//! its doc claimed to be "a pure anti-wedge backstop, never semantics … not shaping guest-visible
//! behavior". It was: a guest asking for 30 s was handed `WAIT_TIMED_OUT` after 10 s, an ordinary
//! finite timeout truncated to a third of itself. Only the oracle did this; bytecode (logical
//! clock) and the JIT (real futex) both honoured the guest, so the one authoritative engine was
//! the wrong one, and no test asked.
//!
//! The kernel waits **longer than `MAX_WAIT`** — that is the whole point, so the gate costs real
//! wall time on the two engines with a real clock. Shorter and it proves nothing.

use temen_run::{instantiate, Backend, Outcome, RunConfig, Value};

/// Comfortably over the 10 s `MAX_WAIT`, so a truncation lands a full second below the floor
/// asserted below, and comfortably under it is nothing a scheduler hiccup can fake.
const GUEST_TIMEOUT_NS: i64 = 12_000_000_000;
/// Above `MAX_WAIT`, below the guest's deadline: a reading in between means the backstop answered.
const FLOOR: std::time::Duration = std::time::Duration::from_millis(11_000);

fn kernel() -> String {
    format!(
        r#"memory 16
export 0 func "_start" 0
func () -> (i64) {{
block 0 () {{
  va = i64.const 16392
  ve = i32.const 0
  vto = i64.const {GUEST_TIMEOUT_NS}
  vst = i32.atomic.wait va ve vto
  v64 = i64.extend_i32_u vst
  return v64
  }}
}}
"#
    )
}

/// `WAIT_TIMED_OUT`.
const TIMED_OUT: i64 = 2;

fn run(backend: Backend) -> (i64, std::time::Duration) {
    let module = temen_text::parse_module(&kernel()).expect("parse");
    temen_verify::verify_module(&module).expect("verify");
    let instance = instantiate(module).expect("instantiate");
    let t0 = std::time::Instant::now();
    let run = instance.run(backend, &RunConfig::default()).expect("run");
    let dt = t0.elapsed();
    match run.outcome {
        Outcome::Returned(v) => match v.as_slice() {
            [Value::I64(x)] => (*x, dt),
            other => panic!("{backend:?}: unexpected result {other:?}"),
        },
        other => panic!("{backend:?}: unexpected outcome {other:?}"),
    }
}

/// Every engine must report the guest's own timeout expiring — never the backstop's.
#[test]
fn a_guest_timeout_longer_than_the_backstop_is_honoured_on_every_backend() {
    for backend in [Backend::TreeWalk, Backend::Bytecode, Backend::Jit] {
        let (status, dt) = run(backend);
        assert_eq!(
            status, TIMED_OUT,
            "{backend:?}: a wait with no notifier must end in WAIT_TIMED_OUT, got {status} in {dt:?}"
        );
    }
}

/// The engines that sleep on a real clock must actually sleep the guest's deadline. Bytecode is
/// excluded **by construction, not by omission**: its cooperative driver advances a *logical* clock
/// to the deadline the moment nothing else can run, so it reports the right status in under a
/// millisecond. Asserting wall time there would pin the clock model, not the semantics — the status
/// assertion above is what binds it, and it is the assertion that was missing.
#[test]
fn the_real_clock_engines_sleep_the_guest_deadline_not_the_backstop() {
    for backend in [Backend::TreeWalk, Backend::Jit] {
        let (status, dt) = run(backend);
        assert_eq!(status, TIMED_OUT, "{backend:?}: expected WAIT_TIMED_OUT");
        assert!(
            dt >= FLOOR,
            "{backend:?}: returned after {dt:?}, under the {FLOOR:?} floor — the 10 s MAX_WAIT \
             backstop decided this, not the guest's {GUEST_TIMEOUT_NS} ns deadline (#1641)"
        );
    }
}
