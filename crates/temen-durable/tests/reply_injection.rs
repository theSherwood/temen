//! #1768 — **reply injection**, the durable half of fork (FORK.md §3, §9.5): a frame frozen at a
//! capability call thaws past that call with a reply the host chose, not the one the call returned.
//! One frozen window, two injected replies, two thaws — the same call site returns twice.
//!
//! Also the **selective** transform a fork compiles with ([`TransformOpts::sites`]): only the calls
//! that can fork are suspend points, so only the functions that reach one are instrumented and every
//! other function — another capability call included — is left byte-identical.
//!
//! On the interpreter, which is the transform's oracle; the JIT runs the same instrumented IR.

use temen_durable::{
    begin_thaw, init_durable_window, inject_leaf_reply, read_state, transform, write_state,
    TransformOpts, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, StreamRole, Value};
use temen_ir::{Inst, Memory, Module};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

// `entry(clock, out)`: a helper does a **non-site** capability call (a zero-length write) and
// returns 5; then the **site** — the `Clock.now` a fork stands in for — whose result is added to
// the helper's, so a thaw's answer is `5 + reply`. The helper's result is live across the site, so
// it rides the leaf frame beside the reply.
const SRC: &str = r#"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = call 1 (v1)
  v3 = i32.const 0
  v4 = call.cap 2 0 (i32) -> (i64) v0 (v3)
  v5 = i64.add v2 v4
  return v5
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 0
  v2 = i64.const 0
  v3 = call.cap 0 1 (i64, i64) -> (i64) v0 (v1, v2)
  v4 = i64.const 5
  return v4
  }
}
"#;

/// The fork site of this test: `Clock.now`. (A real fork is the personality's `fork`; the transform
/// does not care which op it is, only that the predicate names it.)
fn is_site(i: &Inst) -> bool {
    matches!(
        i,
        Inst::CapCall {
            type_id: temen_ir::cap_id::CLOCK,
            op: 0,
            ..
        }
    )
}

fn plain() -> Module {
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    m
}

/// The fork-instrumented module, and where its entry's body now lives (the entry's own slot holds a
/// barrier: no function's address is taken, so the entry is one no instrumented `call.dyn` selects).
fn fork_instrumented() -> (Module, u32) {
    let t = transform(&plain(), &TransformOpts::fork(&is_site)).expect("transform");
    temen_verify::verify_module(&t.module).expect("verify");
    (t.module, t.body[0])
}

/// Run `m` at `entry` from `window` with a fresh host whose clock reads `clock`; the result and
/// final window.
fn run(
    (m, entry): &(Module, u32),
    window: &[u8],
    clock: i64,
) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock;
    let clk = host.grant_clock();
    let out = host.grant_stream(StreamRole::Out);
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(
        m,
        *entry,
        &[Value::I32(clk), Value::I32(out)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    )
}

#[test]
fn only_the_functions_that_reach_a_site_are_instrumented() {
    let (before, (after, entry)) = (plain(), fork_instrumented());
    assert_ne!(
        before.funcs[0], after.funcs[entry as usize],
        "the entry reaches the site"
    );
    assert_eq!(
        before.funcs[1], after.funcs[1],
        "the helper's capability call is not a site: the helper is untouched"
    );
}

#[test]
fn a_frozen_call_thaws_twice_with_two_injected_replies() {
    let m = fork_instrumented();

    // Uninterrupted: 5 + the clock.
    let (r, _) = run(&m, &init_durable_window(WINDOW, TEST_ARENA), 42);
    assert_eq!(r, Ok(vec![Value::I64(47)]));

    // Freeze at the site: the helper's call runs (it is not a poll), the site's poll unwinds.
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (r, frozen) = run(&m, &win, 42);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the unwind returns a placeholder"
    );
    assert_eq!(read_state(&frozen), STATE_UNWINDING);

    // Two copies of the one frozen window, each told a different answer. The clocks are different
    // again, so a thaw that re-issued the call instead of taking the injected reply would show it.
    for (reply, clock) in [(1000, 7), (0, 9)] {
        let mut copy = frozen.clone();
        inject_leaf_reply(&mut copy, TEST_ARENA, 0, reply);
        begin_thaw(&mut copy, TEST_ARENA, 0);
        let (r, _) = run(&m, &copy, clock);
        assert_eq!(
            r,
            Ok(vec![Value::I64(5 + reply)]),
            "the copy told {reply} resumes past the call with {reply}, the helper's 5 reloaded"
        );
    }
}

/// With no injection the leaf reloads what the call returned before the freeze — the ordinary
/// durable thaw, unchanged by the leaf's results-first layout.
#[test]
fn an_uninjected_thaw_reloads_the_calls_own_result() {
    let m = fork_instrumented();
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);
    let (_, mut frozen) = run(&m, &win, 42);
    begin_thaw(&mut frozen, TEST_ARENA, 0);
    let (r, _) = run(&m, &frozen, 9);
    assert_eq!(r, Ok(vec![Value::I64(47)]));
}

/// A fork duplicates the thread alive; a freeze keeps only the window. So the vCPU TLS register —
/// the thread's, outside the window — instruments for a fork ([`TransformOpts::carries_thread`]) and
/// fails closed for a freeze, whose thaw would otherwise run with the register lost.
#[test]
fn vcpu_tls_instruments_for_a_fork_and_fails_closed_for_a_freeze() {
    let mut m = temen_text::parse_module(
        r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i64.const 50000
  vcpu.tls.set v1
  v2 = call.cap 2 0 () -> (i64) v0 ()
  v3 = vcpu.tls.get
  v4 = i64.add v2 v3
  return v4
  }
}
"#,
    )
    .expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let forked = transform(&m, &TransformOpts::fork(&is_site))
        .expect("a fork carries the register")
        .module;
    temen_verify::verify_module(&forked).expect("verify");
    assert_eq!(
        transform(&m, &TransformOpts::DURABLE).err(),
        Some(temen_durable::TransformError::UnsupportedInst),
        "a freeze cannot carry the register, so it refuses the op"
    );
}
