//! #2101 — in a run that is not durable, the bytes at the guard base are the guest's own data, not a
//! freeze phase. A `1` there (a data segment's first word, or a store before a `thread.spawn`) is
//! `STATE_UNWINDING`'s value; the tree-walker used to carry it into the scheduler, where a parked
//! vCPU in that phase reads as a freeze in flight and every park is re-admitted as if for one.

use temen_interp::{bytecode, Host, MemLayout, Value};
use temen_jit::JitOutcome;

/// A timed wait on a word nobody changes, which times out (`2`), in the root or in the thread it
/// spawns. The root first stores `word` at the guard base; `segment` puts a `1` there before the run
/// starts. So a `segment` with a `0` stored is the phase the root starts in, and a `1` stored is the
/// one a thread spawned after it starts in.
fn waiter(segment: bool, word: i32, root_waits: bool) -> temen_ir::Module {
    let data = if segment {
        "data 16384 \"\\x01\\x00\\x00\\x00\"\n"
    } else {
        ""
    };
    // The root waits beside a thread that returns at once, or joins the thread that waits.
    let (spawned, rest) = if root_waits {
        (2, "  vr = call 1 (vsp, va)\n  vj = thread.join vt\n")
    } else {
        (1, "  vr = thread.join vt\n")
    };
    let src = format!(
        "memory 17\n{data}\nfunc () -> (i64) {{\nblock 0 () {{\n  vg = i64.const 16384\n  \
         vw = i32.const {word}\n  i32.store vg vw\n  vsp = i64.const 0\n  va = i64.const 0\n  \
         vt = thread.spawn {spawned} vsp va\n{rest}  return vr\n  }}\n}}\n\n\
         func (i64, i64) -> (i64) {{\nblock 0 (vsp: i64, va: i64) {{\n  vat = i64.const 20480\n  \
         vexp = i32.const 0\n  vns = i64.const 1000000\n  vr = i32.atomic.wait vat vexp vns\n  \
         vr64 = i64.extend_i32_s vr\n  return vr64\n  }}\n}}\n\n\
         func (i64, i64) -> (i64) {{\nblock 0 (vsp: i64, va: i64) {{\n  return va\n  }}\n}}\n"
    );
    let m = temen_text::parse_module(&src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

/// What the wait returned on the tree-walker, the bytecode engine and the Cranelift JIT.
fn on_every_engine(m: &temen_ir::Module) -> [i64; 3] {
    let one = |r: Result<Vec<Value>, temen_interp::Trap>| match r.as_deref() {
        Ok([Value::I64(x)]) => *x,
        other => panic!("the guest ended {other:?}"),
    };
    let mut fuel = u64::MAX / 2;
    let tree_walker = one(temen_interp::run_with_host(
        m,
        0,
        &[],
        &mut fuel,
        &mut Host::new(),
    ));
    let mut fuel = u64::MAX / 2;
    let bytecode = bytecode::compile_and_run_with_host(m, 0, &[], &mut fuel, &mut Host::new())
        .map(one)
        .expect("the bytecode engine runs it");
    let window = MemLayout::image(vec![0; 1 << 17]);
    let reserved = temen_ir::DEFAULT_RESERVED_LOG2;
    let jit = match temen_run::jit_cap_run(m, 0, &[], &window, reserved, 0, &mut Host::new(), None)
        .map(|(outcome, _)| outcome)
    {
        Ok(JitOutcome::Returned(v)) => v[0],
        other => panic!("the JIT run ended {other:?}"),
    };
    [tree_walker, bytecode, jit]
}

const TIMED_OUT: i64 = 2;

/// A data segment whose first word is `1` at the guard base: the root starts in no freeze phase.
#[test]
fn a_one_at_the_guard_base_is_data_to_the_root() {
    assert_eq!(on_every_engine(&waiter(true, 0, true)), [TIMED_OUT; 3]);
}

/// The root stores `1` at the guard base before it spawns: the thread starts in no freeze phase.
#[test]
fn a_one_at_the_guard_base_is_data_to_a_thread_spawned_after_it() {
    assert_eq!(on_every_engine(&waiter(false, 1, false)), [TIMED_OUT; 3]);
}

/// The control: a `0` there.
#[test]
fn the_wait_times_out_over_a_zero() {
    assert_eq!(on_every_engine(&waiter(false, 0, true)), [TIMED_OUT; 3]);
    assert_eq!(on_every_engine(&waiter(false, 0, false)), [TIMED_OUT; 3]);
}
