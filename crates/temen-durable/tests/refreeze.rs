//! A thawed frame is a live frame. A thaw rebuilds a caller by **re-issuing** its call into the
//! frozen callee; once the callee has rewound, the program runs on *inside* that re-issued call, and
//! a second freeze can land beneath it. The re-issued call must poll as the forward path's does —
//! else the caller runs on with the placeholder its unwinding callee returned, and the second
//! artifact has lost the caller's frame. A fork is where it bites first: every fork after a process's
//! first unwinds through frames the previous fork's rewind rebuilt (#1768).
//!
//! The guest: `outer` calls `inner`, which makes two host calls and adds their results; `outer` adds
//! 1000. The host call numbered `freeze_on` asks for a freeze — it writes `UNWINDING`, as a runtime's
//! trigger (the JIT's fork gate) does — and the call's own poll starts the unwind. The first freeze
//! lands at `inner`'s first call; its thaw re-issues `outer`'s call, and the second freeze lands at
//! `inner`'s second call — beneath the re-issued one.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use temen_durable::{
    begin_thaw, init_durable_window, read_state, transform_module, STATE_OFF, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::Memory;

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

const SRC: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call 1 (v0)
  v2 = i64.const 1000
  v3 = i64.add v1 v2
  return v3
  }
}
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = call.cap 13 0 () -> (i64) v0 ()
  v2 = call.cap 13 0 () -> (i64) v0 ()
  v3 = i64.add v1 v2
  return v3
  }
}
"#;

fn module() -> temen_ir::Module {
    let mut m = temen_text::parse_module(SRC).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// Run from `win` with a host whose calls answer `made + 1, made + 2, …` (`made` calls answered
/// before this run) and whose call number `freeze_on` (1-based, this run's) asks for a freeze.
fn run(
    inst: &temen_ir::Module,
    made: i64,
    freeze_on: Option<i64>,
    win: &[u8],
) -> (Result<Vec<Value>, Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.set_durable(true);
    let calls = Arc::new(AtomicI64::new(0));
    let hf = host.grant_host_proc(Box::new(move |_op, _args, mem, _| {
        let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
        if Some(n) == freeze_on {
            let mem = mem.expect("a window");
            mem.write_bytes(STATE_OFF, &STATE_UNWINDING.to_le_bytes())
                .expect("the freeze word is mapped");
        }
        Ok(vec![made + n])
    }));
    let mut fuel = 100_000u64;
    run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(hf)],
        &mut fuel,
        win,
        SIZE_LOG2,
        &mut host,
    )
}

#[test]
fn a_freeze_beneath_a_thaws_reissued_call_unwinds_the_caller_again() {
    let inst = module();
    let (baseline, _) = run(&inst, 0, None, &init_durable_window(WINDOW, TEST_ARENA));
    assert_eq!(baseline, Ok(vec![Value::I64(1003)]), "1000 + 1 + 2");

    // The first freeze: at `inner`'s first call, `outer` frozen at its call.
    let (r, first) = run(&inst, 0, Some(1), &init_durable_window(WINDOW, TEST_ARENA));
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the first freeze unwinds both frames"
    );
    assert_eq!(read_state(&first), STATE_UNWINDING);

    // Its thaw: `outer` re-issues its call, `inner` reloads its first answer and makes its second
    // call, which freezes — beneath `outer`'s re-issued call, which must unwind too.
    let mut win = first.clone();
    begin_thaw(&mut win, TEST_ARENA, 0);
    let (r, second) = run(&inst, 1, Some(1), &win);
    assert_eq!(
        r,
        Ok(vec![Value::I64(0)]),
        "the second freeze unwinds `outer` again, not runs it on with `inner`'s placeholder"
    );
    assert_eq!(read_state(&second), STATE_UNWINDING);

    // The second artifact holds both frames: its thaw finishes the uninterrupted run.
    let mut win = second;
    begin_thaw(&mut win, TEST_ARENA, 0);
    let (r, _) = run(&inst, 2, None, &win);
    assert_eq!(
        r, baseline,
        "the second artifact thaws to the uninterrupted run"
    );
}
