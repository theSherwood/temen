//! The shadow stack traps on overflow instead of corrupting guest memory (R9 / §12.7).
//!
//! The shadow stack mirrors the call stack; the freeze-path `UNWIND` check refuses to push
//! a frame whose top would cross `TEST_ARENA.end` into the guest's region. Reaching a real
//! overflow by natural recursion is impractical for typical frames (`MAX_CALL_DEPTH` caps the
//! reified call stack), so we drive the guard directly: seed the shadow-SP near the top of the
//! reserve, so the very next push would cross it. This is exactly why the check exists — a
//! large-frame guest recursing near the cap must trap here, never write past the reserve.

use temen_durable::{init_durable_window, transform_module, write_state, STATE_UNWINDING};
use temen_interp::{run_capture_reserved_with_host, Host, Trap, Value};
use temen_ir::{Memory, Module};

/// The arena every durable test module declares: the pre-#1503 fixed placement `[guard+64, 1<<16)`.
const TEST_ARENA: temen_ir::durable_abi::ShadowArena = temen_ir::durable_abi::ShadowArena {
    base: 16448,
    end: 65536,
};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

const LEAF: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32) {
  v1 = i32.const 0
  v2 = call.cap 2 0 (i32) -> (i64) v0 (v1)
  v3 = i64.const 100
  v4 = i64.add v2 v3
  return v4
  }
}
"#;

fn instrument() -> Module {
    let mut m = temen_text::parse_module(LEAF).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
        shadow: Some(TEST_ARENA),
    });
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("verify");
    inst
}

/// Freeze with the shadow-SP pre-seeded to `sp` (simulating an already-`sp`-deep stack).
fn freeze_with_sp(inst: &Module, sp: u64) -> Result<Vec<Value>, Trap> {
    let mut win = init_durable_window(WINDOW, TEST_ARENA);
    write_state(&mut win, STATE_UNWINDING);
    // §12.8 4A.5: the root context's shadow-SP word is the first 8 bytes of its region (at
    // `ShadowArena::region_base(0)`), not the legacy global `SHADOW_SP_OFF`.
    win[TEST_ARENA.region_base(0) as usize..TEST_ARENA.region_base(0) as usize + 8]
        .copy_from_slice(&sp.to_le_bytes());
    let mut host = Host::new();
    host.clock_ns = 42;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    let (r, _) = run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        &win,
        SIZE_LOG2,
        &mut host,
    );
    r
}

#[test]
fn shadow_overflow_traps_instead_of_corrupting() {
    let inst = instrument();

    // SP already at the very top of the reserve: the next frame push would cross
    // `TEST_ARENA.end` into guest memory → the check traps instead.
    assert!(
        freeze_with_sp(&inst, TEST_ARENA.end - 8).is_err(),
        "a push past the reserve traps, never writes guest memory"
    );

    // From the base, the same freeze fits and returns its placeholder.
    assert!(
        freeze_with_sp(&inst, TEST_ARENA.region_base(0)).is_ok(),
        "a freeze that fits within the reserve still works"
    );
}
