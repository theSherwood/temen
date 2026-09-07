//! #1300 Phase 2 (item 3) — **scalar conversions in a may-suspend prefix** freeze/thaw on the real
//! interpreter. A value that goes through every conversion family before the suspend point (width
//! extend / wrap, int→float, float cast, saturating and trapping float→int) and is used after it
//! must spill as its converted scalar and reload on thaw; the thawed run on a fresh host equals the
//! uninterrupted run. Before Phase 2 the transform refused the whole module (`UnsupportedInst`) — a
//! durable domain narrowed what its guest (and its guest-JIT units, #1296) could compile.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, read_thaw_state, transform_module, write_state,
    STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, Host, Value};
use temen_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;

fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

fn run(
    inst: &Module,
    clock_ns: i64,
    window: &[u8],
) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(
        inst,
        0,
        &[Value::I32(clk)],
        &mut fuel,
        window,
        SIZE_LOG2,
        &mut host,
    )
}

/// `7` travels `i32 → i64 → i32 → f64 → f32 → i32 (sat) → i64 → f64 → i64 (trap) → i64` before the
/// clock read and is added to it after: baseline = 42 + 7 = 49. Every intermediate but the last is
/// dead across the call; the last (`i64`) is the one the shadow frame must carry.
const CONVERTING_PREFIX: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32)
{
  vc = i32.const 7
  v1 = i64.extend_i32_u vc
  v2 = i32.wrap_i64 v1
  v3 = f64.convert_i32_s v2
  v4 = f32.demote_f64 v3
  v5 = i32.trunc_sat_f32_s v4
  v6 = i64.extend_i32_s v5
  v7 = f64.convert_i64_u v6
  v8 = i64.trunc_f64_s v7
  v9 = i64.reinterpret_f64 v7
  v10 = f64.reinterpret_i64 v9
  v11 = i64.trunc_f64_u v10
  vz = i32.const 0
  vclk = call.cap 2 0 (i32) -> (i64) v0 (vz)
  vsum = i64.add vclk v11
  vsum2 = i64.add vsum v8
  return vsum2
  }
}
"#;

#[test]
fn a_converting_prefix_survives_freeze_and_thaw() {
    let inst = instrument(CONVERTING_PREFIX);
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));
    let baseline = baseline.expect("baseline runs to completion");
    assert_eq!(
        baseline,
        vec![Value::I64(42 + 7 + 7)],
        "baseline: clock + 7 + 7"
    );

    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (frozen, snapshot) = run(&inst, 42, &win);
    assert!(frozen.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(read_state(&snapshot), STATE_UNWINDING);

    // Thaw on a fresh host with the clock at 0: the reloaded frame carries the converted i64s, and
    // the saved clock result (42) is reloaded rather than re-read.
    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, final_win) = run(&inst, 0, &win);
    assert_eq!(thawed, Ok(baseline), "thaw equals the uninterrupted run");
    assert_eq!(read_thaw_state(&final_win, 0), STATE_NORMAL);
}
