//! #1300 Phase 2 (item 3, v128 half) — a **`v128` live across a suspend point** freezes/thaws on the
//! real interpreter. The vector spills through `v128.store` into its 16-byte frame slot and reloads
//! through `v128.load`; a lane extracted after the thaw carries the pre-freeze value, and the thawed
//! run on a fresh host equals the uninterrupted run. Before this the transform refused the module
//! (`UnsupportedInst`), so a durable domain narrowed what SIMD code its guest could run.

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

/// Two vectors are live across the clock read: `i64x2.splat 7` (lane 1 read after) and an
/// `i32x4` sum `[1,2,3,4] + [10,20,30,40]` (lane 2 = 33 read after). Baseline = 42 + 7 + 33 = 82.
const V128_ACROSS: &str = r#"
func (i32) -> (i64) {
block 0 (v0: i32)
{
  vs = i64.const 7
  vv = i64x2.splat vs
  va = v128.const 1 0 0 0 2 0 0 0 3 0 0 0 4 0 0 0
  vb = v128.const 10 0 0 0 20 0 0 0 30 0 0 0 40 0 0 0
  vsum = i32x4.add va vb
  vz = i32.const 0
  vclk = call.cap 2 0 (i32) -> (i64) v0 (vz)
  vl = i64x2.extract_lane 1 vv
  vl2 = i32x4.extract_lane 2 vsum
  vl2w = i64.extend_i32_s vl2
  vr = i64.add vclk vl
  vr2 = i64.add vr vl2w
  return vr2
  }
}
"#;

#[test]
fn a_v128_live_across_the_suspend_point_survives_freeze_and_thaw() {
    let inst = instrument(V128_ACROSS);
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));
    let baseline = baseline.expect("baseline runs to completion");
    assert_eq!(
        baseline,
        vec![Value::I64(42 + 7 + 33)],
        "baseline: clock + 7 + 33"
    );

    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (frozen, snapshot) = run(&inst, 42, &win);
    assert!(frozen.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(read_state(&snapshot), STATE_UNWINDING);

    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, final_win) = run(&inst, 0, &win);
    assert_eq!(thawed, Ok(baseline), "thaw equals the uninterrupted run");
    assert_eq!(read_thaw_state(&final_win, 0), STATE_NORMAL);
}
