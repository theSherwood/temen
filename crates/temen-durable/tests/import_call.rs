//! #1300 Phase 2 — a **`call.import`** (a capability call bound at run time, IMPORTS.md) is a
//! suspend point exactly like `call.cap`: the host effect happens before the freeze, so the thaw
//! reloads the saved result rather than re-reading the clock. Pinned by the same freeze → thaw
//! property as the leaf `call.cap` case, with import slot 0 bound to the clock capability. Before
//! this the transform refused any module with a `call.import` (`UnsupportedInst`) — every
//! C-compiled guest (their `write`/`read`/`exit` are named imports) was outside a durable domain.

use temen_durable::{
    begin_thaw, init_durable_window, read_state, read_thaw_state, transform_module, write_state,
    STATE_NORMAL, STATE_UNWINDING,
};
use temen_interp::{run_capture_reserved_with_host, BoundImport, Host, Value};
use temen_ir::{Memory, Module};

const SIZE_LOG2: u8 = 18;
const WINDOW: usize = 1 << SIZE_LOG2;
/// The clock capability's interface id / `now` op (what the durable tests' `call.cap 2 0` names).
const CLOCK_TYPE_ID: u32 = 2;
const CLOCK_OP: u32 = 0;

fn instrument(src: &str) -> Module {
    let mut m = temen_text::parse_module(src).expect("parse");
    m.memory = Some(Memory {
        size_log2: SIZE_LOG2,
    });
    let inst = transform_module(&m).expect("transform");
    temen_verify::verify_module(&inst).expect("instrumented IR must verify");
    inst
}

/// Run func 0 with import slot 0 bound to the host's clock (`clock_ns` seeded).
fn run(
    inst: &Module,
    clock_ns: i64,
    window: &[u8],
) -> (Result<Vec<Value>, temen_interp::Trap>, Vec<u8>) {
    let mut host = Host::new();
    host.clock_ns = clock_ns;
    let clk = host.grant_clock();
    host.set_import_bindings(vec![BoundImport::required(CLOCK_TYPE_ID, CLOCK_OP, clk)]);
    let mut fuel = 1_000_000u64;
    run_capture_reserved_with_host(inst, 0, &[], &mut fuel, window, SIZE_LOG2, &mut host)
}

/// `v1 = 10` is live across the imported clock read; baseline = 42 + 10 = 52.
const IMPORT_LEAF: &str = r#"
import 0 "clock" (i32) -> (i64)
func () -> (i64) {
block 0 ()
{
  v1 = i64.const 10
  vz = i32.const 0
  vclk = call.import 0 (vz)
  vsum = i64.add vclk v1
  return vsum
  }
}
"#;

#[test]
fn a_call_import_suspend_point_reloads_its_result_on_thaw() {
    let inst = instrument(IMPORT_LEAF);
    let (baseline, _) = run(&inst, 42, &init_durable_window(WINDOW));
    let baseline = baseline.expect("baseline runs to completion");
    assert_eq!(baseline, vec![Value::I64(52)], "baseline: clock + 10");

    let mut win = init_durable_window(WINDOW);
    write_state(&mut win, STATE_UNWINDING);
    let (frozen, snapshot) = run(&inst, 42, &win);
    assert!(frozen.is_ok(), "freeze returns a placeholder, not a trap");
    assert_eq!(read_state(&snapshot), STATE_UNWINDING);

    // Thaw on a fresh host whose clock reads 0: the imported call's result (42) reloads from
    // the frame — it is not re-issued.
    let mut win = snapshot.clone();
    begin_thaw(&mut win, 0);
    let (thawed, final_win) = run(&inst, 0, &win);
    assert_eq!(thawed, Ok(baseline), "thaw equals the uninterrupted run");
    assert_eq!(read_thaw_state(&final_win, 0), STATE_NORMAL);
}
