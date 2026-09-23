//! The opt-in `callprof` per-op histogram: every executed op in the primary module is counted
//! against its IR location `(func, block, inst)`, so a static per-instruction analysis — a call
//! site's spill cost (temen #1627), a function's instruction count — can be weighted by how often it
//! actually ran. Built only with `--features callprof`; the default build has none of this.
#![cfg(feature = "callprof")]

use temen_interp::bytecode::{self, SRC_TERM};
use temen_interp::Value;
use temen_text::parse_module;

/// Func 0 loops five times, calling func 1 once per iteration, then returns the counter.
const LOOP_CALL: &str = r#"func () -> (i64) {
block 0 () {
  v0 = i64.const 0
  br 1(v0)
  }
block 1 (v0: i64) {
  v1 = call 1 (v0)
  v2 = i64.const 1
  v3 = i64.add v0 v2
  v4 = i64.const 5
  v5 = i64.lt_s v3 v4
  br_if v5 1(v3) 2(v3)
  }
block 2 (v0: i64) {
  return v0
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  return v0
  }
}
"#;

fn count(ops: &[((u32, u32, u32), u64)], key: (u32, u32, u32)) -> u64 {
    ops.iter().find(|(k, _)| *k == key).map_or(0, |(_, n)| *n)
}

#[test]
fn op_histogram_counts_each_call_site_and_body_by_ir_location() {
    let m = parse_module(LOOP_CALL).expect("parse");
    bytecode::callprof_reset(m.funcs.len());
    let mut fuel = u64::MAX;
    let r =
        bytecode::compile_and_run(&m, 0, &[], &mut fuel).expect("the engine accepts the module");
    assert_eq!(r.expect("runs"), vec![Value::I64(5)]);

    let ops = bytecode::callprof_op_snapshot();
    // The call site — func 0, block 1, inst 0 — ran once per iteration.
    assert_eq!(
        count(&ops, (0, 1, 0)),
        5,
        "call site count; histogram: {ops:?}"
    );
    // Its callee's only op is the `return` terminator, which carries the SRC_TERM flag.
    assert_eq!(
        count(&ops, (1, 0, SRC_TERM)),
        5,
        "callee body count; histogram: {ops:?}"
    );
    // The loop entry ran once, and the per-function call counter agrees with the call site.
    assert_eq!(count(&ops, (0, 0, 0)), 1);
    assert_eq!(bytecode::callprof_snapshot()[1], 5);

    // A reset clears the per-op histogram along with the per-function one.
    bytecode::callprof_reset(m.funcs.len());
    assert!(bytecode::callprof_op_snapshot().is_empty());
}
