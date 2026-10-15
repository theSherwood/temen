//! Equality harness for the bytecode engine's **§14 executor-child seam** (INTERP_PERF.md Slice
//! 1c-5g): a spawn (`Instantiator.instantiate_rec`, `call.cap 6 17`) and `Instantiator.join`
//! (`call.cap 6 1`). A child runs on the cooperative scheduler in a window of its own, with an
//! attenuated powerbox (an `Instantiator` + an `AddressSpace` over that window), and is joined through
//! the shared §12 thread machinery.
//!
//! Each case is checked **bit-identical** to the tree-walker `run_with_host`; `.expect(Some)` gates
//! that the bytecode engine drove the module (didn't fall back). The host grants the `Instantiator`
//! capability (iface 6); the handle reaches the guest as func 0's argument.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, Host, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;

/// A host for `m` with an `Instantiator` over the low 128 KiB, and the `"budget"` and `"child"` (`m`'s
/// child image of func 1, #2219) of a detached spawn.
fn host(m: &Module) -> (Host, i32) {
    let mut h = Host::new();
    let budget = h.grant_budget(-1, 1 << 20, -1);
    h.register_cap_name("budget", budget);
    let inst = h.grant_instantiator();
    let child = h.grant_module(&temen_ir::child_image_at(m, 1).expect("child image"));
    h.register_cap_name("child", child);
    (h, inst)
}

/// Run `src`'s entry on both engines with an `Instantiator` granted over `[0, 128 KiB)`, and assert
/// the results are identical and equal to `want`.
fn check(src: &str, want: Vec<Value>) {
    let m = parse_module(src).expect("parse");

    let (mut h_tw, inst_tw) = host(&m);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(&m, 0, &[Value::I32(inst_tw)], &mut f_tw, &mut h_tw);

    let (mut h_bc, inst_bc) = host(&m);
    let mut f_bc = 5_000_000u64;
    let bc =
        bytecode::compile_and_run_with_host(&m, 0, &[Value::I32(inst_bc)], &mut f_bc, &mut h_bc)
            .expect("bytecode engine must support instantiate/join (Slice 1c-5g)");

    assert_eq!(tw, bc, "instantiate: tree-walker != bytecode\n{src}");
    assert_eq!(bc, Ok(want), "instantiate result\n{src}");
}

/// A two-arg child receives its starter caps `(Instantiator, AddressSpace)`. It uses the
/// `AddressSpace` (iface 5, op 1 = `unmap`) to decommit the second 16 KiB of its **own** window (the
/// first 16 KiB is its NULL guard, #1206 — a page op there is refused as at the root) and returns the
/// unmap result (0). The parent spawns it (its `"child"`, func 1's child image) detached through a v1
/// record at 17728, paid from the `"budget"` it resolves, and returns what it returns.
const ADDRESS_SPACE: &str = r#"memory 18
data 16640 "budget"
data 16656 "child"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vnp = i64.const 16640
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  q3b = i64.const 17740
  i32.store q3b vb
  vcp = i64.const 16656
  vcl = i64.const 5
  vc = self.resolve vcp vcl
  q3m = i64.const 17736
  i32.store q3m vc
  q3a0 = i64.const 17728
  v5 = call.cap 6 17 (i64) -> (i32) v0 (q3a0)
  v6_st, v6 = call.cap 6 1 (i32) -> (i64, i64) v0 (v5)
  return v6
  }
}
func (i64, i64) -> (i64) {
block 0 (v0: i64, v1: i64) {
  v2 = i32.wrap_i64 v1
  v3 = i64.const 16384
  v4 = i64.const 16384
  v5 = call.cap 5 1 (i64, i64) -> (i64) v2 (v3, v4)
  return v5
  }
}
"#;

#[test]
fn two_arg_child_manages_its_own_pages() {
    let src = format!(
        "{ADDRESS_SPACE}{}",
        rec::segment(17728, &SpawnRec::default())
    );
    check(&src, vec![Value::I64(0)]);
}

/// A child trap (`unreachable`) is its parent's `join` status, never the parent's trap (#2066) —
/// identically on both engines. The parent spawns its `"child"` as [`ADDRESS_SPACE`]'s does, by the
/// spawn record at 17728, and returns the status.
const CHILD_TRAP: &str = r#"memory 17
data 16640 "budget"
data 16656 "child"
func (i32) -> (i64) {
block 0 (v0: i32) {
  vnp = i64.const 16640
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  q3b = i64.const 17740
  i32.store q3b vb
  vcp = i64.const 16656
  vcl = i64.const 5
  vc = self.resolve vcp vcl
  q3m = i64.const 17736
  i32.store q3m vc
  q3a0 = i64.const 17728
  v5 = call.cap 6 17 (i64) -> (i32) v0 (q3a0)
  v6_st, v6 = call.cap 6 1 (i32) -> (i64, i64) v0 (v5)
  return v6_st
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  unreachable
  }
}
"#;

#[test]
fn a_child_trap_is_its_joins_status() {
    let src = format!("{CHILD_TRAP}{}", rec::segment(17728, &SpawnRec::default()));
    check(&src, vec![Value::I64(temen_ir::trap_code::UNREACHABLE)]);
}
