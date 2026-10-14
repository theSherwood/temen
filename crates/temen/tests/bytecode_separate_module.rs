//! Equality harness for the bytecode engine's **§14 separate-module executor child** (INTERP_PERF.md
//! Slice 1c-5h). The host verifies a *different* module and grants the parent a `Module` capability
//! (iface 8); the parent names it in a spawn record (op 17) to spawn a child **domain** running that
//! module — the "plugin-in-plugin" story — and joins it via the shared thread machinery (op 1). A
//! forged module handle is an inert `CapFault`.
//!
//! Each case is checked **bit-identical** to the tree-walker `run_with_host`; `.expect(Some)` gates
//! that the bytecode engine drove the parent module (didn't fall back). The parent entry takes
//! `(instantiator, module handle)`.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{bytecode, run_with_host, Host, Trap, Value};
use temen_ir::{Module, SpawnRec};
use temen_text::parse_module;

/// The child ("plugin") module: a 64 KiB window with a data segment `"VM"` at offset 16 KiB + 100
/// (above the NULL guard its carve reserves like any window, #1206). Its entry
/// (`(i64 instantiator) -> (i64)`) loads its own data byte there, stores a marker at offset 16 KiB, and
/// returns `byte + 1000` — exercising a foreign module's code, data, and window writes, confined.
const CHILD_SRC: &str = r#"memory 16
data 16484 "VM"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.const 16384
  v4 = i32.const 7
  i32.store8 v3 v4
  v5 = i64.extend_i32_u v2
  v6 = i64.const 1000
  v7 = i64.add v5 v6
  return v7
  }
}
"#;

/// A host granting the parent `(instantiator over the whole window, module handle for `child`)`, and
/// a `"budget"` that a detached spawn pays from.
fn host(child: &Module) -> (Host, i32, i32) {
    let mut h = Host::new();
    let budget = h.grant_budget(-1, 1 << 20, -1);
    h.register_cap_name("budget", budget);
    let ih = h.grant_instantiator(0, 128 << 10);
    let mh = h.grant_module(child);
    (h, ih, mh)
}

/// Run `parent_src`'s entry on both engines with `(instantiator over the whole window, module handle
/// for `child_src`)` as its two args, and assert the results are identical and equal to `want`.
fn check(parent_src: &str, child_src: &str, want: Result<Vec<Value>, Trap>) {
    let parent = parse_module(parent_src).expect("parse parent");
    let child = parse_module(child_src).expect("parse child");

    let (mut h_tw, ih_tw, mh_tw) = host(&child);
    let mut f_tw = 5_000_000u64;
    let tw = run_with_host(
        &parent,
        0,
        &[Value::I32(ih_tw), Value::I32(mh_tw)],
        &mut f_tw,
        &mut h_tw,
    );

    let (mut h_bc, ih_bc, mh_bc) = host(&child);
    let mut f_bc = 5_000_000u64;
    let bc = bytecode::compile_and_run_with_host(
        &parent,
        0,
        &[Value::I32(ih_bc), Value::I32(mh_bc)],
        &mut f_bc,
        &mut h_bc,
    )
    .expect("bytecode engine must support instantiate_module (Slice 1c-5h)");

    assert_eq!(
        tw, bc,
        "instantiate_module: tree-walker != bytecode\n{parent_src}"
    );
    assert_eq!(bc, want, "instantiate_module result\n{parent_src}");
}

/// A forged module handle is an inert `CapFault` on both engines: the parent's v1 record at 17408
/// names module 999, which was never granted.
const PARENT_FORGED: &str = r#"memory 17
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  rp = i64.const 17408
  v6 = call.cap 6 17 (i64) -> (i32) v0 (rp)
  v7 = i64.extend_i32_s v6
  return v7
  }
}
"#;

#[test]
fn forged_module_handle_faults_identically() {
    let forged = SpawnRec {
        modh: 999,
        ..SpawnRec::v1(0)
    };
    let parent = format!("{PARENT_FORGED}{}", rec::segment(17408, &forged));
    check(&parent, CHILD_SRC, Err(Trap::CapFault));
}

/// A two-export child ("plugin"): `"alpha"` (func 0) → `byte+1000`, `"beta"` (func 1) → `byte+2000`.
/// Used to prove the bytecode engine drives `Module` op 0 (`resolve_export`, F2) identically to the
/// tree-walker — name → funcidx through the generic `cap_dispatch_slots` seam.
const NAMED_CHILD_SRC: &str = r#"memory 16
data 16484 "VM"
export 0 func "alpha" 0
export 1 func "beta" 1
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 1000
  v5 = i64.add v3 v4
  return v5
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 16484
  v2 = i32.load8_u v1
  v3 = i64.extend_i32_u v2
  v4 = i64.const 2000
  v5 = i64.add v3 v4
  return v5
  }
}
"#;

/// The parent resolves "beta" (func 1, in its own data segment) via `Module` op 0, writes that funcidx
/// into the entry of a v1 record at 17408 with the module and the `"budget"` it resolves, spawns the
/// child detached → join → the *beta* result (`'V'` 86 + 2000).
const PARENT_BY_NAME: &str = r#"memory 17
data 16584 "beta"
data 16640 "budget"
func (i32, i32) -> (i64) {
block 0 (v0: i32, v1: i32) {
  v2 = i64.const 16584
  v3 = i64.const 4
  v4 = call.cap 8 0 (i64, i64) -> (i64) v1 (v2, v3)
  ve = i32.wrap_i64 v4
  rea = i64.const 17412
  i32.store rea ve
  rma = i64.const 17432
  i32.store rma v1
  vnp = i64.const 16640
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  rba = i64.const 17436
  i32.store rba vb
  rp = i64.const 17408
  v9 = call.cap 6 17 (i64) -> (i32) v0 (rp)
  v10_st, v10 = call.cap 6 1 (i32) -> (i64, i64) v0 (v9)
  return v10
  }
}
"#;

#[test]
fn module_child_addressed_by_export_name_matches_treewalker() {
    let parent = format!("{PARENT_BY_NAME}{}", rec::segment(17408, &SpawnRec::v1(0)));
    check(&parent, NAMED_CHILD_SRC, Ok(vec![Value::I64(2086)]));
}
