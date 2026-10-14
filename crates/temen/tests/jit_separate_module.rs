//! §14 **separate-module children on the JIT**, differentially vs. the interpreter — the
//! "plugin-in-plugin" story end to end. The host verifies a *different* module, grants both backends
//! an identical `Module` capability (iface 8), and the parent guest spawns a child domain running
//! that module through a v1 record (op 17). On the JIT the child module is resolved through
//! `temen_run::module_resolver` (a host callback, never guest-reachable) and **compiled at the
//! spawn** — §14's "nesting cost paid at setup" for foreign code. Both backends must agree on the
//! child's result (its data segments seeded into its own window), a forged module handle's
//! `CapFault`, and the final parent window bytes.

#[path = "../../temen-interp/tests/support/rec.rs"]
mod rec;

use temen_interp::{run_capture_reserved_with_host, Host, MemLayout, Trap, Value};
use temen_ir::SpawnRec;
use temen_jit::{JitOutcome, TrapKind};
use temen_text::parse_module;
use temen_verify::verify_module;

type BothOut = (Result<Vec<Value>, Trap>, Vec<u8>, JitOutcome, Vec<u8>);

/// The child ("plugin") module [`both`] grants: a 64 KiB window, `data 16484 "VM"` (above the NULL
/// guard, #1206), and an entry that reads its data byte, stores a marker, and returns `byte + 1000`.
fn child_src() -> &'static str {
    "memory 16
data 16484 \"VM\"
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
"
}

/// Run `parent_src` on both backends with identical grants: an `Instantiator` over the whole 128 KiB
/// window and a `Module` capability for `child_src` (their handles passed as the entry's two args),
/// and a `"budget"` a detached spawn pays from.
fn both(parent_src: &str) -> BothOut {
    both_with(parent_src, child_src())
}

/// [`both`] over a given child module.
fn both_with(parent_src: &str, child_src: &str) -> BothOut {
    let parent = parse_module(parent_src).expect("parse parent");
    verify_module(&parent).expect("verify parent");
    let child = parse_module(child_src).expect("parse child");
    verify_module(&child).expect("verify child");
    let init: Vec<u8> = (0..(128u64 << 10))
        .map(|i| (i as u8).wrapping_mul(31) ^ 0xa5)
        .collect();
    let grant = |h: &mut Host| {
        let budget = h.grant_budget(-1, 1 << 20, -1);
        h.register_cap_name("budget", budget);
        (h.grant_instantiator(0, 128 << 10), h.grant_module(&child))
    };

    let mut hi = Host::new();
    let (ii, mi) = grant(&mut hi);
    let mut hj = Host::new();
    let (ij, mj) = grant(&mut hj);
    assert_eq!((ii, mi), (ij, mj), "grants must encode identically");

    let mut fuel = 5_000_000u64;
    let (ir, imem) = run_capture_reserved_with_host(
        &parent,
        0,
        &[Value::I32(ii), Value::I32(mi)],
        &mut fuel,
        &init,
        0,
        &mut hi,
    );
    let (jo, jmem) = temen_run::jit_cap_run(
        &parent,
        0,
        &[ij as i64, mj as i64],
        &MemLayout::image(init),
        0,
        0,
        &mut hj,
        None,
    )
    .expect("jit");
    (ir, imem, jo, jmem.bytes().to_vec())
}

/// #1720 — a module nests **as built**: a powerbox `_start` (no params — how every program is built
/// to start) is a module child on the JIT exactly as on the interpreter. An `i32` status joins
/// sign-extended (the JIT's trampoline stores it as an interpreter `Reg` holds it), and an empty
/// result joins as `0`.
#[test]
fn jit_powerbox_entry_child_matches_interp() {
    if !temen_jit::fiber_supported() {
        return; // no JIT nesting runtime on this target
    }
    // Spawn the module's entry 0 detached, through a v1 record paid from its `"budget"` → join →
    // the child's result.
    let parent = format!(
        "memory 17
data 16640 \"budget\"
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  rma = i64.const 17432
  i32.store rma v1
  vnp = i64.const 16640
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  rba = i64.const 17436
  i32.store rba vb
  rp = i64.const 17408
  v6 = call.cap 6 17 (i64) -> (i32) v0 (rp)
  v7_st, v7 = call.cap 6 1 (i32) -> (i64, i64) v0 (v6)
  return v7
  }}
}}
{}",
        rec::segment(17408, &SpawnRec::v1())
    );
    let status = "memory 16
export 0 func \"_start\" 0
func () -> (i32) {
block 0 () {
  v0 = i32.const -7
  return v0
  }
}
";
    let nothing = "memory 16
export 0 func \"_start\" 0
func () -> () {
block 0 () {
  return
  }
}
";
    for (child, want) in [(status, -7i64), (nothing, 0)] {
        let (ir, imem, jo, jmem) = both_with(&parent, child);
        assert_eq!(ir, Ok(vec![Value::I64(want)]), "interp, child:\n{child}");
        assert!(
            matches!(jo, JitOutcome::Returned(ref s) if s == &[want]),
            "jit: {jo:?}, child:\n{child}"
        );
        assert_eq!(
            imem, jmem,
            "interp/JIT parent windows diverge, child:\n{child}"
        );
    }
}

/// Validation parity: a forged module handle — the parent's `Instantiator` named as the record's
/// module — is a `CapFault` on both backends. (A window size other than the module's declared memory
/// is `instantiate_record.rs`'s `a_v1_record_with_size_zero_gets_the_declared_window`.)
#[test]
fn a_forged_module_handle_faults_on_both_backends() {
    if !temen_jit::fiber_supported() {
        return;
    }
    let parent = format!(
        "memory 17
data 16640 \"budget\"
func (i32, i32) -> (i64) {{
block 0 (v0: i32, v1: i32) {{
  rma = i64.const 17432
  i32.store rma v0
  vnp = i64.const 16640
  vnl = i64.const 6
  vb = self.resolve vnp vnl
  rba = i64.const 17436
  i32.store rba vb
  rp = i64.const 17408
  v6 = call.cap 6 17 (i64) -> (i32) v0 (rp)
  v7 = i64.extend_i32_s v6
  return v7
  }}
}}
{}",
        rec::segment(17408, &SpawnRec::v1())
    );
    let (ir, _i, jo, _j) = both(&parent);
    assert_eq!(ir, Err(Trap::CapFault), "interp: a forged module handle");
    assert_eq!(jo, JitOutcome::Trapped(TrapKind::CapFault), "jit");
}
