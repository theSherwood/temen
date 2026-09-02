//! #1151 — the browser's §14 codegen entry (`temen_par_enable_inst_codegen`) **declines** a granted
//! unit whose `env.call_interp` leaves are not pure. The Worker services those leaves with
//! `temen_wasmjit_call_interp` (a throwaway window + empty powerbox), which is faithful only for pure
//! leaves; a `map`-calling allocator helper that stores would `CapFault` / write a fresh window there
//! where the interpreter succeeds. Declined ⇒ the child runs on the interpreter, byte-identical. A
//! unit whose only leaf is pure (an `f64.fma` helper) still enables, with the entry eligible.

use temen_browser::{
    temen_par_enable_inst_codegen, temen_par_inst_eligible, temen_par_powerbox_inst,
};

/// Entry `(address_space) -> i64` → helper that `map`s (op 0) and stores: an impure leaf.
const MAP_HELPER: &str = r#"memory 17
func (i32) -> (i64) {
block 0 (v0: i32) {
  va = i64.const 4096
  vr = call 1 (v0, va)
  return vr
  }
}
func (i32, i64) -> (i64) {
block 0 (v0: i32, v1: i64) {
  vlen = i64.const 16384
  vm = call.cap 5 0 (i64, i64) -> (i64) v0 (v1, vlen)
  vk = i64.const 7
  i64.store v1 vk
  return vm
  }
}
"#;

/// Entry `(i64) -> i64` → a pure `f64.fma` helper (out of the nested subset, but faithful on the
/// throwaway servicer).
const PURE_HELPER: &str = r#"memory 17
func (i64) -> (i64) {
block 0 (v0: i64) {
  vr = call 1 (v0)
  return vr
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  vf = f64.reinterpret_i64 v0
  vg = f64.fma vf vf vf
  vi = i64.reinterpret_f64 vg
  return vi
  }
}
"#;

fn encode(src: &str) -> Vec<u8> {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    temen_encode::encode_module(&m)
}

/// One test body: the codegen stash + its once-per-run memoization are process-global (see
/// `par_tierup_driver.rs`), and each `temen_par_powerbox_inst` is the page-side run-generation bump
/// that re-arms the emit — so publish → enable is one serial sequence per recipe.
#[test]
fn inst_codegen_declines_impure_leaves_and_enables_pure_ones() {
    let bytes = encode(MAP_HELPER);
    assert_eq!(temen_par_powerbox_inst(1 << 17, bytes.as_ptr(), bytes.len()), 1);
    assert_eq!(
        temen_par_enable_inst_codegen(),
        0,
        "a map+store helper is an impure leaf: codegen must decline (interpreter child)"
    );
    assert_eq!(temen_par_inst_eligible(0), 0, "no eligibility stashed on decline");

    let bytes = encode(PURE_HELPER);
    assert_eq!(temen_par_powerbox_inst(1 << 17, bytes.as_ptr(), bytes.len()), 1);
    assert_eq!(temen_par_enable_inst_codegen(), 1, "a pure fma leaf is faithful: codegen enables");
    assert_eq!(temen_par_inst_eligible(0), 1, "the entry emits");
    assert_eq!(temen_par_inst_eligible(1), 0, "the fma helper is a cross-tier leaf");
}
