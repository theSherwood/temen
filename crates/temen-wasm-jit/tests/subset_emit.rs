//! `compile_module_reactor_keep` — the subset (profile-guided) emit used by the lazy-emit
//! measurement. Pins that (a) a `keep`-all set reproduces the whole-module `compile_module_reactor`
//! emitted set, (b) dropping a marshallable function moves it off wasm to cross-tier while the module
//! still validates, and (c) a non-marshallable function is force-kept (can't be dropped).

use temen_wasm_jit::{compile_module_reactor, compile_module_reactor_keep};
use wasmi::{Engine, Module as WModule};

fn parse(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).unwrap();
    temen_verify::verify_module(&m).unwrap();
    m
}

fn validates(wasm: &[u8]) {
    WModule::new(&Engine::default(), wasm).expect("emitted subset wasm must validate");
}

fn emitted_set(bm: &[bool]) -> Vec<usize> {
    bm.iter()
        .enumerate()
        .filter(|(_, &b)| b)
        .map(|(i, _)| i)
        .collect()
}

// f0 calls f1 (an int-signature helper) and f2 (also int). All in-subset.
const SRC: &str = r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  v2 = call 2 (v1)
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 3
  v2 = i64.mul v0 v1
  return v2
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i64.add v0 v1
  return v2
  }
}
"#;

#[test]
fn keep_all_matches_whole_module() {
    let m = parse(SRC);
    let (_, whole) = compile_module_reactor(&m, 0, false).unwrap();
    let keep = vec![true; m.funcs.len()];
    let (wasm, sub) = compile_module_reactor_keep(&m, 0, &keep, false).unwrap();
    validates(&wasm);
    assert_eq!(
        emitted_set(&whole),
        emitted_set(&sub),
        "keep-all must emit the whole-module set"
    );
}

#[test]
fn dropping_a_marshallable_function_moves_it_cross_tier() {
    let m = parse(SRC);
    // Keep f0 + f1, drop f2 (int signature ⇒ droppable to the interp).
    let keep = vec![true, true, false];
    let (wasm, sub) = compile_module_reactor_keep(&m, 0, &keep, false).unwrap();
    validates(&wasm);
    assert_eq!(
        emitted_set(&sub),
        vec![0, 1],
        "f2 must be dropped to cross-tier"
    );
}

#[test]
fn entry_is_always_kept() {
    let m = parse(SRC);
    // Even an empty keep set keeps the entry (f0) — the wasm-driven root must be emitted.
    let keep = vec![false, false, false];
    let (wasm, sub) = compile_module_reactor_keep(&m, 0, &keep, false).unwrap();
    validates(&wasm);
    assert!(sub[0], "the entry f0 must always be emitted");
}
