//! The emitted wasm carries a **name section** (issue #865 diagnostics). Each emitted function is named
//! by its **export symbol name** (`lua_checkstack`, `JS_CallInternal` …) — every defined function is
//! exported by its symbol name, so the real names ride in `m.exports` with no `-g` flag and no format
//! change (#907). Precedence: a DWARF source name (a `-g` build's `debug_info.func_names`, the reserved
//! level-2 rung, not exercised here) → the export symbol name → the guest index `gfN` (only for an
//! unexported synthesized helper). So a V8 wasm stack trace / a `temen_warm_jit` decline names the culprit
//! function instead of a bare `wasm-function[N]`. Without this, bisecting a 13 MB emit is guesswork.

use temen_wasm_jit::compile_module;

fn m(src: &str) -> temen_ir::Module {
    let m = temen_text::parse_module(src).expect("parse");
    temen_verify::verify_module(&m).expect("verify");
    m
}

fn uleb(b: &[u8]) -> (u64, usize) {
    let (mut v, mut shift, mut n) = (0u64, 0u32, 0usize);
    loop {
        let byte = b[n];
        v |= ((byte & 0x7f) as u64) << shift;
        n += 1;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
    }
    (v, n)
}

/// The payload of the `"name"` custom section (everything after the `"name"` id string), or None.
fn name_section(wasm: &[u8]) -> Option<Vec<u8>> {
    let mut i = 8; // skip magic (4) + version (4)
    while i < wasm.len() {
        let id = wasm[i];
        i += 1;
        let (len, adv) = uleb(&wasm[i..]);
        i += adv;
        let body = &wasm[i..i + len as usize];
        i += len as usize;
        if id == 0 {
            let (nlen, a2) = uleb(body);
            if &body[a2..a2 + nlen as usize] == b"name" {
                return Some(body[a2 + nlen as usize..].to_vec());
            }
        }
    }
    None
}

#[test]
fn emitted_wasm_names_functions_by_guest_index() {
    // Two in-subset integer functions: gf0 calls gf1 directly. Both are emitted, so both are named.
    let wasm = compile_module(&m(r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i64.add v0 v1
  return v2
  }
}"#))
    .expect("module emits");

    // The `"name"` custom section exists and leads with the function-names subsection (id 1).
    let ns = name_section(&wasm).expect("emitted wasm carries a `name` custom section");
    assert_eq!(
        ns.first().copied(),
        Some(0x01),
        "function-names subsection (id 1)"
    );

    // With no exports declared, both guest functions fall back to `gfN`.
    let has = |needle: &[u8]| wasm.windows(needle.len()).any(|w| w == needle);
    assert!(has(b"gf0"), "names guest function 0 as gf0");
    assert!(has(b"gf1"), "names guest function 1 as gf1");
    // The cross-tier `env.call_interp` import is named too (it appears in mixed-tier stacks).
    assert!(has(b"env.call_interp"), "names the cross-tier import");
}

/// The name section prefers the **export symbol name** over the `gfN` fallback (#907): every defined
/// function is exported by its symbol name, so the real names ride in `m.exports` with no `-g` flag —
/// the same path that turns `gf597` into `lua_checkstack` on the real Lua/QuickJS/Tcl assets. Here
/// func 0 is exported as `compute`, so the trace names `compute` (not `gf0`); the unexported func 1
/// stays `gf1`.
#[test]
fn emitted_wasm_prefers_export_symbol_names() {
    let wasm = compile_module(&m(r#"export 0 func "compute" 0
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i64.add v0 v1
  return v2
  }
}"#))
    .expect("module emits");

    let has = |needle: &[u8]| wasm.windows(needle.len()).any(|w| w == needle);
    assert!(
        name_section(&wasm).is_some(),
        "emitted wasm carries a `name` custom section"
    );
    assert!(
        has(b"compute"),
        "func 0 named by its export symbol `compute`"
    );
    assert!(has(b"gf1"), "unexported func 1 stays gf1");
}

/// A **demoted** module (#907, the `temen-strip` packaging step) still names its functions: the name
/// rides `debug_info.func_names` — the strippable diagnostic table demotion moves it to — instead of
/// the exports table, and the emitter's precedence (`func_names` → export → `gfN`) picks it up with
/// no export entry at all. This is the guardrail that slimming the semantic surface does not degrade
/// a stack trace back to `gfN`.
#[test]
fn emitted_wasm_names_demoted_functions_from_func_names() {
    let mut module = m(r#"
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = call 1 (v0)
  return v1
  }
}
func (i64) -> (i64) {
block 0 (v0: i64) {
  v1 = i64.const 7
  v2 = i64.add v0 v1
  return v2
  }
}"#);
    module.debug_info = Some(temen_ir::DebugInfo {
        func_names: vec![temen_ir::FuncName {
            func: 1,
            name: "emit_demoted".into(),
        }],
        ..Default::default()
    });
    let wasm = compile_module(&module).expect("module emits");

    let has = |needle: &[u8]| wasm.windows(needle.len()).any(|w| w == needle);
    assert!(
        has(b"emit_demoted"),
        "unexported func 1 named from func_names (the demoted-name path)"
    );
    assert!(has(b"gf0"), "unnamed, unexported func 0 stays gf0");
}
