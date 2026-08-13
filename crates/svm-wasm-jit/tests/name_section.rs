//! The emitted wasm carries a **name section** (issue #865 diagnostics). Each emitted function is named
//! by its guest index (`gfN`) — or its source name on a `-g` build (`debug_info.func_names`, not
//! exercised here) — so a V8 wasm stack trace / a `svm_warm_jit` decline names the culprit function
//! instead of a bare `wasm-function[N]`. Without this, bisecting a 13 MB emit is guesswork.

use svm_wasm_jit::compile_module;

fn m(src: &str) -> svm_ir::Module {
    let m = svm_text::parse_module(src).expect("parse");
    svm_verify::verify_module(&m).expect("verify");
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

    // Both guest functions are named `gfN` (no `-g` source names in a parsed svm-text module).
    let has = |needle: &[u8]| wasm.windows(needle.len()).any(|w| w == needle);
    assert!(has(b"gf0"), "names guest function 0 as gf0");
    assert!(has(b"gf1"), "names guest function 1 as gf1");
    // The cross-tier `env.call_interp` import is named too (it appears in mixed-tier stacks).
    assert!(has(b"env.call_interp"), "names the cross-tier import");
}
