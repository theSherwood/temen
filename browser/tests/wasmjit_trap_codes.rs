//! #1735 — `web/wasmjit.js` hand-mirrors the wasm-JIT's `env.trap` codes to classify a trapped
//! call. They are the one trap wire code (`temen_ir::trap_code`) now, and a stale mirror would
//! quietly misclassify every trap, so pin the copy to the source.

fn js_const(src: &str, name: &str) -> i32 {
    let prefix = format!("export const {name} = ");
    let line = src
        .lines()
        .find_map(|l| l.strip_prefix(&prefix))
        .unwrap_or_else(|| panic!("wasmjit.js defines {name}"));
    line.trim_end_matches(';')
        .trim()
        .parse()
        .expect("an integer")
}

#[test]
fn the_js_trap_codes_are_the_emitters() {
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/web/wasmjit.js"))
        .expect("read web/wasmjit.js");
    assert_eq!(
        js_const(&src, "TRAP_OUT_OF_FUEL"),
        temen_wasm_jit::TRAP_OUT_OF_FUEL
    );
    assert_eq!(
        js_const(&src, "TRAP_MEMORY_FAULT"),
        temen_wasm_jit::TRAP_MEMORY_FAULT
    );
}
