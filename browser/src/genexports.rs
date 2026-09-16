//! Regenerate `browser/EXPORTS.md` — the JS ↔ Rust export ABI table (#1414). `cargo run --bin
//! genexports` from `browser/`; `tests/exports_abi.rs` pins the result.
use std::path::Path;

fn main() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let exports = temen_browser::exports_abi::rust_exports(&root.join("src"));
    let refs = temen_browser::exports_abi::js_references(root);
    let out = root.join("EXPORTS.md");
    std::fs::write(
        &out,
        temen_browser::exports_abi::render_markdown(&exports, &refs),
    )
    .expect("write EXPORTS.md");
    eprintln!("wrote {}", out.display());
}
