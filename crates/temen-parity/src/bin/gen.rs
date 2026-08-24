//! Regenerate the checked-in `OPS_PARITY.md` at the repo root: `cargo run -p temen-parity`.

use std::path::PathBuf;

fn main() {
    // crates/temen-parity/ -> repo root.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let out = root.join("OPS_PARITY.md");
    let md = temen_parity::render_markdown();
    std::fs::write(&out, md).expect("write OPS_PARITY.md");
    eprintln!("wrote {}", out.display());
}
