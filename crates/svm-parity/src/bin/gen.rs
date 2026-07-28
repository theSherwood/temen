//! Regenerate the checked-in `OPS_PARITY.md` at the repo root: `cargo run -p svm-parity`.

use std::path::PathBuf;

fn main() {
    // crates/svm-parity/ -> repo root.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let out = root.join("OPS_PARITY.md");
    let md = svm_parity::render_markdown();
    std::fs::write(&out, md).expect("write OPS_PARITY.md");
    eprintln!("wrote {}", out.display());
}
