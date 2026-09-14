//! Regenerate the checked-in parity views: `cargo run -p temen-parity`.
//!
//! Two outputs, one manifest (INVARIANTS #15) — `OPS_PARITY.md` at the repo root for reading in the
//! tree, and `browser/web/assets/ops_parity.json` for the playground's parity page (#1418). Both are
//! pinned by `tests/golden.rs`, so neither can go stale without CI saying so.

use std::path::PathBuf;

fn main() {
    // crates/temen-parity/ -> repo root.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");

    let md_out = root.join("OPS_PARITY.md");
    std::fs::write(&md_out, temen_parity::render_markdown()).expect("write OPS_PARITY.md");
    eprintln!("wrote {}", md_out.display());

    // Lives under `browser/web/` so the Pages deploy picks it up with the rest of the site
    // (`pages.yml` copies `web/.` wholesale) — no workflow change needed to ship the page.
    let json_out = root
        .join("browser")
        .join("web")
        .join("assets")
        .join("ops_parity.json");
    std::fs::write(&json_out, temen_parity::render_json()).expect("write ops_parity.json");
    eprintln!("wrote {}", json_out.display());

    // The capability × axis frontier matrix (#1413) — INVARIANTS #14's other six axes.
    let frontier_out = root.join("FRONTIER.md");
    std::fs::write(&frontier_out, temen_parity::render_frontier_markdown())
        .expect("write FRONTIER.md");
    eprintln!("wrote {}", frontier_out.display());
}
