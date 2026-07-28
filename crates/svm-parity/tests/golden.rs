//! The checked-in `OPS_PARITY.md` must equal what the generator produces. If this fails, the
//! manifest changed without regenerating the doc — run `cargo run -p svm-parity`.

#[test]
fn ops_parity_md_is_up_to_date() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("OPS_PARITY.md");
    let on_disk =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let fresh = svm_parity::render_markdown();
    assert!(
        on_disk == fresh,
        "OPS_PARITY.md is stale — regenerate with `cargo run -p svm-parity`",
    );
}
