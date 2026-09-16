//! **The JS ↔ Rust export ABI pin** (#1414). The page calls the cdylib's exports by name, as strings;
//! nothing at build time relates the two. This does, from the source, in every `cargo test`.

use std::path::Path;
use temen_browser::exports_abi::{js_references, render_markdown, rust_exports};

fn root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

/// Quoted `temen_*` strings in the JS that name something other than an export — each with the reason.
/// Exact, not a prefix list: an entry that stops being quoted anywhere, or starts being exported,
/// fails `the_allowlist_is_exactly_what_it_says`, so this cannot silently widen.
const QUOTED_NON_EXPORTS: &[(&str, &str)] = &[(
    "temen_fs",
    "the memfs data-image kind (`temen-fs`), not a symbol",
)];

#[test]
fn every_temen_name_the_js_touches_is_an_export() {
    let exports = rust_exports(&root().join("src"));
    let refs = js_references(root());
    let missing: Vec<String> = refs
        .iter()
        .filter(|(n, _)| {
            !exports.contains_key(*n) && !QUOTED_NON_EXPORTS.iter().any(|(a, _)| a == n)
        })
        .map(|(n, files)| {
            format!(
                "  {n}  <- {}",
                files
                    .iter()
                    .map(|f| f.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "JS calls these by name but the cdylib exports no such symbol — a runtime failure on the \
         first card to reach them:\n{}",
        missing.join("\n")
    );
}

#[test]
fn the_allowlist_is_exactly_what_it_says() {
    let exports = rust_exports(&root().join("src"));
    let refs = js_references(root());
    for (name, why) in QUOTED_NON_EXPORTS {
        assert!(
            refs.contains_key(*name),
            "{name} ({why}) is allowlisted but no JS quotes it any more — drop the entry"
        );
        assert!(
            !exports.contains_key(*name),
            "{name} ({why}) is allowlisted as a non-export but is now exported — drop the entry"
        );
    }
}

#[test]
fn exports_md_is_up_to_date() {
    let fresh = render_markdown(&rust_exports(&root().join("src")), &js_references(root()));
    let on_disk = std::fs::read_to_string(root().join("EXPORTS.md")).unwrap_or_default();
    assert!(
        on_disk.replace("\r\n", "\n") == fresh,
        "browser/EXPORTS.md is stale — regenerate with `cargo run --bin genexports` (in browser/)"
    );
}

/// The parser's two non-obvious cases, pinned so a refactor of either cannot quietly drop exports:
/// an attribute between `#[no_mangle]` and the `fn`, and the one export-minting macro.
#[test]
fn the_parser_sees_attribute_runs_and_the_macro() {
    let exports = rust_exports(&root().join("src"));
    assert!(
        exports.contains_key("temen_par_ev_b"),
        "`par_ev_getter!` expansions must count as exports"
    );
    let gated = exports.values().filter(|e| e.cfg.is_some()).count();
    assert!(
        gated > 0,
        "some exports are cfg-gated; the parser must record that"
    );
    assert!(
        exports.len() >= 340,
        "export count fell to {}",
        exports.len()
    );
}
