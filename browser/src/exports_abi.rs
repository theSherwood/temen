//! **The JS ↔ Rust export ABI, as data** (#1414, the slice Rust exhaustiveness cannot reach).
//!
//! Every browser driver family is a set of `#[no_mangle] extern "C"` exports (`_open` / `_step` /
//! `_deliver*` / `_ptr`+`_len` / `_close`, …) that the page's JS calls **by name**, as a string. A
//! renamed, removed or `cfg`-gated-away export fails nowhere at build time — it fails in a browser
//! test card, at runtime, when a card happens to reach it. This module makes the surface checkable:
//!
//! - [`rust_exports`] reads the source and returns every export, with its `cfg` gate if any;
//! - [`js_references`] reads every non-vendored `.js` / `.mjs` / `.html` under `browser/` and
//!   returns every `temen_*` name the JS touches as a member access (`ex.temen_x`) or a quoted
//!   string (`ex['temen_x']`, `call1('temen_x', …)`), with the files that touch it;
//! - `tests/exports_abi.rs` pins that the second is a subset of the first, and that
//!   `browser/EXPORTS.md` — the family × op table [`render_markdown`] produces — is fresh.
//!
//! Source-level, deliberately: it needs no wasm build, so it runs in every `cargo test` of this
//! crate. The cost is that a `#[cfg(...)]`-gated export counts as present whichever build is being
//! considered; the table marks those, so a JS reference to a `cfg(atomics)`-only export is at least
//! *visible* as one, which is how the op-13 driver's non-atomics gap was found (#1530).
//!
//! Hand-rolled scanning rather than a regex dependency: the patterns are three, fixed, and the
//! crate's dev-dependencies are deliberately tiny.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// One export: its symbol and the `#[cfg(...)]` attribute gating it, if any.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Export {
    pub name: String,
    pub cfg: Option<String>,
}

/// Every `#[no_mangle] pub extern "C" fn` under `src_dir`, plus the `par_ev_getter!` macro's
/// expansions (the one macro that mints exports). Attribute lines between `#[no_mangle]` and the
/// `fn` (`#[allow(...)]`, `#[cfg(...)]`) are tolerated and, for `cfg`, recorded.
pub fn rust_exports(src_dir: &Path) -> BTreeMap<String, Export> {
    let mut out = BTreeMap::new();
    for file in rs_files(src_dir) {
        let text = std::fs::read_to_string(&file).unwrap_or_default();
        // The contiguous run of `#[...]` lines above the current line. `#[cfg(...)]` conventionally
        // precedes `#[no_mangle]` and `#[allow(...)]` follows it, so the whole run is what decides.
        let mut attrs: Vec<String> = Vec::new();
        for line in text.lines() {
            let t = line.trim_start();
            if let Some(attr) = t.strip_prefix("#[") {
                attrs.push(attr.trim_end_matches(']').to_string());
                continue;
            }
            if attrs.iter().any(|a| a == "no_mangle") {
                if let Some(name) = fn_name(t) {
                    let cfg = attrs.iter().find(|a| a.starts_with("cfg(")).cloned();
                    out.insert(
                        name.to_string(),
                        Export {
                            name: name.to_string(),
                            cfg,
                        },
                    );
                }
            }
            if let Some(rest) = t.strip_prefix("par_ev_getter!(") {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                out.insert(name.clone(), Export { name, cfg: None });
            }
            // Doc comments sit above the attribute run; anything else ends it.
            if !t.starts_with("///") {
                attrs.clear();
            }
        }
    }
    out
}

/// The `fn` name on a `pub [unsafe] extern "C" fn name(` line, if this is one.
fn fn_name(t: &str) -> Option<&str> {
    let t = t.strip_prefix("pub ")?;
    let t = t.strip_prefix("unsafe ").unwrap_or(t);
    let t = t.strip_prefix("extern \"C\" fn ")?;
    let end = t
        .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
        .unwrap_or(t.len());
    (end > 0).then_some(&t[..end])
}

fn rs_files(dir: &Path) -> Vec<PathBuf> {
    let mut v = Vec::new();
    walk(
        dir,
        &mut |p| p.extension().is_some_and(|e| e == "rs"),
        &mut v,
    );
    v.sort();
    v
}

fn walk(dir: &Path, keep: &mut dyn FnMut(&Path) -> bool, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, keep, out);
        } else if keep(&p) {
            out.push(p);
        }
    }
}

/// Every `temen_*` name the JS touches, with the files touching it. Scans `.js`, `.mjs` and `.html`
/// under `browser_dir`, skipping `node_modules/`, `target/` and `web/assets/` (built artifacts and
/// vendored bundles). Comments are stripped first: the docs mention export names in prose constantly.
pub fn js_references(browser_dir: &Path) -> BTreeMap<String, BTreeSet<PathBuf>> {
    let mut files = Vec::new();
    walk(
        browser_dir,
        &mut |p| {
            let s = p.to_string_lossy();
            !s.contains("/node_modules/")
                && !s.contains("/target/")
                && !s.contains("/web/assets/")
                && p.extension()
                    .is_some_and(|e| e == "js" || e == "mjs" || e == "html")
        },
        &mut files,
    );
    let mut out: BTreeMap<String, BTreeSet<PathBuf>> = BTreeMap::new();
    for file in files {
        let text = strip_comments(&std::fs::read_to_string(&file).unwrap_or_default());
        let rel = file
            .strip_prefix(browser_dir)
            .unwrap_or(&file)
            .to_path_buf();
        let b = text.as_bytes();
        let mut i = 0;
        while let Some(off) = text[i..].find("temen_") {
            let start = i + off;
            let end = start
                + text[start..]
                    .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                    .unwrap_or(text.len() - start);
            let before = if start == 0 { b' ' } else { b[start - 1] };
            let after = b.get(end).copied().unwrap_or(b' ');
            // A member access names an export, as does a quoted string that IS the name (a computed
            // `ex[fn]` fed a literal). A template literal that merely starts with the prefix
            // (`` `temen_onramp_${…}` ``) does not, nor does a bare word in prose or a path.
            let quoted = matches!(before, b'\'' | b'"' | b'`') && after == before;
            if before == b'.' || quoted {
                out.entry(text[start..end].to_string())
                    .or_default()
                    .insert(rel.clone());
            }
            i = end.max(start + 1);
        }
    }
    out
}

/// Drop `/* … */` blocks and `// …` line tails (but not the `//` inside a `://` URL).
fn strip_comments(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"/*") {
            i = s[i + 2..].find("*/").map_or(b.len(), |e| i + 2 + e + 2);
        } else if b[i..].starts_with(b"//") && (i == 0 || b[i - 1] != b':') {
            i = s[i..].find('\n').map_or(b.len(), |e| i + e);
        } else {
            out.push(b[i] as char);
            i += 1;
        }
    }
    out
}

/// The family a symbol belongs to: `temen_<family>_…`, or `(corpus runners)` for the handful of
/// non-`temen_` exports the differential corpus drives.
pub fn family(name: &str) -> String {
    name.strip_prefix("temen_")
        .map(|r| r.split('_').next().unwrap_or(r).to_string())
        .unwrap_or_else(|| "(corpus runners)".to_string())
}

/// `EXPORTS.md`: the family × export table. Generated, golden-pinned, never edited by hand.
pub fn render_markdown(
    exports: &BTreeMap<String, Export>,
    refs: &BTreeMap<String, BTreeSet<PathBuf>>,
) -> String {
    let mut fams: BTreeMap<String, Vec<&Export>> = BTreeMap::new();
    for e in exports.values() {
        fams.entry(family(&e.name)).or_default().push(e);
    }
    let mut s = String::new();
    s.push_str("# Browser export ABI\n\n");
    s.push_str(
        "**Generated — do not edit by hand.** Regenerate with `cargo run --bin genexports` (in \
         `browser/`). Every `#[no_mangle] extern \"C\"` export of the `temen-browser` cdylib, by driver \
         family, against what the page's JS actually calls by name. `tests/exports_abi.rs` pins that \
         every name the JS touches is exported, and that this file is fresh (#1414).\n\n",
    );
    let total = exports.len();
    let referenced = exports.keys().filter(|n| refs.contains_key(*n)).count();
    let gated = exports.values().filter(|e| e.cfg.is_some()).count();
    s.push_str(&format!(
        "**{total} exports** in {} families — {referenced} referenced from JS, {} referenced by \
         nothing, {gated} behind a `cfg`.\n\n",
        fams.len(),
        total - referenced,
    ));
    s.push_str("| family | exports | referenced from JS | `cfg`-gated |\n|---|---:|---:|---:|\n");
    for (f, es) in &fams {
        s.push_str(&format!(
            "| `{f}` | {} | {} | {} |\n",
            es.len(),
            es.iter().filter(|e| refs.contains_key(&e.name)).count(),
            es.iter().filter(|e| e.cfg.is_some()).count(),
        ));
    }
    s.push_str("\n## Exports by family\n\n");
    s.push_str(
        "An export marked *unreferenced* is called by no JS or HTML in `browser/`; one marked with a \
         `cfg` exists only in builds where that cfg holds, so a JS caller on another build gets a \
         missing function at runtime (the source-level scan cannot see which build a page runs).\n\n",
    );
    for (f, es) in &fams {
        s.push_str(&format!("### `{f}`\n\n"));
        for e in es {
            s.push_str(&format!("- `{}`", e.name));
            if let Some(c) = &e.cfg {
                s.push_str(&format!(" — `{c}`"));
            }
            if !refs.contains_key(&e.name) {
                s.push_str(" — *unreferenced*");
            }
            s.push('\n');
        }
        s.push('\n');
    }
    s
}
