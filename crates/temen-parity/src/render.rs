//! Render the catalog + manifest as the human-readable `OPS_PARITY.md` matrix and as the
//! machine-readable `OPS_PARITY.json` the playground's parity page reads (#1418). Two renderings,
//! one manifest.

use crate::{catalog, Backend, Status};

/// Render the whole matrix as GitHub-flavored markdown. Deterministic (stable order) so the golden
/// test can byte-compare it against the checked-in file.
pub fn render_markdown() -> String {
    let ops = catalog();
    let mut s = String::new();

    s.push_str("# Op × backend parity matrix\n\n");
    s.push_str(
        "**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity` after \
         changing the manifest (`crates/temen-parity/src/`). This file is the human-readable view of \
         the exhaustive, test-checked classifier in `temen-parity`; the conformance test \
         (`crates/temen-parity/tests/conformance.rs`) pins every non-skipped row against what the \
         backends actually compile.\n\n",
    );
    s.push_str("Backends (DESIGN.md §3): the tree-walk interpreter is the **oracle** (defines \
         observable behavior); the bytecode interpreter is held bit-exact against it; the Cranelift \
         and wasm JITs are fail-closed accelerators that fold their non-subset back to the oracle \
         (INVARIANTS.md #9).\n\n");

    // Legend.
    s.push_str("## Legend\n\n");
    s.push_str(&format!("- {} **Full** — runs the op, observable behavior identical to the oracle (modulo the deliberately unpinned float-NaN bits / backend-local handle indices; DESIGN §3/§3a).\n", Status::Full.glyph()));
    s.push_str(&format!("- {} **Declines (parity not expected)** — folds to the oracle by design: a wasm-JIT concurrency/cap/fiber op (leaf accelerator), or a lowering with no target counterpart.\n", Status::Declines.glyph()));
    s.push_str(&format!("- {} **Not yet (parity not achieved)** — a real gap this backend could close but hasn't.\n", Status::NotYet.glyph()));
    s.push_str(&format!("- {} **Conditional** — Full where a build/target cfg holds, Declines elsewhere (the note names the condition).\n\n", Status::Conditional.glyph()));

    // Summary counts (over the two JIT columns — the interpreters are Full everywhere).
    let mut full = 0usize;
    let mut declines = 0usize;
    let mut notyet = 0usize;
    let mut cond = 0usize;
    let mut unaudited = 0usize;
    for op in &ops {
        for b in [Backend::Cranelift, Backend::WasmJit] {
            match op.cells()[b as usize].status {
                Status::Full => full += 1,
                Status::Declines => declines += 1,
                Status::NotYet => notyet += 1,
                Status::Conditional => cond += 1,
                // The op × backend axis has no unaudited cells — every op is classified for every
                // backend, which is the whole point of an exhaustive classifier. Counted rather than
                // ignored so that claim is checked by the summary line rather than assumed.
                Status::Unaudited => unaudited += 1,
            }
        }
    }
    s.push_str(&format!(
        "**{} ops.** Across the two JIT columns: {} {} Full · {} {} Declines · {} {} Not-yet · {} {} Conditional.\n\n",
        ops.len(),
        full, Status::Full.glyph(),
        declines, Status::Declines.glyph(),
        notyet, Status::NotYet.glyph(),
        cond, Status::Conditional.glyph(),
    ));
    assert_eq!(
        unaudited, 0,
        "the op × backend matrix must classify every cell — `Unaudited` belongs to the frontier          matrix, which is still being filled in"
    );

    // One table per family, in first-seen order.
    let mut families: Vec<&'static str> = Vec::new();
    for op in &ops {
        if !families.contains(&op.family) {
            families.push(op.family);
        }
    }

    for fam in families {
        s.push_str(&format!("## {fam}\n\n"));
        s.push_str(
            "| op | temen-tree-walk | temen-bytecode | temen-jit | temen-wasm-jit | notes |\n",
        );
        s.push_str("|----|:----:|:----:|:----:|:----:|-------|\n");
        for op in ops.iter().filter(|o| o.family == fam) {
            let cells = op.cells();
            // Collect distinct non-empty notes from the JIT columns (the interpreters have none).
            let mut notes: Vec<&str> = Vec::new();
            for c in &cells {
                if !c.note.is_empty() && !notes.contains(&c.note) {
                    notes.push(c.note);
                }
            }
            s.push_str(&format!(
                "| `{}` | {} | {} | {} | {} | {} |\n",
                op.mnemonic,
                cells[Backend::TreeWalk as usize].status.glyph(),
                cells[Backend::Bytecode as usize].status.glyph(),
                cells[Backend::Cranelift as usize].status.glyph(),
                cells[Backend::WasmJit as usize].status.glyph(),
                notes.join("; "),
            ));
        }
        s.push('\n');
    }

    s
}

/// Render the same matrix as JSON, for the playground's parity page (#1418).
///
/// A **second rendering of the one manifest**, not a second source of truth (INVARIANTS #15): both
/// views call `catalog()` + `Op::cells()`, so a classification can never differ between the markdown
/// and the page, and the golden test pins both against the generator.
///
/// Hand-rolled rather than pulled through `serde`: `temen-parity`'s only dependency is `temen-ir`
/// (the classification must not be able to depend on anything else), and the shape here is four
/// scalar fields — a dependency would cost more than it saves.
pub fn render_json() -> String {
    let ops = catalog();
    let mut s = String::new();

    s.push_str("{\n");
    s.push_str("  \"_generated\": \"do not edit by hand — `cargo run -p temen-parity`\",\n");
    s.push_str("  \"backends\": [");
    for (i, b) in Backend::ALL.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push_str(&format!("\"{}\"", b.short()));
    }
    s.push_str("],\n");
    s.push_str("  \"statuses\": {\n");
    // One list, so adding a status cannot leave the JSON map and the trailing-comma logic out of
    // step (the index was hard-coded to 3 before `Unaudited` existed).
    const ALL_STATUSES: [Status; 5] = [
        Status::Full,
        Status::Declines,
        Status::NotYet,
        Status::Conditional,
        Status::Unaudited,
    ];
    for (i, st) in ALL_STATUSES.iter().enumerate() {
        s.push_str(&format!(
            "    \"{}\": {{ \"glyph\": \"{}\", \"label\": \"{}\" }}{}\n",
            st.id(),
            st.glyph(),
            match st {
                Status::Full => "Full",
                Status::Declines => "Declines (parity not expected)",
                Status::NotYet => "Not yet (parity not achieved)",
                Status::Conditional => "Conditional",
                Status::Unaudited => "Unaudited (nobody has established this cell)",
            },
            if i + 1 == ALL_STATUSES.len() { "" } else { "," }
        ));
    }
    s.push_str("  },\n");
    s.push_str("  \"ops\": [\n");
    for (i, op) in ops.iter().enumerate() {
        let cells = op.cells();
        s.push_str("    { \"op\": ");
        json_str(&mut s, &op.mnemonic);
        s.push_str(", \"family\": ");
        json_str(&mut s, op.family);
        s.push_str(", \"cells\": [");
        for (j, c) in cells.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("\"{}\"", c.status.id()));
        }
        s.push_str("], \"notes\": [");
        // Distinct, order-preserving — the same set the markdown's `notes` column joins.
        let mut notes: Vec<&str> = Vec::new();
        for c in &cells {
            if !c.note.is_empty() && !notes.contains(&c.note) {
                notes.push(c.note);
            }
        }
        for (j, n) in notes.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            json_str(&mut s, n);
        }
        s.push_str("] }");
        s.push_str(if i + 1 == ops.len() { "\n" } else { ",\n" });
    }
    s.push_str("  ]\n}\n");
    s
}

/// Append `v` as a JSON string literal, escaping what RFC 8259 requires. The catalog's mnemonics and
/// notes are ASCII prose today, but escaping is not optional for a format a browser parses: one
/// stray quote or backslash in a future note would silently produce an unparseable page.
fn json_str(out: &mut String, v: &str) {
    out.push('"');
    for ch in v.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Render the capability × axis frontier matrix (`FRONTIER.md`) — INVARIANTS #14's other six axes.
///
/// Same two-renderings-one-manifest shape as the op matrix: this reads
/// [`crate::frontier::capability_axes`] and owns no classification of its own.
pub fn render_frontier_markdown() -> String {
    use crate::frontier::{capability_axes, Axis, Capability};
    let mut s = String::new();

    s.push_str("# Capability × axis frontier matrix\n\n");
    s.push_str(
        "**Generated — do not edit by hand.** Regenerate with `cargo run -p temen-parity`. The \
         classification lives in `crates/temen-parity/src/frontier.rs`; this file is its \
         human-readable view.\n\n",
    );
    s.push_str(
        "INVARIANTS.md #14 says an accepted capability must hold across **seven axes**. \
         `OPS_PARITY.md` machine-checks one of them at op granularity; this matrix is the machine \
         for the rest (#1413). Rows are powerbox capability kinds; columns are the seven axes.\n\n",
    );

    // Coverage first: the honest headline is how much of this matrix is actually known.
    let (mut audited, mut total) = (0usize, 0usize);
    for c in Capability::ALL {
        for cell in capability_axes(c) {
            total += 1;
            if cell.status != Status::Unaudited {
                audited += 1;
            }
        }
    }
    // Derived, not stated: the count of conformed columns tracks `Axis::is_conformed`, so filling
    // an axis in cannot leave this sentence claiming the old number (INVARIANTS #15 — one place).
    let conformed: Vec<&str> = Axis::ALL
        .iter()
        .filter(|a| a.is_conformed())
        .map(|a| a.short())
        .collect();
    s.push_str(&format!(
        "**{audited} of {total} cells audited** ({} capabilities × {} axes). An {} cell is not a \
         passing cell — it means nobody has established what it is. {} of the seven columns ({}) \
         are checked against live predicates by the conformance tests in `crates/temen-parity/\
         tests/`; the rest state the manifest's belief and nothing more.\n\n",
        Capability::ALL.len(),
        Axis::ALL.len(),
        Status::Unaudited.glyph(),
        conformed.len(),
        conformed.join(", "),
    ));

    s.push_str("## Legend\n\n");
    for (st, label) in [
        (Status::Full, "**Full** — the capability holds on this axis"),
        (
            Status::Declines,
            "**Declines** — it deliberately does not, and the note says why",
        ),
        (
            Status::NotYet,
            "**Not yet** — a real gap with a tracked plan",
        ),
        (
            Status::Conditional,
            "**Conditional** — holds where the note's condition does",
        ),
        (
            Status::Unaudited,
            "**Unaudited** — nobody has established this cell",
        ),
    ] {
        s.push_str(&format!("- {} {label}\n", st.glyph()));
    }
    s.push('\n');

    s.push_str("## Axes\n\n");
    for a in Axis::ALL {
        s.push_str(&format!(
            "- **{}** — {}{}\n",
            a.short(),
            a.question(),
            if a.is_conformed() {
                " *(conformance-tested)*"
            } else {
                ""
            },
        ));
    }
    s.push('\n');

    s.push_str("## Matrix\n\n| capability |");
    for a in Axis::ALL {
        s.push_str(&format!(" {} |", a.short()));
    }
    s.push_str("\n|----|");
    for _ in Axis::ALL {
        s.push_str(":----:|");
    }
    s.push('\n');
    for c in Capability::ALL {
        s.push_str(&format!("| `{}` |", c.name()));
        for cell in capability_axes(c) {
            s.push_str(&format!(" {} |", cell.status.glyph()));
        }
        s.push('\n');
    }
    s.push('\n');

    // Notes, once each, under the row they belong to — the table stays scannable and the reasoning
    // stays attached.
    s.push_str("## Notes\n\n");
    for c in Capability::ALL {
        let cells = capability_axes(c);
        let mut any = false;
        for (a, cell) in Axis::ALL.iter().zip(cells.iter()) {
            if cell.note.is_empty() {
                continue;
            }
            if !any {
                s.push_str(&format!("**`{}`**\n", c.name()));
                any = true;
            }
            s.push_str(&format!(
                "- *{}* {} — {}\n",
                a.short(),
                cell.status.glyph(),
                cell.note
            ));
        }
        if any {
            s.push('\n');
        }
    }
    s
}
