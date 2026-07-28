//! Render the catalog + manifest as the human-readable `OPS_PARITY.md` matrix.

use crate::{catalog, Backend, Status};

/// Render the whole matrix as GitHub-flavored markdown. Deterministic (stable order) so the golden
/// test can byte-compare it against the checked-in file.
pub fn render_markdown() -> String {
    let ops = catalog();
    let mut s = String::new();

    s.push_str("# Op × backend parity matrix\n\n");
    s.push_str(
        "**Generated — do not edit by hand.** Regenerate with `cargo run -p svm-parity` after \
         changing the manifest (`crates/svm-parity/src/`). This file is the human-readable view of \
         the exhaustive, test-checked classifier in `svm-parity`; the conformance test \
         (`crates/svm-parity/tests/conformance.rs`) pins every non-skipped row against what the \
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
    for op in &ops {
        for b in [Backend::Cranelift, Backend::WasmJit] {
            match op.cells()[b as usize].status {
                Status::Full => full += 1,
                Status::Declines => declines += 1,
                Status::NotYet => notyet += 1,
                Status::Conditional => cond += 1,
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

    // One table per family, in first-seen order.
    let mut families: Vec<&'static str> = Vec::new();
    for op in &ops {
        if !families.contains(&op.family) {
            families.push(op.family);
        }
    }

    for fam in families {
        s.push_str(&format!("## {fam}\n\n"));
        s.push_str("| op | svm-tree-walk | svm-bytecode | svm-jit | svm-wasm-jit | notes |\n");
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
