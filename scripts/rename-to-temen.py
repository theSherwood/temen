#!/usr/bin/env python3
"""
One-shot, re-runnable rename of the project from "SVM" / "Sandbox VM" to "Temen".

Strategy (order matters — landmines first, then a safe global pass):

  1. PROTECT the wire-format MAGIC byte literals from the global pass. The magic is
     deliberately NOT changed in this rename: dozens of committed binary artifacts
     (.svmb / .svmb.gz fixtures and playground assets) embed b"SVM\\0" in their
     bytes, and the gzipped ones can't be byte-patched — flipping the constant would
     make them all fail to load. So the format keeps SVM for now (changing it is a
     separate PR that must regenerate those artifacts). We swap each magic literal to
     a placeholder before the passes and restore it verbatim after, so a naive
     SVM->TEMEN can neither corrupt the [u8; N] lengths nor desync code from the
     committed binaries.
  2. Rewrite file EXTENSIONS as whole tokens, longest first (.svmb -> .temen,
     .svmt/.svm -> .temt). Done before the global pass so ".svmb" doesn't become
     ".temenb". Binary keeps the "prestige" full name (.temen), text gets .temt —
     mirroring wasm's .wasm/.wat and fixing the old scheme where text held the short
     name.
  3. Rewrite prose ("Sandbox VM" -> "Temen", "Codename: TBD" -> "Codename: Temen").
  4. Global, case-preserving token replace of everything left: svm_->temen_,
     svm-->temen-, SVM->TEMEN, Svm->Temen, svm->temen. After steps 1-2 every
     remaining occurrence is an identifier / path / prose token where lengthening is
     harmless. ("temen" contains no "svm", so there is no double-substitution.)

Then (in the shell wrapper) `git mv` the 27 crate dirs, the .svmb/.svmt files, and
the two svm.h headers.

Idempotent: re-running finds nothing to change. Delete this script after the rename
lands if you like; it lives in-tree only so the rename PR is reviewable.
"""
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- 1. wire-format magic literals to PROTECT (kept verbatim; see module docstring) ---
# These embed "SVM" but must survive untouched so code still matches the committed
# binary artifacts. Each is swapped to a NUL-delimited placeholder for the passes,
# then restored. (NUL never appears in source, so the placeholders are collision-safe.)
MAGIC_PROTECT = [
    r'b"SVM\x00"',    # svm-encode MAGIC: [u8; 4]
    r'b"SVM\0"',      # browser module sniff (must match encode)
    r'b"SVMFSIM1"',   # svm-fs IMAGE_MAGIC: [u8; 8]
    r'b"SVMD"',       # svm-snapshot MAGIC: [u8; 4] (+ its doc comment)
    r'b"\x7fSVM"',    # svm-posix ELF-style executable marker
]

# --- 2. file extensions, longest first, as whole tokens ---
EXT = [
    (re.compile(r'\.svmb\b'), '.temen'),
    (re.compile(r'\.svmt\b'), '.temt'),
    (re.compile(r'\.svm\b'), '.temt'),
]

# --- 3. prose ---
# Only the H1 titles ("# Sandbox VM", "# Sandbox VM — Design Notes") become the name.
# Inline "sandbox VM" elsewhere is an accurate common-noun description of what Temen
# *is* — leave it, so we don't get circular phrasing like "a target and Temen".
PROSE = [
    (re.compile(r'# Sandbox VM'), '# Temen'),
    (re.compile(r'Codename: TBD'), 'Codename: Temen'),
]

# --- 4. global case-preserving token replace (applied in this order) ---
# 'svmb'/'svmt' as bare identifier fragments (NOT the .svmb/.svmt extensions, which
# step 2 already handled) map to the format's new short names so e.g. `svmb_strip` ->
# `temen_strip`, not `temenb_strip`.
GLOBAL = [
    (re.compile(r'svm_'), 'temen_'),
    (re.compile(r'svm-'), 'temen-'),
    (re.compile(r'SVM_'), 'TEMEN_'),
    (re.compile(r'SVM-'), 'TEMEN-'),
    (re.compile(r'svmb'), 'temen'),
    (re.compile(r'svmt'), 'temt'),
    (re.compile(r'SVMB'), 'TEMEN'),
    (re.compile(r'SVMT'), 'TEMT'),
    # Bare acronym "SVM" is always prose/strings here (never a code token — env vars
    # are SVM_, magic bytes are protected, Rust idents are Svm/svm_), so title-case
    # "Temen" reads right ("the Temen IR", "Temen guests"), not a shouty "TEMEN".
    (re.compile(r'\bSVM\b'), 'Temen'),
    (re.compile(r'\bSvm'), 'Temen'),   # CamelCase idents: SvmError -> TemenError
    (re.compile(r'\bsvm\b'), 'temen'),
    (re.compile(r'svm'), 'temen'),     # residual (e.g. bsvm/jsvm local vars)
]


def tracked_text_files():
    out = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True)
    for rel in out.splitlines():
        # Cannot push under .github/workflows/ (needs `workflow` scope). The mirror
        # in .github/workflows_src/ IS editable and is handled by the global pass;
        # skip the live copies so we don't stage an unpushable change.
        if rel.startswith(".github/workflows/"):
            continue
        # This script itself contains the literal patterns; never rewrite it.
        if rel == "scripts/rename-to-temen.py":
            continue
        path = os.path.join(ROOT, rel)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
        except (UnicodeDecodeError, IsADirectoryError):
            continue  # binary / non-text
        yield path, data


def apply(data):
    # Protect magic literals: swap to NUL-delimited placeholders.
    for i, lit in enumerate(MAGIC_PROTECT):
        data = data.replace(lit, f"\x00MAGIC{i}\x00")
    for rx, new in EXT:
        data = rx.sub(new, data)
    for rx, new in PROSE:
        data = rx.sub(new, data)
    for rx, new in GLOBAL:
        data = rx.sub(new, data)
    # Restore magic literals verbatim.
    for i, lit in enumerate(MAGIC_PROTECT):
        data = data.replace(f"\x00MAGIC{i}\x00", lit)
    return data


def main():
    # `--path <p>`: print the renamed path (for the git-mv wrapper). Uses the exact
    # same rules as the content pass, so file names and in-code references stay in sync.
    if len(sys.argv) == 3 and sys.argv[1] == "--path":
        sys.stdout.write(apply(sys.argv[2]))
        return
    changed = 0
    for path, data in tracked_text_files():
        new = apply(data)
        if new != data:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new)
            changed += 1
    print(f"rewrote {changed} files")


if __name__ == "__main__":
    main()
