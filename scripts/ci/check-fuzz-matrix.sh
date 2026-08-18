#!/usr/bin/env bash
# Fuzz-target lockstep gate (#923). Three things must name the exact same set of fuzz targets:
#
#   1. fuzz/fuzz_targets/*.rs   — the target source files (cargo-fuzz's unit of coverage)
#   2. fuzz/Cargo.toml [[bin]]  — the entries that actually build a target
#   3. .github/workflows/ci.yml — the `fuzz` job's matrix (what nightly actually runs)
#
# A file with no [[bin]] never builds; a target with no matrix row builds but never runs — zero
# coverage, silently (the exact failure ci.yml's fuzz job warns about). This check fails CI the
# moment the three drift, the same way `workflows-in-sync` gates the workflow mirror.
#
# Runnable locally: `scripts/ci/check-fuzz-matrix.sh`.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

# 1. Basenames of the target source files.
for f in fuzz/fuzz_targets/*.rs; do
  basename "$f" .rs
done | sort >"$tmp/files"

# 2. `name = "..."` under each [[bin]] in fuzz/Cargo.toml (not the [package] name).
awk '
  /^\[\[bin\]\]/ { in_bin = 1; next }
  /^\[/          { in_bin = 0 }
  in_bin && /^name = "/ {
    line = $0
    sub(/^name = "/, "", line)
    sub(/".*$/, "", line)
    print line
  }
' fuzz/Cargo.toml | sort >"$tmp/bins"

# 3. Matrix target identifiers: the bracketed list under the fuzz job's `target:` key. Isolate the
#    fuzz job, then the `target: [ ... ]` block, then strip to bare identifiers.
awk '
  /^  fuzz:/            { in_fuzz = 1 }
  in_fuzz && /^  [a-z]/ && !/^  fuzz:/ { in_fuzz = 0 }   # next top-level job ends the fuzz block
  in_fuzz && /target:/  { in_list = 1; next }
  in_fuzz && in_list {
    if ($0 ~ /\]/) in_list = 0
    gsub(/[][, ]/, "")
    if ($0 != "") print
  }
' .github/workflows/ci.yml | sort >"$tmp/matrix"

fail=0
report() {
  # $1 = human label for set A, $2 = file A, $3 = label B, $4 = file B
  local only_a only_b
  only_a="$(comm -23 "$2" "$4")"
  only_b="$(comm -13 "$2" "$4")"
  if [ -n "$only_a" ]; then
    echo "::error::fuzz targets in $1 but not $3:"; echo "$only_a" | sed 's/^/  - /'
    fail=1
  fi
  if [ -n "$only_b" ]; then
    echo "::error::fuzz targets in $3 but not $1:"; echo "$only_b" | sed 's/^/  - /'
    fail=1
  fi
}

report "fuzz_targets/*.rs" "$tmp/files" "fuzz/Cargo.toml [[bin]]" "$tmp/bins"
report "fuzz_targets/*.rs" "$tmp/files" "ci.yml fuzz matrix" "$tmp/matrix"

if [ "$fail" -eq 0 ]; then
  echo "fuzz targets in lockstep ($(wc -l <"$tmp/files" | tr -d ' ') targets: files == [[bin]] == matrix)."
fi
exit "$fail"
