#!/usr/bin/env bash
# Drive the SVM -> Temen rename: rewrite file *contents*, then `git mv` every path
# whose name changes (crate dirs, the rust-svm custom-target dir, headers, and the
# .svmb/.svmt artifacts) using the SAME rules, so names and references stay in sync.
#
# Re-runnable and safe to inspect. Does NOT touch the wire-format magic bytes (see
# rename-to-temen.py) and does NOT touch .github/workflows/ (unpushable without the
# `workflow` scope — edit the .github/workflows_src/ mirror, which this handles, and
# have the owner copy it over).
set -euo pipefail
cd "$(dirname "$0")/.."

echo ">> rewriting file contents"
python3 scripts/rename-to-temen.py

echo ">> renaming paths (git mv)"
# Snapshot the file list first; git mv mutates the index as we go.
git ls-files > /tmp/temen_files.txt
while IFS= read -r f; do
  [ -f "$f" ] || continue
  nf="$(python3 scripts/rename-to-temen.py --path "$f")"
  if [ "$nf" != "$f" ]; then
    mkdir -p "$(dirname "$nf")"
    git mv "$f" "$nf"
  fi
done < /tmp/temen_files.txt
rm -f /tmp/temen_files.txt

# Drop now-empty directories left behind by the moves (never touch .git).
find . -type d -empty -not -path './.git/*' -delete 2>/dev/null || true

echo ">> done"
