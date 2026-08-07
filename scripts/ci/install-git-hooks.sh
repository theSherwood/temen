#!/usr/bin/env bash
# Point this clone at the shared, version-controlled hooks in .githooks/.
# Opt-in and reversible: `git config --unset core.hooksPath` turns it back off.
# See .githooks/README.md.
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

git config core.hooksPath .githooks
echo "Enabled .githooks (core.hooksPath=.githooks)."
echo "The pre-push hook now runs the fast CI subset before each push."
echo "Bypass once with:  SVM_HOOK_SKIP=1 git push   (or git push --no-verify)"
