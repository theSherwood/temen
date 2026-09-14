#!/usr/bin/env bash
# Prepare the runner's apt for an install, then (optionally) install packages:
#   bash scripts/ci/apt-prep.sh [package…]
# With packages: scrub, `apt-get update`, install them. With none: scrub only — for a step whose
# installer runs its own update (`playwright install --with-deps`) or that adds a repo first
# (`install-llvm.sh`).
#
# THE ONE PLACE THE SCRUB LIVES. `apt-get update` fails hard when *any* configured index is
# inconsistent, and the `ubuntu-latest` image preconfigures third-party repos we never install from —
# so a publish window or an outage on someone else's mirror reds the whole matrix before a line of our
# code runs. Twice now (#1017: the azure mirror; #1374: `dl.google.com` serving a `Packages.gz` that
# did not match its own Release file for ~1 h, taking 7 jobs down at their apt step). This incantation
# used to be copy-pasted into five workflow steps plus `install-llvm.sh`, so #1374's one-word fix
# needed six edits — INVARIANTS #15: one path per behaviour.
set -euo pipefail

# Drop every third-party source the image ships. We install only from Ubuntu's own archive (plus the
# LLVM repo `install-llvm.sh` adds *after* this runs), so none of these can be needed — and each one
# is a mirror whose bad hour becomes our red matrix.
sudo rm -f /etc/apt/sources.list.d/microsoft* \
  /etc/apt/sources.list.d/azure* \
  /etc/apt/sources.list.d/google-chrome*
# The azure mirror also served stale/500s as the *default* archive; point the default at the canonical
# one. Best-effort: the files differ across image revisions, so a miss here is not an error.
sudo sed -i 's|http://azure.archive.ubuntu.com/ubuntu|https://archive.ubuntu.com/ubuntu|g' \
  /etc/apt/apt-mirrors.txt /etc/apt/sources.list.d/*.sources 2>/dev/null || true

if [ "$#" -gt 0 ]; then
  sudo apt-get update
  sudo apt-get install -y "$@"
fi
