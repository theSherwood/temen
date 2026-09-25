#!/usr/bin/env bash
# Check out Embench-IoT at the pinned commit and export `EMBENCH=<checkout>`: appended to $GITHUB_ENV
# on a runner, printed otherwise (`export "$(bash scripts/ci/fetch-embench.sh)"`). Embench is not
# vendored because its kernels carry mixed licenses.
#
# THE ONE PLACE THE EMBENCH REVISION LIVES. The checkout sits in `~/.cache/temen-ci/embench`, which
# CI restores with `actions/cache` keyed on this script, so only a run that finds the cache empty
# fetches. Downloading `master` from codeload on every run was debt (#1840): codeload sometimes
# served an error page instead of the tarball (ISSUES.md I18 class 4), `curl --retry` only hid that,
# and `master` could move under the differential with no change here. The commit id pins the tree:
# git checks what it fetches against it, and the check below reads it again on every run.
set -euo pipefail
EMBENCH_REV=09c2ed8c3b7008c95d08b038de4a3f6dc103ed70 # embench/embench-iot master, 2026-02-13

dir="$HOME/.cache/temen-ci/embench"
if [ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" != "$EMBENCH_REV" ] ||
  [ -n "$(git -C "$dir" status --porcelain)" ]; then
  rm -rf "$dir"
  git init -q "$dir"
  git -C "$dir" fetch -q --depth 1 https://github.com/embench/embench-iot "$EMBENCH_REV"
  git -C "$dir" checkout -q FETCH_HEAD
fi
echo "EMBENCH=$dir" >> "${GITHUB_ENV:-/dev/stdout}"
