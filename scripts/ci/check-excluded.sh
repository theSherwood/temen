#!/usr/bin/env bash
# Build the crates the workspace **excludes** — the ones `cargo …  --workspace` silently skips.
#
# Why this exists: the root `Cargo.toml` carries an `exclude = [...]` list (crates that need their
# own toolchain, feature set, or heavy deps, so they are kept out of the default build). Everything
# else in the repo — the pre-push hook, a contributor's `cargo check --workspace`, and most of the CI
# matrix — reasons in terms of the workspace, so those crates are invisible to all of it. A change
# to a shared signature compiles clean everywhere and breaks them silently.
#
# That is not hypothetical: `temen-webgpu` did not compile from 2026-08-26 (#1116, when `HostProc`
# gained its `RegionMinter` parameter) until it was noticed three weeks later, because no job built
# it. A `temen-llvm` break reached CI the same way.
#
# The exclude list is read from `Cargo.toml` rather than restated here, so a crate added to (or
# removed from) it is covered without anyone remembering to edit this script.
#
# Toolchains: `TEMEN_CHECK_TOOLCHAIN` pins every crate, and `TEMEN_CHECK_TOOLCHAIN_<dir>` — the
# directory with every non-alphanumeric character replaced by `_` — overrides one of them. Some
# excluded crates have their own CI job pinned to a channel (`fuzz` builds on a dated nightly), so
# pinning here makes this check compile what that job compiles:
#
#   TEMEN_CHECK_TOOLCHAIN=1.97.0 TEMEN_CHECK_TOOLCHAIN_fuzz=nightly-2026-07-01 \
#     scripts/ci/check-excluded.sh
#
# Both unset means the default toolchain, which is what you want locally. (CI sets both, because a
# job that installs two toolchains leaves the *last* one as the default — not necessarily the one
# the stable crates should build with.)
#
# Usage:  scripts/ci/check-excluded.sh [extra cargo args…]
#   e.g.  scripts/ci/check-excluded.sh --locked
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

# Pull the `exclude = [ … ]` array out of the root manifest. Handles the array on one line or
# spread over several; ignores comments.
mapfile -t excluded < <(
  awk '
    /^[[:space:]]*exclude[[:space:]]*=/ { inarr = 1 }
    inarr {
      line = $0
      sub(/#.*/, "", line)
      n = split(line, parts, /"/)
      for (i = 2; i <= n; i += 2) print parts[i]
      if (line ~ /\]/) exit
    }
  ' Cargo.toml
)

if [ "${#excluded[@]}" -eq 0 ]; then
  echo "check-excluded: no 'exclude' entries found in Cargo.toml — nothing to do." >&2
  exit 0
fi

# Match CI's posture unless the caller already chose one.
export RUSTFLAGS="${RUSTFLAGS:--D warnings}"

echo "check-excluded: ${#excluded[@]} crate(s) outside the workspace: ${excluded[*]}"
rc=0
for dir in "${excluded[@]}"; do
  if [ ! -f "$dir/Cargo.toml" ]; then
    echo "check-excluded: SKIP $dir (no Cargo.toml)"
    continue
  fi
  var="TEMEN_CHECK_TOOLCHAIN_$(printf '%s' "$dir" | sed 's/[^[:alnum:]]/_/g')"
  toolchain="${!var:-${TEMEN_CHECK_TOOLCHAIN:-}}"
  # Each excluded crate is its own workspace root, so `--manifest-path` builds it standalone —
  # the same way CI's per-crate jobs (and a contributor `cd`-ing into it) would.
  cmd=(cargo)
  [ -n "$toolchain" ] && cmd+=("+$toolchain")
  cmd+=(check --all-targets --manifest-path "$dir/Cargo.toml" "$@")
  echo "check-excluded: \$ ${cmd[*]}"
  if ! "${cmd[@]}"; then
    echo "check-excluded: FAILED $dir" >&2
    rc=1
  fi
done

if [ "$rc" -ne 0 ]; then
  echo "check-excluded: at least one excluded crate does not build." >&2
  exit "$rc"
fi
echo "check-excluded: OK — every excluded crate builds."
