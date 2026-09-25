#!/usr/bin/env bash
# Install the pinned LLVM/clang toolchain on an Ubuntu CI runner (out-of-process build tools only:
# `clang`, `llvm-dis`, `llvm-link`, `opt`, `llvm-as`, `llvm-nm` — the on-ramp reads textual `.ll` with
# an in-house parser and links no libLLVM). Extra apt packages a job needs go on the command line:
#   bash scripts/ci/install-llvm.sh [extra apt packages…]
#
# THE ONE PLACE THE LLVM VERSION LIVES. It must equal the LLVM major of the pinned stable `rustc`
# (`rustc -vV | grep LLVM`; `RUST_STABLE` in ci.yml): the `peval_*` probes emit Rust IR with the default
# `rustc` and feed it to `llvm-link`/`opt`, which can only ingest IR of their own version or older —
# `ci_tool_canary` asserts the two majors agree. Bump both together when rustc moves.
# Ubuntu's own archive stops at clang 18/19, so the toolchain comes from apt.llvm.org.
set -euo pipefail
LLVM_MAJOR=22

# Drop the runner's unused third-party apt sources so a transient outage or publish window on one of
# those mirrors can't fail `apt-get update` before we install anything (I67/#1017, #1374). Scrub
# only: this script runs its own update and install below.
bash "$(dirname "$0")/apt-prep.sh"

# The LLVM packages live in `~/.cache/temen-ci/llvm`, which CI restores with `actions/cache` keyed
# on this script (`.github/actions/install-llvm`), so only a run that finds the cache empty talks to
# apt.llvm.org. Installing from apt.llvm.org on every run was a flake: a runner that could not
# resolve the host went red before a line was built (main, 2026-09-25). A filled cache also pins the
# build: every job installs the exact packages apt.llvm.org served when the cache was filled, until
# this script changes. Only the packages built from `llvm-toolchain-$LLVM_MAJOR` are kept. Their
# Ubuntu dependencies still come from Ubuntu's archive, so a newer runner image is never asked to
# downgrade one of its own libraries. The codename in the path makes a new runner OS fill afresh.
codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
cache="$HOME/.cache/temen-ci/llvm/$codename"
if ! compgen -G "$cache/*.deb" >/dev/null; then
  fill=1
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/llvm.gpg --yes
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/$codename/ llvm-toolchain-$codename-$LLVM_MAJOR main" \
    | sudo tee /etc/apt/sources.list.d/llvm.list >/dev/null
fi
sudo apt-get update
if [ -n "${fill:-}" ]; then
  sudo apt-get install -y --download-only "llvm-$LLVM_MAJOR" "clang-$LLVM_MAJOR"
  mkdir -p "$cache"
  for deb in /var/cache/apt/archives/*.deb; do
    if [ "$(dpkg-deb -f "$deb" Source | cut -d' ' -f1)" = "llvm-toolchain-$LLVM_MAJOR" ]; then
      cp "$deb" "$cache/"
    fi
  done
fi
sudo apt-get install -y "$cache"/*.deb "$@"
# Unversioned tool names (`clang`, `llvm-dis`, …) resolve to the pinned version for the rest of the job.
if [ -n "${GITHUB_PATH:-}" ]; then echo "/usr/lib/llvm-$LLVM_MAJOR/bin" >> "$GITHUB_PATH"; fi
