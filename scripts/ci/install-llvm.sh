#!/usr/bin/env bash
# Install the pinned LLVM/clang toolchain on an Ubuntu CI runner (out-of-process build tools only:
# `clang`, `llvm-dis`, `llvm-link`, `opt`, `llvm-as`, `llvm-nm` — the on-ramp reads textual `.ll` with
# an in-house parser and links no libLLVM). Extra apt packages a job needs go on the command line:
#   bash scripts/ci/install-llvm.sh [extra apt packages…]
#   bash scripts/ci/install-llvm.sh --fetch   # only make sure the pinned packages are here
# CI runs `--fetch` (`.github/actions/install-llvm`), then installs the packages through
# `.github/actions/apt-install`, whose cache holds what Ubuntu's archive serves for them (#2947).
#
# THE ONE PLACE THE LLVM VERSION LIVES. It must equal the LLVM major of the pinned stable `rustc`
# (`rustc -vV | grep LLVM`; `RUST_STABLE` in ci.yml): the `peval_*` probes emit Rust IR with the default
# `rustc` and feed it to `llvm-link`/`opt`, which can only ingest IR of their own version or older —
# `ci_tool_canary` asserts the two majors agree. Bump both together when rustc moves.
# Ubuntu's own archive stops at clang 18/19, so the toolchain comes from apt.llvm.org.
set -euo pipefail
LLVM_MAJOR=22

# The LLVM packages live in `~/.cache/temen-ci/llvm/<codename>`, which CI restores with `actions/cache`
# keyed on the codename and this script (`.github/actions/install-llvm`), so only a run that finds the
# cache empty talks to apt.llvm.org. Installing from apt.llvm.org on every run was a flake: a runner
# that could not resolve the host went red before a line was built (main, 2026-09-25). A filled cache
# also pins the build: every job installs the exact packages apt.llvm.org served when the cache was
# filled, until this script changes. Only the packages built from `llvm-toolchain-$LLVM_MAJOR` are
# kept. What they need from Ubuntu's archive is `apt-prep.sh`'s to cache, per runner image, so a newer
# image is never asked to downgrade one of its own libraries. Until #2947 the key named no codename:
# the `noble` and `resolute` jobs shared one entry, and the codename that did not fill it fetched
# LLVM from apt.llvm.org on every run.
codename=$(. /etc/os-release && echo "$VERSION_CODENAME")
cache="$HOME/.cache/temen-ci/llvm/$codename"
if ! compgen -G "$cache/*.deb" >/dev/null; then
  # Scrub only: the update below needs it, and the LLVM repo is added after it.
  bash "$(dirname "$0")/apt-prep.sh"
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/llvm.gpg --yes
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] http://apt.llvm.org/$codename/ llvm-toolchain-$codename-$LLVM_MAJOR main" \
    | sudo tee /etc/apt/sources.list.d/llvm.list >/dev/null
  sudo apt-get update
  sudo apt-get install -y --download-only "llvm-$LLVM_MAJOR" "clang-$LLVM_MAJOR"
  mkdir -p "$cache"
  for deb in /var/cache/apt/archives/*.deb; do
    if [ "$(dpkg-deb -f "$deb" Source | cut -d' ' -f1)" = "llvm-toolchain-$LLVM_MAJOR" ]; then
      cp "$deb" "$cache/"
    fi
  done
  # apt downloads nothing it has installed already: a cache filled on such a host would be empty.
  if ! compgen -G "$cache/*.deb" >/dev/null; then
    echo "install-llvm: apt.llvm.org served no llvm-toolchain-$LLVM_MAJOR packages (installed already?)" >&2
    exit 1
  fi
fi
# Unversioned tool names (`clang`, `llvm-dis`, …) resolve to the pinned version for the rest of the job.
if [ -n "${GITHUB_PATH:-}" ]; then echo "/usr/lib/llvm-$LLVM_MAJOR/bin" >> "$GITHUB_PATH"; fi
if [ "${1:-}" = --fetch ]; then
  if [ -n "${GITHUB_OUTPUT:-}" ]; then echo "debs=$cache" >> "$GITHUB_OUTPUT"; fi
  exit 0
fi
bash "$(dirname "$0")/apt-prep.sh" "$cache"/*.deb "$@"
