#!/usr/bin/env bash
# Build the nimony toolchain (nifler → nimony → hexer → lengc …) for the Nim-source end-to-end
# tests (`crates/temen-leng/tests/nim_e2e.rs`). Mirrors nim-lang/nimony's own CI build.
#
# Prerequisites: a Nim `devel` compiler on PATH (the caller installs it — e.g. the CI job uses the
# `setup-nim` action). Produces the tools under `<workdir>/nimony/bin` and prints two `KEY=value`
# lines the caller `eval`s / appends to $GITHUB_ENV:
#     NIMONY_BIN=<abs>/nimony/bin
#     NIM_BIN=<dir of the nim on PATH>
#
# The nimony frontend (and its sibling `nativenif`) are vendored as **git submodules** — the exact
# commit is pinned by the gitlink in `.gitmodules`, reproducible in-tree, and bumped deliberately in
# lockstep with any temen-leng change the newer frontend requires (`git -C nimony checkout <ref>` +
# commit the submodule). This replaces the old in-script clone-and-checkout of a hard-coded SHA.
set -euo pipefail

# The caller appends our stdout to $GITHUB_ENV, so ONLY the final `KEY=value` lines may go there —
# send every command's chatter (submodule fetch, the `hastur` build) to stderr, and emit the two
# result lines on the saved stdout (fd 3) at the end.
exec 3>&1 1>&2

WORK="${1:-${GITHUB_WORKSPACE:-$PWD}}"
cd "$WORK"

command -v nim >/dev/null || { echo "error: nim (devel) not on PATH" >&2; exit 1; }
NIM_BIN="$(dirname "$(command -v nim)")"

# Ensure the vendored submodules are checked out at their pinned commits. A CI checkout with
# `submodules: recursive` already does this, so this is a no-op there; it makes a plain checkout or a
# local run work too. Not shallow — the pinned commit need not be a branch tip, so the full fetch is
# required to resolve it. `nativenif` must sit beside `nimony` (the native backend's nim.cfg reaches
# it via `../nativenif`); both are repo-root submodules, so that sibling layout holds.
git submodule update --init nimony nativenif

# setup-nim installs the *prebuilt* devel nightly, cut daily and lagging devel's head by hours to
# a day (or months when nightlies stall). nifler compiles Nim's own parser, so it needs current
# compiler sources; nimony's CI overlays devel HEAD's `compiler/` onto the nightly for that. An
# unpinned HEAD over an older nightly splits the sources in two — Nim#26139 changed `docgen.nim`
# and `rstgen.nim` together, and the nightly's `lib/` no longer matched the overlaid `compiler/`
# (#1220). So: pin the source commit (bump deliberately, like the nimony submodule) and overlay
# both `compiler/` and `lib/` from it — one coherent source tree, the nightly only the bootstrap
# binary (Nim's own bootstrap compiles devel sources with an older binary the same way).
NIM_SRC_REV=973065b279d2ae5b3954c25348c7dc4a02335f2b # nim-lang/Nim devel, 2026-09-03
if [ ! -d nim-src/.git ]; then
  git init -q nim-src
  git -C nim-src remote add origin https://github.com/nim-lang/Nim
fi
git -C nim-src fetch -q --depth 1 origin "$NIM_SRC_REV"
git -C nim-src checkout -q FETCH_HEAD
# The Nim install directory is the parent of its bin/.
NIM_ROOT="$(dirname "$NIM_BIN")"
for d in compiler lib; do
  if [ -d "$NIM_ROOT/$d" ]; then
    cp -a "nim-src/$d/." "$NIM_ROOT/$d/"
  fi
done
# hastur resolves the frontend's NIF libs via `nim/dist/nimony` — point it at our submodule checkout.
mkdir -p "$NIM_ROOT/dist"
rm -rf "$NIM_ROOT/dist/nimony"
ln -sf "$WORK/nimony" "$NIM_ROOT/dist/nimony"

# Build all the tools (C backend; the E2E harness invokes `nimony c`).
( cd nimony && nim c -r src/hastur --release build all )

echo "NIMONY_BIN=$WORK/nimony/bin" >&3
echo "NIM_BIN=$NIM_BIN" >&3
