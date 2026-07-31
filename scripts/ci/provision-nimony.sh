#!/usr/bin/env bash
# Build the nimony toolchain (nifler → nimony → hexer → lengc …) for the Nim-source end-to-end
# tests (`crates/svm-leng/tests/nim_e2e.rs`). Mirrors nim-lang/nimony's own CI build.
#
# Prerequisites: a Nim `devel` compiler on PATH (the caller installs it — e.g. the CI job uses the
# `setup-nim` action). Produces the tools under `<workdir>/nimony/bin` and prints two `KEY=value`
# lines the caller `eval`s / appends to $GITHUB_ENV:
#     NIMONY_BIN=<abs>/nimony/bin
#     NIM_BIN=<dir of the nim on PATH>
#
# NIMONY_REF pins the nimony commit so the emitted Leng (symbol mangling, ABI) is reproducible — bump
# it deliberately, in lockstep with any svm-leng change the newer frontend would require.
set -euo pipefail

# The caller appends our stdout to $GITHUB_ENV, so ONLY the final `KEY=value` lines may go there —
# send every command's chatter (clone progress, the `hastur` build) to stderr, and emit the two
# result lines on the saved stdout (fd 3) at the end.
exec 3>&1 1>&2

# Full 40-char commit SHA (GitHub's fetch/checkout-by-SHA needs the full hash, not an abbreviation).
NIMONY_REF="${NIMONY_REF:-096de9a3dd479c5865f04cd3d595dfdf42e83ef9}"
WORK="${1:-${GITHUB_WORKSPACE:-$PWD}}"
cd "$WORK"

command -v nim >/dev/null || { echo "error: nim (devel) not on PATH" >&2; exit 1; }
NIM_BIN="$(dirname "$(command -v nim)")"

# nimony + its sibling nativenif (the native backend's nim.cfg reach it via `../nativenif`). We need
# master (blob-filtered — all commits, lazy blobs) so the pinned commit is present, then check it out.
# The toolchain cache restores `nimony/bin` (built output) WITHOUT `nimony/.git`, so on a warm cache
# `nimony/` exists but is not a checkout — and `git clone nimony` aborts on the non-empty dir. Init a
# repo in place and fetch instead, which works whether `nimony/` is absent (cold) or holds a restored
# `bin/` (warm); fetching all branches mirrors the old clone's reachability so the pinned SHA is found.
if [ ! -d nimony/.git ]; then
  mkdir -p nimony
  git -C nimony init -q
  git -C nimony remote add origin https://github.com/nim-lang/nimony 2>/dev/null \
    || git -C nimony remote set-url origin https://github.com/nim-lang/nimony
  git -C nimony fetch --filter=blob:none origin
fi
git -C nimony checkout --detach "$NIMONY_REF"
[ -d nativenif/.git ] || git clone --depth 1 https://github.com/nim-lang/nativenif nativenif

# setup-nim installs a *prebuilt* nightly that can lag `devel`; overlay fresh compiler sources so
# nifler (which compiles Nim's own parser) can parse current syntax — exactly what nimony CI does.
if [ ! -d nim-src ]; then
  git clone --depth 1 --branch devel https://github.com/nim-lang/Nim nim-src
fi
# The Nim install directory is the parent of its bin/.
NIM_ROOT="$(dirname "$NIM_BIN")"
if [ -d "$NIM_ROOT/compiler" ]; then
  cp -a nim-src/compiler/. "$NIM_ROOT/compiler/"
fi
# hastur resolves the frontend's NIF libs via `nim/dist/nimony` — point it at our checkout.
mkdir -p "$NIM_ROOT/dist"
rm -rf "$NIM_ROOT/dist/nimony"
ln -sf "$WORK/nimony" "$NIM_ROOT/dist/nimony"

# Build all the tools (C backend; the E2E harness invokes `nimony c`).
( cd nimony && nim c -r src/hastur --release build all )

echo "NIMONY_BIN=$WORK/nimony/bin" >&3
echo "NIM_BIN=$NIM_BIN" >&3
