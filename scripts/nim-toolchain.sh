#!/usr/bin/env bash
# nimony's own toolchain as Temen modules, built with no C compiler (#763): what the self-hosted lane
# (`scripts/ci/nim-selfhost-lane.sh`) builds programs with, and what the playground's nim card ships
# (`scripts/rebuild-assets.sh`, step `nim_card`).
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> bash scripts/nim-toolchain.sh <out>
#
# writes into <out>:
#   nimony/          nimony's tree, laid out as nimony's own is (`bin/ lib/ src/ doc/`), its `bin/`
#                    holding the phases step 1 builds
#   nimony.temen nifmake.temen nifler2.temen nimsem.temen hexer.temen
#                    the tools, through the POSIX edge; nimsem's build in the tree's own nimcache,
#                    each other's in `nimcache-<tool>/`
#   sh.ir            /bin/sh: the POSIX build of demos/shell
#   libc.temeno      the guest libc temen-link links a program against (the committed
#                    `browser/web/assets/pg_libc.temeno`)
#
# Every tool nimony builds from the tree, so a tool nimony builds in-guest from the same tree is one
# this script built: what the lane's fixed point holds nimsem to.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
[ $# -eq 1 ] || { echo "usage: scripts/nim-toolchain.sh <out>" >&2; exit 2; }
mkdir -p "$1"
W="$(cd "$1" && pwd)"

# The tree, from a COPY of the pinned sources (the submodule is never modified), patched: nifmake is a
# classic-Nim-only program upstream, the driver learns the Temen backend, a compiler built for
# Temen builds the programs it runs itself (compile-time evaluation, plugins) for Temen, and a
# program built for Temen starts its processes with Temen's spawn, not a fork. See each patch's own
# preamble. Every build of a tool happens in this tree, and so does the lane's self-build:
# the guest's memfs holds it at the same path, so both record the same paths — nimony records a
# source path relative to its cwd, but writes a compile-time evaluation program's imports and output
# file absolute.
N="$W/nimony"
mkdir -p "$N"
cp -r "$ROOT/nimony/bin" "$ROOT/nimony/lib" "$ROOT/nimony/src" "$ROOT/nimony/doc" "$N/"
for p in "$ROOT"/patches/nimony/*.patch; do (cd "$N" && git apply "$p"); done
cargo build --release -q -p temen-run --example build_nim_hello_temen
B=target/release/examples

echo "[toolchain 1/3] the phases, native: nimony builds nifler2, nimsem and hexer"
# `nimony/bin`'s phases are built by classic Nim — a different compiler, whose hash tables iterate in
# another order (#1753) — so nimony builds each again from the tree, and those replace them: they are
# the reference every artifact is held to, and the compiler of step 2. The driver and nifmake only
# sequence work; theirs stay as they are.
pids=()
for p in nifler2/nifler2 nimony/nimsem hexer/hexer; do
  n="$(basename "$p")"
  (cd "$N" && nimony c --isMain --nimcache:"$W/native-$n" "src/$p.nim" >/dev/null \
    && cp "$(ls "$W/native-$n"/*/"$n" | head -1)" "bin/$n") &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[toolchain 2/3] the tools, through the POSIX edge, built by those phases"
# nimsem builds in the tree's own nimcache, which the lane's self-build compares against; the others
# each get their own, because the builds run at once and would otherwise all write it.
build() { NIMONY_BIN="$N/bin" "$B/build_nim_hello_temen" --posix --root "$N" "$@"; }
pids=()
build src/nimony/nimsem.nim "$W/nimsem.temen" &
pids+=($!)
for p in nifler2/nifler2 hexer/hexer nimony/nimony nifmake/nifmake; do
  n="$(basename "$p")"
  build --nimcache "$W/nimcache-$n" "src/$p.nim" "$W/$n.temen" &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[toolchain 3/3] /bin/sh: the POSIX build of demos/shell"
make -s -C frontend/chibicc
D=crates/temen-run/demos/shell
cat "$D/shim.c" "$D/ring.c" "$D/shell_main.c" >"$W/sh.c"
frontend/chibicc/chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX \
  -cc1-input "$W/sh.c" -cc1-output "$W/sh.ir" "$W/sh.c"
cp browser/web/assets/pg_libc.temeno "$W/libc.temeno"
