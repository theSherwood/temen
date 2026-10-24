#!/usr/bin/env bash
# nimony's own toolchain as Temen modules, built with no C compiler (#763): what the self-hosted lane
# (`scripts/ci/nim-selfhost-lane.sh`) builds programs with, and what the playground's nim card ships
# (`scripts/rebuild-assets.sh`, step `nim_card`).
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> bash scripts/nim-toolchain.sh <out>
#
# writes into <out>:
#   nimony/          nimony's tree, laid out as nimony's own is (`bin/ lib/ src/ doc/`), its `bin/`
#                    holding the phases and the driver step 1 builds: a link into the cache below
#   nimony.temen nifmake.temen nifler2.temen nimsem.temen shoggoth.temen
#                    the tools, through the POSIX edge; nimsem's build in the tree's own nimcache,
#                    each other's in the cache's `nimcache-<tool>/`
#   sh.ir            /bin/sh: the POSIX build of demos/shell
#   libc.temeno      the guest libc temen-link links a program against (the committed
#                    `browser/web/assets/pg_libc.temeno`)
#
# Every tool nimony builds from the tree, so a tool nimony builds in-guest from the same tree is one
# this script built: what the lane's fixed point holds nimsem to.
#
# Everything but the link depends only on nimony: the sources at the pin, the patches, the
# `nimony/bin` that builds them, and this script. So the tree, its native phases and the tools'
# build directories are built once per such input, in `target/nim-toolchain/<key>/`, and each run
# after that reuses them (#2442). nimony's own build finds them up to date and reruns only
# `temen-link`, the one step that is this repo's.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
[ $# -eq 1 ] || { echo "usage: scripts/nim-toolchain.sh <out>" >&2; exit 2; }
mkdir -p "$1"
W="$(cd "$1" && pwd)"

# `temen-link` on the host: the driver plans it as the build's last step (`nimony t`).
cargo build --release -q -p temen-run --example temen_link
export TEMEN_PG_LIBC="$ROOT/browser/web/assets/pg_libc.temeno"

# One build of the cache at a time: a run writes it, even when it only relinks.
mkdir -p target/nim-toolchain
exec 9>target/nim-toolchain/lock
flock 9
KEY="$({
  git -C nimony rev-parse HEAD
  find patches/nimony "$NIMONY_BIN" scripts/nim-toolchain.sh -type f -print0 | sort -z | xargs -0 cat
} | sha256sum | cut -c1-16)"
C="$ROOT/target/nim-toolchain/$KEY"
# The tree, from a COPY of the pinned sources (the submodule is never modified), patched: nifmake is a
# classic-Nim-only program upstream, the driver learns the Temen backend, a compiler built for
# Temen builds the programs it runs itself (compile-time evaluation, plugins) for Temen, and a
# program built for Temen starts its processes with Temen's spawn, not a fork. See each patch's own
# preamble. Every build of a tool happens in this tree, and so does the lane's self-build:
# the guest's memfs holds it at the same path, so both record the same paths — nimony records a
# source path relative to its cwd, but writes a compile-time evaluation program's imports and output
# file absolute.
N="$C/nimony"
# A step's builds run at once and quietly, and nimony prints a build's errors on its stdout. So each
# build writes a log in the cache, and a step names any build that failed and shows its log: a failed
# tool used to stop the script with nothing said (#2613).
pids=() names=()
build() { # <name> <command…>: start one build in the tree, in the background
  local n="$1"
  shift
  (cd "$N" && "$@" >"$C/$n.log" 2>&1) &
  pids+=($!) names+=("$n")
}
await_builds() {
  local i failed=0
  for i in "${!pids[@]}"; do
    if ! wait "${pids[$i]}"; then
      echo "nim-toolchain: building ${names[$i]} failed:" >&2
      cat "$C/${names[$i]}.log" >&2
      failed=1
    fi
  done
  pids=() names=()
  [ "$failed" -eq 0 ]
}
if [ ! -e "$C/built" ]; then
  # Another key's cache is a gigabyte that no build of this checkout uses again.
  find target/nim-toolchain -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} +
  mkdir -p "$N"
  # The sources as the pin has them (`git archive`), not as the checkout is: hastur's build fetches
  # git-ignored files into it (Nim's parser for classic nifler, Nim's cycle collectors for
  # `--mm:orc`), which no tool here imports, and the card's pack ships the tree's `lib/` and `src/`.
  cp -r "$ROOT/nimony/bin" "$N/"
  git -C "$ROOT/nimony" archive HEAD lib src doc | tar -x -C "$N"
  # The cache is inside this repo's worktree, where `git apply` patches paths relative to the
  # repo's root and silently skips the tree's: the ceiling keeps git from finding the repo.
  for p in "$ROOT"/patches/nimony/*.patch; do
    (cd "$N" && GIT_CEILING_DIRECTORIES="$C" git apply "$p") || {
      echo "nim-toolchain: $(basename "$p") does not apply to nimony $(git -C "$ROOT/nimony" rev-parse --short HEAD)" >&2
      exit 1
    }
  done

  echo "[toolchain 1/3] the phases and the driver, native: nimony builds nifler2, nimsem, shoggoth and itself"
  # `nimony/bin`'s phases are built by classic Nim — a different compiler, whose hash tables iterate
  # in another order (#1753) — so nimony builds each again from the tree, and those replace them:
  # they are the reference every artifact is held to, and the compiler of step 2. The driver is
  # rebuilt for its Temen backend, which step 2 plans with; nifmake only sequences work, so `bin/`'s
  # stays.
  for p in nifler2/nifler2 nimony/nimsem lengc/shoggoth/shoggoth nimony/nimony; do
    n="$(basename "$p")"
    build "native-$n" nimony c --isMain --nimcache:"$C/native-$n" "src/$p.nim"
  done
  await_builds
  for n in nifler2 nimsem shoggoth nimony; do
    cp "$(ls "$C/native-$n"/c/*/"$n" | head -1)" "$N/bin/$n"
  done
  touch "$C/built"
fi
cp target/release/examples/temen_link "$N/bin/temen-link"
ln -sfn "$N" "$W/nimony"

echo "[toolchain 2/3] the tools, through the POSIX edge: nimony t builds each, and temen-link links it"
# nimsem builds in the tree's own nimcache, which the lane's self-build compares against; the others
# each get their own, because the builds run at once and would otherwise all write it. Each build's
# module goes first, so its link reruns over what the cache holds.
cache() { if [ "$1" = nimsem ]; then echo "$N/nimcache"; else echo "$C/nimcache-$1"; fi; }
for p in nimony/nimsem nifler2/nifler2 lengc/shoggoth/shoggoth nimony/nimony nifmake/nifmake; do
  n="$(basename "$p")"
  rm -f "$(cache "$n")"/t/*/"$n.temen"
  args=(t --isMain "src/$p.nim")
  [ "$n" = nimsem ] || args=(t --isMain --nimcache:"$(cache "$n")" "src/$p.nim")
  build "$n.temen" env PATH="$N/bin:$PATH" nimony "${args[@]}"
done
await_builds
for n in nimsem nifler2 shoggoth nimony nifmake; do
  cp "$(ls "$(cache "$n")"/t/*/"$n.temen" | head -1)" "$W/$n.temen"
done

echo "[toolchain 3/3] /bin/sh: the POSIX build of demos/shell"
make -s -C frontend/chibicc
D=crates/temen-run/demos/shell
cat "$D/shim.c" "$D/ring.c" "$D/shell_main.c" >"$W/sh.c"
frontend/chibicc/chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX \
  -cc1-input "$W/sh.c" -cc1-output "$W/sh.ir" "$W/sh.c"
cp browser/web/assets/pg_libc.temeno "$W/libc.temeno"
