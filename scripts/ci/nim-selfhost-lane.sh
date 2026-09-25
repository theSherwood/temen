#!/usr/bin/env bash
# The self-hosted nimony lane (#1609) as a CHECK: nimony's own toolchain, compiled to Temen with no
# C compiler, builds a real nim program on Temen, and the program runs.
#
#   nimony      the driver: `nimony t --isMain prog.nim` parses the dependency graph, writes the
#               build plan (its Temen backend), and runs nifmake over it
#   nifmake     forks and execs every step of the plan, in dependency order, through /bin/sh
#   nifler      every module parsed (nimsem also spawns it for each file a module includes)
#   nimsem      system, each dependency, then the program (--isMain)
#   hexer       every module lowered to Leng (.x.nif), then dead-code eliminated (.c.nif)
#   temen-link  the whole program linked into one Temen module (demos/temen_link)
#
# All of it in-guest: the host seeds the sources and reads the linked module, which then runs.
#
# On the engines: the bytecode engine runs the build; the JIT then reruns hexer and runs the program.
# The JIT can run the build too (`--engine jit,bytecode`: it serves fork, #1768), but every process
# first compiles its whole image — a fork twin recompiles its parent (#1825), and nifler2's lexer
# alone takes ~2 minutes (#1831) — which puts it far past this job's budget. The tree-walker — the
# oracle, 4x slower here — is `--engine tree`, for a local run; nim_e2e's
# `nim_shells_out_through_the_posix_sh` differentials the spawning mechanism across all three engines
# on every PR.
#
# Checked against native nimony (`nimony c`) building the same program with the same phases — each
# one built by nimony from the same source as its Temen build, so only the target differs: every
# artifact through the DCE'd `.c.nif` must be byte for byte the same (`nim_selfhost_lane --expect`,
# no normalization, no tolerance), the module linked in-guest must be the host's link of the same
# `.c.nif`s, and the program must print what the native binary prints.
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> bash scripts/ci/nim-selfhost-lane.sh
#
# Needs LLVM (scripts/ci/install-llvm.sh; `LLVM_LINK`/`LLVM_OPT` override the tools) for temen-link,
# which is built from source every run, like every other tool here. Release-mode minutes, so it is a
# step of its own rather than a `cargo test`. The per-PR gate over the same mechanisms, at a size a
# debug test run can afford, is nim_e2e's `nim_shells_out_through_the_posix_sh`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

# The native reference runs from a private copy of the toolchain, so its stdlib sits at `lib/` beside
# its cwd — exactly where the guest's memfs has it. nimony records every source path relative to its
# cwd, so the two then record the same paths. (nimony finds its stdlib at `<bin>/../lib`; 38 MB.)
N="$W/nimony"
mkdir -p "$N"
cp -r "$ROOT/nimony/bin" "$ROOT/nimony/lib" "$N/"

echo "[1/4] the toolchain, through the POSIX edge"
# Each phase twice from one nimony build: the Temen module, and (`--native`) the native binary, which
# replaces the reference toolchain's own. `nimony/bin`'s are built by classic Nim — a different
# compiler, whose hash tables iterate in another order (#1753) — so they are not the reference. The
# driver and nifmake only sequence work; their native copies stay as they are.
cargo build --release -q -p temen-run --example build_nim_hello_temen --example nim_selfhost_lane
B=target/release/examples
# The driver and nifmake are built from a patched COPY of the pinned sources (the submodule is never
# modified): nifmake is a classic-Nim-only program upstream, and the driver learns the Temen backend.
# See each patch's own preamble.
mkdir -p "$W/patched"
cp -r "$ROOT/nimony/src" "$ROOT/nimony/doc" "$W/patched/"
for p in "$ROOT"/patches/nimony/*.patch; do (cd "$W/patched" && git apply "$p"); done
# The six builds are independent, so they run at once; one after another they were ~2 min of this
# step. Each gets its own nimcache, because the in-tree builds would otherwise all write
# `nimony/nimcache`.
build() { "$B/build_nim_hello_temen" --posix --nimcache "$W/nimcache-$1" "${@:2}"; }
pids=()
for p in nifler2/nifler2 nimony/nimsem hexer/hexer; do
  n="$(basename "$p")"
  build "$n" --root nimony --native "$N/bin/$n" "src/$p.nim" "$W/$n.temen" &
  pids+=($!)
done
build nimony --root "$W/patched" src/nimony/nimony.nim "$W/nimony.temen" &
pids+=($!)
build nifmake --root "$W/patched" src/nifmake/nifmake.nim "$W/nifmake.temen" &
pids+=($!)
TEMEN_LINK_CACHE="$W/temen_link_cache" bash crates/temen-run/demos/temen_link/build.sh "$W/temen-link.temen" &
pids+=($!)
for pid in "${pids[@]}"; do wait "$pid"; done

echo "[2/4] the native reference: nimony builds and runs the program"
cat >"$N/prog.nim" <<'NIM'
import std/[strutils, syncio]
let parts = "a,bb,ccc".split(",")
var total = 0
for p in parts: total += p.len
echo "parts=", parts.len, " total=", total, " up=", toUpperAscii("temen")
NIM
(cd "$N" && ./bin/nimony c --isMain prog.nim >/dev/null)
"$(ls "$N"/nimcache/*/prog | head -1)" >"$W/native.out"

echo "[3/4] /bin/sh: the POSIX build of demos/shell"
make -s -C frontend/chibicc
D=crates/temen-run/demos/shell
cat "$D/shim.c" "$D/ring.c" "$D/shell_main.c" >"$W/sh.c"
frontend/chibicc/chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX \
  -cc1-input "$W/sh.c" -cc1-output "$W/sh.ir" "$W/sh.c"

echo "[4/4] the lane: the program built on Temen, by nimony's own toolchain"
"$B/nim_selfhost_lane" --engine bytecode,jit \
  --sh "$W/sh.ir" --nimony "$W/nimony.temen" --nifmake "$W/nifmake.temen" \
  --nimsem "$W/nimsem.temen" --nifler "$W/nifler2.temen" --hexer "$W/hexer.temen" \
  --temen-link "$W/temen-link.temen" \
  --expect "$N/nimcache" nimony/lib "$N/prog.nim" "$W/cache" >"$W/temen.out"
if ! diff -u "$W/native.out" "$W/temen.out"; then
  echo "the program built on Temen does not print what the native build prints (diff above)" >&2
  exit 1
fi
echo "✅ the program built on Temen prints what the native build prints: $(cat "$W/temen.out")"
