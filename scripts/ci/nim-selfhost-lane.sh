#!/usr/bin/env bash
# The self-hosted nimony lane (#1609) as a CHECK: nimony's own toolchain, compiled to Temen with no
# C compiler, builds a real nim program on Temen, and the program runs.
#
#   nimony      the driver: `nimony t --isMain prog.nim` parses the dependency graph, writes the
#               build plan (its Temen backend), and runs nifmake over it
#   nifmake     spawns every step of the plan, in dependency order, through /bin/sh
#   nifler      every module parsed (nimsem also spawns it for each file a module includes)
#   nimsem      system, each dependency, then the program (--isMain)
#   hexer       every module lowered to Leng (.x.nif), then dead-code eliminated (.c.nif)
#   temen-link  the whole program linked into one Temen module (demos/temen_link)
#
# All of it in-guest: the host seeds the sources and reads the linked module, which then runs.
#
# On the engines: the JIT runs the build — it serves pspawn, execve and waitpid, and a run
# compiles each program once, for every process that runs it (#1825) — then the bytecode engine reruns
# hexer, held to the JIT's bytes, and both run the program. The tree-walker — the oracle, 4x slower
# here — is `--engine tree`, for a local run. On every PR, the spawn is differentialled across all
# three engines by `caller_request_parity.rs`, and the fork + exec a program not built for Temen
# takes by nim_e2e's `nim_shells_out_through_the_posix_sh`.
#
# Checked against native nimony (`nimony c -d:temen`) building the same program with the same
# phases — each built by nimony from the same source as its Temen build, so only the target differs:
# every artifact through the DCE'd `.c.nif` must be byte for byte the same (`nim_selfhost_lane
# --expect`, no normalization, no tolerance), the module linked in-guest must be the host's link of
# the same `.c.nif`s, and the program must print what the native binary prints.
#
# With NIM_LANE_SELF=1 the lane also builds a program with a macro in-guest (#2046): nimsem builds
# the macro's plugin for Temen and runs it there, and the program must print what native nimony's
# build of it prints. And it has nimony build each of its own tools in-guest (#763): the driver,
# nifmake, nifler2, nimsem and hexer. Each must be byte for byte the tool that ran in the build
# (`--fixed-point`), and nimsem's artifacts must also match native nimony's. nifler2 takes its
# plugins, parsegen and regex, which nimsem builds for Temen and runs in-guest. The nightly run (and
# a manual dispatch) sets it; a PR run leaves it out.
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> [NIM_LANE_SELF=1] bash scripts/ci/nim-selfhost-lane.sh
#
# Needs LLVM (scripts/ci/install-llvm.sh; `LLVM_LINK`/`LLVM_OPT` override the tools) for temen-link,
# which is built from source every run, like every other tool here. Release-mode minutes, so it is a
# step of its own rather than a `cargo test`. The per-PR gates over its mechanisms, at a size a
# debug test run can afford, are the two named above.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
cargo build --release -q -p temen-run --example nim_selfhost_lane
B=target/release/examples

echo "[1-3/5] the toolchain (scripts/nim-toolchain.sh), and temen-link beside it"
# The phases nimony builds natively, the tools they build through the POSIX edge, and /bin/sh — in
# the patched tree at "$W/nimony" (`N`), which every build here happens in. temen-link is the lane's
# own: the in-guest link, built through the LLVM on-ramp.
TEMEN_LINK_CACHE="$W/temen_link_cache" bash crates/temen-run/demos/temen_link/build.sh "$W/temen-link.temen" &
link=$!
bash scripts/nim-toolchain.sh "$W"
wait "$link"
N="$W/nimony"
lane() {
  "$B/nim_selfhost_lane" \
    --sh "$W/sh.ir" --nimony "$W/nimony.temen" --nifmake "$W/nifmake.temen" \
    --nimsem "$W/nimsem.temen" --nifler "$W/nifler2.temen" --hexer "$W/hexer.temen" \
    --temen-link "$W/temen-link.temen" "$@"
}

echo "[4/5] a program built on Temen by nimony's own toolchain"
# The program is built in a tree of its own — the toolchain's `bin/` and `lib/` beside it — so its
# nimcache holds its build alone. `atCompileTime` is a `const` nimony evaluates by building a program
# and running it, so the build in-guest must build and run one too (#763). Native builds for the
# Temen platform (`-d:temen`), as `nimony t` builds: the C backend's pipeline to the `.c.nif`, which
# is what makes every artifact comparable byte for byte.
P="$W/prog"
mkdir -p "$P"
cp -r "$N/bin" "$N/lib" "$P/"
cat >"$P/prog.nim" <<'NIM'
import std/[strutils, syncio]
proc lengths(s: string): int =
  result = 0
  for p in s.split(","): result += p.len
const atCompileTime = lengths("a,bb,ccc")
let parts = "a,bb,ccc".split(",")
echo "parts=", parts.len, " total=", lengths("a,bb,ccc"), " ctfe=", atCompileTime, " up=", toUpperAscii("temen")
NIM
(cd "$P" && ./bin/nimony c -d:temen --isMain prog.nim >/dev/null)
"$(ls "$P"/nimcache/*/prog | head -1)" >"$W/native.out"
lane --engine jit,bytecode --expect "$P" prog.nim "$W/cache" >"$W/temen.out"
if ! diff -u "$W/native.out" "$W/temen.out"; then
  echo "the program built on Temen does not print what the native build prints (diff above)" >&2
  exit 1
fi
echo "✅ the program built on Temen prints what the native build prints: $(cat "$W/temen.out")"

if [ -z "${NIM_LANE_SELF:-}" ]; then
  echo "[5/5] skipped: a macro, and nimony building its tools in-guest, run with NIM_LANE_SELF=1 (the nightly run)"
  exit 0
fi
echo "[5/5] a macro: nimsem builds its plugin on Temen and runs it there"
# Not `--expect`: the plugin's own build caches are a Temen build in-guest and a C one natively.
# The tree is nimony's whole one: `std/macros` imports nimony's own sources (`src/`), and native
# nimony validates a plugin against `doc/tags.md`.
M="$W/macro"
mkdir -p "$M"
cp -r "$N/bin" "$N/lib" "$N/src" "$N/doc" "$M/"
cat >"$M/prog.nim" <<'NIM'
import std/[syncio, macros]
macro hello(): untyped =
  result = newCall("echo", [newStrLitNode("hello from a macro")])
hello()
NIM
(cd "$M" && ./bin/nimony c -d:temen --isMain prog.nim >/dev/null)
"$(ls "$M"/nimcache/*/prog | head -1)" >"$W/macro-native.out"
lane --engine jit "$M" prog.nim "$W/cache-macro" >"$W/macro-temen.out"
if ! diff -u "$W/macro-native.out" "$W/macro-temen.out"; then
  echo "the macro program built on Temen does not print what the native build prints (diff above)" >&2
  exit 1
fi
echo "✅ the macro program built on Temen prints what the native build prints: $(cat "$W/macro-temen.out")"

echo "[5/5] nimony builds each of its tools on Temen, and each is the tool that built it"
# Each against the .temen the toolchain linked (`--fixed-point`); nimsem also against the
# toolchain's native build of it, in the tree's nimcache (`--expect`).
lane --engine jit --expect --fixed-point "$N" src/nimony/nimsem.nim "$W/cache-nimsem"
for p in nifler2/nifler2 hexer/hexer nimony/nimony nifmake/nifmake; do
  lane --engine jit --fixed-point "$N" "src/$p.nim" "$W/cache-$(basename "$p")"
done
