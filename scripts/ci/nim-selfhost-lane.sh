#!/usr/bin/env bash
# The self-hosted nimony lane (#1609) as a CHECK: nimony's own phases, compiled to Temen with no C
# compiler, build a real nim program on Temen, and the program runs.
#
#   nifler  every module parsed — nimsem spawns it itself: fork → execve("/bin/sh", ["-c",
#           "bin/nifler …"]) → the shell execs nifler in place → nimsem reaps it
#   nimsem  system, each dependency, then the program (--isMain)
#   hexer   every module lowered to Leng (.x.nif)
#   link    temen-leng, and the program runs
#
# Checked against native nimony building the same program: every .s.nif and .x.nif it wrote must
# be here byte for byte (`nim_selfhost_lane --expect`; the one tolerance is #1753's declaration
# order in an .x.nif, reported when used), and the program must print what the native binary prints.
#
#   NIMONY_BIN=<nimony/bin> NIM_BIN=<dir holding nim> bash scripts/ci/nim-selfhost-lane.sh
#
# Release-mode minutes, so it is a step of its own rather than a `cargo test`. The per-PR gate over
# the same mechanisms, at a size a debug test run can afford, is nim_e2e's
# `nim_shells_out_through_the_posix_sh`.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
: "${NIMONY_BIN:?set NIMONY_BIN to the directory holding nimony}"
: "${NIM_BIN:?set NIM_BIN to the directory holding nim}"
export PATH="$NIMONY_BIN:$NIM_BIN:$PATH"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

echo "[1/4] the native reference: nimony builds and runs the program"
# From a private copy of the toolchain, so native nimony's stdlib sits at `lib/` beside its cwd —
# exactly where the guest's memfs has it. nimony records every source path relative to its cwd, so
# the two then record the same paths, and every artifact compares byte for byte, unnormalized.
# (nimony finds its stdlib at `<bin>/../lib`; the copy is 38 MB.)
N="$W/nimony"
mkdir -p "$N"
cp -r "$ROOT/nimony/bin" "$ROOT/nimony/lib" "$N/"
cat >"$N/prog.nim" <<'NIM'
import std/[strutils, syncio]
let parts = "a,bb,ccc".split(",")
var total = 0
for p in parts: total += p.len
echo "parts=", parts.len, " total=", total, " up=", toUpperAscii("temen")
NIM
(cd "$N" && ./bin/nimony c --isMain prog.nim >/dev/null)
SYS_PNIF="$(ls "$N"/nimcache/sys*.p.nif | head -1)"
STEM="$(basename "$SYS_PNIF" .p.nif)"
NATIVE_BIN="$(ls "$N"/nimcache/*/prog | head -1)"
"$NATIVE_BIN" >"$W/native.out"

echo "[2/4] the compiler phases, through the POSIX edge"
cargo build --release -q -p temen-run --example build_nim_hello_temen --example nim_selfhost_lane
B=target/release/examples
for p in nifler2/nifler2 nimony/nimsem hexer/hexer; do
  "$B/build_nim_hello_temen" --posix --root nimony "src/$p.nim" "$W/$(basename "$p").temen"
done

echo "[3/4] /bin/sh: the POSIX build of demos/shell"
make -s -C frontend/chibicc
D=crates/temen-run/demos/shell
cat "$D/shim.c" "$D/ring.c" "$D/shell_main.c" >"$W/sh.c"
frontend/chibicc/chibicc -cc1 --emit-ir --child-entry -DTEMEN_SHELL_POSIX \
  -cc1-input "$W/sh.c" -cc1-output "$W/sh.ir" "$W/sh.c"

echo "[4/4] the lane: the program built on Temen, by nimony's own phases"
"$B/nim_selfhost_lane" --sh "$W/sh.ir" --check "$N/prog.nim" --hexer "$W/hexer.temen" \
  --expect "$N/nimcache" \
  "$W/nimsem.temen" "$W/nifler2.temen" nimony/lib "$SYS_PNIF" "$STEM" "$W/out.s.nif" >"$W/temen.out"
if ! diff -u "$W/native.out" "$W/temen.out"; then
  echo "the program built on Temen does not print what the native build prints (diff above)" >&2
  exit 1
fi
echo "✅ the program built on Temen prints what the native build prints: $(cat "$W/temen.out")"
